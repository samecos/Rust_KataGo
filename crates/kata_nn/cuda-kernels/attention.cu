// 注意力 kernel v1（正确优先版本；M4 换 FlashAttention 风格 both16 tile 版）。
//
// 输入 Q/K/V：[B*H, S, D] 行主序 FP16（S=361, H=12, D=32），non-causal 无掩码。
// 计算：out[i,:] = Σ_j softmax_j(Q_i·K_j · scale) · V[j,:]
//
// `scale` 由调用方指定：ONNX 图里 q/k 各乘 qk_scale=1/∜d，乘积即 1/√d；
// 传统 1/√d 语义则传 rsqrtf(d)（两者数值等价，缩放挪到 score 上做）。
//
// 每个 block 处理一个查询行 i：blockDim = S（向上取 384），
// 线程 j 先算 score_j，shared 树归约做两遍 softmax 归一，再算输出。

#include <cuda_fp16.h>
#include <cstdint>

#define ATT_BLOCK 512

extern "C" __global__ void attention_row_kernel(
    const __half* __restrict__ q,   // [B*H, S, D]
    const __half* __restrict__ k,   // [B*H, S, D]
    const __half* __restrict__ v,   // [B*H, S, D]
    __half* __restrict__ out,       // [B*H, S, D]
    int s, int d, float scale) {
    const int i = blockIdx.x;   // query 行
    const int bh = blockIdx.y;  // batch*head
    const int j = threadIdx.x;  // key 索引

    __shared__ float s_score[ATT_BLOCK];  // 原始 score（保留）
    __shared__ float s_red[ATT_BLOCK];    // 归约工作区
    __shared__ float s_p[ATT_BLOCK];
    __shared__ float s_maxv;
    __shared__ float s_sumv;

    const __half* qrow = q + ((size_t)bh * s + i) * d;

    // --- score_j = Q_i · K_j * scale ---
    float score = 0.0f;
    if (j < s) {
        const __half* krow = k + ((size_t)bh * s + j) * d;
        for (int dd = 0; dd < d; ++dd) {
            score += __half2float(qrow[dd]) * __half2float(krow[dd]);
        }
        score *= scale;
    }
    s_score[j] = (j < s) ? score : -1e30f;
    s_red[j] = s_score[j];
    __syncthreads();

    // --- 最大值归约（树形，shared）---
    for (int off = ATT_BLOCK / 2; off > 0; off >>= 1) {
        if (j < off) {
            float a = s_red[j];
            float b = s_red[j + off];
            s_red[j] = fmaxf(a, b);
        }
        __syncthreads();
    }
    s_maxv = s_red[0];
    __syncthreads();

    // --- p_j = exp(score - max) / sum ---
    s_red[j] = (j < s) ? expf(s_score[j] - s_maxv) : 0.0f;
    __syncthreads();
    for (int off = ATT_BLOCK / 2; off > 0; off >>= 1) {
        if (j < off) {
            s_red[j] += s_red[j + off];
        }
        __syncthreads();
    }
    s_sumv = s_red[0];
    __syncthreads();
    if (j < s) {
        s_p[j] = expf(s_score[j] - s_maxv) / s_sumv;
    }
    __syncthreads();

    // --- out[i][j] = Σ_jj p[jj] * V[jj][j]（线程 j < d 负责输出维 j）---
    if (j < d) {
        float acc = 0.0f;
        const __half* vcol = v + (size_t)bh * s * d + j;
        for (int jj = 0; jj < s; ++jj) {
            acc += s_p[jj] * __half2float(vcol[(size_t)jj * d]);
        }
        out[((size_t)bh * s + i) * d + j] = __float2half(acc);
    }
}

// ---------------------------------------------------------------------------
// 注意力 kernel v2（M4 第一刀）：warp-per-row + 块共享 K/V smem。
//
// 每个 block 处理 ROWS_PER_BLOCK 个查询行，K/V 整块协作加载到 shared 一次
// （v1 每个 block 只处理 1 行、K/V 全从 global 重读，这里消除 ~120× 冗余读）。
// 每个 warp 负责一行：lane 串行处理 12 个 key（寄存器存 score），
// shfl 树归约 max/sum（两遍在线 softmax），再逐 lane 累加 p·V 写输出。
//
// 数值纪律：score/softmax 全部 FP32，输出舍回 half（与 v1 一致）。
// 约束：S ≤ 512（blockDim 覆盖），D ≤ 64，ROWS_PER_BLOCK = 8。

#define ATT2_ROWS 8
#define ATT2_WARPS 8

extern "C" __global__ void attention_row_v2_kernel(
    const __half* __restrict__ q,   // [B*H, S, D]
    const __half* __restrict__ k,   // [B*H, S, D]
    const __half* __restrict__ v,   // [B*H, S, D]
    __half* __restrict__ out,       // [B*H, S, D]
    int s, int d, float scale) {
    extern __shared__ __half smem[];  // [K: S*D] + [V: S*D]
    __half* s_k = smem;
    __half* s_v = smem + (size_t)s * d;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;
    const int row_global = blockIdx.x * ATT2_ROWS + threadIdx.y;  // 块内行号 → 全局 q 行
    // threadIdx.y ∈ [0, ATT2_ROWS)（blockDim.y = ATT2_ROWS）

    // --- 协作加载 K/V（一次，块内所有行共享）---
    const int total = s * d;
    for (int idx = tid; idx < total; idx += nthreads) {
        s_k[idx] = k[(size_t)blockIdx.y * total + idx];
        s_v[idx] = v[(size_t)blockIdx.y * total + idx];
    }
    __syncthreads();

    const int i = row_global;
    if (i >= s) return;
    const int bh = blockIdx.y;
    const int lane = threadIdx.x;

    // q 行（32 half → 16 个 uint2 读进寄存器）
    __half qreg[64];
    const __half* qrow = q + ((size_t)bh * s + i) * d;
    #pragma unroll 8
    for (int dd = 0; dd < d; ++dd) qreg[dd] = qrow[dd];

    // --- pass1：每 lane 串行算 12 个 score，warp 归约 max ---
    const int nseg = (s + 31) / 32;
    float scores[16];  // 每 lane 至多 16 段
    #pragma unroll
    for (int t = 0; t < 16; ++t) scores[t] = -1e30f;
    float maxv = -1e30f;
    for (int t = 0; t < nseg; ++t) {
        const int j = t * 32 + lane;
        if (j < s) {
            float sc = 0.0f;
            const __half* krow = s_k + (size_t)j * d;
            #pragma unroll 8
            for (int dd = 0; dd < d; ++dd) {
                sc += __half2float(qreg[dd]) * __half2float(krow[dd]);
            }
            sc *= scale;
            scores[t] = sc;
            maxv = fmaxf(maxv, sc);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        maxv = fmaxf(maxv, __shfl_xor_sync(0xffffffffu, maxv, off));
    }

    // --- pass2：exp(score-max) 求和，warp 归约 sum ---
    float sumv = 0.0f;
    for (int t = 0; t < nseg; ++t) {
        if (t * 32 + lane < s) {
            float e = expf(scores[t] - maxv);
            scores[t] = e;
            sumv += e;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sumv += __shfl_xor_sync(0xffffffffu, sumv, off);
    }
    const float inv_sum = 1.0f / sumv;

    // --- pass3：out[i][:] = Σ_j p_j · V[j][:]（每 lane 累加自己负责的 key，
    //     然后 warp 树归约合并 32 个 lane 的部分和）---
    float acc[64];
    #pragma unroll
    for (int dd = 0; dd < 64; ++dd) acc[dd] = 0.0f;
    for (int t = 0; t < nseg; ++t) {
        const int j = t * 32 + lane;
        if (j < s) {
            const float p = scores[t] * inv_sum;
            const __half* vrow = s_v + (size_t)j * d;
            #pragma unroll 8
            for (int dd = 0; dd < d; ++dd) {
                acc[dd] += p * __half2float(vrow[dd]);
            }
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        #pragma unroll 8
        for (int dd = 0; dd < d; ++dd) {
            acc[dd] += __shfl_xor_sync(0xffffffffu, acc[dd], off);
        }
    }
    #pragma unroll 8
    for (int dd = 0; dd < d; ++dd) {
        out[((size_t)bh * s + i) * d + dd] = __float2half(acc[dd]);
    }
}

// ---------------------------------------------------------------------------
// 注意力 kernel v3（M4 第二刀）：warp-per-row + K/V smem 分块（128B 打包布局）
// + cp.async.cg 双缓冲流水。b11fix（S=361/H=12/D=32）上目标替代 v1/v2。
//
// 与 v1/v2 的差异与动机：
//   - 每 block 处理 8 个 query 行（8 个 warp，warp w → 行 i），grid = (ceil(S/8),
//     BH) → S=361 时 46×12 = 552 块（块数充足）。v1 每行一块、512 线程做两遍
//     18 次 syncthreads 树归约、且 p·V 只由 32 线程串行 361 次；v2 整段 K/V
//     46KB smem 只 4 块/SM、每 lane 串行依赖链长。
//   - K/V 按 CHUNK=64 键分块，cp.async.cg 16B 双缓冲流水预取（隐藏 L2 延迟）；
//     块内 smem 为 K/V 各 2×4KB tile + s_score 16KB = 33KB（v2 的 46KB 整段
//     K/V → 只 4 块/SM 的天花板消失；此处 4 块/SM 由寄存器 64/线程决定）。
//     CHUNK=128（48KB smem、barrier 减半）实测 attn 0.055 vs 0.056ms 无差异，
//     故保持 64。
//   - smem 用 128B 打包布局（row r → 字节 (r>>1)*128 + (r&1)*64，16B 块内交织）：
//     warp 读同一 key 行的 4×LDS.128 完全无 bank 冲突（行主序直读是 16-way）。
//     加载时 cp.async 直接写打包位置（每线程 1 个 16B chunk，src 线性/dst 置换），
//     无需 staging 中转。
//   - softmax 三趟全在 warp 内（shfl 归约，零跨 warp 树归约）：
//       pass1  QK 点积 → s_score[i][j]（smem，降寄存器提占用），shfl 归 max
//       pass2  exp(score-max) 求和，shfl 归 sum → inv
//       pass3  V tile 流水，acc[32] += p·V（每 lane 12 key 并行），shfl 归约写回
//   - 数值纪律同 v1：Q/K/V half → FP32 计算，score/softmax 全 FP32，输出
//     __float2half_rn（RNE），expf 精确版，无 fast_math。
//   - 约束：S ≤ 512，D == 32（smem/打包布局按 D=32 编译期定），CHUNK = 64。

#define ATT3_ROWS 8
#define ATT3_CHUNK 64   // 每 tile 键数（64 vs 128 实测性能相同；64 smem 更小）
#define ATT3_HALF_GRP (ATT3_CHUNK / 32)     // 每 tile 每组 32 键的组数
#define ATT3_CHUNKS_PT (ATT3_CHUNK / 64)    // 每线程 16B chunk 数

// 128B 打包布局的 16B 块字节偏移：
//   pack：tile 内线性 chunk t → 字节偏移（row = t>>2，c = t&3）。
//   chunk：行 row、行内块 c（0..3）→ 字节偏移。
__device__ __forceinline__ unsigned att3_pack_off(int t) {
    return ((unsigned)(t >> 3) << 7) | ((unsigned)((t >> 2) & 1) << 6) |
           ((unsigned)(t & 3) << 4);
}
__device__ __forceinline__ unsigned att3_chunk_off(int row, int c) {
    return ((unsigned)(row >> 1) << 7) | ((unsigned)(row & 1) << 6) |
           ((unsigned)c << 4);
}

// cp.async.cg 16B：global → smem（saddr 为 cvta 后的 32 位共享地址）。
__device__ __forceinline__ void att3_cp_async16(unsigned saddr,
                                                const void* gaddr) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(saddr),
                 "l"(gaddr));
}
__device__ __forceinline__ void att3_cp_commit() {
    asm volatile("cp.async.commit_group;\n");
}
// 等待直到 ≤ N 组 cp.async 未完成。
template <int N>
__device__ __forceinline__ void att3_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// 16B（8 halfs）→ 4 个 float2（小端：低 16 位为第一个 half）。
__device__ __forceinline__ void att3_h2f8(uint4 u, float2 f[4]) {
    f[0] = __half22float2(__halves2half2(
        __ushort_as_half((unsigned short)(u.x & 0xffffu)),
        __ushort_as_half((unsigned short)(u.x >> 16))));
    f[1] = __half22float2(__halves2half2(
        __ushort_as_half((unsigned short)(u.y & 0xffffu)),
        __ushort_as_half((unsigned short)(u.y >> 16))));
    f[2] = __half22float2(__halves2half2(
        __ushort_as_half((unsigned short)(u.z & 0xffffu)),
        __ushort_as_half((unsigned short)(u.z >> 16))));
    f[3] = __half22float2(__halves2half2(
        __ushort_as_half((unsigned short)(u.w & 0xffffu)),
        __ushort_as_half((unsigned short)(u.w >> 16))));
}

extern "C" __global__ void __launch_bounds__(ATT3_ROWS * 32, 4)
attention_row_v3_kernel(const __half* __restrict__ q,
                        const __half* __restrict__ k,
                        const __half* __restrict__ v,
                        __half* __restrict__ out, int s, int d, float scale) {
    __shared__ __half s_k[2][ATT3_CHUNK * 32];  // K tile 双缓冲（各 4KB）
    __shared__ __half s_v[2][ATT3_CHUNK * 32];  // V tile 双缓冲（各 4KB）
    __shared__ float s_score[ATT3_ROWS][512];  // 原始/exp 得分（warp 私有行）

    const int tid = threadIdx.x;  // 0..255
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int bh = blockIdx.y;
    const int i = blockIdx.x * ATT3_ROWS + warp;  // 本 warp 的 query 行
    const int ntiles = (s + ATT3_CHUNK - 1) / ATT3_CHUNK;
    const size_t bh_off = (size_t)bh * s * 32;  // D=32

    // --- K tile 0 预取 ---
    {
        const int valid = min(ATT3_CHUNK, s);
        const int valid_bytes = valid * 64;  // D=32 → 64B/行
        const __half* base = k + bh_off;
#pragma unroll
        for (int m = 0; m < ATT3_CHUNKS_PT; ++m) {
            const int cid = tid * ATT3_CHUNKS_PT + m;
            if (cid * 16 < valid_bytes) {
                att3_cp_async16(
                    (unsigned)__cvta_generic_to_shared(s_k[0]) +
                        att3_pack_off(cid),
                    base + cid * 8);
            }
        }
    }
    att3_cp_commit();

    // q 行 → fp32 寄存器（32 维）。尾部 block 中无效行（i >= s）跳过加载，
    // 避免越过 q 缓冲尾部（这些行的输出被丢弃，scrow/max 只在本 warp 内自洽）。
    float qf[32];
    if (i < s) {
        const __half* qrow = q + bh_off + (size_t)i * 32;
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            const float2 f = __half22float2(*(const __half2*)(qrow + 2 * m));
            qf[2 * m] = f.x;
            qf[2 * m + 1] = f.y;
        }
    }

    // --- pass 1：QK 点积 → s_score[i][j]，warp 归 max ---
    float maxv = -1e30f;
    float* scrow = s_score[warp];

    for (int t = 0; t < ntiles; ++t) {
        if (t + 1 < ntiles) {
            const int valid = min(ATT3_CHUNK, s - (t + 1) * ATT3_CHUNK);
            const int valid_bytes = valid * 64;
            const __half* base =
                k + bh_off + (size_t)(t + 1) * ATT3_CHUNK * 32;
#pragma unroll
            for (int m = 0; m < ATT3_CHUNKS_PT; ++m) {
                const int cid = tid * ATT3_CHUNKS_PT + m;
                if (cid * 16 < valid_bytes) {
                    att3_cp_async16(
                        (unsigned)__cvta_generic_to_shared(s_k[(t + 1) & 1]) +
                            att3_pack_off(cid),
                        base + cid * 8);
                }
            }
            att3_cp_commit();
            att3_cp_wait<1>();
        } else {
            att3_cp_wait<0>();
        }
        __syncthreads();

        const __half* tk = s_k[t & 1];
#pragma unroll
        for (int half = 0; half < ATT3_HALF_GRP; ++half) {
            const int j = t * ATT3_CHUNK + half * 32 + lane;
            if (j < s) {
                const int lrow = half * 32 + lane;
                float sc = 0.0f;
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const uint4 u = *(const uint4*)((const char*)tk +
                                                    att3_chunk_off(lrow, c));
                    float2 f[4];
                    att3_h2f8(u, f);
                    sc += qf[8 * c] * f[0].x + qf[8 * c + 1] * f[0].y +
                          qf[8 * c + 2] * f[1].x + qf[8 * c + 3] * f[1].y +
                          qf[8 * c + 4] * f[2].x + qf[8 * c + 5] * f[2].y +
                          qf[8 * c + 6] * f[3].x + qf[8 * c + 7] * f[3].y;
                }
                sc *= scale;
                scrow[j] = sc;
                maxv = fmaxf(maxv, sc);
            }
        }
        __syncthreads();
    }

    // shfl 归 max（覆盖全部 361 个 key）
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        maxv = fmaxf(maxv, __shfl_xor_sync(0xffffffffu, maxv, off));

    // --- pass 2：exp(score - max) 求和，warp 归 sum（写回 s_score 为 e）---
    float sumv = 0.0f;
#pragma unroll
    for (int half = 0; half < ATT3_HALF_GRP; ++half) {
        const int j = half * 32 + lane;
        if (j < s) {
            const float e = expf(scrow[j] - maxv);
            scrow[j] = e;
            sumv += e;
        }
    }
    for (int t = 1; t < ntiles; ++t) {
#pragma unroll
        for (int half = 0; half < ATT3_HALF_GRP; ++half) {
            const int j = t * ATT3_CHUNK + half * 32 + lane;
            if (j < s) {
                const float e = expf(scrow[j] - maxv);
                scrow[j] = e;
                sumv += e;
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sumv += __shfl_xor_sync(0xffffffffu, sumv, off);
    const float inv = 1.0f / sumv;

    // --- pass 3：out[i][:] = Σ_j p_j · V[j][:]（V tile 流水 + acc 寄存器）---
    float acc[32];
#pragma unroll
    for (int m = 0; m < 32; ++m) acc[m] = 0.0f;

    // V tile 0 预取
    {
        const int valid = min(ATT3_CHUNK, s);
        const int valid_bytes = valid * 64;
        const __half* base = v + bh_off;
#pragma unroll
        for (int m = 0; m < ATT3_CHUNKS_PT; ++m) {
            const int cid = tid * ATT3_CHUNKS_PT + m;
            if (cid * 16 < valid_bytes) {
                att3_cp_async16(
                    (unsigned)__cvta_generic_to_shared(s_v[0]) +
                        att3_pack_off(cid),
                    base + cid * 8);
            }
        }
    }
    att3_cp_commit();

    for (int t = 0; t < ntiles; ++t) {
        if (t + 1 < ntiles) {
            const int valid = min(ATT3_CHUNK, s - (t + 1) * ATT3_CHUNK);
            const int valid_bytes = valid * 64;
            const __half* base =
                v + bh_off + (size_t)(t + 1) * ATT3_CHUNK * 32;
#pragma unroll
            for (int m = 0; m < ATT3_CHUNKS_PT; ++m) {
                const int cid = tid * ATT3_CHUNKS_PT + m;
                if (cid * 16 < valid_bytes) {
                    att3_cp_async16(
                        (unsigned)__cvta_generic_to_shared(s_v[(t + 1) & 1]) +
                            att3_pack_off(cid),
                        base + cid * 8);
                }
            }
            att3_cp_commit();
            att3_cp_wait<1>();
        } else {
            att3_cp_wait<0>();
        }
        __syncthreads();

        const __half* tv = s_v[t & 1];
#pragma unroll
        for (int half = 0; half < ATT3_HALF_GRP; ++half) {
            const int j = t * ATT3_CHUNK + half * 32 + lane;
            if (j < s) {
                const float p = scrow[j] * inv;
                const int lrow = half * 32 + lane;
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const uint4 u = *(const uint4*)((const char*)tv +
                                                    att3_chunk_off(lrow, c));
                    float2 f[4];
                    att3_h2f8(u, f);
                    acc[8 * c] += p * f[0].x;
                    acc[8 * c + 1] += p * f[0].y;
                    acc[8 * c + 2] += p * f[1].x;
                    acc[8 * c + 3] += p * f[1].y;
                    acc[8 * c + 4] += p * f[2].x;
                    acc[8 * c + 5] += p * f[2].y;
                    acc[8 * c + 6] += p * f[3].x;
                    acc[8 * c + 7] += p * f[3].y;
                }
            }
        }
        __syncthreads();
    }

    // shfl 归约 acc[32]（每 lane 累加了自己的 12 个 key）
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
#pragma unroll
        for (int m = 0; m < 32; ++m)
            acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], off);

    // 写回（D=32，RNE）
    if (i < s) {
        __half* orow = out + bh_off + (size_t)i * 32;
#pragma unroll
        for (int m = 0; m < 32; ++m) orow[m] = __float2half_rn(acc[m]);
    }
}
