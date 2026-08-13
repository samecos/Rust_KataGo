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
