// FP16 张量核 GEMM v2：smem 分块 + cp.async 双缓冲流水 + ldmatrix。
//
// 计算 C[M,N] = alpha * A[M,K] * B[N,K]^T + beta * C[M,N]
//   A: [M,K] 行主序（权重 W[outC, inC] 布局）
//   B: [N,K] 行主序（输入 X[tokens, inC] 布局，即 KataGo 的 RowMajor/NHWC 约定）
//   C: [M,N] 行主序；beta=1 时 C 与 D 同指针（原位残差，先读 C 再写 D）。
//
// 与 v1（hgemm_m16n8k16_kernel，每 k 步直接读 global）的差异：
//   - tile 128×128×32，256 线程（8 warps，4×2 布局：warp 覆盖 32×64）
//   - A/B 各 2 级双缓冲 smem（每级 8KB，共 32KB），cp.async.cg 16B 2-stage 流水
//   - A/B 片段经 ldmatrix.x4 / ldmatrix.x4.trans 从 smem 加载（每个 8×8
//     矩阵连续 128B 存放，ldmatrix 相位内无 bank 冲突）
//   - smem 布局：byte(r, c) = (r>>3)*512 + (c>>3)*128 + (r&7)*16 + (c&7)*2
//     （8 行一组，组内 4 个 128B 窗口按 16B 行交织，保证矩阵行连续）
//   - M/N 任意（越界行/列片段不写回；K 必须是 16 的倍数，最后一个
//     半 tile 的越界 chunk 显式清零）
//   - beta=1 原位 epilogue：float2 先读 C 再写回
//
// SM120 要点：Ampere 风格 mma.sync m16n8k16 + FP32 累加（数值纪律同 v1）。

#include <cuda_fp16.h>
#include <cstdint>

#define V2_WARP 32
#define V2_THREADS 256
#define V2_BM 128
#define V2_BN 128
#define V2_BK 32
#define V2_STAGES 2

// smem 字节偏移（阶段内，A/B 各 [128, 32] f16 = 8KB）：
//   r ∈ [0,128)（A: M 行；B: N 行），c ∈ [0,32)（K 列，f16 下标）
__device__ __forceinline__ unsigned v2_smem_off(int r, int c) {
    return ((unsigned)(r >> 3) << 9) | ((unsigned)(c >> 3) << 7) |
           ((unsigned)(r & 7) << 4) | ((unsigned)(c & 7) << 1);
}

// cp.async.cg 16B：global → smem（saddr 为 cvta 后的 32 位共享地址）。
__device__ __forceinline__ void v2_cp_async16(unsigned saddr, const void* gaddr) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(saddr),
        "l"(gaddr));
}

__device__ __forceinline__ void v2_cp_commit() {
    asm volatile("cp.async.commit_group;\n");
}

// 等待直到 ≤ N 组 cp.async 未完成。
template <int N>
__device__ __forceinline__ void v2_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// smem 16B 清零（K 末半 tile 越界 chunk 必须为 0，否则污染累加器）。
__device__ __forceinline__ void v2_smem_zero16(unsigned saddr) {
    asm volatile("st.shared.v4.b32 [%0], {%1,%1,%1,%1};\n" ::"r"(saddr),
                 "r"(0));
}

// ldmatrix.x4：4 个 8×8 b16 矩阵 → 4 个寄存器。
// 地址语义（PTX ISA）：lane l 提供矩阵 (l/8) 的 行 (l%8) 地址（16B 行）；
// 寄存器 k = 矩阵 k 的片段元素 (row = l/4, col = 2*(l%4))。
// 矩阵基址：m0=base, m1=base+128, m2=base+delta2, m3=base+delta2+128。
//   A：delta2=512（行组 +8）；B：delta2=256（k 窗口 +16）。
__device__ __forceinline__ void v2_ldmatrix_x4(unsigned& r0, unsigned& r1,
                                                unsigned& r2, unsigned& r3,
                                                unsigned base, unsigned delta2,
                                                int lane) {
    const unsigned m = lane >> 3;
    const unsigned addr =
        base + ((m & 1) * 128) + ((m & 2) ? delta2 : 0) + ((lane & 7) << 4);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}

template <bool F16_OUT>
__device__ __forceinline__ void v2_epilogue(
    float* __restrict__ Cf, __half* __restrict__ Ch, const float c[2][8][4],
    int M, int N, int blockM, int blockN, int warpM, int warpN, int lane,
    float alpha, float beta) {
    // float2/half2 向量化要求 (row*N + col) 为偶数：col 恒偶数，故只需 N 偶数；
    // N 奇数（如 ownership N=1）时全走标量（misaligned 会崩）。
    const bool vec = (N & 1) == 0;
    const int row0 = warpM + lane / 4;
    const int col0 = warpN + (lane % 4) * 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int row = row0 + i * 16;
        if (row >= M) continue;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int col = col0 + j * 8;
            if (col >= N) continue;
            const float v0 = alpha * c[i][j][0];
            const float v1 = alpha * c[i][j][1];
            const float v2 = alpha * c[i][j][2];
            const float v3 = alpha * c[i][j][3];
            if (F16_OUT) {
                __half* cp = Ch + (size_t)row * N + col;
                if (vec && col + 1 < N) {
                    *reinterpret_cast<__half2*>(cp) = __floats2half2_rn(v0, v1);
                } else {
                    cp[0] = __float2half_rn(v0);
                    if (col + 1 < N) cp[1] = __float2half_rn(v1);
                }
                __half* cp2 = cp + 8 * N;
                if (row + 8 < M) {
                    if (vec && col + 1 < N) {
                        *reinterpret_cast<__half2*>(cp2) = __floats2half2_rn(v2, v3);
                    } else {
                        cp2[0] = __float2half_rn(v2);
                        if (col + 1 < N) cp2[1] = __float2half_rn(v3);
                    }
                }
            } else {
                float* cp = Cf + (size_t)row * N + col;
                if (vec && col + 1 < N) {
                    const float2 r = *reinterpret_cast<const float2*>(cp);
                    const float2 w = {v0 + beta * r.x, v1 + beta * r.y};
                    *reinterpret_cast<float2*>(cp) = w;
                } else {
                    cp[0] = v0 + beta * cp[0];
                    if (col + 1 < N) cp[1] = v1 + beta * cp[1];
                }
                float* cp2 = cp + 8 * N;
                if (row + 8 < M) {
                    if (vec && col + 1 < N) {
                        const float2 r = *reinterpret_cast<const float2*>(cp2);
                        const float2 w = {v2 + beta * r.x, v3 + beta * r.y};
                        *reinterpret_cast<float2*>(cp2) = w;
                    } else {
                        cp2[0] = v2 + beta * cp2[0];
                        if (col + 1 < N) cp2[1] = v3 + beta * cp2[1];
                    }
                }
            }
        }
    }
}

// 融合逐通道门控 SiLU epilogue：res = alpha*acc + beta*C，silu(res*scale+bias)。
// - F16_OUT=true（up 门）：C 写回 silu 前的 f32 残差 res（供下一块残差累积），
//   Ch 写 silu(affine(res)) 的 f16（gated 流）。
// - F16_OUT=false（down 门）：C 原位写回 silu(affine(res)) 的 f32。
// 数值与独立 kernel 完全一致：原路径 = GEMM 写 res → gate_silu 读 res。
// mma 累加器布局：c[i][j][0..1] = row 的 (col, col+1)；c[i][j][2..3] = row+8。
template <bool F16_OUT>
__device__ __forceinline__ void v2_epilogue_gatesilu(
    float* __restrict__ Cf, __half* __restrict__ Ch, const float c[2][8][4],
    const float* __restrict__ scale, const float* __restrict__ bias,
    int M, int N, int blockM, int blockN, int warpM, int warpN, int lane,
    float alpha, float beta) {
    const int row0 = warpM + lane / 4;
    const int col0 = warpN + (lane % 4) * 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int row = row0 + i * 16;
        if (row >= M) continue;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int col = col0 + j * 8;
            if (col >= N) continue;
            float* cp = Cf + (size_t)row * N + col;
            // 残差（原位读 C）：r0/r1 = row 的 col..col+1。
            const float r0 = alpha * c[i][j][0] + beta * cp[0];
            const float r1 = alpha * c[i][j][1] + beta * cp[1];
            // r2/r3 = row+8（越界行只读不写，与原 epilogue 同规则）。
            const bool has_hi = row + 8 < M;
            float* cp2 = cp + 8 * N;
            const float r2 =
                has_hi ? alpha * c[i][j][2] + beta * cp2[0] : 0.0f;
            const float r3 =
                has_hi ? alpha * c[i][j][3] + beta * cp2[1] : 0.0f;
            // affine + silu（精确 expf，与 gate_silu kernel 同公式）
            float a0 = r0 * scale[col] + bias[col];
            float a1 = r1 * scale[col + 1] + bias[col + 1];
            float a2 = r2 * scale[col] + bias[col];
            float a3 = r3 * scale[col + 1] + bias[col + 1];
            a0 = a0 / (1.0f + expf(-a0));
            a1 = a1 / (1.0f + expf(-a1));
            a2 = a2 / (1.0f + expf(-a2));
            a3 = a3 / (1.0f + expf(-a3));
            if (F16_OUT) {
                // f32 残差保留（silu 前）+ f16 gated 输出
                cp[0] = r0;
                cp[1] = r1;
                if (has_hi) {
                    cp2[0] = r2;
                    cp2[1] = r3;
                }
                __half* hp = Ch + (size_t)row * N + col;
                __half* hp2 = hp + 8 * N;
                hp[0] = __float2half_rn(a0);
                hp[1] = __float2half_rn(a1);
                if (has_hi) {
                    hp2[0] = __float2half_rn(a2);
                    hp2[1] = __float2half_rn(a3);
                }
            } else {
                cp[0] = a0;
                cp[1] = a1;
                if (has_hi) {
                    cp2[0] = a2;
                    cp2[1] = a3;
                }
            }
        }
    }
}

// 加载一个 16B chunk 到 smem（k 越界置零；行越界跳过——越界行只进
// 不会被写回的累加器，无需清零）。
template <bool IS_B>
__device__ __forceinline__ void v2_load_chunk(
    __half* sA, __half* sB, const __half* __restrict__ A,
    const __half* __restrict__ B, int stage, int row, int c16, int kk,
    int M, int N, int K, int blockM, int blockN) {
    const int k_start = kk + c16 * 8;
    const unsigned saddr =
        (unsigned)__cvta_generic_to_shared(IS_B ? sB : sA) +
        stage * (V2_BM * V2_BK * 2) + v2_smem_off(row, c16 * 8);
    if (k_start >= K) {
        // 最后一个 16 宽的半 tile：越界 chunk 清零（保证 mma 的 K 贡献为 0）。
        v2_smem_zero16(saddr);
        return;
    }
    if (IS_B) {
        if (blockN + row < N) {
            const __half* g = B + (size_t)(blockN + row) * K + k_start;
            v2_cp_async16(saddr, g);
        }
    } else {
        if (blockM + row < M) {
            const __half* g = A + (size_t)(blockM + row) * K + k_start;
            v2_cp_async16(saddr, g);
        }
    }
}

template <bool F16_OUT, bool GATESILU>
__device__ __forceinline__ void v2_hgemm_impl(
    const __half* __restrict__ A, const __half* __restrict__ B,
    float* __restrict__ Cf, __half* __restrict__ Ch, int M, int N, int K,
    float alpha, float beta, const float* __restrict__ scale,
    const float* __restrict__ bias) {
    __shared__ __half sA[V2_STAGES][V2_BM * V2_BK];
    __shared__ __half sB[V2_STAGES][V2_BN * V2_BK];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int blockM = blockIdx.x * V2_BM;
    const int blockN = blockIdx.y * V2_BN;
    const int warpM = blockM + (warp >> 1) * 32;  // 4 行组
    const int warpN = blockN + (warp & 1) * 64;   // 2 列组

    // --- 累加器（2 M 步 × 8 N 步 × 4 f32） ---
    float c[2][8][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int r = 0; r < 4; ++r) c[i][j][r] = 0.0f;

    // --- 每个 k-tile：每线程 2 个 A chunk + 2 个 B chunk（16B 粒度） ---
    // 512 chunk / 256 线程 = 2 轮：tid 与 tid+256。
    auto issue_stage = [&](int stage, int kk) {
        const int cid0 = tid;
        const int cid1 = tid + V2_THREADS;
        v2_load_chunk<false>(&sA[0][0], &sB[0][0], A, B, stage, cid0 >> 2,
                             cid0 & 3, kk, M, N, K, blockM, blockN);
        v2_load_chunk<true>(&sA[0][0], &sB[0][0], A, B, stage, cid0 >> 2,
                            cid0 & 3, kk, M, N, K, blockM, blockN);
        v2_load_chunk<false>(&sA[0][0], &sB[0][0], A, B, stage, cid1 >> 2,
                             cid1 & 3, kk, M, N, K, blockM, blockN);
        v2_load_chunk<true>(&sA[0][0], &sB[0][0], A, B, stage, cid1 >> 2,
                            cid1 & 3, kk, M, N, K, blockM, blockN);
    };

    issue_stage(0, 0);
    v2_cp_commit();

    for (int kk = 0; kk < K; kk += V2_BK) {
        const int stage = (kk / V2_BK) & 1;
        if (kk + V2_BK < K) {
            issue_stage(stage ^ 1, kk + V2_BK);
            v2_cp_commit();
            // 已提交下一组：等当前组（stage）就绪即可，预取可继续在飞。
            v2_cp_wait<1>();
        } else {
            // 最后一 tile：没有预取组可挡，必须等当前组完成。
            v2_cp_wait<0>();
        }
        __syncthreads();

        // --- 计算：2 k16 步 × (2 A 片段 + 8 B 片段) + 32 mma ---
        // 注意：smem 是块内 tile，ldmatrix 基址必须用块内相对行号
        // （warp 布局 4×2：warp 行组 = warp>>1（32 行），warp 列组 = warp&1（64 列））。
        const unsigned sA_base =
            (unsigned)__cvta_generic_to_shared(&sA[stage][0]);
        const unsigned sB_base =
            (unsigned)__cvta_generic_to_shared(&sB[stage][0]);
        const int wrow = warp >> 1;  // 块内 warp 行组 0..3
        const int wcol = warp & 1;   // 块内 warp 列组 0..1
        unsigned a_frag[2][2][4];  // [M 步][k16 步]
        unsigned b_frag[8][4];     // [N 步][4: b0(k0), b1(k0+8), b0(k0+16), b1(k0+24)]

#pragma unroll
        for (int t = 0; t < 8; ++t) {
            // B：N 步 t 的 4 个矩阵（n0..n0+8 × 阶段内 k 0..32 的 4 个 8×8
            // 窗口：k∈{0,8,16,24}，覆盖两个 k16 步）。基址只依赖行组。
            // 非 trans：B 原生 [N,K] 行主序，m8n8 片段 (row=lane/4=n,
            // col=2*(lane%4)=k) 恰为 mma B 片段布局（v1 手工构造同款）。
            const unsigned b_base =
                sB_base + (((wcol * 64 + t * 8) >> 3) << 9);
            v2_ldmatrix_x4(b_frag[t][0], b_frag[t][1], b_frag[t][2],
                           b_frag[t][3], b_base, 256, lane);
        }
#pragma unroll
        for (int s = 0; s < 2; ++s) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                // A：M 步 i × k16 步 s 的 16×16 tile（4 个 8×8 窗口：
                //   m0=(r0,k0), m1=(r0,k0+8), m2=(r0+8,k0), m3=(r0+8,k0+8)）。
                // 阶段内局部 k 列：k0 = s*16 → 基址 (s*16/8)*128 = s*256。
                const unsigned a_base =
                    sA_base + (((wrow * 32 + i * 16) >> 3) << 9) + s * 256;
                v2_ldmatrix_x4(a_frag[i][s][0], a_frag[i][s][1], a_frag[i][s][2],
                               a_frag[i][s][3], a_base, 512, lane);
            }
        }
#pragma unroll
        for (int s = 0; s < 2; ++s) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const unsigned a0 = a_frag[i][s][0];  // m0 → a0
                const unsigned a1 = a_frag[i][s][2];  // m2 → a1
                const unsigned a2 = a_frag[i][s][1];  // m1 → a2
                const unsigned a3 = a_frag[i][s][3];  // m3 → a3
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const unsigned b0 = b_frag[t][s * 2];
                    const unsigned b1 = b_frag[t][s * 2 + 1];
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(c[i][t][0]), "+f"(c[i][t][1]), "+f"(c[i][t][2]),
                          "+f"(c[i][t][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                }
            }
        }
        __syncthreads();
    }

    if (GATESILU) {
        v2_epilogue_gatesilu<F16_OUT>(Cf, Ch, c, scale, bias, M, N, blockM,
                                      blockN, warpM, warpN, lane, alpha, beta);
    } else {
        v2_epilogue<F16_OUT>(Cf, Ch, c, M, N, blockM, blockN, warpM, warpN,
                             lane, alpha, beta);
    }
}

extern "C" __global__ void __launch_bounds__(V2_THREADS) hgemm_v2_kernel(
    const __half* __restrict__ A, const __half* __restrict__ B,
    float* __restrict__ C, int M, int N, int K, float alpha, float beta) {
    v2_hgemm_impl<false, false>(A, B, C, nullptr, M, N, K, alpha, beta,
                                nullptr, nullptr);
}

extern "C" __global__ void __launch_bounds__(V2_THREADS)
hgemm_v2_f16out_kernel(const __half* __restrict__ A,
                       const __half* __restrict__ B, __half* __restrict__ C,
                       int M, int N, int K, float alpha, float beta) {
    v2_hgemm_impl<true, false>(A, B, nullptr, C, M, N, K, alpha, beta,
                               nullptr, nullptr);
}

// 融合门控 SiLU（down 门，f32 原位）：C = silu((A@B + C) * scale + bias)。
extern "C" __global__ void __launch_bounds__(V2_THREADS)
hgemm_v2_gatesilu_f32_kernel(const __half* __restrict__ A,
                             const __half* __restrict__ B,
                             float* __restrict__ C,
                             const float* __restrict__ scale,
                             const float* __restrict__ bias, int M, int N,
                             int K, float alpha, float beta) {
    v2_hgemm_impl<false, true>(A, B, C, nullptr, M, N, K, alpha, beta, scale,
                               bias);
}

// 融合门控 SiLU（up 门）：C 保留 f32 残差（silu 前），out 写 f16 gated。
extern "C" __global__ void __launch_bounds__(V2_THREADS)
hgemm_v2_gatesilu_f16_kernel(const __half* __restrict__ A,
                             const __half* __restrict__ B,
                             float* __restrict__ C,
                             __half* __restrict__ out,
                             const float* __restrict__ scale,
                             const float* __restrict__ bias, int M, int N,
                             int K, float alpha, float beta) {
    v2_hgemm_impl<true, true>(A, B, C, out, M, N, K, alpha, beta, scale, bias);
}

// ---------------------------------------------------------------------------
// GEMM tile 64×64×32 变体（小 batch 场景：M=361 时 grid 6×N/64，blocks 是
// 128-tile 的 4 倍，缓解 grid-starved SM 空闲）。warp 布局 4×2（每 warp
// 16 行 × 32 列），ldmatrix/mma 指令形状与 v2 相同，smem 4KB/级/矩阵。
// ---------------------------------------------------------------------------

#define T64_BM 64
#define T64_BN 64
#define T64_BK 64
#define T64_STAGES 2

// 64 列 tile 打包偏移（行组 = 8 行 × 128B = 1024B）。
__device__ __forceinline__ unsigned t64_smem_off64(int r, int c) {
    return ((unsigned)(r >> 3) << 10) | ((unsigned)(c >> 3) << 7) |
           ((unsigned)(r & 7) << 4) | ((unsigned)(c & 7) << 1);
}

// 每 k-tile：A/B 各 64×64 f16 = 8KB = 512 个 16B chunk（每线程 2 个）。
template <bool IS_B>
__device__ __forceinline__ void t64_load_chunk(
    __half* sA, __half* sB, const __half* __restrict__ A,
    const __half* __restrict__ B, int stage, int cid, int kk,
    int M, int N, int K, int blockM, int blockN) {
    const int row = cid >> 3;
    const int c16 = cid & 7;
    const int k_start = kk + c16 * 8;
    const unsigned saddr =
        (unsigned)__cvta_generic_to_shared(IS_B ? sB : sA) +
        stage * (T64_BM * T64_BK * 2) + t64_smem_off64(row, c16 * 8);
    if (k_start >= K) {
        v2_smem_zero16(saddr);
        return;
    }
    if (IS_B) {
        if (blockN + row < N) {
            const __half* g = B + (size_t)(blockN + row) * K + k_start;
            v2_cp_async16(saddr, g);
        }
    } else {
        if (blockM + row < M) {
            const __half* g = A + (size_t)(blockM + row) * K + k_start;
            v2_cp_async16(saddr, g);
        }
    }
}

template <bool F16_OUT>
__device__ __forceinline__ void t64_hgemm_impl(
    const __half* __restrict__ A, const __half* __restrict__ B,
    float* __restrict__ Cf, __half* __restrict__ Ch, int M, int N, int K,
    float alpha, float beta) {
    __shared__ __half sA[T64_STAGES][T64_BM * T64_BK];
    __shared__ __half sB[T64_STAGES][T64_BN * T64_BK];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int blockM = blockIdx.x * T64_BM;
    const int blockN = blockIdx.y * T64_BN;
    const int warpM = blockM + (warp >> 1) * 16;  // 4 行组 × 16 行
    const int warpN = blockN + (warp & 1) * 32;   // 2 列组 × 32 列

    // 累加器：1 M 步 × 4 N 步 × 4 f32
    float c[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) c[j][r] = 0.0f;

    auto issue_stage = [&](int stage, int kk) {
        // 每级 512 chunk，每线程 2 个 A + 2 个 B
        t64_load_chunk<false>(&sA[0][0], &sB[0][0], A, B, stage, tid, kk, M,
                              N, K, blockM, blockN);
        t64_load_chunk<false>(&sA[0][0], &sB[0][0], A, B, stage, tid + 256,
                              kk, M, N, K, blockM, blockN);
        t64_load_chunk<true>(&sA[0][0], &sB[0][0], A, B, stage, tid, kk, M, N,
                             K, blockM, blockN);
        t64_load_chunk<true>(&sA[0][0], &sB[0][0], A, B, stage, tid + 256, kk,
                             M, N, K, blockM, blockN);
    };

    issue_stage(0, 0);
    v2_cp_commit();

    for (int kk = 0; kk < K; kk += T64_BK) {
        const int stage = (kk / T64_BK) & 1;
        if (kk + T64_BK < K) {
            issue_stage(stage ^ 1, kk + T64_BK);
            v2_cp_commit();
            v2_cp_wait<1>();
        } else {
            v2_cp_wait<0>();
        }
        __syncthreads();

        const unsigned sA_base =
            (unsigned)__cvta_generic_to_shared(&sA[stage][0]);
        const unsigned sB_base =
            (unsigned)__cvta_generic_to_shared(&sB[stage][0]);
        const int wrow = warp >> 1;  // 0..3（16 行组）
        const int wcol = warp & 1;   // 0..1（32 列组）

        // B：4 个 N 步 × 8 个 k8 窗口（64 列：2 个 ldmatrix_x4 各覆盖 32 列）。
        unsigned b_frag[4][8];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const unsigned b_base =
                sB_base + (((wcol * 32 + t * 8) >> 3) << 10);
            v2_ldmatrix_x4(b_frag[t][0], b_frag[t][1], b_frag[t][2],
                           b_frag[t][3], b_base, 256, lane);
            v2_ldmatrix_x4(b_frag[t][4], b_frag[t][5], b_frag[t][6],
                           b_frag[t][7], b_base + 512, 256, lane);
        }
        // A：1 个 M 步 × 4 个 k16 步（每步 4 个 8×8）。
        // 64 列布局行组 = 8 行 × 128B = 1024B → delta2=1024（r+8 行组）。
        unsigned a_frag[4][4];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const unsigned a_base =
                sA_base + (((wrow * 16) >> 3) << 10) + s * 256;
            v2_ldmatrix_x4(a_frag[s][0], a_frag[s][1], a_frag[s][2],
                           a_frag[s][3], a_base, 1024, lane);
        }
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const unsigned a0 = a_frag[s][0];
            const unsigned a1 = a_frag[s][2];
            const unsigned a2 = a_frag[s][1];
            const unsigned a3 = a_frag[s][3];
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const unsigned b0 = b_frag[t][s * 2];
                const unsigned b1 = b_frag[t][s * 2 + 1];
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(c[t][0]), "+f"(c[t][1]), "+f"(c[t][2]),
                      "+f"(c[t][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
        __syncthreads();
    }

    // epilogue 复用 v2（2 M 步 × 8 N 步的形状不匹配，此处 1×4，直接内联）。
    const int row0 = warpM + lane / 4;
    const int col0 = warpN + (lane % 4) * 2;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int row = row0;
        const int col = col0 + t * 8;
        if (row >= M || col >= N) continue;
        const float v0 = alpha * c[t][0];
        const float v1 = alpha * c[t][1];
        const float v2 = alpha * c[t][2];
        const float v3 = alpha * c[t][3];
        if (F16_OUT) {
            __half* cp = Ch + (size_t)row * N + col;
            cp[0] = __float2half_rn(v0);
            if (col + 1 < N) cp[1] = __float2half_rn(v1);
            if (row + 8 < M) {
                cp[8 * N] = __float2half_rn(v2);
                if (col + 1 < N) cp[8 * N + 1] = __float2half_rn(v3);
            }
        } else {
            float* cp = Cf + (size_t)row * N + col;
            cp[0] = v0 + beta * cp[0];
            if (col + 1 < N) cp[1] = v1 + beta * cp[1];
            if (row + 8 < M) {
                cp[8 * N] = v2 + beta * cp[8 * N];
                if (col + 1 < N) cp[8 * N + 1] = v3 + beta * cp[8 * N + 1];
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(V2_THREADS) hgemm_t64_kernel(
    const __half* __restrict__ A, const __half* __restrict__ B,
    float* __restrict__ C, int M, int N, int K, float alpha, float beta) {
    t64_hgemm_impl<false>(A, B, C, nullptr, M, N, K, alpha, beta);
}

extern "C" __global__ void __launch_bounds__(V2_THREADS)
hgemm_t64_f16out_kernel(const __half* __restrict__ A,
                        const __half* __restrict__ B, __half* __restrict__ C,
                        int M, int N, int K, float alpha, float beta) {
    t64_hgemm_impl<true>(A, B, nullptr, C, M, N, K, alpha, beta);
}
