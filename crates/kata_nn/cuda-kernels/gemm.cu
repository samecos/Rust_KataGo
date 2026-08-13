// FP16 张量核 GEMM（手写内联 PTX mma.sync m16n8k16）。
//
// 计算 C[M,N] = alpha * A[M,K] * B[N,K]^T + beta * C[M,N]
//   A: [M,K] 行主序（权重 W[outC, inC] 布局）
//   B: [N,K] 行主序（输入 X[tokens, inC] 布局，即 KataGo 的 RowMajor/NHWC 约定）
//   C: [M,N] 行主序；beta=1 时 C 与 D 同指针（原位残差，KataGomo_fork 验证的
//      beta=1 epilogue 融合路径）。
//
// SM120 要点（fork 实证）：Ampere 风格 mma.sync m16n8k16 即可拿到张量核收益，
// 无需 tcgen05/wgmma；FP32 累加保证数值纪律。
//
// 约束：K 必须是 16 的倍数（调用方负责 padding）；M、N 任意。

#include <cuda_fp16.h>
#include <cstdint>

#define WARP 32
#define BLOCK_TILE_M 64
#define BLOCK_TILE_N 64
#define WARPS_PER_BLOCK 4

// 每个 warp 覆盖 16x16 的子块（2 个 m16n8k16 mma），块内 4 个 warp 沿 M 排列：
// warp w → rows [blockM + w*16, +16)，cols 循环覆盖 [blockN, blockN+64)。
extern "C" __global__ void __launch_bounds__(WARPS_PER_BLOCK * WARP)
hgemm_m16n8k16_kernel(const __half* __restrict__ A,
                      const __half* __restrict__ B,
                      float* __restrict__ C,
                      int M, int N, int K,
                      float alpha, float beta) {
    const int warp = threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;

    const int m0 = blockIdx.x * BLOCK_TILE_M + warp * 16;
    const int n0 = blockIdx.y * BLOCK_TILE_N;
    if (m0 >= M || n0 >= N) return;

    // 该 warp 沿 N 循环的 tile 数（每 tile 16 列，覆盖 BLOCK_TILE_N）。
    const int n_tiles = BLOCK_TILE_N / 16;

    // A 片段（固定 4 个 .f16x2 寄存器）：行 m0 + lane/4 与 m0 + 8 + lane/4，
    // 列 kk + (lane%4)*2 与 kk + 8 + (lane%4)*2。
    uint32_t a[4];
    // B 片段：每个 16x8 tile 2 个 .f16x2 寄存器。
    uint32_t b[n_tiles][2];
    float c[n_tiles][2][4];

    for (int t = 0; t < n_tiles; ++t)
        for (int j = 0; j < 2; ++j)
            for (int r = 0; r < 4; ++r)
                c[t][j][r] = 0.0f;

    const int a_row0 = m0 + lane / 4;
    const int a_row1 = a_row0 + 8;
    const int a_col = (lane % 4) * 2;

    for (int kk = 0; kk < K; kk += 16) {
        // --- 加载 A 片段（行主序，越界置零）---
        if (a_row0 < M) {
            const __half* ap = A + (size_t)a_row0 * K + kk + a_col;
            a[0] = *reinterpret_cast<const uint32_t*>(ap);
            a[1] = *reinterpret_cast<const uint32_t*>(ap + 8);
        } else {
            a[0] = a[1] = 0;
        }
        if (a_row1 < M) {
            const __half* ap = A + (size_t)a_row1 * K + kk + a_col;
            a[2] = *reinterpret_cast<const uint32_t*>(ap);
            a[3] = *reinterpret_cast<const uint32_t*>(ap + 8);
        } else {
            a[2] = a[3] = 0;
        }

        // --- 每个 N-tile 加载 B 片段（B 为 [N,K] 行主序 → 列主序片段）---
        const int b_row = kk + (lane / 4) * 2;
        #pragma unroll
        for (int t = 0; t < n_tiles; ++t) {
            const int b_col = n0 + t * 16 + (lane % 4);
            if (b_col < N) {
                const __half* bp = B + (size_t)b_col * K + b_row;
                b[t][0] = *reinterpret_cast<const uint32_t*>(bp);
                b[t][1] = *reinterpret_cast<const uint32_t*>(bp + 8);
            } else {
                b[t][0] = 0;
                b[t][1] = 0;
            }
        }

        // --- mma.sync m16n8k16 ---
        #pragma unroll
        for (int t = 0; t < n_tiles; ++t) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                uint32_t bb0 = b[t][0];
                uint32_t bb1 = b[t][1];
                // B 片段的 2 个寄存器对应 k 的 8..15 行？mma 布局：B frag 2 个
                // reg：reg0 = (row kk+lane/4*2+0, col n+lane%4)，(row +1, col)；
                // reg1 = (row +8, col)，(row +9, col)。
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(c[t][j][0]), "+f"(c[t][j][1]), "+f"(c[t][j][2]), "+f"(c[t][j][3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(bb0), "r"(bb1));
            }
        }
    }

    // --- 写出 C（含 beta 残差）---
    const int c_row = m0 + lane / 4;
    const int c_col = n0 + (lane % 4) * 2;
    #pragma unroll
    for (int t = 0; t < n_tiles; ++t) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int col = c_col + t * 16 + j * 8;
            if (c_row < M && col < N) {
                float* cp = C + (size_t)c_row * N + col;
                cp[0] = alpha * c[t][j][0] + beta * cp[0];
                cp[1] = alpha * c[t][j][1] + beta * cp[1];
            }
            if (c_row + 8 < M && col < N) {
                float* cp = C + (size_t)(c_row + 8) * N + col;
                cp[0] = alpha * c[t][j][2] + beta * cp[0];
                cp[1] = alpha * c[t][j][3] + beta * cp[1];
            }
        }
    }
}
