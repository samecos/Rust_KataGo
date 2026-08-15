// INT8 张量核 GEMM（手写内联 PTX mma.sync m16n8k32，s32 累加）。
//
// 计算 C16[M,N] = dequant(A[M,K] @ B[N,K]^T) = dot * sa[row] * sb[col]
//   A: [M,K] 行主序 int8（量化激活）；sa: [M] f32 每行 scale
//   B: [N,K] 行主序 int8（量化权重）；sb: [N] f32 每输出通道 scale
//   C: [M,N] 行主序 f16（epilogue 内反量化直出，消费者零额外 pass）
//
// 背景：cublasLt INT8(imma) 在 CUDA13/SM120 已不可用（全部布局组合
// heuristic 拒绝）；FP8 E4M3 端到端精度不达标（KL 2e-2，翻非平局点）。
// INT8 per-row×per-channel 模拟精度达标（policy KL 1.2e-3，top-1 翻转
// 仅限 margin<0.01 平局），故自研 imma kernel。
//
// 结构（t64 骨架的 int8 变体）：
//   - tile 64×64×64 int8，2 级 cp.async 双缓冲，256 线程（8 warps 4×2）
//   - smem 行 stride 80B：16B 对齐（cp.async 要求）且 4B 片段读无 bank
//     冲突（行 r 起始 bank = 20r mod 32，8 行全错位）
//   - 片段用 4B 直接读（v1 不用 ldmatrix；k32 A 片段 = (lane%4)*4 与
//     +16 偏移两处 4B，B 片段同理）
//   - K 须为 64 的倍数（trunk K∈{384,1152} 均满足；末 chunk 越界清零）
//
// SM120：imma m16n8k32 s8 峰值 = 2× FP16（5070Ti 175.76 TOPS）。

#include <cuda_fp16.h>
#include <cstdint>

#define I8_THREADS 256
#define I8_BM 64
#define I8_BN 64
#define I8_BK 64
#define I8_STAGES 2
#define I8_STRIDE 80  // smem 行字节 stride（64B 数据 + 16B pad）

__device__ __forceinline__ void i8_cp_async16(unsigned saddr, const void* gaddr) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(saddr),
        "l"(gaddr));
}

__device__ __forceinline__ void i8_cp_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void i8_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void i8_smem_zero16(unsigned saddr) {
    asm volatile("st.shared.v4.b32 [%0], {%1,%1,%1,%1};\n" ::"r"(saddr),
                 "r"(0));
}

// 每级每矩阵 64 行 × 64B = 256 个 16B chunk，256 线程每线程 1 个。
// cid → row = cid>>2，chunk = cid&3。
template <bool IS_B>
__device__ __forceinline__ void i8_load_chunk(
    int8_t* sA, int8_t* sB, const int8_t* __restrict__ A,
    const int8_t* __restrict__ B, int stage, int cid, int kk,
    int M, int N, int K, int blockM, int blockN) {
    const int row = cid >> 2;
    const int c16 = cid & 3;
    const int k_start = kk + c16 * 16;
    const unsigned saddr =
        (unsigned)__cvta_generic_to_shared(IS_B ? sB : sA) +
        stage * (I8_BM * I8_STRIDE) + (unsigned)(row * I8_STRIDE + c16 * 16);
    const int limit = IS_B ? N : M;
    const int base = IS_B ? blockN : blockM;
    if (k_start >= K || base + row >= limit) {
        i8_smem_zero16(saddr);
        return;
    }
    const int8_t* g =
        (IS_B ? B : A) + (size_t)(base + row) * (size_t)K + k_start;
    i8_cp_async16(saddr, g);
}

extern "C" __global__ void __launch_bounds__(I8_THREADS)
igemm_t64_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                 const float* __restrict__ sa, const float* __restrict__ sb,
                 __half* __restrict__ C, int M, int N, int K) {
    __shared__ int8_t sA[I8_STAGES][I8_BM * I8_STRIDE];
    __shared__ int8_t sB[I8_STAGES][I8_BN * I8_STRIDE];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int blockM = blockIdx.x * I8_BM;
    const int blockN = blockIdx.y * I8_BN;
    const int warpM = blockM + (warp >> 1) * 16;  // 4 行组 × 16 行
    const int warpN = blockN + (warp & 1) * 32;   // 2 列组 × 32 列

    // 累加器：4 个 N 步（8 列）× 4 s32
    int32_t c[4][4];
#pragma unroll
    for (int t = 0; t < 4; ++t)
#pragma unroll
        for (int r = 0; r < 4; ++r) c[t][r] = 0;

    auto issue_stage = [&](int stage, int kk) {
        i8_load_chunk<false>(&sA[0][0], &sB[0][0], A, B, stage, tid, kk,
                             M, N, K, blockM, blockN);
        i8_load_chunk<true>(&sA[0][0], &sB[0][0], A, B, stage, tid, kk,
                            M, N, K, blockM, blockN);
    };

    issue_stage(0, 0);
    i8_cp_commit();

    // 片段读地址（线程固定部分）
    const int a_row0 = warpM - blockM + lane / 4;      // smem 内行 0..63
    const int b_row0 = warpN - blockN + lane / 4;      // smem 内行 0..63
    const int kq = (lane & 3) * 4;                     // k 四元组偏移

    for (int kk = 0; kk < K; kk += I8_BK) {
        const int stage = (kk / I8_BK) & 1;
        if (kk + I8_BK < K) {
            issue_stage(stage ^ 1, kk + I8_BK);
            i8_cp_commit();
            i8_cp_wait<1>();
        } else {
            i8_cp_wait<0>();
        }
        __syncthreads();

        const int8_t* sAs = &sA[stage][0];
        const int8_t* sBs = &sB[stage][0];

#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {  // 2 个 k32 步
            const int k0 = ks * 32;
            // A 片段：a0=(r,c) a1=(r+8,c) a2=(r,c+16) a3=(r+8,c+16)
            uint32_t a0, a1, a2, a3;
            {
                const int8_t* p0 = sAs + a_row0 * I8_STRIDE + k0 + kq;
                const int8_t* p1 = p0 + 8 * I8_STRIDE;
                a0 = *reinterpret_cast<const uint32_t*>(p0);
                a1 = *reinterpret_cast<const uint32_t*>(p1);
                a2 = *reinterpret_cast<const uint32_t*>(p0 + 16);
                a3 = *reinterpret_cast<const uint32_t*>(p1 + 16);
            }
#pragma unroll
            for (int t = 0; t < 4; ++t) {  // 4 个 8 列步
                const int8_t* pb =
                    sBs + (b_row0 + t * 8) * I8_STRIDE + k0 + kq;
                uint32_t b0 = *reinterpret_cast<const uint32_t*>(pb);
                uint32_t b1 = *reinterpret_cast<const uint32_t*>(pb + 16);
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                    "{%0,%1,%2,%3};\n"
                    : "+r"(c[t][0]), "+r"(c[t][1]), "+r"(c[t][2]),
                      "+r"(c[t][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0),
                      "r"(b1));
            }
        }
        __syncthreads();
    }

    // epilogue：反量化直出 f16。C 片段：c0=(r,c) c1=(r,c+1) c2=(r+8,c)
    // c3=(r+8,c+1)，r = warpM + lane/4，c = warpN + (lane%4)*2。
    const int r0 = warpM + lane / 4;
    const int c0 = warpN + (lane & 3) * 2;
    const float sa0 = (r0 < M) ? sa[r0] : 0.0f;
    const float sa1 = (r0 + 8 < M) ? sa[r0 + 8] : 0.0f;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int col = c0 + t * 8;
        if (col >= N) continue;
        const float sb0 = sb[col];
        const float sb1 = (col + 1 < N) ? sb[col + 1] : 0.0f;
        if (r0 < M) {
            __half* cp = C + (size_t)r0 * N + col;
            cp[0] = __float2half_rn((float)c[t][0] * sa0 * sb0);
            if (col + 1 < N) cp[1] = __float2half_rn((float)c[t][1] * sa0 * sb1);
        }
        if (r0 + 8 < M) {
            __half* cp = C + (size_t)(r0 + 8) * N + col;
            cp[0] = __float2half_rn((float)c[t][2] * sa1 * sb0);
            if (col + 1 < N) cp[1] = __float2half_rn((float)c[t][3] * sa1 * sb1);
        }
    }
}
