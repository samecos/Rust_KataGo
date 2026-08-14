// ---------------------------------------------------------------------------
// FlashAttention-2 风格 tensor-core attention（b11fix：S=361, H=12, D=32）。
//
// 与 v3（warp-per-row 标量点积）的差异：QK/PV 全部走 mma.sync m16n8k16
// 张量核，tile 128 行 query × 64 keys，smem 双缓冲 K/V 流水。
//
// 布局：
//   - 输入 q/k/v [B*H, S, D] f16 行主序（与 v3 相同）
//   - 输出 out [B*S, H*D] f16（merge 后布局，省 attn_merge kernel）
//   - smem 用 128B 打包布局（每 8×8 矩阵连续 128B，ldmatrix.x4 无冲突）
//
// 数值纪律：QK/PV 张量核 FP32 累加（f16 输入），online softmax 全程 FP32
// （精确 expf，与 v3 一致），输出 __float2half_rn。
//
// 每 block 128 行 × 全部 keys；每 warp 16 行 × 64 keys 的 score tile。
// mma C 寄存器布局（每线程 4 f32）：c0,c1 = 行 (lane/4) 的 (col,col+1)；
// c2,c3 = 行 (lane/4)+8 的 (col,col+1)，col = (lane%4)*2 + j*8。
//
// 约束：S ≤ 512，D == 32，每 block 128 行。
// ---------------------------------------------------------------------------

#include <cuda_fp16.h>
#include <cstdint>

#define FA2_ROWS 128      // 每 block query 行
#define FA2_KTILE 64      // key tile
#define FA2_THREADS 256   // 8 warps
#define FA2_STAGES 2

// ---- 与 gemm_v2.cu 相同的辅助（独立拷贝，避免跨文件链接） ----------------

// smem 字节偏移（8 行一组打包）：r 行、c 列（f16 下标）。
__device__ __forceinline__ unsigned fa2_smem_off(int r, int c) {
    return ((unsigned)(r >> 3) << 9) | ((unsigned)(c >> 3) << 7) |
           ((unsigned)(r & 7) << 4) | ((unsigned)(c & 7) << 1);
}

__device__ __forceinline__ void fa2_cp_async16(unsigned saddr,
                                               const void* gaddr) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(saddr),
                 "l"(gaddr));
}
__device__ __forceinline__ void fa2_cp_commit() {
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void fa2_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void fa2_ldmatrix_x4(unsigned& r0, unsigned& r1,
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

// ---- 主 kernel ------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(FA2_THREADS, 2)
attention_fa2_kernel(const __half* __restrict__ q,  // [B*H, S, 32]
                     const __half* __restrict__ k,  // [B*H, S, 32]
                     const __half* __restrict__ v,  // [B*H, S, 32]
                     __half* __restrict__ out,      // [B*S, H*32]
                     int s, int d, float scale, int heads) {
    // d 仅作参数占位（打包布局按 D=32 编译期定），调用方保证 d==32。
    (void)d;
    __shared__ __half s_q[FA2_ROWS * 32];
    __shared__ __half s_k[FA2_STAGES][FA2_KTILE * 32];
    __shared__ __half s_v[FA2_STAGES][FA2_KTILE * 32];
    __shared__ __half s_p[FA2_ROWS * FA2_KTILE];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int bh = blockIdx.y;               // 0..B*H-1
    const int row0 = blockIdx.x * FA2_ROWS;  // 本块 query 起始行（S 内）
    const size_t bh_off = (size_t)bh * s * 32;
    const int ntiles = (s + FA2_KTILE - 1) / FA2_KTILE;
    int ntiles_eff = ntiles;

    // 输出 merge 布局：attn [b,h,s,d] → out [b,s,h,d]
    const int h_l = bh % heads;
    const int b = bh / heads;
    const int warpM = warp * 16;  // 块内行组 0..127

    // ---- Q 加载（128 行 × 32 列 = 512 chunk，每线程 2 个） ----
    {
        const unsigned sbase = (unsigned)__cvta_generic_to_shared(s_q);
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const int cid = tid + m * FA2_THREADS;  // 0..511
            const int r = cid >> 2;
            const int c16 = cid & 3;
            if (row0 + r < s) {
                const __half* g = q + bh_off + (size_t)(row0 + r) * 32;
                fa2_cp_async16(sbase + fa2_smem_off(r, c16 * 8), g + c16 * 8);
            }
        }
    }
    // K/V tile 0 预取（每 tile 256 chunk，每线程 1 个 K + 1 个 V）
    {
        const unsigned kbase = (unsigned)__cvta_generic_to_shared(s_k[0]);
        const unsigned vbase = (unsigned)__cvta_generic_to_shared(s_v[0]);
        const int r = tid >> 2;
        const int c16 = tid & 3;
        if (r < s) {
            const __half* gk = k + bh_off + (size_t)r * 32;
            const __half* gv = v + bh_off + (size_t)r * 32;
            fa2_cp_async16(kbase + fa2_smem_off(r, c16 * 8), gk + c16 * 8);
            fa2_cp_async16(vbase + fa2_smem_off(r, c16 * 8), gv + c16 * 8);
        }
    }
    fa2_cp_commit();

    // ---- PV 累加器（1 M 步 × 4 N 步 × 4 f32）与 online softmax 状态 ----
    float acc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[j][r] = 0.0f;
    // 每 lane 2 行：low = warpM + lane/4，high = low + 8。
    float m_lo = -1e30f, m_hi = -1e30f;
    float sum_lo = 0.0f, sum_hi = 0.0f;

    const unsigned q_base = (unsigned)__cvta_generic_to_shared(s_q);

    for (int t = 0; t < ntiles_eff; ++t) {
        const int stage = t & 1;
        // 预取下一 K/V tile
        if (t + 1 < ntiles) {
            const unsigned kbase =
                (unsigned)__cvta_generic_to_shared(s_k[(t + 1) & 1]);
            const unsigned vbase =
                (unsigned)__cvta_generic_to_shared(s_v[(t + 1) & 1]);
            const int r = tid >> 2;
            const int c16 = tid & 3;
            const int kr = (t + 1) * FA2_KTILE + r;
            if (kr < s) {
                const __half* gk = k + bh_off + (size_t)kr * 32;
                const __half* gv = v + bh_off + (size_t)kr * 32;
                fa2_cp_async16(kbase + fa2_smem_off(r, c16 * 8), gk + c16 * 8);
                fa2_cp_async16(vbase + fa2_smem_off(r, c16 * 8), gv + c16 * 8);
            }
            fa2_cp_commit();
            fa2_cp_wait<1>();
        } else {
            fa2_cp_wait<0>();
        }
        __syncthreads();

        // ---- QK^T：S[128,64] = Q[128,32] @ K^T[32,64] ----
        // 每 warp：16 行 × 64 列 = 1 M 步 × 8 N 步。
        // B = K 片段（8 个 N 步 × 4 个 k8 窗口）。
        const unsigned kbase = (unsigned)__cvta_generic_to_shared(s_k[stage]);
        unsigned k_frag[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned b_base = kbase + (((j * 8) >> 3) << 9);
            fa2_ldmatrix_x4(k_frag[j][0], k_frag[j][1], k_frag[j][2],
                            k_frag[j][3], b_base, 256, lane);
        }
        // A = Q 片段（2 个 k16 步）。
        unsigned q_frag[2][4];
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            const unsigned a_base = q_base + (((warpM) >> 3) << 9) + ks * 256;
            fa2_ldmatrix_x4(q_frag[ks][0], q_frag[ks][1], q_frag[ks][2],
                            q_frag[ks][3], a_base, 512, lane);
        }
        // scores：8 N 步 × 4 regs（c0,c1 = low 行 (col,col+1)；c2,c3 = high）。
        float sc[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int r = 0; r < 4; ++r) sc[j][r] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            // ldmatrix 输出映射：a0=m0(r0,k0), a1=m2(r0+8,k0),
            // a2=m1(r0,k0+8), a3=m3(r0+8,k0+8)
            const unsigned a0 = q_frag[ks][0];
            const unsigned a1 = q_frag[ks][2];
            const unsigned a2 = q_frag[ks][1];
            const unsigned a3 = q_frag[ks][3];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned b0 = k_frag[j][ks * 2];
                const unsigned b1 = k_frag[j][ks * 2 + 1];
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(sc[j][0]), "+f"(sc[j][1]), "+f"(sc[j][2]),
                      "+f"(sc[j][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }

        // ---- online softmax（行内 64 列 = 4 lane（lane%4 组）× 8 j × 2）----
        {
            float mx_lo = -1e30f, mx_hi = -1e30f;
            float sm_lo = 0.0f, sm_hi = 0.0f;
            // 本 tile 的 key 全局起始：越界 key 的 smem 是残留垃圾（只加载了
            // kr < s 的行），必须把 score 压到 -1e30 使其 exp 贡献为 0。
            const int key_base = t * FA2_KTILE;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int key0 = key_base + (lane % 4) * 2 + j * 8;
                float s0 = sc[j][0] * scale;  // low, col
                float s1 = sc[j][1] * scale;  // low, col+1
                float s2 = sc[j][2] * scale;  // high, col
                float s3 = sc[j][3] * scale;  // high, col+1
                if (key0 >= s) {
                    s0 = -1e30f;
                    s1 = -1e30f;
                    s2 = -1e30f;
                    s3 = -1e30f;
                }
                if (key0 + 1 >= s) {
                    s1 = -1e30f;
                    s3 = -1e30f;
                }
                sc[j][0] = s0;
                sc[j][1] = s1;
                sc[j][2] = s2;
                sc[j][3] = s3;
                mx_lo = fmaxf(mx_lo, fmaxf(s0, s1));
                mx_hi = fmaxf(mx_hi, fmaxf(s2, s3));
            }
            // 4-lane 归约（lane%4 组）
#pragma unroll
            for (int off = 1; off < 4; off <<= 1) {
                mx_lo = fmaxf(mx_lo, __shfl_xor_sync(0xffffffffu, mx_lo, off));
                mx_hi = fmaxf(mx_hi, __shfl_xor_sync(0xffffffffu, mx_hi, off));
            }
            // online 合并（精确 expf，与 v3 数值纪律一致）
            const float n_lo = fmaxf(m_lo, mx_lo);
            const float n_hi = fmaxf(m_hi, mx_hi);
            const float a_lo = expf(m_lo - n_lo);
            const float a_hi = expf(m_hi - n_hi);
            sum_lo = sum_lo * a_lo;
            sum_hi = sum_hi * a_hi;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float e0 = expf(sc[j][0] - n_lo);
                const float e1 = expf(sc[j][1] - n_lo);
                const float e2 = expf(sc[j][2] - n_hi);
                const float e3 = expf(sc[j][3] - n_hi);
                sc[j][0] = e0;
                sc[j][1] = e1;
                sc[j][2] = e2;
                sc[j][3] = e3;
                sm_lo += e0 + e1;
                sm_hi += e2 + e3;
            }
            // 4-lane 归约 sum
#pragma unroll
            for (int off = 1; off < 4; off <<= 1) {
                sm_lo += __shfl_xor_sync(0xffffffffu, sm_lo, off);
                sm_hi += __shfl_xor_sync(0xffffffffu, sm_hi, off);
            }
            sum_lo += sm_lo;
            sum_hi += sm_hi;
            m_lo = n_lo;
            m_hi = n_hi;
            // 旧 acc rescale
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[j][0] *= a_lo;
                acc[j][1] *= a_lo;
                acc[j][2] *= a_hi;
                acc[j][3] *= a_hi;
            }
        }

        // ---- P 写 smem（打包布局，供 PV 的 ldmatrix） ----
        {
            const int prow_lo = warpM + lane / 4;
            const int prow_hi = prow_lo + 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int pcol = (lane % 4) * 2 + j * 8;
                *reinterpret_cast<__half2*>(
                    reinterpret_cast<char*>(s_p) + fa2_smem_off(prow_lo, pcol)) =
                    __floats2half2_rn(sc[j][0], sc[j][1]);
                *reinterpret_cast<__half2*>(
                    reinterpret_cast<char*>(s_p) + fa2_smem_off(prow_hi, pcol)) =
                    __floats2half2_rn(sc[j][2], sc[j][3]);
            }
        }
        __syncthreads();

        // ---- PV：O[128,32] += P[128,64] @ V[64,32] ----
        // mma B 是 [k16, n8]（k = keys 方向，n = d 方向）：
        // 每 ks 步（16 keys）的 B 片段 = 2 个 8-key 行组 × 8 d 列。
        // V smem [64 keys, 32 d] 打包：ldmatrix m0=(k0..7,d0..7),
        // m1=(k0..7,d8..15), m2=(k8..15,d0..7), m3=(k8..15,d8..15)。
        // B(k0..15, d0..7) = (m0, m2)；B(k0..15, d8..15) = (m1, m3)。
        const unsigned vbase = (unsigned)__cvta_generic_to_shared(s_v[stage]);
        const unsigned pbase = (unsigned)__cvta_generic_to_shared(s_p);
        unsigned v_frag[4][2][4];  // [ks][d 半窗口(0..15 / 16..31)][m0..m3]
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
            for (int dh = 0; dh < 2; ++dh) {
                const unsigned b_base =
                    vbase + ((ks * 16) >> 3) * 512 + dh * 256;
                fa2_ldmatrix_x4(v_frag[ks][dh][0], v_frag[ks][dh][1],
                                v_frag[ks][dh][2], v_frag[ks][dh][3],
                                b_base, 512, lane);
            }
        }
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            unsigned p_frag[4];
            const unsigned a_base =
                pbase + (((warpM) >> 3) << 9) + ks * 256;
            fa2_ldmatrix_x4(p_frag[0], p_frag[1], p_frag[2], p_frag[3],
                            a_base, 512, lane);
            const unsigned a0 = p_frag[0];
            const unsigned a1 = p_frag[2];
            const unsigned a2 = p_frag[1];
            const unsigned a3 = p_frag[3];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int dh = j >> 1;
                const unsigned b0 = (j & 1) ? v_frag[ks][dh][1]
                                            : v_frag[ks][dh][0];
                const unsigned b1 = (j & 1) ? v_frag[ks][dh][3]
                                            : v_frag[ks][dh][2];
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]),
                      "+f"(acc[j][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
        __syncthreads();
    }

    // ---- 最终归一 + 写回（merge 布局 [B*S, H*D]） ----
    {
        const float inv_lo = 1.0f / sum_lo;
        const float inv_hi = 1.0f / sum_hi;
        const int prow_lo = warpM + lane / 4;
        const int prow_hi = prow_lo + 8;
        __half* orow_lo =
            out + (((size_t)b * s + (row0 + prow_lo)) * heads + h_l) * 32;
        __half* orow_hi = orow_lo + 8 * heads * 32;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int pcol = (lane % 4) * 2 + j * 8;
            if (row0 + prow_lo < s) {
                orow_lo[pcol] = __float2half_rn(acc[j][0] * inv_lo);
                orow_lo[pcol + 1] = __float2half_rn(acc[j][1] * inv_lo);
            }
            if (row0 + prow_hi < s) {
                orow_hi[pcol] = __float2half_rn(acc[j][2] * inv_hi);
                orow_hi[pcol + 1] = __float2half_rn(acc[j][3] * inv_hi);
            }
        }
    }
}
