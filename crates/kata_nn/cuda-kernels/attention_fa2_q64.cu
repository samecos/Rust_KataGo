// Keep the C384 specialization; generic entry points supply the actual mid.
#ifndef FA64_MID
#define FA64_MID 384
#endif
// 64Q x 64K, 4-warp tensor-core attention candidate for S=361,H=12,D=32.
//
// This follows KataGo cudarocmopt's register-resident score/P organization,
// adapted to this backend's packed [Q|K|V] projection and load-time RoPE.
// QK and PV use f16 inputs with FP32 mma accumulation. Online softmax uses
// exact expf; no fast-math or approximate exponent instructions are used.

#include <cuda_fp16.h>
#include <cstdint>

#define FA64_BQ 64
#define FA64_BKV 64
#define FA64_WARPS 4
#define FA64_THREADS (FA64_WARPS * 32)
#define FA64_D 32
#define FA64_ST (FA64_D + 8)

// The separate q64-serial translation unit selects only the softmax reduction
// order. Keep the established q64 entry and its arithmetic unchanged.
#ifndef FA64_SERIAL_SUM
#define FA64_SERIAL_SUM 0
#endif
#ifndef FA64_KERNEL_NAME
#define FA64_KERNEL_NAME attention_fa2_q64_kernel
#endif

__device__ __forceinline__ void fa64_mma(float c[4], const uint32_t a[4],
                                         const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
          "r"(b[1]));
}

__device__ __forceinline__ uint32_t fa64_pack_half2(float lo, float hi) {
    __half2 h = __float22half2_rn(make_float2(lo, hi));
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ void fa64_ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    const __half* row_ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(row_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}

__device__ __forceinline__ void fa64_ldmatrix_x4_trans(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    const __half* row_ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(row_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}

__device__ __forceinline__ void fa64_cp_async16(void* dst, const void* src,
                                                 bool valid) {
    uint32_t saddr = (uint32_t)__cvta_generic_to_shared(dst);
    int src_size = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::
                     "r"(saddr), "l"(src), "r"(src_size));
}

__device__ __forceinline__ void fa64_cp_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void fa64_cp_wait_all() {
    asm volatile("cp.async.wait_group 0;\n");
}

extern "C" __global__ void __launch_bounds__(FA64_THREADS)
FA64_KERNEL_NAME(const __half* __restrict__ qkv,
                         const float* __restrict__ rope_cos,
                         const float* __restrict__ rope_sin,
                         __half* __restrict__ out, int s, int d, float scale,
                         int heads) {
    (void)d;
    constexpr int kt = FA64_D / 16;
    constexpr int nt_s = FA64_BKV / 8;
    constexpr int kt_pv = FA64_BKV / 16;
    constexpr int nt_o = FA64_D / 8;
    constexpr int vecs_per_tile = FA64_BKV * (FA64_D / 8);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gr = lane >> 2;
    const int q4 = lane & 3;
    const int bh = blockIdx.y;
    const int h = bh % heads;
    const int b = bh / heads;
    const int q_block = blockIdx.x * FA64_BQ;
    const int ntiles = (s + FA64_BKV - 1) / FA64_BKV;

    __shared__ __align__(16) __half k_tiles[2][FA64_BKV * FA64_ST];
    __shared__ __align__(16) __half v_tiles[2][FA64_BKV * FA64_ST];

    auto issue_tile = [&](int tile, int buf) {
        const int key_start = tile * FA64_BKV;
#pragma unroll
        for (int t = tid; t < vecs_per_tile; t += FA64_THREADS) {
            const int row = t / (FA64_D / 8);
            const int d0 = (t % (FA64_D / 8)) * 8;
            const int key = key_start + row;
            const bool valid = key < s;
            uint4 rotated = make_uint4(0, 0, 0, 0);
            if (valid) {
                const __half* gk = qkv + ((size_t)b * s + key) * (3 * FA64_MID) +
                                   FA64_MID + h * FA64_D + d0;
                __half2 pairs[4];
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const float x = __half2float(gk[p * 2]);
                    const float y = __half2float(gk[p * 2 + 1]);
                    const int pair = d0 / 2 + p;
                    const size_t ro = (size_t)key * heads * 16 + h * 16 + pair;
                    const float co = rope_cos[ro];
                    const float sn = rope_sin[ro];
                    pairs[p] = __floats2half2_rn(x * co - y * sn,
                                                 x * sn + y * co);
                }
                rotated = *reinterpret_cast<uint4*>(pairs);
            }
            *reinterpret_cast<uint4*>(&k_tiles[buf][row * FA64_ST + d0]) =
                rotated;
            const __half* gv = qkv + ((size_t)b * s + (valid ? key : 0)) *
                                         (3 * FA64_MID) +
                               (2 * FA64_MID) + h * FA64_D + d0;
            fa64_cp_async16(&v_tiles[buf][row * FA64_ST + d0], gv, valid);
        }
        fa64_cp_commit();
    };

    issue_tile(0, 0);

    // Stage and rotate this block's Q rows through buffer 1, then retain the
    // mma A fragments in registers for the full online-softmax loop.
#pragma unroll
    for (int t = tid; t < FA64_BQ * (FA64_D / 8); t += FA64_THREADS) {
        const int row = t / (FA64_D / 8);
        const int d0 = (t % (FA64_D / 8)) * 8;
        const int qpos = q_block + row;
        uint4 rotated = make_uint4(0, 0, 0, 0);
        if (qpos < s) {
            const __half* gq = qkv + ((size_t)b * s + qpos) * (3 * FA64_MID) +
                               h * FA64_D + d0;
            __half2 pairs[4];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const float x = __half2float(gq[p * 2]);
                const float y = __half2float(gq[p * 2 + 1]);
                const int pair = d0 / 2 + p;
                const size_t ro = (size_t)qpos * heads * 16 + h * 16 + pair;
                const float co = rope_cos[ro];
                const float sn = rope_sin[ro];
                pairs[p] = __floats2half2_rn(x * co - y * sn,
                                             x * sn + y * co);
            }
            rotated = *reinterpret_cast<uint4*>(pairs);
        }
        *reinterpret_cast<uint4*>(&k_tiles[1][row * FA64_ST + d0]) = rotated;
    }
    __syncthreads();

    uint32_t q_a[kt][4];
    {
        const int r0 = warp * 16 + gr;
#pragma unroll
        for (int kstep = 0; kstep < kt; ++kstep) {
            const int d0 = kstep * 16 + q4 * 2;
            q_a[kstep][0] = *reinterpret_cast<const uint32_t*>(
                &k_tiles[1][r0 * FA64_ST + d0]);
            q_a[kstep][1] = *reinterpret_cast<const uint32_t*>(
                &k_tiles[1][(r0 + 8) * FA64_ST + d0]);
            q_a[kstep][2] = *reinterpret_cast<const uint32_t*>(
                &k_tiles[1][r0 * FA64_ST + d0 + 8]);
            q_a[kstep][3] = *reinterpret_cast<const uint32_t*>(
                &k_tiles[1][(r0 + 8) * FA64_ST + d0 + 8]);
        }
    }

    float out_frag[nt_o][4];
    float row_max[2] = {-1e30f, -1e30f};
    float row_sum[2] = {0.0f, 0.0f};
#pragma unroll
    for (int n = 0; n < nt_o; ++n)
#pragma unroll
        for (int r = 0; r < 4; ++r) out_frag[n][r] = 0.0f;

    for (int tile = 0; tile < ntiles; ++tile) {
        fa64_cp_wait_all();
        __syncthreads();
        if (tile + 1 < ntiles) issue_tile(tile + 1, (tile + 1) & 1);

        const __half* k_tile = k_tiles[tile & 1];
        const __half* v_tile = v_tiles[tile & 1];
        float score[nt_s][4];
#pragma unroll
        for (int n = 0; n < nt_s; ++n)
#pragma unroll
            for (int r = 0; r < 4; ++r) score[n][r] = 0.0f;

#pragma unroll
        for (int kstep = 0; kstep < kt; ++kstep) {
#pragma unroll
            for (int np = 0; np < nt_s / 2; ++np) {
                uint32_t kb[4];
                const __half* rp =
                    &k_tile[(np * 16 + ((lane >> 4) & 1) * 8 + (lane & 7)) *
                                FA64_ST +
                            kstep * 16 + ((lane >> 3) & 1) * 8];
                fa64_ldmatrix_x4(kb[0], kb[1], kb[2], kb[3], rp);
                fa64_mma(score[2 * np], q_a[kstep], kb);
                fa64_mma(score[2 * np + 1], q_a[kstep], kb + 2);
            }
        }

        float part_max0[nt_s], part_max1[nt_s];
#if FA64_SERIAL_SUM
        float serial_max0 = -1e30f, serial_max1 = -1e30f;
#endif
        const int key_start = tile * FA64_BKV;
#pragma unroll
        for (int n = 0; n < nt_s; ++n) {
            const int key0 = key_start + n * 8 + q4 * 2;
            float s0 = score[n][0] * scale;
            float s1 = score[n][1] * scale;
            float s2 = score[n][2] * scale;
            float s3 = score[n][3] * scale;
            if (key0 >= s) s0 = s2 = -1e30f;
            if (key0 + 1 >= s) s1 = s3 = -1e30f;
            score[n][0] = s0;
            score[n][1] = s1;
            score[n][2] = s2;
            score[n][3] = s3;
            part_max0[n] = fmaxf(s0, s1);
            part_max1[n] = fmaxf(s2, s3);
#if FA64_SERIAL_SUM
            serial_max0 = fmaxf(serial_max0, part_max0[n]);
            serial_max1 = fmaxf(serial_max1, part_max1[n]);
#endif
        }
#if FA64_SERIAL_SUM
        float tile_max0 = serial_max0;
        float tile_max1 = serial_max1;
#else
#pragma unroll
        for (int width = nt_s / 2; width >= 1; width >>= 1) {
#pragma unroll
            for (int i = 0; i < width; ++i) {
                part_max0[i] = fmaxf(part_max0[i], part_max0[i + width]);
                part_max1[i] = fmaxf(part_max1[i], part_max1[i + width]);
            }
        }
        float tile_max0 = part_max0[0];
        float tile_max1 = part_max1[0];
#endif
        tile_max0 = fmaxf(tile_max0,
                          __shfl_xor_sync(0xffffffffu, tile_max0, 1));
        tile_max0 = fmaxf(tile_max0,
                          __shfl_xor_sync(0xffffffffu, tile_max0, 2));
        tile_max1 = fmaxf(tile_max1,
                          __shfl_xor_sync(0xffffffffu, tile_max1, 1));
        tile_max1 = fmaxf(tile_max1,
                          __shfl_xor_sync(0xffffffffu, tile_max1, 2));

        const float next_max0 = fmaxf(row_max[0], tile_max0);
        const float next_max1 = fmaxf(row_max[1], tile_max1);
        const float alpha0 = expf(row_max[0] - next_max0);
        const float alpha1 = expf(row_max[1] - next_max1);
        row_max[0] = next_max0;
        row_max[1] = next_max1;

        float part_sum0[nt_s], part_sum1[nt_s];
#if FA64_SERIAL_SUM
        // Match q128's chronological arithmetic, including the separate scale
        // and add of the previous normalizer. Do not introduce approximate
        // exponentials, change the half P boundary, or reorder the PV MMAs.
        float tile_sum0 = 0.0f, tile_sum1 = 0.0f;
        row_sum[0] = row_sum[0] * alpha0;
        row_sum[1] = row_sum[1] * alpha1;
#endif
#pragma unroll
        for (int n = 0; n < nt_s; ++n) {
            const float p0 = expf(score[n][0] - next_max0);
            const float p1 = expf(score[n][1] - next_max0);
            const float p2 = expf(score[n][2] - next_max1);
            const float p3 = expf(score[n][3] - next_max1);
            score[n][0] = p0;
            score[n][1] = p1;
            score[n][2] = p2;
            score[n][3] = p3;
            part_sum0[n] = p0 + p1;
            part_sum1[n] = p2 + p3;
#if FA64_SERIAL_SUM
            tile_sum0 += part_sum0[n];
            tile_sum1 += part_sum1[n];
#endif
        }
#if !FA64_SERIAL_SUM
#pragma unroll
        for (int width = nt_s / 2; width >= 1; width >>= 1) {
#pragma unroll
            for (int i = 0; i < width; ++i) {
                part_sum0[i] += part_sum0[i + width];
                part_sum1[i] += part_sum1[i + width];
            }
        }
        float tile_sum0 = part_sum0[0];
        float tile_sum1 = part_sum1[0];
#endif
        tile_sum0 += __shfl_xor_sync(0xffffffffu, tile_sum0, 1);
        tile_sum0 += __shfl_xor_sync(0xffffffffu, tile_sum0, 2);
        tile_sum1 += __shfl_xor_sync(0xffffffffu, tile_sum1, 1);
        tile_sum1 += __shfl_xor_sync(0xffffffffu, tile_sum1, 2);
#if FA64_SERIAL_SUM
        row_sum[0] += tile_sum0;
        row_sum[1] += tile_sum1;
#else
        row_sum[0] = row_sum[0] * alpha0 + tile_sum0;
        row_sum[1] = row_sum[1] * alpha1 + tile_sum1;
#endif

#pragma unroll
        for (int n = 0; n < nt_o; ++n) {
            out_frag[n][0] *= alpha0;
            out_frag[n][1] *= alpha0;
            out_frag[n][2] *= alpha1;
            out_frag[n][3] *= alpha1;
        }

        uint32_t p_a[kt_pv][4];
#pragma unroll
        for (int kstep = 0; kstep < kt_pv; ++kstep) {
            p_a[kstep][0] =
                fa64_pack_half2(score[2 * kstep][0], score[2 * kstep][1]);
            p_a[kstep][1] =
                fa64_pack_half2(score[2 * kstep][2], score[2 * kstep][3]);
            p_a[kstep][2] = fa64_pack_half2(score[2 * kstep + 1][0],
                                             score[2 * kstep + 1][1]);
            p_a[kstep][3] = fa64_pack_half2(score[2 * kstep + 1][2],
                                             score[2 * kstep + 1][3]);
        }

#pragma unroll
        for (int c = 0; c < kt_pv / 2; ++c) {
#pragma unroll
            for (int n = 0; n < nt_o; ++n) {
                uint32_t vb[4];
                const __half* rp =
                    &v_tile[(c * 32 + lane) * FA64_ST + n * 8];
                fa64_ldmatrix_x4_trans(vb[0], vb[1], vb[2], vb[3], rp);
                fa64_mma(out_frag[n], p_a[2 * c], vb);
                fa64_mma(out_frag[n], p_a[2 * c + 1], vb + 2);
            }
        }
    }

    const float inv0 = 1.0f / row_sum[0];
    const float inv1 = 1.0f / row_sum[1];
    const int r0 = q_block + warp * 16 + gr;
    const int r1 = r0 + 8;
    if (r0 < s) {
        __half* dst = out + (((size_t)b * s + r0) * heads + h) * FA64_D;
#pragma unroll
        for (int n = 0; n < nt_o; ++n) {
            const int d0 = n * 8 + q4 * 2;
            *reinterpret_cast<uint32_t*>(dst + d0) =
                fa64_pack_half2(out_frag[n][0] * inv0,
                                out_frag[n][1] * inv0);
        }
    }
    if (r1 < s) {
        __half* dst = out + (((size_t)b * s + r1) * heads + h) * FA64_D;
#pragma unroll
        for (int n = 0; n < nt_o; ++n) {
            const int d0 = n * 8 + q4 * 2;
            *reinterpret_cast<uint32_t*>(dst + d0) =
                fa64_pack_half2(out_frag[n][2] * inv1,
                                out_frag[n][3] * inv1);
        }
    }
}
