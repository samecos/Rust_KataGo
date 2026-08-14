// Rust_KataGo CUDA 推理执行器（cuda_exec）的补充 kernel。
//
// 覆盖层图执行所需、basic/gemm/elementwise/attention 未提供的算子：
// 初始卷积 im2col、conv+全局偏置+门控 SiLU、逐通道 gate SiLU、RoPE+QKV 切分、
// attention 输出拼回、BN 仿射+SiLU、头池化（mean/meanScale/max 与
// mean/meanScale/meanQuad）、策略头 g 分支合并、策略输出拼接等。
//
// 数值纪律（对齐 KataGomo_fork）：FP16 输入先转 FP32 计算、激活用精确形式
// （SiLU = x/(1+exp(-x))）、结果舍入回 FP16；不使用近似指令。
//
// 布局约定（与 cuda_exec.rs 一致）：
//   - 中间激活一律 NHWC 行主序：x[M, C]，M = batch * seq_len。
//   - GEMM 权重 out-first：[N, K]（K 已 pad 到 16 的倍数，pad 列恒零）。
//   - attention 的 q/k/v：[B*H, S, D]。
//   - RoPE 表：[S, H*D/2] f32（保留 f32 精度，见 IR 注释）。

#include <cuda_fp16.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// 初始卷积
// ---------------------------------------------------------------------------

// 3x3 same 卷积（pad=1）的 im2col：spatial [B, C, W, W] NCHW f32
// -> cols [B*S, KP] f16（K = C*9 实列，K..KP 保持零）。
extern "C" __global__ void im2col_f16_kernel(const float* __restrict__ spatial,
                                             __half* __restrict__ cols,
                                             int B, int S, int W, int C, int K, int KP) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * K;
    if (i >= total) return;
    int col = i % K;
    int row = (i / K) % S;
    int b = i / (S * K);
    int y = row / W;
    int x = row % W;
    int c = col / 9;
    int ky = (col % 9) / 3;
    int kx = col % 3;
    int sy = y + ky - 1;
    int sx = x + kx - 1;
    float v = 0.0f;
    if (sy >= 0 && sy < W && sx >= 0 && sx < W) {
        v = spatial[((size_t)(b * C + c) * W + sy) * W + sx];
    }
    cols[(size_t)(b * S + row) * KP + col] = __float2half(v);
}

// conv GEMM 输出 + 全局输入线性变换（逐通道广播）→ raw 768 流 f32。
// （门控 SiLU 在下一 kernel 单独做：导出图里块残差加的是 gate **前**的
//   raw 流，raw 与 gated 两条 768 流都要保留。raw 流保持 f32 累积以抑制
//   11 个块的残差舍入漂移；gated 流仍 f16，喂给下游张量核 GEMM。）
// conv [M, C] f32（gemm_out）、glob [B, G] f32、gw [C, G] f32、out [M, C] f32。
extern "C" __global__ void conv_bias_gate_kernel(const float* __restrict__ conv,
                                                 const float* __restrict__ glob,
                                                 const float* __restrict__ gw,
                                                 float* __restrict__ out,
                                                 int B, int S, int C, int G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * S * C) return;
    int c = i % C;
    int s = (i / C) % S;
    int b = i / (S * C);
    float v = conv[i];
    const float* g = glob + (size_t)b * G;
    const float* w = gw + (size_t)c * G;
    for (int k = 0; k < G; ++k) v += g[k] * w[k];
    out[i] = v;
}

// 融合版（G3）：conv+global→out f32（残差保留）+ gated f16（silu(affine)）。
// 数值与独立 gate_silu_out_f16_kernel 完全一致（同公式同舍入）。
extern "C" __global__ void conv_bias_gate_silu_kernel(
    const float* __restrict__ conv, const float* __restrict__ glob,
    const float* __restrict__ gw, float* __restrict__ out,
    __half* __restrict__ gated, const float* __restrict__ scale,
    const float* __restrict__ bias, int B, int S, int C, int G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * S * C) return;
    int c = i % C;
    int s = (i / C) % S;
    int b = i / (S * C);
    float v = conv[i];
    const float* g = glob + (size_t)b * G;
    const float* w = gw + (size_t)c * G;
    for (int k = 0; k < G; ++k) v += g[k] * w[k];
    out[i] = v;
    float a = v * scale[c] + bias[c];
    a = a / (1.0f + expf(-a));
    gated[i] = __float2half(a);
}

// ---------------------------------------------------------------------------
// 通用逐元素 / 布局辅助
// ---------------------------------------------------------------------------

// 把 f16 缓冲的 [k, stride) 列清零（GEMM 前保证 A 的 K-pad 区域为零）。
extern "C" __global__ void zero_pad_f16_kernel(__half* __restrict__ a,
                                               int m, int k, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int w = stride - k;
    if (w <= 0 || i >= m * w) return;
    int row = i / w;
    a[(size_t)row * stride + k + (i % w)] = __half{0};
}

// f32 逐元素加逐通道 bias（原位）：x[i] = x[i] + bias[i % c]。
extern "C" __global__ void f32_bias_add_kernel(float* __restrict__ x,
                                               const float* __restrict__ bias,
                                               int n, int c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = x[i] + bias[i % c];
}

// RMSNorm（f32 入 f16 出）：每行 ncols 元素。
// out = in / sqrt(mean(in^2) + eps) * scale；每 block 一行，128 线程。
extern "C" __global__ void rms_norm_f32_kernel(const float* __restrict__ in,
                                               const float* __restrict__ scale,
                                               __half* __restrict__ out,
                                               float eps, int ncols) {
    int row = blockIdx.x;
    const float* x = in + (size_t)row * ncols;
    __half* y = out + (size_t)row * ncols;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) {
        sum += x[i] * x[i];
    }
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }
    __shared__ float s_warpsum[8];
    if ((threadIdx.x & 31) == 0) s_warpsum[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < blockDim.x / 32; ++i) total += s_warpsum[i];
        s_warpsum[1] = rsqrtf(total / (float)ncols + eps);
    }
    __syncthreads();
    float rstd = s_warpsum[1];
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) {
        y[i] = __float2half(x[i] * rstd * scale[i]);
    }
}

// 逐通道门控 SiLU（f32 原位）：x[i] = silu(x[i] * scale[i%c] + bias[i%c])。
// 384 维块尾门用（f32 残差流上直接做，上投影前再转 f16）。
extern "C" __global__ void gate_silu_f32_kernel(float* __restrict__ x,
                                                const float* __restrict__ scale,
                                                const float* __restrict__ bias,
                                                int n, int c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i] * scale[i % c] + bias[i % c];
    v = v / (1.0f + expf(-v));
    x[i] = v;
}

// 逐通道门控 SiLU（原位 f16）：x[i] = silu(x[i] * scale[i%c] + bias[i%c])
// = (x*s+b) * sigmoid(x*s+b)（严格按导出图 match_gated_silu 语义）。
extern "C" __global__ void gate_silu_f16_kernel(__half* __restrict__ x,
                                                const float* __restrict__ scale,
                                                const float* __restrict__ bias,
                                                int n, int c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(x[i]) * scale[i % c] + bias[i % c];
    v = v / (1.0f + expf(-v));
    x[i] = __float2half(v);
}

// 逐通道门控 SiLU（异位，f32 入 f16 出）：
// out[i] = silu(in[i] * scale[i%c] + bias[i%c])。
// 768 维块边界门用：raw 流（f32 残差）与 gated 流（f16 下一块输入）分开保存。
extern "C" __global__ void gate_silu_out_f16_kernel(const float* __restrict__ in,
                                                    const float* __restrict__ scale,
                                                    const float* __restrict__ bias,
                                                    __half* __restrict__ out,
                                                    int n, int c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i] * scale[i % c] + bias[i % c];
    v = v / (1.0f + expf(-v));
    out[i] = __float2half(v);
}

// f32 原位累加：x[i] += a[i]（768 上投影残差：f32 raw 流 + f32 GEMM 输出）。
extern "C" __global__ void f32_add_inplace_kernel(float* __restrict__ x,
                                                  const float* __restrict__ a,
                                                  int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += a[i];
}

// bias + SiLU（f32 输入，f16 输出）：out[i] = silu(in[i] + bias[i%c])。
extern "C" __global__ void add_bias_silu_f16_kernel(const float* __restrict__ in,
                                                    const float* __restrict__ bias,
                                                    __half* __restrict__ out,
                                                    int n, int c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i] + bias[i % c];
    v = v / (1.0f + expf(-v));
    out[i] = __float2half(v);
}

// bias + SiLU（f32 就地）：x[i] = silu(x[i] + bias[i%c])。
extern "C" __global__ void bias_silu_f32_kernel(float* __restrict__ x,
                                                const float* __restrict__ bias,
                                                int n, int c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i] + bias[i % c];
    v = v / (1.0f + expf(-v));
    x[i] = v;
}

// SiLU（f32 输入，f16 输出）。
extern "C" __global__ void silu_f32_to_f16_kernel(const float* __restrict__ in,
                                                  __half* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
    v = v / (1.0f + expf(-v));
    out[i] = __float2half(v);
}

// 残差（原位）：b[i] = a[i] + half2float(b[i])（f32 计算 f16 存储）。
extern "C" __global__ void f32_add_f16_kernel(const float* __restrict__ a,
                                              __half* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    b[i] = __float2half(a[i] + __half2float(b[i]));
}

// trunk 末端 BN（eval 仿射）+ SiLU（f32 入 f16 出，写 gated 流供头部 GEMM）：
// out[i] = silu((x[i]-mean[c])/std[c]*gamma[c]+beta[c])。
// 注：ONNX 里 BN 之后还有 ×on_board 掩码乘法，19 路 on_board 恒 1，故省略。
extern "C" __global__ void bn_silu_f16_kernel(const float* __restrict__ x,
                                              const float* __restrict__ mean,
                                              const float* __restrict__ std,
                                              const float* __restrict__ gamma,
                                              const float* __restrict__ beta,
                                              __half* __restrict__ out,
                                              int n, int c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int ch = i % c;
    float v = (x[i] - mean[ch]) / std[ch] * gamma[ch] + beta[ch];
    v = v / (1.0f + expf(-v));
    out[i] = __float2half(v);
}

// ---------------------------------------------------------------------------
// attention：QKV 切分 + RoPE
// ---------------------------------------------------------------------------

// 从 qkv GEMM 输出 [M, 3*H*D] f32 拆出 Q/K/V，Q/K 做 RoPE，写成 [B*H, S, D] f16。
// V 段不做 RoPE（导出图 v 路径仅 Transpose [0,2,1,3]）。
// RoPE（严格按 IR）：末维拆 (D/2, 2) 半维对，a=偶下标、b=奇下标：
//   rot(a,b) = (a*cos[s, h*D/2+p] - b*sin[s, h*D/2+p],
//               a*sin[s, h*D/2+p] + b*cos[s, h*D/2+p])
// cos/sin 为 [S, H*D/2] f32。grid=(S, B*H)，block=32（每线程一个输出维）。
extern "C" __global__ void qkv_rope_kernel(const float* __restrict__ qkv,
                                           const float* __restrict__ cos,
                                           const float* __restrict__ sin,
                                           __half* __restrict__ q,
                                           __half* __restrict__ k,
                                           __half* __restrict__ v,
                                           int B, int H, int S, int D) {
    int s = blockIdx.x;
    int bh = blockIdx.y;
    int d = threadIdx.x;
    if (d >= D) return;
    int b = bh / H;
    int h = bh % H;
    int p = d >> 1;                 // 半维对下标
    int row = b * S + s;            // qkv 行（[M, 3*H*D] 布局）
    float co = cos[s * (H * D / 2) + h * (D / 2) + p];
    float sn = sin[s * (H * D / 2) + h * (D / 2) + p];
    // qkv 行内三段的列基址：seg*H*D + h*D + (p*2 | p*2+1)
    int base = row * (3 * H * D) + h * D + p * 2;
    int out_row = (size_t)bh * S + s;
    // Q/K：RoPE
    for (int seg = 0; seg < 2; ++seg) {
        float a = qkv[base + seg * (H * D)];
        float bb = qkv[base + seg * (H * D) + 1];
        float rot_a = a * co - bb * sn;
        float rot_b = a * sn + bb * co;
        __half* dst = seg == 0 ? q : k;
        dst[(size_t)out_row * D + d] = __float2half((d & 1) ? rot_b : rot_a);
    }
    // V：原样拷贝（无 RoPE）
    v[(size_t)out_row * D + d] =
        __float2half(qkv[(size_t)row * (3 * H * D) + 2 * (H * D) + h * D + d]);
}

// attention 输出拼回 NHWC：attn [B*H, S, D] f16 -> out [M, H*D] f16。
extern "C" __global__ void attn_merge_kernel(const __half* __restrict__ attn,
                                             __half* __restrict__ out,
                                             int B, int H, int S, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * H * D;
    if (i >= total) return;
    int d = i % D;
    int h = (i / D) % H;
    int s = (i / (D * H)) % S;
    int b = i / (D * H * S);
    out[i] = attn[((size_t)(b * H + h) * S + s) * D + d];
}

// ---------------------------------------------------------------------------
// 头池化
// ---------------------------------------------------------------------------

// 策略头池化：in [B, S, C]（[M, C] 行主序）f16 -> out [B, 3C] f32，
// 布局 [mean(C), mean*scale(C), max(C)]（与导出图 Concat 顺序一致）。
extern "C" __global__ void pool_mean_max_kernel(const __half* __restrict__ in,
                                                float* __restrict__ out,
                                                int B, int S, int C, float scale) {
    int bc = blockIdx.x * blockDim.x + threadIdx.x;
    if (bc >= B * C) return;
    int b = bc / C;
    int c = bc % C;
    float sum = 0.0f;
    float mx = -1e30f;
    for (int s = 0; s < S; ++s) {
        float v = __half2float(in[((size_t)b * S + s) * C + c]);
        sum += v;
        mx = fmaxf(mx, v);
    }
    float mean = sum / (float)S;
    out[(size_t)b * (3 * C) + c] = mean;
    out[(size_t)b * (3 * C) + C + c] = mean * scale;
    out[(size_t)b * (3 * C) + 2 * C + c] = mx;
}

// 价值头池化：out [B, 3C] = [mean(C), mean*scale(C), mean*quad(C)]。
extern "C" __global__ void pool_mean3_kernel(const __half* __restrict__ in,
                                             float* __restrict__ out,
                                             int B, int S, int C,
                                             float scale, float quad) {
    int bc = blockIdx.x * blockDim.x + threadIdx.x;
    if (bc >= B * C) return;
    int b = bc / C;
    int c = bc % C;
    float sum = 0.0f;
    for (int s = 0; s < S; ++s) {
        sum += __half2float(in[((size_t)b * S + s) * C + c]);
    }
    float mean = sum / (float)S;
    out[(size_t)b * (3 * C) + c] = mean;
    out[(size_t)b * (3 * C) + C + c] = mean * scale;
    out[(size_t)b * (3 * C) + 2 * C + c] = mean * quad;
}

// ---------------------------------------------------------------------------
// 策略头 / 输出拼接
// ---------------------------------------------------------------------------

// 策略头 g 分支：out[i] = silu(conv1p[i] + gproj[b, c] + bias2[c])。
// 注：ONNX 里此加法之后还有 ×on_board 掩码乘法，19 路恒 1，故省略。
// conv1p [M, C] f32、gproj [B, C] f32、bias2 [C] f32、out [M, C] f32
// （小头链路保持 f32 精度）。
extern "C" __global__ void policy_g_kernel(const float* __restrict__ conv1p,
                                           const float* __restrict__ gproj,
                                           const float* __restrict__ bias2,
                                           float* __restrict__ out,
                                           int B, int S, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * S * C) return;
    int c = i % C;
    int b = i / (S * C);
    float v = conv1p[i] + gproj[(size_t)b * C + c] + bias2[c];
    v = v / (1.0f + expf(-v));
    out[i] = v;
}

// 策略输出拼接：out [B, 6, S+1] = concat(moves [M, 6], pass [B, 6], axis=2)。
// penalty 为落点屏蔽常数（logits -= penalty*(1-on_board)；19 路全合法，
// (1-on_board) 恒 0，调用方传 0.0 —— 结构保留）。
extern "C" __global__ void policy_concat_kernel(const float* __restrict__ moves,
                                                const float* __restrict__ pass,
                                                float* __restrict__ out,
                                                int B, int S, int C, float penalty) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * C * (S + 1)) return;
    int p = i % (S + 1);
    int c = (i / (S + 1)) % C;
    int b = i / ((S + 1) * C);
    if (p < S) {
        out[i] = moves[((size_t)b * S + p) * C + c] - penalty;
    } else {
        out[i] = pass[(size_t)b * C + c];
    }
}

// ---------------------------------------------------------------------------
// 小头 GEMM（精度优先）：C[m,n] = Σ_k A[m,k] * B[n,k]，A f32、B f16、C f32。
// 池化向量等小张量保持 f32，避免 f16 存储误差放大（value 对拍门 0.01）。
// 一线程一输出元素；仅用于 M×N 很小的头层 GEMM。
extern "C" __global__ void sgemm_f16b_kernel(const float* __restrict__ A,
                                             const __half* __restrict__ B,
                                             float* __restrict__ C,
                                             int M, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * N) return;
    int row = i / N;
    int col = i % N;
    float acc = 0.0f;
    const float* a = A + (size_t)row * K;
    const __half* b = B + (size_t)col * K;
    for (int k = 0; k < K; ++k) {
        acc += a[k] * __half2float(b[k]);
    }
    C[i] = acc;
}

// RMSNorm C384 warp4-vec8（fork G3 复刻）：每块 4 行、每 warp 1 行，
// 每 lane 载 12 f32（3×float4 语义，循环展开），纯 shfl 归约零 smem。
// 数值与 rms_norm_f32_kernel 同公式（sum/n + eps → rsqrtf → ×scale）。
extern "C" __global__ void rms_norm_f32_w4_kernel(
    const float* __restrict__ x, const float* __restrict__ scale,
    __half* __restrict__ y, float eps, int ncols, int rows) {
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    x += (size_t)row * ncols;
    y += (size_t)row * ncols;
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 12; ++i) {
        const float v = x[lane + i * 32];
        sum += v * v;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    const float rstd = rsqrtf(sum / (float)ncols + eps);
#pragma unroll
    for (int i = 0; i < 12; ++i) {
        const int c = lane + i * 32;
        y[c] = __float2half(x[c] * rstd * scale[c]);
    }
}
