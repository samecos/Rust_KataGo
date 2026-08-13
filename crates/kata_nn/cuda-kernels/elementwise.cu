// Rust_KataGo 手写 CUDA 逐元素与归一化 kernel。
//
// 数值纪律（对齐 KataGomo_fork）：FP16 输入先转 FP32 计算、激活用精确形式
// （SiLU = x/(1+exp(-x))）、结果再舍入回 FP16；不使用近似指令。

#include <cuda_fp16.h>
#include <cstdint>

// SiLU：out = x / (1 + exp(-x))（FP32 计算，FP16 存储）。
extern "C" __global__ void silu_f16_kernel(const __half* __restrict__ in,
                                           __half* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(in[i]);
    v = v / (1.0f + expf(-v));
    out[i] = __float2half(v);
}

// affine + SiLU：out = silu(in * scale + bias)。
extern "C" __global__ void affine_silu_f16_kernel(const __half* __restrict__ in,
                                                  const __half* __restrict__ scale,
                                                  const __half* __restrict__ bias,
                                                  __half* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(in[i]) * __half2float(scale[i]) + __half2float(bias[i]);
    v = v / (1.0f + expf(-v));
    out[i] = __float2half(v);
}

// 原位残差：res += in（FP32 计算，FP16 存储；beta=1 残差 GEMM 的逐元素版本）。
extern "C" __global__ void add_residual_f16_kernel(const __half* __restrict__ in,
                                                   __half* __restrict__ res, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(in[i]) + __half2float(res[i]);
    res[i] = __float2half(v);
}

// RMSNorm：每行 ncols 个元素，out = in / sqrt(mean(x^2) + eps) * scale。
// 每 block 处理一行：128 线程，FP32 累加（先线程内串行、再跨线程归约）。
extern "C" __global__ void rms_norm_f16_kernel(const __half* __restrict__ in,
                                               const __half* __restrict__ scale,
                                               __half* __restrict__ out,
                                               float eps, int ncols) {
    int row = blockIdx.x;
    const __half* x = in + (size_t)row * ncols;
    __half* y = out + (size_t)row * ncols;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) {
        float v = __half2float(x[i]);
        sum += v * v;
    }
    // warp 内归约
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }
    // warp 间经共享内存
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
        float v = __half2float(x[i]) * rstd * __half2float(scale[i]);
        y[i] = __float2half(v);
    }
}

// SwiGLU：out = x_up * sigmoid(x_gate) * 0.5? 否——本模型 FFN 为
// silu(gate)*up（SiLU 门控）。此处提供逐元素 SwiGLU：
// out[i] = up[i] * silu(gate[i])。
extern "C" __global__ void swiglu_f16_kernel(const __half* __restrict__ up,
                                             const __half* __restrict__ gate,
                                             __half* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(gate[i]);
    g = g / (1.0f + expf(-g));
    out[i] = __float2half(__half2float(up[i]) * g);
}

// f16 -> f32（head 输出用）。
extern "C" __global__ void half_to_f32_region_kernel(const __half* __restrict__ in,
                                                     float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __half2float(in[i]);
}
