// W8A8 inference boundaries. Reduction, scaling and residual math are FP32.
// The matrix multiply itself is INT8 x INT8 -> INT32 (cuBLASLt).
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

// Match the warp4 RMS sum order and FP16 boundary, then quantize in registers.
template<int Width> __device__ __forceinline__ void rms_quantize(
    const float* x, const float* gamma, int8_t* q, float* scales,
    float eps, int rows) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (row >= rows) return;
    x += (size_t)row * Width;
    float values[Width / 32];
    float sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < Width / 32; ++j) {
        const float v = x[lane + j * 32];
        values[j] = v;
        sum += v * v;
    }
    #pragma unroll
    for (int step = 16; step; step >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, step);
    const float rstd = rsqrtf(sum / (float)Width + eps);
    float amax = 0.0f;
    int bad = 0;
    #pragma unroll
    for (int j = 0; j < Width / 32; ++j) {
        const float v = __half2float(__float2half_rn(values[j] * rstd * gamma[lane + j * 32]));
        values[j] = v;
        amax = fmaxf(amax, fabsf(v));
        bad |= !isfinite(v);
    }
    for (int step = 16; step; step >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, step));
        bad |= __shfl_xor_sync(0xffffffff, bad, step);
    }
    const float scale = bad ? NAN : (amax == 0.0f ? 1.0f : amax / 127.0f);
    if (lane == 0) scales[row] = scale;
    #pragma unroll
    for (int j = 0; j < Width / 32; ++j) {
        const int v = bad ? 0 : max(-127, min(127, __float2int_rn(values[j] / scale)));
        q[(size_t)row * Width + lane + j * 32] = (int8_t)v;
    }
}

extern "C" __global__ void int8_rms_quantize384_kernel(
    const float* x, const float* gamma, int8_t* q, float* scales, float eps, int rows) {
    rms_quantize<384>(x, gamma, q, scales, eps, rows);
}

extern "C" __global__ void int8_rms_quantize512_kernel(
    const float* x, const float* gamma, int8_t* q, float* scales, float eps, int rows) {
    rms_quantize<512>(x, gamma, q, scales, eps, rows);
}

// Four independent token rows per CTA. The maximum reduction has no summation
// error and needs no shared memory or block-wide barriers. Padding and RNE are
// identical to the original block-per-row kernel retained below as an oracle.
extern "C" __global__ void int8_quantize_rows_warp_kernel(
    const __half* x, int8_t* q, float* scales,
    int rows, int width, int input_stride, int padded_width) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (row >= rows) return;
    float amax = 0.0f;
    int bad = 0;
    for (int col = lane; col < width; col += 32) {
        const float v = __half2float(x[(size_t)row * input_stride + col]);
        amax = fmaxf(amax, fabsf(v));
        bad |= !isfinite(v);
    }
    for (int step = 16; step; step >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, step));
        bad |= __shfl_xor_sync(0xffffffff, bad, step);
    }
    const float scale = bad ? NAN : (amax == 0.0f ? 1.0f : amax / 127.0f);
    if (lane == 0) scales[row] = scale;
    for (int col = lane; col < padded_width; col += 32) {
        int v = 0;
        if (col < width && !bad) {
            const float value = __half2float(x[(size_t)row * input_stride + col]);
            v = max(-127, min(127, __float2int_rn(value / scale)));
        }
        q[(size_t)row * padded_width + col] = (int8_t)v;
    }
}

// Register-resident SwiGLU values avoid the shared-memory round trip. Each
// warp owns one complete row, including its scale read and subsequent write.
template<int Values> __device__ __forceinline__ void swiglu_quantize_warp(
    const int32_t* dot, float* scales, const float* sw, int8_t* q,
    int rows, int hidden, int padded_hidden) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (row >= rows) return;
    const float sa = __shfl_sync(0xffffffff, lane == 0 ? scales[row] : 0.0f, 0);
    float values[Values];
    float amax = 0.0f;
    int bad = 0;
    #pragma unroll
    for (int j = 0; j < Values; ++j) {
        const int col = lane + j * 32;
        float value = 0.0f;
        if (col < hidden) {
            const size_t i = (size_t)row * (2 * hidden) + col;
            const float g = __half2float(__float2half_rn(__fmul_rn(__fmul_rn((float)dot[i], sa), sw[col])));
            const float u = __half2float(__float2half_rn(__fmul_rn(__fmul_rn((float)dot[i + hidden], sa), sw[col + hidden])));
            value = __half2float(__float2half_rn(u * (g / (1.0f + expf(-g)))));
            amax = fmaxf(amax, fabsf(value));
            bad |= !isfinite(value);
        }
        values[j] = value;
    }
    for (int step = 16; step; step >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, step));
        bad |= __shfl_xor_sync(0xffffffff, bad, step);
    }
    const float scale = bad ? NAN : (amax == 0.0f ? 1.0f : amax / 127.0f);
    if (lane == 0) scales[row] = scale;
    #pragma unroll
    for (int j = 0; j < Values; ++j) {
        const int col = lane + j * 32;
        if (col < padded_hidden) {
            int v = 0;
            if (col < hidden && !bad) v = max(-127, min(127, __float2int_rn(values[j] / scale)));
            q[(size_t)row * padded_hidden + col] = (int8_t)v;
        }
    }
}

extern "C" __global__ void int8_swiglu_quantize_warp512_kernel(
    const int32_t* dot, float* scales, const float* sw, int8_t* q,
    int rows, int hidden, int padded_hidden) {
    swiglu_quantize_warp<16>(dot, scales, sw, q, rows, hidden, padded_hidden);
}

extern "C" __global__ void int8_quantize_rows_kernel(
    const __half* x, int8_t* q, float* scales,
    int rows, int width, int input_stride, int padded_width) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float maxima[256];
    __shared__ int invalid[256];
    float amax = 0.0f;
    int bad = 0;
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        const float v = __half2float(x[(size_t)row * input_stride + col]);
        bad |= !isfinite(v);
        amax = fmaxf(amax, fabsf(v));
    }
    maxima[threadIdx.x] = amax;
    invalid[threadIdx.x] = bad;
    __syncthreads();
    for (int step = 128; step; step >>= 1) {
        if (threadIdx.x < step) {
            maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + step]);
            invalid[threadIdx.x] |= invalid[threadIdx.x + step];
        }
        __syncthreads();
    }
    // Nonfinite activations propagate to the output instead of becoming zeros.
    const float scale = invalid[0] ? NAN : (maxima[0] == 0.0f ? 1.0f : maxima[0] / 127.0f);
    if (threadIdx.x == 0) scales[row] = scale;
    for (int col = threadIdx.x; col < padded_width; col += blockDim.x) {
        int v = 0;
        if (col < width && !invalid[0]) {
            const float value = __half2float(x[(size_t)row * input_stride + col]);
            v = __float2int_rn(value / scale);
            v = max(-127, min(127, v));
        }
        q[(size_t)row * padded_width + col] = (int8_t)v;
    }
}

extern "C" __global__ void int8_dequantize_half_kernel(
    const int32_t* dot, const float* sa, const float* sw, __half* output,
    int rows, int cols) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * cols) return;
    const float value = __fmul_rn(__fmul_rn((float)dot[i], sa[i / cols]), sw[i % cols]);
    output[i] = __float2half_rn(value);
}

extern "C" __global__ void int8_dequantize_residual_kernel(
    const int32_t* dot, const float* sa, const float* sw, float* residual,
    int rows, int cols) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * cols) return;
    const float value = __fmul_rn(__fmul_rn((float)dot[i], sa[i / cols]), sw[i % cols]);
    residual[i] = __fadd_rn(residual[i], value);
}

// Preserve both existing half boundaries, but keep intermediates in shared
// memory instead of writing dual/activated half tensors to device memory.
// scales is read once per CTA, then replaced by the next projection's scale.
extern "C" __global__ void int8_swiglu_quantize_kernel(
    const int32_t* dot, float* scales, const float* sw, int8_t* q,
    int rows, int hidden, int padded_hidden) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float activated[];
    __shared__ float maxima[256];
    __shared__ int invalid[256];
    __shared__ float input_scale;
    if (threadIdx.x == 0) input_scale = scales[row];
    __syncthreads();
    float amax = 0.0f;
    int bad = 0;
    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
        const size_t i = (size_t)row * (2 * hidden) + col;
        const float g = __half2float(__float2half_rn(__fmul_rn(__fmul_rn((float)dot[i], input_scale), sw[col])));
        const float u = __half2float(__float2half_rn(__fmul_rn(__fmul_rn((float)dot[i + hidden], input_scale), sw[col + hidden])));
        const float value = __half2float(__float2half_rn(u * (g / (1.0f + expf(-g)))));
        activated[col] = value;
        amax = fmaxf(amax, fabsf(value));
        bad |= !isfinite(value);
    }
    maxima[threadIdx.x] = amax;
    invalid[threadIdx.x] = bad;
    __syncthreads();
    for (int step = 128; step; step >>= 1) {
        if (threadIdx.x < step) {
            maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + step]);
            invalid[threadIdx.x] |= invalid[threadIdx.x + step];
        }
        __syncthreads();
    }
    const float scale = invalid[0] ? NAN : (maxima[0] == 0.0f ? 1.0f : maxima[0] / 127.0f);
    if (threadIdx.x == 0) scales[row] = scale;
    for (int col = threadIdx.x; col < padded_hidden; col += blockDim.x) {
        int v = 0;
        if (col < hidden && !invalid[0]) v = max(-127, min(127, __float2int_rn(activated[col] / scale)));
        q[(size_t)row * padded_hidden + col] = (int8_t)v;
    }
}
