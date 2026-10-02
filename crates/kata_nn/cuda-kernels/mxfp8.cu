// SM120 MXFP8 primitives. No --use_fast_math: the FP32 division and multiply
// are explicitly rounded, and subnormal inputs/scales must be preserved.
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <math.h>

namespace {
constexpr unsigned kNan = 1U;
constexpr unsigned kInf = 2U;
constexpr unsigned kOutputNonfinite = 4U;
constexpr unsigned kWeightOverflow = 8U;
constexpr unsigned kUnsupportedArchitecture = 16U;

__device__ __forceinline__ void record_error(unsigned* status, unsigned flags,
                                              unsigned layer) {
  atomicOr(status, flags);
  atomicCAS(status + 1, 0xffffffffU, layer);
}

template <typename T> __device__ __forceinline__ float as_float(T value);
template <> __device__ __forceinline__ float as_float<float>(float value) { return value; }
template <> __device__ __forceinline__ float as_float<__half>(__half value) {
  return __half2float(value);
}

// A finite E4M3 byte decoded without a second low-precision conversion.
__device__ __forceinline__ float decoded_magnitude(uint8_t value) {
  const unsigned magnitude = value & 127U;
  const int exponent = static_cast<int>(magnitude >> 3);
  const int mantissa = static_cast<int>(magnitude & 7U);
  return exponent == 0 ? ldexpf(static_cast<float>(mantissa), -9)
      : ldexpf(1.0f + static_cast<float>(mantissa) * 0.125f, exponent - 7);
}

template <typename T, bool CheckDecodedWeight>
__device__ __forceinline__ void quantize_block32(
    const T* source, uint8_t* quantized, uint8_t* scales, unsigned* status,
    int k, int stride, int padded_k, int padded_blocks, unsigned layer) {
  const int blocks = padded_k / 32;
  // A flat grid avoids CUDA's 65535 grid.y limit for large physical batches.
  const int row = static_cast<int>(blockIdx.x / blocks);
  const int block = static_cast<int>(blockIdx.x % blocks);
  const int lane = threadIdx.x;
  const int col = block * 32 + lane;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
  float x = col < k ? as_float(source[static_cast<size_t>(row) * stride + col]) : 0.0f;
  const unsigned error = isnan(x) ? kNan : (isinf(x) ? kInf : 0U);
  if (error) record_error(status, error, layer);
  const bool bad_block = __any_sync(0xffffffffU, error != 0);
  if (error) x = 0.0f;
  float maximum = fabsf(x);
  for (int offset = 16; offset; offset /= 2) {
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, offset));
  }
  const bool zero = maximum == 0.0f || bad_block;
  const uint8_t scale = zero ? 127U : __nv_cvt_float_to_e8m0(
      __fdiv_rn(maximum, 448.0f), __NV_SATFINITE, cudaRoundPosInf);
  const float reciprocal = ldexpf(1.0f, 127 - static_cast<int>(scale));
  const uint8_t value = zero || col >= k ? 0U
      : __nv_cvt_float_to_fp8(__fmul_rn(x, reciprocal), __NV_SATFINITE, __NV_E4M3);
  if (CheckDecodedWeight && !isfinite(ldexpf(decoded_magnitude(value),
                                            static_cast<int>(scale) - 127))) {
    record_error(status, kWeightOverflow, layer);
  }
#else
  // build.rs also compiles SM89. Keep that target buildable, but never silently
  // provide an alternative MXFP8 implementation. Rust rejects it before load.
  const uint8_t scale = 127U;
  const uint8_t value = 0U;
  if (lane == 0) record_error(status, kUnsupportedArchitecture, layer);
#endif
  quantized[static_cast<size_t>(row) * padded_k + col] = value;
  if (lane == 0) {
    const size_t tile = static_cast<size_t>(row / 128) * (padded_blocks / 4) + block / 4;
    const size_t local = (row & 31) * 16 + ((row & 127) >> 5) * 4 + (block & 3);
    scales[tile * 512 + local] = scale;
  }
}

template <typename T>
__device__ __forceinline__ void check_output(const T* data, unsigned* status,
                                            uint64_t count, unsigned layer) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count && !isfinite(as_float(data[index]))) {
    record_error(status, kOutputNonfinite, layer);
  }
}
}  // namespace

extern "C" __global__ void mxfp8_begin_forward_kernel(unsigned* status) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    status[0] = 0;
    status[1] = 0xffffffffU;
  }
}

extern "C" __global__ void mxfp8_clear_scales_kernel(uint8_t* scales, uint64_t count) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) scales[index] = 0;
}

extern "C" __global__ void mxfp8_quantize_half_kernel(
    const __half* source, uint8_t* quantized, uint8_t* scales, unsigned* status,
    int k, int stride, int padded_k, int padded_blocks, unsigned layer) {
  quantize_block32<__half, false>(source, quantized, scales, status,
                                k, stride, padded_k, padded_blocks, layer);
}

// Experimental half-only route: clear padding without a separate launch.
// Forward packing is s=512*t + 16*(row%32) + 4*((row%128)/32) + block%4,
// t=(row/128)*quads + block/4, quads=padded_blocks/4. Its inverse below is
// bijective: the low nine bits independently encode row%32, row/32%4 and
// block%4; the remaining tile index independently encodes row/128 and block/4.
// Thus (row<rows && block<padded_k/32) identifies exactly the bytes written
// by lane 0 in quantize_block32. Clearing only the complement cannot race an
// active scale write, even across CTAs. The grid-stride partition visits every
// byte of the maximum allocation exactly once, including storage beyond this
// shape's packed rectangle after a large-K -> small-K transition. No barrier
// or atomic is needed for scale bytes; the following GEMM uses the same stream.
extern "C" __global__ void mxfp8_quantize_half_clear_fused_kernel(
    const __half* source, uint8_t* quantized, uint8_t* scales, unsigned* status,
    int k, int stride, int padded_k, int padded_blocks, unsigned layer,
    int rows, uint64_t scale_capacity) {
  const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t step = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  const unsigned quads = static_cast<unsigned>(padded_blocks / 4);
  for (uint64_t index = start; index < scale_capacity; index += step) {
    // Rust bounds allocation capacity to u32::MAX, so tile/row fit u32.
    const unsigned tile = static_cast<unsigned>(index >> 9);
    const unsigned local = static_cast<unsigned>(index & 511U);
    const unsigned row = (tile / quads) * 128U + ((local >> 2) & 3U) * 32U + (local >> 4);
    const unsigned block = (tile % quads) * 4U + (local & 3U);
    if (row >= static_cast<unsigned>(rows) || block >= static_cast<unsigned>(padded_k / 32)) {
      scales[index] = 0;
    }
  }
  // Same implementation, expressions, half boundary, FP32 rounding and sticky
  // error semantics as the separate route. Static FP32 weights never use this.
  quantize_block32<__half, false>(source, quantized, scales, status,
                                k, stride, padded_k, padded_blocks, layer);
}

extern "C" __global__ void mxfp8_quantize_float_kernel(
    const float* source, uint8_t* quantized, uint8_t* scales, unsigned* status,
    int k, int stride, int padded_k, int padded_blocks, unsigned layer) {
  quantize_block32<float, true>(source, quantized, scales, status,
                              k, stride, padded_k, padded_blocks, layer);
}

extern "C" __global__ void mxfp8_check_half_output_kernel(
    const __half* data, unsigned* status, uint64_t count, unsigned layer) {
  check_output(data, status, count, layer);
}

extern "C" __global__ void mxfp8_check_float_output_kernel(
    const float* data, unsigned* status, uint64_t count, unsigned layer) {
  check_output(data, status, count, layer);
}

// One launch for the five final raw heads, including FP16/FP32 work after the
// last MXFP8 projection. This is part of the captured full-forward contract.
extern "C" __global__ void mxfp8_check_final_outputs_kernel(
    const float* policy, const float* value, const float* misc,
    const float* moremisc, const float* ownership, unsigned* status,
    uint64_t policy_count, uint64_t value_count, uint64_t misc_count,
    uint64_t moremisc_count, uint64_t ownership_count, unsigned layer) {
  uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const float* data;
  if (index < policy_count) data = policy;
  else if ((index -= policy_count) < value_count) data = value;
  else if ((index -= value_count) < misc_count) data = misc;
  else if ((index -= misc_count) < moremisc_count) data = moremisc;
  else if ((index -= moremisc_count) < ownership_count) data = ownership;
  else return;
  if (!isfinite(data[index])) record_error(status, kOutputNonfinite, layer);
}
