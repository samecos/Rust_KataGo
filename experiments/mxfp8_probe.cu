// Standalone G3 correctness probe. Not linked into RustGo; no model, PTQ
// calibration, performance claim, or runtime fallback is provided here.
// Build instructions and interpretation: experiments/mxfp8_probe.md.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {
constexpr size_t kGuard = 256;
constexpr uint8_t kCanary = 0xcd;
constexpr size_t kWorkspace = 32 * 1024 * 1024;

size_t round_up(size_t n, size_t alignment) {
  return (n + alignment - 1) / alignment * alignment;
}

const char* lt_name(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
    default: return "OTHER";
  }
}

struct Unsupported : std::runtime_error {
  using std::runtime_error::runtime_error;
};

void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorName(status)
                             + " / " + cudaGetErrorString(status));
  }
}

void lt_check(cublasStatus_t status, const char* what) {
  if (status == CUBLAS_STATUS_SUCCESS) return;
  const auto reason = std::string(what) + ": " + lt_name(status)
                    + " (" + std::to_string(static_cast<int>(status)) + ")";
  if (status == CUBLAS_STATUS_NOT_SUPPORTED || status == CUBLAS_STATUS_ARCH_MISMATCH) {
    throw Unsupported(reason);
  }
  throw std::runtime_error(reason);
}

void require(bool condition, const std::string& reason) {
  if (!condition) throw std::runtime_error(reason);
}

uint32_t float_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

// Independent CPU decoder: E4M3FN, bias 7, finite exponent 15 except 0x7f.
float decode_fp8(uint8_t bits) {
  const int exponent = (bits >> 3) & 15;
  const int mantissa = bits & 7;
  require(exponent != 15 || mantissa != 7, "FP8 oracle encountered NaN");
  const float magnitude = exponent == 0
      ? std::ldexp(static_cast<float>(mantissa), -9)
      : std::ldexp(static_cast<float>(8 + mantissa), exponent - 10);
  return (bits & 128) ? -magnitude : magnitude;
}

float decode_scale(uint8_t bits) {
  require(bits != 255, "UE8M0 oracle encountered NaN");
  return std::ldexp(1.0f, static_cast<int>(bits) - 127);
}

// IEEE binary16 RNE conversion implemented on the CPU, independently of CUDA
// conversion intrinsics and the cuBLAS epilogue used by the operation tested.
uint16_t half_bits_rne(float value) {
  const uint32_t bits = float_bits(value);
  const uint32_t sign = (bits >> 16) & 0x8000;
  const uint32_t exponent = (bits >> 23) & 255;
  uint32_t mantissa = bits & 0x7fffff;
  if (exponent == 255) return static_cast<uint16_t>(sign | 0x7c00 | (mantissa ? 0x200 : 0));
  int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00);
  if (half_exponent <= 0) {
    if (half_exponent < -10) return static_cast<uint16_t>(sign);
    mantissa |= 0x800000;
    const unsigned shift = static_cast<unsigned>(14 - half_exponent);
    uint32_t rounded = mantissa >> shift;
    const uint32_t remainder = mantissa & ((1u << shift) - 1);
    const uint32_t midpoint = 1u << (shift - 1);
    if (remainder > midpoint || (remainder == midpoint && (rounded & 1))) ++rounded;
    return static_cast<uint16_t>(sign | rounded);
  }
  uint32_t rounded = mantissa >> 13;
  const uint32_t remainder = mantissa & 0x1fff;
  if (remainder > 0x1000 || (remainder == 0x1000 && (rounded & 1))) {
    if (++rounded == 0x400) {
      rounded = 0;
      ++half_exponent;
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(half_exponent) << 10) | rounded);
}

float decode_half(uint16_t bits) {
  const int exponent = (bits >> 10) & 31;
  const int mantissa = bits & 1023;
  if (exponent == 31) return mantissa ? std::numeric_limits<float>::quiet_NaN()
                                    : ((bits & 0x8000) ? -INFINITY : INFINITY);
  const float magnitude = exponent == 0
      ? std::ldexp(static_cast<float>(mantissa), -24)
      : std::ldexp(static_cast<float>(1024 + mantissa), exponent - 25);
  return (bits & 0x8000) ? -magnitude : magnitude;
}

uint32_t mix(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  return value ^ (value >> 16);
}

// cuBLAS 128-outer x 4-scale tile (512 bytes); each scale covers K=32.
size_t scale_offset(size_t outer, size_t block, size_t padded_blocks) {
  return ((outer / 128) * (padded_blocks / 4) + block / 4) * 512
       + (outer % 32) * 16 + ((outer % 128) / 32) * 4 + block % 4;
}

struct Matrix {
  int outer, logical_k, padded_k, blocks;
  std::vector<uint8_t> values;
  std::vector<uint8_t> logical_scales; // Oracle only: never read packed indices.
  std::vector<uint8_t> packed_scales;  // GPU only: all outside-tile slots zero.

  Matrix(int rows, int k)
      : outer(rows), logical_k(k), padded_k(static_cast<int>(round_up(k, 32))),
        blocks(padded_k / 32), values(static_cast<size_t>(rows) * padded_k, 0),
        logical_scales(static_cast<size_t>(rows) * blocks),
        packed_scales(round_up(rows, 128) * round_up(blocks, 4), 0) {}

  Matrix(int rows, int k, uint32_t seed) : Matrix(rows, k) {
    // Binary-exact palette with signs, zeros and three mantissa bits. Combined
    // with scales 1/4..2 and K<=384, every product/sum is exactly FP32 in any
    // reduction order. This isolates layout/scale application from PTQ error.
    const std::array<uint8_t, 13> palette = {
        0x00, 0x20, 0xa0, 0x28, 0xa8, 0x30, 0xb0,
        0x34, 0xb4, 0x38, 0xb8, 0x3c, 0xbc};
    std::vector<bool> used(packed_scales.size(), false);
    for (int row = 0; row < rows; ++row) {
      for (int block = 0; block < blocks; ++block) {
        const uint8_t scale = static_cast<uint8_t>(125 + mix(seed ^ (row * 131U) ^ (block * 977U)) % 4);
        logical_scales[static_cast<size_t>(row) * blocks + block] = scale;
        const size_t offset = scale_offset(row, block, round_up(blocks, 4));
        require(offset < packed_scales.size() && !used[offset], "scale packing collision/out-of-bounds");
        used[offset] = true;
        packed_scales[offset] = scale;
      }
      for (int col = 0; col < logical_k; ++col) {
        values[static_cast<size_t>(row) * padded_k + col] =
            palette[mix(seed + row * 65537U + col * 257U) % palette.size()];
      }
    }
  }

  float element(int row, int col) const {
    return decode_fp8(values[static_cast<size_t>(row) * padded_k + col])
         * decode_scale(logical_scales[static_cast<size_t>(row) * blocks + col / 32]);
  }
};

void cpu_self_test() {
  require(decode_fp8(0x38) == 1.0f && decode_fp8(0xb8) == -1.0f, "FP8 sign/bias oracle");
  require(decode_fp8(0x01) == std::ldexp(1.0f, -9), "FP8 subnormal oracle");
  require(decode_fp8(0x7e) == 448.0f, "FP8 finite maximum oracle");
  require(decode_scale(125) == 0.25f && decode_scale(128) == 2.0f, "scale oracle");
  require(half_bits_rne(1.0f) == 0x3c00 && half_bits_rne(-1.0f) == 0xbc00, "half oracle");
  require(half_bits_rne(1.0f + std::ldexp(1.0f, -11)) == 0x3c00, "half even tie");
  require(half_bits_rne(1.0f + 3 * std::ldexp(1.0f, -11)) == 0x3c02, "half odd tie");
  require(half_bits_rne(std::ldexp(1.0f, -24)) == 1, "half subnormal");
  for (uint32_t bits = 0; bits < 0x7c00; ++bits) {
    require(half_bits_rne(decode_half(static_cast<uint16_t>(bits))) == bits, "half exhaustive finite roundtrip");
  }
  Matrix test(136, 344, 17);
  require(test.padded_k == 352, "B15 tail padding");
  for (int row = 0; row < test.outer; ++row) {
    for (int col = test.logical_k; col < test.padded_k; ++col) {
      require(test.values[static_cast<size_t>(row) * test.padded_k + col] == 0, "nonzero K tail");
    }
  }
  std::cout << "[mxfp8-cpu] independent_decoders_and_half_roundtrip=PASS\n";
}

struct DeviceBuffer {
  uint8_t* allocation = nullptr;
  size_t bytes;
  cudaStream_t owner;
  DeviceBuffer(size_t count, cudaStream_t stream) : bytes(count), owner(stream) {
    require(owner != nullptr, "DeviceBuffer requires an explicit owning stream");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&allocation), bytes + 2 * kGuard), "cudaMalloc");
    const auto status = cudaMemsetAsync(allocation, kCanary, bytes + 2 * kGuard, owner);
    if (status != cudaSuccess) {
      cudaFree(allocation);
      allocation = nullptr;
      cuda_check(status, "initialize buffer guards");
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (allocation) {
      cudaStreamSynchronize(owner);
      cudaFree(allocation);
    }
  }
  uint8_t* data() const { return allocation + kGuard; }
  void upload(const void* source, size_t count) const {
    require(count == bytes, "upload length mismatch");
    // A pageable H2D cudaMemcpy on the default stream can return after staging,
    // before device completion. Our nonblocking compute stream does not wait
    // for that stream. Queue on the actual owner and complete before temporary
    // host vectors go out of scope. These helpers are outside graph capture.
    cuda_check(cudaMemcpyAsync(data(), source, bytes, cudaMemcpyHostToDevice, owner), "upload buffer");
    cuda_check(cudaStreamSynchronize(owner), "finish upload and host buffer lifetime");
  }
  std::vector<uint8_t> download() const {
    std::vector<uint8_t> host(bytes);
    cuda_check(cudaMemcpyAsync(host.data(), data(), bytes, cudaMemcpyDeviceToHost, owner), "download buffer");
    cuda_check(cudaStreamSynchronize(owner), "finish download");
    return host;
  }
  void check_guards(const char* name) const {
    std::array<uint8_t, kGuard> before{}, after{};
    cuda_check(cudaMemcpyAsync(before.data(), allocation, kGuard, cudaMemcpyDeviceToHost, owner), "read prefix guard");
    cuda_check(cudaMemcpyAsync(after.data(), data() + bytes, kGuard, cudaMemcpyDeviceToHost, owner), "read suffix guard");
    cuda_check(cudaStreamSynchronize(owner), "finish guard download");
    for (size_t i = 0; i < kGuard; ++i) {
      require(before[i] == kCanary && after[i] == kCanary, std::string(name) + " guard changed");
    }
  }
};

struct Resources {
  cudaStream_t stream = nullptr;
  cublasLtHandle_t handle = nullptr;
  ~Resources() {
    if (stream) cudaStreamSynchronize(stream);
    if (handle) cublasLtDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
  }
};

struct Descriptors {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  ~Descriptors() {
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (operation) cublasLtMatmulDescDestroy(operation);
  }
};

template <class T>
void set_attribute(cublasLtMatmulDesc_t descriptor, cublasLtMatmulDescAttributes_t key, const T& value) {
  lt_check(cublasLtMatmulDescSetAttribute(descriptor, key, &value, sizeof(value)), "set matmul attribute");
}

int32_t algo_config(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t key) {
  int32_t result = -1;
  size_t written = 0;
  lt_check(cublasLtMatmulAlgoConfigGetAttribute(&algo, key, &result, sizeof(result), &written), "read algorithm config");
  require(written == sizeof(result), "unexpected algorithm config size");
  return result;
}

struct ProbeCase {
  const char* name;
  int tokens, channels, logical_k;
  bool fp32_output;
  float beta;
};

std::vector<float> cpu_reference(const Matrix& weight, const Matrix& activation,
                                 const std::vector<float>& residual, float beta) {
  std::vector<float> result(static_cast<size_t>(activation.outer) * weight.outer);
  std::vector<float> decoded_weight(static_cast<size_t>(weight.outer) * weight.logical_k);
  std::vector<float> decoded_activation(static_cast<size_t>(activation.outer) * weight.logical_k);
  for (int row = 0; row < weight.outer; ++row) {
    for (int k = 0; k < weight.logical_k; ++k) {
      decoded_weight[static_cast<size_t>(row) * weight.logical_k + k] = weight.element(row, k);
    }
  }
  for (int row = 0; row < activation.outer; ++row) {
    for (int k = 0; k < weight.logical_k; ++k) {
      decoded_activation[static_cast<size_t>(row) * weight.logical_k + k] = activation.element(row, k);
    }
  }
  for (int token = 0; token < activation.outer; ++token) {
    for (int channel = 0; channel < weight.outer; ++channel) {
      float accumulator = 0.0f;
      double exact = 0.0;
      for (int k = 0; k < weight.logical_k; ++k) {
        const float a = decoded_weight[static_cast<size_t>(channel) * weight.logical_k + k];
        const float b = decoded_activation[static_cast<size_t>(token) * weight.logical_k + k];
        accumulator = std::fma(a, b, accumulator);
        exact += static_cast<double>(a) * b;
      }
      const size_t offset = static_cast<size_t>(token) * weight.outer + channel;
      result[offset] = std::fma(beta, residual[offset], accumulator);
      exact += static_cast<double>(beta) * residual[offset];
      require(static_cast<double>(result[offset]) == exact, "synthetic CPU reference unexpectedly rounded in FP32");
      require(std::isfinite(result[offset]), "nonfinite CPU reference");
    }
  }
  return result;
}

cublasLtMatmulAlgo_t prepare_gemm(Descriptors& desc, Resources& rt, DeviceBuffer& workspace,
                                      const char* name, int channels, int tokens, int k,
                                      bool fp32_output, const void* scales_a, const void* scales_b) {
  lt_check(cublasLtMatmulDescCreate(&desc.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F), "create operation");
  const cublasOperation_t transpose = CUBLAS_OP_T, normal = CUBLAS_OP_N;
  const cublasLtMatmulMatrixScale_t scaling = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
  const cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_HOST;
  const int8_t fast_accum = 0;
  set_attribute(desc.operation, CUBLASLT_MATMUL_DESC_TRANSA, transpose);
  set_attribute(desc.operation, CUBLASLT_MATMUL_DESC_TRANSB, normal);
  set_attribute(desc.operation, CUBLASLT_MATMUL_DESC_POINTER_MODE, pointer_mode);
  set_attribute(desc.operation, CUBLASLT_MATMUL_DESC_FAST_ACCUM, fast_accum);
  set_attribute(desc.operation, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, scaling);
  set_attribute(desc.operation, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, scaling);
  const void* scale_a = scales_a;
  const void* scale_b = scales_b;
  set_attribute(desc.operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, scale_a);
  set_attribute(desc.operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, scale_b);
  // Default column-major descriptors reinterpret row-major engine arrays.
  lt_check(cublasLtMatrixLayoutCreate(&desc.a, CUDA_R_8F_E4M3, k, channels, k), "create A layout");
  lt_check(cublasLtMatrixLayoutCreate(&desc.b, CUDA_R_8F_E4M3, k, tokens, k), "create B layout");
  const cudaDataType_t output_type = fp32_output ? CUDA_R_32F : CUDA_R_16F;
  lt_check(cublasLtMatrixLayoutCreate(&desc.c, output_type, channels, tokens, channels), "create C/D layout");
  lt_check(cublasLtMatmulPreferenceCreate(&desc.preference), "create preference");
  const size_t workspace_bytes = workspace.bytes;
  lt_check(cublasLtMatmulPreferenceSetAttribute(desc.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &workspace_bytes, sizeof(workspace_bytes)), "set workspace preference");

  std::array<cublasLtMatmulHeuristicResult_t, 32> candidates{};
  int returned = 0;
  lt_check(cublasLtMatmulAlgoGetHeuristic(rt.handle, desc.operation, desc.a, desc.b, desc.c, desc.c,
                                         desc.preference, static_cast<int>(candidates.size()), candidates.data(), &returned), "heuristic");
  std::cout << "[mxfp8-heuristic] name=" << name << " returned=" << returned << '\n';
  if (returned == 0) throw Unsupported("no heuristic for requested MXFP8 shape/output/beta");
  int selected = -1;
  cublasLtMatmulHeuristicResult_t checked{};
  uint64_t selected_flags = 0;
  for (int rank = 0; rank < returned; ++rank) {
    const auto& candidate = candidates[rank];
    cublasLtMatmulHeuristicResult_t check{};
    const auto status = cublasLtMatmulAlgoCheck(rt.handle, desc.operation, desc.a, desc.b, desc.c, desc.c,
                                              &candidate.algo, &check);
    uint64_t flags = 0;
    size_t written = 0;
    const auto cap_status = cublasLtMatmulAlgoCapGetAttribute(&candidate.algo, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
                                                            &flags, sizeof(flags), &written);
    const bool audited = cap_status == CUBLAS_STATUS_SUCCESS && written == sizeof(flags)
        && (flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_TENSOR_OP_MASK)
        && (flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F)
        && (flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8F_E4M3);
    std::cout << "[mxfp8-algocheck] name=" << name << " rank=" << rank
              << " heuristic_state=" << lt_name(candidate.state) << " status=" << lt_name(status)
              << " checked_state=" << lt_name(check.state) << " workspace=" << check.workspaceSize
              << " cap_status=" << lt_name(cap_status) << " numerical_flags=0x" << std::hex << flags
              << std::dec << " tensor_fp32_e4m3=" << audited << '\n';
    if (candidate.state == CUBLAS_STATUS_SUCCESS && status == CUBLAS_STATUS_SUCCESS
        && check.state == CUBLAS_STATUS_SUCCESS && check.workspaceSize <= workspace.bytes && audited) {
      selected = rank;
      checked = check;
      selected_flags = flags;
      break;
    }
  }
  if (selected < 0) throw Unsupported("no AlgoCheck-approved Tensor Core / FP32 / E4M3 candidate within workspace");
  const auto& algorithm = candidates[selected].algo;
  std::cout << "[mxfp8-algorithm] name=" << name << " rank=" << selected
            << " id=" << algo_config(algorithm, CUBLASLT_ALGO_CONFIG_ID)
            << " tile=" << algo_config(algorithm, CUBLASLT_ALGO_CONFIG_TILE_ID)
            << " split_k=" << algo_config(algorithm, CUBLASLT_ALGO_CONFIG_SPLITK_NUM)
            << " reduction=" << algo_config(algorithm, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)
            << " workspace=" << checked.workspaceSize << " numerical_flags=0x" << std::hex
            << selected_flags << std::dec << '\n';

  return algorithm;
}

void run_case(const ProbeCase& test, Resources& rt, DeviceBuffer& workspace) {
  Matrix weight(test.channels, test.logical_k, 0x31415926U);
  Matrix activation(test.tokens, test.logical_k, 0x27182818U);
  const int k = weight.padded_k;
  const size_t count = static_cast<size_t>(test.tokens) * test.channels;
  const size_t output_bytes = count * (test.fp32_output ? sizeof(float) : sizeof(uint16_t));
  std::cout << "[mxfp8-case] name=" << test.name << " tokens=" << test.tokens
            << " channels=" << test.channels << " logical_k=" << test.logical_k
            << " padded_k=" << k << " cublas_m=" << test.channels
            << " cublas_n=" << test.tokens << " trans=TN compute=FP32"
            << " output=" << (test.fp32_output ? "FP32" : "FP16")
            << " alpha=1 beta=" << test.beta << " c_equals_d=1"
            << " fast_accum=0"
            << " A_B=E4M3 A_B_scale=VEC32_UE8M0 scale_bytes="
            << weight.packed_scales.size() << ',' << activation.packed_scales.size() << '\n';

  std::vector<float> residual(count);
  std::vector<uint8_t> initial_output(output_bytes);
  for (size_t i = 0; i < count; ++i) {
    residual[i] = (static_cast<int>(mix(static_cast<uint32_t>(i)) % 17) - 8) * 0.125f;
    if (test.fp32_output) std::memcpy(initial_output.data() + i * 4, &residual[i], 4);
    else {
      const uint16_t bits = half_bits_rne(residual[i]);
      std::memcpy(initial_output.data() + i * 2, &bits, 2);
    }
  }
  const auto reference = cpu_reference(weight, activation, residual, test.beta);

  // Buffers outlive descriptors containing their scale pointers. Each case has
  // its own descriptor; there is no shape-only descriptor cache or mutation.
  DeviceBuffer a(weight.values.size(), rt.stream), b(activation.values.size(), rt.stream);
  DeviceBuffer sa(weight.packed_scales.size(), rt.stream), sb(activation.packed_scales.size(), rt.stream);
  DeviceBuffer output(output_bytes, rt.stream);
  a.upload(weight.values.data(), weight.values.size());
  b.upload(activation.values.data(), activation.values.size());
  sa.upload(weight.packed_scales.data(), weight.packed_scales.size());
  sb.upload(activation.packed_scales.data(), activation.packed_scales.size());
  output.upload(initial_output.data(), initial_output.size());

  Descriptors desc;
  const auto algorithm = prepare_gemm(desc, rt, workspace, test.name, test.channels,
                                      test.tokens, k, test.fp32_output, sa.data(), sb.data());

  const float alpha = 1.0f, beta = test.beta;
  // Deliberately one real execution: this is correctness, not a benchmark.
  lt_check(cublasLtMatmul(rt.handle, desc.operation, &alpha, a.data(), desc.a, b.data(), desc.b,
                         &beta, output.data(), desc.c, output.data(), desc.c, &algorithm,
                         workspace.data(), workspace.bytes, rt.stream), "MXFP8 launch");
  cuda_check(cudaStreamSynchronize(rt.stream), "MXFP8 execution synchronize");
  const auto actual = output.download();
  size_t mismatches = 0, nonfinite = 0;
  double max_abs_vs_stored = 0.0, max_abs_vs_fp32 = 0.0;
  for (size_t i = 0; i < count; ++i) {
    float got, expected_stored;
    uint32_t got_bits, expected_bits;
    if (test.fp32_output) {
      std::memcpy(&got, actual.data() + 4 * i, 4);
      got_bits = float_bits(got);
      expected_stored = reference[i];
      expected_bits = float_bits(reference[i]);
    } else {
      uint16_t half;
      std::memcpy(&half, actual.data() + 2 * i, 2);
      got_bits = half;
      expected_bits = half_bits_rne(reference[i]);
      got = decode_half(half);
      expected_stored = decode_half(static_cast<uint16_t>(expected_bits));
    }
    if (!std::isfinite(got)) ++nonfinite;
    else {
      max_abs_vs_stored = std::max(max_abs_vs_stored, std::abs(static_cast<double>(got) - expected_stored));
      max_abs_vs_fp32 = std::max(max_abs_vs_fp32, std::abs(static_cast<double>(got) - reference[i]));
    }
    if (got_bits != expected_bits) {
      if (mismatches < 4) {
        std::cout << "[mxfp8-mismatch] name=" << test.name << " token=" << i / test.channels
                  << " channel=" << i % test.channels << " expected=" << expected_stored
                  << " got=" << got << " expected_bits=0x" << std::hex << expected_bits
                  << " got_bits=0x" << got_bits << std::dec << '\n';
      }
      ++mismatches;
    }
  }
  a.check_guards("A"); b.check_guards("B"); sa.check_guards("A scales"); sb.check_guards("B scales");
  output.check_guards("C/D"); workspace.check_guards("workspace");
  require(a.download() == weight.values && b.download() == activation.values, "read-only FP8 inputs changed");
  require(sa.download() == weight.packed_scales && sb.download() == activation.packed_scales, "read-only scale inputs changed");
  std::cout << "[mxfp8-numeric] name=" << test.name << " elements=" << count
            << " nonfinite=" << nonfinite << " bit_mismatches=" << mismatches
            << " max_abs_vs_rounded_oracle=" << max_abs_vs_stored
            << " max_abs_vs_fp32_oracle=" << max_abs_vs_fp32 << " guards=PASS inputs_unchanged=PASS\n";
  require(nonfinite == 0 && mismatches == 0, "independent exact oracle mismatch (no tolerance widening)");
}

// CPU encoding oracle deliberately does not call CUDA conversion intrinsics.
// Searching all 127 positive finite codes gives RNE, including subnormal ties.
uint8_t encode_fp8_cpu(float value) {
  require(std::isfinite(value), "CPU quantizer rejects NaN/Inf");
  static const std::array<float, 127> levels = [] {
    std::array<float, 127> table{};
    for (size_t i = 0; i < table.size(); ++i) table[i] = decode_fp8(static_cast<uint8_t>(i));
    return table;
  }();
  const uint8_t sign = std::signbit(value) ? 128 : 0;
  const double magnitude = std::abs(static_cast<double>(value));
  double best_distance = std::numeric_limits<double>::infinity();
  uint8_t best = 0;
  for (int code = 0; code <= 126; ++code) {
    const double distance = std::abs(magnitude - levels[code]);
    if (distance < best_distance || (distance == best_distance && (code & 1) == 0)) {
      best = static_cast<uint8_t>(code);
      best_distance = distance;
    }
  }
  // Very large finite FP32 inputs can make all distances indistinguishable
  // even in double, so saturate explicitly before returning the nearest code.
  if (magnitude >= 448.0) best = 126;
  return sign | best;
}

uint8_t encode_scale_cpu(float value) {
  require(std::isfinite(value), "CPU scale encoder rejects NaN/Inf");
  const double magnitude = std::abs(static_cast<double>(value));
  if (magnitude <= std::ldexp(1.0, -127)) return 0;
  if (magnitude >= std::ldexp(1.0, 127)) return 254;
  int exponent = 0;
  const double mantissa = std::frexp(magnitude, &exponent);
  const int rounded_exponent = mantissa == 0.5 ? exponent - 1 : exponent;
  return static_cast<uint8_t>(rounded_exponent + 127);
}

struct QuantInput {
  int rows, k, stride;
  bool half;
  // Activation values are already rounded through the existing half boundary;
  // weight values are original FP32. Padding is intentionally NaN, never read.
  std::vector<float> values;
  QuantInput(int r, int columns, int pitch, bool h)
      : rows(r), k(columns), stride(pitch), half(h),
        values(static_cast<size_t>(r) * pitch, std::numeric_limits<float>::quiet_NaN()) {
    require(rows > 0 && k > 0 && stride >= k, "invalid quantization shape/stride");
  }
  void set(int row, int col, float value) {
    values[static_cast<size_t>(row) * stride + col] = half ? decode_half(half_bits_rne(value)) : value;
  }
  std::vector<uint8_t> bytes() const {
    std::vector<uint8_t> result(values.size() * (half ? 2 : 4));
    for (size_t i = 0; i < values.size(); ++i) {
      if (half) {
        const uint16_t bits = half_bits_rne(values[i]);
        std::memcpy(result.data() + 2 * i, &bits, 2);
      } else std::memcpy(result.data() + 4 * i, &values[i], 4);
    }
    return result;
  }
};

Matrix quantize_cpu(const QuantInput& input) {
  Matrix result(input.rows, input.k);
  for (int row = 0; row < input.rows; ++row) {
    for (int block = 0; block < result.blocks; ++block) {
      float maximum = 0.0f;
      for (int lane = 0; lane < 32 && block * 32 + lane < input.k; ++lane) {
        const float x = input.values[static_cast<size_t>(row) * input.stride + block * 32 + lane];
        require(std::isfinite(x), "CPU quantizer rejects NaN/Inf");
        maximum = std::max(maximum, std::abs(x));
      }
      // FP32 division first, then scale rounding. Avoid logarithms of maximum
      // or an FP64 division: those can disagree precisely at scale boundaries.
      const float ratio = maximum / 448.0f;
      const uint8_t scale = maximum == 0.0f ? 127 : encode_scale_cpu(ratio);
      result.logical_scales[static_cast<size_t>(row) * result.blocks + block] = scale;
      result.packed_scales[scale_offset(row, block, round_up(result.blocks, 4))] = scale;
      for (int lane = 0; lane < 32 && block * 32 + lane < input.k; ++lane) {
        const int col = block * 32 + lane;
        const float x = input.values[static_cast<size_t>(row) * input.stride + col];
        const float scaled = std::ldexp(x, 127 - static_cast<int>(scale));
        result.values[static_cast<size_t>(row) * result.padded_k + col] =
            maximum == 0.0f ? 0 : encode_fp8_cpu(scaled);
      }
    }
  }
  return result;
}

QuantInput make_quant_input(int rows, int k, int stride, bool half, uint32_t variant, bool edges) {
  QuantInput input(rows, k, stride, half);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < k; ++col) {
      const int block = col / 32, lane = col % 32;
      const uint32_t noise = mix(variant + row * 65537U + col * 257U);
      const int exponent = static_cast<int>(mix(row * 131U + block * 977U + variant) % 13) - 8;
      float x = std::ldexp((static_cast<int>(noise % 20001) - 10000) / 10000.0f, exponent);
      if (row == 0 && block == 0) x = (lane & 1) ? -0.0f : 0.0f;
      if (edges) {
        switch ((row * ((k + 31) / 32) + block) % 12) {
          case 0: x = (lane & 1) ? -0.0f : 0.0f; break;
          case 1: x = std::numeric_limits<float>::denorm_min() * (lane + 1); break;
          case 2: x = std::numeric_limits<float>::min() * (lane + 1) / 32.0f; break;
          case 3: x = std::ldexp(448.0f, -127); if (lane == 31) x = std::nextafter(x, INFINITY); break;
          case 4: x = 0.875f; if (lane == 31) x = half ? 0.87548828125f : std::nextafter(x, INFINITY); break;
          case 5: x = half ? 65504.0f : std::numeric_limits<float>::max(); break;
          case 6: x = std::ldexp(1.0f + lane / 32.0f, half ? -20 : -100); break;
          case 7: x = std::ldexp(static_cast<float>(lane + 1), -24); break;
          case 8: x = lane == 31 ? 448.0f : ((lane & 1) ? 1.0625f : 1.1875f); break;
          case 9: x = lane == 31 ? 448.0f : std::ldexp(static_cast<float>(lane % 5), -10); break;
          case 10: x = 7.25f; break;
          case 11: x = lane == 31 ? std::nextafter(112.0f, 0.0f) : 3.375f; break;
        }
        if (lane & 2) x = -x;
      }
      input.set(row, col, x);
    }
  }
  return input;
}

std::vector<float> conversion_inputs() {
  std::vector<float> input = {0.0f, -0.0f, 448.0f, -448.0f, 449.0f, -500.0f,
      std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
      std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::denorm_min()};
  for (int code = 0; code < 126; ++code) {
    const float midpoint = (decode_fp8(static_cast<uint8_t>(code)) + decode_fp8(static_cast<uint8_t>(code + 1))) * 0.5f;
    for (float x : {std::nextafter(midpoint, 0.0f), midpoint, std::nextafter(midpoint, INFINITY)}) {
      input.push_back(x);
      input.push_back(-x);
    }
  }
  for (int exponent : {-127, -126, -120, -20, -1, 0, 1, 20, 120, 127}) {
    const float x = std::ldexp(1.0f, exponent);
    input.push_back(std::nextafter(x, 0.0f));
    input.push_back(x);
    input.push_back(std::nextafter(x, INFINITY));
  }
  return input;
}

void quantizer_cpu_self_test() {
  require(encode_fp8_cpu(1.0625f) == 0x38 && encode_fp8_cpu(1.1875f) == 0x3a, "E4M3 ties-even CPU oracle");
  require(encode_fp8_cpu(500.0f) == 0x7e && encode_fp8_cpu(-500.0f) == 0xfe, "E4M3 saturation CPU oracle");
  require(encode_fp8_cpu(std::ldexp(1.0f, -10)) == 0 && encode_fp8_cpu(-0.0f) == 0x80, "E4M3 subnormal/negative-zero oracle");
  require(encode_scale_cpu(std::ldexp(1.0f, -127)) == 0 && encode_scale_cpu(std::nextafter(1.0f, INFINITY)) == 128,
          "E8M0 ceil CPU oracle");
  for (int code = 0; code <= 126; ++code) {
    require(encode_fp8_cpu(decode_fp8(static_cast<uint8_t>(code))) == code, "positive E4M3 roundtrip");
  }
  for (float bad : {INFINITY, -INFINITY, std::numeric_limits<float>::quiet_NaN()}) {
    bool rejected = false;
    try { (void)encode_fp8_cpu(bad); } catch (const std::exception&) { rejected = true; }
    require(rejected, "CPU encoder accepted nonfinite input");
  }
  for (bool half : {false, true}) {
    const auto input = make_quant_input(12, 65, 79, half, 19, true);
    const auto encoded = quantize_cpu(input);
    require(encoded.logical_scales[0] == 127, "zero block must have scale=1");
    for (int i = 0; i < 32; ++i) require(encoded.values[i] == 0, "zero block must have canonical q=0");
  }
  std::cout << "[mxfp8-cpu] independent_RNE_SATFINITE_scale_ceil=PASS\n";
}

template <typename T> __device__ float source_float(T value);
template <> __device__ float source_float<float>(float value) { return value; }
template <> __device__ float source_float<__half>(__half value) { return __half2float(value); }

// One warp per K32 block. This initial implementation favors auditable
// semantics over occupancy/throughput; it is not a tuned quantization kernel.
template <typename T>
__global__ void quantize_block32_kernel(const T* source, uint8_t* quantized,
                                       uint8_t* scales, unsigned* invalid,
                                       int k, int stride, int padded_k, int padded_blocks) {
  const int lane = threadIdx.x;
  const int row = blockIdx.y, block = blockIdx.x, col = block * 32 + lane;
  float x = col < k ? source_float(source[static_cast<size_t>(row) * stride + col]) : 0.0f;
  const unsigned error = isnan(x) ? 1U : (isinf(x) ? 2U : 0U);
  if (error) atomicOr(invalid, error);
  const bool bad_block = __any_sync(0xffffffffU, error != 0);
  if (error) x = 0.0f;
  float maximum = fabsf(x);
  for (int offset = 16; offset > 0; offset /= 2) {
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, offset));
  }
  const bool zero = maximum == 0.0f || bad_block;
  const uint8_t scale = zero ? 127 : __nv_cvt_float_to_e8m0(
      __fdiv_rn(maximum, 448.0f), __NV_SATFINITE, cudaRoundPosInf);
  const float reciprocal = ldexpf(1.0f, 127 - static_cast<int>(scale));
  quantized[static_cast<size_t>(row) * padded_k + col] = zero || col >= k ? 0
      : __nv_cvt_float_to_fp8(__fmul_rn(x, reciprocal), __NV_SATFINITE, __NV_E4M3);
  if (lane == 0) {
    // Independently written GPU address calculation; CPU oracle reads its
    // logical table, so an incorrect shared packing formula cannot mask drift.
    const size_t tile = static_cast<size_t>(row / 128) * (padded_blocks / 4) + block / 4;
    const size_t local = (row & 31) * 16 + ((row & 127) >> 5) * 4 + (block & 3);
    scales[tile * 512 + local] = scale;
  }
}

__global__ void conversion_edges_kernel(const float* source, uint8_t* q, uint8_t* scale, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    q[index] = __nv_cvt_float_to_fp8(source[index], __NV_SATFINITE, __NV_E4M3);
    scale[index] = __nv_cvt_float_to_e8m0(source[index], __NV_SATFINITE, cudaRoundPosInf);
  }
}

struct QuantBuffers {
  int rows, k, stride, padded_k, padded_blocks;
  bool half;
  DeviceBuffer source, q, scale, invalid;
  QuantBuffers(const QuantInput& input, cudaStream_t stream)
      : rows(input.rows), k(input.k), stride(input.stride),
        padded_k(static_cast<int>(round_up(k, 32))),
        padded_blocks(static_cast<int>(round_up(padded_k / 32, 4))), half(input.half),
        source(input.values.size() * (half ? 2 : 4), stream),
        q(static_cast<size_t>(rows) * padded_k, stream),
        scale(round_up(rows, 128) * padded_blocks, stream), invalid(sizeof(unsigned), stream) {}

  void upload(const QuantInput& input) {
    require(input.rows == rows && input.k == k && input.stride == stride && input.half == half,
            "cannot change quantizer shape/stride/type after allocation");
    const auto bytes = input.bytes();
    source.upload(bytes.data(), bytes.size());
  }
  void launch(cudaStream_t stream) {
    require(stream == source.owner, "quantization submitted to a different buffer owner stream");
    cuda_check(cudaMemsetAsync(invalid.data(), 0, invalid.bytes, stream), "clear quantizer error status");
    cuda_check(cudaMemsetAsync(scale.data(), 0, scale.bytes, stream), "clear packed scale padding");
    const dim3 grid(padded_k / 32, rows, 1);
    if (half) {
      quantize_block32_kernel<<<grid, 32, 0, stream>>>(reinterpret_cast<const __half*>(source.data()),
          q.data(), scale.data(), reinterpret_cast<unsigned*>(invalid.data()), k, stride, padded_k, padded_blocks);
    } else {
      quantize_block32_kernel<<<grid, 32, 0, stream>>>(reinterpret_cast<const float*>(source.data()),
          q.data(), scale.data(), reinterpret_cast<unsigned*>(invalid.data()), k, stride, padded_k, padded_blocks);
    }
    cuda_check(cudaGetLastError(), "launch block32 quantizer");
  }
  unsigned error_status() const {
    const auto bytes = invalid.download();
    unsigned result = 0;
    std::memcpy(&result, bytes.data(), sizeof(result));
    return result;
  }
  void check_guards() const {
    source.check_guards("quant source"); q.check_guards("quant values");
    scale.check_guards("quant scales"); invalid.check_guards("quant status");
  }
};

void compare_bytes(const std::vector<uint8_t>& expected, const std::vector<uint8_t>& actual,
                   const std::string& label) {
  require(expected.size() == actual.size(), label + " size mismatch");
  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != actual[i]) {
      throw std::runtime_error(label + " byte " + std::to_string(i) + " expected="
          + std::to_string(expected[i]) + " actual=" + std::to_string(actual[i]));
    }
  }
}

void check_quantization(const char* name, const QuantInput& input, const Matrix& expected,
                        const QuantBuffers& gpu) {
  require(gpu.error_status() == 0, std::string(name) + " rejected input (NaN/Inf)");
  compare_bytes(expected.values, gpu.q.download(), std::string(name) + "/FP8");
  compare_bytes(expected.packed_scales, gpu.scale.download(), std::string(name) + "/scales");
  compare_bytes(input.bytes(), gpu.source.download(), std::string(name) + "/readonly-source");
  gpu.check_guards();
  double max_error = 0.0, squared_error = 0.0;
  for (int row = 0; row < input.rows; ++row) {
    for (int col = 0; col < input.k; ++col) {
      // Double dequantization also reports finite FLT_MAX inputs that round
      // to a mathematical MX value above FP32_MAX; such edge fixtures are
      // quantizer-only, never passed to the bounded finite GEMM cases.
      const double decoded = static_cast<double>(decode_fp8(expected.values[static_cast<size_t>(row) * expected.padded_k + col]))
          * std::ldexp(1.0, static_cast<int>(expected.logical_scales[static_cast<size_t>(row) * expected.blocks + col / 32]) - 127);
      const double error = std::abs(decoded - input.values[static_cast<size_t>(row) * input.stride + col]);
      max_error = std::max(max_error, error);
      squared_error += error * error;
    }
  }
  std::cout << "[mxfp8-quantization] name=" << name << " source=" << (input.half ? "half_boundary" : "original_FP32")
            << " rows=" << input.rows << " k=" << input.k << " stride=" << input.stride
            << " padded_k=" << expected.padded_k << " fp8_and_scales=BITWISE_PASS"
            << " guards=PASS readonly_source=PASS quantization_max_abs=" << max_error
            << " quantization_rmse=" << std::sqrt(squared_error / (input.rows * input.k)) << '\n';
}

void test_gpu_conversions(Resources& rt) {
  const auto values = conversion_inputs();
  DeviceBuffer source(values.size() * sizeof(float), rt.stream), q(values.size(), rt.stream), scales(values.size(), rt.stream);
  source.upload(values.data(), source.bytes);
  conversion_edges_kernel<<<static_cast<unsigned>((values.size() + 127) / 128), 128, 0, rt.stream>>>(
      reinterpret_cast<const float*>(source.data()), q.data(), scales.data(), static_cast<int>(values.size()));
  cuda_check(cudaGetLastError(), "conversion edge kernel");
  cuda_check(cudaStreamSynchronize(rt.stream), "conversion edges synchronize");
  std::vector<uint8_t> expected_q, expected_scales;
  for (float x : values) { expected_q.push_back(encode_fp8_cpu(x)); expected_scales.push_back(encode_scale_cpu(x)); }
  compare_bytes(expected_q, q.download(), "conversion RNE/SATFINITE");
  compare_bytes(expected_scales, scales.download(), "conversion E8M0 ceil");
  source.check_guards("conversion source"); q.check_guards("conversion q"); scales.check_guards("conversion scales");
  std::cout << "[mxfp8-conversion] finite_vectors=" << values.size() << " all_midpoints_and_neighbors=BITWISE_PASS"
            << " saturation=PASS subnormal=PASS scale_boundaries=PASS\n";
}

void test_gpu_quantizers(Resources& rt) {
  for (bool half : {false, true}) {
    for (bool edges : {true, false}) {
      auto input = edges ? make_quant_input(12, 65, half ? 73 : 79, half, 19, true)
                         : make_quant_input(129, 344, half ? 351 : 357, half, 23, false);
      const auto expected = quantize_cpu(input);
      QuantBuffers gpu(input, rt.stream);
      gpu.upload(input);
      gpu.launch(rt.stream);
      cuda_check(cudaStreamSynchronize(rt.stream), "quantization synchronize");
      check_quantization(edges ? "edge_blocks" : "strided_k344", input, expected, gpu);
    }
    // K31/32/33 exercise both full and partial single blocks, and extra K32
    // output padding. NaN source row-padding must be ignored in every case.
    for (int k : {31, 32, 33}) {
      auto input = make_quant_input(2, k, k + 5, half, 31, false);
      const auto expected = quantize_cpu(input);
      QuantBuffers gpu(input, rt.stream);
      gpu.upload(input); gpu.launch(rt.stream);
      cuda_check(cudaStreamSynchronize(rt.stream), "K boundary quantization synchronize");
      check_quantization("K31_32_33", input, expected, gpu);
    }
    for (float bad : {INFINITY, -INFINITY, std::numeric_limits<float>::quiet_NaN()}) {
      auto input = make_quant_input(2, 33, 41, half, 37, false);
      input.set(1, 32, bad);
      bool cpu_rejected = false;
      try { (void)quantize_cpu(input); } catch (const std::exception&) { cpu_rejected = true; }
      require(cpu_rejected, "CPU quantizer accepted NaN/Inf");
      QuantBuffers gpu(input, rt.stream);
      gpu.upload(input); gpu.launch(rt.stream);
      cuda_check(cudaStreamSynchronize(rt.stream), "invalid input quantization synchronize");
      const unsigned error = gpu.error_status();
      require(error == (std::isnan(bad) ? 1U : 2U), "GPU quantizer failed to reject NaN/Inf");
      gpu.check_guards();
      std::cout << "[mxfp8-rejection] source=" << (half ? "half" : "FP32")
                << " input=" << (std::isnan(bad) ? "NaN" : (bad < 0 ? "-Inf" : "+Inf"))
                << " status=" << error << " result=REJECTED gemm_submitted=0\n";
    }
  }
}

struct QuantGemmReference {
  std::vector<double> exact, sum_abs;
  std::vector<float> sequential_fp32;
  double quantization_output_max_abs = 0.0;
};

QuantGemmReference quantized_reference(const QuantInput& weight_source, const Matrix& weight,
                                        const QuantInput& input_source, const Matrix& input,
                                        const std::vector<float>& residual, float beta) {
  const size_t count = static_cast<size_t>(input.outer) * weight.outer;
  QuantGemmReference reference{std::vector<double>(count), std::vector<double>(count), std::vector<float>(count)};
  std::vector<float> w(static_cast<size_t>(weight.outer) * weight.logical_k);
  std::vector<float> x(static_cast<size_t>(input.outer) * input.logical_k);
  for (int row = 0; row < weight.outer; ++row) for (int k = 0; k < weight.logical_k; ++k) {
    w[static_cast<size_t>(row) * weight.logical_k + k] = weight.element(row, k);
  }
  for (int row = 0; row < input.outer; ++row) for (int k = 0; k < input.logical_k; ++k) {
    x[static_cast<size_t>(row) * input.logical_k + k] = input.element(row, k);
  }
  for (int token = 0; token < input.outer; ++token) {
    for (int channel = 0; channel < weight.outer; ++channel) {
      const size_t index = static_cast<size_t>(token) * weight.outer + channel;
      double exact = 0.0, original = 0.0, absolute = 0.0;
      float accumulator = 0.0f;
      for (int k = 0; k < weight.logical_k; ++k) {
        const float a = w[static_cast<size_t>(channel) * weight.logical_k + k];
        const float b = x[static_cast<size_t>(token) * input.logical_k + k];
        const double product = static_cast<double>(a) * b;
        exact += product;
        absolute += std::abs(product);
        accumulator = std::fma(a, b, accumulator);
        original += static_cast<double>(weight_source.values[static_cast<size_t>(channel) * weight_source.stride + k])
                  * input_source.values[static_cast<size_t>(token) * input_source.stride + k];
      }
      reference.exact[index] = exact + static_cast<double>(beta) * residual[index];
      reference.sum_abs[index] = absolute + std::abs(static_cast<double>(beta) * residual[index]);
      reference.sequential_fp32[index] = std::fma(beta, residual[index], accumulator);
      reference.quantization_output_max_abs = std::max(reference.quantization_output_max_abs, std::abs(exact - original));
      require(std::isfinite(reference.exact[index]) && std::isfinite(reference.sequential_fp32[index]), "nonfinite quantized GEMM oracle");
    }
  }
  return reference;
}

void check_quantized_gemm(const char* name, const ProbeCase& test, const QuantGemmReference& reference,
                          const std::vector<uint8_t>& bytes) {
  const size_t count = static_cast<size_t>(test.tokens) * test.channels;
  require(bytes.size() == count * (test.fp32_output ? 4 : 2), "quantized GEMM output shape");
  // A priori FP32 error bound for arbitrary summation order. All tested
  // decoded products are normal/exactly representable. Include the much
  // smaller FP64 reference-sum uncertainty too; PTQ error is NOT in the bound.
  const double u = std::ldexp(1.0, -24);
  const double operations = test.logical_k + 2.0;
  const double gamma = operations * u / (1.0 - operations * u);
  const double double_u = std::ldexp(1.0, -53);
  const double reference_gamma = operations * double_u / (1.0 - operations * double_u);
  size_t violations = 0;
  double max_exact_error = 0.0, max_cpu32_error = 0.0, max_interval_width = 0.0;
  for (size_t i = 0; i < count; ++i) {
    float actual;
    if (test.fp32_output) std::memcpy(&actual, bytes.data() + 4 * i, 4);
    else {
      uint16_t bits;
      std::memcpy(&bits, bytes.data() + 2 * i, 2);
      actual = decode_half(bits);
    }
    const double arithmetic_bound = (gamma + reference_gamma) * reference.sum_abs[i];
    double lower = reference.exact[i] - arithmetic_bound;
    double upper = reference.exact[i] + arithmetic_bound;
    if (!test.fp32_output) {
      // Round the two interval endpoints through the same IEEE storage
      // contract, using the independent CPU encoder. Outward FP32 rounding
      // avoids accidentally shrinking the allowed interval on the host.
      const float outward_low = std::nextafter(static_cast<float>(lower), -INFINITY);
      const float outward_high = std::nextafter(static_cast<float>(upper), INFINITY);
      lower = decode_half(half_bits_rne(outward_low));
      upper = decode_half(half_bits_rne(outward_high));
    }
    require(std::isfinite(lower) && std::isfinite(upper), "test interval overflows output format");
    max_interval_width = std::max(max_interval_width, upper - lower);
    if (!std::isfinite(actual) || actual < lower || actual > upper) {
      if (violations < 4) std::cout << "[mxfp8-quant-gemm-mismatch] name=" << name << " index=" << i
          << " actual=" << actual << " allowed=[" << lower << ',' << upper << "] exact=" << reference.exact[i] << '\n';
      ++violations;
    }
    if (std::isfinite(actual)) {
      max_exact_error = std::max(max_exact_error, std::abs(actual - reference.exact[i]));
      max_cpu32_error = std::max(max_cpu32_error, std::abs(static_cast<double>(actual) - reference.sequential_fp32[i]));
    }
  }
  std::cout << "[mxfp8-quant-gemm] name=" << name << " output=" << (test.fp32_output ? "FP32" : "FP16")
            << " bound=gamma_Kplus2_sum_abs_then_storage_RNE gamma=" << gamma
            << " reference_gamma=" << reference_gamma
            << " interval_violations=" << violations << " max_abs_vs_decoded_FP64=" << max_exact_error
            << " max_abs_vs_decoded_CPU_FP32=" << max_cpu32_error << " max_allowed_interval_width=" << max_interval_width
            << " separate_quantization_output_max_abs=" << reference.quantization_output_max_abs << '\n';
  require(violations == 0, "quantized GEMM exceeds a priori arithmetic/storage bound");
}

struct QuantGemmWork {
  QuantBuffers activation;
  DeviceBuffer output, workspace;
  Descriptors desc;
  cublasLtMatmulAlgo_t algorithm{};
  QuantGemmWork(const QuantInput& input, const QuantBuffers& weight, const ProbeCase& test, Resources& rt)
      : activation(input, rt.stream), output(static_cast<size_t>(test.tokens) * test.channels * (test.fp32_output ? 4 : 2), rt.stream),
        workspace(kWorkspace, rt.stream) {
    algorithm = prepare_gemm(desc, rt, workspace, test.name, test.channels, test.tokens,
                            activation.padded_k, test.fp32_output, weight.scale.data(), activation.scale.data());
  }
  void submit(const QuantBuffers& weight, const DeviceBuffer& initial_output,
              const ProbeCase& test, Resources& rt) {
    activation.launch(rt.stream);
    // Reset C every invocation, including graph replay with beta=1. A captured
    // memcpy references stable DEVICE storage, never pageable host memory.
    cuda_check(cudaMemcpyAsync(output.data(), initial_output.data(), output.bytes,
                                cudaMemcpyDeviceToDevice, rt.stream), "restore GEMM residual");
    const float alpha = 1.0f, beta = test.beta;
    lt_check(cublasLtMatmul(rt.handle, desc.operation, &alpha, weight.q.data(), desc.a,
                           activation.q.data(), desc.b, &beta, output.data(), desc.c,
                           output.data(), desc.c, &algorithm, workspace.data(), workspace.bytes, rt.stream), "quantized MXFP8 launch");
  }
  void check_scale_pointers(const QuantBuffers& weight) const {
    for (const auto pair : {
        std::make_pair(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, static_cast<const void*>(weight.scale.data())),
        std::make_pair(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, static_cast<const void*>(activation.scale.data()))}) {
      void* pointer = nullptr;
      size_t written = 0;
      lt_check(cublasLtMatmulDescGetAttribute(desc.operation, pair.first, &pointer, sizeof(pointer), &written), "read scale descriptor pointer");
      require(written == sizeof(pointer) && pointer == pair.second, "descriptor scale address changed");
    }
  }
};

struct CapturedGraph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  ~CapturedGraph() {
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
  }
};

void test_quantized_graph(const ProbeCase& test, Resources& rt) {
  auto weight_input = make_quant_input(test.channels, test.logical_k, test.logical_k + 13, false, 41, false);
  const auto weight_cpu = quantize_cpu(weight_input);
  QuantBuffers weight(weight_input, rt.stream);
  weight.upload(weight_input); weight.launch(rt.stream);
  cuda_check(cudaStreamSynchronize(rt.stream), "static weight quantization synchronize");
  check_quantization("static_original_FP32_weight", weight_input, weight_cpu, weight);
  const std::array<QuantInput, 2> inputs = {
      make_quant_input(test.tokens, test.logical_k, test.logical_k + 7, true, 43, false),
      make_quant_input(test.tokens, test.logical_k, test.logical_k + 7, true, 47, false)};
  const std::array<Matrix, 2> quantized = {quantize_cpu(inputs[0]), quantize_cpu(inputs[1])};
  require(quantized[0].values != quantized[1].values && quantized[0].packed_scales != quantized[1].packed_scales,
          "graph inputs must change BOTH FP8 values and block scales");
  const size_t count = static_cast<size_t>(test.tokens) * test.channels;
  std::vector<float> residual(count);
  std::vector<uint8_t> initial(count * (test.fp32_output ? 4 : 2));
  for (size_t i = 0; i < count; ++i) {
    residual[i] = (static_cast<int>(i % 17) - 8) * 0.125f;
    if (test.fp32_output) std::memcpy(initial.data() + 4 * i, &residual[i], 4);
    else { const auto bits = half_bits_rne(residual[i]); std::memcpy(initial.data() + 2 * i, &bits, 2); }
  }
  DeviceBuffer initial_gpu(initial.size(), rt.stream);
  initial_gpu.upload(initial.data(), initial.size());
  const std::array<QuantGemmReference, 2> references = {
      quantized_reference(weight_input, weight_cpu, inputs[0], quantized[0], residual, test.beta),
      quantized_reference(weight_input, weight_cpu, inputs[1], quantized[1], residual, test.beta)};
  // Complete separation of activation/source/output/scale/status/workspace
  // storage prevents a direct call from repairing stale captured state.
  QuantGemmWork graph_work(inputs[0], weight, test, rt), direct_work(inputs[0], weight, test, rt);
  graph_work.check_scale_pointers(weight); direct_work.check_scale_pointers(weight);
  for (auto* work : {&graph_work, &direct_work}) {
    work->activation.upload(inputs[0]); work->activation.launch(rt.stream);
    cuda_check(cudaStreamSynchronize(rt.stream), "pre-GEMM quantization validation synchronize");
    check_quantization("before_first_GEMM", inputs[0], quantized[0], work->activation);
    work->submit(weight, initial_gpu, test, rt);
    cuda_check(cudaStreamSynchronize(rt.stream), "pre-capture pipeline warm");
  }
  CapturedGraph graph;
  cuda_check(cudaStreamBeginCapture(rt.stream, cudaStreamCaptureModeThreadLocal), "begin quantization graph capture");
  try {
    graph_work.submit(weight, initial_gpu, test, rt);
  } catch (...) {
    cudaGraph_t discarded = nullptr;
    cudaStreamEndCapture(rt.stream, &discarded);
    if (discarded) cudaGraphDestroy(discarded);
    throw;
  }
  cuda_check(cudaStreamEndCapture(rt.stream, &graph.graph), "end quantization graph capture");
  require(graph.graph != nullptr, "empty quantization graph");
  cuda_check(cudaGraphInstantiate(&graph.executable, graph.graph, 0), "instantiate quantization graph");
  cuda_check(cudaGraphUpload(graph.executable, rt.stream), "upload quantization graph");
  cuda_check(cudaStreamSynchronize(rt.stream), "graph upload synchronize");
  std::vector<uint8_t> first_output;
  int round = 0;
  for (int variant : {0, 1, 0}) {
    direct_work.activation.upload(inputs[variant]);
    direct_work.activation.launch(rt.stream);
    cuda_check(cudaStreamSynchronize(rt.stream), "validate dynamic quantization before GEMM");
    check_quantization("dynamic_before_GEMM", inputs[variant], quantized[variant], direct_work.activation);
    direct_work.submit(weight, initial_gpu, test, rt);
    cuda_check(cudaStreamSynchronize(rt.stream), "fresh direct quantization/GEMM synchronize");
    require(direct_work.activation.error_status() == 0, "direct pipeline rejected nonfinite input; discard output");
    const auto direct = direct_work.output.download();
    check_quantized_gemm(test.name, test, references[variant], direct);
    graph_work.activation.upload(inputs[variant]);
    cuda_check(cudaGraphLaunch(graph.executable, rt.stream), "launch quantization graph replay");
    cuda_check(cudaStreamSynchronize(rt.stream), "quantization graph replay synchronize");
    // Captured pipelines report an error bit, checked before exposing output.
    // A future production backend must enforce the same discard-on-error rule.
    check_quantization("graph_dynamic_quantization", inputs[variant], quantized[variant], graph_work.activation);
    const auto replay = graph_work.output.download();
    compare_bytes(direct, replay, "graph/direct GEMM output bits");
    if (round == 0) first_output = replay;
    else if (round == 1) require(first_output != replay, "graph B produced stale A output");
    else compare_bytes(first_output, replay, "graph A return output bits");
    for (auto* work : {&graph_work, &direct_work}) {
      work->output.check_guards("pipeline output"); work->workspace.check_guards("pipeline workspace");
      work->activation.check_guards(); work->check_scale_pointers(weight);
    }
    std::cout << "[mxfp8-graph] name=" << test.name << " round=" << round++ << " input=" << variant
              << " dynamic_quantization_and_GEMM=BITWISE_DIRECT_PASS scale_addresses=UNCHANGED\n";
  }
  // Captured work cannot perform a host check between quantization and GEMM.
  // Explicitly prove that invalid dynamic input is flagged and discarded,
  // then that the next valid replay clears the flag and reproduces A exactly.
  auto invalid_input = inputs[0];
  invalid_input.set(0, 1, std::numeric_limits<float>::quiet_NaN());
  graph_work.activation.upload(invalid_input);
  cuda_check(cudaGraphLaunch(graph.executable, rt.stream), "launch graph invalid-input rejection");
  cuda_check(cudaStreamSynchronize(rt.stream), "graph invalid-input synchronize");
  require(graph_work.activation.error_status() == 1, "captured quantizer failed to flag NaN");
  std::cout << "[mxfp8-graph-rejection] name=" << test.name
            << " input=NaN status=1 output_discarded=1 gemm_was_captured=1\n";
  graph_work.activation.upload(inputs[0]);
  cuda_check(cudaGraphLaunch(graph.executable, rt.stream), "replay valid A after rejection");
  cuda_check(cudaStreamSynchronize(rt.stream), "graph recovery synchronize");
  check_quantization("graph_after_rejection", inputs[0], quantized[0], graph_work.activation);
  compare_bytes(first_output, graph_work.output.download(), "graph A after rejected input");
  graph_work.output.check_guards("graph output after rejection");
  graph_work.workspace.check_guards("graph workspace after rejection");
  graph_work.check_scale_pointers(weight);
  check_quantization("static_weight_after_graph", weight_input, weight_cpu, weight);
  compare_bytes(initial, initial_gpu.download(), "readonly residual source");
  initial_gpu.check_guards("residual source");
  std::cout << "[mxfp8-graph-result] name=" << test.name << " result=PASS sequence=A_B_A"
            << " graph_buffers_independent=1 scales_regenerated_each_replay=1\n";
}

void print_versions(int device, const cudaDeviceProp& prop) {
  int driver = 0, runtime = 0;
  cuda_check(cudaDriverGetVersion(&driver), "driver version");
  cuda_check(cudaRuntimeGetVersion(&runtime), "runtime version");
  std::cout << "[mxfp8-runtime] device=" << device << " name=\"" << prop.name
            << "\" sm=" << prop.major << '.' << prop.minor << " sm_count=" << prop.multiProcessorCount
            << " driver_api=" << driver << " runtime=" << runtime << " compiled_cudart=" << CUDART_VERSION
            << " cublaslt_runtime=" << cublasLtGetVersion() << " cublas_headers="
            << CUBLAS_VER_MAJOR << '.' << CUBLAS_VER_MINOR << '.' << CUBLAS_VER_PATCH << '.' << CUBLAS_VER_BUILD
            << " nvcc=" << __CUDACC_VER_MAJOR__ << '.' << __CUDACC_VER_MINOR__ << '.' << __CUDACC_VER_BUILD__ << '\n';
#ifdef _WIN32
  for (const char* library : {"cublasLt64_13.dll", "cublas64_13.dll"}) {
    char path[32768] = {};
    const auto module = GetModuleHandleA(library);
    if (module && GetModuleFileNameA(module, path, static_cast<DWORD>(sizeof(path)))) {
      std::cout << "[mxfp8-library] name=" << library << " loaded_path=\"" << path << "\"\n";
    }
  }
#endif
}
} // namespace

int main(int argc, char** argv) {
  try {
    std::cout << std::setprecision(12);
    bool cpu_only = false;
    int device = 0;
    for (int i = 1; i < argc; ++i) {
      const std::string arg(argv[i]);
      if (arg == "--cpu-only") cpu_only = true;
      else if (arg == "--device" && i + 1 < argc) device = std::stoi(argv[++i]);
      else throw std::runtime_error("usage: mxfp8_probe [--cpu-only] [--device N]");
    }
    cpu_self_test();
    quantizer_cpu_self_test();
    if (cpu_only) {
      std::cout << "[mxfp8-summary] CPU_ONLY_NO_GPU_VALIDATION\n";
      return 0;
    }
    cuda_check(cudaSetDevice(device), "select CUDA device");
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, device), "device properties");
    print_versions(device, prop);
    if (prop.major != 12 || prop.minor != 0) throw Unsupported("this prototype requires SM120; other architectures are not validated");
    Resources rt;
    cuda_check(cudaStreamCreateWithFlags(&rt.stream, cudaStreamNonBlocking), "create stream");
    std::cout << "[mxfp8-stream] compute=nonblocking buffer_init_and_IO=explicit_owner_stream"
              << " host_upload_download_completion=stream_synchronized\n";
    lt_check(cublasLtCreate(&rt.handle), "create cuBLASLt handle");
    DeviceBuffer workspace(kWorkspace, rt.stream);
    const std::array<ProbeCase, 6> cases = {{
        {"fp16_cross_tiles_361", 361, 136, 160, false, 0.0f},
        {"fp16_b11_shape", 361, 384, 384, false, 0.0f},
        {"fp16_b15_k344_pad352", 17, 384, 344, false, 0.0f},
        {"fp16_small_n24_k72_pad96", 129, 24, 72, false, 0.0f},
        {"fp32_cross_tiles_361", 361, 136, 160, true, 0.0f},
        {"fp32_residual_cross_tiles_361", 361, 136, 160, true, 1.0f},
    }};
    int passed = 0, unsupported = 0, failed = 0;
    for (const auto& test : cases) {
      try {
        run_case(test, rt, workspace);
        ++passed;
        std::cout << "[mxfp8-result] name=" << test.name << " result=PASS\n";
      } catch (const Unsupported& error) {
        ++unsupported;
        std::cout << "[mxfp8-result] name=" << test.name << " result=UNSUPPORTED reason=\"" << error.what() << "\"\n";
      } catch (const std::exception& error) {
        ++failed;
        std::cout << "[mxfp8-result] name=" << test.name << " result=FAIL reason=\"" << error.what() << "\"\n";
      }
    }
    const auto run_extension = [&](const char* name, auto&& body) {
      try {
        body();
        ++passed;
        std::cout << "[mxfp8-result] name=" << name << " result=PASS\n";
      } catch (const Unsupported& error) {
        ++unsupported;
        std::cout << "[mxfp8-result] name=" << name << " result=UNSUPPORTED reason=\"" << error.what() << "\"\n";
      } catch (const std::exception& error) {
        ++failed;
        std::cout << "[mxfp8-result] name=" << name << " result=FAIL reason=\"" << error.what() << "\"\n";
      }
    };
    run_extension("GPU_conversion_boundaries", [&] { test_gpu_conversions(rt); });
    run_extension("GPU_block32_quantizers", [&] { test_gpu_quantizers(rt); });
    const std::array<ProbeCase, 2> graph_cases = {{
        {"quantized_graph_fp16_k344", 361, 136, 344, false, 0.0f},
        {"quantized_graph_fp32_residual_k168", 129, 24, 168, true, 1.0f},
    }};
    for (const auto& test : graph_cases) {
      run_extension(test.name, [&] { test_quantized_graph(test, rt); });
    }
    std::cout << "[mxfp8-summary] passed=" << passed << " unsupported=" << unsupported << " failed=" << failed
              << " scope=synthetic_quantizer_GEMM_Graph_NOT_MODEL_PTQ_OR_PERFORMANCE\n";
    return failed ? 1 : (unsupported ? 2 : 0);
  } catch (const Unsupported& error) {
    std::cerr << "[mxfp8-fatal] result=UNSUPPORTED reason=\"" << error.what() << "\"\n";
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "[mxfp8-fatal] result=FAIL reason=\"" << error.what() << "\"\n";
    return 1;
  }
}
