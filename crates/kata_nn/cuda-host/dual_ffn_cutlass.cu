// ---------------------------------------------------------------------------
// B2:dual FFN + SwiGLU epilogue(CUTLASS DualGemm,Sm80 mma.sync 路径)。
//
// 形状:b11fix FFN 上投影 —— A[M,384](RmsNorm 输出 f16)共享,B0/B1 各
// [1152,384](packed `dual` 权重的 gate/up 两半,零拷贝切片),D[M,1152] f16
// 直接是 SwiGLU 结果,省掉 2304 宽中间缓冲的整趟往返 + 独立 swiglu kernel。
//
// 瓦片族 = fork dual_ffn(t128 n64 k32 s3,warp 64x32x32,instr 16x8x16);
// 门槛实测(cuda-optimization-plan.md M1/M2 节):同形状主循环 0.1116ms
// 优于 cuBLASLt 0.1303(heuristic)/ 0.117(C1 计时)。
//
// 数值纪律(与现行 hgemm_f16 + swiglu_dual 两 kernel 路径逐位一致):
//   1. 两份 FP32 累加器先各自 __float2half_rn(复刻 f16 GEMM 输出的存储
//      舍入边界),再转回 FP32;
//   2. SiLU 用精确形式 g/(1+expf(-g))(非 __expf/fast_tanh);
//   3. 乘积 __float2half_rn 写出。
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <memory>
#include <new>
#include <unordered_map>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "device/dual_gemm.h"

namespace {

constexpr int kChannels = 384;    // K(瓶颈宽)
constexpr int kFfn = 1152;        // 每个投影的 N(gate/up 各一)

using Element = cutlass::half_t;
using Layout = cutlass::layout::RowMajor;
// 本仓库权重是 out-first [N,K] 行主序(K 连续)= CUTLASS 约定的
// B[K,N] ColumnMajor(K 连续),即 tensorop 友好的 TN 布局;fork 源码的
// RowMajor+ld=N 对应其 [K,N] 权重存储,不能照搬。
using LayoutB = cutlass::layout::ColumnMajor;

// 精确数值版 SwiGLU 合并 op(接口与 LeftSiLUAndMul 相同)。
// lhs = gate 投影 acc,silu(lhs)*rhs(up 投影)。
template <
  typename ElementOutput_,
  int Count,
  typename ElementAccumulator_ = ElementOutput_,
  typename ElementCompute_ = ElementOutput_,
  cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest>
class ExactRoundSiLUMul {
public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static int const kCount = Count;
  using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;

  struct Params {};

  CUTLASS_HOST_DEVICE
  ExactRoundSiLUMul(Params const&) {}

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return true; }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      FragmentAccumulator const& lhs, FragmentAccumulator const& rhs) const {
    FragmentOutput out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      // 先复刻现行路径的中间 half 舍入边界
      __half g16 = __float2half_rn(float(lhs[i]));
      __half u16 = __float2half_rn(float(rhs[i]));
      float g = __half2float(g16);
      float u = __half2float(u16);
      float s = g / (1.0f + expf(-g));  // 精确 SiLU
      __half r = __float2half_rn(u * s);
      out[i] = *reinterpret_cast<ElementOutput*>(&r);
    }
    return out;
  }
};

using ProjectionOutput = cutlass::epilogue::thread::LinearCombination<
  Element, 8, float, float, cutlass::epilogue::thread::ScaleType::Nothing>;
using SwiGLU = ExactRoundSiLUMul<Element, 8, Element, float>;
using DualGemm = cutlass::gemm::device::DualGemm<
  Element, Layout, Element, LayoutB,
  LayoutB, Element, Layout, float,
  cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128, 64, 32>,
  cutlass::gemm::GemmShape<64, 32, 32>,
  cutlass::gemm::GemmShape<16, 8, 16>,
  ProjectionOutput, ProjectionOutput, SwiGLU,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
  3, false, false, false, 8, 8>;

DualGemm::Arguments makeArguments(
  const void* input, const void* gateWeights, const void* upWeights,
  void* output, int tokens
) {
  DualGemm::TensorRefC nullC;
  DualGemm::TensorRefD nullD;
  return {
    cutlass::gemm::DualGemmMode::kGemm,
    {tokens, kFfn, kChannels},
    {reinterpret_cast<const Element*>(input), Layout(kChannels)},
    // B0 = gate(silu 作用于 D0),B1 = up;ColumnMajor ld=K(K 连续)
    {reinterpret_cast<const Element*>(gateWeights), LayoutB(kChannels)},
    nullC, nullD,
    {reinterpret_cast<const Element*>(upWeights), LayoutB(kChannels)},
    nullC, nullD,
    {reinterpret_cast<Element*>(output), Layout(kFfn)},
    // ScaleType::Nothing 的 Params 仅含 alpha(单参构造),beta 不参与。
    {1.0f}, {1.0f}, {}, 1
  };
}

struct State {
  DualGemm op;
  bool initialized = false;
};

struct Handle {
  std::unordered_map<int, std::unique_ptr<State>> byTokens;
};

int statusCode(cutlass::Status status) {
  return 100 + static_cast<int>(status);
}

}  // namespace

extern "C" void* katago_dual_ffn_create() {
  return new (std::nothrow) Handle();
}

extern "C" void katago_dual_ffn_destroy(void* opaque) {
  delete static_cast<Handle*>(opaque);
}

extern "C" int katago_dual_ffn_exec(
  void* opaque,
  const void* input,
  const void* gateWeights,
  const void* upWeights,
  void* output,
  int tokens,
  uintptr_t stream
) {
  if (opaque == nullptr || input == nullptr || gateWeights == nullptr ||
      upWeights == nullptr || output == nullptr || tokens <= 0)
    return 1;
  Handle* handle = static_cast<Handle*>(opaque);
  auto& slot = handle->byTokens[tokens];
  if (slot == nullptr)
    slot = std::make_unique<State>();
  DualGemm::Arguments args =
      makeArguments(input, gateWeights, upWeights, output, tokens);
  cutlass::Status status;
  if (!slot->initialized) {
    status = slot->op.can_implement(args);
    if (status != cutlass::Status::kSuccess)
      return statusCode(status);
    if (DualGemm::get_workspace_size(args) != 0)
      return 50;
    status = slot->op.initialize(args, nullptr, reinterpret_cast<cudaStream_t>(stream));
    if (status != cutlass::Status::kSuccess)
      return statusCode(status);
    slot->initialized = true;
  } else {
    status = slot->op.update(args, nullptr);
    if (status != cutlass::Status::kSuccess)
      return statusCode(status);
  }
  status = slot->op.run(reinterpret_cast<cudaStream_t>(stream));
  if (status != cutlass::Status::kSuccess)
    return statusCode(status);
  cudaError_t cudaStatus = cudaPeekAtLastError();
  return cudaStatus == cudaSuccess ? 0 : 200 + static_cast<int>(cudaStatus);
}

// Capability probe for the exact production kernel. This deliberately runs a
// real M=16,N=1152,K=384 DualGemm rather than a trivial CUDA kernel, so it
// verifies CUTLASS initialization, dynamic shared-memory opt-in, launch, and
// the exact SwiGLU epilogue. Zero inputs/weights must overwrite a nonzero
// sentinel output with exactly zero.
extern "C" int katago_dual_ffn_probe() {
  int device = 0;
  int major = 0;
  if (cudaGetDevice(&device) != cudaSuccess)
    return 301;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess)
    return 302;
  if (major < 8)
    return 303;

  constexpr int M = 16;
  constexpr size_t inputElts = static_cast<size_t>(M) * kChannels;
  constexpr size_t weightElts = static_cast<size_t>(kFfn) * kChannels;
  constexpr size_t outputElts = static_cast<size_t>(M) * kFfn;
  constexpr size_t totalElts = inputElts + 2 * weightElts + outputElts;

  half* buffer = nullptr;
  if (cudaMalloc(&buffer, totalElts * sizeof(half)) != cudaSuccess)
    return 304;
  half* input = buffer;
  half* gate = input + inputElts;
  half* up = gate + weightElts;
  half* output = up + weightElts;

  int result = 0;
  if (cudaMemset(buffer, 0, (inputElts + 2 * weightElts) * sizeof(half)) != cudaSuccess)
    result = 305;
  if (result == 0 && cudaMemset(output, 0xFF, outputElts * sizeof(half)) != cudaSuccess)
    result = 306;

  if (result == 0) {
    void* handle = katago_dual_ffn_create();
    if (handle == nullptr)
      result = 307;
    else {
      const int execResult = katago_dual_ffn_exec(
          handle, input, gate, up, output, M, reinterpret_cast<uintptr_t>(nullptr));
      katago_dual_ffn_destroy(handle);
      if (execResult != 0)
        result = 400 + execResult;
    }
  }
  if (result == 0 && cudaDeviceSynchronize() != cudaSuccess)
    result = 308;

  std::vector<half> hostOutput;
  if (result == 0) {
    hostOutput.resize(outputElts);
    if (cudaMemcpy(hostOutput.data(), output, outputElts * sizeof(half), cudaMemcpyDeviceToHost) != cudaSuccess)
      result = 309;
  }
  if (result == 0) {
    for (half value : hostOutput) {
      if (__half2float(value) != 0.0f) {
        result = 310;
        break;
      }
    }
  }

  (void)cudaFree(buffer);
  (void)cudaGetLastError();
  return result;
}
