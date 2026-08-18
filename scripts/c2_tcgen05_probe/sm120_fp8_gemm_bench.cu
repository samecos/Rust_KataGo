// C2 证伪后的工具链验证:CUTLASS 4.7 SM120 dense FP8(f8f6f4 mma.sync + TMA)
// 在 sm_120f 下编译并运行,与 cublas FP16 同形状对照计时。
// 形状取生产 GEMM:ffn_up M=5776 N=2304 K=384 / qkv N=1152 / ffn_down N=384 K=2304。
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

using namespace cute;

using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = float;
using ElementD = float;
using ElementAccumulator = float;
using ElementCompute = float;
using LayoutA = cutlass::layout::RowMajor;      // [M,K] 激活
using LayoutB = cutlass::layout::ColumnMajor;   // [K,N] 列主 = 权重 [N,K] out-first
using LayoutC = cutlass::layout::RowMajor;      // [M,N] 生产布局
using LayoutD = cutlass::layout::RowMajor;

static constexpr int Alignment = 16;
static constexpr int AlignmentCD = 128 / cutlass::sizeof_bits<ElementC>::value;

using TileShape = Shape<_128, _64, _64>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentCD,
    ElementD, LayoutD, AlignmentCD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, Alignment,
    ElementB, LayoutB, Alignment,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

#define CHECK_CUDA(x)                                                        \
  do {                                                                       \
    cudaError_t e_ = (x);                                                    \
    if (e_ != cudaSuccess) {                                                 \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

static float bench_fp8(const ElementA* dA, const ElementB* dB, float* dD,
                       int M, int N, int K, float alpha = 1.0f, float beta = 0.0f,
                       float* out_first_ms = nullptr) {
  Gemm gemm;
  StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  StrideB sB = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {dA, sA, dB, sB},
      {{alpha, beta}, nullptr, sC, dD, sD}};
  size_t ws_size = Gemm::get_workspace_size(args);
  void* ws = nullptr;
  if (ws_size) CHECK_CUDA(cudaMalloc(&ws, ws_size));
  cutlass::Status st = gemm.initialize(args, ws);
  if (st != cutlass::Status::kSuccess) {
    printf("CUTLASS initialize failed: %d\n", int(st));
    exit(1);
  }
  st = gemm.run();
  if (st != cutlass::Status::kSuccess) {
    printf("CUTLASS run failed: %d\n", int(st));
    exit(1);
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  if (out_first_ms) { *out_first_ms = 0.f; }

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  for (int i = 0; i < 5; ++i) gemm.run();
  CHECK_CUDA(cudaDeviceSynchronize());
  cudaEventRecord(t0);
  for (int i = 0; i < 50; ++i) gemm.run();
  cudaEventRecord(t1);
  CHECK_CUDA(cudaEventSynchronize(t1));
  float ms = 0;
  cudaEventElapsedTime(&ms, t0, t1);
  ms /= 50.f;
  if (ws) cudaFree(ws);
  return ms;
}

static float bench_cublas_f16(const __half* dA, const __half* dB, float* dD,
                              cublasHandle_t h, int M, int N, int K) {
  // A [M,K] row, B [N,K] row(= [K,N] col), D [M,N] row
  // cublas 列主视角:C' = B * A';用 CUBLAS_OP_T/N 组合表达 TN。
  const float alpha = 1.f, beta = 0.f;
  // D_row[M,N] = A[M,K] * B[N,K]^T  <=>  D_col[N,M] = B_col? 直接用列主记号:
  // cublas: C_col[M,N] = opA(A_col) * opB(B_col)。
  // 我们传 A_col = A_row^T (K x M),即 opA=T 得 (M x K);B_col = B_row^T (K x N)? 
  // 更简单:用经典 trick —— 计算 C_col[N,M] = Bmat[N,K](col of [K,N]?) ...
  // 直接显式:cublasSgemm 风格,行主 C = A*B^T:
  //   cublasGemmEx(h, OP_T, OP_N, N, M, K, B_dev(NxK col-major as KxN row?), ...)
  // 采用已验证写法:C_col(N x M) = A_col?? —— 避免绕晕,这里使用:
  //   m=N, n=M, k=K; A = dB 以 [N,K] row-major 存储 → 视作列主 [K,N],lda=K,opA=N? 
  // 最终等价:D_row = A_row * B_row^T  ⟺  cublas: C=dD(N x M col), A=dB(K x N col? )
  // —— 下面这行是本仓库 cublas 探针沿用多年的形式,直接复用:
  cublasStatus_t st = cublasGemmEx(
      h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
      dB, CUDA_R_16F, K,   // B: [N,K] row-major → 列主看是 [K,N],OP_T → (N x K)
      dA, CUDA_R_16F, K,   // A: [M,K] row-major → 列主看是 [K,M],OP_N → (K x M)
      &beta, dD, CUDA_R_32F, N,  // C: 列主 (N x M) = 行主 (M x N)
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  if (st != CUBLAS_STATUS_SUCCESS) { printf("cublas err %d\n", int(st)); exit(1); }
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  for (int i = 0; i < 5; ++i)
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_16F, K,
                 dA, CUDA_R_16F, K, &beta, dD, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT);
  CHECK_CUDA(cudaDeviceSynchronize());
  cudaEventRecord(t0);
  for (int i = 0; i < 50; ++i)
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_16F, K,
                 dA, CUDA_R_16F, K, &beta, dD, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT);
  cudaEventRecord(t1);
  CHECK_CUDA(cudaEventSynchronize(t1));
  float ms = 0;
  cudaEventElapsedTime(&ms, t0, t1);
  return ms / 50.f;
}

int main() {
  // ---- 1) 小形状数值自检:CPU 参考(双精度,反量化同一字节流) ----
  {
    const int M = 128, N = 128, K = 64;
    std::vector<ElementA> hA(M * K);
    std::vector<ElementB> hB(N * K);
    srand(42);
    for (auto& x : hA) x = ElementA((rand() % 1000 - 500) / 500.f);
    for (auto& x : hB) x = ElementB((rand() % 1000 - 500) / 500.f);
    ElementA* dA; ElementB* dB; float* dD;
    CHECK_CUDA(cudaMalloc(&dA, M * K));
    CHECK_CUDA(cudaMalloc(&dB, N * K));
    CHECK_CUDA(cudaMalloc(&dD, M * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), M * K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), N * K, cudaMemcpyHostToDevice));
    bench_fp8(dA, dB, dD, M, N, K);
    std::vector<float> hD(M * N);
    CHECK_CUDA(cudaMemcpy(hD.data(), dD, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    double max_rel = 0;
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        double ref = 0;
        for (int k = 0; k < K; ++k)
          ref += double(float(hA[m * K + k])) * double(float(hB[n * K + k]));
        double got = hD[m * N + n];
        double rel = std::abs(got - ref) / (std::abs(ref) + 1e-6);
        max_rel = fmax(max_rel, rel);
      }
    printf("sanity 128x128x64: max_rel_err=%.3e %s\n", max_rel,
           max_rel < 1e-3 ? "PASS" : "FAIL");
    cudaFree(dA); cudaFree(dB); cudaFree(dD);
  }

  // ---- 2) 三生产形状计时:FP8 CUTLASS vs FP16 cublas ----
  struct Shape { int M, N, K; const char* name; };
  Shape shapes[] = {
      {5776, 2304, 384, "ffn_up"},
      {5776, 1152, 384, "qkv"},
      {5776, 384, 2304, "ffn_down"},
      {361, 2304, 384, "ffn_up_B1"},
      {361, 384, 384, "outproj_B1"},
  };
  cublasHandle_t h;
  cublasCreate(&h);
  for (auto& s : shapes) {
    size_t szA = size_t(s.M) * s.K, szB = size_t(s.N) * s.K, szD = size_t(s.M) * s.N;
    std::vector<ElementA> hA(szA);
    std::vector<ElementB> hB(szB);
    srand(7);
    for (auto& x : hA) x = ElementA((rand() % 1000 - 500) / 500.f);
    for (auto& x : hB) x = ElementB((rand() % 1000 - 500) / 500.f);
    ElementA* dA8; ElementB* dB8; float* dD;
    CHECK_CUDA(cudaMalloc(&dA8, szA));
    CHECK_CUDA(cudaMalloc(&dB8, szB));
    CHECK_CUDA(cudaMalloc(&dD, szD * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA8, hA.data(), szA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB8, hB.data(), szB, cudaMemcpyHostToDevice));

    __half *dA16, *dB16;
    CHECK_CUDA(cudaMalloc(&dA16, szA * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB16, szB * sizeof(__half)));
    // f16 侧内容不影响计时,填 0 即可
    CHECK_CUDA(cudaMemset(dA16, 0, szA * sizeof(__half)));
    CHECK_CUDA(cudaMemset(dB16, 0, szB * sizeof(__half)));

    float ms8 = bench_fp8(dA8, dB8, dD, s.M, s.N, s.K);
    float ms16 = bench_cublas_f16(dA16, dB16, dD, h, s.M, s.N, s.K);
    double flop = 2.0 * s.M * s.N * s.K;
    printf("%-10s M=%d N=%d K=%d  fp8 %.4f ms (%.1f TF)  fp16 %.4f ms (%.1f TF)  speedup %.2fx\n",
           s.name, s.M, s.N, s.K, ms8, flop / ms8 / 1e9, ms16, flop / ms16 / 1e9,
           ms16 / ms8);
    cudaFree(dA8); cudaFree(dB8); cudaFree(dD); cudaFree(dA16); cudaFree(dB16);
  }
  cublasDestroy(h);
  return 0;
}
