// ---------------------------------------------------------------------------
// B4 门槛微基准:CUTLASS beta=1 原位残差 GEMM vs cuBLASLt(RANK=time 语义)。
//
// 生产形状(b11fix @ B16, M=5776=B16×361):
//   outproj   N=384 K=384   (attention 输出投影残差进 act384)
//   linear-up N=768 K=384   (bottleneck 上投影残差进 act768)
//   ffn-down  N=384 K=2304  (FFN 下投影残差进 act384)
// 另附 B1(M=361)outproj 作参照。
//
// 布局(与 cuda_exec::hgemm_residual 一致):A=[M,K] f16 行主;B=权重 out-first
// [N,K] f16(K 连续)= CUTLASS ColumnMajor ld=K;C=D=[M,N] f32 原位。
// cuBLASLt 侧逐位复刻 cuda.rs::cublaslt_select_algos/exec(COMPUTE_32F、
// 列主映射 A_cm=B[k,n] OP_T / B_cm=A[k,m] OP_N、top-8 计时取最优)。
//
// 瓦片:fork linear2 认证配置 128x128x32(warp 64x64x32,3 stages)+ 对照变体。
// 编译(同 build.rs dual_ffn 链路,exe 供本机直接运行):
//   nvcc -O2 -std=c++17 --expt-relaxed-constexpr \
//     -gencode arch=compute_120,code=sm_120 \
//     -ccbin <MSVC Hostx64/x64> -I<D:/code/cutlass>/include \
//     -Xcompiler "/EHsc /bigobj /std:c++17 /Zc:preprocessor /Zc:__cplusplus" \
//     residual_bench.cu -o residual_bench.exe -lcublasLt
// ---------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <utility>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#define CHECK_CUDA(x)                                                          \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      printf("CUDA error %s @%d\n", cudaGetErrorString(e_), __LINE__);         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_LT(x)                                                            \
  do {                                                                         \
    cublasStatus_t s_ = (x);                                                   \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                         \
      printf("cublasLt error %d @%d\n", (int)s_, __LINE__);                    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ---- CUTLASS 变体(A/B half,C/D f32,beta=1 原位) ----
template <class TileShape, class WarpShape, int Stages>
struct GemmVariant {
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor,    // A [M,K]
      cutlass::half_t, cutlass::layout::ColumnMajor, // B [K,N](权重 out-first)
      float, cutlass::layout::RowMajor,              // C = D [M,N] f32
      float,                                         // 累加 f32
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      TileShape, WarpShape, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<float, 8, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages>;
};

using V1 = GemmVariant<cutlass::gemm::GemmShape<128, 128, 32>,
                       cutlass::gemm::GemmShape<64, 64, 32>, 3>::Gemm;  // fork linear2
using V2 = GemmVariant<cutlass::gemm::GemmShape<128, 64, 32>,
                       cutlass::gemm::GemmShape<64, 32, 32>, 3>::Gemm;  // dual_ffn 族
using V3 = GemmVariant<cutlass::gemm::GemmShape<128, 256, 32>,
                       cutlass::gemm::GemmShape<64, 64, 32>, 3>::Gemm;  // 大 N

template <class Gemm>
bool run_cutlass(const half* A, const half* B, float* C, int m, int n, int k,
                 cudaStream_t stream) {
  typename Gemm::Arguments args{
      {m, n, k},
      {reinterpret_cast<const cutlass::half_t*>(A),
       cutlass::layout::RowMajor(k)},
      {reinterpret_cast<const cutlass::half_t*>(B),
       cutlass::layout::ColumnMajor(k)},
      {C, cutlass::layout::RowMajor(n)},
      {C, cutlass::layout::RowMajor(n)},
      {1.0f, 1.0f}};
  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
  if (gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess)
    return false;
  return gemm.run(stream) == cutlass::Status::kSuccess;
}

template <class Gemm>
float bench_cutlass(const half* A, const half* B, float* C, int m, int n, int k,
                    cudaStream_t stream, int iters = 50) {
  for (int i = 0; i < 5; ++i) run_cutlass<Gemm>(A, B, C, m, n, k, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0, stream);
  for (int i = 0; i < iters; ++i) run_cutlass<Gemm>(A, B, C, m, n, k, stream);
  cudaEventRecord(t1, stream);
  CHECK_CUDA(cudaEventSynchronize(t1));
  float ms = 0;
  cudaEventElapsedTime(&ms, t0, t1);
  return ms / iters;
}

// ---- cuBLASLt(复刻 cuda.rs 生产路径,含 RANK=time 的 top-8 计时) ----
namespace lt {

cublasLtHandle_t handle;
void* workspace;
constexpr size_t kWsSize = 32u << 20;

void init() {
  CHECK_LT(cublasLtCreate(&handle));
  CHECK_CUDA(cudaMalloc(&workspace, kWsSize));
}

// 单次执行(列主映射:C_cm[n,m] = OP_T(B[k,n]) * OP_N(A[k,m]))。
void exec_one(cublasLtMatmulDesc_t desc, cublasLtMatrixLayout_t a_lay,
              cublasLtMatrixLayout_t b_lay, cublasLtMatrixLayout_t c_lay,
              const cublasLtMatmulAlgo_t* algo, const half* B_w, const half* A,
              float* C, int m, int n, int k, float beta, cudaStream_t stream) {
  float alpha = 1.0f;
  CHECK_LT(cublasLtMatmul(handle, desc, &alpha, B_w, a_lay, A, b_lay, &beta,
                          C, c_lay, C, c_lay, algo, workspace, kWsSize,
                          stream));
}

struct DescSet {
  cublasLtMatmulDesc_t desc;
  cublasLtMatrixLayout_t a_lay, b_lay, c_lay;
  void create(int m, int n, int k) {
    CHECK_LT(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    CHECK_LT(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    CHECK_LT(cublasLtMatrixLayoutCreate(&a_lay, CUDA_R_16F, k, n, k));
    CHECK_LT(cublasLtMatrixLayoutCreate(&b_lay, CUDA_R_16F, k, m, k));
    CHECK_LT(cublasLtMatrixLayoutCreate(&c_lay, CUDA_R_32F, n, m, n));
  }
  void destroy() {
    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(a_lay);
    cublasLtMatrixLayoutDestroy(b_lay);
    cublasLtMatrixLayoutDestroy(c_lay);
  }
};

// 返回 {启发式首选 ms, top-8 计时最优 ms}。
std::pair<float, float> bench(const half* A, const half* B, float* C, int m,
                              int n, int k, cudaStream_t stream,
                              int iters = 50) {
  DescSet ds;
  ds.create(m, n, k);
  cublasLtMatmulPreference_t pref;
  CHECK_LT(cublasLtMatmulPreferenceCreate(&pref));
  size_t wsz = kWsSize;
  CHECK_LT(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz)));
  cublasLtMatmulHeuristicResult_t heur[8];
  int cnt = 0;
  CHECK_LT(cublasLtMatmulAlgoGetHeuristic(handle, ds.desc, ds.a_lay, ds.b_lay,
                                          ds.c_lay, ds.c_lay, pref, 8, heur,
                                          &cnt));
  cublasLtMatmulPreferenceDestroy(pref);
  if (cnt == 0) {
    printf("cublasLt: no heuristic result\n");
    exit(1);
  }
  float first_ms = -1.f, best_ms = 1e30f;
  for (int c = 0; c < cnt; ++c) {
    if (heur[c].workspaceSize > kWsSize) continue;
    // 计时(beta=1 原位,输出数值会累积;不影响计时)
    for (int i = 0; i < 5; ++i)
      exec_one(ds.desc, ds.a_lay, ds.b_lay, ds.c_lay, &heur[c].algo, B, A, C,
               m, n, k, 1.0f, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0, stream);
    for (int i = 0; i < iters; ++i)
      exec_one(ds.desc, ds.a_lay, ds.b_lay, ds.c_lay, &heur[c].algo, B, A, C,
               m, n, k, 1.0f, stream);
    cudaEventRecord(t1, stream);
    CHECK_CUDA(cudaEventSynchronize(t1));
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= iters;
    if (c == 0) first_ms = ms;
    best_ms = fminf(best_ms, ms);
  }
  ds.destroy();
  return {first_ms, best_ms};
}

}  // namespace lt

// ---- 数值自检(小形状,CPU 双精度参考) ----
static bool sanity(cudaStream_t stream) {
  const int M = 128, N = 384, K = 384;
  std::vector<half> hA(M * K), hB(N * K);
  std::vector<float> hC0(M * N), hD(M * N);
  srand(42);
  for (auto& x : hA) x = __float2half((rand() % 2000 - 1000) / 2000.f);
  for (auto& x : hB) x = __float2half((rand() % 2000 - 1000) / 2000.f);
  for (auto& x : hC0) x = (rand() % 2000 - 1000) / 500.f;
  half *dA, *dB;
  float* dC;
  CHECK_CUDA(cudaMalloc(&dA, hA.size() * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&dB, hB.size() * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&dC, hC0.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(half),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(half),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dC, hC0.data(), hC0.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  bool ok = run_cutlass<V1>(dA, dB, dC, M, N, K, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaMemcpy(hD.data(), dC, hD.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  double max_rel = 0;
  for (int m = 0; m < M && ok; ++m)
    for (int n = 0; n < N; ++n) {
      double ref = hC0[m * N + n];
      for (int k = 0; k < K; ++k)
        ref += double(__half2float(hA[m * K + k])) *
               double(__half2float(hB[n * K + k]));
      double rel = std::abs(hD[m * N + n] - ref) / (std::abs(ref) + 1e-3);
      max_rel = fmax(max_rel, rel);
    }
  // cuBLASLt 同数据对照(重置 C)
  CHECK_CUDA(cudaMemcpy(dC, hC0.data(), hC0.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  lt::DescSet ds;
  ds.create(M, N, K);
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  cublasLtMatmulHeuristicResult_t h1;
  int cnt = 0;
  cublasLtMatmulAlgoGetHeuristic(lt::handle, ds.desc, ds.a_lay, ds.b_lay,
                                 ds.c_lay, ds.c_lay, pref, 1, &h1, &cnt);
  lt::exec_one(ds.desc, ds.a_lay, ds.b_lay, ds.c_lay, &h1.algo, dB, dA, dC, M,
               N, K, 1.0f, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  std::vector<float> hD2(M * N);
  CHECK_CUDA(cudaMemcpy(hD2.data(), dC, hD2.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  double max_diff = 0;
  for (size_t i = 0; i < hD.size(); ++i)
    max_diff = fmax(max_diff, std::abs(hD[i] - hD2[i]) / (std::abs(hD[i]) + 1.0));
  ds.destroy();
  cublasLtMatmulPreferenceDestroy(pref);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  printf("sanity 128x384x384: cutlass_vs_cpu max_rel=%.3e (%s)  "
         "cutlass_vs_lt max_rel=%.3e\n",
         max_rel, max_rel < 2e-3 ? "PASS" : "FAIL", max_diff);
  return max_rel < 2e-3;
}

int main() {
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));
  lt::init();
  if (!sanity(stream)) return 1;

  struct Shape { int m, n, k; const char* name; };
  const Shape shapes[] = {
      {5776, 384, 384, "outproj_B16"},
      {5776, 768, 384, "linup_B16"},
      {5776, 384, 2304, "ffndown_B16"},
      {361, 384, 384, "outproj_B1"},
      {722, 384, 384, "outproj_B2"},
  };
  printf("%-12s %5s %5s %5s  %-26s %-26s %-26s %-26s\n", "shape", "M", "N",
         "K", "Lt#0(heuristic)", "Lt best(top8)", "CUTLASS 128x128 s3",
         "CUTLASS 128x64 s3");
  for (const auto& s : shapes) {
    size_t szA = size_t(s.m) * s.k, szB = size_t(s.n) * s.k,
           szC = size_t(s.m) * s.n;
    half *dA, *dB;
    float* dC;
    CHECK_CUDA(cudaMalloc(&dA, szA * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dB, szB * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dC, szC * sizeof(float)));
    CHECK_CUDA(cudaMemset(dA, 0x3c, szA * sizeof(half)));
    CHECK_CUDA(cudaMemset(dB, 0x3c, szB * sizeof(half)));
    CHECK_CUDA(cudaMemset(dC, 0, szC * sizeof(float)));
    auto [lt0, ltb] = lt::bench(dA, dB, dC, s.m, s.n, s.k, stream);
    float v1 = bench_cutlass<V1>(dA, dB, dC, s.m, s.n, s.k, stream);
    float v2 = bench_cutlass<V2>(dA, dB, dC, s.m, s.n, s.k, stream);
    float v3 = bench_cutlass<V3>(dA, dB, dC, s.m, s.n, s.k, stream);
    double tf = 2.0 * s.m * s.n * s.k / 1e9;
    printf("%-12s %5d %5d %5d  %8.4fms %5.1fTF  %8.4fms %5.1fTF  %8.4fms "
           "%5.1fTF  %8.4fms %5.1fTF  %8.4fms %5.1fTF\n",
           s.name, s.m, s.n, s.k, lt0, tf / lt0, ltb, tf / ltb, v1, tf / v1,
           v2, tf / v2, v3, tf / v3);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
  }
  return 0;
}
