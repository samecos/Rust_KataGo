// Standalone, opt-in TF3 residual GEMM probe. Never linked into production.
// Shape/tile references: KataGomo_fork sm120_aot/linear2_residual_cutlass.cu
// (128x128x32, warp 64x64x32, stages=3) and this directory's residual_bench.cu.
// A/B are half; accumulator, epilogue arithmetic and C=D are float throughout.
// Existing Rust TN weights [N,K] are used directly as column-major B[K,N].
// No fast-math. See README.md for measurement and numerical boundaries.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include "cutlass/cutlass.h"
#include "cutlass/version.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"

namespace {
constexpr size_t kWorkspace = 32ULL * 1024 * 1024;
constexpr double kAbsTolerance = 5e-5;
constexpr double kRelTolerance = 5e-5;
constexpr int kOracleSamples = 256;

void cuda_check(cudaError_t e, const char* where) {
  if(e != cudaSuccess)
    throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(e));
}
void lt_check(cublasStatus_t e, const char* where) {
  if(e != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(where) + ": cuBLAS status " + std::to_string(int(e)));
}
std::string quote(const std::string& s) {
  std::string out = "\"";
  for(char c : s) {
    if(c == '\\' || c == '"') { out += '\\'; out += c; }
    else if(c == '\n') out += "\\n";
    else if(static_cast<unsigned char>(c) < 32) out += '?';
    else out += c;
  }
  return out + '"';
}
struct Shape { int m, n, k; std::string name; };
struct Options {
  std::string suite = "tf3", candidates = "cutlass", output;
  int warmup = 10, iterations = 200, rounds = 3;
  bool metadata_only = false;
};
Options parse(int argc, char** argv) {
  Options o;
  for(int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if(arg == "--help") {
      std::cout << "tf3_residual_bench [--suite tf3|legacy] [--candidates cutlass|lt|all] [--warmup 10] "
                   "[--iterations 200] [--rounds 3] [--metadata-only] [--output report.json]\n";
      std::exit(0);
    }
    if(arg == "--metadata-only") { o.metadata_only = true; continue; }
    if(i + 1 == argc) throw std::runtime_error("missing value for " + arg);
    std::string value = argv[++i];
    if(arg == "--suite") o.suite = value;
    else if(arg == "--candidates") o.candidates = value;
    else if(arg == "--output") o.output = value;
    else if(arg == "--warmup") o.warmup = std::stoi(value);
    else if(arg == "--iterations") o.iterations = std::stoi(value);
    else if(arg == "--rounds") o.rounds = std::stoi(value);
    else throw std::runtime_error("unknown option " + arg);
  }
  if((o.suite != "tf3" && o.suite != "legacy") || o.warmup < 1 ||
     o.iterations < 1 || o.iterations > 100000 || o.rounds < 1 || o.rounds > 100)
    throw std::runtime_error("invalid suite or measurement bounds");
  if(o.candidates != "cutlass" && o.candidates != "lt" && o.candidates != "all")
    throw std::runtime_error("candidates must be cutlass, lt or all");
  if(o.suite != "tf3" && o.candidates != "cutlass")
    throw std::runtime_error("fixed Lt candidate indices are registered only for the tf3 suite");
  return o;
}
// Pre-registered from tf3-residual-report.json, SHA256
// 83859747256f74e2e910805ab723e6820b18e5d7183337a80bcce7b2a37b1ca4.
// Indices are after the production workspace-only filter. They are never
// chosen from any timing performed by this invocation.
int fixed_lt_candidate(const Shape& s) {
  if(s.m == 5054 && s.n == 384 && s.k == 1152) return 1;
  if(s.m == 5776 && s.n == 384 && s.k == 1152) return 3;
  if(s.m == 5776 && s.n == 384 && s.k == 384) return 2;
  return -1;
}
std::string algorithm_hex(const cublasLtMatmulAlgo_t& algorithm) {
  std::ostringstream result;
  const auto* bytes = reinterpret_cast<const unsigned char*>(&algorithm);
  for(size_t i = 0; i < sizeof(algorithm); ++i)
    result << std::hex << std::setfill('0') << std::setw(2) << unsigned(bytes[i]);
  return result.str();
}
// Query only documented API attributes; never decode or mask opaque data[] bits.
// Keep unsupported attributes visible, including API status and exact byte sizes.
template<class T, class Getter> std::string attribute_json(const char* type, Getter get) {
  T value{}; size_t written = 0;
  auto status = get(&value,sizeof(value),&written);
  bool success = status == CUBLAS_STATUS_SUCCESS;
  bool complete = success && written == sizeof(value);
  std::ostringstream r;
  r << "{\"status\":" << int(status) << ",\"type\":" << quote(type)
    << ",\"requested_bytes\":" << sizeof(value) << ",\"written_bytes\":";
  if(success) r << written; else r << "null";
  r << ",\"complete\":" << (complete ? "true" : "false") << ",\"value\":";
  if(complete) r << value; else r << "null";
  r << '}'; return r.str();
}
template<class T> std::string config_attribute(const cublasLtMatmulAlgo_t& algorithm,
                                              cublasLtMatmulAlgoConfigAttributes_t attr, const char* type) {
  return attribute_json<T>(type,[&](void* value,size_t size,size_t* written) {
    return cublasLtMatmulAlgoConfigGetAttribute(&algorithm,attr,value,size,written);
  });
}
template<class T> std::string cap_attribute(const cublasLtMatmulAlgo_t& algorithm,
                                           cublasLtMatmulAlgoCapAttributes_t attr, const char* type) {
  return attribute_json<T>(type,[&](void* value,size_t size,size_t* written) {
    return cublasLtMatmulAlgoCapGetAttribute(&algorithm,attr,value,size,written);
  });
}
std::string runtime_property(libraryPropertyType attr) {
  int value = 0;
  auto status = cublasLtGetProperty(attr,&value);
  std::ostringstream r; r << "{\"status\":" << int(status) << ",\"value\":";
  if(status == CUBLAS_STATUS_SUCCESS) r << value; else r << "null";
  r << '}'; return r.str();
}
const char* platform_os() {
#if defined(_WIN32)
  return "windows";
#elif defined(__linux__)
  return "linux";
#else
  return "unknown";
#endif
}
const char* platform_arch() {
#if defined(_M_X64) || defined(__x86_64__)
  return "x86_64";
#elif defined(_M_ARM64) || defined(__aarch64__)
  return "aarch64";
#else
  return "unknown";
#endif
}
template<class T> struct DeviceBuffer {
  T* p = nullptr;
  explicit DeviceBuffer(size_t n) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)), "cudaMalloc");
  }
  ~DeviceBuffer() { if(p) cudaFree(p); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
struct Stream {
  cudaStream_t p = nullptr;
  Stream() { cuda_check(cudaStreamCreateWithFlags(&p, cudaStreamNonBlocking), "stream create"); }
  ~Stream() { if(p) { cudaStreamSynchronize(p); cudaStreamDestroy(p); } }
};
struct Event {
  cudaEvent_t p = nullptr;
  Event() { cuda_check(cudaEventCreate(&p), "event create"); }
  ~Event() { if(p) cudaEventDestroy(p); }
};
// Exact reproducible host inputs without implementation-defined RNG distributions.
float random_value(uint32_t& state) {
  state ^= state << 13; state ^= state >> 17; state ^= state << 5;
  return (float(int(state & 65535U) - 32768)) / 131072.0f;
}
struct Case {
  Shape s;
  Stream stream;
  std::vector<half> a, b;
  std::vector<float> c0;
  DeviceBuffer<half> da, db;
  DeviceBuffer<float> dc, dc0;
  explicit Case(Shape shape) : s(shape), a(size_t(s.m)*s.k), b(size_t(s.n)*s.k),
      c0(size_t(s.m)*s.n), da(a.size()), db(b.size()), dc(c0.size()), dc0(c0.size()) {
    uint32_t seed = 0x6d2b79f5U;
    for(auto& x : a) x = __float2half_rn(random_value(seed));
    for(auto& x : b) x = __float2half_rn(random_value(seed));
    for(auto& x : c0) x = random_value(seed);
    cuda_check(cudaMemcpyAsync(da.p,a.data(),a.size()*sizeof(half),cudaMemcpyHostToDevice,stream.p), "upload A");
    cuda_check(cudaMemcpyAsync(db.p,b.data(),b.size()*sizeof(half),cudaMemcpyHostToDevice,stream.p), "upload B");
    cuda_check(cudaMemcpyAsync(dc0.p,c0.data(),c0.size()*sizeof(float),cudaMemcpyHostToDevice,stream.p), "upload C0");
    reset(); sync();
  }
  void sync() { cuda_check(cudaStreamSynchronize(stream.p), "stream synchronize"); }
  void reset() {
    cuda_check(cudaMemcpyAsync(dc.p,dc0.p,c0.size()*sizeof(float),cudaMemcpyDeviceToDevice,stream.p), "reset residual");
  }
  std::vector<float> download() {
    std::vector<float> result(c0.size());
    cuda_check(cudaMemcpyAsync(result.data(),dc.p,result.size()*sizeof(float),cudaMemcpyDeviceToHost,stream.p), "download");
    sync(); return result;
  }
};
struct Lt {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  DeviceBuffer<unsigned char> workspace{kWorkspace};
  std::vector<cublasLtMatmulHeuristicResult_t> heur;
  int returned_count = 0;
  explicit Lt(const Shape& s) {
    try {
      lt_check(cublasLtCreate(&handle), "Lt create");
      lt_check(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F), "Lt op");
      cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
      lt_check(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)), "Lt transA");
      lt_check(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)), "Lt transB");
      lt_check(cublasLtMatrixLayoutCreate(&a,CUDA_R_16F,s.k,s.n,s.k), "Lt A layout");
      lt_check(cublasLtMatrixLayoutCreate(&b,CUDA_R_16F,s.k,s.m,s.k), "Lt B layout");
      lt_check(cublasLtMatrixLayoutCreate(&c,CUDA_R_32F,s.n,s.m,s.n), "Lt C layout");
      lt_check(cublasLtMatmulPreferenceCreate(&pref), "Lt preference");
      size_t bytes = kWorkspace;
      lt_check(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes)), "Lt workspace preference");
      cublasLtMatmulHeuristicResult_t found[8]{};
      int count = 0;
      lt_check(cublasLtMatmulAlgoGetHeuristic(handle,op,a,b,c,c,pref,8,found,&count), "Lt heuristics");
      returned_count = count;
      // Mirrors cuda.rs: preserve order and filter only by workspace capacity.
      for(int i = 0; i < count; ++i) if(found[i].workspaceSize <= bytes) heur.push_back(found[i]);
      if(heur.empty()) throw std::runtime_error("Lt returned no production-compatible heuristic");
    } catch(...) { cleanup(); throw; }
  }
  void cleanup() {
    if(pref) { cublasLtMatmulPreferenceDestroy(pref); pref = nullptr; }
    if(c) { cublasLtMatrixLayoutDestroy(c); c = nullptr; }
    if(b) { cublasLtMatrixLayoutDestroy(b); b = nullptr; }
    if(a) { cublasLtMatrixLayoutDestroy(a); a = nullptr; }
    if(op) { cublasLtMatmulDescDestroy(op); op = nullptr; }
    if(handle) { cublasLtDestroy(handle); handle = nullptr; }
  }
  ~Lt() { cleanup(); }
  void run(Case& data, size_t index = 0) {
    const float alpha = 1, beta = 1;
    lt_check(cublasLtMatmul(handle,op,&alpha,data.db.p,a,data.da.p,b,&beta,
        data.dc.p,c,data.dc.p,c,&heur.at(index).algo,workspace.p,kWorkspace,data.stream.p), "Lt matmul");
  }
};
std::string heuristic_metadata(Case& d, Lt& lt) {
  std::ostringstream r; r << std::setprecision(12)
    << "{\"schema\":1,\"requested_count\":8,\"returned_count\":" << lt.returned_count
    << ",\"filtered_count\":" << lt.heur.size()
    << ",\"filter\":\"workspaceSize<=32MiB_preserve_order\",\"candidates\":[";
  for(size_t i = 0; i < lt.heur.size(); ++i) {
    if(i) r << ',';
    const auto& h = lt.heur[i]; const auto& algorithm = h.algo;
    r << "{\"filtered_index\":" << i << ",\"heuristic_state\":" << int(h.state)
      << ",\"algorithm_workspace_bytes\":" << h.workspaceSize
      << ",\"algorithm_bytes_hex\":" << quote(algorithm_hex(algorithm)) << ",\"config\":{";
#define CONFIG_FIELD(name, type, label) \
    r << quote(#name) << ':' << config_attribute<type>(algorithm,CUBLASLT_ALGO_CONFIG_##name,label)
    CONFIG_FIELD(ID,int32_t,"int32_t"); r << ',';
    CONFIG_FIELD(TILE_ID,uint32_t,"uint32_t"); r << ',';
    CONFIG_FIELD(SPLITK_NUM,int32_t,"int32_t"); r << ',';
    CONFIG_FIELD(REDUCTION_SCHEME,uint32_t,"uint32_t"); r << ',';
    CONFIG_FIELD(CTA_SWIZZLING,uint32_t,"uint32_t"); r << ',';
    CONFIG_FIELD(CUSTOM_OPTION,uint32_t,"uint32_t"); r << ',';
    CONFIG_FIELD(STAGES_ID,uint32_t,"uint32_t"); r << ',';
    CONFIG_FIELD(INNER_SHAPE_ID,uint16_t,"uint16_t"); r << ',';
    CONFIG_FIELD(CLUSTER_SHAPE_ID,uint16_t,"uint16_t");
#undef CONFIG_FIELD
    r << "},\"capabilities\":{";
#define CAP_FIELD(name, type, label) \
    r << quote(#name) << ':' << cap_attribute<type>(algorithm,CUBLASLT_ALGO_CAP_##name,label)
    CAP_FIELD(NUMERICAL_IMPL_FLAGS,uint64_t,"uint64_t"); r << ',';
    CAP_FIELD(MIN_ALIGNMENT_A_BYTES,uint32_t,"uint32_t"); r << ',';
    CAP_FIELD(MIN_ALIGNMENT_B_BYTES,uint32_t,"uint32_t"); r << ',';
    CAP_FIELD(MIN_ALIGNMENT_C_BYTES,uint32_t,"uint32_t"); r << ',';
    CAP_FIELD(MIN_ALIGNMENT_D_BYTES,uint32_t,"uint32_t");
#undef CAP_FIELD
    cublasLtMatmulHeuristicResult_t checked{};
    auto status = cublasLtMatmulAlgoCheckForStream(lt.handle,lt.op,lt.a,lt.b,lt.c,lt.c,
                                                 &algorithm,&checked,d.stream.p);
    r << "},\"algo_check_for_stream\":{\"status\":" << int(status) << ",\"result_state\":";
    if(status == CUBLAS_STATUS_SUCCESS) r << int(checked.state); else r << "null";
    r << ",\"workspace_bytes\":";
    if(status == CUBLAS_STATUS_SUCCESS) r << checked.workspaceSize; else r << "null";
    r << ",\"waves_count\":";
    if(status == CUBLAS_STATUS_SUCCESS && std::isfinite(checked.wavesCount)) r << checked.wavesCount;
    else r << "null";
    r << "}}";
  }
  r << "]}"; return r.str();
}
template<int TM, int TN, int WM, int WN> struct CutlassOp {
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t,cutlass::layout::RowMajor,
      cutlass::half_t,cutlass::layout::ColumnMajor,
      float,cutlass::layout::RowMajor,float,
      cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<TM,TN,32>,cutlass::gemm::GemmShape<WM,WN,32>,
      cutlass::gemm::GemmShape<16,8,16>,
      cutlass::epilogue::thread::LinearCombination<float,4,float,float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,3,8,8,false>;
  using Kernel = typename Gemm::GemmKernel;
  typename Kernel::Params params;
  dim3 grid, block;
  cudaFuncAttributes attributes{};
  static constexpr int smem = sizeof(typename Kernel::SharedStorage);
  explicit CutlassOp(Case& d) {
    typename Gemm::Arguments args(
        {d.s.m,d.s.n,d.s.k},
        {reinterpret_cast<const cutlass::half_t*>(d.da.p),d.s.k},
        {reinterpret_cast<const cutlass::half_t*>(d.db.p),d.s.k},
        {d.dc.p,d.s.n},{d.dc.p,d.s.n},{1.0f,1.0f},1);
    if(Gemm::can_implement(args) != cutlass::Status::kSuccess)
      throw std::runtime_error("CUTLASS cannot implement shape");
    typename Gemm::ThreadblockSwizzle swizzle;
    auto tiled = swizzle.get_tiled_shape(args.problem_size,{TM,TN,32},1);
    params = typename Kernel::Params{
        args.problem_size,tiled,args.ref_A.non_const_ref(),args.ref_B.non_const_ref(),
        args.ref_C.non_const_ref(),args.ref_D,args.epilogue,nullptr,
        args.gather_A_indices,args.gather_B_indices,args.scatter_D_indices};
    grid = swizzle.get_grid_shape(tiled);
    block = dim3(Kernel::kThreadCount,1,1);
    cuda_check(cudaFuncSetAttribute(cutlass::Kernel<Kernel>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem), "CUTLASS shared-memory attribute");
    cuda_check(cudaFuncGetAttributes(&attributes,cutlass::Kernel<Kernel>), "CUTLASS function attributes");
  }
  void run(Case& d) {
    // Same cached-Params launch pattern as Fork; no initialize/SetAttribute in timing.
    cutlass::Kernel<Kernel><<<grid,block,smem,d.stream.p>>>(params);
    cuda_check(cudaGetLastError(), "CUTLASS launch");
  }
};
struct ErrorStats {
  double max_abs = 0, max_scaled = 0;
  size_t count = 0, failures = 0;
  void add(double actual, double expected) {
    ++count;
    if(!std::isfinite(actual) || !std::isfinite(expected)) { ++failures; return; }
    double error = std::abs(actual-expected);
    double scaled = error/(kAbsTolerance+kRelTolerance*std::abs(expected));
    max_abs = std::max(max_abs,error); max_scaled = std::max(max_scaled,scaled);
    if(scaled > 1) ++failures;
  }
  std::string json() const {
    std::ostringstream o; o << std::setprecision(12)
      << "{\"count\":" << count << ",\"failures\":" << failures
      << ",\"max_abs\":" << max_abs << ",\"max_gate_ratio\":" << max_scaled << '}'; return o.str();
  }
};
ErrorStats oracle(const Case& d, const std::vector<float>& result) {
  ErrorStats error;
  for(int i = 0; i < kOracleSamples; ++i) {
    // Includes first/last output and spread-out tail rows, rather than only a small tile.
    size_t index = size_t(i)*(result.size()-1)/(kOracleSamples-1);
    size_t row = index/d.s.n, col = index%d.s.n;
    double ref = d.c0[index];
    for(int k = 0; k < d.s.k; ++k)
      ref += double(__half2float(d.a[row*d.s.k+k]))*double(__half2float(d.b[col*d.s.k+k]));
    error.add(result[index],ref);
  }
  return error;
}
ErrorStats compare(const std::vector<float>& actual, const std::vector<float>& reference) {
  ErrorStats error;
  for(size_t i = 0; i < actual.size(); ++i) error.add(actual[i],reference[i]);
  return error;
}
struct Timing { double event_ms, wall_ms; };
Timing measure(Case& d, const std::function<void()>& run, const Options& o) {
  d.reset();
  for(int i = 0; i < o.warmup; ++i) run();
  d.reset(); d.sync();
  Event start, stop;
  auto wall = std::chrono::steady_clock::now();
  cuda_check(cudaEventRecord(start.p,d.stream.p), "event start");
  for(int i = 0; i < o.iterations; ++i) run();
  cuda_check(cudaEventRecord(stop.p,d.stream.p), "event stop");
  cuda_check(cudaEventSynchronize(stop.p), "event sync");
  double wall_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-wall).count();
  float event_ms = 0;
  cuda_check(cudaEventElapsedTime(&event_ms,start.p,stop.p), "event elapsed");
  if(!std::isfinite(event_ms) || event_ms <= 0) throw std::runtime_error("invalid event timing");
  return {double(event_ms)/o.iterations,wall_ms/o.iterations};
}
double geo(const std::vector<double>& x) {
  double sum = 0; for(double v : x) sum += std::log(v); return std::exp(sum/x.size());
}
double spread(const std::vector<double>& x) {
  return *std::max_element(x.begin(),x.end()) / *std::min_element(x.begin(),x.end()) - 1;
}
void append_pair(std::ostream& result, Case& d, Lt& lt, const Options& o,
                 const std::function<void()>& trial_run, const char* trial_name) {
  result << ",\"samples\":[";
  std::vector<double> baseline, trial;
  bool comma = false;
  for(int round = 0; round < o.rounds; ++round) {
    for(int phase = 0; phase < 4; ++phase) {
      bool is_a = phase == 0 || phase == 3;
      Timing t = is_a ? measure(d,[&]{lt.run(d);},o) : measure(d,trial_run,o);
      (is_a ? baseline : trial).push_back(t.event_ms);
      if(comma) result << ','; comma = true;
      result << "{\"round\":" << round << ",\"phase\":" << phase
        << ",\"variant\":" << quote(is_a ? "A_production_lt_heuristic" : trial_name)
        << ",\"event_ms_per_op\":" << t.event_ms << ",\"wall_ms_per_op\":" << t.wall_ms << '}';
    }
  }
  double a = geo(baseline), b = geo(trial), sa = spread(baseline), sb = spread(trial);
  result << "],\"baseline_geo_ms\":" << a << ",\"candidate_geo_ms\":" << b
    << ",\"speedup_percent\":" << 100*(a/b-1)
    << ",\"baseline_spread\":" << sa << ",\"candidate_spread\":" << sb
    << ",\"status\":" << quote((sa > .1 || sb > .1) ? "DIAGNOSTIC_UNSTABLE" : "STABLE")
    << ",\"production_eligible\":false";
}
template<class Op> std::string candidate(Case& d, Lt& lt, const std::vector<float>& reference,
                                       const Options& o, const char* name, bool& all_numeric) {
  std::ostringstream result; result << std::setprecision(12) << "{\"tile\":" << quote(name);
  try {
    Op op(d);
    d.reset(); op.run(d);
    auto values = d.download();
    auto full = compare(values,reference), cpu = oracle(d,values);
    result << ",\"numeric_vs_production_lt\":" << full.json() << ",\"numeric_vs_fp64_samples\":" << cpu.json();
    if(full.failures || cpu.failures) {
      all_numeric = false;
      result << ",\"status\":\"NUMERIC_FAIL\"}";
      return result.str();
    }
    result << ",\"registers\":" << op.attributes.numRegs << ",\"dynamic_smem_bytes\":" << Op::smem
      << ",\"threads\":" << op.block.x;
    append_pair(result,d,lt,o,[&]{op.run(d);},"B_cutlass");
    result << '}';
  } catch(const std::exception& e) {
    all_numeric = false;
    // API failures are fatal: do not emit a partial sample array or silently skip a tile.
    throw std::runtime_error(std::string(name) + ": " + e.what());
  }
  return result.str();
}
std::string lt_candidate(Case& d, Lt& lt, const std::vector<float>& reference,
                        const Options& o, int index, bool& all_numeric) {
  if(index <= 0 || size_t(index) >= lt.heur.size())
    throw std::runtime_error("pre-registered Lt candidate is absent; do not substitute another index");
  if(lt.heur[index].state != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("pre-registered Lt candidate heuristic state is not SUCCESS");
  std::ostringstream result; result << std::setprecision(12)
    << "{\"filtered_index\":" << index << ",\"selection\":\"fixed_from_prior_diagnostic\""
    << ",\"algorithm_workspace_bytes\":" << lt.heur[index].workspaceSize
    << ",\"algorithm_bytes_hex\":" << quote(algorithm_hex(lt.heur[index].algo));
  d.reset(); lt.run(d,size_t(index));
  auto values = d.download();
  auto full = compare(values,reference), cpu = oracle(d,values);
  result << ",\"numeric_vs_production_lt\":" << full.json() << ",\"numeric_vs_fp64_samples\":" << cpu.json();
  if(full.failures || cpu.failures) {
    all_numeric = false;
    result << ",\"status\":\"NUMERIC_FAIL\",\"production_eligible\":false}";
    return result.str();
  }
  append_pair(result,d,lt,o,[&]{lt.run(d,size_t(index));},"B_fixed_lt_candidate");
  result << '}'; return result.str();
}
std::string shape_report(Shape s, const Options& o, bool& all_cutlass_numeric, bool& all_lt_numeric) {
  Case d(s); Lt lt(s);
  if(o.metadata_only) {
    std::ostringstream r;
    r << "{\"name\":" << quote(s.name) << ",\"m\":" << s.m << ",\"n\":" << s.n << ",\"k\":" << s.k
      << ",\"status\":\"METADATA_ONLY\",\"heuristic_metadata\":" << heuristic_metadata(d,lt) << '}';
    return r.str();
  }
  std::cerr << "Numeric gate then ABBA: " << s.name << " M=" << s.m << " N=" << s.n << " K=" << s.k << '\n';
  d.reset(); lt.run(d);
  auto reference = d.download();
  auto base_oracle = oracle(d,reference);
  size_t nonfinite = 0; for(float x : reference) if(!std::isfinite(x)) ++nonfinite;
  if(base_oracle.failures || nonfinite) throw std::runtime_error("production Lt baseline failed FP64/finite gate");
  std::ostringstream r; r << std::setprecision(12)
    << "{\"name\":" << quote(s.name) << ",\"m\":" << s.m << ",\"n\":" << s.n << ",\"k\":" << s.k
    << ",\"baseline_fp64_gate\":" << base_oracle.json()
    << ",\"baseline_filtered_heuristic_index\":0,\"baseline_heuristic_state\":" << int(lt.heur[0].state)
    << ",\"baseline_algorithm_workspace_bytes\":" << lt.heur[0].workspaceSize
    << ",\"baseline_algorithm_bytes_hex\":" << quote(algorithm_hex(lt.heur[0].algo))
    << ",\"candidates\":[";
  if(o.candidates != "lt") {
    r << candidate<CutlassOp<128,128,64,64>>(d,lt,reference,o,"128x128x32_w64x64_s3",all_cutlass_numeric) << ',';
    r << candidate<CutlassOp<128,64,64,32>>(d,lt,reference,o,"128x64x32_w64x32_s3",all_cutlass_numeric) << ',';
    r << candidate<CutlassOp<128,256,64,64>>(d,lt,reference,o,"128x256x32_w64x64_s3",all_cutlass_numeric);
  }
  r << "],\"lt_candidates\":[";
  if(o.candidates != "cutlass" && fixed_lt_candidate(s) >= 0)
    r << lt_candidate(d,lt,reference,o,fixed_lt_candidate(s),all_lt_numeric);
  r << ']';
  // Separate diagnostic; never substitutes a faster heuristic for A in ABBA.
  r << ",\"top8_diagnostic\":[";
  double best = std::numeric_limits<double>::infinity(); int best_index = -1;
  for(size_t i = 0; o.candidates != "lt" && i < lt.heur.size(); ++i) {
    if(i) r << ',';
    r << "{\"filtered_index\":" << i << ",\"heuristic_state\":" << int(lt.heur[i].state);
    if(lt.heur[i].state != CUBLAS_STATUS_SUCCESS) { r << ",\"status\":\"HEURISTIC_REJECTED\"}"; continue; }
    d.reset(); lt.run(d,i);
    auto values = d.download();
    auto full = compare(values,reference), cpu = oracle(d,values);
    r << ",\"numeric_vs_production_lt\":" << full.json() << ",\"numeric_vs_fp64_samples\":" << cpu.json();
    if(full.failures || cpu.failures) { r << ",\"status\":\"NUMERIC_FAIL\"}"; continue; }
    Timing t = measure(d,[&]{lt.run(d,i);},o);
    r << ",\"status\":\"DIAGNOSTIC_ONLY\",\"event_ms_per_op\":" << t.event_ms
      << ",\"wall_ms_per_op\":" << t.wall_ms << '}';
    if(t.event_ms < best) { best = t.event_ms; best_index = int(i); }
  }
  r << "],\"top8_best_filtered_index\":" << best_index << ",\"top8_best_ms\":";
  if(best_index < 0) r << "null"; else r << best;
  // After this shape's unchanged numerical gates and all timing. Attribute and
  // check APIs never execute inside measure() or append_pair().
  r << ",\"heuristic_metadata\":" << heuristic_metadata(d,lt);
  r << '}'; return r.str();
}
} // namespace

#ifndef KATAGO_RESIDUAL_PROBE_LIBRARY
int main(int argc, char** argv) {
  try {
    Options o = parse(argc,argv);
    cuda_check(cudaSetDevice(0), "select device");
    cudaDeviceProp device{}; cuda_check(cudaGetDeviceProperties(&device,0), "device properties");
    int driver = 0, runtime = 0;
    cuda_check(cudaDriverGetVersion(&driver), "driver version");
    cuda_check(cudaRuntimeGetVersion(&runtime), "runtime version");
    std::vector<Shape> shapes = o.suite == "tf3" ? std::vector<Shape>{
        {5054,384,1152,"tf3_ffndown_B14"},{5776,384,1152,"tf3_ffndown_B16"},
        {5054,384,384,"tf3_outproj_B14"},{5776,384,384,"tf3_outproj_B16"}} : std::vector<Shape>{
        {5776,384,384,"legacy_outproj_B16"},{5776,768,384,"legacy_linup_B16"},
        {5776,384,2304,"legacy_ffndown_B16"},{361,384,384,"legacy_outproj_B1"},
        {722,384,384,"legacy_outproj_B2"}};
    if(o.candidates == "lt")
      shapes.erase(std::remove_if(shapes.begin(),shapes.end(),[](const Shape& s){return fixed_lt_candidate(s) < 0;}),shapes.end());
    bool all_cutlass_numeric = true, all_lt_numeric = true;
    std::ostringstream report; report << std::setprecision(12)
      << "{\"schema\":2,\"kind\":\"tf3_residual_gemm_probe\",\"production_eligible\":false"
      << ",\"device\":" << quote(device.name) << ",\"sm\":" << (10*device.major+device.minor)
      << ",\"driver_version\":" << driver << ",\"runtime_version\":" << runtime
      << ",\"cublaslt_version\":" << cublasLtGetVersion()
      << ",\"platform\":{\"os_macro\":" << quote(platform_os()) << ",\"arch_macro\":" << quote(platform_arch()) << '}'
      << ",\"cublas_header_version\":{\"major\":" << CUBLAS_VER_MAJOR << ",\"minor\":" << CUBLAS_VER_MINOR
      << ",\"patch\":" << CUBLAS_VER_PATCH << ",\"build\":" << CUBLAS_VER_BUILD << '}'
      << ",\"cublaslt_runtime_properties\":{\"major\":" << runtime_property(MAJOR_VERSION)
      << ",\"minor\":" << runtime_property(MINOR_VERSION) << ",\"patch\":" << runtime_property(PATCH_LEVEL)
      << ",\"build\":null,\"build_note\":\"GetProperty_has_no_build_property_header_build_is_compile_time_only\"}"
      << ",\"nvcc_version\":\"" << __CUDACC_VER_MAJOR__ << '.' << __CUDACC_VER_MINOR__ << '.' << __CUDACC_VER_BUILD__ << '"'
      << ",\"cutlass_version\":\"" << CUTLASS_MAJOR << '.' << CUTLASS_MINOR << '.' << CUTLASS_PATCH << '"'
      << ",\"suite\":" << quote(o.suite) << ",\"streams\":1,\"warmup\":" << o.warmup
      << ",\"candidate_set\":" << quote(o.candidates)
      << ",\"lt_candidate_source_report\":\"target/fork-parity-20260908/tf3-residual-report.json\""
      << ",\"lt_candidate_source_sha256\":\"83859747256f74e2e910805ab723e6820b18e5d7183337a80bcce7b2a37b1ca4\""
      << ",\"metadata_only\":" << (o.metadata_only ? "true" : "false")
      << ",\"top8_diagnostic_enabled\":" << (o.metadata_only || o.candidates == "lt" ? "false" : "true")
      << ",\"iterations\":" << o.iterations << ",\"abba_rounds\":" << (o.metadata_only ? 0 : o.rounds)
      << ",\"input\":\"seeded_random_half_A_B_nonzero_float_residual\",\"seed\":1831565813"
      << ",\"layout\":\"Rust_TN_weights_NK\",\"workspace_bytes\":" << kWorkspace
      << ",\"precision\":\"half_inputs_weights_float_accumulator_epilogue_residual_output\""
      << ",\"cutlass_epilogue_vector_elements\":" << (o.candidates == "lt" ? "null" : "4")
      << ",\"alpha\":1,\"beta\":1,\"c_equals_d\":true"
      << ",\"numeric_abs_tolerance\":" << kAbsTolerance << ",\"numeric_rel_tolerance\":" << kRelTolerance
      << ",\"fp64_samples_per_shape\":" << kOracleSamples
      << ",\"timing\":\"CUDA_event_span_divided_by_iterations_not_per_op_median\""
      << ",\"max_allowed_spread\":0.1,\"shapes\":[";
    for(size_t i = 0; i < shapes.size(); ++i) {
      if(i) report << ','; report << shape_report(shapes[i],o,all_cutlass_numeric,all_lt_numeric);
    }
    bool all_numeric = all_cutlass_numeric && all_lt_numeric;
    report << "],\"all_selected_numeric_pass\":" << (o.metadata_only ? "null" : (all_numeric ? "true" : "false"))
      << ",\"all_cutlass_numeric_pass\":" << (o.metadata_only || o.candidates == "lt" ? "null" : (all_cutlass_numeric ? "true" : "false"))
      << ",\"all_lt_numeric_pass\":" << (o.metadata_only || o.candidates == "cutlass" ? "null" : (all_lt_numeric ? "true" : "false")) << "}\n";
    if(o.output.empty()) std::cout << report.str();
    else {
      std::ofstream file(o.output,std::ios::binary);
      if(!file || !(file << report.str())) throw std::runtime_error("cannot write report " + o.output);
    }
    return all_numeric ? 0 : 2;
  } catch(const std::exception& e) {
    std::cerr << "tf3_residual_bench ERROR: " << e.what() << '\n';
    return 1;
  }
}
#endif
