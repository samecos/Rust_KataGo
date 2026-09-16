// Independent experimental candidate; never linked into the production backend.
// Reuse precisely the residual probe's inputs, FP64 gates and ABBA timing.
#define KATAGO_RESIDUAL_PROBE_LIBRARY
#include "tf3_residual_bench.cu"
#include <cstring>

namespace {
template<class T> T require_config(cublasLtMatmulAlgo_t& algo,
    cublasLtMatmulAlgoConfigAttributes_t attr, T expected, bool set) {
  if(set) lt_check(cublasLtMatmulAlgoConfigSetAttribute(&algo,
      attr,&expected,sizeof(expected)), "initialized algo config set");
  T actual{}; size_t written = 0;
  lt_check(cublasLtMatmulAlgoConfigGetAttribute(&algo,attr,&actual,sizeof(actual),&written),
      "initialized algo config get");
  if(written != sizeof(actual) || actual != expected)
    throw std::runtime_error("initialized algo config readback mismatch: " + std::to_string(int(attr)));
  return actual;
}
template<class T> T require_cap(const cublasLtMatmulAlgo_t& algo,
    cublasLtMatmulAlgoCapAttributes_t attr) {
  T value{}; size_t written = 0;
  lt_check(cublasLtMatmulAlgoCapGetAttribute(&algo,attr,&value,sizeof(value),&written),
      "initialized algo capability");
  if(written != sizeof(value)) throw std::runtime_error("incomplete initialized algo capability");
  return value;
}
struct InitializedLt {
  cublasLtMatmulAlgo_t algo{};
  std::string after_init, after_config, after_check;
  size_t workspace_bytes = 0;
  float waves = 0;
  uint64_t numerical_flags = 0;
  uint32_t alignments[4]{};
  int available_ids = 0;
  explicit InitializedLt(Case& d, Lt& lt) {
    if(cublasLtGetVersion() != 130600)
      throw std::runtime_error("this independent candidate requires Lt 130600");
    int ids[1024]{}, count = 0;
    lt_check(cublasLtMatmulAlgoGetIds(lt.handle,CUBLAS_COMPUTE_32F,CUDA_R_32F,
        CUDA_R_16F,CUDA_R_16F,CUDA_R_32F,CUDA_R_32F,1024,ids,&count), "initialized algo IDs");
    if(count < 1 || count > 1024 || std::find(ids,ids+count,21) == ids+count)
      throw std::runtime_error("required initialized algorithm ID 21 not available");
    available_ids = count;
    lt_check(cublasLtMatmulAlgoInit(lt.handle,CUBLAS_COMPUTE_32F,CUDA_R_32F,
        CUDA_R_16F,CUDA_R_16F,CUDA_R_32F,CUDA_R_32F,21,&algo), "initialized algo init");
    after_init = algorithm_hex(algo);
    config(true);
    after_config = algorithm_hex(algo);
    config(false);
    numerical_flags = require_cap<uint64_t>(algo,CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS);
    // Publicly documented flags: HMMA | ACCUMULATOR_32F | INPUT_16F.
    if(numerical_flags != uint64_t(66050))
      throw std::runtime_error("initialized candidate numerical implementation differs");
    const cublasLtMatmulAlgoCapAttributes_t attrs[] = {
      CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES,
      CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES};
    const void* pointers[] = {d.db.p,d.da.p,d.dc.p,d.dc.p};
    for(int i = 0; i < 4; ++i) {
      alignments[i] = require_cap<uint32_t>(algo,attrs[i]);
      const size_t strides[] = {size_t(d.s.k)*sizeof(half),size_t(d.s.k)*sizeof(half),
        size_t(d.s.n)*sizeof(float),size_t(d.s.n)*sizeof(float)};
      if(alignments[i] != 16 || reinterpret_cast<uintptr_t>(pointers[i]) % alignments[i] ||
          strides[i] % alignments[i])
        throw std::runtime_error("initialized candidate pointer alignment mismatch");
    }
    if(reinterpret_cast<uintptr_t>(lt.workspace.p) % 256)
      throw std::runtime_error("initialized candidate workspace alignment mismatch");
    cublasLtMatmulHeuristicResult_t check{};
    lt_check(cublasLtMatmulAlgoCheckForStream(lt.handle,lt.op,lt.a,lt.b,lt.c,lt.c,
        &algo,&check,d.stream.p), "initialized algo check");
    if(check.state != CUBLAS_STATUS_SUCCESS || check.workspaceSize > kWorkspace ||
        !std::isfinite(check.wavesCount))
      throw std::runtime_error("initialized candidate check result invalid");
    // The API explicitly does not initialize check.algo. Execute our original object.
    workspace_bytes = check.workspaceSize; waves = check.wavesCount;
    after_check = algorithm_hex(algo);
    if(after_config != after_check)
      throw std::runtime_error("initialized candidate changed during read-only metadata/check");
  }
  void config(bool set) {
    require_config<int32_t>(algo,CUBLASLT_ALGO_CONFIG_ID,21,false);
    require_config<uint32_t>(algo,CUBLASLT_ALGO_CONFIG_TILE_ID,15,set);
    require_config<int32_t>(algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,1,set);
    require_config<uint32_t>(algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,0,set);
    require_config<uint32_t>(algo,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,0,set);
    require_config<uint32_t>(algo,CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,0,set);
    require_config<uint32_t>(algo,CUBLASLT_ALGO_CONFIG_STAGES_ID,12,set);
    // These documented zero values leave inner shape undefined and cluster AUTO.
    require_config<uint16_t>(algo,CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,0,set);
    require_config<uint16_t>(algo,CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,0,set);
  }
  void run(Case& d, Lt& lt) {
    const float alpha = 1, beta = 1;
    lt_check(cublasLtMatmul(lt.handle,lt.op,&alpha,d.db.p,lt.a,d.da.p,lt.b,&beta,
        d.dc.p,lt.c,d.dc.p,lt.c,&algo,lt.workspace.p,kWorkspace,d.stream.p), "initialized Lt matmul");
  }
  std::string metadata() const {
    std::ostringstream r;
    r << "{\"id\":21,\"tile\":15,\"split_k\":1,\"reduction\":0,\"swizzle\":0,\"custom\":0,"
      << "\"stages\":12,\"inner_shape\":0,\"cluster_shape\":0,\"available_id_count\":" << available_ids
      << ",\"config_set_and_readback\":\"PASS\",\"check_status\":0,\"check_state\":0"
      << ",\"workspace_bytes\":" << workspace_bytes << ",\"waves\":" << waves
      << ",\"numerical_impl_flags\":" << numerical_flags << ",\"min_alignments\":[";
    for(int i = 0; i < 4; ++i) { if(i) r << ','; r << alignments[i]; }
    r << "],\"after_init_hex\":" << quote(after_init)
      << ",\"after_config_hex\":" << quote(after_config)
      << ",\"after_check_hex\":" << quote(after_check)
      << ",\"current_hex\":" << quote(algorithm_hex(algo)) << '}';
    return r.str();
  }
};
std::string initialized_shape(Shape s, const Options& o, bool& all_numeric) {
  Case d(s); Lt lt(s); InitializedLt trial(d,lt);
  std::ostringstream r; r << std::setprecision(12)
    << "{\"name\":" << quote(s.name) << ",\"m\":" << s.m << ",\"n\":" << s.n << ",\"k\":" << s.k;
  if(o.metadata_only) {
    r << ",\"status\":\"METADATA_ONLY\",\"initialized_candidate\":" << trial.metadata()
      << ",\"heuristic_metadata\":" << heuristic_metadata(d,lt) << '}';
    return r.str();
  }
  std::cerr << "Initialized ID21: numeric then ABBA " << s.name << '\n';
  if(lt.heur[0].state != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("baseline heuristic state is not SUCCESS");
  d.reset(); lt.run(d); auto reference = d.download();
  auto base_oracle = oracle(d,reference);
  for(float v : reference) if(!std::isfinite(v)) throw std::runtime_error("non-finite baseline");
  if(base_oracle.failures) throw std::runtime_error("baseline FP64 gate failed");
  d.reset(); trial.run(d,lt); auto values = d.download();
  auto full = compare(values,reference), cpu = oracle(d,values);
  size_t bits_different = 0;
  for(size_t i = 0; i < values.size(); ++i)
    if(std::memcmp(&values[i],&reference[i],sizeof(float))) ++bits_different;
  r << ",\"baseline_fp64_gate\":" << base_oracle.json()
    << ",\"numeric_vs_production_lt\":" << full.json()
    << ",\"numeric_vs_fp64_samples\":" << cpu.json()
    << ",\"output_float_bits_different\":" << bits_different;
  if(full.failures || cpu.failures) {
    all_numeric = false;
    r << ",\"status\":\"NUMERIC_FAIL\",\"production_eligible\":false";
  } else {
    append_pair(r,d,lt,o,[&]{trial.run(d,lt);},"B_initialized_ID21_tile15_stages12");
  }
  if(algorithm_hex(trial.algo) != trial.after_check)
    throw std::runtime_error("initialized algorithm changed after launches");
  r << ",\"initialized_candidate\":" << trial.metadata()
    << ",\"heuristic_metadata\":" << heuristic_metadata(d,lt) << '}';
  return r.str();
}
}
int main(int argc, char** argv) {
  try {
    Options o = parse(argc,argv);
    if(o.suite != "tf3" || o.candidates != "lt")
      throw std::runtime_error("initialized probe requires --suite tf3 --candidates lt");
    if(!o.output.empty() && std::ifstream(o.output).good())
      throw std::runtime_error("refusing to overwrite existing report");
    cuda_check(cudaSetDevice(0), "select device");
    cudaDeviceProp device{}; cuda_check(cudaGetDeviceProperties(&device,0), "device properties");
    int driver = 0, runtime = 0;
    cuda_check(cudaDriverGetVersion(&driver), "driver version");
    cuda_check(cudaRuntimeGetVersion(&runtime), "runtime version");
    if(std::string(device.name) != "NVIDIA GeForce RTX 5070 Ti" || device.major != 12 ||
        device.minor != 0 || device.multiProcessorCount != 70 || device.l2CacheSize != 50331648)
      throw std::runtime_error("initialized candidate was registered for RTX 5070 Ti SM120 only");
    std::vector<Shape> shapes{{5054,384,1152,"tf3_ffndown_B14"},
      {5776,384,1152,"tf3_ffndown_B16"},{5776,384,384,"tf3_outproj_B16"}};
    bool all_numeric = true;
    std::ostringstream r; r << std::setprecision(12)
      << "{\"schema\":1,\"kind\":\"independent_initialized_lt_residual_probe\",\"production_eligible\":false"
      << ",\"platform\":" << quote(platform_os()) << ",\"device\":" << quote(device.name)
      << ",\"driver_version\":" << driver << ",\"runtime_version\":" << runtime
      << ",\"cublaslt_version\":" << cublasLtGetVersion()
      << ",\"cublas_header_version\":{\"major\":" << CUBLAS_VER_MAJOR << ",\"minor\":" << CUBLAS_VER_MINOR
      << ",\"patch\":" << CUBLAS_VER_PATCH << ",\"build\":" << CUBLAS_VER_BUILD << '}'
      << ",\"cublaslt_runtime_properties\":{\"major\":" << runtime_property(MAJOR_VERSION)
      << ",\"minor\":" << runtime_property(MINOR_VERSION) << ",\"patch\":" << runtime_property(PATCH_LEVEL)
      << ",\"build\":null}"
      << ",\"nvcc_version\":\"" << __CUDACC_VER_MAJOR__ << '.' << __CUDACC_VER_MINOR__ << '.' << __CUDACC_VER_BUILD__ << '"'
      << ",\"candidate\":\"public_init_ID21_tile15_splitK1_stages12_r1\""
      << ",\"precision\":\"compute32F_scale32F_AB16F_CD32F_beta1_C_equals_D\""
      << ",\"input\":\"seeded_random_half_A_B_nonzero_float_residual\",\"seed\":1831565813"
      << ",\"layout\":\"Rust_TN_weights_NK\",\"workspace_capacity_bytes\":" << kWorkspace
      << ",\"streams\":1,\"alpha\":1,\"beta\":1,\"c_equals_d\":true"
      << ",\"timing\":\"CUDA_event_span_divided_by_iterations_not_per_op_median\",\"max_allowed_spread\":0.1"
      << ",\"numeric_abs_tolerance\":" << kAbsTolerance << ",\"numeric_rel_tolerance\":" << kRelTolerance
      << ",\"fp64_samples_per_shape\":" << kOracleSamples
      << ",\"metadata_only\":" << (o.metadata_only ? "true" : "false")
      << ",\"warmup\":" << o.warmup << ",\"iterations\":" << o.iterations
      << ",\"abba_rounds\":" << (o.metadata_only ? 0 : o.rounds) << ",\"shapes\":[";
    for(size_t i = 0; i < shapes.size(); ++i) {
      if(i) r << ','; r << initialized_shape(shapes[i],o,all_numeric);
    }
    r << "],\"all_selected_numeric_pass\":" << (o.metadata_only ? "null" : (all_numeric ? "true" : "false")) << "}\n";
    if(o.output.empty()) std::cout << r.str();
    else { std::ofstream f(o.output,std::ios::binary); if(!f || !(f << r.str())) throw std::runtime_error("report write failed"); }
    return all_numeric ? 0 : 2;
  } catch(const std::exception& e) {
    std::cerr << "initialized_lt_probe ERROR: " << e.what() << '\n'; return 1;
  }
}
