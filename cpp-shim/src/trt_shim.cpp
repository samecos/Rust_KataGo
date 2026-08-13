// TensorRT backend C-shim implementation.
//
// Provides the functions declared in trt_shim.h.  This file is intended to be
// compiled by a Rust build.rs script using the `cc` crate.

#include "trt_shim.h"

#include <cuda_runtime.h>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ==========================================================================
// Helpers
// ==========================================================================

// Thread-local error buffer so that katago_trt_last_error() is always valid
// on the calling thread even after the outer handle has been destroyed.
static thread_local std::string tls_last_error;

static void set_error(const std::string& msg) {
    tls_last_error = msg;
}

static void clear_error() { tls_last_error.clear(); }

const char* katago_trt_last_error(void) { return tls_last_error.c_str(); }

// Logger that forwards TRT messages to stderr (quiet at INFO level).
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Suppress info messages to keep output clean.
        if (severity <= Severity::kINFO) return;
        const char* prefix = "";
        switch (severity) {
        case Severity::kINTERNAL_ERROR: prefix = "[TRT INTERNAL] "; break;
        case Severity::kERROR:          prefix = "[TRT ERROR] ";    break;
        case Severity::kWARNING:        prefix = "[TRT WARN] ";     break;
        case Severity::kINFO:           prefix = "[TRT INFO] ";     break;
        case Severity::kVERBOSE:        return; // too noisy
        }
        if (severity >= Severity::kERROR) {
            set_error(std::string(prefix) + msg);
        }
        std::cerr << prefix << msg << std::endl;
    }
};

static TrtLogger g_trt_logger;

// Serialize a vector of bytes to a binary file.
static bool write_binary_file(const std::string& path, const std::vector<uint8_t>& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(data.data()), data.size());
    return out.good();
}

// Read an entire binary file into a vector.
static std::vector<uint8_t> read_binary_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return {};
    auto size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    in.read(reinterpret_cast<char*>(data.data()), size);
    if (!in.good()) return {};
    return data;
}

// ==========================================================================
// Engine
// ==========================================================================

struct KatagoTrtEngine {
    KatagoTrtEngineInfo info;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;

    // Generic-engine I/O snapshots (creation order), filled by
    // katago_trt_engine_create_generic. Empty for the name-based engines.
    std::vector<std::string> generic_input_names;
    std::vector<std::string> generic_output_names;
    std::vector<std::vector<int64_t>> generic_input_dims;
    std::vector<std::vector<int64_t>> generic_output_dims;

    ~KatagoTrtEngine() { /* runtime & engine auto-destroy */ }
};

KatagoTrtEngine* katago_trt_engine_create_from_onnx(
    const uint8_t* onnx_data, size_t onnx_size,
    int max_batch_size, int use_fp16, int require_exact_nn_len) {
    clear_error();

    if (!onnx_data || onnx_size == 0) {
        set_error("ONNX data is null or empty");
        return nullptr;
    }
    if (max_batch_size <= 0) {
        set_error("max_batch_size must be positive");
        return nullptr;
    }

    // --- Build TRT engine from ONNX ---------------------------------------------------
    const auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(g_trt_logger));
    if (!builder) {
        set_error("createInferBuilder failed");
        return nullptr;
    }

    const auto networkFlags =
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    const auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(networkFlags));
    if (!network) {
        set_error("createNetworkV2 failed");
        return nullptr;
    }

    const auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, g_trt_logger));
    if (!parser) {
        set_error("createParser (nvonnxparser) failed");
        return nullptr;
    }

    if (!parser->parse(onnx_data, onnx_size)) {
        std::ostringstream oss;
        oss << "ONNX parsing failed";
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            oss << "\n  " << parser->getError(i)->desc();
        }
        set_error(oss.str());
        return nullptr;
    }

    // Query I/O tensors from the parsed network.
    // We expect these tensor names (from KataGo's OnnxModelBuilder):
    //   InputMask, InputSpatial, InputGlobal, InputMeta (optional),
    //   OutputPolicyPass, OutputPolicy, OutputValue, OutputScoreValue, OutputOwnership
    int num_inputs = network->getNbInputs();
    std::map<std::string, nvinfer1::Dims> input_dims;
    for (int i = 0; i < num_inputs; ++i) {
        const auto* tensor = network->getInput(i);
        input_dims[tensor->getName()] = tensor->getDimensions();
    }

    // Attempt to deduce nn_x_len and nn_y_len from InputSpatial shape [N, C, Y, X].
    int nn_x_len = 0, nn_y_len = 0;
    if (input_dims.count("InputSpatial")) {
        auto dims = input_dims["InputSpatial"];
        // Expect 4D: [batch, channels, height, width]
        if (dims.nbDims >= 4) {
            nn_y_len = dims.d[dims.nbDims - 2];
            nn_x_len = dims.d[dims.nbDims - 1];
        }
    }
    if (nn_x_len <= 0 || nn_y_len <= 0) {
        set_error("Could not deduce nn_x_len/nn_y_len from ONNX model; "
                  "ensure InputSpatial has shape [N, C, Y, X]");
        return nullptr;
    }

    int num_input_channels = 0;
    if (input_dims.count("InputSpatial") && input_dims["InputSpatial"].nbDims >= 3) {
        num_input_channels = input_dims["InputSpatial"].d[1];
    }
    int num_input_global_channels = 0;
    if (input_dims.count("InputGlobal") && input_dims["InputGlobal"].nbDims >= 2) {
        num_input_global_channels = input_dims["InputGlobal"].d[1];
    }
    int num_input_meta_channels = 0;
    if (input_dims.count("InputMeta") && input_dims["InputMeta"].nbDims >= 2) {
        num_input_meta_channels = input_dims["InputMeta"].d[1];
    }

    // Query output dims.
    int num_policy_channels = 1;
    int num_value_channels = 3;
    int num_score_value_channels = 1;
    int num_ownership_channels = 1;

    for (int i = 0; i < network->getNbOutputs(); ++i) {
        const auto* tensor = network->getOutput(i);
        std::string name = tensor->getName();
        auto dims = tensor->getDimensions();
        if (name == "OutputPolicy" && dims.nbDims >= 2) {
            num_policy_channels = dims.d[1];
        } else if (name == "OutputValue" && dims.nbDims >= 2) {
            num_value_channels = dims.d[1];
        } else if (name == "OutputScoreValue" && dims.nbDims >= 2) {
            num_score_value_channels = dims.d[1];
        } else if (name == "OutputOwnership" && dims.nbDims >= 2) {
            num_ownership_channels = dims.d[1];
        }
    }

    // --- Builder config and optimization profile --------------------------------------
    const auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    if (!config) {
        set_error("createBuilderConfig failed");
        return nullptr;
    }

    // Set a moderate workspace size (2 GiB).
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 2ULL << 30);

    if (use_fp16) {
        if (!builder->platformHasFastFp16()) {
            set_error("FP16 requested but not supported on this platform");
            return nullptr;
        }
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // Create an optimization profile for dynamic batch size [1 .. max_batch_size].
    const auto profile = builder->createOptimizationProfile();
    for (int i = 0; i < num_inputs; ++i) {
        const auto* tensor = network->getInput(i);
        auto dims = tensor->getDimensions();
        // First dim is batch. Set it to -1 for dynamic batch.
        dims.d[0] = -1;
        network->getInput(i)->setDimensions(dims);

        auto minDims = dims, optDims = dims, maxDims = dims;
        minDims.d[0] = 1;
        optDims.d[0] = std::max(1, max_batch_size / 2);
        maxDims.d[0] = max_batch_size;
        profile->setDimensions(tensor->getName(),
                               nvinfer1::OptProfileSelector::kMIN, minDims);
        profile->setDimensions(tensor->getName(),
                               nvinfer1::OptProfileSelector::kOPT, optDims);
        profile->setDimensions(tensor->getName(),
                               nvinfer1::OptProfileSelector::kMAX, maxDims);
    }
    config->addOptimizationProfile(profile);

    // Build the engine.
    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
        set_error("buildSerializedNetwork failed");
        return nullptr;
    }

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(g_trt_logger));
    if (!runtime) {
        set_error("createInferRuntime failed");
        return nullptr;
    }

    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(serialized->data(), serialized->size()));
    if (!engine) {
        set_error("deserializeCudaEngine failed");
        return nullptr;
    }

    auto ret = std::unique_ptr<KatagoTrtEngine>(new KatagoTrtEngine());
    ret->info.nn_x_len = nn_x_len;
    ret->info.nn_y_len = nn_y_len;
    ret->info.num_input_channels = num_input_channels;
    ret->info.num_input_global_channels = num_input_global_channels;
    ret->info.num_input_meta_channels = num_input_meta_channels;
    ret->info.num_policy_channels = num_policy_channels;
    ret->info.num_value_channels = num_value_channels;
    ret->info.num_score_value_channels = num_score_value_channels;
    ret->info.num_ownership_channels = num_ownership_channels;
    ret->info.max_batch_size = max_batch_size;
    ret->runtime = std::move(runtime);
    ret->engine = std::move(engine);

    clear_error();
    return ret.release();
}

KatagoTrtEngine* katago_trt_engine_create_from_onnx_file(
    const char* onnx_file_path,
    int max_batch_size, int use_fp16, int require_exact_nn_len) {
    auto data = read_binary_file(onnx_file_path);
    if (data.empty()) {
        set_error(std::string("Failed to read ONNX file: ") + onnx_file_path);
        return nullptr;
    }
    return katago_trt_engine_create_from_onnx(
        data.data(), data.size(), max_batch_size, use_fp16, require_exact_nn_len);
}

void katago_trt_engine_destroy(KatagoTrtEngine* engine) { delete engine; }

int katago_trt_engine_get_info(const KatagoTrtEngine* engine,
                               KatagoTrtEngineInfo* info_out) {
    if (!engine || !info_out) return 0;
    *info_out = engine->info;
    return 1;
}

int katago_trt_engine_serialize_to_file(const KatagoTrtEngine* engine,
                                         const char* file_path) {
    if (!engine || !engine->engine) {
        set_error("Invalid engine");
        return 0;
    }
    const auto serialized = engine->engine->serialize();
    if (!serialized) {
        set_error("Engine serialization failed");
        return 0;
    }
    auto* raw_data = static_cast<const uint8_t*>(serialized->data());
    std::vector<uint8_t> data(raw_data, raw_data + serialized->size());
    delete serialized;
    if (!write_binary_file(file_path, data)) {
        set_error(std::string("Failed to write plan file: ") + file_path);
        return 0;
    }
    return 1;
}

KatagoTrtEngine* katago_trt_engine_deserialize_from_file(const char* file_path) {
    clear_error();
    auto data = read_binary_file(file_path);
    if (data.empty()) {
        set_error(std::string("Failed to read plan file: ") + file_path);
        return nullptr;
    }
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(g_trt_logger));
    if (!runtime) {
        set_error("createInferRuntime failed");
        return nullptr;
    }
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(data.data(), data.size()));
    if (!engine) {
        set_error("deserializeCudaEngine (from plan) failed");
        return nullptr;
    }

    // Reconstruct engine info by querying tensor dims.
    // We have the engine but not the info we stored in KatagoTrtEngine.
    auto ret = std::unique_ptr<KatagoTrtEngine>(new KatagoTrtEngine());

    // Query the first input tensor for batch-independent dims.
    ret->info.max_batch_size = 1;
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        auto name = engine->getIOTensorName(i);
        auto dims = engine->getTensorShape(name);
        std::string sname(name);
        if (sname == "InputSpatial" && dims.nbDims >= 4) {
            ret->info.nn_y_len = dims.d[2];
            ret->info.nn_x_len = dims.d[3];
            ret->info.num_input_channels = dims.d[1];
            ret->info.max_batch_size = dims.d[0];  // from profile max
        } else if (sname == "InputGlobal" && dims.nbDims >= 2) {
            ret->info.num_input_global_channels = dims.d[1];
        } else if (sname == "InputMeta" && dims.nbDims >= 2) {
            ret->info.num_input_meta_channels = dims.d[1];
        } else if (sname == "OutputPolicy" && dims.nbDims >= 2) {
            ret->info.num_policy_channels = dims.d[1];
        } else if (sname == "OutputValue" && dims.nbDims >= 2) {
            ret->info.num_value_channels = dims.d[1];
        } else if (sname == "OutputScoreValue" && dims.nbDims >= 2) {
            ret->info.num_score_value_channels = dims.d[1];
        } else if (sname == "OutputOwnership" && dims.nbDims >= 2) {
            ret->info.num_ownership_channels = dims.d[1];
        }
    }

    ret->runtime = std::move(runtime);
    ret->engine = std::move(engine);

    clear_error();
    return ret.release();
}

// ==========================================================================
// Execution context
// ==========================================================================

struct KatagoTrtContext {
    const KatagoTrtEngine* engine;
    std::unique_ptr<nvinfer1::IExecutionContext> exec;
    int gpu_idx;
};

KatagoTrtContext* katago_trt_context_create(const KatagoTrtEngine* engine,
                                             int gpu_idx) {
    clear_error();
    if (!engine || !engine->engine) {
        set_error("Invalid engine");
        return nullptr;
    }
    auto exec = std::unique_ptr<nvinfer1::IExecutionContext>(
        engine->engine->createExecutionContext());
    if (!exec) {
        set_error("createExecutionContext failed");
        return nullptr;
    }
    auto ctx = std::unique_ptr<KatagoTrtContext>(new KatagoTrtContext());
    ctx->engine = engine;
    ctx->exec = std::move(exec);
    ctx->gpu_idx = gpu_idx;
    clear_error();
    return ctx.release();
}

void katago_trt_context_destroy(KatagoTrtContext* ctx) { delete ctx; }

int katago_trt_context_is_fp16(const KatagoTrtContext* ctx) {
    // Engine-level precision can be queried; for now we return 0.
    // A real implementation would track this from engine creation.
    (void)ctx;
    return 0;
}

// ==========================================================================
// I/O buffers (host staging)
// ==========================================================================

struct KatagoTrtBuffers {
    KatagoTrtEngineInfo info;
    int max_batch_size;

    // Per-batch-element sizes
    int single_mask_elts;
    int single_spatial_elts;
    int single_global_elts;
    int single_meta_elts;
    int single_policy_pass_elts;
    int single_policy_elts;
    int single_value_elts;
    int single_score_value_elts;
    int single_ownership_elts;

    // Host buffers (flat, interleaved by batch: [batch0, batch1, ...])
    std::vector<float> input_mask;
    std::vector<float> input_spatial;
    std::vector<float> input_global;
    std::vector<float> input_meta;
    std::vector<float> output_policy_pass;
    std::vector<float> output_policy;
    std::vector<float> output_value;
    std::vector<float> output_score_value;
    std::vector<float> output_ownership;
};

KatagoTrtBuffers* katago_trt_buffers_create(const KatagoTrtEngine* engine,
                                             int max_batch_size) {
    if (!engine || max_batch_size <= 0) return nullptr;

    const auto& info = engine->info;
    auto bufs = std::unique_ptr<KatagoTrtBuffers>(new KatagoTrtBuffers());
    bufs->info = info;
    bufs->max_batch_size = max_batch_size;

    bufs->single_mask_elts = info.nn_x_len * info.nn_y_len;
    bufs->single_spatial_elts = info.num_input_channels * info.nn_x_len * info.nn_y_len;
    bufs->single_global_elts = info.num_input_global_channels;
    bufs->single_meta_elts = info.num_input_meta_channels;
    bufs->single_policy_pass_elts = info.num_policy_channels;
    bufs->single_policy_elts = info.num_policy_channels * info.nn_x_len * info.nn_y_len;
    bufs->single_value_elts = info.num_value_channels;
    bufs->single_score_value_elts = info.num_score_value_channels;
    bufs->single_ownership_elts = info.num_ownership_channels * info.nn_x_len * info.nn_y_len;

    size_t total_mask = size_t(max_batch_size) * bufs->single_mask_elts;
    size_t total_spatial = size_t(max_batch_size) * bufs->single_spatial_elts;
    size_t total_global = size_t(max_batch_size) * bufs->single_global_elts;
    size_t total_meta = size_t(max_batch_size) * bufs->single_meta_elts;
    size_t total_policy_pass = size_t(max_batch_size) * bufs->single_policy_pass_elts;
    size_t total_policy = size_t(max_batch_size) * bufs->single_policy_elts;
    size_t total_value = size_t(max_batch_size) * bufs->single_value_elts;
    size_t total_score_value = size_t(max_batch_size) * bufs->single_score_value_elts;
    size_t total_ownership = size_t(max_batch_size) * bufs->single_ownership_elts;

    bufs->input_mask.resize(total_mask);
    bufs->input_spatial.resize(total_spatial);
    bufs->input_global.resize(total_global);
    bufs->input_meta.resize(total_meta);
    bufs->output_policy_pass.resize(total_policy_pass);
    bufs->output_policy.resize(total_policy);
    bufs->output_value.resize(total_value);
    bufs->output_score_value.resize(total_score_value);
    bufs->output_ownership.resize(total_ownership);

    return bufs.release();
}

void katago_trt_buffers_destroy(KatagoTrtBuffers* bufs) { delete bufs; }

#define IMPL_PTR_ACCESSOR(RetType, Field)           \
    RetType* katago_trt_buffers_##Field(KatagoTrtBuffers* b) { \
        return b ? b->Field.data() : nullptr;                     \
    }
#define IMPL_CONST_PTR_ACCESSOR(RetType, Field)                      \
    const RetType* katago_trt_buffers_##Field(const KatagoTrtBuffers* b) { \
        return b ? b->Field.data() : nullptr;                                     \
    }
#define IMPL_ELTS_ACCESSOR(Name, Field)                             \
    int katago_trt_buffers_##Name(const KatagoTrtBuffers* b) {     \
        return b ? b->Field : 0;                                   \
    }

IMPL_PTR_ACCESSOR(float, input_spatial)
IMPL_PTR_ACCESSOR(float, input_global)
IMPL_PTR_ACCESSOR(float, input_meta)
IMPL_PTR_ACCESSOR(float, input_mask)
IMPL_CONST_PTR_ACCESSOR(float, output_policy_pass)
IMPL_CONST_PTR_ACCESSOR(float, output_policy)
IMPL_CONST_PTR_ACCESSOR(float, output_value)
IMPL_CONST_PTR_ACCESSOR(float, output_score_value)
IMPL_CONST_PTR_ACCESSOR(float, output_ownership)

IMPL_ELTS_ACCESSOR(single_spatial_elts, single_spatial_elts)
IMPL_ELTS_ACCESSOR(single_global_elts, single_global_elts)
IMPL_ELTS_ACCESSOR(single_meta_elts, single_meta_elts)
IMPL_ELTS_ACCESSOR(single_mask_elts, single_mask_elts)
IMPL_ELTS_ACCESSOR(single_policy_pass_elts, single_policy_pass_elts)
IMPL_ELTS_ACCESSOR(single_policy_elts, single_policy_elts)
IMPL_ELTS_ACCESSOR(single_value_elts, single_value_elts)
IMPL_ELTS_ACCESSOR(single_score_value_elts, single_score_value_elts)
IMPL_ELTS_ACCESSOR(single_ownership_elts, single_ownership_elts)

#undef IMPL_PTR_ACCESSOR
#undef IMPL_CONST_PTR_ACCESSOR
#undef IMPL_ELTS_ACCESSOR

// ==========================================================================
// Inference
// ==========================================================================

// Helper: get the element count for a named tensor.
static int64_t tensor_element_count(const nvinfer1::Dims& dims) {
    int64_t n = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] >= 0)
            n *= dims.d[i];
    }
    return n;
}

int katago_trt_infer(KatagoTrtContext* ctx, KatagoTrtBuffers* bufs,
                     int batch_size) {
    clear_error();

    if (!ctx || !ctx->exec || !bufs) {
        set_error("Invalid context or buffers");
        return 0;
    }
    if (batch_size <= 0 || batch_size > bufs->max_batch_size) {
        set_error("batch_size out of range");
        return 0;
    }

    const auto& eng = ctx->engine->engine;
    const auto& info = ctx->engine->info;

    // Set device and stream.
    cudaError_t cu_err = cudaSetDevice(ctx->gpu_idx);
    if (cu_err != cudaSuccess) {
        set_error(std::string("cudaSetDevice failed: ") + cudaGetErrorString(cu_err));
        return 0;
    }

    cudaStream_t stream = nullptr;
    cu_err = cudaStreamCreate(&stream);
    if (cu_err != cudaSuccess) {
        set_error(std::string("cudaStreamCreate failed: ") + cudaGetErrorString(cu_err));
        return 0;
    }

    // ----------------------------------------------------------------------
    // Collect per-batch-element sizes
    // ----------------------------------------------------------------------
    auto sb = size_t(batch_size);
    size_t mask_bytes     = sb * sizeof(float) * bufs->single_mask_elts;
    size_t spatial_bytes  = sb * sizeof(float) * bufs->single_spatial_elts;
    size_t global_bytes   = sb * sizeof(float) * bufs->single_global_elts;
    size_t meta_bytes     = sb * sizeof(float) * bufs->single_meta_elts;
    size_t policy_pass_bytes = sb * sizeof(float) * bufs->single_policy_pass_elts;
    size_t policy_bytes   = sb * sizeof(float) * bufs->single_policy_elts;
    size_t value_bytes    = sb * sizeof(float) * bufs->single_value_elts;
    size_t score_value_bytes = sb * sizeof(float) * bufs->single_score_value_elts;
    size_t ownership_bytes = sb * sizeof(float) * bufs->single_ownership_elts;

    // ----------------------------------------------------------------------
    // Allocate device memory
    // ----------------------------------------------------------------------
    float *d_mask = nullptr, *d_spatial = nullptr, *d_global = nullptr, *d_meta = nullptr;
    float *d_policy_pass = nullptr, *d_policy = nullptr, *d_value = nullptr;
    float *d_score_value = nullptr, *d_ownership = nullptr;

    // Declared before goto cleanup to avoid initialization-skipped warnings.
    nvinfer1::Dims template4 = {4, {1, 1, info.nn_y_len, info.nn_x_len}};
    nvinfer1::Dims template2 = {2, {1, 1}};
    (void)template4;
    (void)template2;

    auto set_input = [&](const char* name, const nvinfer1::Dims& template_dims,
                         int batch, void* dev_ptr) -> bool {
        auto dims = template_dims;
        dims.d[0] = batch;
        if (!ctx->exec->setInputShape(name, dims)) {
            set_error(std::string("setInputShape failed for ") + name);
            return false;
        }
        if (!ctx->exec->setTensorAddress(name, dev_ptr)) {
            set_error(std::string("setTensorAddress failed for ") + name);
            return false;
        }
        return true;
    };
    (void)set_input;

#define CUDA_ALLOC(name, bytes) \
    if (bytes > 0) {                                                          \
        cu_err = cudaMalloc(&d_##name, bytes);                                \
        if (cu_err != cudaSuccess) {                                          \
            set_error(std::string("cudaMalloc " #name " failed: ") + cudaGetErrorString(cu_err)); \
            goto cleanup;                                                     \
        }                                                                     \
    }

    CUDA_ALLOC(mask, mask_bytes);
    CUDA_ALLOC(spatial, spatial_bytes);
    CUDA_ALLOC(global, global_bytes);
    if (meta_bytes > 0) CUDA_ALLOC(meta, meta_bytes);
    CUDA_ALLOC(policy_pass, policy_pass_bytes);
    CUDA_ALLOC(policy, policy_bytes);
    CUDA_ALLOC(value, value_bytes);
    CUDA_ALLOC(score_value, score_value_bytes);
    CUDA_ALLOC(ownership, ownership_bytes);
#undef CUDA_ALLOC

    // ----------------------------------------------------------------------
    // Copy inputs Host → Device
    // ----------------------------------------------------------------------
    cu_err = cudaMemcpyAsync(d_mask, bufs->input_mask.data(), mask_bytes,
                             cudaMemcpyHostToDevice, stream);
    if (cu_err != cudaSuccess) {
        set_error(std::string("cudaMemcpyAsync mask H2D failed: ") + cudaGetErrorString(cu_err));
        goto cleanup;
    }
    cu_err = cudaMemcpyAsync(d_spatial, bufs->input_spatial.data(), spatial_bytes,
                             cudaMemcpyHostToDevice, stream);
    if (cu_err != cudaSuccess) { set_error("cudaMemcpyAsync spatial H2D failed"); goto cleanup; }
    cu_err = cudaMemcpyAsync(d_global, bufs->input_global.data(), global_bytes,
                             cudaMemcpyHostToDevice, stream);
    if (cu_err != cudaSuccess) { set_error("cudaMemcpyAsync global H2D failed"); goto cleanup; }
    if (meta_bytes > 0) {
        cu_err = cudaMemcpyAsync(d_meta, bufs->input_meta.data(), meta_bytes,
                                 cudaMemcpyHostToDevice, stream);
        if (cu_err != cudaSuccess) { set_error("cudaMemcpyAsync meta H2D failed"); goto cleanup; }
    }

    // ----------------------------------------------------------------------
    // Set input shapes (dynamic batch) and tensor addresses
    // ----------------------------------------------------------------------

    if (d_spatial) {
        template4.d[1] = info.num_input_channels;
        if (!set_input("InputMask",    template4, batch_size, d_mask))    goto cleanup;
        template4.d[1] = info.num_input_channels;
        if (!set_input("InputSpatial", template4, batch_size, d_spatial)) goto cleanup;
    }
    if (d_global) {
        template2.d[1] = info.num_input_global_channels;
        if (!set_input("InputGlobal",  template2, batch_size, d_global))  goto cleanup;
    }
    if (d_meta && info.num_input_meta_channels > 0) {
        template2.d[1] = info.num_input_meta_channels;
        if (!set_input("InputMeta",    template2, batch_size, d_meta))    goto cleanup;
    }

    // Set output tensor addresses.
    if (!ctx->exec->setTensorAddress("OutputPolicyPass", d_policy_pass)) { set_error("setTensorAddress OutputPolicyPass failed"); goto cleanup; }
    if (!ctx->exec->setTensorAddress("OutputPolicy",     d_policy))      { set_error("setTensorAddress OutputPolicy failed"); goto cleanup; }
    if (!ctx->exec->setTensorAddress("OutputValue",      d_value))       { set_error("setTensorAddress OutputValue failed"); goto cleanup; }
    if (!ctx->exec->setTensorAddress("OutputScoreValue", d_score_value)) { set_error("setTensorAddress OutputScoreValue failed"); goto cleanup; }
    if (!ctx->exec->setTensorAddress("OutputOwnership",  d_ownership))   { set_error("setTensorAddress OutputOwnership failed"); goto cleanup; }

    // ----------------------------------------------------------------------
    // Execute
    // ----------------------------------------------------------------------
    if (!ctx->exec->enqueueV3(stream)) {
        set_error("enqueueV3 failed");
        goto cleanup;
    }

    // ----------------------------------------------------------------------
    // Copy outputs Device → Host (synchronize)
    // ----------------------------------------------------------------------
    cu_err = cudaStreamSynchronize(stream);
    if (cu_err != cudaSuccess) {
        set_error(std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(cu_err));
        goto cleanup;
    }

    cu_err = cudaMemcpy(bufs->output_policy_pass.data(), d_policy_pass, policy_pass_bytes,
                        cudaMemcpyDeviceToHost);
    if (cu_err != cudaSuccess) { set_error("cudaMemcpy policy_pass D2H failed"); goto cleanup; }
    cu_err = cudaMemcpy(bufs->output_policy.data(), d_policy, policy_bytes,
                        cudaMemcpyDeviceToHost);
    if (cu_err != cudaSuccess) { set_error("cudaMemcpy policy D2H failed"); goto cleanup; }
    cu_err = cudaMemcpy(bufs->output_value.data(), d_value, value_bytes,
                        cudaMemcpyDeviceToHost);
    if (cu_err != cudaSuccess) { set_error("cudaMemcpy value D2H failed"); goto cleanup; }
    cu_err = cudaMemcpy(bufs->output_score_value.data(), d_score_value, score_value_bytes,
                        cudaMemcpyDeviceToHost);
    if (cu_err != cudaSuccess) { set_error("cudaMemcpy score_value D2H failed"); goto cleanup; }
    cu_err = cudaMemcpy(bufs->output_ownership.data(), d_ownership, ownership_bytes,
                        cudaMemcpyDeviceToHost);
    if (cu_err != cudaSuccess) { set_error("cudaMemcpy ownership D2H failed"); goto cleanup; }

    // ----------------------------------------------------------------------
    // Cleanup
    // ----------------------------------------------------------------------
cleanup: {
        cudaError_t local_err = cudaStreamDestroy(stream);
        if (local_err != cudaSuccess) {
            std::cerr << "cudaStreamDestroy warning: " << cudaGetErrorString(local_err) << std::endl;
        }
    }
#define CUDA_FREE(name) if (d_##name) { cudaFree(d_##name); d_##name = nullptr; }
    CUDA_FREE(mask);
    CUDA_FREE(spatial);
    CUDA_FREE(global);
    if (d_meta) { cudaFree(d_meta); d_meta = nullptr; }
    CUDA_FREE(policy_pass);
    CUDA_FREE(policy);
    CUDA_FREE(value);
    CUDA_FREE(score_value);
    CUDA_FREE(ownership);
#undef CUDA_FREE

    return tls_last_error.empty() ? 1 : 0;
}

// ==========================================================================
// Generic ONNX engine + inference (name-agnostic)
// ==========================================================================

KatagoTrtEngine* katago_trt_engine_deserialize_generic(const char* file_path) {
    clear_error();
    auto data = read_binary_file(file_path);
    if (data.empty()) {
        set_error(std::string("Failed to read plan file: ") + file_path);
        return nullptr;
    }
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(g_trt_logger));
    if (!runtime) {
        set_error("createInferRuntime failed");
        return nullptr;
    }
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(data.data(), data.size()));
    if (!engine) {
        set_error("deserializeCudaEngine (generic) failed");
        return nullptr;
    }

    auto ret = std::unique_ptr<KatagoTrtEngine>(new KatagoTrtEngine());
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        auto name = engine->getIOTensorName(i);
        if (!name) continue;
        auto mode = engine->getTensorIOMode(name);
        auto dims = engine->getTensorShape(name);
        std::vector<int64_t> d(dims.d, dims.d + dims.nbDims);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            ret->generic_input_names.emplace_back(name);
            ret->generic_input_dims.push_back(std::move(d));
        } else {
            ret->generic_output_names.emplace_back(name);
            ret->generic_output_dims.push_back(std::move(d));
        }
    }
    ret->runtime = std::move(runtime);
    ret->engine = std::move(engine);
    clear_error();
    return ret.release();
}

KatagoTrtEngine* katago_trt_engine_create_generic(
    const uint8_t* onnx_data, size_t onnx_size,
    int max_batch_size, int use_fp16) {
    clear_error();

    if (!onnx_data || onnx_size == 0) {
        set_error("ONNX data is null or empty");
        return nullptr;
    }
    if (max_batch_size <= 0) {
        set_error("max_batch_size must be positive");
        return nullptr;
    }

    const auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(g_trt_logger));
    if (!builder) {
        set_error("createInferBuilder failed");
        return nullptr;
    }

    const auto networkFlags =
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    const auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(networkFlags));
    if (!network) {
        set_error("createNetworkV2 failed");
        return nullptr;
    }

    const auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, g_trt_logger));
    if (!parser) {
        set_error("createParser (nvonnxparser) failed");
        return nullptr;
    }

    if (!parser->parse(onnx_data, onnx_size)) {
        std::ostringstream oss;
        oss << "ONNX parsing failed";
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            oss << "\n  " << parser->getError(i)->desc();
        }
        set_error(oss.str());
        return nullptr;
    }

    // Snapshot I/O names and dims in creation order.
    std::vector<std::string> input_names, output_names;
    std::vector<std::vector<int64_t>> input_dims, output_dims;
    for (int i = 0; i < network->getNbInputs(); ++i) {
        const auto* t = network->getInput(i);
        auto d = t->getDimensions();
        input_names.emplace_back(t->getName());
        input_dims.emplace_back(d.d, d.d + d.nbDims);
    }
    for (int i = 0; i < network->getNbOutputs(); ++i) {
        const auto* t = network->getOutput(i);
        auto d = t->getDimensions();
        output_names.emplace_back(t->getName());
        output_dims.emplace_back(d.d, d.d + d.nbDims);
    }

    const auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    if (!config) {
        set_error("createBuilderConfig failed");
        return nullptr;
    }
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 2ULL << 30);
    if (use_fp16) {
        if (!builder->platformHasFastFp16()) {
            set_error("FP16 requested but not supported on this platform");
            return nullptr;
        }
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // Dynamic-batch optimization profile over all inputs.
    const auto profile = builder->createOptimizationProfile();
    for (int i = 0; i < network->getNbInputs(); ++i) {
        const auto* t = network->getInput(i);
        auto dims = t->getDimensions();
        if (dims.nbDims < 1) {
            set_error("input tensor has no batch dimension");
            return nullptr;
        }
        dims.d[0] = -1;
        network->getInput(i)->setDimensions(dims);
        auto minDims = dims, optDims = dims, maxDims = dims;
        minDims.d[0] = 1;
        optDims.d[0] = std::max(1, max_batch_size / 2);
        maxDims.d[0] = max_batch_size;
        profile->setDimensions(t->getName(),
                               nvinfer1::OptProfileSelector::kMIN, minDims);
        profile->setDimensions(t->getName(),
                               nvinfer1::OptProfileSelector::kOPT, optDims);
        profile->setDimensions(t->getName(),
                               nvinfer1::OptProfileSelector::kMAX, maxDims);
    }
    config->addOptimizationProfile(profile);

    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
        set_error("buildSerializedNetwork failed");
        return nullptr;
    }

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(g_trt_logger));
    if (!runtime) {
        set_error("createInferRuntime failed");
        return nullptr;
    }

    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(serialized->data(), serialized->size()));
    if (!engine) {
        set_error("deserializeCudaEngine failed");
        return nullptr;
    }

    auto ret = std::unique_ptr<KatagoTrtEngine>(new KatagoTrtEngine());
    ret->info.max_batch_size = max_batch_size;
    ret->generic_input_names = std::move(input_names);
    ret->generic_output_names = std::move(output_names);
    ret->generic_input_dims = std::move(input_dims);
    ret->generic_output_dims = std::move(output_dims);
    ret->runtime = std::move(runtime);
    ret->engine = std::move(engine);

    clear_error();
    return ret.release();
}

int katago_trt_engine_num_inputs(const KatagoTrtEngine* engine) {
    return engine ? static_cast<int>(engine->generic_input_names.size()) : 0;
}

int katago_trt_engine_num_outputs(const KatagoTrtEngine* engine) {
    return engine ? static_cast<int>(engine->generic_output_names.size()) : 0;
}

static int fill_tensor_info(const std::string& name,
                            const std::vector<int64_t>& dims,
                            KatagoTrtTensorInfo* out) {
    if (!out) return 0;
    size_t n = name.copy(out->name, sizeof(out->name) - 1);
    out->name[n] = '\0';
    out->nb_dims = static_cast<int>(dims.size());
    if (out->nb_dims > 8) out->nb_dims = 8;
    for (int i = 0; i < out->nb_dims; ++i) out->dims[i] = dims[i];
    return 1;
}

int katago_trt_engine_input_info(const KatagoTrtEngine* engine, int idx,
                                 KatagoTrtTensorInfo* out) {
    if (!engine || idx < 0 || idx >= static_cast<int>(engine->generic_input_names.size()))
        return 0;
    return fill_tensor_info(engine->generic_input_names[idx],
                            engine->generic_input_dims[idx], out);
}

int katago_trt_engine_output_info(const KatagoTrtEngine* engine, int idx,
                                  KatagoTrtTensorInfo* out) {
    if (!engine || idx < 0 || idx >= static_cast<int>(engine->generic_output_names.size()))
        return 0;
    return fill_tensor_info(engine->generic_output_names[idx],
                            engine->generic_output_dims[idx], out);
}

int katago_trt_infer_generic(KatagoTrtContext* ctx, int batch_size,
                             const float* const* input_ptrs,
                             float* const* output_ptrs) {
    clear_error();
    if (!ctx || !ctx->exec) {
        set_error("Invalid context");
        return 0;
    }
    if (batch_size <= 0) {
        set_error("batch_size must be positive");
        return 0;
    }

    const auto& in_names = ctx->engine->generic_input_names;
    const auto& out_names = ctx->engine->generic_output_names;
    const auto& in_dims = ctx->engine->generic_input_dims;
    const auto& out_dims = ctx->engine->generic_output_dims;
    if (in_names.empty()) {
        set_error("Not a generic engine");
        return 0;
    }

    cudaError_t cu_err = cudaSetDevice(ctx->gpu_idx);
    if (cu_err != cudaSuccess) {
        set_error(std::string("cudaSetDevice failed: ") + cudaGetErrorString(cu_err));
        return 0;
    }

    cudaStream_t stream = nullptr;
    cu_err = cudaStreamCreate(&stream);
    if (cu_err != cudaSuccess) {
        set_error(std::string("cudaStreamCreate failed: ") + cudaGetErrorString(cu_err));
        return 0;
    }

    auto prod_after_batch = [](const std::vector<int64_t>& dims) -> int64_t {
        int64_t n = 1;
        for (size_t i = 1; i < dims.size(); ++i) {
            if (dims[i] >= 0) n *= dims[i];
        }
        return n;
    };

    std::vector<void*> d_inputs(in_names.size(), nullptr);
    std::vector<void*> d_outputs(out_names.size(), nullptr);
    std::vector<int64_t> in_elts(in_names.size(), 0);
    std::vector<int64_t> out_elts(out_names.size(), 0);
    bool ok = true;

    for (size_t i = 0; i < in_names.size(); ++i) {
        int64_t elts = batch_size * prod_after_batch(in_dims[i]);
        in_elts[i] = elts;
        if (!input_ptrs || !input_ptrs[i]) {
            set_error("null input buffer");
            ok = false;
            break;
        }
        cu_err = cudaMalloc(&d_inputs[i], static_cast<size_t>(elts) * sizeof(float));
        if (cu_err != cudaSuccess) {
            set_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(cu_err));
            ok = false;
            break;
        }
        cu_err = cudaMemcpyAsync(d_inputs[i], input_ptrs[i],
                                 static_cast<size_t>(elts) * sizeof(float),
                                 cudaMemcpyHostToDevice, stream);
        if (cu_err != cudaSuccess) {
            set_error(std::string("cudaMemcpyAsync H2D failed: ") + cudaGetErrorString(cu_err));
            ok = false;
            break;
        }
    }
    for (size_t i = 0; ok && i < out_names.size(); ++i) {
        int64_t elts = batch_size * prod_after_batch(out_dims[i]);
        out_elts[i] = elts;
        if (!output_ptrs || !output_ptrs[i]) {
            set_error("null output buffer");
            ok = false;
            break;
        }
        cu_err = cudaMalloc(&d_outputs[i], static_cast<size_t>(elts) * sizeof(float));
        if (cu_err != cudaSuccess) {
            set_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(cu_err));
            ok = false;
            break;
        }
    }

    if (ok) {
        for (size_t i = 0; i < in_names.size(); ++i) {
            nvinfer1::Dims dims;
            dims.nbDims = static_cast<int>(in_dims[i].size());
            for (int d = 0; d < dims.nbDims; ++d) dims.d[d] = in_dims[i][d];
            dims.d[0] = batch_size;
            if (!ctx->exec->setInputShape(in_names[i].c_str(), dims)) {
                set_error(std::string("setInputShape failed for ") + in_names[i]);
                ok = false;
                break;
            }
            if (!ctx->exec->setTensorAddress(in_names[i].c_str(), d_inputs[i])) {
                set_error(std::string("setTensorAddress failed for ") + in_names[i]);
                ok = false;
                break;
            }
        }
    }
    if (ok) {
        for (size_t i = 0; i < out_names.size(); ++i) {
            if (!ctx->exec->setTensorAddress(out_names[i].c_str(), d_outputs[i])) {
                set_error(std::string("setTensorAddress failed for ") + out_names[i]);
                ok = false;
                break;
            }
        }
    }
    if (ok) {
        if (!ctx->exec->enqueueV3(stream)) {
            set_error("enqueueV3 failed");
            ok = false;
        }
    }
    if (ok) {
        cu_err = cudaStreamSynchronize(stream);
        if (cu_err != cudaSuccess) {
            set_error(std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(cu_err));
            ok = false;
        }
    }
    if (ok) {
        for (size_t i = 0; i < out_names.size(); ++i) {
            cu_err = cudaMemcpy(output_ptrs[i], d_outputs[i],
                                static_cast<size_t>(out_elts[i]) * sizeof(float),
                                cudaMemcpyDeviceToHost);
            if (cu_err != cudaSuccess) {
                set_error(std::string("cudaMemcpy D2H failed: ") + cudaGetErrorString(cu_err));
                ok = false;
                break;
            }
        }
    }

    for (void* p : d_inputs) if (p) cudaFree(p);
    for (void* p : d_outputs) if (p) cudaFree(p);
    cudaStreamDestroy(stream);

    return ok ? 1 : 0;
}
