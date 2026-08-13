#pragma once
// TensorRT backend C-shim API.
//
// This header exposes a minimal C-compatible interface for building TensorRT
// engines from ONNX models and running batched neural-net inference. It is
// designed to be consumed by Rust via FFI.
//
// All functions use C linkage and opaque handle types so that ownership and
// lifecycle are explicit.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Opaque handle types
// ---------------------------------------------------------------------------

typedef struct KatagoTrtEngine KatagoTrtEngine;
typedef struct KatagoTrtContext KatagoTrtContext;
typedef struct KatagoTrtBuffers KatagoTrtBuffers;

// ---------------------------------------------------------------------------
// Error reporting
// ---------------------------------------------------------------------------

// Returns the last error message as a null-terminated string. The buffer is
// owned by the library and is valid until the next shim call on the same
// thread.
const char* katago_trt_last_error(void);

// ---------------------------------------------------------------------------
// Engine lifecycle
// ---------------------------------------------------------------------------

// Describes the I/O dimensions of a loaded engine.
typedef struct {
    int nn_x_len;
    int nn_y_len;
    int num_input_channels;      // spatial features  (e.g. 22)
    int num_input_global_channels; // global features (e.g. 19)
    int num_input_meta_channels;   // SGF metadata channels (0 or 4)
    int num_policy_channels;       // policy head channels
    int num_value_channels;        // value head channels (always 3)
    int num_score_value_channels;  // score/value head channels
    int num_ownership_channels;    // ownership channels (always 1)
    int max_batch_size;            // build-time max batch size
} KatagoTrtEngineInfo;

// Create an engine from serialized ONNX model data.
// - onnx_data / onnx_size: the ONNX ModelProto bytes
// - max_batch_size: maximum batch size the engine must support
// - use_fp16: 1 to build with FP16, 0 for FP32
// - require_exact_nn_len: 1 to enforce exact spatial dims, 0 for smaller boards
// Returns NULL on failure; call katago_trt_last_error() for details.
KatagoTrtEngine* katago_trt_engine_create_from_onnx(
    const uint8_t* onnx_data, size_t onnx_size,
    int max_batch_size, int use_fp16, int require_exact_nn_len);

// Create an engine from an ONNX model file on disk.
KatagoTrtEngine* katago_trt_engine_create_from_onnx_file(
    const char* onnx_file_path,
    int max_batch_size, int use_fp16, int require_exact_nn_len);

// Destroy an engine and free all associated GPU resources.
void katago_trt_engine_destroy(KatagoTrtEngine* engine);

// Query engine I/O dimensions.
int katago_trt_engine_get_info(const KatagoTrtEngine* engine,
                               KatagoTrtEngineInfo* info_out);

// Serialize the engine to a plan file for faster future loading.
// Returns 1 on success, 0 on failure.
int katago_trt_engine_serialize_to_file(const KatagoTrtEngine* engine,
                                        const char* file_path);

// Create an engine from a previously serialized plan file.
KatagoTrtEngine* katago_trt_engine_deserialize_from_file(const char* file_path);

// Deserialize a generic engine from a plan file, re-enumerating its I/O
// tensors from the engine itself (so the generic inference API works without
// the original ONNX model present).
KatagoTrtEngine* katago_trt_engine_deserialize_generic(const char* file_path);

// ---------------------------------------------------------------------------
// Execution context (per thread)
// ---------------------------------------------------------------------------

// Create an execution context bound to the given CUDA device index.
// The context is NOT thread-safe; each server thread needs its own.
KatagoTrtContext* katago_trt_context_create(const KatagoTrtEngine* engine,
                                             int gpu_idx);

void katago_trt_context_destroy(KatagoTrtContext* ctx);

// Returns 1 if the context (and its engine) uses FP16.
int katago_trt_context_is_fp16(const KatagoTrtContext* ctx);

// ---------------------------------------------------------------------------
// I/O buffers (pre-allocated host staging buffers)
// ---------------------------------------------------------------------------

// Create pre-allocated host-side input/output buffers for the given engine.
// These buffers avoid per-call allocations.
KatagoTrtBuffers* katago_trt_buffers_create(const KatagoTrtEngine* engine,
                                             int max_batch_size);

void katago_trt_buffers_destroy(KatagoTrtBuffers* bufs);

// Get raw pointers into the staging buffers so the caller can fill inputs
// and read outputs without extra copies.

// Input pointers (caller fills these before katago_trt_infer):
float* katago_trt_buffers_input_spatial(KatagoTrtBuffers* bufs);
float* katago_trt_buffers_input_global(KatagoTrtBuffers* bufs);
float* katago_trt_buffers_input_meta(KatagoTrtBuffers* bufs);
float* katago_trt_buffers_input_mask(KatagoTrtBuffers* bufs);

// Element counts per single batch element:
int katago_trt_buffers_single_spatial_elts(const KatagoTrtBuffers* bufs);
int katago_trt_buffers_single_global_elts(const KatagoTrtBuffers* bufs);
int katago_trt_buffers_single_meta_elts(const KatagoTrtBuffers* bufs);
int katago_trt_buffers_single_mask_elts(const KatagoTrtBuffers* bufs);

// Output pointers (filled by katago_trt_infer):
const float* katago_trt_buffers_output_policy_pass(const KatagoTrtBuffers* bufs);
const float* katago_trt_buffers_output_policy(const KatagoTrtBuffers* bufs);
const float* katago_trt_buffers_output_value(const KatagoTrtBuffers* bufs);
const float* katago_trt_buffers_output_score_value(const KatagoTrtBuffers* bufs);
const float* katago_trt_buffers_output_ownership(const KatagoTrtBuffers* bufs);

int katago_trt_buffers_single_policy_pass_elts(const KatagoTrtBuffers* bufs);
int katago_trt_buffers_single_policy_elts(const KatagoTrtBuffers* bufs);
int katago_trt_buffers_single_value_elts(const KatagoTrtBuffers* bufs);
int katago_trt_buffers_single_score_value_elts(const KatagoTrtBuffers* bufs);
int katago_trt_buffers_single_ownership_elts(const KatagoTrtBuffers* bufs);

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

// Run a batched inference.
//
// The caller must have already filled:
//   spatial_inputs  (N * single_spatial_elts)  — NCHW layout
//   global_inputs   (N * single_global_elts)
//   mask_inputs     (N * single_mask_elts)     — copied from first spatial channel
// and optionally meta_inputs if single_meta_elts > 0.
//
// On return, the output buffers contain the raw float results for all N
// batch elements (no post-processing, symmetry decoding, or activation).
//
// Returns 1 on success, 0 on failure.
int katago_trt_infer(KatagoTrtContext* ctx, KatagoTrtBuffers* bufs,
                     int batch_size);

// ---------------------------------------------------------------------------
// Generic ONNX engine + inference (name-agnostic, for directly loading
// exported KataGo .onnx models such as b11fix.onnx)
// ---------------------------------------------------------------------------

// Describes a single I/O tensor of a generic engine.
typedef struct {
    char name[128];
    int64_t dims[8];
    int nb_dims;
} KatagoTrtTensorInfo;

// Create an engine from serialized ONNX model data without assuming any
// specific tensor names. All inputs are treated as having a dynamic batch
// (first) dimension, with an optimization profile over [1 .. max_batch_size].
KatagoTrtEngine* katago_trt_engine_create_generic(
    const uint8_t* onnx_data, size_t onnx_size,
    int max_batch_size, int use_fp16);

// Number of input / output tensors of a generic engine.
int katago_trt_engine_num_inputs(const KatagoTrtEngine* engine);
int katago_trt_engine_num_outputs(const KatagoTrtEngine* engine);

// Query input/output tensor info by creation-order index.
// Returns 1 on success, 0 on failure.
int katago_trt_engine_input_info(const KatagoTrtEngine* engine, int idx,
                                 KatagoTrtTensorInfo* out);
int katago_trt_engine_output_info(const KatagoTrtEngine* engine, int idx,
                                  KatagoTrtTensorInfo* out);

// Run batched inference with host-staging buffers supplied by the caller.
// input_ptrs[i] / output_ptrs[i] are flat float arrays in creation-order
// index, each sized batch_size * prod(dims[1..]) elements. The shim copies
// inputs to device, enqueues on a private stream, synchronizes, and copies
// outputs back.
// Returns 1 on success, 0 on failure.
int katago_trt_infer_generic(KatagoTrtContext* ctx, int batch_size,
                             const float* const* input_ptrs,
                             float* const* output_ptrs);

#ifdef __cplusplus
}
#endif
