//! Low-level FFI bindings to the TensorRT C++ shim (`cpp-shim/src/trt_shim.cpp`).
//!
//! When the `trt` Cargo feature is enabled, the build script compiles the
//! C++ shim and links against CUDA / TensorRT.  These `extern "C"` declarations
//! mirror the functions exposed by `trt_shim.h`.
//!
//! All functions return `None`/`false` when the shim is not available (e.g.
//! when the `trt` feature was enabled but CUDA could not be found), allowing
//! callers to fall back gracefully.

#![allow(non_camel_case_types, dead_code)]

use std::ffi::{c_char, c_float, c_int, c_uchar};

// ---------------------------------------------------------------------------
// Opaque handle types
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct KatagoTrtEngine {
    _private: [u8; 0],
}
#[repr(C)]
pub struct KatagoTrtContext {
    _private: [u8; 0],
}
#[repr(C)]
pub struct KatagoTrtBuffers {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// Engine info
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct KatagoTrtEngineInfo {
    pub nn_x_len: c_int,
    pub nn_y_len: c_int,
    pub num_input_channels: c_int,
    pub num_input_global_channels: c_int,
    pub num_input_meta_channels: c_int,
    pub num_policy_channels: c_int,
    pub num_value_channels: c_int,
    pub num_score_value_channels: c_int,
    pub num_ownership_channels: c_int,
    pub max_batch_size: c_int,
}

// ---------------------------------------------------------------------------
// Generic engine info
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KatagoTrtTensorInfo {
    pub name: [c_char; 128],
    pub dims: [i64; 8],
    pub nb_dims: c_int,
}

impl Default for KatagoTrtTensorInfo {
    fn default() -> Self {
        Self {
            name: [0; 128],
            dims: [0; 8],
            nb_dims: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// FFI declarations
// ---------------------------------------------------------------------------

unsafe extern "C" {
    // Error reporting.
    pub fn katago_trt_last_error() -> *const c_char;

    // Engine lifecycle.
    pub fn katago_trt_engine_create_from_onnx(
        onnx_data: *const c_uchar,
        onnx_size: usize,
        max_batch_size: c_int,
        use_fp16: c_int,
        require_exact_nn_len: c_int,
    ) -> *mut KatagoTrtEngine;

    pub fn katago_trt_engine_create_from_onnx_file(
        onnx_file_path: *const c_char,
        max_batch_size: c_int,
        use_fp16: c_int,
        require_exact_nn_len: c_int,
    ) -> *mut KatagoTrtEngine;

    pub fn katago_trt_engine_destroy(engine: *mut KatagoTrtEngine);

    pub fn katago_trt_engine_get_info(
        engine: *const KatagoTrtEngine,
        info_out: *mut KatagoTrtEngineInfo,
    ) -> c_int;

    pub fn katago_trt_engine_serialize_to_file(
        engine: *const KatagoTrtEngine,
        file_path: *const c_char,
    ) -> c_int;

    pub fn katago_trt_engine_deserialize_from_file(
        file_path: *const c_char,
    ) -> *mut KatagoTrtEngine;

    pub fn katago_trt_engine_deserialize_generic(
        file_path: *const c_char,
    ) -> *mut KatagoTrtEngine;

    // Execution context.
    pub fn katago_trt_context_create(
        engine: *const KatagoTrtEngine,
        gpu_idx: c_int,
    ) -> *mut KatagoTrtContext;

    pub fn katago_trt_context_destroy(ctx: *mut KatagoTrtContext);

    pub fn katago_trt_context_is_fp16(ctx: *const KatagoTrtContext) -> c_int;

    // I/O buffers.
    pub fn katago_trt_buffers_create(
        engine: *const KatagoTrtEngine,
        max_batch_size: c_int,
    ) -> *mut KatagoTrtBuffers;

    pub fn katago_trt_buffers_destroy(bufs: *mut KatagoTrtBuffers);

    // Input pointers (float* → caller fills).
    pub fn katago_trt_buffers_input_spatial(bufs: *mut KatagoTrtBuffers) -> *mut c_float;
    pub fn katago_trt_buffers_input_global(bufs: *mut KatagoTrtBuffers) -> *mut c_float;
    pub fn katago_trt_buffers_input_meta(bufs: *mut KatagoTrtBuffers) -> *mut c_float;
    pub fn katago_trt_buffers_input_mask(bufs: *mut KatagoTrtBuffers) -> *mut c_float;

    // Per-element counts.
    pub fn katago_trt_buffers_single_spatial_elts(bufs: *const KatagoTrtBuffers) -> c_int;
    pub fn katago_trt_buffers_single_global_elts(bufs: *const KatagoTrtBuffers) -> c_int;
    pub fn katago_trt_buffers_single_meta_elts(bufs: *const KatagoTrtBuffers) -> c_int;
    pub fn katago_trt_buffers_single_mask_elts(bufs: *const KatagoTrtBuffers) -> c_int;

    // Output pointers (const float* → caller reads after infer).
    pub fn katago_trt_buffers_output_policy_pass(bufs: *const KatagoTrtBuffers) -> *const c_float;
    pub fn katago_trt_buffers_output_policy(bufs: *const KatagoTrtBuffers) -> *const c_float;
    pub fn katago_trt_buffers_output_value(bufs: *const KatagoTrtBuffers) -> *const c_float;
    pub fn katago_trt_buffers_output_score_value(bufs: *const KatagoTrtBuffers) -> *const c_float;
    pub fn katago_trt_buffers_output_ownership(bufs: *const KatagoTrtBuffers) -> *const c_float;

    pub fn katago_trt_buffers_single_policy_pass_elts(bufs: *const KatagoTrtBuffers) -> c_int;
    pub fn katago_trt_buffers_single_policy_elts(bufs: *const KatagoTrtBuffers) -> c_int;
    pub fn katago_trt_buffers_single_value_elts(bufs: *const KatagoTrtBuffers) -> c_int;
    pub fn katago_trt_buffers_single_score_value_elts(bufs: *const KatagoTrtBuffers) -> c_int;
    pub fn katago_trt_buffers_single_ownership_elts(bufs: *const KatagoTrtBuffers) -> c_int;

    // Inference.
    pub fn katago_trt_infer(
        ctx: *mut KatagoTrtContext,
        bufs: *mut KatagoTrtBuffers,
        batch_size: c_int,
    ) -> c_int;

    // Generic ONNX engine + inference.
    pub fn katago_trt_engine_create_generic(
        onnx_data: *const c_uchar,
        onnx_size: usize,
        max_batch_size: c_int,
        use_fp16: c_int,
    ) -> *mut KatagoTrtEngine;

    pub fn katago_trt_engine_num_inputs(engine: *const KatagoTrtEngine) -> c_int;
    pub fn katago_trt_engine_num_outputs(engine: *const KatagoTrtEngine) -> c_int;

    pub fn katago_trt_engine_input_info(
        engine: *const KatagoTrtEngine,
        idx: c_int,
        out: *mut KatagoTrtTensorInfo,
    ) -> c_int;

    pub fn katago_trt_engine_output_info(
        engine: *const KatagoTrtEngine,
        idx: c_int,
        out: *mut KatagoTrtTensorInfo,
    ) -> c_int;

    pub fn katago_trt_infer_generic(
        ctx: *mut KatagoTrtContext,
        batch_size: c_int,
        input_ptrs: *const *const c_float,
        output_ptrs: *const *mut c_float,
    ) -> c_int;
}

// ---------------------------------------------------------------------------
// Safe wrappers
// ---------------------------------------------------------------------------

/// Last error message from the C++ shim (thread-local).
pub fn last_error() -> Option<String> {
    unsafe {
        let ptr = katago_trt_last_error();
        if ptr.is_null() {
            return None;
        }
        let s = std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned();
        if s.is_empty() { None } else { Some(s) }
    }
}
