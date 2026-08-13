//! TensorRT neural-network backend.
//!
//! Corresponds to `cpp/neuralnet/trtbackend.cpp`.
//!
//! When the `trt` Cargo feature is enabled **and** the CUDA/TensorRT SDK are
//! found at build time, this module drives real GPU inference through a C++
//! shim (`katago-rs/cpp-shim/`).  If the SDK is missing, the `trt` feature
//! still compiles but every operation returns a descriptive error.

use std::any::Any;

use kata_core::config::Config;
use kata_core::logger::Logger;

use crate::backend::{
    Backend, ComputeContext, ComputeHandle, Enabled, InputBuffers, LoadedModel, NNOutput,
    NNResultBuf, NeuralNetError,
};
use crate::desc::ModelDesc;

// ==========================================================================
// Module: shim available (CUDA/TensorRT SDK found at build time)
// ==========================================================================

#[cfg(all(feature = "trt", trt_shim_available))]
mod imp {
    use super::*;
    use crate::backends::trt_ffi;
    use kata_game::symmetry::{copy_inputs_with_symmetry, copy_outputs_with_symmetry, invert};
    use std::ffi::CString;
    use std::sync::Mutex;

    // ------------------------------------------------------------------
    // Safe Rust wrappers around the C++ shim handles
    // ------------------------------------------------------------------

    struct TrtEngine {
        ptr: *mut trt_ffi::KatagoTrtEngine,
        info: trt_ffi::KatagoTrtEngineInfo,
    }

    unsafe impl Send for TrtEngine {}
    unsafe impl Sync for TrtEngine {}

    impl TrtEngine {
        fn from_onnx_bytes(
            onnx: &[u8],
            max_batch_size: i32,
            use_fp16: bool,
        ) -> Result<Self, String> {
            let ptr = unsafe {
                trt_ffi::katago_trt_engine_create_from_onnx(
                    onnx.as_ptr(),
                    onnx.len(),
                    max_batch_size,
                    use_fp16 as i32,
                    0,
                )
            };
            if ptr.is_null() {
                return Err(trt_ffi::last_error().unwrap_or_else(|| "unknown engine error".into()));
            }
            let mut info = trt_ffi::KatagoTrtEngineInfo::default();
            unsafe { trt_ffi::katago_trt_engine_get_info(ptr, &mut info) };
            Ok(Self { ptr, info })
        }

        #[allow(dead_code)]
        fn from_onnx_file(path: &str, max_batch_size: i32, use_fp16: bool) -> Result<Self, String> {
            let c_path = CString::new(path).map_err(|_| "invalid path".to_string())?;
            let ptr = unsafe {
                trt_ffi::katago_trt_engine_create_from_onnx_file(
                    c_path.as_ptr(),
                    max_batch_size,
                    use_fp16 as i32,
                    0,
                )
            };
            if ptr.is_null() {
                return Err(trt_ffi::last_error().unwrap_or_else(|| "unknown engine error".into()));
            }
            let mut info = trt_ffi::KatagoTrtEngineInfo::default();
            unsafe { trt_ffi::katago_trt_engine_get_info(ptr, &mut info) };
            Ok(Self { ptr, info })
        }

        /// Build a name-agnostic engine for a directly loaded exported `.onnx`
        /// model, returning the engine together with its I/O tensor layout.
        fn from_onnx_bytes_generic(
            onnx: &[u8],
            max_batch_size: i32,
            use_fp16: bool,
        ) -> Result<(Self, GenericIo), String> {
            let ptr = unsafe {
                trt_ffi::katago_trt_engine_create_generic(
                    onnx.as_ptr(),
                    onnx.len(),
                    max_batch_size,
                    use_fp16 as i32,
                )
            };
            if ptr.is_null() {
                return Err(trt_ffi::last_error().unwrap_or_else(|| "unknown engine error".into()));
            }
            let mut info = trt_ffi::KatagoTrtEngineInfo::default();
            unsafe { trt_ffi::katago_trt_engine_get_info(ptr, &mut info) };

            let num_inputs = unsafe { trt_ffi::katago_trt_engine_num_inputs(ptr) };
            let num_outputs = unsafe { trt_ffi::katago_trt_engine_num_outputs(ptr) };
            let mut input_names = Vec::with_capacity(num_inputs.max(0) as usize);
            let mut input_dims = Vec::with_capacity(num_inputs.max(0) as usize);
            for i in 0..num_inputs.max(0) {
                let mut t = trt_ffi::KatagoTrtTensorInfo::default();
                if unsafe { trt_ffi::katago_trt_engine_input_info(ptr, i, &mut t) } == 0 {
                    return Err("failed to query generic engine input info".to_string());
                }
                let name = unsafe { std::ffi::CStr::from_ptr(t.name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();
                input_names.push(name);
                input_dims.push(t.dims[..t.nb_dims.max(0) as usize].to_vec());
            }
            let mut output_names = Vec::with_capacity(num_outputs.max(0) as usize);
            let mut output_dims = Vec::with_capacity(num_outputs.max(0) as usize);
            for i in 0..num_outputs.max(0) {
                let mut t = trt_ffi::KatagoTrtTensorInfo::default();
                if unsafe { trt_ffi::katago_trt_engine_output_info(ptr, i, &mut t) } == 0 {
                    return Err("failed to query generic engine output info".to_string());
                }
                let name = unsafe { std::ffi::CStr::from_ptr(t.name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();
                output_names.push(name);
                output_dims.push(t.dims[..t.nb_dims.max(0) as usize].to_vec());
            }

            let gio = GenericIo {
                input_names,
                output_names,
                input_dims,
                output_dims,
            };
            Ok((Self { ptr, info }, gio))
        }
    }

    impl Drop for TrtEngine {
        fn drop(&mut self) {
            unsafe { trt_ffi::katago_trt_engine_destroy(self.ptr) };
        }
    }

    struct TrtContext {
        ptr: *mut trt_ffi::KatagoTrtContext,
    }

    unsafe impl Send for TrtContext {}

    impl TrtContext {
        fn new(engine: &TrtEngine, gpu_idx: i32) -> Result<Self, String> {
            let ptr = unsafe { trt_ffi::katago_trt_context_create(engine.ptr, gpu_idx) };
            if ptr.is_null() {
                return Err(trt_ffi::last_error().unwrap_or_else(|| "unknown context error".into()));
            }
            Ok(Self { ptr })
        }
    }

    impl Drop for TrtContext {
        fn drop(&mut self) {
            unsafe { trt_ffi::katago_trt_context_destroy(self.ptr) };
        }
    }

    struct TrtBuffers {
        ptr: *mut trt_ffi::KatagoTrtBuffers,
    }

    unsafe impl Send for TrtBuffers {}

    impl TrtBuffers {
        fn new(engine: &TrtEngine, max_batch_size: i32) -> Result<Self, String> {
            let ptr = unsafe { trt_ffi::katago_trt_buffers_create(engine.ptr, max_batch_size) };
            if ptr.is_null() {
                return Err(trt_ffi::last_error().unwrap_or_else(|| "unknown buffers error".into()));
            }
            Ok(Self { ptr })
        }
    }

    impl Drop for TrtBuffers {
        fn drop(&mut self) {
            unsafe { trt_ffi::katago_trt_buffers_destroy(self.ptr) };
        }
    }

    // ------------------------------------------------------------------
    // Backend types
    // ------------------------------------------------------------------

    pub struct TensorRtLoadedModel {
        model_path: String,
        model_desc: ModelDesc,
    }

    impl LoadedModel for TensorRtLoadedModel {
        fn model_desc(&self) -> &ModelDesc {
            &self.model_desc
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    pub struct TensorRtComputeContext {
        engine: TrtEngine,
        engine_info: trt_ffi::KatagoTrtEngineInfo,
        gpu_idxs: Vec<i32>,
        /// Present when the engine was built from a directly loaded `.onnx`
        /// model via the generic shim path.
        generic: Option<GenericIo>,
    }

    impl ComputeContext for TensorRtComputeContext {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    pub struct TensorRtComputeHandle {
        ctx: TrtContext,
        engine_info: trt_ffi::KatagoTrtEngineInfo,
        max_batch_size: i32,
        /// 求值器输入缓冲的布局（NCHW=false / NHWC=true），随 handle 记录。
        inputs_use_nhwc: bool,
        /// Name-based staging buffers (onnx_builder models); `None` for the
        /// generic onnx path, which uses `generic_bufs` instead.
        bufs: Mutex<Option<TrtBuffers>>,
        generic: Option<GenericIo>,
        generic_bufs: Mutex<Option<GenericHostBufs>>,
    }

    impl ComputeHandle for TensorRtComputeHandle {
        fn is_using_fp16(&self) -> bool {
            unsafe { trt_ffi::katago_trt_context_is_fp16(self.ctx.ptr) != 0 }
        }
        fn set_is_warmup(&mut self, _is_warmup: bool) -> bool {
            false
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// Name-agnostic I/O layout of a generic engine.
    #[derive(Debug, Clone)]
    struct GenericIo {
        input_names: Vec<String>,
        output_names: Vec<String>,
        input_dims: Vec<Vec<i64>>,
        output_dims: Vec<Vec<i64>>,
    }

    impl GenericIo {
        fn input_index(&self, name: &str) -> Option<usize> {
            self.input_names.iter().position(|n| n == name)
        }
        fn output_index(&self, name: &str) -> Option<usize> {
            self.output_names.iter().position(|n| n == name)
        }
        fn single_elts(dims: &[i64]) -> usize {
            dims.iter().skip(1).filter(|&&d| d >= 0).product::<i64>().max(1) as usize
        }

        /// Serialize the layout to a compact text sidecar (one tensor per
        /// line: `input|output <name> <dim0> <dim1> ...`).
        fn to_sidecar(&self) -> String {
            let mut s = String::new();
            for (i, n) in self.input_names.iter().enumerate() {
                s.push_str(&format!("input {n}"));
                for d in &self.input_dims[i] {
                    s.push_str(&format!(" {d}"));
                }
                s.push('\n');
            }
            for (i, n) in self.output_names.iter().enumerate() {
                s.push_str(&format!("output {n}"));
                for d in &self.output_dims[i] {
                    s.push_str(&format!(" {d}"));
                }
                s.push('\n');
            }
            s
        }

        fn from_sidecar(text: &str) -> Result<Self, String> {
            let mut input_names = Vec::new();
            let mut output_names = Vec::new();
            let mut input_dims = Vec::new();
            let mut output_dims = Vec::new();
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let mut parts = line.split_whitespace();
                let kind = parts.next().unwrap_or("");
                let name = parts.next().unwrap_or("");
                let dims: Vec<i64> = parts
                    .map(|d| d.parse::<i64>().map_err(|_| "bad dim".to_string()))
                    .collect::<Result<_, _>>()?;
                match kind {
                    "input" => {
                        input_names.push(name.to_string());
                        input_dims.push(dims);
                    }
                    "output" => {
                        output_names.push(name.to_string());
                        output_dims.push(dims);
                    }
                    _ => return Err(format!("bad sidecar kind: {kind}")),
                }
            }
            if input_names.is_empty() {
                return Err("sidecar has no inputs".to_string());
            }
            Ok(Self {
                input_names,
                output_names,
                input_dims,
                output_dims,
            })
        }
    }

    /// Host staging buffers for the generic inference path.
    struct GenericHostBufs {
        inputs: Vec<Vec<f32>>,
        outputs: Vec<Vec<f32>>,
    }

    /// Build (or load from a plan cache next to the model file) a generic
    /// engine for a directly exported `.onnx` model.
    fn generic_engine_for_model(
        model_path: &str,
        onnx_bytes: &[u8],
        max_batch_size: i32,
        use_fp16: bool,
    ) -> Result<(TrtEngine, GenericIo), String> {
        let plan_path = format!(
            "{model_path}.trtplan.b{}.{}",
            max_batch_size,
            if use_fp16 { "fp16" } else { "fp32" }
        );
        let sidecar_path = format!("{plan_path}.io");

        // Try the cache first.
        if let Ok(text) = std::fs::read_to_string(&sidecar_path) {
            let c_path = CString::new(plan_path.as_str()).map_err(|_| "invalid plan path".to_string())?;
            let ptr = unsafe { trt_ffi::katago_trt_engine_deserialize_generic(c_path.as_ptr()) };
            if !ptr.is_null() {
                if let Ok(gio) = GenericIo::from_sidecar(&text) {
                    let mut info = trt_ffi::KatagoTrtEngineInfo::default();
                    unsafe { trt_ffi::katago_trt_engine_get_info(ptr, &mut info) };
                    return Ok((TrtEngine { ptr, info }, gio));
                }
                unsafe { trt_ffi::katago_trt_engine_destroy(ptr) };
            }
        }

        // Build from ONNX and persist the plan + sidecar.
        let (engine, gio) = TrtEngine::from_onnx_bytes_generic(onnx_bytes, max_batch_size, use_fp16)?;
        let c_path = CString::new(plan_path.as_str()).map_err(|_| "invalid plan path".to_string())?;
        if unsafe { trt_ffi::katago_trt_engine_serialize_to_file(engine.ptr, c_path.as_ptr()) } == 0 {
            // Cache write failure is non-fatal; just remove any partial file.
            let _ = std::fs::remove_file(&plan_path);
        } else {
            let _ = std::fs::write(&sidecar_path, gio.to_sidecar());
        }
        Ok((engine, gio))
    }

    impl GenericHostBufs {
        fn new(gio: &GenericIo, max_batch_size: i32) -> Self {
            let inputs = gio
                .input_dims
                .iter()
                .map(|d| vec![0.0f32; max_batch_size as usize * GenericIo::single_elts(d)])
                .collect();
            let outputs = gio
                .output_dims
                .iter()
                .map(|d| vec![0.0f32; max_batch_size as usize * GenericIo::single_elts(d)])
                .collect();
            Self { inputs, outputs }
        }
    }

    pub struct TensorRtInputBuffers;

    impl InputBuffers for TensorRtInputBuffers {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    pub struct TensorRtBackend;

    impl Backend for TensorRtBackend {
        fn global_initialize(&self) {}
        fn global_cleanup(&self) {}

        fn print_devices(&self) {
            println!("TensorRT backend (C++ shim linked)");
        }

        fn load_model_file(
            &self,
            file: &str,
            _expected_sha256: &str,
        ) -> Result<Box<dyn LoadedModel>, NeuralNetError> {
            // Directly exported KataGo .onnx models are handled via their
            // custom metadata; other formats go through the KataGo model parser.
            if file.ends_with(".onnx") {
                let bytes = std::fs::read(file)
                    .map_err(|e| NeuralNetError(format!("could not read {file}: {e}")))?;
                let parsed = crate::onnx_model::parse_onnx_model(&bytes)
                    .map_err(|e| NeuralNetError(format!("onnx metadata parse failed: {e}")))?;
                return Ok(Box::new(TensorRtLoadedModel {
                    model_path: file.to_string(),
                    model_desc: parsed.model_desc,
                }));
            }
            let model_desc = crate::model_parser::load_model_file(file)
                .map_err(|e| NeuralNetError(format!("model load failed: {e}")))?;
            Ok(Box::new(TensorRtLoadedModel {
                model_path: file.to_string(),
                model_desc,
            }))
        }

        fn create_compute_context(
            &self,
            gpu_idxs: &[i32],
            _logger: &Logger,
            nn_x_len: i32,
            nn_y_len: i32,
            _home_data_dir_override: &str,
            use_fp16_mode: Enabled,
            loaded_model: &dyn LoadedModel,
            cfg: &Config,
        ) -> Result<Box<dyn ComputeContext>, NeuralNetError> {
            let model = loaded_model
                .as_any()
                .downcast_ref::<TensorRtLoadedModel>()
                .ok_or_else(|| NeuralNetError("Wrong loaded model type".to_string()))?;

            // Engine optimization-profile batch size; mirrors the evaluator's
            // nnMaxBatchSize when present.
            let max_batch_size: i32 = if cfg.contains("nnMaxBatchSize") {
                cfg.get_int("nnMaxBatchSize", 1, 65536).unwrap_or(8)
            } else {
                8
            };
            let use_fp16 = !matches!(use_fp16_mode, Enabled::False);

            // 1. Use the ModelDesc parsed during load_model_file.
            let model_desc = model.model_desc.clone();

            // Directly exported KataGo .onnx model: build a generic engine
            // straight from the file (no onnx_builder round-trip).
            if model.model_path.ends_with(".onnx") {
                let bytes = std::fs::read(&model.model_path).map_err(|e| {
                    NeuralNetError(format!("could not read {}: {e}", model.model_path))
                })?;
                let parsed = crate::onnx_model::parse_onnx_model(&bytes)
                    .map_err(|e| NeuralNetError(format!("onnx metadata parse failed: {e}")))?;
                if parsed.inputs.len() != 2 {
                    return Err(NeuralNetError(format!(
                        "expected 2 inputs (spatial+global), found {}",
                        parsed.inputs.len()
                    )));
                }
                let (mut engine, gio) = generic_engine_for_model(
                    &model.model_path,
                    &bytes,
                    max_batch_size,
                    use_fp16,
                )
                .map_err(|e| NeuralNetError(format!("TensorRT engine creation failed: {e}")))?;

                let d = &model_desc;
                engine.info = trt_ffi::KatagoTrtEngineInfo {
                    nn_x_len,
                    nn_y_len,
                    num_input_channels: d.num_input_channels,
                    num_input_global_channels: d.num_input_global_channels,
                    num_input_meta_channels: d.num_input_meta_channels,
                    num_policy_channels: d.num_policy_channels,
                    num_value_channels: d.num_value_channels,
                    num_score_value_channels: d.num_score_value_channels,
                    num_ownership_channels: d.num_ownership_channels,
                    max_batch_size,
                };
                let engine_info = engine.info;

                return Ok(Box::new(TensorRtComputeContext {
                    engine,
                    engine_info,
                    gpu_idxs: gpu_idxs.to_vec(),
                    generic: Some(gio),
                }));
            }

            // 2. Build ONNX from ModelDesc
            let onnx_result = crate::onnx_builder::build(
                &model_desc,
                nn_x_len,
                nn_y_len,
                true,  // require_exact_nn_len
                false, // transformer_nhwc
                _logger,
            )
            .map_err(|e| NeuralNetError(format!("ONNX build failed: {e}")))?;

            // 3. Create TRT engine from ONNX bytes
            let engine =
                TrtEngine::from_onnx_bytes(&onnx_result.serialized_model, max_batch_size, use_fp16)
                    .map_err(|e| NeuralNetError(format!("TensorRT engine creation failed: {e}")))?;
            let engine_info = engine.info;

            Ok(Box::new(TensorRtComputeContext {
                engine,
                engine_info,
                gpu_idxs: gpu_idxs.to_vec(),
                generic: None,
            }))
        }

        fn create_compute_handle(
            &self,
            ctx: &dyn ComputeContext,
            _loaded_model: &dyn LoadedModel,
            _logger: &Logger,
            max_batch_size: i32,
            _require_exact_nn_len: bool,
            inputs_use_nhwc: bool,
            gpu_idx_for_this_thread: i32,
            _server_thread_idx: i32,
        ) -> Result<Box<dyn ComputeHandle>, NeuralNetError> {
            let trt_ctx = ctx
                .as_any()
                .downcast_ref::<TensorRtComputeContext>()
                .ok_or_else(|| NeuralNetError("Wrong compute context type".to_string()))?;

            let gpu = if gpu_idx_for_this_thread >= 0 {
                gpu_idx_for_this_thread
            } else {
                // -1 means "auto": pick the first valid configured device,
                // falling back to device 0.
                trt_ctx
                    .gpu_idxs
                    .iter()
                    .copied()
                    .find(|&g| g >= 0)
                    .unwrap_or(0)
            };
            let engine_info = trt_ctx.engine_info;

            let trt_handle = TrtContext::new(&trt_ctx.engine, gpu)
                .map_err(|e| NeuralNetError(format!("TensorRT context creation failed: {e}")))?;

            let (bufs, generic_bufs) = if let Some(gio) = &trt_ctx.generic {
                (None, Some(GenericHostBufs::new(gio, max_batch_size)))
            } else {
                let bufs = TrtBuffers::new(&trt_ctx.engine, max_batch_size)
                    .map_err(|e| NeuralNetError(format!("TensorRT buffers creation failed: {e}")))?;
                (Some(bufs), None)
            };

            Ok(Box::new(TensorRtComputeHandle {
                ctx: trt_handle,
                engine_info,
                max_batch_size,
                inputs_use_nhwc: inputs_use_nhwc,
                bufs: Mutex::new(bufs),
                generic: trt_ctx.generic.clone(),
                generic_bufs: Mutex::new(generic_bufs),
            }))
        }

        fn is_using_fp16(&self, handle: &dyn ComputeHandle) -> bool {
            handle.is_using_fp16()
        }

        fn set_is_warmup(&self, handle: &mut dyn ComputeHandle, is_warmup: bool) -> bool {
            handle.set_is_warmup(is_warmup)
        }

        fn create_input_buffers(
            &self,
            _loaded_model: &dyn LoadedModel,
            _max_batch_size: i32,
            _nn_x_len: i32,
            _nn_y_len: i32,
        ) -> Result<Box<dyn InputBuffers>, NeuralNetError> {
            Ok(Box::new(TensorRtInputBuffers))
        }

        fn get_output(
            &self,
            handle: &dyn ComputeHandle,
            _buffers: &dyn InputBuffers,
            num_batch_elts: i32,
            input_bufs: &mut [&mut NNResultBuf],
            outputs: &mut [&mut NNOutput],
        ) -> Result<(), NeuralNetError> {
            let h = handle
                .as_any()
                .downcast_ref::<TensorRtComputeHandle>()
                .ok_or_else(|| NeuralNetError("Wrong compute handle type".to_string()))?;

            let n = num_batch_elts as usize;
            if n == 0 {
                return Ok(());
            }
            if n > h.max_batch_size as usize {
                return Err(NeuralNetError(format!(
                    "batch size {n} exceeds handle max {max_batch}",
                    max_batch = h.max_batch_size
                )));
            }

            // Generic directly-loaded .onnx path.
            if h.generic.is_some() {
                return generic_get_output(h, n, input_bufs, outputs);
            }
            let info = h.engine_info;
            let nn_x_len = info.nn_x_len;
            let nn_y_len = info.nn_y_len;
            let num_spatial_channels = info.num_input_channels;
            let _num_global_channels = info.num_input_global_channels;
            let num_policy_channels = info.num_policy_channels;
            let num_value_channels = info.num_value_channels;
            let _num_score_value_channels = info.num_score_value_channels;

            let lock = h.bufs.lock().unwrap();
            let lock = lock
                .as_ref()
                .ok_or_else(|| NeuralNetError("name-based buffers not initialized".to_string()))?;

            // --- Fill input buffers -----------------------------------------------
            // TRT always uses NCHW layout.
            let bufs_ptr = lock.ptr;

            let single_spatial_elts =
                unsafe { trt_ffi::katago_trt_buffers_single_spatial_elts(bufs_ptr) } as usize;
            let single_global_elts =
                unsafe { trt_ffi::katago_trt_buffers_single_global_elts(bufs_ptr) } as usize;
            let single_mask_elts =
                unsafe { trt_ffi::katago_trt_buffers_single_mask_elts(bufs_ptr) } as usize;

            let spatial_ptr = unsafe { trt_ffi::katago_trt_buffers_input_spatial(bufs_ptr) };
            let global_ptr = unsafe { trt_ffi::katago_trt_buffers_input_global(bufs_ptr) };
            let mask_ptr = unsafe { trt_ffi::katago_trt_buffers_input_mask(bufs_ptr) };

            if spatial_ptr.is_null() || global_ptr.is_null() || mask_ptr.is_null() {
                return Err(NeuralNetError(
                    "TensorRT input buffer pointers are null".into(),
                ));
            }

            let spatial_slice =
                unsafe { std::slice::from_raw_parts_mut(spatial_ptr, n * single_spatial_elts) };
            let global_slice =
                unsafe { std::slice::from_raw_parts_mut(global_ptr, n * single_global_elts) };
            let mask_slice =
                unsafe { std::slice::from_raw_parts_mut(mask_ptr, n * single_mask_elts) };

            for i in 0..n {
                let sym_idx = input_bufs[i].symmetry;
                let spatial_offset = i * single_spatial_elts;
                let global_offset = i * single_global_elts;
                let mask_offset = i * single_mask_elts;

                // Copy spatial features with symmetry (layout = evaluator buffers).
                copy_inputs_with_symmetry(
                    &input_bufs[i].row_spatial_buf,
                    &mut spatial_slice[spatial_offset..spatial_offset + single_spatial_elts],
                    1,
                    nn_y_len,
                    nn_x_len,
                    num_spatial_channels,
                    h.inputs_use_nhwc,
                    sym_idx,
                );

                // Copy global features (no symmetry).
                let gb = &input_bufs[i].row_global_buf;
                let copy_len = single_global_elts.min(gb.len());
                global_slice[global_offset..global_offset + copy_len]
                    .copy_from_slice(&gb[..copy_len]);

                // Mask = first channel of spatial input (already symmetry-applied).
                // After copy_inputs_with_symmetry, the first `single_mask_elts` values
                // in the spatial slice correspond to the first channel.
                // For NCHW: channel 0 occupies positions [0 .. mask_elts).
                let src_mask_start = spatial_offset;
                mask_slice[mask_offset..mask_offset + single_mask_elts].copy_from_slice(
                    &spatial_slice[src_mask_start..src_mask_start + single_mask_elts],
                );
            }

            // --- Inference ----------------------------------------------------------
            let ok = unsafe { trt_ffi::katago_trt_infer(h.ctx.ptr, bufs_ptr, num_batch_elts) };
            if ok == 0 {
                return Err(NeuralNetError(
                    trt_ffi::last_error().unwrap_or_else(|| "TensorRT inference failed".into()),
                ));
            }

            // --- Decode outputs -----------------------------------------------------
            let bufs_cptr = bufs_ptr as *const trt_ffi::KatagoTrtBuffers;

            let single_policy_elts =
                unsafe { trt_ffi::katago_trt_buffers_single_policy_elts(bufs_cptr) } as usize;
            let single_policy_pass_elts =
                unsafe { trt_ffi::katago_trt_buffers_single_policy_pass_elts(bufs_cptr) } as usize;
            let single_value_elts =
                unsafe { trt_ffi::katago_trt_buffers_single_value_elts(bufs_cptr) } as usize;
            let single_score_value_elts =
                unsafe { trt_ffi::katago_trt_buffers_single_score_value_elts(bufs_cptr) } as usize;
            let single_ownership_elts =
                unsafe { trt_ffi::katago_trt_buffers_single_ownership_elts(bufs_cptr) } as usize;

            let policy_ptr = unsafe { trt_ffi::katago_trt_buffers_output_policy(bufs_cptr) };
            let policy_pass_ptr =
                unsafe { trt_ffi::katago_trt_buffers_output_policy_pass(bufs_cptr) };
            let value_ptr = unsafe { trt_ffi::katago_trt_buffers_output_value(bufs_cptr) };
            let score_value_ptr =
                unsafe { trt_ffi::katago_trt_buffers_output_score_value(bufs_cptr) };
            let ownership_ptr = unsafe { trt_ffi::katago_trt_buffers_output_ownership(bufs_cptr) };

            if policy_ptr.is_null()
                || policy_pass_ptr.is_null()
                || value_ptr.is_null()
                || score_value_ptr.is_null()
            {
                return Err(NeuralNetError(
                    "TensorRT output buffer pointers are null".into(),
                ));
            }

            let policy_slice =
                unsafe { std::slice::from_raw_parts(policy_ptr, n * single_policy_elts) };
            let policy_pass_slice =
                unsafe { std::slice::from_raw_parts(policy_pass_ptr, n * single_policy_pass_elts) };
            let value_slice =
                unsafe { std::slice::from_raw_parts(value_ptr, n * single_value_elts) };
            let score_value_slice =
                unsafe { std::slice::from_raw_parts(score_value_ptr, n * single_score_value_elts) };
            let ownership_slice = if ownership_ptr.is_null() {
                &[][..]
            } else {
                unsafe { std::slice::from_raw_parts(ownership_ptr, n * single_ownership_elts) }
            };

            // Pre-allocate temporary buffers for symmetry output decoding.
            let mut tmp_policy = vec![0.0f32; single_policy_elts];
            let mut tmp_ownership = if !ownership_slice.is_empty() {
                vec![0.0f32; single_ownership_elts]
            } else {
                Vec::new()
            };

            let policy_area = (nn_x_len * nn_y_len) as usize;
            let area = policy_area; // ownership has same spatial dims

            for i in 0..n {
                let sym_idx = input_bufs[i].symmetry;
                let inv_sym = if sym_idx != 0 { invert(sym_idx) } else { 0 };

                // --- Policy (spatial logits) -----------------------------------------
                let p_off = i * single_policy_elts;
                // Apply inverse symmetry to the raw policy logits.
                if inv_sym != 0 {
                    copy_outputs_with_symmetry(
                        &policy_slice[p_off..p_off + single_policy_elts],
                        &mut tmp_policy,
                        1,
                        nn_y_len,
                        nn_x_len,
                        inv_sym,
                    );
                } else {
                    tmp_policy[..single_policy_elts]
                        .copy_from_slice(&policy_slice[p_off..p_off + single_policy_elts]);
                }

                // For plain-policy models (num_policy_channels == 1), copy directly.
                // For multi-channel models (optimism), blend channels.
                let step = policy_area;
                if num_policy_channels == 1 {
                    for pos in 0..policy_area {
                        outputs[i].policy_probs[pos] = tmp_policy[0 * step + pos];
                    }
                } else {
                    // Channel 0: base policy, channel 1: pass-aware policy.
                    let optimism = input_bufs[i].policy_optimism as f32;
                    for pos in 0..policy_area {
                        let base = tmp_policy[0 * step + pos];
                        let pass_aware = if num_policy_channels >= 2 {
                            tmp_policy[1 * step + pos]
                        } else {
                            0.0
                        };
                        outputs[i].policy_probs[pos] = base + optimism * pass_aware;
                    }
                }

                // --- Pass policy ----------------------------------------------------
                let pp_off = i * single_policy_pass_elts;
                let pass_step = single_policy_pass_elts / (num_policy_channels as usize).max(1);
                let pass_idx = policy_area; // pass position is at [nnXLen * nnYLen]
                if num_policy_channels == 1 {
                    outputs[i].policy_probs[pass_idx] = policy_pass_slice[pp_off];
                } else {
                    let base_pass = policy_pass_slice[pp_off];
                    let pass_aware_pass = if num_policy_channels >= 2 {
                        policy_pass_slice[pp_off + pass_step]
                    } else {
                        0.0
                    };
                    let optimism = input_bufs[i].policy_optimism as f32;
                    outputs[i].policy_probs[pass_idx] = base_pass + optimism * pass_aware_pass;
                }

                // --- Value ----------------------------------------------------------
                let v_off = i * single_value_elts;
                outputs[i].white_win_prob = value_slice[v_off];
                outputs[i].white_loss_prob = if num_value_channels >= 2 {
                    value_slice[v_off + 1]
                } else {
                    1.0 - value_slice[v_off]
                };
                outputs[i].white_no_result_prob = if num_value_channels >= 3 {
                    value_slice[v_off + 2]
                } else {
                    0.0
                };

                // --- Score value ----------------------------------------------------
                let sv_off = i * single_score_value_elts;
                let sv = &score_value_slice[sv_off..sv_off + single_score_value_elts];
                outputs[i].white_score_mean = sv.first().copied().unwrap_or(0.0);
                outputs[i].white_score_mean_sq = sv.get(1).copied().unwrap_or(0.0);
                outputs[i].white_lead = sv.get(2).copied().unwrap_or(0.0);
                outputs[i].var_time_left = sv.get(3).copied().unwrap_or(0.0);
                outputs[i].shortterm_winloss_error = sv.get(4).copied().unwrap_or(0.0);
                outputs[i].shortterm_score_error = sv.get(5).copied().unwrap_or(0.0);

                // --- Ownership ------------------------------------------------------
                if !ownership_slice.is_empty() && !tmp_ownership.is_empty() {
                    let o_off = i * single_ownership_elts;
                    if inv_sym != 0 {
                        copy_outputs_with_symmetry(
                            &ownership_slice[o_off..o_off + single_ownership_elts],
                            &mut tmp_ownership,
                            1,
                            nn_y_len,
                            nn_x_len,
                            inv_sym,
                        );
                    } else {
                        tmp_ownership[..single_ownership_elts].copy_from_slice(
                            &ownership_slice[o_off..o_off + single_ownership_elts],
                        );
                    }
                    outputs[i].white_owner_map =
                        Some(tmp_ownership[..area].to_vec().into_boxed_slice());
                }

                // --- Common fields --------------------------------------------------
                outputs[i].nn_x_len = nn_x_len;
                outputs[i].nn_y_len = nn_y_len;
                outputs[i].policy_optimism_used = input_bufs[i].policy_optimism as f32;
            }

            Ok(())
        }
    }

    /// Inference + decode for a directly loaded exported `.onnx` model
    /// (v15 head layout: `out_policy[6,362]`, `out_value[3]`, `out_miscvalue[10]`,
    /// `out_moremiscvalue[8]`, `out_ownership[1,19,19]`).
    fn generic_get_output(
        h: &TensorRtComputeHandle,
        n: usize,
        input_bufs: &mut [&mut NNResultBuf],
        outputs: &mut [&mut NNOutput],
    ) -> Result<(), NeuralNetError> {
        let gio = h.generic.as_ref().expect("generic path requires GenericIo");
        let nn_x_len = h.engine_info.nn_x_len;
        let nn_y_len = h.engine_info.nn_y_len;

        let idx_spatial = gio
            .input_index("input_spatial")
            .ok_or_else(|| NeuralNetError("missing input tensor 'input_spatial'".into()))?;
        let idx_global = gio
            .input_index("input_global")
            .ok_or_else(|| NeuralNetError("missing input tensor 'input_global'".into()))?;
        let idx_policy = gio
            .output_index("out_policy")
            .ok_or_else(|| NeuralNetError("missing output tensor 'out_policy'".into()))?;
        let idx_value = gio
            .output_index("out_value")
            .ok_or_else(|| NeuralNetError("missing output tensor 'out_value'".into()))?;
        let idx_misc = gio
            .output_index("out_miscvalue")
            .ok_or_else(|| NeuralNetError("missing output tensor 'out_miscvalue'".into()))?;
        let idx_moremisc = gio
            .output_index("out_moremiscvalue")
            .ok_or_else(|| NeuralNetError("missing output tensor 'out_moremiscvalue'".into()))?;
        let idx_ownership = gio
            .output_index("out_ownership")
            .ok_or_else(|| NeuralNetError("missing output tensor 'out_ownership'".into()))?;

        let mut gb = h.generic_bufs.lock().unwrap();
        let bufs = gb.as_mut().expect("generic buffers must be initialized");

        let single_spatial = GenericIo::single_elts(&gio.input_dims[idx_spatial]);
        let single_global = GenericIo::single_elts(&gio.input_dims[idx_global]);

        // Fill inputs (NCHW spatial + global, with symmetry applied to spatial).
        for i in 0..n {
            let sym_idx = input_bufs[i].symmetry;
            let sp_off = i * single_spatial;
            copy_inputs_with_symmetry(
                &input_bufs[i].row_spatial_buf,
                &mut bufs.inputs[idx_spatial][sp_off..sp_off + single_spatial],
                1,
                nn_y_len,
                nn_x_len,
                h.engine_info.num_input_channels,
                h.inputs_use_nhwc,
                sym_idx,
            );
            let gl_off = i * single_global;
            let gb_row = &input_bufs[i].row_global_buf;
            let copy_len = single_global.min(gb_row.len());
            bufs.inputs[idx_global][gl_off..gl_off + copy_len]
                .copy_from_slice(&gb_row[..copy_len]);
        }

        // Infer.
        let in_ptrs: Vec<*const f32> = bufs.inputs.iter().map(|v| v.as_ptr()).collect();
        let out_ptrs: Vec<*mut f32> = bufs.outputs.iter_mut().map(|v| v.as_mut_ptr()).collect();

        // Debug dump (KATAGO_DEBUG_DUMP set): write the first batch element's
        // inputs and raw outputs for offline inspection.
        if let Ok(d) = std::env::var("KATAGO_DEBUG_DUMP") {
            let w = |name: &str, data: &[f32]| {
                let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
                let _ = std::fs::write(format!("{d}/{name}.bin"), bytes);
            };
            let _ = std::fs::create_dir_all(&d);
            w("dbg_spatial", &bufs.inputs[idx_spatial][..single_spatial]);
            w("dbg_global", &bufs.inputs[idx_global][..single_global]);
        }

        let ok = unsafe {
            trt_ffi::katago_trt_infer_generic(
                h.ctx.ptr,
                n as i32,
                in_ptrs.as_ptr(),
                out_ptrs.as_ptr(),
            )
        };
        if ok == 0 {
            return Err(NeuralNetError(
                trt_ffi::last_error().unwrap_or_else(|| "generic inference failed".into()),
            ));
        }

        if let Ok(d) = std::env::var("KATAGO_DEBUG_DUMP") {
            let w = |name: &str, data: &[f32]| {
                let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
                let _ = std::fs::write(format!("{d}/{name}.bin"), bytes);
            };
            let sp = GenericIo::single_elts(&gio.output_dims[idx_policy]);
            let sv = GenericIo::single_elts(&gio.output_dims[idx_value]);
            let sm = GenericIo::single_elts(&gio.output_dims[idx_misc]);
            let smm = GenericIo::single_elts(&gio.output_dims[idx_moremisc]);
            let so = GenericIo::single_elts(&gio.output_dims[idx_ownership]);
            w("dbg_policy", &bufs.outputs[idx_policy][..sp]);
            w("dbg_value", &bufs.outputs[idx_value][..sv]);
            w("dbg_misc", &bufs.outputs[idx_misc][..sm]);
            w("dbg_moremisc", &bufs.outputs[idx_moremisc][..smm]);
            w("dbg_ownership", &bufs.outputs[idx_ownership][..so]);
        }

        // Decode v15 outputs.
        let policy_area = (nn_x_len * nn_y_len) as usize;
        let single_policy = GenericIo::single_elts(&gio.output_dims[idx_policy]); // 6*362
        let policy_channels = gio.output_dims[idx_policy].get(1).copied().unwrap_or(6) as usize;
        let single_value = GenericIo::single_elts(&gio.output_dims[idx_value]);
        let single_misc = GenericIo::single_elts(&gio.output_dims[idx_misc]);
        let single_moremisc = GenericIo::single_elts(&gio.output_dims[idx_moremisc]);
        let single_ownership = GenericIo::single_elts(&gio.output_dims[idx_ownership]);

        let mut tmp_policy_base = vec![0.0f32; policy_area];
        let mut tmp_policy_opt = vec![0.0f32; policy_area];
        let mut tmp_ownership = vec![0.0f32; single_ownership];

        for i in 0..n {
            let sym_idx = input_bufs[i].symmetry;
            let inv_sym = if sym_idx != 0 { invert(sym_idx) } else { 0 };

            // --- Policy ----------------------------------------------------
            let p_off = i * single_policy;
            let base_src = &bufs.outputs[idx_policy][p_off..p_off + policy_area];
            // Short-term-optimistic logits are channel 5 of the exported head.
            let opt_ch = 5usize.min(policy_channels.saturating_sub(1));
            let opt_src = &bufs.outputs[idx_policy]
                [p_off + opt_ch * policy_area..p_off + (opt_ch + 1) * policy_area];
            if inv_sym != 0 {
                copy_outputs_with_symmetry(
                    base_src,
                    &mut tmp_policy_base,
                    1,
                    nn_y_len,
                    nn_x_len,
                    inv_sym,
                );
                copy_outputs_with_symmetry(
                    opt_src,
                    &mut tmp_policy_opt,
                    1,
                    nn_y_len,
                    nn_x_len,
                    inv_sym,
                );
            } else {
                tmp_policy_base.copy_from_slice(base_src);
                tmp_policy_opt.copy_from_slice(opt_src);
            }
            let optimism = input_bufs[i].policy_optimism as f32;
            for pos in 0..policy_area {
                let base = tmp_policy_base[pos];
                let opt = tmp_policy_opt[pos];
                outputs[i].policy_probs[pos] = base + (opt - base) * optimism;
            }
            // Pass logit is at index policy_area within each channel.
            let base_pass = bufs.outputs[idx_policy][p_off + policy_area];
            let opt_pass = bufs.outputs[idx_policy][p_off + opt_ch * policy_area + policy_area];
            outputs[i].policy_probs[policy_area] = base_pass + (opt_pass - base_pass) * optimism;

            // --- Value -----------------------------------------------------
            let v_off = i * single_value;
            outputs[i].white_win_prob = bufs.outputs[idx_value][v_off];
            outputs[i].white_loss_prob = bufs.outputs[idx_value][v_off + 1];
            outputs[i].white_no_result_prob = bufs.outputs[idx_value][v_off + 2];

            // --- Score value -------------------------------------------------
            let m_off = i * single_misc;
            outputs[i].white_score_mean = bufs.outputs[idx_misc][m_off];
            outputs[i].white_score_mean_sq = bufs.outputs[idx_misc][m_off + 1];
            outputs[i].white_lead = bufs.outputs[idx_misc][m_off + 2];
            outputs[i].var_time_left = bufs.outputs[idx_misc][m_off + 3];
            let mm_off = i * single_moremisc;
            outputs[i].shortterm_winloss_error = bufs.outputs[idx_moremisc][mm_off];
            outputs[i].shortterm_score_error = bufs.outputs[idx_moremisc][mm_off + 1];

            // --- Ownership ---------------------------------------------------
            if input_bufs[i].include_owner_map {
                let o_off = i * single_ownership;
                let src = &bufs.outputs[idx_ownership][o_off..o_off + single_ownership];
                if inv_sym != 0 {
                    copy_outputs_with_symmetry(
                        src,
                        &mut tmp_ownership,
                        1,
                        nn_y_len,
                        nn_x_len,
                        inv_sym,
                    );
                } else {
                    tmp_ownership.copy_from_slice(src);
                }
                outputs[i].white_owner_map = Some(tmp_ownership.clone().into_boxed_slice());
            }

            outputs[i].nn_x_len = nn_x_len;
            outputs[i].nn_y_len = nn_y_len;
            outputs[i].policy_optimism_used = input_bufs[i].policy_optimism as f32;
        }

        Ok(())
    }
}

// ==========================================================================
// Module: shim NOT compiled (CUDA/TensorRT SDK missing)
// ==========================================================================

#[cfg(all(feature = "trt", not(trt_shim_available)))]
mod imp {
    use super::*;

    pub struct TensorRtLoadedModel {
        model_desc: ModelDesc,
    }
    impl TensorRtLoadedModel {
        pub fn new() -> Self {
            Self {
                model_desc: ModelDesc::default(),
            }
        }
    }
    impl LoadedModel for TensorRtLoadedModel {
        fn model_desc(&self) -> &ModelDesc {
            &self.model_desc
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    pub struct TensorRtComputeContext;
    impl ComputeContext for TensorRtComputeContext {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    pub struct TensorRtComputeHandle;
    impl ComputeHandle for TensorRtComputeHandle {
        fn is_using_fp16(&self) -> bool {
            false
        }
        fn set_is_warmup(&mut self, _: bool) -> bool {
            false
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    pub struct TensorRtInputBuffers;
    impl InputBuffers for TensorRtInputBuffers {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    pub struct TensorRtBackend;

    impl Backend for TensorRtBackend {
        fn global_initialize(&self) {}
        fn global_cleanup(&self) {}
        fn print_devices(&self) {
            println!(
                "TensorRT backend selected, but C++ shim not compiled (CUDA/TensorRT SDK not found)"
            );
        }

        fn load_model_file(
            &self,
            _f: &str,
            _s: &str,
        ) -> Result<Box<dyn LoadedModel>, NeuralNetError> {
            Err(NeuralNetError(
                "TensorRT C++ shim not compiled (CUDA/TensorRT SDK not found at build time)".into(),
            ))
        }
        fn create_compute_context(
            &self,
            _g: &[i32],
            _l: &Logger,
            _x: i32,
            _y: i32,
            _h: &str,
            _u: Enabled,
            _m: &dyn LoadedModel,
            _c: &Config,
        ) -> Result<Box<dyn ComputeContext>, NeuralNetError> {
            Err(NeuralNetError("TensorRT C++ shim not compiled".into()))
        }
        fn create_compute_handle(
            &self,
            _c: &dyn ComputeContext,
            _m: &dyn LoadedModel,
            _l: &Logger,
            _b: i32,
            _r: bool,
            _n: bool,
            _g: i32,
            _s: i32,
        ) -> Result<Box<dyn ComputeHandle>, NeuralNetError> {
            Err(NeuralNetError("TensorRT C++ shim not compiled".into()))
        }
        fn is_using_fp16(&self, _h: &dyn ComputeHandle) -> bool {
            false
        }
        fn set_is_warmup(&self, _h: &mut dyn ComputeHandle, _w: bool) -> bool {
            false
        }
        fn create_input_buffers(
            &self,
            _m: &dyn LoadedModel,
            _b: i32,
            _x: i32,
            _y: i32,
        ) -> Result<Box<dyn InputBuffers>, NeuralNetError> {
            Err(NeuralNetError("TensorRT C++ shim not compiled".into()))
        }
        fn get_output(
            &self,
            _h: &dyn ComputeHandle,
            _b: &dyn InputBuffers,
            _n: i32,
            _i: &mut [&mut NNResultBuf],
            _o: &mut [&mut NNOutput],
        ) -> Result<(), NeuralNetError> {
            Err(NeuralNetError("TensorRT C++ shim not compiled".into()))
        }
    }
}

// ==========================================================================
// Module: `trt` feature not enabled
// ==========================================================================

#[cfg(not(feature = "trt"))]
#[allow(dead_code)]
mod imp {
    use super::*;

    pub struct TensorRtLoadedModel {
        model_desc: ModelDesc,
    }
    impl TensorRtLoadedModel {
        pub fn new() -> Self {
            Self {
                model_desc: ModelDesc::default(),
            }
        }
    }
    impl LoadedModel for TensorRtLoadedModel {
        fn model_desc(&self) -> &ModelDesc {
            &self.model_desc
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    pub struct TensorRtComputeContext;
    impl ComputeContext for TensorRtComputeContext {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    pub struct TensorRtComputeHandle;
    impl ComputeHandle for TensorRtComputeHandle {
        fn is_using_fp16(&self) -> bool {
            false
        }
        fn set_is_warmup(&mut self, _: bool) -> bool {
            false
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    pub struct TensorRtInputBuffers;
    impl InputBuffers for TensorRtInputBuffers {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    pub struct TensorRtBackend;

    impl Backend for TensorRtBackend {
        fn global_initialize(&self) {}
        fn global_cleanup(&self) {}
        fn print_devices(&self) {
            println!("TensorRT backend selected, but compiled without the 'trt' feature");
        }

        fn load_model_file(
            &self,
            _f: &str,
            _s: &str,
        ) -> Result<Box<dyn LoadedModel>, NeuralNetError> {
            Err(NeuralNetError(
                "TensorRT support not enabled (enable the 'trt' Cargo feature)".into(),
            ))
        }
        fn create_compute_context(
            &self,
            _g: &[i32],
            _l: &Logger,
            _x: i32,
            _y: i32,
            _h: &str,
            _u: Enabled,
            _m: &dyn LoadedModel,
            _c: &Config,
        ) -> Result<Box<dyn ComputeContext>, NeuralNetError> {
            Err(NeuralNetError(
                "TensorRT support not enabled (enable the 'trt' feature)".into(),
            ))
        }
        fn create_compute_handle(
            &self,
            _c: &dyn ComputeContext,
            _m: &dyn LoadedModel,
            _l: &Logger,
            _b: i32,
            _r: bool,
            _n: bool,
            _g: i32,
            _s: i32,
        ) -> Result<Box<dyn ComputeHandle>, NeuralNetError> {
            Err(NeuralNetError(
                "TensorRT support not enabled (enable the 'trt' feature)".into(),
            ))
        }
        fn is_using_fp16(&self, _h: &dyn ComputeHandle) -> bool {
            false
        }
        fn set_is_warmup(&self, _h: &mut dyn ComputeHandle, _w: bool) -> bool {
            false
        }
        fn create_input_buffers(
            &self,
            _m: &dyn LoadedModel,
            _b: i32,
            _x: i32,
            _y: i32,
        ) -> Result<Box<dyn InputBuffers>, NeuralNetError> {
            Err(NeuralNetError(
                "TensorRT support not enabled (enable the 'trt' feature)".into(),
            ))
        }
        fn get_output(
            &self,
            _h: &dyn ComputeHandle,
            _b: &dyn InputBuffers,
            _n: i32,
            _i: &mut [&mut NNResultBuf],
            _o: &mut [&mut NNOutput],
        ) -> Result<(), NeuralNetError> {
            Err(NeuralNetError(
                "TensorRT support not enabled (enable the 'trt' feature)".into(),
            ))
        }
    }
}

pub use imp::TensorRtBackend;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trt_backend_lifecycle_stub() {
        let backend = TensorRtBackend;
        backend.global_initialize();
        backend.print_devices();
        backend.global_cleanup();
    }

    #[test]
    #[cfg(not(trt_shim_available))]
    fn test_trt_backend_load_model_file_error() {
        let backend = TensorRtBackend;
        let result = backend.load_model_file("model.bin", "");
        assert!(
            result.is_err(),
            "load_model_file should produce an error without an ONNX model"
        );
    }
}
