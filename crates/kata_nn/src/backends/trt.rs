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
        bufs: Mutex<TrtBuffers>,
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
            Ok(Box::new(TensorRtLoadedModel {
                model_path: file.to_string(),
                model_desc: ModelDesc::default(),
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
            _cfg: &Config,
        ) -> Result<Box<dyn ComputeContext>, NeuralNetError> {
            let model = loaded_model
                .as_any()
                .downcast_ref::<TensorRtLoadedModel>()
                .ok_or_else(|| NeuralNetError("Wrong loaded model type".to_string()))?;

            let max_batch_size: i32 = 8; // TODO: from config
            let use_fp16 = !matches!(use_fp16_mode, Enabled::False);

            // 1. Load the model file → ModelDesc
            let model_desc = crate::model_parser::load_model_file(&model.model_path)
                .map_err(|e| NeuralNetError(format!("model load failed: {e}")))?;

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
            }))
        }

        fn create_compute_handle(
            &self,
            ctx: &dyn ComputeContext,
            _loaded_model: &dyn LoadedModel,
            _logger: &Logger,
            max_batch_size: i32,
            _require_exact_nn_len: bool,
            _inputs_use_nhwc: bool,
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
                trt_ctx.gpu_idxs.first().copied().unwrap_or(0)
            };
            let engine_info = trt_ctx.engine_info;

            let trt_handle = TrtContext::new(&trt_ctx.engine, gpu)
                .map_err(|e| NeuralNetError(format!("TensorRT context creation failed: {e}")))?;
            let bufs = TrtBuffers::new(&trt_ctx.engine, max_batch_size)
                .map_err(|e| NeuralNetError(format!("TensorRT buffers creation failed: {e}")))?;

            Ok(Box::new(TensorRtComputeHandle {
                ctx: trt_handle,
                engine_info,
                max_batch_size,
                bufs: Mutex::new(bufs),
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

            let info = h.engine_info;
            let nn_x_len = info.nn_x_len;
            let nn_y_len = info.nn_y_len;
            let num_spatial_channels = info.num_input_channels;
            let _num_global_channels = info.num_input_global_channels;
            let num_policy_channels = info.num_policy_channels;
            let num_value_channels = info.num_value_channels;
            let _num_score_value_channels = info.num_score_value_channels;

            let lock = h.bufs.lock().unwrap();

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

                // Copy spatial features with symmetry (NCHW, c=h.stack, h,w per channel).
                copy_inputs_with_symmetry(
                    &input_bufs[i].row_spatial_buf,
                    &mut spatial_slice[spatial_offset..spatial_offset + single_spatial_elts],
                    1,
                    nn_y_len,
                    nn_x_len,
                    num_spatial_channels,
                    false, // NCHW
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

                input_bufs[i].has_result = true;
            }

            Ok(())
        }
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
