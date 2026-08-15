//! Generic neural-net backend interface.
//!
//! Corresponds to `cpp/neuralnet/nninterface.h`.
//! This module defines the object-safe traits and data structures that abstract
//! neural-network inference backends (CUDA, OpenCL, Eigen, dummy, etc.).

#![allow(clippy::too_many_arguments)]

use std::any::Any;
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use kata_core::config::Config;
use kata_core::logger::Logger;

use crate::desc::ModelDesc;
pub use crate::inputs::NNOutput;

// ---------------------------------------------------------------------------
// enabled_t
// ---------------------------------------------------------------------------

/// Tri-state value for backend options that can be enabled, disabled, or left
/// on auto.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Enabled {
    False,
    True,
    #[default]
    Auto,
}

impl Enabled {
    /// Parse an enabled value, accepting the same spellings as the C++
    /// `enabled_t::tryParse`.
    pub fn try_parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "1" | "t" | "true" | "enabled" | "y" | "yes" => Some(Enabled::True),
            "0" | "f" | "false" | "disabled" | "n" | "no" => Some(Enabled::False),
            "auto" => Some(Enabled::Auto),
            _ => None,
        }
    }
}

impl fmt::Display for Enabled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Enabled::True => write!(f, "true"),
            Enabled::False => write!(f, "false"),
            Enabled::Auto => write!(f, "auto"),
        }
    }
}

impl FromStr for Enabled {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::try_parse(s).ok_or(())
    }
}

// ---------------------------------------------------------------------------
// Marker traits
// ---------------------------------------------------------------------------

/// A handle to a loaded neural-network model.
///
/// Mirrors C++ `LoadedModel`.
pub trait LoadedModel: Send {
    /// Returns the model descriptor.
    fn model_desc(&self) -> &ModelDesc;

    /// Enable downcasting to the concrete implementor.
    fn as_any(&self) -> &dyn Any;
}

/// A handle to cross-thread cross-GPU initialization state.
///
/// Mirrors C++ `ComputeContext`.
pub trait ComputeContext: Send {
    /// Enable downcasting to the concrete implementor.
    fn as_any(&self) -> &dyn Any;
}

/// A handle to the local compute backend. Not thread-safe; each handle should
/// only be used by one thread.
///
/// Mirrors C++ `ComputeHandle`.
pub trait ComputeHandle: Send {
    /// Returns true if the handle evaluates in FP16.
    fn is_using_fp16(&self) -> bool;

    /// Sets whether the handle is currently being used in warmup mode, returning
    /// the previous value.
    fn set_is_warmup(&mut self, is_warmup: bool) -> bool;

    /// Enable downcasting to the concrete implementor.
    fn as_any(&self) -> &dyn Any;
}

/// Input buffers used to pass data into the neural network for computation.
///
/// Mirrors C++ `InputBuffers`.
pub trait InputBuffers: Send + Sync {
    /// Enable downcasting to the concrete implementor.
    fn as_any(&self) -> &dyn Any;
}

// ---------------------------------------------------------------------------
// NNResultBuf
// ---------------------------------------------------------------------------

/// Buffer that holds input rows for a neural-net evaluation and, once the
/// evaluation completes, the resulting output.
///
/// Mirrors the `NNResultBuf` declared in C++ `nneval.h`.
#[derive(Debug, Clone)]
pub struct NNResultBuf {
    pub client_waiting_for_result: bool,
    pub has_result: bool,
    pub include_owner_map: bool,
    pub board_x_size_for_server: i32,
    pub board_y_size_for_server: i32,
    pub row_spatial_buf: Vec<f32>,
    pub row_global_buf: Vec<f32>,
    pub row_meta_buf: Vec<f32>,
    pub has_row_meta: bool,
    pub result: Option<Arc<NNOutput>>,
    pub error_log_lockout: bool,
    pub symmetry: i32,
    pub policy_optimism: f64,
}

impl NNResultBuf {
    /// Creates a new result buffer with default values.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for NNResultBuf {
    fn default() -> Self {
        Self {
            client_waiting_for_result: false,
            has_result: false,
            include_owner_map: false,
            board_x_size_for_server: 0,
            board_y_size_for_server: 0,
            row_spatial_buf: Vec::new(),
            row_global_buf: Vec::new(),
            row_meta_buf: Vec::new(),
            has_row_meta: false,
            result: None,
            error_log_lockout: false,
            symmetry: 0,
            policy_optimism: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// NeuralNetError
// ---------------------------------------------------------------------------

/// Error type for neural-network backend operations.
#[derive(Debug, Clone)]
pub struct NeuralNetError(pub String);

impl fmt::Display for NeuralNetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NeuralNetError: {}", self.0)
    }
}

impl std::error::Error for NeuralNetError {}

// ---------------------------------------------------------------------------
// Backend trait
// ---------------------------------------------------------------------------

/// Generic interface to neural-net inference.
///
/// Mirrors the functions in the C++ `NeuralNet` namespace. All methods take
/// `&self` so that the trait remains object-safe and can be used as
/// `Box<dyn Backend>`.
pub trait Backend: Send + Sync {
    /// Called once upon program startup to initialize global backend state.
    fn global_initialize(&self);

    /// Called at program termination to clean up global backend state.
    fn global_cleanup(&self);

    /// Print available backend devices.
    fn print_devices(&self);

    /// Load a model file and return a handle to it.
    fn load_model_file(
        &self,
        file: &str,
        expected_sha256: &str,
    ) -> Result<Box<dyn LoadedModel>, NeuralNetError>;

    /// Create a compute context shared across threads and GPUs.
    fn create_compute_context(
        &self,
        gpu_idxs: &[i32],
        logger: &Logger,
        nn_x_len: i32,
        nn_y_len: i32,
        home_data_dir_override: &str,
        use_fp16_mode: Enabled,
        loaded_model: &dyn LoadedModel,
        cfg: &Config,
    ) -> Result<Box<dyn ComputeContext>, NeuralNetError>;

    /// Create a per-thread compute handle.
    fn create_compute_handle(
        &self,
        ctx: &dyn ComputeContext,
        loaded_model: &dyn LoadedModel,
        logger: &Logger,
        max_batch_size: i32,
        require_exact_nn_len: bool,
        inputs_use_nhwc: bool,
        gpu_idx_for_this_thread: i32,
        server_thread_idx: i32,
    ) -> Result<Box<dyn ComputeHandle>, NeuralNetError>;

    /// Returns true if the handle evaluates in FP16.
    fn is_using_fp16(&self, handle: &dyn ComputeHandle) -> bool;

    /// Sets whether the handle is currently being used in warmup mode, returning
    /// the previous value.
    fn set_is_warmup(&self, handle: &mut dyn ComputeHandle, is_warmup: bool) -> bool;

    /// Create input buffers for batched neural-net evaluation.
    fn create_input_buffers(
        &self,
        loaded_model: &dyn LoadedModel,
        max_batch_size: i32,
        nn_x_len: i32,
        nn_y_len: i32,
    ) -> Result<Box<dyn InputBuffers>, NeuralNetError>;

    /// Perform a batched neural-net evaluation.
    ///
    /// Preconditions: `input_bufs[n]->{row_spatial_buf,row_global_buf}` have been
    /// filled with input data for all `n` in `[0, num_batch_elts)`. `outputs` has
    /// length `num_batch_elts` containing allocated but possibly-uninitialized
    /// `NNOutput` structs.
    ///
    /// Result: mutably writes the results of the parallel neural-net evaluations
    /// into the `NNOutput` structs. All outputs are in logits; final activation
    /// functions are not applied.
    fn get_output(
        &self,
        handle: &dyn ComputeHandle,
        buffers: &dyn InputBuffers,
        num_batch_elts: i32,
        input_bufs: &mut [&mut NNResultBuf],
        outputs: &mut [&mut NNOutput],
    ) -> Result<(), NeuralNetError>;

    // ASYNC PIPELINE (fork `cudaAsyncInferPipeline`) ----------------------------

    /// 是否支持事件门控异步流水线：submit 非阻塞提交 GPU 工作，finish 等待
    /// 完成事件并解码。serve 循环借此让"下一批的输入填充/上传提交"与
    /// "上一批的 GPU 执行"重叠，消除批间 GPU 空闲。
    /// 默认 false：后端只有同步 get_output。
    fn supports_async_pipeline(&self) -> bool {
        false
    }

    /// 非阻塞提交一批推理（填 pinned 输入 → 异步上传 → 启动前向 → 异步回传
    /// → record 完成事件），返回批次令牌（供 finish/query 使用）。
    /// 不会等待 GPU 完成；`input_bufs` 仅在本调用内被读取。
    ///
    /// 流水线契约：两次连续 submit 之间必须 finish 上一批，除非两批的
    /// `num_batch_elts` 相同（CUDA 后端的 graph/工作区/pinned 缓冲绑定固定
    /// batch 尺寸；同尺寸连交可重叠，异尺寸须先收尾触发重建）。
    /// 默认实现：什么都不做（配合默认 finish_output 退化为同步）。
    fn submit_output(
        &self,
        _handle: &dyn ComputeHandle,
        _buffers: &dyn InputBuffers,
        _num_batch_elts: i32,
        _input_bufs: &mut [&mut NNResultBuf],
    ) -> Result<usize, NeuralNetError> {
        Ok(0)
    }

    /// 非阻塞查询令牌对应批次是否已在 GPU 完成（完成事件已触发）。
    /// 默认实现：总是就绪（同步语义）。
    fn query_output_done(&self, _handle: &dyn ComputeHandle, _token: usize) -> bool {
        true
    }

    /// 等待令牌对应批次完成并解码到 `outputs`（阻塞 host 直至完成事件触发）。
    /// 默认实现：退化为同步 get_output（忽略令牌）。
    fn finish_output(
        &self,
        handle: &dyn ComputeHandle,
        buffers: &dyn InputBuffers,
        _token: usize,
        num_batch_elts: i32,
        input_bufs: &mut [&mut NNResultBuf],
        outputs: &mut [&mut NNOutput],
    ) -> Result<(), NeuralNetError> {
        self.get_output(handle, buffers, num_batch_elts, input_bufs, outputs)
    }

    // FOR TESTING ----------------------------------------------------------------

    /// If implemented, evaluate a convolution layer on the input buffer.
    /// Returns false if not implemented by the backend.
    fn test_evaluate_conv(
        &self,
        _desc: &crate::desc::ConvLayerDesc,
        _batch_size: i32,
        _nn_x_len: i32,
        _nn_y_len: i32,
        _use_fp16: bool,
        _use_nhwc: bool,
        _input_buffer: &[f32],
        _output_buffer: &mut Vec<f32>,
    ) -> bool {
        false
    }

    /// If implemented, evaluate a batch-norm layer on the input buffer.
    fn test_evaluate_batch_norm(
        &self,
        _desc: &crate::desc::BatchNormLayerDesc,
        _batch_size: i32,
        _nn_x_len: i32,
        _nn_y_len: i32,
        _use_fp16: bool,
        _use_nhwc: bool,
        _input_buffer: &[f32],
        _mask_buffer: &[f32],
        _output_buffer: &mut Vec<f32>,
    ) -> bool {
        false
    }

    /// If implemented, evaluate a residual block on the input buffer.
    fn test_evaluate_residual_block(
        &self,
        _desc: &crate::desc::ResidualBlockDesc,
        _batch_size: i32,
        _nn_x_len: i32,
        _nn_y_len: i32,
        _use_fp16: bool,
        _use_nhwc: bool,
        _input_buffer: &[f32],
        _mask_buffer: &[f32],
        _output_buffer: &mut Vec<f32>,
    ) -> bool {
        false
    }

    /// If implemented, evaluate a global-pooling residual block on the input buffer.
    fn test_evaluate_global_pooling_residual_block(
        &self,
        _desc: &crate::desc::GlobalPoolingResidualBlockDesc,
        _batch_size: i32,
        _nn_x_len: i32,
        _nn_y_len: i32,
        _use_fp16: bool,
        _use_nhwc: bool,
        _input_buffer: &[f32],
        _mask_buffer: &[f32],
        _output_buffer: &mut Vec<f32>,
    ) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Dummy backend for tests
// ---------------------------------------------------------------------------

pub mod dummy {
    use super::*;

    /// A loaded model that wraps a default [`ModelDesc`] with a valid model
    /// version so that downstream code can derive the inputs version.
    pub struct DummyLoadedModel {
        model_desc: ModelDesc,
    }

    impl DummyLoadedModel {
        pub fn new() -> Self {
            Self {
                model_desc: ModelDesc {
                    model_version: 8,
                    ..ModelDesc::default()
                },
            }
        }
    }

    impl Default for DummyLoadedModel {
        fn default() -> Self {
            Self::new()
        }
    }

    impl LoadedModel for DummyLoadedModel {
        fn model_desc(&self) -> &ModelDesc {
            &self.model_desc
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// Trivial compute context.
    pub struct DummyComputeContext;

    impl ComputeContext for DummyComputeContext {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// Trivial compute handle.
    pub struct DummyComputeHandle {
        is_warmup: bool,
    }

    impl DummyComputeHandle {
        pub fn new() -> Self {
            Self { is_warmup: false }
        }
    }

    impl Default for DummyComputeHandle {
        fn default() -> Self {
            Self::new()
        }
    }

    impl ComputeHandle for DummyComputeHandle {
        fn is_using_fp16(&self) -> bool {
            false
        }

        fn set_is_warmup(&mut self, is_warmup: bool) -> bool {
            let prev = self.is_warmup;
            self.is_warmup = is_warmup;
            prev
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// Trivial input buffers.
    pub struct DummyInputBuffers;

    impl InputBuffers for DummyInputBuffers {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// A backend that produces trivial outputs for testing.
    pub struct DummyBackend;

    impl Backend for DummyBackend {
        fn global_initialize(&self) {}

        fn global_cleanup(&self) {}

        fn print_devices(&self) {}

        fn load_model_file(
            &self,
            _file: &str,
            _expected_sha256: &str,
        ) -> Result<Box<dyn LoadedModel>, NeuralNetError> {
            Ok(Box::new(DummyLoadedModel::new()))
        }

        fn create_compute_context(
            &self,
            _gpu_idxs: &[i32],
            _logger: &Logger,
            _nn_x_len: i32,
            _nn_y_len: i32,
            _home_data_dir_override: &str,
            _use_fp16_mode: Enabled,
            _loaded_model: &dyn LoadedModel,
            _cfg: &Config,
        ) -> Result<Box<dyn ComputeContext>, NeuralNetError> {
            Ok(Box::new(DummyComputeContext))
        }

        fn create_compute_handle(
            &self,
            _ctx: &dyn ComputeContext,
            _loaded_model: &dyn LoadedModel,
            _logger: &Logger,
            _max_batch_size: i32,
            _require_exact_nn_len: bool,
            _inputs_use_nhwc: bool,
            _gpu_idx_for_this_thread: i32,
            _server_thread_idx: i32,
        ) -> Result<Box<dyn ComputeHandle>, NeuralNetError> {
            Ok(Box::new(DummyComputeHandle::new()))
        }

        fn is_using_fp16(&self, _handle: &dyn ComputeHandle) -> bool {
            false
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
            Ok(Box::new(DummyInputBuffers))
        }

        fn get_output(
            &self,
            _handle: &dyn ComputeHandle,
            _buffers: &dyn InputBuffers,
            num_batch_elts: i32,
            input_bufs: &mut [&mut NNResultBuf],
            outputs: &mut [&mut NNOutput],
        ) -> Result<(), NeuralNetError> {
            let n = num_batch_elts as usize;
            for (input_buf, output) in input_bufs
                .iter_mut()
                .take(n)
                .zip(outputs.iter_mut().take(n))
            {
                **output = NNOutput::default();
                input_buf.has_result = true;
                input_buf.result = Some(Arc::new(NNOutput::default()));
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enabled_display() {
        assert_eq!(Enabled::True.to_string(), "true");
        assert_eq!(Enabled::False.to_string(), "false");
        assert_eq!(Enabled::Auto.to_string(), "auto");
    }

    #[test]
    fn test_enabled_parse() {
        assert_eq!("true".parse::<Enabled>(), Ok(Enabled::True));
        assert_eq!("TRUE".parse::<Enabled>(), Ok(Enabled::True));
        assert_eq!("t".parse::<Enabled>(), Ok(Enabled::True));
        assert_eq!("yes".parse::<Enabled>(), Ok(Enabled::True));

        assert_eq!("false".parse::<Enabled>(), Ok(Enabled::False));
        assert_eq!("FALSE".parse::<Enabled>(), Ok(Enabled::False));
        assert_eq!("f".parse::<Enabled>(), Ok(Enabled::False));
        assert_eq!("no".parse::<Enabled>(), Ok(Enabled::False));

        assert_eq!("auto".parse::<Enabled>(), Ok(Enabled::Auto));
        assert_eq!("AUTO".parse::<Enabled>(), Ok(Enabled::Auto));

        assert_eq!("maybe".parse::<Enabled>(), Err(()));
    }

    #[test]
    fn test_enabled_try_parse() {
        assert_eq!(Enabled::try_parse("  enabled  "), Some(Enabled::True));
        assert_eq!(Enabled::try_parse("0"), Some(Enabled::False));
        assert_eq!(Enabled::try_parse("unknown"), None);
    }

    #[test]
    fn test_enabled_default() {
        assert_eq!(Enabled::default(), Enabled::Auto);
    }

    #[test]
    fn test_dummy_backend_lifecycle() {
        use dummy::DummyBackend;

        let backend = DummyBackend;
        backend.global_initialize();

        let model = backend
            .load_model_file("dummy.bin", "")
            .expect("load_model_file should succeed");
        assert_eq!(model.model_desc().get_num_parameters(), 0);

        let logger = Logger::new(kata_core::logger::LoggerOptions::default(), None);
        let ctx = backend
            .create_compute_context(
                &[],
                &logger,
                19,
                19,
                "",
                Enabled::Auto,
                &*model,
                &Config::new(false, true),
            )
            .expect("create_compute_context should succeed");

        let mut handle = backend
            .create_compute_handle(&*ctx, &*model, &logger, 1, false, false, 0, 0)
            .expect("create_compute_handle should succeed");

        assert!(!backend.is_using_fp16(&*handle));
        assert!(!handle.is_using_fp16());
        assert!(!backend.set_is_warmup(&mut *handle, true));
        assert!(handle.set_is_warmup(false));

        let _buffers = backend
            .create_input_buffers(&*model, 1, 19, 19)
            .expect("create_input_buffers should succeed");

        backend.global_cleanup();
    }

    #[test]
    fn test_dummy_get_output_sets_results() {
        use dummy::DummyBackend;

        let backend = DummyBackend;
        let model = backend.load_model_file("dummy.bin", "").unwrap();
        let logger = Logger::new(kata_core::logger::LoggerOptions::default(), None);
        let ctx = backend
            .create_compute_context(
                &[],
                &logger,
                19,
                19,
                "",
                Enabled::Auto,
                &*model,
                &Config::new(false, true),
            )
            .unwrap();
        let handle = backend
            .create_compute_handle(&*ctx, &*model, &logger, 2, false, false, 0, 0)
            .unwrap();
        let buffers = backend.create_input_buffers(&*model, 2, 19, 19).unwrap();

        let mut input0 = NNResultBuf::new();
        let mut input1 = NNResultBuf::new();
        let mut output0 = NNOutput::default();
        let mut output1 = NNOutput::default();

        {
            let mut input_refs: Vec<&mut NNResultBuf> = vec![&mut input0, &mut input1];
            let mut output_refs: Vec<&mut NNOutput> = vec![&mut output0, &mut output1];
            backend
                .get_output(&*handle, &*buffers, 2, &mut input_refs, &mut output_refs)
                .expect("get_output should succeed");
        }

        assert!(input0.has_result);
        assert!(input1.has_result);
        assert!(input0.result.is_some());
        assert!(input1.result.is_some());
        assert_eq!(output0.white_win_prob, 0.0);
        assert_eq!(output1.white_win_prob, 0.0);
    }
}
