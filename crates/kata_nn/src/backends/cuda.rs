//! Hand-written CUDA backend (M3 骨架：kernel 加载与基础运行时）。
//!
//! kernel 由 `build.rs` 按 `configs/sm-targets.json` 编译为 PTX 并嵌入二进制
//! （`CUDA_TARGETS` 表）；运行时按设备 compute capability 选择最优目标。
//!
//! 对应 KataGo 官方 `cpp/neuralnet/cudabackend.cpp`，并参考 KataGomo_fork 的
//! SM120 优化路径（plan 驱动、fail-closed 见 M4）。

#[cfg(feature = "cuda")]
mod imp {
    use cudarc::driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg,
    };
    use cudarc::driver::sys::CUdevice_attribute;
    use std::sync::Arc;

    // 由 build.rs 生成的 PTX 表：`(target_id, compute_capability, [(kernel_name, ptx_bytes)])`。
    include!(concat!(env!("OUT_DIR"), "/cuda_kernels.rs"));

    /// 按设备 compute capability 选择编译入的最优 SM 目标。
    fn select_target(dev: &Arc<CudaContext>) -> Result<(&'static str, &'static [(&'static str, &'static [u8])]), String> {
        let major = dev
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| format!("failed to query compute capability: {e}"))?;
        let minor = dev
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| format!("failed to query compute capability: {e}"))?;
        let cap = format!("{major}.{minor}");
        let _ = &cap;
        // 精确匹配优先，否则选数值上不大于设备能力的最新目标。
        for (id, target_cap, kernels) in CUDA_TARGETS {
            if *target_cap == cap {
                return Ok((id, kernels));
            }
        }
        let mut best: Option<(&str, &str, &[(&str, &[u8])])> = None;
        for (id, target_cap, kernels) in CUDA_TARGETS {
            let compatible = target_cap
                .parse::<f32>()
                .map(|t| t <= major as f32 + minor as f32 / 10.0)
                .unwrap_or(false);
            if compatible {
                let better = best
                    .map(|(_, bcap, _)| target_cap.parse::<f32>().unwrap_or(0.0) > bcap.parse::<f32>().unwrap_or(0.0))
                    .unwrap_or(true);
                if better {
                    best = Some((id, target_cap, kernels));
                }
            }
        }
        match best {
            Some((id, _, kernels)) => Ok((id, kernels)),
            None => Err(format!("no compiled SM target compatible with device capability {cap}")),
        }
    }

    /// CUDA 运行时：设备 + 按 SM 目标加载的 kernel 模块。
    pub struct CudaRuntime {
        pub device: Arc<CudaContext>,
        pub target_id: String,
        modules: Vec<(String, Arc<CudaModule>)>,
    }

    impl CudaRuntime {
        /// 初始化设备 0 并加载最优 SM 目标的全部 kernel（fail-closed）。
        pub fn new() -> Result<Self, String> {
            let device = CudaContext::new(0).map_err(|e| format!("CUDA device 0 init failed: {e}"))?;
            let (target_id, kernels) = select_target(&device)?;
            let mut modules = Vec::new();
            for (name, ptx) in kernels {
                let ptx = cudarc::nvrtc::Ptx::from_src(
                    std::str::from_utf8(ptx).map_err(|_| "PTX not UTF-8".to_string())?.to_string(),
                );
                let module = device
                    .load_module(ptx)
                    .map_err(|e| format!("failed to load PTX for kernel {name}: {e}"))?;
                modules.push((name.to_string(), module));
            }
            Ok(Self {
                device,
                target_id: target_id.to_string(),
                modules,
            })
        }

        /// 按名字查找 kernel 函数。
        pub fn get_func(&self, name: &str) -> Result<CudaFunction, String> {
            for (kname, module) in &self.modules {
                if let Ok(f) = module.load_function(name) {
                    let _ = kname;
                    return Ok(f);
                }
            }
            Err(format!("kernel {name} not found"))
        }

        /// 冒烟/自检：`out[i] = a[i] + b[i]`。
        pub fn f32_add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
            assert_eq!(a.len(), b.len());
            let n = a.len();
            let f = self.get_func("f32_add_kernel")?;
            let stream = self.device.default_stream();
            let mut d_a: CudaSlice<f32> =
                unsafe { stream.alloc(n) }.map_err(|e| e.to_string())?;
            let mut d_b: CudaSlice<f32> =
                unsafe { stream.alloc(n) }.map_err(|e| e.to_string())?;
            stream
                .memcpy_htod(a, &mut d_a)
                .map_err(|e| e.to_string())?;
            stream
                .memcpy_htod(b, &mut d_b)
                .map_err(|e| e.to_string())?;
            let mut d_out: CudaSlice<f32> = stream.alloc_zeros(n).map_err(|e| e.to_string())?;
            let cfg = LaunchConfig::for_num_elems(n as u32);
            unsafe {
                stream
                    .launch_builder(&f)
                    .arg(&d_a)
                    .arg(&d_b)
                    .arg(&mut d_out)
                    .arg(&(n as i32))
                    .launch(cfg)
            }
            .map_err(|e| format!("launch failed: {e}"))?;
            let mut out = vec![0.0f32; n];
            stream
                .memcpy_dtoh(&d_out, &mut out)
                .map_err(|e| e.to_string())?;
            Ok(out)
        }

        /// FP16 张量核 GEMM：`C[M,N] = alpha * A[M,K] * B[N,K]^T + beta * C`。
        ///
        /// `a`/`b` 为 f32 主机数据（主机侧转 half），`c` 就地读写；K 必须是
        /// 16 的倍数（调用方负责 padding）。对应 `cuda-kernels/gemm.cu` 的
        /// `hgemm_m16n8k16_kernel`（内联 PTX mma.sync m16n8k16）。
        #[allow(clippy::too_many_arguments)]
        pub fn hgemm_m16n8k16(
            &self,
            a: &[f32],
            b: &[f32],
            c: &mut [f32],
            m: usize,
            n: usize,
            k: usize,
            alpha: f32,
            beta: f32,
        ) -> Result<(), String> {
            assert_eq!(a.len(), m * k, "A 尺寸不符");
            assert_eq!(b.len(), n * k, "B 尺寸不符");
            assert_eq!(c.len(), m * n, "C 尺寸不符");
            assert_eq!(k % 16, 0, "K 必须是 16 的倍数");

            let a_half: Vec<u16> = a.iter().map(|&x| f32_to_f16_bits(x)).collect();
            let b_half: Vec<u16> = b.iter().map(|&x| f32_to_f16_bits(x)).collect();

            let f = self.get_func("hgemm_m16n8k16_kernel")?;
            let stream = self.device.default_stream();
            let mut d_a: CudaSlice<u16> =
                unsafe { stream.alloc(a_half.len()) }.map_err(|e| e.to_string())?;
            let mut d_b: CudaSlice<u16> =
                unsafe { stream.alloc(b_half.len()) }.map_err(|e| e.to_string())?;
            let mut d_c: CudaSlice<f32> =
                unsafe { stream.alloc(c.len()) }.map_err(|e| e.to_string())?;
            stream
                .memcpy_htod(a_half.as_slice(), &mut d_a)
                .map_err(|e| e.to_string())?;
            stream
                .memcpy_htod(b_half.as_slice(), &mut d_b)
                .map_err(|e| e.to_string())?;
            stream
                .memcpy_htod(c, &mut d_c)
                .map_err(|e| e.to_string())?;

            let grid = (
                m.div_ceil(64) as u32,
                n.div_ceil(64) as u32,
                1u32,
            );
            let block = (4 * 32) as u32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: grid,
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(&f)
                    .arg(&d_a)
                    .arg(&d_b)
                    .arg(&mut d_c)
                    .arg(&(m as i32))
                    .arg(&(n as i32))
                    .arg(&(k as i32))
                    .arg(&alpha)
                    .arg(&beta)
                    .launch(cfg)
            }
            .map_err(|e| format!("hgemm launch failed: {e}"))?;

            stream
                .memcpy_dtoh(&d_c, c)
                .map_err(|e| e.to_string())?;
            Ok(())
        }
    }

    /// f32 → f16 位模式（round-to-nearest-even，与 CUDA `__float2half` 一致）。
    pub fn f32_to_f16_bits(x: f32) -> u16 {        let b = x.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32;
        let mant = b & 0x7fffff;
        if exp == 0xff {
            return if mant != 0 { sign | 0x7e00 } else { sign | 0x7c00 };
        }
        let e = exp - 127 + 15;
        if e >= 0x1f {
            return sign | 0x7c00;
        }
        if e <= 0 {
            if e < -10 {
                return sign;
            }
            let m = (mant | 0x800000) >> (14 - e);
            let m = (m + 1) >> 1;
            return sign | (m as u16);
        }
        let m = mant >> 13;
        let round_bits = mant & 0x1fff;
        let mut m = m as u16;
        if round_bits > 0x1000 || (round_bits == 0x1000 && (m & 1) == 1) {
            m += 1;
            if m == 0x400 {
                if e + 1 >= 0x1f {
                    return sign | 0x7c00;
                }
                return sign | (((e + 1) as u16) << 10);
            }
        }
        sign | ((e as u16) << 10) | m
    }

    /// f16 位模式 → f32（主机侧，测试/参考用）。
    pub fn f16_to_f32_bits(bits: u16) -> f32 {
        let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0f32 };
        let e = ((bits >> 10) & 0x1f) as i32;
        let m = (bits & 0x3ff) as u32;
        if e == 0 {
            if m == 0 {
                return 0.0 * sign;
            }
            sign * (m as f32) * 2f32.powi(-24)
        } else if e == 0x1f {
            f32::INFINITY * sign
        } else {
            sign * (1.0 + (m as f32) / 1024.0) * 2f32.powi(e - 15)
        }
    }

    impl CudaRuntime {
    /// 通用 1D f16 逐元素 kernel：`in -> out`（单输入单输出）。
    fn run_elem1(&self, kernel: &str, x: &[u16], n: usize) -> Result<Vec<u16>, String> {        let f = self.get_func(kernel)?;
        let stream = self.device.default_stream();
        let mut d_in: CudaSlice<u16> =
            unsafe { stream.alloc(n) }.map_err(|e| e.to_string())?;
        let mut d_out: CudaSlice<u16> = stream.alloc_zeros(n).map_err(|e| e.to_string())?;
        stream.memcpy_htod(x, &mut d_in).map_err(|e| e.to_string())?;
        unsafe {
            stream
                .launch_builder(&f)
                .arg(&d_in)
                .arg(&mut d_out)
                .arg(&(n as i32))
                .launch(LaunchConfig::for_num_elems(n as u32))
        }
        .map_err(|e| format!("{kernel} launch failed: {e}"))?;
        let mut out = vec![0u16; n];
        stream.memcpy_dtoh(&d_out, &mut out).map_err(|e| e.to_string())?;
        Ok(out)
    }

    /// SiLU（f16 逐元素，FP32 计算）。
    pub fn silu_f16(&self, x: &[f32]) -> Result<Vec<f32>, String> {
        let xh: Vec<u16> = x.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let out = self.run_elem1("silu_f16_kernel", &xh, x.len())?;
        Ok(out.iter().map(|&b| f16_to_f32_bits(b)).collect())
    }

    /// 原位残差 `res += in`（f16）。
    pub fn add_residual_f16(&self, x: &[f32], res: &mut [f32]) -> Result<(), String> {
        assert_eq!(x.len(), res.len());
        let n = x.len();
        let xh: Vec<u16> = x.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let rh: Vec<u16> = res.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let f = self.get_func("add_residual_f16_kernel")?;
        let stream = self.device.default_stream();
        let mut d_in: CudaSlice<u16> =
            unsafe { stream.alloc(n) }.map_err(|e| e.to_string())?;
        let mut d_res: CudaSlice<u16> =
            unsafe { stream.alloc(n) }.map_err(|e| e.to_string())?;
        stream.memcpy_htod(xh.as_slice(), &mut d_in).map_err(|e| e.to_string())?;
        stream.memcpy_htod(rh.as_slice(), &mut d_res).map_err(|e| e.to_string())?;
        unsafe {
            stream
                .launch_builder(&f)
                .arg(&d_in)
                .arg(&mut d_res)
                .arg(&(n as i32))
                .launch(LaunchConfig::for_num_elems(n as u32))
        }
        .map_err(|e| format!("add_residual launch failed: {e}"))?;
        let mut out = vec![0u16; n];
        stream.memcpy_dtoh(&d_res, &mut out).map_err(|e| e.to_string())?;
        for (r, b) in res.iter_mut().zip(out.iter()) {
            *r = f16_to_f32_bits(*b);
        }
        Ok(())
    }

    /// RMSNorm（每行 ncols 元素，f16）。
    pub fn rms_norm_f16(
        &self,
        x: &[f32],
        scale: &[f32],
        eps: f32,
        ncols: usize,
    ) -> Result<Vec<f32>, String> {
        assert_eq!(x.len() % ncols, 0);
        assert_eq!(scale.len(), ncols);
        let rows = x.len() / ncols;
        let xh: Vec<u16> = x.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let f = self.get_func("rms_norm_f16_kernel")?;
        let stream = self.device.default_stream();
        let mut d_x: CudaSlice<u16> =
            unsafe { stream.alloc(x.len()) }.map_err(|e| e.to_string())?;
        let mut d_s: CudaSlice<f32> =
            unsafe { stream.alloc(ncols) }.map_err(|e| e.to_string())?;
        let mut d_y: CudaSlice<u16> =
            stream.alloc_zeros(x.len()).map_err(|e| e.to_string())?;
        stream.memcpy_htod(xh.as_slice(), &mut d_x).map_err(|e| e.to_string())?;
        stream.memcpy_htod(scale, &mut d_s).map_err(|e| e.to_string())?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&f)
                .arg(&d_x)
                .arg(&d_s)
                .arg(&mut d_y)
                .arg(&eps)
                .arg(&(ncols as i32))
                .launch(cfg)
        }
        .map_err(|e| format!("rms_norm launch failed: {e}"))?;
        let mut out = vec![0u16; x.len()];
        stream.memcpy_dtoh(&d_y, &mut out).map_err(|e| e.to_string())?;
        Ok(out.iter().map(|&b| f16_to_f32_bits(b)).collect())
    }

    /// 注意力 v1（正确优先）：`out = softmax(Q·K^T/sqrt(D))·V`，non-causal 无掩码。
    /// q/k/v：[B*H, S, D] 行主序 f32（主机侧转 half）。
    pub fn attention_row(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        s: usize,
        d: usize,
    ) -> Result<Vec<f32>, String> {
        let bh = q.len() / (s * d);
        assert_eq!(q.len(), bh * s * d);
        assert_eq!(k.len(), bh * s * d);
        assert_eq!(v.len(), bh * s * d);
        let n = q.len();
        let qh: Vec<u16> = q.iter().map(|&x| f32_to_f16_bits(x)).collect();
        let kh: Vec<u16> = k.iter().map(|&x| f32_to_f16_bits(x)).collect();
        let vh: Vec<u16> = v.iter().map(|&x| f32_to_f16_bits(x)).collect();
        let f = self.get_func("attention_row_kernel")?;
        let stream = self.device.default_stream();
        let mut d_q: CudaSlice<u16> =
            unsafe { stream.alloc(n) }.map_err(|e| e.to_string())?;
        let mut d_k: CudaSlice<u16> =
            unsafe { stream.alloc(n) }.map_err(|e| e.to_string())?;
        let mut d_v: CudaSlice<u16> =
            unsafe { stream.alloc(n) }.map_err(|e| e.to_string())?;
        let mut d_o: CudaSlice<u16> =
            stream.alloc_zeros(n).map_err(|e| e.to_string())?;
        stream.memcpy_htod(qh.as_slice(), &mut d_q).map_err(|e| e.to_string())?;
        stream.memcpy_htod(kh.as_slice(), &mut d_k).map_err(|e| e.to_string())?;
        stream.memcpy_htod(vh.as_slice(), &mut d_v).map_err(|e| e.to_string())?;
        // kernel 归约工作区固定 ATT_BLOCK=384，且要求 s ≤ 384。
        assert!(s <= 512, "attention_row v1 requires S <= 512");
        let block = 512u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (s as u32, bh as u32, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let scale = 1.0 / (d as f32).sqrt();
        unsafe {
            stream
                .launch_builder(&f)
                .arg(&d_q)
                .arg(&d_k)
                .arg(&d_v)
                .arg(&mut d_o)
                .arg(&(s as i32))
                .arg(&(d as i32))
                .arg(&scale)
                .launch(cfg)
        }
        .map_err(|e| format!("attention launch failed: {e}"))?;
        let mut out = vec![0u16; n];
        stream.memcpy_dtoh(&d_o, &mut out).map_err(|e| e.to_string())?;
        Ok(out.iter().map(|&b| f16_to_f32_bits(b)).collect())
    }
    }
}

/// CUDA 后端入口（无 feature 时的占位实现）。
#[cfg(not(feature = "cuda"))]
pub struct CudaRuntime;

#[cfg(not(feature = "cuda"))]
impl CudaRuntime {
    pub fn new() -> Result<Self, String> {
        Err("CUDA backend not enabled (enable the 'cuda' Cargo feature)".to_string())
    }
}

#[cfg(feature = "cuda")]
pub use imp::{f16_to_f32_bits, f32_to_f16_bits, CudaRuntime};

// ---------------------------------------------------------------------------
// Backend trait 对接：把 cuda_exec 的整图执行器接入 NN 求值器
// ---------------------------------------------------------------------------
#[cfg(feature = "cuda")]
mod backend_impl {
    use super::imp::CudaRuntime;
    use crate::backend::{
        Backend, ComputeContext, ComputeHandle, Enabled, InputBuffers, LoadedModel, NNOutput,
        NNResultBuf, NeuralNetError,
    };
    use crate::backends::cuda_exec::CudaModel;
    use crate::desc::ModelDesc;
    use kata_core::config::Config;
    use kata_core::logger::Logger;
    use kata_game::symmetry::{copy_inputs_with_symmetry, copy_outputs_with_symmetry, invert};
    use std::any::Any;
    use std::sync::Arc;

    /// CUDA 后端（手写 kernel，sm-targets.json 选择编译目标）。
    pub struct CudaBackend;

    /// 已加载模型：层图权重常驻设备 + 运行时（设备/模块）。
    pub struct CudaLoadedModel {
        model_desc: ModelDesc,
        model: Arc<CudaModel>,
        rt: Arc<CudaRuntime>,
    }

    impl LoadedModel for CudaLoadedModel {
        fn model_desc(&self) -> &ModelDesc {
            &self.model_desc
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// 跨线程共享的推理上下文：模型 + 运行时 + 棋盘尺寸。
    pub struct CudaComputeContext {
        model: Arc<CudaModel>,
        rt: Arc<CudaRuntime>,
        nn_x_len: i32,
        nn_y_len: i32,
    }

    impl ComputeContext for CudaComputeContext {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// 每线程计算句柄（本实现无需 per-thread 状态，共享模型/运行时）。
    pub struct CudaComputeHandle {
        model: Arc<CudaModel>,
        rt: Arc<CudaRuntime>,
        nn_x_len: i32,
        nn_y_len: i32,
        max_batch_size: i32,
        inputs_use_nhwc: bool,
    }

    impl ComputeHandle for CudaComputeHandle {
        fn is_using_fp16(&self) -> bool {
            true
        }
        fn set_is_warmup(&mut self, _is_warmup: bool) -> bool {
            false
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    pub struct CudaInputBuffers;

    impl InputBuffers for CudaInputBuffers {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    impl Backend for CudaBackend {
        fn global_initialize(&self) {}
        fn global_cleanup(&self) {}

        fn print_devices(&self) {
            println!("Hand-written CUDA backend (cuda_exec + embedded PTX, SM120-first)");
        }

        fn load_model_file(
            &self,
            file: &str,
            _expected_sha256: &str,
        ) -> Result<Box<dyn LoadedModel>, NeuralNetError> {
            let bytes = std::fs::read(file)
                .map_err(|e| NeuralNetError(format!("could not read {file}: {e}")))?;
            let parsed = crate::onnx_model::parse_onnx_model(&bytes)
                .map_err(|e| NeuralNetError(format!("onnx metadata parse failed: {e}")))?;
            let graph = crate::onnx_parser::parse_layer_graph(&bytes)
                .map_err(|e| NeuralNetError(format!("layer graph parse failed: {e}")))?;
            let rt = Arc::new(
                CudaRuntime::new().map_err(|e| NeuralNetError(format!("CUDA init failed: {e}")))?,
            );
            let model = Arc::new(
                CudaModel::load(&graph, &rt)
                    .map_err(|e| NeuralNetError(format!("CUDA model upload failed: {e}")))?,
            );
            Ok(Box::new(CudaLoadedModel {
                model_desc: parsed.model_desc,
                model,
                rt,
            }))
        }

        fn create_compute_context(
            &self,
            _gpu_idxs: &[i32],
            _logger: &Logger,
            nn_x_len: i32,
            nn_y_len: i32,
            _home_data_dir_override: &str,
            _use_fp16_mode: Enabled,
            loaded_model: &dyn LoadedModel,
            _cfg: &Config,
        ) -> Result<Box<dyn ComputeContext>, NeuralNetError> {
            let model = loaded_model
                .as_any()
                .downcast_ref::<CudaLoadedModel>()
                .ok_or_else(|| NeuralNetError("Wrong loaded model type".to_string()))?;
            Ok(Box::new(CudaComputeContext {
                model: model.model.clone(),
                rt: model.rt.clone(),
                nn_x_len,
                nn_y_len,
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
            _gpu_idx_for_this_thread: i32,
            _server_thread_idx: i32,
        ) -> Result<Box<dyn ComputeHandle>, NeuralNetError> {
            let c = ctx
                .as_any()
                .downcast_ref::<CudaComputeContext>()
                .ok_or_else(|| NeuralNetError("Wrong compute context type".to_string()))?;
            Ok(Box::new(CudaComputeHandle {
                model: c.model.clone(),
                rt: c.rt.clone(),
                nn_x_len: c.nn_x_len,
                nn_y_len: c.nn_y_len,
                max_batch_size,
                inputs_use_nhwc,
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
            Ok(Box::new(CudaInputBuffers))
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
                .downcast_ref::<CudaComputeHandle>()
                .ok_or_else(|| NeuralNetError("Wrong compute handle type".to_string()))?;

            let n = num_batch_elts as usize;
            if n == 0 {
                return Ok(());
            }
            if n > h.max_batch_size as usize {
                return Err(NeuralNetError(format!(
                    "batch size {n} exceeds handle max {}",
                    h.max_batch_size
                )));
            }
            let nn_x_len = h.nn_x_len;
            let nn_y_len = h.nn_y_len;
            let policy_area = (nn_x_len * nn_y_len) as usize;
            if policy_area != 361 {
                return Err(NeuralNetError(format!(
                    "CUDA backend only supports the 19x19 board (got {nn_x_len}x{nn_y_len})"
                )));
            }

            // --- 填输入（NCHW spatial + global，空间特征带对称性） ----------
            const NUM_SPATIAL_CHANNELS: i32 = 22;
            const NUM_GLOBAL_CHANNELS: usize = 19;
            let single_spatial = (NUM_SPATIAL_CHANNELS * nn_x_len * nn_y_len) as usize;

            let mut spatial_host = vec![0.0f32; n * single_spatial];
            let mut global_host = vec![0.0f32; n * NUM_GLOBAL_CHANNELS];
            for i in 0..n {
                let sym_idx = input_bufs[i].symmetry;
                let sp_off = i * single_spatial;
                copy_inputs_with_symmetry(
                    &input_bufs[i].row_spatial_buf,
                    &mut spatial_host[sp_off..sp_off + single_spatial],
                    1,
                    nn_y_len,
                    nn_x_len,
                    NUM_SPATIAL_CHANNELS,
                    h.inputs_use_nhwc,
                    sym_idx,
                );
                let gl_off = i * NUM_GLOBAL_CHANNELS;
                let gb = &input_bufs[i].row_global_buf;
                let copy_len = NUM_GLOBAL_CHANNELS.min(gb.len());
                global_host[gl_off..gl_off + copy_len].copy_from_slice(&gb[..copy_len]);
            }

            // --- 上传 + 前向 -------------------------------------------------
            let stream = h.rt.device.default_stream();
            let mut d_spatial: cudarc::driver::CudaSlice<f32> =
                unsafe { stream.alloc(spatial_host.len()) }
                    .map_err(|e| NeuralNetError(format!("alloc spatial: {e}")))?;
            let mut d_global: cudarc::driver::CudaSlice<f32> =
                unsafe { stream.alloc(global_host.len()) }
                    .map_err(|e| NeuralNetError(format!("alloc global: {e}")))?;
            stream
                .memcpy_htod(spatial_host.as_slice(), &mut d_spatial)
                .map_err(|e| NeuralNetError(format!("upload spatial: {e}")))?;
            stream
                .memcpy_htod(global_host.as_slice(), &mut d_global)
                .map_err(|e| NeuralNetError(format!("upload global: {e}")))?;

            let outs = h
                .model
                .apply(&h.rt, &d_spatial, &d_global, n)
                .map_err(|e| NeuralNetError(format!("CUDA forward failed: {e}")))?;
            let host = outs
                .to_host(&h.rt)
                .map_err(|e| NeuralNetError(format!("CUDA download failed: {e}")))?;

            // --- 解码 v15 输出（与 trt.rs generic_get_output 同口径） --------
            let single_policy = 6 * (policy_area + 1);
            let mut tmp_policy_base = vec![0.0f32; policy_area];
            let mut tmp_policy_opt = vec![0.0f32; policy_area];
            let mut tmp_ownership = vec![0.0f32; policy_area];

            for i in 0..n {
                let sym_idx = input_bufs[i].symmetry;
                let inv_sym = if sym_idx != 0 { invert(sym_idx) } else { 0 };

                // Policy：通道 0 基策略 + 通道 5 乐观策略，pass 位 361。
                let p_off = i * single_policy;
                let base_src = &host.policy[p_off..p_off + policy_area];
                let opt_src = &host.policy[p_off + 5 * policy_area..p_off + 6 * policy_area];
                if inv_sym != 0 {
                    copy_outputs_with_symmetry(base_src, &mut tmp_policy_base, 1, nn_y_len, nn_x_len, inv_sym);
                    copy_outputs_with_symmetry(opt_src, &mut tmp_policy_opt, 1, nn_y_len, nn_x_len, inv_sym);
                } else {
                    tmp_policy_base.copy_from_slice(base_src);
                    tmp_policy_opt.copy_from_slice(opt_src);
                }
                let optimism = input_bufs[i].policy_optimism as f32;
                for pos in 0..policy_area {
                    outputs[i].policy_probs[pos] =
                        tmp_policy_base[pos] + (tmp_policy_opt[pos] - tmp_policy_base[pos]) * optimism;
                }
                let base_pass = host.policy[p_off + policy_area];
                let opt_pass = host.policy[p_off + 5 * policy_area + policy_area];
                outputs[i].policy_probs[policy_area] = base_pass + (opt_pass - base_pass) * optimism;

                // Value：3 通道。
                let v_off = i * 3;
                outputs[i].white_win_prob = host.value[v_off];
                outputs[i].white_loss_prob = host.value[v_off + 1];
                outputs[i].white_no_result_prob = host.value[v_off + 2];

                // Score value（misc 前 4 个）。
                let m_off = i * 10;
                outputs[i].white_score_mean = host.misc[m_off];
                outputs[i].white_score_mean_sq = host.misc[m_off + 1];
                outputs[i].white_lead = host.misc[m_off + 2];
                outputs[i].var_time_left = host.misc[m_off + 3];

                // Shortterm 误差（moremisc 前 2 个）。
                let mm_off = i * 8;
                outputs[i].shortterm_winloss_error = host.moremisc[mm_off];
                outputs[i].shortterm_score_error = host.moremisc[mm_off + 1];

                // Ownership。
                if input_bufs[i].include_owner_map {
                    let o_off = i * policy_area;
                    let src = &host.ownership[o_off..o_off + policy_area];
                    if inv_sym != 0 {
                        copy_outputs_with_symmetry(src, &mut tmp_ownership, 1, nn_y_len, nn_x_len, inv_sym);
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
}

#[cfg(feature = "cuda")]
pub use backend_impl::CudaBackend;
