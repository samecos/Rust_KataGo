//! Hand-written CUDA backend (M3 骨架：kernel 加载与基础运行时）。
//!
//! kernel 由 `build.rs` 按 `configs/sm-targets.json` 编译为 PTX 并嵌入二进制
//! （`CUDA_TARGETS` 表）；运行时按设备 compute capability 选择最优目标。
//!
//! 对应 KataGo 官方 `cpp/neuralnet/cudabackend.cpp`，并参考 KataGomo_fork 的
//! SM120 优化路径（plan 驱动、fail-closed 见 M4）。

#[cfg(feature = "cuda")]
#[path = "strict_attention.rs"]
pub(crate) mod strict_attention;
#[cfg(feature = "cuda")]
#[path = "qkv_immutable.rs"]
pub(crate) mod qkv_immutable;

#[cfg(feature = "cuda")]
#[path = "ffn_compact.rs"]
pub(crate) mod ffn_compact;
#[cfg(feature = "cuda")]
#[path = "outproj_n128.rs"]
pub(crate) mod outproj_n128;

#[cfg(feature = "cuda")]
mod imp {
    use cudarc::driver::sys::CUdevice_attribute;
    use cudarc::driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg,
    };
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    // 由 build.rs 生成的 PTX 表：`(target_id, compute_capability, [(kernel_name, ptx_bytes)])`。
    include!(concat!(env!("OUT_DIR"), "/cuda_kernels.rs"));
    // 由 build.rs 生成的构建指纹和编译能力表。
    include!(concat!(env!("OUT_DIR"), "/cuda_build.rs"));

    /// CUDA build and loaded-library fingerprint, shared by plans and diagnostics.
    pub fn backend_build_fingerprint() -> crate::tactic_plan::BackendBuildFingerprint {
        crate::tactic_plan::BackendBuildFingerprint {
            kernel_build_id: CUDA_KERNEL_BUILD_ID.to_string(),
            cuda_compiler: CUDA_COMPILER.to_string(),
            cutlass_version: CUDA_CUTLASS_VERSION.to_string(),
            cutlass_commit: CUDA_CUTLASS_COMMIT.to_string(),
            compiled_sm: CUDA_COMPILED_SMS.iter().map(|s| (*s).to_string()).collect(),
            capabilities: crate::tactic_plan::BackendCapabilities {
                dual_ffn: CUDA_CAP_DUAL_FFN,
                attention_q64: CUDA_CAP_ATTENTION_Q64,
                attention_q64_serial: CUDA_CAP_ATTENTION_Q64_SERIAL,
            },
            fp16_encoding_revision: Some(crate::tactic_plan::CUDA_FP16_ENCODING_REVISION),
            strict_attention_artifact: super::strict_attention::fingerprint(),
            qkv_immutable_artifact: super::qkv_immutable::fingerprint(),
            ffn_compact_artifact: super::ffn_compact::fingerprint(),
            outproj_n128_artifact: super::outproj_n128::fingerprint(),
            host_tactic_revision: Some(crate::tactic_plan::CUDA_HOST_TACTIC_REVISION),
            cublaslt_version: Some(unsafe { cudarc::cublaslt::sys::cublasLtGetVersion() } as u64),
        }
    }

    /// Whether the optional q64 attention PTX was compiled into this binary.
    /// Keep the hot-path capability check allocation-free; the full fingerprint
    /// is intentionally reserved for plan validation and diagnostics.
    pub fn attention_q64_available() -> bool {
        CUDA_CAP_ATTENTION_Q64
    }

    /// Independent capability: q64 may be present without the serial entry.
    pub fn attention_q64_serial_available() -> bool {
        CUDA_CAP_ATTENTION_Q64_SERIAL
    }

    /// The same logical W[N,K], stored either row-major (TN) or transposed
    /// row-major [K,N] (NN). Only storage/descriptor layout changes; both use
    /// FP32 compute and the caller's existing output and residual types.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum CublasLtWeightLayout {
        Tn,
        Nn,
    }

    impl CublasLtWeightLayout {
        fn descriptor(self, n: usize, k: usize) -> (u32, u64, u64, i64) {
            match self {
                Self::Tn => (1, k as u64, n as u64, k as i64),
                Self::Nn => (0, n as u64, k as u64, n as i64),
            }
        }
    }

    /// Output type, residual mode and weight layout must have separate cache
    /// fields: FP16 output is not interchangeable with FP32 beta=1 output.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct CublasLtAlgoKey {
        m: usize,
        n: usize,
        k: usize,
        f16out: bool,
        residual: bool,
        layout: CublasLtWeightLayout,
        /// Do not reuse a heuristic/time choice when a fixed preset is enabled,
        /// or a preset choice for beta != 1 on the same residual shape.
        residual_preset: bool,
    }

    fn tf3_residual_candidate_index(
        m: usize,
        n: usize,
        k: usize,
        beta: f32,
        layout: CublasLtWeightLayout,
    ) -> Option<usize> {
        if beta != 1.0 || layout != CublasLtWeightLayout::Tn {
            return None;
        }
        match (m, n, k) {
            (5054, 384, 1152) => Some(1),
            (5776, 384, 1152) => Some(3),
            (5776, 384, 384) => Some(2),
            _ => None,
        }
    }

    // Exact opaque data of the three pre-registered candidates, not a portable
    // algorithm ID. Accepted only with the measured GPU and Lt 130600. These
    // are u64 values decoded from the little-endian bytes in
    // target/fork-parity-20260908/tf3-residual-lt-abba-report.json
    // SHA256 62a331f20a511dcd6d4b69f6b83b204ecac92f6d4cfca61b182455b9a6cb97e2.
    const TF3_RESIDUAL_ALGO_DATA: [u64; 8] = [
        0x0000000f00000015,
        0x000000010000000c,
        0,
        0,
        0x0001fe2800000001,
        0x0000000200000002,
        0x0000004400000000,
        0,
    ];

    fn fixed_residual_candidate(
        candidates: &[cudarc::cublaslt::sys::cublasLtMatmulAlgo_t],
        index: usize,
    ) -> Result<cudarc::cublaslt::sys::cublasLtMatmulAlgo_t, String> {
        let candidate = candidates.get(index).ok_or_else(|| format!(
            "tf3_5070ti_r1 requires filtered heuristic index {index}, only {} candidates available",
            candidates.len()
        ))?;
        if candidate.data != TF3_RESIDUAL_ALGO_DATA {
            return Err(format!(
                "tf3_5070ti_r1 algorithm identity mismatch at filtered heuristic index {index}: actual {:x?}",
                candidate.data
            ));
        }
        Ok(*candidate)
    }

    static RESIDUAL_B14_DOWN_REPORT: OnceLock<()> = OnceLock::new();
    static RESIDUAL_B16_DOWN_REPORT: OnceLock<()> = OnceLock::new();
    static RESIDUAL_B16_OUT_REPORT: OnceLock<()> = OnceLock::new();

    fn report_residual_preset(m: usize, n: usize, k: usize, index: usize) {
        let slot = match (m, n, k) {
            (5054, 384, 1152) => &RESIDUAL_B14_DOWN_REPORT,
            (5776, 384, 1152) => &RESIDUAL_B16_DOWN_REPORT,
            (5776, 384, 384) => &RESIDUAL_B16_OUT_REPORT,
            _ => return,
        };
        slot.get_or_init(|| {
            eprintln!(
                "[cuda-tactic] name=residual_algo engine=tf3_5070ti_r1 m={m} n={n} k={k} layout=tn output=f32 beta=1 filtered_index={index} cublaslt_version={} algorithm_identity=matched",
                crate::tactic_plan::TF3_RESIDUAL_CUBLASLT_VERSION
            );
        });
    }

    #[cfg(test)]
    mod residual_algo_tests {
        use super::*;

        #[test]
        fn residual_preset_is_limited_to_measured_tn_beta_one_shapes() {
            for (m, n, k, index) in [
                (5054, 384, 1152, 1),
                (5776, 384, 1152, 3),
                (5776, 384, 384, 2),
            ] {
                assert_eq!(
                    tf3_residual_candidate_index(m, n, k, 1.0, CublasLtWeightLayout::Tn),
                    Some(index)
                );
                assert_eq!(
                    tf3_residual_candidate_index(m, n, k, 1.0, CublasLtWeightLayout::Nn),
                    None
                );
                for beta in [0.0, -1.0, 0.5, 2.0, f32::NAN] {
                    assert_eq!(
                        tf3_residual_candidate_index(m, n, k, beta, CublasLtWeightLayout::Tn),
                        None
                    );
                }
            }
            for (m, n, k) in [
                (5054, 384, 384),
                (2888, 384, 1152),
                (5776, 768, 1152),
                (5776, 384, 2304),
                (5775, 384, 1152),
            ] {
                assert_eq!(
                    tf3_residual_candidate_index(m, n, k, 1.0, CublasLtWeightLayout::Tn),
                    None
                );
            }
        }

        #[test]
        fn fixed_residual_candidate_rejects_missing_or_changed_algorithm() {
            use cudarc::cublaslt::sys::cublasLtMatmulAlgo_t;
            let expected = cublasLtMatmulAlgo_t {
                data: TF3_RESIDUAL_ALGO_DATA,
            };
            let mut candidates = vec![expected; 4];
            assert_eq!(fixed_residual_candidate(&candidates, 3).unwrap(), expected);
            assert!(
                fixed_residual_candidate(&candidates[..3], 3)
                    .unwrap_err()
                    .contains("only 3 candidates")
            );
            candidates[3].data[0] ^= 1;
            assert!(
                fixed_residual_candidate(&candidates, 3)
                    .unwrap_err()
                    .contains("identity mismatch")
            );
            // Never search another index for the expected data after a mismatch.
            assert_eq!(fixed_residual_candidate(&candidates, 1).unwrap(), expected);
        }

        #[test]
        fn residual_preset_cache_does_not_alias_existing_choices() {
            let base = CublasLtAlgoKey {
                m: 5776,
                n: 384,
                k: 1152,
                f16out: false,
                residual: true,
                layout: CublasLtWeightLayout::Tn,
                residual_preset: false,
            };
            let preset = CublasLtAlgoKey {
                residual_preset: true,
                ..base
            };
            let mut cache = HashMap::new();
            cache.insert(base, 0);
            assert_eq!(cache.get(&preset), None);
            cache.insert(preset, 3);
            assert_eq!(cache.get(&base), Some(&0));
            assert_eq!(cache.get(&preset), Some(&3));
        }
    }

    /// cudarc 0.19 models graph instantiate bitflags as an enum without a zero
    /// variant. `transmute(0)` is undefined behavior and aborts debug builds.
    /// Using node priority is equivalent to defaults when nodes have no explicit
    /// priority attributes, while remaining a valid CUDA flag value.
    pub fn graph_instantiate_flags() -> cudarc::driver::sys::CUgraphInstantiate_flags {
        cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY
    }

    /// 按设备 compute capability 选择编译入的最优 SM 目标。
    fn select_target(
        dev: &Arc<CudaContext>,
    ) -> Result<(&'static str, &'static [(&'static str, &'static [u8])]), String> {
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
                    .map(|(_, bcap, _)| {
                        target_cap.parse::<f32>().unwrap_or(0.0)
                            > bcap.parse::<f32>().unwrap_or(0.0)
                    })
                    .unwrap_or(true);
                if better {
                    best = Some((id, target_cap, kernels));
                }
            }
        }
        match best {
            Some((id, _, kernels)) => Ok((id, kernels)),
            None => Err(format!(
                "no compiled SM target compatible with device capability {cap}"
            )),
        }
    }

    /// CUDA 运行时：设备 + 按 SM 目标加载的 kernel 模块。
    pub struct CudaRuntime {
        pub device: Arc<CudaContext>,
        pub target_id: String,
        modules: Vec<(String, Arc<CudaModule>)>,
        /// cuBLASLt GEMM,按形状、输出类型、残差模式和权重布局缓存算法。
        /// graph capture 前需 warmup(第一次调用完成算法选择)。
        cublaslt: Option<CublasLtState>,
        /// Optional, independently fingerprinted AOT modules. Preparation is
        /// explicit at model load, after any plan has been installed.
        strict_attention: OnceLock<Result<super::strict_attention::StrictAttention, String>>,
    }

    /// cuBLASLt 状态:handle + workspace + 算法缓存。
    struct CublasLtState {
        handle: cudarc::cublaslt::sys::cublasLtHandle_t,
        workspace_len: usize,
        /// cuBLASLt workspace cannot be shared by overlapping matmuls on
        /// different streams. Keep one allocation per stream while retaining
        /// the single runtime/model shared by all benchmark handles.
        workspaces: Mutex<HashMap<usize, CudaSlice<u8>>>,
        algo_cache: Mutex<HashMap<CublasLtAlgoKey, cudarc::cublaslt::sys::cublasLtMatmulAlgo_t>>,
    }
    // cuBLASLt handle 是 opaque 指针,create 后可跨线程使用(文档:线程安全)。
    unsafe impl Send for CublasLtState {}
    unsafe impl Sync for CublasLtState {}

    /// C1 开关(KATAGO_CUDA_CUBLASLT_RANK=time):算法选择从"启发式首选"
    /// 改为"top-8 候选实测计时取最优"。M0 探针实测 qkv 形状首选非最优(-26%)。
    /// 经 tactic_plan::tactic_var 读取(plan > env > 默认)。
    pub(crate) fn cublaslt_rank_time() -> bool {
        crate::tactic_plan::tactic_var("KATAGO_CUDA_CUBLASLT_RANK")
            .map(|v| v == "time")
            .unwrap_or(false)
    }

    impl CudaRuntime {
        pub(crate) fn prepare_strict_attention(
            &self,
            sequence: usize,
            heads: usize,
            head_dim: usize,
            width: usize,
        ) -> Result<bool, String> {
            let requested = crate::tactic_plan::strict_attention_requested()?;
            if !requested {
                return Ok(false);
            }
            if crate::backends::cuda_exec::capturing() {
                return Err("strict attention cannot be prepared during graph capture".into());
            }
            super::strict_attention::validate_shape(sequence, heads, head_dim, width)?;
            crate::tactic_plan::validate_strict_attention_target(&super::device_fingerprint(
                &self.device,
            )?)?;
            if !attention_q64_serial_available() {
                return Err("fa4-strict-b14-r1 requires the q64-serial fallback kernel".into());
            }
            // Resolve the fallback before inference too. A compile capability
            // alone does not prove that this runtime loaded the symbol.
            self.get_func("attention_fa2_q64_serial_kernel")?;
            self.strict_attention
                .get_or_init(|| super::strict_attention::StrictAttention::load(&self.device))
                .as_ref()
                .map_err(Clone::clone)?;
            Ok(true)
        }

        pub(crate) fn prepared_strict_attention(
            &self,
        ) -> Result<&super::strict_attention::StrictAttention, String> {
            self.strict_attention
                .get()
                .ok_or_else(|| {
                    "fa4-strict-b14-r1 was not prepared at model load; reload the model".to_string()
                })?
                .as_ref()
                .map_err(Clone::clone)
        }

        /// Call after plan installation, before preparing inference. Direct
        /// callers also validate on the first matching GEMM cache miss.
        pub fn validate_residual_algo_request(&self) -> Result<(), String> {
            if !crate::tactic_plan::residual_preset_requested()? {
                return Ok(());
            }
            let st = self
                .cublaslt
                .as_ref()
                .ok_or_else(|| "tf3_5070ti_r1 requires an available cuBLASLt handle".to_string())?;
            if st.workspace_len != 32 * 1024 * 1024 {
                return Err(
                    "tf3_5070ti_r1 requires the measured 32 MiB workspace preference".into(),
                );
            }
            let device = super::device_fingerprint(&self.device)?;
            let version = unsafe { cudarc::cublaslt::sys::cublasLtGetVersion() } as u64;
            crate::tactic_plan::validate_tf3_residual_target(&device, Some(version))
        }

        /// cublasLt 句柄（实验/测试路径用；生产 GEMM 走 cublaslt_gemm*）。
        #[doc(hidden)]
        pub fn cublaslt_handle(&self) -> Option<cudarc::cublaslt::sys::cublasLtHandle_t> {
            self.cublaslt.as_ref().map(|st| st.handle)
        }

        /// cublasLt 工作区大小（字节）。
        #[doc(hidden)]
        pub fn cublaslt_workspace_len(&self) -> usize {
            self.cublaslt
                .as_ref()
                .map(|st| st.workspace_len)
                .unwrap_or(0)
        }

        /// cublasLt 工作区设备指针（调用方须保证 rt 存活）。
        #[doc(hidden)]
        pub fn cublaslt_workspace_ptr(
            &self,
            stream: &Arc<cudarc::driver::CudaStream>,
        ) -> (u64, u64) {
            use cudarc::driver::DevicePtr;
            let Some(st) = &self.cublaslt else {
                return (0, 0);
            };
            let key = stream.cu_stream() as usize;
            let mut workspaces = st.workspaces.lock().unwrap();
            if !workspaces.contains_key(&key) {
                crate::backends::group_cost::setup_observed();
                // cudarc uses stream-ordered allocation when supported, so
                // allocate on the same stream that will consume the buffer.
                let ws = unsafe { stream.alloc(st.workspace_len) };
                let Ok(ws) = ws else { return (0, 0) };
                workspaces.insert(key, ws);
            }
            let (p, _g) = workspaces.get(&key).unwrap().device_ptr(stream);
            (p, 0)
        }

        /// 初始化设备 0 并加载最优 SM 目标的全部 kernel（fail-closed）。
        pub fn new() -> Result<Self, String> {
            let device =
                CudaContext::new(0).map_err(|e| format!("CUDA device 0 init failed: {e}"))?;
            // 关闭 cudarc 自动 event 跟踪：graph capture 会因未记录 event 的
            // 跨流 wait 报 CUDA_ERROR_STREAM_CAPTURE_ISOLATION。本后端显式
            // 管理同步（每 handle 单流、加载后 device 级同步、每次前向后
            // stream.synchronize），host 回拷由 SyncOnDrop 直接同步流保证。
            unsafe { device.disable_event_tracking() };
            let (target_id, kernels) = select_target(&device)?;
            let mut modules = Vec::new();
            for (name, ptx) in kernels {
                let ptx = cudarc::nvrtc::Ptx::from_src(
                    std::str::from_utf8(ptx)
                        .map_err(|_| "PTX not UTF-8".to_string())?
                        .to_string(),
                );
                let module = device
                    .load_module(ptx)
                    .map_err(|e| format!("failed to load PTX for kernel {name}: {e}"))?;
                modules.push((name.to_string(), module));
            }
            // cuBLASLt 初始化(可选;失败降级为手写 kernel)
            let cublaslt = Self::init_cublaslt(&device);
            Ok(Self {
                device,
                target_id: target_id.to_string(),
                modules,
                cublaslt,
                strict_attention: OnceLock::new(),
            })
        }

        /// 初始化 cuBLASLt handle + workspace。
        fn init_cublaslt(device: &Arc<CudaContext>) -> Option<CublasLtState> {
            use cudarc::cublaslt::sys;
            unsafe {
                let mut handle: sys::cublasLtHandle_t = std::ptr::null_mut();
                let r = sys::cublasLtCreate(&mut handle);
                if r != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    eprintln!("note: cublasLtCreate failed ({r:?}), cublaslt disabled");
                    return None;
                }
                let workspace_len = 32 * 1024 * 1024;
                Some(CublasLtState {
                    handle,
                    workspace_len,
                    workspaces: Mutex::new(HashMap::new()),
                    algo_cache: Mutex::new(HashMap::new()),
                })
            }
        }

        /// cuBLASLt GEMM：`C[m,n] = alpha * A[m,k] @ B[n,k]^T + beta*C`。
        /// f16 输入、f32 累加、f32 输出。返回是否成功（false → 调用方回退手写）。
        /// graph capture 前须至少调用一次同 shape（完成算法选择）。
        #[allow(clippy::too_many_arguments)]
        pub fn cublaslt_gemm(
            &self,
            stream: &Arc<cudarc::driver::CudaStream>,
            a: &CudaSlice<u16>,
            b: &CudaSlice<u16>,
            c: &mut CudaSlice<f32>,
            m: usize,
            n: usize,
            k: usize,
            beta: f32,
        ) -> Result<bool, String> {
            self.cublaslt_gemm_with_layout(stream, a, b, c, m, n, k, beta, CublasLtWeightLayout::Tn)
        }

        /// Explicit layout variant used by the opt-in K384 candidate and its
        /// diagnostic. `b` must have the storage declared by `layout`.
        #[allow(clippy::too_many_arguments)]
        pub fn cublaslt_gemm_with_layout(
            &self,
            stream: &Arc<cudarc::driver::CudaStream>,
            a: &CudaSlice<u16>,
            b: &CudaSlice<u16>,
            c: &mut CudaSlice<f32>,
            m: usize,
            n: usize,
            k: usize,
            beta: f32,
            layout: CublasLtWeightLayout,
        ) -> Result<bool, String> {
            let preset_index = match tf3_residual_candidate_index(m, n, k, beta, layout) {
                Some(index) if crate::tactic_plan::residual_preset_requested()? => Some(index),
                _ => None,
            };
            let Some(st) = &self.cublaslt else {
                if preset_index.is_some() {
                    return Err("tf3_5070ti_r1 requires an available cuBLASLt handle".into());
                }
                return Ok(false);
            };
            let key = CublasLtAlgoKey {
                m,
                n,
                k,
                f16out: false,
                residual: beta != 0.0,
                layout,
                residual_preset: preset_index.is_some(),
            };
            let cached = st.algo_cache.lock().unwrap().get(&key).copied();
            let algo = match cached {
                Some(a) => Some(a),
                None => {
                    crate::backends::group_cost::setup_observed();
                    let cands = self.cublaslt_select_algos(st, m, n, k, false, layout)?;
                    if let Some(index) = preset_index {
                        // No timing, precision changes, or fallback: the fixed
                        // choice wins over global rank=time only for this key.
                        self.validate_residual_algo_request()?;
                        let pick = fixed_residual_candidate(&cands, index)
                            .map_err(|e| format!("residual GEMM ({m},{n},{k}): {e}"))?;
                        st.algo_cache.lock().unwrap().insert(key, pick);
                        Some(pick)
                    } else if cands.is_empty() {
                        None
                    } else if cublaslt_rank_time() && !crate::backends::cuda_exec::capturing() {
                        // C1 计时重排:在当前活跃流上逐候选计时(scratch 承载输出,
                        // beta=0,不碰真实 C);与真实工作同流提交,天然有序。
                        crate::backends::cuda_exec::report_tactic_once(
                            &crate::backends::cuda_exec::CUBLASLT_RANK_TIME_REPORT,
                            "name=cublaslt_rank engine=time",
                        );
                        let tstream = stream.clone();
                        let mut scratch: CudaSlice<f32> = unsafe { tstream.alloc(c.len()) }
                            .map_err(|e| format!("rank scratch alloc: {e}"))?;
                        let pick = self
                            .cublaslt_time_candidates_f32(
                                st,
                                &tstream,
                                &cands,
                                a,
                                b,
                                &mut scratch,
                                m,
                                n,
                                k,
                                layout,
                            )
                            .unwrap_or(cands[0]);
                        st.algo_cache.lock().unwrap().insert(key, pick);
                        Some(pick)
                    } else if crate::backends::cuda_exec::capturing() {
                        // capture 中同步非法,无法计时:用启发式首选且不写缓存,
                        // 避免把未计时的选择固化进缓存。
                        Some(cands[0])
                    } else {
                        crate::backends::cuda_exec::report_cublaslt_rank_heuristic_once();
                        st.algo_cache.lock().unwrap().insert(key, cands[0]);
                        Some(cands[0])
                    }
                }
            };
            let Some(algo) = algo else { return Ok(false) };
            self.cublaslt_exec(st, stream, &algo, a, b, c, m, n, k, beta, layout)?;
            if let Some(index) = preset_index {
                report_residual_preset(m, n, k, index);
            }
            Ok(true)
        }

        /// cuBLASLt GEMM 的 f16 输出变体：`C[m,n] f16 = A @ B^T`（epilogue
        /// 直接 f16，对应手写 hgemm_f16）。用于 qkv packed / dual FFN。
        #[allow(clippy::too_many_arguments)]
        pub fn cublaslt_gemm_f16out(
            &self,
            stream: &Arc<cudarc::driver::CudaStream>,
            a: &CudaSlice<u16>,
            b: &CudaSlice<u16>,
            c: &mut CudaSlice<u16>,
            m: usize,
            n: usize,
            k: usize,
        ) -> Result<bool, String> {
            self.cublaslt_gemm_f16out_with_layout(
                stream,
                a,
                b,
                c,
                m,
                n,
                k,
                CublasLtWeightLayout::Tn,
            )
        }

        /// FP32 compute, FP16 output, with an explicit weight storage layout.
        #[allow(clippy::too_many_arguments)]
        pub fn cublaslt_gemm_f16out_with_layout(
            &self,
            stream: &Arc<cudarc::driver::CudaStream>,
            a: &CudaSlice<u16>,
            b: &CudaSlice<u16>,
            c: &mut CudaSlice<u16>,
            m: usize,
            n: usize,
            k: usize,
            layout: CublasLtWeightLayout,
        ) -> Result<bool, String> {
            let Some(st) = &self.cublaslt else {
                return Ok(false);
            };
            let key = CublasLtAlgoKey {
                m,
                n,
                k,
                f16out: true,
                residual: false,
                layout,
                residual_preset: false,
            };
            let cached = st.algo_cache.lock().unwrap().get(&key).copied();
            let algo = match cached {
                Some(a) => Some(a),
                None => {
                    crate::backends::group_cost::setup_observed();
                    let cands = self.cublaslt_select_algos(st, m, n, k, true, layout)?;
                    if cands.is_empty() {
                        None
                    } else if cublaslt_rank_time() && !crate::backends::cuda_exec::capturing() {
                        crate::backends::cuda_exec::report_tactic_once(
                            &crate::backends::cuda_exec::CUBLASLT_RANK_TIME_REPORT,
                            "name=cublaslt_rank engine=time",
                        );
                        let tstream = stream.clone();
                        let mut scratch: CudaSlice<u16> = unsafe { tstream.alloc(c.len()) }
                            .map_err(|e| format!("rank scratch alloc: {e}"))?;
                        let pick = self
                            .cublaslt_time_candidates_f16(
                                st,
                                &tstream,
                                &cands,
                                a,
                                b,
                                &mut scratch,
                                m,
                                n,
                                k,
                                layout,
                            )
                            .unwrap_or(cands[0]);
                        st.algo_cache.lock().unwrap().insert(key, pick);
                        Some(pick)
                    } else if crate::backends::cuda_exec::capturing() {
                        Some(cands[0])
                    } else {
                        crate::backends::cuda_exec::report_cublaslt_rank_heuristic_once();
                        st.algo_cache.lock().unwrap().insert(key, cands[0]);
                        Some(cands[0])
                    }
                }
            };
            let Some(algo) = algo else { return Ok(false) };
            self.cublaslt_exec_f16(st, stream, &algo, a, b, c, m, n, k, layout)?;
            Ok(true)
        }

        /// C1:候选逐个计时(3 预热 + 12 计时,beta=0 写 scratch),返回最快者。
        /// 仅相对排名有意义;首选与最优差 <2% 时不打日志。
        #[allow(clippy::too_many_arguments)]
        fn cublaslt_time_candidates_f32(
            &self,
            st: &CublasLtState,
            stream: &Arc<cudarc::driver::CudaStream>,
            cands: &[cudarc::cublaslt::sys::cublasLtMatmulAlgo_t],
            a: &CudaSlice<u16>,
            b: &CudaSlice<u16>,
            scratch: &mut CudaSlice<f32>,
            m: usize,
            n: usize,
            k: usize,
            layout: CublasLtWeightLayout,
        ) -> Option<cudarc::cublaslt::sys::cublasLtMatmulAlgo_t> {
            let mut best: Option<(f64, cudarc::cublaslt::sys::cublasLtMatmulAlgo_t)> = None;
            let mut first_ms = f64::NAN;
            for (i, cand) in cands.iter().enumerate() {
                let mut failed = false;
                for _ in 0..3 {
                    if self
                        .cublaslt_exec(st, stream, cand, a, b, scratch, m, n, k, 0.0, layout)
                        .is_err()
                    {
                        failed = true;
                        break;
                    }
                }
                if failed || stream.synchronize().is_err() {
                    continue;
                }
                let t0 = std::time::Instant::now();
                for _ in 0..12 {
                    if self
                        .cublaslt_exec(st, stream, cand, a, b, scratch, m, n, k, 0.0, layout)
                        .is_err()
                    {
                        failed = true;
                        break;
                    }
                }
                if failed || stream.synchronize().is_err() {
                    continue;
                }
                let ms = t0.elapsed().as_secs_f64() * 1000.0 / 12.0;
                if i == 0 {
                    first_ms = ms;
                }
                if best.as_ref().map(|(t, _)| ms < *t).unwrap_or(true) {
                    best = Some((ms, *cand));
                }
            }
            if let Some((bms, _)) = &best {
                if first_ms.is_finite() && *bms < first_ms * 0.98 {
                    eprintln!(
                        "[cublaslt-rank] m={m} n={n} k={k} f32out: #0 {first_ms:.4}ms → best {:.4}ms (-{:.0}%)",
                        bms,
                        (first_ms - bms) / first_ms * 100.0
                    );
                }
            }
            best.map(|(_, a)| a)
        }

        /// f16 输出版本的候选计时(同 f32 版逻辑)。
        #[allow(clippy::too_many_arguments)]
        fn cublaslt_time_candidates_f16(
            &self,
            st: &CublasLtState,
            stream: &Arc<cudarc::driver::CudaStream>,
            cands: &[cudarc::cublaslt::sys::cublasLtMatmulAlgo_t],
            a: &CudaSlice<u16>,
            b: &CudaSlice<u16>,
            scratch: &mut CudaSlice<u16>,
            m: usize,
            n: usize,
            k: usize,
            layout: CublasLtWeightLayout,
        ) -> Option<cudarc::cublaslt::sys::cublasLtMatmulAlgo_t> {
            let mut best: Option<(f64, cudarc::cublaslt::sys::cublasLtMatmulAlgo_t)> = None;
            let mut first_ms = f64::NAN;
            for (i, cand) in cands.iter().enumerate() {
                let mut failed = false;
                for _ in 0..3 {
                    if self
                        .cublaslt_exec_f16(st, stream, cand, a, b, scratch, m, n, k, layout)
                        .is_err()
                    {
                        failed = true;
                        break;
                    }
                }
                if failed || stream.synchronize().is_err() {
                    continue;
                }
                let t0 = std::time::Instant::now();
                for _ in 0..12 {
                    if self
                        .cublaslt_exec_f16(st, stream, cand, a, b, scratch, m, n, k, layout)
                        .is_err()
                    {
                        failed = true;
                        break;
                    }
                }
                if failed || stream.synchronize().is_err() {
                    continue;
                }
                let ms = t0.elapsed().as_secs_f64() * 1000.0 / 12.0;
                if i == 0 {
                    first_ms = ms;
                }
                if best.as_ref().map(|(t, _)| ms < *t).unwrap_or(true) {
                    best = Some((ms, *cand));
                }
            }
            if let Some((bms, _)) = &best {
                if first_ms.is_finite() && *bms < first_ms * 0.98 {
                    eprintln!(
                        "[cublaslt-rank] m={m} n={n} k={k} f16out: #0 {first_ms:.4}ms → best {:.4}ms (-{:.0}%)",
                        bms,
                        (first_ms - bms) / first_ms * 100.0
                    );
                }
            }
            best.map(|(_, a)| a)
        }

        /// 选算法:heuristic 查询 top-8,按原顺序返回(索引 0 = 启发式首选)。
        /// f16out=true 时 C/D 布局为 CUDA_R_16F(qkv packed / dual FFN 用)。
        fn cublaslt_select_algos(
            &self,
            st: &CublasLtState,
            m: usize,
            n: usize,
            k: usize,
            f16out: bool,
            layout: CublasLtWeightLayout,
        ) -> Result<Vec<cudarc::cublaslt::sys::cublasLtMatmulAlgo_t>, String> {
            use cudarc::cublaslt::sys;
            unsafe {
                let mut desc: sys::cublasLtMatmulDesc_t = std::ptr::null_mut();
                sys::cublasLtMatmulDescCreate(
                    &mut desc,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    sys::cudaDataType_t::CUDA_R_32F,
                );
                let (transa, weight_rows, weight_cols, weight_ld) = layout.descriptor(n, k);
                let transb: u32 = 0; // CUBLAS_OP_N
                sys::cublasLtMatmulDescSetAttribute(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                    &transa as *const _ as *const _,
                    std::mem::size_of_val(&transa),
                );
                sys::cublasLtMatmulDescSetAttribute(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                    &transb as *const _ as *const _,
                    std::mem::size_of_val(&transb),
                );
                // C_cm[N,M] = op(W_cm) @ X_cm[K,M]. TN uses W_cm[K,N],
                // NN uses the load-time transposed W_cm[N,K]; X/C are unchanged.
                let cdtype = if f16out {
                    sys::cudaDataType_t::CUDA_R_16F
                } else {
                    sys::cudaDataType_t::CUDA_R_32F
                };
                let mut a_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                let mut b_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                let mut c_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                sys::cublasLtMatrixLayoutCreate(
                    &mut a_lay,
                    sys::cudaDataType_t::CUDA_R_16F,
                    weight_rows,
                    weight_cols,
                    weight_ld,
                );
                sys::cublasLtMatrixLayoutCreate(
                    &mut b_lay,
                    sys::cudaDataType_t::CUDA_R_16F,
                    k as u64,
                    m as u64,
                    k as i64,
                );
                sys::cublasLtMatrixLayoutCreate(&mut c_lay, cdtype, n as u64, m as u64, n as i64);
                let mut pref: sys::cublasLtMatmulPreference_t = std::ptr::null_mut();
                sys::cublasLtMatmulPreferenceCreate(&mut pref);
                let ws_size = st.workspace_len;
                sys::cublasLtMatmulPreferenceSetAttribute(
                    pref,
                    sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    &ws_size as *const _ as *const _, std::mem::size_of_val(&ws_size));
                const REQ: usize = 8;
                let mut heur: Vec<sys::cublasLtMatmulHeuristicResult_t> =
                    vec![std::mem::zeroed(); REQ];
                let mut cnt = 0i32;
                let r = sys::cublasLtMatmulAlgoGetHeuristic(
                    st.handle,
                    desc,
                    a_lay,
                    b_lay,
                    c_lay,
                    c_lay,
                    pref,
                    REQ as i32,
                    heur.as_mut_ptr(),
                    &mut cnt,
                );
                sys::cublasLtMatmulDescDestroy(desc);
                sys::cublasLtMatrixLayoutDestroy(a_lay);
                sys::cublasLtMatrixLayoutDestroy(b_lay);
                sys::cublasLtMatrixLayoutDestroy(c_lay);
                sys::cublasLtMatmulPreferenceDestroy(pref);
                if r != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS || cnt == 0 {
                    return Ok(Vec::new());
                }
                Ok(heur
                    .into_iter()
                    .take(cnt as usize)
                    .filter(|h| h.workspaceSize <= ws_size)
                    .map(|h| h.algo)
                    .collect())
            }
        }

        /// 执行 cuBLASLt GEMM(已选算法)。
        #[allow(clippy::too_many_arguments)]
        fn cublaslt_exec(
            &self,
            st: &CublasLtState,
            stream: &Arc<cudarc::driver::CudaStream>,
            algo: &cudarc::cublaslt::sys::cublasLtMatmulAlgo_t,
            a: &CudaSlice<u16>,
            b: &CudaSlice<u16>,
            c: &mut CudaSlice<f32>,
            m: usize,
            n: usize,
            k: usize,
            beta: f32,
            layout: CublasLtWeightLayout,
        ) -> Result<(), String> {
            use cudarc::cublaslt::sys;
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            unsafe {
                let mut desc: sys::cublasLtMatmulDesc_t = std::ptr::null_mut();
                sys::cublasLtMatmulDescCreate(
                    &mut desc,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    sys::cudaDataType_t::CUDA_R_32F,
                );
                let (transa, weight_rows, weight_cols, weight_ld) = layout.descriptor(n, k);
                let transb: u32 = 0; // CUBLAS_OP_N
                sys::cublasLtMatmulDescSetAttribute(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                    &transa as *const _ as *const _,
                    std::mem::size_of_val(&transa),
                );
                sys::cublasLtMatmulDescSetAttribute(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                    &transb as *const _ as *const _,
                    std::mem::size_of_val(&transb),
                );
                let mut a_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                let mut b_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                let mut c_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                sys::cublasLtMatrixLayoutCreate(
                    &mut a_lay,
                    sys::cudaDataType_t::CUDA_R_16F,
                    weight_rows,
                    weight_cols,
                    weight_ld,
                );
                sys::cublasLtMatrixLayoutCreate(
                    &mut b_lay,
                    sys::cudaDataType_t::CUDA_R_16F,
                    k as u64,
                    m as u64,
                    k as i64,
                );
                sys::cublasLtMatrixLayoutCreate(
                    &mut c_lay,
                    sys::cudaDataType_t::CUDA_R_32F,
                    n as u64,
                    m as u64,
                    n as i64,
                );
                let alpha = 1.0f32;
                let (a_ptr, _ga) = a.device_ptr(stream);
                let (b_ptr, _gb) = b.device_ptr(stream);
                let (c_ptr, _gc) = c.device_ptr_mut(stream);
                let (ws_ptr, _) = self.cublaslt_workspace_ptr(stream);
                if ws_ptr == 0 {
                    return Err("cublasLt workspace allocation failed".to_string());
                }
                let r = sys::cublasLtMatmul(
                    st.handle,
                    desc,
                    &alpha as *const _ as *const _,
                    b_ptr as *const _, // A_cm = B
                    a_lay,
                    a_ptr as *const _, // B_cm = A
                    b_lay,
                    &beta as *const _ as *const _,
                    c_ptr as *const _,
                    c_lay,
                    c_ptr as *mut _,
                    c_lay,
                    algo,
                    ws_ptr as *mut _,
                    st.workspace_len,
                    stream.cu_stream() as _,
                );
                sys::cublasLtMatmulDescDestroy(desc);
                sys::cublasLtMatrixLayoutDestroy(a_lay);
                sys::cublasLtMatrixLayoutDestroy(b_lay);
                sys::cublasLtMatrixLayoutDestroy(c_lay);
                if r != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    return Err(format!("cublasLtMatmul failed: {r:?}"));
                }
            }
            Ok(())
        }

        /// f16 输出的执行(Cdesc = CUDA_R_16F,C 为 f16)。
        #[allow(clippy::too_many_arguments)]
        fn cublaslt_exec_f16(
            &self,
            st: &CublasLtState,
            stream: &Arc<cudarc::driver::CudaStream>,
            algo: &cudarc::cublaslt::sys::cublasLtMatmulAlgo_t,
            a: &CudaSlice<u16>,
            b: &CudaSlice<u16>,
            c: &mut CudaSlice<u16>,
            m: usize,
            n: usize,
            k: usize,
            layout: CublasLtWeightLayout,
        ) -> Result<(), String> {
            use cudarc::cublaslt::sys;
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            unsafe {
                let mut desc: sys::cublasLtMatmulDesc_t = std::ptr::null_mut();
                sys::cublasLtMatmulDescCreate(
                    &mut desc,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    sys::cudaDataType_t::CUDA_R_32F,
                );
                let (transa, weight_rows, weight_cols, weight_ld) = layout.descriptor(n, k);
                let transb: u32 = 0;
                sys::cublasLtMatmulDescSetAttribute(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                    &transa as *const _ as *const _,
                    std::mem::size_of_val(&transa),
                );
                sys::cublasLtMatmulDescSetAttribute(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                    &transb as *const _ as *const _,
                    std::mem::size_of_val(&transb),
                );
                let mut a_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                let mut b_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                let mut c_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                sys::cublasLtMatrixLayoutCreate(
                    &mut a_lay,
                    sys::cudaDataType_t::CUDA_R_16F,
                    weight_rows,
                    weight_cols,
                    weight_ld,
                );
                sys::cublasLtMatrixLayoutCreate(
                    &mut b_lay,
                    sys::cudaDataType_t::CUDA_R_16F,
                    k as u64,
                    m as u64,
                    k as i64,
                );
                sys::cublasLtMatrixLayoutCreate(
                    &mut c_lay,
                    sys::cudaDataType_t::CUDA_R_16F,
                    n as u64,
                    m as u64,
                    n as i64,
                );
                let alpha = 1.0f32;
                let beta = 0.0f32;
                let (a_ptr, _ga) = a.device_ptr(stream);
                let (b_ptr, _gb) = b.device_ptr(stream);
                let (c_ptr, _gc) = c.device_ptr_mut(stream);
                let (ws_ptr, _) = self.cublaslt_workspace_ptr(stream);
                if ws_ptr == 0 {
                    return Err("cublasLt workspace allocation failed".to_string());
                }
                let r = sys::cublasLtMatmul(
                    st.handle,
                    desc,
                    &alpha as *const _ as *const _,
                    b_ptr as *const _,
                    a_lay,
                    a_ptr as *const _,
                    b_lay,
                    &beta as *const _ as *const _,
                    c_ptr as *const _,
                    c_lay,
                    c_ptr as *mut _,
                    c_lay,
                    algo,
                    ws_ptr as *mut _,
                    st.workspace_len,
                    stream.cu_stream() as _,
                );
                sys::cublasLtMatmulDescDestroy(desc);
                sys::cublasLtMatrixLayoutDestroy(a_lay);
                sys::cublasLtMatrixLayoutDestroy(b_lay);
                sys::cublasLtMatrixLayoutDestroy(c_lay);
                if r != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    return Err(format!("cublasLtMatmul f16 failed: {r:?}"));
                }
            }
            Ok(())
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

        /// 设置 persisting-L2 访问窗口（fork G6 复刻）：
        /// `cuCtxSetLimit(PERSISTING_L2_CACHE_SIZE)` + `cuStreamSetAttribute(
        /// ACCESS_POLICY_WINDOW)`。窗口内地址的访问按 PERSISTING 驻留 L2，
        /// 窗口外按 STREAMING。返回是否成功（不支持时静默跳过，不影响正确性）。
        /// G6 实验：当前 batch(≤16)工作集 ~13MB 本已驻留 48MB L2,无收益;
        /// 保留供大 batch/多模型场景。ABBA 实测 t=1/t=8 均持平。
        #[cfg(feature = "cuda")]
        #[allow(dead_code)]
        pub fn set_persisting_l2_window(
            &self,
            stream: &cudarc::driver::CudaStream,
            base_ptr: u64,
            num_bytes: usize,
        ) -> Result<(), String> {
            use cudarc::driver::sys;
            // 1. 提升 persisting L2 上限到窗口大小（钳制到设备上限）
            let max_persisting =
                self.device
                    .attribute(
                        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE,
                    )
                    .map_err(|e| format!("query max persisting L2: {e}"))? as usize;
            let window_cap =
                self.device
                    .attribute(
                        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE,
                    )
                    .map_err(|e| format!("query max access window: {e}"))? as usize;
            let eff_bytes = num_bytes.min(window_cap);
            let request = max_persisting.min(eff_bytes);
            self.device
                .set_limit(sys::CUlimit::CU_LIMIT_PERSISTING_L2_CACHE_SIZE, request)
                .map_err(|e| format!("set persisting L2 limit: {e}"))?;
            // 2. 设置流的访问策略窗口
            let mut value: sys::CUstreamAttrValue = unsafe { std::mem::zeroed() };
            unsafe {
                value.accessPolicyWindow = sys::CUaccessPolicyWindow {
                    base_ptr: base_ptr as *mut core::ffi::c_void,
                    num_bytes: eff_bytes,
                    hitRatio: 1.0,
                    hitProp: sys::CUaccessProperty::CU_ACCESS_PROPERTY_PERSISTING,
                    missProp: sys::CUaccessProperty::CU_ACCESS_PROPERTY_STREAMING,
                };
            }
            let res = unsafe {
                sys::cuStreamSetAttribute(
                    stream.cu_stream(),
                    sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW,
                    &value,
                )
            };
            if res != sys::CUresult::CUDA_SUCCESS {
                return Err(format!("cuStreamSetAttribute failed: {res:?}"));
            }
            Ok(())
        }

        /// 清除流的 persisting-L2 窗口（恢复正常访问属性）。
        #[cfg(feature = "cuda")]
        #[allow(dead_code)]
        pub fn clear_l2_window(&self, stream: &cudarc::driver::CudaStream) -> Result<(), String> {
            use cudarc::driver::sys;
            let mut value: sys::CUstreamAttrValue = unsafe { std::mem::zeroed() };
            unsafe {
                value.accessPolicyWindow = sys::CUaccessPolicyWindow {
                    base_ptr: core::ptr::null_mut(),
                    num_bytes: 0,
                    hitRatio: 0.0,
                    hitProp: sys::CUaccessProperty::CU_ACCESS_PROPERTY_NORMAL,
                    missProp: sys::CUaccessProperty::CU_ACCESS_PROPERTY_NORMAL,
                };
            }
            let res = unsafe {
                sys::cuStreamSetAttribute(
                    stream.cu_stream(),
                    sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW,
                    &value,
                )
            };
            if res != sys::CUresult::CUDA_SUCCESS {
                return Err(format!("clear L2 window failed: {res:?}"));
            }
            Ok(())
        }

        /// 冒烟/自检：`out[i] = a[i] + b[i]`。
        pub fn f32_add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
            assert_eq!(a.len(), b.len());
            let n = a.len();
            let f = self.get_func("f32_add_kernel")?;
            let stream = self.device.default_stream();
            let mut d_a: CudaSlice<f32> = unsafe { stream.alloc(n) }.map_err(|e| e.to_string())?;
            let mut d_b: CudaSlice<f32> = unsafe { stream.alloc(n) }.map_err(|e| e.to_string())?;
            stream.memcpy_htod(a, &mut d_a).map_err(|e| e.to_string())?;
            stream.memcpy_htod(b, &mut d_b).map_err(|e| e.to_string())?;
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
        /// 16 的倍数（调用方负责 padding）。对应 `cuda-kernels/gemm_v2.cu` 的
        /// `hgemm_v2_kernel`（tile 128×128×32 + smem 双缓冲 + cp.async +
        /// 内联 PTX mma.sync m16n8k16）。v1（`hgemm_m16n8k16_kernel`，无 smem）
        /// 保留在 gemm.cu 中，ABBA 对比后退位。
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

            let f = self.get_func("hgemm_v2_kernel")?;
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
            stream.memcpy_htod(c, &mut d_c).map_err(|e| e.to_string())?;

            let grid = (m.div_ceil(128) as u32, n.div_ceil(128) as u32, 1u32);
            let block = (8 * 32) as u32;
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

            stream.memcpy_dtoh(&d_c, c).map_err(|e| e.to_string())?;
            Ok(())
        }
    }

    /// f32 → f16 位模式（round-to-nearest-even，与 CUDA `__float2half` 一致）。
    pub fn f32_to_f16_bits(x: f32) -> u16 {
        let b = x.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32;
        let mant = b & 0x7fffff;
        if exp == 0xff {
            return if mant != 0 {
                sign | 0x7e00
            } else {
                sign | 0x7c00
            };
        }
        let e = exp - 127 + 15;
        if e >= 0x1f {
            return sign | 0x7c00;
        }
        if e <= 0 {
            if e < -10 {
                return sign;
            }
            // Half subnormals have a fixed 2^-24 spacing. Shift the full
            // significand once, then round using all discarded bits. A carry
            // to 0x400 correctly becomes the minimum normal half value.
            let significand = mant | 0x800000;
            let shift = (14 - e) as u32;
            let m = significand >> shift;
            let discarded = significand & ((1u32 << shift) - 1);
            let midpoint = 1u32 << (shift - 1);
            let m = m + u32::from(discarded > midpoint || (discarded == midpoint && (m & 1) != 0));
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

    #[cfg(test)]
    mod half_encoding_tests {
        use super::{f16_to_f32_bits, f32_to_f16_bits};

        // Build the exact positive binary16 number line from its spacing, using
        // FP64 arithmetic. This oracle does not use either production converter
        // or repeat the FP32 significand-shift/rounding implementation.
        fn positive_finite_half_values() -> Vec<f32> {
            let mut values = Vec::with_capacity(0x7c00);
            let subnormal_step = 2.0f64.powi(-24);
            for fraction in 0..1024 {
                values.push((fraction as f64 * subnormal_step) as f32);
            }
            for exponent in -14..=15 {
                let step = 2.0f64.powi(exponent - 10);
                for significand in 1024..2048 {
                    values.push((significand as f64 * step) as f32);
                }
            }
            values
        }

        #[test]
        fn every_finite_half_roundtrips_without_a_gpu() {
            let values = positive_finite_half_values();
            assert_eq!(values.len(), 0x7c00);
            for (magnitude, &positive) in values.iter().enumerate() {
                for (value, sign) in [(positive, 0u16), (-positive, 0x8000)] {
                    let bits = magnitude as u16 | sign;
                    assert_eq!(
                        f16_to_f32_bits(bits).to_bits(),
                        value.to_bits(),
                        "half decode {bits:#06x}"
                    );
                    assert_eq!(f32_to_f16_bits(value), bits, "half encode {bits:#06x}");
                }
            }
        }

        #[test]
        fn every_finite_half_midpoint_and_neighboring_f32_rounds_to_even() {
            let values = positive_finite_half_values();
            for (low, pair) in values.windows(2).enumerate() {
                // Every adjacent-half midpoint is exactly representable in
                // FP32. Its immediate FP32 neighbors lie on opposite sides.
                let midpoint = ((pair[0] as f64 + pair[1] as f64) * 0.5) as f32;
                let low = low as u16;
                let even = if low & 1 == 0 { low } else { low + 1 };
                for (value, expected) in [
                    (f32::from_bits(midpoint.to_bits() - 1), low),
                    (midpoint, even),
                    (f32::from_bits(midpoint.to_bits() + 1), low + 1),
                ] {
                    assert_eq!(
                        f32_to_f16_bits(value),
                        expected,
                        "positive midpoint neighbor {:#010x}",
                        value.to_bits()
                    );
                    assert_eq!(
                        f32_to_f16_bits(-value),
                        expected | 0x8000,
                        "negative midpoint neighbor {:#010x}",
                        (-value).to_bits()
                    );
                }
            }
        }

        #[test]
        fn half_underflow_normal_boundary_overflow_and_special_values() {
            // Known IEEE encodings cover sticky-bit rounding, signed zero,
            // subnormal-to-normal carry, overflow ties and existing NaN policy.
            for (fp32_bits, half_bits) in [
                (0x00000000u32, 0x0000u16),
                (0x00000001, 0x0000), // smallest FP32 subnormal
                (0x007fffff, 0x0000), // largest FP32 subnormal
                (0x00800000, 0x0000), // smallest FP32 normal
                (0x32ffffff, 0x0000), // just below 2^-25
                (0x33000000, 0x0000), // 2^-25: tie to signed zero
                (0x33000001, 0x0001), // just above underflow midpoint
                (0x33800000, 0x0001), // 2^-24: smallest half subnormal
                (0x38000000, 0x0200), // 2^-15: formerly encoded as 0x0100
                (0x387fc000, 0x03ff), // largest half subnormal
                (0x387fdfff, 0x03ff), // just below minimum-normal midpoint
                (0x387fe000, 0x0400), // tie rounds into minimum normal
                (0x387fe001, 0x0400),
                (0x38800000, 0x0400), // 2^-14: smallest half normal
                (0x3f800000, 0x3c00), // 1.0
                (0x477fe000, 0x7bff), // 65504: largest finite half
                (0x477fefff, 0x7bff), // just below overflow midpoint
                (0x477ff000, 0x7c00), // 65520: overflow tie
                (0x477ff001, 0x7c00),
                (0x7f7fffff, 0x7c00), // largest finite FP32
                (0x7f800000, 0x7c00), // infinity
                (0x7f800001, 0x7e00), // signaling NaN: existing canonicalization
                (0x7fc00000, 0x7e00), // quiet NaN
            ] {
                for sign in [0u32, 0x80000000] {
                    assert_eq!(
                        f32_to_f16_bits(f32::from_bits(fp32_bits | sign)),
                        half_bits | ((sign >> 16) as u16),
                        "boundary FP32 {:#010x}",
                        fp32_bits | sign
                    );
                }
            }
        }
    }

    impl CudaRuntime {
        /// 通用 1D f16 逐元素 kernel：`in -> out`（单输入单输出）。
        fn run_elem1(&self, kernel: &str, x: &[u16], n: usize) -> Result<Vec<u16>, String> {
            let f = self.get_func(kernel)?;
            let stream = self.device.default_stream();
            let mut d_in: CudaSlice<u16> = unsafe { stream.alloc(n) }.map_err(|e| e.to_string())?;
            let mut d_out: CudaSlice<u16> = stream.alloc_zeros(n).map_err(|e| e.to_string())?;
            stream
                .memcpy_htod(x, &mut d_in)
                .map_err(|e| e.to_string())?;
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
            stream
                .memcpy_dtoh(&d_out, &mut out)
                .map_err(|e| e.to_string())?;
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
            let mut d_in: CudaSlice<u16> = unsafe { stream.alloc(n) }.map_err(|e| e.to_string())?;
            let mut d_res: CudaSlice<u16> =
                unsafe { stream.alloc(n) }.map_err(|e| e.to_string())?;
            stream
                .memcpy_htod(xh.as_slice(), &mut d_in)
                .map_err(|e| e.to_string())?;
            stream
                .memcpy_htod(rh.as_slice(), &mut d_res)
                .map_err(|e| e.to_string())?;
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
            stream
                .memcpy_dtoh(&d_res, &mut out)
                .map_err(|e| e.to_string())?;
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
            let mut d_y: CudaSlice<u16> = stream.alloc_zeros(x.len()).map_err(|e| e.to_string())?;
            stream
                .memcpy_htod(xh.as_slice(), &mut d_x)
                .map_err(|e| e.to_string())?;
            stream
                .memcpy_htod(scale, &mut d_s)
                .map_err(|e| e.to_string())?;
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
            stream
                .memcpy_dtoh(&d_y, &mut out)
                .map_err(|e| e.to_string())?;
            Ok(out.iter().map(|&b| f16_to_f32_bits(b)).collect())
        }

        /// 注意力 v3（warp-per-row + cp.async 分块）：`out = softmax(Q·K^T/sqrt(D))·V`，non-causal 无掩码。
        /// q/k/v：[B*H, S, D] 行主序 f32（主机侧转 half）。要求 S ≤ 512、D == 32。
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
            assert!(s <= 512, "attention_row v3 requires S <= 512");
            assert_eq!(d, 32, "attention_row v3 requires D=32");
            let f = self.get_func("attention_row_v3_kernel")?;
            let stream = self.device.default_stream();
            let mut d_q: CudaSlice<u16> = unsafe { stream.alloc(n) }.map_err(|e| e.to_string())?;
            let mut d_k: CudaSlice<u16> = unsafe { stream.alloc(n) }.map_err(|e| e.to_string())?;
            let mut d_v: CudaSlice<u16> = unsafe { stream.alloc(n) }.map_err(|e| e.to_string())?;
            let mut d_o: CudaSlice<u16> = stream.alloc_zeros(n).map_err(|e| e.to_string())?;
            stream
                .memcpy_htod(qh.as_slice(), &mut d_q)
                .map_err(|e| e.to_string())?;
            stream
                .memcpy_htod(kh.as_slice(), &mut d_k)
                .map_err(|e| e.to_string())?;
            stream
                .memcpy_htod(vh.as_slice(), &mut d_v)
                .map_err(|e| e.to_string())?;
            let block = 256u32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (((s + 7) / 8) as u32, bh as u32, 1),
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
                    .arg(&(bh as i32)) // heads：输出写 [b*s*H + h] 布局（bh 全为 batch 时=恒等）
                    .launch(cfg)
            }
            .map_err(|e| format!("attention launch failed: {e}"))?;
            let mut out = vec![0u16; n];
            stream
                .memcpy_dtoh(&d_o, &mut out)
                .map_err(|e| e.to_string())?;
            Ok(out.iter().map(|&b| f16_to_f32_bits(b)).collect())
        }
    }
}

/// 当前设备的 tactic-plan 指纹（`cuda-fingerprint` 子命令与 plan 校验共用）。
/// `gpu_idx` 传 `None` 用默认设备。
#[cfg(feature = "cuda")]
pub fn current_device_fingerprint(
    gpu_idx: Option<i32>,
) -> Result<crate::tactic_plan::DeviceFingerprint, String> {
    let dev = match gpu_idx {
        Some(i) => cudarc::driver::CudaContext::new(i as usize).map_err(|e| e.to_string())?,
        None => cudarc::driver::CudaContext::new(0).map_err(|e| e.to_string())?,
    };
    device_fingerprint(&dev)
}

/// 当前设备的 tactic-plan 指纹（`cuda-fingerprint` 子命令与 plan 校验共用）。
#[cfg(feature = "cuda")]
pub fn device_fingerprint(
    dev: &cudarc::driver::CudaContext,
) -> Result<crate::tactic_plan::DeviceFingerprint, String> {
    use cudarc::driver::sys::CUdevice_attribute as Attr;
    let major = dev
        .attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(|e| format!("query cc major: {e}"))?;
    let minor = dev
        .attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .map_err(|e| format!("query cc minor: {e}"))?;
    Ok(crate::tactic_plan::DeviceFingerprint {
        gpu_name: dev.name().map_err(|e| format!("query device name: {e}"))?,
        compute_capability: format!("{major}.{minor}"),
        sm_count: dev
            .attribute(Attr::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(|e| format!("query sm count: {e}"))? as u32,
        l2_cache_bytes: dev
            .attribute(Attr::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
            .map_err(|e| format!("query l2 size: {e}"))? as u64,
    })
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
pub use imp::{
    CublasLtWeightLayout, CudaRuntime, attention_q64_available, attention_q64_serial_available,
    backend_build_fingerprint, f16_to_f32_bits, f32_to_f16_bits, graph_instantiate_flags,
};

// ---------------------------------------------------------------------------
// Backend trait 对接：把 cuda_exec 的整图执行器接入 NN 求值器
// ---------------------------------------------------------------------------
#[cfg(feature = "cuda")]
mod backend_impl {
    use super::imp::{CudaRuntime, graph_instantiate_flags};
    use crate::backend::{
        Backend, ComputeContext, ComputeHandle, Enabled, InputBuffers, LoadedModel, NNOutput,
        NNResultBuf, NeuralNetError,
    };
    use crate::backends::cuda_exec::{CudaModel, CudaOutputsHost, CudaWorkspace, set_capturing};
    use crate::desc::ModelDesc;
    use kata_core::config::Config;
    use kata_core::logger::Logger;
    use kata_game::symmetry::copy_inputs_with_symmetry;
    use std::any::Any;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    /// CUDA 后端（手写 kernel，sm-targets.json 选择编译目标）。
    pub struct CudaBackend { int8: bool, quantized: bool }

    // Preserve the existing unit-value API used by callers while selecting
    // precision explicitly through a separate backend value.
    #[allow(non_upper_case_globals)]
    pub const CudaBackend: CudaBackend = CudaBackend { int8: false, quantized: false };
    #[allow(non_upper_case_globals)]
    pub const CudaInt8Backend: CudaBackend = CudaBackend { int8: true, quantized: false };
    /// Immutable per-projection precision recipe on the shared CUDA executor.
    #[allow(non_upper_case_globals)]
    pub const CudaQuantBackend: CudaBackend = CudaBackend { int8: false, quantized: true };

    /// Identify the actual loaded recipe and its execution environment. This is
    /// not a numerical/performance certification: it is the immutable key to
    /// which those separately collected records must refer.
    fn quantized_profile_id(
        model: &CudaLoadedModel,
        execution_model: &CudaModel,
        recipe_sha256: &str,
        cfg: &Config,
    ) -> Result<String, NeuralNetError> {
        use sha2::{Digest, Sha256};
        use std::collections::BTreeMap;
        static EXECUTABLE_SHA: std::sync::OnceLock<Result<String, String>> = std::sync::OnceLock::new();
        let executable_sha = EXECUTABLE_SHA.get_or_init(|| {
            let path = std::env::current_exe().map_err(|e| format!("quantized executable identity: {e}"))?;
            crate::tactic_plan::sha256_file(&path)
        }).as_ref().map_err(|e| NeuralNetError(e.clone()))?;
        let device = super::device_fingerprint(&model.rt.device).map_err(NeuralNetError)?;
        let mut tactics = BTreeMap::new();
        for key in crate::tactic_plan::ALLOWED_TACTIC_KEYS.iter().copied().chain([
            "KATAGO_CUDA_INT8_GEMM_TUNE", "KATAGO_CUDA_INT8_RMS_FUSION",
        ]) {
            let value = match crate::tactic_plan::tactic_var(key) {
                Ok(value) => Some(value),
                Err(std::env::VarError::NotPresent) => None,
                Err(_) => return Err(NeuralNetError(format!("non-Unicode tactic {key}"))),
            };
            tactics.insert(key, value);
        }
        let mut execution_config = BTreeMap::new();
        for key in ["nnMaxBatchSize", "numNNServerThreadsPerModel", "nnUseFP16", "nnUseNHWC"] {
            let value = if cfg.contains(key) {
                Some(cfg.get_string(key).map_err(|e| NeuralNetError(e.to_string()))?)
            } else { None };
            execution_config.insert(key, value);
        }
        let mut driver_version = 0;
        let status = unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut driver_version) };
        if status != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
            return Err(NeuralNetError(format!("quantized driver identity: {status:?}")));
        }
        let contract = serde_json::json!({
            "schema": "rustgo-quantized-execution-v1",
            "model_sha256": model.source_model_sha256,
            "recipe_sha256": recipe_sha256,
            "executable_sha256": executable_sha,
            "backend_build": super::backend_build_fingerprint(),
            "device": {"gpu_name": device.gpu_name, "compute_capability": device.compute_capability,
                "sm_count": device.sm_count, "l2_cache_bytes": device.l2_cache_bytes},
            "driver_version": driver_version,
            "tactics": tactics,
            "actual_execution": {
                "mxfp8_scale_clear_fusion": execution_model.mxfp8_scale_clear_fusion_enabled(),
            },
            "execution_config": execution_config,
        });
        let bytes = serde_json::to_vec(&contract).map_err(|e| NeuralNetError(e.to_string()))?;
        Ok(format!("rustgo-quant-v1:{}", hex::encode(Sha256::digest(&bytes))))
    }

    #[cfg(test)]
    use crate::output_postprocess::policy_channel;

    fn validate_expected_quant_profile(cfg: &Config, actual: &str) -> Result<(), NeuralNetError> {
        if cfg.contains("cudaQuantExpectedProfile") {
            let expected = cfg.get_string("cudaQuantExpectedProfile")
                .map_err(|e| NeuralNetError(e.to_string()))?;
            if expected != actual {
                return Err(NeuralNetError(format!("quantized inference profile mismatch: expected {expected}, actual {actual}")));
            }
        }
        Ok(())
    }

    /// Bind facts from the successful upload and the frozen policy it uses.
    /// This does not assert that graph capture/fallback chose one kernel for
    /// every shape. Timing-based algorithm caches need a future observed key.
    fn legacy_execution_profile_id(
        model: &CudaLoadedModel, uploaded: &CudaModel, cfg: &Config,
        nn_x_len: i32, nn_y_len: i32,
        tactics: &std::collections::BTreeMap<String, Option<String>>,
    ) -> Result<Option<String>, NeuralNetError> {
        use std::collections::BTreeMap;
        if crate::execution_identity::timing_selected(tactics)
            || !cfg.contains("rustgoResolvedInputsUseNHWC")
            || !cfg.contains("rustgoResolvedRequireExactNNLen") {
            return Ok(None);
        }
        static EXECUTABLE_SHA: std::sync::OnceLock<Result<String, String>> = std::sync::OnceLock::new();
        let executable_sha256 = EXECUTABLE_SHA.get_or_init(|| {
            let path = std::env::current_exe().map_err(|e| e.to_string())?;
            crate::tactic_plan::sha256_file(&path)
        }).as_ref().map_err(|e| NeuralNetError(e.clone()))?.clone();
        let device = super::device_fingerprint(&model.rt.device).map_err(NeuralNetError)?;
        let uuid = model.rt.device.uuid().map_err(|e| NeuralNetError(format!("CUDA device UUID: {e}")))?;
        let uuid_bytes: Vec<u8> = uuid.bytes.iter().map(|b| *b as u8).collect();
        let mut driver_version = 0;
        let status = unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut driver_version) };
        if status != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
            return Err(NeuralNetError(format!("CUDA driver identity: {status:?}")));
        }
        let mut execution_config = BTreeMap::new();
        for (key, min, max) in [("nnMaxBatchSize", 1, 65536), ("numNNServerThreadsPerModel", 1, 65536)] {
            execution_config.insert(key.into(), cfg.get_int(key, min, max)
                .map_err(|e| NeuralNetError(e.to_string()))?.to_string());
        }
        for key in ["rustgoResolvedInputsUseNHWC", "rustgoResolvedRequireExactNNLen"] {
            execution_config.insert(key.into(), cfg.get_bool(key).map_err(|e| NeuralNetError(e.to_string()))?.to_string());
        }
        execution_config.insert("nn_x_len".into(), nn_x_len.to_string());
        execution_config.insert("nn_y_len".into(), nn_y_len.to_string());
        let contract = crate::execution_identity::LegacyCudaExecution {
            model_sha256: model.source_model_sha256.clone(), executable_sha256,
            backend_mode: if model.int8 { "cuda-int8" } else { "cuda-fp16" }.into(),
            backend_build: serde_json::to_value(super::backend_build_fingerprint()).map_err(|e| NeuralNetError(e.to_string()))?,
            device: serde_json::json!({"uuid":hex::encode(uuid_bytes), "ordinal":model.rt.device.ordinal(),
                "gpu_name":device.gpu_name, "compute_capability":device.compute_capability,
                "sm_count":device.sm_count, "l2_cache_bytes":device.l2_cache_bytes}),
            driver_version, kernel_target: model.rt.target_id.clone(),
            installed_plan_id: crate::tactic_plan::installed_plan_id().map(str::to_owned),
            tactics: tactics.clone(), execution_config,
            loaded_execution: serde_json::json!({
                "gemm_layout": crate::backends::cuda_exec::selected_gemm_layout().map_err(NeuralNetError)?,
                "int8_scope":uploaded.int8_scope().map(|s| s.name()),
                "int8_min_ffn_width":uploaded.int8_min_ffn_width(), "int8_ffn_count":uploaded.int8_ffn_count(),
                "int8_quantization_version": if model.int8 { Some(crate::backends::int8::QUANTIZATION_VERSION) } else { None },
                "ffn_compact":uploaded.ffn_compact_enabled(),
                "dual_ffn_handle_ready":uploaded.dual_ffn_handle_ready(),
                "dual_ffn_effective":crate::backends::cuda_exec::dual_ffn_enabled(),
                "cublaslt_available":model.rt.cublaslt_handle().is_some(),
                "cublaslt_workspace_bytes":model.rt.cublaslt_workspace_len(),
                "mixed_fp16_fp32":true,
            }),
        };
        contract.profile_id().map(Some).map_err(NeuralNetError)
    }

    // Diagnostic only: match the same encoded request, physical batch and row
    // across processes without exporting board contents. Disabled measurements
    // incur only a cached flag read; traced runs are never performance evidence.
    fn trace_completed_batch(physical_batch: usize, inputs: &[&mut NNResultBuf], host: &CudaOutputsHost) {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ENABLED.get_or_init(|| crate::tactic_plan::execution_env_var("KATAGO_CUDA_BATCH_TRACE").as_deref() == Ok("1")) { return; }
        use sha2::{Digest, Sha256};
        let rows = inputs.iter().enumerate().map(|(row, input)| {
            let mut encoded = Sha256::new();
            for values in [&input.row_spatial_buf, &input.row_global_buf] {
                encoded.update((values.len() as u64).to_le_bytes());
                for value in values { encoded.update(value.to_bits().to_le_bytes()); }
            }
            encoded.update(input.symmetry.to_le_bytes());
            encoded.update(input.policy_optimism.to_bits().to_le_bytes());
            let mut raw = Sha256::new();
            for (values, width) in [(&host.policy, 6 * 362), (&host.value, 3),
                (&host.misc, 10), (&host.moremisc, 8), (&host.ownership, 361)] {
                for value in &values[row * width..(row + 1) * width] {
                    raw.update(value.to_bits().to_le_bytes());
                }
            }
            serde_json::json!({"row":row,"input_sha256":hex::encode(encoded.finalize()),
                "raw_output_sha256":hex::encode(raw.finalize())})
        }).collect::<Vec<_>>();
        eprintln!("[cuda-batch-trace] {}", serde_json::json!({"physical_batch":physical_batch,"logical_rows":inputs.len(),"rows":rows}));
    }

    #[cfg(test)]
    mod policy_tests {
        use super::policy_channel;

        #[test]
        fn channel_stride_includes_pass_for_every_batch_row() {
            let logits: Vec<f32> = (0..2 * 6 * 362).map(|i| i as f32).collect();
            for row in 0..2 {
                for ch in [0, 5] {
                    let slice = policy_channel(&logits, row, ch, 361);
                    let offset = (row * 6 + ch) * 362;
                    assert_eq!(slice.len(), 362);
                    assert_eq!(slice[0], offset as f32);
                    assert_eq!(slice[360], (offset + 360) as f32);
                    assert_eq!(slice[361], (offset + 361) as f32);
                }
            }
        }
    }

    /// Validate and, when requested, execute the real production DualFFN
    /// capability probe before any serve thread can submit inference.
    pub fn ensure_requested_cuda_capabilities(model: &CudaModel) -> Result<(), String> {
        crate::tactic_plan::validate_mxfp8_scale_clear_fusion_execution(
            crate::tactic_plan::mxfp8_scale_clear_fusion_requested()?,
            model.has_mxfp8(),
            model.mxfp8_scale_clear_fusion_enabled(),
        )?;
        if crate::tactic_plan::tactic_enabled("KATAGO_CUDA_DUALFFN") {
            let build = super::backend_build_fingerprint();
            let probe = if build.capabilities.dual_ffn {
                crate::backends::cuda_exec::dual_ffn_probe_once()
            } else {
                Err("backend was built without CUTLASS DualFFN".to_string())
            };
            let ready = probe.is_ok() && model.dual_ffn_handle_ready();
            if ready {
                crate::backends::cuda_exec::set_dual_ffn_runtime_ready(true);
                eprintln!(
                    "[cuda-tactic] name=dual_ffn requested=1 compiled=1 probe=pass handle=ready effective=1"
                );
            } else {
                let reason = probe
                    .err()
                    .unwrap_or_else(|| "model DualFFN handle creation failed".to_string());
                if crate::tactic_plan::installed_plan_id().is_some() || model.quantization_recipe_id().is_some() {
                    return Err(format!(
                        "precision recipe or certified plan requests DualFFN but it is unavailable: {reason}"
                    ));
                }
                crate::backends::cuda_exec::set_dual_ffn_runtime_ready(false);
                eprintln!(
                    "WARNING: [cuda-tactic] name=dual_ffn requested=1 compiled={} probe=fail effective=0 fallback=unfused reason={reason}",
                    u8::from(build.capabilities.dual_ffn)
                );
            }
        }
        Ok(())
    }

    /// Common admission for evaluator and direct recipe benchmarks. Recipe
    /// execution has fixed tactics; timing-based selection needs G2 artifacts.
    pub fn validate_quantized_runtime(rt: &CudaRuntime) -> Result<(), String> {
        crate::tactic_plan::freeze_quantized_tactics()?;
        crate::tactic_plan::mxfp8_scale_clear_fusion_requested()?;
        if crate::tactic_plan::tactic_var("KATAGO_CUDA_CUBLASLT_RANK").as_deref() == Ok("time")
            || crate::tactic_plan::tactic_enabled("KATAGO_CUDA_INT8_GEMM_TUNE") {
            return Err("cudaquantbackend v1 requires deterministic algorithm selection; runtime timing choices need a persisted execution plan".into());
        }
        if crate::tactic_plan::tactic_var("KATAGO_CUDA_CUBLASLT").as_deref() != Ok("0")
            && rt.cublaslt_handle().is_none() {
            return Err("cudaquantbackend requested cuBLASLt but its handle is unavailable".into());
        }
        Ok(())
    }

    /// Bind model-specific artifacts to the exact source bytes already parsed.
    pub fn validate_quantized_model_source(model_sha256: &str) -> Result<(), String> {
        if crate::tactic_plan::ffn_compact_requested()?
            && model_sha256 != crate::tactic_plan::FFN_COMPACT_MODEL {
            return Err("FFN compact source model SHA mismatch".into());
        }
        crate::tactic_plan::validate_outproj_model_sha(model_sha256)
    }

    /// GPU weights are uploaded only after the context's certified plan is
    /// installed. The parsed graph is discarded after the successful upload.
    enum CudaModelLoadState {
        Parsed(crate::onnx_parser::LayerGraph),
        Uploaded {
            model: Arc<CudaModel>,
            gemm_layout: &'static str,
            quant_recipe_source: Option<String>,
            inference_profile_id: Option<String>,
            execution_profile_id: Option<String>,
        },
    }

    /// Parsed model and CUDA runtime. Context creation completes the deferred
    /// upload, before any compute handle, warmup, graph capture or serve thread.
    pub struct CudaLoadedModel {
        int8: bool,
        quantized: bool,
        model_desc: ModelDesc,
        model: Mutex<CudaModelLoadState>,
        rt: Arc<CudaRuntime>,
        /// 模型文件路径（cudaTacticPlan 指纹校验时算 SHA-256 用）。
        model_path: String,
        /// Hash of the exact bytes parsed, including native compression.
        source_model_sha256: String,
    }

    impl LoadedModel for CudaLoadedModel {
        fn model_desc(&self) -> &ModelDesc {
            &self.model_desc
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn inference_profile_id(&self) -> Option<String> {
            match &*self.model.lock().ok()? {
                CudaModelLoadState::Uploaded { inference_profile_id, .. } => inference_profile_id.clone(),
                CudaModelLoadState::Parsed(_) => None,
            }
        }
        fn execution_profile_id(&self) -> Option<String> {
            match &*self.model.lock().ok()? {
                CudaModelLoadState::Uploaded { execution_profile_id, .. } => execution_profile_id.clone(),
                CudaModelLoadState::Parsed(_) => None,
            }
        }
    }

    /// 跨线程共享的推理上下文：模型 + 运行时 + 棋盘尺寸。
    pub struct CudaComputeContext {
        model: Arc<CudaModel>,
        rt: Arc<CudaRuntime>,
        nn_x_len: i32,
        nn_y_len: i32,
        quant_max_batch_size: Option<i32>,
        legacy_handle_identity: Option<(i32, bool)>,
    }

    impl ComputeContext for CudaComputeContext {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// per-slot 的 graph 状态：固定 batch 的输入缓冲 + 已捕获图 + 完成事件。
    /// 双槽（pipeline）：submit(N+1) 的 htod 与 graph(N) 执行重叠。
    struct CudaSlot {
        /// 已捕获图；None = capture 失败降级直连。
        graph: Option<cudarc::driver::CudaGraph>,
        /// Graph drops first; keep its MXFP8 descriptors, weights and runtime alive.
        _quantized_graph_resources: Option<crate::backends::mxfp8::Mxfp8GraphResources>,
        in_spatial: cudarc::driver::CudaSlice<f32>,
        in_global: cudarc::driver::CudaSlice<f32>,
        in_sp_pin: cudarc::driver::PinnedHostSlice<f32>,
        in_gl_pin: cudarc::driver::PinnedHostSlice<f32>,
        out_policy_pin: cudarc::driver::PinnedHostSlice<f32>,
        out_value_pin: cudarc::driver::PinnedHostSlice<f32>,
        out_misc_pin: cudarc::driver::PinnedHostSlice<f32>,
        out_moremisc_pin: cudarc::driver::PinnedHostSlice<f32>,
        out_own_pin: cudarc::driver::PinnedHostSlice<f32>,
        /// Per-submission snapshot: the next forward can reset the shared status.
        quantized_status_pin: Option<cudarc::driver::PinnedHostSlice<u32>>,
        /// 前向完成事件（record 在 dtoh 后）。
        done_event: Option<cudarc::driver::CudaEvent>,
    }

    /// per-handle 的 CUDA Graph 状态：按物理 batch 缓存的双槽 graph。
    /// 尺寸变化不再销毁旧状态——消除攒批流水线中尺寸抖动引发的重捕获
    /// 抖动；不同尺寸的批次可同时安全在途（各自独立 ws/pins/event，
    /// 同一流串行执行互不干扰）。
    struct CudaGraphState {
        by_size: std::collections::HashMap<usize, CudaBatchState>,
    }

    /// 某物理 batch 的双槽 graph + 共享工作区。
    struct CudaBatchState {
        slots: [CudaSlot; 2],
        /// 共享工作区（流串行执行，两 graph 复用同一中间缓冲安全）。
        ws: CudaWorkspace,
        /// 当前提交槽（0/1 交替）。
        cur: usize,
    }

    // 图与其中指针仅在其归属的 serve 线程内创建/launch；Mutex 只是
    // ComputeHandle 内部可变性的手段，跨线程移动本身安全。
    unsafe impl Send for CudaGraphState {}

    /// 每线程计算句柄：专属 non-blocking 流（多 server 线程可并行提交推理）。
    pub struct CudaComputeHandle {
        model: Arc<CudaModel>,
        rt: Arc<CudaRuntime>,
        stream: Arc<cudarc::driver::CudaStream>,
        /// 捕获的 graph + 固定地址输入缓冲 + 工作区（按 batch 惰性创建）。
        graph_state: Mutex<Option<CudaGraphState>>,
        /// 每个 handle 的实际 graph/direct launch 只报告一次。
        graph_launch_reported: AtomicBool,
        direct_launch_reported: AtomicBool,
        nn_x_len: i32,
        nn_y_len: i32,
        max_batch_size: i32,
        inputs_use_nhwc: bool,
        /// B16 凑满：n>1 时物理 batch 补齐到 max_batch_size（尾批复制）。
        pad_to_max: bool,
    }

    impl Drop for CudaComputeHandle {
        fn drop(&mut self) {
            // A caller may cancel after submit without finish and release the
            // loaded model/context first. Synchronize before Rust drops ANY
            // fields: `model` owns non-MX weights uploaded on another stream,
            // whereas each graph token only retains its MXFP8 resources.
            // Graph destruction itself does not wait for an in-flight launch.
            if let Err(error) = self.stream.synchronize() {
                // Drop cannot return an error, and even stderr failure must not
                // panic during unwinding. Do not imply pending outputs passed
                // the normal completion/status validation on this error path.
                use std::io::Write;
                let _ = writeln!(
                    std::io::stderr().lock(),
                    "[cuda-handle-drop] owner-stream synchronization failed: {error}; outstanding outputs were not validated"
                );
            }
        }
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

    impl CudaBackend {
        /// 事件门控 pipeline —— 提交（非阻塞）：填 pinned → htod → graph/直连
        /// → dtoh → record done_event。返回槽位（供 finish_output）。
        fn submit_output(
            &self,
            h: &CudaComputeHandle,
            n: usize,
            input_bufs: &mut [&mut NNResultBuf],
        ) -> Result<usize, NeuralNetError> {
            let nn_x_len = h.nn_x_len;
            let nn_y_len = h.nn_y_len;
            let policy_area = (nn_x_len * nn_y_len) as usize;
            if policy_area != 361 {
                return Err(NeuralNetError(format!(
                    "CUDA backend only supports the 19x19 board (got {nn_x_len}x{nn_y_len})"
                )));
            }
            const NUM_SPATIAL_CHANNELS: i32 = 22;
            const NUM_GLOBAL_CHANNELS: usize = 19;
            let single_spatial = (NUM_SPATIAL_CHANNELS * nn_x_len * nn_y_len) as usize;
            // 物理 batch 策略（2026-08-15 ABBA 定论）：精确尺寸（per-size
            // graph 缓存使变尺寸零成本）。B16 凑满（KATAGO_CUDA_PADBATCH=1,
            // n>1 补齐到 handle 上限、尾批复制）已实测证伪——T(B) 曲线
            // 过陡：B16≈12.5ms vs B8≈11.6ms vs B4≈6.7ms vs B2≈4.4ms,
            // t=8 时 PAD 452 vs NOPAD 1022 visits/s。每行成本 B16(0.78ms)
            // 仅为 B4(1.7ms)一半，但中低并发下需求不足以填满大 batch。
            // n==1 保持小图（单线程延迟优先）。
            let phys_batch = if n == 1 || !h.pad_to_max {
                n
            } else {
                h.max_batch_size as usize
            };
            if phys_batch > n {
                crate::backends::cuda_exec::report_tactic_once(
                    &crate::backends::cuda_exec::PADBATCH_REPORT,
                    &format!("name=padbatch launch=on phys={phys_batch} rows={n}"),
                );
            }
            let mut spatial_host = vec![0.0f32; phys_batch * single_spatial];
            let mut global_host = vec![0.0f32; phys_batch * NUM_GLOBAL_CHANNELS];
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
            // 尾批复制 padding：重复最后一行真实数据（输出时丢弃 padding 行）。
            for i in n..phys_batch {
                let src = (n - 1) * single_spatial;
                let dst = i * single_spatial;
                spatial_host.copy_within(src..src + single_spatial, dst);
                let gsrc = (n - 1) * NUM_GLOBAL_CHANNELS;
                let gdst = i * NUM_GLOBAL_CHANNELS;
                global_host.copy_within(gsrc..gsrc + NUM_GLOBAL_CHANNELS, gdst);
            }

            let stream = &h.stream;
            let mut g = h.graph_state.lock().unwrap();
            let g = g.get_or_insert_with(|| CudaGraphState {
                by_size: std::collections::HashMap::new(),
            });
            let force_direct = crate::tactic_plan::tactic_enabled("KATAGO_CUDA_NOGRAPH");
            // per-size 缓存：新尺寸才创建（capture 期间全局互斥 + 设备清场）。
            // 已有尺寸直接复用，零重捕获；旧尺寸状态永久保留，在途批不失效。
            if !g.by_size.contains_key(&phys_batch) {
                // capture 期间全局互斥：WDDM/CUDA13 下同 context 其他流的并发
                // 活动会触发 STREAM_CAPTURE_INVALIDATED（实测 100% 复现）。
                // 仅创建期持有；正常运行路径（htod/launch/dtoh）不锁。
                static CUDA_EXEC_LOCK: Mutex<()> = Mutex::new(());
                let _rebuild_guard = CUDA_EXEC_LOCK.lock().unwrap();
                if !force_direct {
                    h.rt.device
                        .synchronize()
                        .map_err(|e| NeuralNetError(format!("pre-capture sync: {e}")))?;
                }
                let mut ws = CudaWorkspace::new_with_runtime(h.rt.clone(), stream, &h.model, phys_batch)
                    .map_err(|e| NeuralNetError(format!("CUDA workspace alloc failed: {e}")))?;
                let mut slot_vec: Vec<CudaSlot> = Vec::with_capacity(2);
                for slot_idx in 0..2 {
                    let mut in_spatial: cudarc::driver::CudaSlice<f32> =
                        unsafe { stream.alloc(spatial_host.len()) }
                            .map_err(|e| NeuralNetError(format!("alloc in_spatial: {e}")))?;
                    let mut in_global: cudarc::driver::CudaSlice<f32> =
                        unsafe { stream.alloc(global_host.len()) }
                            .map_err(|e| NeuralNetError(format!("alloc in_global: {e}")))?;
                    let in_sp_pin = unsafe { h.rt.device.alloc_pinned::<f32>(spatial_host.len()) }
                        .map_err(|e| NeuralNetError(format!("pinned in_sp: {e}")))?;
                    let in_gl_pin = unsafe { h.rt.device.alloc_pinned::<f32>(global_host.len()) }
                        .map_err(|e| NeuralNetError(format!("pinned in_gl: {e}")))?;
                    let out_policy_pin =
                        unsafe { h.rt.device.alloc_pinned::<f32>(phys_batch * 6 * 362) }
                            .map_err(|e| NeuralNetError(format!("pinned out_policy: {e}")))?;
                    let out_value_pin = unsafe { h.rt.device.alloc_pinned::<f32>(phys_batch * 3) }
                        .map_err(|e| NeuralNetError(format!("pinned out_value: {e}")))?;
                    let out_misc_pin = unsafe { h.rt.device.alloc_pinned::<f32>(phys_batch * 10) }
                        .map_err(|e| NeuralNetError(format!("pinned out_misc: {e}")))?;
                    let out_moremisc_pin =
                        unsafe { h.rt.device.alloc_pinned::<f32>(phys_batch * 8) }
                            .map_err(|e| NeuralNetError(format!("pinned out_moremisc: {e}")))?;
                    let out_own_pin =
                        unsafe { h.rt.device.alloc_pinned::<f32>(phys_batch * policy_area) }
                            .map_err(|e| NeuralNetError(format!("pinned out_own: {e}")))?;
                    let quantized_status_pin = if ws.has_mxfp8() {
                        Some(unsafe { h.rt.device.alloc_pinned::<u32>(crate::backends::mxfp8::STATUS_WORDS) }
                            .map_err(|e| NeuralNetError(format!("pinned quantized status: {e}")))?)
                    } else {
                        None
                    };
                    // 完成事件无条件创建：直连（NOGRAPH）模式下流水线
                    // 仍需要事件做完成门控（finish/query 不再退化）。
                    let done_event = Some(
                        h.rt.device
                            .new_event(Some(
                                cudarc::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC,
                            ))
                            .map_err(|e| NeuralNetError(format!("event create: {e}")))?,
                    );
                    let graph = if force_direct {
                        None
                    } else {
                        // 捕获前先直连跑一次(无条件,2026-08-24 修复):
                        // - C1(KATAGO_CUDA_CUBLASLT_RANK=time):计时重排在非
                        //   capture 语境完成算法选择并写缓存,capture 把优胜
                        //   kernel 烙进 graph;
                        // - B2(KATAGO_CUDA_DUALFFN=1):DualGemm 首调用的
                        //   initialize(cudaFuncSetAttribute 等)在 capture 外
                        //   完成;
                        // - 无 warm 时,进程内首次捕获的 graph exec 会被后续
                        //   同流捕获作废(CUDA lazy module loading × capture,
                        //   graph_launch_repro.rs 复现:首发 OK、二次捕获后
                        //   cuGraphLaunch 恒 CUDA_ERROR_INVALID_VALUE,重
                        //   upload 无效;生产表现为 warmup/首批 eval 丢弃)。
                        if slot_idx == 0 {
                            // Warmup may inspect data while selecting an INT8
                            // algorithm. Supply the actual request instead of
                            // reading the freshly allocated device buffers.
                            stream.memcpy_htod(&spatial_host, &mut in_spatial)
                                .map_err(|e| NeuralNetError(format!("warm spatial upload: {e}")))?;
                            stream.memcpy_htod(&global_host, &mut in_global)
                                .map_err(|e| NeuralNetError(format!("warm global upload: {e}")))?;
                            h.model
                                .apply(&h.rt, stream, &mut ws, &in_spatial, &in_global)
                                .map_err(|e| NeuralNetError(format!("pre-capture warm: {e}")))?;
                            ws.complete_quantized_warmup()
                                .map_err(|e| NeuralNetError(format!("pre-capture quantized validation: {e}")))?;
                            h.rt.device.synchronize().map_err(|e| {
                                NeuralNetError(format!("pre-capture warm sync: {e}"))
                            })?;
                        }
                        h.rt.device
                            .synchronize()
                            .map_err(|e| NeuralNetError(format!("pre-capture sync: {e}")))?;
                        set_capturing(true);
                        let cap_result =
                            (|| -> Result<cudarc::driver::CudaGraph, NeuralNetError> {
                                stream.begin_capture(
                        // RELAXED(2026-08-18):GLOBAL 模式下其他流任何并发活动都
                        // 使 capture 失效——serve=2 双流时双方 capture 互相打挂
                        // (Linux 亦然)。RELAXED 允许他流并发;本后端每 handle
                        // 单流、apply 内无跨流依赖,捕获内容不变,安全。
                        // Legacy backends may fall back to direct launch. A
                        // quantized profile binds this launch choice and fails
                        // closed if its requested Graph cannot be captured.
                        cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                    ).map_err(|e| NeuralNetError(format!("begin_capture: {e}")))?;
                                h.model
                                    .apply(&h.rt, stream, &mut ws, &in_spatial, &in_global)
                                    .map_err(|e| {
                                        NeuralNetError(format!(
                                            "CUDA forward (capture) failed: {e}"
                                        ))
                                    })?;
                                stream
                                    .end_capture(graph_instantiate_flags())
                                    .map_err(|e| NeuralNetError(format!("end_capture: {e}")))?
                                    .ok_or_else(|| {
                                        NeuralNetError("capture produced no graph".to_string())
                                    })
                            })();
                        set_capturing(false);
                        match cap_result {
                            Ok(g) => Some(g),
                            Err(e) => {
                                let _ = stream.end_capture(graph_instantiate_flags());
                                if h.model.quantization_recipe_id().is_some() {
                                    return Err(NeuralNetError(format!(
                                        "quantized profile requires CUDA Graph; capture failed: {e}"
                                    )));
                                }
                                eprintln!(
                                    "WARNING: CUDA graph capture failed ({e}); falling back to direct launch path"
                                );
                                None
                            }
                        }
                    };
                    if let Some(g) = graph.as_ref() {
                        g.upload()
                            .map_err(|e| NeuralNetError(format!("graph upload: {e}")))?;
                    }
                    slot_vec.push(CudaSlot {
                        graph,
                        _quantized_graph_resources: ws.quantized_graph_resources()
                            .map_err(|e| NeuralNetError(format!("quantized graph resources: {e}")))?,
                        in_spatial,
                        in_global,
                        in_sp_pin,
                        in_gl_pin,
                        out_policy_pin,
                        out_value_pin,
                        out_misc_pin,
                        out_moremisc_pin,
                        out_own_pin,
                        quantized_status_pin,
                        done_event,
                    });
                }
                let mut it = slot_vec.into_iter();
                let s0 = it.next().unwrap();
                let s1 = it.next().unwrap();
                g.by_size.insert(
                    phys_batch,
                    CudaBatchState {
                        slots: [s0, s1],
                        ws,
                        cur: 0,
                    },
                );
            }

            let st = g.by_size.get_mut(&phys_batch).unwrap();
            let slot = st.cur;
            st.cur ^= 1;
            let sl = &mut st.slots[slot];
            {
                let sp = sl
                    .in_sp_pin
                    .as_mut_slice()
                    .map_err(|e| NeuralNetError(format!("pin in_sp: {e}")))?;
                sp.copy_from_slice(&spatial_host);
                let gl = sl
                    .in_gl_pin
                    .as_mut_slice()
                    .map_err(|e| NeuralNetError(format!("pin in_gl: {e}")))?;
                gl.copy_from_slice(&global_host);
            }
            stream
                .memcpy_htod(&sl.in_sp_pin, &mut sl.in_spatial)
                .map_err(|e| NeuralNetError(format!("upload spatial: {e}")))?;
            stream
                .memcpy_htod(&sl.in_gl_pin, &mut sl.in_global)
                .map_err(|e| NeuralNetError(format!("upload global: {e}")))?;
            match sl.graph.as_ref() {
                Some(graph) => {
                    graph
                        .launch()
                        .map_err(|e| NeuralNetError(format!("graph launch: {e}")))?;
                    if !h.graph_launch_reported.swap(true, Ordering::Relaxed) {
                        eprintln!(
                            "[cuda-tactic] name=graph requested=graph launch=graph effective=graph batch={phys_batch}"
                        );
                    }
                }
                None => {
                    h.model
                        .apply(&h.rt, stream, &mut st.ws, &sl.in_spatial, &sl.in_global)
                        .map_err(|e| NeuralNetError(format!("CUDA forward failed: {e}")))?;
                    if !h.direct_launch_reported.swap(true, Ordering::Relaxed) {
                        eprintln!(
                            "[cuda-tactic] name=graph requested={} launch=direct effective=direct fallback={} batch={phys_batch}",
                            if force_direct { "direct" } else { "graph" },
                            u8::from(!force_direct),
                        );
                    }
                }
            }
            stream
                .memcpy_dtoh(&st.ws.out_policy, &mut sl.out_policy_pin)
                .map_err(|e| NeuralNetError(format!("dtoh policy: {e}")))?;
            stream
                .memcpy_dtoh(&st.ws.out_value, &mut sl.out_value_pin)
                .map_err(|e| NeuralNetError(format!("dtoh value: {e}")))?;
            stream
                .memcpy_dtoh(&st.ws.out_misc, &mut sl.out_misc_pin)
                .map_err(|e| NeuralNetError(format!("dtoh misc: {e}")))?;
            stream
                .memcpy_dtoh(&st.ws.out_moremisc, &mut sl.out_moremisc_pin)
                .map_err(|e| NeuralNetError(format!("dtoh moremisc: {e}")))?;
            stream
                .memcpy_dtoh(&st.ws.out_ownership, &mut sl.out_own_pin)
                .map_err(|e| NeuralNetError(format!("dtoh ownership: {e}")))?;
            if let Some(status) = &mut sl.quantized_status_pin {
                st.ws.enqueue_quantized_status_copy(status)
                    .map_err(|e| NeuralNetError(format!("quantized status snapshot: {e}")))?;
            }
            if let Some(ev) = sl.done_event.as_ref() {
                ev.record(stream)
                    .map_err(|e| NeuralNetError(format!("event record: {e}")))?;
            }
            // token = 物理 batch * 2 + 槽位（finish/query 据此定位 per-size 状态）。
            Ok(phys_batch * 2 + slot)
        }

        /// 完成：等 done_event → 读 pinned → 解码到 outputs。
        /// `token` = submit 返回值（物理 batch * 2 + 槽位）；`n` 为批行数。
        fn finish_output(
            &self,
            h: &CudaComputeHandle,
            token: usize,
            n: usize,
            input_bufs: &mut [&mut NNResultBuf],
            outputs: &mut [&mut NNOutput],
        ) -> Result<(), NeuralNetError> {
            let nn_x_len = h.nn_x_len;
            let nn_y_len = h.nn_y_len;
            let slot = token & 1;
            // 状态按物理 batch（token 高位）索引——B16 凑满时 n 是逻辑行数,
            // 与 phys_batch 不同,必须用 token 解码而非 n。
            let phys_batch = token >> 1;
            let mut g = h.graph_state.lock().unwrap();
            let st = g
                .as_mut()
                .and_then(|g| g.by_size.get_mut(&phys_batch))
                .ok_or_else(|| {
                    NeuralNetError(format!(
                        "finish: no graph state for phys batch {phys_batch}"
                    ))
                })?;
            let sl = &mut st.slots[slot];
            match sl.done_event.as_ref() {
                Some(ev) => ev
                    .synchronize()
                    .map_err(|e| NeuralNetError(format!("event sync: {e}")))?,
                None => h
                    .stream
                    .synchronize()
                    .map_err(|e| NeuralNetError(format!("sync: {e}")))?,
            }
            // Reject the entire submitted batch before decoding or exposing any output.
            // Never read shared workspace status here: another slot may already run.
            if let Some(status) = &sl.quantized_status_pin {
                CudaWorkspace::validate_quantized_status(status.as_slice()
                    .map_err(|e| NeuralNetError(format!("sync quantized status: {e}")))?)
                    .map_err(NeuralNetError)?;
            }
            let host = CudaOutputsHost {
                policy: sl
                    .out_policy_pin
                    .as_slice()
                    .map_err(|e| NeuralNetError(format!("sync policy: {e}")))?
                    .to_vec(),
                value: sl
                    .out_value_pin
                    .as_slice()
                    .map_err(|e| NeuralNetError(format!("sync value: {e}")))?
                    .to_vec(),
                misc: sl
                    .out_misc_pin
                    .as_slice()
                    .map_err(|e| NeuralNetError(format!("sync misc: {e}")))?
                    .to_vec(),
                moremisc: sl
                    .out_moremisc_pin
                    .as_slice()
                    .map_err(|e| NeuralNetError(format!("sync moremisc: {e}")))?
                    .to_vec(),
                ownership: sl
                    .out_own_pin
                    .as_slice()
                    .map_err(|e| NeuralNetError(format!("sync ownership: {e}")))?
                    .to_vec(),
            };
            drop(g);

            trace_completed_batch(phys_batch, &input_bufs[..n], &host);

            crate::output_postprocess::decode_raw_outputs(
                nn_x_len, nn_y_len, n, input_bufs,
                crate::output_postprocess::RawHeads { policy: &host.policy, value: &host.value,
                    misc: &host.misc, moremisc: &host.moremisc, ownership: &host.ownership },
                outputs,
            );
            Ok(())
        }

        /// 非阻塞查询批次是否完成（cuEventQuery）。token = 物理 batch*2+槽位。
        /// 状态缺失（不可能：submit 先于 query）或事件缺失（直连模式）视为完成。
        fn query_slot_done(&self, h: &CudaComputeHandle, token: usize) -> bool {
            let g = h.graph_state.lock().unwrap();
            let Some(st) = g.as_ref().and_then(|g| g.by_size.get(&(token >> 1))) else {
                return true;
            };
            let Some(ev) = st.slots[token & 1].done_event.as_ref() else {
                return true;
            };
            // 直接调 driver API：cudarc 0.19 的 CudaEvent 未暴露 query。
            // CUDA_ERROR_NOT_READY → false；其余错误按已完成处理
            // （错误会在 finish 的 event synchronize 中正式上报）。
            match unsafe { cudarc::driver::result::event::query(ev.cu_event()) } {
                Ok(()) => true,
                Err(e) => e.0 != cudarc::driver::sys::cudaError_enum::CUDA_ERROR_NOT_READY,
            }
        }
    }

    impl Backend for CudaBackend {
        fn global_initialize(&self) {}
        fn global_cleanup(&self) {}

        fn print_devices(&self) {
            println!("CUDA backend precision={} (SM120-first)", if self.quantized { "per-projection recipe" } else if self.int8 { "W8A8 mixed" } else { "FP16 mixed" });
        }

        fn load_model_file(
            &self,
            file: &str,
            expected_sha256: &str,
        ) -> Result<Box<dyn LoadedModel>, NeuralNetError> {
            let bytes = std::fs::read(file)
                .map_err(|e| NeuralNetError(format!("could not read {file}: {e}")))?;
            use sha2::{Digest, Sha256};
            let source_model_sha256 = hex::encode(Sha256::digest(&bytes));
            if !expected_sha256.is_empty() {
                if !source_model_sha256.eq_ignore_ascii_case(expected_sha256) {
                    return Err(NeuralNetError(format!(
                        "model SHA256 mismatch for {file}: expected {expected_sha256}, got {source_model_sha256}"
                    )));
                }
            }
            let lower_file = file.to_ascii_lowercase();
            let (model_desc, graph) = if lower_file.ends_with(".onnx") {
                let parsed = crate::onnx_model::parse_onnx_model(&bytes)
                    .map_err(|e| NeuralNetError(format!("onnx metadata parse failed: {e}")))?;
                let graph = crate::onnx_parser::parse_layer_graph(&bytes)
                    .map_err(|e| NeuralNetError(format!("layer graph parse failed: {e}")))?;
                (parsed.model_desc, graph)
            } else {
                let compressed = lower_file.ends_with(".gz");
                let inner = lower_file.strip_suffix(".gz").unwrap_or(&lower_file);
                let binary = if inner.ends_with(".bin") {
                    true
                } else if inner.ends_with(".txt") {
                    false
                } else {
                    return Err(NeuralNetError(
                        "CUDA model must be .onnx, .bin[.gz], or .txt[.gz]".into(),
                    ));
                };
                // Parse the exact bytes that were verified above. The worker
                // and tactic plan retain the original file's identity; this
                // in-memory lowering never substitutes an exported model.
                let desc =
                    crate::model_parser::load_model_from_bytes(&bytes, binary, compressed)
                        .map_err(|e| NeuralNetError(format!("native model parse failed: {e}")))?;
                let graph = crate::native_model::lower_model(&desc)
                    .map_err(|e| NeuralNetError(format!("native CUDA model unsupported: {e}")))?;
                (desc, graph)
            };
            let rt = Arc::new(
                CudaRuntime::new().map_err(|e| NeuralNetError(format!("CUDA init failed: {e}")))?,
            );
            // cudaTacticPlan is unavailable until create_compute_context.
            // Reading load-time tactics here would bind the environment first,
            // defeating both plan NN/env TN and plan TN/env NN precedence.
            // Retain the parsed graph and defer all weight uploads instead.
            Ok(Box::new(CudaLoadedModel {
                int8: self.int8,
                quantized: self.quantized,
                model_desc,
                model: Mutex::new(CudaModelLoadState::Parsed(graph)),
                rt,
                model_path: file.to_string(),
                source_model_sha256,
            }))
        }

        fn create_compute_context(
            &self,
            _gpu_idxs: &[i32],
            logger: &Logger,
            nn_x_len: i32,
            nn_y_len: i32,
            _home_data_dir_override: &str,
            _use_fp16_mode: Enabled,
            loaded_model: &dyn LoadedModel,
            cfg: &Config,
        ) -> Result<Box<dyn ComputeContext>, NeuralNetError> {
            let model = loaded_model
                .as_any()
                .downcast_ref::<CudaLoadedModel>()
                .ok_or_else(|| NeuralNetError("Wrong loaded model type".to_string()))?;
            if model.int8 != self.int8 || model.quantized != self.quantized {
                return Err(NeuralNetError("CUDA loaded model/backend precision mismatch".into()));
            }
            let quant_recipe_path = if self.quantized {
                for key in ["cudaTacticPlan", "cudaInt8Scope", "cudaInt8MinFfnWidth"] {
                    if cfg.contains(key) {
                        return Err(NeuralNetError(format!("cudaquantbackend uses cudaQuantPlan; incompatible option {key}")));
                    }
                }
                if crate::tactic_plan::installed_plan_id().is_some() {
                    return Err(NeuralNetError("cudaquantbackend cannot inherit an installed FP16 tactic plan; use a separate process".into()));
                }
                validate_quantized_runtime(&model.rt).map_err(NeuralNetError)?;
                if cfg.contains("cudaQuantPlan") {
                    let path = cfg.get_string("cudaQuantPlan").map_err(|e| NeuralNetError(e.to_string()))?;
                    if path.trim().is_empty() {
                        return Err(NeuralNetError("cudaQuantPlan must not be empty; omit it for the FP16 base recipe".into()));
                    }
                    Some(std::path::PathBuf::from(path))
                } else { None }
            } else {
                if cfg.contains("cudaQuantPlan") || cfg.contains("cudaQuantExpectedProfile") {
                    return Err(NeuralNetError("cudaQuantPlan/cudaQuantExpectedProfile requires nnBackend=cudaquantbackend".into()));
                }
                None
            };
            let quant_max_batch_size = if self.quantized {
                Some(cfg.get_int("nnMaxBatchSize", 1, 65536).map_err(|e| NeuralNetError(format!("cudaquantbackend requires resolved batch capacity: {e}")))?)
            } else { None };
            // Hash and resolve the same bytes, even if an external publisher
            // replaces the recipe path during context creation.
            let quant_recipe_bytes = quant_recipe_path.as_ref().map(|path| {
                std::fs::read(path).map_err(|e| NeuralNetError(format!("cudaQuantPlan {}: {e}", path.display())))
            }).transpose()?;
            let quant_recipe_source = if self.quantized {
                use sha2::{Digest, Sha256};
                Some(match &quant_recipe_bytes {
                    Some(bytes) => hex::encode(Sha256::digest(bytes)),
                    None => "builtin-fp16-v1".into(),
                })
            } else { None };
            let int8_scope = if self.int8 {
                if cfg.contains("cudaTacticPlan") {
                    return Err(NeuralNetError("cudaint8backend cannot reuse a FP16 cudaTacticPlan; omit it and validate the quantized model separately".into()));
                }
                let name = if cfg.contains("cudaInt8Scope") {
                    cfg.get_string("cudaInt8Scope").map_err(|e| NeuralNetError(e.to_string()))?
                } else { "ffn".to_string() };
                Some(crate::backends::int8::Int8Scope::parse(&name).map_err(NeuralNetError)?)
            } else {
                if cfg.contains("cudaInt8Scope") || cfg.contains("cudaInt8MinFfnWidth") {
                    return Err(NeuralNetError("cudaInt8Scope/cudaInt8MinFfnWidth requires nnBackend=cudaint8backend".into()));
                }
                None
            };
            let int8_min_ffn_width = if cfg.contains("cudaInt8MinFfnWidth") {
                crate::backends::int8::parse_min_ffn_width(&cfg.get_string("cudaInt8MinFfnWidth").map_err(|e| NeuralNetError(e.to_string()))?).map_err(NeuralNetError)?
            } else { 0 };
            // cudaTacticPlan：离线 autotune 认证 plan，fail-closed 安装
            // tactic 覆盖（先于 serve 线程 spawn 与一切 tactic 读取）。
            if cfg.contains("cudaTacticPlan") {
                let path = cfg
                    .get_string("cudaTacticPlan")
                    .map_err(|e| NeuralNetError(format!("cudaTacticPlan: {e}")))?;
                let fp =
                    super::device_fingerprint(&model.rt.device).map_err(|e| NeuralNetError(e))?;
                let model_sha =
                    crate::tactic_plan::sha256_file(std::path::Path::new(&model.model_path))
                        .map_err(|e| NeuralNetError(e))?;
                crate::tactic_plan::load_and_install(
                    std::path::Path::new(&path),
                    &fp,
                    &model_sha,
                    &super::backend_build_fingerprint(),
                )
                .map_err(|e| NeuralNetError(e))?;
                logger.write(&format!(
                    "cudaTacticPlan '{}' installed (plan id: {})\n",
                    path,
                    crate::tactic_plan::installed_plan_id().unwrap_or("?")
                ));
            }
            let legacy_tactics = if self.quantized { None } else {
                Some(crate::tactic_plan::freeze_legacy_tactics().map_err(NeuralNetError)?)
            };
            let legacy_handle_identity = if !self.quantized && cfg.contains("rustgoResolvedInputsUseNHWC") {
                Some((cfg.get_int("nnMaxBatchSize", 1, 65536).map_err(|e| NeuralNetError(e.to_string()))?,
                    cfg.get_bool("rustgoResolvedInputsUseNHWC").map_err(|e| NeuralNetError(e.to_string()))?))
            } else { None };
            model
                .rt
                .validate_residual_algo_request()
                .map_err(NeuralNetError)?;
            if self.quantized {
                validate_quantized_model_source(&model.source_model_sha256).map_err(NeuralNetError)?;
            } else if crate::tactic_plan::ffn_compact_requested().map_err(NeuralNetError)? {
                let sha = crate::tactic_plan::sha256_file(std::path::Path::new(&model.model_path)).map_err(NeuralNetError)?;
                if sha != crate::tactic_plan::FFN_COMPACT_MODEL { return Err(NeuralNetError("FFN compact source model SHA mismatch".into())); }
            }
            if !self.quantized && crate::tactic_plan::outproj_n128_requested().map_err(NeuralNetError)? {
                let sha=crate::tactic_plan::sha256_file(std::path::Path::new(&model.model_path)).map_err(NeuralNetError)?;
                crate::tactic_plan::validate_outproj_model_sha(&sha).map_err(NeuralNetError)?;
            }
            let gemm_layout =
                crate::backends::cuda_exec::selected_gemm_layout().map_err(NeuralNetError)?;
            let uploaded_model = {
                let mut state = model.model.lock().map_err(|_| {
                    NeuralNetError("CUDA model upload state was poisoned".to_string())
                })?;
                match &*state {
                    CudaModelLoadState::Uploaded {
                        model,
                        gemm_layout: uploaded_layout,
                        quant_recipe_source: uploaded_source,
                        inference_profile_id,
                        execution_profile_id,
                    } => {
                        if !self.quantized && (model.int8_scope() != int8_scope || model.int8_min_ffn_width() != int8_min_ffn_width) {
                            return Err(NeuralNetError("CUDA quantization scope/layer selection changed after upload; reload the model".into()));
                        }
                        if *uploaded_source != quant_recipe_source {
                            return Err(NeuralNetError("CUDA quantization recipe changed after upload; reload the model".into()));
                        }
                        if self.quantized {
                            let recipe_id = model.quantization_recipe_id().ok_or_else(|| NeuralNetError("missing uploaded quantization recipe identity".into()))?;
                            // The outer loaded model is shadowed by this match binding.
                            let loaded = loaded_model.as_any().downcast_ref::<CudaLoadedModel>().unwrap();
                            let actual = quantized_profile_id(loaded, model, recipe_id, cfg)?;
                            validate_expected_quant_profile(cfg, &actual)?;
                            if inference_profile_id.as_deref() != Some(actual.as_str()) {
                                return Err(NeuralNetError("CUDA quantized execution options changed after upload; reload in a new process".into()));
                            }
                        }
                        if let Some(tactics) = &legacy_tactics {
                            let loaded = loaded_model.as_any().downcast_ref::<CudaLoadedModel>().unwrap();
                            let actual = legacy_execution_profile_id(loaded, model, cfg, nn_x_len, nn_y_len, tactics)?;
                            if execution_profile_id != &actual {
                                return Err(NeuralNetError("CUDA legacy execution options changed after upload; reload in a new process".into()));
                            }
                        }
                        if *uploaded_layout != gemm_layout {
                            return Err(NeuralNetError(format!(
                                "CUDA model weights were prepared for GEMM layout '{uploaded_layout}', but this context requests '{gemm_layout}'; reload the model before changing its load-time layout"
                            )));
                        }
                        model.clone()
                    }
                    CudaModelLoadState::Parsed(graph) => {
                        // An independent stream plus a device sync makes the
                        // finished weights visible to every later handle.
                        let load_stream = model.rt.device.new_stream().map_err(|e| {
                            NeuralNetError(format!("CUDA stream create failed: {e}"))
                        })?;
                        let recipe = if self.quantized {
                            Some(match &quant_recipe_bytes {
                                Some(bytes) => {
                                    let parsed: crate::quantization_plan::PrecisionRecipe = serde_json::from_slice(bytes)
                                        .map_err(|e| NeuralNetError(format!("cudaQuantPlan JSON: {e}")))?;
                                    crate::quantization_plan::resolve_recipe(&parsed, graph, &model.source_model_sha256)
                                },
                                None => crate::quantization_plan::fp16_recipe(graph, &model.source_model_sha256),
                            }.map_err(NeuralNetError)?)
                        } else { None };
                        let uploaded = Arc::new(if let Some(recipe) = &recipe {
                            CudaModel::load_quantized(graph, &model.rt, &load_stream, recipe)
                        } else { match int8_scope {
                            Some(scope) => CudaModel::load_int8_selective(graph, &model.rt, &load_stream, scope, int8_min_ffn_width),
                            None => CudaModel::load(graph, &model.rt, &load_stream),
                        } }.map_err(
                                |e| NeuralNetError(format!("CUDA model upload failed: {e}")),
                            )?);
                        model
                            .rt
                            .device
                            .synchronize()
                            .map_err(|e| NeuralNetError(format!("CUDA sync failed: {e}")))?;
                        let inference_profile_id = if let Some(recipe) = &recipe {
                            ensure_requested_cuda_capabilities(&uploaded).map_err(NeuralNetError)?;
                            let identity = quantized_profile_id(model, &uploaded, &recipe.recipe_sha256, cfg)?;
                            validate_expected_quant_profile(cfg, &identity)?;
                            logger.write(&format!("[cuda-quant] model_sha256={} graph_sha256={} recipe_sha256={} inference_profile={} validation=unverified\n",
                                model.source_model_sha256, recipe.graph_sha256, recipe.recipe_sha256, identity));
                            Some(identity)
                        } else { None };
                        let execution_profile_id = if let Some(tactics) = &legacy_tactics {
                            // Observe the existing capability/fallback result;
                            // the getter itself never initializes GPU state.
                            ensure_requested_cuda_capabilities(&uploaded).map_err(NeuralNetError)?;
                            legacy_execution_profile_id(model, &uploaded, cfg, nn_x_len, nn_y_len, tactics)?
                        } else { inference_profile_id.clone() };
                        *state = CudaModelLoadState::Uploaded {
                            model: uploaded.clone(),
                            gemm_layout,
                            quant_recipe_source: quant_recipe_source.clone(),
                            inference_profile_id,
                            execution_profile_id,
                        };
                        uploaded
                    }
                }
            };
            if uploaded_model.ffn_compact_enabled() != crate::tactic_plan::ffn_compact_requested().map_err(NeuralNetError)? {
                return Err(NeuralNetError("CUDA model FFN compaction changed after upload; reload the model".into()));
            }
            ensure_requested_cuda_capabilities(&uploaded_model).map_err(NeuralNetError)?;
            if let Some(scope) = int8_scope {
                logger.write(&format!("[cuda-int8] precision=W8A8 accumulation=INT32 scope={} min_ffn_width={} int8_ffn_layers={} quantization={} model_sha256={} heads=FP16/FP32 activation_scaling=dynamic-per-row calibration=none\n", scope.name(), int8_min_ffn_width, uploaded_model.int8_ffn_count(), crate::backends::int8::QUANTIZATION_VERSION, model.model_desc.sha256));
            }
            Ok(Box::new(CudaComputeContext {
                model: uploaded_model,
                rt: model.rt.clone(),
                nn_x_len,
                nn_y_len,
                quant_max_batch_size,
                legacy_handle_identity,
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
            if let Some(expected_batch) = c.quant_max_batch_size {
                if max_batch_size != expected_batch || inputs_use_nhwc {
                    return Err(NeuralNetError(format!("quantized compute handle differs from its execution identity: expected batch={expected_batch}, NCHW; got batch={max_batch_size}, NHWC={inputs_use_nhwc}")));
                }
            }
            if let Some((batch, nhwc)) = c.legacy_handle_identity {
                if max_batch_size != batch || inputs_use_nhwc != nhwc {
                    return Err(NeuralNetError("legacy CUDA handle differs from its loaded execution identity".into()));
                }
            }
            // 每 handle 独立 non-blocking 流：多 server 线程并行提交推理，
            // 避免 legacy 默认流全设备串行化。
            let stream =
                c.rt.device
                    .new_stream()
                    .map_err(|e| NeuralNetError(format!("CUDA stream create failed: {e}")))?;
            Ok(Box::new(CudaComputeHandle {
                model: c.model.clone(),
                rt: c.rt.clone(),
                stream,
                graph_state: Mutex::new(None),
                graph_launch_reported: AtomicBool::new(false),
                direct_launch_reported: AtomicBool::new(false),
                nn_x_len: c.nn_x_len,
                nn_y_len: c.nn_y_len,
                max_batch_size,
                inputs_use_nhwc,
                pad_to_max: crate::tactic_plan::tactic_enabled("KATAGO_CUDA_PADBATCH"),
            }))
        }

        fn is_using_fp16(&self, handle: &dyn ComputeHandle) -> bool {
            handle.is_using_fp16()
        }

        fn set_is_warmup(&self, handle: &mut dyn ComputeHandle, is_warmup: bool) -> bool {
            handle.set_is_warmup(is_warmup)
        }

        fn warmup_batches(&self, max_batch_size: i32) -> Vec<i32> {
            if self.int8 && crate::tactic_plan::tactic_var("KATAGO_CUDA_INT8_GEMM_TUNE").is_ok() {
                // Pay opt-in algorithm selection cost before accepting work,
                // including intermediate batches used by local search. An
                // explicit 0 warms the same shapes for fair A/B measurement;
                // an absent key retains the normal lightweight startup.
                return (1..=max_batch_size.max(1)).collect();
            }
            if max_batch_size <= 1 {
                vec![1]
            } else {
                vec![1, max_batch_size]
            }
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
            // 同步语义 = 提交后立即完成（eval serve 流水线直接调 submit/finish
            // 以获得批间重叠；此处保持 trait 契约供其余调用方使用）。
            let slot = self.submit_output(h, n, input_bufs)?;
            self.finish_output(h, slot, n, input_bufs, outputs)
        }

        fn supports_async_pipeline(&self) -> bool {
            // KATAGO_CUDA_NOPIPELINE=1：回退同步 serve 循环（ABBA 对照/调试）。
            !crate::tactic_plan::tactic_enabled("KATAGO_CUDA_NOPIPELINE")
        }

        fn submit_output(
            &self,
            handle: &dyn ComputeHandle,
            _buffers: &dyn InputBuffers,
            num_batch_elts: i32,
            input_bufs: &mut [&mut NNResultBuf],
        ) -> Result<usize, NeuralNetError> {
            let h = handle
                .as_any()
                .downcast_ref::<CudaComputeHandle>()
                .ok_or_else(|| NeuralNetError("Wrong compute handle type".to_string()))?;
            let n = num_batch_elts as usize;
            if n == 0 {
                return Err(NeuralNetError("submit_output with empty batch".to_string()));
            }
            if n > h.max_batch_size as usize {
                return Err(NeuralNetError(format!(
                    "batch size {n} exceeds handle max {}",
                    h.max_batch_size
                )));
            }
            CudaBackend::submit_output(self, h, n, input_bufs)
        }

        fn query_output_done(&self, handle: &dyn ComputeHandle, token: usize) -> bool {
            let Some(h) = handle.as_any().downcast_ref::<CudaComputeHandle>() else {
                return true;
            };
            self.query_slot_done(h, token)
        }

        fn finish_output(
            &self,
            handle: &dyn ComputeHandle,
            _buffers: &dyn InputBuffers,
            token: usize,
            num_batch_elts: i32,
            input_bufs: &mut [&mut NNResultBuf],
            outputs: &mut [&mut NNOutput],
        ) -> Result<(), NeuralNetError> {
            let h = handle
                .as_any()
                .downcast_ref::<CudaComputeHandle>()
                .ok_or_else(|| NeuralNetError("Wrong compute handle type".to_string()))?;
            CudaBackend::finish_output(self, h, token, num_batch_elts as usize, input_bufs, outputs)
        }
    }
}

#[cfg(feature = "cuda")]
pub use backend_impl::{CudaBackend, CudaInt8Backend, CudaQuantBackend, ensure_requested_cuda_capabilities, validate_quantized_runtime, validate_quantized_model_source};
