//! Hand-written CUDA backend (M3 骨架：kernel 加载与基础运行时）。
//!
//! kernel 由 `build.rs` 按 `configs/sm-targets.json` 编译为 PTX 并嵌入二进制
//! （`CUDA_TARGETS` 表）；运行时按设备 compute capability 选择最优目标。
//!
//! 对应 KataGo 官方 `cpp/neuralnet/cudabackend.cpp`，并参考 KataGomo_fork 的
//! SM120 优化路径（plan 驱动、fail-closed 见 M4）。

#[cfg(feature = "cuda")]
mod imp {
    use cudarc::driver::sys::CUdevice_attribute;
    use cudarc::driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg,
    };
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    // 由 build.rs 生成的 PTX 表：`(target_id, compute_capability, [(kernel_name, ptx_bytes)])`。
    include!(concat!(env!("OUT_DIR"), "/cuda_kernels.rs"));
    // 由 build.rs 生成的构建指纹和编译能力表。
    include!(concat!(env!("OUT_DIR"), "/cuda_build.rs"));

    /// 构建期 CUDA 后端指纹，供认证 plan 和 `cuda-fingerprint` 共用。
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
            },
        }
    }

    /// Whether the optional q64 attention PTX was compiled into this binary.
    /// Keep the hot-path capability check allocation-free; the full fingerprint
    /// is intentionally reserved for plan validation and diagnostics.
    pub fn attention_q64_available() -> bool {
        CUDA_CAP_ATTENTION_Q64
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
        /// cuBLASLt GEMM(f16 输入 f32 累加输出),按 (m,n,k) 缓存启发式算法。
        /// graph capture 前需 warmup(第一次调用完成算法选择)。
        cublaslt: Option<CublasLtState>,
    }

    /// cuBLASLt 状态:handle + workspace + 算法缓存。
    struct CublasLtState {
        handle: cudarc::cublaslt::sys::cublasLtHandle_t,
        workspace_len: usize,
        /// cuBLASLt workspace cannot be shared by overlapping matmuls on
        /// different streams. Keep one allocation per stream while retaining
        /// the single runtime/model shared by all benchmark handles.
        workspaces: Mutex<HashMap<usize, CudaSlice<u8>>>,
        algo_cache: Mutex<
            HashMap<(usize, usize, usize, bool), cudarc::cublaslt::sys::cublasLtMatmulAlgo_t>,
        >,
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
            let Some(st) = &self.cublaslt else {
                return Ok(false);
            };
            let key = (m, n, k, beta != 0.0);
            let cached = st.algo_cache.lock().unwrap().get(&key).copied();
            let algo = match cached {
                Some(a) => Some(a),
                None => {
                    let cands = self.cublaslt_select_algos(st, m, n, k, false)?;
                    if cands.is_empty() {
                        None
                    } else if cublaslt_rank_time() && !crate::backends::cuda_exec::capturing() {
                        // C1 计时重排:在当前活跃流上逐候选计时(scratch 承载输出,
                        // beta=0,不碰真实 C);与真实工作同流提交,天然有序。
                        crate::backends::cuda_exec::report_tactic_once(
                            &crate::backends::cuda_exec::CUBLASLT_RANK_TIME_REPORT,
                            "name=cublaslt_rank engine=time",
                        );
                        let tstream = crate::backends::cuda_exec::active_stream_clone()
                            .unwrap_or_else(|| self.device.default_stream());
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
            self.cublaslt_exec(st, stream, &algo, a, b, c, m, n, k, beta)?;
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
            let Some(st) = &self.cublaslt else {
                return Ok(false);
            };
            let key = (m, n, k, true); // f16out 用 beta=true 槽位区分
            let cached = st.algo_cache.lock().unwrap().get(&key).copied();
            let algo = match cached {
                Some(a) => Some(a),
                None => {
                    let cands = self.cublaslt_select_algos(st, m, n, k, true)?;
                    if cands.is_empty() {
                        None
                    } else if cublaslt_rank_time() && !crate::backends::cuda_exec::capturing() {
                        crate::backends::cuda_exec::report_tactic_once(
                            &crate::backends::cuda_exec::CUBLASLT_RANK_TIME_REPORT,
                            "name=cublaslt_rank engine=time",
                        );
                        let tstream = crate::backends::cuda_exec::active_stream_clone()
                            .unwrap_or_else(|| self.device.default_stream());
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
            self.cublaslt_exec_f16(st, stream, &algo, a, b, c, m, n, k)?;
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
        ) -> Option<cudarc::cublaslt::sys::cublasLtMatmulAlgo_t> {
            let mut best: Option<(f64, cudarc::cublaslt::sys::cublasLtMatmulAlgo_t)> = None;
            let mut first_ms = f64::NAN;
            for (i, cand) in cands.iter().enumerate() {
                let mut failed = false;
                for _ in 0..3 {
                    if self
                        .cublaslt_exec(st, stream, cand, a, b, scratch, m, n, k, 0.0)
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
                        .cublaslt_exec(st, stream, cand, a, b, scratch, m, n, k, 0.0)
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
        ) -> Option<cudarc::cublaslt::sys::cublasLtMatmulAlgo_t> {
            let mut best: Option<(f64, cudarc::cublaslt::sys::cublasLtMatmulAlgo_t)> = None;
            let mut first_ms = f64::NAN;
            for (i, cand) in cands.iter().enumerate() {
                let mut failed = false;
                for _ in 0..3 {
                    if self
                        .cublaslt_exec_f16(st, stream, cand, a, b, scratch, m, n, k)
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
                        .cublaslt_exec_f16(st, stream, cand, a, b, scratch, m, n, k)
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
        ) -> Result<Vec<cudarc::cublaslt::sys::cublasLtMatmulAlgo_t>, String> {
            use cudarc::cublaslt::sys;
            unsafe {
                let mut desc: sys::cublasLtMatmulDesc_t = std::ptr::null_mut();
                sys::cublasLtMatmulDescCreate(
                    &mut desc,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    sys::cudaDataType_t::CUDA_R_32F,
                );
                let transa: u32 = 1; // CUBLAS_OP_T
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
                // 布局(列主序映射):A_cm=B 内存 [k,n] ld=k;B_cm=A 内存 [k,m] ld=k;C [n,m] ld=n
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
                    k as u64,
                    n as u64,
                    k as i64,
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
                let transa: u32 = 1; // CUBLAS_OP_T
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
                    k as u64,
                    n as u64,
                    k as i64,
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
                let transa: u32 = 1;
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
                    k as u64,
                    n as u64,
                    k as i64,
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
    CudaRuntime, attention_q64_available, backend_build_fingerprint, f16_to_f32_bits,
    f32_to_f16_bits, graph_instantiate_flags,
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
    use kata_game::symmetry::{copy_inputs_with_symmetry, copy_outputs_with_symmetry, invert};
    use std::any::Any;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    /// CUDA 后端（手写 kernel，sm-targets.json 选择编译目标）。
    pub struct CudaBackend;

    /// Validate and, when requested, execute the real production DualFFN
    /// capability probe before any serve thread can submit inference.
    pub(crate) fn ensure_requested_cuda_capabilities(model: &CudaModel) -> Result<(), String> {
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
                if crate::tactic_plan::installed_plan_id().is_some() {
                    return Err(format!(
                        "certified plan requests DualFFN but it is unavailable: {reason}"
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

    /// 已加载模型：层图权重常驻设备 + 运行时（设备/模块）。
    pub struct CudaLoadedModel {
        model_desc: ModelDesc,
        model: Arc<CudaModel>,
        rt: Arc<CudaRuntime>,
        /// 模型文件路径（cudaTacticPlan 指纹校验时算 SHA-256 用）。
        model_path: String,
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

    /// per-slot 的 graph 状态：固定 batch 的输入缓冲 + 已捕获图 + 完成事件。
    /// 双槽（pipeline）：submit(N+1) 的 htod 与 graph(N) 执行重叠。
    struct CudaSlot {
        /// 已捕获图；None = capture 失败降级直连。
        graph: Option<cudarc::driver::CudaGraph>,
        in_spatial: cudarc::driver::CudaSlice<f32>,
        in_global: cudarc::driver::CudaSlice<f32>,
        in_sp_pin: cudarc::driver::PinnedHostSlice<f32>,
        in_gl_pin: cudarc::driver::PinnedHostSlice<f32>,
        out_policy_pin: cudarc::driver::PinnedHostSlice<f32>,
        out_value_pin: cudarc::driver::PinnedHostSlice<f32>,
        out_misc_pin: cudarc::driver::PinnedHostSlice<f32>,
        out_moremisc_pin: cudarc::driver::PinnedHostSlice<f32>,
        out_own_pin: cudarc::driver::PinnedHostSlice<f32>,
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
                let mut ws = CudaWorkspace::new(stream, &h.model, phys_batch)
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
                            h.model
                                .apply(&h.rt, stream, &mut ws, &in_spatial, &in_global)
                                .map_err(|e| NeuralNetError(format!("pre-capture warm: {e}")))?;
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
                        // 若 RELAXED 仍失效(驱动差异)则自动回退 nograph(下方
                        // match 分支),行为与旧版一致。
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
                        in_spatial,
                        in_global,
                        in_sp_pin,
                        in_gl_pin,
                        out_policy_pin,
                        out_value_pin,
                        out_misc_pin,
                        out_moremisc_pin,
                        out_own_pin,
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
            let policy_area = (nn_x_len * nn_y_len) as usize;
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

            let single_policy = 6 * (policy_area + 1);
            let mut tmp_policy_base = vec![0.0f32; policy_area];
            let mut tmp_policy_opt = vec![0.0f32; policy_area];
            let mut tmp_ownership = vec![0.0f32; policy_area];
            for i in 0..n {
                let sym_idx = input_bufs[i].symmetry;
                let inv_sym = if sym_idx != 0 { invert(sym_idx) } else { 0 };
                let p_off = i * single_policy;
                let base_src = &host.policy[p_off..p_off + policy_area];
                let opt_src = &host.policy[p_off + 5 * policy_area..p_off + 6 * policy_area];
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
                    outputs[i].policy_probs[pos] = tmp_policy_base[pos]
                        + (tmp_policy_opt[pos] - tmp_policy_base[pos]) * optimism;
                }
                let base_pass = host.policy[p_off + policy_area];
                let opt_pass = host.policy[p_off + 5 * policy_area + policy_area];
                outputs[i].policy_probs[policy_area] =
                    base_pass + (opt_pass - base_pass) * optimism;
                let v_off = i * 3;
                outputs[i].white_win_prob = host.value[v_off];
                outputs[i].white_loss_prob = host.value[v_off + 1];
                outputs[i].white_no_result_prob = host.value[v_off + 2];
                let m_off = i * 10;
                outputs[i].white_score_mean = host.misc[m_off];
                outputs[i].white_score_mean_sq = host.misc[m_off + 1];
                outputs[i].white_lead = host.misc[m_off + 2];
                outputs[i].var_time_left = host.misc[m_off + 3];
                let mm_off = i * 8;
                outputs[i].shortterm_winloss_error = host.moremisc[mm_off];
                outputs[i].shortterm_score_error = host.moremisc[mm_off + 1];
                if input_bufs[i].include_owner_map {
                    let o_off = i * policy_area;
                    let src = &host.ownership[o_off..o_off + policy_area];
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
            // 权重上传用独立流，完成后设备级同步，保证后续任意流可见。
            let load_stream = rt
                .device
                .new_stream()
                .map_err(|e| NeuralNetError(format!("CUDA stream create failed: {e}")))?;
            let model = Arc::new(
                CudaModel::load(&graph, &rt, &load_stream)
                    .map_err(|e| NeuralNetError(format!("CUDA model upload failed: {e}")))?,
            );
            rt.device
                .synchronize()
                .map_err(|e| NeuralNetError(format!("CUDA sync failed: {e}")))?;
            Ok(Box::new(CudaLoadedModel {
                model_desc: parsed.model_desc,
                model,
                rt,
                model_path: file.to_string(),
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
            ensure_requested_cuda_capabilities(&model.model).map_err(NeuralNetError)?;
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
pub use backend_impl::CudaBackend;
