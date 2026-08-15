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
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

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
        /// cuBLASLt GEMM(f16 输入 f32 累加输出),按 (m,n,k) 缓存启发式算法。
        /// graph capture 前需 warmup(第一次调用完成算法选择)。
        cublaslt: Option<CublasLtState>,
    }

    /// cuBLASLt 状态:handle + workspace + 算法缓存。
    struct CublasLtState {
        handle: cudarc::cublaslt::sys::cublasLtHandle_t,
        workspace: CudaSlice<u8>,
        algo_cache: Mutex<HashMap<(usize, usize, usize, bool), cudarc::cublaslt::sys::cublasLtMatmulAlgo_t>>,
    }
    // cuBLASLt handle 是 opaque 指针,create 后可跨线程使用(文档:线程安全)。
    unsafe impl Send for CublasLtState {}
    unsafe impl Sync for CublasLtState {}

    impl CudaRuntime {
        /// 初始化设备 0 并加载最优 SM 目标的全部 kernel（fail-closed）。
        pub fn new() -> Result<Self, String> {
            let device = CudaContext::new(0).map_err(|e| format!("CUDA device 0 init failed: {e}"))?;
            // 关闭 cudarc 自动 event 跟踪：graph capture 会因未记录 event 的
            // 跨流 wait 报 CUDA_ERROR_STREAM_CAPTURE_ISOLATION。本后端显式
            // 管理同步（每 handle 单流、加载后 device 级同步、每次前向后
            // stream.synchronize），host 回拷由 SyncOnDrop 直接同步流保证。
            unsafe { device.disable_event_tracking() };
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
                let stream = device.default_stream();
                let ws: CudaSlice<u8> = match stream.alloc(32 * 1024 * 1024) {
                    Ok(w) => w,
                    Err(e) => {
                        eprintln!("note: cublaslt workspace alloc failed ({e}), disabled");
                        return None;
                    }
                };
                Some(CublasLtState {
                    handle,
                    workspace: ws,
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
            stream: &cudarc::driver::CudaStream,
            a: &CudaSlice<u16>,
            b: &CudaSlice<u16>,
            c: &mut CudaSlice<f32>,
            m: usize,
            n: usize,
            k: usize,
            beta: f32,
        ) -> Result<bool, String> {
            use cudarc::cublaslt::sys;
            let Some(st) = &self.cublaslt else { return Ok(false) };
            let key = (m, n, k, beta != 0.0);
            let algo = {
                let cache = st.algo_cache.lock().unwrap();
                cache.get(&key).copied()
            };
            let algo = match algo {
                Some(a) => Some(a),
                None => match self.cublaslt_select_algo(st, m, n, k)? {
                    Some(a) => {
                        st.algo_cache.lock().unwrap().insert(key, a);
                        Some(a)
                    }
                    None => None,
                },
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
            stream: &cudarc::driver::CudaStream,
            a: &CudaSlice<u16>,
            b: &CudaSlice<u16>,
            c: &mut CudaSlice<u16>,
            m: usize,
            n: usize,
            k: usize,
        ) -> Result<bool, String> {
            use cudarc::cublaslt::sys;
            let Some(st) = &self.cublaslt else { return Ok(false) };
            let key = (m, n, k, true); // f16out 用 beta=true 槽位区分
            let algo = {
                let cache = st.algo_cache.lock().unwrap();
                cache.get(&key).copied()
            };
            let algo = match algo {
                Some(a) => Some(a),
                None => match self.cublaslt_select_algo_f16(st, m, n, k)? {
                    Some(a) => {
                        st.algo_cache.lock().unwrap().insert(key, a);
                        Some(a)
                    }
                    None => None,
                },
            };
            let Some(algo) = algo else { return Ok(false) };
            self.cublaslt_exec_f16(st, stream, &algo, a, b, c, m, n, k)?;
            Ok(true)
        }

        /// 选算法(heuristic 查询)。
        fn cublaslt_select_algo(
            &self,
            st: &CublasLtState,
            m: usize,
            n: usize,
            k: usize,
        ) -> Result<Option<cudarc::cublaslt::sys::cublasLtMatmulAlgo_t>, String> {
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
                    desc, sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                    &transa as *const _ as *const _, std::mem::size_of_val(&transa));
                sys::cublasLtMatmulDescSetAttribute(
                    desc, sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                    &transb as *const _ as *const _, std::mem::size_of_val(&transb));
                // 布局(列主序映射):A_cm=B 内存 [k,n] ld=k;B_cm=A 内存 [k,m] ld=k;C [n,m] ld=n
                let mut a_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                let mut b_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                let mut c_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                sys::cublasLtMatrixLayoutCreate(&mut a_lay, sys::cudaDataType_t::CUDA_R_16F, k as u64, n as u64, k as i64);
                sys::cublasLtMatrixLayoutCreate(&mut b_lay, sys::cudaDataType_t::CUDA_R_16F, k as u64, m as u64, k as i64);
                sys::cublasLtMatrixLayoutCreate(&mut c_lay, sys::cudaDataType_t::CUDA_R_32F, n as u64, m as u64, n as i64);
                let mut pref: sys::cublasLtMatmulPreference_t = std::ptr::null_mut();
                sys::cublasLtMatmulPreferenceCreate(&mut pref);
                let ws_size = st.workspace.len();
                sys::cublasLtMatmulPreferenceSetAttribute(
                    pref,
                    sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    &ws_size as *const _ as *const _, std::mem::size_of_val(&ws_size));
                let mut heur: sys::cublasLtMatmulHeuristicResult_t = std::mem::zeroed();
                let mut cnt = 0i32;
                let r = sys::cublasLtMatmulAlgoGetHeuristic(
                    st.handle, desc, a_lay, b_lay, c_lay, c_lay, pref, 1, &mut heur, &mut cnt,
                );
                sys::cublasLtMatmulDescDestroy(desc);
                sys::cublasLtMatrixLayoutDestroy(a_lay);
                sys::cublasLtMatrixLayoutDestroy(b_lay);
                sys::cublasLtMatrixLayoutDestroy(c_lay);
                sys::cublasLtMatmulPreferenceDestroy(pref);
                if r != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS || cnt == 0 {
                    return Ok(None);
                }
                Ok(Some(heur.algo))
            }
        }

        /// 执行 cuBLASLt GEMM(已选算法)。
        #[allow(clippy::too_many_arguments)]
        fn cublaslt_exec(
            &self,
            st: &CublasLtState,
            stream: &cudarc::driver::CudaStream,
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
                    desc, sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                    &transa as *const _ as *const _, std::mem::size_of_val(&transa));
                sys::cublasLtMatmulDescSetAttribute(
                    desc, sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                    &transb as *const _ as *const _, std::mem::size_of_val(&transb));
                let mut a_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                let mut b_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                let mut c_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                sys::cublasLtMatrixLayoutCreate(&mut a_lay, sys::cudaDataType_t::CUDA_R_16F, k as u64, n as u64, k as i64);
                sys::cublasLtMatrixLayoutCreate(&mut b_lay, sys::cudaDataType_t::CUDA_R_16F, k as u64, m as u64, k as i64);
                sys::cublasLtMatrixLayoutCreate(&mut c_lay, sys::cudaDataType_t::CUDA_R_32F, n as u64, m as u64, n as i64);
                let alpha = 1.0f32;
                let (a_ptr, _ga) = a.device_ptr(stream);
                let (b_ptr, _gb) = b.device_ptr(stream);
                let (c_ptr, _gc) = c.device_ptr_mut(stream);
                let (ws_ptr, _gw) = st.workspace.device_ptr(stream);
                let r = sys::cublasLtMatmul(
                    st.handle,
                    desc,
                    &alpha as *const _ as *const _,
                    b_ptr as *const _,   // A_cm = B
                    a_lay,
                    a_ptr as *const _,   // B_cm = A
                    b_lay,
                    &beta as *const _ as *const _,
                    c_ptr as *const _,
                    c_lay,
                    c_ptr as *mut _,
                    c_lay,
                    algo,
                    ws_ptr as *mut _,
                    st.workspace.len(),
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

        /// f16 输出的算法选择(Cdesc = CUDA_R_16F)。
        fn cublaslt_select_algo_f16(
            &self,
            st: &CublasLtState,
            m: usize,
            n: usize,
            k: usize,
        ) -> Result<Option<cudarc::cublaslt::sys::cublasLtMatmulAlgo_t>, String> {
            use cudarc::cublaslt::sys;
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
                    desc, sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                    &transa as *const _ as *const _, std::mem::size_of_val(&transa));
                sys::cublasLtMatmulDescSetAttribute(
                    desc, sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                    &transb as *const _ as *const _, std::mem::size_of_val(&transb));
                let mut a_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                let mut b_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                let mut c_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                sys::cublasLtMatrixLayoutCreate(&mut a_lay, sys::cudaDataType_t::CUDA_R_16F, k as u64, n as u64, k as i64);
                sys::cublasLtMatrixLayoutCreate(&mut b_lay, sys::cudaDataType_t::CUDA_R_16F, k as u64, m as u64, k as i64);
                sys::cublasLtMatrixLayoutCreate(&mut c_lay, sys::cudaDataType_t::CUDA_R_16F, n as u64, m as u64, n as i64);
                let mut pref: sys::cublasLtMatmulPreference_t = std::ptr::null_mut();
                sys::cublasLtMatmulPreferenceCreate(&mut pref);
                let ws_size = st.workspace.len();
                sys::cublasLtMatmulPreferenceSetAttribute(
                    pref,
                    sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    &ws_size as *const _ as *const _, std::mem::size_of_val(&ws_size));
                let mut heur: sys::cublasLtMatmulHeuristicResult_t = std::mem::zeroed();
                let mut cnt = 0i32;
                let r = sys::cublasLtMatmulAlgoGetHeuristic(
                    st.handle, desc, a_lay, b_lay, c_lay, c_lay, pref, 1, &mut heur, &mut cnt,
                );
                sys::cublasLtMatmulDescDestroy(desc);
                sys::cublasLtMatrixLayoutDestroy(a_lay);
                sys::cublasLtMatrixLayoutDestroy(b_lay);
                sys::cublasLtMatrixLayoutDestroy(c_lay);
                sys::cublasLtMatmulPreferenceDestroy(pref);
                if r != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS || cnt == 0 {
                    return Ok(None);
                }
                Ok(Some(heur.algo))
            }
        }

        /// f16 输出的执行(Cdesc = CUDA_R_16F,C 为 f16)。
        #[allow(clippy::too_many_arguments)]
        fn cublaslt_exec_f16(
            &self,
            st: &CublasLtState,
            stream: &cudarc::driver::CudaStream,
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
                    desc, sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                    &transa as *const _ as *const _, std::mem::size_of_val(&transa));
                sys::cublasLtMatmulDescSetAttribute(
                    desc, sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                    &transb as *const _ as *const _, std::mem::size_of_val(&transb));
                let mut a_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                let mut b_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                let mut c_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
                sys::cublasLtMatrixLayoutCreate(&mut a_lay, sys::cudaDataType_t::CUDA_R_16F, k as u64, n as u64, k as i64);
                sys::cublasLtMatrixLayoutCreate(&mut b_lay, sys::cudaDataType_t::CUDA_R_16F, k as u64, m as u64, k as i64);
                sys::cublasLtMatrixLayoutCreate(&mut c_lay, sys::cudaDataType_t::CUDA_R_16F, n as u64, m as u64, n as i64);
                let alpha = 1.0f32;
                let beta = 0.0f32;
                let (a_ptr, _ga) = a.device_ptr(stream);
                let (b_ptr, _gb) = b.device_ptr(stream);
                let (c_ptr, _gc) = c.device_ptr_mut(stream);
                let (ws_ptr, _gw) = st.workspace.device_ptr(stream);
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
                    st.workspace.len(),
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
            let max_persisting = self
                .device
                .attribute(
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE,
                )
                .map_err(|e| format!("query max persisting L2: {e}"))? as usize;
            let window_cap = self
                .device
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
        pub fn clear_l2_window(
            &self,
            stream: &cudarc::driver::CudaStream,
        ) -> Result<(), String> {
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
            stream
                .memcpy_htod(c, &mut d_c)
                .map_err(|e| e.to_string())?;

            let grid = (
                m.div_ceil(128) as u32,
                n.div_ceil(128) as u32,
                1u32,
            );
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
    use crate::backends::cuda_exec::{set_capturing, CudaModel, CudaOutputsHost, CudaWorkspace};
    use crate::desc::ModelDesc;
    use kata_core::config::Config;
    use kata_core::logger::Logger;
    use kata_game::symmetry::{copy_inputs_with_symmetry, copy_outputs_with_symmetry, invert};
    use std::any::Any;
    use std::sync::{Arc, Mutex};

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

    /// per-handle 的 CUDA Graph 状态：固定 batch 的输入缓冲 + 工作区 + 已捕获图。
    /// 首次前向捕获（180 kernel → 1 次 graph 提交），后续复用。
    /// 输入/输出走 pinned host 内存（htod/dtoh 真正异步，避免非 pinned
    /// 的退化同步往返；每次前向仅一次 event 同步）。
    struct CudaGraphState {
        /// 已捕获图；None = capture 失败降级直连（每前向逐 kernel 提交）。
        graph: Option<cudarc::driver::CudaGraph>,
        ws: CudaWorkspace,
        in_spatial: cudarc::driver::CudaSlice<f32>,
        in_global: cudarc::driver::CudaSlice<f32>,
        in_sp_pin: cudarc::driver::PinnedHostSlice<f32>,
        in_gl_pin: cudarc::driver::PinnedHostSlice<f32>,
        out_policy_pin: cudarc::driver::PinnedHostSlice<f32>,
        out_value_pin: cudarc::driver::PinnedHostSlice<f32>,
        out_misc_pin: cudarc::driver::PinnedHostSlice<f32>,
        out_moremisc_pin: cudarc::driver::PinnedHostSlice<f32>,
        out_own_pin: cudarc::driver::PinnedHostSlice<f32>,
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
            // 每 handle 独立 non-blocking 流：多 server 线程并行提交推理，
            // 避免 legacy 默认流全设备串行化。
            let stream = c
                .rt
                .device
                .new_stream()
                .map_err(|e| NeuralNetError(format!("CUDA stream create failed: {e}")))?;
            Ok(Box::new(CudaComputeHandle {
                model: c.model.clone(),
                rt: c.rt.clone(),
                stream,
                graph_state: Mutex::new(None),
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

            // host 计时（KATAGO_CUDA_HOST_PROFILE=1）：fill / exec / decode 三段累计。
            let host_prof = std::env::var("KATAGO_CUDA_HOST_PROFILE").is_ok();
            let t0 = std::time::Instant::now();

            // --- 填输入（NCHW spatial + global，空间特征带对称性） ----------
            const NUM_SPATIAL_CHANNELS: i32 = 22;
            const NUM_GLOBAL_CHANNELS: usize = 19;
            let single_spatial = (NUM_SPATIAL_CHANNELS * nn_x_len * nn_y_len) as usize;

            // 物理 batch：batch=1 用专用小图（单线程/串行请求不浪费 GPU）；
            // 其余按实际 batch（kernel 计算随 batch 线性增长，padding 会
            // 放大 GPU 时间——ABBA 证伪了固定 max 物理 batch 的尾批复制）。
            let phys_batch = n;

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
            // padding 行（仅 phys_batch > n 时；当前 phys_batch == n，循环为空）。
            for i in n..phys_batch {
                let src = (i - 1) * single_spatial;
                let dst = i * single_spatial;
                spatial_host.copy_within(src..src + single_spatial, dst);
                let gsrc = (i - 1) * NUM_GLOBAL_CHANNELS;
                let gdst = i * NUM_GLOBAL_CHANNELS;
                global_host.copy_within(gsrc..gsrc + NUM_GLOBAL_CHANNELS, gdst);
            }

            // --- 上传 + 前向 -------------------------------------------------
            let stream = &h.stream;
            let t1 = std::time::Instant::now();

            // 调试钩子：dump 后端收到的输入（KATAGO_CUDA_DUMP_INPUT=<dir>）。
            if let Ok(d) = std::env::var("KATAGO_CUDA_DUMP_INPUT") {
                let _ = std::fs::create_dir_all(&d);
                let w = |name: &str, data: &[f32]| {
                    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
                    let _ = std::fs::write(format!("{d}/{name}.bin"), bytes);
                };
                w("spatial", &spatial_host);
                w("global", &global_host);
            }

            // --- 上传 + 前向（CUDA Graph：首次捕获，后续 1 次提交） ----------
            // WDDM/CUDA13 下 capture 与同 context 其他流的并发活动会互相
            // 干扰（实测 STREAM_CAPTURE_INVALIDATED，100% 复现），故 get_output
            // 全程全局互斥：批量已在 serve 层凑好，batch 内并行不受影响，
            // 串行开销（后端 ~0.15ms/批）远小于稳定性损失。
            static CUDA_EXEC_LOCK: Mutex<()> = Mutex::new(());
            let _exec_guard = CUDA_EXEC_LOCK.lock().unwrap();
            let host = {
                let mut g = h.graph_state.lock().unwrap();
                // KATAGO_CUDA_NOGRAPH=1：强制直连路径（逐 kernel 提交），
                // 配合 KATAGO_CUDA_PROFILE 定位 graph 重放外的真实 kernel 耗时。
                let force_direct = std::env::var("KATAGO_CUDA_NOGRAPH").is_ok();
                // workspace/缓冲 需要（重新）创建（物理 batch 变化或首次）。
                let need_rebuild = match g.as_ref() {
                    Some(s) => s.ws.batch() != phys_batch,
                    None => true,
                };
                if need_rebuild {
                    // 清场：等待设备全部工作完成，避免与其他流的未完成
                    // 工作产生 capture 依赖（STREAM_CAPTURE_INVALIDATED）。
                    // 直连模式无需（无 capture）。
                    if !force_direct {
                        h.rt
                            .device
                            .synchronize()
                            .map_err(|e| NeuralNetError(format!("pre-capture sync: {e}")))?;
                    }
                    // 固定地址输入缓冲 + 工作区 + pinned host 缓冲，
                    // 捕获整图（htod/dtoh 在图外）。
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
                    let out_policy_pin = unsafe { h.rt.device.alloc_pinned::<f32>(phys_batch * 6 * 362) }
                        .map_err(|e| NeuralNetError(format!("pinned out_policy: {e}")))?;
                    let out_value_pin = unsafe { h.rt.device.alloc_pinned::<f32>(phys_batch * 3) }
                        .map_err(|e| NeuralNetError(format!("pinned out_value: {e}")))?;
                    let out_misc_pin = unsafe { h.rt.device.alloc_pinned::<f32>(phys_batch * 10) }
                        .map_err(|e| NeuralNetError(format!("pinned out_misc: {e}")))?;
                    let out_moremisc_pin = unsafe { h.rt.device.alloc_pinned::<f32>(phys_batch * 8) }
                        .map_err(|e| NeuralNetError(format!("pinned out_moremisc: {e}")))?;
                    let out_own_pin = unsafe { h.rt.device.alloc_pinned::<f32>(phys_batch * policy_area) }
                        .map_err(|e| NeuralNetError(format!("pinned out_own: {e}")))?;
                    let mut ws = CudaWorkspace::new(stream, &h.model, phys_batch)
                        .map_err(|e| NeuralNetError(format!("CUDA workspace alloc failed: {e}")))?;
                    let graph = if force_direct {
                        None
                    } else {
                        set_capturing(true);
                        let cap_result = (|| -> Result<cudarc::driver::CudaGraph, NeuralNetError> {
                            stream.begin_capture(
                                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
                            ).map_err(|e| NeuralNetError(format!("begin_capture: {e}")))?;
                            h.model
                                .apply(&h.rt, stream, &mut ws, &in_spatial, &in_global)
                                .map_err(|e| NeuralNetError(format!("CUDA forward (capture) failed: {e}")))?;
                            let flags: cudarc::driver::sys::CUgraphInstantiate_flags =
                                unsafe { std::mem::transmute(0u32) };
                            stream
                                .end_capture(flags)
                                .map_err(|e| NeuralNetError(format!("end_capture: {e}")))?
                                .ok_or_else(|| NeuralNetError("capture produced no graph".to_string()))
                        })();
                        set_capturing(false);
                        match cap_result {
                            Ok(g) => Some(g),
                            Err(e) => {
                                // capture 失败：清理流的 capture 状态（丢弃残余图），
                                // 降级为直连路径（每前向逐 kernel 提交，慢但正确）。
                                let flags: cudarc::driver::sys::CUgraphInstantiate_flags =
                                    unsafe { std::mem::transmute(0u32) };
                                let _ = stream.end_capture(flags);
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
                    *g = Some(CudaGraphState {
                        graph,
                        ws,
                        in_spatial,
                        in_global,
                        in_sp_pin,
                        in_gl_pin,
                        out_policy_pin,
                        out_value_pin,
                        out_misc_pin,
                        out_moremisc_pin,
                        out_own_pin,
                    });
                }
                let st = g.as_mut().unwrap();
                // 填 pinned 输入 → 异步 htod（固定设备地址）→ graph 提交 →
                // 异步 dtoh 到 pinned → 一次同步后读。
                let e0 = std::time::Instant::now();
                {
                    let sp = st
                        .in_sp_pin
                        .as_mut_slice()
                        .map_err(|e| NeuralNetError(format!("pin in_sp: {e}")))?;
                    sp.copy_from_slice(&spatial_host);
                    let gl = st
                        .in_gl_pin
                        .as_mut_slice()
                        .map_err(|e| NeuralNetError(format!("pin in_gl: {e}")))?;
                    gl.copy_from_slice(&global_host);
                }
                let e1 = std::time::Instant::now();
                stream
                    .memcpy_htod(&st.in_sp_pin, &mut st.in_spatial)
                    .map_err(|e| NeuralNetError(format!("upload spatial: {e}")))?;
                stream
                    .memcpy_htod(&st.in_gl_pin, &mut st.in_global)
                    .map_err(|e| NeuralNetError(format!("upload global: {e}")))?;
                let e2 = std::time::Instant::now();
                match st.graph.as_ref() {
                    Some(graph) if !force_direct => graph
                        .launch()
                        .map_err(|e| NeuralNetError(format!("graph launch: {e}")))?,
                    // 降级直连：逐 kernel 提交（慢但正确）。
                    _ => h
                        .model
                        .apply(&h.rt, stream, &mut st.ws, &st.in_spatial, &st.in_global)
                        .map_err(|e| NeuralNetError(format!("CUDA forward failed: {e}")))?,
                }
                let e3 = std::time::Instant::now();
                stream
                    .memcpy_dtoh(&st.ws.out_policy, &mut st.out_policy_pin)
                    .map_err(|e| NeuralNetError(format!("dtoh policy: {e}")))?;
                stream
                    .memcpy_dtoh(&st.ws.out_value, &mut st.out_value_pin)
                    .map_err(|e| NeuralNetError(format!("dtoh value: {e}")))?;
                stream
                    .memcpy_dtoh(&st.ws.out_misc, &mut st.out_misc_pin)
                    .map_err(|e| NeuralNetError(format!("dtoh misc: {e}")))?;
                stream
                    .memcpy_dtoh(&st.ws.out_moremisc, &mut st.out_moremisc_pin)
                    .map_err(|e| NeuralNetError(format!("dtoh moremisc: {e}")))?;
                stream
                    .memcpy_dtoh(&st.ws.out_ownership, &mut st.out_own_pin)
                    .map_err(|e| NeuralNetError(format!("dtoh ownership: {e}")))?;
                let e4 = std::time::Instant::now();
                // 流级同步（WC pinned 内存的 host 读需设备写全局可见）
                stream
                    .synchronize()
                    .map_err(|e| NeuralNetError(format!("sync: {e}")))?;
                let e5 = std::time::Instant::now();
                let host = CudaOutputsHost {
                    policy: st
                        .out_policy_pin
                        .as_slice()
                        .map_err(|e| NeuralNetError(format!("sync policy: {e}")))?
                        .to_vec(),
                    value: st
                        .out_value_pin
                        .as_slice()
                        .map_err(|e| NeuralNetError(format!("sync value: {e}")))?
                        .to_vec(),
                    misc: st
                        .out_misc_pin
                        .as_slice()
                        .map_err(|e| NeuralNetError(format!("sync misc: {e}")))?
                        .to_vec(),
                    moremisc: st
                        .out_moremisc_pin
                        .as_slice()
                        .map_err(|e| NeuralNetError(format!("sync moremisc: {e}")))?
                        .to_vec(),
                    ownership: st
                        .out_own_pin
                        .as_slice()
                        .map_err(|e| NeuralNetError(format!("sync ownership: {e}")))?
                        .to_vec(),
                };
                let e6 = std::time::Instant::now();
                if host_prof {
                    use std::sync::atomic::{AtomicU64, Ordering};
                    static CNT: AtomicU64 = AtomicU64::new(0);
                    static A_FILLPIN: AtomicU64 = AtomicU64::new(0);
                    static A_HTOD: AtomicU64 = AtomicU64::new(0);
                    static A_LAUNCH: AtomicU64 = AtomicU64::new(0);
                    static A_DTOH: AtomicU64 = AtomicU64::new(0);
                    static A_SYNC: AtomicU64 = AtomicU64::new(0);
                    static A_VEC: AtomicU64 = AtomicU64::new(0);
                    let c = CNT.fetch_add(1, Ordering::Relaxed) + 1;
                    let push = |acc: &AtomicU64, d: std::time::Duration| {
                        let us = d.as_micros() as u64;
                        acc.fetch_add(us, Ordering::Relaxed) + us
                    };
                    let a1 = push(&A_FILLPIN, e1 - e0);
                    let a2 = push(&A_HTOD, e2 - e1);
                    let a3 = push(&A_LAUNCH, e3 - e2);
                    let a4 = push(&A_DTOH, e4 - e3);
                    let a5 = push(&A_SYNC, e5 - e4);
                    let a6 = push(&A_VEC, e6 - e5);
                    if c % 100 == 0 {
                        eprintln!(
                            "[cuda-exec] n={c} fillpin={:.1}us htod={:.1}us launch={:.1}us dtoh={:.1}us sync={:.1}us tovec={:.1}us",
                            a1 as f64 / c as f64, a2 as f64 / c as f64,
                            a3 as f64 / c as f64, a4 as f64 / c as f64,
                            a5 as f64 / c as f64, a6 as f64 / c as f64
                        );
                    }
                }
                host
            };
            let t2 = std::time::Instant::now();

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

            let t3 = std::time::Instant::now();
            if host_prof {
                use std::sync::atomic::{AtomicU64, Ordering};
                static CNT: AtomicU64 = AtomicU64::new(0);
                static ACC_FILL: AtomicU64 = AtomicU64::new(0);
                static ACC_EXEC: AtomicU64 = AtomicU64::new(0);
                static ACC_DECODE: AtomicU64 = AtomicU64::new(0);
                let c = CNT.fetch_add(1, Ordering::Relaxed) + 1;
                let fill = (t1 - t0).as_micros() as u64;
                let exec = (t2 - t1).as_micros() as u64;
                let decode = (t3 - t2).as_micros() as u64;
                let a_fill = ACC_FILL.fetch_add(fill, Ordering::Relaxed) + fill;
                let a_exec = ACC_EXEC.fetch_add(exec, Ordering::Relaxed) + exec;
                let a_decode = ACC_DECODE.fetch_add(decode, Ordering::Relaxed) + decode;
                if c % 100 == 0 {
                    eprintln!(
                        "[cuda-host] n={c} avg fill={:.1}us exec={:.1}us decode={:.1}us total={:.1}us",
                        a_fill as f64 / c as f64,
                        a_exec as f64 / c as f64,
                        a_decode as f64 / c as f64,
                        (a_fill + a_exec + a_decode) as f64 / c as f64
                    );
                }
            }

            Ok(())
        }
    }
}

#[cfg(feature = "cuda")]
pub use backend_impl::CudaBackend;
