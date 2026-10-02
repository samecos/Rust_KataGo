//! Independent SM120 MXFP8 projections. This module does not select model
//! precision or change existing FP16/INT8 execution. All activation inputs are
//! existing half boundaries; static weights are quantized from original FP32.
//!
//! Prepare outside capture, warm and explicitly `complete_warmup`, then capture
//! `begin_forward -> projections -> remaining heads -> check_final_outputs`
//! when executing a full network. Copy the sticky status to each submission's
//! own pinned slot before recording its completion event. Validate that snapshot
//! after completion and before publishing outputs. No projection resets status.

use super::cuda::CudaRuntime;
use cudarc::cublaslt::sys;
use cudarc::driver::{
    sys as driver_sys, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig,
    PinnedHostSlice, PushKernelArg,
};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};

pub const QUANTIZATION_VERSION: &str = crate::quantization_plan::MXFP8_QUANTIZATION_VERSION;
pub const STATUS_WORDS: usize = 2;
pub const STATUS_NAN: u32 = 1;
pub const STATUS_INFINITY: u32 = 2;
pub const STATUS_OUTPUT_NONFINITE: u32 = 4;
pub const STATUS_WEIGHT_OVERFLOW: u32 = 8;
pub const STATUS_UNSUPPORTED_ARCH: u32 = 16;
/// Distinct from every projection ID and the u32::MAX "no error" sentinel.
pub const FINAL_OUTPUT_LAYER: u32 = u32::MAX - 1;
const MAX_K: usize = 8192;
const HEURISTIC_LIMIT: usize = 32;
// cublasLt.h numerical implementation masks (not exported by cudarc bindgen).
const TENSOR_OP_MASK: u64 = 0xfe;
const ACCUMULATOR_32F: u64 = 0x02 << 8;
const INPUT_8F_E4M3: u64 = 0x40 << 16;

fn check(status: sys::cublasStatus_t, operation: &str) -> Result<(), String> {
    if status == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(format!("MXFP8 {operation}: {status:?}; no FP16 fallback"))
    }
}

fn outside_capture(stream: &Arc<CudaStream>, operation: &str) -> Result<(), String> {
    if stream.capture_status().map_err(|e| e.to_string())?
        != driver_sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
    {
        Err(format!("MXFP8 {operation} must run before graph capture"))
    } else {
        Ok(())
    }
}

fn dimensions(rows: usize, k: usize) -> Result<(usize, usize, usize, usize), String> {
    if rows == 0 || rows > i32::MAX as usize || k == 0 || k > MAX_K {
        return Err(format!(
            "invalid MXFP8 dimensions rows={rows} K={k}; K must be 1..={MAX_K}"
        ));
    }
    let kp = k.div_ceil(32) * 32;
    let blocks4 = (kp / 32).div_ceil(4) * 4;
    let data_len = rows
        .checked_mul(kp)
        .filter(|&x| x <= u32::MAX as usize)
        .ok_or("MXFP8 data size exceeds kernel indexing limit")?;
    let scale_len = rows
        .div_ceil(128)
        .checked_mul(128)
        .and_then(|x| x.checked_mul(blocks4))
        .filter(|&x| x <= u32::MAX as usize)
        .ok_or("MXFP8 scale size exceeds kernel indexing limit")?;
    Ok((kp, blocks4, data_len, scale_len))
}

fn required_input(rows: usize, k: usize, stride: usize) -> Result<usize, String> {
    if stride < k || stride > i32::MAX as usize {
        return Err("MXFP8 input stride is smaller than K or exceeds i32".into());
    }
    (rows - 1)
        .checked_mul(stride)
        .and_then(|x| x.checked_add(k))
        .ok_or_else(|| "MXFP8 input size overflow".to_string())
}

fn aligned(pointer: u64, alignment: u64, name: &str) -> Result<(), String> {
    if pointer == 0 || pointer % alignment != 0 {
        Err(format!(
            "MXFP8 {name} pointer must be nonzero and {alignment}-byte aligned"
        ))
    } else {
        Ok(())
    }
}

#[derive(Clone)]
pub struct Mxfp8Kernels {
    quantize_half: CudaFunction,
    quantize_half_clear_fused: Option<CudaFunction>,
    quantize_float: CudaFunction,
    clear_scales: CudaFunction,
    begin: CudaFunction,
    check_half: CudaFunction,
    check_float: CudaFunction,
    check_final: CudaFunction,
    context: usize,
    runtime_version: usize,
}

impl Mxfp8Kernels {
    pub fn load(rt: &CudaRuntime) -> Result<Self, String> {
        let scale_clear_fusion = crate::tactic_plan::mxfp8_scale_clear_fusion_requested()?;
        rt.device.bind_to_thread().map_err(|e| e.to_string())?;
        let major = rt
            .device
            .attribute(driver_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| e.to_string())?;
        let minor = rt
            .device
            .attribute(driver_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| e.to_string())?;
        if (major, minor) != (12, 0) || rt.target_id != "sm_120" {
            return Err(format!(
                "MXFP8 requires SM120 and its compiled kernels; device={major}.{minor} target={}",
                rt.target_id
            ));
        }
        if rt.cublaslt_handle().is_none() {
            return Err("MXFP8 requires an available cuBLASLt handle".into());
        }
        let runtime_version = unsafe { sys::cublasLtGetVersion() };
        // Version alone is insufficient: descriptors and every actual shape
        // also pass the runtime's heuristic/AlgoCheck and real warm launch.
        if runtime_version < 120800 {
            return Err(format!(
                "MXFP8 requires block-scale-capable cuBLASLt; runtime={runtime_version}"
            ));
        }
        let kernels = Self {
            quantize_half: rt.get_func("mxfp8_quantize_half_kernel")?,
            quantize_half_clear_fused: if scale_clear_fusion {
                Some(rt.get_func("mxfp8_quantize_half_clear_fused_kernel")?)
            } else { None },
            quantize_float: rt.get_func("mxfp8_quantize_float_kernel")?,
            clear_scales: rt.get_func("mxfp8_clear_scales_kernel")?,
            begin: rt.get_func("mxfp8_begin_forward_kernel")?,
            check_half: rt.get_func("mxfp8_check_half_output_kernel")?,
            check_float: rt.get_func("mxfp8_check_float_output_kernel")?,
            check_final: rt.get_func("mxfp8_check_final_outputs_kernel")?,
            context: rt.device.cu_ctx() as usize,
            runtime_version,
        };
        eprintln!("[cuda-mxfp8] capability=sm120 cublaslt_runtime={runtime_version} quantization={QUANTIZATION_VERSION} fallback=none");
        Ok(kernels)
    }

    pub fn runtime_version(&self) -> usize {
        self.runtime_version
    }

    /// Immutable selected implementation, not a fresh environment lookup.
    pub fn scale_clear_fusion_enabled(&self) -> bool {
        self.quantize_half_clear_fused.is_some()
    }

    fn validate_selection(&self) -> Result<(), String> {
        if crate::tactic_plan::mxfp8_scale_clear_fusion_requested()? != self.scale_clear_fusion_enabled() {
            return Err("MXFP8 scale-clear fusion selection changed after preparation; reload the model/workspace".into());
        }
        Ok(())
    }

    fn report_half_quantize_path(&self) {
        static SEPARATE: AtomicBool = AtomicBool::new(false);
        static FUSED: AtomicBool = AtomicBool::new(false);
        let fused = self.scale_clear_fusion_enabled();
        let reported = if fused { &FUSED } else { &SEPARATE };
        if !reported.load(Ordering::Relaxed) && !reported.swap(true, Ordering::Relaxed) {
            eprintln!("[cuda-tactic] name=mxfp8_scale_clear_fusion requested={} launch={} effective={}",
                u8::from(fused), if fused { "fused" } else { "separate" }, u8::from(fused));
        }
    }

    fn check_context(&self, rt: &CudaRuntime, stream: &Arc<CudaStream>) -> Result<(), String> {
        if self.context != rt.device.cu_ctx() as usize || !Arc::ptr_eq(&rt.device, stream.context())
        {
            return Err("MXFP8 runtime, kernels and stream belong to different contexts".into());
        }
        Ok(())
    }

    fn reset(&self, stream: &Arc<CudaStream>, status: u64) -> Result<(), String> {
        unsafe {
            stream
                .launch_builder(&self.begin)
                .arg(&status)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| format!("MXFP8 clear forward status: {e}"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn quantize(
        &self,
        stream: &Arc<CudaStream>,
        half: bool,
        source: u64,
        data: u64,
        scale: u64,
        status: u64,
        rows: usize,
        k: usize,
        stride: usize,
        kp: usize,
        blocks4: usize,
        scale_capacity: usize,
        layer: u32,
    ) -> Result<(), String> {
        if half && let Some(fused) = &self.quantize_half_clear_fused {
            unsafe {
                stream.launch_builder(fused)
                    .arg(&source).arg(&data).arg(&scale).arg(&status)
                    .arg(&(k as i32)).arg(&(stride as i32)).arg(&(kp as i32))
                    .arg(&(blocks4 as i32)).arg(&layer).arg(&(rows as i32))
                    .arg(&(scale_capacity as u64))
                    .launch(LaunchConfig {
                        grid_dim: ((rows * (kp / 32)) as u32, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    })
            }.map_err(|e| format!("MXFP8 fused scale padding/block32 quantization: {e}"))?;
            self.report_half_quantize_path();
            return Ok(());
        }
        // Clear the whole allocation, including tiles unused by this shape.
        // A shared workspace may switch K in either direction between layers.
        unsafe {
            stream
                .launch_builder(&self.clear_scales)
                .arg(&scale)
                .arg(&(scale_capacity as u64))
                .launch(LaunchConfig::for_num_elems(scale_capacity as u32))
                .map_err(|e| format!("MXFP8 clear scale padding: {e}"))?;
            stream
                .launch_builder(if half {
                    &self.quantize_half
                } else {
                    &self.quantize_float
                })
                .arg(&source)
                .arg(&data)
                .arg(&scale)
                .arg(&status)
                .arg(&(k as i32))
                .arg(&(stride as i32))
                .arg(&(kp as i32))
                .arg(&(blocks4 as i32))
                .arg(&layer)
                .launch(LaunchConfig {
                    grid_dim: ((rows * (kp / 32)) as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| format!("MXFP8 block32 quantization: {e}"))?;
        }
        if half { self.report_half_quantize_path(); }
        Ok(())
    }
}

/// Immutable device weights. Upload completes before return so the same
/// weights may be read on other streams in this context without event tracking.
pub struct Mxfp8Weight {
    data: CudaSlice<u8>,
    scales: CudaSlice<u8>,
    n: usize,
    k: usize,
    kp: usize,
    context: usize,
}

impl Mxfp8Weight {
    #[allow(clippy::too_many_arguments)]
    pub fn upload_from_f32(
        rt: &CudaRuntime,
        stream: &Arc<CudaStream>,
        kernels: &Mxfp8Kernels,
        source: &[f32],
        n: usize,
        k: usize,
        stride: usize,
    ) -> Result<Arc<Self>, String> {
        kernels.check_context(rt, stream)?;
        outside_capture(stream, "weight upload")?;
        let (kp, blocks4, count, scale_count) = dimensions(n, k)?;
        if source.len() < required_input(n, k, stride)? {
            return Err("MXFP8 original FP32 weight buffer is too small".into());
        }
        for row in 0..n {
            if source[row * stride..row * stride + k]
                .iter()
                .any(|x| !x.is_finite())
            {
                return Err(format!(
                    "MXFP8 original FP32 weights contain NaN or infinity in row {row}"
                ));
            }
        }
        let input = stream.clone_htod(source).map_err(|e| e.to_string())?;
        let data = stream.alloc_zeros::<u8>(count).map_err(|e| e.to_string())?;
        let scales = stream
            .alloc_zeros::<u8>(scale_count)
            .map_err(|e| e.to_string())?;
        let status = stream
            .alloc_zeros::<u32>(STATUS_WORDS)
            .map_err(|e| e.to_string())?;
        let (x, _x) = input.device_ptr(stream);
        let (q, _q) = data.device_ptr(stream);
        let (s, _s) = scales.device_ptr(stream);
        let (invalid, _invalid) = status.device_ptr(stream);
        kernels.reset(stream, invalid)?;
        kernels.quantize(
            stream,
            false,
            x,
            q,
            s,
            invalid,
            n,
            k,
            stride,
            kp,
            blocks4,
            scale_count,
            0,
        )?;
        let mut host_status = [0u32; STATUS_WORDS];
        stream
            .memcpy_dtoh(&status, &mut host_status)
            .map_err(|e| e.to_string())?;
        stream
            .synchronize()
            .map_err(|e| format!("MXFP8 static weight upload: {e}"))?;
        validate_completed_status(&host_status)?;
        // DevicePtr guards can borrow their owning allocations. Release them
        // before moving the completed immutable allocations into the weight.
        drop((_x, _q, _s, _invalid));
        Ok(Arc::new(Self {
            data,
            scales,
            n,
            k,
            kp,
            context: kernels.context,
        }))
    }

    pub fn n(&self) -> usize {
        self.n
    }
    pub fn k(&self) -> usize {
        self.k
    }
    pub fn padded_k(&self) -> usize {
        self.kp
    }
    pub fn quantized_data(&self) -> &CudaSlice<u8> {
        &self.data
    }
    pub fn packed_scales(&self) -> &CudaSlice<u8> {
        &self.scales
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mxfp8Output {
    Half,
    Float,
    FloatResidual,
}

impl Mxfp8Output {
    fn data_type(self) -> sys::cudaDataType_t {
        match self {
            Self::Half => sys::cudaDataType_t::CUDA_R_16F,
            Self::Float | Self::FloatResidual => sys::cudaDataType_t::CUDA_R_32F,
        }
    }
    fn beta(self) -> f32 {
        if self == Self::FloatResidual {
            1.0
        } else {
            0.0
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedProjectionId {
    workspace: u64,
    index: usize,
}

#[derive(Clone, Debug)]
pub struct Mxfp8PlanInfo {
    pub rows: usize,
    pub n: usize,
    pub k: usize,
    pub padded_k: usize,
    pub output: Mxfp8Output,
    pub algorithm_id: i32,
    pub tile_id: i32,
    pub split_k: i32,
    pub reduction_scheme: i32,
    pub numerical_flags: u64,
    pub workspace_bytes: usize,
    pub scale_a: u64,
    pub scale_b: u64,
    pub runtime_version: usize,
}

struct LtPreference(sys::cublasLtMatmulPreference_t);
impl Drop for LtPreference {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { sys::cublasLtMatmulPreferenceDestroy(self.0) };
        }
    }
}

struct LtPlan {
    desc: sys::cublasLtMatmulDesc_t,
    a: sys::cublasLtMatrixLayout_t,
    b: sys::cublasLtMatrixLayout_t,
    c: sys::cublasLtMatrixLayout_t,
    algo: sys::cublasLtMatmulAlgo_t,
}
// Immutable descriptors are shared only as keepalive references. All actual
// submissions use their workspace's fixed stream; there is no rebinding API.
unsafe impl Send for LtPlan {}
unsafe impl Sync for LtPlan {}
impl Drop for LtPlan {
    fn drop(&mut self) {
        unsafe {
            if !self.desc.is_null() {
                sys::cublasLtMatmulDescDestroy(self.desc);
            }
            for layout in [self.a, self.b, self.c] {
                if !layout.is_null() {
                    sys::cublasLtMatrixLayoutDestroy(layout);
                }
            }
        }
    }
}

impl LtPlan {
    fn set<T>(
        &self,
        attribute: sys::cublasLtMatmulDescAttributes_t,
        value: &T,
    ) -> Result<(), String> {
        check(
            unsafe {
                sys::cublasLtMatmulDescSetAttribute(
                    self.desc,
                    attribute,
                    value as *const T as *const _,
                    std::mem::size_of::<T>(),
                )
            },
            "set matmul descriptor attribute",
        )
    }

    fn config(&self, attribute: sys::cublasLtMatmulAlgoConfigAttributes_t) -> Result<i32, String> {
        let mut result = 0i32;
        let mut written = 0usize;
        check(
            unsafe {
                sys::cublasLtMatmulAlgoConfigGetAttribute(
                    &self.algo,
                    attribute,
                    &mut result as *mut _ as *mut _,
                    std::mem::size_of_val(&result),
                    &mut written,
                )
            },
            "read selected algorithm configuration",
        )?;
        if written != std::mem::size_of_val(&result) {
            return Err("MXFP8 algorithm configuration has an unexpected size".into());
        }
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        rt: &CudaRuntime,
        weight: &Mxfp8Weight,
        rows: usize,
        output: Mxfp8Output,
        scale_a: u64,
        scale_b: u64,
        workspace_len: usize,
        runtime_version: usize,
    ) -> Result<(Self, Mxfp8PlanInfo), String> {
        let mut plan = Self {
            desc: std::ptr::null_mut(),
            a: std::ptr::null_mut(),
            b: std::ptr::null_mut(),
            c: std::ptr::null_mut(),
            algo: unsafe { std::mem::zeroed() },
        };
        rt.device.bind_to_thread().map_err(|e| e.to_string())?;
        unsafe {
            check(
                sys::cublasLtMatmulDescCreate(
                    &mut plan.desc,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    sys::cudaDataType_t::CUDA_R_32F,
                ),
                "create descriptor",
            )?;
        }
        use sys::cublasLtMatmulDescAttributes_t as Attr;
        plan.set(Attr::CUBLASLT_MATMUL_DESC_TRANSA, &1i32)?;
        plan.set(Attr::CUBLASLT_MATMUL_DESC_TRANSB, &0i32)?;
        plan.set(
            Attr::CUBLASLT_MATMUL_DESC_POINTER_MODE,
            &sys::cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_HOST,
        )?;
        plan.set(Attr::CUBLASLT_MATMUL_DESC_FAST_ACCUM, &0i8)?;
        let mode = sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        plan.set(Attr::CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode)?;
        plan.set(Attr::CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode)?;
        plan.set(
            Attr::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            &(scale_a as *const std::ffi::c_void),
        )?;
        plan.set(
            Attr::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            &(scale_b as *const std::ffi::c_void),
        )?;
        unsafe {
            check(
                sys::cublasLtMatrixLayoutCreate(
                    &mut plan.a,
                    sys::cudaDataType_t::CUDA_R_8F_E4M3,
                    weight.kp as u64,
                    weight.n as u64,
                    weight.kp as i64,
                ),
                "create weight layout",
            )?;
            check(
                sys::cublasLtMatrixLayoutCreate(
                    &mut plan.b,
                    sys::cudaDataType_t::CUDA_R_8F_E4M3,
                    weight.kp as u64,
                    rows as u64,
                    weight.kp as i64,
                ),
                "create activation layout",
            )?;
            check(
                sys::cublasLtMatrixLayoutCreate(
                    &mut plan.c,
                    output.data_type(),
                    weight.n as u64,
                    rows as u64,
                    weight.n as i64,
                ),
                "create output layout",
            )?;
        }
        let handle = rt.cublaslt_handle().ok_or("MXFP8 requires cuBLASLt")?;
        let mut preference = LtPreference(std::ptr::null_mut());
        let mut candidates =
            vec![
                unsafe { std::mem::zeroed::<sys::cublasLtMatmulHeuristicResult_t>() };
                HEURISTIC_LIMIT
            ];
        let mut returned = 0;
        unsafe {
            check(
                sys::cublasLtMatmulPreferenceCreate(&mut preference.0),
                "create preference",
            )?;
            check(sys::cublasLtMatmulPreferenceSetAttribute(preference.0,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &workspace_len as *const _ as *const _, std::mem::size_of_val(&workspace_len)), "set workspace preference")?;
            check(
                sys::cublasLtMatmulAlgoGetHeuristic(
                    handle,
                    plan.desc,
                    plan.a,
                    plan.b,
                    plan.c,
                    plan.c,
                    preference.0,
                    HEURISTIC_LIMIT as i32,
                    candidates.as_mut_ptr(),
                    &mut returned,
                ),
                "get heuristic",
            )?;
        }
        if returned < 0 || returned as usize > candidates.len() {
            return Err("MXFP8 heuristic returned an invalid result count".into());
        }
        let mut selected = None;
        for (rank, candidate) in candidates.iter().take(returned as usize).enumerate() {
            let mut checked = unsafe { std::mem::zeroed::<sys::cublasLtMatmulHeuristicResult_t>() };
            let status = unsafe {
                sys::cublasLtMatmulAlgoCheck(
                    handle,
                    plan.desc,
                    plan.a,
                    plan.b,
                    plan.c,
                    plan.c,
                    &candidate.algo,
                    &mut checked,
                )
            };
            let mut flags = 0u64;
            let mut written = 0usize;
            let cap_status =
                unsafe {
                    sys::cublasLtMatmulAlgoCapGetAttribute(&candidate.algo,
                sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
                &mut flags as *mut _ as *mut _, std::mem::size_of_val(&flags), &mut written)
                };
            let audited = cap_status == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                && written == std::mem::size_of_val(&flags)
                && flags & TENSOR_OP_MASK != 0
                && flags & ACCUMULATOR_32F != 0
                && flags & INPUT_8F_E4M3 != 0;
            if candidate.state == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                && status == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                && checked.state == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                && checked.workspaceSize <= workspace_len
                && audited
            {
                selected = Some((candidate.algo, flags, checked.workspaceSize));
                break;
            }
            eprintln!("[cuda-mxfp8-candidate] rows={rows} n={} k={} rank={rank} heuristic={:?} check={status:?} state={:?} flags={flags:#x} audited={audited} workspace={}",
                weight.n, weight.kp, candidate.state, checked.state, checked.workspaceSize);
        }
        let (algo, flags, workspace_bytes) = selected.ok_or_else(|| format!(
            "no AlgoCheck-approved MXFP8 Tensor Core/E4M3/FP32 algorithm for rows={rows} N={} K={} output={output:?}, candidates={returned}; no FP16 fallback",
            weight.n, weight.kp))?;
        plan.algo = algo;
        use sys::cublasLtMatmulAlgoConfigAttributes_t as Config;
        let info = Mxfp8PlanInfo {
            rows,
            n: weight.n,
            k: weight.k,
            padded_k: weight.kp,
            output,
            algorithm_id: plan.config(Config::CUBLASLT_ALGO_CONFIG_ID)?,
            tile_id: plan.config(Config::CUBLASLT_ALGO_CONFIG_TILE_ID)?,
            split_k: plan.config(Config::CUBLASLT_ALGO_CONFIG_SPLITK_NUM)?,
            reduction_scheme: plan.config(Config::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)?,
            numerical_flags: flags,
            workspace_bytes,
            scale_a,
            scale_b,
            runtime_version,
        };
        Ok((plan, info))
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        rt: &CudaRuntime,
        stream: &Arc<CudaStream>,
        weight: u64,
        activation: u64,
        output: u64,
        kind: Mxfp8Output,
        workspace: u64,
        workspace_bytes: usize,
    ) -> Result<(), String> {
        let alpha = 1.0f32;
        let beta = kind.beta();
        check(
            unsafe {
                sys::cublasLtMatmul(
                    rt.cublaslt_handle().ok_or("MXFP8 requires cuBLASLt")?,
                    self.desc,
                    &alpha as *const _ as *const _,
                    weight as *const _,
                    self.a,
                    activation as *const _,
                    self.b,
                    &beta as *const _ as *const _,
                    output as *const _,
                    self.c,
                    output as *mut _,
                    self.c,
                    &self.algo,
                    workspace as *mut _,
                    workspace_bytes,
                    stream.cu_stream() as *mut _,
                )
            },
            "Tensor Core GEMM",
        )
    }
}

/// Shared only to keep graph device addresses alive. Rust never resizes these
/// allocations. Internal raw-pointer kernel writes are ordered on owner stream.
struct WorkspaceStorage {
    data: CudaSlice<u8>,
    scales: CudaSlice<u8>,
    status: CudaSlice<u32>,
    stream: Arc<CudaStream>,
}
impl Drop for WorkspaceStorage {
    fn drop(&mut self) {
        // Destruction, not the inference path. Keep outstanding kernels from
        // observing freed storage even if a caller drops a workspace early.
        let _ = self.stream.synchronize();
    }
}

struct PreparedProjection {
    plan: LtPlan,
    weight: Arc<Mxfp8Weight>,
    info: Mxfp8PlanInfo,
    launched: AtomicBool,
    warmed: AtomicBool,
    last_submitted_forward: AtomicU64,
}

/// Retain alongside a captured graph and drop only after destroying that graph.
/// It keeps descriptors, weights, scratch and the runtime's Lt workspace alive
/// even if the Rust workspace wrapper is moved or dropped. Input/output buffers
/// captured by the caller still require their own keepalive ownership.
pub struct Mxfp8GraphResources {
    projections: Vec<Arc<PreparedProjection>>,
    storage: Arc<WorkspaceStorage>,
    runtime: Arc<CudaRuntime>,
}
impl Drop for Mxfp8GraphResources {
    fn drop(&mut self) {
        let _ = self.storage.stream.synchronize();
        // Explicit reads document the keepalive-only fields and suppress dead
        // field warnings without exposing mutable raw descriptors.
        let _ = (&self.projections, &self.runtime);
    }
}

pub struct Mxfp8Workspace {
    // Plans drop before storage/runtime. GraphResources may prolong all three.
    projections: Vec<Arc<PreparedProjection>>,
    storage: Arc<WorkspaceStorage>,
    rt: Arc<CudaRuntime>,
    stream: Arc<CudaStream>,
    kernels: Mxfp8Kernels,
    rows: usize,
    max_k: usize,
    id: u64,
    lt_workspace: u64,
    lt_workspace_len: usize,
    began: bool,
    forward_generation: u64,
}

impl Drop for Mxfp8Workspace {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

impl Mxfp8Workspace {
    pub fn new(
        rt: Arc<CudaRuntime>,
        stream: Arc<CudaStream>,
        kernels: Mxfp8Kernels,
        rows: usize,
        max_k: usize,
    ) -> Result<Self, String> {
        kernels.check_context(&rt, &stream)?;
        kernels.validate_selection()?;
        outside_capture(&stream, "workspace allocation")?;
        let (_, _, count, scale_count) = dimensions(rows, max_k)?;
        let storage = Arc::new(WorkspaceStorage {
            data: stream.alloc_zeros(count).map_err(|e| e.to_string())?,
            scales: stream.alloc_zeros(scale_count).map_err(|e| e.to_string())?,
            status: stream
                .alloc_zeros(STATUS_WORDS)
                .map_err(|e| e.to_string())?,
            stream: stream.clone(),
        });
        let (lt_workspace, _) = rt.cublaslt_workspace_ptr(&stream);
        aligned(lt_workspace, 256, "cuBLASLt workspace")?;
        let lt_workspace_len = rt.cublaslt_workspace_len();
        if lt_workspace_len == 0 {
            return Err("MXFP8 cuBLASLt workspace is empty".into());
        }
        static NEXT_WORKSPACE: AtomicU64 = AtomicU64::new(1);
        let id = NEXT_WORKSPACE.fetch_add(1, Ordering::Relaxed);
        if id == 0 {
            return Err("MXFP8 workspace identity exhausted".into());
        }
        Ok(Self {
            projections: Vec::new(),
            storage,
            rt,
            stream,
            kernels,
            rows,
            max_k,
            id,
            lt_workspace,
            lt_workspace_len,
            began: false,
            forward_generation: 0,
        })
    }

    pub fn prepare_projection(
        &mut self,
        weight: Arc<Mxfp8Weight>,
        output: Mxfp8Output,
    ) -> Result<PreparedProjectionId, String> {
        self.kernels.validate_selection()?;
        outside_capture(&self.stream, "projection preparation")?;
        if weight.context != self.kernels.context || weight.k > self.max_k {
            return Err("MXFP8 projection weight context or K exceeds workspace capacity".into());
        }
        let alignment = if output == Mxfp8Output::Half { 8 } else { 4 };
        if weight.n % alignment != 0 {
            return Err(format!(
                "MXFP8 output N={} must be {alignment}-aligned; no implicit output padding",
                weight.n
            ));
        }
        self.rows
            .checked_mul(weight.n)
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or("MXFP8 output exceeds kernel indexing limit")?;
        let (a, _a) = weight.data.device_ptr(&self.stream);
        let (scale_a, _sa) = weight.scales.device_ptr(&self.stream);
        let (b, _b) = self.storage.data.device_ptr(&self.stream);
        let (scale_b, _sb) = self.storage.scales.device_ptr(&self.stream);
        for (pointer, name) in [
            (a, "weight"),
            (b, "activation"),
            (scale_a, "weight scale"),
            (scale_b, "activation scale"),
        ] {
            aligned(pointer, 16, name)?;
        }
        let (plan, info) = LtPlan::new(
            &self.rt,
            &weight,
            self.rows,
            output,
            scale_a,
            scale_b,
            self.lt_workspace_len,
            self.kernels.runtime_version,
        )?;
        drop((_a, _sa, _b, _sb));
        let id = PreparedProjectionId {
            workspace: self.id,
            index: self.projections.len(),
        };
        self.projections.push(Arc::new(PreparedProjection {
            plan,
            weight,
            info,
            launched: AtomicBool::new(false),
            warmed: AtomicBool::new(false),
            last_submitted_forward: AtomicU64::new(0),
        }));
        Ok(id)
    }

    fn projection(&self, id: PreparedProjectionId) -> Result<&PreparedProjection, String> {
        if id.workspace != self.id {
            return Err("MXFP8 projection belongs to another workspace".into());
        }
        self.projections
            .get(id.index)
            .map(Arc::as_ref)
            .ok_or_else(|| "invalid MXFP8 prepared projection ID".to_string())
    }

    pub fn plan_info(&self, id: PreparedProjectionId) -> Result<&Mxfp8PlanInfo, String> {
        Ok(&self.projection(id)?.info)
    }
    pub fn quantized_data(&self) -> &CudaSlice<u8> {
        &self.storage.data
    }
    pub fn packed_scales(&self) -> &CudaSlice<u8> {
        &self.storage.scales
    }
    pub fn status(&self) -> &CudaSlice<u32> {
        &self.storage.status
    }

    pub fn scale_clear_fusion_enabled(&self) -> bool {
        self.kernels.scale_clear_fusion_enabled()
    }

    /// Raw default/per-thread stream handles can be equal in distinct contexts.
    /// Validate both identities before obtaining pointers or scheduling work.
    pub fn validate_owner_stream(&self, stream: &Arc<CudaStream>) -> Result<(), String> {
        if stream.context().cu_ctx() != self.stream.context().cu_ctx()
            || stream.cu_stream() != self.stream.cu_stream()
        {
            return Err("MXFP8 buffer/operation requires the workspace owner stream and CUDA context".into());
        }
        Ok(())
    }

    /// Append once after all five raw heads have been produced, before any
    /// status snapshot or completion event. One asynchronous kernel; no reset,
    /// allocation or synchronization. External output writers must recheck.
    pub fn check_final_outputs(&self, outputs: [&CudaSlice<f32>; 5]) -> Result<(), String> {
        if !self.began {
            return Err("MXFP8 final output check requires begin_forward".into());
        }
        let mut total = 0usize;
        for output in outputs {
            self.validate_owner_stream(output.stream())?;
            if output.is_empty() {
                return Err("MXFP8 final output is empty".into());
            }
            total = total.checked_add(output.len()).filter(|&n| n <= u32::MAX as usize)
                .ok_or("MXFP8 final output size exceeds kernel indexing limit")?;
        }
        let (policy, _policy) = outputs[0].device_ptr(&self.stream);
        let (value, _value) = outputs[1].device_ptr(&self.stream);
        let (misc, _misc) = outputs[2].device_ptr(&self.stream);
        let (moremisc, _moremisc) = outputs[3].device_ptr(&self.stream);
        let (ownership, _ownership) = outputs[4].device_ptr(&self.stream);
        let (status, _status) = self.storage.status.device_ptr(&self.stream);
        unsafe {
            self.stream.launch_builder(&self.kernels.check_final)
                .arg(&policy).arg(&value).arg(&misc).arg(&moremisc).arg(&ownership).arg(&status)
                .arg(&(outputs[0].len() as u64)).arg(&(outputs[1].len() as u64))
                .arg(&(outputs[2].len() as u64)).arg(&(outputs[3].len() as u64))
                .arg(&(outputs[4].len() as u64)).arg(&FINAL_OUTPUT_LAYER)
                .launch(LaunchConfig::for_num_elems(total as u32))
        }.map_err(|e| format!("MXFP8 final raw output finite check: {e}"))?;
        Ok(())
    }

    /// Call outside capture once all projections are prepared. This token must
    /// live at least as long as the captured graph; it cannot protect resources
    /// added by a later prepare call, so obtain a new token for each new graph.
    pub fn graph_resources(&self) -> Result<Mxfp8GraphResources, String> {
        outside_capture(&self.stream, "graph resource retention")?;
        Ok(Mxfp8GraphResources {
            projections: self.projections.clone(),
            storage: self.storage.clone(),
            runtime: self.rt.clone(),
        })
    }

    /// Exactly once per full forward, including inside its captured graph.
    pub fn begin_forward(&mut self) -> Result<(), String> {
        // Selection was checked while preparing this immutable workspace. Full
        // network execution also freezes tactic_var globally; do not allocate
        // environment strings or choose a different route during a forward.
        let generation = self
            .forward_generation
            .checked_add(1)
            .ok_or("MXFP8 forward generation exhausted")?;
        let (status, _guard) = self.storage.status.device_ptr(&self.stream);
        self.kernels.reset(&self.stream, status)?;
        self.began = true;
        self.forward_generation = generation;
        Ok(())
    }

    /// Preparation-only completion barrier. A successful asynchronous submission
    /// is not enough to authorize capture: confirm both GPU execution and the
    /// sticky status of this exact forward. Earlier forwards erased by a reset
    /// cannot accidentally certify an unvalidated projection.
    pub fn complete_warmup(&mut self) -> Result<(), String> {
        outside_capture(&self.stream, "warmup completion")?;
        if !self.began {
            return Err("MXFP8 warmup requires begin_forward and actual projections".into());
        }
        let mut status = [0u32; STATUS_WORDS];
        self.stream
            .memcpy_dtoh(&self.storage.status, &mut status)
            .map_err(|e| format!("MXFP8 warmup status: {e}"))?;
        self.stream
            .synchronize()
            .map_err(|e| format!("MXFP8 warmup completion: {e}"))?;
        validate_completed_status(&status)?;
        let mut ran = false;
        for projection in &self.projections {
            if projection.last_submitted_forward.load(Ordering::Relaxed) == self.forward_generation
            {
                projection.warmed.store(true, Ordering::Relaxed);
                ran = true;
            }
        }
        if !ran {
            return Err("MXFP8 warmup did not execute a projection in this forward".into());
        }
        Ok(())
    }

    fn check_input(
        &self,
        projection: &PreparedProjection,
        input: &CudaSlice<u16>,
        stride: usize,
        output_len: usize,
        layer: u32,
    ) -> Result<(), String> {
        if !self.began {
            return Err("MXFP8 begin_forward must precede projections".into());
        }
        if layer >= FINAL_OUTPUT_LAYER {
            return Err("MXFP8 layer IDs u32::MAX-1/MAX are reserved for final outputs/no error".into());
        }
        self.validate_owner_stream(input.stream())?;
        if input.len() < required_input(self.rows, projection.weight.k, stride)?
            || output_len < self.rows * projection.weight.n
        {
            return Err("MXFP8 input/output buffer is too small".into());
        }
        let capturing = self.stream.capture_status().map_err(|e| e.to_string())?
            != driver_sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        if capturing && !projection.warmed.load(Ordering::Relaxed) {
            return Err(
                "MXFP8 projection requires a direct launch and complete_warmup before capture"
                    .into(),
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn project(
        &self,
        projection: &PreparedProjection,
        input: &CudaSlice<u16>,
        stride: usize,
        output: u64,
        layer: u32,
    ) -> Result<(), String> {
        aligned(output, 16, "output")?;
        let (source, _source) = input.device_ptr(&self.stream);
        let (data, _data) = self.storage.data.device_ptr(&self.stream);
        let (scale, _scale) = self.storage.scales.device_ptr(&self.stream);
        let (status, _status) = self.storage.status.device_ptr(&self.stream);
        let (weight, _weight) = projection.weight.data.device_ptr(&self.stream);
        let kp = projection.weight.kp;
        self.kernels.quantize(
            &self.stream,
            true,
            source,
            data,
            scale,
            status,
            self.rows,
            projection.weight.k,
            stride,
            kp,
            (kp / 32).div_ceil(4) * 4,
            self.storage.scales.len(),
            layer,
        )?;
        projection.plan.run(
            &self.rt,
            &self.stream,
            weight,
            data,
            output,
            projection.info.output,
            self.lt_workspace,
            self.lt_workspace_len,
        )?;
        let count = self.rows * projection.weight.n;
        unsafe {
            self.stream
                .launch_builder(if projection.info.output == Mxfp8Output::Half {
                    &self.kernels.check_half
                } else {
                    &self.kernels.check_float
                })
                .arg(&output)
                .arg(&status)
                .arg(&(count as u64))
                .arg(&layer)
                .launch(LaunchConfig::for_num_elems(count as u32))
        }
        .map_err(|e| format!("MXFP8 output finite check: {e}"))?;
        projection
            .last_submitted_forward
            .store(self.forward_generation, Ordering::Relaxed);
        if !projection.launched.swap(true, Ordering::Relaxed) {
            let info = &projection.info;
            eprintln!("[cuda-mxfp8] launch=cublaslt-mxfp8 rows={} n={} k={} kp={} output={:?} algorithm={} tile={} split_k={} reduction={} flags={:#x} workspace={} cublaslt_runtime={}",
                info.rows, info.n, info.k, info.padded_k, info.output, info.algorithm_id,
                info.tile_id, info.split_k, info.reduction_scheme, info.numerical_flags,
                info.workspace_bytes, info.runtime_version);
        }
        Ok(())
    }

    pub fn project_half(
        &mut self,
        id: PreparedProjectionId,
        layer_id: u32,
        input: &CudaSlice<u16>,
        stride: usize,
        output: &mut CudaSlice<u16>,
    ) -> Result<(), String> {
        let projection = self.projection(id)?;
        if projection.info.output != Mxfp8Output::Half {
            return Err("MXFP8 project_half received a non-half projection".into());
        }
        self.validate_owner_stream(output.stream())?;
        self.check_input(projection, input, stride, output.len(), layer_id)?;
        let (destination, _guard) = output.device_ptr_mut(&self.stream);
        self.project(projection, input, stride, destination, layer_id)
    }

    pub fn project_f32(
        &mut self,
        id: PreparedProjectionId,
        layer_id: u32,
        input: &CudaSlice<u16>,
        stride: usize,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), String> {
        let projection = self.projection(id)?;
        if projection.info.output == Mxfp8Output::Half {
            return Err("MXFP8 project_f32 received a half projection".into());
        }
        self.validate_owner_stream(output.stream())?;
        self.check_input(projection, input, stride, output.len(), layer_id)?;
        let (destination, _guard) = output.device_ptr_mut(&self.stream);
        self.project(projection, input, stride, destination, layer_id)
    }

    /// Enqueue before this slot's done_event and before the next forward reset.
    /// A different pinned allocation is required for each in-flight slot.
    pub fn enqueue_status_copy(
        &self,
        destination: &mut PinnedHostSlice<u32>,
    ) -> Result<(), String> {
        if destination.len() != STATUS_WORDS {
            return Err(format!(
                "MXFP8 pinned status must contain {STATUS_WORDS} words"
            ));
        }
        if !Arc::ptr_eq(destination.context(), &self.rt.device) {
            return Err("MXFP8 pinned status belongs to another CUDA context".into());
        }
        self.stream
            .memcpy_dtoh(&self.storage.status, destination)
            .map_err(|e| format!("MXFP8 status snapshot: {e}"))
    }
}

/// Call only after the corresponding DTOH copy's completion event/stream has
/// completed. Safe placeholder results do not make an invalid forward valid.
pub fn validate_completed_status(status: &[u32]) -> Result<(), String> {
    if status.len() != STATUS_WORDS {
        return Err("invalid MXFP8 status snapshot length".into());
    }
    if status[0] == 0 && status[1] == u32::MAX {
        return Ok(());
    }
    Err(format!("MXFP8 forward rejected: flags={:#x} first_layer={}{} (NaN=1 infinity=2 output_nonfinite=4 weight_overflow=8 unsupported_arch=16); discard all outputs",
        status[0], status[1], if status[1] == FINAL_OUTPUT_LAYER { " (final_raw_heads)" } else { "" }))
}
