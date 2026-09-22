//! Native W8A8 projections: per-output-channel weights, dynamic per-row
//! activations, INT32 dot products, FP32 scaling and residual accumulation.
//! Shape-dependent state belongs to a workspace/stream, never to a global
//! scratch buffer. No floating point GEMM fallback can masquerade as INT8.

use super::cuda::CudaRuntime;
use cudarc::cublaslt::sys;
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
};
use std::{collections::HashMap, sync::Arc};

pub const QUANTIZATION_VERSION: &str = "w8a8-row-out-rne-v1";
const WARP_MIN_ROWS: usize = 1024;
const WARP_MAX_HIDDEN: usize = 512;

fn report_row_kernel(warp: bool, swiglu: bool, rows: usize, width: usize) {
    if super::cuda_exec::capturing() {
        return;
    }
    static QUANT_BLOCK: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    static QUANT_WARP: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    static SWIGLU_BLOCK: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    static SWIGLU_WARP: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    let slot = match (swiglu, warp) {
        (false, false) => &QUANT_BLOCK,
        (false, true) => &QUANT_WARP,
        (true, false) => &SWIGLU_BLOCK,
        (true, true) => &SWIGLU_WARP,
    };
    slot.get_or_init(|| {
        let name = if swiglu {
            "int8_swiglu_quantize"
        } else {
            "int8_quantize"
        };
        let launch = if warp { "warp4" } else { "cta256" };
        eprintln!("[cuda-tactic] name={name} launch={launch} rows={rows} width={width}");
    });
}

/// Explicit load-time mixed-precision policy. Zero retains the original all-FFN
/// path; the threshold is inclusive and expressed in the model's unpadded width.
pub fn parse_min_ffn_width(value: &str) -> Result<usize, String> {
    let width = value
        .parse::<usize>()
        .map_err(|_| "cudaInt8MinFfnWidth must be an integer in 0..=8192".to_string())?;
    if width > 8192 || width % 8 != 0 {
        return Err("cudaInt8MinFfnWidth must be 8-aligned and in 0..=8192".into());
    }
    Ok(width)
}

#[derive(Clone)]
pub struct Int8Kernels {
    quantize: CudaFunction,
    quantize_warp: CudaFunction,
    half: CudaFunction,
    residual: CudaFunction,
    swiglu_quantize: CudaFunction,
    swiglu_warp512: CudaFunction,
}

impl Int8Kernels {
    pub fn load(rt: &CudaRuntime) -> Result<Self, String> {
        if rt.cublaslt_handle().is_none() {
            return Err("INT8 requires cuBLASLt".into());
        }
        Ok(Self {
            quantize: rt.get_func("int8_quantize_rows_kernel")?,
            quantize_warp: rt.get_func("int8_quantize_rows_warp_kernel")?,
            half: rt.get_func("int8_dequantize_half_kernel")?,
            residual: rt.get_func("int8_dequantize_residual_kernel")?,
            swiglu_quantize: rt.get_func("int8_swiglu_quantize_kernel")?,
            swiglu_warp512: rt.get_func("int8_swiglu_quantize_warp512_kernel")?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Int8Scope {
    Ffn,
    Transformer,
}

impl Int8Scope {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "ffn" => Ok(Self::Ffn),
            "transformer" => Ok(Self::Transformer),
            _ => Err(format!(
                "cudaInt8Scope={value:?}; expected ffn or transformer"
            )),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Ffn => "ffn",
            Self::Transformer => "transformer",
        }
    }
}

/// CPU representation is public for independent numerical/layout tests.
pub struct QuantizedWeights {
    pub values: Vec<i8>,
    pub scales: Vec<f32>,
    pub n: usize,
    pub k: usize,
    pub kp: usize,
}

impl QuantizedWeights {
    pub fn new(weights: &[f32], n: usize, k: usize) -> Result<Self, String> {
        if n == 0
            || n > i32::MAX as usize
            || k == 0
            || k > 8192
            || n % 4 != 0
            || n.checked_mul(k) != Some(weights.len())
        {
            return Err(format!("invalid INT8 weight shape N={n} K={k}"));
        }
        if !weights.iter().all(|v| v.is_finite()) {
            return Err("INT8 weights contain NaN or infinity".into());
        }
        let kp = k.div_ceil(16) * 16;
        let count = n.checked_mul(kp).ok_or("INT8 weight size overflow")?;
        let mut values = vec![0; count];
        let mut scales = Vec::with_capacity(n);
        for row in 0..n {
            let w = &weights[row * k..(row + 1) * k];
            let amax = w.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            let scale = if amax == 0.0 {
                1.0
            } else {
                (amax / 127.0).max(f32::MIN_POSITIVE)
            };
            scales.push(scale);
            for col in 0..k {
                values[row * kp + col] =
                    (w[col] / scale).round_ties_even().clamp(-127.0, 127.0) as i8;
            }
        }
        Ok(Self {
            values,
            scales,
            n,
            k,
            kp,
        })
    }
}

pub struct Int8Weight {
    pub data: CudaSlice<i8>,
    pub scales: CudaSlice<f32>,
    pub n: usize,
    pub k: usize,
    pub kp: usize,
}

impl Int8Weight {
    pub fn upload(
        stream: &Arc<CudaStream>,
        weights: &[f32],
        n: usize,
        k: usize,
    ) -> Result<Self, String> {
        let q = QuantizedWeights::new(weights, n, k)?;
        Ok(Self {
            data: stream.clone_htod(&q.values).map_err(|e| e.to_string())?,
            scales: stream.clone_htod(&q.scales).map_err(|e| e.to_string())?,
            n,
            k,
            kp: q.kp,
        })
    }
}

struct LtPlan {
    desc: sys::cublasLtMatmulDesc_t,
    a: sys::cublasLtMatrixLayout_t,
    b: sys::cublasLtMatrixLayout_t,
    c: sys::cublasLtMatrixLayout_t,
    algo: sys::cublasLtMatmulAlgo_t,
}

// These opaque descriptors are immutable after construction. The containing
// workspace is moved with its handle and used on one stream at a time.
unsafe impl Send for LtPlan {}

impl Drop for LtPlan {
    fn drop(&mut self) {
        unsafe {
            if !self.desc.is_null() {
                sys::cublasLtMatmulDescDestroy(self.desc);
            }
            for p in [self.a, self.b, self.c] {
                if !p.is_null() {
                    sys::cublasLtMatrixLayoutDestroy(p);
                }
            }
        }
    }
}

fn check(status: sys::cublasStatus_t, operation: &str) -> Result<(), String> {
    if status == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(format!("INT8 {operation}: {status:?}"))
    }
}

impl LtPlan {
    fn new(rt: &CudaRuntime, m: usize, n: usize, k: usize) -> Result<Self, String> {
        let handle = rt.cublaslt_handle().ok_or("INT8 requires cuBLASLt")?;
        let mut p = Self {
            desc: std::ptr::null_mut(),
            a: std::ptr::null_mut(),
            b: std::ptr::null_mut(),
            c: std::ptr::null_mut(),
            algo: unsafe { std::mem::zeroed() },
        };
        unsafe {
            // INT32 output requires INT32 alpha/beta. FP32 scale + INT32
            // output was the invalid combination in the old INT8 probe.
            check(
                sys::cublasLtMatmulDescCreate(
                    &mut p.desc,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32I,
                    sys::cudaDataType_t::CUDA_R_32I,
                ),
                "create descriptor",
            )?;
            let ta = 1u32; // cublasOperation_t::CUBLAS_OP_T
            let tb = 0u32; // cublasOperation_t::CUBLAS_OP_N
            for (attr, value) in [
                (
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                    ta,
                ),
                (
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                    tb,
                ),
            ] {
                check(
                    sys::cublasLtMatmulDescSetAttribute(
                        p.desc,
                        attr,
                        &value as *const _ as *const _,
                        std::mem::size_of_val(&value),
                    ),
                    "set transpose",
                )?;
            }
            check(
                sys::cublasLtMatrixLayoutCreate(
                    &mut p.a,
                    sys::cudaDataType_t::CUDA_R_8I,
                    k as u64,
                    n as u64,
                    k as i64,
                ),
                "weight layout",
            )?;
            check(
                sys::cublasLtMatrixLayoutCreate(
                    &mut p.b,
                    sys::cudaDataType_t::CUDA_R_8I,
                    k as u64,
                    m as u64,
                    k as i64,
                ),
                "activation layout",
            )?;
            check(
                sys::cublasLtMatrixLayoutCreate(
                    &mut p.c,
                    sys::cudaDataType_t::CUDA_R_32I,
                    n as u64,
                    m as u64,
                    n as i64,
                ),
                "output layout",
            )?;
            let mut pref = std::ptr::null_mut();
            check(
                sys::cublasLtMatmulPreferenceCreate(&mut pref),
                "create preference",
            )?;
            let ws = rt.cublaslt_workspace_len();
            let status = sys::cublasLtMatmulPreferenceSetAttribute(
                pref,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws as *const _ as *const _,
                std::mem::size_of_val(&ws),
            );
            if let Err(e) = check(status, "set workspace") {
                sys::cublasLtMatmulPreferenceDestroy(pref);
                return Err(e);
            }
            let mut heuristic: sys::cublasLtMatmulHeuristicResult_t = std::mem::zeroed();
            let mut count = 0;
            let status = sys::cublasLtMatmulAlgoGetHeuristic(
                handle,
                p.desc,
                p.a,
                p.b,
                p.c,
                p.c,
                pref,
                1,
                &mut heuristic,
                &mut count,
            );
            sys::cublasLtMatmulPreferenceDestroy(pref);
            check(status, "select integer GEMM")?;
            if count != 1 || heuristic.state != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(format!(
                    "no INT8 GEMM for M={m} N={n} K={k}; no FP16 fallback"
                ));
            }
            p.algo = heuristic.algo;
            let mut flags = 0u64;
            let mut written = 0usize;
            check(
                sys::cublasLtMatmulAlgoCapGetAttribute(
                    &p.algo,
                    sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
                    &mut flags as *mut _ as *mut _,
                    std::mem::size_of_val(&flags),
                    &mut written,
                ),
                "inspect integer algorithm",
            )?;
            // cublasLt.h: IMMA = 0x04; make the actual selected route visible.
            static REPORTED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
            REPORTED.get_or_init(|| eprintln!("[cuda-int8] launch=cublaslt-int8 accumulator=int32 imma={} numerical_flags={flags:#x} M={m} N={n} K={k}", flags & 4 != 0));
        }
        Ok(p)
    }
}

pub struct Int8Workspace {
    pub quantized: CudaSlice<i8>,
    pub scales: CudaSlice<f32>,
    pub dots: CudaSlice<i32>,
    plans: HashMap<(usize, usize, usize), LtPlan>,
    kernels: Int8Kernels,
    stream_id: usize,
}

impl Int8Workspace {
    pub fn new(
        rt: &CudaRuntime,
        stream: &Arc<CudaStream>,
        rows: usize,
        max_k: usize,
        max_n: usize,
    ) -> Result<Self, String> {
        Self::from_kernels(&Int8Kernels::load(rt)?, stream, rows, max_k, max_n)
    }

    pub fn from_kernels(
        kernels: &Int8Kernels,
        stream: &Arc<CudaStream>,
        rows: usize,
        max_k: usize,
        max_n: usize,
    ) -> Result<Self, String> {
        if rows == 0
            || rows > i32::MAX as usize
            || max_k == 0
            || max_k > 8192
            || max_n == 0
            || max_n > i32::MAX as usize
        {
            return Err("invalid INT8 workspace dimensions".into());
        }
        let k_count = rows
            .checked_mul(max_k.div_ceil(16) * 16)
            .ok_or("INT8 workspace size overflow")?;
        let n_count = rows
            .checked_mul(max_n)
            .filter(|&v| v <= u32::MAX as usize)
            .ok_or("INT8 output exceeds kernel indexing limit")?;
        Ok(Self {
            quantized: stream.alloc_zeros(k_count).map_err(|e| e.to_string())?,
            scales: stream.alloc_zeros(rows).map_err(|e| e.to_string())?,
            dots: stream.alloc_zeros(n_count).map_err(|e| e.to_string())?,
            plans: HashMap::new(),
            kernels: kernels.clone(),
            stream_id: stream.cu_stream() as usize,
        })
    }

    fn multiply(
        &mut self,
        rt: &CudaRuntime,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<u16>,
        stride: usize,
        weight: &Int8Weight,
        rows: usize,
    ) -> Result<(), String> {
        let (n, k) = (weight.n, weight.kp);
        if stream.cu_stream() as usize != self.stream_id {
            return Err("INT8 workspace belongs to another CUDA stream".into());
        }
        if rows == 0
            || rows > i32::MAX as usize
            || stride > i32::MAX as usize
            || stride < weight.k
            || input.len() < rows.checked_mul(stride).ok_or("INT8 input size overflow")?
            || self.quantized.len() < rows.checked_mul(k).ok_or("INT8 activation size overflow")?
            || self.scales.len() < rows
            || self.dots.len() < rows.checked_mul(n).ok_or("INT8 output size overflow")?
        {
            return Err("INT8 projection buffer/stride mismatch".into());
        }
        // B1/B2 need more CTAs to hide latency. Larger token batches can use
        // four rows per block and avoid block-wide reduction barriers.
        let warp = rows >= WARP_MIN_ROWS;
        report_row_kernel(warp, false, rows, weight.k);
        unsafe {
            stream
                .launch_builder(if warp {
                    &self.kernels.quantize_warp
                } else {
                    &self.kernels.quantize
                })
                .arg(input)
                .arg(&mut self.quantized)
                .arg(&mut self.scales)
                .arg(&(rows as i32))
                .arg(&(weight.k as i32))
                .arg(&(stride as i32))
                .arg(&(k as i32))
                .launch(LaunchConfig {
                    grid_dim: (if warp { rows.div_ceil(4) } else { rows } as u32, 1, 1),
                    block_dim: (if warp { 128 } else { 256 }, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        self.integer_gemm(rt, stream, weight, rows)
    }

    fn integer_gemm(
        &mut self,
        rt: &CudaRuntime,
        stream: &Arc<CudaStream>,
        weight: &Int8Weight,
        rows: usize,
    ) -> Result<(), String> {
        let key = (rows, weight.n, weight.kp);
        if !self.plans.contains_key(&key) {
            if super::cuda_exec::capturing() {
                return Err("INT8 GEMM must be warmed before graph capture".into());
            }
            self.plans
                .insert(key, LtPlan::new(rt, rows, weight.n, weight.kp)?);
        }
        let plan = &self.plans[&key];
        unsafe {
            let (a, _a) = weight.data.device_ptr(stream);
            let (b, _b) = self.quantized.device_ptr(stream);
            let (c, _c) = self.dots.device_ptr_mut(stream);
            let (workspace, _) = rt.cublaslt_workspace_ptr(stream);
            if workspace == 0 {
                return Err("INT8 cuBLASLt workspace allocation failed".into());
            }
            let alpha = 1i32;
            let beta = 0i32;
            check(
                sys::cublasLtMatmul(
                    rt.cublaslt_handle().ok_or("INT8 requires cuBLASLt")?,
                    plan.desc,
                    &alpha as *const _ as *const _,
                    a as *const _,
                    plan.a,
                    b as *const _,
                    plan.b,
                    &beta as *const _ as *const _,
                    c as *const _,
                    plan.c,
                    c as *mut _,
                    plan.c,
                    &plan.algo,
                    workspace as *mut _,
                    rt.cublaslt_workspace_len(),
                    stream.cu_stream() as *mut _,
                ),
                "integer GEMM",
            )?;
        }
        Ok(())
    }

    pub fn project_half(
        &mut self,
        rt: &CudaRuntime,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<u16>,
        stride: usize,
        weight: &Int8Weight,
        output: &mut CudaSlice<u16>,
        rows: usize,
    ) -> Result<(), String> {
        let count = rows
            .checked_mul(weight.n)
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or("INT8 output size overflow")?;
        if output.len() < count {
            return Err("INT8 half output too small".into());
        }
        self.multiply(rt, stream, input, stride, weight, rows)?;
        unsafe {
            stream
                .launch_builder(&self.kernels.half)
                .arg(&self.dots)
                .arg(&self.scales)
                .arg(&weight.scales)
                .arg(output)
                .arg(&(rows as i32))
                .arg(&(weight.n as i32))
                .launch(LaunchConfig::for_num_elems(count as u32))
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn project_residual(
        &mut self,
        rt: &CudaRuntime,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<u16>,
        stride: usize,
        weight: &Int8Weight,
        output: &mut CudaSlice<f32>,
        rows: usize,
    ) -> Result<(), String> {
        let count = rows
            .checked_mul(weight.n)
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or("INT8 output size overflow")?;
        if output.len() < count {
            return Err("INT8 residual output too small".into());
        }
        self.multiply(rt, stream, input, stride, weight, rows)?;
        self.write_residual(stream, weight, output, rows, count)
    }

    fn write_residual(
        &mut self,
        stream: &Arc<CudaStream>,
        weight: &Int8Weight,
        output: &mut CudaSlice<f32>,
        rows: usize,
        count: usize,
    ) -> Result<(), String> {
        unsafe {
            stream
                .launch_builder(&self.kernels.residual)
                .arg(&self.dots)
                .arg(&self.scales)
                .arg(&weight.scales)
                .arg(output)
                .arg(&(rows as i32))
                .arg(&(weight.n as i32))
                .launch(LaunchConfig::for_num_elems(count as u32))
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Fused FFN preserves the unfused projection/activation half rounding.
    pub fn ffn(
        &mut self,
        rt: &CudaRuntime,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<u16>,
        dual: &Int8Weight,
        down: &Int8Weight,
        output: &mut CudaSlice<f32>,
        rows: usize,
    ) -> Result<(), String> {
        let count = rows
            .checked_mul(down.n)
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or("INT8 FFN size overflow")?;
        if dual.n != 2 * down.k
            || dual.k != down.n
            || output.len() < count
            || self.quantized.len() < rows.checked_mul(down.kp).ok_or("INT8 FFN size overflow")?
            || self.dots.len() < count
        {
            return Err("INT8 FFN shape/buffer mismatch".into());
        }
        self.multiply(rt, stream, input, dual.k, dual, rows)?;
        // Wide FFNs regress with one warp per row (register/latency pressure).
        let warp = down.k <= WARP_MAX_HIDDEN && rows >= WARP_MIN_ROWS;
        report_row_kernel(warp, true, rows, down.k);
        unsafe {
            stream
                .launch_builder(if warp {
                    &self.kernels.swiglu_warp512
                } else {
                    &self.kernels.swiglu_quantize
                })
                .arg(&self.dots)
                .arg(&mut self.scales)
                .arg(&dual.scales)
                .arg(&mut self.quantized)
                .arg(&(rows as i32))
                .arg(&(down.k as i32))
                .arg(&(down.kp as i32))
                .launch(LaunchConfig {
                    grid_dim: (if warp { rows.div_ceil(4) } else { rows } as u32, 1, 1),
                    block_dim: (if warp { 128 } else { 256 }, 1, 1),
                    shared_mem_bytes: if warp { 0 } else { (down.k * 4) as u32 },
                })
                .map_err(|e| e.to_string())?;
        }
        self.integer_gemm(rt, stream, down, rows)?;
        self.write_residual(stream, down, output, rows, count)
    }
}
