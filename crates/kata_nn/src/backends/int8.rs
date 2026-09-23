//! Native W8A8 projections: per-output-channel weights, dynamic per-row
//! activations, INT32 dot products, FP32 scaling and residual accumulation.
//! Shape-dependent state belongs to a workspace/stream, never to a global
//! scratch buffer. No floating point GEMM fallback can masquerade as INT8.

use super::cuda::CudaRuntime;
use cudarc::cublaslt::sys;
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
};
use cudarc::driver::{result as driver_result, sys as driver_sys};
use std::{collections::HashMap, sync::Arc, time::Instant};

pub const QUANTIZATION_VERSION: &str = "w8a8-row-out-rne-v1";
const WARP_MIN_ROWS: usize = 1024;
const WARP_MAX_HIDDEN: usize = 512;
const GEMM_TUNE_CANDIDATES: usize = 16;
const GEMM_TUNE_REPEATS: usize = 128;

fn gemm_tune_requested() -> Result<bool, String> {
    match crate::tactic_plan::tactic_var("KATAGO_CUDA_INT8_GEMM_TUNE") {
        Ok(v) if v == "1" => Ok(true),
        Ok(v) if v == "0" => Ok(false),
        Err(std::env::VarError::NotPresent) => Ok(false),
        _ => Err("invalid KATAGO_CUDA_INT8_GEMM_TUNE; expected 0 or 1".into()),
    }
}

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
    rms384: CudaFunction,
    rms512: CudaFunction,
    rms_fusion: Option<bool>,
    gemm_tune: bool,
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
        // Independent INT8 tactic: never imported from an FP16 plan.
        let rms_fusion = match crate::tactic_plan::tactic_var("KATAGO_CUDA_INT8_RMS_FUSION")
            .ok()
            .as_deref()
        {
            None => None,
            Some("1") => Some(true),
            Some("0") => Some(false),
            Some(v) => {
                return Err(format!(
                    "invalid KATAGO_CUDA_INT8_RMS_FUSION={v}; expected 0 or 1"
                ));
            }
        };
        let gemm_tune = gemm_tune_requested()?;
        eprintln!(
            "[cuda-tactic] name=int8_gemm_tune enabled={} candidate_pool={GEMM_TUNE_CANDIDATES} default=0",
            u8::from(gemm_tune)
        );
        Ok(Self {
            rms384: rt.get_func("int8_rms_quantize384_kernel")?,
            rms512: rt.get_func("int8_rms_quantize512_kernel")?,
            rms_fusion,
            gemm_tune,
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

struct LtPreference(sys::cublasLtMatmulPreference_t);

impl Drop for LtPreference {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { sys::cublasLtMatmulPreferenceDestroy(self.0) };
        }
    }
}

/// End capture even when a candidate launch fails. Use the raw graph API so
/// that an instantiation error also releases the uninstantiated CUDA graph.
struct TuneCapture<'a> {
    stream: &'a Arc<CudaStream>,
    active: bool,
}

impl Drop for TuneCapture<'_> {
    fn drop(&mut self) {
        if self.active {
            unsafe {
                if let Ok(graph) = driver_result::stream::end_capture(self.stream.cu_stream()) {
                    if !graph.is_null() {
                        let _ = driver_result::graph::destroy(graph);
                    }
                }
            }
        }
    }
}

struct TuneGraph {
    graph: driver_sys::CUgraph,
    exec: driver_sys::CUgraphExec,
    stream: Arc<CudaStream>,
}

impl Drop for TuneGraph {
    fn drop(&mut self) {
        unsafe {
            if !self.exec.is_null() {
                let _ = driver_result::graph::exec_destroy(self.exec);
            }
            if !self.graph.is_null() {
                let _ = driver_result::graph::destroy(self.graph);
            }
        }
    }
}

impl TuneGraph {
    fn capture(
        stream: &Arc<CudaStream>,
        mut run: impl FnMut() -> Result<(), String>,
    ) -> Result<Self, String> {
        // cuBLAS must initialize each algorithm before capture, including
        // candidates that share the same descriptors as a previous graph.
        for _ in 0..4 {
            run()?;
        }
        stream.synchronize().map_err(|e| e.to_string())?;
        stream
            .begin_capture(driver_sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(|e| e.to_string())?;
        let mut capture = TuneCapture {
            stream,
            active: true,
        };
        for _ in 0..GEMM_TUNE_REPEATS {
            run()?;
        }
        let graph = unsafe { driver_result::stream::end_capture(stream.cu_stream()) };
        capture.active = false;
        let mut graph = Self {
            graph: graph.map_err(|e| e.to_string())?,
            exec: std::ptr::null_mut(),
            stream: stream.clone(),
        };
        if graph.graph.is_null() {
            return Err("INT8 tuning captured an empty graph".into());
        }
        graph.exec = unsafe {
            driver_result::graph::instantiate(graph.graph, super::cuda::graph_instantiate_flags())
        }
        .map_err(|e| e.to_string())?;
        graph.launch()?;
        stream.synchronize().map_err(|e| e.to_string())?;
        Ok(graph)
    }

    fn launch(&self) -> Result<(), String> {
        unsafe { driver_result::graph::launch(self.exec, self.stream.cu_stream()) }
            .map_err(|e| e.to_string())
    }
}

fn timing_summary(samples: &[f32]) -> Option<(f32, f32)> {
    if samples.is_empty() || samples.iter().any(|t| !t.is_finite() || *t <= 0.0) {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f32::total_cmp);
    let mid = sorted.len() / 2;
    let median = (sorted[mid] + sorted[(sorted.len() - 1) / 2]) * 0.5;
    let spread = (sorted[sorted.len() - 1] - sorted[0]) / median;
    Some((median, spread))
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
            // A count=1 request is the original production algorithm. A
            // count=16 request can return a different first algorithm.
            let production = p.heuristics(rt, 1)?;
            if production.len() != 1
                || production[0].state != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                || production[0].workspaceSize > rt.cublaslt_workspace_len()
            {
                return Err(format!(
                    "no INT8 GEMM for M={m} N={n} K={k}; no FP16 fallback"
                ));
            }
            p.algo = production[0].algo;
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

    fn heuristics(
        &self,
        rt: &CudaRuntime,
        limit: usize,
    ) -> Result<Vec<sys::cublasLtMatmulHeuristicResult_t>, String> {
        let handle = rt.cublaslt_handle().ok_or("INT8 requires cuBLASLt")?;
        let mut pref = LtPreference(std::ptr::null_mut());
        let mut results = vec![unsafe { std::mem::zeroed() }; limit];
        let mut count = 0;
        let workspace_len = rt.cublaslt_workspace_len();
        unsafe {
            check(
                sys::cublasLtMatmulPreferenceCreate(&mut pref.0),
                "create preference",
            )?;
            check(
                sys::cublasLtMatmulPreferenceSetAttribute(
                    pref.0,
                    sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    &workspace_len as *const _ as *const _,
                    std::mem::size_of_val(&workspace_len),
                ),
                "set workspace",
            )?;
            check(
                sys::cublasLtMatmulAlgoGetHeuristic(
                    handle,
                    self.desc,
                    self.a,
                    self.b,
                    self.c,
                    self.c,
                    pref.0,
                    limit as i32,
                    results.as_mut_ptr(),
                    &mut count,
                ),
                "select integer GEMM",
            )?;
        }
        if count < 0 || count as usize > limit {
            return Err("INT8 heuristic returned an invalid candidate count".into());
        }
        results.truncate(count as usize);
        Ok(results)
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        rt: &CudaRuntime,
        stream: &Arc<CudaStream>,
        weight: &CudaSlice<i8>,
        input: &CudaSlice<i8>,
        output: &mut CudaSlice<i32>,
        algo: &sys::cublasLtMatmulAlgo_t,
        workspace: u64,
        workspace_len: usize,
    ) -> Result<(), String> {
        if workspace == 0 {
            return Err("INT8 cuBLASLt workspace allocation failed".into());
        }
        unsafe {
            let (a, _a) = weight.device_ptr(stream);
            let (b, _b) = input.device_ptr(stream);
            let (c, _c) = output.device_ptr_mut(stream);
            let alpha = 1i32;
            let beta = 0i32;
            check(
                sys::cublasLtMatmul(
                    rt.cublaslt_handle().ok_or("INT8 requires cuBLASLt")?,
                    self.desc,
                    &alpha as *const _ as *const _,
                    a as *const _,
                    self.a,
                    b as *const _,
                    self.b,
                    &beta as *const _ as *const _,
                    c as *const _,
                    self.c,
                    c as *mut _,
                    self.c,
                    algo,
                    workspace as *mut _,
                    workspace_len,
                    stream.cu_stream() as *mut _,
                ),
                "integer GEMM",
            )
        }
    }

    fn tune(
        &mut self,
        rt: &CudaRuntime,
        source_stream: &Arc<CudaStream>,
        weight: &Int8Weight,
        input: &CudaSlice<i8>,
        rows: usize,
    ) -> Result<(), String> {
        let started = Instant::now();
        let (m, n, k) = (rows, weight.n, weight.kp);
        // cudarc event tracking is disabled by CudaRuntime. Finish the actual
        // quantizer before copying its input into a private tuning stream.
        // No inference output, scale, input or cached inference graph is used
        // as tuning scratch; all temporary device buffers are stream-local.
        source_stream.synchronize().map_err(|e| e.to_string())?;
        let stream = rt.device.new_stream().map_err(|e| e.to_string())?;
        let weights = stream.clone_dtod(&weight.data).map_err(|e| e.to_string())?;
        let inputs = stream
            .clone_dtod(&input.slice(0..m * k))
            .map_err(|e| e.to_string())?;
        let mut output = stream
            .alloc_zeros::<i32>(m * n)
            .map_err(|e| e.to_string())?;
        let workspace_len = rt.cublaslt_workspace_len();
        let workspace = stream
            .alloc_zeros::<u8>(workspace_len)
            .map_err(|e| e.to_string())?;
        let (workspace_ptr, _workspace_guard) = workspace.device_ptr(&stream);
        let production = self.algo;
        self.run(
            rt,
            &stream,
            &weights,
            &inputs,
            &mut output,
            &production,
            workspace_ptr,
            workspace_len,
        )?;
        let reference = stream.clone_dtoh(&output).map_err(|e| e.to_string())?;
        // A partial write must not accidentally pass because the preceding
        // candidate left the same answer in the output buffer.
        let poison: Vec<i32> = reference.iter().map(|v| !v).collect();
        let baseline_graph = TuneGraph::capture(&stream, || {
            self.run(
                rt,
                &stream,
                &weights,
                &inputs,
                &mut output,
                &production,
                workspace_ptr,
                workspace_len,
            )
        })?;
        stream
            .memcpy_htod(&poison, &mut output)
            .map_err(|e| e.to_string())?;
        baseline_graph.launch()?;
        if stream.clone_dtoh(&output).map_err(|e| e.to_string())? != reference {
            return Err(format!(
                "INT8 production graph validation failed M={m} N={n} K={k}"
            ));
        }
        let flags = Some(driver_sys::CUevent_flags::CU_EVENT_DEFAULT);
        let start = rt.device.new_event(flags).map_err(|e| e.to_string())?;
        let end = rt.device.new_event(flags).map_err(|e| e.to_string())?;
        let measure = |graph: &TuneGraph| -> Result<f32, String> {
            start.record(&stream).map_err(|e| e.to_string())?;
            graph.launch()?;
            end.record(&stream).map_err(|e| e.to_string())?;
            end.synchronize().map_err(|e| e.to_string())?;
            Ok(start.elapsed_ms(&end).map_err(|e| e.to_string())? * 1000.0
                / GEMM_TUNE_REPEATS as f32)
        };
        let mut initial_samples = Vec::new();
        for _ in 0..3 {
            initial_samples.push(measure(&baseline_graph)?);
        }
        let (mut baseline_us, _) = timing_summary(&initial_samples)
            .ok_or("INT8 tuning received invalid CUDA event timings")?;
        let mut chosen_us = baseline_us;
        let mut best_ratio = 1.0f32;
        let mut chosen_rank = None;
        let mut verified = 0;
        let candidates = self.heuristics(rt, GEMM_TUNE_CANDIDATES)?;
        for (rank, candidate) in candidates.iter().enumerate() {
            if candidate.state != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                || candidate.workspaceSize > workspace_len
            {
                eprintln!(
                    "[cuda-int8-gemm-candidate] M={m} N={n} K={k} rank={rank} valid=0 reason=heuristic_state_or_workspace"
                );
                continue;
            }
            stream
                .memcpy_htod(&poison, &mut output)
                .map_err(|e| e.to_string())?;
            if let Err(error) = self.run(
                rt,
                &stream,
                &weights,
                &inputs,
                &mut output,
                &candidate.algo,
                workspace_ptr,
                workspace_len,
            ) {
                stream.synchronize().map_err(|e| e.to_string())?;
                eprintln!(
                    "[cuda-int8-gemm-candidate] M={m} N={n} K={k} rank={rank} valid=0 reason=launch error={error:?}"
                );
                continue;
            }
            if stream.clone_dtoh(&output).map_err(|e| e.to_string())? != reference {
                eprintln!(
                    "[cuda-int8-gemm-candidate] M={m} N={n} K={k} rank={rank} valid=0 reason=int32_mismatch"
                );
                continue;
            }
            let graph = match TuneGraph::capture(&stream, || {
                self.run(
                    rt,
                    &stream,
                    &weights,
                    &inputs,
                    &mut output,
                    &candidate.algo,
                    workspace_ptr,
                    workspace_len,
                )
            }) {
                Ok(graph) => graph,
                Err(error) => {
                    stream.synchronize().map_err(|e| e.to_string())?;
                    eprintln!(
                        "[cuda-int8-gemm-candidate] M={m} N={n} K={k} rank={rank} valid=0 reason=graph error={error:?}"
                    );
                    continue;
                }
            };
            stream
                .memcpy_htod(&poison, &mut output)
                .map_err(|e| e.to_string())?;
            graph.launch()?;
            if stream.clone_dtoh(&output).map_err(|e| e.to_string())? != reference {
                eprintln!(
                    "[cuda-int8-gemm-candidate] M={m} N={n} K={k} rank={rank} valid=0 reason=graph_int32_mismatch"
                );
                continue;
            }
            verified += 1;
            let mut base_samples = Vec::new();
            let mut candidate_samples = Vec::new();
            let mut every_round_faster = true;
            // Pair each candidate with the separately queried production
            // algorithm, reducing clock drift/temperature ordering bias.
            for _ in 0..3 {
                let a1 = measure(&baseline_graph)?;
                let b1 = measure(&graph)?;
                let b2 = measure(&graph)?;
                let a2 = measure(&baseline_graph)?;
                base_samples.extend([a1, a2]);
                candidate_samples.extend([b1, b2]);
                every_round_faster &= b1 + b2 < a1 + a2;
            }
            let summaries = timing_summary(&base_samples).zip(timing_summary(&candidate_samples));
            let Some(((base, base_spread), (candidate_us, candidate_spread))) = summaries else {
                eprintln!(
                    "[cuda-int8-gemm-candidate] M={m} N={n} K={k} rank={rank} valid=1 accepted=0 reason=invalid_timing"
                );
                continue;
            };
            let ratio = candidate_us / base;
            let accepted = every_round_faster
                && base_spread <= 0.05
                && candidate_spread <= 0.05
                && ratio <= 0.97;
            eprintln!(
                "[cuda-int8-gemm-candidate] M={m} N={n} K={k} rank={rank} valid=1 int32=bitwise graph=bitwise accepted={} baseline_us={base:.4} candidate_us={candidate_us:.4} baseline_spread={base_spread:.5} candidate_spread={candidate_spread:.5} every_round_faster={every_round_faster} baseline_samples_us={base_samples:?} candidate_samples_us={candidate_samples:?}",
                u8::from(accepted)
            );
            if accepted && ratio < best_ratio {
                best_ratio = ratio;
                chosen_rank = Some(rank);
                baseline_us = base;
                chosen_us = candidate_us;
            }
        }
        // Commit only after every candidate has completed. integer_gemm will
        // submit the selected algorithm once to the real output afterwards.
        if let Some(rank) = chosen_rank {
            self.algo = candidates[rank].algo;
        }
        let chosen =
            chosen_rank.map_or_else(|| "production".to_string(), |rank| format!("pool:{rank}"));
        eprintln!(
            "[cuda-int8-gemm-tune] M={m} N={n} K={k} chosen={chosen} baseline_us={baseline_us:.4} chosen_us={chosen_us:.4} verified={verified} returned={} elapsed_ms={:.3} repeats={GEMM_TUNE_REPEATS} rounds=3 int32=bitwise cache=workspace_shape",
            candidates.len(),
            started.elapsed().as_secs_f64() * 1000.0
        );
        Ok(())
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
            if super::cuda_exec::capturing()
                || stream.capture_status().map_err(|e| e.to_string())?
                    != driver_sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
            {
                return Err("INT8 GEMM must be warmed before graph capture".into());
            }
            let mut plan = LtPlan::new(rt, rows, weight.n, weight.kp)?;
            if self.kernels.gemm_tune {
                plan.tune(rt, stream, weight, &self.quantized, rows)?;
            }
            self.plans.insert(key, plan);
        }
        let plan = &self.plans[&key];
        let (workspace, _) = rt.cublaslt_workspace_ptr(stream);
        plan.run(
            rt,
            stream,
            &weight.data,
            &self.quantized,
            &mut self.dots,
            &plan.algo,
            workspace,
            rt.cublaslt_workspace_len(),
        )
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
        self.validate_ffn(stream, dual, down, output, rows)?;
        self.multiply(rt, stream, input, dual.k, dual, rows)?;
        self.finish_ffn(rt, stream, dual, down, output, rows)
    }

    pub fn rms_fusion_enabled(&self, width: usize, min_ffn_width: usize) -> bool {
        // ABBA passed for all-FFN profiles. The selective B15 trial did not
        // reach 1% and slightly regressed at C32, so keep its original path.
        self.kernels.rms_fusion.unwrap_or(min_ffn_width == 0) && matches!(width, 384 | 512)
    }

    /// RMSNorm and input quantization preserve the FP16 boundary in registers.
    pub fn ffn_rms(
        &mut self,
        rt: &CudaRuntime,
        stream: &Arc<CudaStream>,
        gamma: &CudaSlice<f32>,
        eps: f32,
        dual: &Int8Weight,
        down: &Int8Weight,
        residual: &mut CudaSlice<f32>,
        rows: usize,
    ) -> Result<(), String> {
        self.validate_ffn(stream, dual, down, residual, rows)?;
        if !matches!(dual.k, 384 | 512) || gamma.len() < dual.k || !eps.is_finite() || eps <= 0.0 {
            return Err("INT8 RMS shape/epsilon mismatch".into());
        }
        if !super::cuda_exec::capturing() {
            static REPORTED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
            REPORTED.get_or_init(|| eprintln!("[cuda-tactic] name=int8_rms_quantize launch=warp4 width={} rows={rows} half_boundary=preserved", dual.k));
        }
        unsafe {
            stream
                .launch_builder(if dual.k == 384 {
                    &self.kernels.rms384
                } else {
                    &self.kernels.rms512
                })
                .arg(&*residual)
                .arg(gamma)
                .arg(&mut self.quantized)
                .arg(&mut self.scales)
                .arg(&eps)
                .arg(&(rows as i32))
                .launch(LaunchConfig {
                    grid_dim: (rows.div_ceil(4) as u32, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        self.integer_gemm(rt, stream, dual, rows)?;
        self.finish_ffn(rt, stream, dual, down, residual, rows)
    }

    fn validate_ffn(
        &self,
        stream: &Arc<CudaStream>,
        dual: &Int8Weight,
        down: &Int8Weight,
        output: &CudaSlice<f32>,
        rows: usize,
    ) -> Result<(), String> {
        let count = rows
            .checked_mul(down.n)
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or("INT8 FFN size overflow")?;
        if stream.cu_stream() as usize != self.stream_id
            || rows == 0
            || rows > i32::MAX as usize
            || dual.n != 2 * down.k
            || dual.k != down.n
            || output.len() < count
            || self.quantized.len() < rows.checked_mul(down.kp).ok_or("INT8 FFN size overflow")?
            || self.quantized.len() < rows.checked_mul(dual.kp).ok_or("INT8 FFN size overflow")?
            || self.scales.len() < rows
            || self.dots.len() < rows.checked_mul(dual.n).ok_or("INT8 FFN size overflow")?
            || self.dots.len() < count
        {
            return Err("INT8 FFN shape/buffer mismatch".into());
        }
        Ok(())
    }

    fn finish_ffn(
        &mut self,
        rt: &CudaRuntime,
        stream: &Arc<CudaStream>,
        dual: &Int8Weight,
        down: &Int8Weight,
        output: &mut CudaSlice<f32>,
        rows: usize,
    ) -> Result<(), String> {
        let count = rows * down.n; // checked before submission
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
