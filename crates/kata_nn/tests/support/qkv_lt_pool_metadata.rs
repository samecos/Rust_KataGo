//! Bounded REQ8 candidate-pool diagnostic for B14 QKV TN, FP16 output.
//! Copied from frozen qkv_lt_metadata.rs; the original is unchanged.
//! Adapted from frozen r4 metadata helper; no production algorithm cache is read.
//! This does not inspect or change CudaRuntime's private algorithm cache.

use cudarc::cublaslt::sys;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use kata_nn::backends::cuda::CudaRuntime;
use serde::Serialize;
use serde_json::{Value, json};
use std::ffi::c_void;
use std::sync::{Arc, OnceLock};

#[path = "lt_replay_precision.rs"]
mod precision;
use precision::{
    ACCUMULATOR_32F, ACCUMULATOR_TYPE_MASK, INPUT_16F, INPUT_TYPE_MASK, flags_support_fp16_fp32,
};

const WORKSPACE_BYTES: usize = 32 * 1024 * 1024;
const REQUESTED: usize = 8;

fn status(value: sys::cublasStatus_t, operation: &str) -> Result<(), String> {
    if value == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(format!("{operation}: {value:?}"))
    }
}

fn status_json(value: sys::cublasStatus_t) -> Value {
    json!({"name": format!("{value:?}"), "code": value as u32})
}

fn opaque(algo: &sys::cublasLtMatmulAlgo_t) -> Value {
    let bytes: Vec<u8> = algo
        .data
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    json!({
        "bytes": bytes.len(),
        "u64_hex": algo.data.iter().map(|v| format!("0x{v:016x}")).collect::<Vec<_>>(),
        "little_endian_hex": hex::encode(bytes),
    })
}

fn heuristic_json(value: &sys::cublasLtMatmulHeuristicResult_t) -> Value {
    json!({
        "state": status_json(value.state),
        "workspace_bytes": value.workspaceSize,
        "waves_count": value.wavesCount.is_finite().then_some(value.wavesCount),
        "waves_count_f32_bits": format!("0x{:08x}", value.wavesCount.to_bits()),
    })
}

/// Own every descriptor immediately, including partially completed creation.
struct Descriptors {
    operation: sys::cublasLtMatmulDesc_t,
    a: sys::cublasLtMatrixLayout_t,
    b: sys::cublasLtMatrixLayout_t,
    cd: sys::cublasLtMatrixLayout_t,
    preference: sys::cublasLtMatmulPreference_t,
}

impl Drop for Descriptors {
    fn drop(&mut self) {
        unsafe {
            if !self.preference.is_null() {
                sys::cublasLtMatmulPreferenceDestroy(self.preference);
            }
            if !self.cd.is_null() {
                sys::cublasLtMatrixLayoutDestroy(self.cd);
            }
            if !self.b.is_null() {
                sys::cublasLtMatrixLayoutDestroy(self.b);
            }
            if !self.a.is_null() {
                sys::cublasLtMatrixLayoutDestroy(self.a);
            }
            if !self.operation.is_null() {
                sys::cublasLtMatmulDescDestroy(self.operation);
            }
        }
    }
}

impl Descriptors {
    fn new(m: usize, n: usize, k: usize) -> Result<Self, String> {
        let mut result = Self {
            operation: std::ptr::null_mut(),
            a: std::ptr::null_mut(),
            b: std::ptr::null_mut(),
            cd: std::ptr::null_mut(),
            preference: std::ptr::null_mut(),
        };
        unsafe {
            status(
                sys::cublasLtMatmulDescCreate(
                    &mut result.operation,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    sys::cudaDataType_t::CUDA_R_32F,
                ),
                "operation create",
            )?;
            let transa = 1u32;
            let transb = 0u32;
            for (attribute, value) in [
                (
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                    &transa,
                ),
                (
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                    &transb,
                ),
            ] {
                status(
                    sys::cublasLtMatmulDescSetAttribute(
                        result.operation,
                        attribute,
                        (value as *const u32).cast(),
                        std::mem::size_of::<u32>(),
                    ),
                    "transpose attribute",
                )?;
            }
            status(
                sys::cublasLtMatrixLayoutCreate(
                    &mut result.a,
                    sys::cudaDataType_t::CUDA_R_16F,
                    k as u64,
                    n as u64,
                    k as i64,
                ),
                "weight A layout",
            )?;
            status(
                sys::cublasLtMatrixLayoutCreate(
                    &mut result.b,
                    sys::cudaDataType_t::CUDA_R_16F,
                    k as u64,
                    m as u64,
                    k as i64,
                ),
                "input B layout",
            )?;
            status(
                sys::cublasLtMatrixLayoutCreate(
                    &mut result.cd,
                    sys::cudaDataType_t::CUDA_R_16F,
                    n as u64,
                    m as u64,
                    n as i64,
                ),
                "shared C/D layout",
            )?;
            status(
                sys::cublasLtMatmulPreferenceCreate(&mut result.preference),
                "preference create",
            )?;
            // Exactly the production preference: no alignment or numerical
            // preference is added before the REQ8 non-stream heuristic query.
            status(sys::cublasLtMatmulPreferenceSetAttribute(
                result.preference,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&WORKSPACE_BYTES as *const usize).cast(),
                std::mem::size_of::<usize>(),
            ), "workspace preference")?;
        }
        Ok(result)
    }

    fn precision_readback(&self) -> Result<(bool, Value), String> {
        use sys::cublasLtMatmulDescAttributes_t as Op;
        // CUDA v13.3 cublasLt.h declares compute/scale as int32_t, and
        // MATRIX_LAYOUT_TYPE as uint32_t. typed() checks status and byte count.
        let (compute, compute_report) =
            typed::<i32>("descriptor compute type", |p, size, written| unsafe {
                sys::cublasLtMatmulDescGetAttribute(
                    self.operation,
                    Op::CUBLASLT_MATMUL_DESC_COMPUTE_TYPE,
                    p,
                    size,
                    written,
                )
            })?;
        let (scale, scale_report) =
            typed::<i32>("descriptor scale type", |p, size, written| unsafe {
                sys::cublasLtMatmulDescGetAttribute(
                    self.operation,
                    Op::CUBLASLT_MATMUL_DESC_SCALE_TYPE,
                    p,
                    size,
                    written,
                )
            })?;
        let mut matches = compute == sys::cublasComputeType_t::CUBLAS_COMPUTE_32F as i32
            && scale == sys::cudaDataType_t::CUDA_R_32F as i32;
        let mut matrices = serde_json::Map::new();
        for (name, layout, expected) in [
            ("A", self.a, sys::cudaDataType_t::CUDA_R_16F),
            ("B", self.b, sys::cudaDataType_t::CUDA_R_16F),
            ("C", self.cd, sys::cudaDataType_t::CUDA_R_16F),
            ("D", self.cd, sys::cudaDataType_t::CUDA_R_16F),
        ] {
            let (actual, detail) = typed::<u32>(
                &format!("{name} descriptor matrix type"),
                |p, size, written| unsafe {
                    sys::cublasLtMatrixLayoutGetAttribute(
                        layout,
                        sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_TYPE,
                        p,
                        size,
                        written,
                    )
                },
            )?;
            matches &= actual == expected as u32;
            matrices.insert(
                name.into(),
                json!({"dtype": detail, "expected": expected as u32,
                "expected_name": format!("{expected:?}"), "matches": actual == expected as u32}),
            );
        }
        Ok((
            matches,
            json!({"complete": true, "matches_required_precision": matches,
            "compute": compute_report, "scale": scale_report, "matrices": matrices,
            "C_D_same_descriptor": true, "source": "actual cublasLt GetAttribute calls"}),
        ))
    }
}

fn typed<T: Default + Serialize>(
    label: &str,
    get: impl FnOnce(*mut c_void, usize, *mut usize) -> sys::cublasStatus_t,
) -> Result<(T, Value), String> {
    let mut value = T::default();
    let mut written = 0usize;
    let size = std::mem::size_of::<T>();
    let rc = get((&mut value as *mut T).cast(), size, &mut written);
    status(rc, label)?;
    if written != size {
        return Err(format!("{label}: sizeWritten={written}, expected {size}"));
    }
    let report = json!({"type": std::any::type_name::<T>(), "value": &value,
        "size_written": written, "status": status_json(rc)});
    Ok((value, report))
}

fn config<T: Default + Serialize>(
    algo: &sys::cublasLtMatmulAlgo_t,
    attr: sys::cublasLtMatmulAlgoConfigAttributes_t,
) -> Result<Value, String> {
    let (_, report) = typed::<T>(&format!("config {attr:?}"), |p, n, written| unsafe {
        sys::cublasLtMatmulAlgoConfigGetAttribute(algo, attr, p, n, written)
    })?;
    Ok(report)
}

fn cap<T: Default + Serialize>(
    algo: &sys::cublasLtMatmulAlgo_t,
    attr: sys::cublasLtMatmulAlgoCapAttributes_t,
) -> Result<(T, Value), String> {
    typed::<T>(&format!("capability {attr:?}"), |p, n, written| unsafe {
        sys::cublasLtMatmulAlgoCapGetAttribute(algo, attr, p, n, written)
    })
}

type CheckForStream = unsafe extern "C" fn(
    sys::cublasLtHandle_t,
    sys::cublasLtMatmulDesc_t,
    sys::cublasLtMatrixLayout_t,
    sys::cublasLtMatrixLayout_t,
    sys::cublasLtMatrixLayout_t,
    sys::cublasLtMatrixLayout_t,
    *const sys::cublasLtMatmulAlgo_t,
    *mut sys::cublasLtMatmulHeuristicResult_t,
    sys::cudaStream_t,
) -> sys::cublasStatus_t;

fn check_function() -> Result<CheckForStream, String> {
    static FUNCTION: OnceLock<Result<CheckForStream, String>> = OnceLock::new();
    FUNCTION
        .get_or_init(|| unsafe {
            // culib is held in cudarc's static OnceLock; copying this symbol never
            // outlives its library. Absence is an error, not a non-stream fallback.
            sys::culib()
                .get::<CheckForStream>(b"cublasLtMatmulAlgoCheckForStream\0")
                .map(|symbol| *symbol)
                .map_err(|error| format!("CheckForStream symbol unavailable: {error}"))
        })
        .clone()
}

/// A real, unmodified source-shape heuristic object. Metadata is observational;
/// execution gates use the private source shape, flags, alignment and bytes.
pub struct Selection {
    algo: sys::cublasLtMatmulAlgo_t,
    original: [u64; 8],
    source_shape: [usize; 3],
    stream: Arc<CudaStream>,
    handle: usize,
    numerical_flags: u64,
    source_state: u32,
    source_workspace_bytes: usize,
    splitk_num: i32,
    reduction_scheme: u32,
    min_align: [u32; 4],
    pub metadata: Value,
}

fn shape_json(m: usize, n: usize, k: usize) -> Value {
    json!({"m": m, "n": n, "k": k})
}

fn descriptor_json(m: usize, n: usize, k: usize) -> Value {
    json!({"compute": "CUBLAS_COMPUTE_32F", "scale": "CUDA_R_32F",
        "layout": "column-major descriptors over production row-major TN storage",
        "A": {"source": "weights", "dtype": "f16", "rows": k, "cols": n, "ld": k, "op": "T"},
        "B": {"source": "inputs", "dtype": "f16", "rows": k, "cols": m, "ld": k, "op": "N"},
        "C_D": {"same_pointer": true, "dtype": "f16", "rows": n, "cols": m, "ld": n},
        "alpha": 1.0, "beta": 0.0})
}

/// One entry from the SAME request8 call. Metadata failure prevents execution.
pub struct Candidate {
    pub original_index: usize,
    pub selection: Option<Selection>,
    pub metadata: Value,
}
pub struct Pool {
    pub metadata: Value,
    pub candidates: Vec<Candidate>,
}

/// Preserve all original objects/order; do not alter heuristic preferences.
pub fn select_pool(rt: &CudaRuntime, stream: &Arc<CudaStream>, m: usize, n: usize, k: usize) -> Result<Pool, String> {
    if [m, n, k] != [5054, 1152, 384] {
        return Err("registered B14 QKV requires M5054 N1152 K384".into());
    }
    use sys::cublasLtMatmulAlgoCapAttributes_t as Cap;
    use sys::cublasLtMatmulAlgoConfigAttributes_t as Config;
    if [m, n, k].iter().any(|&v| v == 0 || v > i64::MAX as usize) {
        return Err("positive dimensions fitting signed leading dimensions required".into());
    }
    if !Arc::ptr_eq(stream.context(), &rt.device) {
        return Err("runtime and selection stream must share the same CUDA context".into());
    }
    stream
        .context()
        .bind_to_thread()
        .map_err(|e| format!("bind context: {e}"))?;
    let handle = rt.cublaslt_handle().ok_or("cuBLASLt is unavailable")?;
    if rt.cublaslt_workspace_len() != WORKSPACE_BYTES {
        return Err("production replay requires exactly 32 MiB workspace".into());
    }
    let descriptors = Descriptors::new(m, n, k)?;
    let mut heuristics: Vec<sys::cublasLtMatmulHeuristicResult_t> =
        vec![unsafe { std::mem::zeroed() }; REQUESTED];
    let mut count = 0i32;
    let query_status = unsafe {
        sys::cublasLtMatmulAlgoGetHeuristic(
            handle,
            descriptors.operation,
            descriptors.a,
            descriptors.b,
            descriptors.cd,
            descriptors.cd,
            descriptors.preference,
            REQUESTED as i32,
            heuristics.as_mut_ptr(),
            &mut count,
        )
    };
    status(query_status, "REQ8 heuristic query")?;
    if count != REQUESTED as i32 {
        return Err(format!("bounded pool requires exactly eight candidates; returned: {count}"));
    }

    let unfiltered: Vec<Value> = heuristics.iter().enumerate().map(|(index,h)| json!({
        "original_index":index,"result":heuristic_json(h),"opaque":opaque(&h.algo),
        "workspace_eligible":h.workspaceSize<=WORKSPACE_BYTES})).collect();
    let query_record = json!({"api":"cublasLtMatmulAlgoGetHeuristic","query_calls":1,
        "status":status_json(query_status),"requested":REQUESTED,"returned":count,
        "unfiltered":unfiltered,"selection_policy":"all eight original indices; no pool filtering or re-query",
        "preference_max_workspace_bytes":WORKSPACE_BYTES});
    let mut candidates = Vec::new();
    for (index, heuristic) in heuristics.iter().enumerate() {
        let described = (|| -> Result<Selection,String> {
            let algo = heuristic.algo;
            let before = algo.data;
    let attributes = json!({
        "id": config::<i32>(&algo, Config::CUBLASLT_ALGO_CONFIG_ID)?,
        "tile_id": config::<u32>(&algo, Config::CUBLASLT_ALGO_CONFIG_TILE_ID)?,
        "splitk_num": config::<i32>(&algo, Config::CUBLASLT_ALGO_CONFIG_SPLITK_NUM)?,
        "reduction_scheme": config::<u32>(&algo, Config::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)?,
        "cta_swizzling": config::<u32>(&algo, Config::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING)?,
        "custom_option": config::<u32>(&algo, Config::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION)?,
        "stages_id": config::<u32>(&algo, Config::CUBLASLT_ALGO_CONFIG_STAGES_ID)?,
        "inner_shape_id": config::<u16>(&algo, Config::CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID)?,
        "cluster_shape_id": config::<u16>(&algo, Config::CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID)?,
    });
    let (numerical_flags, numerical) =
        cap::<u64>(&algo, Cap::CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS)?;
    let (min_a, cap_a) = cap::<u32>(&algo, Cap::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES)?;
    let (min_b, cap_b) = cap::<u32>(&algo, Cap::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES)?;
    let (min_c, cap_c) = cap::<u32>(&algo, Cap::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES)?;
    let (min_d, cap_d) = cap::<u32>(&algo, Cap::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES)?;
    if algo.data != before {
        return Err("attribute/capability queries changed selected opaque bytes".into());
    }
    let after_queries = opaque(&algo);

            let splitk_num = attributes["splitk_num"]["value"].as_i64().ok_or("splitK attribute missing")? as i32;
            let reduction_scheme = attributes["reduction_scheme"]["value"].as_u64().ok_or("reduction attribute missing")? as u32;
            let metadata = json!({"status":"SELECTED_NOT_EXECUTED","executed":false,
                "original_index":index,"shape":shape_json(m,n,k),"selection_shape":shape_json(m,n,k),
                "execution_shape":null,"descriptor":descriptor_json(m,n,k),
                "stream":format!("0x{:016x}",stream.cu_stream() as usize),
                "cublaslt_version":unsafe{sys::cublasLtGetVersion()},
                "source_heuristic":heuristic_json(heuristic),
                "selected":{"opaque":opaque(&algo),"opaque_before":opaque(&heuristic.algo),
                    "opaque_after_queries":after_queries,"nine_config_attributes":attributes,
                    "numerical_impl_flags":numerical,"minimum_alignment":{"A":cap_a,"B":cap_b,"C":cap_c,"D":cap_d}},
                "arithmetic_scope":{"allowed_splitk":[0,1],"required_reduction_scheme":0,
                    "reason":"exclude half intermediate split-K/reduction; FP32 capability alone is insufficient"}});
            Ok(Selection{algo,original:before,source_shape:[m,n,k],stream:Arc::clone(stream),
                handle:handle as usize,numerical_flags,min_align:[min_a,min_b,min_c,min_d],
                source_state:heuristic.state as u32,source_workspace_bytes:heuristic.workspaceSize,
                splitk_num,reduction_scheme,metadata})
        })();
        let (selection,metadata) = match described {
            Ok(selection) => { let metadata=selection.metadata.clone(); (Some(selection),metadata) },
            Err(error) => (None,json!({"status":"METADATA_FAILED_NO_EXECUTION","executed":false,
                "original_index":index,"error":error,"source_heuristic":heuristic_json(heuristic),
                "opaque":opaque(&heuristic.algo)})),
        };
        candidates.push(Candidate{original_index:index,selection,metadata});
    }
    Ok(Pool{metadata:query_record,candidates})
}

fn not_supported(mut report: Value, reason: impl Into<String>) -> Value {
    report["status"] = json!("NOT_SUPPORTED");
    report["executed"] = json!(false);
    report["reason"] = json!(reason.into());
    report["execution"]["success"] = json!(false);
    report["execution"]["calls"] = json!(0);
    report
}

/// Execute the exact source object at the registered fixed QKV shape. An unsupported Check, dtype,
/// workspace or alignment returns structured NOT_SUPPORTED with zero launches.
#[allow(clippy::too_many_arguments)]
pub fn execute_selected(
    rt: &CudaRuntime,
    stream: &Arc<CudaStream>,
    inputs: &CudaSlice<u16>,
    weights: &CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    m: usize,
    n: usize,
    k: usize,
    selection: &Selection,
) -> Result<Value, String> {
    let mut report = selection.metadata.clone();
    report["status"] = json!("NOT_SUPPORTED");
    report["executed"] = json!(false);
    report["reason"] = Value::Null;
    report["shape"] = shape_json(m, n, k);
    report["selection_shape"] = shape_json(
        selection.source_shape[0],
        selection.source_shape[1],
        selection.source_shape[2],
    );
    report["execution_shape"] = shape_json(m, n, k);
    report["descriptor"] = descriptor_json(m, n, k);
    report["execution_status"] = Value::Null;
    report["execution"] = json!({"success": false, "calls": 0, "stream_synchronized": false,
        "algo_source": "original source-shape heuristic object, not CheckForStream result.algo"});
    report["check_for_stream"] =
        json!({"called": false, "status": null, "state": null, "result_algo_ignored": true});
    if [m, n, k] != [5054, 1152, 384] || [m, n, k] != selection.source_shape {
        return Ok(not_supported(
            report,
            "registered QKV target and selection must both be M5054 N1152 K384",
        ));
    }
    report["source_execution_guard"] = json!({"source_state":selection.source_state,
        "source_workspace_bytes":selection.source_workspace_bytes,"splitk_num":selection.splitk_num,
        "reduction_scheme":selection.reduction_scheme,"required_reduction_scheme":0,"allowed_splitk":[0,1]});
    if selection.source_state != 0 || selection.source_workspace_bytes > WORKSPACE_BYTES {
        return Ok(not_supported(report,"source heuristic state/workspace rejected before launch"));
    }
    if !(0..=1).contains(&selection.splitk_num) || selection.reduction_scheme != 0 {
        return Ok(not_supported(report,"split-K/reduction excluded: only splitk 0/1 and reduction 0 are admitted"));
    }
    let input_elements = m.checked_mul(k).ok_or("input element overflow")?;
    let weight_elements = n.checked_mul(k).ok_or("weight element overflow")?;
    let output_elements = m.checked_mul(n).ok_or("output element overflow")?;
    let input_stride = k.checked_mul(2).ok_or("half stride overflow")?;
    let output_stride = n.checked_mul(2).ok_or("half output stride overflow")?;
    if inputs.len() < input_elements
        || weights.len() < weight_elements
        || out.len() < output_elements
    {
        return Ok(not_supported(
            report,
            "target matrices exceed supplied allocations",
        ));
    }
    if !Arc::ptr_eq(stream.context(), &rt.device)
        || !Arc::ptr_eq(selection.stream.context(), &rt.device)
        || stream.cu_stream() != selection.stream.cu_stream()
        || !Arc::ptr_eq(inputs.context(), &rt.device)
        || !Arc::ptr_eq(weights.context(), &rt.device)
        || !Arc::ptr_eq(out.context(), &rt.device)
    {
        return Ok(not_supported(
            report,
            "selection, execution and buffers must use the same runtime context and stream",
        ));
    }
    stream
        .context()
        .bind_to_thread()
        .map_err(|e| format!("bind context: {e}"))?;
    let handle = rt.cublaslt_handle().ok_or("cuBLASLt is unavailable")?;
    if handle as usize != selection.handle || rt.cublaslt_workspace_len() != WORKSPACE_BYTES {
        return Ok(not_supported(
            report,
            "same Lt handle and exactly 32 MiB workspace required",
        ));
    }
    let descriptors = match Descriptors::new(m, n, k) {
        Ok(value) => value,
        Err(error) => {
            return Ok(not_supported(
                report,
                format!("target descriptors: {error}"),
            ));
        }
    };
    let (descriptor_precision_ok, descriptor_readback) = match descriptors.precision_readback() {
        Ok(value) => value,
        Err(error) => {
            report["descriptor_readback"] = json!({"complete": false, "error": error});
            return Ok(not_supported(report, "target descriptor readback failed"));
        }
    };
    report["descriptor_readback"] = descriptor_readback;
    let algo = &selection.algo;
    let before = selection.original;
    if algo.data != before {
        return Err("source selection opaque changed before target check".into());
    }
    report["selected"]["opaque_before_target_check"] = opaque(algo);
    let [min_a, min_b, min_c, min_d] = selection.min_align;
    let fp32 = selection.numerical_flags & ACCUMULATOR_TYPE_MASK == ACCUMULATOR_32F;
    let half_input = selection.numerical_flags & INPUT_16F != 0;
    let flags_ok = flags_support_fp16_fp32(selection.numerical_flags);
    report["precision_guard"] = json!({"flags": selection.numerical_flags,
        "accumulator_32f_mask": ACCUMULATOR_32F, "accumulator_type_mask": ACCUMULATOR_TYPE_MASK,
        "input_16f_mask": INPUT_16F, "input_type_mask": INPUT_TYPE_MASK,
        "source": "CUDA v13.3 cublasLt.h:995-1009", "accumulator_is_exactly_fp32": fp32,
        "input_supports_fp16": half_input, "flags_support_fp16_fp32": flags_ok,
        "descriptor_precision_readback_matches": descriptor_precision_ok,
        "input_rule": "INPUT_16F bit is present; other input capability bits are allowed",
        "descriptor_compute": "CUBLAS_COMPUTE_32F",
        "descriptor_scale": "CUDA_R_32F", "descriptor_output": "CUDA_R_16F"});
    let (input_ptr, _input_guard) = inputs.device_ptr(stream);
    let (weight_ptr, _weight_guard) = weights.device_ptr(stream);
    let (output_ptr, _output_guard) = out.device_ptr_mut(stream);
    let (workspace_ptr, _) = rt.cublaslt_workspace_ptr(stream);
    let mut alignment = Vec::new();
    let mut alignment_ok = true;
    for (name, pointer, stride, minimum) in [
        ("A_weights", weight_ptr, input_stride, min_a),
        ("B_inputs", input_ptr, input_stride, min_b),
        ("C_output", output_ptr, output_stride, min_c),
        ("D_output", output_ptr, output_stride, min_d),
    ] {
        let pointer_ok = minimum > 0 && pointer != 0 && pointer % u64::from(minimum) == 0;
        let stride_ok = minimum > 0 && stride % minimum as usize == 0;
        alignment.push(
            json!({"matrix": name, "pointer": format!("0x{pointer:016x}"),
            "leading_dimension_bytes": stride, "minimum_alignment_bytes": minimum,
            "pointer_aligned": pointer_ok, "stride_aligned": stride_ok}),
        );
        if !pointer_ok || !stride_ok {
            alignment_ok = false;
        }
    }
    report["alignment"] = json!(alignment);
    let workspace_ok = workspace_ptr != 0 && workspace_ptr % 256 == 0;
    report["workspace"] = json!({"pointer": format!("0x{workspace_ptr:016x}"), "bytes": WORKSPACE_BYTES,
        "same_runtime_and_stream": true, "aligned_256": workspace_ok});
    let mut checked: sys::cublasLtMatmulHeuristicResult_t = unsafe { std::mem::zeroed() };
    let check_status = unsafe {
        check_function()?(
            handle,
            descriptors.operation,
            descriptors.a,
            descriptors.b,
            descriptors.cd,
            descriptors.cd,
            algo,
            &mut checked,
            stream.cu_stream() as sys::cudaStream_t,
        )
    };
    if algo.data != before {
        return Err("CheckForStream changed selected opaque bytes".into());
    }
    let after_check = opaque(algo);
    report["selected"]["opaque_after_check"] = after_check;
    report["check_for_stream"] = json!({"called": true, "status": check_status as u32,
        "state": checked.state as u32, "api_status": status_json(check_status),
        "result": heuristic_json(&checked), "result_algo_ignored": true,
        "result_fields_valid": check_status == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS});
    let mut rejected = Vec::new();
    if check_status != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        rejected.push(format!("CheckForStream API returned {check_status:?}"));
    }
    if checked.state != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        rejected.push(format!("CheckForStream state returned {:?}", checked.state));
    }
    if checked.workspaceSize > WORKSPACE_BYTES || !checked.wavesCount.is_finite() {
        rejected.push(format!(
            "workspace/waves rejected: {}",
            heuristic_json(&checked)
        ));
    }
    if !flags_ok {
        rejected.push(
            "algorithm numerical flags lack FP16 input capability or sole FP32 accumulation".into(),
        );
    }
    if !descriptor_precision_ok {
        rejected.push(
            "actual target descriptors do not specify FP16 A/B/C/D, COMPUTE_32F and FP32 scale"
                .into(),
        );
    }
    if !alignment_ok || !workspace_ok {
        rejected.push("target pointer/stride/workspace alignment is not supported".into());
    }
    if !rejected.is_empty() {
        return Ok(not_supported(report, rejected.join("; ")));
    }
    report["selected"]["opaque_before_execution"] = opaque(algo);
    let alpha = 1.0f32;
    let beta = 0.0f32;
    // checked.algo is deliberately never read or executed: the API documents
    // it as not updated. Pass the original workspace-selected heuristic algo.
    let launch_status = unsafe {
        sys::cublasLtMatmul(
            handle,
            descriptors.operation,
            (&alpha as *const f32).cast(),
            weight_ptr as *const c_void,
            descriptors.a,
            input_ptr as *const c_void,
            descriptors.b,
            (&beta as *const f32).cast(),
            output_ptr as *const c_void,
            descriptors.cd,
            output_ptr as *mut c_void,
            descriptors.cd,
            algo,
            workspace_ptr as *mut c_void,
            WORKSPACE_BYTES,
            stream.cu_stream() as sys::cudaStream_t,
        )
    };
    // Keep descriptors, matrices, pointer guards and workspace alive through
    // completion even if the launch API reports an error.
    let synchronized = stream.synchronize();
    if algo.data != before {
        return Err("execution changed selected opaque bytes".into());
    }
    let success =
        launch_status == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS && synchronized.is_ok();
    report["status"] = json!(if success {
        "PASS_REPLAY_EXECUTED"
    } else {
        "EXECUTION_FAILED"
    });
    report["executed"] = json!(true);
    report["execution_status"] = json!(launch_status as u32);
    report["selected"]["opaque_after_execution"] = opaque(algo);
    report["selected"]["opaque_unchanged_after_queries_check_and_execution"] = json!(true);
    report["execution"] = json!({"success": success, "launch_status": status_json(launch_status),
        "stream_synchronized": synchronized.is_ok(), "execution_attempted": true,
        "synchronize_error": synchronized.as_ref().err().map(|e| e.to_string()),
        "algo_source": "original selected heuristic, not CheckForStream result.algo", "calls": 1});
    if !success {
        report["reason"] = json!(format!(
            "Matmul {launch_status:?}; synchronize {synchronized:?}"
        ));
    }
    Ok(report)
}

