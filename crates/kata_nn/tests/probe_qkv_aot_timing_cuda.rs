//! WSL-only, opt-in, standalone QKV DIAGNOSTIC timing. No model certification.
//! The numerical test is deliberately not included: only its frozen operand helper is reused.
#![cfg(all(feature = "cuda", target_os = "linux"))]

#[path = "support/qkv_lt_baseline.rs"]
mod operands;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use kata_nn::backends::cuda::{backend_build_fingerprint, f16_to_f32_bits, CublasLtWeightLayout, CudaRuntime};
use kata_nn::tactic_plan;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const BASE: &str = "target/fork-parity-20260908/qkv-fp32-aot-r1";
const ADMISSION_SHA: &str = "001316ac9a5337a6793d6216512ad1521ff269ab6e231e64f44efcc992def2f3";
const NUMERIC_SHA: &str = "2a18c4d8b3fd1f09f17712faf6da688c66a8dc35fa2560bc887c275beaeea4fb";
const ASSETS_SHA: &str = "d3469f72e305e80f56375185d1cfc11c18e5a46dd908c9439d2c0c13bc8ec1ac";
const HELPER_SHA: &str = "00e87d3e80e828506020865fd7195cf59904b0868bbc1a40f1503fa1a0a7d891";
const ACTIVE: usize = operands::M * operands::N;
const TAIL: usize = 256;
const WARMUP: usize = 80;
const ITERATIONS: usize = 1000;
const ORDER: [&str; 12] = ["A", "B", "B", "A", "A", "B", "B", "A", "A", "B", "B", "A"];

fn require(ok: bool, message: &str) -> Result<(), String> {
    if ok { Ok(()) } else { Err(message.into()) }
}
fn hash(raw: &[u8]) -> String { hex::encode(Sha256::digest(raw)) }
fn half_bytes(bits: &[u16]) -> Vec<u8> { bits.iter().flat_map(|v| v.to_le_bytes()).collect() }
fn local(root: &Path, raw: &str) -> PathBuf {
    let text = raw.replace('\\', "/");
    let bytes = text.as_bytes();
    if bytes.len() > 3 && bytes[0].is_ascii_alphabetic() && bytes[1..3] == *b":/" {
        PathBuf::from(format!("/mnt/{}/{}", (bytes[0] as char).to_ascii_lowercase(), &text[3..]))
    } else {
        let path = PathBuf::from(text);
        if path.is_absolute() { path } else { root.join(path) }
    }
}
fn record(path: &Path) -> Result<Value, String> {
    let raw = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(json!({"path":path,"sha256":hash(&raw),"bytes":raw.len()}))
}
fn bound(root: &Path, item: &Value) -> Result<Vec<u8>, String> {
    let path = local(root, item["path"].as_str().ok_or("bound path missing")?);
    let raw = fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    require(hash(&raw) == item["sha256"].as_str().ok_or("bound SHA missing")?, "bound file SHA changed")?;
    if let Some(len) = item["bytes"].as_u64() {
        require(raw.len() as u64 == len, "bound file length changed")?;
    }
    Ok(raw)
}
fn save(dir: &Path, report: &Value) -> Result<(), String> {
    fs::write(dir.join("report.json"), serde_json::to_vec_pretty(report).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}
fn export(dir: &Path, file: &str, bits: &[u16]) -> Result<Value, String> {
    let raw = half_bytes(bits);
    OpenOptions::new().write(true).create_new(true).open(dir.join(file))
        .map_err(|e| e.to_string())?.write_all(&raw).map_err(|e| e.to_string())?;
    Ok(json!({"file":file,"sha256":hash(&raw),"bytes":raw.len(),"elements":bits.len(),
        "dtype":"float16","byte_order":"little-endian","raw_bits_preserved":true}))
}

struct Admission { value: Value, assets: Value, raw: Value, expected: [Vec<u8>; 2], bridge: PathBuf }
fn admission(root: &Path) -> Result<Admission, String> {
    let entry = json!({"path":format!("{BASE}/timing-r1/admission.json"),"sha256":ADMISSION_SHA});
    let value: Value = serde_json::from_slice(&bound(root, &entry)?).map_err(|e| e.to_string())?;
    require(value["status"] == "ADMITTED_STANDALONE_DIAGNOSTIC_TIMING_NOT_CERTIFIED"
        && value["numeric_report"]["sha256"] == NUMERIC_SHA
        && value["numeric_assets"]["sha256"] == ASSETS_SHA
        && value["registered_order"] == json!(ORDER)
        && value["warmup_per_phase"] == WARMUP && value["iterations_per_phase"] == ITERATIONS,
        "frozen admission contract mismatch")?;
    let numeric: Value = serde_json::from_slice(&bound(root, &value["numeric_report"])?).map_err(|e| e.to_string())?;
    require(numeric["status"] == "PASS_COMPLETE_QKV_AOT_NUMERIC_DIAGNOSTIC_NOT_CERTIFIED"
        && numeric["pass_all"] == true && numeric["controls_pass"] == true
        && numeric["comparison"]["complete_elements"] == ACTIVE
        && numeric["comparison"]["operator_gate"]["pass_all"] == true
        && numeric["comparison"]["operator_gate"]["strict_max_abs_limit"] == 0.01,
        "complete numerical CPU PASS required before any CUDA initialization")?;
    for key in ["candidate_vs_lt", "candidate_vs_fp64", "lt_vs_fp64"] {
        let error = numeric["comparison"]["operator_gate"]["maxima"][key].as_f64().ok_or("numeric maximum missing")?;
        require(error.is_finite() && error >= 0.0 && error < 0.01, "full operator numerical gate")?;
    }
    for key in ["raw_report", "binary", "contract", "assets", "console", "fp64_report"] {
        bound(root, &numeric[key])?;
    }
    let raw: Value = serde_json::from_slice(&bound(root, &value["numeric_raw_report"])?).map_err(|e| e.to_string())?;
    require(raw["status"] == "PASS_AOT_VS_LT_PENDING_FP64_NOT_CERTIFIED"
        && raw["final_stream_synchronized"] == true
        && raw["repeated_runs_deterministic_raw_half"] == true,
        "numeric raw execution proof")?;
    let expected = [bound(root, &value["expected_outputs"]["baseline"])?,
        bound(root, &value["expected_outputs"]["candidate"])?];
    let repeated = bound(root, &value["expected_outputs"]["candidate_repeat"])?;
    require(expected.iter().all(|v| v.len() == ACTIVE * 2) && repeated == expected[1], "complete numerical raw identities")?;
    for key in ["baseline", "candidate", "candidate_repeat"] {
        require(numeric["raw_outputs"][key]["sha256"] == value["expected_outputs"][key]["sha256"], "numeric raw reference binding")?;
        bound(root, &numeric["raw_outputs"][key])?;
    }
    for key in ["operator_contract", "numeric_binary", "bridge"] { bound(root, &value[key])?; }
    let assets: Value = serde_json::from_slice(&bound(root, &value["numeric_assets"])?).map_err(|e| e.to_string())?;
    require(assets["status"] == "REGISTERED_QKV_FP32_AOT_NUMERIC_ASSETS", "asset status")?;
    for item in assets["assets"]["files"].as_array().ok_or("asset files missing")? { bound(root, item)?; }
    bound(root, &assets["assets"]["bridge"])?;
    require(assets["assets"]["bridge"]["sha256"] == value["bridge"]["sha256"], "bridge binding")?;
    require(hash(include_bytes!("support/qkv_lt_baseline.rs")) == HELPER_SHA, "frozen compiled operand support changed")?;
    let bridge = local(root, value["bridge"]["path"].as_str().ok_or("bridge path")?);
    Ok(Admission { value, assets, raw, expected, bridge })
}

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}
type Init = unsafe extern "C" fn() -> c_int;
type Launch = unsafe extern "C" fn(*const u16, *const u16, *mut u16, *mut c_void) -> c_int;
struct Bridge { _handle: *mut c_void, init: Init, launch: Launch }
// The generated ABI has no destroy operation. Keep the library/module loaded until exit.
impl Bridge {
    unsafe fn error(label: &str) -> String {
        let ptr = unsafe { dlerror() };
        if ptr.is_null() { label.to_owned() }
        else { format!("{label}: {}", unsafe { CStr::from_ptr(ptr) }.to_string_lossy()) }
    }
    fn load(path: &Path) -> Result<Self, String> {
        let name = CString::new(path.to_str().ok_or("bridge UTF8")?).map_err(|e| e.to_string())?;
        unsafe {
            let handle = dlopen(name.as_ptr(), 2); // RTLD_NOW | RTLD_LOCAL
            if handle.is_null() { return Err(Self::error("dlopen")); }
            dlerror();
            let init = dlsym(handle, b"qkv_fp32_init\0".as_ptr().cast());
            if init.is_null() { return Err(Self::error("qkv_fp32_init")); }
            dlerror();
            let launch = dlsym(handle, b"qkv_fp32_launch\0".as_ptr().cast());
            if launch.is_null() { return Err(Self::error("qkv_fp32_launch")); }
            Ok(Self { _handle: handle, init: std::mem::transmute::<*mut c_void, Init>(init),
                launch: std::mem::transmute::<*mut c_void, Launch>(launch) })
        }
    }
    fn initialize(&self, rt: &CudaRuntime, stream: &Arc<CudaStream>) -> Result<(), String> {
        require(Arc::ptr_eq(stream.context(), &rt.device), "init context mismatch")?;
        rt.device.bind_to_thread().map_err(|e| e.to_string())?;
        let rc = unsafe { (self.init)() };
        let completion = stream.synchronize().map_err(|e| e.to_string());
        require(rc == 0, &format!("qkv_fp32_init={rc}; completion={completion:?}"))?;
        completion
    }
    // No synchronization here: the caller records stop immediately after enqueue,
    // and synchronizes that stop on EVERY measured iteration. Access guards survive FFI.
    fn submit(&self, stream: &Arc<CudaStream>, device: &operands::DeviceOperands,
        output: &mut CudaSlice<u16>) -> Result<(), String> {
        let (input, _input_guard) = device.input.device_ptr(stream);
        let (weights, _weight_guard) = device.weights.device_ptr(stream);
        let (out, _output_guard) = output.device_ptr_mut(stream);
        let rc = unsafe { (self.launch)(input as *const u16, weights as *const u16,
            out as *mut u16, stream.cu_stream() as *mut c_void) };
        if rc == 0 { Ok(()) } else { Err(format!("qkv_fp32_launch={rc}")) }
    }
}

fn submit(arm: &str, rt: &CudaRuntime, stream: &Arc<CudaStream>,
    device: &operands::DeviceOperands, output: &mut CudaSlice<u16>, bridge: &Bridge) -> Result<(), String> {
    match arm {
        "A" => require(rt.cublaslt_gemm_f16out_with_layout(stream, &device.input, &device.weights,
            output, operands::M, operands::N, operands::K, CublasLtWeightLayout::Tn)?, "public Lt unavailable; fallback forbidden"),
        "B" => bridge.submit(stream, device, output),
        _ => Err("unregistered arm".into()),
    }
}

fn reset_output(stream: &Arc<CudaStream>, output: &mut CudaSlice<u16>, initial: &[u16]) -> Result<(), String> {
    stream.memcpy_htod(initial, output).map_err(|e| e.to_string())?;
    require(stream.clone_dtoh(output).map_err(|e| e.to_string())? == initial,
        "actual pre-arm NaN/guard reset mismatch")
}

fn checkpoint(dir: &Path, name: &str, stream: &Arc<CudaStream>, device: &operands::DeviceOperands,
    host: &operands::HostOperands, output: &CudaSlice<u16>, expected: &[u8]) -> Result<Value, String> {
    let actual = stream.clone_dtoh(output).map_err(|e| e.to_string())?;
    let input = stream.clone_dtoh(&device.input).map_err(|e| e.to_string())?;
    let weights = stream.clone_dtoh(&device.weights).map_err(|e| e.to_string())?;
    stream.synchronize().map_err(|e| e.to_string())?;
    let tail: Vec<u16> = (0..TAIL).map(|i| 0x5000 ^ i as u16).collect();
    let output_file = export(dir, &format!("{name}-output.f16le"), &actual[..ACTIVE])?;
    let guard_file = export(dir, &format!("{name}-guard.f16le"), &actual[ACTIVE..])?;
    let bits_equal = half_bytes(&actual[..ACTIVE]) == expected;
    let finite = actual[..ACTIVE].iter().all(|v| f16_to_f32_bits(*v).is_finite());
    let guard_ok = actual[ACTIVE..] == tail;
    let inputs_ok = input == host.input && weights == host.weights;
    let result = json!({"output":output_file,"tail_guard":guard_file,"all_active_bits_equal_numeric_pass":bits_equal,
        "all_active_finite":finite,"guard_unchanged":guard_ok,"input_unchanged":input==host.input,
        "weights_unchanged":weights==host.weights,"input_sha256":hash(&half_bytes(&input)),
        "weights_sha256":hash(&half_bytes(&weights)),"checked_active_elements":ACTIVE});
    // Preserve failed complete output and its checkpoint evidence before failing closed.
    OpenOptions::new().write(true).create_new(true).open(dir.join(format!("{name}-check.json")))
        .map_err(|e| e.to_string())?.write_all(&serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    require(bits_equal && finite && guard_ok && inputs_ok, "full raw/finite/guard/immutable-input checkpoint failed")?;
    Ok(result)
}

fn tactics() -> Result<Value, String> {
    require(tactic_plan::installed_plan_id().is_none(), "standalone timing must not install a tactic plan")?;
    let mut effective = serde_json::Map::new();
    for (key, expected) in [("KATAGO_CUDA_CUBLASLT", "1"), ("KATAGO_CUDA_CUBLASLT_RANK", "heuristic"),
        ("KATAGO_CUDA_GEMM_LAYOUT", "tn"), ("KATAGO_CUDA_RESIDUAL_ALGO", "heuristic"), ("KATAGO_CUDA_NOGRAPH", "1")] {
        let actual = tactic_plan::tactic_var(key).map_err(|e| format!("explicit {key} required: {e}"))?;
        require(actual == expected, &format!("{key} must equal {expected}"))?;
        effective.insert(key.to_owned(), json!(actual));
    }
    Ok(Value::Object(effective))
}

fn loaded_cuda_libraries() -> Result<Value, String> {
    let maps = fs::read_to_string("/proc/self/maps").map_err(|e| e.to_string())?;
    let mut files = std::collections::BTreeSet::new();
    for line in maps.lines() {
        if let Some(path) = line.split_whitespace().last() {
            if path.starts_with('/') && ["libcublasLt.so", "libcublas.so", "libcudart.so", "libcuda.so"]
                .iter().any(|name| path.contains(name)) { files.insert(PathBuf::from(path)); }
        }
    }
    require(files.iter().any(|p| p.to_string_lossy().contains("libcublasLt.so")), "actual loaded cuBLASLt path missing")?;
    files.iter().map(|p| record(p)).collect::<Result<Vec<_>, _>>().map(|v| json!(v))
}

fn execute(root: &Path, dir: &Path, model: &Path, report: &mut Value) -> Result<(), String> {
    let admitted = admission(root)?; // All CPU gates run before CudaRuntime::new.
    report["admission"] = admitted.value.clone();
    report["numeric_assets"] = admitted.assets;
    report["effective_tactics"] = tactics()?;
    let host = operands::load_operands(root, model)?;
    report["operand_provenance"] = host.provenance.clone();
    let rt = CudaRuntime::new()?;
    let stream = rt.device.new_stream().map_err(|e| e.to_string())?;
    let device = operands::upload_operands(&rt, &stream, &host)?;
    let fingerprint = serde_json::to_value(backend_build_fingerprint()).map_err(|e| e.to_string())?;
    require(fingerprint == admitted.raw["backend_build_fingerprint"], "numeric/timing compiled backend fingerprint mismatch")?;
    report["backend_build_fingerprint"] = fingerprint;
    report["stream_id"] = json!(stream.cu_stream() as usize);
    rt.device.bind_to_thread().map_err(|e| e.to_string())?;
    let bridge = Bridge::load(&admitted.bridge)?;
    bridge.initialize(&rt, &stream)?;
    let mut initial = vec![0x7e00u16; ACTIVE];
    initial.extend((0..TAIL).map(|i| 0x5000 ^ i as u16));
    let mut output = stream.clone_htod(&initial).map_err(|e| e.to_string())?;
    require(Arc::ptr_eq(device.input.context(), &rt.device) && Arc::ptr_eq(device.weights.context(), &rt.device)
        && Arc::ptr_eq(output.context(), &rt.device) && device.input.len() == operands::M*operands::K
        && device.weights.len() == operands::N*operands::K && output.len() == ACTIVE+TAIL, "device shape/context contract")?;
    let (ip, ig) = device.input.device_ptr(&stream);
    let (wp, wg) = device.weights.device_ptr(&stream);
    let (op, og) = output.device_ptr(&stream);
    require(ip % 16 == 0 && wp % 16 == 0 && op % 16 == 0, "device pointer alignment")?;
    report["device_pointers"] = json!({"input":format!("0x{ip:016x}"),"weights":format!("0x{wp:016x}"),
        "output":format!("0x{op:016x}"),"reused_for_all_phases":true});
    drop((ig, wg, og));
    require(stream.clone_dtoh(&output).map_err(|e| e.to_string())? == initial, "actual NaN/guard initialization")?;
    report["initial_output"] = export(dir, "initial-output-with-guard.f16le", &initial)?;
    // Both kernels initialize/cache outside timing, and must reproduce the complete PASS raw outputs.
    for (i, arm) in ["A", "B"].iter().enumerate() {
        reset_output(&stream, &mut output, &initial)?;
        submit(arm, &rt, &stream, &device, &mut output, &bridge)?;
        stream.synchronize().map_err(|e| e.to_string())?;
        report["preflight"][*arm] = checkpoint(dir, &format!("pre-{arm}"), &stream, &device, &host, &output, &admitted.expected[i])?;
        save(dir, report)?;
    }
    report["loaded_cuda_libraries"] = loaded_cuda_libraries()?;
    // Event allocation and elapsed-time readback are outside every measured wall interval.
    let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    let events = (0..ITERATIONS).map(|_| Ok((rt.device.new_event(flags).map_err(|e| e.to_string())?,
        rt.device.new_event(flags).map_err(|e| e.to_string())?)))
        .collect::<Result<Vec<_>, String>>()?;
    for (phase, arm) in ORDER.iter().enumerate() {
        report["phase"] = json!(phase);
        reset_output(&stream, &mut output, &initial)?;
        for _ in 0..WARMUP { submit(arm, &rt, &stream, &device, &mut output, &bridge)?; }
        stream.synchronize().map_err(|e| e.to_string())?;
        eprintln!("[qkv-aot-timing] phase={phase} arm={arm} batch=14 m=5054 n=1152 k=384 warmup=80 iterations=1000 lane=0 stop_sync=every_iteration scope=DIAGNOSTIC");
        let mut completed = 0usize;
        let started = Instant::now();
        let measured = (|| -> Result<(), String> {
            for (start, stop) in &events {
                start.record(&stream).map_err(|e| e.to_string())?;
                submit(arm, &rt, &stream, &device, &mut output, &bridge)?;
                stop.record(&stream).map_err(|e| e.to_string())?;
                stop.synchronize().map_err(|e| e.to_string())?;
                completed += 1;
            }
            Ok(())
        })();
        let wall_seconds = started.elapsed().as_secs_f64();
        let completion = stream.synchronize().map_err(|e| e.to_string());
        let samples = events[..completed].iter().map(|(a,b)| a.elapsed_ms(b).map(|v| v as f64).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, String>>()?;
        let item = json!({"phase":phase,"arm":arm,"round":phase/4,"warmup":WARMUP,"requested_iterations":ITERATIONS,
            "completed_iterations":completed,"cuda_event_ms":samples,"wall_seconds":wall_seconds,
            "wall_calls_per_second":completed as f64/wall_seconds,"stop_sync_every_iteration":true,
            "actual_nan_guard_reset_before_warmup":true,
            "measured_enqueue_error":measured.as_ref().err(),"completion_error":completion.as_ref().err()});
        report["phases"].as_array_mut().unwrap().push(item);
        save(dir, report)?;
        measured?;
        completion?;
        require(samples.len() == ITERATIONS && samples.iter().all(|v| v.is_finite() && *v > 0.0), "complete positive CUDA event samples")?;
        let index = usize::from(*arm == "B");
        let check = checkpoint(dir, &format!("phase-{phase:02}-{arm}"), &stream, &device, &host, &output, &admitted.expected[index])?;
        report["phases"][phase]["post_check"] = check;
        save(dir, report)?;
    }
    for (i, arm) in ["A", "B"].iter().enumerate() {
        reset_output(&stream, &mut output, &initial)?;
        submit(arm, &rt, &stream, &device, &mut output, &bridge)?;
        stream.synchronize().map_err(|e| e.to_string())?;
        report["postflight"][*arm] = checkpoint(dir, &format!("post-{arm}"), &stream, &device, &host, &output, &admitted.expected[i])?;
    }
    require(tactics()? == report["effective_tactics"], "tactics changed during timing")?;
    admission(root)?; // Revalidate immutable admission/outputs/bridge/assets after the last GPU operation.
    report["assets_revalidated_after_timing"] = json!(true);
    report["final_stream_synchronized"] = json!(true);
    report["status"] = json!("COMPLETE_RAW_QKV_AOT_TIMING_DIAGNOSTIC_NOT_CERTIFIED");
    report["phase"] = json!("complete");
    save(dir, report)
}

#[test]
fn probe_qkv_aot_timing_cuda() {
    match std::env::var("KATAGO_RUN_QKV_AOT_TIMING").as_deref() {
        Err(std::env::VarError::NotPresent) => { eprintln!("SKIP: set KATAGO_RUN_QKV_AOT_TIMING=1"); return; }
        Ok("1") => {}
        value => panic!("invalid explicit QKV timing opt-in {value:?}"),
    }
    let release = fs::read_to_string("/proc/sys/kernel/osrelease").unwrap();
    assert!(release.to_lowercase().contains("microsoft"), "frozen WSL platform required");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap();
    let dir = local(&root, &std::env::var("KATAGO_QKV_AOT_TIMING_DIR").expect("fresh explicit timing output directory required"));
    assert!(!dir.exists(), "timing evidence directory must be fresh");
    assert!(dir.parent().unwrap().canonicalize().unwrap().starts_with(root.join("target").canonicalize().unwrap()), "timing output must stay under workspace target");
    fs::create_dir(&dir).unwrap();
    let model = PathBuf::from(std::env::var_os("KATAGO_TEST_MODEL_DIR").expect("explicit model directory required"))
        .join("kata1-tf3-b11c768-s11001M-d5973M.bin.gz");
    let mut report = json!({"schema":1,"status":"PREPARING_NOT_COMPLETE","phase":"CPU_admission",
        "scope":"standalone real-operand QKV DIAGNOSTIC; no model/Worker throughput or certification",
        "test_executable":record(&std::env::current_exe().unwrap()).unwrap(),
        "source":{"test_sha256":hash(include_bytes!("probe_qkv_aot_timing_cuda.rs")),"operand_support_sha256":hash(include_bytes!("support/qkv_lt_baseline.rs"))},
        "admission_sha256":ADMISSION_SHA,"kernel_release":release.trim(),"lanes":1,
        "shape":{"batch":14,"m":5054,"n":1152,"k":384},"registered_order":ORDER,
        "warmup_per_phase":WARMUP,"iterations_per_phase":ITERATIONS,
        "timing":"CUDA start -> actual enqueue -> stop -> stop synchronize; each iteration",
        "wall_scope":"1000 iteration event-record/enqueue/stop-sync host loop; excludes event allocation/elapsed-query and all transfers/checks",
        "raw_events_in_original_order":true,"discard_or_replace_samples":false,
        "production_certified":false,"is_full_network_gate":false,"is_cpp_golden_gate":false,
        "environment":std::env::vars().filter(|(k,_)| k.starts_with("KATAGO_CUDA_")).collect::<std::collections::BTreeMap<_,_>>(),
        "preflight":{},"postflight":{},"phases":[]});
    save(&dir, &report).unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| execute(&root, &dir, &model, &mut report)));
    match result {
        Ok(Ok(())) => eprintln!("[qkv-aot-timing] status=COMPLETE_RAW_QKV_AOT_TIMING_DIAGNOSTIC_NOT_CERTIFIED report={}",dir.join("report.json").display()),
        Ok(Err(error)) => {
            report["status"] = json!("FAILED_QKV_AOT_TIMING_DIAGNOSTIC_NOT_CERTIFIED");
            report["failure"] = json!(error);
            save(&dir, &report).unwrap();
            panic!("QKV timing failed; partial evidence retained");
        }
        Err(panic) => {
            report["status"] = json!("FAILED_QKV_AOT_TIMING_DIAGNOSTIC_NOT_CERTIFIED");
            report["failure"] = json!(panic.downcast_ref::<String>().map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied()).unwrap_or("non-string panic"));
            save(&dir, &report).unwrap();
            std::panic::resume_unwind(panic);
        }
    }
}
