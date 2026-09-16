//! WSL-only real-operand QKV numerical harness. No timing or production change.
//! The opt-in entry point is attached only after the asset SHA is frozen.
#![cfg(all(feature = "cuda", target_os = "linux"))]

#[path = "support/qkv_lt_baseline.rs"]
mod operands;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use kata_nn::backends::cuda::{backend_build_fingerprint, f16_to_f32_bits, CudaRuntime};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const ACTIVE: usize = operands::M * operands::N;
const TAIL: usize = 256;
const ASSET_RELATIVE: &str = "target/fork-parity-20260908/qkv-fp32-aot-r1/numeric-r1/assets.json";
const BASELINE_OUTPUT_SHA: &str =
    "57be090aa964a5e855e168eb26b1d562976ae3c35664ada3b16bff24e1b0d8c6";
const BASELINE_RECEIPT_SHA: &str =
    "ee711a037caf8cff0cfddf50572a6ac4b6c45a59cd7b74dcd9a39385b2b5b16c";
const BASELINE_REPORT_SHA: &str =
    "38276f4e3aa2a36b5e77deba95240fd754ecf499a9dc6720f5dfd7f41c552313";
const BASELINE_BINARY_SHA: &str =
    "0329a1aae7ce393c953eb50617b8c3a916b57fc18a1523383649cda9dcb93686";
const BASELINE_HELPER_SHA: &str =
    "00e87d3e80e828506020865fd7195cf59904b0868bbc1a40f1503fa1a0a7d891";

fn hash(raw: &[u8]) -> String {
    hex::encode(Sha256::digest(raw))
}
fn half_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn require(ok: bool, message: &str) -> Result<(), String> {
    if ok {
        Ok(())
    } else {
        Err(message.into())
    }
}
fn record(path: &Path) -> Result<Value, String> {
    let raw = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(json!({"path":path,"sha256":hash(&raw),"bytes":raw.len()}))
}
fn bound(path: &Path, expected: &str) -> Result<Vec<u8>, String> {
    let raw = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    require(
        hash(&raw) == expected,
        &format!("SHA mismatch {}", path.display()),
    )?;
    Ok(raw)
}
fn save(dir: &Path, report: &Value) {
    fs::write(
        dir.join("report.json"),
        serde_json::to_vec_pretty(report).unwrap(),
    )
    .unwrap();
}
fn export(dir: &Path, file: &str, bits: &[u16], shape: &[usize]) -> Result<Value, String> {
    require(
        bits.len() == shape.iter().product::<usize>(),
        "raw tensor shape/length",
    )?;
    let raw = half_bytes(bits);
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(dir.join(file))
        .map_err(|e| e.to_string())?
        .write_all(&raw)
        .map_err(|e| e.to_string())?;
    Ok(
        json!({"file":file,"sha256":hash(&raw),"bytes":raw.len(),"shape":shape,
        "elements":bits.len(),"dtype":"float16","byte_order":"little-endian","raw_bits_preserved":true}),
    )
}
fn local_path(root: &Path, raw: &str) -> PathBuf {
    let normalized = raw.replace('\\', "/");
    let bytes = normalized.as_bytes();
    if bytes.len() > 3 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' && bytes[2] == b'/' {
        PathBuf::from(format!(
            "/mnt/{}/{}",
            (bytes[0] as char).to_ascii_lowercase(),
            &normalized[3..]
        ))
    } else {
        let path = PathBuf::from(normalized);
        if path.is_absolute() {
            path
        } else {
            root.join(path)
        }
    }
}

fn validate_assets(root: &Path, assets_sha: &str) -> Result<(Value, PathBuf), String> {
    let raw = bound(&root.join(ASSET_RELATIVE), assets_sha)?;
    let assets: Value = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    require(
        assets["status"] == "REGISTERED_QKV_FP32_AOT_NUMERIC_ASSETS",
        "asset status",
    )?;
    let files = assets["assets"]["files"]
        .as_array()
        .ok_or("asset file list missing")?;
    require(!files.is_empty(), "asset file list empty")?;
    for item in files {
        let path = local_path(root, item["path"].as_str().ok_or("asset path")?);
        bound(&path, item["sha256"].as_str().ok_or("asset SHA")?)?;
    }
    let bridge = &assets["assets"]["bridge"];
    let bridge_path = local_path(root, bridge["path"].as_str().ok_or("bridge path")?);
    let bridge_raw = bound(&bridge_path, bridge["sha256"].as_str().ok_or("bridge SHA")?)?;
    require(
        bridge_raw.len() >= 20
            && bridge_raw.starts_with(b"\x7fELF")
            && bridge_raw[4] == 2
            && bridge_raw[5] == 1
            && bridge_raw[18..20] == [62, 0],
        "bridge must be native ELF64 x86-64",
    )?;
    require(
        assets["baseline"]["receipt"]["sha256"] == BASELINE_RECEIPT_SHA
            && assets["baseline"]["output"]["sha256"] == BASELINE_OUTPUT_SHA,
        "registered baseline identity",
    )?;
    for key in ["receipt", "output"] {
        let item = &assets["baseline"][key];
        bound(
            &local_path(root, item["path"].as_str().ok_or("baseline asset path")?),
            item["sha256"].as_str().ok_or("baseline asset SHA")?,
        )?;
    }
    Ok((assets, bridge_path))
}

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}
type Init = unsafe extern "C" fn() -> c_int;
type Launch = unsafe extern "C" fn(*const u16, *const u16, *mut u16, *mut c_void) -> c_int;

struct Bridge {
    _handle: *mut c_void,
    init: Init,
    launch: Launch,
}
// No Drop/dlclose: the bridge owns a loaded CUDA module and has no destroy ABI.
// Its native library and CUDA module deliberately remain loaded until exit.
impl Bridge {
    unsafe fn loader_error(label: &str) -> String {
        let error = unsafe { dlerror() };
        if error.is_null() {
            format!("{label}: missing symbol or loader failure")
        } else {
            format!(
                "{label}: {}",
                unsafe { CStr::from_ptr(error) }.to_string_lossy()
            )
        }
    }
    fn load(path: &Path) -> Result<Self, String> {
        let path = CString::new(path.to_str().ok_or("non-UTF8 bridge path")?)
            .map_err(|e| e.to_string())?;
        unsafe {
            let handle = dlopen(path.as_ptr(), 2); // RTLD_NOW | RTLD_LOCAL
            if handle.is_null() {
                return Err(Self::loader_error("dlopen"));
            }
            dlerror();
            let init = dlsym(handle, b"qkv_fp32_init\0".as_ptr().cast());
            if init.is_null() {
                return Err(Self::loader_error("qkv_fp32_init"));
            }
            dlerror();
            let launch = dlsym(handle, b"qkv_fp32_launch\0".as_ptr().cast());
            if launch.is_null() {
                return Err(Self::loader_error("qkv_fp32_launch"));
            }
            Ok(Self {
                _handle: handle,
                init: std::mem::transmute::<*mut c_void, Init>(init),
                launch: std::mem::transmute::<*mut c_void, Launch>(launch),
            })
        }
    }
    fn initialize(&self, rt: &CudaRuntime, stream: &Arc<CudaStream>) -> Result<(), String> {
        require(
            Arc::ptr_eq(stream.context(), &rt.device),
            "init CUDA context mismatch",
        )?;
        stream
            .context()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        let status = unsafe { (self.init)() };
        stream
            .synchronize()
            .map_err(|e| format!("init synchronization: {e}"))?;
        require(status == 0, &format!("qkv_fp32_init returned {status}"))
    }
    fn launch(
        &self,
        rt: &CudaRuntime,
        stream: &Arc<CudaStream>,
        device: &operands::DeviceOperands,
        output: &mut CudaSlice<u16>,
    ) -> Result<Value, String> {
        require(
            device.input.len() == operands::M * operands::K
                && device.weights.len() == operands::N * operands::K
                && output.len() == ACTIVE + TAIL,
            "AOT matrix/guard dimensions",
        )?;
        require(
            Arc::ptr_eq(stream.context(), &rt.device)
                && Arc::ptr_eq(device.input.context(), &rt.device)
                && Arc::ptr_eq(device.weights.context(), &rt.device)
                && Arc::ptr_eq(output.context(), &rt.device),
            "AOT context mismatch",
        )?;
        stream
            .context()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        let (input, _input_guard) = device.input.device_ptr(stream);
        let (weights, _weight_guard) = device.weights.device_ptr(stream);
        let (out, _output_guard) = output.device_ptr_mut(stream);
        let rc = unsafe {
            (self.launch)(
                input as *const u16,
                weights as *const u16,
                out as *mut u16,
                stream.cu_stream() as *mut c_void,
            )
        };
        // Keep all cudarc access guards alive through the actual FFI submission
        // and completion. No naked .0 pointer extraction across this boundary.
        let completion = stream
            .synchronize()
            .map_err(|e| format!("AOT completion: {e}"));
        require(
            rc == 0,
            &format!("qkv_fp32_launch returned {rc}; synchronization={completion:?}"),
        )?;
        completion?;
        Ok(json!({"status":rc,"calls":1,"stream_synchronized":true,
            "input_pointer":format!("0x{input:016x}"),"weights_pointer":format!("0x{weights:016x}"),
            "output_pointer":format!("0x{out:016x}"),"stream":stream.cu_stream() as usize,
            "abi":"int qkv_fp32_launch(const uint16_t*,const uint16_t*,uint16_t*,void* cudaStream_t)"}))
    }
}

fn baseline_reference(root: &Path) -> Result<(Vec<u8>, Value), String> {
    let dir = root
        .join("target/fork-parity-20260908/qkv-fp32-aot-r1/operands/passed-baseline-r1-snapshot");
    let receipt_raw = bound(&dir.join("receipt.json"), BASELINE_RECEIPT_SHA)?;
    let receipt: Value = serde_json::from_slice(&receipt_raw).map_err(|e| e.to_string())?;
    require(
        receipt["status"] == "PASS_FROZEN_QKV_LT_BASELINE_DIAGNOSTIC_ONLY",
        "baseline receipt status",
    )?;
    let raw = bound(&dir.join("raw-report.json"), BASELINE_REPORT_SHA)?;
    let report: Value = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    require(
        report["status"] == "PASS_QKV_LT_BASELINE_REPLAY_NOT_CERTIFIED"
            && report["test_executable"]["sha256"] == BASELINE_BINARY_SHA
            && report["operand_source_sha256"] == BASELINE_HELPER_SHA
            && report["public_output"]["sha256"] == BASELINE_OUTPUT_SHA,
        "baseline report provenance",
    )?;
    bound(&dir.join("probe-passed-baseline-wsl"), BASELINE_BINARY_SHA)?;
    bound(&dir.join("qkv_lt_baseline.rs"), BASELINE_HELPER_SHA)?;
    require(
        hash(include_bytes!("support/qkv_lt_baseline.rs")) == BASELINE_HELPER_SHA,
        "compiled baseline helper changed",
    )?;
    let output = bound(&dir.join("public-output.f16le"), BASELINE_OUTPUT_SHA)?;
    require(output.len() == ACTIVE * 2, "baseline archive output length")?;
    Ok((
        output,
        json!({"receipt":record(&dir.join("receipt.json"))?,"report":record(&dir.join("raw-report.json"))?,
        "output":record(&dir.join("public-output.f16le"))?,"executed_binary":record(&dir.join("probe-passed-baseline-wsl"))?}),
    ))
}

#[allow(clippy::too_many_arguments)]
fn run_phase(
    dir: &Path,
    name: &str,
    stream: &Arc<CudaStream>,
    device: &operands::DeviceOperands,
    host: &operands::HostOperands,
    report: &mut Value,
    launch: impl FnOnce(&mut CudaSlice<u16>) -> Result<Value, String>,
) -> Result<Vec<u16>, String> {
    fs::create_dir(dir.join(name)).map_err(|e| e.to_string())?;
    let initial_tail: Vec<u16> = (0..TAIL).map(|i| 0x5000u16 ^ i as u16).collect();
    let mut initial = vec![0x7e00u16; ACTIVE];
    initial.extend_from_slice(&initial_tail);
    let mut output = stream.clone_htod(&initial).map_err(|e| e.to_string())?;
    let read_initial = stream.clone_dtoh(&output).map_err(|e| e.to_string())?;
    let input_before = stream
        .clone_dtoh(&device.input)
        .map_err(|e| e.to_string())?;
    let weights_before = stream
        .clone_dtoh(&device.weights)
        .map_err(|e| e.to_string())?;
    require(
        read_initial == initial && input_before == host.input && weights_before == host.weights,
        "pre-launch actual buffer identity",
    )?;
    report["runs"][name] = json!({"status":"PREPARED_NOT_EXECUTED","input_before":export(dir,&format!("{name}/input-before.f16le"),&input_before,&[operands::M,operands::K])?,
        "weights_before":export(dir,&format!("{name}/weights-before.f16le"),&weights_before,&[operands::N,operands::K])?,
        "output":{"initial":export(dir,&format!("{name}/output-initial.f16le"),&read_initial,&[ACTIVE+TAIL])?,
            "guard_before":export(dir,&format!("{name}/guard-before.f16le"),&initial_tail,&[TAIL])?}});
    save(dir, report);
    eprintln!(
        "[qkv-aot-numeric] phase=before_launch arm={name} m={} n={} k={} lane=0",
        operands::M,
        operands::N,
        operands::K
    );
    let execution = match launch(&mut output) {
        Ok(value) => value,
        Err(error) => {
            report["runs"][name]["status"] = json!("EXECUTION_FAILED");
            report["runs"][name]["execution_error"] = json!(error);
            save(dir, report);
            return Err(format!(
                "{name} launch/completion failed; no further DTOH or candidate launch"
            ));
        }
    };
    let actual = stream.clone_dtoh(&output).map_err(|e| e.to_string())?;
    let input_after = stream
        .clone_dtoh(&device.input)
        .map_err(|e| e.to_string())?;
    let weights_after = stream
        .clone_dtoh(&device.weights)
        .map_err(|e| e.to_string())?;
    stream.synchronize().map_err(|e| e.to_string())?;
    let all_finite = actual[..ACTIVE]
        .iter()
        .all(|v| f16_to_f32_bits(*v).is_finite());
    let guard_ok = actual[ACTIVE..] == initial_tail;
    let input_ok = input_after == input_before;
    let weights_ok = weights_after == weights_before;
    let item = &mut report["runs"][name];
    item["execution"] = execution;
    item["input_after"] = export(
        dir,
        &format!("{name}/input-after.f16le"),
        &input_after,
        &[operands::M, operands::K],
    )?;
    item["weights_after"] = export(
        dir,
        &format!("{name}/weights-after.f16le"),
        &weights_after,
        &[operands::N, operands::K],
    )?;
    item["input_unchanged"] = json!(input_ok);
    item["weights_unchanged"] = json!(weights_ok);
    item["output"]["active"] = export(
        dir,
        &format!("{name}/output.f16le"),
        &actual[..ACTIVE],
        &[operands::M, operands::N],
    )?;
    item["output"]["guard_after"] = export(
        dir,
        &format!("{name}/guard-after.f16le"),
        &actual[ACTIVE..],
        &[TAIL],
    )?;
    item["output"]["all_active_finite"] = json!(all_finite);
    item["output"]["guard_unchanged"] = json!(guard_ok);
    item["status"] = json!("NUMERIC_OUTPUT_COLLECTED");
    save(dir, report);
    require(
        all_finite && guard_ok && input_ok && weights_ok,
        "active finite/NaN overwrite/guard/immutable operands gate",
    )?;
    eprintln!("[qkv-aot-numeric] phase=complete arm={name} full_m_tail=1 guard_unchanged=1 operands_unchanged=1");
    Ok(actual[..ACTIVE].to_vec())
}

fn execute(
    root: &Path,
    dir: &Path,
    model: &Path,
    assets_sha: &str,
    report: &mut Value,
) -> Result<(), String> {
    let (assets, bridge_path) = validate_assets(root, assets_sha)?;
    report["assets"] = assets;
    report["assets_manifest"] = record(&root.join(ASSET_RELATIVE))?;
    let (expected, archive) = baseline_reference(root)?;
    report["baseline_archive"] = archive;
    let host = operands::load_operands(root, model)?;
    report["operand_provenance"] = host.provenance.clone();
    let rt = CudaRuntime::new()?;
    let stream = rt.device.new_stream().map_err(|e| e.to_string())?;
    let device = operands::upload_operands(&rt, &stream, &host)?;
    report["backend_build_fingerprint"] =
        serde_json::to_value(backend_build_fingerprint()).map_err(|e| e.to_string())?;
    report["stream_id"] = json!(stream.cu_stream() as usize);
    stream
        .context()
        .bind_to_thread()
        .map_err(|e| e.to_string())?;
    let bridge = Bridge::load(&bridge_path)?;
    bridge.initialize(&rt, &stream)?;
    report["bridge"] = json!({"file":record(&bridge_path)?,"init_status":0,
        "init_context_bound":true,"library_lifetime":"kept loaded until process exit; no destroy ABI",
        "launch_abi":"int qkv_fp32_launch(const uint16_t*,const uint16_t*,uint16_t*,void* cudaStream_t)"});
    report["phase"] = json!("baseline");
    save(dir, report);
    let baseline = run_phase(dir, "baseline", &stream, &device, &host, report, |output| {
        operands::run_lt_baseline(&rt, &stream, &device, output)?;
        Ok(json!({"status":0,"calls":1,"stream_synchronized":true,
            "api":"public cublaslt_gemm_f16out_with_layout TN","compute":"CUBLAS_COMPUTE_32F"}))
    })?;
    let matched = half_bytes(&baseline) == expected;
    report["runs"]["baseline"]["archived_active_all_bits_equal"] = json!(matched);
    save(dir, report);
    require(
        matched,
        "baseline differs from frozen output; candidate is not launched",
    )?;
    report["phase"] = json!("candidate");
    let candidate = run_phase(
        dir,
        "candidate",
        &stream,
        &device,
        &host,
        report,
        |output| bridge.launch(&rt, &stream, &device, output),
    )?;
    report["phase"] = json!("candidate_repeat");
    let repeated = run_phase(
        dir,
        "candidate_repeat",
        &stream,
        &device,
        &host,
        report,
        |output| bridge.launch(&rt, &stream, &device, output),
    )?;
    let repeat_mismatches = candidate
        .iter()
        .zip(&repeated)
        .filter(|(a, b)| a != b)
        .count();
    report["candidate_repeat_comparison"] = json!({"elements":ACTIVE,
        "raw_bit_mismatches":repeat_mismatches,"all_bits_equal":repeat_mismatches==0});
    report["repeated_runs_deterministic_raw_half"] = json!(repeat_mismatches == 0);
    save(dir, report);
    require(
        repeat_mismatches == 0,
        "candidate numerical repeats must match every active half bit",
    )?;
    let mut max_abs = 0.0f64;
    let mut worst = 0usize;
    let mut bit_mismatches = 0usize;
    for (index, (&a, &b)) in baseline.iter().zip(&candidate).enumerate() {
        bit_mismatches += usize::from(a != b);
        let abs = (f16_to_f32_bits(a) as f64 - f16_to_f32_bits(b) as f64).abs();
        if abs > max_abs {
            max_abs = abs;
            worst = index;
        }
    }
    report["candidate_vs_lt"] = json!({"elements":ACTIVE,"max_abs":max_abs,"worst_flat_index":worst,
        "raw_bit_mismatches":bit_mismatches,"strict_max_abs_limit":0.01,"pass":max_abs<0.01,
        "scope":"all physical M5054 rows of half outputs; full FP64 reference checked by CPU consumer"});
    stream.synchronize().map_err(|e| e.to_string())?;
    report["final_stream_synchronized"] = json!(true);
    save(dir, report);
    require(
        max_abs < 0.01,
        "candidate/Lt maxabs must be strictly below 0.01",
    )?;
    report["status"] = json!("PASS_AOT_VS_LT_PENDING_FP64_NOT_CERTIFIED");
    report["phase"] = json!("complete");
    save(dir, report);
    Ok(())
}

fn numeric_entry(assets_sha: &str) {
    match std::env::var("KATAGO_RUN_QKV_AOT_NUMERIC").as_deref() {
        Err(std::env::VarError::NotPresent) => {
            eprintln!("SKIP: set KATAGO_RUN_QKV_AOT_NUMERIC=1");
            return;
        }
        Ok("1") => {}
        value => panic!("invalid explicit QKV numerical opt-in {value:?}"),
    }
    let kernel = fs::read_to_string("/proc/sys/kernel/osrelease").unwrap();
    assert!(
        kernel.to_lowercase().contains("microsoft"),
        "frozen WSL platform required"
    );
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let output = PathBuf::from(
        std::env::var_os("KATAGO_QKV_AOT_NUMERIC_DIR")
            .expect("fresh explicit numerical output directory required"),
    );
    let output = if output.is_absolute() {
        output
    } else {
        root.join(output)
    };
    assert!(!output.exists(), "output evidence must not already exist");
    assert!(
        output
            .parent()
            .unwrap()
            .canonicalize()
            .unwrap()
            .starts_with(root.join("target").canonicalize().unwrap()),
        "numerical output must stay under workspace target"
    );
    fs::create_dir(&output).unwrap();
    let model = PathBuf::from(
        std::env::var_os("KATAGO_TEST_MODEL_DIR").expect("explicit model directory required"),
    )
    .join("kata1-tf3-b11c768-s11001M-d5973M.bin.gz");
    let mut report = json!({"schema":1,"status":"PREPARING_NOT_COMPLETE","phase":"preflight",
        "scope":"real first-QKV standalone AOT numerical comparison; full FP64 gate is performed by external CPU consumer",
        "test_executable":record(&std::env::current_exe().unwrap()).unwrap(),
        "source":{"test_sha256":hash(include_bytes!("probe_qkv_aot_numeric_cuda.rs")),
            "baseline_helper_sha256":hash(include_bytes!("support/qkv_lt_baseline.rs"))},
        "assets_manifest_expected_sha256":assets_sha,"platform":std::env::consts::OS,"kernel_release":kernel.trim(),
        "shape":{"batch":operands::BATCH,"m":operands::M,"n":operands::N,"k":operands::K},
        "registered_order":["baseline","candidate","candidate_repeat"],"lanes":1,
        "output_guard":{"active_elements":ACTIVE,"tail_elements":TAIL,"active_initial_half_bits":0x7e00,
            "tail_pattern":"u16 0x5000 xor index0..255"},
        "numeric_gate":{"candidate_vs_lt_max_abs_strictly_less_than":0.01,
            "candidate_vs_fp64_max_abs_strictly_less_than":0.01,"fp64_gate_location":"external CPU consumer",
            "repeated_runs_deterministic_raw_half":true},
        "timing_performed":false,"warmup_performed":false,"production_certified":false,
        "is_full_network_gate":false,"is_cpp_golden_gate":false,"runs":{}});
    save(&output, &report);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        execute(&root, &output, &model, assets_sha, &mut report)
    }));
    match result {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            report["status"] = json!("FAILED_NOT_CERTIFIED");
            report["failure"] = json!(error);
            save(&output, &report);
            panic!("QKV numerical harness failed; inspect frozen output report");
        }
        Err(panic) => {
            report["status"] = json!("FAILED_NOT_CERTIFIED");
            report["failure"] = json!(panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("non-string panic"));
            save(&output, &report);
            std::panic::resume_unwind(panic);
        }
    }
    eprintln!(
        "[qkv-aot-numeric] status=PASS_AOT_VS_LT_PENDING_FP64_NOT_CERTIFIED raw_report={}",
        output.join("report.json").display()
    );
}

#[test]
fn probe_qkv_aot_numeric_cuda() {
    numeric_entry("d3469f72e305e80f56375185d1cfc11c18e5a46dd908c9439d2c0c13bc8ec1ac");
}
