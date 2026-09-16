//! Opt-in Windows B14 QKV REQ8 numerical collection. No timing or production change.
//! All candidates come from one independent query; the public API control has its
//! own ordinary query/cache. The separate CPU FP64/Lt gate must admit any timing.
#![cfg(feature = "cuda")]

#[path = "support/qkv_lt_pool_metadata.rs"]
mod metadata;
#[path = "support/qkv_lt_baseline.rs"]
mod operands;

use cudarc::driver::{CudaSlice, DevicePtr};
use kata_nn::backends::cuda::{CudaRuntime, backend_build_fingerprint, device_fingerprint};
use kata_nn::tactic_plan::{installed_plan_id, sha256_file, tactic_var};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const FINGERPRINT_SHA: &str = "17f0b3b99aed726467316f033df63debef2dd1c8a3a2b389f81eb3e404f6714c";
const MODEL_SHA: &str = "1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6";

fn hash(bytes: &[u8]) -> String { hex::encode(Sha256::digest(bytes)) }
fn half_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn finite(values: &[u16]) -> bool {
    values.iter().all(|x| x & 0x7c00 != 0x7c00)
}
fn record(path: &Path) -> Value {
    json!({"path":path,"sha256":sha256_file(path).unwrap(),"bytes":fs::metadata(path).unwrap().len()})
}
fn raw(path: &Path, bytes: &[u8]) -> Value {
    OpenOptions::new().write(true).create_new(true).open(path).unwrap()
        .write_all(bytes).unwrap();
    json!({"path":path,"sha256":hash(bytes),"bytes":bytes.len()})
}
fn save(dir: &Path, report: &Value) {
    fs::write(dir.join("report.json"), serde_json::to_vec_pretty(report).unwrap()).unwrap();
}

#[cfg(windows)]
fn loaded_libraries() -> Value {
    use std::ffi::c_void;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetModuleHandleW(name: *const u16) -> *mut c_void;
        fn GetModuleFileNameW(module: *mut c_void, file: *mut u16, size: u32) -> u32;
    }
    let name: Vec<u16> = "cublasLt64_13.dll".encode_utf16().chain(Some(0)).collect();
    let module = unsafe { GetModuleHandleW(name.as_ptr()) };
    assert!(!module.is_null(), "the actual loaded CUDA 13 cuBLASLt DLL is required");
    let mut buffer = vec![0u16; 32768];
    let count = unsafe { GetModuleFileNameW(module, buffer.as_mut_ptr(), buffer.len() as u32) };
    assert!(count > 0 && (count as usize) < buffer.len(), "actual Lt DLL path query failed/truncated");
    let path = PathBuf::from(String::from_utf16(&buffer[..count as usize]).unwrap());
    assert!(path.is_absolute() && path.is_file(), "actual loaded Lt DLL path must resolve");
    json!({"cublasLt":record(&path),"source":"GetModuleHandleW + GetModuleFileNameW on the loaded process module",
        "version":unsafe{cudarc::cublaslt::sys::cublasLtGetVersion()},"platform":"windows"})
}
#[cfg(not(windows))]
fn loaded_libraries() -> Value { panic!("this bounded candidate-pool registration targets Windows") }

#[test]
fn probe_qkv_lt_pool_cuda() {
    match std::env::var("KATAGO_RUN_QKV_LT_POOL").as_deref() {
        Err(std::env::VarError::NotPresent) => {
            eprintln!("SKIP: set KATAGO_RUN_QKV_LT_POOL=1");
            return;
        }
        Ok("1") => {}
        value => panic!("invalid explicit QKV pool opt-in {value:?}"),
    }
    assert_eq!(std::env::consts::OS, "windows", "registered current Windows pool only");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap();
    let model = PathBuf::from(std::env::var_os("KATAGO_TEST_MODEL_DIR")
        .expect("explicit model directory required")).join("kata1-tf3-b11c768-s11001M-d5973M.bin.gz");
    let fingerprint_path = PathBuf::from(std::env::var_os("KATAGO_QKV_POOL_FINGERPRINT")
        .expect("explicit frozen Windows fingerprint required"));
    let fingerprint_raw = fs::read(&fingerprint_path).unwrap();
    assert_eq!(hash(&fingerprint_raw), FINGERPRINT_SHA, "unregistered fingerprint file");
    let fingerprint: Value = serde_json::from_slice(&fingerprint_raw).unwrap();
    assert_eq!(fingerprint["model_sha256"], MODEL_SHA);
    assert_eq!(sha256_file(&model).unwrap(), MODEL_SHA);
    let output = PathBuf::from(std::env::var_os("KATAGO_QKV_LT_POOL_DIR")
        .expect("fresh explicit output directory required"));
    assert!(!output.exists() && output.parent().unwrap().canonicalize().unwrap()
        .starts_with(root.join("target").canonicalize().unwrap()), "fresh target output required");
    fs::create_dir(&output).unwrap();
    let mut report = json!({"schema":1,"status":"PREPARING_NOT_COMPLETE","production_certified":false,
        "timing_performed":false,"timing_admitted":false,"cpu_fp64_gate":"NOT_RUN",
        "scope":"Windows standalone exact B14 QKV pool; immutable operands originate from the hash-bound WSL prefix; no whole-network or cross-platform numerical certification",
        "model":record(&model),"test_executable":record(&std::env::current_exe().unwrap()),
        "fingerprint":record(&fingerprint_path),"registered_fingerprint":fingerprint,
        "test_source_sha256":hash(include_bytes!("probe_qkv_lt_pool_cuda.rs")),
        "metadata_source_sha256":hash(include_bytes!("support/qkv_lt_pool_metadata.rs")),
        "operand_source_sha256":hash(include_bytes!("support/qkv_lt_baseline.rs")),
        "precision_source_sha256":hash(include_bytes!("support/lt_replay_precision.rs")),
        "platform":std::env::consts::OS,"shape":{"m":operands::M,"n":operands::N,"k":operands::K,"batch":operands::BATCH},
        "compute":"CUBLAS_COMPUTE_32F","scale":"CUDA_R_32F","input":"CUDA_R_16F","output":"CUDA_R_16F",
        "weight_layout":"TN row-major [1152,384] Q|K|V","alpha":1,"beta":0,
        "candidates":[]});
    save(&output, &report);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert!(installed_plan_id().is_none(), "diagnostic must not install or mutate a tactic plan");
        let mut actual_tactics = serde_json::Map::new();
        for (key, expected) in [
            ("KATAGO_CUDA_CUBLASLT","1"),("KATAGO_CUDA_CUBLASLT_RANK","heuristic"),
            ("KATAGO_CUDA_GEMM_LAYOUT","tn"),
        ] {
            let value = tactic_var(key).expect("required explicit pool tactic");
            assert_eq!(value, expected, "unexpected {key}");
            actual_tactics.insert(key.into(), json!(value));
        }
        report["tactics"] = Value::Object(actual_tactics);
        // Reuse only CPU provenance/packing from the frozen helper. Its upload
        // routine deliberately requires the old WSL build, so Windows instead
        // binds its actual build/device to the separately frozen fingerprint.
        let host = operands::load_operands(&root, &model).expect("frozen CPU operand provenance");
        assert!(host.input.chunks_exact(361*operands::K)
            .all(|board| board == &host.input[..361*operands::K]), "all 14 reference boards must repeat");
        report["operand_provenance"] = host.provenance.clone();
        report["input_host"] = raw(&output.join("input-host.f16le"), &half_bytes(&host.input));
        report["weights_host"] = raw(&output.join("weights-host.f16le"), &half_bytes(&host.weights));
        report["weights_source"] = raw(&output.join("weights-source.f32le"),
            &host.weights_fp32.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>());
        save(&output, &report);
        let rt = CudaRuntime::new().expect("working CUDA required by explicit pool opt-in");
        assert!(!kata_nn::backends::cuda_exec::capturing());
        let build = serde_json::to_value(backend_build_fingerprint()).unwrap();
        assert_eq!(build, report["registered_fingerprint"]["backend_build"], "actual Windows build identity");
        assert_eq!(build["fp16_encoding_revision"], 1);
        assert_eq!(build["cublaslt_version"], 130600);
        let device = device_fingerprint(&rt.device).unwrap();
        let dev = json!({"gpu_name":device.gpu_name,"compute_capability":device.compute_capability,
            "sm_count":device.sm_count,"l2_cache_bytes":device.l2_cache_bytes});
        for key in ["gpu_name","compute_capability","sm_count","l2_cache_bytes"] {
            assert_eq!(dev[key], report["registered_fingerprint"][key], "actual GPU {key}");
        }
        report["backend_build_fingerprint"] = build;
        report["device_fingerprint"] = dev;
        report["loaded_libraries"] = loaded_libraries();
        assert_eq!(report["loaded_libraries"]["version"], 130600);
        save(&output, &report);
        let stream = rt.device.new_stream().unwrap();
        let gpu = operands::DeviceOperands {
            input:stream.clone_htod(&host.input).unwrap(),
            weights:stream.clone_htod(&host.weights).unwrap(),
        };
        let input_device = stream.clone_dtoh(&gpu.input).unwrap();
        let weights_device = stream.clone_dtoh(&gpu.weights).unwrap();
        stream.synchronize().unwrap();
        report["input_device"] = raw(&output.join("input-device.f16le"), &half_bytes(&input_device));
        report["weights_device"] = raw(&output.join("weights-device.f16le"), &half_bytes(&weights_device));
        assert_eq!(input_device, host.input);
        assert_eq!(weights_device, host.weights);
        let active = operands::M * operands::N;
        let guard: Vec<u16> = (0..256).map(|i| 0x5000 ^ i).collect();
        let mut initial = vec![0x7e00u16; active];
        initial.extend_from_slice(&guard);
        let mut device_output: CudaSlice<u16> = stream.clone_htod(&initial).unwrap();
        report["output_guard"] = json!({"active_elements":active,"guard_elements":256,
            "active_initial_half_bits":0x7e00,"tail_pattern":"u16 0x5000 xor index0..255"});
        report["output_pointer"] = {
            let (pointer, _guard) = device_output.device_ptr(&stream);
            json!(format!("0x{pointer:016x}"))
        };
        report["stream"] = json!(format!("0x{:016x}",stream.cu_stream() as usize));
        report["same_output_pointer_for_public_and_all_candidates"] = json!(true);
        let public_initial = stream.clone_dtoh(&device_output).unwrap();
        stream.synchronize().unwrap();
        report["public"] = json!({"initial":raw(&output.join("public-initial.f16le"), &half_bytes(&public_initial))});
        assert_eq!(public_initial, initial);
        operands::run_lt_baseline(&rt, &stream, &gpu, &mut device_output)
            .expect("fresh public half-output Lt control; no fallback");
        let public_all = stream.clone_dtoh(&device_output).unwrap();
        stream.synchronize().unwrap();
        let public = public_all[..active].to_vec();
        report["public"]["output"] = raw(&output.join("public-output.f16le"), &half_bytes(&public));
        report["public"]["guard_after"] = raw(&output.join("public-guard-after.f16le"), &half_bytes(&public_all[active..]));
        report["public"]["all_finite"] = json!(finite(&public));
        report["public"]["api_success"] = json!(true);
        save(&output, &report);
        assert_eq!(&public_all[active..], guard.as_slice(), "public control tail modified");
        assert!(finite(&public), "public control failed complete active NaN overwrite");
        let pool = metadata::select_pool(&rt, &stream, operands::M, operands::N, operands::K)
            .expect("one independent exact-shape REQ8 query");
        assert_eq!(pool.candidates.len(), 8);
        report["pool_query"] = pool.metadata;
        save(&output, &report);
        for candidate in pool.candidates {
            let index = candidate.original_index;
            assert_eq!(index, report["candidates"].as_array().unwrap().len(), "original pool order");
            report["candidates"].as_array_mut().unwrap().push(json!({"original_index":index,
                "metadata":candidate.metadata,"status":"PREPARING_NOT_COMPLETE","runs":[],
                "timing_admitted":false,"repeat_bits_equal":null,"vs_public":null}));
            save(&output, &report);
            let Some(selection) = candidate.selection else {
                report["candidates"][index]["status"] = json!("METADATA_FAILED_NO_EXECUTION");
                save(&output, &report);
                assert_ne!(index, 0, "public baseline self-control metadata is mandatory");
                continue;
            };
            let mut first: Option<Vec<u16>> = None;
            for repeat_index in 0..2 {
                let stem = format!("candidate-{index:02}-r{repeat_index}");
                stream.memcpy_htod(&initial, &mut device_output).unwrap();
                let before = stream.clone_dtoh(&device_output).unwrap();
                stream.synchronize().unwrap();
                let mut run = json!({"repeat_index":repeat_index,
                    "initial":raw(&output.join(format!("{stem}-initial.f16le")), &half_bytes(&before))});
                assert_eq!(before, initial, "actual candidate NaN/tail initialization");
                let execution = metadata::execute_selected(&rt, &stream, &gpu.input, &gpu.weights,
                    &mut device_output, operands::M, operands::N, operands::K, &selection)
                    .expect("candidate original-object check/execution");
                run["execution"] = execution;
                let after = stream.clone_dtoh(&device_output).unwrap();
                let input_after = stream.clone_dtoh(&gpu.input).unwrap();
                let weights_after = stream.clone_dtoh(&gpu.weights).unwrap();
                stream.synchronize().unwrap();
                run["input_after_sha256"] = json!(hash(&half_bytes(&input_after)));
                run["weights_after_sha256"] = json!(hash(&half_bytes(&weights_after)));
                run["operands_unchanged"] = json!(input_after == host.input && weights_after == host.weights);
                run["output"] = raw(&output.join(format!("{stem}-output.f16le")), &half_bytes(&after[..active]));
                run["guard_after"] = raw(&output.join(format!("{stem}-guard-after.f16le")), &half_bytes(&after[active..]));
                run["all_finite"] = json!(finite(&after[..active]));
                let success = run["execution"]["status"] == "PASS_REPLAY_EXECUTED";
                let unsupported = run["execution"]["status"] == "NOT_SUPPORTED";
                report["candidates"][index]["runs"].as_array_mut().unwrap().push(run);
                save(&output, &report);
                assert_eq!(input_after, host.input, "candidate changed inputs");
                assert_eq!(weights_after, host.weights, "candidate changed weights");
                assert_eq!(&after[active..], guard.as_slice(), "candidate wrote outside active descriptor");
                if unsupported {
                    assert_eq!(after, initial, "zero-launch rejection changed the output");
                    report["candidates"][index]["status"] = json!("NOT_SUPPORTED");
                    save(&output, &report);
                    assert_ne!(index, 0, "public baseline self-control must execute");
                    break;
                }
                assert!(success, "candidate execution error; evidence preserved");
                if repeat_index == 0 {
                    let mismatches = public.iter().zip(&after[..active]).filter(|(a,b)| a != b).count();
                    report["candidates"][index]["vs_public"] = json!({"elements":active,
                        "raw_half_bit_mismatches":mismatches,"all_bits_equal":mismatches==0});
                    if index == 0 {
                        save(&output, &report);
                        assert_eq!(mismatches, 0, "pool #0 must exactly reproduce fresh public API output");
                    }
                    first = Some(after[..active].to_vec());
                } else {
                    let equal = first.as_ref().unwrap().as_slice() == &after[..active];
                    report["candidates"][index]["repeat_bits_equal"] = json!(equal);
                    let all_finite = report["candidates"][index]["runs"].as_array().unwrap()
                        .iter().all(|run| run["all_finite"] == true);
                    report["candidates"][index]["status"] = json!(if equal && all_finite {
                        "COMPLETE_FINITE_REPEAT_PENDING_CPU"
                    } else { "FAILED_NUMERIC_CONTROLS" });
                    if index == 0 {
                        save(&output, &report);
                        assert!(equal && all_finite, "public baseline self-control repeat/finite failure");
                    }
                }
                save(&output, &report);
            }
        }
        let input_after = stream.clone_dtoh(&gpu.input).unwrap();
        let weights_after = stream.clone_dtoh(&gpu.weights).unwrap();
        stream.synchronize().unwrap();
        report["input_after"] = raw(&output.join("input-after.f16le"), &half_bytes(&input_after));
        report["weights_after"] = raw(&output.join("weights-after.f16le"), &half_bytes(&weights_after));
        assert_eq!(input_after, host.input);
        assert_eq!(weights_after, host.weights);
        report["operands_unchanged"] = json!(true);
        report["stream_synchronized"] = json!(true);
        report["status"] = json!("COMPLETE_QKV_LT_POOL_PENDING_CPU");
        save(&output, &report);
    }));
    if let Err(panic) = result {
        report["status"] = json!("FAILED_QKV_LT_POOL_EVIDENCE_PRESERVED");
        report["failure"] = json!(panic.downcast_ref::<String>().map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied()).unwrap_or("non-string panic"));
        save(&output, &report);
        std::panic::resume_unwind(panic);
    }
    eprintln!("[qkv-lt-pool] status=COMPLETE_QKV_LT_POOL_PENDING_CPU physical_batch=14 m=5054 n=1152 k=384 pool_query_calls=1 candidates=8 compute=32f input=16f output=16f layout=tn timing=0");
}
