//! Opt-in real-operand B14 QKV baseline. No AOT kernel, performance measurement,
//! or production integration. Explicit independent Lt replay must match every
//! packed half output bit of the public FP32-compute runtime API.
#![cfg(feature = "cuda")]

#[path = "support/qkv_lt_metadata.rs"]
mod metadata;
#[path = "support/qkv_lt_baseline.rs"]
mod operands;

use cudarc::driver::CudaSlice;
use kata_nn::backends::cuda::{
    CudaRuntime, backend_build_fingerprint, device_fingerprint, f16_to_f32_bits,
};
use kata_nn::tactic_plan::sha256_file;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn half_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn record(path: &Path) -> Value {
    json!({"path":path,"sha256":sha256_file(path).unwrap()})
}
fn raw(path: &Path, bytes: &[u8]) -> Value {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap()
        .write_all(bytes)
        .unwrap();
    json!({"path":path,"sha256":hash(bytes),"bytes":bytes.len()})
}
fn save(dir: &Path, report: &Value) {
    fs::write(
        dir.join("report.json"),
        serde_json::to_vec_pretty(report).unwrap(),
    )
    .unwrap();
}

#[test]
fn probe_qkv_lt_baseline_cuda() {
    match std::env::var("KATAGO_RUN_QKV_LT_BASELINE").as_deref() {
        Err(std::env::VarError::NotPresent) => {
            eprintln!("SKIP: set KATAGO_RUN_QKV_LT_BASELINE=1");
            return;
        }
        Ok("1") => {}
        value => panic!("invalid explicit QKV baseline opt-in {value:?}"),
    }
    assert_eq!(
        std::env::consts::OS,
        "linux",
        "frozen real input requires WSL"
    );
    let kernel = fs::read_to_string("/proc/sys/kernel/osrelease").unwrap();
    assert!(
        kernel.to_lowercase().contains("microsoft"),
        "WSL runtime required"
    );
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let model = PathBuf::from(
        std::env::var_os("KATAGO_TEST_MODEL_DIR").expect("explicit model directory required"),
    )
    .join("kata1-tf3-b11c768-s11001M-d5973M.bin.gz");
    let output = PathBuf::from(
        std::env::var_os("KATAGO_QKV_LT_BASELINE_DIR")
            .expect("fresh explicit output directory required"),
    );
    assert!(
        !output.exists()
            && output
                .parent()
                .unwrap()
                .canonicalize()
                .unwrap()
                .starts_with(root.join("target").canonicalize().unwrap())
    );
    fs::create_dir(&output).unwrap();
    let mut report = json!({"schema":1,"status":"PREPARING_NOT_COMPLETE","production_certified":false,
        "scope":"real first-QKV standalone baseline and independent f16out Lt replay; not intercepted production cache or whole-network gate",
        "model":record(&model),"test_executable":record(&std::env::current_exe().unwrap()),
        "test_source_sha256":hash(include_bytes!("probe_qkv_lt_baseline_cuda.rs")),
        "operand_source_sha256":hash(include_bytes!("support/qkv_lt_baseline.rs")),
        "metadata_source_sha256":hash(include_bytes!("support/qkv_lt_metadata.rs")),
        "precision_source_sha256":hash(include_bytes!("support/lt_replay_precision.rs")),
        "platform":std::env::consts::OS,"kernel_release":kernel.trim(),
        "shape":{"m":operands::M,"n":operands::N,"k":operands::K,"batch":operands::BATCH},
        "compute":"CUBLAS_COMPUTE_32F","scale":"CUDA_R_32F","input":"CUDA_R_16F","output":"CUDA_R_16F",
        "weight_layout":"TN row-major [1152,384] Q|K|V","alpha":1,"beta":0,"timing_performed":false});
    save(&output, &report);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let host = operands::load_operands(&root, &model).expect("frozen CPU operand provenance");
        report["operand_provenance"] = host.provenance.clone();
        report["input_host"] = raw(&output.join("input-host.f16le"), &half_bytes(&host.input));
        report["weights_host"] = raw(
            &output.join("weights-host.f16le"),
            &half_bytes(&host.weights),
        );
        report["weights_source"] = raw(
            &output.join("weights-source.f32le"),
            &host
                .weights_fp32
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        save(&output, &report);
        let rt = CudaRuntime::new().expect("working CUDA required by explicit probe");
        let prefix: Value = serde_json::from_slice(
            &fs::read(root.join("target/fork-parity-20260908/g3-prefix-trace-wsl-r2/meta.json"))
                .unwrap(),
        )
        .unwrap();
        let build = serde_json::to_value(backend_build_fingerprint()).unwrap();
        assert_eq!(build, prefix["backend_build_fingerprint"]);
        let device = device_fingerprint(&rt.device).unwrap();
        let dev = json!({"gpu_name":device.gpu_name,"compute_capability":device.compute_capability,"sm_count":device.sm_count,"l2_cache_bytes":device.l2_cache_bytes});
        assert_eq!(dev, prefix["device_fingerprint"]);
        report["backend_build_fingerprint"] = build;
        report["device_fingerprint"] = dev;
        let stream = rt.device.new_stream().unwrap();
        let gpu = operands::upload_operands(&rt, &stream, &host)
            .expect("actual input and half weight upload roundtrip");
        report["input_device"] = raw(
            &output.join("input-device.f16le"),
            &half_bytes(&stream.clone_dtoh(&gpu.input).unwrap()),
        );
        report["weights_device"] = raw(
            &output.join("weights-device.f16le"),
            &half_bytes(&stream.clone_dtoh(&gpu.weights).unwrap()),
        );
        assert_eq!(
            report["input_host"]["sha256"],
            report["input_device"]["sha256"]
        );
        assert_eq!(
            report["weights_host"]["sha256"],
            report["weights_device"]["sha256"]
        );
        let active = operands::M * operands::N;
        let guard: Vec<u16> = (0..256).map(|i| 0x5000 ^ i).collect();
        let mut initial = vec![0x7e00u16; active];
        initial.extend_from_slice(&guard);
        let mut public: CudaSlice<u16> = stream.clone_htod(&initial).unwrap();
        let public_initial = stream.clone_dtoh(&public).unwrap();
        stream.synchronize().unwrap();
        report["public_initial"] = raw(
            &output.join("public-initial.f16le"),
            &half_bytes(&public_initial),
        );
        assert_eq!(public_initial, initial);
        report["output_guard"] = json!({"active_elements":active,"guard_elements":256,"active_initial_half_bits":0x7e00,"tail_pattern":"u16 0x5000 xor index0..255"});
        operands::run_lt_baseline(&rt, &stream, &gpu, &mut public)
            .expect("public Lt FP32 compute/packed half output; no fallback");
        stream.synchronize().unwrap();
        let reference_all = stream.clone_dtoh(&public).unwrap();
        stream.synchronize().unwrap();
        let reference = &reference_all[..active];
        report["public_output"] = raw(&output.join("public-output.f16le"), &half_bytes(&reference));
        report["public_guard_after"] = raw(
            &output.join("public-guard-after.f16le"),
            &half_bytes(&reference_all[active..]),
        );
        save(&output, &report);
        assert_eq!(
            &reference_all[active..],
            guard.as_slice(),
            "public Lt wrote outside active descriptors"
        );
        assert!(reference.iter().all(|x| f16_to_f32_bits(*x).is_finite()));
        report["public_api_success"] = json!(true);
        save(&output, &report);
        let mut explicit: CudaSlice<u16> = stream.clone_htod(&initial).unwrap();
        let explicit_initial = stream.clone_dtoh(&explicit).unwrap();
        stream.synchronize().unwrap();
        report["explicit_initial"] = raw(
            &output.join("explicit-initial.f16le"),
            &half_bytes(&explicit_initial),
        );
        assert_eq!(explicit_initial, initial);
        let selected = metadata::select(&rt, &stream, operands::M, operands::N, operands::K)
            .expect("independent REQ8 first workspace-eligible QKV query");
        let detail = metadata::execute_selected(
            &rt,
            &stream,
            &gpu.input,
            &gpu.weights,
            &mut explicit,
            operands::M,
            operands::N,
            operands::K,
            &selected,
        )
        .expect("independent target stream check/replay");
        report["lt_replay"] = detail;
        save(&output, &report);
        assert_eq!(
            report["lt_replay"]["status"], "PASS_REPLAY_EXECUTED",
            "no unsupported replay can be baseline proof"
        );
        let actual_all = stream.clone_dtoh(&explicit).unwrap();
        stream.synchronize().unwrap();
        let actual = &actual_all[..active];
        report["explicit_output"] =
            raw(&output.join("explicit-output.f16le"), &half_bytes(&actual));
        report["explicit_guard_after"] = raw(
            &output.join("explicit-guard-after.f16le"),
            &half_bytes(&actual_all[active..]),
        );
        save(&output, &report);
        assert_eq!(
            &actual_all[active..],
            guard.as_slice(),
            "independent Lt replay wrote outside active descriptors"
        );
        assert!(
            actual.iter().all(|x| f16_to_f32_bits(*x).is_finite()),
            "all active half outputs must overwrite NaN initialization"
        );
        let mismatches = reference.iter().zip(actual).filter(|(a, b)| a != b).count();
        report["output_comparison"] = json!({"elements":actual.len(),"raw_half_bit_mismatches":mismatches,"all_bits_equal":mismatches==0});
        let input_after = stream.clone_dtoh(&gpu.input).unwrap();
        let weights_after = stream.clone_dtoh(&gpu.weights).unwrap();
        stream.synchronize().unwrap();
        report["input_after"] = raw(&output.join("input-after.f16le"), &half_bytes(&input_after));
        report["weights_after"] = raw(
            &output.join("weights-after.f16le"),
            &half_bytes(&weights_after),
        );
        save(&output, &report);
        assert_eq!(input_after, host.input);
        assert_eq!(weights_after, host.weights);
        assert_eq!(
            mismatches, 0,
            "independent selected algorithm must reproduce public API output bits"
        );
        report["operands_unchanged"] = json!(true);
        report["status"] = json!("PASS_QKV_LT_BASELINE_REPLAY_NOT_CERTIFIED");
        save(&output, &report);
    }));
    if let Err(panic) = result {
        report["status"] = json!("FAILED_QKV_BASELINE_EVIDENCE_PRESERVED");
        report["failure"] = json!(
            panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("non-string panic")
        );
        save(&output, &report);
        std::panic::resume_unwind(panic);
    }
    eprintln!(
        "[qkv-lt-baseline] status=PASS_QKV_LT_BASELINE_REPLAY_NOT_CERTIFIED physical_batch=14 m=5054 n=1152 k=384 compute=32f input=16f output=16f layout=tn"
    );
}
