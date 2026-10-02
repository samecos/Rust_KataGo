//! CPU-only process-state tests. Each scenario runs in a fresh copy of this
//! integration-test executable because the production guards are OnceLocks.
//! No CUDA feature, model files, hardware or private test helpers are needed.

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Barrier};

use kata_nn::tactic_plan::{
    ALLOWED_TACTIC_KEYS, BackendBuildFingerprint, BackendCapabilities, CUDA_FP16_ENCODING_REVISION,
    CUDA_HOST_TACTIC_REVISION, DeviceFingerprint, freeze_quantized_tactics, installed_plan_id,
    load_and_install, tactic_var,
};

const CHILD_SCENARIO: &str = "RUSTGO_TEST_QUANTIZED_TACTIC_FREEZE_SCENARIO";
const TEST_NAME: &str = "quantized_tactics_are_frozen_and_legacy_installation_is_exclusive";
const PLAN_ID: &str = "cpu-only-quantized-tactic-freeze-fixture";

struct PlanFixture {
    path: PathBuf,
    device: DeviceFingerprint,
    build: BackendBuildFingerprint,
    model_sha256: String,
}

impl PlanFixture {
    fn new() -> Self {
        let device = DeviceFingerprint {
            gpu_name: "NVIDIA GeForce RTX 5070 Ti".into(),
            compute_capability: "12.0".into(),
            sm_count: 70,
            l2_cache_bytes: 50331648,
        };
        let build = BackendBuildFingerprint {
            kernel_build_id: format!("sha256:{}", "11".repeat(32)),
            cuda_compiler: "cpu-test-fixture".into(),
            cutlass_version: "3.9.2".into(),
            cutlass_commit: "fixture".into(),
            compiled_sm: vec!["120".into()],
            capabilities: BackendCapabilities {
                dual_ffn: false,
                attention_q64: false,
                attention_q64_serial: false,
            },
            fp16_encoding_revision: Some(CUDA_FP16_ENCODING_REVISION),
            host_tactic_revision: Some(CUDA_HOST_TACTIC_REVISION),
            cublaslt_version: Some(130600),
            strict_attention_artifact: None,
            qkv_immutable_artifact: None,
            ffn_compact_artifact: None,
            outproj_n128_artifact: None,
        };
        let model_sha256 = "aa".repeat(32);
        let plan = serde_json::json!({
            "schema": 2,
            "kind": "cuda-tactic-plan",
            "plan_id": PLAN_ID,
            "backend_build": build,
            "target": {
                "architecture": "sm_120",
                "gpu_name": device.gpu_name,
                "compute_capability": device.compute_capability,
                "sm_count": device.sm_count,
                "l2_cache_bytes": device.l2_cache_bytes,
                "model_sha256": model_sha256,
                "max_batch_size": 16,
            },
            "apply": { "tactic_overrides": { "KATAGO_CUDA_RMS": "v1" } },
        });
        let path = std::env::temp_dir().join(format!(
            "rustgo-quantized-tactic-freeze-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, serde_json::to_vec(&plan).unwrap()).unwrap();
        Self {
            path,
            device,
            build,
            model_sha256,
        }
    }

    fn install(&self) -> Result<(), String> {
        load_and_install(&self.path, &self.device, &self.model_sha256, &self.build)
    }
}

impl Drop for PlanFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn clear_inherited_tactics() {
    // This isolated child has one test and has not spawned reader threads.
    // Never mutate the parent cargo/test runner's environment.
    for key in ALLOWED_TACTIC_KEYS
        .iter()
        .copied()
        .chain(["KATAGO_CUDA_INT8_GEMM_TUNE", "KATAGO_CUDA_INT8_RMS_FUSION"])
    {
        unsafe {
            std::env::remove_var(key);
        }
    }
}

fn freeze_first() {
    let plan = PlanFixture::new();
    unsafe {
        std::env::set_var("KATAGO_CUDA_RMS", "v1");
        std::env::set_var("KATAGO_CUDA_INT8_RMS_FUSION", "0");
    }
    freeze_quantized_tactics().unwrap();
    freeze_quantized_tactics().unwrap(); // An unchanged snapshot is idempotent.
    unsafe {
        std::env::set_var("KATAGO_CUDA_RMS", "w4");
        std::env::set_var("KATAGO_CUDA_INT8_RMS_FUSION", "1");
        std::env::set_var("KATAGO_CUDA_INT8_GEMM_TUNE", "1");
        std::env::set_var("KATAGO_CUDA_PADBATCH", "1");
    }
    assert_eq!(tactic_var("KATAGO_CUDA_RMS").unwrap(), "v1");
    assert_eq!(tactic_var("KATAGO_CUDA_INT8_RMS_FUSION").unwrap(), "0");
    for key in ["KATAGO_CUDA_PADBATCH", "KATAGO_CUDA_INT8_GEMM_TUNE"] {
        assert!(matches!(
            tactic_var(key),
            Err(std::env::VarError::NotPresent)
        ));
    }
    let err = freeze_quantized_tactics().unwrap_err();
    assert!(err.contains("changed after initialization"), "{err}");
    // Readers must see the original snapshot even after the attempted refreeze
    // fails. No environment mutation occurs while these readers are running.
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                for _ in 0..1000 {
                    assert_eq!(tactic_var("KATAGO_CUDA_RMS").unwrap(), "v1");
                    assert_eq!(tactic_var("KATAGO_CUDA_INT8_RMS_FUSION").unwrap(), "0");
                    assert!(matches!(
                        tactic_var("KATAGO_CUDA_PADBATCH"),
                        Err(std::env::VarError::NotPresent)
                    ));
                }
            });
        }
    });
    let err = plan.install().unwrap_err();
    assert!(
        err.contains("after initializing precision recipes"),
        "failed before installation guard: {err}"
    );
    assert_eq!(installed_plan_id(), None);
}

fn plan_first() {
    let plan = PlanFixture::new();
    // This successful public load also proves the fixture used in freeze_first
    // is valid; an invalid plan would not exercise the installation guard.
    plan.install().unwrap();
    assert_eq!(installed_plan_id(), Some(PLAN_ID));
    assert_eq!(tactic_var("KATAGO_CUDA_RMS").unwrap(), "v1");
    let err = freeze_quantized_tactics().unwrap_err();
    assert!(err.contains("require separate processes"), "{err}");
    plan.install().unwrap(); // Failed freeze must not break legacy idempotency.
}

fn concurrent_install_and_freeze() {
    let plan = PlanFixture::new();
    let barrier = Arc::new(Barrier::new(2));
    let (frozen, installed) = std::thread::scope(|scope| {
        let gate = barrier.clone();
        let freeze = scope.spawn(move || {
            gate.wait();
            freeze_quantized_tactics()
        });
        let install = scope.spawn(|| {
            barrier.wait();
            plan.install()
        });
        (freeze.join().unwrap(), install.join().unwrap())
    });
    match (frozen, installed) {
        (Ok(()), Err(err)) => {
            assert!(
                err.contains("after initializing precision recipes"),
                "{err}"
            );
            assert_eq!(installed_plan_id(), None);
        }
        (Err(err), Ok(())) => {
            assert!(err.contains("require separate processes"), "{err}");
            assert_eq!(installed_plan_id(), Some(PLAN_ID));
        }
        result => panic!("exactly one process execution mode must win: {result:?}"),
    }
}

#[test]
fn quantized_tactics_are_frozen_and_legacy_installation_is_exclusive() {
    if let Ok(scenario) = std::env::var(CHILD_SCENARIO) {
        clear_inherited_tactics();
        match scenario.as_str() {
            "freeze-first" => freeze_first(),
            "plan-first" => plan_first(),
            "concurrent" => concurrent_install_and_freeze(),
            other => panic!("unknown child scenario {other}"),
        }
        return;
    }
    for scenario in ["freeze-first", "plan-first", "concurrent"] {
        let result = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", TEST_NAME, "--nocapture", "--test-threads=1"])
            .env(CHILD_SCENARIO, scenario)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{scenario} failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
    }
}
