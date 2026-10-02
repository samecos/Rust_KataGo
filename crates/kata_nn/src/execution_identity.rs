//! Immutable loaded CUDA execution-policy identity. This is a routing key,
//! not a claim that every batch used a particular kernel, graph, or precision.
//! Shape-dependent deterministic selection and existing fallback rules remain
//! part of the bound executable. Timing-selected algorithms are not certified.

use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct LegacyCudaExecution {
    pub model_sha256: String,
    pub executable_sha256: String,
    pub backend_mode: String,
    pub backend_build: serde_json::Value,
    pub device: serde_json::Value,
    pub driver_version: i32,
    pub kernel_target: String,
    pub installed_plan_id: Option<String>,
    pub tactics: BTreeMap<String, Option<String>>,
    pub execution_config: BTreeMap<String, String>,
    pub loaded_execution: serde_json::Value,
}

impl LegacyCudaExecution {
    pub(crate) fn profile_id(&self) -> Result<String, String> {
        for (name, value) in [
            ("model", &self.model_sha256),
            ("executable", &self.executable_sha256),
        ] {
            if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(format!("invalid {name} SHA256 for loaded CUDA identity"));
            }
        }
        if !matches!(self.backend_mode.as_str(), "cuda-fp16" | "cuda-int8") {
            return Err("unsupported legacy CUDA execution mode".into());
        }
        if self.driver_version <= 0
            || self.kernel_target.is_empty()
            || !self.backend_build.is_object()
            || !self.device.is_object()
            || !self.loaded_execution.is_object()
            || self.execution_config.is_empty()
        {
            return Err("incomplete loaded CUDA execution identity".into());
        }
        if timing_selected(&self.tactics) {
            return Err("timing-selected CUDA algorithms have no fixed loaded identity".into());
        }
        use sha2::{Digest, Sha256};
        let bytes = serde_json::to_vec(&serde_json::json!({
            "schema": "rustgo-loaded-cuda-execution-v1", "execution": self,
        }))
        .map_err(|e| e.to_string())?;
        Ok(format!(
            "rustgo-cuda-exec-v1:{}",
            hex::encode(Sha256::digest(bytes))
        ))
    }
}

pub(crate) fn timing_selected(tactics: &BTreeMap<String, Option<String>>) -> bool {
    tactics
        .get("KATAGO_CUDA_CUBLASLT_RANK")
        .and_then(Option::as_deref)
        == Some("time")
        || tactics
            .get("KATAGO_CUDA_INT8_GEMM_TUNE")
            .and_then(Option::as_deref)
            == Some("1")
}

/// A previously advertised identity cannot be silently removed or rebound.
pub(crate) fn require_unchanged(
    previous: Option<&str>,
    current: Option<&str>,
) -> Result<(), String> {
    if previous.is_some() && previous != current {
        return Err("loaded execution identity changed; create a new evaluator/process".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> LegacyCudaExecution {
        LegacyCudaExecution {
            model_sha256: "a".repeat(64),
            executable_sha256: "b".repeat(64),
            backend_mode: "cuda-fp16".into(),
            backend_build: serde_json::json!({"kernel":"fixture"}),
            device: serde_json::json!({"uuid":"fixture","cc":"12.0"}),
            driver_version: 13010,
            kernel_target: "fixture-sm120".into(),
            installed_plan_id: Some("fixture-plan".into()),
            tactics: BTreeMap::from([(
                "KATAGO_CUDA_CUBLASLT_RANK".into(),
                Some("heuristic".into()),
            )]),
            execution_config: BTreeMap::from([("max_batch_size".into(), "8".into())]),
            loaded_execution: serde_json::json!({"scope":null,"int8_ffn_count":0,"layout":"tn"}),
        }
    }
    #[test]
    fn execution_identity_binds_actual_inputs_and_not_file_locations() {
        let original = fixture();
        let expected = original.profile_id().unwrap();
        assert!(expected.starts_with("rustgo-cuda-exec-v1:"));
        let changes: Vec<Box<dyn Fn(&mut LegacyCudaExecution)>> = vec![
            Box::new(|x| x.model_sha256 = "c".repeat(64)),
            Box::new(|x| x.executable_sha256 = "d".repeat(64)),
            Box::new(|x| x.backend_mode = "cuda-int8".into()),
            Box::new(|x| x.backend_build["kernel"] = "another-build".into()),
            Box::new(|x| x.device["uuid"] = "another-device".into()),
            Box::new(|x| x.driver_version += 1),
            Box::new(|x| x.kernel_target = "another-target".into()),
            Box::new(|x| x.installed_plan_id = None),
            Box::new(|x| {
                x.tactics
                    .insert("KATAGO_CUDA_ATTN_TILE".into(), Some("q64".into()));
            }),
            Box::new(|x| {
                x.execution_config
                    .insert("max_batch_size".into(), "3".into());
            }),
            Box::new(|x| x.loaded_execution["int8_ffn_count"] = 33.into()),
        ];
        for change in changes {
            let mut x = original.clone();
            change(&mut x);
            assert_ne!(x.profile_id().unwrap(), expected);
        }
        // Paths are deliberately absent; identical loaded inputs have one key.
        assert_eq!(original.clone().profile_id().unwrap(), expected);
    }
    #[test]
    fn execution_identity_preserves_absent_versus_explicit_zero() {
        let mut x = fixture();
        x.tactics.insert("KATAGO_CUDA_INT8_GEMM_TUNE".into(), None);
        let absent = x.profile_id().unwrap();
        x.tactics
            .insert("KATAGO_CUDA_INT8_GEMM_TUNE".into(), Some("0".into()));
        assert_ne!(absent, x.profile_id().unwrap());
        x.tactics
            .insert("KATAGO_CUDA_INT8_GEMM_TUNE".into(), Some("1".into()));
        assert!(x.profile_id().is_err());
        let mut x = fixture();
        x.tactics
            .insert("KATAGO_CUDA_CUBLASLT_RANK".into(), Some("time".into()));
        assert!(x.profile_id().is_err());
    }
    #[test]
    fn execution_identity_rejects_missing_facts_and_rebinding() {
        let mut x = fixture();
        x.model_sha256.clear();
        assert!(x.profile_id().is_err());
        let mut x = fixture();
        x.backend_mode = "onnx-unknown".into();
        assert!(x.profile_id().is_err());
        let mut x = fixture();
        x.driver_version = 0;
        assert!(x.profile_id().is_err());
        assert!(require_unchanged(None, None).is_ok());
        assert!(require_unchanged(None, Some("loaded")).is_ok());
        assert!(require_unchanged(Some("loaded"), Some("loaded")).is_ok());
        assert!(require_unchanged(Some("loaded"), None).is_err());
        assert!(require_unchanged(Some("loaded"), Some("other")).is_err());
    }
}
