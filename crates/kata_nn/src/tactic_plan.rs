//! CUDA tactic plan（KataGomo_fork plan JSON 机制的本地化，fail-closed）。
//!
//! plan 是离线 autotune（`scripts/autotune.py`）产出的认证配置：设备指纹 +
//! 模型 SHA-256 + tactic overrides。加载时任何字段与运行环境不匹配即报错，
//! 绝不静默回退（fork `cudatacticplan.cpp` 同款契约）。
//!
//! 接入点：cudabackend `create_compute_context`（配置键 `cudaTacticPlan`，
//! 每模型可 `cudaTacticPlan{i}` 覆盖）。此时 serve 线程尚未 spawn，先于
//! 一切 tactic 读取安装覆盖。
//!
//! 优先级：plan 覆盖 > 环境变量 > 代码默认。诊断开关
//! （`KATAGO_CUDA_PROFILE` 等）不属于 tactic，不进 plan。
//!
//! 多模型约束：全进程只允许安装一份 plan 覆盖；第二次安装必须逐键相同
//! （分析引擎多模型共用同一 plan 是常态），冲突即报错。

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

/// plan `apply.tactic_overrides` 允许的 tactic 键（= 环境变量名）。
/// 出现未知键 = 加载失败（防 plan 与代码版本漂移后静默失配）。
pub const ALLOWED_TACTIC_KEYS: &[&str] = &[
    "KATAGO_CUDA_ATTN",
    "KATAGO_CUDA_ATTN_TILE",
    "KATAGO_CUDA_CUBLASLT",
    "KATAGO_CUDA_CUBLASLT_RANK",
    "KATAGO_CUDA_DUALFFN",
    "KATAGO_CUDA_FUSION",
    "KATAGO_CUDA_NOGRAPH",
    "KATAGO_CUDA_NOPIPELINE",
    "KATAGO_CUDA_PADBATCH",
    "KATAGO_CUDA_RMS",
    "KATAGO_CUDA_SPLITK",
    "KATAGO_CUDA_T32",
    "KATAGO_CUDA_T64N32",
    "KATAGO_NN_BATCH_WINDOW_US",
];

/// 设备指纹（plan target 侧的运行时比对来源）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceFingerprint {
    pub gpu_name: String,
    pub compute_capability: String,
    pub sm_count: u32,
    pub l2_cache_bytes: u64,
}

/// Capabilities compiled into this CUDA backend binary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendCapabilities {
    pub dual_ffn: bool,
    pub attention_q64: bool,
}

/// Device-code and native CUDA wrapper identity embedded by `build.rs`.
/// Schema 2 plans compare this value exactly before installing any tactic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendBuildFingerprint {
    pub kernel_build_id: String,
    pub cuda_compiler: String,
    pub cutlass_version: String,
    pub cutlass_commit: String,
    pub compiled_sm: Vec<String>,
    pub capabilities: BackendCapabilities,
}

#[derive(Debug, Deserialize)]
struct PlanFile {
    schema: u32,
    kind: String,
    plan_id: String,
    #[serde(default)]
    backend_build: Option<BackendBuildFingerprint>,
    target: PlanTarget,
    apply: PlanApply,
}

#[derive(Debug, Deserialize)]
struct PlanTarget {
    #[allow(dead_code)]
    architecture: String,
    gpu_name: String,
    compute_capability: String,
    sm_count: u32,
    l2_cache_bytes: u64,
    model_sha256: String,
    #[allow(dead_code)]
    max_batch_size: Option<i32>,
}

#[derive(Debug, Deserialize)]
struct PlanApply {
    #[serde(default)]
    tactic_overrides: HashMap<String, String>,
}

struct Installed {
    plan_id: String,
    overrides: HashMap<String, String>,
}

static INSTALLED: OnceLock<Installed> = OnceLock::new();

/// 读一个 tactic 配置：plan 覆盖优先，其次环境变量。
/// 语义与 `std::env::var` 一致（未设置返回 `Err`）。
pub fn tactic_var(key: &str) -> Result<String, std::env::VarError> {
    if let Some(inst) = INSTALLED.get() {
        if let Some(v) = inst.overrides.get(key) {
            return Ok(v.clone());
        }
    }
    std::env::var(key)
}

/// Read a boolean tactic using its validated `0|1` value.
///
/// Checking only whether the variable exists makes an explicit `0` enable the
/// tactic, which breaks plan-driven negative overrides.
pub fn tactic_enabled(key: &str) -> bool {
    let value = tactic_var(key).ok();
    bool_value_enabled(value.as_deref())
}

fn bool_value_enabled(value: Option<&str>) -> bool {
    value == Some("1")
}

/// 当前已安装 plan 的 id（日志/诊断用）。
pub fn installed_plan_id() -> Option<&'static str> {
    INSTALLED.get().map(|i| i.plan_id.as_str())
}

/// 校验 tactic 值的取值域（与各读取点的分支保持一致）。
fn validate_value(key: &str, value: &str) -> Result<(), String> {
    let ok = match key {
        // 0/1 开关
        "KATAGO_CUDA_CUBLASLT"
        | "KATAGO_CUDA_DUALFFN"
        | "KATAGO_CUDA_NOGRAPH"
        | "KATAGO_CUDA_NOPIPELINE"
        | "KATAGO_CUDA_PADBATCH"
        | "KATAGO_CUDA_SPLITK"
        | "KATAGO_CUDA_T32"
        | "KATAGO_CUDA_T64N32" => value == "0" || value == "1",
        // 旧路径回退开关
        "KATAGO_CUDA_ATTN" => value == "v3",
        // FA2 query tile: production q128 or the 4-warp q64 candidate.
        "KATAGO_CUDA_ATTN_TILE" => matches!(value, "q64" | "q128"),
        "KATAGO_CUDA_RMS" => value == "v1",
        "KATAGO_CUDA_FUSION" => matches!(value, "none" | "up" | "down" | "all"),
        // cuBLASLt 算法选择策略:heuristic=首选(默认),time=top-N 计时重排
        "KATAGO_CUDA_CUBLASLT_RANK" => matches!(value, "heuristic" | "time"),
        // 微秒窗口
        "KATAGO_NN_BATCH_WINDOW_US" => value
            .parse::<u64>()
            .map(|v| v <= 1_000_000)
            .unwrap_or(false),
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(format!(
            "invalid value '{value}' for tactic key {key} in plan"
        ))
    }
}

/// 加载并安装 plan：全字段校验，任何不匹配即报错（fail-closed）。
///
/// `model_sha256` 为调用方对模型文件实时计算的结果；`device` 为当前
/// CUDA 设备指纹。
pub fn load_and_install(
    path: &Path,
    device: &DeviceFingerprint,
    model_sha256: &str,
    backend_build: &BackendBuildFingerprint,
) -> Result<(), String> {
    let ctx = |m: String| format!("cudaTacticPlan {path:?}: {m}");
    let text = std::fs::read_to_string(path).map_err(|e| ctx(format!("read failed: {e}")))?;
    let plan: PlanFile =
        serde_json::from_str(&text).map_err(|e| ctx(format!("JSON parse failed: {e}")))?;
    if plan.schema != 1 && plan.schema != 2 {
        return Err(ctx(format!("unsupported schema {}", plan.schema)));
    }
    if plan.kind != "cuda-tactic-plan" {
        return Err(ctx(format!("unsupported kind '{}'", plan.kind)));
    }
    for (key, value) in &plan.apply.tactic_overrides {
        if !ALLOWED_TACTIC_KEYS.contains(&key.as_str()) {
            return Err(ctx(format!("unknown tactic key '{key}'")));
        }
        validate_value(key, value).map_err(&ctx)?;
    }
    if plan.schema == 2 {
        let expected = plan
            .backend_build
            .as_ref()
            .ok_or_else(|| ctx("schema 2 requires backend_build".to_string()))?;
        validate_backend_build(expected, backend_build).map_err(&ctx)?;
    }
    validate_required_capabilities(&plan.apply.tactic_overrides, backend_build).map_err(&ctx)?;
    let t = &plan.target;
    if t.gpu_name != device.gpu_name {
        return Err(ctx(format!(
            "gpu name mismatch: plan '{}' vs device '{}'",
            t.gpu_name, device.gpu_name
        )));
    }
    if t.compute_capability != device.compute_capability {
        return Err(ctx(format!(
            "compute capability mismatch: plan '{}' vs device '{}'",
            t.compute_capability, device.compute_capability
        )));
    }
    if t.sm_count != device.sm_count {
        return Err(ctx(format!(
            "SM count mismatch: plan {} vs device {}",
            t.sm_count, device.sm_count
        )));
    }
    if t.l2_cache_bytes != device.l2_cache_bytes {
        return Err(ctx(format!(
            "L2 size mismatch: plan {} vs device {}",
            t.l2_cache_bytes, device.l2_cache_bytes
        )));
    }
    if !t.model_sha256.eq_ignore_ascii_case(model_sha256) {
        return Err(ctx(format!(
            "model sha256 mismatch: plan '{}' vs actual '{}'",
            t.model_sha256, model_sha256
        )));
    }
    install(plan.plan_id.clone(), plan.apply.tactic_overrides).map_err(ctx)
}

fn validate_backend_build(
    expected: &BackendBuildFingerprint,
    actual: &BackendBuildFingerprint,
) -> Result<(), String> {
    macro_rules! exact {
        ($field:ident, $label:literal) => {
            if expected.$field != actual.$field {
                return Err(format!(
                    "backend build {} mismatch: plan '{:?}' vs actual '{:?}'",
                    $label, expected.$field, actual.$field
                ));
            }
        };
    }
    exact!(kernel_build_id, "kernel_build_id");
    exact!(cuda_compiler, "cuda_compiler");
    exact!(cutlass_version, "cutlass_version");
    exact!(cutlass_commit, "cutlass_commit");
    exact!(compiled_sm, "compiled_sm");
    exact!(capabilities, "capabilities");
    Ok(())
}

fn validate_required_capabilities(
    overrides: &HashMap<String, String>,
    backend_build: &BackendBuildFingerprint,
) -> Result<(), String> {
    if overrides.get("KATAGO_CUDA_DUALFFN").map(String::as_str) == Some("1")
        && !backend_build.capabilities.dual_ffn
    {
        return Err(
            "plan requests KATAGO_CUDA_DUALFFN=1 but backend capability dual_ffn=false".to_string(),
        );
    }
    if overrides.get("KATAGO_CUDA_ATTN_TILE").map(String::as_str) == Some("q64")
        && !backend_build.capabilities.attention_q64
    {
        return Err(
            "plan requests KATAGO_CUDA_ATTN_TILE=q64 but backend capability attention_q64=false"
                .to_string(),
        );
    }
    Ok(())
}

/// 安装覆盖（幂等：与已安装内容逐键相同则成功，冲突则报错）。
fn install(plan_id: String, overrides: HashMap<String, String>) -> Result<(), String> {
    if INSTALLED
        .set(Installed {
            plan_id: plan_id.clone(),
            overrides: overrides.clone(),
        })
        .is_ok()
    {
        return Ok(());
    }
    // 已有安装：plan_id 与覆盖逐键相同才放行（多模型共用同一 plan）。
    let existing = INSTALLED.get().unwrap();
    if existing.plan_id == plan_id && same_overrides(&existing.overrides, &overrides) {
        Ok(())
    } else {
        Err(format!(
            "conflicting tactic plan '{plan_id}': plan '{}' already installed",
            existing.plan_id
        ))
    }
}

fn same_overrides(a: &HashMap<String, String>, b: &HashMap<String, String>) -> bool {
    a.len() == b.len() && a.iter().all(|(k, v)| b.get(k) == Some(v))
}

/// 计算文件 SHA-256（plan 校验与 `cuda-fingerprint` 子命令共用）。
pub fn sha256_file(path: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    use std::io::Read;
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("read {path:?}: {e}"))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boolean_tactic_requires_one() {
        assert!(bool_value_enabled(Some("1")));
        assert!(!bool_value_enabled(Some("0")));
        assert!(!bool_value_enabled(None));
    }

    fn device() -> DeviceFingerprint {
        DeviceFingerprint {
            gpu_name: "NVIDIA GeForce RTX 5070 Ti".into(),
            compute_capability: "12.0".into(),
            sm_count: 96,
            l2_cache_bytes: 50331648,
        }
    }

    fn backend_build() -> BackendBuildFingerprint {
        BackendBuildFingerprint {
            kernel_build_id: format!("sha256:{}", "11".repeat(32)),
            cuda_compiler: "13.3.73".into(),
            cutlass_version: "3.9.2".into(),
            cutlass_commit: "ad7b2f5".into(),
            compiled_sm: vec!["120".into(), "89".into()],
            capabilities: BackendCapabilities {
                dual_ffn: true,
                attention_q64: false,
            },
        }
    }

    fn plan_json(overrides: &str) -> String {
        format!(
            r#"{{
              "schema": 1,
              "kind": "cuda-tactic-plan",
              "plan_id": "test-plan",
              "target": {{
                "architecture": "sm_120",
                "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
                "compute_capability": "12.0",
                "sm_count": 96,
                "l2_cache_bytes": 50331648,
                "model_sha256": "{MODEL}",
                "max_batch_size": 16
              }},
              "apply": {{ "tactic_overrides": {overrides} }}
            }}"#,
            MODEL = "aa".repeat(32),
            overrides = overrides
        )
    }

    fn plan_json_v2(overrides: &str, build: &BackendBuildFingerprint) -> String {
        let mut value: serde_json::Value = serde_json::from_str(&plan_json(overrides)).unwrap();
        value["schema"] = serde_json::json!(2);
        value["backend_build"] = serde_json::to_value(build).unwrap();
        serde_json::to_string_pretty(&value).unwrap()
    }

    fn write_plan(dir: &std::path::Path, name: &str, body: &str) -> std::path::PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    #[test]
    fn rejects_unknown_tactic_key() {
        let dir = std::env::temp_dir().join("kata_tactic_plan_test_1");
        std::fs::create_dir_all(&dir).unwrap();
        let p = write_plan(
            &dir,
            "a.json",
            &plan_json(r#"{ "KATAGO_CUDA_BOGUS": "1" }"#),
        );
        let err = load_and_install(&p, &device(), &"aa".repeat(32), &backend_build()).unwrap_err();
        assert!(err.contains("unknown tactic key"), "{err}");
    }

    #[test]
    fn rejects_bad_value() {
        let dir = std::env::temp_dir().join("kata_tactic_plan_test_2");
        std::fs::create_dir_all(&dir).unwrap();
        let p = write_plan(
            &dir,
            "a.json",
            &plan_json(r#"{ "KATAGO_CUDA_FUSION": "bogus" }"#),
        );
        let err = load_and_install(&p, &device(), &"aa".repeat(32), &backend_build()).unwrap_err();
        assert!(err.contains("invalid value"), "{err}");
    }

    #[test]
    fn rejects_device_mismatch() {
        let dir = std::env::temp_dir().join("kata_tactic_plan_test_3");
        std::fs::create_dir_all(&dir).unwrap();
        let p = write_plan(&dir, "a.json", &plan_json("{}"));
        let mut dev = device();
        dev.sm_count = 84;
        let err = load_and_install(&p, &dev, &"aa".repeat(32), &backend_build()).unwrap_err();
        assert!(err.contains("SM count mismatch"), "{err}");
    }

    #[test]
    fn rejects_model_mismatch() {
        let dir = std::env::temp_dir().join("kata_tactic_plan_test_4");
        std::fs::create_dir_all(&dir).unwrap();
        let p = write_plan(&dir, "a.json", &plan_json("{}"));
        let err = load_and_install(&p, &device(), &"bb".repeat(32), &backend_build()).unwrap_err();
        assert!(err.contains("model sha256 mismatch"), "{err}");
    }

    #[test]
    fn schema2_requires_backend_build() {
        let dir = std::env::temp_dir().join("kata_tactic_plan_test_schema2_missing");
        std::fs::create_dir_all(&dir).unwrap();
        let body = plan_json("{}").replacen("\"schema\": 1", "\"schema\": 2", 1);
        let p = write_plan(&dir, "missing.json", &body);
        let err = load_and_install(&p, &device(), &"aa".repeat(32), &backend_build()).unwrap_err();
        assert!(err.contains("schema 2 requires backend_build"), "{err}");
    }

    #[test]
    fn schema2_rejects_backend_build_mismatches() {
        let dir = std::env::temp_dir().join("kata_tactic_plan_test_schema2_build");
        std::fs::create_dir_all(&dir).unwrap();
        let expected = backend_build();
        for (name, actual, field) in [
            (
                "build-id",
                BackendBuildFingerprint {
                    kernel_build_id: format!("sha256:{}", "22".repeat(32)),
                    ..expected.clone()
                },
                "kernel_build_id",
            ),
            (
                "cutlass-version",
                BackendBuildFingerprint {
                    cutlass_version: "4.2.2".into(),
                    ..expected.clone()
                },
                "cutlass_version",
            ),
            (
                "capabilities",
                BackendBuildFingerprint {
                    capabilities: BackendCapabilities {
                        dual_ffn: false,
                        ..expected.capabilities.clone()
                    },
                    ..expected.clone()
                },
                "capabilities",
            ),
        ] {
            let p = write_plan(
                &dir,
                &format!("{name}.json"),
                &plan_json_v2("{}", &expected),
            );
            let err = load_and_install(&p, &device(), &"aa".repeat(32), &actual).unwrap_err();
            assert!(err.contains(field), "{name}: {err}");
        }
    }

    #[test]
    fn rejects_requested_missing_capability() {
        let dir = std::env::temp_dir().join("kata_tactic_plan_test_capability");
        std::fs::create_dir_all(&dir).unwrap();
        let mut actual = backend_build();
        actual.capabilities.dual_ffn = false;
        let p = write_plan(
            &dir,
            "dual.json",
            &plan_json(r#"{ "KATAGO_CUDA_DUALFFN": "1" }"#),
        );
        let err = load_and_install(&p, &device(), &"aa".repeat(32), &actual).unwrap_err();
        assert!(err.contains("capability dual_ffn=false"), "{err}");
    }

    #[test]
    fn rejects_q64_when_backend_did_not_compile_it() {
        let dir = std::env::temp_dir().join("kata_tactic_plan_test_q64_capability");
        std::fs::create_dir_all(&dir).unwrap();
        let p = write_plan(
            &dir,
            "q64.json",
            &plan_json(r#"{ "KATAGO_CUDA_ATTN_TILE": "q64" }"#),
        );
        let err = load_and_install(&p, &device(), &"aa".repeat(32), &backend_build()).unwrap_err();
        assert!(err.contains("capability attention_q64=false"), "{err}");
    }

    // 注意：成功路径会写入进程级 OnceLock。cargo 默认并行跑测试，凡触及
    // install 的路径必须收敛到单个用例里顺序执行（拒绝类用例都在 install
    // 之前失败，互不干扰）。
    #[test]
    fn installs_idempotent_and_conflict_detection() {
        let dir = std::env::temp_dir().join("kata_tactic_plan_test_5");
        std::fs::create_dir_all(&dir).unwrap();
        let body = plan_json(
            r#"{ "KATAGO_CUDA_FUSION": "all", "KATAGO_CUDA_CUBLASLT": "1", "KATAGO_CUDA_PADBATCH": "0" }"#,
        );
        let p = write_plan(&dir, "ok.json", &body);
        load_and_install(&p, &device(), &"aa".repeat(32), &backend_build()).unwrap();
        // 同 plan 重装（多模型场景）幂等成功。
        load_and_install(&p, &device(), &"aa".repeat(32), &backend_build()).unwrap();
        assert_eq!(tactic_var("KATAGO_CUDA_FUSION").unwrap(), "all");
        assert!(!tactic_enabled("KATAGO_CUDA_PADBATCH"));
        // plan_id 相同但覆盖不同 → 冲突拒绝。
        let p2 = write_plan(
            &dir,
            "conflict.json",
            &plan_json(r#"{ "KATAGO_CUDA_FUSION": "none" }"#),
        );
        let err = load_and_install(&p2, &device(), &"aa".repeat(32), &backend_build()).unwrap_err();
        assert!(err.contains("conflicting tactic plan"), "{err}");
    }
}
