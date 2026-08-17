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

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

/// plan `apply.tactic_overrides` 允许的 tactic 键（= 环境变量名）。
/// 出现未知键 = 加载失败（防 plan 与代码版本漂移后静默失配）。
pub const ALLOWED_TACTIC_KEYS: &[&str] = &[
    "KATAGO_CUDA_ATTN",
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

#[derive(Debug, Deserialize)]
struct PlanFile {
    schema: u32,
    kind: String,
    plan_id: String,
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
        "KATAGO_CUDA_RMS" => value == "v1",
        "KATAGO_CUDA_FUSION" => matches!(value, "none" | "up" | "down" | "all"),
        // cuBLASLt 算法选择策略:heuristic=首选(默认),time=top-N 计时重排
        "KATAGO_CUDA_CUBLASLT_RANK" => matches!(value, "heuristic" | "time"),
        // 微秒窗口
        "KATAGO_NN_BATCH_WINDOW_US" => {
            value.parse::<u64>().map(|v| v <= 1_000_000).unwrap_or(false)
        }
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(format!("invalid value '{value}' for tactic key {key} in plan"))
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
) -> Result<(), String> {
    let ctx = |m: String| format!("cudaTacticPlan {path:?}: {m}");
    let text = std::fs::read_to_string(path).map_err(|e| ctx(format!("read failed: {e}")))?;
    let plan: PlanFile =
        serde_json::from_str(&text).map_err(|e| ctx(format!("JSON parse failed: {e}")))?;
    if plan.schema != 1 {
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
        let n = file.read(&mut buf).map_err(|e| format!("read {path:?}: {e}"))?;
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

    fn device() -> DeviceFingerprint {
        DeviceFingerprint {
            gpu_name: "NVIDIA GeForce RTX 5070 Ti".into(),
            compute_capability: "12.0".into(),
            sm_count: 96,
            l2_cache_bytes: 50331648,
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

    fn write_plan(dir: &std::path::Path, name: &str, body: &str) -> std::path::PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    #[test]
    fn rejects_unknown_tactic_key() {
        let dir = std::env::temp_dir().join("kata_tactic_plan_test_1");
        std::fs::create_dir_all(&dir).unwrap();
        let p = write_plan(&dir, "a.json", &plan_json(r#"{ "KATAGO_CUDA_BOGUS": "1" }"#));
        let err = load_and_install(&p, &device(), &"aa".repeat(32)).unwrap_err();
        assert!(err.contains("unknown tactic key"), "{err}");
    }

    #[test]
    fn rejects_bad_value() {
        let dir = std::env::temp_dir().join("kata_tactic_plan_test_2");
        std::fs::create_dir_all(&dir).unwrap();
        let p = write_plan(&dir, "a.json", &plan_json(r#"{ "KATAGO_CUDA_FUSION": "bogus" }"#));
        let err = load_and_install(&p, &device(), &"aa".repeat(32)).unwrap_err();
        assert!(err.contains("invalid value"), "{err}");
    }

    #[test]
    fn rejects_device_mismatch() {
        let dir = std::env::temp_dir().join("kata_tactic_plan_test_3");
        std::fs::create_dir_all(&dir).unwrap();
        let p = write_plan(&dir, "a.json", &plan_json("{}"));
        let mut dev = device();
        dev.sm_count = 84;
        let err = load_and_install(&p, &dev, &"aa".repeat(32)).unwrap_err();
        assert!(err.contains("SM count mismatch"), "{err}");
    }

    #[test]
    fn rejects_model_mismatch() {
        let dir = std::env::temp_dir().join("kata_tactic_plan_test_4");
        std::fs::create_dir_all(&dir).unwrap();
        let p = write_plan(&dir, "a.json", &plan_json("{}"));
        let err = load_and_install(&p, &device(), &"bb".repeat(32)).unwrap_err();
        assert!(err.contains("model sha256 mismatch"), "{err}");
    }

    // 注意：成功路径会写入进程级 OnceLock。cargo 默认并行跑测试，凡触及
    // install 的路径必须收敛到单个用例里顺序执行（拒绝类用例都在 install
    // 之前失败，互不干扰）。
    #[test]
    fn installs_idempotent_and_conflict_detection() {
        let dir = std::env::temp_dir().join("kata_tactic_plan_test_5");
        std::fs::create_dir_all(&dir).unwrap();
        let body = plan_json(r#"{ "KATAGO_CUDA_FUSION": "all", "KATAGO_CUDA_CUBLASLT": "1" }"#);
        let p = write_plan(&dir, "ok.json", &body);
        load_and_install(&p, &device(), &"aa".repeat(32)).unwrap();
        // 同 plan 重装（多模型场景）幂等成功。
        load_and_install(&p, &device(), &"aa".repeat(32)).unwrap();
        assert_eq!(tactic_var("KATAGO_CUDA_FUSION").unwrap(), "all");
        // plan_id 相同但覆盖不同 → 冲突拒绝。
        let p2 = write_plan(&dir, "conflict.json", &plan_json(r#"{ "KATAGO_CUDA_FUSION": "none" }"#));
        let err = load_and_install(&p2, &device(), &"aa".repeat(32)).unwrap_err();
        assert!(err.contains("conflicting tactic plan"), "{err}");
    }
}
