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
    "KATAGO_CUDA_GEMM_LAYOUT",
    "KATAGO_CUDA_NOGRAPH",
    "KATAGO_CUDA_NOPIPELINE",
    "KATAGO_CUDA_PADBATCH",
    "KATAGO_CUDA_RESIDUAL_ALGO",
    "KATAGO_CUDA_RMS",
    "KATAGO_CUDA_SPLITK",
    "KATAGO_CUDA_STEM_MAP3D",
    "KATAGO_CUDA_GATE_ROWQUAD_R1",
    "KATAGO_CUDA_QKV_IMMUTABLE_R1",
    "KATAGO_CUDA_QKV_CLASSIC_N128_R1",
    "KATAGO_CUDA_OUTPROJ_CLASSIC_N128_R1",
    "KATAGO_CUDA_FFN_COMPACT_R1",
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
    #[serde(default)]
    pub attention_q64_serial: bool,
}

/// Host-side tactic contract, independent of the CUDA device-code build hash.
/// Revision 1 installs the plan before preparing GEMM layout-specific weights.
/// Revision 2 adds native variable-width/pruned TF3 execution and shape admission.
/// Bump this when a host-side change invalidates certification of these tactics.
pub const CUDA_HOST_TACTIC_REVISION: u32 = 2;

/// Host FP32-to-FP16 encoding contract, independent of tactics and device code.
/// Revision 1 correctly rounds half subnormals to nearest, ties to even.
/// Every CUDA plan must bind this revision after numerical revalidation.
pub const CUDA_FP16_ENCODING_REVISION: u32 = 1;

/// CUDA build identity and independently versioned host contracts.
/// Both plan schemas require a matching FP16 encoding revision. Schema 2 also
/// compares CUDA build fields exactly. The host tactic revision remains optional
/// only for plans that do not control a tactic requiring that separate contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendBuildFingerprint {
    pub kernel_build_id: String,
    pub cuda_compiler: String,
    pub cutlass_version: String,
    pub cutlass_commit: String,
    pub compiled_sm: Vec<String>,
    pub capabilities: BackendCapabilities,
    /// Optional for parsing old archives, mandatory when installing any plan.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fp16_encoding_revision: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_tactic_revision: Option<u32>,
    /// Runtime library identity, required by version-specific Lt presets.
    /// Its omission is independent of the mandatory FP16 encoding contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cublaslt_version: Option<u64>,
    /// Independently hashed AOT code and ABI. Legacy plans may omit this only
    /// while leaving the opt-in strict attention path disabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict_attention_artifact: Option<String>,
    /// Exact QKV CUBIN, parameter image, and host packed-attention integration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qkv_immutable_artifact: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ffn_compact_artifact: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outproj_n128_artifact: Option<String>,
}

pub(crate) const STRICT_ATTENTION_KEY: &str = "KATAGO_CUDA_ATTN";
pub(crate) const STRICT_ATTENTION_PRESET: &str = "fa4-strict-b14-r1";

pub(crate) const QKV_IMMUTABLE_KEY:&str="KATAGO_CUDA_QKV_IMMUTABLE_R1";
fn qkv_immutable_value(value:Option<&str>)->Result<bool,String>{match value{None|Some("0")=>Ok(false),Some("1")=>Ok(true),Some(v)=>Err(format!("invalid value '{v}' for {QKV_IMMUTABLE_KEY}"))}}
fn validate_qkv_immutable_options(requested:bool,strict:bool,layout:Option<&str>,installed:Option<&HashMap<String,String>>)->Result<(),String>{
    if !requested{return Ok(());}
    if !strict{return Err(format!("{QKV_IMMUTABLE_KEY}=1 requires {STRICT_ATTENTION_KEY}={STRICT_ATTENTION_PRESET}"));}
    if !matches!(layout,None|Some("tn")){return Err(format!("{QKV_IMMUTABLE_KEY}=1 requires TN QKV weights"));}
    if let Some(o)=installed{for(k,v)in[(QKV_IMMUTABLE_KEY,"1"),("KATAGO_CUDA_GEMM_LAYOUT","tn")]{if o.get(k).map(String::as_str)!=Some(v){return Err(format!("{QKV_IMMUTABLE_KEY}=1 requires {k}={v} explicitly bound by the installed plan"));}}}
    Ok(())
}
pub(crate) const QKV_N128_KEY:&str="KATAGO_CUDA_QKV_CLASSIC_N128_R1";
fn qkv_n128_value(value:Option<&str>)->Result<bool,String>{match value{None|Some("0")=>Ok(false),Some("1")=>Ok(true),Some(v)=>Err(format!("invalid value '{v}' for {QKV_N128_KEY}"))}}
fn validate_qkv_n128_options(requested:bool,qkv:bool,installed:Option<&HashMap<String,String>>)->Result<(),String>{
    if !requested{return Ok(());}
    if !qkv{return Err(format!("{QKV_N128_KEY}=1 requires {QKV_IMMUTABLE_KEY}=1"));}
    if let Some(o)=installed{if o.get(QKV_N128_KEY).map(String::as_str)!=Some("1"){return Err(format!("{QKV_N128_KEY}=1 must be explicitly bound by the installed plan"));}}
    Ok(())
}
pub(crate) fn qkv_n128_requested()->Result<bool,String>{
    let enabled=qkv_n128_value(tactic_var(QKV_N128_KEY).ok().as_deref())?;
    validate_qkv_n128_options(enabled,qkv_immutable_value(tactic_var(QKV_IMMUTABLE_KEY).ok().as_deref())?,INSTALLED.get().map(|i|&i.overrides))?;
    Ok(enabled)
}
pub(crate) fn qkv_immutable_requested()->Result<bool,String>{
    let enabled=qkv_immutable_value(tactic_var(QKV_IMMUTABLE_KEY).ok().as_deref())?;
    qkv_n128_requested()?;
    validate_qkv_immutable_options(enabled,strict_attention_value(tactic_var(STRICT_ATTENTION_KEY).ok().as_deref())?,tactic_var("KATAGO_CUDA_GEMM_LAYOUT").ok().as_deref(),INSTALLED.get().map(|i|&i.overrides))?;
    if enabled{
        #[cfg(feature="cuda")]let artifact=crate::backends::cuda::qkv_immutable::fingerprint();
        #[cfg(not(feature="cuda"))]let artifact:Option<String>=None;
        if artifact.is_none(){return Err(format!("{QKV_IMMUTABLE_KEY}=1 requires a valid compiled qkv_immutable_artifact"));}
    }
    Ok(enabled)
}

pub(crate) fn validate_strict_attention_target(device: &DeviceFingerprint) -> Result<(), String> {
    if device.gpu_name != "NVIDIA GeForce RTX 5070 Ti"
        || device.compute_capability != "12.0"
        || device.sm_count != 70
        || device.l2_cache_bytes != 50331648
    {
        return Err(format!(
            "{STRICT_ATTENTION_PRESET} requires RTX 5070 Ti SM120 (70 SM, 50331648-byte L2), actual {device:?}"
        ));
    }
    Ok(())
}

fn strict_attention_value(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("fa2" | "v3") => Ok(false),
        Some(STRICT_ATTENTION_PRESET) => Ok(true),
        Some(value) => Err(format!(
            "invalid value '{value}' for {STRICT_ATTENTION_KEY}"
        )),
    }
}

fn validate_strict_attention_options(
    requested: bool,
    tile: Option<&str>,
    nograph: Option<&str>,
    installed_overrides: Option<&HashMap<String, String>>,
) -> Result<(), String> {
    if !requested {
        return Ok(());
    }
    if let Some(overrides) = installed_overrides {
        for (key, expected) in [
            (STRICT_ATTENTION_KEY, STRICT_ATTENTION_PRESET),
            ("KATAGO_CUDA_ATTN_TILE", "q64-serial"),
            ("KATAGO_CUDA_NOGRAPH", "1"),
        ] {
            if overrides.get(key).map(String::as_str) != Some(expected) {
                return Err(format!(
                    "{STRICT_ATTENTION_PRESET} requires {key}={expected} explicitly bound by the installed plan; environment variables cannot extend its certification"
                ));
            }
        }
    }
    if tile != Some("q64-serial") || nograph != Some("1") {
        return Err(format!(
            "{STRICT_ATTENTION_PRESET} requires KATAGO_CUDA_ATTN_TILE=q64-serial and KATAGO_CUDA_NOGRAPH=1"
        ));
    }
    Ok(())
}

/// Resolve the opt-in path with normal plan > environment precedence. A plan
/// must bind the candidate, fallback and graph policy together.
pub(crate) fn strict_attention_requested() -> Result<bool, String> {
    let requested = strict_attention_value(tactic_var(STRICT_ATTENTION_KEY).ok().as_deref())?;
    validate_strict_attention_options(
        requested,
        tactic_var("KATAGO_CUDA_ATTN_TILE").ok().as_deref(),
        tactic_var("KATAGO_CUDA_NOGRAPH").ok().as_deref(),
        INSTALLED.get().map(|i| &i.overrides),
    )?;
    if requested {
        #[cfg(feature = "cuda")]
        let artifact = crate::backends::cuda::strict_attention::fingerprint();
        #[cfg(not(feature = "cuda"))]
        let artifact: Option<String> = None;
        if artifact.is_none() {
            return Err(format!(
                "{STRICT_ATTENTION_PRESET} requires a valid compiled strict_attention_artifact"
            ));
        }
    }
    Ok(requested)
}

pub(crate) const STEM_MAP3D_KEY: &str = "KATAGO_CUDA_STEM_MAP3D";

fn stem_map3d_value(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(value) => Err(format!("invalid value '{value}' for {STEM_MAP3D_KEY}")),
    }
}

fn validate_stem_map3d_binding(
    requested: bool,
    installed_overrides: Option<&HashMap<String, String>>,
) -> Result<(), String> {
    if requested
        && let Some(overrides) = installed_overrides
        && overrides.get(STEM_MAP3D_KEY).map(String::as_str) != Some("1")
    {
        return Err(format!(
            "{STEM_MAP3D_KEY}=1 must be explicitly bound by the installed plan; environment variables cannot extend certification"
        ));
    }
    Ok(())
}

/// No plan: explicit environment diagnostics are allowed. With a plan, the
/// candidate must be bound; normal plan > environment precedence is retained.
pub(crate) fn stem_map3d_requested() -> Result<bool, String> {
    let requested = stem_map3d_value(tactic_var(STEM_MAP3D_KEY).ok().as_deref())?;
    validate_stem_map3d_binding(requested, INSTALLED.get().map(|i| &i.overrides))?;
    Ok(requested)
}

pub(crate) const GATE_ROWQUAD_KEY: &str = "KATAGO_CUDA_GATE_ROWQUAD_R1";

fn gate_rowquad_value(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(value) => Err(format!("invalid value '{value}' for {GATE_ROWQUAD_KEY}")),
    }
}

fn validate_gate_rowquad_binding(
    requested: bool,
    installed_overrides: Option<&HashMap<String, String>>,
) -> Result<(), String> {
    if requested
        && let Some(overrides) = installed_overrides
        && overrides.get(GATE_ROWQUAD_KEY).map(String::as_str) != Some("1")
    {
        return Err(format!(
            "{GATE_ROWQUAD_KEY}=1 must be explicitly bound by the installed plan; environment variables cannot extend certification"
        ));
    }
    Ok(())
}

/// No plan: explicit environment diagnostics are allowed. With a plan, the
/// candidate must be bound; normal plan > environment precedence is retained.
pub(crate) fn gate_rowquad_requested() -> Result<bool, String> {
    let requested = gate_rowquad_value(tactic_var(GATE_ROWQUAD_KEY).ok().as_deref())?;
    validate_gate_rowquad_binding(requested, INSTALLED.get().map(|i| &i.overrides))?;
    Ok(requested)
}

pub(crate) const RESIDUAL_ALGO_KEY: &str = "KATAGO_CUDA_RESIDUAL_ALGO";
pub(crate) const TF3_RESIDUAL_PRESET: &str = "tf3_5070ti_r1";
pub(crate) const TF3_RESIDUAL_CUBLASLT_VERSION: u64 = 130600;

/// This preset was measured only on this GPU and library. A matching heuristic
/// index on a different target is not evidence of the same algorithm.
pub(crate) fn validate_tf3_residual_target(
    device: &DeviceFingerprint,
    cublaslt_version: Option<u64>,
) -> Result<(), String> {
    if cublaslt_version != Some(TF3_RESIDUAL_CUBLASLT_VERSION) {
        return Err(format!(
            "{RESIDUAL_ALGO_KEY}={TF3_RESIDUAL_PRESET} requires cuBLASLt {TF3_RESIDUAL_CUBLASLT_VERSION}, actual {cublaslt_version:?}"
        ));
    }
    if device.gpu_name != "NVIDIA GeForce RTX 5070 Ti"
        || device.compute_capability != "12.0"
        || device.sm_count != 70
        || device.l2_cache_bytes != 50331648
    {
        return Err(format!(
            "{RESIDUAL_ALGO_KEY}={TF3_RESIDUAL_PRESET} requires RTX 5070 Ti SM120 (70 SM, 50331648-byte L2), actual {device:?}"
        ));
    }
    Ok(())
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

fn residual_preset_value(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("heuristic") => Ok(false),
        Some(TF3_RESIDUAL_PRESET) => Ok(true),
        Some(value) => Err(format!("invalid value '{value}' for {RESIDUAL_ALGO_KEY}")),
    }
}

fn validate_residual_plan_override(
    requested: bool,
    installed_overrides: Option<&HashMap<String, String>>,
) -> Result<(), String> {
    if requested
        && let Some(overrides) = installed_overrides
        && overrides.get(RESIDUAL_ALGO_KEY).map(String::as_str) != Some(TF3_RESIDUAL_PRESET)
    {
        return Err(format!(
            "{RESIDUAL_ALGO_KEY}={TF3_RESIDUAL_PRESET} must be explicitly bound by the installed plan; an environment override cannot add an unbound library-specific preset"
        ));
    }
    Ok(())
}

/// Keep normal plan > environment precedence, but do not let an old plan's
/// omission smuggle an unversioned new preset in through the environment.
pub(crate) fn residual_preset_requested() -> Result<bool, String> {
    let value = tactic_var(RESIDUAL_ALGO_KEY).ok();
    let requested = residual_preset_value(value.as_deref())?;
    validate_residual_plan_override(requested, INSTALLED.get().map(|i| &i.overrides))?;
    if requested && tactic_var("KATAGO_CUDA_CUBLASLT").as_deref() == Ok("0") {
        return Err(format!(
            "{RESIDUAL_ALGO_KEY}={TF3_RESIDUAL_PRESET} requires KATAGO_CUDA_CUBLASLT enabled"
        ));
    }
    Ok(requested)
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
        | STEM_MAP3D_KEY
        | GATE_ROWQUAD_KEY
        | QKV_IMMUTABLE_KEY
        | QKV_N128_KEY
        | OUTPROJ_N128_KEY
        | FFN_COMPACT_KEY
        | "KATAGO_CUDA_T32"
        | "KATAGO_CUDA_T64N32" => value == "0" || value == "1",
        // 旧路径回退开关
        "KATAGO_CUDA_ATTN" => matches!(value, "fa2" | "v3" | STRICT_ATTENTION_PRESET),
        // q64-serial retains q64's layout and uses q128's reduction order.
        "KATAGO_CUDA_ATTN_TILE" => matches!(value, "q64" | "q128" | "q64-serial"),
        "KATAGO_CUDA_RMS" => value == "v1",
        "KATAGO_CUDA_FUSION" => matches!(value, "none" | "up" | "down" | "all"),
        // cuBLASLt 算法选择策略:heuristic=首选(默认),time=top-N 计时重排
        "KATAGO_CUDA_CUBLASLT_RANK" => matches!(value, "heuristic" | "time"),
        RESIDUAL_ALGO_KEY => matches!(value, "heuristic" | TF3_RESIDUAL_PRESET),
        "KATAGO_CUDA_GEMM_LAYOUT" => {
            matches!(value, "tn" | "nn_k384" | "nn_k384_b8" | "nn_k384_b16")
        }
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
    validate_plan_backend(&plan, backend_build).map_err(&ctx)?;
    // Validate the effective request before installation, including the case
    // where an old plan omits this key and the environment tries to enable it.
    let residual_value = plan
        .apply
        .tactic_overrides
        .get(RESIDUAL_ALGO_KEY)
        .cloned()
        .or_else(|| tactic_var(RESIDUAL_ALGO_KEY).ok());
    let residual_requested = residual_preset_value(residual_value.as_deref()).map_err(&ctx)?;
    validate_residual_plan_override(residual_requested, Some(&plan.apply.tactic_overrides))
        .map_err(&ctx)?;
    if residual_requested {
        validate_tf3_residual_target(device, backend_build.cublaslt_version).map_err(&ctx)?;
    }
    let effective = |key: &str| {
        plan.apply
            .tactic_overrides
            .get(key)
            .cloned()
            .or_else(|| tactic_var(key).ok())
    };
    let gate_requested = gate_rowquad_value(effective(GATE_ROWQUAD_KEY).as_deref()).map_err(&ctx)?;
    validate_gate_rowquad_binding(gate_requested, Some(&plan.apply.tactic_overrides)).map_err(&ctx)?;
    let stem_requested = stem_map3d_value(effective(STEM_MAP3D_KEY).as_deref()).map_err(&ctx)?;
    validate_stem_map3d_binding(stem_requested, Some(&plan.apply.tactic_overrides))
        .map_err(&ctx)?;
    let strict_requested =
        strict_attention_value(effective(STRICT_ATTENTION_KEY).as_deref()).map_err(&ctx)?;
    validate_strict_attention_options(
        strict_requested,
        effective("KATAGO_CUDA_ATTN_TILE").as_deref(),
        effective("KATAGO_CUDA_NOGRAPH").as_deref(),
        Some(&plan.apply.tactic_overrides),
    )
    .map_err(&ctx)?;
    if strict_requested {
        validate_strict_attention_target(device).map_err(&ctx)?;
    }
    validate_qkv_immutable_options(qkv_immutable_value(effective(QKV_IMMUTABLE_KEY).as_deref()).map_err(&ctx)?,strict_requested,effective("KATAGO_CUDA_GEMM_LAYOUT").as_deref(),Some(&plan.apply.tactic_overrides)).map_err(&ctx)?;
    validate_qkv_n128_options(qkv_n128_value(effective(QKV_N128_KEY).as_deref()).map_err(&ctx)?,qkv_immutable_value(effective(QKV_IMMUTABLE_KEY).as_deref()).map_err(&ctx)?,Some(&plan.apply.tactic_overrides)).map_err(&ctx)?;
    let outproj=outproj_value(effective(OUTPROJ_N128_KEY).as_deref()).map_err(&ctx)?;
    validate_outproj_options(outproj,&effective,Some(&plan.apply.tactic_overrides)).map_err(&ctx)?;
    if outproj {validate_strict_attention_target(device).map_err(&ctx)?;}
    validate_compact_options(compact_value(effective(FFN_COMPACT_KEY).as_deref()).map_err(&ctx)?, &effective, Some(&plan.apply.tactic_overrides)).map_err(&ctx)?;
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

fn validate_plan_backend(plan: &PlanFile, actual: &BackendBuildFingerprint) -> Result<(), String> {
    let outproj=outproj_value(plan.apply.tactic_overrides.get(OUTPROJ_N128_KEY).map(String::as_str))?;
    validate_outproj_options(outproj,&|k|plan.apply.tactic_overrides.get(k).cloned(),Some(&plan.apply.tactic_overrides))?;
    let expected_outproj=plan.backend_build.as_ref().and_then(|b|b.outproj_n128_artifact.as_deref());
    if outproj && (plan.schema!=2 || expected_outproj.is_none()) {return Err("outproj N128 requires schema2 and outproj_n128_artifact".into());}
    if outproj && !plan.target.model_sha256.eq_ignore_ascii_case(OUTPROJ_N128_MODEL) {return Err("outproj N128 model identity mismatch".into());}
    if expected_outproj.is_some() && expected_outproj!=actual.outproj_n128_artifact.as_deref() {return Err("backend build outproj_n128_artifact mismatch".into());}

    let compact = compact_value(plan.apply.tactic_overrides.get(FFN_COMPACT_KEY).map(String::as_str))?;
    let expected = plan.backend_build.as_ref().and_then(|b| b.ffn_compact_artifact.as_deref());
    if compact {
        if plan.schema != 2 || expected.is_none() { return Err("FFN compact requires schema 2 and ffn_compact_artifact".into()); }
        if !plan.target.model_sha256.eq_ignore_ascii_case(FFN_COMPACT_MODEL) { return Err("FFN compact model identity mismatch".into()); }
        if !actual.capabilities.dual_ffn || actual.cublaslt_version != Some(130600) { return Err("FFN compact requires CUTLASS and cuBLASLt 130600".into()); }
        validate_compact_options(true, &|k| plan.apply.tactic_overrides.get(k).cloned(), Some(&plan.apply.tactic_overrides))?;
    }
    if expected.is_some() && expected != actual.ffn_compact_artifact.as_deref() { return Err("backend build ffn_compact_artifact mismatch".into()); }

    let qkv_requested=qkv_immutable_value(plan.apply.tactic_overrides.get(QKV_IMMUTABLE_KEY).map(String::as_str))?;
    validate_qkv_n128_options(qkv_n128_value(plan.apply.tactic_overrides.get(QKV_N128_KEY).map(String::as_str))?,qkv_requested,Some(&plan.apply.tactic_overrides))?;
    if qkv_requested && plan.schema!=2{return Err(format!("{QKV_IMMUTABLE_KEY}=1 requires schema 2 and exact CUDA/artifact identities"));}
    let expected_qkv=plan.backend_build.as_ref().and_then(|b|b.qkv_immutable_artifact.as_deref());
    if qkv_requested && expected_qkv.is_none(){return Err(format!("{QKV_IMMUTABLE_KEY}=1 requires backend_build.qkv_immutable_artifact"));}
    if expected_qkv.is_some() && expected_qkv!=actual.qkv_immutable_artifact.as_deref(){return Err("backend build qkv_immutable_artifact mismatch".into());}
    validate_qkv_immutable_options(qkv_requested,strict_attention_value(plan.apply.tactic_overrides.get(STRICT_ATTENTION_KEY).map(String::as_str))?,plan.apply.tactic_overrides.get("KATAGO_CUDA_GEMM_LAYOUT").map(String::as_str),Some(&plan.apply.tactic_overrides))?;

    // executor.cu is covered by kernel_build_id. Require its existing exact
    // schema-2 comparison for the opt-in kernel; no global host revision bump.
    if stem_map3d_value(plan.apply.tactic_overrides.get(STEM_MAP3D_KEY).map(String::as_str))?
        && plan.schema != 2
    {
        return Err(format!("{STEM_MAP3D_KEY}=1 requires schema 2 and its exact CUDA build identity"));
    }
    if gate_rowquad_value(plan.apply.tactic_overrides.get(GATE_ROWQUAD_KEY).map(String::as_str))? && plan.schema != 2 {
        return Err(format!("{GATE_ROWQUAD_KEY}=1 requires schema 2 and its exact CUDA build identity"));
    }
    // Weight encoding affects every tactic, including empty/negative overrides.
    // Check both schemas before legacy build/layout/library checks so an old
    // archive cannot silently inherit certification under corrected weights.
    let expected_encoding = plan
        .backend_build
        .as_ref()
        .and_then(|build| build.fp16_encoding_revision)
        .ok_or_else(|| {
            "CUDA plan requires backend_build.fp16_encoding_revision; legacy FP16 encoding is not certified for this binary; rerun numerical validation and migrate the plan".to_string()
        })?;
    if Some(expected_encoding) != actual.fp16_encoding_revision {
        return Err(format!(
            "backend build fp16_encoding_revision mismatch: plan '{:?}' vs actual '{:?}'; rerun numerical validation and migrate the plan",
            Some(expected_encoding),
            actual.fp16_encoding_revision
        ));
    }
    if plan.schema == 2 {
        let expected = plan
            .backend_build
            .as_ref()
            .ok_or_else(|| "schema 2 requires backend_build".to_string())?;
        validate_backend_build(expected, actual)?;
    }
    let expected_artifact = plan
        .backend_build
        .as_ref()
        .and_then(|build| build.strict_attention_artifact.as_deref());
    let strict_requested = strict_attention_value(
        plan.apply
            .tactic_overrides
            .get(STRICT_ATTENTION_KEY)
            .map(String::as_str),
    )?;
    if strict_requested && expected_artifact.is_none() {
        return Err(format!(
            "{STRICT_ATTENTION_PRESET} requires backend_build.strict_attention_artifact; rerun numerical and performance validation"
        ));
    }
    if expected_artifact.is_some()
        && expected_artifact != actual.strict_attention_artifact.as_deref()
    {
        return Err(format!(
            "backend build strict_attention_artifact mismatch: plan {expected_artifact:?} vs actual {:?}",
            actual.strict_attention_artifact
        ));
    }
    validate_strict_attention_options(
        strict_requested,
        plan.apply
            .tactic_overrides
            .get("KATAGO_CUDA_ATTN_TILE")
            .map(String::as_str),
        plan.apply
            .tactic_overrides
            .get("KATAGO_CUDA_NOGRAPH")
            .map(String::as_str),
        Some(&plan.apply.tactic_overrides),
    )?;
    let expected_revision = plan
        .backend_build
        .as_ref()
        .and_then(|build| build.host_tactic_revision);
    // This applies to both schemas and to an explicit TN override as well:
    // negative overrides also rely on installing the plan before weight upload.
    if plan
        .apply
        .tactic_overrides
        .contains_key("KATAGO_CUDA_GEMM_LAYOUT")
        && expected_revision.is_none()
    {
        return Err(
            "plan controls KATAGO_CUDA_GEMM_LAYOUT but requires backend_build.host_tactic_revision"
                .to_string(),
        );
    }
    if expected_revision.is_some() && expected_revision != actual.host_tactic_revision {
        return Err(format!(
            "backend build host_tactic_revision mismatch: plan '{expected_revision:?}' vs actual '{:?}'",
            actual.host_tactic_revision
        ));
    }
    let expected_library = plan
        .backend_build
        .as_ref()
        .and_then(|build| build.cublaslt_version);
    if expected_library.is_some() && expected_library != actual.cublaslt_version {
        return Err(format!(
            "backend build cublaslt_version mismatch: plan {expected_library:?} vs actual {:?}",
            actual.cublaslt_version
        ));
    }
    if plan
        .apply
        .tactic_overrides
        .get(RESIDUAL_ALGO_KEY)
        .map(String::as_str)
        == Some(TF3_RESIDUAL_PRESET)
    {
        let expected_version = expected_library
            .ok_or_else(|| format!(
                "plan controls {RESIDUAL_ALGO_KEY}={TF3_RESIDUAL_PRESET} but requires backend_build.cublaslt_version"
            ))?;
        if expected_version != TF3_RESIDUAL_CUBLASLT_VERSION {
            return Err(format!(
                "{TF3_RESIDUAL_PRESET} requires cuBLASLt {TF3_RESIDUAL_CUBLASLT_VERSION}, plan {expected_version}"
            ));
        }
        if plan
            .apply
            .tactic_overrides
            .get("KATAGO_CUDA_CUBLASLT")
            .map(String::as_str)
            == Some("0")
        {
            return Err(format!(
                "{TF3_RESIDUAL_PRESET} requires KATAGO_CUDA_CUBLASLT enabled"
            ));
        }
    }
    validate_required_capabilities(&plan.apply.tactic_overrides, actual)
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
    if overrides.get("KATAGO_CUDA_ATTN_TILE").map(String::as_str) == Some("q64-serial")
        && !backend_build.capabilities.attention_q64_serial
    {
        return Err(
            "plan requests KATAGO_CUDA_ATTN_TILE=q64-serial but backend capability attention_q64_serial=false".to_string(),
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


    #[test] fn compact_values_bindings_and_artifacts() {
        assert!(!compact_value(None).unwrap()); assert!(!compact_value(Some("0")).unwrap());
        assert!(compact_value(Some("1")).unwrap());
        for v in ["true", "2", ""] { assert!(compact_value(Some(v)).is_err()); assert!(validate_value(FFN_COMPACT_KEY,v).is_err()); }
        let mut o:HashMap<String,String> = COMPACT_DEPS.iter().map(|&(k,v)|(k.into(),v.into())).collect();
        o.insert(FFN_COMPACT_KEY.into(),"1".into());
        validate_compact_options(true,&|k|o.get(k).cloned(),Some(&o)).unwrap();
        for key in o.keys() {
            let mut missing=o.clone(); missing.remove(key);
            assert!(validate_compact_options(true,&|k|o.get(k).cloned(),Some(&missing)).is_err());
        }
        validate_compact_options(false,&|_|Some("1".into()),Some(&HashMap::new())).unwrap();
        let mut b=backend_build(); b.ffn_compact_artifact=Some("sha256:compact-test".into());
        let mut p=backend_plan(2,&serde_json::to_string(&o).unwrap(),Some(&b));
        p.target.model_sha256=FFN_COMPACT_MODEL.into();
        validate_plan_backend(&p,&b).unwrap();
        p.schema=1; assert!(validate_plan_backend(&p,&b).is_err()); p.schema=2;
        p.target.model_sha256="00".repeat(32); assert!(validate_plan_backend(&p,&b).is_err()); p.target.model_sha256=FFN_COMPACT_MODEL.into();
        let mut other=b.clone(); other.ffn_compact_artifact=Some("different".into()); assert!(validate_plan_backend(&p,&other).is_err());
        other.ffn_compact_artifact=None; assert!(validate_plan_backend(&p,&other).is_err());
        p.backend_build.as_mut().unwrap().ffn_compact_artifact=None; assert!(validate_plan_backend(&p,&b).is_err());
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
                attention_q64_serial: false,
            },
            fp16_encoding_revision: Some(CUDA_FP16_ENCODING_REVISION),
            host_tactic_revision: Some(CUDA_HOST_TACTIC_REVISION),
            cublaslt_version: Some(TF3_RESIDUAL_CUBLASLT_VERSION),
            strict_attention_artifact: None,
            qkv_immutable_artifact: None,
            ffn_compact_artifact: None,
            outproj_n128_artifact: None,
        }
    }

    fn plan_json(overrides: &str) -> String {
        format!(
            r#"{{
              "schema": 1,
              "kind": "cuda-tactic-plan",
              "plan_id": "test-plan",
              "backend_build": {BUILD},
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
            BUILD = serde_json::to_string(&backend_build()).unwrap(),
            overrides = overrides
        )
    }

    fn plan_json_v2(overrides: &str, build: &BackendBuildFingerprint) -> String {
        let mut value: serde_json::Value = serde_json::from_str(&plan_json(overrides)).unwrap();
        value["schema"] = serde_json::json!(2);
        value["backend_build"] = serde_json::to_value(build).unwrap();
        serde_json::to_string_pretty(&value).unwrap()
    }

    fn backend_plan(
        schema: u32,
        overrides: &str,
        build: Option<&BackendBuildFingerprint>,
    ) -> PlanFile {
        let mut value: serde_json::Value = serde_json::from_str(&plan_json(overrides)).unwrap();
        value["schema"] = serde_json::json!(schema);
        if let Some(build) = build {
            value["backend_build"] = serde_json::to_value(build).unwrap();
        } else {
            value.as_object_mut().unwrap().remove("backend_build");
        }
        serde_json::from_value(value).unwrap()
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
    fn gemm_layout_accepts_only_implemented_contracts() {
        for value in ["tn", "nn_k384", "nn_k384_b8", "nn_k384_b16"] {
            assert!(validate_value("KATAGO_CUDA_GEMM_LAYOUT", value).is_ok());
        }
        for value in ["", "nn", "fp16", "NN_K384"] {
            assert!(validate_value("KATAGO_CUDA_GEMM_LAYOUT", value).is_err());
        }
    }

    #[test]
    fn stem_map3d_value_and_plan_binding() {
        assert!(ALLOWED_TACTIC_KEYS.contains(&STEM_MAP3D_KEY));
        assert!(!stem_map3d_value(None).unwrap());
        for value in ["0", "1"] {
            validate_value(STEM_MAP3D_KEY, value).unwrap();
            assert_eq!(stem_map3d_value(Some(value)).unwrap(), value == "1");
        }
        for value in ["", "true", "map3d", "2"] {
            assert!(validate_value(STEM_MAP3D_KEY, value).is_err());
            assert!(stem_map3d_value(Some(value)).is_err());
        }
        validate_stem_map3d_binding(true, None).unwrap();
        let missing = HashMap::new();
        let off = HashMap::from([(STEM_MAP3D_KEY.into(), "0".into())]);
        let on = HashMap::from([(STEM_MAP3D_KEY.into(), "1".into())]);
        assert!(validate_stem_map3d_binding(true, Some(&missing)).is_err());
        assert!(validate_stem_map3d_binding(true, Some(&off)).is_err());
        validate_stem_map3d_binding(true, Some(&on)).unwrap();
        validate_stem_map3d_binding(false, Some(&missing)).unwrap();
    }

    #[test]
    fn stem_map3d_requires_existing_exact_schema2_build_binding() {
        let build = backend_build();
        let on = r#"{ "KATAGO_CUDA_STEM_MAP3D": "1" }"#;
        assert!(validate_plan_backend(&backend_plan(1, on, Some(&build)), &build).is_err());
        validate_plan_backend(&backend_plan(2, on, Some(&build)), &build).unwrap();
        let mut stale = build.clone();
        stale.kernel_build_id = "sha256:stale".into();
        assert!(validate_plan_backend(&backend_plan(2, on, Some(&stale)), &build).is_err());
        validate_plan_backend(&backend_plan(1, r#"{ "KATAGO_CUDA_STEM_MAP3D": "0" }"#, Some(&build)), &build).unwrap();
    }

    #[test]
    fn gate_rowquad_value_and_plan_binding() {
        assert!(ALLOWED_TACTIC_KEYS.contains(&GATE_ROWQUAD_KEY));
        assert!(!gate_rowquad_value(None).unwrap());
        for value in ["0", "1"] {
            validate_value(GATE_ROWQUAD_KEY, value).unwrap();
            assert_eq!(gate_rowquad_value(Some(value)).unwrap(), value == "1");
        }
        for value in ["", "true", "map3d", "2"] {
            assert!(validate_value(GATE_ROWQUAD_KEY, value).is_err());
            assert!(gate_rowquad_value(Some(value)).is_err());
        }
        validate_gate_rowquad_binding(true, None).unwrap();
        let missing = HashMap::new();
        let off = HashMap::from([(GATE_ROWQUAD_KEY.into(), "0".into())]);
        let on = HashMap::from([(GATE_ROWQUAD_KEY.into(), "1".into())]);
        assert!(validate_gate_rowquad_binding(true, Some(&missing)).is_err());
        assert!(validate_gate_rowquad_binding(true, Some(&off)).is_err());
        validate_gate_rowquad_binding(true, Some(&on)).unwrap();
        validate_gate_rowquad_binding(false, Some(&missing)).unwrap();
    }

    #[test]
    fn gate_rowquad_requires_existing_exact_schema2_build_binding() {
        let build = backend_build();
        let on = r#"{ "KATAGO_CUDA_GATE_ROWQUAD_R1": "1" }"#;
        assert!(validate_plan_backend(&backend_plan(1, on, Some(&build)), &build).is_err());
        validate_plan_backend(&backend_plan(2, on, Some(&build)), &build).unwrap();
        let mut stale = build.clone();
        stale.kernel_build_id = "sha256:stale".into();
        assert!(validate_plan_backend(&backend_plan(2, on, Some(&stale)), &build).is_err());
        validate_plan_backend(&backend_plan(1, r#"{ "KATAGO_CUDA_GATE_ROWQUAD_R1": "0" }"#, Some(&build)), &build).unwrap();
    }

    #[test]
    fn residual_algo_accepts_only_registered_presets() {
        assert!(ALLOWED_TACTIC_KEYS.contains(&RESIDUAL_ALGO_KEY));
        assert!(!residual_preset_value(None).unwrap());
        for value in ["heuristic", TF3_RESIDUAL_PRESET] {
            validate_value(RESIDUAL_ALGO_KEY, value).unwrap();
            assert_eq!(
                residual_preset_value(Some(value)).unwrap(),
                value == TF3_RESIDUAL_PRESET
            );
        }
        for value in ["", "time", "tf3", "1"] {
            assert!(validate_value(RESIDUAL_ALGO_KEY, value).is_err());
            assert!(residual_preset_value(Some(value)).is_err());
        }
    }

    #[test]
    fn strict_attention_accepts_only_registered_paths() {
        assert!(!strict_attention_value(None).unwrap());
        for value in ["fa2", "v3", STRICT_ATTENTION_PRESET] {
            validate_value(STRICT_ATTENTION_KEY, value).unwrap();
            assert_eq!(
                strict_attention_value(Some(value)).unwrap(),
                value == STRICT_ATTENTION_PRESET
            );
        }
        for value in ["", "fa4", "fa4-strict", "1", "FA2"] {
            assert!(validate_value(STRICT_ATTENTION_KEY, value).is_err());
            assert!(strict_attention_value(Some(value)).is_err());
        }
    }

    fn strict_overrides() -> HashMap<String, String> {
        HashMap::from([
            (STRICT_ATTENTION_KEY.into(), STRICT_ATTENTION_PRESET.into()),
            ("KATAGO_CUDA_ATTN_TILE".into(), "q64-serial".into()),
            ("KATAGO_CUDA_NOGRAPH".into(), "1".into()),
        ])
    }

    #[test]
    fn strict_attention_requires_bound_fallback_and_graph_policy() {
        let overrides = strict_overrides();
        validate_strict_attention_options(true, Some("q64-serial"), Some("1"), Some(&overrides))
            .unwrap();
        validate_strict_attention_options(true, Some("q64-serial"), Some("1"), None).unwrap();
        for key in [
            STRICT_ATTENTION_KEY,
            "KATAGO_CUDA_ATTN_TILE",
            "KATAGO_CUDA_NOGRAPH",
        ] {
            let mut missing = overrides.clone();
            missing.remove(key);
            // Even correctly resolved environment values cannot fill omissions.
            let err = validate_strict_attention_options(
                true,
                Some("q64-serial"),
                Some("1"),
                Some(&missing),
            )
            .unwrap_err();
            assert!(err.contains("explicitly bound"), "{err}");
        }
        for tile in [None, Some("q64"), Some("q128")] {
            assert!(validate_strict_attention_options(true, tile, Some("1"), None).is_err());
        }
        for nograph in [None, Some("0")] {
            assert!(
                validate_strict_attention_options(true, Some("q64-serial"), nograph, None).is_err()
            );
        }
        // A negative plan override wins over a conflicting environment request.
        let legacy = HashMap::from([(STRICT_ATTENTION_KEY.into(), "fa2".into())]);
        validate_strict_attention_options(false, None, None, Some(&legacy)).unwrap();
    }

    #[test]
    fn strict_attention_artifact_binding_is_required_in_both_schemas() {
        let mut actual = backend_build();
        actual.capabilities.attention_q64_serial = true;
        actual.strict_attention_artifact = Some(format!("sha256:{}", "77".repeat(32)));
        let overrides = serde_json::to_string(&strict_overrides()).unwrap();
        for schema in [1, 2] {
            validate_plan_backend(&backend_plan(schema, &overrides, Some(&actual)), &actual)
                .unwrap();
            for explicit_null in [false, true] {
                let mut value: serde_json::Value =
                    serde_json::from_str(&plan_json_v2(&overrides, &actual)).unwrap();
                value["schema"] = serde_json::json!(schema);
                if explicit_null {
                    value["backend_build"]["strict_attention_artifact"] = serde_json::Value::Null;
                } else {
                    value["backend_build"]
                        .as_object_mut()
                        .unwrap()
                        .remove("strict_attention_artifact");
                }
                let plan: PlanFile = serde_json::from_value(value).unwrap();
                assert!(
                    validate_plan_backend(&plan, &actual)
                        .unwrap_err()
                        .contains("requires backend_build.strict_attention_artifact")
                );
            }
            for artifact in [None, Some(format!("sha256:{}", "88".repeat(32)))] {
                let mut other = actual.clone();
                other.strict_attention_artifact = artifact;
                let plan = backend_plan(schema, &overrides, Some(&actual));
                assert!(
                    validate_plan_backend(&plan, &other)
                        .unwrap_err()
                        .contains("strict_attention_artifact mismatch")
                );
            }
            // Explicitly bound assets are checked even when the tactic is off.
            let mut wrong = actual.clone();
            wrong.strict_attention_artifact = Some("wrong".into());
            assert!(
                validate_plan_backend(&backend_plan(schema, "{}", Some(&wrong)), &actual)
                    .unwrap_err()
                    .contains("strict_attention_artifact mismatch")
            );
        }
    }

    #[test]
    fn legacy_plans_do_not_claim_strict_attention_certification() {
        let legacy = backend_build();
        let mut with_artifact = legacy.clone();
        with_artifact.strict_attention_artifact = Some(format!("sha256:{}", "77".repeat(32)));
        assert!(
            serde_json::to_value(&legacy)
                .unwrap()
                .get("strict_attention_artifact")
                .is_none()
        );
        for schema in [1, 2] {
            for overrides in [
                "{}",
                r#"{"KATAGO_CUDA_ATTN":"fa2"}"#,
                r#"{"KATAGO_CUDA_ATTN":"v3"}"#,
            ] {
                validate_plan_backend(
                    &backend_plan(schema, overrides, Some(&legacy)),
                    &with_artifact,
                )
                .unwrap();
            }
        }
    }

    #[test]
    fn strict_attention_rejects_unmeasured_device() {
        let measured = DeviceFingerprint {
            sm_count: 70,
            ..device()
        };
        validate_strict_attention_target(&measured).unwrap();
        for other in [
            DeviceFingerprint {
                gpu_name: "NVIDIA GeForce RTX 5080".into(),
                ..measured.clone()
            },
            DeviceFingerprint {
                compute_capability: "12.1".into(),
                ..measured.clone()
            },
            DeviceFingerprint {
                sm_count: 84,
                ..measured.clone()
            },
            DeviceFingerprint {
                l2_cache_bytes: 67108864,
                ..measured.clone()
            },
        ] {
            assert!(
                validate_strict_attention_target(&other)
                    .unwrap_err()
                    .contains("requires RTX 5070 Ti")
            );
        }
    }

    #[test]
    fn revalidated_plans_without_library_version_keep_optional_library_contract() {
        let actual = backend_build();
        let mut legacy = actual.clone();
        legacy.cublaslt_version = None;
        assert!(
            serde_json::to_value(&legacy)
                .unwrap()
                .get("cublaslt_version")
                .is_none()
        );
        for schema in [1, 2] {
            for overrides in [
                "{}",
                r#"{"KATAGO_CUDA_GEMM_LAYOUT":"nn_k384_b16"}"#,
                r#"{"KATAGO_CUDA_RESIDUAL_ALGO":"heuristic"}"#,
            ] {
                let plan = backend_plan(schema, overrides, Some(&legacy));
                assert_eq!(plan.backend_build.as_ref().unwrap().cublaslt_version, None);
                validate_plan_backend(&plan, &actual).unwrap();
            }
        }
        assert_eq!(actual.host_tactic_revision, Some(CUDA_HOST_TACTIC_REVISION));
        // The FP16 contract is already present. Only omission preserves this
        // optional library contract; an explicit library field must match even
        // if this plan never enables the new preset.
        let mut other_library = actual.clone();
        other_library.cublaslt_version = Some(130500);
        validate_plan_backend(&backend_plan(2, "{}", Some(&legacy)), &other_library).unwrap();
        for schema in [1, 2] {
            for overrides in ["{}", r#"{"KATAGO_CUDA_RESIDUAL_ALGO":"heuristic"}"#] {
                assert!(
                    validate_plan_backend(
                        &backend_plan(schema, overrides, Some(&actual)),
                        &other_library
                    )
                    .unwrap_err()
                    .contains("cublaslt_version mismatch")
                );
            }
        }
    }

    #[test]
    fn residual_preset_requires_explicit_library_binding_in_both_schemas() {
        let actual = backend_build();
        let overrides = r#"{"KATAGO_CUDA_RESIDUAL_ALGO":"tf3_5070ti_r1"}"#;
        let mut legacy = actual.clone();
        legacy.cublaslt_version = None;
        for schema in [1, 2] {
            let err =
                validate_plan_backend(&backend_plan(schema, overrides, Some(&legacy)), &actual)
                    .unwrap_err();
            assert!(
                err.contains("requires backend_build.cublaslt_version"),
                "{err}"
            );
            validate_plan_backend(&backend_plan(schema, overrides, Some(&actual)), &actual)
                .unwrap();
            let mut value: serde_json::Value =
                serde_json::from_str(&plan_json_v2(overrides, &actual)).unwrap();
            value["schema"] = serde_json::json!(schema);
            value["backend_build"]["cublaslt_version"] = serde_json::Value::Null;
            let plan: PlanFile = serde_json::from_value(value).unwrap();
            assert!(
                validate_plan_backend(&plan, &actual)
                    .unwrap_err()
                    .contains("requires backend_build.cublaslt_version")
            );
        }
    }

    #[test]
    fn residual_preset_rejects_mismatched_or_unmeasured_library() {
        let expected = backend_build();
        let overrides = r#"{"KATAGO_CUDA_RESIDUAL_ALGO":"tf3_5070ti_r1"}"#;
        for schema in [1, 2] {
            let plan = backend_plan(schema, overrides, Some(&expected));
            for version in [None, Some(130500), Some(130601)] {
                let mut actual = expected.clone();
                actual.cublaslt_version = version;
                assert!(
                    validate_plan_backend(&plan, &actual)
                        .unwrap_err()
                        .contains("cublaslt_version mismatch")
                );
            }
            let mut other = expected.clone();
            other.cublaslt_version = Some(130500);
            assert!(
                validate_plan_backend(&backend_plan(schema, overrides, Some(&other)), &other)
                    .unwrap_err()
                    .contains("requires cuBLASLt 130600")
            );
            let disabled =
                r#"{"KATAGO_CUDA_RESIDUAL_ALGO":"tf3_5070ti_r1","KATAGO_CUDA_CUBLASLT":"0"}"#;
            assert!(
                validate_plan_backend(&backend_plan(schema, disabled, Some(&expected)), &expected)
                    .unwrap_err()
                    .contains("requires KATAGO_CUDA_CUBLASLT enabled")
            );
        }
    }

    #[test]
    fn residual_preset_rejects_other_gpu_or_device_attributes() {
        let actual = DeviceFingerprint {
            sm_count: 70,
            ..device()
        };
        validate_tf3_residual_target(&actual, Some(TF3_RESIDUAL_CUBLASLT_VERSION)).unwrap();
        for other in [
            DeviceFingerprint {
                gpu_name: "NVIDIA GeForce RTX 5080".into(),
                ..actual.clone()
            },
            DeviceFingerprint {
                compute_capability: "12.1".into(),
                ..actual.clone()
            },
            DeviceFingerprint {
                sm_count: 84,
                ..actual.clone()
            },
            DeviceFingerprint {
                l2_cache_bytes: 67108864,
                ..actual.clone()
            },
        ] {
            assert!(
                validate_tf3_residual_target(&other, Some(TF3_RESIDUAL_CUBLASLT_VERSION))
                    .unwrap_err()
                    .contains("requires RTX 5070 Ti")
            );
        }
        for version in [None, Some(130500)] {
            assert!(
                validate_tf3_residual_target(&actual, version)
                    .unwrap_err()
                    .contains("requires cuBLASLt 130600")
            );
        }
    }

    #[test]
    fn environment_cannot_enable_unbound_preset_under_old_plan() {
        let empty = HashMap::new();
        validate_residual_plan_override(false, Some(&empty)).unwrap();
        // A standalone environment experiment still validates actual runtime
        // target and algorithm identity; a certified plan must record the key.
        validate_residual_plan_override(true, None).unwrap();
        assert!(
            validate_residual_plan_override(true, Some(&empty))
                .unwrap_err()
                .contains("explicitly bound")
        );
        let enabled = HashMap::from([(RESIDUAL_ALGO_KEY.into(), TF3_RESIDUAL_PRESET.into())]);
        validate_residual_plan_override(true, Some(&enabled)).unwrap();
        let disabled = HashMap::from([(RESIDUAL_ALGO_KEY.into(), "heuristic".into())]);
        validate_residual_plan_override(false, Some(&disabled)).unwrap();
        assert!(validate_residual_plan_override(true, Some(&disabled)).is_err());
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
    fn all_schemas_reject_missing_encoding_contract_before_installation() {
        let dir = std::env::temp_dir().join("kata_tactic_plan_test_missing_encoding");
        std::fs::create_dir_all(&dir).unwrap();
        for schema in [1, 2] {
            for missing_build in [false, true] {
                let mut value: serde_json::Value = serde_json::from_str(&plan_json("{}")).unwrap();
                value["schema"] = serde_json::json!(schema);
                if missing_build {
                    value.as_object_mut().unwrap().remove("backend_build");
                } else {
                    value["backend_build"]
                        .as_object_mut()
                        .unwrap()
                        .remove("fp16_encoding_revision");
                }
                let p = write_plan(
                    &dir,
                    &format!("schema{schema}-missing-build-{missing_build}.json"),
                    &serde_json::to_string(&value).unwrap(),
                );
                let err = load_and_install(&p, &device(), &"aa".repeat(32), &backend_build())
                    .unwrap_err();
                assert!(
                    err.contains("requires backend_build.fp16_encoding_revision"),
                    "{err}"
                );
                assert!(
                    err.contains("rerun numerical validation and migrate the plan"),
                    "{err}"
                );
            }
        }
    }

    #[test]
    fn every_tactic_requires_explicit_fp16_encoding_in_both_schemas() {
        let actual = backend_build();
        assert_eq!(actual.fp16_encoding_revision, Some(1));
        assert_eq!(
            serde_json::to_value(&actual).unwrap()["fp16_encoding_revision"],
            serde_json::json!(1)
        );
        for schema in [1, 2] {
            for host_revision in [None, Some(CUDA_HOST_TACTIC_REVISION)] {
                let mut legacy = actual.clone();
                legacy.host_tactic_revision = host_revision;
                legacy.fp16_encoding_revision = None;
                assert!(
                    serde_json::to_value(&legacy)
                        .unwrap()
                        .get("fp16_encoding_revision")
                        .is_none()
                );
                for overrides in [
                    "{}",
                    r#"{"KATAGO_CUDA_DUALFFN":"1"}"#,
                    r#"{"KATAGO_CUDA_CUBLASLT":"0"}"#,
                    r#"{"KATAGO_CUDA_GEMM_LAYOUT":"tn"}"#,
                    r#"{"KATAGO_CUDA_GEMM_LAYOUT":"nn_k384_b16"}"#,
                    r#"{"KATAGO_CUDA_RESIDUAL_ALGO":"heuristic"}"#,
                    r#"{"KATAGO_CUDA_RESIDUAL_ALGO":"tf3_5070ti_r1"}"#,
                ] {
                    for explicit_null in [false, true] {
                        let mut value: serde_json::Value =
                            serde_json::from_str(&plan_json_v2(overrides, &legacy)).unwrap();
                        value["schema"] = serde_json::json!(schema);
                        if explicit_null {
                            value["backend_build"]["fp16_encoding_revision"] =
                                serde_json::Value::Null;
                        }
                        // Old archives remain readable, but are never eligible
                        // for installation under the corrected weight encoder.
                        let plan: PlanFile = serde_json::from_value(value).unwrap();
                        let err = validate_plan_backend(&plan, &actual).unwrap_err();
                        assert!(
                            err.contains("requires backend_build.fp16_encoding_revision"),
                            "schema={schema} overrides={overrides}: {err}"
                        );
                        assert!(
                            err.contains("rerun numerical validation and migrate"),
                            "{err}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn fp16_encoding_must_match_even_without_host_revision_or_with_lt_preset() {
        let actual = backend_build();
        for schema in [1, 2] {
            for host_revision in [None, Some(CUDA_HOST_TACTIC_REVISION)] {
                for overrides in [
                    "{}",
                    r#"{"KATAGO_CUDA_DUALFFN":"1"}"#,
                    r#"{"KATAGO_CUDA_GEMM_LAYOUT":"tn"}"#,
                    r#"{"KATAGO_CUDA_RESIDUAL_ALGO":"tf3_5070ti_r1"}"#,
                ] {
                    for revision in [0, 2, u32::MAX] {
                        let mut expected = actual.clone();
                        expected.host_tactic_revision = host_revision;
                        expected.fp16_encoding_revision = Some(revision);
                        let err = validate_plan_backend(
                            &backend_plan(schema, overrides, Some(&expected)),
                            &actual,
                        )
                        .unwrap_err();
                        assert!(err.contains("fp16_encoding_revision mismatch"), "{err}");
                        assert!(
                            err.contains("rerun numerical validation and migrate"),
                            "{err}"
                        );
                    }
                }
            }
            let plan = backend_plan(schema, "{}", Some(&actual));
            for revision in [None, Some(0), Some(CUDA_FP16_ENCODING_REVISION + 1)] {
                let mut unavailable_or_other = actual.clone();
                unavailable_or_other.fp16_encoding_revision = revision;
                let err = validate_plan_backend(&plan, &unavailable_or_other).unwrap_err();
                assert!(err.contains("fp16_encoding_revision mismatch"), "{err}");
            }
        }
    }

    #[test]
    fn corrected_encoding_preserves_independent_host_and_library_requirements() {
        let actual = backend_build();
        let mut minimal = actual.clone();
        minimal.host_tactic_revision = None;
        minimal.cublaslt_version = None;
        for schema in [1, 2] {
            for overrides in [
                "{}",
                r#"{"KATAGO_CUDA_DUALFFN":"1"}"#,
                r#"{"KATAGO_CUDA_RESIDUAL_ALGO":"heuristic"}"#,
            ] {
                validate_plan_backend(&backend_plan(schema, overrides, Some(&minimal)), &actual)
                    .unwrap();
            }
            let layout = r#"{"KATAGO_CUDA_GEMM_LAYOUT":"nn_k384_b16"}"#;
            let err = validate_plan_backend(&backend_plan(schema, layout, Some(&minimal)), &actual)
                .unwrap_err();
            assert!(
                err.contains("requires backend_build.host_tactic_revision"),
                "{err}"
            );
            let preset = r#"{"KATAGO_CUDA_RESIDUAL_ALGO":"tf3_5070ti_r1"}"#;
            let err = validate_plan_backend(&backend_plan(schema, preset, Some(&minimal)), &actual)
                .unwrap_err();
            assert!(
                err.contains("requires backend_build.cublaslt_version"),
                "{err}"
            );
            let both = r#"{"KATAGO_CUDA_GEMM_LAYOUT":"nn_k384_b16","KATAGO_CUDA_RESIDUAL_ALGO":"tf3_5070ti_r1"}"#;
            validate_plan_backend(&backend_plan(schema, both, Some(&actual)), &actual).unwrap();
        }
    }

    #[test]
    fn revalidated_plans_without_host_revision_accept_original_tactics() {
        let actual = backend_build();
        let mut legacy = actual.clone();
        legacy.host_tactic_revision = None;
        let legacy_json = serde_json::to_value(&legacy).unwrap();
        assert!(legacy_json.get("host_tactic_revision").is_none());
        let overrides = r#"{ "KATAGO_CUDA_DUALFFN": "1", "KATAGO_CUDA_CUBLASLT": "1" }"#;
        for schema in [1, 2] {
            let plan = backend_plan(schema, overrides, Some(&legacy));
            assert_eq!(
                plan.backend_build.as_ref().unwrap().host_tactic_revision,
                None
            );
            validate_plan_backend(&plan, &actual).unwrap();
        }
        // Omitting the host revision must not weaken schema 2's original gates.
        legacy.kernel_build_id = "different-device-code".into();
        let err =
            validate_plan_backend(&backend_plan(2, overrides, Some(&legacy)), &actual).unwrap_err();
        assert!(err.contains("kernel_build_id mismatch"), "{err}");
    }

    #[test]
    fn every_layout_override_requires_host_revision_in_both_schemas() {
        let actual = backend_build();
        let mut legacy = actual.clone();
        legacy.host_tactic_revision = None;
        for schema in [1, 2] {
            for layout in ["tn", "nn_k384", "nn_k384_b8", "nn_k384_b16"] {
                let overrides = format!(r#"{{ "KATAGO_CUDA_GEMM_LAYOUT": "{layout}" }}"#);
                let err = validate_plan_backend(
                    &backend_plan(schema, &overrides, Some(&legacy)),
                    &actual,
                )
                .unwrap_err();
                assert!(
                    err.contains("requires backend_build.host_tactic_revision"),
                    "schema={schema} layout={layout}: {err}"
                );
                // An explicit JSON null is the same absence, not a version match.
                let mut value: serde_json::Value =
                    serde_json::from_str(&plan_json_v2(&overrides, &actual)).unwrap();
                value["schema"] = serde_json::json!(schema);
                value["backend_build"]["host_tactic_revision"] = serde_json::Value::Null;
                let plan = serde_json::from_value(value).unwrap();
                let err = validate_plan_backend(&plan, &actual).unwrap_err();
                assert!(
                    err.contains("requires backend_build.host_tactic_revision"),
                    "{err}"
                );
            }
        }
    }

    #[test]
    fn matched_host_revision_accepts_layouts_in_both_schemas() {
        let actual = backend_build();
        assert_eq!(actual.host_tactic_revision, Some(CUDA_HOST_TACTIC_REVISION));
        let fingerprint_json = serde_json::to_value(&actual).unwrap();
        assert_eq!(
            fingerprint_json["host_tactic_revision"],
            serde_json::json!(CUDA_HOST_TACTIC_REVISION)
        );
        for schema in [1, 2] {
            for layout in ["tn", "nn_k384", "nn_k384_b8", "nn_k384_b16"] {
                let overrides = format!(r#"{{ "KATAGO_CUDA_GEMM_LAYOUT": "{layout}" }}"#);
                validate_plan_backend(&backend_plan(schema, &overrides, Some(&actual)), &actual)
                    .unwrap();
            }
        }
    }

    #[test]
    fn explicit_host_revision_requires_exact_match_even_for_legacy_tactics() {
        let expected = backend_build();
        for schema in [1, 2] {
            for overrides in [
                r#"{ "KATAGO_CUDA_DUALFFN": "1" }"#,
                r#"{ "KATAGO_CUDA_GEMM_LAYOUT": "tn" }"#,
                r#"{ "KATAGO_CUDA_GEMM_LAYOUT": "nn_k384_b8" }"#,
                r#"{ "KATAGO_CUDA_GEMM_LAYOUT": "nn_k384_b16" }"#,
            ] {
                let plan = backend_plan(schema, overrides, Some(&expected));
                for revision in [None, Some(0), Some(CUDA_HOST_TACTIC_REVISION - 1), Some(CUDA_HOST_TACTIC_REVISION + 1)] {
                    let mut actual = expected.clone();
                    actual.host_tactic_revision = revision;
                    let err = validate_plan_backend(&plan, &actual).unwrap_err();
                    assert!(err.contains("host_tactic_revision mismatch"), "{err}");
                }
            }
        }
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
                "host-tactic-revision",
                BackendBuildFingerprint {
                    host_tactic_revision: Some(CUDA_HOST_TACTIC_REVISION + 1),
                    ..expected.clone()
                },
                "host_tactic_revision mismatch",
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

    #[test]
    fn q64_serial_requires_its_own_compiled_capability() {
        validate_value("KATAGO_CUDA_ATTN_TILE", "q64-serial").unwrap();
        let overrides = HashMap::from([(
            "KATAGO_CUDA_ATTN_TILE".to_string(),
            "q64-serial".to_string(),
        )]);
        let mut actual = backend_build();
        actual.capabilities.attention_q64 = true;
        let err = validate_required_capabilities(&overrides, &actual).unwrap_err();
        assert!(
            err.contains("capability attention_q64_serial=false"),
            "{err}"
        );
        let ordinary_q64 =
            HashMap::from([("KATAGO_CUDA_ATTN_TILE".to_string(), "q64".to_string())]);
        validate_required_capabilities(&ordinary_q64, &actual).unwrap();
        actual.capabilities.attention_q64_serial = true;
        validate_required_capabilities(&overrides, &actual).unwrap();
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
    #[test]
    fn outproj_options_artifact_and_model_binding() {
        assert!(!outproj_value(None).unwrap());for v in ["0","1"] {validate_value(OUTPROJ_N128_KEY,v).unwrap();}
        for v in ["2","true",""] {assert!(outproj_value(Some(v)).is_err());}
        let mut o:HashMap<String,String>=OUTPROJ_DEPS.iter().map(|&(k,v)|(k.into(),v.into())).collect();o.insert(OUTPROJ_N128_KEY.into(),"1".into());
        validate_outproj_options(true,&|k|o.get(k).cloned(),Some(&o)).unwrap();
        for key in OUTPROJ_DEPS.iter().map(|&(k,_)|k).chain(std::iter::once(OUTPROJ_N128_KEY)) {let mut missing=o.clone();missing.remove(key);assert!(validate_outproj_options(true,&|k|o.get(k).cloned(),Some(&missing)).is_err());}
        let mut actual=backend_build();actual.outproj_n128_artifact=Some("outproj".into());
        let mut plan=backend_plan(2,&serde_json::to_string(&o).unwrap(),Some(&actual));plan.target.model_sha256=OUTPROJ_N128_MODEL.into();
        validate_plan_backend(&plan,&actual).unwrap();plan.schema=1;assert!(validate_plan_backend(&plan,&actual).is_err());plan.schema=2;
        plan.target.model_sha256="00".repeat(32);assert!(validate_plan_backend(&plan,&actual).is_err());plan.target.model_sha256=OUTPROJ_N128_MODEL.into();
        for identity in [None,Some("wrong".into())] {let mut mismatch=actual.clone();mismatch.outproj_n128_artifact=identity;assert!(validate_plan_backend(&plan,&mismatch).is_err());}
        plan.backend_build.as_mut().unwrap().outproj_n128_artifact=None;assert!(validate_plan_backend(&plan,&actual).is_err());
    }
    #[test]
    fn qkv_n128_requires_base_and_explicit_binding(){
        assert!(!qkv_n128_value(None).unwrap());
        for v in ["0","1"]{validate_value(QKV_N128_KEY,v).unwrap();}
        assert!(qkv_n128_value(Some("2")).is_err());
        assert!(validate_qkv_n128_options(true,false,None).is_err());
        let mut o=HashMap::new();assert!(validate_qkv_n128_options(true,true,Some(&o)).is_err());
        o.insert(QKV_N128_KEY.into(),"1".into());validate_qkv_n128_options(true,true,Some(&o)).unwrap();
        validate_qkv_n128_options(false,false,None).unwrap();
        let mut a=backend_build();a.capabilities.attention_q64_serial=true;a.strict_attention_artifact=Some("strict".into());a.qkv_immutable_artifact=Some("n128-combined".into());
        let mut o=strict_overrides();o.insert("KATAGO_CUDA_GEMM_LAYOUT".into(),"tn".into());o.insert(QKV_IMMUTABLE_KEY.into(),"1".into());o.insert(QKV_N128_KEY.into(),"1".into());
        let text=serde_json::to_string(&o).unwrap();validate_plan_backend(&backend_plan(2,&text,Some(&a)),&a).unwrap();
        for value in [None,Some("wrong".into())]{let mut b=a.clone();b.qkv_immutable_artifact=value;assert!(validate_plan_backend(&backend_plan(2,&text,Some(&b)),&a).is_err());}
        assert!(validate_plan_backend(&backend_plan(1,&text,Some(&a)),&a).is_err());
    }
    #[test]
    fn qkv_immutable_default_and_binding(){
        assert!(!qkv_immutable_value(None).unwrap());
        for v in ["0","1"]{validate_value(QKV_IMMUTABLE_KEY,v).unwrap();assert_eq!(qkv_immutable_value(Some(v)).unwrap(),v=="1");}
        for v in ["","true","2"]{assert!(qkv_immutable_value(Some(v)).is_err());assert!(validate_value(QKV_IMMUTABLE_KEY,v).is_err());}
        validate_qkv_immutable_options(true,true,Some("tn"),None).unwrap();
        assert!(validate_qkv_immutable_options(true,false,Some("tn"),None).is_err());
        assert!(validate_qkv_immutable_options(true,true,Some("nn_k384"),None).is_err());
        let mut o=strict_overrides();o.insert("KATAGO_CUDA_GEMM_LAYOUT".into(),"tn".into());
        assert!(validate_qkv_immutable_options(true,true,Some("tn"),Some(&o)).is_err());
        o.insert(QKV_IMMUTABLE_KEY.into(),"0".into());validate_qkv_immutable_options(false,true,Some("tn"),Some(&o)).unwrap();
        assert!(validate_qkv_immutable_options(true,true,Some("tn"),Some(&o)).is_err());
        o.insert(QKV_IMMUTABLE_KEY.into(),"1".into());validate_qkv_immutable_options(true,true,Some("tn"),Some(&o)).unwrap();
    }
    #[test]
    fn qkv_immutable_schema_and_artifact(){
        let mut a=backend_build();a.capabilities.attention_q64_serial=true;a.strict_attention_artifact=Some("strict".into());a.qkv_immutable_artifact=Some("qkv".into());
        let mut o=strict_overrides();o.insert("KATAGO_CUDA_GEMM_LAYOUT".into(),"tn".into());o.insert(QKV_IMMUTABLE_KEY.into(),"1".into());let text=serde_json::to_string(&o).unwrap();
        validate_plan_backend(&backend_plan(2,&text,Some(&a)),&a).unwrap();
        assert!(validate_plan_backend(&backend_plan(1,&text,Some(&a)),&a).unwrap_err().contains("requires schema 2"));
        for value in [None,Some("wrong".into())]{let mut e=a.clone();e.qkv_immutable_artifact=value;assert!(validate_plan_backend(&backend_plan(2,&text,Some(&e)),&a).unwrap_err().contains("qkv_immutable_artifact"));assert!(validate_plan_backend(&backend_plan(2,&text,Some(&a)),&e).is_err());}
        let mut legacy=a.clone();legacy.qkv_immutable_artifact=None;let old=serde_json::to_string(&strict_overrides()).unwrap();validate_plan_backend(&backend_plan(2,&old,Some(&legacy)),&a).unwrap();
        o.insert("KATAGO_CUDA_GEMM_LAYOUT".into(),"nn_k384".into());let text=serde_json::to_string(&o).unwrap();assert!(validate_plan_backend(&backend_plan(2,&text,Some(&a)),&a).unwrap_err().contains("TN QKV"));
    }

}

pub(crate) const FFN_COMPACT_KEY: &str = "KATAGO_CUDA_FFN_COMPACT_R1";
pub(crate) const FFN_COMPACT_MODEL: &str = "1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6";
const COMPACT_DEPS: &[(&str,&str)] = &[("KATAGO_CUDA_DUALFFN","1"),("KATAGO_CUDA_CUBLASLT","1"),("KATAGO_CUDA_GEMM_LAYOUT","tn"),("KATAGO_CUDA_NOGRAPH","1"),("KATAGO_CUDA_FUSION","none")];
fn compact_value(v:Option<&str>)->Result<bool,String> { match v { None|Some("0")=>Ok(false),Some("1")=>Ok(true),Some(v)=>Err(format!("invalid {FFN_COMPACT_KEY}: {v}")) } }
fn validate_compact_options(enabled:bool,get:&impl Fn(&str)->Option<String>,installed:Option<&HashMap<String,String>>)->Result<(),String> {
    if !enabled { return Ok(()); }
    for &(k,v) in COMPACT_DEPS {
        if get(k).as_deref()!=Some(v) { return Err(format!("{FFN_COMPACT_KEY}=1 requires {k}={v}")); }
    }
    if let Some(o)=installed {
        for (k,v) in COMPACT_DEPS.iter().copied().chain(std::iter::once((FFN_COMPACT_KEY,"1"))) {
            if o.get(k).map(String::as_str)!=Some(v) { return Err(format!("{FFN_COMPACT_KEY}=1 requires {k}={v} explicitly bound by the installed plan")); }
        }
    } Ok(())
}
pub(crate) fn ffn_compact_requested()->Result<bool,String> {
    let enabled=compact_value(tactic_var(FFN_COMPACT_KEY).ok().as_deref())?;
    validate_compact_options(enabled,&|k|tactic_var(k).ok(),INSTALLED.get().map(|i|&i.overrides))?;
    if enabled {
        #[cfg(feature="cuda")]let artifact=crate::backends::cuda::ffn_compact::fingerprint();
        #[cfg(not(feature="cuda"))]let artifact:Option<String>=None;
        if artifact.is_none() { return Err("FFN compact requires valid compiled ffn_compact_artifact".into()); }
    } Ok(enabled)
}

pub const OUTPROJ_N128_KEY: &str = "KATAGO_CUDA_OUTPROJ_CLASSIC_N128_R1";
pub const OUTPROJ_N128_MODEL: &str = "1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6";
const OUTPROJ_DEPS: &[(&str,&str)] = &[("KATAGO_CUDA_GEMM_LAYOUT","tn"),("KATAGO_CUDA_NOGRAPH","1"),("KATAGO_CUDA_SPLITK","0"),("KATAGO_CUDA_PADBATCH","0"),("KATAGO_CUDA_FUSION","none"),("KATAGO_CUDA_CUBLASLT","1")];
fn outproj_value(value:Option<&str>)->Result<bool,String> { match value {None|Some("0")=>Ok(false),Some("1")=>Ok(true),Some(v)=>Err(format!("invalid {OUTPROJ_N128_KEY}: {v}"))} }
fn validate_outproj_options(enabled:bool,get:&impl Fn(&str)->Option<String>,installed:Option<&HashMap<String,String>>)->Result<(),String> {
    if !enabled {return Ok(());}
    for &(key,value) in OUTPROJ_DEPS {
        if get(key).as_deref()!=Some(value) {return Err(format!("{OUTPROJ_N128_KEY}=1 requires {key}={value}"));}
    }
    if let Some(overrides)=installed {
        for (key,value) in OUTPROJ_DEPS.iter().copied().chain(std::iter::once((OUTPROJ_N128_KEY,"1"))) {
            if overrides.get(key).map(String::as_str)!=Some(value) {return Err(format!("{OUTPROJ_N128_KEY}=1 requires {key}={value} explicitly bound by the installed plan"));}
        }
    } Ok(())
}
pub fn outproj_n128_requested()->Result<bool,String> {
    let enabled=outproj_value(tactic_var(OUTPROJ_N128_KEY).ok().as_deref())?;
    validate_outproj_options(enabled,&|key|tactic_var(key).ok(),INSTALLED.get().map(|i|&i.overrides))?;
    if enabled {
        #[cfg(feature="cuda")]let artifact=crate::backends::cuda::outproj_n128::fingerprint();
        #[cfg(not(feature="cuda"))]let artifact:Option<String>=None;
        if artifact.is_none() {return Err("outproj N128 requires valid compiled outproj_n128_artifact".into());}
    } Ok(enabled)
}
pub fn validate_outproj_model_sha(sha:&str)->Result<(),String> {
    if outproj_n128_requested()? && !sha.eq_ignore_ascii_case(OUTPROJ_N128_MODEL) {return Err("outproj N128 source model SHA mismatch".into());} Ok(())
}
