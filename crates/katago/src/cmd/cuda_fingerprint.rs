//! `cuda-fingerprint` 子命令：打印当前 CUDA 设备指纹与模型 SHA-256
//! （JSON），供 `scripts/autotune.py` 生成 plan 的 target 字段。

use clap::Parser;

#[derive(Parser, Debug)]
struct FingerprintArgs {
    /// 模型文件（给出则同时打印其 SHA-256）。
    #[arg(long = "model", value_name = "FILE")]
    model: Option<String>,
}

#[cfg(feature = "cuda")]
pub fn cuda_fingerprint(args: &[String]) -> i32 {
    let args = FingerprintArgs::parse_from(std::iter::once("cuda-fingerprint").chain(args.iter().map(|s| s.as_str())));
    let fp = match kata_nn::backends::cuda::current_device_fingerprint(None) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    let mut out = serde_json::json!({
        "gpu_name": fp.gpu_name,
        "architecture": format!("sm_{}", fp.compute_capability.replace('.', "")),
        "compute_capability": fp.compute_capability,
        "sm_count": fp.sm_count,
        "l2_cache_bytes": fp.l2_cache_bytes,
    });
    if let Some(m) = &args.model {
        match kata_nn::tactic_plan::sha256_file(std::path::Path::new(m)) {
            Ok(sha) => out["model_sha256"] = serde_json::json!(sha),
            Err(e) => {
                eprintln!("{e}");
                return 1;
            }
        }
    }
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
    0
}

#[cfg(not(feature = "cuda"))]
pub fn cuda_fingerprint(_args: &[String]) -> i32 {
    eprintln!("cuda-fingerprint requires building with --features cuda");
    1
}
