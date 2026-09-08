fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::process::Command;
    let revision = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".to_owned());
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .map(|o| !o.status.success() || !o.stdout.is_empty())
        .unwrap_or(true);
    println!(
        "cargo:rustc-env=KATAGO_WORKER_ENGINE_REVISION={revision}{}",
        if dirty { "+dirty" } else { "" }
    );
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/index");
    let mut config = tonic_prost_build::Config::new();
    config.protoc_executable(protoc_bin_vendored::protoc_bin_path()?);
    // Connect is the RPC name, so disable the conflicting transport constructor.
    tonic_prost_build::configure()
        .build_transport(false)
        .compile_with_config(config, &["proto/worker.proto"], &["proto"])?;
    println!("cargo:rerun-if-changed=proto/worker.proto");
    Ok(())
}
