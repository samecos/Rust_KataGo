//! Sandbox command for backend-specific smoke tests.
//!
//! Corresponds to `MainCmds::sandbox` in `cpp/command/sandbox.cpp`.
//!
//! The C++ implementation is currently a TensorRT-only ONNX smoke test guarded
//! by `USE_TENSORRT_BACKEND`. The Rust port does not implement the TensorRT
//! backend, so this command simply reports that it is unavailable.

use kata_core::global::StringError;

/// Public CLI entry point.
pub fn sandbox(_args: &[String]) -> i32 {
    match sandbox_impl() {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {}", e);
            1
        }
    }
}

fn sandbox_impl() -> Result<(), StringError> {
    println!(
        "sandbox: ONNX smoke test requires the TensorRT backend (build with -DUSE_BACKEND=TENSORRT)."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sandbox_reports_tensorrt_only() {
        assert_eq!(sandbox(&[]), 0);
    }
}
