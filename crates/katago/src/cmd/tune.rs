//! GPU tuning command for OpenCL.
//!
//! Corresponds to `MainCmds::tuner` in `cpp/command/tune.cpp`.
//!
//! The C++ implementation is guarded by `USE_OPENCL_BACKEND`; in non-OpenCL
//! builds the command simply reports that it does nothing. The Rust port keeps
//! the same behavior: it parses the expected command-line flags and then prints
//! a message explaining that OpenCL tuning is not implemented.

#![allow(dead_code)]

use clap::Parser;

use std::io::Write;

use kata_core::global::StringError;

use crate::cli::CommonArgs;

/// CLI arguments for the `tune` subcommand.
#[derive(Parser, Debug, Clone)]
struct TuneArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// Filename to output tuning configuration to.
    #[arg(long = "output", value_name = "FILE")]
    output: Option<String>,

    /// Specific GPU/device number(s) to tune, comma-separated (default all).
    #[arg(long = "gpus", value_name = "GPUS")]
    gpus: Option<String>,

    /// Width of board to tune for.
    #[arg(long = "xsize", value_name = "INT", default_value_t = 19)]
    x_size: i32,

    /// Height of board to tune for.
    #[arg(long = "ysize", value_name = "INT", default_value_t = 19)]
    y_size: i32,

    /// Test FP16? true|false|auto (default auto).
    #[arg(long = "testFP16", value_name = "BOOL_OR_AUTO", default_value = "auto")]
    test_fp16: String,

    /// Test FP16 storage? true|false|auto (default auto).
    #[arg(
        long = "testFP16Storage",
        value_name = "BOOL_OR_AUTO",
        default_value = "auto"
    )]
    test_fp16_storage: String,

    /// Test FP16 compute? true|false|auto (default auto).
    #[arg(
        long = "testFP16Compute",
        value_name = "BOOL_OR_AUTO",
        default_value = "auto"
    )]
    test_fp16_compute: String,

    /// Test FP16 tensor cores? true|false|auto (default auto).
    #[arg(
        long = "testFP16TensorCores",
        value_name = "BOOL_OR_AUTO",
        default_value = "auto"
    )]
    test_fp16_tensor_cores: String,

    /// Batch size to tune for.
    #[arg(long = "batchsize", value_name = "INT", default_value_t = 8)]
    batch_size: i32,

    /// Winograd 3x3 tile size.
    #[arg(long = "winograd3x3tilesize", value_name = "INT", default_value_t = 4)]
    winograd_3x3_tile_size: i32,

    /// Test more possible configurations.
    #[arg(long = "full")]
    full: bool,

    /// Verbosely print out errors for configurations that fail.
    #[arg(long = "verboseErrors")]
    verbose_errors: bool,

    /// Verbosely print out tuner results even if they don't improve the best.
    #[arg(long = "verboseTuner")]
    verbose_tuner: bool,
}

/// Parse tune flags and report that OpenCL tuning is not implemented.
pub fn tune(args: &[String]) -> i32 {
    let mut out = std::io::stdout();
    match tune_impl(args, &mut out) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {}", e);
            1
        }
    }
}

fn tune_impl(args: &[String], out: &mut dyn Write) -> Result<(), StringError> {
    let _parsed = TuneArgs::try_parse_from(std::iter::once(&"tune".to_string()).chain(args.iter()))
        .map_err(|e| StringError::new(format!("Argument error: {}", e)))?;

    writeln!(
        out,
        "Currently this command only does anything for the OpenCL version of KataGo"
    )
    .map_err(|e| StringError::new(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tune_runs_and_reports_opencl_only() {
        let mut out = Vec::new();
        let result = tune_impl(&[], &mut out);
        assert!(result.is_ok());
        let output = String::from_utf8_lossy(&out);
        assert!(output.contains("OpenCL"));
    }

    #[test]
    fn test_tune_parses_flags() {
        let args = TuneArgs::parse_from([
            "tune",
            "--model",
            "model.bin.gz",
            "--output",
            "tune.cfg",
            "--gpus",
            "0,1",
            "--batchsize",
            "16",
        ]);
        assert_eq!(args.common.model, Some("model.bin.gz".to_string()));
        assert_eq!(args.output, Some("tune.cfg".to_string()));
        assert_eq!(args.gpus, Some("0,1".to_string()));
        assert_eq!(args.batch_size, 16);
    }
}
