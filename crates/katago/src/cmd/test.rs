//! Run internal integration tests.
//!
//! Corresponds to the various `run*tests` helpers in `cpp/command/runtests.cpp`.
//!
//! The C++ file defines many specialized test commands (runtests, runoutputtests,
//! runsearchtests, runselfplayinittests, etc.) that exercise the C++ test suite.
//! In the Rust port the equivalent tests live in each crate's `#[cfg(test)]`
//! modules and are normally executed via `cargo test`. This command provides a
//! placeholder that reports the recommended way to run those tests.

#![allow(dead_code)]

use std::io::Write;

use clap::Parser;

use kata_core::global::StringError;

/// CLI arguments for the `runtests` subcommand.
///
/// The original C++ command takes no arguments.
#[derive(Parser, Debug, Clone)]
struct RuntestsArgs {
    /// Forwarded arguments are accepted for compatibility but ignored.
    #[arg(allow_hyphen_values = true, trailing_var_arg = true)]
    forwarded: Vec<String>,
}

/// Public CLI entry point for `runtests`.
pub fn runtests(args: &[String]) -> i32 {
    let mut out = std::io::stdout();
    match runtests_impl(args, &mut out) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {}", e);
            1
        }
    }
}

fn runtests_impl(args: &[String], out: &mut dyn Write) -> Result<(), StringError> {
    let _parsed =
        RuntestsArgs::try_parse_from(std::iter::once(&"runtests".to_string()).chain(args.iter()))
            .map_err(|e| StringError::new(format!("Argument error: {}", e)))?;

    writeln!(out, "Run the Rust test suite with: cargo test --workspace")
        .map_err(|e| StringError::new(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtests_reports_cargo_test() {
        let mut out = Vec::new();
        let result = runtests_impl(&[], &mut out);
        assert!(result.is_ok());
        let output = String::from_utf8_lossy(&out);
        assert!(output.contains("cargo test --workspace"));
    }

    #[test]
    fn test_runtests_accepts_forwarded_args() {
        let mut out = Vec::new();
        let result = runtests_impl(&["--".to_string(), "some-filter".to_string()], &mut out);
        assert!(result.is_ok());
    }
}
