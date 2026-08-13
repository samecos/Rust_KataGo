//! Write training data from SGFs or other sources.
//!
//! Corresponds to `MainCmds::writetrainingdata` in `cpp/command/writetrainingdata.cpp`.
//!
//! The C++ implementation loads SGF files, runs neural net evaluation, and
//! writes KataGo training data (policy/value targets, etc.). The Rust port does
//! not yet implement this pipeline, so this command reports that it is
//! unavailable.

use kata_core::global::StringError;

/// Public CLI entry point.
pub fn writetrainingdata(_args: &[String]) -> i32 {
    match writetrainingdata_impl() {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {}", e);
            1
        }
    }
}

fn writetrainingdata_impl() -> Result<(), StringError> {
    println!(
        "writetrainingdata: converting SGFs or other sources into KataGo training data is not yet implemented in the Rust port."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_writetrainingdata_reports_not_implemented() {
        assert_eq!(writetrainingdata(&[]), 0);
    }
}
