//! Starting-position and SGF sampling commands.
//!
//! Corresponds to the subcommands in `cpp/command/startposes.cpp`:
//! `samplesgfs`, `dataminesgfs`, `trystartposes`, `viewstartposes`,
//! `checksgfhintpolicy`, and `genposesfromselfplayinit`.
//!
//! The C++ implementations perform SGF sampling, data mining for opening
//! positions, and related utilities. The Rust port does not yet implement this
//! pipeline, so all of these subcommands are placeholders.

use kata_core::global::StringError;

fn report_not_implemented(name: &str) -> Result<(), StringError> {
    println!(
        "{}: Starting-position / SGF sampling support is not yet implemented in the Rust port.",
        name
    );
    Ok(())
}

/// Public CLI entry point for `samplesgfs`.
pub fn samplesgfs(_args: &[String]) -> i32 {
    run("samplesgfs")
}

/// Public CLI entry point for `dataminesgfs`.
pub fn dataminesgfs(_args: &[String]) -> i32 {
    run("dataminesgfs")
}

/// Public CLI entry point for `trystartposes`.
pub fn trystartposes(_args: &[String]) -> i32 {
    run("trystartposes")
}

/// Public CLI entry point for `viewstartposes`.
pub fn viewstartposes(_args: &[String]) -> i32 {
    run("viewstartposes")
}

/// Public CLI entry point for `checksgfhintpolicy`.
pub fn checksgfhintpolicy(_args: &[String]) -> i32 {
    run("checksgfhintpolicy")
}

/// Public CLI entry point for `genposesfromselfplayinit`.
pub fn genposesfromselfplayinit(_args: &[String]) -> i32 {
    run("genposesfromselfplayinit")
}

fn run(name: &str) -> i32 {
    match report_not_implemented(name) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {}", e);
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_samplesgfs_reports_not_implemented() {
        assert_eq!(samplesgfs(&[]), 0);
    }

    #[test]
    fn test_dataminesgfs_reports_not_implemented() {
        assert_eq!(dataminesgfs(&[]), 0);
    }

    #[test]
    fn test_trystartposes_reports_not_implemented() {
        assert_eq!(trystartposes(&[]), 0);
    }

    #[test]
    fn test_viewstartposes_reports_not_implemented() {
        assert_eq!(viewstartposes(&[]), 0);
    }

    #[test]
    fn test_checksgfhintpolicy_reports_not_implemented() {
        assert_eq!(checksgfhintpolicy(&[]), 0);
    }

    #[test]
    fn test_genposesfromselfplayinit_reports_not_implemented() {
        assert_eq!(genposesfromselfplayinit(&[]), 0);
    }
}
