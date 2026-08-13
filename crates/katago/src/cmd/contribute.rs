//! Distributed training contribution command.
//!
//! Corresponds to `MainCmds::contribute` in `cpp/command/contribute.cpp`.
//!
//! The C++ implementation is guarded by `BUILD_DISTRIBUTED`; in non-distributed
//! builds the command simply reports that distributed training is not enabled.
//! The Rust port keeps the same behavior.

use std::sync::Arc;

use kata_core::global::StringError;
use kata_core::logger::{Logger, LoggerOptions};
use kata_distributed::client::{Connection, Url};

/// Public CLI entry point.
pub fn contribute(_args: &[String]) -> i32 {
    match contribute_impl() {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {}", e);
            1
        }
    }
}

fn contribute_impl() -> Result<(), StringError> {
    // Keep the distributed client crate linked: instantiate a stub connection.
    let _conn = Connection::new(
        "",
        "",
        "",
        "",
        &Url::default(),
        "",
        false,
        Arc::new(Logger::new(LoggerOptions::default(), None)),
    );
    println!("This version of KataGo was NOT compiled with support for distributed training.");
    println!(
        "Compile with -DBUILD_DISTRIBUTED=1 in CMake, and/or see notes at https://github.com/lightvector/KataGo#compiling-katago"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contribute_reports_not_supported() {
        assert_eq!(contribute(&[]), 0);
    }
}
