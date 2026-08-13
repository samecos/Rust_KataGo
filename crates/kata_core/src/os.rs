//! Operating-system detection constants.
//!
//! Corresponds to `cpp/core/os.h`. In Rust these are expressed with `cfg!`
//! rather than C preprocessor macros.

/// True when compiling for a Windows target.
pub const IS_WINDOWS: bool = cfg!(windows);

/// True when compiling for a Unix-like target (including Linux and macOS).
pub const IS_UNIX_OR_APPLE: bool = cfg!(unix);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exactly_one_os_family() {
        assert_ne!(IS_WINDOWS, IS_UNIX_OR_APPLE);
    }
}
