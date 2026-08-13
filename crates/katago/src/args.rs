//! UTF-8 command-line argument and console helpers.
//!
//! Corresponds to `cpp/core/mainargs.h` and `cpp/core/mainargs.cpp`.

/// Returns the program's command-line arguments as UTF-8 strings.
///
/// On Windows this uses the wide-character command line and converts it to
/// UTF-8, matching the behavior of the C++ implementation. On other platforms
/// it simply returns `std::env::args`.
pub fn get_command_line_args_utf8() -> Vec<String> {
    std::env::args().collect()
}

/// Prepares the standard output and error streams for UTF-8 text.
///
/// On Windows this sets the console code page to UTF-8. On Unix-like systems
/// this is a no-op because UTF-8 is already the default.
pub fn make_cout_and_cerr_accept_utf8() {
    #[cfg(windows)]
    {
        const CP_UTF8: u32 = 65001;

        unsafe extern "system" {
            fn SetConsoleOutputCP(code_page: u32) -> i32;
            fn SetConsoleCP(code_page: u32) -> i32;
        }

        unsafe {
            // We ignore errors here; if the process does not have a console,
            // there is nothing useful to do.
            let _ = SetConsoleOutputCP(CP_UTF8);
            let _ = SetConsoleCP(CP_UTF8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_command_line_args_contains_program() {
        let args = get_command_line_args_utf8();
        assert!(
            !args.is_empty(),
            "command-line args should contain at least the program name"
        );
    }

    #[test]
    fn test_make_cout_and_cerr_accept_utf8_is_callable() {
        // Should not panic; on Windows it may set the console code page.
        make_cout_and_cerr_accept_utf8();
    }
}
