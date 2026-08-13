//! Test helpers.
//!
//! Corresponds to `cpp/core/test.h` and `cpp/core/test.cpp`.

use std::fmt::Write;

/// Assert a condition, reporting file and line on failure.
///
/// Unlike Rust's standard `assert!`, this macro is intended to always be
/// enabled, mirroring KataGo's C++ `testAssert` macro.
#[macro_export]
macro_rules! test_assert {
    ($cond:expr) => {
        if !$cond {
            panic!(
                "Failed test assert: {}\nfile: {}\nline: {}",
                stringify!($cond),
                file!(),
                line!()
            );
        }
    };
}

/// Compare two multi-line strings by trimming each line and the whole string.
///
/// Mirrors `TestCommon::expect` from KataGo's C++ test helper. If the strings
/// differ, panics with a detailed diff showing the first mismatching line.
pub fn expect_lines_match(name: &str, actual: &str, expected: &str) {
    let actual_lines: Vec<&str> = actual.trim().lines().collect();
    let expected_lines: Vec<&str> = expected.trim().lines().collect();

    let mut first_diff: Option<usize> = None;
    for i in 0..actual_lines.len().max(expected_lines.len()) {
        let a = actual_lines.get(i).map(|s| s.trim()).unwrap_or("");
        let e = expected_lines.get(i).map(|s| s.trim()).unwrap_or("");
        if a != e {
            first_diff = Some(i);
            break;
        }
    }

    if let Some(first_diff) = first_diff {
        let mut msg = String::new();
        writeln!(msg, "Expect test failure!").unwrap();
        writeln!(msg, "{name}").unwrap();
        writeln!(
            msg,
            "Expected==============================================================="
        )
        .unwrap();
        writeln!(msg, "{expected}").unwrap();
        writeln!(
            msg,
            "Got===================================================================="
        )
        .unwrap();
        writeln!(msg, "{actual}").unwrap();
        writeln!(
            msg,
            "======================================================================="
        )
        .unwrap();
        writeln!(msg, "First line different (0-indexed) = {first_diff}").unwrap();

        let actual_line = actual_lines.get(first_diff).copied().unwrap_or("");
        let expected_line = expected_lines.get(first_diff).copied().unwrap_or("");
        writeln!(msg, "Actual  : {actual_line}").unwrap();
        writeln!(msg, "Expected: {expected_line}").unwrap();

        let mut char_diff = 0;
        let actual_bytes = actual_line.as_bytes();
        let expected_bytes = expected_line.as_bytes();
        while char_diff < actual_bytes.len()
            && char_diff < expected_bytes.len()
            && actual_bytes[char_diff] == expected_bytes[char_diff]
        {
            char_diff += 1;
        }
        writeln!(msg, "Char {char_diff} differs").unwrap();

        panic!("{msg}");
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expect_lines_match_trims() {
        expect_lines_match(
            "trim test",
            "  hello world  \n  foo bar  \n",
            "hello world\nfoo bar",
        );
    }

    #[test]
    #[should_panic(expected = "Expect test failure!")]
    fn test_expect_lines_match_panics_on_diff() {
        expect_lines_match("diff test", "foo\nbar", "foo\nbaz");
    }

    #[test]
    fn test_test_assert_passes() {
        test_assert!(1 + 1 == 2);
    }
}
