//! Single command-line preprocessing for the command loop.
//!
//! Corresponds to `cpp/core/commandloop.h` and `cpp/core/commandloop.cpp`.

use crate::global::trim;

/// Preprocess a raw command line for the command loop.
///
/// The returned string:
/// * contains only ASCII printables (`0x20..=0x7E`) and tabs,
/// * has any trailing `#` comment removed,
/// * has tabs converted to spaces,
/// * is trimmed of leading/trailing whitespace.
pub fn process_single_command_line(s: &str) -> String {
    let mut line = trim(s)
        .chars()
        .filter(|c| {
            let cp = *c as u32;
            (32..=126).contains(&cp) || *c == '\t'
        })
        .collect::<String>();

    // Remove comments.
    if let Some(pos) = line.find('#') {
        line.truncate(pos);
    }

    // Convert tabs to spaces.
    let line = line.replace('\t', " ");

    trim(&line).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trim_and_comment_removal() {
        assert_eq!(
            process_single_command_line("  genmove b # this is a comment  "),
            "genmove b"
        );
    }

    #[test]
    fn test_tabs_to_spaces() {
        assert_eq!(
            process_single_command_line("\tplay\tB\tQ16\t"),
            "play B Q16"
        );
    }

    #[test]
    fn test_non_ascii_filtered() {
        assert_eq!(process_single_command_line("play B 你好\r\n"), "play B");
    }

    #[test]
    fn test_control_chars_filtered() {
        assert_eq!(
            process_single_command_line("\x01showboard\x7F"),
            "showboard"
        );
    }

    #[test]
    fn test_empty_and_whitespace_only() {
        assert_eq!(process_single_command_line("   "), "");
        assert_eq!(process_single_command_line("# comment only"), "");
    }
}
