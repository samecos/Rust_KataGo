//! Generic utility functions used throughout KataGo.
//!
//! This module corresponds to `cpp/core/global.h` and `cpp/core/global.cpp`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Base error type carrying a string message.
#[derive(Debug, Error)]
#[error("{message}")]
pub struct StringError {
    pub message: String,
}

impl StringError {
    pub fn new<S: Into<String>>(message: S) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// I/O error.
#[derive(Debug, Error)]
#[error("IOError: {0}")]
pub struct IOError(pub String);

impl From<StringError> for IOError {
    fn from(e: StringError) -> Self {
        IOError(e.message)
    }
}

impl From<&str> for IOError {
    fn from(s: &str) -> Self {
        IOError(s.to_string())
    }
}

/// Error for config parsing failures.
#[derive(Debug, Error)]
#[error("ConfigParsingError: {0}")]
pub struct ConfigParsingError(pub String);

impl From<&str> for ConfigParsingError {
    fn from(s: &str) -> Self {
        ConfigParsingError(s.to_string())
    }
}

/// Error for invalid parameter values.
#[derive(Debug, Error)]
#[error("ValueError: {0}")]
pub struct ValueError(pub String);

impl From<&str> for ValueError {
    fn from(s: &str) -> Self {
        ValueError(s.to_string())
    }
}

/// Error for command-line argument handling.
#[derive(Debug, Error)]
#[error("CommandError: {0}")]
pub struct CommandError(pub String);

impl From<&str> for CommandError {
    fn from(s: &str) -> Self {
        CommandError(s.to_string())
    }
}

// ---------------------------------------------------------------------------
// Fatal errors
// ---------------------------------------------------------------------------

/// Report a fatal error message and exit the process.
///
/// Corresponds to `Global::fatalError`. Prefer returning a `Result` in new
/// Rust code; this function exists for compatibility with C++ control flow.
pub fn fatal_error<S: AsRef<str>>(s: S) -> ! {
    eprintln!("\nFATAL ERROR:\n{}", s.as_ref());
    std::process::exit(1);
}

/// Marker for code paths that should be unreachable.
#[macro_export]
macro_rules! assert_unreachable {
    () => {
        $crate::global::fatal_error(format!(
            "BUG? Reached asserted-unreachable point of the code: {}:{}",
            file!(),
            line!()
        ))
    };
}

// ---------------------------------------------------------------------------
// Time
// ---------------------------------------------------------------------------

/// Get a string describing the current date, suitable for filenames.
///
/// Format: `YYYY-MM-DD_HH-MM-SS`.
pub fn get_date_string() -> String {
    let now = SystemTime::now();
    let duration = now.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = duration.as_secs() as i64;
    let datetime = time_from_seconds(secs);
    format!(
        "{:04}-{:02}-{:02}_{:02}-{:02}-{:02}",
        datetime.year,
        datetime.month,
        datetime.day,
        datetime.hour,
        datetime.minute,
        datetime.second
    )
}

pub(crate) struct DateTime {
    pub year: i32,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
}

pub(crate) fn time_from_seconds(mut secs: i64) -> DateTime {
    // Days since 1970-01-01.
    let mut days = secs / 86400;
    secs %= 86400;
    if secs < 0 {
        days -= 1;
        secs += 86400;
    }

    let mut year = 1970i32;
    loop {
        let year_len = if is_leap_year(year) { 366 } else { 365 };
        if days >= year_len as i64 {
            days -= year_len as i64;
            year += 1;
        } else if days < 0 {
            year -= 1;
            let prev_len = if is_leap_year(year) { 366 } else { 365 };
            days += prev_len as i64;
        } else {
            break;
        }
    }

    let month_lengths = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1u8;
    for (idx, len) in month_lengths.iter().enumerate() {
        if days < *len as i64 {
            month = (idx + 1) as u8;
            break;
        }
        days -= *len as i64;
        month = (idx + 2) as u8;
    }

    let day = (days + 1) as u8;
    let hour = (secs / 3600) as u8;
    secs %= 3600;
    let minute = (secs / 60) as u8;
    let second = (secs % 60) as u8;

    DateTime {
        year,
        month,
        day,
        hour,
        minute,
        second,
    }
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

// ---------------------------------------------------------------------------
// Conversions to string
// ---------------------------------------------------------------------------

pub fn bool_to_string(b: bool) -> String {
    if b {
        "true".to_string()
    } else {
        "false".to_string()
    }
}

pub fn char_to_string(c: char) -> String {
    c.to_string()
}

pub fn int_to_string(x: i32) -> String {
    x.to_string()
}

pub fn float_to_string(x: f32) -> String {
    x.to_string()
}

pub fn double_to_string(x: f64) -> String {
    x.to_string()
}

pub fn double_to_string_high_precision(x: f64) -> String {
    format!("{:.17}", x)
}

pub fn int64_to_string(x: i64) -> String {
    x.to_string()
}

pub fn uint32_to_string(x: u32) -> String {
    x.to_string()
}

pub fn uint64_to_string(x: u64) -> String {
    x.to_string()
}

pub fn uint32_to_hex_string(x: u32) -> String {
    format!("{:08X}", x)
}

pub fn uint64_to_hex_string(x: u64) -> String {
    format!("{:016X}", x)
}

pub fn size_to_string(x: usize) -> String {
    x.to_string()
}

// ---------------------------------------------------------------------------
// Conversions from string
// ---------------------------------------------------------------------------

macro_rules! impl_try_parse {
    ($name:ident, $ty:ty, $err:literal) => {
        pub fn $name(s: &str) -> Result<$ty, IOError> {
            let trimmed = trim(s);
            trimmed
                .parse::<$ty>()
                .map_err(|_| IOError(format!("{}: {}", $err, s)))
        }
    };
}

impl_try_parse!(string_to_int, i32, "could not parse int");
impl_try_parse!(string_to_int64, i64, "could not parse int64");
impl_try_parse!(string_to_float, f32, "could not parse float");
impl_try_parse!(string_to_double, f64, "could not parse double");

pub fn try_string_to_int(s: &str) -> Option<i32> {
    trim(s).parse().ok()
}

pub fn try_string_to_int64(s: &str) -> Option<i64> {
    trim(s).parse().ok()
}

pub fn try_string_to_float(s: &str) -> Option<f32> {
    trim(s).parse().ok()
}

pub fn try_string_to_double(s: &str) -> Option<f64> {
    trim(s).parse().ok()
}

pub fn string_to_uint64(s: &str) -> Result<u64, IOError> {
    let trimmed = trim(s);
    if trimmed.starts_with('-') {
        return Err(IOError(format!("could not parse uint64: {}", s)));
    }
    trimmed
        .parse::<u64>()
        .map_err(|_| IOError(format!("could not parse uint64: {}", s)))
}

pub fn try_string_to_uint64(s: &str) -> Option<u64> {
    let trimmed = trim(s);
    if trimmed.starts_with('-') {
        return None;
    }
    trimmed.parse().ok()
}

pub fn string_to_bool(s: &str) -> Result<bool, IOError> {
    match to_lower(trim(s)).as_str() {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(IOError(format!("could not parse bool: {}", s))),
    }
}

pub fn try_string_to_bool(s: &str) -> Option<bool> {
    match to_lower(trim(s)).as_str() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

pub fn hex_string_to_uint64(s: &str) -> Result<u64, IOError> {
    if !s.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(IOError(format!("could not parse uint64 from hex: {}", s)));
    }
    u64::from_str_radix(s, 16)
        .map_err(|_| IOError(format!("could not parse uint64 from hex: {}", s)))
}

pub fn try_hex_string_to_uint64(s: &str) -> Option<u64> {
    if !s.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(s, 16).ok()
}

// ---------------------------------------------------------------------------
// String inspection
// ---------------------------------------------------------------------------

pub fn is_whitespace_char(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\r' | '\n' | '\x0B' | '\x0C')
}

pub fn is_whitespace(s: &str) -> bool {
    s.chars().all(is_whitespace_char)
}

pub fn is_prefix(s: &str, prefix: &str) -> bool {
    s.starts_with(prefix)
}

pub fn is_suffix(s: &str, suffix: &str) -> bool {
    s.ends_with(suffix)
}

pub fn chop_prefix<'a>(s: &'a str, prefix: &str) -> Result<&'a str, StringError> {
    s.strip_prefix(prefix).ok_or_else(|| {
        StringError::new(format!(
            "chopPrefix:\n{}\nis not a prefix of\n{}",
            prefix, s
        ))
    })
}

pub fn chop_suffix<'a>(s: &'a str, suffix: &str) -> Result<&'a str, StringError> {
    s.strip_suffix(suffix).ok_or_else(|| {
        StringError::new(format!(
            "chopSuffix:\n{}\nis not a suffix of\n{}",
            suffix, s
        ))
    })
}

pub fn trim(s: &str) -> &str {
    s.trim()
}

pub fn trim_with_delims<'a>(s: &'a str, delims: &str) -> &'a str {
    s.trim_matches(|c: char| delims.contains(c))
}

// ---------------------------------------------------------------------------
// Join / split
// ---------------------------------------------------------------------------

pub fn concat_slice(strs: &[&str], delim: &str) -> String {
    strs.join(delim)
}

pub fn concat_vec(strs: &[String], delim: &str) -> String {
    strs.join(delim)
}

pub fn concat_vec_range(strs: &[String], delim: &str, start: usize, end: usize) -> String {
    let start = start.min(strs.len());
    let end = end.min(strs.len());
    if start >= end {
        return String::new();
    }
    let slice = &strs[start..end];
    slice.join(delim)
}

pub fn concat_set(strs: &BTreeSet<String>, delim: &str) -> String {
    strs.iter().cloned().collect::<Vec<_>>().join(delim)
}

/// Split a string into tokens, trimming whitespace off each token.
pub fn split(s: &str) -> Vec<String> {
    s.split_whitespace().map(|t| t.to_string()).collect()
}

/// Split a string based on the given delimiter, without trimming.
pub fn split_by(s: &str, delim: char) -> Vec<String> {
    s.split(delim).map(|t| t.to_string()).collect()
}

// ---------------------------------------------------------------------------
// Case conversion and comparison
// ---------------------------------------------------------------------------

pub fn to_upper(s: &str) -> String {
    s.to_uppercase()
}

pub fn to_lower(s: &str) -> String {
    s.to_lowercase()
}

pub fn is_equal_case_insensitive(s0: &str, s1: &str) -> bool {
    s0.eq_ignore_ascii_case(s1)
}

// ---------------------------------------------------------------------------
// Formatted strings
// ---------------------------------------------------------------------------

/// Runtime `printf`-style formatting helper.
///
/// The C++ original used variadic formatting. Rust does not support runtime
/// format strings safely, so this macro forwards to `format!`.
#[macro_export]
macro_rules! strprintf {
    ($fmt:literal $(, $args:expr)*) => {
        format!($fmt $(, $args)*)
    };
}

// ---------------------------------------------------------------------------
// Digits
// ---------------------------------------------------------------------------

pub fn is_digit(c: char) -> bool {
    c.is_ascii_digit()
}

pub fn is_alpha(c: char) -> bool {
    c.is_ascii_alphabetic()
}

pub fn is_digits(s: &str) -> bool {
    is_digits_range(s, 0, s.len())
}

pub fn is_digits_range(s: &str, start: usize, end: usize) -> bool {
    if end <= start || end - start > 9 {
        return false;
    }
    let mut value: i64 = 0;
    for c in s[start..end.min(s.len())].chars() {
        if !is_digit(c) {
            return false;
        }
        value = value * 10 + i64::from(c as u8 - b'0');
    }
    (value & 0x7FFF_FFFF) == value
}

pub fn parse_digits(s: &str) -> Result<i32, IOError> {
    parse_digits_range(s, 0, s.len())
}

pub fn parse_digits_range(s: &str, start: usize, end: usize) -> Result<i32, IOError> {
    if end <= start {
        return Err(IOError(
            "Could not parse digits, end <= start, or empty string".to_string(),
        ));
    }
    if end - start > 9 {
        return Err(IOError(format!(
            "Could not parse digits, overflow: {}",
            &s[start..end.min(s.len())]
        )));
    }
    let mut value: i64 = 0;
    for c in s[start..end.min(s.len())].chars() {
        if !is_digit(c) {
            return Ok(0);
        }
        value = value * 10 + i64::from(c as u8 - b'0');
    }
    if (value & 0x7FFF_FFFF) != value {
        return Err(IOError(format!(
            "Could not parse digits, overflow: {}",
            &s[start..end.min(s.len())]
        )));
    }
    Ok(value as i32)
}

// ---------------------------------------------------------------------------
// Character set checks
// ---------------------------------------------------------------------------

pub fn string_chars_all_allowed(s: &str, allowed: &str) -> bool {
    s.chars().all(|c| allowed.contains(c))
}

// ---------------------------------------------------------------------------
// Comments
// ---------------------------------------------------------------------------

/// Strip `#` rest-of-line style comments from a string.
pub fn strip_comments(s: &str) -> String {
    if !s.contains('#') {
        return s.to_string();
    }
    s.lines()
        .map(|line| match line.find('#') {
            Some(pos) => &line[..pos],
            None => line,
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

// ---------------------------------------------------------------------------
// Key-value parsing
// ---------------------------------------------------------------------------

/// Parse comma- or newline-separated `key=value` pairs.
///
/// Whitespace around keys and values is trimmed. Duplicate keys raise an
/// `IOError`.
pub fn read_key_values(contents: &str) -> Result<BTreeMap<String, String>, IOError> {
    let mut map = BTreeMap::new();
    for line in contents.lines() {
        if line.is_empty() {
            continue;
        }
        for chunk in line.split(',') {
            if chunk.is_empty() {
                continue;
            }
            let Some(pos) = chunk.find('=') else {
                continue;
            };
            let key = trim(&chunk[..pos]);
            let value = trim(&chunk[pos + 1..]);
            if key.is_empty() {
                return Err(IOError(format!(
                    "readKeyValues: key value pair without key: {}",
                    line
                )));
            }
            if value.is_empty() {
                return Err(IOError(format!(
                    "readKeyValues: key value pair without value: {}",
                    line
                )));
            }
            if map.contains_key(key) {
                return Err(IOError(format!("readKeyValues: duplicate key: {}", key)));
            }
            map.insert(key.to_string(), value.to_string());
        }
    }
    Ok(map)
}

// ---------------------------------------------------------------------------
// Memory size parsing
// ---------------------------------------------------------------------------

/// Read a memory value such as `16G`, `256K`, `512MB`, or `1024B`.
pub fn read_mem(s: &str) -> Result<u64, IOError> {
    if s.len() < 2 {
        return Err(IOError(format!(
            "readMem: Could not parse amount of memory: {}",
            s
        )));
    }

    let (shift, numeric): (u32, &str) = if let Some(n) = s.strip_suffix("PB") {
        (50, n)
    } else if let Some(n) = s.strip_suffix("TB") {
        (40, n)
    } else if let Some(n) = s.strip_suffix("GB") {
        (30, n)
    } else if let Some(n) = s.strip_suffix("MB") {
        (20, n)
    } else if let Some(n) = s.strip_suffix("KB") {
        (10, n)
    } else if let Some(n) = s.strip_suffix('P') {
        (50, n)
    } else if let Some(n) = s.strip_suffix('T') {
        (40, n)
    } else if let Some(n) = s.strip_suffix('G') {
        (30, n)
    } else if let Some(n) = s.strip_suffix('M') {
        (20, n)
    } else if let Some(n) = s.strip_suffix('K') {
        (10, n)
    } else if let Some(n) = s.strip_suffix('B') {
        (0, n)
    } else {
        (0, s)
    };

    if !is_digits(numeric) {
        return Err(IOError(format!(
            "readMem: Could not parse amount of memory: {}",
            s
        )));
    }
    let value: u64 = numeric
        .parse()
        .map_err(|_| IOError(format!("readMem: Could not parse amount of memory: {}", s)))?;

    value.checked_shl(shift).ok_or_else(|| {
        IOError(format!(
            "readMem: Could not parse amount of memory (too large): {}",
            s
        ))
    })
}

// ---------------------------------------------------------------------------
// User interaction
// ---------------------------------------------------------------------------

/// Display a message and wait for the user to press Enter.
pub fn pause_for_key() {
    println!("Press any key to continue...");
    let mut buf = String::new();
    let _ = std::io::stdin().read_line(&mut buf);
}

// ---------------------------------------------------------------------------
// Rounding
// ---------------------------------------------------------------------------

pub fn round_static(x: f64, inverse_scale: f64) -> f64 {
    (x * inverse_scale).round() / inverse_scale
}

pub fn round_dynamic(x: f64, precision: i32) -> f64 {
    let absx = x.abs();
    if absx <= 1e-60 {
        return x;
    }
    let order = absx.log10().floor() as i32;
    let rounding_magnitude = order - precision;
    if rounding_magnitude >= 0 {
        return x.round();
    }
    let inverse_scale = 10f64.powi(-rounding_magnitude);
    round_static(x, inverse_scale)
}

// ---------------------------------------------------------------------------
// Container helpers
// ---------------------------------------------------------------------------

pub fn contains_char(s: &str, c: char) -> bool {
    s.contains(c)
}

pub fn contains<T: PartialEq>(slice: &[T], elt: &T) -> bool {
    slice.contains(elt)
}

pub fn contains_str(slice: &[String], elt: &str) -> bool {
    slice.iter().any(|s| s == elt)
}

pub fn index_of<T: PartialEq>(slice: &[T], elt: &T) -> Option<usize> {
    slice.iter().position(|x| x == elt)
}

pub fn index_of_str(slice: &[String], elt: &str) -> Option<usize> {
    slice.iter().position(|s| s == elt)
}

pub fn contains_key<K: Eq + std::hash::Hash, V>(map: &HashMap<K, V>, key: &K) -> bool {
    map.contains_key(key)
}

pub fn contains_key_btree<K: Ord, V>(map: &BTreeMap<K, V>, key: &K) -> bool {
    map.contains_key(key)
}

pub fn map_get<K: Eq + std::hash::Hash + fmt::Debug + Clone, V: Clone>(
    map: &HashMap<K, V>,
    key: &K,
) -> Result<V, IOError> {
    map.get(key)
        .cloned()
        .ok_or_else(|| IOError(format!("map_get: key {:?} not found", key)))
}

pub fn map_get_btree<K: Ord + fmt::Debug + Clone, V: Clone>(
    map: &BTreeMap<K, V>,
    key: &K,
) -> Result<V, IOError> {
    map.get(key)
        .cloned()
        .ok_or_else(|| IOError(format!("map_get: key {:?} not found", key)))
}

pub fn map_get_defaulting<K: Eq + std::hash::Hash, V: Clone>(
    map: &HashMap<K, V>,
    key: &K,
    default: &V,
) -> V {
    map.get(key).cloned().unwrap_or_else(|| default.clone())
}

pub fn map_get_defaulting_btree<K: Ord, V: Clone>(map: &BTreeMap<K, V>, key: &K, default: &V) -> V {
    map.get(key).cloned().unwrap_or_else(|| default.clone())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bool_to_string() {
        assert_eq!(bool_to_string(true), "true");
        assert_eq!(bool_to_string(false), "false");
    }

    #[test]
    fn test_int_to_string() {
        assert_eq!(int_to_string(-42), "-42");
        assert_eq!(uint64_to_string(u64::MAX), "18446744073709551615");
    }

    #[test]
    fn test_hex_strings() {
        assert_eq!(uint32_to_hex_string(0xABCD), "0000ABCD");
        assert_eq!(uint64_to_hex_string(0x1234), "0000000000001234");
        assert_eq!(hex_string_to_uint64("AB").unwrap(), 0xAB);
        assert!(hex_string_to_uint64("GH").is_err());
    }

    #[test]
    fn test_string_to_int() {
        assert_eq!(string_to_int("  42  ").unwrap(), 42);
        assert!(string_to_int("abc").is_err());
    }

    #[test]
    fn test_string_to_bool() {
        assert!(string_to_bool("TRUE").unwrap());
        assert!(!string_to_bool("false").unwrap());
        assert!(string_to_bool("maybe").is_err());
    }

    #[test]
    fn test_trim_split_join() {
        assert_eq!(trim("  hello  "), "hello");
        assert_eq!(split("  a   b c "), vec!["a", "b", "c"]);
        assert_eq!(split_by("a,b,c", ','), vec!["a", "b", "c"]);
        assert_eq!(concat_vec(&["a".into(), "b".into()], ","), "a,b");
    }

    #[test]
    fn test_prefix_suffix() {
        assert!(is_prefix("hello", "he"));
        assert!(is_suffix("hello", "lo"));
        assert_eq!(chop_prefix("hello", "he").unwrap(), "llo");
        assert_eq!(chop_suffix("hello", "lo").unwrap(), "hel");
    }

    #[test]
    fn test_case() {
        assert_eq!(to_upper("Hello"), "HELLO");
        assert_eq!(to_lower("Hello"), "hello");
        assert!(is_equal_case_insensitive("Hello", "HELLO"));
    }

    #[test]
    fn test_digits() {
        assert!(is_digits("123"));
        assert!(!is_digits("12a"));
        assert!(!is_digits("1234567890")); // 10 digits -> overflow
        assert_eq!(parse_digits("123").unwrap(), 123);
        assert_eq!(parse_digits("12a").unwrap(), 0);
    }

    #[test]
    fn test_strip_comments() {
        assert_eq!(strip_comments("a # comment\nb"), "a \nb\n");
        assert_eq!(strip_comments("no comments"), "no comments");
    }

    #[test]
    fn test_read_key_values() {
        let input = "a=1, b = 2\nc=3";
        let map = read_key_values(input).unwrap();
        assert_eq!(map.get("a").unwrap(), "1");
        assert_eq!(map.get("b").unwrap(), "2");
        assert_eq!(map.get("c").unwrap(), "3");
    }

    #[test]
    fn test_read_mem() {
        assert_eq!(read_mem("16K").unwrap(), 16 << 10);
        assert_eq!(read_mem("1MB").unwrap(), 1 << 20);
        assert_eq!(read_mem("2G").unwrap(), 2 << 30);
        assert!(read_mem("PB99999999999999999999").is_err());
    }

    #[test]
    fn test_rounding() {
        assert_eq!(round_static(1.234, 100.0), 1.23);
        assert!((round_dynamic(1234.567, 2) - 1235.0).abs() < 1e-9);
        assert!((round_dynamic(1.234567, 2) - 1.23).abs() < 1e-9);
    }

    #[test]
    fn test_date_string_format() {
        let s = get_date_string();
        assert_eq!(s.len(), 19);
        assert!(s.contains('_'));
    }
}
