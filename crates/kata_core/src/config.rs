//! Simple key-value config parser.
//!
//! Corresponds to `cpp/core/config_parser.h` and `cpp/core/config_parser.cpp`.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;

use parking_lot::Mutex;
use thiserror::Error;

use crate::global;
use crate::logger::Logger;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur when parsing or querying a config.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// An underlying I/O operation failed.
    #[error("IOError: {0}")]
    Io(#[from] std::io::Error),

    /// The config file could not be parsed.
    #[error("ConfigParsingError: {0}")]
    Parsing(String),

    /// A requested key is missing or has an invalid value.
    #[error("Config key error: {0}")]
    Key(String),
}

impl ConfigError {
    fn parsing<S: Into<String>>(msg: S) -> Self {
        ConfigError::Parsing(msg.into())
    }

    fn key<S: Into<String>>(msg: S) -> Self {
        ConfigError::Key(msg.into())
    }
}

// ---------------------------------------------------------------------------
// enabled_t
// ---------------------------------------------------------------------------

/// Tri-state value used for config options that can be enabled, disabled, or
/// left on auto.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Enabled {
    False,
    True,
    Auto,
}

impl Enabled {
    /// Parse an enabled value, accepting the same spellings as the C++
    /// `enabled_t::tryParse`.
    pub fn try_parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "1" | "t" | "true" | "enabled" | "y" | "yes" => Some(Enabled::True),
            "0" | "f" | "false" | "disabled" | "n" | "no" => Some(Enabled::False),
            "auto" => Some(Enabled::Auto),
            _ => None,
        }
    }
}

impl fmt::Display for Enabled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Enabled::True => write!(f, "true"),
            Enabled::False => write!(f, "false"),
            Enabled::Auto => write!(f, "auto"),
        }
    }
}

// ---------------------------------------------------------------------------
// ConfigParser
// ---------------------------------------------------------------------------

/// A parser for KataGo's simple `key = value` configuration files.
pub struct ConfigParser {
    file_name: String,
    contents: String,
    key_values: BTreeMap<String, String>,
    keys_override_enabled: bool,
    keys_override_from_includes: bool,
    cur_line_num: usize,
    cur_filename: String,
    included_files: Vec<String>,
    base_dirs: Vec<String>,
    #[allow(dead_code)]
    log_messages: Vec<String>,
    used_keys: Mutex<HashSet<String>>,
}

impl ConfigParser {
    /// Create an empty parser.
    ///
    /// Equivalent to the C++ default constructor before `initialize` is called.
    pub fn new(keys_override: bool, keys_override_from_includes: bool) -> Self {
        Self {
            file_name: String::new(),
            contents: String::new(),
            key_values: BTreeMap::new(),
            keys_override_enabled: keys_override,
            keys_override_from_includes,
            cur_line_num: 0,
            cur_filename: String::new(),
            included_files: Vec::new(),
            base_dirs: Vec::new(),
            log_messages: Vec::new(),
            used_keys: Mutex::new(HashSet::new()),
        }
    }

    /// Read a config from a file.
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        keys_override: bool,
        keys_override_from_includes: bool,
    ) -> Result<Self, ConfigError> {
        let mut parser = Self::new(keys_override, keys_override_from_includes);
        parser.initialize_file(path)?;
        Ok(parser)
    }

    /// Read a config from a string.
    pub fn from_str(
        contents: &str,
        keys_override: bool,
        keys_override_from_includes: bool,
    ) -> Result<Self, ConfigError> {
        let mut parser = Self::new(keys_override, keys_override_from_includes);
        parser.initialize_str(contents)?;
        Ok(parser)
    }

    /// Build a config directly from a key-value map.
    pub fn from_map(map: BTreeMap<String, String>) -> Self {
        let mut parser = Self::new(false, true);
        parser.key_values = map;
        parser
    }

    /// Re-initialize this parser from a file.
    pub fn initialize_file<P: AsRef<Path>>(&mut self, path: P) -> Result<(), ConfigError> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        self.file_name = path.to_string_lossy().to_string();
        let base_dir = extract_base_dir(&self.file_name);
        if !base_dir.is_empty() {
            self.base_dirs.push(base_dir);
        }
        self.initialize_internal(reader)?;
        Ok(())
    }

    /// Re-initialize this parser from a string.
    pub fn initialize_str(&mut self, contents: &str) -> Result<(), ConfigError> {
        self.initialize_internal(contents.as_bytes())
    }

    /// Re-initialize this parser from a key-value map.
    pub fn initialize_map(&mut self, map: BTreeMap<String, String>) {
        self.key_values = map;
    }

    fn initialize_internal<R: BufRead>(&mut self, reader: R) -> Result<(), ConfigError> {
        self.key_values.clear();
        self.contents.clear();
        self.cur_filename.clone_from(&self.file_name);
        self.read_stream_content(reader)?;
        Ok(())
    }

    fn read_stream_content<R: BufRead>(&mut self, reader: R) -> Result<(), ConfigError> {
        let mut line_num = self.cur_line_num;
        let mut content_stream = String::new();
        let mut cur_file_keys: HashSet<String> = HashSet::new();

        for line_result in reader.lines() {
            let line = line_result?;
            content_stream.push_str(&line);
            content_stream.push('\n');
            line_num += 1;
            self.cur_line_num = line_num;

            let trimmed = global::trim(&line);
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            if trimmed.starts_with('@') {
                self.process_include_line(trimmed)?;
                continue;
            }

            let (key, value) = match self.parse_key_value(trimmed)? {
                Some(kv) => kv,
                None => continue,
            };

            if cur_file_keys.contains(&key) {
                if !self.keys_override_enabled {
                    return Err(ConfigError::parsing(format!(
                        "Key '{}' was specified multiple times in {}, you probably didn't mean to do this, please delete one of them",
                        key, self.cur_filename
                    )));
                }
                self.log_messages.push(format!(
                    "Key '{}' was overriden by new value '{}'{}",
                    key,
                    value,
                    self.line_and_file_info()
                ));
            }
            if self.key_values.contains_key(&key) {
                if !self.keys_override_from_includes {
                    return Err(ConfigError::parsing(format!(
                        "Key '{}' was specified multiple times in {} or its included files, and key overriding is disabled",
                        key, self.cur_filename
                    )));
                }
                self.log_messages.push(format!(
                    "Key '{}' was overriden by new value '{}'{}",
                    key,
                    value,
                    self.line_and_file_info()
                ));
            }
            self.key_values.insert(key.clone(), value);
            cur_file_keys.insert(key);
        }

        self.contents.push_str(&content_stream);
        Ok(())
    }

    fn process_include_line(&mut self, mut line: &str) -> Result<(), ConfigError> {
        // Strip any trailing comment from the @ directive line.
        if let Some(pos) = line.find('#') {
            line = &line[..pos];
        }

        if line.len() < 9 {
            return Err(ConfigError::parsing(format!(
                "Unsupported @ directive{}",
                self.line_and_file_info()
            )));
        }

        let pos0 = line.find([' ', '\t', '\x0B', '\x0C', '=']).ok_or_else(|| {
            ConfigError::parsing(format!(
                "@ directive without value (key-val separator is not found){}",
                self.line_and_file_info()
            ))
        })?;

        let key = global::trim(&line[..pos0]);
        if key != "@include" {
            return Err(ConfigError::parsing(format!(
                "Unsupported @ directive '{}'{}",
                key,
                self.line_and_file_info()
            )));
        }

        let mut value = &line[pos0 + 1..];
        let pos1 = value
            .find(|c: char| !matches!(c, ' ' | '\t' | '\x0B' | '\x0C' | '='))
            .ok_or_else(|| {
                ConfigError::parsing(format!(
                    "@ directive without value (value after key-val separator is not found){}",
                    self.line_and_file_info()
                ))
            })?;

        value = global::trim(&value[pos1..]);
        value = global::trim_with_delims(value, "'");
        value = global::trim_with_delims(value, "\"");

        let saved_line_num = self.cur_line_num;
        self.process_included_file(value.to_string())?;
        self.cur_line_num = saved_line_num;
        Ok(())
    }

    fn process_included_file(&mut self, fname: String) -> Result<(), ConfigError> {
        if fname == self.file_name || self.included_files.contains(&fname) {
            return Err(ConfigError::parsing(format!(
                "Circular or multiple inclusion of the same file: '{}'{}",
                fname,
                self.line_and_file_info()
            )));
        }
        self.included_files.push(fname.clone());
        self.cur_filename.clone_from(&fname);

        let mut fpath = String::new();
        for base in &self.base_dirs {
            fpath.push_str(base);
        }
        fpath.push_str(&fname);

        let base_dir = extract_base_dir(&fname);
        if !base_dir.is_empty() {
            if base_dir.starts_with('\\') || base_dir.starts_with('/') {
                return Err(ConfigError::parsing(format!(
                    "Absolute paths in the included files are not supported yet{}",
                    self.line_and_file_info()
                )));
            }
            self.base_dirs.push(base_dir.clone());
        }

        let file = File::open(&fpath)?;
        let reader = BufReader::new(file);
        self.read_stream_content(reader)?;

        if !base_dir.is_empty() {
            self.base_dirs.pop();
        }
        Ok(())
    }

    fn parse_key_value(&self, trimmed_line: &str) -> Result<Option<(String, String)>, ConfigError> {
        let mut key = String::new();
        let mut value = String::new();
        let mut found_key = false;
        let mut i = 0;
        let chars: Vec<char> = trimmed_line.chars().collect();

        // Parse key.
        while i < chars.len() {
            let c = chars[i];
            if global::is_alpha(c) || global::is_digit(c) || c == '_' || c == '-' {
                key.push(c);
                found_key = true;
                i += 1;
                continue;
            }
            if c == '#' {
                if found_key {
                    return Err(ConfigError::parsing(format!(
                        "Could not parse key value pair{}",
                        self.line_and_file_info()
                    )));
                }
                return Ok(None);
            }
            if global::is_whitespace_char(c) || c == '=' {
                break;
            }
            return Err(ConfigError::parsing(format!(
                "Could not parse key value pair{}",
                self.line_and_file_info()
            )));
        }

        // Skip whitespace after key.
        while i < chars.len() {
            let c = chars[i];
            if global::is_whitespace_char(c) {
                i += 1;
                continue;
            }
            if c == '#' {
                if found_key {
                    return Err(ConfigError::parsing(format!(
                        "Could not parse key value pair{}",
                        self.line_and_file_info()
                    )));
                }
                return Ok(None);
            }
            if c == '=' {
                break;
            }
            return Err(ConfigError::parsing(format!(
                "Could not parse key value pair{}",
                self.line_and_file_info()
            )));
        }

        // Skip equals sign.
        let mut found_equals = false;
        if i < chars.len() {
            debug_assert_eq!(chars[i], '=');
            found_equals = true;
            i += 1;
        }

        // Skip whitespace after equals sign.
        while i < chars.len() {
            let c = chars[i];
            if global::is_whitespace_char(c) {
                i += 1;
                continue;
            }
            if c == '#' {
                if found_key || found_equals {
                    return Err(ConfigError::parsing(format!(
                        "Could not parse key value pair{}",
                        self.line_and_file_info()
                    )));
                }
                return Ok(None);
            }
            break;
        }

        // Maybe parse double quotes.
        let mut is_double_quotes = false;
        if i < chars.len() && chars[i] == '"' {
            is_double_quotes = true;
            i += 1;
        }

        // Parse value.
        let mut found_value = false;
        while i < chars.len() {
            let c = chars[i];
            if is_double_quotes {
                if c == '\\' {
                    if i + 1 >= chars.len() {
                        return Err(ConfigError::parsing(format!(
                            "Could not parse key value pair{}",
                            self.line_and_file_info()
                        )));
                    }
                    i += 1;
                    value.push(chars[i]);
                    found_value = true;
                } else if c == '"' {
                    break;
                } else {
                    value.push(c);
                    found_value = true;
                }
            } else if c == '#' {
                break;
            } else {
                value.push(c);
                found_value = true;
            }
            i += 1;
        }

        if is_double_quotes {
            if i >= chars.len() || chars[i] != '"' {
                return Err(ConfigError::parsing(format!(
                    "Could not parse key value pair{}",
                    self.line_and_file_info()
                )));
            }
            i += 1;
            let remainder = global::trim(&trimmed_line[i..]);
            if !remainder.is_empty() && !remainder.starts_with('#') {
                return Err(ConfigError::parsing(format!(
                    "Could not parse key value pair{}",
                    self.line_and_file_info()
                )));
            }
        } else {
            value = global::trim(&value).to_string();
        }

        if is_double_quotes && !(found_key && found_value) {
            return Err(ConfigError::parsing(format!(
                "Could not parse key value pair{}",
                self.line_and_file_info()
            )));
        }
        if found_equals && !(found_key && found_value) {
            return Err(ConfigError::parsing(format!(
                "Could not parse key value pair{}",
                self.line_and_file_info()
            )));
        }
        if found_key != found_value {
            return Err(ConfigError::parsing(format!(
                "Could not parse key value pair{}",
                self.line_and_file_info()
            )));
        }

        if found_key {
            Ok(Some((key, value)))
        } else {
            Ok(None)
        }
    }

    fn line_and_file_info(&self) -> String {
        format!(", line {} in '{}'", self.cur_line_num, self.cur_filename)
    }

    // -----------------------------------------------------------------------
    // Key/value inspection
    // -----------------------------------------------------------------------

    /// The original file name passed to the parser, if any.
    pub fn file_name(&self) -> &str {
        &self.file_name
    }

    /// The raw contents of the config and any included files.
    pub fn contents(&self) -> &str {
        &self.contents
    }

    /// All key-value pairs as a human-readable string.
    pub fn all_key_vals(&self) -> String {
        if self.key_values.is_empty() {
            return String::new();
        }
        self.key_values
            .iter()
            .map(|(k, v)| format!("{} = {}", k, v))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    }

    /// Returns `true` if the given key is present.
    pub fn contains(&self, key: &str) -> bool {
        self.key_values.contains_key(key)
    }

    /// Returns `true` if any of the given keys is present.
    pub fn contains_any(&self, possible_keys: &[String]) -> bool {
        possible_keys.iter().any(|k| self.contains(k))
    }

    /// Returns the first key from `possible_keys` that is present, or fails.
    pub fn first_found_or_fail(&self, possible_keys: &[String]) -> Result<String, ConfigError> {
        for key in possible_keys {
            if self.contains(key) {
                return Ok(key.clone());
            }
        }
        let message = possible_keys
            .iter()
            .fold("Could not find key".to_string(), |acc, k| {
                acc + " '" + k + "'"
            });
        Err(ConfigError::key(format!(
            "{} in config file {}",
            message, self.file_name
        )))
    }

    /// Returns the first key from `possible_keys` that is present, or an empty
    /// string.
    pub fn first_found_or_empty(&self, possible_keys: &[String]) -> String {
        for key in possible_keys {
            if self.contains(key) {
                return key.clone();
            }
        }
        String::new()
    }

    // -----------------------------------------------------------------------
    // Getters
    // -----------------------------------------------------------------------

    /// Get a string value, marking the key as used.
    pub fn get_string(&self, key: &str) -> Result<String, ConfigError> {
        let value = self
            .key_values
            .get(key)
            .ok_or_else(|| {
                ConfigError::key(format!(
                    "Could not find key '{}' in config file {}",
                    key, self.file_name
                ))
            })?
            .clone();
        self.used_keys.lock().insert(key.to_string());
        Ok(value)
    }

    /// Get a string value restricted to a set of allowed values.
    pub fn get_string_set(
        &self,
        key: &str,
        possibles: &BTreeSet<String>,
    ) -> Result<String, ConfigError> {
        let value = self.get_string(key)?;
        if !possibles.contains(&value) {
            return Err(ConfigError::key(format!(
                "Key '{}' must be one of ({}) in config file {}",
                key,
                global::concat_set(possibles, "|"),
                self.file_name
            )));
        }
        Ok(value)
    }

    /// Get a comma-separated list of string values.
    pub fn get_strings(&self, key: &str) -> Result<Vec<String>, ConfigError> {
        Ok(global::split_by(&self.get_string(key)?, ','))
    }

    /// Get a comma-separated list of non-empty, trimmed string values.
    pub fn get_strings_non_empty_trim(&self, key: &str) -> Result<Vec<String>, ConfigError> {
        let raw = self.get_strings(key)?;
        Ok(raw
            .into_iter()
            .map(|s| global::trim(&s).to_string())
            .filter(|s| !s.is_empty())
            .collect())
    }

    /// Get a comma-separated list of string values restricted to allowed values.
    pub fn get_strings_set(
        &self,
        key: &str,
        possibles: &BTreeSet<String>,
    ) -> Result<Vec<String>, ConfigError> {
        let values = self.get_strings(key)?;
        for value in &values {
            if !possibles.contains(value) {
                return Err(ConfigError::key(format!(
                    "Key '{}' must be one of ({}) in config file {}",
                    key,
                    global::concat_set(possibles, "|"),
                    self.file_name
                )));
            }
        }
        Ok(values)
    }

    /// Get a boolean value.
    pub fn get_bool(&self, key: &str) -> Result<bool, ConfigError> {
        let value = self.get_string(key)?;
        global::try_string_to_bool(&value).ok_or_else(|| {
            ConfigError::key(format!(
                "Could not parse '{}' as bool for key '{}' in config file {}",
                value, key, self.file_name
            ))
        })
    }

    /// Get a list of boolean values.
    pub fn get_bools(&self, key: &str) -> Result<Vec<bool>, ConfigError> {
        let values = self.get_strings(key)?;
        let mut ret = Vec::with_capacity(values.len());
        for value in values {
            match global::try_string_to_bool(&value) {
                Some(b) => ret.push(b),
                None => {
                    return Err(ConfigError::key(format!(
                        "Could not parse '{}' as bool for key '{}' in config file {}",
                        value, key, self.file_name
                    )));
                }
            }
        }
        Ok(ret)
    }

    /// Get an `Enabled` tri-state value.
    pub fn get_enabled(&self, key: &str) -> Result<Enabled, ConfigError> {
        let value = self.get_string(key)?.to_lowercase();
        let value = global::trim(&value);
        Enabled::try_parse(value).ok_or_else(|| {
            ConfigError::key(format!(
                "Could not parse '{}' as bool or auto for key '{}' in config file {}",
                value, key, self.file_name
            ))
        })
    }

    /// Get an `i32` value, optionally constrained to `[min, max]`.
    pub fn get_int(&self, key: &str, min: i32, max: i32) -> Result<i32, ConfigError> {
        debug_assert!(min <= max);
        let value = self.get_string(key)?;
        let x = global::try_string_to_int(&value).ok_or_else(|| {
            ConfigError::key(format!(
                "Could not parse '{}' as int for key '{}' in config file {}",
                value, key, self.file_name
            ))
        })?;
        if x < min || x > max {
            return Err(ConfigError::key(format!(
                "Key '{}' must be in the range {} to {} in config file {}",
                key,
                global::int_to_string(min),
                global::int_to_string(max),
                self.file_name
            )));
        }
        Ok(x)
    }

    /// Get a list of `i32` values.
    pub fn get_ints(&self, key: &str, min: i32, max: i32) -> Result<Vec<i32>, ConfigError> {
        let values = self.get_strings(key)?;
        let mut ret = Vec::with_capacity(values.len());
        for value in values {
            let x = global::try_string_to_int(&value).ok_or_else(|| {
                ConfigError::key(format!(
                    "Could not parse '{}' as int for key '{}' in config file {}",
                    value, key, self.file_name
                ))
            })?;
            if x < min || x > max {
                return Err(ConfigError::key(format!(
                    "Key '{}' must be in the range {} to {} in config file {}",
                    key,
                    global::int_to_string(min),
                    global::int_to_string(max),
                    self.file_name
                )));
            }
            ret.push(x);
        }
        Ok(ret)
    }

    /// Get a list of non-negative integer dashed pairs.
    pub fn get_non_negative_int_dashed_pairs(
        &self,
        key: &str,
        min: i32,
        max: i32,
    ) -> Result<Vec<(i32, i32)>, ConfigError> {
        let pair_strs = self.get_strings(key)?;
        let mut ret = Vec::with_capacity(pair_strs.len());
        for pair_str in pair_strs {
            if global::trim(&pair_str).is_empty() {
                continue;
            }
            let pieces = global::split_by(global::trim(&pair_str), '-');
            if pieces.len() != 2 {
                return Err(ConfigError::key(format!(
                    "Could not parse '{}' as a pair of integers separated by a dash for key '{}' in config file {}",
                    pair_str, key, self.file_name
                )));
            }
            let p0 = global::try_string_to_int(&pieces[0]).ok_or_else(|| {
                ConfigError::key(format!(
                    "Could not parse '{}' as a pair of integers separated by a dash for key '{}' in config file {}",
                    pair_str, key, self.file_name
                ))
            })?;
            let p1 = global::try_string_to_int(&pieces[1]).ok_or_else(|| {
                ConfigError::key(format!(
                    "Could not parse '{}' as a pair of integers separated by a dash for key '{}' in config file {}",
                    pair_str, key, self.file_name
                ))
            })?;
            if p0 < min || p0 > max || p1 < min || p1 > max {
                return Err(ConfigError::key(format!(
                    "Expected key '{}' to have all values range {} to {} in config file {}",
                    key,
                    global::int_to_string(min),
                    global::int_to_string(max),
                    self.file_name
                )));
            }
            ret.push((p0, p1));
        }
        Ok(ret)
    }

    /// Get an `i64` value, optionally constrained to `[min, max]`.
    pub fn get_int64(&self, key: &str, min: i64, max: i64) -> Result<i64, ConfigError> {
        debug_assert!(min <= max);
        let value = self.get_string(key)?;
        let x = global::try_string_to_int64(&value).ok_or_else(|| {
            ConfigError::key(format!(
                "Could not parse '{}' as int64_t for key '{}' in config file {}",
                value, key, self.file_name
            ))
        })?;
        if x < min || x > max {
            return Err(ConfigError::key(format!(
                "Key '{}' must be in the range {} to {} in config file {}",
                key,
                global::int64_to_string(min),
                global::int64_to_string(max),
                self.file_name
            )));
        }
        Ok(x)
    }

    /// Get a list of `i64` values.
    pub fn get_int64s(&self, key: &str, min: i64, max: i64) -> Result<Vec<i64>, ConfigError> {
        let values = self.get_strings(key)?;
        let mut ret = Vec::with_capacity(values.len());
        for value in values {
            let x = global::try_string_to_int64(&value).ok_or_else(|| {
                ConfigError::key(format!(
                    "Could not parse '{}' as int64_t for key '{}' in config file {}",
                    value, key, self.file_name
                ))
            })?;
            if x < min || x > max {
                return Err(ConfigError::key(format!(
                    "Key '{}' must be in the range {} to {} in config file {}",
                    key,
                    global::int64_to_string(min),
                    global::int64_to_string(max),
                    self.file_name
                )));
            }
            ret.push(x);
        }
        Ok(ret)
    }

    /// Get a `u64` value, optionally constrained to `[min, max]`.
    pub fn get_uint64(&self, key: &str, min: u64, max: u64) -> Result<u64, ConfigError> {
        debug_assert!(min <= max);
        let value = self.get_string(key)?;
        let x = global::try_string_to_uint64(&value).ok_or_else(|| {
            ConfigError::key(format!(
                "Could not parse '{}' as uint64_t for key '{}' in config file {}",
                value, key, self.file_name
            ))
        })?;
        if x < min || x > max {
            return Err(ConfigError::key(format!(
                "Key '{}' must be in the range {} to {} in config file {}",
                key,
                global::uint64_to_string(min),
                global::uint64_to_string(max),
                self.file_name
            )));
        }
        Ok(x)
    }

    /// Get a list of `u64` values.
    pub fn get_uint64s(&self, key: &str, min: u64, max: u64) -> Result<Vec<u64>, ConfigError> {
        let values = self.get_strings(key)?;
        let mut ret = Vec::with_capacity(values.len());
        for value in values {
            let x = global::try_string_to_uint64(&value).ok_or_else(|| {
                ConfigError::key(format!(
                    "Could not parse '{}' as uint64_t for key '{}' in config file {}",
                    value, key, self.file_name
                ))
            })?;
            if x < min || x > max {
                return Err(ConfigError::key(format!(
                    "Key '{}' must be in the range {} to {} in config file {}",
                    key,
                    global::uint64_to_string(min),
                    global::uint64_to_string(max),
                    self.file_name
                )));
            }
            ret.push(x);
        }
        Ok(ret)
    }

    /// Get an `f32` value, optionally constrained to `[min, max]`.
    pub fn get_float(&self, key: &str, min: f32, max: f32) -> Result<f32, ConfigError> {
        debug_assert!(min <= max);
        let value = self.get_string(key)?;
        let x = global::try_string_to_float(&value).ok_or_else(|| {
            ConfigError::key(format!(
                "Could not parse '{}' as float for key '{}' in config file {}",
                value, key, self.file_name
            ))
        })?;
        if x.is_nan() {
            return Err(ConfigError::key(format!(
                "Key '{}' is nan in config file {}",
                key, self.file_name
            )));
        }
        if x < min || x > max {
            return Err(ConfigError::key(format!(
                "Key '{}' must be in the range {} to {} in config file {}",
                key,
                global::float_to_string(min),
                global::float_to_string(max),
                self.file_name
            )));
        }
        Ok(x)
    }

    /// Get a list of `f32` values.
    pub fn get_floats(&self, key: &str, min: f32, max: f32) -> Result<Vec<f32>, ConfigError> {
        let values = self.get_strings(key)?;
        let mut ret = Vec::with_capacity(values.len());
        for value in values {
            let x = global::try_string_to_float(&value).ok_or_else(|| {
                ConfigError::key(format!(
                    "Could not parse '{}' as float for key '{}' in config file {}",
                    value, key, self.file_name
                ))
            })?;
            if x.is_nan() {
                return Err(ConfigError::key(format!(
                    "Key '{}' is nan in config file {}",
                    key, self.file_name
                )));
            }
            if x < min || x > max {
                return Err(ConfigError::key(format!(
                    "Key '{}' must be in the range {} to {} in config file {}",
                    key,
                    global::float_to_string(min),
                    global::float_to_string(max),
                    self.file_name
                )));
            }
            ret.push(x);
        }
        Ok(ret)
    }

    /// Get an `f64` value, optionally constrained to `[min, max]`.
    pub fn get_double(&self, key: &str, min: f64, max: f64) -> Result<f64, ConfigError> {
        debug_assert!(min <= max);
        let value = self.get_string(key)?;
        let x = global::try_string_to_double(&value).ok_or_else(|| {
            ConfigError::key(format!(
                "Could not parse '{}' as double for key '{}' in config file {}",
                value, key, self.file_name
            ))
        })?;
        if x.is_nan() {
            return Err(ConfigError::key(format!(
                "Key '{}' is nan in config file {}",
                key, self.file_name
            )));
        }
        if x < min || x > max {
            return Err(ConfigError::key(format!(
                "Key '{}' must be in the range {} to {} in config file {}",
                key,
                global::double_to_string(min),
                global::double_to_string(max),
                self.file_name
            )));
        }
        Ok(x)
    }

    /// Get a list of `f64` values.
    pub fn get_doubles(&self, key: &str, min: f64, max: f64) -> Result<Vec<f64>, ConfigError> {
        let values = self.get_strings(key)?;
        let mut ret = Vec::with_capacity(values.len());
        for value in values {
            let x = global::try_string_to_double(&value).ok_or_else(|| {
                ConfigError::key(format!(
                    "Could not parse '{}' as double for key '{}' in config file {}",
                    value, key, self.file_name
                ))
            })?;
            if x.is_nan() {
                return Err(ConfigError::key(format!(
                    "Key '{}' is nan in config file {}",
                    key, self.file_name
                )));
            }
            if x < min || x > max {
                return Err(ConfigError::key(format!(
                    "Key '{}' must be in the range {} to {} in config file {}",
                    key,
                    global::double_to_string(min),
                    global::double_to_string(max),
                    self.file_name
                )));
            }
            ret.push(x);
        }
        Ok(ret)
    }

    // -----------------------------------------------------------------------
    // Overrides and aliases
    // -----------------------------------------------------------------------

    /// Remove a single key or set it to a new value. An empty value deletes the
    /// key.
    pub fn override_key(&mut self, key: &str, value: &str) {
        if value.is_empty() {
            self.key_values.remove(key);
        } else {
            self.key_values.insert(key.to_string(), value.to_string());
        }
    }

    /// Override keys by loading another config file.
    pub fn override_keys_file(&mut self, fname: &str) -> Result<(), ConfigError> {
        self.base_dirs.clear();
        self.process_included_file(fname.to_string())
    }

    /// Override keys from a map. Empty values delete keys.
    pub fn override_keys_map(&mut self, new_kvs: &BTreeMap<String, String>) {
        for (key, value) in new_kvs {
            if value.is_empty() {
                self.key_values.remove(key);
            } else {
                self.key_values.insert(key.clone(), value.clone());
            }
        }
        self.file_name
            .push_str(" and/or command-line and query overrides");
    }

    /// Override keys from a map, erasing mutually exclusive keys first.
    pub fn override_keys_map_with_mutex(
        &mut self,
        new_kvs: &BTreeMap<String, String>,
        mutex_key_sets: &[(BTreeSet<String>, BTreeSet<String>)],
    ) {
        for (a, b) in mutex_key_sets {
            let has_a = a.iter().any(|k| new_kvs.contains_key(k));
            let has_b = b.iter().any(|k| new_kvs.contains_key(k));
            if has_a {
                for k in b {
                    self.key_values.remove(k);
                }
            }
            if has_b {
                for k in a {
                    self.key_values.remove(k);
                }
            }
        }
        self.override_keys_map(new_kvs);
    }

    /// Parse a comma-separated `key=value` string into a map.
    pub fn parse_comma_separated(
        comma_separated_values: &str,
    ) -> Result<BTreeMap<String, String>, ConfigError> {
        let mut key_values = BTreeMap::new();
        for piece in global::split_by(comma_separated_values, ',') {
            let s = global::trim(&piece);
            if s.is_empty() {
                continue;
            }
            let pos = s.find('=').ok_or_else(|| {
                ConfigError::parsing(format!(
                    "Could not parse kv pair, could not find '=' in: {}",
                    s
                ))
            })?;
            let key = global::trim(&s[..pos]).to_string();
            let value = global::trim(&s[pos + 1..]).to_string();
            key_values.insert(key, value);
        }
        Ok(key_values)
    }

    /// Rename a key to another key. It is an error for both to be present.
    pub fn apply_alias(
        &mut self,
        map_this_key: &str,
        to_this_key: &str,
    ) -> Result<(), ConfigError> {
        if self.contains(map_this_key) && self.contains(to_this_key) {
            return Err(ConfigError::key(format!(
                "Cannot specify both {} and {} in the same config",
                map_this_key, to_this_key
            )));
        }
        if self.contains(map_this_key) {
            let value = self.key_values.remove(map_this_key).unwrap();
            self.key_values.insert(to_this_key.to_string(), value);
            let mut used = self.used_keys.lock();
            if used.remove(map_this_key) {
                used.insert(to_this_key.to_string());
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Used-key tracking
    // -----------------------------------------------------------------------

    /// Mark a key as used.
    pub fn mark_key_used(&self, key: &str) {
        self.used_keys.lock().insert(key.to_string());
    }

    /// Unmark a key as used.
    pub fn unset_used_key(&self, key: &str) {
        self.used_keys.lock().remove(key);
    }

    /// Mark every key with the given prefix as used.
    pub fn mark_all_keys_used_with_prefix(&self, prefix: &str) {
        let mut used = self.used_keys.lock();
        for key in self.key_values.keys() {
            if global::is_prefix(key, prefix) {
                used.insert(key.clone());
            }
        }
    }

    /// Return the list of keys that were set but never read.
    pub fn unused_keys(&self) -> Vec<String> {
        let used = self.used_keys.lock();
        self.key_values
            .keys()
            .filter(|k| !used.contains(*k))
            .cloned()
            .collect()
    }

    /// Write warnings about unused keys to the given writer and optional
    /// logger.
    pub fn warn_unused_keys(
        &self,
        writer: &mut dyn Write,
        logger: Option<&Logger>,
    ) -> io::Result<()> {
        let unused = self.unused_keys();
        if unused.is_empty() {
            return Ok(());
        }

        let mut messages: Vec<String> = Vec::new();
        messages.push("--------------".to_string());
        messages.push(format!(
            "WARNING: Config had unused keys! You may have a typo, an option you specified is being unused from {}",
            self.file_name
        ));
        for key in &unused {
            messages.push(format!(
                "WARNING: Unused key '{}' in {}",
                key, self.file_name
            ));
        }
        messages.push("--------------".to_string());

        if let Some(logger) = logger {
            for msg in &messages {
                logger.write(msg);
            }
        }
        for msg in &messages {
            writeln!(writer, "{}", msg)?;
        }
        Ok(())
    }
}

impl Clone for ConfigParser {
    fn clone(&self) -> Self {
        Self {
            file_name: self.file_name.clone(),
            contents: self.contents.clone(),
            key_values: self.key_values.clone(),
            keys_override_enabled: self.keys_override_enabled,
            keys_override_from_includes: self.keys_override_from_includes,
            cur_line_num: self.cur_line_num,
            cur_filename: self.cur_filename.clone(),
            included_files: self.included_files.clone(),
            base_dirs: self.base_dirs.clone(),
            log_messages: self.log_messages.clone(),
            used_keys: Mutex::new(self.used_keys.lock().clone()),
        }
    }
}

fn extract_base_dir(fname: &str) -> String {
    match fname.rfind(['/', '\\']) {
        Some(idx) => fname[..=idx].to_string(),
        None => String::new(),
    }
}

/// Type alias matching the informal `Config` name used in the C++ codebase.
pub type Config = ConfigParser;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn data_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("KataGo")
            .join("cpp")
            .join("tests")
            .join("data")
            .join("configs")
    }

    #[test]
    fn test_empty_config() {
        let cfg = ConfigParser::from_str("", false, true).unwrap();
        assert_eq!(cfg.all_key_vals(), "");
    }

    #[test]
    fn test_inline_config() {
        let s = r#"
a1 = k2
#comment
 #comment
  #= == == ayay
  #a = b
  b1 = c5
_c_ = 43
d_= 5
e=6
f =7
abc =    def
bcd    =  g#foo
c-de =  g  #"test's"=== =
_a = "quoted"
_b= "quoted "  #hmm##
 _c =" quoted "
_d =" some # symbols \" yay " # later comment
 _e  = "\"\"\\"  # comment
# _f  = "\"\"\\"  # comment
key =  with spaces
quotes =  i'm a value " with " quotes! # hmmm"!
 test=back\slashes don't \escape \\here\
 test2=back\slashes don't \escape \\here\#comment
"#;
        let cfg = ConfigParser::from_str(s, false, true).unwrap();
        let expected = concat!(
            "_a = quoted\n",
            "_b = quoted \n",
            "_c =  quoted \n",
            "_c_ = 43\n",
            "_d =  some # symbols \" yay \n",
            "_e = \"\"\\\n",
            "a1 = k2\n",
            "abc = def\n",
            "b1 = c5\n",
            "bcd = g\n",
            "c-de = g\n",
            "d_ = 5\n",
            "e = 6\n",
            "f = 7\n",
            "key = with spaces\n",
            "quotes = i'm a value \" with \" quotes!\n",
            "test = back\\slashes don't \\escape \\\\here\\\n",
            "test2 = back\\slashes don't \\escape \\\\here\\\n",
        );
        assert_eq!(cfg.all_key_vals(), expected);
    }

    fn fails(s: &str) -> bool {
        ConfigParser::from_str(s, false, true).is_err()
    }

    #[test]
    fn test_parse_failures() {
        assert!(fails("abc\n"));
        assert!(fails("abc =\n"));
        assert!(fails("abc = # comment\n"));
        assert!(fails("abc = \"\"\n"));
        assert!(fails("abc = \"\"def\n"));
        assert!(fails("abc = \"data\"def\n"));
        assert!(fails("abc = \"data\" def\n"));
        assert!(!fails("abc = \"data\"# def\n"));
        assert!(!fails("abc = \"data\" #def\n"));
        assert!(fails(" =\n"));
        assert!(fails("=\n"));
        assert!(fails("= # foo\n"));
        assert!(fails("\"abc\" = def\n"));
        assert!(fails("a!b = def\n"));
        assert!(fails("a#b = def\n"));
        assert!(fails("a$b = def\n"));
        assert!(fails("a%b = def\n"));
        assert!(fails("a@b = def\n"));
        assert!(!fails("#ab = def\n"));
        assert!(!fails("0ab = def\n"));
        assert!(fails("!ab = def\n"));
        assert!(!fails("a-x = c-y\n"));
        assert!(!fails("notrailing = newline is okay"));
    }

    #[test]
    fn test_numeric_getters() {
        let s = r#"
negInt = -42
zeroInt = 0
posInt = 100
bigInt = 2000000000
negInt64 = -9000000000000
zeroInt64 = 0
posUInt64 = 18000000000000
negFloat = -3.5
zeroFloat = 0.0
posFloat = 1.5e20
negDouble = -1.0e100
zeroDouble = 0.0
posDouble = 1.0e100
"#;
        let cfg = ConfigParser::from_str(s, false, true).unwrap();
        assert_eq!(cfg.get_int("negInt", i32::MIN, i32::MAX).unwrap(), -42);
        assert_eq!(cfg.get_int("zeroInt", i32::MIN, i32::MAX).unwrap(), 0);
        assert_eq!(cfg.get_int("posInt", i32::MIN, i32::MAX).unwrap(), 100);
        assert_eq!(
            cfg.get_int("bigInt", i32::MIN, i32::MAX).unwrap(),
            2_000_000_000
        );
        assert_eq!(
            cfg.get_int64("negInt64", i64::MIN, i64::MAX).unwrap(),
            -9_000_000_000_000
        );
        assert_eq!(cfg.get_int64("zeroInt64", i64::MIN, i64::MAX).unwrap(), 0);
        assert_eq!(
            cfg.get_uint64("posUInt64", u64::MIN, u64::MAX).unwrap(),
            18_000_000_000_000
        );
        assert_eq!(cfg.get_float("negFloat", f32::MIN, f32::MAX).unwrap(), -3.5);
        assert_eq!(cfg.get_float("zeroFloat", f32::MIN, f32::MAX).unwrap(), 0.0);
        assert_eq!(
            cfg.get_float("posFloat", f32::MIN, f32::MAX).unwrap(),
            1.5e20
        );
        assert_eq!(
            cfg.get_double("negDouble", f64::MIN, f64::MAX).unwrap(),
            -1.0e100
        );
        assert_eq!(
            cfg.get_double("zeroDouble", f64::MIN, f64::MAX).unwrap(),
            0.0
        );
        assert_eq!(
            cfg.get_double("posDouble", f64::MIN, f64::MAX).unwrap(),
            1.0e100
        );
    }

    #[test]
    fn test_numeric_ranges() {
        let s = "val = 5\nfval = 2.5\n";
        let cfg = ConfigParser::from_str(s, false, true).unwrap();
        assert_eq!(cfg.get_int("val", 0, 10).unwrap(), 5);
        assert_eq!(cfg.get_float("fval", 0.0, 5.0).unwrap(), 2.5);
        assert_eq!(cfg.get_double("fval", 0.0, 5.0).unwrap(), 2.5);
        assert!(cfg.get_int("val", 10, 20).is_err());
        assert!(cfg.get_float("fval", 5.0, 10.0).is_err());
        assert!(cfg.get_double("fval", 5.0, 10.0).is_err());
    }

    #[test]
    fn test_numeric_vectors() {
        let s = r#"
ints = -100, 0, 100
floats = -1.5, 0.0, 1.5
doubles = -1.0e50, 0.0, 1.0e50
"#;
        let cfg = ConfigParser::from_str(s, false, true).unwrap();
        assert_eq!(
            cfg.get_ints("ints", i32::MIN, i32::MAX).unwrap(),
            vec![-100, 0, 100]
        );
        assert_eq!(
            cfg.get_floats("floats", f32::MIN, f32::MAX).unwrap(),
            vec![-1.5f32, 0.0, 1.5]
        );
        assert_eq!(
            cfg.get_doubles("doubles", f64::MIN, f64::MAX).unwrap(),
            vec![-1.0e50, 0.0, 1.0e50]
        );
    }

    #[test]
    fn test_override_and_alias() {
        let mut cfg = ConfigParser::from_str("old = 1\n", false, true).unwrap();
        cfg.apply_alias("old", "new").unwrap();
        assert!(!cfg.contains("old"));
        assert_eq!(cfg.get_int("new", 0, 100).unwrap(), 1);

        cfg.override_key("new", "");
        assert!(!cfg.contains("new"));

        cfg.override_key("other", "42");
        assert_eq!(cfg.get_int("other", 0, 100).unwrap(), 42);
    }

    #[test]
    fn test_unused_keys() {
        let cfg = ConfigParser::from_str("a = 1\nb = 2\n", false, true).unwrap();
        let _ = cfg.get_int("a", 0, 10);
        assert_eq!(cfg.unused_keys(), vec!["b".to_string()]);
        cfg.mark_key_used("b");
        assert!(cfg.unused_keys().is_empty());
    }

    #[test]
    fn test_parse_comma_separated() {
        let map =
            ConfigParser::parse_comma_separated("startPosesPolicyInitAreaProp=0.25,rules=Japanese")
                .unwrap();
        assert_eq!(map.get("rules").unwrap(), "Japanese");
        assert_eq!(map.get("startPosesPolicyInitAreaProp").unwrap(), "0.25");
    }

    #[test]
    fn test_file_config() {
        let path = data_dir().join("analysis_example.cfg");
        let cfg = ConfigParser::from_file(&path, false, true).unwrap();
        assert_eq!(cfg.get_int("nnMaxBatchSize", 0, 10000).unwrap(), 64);
    }

    #[test]
    fn test_duplicate_key_errors() {
        let path = data_dir().join("test-duplicate.cfg");
        assert!(ConfigParser::from_file(&path, false, true).is_err());
        let cfg = ConfigParser::from_file(&path, true, true).unwrap();
        assert_eq!(cfg.get_string("logDir").unwrap(), "more_logs");
    }

    #[test]
    fn test_override_from_includes_disabled() {
        let path = data_dir().join("test.cfg");
        assert!(ConfigParser::from_file(&path, false, false).is_err());
    }

    #[test]
    fn test_includes_and_overrides() {
        let path = data_dir().join("test.cfg");
        let cfg = ConfigParser::from_file(&path, false, true).unwrap();
        assert!(cfg.contains("reportAnalysisWinratesAs"));
        assert_eq!(cfg.get_int("maxVisits", 0, 10000).unwrap(), 1000);
        assert_eq!(cfg.get_string("logDir").unwrap(), "more_logs");
        assert_eq!(cfg.get_int("nnMaxBatchSize", 0, 200_000).unwrap(), 100_500);
    }
}
