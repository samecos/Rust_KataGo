//! Common command-line argument handling for the KataGo CLI.
//!
//! Corresponds to `cpp/command/commandline.h` and `cpp/command/commandline.cpp`.
//! This module provides the shared `--model`, `--human-model`, `--config`, and
//! `--override-config` flags used by most subcommands, plus helpers to resolve
//! default file paths and build a [`ConfigParser`].

#![allow(dead_code)]

use std::collections::BTreeMap;

use clap::Parser;
use kata_core::config::ConfigParser;
use kata_core::fs;
use kata_core::global::StringError;
use kata_core::logger::Logger;
use kata_data::home;
use kata_program::setup::get_mutex_key_sets;

fn default_model_help() -> String {
    format!(
        "Neural net model file. Defaults to: {}/default_model.bin.gz",
        home::get_default_files_dir_for_help_message()
    )
}

fn default_config_help() -> String {
    "Config file(s) to use, can be one or multiple files.".to_string()
}

/// Common CLI arguments shared by most KataGo subcommands.
///
/// Mirrors `KataGoCommandLine` in `cpp/command/commandline.cpp`.
#[derive(Parser, Debug, Clone, Default)]
pub struct CommonArgs {
    /// Neural net model file.
    #[arg(long = "model", value_name = "FILE", help = default_model_help())]
    pub model: Option<String>,

    /// Human SL neural net model file.
    #[arg(long = "human-model", value_name = "FILE")]
    pub human_model: Option<String>,

    /// Config file(s) to use.
    #[arg(long = "config", value_name = "FILE", num_args = 1.., help = default_config_help())]
    pub config: Vec<String>,

    /// Override config parameters. Format: "key=value, key=value,..."
    #[arg(long = "override-config", value_name = "KEYVALUEPAIRS", num_args = 1..)]
    pub override_config: Vec<String>,
}

impl CommonArgs {
    /// Resolve the model file path, searching default locations if none was given.
    pub fn get_model_file(&self) -> Result<String, StringError> {
        if let Some(model) = &self.model {
            if !model.is_empty() {
                return Ok(model.clone());
            }
        }

        let paths = get_default_model_paths()?;
        let path_for_err_msg = paths
            .first()
            .cloned()
            .unwrap_or_else(get_default_model_path_for_help);
        for path in &paths {
            if fs::exists(path) {
                return Ok(path.clone());
            }
        }
        Err(StringError::new(format!(
            "-model MODELFILENAME.bin.gz was not specified to tell KataGo where to find the neural net model, and default was not found at {}",
            path_for_err_msg
        )))
    }

    /// True if no explicit model file was provided.
    pub fn model_file_is_default(&self) -> bool {
        self.model.as_ref().map(|s| s.is_empty()).unwrap_or(true)
    }

    /// Return the human SL model file, if any.
    pub fn get_human_model_file(&self) -> Option<String> {
        self.human_model.clone()
    }

    /// Resolve the config file paths, searching default locations if none were given.
    pub fn get_config_files(
        &self,
        default_config_file_name: &str,
    ) -> Result<Vec<String>, StringError> {
        if !self.config.is_empty() {
            return Ok(self.config.clone());
        }
        if default_config_file_name.is_empty() {
            return Ok(Vec::new());
        }

        let paths = get_default_config_paths(default_config_file_name)?;
        let path_for_err_msg = paths
            .first()
            .cloned()
            .unwrap_or_else(|| get_default_config_path_for_help(default_config_file_name));
        for path in &paths {
            if fs::exists(path) {
                return Ok(vec![path.clone()]);
            }
        }
        Err(StringError::new(format!(
            "-config CONFIG_FILE_NAME.cfg was not specified to tell KataGo where to find the config, and default was not found at {}",
            path_for_err_msg
        )))
    }

    /// Build a [`ConfigParser`] from the provided config files and overrides.
    pub fn get_config(&self, default_config_file_name: &str) -> Result<ConfigParser, StringError> {
        let config_files = self.get_config_files(default_config_file_name)?;
        if config_files.is_empty() {
            return Err(StringError::new(
                "No config file specified and no default available".to_string(),
            ));
        }

        let mut cfg = ConfigParser::new(false, false);
        cfg.initialize_file(&config_files[0])
            .map_err(|e| StringError::new(format!("Could not initialize config: {}", e)))?;
        for override_file in &config_files[1..] {
            cfg.override_keys_file(override_file).map_err(|e| {
                StringError::new(format!(
                    "Could not override config from {}: {}",
                    override_file, e
                ))
            })?;
        }
        self.maybe_apply_override_config_arg(&mut cfg)?;
        Ok(cfg)
    }

    /// Like [`Self::get_config`], but allows an empty config when no default is configured.
    pub fn get_config_allow_empty(
        &self,
        default_config_file_name: Option<&str>,
    ) -> Result<ConfigParser, StringError> {
        if self.config.is_empty() && default_config_file_name.is_none() {
            let mut cfg = ConfigParser::new(false, false);
            cfg.initialize_map(BTreeMap::new());
            self.maybe_apply_override_config_arg(&mut cfg)?;
            Ok(cfg)
        } else {
            self.get_config(default_config_file_name.unwrap_or(""))
        }
    }

    /// Apply `--override-config` key/value pairs to an existing config.
    pub fn maybe_apply_override_config_arg(
        &self,
        cfg: &mut ConfigParser,
    ) -> Result<(), StringError> {
        for override_config in &self.override_config {
            if override_config.is_empty() {
                continue;
            }
            let new_kvs = ConfigParser::parse_comma_separated(override_config)
                .map_err(|e| StringError::new(format!("Could not parse override-config: {}", e)))?;
            let mutex_key_sets = get_mutex_key_sets();
            cfg.override_keys_map_with_mutex(&new_kvs, &mutex_key_sets);
        }
        Ok(())
    }

    /// Log all `--override-config` key/value pairs.
    pub fn log_overrides(&self, logger: &Logger) {
        for override_config in &self.override_config {
            if override_config.is_empty() {
                continue;
            }
            if let Ok(new_kvs) = ConfigParser::parse_comma_separated(override_config) {
                for (key, value) in new_kvs {
                    logger.write(&format!("Config override: {} = {}", key, value));
                }
            }
        }
    }
}

fn get_default_model_path_for_help() -> String {
    format!(
        "{}/default_model.bin.gz",
        home::get_default_files_dir_for_help_message()
    )
}

fn get_default_model_paths() -> Result<Vec<String>, StringError> {
    let mut ret = Vec::new();
    for dir in home::get_default_files_dirs().map_err(|e| StringError::new(e.0.clone()))? {
        ret.push(format!("{}/default_model.bin.gz", dir));
        ret.push(format!("{}/default_model.txt.gz", dir));
    }
    Ok(ret)
}

fn get_default_config_path_for_help(default_config_file_name: &str) -> String {
    format!(
        "{}/{}",
        home::get_default_files_dir_for_help_message(),
        default_config_file_name
    )
}

fn get_default_config_paths(default_config_file_name: &str) -> Result<Vec<String>, StringError> {
    let mut ret = Vec::new();
    for dir in home::get_default_files_dirs().map_err(|e| StringError::new(e.0.clone()))? {
        ret.push(format!("{}/{}", dir, default_config_file_name));
    }
    Ok(ret)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_common_args_parsing() {
        let args = CommonArgs::parse_from([
            "katago",
            "--model",
            "model.bin.gz",
            "--config",
            "cfg.cfg",
            "--override-config",
            "foo=bar",
        ]);
        assert_eq!(args.model, Some("model.bin.gz".to_string()));
        assert_eq!(args.config, vec!["cfg.cfg".to_string()]);
        assert_eq!(args.override_config, vec!["foo=bar".to_string()]);
    }

    #[test]
    fn test_get_model_file_explicit() {
        let args = CommonArgs {
            model: Some("my-model.bin.gz".to_string()),
            ..CommonArgs::default()
        };
        assert_eq!(args.get_model_file().unwrap(), "my-model.bin.gz");
    }

    #[test]
    fn test_model_file_is_default() {
        let args = CommonArgs::default();
        assert!(args.model_file_is_default());
    }

    #[test]
    fn test_get_config_files_explicit() {
        let args = CommonArgs {
            config: vec!["a.cfg".to_string(), "b.cfg".to_string()],
            ..CommonArgs::default()
        };
        assert_eq!(
            args.get_config_files("default.cfg").unwrap(),
            vec!["a.cfg".to_string(), "b.cfg".to_string()]
        );
    }

    #[test]
    fn test_override_config_applies_mutex() {
        let mut cfg = ConfigParser::new(false, false);
        cfg.initialize_str("rules = chinese\nkoRule = positional\n")
            .unwrap();

        let args = CommonArgs {
            override_config: vec!["rules=japanese".to_string()],
            ..CommonArgs::default()
        };
        args.maybe_apply_override_config_arg(&mut cfg).unwrap();

        // The expanded rule keys should have been erased because "rules" was specified.
        assert!(cfg.get_string("rules").is_ok());
        assert!(cfg.get_string("koRule").is_err());
    }

    #[test]
    fn test_override_config_erases_rules() {
        let mut cfg = ConfigParser::new(false, false);
        cfg.initialize_str("koRule = positional\nscoringRule = area\n")
            .unwrap();

        let args = CommonArgs {
            override_config: vec!["koRule=situational".to_string()],
            ..CommonArgs::default()
        };
        args.maybe_apply_override_config_arg(&mut cfg).unwrap();

        // "rules" is not present, so only the conflicting expanded keys matter.
        assert_eq!(cfg.get_string("koRule").unwrap(), "situational");
    }
}
