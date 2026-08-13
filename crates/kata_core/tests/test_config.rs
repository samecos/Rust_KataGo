//! Integration tests ported from `cpp/tests/testconfig.cpp`.
//!
//! Covers `ConfigParser` behavior on real fixture files: inclusion, overriding,
//! circular inclusion, and parsing all example configs.

use kata_core::config::ConfigParser;
use std::path::PathBuf;

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/configs")
}

fn cpp_configs_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("KataGo")
        .join("cpp")
        .join("configs")
}

#[test]
fn run_config_tests() {
    let data = data_dir();

    {
        let cfg = ConfigParser::from_file(data.join("analysis_example.cfg"), false, true).unwrap();
        assert_eq!(cfg.get_int("nnMaxBatchSize", 0, 10000).unwrap(), 64);
    }

    {
        assert!(ConfigParser::from_file(data.join("test-duplicate.cfg"), false, true).is_err());
    }

    {
        let cfg = ConfigParser::from_file(data.join("test-duplicate.cfg"), true, true).unwrap();
        assert_eq!(cfg.get_string("logDir").unwrap(), "more_logs");
    }

    {
        assert!(ConfigParser::from_file(data.join("test.cfg"), false, false).is_err());
    }

    {
        let cfg = ConfigParser::from_file(data.join("test.cfg"), false, true).unwrap();
        assert!(cfg.contains("reportAnalysisWinratesAs"));
        assert_eq!(cfg.get_int("maxVisits", 0, 10000).unwrap(), 1000);
        assert_eq!(cfg.get_string("logDir").unwrap(), "more_logs");
        assert_eq!(cfg.get_int("nnMaxBatchSize", 0, 200_000).unwrap(), 100_500);
    }

    {
        assert!(ConfigParser::from_file(data.join("test-circular0.cfg"), false, true).is_err());
    }

    {
        let cfg =
            ConfigParser::from_file(data.join("folded/test-parent.cfg"), false, true).unwrap();
        assert_eq!(cfg.get_string("param").unwrap(), "value");
        assert_eq!(cfg.get_string("logDir").unwrap(), "more_logs");
    }

    // Approximate the C++ command-line multi-config test without invoking the
    // full CLI parser.
    {
        let mut cfg =
            ConfigParser::from_file(data.join("analysis_example.cfg"), false, true).unwrap();
        cfg.override_keys_file(data.join("test2.cfg").to_str().unwrap())
            .unwrap();
        assert!(cfg.contains("logDir"));
        assert_eq!(cfg.get_int("nnMaxBatchSize", 0, 200).unwrap(), 100);
    }
}

#[test]
fn run_parse_all_configs_test() {
    let dir = cpp_configs_dir();
    if !dir.exists() {
        return;
    }

    let mut paths: Vec<PathBuf> = Vec::new();

    fn collect_cfgs(dir: &std::path::Path, paths: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                collect_cfgs(&path, paths);
            } else if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("cfg") {
                paths.push(path);
            }
        }
    }

    collect_cfgs(&dir, &mut paths);

    paths.sort();
    for path in paths {
        let path_str = path.to_string_lossy();
        if path_str.contains("ringmaster") {
            continue;
        }
        let cfg = ConfigParser::from_file(&path, false, true).unwrap_or_else(|e| {
            panic!("Failed to parse example config {}: {}", path_str, e);
        });
        if !cfg.contains("password") {
            // Mirror C++ behavior: print the config contents when no password
            // key is present.
            println!("======================================================");
            println!("{}", path_str);
            println!("{}", cfg.all_key_vals());
        }
    }
}
