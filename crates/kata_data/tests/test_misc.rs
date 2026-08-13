//! Integration tests ported from `cpp/tests/testmisc.cpp`.
//!
//! Covers file collection helpers (`collectSgfsFromDir`, `collectFiles`) and
//! model discovery helpers (`findLatestModel`, `setLastModifiedTimeToNow`).

use kata_core::fs::weakly_canonical;
use kata_core::logger::{Logger, LoggerOptions};
use kata_data::files::collect_sgfs_from_dir;
use kata_data::model_loader::{find_latest_model, set_last_modified_time_to_now};
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

fn logger() -> Logger {
    Logger::new(
        LoggerOptions {
            log_to_stdout: false,
            log_to_stderr: false,
            log_time: false,
        },
        None,
    )
}

fn tmp_dir(prefix: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "katago_misc_test_{}_{}",
        prefix,
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn run_collect_files_tests() {
    let root = tmp_dir("collect");
    let sub = root.join("sub");
    fs::create_dir_all(&sub).unwrap();

    fs::File::create(root.join("a.sgf")).unwrap();
    fs::File::create(root.join("b.SGF")).unwrap();
    fs::File::create(root.join("not.txt")).unwrap();
    fs::File::create(sub.join("c.sgf")).unwrap();
    fs::File::create(root.join("x.cfg")).unwrap();
    fs::File::create(root.join("y.CFG")).unwrap();
    fs::File::create(sub.join("z.cfg")).unwrap();

    {
        let mut collected = Vec::new();
        collect_sgfs_from_dir(root.to_str().unwrap(), &mut collected).unwrap();
        collected.sort();
        assert_eq!(collected.len(), 3);
        assert!(collected.iter().any(|p| p.ends_with("a.sgf")));
        assert!(collected.iter().any(|p| p.ends_with("b.SGF")));
        assert!(collected.iter().any(|p| p.ends_with("c.sgf")));
    }

    {
        let collected =
            kata_core::fs::collect_files(&root, &|s| s.ends_with(".cfg") || s.ends_with(".CFG"))
                .unwrap();
        let mut collected = collected;
        collected.sort();
        assert_eq!(collected.len(), 3);
        assert!(collected.iter().any(|p| p.ends_with("x.cfg")));
        assert!(collected.iter().any(|p| p.ends_with("y.CFG")));
        assert!(collected.iter().any(|p| p.ends_with("z.cfg")));
    }

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn run_load_model_tests() {
    let models_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/models");

    {
        let dir = models_dir.join("findLatestModelTest1");
        let info = find_latest_model(dir.to_str().unwrap(), &logger())
            .unwrap()
            .unwrap();
        assert_eq!(info.model_time, 0);
        assert_eq!(info.model_name, "random");
        assert_eq!(info.model_dir, "/dev/null");
        assert_eq!(info.model_file, "/dev/null");
    }

    {
        let dir = models_dir.join("findLatestModelTest2");
        let info = find_latest_model(dir.to_str().unwrap(), &logger())
            .unwrap()
            .unwrap();
        assert!(info.model_time > 0);
        assert_eq!(info.model_name, "abc.bin.gz");
        assert_eq!(weakly_canonical(&info.model_dir), weakly_canonical(&dir));
        assert!(weakly_canonical(&info.model_dir).starts_with(weakly_canonical(&dir)));
        assert!(
            Path::new(&info.model_file)
                .file_name()
                .is_some_and(|n| n == "abc.bin.gz")
        );
    }

    {
        let dir = models_dir.join("findLatestModelTest3");
        let info = find_latest_model(dir.to_str().unwrap(), &logger())
            .unwrap()
            .unwrap();
        assert!(info.model_time > 0);
        assert_eq!(info.model_name, "def");
        assert!(Path::new(&info.model_dir).ends_with("def"));
        assert!(
            Path::new(&info.model_file)
                .file_name()
                .is_some_and(|n| n == "model.bin.gz")
        );
        assert_ne!(weakly_canonical(&info.model_dir), weakly_canonical(&dir));
        assert!(weakly_canonical(&info.model_dir).starts_with(weakly_canonical(&dir)));
    }

    {
        let dir = models_dir.join("findLatestModelTest4");
        let file = dir.join("abc.bin.gz");
        set_last_modified_time_to_now(file.to_str().unwrap(), &logger());

        let info = find_latest_model(dir.to_str().unwrap(), &logger())
            .unwrap()
            .unwrap();
        assert!(info.model_time > 0);
        assert_eq!(info.model_name, "abc.bin.gz");
        assert_eq!(weakly_canonical(&info.model_dir), weakly_canonical(&dir));
        assert!(weakly_canonical(&info.model_dir).starts_with(weakly_canonical(&dir)));
        assert!(
            Path::new(&info.model_file)
                .file_name()
                .is_some_and(|n| n == "abc.bin.gz")
        );
    }

    thread::sleep(Duration::from_millis(1500));
    {
        let dir = models_dir.join("findLatestModelTest4");
        let file = dir.join("def/model.bin.gz");
        set_last_modified_time_to_now(file.to_str().unwrap(), &logger());

        let info = find_latest_model(dir.to_str().unwrap(), &logger())
            .unwrap()
            .unwrap();
        assert!(info.model_time > 0);
        assert_eq!(info.model_name, "def");
        assert!(Path::new(&info.model_dir).ends_with("def"));
        assert!(
            Path::new(&info.model_file)
                .file_name()
                .is_some_and(|n| n == "model.bin.gz")
        );
        assert_ne!(weakly_canonical(&info.model_dir), weakly_canonical(&dir));
        assert!(weakly_canonical(&info.model_dir).starts_with(weakly_canonical(&dir)));
    }

    thread::sleep(Duration::from_millis(1500));
    {
        let dir = models_dir.join("findLatestModelTest4");
        let file = dir.join("def/ghi.bin.gz");
        set_last_modified_time_to_now(file.to_str().unwrap(), &logger());

        let info = find_latest_model(dir.to_str().unwrap(), &logger())
            .unwrap()
            .unwrap();
        assert!(info.model_time > 0);
        assert_eq!(info.model_name, "ghi.bin.gz");
        assert!(Path::new(&info.model_dir).ends_with("def"));
        assert!(
            Path::new(&info.model_file)
                .file_name()
                .is_some_and(|n| n == "ghi.bin.gz")
        );
        assert_ne!(weakly_canonical(&info.model_dir), weakly_canonical(&dir));
        assert!(weakly_canonical(&info.model_dir).starts_with(weakly_canonical(&dir)));
    }
}
