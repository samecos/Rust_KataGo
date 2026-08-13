//! Model file discovery and cleanup helpers.
//!
//! Corresponds to `cpp/dataio/loadmodel.h` and `cpp/dataio/loadmodel.cpp`.

use kata_core::global::IOError;
use kata_core::logger::Logger;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const ACCEPTABLE_MODEL_SUFFIXES: &[&str] = &[".bin.gz", ".bin", "model.txt.gz", "model.txt"];

const GENERIC_MODEL_NAMES: &[&str] = &[
    "model.bin.gz",
    "model.bin",
    "model.txt.gz",
    "model.txt",
    "Model.bin.gz",
    "Model.bin",
    "Model.txt.gz",
    "Model.txt",
    "MODEL.bin.gz",
    "MODEL.bin",
    "MODEL.txt.gz",
    "MODEL.txt",
    "model.ckpt",
    "Model.ckpt",
    "MODEL.ckpt",
    "model.checkpoint",
    "Model.checkpoint",
    "MODEL.checkpoint",
    "model",
    "Model",
    "MODEL",
];

fn has_acceptable_suffix(path: &str) -> bool {
    ACCEPTABLE_MODEL_SUFFIXES.iter().any(|s| path.ends_with(s))
}

fn is_generic_model_name(name: &str) -> bool {
    GENERIC_MODEL_NAMES.contains(&name)
}

fn system_time_to_time_t(t: SystemTime) -> i64 {
    t.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64
}

/// Information about the latest model file found in a directory tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    pub model_name: String,
    pub model_file: String,
    pub model_dir: String,
    pub model_time: i64,
}

/// Recursively find the newest model file in `models_dir`.
pub fn find_latest_model(models_dir: &str, _logger: &Logger) -> Result<Option<ModelInfo>, IOError> {
    let root = Path::new(models_dir);
    if !root.exists() {
        return Ok(None);
    }

    let mut latest: Option<(PathBuf, SystemTime)> = None;
    for entry in walkdir(models_dir)? {
        let meta = fs::metadata(&entry).map_err(|e| {
            IOError(format!(
                "Could not read metadata for {}: {}",
                entry.display(),
                e
            ))
        })?;
        if !meta.is_file() {
            continue;
        }
        let name = entry.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !has_acceptable_suffix(name) {
            continue;
        }
        let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        match latest {
            None => latest = Some((entry, modified)),
            Some((_, t)) if modified > t => latest = Some((entry, modified)),
            _ => {}
        }
    }

    let Some((path, modified)) = latest else {
        return Ok(Some(ModelInfo {
            model_name: "random".to_string(),
            model_file: "/dev/null".to_string(),
            model_dir: "/dev/null".to_string(),
            model_time: 0,
        }));
    };

    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model")
        .to_string();
    let model_name = if is_generic_model_name(&file_name) {
        path.parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("model")
            .to_string()
    } else {
        file_name
    };
    let model_file = path.to_string_lossy().into_owned();
    let model_dir = path
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| models_dir.to_string());
    let model_time = system_time_to_time_t(modified);

    Ok(Some(ModelInfo {
        model_name,
        model_file,
        model_dir,
        model_time,
    }))
}

/// Set a file's last-modified time to now.
pub fn set_last_modified_time_to_now(file_path: &str, logger: &Logger) {
    let now = SystemTime::now();
    match fs::OpenOptions::new().write(true).open(file_path) {
        Ok(file) => {
            if let Err(e) = file.set_modified(now) {
                logger.write(&format!(
                    "Warning: could not set last modified time for {}: {}",
                    file_path, e
                ));
            }
        }
        Err(e) => {
            logger.write(&format!(
                "Warning: could not open {} to set last modified time: {}",
                file_path, e
            ));
        }
    }
}

/// Delete model files in `models_dir` whose modification time is older than `time`.
pub fn delete_models_older_than(models_dir: &str, logger: &Logger, time: i64) {
    let root = Path::new(models_dir);
    if !root.exists() {
        return;
    }
    let entries = match fs::read_dir(root) {
        Ok(e) => e,
        Err(e) => {
            logger.write(&format!(
                "Warning: could not read models directory {}: {}",
                models_dir, e
            ));
            return;
        }
    };

    let cutoff = UNIX_EPOCH + std::time::Duration::from_secs(time as u64);
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                logger.write(&format!(
                    "Warning: could not read entry in {}: {}",
                    models_dir, e
                ));
                continue;
            }
        };
        let path = entry.path();
        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(e) => {
                logger.write(&format!(
                    "Warning: could not read metadata for {}: {}",
                    path.display(),
                    e
                ));
                continue;
            }
        };
        if !meta.is_file() {
            continue;
        }
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !(name.ends_with(".bin.gz")
            || name.ends_with(".txt.gz")
            || name.ends_with(".bin")
            || name.ends_with(".txt"))
        {
            continue;
        }
        let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        if modified < cutoff {
            logger.write(&format!(
                "Deleting old unused model file: {}",
                path.display()
            ));
            if let Err(e) = fs::remove_file(&path) {
                logger.write(&format!(
                    "Warning: could not delete {}: {}",
                    path.display(),
                    e
                ));
            }
        }
    }
}

fn walkdir(dir: &str) -> Result<Vec<PathBuf>, IOError> {
    let mut result = Vec::new();
    let mut stack = vec![PathBuf::from(dir)];
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir)
            .map_err(|e| IOError(format!("Could not read directory {}: {}", dir.display(), e)))?;
        for entry in entries {
            let entry = entry.map_err(|e| {
                IOError(format!("Could not read entry in {}: {}", dir.display(), e))
            })?;
            let path = entry.path();
            let file_type = entry.file_type().map_err(|e| {
                IOError(format!(
                    "Could not get file type for {}: {}",
                    path.display(),
                    e
                ))
            })?;
            if file_type.is_dir() {
                stack.push(path);
            } else {
                result.push(path);
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kata_core::logger::{Logger, LoggerOptions};
    use std::fs::File;
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

    #[test]
    fn test_find_latest_model() {
        let root = std::env::temp_dir().join(format!("katago_model_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();

        let old = root.join("old.bin.gz");
        let latest = root.join("latest.bin.gz");
        File::create(&old).unwrap();
        thread::sleep(Duration::from_millis(60));
        File::create(&latest).unwrap();

        let info = find_latest_model(root.to_str().unwrap(), &logger())
            .unwrap()
            .unwrap();
        assert!(info.model_file.ends_with("latest.bin.gz"));
        assert_eq!(info.model_name, "latest.bin.gz");
        assert_eq!(info.model_dir, root.to_string_lossy());
        assert!(info.model_time > 0);

        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn test_find_latest_model_generic_name() {
        let root =
            std::env::temp_dir().join(format!("katago_model_generic_test_{}", std::process::id()));
        let sub = root.join("my-model");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&sub).unwrap();

        File::create(sub.join("model.bin.gz")).unwrap();

        let info = find_latest_model(root.to_str().unwrap(), &logger())
            .unwrap()
            .unwrap();
        assert_eq!(info.model_name, "my-model");
        assert!(info.model_file.ends_with("model.bin.gz"));

        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn test_find_latest_model_empty_dir() {
        let root =
            std::env::temp_dir().join(format!("katago_model_empty_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();

        let info = find_latest_model(root.to_str().unwrap(), &logger())
            .unwrap()
            .unwrap();
        assert_eq!(info.model_name, "random");
        assert_eq!(info.model_file, "/dev/null");
        assert_eq!(info.model_dir, "/dev/null");
        assert_eq!(info.model_time, 0);

        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn test_delete_models_older_than() {
        let root =
            std::env::temp_dir().join(format!("katago_model_delete_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();

        let old = root.join("old.bin.gz");
        let keep = root.join("keep.bin.gz");
        File::create(&old).unwrap();
        File::create(&keep).unwrap();

        let old_time = UNIX_EPOCH + Duration::from_secs(1_000);
        let keep_time = UNIX_EPOCH + Duration::from_secs(2_000);
        fs::OpenOptions::new()
            .write(true)
            .open(&old)
            .unwrap()
            .set_modified(old_time)
            .unwrap();
        fs::OpenOptions::new()
            .write(true)
            .open(&keep)
            .unwrap()
            .set_modified(keep_time)
            .unwrap();

        delete_models_older_than(root.to_str().unwrap(), &logger(), 1_500);

        assert!(!old.exists());
        assert!(keep.exists());

        fs::remove_dir_all(&root).unwrap();
    }
}
