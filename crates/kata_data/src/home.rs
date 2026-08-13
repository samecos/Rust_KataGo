//! Home directory and default file location helpers.
//!
//! Corresponds to `cpp/dataio/homedata.h` and `cpp/dataio/homedata.cpp`.

use kata_core::global::IOError;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn current_exe_dir() -> Result<PathBuf, IOError> {
    let exe = env::current_exe().map_err(|e| {
        IOError(format!(
            "Could not find containing directory of executable: {}",
            e
        ))
    })?;
    exe.parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| IOError("Could not find containing directory of executable".to_string()))
}

fn ensure_dir(path: &Path) -> Result<(), IOError> {
    fs::create_dir_all(path).map_err(|e| {
        IOError(format!(
            "Could not create directory {}: {}",
            path.display(),
            e
        ))
    })
}

/// Returns directories for reading default files.
///
/// On Windows this is the executable's directory. On Unix it returns both the
/// executable's directory and the home data directory.
pub fn get_default_files_dirs() -> Result<Vec<String>, IOError> {
    let exe_dir = current_exe_dir()?;
    let exe_dir_str = exe_dir.to_string_lossy().into_owned();

    #[cfg(unix)]
    {
        let mut dirs = vec![exe_dir_str];
        if let Ok(home_dir) = get_home_data_dir(false, "") {
            dirs.push(home_dir);
        }
        Ok(dirs)
    }

    #[cfg(not(unix))]
    {
        Ok(vec![exe_dir_str])
    }
}

/// Returns a help-message string describing the default files directory.
pub fn get_default_files_dir_for_help_message() -> &'static str {
    #[cfg(windows)]
    {
        "(dir containing katago.exe)"
    }
    #[cfg(unix)]
    {
        "(dir containing katago.exe, or else ~/.katago)"
    }
    #[cfg(not(any(windows, unix)))]
    {
        "(dir containing executable)"
    }
}

/// Returns a directory suitable for writing automatically-generated data.
///
/// If `home_data_dir_override` is non-empty, it is used directly. Otherwise:
/// - On Windows: `<executable dir>/KataGoData`.
/// - On Unix: `$HOME/.katago`, falling back to `./.katago` if `HOME` is not set.
pub fn get_home_data_dir(make_dir: bool, home_data_dir_override: &str) -> Result<String, IOError> {
    if !home_data_dir_override.is_empty() {
        let path = Path::new(home_data_dir_override);
        if make_dir {
            ensure_dir(path)?;
        }
        return Ok(path.to_string_lossy().into_owned());
    }

    #[cfg(windows)]
    {
        let mut path = current_exe_dir()?;
        path.push("KataGoData");
        if make_dir {
            ensure_dir(&path)?;
        }
        Ok(path.to_string_lossy().into_owned())
    }

    #[cfg(unix)]
    {
        let home = env::var("HOME");
        let path = if let Ok(home) = home {
            PathBuf::from(home).join(".katago")
        } else {
            PathBuf::from("./.katago")
        };
        if make_dir {
            ensure_dir(&path)?;
        }
        Ok(path.to_string_lossy().into_owned())
    }

    #[cfg(not(any(windows, unix)))]
    {
        let mut path = current_exe_dir()?;
        path.push("KataGoData");
        if make_dir {
            ensure_dir(&path)?;
        }
        Ok(path.to_string_lossy().into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_override_home_data_dir() {
        let tmp = std::env::temp_dir().join(format!("katago_home_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        assert!(!tmp.exists());
        let dir = get_home_data_dir(true, tmp.to_str().unwrap()).unwrap();
        assert!(tmp.exists());
        assert_eq!(dir, tmp.to_string_lossy());
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_override_no_make_dir() {
        let tmp =
            std::env::temp_dir().join(format!("katago_home_test_no_mkdir_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let dir = get_home_data_dir(false, tmp.to_str().unwrap()).unwrap();
        assert!(!tmp.exists());
        assert_eq!(dir, tmp.to_string_lossy());
    }

    #[test]
    fn test_default_files_dirs_non_empty() {
        let dirs = get_default_files_dirs().unwrap();
        assert!(!dirs.is_empty());
        for d in &dirs {
            assert!(!d.is_empty());
        }
    }

    #[test]
    fn test_help_message_non_empty() {
        assert!(!get_default_files_dir_for_help_message().is_empty());
    }
}
