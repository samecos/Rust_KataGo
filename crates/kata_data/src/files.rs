//! File collection helpers for SGF, multi-SGF, and position files.
//!
//! Corresponds to `cpp/dataio/files.h` and `cpp/dataio/files.cpp`.

use kata_core::global::{self, IOError};
use std::fs;
use std::path::Path;
use std::time::SystemTime;

const SGF_SUFFIX: &str = ".sgf";
const SGF_SUFFIX_UPPER: &str = ".SGF";
const MULTI_SGF_SUFFIX: &str = ".sgfs";
const MULTI_SGF_SUFFIX_UPPER: &str = ".SGFS";
const POS_SUFFIX: &str = "poses.txt";

fn is_sgf(name: &str) -> bool {
    name.ends_with(SGF_SUFFIX) || name.ends_with(SGF_SUFFIX_UPPER)
}

fn is_multi_sgfs(name: &str) -> bool {
    name.ends_with(MULTI_SGF_SUFFIX) || name.ends_with(MULTI_SGF_SUFFIX_UPPER)
}

fn is_pos_file(name: &str) -> bool {
    name.ends_with(POS_SUFFIX)
}

type FileFilter = fn(&str) -> bool;

fn collect_files_matching(
    dir: &Path,
    filter: FileFilter,
    collected: &mut Vec<String>,
) -> Result<(), IOError> {
    let entries = fs::read_dir(dir)
        .map_err(|e| IOError(format!("Error reading directory {}: {}", dir.display(), e)))?;

    for entry in entries {
        let entry = entry.map_err(|e| {
            IOError(format!(
                "Error reading entry in directory {}: {}",
                dir.display(),
                e
            ))
        })?;
        let path = entry.path();
        let file_type = entry.file_type().map_err(|e| {
            IOError(format!(
                "Error getting file type for {}: {}",
                path.display(),
                e
            ))
        })?;

        if file_type.is_dir() {
            collect_files_matching(&path, filter, collected)?;
        } else if file_type.is_file() {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if filter(name) {
                    collected.push(path.to_string_lossy().into_owned());
                }
            }
        }
    }
    Ok(())
}

fn collect_from_dir_or_file(
    dir_or_file: &str,
    filter: FileFilter,
    file_kind: &str,
    expected_suffix: &str,
    collected: &mut Vec<String>,
) -> Result<(), IOError> {
    let path = Path::new(dir_or_file);
    if path.exists() && !path.is_dir() {
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if filter(name) {
                collected.push(dir_or_file.to_string());
            } else {
                return Err(IOError(format!(
                    "Error collecting {} files: File does not end in {}: {}",
                    file_kind, expected_suffix, dir_or_file
                )));
            }
        }
        return Ok(());
    }
    collect_files_matching(path, filter, collected)
}

/// Returns true if the file name ends with `.sgfs`/`.SGFS`.
pub fn is_multi_sgfs_file(name: &str) -> bool {
    is_multi_sgfs(name)
}

/// Recursively collect `.sgf`/`.SGF` files from a directory.
pub fn collect_sgfs_from_dir(dir: &str, collected: &mut Vec<String>) -> Result<(), IOError> {
    collect_files_matching(Path::new(dir), is_sgf, collected)
}

/// Collect a single SGF file or recursively collect SGFs from a directory.
pub fn collect_sgfs_from_dir_or_file(
    dir_or_file: &str,
    collected: &mut Vec<String>,
) -> Result<(), IOError> {
    collect_from_dir_or_file(dir_or_file, is_sgf, "sgf", ".sgf or .SGF", collected)
}

/// Collect SGFs from multiple directories or files.
pub fn collect_sgfs_from_dirs(
    dirs: &[impl AsRef<str>],
    collected: &mut Vec<String>,
) -> Result<(), IOError> {
    for d in dirs {
        let original = d.as_ref();
        let trimmed = global::trim(original);
        if trimmed.is_empty() {
            continue;
        }
        let to_use = if Path::new(original).exists() {
            original
        } else {
            trimmed
        };
        collect_sgfs_from_dir(to_use, collected)?;
    }
    Ok(())
}

/// Collect SGFs from multiple directories or files, allowing each entry to be either.
pub fn collect_sgfs_from_dirs_or_files(
    dirs_or_files: &[impl AsRef<str>],
    collected: &mut Vec<String>,
) -> Result<(), IOError> {
    for d in dirs_or_files {
        let original = d.as_ref();
        let trimmed = global::trim(original);
        if trimmed.is_empty() {
            continue;
        }
        let to_use = if Path::new(original).exists() {
            original
        } else {
            trimmed
        };
        collect_sgfs_from_dir_or_file(to_use, collected)?;
    }
    Ok(())
}

/// Recursively collect `.sgfs`/`.SGFS` files from a directory.
pub fn collect_multi_sgfs_from_dir(dir: &str, collected: &mut Vec<String>) -> Result<(), IOError> {
    collect_files_matching(Path::new(dir), is_multi_sgfs, collected)
}

/// Collect a single multi-SGF file or recursively collect multi-SGFs from a directory.
pub fn collect_multi_sgfs_from_dir_or_file(
    dir_or_file: &str,
    collected: &mut Vec<String>,
) -> Result<(), IOError> {
    collect_from_dir_or_file(
        dir_or_file,
        is_multi_sgfs,
        "sgfs",
        ".sgfs or .SGFS",
        collected,
    )
}

/// Collect multi-SGFs from multiple directories or files.
pub fn collect_multi_sgfs_from_dirs(
    dirs: &[impl AsRef<str>],
    collected: &mut Vec<String>,
) -> Result<(), IOError> {
    for d in dirs {
        let original = d.as_ref();
        let trimmed = global::trim(original);
        if trimmed.is_empty() {
            continue;
        }
        let to_use = if Path::new(original).exists() {
            original
        } else {
            trimmed
        };
        collect_multi_sgfs_from_dir(to_use, collected)?;
    }
    Ok(())
}

/// Collect multi-SGFs from multiple directories or files, allowing each entry to be either.
pub fn collect_multi_sgfs_from_dirs_or_files(
    dirs_or_files: &[impl AsRef<str>],
    collected: &mut Vec<String>,
) -> Result<(), IOError> {
    for d in dirs_or_files {
        let original = d.as_ref();
        let trimmed = global::trim(original);
        if trimmed.is_empty() {
            continue;
        }
        let to_use = if Path::new(original).exists() {
            original
        } else {
            trimmed
        };
        collect_multi_sgfs_from_dir_or_file(to_use, collected)?;
    }
    Ok(())
}

/// Recursively collect `poses.txt` files from a directory.
pub fn collect_poses_from_dir(dir: &str, collected: &mut Vec<String>) -> Result<(), IOError> {
    collect_files_matching(Path::new(dir), is_pos_file, collected)
}

/// Collect a single position file or recursively collect position files from a directory.
pub fn collect_poses_from_dir_or_file(
    dir_or_file: &str,
    collected: &mut Vec<String>,
) -> Result<(), IOError> {
    collect_from_dir_or_file(dir_or_file, is_pos_file, "pos", "poses.txt", collected)
}

/// Collect position files from multiple directories or files.
pub fn collect_poses_from_dirs(
    dirs: &[impl AsRef<str>],
    collected: &mut Vec<String>,
) -> Result<(), IOError> {
    for d in dirs {
        let original = d.as_ref();
        let trimmed = global::trim(original);
        if trimmed.is_empty() {
            continue;
        }
        let to_use = if Path::new(original).exists() {
            original
        } else {
            trimmed
        };
        collect_poses_from_dir(to_use, collected)?;
    }
    Ok(())
}

/// Collect position files from multiple directories or files, allowing each entry to be either.
pub fn collect_poses_from_dirs_or_files(
    dirs_or_files: &[impl AsRef<str>],
    collected: &mut Vec<String>,
) -> Result<(), IOError> {
    for d in dirs_or_files {
        let original = d.as_ref();
        let trimmed = global::trim(original);
        if trimmed.is_empty() {
            continue;
        }
        let to_use = if Path::new(original).exists() {
            original
        } else {
            trimmed
        };
        collect_poses_from_dir_or_file(to_use, collected)?;
    }
    Ok(())
}

/// Sort files by modification time, newest first.
pub fn sort_newest_to_oldest(files: &mut [String]) -> Result<(), IOError> {
    let mut with_time: Vec<(String, SystemTime)> = Vec::with_capacity(files.len());
    for path in files.iter() {
        let metadata = fs::metadata(path)
            .map_err(|e| IOError(format!("Error getting metadata for {}: {}", path, e)))?;
        let modified = metadata
            .modified()
            .map_err(|e| IOError(format!("Error getting modified time for {}: {}", path, e)))?;
        with_time.push((path.clone(), modified));
    }
    with_time.sort_by_key(|b| std::cmp::Reverse(b.1));
    for (i, (path, _)) in with_time.into_iter().enumerate() {
        files[i] = path;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::path::PathBuf;
    use std::thread;
    use std::time::Duration;

    fn tmp_dir(prefix: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "katago_files_test_{}_{}",
            prefix,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn test_is_multi_sgfs() {
        assert!(is_multi_sgfs_file("foo.sgfs"));
        assert!(is_multi_sgfs_file("foo.SGFS"));
        assert!(!is_multi_sgfs_file("foo.sgf"));
    }

    #[test]
    fn test_collect_sgfs_recursive() {
        let root = tmp_dir("sgfs");
        let sub = root.join("sub");
        fs::create_dir_all(&sub).unwrap();
        File::create(root.join("a.sgf")).unwrap();
        File::create(root.join("b.SGF")).unwrap();
        File::create(root.join("not.txt")).unwrap();
        File::create(sub.join("c.sgf")).unwrap();

        let mut collected = Vec::new();
        collect_sgfs_from_dir(root.to_str().unwrap(), &mut collected).unwrap();
        assert_eq!(collected.len(), 3);
        assert!(collected.iter().any(|p| p.ends_with("a.sgf")));
        assert!(collected.iter().any(|p| p.ends_with("b.SGF")));
        assert!(collected.iter().any(|p| p.ends_with("c.sgf")));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_collect_sgfs_from_dir_or_file() {
        let root = tmp_dir("sgfs_dof");
        let file = root.join("single.sgf");
        File::create(&file).unwrap();

        let mut collected = Vec::new();
        collect_sgfs_from_dir_or_file(file.to_str().unwrap(), &mut collected).unwrap();
        assert_eq!(collected.len(), 1);

        let bad = root.join("bad.txt");
        File::create(&bad).unwrap();
        let mut collected2 = Vec::new();
        assert!(collect_sgfs_from_dir_or_file(bad.to_str().unwrap(), &mut collected2).is_err());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_collect_poses() {
        let root = tmp_dir("poses");
        File::create(root.join("game1_poses.txt")).unwrap();
        File::create(root.join("game2_poses.txt")).unwrap();
        File::create(root.join("game.sgf")).unwrap();

        let mut collected = Vec::new();
        collect_poses_from_dir(root.to_str().unwrap(), &mut collected).unwrap();
        assert_eq!(collected.len(), 2);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_sort_newest_to_oldest() {
        let root = tmp_dir("sort");
        let a = root.join("a.txt");
        let b = root.join("b.txt");
        let c = root.join("c.txt");
        File::create(&a).unwrap();
        thread::sleep(Duration::from_millis(60));
        File::create(&b).unwrap();
        thread::sleep(Duration::from_millis(60));
        File::create(&c).unwrap();

        let mut files = vec![
            a.to_string_lossy().into_owned(),
            b.to_string_lossy().into_owned(),
            c.to_string_lossy().into_owned(),
        ];
        sort_newest_to_oldest(&mut files).unwrap();
        assert!(files[0].ends_with("c.txt"));
        assert!(files[1].ends_with("b.txt"));
        assert!(files[2].ends_with("a.txt"));

        let _ = fs::remove_dir_all(&root);
    }
}
