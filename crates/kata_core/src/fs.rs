//! File system helpers.
//!
//! Corresponds to `cpp/core/fileutils.h`, `cpp/core/fileutils.cpp`,
//! `cpp/core/makedir.h`, and `cpp/core/makedir.cpp`.

use std::fs;
use std::io::{self, BufRead, Read};
use std::path::{Component, Path, PathBuf};

use crate::global;
use crate::hash::sha2::sha256_hex;

// ---------------------------------------------------------------------------
// Basic queries
// ---------------------------------------------------------------------------

/// Returns `true` if `path` exists and is accessible.
pub fn exists<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().exists()
}

/// Returns `true` if `path` is an existing directory.
pub fn is_directory<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().is_dir()
}

// ---------------------------------------------------------------------------
// Directory creation
// ---------------------------------------------------------------------------

/// Create a directory if it does not already exist.
pub fn make_dir<P: AsRef<Path>>(path: P) -> Result<(), global::StringError> {
    let path = path.as_ref();
    match fs::create_dir(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => Ok(()),
        Err(e) => Err(global::StringError::new(format!(
            "Error creating directory {}: {}",
            path.display(),
            e
        ))),
    }
}

// ---------------------------------------------------------------------------
// File removal and renaming
// ---------------------------------------------------------------------------

/// Try to remove a file. Returns `false` on failure.
pub fn try_remove_file<P: AsRef<Path>>(path: P) -> bool {
    fs::remove_file(path.as_ref()).is_ok()
}

/// Try to rename a file. Returns `false` on failure.
pub fn try_rename<P: AsRef<Path>, Q: AsRef<Path>>(src: P, dst: Q) -> bool {
    fs::rename(src.as_ref(), dst.as_ref()).is_ok()
}

/// Rename a file, raising an error on failure.
pub fn rename<P: AsRef<Path>, Q: AsRef<Path>>(src: P, dst: Q) -> Result<(), global::IOError> {
    fs::rename(src.as_ref(), dst.as_ref()).map_err(|e| {
        global::IOError(format!(
            "Could not rename {} to {}: {}",
            src.as_ref().display(),
            dst.as_ref().display(),
            e
        ))
    })
}

// ---------------------------------------------------------------------------
// Path normalization
// ---------------------------------------------------------------------------

/// A weak version of `canonicalize`: resolves `.`, `..`, and existing symlink
/// prefixes, but does not require the entire path to exist.
pub fn weakly_canonical<P: AsRef<Path>>(path: P) -> PathBuf {
    let path = path.as_ref();

    // If the path exists, std::fs::canonicalize is the most correct answer.
    if let Ok(canonical) = fs::canonicalize(path) {
        return canonical;
    }

    // Otherwise normalize lexically, using the current directory for relative
    // paths.
    let base = if path.is_absolute() {
        PathBuf::from("/")
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    };

    let mut result = base;
    for component in path.components() {
        match component {
            Component::Prefix(_) | Component::RootDir => {
                result = PathBuf::from(component.as_os_str());
            }
            Component::CurDir => {}
            Component::ParentDir => {
                result.pop();
            }
            Component::Normal(name) => {
                result.push(name);
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Reading files
// ---------------------------------------------------------------------------

fn io_error<E: std::fmt::Display, P: AsRef<Path>>(msg: &str, path: P, err: E) -> global::IOError {
    global::IOError(format!("{} {}: {}", msg, path.as_ref().display(), err))
}

/// Read an entire text file into a `String`.
pub fn read_file<P: AsRef<Path>>(path: P) -> Result<String, global::IOError> {
    fs::read_to_string(path.as_ref()).map_err(|e| io_error("Could not read file", path, e))
}

/// Read an entire file into a byte vector.
pub fn read_file_binary<P: AsRef<Path>>(path: P) -> Result<Vec<u8>, global::IOError> {
    fs::read(path.as_ref()).map_err(|e| io_error("Could not read file", path, e))
}

/// Read a file and split it on `delimiter`. The delimiter is not included.
pub fn read_file_lines<P: AsRef<Path>>(
    path: P,
    delimiter: u8,
) -> Result<Vec<String>, global::IOError> {
    let file =
        fs::File::open(path.as_ref()).map_err(|e| io_error("Could not read file", &path, e))?;
    let reader = io::BufReader::new(file);
    let mut lines = Vec::new();
    for part in reader.split(delimiter) {
        let part = part.map_err(|e| io_error("Could not read file", &path, e))?;
        lines.push(String::from_utf8_lossy(&part).into_owned());
    }
    Ok(lines)
}

// ---------------------------------------------------------------------------
// Loading with optional SHA-256 verification
// ---------------------------------------------------------------------------

fn verify_sha256(
    data: &[u8],
    expected_sha256: &str,
    actual_sha256: Option<&mut String>,
    path: &Path,
) -> Result<(), global::IOError> {
    let hash = sha256_hex(data);
    if let Some(buf) = actual_sha256 {
        buf.clone_from(&hash);
    }
    if !expected_sha256.is_empty() {
        let expected = global::to_lower(expected_sha256);
        if expected != hash {
            return Err(global::IOError(format!(
                "File {} sha256 was {} which does not match the expected sha256 {}",
                path.display(),
                hash,
                expected_sha256
            )));
        }
    }
    Ok(())
}

/// Read a file into a byte vector, optionally verifying its SHA-256.
pub fn load_file_into_bytes<P: AsRef<Path>>(
    path: P,
    expected_sha256: &str,
) -> Result<Vec<u8>, global::IOError> {
    let path = path.as_ref();
    let data = read_file_binary(path)?;
    verify_sha256(&data, expected_sha256, None, path)?;
    Ok(data)
}

/// Read a file into a byte vector, optionally verifying its SHA-256 and
/// returning the computed hash.
pub fn load_file_into_bytes_with_hash<P: AsRef<Path>>(
    path: P,
    expected_sha256: &str,
) -> Result<(Vec<u8>, Option<String>), global::IOError> {
    let path = path.as_ref();
    let data = read_file_binary(path)?;
    let mut actual_hash = String::new();
    verify_sha256(&data, expected_sha256, Some(&mut actual_hash), path)?;
    let hash = if actual_hash.is_empty() {
        None
    } else {
        Some(actual_hash)
    };
    Ok((data, hash))
}

/// Read a gzip-compressed file into a byte vector, optionally verifying the
/// SHA-256 of the uncompressed contents.
pub fn uncompress_and_load_file_into_bytes<P: AsRef<Path>>(
    path: P,
    expected_sha256: &str,
) -> Result<Vec<u8>, global::IOError> {
    let path = path.as_ref();
    let compressed = read_file_binary(path)?;
    let mut decoder = flate2::read::GzDecoder::new(&compressed[..]);
    let mut data = Vec::new();
    decoder.read_to_end(&mut data).map_err(|e| {
        global::IOError(format!(
            "Error while ungzipping file {}: {}",
            path.display(),
            e
        ))
    })?;
    verify_sha256(&data, expected_sha256, None, path)?;
    Ok(data)
}

/// Read a gzip-compressed file into a byte vector, optionally verifying the
/// SHA-256 of the uncompressed contents and returning the computed hash.
pub fn uncompress_and_load_file_into_bytes_with_hash<P: AsRef<Path>>(
    path: P,
    expected_sha256: &str,
) -> Result<(Vec<u8>, Option<String>), global::IOError> {
    let path = path.as_ref();
    let compressed = read_file_binary(path)?;
    let mut decoder = flate2::read::GzDecoder::new(&compressed[..]);
    let mut data = Vec::new();
    decoder.read_to_end(&mut data).map_err(|e| {
        global::IOError(format!(
            "Error while ungzipping file {}: {}",
            path.display(),
            e
        ))
    })?;
    let mut actual_hash = String::new();
    verify_sha256(&data, expected_sha256, Some(&mut actual_hash), path)?;
    let hash = if actual_hash.is_empty() {
        None
    } else {
        Some(actual_hash)
    };
    Ok((data, hash))
}

// ---------------------------------------------------------------------------
// Directory listing
// ---------------------------------------------------------------------------

/// List the names of files and directories in `dirname`.
pub fn list_files<P: AsRef<Path>>(dirname: P) -> Result<Vec<String>, global::StringError> {
    let dirname = dirname.as_ref();
    let mut collected = Vec::new();
    for entry in fs::read_dir(dirname).map_err(|e| {
        global::StringError::new(format!(
            "Error listing files in {}: {}",
            dirname.display(),
            e
        ))
    })? {
        let entry = entry.map_err(|e| {
            global::StringError::new(format!(
                "Error listing files in {}: {}",
                dirname.display(),
                e
            ))
        })?;
        collected.push(entry.file_name().to_string_lossy().into_owned());
    }
    Ok(collected)
}

/// Recursively walk `dirname` and collect the full paths of files whose names
/// pass `file_filter`.
pub fn collect_files<P: AsRef<Path>>(
    dirname: P,
    file_filter: &dyn Fn(&str) -> bool,
) -> Result<Vec<String>, global::StringError> {
    let dirname = dirname.as_ref();
    let mut collected = Vec::new();
    for entry in walkdir::WalkDir::new(dirname) {
        let entry = entry.map_err(|e| {
            global::StringError::new(format!(
                "Error recursively collecting files in {}: {}",
                dirname.display(),
                e
            ))
        })?;
        if entry.file_type().is_file() {
            let file_name = entry.file_name().to_string_lossy();
            if file_filter(&file_name) {
                collected.push(entry.path().to_string_lossy().into_owned());
            }
        }
    }
    Ok(collected)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp_dir() -> PathBuf {
        let mut dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        dir.push(format!(
            "katago_fs_test_{}_{}_{}",
            std::process::id(),
            global::get_date_string(),
            nanos
        ));
        let _ = fs::create_dir_all(&dir);
        dir
    }

    fn cleanup(dir: &Path) {
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_make_dir_and_exists() {
        let dir = tmp_dir();
        cleanup(&dir);
        assert!(!exists(&dir));
        make_dir(&dir).unwrap();
        assert!(exists(&dir));
        assert!(is_directory(&dir));
        make_dir(&dir).unwrap(); // idempotent
        cleanup(&dir);
    }

    #[test]
    fn test_read_and_write() {
        let dir = tmp_dir();
        let file = dir.join("test.txt");
        cleanup(&dir);
        make_dir(&dir).unwrap();

        let mut f = fs::File::create(&file).unwrap();
        f.write_all(b"hello\nworld").unwrap();
        drop(f);

        assert_eq!(read_file(&file).unwrap(), "hello\nworld");
        assert_eq!(read_file_binary(&file).unwrap(), b"hello\nworld");

        let lines = read_file_lines(&file, b'\n').unwrap();
        assert_eq!(lines, vec!["hello".to_string(), "world".to_string()]);

        cleanup(&dir);
    }

    #[test]
    fn test_list_and_collect() {
        let dir = tmp_dir();
        cleanup(&dir);
        make_dir(&dir).unwrap();
        fs::File::create(dir.join("a.txt")).unwrap();
        fs::File::create(dir.join("b.txt")).unwrap();
        fs::File::create(dir.join("c.log")).unwrap();

        let mut names = list_files(&dir).unwrap();
        names.sort();
        assert_eq!(names, vec!["a.txt", "b.txt", "c.log"]);

        let mut collected = collect_files(&dir, &|name| name.ends_with(".txt")).unwrap();
        collected.sort();
        assert_eq!(collected.len(), 2);
        assert!(collected.iter().all(|p| p.ends_with(".txt")));

        cleanup(&dir);
    }

    #[test]
    fn test_rename_and_remove() {
        let dir = tmp_dir();
        cleanup(&dir);
        make_dir(&dir).unwrap();
        let src = dir.join("src.txt");
        let dst = dir.join("dst.txt");
        fs::write(&src, b"x").unwrap();

        assert!(try_rename(&src, &dst));
        assert!(!exists(&src));
        assert!(exists(&dst));

        assert!(try_remove_file(&dst));
        assert!(!exists(&dst));

        cleanup(&dir);
    }

    #[test]
    fn test_weakly_canonical() {
        let cwd = std::env::current_dir().unwrap();
        let rel = weakly_canonical("./foo/../bar");
        assert!(rel.is_absolute());
        assert!(rel.to_string_lossy().contains("bar"));

        let abs = weakly_canonical(cwd.join("."));
        assert!(abs.is_absolute());
    }

    #[test]
    fn test_load_with_sha256() {
        let dir = tmp_dir();
        cleanup(&dir);
        make_dir(&dir).unwrap();
        let file = dir.join("data.bin");
        fs::write(&file, b"hello world").unwrap();

        let expected = sha256_hex(b"hello world");
        let data = load_file_into_bytes(&file, &expected).unwrap();
        assert_eq!(data, b"hello world");

        let (data, hash) = load_file_into_bytes_with_hash(&file, "").unwrap();
        assert_eq!(data, b"hello world");
        assert_eq!(hash.unwrap(), expected);

        cleanup(&dir);
    }

    #[test]
    fn test_uncompress_and_load() {
        let dir = tmp_dir();
        cleanup(&dir);
        make_dir(&dir).unwrap();
        let file = dir.join("data.gz");

        let uncompressed = b"hello gzip world";
        let expected = sha256_hex(uncompressed);
        {
            let f = fs::File::create(&file).unwrap();
            let mut encoder = flate2::write::GzEncoder::new(f, flate2::Compression::default());
            encoder.write_all(uncompressed).unwrap();
            encoder.finish().unwrap();
        }

        let data = uncompress_and_load_file_into_bytes(&file, &expected).unwrap();
        assert_eq!(data, uncompressed);

        cleanup(&dir);
    }
}
