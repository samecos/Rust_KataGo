//! Bounded, no-overwrite artifact I/O for the diagnostic collector.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    path::{Component, Path, PathBuf},
};

pub fn hash(raw: &[u8]) -> String {
    hex::encode(Sha256::digest(raw))
}
pub fn valid_sha(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}
impl Source {
    pub fn capture(path: &Path) -> Result<Self> {
        let path = path.canonicalize()?;
        let mut file = fs::File::open(&path)?;
        let bytes = file.metadata()?.len();
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 65536];
        let mut total = 0u64;
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            total += n as u64;
            hasher.update(&buffer[..n]);
        }
        ensure!(total == bytes, "source length changed during read");
        Ok(Self {
            path,
            bytes,
            sha256: hex::encode(hasher.finalize()),
        })
    }
    pub fn read(&self, cap: u64) -> Result<Vec<u8>> {
        ensure!(
            self.path.is_absolute() && valid_sha(&self.sha256) && self.bytes <= cap,
            "invalid/oversized source"
        );
        ensure!(
            fs::metadata(&self.path)?.len() == self.bytes,
            "source length changed"
        );
        let raw = fs::read(&self.path)?;
        ensure!(
            raw.len() as u64 == self.bytes && hash(&raw) == self.sha256,
            "source SHA changed"
        );
        Ok(raw)
    }
    pub fn recheck(&self) -> Result<()> {
        let actual = Self::capture(&self.path)?;
        ensure!(
            actual.bytes == self.bytes && actual.sha256 == self.sha256,
            "bound source changed: {}",
            self.path.display()
        );
        Ok(())
    }
}

pub fn write_new(root: &Path, name: &str, bytes: &[u8]) -> Result<Source> {
    let path = Path::new(name);
    ensure!(
        !name.is_empty() && path.components().all(|c| matches!(c, Component::Normal(_))),
        "invalid output path"
    );
    let full = root.join(path);
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&full)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    let source = Source::capture(&full)?;
    ensure!(
        source.bytes == bytes.len() as u64 && source.sha256 == hash(bytes),
        "output write changed"
    );
    Ok(source)
}
pub fn write_json(root: &Path, name: &str, value: &impl Serialize) -> Result<Source> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    write_new(root, name, &bytes)
}
pub fn commit_json(root: &Path, name: &str, value: &impl Serialize) -> Result<()> {
    // Pending data is flushed before a no-replace atomic hard-link publication.
    let pending = write_json(root, &format!("{name}.pending"), value)?;
    fs::hard_link(pending.path, root.join(name))?;
    Ok(())
}

pub struct Journal {
    file: fs::File,
    path: PathBuf,
    digest: Sha256,
    bytes: u64,
    pub consumed: usize,
    pub limit: usize,
}
impl Journal {
    pub fn new(path: &Path, limit: usize) -> Result<Self> {
        Ok(Self {
            file: fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(path)?,
            path: path.into(),
            digest: Sha256::new(),
            bytes: 0,
            consumed: 0,
            limit,
        })
    }
    /// A durable attempt precedes GPU submission. An uncertain/failed call is consumed.
    pub fn consume(&mut self, event: &serde_json::Value) -> Result<()> {
        ensure!(self.consumed < self.limit, "forward budget exhausted");
        let raw = serde_json::to_vec(&serde_json::json!({"ordinal":self.consumed,"event":event}))?;
        self.file.write_all(&raw)?;
        self.file.write_all(b"\n")?;
        self.file.sync_all()?;
        self.digest.update(&raw);
        self.digest.update(b"\n");
        self.bytes += raw.len() as u64 + 1;
        self.consumed += 1;
        Ok(())
    }
    pub fn seal(self) -> Result<Source> {
        self.file.sync_all()?;
        drop(self.file);
        let actual = Source::capture(&self.path)?;
        ensure!(
            actual.bytes == self.bytes && actual.sha256 == hex::encode(self.digest.finalize()),
            "durable forward journal changed"
        );
        Ok(actual)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_rejects_same_size_mutation_and_journal_exhaustion() {
        let dir = std::env::temp_dir().join(format!(
            "rustgo-calibration-artifacts-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&dir).unwrap();
        let source = write_new(&dir, "source", b"one").unwrap();
        assert!(write_new(&dir, "source", b"two").is_err());
        fs::write(&source.path, b"two").unwrap();
        assert!(source.read(3).is_err());
        assert!(source.recheck().is_err());
        let mut journal = Journal::new(&dir.join("journal"), 1).unwrap();
        journal
            .consume(&serde_json::json!({"phase":"numeric"}))
            .unwrap();
        assert!(
            journal
                .consume(&serde_json::json!({"phase":"numeric"}))
                .is_err()
        );
        assert_eq!(journal.consumed, 1);
        journal.seal().unwrap();
        let mut damaged = Journal::new(&dir.join("damaged"), 1).unwrap();
        damaged
            .consume(&serde_json::json!({"phase":"numeric"}))
            .unwrap();
        fs::OpenOptions::new()
            .append(true)
            .open(dir.join("damaged"))
            .unwrap()
            .write_all(b"damage")
            .unwrap();
        assert!(damaged.seal().is_err());
        // Delete only the explicitly created files and directory, never recursively.
        fs::remove_file(dir.join("source")).unwrap();
        fs::remove_file(dir.join("journal")).unwrap();
        fs::remove_file(dir.join("damaged")).unwrap();
        fs::remove_dir(dir).unwrap();
    }
}
