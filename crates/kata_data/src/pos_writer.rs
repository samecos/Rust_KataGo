//! Multi-threaded position sample writer.
//!
//! Corresponds to `cpp/dataio/poswriter.h` and `cpp/dataio/poswriter.cpp`.
//! Batches JSON-encoded `Sgf::PositionSample` lines into numbered output files.

use crate::sgf::PositionSample;
use kata_core::global::IOError;
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::thread::{self, JoinHandle};

use crossbeam::channel::{Receiver, Sender, bounded};

/// Writes position-sample JSON lines to numbered files in a background thread.
pub struct PosWriter {
    suffix: String,
    out_dir: PathBuf,
    sgf_split_count: i32,
    sgf_split_idx: i32,
    max_poses_per_out_file: i32,
    sender: Option<Sender<String>>,
    handle: Option<JoinHandle<()>>,
}

impl PosWriter {
    /// Create a new writer. It must be started with [`Self::start`] before use.
    pub fn new(
        suffix: &str,
        out_dir: impl AsRef<Path>,
        sgf_split_count: i32,
        sgf_split_idx: i32,
        max_poses_per_out_file: i32,
    ) -> Self {
        Self {
            suffix: suffix.to_string(),
            out_dir: out_dir.as_ref().to_path_buf(),
            sgf_split_count,
            sgf_split_idx,
            max_poses_per_out_file,
            sender: None,
            handle: None,
        }
    }

    /// Start the background writer thread.
    pub fn start(&mut self) -> Result<(), IOError> {
        if self.sender.is_some() {
            return Err(IOError("PosWriter::start - already started".to_string()));
        }

        create_dir_all(&self.out_dir).map_err(|e| {
            IOError(format!(
                "PosWriter::start - could not create output directory {}: {}",
                self.out_dir.display(),
                e
            ))
        })?;

        let (tx, rx) = bounded(1024);
        let suffix = self.suffix.clone();
        let out_dir = self.out_dir.clone();
        let sgf_split_count = self.sgf_split_count;
        let sgf_split_idx = self.sgf_split_idx;
        let max_poses_per_out_file = self.max_poses_per_out_file;

        let handle = thread::spawn(move || {
            write_loop(
                &suffix,
                &out_dir,
                sgf_split_count,
                sgf_split_idx,
                max_poses_per_out_file,
                rx,
            );
        });

        self.sender = Some(tx);
        self.handle = Some(handle);
        Ok(())
    }

    /// Stop accepting new lines, wait for the background thread to finish, and
    /// flush any remaining output.
    pub fn flush_and_stop(&mut self) -> Result<(), IOError> {
        // Dropping the sender signals the receiver to terminate.
        self.sender.take();
        if let Some(handle) = self.handle.take() {
            handle.join().map_err(|e| {
                IOError(format!(
                    "PosWriter::flush_and_stop - writer thread panicked: {:?}",
                    e
                ))
            })?;
        }
        Ok(())
    }

    /// Write a raw JSON line to the output files.
    pub fn write_line(&self, line: &str) -> Result<(), IOError> {
        let sender = self.sender.as_ref().ok_or_else(|| {
            IOError("PosWriter::write_line - writer has not been started".to_string())
        })?;
        sender
            .send(line.to_string())
            .map_err(|e| IOError(format!("PosWriter::write_line - send failed: {}", e)))
    }

    /// Serialize a position sample and write it as one JSON line.
    pub fn write_pos(&self, pos: &PositionSample) -> Result<(), IOError> {
        self.write_line(&PositionSample::to_json_line(pos))
    }
}

impl Drop for PosWriter {
    fn drop(&mut self) {
        let _ = self.flush_and_stop();
    }
}

fn write_loop(
    suffix: &str,
    out_dir: &Path,
    sgf_split_count: i32,
    sgf_split_idx: i32,
    max_poses_per_out_file: i32,
    rx: Receiver<String>,
) {
    let mut file_counter = 0;
    let mut num_written_this_file = 0;
    let mut out: Option<BufWriter<File>> = None;

    while let Ok(message) = rx.recv() {
        if out.is_none() || num_written_this_file > max_poses_per_out_file {
            if let Some(mut writer) = out.take() {
                let _ = writer.flush();
            }

            let file_name = if sgf_split_count > 1 {
                format!("{}.{}.{}", file_counter, sgf_split_idx, suffix)
            } else {
                format!("{}.{}", file_counter, suffix)
            };
            let path = out_dir.join(file_name);
            let file = File::create(&path).expect("PosWriter - failed to create output file");
            out = Some(BufWriter::new(file));
            file_counter += 1;
            num_written_this_file = 0;
        }

        if let Some(writer) = &mut out {
            writeln!(writer, "{}", message).expect("PosWriter - failed to write line");
        }
        num_written_this_file += 1;
    }

    if let Some(mut writer) = out.take() {
        let _ = writer.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sgf::{PositionSample, Sgf};
    use std::io::Read;

    #[test]
    fn test_pos_writer_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let mut writer = PosWriter::new("poses.json", tmp.path(), 1, 0, 2);
        writer.start().unwrap();

        let sgf = Sgf::parse("(;SZ[5]KM[7.5]RU[Chinese];B[cc];W[bb])").unwrap();
        let mut samples = Vec::new();
        sgf.iter_all_positions(
            false,
            false,
            None,
            &mut |sample, _hist, _comments| {
                samples.push(sample.clone());
            },
            true,
        )
        .unwrap();

        for sample in &samples {
            writer.write_pos(sample).unwrap();
        }
        writer.flush_and_stop().unwrap();

        let mut file = File::open(tmp.path().join("0.poses.json")).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), samples.len());
        for (line, sample) in lines.iter().zip(&samples) {
            let parsed = PositionSample::of_json_line(line).unwrap();
            assert!(sample.is_equal_for_testing(&parsed, false, false));
        }
    }
}
