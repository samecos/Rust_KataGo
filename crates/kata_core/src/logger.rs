//! Thread-safe logging utility.
//!
//! Corresponds to `cpp/core/logger.h` and `cpp/core/logger.cpp`.
//!
//! The original C++ logger reads `ConfigParser` settings directly. To break the
//! dependency cycle during incremental porting, this Rust version exposes an
//! explicit `LoggerOptions` struct; configuration parsing will be layered on
//! top once `config_parser.rs` is ported.

use parking_lot::Mutex;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::global;

/// Options controlling logger behavior.
#[derive(Debug, Clone, Copy)]
pub struct LoggerOptions {
    pub log_to_stdout: bool,
    pub log_to_stderr: bool,
    pub log_time: bool,
}

impl Default for LoggerOptions {
    fn default() -> Self {
        Self {
            log_to_stdout: false,
            log_to_stderr: false,
            log_time: true,
        }
    }
}

/// Thread-safe multi-target logger.
pub struct Logger {
    inner: Arc<Mutex<LoggerInner>>,
}

impl Clone for Logger {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

struct LoggerInner {
    options: LoggerOptions,
    ostreams: Vec<Box<dyn Write + Send>>,
    files: Vec<BufWriter<File>>,
    header: String,
    is_disabled: bool,
}

impl Logger {
    /// Create a new logger with the given options and optional header text.
    pub fn new(options: LoggerOptions, header: Option<String>) -> Self {
        let header = header.unwrap_or_default();
        let inner = LoggerInner {
            options,
            ostreams: Vec::new(),
            files: Vec::new(),
            header: header.clone(),
            is_disabled: false,
        };
        let logger = Self {
            inner: Arc::new(Mutex::new(inner)),
        };
        if !header.is_empty() {
            logger.write(&header);
        }
        logger
    }

    /// Returns whether the logger writes to stdout.
    pub fn is_logging_to_stdout(&self) -> bool {
        self.inner.lock().options.log_to_stdout
    }

    /// Returns whether the logger writes to stderr.
    pub fn is_logging_to_stderr(&self) -> bool {
        self.inner.lock().options.log_to_stderr
    }

    /// Add an arbitrary output stream. The logger takes ownership.
    ///
    /// If `after_creation` is true and a header was configured, the header is
    /// written to the new stream.
    pub fn add_ostream<W: Write + Send + 'static>(&self, out: W, after_creation: bool) {
        let mut inner = self.inner.lock();
        let mut out: Box<dyn Write + Send> = Box::new(out);
        if after_creation && !inner.header.is_empty() {
            let now = now_seconds();
            write_locked(&inner.header, true, &mut out, now, inner.options.log_time);
        }
        inner.ostreams.push(out);
    }

    /// Add a file output. The file is opened in append mode.
    ///
    /// If the file cannot be opened, a warning is logged to other targets and
    /// stderr.
    pub fn add_file<P: AsRef<Path>>(&self, path: P, after_creation: bool) {
        let path = path.as_ref();
        if path.as_os_str().is_empty() {
            return;
        }
        let file = match OpenOptions::new().create(true).append(true).open(path) {
            Ok(f) => f,
            Err(e) => {
                let msg = format!("WARNING: could not open file for logging: {}", e);
                self.write(&msg);
                eprintln!("{}", msg);
                return;
            }
        };

        let mut inner = self.inner.lock();
        let mut writer = BufWriter::new(file);
        if after_creation && !inner.header.is_empty() {
            let now = now_seconds();
            write_locked(
                &inner.header,
                true,
                &mut writer,
                now,
                inner.options.log_time,
            );
        }
        inner.files.push(writer);
    }

    /// Enable or disable all output.
    pub fn set_disabled(&self, disabled: bool) {
        self.inner.lock().is_disabled = disabled;
    }

    /// Write a line to all enabled outputs.
    pub fn write(&self, s: &str) {
        self.write_inner(s, true);
    }

    /// Write to all enabled outputs without appending a newline.
    pub fn write_no_endline(&self, s: &str) {
        self.write_inner(s, false);
    }

    fn write_inner(&self, s: &str, end_line: bool) {
        let mut inner = self.inner.lock();
        if inner.is_disabled {
            return;
        }
        let now = now_seconds();
        let log_time = inner.options.log_time;
        let log_to_stdout = inner.options.log_to_stdout;
        let log_to_stderr = inner.options.log_to_stderr;

        if log_to_stdout {
            let mut stdout = io::stdout().lock();
            write_locked(s, end_line, &mut stdout, now, log_time);
        }
        if log_to_stderr {
            let mut stderr = io::stderr().lock();
            write_locked(s, end_line, &mut stderr, now, log_time);
        }
        for out in &mut inner.ostreams {
            write_locked(s, end_line, out, now, log_time);
        }
        for file in &mut inner.files {
            write_locked(s, end_line, file, now, log_time);
        }
    }

    /// Create a `Write` handle whose buffered content is flushed through this
    /// logger. The returned handle shares the logger lifetime via `Arc`.
    pub fn create_ostream(&self) -> LogStream {
        LogStream {
            logger: self.inner.clone(),
            buffer: Vec::new(),
        }
    }

    /// Run a closure and log any uncaught panic.
    pub fn log_thread_uncaught<F, T>(name: &str, logger: Option<&Logger>, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
            Ok(v) => v,
            Err(payload) => {
                let msg = panic_message(&payload);
                let full = format!("ERROR: {} loop thread failed: {}", name, msg);
                if let Some(logger) = logger {
                    logger.write(&full);
                } else {
                    eprintln!("{}", full);
                }
                std::thread::sleep(std::time::Duration::from_secs_f64(5.0));
                std::panic::resume_unwind(payload);
            }
        }
    }
}

fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// A `Write` adapter that flushes complete lines through a `Logger`.
pub struct LogStream {
    logger: Arc<Mutex<LoggerInner>>,
    buffer: Vec<u8>,
}

impl Write for LogStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        // Flush any complete lines.
        while let Some(pos) = self.buffer.iter().position(|&b| b == b'\n') {
            let line = String::from_utf8_lossy(&self.buffer[..pos]);
            self.write_no_endline(&line);
            self.buffer.drain(..=pos);
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if !self.buffer.is_empty() {
            let line = String::from_utf8_lossy(&self.buffer);
            self.write_no_endline(&line);
            self.buffer.clear();
        }
        Ok(())
    }
}

impl LogStream {
    fn write_no_endline(&self, s: &str) {
        let mut inner = self.logger.lock();
        if inner.is_disabled {
            return;
        }
        let now = now_seconds();
        let log_time = inner.options.log_time;
        let log_to_stdout = inner.options.log_to_stdout;
        let log_to_stderr = inner.options.log_to_stderr;

        if log_to_stdout {
            let mut stdout = io::stdout().lock();
            write_locked(s, false, &mut stdout, now, log_time);
        }
        if log_to_stderr {
            let mut stderr = io::stderr().lock();
            write_locked(s, false, &mut stderr, now, log_time);
        }
        for out in &mut inner.ostreams {
            write_locked(s, false, out, now, log_time);
        }
        for file in &mut inner.files {
            write_locked(s, false, file, now, log_time);
        }
    }
}

fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn format_time(seconds: u64) -> String {
    let dt = global::time_from_seconds(seconds as i64);
    format!(
        "[{:04}-{:02}-{:02} {:02}:{:02}:{:02}] ",
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
    )
}

fn write_locked<W: Write + ?Sized>(s: &str, end_line: bool, out: &mut W, now: u64, log_time: bool) {
    if log_time {
        let _ = out.write_all(format_time(now).as_bytes());
        let _ = out.write_all(s.as_bytes());
    } else {
        let _ = out.write_all(b": ");
        let _ = out.write_all(s.as_bytes());
    }
    if end_line {
        let _ = out.write_all(b"\n");
    }
    let _ = out.flush();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    /// A test writer that records bytes to a shared buffer.
    #[derive(Clone)]
    struct VecWriter(Arc<Mutex<Vec<u8>>>);

    impl VecWriter {
        fn new() -> Self {
            Self(Arc::new(Mutex::new(Vec::new())))
        }

        fn get(&self) -> String {
            let bytes = self.0.lock().unwrap().clone();
            String::from_utf8(bytes).unwrap()
        }
    }

    impl Write for VecWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_logger_writes_to_stream() {
        let writer = VecWriter::new();
        let logger = Logger::new(
            LoggerOptions {
                log_to_stdout: false,
                log_to_stderr: false,
                log_time: false,
            },
            None,
        );
        logger.add_ostream(writer.clone(), false);
        logger.write("hello");

        assert!(writer.get().contains(": hello\n"));
    }

    #[test]
    fn test_logger_disabled() {
        let writer = VecWriter::new();
        let logger = Logger::new(LoggerOptions::default(), None);
        logger.add_ostream(writer.clone(), false);
        logger.set_disabled(true);
        logger.write("hidden");

        assert!(writer.get().is_empty());
    }

    #[test]
    fn test_header_written_at_creation() {
        let writer = VecWriter::new();
        let logger = Logger::new(LoggerOptions::default(), Some("header".to_string()));
        logger.add_ostream(writer.clone(), true);

        assert!(writer.get().contains("header"));
    }

    #[test]
    fn test_log_stream() {
        let writer = VecWriter::new();
        let logger = Logger::new(
            LoggerOptions {
                log_to_stdout: false,
                log_to_stderr: false,
                log_time: false,
            },
            None,
        );
        logger.add_ostream(writer.clone(), false);
        let mut stream = logger.create_ostream();
        write!(stream, "line1\nline2").unwrap();
        stream.flush().unwrap();

        let output = writer.get();
        assert!(output.contains(": line1"));
        assert!(output.contains(": line2"));
    }
}
