//! Parallel range iteration.
//!
//! Corresponds to `cpp/core/parallel.h` and `cpp/core/parallel.cpp`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::logger::Logger;

/// Invoke `f(thread_idx, i)` for each `i` in `0..size`, using `num_threads`
/// worker threads and a shared atomic counter to hand out indices.
///
/// The C++ implementation spawns `num_threads` `std::thread`s and joins them
/// before returning. This Rust port preserves the same blocking behaviour.
pub fn iter_range<F>(num_threads: usize, size: usize, f: F)
where
    F: FnMut(usize, usize) + Send,
{
    let counter = AtomicUsize::new(0);
    let f = Arc::new(Mutex::new(f));
    thread::scope(|s| {
        for thread_idx in 0..num_threads {
            let counter = &counter;
            let f = Arc::clone(&f);
            s.spawn(move || {
                loop {
                    let old_value = counter.fetch_add(1, Ordering::Relaxed);
                    if old_value >= size {
                        break;
                    }
                    f.lock().unwrap()(thread_idx, old_value);
                }
            });
        }
    });
}

/// Like [`iter_range`], but wraps each worker in the logger's uncaught-exception
/// handler.
pub fn iter_range_with_logger<F>(num_threads: usize, size: usize, logger: &Logger, f: F)
where
    F: FnMut(usize, usize) + Send,
{
    let counter = AtomicUsize::new(0);
    let f = Arc::new(Mutex::new(f));
    thread::scope(|s| {
        for thread_idx in 0..num_threads {
            let counter = &counter;
            let f = Arc::clone(&f);
            s.spawn(move || {
                Logger::log_thread_uncaught("parallel iter range loop", Some(logger), || {
                    loop {
                        let old_value = counter.fetch_add(1, Ordering::Relaxed);
                        if old_value >= size {
                            break;
                        }
                        f.lock().unwrap()(thread_idx, old_value);
                    }
                });
            });
        }
    });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[test]
    fn test_iter_range_visits_all_indices() {
        let size = 1000;
        let visited = AtomicUsize::new(0);
        iter_range(4, size, |_thread_idx, i| {
            assert!(i < size);
            visited.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(visited.load(Ordering::Relaxed), size);
    }

    #[test]
    fn test_iter_range_empty() {
        let called = AtomicUsize::new(0);
        iter_range(4, 0, |_thread_idx, _i| {
            called.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(called.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_iter_range_single_thread() {
        let mut sum = 0;
        iter_range(1, 10, |_thread_idx, i| {
            sum += i;
        });
        assert_eq!(sum, 45);
    }
}
