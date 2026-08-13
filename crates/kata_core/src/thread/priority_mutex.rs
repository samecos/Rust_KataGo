//! Priority mutex favouring high-priority lockers.
//!
//! Corresponds to `cpp/core/prioritymutex.h`.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};

/// A mutex with two priority levels.
///
/// High-priority lockers take precedence: once a high-priority thread attempts
/// to lock, newly arriving low-priority threads block until all pending
/// high-priority threads have acquired and released the mutex.
pub struct PriorityMutex {
    mutex: Mutex<()>,
    low_priority_okay: Condvar,
    num_high_priority_threads: AtomicI32,
}

impl PriorityMutex {
    /// Create a new unlocked priority mutex.
    pub fn new() -> Self {
        Self {
            mutex: Mutex::new(()),
            low_priority_okay: Condvar::new(),
            num_high_priority_threads: AtomicI32::new(0),
        }
    }

    /// Acquire the mutex with high priority.
    pub fn lock_high_priority(&self) -> PriorityMutexGuard<'_> {
        self.num_high_priority_threads
            .fetch_add(1, Ordering::Relaxed);
        let guard = self.mutex.lock().unwrap();
        PriorityMutexGuard {
            mutex: self,
            is_high_priority: true,
            _guard: Some(guard),
        }
    }

    /// Acquire the mutex with low priority, waiting until no high-priority
    /// threads are pending.
    pub fn lock_low_priority(&self) -> PriorityMutexGuard<'_> {
        let guard = self.mutex.lock().unwrap();
        let mut guard = guard;
        while self.num_high_priority_threads.load(Ordering::Relaxed) > 0 {
            guard = self.low_priority_okay.wait(guard).unwrap();
        }
        // `MutexGuard` is kept inside `PriorityMutexGuard`; unlocking is done
        // via `PriorityMutexGuard::unlock` or on drop.
        PriorityMutexGuard {
            mutex: self,
            is_high_priority: false,
            _guard: Some(guard),
        }
    }

    fn unlock_high_priority(&self) {
        let old_value = self
            .num_high_priority_threads
            .fetch_sub(1, Ordering::Relaxed);
        let new_value = old_value - 1;
        debug_assert!(new_value >= 0);
        if new_value <= 0 {
            self.low_priority_okay.notify_all();
        }
    }
}

impl Default for PriorityMutex {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard for [`PriorityMutex`].
pub struct PriorityMutexGuard<'a> {
    mutex: &'a PriorityMutex,
    is_high_priority: bool,
    _guard: Option<MutexGuard<'a, ()>>,
}

impl PriorityMutexGuard<'_> {
    /// Explicitly unlock before the guard goes out of scope.
    pub fn unlock(mut self) {
        // Drop the inner guard first, then update priority state.
        self._guard.take();
        if self.is_high_priority {
            self.mutex.unlock_high_priority();
        }
        // Prevent the normal Drop impl from running again.
        std::mem::forget(self);
    }
}

impl Drop for PriorityMutexGuard<'_> {
    fn drop(&mut self) {
        if self._guard.take().is_some() && self.is_high_priority {
            self.mutex.unlock_high_priority();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use super::*;

    #[test]
    fn test_high_priority_acquires_after_low_priority_releases() {
        let mutex = Arc::new(PriorityMutex::new());
        let acquired = Arc::new(std::sync::atomic::AtomicBool::new(false));

        // Main thread holds the mutex with low priority.
        let low_guard = mutex.lock_low_priority();

        let mutex2 = Arc::clone(&mutex);
        let acquired2 = Arc::clone(&acquired);
        let high = thread::spawn(move || {
            let _guard = mutex2.lock_high_priority();
            acquired2.store(true, Ordering::Relaxed);
        });

        // Give the high-priority thread time to start waiting on the mutex.
        thread::sleep(Duration::from_millis(30));
        assert!(!acquired.load(Ordering::Relaxed));

        // Release low priority; the waiting high-priority thread should acquire.
        drop(low_guard);
        high.join().unwrap();
        assert!(acquired.load(Ordering::Relaxed));
    }

    #[test]
    fn test_multiple_high_priority() {
        let mutex = Arc::new(PriorityMutex::new());
        let counter = Arc::new(std::sync::atomic::AtomicI32::new(0));

        let mut handles = Vec::new();
        for _ in 0..4 {
            let mutex = Arc::clone(&mutex);
            let counter = Arc::clone(&counter);
            handles.push(thread::spawn(move || {
                let _guard = mutex.lock_high_priority();
                counter.fetch_add(1, Ordering::Relaxed);
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(counter.load(Ordering::Relaxed), 4);
    }
}
