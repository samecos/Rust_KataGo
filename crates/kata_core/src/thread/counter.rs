//! Thread-safe counter and waitable flag.
//!
//! Corresponds to `cpp/core/threadsafecounter.h` and
//! `cpp/core/threadsafecounter.cpp`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};

/// A thread-safe counter that can be incremented and waited on until zero.
pub struct ThreadSafeCounter {
    value: Mutex<i64>,
    zero_condvar: Condvar,
}

impl ThreadSafeCounter {
    /// Create a new counter initialized to zero.
    pub fn new() -> Self {
        Self {
            value: Mutex::new(0),
            zero_condvar: Condvar::new(),
        }
    }

    /// Add `x` to the counter. If the result is zero, wake all waiters.
    pub fn add(&self, x: i64) {
        let mut value = self.value.lock().unwrap();
        *value += x;
        if *value == 0 {
            self.zero_condvar.notify_all();
        }
    }

    /// Set the counter to zero and wake all waiters.
    pub fn set_zero(&self) {
        let mut value = self.value.lock().unwrap();
        *value = 0;
        self.zero_condvar.notify_all();
    }

    /// Block until the counter reaches zero.
    pub fn wait_until_zero(&self) {
        let mut value = self.value.lock().unwrap();
        while *value != 0 {
            value = self.zero_condvar.wait(value).unwrap();
        }
    }
}

impl Default for ThreadSafeCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// A boolean flag that supports blocking until it becomes true or false.
pub struct WaitableFlag {
    value: AtomicBool,
    finished: Mutex<bool>,
    false_condvar: Condvar,
    true_condvar: Condvar,
}

impl WaitableFlag {
    /// Create a new flag initially set to `false` and not finished.
    pub fn new() -> Self {
        Self {
            value: AtomicBool::new(false),
            finished: Mutex::new(false),
            false_condvar: Condvar::new(),
            true_condvar: Condvar::new(),
        }
    }

    /// Set the flag to `b` and wake the appropriate waiters.
    ///
    /// Has no effect if the flag has been permanently set.
    pub fn set(&self, b: bool) {
        let finished = self.finished.lock().unwrap();
        if *finished {
            return;
        }
        self.value.store(b, Ordering::Release);
        if b {
            self.true_condvar.notify_all();
        } else {
            self.false_condvar.notify_all();
        }
    }

    /// Permanently set the flag to `b` and wake all waiters.
    ///
    /// Subsequent calls to [`set`] and [`set_permanently`] are ignored.
    pub fn set_permanently(&self, b: bool) {
        let mut finished = self.finished.lock().unwrap();
        if *finished {
            return;
        }
        *finished = true;
        self.value.store(b, Ordering::Release);
        if b {
            self.true_condvar.notify_all();
        } else {
            self.false_condvar.notify_all();
        }
    }

    /// Return the current value.
    pub fn get(&self) -> bool {
        self.value.load(Ordering::Acquire)
    }

    /// Block until the flag becomes `false`.
    pub fn wait_until_false(&self) {
        let mut b = self.get();
        if !b {
            return;
        }
        let lock = self.finished.lock().unwrap();
        let mut guard = lock;
        while b {
            guard = self.false_condvar.wait(guard).unwrap();
            b = self.get();
        }
    }

    /// Block until the flag becomes `true`.
    pub fn wait_until_true(&self) {
        let mut b = self.get();
        if b {
            return;
        }
        let lock = self.finished.lock().unwrap();
        let mut guard = lock;
        while !b {
            guard = self.true_condvar.wait(guard).unwrap();
            b = self.get();
        }
    }
}

impl Default for WaitableFlag {
    fn default() -> Self {
        Self::new()
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
    fn test_counter_wait_until_zero() {
        let counter = Arc::new(ThreadSafeCounter::new());
        let counter2 = Arc::clone(&counter);

        counter.add(2);

        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            counter2.add(-1);
            thread::sleep(Duration::from_millis(20));
            counter2.add(-1);
        });

        counter.wait_until_zero();
        handle.join().unwrap();
        assert_eq!(*counter.value.lock().unwrap(), 0);
    }

    #[test]
    fn test_counter_set_zero() {
        let counter = ThreadSafeCounter::new();
        counter.add(5);
        counter.set_zero();
        counter.wait_until_zero(); // should return immediately
    }

    #[test]
    fn test_waitable_flag_wait_until_true() {
        let flag = Arc::new(WaitableFlag::new());
        let flag2 = Arc::clone(&flag);

        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            flag2.set(true);
        });

        flag.wait_until_true();
        handle.join().unwrap();
        assert!(flag.get());
    }

    #[test]
    fn test_waitable_flag_wait_until_false() {
        let flag = Arc::new(WaitableFlag::new());
        flag.set(true);
        let flag2 = Arc::clone(&flag);

        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            flag2.set(false);
        });

        flag.wait_until_false();
        handle.join().unwrap();
        assert!(!flag.get());
    }

    #[test]
    fn test_waitable_flag_permanent() {
        let flag = Arc::new(WaitableFlag::new());
        flag.set_permanently(true);
        assert!(flag.get());
        flag.set(false); // ignored
        assert!(flag.get());
        flag.set_permanently(false); // ignored
        assert!(flag.get());
    }
}
