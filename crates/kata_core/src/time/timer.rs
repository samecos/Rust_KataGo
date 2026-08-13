//! Monotonic wall-clock timer.
//!
//! Corresponds to `cpp/core/timer.h` and `cpp/core/timer.cpp`.

use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// A simple timer measuring elapsed wall-clock time.
#[derive(Debug, Clone, Copy)]
pub struct ClockTimer {
    initial_time: Instant,
}

impl ClockTimer {
    /// Create and reset a new timer.
    pub fn new() -> Self {
        Self {
            initial_time: Instant::now(),
        }
    }

    /// Reset the timer to the current time.
    pub fn reset(&mut self) {
        self.initial_time = Instant::now();
    }

    /// Return the number of seconds elapsed since the timer was reset.
    pub fn get_seconds(&self) -> f64 {
        self.initial_time.elapsed().as_secs_f64()
    }

    /// Return a high-resolution integer timestamp suitable for seeds/hashes.
    ///
    /// The exact epoch is platform-dependent, matching the C++ implementation
    /// (which uses `GetTickCount64` on Windows and `steady_clock` on Unix).
    pub fn get_precision_system_time() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as i64
    }
}

impl Default for ClockTimer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer_elapses() {
        let mut timer = ClockTimer::new();
        assert!(timer.get_seconds() >= 0.0);

        std::thread::sleep(std::time::Duration::from_millis(50));
        let elapsed = timer.get_seconds();
        assert!(elapsed >= 0.04, "elapsed too small: {}", elapsed);

        timer.reset();
        assert!(timer.get_seconds() < elapsed);
    }

    #[test]
    fn test_precision_system_time_increases() {
        let t1 = ClockTimer::get_precision_system_time();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let t2 = ClockTimer::get_precision_system_time();
        assert!(
            t2 >= t1,
            "precision time did not increase: {} -> {}",
            t1,
            t2
        );
    }
}
