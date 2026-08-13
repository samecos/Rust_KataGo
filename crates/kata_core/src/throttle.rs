use std::fmt;
use std::sync::{Condvar, Mutex};

/// A semaphore-like throttle that limits the number of concurrently active
/// threads.
///
/// Corresponds to `cpp/core/throttle.h`.
pub struct Throttle {
    max_threads_active: usize,
    state: Mutex<ThrottleState>,
    okay_for_more: Condvar,
}

struct ThrottleState {
    num_threads_active: usize,
}

impl Throttle {
    /// Create a throttle allowing at most `max_threads_at_a_time` active
    /// threads. Panics if `max_threads_at_a_time` is zero.
    pub fn new(max_threads_at_a_time: usize) -> Self {
        assert!(
            max_threads_at_a_time > 0,
            "Throttle: maxThreadsAtATime must be > 0"
        );
        Self {
            max_threads_active: max_threads_at_a_time,
            state: Mutex::new(ThrottleState {
                num_threads_active: 0,
            }),
            okay_for_more: Condvar::new(),
        }
    }

    /// Acquire a slot, blocking until one is available.
    pub fn lock(&self) -> ThrottleGuard<'_> {
        let mut state = self.state.lock().unwrap();
        while state.num_threads_active >= self.max_threads_active {
            state = self.okay_for_more.wait(state).unwrap();
        }
        state.num_threads_active += 1;
        debug_assert!(state.num_threads_active <= self.max_threads_active);
        ThrottleGuard { throttle: self }
    }

    fn unlock(&self) {
        let mut state = self.state.lock().unwrap();
        debug_assert!(state.num_threads_active > 0);
        state.num_threads_active -= 1;
        self.okay_for_more.notify_one();
    }
}

impl fmt::Debug for Throttle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.state.lock().unwrap();
        f.debug_struct("Throttle")
            .field("num_threads_active", &state.num_threads_active)
            .field("max_threads_active", &self.max_threads_active)
            .finish()
    }
}

/// RAII guard that releases the throttle slot on drop.
pub struct ThrottleGuard<'a> {
    throttle: &'a Throttle,
}

impl Drop for ThrottleGuard<'_> {
    fn drop(&mut self) {
        self.throttle.unlock();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    use super::*;

    #[test]
    fn test_throttle_limits_concurrency() {
        let throttle = Arc::new(Throttle::new(2));
        let active = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let max_observed = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..6 {
            let throttle = Arc::clone(&throttle);
            let active = Arc::clone(&active);
            let max_observed = Arc::clone(&max_observed);
            handles.push(thread::spawn(move || {
                let _guard = throttle.lock();
                let n = active.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                loop {
                    let current_max = max_observed.load(std::sync::atomic::Ordering::Relaxed);
                    if n > current_max {
                        if max_observed
                            .compare_exchange(
                                current_max,
                                n,
                                std::sync::atomic::Ordering::Relaxed,
                                std::sync::atomic::Ordering::Relaxed,
                            )
                            .is_ok()
                        {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                thread::sleep(Duration::from_millis(30));
                active.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(max_observed.load(std::sync::atomic::Ordering::Relaxed), 2);
    }

    #[test]
    fn test_throttle_unblocks_waiters() {
        let throttle = Arc::new(Throttle::new(1));
        let t = Arc::clone(&throttle);
        let start = Instant::now();

        let h = thread::spawn(move || {
            let _guard = t.lock();
            thread::sleep(Duration::from_millis(50));
        });

        thread::sleep(Duration::from_millis(10));
        let _guard = throttle.lock();
        h.join().unwrap();
        assert!(start.elapsed() >= Duration::from_millis(50));
    }
}
