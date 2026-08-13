//! A pool of mutexes used to shard lock contention.
//!
//! Corresponds to `cpp/search/mutexpool.h` and `cpp/search/mutexpool.cpp`.

use kata_core::hash::Hash128;
use parking_lot::Mutex;

/// A fixed-size pool of independent mutexes.
///
/// Callers pick a mutex either by index or by hashing a key, reducing contention
/// when many threads need to synchronize on different logical resources.
pub struct MutexPool {
    mutexes: Vec<Mutex<()>>,
    num_mutexes: u32,
}

impl MutexPool {
    /// Create a pool with `n` mutexes.
    pub fn new(n: u32) -> Self {
        let mut mutexes = Vec::with_capacity(n as usize);
        for _ in 0..n {
            mutexes.push(Mutex::new(()));
        }
        Self {
            mutexes,
            num_mutexes: n,
        }
    }

    /// Number of mutexes in the pool.
    pub fn num_mutexes(&self) -> u32 {
        self.num_mutexes
    }

    /// Direct access to the mutex at `idx`.
    ///
    /// # Panics
    /// Panics if `idx >= num_mutexes`.
    pub fn get(&self, idx: u32) -> &Mutex<()> {
        &self.mutexes[idx as usize]
    }

    /// Return the mutex selected by `idx mod num_mutexes`.
    pub fn get_with_modulo(&self, idx: u32) -> &Mutex<()> {
        &self.mutexes[(idx % self.num_mutexes) as usize]
    }

    /// Return the mutex selected by `idx mod num_mutexes`.
    pub fn get_with_modulo_u64(&self, idx: u64) -> &Mutex<()> {
        &self.mutexes[(idx % self.num_mutexes as u64) as usize]
    }

    /// Return the mutex selected by `hash.hash0 mod num_mutexes`.
    pub fn get_with_modulo_hash(&self, hash: Hash128) -> &Mutex<()> {
        &self.mutexes[(hash.hash0 % self.num_mutexes as u64) as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_num_mutexes() {
        let pool = MutexPool::new(4);
        assert_eq!(pool.num_mutexes(), 4);
    }

    #[test]
    fn test_get_and_lock() {
        let pool = MutexPool::new(4);
        let _guard = pool.get(2).lock();
    }

    #[test]
    fn test_modulo_indexing() {
        let pool = MutexPool::new(4);
        assert_eq!(
            pool.get_with_modulo(2) as *const Mutex<()>,
            pool.get_with_modulo(6) as *const Mutex<()>
        );
        assert_eq!(
            pool.get_with_modulo_u64(10) as *const Mutex<()>,
            pool.get(2) as *const Mutex<()>
        );
    }

    #[test]
    fn test_modulo_hash_indexing() {
        let pool = MutexPool::new(8);
        let hash = Hash128::new(42, 0);
        let idx = (hash.hash0 % 8) as u32;
        assert_eq!(
            pool.get_with_modulo_hash(hash) as *const Mutex<()>,
            pool.get(idx) as *const Mutex<()>
        );
    }
}
