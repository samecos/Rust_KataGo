//! Table mapping graph hashes to search nodes, used for transposition/graph search.
//!
//! Corresponds to `cpp/search/searchnodetable.h` and `cpp/search/searchnodetable.cpp`.

use std::collections::HashMap;
use std::marker::PhantomData;

use kata_core::hash::Hash128;
use parking_lot::Mutex;

use crate::mutex_pool::MutexPool;

/// A sharded table of non-owning pointers to search nodes keyed by graph hash.
///
/// The table does not own the nodes it stores; the caller is responsible for
/// keeping nodes alive at least as long as they remain in the table.
pub struct SearchNodeTable<T> {
    pub entries: Vec<Mutex<HashMap<Hash128, *mut T>>>,
    pub mutex_pool: MutexPool,
    pub num_shards: u32,
    _marker: PhantomData<T>,
}

impl<T> SearchNodeTable<T> {
    /// Create a table with `2^num_shards_power_of_two` shards.
    pub fn new(num_shards_power_of_two: u32) -> Self {
        assert!(
            num_shards_power_of_two < 32,
            "SearchNodeTable shift too large"
        );
        let num_shards = 1u32 << num_shards_power_of_two;
        let mutex_pool = MutexPool::new(num_shards);
        let entries = (0..num_shards)
            .map(|_| Mutex::new(HashMap::new()))
            .collect();
        Self {
            entries,
            mutex_pool,
            num_shards,
            _marker: PhantomData,
        }
    }

    /// Return the shard index for `hash`.
    ///
    /// Panics if the table was created with zero shards (which is impossible via `new`).
    pub fn get_index(&self, hash: u64) -> u32 {
        let mask = self.num_shards - 1;
        (hash & u64::from(mask)) as u32
    }

    /// Return the shard index for `graph_hash`.
    pub fn get_index_for_hash(&self, graph_hash: Hash128) -> u32 {
        self.get_index(graph_hash.hash0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyNode {
        _value: i32,
    }

    #[test]
    fn test_new_power_of_two_shards() {
        let table = SearchNodeTable::<DummyNode>::new(4);
        assert_eq!(table.num_shards, 16);
        assert_eq!(table.entries.len(), 16);
        assert_eq!(table.mutex_pool.num_mutexes(), 16);
    }

    #[test]
    fn test_get_index_within_range() {
        let table = SearchNodeTable::<DummyNode>::new(3);
        for h in 0..100u64 {
            let idx = table.get_index(h);
            assert!(idx < table.num_shards);
        }
    }

    #[test]
    fn test_get_index_uses_low_bits() {
        let table = SearchNodeTable::<DummyNode>::new(2);
        assert_eq!(table.get_index(0b0001), 1);
        assert_eq!(table.get_index(0b0101), 1);
        assert_eq!(table.get_index(0b0010), 2);
        assert_eq!(table.get_index(0b1111), 3);
    }

    #[test]
    fn test_store_and_retrieve_raw_pointer() {
        let table = SearchNodeTable::<DummyNode>::new(4);
        let mut node = DummyNode { _value: 42 };
        let node_ptr: *mut DummyNode = &mut node;
        let hash = Hash128::new(123, 0);
        let idx = table.get_index_for_hash(hash) as usize;

        {
            let _guard = table.mutex_pool.get_with_modulo(idx as u32).lock();
            table.entries[idx].lock().insert(hash, node_ptr);
        }

        {
            let _guard = table.mutex_pool.get_with_modulo(idx as u32).lock();
            let map = table.entries[idx].lock();
            let retrieved = map.get(&hash).copied().unwrap();
            assert_eq!(retrieved, node_ptr);
            unsafe {
                assert_eq!((*retrieved)._value, 42);
            }
        }
    }

    #[test]
    fn test_clear_removes_entries() {
        let table = SearchNodeTable::<DummyNode>::new(4);
        let mut node = DummyNode { _value: 7 };
        let hash = Hash128::new(99, 0);
        let idx = table.get_index_for_hash(hash) as usize;

        {
            let _guard = table.mutex_pool.get_with_modulo(idx as u32).lock();
            table.entries[idx].lock().insert(hash, &mut node);
        }
        {
            let _guard = table.mutex_pool.get_with_modulo(idx as u32).lock();
            table.entries[idx].lock().clear();
            assert!(table.entries[idx].lock().is_empty());
        }
    }
}
