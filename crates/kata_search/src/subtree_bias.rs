//! Per-subtree empirical value-bias table.
//!
//! Corresponds to `cpp/search/subtreevaluebiastable.h` and
//! `cpp/search/subtreevaluebiastable.cpp`.

use kata_core::hash::Hash128;
use kata_core::rng::Rand;
use kata_game::board::{Board, Loc, MAX_ARR_SIZE, NULL_LOC, Player};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use crate::local_pattern::LocalPatternHasher;

struct ZobristTables {
    pattern_hasher: LocalPatternHasher,
    move_locs: [[Hash128; 2]; MAX_ARR_SIZE],
    ko_ban: [Hash128; MAX_ARR_SIZE],
}

static ZOBRIST: OnceLock<ZobristTables> = OnceLock::new();

fn zobrist_tables() -> &'static ZobristTables {
    ZOBRIST.get_or_init(|| {
        let mut rand = Rand::new();
        rand.init_from_seed("ValueBiasTable ZOBRIST STUFF");

        let mut pattern_hasher = LocalPatternHasher::new();
        pattern_hasher.init(5, 5, &mut rand);

        let mut move_locs = [[Hash128::default(); 2]; MAX_ARR_SIZE];
        for slot in &mut move_locs {
            for cell in slot.iter_mut() {
                *cell = Hash128::new(rand.next_u64(), rand.next_u64());
            }
        }

        rand.init_from_seed(
            "Reseed ValueBiasTable zobrist so that zobrists don't change when MAX_ARR_SIZE changes",
        );
        let mut ko_ban = [Hash128::default(); MAX_ARR_SIZE];
        for cell in &mut ko_ban {
            *cell = Hash128::new(rand.next_u64(), rand.next_u64());
        }

        ZobristTables {
            pattern_hasher,
            move_locs,
            ko_ban,
        }
    })
}

/// Accumulated statistics used to correct neural-net utility estimates.
#[derive(Debug, Default)]
pub struct SubtreeValueBiasEntry {
    pub delta_utility_sum: f64,
    pub weight_sum: f64,
}

/// Sharded table mapping board patterns to empirical bias entries.
pub struct SubtreeValueBiasTable {
    entries: Vec<Mutex<HashMap<Hash128, Arc<SubtreeValueBiasEntry>>>>,
}

impl SubtreeValueBiasTable {
    /// Create a table with `num_shards` shards.
    pub fn new(num_shards: i32) -> Self {
        assert!(
            num_shards > 0,
            "SubtreeValueBiasTable must have at least one shard"
        );
        let _ = zobrist_tables();
        let entries = (0..num_shards)
            .map(|_| Mutex::new(HashMap::new()))
            .collect();
        Self { entries }
    }

    /// Compute the hash key for a position.
    fn hash_key(
        pla: Player,
        parent_prev_move_loc: Loc,
        prev_move_loc: Loc,
        prev_board: &Board,
    ) -> Hash128 {
        let z = zobrist_tables();
        let mut hash =
            z.move_locs[parent_prev_move_loc as usize][0] ^ z.move_locs[prev_move_loc as usize][1];
        hash ^= z.pattern_hasher.get_hash(prev_board, prev_move_loc, pla);
        if prev_board.ko_loc != NULL_LOC {
            hash ^= z.ko_ban[prev_board.ko_loc as usize];
        }
        hash
    }

    /// Return the bias entry for the given position, creating it if necessary.
    pub fn get(
        &self,
        pla: Player,
        parent_prev_move_loc: Loc,
        prev_move_loc: Loc,
        prev_board: &Board,
    ) -> Arc<SubtreeValueBiasEntry> {
        let hash = Self::hash_key(pla, parent_prev_move_loc, prev_move_loc, prev_board);
        let sub_map_idx = (hash.hash0 % self.entries.len() as u64) as usize;
        let mut map = self.entries[sub_map_idx].lock();
        map.entry(hash)
            .or_insert_with(|| Arc::new(SubtreeValueBiasEntry::default()))
            .clone()
    }

    /// Remove entries that are no longer referenced outside the table.
    ///
    /// Must not be called concurrently with any other access to the table.
    pub fn clear_unused_synchronous(&self) {
        for map in &self.entries {
            let mut map = map.lock();
            map.retain(|_, entry| Arc::strong_count(entry) > 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kata_game::board::{Board, P_BLACK, P_WHITE, location};

    fn empty_board() -> Board {
        Board::new(9, 9)
    }

    #[test]
    fn test_get_returns_same_entry_for_same_key() {
        let table = SubtreeValueBiasTable::new(4);
        let board = empty_board();
        let loc = location::get_loc(4, 4, board.x_size);
        let parent = location::get_loc(3, 4, board.x_size);

        let e1 = table.get(P_BLACK, parent, loc, &board);
        let e2 = table.get(P_BLACK, parent, loc, &board);
        assert!(Arc::ptr_eq(&e1, &e2));
    }

    #[test]
    fn test_get_returns_different_entry_for_different_key() {
        let table = SubtreeValueBiasTable::new(4);
        let board = empty_board();
        let loc = location::get_loc(4, 4, board.x_size);

        let e1 = table.get(P_BLACK, NULL_LOC, loc, &board);
        let e2 = table.get(P_WHITE, NULL_LOC, loc, &board);
        assert!(!Arc::ptr_eq(&e1, &e2));
    }

    #[test]
    fn test_clear_unused_removes_unreferenced_entries() {
        let table = SubtreeValueBiasTable::new(2);
        let board = empty_board();
        let loc = location::get_loc(4, 4, board.x_size);

        {
            let _entry = table.get(P_BLACK, NULL_LOC, loc, &board);
        }
        assert_eq!(total_entries(&table), 1);

        table.clear_unused_synchronous();
        assert_eq!(total_entries(&table), 0);
    }

    #[test]
    fn test_clear_unused_keeps_referenced_entries() {
        let table = SubtreeValueBiasTable::new(2);
        let board = empty_board();
        let loc = location::get_loc(4, 4, board.x_size);

        let _entry = table.get(P_BLACK, NULL_LOC, loc, &board);
        table.clear_unused_synchronous();
        assert_eq!(total_entries(&table), 1);
    }

    #[test]
    fn test_global_init_is_stable_across_tables() {
        let board = empty_board();
        let loc = location::get_loc(4, 4, board.x_size);

        let t1 = SubtreeValueBiasTable::new(2);
        let e1 = t1.get(P_BLACK, NULL_LOC, loc, &board);

        let t2 = SubtreeValueBiasTable::new(4);
        let e2 = t2.get(P_BLACK, NULL_LOC, loc, &board);

        // Same global zobrist tables mean the same key, but the tables are independent,
        // so the entries are different objects.
        assert!(!Arc::ptr_eq(&e1, &e2));
    }

    fn total_entries(table: &SubtreeValueBiasTable) -> usize {
        table.entries.iter().map(|m| m.lock().len()).sum()
    }
}
