//! Cache of evaluated search-node statistics for reuse across graph-equivalent positions.
//!
//! Corresponds to `cpp/search/evalcache.h` and `cpp/search/evalcache.cpp`.
//!
//! To avoid depending on the not-yet-ported `SearchNode`, the cache operates on a
//! small trait (`EvalCacheNode`) that `SearchNode` will implement later.

use std::collections::BTreeMap;
use std::sync::Arc;

use kata_core::hash::Hash128;
use kata_game::board::{Loc, P_WHITE, PASS_LOC, Player};
use parking_lot::Mutex;

use crate::mutex_pool::MutexPool;

/// Per-child first-play evaluation cached from a previous search.
#[derive(Debug, Clone, Copy, Default)]
pub struct FirstExploreEval {
    pub avg_win_loss: f32,
    pub avg_score_mean: f32,
    pub cache_weight: f32,
}

impl FirstExploreEval {
    pub fn new(avg_win_loss: f32, avg_score_mean: f32, cache_weight: f32) -> Self {
        Self {
            avg_win_loss,
            avg_score_mean,
            cache_weight,
        }
    }
}

/// Trait implemented by child nodes exposed to the eval cache.
pub trait EvalCacheChildNode {
    /// Number of visits accumulated by this child.
    fn visits(&self) -> i64;
    /// Average win-loss value (from white's perspective).
    fn win_loss_value_avg(&self) -> f64;
    /// Average score mean (from white's perspective).
    fn score_mean_avg(&self) -> f64;
    /// Average utility (from white's perspective).
    fn utility_avg(&self) -> f64;
}

/// Trait implemented by search nodes that can be cached.
pub trait EvalCacheNode {
    /// Player to move at this node.
    fn next_pla(&self) -> Player;
    /// Number of visits accumulated by this node.
    fn visits(&self) -> i64;
    /// Average win-loss value (from white's perspective).
    fn win_loss_value_avg(&self) -> f64;
    /// Average no-result value.
    fn no_result_value_avg(&self) -> f64;
    /// Average score mean (from white's perspective).
    fn score_mean_avg(&self) -> f64;
    /// Average lead (from white's perspective).
    fn lead_avg(&self) -> f64;
    /// Average utility (from white's perspective).
    fn utility_avg(&self) -> f64;
    /// Call `f` for every allocated child, passing the child's move location,
    /// edge visits, and the child node itself.
    fn for_each_child(&self, f: &mut dyn FnMut(Loc, i64, &dyn EvalCacheChildNode));
}

/// Cached aggregate evaluation and first-explore evals for a graph hash.
#[derive(Debug, Default)]
pub struct EvalCacheEntry {
    pub avg_win_loss: f32,
    pub avg_no_result: f32,
    pub avg_score_mean: f32,
    pub avg_lead: f32,
    pub cache_weight: f32,
    pub first_explore_evals: BTreeMap<Loc, FirstExploreEval>,
}

/// Sharded cache mapping graph hashes to evaluated node statistics.
pub struct EvalCacheTable {
    entries: Vec<Mutex<BTreeMap<Hash128, Arc<EvalCacheEntry>>>>,
    mutex_pool: MutexPool,
}

impl EvalCacheTable {
    /// Create a cache with `num_shards` shards.
    pub fn new(num_shards: u32) -> Self {
        assert!(
            num_shards > 0,
            "EvalCacheTable must have at least one shard"
        );
        let mutex_pool = MutexPool::new(num_shards);
        let entries = (0..num_shards)
            .map(|_| Mutex::new(BTreeMap::new()))
            .collect();
        Self {
            entries,
            mutex_pool,
        }
    }

    fn shard_idx(&self, hash: Hash128) -> usize {
        (hash.hash0 % self.entries.len() as u64) as usize
    }

    /// Look up a cached entry by graph hash.
    pub fn find(&self, graph_hash: Hash128) -> Option<Arc<EvalCacheEntry>> {
        let idx = self.shard_idx(graph_hash);
        let _guard = self.mutex_pool.get_with_modulo_u64(graph_hash.hash0).lock();
        self.entries[idx].lock().get(&graph_hash).cloned()
    }

    /// Update the cache from `node`, replacing the existing entry only if the
    /// new visit count is large enough.
    pub fn update<N: EvalCacheNode>(
        &self,
        graph_hash: Hash128,
        node: &N,
        eval_cache_min_visits: i64,
        is_root_node: bool,
    ) {
        let idx = self.shard_idx(graph_hash);
        let mutex = self.mutex_pool.get_with_modulo_u64(graph_hash.hash0);

        let old_entry = {
            let _guard = mutex.lock();
            self.entries[idx].lock().get(&graph_hash).cloned()
        };

        let new_cache_weight = node.visits() as f32;
        if let Some(ref old) = old_entry {
            if new_cache_weight < old.cache_weight * 0.75 {
                return;
            }
        }

        let mut new_entry = if let Some(ref old) = old_entry {
            EvalCacheEntry {
                cache_weight: old.cache_weight,
                avg_win_loss: old.avg_win_loss,
                avg_no_result: old.avg_no_result,
                avg_score_mean: old.avg_score_mean,
                avg_lead: old.avg_lead,
                first_explore_evals: old.first_explore_evals.clone(),
            }
        } else {
            EvalCacheEntry::default()
        };

        node.for_each_child(&mut |move_loc, _edge_visits, child| {
            let child_num_visits = child.visits();
            if child_num_visits >= eval_cache_min_visits {
                let child_cache_weight = child_num_visits as f32;
                let eval = new_entry
                    .first_explore_evals
                    .entry(move_loc)
                    .or_insert_with(FirstExploreEval::default);
                if child_cache_weight >= eval.cache_weight {
                    *eval = FirstExploreEval::new(
                        child.win_loss_value_avg() as f32,
                        child.score_mean_avg() as f32,
                        child_cache_weight,
                    );
                }
            }
        });

        let mut should_record_evals = true;
        if is_root_node {
            let mut total_edge_visits: i64 = 0;
            let mut pass_edge_visits: i64 = 0;
            let mut max_self_utility: f64 = -1e50;
            let mut pass_self_utility: f64 = -1e50;

            node.for_each_child(&mut |move_loc, edge_visits, child| {
                let child_utility = child.utility_avg();
                let self_utility = if node.next_pla() == P_WHITE {
                    child_utility
                } else {
                    -child_utility
                };
                total_edge_visits += edge_visits;
                if self_utility > max_self_utility {
                    max_self_utility = self_utility;
                }
                if move_loc == PASS_LOC {
                    pass_edge_visits += edge_visits;
                    pass_self_utility = self_utility;
                }
            });

            if pass_edge_visits * 8 >= total_edge_visits
                || pass_self_utility + 0.05 >= max_self_utility
            {
                should_record_evals = false;
            }
        }

        if should_record_evals {
            new_entry.cache_weight = new_cache_weight;
            new_entry.avg_win_loss = node.win_loss_value_avg() as f32;
            new_entry.avg_no_result = node.no_result_value_avg() as f32;
            new_entry.avg_score_mean = node.score_mean_avg() as f32;
            new_entry.avg_lead = node.lead_avg() as f32;
        }

        let _guard = mutex.lock();
        let mut map = self.entries[idx].lock();
        map.insert(graph_hash, Arc::new(new_entry));
    }

    /// Clear all cached entries.
    pub fn clear(&self) {
        for (i, map) in self.entries.iter().enumerate() {
            let _guard = self.mutex_pool.get(i as u32).lock();
            map.lock().clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kata_game::board::P_BLACK;

    struct MockChild {
        visits: i64,
        win_loss: f64,
        score_mean: f64,
        utility: f64,
    }

    impl EvalCacheChildNode for MockChild {
        fn visits(&self) -> i64 {
            self.visits
        }
        fn win_loss_value_avg(&self) -> f64 {
            self.win_loss
        }
        fn score_mean_avg(&self) -> f64 {
            self.score_mean
        }
        fn utility_avg(&self) -> f64 {
            self.utility
        }
    }

    struct MockNode {
        next_pla: Player,
        visits: i64,
        win_loss: f64,
        no_result: f64,
        score_mean: f64,
        lead: f64,
        utility: f64,
        children: Vec<(Loc, i64, MockChild)>,
    }

    impl EvalCacheNode for MockNode {
        fn next_pla(&self) -> Player {
            self.next_pla
        }
        fn visits(&self) -> i64 {
            self.visits
        }
        fn win_loss_value_avg(&self) -> f64 {
            self.win_loss
        }
        fn no_result_value_avg(&self) -> f64 {
            self.no_result
        }
        fn score_mean_avg(&self) -> f64 {
            self.score_mean
        }
        fn lead_avg(&self) -> f64 {
            self.lead
        }
        fn utility_avg(&self) -> f64 {
            self.utility
        }
        fn for_each_child(&self, f: &mut dyn FnMut(Loc, i64, &dyn EvalCacheChildNode)) {
            for (loc, edge_visits, child) in &self.children {
                f(*loc, *edge_visits, child);
            }
        }
    }

    fn test_node_with_child(next_pla: Player, child_visits: i64) -> MockNode {
        MockNode {
            next_pla,
            visits: child_visits.max(10),
            win_loss: 0.3,
            no_result: 0.0,
            score_mean: 2.5,
            lead: 2.0,
            utility: 0.25,
            children: vec![(
                100,
                child_visits,
                MockChild {
                    visits: child_visits,
                    win_loss: 0.35,
                    score_mean: 3.0,
                    utility: 0.3,
                },
            )],
        }
    }

    #[test]
    fn test_find_missing_returns_none() {
        let table = EvalCacheTable::new(4);
        assert!(table.find(Hash128::new(1, 2)).is_none());
    }

    #[test]
    fn test_update_and_find() {
        let table = EvalCacheTable::new(4);
        let hash = Hash128::new(123, 0);
        let node = test_node_with_child(P_BLACK, 10);

        table.update(hash, &node, 5, false);
        let entry = table.find(hash).unwrap();
        assert!((entry.avg_win_loss - 0.3).abs() < 1e-6);
        assert!((entry.cache_weight - node.visits as f32).abs() < 1e-6);
        assert!(entry.first_explore_evals.contains_key(&100));
    }

    #[test]
    fn test_update_ignored_when_weight_too_low() {
        let table = EvalCacheTable::new(4);
        let hash = Hash128::new(123, 0);
        let node1 = test_node_with_child(P_BLACK, 100);
        table.update(hash, &node1, 5, false);

        let node2 = MockNode {
            visits: 10,
            ..test_node_with_child(P_BLACK, 10)
        };
        table.update(hash, &node2, 5, false);

        let entry = table.find(hash).unwrap();
        assert!((entry.cache_weight - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_update_replaces_when_weight_high_enough() {
        let table = EvalCacheTable::new(4);
        let hash = Hash128::new(123, 0);
        let node1 = test_node_with_child(P_BLACK, 100);
        table.update(hash, &node1, 5, false);

        let node2 = MockNode {
            visits: 200,
            win_loss: 0.5,
            ..test_node_with_child(P_BLACK, 200)
        };
        table.update(hash, &node2, 5, false);

        let entry = table.find(hash).unwrap();
        assert!((entry.avg_win_loss - 0.5).abs() < 1e-6);
        assert!((entry.cache_weight - 200.0).abs() < 1e-6);
    }

    #[test]
    fn test_child_with_too_few_visits_not_cached() {
        let table = EvalCacheTable::new(4);
        let hash = Hash128::new(123, 0);
        let node = test_node_with_child(P_BLACK, 3);
        table.update(hash, &node, 5, false);

        let entry = table.find(hash).unwrap();
        assert!(entry.first_explore_evals.is_empty());
    }

    #[test]
    fn test_root_skips_evals_when_pass_is_strong() {
        let table = EvalCacheTable::new(4);
        let hash = Hash128::new(123, 0);
        let node = MockNode {
            next_pla: P_BLACK,
            visits: 100,
            win_loss: 0.3,
            no_result: 0.0,
            score_mean: 2.5,
            lead: 2.0,
            utility: 0.25,
            children: vec![
                (
                    PASS_LOC,
                    50,
                    MockChild {
                        visits: 50,
                        win_loss: 0.3,
                        score_mean: 2.5,
                        utility: 0.25,
                    },
                ),
                (
                    100,
                    50,
                    MockChild {
                        visits: 50,
                        win_loss: 0.3,
                        score_mean: 2.5,
                        utility: 0.25,
                    },
                ),
            ],
        };

        table.update(hash, &node, 5, true);
        let entry = table.find(hash).unwrap();
        assert!((entry.cache_weight - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_clear_removes_entries() {
        let table = EvalCacheTable::new(4);
        let hash = Hash128::new(123, 0);
        table.update(hash, &test_node_with_child(P_BLACK, 10), 5, false);
        assert!(table.find(hash).is_some());

        table.clear();
        assert!(table.find(hash).is_none());
    }

    #[test]
    fn test_first_explore_eval_weight_update() {
        let table = EvalCacheTable::new(4);
        let hash = Hash128::new(123, 0);
        let node1 = test_node_with_child(P_BLACK, 10);
        table.update(hash, &node1, 5, false);

        let node2 = MockNode {
            visits: 100,
            children: vec![(
                100,
                100,
                MockChild {
                    visits: 100,
                    win_loss: 0.8,
                    score_mean: 5.0,
                    utility: 0.7,
                },
            )],
            ..test_node_with_child(P_BLACK, 100)
        };
        table.update(hash, &node2, 5, false);

        let entry = table.find(hash).unwrap();
        let eval = entry.first_explore_evals.get(&100).unwrap();
        assert!((eval.avg_win_loss - 0.8).abs() < 1e-6);
        assert!((eval.cache_weight - 100.0).abs() < 1e-6);
    }
}
