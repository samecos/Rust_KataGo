//! MCTS search node and per-node statistics.
//!
//! Corresponds to `cpp/search/searchnode.h` and `cpp/search/searchnode.cpp`.
//! Methods that require the not-yet-ported `SearchThread` are omitted for now
//! (notably `store_nn_output` / `store_human_output` with thread cleanup).

use std::sync::Arc;
use std::sync::atomic::{
    AtomicBool, AtomicI16, AtomicI32, AtomicI64, AtomicPtr, AtomicU32, AtomicU64, Ordering,
};

use kata_core::hash::Hash128;
use kata_game::board::{Loc, NULL_LOC, Player};
use kata_nn::inputs::NNOutput;

use crate::eval_cache::{EvalCacheChildNode, EvalCacheEntry, EvalCacheNode};
use crate::subtree_bias::SubtreeValueBiasEntry;

/// Stable `f64` atomic implemented by storing the IEEE-754 bit pattern.
#[derive(Debug)]
pub struct AtomicF64 {
    bits: AtomicU64,
}

impl AtomicF64 {
    pub fn new(value: f64) -> Self {
        Self {
            bits: AtomicU64::new(value.to_bits()),
        }
    }

    pub fn load(&self, order: Ordering) -> f64 {
        f64::from_bits(self.bits.load(order))
    }

    pub fn store(&self, value: f64, order: Ordering) {
        self.bits.store(value.to_bits(), order);
    }
}

pub type SearchNodeState = i32;

pub const STATE_UNEVALUATED: SearchNodeState = 0;
pub const STATE_EVALUATING: SearchNodeState = 1;
pub const STATE_EXPANDED0: SearchNodeState = 2;
pub const STATE_GROWING1: SearchNodeState = 3;
pub const STATE_EXPANDED1: SearchNodeState = 4;
pub const STATE_GROWING2: SearchNodeState = 5;
pub const STATE_EXPANDED2: SearchNodeState = 6;

pub mod children_sizes {
    pub const SIZE0_TOTAL: usize = 8;
    pub const SIZE1_TOTAL: usize = 64;
    pub const SIZE2_TOTAL: usize = kata_nn::inputs::nn_pos::MAX_NN_POLICY_SIZE;
    pub const SIZE0_OVERFLOW: usize = SIZE0_TOTAL;
    pub const SIZE1_OVERFLOW: usize = SIZE1_TOTAL - SIZE0_TOTAL;
    pub const SIZE2_OVERFLOW: usize = SIZE2_TOTAL - SIZE1_TOTAL;
}

/// Thread-safe mutable statistics stored on a search node.
#[derive(Debug)]
pub struct NodeStatsAtomic {
    pub visits: AtomicI64,
    pub win_loss_value_avg: AtomicF64,
    pub no_result_value_avg: AtomicF64,
    pub score_mean_avg: AtomicF64,
    pub score_mean_sq_avg: AtomicF64,
    pub lead_avg: AtomicF64,
    pub utility_avg: AtomicF64,
    pub utility_sq_avg: AtomicF64,
    pub weight_sum: AtomicF64,
    pub weight_sq_sum: AtomicF64,
}

impl Default for NodeStatsAtomic {
    fn default() -> Self {
        Self {
            visits: AtomicI64::new(0),
            win_loss_value_avg: AtomicF64::new(0.0),
            no_result_value_avg: AtomicF64::new(0.0),
            score_mean_avg: AtomicF64::new(0.0),
            score_mean_sq_avg: AtomicF64::new(0.0),
            lead_avg: AtomicF64::new(0.0),
            utility_avg: AtomicF64::new(0.0),
            utility_sq_avg: AtomicF64::new(0.0),
            weight_sum: AtomicF64::new(0.0),
            weight_sq_sum: AtomicF64::new(0.0),
        }
    }
}

impl Clone for NodeStatsAtomic {
    fn clone(&self) -> Self {
        Self {
            visits: AtomicI64::new(self.visits.load(Ordering::Acquire)),
            win_loss_value_avg: AtomicF64::new(self.win_loss_value_avg.load(Ordering::Acquire)),
            no_result_value_avg: AtomicF64::new(self.no_result_value_avg.load(Ordering::Acquire)),
            score_mean_avg: AtomicF64::new(self.score_mean_avg.load(Ordering::Acquire)),
            score_mean_sq_avg: AtomicF64::new(self.score_mean_sq_avg.load(Ordering::Acquire)),
            lead_avg: AtomicF64::new(self.lead_avg.load(Ordering::Acquire)),
            utility_avg: AtomicF64::new(self.utility_avg.load(Ordering::Acquire)),
            utility_sq_avg: AtomicF64::new(self.utility_sq_avg.load(Ordering::Acquire)),
            weight_sum: AtomicF64::new(self.weight_sum.load(Ordering::Acquire)),
            weight_sq_sum: AtomicF64::new(self.weight_sq_sum.load(Ordering::Acquire)),
        }
    }
}

impl NodeStatsAtomic {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_child_weight(&self, edge_visits: i64) -> f64 {
        NodeStats::child_weight(
            edge_visits,
            self.visits.load(Ordering::Acquire),
            self.weight_sum.load(Ordering::Acquire),
        )
    }

    pub fn get_child_weight_with_child_visits(&self, edge_visits: i64, child_visits: i64) -> f64 {
        NodeStats::child_weight(
            edge_visits,
            child_visits,
            self.weight_sum.load(Ordering::Acquire),
        )
    }

    pub fn get_child_weight_sq(&self, edge_visits: i64) -> f64 {
        NodeStats::child_weight_sq(
            edge_visits,
            self.visits.load(Ordering::Acquire),
            self.weight_sq_sum.load(Ordering::Acquire),
        )
    }

    pub fn get_child_weight_sq_with_child_visits(
        &self,
        edge_visits: i64,
        child_visits: i64,
    ) -> f64 {
        NodeStats::child_weight_sq(
            edge_visits,
            child_visits,
            self.weight_sq_sum.load(Ordering::Acquire),
        )
    }
}

/// Snapshot of node statistics, suitable for copying out of the search tree.
#[derive(Debug, Clone, Default)]
pub struct NodeStats {
    pub visits: i64,
    pub win_loss_value_avg: f64,
    pub no_result_value_avg: f64,
    pub score_mean_avg: f64,
    pub score_mean_sq_avg: f64,
    pub lead_avg: f64,
    pub utility_avg: f64,
    pub utility_sq_avg: f64,
    pub weight_sum: f64,
    pub weight_sq_sum: f64,
}

impl NodeStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_atomic(other: &NodeStatsAtomic) -> Self {
        Self {
            visits: other.visits.load(Ordering::Acquire),
            win_loss_value_avg: other.win_loss_value_avg.load(Ordering::Acquire),
            no_result_value_avg: other.no_result_value_avg.load(Ordering::Acquire),
            score_mean_avg: other.score_mean_avg.load(Ordering::Acquire),
            score_mean_sq_avg: other.score_mean_sq_avg.load(Ordering::Acquire),
            lead_avg: other.lead_avg.load(Ordering::Acquire),
            utility_avg: other.utility_avg.load(Ordering::Acquire),
            utility_sq_avg: other.utility_sq_avg.load(Ordering::Acquire),
            weight_sum: other.weight_sum.load(Ordering::Acquire),
            weight_sq_sum: other.weight_sq_sum.load(Ordering::Acquire),
        }
    }

    pub fn child_weight(edge_visits: i64, child_visits: i64, raw_child_weight: f64) -> f64 {
        raw_child_weight * (edge_visits as f64 / child_visits.max(1) as f64)
    }

    pub fn child_weight_sq(edge_visits: i64, child_visits: i64, raw_child_weight_sq: f64) -> f64 {
        raw_child_weight_sq * (edge_visits as f64 / child_visits.max(1) as f64)
    }

    pub fn get_child_weight(&self, edge_visits: i64) -> f64 {
        Self::child_weight(edge_visits, self.visits, self.weight_sum)
    }
}

/// Additional per-node values computed during search-result extraction.
#[derive(Debug, Clone, Default)]
pub struct MoreNodeStats {
    pub stats: NodeStats,
    pub self_utility: f64,
    pub weight_adjusted: f64,
    pub prev_move_loc: Loc,
}

impl MoreNodeStats {
    pub fn new() -> Self {
        Self::default()
    }
}

/// A child edge: atomic pointer to the child node plus edge metadata.
#[derive(Debug)]
pub struct SearchChildPointer {
    data: AtomicPtr<SearchNode>,
    edge_visits: AtomicI64,
    move_loc: AtomicI16,
}

impl Default for SearchChildPointer {
    fn default() -> Self {
        Self {
            data: AtomicPtr::new(std::ptr::null_mut()),
            edge_visits: AtomicI64::new(0),
            move_loc: AtomicI16::new(NULL_LOC),
        }
    }
}

impl SearchChildPointer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn store_all(&self, other: &SearchChildPointer) {
        let d = other.data.load(Ordering::Acquire);
        let e = other.edge_visits.load(Ordering::Acquire);
        let m = other.move_loc.load(Ordering::Acquire);
        self.move_loc.store(m, Ordering::Release);
        self.edge_visits.store(e, Ordering::Release);
        self.data.store(d, Ordering::Release);
    }

    pub fn get_if_allocated(&self) -> Option<&SearchNode> {
        let ptr = self.data.load(Ordering::Acquire);
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { &*ptr })
        }
    }

    /// Raw pointer to the allocated child node, or null.
    pub fn get_raw_ptr(&self) -> *mut SearchNode {
        self.data.load(Ordering::Acquire)
    }

    pub fn get_if_allocated_relaxed(&self) -> Option<&SearchNode> {
        let ptr = self.data.load(Ordering::Relaxed);
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { &*ptr })
        }
    }

    pub fn is_allocated_relaxed(&self) -> bool {
        !self.data.load(Ordering::Relaxed).is_null()
    }

    pub fn store(&self, node: *mut SearchNode) {
        self.data.store(node, Ordering::Release);
    }

    pub fn store_relaxed(&self, node: *mut SearchNode) {
        self.data.store(node, Ordering::Relaxed);
    }

    pub fn store_if_null(&self, node: *mut SearchNode) -> bool {
        self.data
            .compare_exchange(
                std::ptr::null_mut(),
                node,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    pub fn get_edge_visits(&self) -> i64 {
        self.edge_visits.load(Ordering::Acquire)
    }

    pub fn get_edge_visits_relaxed(&self) -> i64 {
        self.edge_visits.load(Ordering::Relaxed)
    }

    pub fn set_edge_visits(&self, x: i64) {
        self.edge_visits.store(x, Ordering::Release);
    }

    pub fn set_edge_visits_relaxed(&self, x: i64) {
        self.edge_visits.store(x, Ordering::Relaxed);
    }

    pub fn add_edge_visits(&self, delta: i64) {
        self.edge_visits.fetch_add(delta, Ordering::AcqRel);
    }

    pub fn compare_exchange_weak_edge_visits(&self, expected: &mut i64, desired: i64) -> bool {
        // On a failed CAS the atomic reports the actual value in `Err`; write
        // it back into `expected` so that callers looping on this (e.g.
        // `maybe_catch_up_edge_visits`) see the real counter and cannot spin
        // forever when another thread moves it concurrently.
        match self
            .edge_visits
            .compare_exchange_weak(*expected, desired, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => true,
            Err(actual) => {
                *expected = actual;
                false
            }
        }
    }

    pub fn get_move_loc(&self) -> Loc {
        self.move_loc.load(Ordering::Acquire)
    }

    pub fn get_move_loc_relaxed(&self) -> Loc {
        self.move_loc.load(Ordering::Relaxed)
    }

    pub fn set_move_loc(&self, loc: Loc) {
        self.move_loc.store(loc, Ordering::Release);
    }

    pub fn set_move_loc_relaxed(&self, loc: Loc) {
        self.move_loc.store(loc, Ordering::Relaxed);
    }
}

/// Trait used by `SearchNode` to hand off replaced NN outputs for deferred cleanup.
pub trait SearchThread {
    /// Raw pointers to `Box<Arc<NNOutput>>` values that the search will drop later.
    fn old_nn_outputs_to_clean_up(&mut self) -> &mut Vec<*mut Arc<NNOutput>>;
}

/// A view onto a `SearchNode`'s logically concatenated children arrays.
pub struct SearchNodeChildrenReference<'a> {
    pub snapshotted_state: SearchNodeState,
    node: &'a SearchNode,
}

impl<'a> SearchNodeChildrenReference<'a> {
    pub fn new(snapshotted_state: SearchNodeState, node: &'a SearchNode) -> Self {
        Self {
            snapshotted_state,
            node,
        }
    }

    pub fn get(&self, i: usize) -> &SearchChildPointer {
        assert!(i < self.get_capacity(), "child index {} out of capacity", i);
        if i < children_sizes::SIZE0_TOTAL {
            &self.node.children0.as_ref().unwrap()[i]
        } else if i < children_sizes::SIZE1_TOTAL {
            &self.node.children1.as_ref().unwrap()[i - children_sizes::SIZE0_TOTAL]
        } else {
            &self.node.children2.as_ref().unwrap()[i - children_sizes::SIZE1_TOTAL]
        }
    }

    pub fn get_capacity(&self) -> usize {
        children_capacity(self.snapshotted_state)
    }

    pub fn iterate_and_count_children(&self) -> usize {
        let (arr, offset) = if self.snapshotted_state < STATE_EXPANDED0 {
            return 0;
        } else if self.snapshotted_state < STATE_EXPANDED1 {
            (self.node.children0.as_ref().unwrap().as_ref(), 0usize)
        } else if self.snapshotted_state < STATE_EXPANDED2 {
            (
                self.node.children1.as_ref().unwrap().as_ref(),
                children_sizes::SIZE0_TOTAL,
            )
        } else {
            (
                self.node.children2.as_ref().unwrap().as_ref(),
                children_sizes::SIZE1_TOTAL,
            )
        };
        for (i, child) in arr.iter().enumerate() {
            if child.get_if_allocated().is_none() {
                return offset + i;
            }
        }
        offset + arr.len()
    }
}

fn children_capacity(state_value: SearchNodeState) -> usize {
    if state_value < STATE_EXPANDED0 {
        0
    } else if state_value < STATE_EXPANDED1 {
        children_sizes::SIZE0_TOTAL
    } else if state_value < STATE_EXPANDED2 {
        children_sizes::SIZE1_TOTAL
    } else {
        children_sizes::SIZE2_TOTAL
    }
}

/// A node in the MCTS search tree.
pub struct SearchNode {
    pub stats_lock: AtomicBool,

    pub next_pla: Player,
    pub force_non_terminal: bool,
    pub pattern_bonus_hash: Hash128,
    pub mutex_idx: u32,

    pub state: AtomicI32,

    pub nn_output: AtomicPtr<Arc<NNOutput>>,
    pub human_output: AtomicPtr<Arc<NNOutput>>,

    pub node_age: AtomicU32,

    pub children0: Option<Box<[SearchChildPointer]>>,
    pub children1: Option<Box<[SearchChildPointer]>>,
    pub children2: Option<Box<[SearchChildPointer]>>,

    pub stats: NodeStatsAtomic,
    pub virtual_losses: AtomicI32,

    pub last_subtree_value_bias_delta_sum: f64,
    pub last_subtree_value_bias_weight: f64,
    pub subtree_value_bias_table_entry: Option<Arc<SubtreeValueBiasEntry>>,

    pub graph_hash: Hash128,
    pub eval_cache_entry: Option<Arc<EvalCacheEntry>>,

    pub dirty_counter: AtomicI32,
}

impl SearchNode {
    pub fn new(
        next_pla: Player,
        force_non_terminal: bool,
        mutex_idx: u32,
        graph_hash: Hash128,
    ) -> Self {
        Self {
            stats_lock: AtomicBool::new(false),
            next_pla,
            force_non_terminal,
            pattern_bonus_hash: Hash128::default(),
            mutex_idx,
            state: AtomicI32::new(STATE_UNEVALUATED),
            nn_output: AtomicPtr::new(std::ptr::null_mut()),
            human_output: AtomicPtr::new(std::ptr::null_mut()),
            node_age: AtomicU32::new(0),
            children0: None,
            children1: None,
            children2: None,
            stats: NodeStatsAtomic::default(),
            virtual_losses: AtomicI32::new(0),
            last_subtree_value_bias_delta_sum: 0.0,
            last_subtree_value_bias_weight: 0.0,
            subtree_value_bias_table_entry: None,
            graph_hash,
            eval_cache_entry: None,
            dirty_counter: AtomicI32::new(0),
        }
    }

    /// Build a partial copy of `other` for a new search root.
    /// `copy_subtree_value_bias` is not supported and will panic if true.
    pub fn clone_for_tree(
        other: &SearchNode,
        force_non_terminal: bool,
        copy_subtree_value_bias: bool,
    ) -> Self {
        if copy_subtree_value_bias {
            panic!("copy_subtree_value_bias is not supported");
        }

        let mut node = Self {
            stats_lock: AtomicBool::new(false),
            next_pla: other.next_pla,
            force_non_terminal,
            pattern_bonus_hash: other.pattern_bonus_hash,
            mutex_idx: other.mutex_idx,
            state: AtomicI32::new(other.state.load(Ordering::Acquire)),
            nn_output: AtomicPtr::new(std::ptr::null_mut()),
            human_output: AtomicPtr::new(std::ptr::null_mut()),
            node_age: AtomicU32::new(other.node_age.load(Ordering::Acquire)),
            children0: None,
            children1: None,
            children2: None,
            stats: other.stats.clone(),
            virtual_losses: AtomicI32::new(other.virtual_losses.load(Ordering::Acquire)),
            last_subtree_value_bias_delta_sum: 0.0,
            last_subtree_value_bias_weight: 0.0,
            subtree_value_bias_table_entry: None,
            graph_hash: other.graph_hash,
            eval_cache_entry: other.eval_cache_entry.clone(),
            dirty_counter: AtomicI32::new(other.dirty_counter.load(Ordering::Acquire)),
        };

        let other_nn = other.nn_output.load(Ordering::Acquire);
        if !other_nn.is_null() {
            let cloned = Arc::new((**unsafe { &*other_nn }).clone());
            node.nn_output
                .store(Box::into_raw(Box::new(cloned)), Ordering::Release);
        }
        let other_human = other.human_output.load(Ordering::Acquire);
        if !other_human.is_null() {
            let cloned = Arc::new((**unsafe { &*other_human }).clone());
            node.human_output
                .store(Box::into_raw(Box::new(cloned)), Ordering::Release);
        }

        if let Some(ref children) = other.children0 {
            let copy: Vec<SearchChildPointer> = (0..children_sizes::SIZE0_OVERFLOW)
                .map(|_| SearchChildPointer::new())
                .collect();
            for (i, c) in children.iter().enumerate() {
                copy[i].store_all(c);
            }
            node.children0 = Some(copy.into_boxed_slice());
        }
        if let Some(ref children) = other.children1 {
            let copy: Vec<SearchChildPointer> = (0..children_sizes::SIZE1_OVERFLOW)
                .map(|_| SearchChildPointer::new())
                .collect();
            for (i, c) in children.iter().enumerate() {
                copy[i].store_all(c);
            }
            node.children1 = Some(copy.into_boxed_slice());
        }
        if let Some(ref children) = other.children2 {
            let copy: Vec<SearchChildPointer> = (0..children_sizes::SIZE2_OVERFLOW)
                .map(|_| SearchChildPointer::new())
                .collect();
            for (i, c) in children.iter().enumerate() {
                copy[i].store_all(c);
            }
            node.children2 = Some(copy.into_boxed_slice());
        }

        node
    }

    pub fn get_children(&self) -> SearchNodeChildrenReference<'_> {
        SearchNodeChildrenReference::new(self.state.load(Ordering::Acquire), self)
    }

    pub fn get_children_with_state(
        &self,
        state_value: SearchNodeState,
    ) -> SearchNodeChildrenReference<'_> {
        SearchNodeChildrenReference::new(state_value, self)
    }

    pub fn initialize_children(&mut self) {
        assert!(self.children0.is_none());
        let v: Vec<SearchChildPointer> = (0..children_sizes::SIZE0_OVERFLOW)
            .map(|_| SearchChildPointer::new())
            .collect();
        self.children0 = Some(v.into_boxed_slice());
    }

    pub fn maybe_expand_children_capacity_for_new_child(
        &mut self,
        state_value: &mut SearchNodeState,
        num_children_full_plus_one: usize,
    ) -> bool {
        let capacity = children_capacity(*state_value);
        if capacity >= num_children_full_plus_one {
            return true;
        }
        assert_eq!(capacity, num_children_full_plus_one - 1);
        self.try_expanding_children_capacity_assume_full(state_value)
    }

    pub fn collapse_children_capacity(&mut self, num_good_children: usize) {
        let mut state_value = self.state.load(Ordering::Acquire);
        if num_good_children <= children_sizes::SIZE1_TOTAL && state_value > STATE_EXPANDED1 {
            assert_eq!(state_value, STATE_EXPANDED2);
            assert!(self.children2.is_some());
            for child in self.children2.as_ref().unwrap().iter() {
                assert!(!child.is_allocated_relaxed());
            }
            self.children2 = None;
            state_value = STATE_EXPANDED1;
            self.state.store(state_value, Ordering::Release);
        }
        if num_good_children <= children_sizes::SIZE0_TOTAL && state_value > STATE_EXPANDED0 {
            assert_eq!(state_value, STATE_EXPANDED1);
            assert!(self.children1.is_some());
            for child in self.children1.as_ref().unwrap().iter() {
                assert!(!child.is_allocated_relaxed());
            }
            self.children1 = None;
            state_value = STATE_EXPANDED0;
            self.state.store(state_value, Ordering::Release);
        }
    }

    fn try_expanding_children_capacity_assume_full(
        &mut self,
        state_value: &mut SearchNodeState,
    ) -> bool {
        if *state_value < STATE_EXPANDED1 {
            if *state_value == STATE_GROWING1 {
                return false;
            }
            assert_eq!(*state_value, STATE_EXPANDED0);
            match self.state.compare_exchange(
                STATE_EXPANDED0,
                STATE_GROWING1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    *state_value = STATE_GROWING1;
                }
                Err(actual) => {
                    *state_value = actual;
                    return false;
                }
            }
            assert!(self.children1.is_none());
            let v: Vec<SearchChildPointer> = (0..children_sizes::SIZE1_OVERFLOW)
                .map(|_| SearchChildPointer::new())
                .collect();
            self.children1 = Some(v.into_boxed_slice());
            self.state.store(STATE_EXPANDED1, Ordering::Release);
            *state_value = STATE_EXPANDED1;
        } else if *state_value < STATE_EXPANDED2 {
            if *state_value == STATE_GROWING2 {
                return false;
            }
            assert_eq!(*state_value, STATE_EXPANDED1);
            match self.state.compare_exchange(
                STATE_EXPANDED1,
                STATE_GROWING2,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    *state_value = STATE_GROWING2;
                }
                Err(actual) => {
                    *state_value = actual;
                    return false;
                }
            }
            assert!(self.children2.is_none());
            let v: Vec<SearchChildPointer> = (0..children_sizes::SIZE2_OVERFLOW)
                .map(|_| SearchChildPointer::new())
                .collect();
            self.children2 = Some(v.into_boxed_slice());
            self.state.store(STATE_EXPANDED2, Ordering::Release);
            *state_value = STATE_EXPANDED2;
        } else {
            panic!("try_expanding_children_capacity_assume_full: already at max capacity");
        }
        true
    }

    pub fn get_nn_output(&self) -> Option<Arc<NNOutput>> {
        let ptr = self.nn_output.load(Ordering::Acquire);
        if ptr.is_null() {
            None
        } else {
            Some(Arc::clone(unsafe { &*ptr }))
        }
    }

    pub fn get_human_output(&self) -> Option<Arc<NNOutput>> {
        let ptr = self.human_output.load(Ordering::Acquire);
        if ptr.is_null() {
            None
        } else {
            Some(Arc::clone(unsafe { &*ptr }))
        }
    }

    pub fn store_nn_output(
        &self,
        new_nn_output: *mut Arc<NNOutput>,
        thread: &mut dyn SearchThread,
    ) -> bool {
        let to_clean_up = self.nn_output.swap(new_nn_output, Ordering::AcqRel);
        if !to_clean_up.is_null() {
            thread.old_nn_outputs_to_clean_up().push(to_clean_up);
            false
        } else {
            true
        }
    }

    pub fn store_nn_output_if_null(&self, new_nn_output: *mut Arc<NNOutput>) -> bool {
        self.nn_output
            .compare_exchange(
                std::ptr::null_mut(),
                new_nn_output,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    pub fn store_human_output(
        &self,
        new_human_output: *mut Arc<NNOutput>,
        thread: &mut dyn SearchThread,
    ) -> bool {
        let to_clean_up = self.human_output.swap(new_human_output, Ordering::AcqRel);
        if !to_clean_up.is_null() {
            thread.old_nn_outputs_to_clean_up().push(to_clean_up);
            false
        } else {
            true
        }
    }

    pub fn store_human_output_if_null(&self, new_human_output: *mut Arc<NNOutput>) -> bool {
        self.human_output
            .compare_exchange(
                std::ptr::null_mut(),
                new_human_output,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }
}

impl Drop for SearchNode {
    fn drop(&mut self) {
        self.children2 = None;
        self.children1 = None;
        self.children0 = None;

        let nn = self.nn_output.load(Ordering::Relaxed);
        if !nn.is_null() {
            unsafe {
                drop(Box::from_raw(nn));
            }
        }
        let human = self.human_output.load(Ordering::Relaxed);
        if !human.is_null() {
            unsafe {
                drop(Box::from_raw(human));
            }
        }
    }
}

impl EvalCacheChildNode for SearchNode {
    fn visits(&self) -> i64 {
        self.stats.visits.load(Ordering::Acquire)
    }
    fn win_loss_value_avg(&self) -> f64 {
        self.stats.win_loss_value_avg.load(Ordering::Acquire)
    }
    fn score_mean_avg(&self) -> f64 {
        self.stats.score_mean_avg.load(Ordering::Acquire)
    }
    fn utility_avg(&self) -> f64 {
        self.stats.utility_avg.load(Ordering::Acquire)
    }
}

impl EvalCacheNode for SearchNode {
    fn next_pla(&self) -> Player {
        self.next_pla
    }
    fn visits(&self) -> i64 {
        self.stats.visits.load(Ordering::Acquire)
    }
    fn win_loss_value_avg(&self) -> f64 {
        self.stats.win_loss_value_avg.load(Ordering::Acquire)
    }
    fn no_result_value_avg(&self) -> f64 {
        self.stats.no_result_value_avg.load(Ordering::Acquire)
    }
    fn score_mean_avg(&self) -> f64 {
        self.stats.score_mean_avg.load(Ordering::Acquire)
    }
    fn lead_avg(&self) -> f64 {
        self.stats.lead_avg.load(Ordering::Acquire)
    }
    fn utility_avg(&self) -> f64 {
        self.stats.utility_avg.load(Ordering::Acquire)
    }
    fn for_each_child(&self, f: &mut dyn FnMut(Loc, i64, &dyn EvalCacheChildNode)) {
        let children = self.get_children();
        let capacity = children.get_capacity();
        for i in 0..capacity {
            let child_pointer = children.get(i);
            let child = match child_pointer.get_if_allocated_relaxed() {
                Some(c) => c,
                None => break,
            };
            let move_loc = child_pointer.get_move_loc_relaxed();
            let edge_visits = child_pointer.get_edge_visits();
            f(move_loc, edge_visits, child);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kata_game::board::{P_BLACK, P_WHITE};

    fn make_output() -> *mut Arc<NNOutput> {
        Box::into_raw(Box::new(Arc::new(NNOutput::default())))
    }

    #[test]
    fn test_node_stats_child_weight() {
        let stats = NodeStats {
            visits: 10,
            weight_sum: 5.0,
            ..NodeStats::default()
        };
        assert!((stats.get_child_weight(20) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_node_stats_atomic_clone() {
        let a = NodeStatsAtomic::new();
        a.visits.store(7, Ordering::Relaxed);
        a.weight_sum.store(3.0, Ordering::Relaxed);
        let b = a.clone();
        assert_eq!(b.visits.load(Ordering::Acquire), 7);
        assert!((b.weight_sum.load(Ordering::Acquire) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_search_child_pointer_store_if_null() {
        let parent = Box::into_raw(Box::new(SearchNode::new(
            P_WHITE,
            false,
            0,
            Hash128::default(),
        )));
        let child = SearchChildPointer::new();
        assert!(child.store_if_null(parent));
        assert!(!child.store_if_null(parent));
        assert!(child.get_if_allocated().is_some());
        unsafe {
            drop(Box::from_raw(parent));
        }
    }

    #[test]
    fn test_search_child_pointer_edge_visits() {
        let child = SearchChildPointer::new();
        child.set_edge_visits(5);
        child.add_edge_visits(3);
        assert_eq!(child.get_edge_visits(), 8);
    }

    #[test]
    fn test_search_node_default_state() {
        let node = SearchNode::new(P_BLACK, false, 3, Hash128::new(1, 2));
        assert_eq!(node.state.load(Ordering::Acquire), STATE_UNEVALUATED);
        assert_eq!(node.mutex_idx, 3);
        assert!(!node.force_non_terminal);
        assert_eq!(node.next_pla, P_BLACK);
        assert!(node.get_nn_output().is_none());
    }

    #[test]
    fn test_initialize_children_capacity() {
        let mut node = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        node.initialize_children();
        assert!(node.children0.is_some());
        assert_eq!(
            node.children0.as_ref().unwrap().len(),
            children_sizes::SIZE0_OVERFLOW
        );
    }

    #[test]
    fn test_expand_children_capacity() {
        let mut node = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        node.initialize_children();
        node.state.store(STATE_EXPANDED0, Ordering::Release);

        let mut state_value = STATE_EXPANDED0;
        assert!(node.maybe_expand_children_capacity_for_new_child(
            &mut state_value,
            children_sizes::SIZE0_TOTAL + 1
        ));
        assert_eq!(state_value, STATE_EXPANDED1);
        assert!(node.children1.is_some());
        assert_eq!(
            node.children1.as_ref().unwrap().len(),
            children_sizes::SIZE1_OVERFLOW
        );

        assert!(node.maybe_expand_children_capacity_for_new_child(
            &mut state_value,
            children_sizes::SIZE1_TOTAL + 1
        ));
        assert_eq!(state_value, STATE_EXPANDED2);
        assert!(node.children2.is_some());
        assert_eq!(
            node.children2.as_ref().unwrap().len(),
            children_sizes::SIZE2_OVERFLOW
        );
    }

    #[test]
    fn test_iterate_and_count_children() {
        let mut node = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        node.initialize_children();
        node.state.store(STATE_EXPANDED0, Ordering::Release);

        let children = node.get_children();
        assert_eq!(children.get_capacity(), children_sizes::SIZE0_TOTAL);
        assert_eq!(children.iterate_and_count_children(), 0);

        let child_ptr = Box::into_raw(Box::new(SearchNode::new(
            P_WHITE,
            false,
            0,
            Hash128::default(),
        )));
        children.get(0).store(child_ptr);
        children.get(1).store(child_ptr);
        assert_eq!(children.iterate_and_count_children(), 2);

        unsafe {
            drop(Box::from_raw(child_ptr));
        }
    }

    #[test]
    fn test_collapse_children_capacity() {
        let mut node = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        node.initialize_children();
        node.state.store(STATE_EXPANDED0, Ordering::Release);

        let mut state_value = STATE_EXPANDED0;
        node.maybe_expand_children_capacity_for_new_child(
            &mut state_value,
            children_sizes::SIZE0_TOTAL + 1,
        );
        assert_eq!(state_value, STATE_EXPANDED1);

        node.collapse_children_capacity(children_sizes::SIZE0_TOTAL);
        assert_eq!(node.state.load(Ordering::Acquire), STATE_EXPANDED0);
        assert!(node.children1.is_none());
        assert!(node.children2.is_none());
    }

    #[test]
    fn test_store_nn_output_if_null() {
        let node = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        let out = make_output();
        assert!(node.store_nn_output_if_null(out));
        assert!(node.get_nn_output().is_some());
        let out2 = make_output();
        assert!(!node.store_nn_output_if_null(out2));
        unsafe {
            drop(Box::from_raw(out2));
        }
    }

    struct DummyThread {
        cleanup: Vec<*mut Arc<NNOutput>>,
    }

    impl SearchThread for DummyThread {
        fn old_nn_outputs_to_clean_up(&mut self) -> &mut Vec<*mut Arc<NNOutput>> {
            &mut self.cleanup
        }
    }

    #[test]
    fn test_store_nn_output_replaces_and_queues_cleanup() {
        let node = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        let first = make_output();
        assert!(node.store_nn_output_if_null(first));

        let second = make_output();
        let mut thread = DummyThread {
            cleanup: Vec::new(),
        };
        assert!(!node.store_nn_output(second, &mut thread));
        assert_eq!(thread.cleanup.len(), 1);
        assert_eq!(thread.cleanup[0], first);

        for ptr in thread.cleanup {
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }

    #[test]
    fn test_clone_for_tree_preserves_outputs() {
        let mut node = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        node.virtual_losses.store(5, Ordering::Relaxed);
        let out = make_output();
        node.store_nn_output_if_null(out);

        let child_ptr = Box::into_raw(Box::new(SearchNode::new(
            P_WHITE,
            false,
            0,
            Hash128::default(),
        )));
        node.initialize_children();
        node.state.store(STATE_EXPANDED0, Ordering::Relaxed);
        node.get_children().get(0).store(child_ptr);

        let cloned = SearchNode::clone_for_tree(&node, true, false);
        assert_eq!(cloned.virtual_losses.load(Ordering::Acquire), 5);
        assert!(cloned.force_non_terminal);
        assert!(cloned.get_nn_output().is_some());
        assert!(cloned.children0.is_some());
        assert!(cloned.get_children().get(0).get_if_allocated().is_some());

        unsafe {
            drop(Box::from_raw(child_ptr));
        }
    }
}
