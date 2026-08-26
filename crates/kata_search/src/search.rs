//! MCTS search skeleton.
//!
//! Corresponds to `cpp/search/search.h` and the family of `search*.cpp` files.
//! This slice provides the `Search` and `SearchThread` structs, their
//! constructors/destructor, and a method skeleton for every declaration in the
//! header. Non-trivial algorithms are left as `todo!()` or a safe default.

#![allow(
    clippy::all,
    dead_code,
    missing_docs,
    unused_assignments,
    unused_variables,
    clippy::too_many_arguments
)]

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, AtomicPtr, Ordering};
use std::thread::JoinHandle;

use kata_core::hash::Hash128;
use kata_core::logger::Logger;
use kata_core::rng::Rand;
use kata_core::thread::counter::ThreadSafeCounter;
use kata_core::thread::queue::ThreadSafeQueue;
use kata_core::time::timer::ClockTimer;
use kata_game::board::location::{
    self as location, euclidean_distance_squared, get_center_loc, get_mirror_loc, is_central,
    is_near_central,
};
use kata_game::board::{
    Board, C_EMPTY, Color, Loc, MAX_ARR_SIZE, NULL_LOC, P_BLACK, P_WHITE, PASS_LOC, Player,
    get_opp, player_io,
};
use kata_game::graph_hash;
use kata_game::history::BoardHistory;
use kata_game::rules::{Rules, ScoringRule, WhiteHandicapBonusRule};
use kata_game::symmetry;
use kata_nn::backend::NNResultBuf;
use kata_nn::eval::NnEvaluator;
use kata_nn::inputs::{MiscNNInputParams, NNOutput, nn_pos};
use kata_nn::score_value;
use parking_lot::Mutex;
use serde_json::Value;

use crate::analysis::AnalysisData;
use crate::distribution::DistributionTable;
use crate::eval_cache::EvalCacheTable;
use crate::mutex_pool::MutexPool;
use crate::node::{
    MoreNodeStats, NodeStats, STATE_EVALUATING, STATE_EXPANDED0, STATE_UNEVALUATED, SearchNode,
    SearchNodeChildrenReference, SearchNodeState,
};
use crate::node_table::SearchNodeTable;
use crate::params::SearchParams;
use crate::pattern_bonus::PatternBonusTable;
use crate::reported_values::{ReportedSearchContext, ReportedSearchStats, ReportedSearchValues};
use crate::subtree_bias::SubtreeValueBiasTable;
use crate::time_control::TimeControls;

/// Opaque stub for the C++ `KoHashTable` from `game/boardhistory.h`.
///
/// This type is intentionally empty; a real port will replace it with the
/// history-based ko hash tracking logic.
#[derive(Debug, Default)]
pub struct KoHashTable;

impl KoHashTable {
    /// Create a new empty stub table.
    pub fn new() -> Self {
        Self
    }
}

/// Options controlling how the search tree is printed for debugging.
///
/// Corresponds to the C++ `PrintTreeOptions` from `search/searchprint.h`.
#[derive(Debug, Default, Clone)]
pub struct PrintTreeOptions {
    pub max_depth: i32,
    pub max_children_to_show: i32,
    pub min_visits_to_show: i64,
    pub min_visits_to_expand: i64,
    pub min_visits_prop_to_show: f64,
    pub min_visits_prop_to_expand: f64,
    pub max_pv_depth: i32,
    pub print_raw_nn: bool,
    pub print_sqs: bool,
    pub print_avg_shortterm_error: bool,
    pub branch: Vec<Loc>,
    pub also_branch: bool,
}

impl PrintTreeOptions {
    /// Create the default set of tree-printing options.
    pub fn new() -> Self {
        Self {
            max_depth: 1,
            max_children_to_show: 100000,
            min_visits_to_show: 0,
            min_visits_to_expand: 1,
            min_visits_prop_to_show: 0.0,
            min_visits_prop_to_expand: 0.0,
            max_pv_depth: 7,
            print_raw_nn: false,
            print_sqs: false,
            print_avg_shortterm_error: false,
            branch: Vec::new(),
            also_branch: false,
        }
    }

    /// Set the maximum recursion depth when printing the tree.
    pub fn max_depth(mut self, d: i32) -> Self {
        self.max_depth = d;
        self
    }

    /// Set the maximum number of children to display per node.
    pub fn max_children_to_show(mut self, c: i32) -> Self {
        self.max_children_to_show = c;
        self
    }

    /// Set the minimum number of visits a child must have to be shown.
    pub fn min_visits_to_show(mut self, v: i64) -> Self {
        self.min_visits_to_show = v;
        self
    }

    /// Set the minimum number of visits a node must have to be expanded.
    pub fn min_visits_to_expand(mut self, v: i64) -> Self {
        self.min_visits_to_expand = v;
        self
    }

    /// Set the minimum proportion of root visits a child must have to be shown.
    pub fn min_visits_prop_to_show(mut self, p: f64) -> Self {
        self.min_visits_prop_to_show = p;
        self
    }

    /// Set the minimum proportion of root visits a node must have to be expanded.
    pub fn min_visits_prop_to_expand(mut self, p: f64) -> Self {
        self.min_visits_prop_to_expand = p;
        self
    }

    /// Set the maximum principal-variation depth stored in analysis data.
    pub fn max_pv_depth(mut self, d: i32) -> Self {
        self.max_pv_depth = d;
        self
    }

    /// Enable printing of raw NN values (placeholder, currently ignored).
    pub fn print_raw_nn(mut self, b: bool) -> Self {
        self.print_raw_nn = b;
        self
    }

    /// Enable printing of score-mean-squared and weight-squared statistics.
    pub fn print_sqs(mut self, b: bool) -> Self {
        self.print_sqs = b;
        self
    }

    /// Enable printing of average short-term error statistics.
    pub fn print_avg_shortterm_error(mut self, b: bool) -> Self {
        self.print_avg_shortterm_error = b;
        self
    }

    /// Restrict printing to the given move sequence.
    pub fn only_branch(mut self, board: &Board, moves: &str) -> Self {
        self.branch =
            location::parse_sequence(moves, board.x_size, board.y_size).unwrap_or_default();
        self
    }

    /// Like [`PrintTreeOptions::only_branch`], but also display the root branch.
    pub fn also_branch(mut self, board: &Board, moves: &str) -> Self {
        self.branch =
            location::parse_sequence(moves, board.x_size, board.y_size).unwrap_or_default();
        self.also_branch = true;
        self
    }
}

/// Per-thread state used during a search.
pub struct SearchThread {
    pub thread_idx: i32,

    pub pla: Player,
    pub board: Board,
    pub history: BoardHistory,
    pub graph_hash: Hash128,

    /// Path traced down the graph during a playout.
    pub graph_path: PtrHashSet<*const SearchNode>,

    pub should_count_playout: bool,
    pub rand: Rand,

    pub nn_result_buf: NNResultBuf,
    pub stats_buf: Vec<MoreNodeStats>,

    /// Scratch buffer for downweight_bad_children_and_normalize_weight,
    /// owned per thread so backups don't allocate (upstream uses a stack
    /// array; a Vec avoids the fixed MAX_NN_POLICY_SIZE zeroing cost).
    pub stdevs_buf: Vec<f64>,

    pub upper_bound_visits_left: f64,

    /// Replaced `NNOutput` values queued for lazy cleanup.
    pub old_nn_outputs_to_clean_up: Vec<*mut Arc<NNOutput>>,

    /// Debug-only hashes of positions with illegal moves.
    pub illegal_move_hashes: HashSet<Hash128>,
}

/// Cheap hasher for sets keyed by raw node pointers (graph path, tree-walk
/// dedup). Upstream C++ uses identity-hashed `unordered_set`; the default
/// SipHash costs far more than the set operations themselves on these hot
/// paths. The multiply breaks the low-zero-bit alignment pattern of heap
/// pointers.
#[derive(Default)]
pub struct PtrHasher(u64);

impl std::hash::Hasher for PtrHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    }
    fn write_usize(&mut self, n: usize) {
        self.0 = (n as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    fn write_isize(&mut self, n: isize) {
        self.0 = (n as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
}

pub type PtrHashSet<T> = HashSet<T, std::hash::BuildHasherDefault<PtrHasher>>;

impl SearchThread {
    /// Create per-thread state from the current search configuration.
    pub fn new(thread_idx: i32, search: &Search) -> Self {
        // Mirrors upstream makeSeed: the stream differs per thread, per
        // position, per move count, and per search, so noise sequences do not
        // repeat when the same root is searched again.
        let seed = format!(
            "{}$searchThread${}${}${}${}${}",
            search.rand_seed,
            thread_idx,
            search.root_board.pos_hash.hash0,
            search.root_board.pos_hash.hash1,
            search.root_history.move_history.len(),
            search.num_searches_begun,
        );
        Self {
            thread_idx,
            pla: search.root_pla,
            board: search.root_board.clone(),
            history: search.root_history.clone(),
            graph_hash: search.root_graph_hash,
            graph_path: PtrHashSet::with_capacity_and_hasher(256, Default::default()),
            should_count_playout: false,
            rand: Rand::new_from_seed(&seed),
            nn_result_buf: NNResultBuf::new(),
            stats_buf: Vec::with_capacity(nn_pos::MAX_NN_POLICY_SIZE),
            stdevs_buf: Vec::with_capacity(nn_pos::MAX_NN_POLICY_SIZE),
            upper_bound_visits_left: 0.0,
            old_nn_outputs_to_clean_up: Vec::with_capacity(8),
            illegal_move_hashes: HashSet::new(),
        }
    }
}

impl Drop for SearchThread {
    fn drop(&mut self) {
        for ptr in &self.old_nn_outputs_to_clean_up {
            if !ptr.is_null() {
                unsafe {
                    drop(Box::from_raw(*ptr));
                }
            }
        }
    }
}

// SAFETY: A `SearchThread` is owned by exactly one playout worker for the
// duration of a search. Its otherwise-`!Send` fields are thread-private
// scratch state that never outlives the owning worker:
// - `graph_path` holds raw pointers to tree nodes that stay alive for the
//   whole search (owned by `Search::root_node` / the node table) and is only
//   read by the owning thread;
// - `old_nn_outputs_to_clean_up` owns `Box<Arc<NNOutput>>` allocations that
//   are transferred to the shared cleanup buffer (or dropped by `Drop`) before
//   the thread terminates, never concurrently.
unsafe impl Send for SearchThread {}

impl crate::node::SearchThread for SearchThread {
    fn old_nn_outputs_to_clean_up(&mut self) -> &mut Vec<*mut Arc<NNOutput>> {
        &mut self.old_nn_outputs_to_clean_up
    }
}

/// The main MCTS search structure.
pub struct Search<'a> {
    // Constant/immutable during search
    pub root_pla: Player,
    pub root_board: Board,
    pub root_history: BoardHistory,
    pub root_graph_hash: Hash128,
    pub eval_cache_params_hash: Hash128,
    pub root_hint_loc: Loc,

    pub avoid_move_until_by_loc_black: Vec<i32>,
    pub avoid_move_until_by_loc_white: Vec<i32>,
    pub avoid_move_until_rescale_root: bool,

    pub root_sym_dup_loc: [bool; MAX_ARR_SIZE],
    pub root_symmetries: Vec<i32>,
    pub root_prune_only_symmetries: Vec<i32>,

    pub root_safe_area: Option<Box<[Color]>>,
    pub recent_score_center: f64,

    pub mirroring_pla: Player,
    pub mirror_advantage: f64,
    pub mirror_center_symmetry_error: f64,

    pub always_include_owner_map: bool,

    pub search_params: SearchParams,
    pub num_searches_begun: i64,
    pub search_node_age: u32,
    pub pla_that_search_is_for: Player,
    pub pla_that_search_is_for_last_search: Player,
    pub last_search_num_playouts: i64,
    pub effective_search_time_carried_over: f64,

    pub rand_seed: String,

    pub root_ko_hash_table: Option<Box<KoHashTable>>,

    pub value_weight_distribution: Option<Box<DistributionTable>>,

    pub norm_to_t_approx_z: f64,
    pub norm_to_t_approx_table: Vec<f64>,

    pub pattern_bonus_table: Option<Box<PatternBonusTable>>,
    pub external_pattern_bonus_table: Option<Box<PatternBonusTable>>,

    pub eval_cache: Option<Arc<EvalCacheTable>>,

    pub non_search_rand: Rand,

    // Externally owned values
    pub logger: Option<&'a Logger>,
    pub nn_evaluator: Option<&'a NnEvaluator>,
    pub human_evaluator: Option<&'a NnEvaluator>,
    pub nn_x_len: i32,
    pub nn_y_len: i32,
    pub policy_size: i32,

    // Mutated during search
    pub root_node: Option<Box<SearchNode>>,
    pub node_table: Option<Box<SearchNodeTable<SearchNode>>>,
    pub mutex_pool: Option<Box<MutexPool>>,
    pub subtree_value_bias_table: Option<Box<SubtreeValueBiasTable>>,

    // Thread pool
    pub num_threads_spawned: i32,
    pub threads: Option<Vec<JoinHandle<()>>>,
    pub thread_tasks: Option<Box<ThreadSafeQueue<Box<dyn Fn(i32) + Send + Sync>>>>,
    pub thread_tasks_remaining: Option<ThreadSafeCounter>,

    // Lazy cleanup of replaced NN outputs.
    pub old_nn_outputs_to_clean_up_mutex: Mutex<()>,
    pub old_nn_outputs_to_clean_up: Vec<*mut Arc<NNOutput>>,
}

/// Shared access to a `Search` from the parallel playout worker threads.
///
/// The playout path (`run_single_playout` and its helpers) only reads the
/// `Search` configuration and tables; every piece of tree state it mutates is
/// reached through interior mutability: node fields are atomics, stats are
/// updated under the per-node spinlock, the child-array growth is serialized
/// by the node state machine (CAS on `state`), and the node table / mutex pool
/// shards are `parking_lot` mutexes. Fields that genuinely need `&mut Search`
/// (root node, node table, `non_search_rand`, `old_nn_outputs_to_clean_up`, …)
/// are not touched by the workers; the main thread only writes the shared
/// stop/limit atomics during the parallel phase and resumes exclusive `&mut`
/// access after all workers have joined via `std::thread::scope`.
///
/// # Safety
///
/// - `Send`: workers only call `&self` methods through this reference, and all
///   writes performed on their behalf target interior-mutable state, so moving
///   the reference into a worker thread introduces no aliasing `&mut`.
/// - `Sync`: multiple workers may hold copies of the same `&Search`; the same
///   interior-mutability argument applies for concurrent access.
struct SearchPlayoutRef<'b, 'a>(&'b Search<'a>);

unsafe impl<'b, 'a> Send for SearchPlayoutRef<'b, 'a> {}
unsafe impl<'b, 'a> Sync for SearchPlayoutRef<'b, 'a> {}

impl<'b, 'a> Clone for SearchPlayoutRef<'b, 'a> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'b, 'a> Copy for SearchPlayoutRef<'b, 'a> {}

impl<'b, 'a> SearchPlayoutRef<'b, 'a> {
    /// Run one playout against the shared tree.
    ///
    /// Method (rather than direct field access) so that closures capture the
    /// whole `SearchPlayoutRef` — which carries the `unsafe impl Send/Sync` —
    /// instead of precisely capturing the inner `&Search` reference.
    fn run_single_playout(
        &self,
        thread: &mut SearchThread,
        upper_bound_visits_left: f64,
        root_ptr: RootNodePtr,
    ) -> bool {
        self.0
            .run_single_playout(thread, upper_bound_visits_left, root_ptr.0)
    }
}

/// Raw pointer to the search root node, shared with playout worker threads.
///
/// # Safety
///
/// The root `Box<SearchNode>` is owned by `Search::root_node`, which is never
/// replaced or dropped during a search, so the pointer stays valid for the
/// whole parallel phase. Workers only access the node through interior
/// mutability (atomics, the node state-machine CAS, the stats spinlock), which
/// is what makes concurrent playouts through the same tree sound — the same
/// argument as in the C++ original.
#[derive(Clone, Copy)]
struct RootNodePtr(*mut SearchNode);

unsafe impl Send for RootNodePtr {}
unsafe impl Sync for RootNodePtr {}

/// Buffer for old NN outputs collected by parallel playout workers.
///
/// # Safety
///
/// The raw pointers are `Box<Arc<NNOutput>>` allocations owned by exactly one
/// `Vec` at a time: each worker drains its thread-local
/// `SearchThread::old_nn_outputs_to_clean_up` into this shared buffer, and the
/// main thread drains it into `Search::old_nn_outputs_to_clean_up` only after
/// all workers have joined. Ownership never overlaps, so moving the pointers
/// between threads cannot create aliasing or double frees.
struct OldNNOutputBuffer(Vec<*mut Arc<NNOutput>>);

unsafe impl Send for OldNNOutputBuffer {}

impl<'a> Search<'a> {
    const POLICY_ILLEGAL_SELECTION_VALUE: f64 = -1e50;
    const FUTILE_VISITS_PRUNE_VALUE: f64 = -1e40;
    /// Based on sha256 of "search.cpp FORCE_NON_TERMINAL_HASH".
    const FORCE_NON_TERMINAL_HASH: Hash128 = Hash128::new(0xd4c31800cb8809e2, 0xf75f9d2083f2ffca);
    const EVALUATING_SELECTION_VALUE_PENALTY: f64 = 1e20;

    /// Construct a search with a single neural-net evaluator.
    pub fn new(
        params: SearchParams,
        nn_eval: &'a NnEvaluator,
        logger: &'a Logger,
        rand_seed: &str,
    ) -> Self {
        Self::new_with_human(params, nn_eval, None, logger, rand_seed)
    }

    /// Construct a search with optional separate human-style evaluator.
    pub fn new_with_human(
        params: SearchParams,
        nn_eval: &'a NnEvaluator,
        human_eval: Option<&'a NnEvaluator>,
        logger: &'a Logger,
        rand_seed: &str,
    ) -> Self {
        let nn_x_len = nn_eval.nn_x_len();
        let nn_y_len = nn_eval.nn_y_len();
        let policy_size = nn_x_len * nn_y_len + 1;

        let eval_cache = if params.use_eval_cache {
            Some(Arc::new(EvalCacheTable::new(1024)))
        } else {
            None
        };

        let node_table = Box::new(SearchNodeTable::<SearchNode>::new(
            params.node_table_shards_power_of_two as u32,
        ));
        let num_node_table_shards = node_table.num_shards;
        let mutex_pool = Box::new(MutexPool::new(num_node_table_shards));

        let subtree_value_bias_table = if params.subtree_value_bias_factor != 0.0 {
            Some(Box::new(SubtreeValueBiasTable::new(
                params.subtree_value_bias_table_num_shards,
            )))
        } else {
            None
        };

        let root_board = Board::new(19, 19);
        let root_history = BoardHistory::new(root_board.clone(), C_EMPTY, Rules::default(), 0);

        // 值权重分布表（C++ search.cpp:131 无条件创建；valueWeightExponent!=0 时
        // getPlaySelectionValues 需要它，缺失会导致搜索线程 panic）。
        const VALUE_WEIGHT_DEGREES_OF_FREEDOM: f64 = 3.0;
        let value_weight_distribution = Some(Box::new(DistributionTable::new(
            |z| kata_core::math::t_dist_pdf(z, VALUE_WEIGHT_DEGREES_OF_FREEDOM),
            |z| kata_core::math::t_dist_cdf(z, VALUE_WEIGHT_DEGREES_OF_FREEDOM),
            -50.0,
            50.0,
            2000,
        )));

        Self {
            root_pla: C_EMPTY,
            root_board,
            root_history,
            root_graph_hash: Hash128::default(),
            eval_cache_params_hash: params.get_hash(),
            root_hint_loc: NULL_LOC,
            avoid_move_until_by_loc_black: Vec::new(),
            avoid_move_until_by_loc_white: Vec::new(),
            avoid_move_until_rescale_root: false,
            root_sym_dup_loc: [false; MAX_ARR_SIZE],
            root_symmetries: Vec::new(),
            root_prune_only_symmetries: Vec::new(),
            root_safe_area: None,
            recent_score_center: 0.0,
            mirroring_pla: C_EMPTY,
            mirror_advantage: 0.0,
            mirror_center_symmetry_error: 0.0,
            always_include_owner_map: false,
            search_params: params,
            num_searches_begun: 0,
            search_node_age: 0,
            pla_that_search_is_for: C_EMPTY,
            pla_that_search_is_for_last_search: C_EMPTY,
            last_search_num_playouts: 0,
            effective_search_time_carried_over: 0.0,
            rand_seed: rand_seed.to_string(),
            root_ko_hash_table: None,
            value_weight_distribution,
            norm_to_t_approx_z: 0.0,
            norm_to_t_approx_table: Vec::new(),
            pattern_bonus_table: None,
            external_pattern_bonus_table: None,
            eval_cache,
            non_search_rand: Rand::new_from_seed(rand_seed),
            logger: Some(logger),
            nn_evaluator: Some(nn_eval),
            human_evaluator: human_eval,
            nn_x_len,
            nn_y_len,
            policy_size,
            root_node: None,
            node_table: Some(node_table),
            mutex_pool: Some(mutex_pool),
            subtree_value_bias_table,
            num_threads_spawned: 0,
            threads: None,
            thread_tasks: None,
            thread_tasks_remaining: None,
            old_nn_outputs_to_clean_up_mutex: Mutex::new(()),
            old_nn_outputs_to_clean_up: Vec::new(),
        }
    }
}

impl<'a> Search<'a> {
    pub fn get_root_board(&self) -> &Board {
        &self.root_board
    }

    pub fn get_root_hist(&self) -> &BoardHistory {
        &self.root_history
    }

    pub fn get_root_pla(&self) -> Player {
        self.root_pla
    }

    pub fn get_playout_doubling_advantage_pla(&self) -> Player {
        self.search_params.playout_doubling_advantage_pla
    }

    /// Convert a board location to the neural-net flat position for the root board.
    pub fn get_pos(&self, move_loc: Loc) -> i32 {
        nn_pos::loc_to_pos(
            move_loc,
            self.root_board.x_size,
            self.nn_x_len,
            self.nn_y_len,
        )
    }

    pub fn set_position(&mut self, pla: Player, board: &Board, history: &BoardHistory) {
        self.root_pla = pla;
        self.root_board = board.clone();
        self.root_history = history.clone();
        self.root_graph_hash = Hash128::default();
        self.clear_search();
    }

    pub fn set_player_and_clear_history(&mut self, pla: Player) {
        self.root_pla = pla;
        self.root_history = BoardHistory::new(
            self.root_board.clone(),
            pla,
            self.root_history.rules,
            self.root_history.encore_phase,
        );
        self.clear_search();
    }

    pub fn set_player_if_new(&mut self, pla: Player) {
        if self.root_pla != pla {
            self.root_pla = pla;
            self.clear_search();
        }
    }

    pub fn set_komi_if_new(&mut self, new_komi: f64) {
        let old_komi = self.root_history.rules.komi;
        self.root_history.rules.set_komi(new_komi as f32);
        if self.root_history.rules.komi != old_komi {
            self.clear_search();
        }
    }

    pub fn set_root_hint_loc(&mut self, hint_loc: Loc) {
        // When we positively change the hint loc, we clear the search to make
        // absolutely sure that the hintloc takes effect, and that all nnevals
        // (including the root noise that adds the hintloc) have a chance to
        // happen.
        if hint_loc != NULL_LOC && self.root_hint_loc != hint_loc {
            self.clear_search();
        }
        self.root_hint_loc = hint_loc;
    }

    pub fn set_avoid_move_until_by_loc(&mut self, b_vec: &[i32], w_vec: &[i32]) {
        if self.avoid_move_until_by_loc_black == b_vec
            && self.avoid_move_until_by_loc_white == w_vec
        {
            return;
        }
        self.clear_search();
        self.avoid_move_until_by_loc_black = b_vec.to_vec();
        self.avoid_move_until_by_loc_white = w_vec.to_vec();
    }

    pub fn set_avoid_move_until_rescale_root(&mut self, b: bool) {
        self.avoid_move_until_rescale_root = b;
    }

    pub fn set_always_include_owner_map(&mut self, b: bool) {
        if !self.always_include_owner_map && b {
            self.clear_search();
        }
        self.always_include_owner_map = b;
    }

    pub fn set_root_symmetry_pruning_only(&mut self, root_prune_only_symmetries: &[i32]) {
        if self.root_prune_only_symmetries == root_prune_only_symmetries {
            return;
        }
        self.clear_search();
        self.root_prune_only_symmetries = root_prune_only_symmetries.to_vec();
    }

    pub fn set_params(&mut self, params: &SearchParams) {
        self.search_params = params.clone();
        self.eval_cache_params_hash = params.get_hash();
        self.clear_search();
    }

    pub fn set_params_no_clearing(&mut self, params: &SearchParams) {
        self.search_params = params.clone();
        self.eval_cache_params_hash = params.get_hash();
    }

    pub fn set_external_pattern_bonus_table(&mut self, table: Option<Box<PatternBonusTable>>) {
        self.external_pattern_bonus_table = table;
    }

    pub fn set_copy_of_external_pattern_bonus_table(
        &mut self,
        table: &Option<Box<PatternBonusTable>>,
    ) {
        self.external_pattern_bonus_table = table.clone();
    }

    pub fn set_external_eval_cache(&mut self, cache: Option<Arc<EvalCacheTable>>) {
        self.eval_cache = cache;
    }

    pub fn set_nn_eval(&mut self, nn_eval: Option<&'a NnEvaluator>) {
        self.nn_evaluator = nn_eval;
        if let Some(eval) = nn_eval {
            self.nn_x_len = eval.nn_x_len();
            self.nn_y_len = eval.nn_y_len();
            self.policy_size = self.nn_x_len * self.nn_y_len + 1;
        }
    }

    pub fn respawn_threads(&mut self) {
        self.kill_threads();
        self.spawn_threads_if_needed();
    }

    pub fn clear_search(&mut self) {
        self.effective_search_time_carried_over = 0.0;
        if self.root_node.is_some() {
            self.delete_all_table_nodes_multithreaded();
            self.root_node = None;
        }
        self.clear_old_nn_outputs();
        self.search_node_age = 0;
        self.last_search_num_playouts = 0;
        self.num_searches_begun = 0;
    }

    pub fn make_move(&mut self, move_loc: Loc, move_pla: Player) -> bool {
        self.make_move_with_prevent(move_loc, move_pla, false)
    }

    pub fn make_move_with_prevent(
        &mut self,
        move_loc: Loc,
        move_pla: Player,
        prevent_encore: bool,
    ) -> bool {
        if !self.is_legal_tolerant(move_loc, move_pla) {
            return false;
        }

        if move_pla != self.root_pla {
            self.set_player_and_clear_history(move_pla);
        }

        let old_white_handicap_bonus_score = self.root_history.white_handicap_bonus_score;

        let made = self.root_history.make_board_move_tolerant_with_prevent(
            &mut self.root_board,
            move_loc,
            self.root_pla,
            prevent_encore,
        );
        if !made {
            return false;
        }
        self.root_pla = get_opp(self.root_pla);

        if self.root_node.is_some() {
            let root_ptr = self.root_node.as_deref().unwrap() as *const SearchNode;
            let children = unsafe { &*root_ptr }.get_children();
            let capacity = children.get_capacity();
            let mut child_ptr: *mut SearchNode = std::ptr::null_mut();
            for i in 0..capacity {
                let child_pointer = children.get(i);
                let candidate = match child_pointer.get_if_allocated_relaxed() {
                    Some(c) => c,
                    None => break,
                };
                if child_pointer.get_move_loc_relaxed() == move_loc {
                    if candidate.get_nn_output().is_some() {
                        child_ptr = candidate as *const SearchNode as *mut SearchNode;
                    }
                    break;
                }
            }

            if !child_ptr.is_null() {
                let root_visits = unsafe { &*root_ptr }.stats.visits.load(Ordering::Acquire);
                let child_visits = unsafe { &*child_ptr }.stats.visits.load(Ordering::Acquire);
                let mut visit_proportion = if root_visits > 0 {
                    child_visits as f64 / root_visits as f64
                } else {
                    0.0
                };
                if visit_proportion > 1.0 {
                    visit_proportion = 1.0;
                }
                self.effective_search_time_carried_over *=
                    visit_proportion * self.search_params.tree_reuse_carry_over_time_factor;

                let old_root = self.root_node.take().unwrap();
                let force_non_terminal = self.root_history.is_game_finished;
                let new_root = Box::new(SearchNode::clone_for_tree(
                    unsafe { &*child_ptr },
                    force_non_terminal,
                    false,
                ));
                self.root_node = Some(new_root);

                {
                    let new_root_ptr = self.root_node.as_deref().unwrap() as *const SearchNode;
                    self.apply_recursively_any_order_multithreaded(
                        &[unsafe { &*new_root_ptr }],
                        &|_node, _idx| {},
                    );
                }

                self.delete_all_old_or_all_new_table_nodes_and_subtree_value_bias_multithreaded(
                    true,
                );
                let _ = old_root;
            } else {
                self.clear_search();
            }
        }

        self.avoid_move_until_by_loc_black.clear();
        self.avoid_move_until_by_loc_white.clear();

        if self.root_history.white_handicap_bonus_score != old_white_handicap_bonus_score {
            self.clear_search();
        }
        if self.search_params.conservative_pass
            && self
                .root_history
                .pass_would_end_game(&self.root_board, self.root_pla)
        {
            self.clear_search();
        }
        if prevent_encore
            && self
                .root_history
                .pass_would_end_phase(&self.root_board, self.root_pla)
        {
            self.clear_search();
        }

        true
    }

    pub fn is_legal_tolerant(&self, move_loc: Loc, move_pla: Player) -> bool {
        self.root_board.is_legal(move_loc, move_pla, true)
    }

    pub fn is_legal_strict(&self, move_loc: Loc, move_pla: Player) -> bool {
        self.root_history
            .is_legal(&self.root_board, move_loc, move_pla)
    }

    pub fn run_whole_search_and_get_move(&mut self, move_pla: Player) -> Loc {
        self.run_whole_search_and_get_move_impl(move_pla, false, None)
    }

    pub fn run_whole_search(&mut self, move_pla: Player) {
        self.run_whole_search_impl(move_pla, false, None);
    }

    pub fn run_whole_search_with_stop(
        &mut self,
        move_pla: Player,
        should_stop_early: Option<&dyn Fn() -> bool>,
    ) {
        self.run_whole_search_impl(move_pla, false, should_stop_early);
    }

    pub fn run_whole_search_and_get_move_pondering(
        &mut self,
        move_pla: Player,
        pondering: bool,
    ) -> Loc {
        self.run_whole_search_and_get_move_impl(move_pla, pondering, None)
    }

    pub fn run_whole_search_pondering(&mut self, move_pla: Player, pondering: bool) {
        self.run_whole_search_impl(move_pla, pondering, None);
    }

    pub fn run_whole_search_with_stop_pondering(
        &mut self,
        move_pla: Player,
        pondering: bool,
        should_stop_early: Option<&dyn Fn() -> bool>,
    ) {
        self.run_whole_search_impl(move_pla, pondering, should_stop_early);
    }

    fn run_whole_search_impl(
        &mut self,
        move_pla: Player,
        pondering: bool,
        should_stop_early: Option<&dyn Fn() -> bool>,
    ) {
        let _ = move_pla;
        self.run_whole_search_full(
            None,
            should_stop_early,
            pondering,
            &TimeControls::new(),
            1.0,
        );
    }

    fn run_whole_search_and_get_move_impl(
        &mut self,
        move_pla: Player,
        pondering: bool,
        should_stop_early: Option<&dyn Fn() -> bool>,
    ) -> Loc {
        self.run_whole_search_impl(move_pla, pondering, should_stop_early);
        self.get_chosen_move_loc()
    }

    pub fn run_whole_search_full(
        &mut self,
        search_begun: Option<&dyn Fn()>,
        should_stop_early: Option<&dyn Fn() -> bool>,
        pondering: bool,
        tc: &TimeControls,
        mut search_factor: f64,
    ) {
        use std::sync::atomic::{AtomicBool, AtomicI64, Ordering as AtomicOrdering};

        let timer = ClockTimer::new();
        // Shared stop/limit state: `Arc` so the parallel playout workers and the
        // controller thread can each hold their own copy.
        let num_playouts_shared = std::sync::Arc::new(AtomicI64::new(0));
        let should_stop_now = std::sync::Arc::new(AtomicBool::new(false));

        self.begin_search(pondering);
        if let Some(cb) = search_begun {
            cb();
        }
        let num_non_playout_visits = self.get_root_visits();

        let mut max_visits = if pondering {
            self.search_params.max_visits_pondering
        } else {
            self.search_params.max_visits
        };
        let mut max_playouts = if pondering {
            self.search_params.max_playouts_pondering
        } else {
            self.search_params.max_playouts
        };
        let mut max_time = if pondering {
            self.search_params.max_time_pondering
        } else {
            self.search_params.max_time
        };

        {
            let move_history = &self.root_history.move_history;
            if move_history.len() >= 1 && move_history[move_history.len() - 1].loc == PASS_LOC {
                if move_history.len() >= 3 && move_history[move_history.len() - 3].loc == PASS_LOC {
                    search_factor *= self.search_params.search_factor_after_two_pass;
                } else {
                    search_factor *= self.search_params.search_factor_after_one_pass;
                }
            }

            if search_factor != 1.0 {
                let cap = (1_i64 << 62) as f64;
                max_visits = ((max_visits as f64 * search_factor).min(cap)).ceil() as i64;
                max_playouts = ((max_playouts as f64 * search_factor).min(cap)).ceil() as i64;
                max_time *= search_factor;
            }
        }

        let cap_threads = 0x3fff_ffff_i32;
        let _ = cap_threads;

        let upper_bound_visits_left_due_to_time = std::sync::Arc::new(AtomicI64::new(0));
        let has_max_time = max_time < 1.0e12;
        let has_tc = !pondering && !tc.is_effectively_unlimited_time();
        let tc_max_time_value = std::sync::Arc::new(AtomicI64::new(0));

        if !pondering && (has_tc || has_max_time) {
            let num_playouts = num_playouts_shared.load(AtomicOrdering::Relaxed);
            let root_visits = num_playouts + num_non_playout_visits;
            let time_used = timer.get_seconds();
            let mut tc_limit = 1e30;
            if has_tc {
                tc_limit =
                    self.recompute_search_time_limit(tc, time_used, search_factor, root_visits);
            }
            let upper_bound = self.compute_upper_bound_visits_left_due_to_time(
                root_visits,
                time_used,
                tc_limit.min(max_time),
            );
            tc_max_time_value.store(tc_limit.to_bits() as i64, AtomicOrdering::Release);
            upper_bound_visits_left_due_to_time
                .store(upper_bound.to_bits() as i64, AtomicOrdering::Release);
        }

        let actual_search_start_time = timer.get_seconds();
        // The root node is guaranteed to exist after `begin_search`; the
        // pointer stays valid for the whole search (the root Box is never
        // replaced mid-search), so workers can share it without re-deriving it
        // from `&self`.
        let root_ptr = RootNodePtr(self.root_node.as_deref_mut().unwrap() as *mut SearchNode);

        let num_threads = self.search_params.num_threads.max(1);
        if num_threads <= 1 {
            // Single-threaded fallback: mirrors the upstream serial loop,
            // which builds one SearchThread for the whole search (the RNG
            // reseeding, board/history clones, and Vec growth of a
            // per-playout construction are pure overhead).
            let mut last_time_used_recomputing_tc_limit = 0.0;
            let mut num_playouts = 0_i64;
            let mut thread = SearchThread::new(0, &*self);

            loop {
                let time_used = if has_tc || has_max_time {
                    timer.get_seconds()
                } else {
                    0.0
                };

                let tc_max_time_limit = if has_tc {
                    f64::from_bits(tc_max_time_value.load(AtomicOrdering::Acquire) as u64)
                } else {
                    0.0
                };

                let mut should_stop = num_playouts >= max_playouts
                    || num_playouts + num_non_playout_visits >= max_visits;

                if has_max_time && num_playouts >= 2 && time_used >= max_time {
                    should_stop = true;
                }
                if has_tc && num_playouts >= 2 && time_used >= tc_max_time_limit {
                    should_stop = true;
                }
                if let Some(stop) = should_stop_early {
                    if stop() {
                        should_stop = true;
                    }
                }

                if should_stop || should_stop_now.load(AtomicOrdering::Relaxed) {
                    should_stop_now.store(true, AtomicOrdering::Relaxed);
                    break;
                }

                if !pondering
                    && (has_tc || has_max_time)
                    && time_used >= last_time_used_recomputing_tc_limit + 0.1
                {
                    let root_visits = num_playouts + num_non_playout_visits;
                    let mut tc_limit = 1e30;
                    if has_tc {
                        tc_limit = self
                            .recompute_search_time_limit(tc, time_used, search_factor, root_visits);
                        tc_max_time_value.store(tc_limit.to_bits() as i64, AtomicOrdering::Release);
                    }
                    let upper_bound = self.compute_upper_bound_visits_left_due_to_time(
                        root_visits,
                        time_used,
                        tc_limit.min(max_time),
                    );
                    upper_bound_visits_left_due_to_time
                        .store(upper_bound.to_bits() as i64, AtomicOrdering::Release);
                    last_time_used_recomputing_tc_limit = time_used;
                }

                let mut upper_bound_visits_left = 1e30;
                if has_tc {
                    upper_bound_visits_left = f64::from_bits(
                        upper_bound_visits_left_due_to_time.load(AtomicOrdering::Acquire) as u64,
                    );
                }
                upper_bound_visits_left =
                    upper_bound_visits_left.min(max_playouts as f64 - num_playouts as f64);
                upper_bound_visits_left = upper_bound_visits_left
                    .min(max_visits as f64 - num_playouts as f64 - num_non_playout_visits as f64);

                let finished =
                    self.run_single_playout(&mut thread, upper_bound_visits_left, root_ptr.0);
                if finished {
                    num_playouts += 1;
                    num_playouts_shared.fetch_add(1, AtomicOrdering::Relaxed);
                } else {
                    std::thread::yield_now();
                }
                self.transfer_old_nn_outputs(&mut thread);
            }
        } else {
            // Parallel playout: `num_threads` workers each run playouts against
            // the shared tree (node access is already thread-safe via the CAS
            // state machine, per-node stats spinlock and sharded node table, as
            // in the C++ original), so NN requests now arrive at the backend
            // concurrently and can be batched. The main thread acts as a
            // controller: it enforces the stop conditions
            // (max_visits/max_playouts/max_time/should_stop_early) and
            // periodically recomputes the time-control limit, exactly as the
            // serial loop did per-iteration.
            let search_shared = SearchPlayoutRef(&*self);
            let old_outputs =
                std::sync::Arc::new(std::sync::Mutex::new(OldNNOutputBuffer(Vec::new())));
            let mut last_time_used_recomputing_tc_limit = 0.0;

            std::thread::scope(|s| {
                for worker_idx in 0..num_threads {
                    let thread = SearchThread::new(worker_idx, &*self);
                    let search_shared = search_shared;
                    let old_outputs = std::sync::Arc::clone(&old_outputs);
                    let should_stop_now = std::sync::Arc::clone(&should_stop_now);
                    let num_playouts_shared = std::sync::Arc::clone(&num_playouts_shared);
                    let upper_bound_visits_left_due_to_time =
                        std::sync::Arc::clone(&upper_bound_visits_left_due_to_time);
                    s.spawn(move || {
                        let mut thread = thread;
                        loop {
                            if should_stop_now.load(AtomicOrdering::Relaxed) {
                                break;
                            }
                            let num_playouts =
                                num_playouts_shared.load(AtomicOrdering::Relaxed);
                            let mut upper_bound_visits_left = if has_tc {
                                f64::from_bits(
                                    upper_bound_visits_left_due_to_time
                                    .load(AtomicOrdering::Acquire) as u64,
                                )
                            } else {
                                1e30
                            };
                            upper_bound_visits_left = upper_bound_visits_left
                                .min(max_playouts as f64 - num_playouts as f64);
                            upper_bound_visits_left = upper_bound_visits_left.min(
                                max_visits as f64 - num_playouts as f64
                                    - num_non_playout_visits as f64,
                            );

                            let finished = search_shared.run_single_playout(
                                &mut thread,
                                upper_bound_visits_left,
                                root_ptr,
                            );
                            if finished {
                                num_playouts_shared.fetch_add(1, AtomicOrdering::Relaxed);
                            } else {
                                std::thread::yield_now();
                            }
                        }
                        // Hand the worker's lazily-collected old NN outputs to
                        // the shared buffer for cleanup by the main thread.
                        old_outputs
                            .lock()
                            .unwrap()
                            .0
                            .append(&mut thread.old_nn_outputs_to_clean_up);
                    });
                }

                // Controller: same stop logic as the serial loop, driven by the
                // shared playout counter.
                loop {
                    if should_stop_now.load(AtomicOrdering::Relaxed) {
                        break;
                    }
                    let time_used = if has_tc || has_max_time {
                        timer.get_seconds()
                    } else {
                        0.0
                    };
                    let tc_max_time_limit = if has_tc {
                        f64::from_bits(tc_max_time_value.load(AtomicOrdering::Acquire) as u64)
                    } else {
                        0.0
                    };

                    let num_playouts = num_playouts_shared.load(AtomicOrdering::Relaxed);
                    let mut should_stop = num_playouts >= max_playouts
                        || num_playouts + num_non_playout_visits >= max_visits;
                    if has_max_time && num_playouts >= 2 && time_used >= max_time {
                        should_stop = true;
                    }
                    if has_tc && num_playouts >= 2 && time_used >= tc_max_time_limit {
                        should_stop = true;
                    }
                    if let Some(stop) = should_stop_early {
                        if stop() {
                            should_stop = true;
                        }
                    }
                    if should_stop {
                        should_stop_now.store(true, AtomicOrdering::Relaxed);
                        break;
                    }

                    if !pondering
                        && (has_tc || has_max_time)
                        && time_used >= last_time_used_recomputing_tc_limit + 0.1
                    {
                        let root_visits = num_playouts + num_non_playout_visits;
                        let mut tc_limit = 1e30;
                        if has_tc {
                            tc_limit = self.recompute_search_time_limit(
                                tc,
                                time_used,
                                search_factor,
                                root_visits,
                            );
                            tc_max_time_value
                                .store(tc_limit.to_bits() as i64, AtomicOrdering::Release);
                        }
                        let upper_bound = self.compute_upper_bound_visits_left_due_to_time(
                            root_visits,
                            time_used,
                            tc_limit.min(max_time),
                        );
                        upper_bound_visits_left_due_to_time
                            .store(upper_bound.to_bits() as i64, AtomicOrdering::Release);
                        last_time_used_recomputing_tc_limit = time_used;
                    }

                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
            });

            self.old_nn_outputs_to_clean_up
                .append(&mut old_outputs.lock().unwrap().0);
        }

        if self.root_node.is_some() {
            let root_ptr = self.root_node.as_deref_mut().unwrap() as *mut SearchNode;
            let root_age = unsafe { &*root_ptr }.node_age.load(Ordering::Acquire);
            if root_age != self.search_node_age && unsafe { &*root_ptr }.get_nn_output().is_some() {
                let mut thread = SearchThread::new(0, &*self);
                self.maybe_recompute_existing_nn_output(
                    &mut thread,
                    unsafe { &mut *root_ptr },
                    true,
                );
                self.transfer_old_nn_outputs(&mut thread);
            }
        }

        if self.search_params.use_eval_cache
            && self.search_params.use_graph_search
            && self.eval_cache.is_some()
            && self.root_node.is_some()
            && self.mirroring_pla == C_EMPTY
        {
            let root_ptr = self.root_node.as_deref_mut().unwrap() as *mut SearchNode;
            self.recursively_record_eval_cache(unsafe { &mut *root_ptr });
        }

        self.last_search_num_playouts = num_playouts_shared.load(AtomicOrdering::Relaxed);
        self.effective_search_time_carried_over += timer.get_seconds() - actual_search_start_time;
    }

    pub fn maybe_recompute_root_nn_output(&mut self) {
        if self.root_node.is_none() {
            return;
        }

        let root_ptr = self.root_node.as_deref_mut().unwrap() as *mut SearchNode;
        let root_age = unsafe { &*root_ptr }.node_age.load(Ordering::Acquire);
        if root_age != self.search_node_age && unsafe { &*root_ptr }.get_nn_output().is_some() {
            let mut thread = SearchThread::new(0, &*self);
            self.maybe_recompute_existing_nn_output(&mut thread, unsafe { &mut *root_ptr }, true);
            self.transfer_old_nn_outputs(&mut thread);
        }
    }

    pub fn begin_search(&mut self, pondering: bool) {
        if self.root_board.x_size > self.nn_x_len || self.root_board.y_size > self.nn_y_len {
            panic!(
                "Search got NNEval nnXLen = {} nnYLen = {} but was asked to search board with larger x or y size",
                self.nn_x_len, self.nn_y_len
            );
        }

        self.num_searches_begun += 1;

        // Avoid any issues in principle from rolling over
        if self.search_node_age > 0x3FFFFFFF {
            self.clear_search();
        }

        if !pondering {
            self.pla_that_search_is_for = self.root_pla;
        }
        if self.pla_that_search_is_for == C_EMPTY {
            self.pla_that_search_is_for = get_opp(self.root_pla);
        }

        if self.pla_that_search_is_for_last_search != self.pla_that_search_is_for {
            if self.search_params.playout_doubling_advantage != 0.0
                && self.search_params.playout_doubling_advantage_pla == C_EMPTY
            {
                self.clear_search();
            }
            if self.search_params.avoid_repeated_pattern_utility != 0.0
                || self.external_pattern_bonus_table.is_some()
            {
                self.clear_search();
            }
            if self.human_evaluator.is_some() {
                if self.search_params.human_sl_pla_explore_prob_weightless
                    != self.search_params.human_sl_opp_explore_prob_weightless
                    || self.search_params.human_sl_pla_explore_prob_weightful
                        != self.search_params.human_sl_opp_explore_prob_weightful
                    || self.search_params.human_sl_pla_explore_prob_weightless
                        != self.search_params.human_sl_root_explore_prob_weightless
                    || self.search_params.human_sl_pla_explore_prob_weightful
                        != self.search_params.human_sl_root_explore_prob_weightful
                {
                    self.clear_search();
                }
            }
        }
        self.pla_that_search_is_for_last_search = self.pla_that_search_is_for;

        self.clear_old_nn_outputs();
        self.compute_root_values();

        if self.search_params.subtree_value_bias_factor != 0.0
            && self.subtree_value_bias_table.is_none()
            && !(self.search_params.anti_mirror && self.mirroring_pla != C_EMPTY)
        {
            self.subtree_value_bias_table = Some(Box::new(SubtreeValueBiasTable::new(
                self.search_params.subtree_value_bias_table_num_shards,
            )));
        }

        if self.search_params.use_eval_cache
            && self.search_params.use_graph_search
            && self.eval_cache.is_none()
            && self.mirroring_pla == C_EMPTY
        {
            self.eval_cache = Some(Arc::new(EvalCacheTable::new(
                self.search_params.subtree_value_bias_table_num_shards as u32,
            )));
        }

        if self.pattern_bonus_table.is_some() {
            self.pattern_bonus_table = None;
        }
        if self.search_params.avoid_repeated_pattern_utility != 0.0
            || self.external_pattern_bonus_table.is_some()
        {
            if let Some(external) = &self.external_pattern_bonus_table {
                self.pattern_bonus_table = Some(external.clone());
            } else {
                self.pattern_bonus_table = Some(Box::new(PatternBonusTable::new()));
            }
            if self.search_params.avoid_repeated_pattern_utility != 0.0 {
                let bonus = if self.pla_that_search_is_for == P_WHITE {
                    -self.search_params.avoid_repeated_pattern_utility
                } else {
                    self.search_params.avoid_repeated_pattern_utility
                };
                self.pattern_bonus_table
                    .as_ref()
                    .unwrap()
                    .add_bonus_for_game_moves_player(
                        &self.root_history,
                        bonus,
                        self.pla_that_search_is_for,
                    );
            }
            if let Some(root) = self.root_node.as_mut() {
                root.pattern_bonus_hash = Hash128::default();
            }
        }

        if self.search_params.root_symmetry_pruning {
            let avoid = if self.root_pla == P_BLACK {
                &self.avoid_move_until_by_loc_black
            } else {
                &self.avoid_move_until_by_loc_white
            };
            let only = if self.root_prune_only_symmetries.is_empty() {
                None
            } else {
                Some(self.root_prune_only_symmetries.as_slice())
            };
            let (dup_loc, symmetries) = symmetry::mark_duplicate_move_locs(
                &self.root_board,
                &self.root_history,
                only,
                avoid,
            );
            self.root_sym_dup_loc = dup_loc.try_into().expect("dup_loc length mismatch");
            self.root_symmetries = symmetries;
        } else {
            self.root_sym_dup_loc = [false; MAX_ARR_SIZE];
            self.root_symmetries.clear();
            self.root_symmetries.push(0);
        }

        let mut dummy_thread = SearchThread::new(-1, self);

        if self.search_params.use_graph_search {
            self.root_graph_hash = graph_hash::get_graph_hash_from_scratch(
                &self.root_history,
                self.root_pla,
                self.search_params.graph_search_rep_bound,
                self.search_params.draw_equivalent_wins_for_white,
            );
        } else {
            self.root_graph_hash = Hash128::default();
        }

        if self.search_params.use_eval_cache && self.search_params.use_graph_search {
            self.eval_cache_params_hash = self.search_params.get_hash();
        } else {
            self.eval_cache_params_hash = Hash128::default();
        }

        if self.root_node.is_none() {
            let force_non_terminal = self.root_history.is_game_finished;
            let mutex_idx = self.create_mutex_idx_for_node(&mut dummy_thread);
            let mut root = Box::new(SearchNode::new(
                self.root_pla,
                force_non_terminal,
                mutex_idx,
                self.root_graph_hash,
            ));
            if self.search_params.use_eval_cache
                && self.search_params.use_graph_search
                && self.eval_cache.is_some()
                && self.mirroring_pla == C_EMPTY
            {
                let key = self.get_eval_cache_key(self.root_graph_hash);
                root.eval_cache_entry = self.eval_cache.as_ref().and_then(|c| c.find(key));
            }
            self.root_node = Some(root);
        } else {
            let root_ptr = self.root_node.as_deref_mut().unwrap() as *mut SearchNode;
            let any_filtered = unsafe {
                let root = &mut *root_ptr;
                let children = root.get_children();
                let children_capacity = children.get_capacity();
                let mut any_filtered = false;
                if children_capacity > 0 {
                    let mut num_good_children = 0;
                    let mut filtered_nodes: Vec<*mut SearchNode> = Vec::new();
                    let mut i = 0;
                    while i < children_capacity {
                        let child_ptr = children.get(i).get_raw_ptr();
                        let edge_visits = children.get(i).get_edge_visits();
                        let move_loc = children.get(i).get_move_loc();
                        if child_ptr.is_null() {
                            break;
                        }
                        children.get(i).store(std::ptr::null_mut());
                        children.get(i).set_edge_visits(0);
                        children.get(i).set_move_loc(NULL_LOC);
                        if self
                            .root_history
                            .is_legal(&self.root_board, move_loc, self.root_pla)
                            && self.is_allowed_root_move(move_loc)
                        {
                            children.get(num_good_children).store(child_ptr);
                            children.get(num_good_children).set_edge_visits(edge_visits);
                            children.get(num_good_children).set_move_loc(move_loc);
                            num_good_children += 1;
                        } else {
                            any_filtered = true;
                            filtered_nodes.push(child_ptr);
                        }
                        i += 1;
                    }
                    while i < children_capacity {
                        debug_assert!(children.get(i).get_raw_ptr().is_null());
                        i += 1;
                    }

                    if any_filtered {
                        root.collapse_children_capacity(num_good_children);
                        let children = root.get_children();
                        let new_capacity = children.get_capacity();
                        let mut new_num_visits = 0_i64;
                        for i in 0..new_capacity {
                            let child = children.get(i).get_raw_ptr();
                            if child.is_null() {
                                break;
                            }
                            new_num_visits += children.get(i).get_edge_visits();
                        }
                        new_num_visits += 1;

                        while root.stats_lock.swap(true, Ordering::Acquire) {}
                        root.stats.visits.store(new_num_visits, Ordering::Release);
                        root.stats_lock.store(false, Ordering::Release);

                        self.recompute_node_stats(root, &mut dummy_thread, 0, true);
                    }
                }
                any_filtered
            };

            if self.search_params.dynamic_score_utility_factor != 0.0
                || self.search_params.subtree_value_bias_factor != 0.0
                || self.pattern_bonus_table.is_some()
            {
                unsafe {
                    self.recursively_recompute_stats(&mut *root_ptr);
                }
                if any_filtered {
                    // Recursive stats recomputation resulted in us marking all
                    // nodes we have, mirroring upstream. Anything filtered is
                    // old now, delete it.
                    self.delete_all_old_or_all_new_table_nodes_and_subtree_value_bias_multithreaded(
                        true,
                    );
                }
            } else if any_filtered {
                // Sweep over the tree marking the kept subtree as good (calling
                // a NULL function), then delete anything unmarked — exactly the
                // upstream branch. Marking is NOT optional: deleting by age
                // without first bumping the kept subtree's age would free nodes
                // still referenced by root child pointers (search_node_age was
                // incremented at the end of the previous begin_search, so the
                // retained tree is "old" until re-marked here).
                {
                    // SAFETY: the root node outlives this call (it is owned by
                    // self.root_node and only freed via clear_search / the
                    // delete-old sweep below, which runs strictly after).
                    let new_root_ptr = self.root_node.as_deref().unwrap() as *const SearchNode;
                    let new_root_ref = unsafe { &*new_root_ptr };
                    self.apply_recursively_any_order_multithreaded(
                        std::slice::from_ref(&new_root_ref),
                        &|_node, _idx| {},
                    );
                }
                self.delete_all_old_or_all_new_table_nodes_and_subtree_value_bias_multithreaded(
                    true,
                );
            }
        }

        if self.search_params.subtree_value_bias_factor != 0.0
            && self.subtree_value_bias_table.is_some()
        {
            self.subtree_value_bias_table
                .as_ref()
                .unwrap()
                .clear_unused_synchronous();
        }

        self.search_node_age = self.search_node_age.wrapping_add(1);
    }

    /// Run a single playout from the root, returning whether it counted toward
    /// the playout total.
    ///
    /// `root_ptr` must point at the root node owned by `self.root_node`, which
    /// is stable for the duration of a search. It is passed in (rather than
    /// derived from `&self`) so that the playout path can be called through a
    /// shared `&Search` reference from multiple worker threads; the tree nodes
    /// are accessed exclusively through interior mutability (atomics, spinlock
    /// stats, CAS state machine) as in the C++ original.
    pub fn run_single_playout(
        &self,
        thread: &mut SearchThread,
        upper_bound_visits_left: f64,
        root_ptr: *mut SearchNode,
    ) -> bool {
        thread.upper_bound_visits_left = upper_bound_visits_left;
        thread.should_count_playout = true;

        let _finished = unsafe { self.playout_descend(thread, &mut *root_ptr, true) };

        thread.pla = self.root_pla;
        thread.board = self.root_board.clone();
        thread.history = self.root_history.clone();
        thread.graph_hash = self.root_graph_hash;
        thread.graph_path.clear();

        thread.should_count_playout
    }
}

// Accumulators used by tree ownership traversal.
trait OwnershipAccumulator {
    fn accumulate(&mut self, ownership: &[f32], self_prop: f64);
}

struct SumOwnership<'a> {
    vec: &'a mut Vec<f64>,
}

impl OwnershipAccumulator for SumOwnership<'_> {
    fn accumulate(&mut self, ownership: &[f32], self_prop: f64) {
        for (i, &v) in ownership.iter().enumerate() {
            self.vec[i] += self_prop * v as f64;
        }
    }
}

struct SumOwnershipAndSq<'a> {
    avg: &'a mut Vec<f64>,
    sq: &'a mut Vec<f64>,
}

impl OwnershipAccumulator for SumOwnershipAndSq<'_> {
    fn accumulate(&mut self, ownership: &[f32], self_prop: f64) {
        for (i, &v) in ownership.iter().enumerate() {
            let value = v as f64;
            self.avg[i] += self_prop * value;
            self.sq[i] += self_prop * value * value;
        }
    }
}

// Search results and tree inspection
impl<'a> Search<'a> {
    pub fn get_chosen_move_loc(&mut self) -> Loc {
        let root = match self.root_node.as_deref() {
            Some(r) => r,
            None => return NULL_LOC,
        };

        let mut locs = Vec::new();
        let mut play_selection_values = Vec::new();
        if !self.get_play_selection_values(&mut locs, &mut play_selection_values, 0.0) {
            return NULL_LOC;
        }
        assert_eq!(locs.len(), play_selection_values.len());

        let temperature = self.interpolate_early(
            self.search_params.chosen_move_temperature_halflife,
            self.search_params.chosen_move_temperature_early,
            self.search_params.chosen_move_temperature,
        );

        let idx = Self::choose_index_with_temperature(
            &mut self.non_search_rand,
            &play_selection_values,
            temperature,
            self.search_params.chosen_move_temperature_only_below_prob,
            None,
        ) as usize;
        locs[idx]
    }

    pub fn get_play_selection_values(
        &self,
        locs: &mut Vec<Loc>,
        play_selection_values: &mut Vec<f64>,
        scale_max_to_at_least: f64,
    ) -> bool {
        self.get_play_selection_values_with_counts(
            locs,
            play_selection_values,
            None,
            scale_max_to_at_least,
        )
    }

    pub fn get_play_selection_values_with_counts(
        &self,
        locs: &mut Vec<Loc>,
        play_selection_values: &mut Vec<f64>,
        mut ret_visit_counts: Option<&mut Vec<f64>>,
        scale_max_to_at_least: f64,
    ) -> bool {
        locs.clear();
        play_selection_values.clear();
        if let Some(ref mut v) = ret_visit_counts {
            v.clear();
        }
        let root = match self.root_node.as_deref() {
            Some(r) => r,
            None => return false,
        };
        self.get_play_selection_values_for_node(
            root,
            locs,
            play_selection_values,
            ret_visit_counts,
            scale_max_to_at_least,
            true,
        )
    }

    pub fn get_play_selection_values_for_node(
        &self,
        node: &SearchNode,
        locs: &mut Vec<Loc>,
        play_selection_values: &mut Vec<f64>,
        ret_visit_counts: Option<&mut Vec<f64>>,
        scale_max_to_at_least: f64,
        allow_direct_policy_moves: bool,
    ) -> bool {
        let mut lcb_buf = vec![0.0f64; nn_pos::MAX_NN_POLICY_SIZE];
        let mut radius_buf = vec![0.0f64; nn_pos::MAX_NN_POLICY_SIZE];
        self.get_play_selection_values_internal(
            node,
            locs,
            play_selection_values,
            ret_visit_counts,
            scale_max_to_at_least,
            allow_direct_policy_moves,
            false,
            false,
            &mut lcb_buf,
            &mut radius_buf,
        )
    }

    fn get_play_selection_values_for_node_no_lcb(
        &self,
        node: &SearchNode,
        locs: &mut Vec<Loc>,
        play_selection_values: &mut Vec<f64>,
        scale_max_to_at_least: f64,
    ) -> bool {
        let mut lcb_buf = vec![0.0f64; nn_pos::MAX_NN_POLICY_SIZE];
        let mut radius_buf = vec![0.0f64; nn_pos::MAX_NN_POLICY_SIZE];
        self.get_play_selection_values_internal(
            node,
            locs,
            play_selection_values,
            None,
            scale_max_to_at_least,
            false,
            false,
            true,
            &mut lcb_buf,
            &mut radius_buf,
        )
    }

    pub fn get_root_values(&self, values: &mut ReportedSearchValues) -> bool {
        *values = ReportedSearchValues::new();
        match self.root_node.as_deref() {
            Some(root) => self.get_node_values(root, values),
            None => false,
        }
    }

    pub fn get_root_values_require_success(&self) -> ReportedSearchValues {
        if self.root_node.is_none() {
            panic!("Bug? Bot search root was null");
        }
        let mut values = ReportedSearchValues::new();
        if !self.get_root_values(&mut values) {
            panic!("Bug? Bot search returned no root values");
        }
        values
    }

    pub fn get_node_values(&self, node: &SearchNode, values: &mut ReportedSearchValues) -> bool {
        *values = ReportedSearchValues::new();
        let visits = node.stats.visits.load(Ordering::Acquire);
        let weight_sum = node.stats.weight_sum.load(Ordering::Acquire);
        if weight_sum <= 0.0 || visits < 0 {
            return false;
        }

        let stats = ReportedSearchStats {
            win_loss_value_avg: node.stats.win_loss_value_avg.load(Ordering::Acquire),
            no_result_value_avg: node.stats.no_result_value_avg.load(Ordering::Acquire),
            score_mean_avg: node.stats.score_mean_avg.load(Ordering::Acquire),
            score_mean_sq_avg: node.stats.score_mean_sq_avg.load(Ordering::Acquire),
            lead_avg: node.stats.lead_avg.load(Ordering::Acquire),
            utility_avg: node.stats.utility_avg.load(Ordering::Acquire),
            total_weight: weight_sum,
            total_visits: visits,
        };
        let ctx = ReportedSearchContext {
            sqrt_board_area: self.root_board.sqrt_board_area(),
            recent_score_center: self.recent_score_center,
            dynamic_score_center_scale: self.search_params.dynamic_score_center_scale,
        };
        *values = ReportedSearchValues::from_stats(&stats, &ctx);
        true
    }

    pub fn get_pruned_root_values(&self, values: &mut ReportedSearchValues) -> bool {
        *values = ReportedSearchValues::new();
        match self.root_node.as_deref() {
            Some(root) => self.get_pruned_node_values(root, values),
            None => false,
        }
    }

    pub fn get_pruned_node_values(
        &self,
        node: &SearchNode,
        values: &mut ReportedSearchValues,
    ) -> bool {
        *values = ReportedSearchValues::new();

        let mut locs = Vec::new();
        let mut play_selection_values = Vec::new();
        if !self.get_play_selection_values_for_node_no_lcb(
            node,
            &mut locs,
            &mut play_selection_values,
            1.0,
        ) {
            return self.get_node_values(node, values);
        }

        let children = node.get_children();
        let children_capacity = children.get_capacity();

        let mut win_loss_value_sum = 0.0;
        let mut no_result_value_sum = 0.0;
        let mut score_mean_sum = 0.0;
        let mut score_mean_sq_sum = 0.0;
        let mut lead_sum = 0.0;
        let mut utility_sum = 0.0;
        let mut weight_sum = 0.0;

        for i in 0..children_capacity {
            let child_ptr = children.get(i);
            let child = match child_ptr.get_if_allocated() {
                Some(c) => c,
                None => break,
            };
            let edge_visits = child_ptr.get_edge_visits();
            let stats = NodeStats::from_atomic(&child.stats);
            if stats.visits <= 0 || stats.weight_sum <= 0.0 || edge_visits <= 0 {
                continue;
            }
            let weight = play_selection_values.get(i).copied().unwrap_or(0.0);
            win_loss_value_sum += weight * stats.win_loss_value_avg;
            no_result_value_sum += weight * stats.no_result_value_avg;
            score_mean_sum += weight * stats.score_mean_avg;
            score_mean_sq_sum += weight * stats.score_mean_sq_avg;
            lead_sum += weight * stats.lead_avg;
            utility_sum += weight * stats.utility_avg;
            weight_sum += weight;
        }

        // Also add in the direct evaluation of this node.
        let nn_output = match node.get_nn_output() {
            Some(o) => o,
            None => return false,
        };
        let win_prob = nn_output.white_win_prob as f64;
        let loss_prob = nn_output.white_loss_prob as f64;
        let no_result_prob = nn_output.white_no_result_prob as f64;
        let score_mean = nn_output.white_score_mean as f64;
        let score_mean_sq = nn_output.white_score_mean_sq as f64;
        let lead = nn_output.white_lead as f64;
        let utility = self.get_result_utility(win_prob - loss_prob, no_result_prob)
            + self.get_score_utility(score_mean, score_mean_sq);
        let weight = self.compute_weight_from_nn_output(Some(&nn_output));
        win_loss_value_sum += (win_prob - loss_prob) * weight;
        no_result_value_sum += no_result_prob * weight;
        score_mean_sum += score_mean * weight;
        score_mean_sq_sum += score_mean_sq * weight;
        lead_sum += lead * weight;
        utility_sum += utility * weight;
        weight_sum += weight;

        if weight_sum <= 0.0 {
            return false;
        }

        let stats = ReportedSearchStats {
            win_loss_value_avg: win_loss_value_sum / weight_sum,
            no_result_value_avg: no_result_value_sum / weight_sum,
            score_mean_avg: score_mean_sum / weight_sum,
            score_mean_sq_avg: score_mean_sq_sum / weight_sum,
            lead_avg: lead_sum / weight_sum,
            utility_avg: utility_sum / weight_sum,
            total_weight: node.stats.weight_sum.load(Ordering::Acquire),
            total_visits: node.stats.visits.load(Ordering::Acquire),
        };
        let ctx = ReportedSearchContext {
            sqrt_board_area: self.root_board.sqrt_board_area(),
            recent_score_center: self.recent_score_center,
            dynamic_score_center_scale: self.search_params.dynamic_score_center_scale,
        };
        *values = ReportedSearchValues::from_stats(&stats, &ctx);
        true
    }

    pub fn get_root_node(&self) -> Option<&SearchNode> {
        self.root_node.as_deref()
    }

    pub fn get_child_for_move<'b>(
        &self,
        node: &'b SearchNode,
        move_loc: Loc,
    ) -> Option<&'b SearchNode> {
        let children = node.get_children();
        let count = children.iterate_and_count_children();
        for i in 0..count {
            let child = children.get(i);
            if child.get_move_loc() == move_loc {
                // Safety: the child pointer lives inside `node`, whose lifetime is `'b`.
                return unsafe {
                    std::mem::transmute::<Option<&SearchNode>, Option<&'b SearchNode>>(
                        child.get_if_allocated(),
                    )
                };
            }
        }
        None
    }

    pub fn get_root_raw_nn_values(&self, values: &mut ReportedSearchValues) -> bool {
        *values = ReportedSearchValues::new();
        match self.root_node.as_deref() {
            Some(root) => self.get_node_raw_nn_values(root, values),
            None => false,
        }
    }

    pub fn get_root_raw_nn_values_require_success(&self) -> ReportedSearchValues {
        if self.root_node.is_none() {
            panic!("Bug? Bot search root was null");
        }
        let mut values = ReportedSearchValues::new();
        if !self.get_root_raw_nn_values(&mut values) {
            panic!("Bug? Bot search returned no root values");
        }
        values
    }

    pub fn get_node_raw_nn_values(
        &self,
        node: &SearchNode,
        values: &mut ReportedSearchValues,
    ) -> bool {
        *values = ReportedSearchValues::new();
        let nn_output = match node.get_nn_output() {
            Some(o) => o,
            None => return false,
        };

        values.win_value = nn_output.white_win_prob as f64;
        values.loss_value = nn_output.white_loss_prob as f64;
        values.no_result_value = nn_output.white_no_result_prob as f64;

        let score_mean = nn_output.white_score_mean as f64;
        let score_mean_sq = nn_output.white_score_mean_sq as f64;
        let score_stdev = score_value::get_score_stdev(score_mean, score_mean_sq);
        let sqrt_board_area = self.root_board.sqrt_board_area();
        values.static_score_value = score_value::expected_white_score_value(
            score_mean,
            score_stdev,
            0.0,
            2.0,
            sqrt_board_area,
        );
        values.dynamic_score_value = score_value::expected_white_score_value(
            score_mean,
            score_stdev,
            self.recent_score_center,
            self.search_params.dynamic_score_center_scale,
            sqrt_board_area,
        );
        values.expected_score = score_mean;
        values.expected_score_stdev = score_stdev;
        values.lead = nn_output.white_lead as f64;

        debug_assert!(values.win_value >= 0.0);
        debug_assert!(values.loss_value >= 0.0);
        debug_assert!(values.no_result_value >= 0.0);
        debug_assert!(values.win_value + values.loss_value + values.no_result_value < 1.001);

        let mut win_loss_value = values.win_value - values.loss_value;
        win_loss_value = win_loss_value.clamp(-1.0, 1.0);
        values.win_loss_value = win_loss_value;

        values.weight = self.compute_weight_from_nn_output(Some(&nn_output));
        values.visits = 1;

        true
    }

    pub fn get_root_visits(&self) -> i64 {
        self.root_node
            .as_ref()
            .map(|n| n.stats.visits.load(Ordering::Acquire))
            .unwrap_or(0)
    }

    pub fn get_policy(&self, policy_probs: &mut [f32]) -> bool {
        self.get_policy_for_node(self.root_node.as_deref(), policy_probs)
    }

    pub fn get_policy_for_node(&self, node: Option<&SearchNode>, policy_probs: &mut [f32]) -> bool {
        if let Some(n) = node {
            if let Some(output) = n.get_nn_output() {
                let len = policy_probs.len().min(output.policy_probs.len());
                policy_probs[..len].copy_from_slice(&output.policy_probs[..len]);
                return true;
            }
        }
        false
    }

    pub fn get_policy_surprise_and_entropy(
        &self,
        surprise_ret: &mut f64,
        search_entropy_ret: &mut f64,
        policy_entropy_ret: &mut f64,
    ) -> bool {
        match self.root_node.as_deref() {
            Some(root) => self.get_policy_surprise_and_entropy_for_node(
                surprise_ret,
                search_entropy_ret,
                policy_entropy_ret,
                root,
            ),
            None => {
                *surprise_ret = 0.0;
                *search_entropy_ret = 0.0;
                *policy_entropy_ret = 0.0;
                false
            }
        }
    }

    pub fn get_policy_surprise_and_entropy_for_node(
        &self,
        surprise_ret: &mut f64,
        search_entropy_ret: &mut f64,
        policy_entropy_ret: &mut f64,
        node: &SearchNode,
    ) -> bool {
        *surprise_ret = 0.0;
        *search_entropy_ret = 0.0;
        *policy_entropy_ret = 0.0;

        let nn_output = match node.get_nn_output() {
            Some(o) => o,
            None => return false,
        };

        let mut locs = Vec::new();
        let mut play_selection_values = Vec::new();
        if !self.get_play_selection_values_for_node(
            node,
            &mut locs,
            &mut play_selection_values,
            None,
            1.0,
            true,
        ) {
            return false;
        }

        let policy_probs = nn_output.get_policy_probs_maybe_noised();

        let sum_play_selection_values: f64 = play_selection_values.iter().sum();
        if sum_play_selection_values <= 0.0 {
            return false;
        }

        let mut surprise = 0.0;
        let mut search_entropy = 0.0;
        for i in 0..play_selection_values.len() {
            let pos = self.get_pos(locs[i]) as usize;
            let policy = (policy_probs.get(pos).copied().unwrap_or(0.0f32) as f64).max(1e-100);
            let target = play_selection_values[i] / sum_play_selection_values;
            if target > 1e-100 {
                let log_target = target.ln();
                let log_policy = policy.ln();
                surprise += target * (log_target - log_policy);
                search_entropy += -target * log_target;
            }
        }

        let mut policy_entropy = 0.0;
        for pos in 0..nn_pos::MAX_NN_POLICY_SIZE {
            let policy = policy_probs.get(pos).copied().unwrap_or(0.0f32) as f64;
            if policy > 1e-100 {
                policy_entropy += -policy * policy.ln();
            }
        }

        if surprise < 0.0 {
            surprise = 0.0;
        }
        if search_entropy < 0.0 {
            search_entropy = 0.0;
        }
        if policy_entropy < 0.0 {
            policy_entropy = 0.0;
        }

        *surprise_ret = surprise;
        *search_entropy_ret = search_entropy;
        *policy_entropy_ret = policy_entropy;
        true
    }

    pub fn get_policy_surprise(&self) -> f64 {
        let mut surprise = 0.0;
        let mut search_entropy = 0.0;
        let mut policy_entropy = 0.0;
        if self.get_policy_surprise_and_entropy(
            &mut surprise,
            &mut search_entropy,
            &mut policy_entropy,
        ) {
            surprise
        } else {
            0.0
        }
    }

    pub fn print_pv(&self, out: &mut String, node: Option<&SearchNode>, max_depth: i32) {
        out.clear();
        let mut buf = Vec::new();
        let mut visits_buf = Vec::new();
        let mut edge_visits_buf = Vec::new();
        let mut scratch_locs = Vec::new();
        let mut scratch_values = Vec::new();
        self.append_pv(
            &mut buf,
            &mut visits_buf,
            &mut edge_visits_buf,
            &mut scratch_locs,
            &mut scratch_values,
            node,
            max_depth,
        );
        self.format_pv(out, &buf);
    }

    pub fn print_pv_for_move(
        &self,
        out: &mut String,
        node: &SearchNode,
        move_loc: Loc,
        max_depth: i32,
    ) {
        out.clear();
        let mut buf = Vec::new();
        let mut visits_buf = Vec::new();
        let mut edge_visits_buf = Vec::new();
        let mut scratch_locs = Vec::new();
        let mut scratch_values = Vec::new();
        self.append_pv_for_move(
            &mut buf,
            &mut visits_buf,
            &mut edge_visits_buf,
            &mut scratch_locs,
            &mut scratch_values,
            node,
            move_loc,
            max_depth,
        );
        self.format_pv(out, &buf);
    }

    fn format_pv(&self, out: &mut String, buf: &[Loc]) {
        let mut printed_anything = false;
        for &loc in buf {
            if loc == NULL_LOC {
                continue;
            }
            if printed_anything {
                out.push(' ');
            }
            out.push_str(&location::to_string(
                loc,
                self.root_board.x_size,
                self.root_board.y_size,
            ));
            printed_anything = true;
        }
    }

    pub fn print_tree(
        &self,
        out: &mut String,
        node: Option<&SearchNode>,
        options: &PrintTreeOptions,
        perspective: Player,
    ) {
        out.clear();
        let node = match node {
            Some(n) => n,
            None => return,
        };

        let mut scratch_locs: Vec<Loc> = Vec::new();
        let mut scratch_values: Vec<f64> = Vec::new();
        let edge_visits = node.stats.visits.load(Ordering::Acquire);
        let mut data = self.get_analysis_data_of_single_child(
            Some(node),
            edge_visits,
            &mut scratch_locs,
            &mut scratch_values,
            NULL_LOC,
            f64::NAN,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            options.max_pv_depth,
        );
        data.weight_factor = f64::NAN;

        let perspective_to_use = if perspective != P_BLACK && perspective != P_WHITE {
            node.next_pla
        } else {
            perspective
        };

        let mut prefix = String::new();
        self.print_tree_helper(
            out,
            Some(node),
            options,
            &mut prefix,
            0,
            0,
            &data,
            perspective_to_use,
        );
    }

    fn print_tree_helper(
        &self,
        out: &mut String,
        node: Option<&SearchNode>,
        options: &PrintTreeOptions,
        prefix: &mut String,
        orig_visits: i64,
        depth: i32,
        data: &AnalysisData,
        perspective: Player,
    ) {
        let node = match node {
            Some(n) => n,
            None => return,
        };

        let perspective_to_use = if perspective != P_BLACK && perspective != P_WHITE {
            node.next_pla
        } else {
            perspective
        };
        let perspective_factor = if perspective_to_use == P_BLACK {
            -1.0
        } else {
            1.0
        };

        let mut orig_visits = orig_visits;
        if depth == 0 {
            orig_visits = data.num_visits;
        }

        // Output for this node.
        {
            out.push_str(prefix);
            out.push_str(": ");

            if data.child_visits > 0 {
                out.push_str(&format!(
                    "T {:6.2}c ",
                    perspective_factor * data.utility * 100.0
                ));
                out.push_str(&format!(
                    "W {:6.2}c ",
                    perspective_factor * data.result_utility * 100.0
                ));
                out.push_str(&format!(
                    "S {:6.2}c ({:+5.1} L {:+5.1}) ",
                    perspective_factor * data.score_utility * 100.0,
                    perspective_factor * data.score_mean,
                    perspective_factor * data.lead
                ));
            }

            if depth > 0 && !data.lcb.is_nan() {
                out.push_str(&format!(
                    "LCB {:7.2}c ",
                    perspective_factor * data.lcb * 100.0
                ));
            }

            if !data.policy_prior.is_nan() {
                out.push_str(&format!("P {:5.2}% ", data.policy_prior * 100.0));
            }
            if !data.weight_factor.is_nan() {
                out.push_str(&format!("WF {:5.1} ", data.weight_factor));
            }
            if data.play_selection_value >= 0.0 && depth > 0 {
                out.push_str(&format!("PSV {:7.0} ", data.play_selection_value));
            }

            if options.print_sqs {
                out.push_str(&format!(
                    "SMSQ {:5.1} USQ {:7.5} W {:6.2} WSQ {:8.2} ",
                    data.score_mean_sq_avg,
                    data.utility_sq_avg,
                    data.weight_sum,
                    data.weight_sq_sum
                ));
            }

            if options.print_avg_shortterm_error {
                let (wl_error, score_error) =
                    self.get_shallow_average_shortterm_wl_and_score_error(Some(node));
                out.push_str(&format!(
                    "STWL {:6.2}c STS {:5.1} ",
                    wl_error * 100.0,
                    score_error
                ));
            }

            out.push_str(&format!("N {:7}  --  ", data.child_visits));
            self.format_pv(out, &data.pv);
            out.push('\n');
        }

        let branch_len = options.branch.len() as i32;
        if depth >= branch_len {
            if depth >= options.max_depth + branch_len {
                return;
            }
            if data.num_visits < options.min_visits_to_expand {
                return;
            }
            if (data.num_visits as f64) < orig_visits as f64 * options.min_visits_prop_to_expand {
                return;
            }
        }
        if (options.also_branch && depth == 0) || (!options.also_branch && depth == branch_len) {
            let arrow = if node.next_pla == perspective_to_use {
                "^"
            } else {
                "v"
            };
            out.push_str(&format!(
                "---{}({})---\n",
                player_io::player_to_string(node.next_pla),
                arrow
            ));
        }

        let mut analysis_data = Vec::new();
        self.get_analysis_data_for_node(
            node,
            &mut analysis_data,
            0,
            true,
            options.max_pv_depth,
            false,
        );

        let num_children = analysis_data.len();

        // Find the last child that meets the visit filter, so that children after
        // an included one are still shown (in case the filter is more complex than visits).
        let mut last_idx_with_enough_visits: i32 = num_children as i32 - 1;
        loop {
            if last_idx_with_enough_visits <= 0 {
                break;
            }
            let child_visits = analysis_data[last_idx_with_enough_visits as usize].num_visits;
            let has_enough_visits = child_visits >= options.min_visits_to_show
                && child_visits as f64 >= orig_visits as f64 * options.min_visits_prop_to_show;
            if has_enough_visits {
                break;
            }
            last_idx_with_enough_visits -= 1;
        }

        let mut num_children_to_recurse_on = num_children as i32;
        if options.max_children_to_show < num_children_to_recurse_on {
            num_children_to_recurse_on = options.max_children_to_show;
        }
        if last_idx_with_enough_visits + 1 < num_children_to_recurse_on {
            num_children_to_recurse_on = last_idx_with_enough_visits + 1;
        }

        for (i, child_data) in analysis_data.iter().enumerate() {
            let i = i as i32;
            let move_loc = child_data.move_loc;

            let should_recurse = (depth >= branch_len && i < num_children_to_recurse_on)
                || (depth < branch_len && move_loc == options.branch[depth as usize])
                || (depth < branch_len && options.also_branch && i < num_children_to_recurse_on);

            if should_recurse {
                let child = self.get_child_by_move_loc(node, move_loc);

                let old_len = prefix.len();
                let loc_str =
                    location::to_string(move_loc, self.root_board.x_size, self.root_board.y_size);
                if loc_str == "pass" {
                    prefix.push_str("pss");
                } else {
                    prefix.push_str(&loc_str);
                }
                prefix.push(' ');
                while prefix.len() < old_len + 4 {
                    prefix.push(' ');
                }

                let mut next_depth = depth + 1;
                if depth < branch_len && move_loc != options.branch[depth as usize] {
                    next_depth = branch_len + 1;
                }

                self.print_tree_helper(
                    out,
                    child,
                    options,
                    prefix,
                    orig_visits,
                    next_depth,
                    child_data,
                    perspective_to_use,
                );
                prefix.truncate(old_len);
            }
        }
    }

    fn get_child_by_move_loc<'b>(
        &self,
        node: &'b SearchNode,
        move_loc: Loc,
    ) -> Option<&'b SearchNode> {
        let children = node.get_children();
        let mut child_ptr: Option<*const SearchNode> = None;
        for i in 0..children.get_capacity() {
            let child_pointer = children.get(i);
            if let Some(child) = child_pointer.get_if_allocated() {
                if child_pointer.get_move_loc_relaxed() == move_loc {
                    child_ptr = Some(child);
                    break;
                }
            } else {
                break;
            }
        }
        child_ptr.map(|p| unsafe { &*p })
    }

    pub fn print_root_policy_map(&self, out: &mut String) {
        out.clear();
        let root = match self.root_node.as_deref() {
            Some(r) => r,
            None => return,
        };
        let nn_output = match root.get_nn_output() {
            Some(o) => o,
            None => return,
        };
        let policy_probs = nn_output.get_policy_probs_maybe_noised();
        for y in 0..self.root_board.y_size {
            for x in 0..self.root_board.x_size {
                let pos = nn_pos::xy_to_pos(x, y, nn_output.nn_x_len) as usize;
                out.push_str(&format!("{:6.1} ", policy_probs[pos] * 100.0));
            }
            out.push('\n');
        }
        out.push('\n');
    }

    pub fn print_root_ownership_map(&self, out: &mut String, perspective: Player) {
        out.clear();
        let root = match self.root_node.as_deref() {
            Some(r) => r,
            None => return,
        };
        let nn_output = match root.get_nn_output() {
            Some(o) => o,
            None => return,
        };
        let owner_map = match nn_output.white_owner_map.as_ref() {
            Some(m) => m,
            None => return,
        };

        let perspective_to_use = if perspective != P_BLACK && perspective != P_WHITE {
            self.root_pla
        } else {
            perspective
        };
        let perspective_factor = if perspective_to_use == P_BLACK {
            -1.0
        } else {
            1.0
        };

        for y in 0..self.root_board.y_size {
            for x in 0..self.root_board.x_size {
                let pos = nn_pos::xy_to_pos(x, y, nn_output.nn_x_len) as usize;
                out.push_str(&format!(
                    "{:6.1} ",
                    perspective_factor * owner_map[pos] * 100.0
                ));
            }
            out.push('\n');
        }
        out.push('\n');
    }

    pub fn print_root_ending_score_value_bonus(&self, out: &mut String) {
        out.clear();
        let root = match self.root_node.as_deref() {
            Some(r) => r,
            None => return,
        };
        let nn_output = match root.get_nn_output() {
            Some(o) => o,
            None => return,
        };
        if nn_output.white_owner_map.is_none() {
            return;
        }

        let children = root.get_children();
        let children_capacity = children.get_capacity();
        for i in 0..children_capacity {
            let child_pointer = children.get(i);
            let child = match child_pointer.get_if_allocated() {
                Some(c) => c,
                None => break,
            };
            let edge_visits = child_pointer.get_edge_visits();
            let move_loc = child_pointer.get_move_loc_relaxed();
            let child_visits = child.stats.visits.load(Ordering::Acquire);
            let score_mean_avg = child.stats.score_mean_avg.load(Ordering::Acquire);
            let score_mean_sq_avg = child.stats.score_mean_sq_avg.load(Ordering::Acquire);
            let utility_avg = child.stats.utility_avg.load(Ordering::Acquire);

            let utility_no_bonus = utility_avg;
            let ending_score_bonus = self.get_ending_white_score_bonus(root, move_loc);
            let utility_diff =
                self.get_score_utility_diff(score_mean_avg, score_mean_sq_avg, ending_score_bonus);
            let utility_with_bonus = utility_no_bonus + utility_diff;

            out.push_str(&location::to_string(
                move_loc,
                self.root_board.x_size,
                self.root_board.y_size,
            ));
            out.push(' ');
            out.push_str(&format!(
                "visits {} edgeVisits {} utilityNoBonus {:.2}c utilityWithBonus {:.2}c endingScoreBonus {:.2}\n",
                child_visits,
                edge_visits,
                utility_no_bonus * 100.0,
                utility_with_bonus * 100.0,
                ending_score_bonus
            ));
        }
    }

    pub fn get_analysis_data(
        &self,
        buf: &mut Vec<AnalysisData>,
        min_moves_to_try_to_get: i32,
        include_weight_factors: bool,
        max_pv_depth: i32,
        duplicate_for_symmetries: bool,
    ) {
        buf.clear();
        if let Some(root) = self.root_node.as_deref() {
            self.get_analysis_data_for_node(
                root,
                buf,
                min_moves_to_try_to_get,
                include_weight_factors,
                max_pv_depth,
                duplicate_for_symmetries,
            );
        }
    }

    pub fn get_analysis_data_for_node(
        &self,
        node: &SearchNode,
        buf: &mut Vec<AnalysisData>,
        min_moves_to_try_to_get: i32,
        include_weight_factors: bool,
        max_pv_depth: i32,
        duplicate_for_symmetries: bool,
    ) {
        buf.clear();

        let mut scratch_locs: Vec<Loc> = Vec::new();
        let mut scratch_values: Vec<f64> = Vec::new();
        let mut lcb_buf = [0.0f64; nn_pos::MAX_NN_POLICY_SIZE];
        let mut radius_buf = [0.0f64; nn_pos::MAX_NN_POLICY_SIZE];
        let mut policy_probs = [0.0f32; nn_pos::MAX_NN_POLICY_SIZE];

        let num_children: usize;
        let mut children_edge_visits: Vec<i64>;
        let mut children_move_locs: Vec<Loc>;
        let play_selection_values: Vec<f64>;
        let policy_prob_mass_visited: f64;
        let parent_win_loss_value: f64;
        let parent_score_mean: f64;
        let parent_score_stdev: f64;
        let parent_lead: f64;
        let mut parent_utility = 0.0;
        let mut parent_weight_per_visit = 0.0;
        let mut parent_utility_stdev_factor = 0.0;
        let fpu_value: f64;

        {
            let children_ref = node.get_children();
            let children_capacity = children_ref.get_capacity();

            let mut children: Vec<&SearchNode> = Vec::new();
            children_edge_visits = Vec::new();
            children_move_locs = Vec::new();
            children.reserve((self.root_board.x_size * self.root_board.y_size + 1) as usize);
            children_edge_visits
                .reserve((self.root_board.x_size * self.root_board.y_size + 1) as usize);
            children_move_locs
                .reserve((self.root_board.x_size * self.root_board.y_size + 1) as usize);

            for i in 0..children_capacity {
                let child_pointer = children_ref.get(i);
                let child = match child_pointer.get_if_allocated() {
                    Some(c) => c,
                    None => break,
                };
                children.push(child);
                children_edge_visits.push(child_pointer.get_edge_visits());
                children_move_locs.push(child_pointer.get_move_loc_relaxed());
            }
            num_children = children.len();

            if num_children == 0 {
                return;
            }
            debug_assert!(num_children <= nn_pos::MAX_NN_POLICY_SIZE);

            let always_compute_lcb = true;
            let got_play_selection_values = self.get_play_selection_values_internal(
                node,
                &mut scratch_locs,
                &mut scratch_values,
                None,
                1.0,
                false,
                always_compute_lcb,
                false,
                &mut lcb_buf,
                &mut radius_buf,
            );

            if !got_play_selection_values {
                scratch_locs.clear();
                scratch_values.clear();
                for i in 0..num_children {
                    scratch_locs.push(children_move_locs[i]);
                    scratch_values.push(0.0);
                }
                let mut lcb_buf_value = 0.0;
                let mut radius_buf_value = 0.0;
                self.get_self_utility_lcb_and_radius_zero_visits(
                    &mut lcb_buf_value,
                    &mut radius_buf_value,
                );
                for i in 0..num_children {
                    lcb_buf[i] = lcb_buf_value;
                    radius_buf[i] = radius_buf_value;
                }
            }

            let nn_output = node
                .get_nn_output()
                .expect("get_analysis_data_for_node: node has no nn output");
            let policy_probs_from_nn = nn_output.get_policy_probs_maybe_noised();
            for i in 0..nn_pos::MAX_NN_POLICY_SIZE {
                policy_probs[i] = policy_probs_from_nn[i];
            }

            play_selection_values = scratch_values.clone();

            policy_prob_mass_visited = {
                let mut mass = 0.0;
                for i in 0..num_children {
                    let pos = self.get_pos(children_move_locs[i]) as usize;
                    mass += (policy_probs[pos] as f64).max(0.0);
                }
                debug_assert!(mass <= 1.0001);
                mass
            };

            {
                let weight_sum = node.stats.weight_sum.load(Ordering::Acquire);
                let win_loss_value_avg = node.stats.win_loss_value_avg.load(Ordering::Acquire);
                let score_mean_avg = node.stats.score_mean_avg.load(Ordering::Acquire);
                let score_mean_sq_avg = node.stats.score_mean_sq_avg.load(Ordering::Acquire);
                let lead_avg = node.stats.lead_avg.load(Ordering::Acquire);
                assert!(weight_sum > 0.0);

                parent_win_loss_value = win_loss_value_avg;
                parent_score_mean = score_mean_avg;
                parent_score_stdev =
                    score_value::get_score_stdev(parent_score_mean, score_mean_sq_avg);
                parent_lead = lead_avg;
            }

            fpu_value = self.get_fpu_value_for_children_assume_visited(
                node,
                node.next_pla,
                true,
                policy_prob_mass_visited,
                &mut parent_utility,
                &mut parent_weight_per_visit,
                &mut parent_utility_stdev_factor,
            );

            let mut stats_buf = vec![MoreNodeStats::new(); num_children];
            for i in 0..num_children {
                let child = children[i];
                let edge_visits = children_edge_visits[i];
                let move_loc = children_move_locs[i];
                let policy_prob = policy_probs[self.get_pos(move_loc) as usize] as f64;
                let mut data = self.get_analysis_data_of_single_child(
                    Some(child),
                    edge_visits,
                    &mut scratch_locs,
                    &mut scratch_values,
                    move_loc,
                    policy_prob,
                    fpu_value,
                    parent_utility,
                    parent_win_loss_value,
                    parent_score_mean,
                    parent_score_stdev,
                    parent_lead,
                    max_pv_depth,
                );
                data.play_selection_value = play_selection_values[i];
                data.lcb = if node.next_pla == P_BLACK {
                    -lcb_buf[i]
                } else {
                    lcb_buf[i]
                };
                data.radius = radius_buf[i];

                if include_weight_factors {
                    stats_buf[i].stats = NodeStats::from_atomic(&child.stats);
                    stats_buf[i].self_utility = if node.next_pla == P_WHITE {
                        data.utility
                    } else {
                        -data.utility
                    };
                    stats_buf[i].weight_adjusted = stats_buf[i].stats.get_child_weight(edge_visits);
                    stats_buf[i].prev_move_loc = move_loc;
                }

                buf.push(data);
            }

            if include_weight_factors {
                let mut total_child_weight = 0.0;
                for i in 0..num_children {
                    total_child_weight += stats_buf[i].weight_adjusted;
                }
                if self.search_params.use_noise_pruning {
                    let mut policy_probs_buf = [0.0f64; nn_pos::MAX_NN_POLICY_SIZE];
                    for i in 0..num_children {
                        let pos = self.get_pos(stats_buf[i].prev_move_loc) as usize;
                        policy_probs_buf[i] = (policy_probs[pos] as f64).max(1e-30);
                    }
                    total_child_weight = self.prune_noise_weight(
                        &mut stats_buf,
                        num_children as i32,
                        total_child_weight,
                        &policy_probs_buf,
                    );
                }
                let amount_to_subtract = 0.0;
                let amount_to_prune = 0.0;
                let mut stdevs_scratch = Vec::with_capacity(num_children);
                self.downweight_bad_children_and_normalize_weight(
                    num_children as i32,
                    total_child_weight,
                    total_child_weight,
                    amount_to_subtract,
                    amount_to_prune,
                    &mut stats_buf,
                    &mut stdevs_scratch,
                );
                for i in 0..num_children {
                    buf[i].weight_factor = stats_buf[i].weight_adjusted;
                }
            }
        }

        if (num_children as i32) < min_moves_to_try_to_get {
            for _ in 0..(min_moves_to_try_to_get - num_children as i32) {
                let mut best_pos = -1i32;
                let mut best_policy = -1.0f64;
                for pos in 0..nn_pos::MAX_NN_POLICY_SIZE {
                    let policy = policy_probs[pos] as f64;
                    if policy < best_policy {
                        continue;
                    }
                    let mut already_used = false;
                    for data in buf.iter() {
                        if self.get_pos(data.move_loc) == pos as i32 {
                            already_used = true;
                            break;
                        }
                    }
                    if already_used {
                        continue;
                    }
                    best_pos = pos as i32;
                    best_policy = policy;
                }
                if best_pos < 0 || best_policy < 0.0 {
                    break;
                }

                let best_move = nn_pos::pos_to_loc(
                    best_pos,
                    self.root_board.x_size,
                    self.root_board.y_size,
                    self.nn_x_len,
                    self.nn_y_len,
                );
                let data = self.get_analysis_data_of_single_child(
                    None,
                    0,
                    &mut scratch_locs,
                    &mut scratch_values,
                    best_move,
                    best_policy,
                    fpu_value,
                    parent_utility,
                    parent_win_loss_value,
                    parent_score_mean,
                    parent_score_stdev,
                    parent_lead,
                    max_pv_depth,
                );
                buf.push(data);
            }
        }

        buf.sort();

        if duplicate_for_symmetries
            && self.search_params.root_symmetry_pruning
            && self.root_symmetries.len() > 1
        {
            let mut new_buf = Vec::new();
            let mut is_done: HashSet<Loc> = HashSet::new();
            for data in buf.iter() {
                for &symmetry in self.root_symmetries.iter() {
                    let sym_move = symmetry::get_sym_loc(data.move_loc, &self.root_board, symmetry);
                    if is_done.contains(&sym_move) {
                        continue;
                    }
                    let avoid = if self.root_pla == P_BLACK {
                        &self.avoid_move_until_by_loc_black
                    } else {
                        &self.avoid_move_until_by_loc_white
                    };
                    if !avoid.is_empty()
                        && (sym_move as usize) < avoid.len()
                        && avoid[sym_move as usize] > 0
                    {
                        continue;
                    }

                    is_done.insert(sym_move);
                    new_buf.push(data.clone());
                    let new_data = new_buf.last_mut().unwrap();
                    new_data.move_loc = sym_move;
                    if symmetry != 0 {
                        new_data.is_symmetry_of = data.move_loc;
                    }
                    new_data.symmetry = symmetry;
                    for j in 0..new_data.pv.len() {
                        new_data.pv[j] =
                            symmetry::get_sym_loc(new_data.pv[j], &self.root_board, symmetry);
                    }
                }
            }
            *buf = new_buf;
        }

        for i in 0..buf.len() {
            buf[i].order = i as i32;
        }
    }

    pub fn append_pv(
        &self,
        buf: &mut Vec<Loc>,
        visits_buf: &mut Vec<i64>,
        edge_visits_buf: &mut Vec<i64>,
        scratch_locs: &mut Vec<Loc>,
        scratch_values: &mut Vec<f64>,
        node: Option<&SearchNode>,
        max_depth: i32,
    ) {
        let node = match node {
            Some(n) => n,
            None => {
                buf.clear();
                visits_buf.clear();
                edge_visits_buf.clear();
                scratch_locs.clear();
                scratch_values.clear();
                return;
            }
        };
        self.append_pv_for_move(
            buf,
            visits_buf,
            edge_visits_buf,
            scratch_locs,
            scratch_values,
            node,
            NULL_LOC,
            max_depth,
        );
    }

    pub fn append_pv_for_move(
        &self,
        buf: &mut Vec<Loc>,
        visits_buf: &mut Vec<i64>,
        edge_visits_buf: &mut Vec<i64>,
        scratch_locs: &mut Vec<Loc>,
        scratch_values: &mut Vec<f64>,
        node: &SearchNode,
        move_loc: Loc,
        max_depth: i32,
    ) {
        let target_move = move_loc;
        let mut node = node;

        for depth in 0..max_depth {
            scratch_locs.clear();
            scratch_values.clear();
            if !self.get_play_selection_values_for_node(
                node,
                scratch_locs,
                scratch_values,
                None,
                1.0,
                true,
            ) {
                return;
            }

            let mut best_idx: i32 = -1;
            let mut best_move_loc: Loc = NULL_LOC;
            let mut max_selection_value = Self::POLICY_ILLEGAL_SELECTION_VALUE;

            for (i, &loc) in scratch_locs.iter().enumerate() {
                let selection_value = scratch_values[i];
                if depth == 0 && loc == target_move {
                    best_idx = i as i32;
                    best_move_loc = loc;
                    break;
                }
                if selection_value > max_selection_value {
                    max_selection_value = selection_value;
                    best_idx = i as i32;
                    best_move_loc = loc;
                }
            }

            if best_idx < 0 || best_move_loc == NULL_LOC {
                return;
            }
            if depth == 0 && target_move != NULL_LOC && best_move_loc != target_move {
                return;
            }

            let children = node.get_children();
            let children_capacity = children.get_capacity();

            // Direct policy move (not a real child).
            if best_idx as usize >= children_capacity {
                buf.push(best_move_loc);
                visits_buf.push(0);
                edge_visits_buf.push(0);
                return;
            }

            let child_pointer = children.get(best_idx as usize);
            let child_raw = child_pointer.get_raw_ptr();
            let edge_visits = child_pointer.get_edge_visits();
            drop(children);

            if child_raw.is_null() {
                buf.push(best_move_loc);
                visits_buf.push(0);
                edge_visits_buf.push(0);
                return;
            }
            let child = unsafe { &*child_raw };

            let visits = child.stats.visits.load(Ordering::Acquire);

            buf.push(best_move_loc);
            visits_buf.push(visits);
            edge_visits_buf.push(edge_visits);

            node = child;
        }
    }

    pub fn get_average_tree_ownership(&self, node: Option<&SearchNode>) -> Vec<f64> {
        let node = node.or_else(|| self.root_node.as_deref());
        let node = match node {
            Some(n) => n,
            None => return Vec::new(),
        };
        if !self.always_include_owner_map {
            panic!("Called get_average_tree_ownership when always_include_owner_map is false");
        }
        let mut vec = vec![0.0f64; (self.nn_x_len * self.nn_y_len) as usize];
        let visits = node.stats.visits.load(Ordering::Acquire);
        let min_prop = 0.5 / (visits as f64).max(1.0).powf(0.75);
        let prune_prop = min_prop * 0.01;
        let mut graph_path = HashSet::new();
        let mut accumulator = SumOwnership { vec: &mut vec };
        self.traverse_tree_for_ownership(
            min_prop,
            prune_prop,
            1.0,
            Some(node),
            &mut graph_path,
            &mut accumulator,
        );
        vec
    }

    pub fn get_average_tree_ownership_with_sym(
        &self,
        perspective: Player,
        node: Option<&SearchNode>,
        symmetry: i32,
    ) -> Vec<f64> {
        let ownership = self.get_average_tree_ownership(node);
        let board = &self.root_board;
        let mut output = vec![0.0f64; (board.y_size * board.x_size) as usize];
        for y in 0..board.y_size {
            for x in 0..board.x_size {
                let pos = nn_pos::xy_to_pos(x, y, self.nn_x_len) as usize;
                let (sx, sy) = symmetry::get_sym_loc_xy(x, y, board.x_size, board.y_size, symmetry);
                let sym_pos = (sy * board.x_size + sx) as usize;
                let o = if perspective == P_BLACK
                    || (perspective != P_BLACK
                        && perspective != P_WHITE
                        && self.root_pla == P_BLACK)
                {
                    -ownership[pos]
                } else {
                    ownership[pos]
                };
                output[sym_pos] = kata_core::global::round_static(o, 1_000_000.0);
            }
        }
        output
    }

    pub fn get_average_and_std_dev_tree_ownership(
        &self,
        node: Option<&SearchNode>,
    ) -> (Vec<f64>, Vec<f64>) {
        let node = node.or_else(|| self.root_node.as_deref());
        let node = match node {
            Some(n) => n,
            None => return (Vec::new(), Vec::new()),
        };
        let mut average = vec![0.0f64; (self.nn_x_len * self.nn_y_len) as usize];
        let mut stdev = vec![0.0f64; (self.nn_x_len * self.nn_y_len) as usize];
        let visits = node.stats.visits.load(Ordering::Acquire);
        let min_prop = 0.5 / (visits as f64).max(1.0).powf(0.75);
        let prune_prop = min_prop * 0.01;
        let mut graph_path = HashSet::new();
        let mut accumulator = SumOwnershipAndSq {
            avg: &mut average,
            sq: &mut stdev,
        };
        self.traverse_tree_for_ownership(
            min_prop,
            prune_prop,
            1.0,
            Some(node),
            &mut graph_path,
            &mut accumulator,
        );
        for i in 0..average.len() {
            let avg = average[i];
            stdev[i] = (stdev[i] - avg * avg).max(0.0).sqrt();
        }
        (average, stdev)
    }

    pub fn get_average_and_std_dev_tree_ownership_with_sym(
        &self,
        perspective: Player,
        node: Option<&SearchNode>,
        symmetry: i32,
    ) -> (Vec<f64>, Vec<f64>) {
        let (average, stdev) = self.get_average_and_std_dev_tree_ownership(node);
        let board = &self.root_board;
        let mut avg_output = vec![0.0f64; (board.y_size * board.x_size) as usize];
        let mut stdev_output = vec![0.0f64; (board.y_size * board.x_size) as usize];
        for y in 0..board.y_size {
            for x in 0..board.x_size {
                let pos = nn_pos::xy_to_pos(x, y, self.nn_x_len) as usize;
                let (sx, sy) = symmetry::get_sym_loc_xy(x, y, board.x_size, board.y_size, symmetry);
                let sym_pos = (sy * board.x_size + sx) as usize;
                let o = if perspective == P_BLACK
                    || (perspective != P_BLACK
                        && perspective != P_WHITE
                        && self.root_pla == P_BLACK)
                {
                    -average[pos]
                } else {
                    average[pos]
                };
                avg_output[sym_pos] = kata_core::global::round_static(o, 1_000_000.0);
                stdev_output[sym_pos] = kata_core::global::round_static(stdev[pos], 1_000_000.0);
            }
        }
        (avg_output, stdev_output)
    }

    pub fn get_shallow_average_shortterm_wl_and_score_error(
        &self,
        node: Option<&SearchNode>,
    ) -> (f64, f64) {
        let node = match node {
            Some(n) => n,
            None => match self.root_node.as_deref() {
                Some(r) => r,
                None => return (0.0, 0.0),
            },
        };

        let supports_shortterm_error = self
            .nn_evaluator
            .map_or(false, |eval| eval.supports_shortterm_error());
        if !supports_shortterm_error {
            return (-1.0, -1.0);
        }

        let mut graph_path = HashSet::new();
        let mut policy_probs_buf = [0.0f64; nn_pos::MAX_NN_POLICY_SIZE];
        let mut wl_error = 0.0;
        let mut score_error = 0.0;

        let visits = node.stats.visits.load(Ordering::Acquire);
        let min_prop = 0.25 / (visits.max(1) as f64).powf(0.625);
        let desired_prop = 1.0;

        self.get_shallow_average_shortterm_wl_and_score_error_helper(
            node,
            &mut graph_path,
            &mut policy_probs_buf,
            min_prop,
            desired_prop,
            &mut wl_error,
            &mut score_error,
        );

        (wl_error, score_error)
    }

    pub fn get_sharp_score(&self, node: &SearchNode, ret: &mut f64) -> bool {
        let visits = node.stats.visits.load(Ordering::Acquire);
        let min_prop = 0.25 / (visits.max(1) as f64).powf(0.5);
        let desired_prop = 1.0;
        *ret = 0.0;

        let mut graph_path = HashSet::new();
        let mut policy_probs_buf = [0.0f64; nn_pos::MAX_NN_POLICY_SIZE];

        let is_root = self
            .root_node
            .as_deref()
            .map_or(false, |root| std::ptr::eq(node, root));
        if !is_root {
            return self.get_sharp_score_helper(
                node,
                &mut graph_path,
                &mut policy_probs_buf,
                min_prop,
                desired_prop,
                ret,
            );
        }

        let nn_output = match node.get_nn_output() {
            Some(o) => o,
            None => return false,
        };

        let mut locs = Vec::new();
        let mut play_selection_values = Vec::new();
        let mut lcb_buf = vec![0.0f64; nn_pos::MAX_NN_POLICY_SIZE];
        let mut radius_buf = vec![0.0f64; nn_pos::MAX_NN_POLICY_SIZE];
        let suc = self.get_play_selection_values_internal(
            node,
            &mut locs,
            &mut play_selection_values,
            None,
            1.0,
            false,
            false,
            true,
            &mut lcb_buf,
            &mut radius_buf,
        );

        if !suc {
            let mut values = ReportedSearchValues::new();
            if self.get_node_values(node, &mut values) {
                *ret = values.expected_score;
                return true;
            }
            return false;
        }

        let num_children = play_selection_values.len();
        let children = node.get_children();

        let mut relative_children_weight_sum = 0.0;
        let mut children_weight_sum = 0.0;
        for i in 0..num_children {
            let child_weight = play_selection_values[i];
            relative_children_weight_sum += child_weight * child_weight * child_weight;
            children_weight_sum += child_weight;
        }

        let mut parent_nn_weight = self.compute_weight_from_nn_output(Some(&*nn_output));
        parent_nn_weight = parent_nn_weight.max(1e-10);
        let desired_prop_from_children =
            desired_prop * children_weight_sum / (children_weight_sum + parent_nn_weight);
        let mut self_prop =
            desired_prop * parent_nn_weight / (children_weight_sum + parent_nn_weight);

        if desired_prop_from_children <= 0.0 || relative_children_weight_sum <= 0.0 {
            self_prop += desired_prop_from_children;
        } else {
            graph_path.insert(node as *const SearchNode);

            for i in 0..num_children {
                let child = children.get(i).get_if_allocated().unwrap();
                let child_weight = play_selection_values[i];
                let desired_prop_from_child = child_weight * child_weight * child_weight
                    / relative_children_weight_sum
                    * desired_prop_from_children;
                let accumulated = self.get_sharp_score_helper(
                    child,
                    &mut graph_path,
                    &mut policy_probs_buf,
                    min_prop,
                    desired_prop_from_child,
                    ret,
                );
                if !accumulated {
                    self_prop += desired_prop_from_child;
                }
            }

            graph_path.remove(&(node as *const SearchNode));
        }

        let score_mean = nn_output.white_score_mean as f64;
        *ret += score_mean * self_prop;
        true
    }

    pub fn get_analysis_json(
        &self,
        perspective: Player,
        analysis_pv_len: i32,
        prevent_encore: bool,
        include_policy: bool,
        include_ownership: bool,
        include_ownership_stdev: bool,
        include_moves_ownership: bool,
        include_moves_ownership_stdev: bool,
        include_pv_visits: bool,
        include_no_result_value: bool,
        ret: &mut Value,
    ) -> bool {
        const MIN_MOVES: i32 = 0;
        const OUTPUT_PRECISION: i32 = 8;

        let mut buf = Vec::new();
        let duplicate_for_symmetries = true;
        self.get_analysis_data(
            &mut buf,
            MIN_MOVES,
            false,
            analysis_pv_len,
            duplicate_for_symmetries,
        );

        let board = &self.root_board;
        let hist = &self.root_history;
        let root_node_opt = self.root_node.as_deref();
        let nn_output = root_node_opt.and_then(|n| n.get_nn_output());
        let human_output = root_node_opt.and_then(|n| n.get_human_output());

        let flip = perspective == P_BLACK
            || (perspective != P_BLACK && perspective != P_WHITE && self.root_pla == P_BLACK);

        let json_f64 = |x: f64| -> Value {
            serde_json::Number::from_f64(x)
                .map(Value::Number)
                .unwrap_or(Value::Null)
        };

        let mut move_infos = Vec::new();
        for data in &buf {
            let mut winrate = 0.5 * (1.0 + data.win_loss_value);
            let mut utility = data.utility;
            let radius_scale_hack_factor = self.search_params.win_loss_utility_factor
                / (self.search_params.win_loss_utility_factor
                    + self.search_params.static_score_utility_factor
                    + self.search_params.dynamic_score_utility_factor
                    + 1.0e-20)
                * 0.5;
            let mut lcb = if self.root_pla == P_WHITE {
                winrate - data.radius * radius_scale_hack_factor
            } else {
                winrate + data.radius * radius_scale_hack_factor
            };
            let mut utility_lcb = data.lcb;
            let mut score_mean = data.score_mean;
            let mut lead = data.lead;
            if flip {
                winrate = 1.0 - winrate;
                lcb = 1.0 - lcb;
                utility = -utility;
                score_mean = -score_mean;
                lead = -lead;
                utility_lcb = -utility_lcb;
            }

            let mut move_info = serde_json::Map::new();
            move_info.insert(
                "move".to_string(),
                Value::String(location::to_string(
                    data.move_loc,
                    board.x_size,
                    board.y_size,
                )),
            );
            move_info.insert(
                "visits".to_string(),
                Value::Number(data.child_visits.into()),
            );
            move_info.insert(
                "weight".to_string(),
                json_f64(kata_core::global::round_dynamic(
                    data.child_weight_sum,
                    OUTPUT_PRECISION,
                )),
            );
            move_info.insert(
                "utility".to_string(),
                json_f64(kata_core::global::round_dynamic(utility, OUTPUT_PRECISION)),
            );
            move_info.insert(
                "winrate".to_string(),
                json_f64(kata_core::global::round_dynamic(winrate, OUTPUT_PRECISION)),
            );
            move_info.insert(
                "scoreMean".to_string(),
                json_f64(kata_core::global::round_dynamic(lead, OUTPUT_PRECISION)),
            );
            move_info.insert(
                "scoreSelfplay".to_string(),
                json_f64(kata_core::global::round_dynamic(
                    score_mean,
                    OUTPUT_PRECISION,
                )),
            );
            move_info.insert(
                "scoreLead".to_string(),
                json_f64(kata_core::global::round_dynamic(lead, OUTPUT_PRECISION)),
            );
            move_info.insert(
                "scoreStdev".to_string(),
                json_f64(kata_core::global::round_dynamic(
                    data.score_stdev,
                    OUTPUT_PRECISION,
                )),
            );
            if include_no_result_value {
                move_info.insert(
                    "noResultValue".to_string(),
                    json_f64(kata_core::global::round_dynamic(
                        data.no_result_value,
                        OUTPUT_PRECISION,
                    )),
                );
            }
            move_info.insert(
                "prior".to_string(),
                json_f64(kata_core::global::round_dynamic(
                    data.policy_prior,
                    OUTPUT_PRECISION,
                )),
            );
            if let Some(ref human) = human_output {
                let human_prior = human
                    .get_policy_probs_maybe_noised()
                    .get(self.get_pos(data.move_loc) as usize)
                    .copied()
                    .unwrap_or(0.0f32) as f64;
                move_info.insert(
                    "humanPrior".to_string(),
                    json_f64(kata_core::global::round_dynamic(
                        human_prior.max(0.0),
                        OUTPUT_PRECISION,
                    )),
                );
            }
            move_info.insert(
                "lcb".to_string(),
                json_f64(kata_core::global::round_dynamic(lcb, OUTPUT_PRECISION)),
            );
            move_info.insert(
                "utilityLcb".to_string(),
                json_f64(kata_core::global::round_dynamic(
                    utility_lcb,
                    OUTPUT_PRECISION,
                )),
            );
            move_info.insert(
                "order".to_string(),
                Value::Number((data.order as i64).into()),
            );
            if data.is_symmetry_of != NULL_LOC {
                move_info.insert(
                    "isSymmetryOf".to_string(),
                    Value::String(location::to_string(
                        data.is_symmetry_of,
                        board.x_size,
                        board.y_size,
                    )),
                );
            }
            move_info.insert(
                "edgeVisits".to_string(),
                Value::Number(data.num_visits.into()),
            );
            move_info.insert(
                "edgeWeight".to_string(),
                json_f64(kata_core::global::round_dynamic(
                    data.weight_sum,
                    OUTPUT_PRECISION,
                )),
            );
            move_info.insert(
                "playSelectionValue".to_string(),
                json_f64(kata_core::global::round_dynamic(
                    data.play_selection_value,
                    OUTPUT_PRECISION,
                )),
            );

            let pv_len = if prevent_encore && data.pv_contains_pass() {
                data.get_pv_len_up_to_phase_end(board, hist, self.root_pla)
            } else {
                data.pv.len()
            };
            let pv: Vec<Value> = data.pv[..pv_len]
                .iter()
                .map(|&loc| Value::String(location::to_string(loc, board.x_size, board.y_size)))
                .collect();
            move_info.insert("pv".to_string(), Value::Array(pv));

            if include_pv_visits {
                let pv_visits: Vec<Value> = data.pv_visits[..pv_len]
                    .iter()
                    .map(|&v| Value::Number(v.into()))
                    .collect();
                move_info.insert("pvVisits".to_string(), Value::Array(pv_visits));
                let pv_edge_visits: Vec<Value> = data.pv_edge_visits[..pv_len]
                    .iter()
                    .map(|&v| Value::Number(v.into()))
                    .collect();
                move_info.insert("pvEdgeVisits".to_string(), Value::Array(pv_edge_visits));
            }

            if include_moves_ownership && include_moves_ownership_stdev {
                if !data.node.is_null() {
                    let (avg, stdev) = self.get_average_and_std_dev_tree_ownership_with_sym(
                        perspective,
                        Some(unsafe { &*data.node }),
                        data.symmetry,
                    );
                    move_info.insert(
                        "ownership".to_string(),
                        Value::Array(avg.into_iter().map(json_f64).collect()),
                    );
                    move_info.insert(
                        "ownershipStdev".to_string(),
                        Value::Array(stdev.into_iter().map(json_f64).collect()),
                    );
                }
            } else if include_moves_ownership_stdev {
                if !data.node.is_null() {
                    let (_, stdev) = self.get_average_and_std_dev_tree_ownership_with_sym(
                        perspective,
                        Some(unsafe { &*data.node }),
                        data.symmetry,
                    );
                    move_info.insert(
                        "ownershipStdev".to_string(),
                        Value::Array(stdev.into_iter().map(json_f64).collect()),
                    );
                }
            } else if include_moves_ownership {
                if !data.node.is_null() {
                    let avg = self.get_average_tree_ownership_with_sym(
                        perspective,
                        Some(unsafe { &*data.node }),
                        data.symmetry,
                    );
                    move_info.insert(
                        "ownership".to_string(),
                        Value::Array(avg.into_iter().map(json_f64).collect()),
                    );
                }
            }

            move_infos.push(Value::Object(move_info));
        }

        let mut root_info = serde_json::Map::new();
        {
            let mut root_vals = ReportedSearchValues::new();
            if !self.get_pruned_root_values(&mut root_vals) {
                return false;
            }
            let winloss = root_vals.win_loss_value;
            let score_mean = root_vals.expected_score;
            let lead = root_vals.lead;
            let utility = root_vals.utility;
            let flip_factor = if flip { -1.0 } else { 1.0 };

            root_info.insert("visits".to_string(), Value::Number(root_vals.visits.into()));
            root_info.insert(
                "weight".to_string(),
                json_f64(kata_core::global::round_dynamic(
                    root_vals.weight,
                    OUTPUT_PRECISION,
                )),
            );
            root_info.insert(
                "winrate".to_string(),
                json_f64(kata_core::global::round_dynamic(
                    0.5 + 0.5 * winloss * flip_factor,
                    OUTPUT_PRECISION,
                )),
            );
            root_info.insert(
                "scoreSelfplay".to_string(),
                json_f64(kata_core::global::round_dynamic(
                    score_mean * flip_factor,
                    OUTPUT_PRECISION,
                )),
            );
            root_info.insert(
                "scoreLead".to_string(),
                json_f64(kata_core::global::round_dynamic(
                    lead * flip_factor,
                    OUTPUT_PRECISION,
                )),
            );
            root_info.insert(
                "scoreStdev".to_string(),
                json_f64(kata_core::global::round_dynamic(
                    root_vals.expected_score_stdev,
                    OUTPUT_PRECISION,
                )),
            );
            root_info.insert(
                "utility".to_string(),
                json_f64(kata_core::global::round_dynamic(
                    utility * flip_factor,
                    OUTPUT_PRECISION,
                )),
            );

            if let Some(ref nn) = nn_output {
                let white_win_prob = nn.white_win_prob as f64;
                let white_loss_prob = nn.white_loss_prob as f64;
                let white_score_mean = nn.white_score_mean as f64;
                root_info.insert(
                    "rawWinrate".to_string(),
                    json_f64(kata_core::global::round_dynamic(
                        0.5 + 0.5 * (white_win_prob - white_loss_prob) * flip_factor,
                        OUTPUT_PRECISION,
                    )),
                );
                root_info.insert(
                    "rawLead".to_string(),
                    json_f64(kata_core::global::round_dynamic(
                        nn.white_lead as f64 * flip_factor,
                        OUTPUT_PRECISION,
                    )),
                );
                root_info.insert(
                    "rawScoreSelfplay".to_string(),
                    json_f64(kata_core::global::round_dynamic(
                        white_score_mean * flip_factor,
                        OUTPUT_PRECISION,
                    )),
                );
                let raw_stdev = (nn.white_score_mean_sq as f64
                    - white_score_mean * white_score_mean)
                    .max(0.0)
                    .sqrt();
                root_info.insert(
                    "rawScoreSelfplayStdev".to_string(),
                    json_f64(kata_core::global::round_dynamic(
                        raw_stdev,
                        OUTPUT_PRECISION,
                    )),
                );
                root_info.insert(
                    "rawNoResultProb".to_string(),
                    json_f64(kata_core::global::round_dynamic(
                        nn.white_no_result_prob as f64,
                        OUTPUT_PRECISION,
                    )),
                );
                root_info.insert(
                    "rawStWrError".to_string(),
                    json_f64(kata_core::global::round_dynamic(
                        nn.shortterm_winloss_error as f64 * 0.5,
                        OUTPUT_PRECISION,
                    )),
                );
                root_info.insert(
                    "rawStScoreError".to_string(),
                    json_f64(kata_core::global::round_dynamic(
                        nn.shortterm_score_error as f64,
                        OUTPUT_PRECISION,
                    )),
                );
                root_info.insert(
                    "rawVarTimeLeft".to_string(),
                    json_f64(kata_core::global::round_dynamic(
                        nn.var_time_left as f64,
                        OUTPUT_PRECISION,
                    )),
                );
            }
            if let Some(ref human) = human_output {
                let white_win_prob = human.white_win_prob as f64;
                let white_loss_prob = human.white_loss_prob as f64;
                let white_score_mean = human.white_score_mean as f64;
                root_info.insert(
                    "humanWinrate".to_string(),
                    json_f64(kata_core::global::round_dynamic(
                        0.5 + 0.5 * (white_win_prob - white_loss_prob) * flip_factor,
                        OUTPUT_PRECISION,
                    )),
                );
                root_info.insert(
                    "humanScoreMean".to_string(),
                    json_f64(kata_core::global::round_dynamic(
                        white_score_mean * flip_factor,
                        OUTPUT_PRECISION,
                    )),
                );
                let human_stdev = (human.white_score_mean_sq as f64
                    - white_score_mean * white_score_mean)
                    .max(0.0)
                    .sqrt();
                root_info.insert(
                    "humanScoreStdev".to_string(),
                    json_f64(kata_core::global::round_dynamic(
                        human_stdev,
                        OUTPUT_PRECISION,
                    )),
                );
                root_info.insert(
                    "humanStWrError".to_string(),
                    json_f64(kata_core::global::round_dynamic(
                        human.shortterm_winloss_error as f64 * 0.5,
                        OUTPUT_PRECISION,
                    )),
                );
                root_info.insert(
                    "humanStScoreError".to_string(),
                    json_f64(kata_core::global::round_dynamic(
                        human.shortterm_score_error as f64,
                        OUTPUT_PRECISION,
                    )),
                );
            }

            let mut this_hash = Hash128::default();
            let mut sym_hash = Hash128::default();
            for symmetry in 0..symmetry::NUM_SYMMETRIES {
                let sym_board = symmetry::get_sym_board(board, symmetry);
                let hash = sym_board.get_sit_hash_with_simple_ko(self.root_pla);
                if symmetry == 0 {
                    this_hash = hash;
                    sym_hash = hash;
                } else if hash < sym_hash {
                    sym_hash = hash;
                }
            }
            root_info.insert(
                "thisHash".to_string(),
                Value::String(
                    kata_core::global::uint64_to_hex_string(this_hash.hash1)
                        + &kata_core::global::uint64_to_hex_string(this_hash.hash0),
                ),
            );
            root_info.insert(
                "symHash".to_string(),
                Value::String(
                    kata_core::global::uint64_to_hex_string(sym_hash.hash1)
                        + &kata_core::global::uint64_to_hex_string(sym_hash.hash0),
                ),
            );
            root_info.insert(
                "currentPlayer".to_string(),
                Value::String(player_io::player_to_string_short(self.root_pla).to_string()),
            );
        }

        let mut result = serde_json::Map::new();
        result.insert("moveInfos".to_string(), Value::Array(move_infos));
        result.insert("rootInfo".to_string(), Value::Object(root_info));

        if include_policy {
            let mut policy_probs = [0.0f32; nn_pos::MAX_NN_POLICY_SIZE];
            if !self.get_policy(&mut policy_probs) {
                return false;
            }
            let mut policy = Vec::new();
            for y in 0..board.y_size {
                for x in 0..board.x_size {
                    let pos = nn_pos::xy_to_pos(x, y, self.nn_x_len) as usize;
                    policy.push(json_f64(kata_core::global::round_dynamic(
                        policy_probs[pos] as f64,
                        OUTPUT_PRECISION,
                    )));
                }
            }
            let pass_pos =
                nn_pos::loc_to_pos(PASS_LOC, board.x_size, self.nn_x_len, self.nn_y_len) as usize;
            policy.push(json_f64(kata_core::global::round_dynamic(
                policy_probs[pass_pos] as f64,
                OUTPUT_PRECISION,
            )));
            result.insert("policy".to_string(), Value::Array(policy));

            if let Some(ref human) = human_output {
                let human_policy_probs = human.get_policy_probs_maybe_noised();
                let mut policy = Vec::new();
                for y in 0..board.y_size {
                    for x in 0..board.x_size {
                        let pos = nn_pos::xy_to_pos(x, y, self.nn_x_len) as usize;
                        policy.push(json_f64(kata_core::global::round_dynamic(
                            human_policy_probs[pos] as f64,
                            OUTPUT_PRECISION,
                        )));
                    }
                }
                let pass_pos =
                    nn_pos::loc_to_pos(PASS_LOC, board.x_size, self.nn_x_len, self.nn_y_len)
                        as usize;
                policy.push(json_f64(kata_core::global::round_dynamic(
                    human_policy_probs[pass_pos] as f64,
                    OUTPUT_PRECISION,
                )));
                result.insert("humanPolicy".to_string(), Value::Array(policy));
            }
        }

        if include_ownership && include_ownership_stdev {
            let (avg, stdev) =
                self.get_average_and_std_dev_tree_ownership_with_sym(perspective, root_node_opt, 0);
            result.insert(
                "ownership".to_string(),
                Value::Array(avg.into_iter().map(json_f64).collect()),
            );
            result.insert(
                "ownershipStdev".to_string(),
                Value::Array(stdev.into_iter().map(json_f64).collect()),
            );
        } else if include_ownership_stdev {
            let (_, stdev) =
                self.get_average_and_std_dev_tree_ownership_with_sym(perspective, root_node_opt, 0);
            result.insert(
                "ownershipStdev".to_string(),
                Value::Array(stdev.into_iter().map(json_f64).collect()),
            );
        } else if include_ownership {
            let avg = self.get_average_tree_ownership_with_sym(perspective, root_node_opt, 0);
            result.insert(
                "ownership".to_string(),
                Value::Array(avg.into_iter().map(json_f64).collect()),
            );
        }

        *ret = Value::Object(result);
        true
    }
}

// Dirichlet noise and temperature
impl<'a> Search<'a> {
    pub fn choose_index_with_temperature(
        rand: &mut Rand,
        relative_probs: &[f64],
        temperature: f64,
        only_below_prob: f64,
        processed_rel_probs_buf: Option<&mut [f64]>,
    ) -> u32 {
        let n = relative_probs.len();
        if n == 0 {
            return 0;
        }

        let mut processed = vec![0.0f64; n];

        let mut max_rel_prob = 0.0;
        let mut sum_rel_prob = 0.0;
        for &p in relative_probs.iter() {
            sum_rel_prob += p.max(0.0);
            if p > max_rel_prob {
                max_rel_prob = p;
            }
        }
        if max_rel_prob <= 0.0 || sum_rel_prob <= 0.0 {
            return 0;
        }

        // Temperature so close to 0 that we just calculate the max directly.
        if temperature <= 1.0e-4 && only_below_prob >= 1.0 {
            let mut best_idx = 0;
            let mut best_prob = relative_probs[0];
            processed[0] = 0.0;
            for i in 1..n {
                processed[i] = 0.0;
                if relative_probs[i] > best_prob {
                    best_prob = relative_probs[i];
                    best_idx = i;
                }
            }
            processed[best_idx] = 1.0;
            if let Some(buf) = processed_rel_probs_buf {
                if buf.len() >= n {
                    buf[..n].copy_from_slice(&processed);
                }
            }
            return best_idx as u32;
        }

        // Actual temperature.
        let log_max_rel_prob = max_rel_prob.ln();
        let log_sum_rel_prob = sum_rel_prob.ln();
        let log_only_below_prob = only_below_prob.max(1e-50).ln();
        let log_rel_prob_threshold =
            0.0f64.min(log_only_below_prob + log_sum_rel_prob - log_max_rel_prob);

        let mut sum = 0.0;
        for i in 0..n {
            let p = relative_probs[i];
            if p <= 0.0 {
                processed[i] = 0.0;
            } else {
                let log_rel_prob = p.ln() - log_max_rel_prob;
                let new_log_rel_prob = if log_rel_prob > log_rel_prob_threshold {
                    log_rel_prob
                } else {
                    (log_rel_prob - log_rel_prob_threshold) / temperature + log_rel_prob_threshold
                };
                processed[i] = new_log_rel_prob.exp();
            }
            sum += processed[i];
        }

        if let Some(buf) = processed_rel_probs_buf {
            if buf.len() >= n {
                buf[..n].copy_from_slice(&processed);
            }
        }

        if sum <= 0.0 {
            return 0;
        }

        let draw = rand.next_double_up_to(sum);
        let mut cumulative = 0.0;
        for (i, &p) in processed.iter().enumerate() {
            cumulative += p;
            if draw < cumulative {
                return i as u32;
            }
        }
        (n - 1) as u32
    }

    pub fn compute_dirichlet_alpha_distribution(
        policy_size: i32,
        policy_probs: &[f32],
        alpha_distr: &mut [f64],
    ) {
        let n = policy_size as usize;
        if alpha_distr.len() < n {
            return;
        }
        alpha_distr[..n].fill(0.0);

        let mut legal_count = 0;
        for i in 0..n {
            if policy_probs[i] >= 0.0 {
                legal_count += 1;
            }
        }
        if legal_count == 0 {
            return;
        }

        let mut log_policy_sum = 0.0;
        for i in 0..n {
            if policy_probs[i] >= 0.0 {
                alpha_distr[i] = (policy_probs[i].min(0.01f32) as f64 + 1e-20).ln();
                log_policy_sum += alpha_distr[i];
            }
        }
        let log_policy_mean = log_policy_sum / legal_count as f64;

        let mut alpha_prop_sum = 0.0;
        for i in 0..n {
            if policy_probs[i] >= 0.0 {
                alpha_distr[i] = (alpha_distr[i] - log_policy_mean).max(0.0);
                alpha_prop_sum += alpha_distr[i];
            }
        }

        let uniform_prob = 1.0 / legal_count as f64;
        if alpha_prop_sum <= 0.0 {
            for i in 0..n {
                if policy_probs[i] >= 0.0 {
                    alpha_distr[i] = uniform_prob;
                }
            }
        } else {
            for i in 0..n {
                if policy_probs[i] >= 0.0 {
                    alpha_distr[i] = 0.5 * (alpha_distr[i] / alpha_prop_sum + uniform_prob);
                }
            }
        }
    }

    pub fn add_dirichlet_noise(
        search_params: &SearchParams,
        rand: &mut Rand,
        policy_size: i32,
        policy_probs: &mut [f32],
    ) {
        let n = policy_size as usize;
        if n == 0 || policy_probs.len() < n {
            return;
        }

        let mut r = vec![0.0f64; n];
        Self::compute_dirichlet_alpha_distribution(policy_size, &policy_probs[..n], &mut r);

        let concentration = search_params.root_dirichlet_noise_total_concentration;
        let mut r_sum = 0.0;
        for i in 0..n {
            if policy_probs[i] >= 0.0 {
                r[i] = rand.next_gamma(r[i] * concentration);
                r_sum += r[i];
            } else {
                r[i] = 0.0;
            }
        }

        if r_sum <= 0.0 {
            return;
        }
        for x in r.iter_mut() {
            *x /= r_sum;
        }

        let weight = search_params.root_dirichlet_noise_weight;
        for i in 0..n {
            if policy_probs[i] >= 0.0 {
                policy_probs[i] = (r[i] * weight + policy_probs[i] as f64 * (1.0 - weight)) as f32;
            }
        }
    }

    fn maybe_add_policy_noise_and_temp(
        &self,
        thread: &mut SearchThread,
        is_root: bool,
        old_nn_output: &NNOutput,
    ) -> *mut Arc<NNOutput> {
        if !is_root {
            return std::ptr::null_mut();
        }
        if !self.search_params.root_noise_enabled
            && self.search_params.root_policy_temperature == 1.0
            && self.search_params.root_policy_temperature_early == 1.0
            && self.root_hint_loc == NULL_LOC
            && !self.avoid_move_until_rescale_root
        {
            return std::ptr::null_mut();
        }

        let n = self.policy_size as usize;

        let mut new_output = old_nn_output.clone();
        let mut noised = vec![0.0f32; nn_pos::MAX_NN_POLICY_SIZE].into_boxed_slice();
        noised[..n].copy_from_slice(&old_nn_output.policy_probs[..n]);
        new_output.noised_policy_probs = Some(noised);

        // Apply the optional root policy temperature.
        if self.search_params.root_policy_temperature != 1.0
            || self.search_params.root_policy_temperature_early != 1.0
        {
            let root_policy_temperature = self.interpolate_early(
                self.search_params.chosen_move_temperature_halflife,
                self.search_params.root_policy_temperature_early,
                self.search_params.root_policy_temperature,
            );

            let buf = new_output.noised_policy_probs.as_mut().unwrap();
            let mut max_value = 0.0f64;
            for i in 0..n {
                let p = buf[i] as f64;
                if p > max_value {
                    max_value = p;
                }
            }
            if max_value <= 0.0 {
                return std::ptr::null_mut();
            }

            let log_max_value = max_value.ln();
            let inv_temp = 1.0 / root_policy_temperature;
            let mut sum = 0.0;
            for i in 0..n {
                if buf[i] > 0.0 {
                    let p = ((buf[i] as f64).ln() - log_max_value) * inv_temp;
                    buf[i] = p.exp() as f32;
                    sum += buf[i] as f64;
                }
            }
            if sum > 0.0 {
                for i in 0..n {
                    if buf[i] >= 0.0 {
                        buf[i] = (buf[i] as f64 / sum) as f32;
                    }
                }
            }
        }

        if self.search_params.root_noise_enabled {
            let buf = new_output.noised_policy_probs.as_mut().unwrap();
            Self::add_dirichlet_noise(
                &self.search_params,
                &mut thread.rand,
                self.policy_size,
                &mut buf[..n],
            );
        }

        if self.avoid_move_until_rescale_root {
            let avoid = if self.root_pla == P_BLACK {
                &self.avoid_move_until_by_loc_black
            } else {
                &self.avoid_move_until_by_loc_white
            };
            if !avoid.is_empty() {
                let buf = new_output.noised_policy_probs.as_mut().unwrap();
                let mut policy_sum = 0.0;
                for loc in 0..MAX_ARR_SIZE as Loc {
                    if (self.root_board.is_on_board(loc) || loc == PASS_LOC)
                        && (loc as usize) < avoid.len()
                        && avoid[loc as usize] <= 0
                    {
                        let pos = self.get_pos(loc) as usize;
                        if pos < n && buf[pos] > 0.0 {
                            policy_sum += buf[pos] as f64;
                        }
                    }
                }
                if policy_sum > 0.0 {
                    for i in 0..n {
                        if buf[i] > 0.0 {
                            buf[i] = (buf[i] as f64 / policy_sum) as f32;
                        }
                    }
                }
            }
        }

        // Move a small amount of policy to the hint move.
        if self.root_hint_loc != NULL_LOC {
            let buf = new_output.noised_policy_probs.as_mut().unwrap();
            let pos = self.get_pos(self.root_hint_loc) as usize;
            if pos < n && buf[pos] >= 0.0 {
                const PROP_TO_MOVE: f64 = 0.02;
                let mut amount_to_move = 0.0;
                for i in 0..n {
                    if buf[i] >= 0.0 {
                        amount_to_move += buf[i] as f64 * PROP_TO_MOVE;
                        buf[i] = (buf[i] as f64 * (1.0 - PROP_TO_MOVE)) as f32;
                    }
                }
                buf[pos] += amount_to_move as f32;
            }
        }

        let new_arc = Arc::new(new_output);
        Box::into_raw(Box::new(new_arc))
    }
}

// Utility and scores
impl<'a> Search<'a> {
    pub fn get_result_utility(&self, win_loss_value: f64, no_result_value: f64) -> f64 {
        win_loss_value * self.search_params.win_loss_utility_factor
            + no_result_value * self.search_params.no_result_utility_for_white
    }

    pub fn get_result_utility_from_nn(&self, nn_output: &NNOutput) -> f64 {
        let win_loss_value = nn_output.white_win_prob as f64 - nn_output.white_loss_prob as f64;
        let no_result_value = nn_output.white_no_result_prob as f64;
        win_loss_value * self.search_params.win_loss_utility_factor
            + no_result_value * self.search_params.no_result_utility_for_white
    }

    pub fn get_score_utility(&self, score_mean_avg: f64, score_mean_sq_avg: f64) -> f64 {
        let score_stdev = score_value::get_score_stdev(score_mean_avg, score_mean_sq_avg);
        let sqrt_board_area = self.root_board.sqrt_board_area();
        let static_score_value = score_value::expected_white_score_value(
            score_mean_avg,
            score_stdev,
            0.0,
            2.0,
            sqrt_board_area,
        );
        let dynamic_score_value = score_value::expected_white_score_value(
            score_mean_avg,
            score_stdev,
            self.recent_score_center,
            self.search_params.dynamic_score_center_scale,
            sqrt_board_area,
        );
        static_score_value * self.search_params.static_score_utility_factor
            + dynamic_score_value * self.search_params.dynamic_score_utility_factor
    }

    pub fn get_score_utility_diff(
        &self,
        score_mean_avg: f64,
        score_mean_sq_avg: f64,
        delta: f64,
    ) -> f64 {
        let score_stdev = score_value::get_score_stdev(score_mean_avg, score_mean_sq_avg);
        let sqrt_board_area = self.root_board.sqrt_board_area();
        let static_diff = score_value::expected_white_score_value(
            score_mean_avg + delta,
            score_stdev,
            0.0,
            2.0,
            sqrt_board_area,
        ) - score_value::expected_white_score_value(
            score_mean_avg,
            score_stdev,
            0.0,
            2.0,
            sqrt_board_area,
        );
        let dynamic_diff = score_value::expected_white_score_value(
            score_mean_avg + delta,
            score_stdev,
            self.recent_score_center,
            self.search_params.dynamic_score_center_scale,
            sqrt_board_area,
        ) - score_value::expected_white_score_value(
            score_mean_avg,
            score_stdev,
            self.recent_score_center,
            self.search_params.dynamic_score_center_scale,
            sqrt_board_area,
        );
        static_diff * self.search_params.static_score_utility_factor
            + dynamic_diff * self.search_params.dynamic_score_utility_factor
    }

    pub fn get_approx_score_utility_derivative(&self, score_mean: f64) -> f64 {
        let sqrt_board_area = self.root_board.sqrt_board_area();

        fn derivative(score_mean: f64, center: f64, scale: f64, sqrt_board_area: f64) -> f64 {
            let adjusted = score_mean - center;
            let denom = scale * sqrt_board_area;
            let a = denom;
            (2.0 / std::f64::consts::PI) * a / (adjusted * adjusted + a * a)
        }

        let static_derivative = derivative(score_mean, 0.0, 2.0, sqrt_board_area);
        let dynamic_derivative = derivative(
            score_mean,
            self.recent_score_center,
            self.search_params.dynamic_score_center_scale,
            sqrt_board_area,
        );
        static_derivative * self.search_params.static_score_utility_factor
            + dynamic_derivative * self.search_params.dynamic_score_utility_factor
    }

    pub fn get_utility_from_nn(&self, nn_output: &NNOutput) -> f64 {
        self.get_result_utility_from_nn(nn_output)
            + self.get_score_utility(
                nn_output.white_score_mean as f64,
                nn_output.white_score_mean_sq as f64,
            )
    }
}

// Search biasing helpers
impl<'a> Search<'a> {
    pub fn is_allowed_root_move(&self, move_loc: Loc) -> bool {
        if move_loc == NULL_LOC {
            return false;
        }
        if move_loc != PASS_LOC && !self.root_board.is_on_board(move_loc) {
            return false;
        }
        if !self
            .root_history
            .is_legal(&self.root_board, move_loc, self.root_pla)
        {
            return false;
        }

        // Avoid-move-until vectors: a positive value forbids the move at the root.
        let avoid = if self.root_pla == P_BLACK {
            &self.avoid_move_until_by_loc_black
        } else {
            &self.avoid_move_until_by_loc_white
        };
        if (move_loc as usize) < avoid.len() && avoid[move_loc as usize] > 0 {
            return false;
        }

        // Pass-alive safe-area pruning after repeated opponent passes.
        if self.search_params.root_prune_useless_moves
            && !self.root_history.move_history.is_empty()
            && move_loc != PASS_LOC
        {
            let hist = &self.root_history.move_history;
            let last_idx = hist.len() - 1;
            let opp = get_opp(self.root_pla);
            if last_idx >= 6
                && hist[last_idx].loc == PASS_LOC
                && hist[last_idx - 2].loc == PASS_LOC
                && hist[last_idx - 4].loc == PASS_LOC
                && hist[last_idx - 6].loc == PASS_LOC
                && hist[last_idx].pla == opp
                && hist[last_idx - 2].pla == opp
                && hist[last_idx - 4].pla == opp
                && hist[last_idx - 6].pla == opp
            {
                if let Some(area) = &self.root_safe_area {
                    let c = area[move_loc as usize];
                    if c == opp || c == self.root_pla {
                        return false;
                    }
                }
            }
        }

        if self.search_params.root_symmetry_pruning
            && move_loc != PASS_LOC
            && self.root_sym_dup_loc[move_loc as usize]
        {
            return false;
        }

        true
    }

    pub fn get_pattern_bonus(&self, pattern_bonus_hash: Hash128, prev_move_pla: Player) -> f64 {
        if prev_move_pla != self.pla_that_search_is_for {
            return 0.0;
        }
        if let Some(table) = &self.pattern_bonus_table {
            table.get(pattern_bonus_hash).utility_bonus
        } else {
            0.0
        }
    }

    pub fn get_ending_white_score_bonus(&self, parent: &SearchNode, move_loc: Loc) -> f64 {
        if move_loc == NULL_LOC {
            return 0.0;
        }
        let is_root = self
            .root_node
            .as_deref()
            .map_or(false, |root| std::ptr::eq(parent, root));
        if !is_root {
            return 0.0;
        }

        // Borrow without an Arc refcount round trip: this runs once per root
        // child per playout. Safe because replaced outputs are deferred to
        // thread cleanup and only freed at the next begin_search, so the pointee
        // outlives this call regardless of concurrent swaps.
        let nn_output = match unsafe { parent.nn_output_ref() } {
            Some(output) => output,
            None => return 0.0,
        };
        let white_owner_map = match &nn_output.white_owner_map {
            Some(map) => map,
            None => return 0.0,
        };
        if nn_output.nn_x_len != self.nn_x_len || nn_output.nn_y_len != self.nn_y_len {
            return 0.0;
        }

        let is_area_ish = self.root_history.rules.scoring_rule == ScoringRule::Area
            || (self.root_history.rules.scoring_rule == ScoringRule::Territory
                && self.root_history.encore_phase >= 2);

        const EXTREME: f64 = 0.95;
        const TAIL: f64 = 0.05;

        // Extra points from the perspective of the root player.
        let mut extra_root_points = 0.0;
        if is_area_ish {
            if move_loc != PASS_LOC && self.root_board.ko_loc == NULL_LOC {
                let pos = nn_pos::loc_to_pos(
                    move_loc,
                    self.root_board.x_size,
                    self.nn_x_len,
                    self.nn_y_len,
                ) as usize;
                let white_ownership = white_owner_map[pos] as f64;
                let pla_ownership = if self.root_pla == P_WHITE {
                    white_ownership
                } else {
                    -white_ownership
                };
                if pla_ownership <= -EXTREME {
                    if !self.root_board.would_be_capture(move_loc, self.root_pla) {
                        extra_root_points -= self.search_params.root_ending_bonus_points
                            * ((-EXTREME - pla_ownership) / TAIL);
                    }
                } else if pla_ownership >= EXTREME {
                    if !self
                        .root_board
                        .is_adjacent_to_pla(move_loc, get_opp(self.root_pla))
                    {
                        // TODO: C++ also checks rootBoard.isNonPassAliveSelfConnection(...),
                        // which is not yet available in the Rust Board API. Treat as false.
                        let is_non_pass_alive_self_connection = false;
                        if !is_non_pass_alive_self_connection {
                            extra_root_points -= self.search_params.root_ending_bonus_points
                                * ((pla_ownership - EXTREME) / TAIL);
                        }
                    }
                }
            }
            if move_loc == PASS_LOC && self.root_history.has_button {
                extra_root_points -= self.search_params.root_ending_bonus_points * 0.5;
            }
        } else {
            if move_loc == PASS_LOC {
                extra_root_points -= self.search_params.root_ending_bonus_points * (2.0 / 3.0);
            } else if self.root_board.ko_loc == NULL_LOC {
                let pos = nn_pos::loc_to_pos(
                    move_loc,
                    self.root_board.x_size,
                    self.nn_x_len,
                    self.nn_y_len,
                ) as usize;
                let white_ownership = white_owner_map[pos] as f64;
                let pla_ownership = if self.root_pla == P_WHITE {
                    white_ownership
                } else {
                    -white_ownership
                };
                if pla_ownership <= -EXTREME {
                    extra_root_points -= self.search_params.root_ending_bonus_points
                        * ((-EXTREME - pla_ownership) / TAIL);
                } else if pla_ownership >= EXTREME {
                    if !self
                        .root_board
                        .is_adjacent_to_pla(move_loc, get_opp(self.root_pla))
                    {
                        // TODO: C++ also checks rootBoard.isNonPassAliveSelfConnection(...),
                        // which is not yet available in the Rust Board API. Treat as false.
                        let is_non_pass_alive_self_connection = false;
                        if !is_non_pass_alive_self_connection {
                            extra_root_points -= self.search_params.root_ending_bonus_points
                                * ((pla_ownership - EXTREME) / TAIL);
                        }
                    }
                }
            }
        }

        if self.root_pla == P_WHITE {
            extra_root_points
        } else {
            -extra_root_points
        }
    }

    pub fn should_suppress_pass(&self, node: Option<&SearchNode>) -> bool {
        if !self.search_params.fill_dame_before_pass {
            return false;
        }
        let n = match node {
            Some(n) => n,
            None => return false,
        };
        let is_root = self
            .root_node
            .as_deref()
            .map_or(false, |root| std::ptr::eq(n, root));
        if !is_root {
            return false;
        }
        if self.root_history.rules.scoring_rule != ScoringRule::Territory
            || self.root_history.encore_phase > 0
        {
            return false;
        }

        let nn_output = match n.get_nn_output() {
            Some(output) => output,
            None => return false,
        };
        let white_owner_map = match &nn_output.white_owner_map {
            Some(map) => map,
            None => return false,
        };
        if nn_output.nn_x_len != self.nn_x_len || nn_output.nn_y_len != self.nn_y_len {
            return false;
        }

        // Find the pass child.
        let children = n.get_children();
        let capacity = children.get_capacity();
        let mut pass_node: Option<&SearchNode> = None;
        let mut pass_edge_visits: i64 = 0;
        for i in 0..capacity {
            let child_ptr = children.get(i);
            let child = match child_ptr.get_if_allocated() {
                Some(child) => child,
                None => break,
            };
            if child_ptr.get_move_loc_relaxed() == PASS_LOC {
                pass_node = Some(child);
                pass_edge_visits = child_ptr.get_edge_visits();
                break;
            }
        }
        let pass_node = match pass_node {
            Some(node) => node,
            None => return false,
        };

        let pass_visits = pass_node.stats.visits.load(Ordering::Acquire);
        let pass_score_mean = pass_node.stats.score_mean_avg.load(Ordering::Acquire);
        let pass_lead = pass_node.stats.lead_avg.load(Ordering::Acquire);
        let pass_utility = pass_node.stats.utility_avg.load(Ordering::Acquire);
        let pass_weight = pass_node
            .stats
            .get_child_weight_with_child_visits(pass_edge_visits, pass_visits);
        if pass_visits <= 0 || pass_weight <= 1e-10 {
            return false;
        }

        const EXTREME: f64 = 0.95;

        for i in 0..capacity {
            let child_ptr = children.get(i);
            let child = match child_ptr.get_if_allocated() {
                Some(child) => child,
                None => break,
            };
            let move_loc = child_ptr.get_move_loc_relaxed();
            if move_loc == PASS_LOC {
                continue;
            }
            let pos = nn_pos::loc_to_pos(
                move_loc,
                self.root_board.x_size,
                self.nn_x_len,
                self.nn_y_len,
            ) as usize;
            let white_ownership = white_owner_map[pos] as f64;
            let pla_ownership = if self.root_pla == P_WHITE {
                white_ownership
            } else {
                -white_ownership
            };
            let opp_owned = pla_ownership < -EXTREME;
            let mut adj_to_pla_owned = false;
            for j in 0..4 {
                let adj = move_loc + self.root_board.adj_offsets()[j];
                if self.root_board.is_on_board(adj) {
                    let adj_pos = nn_pos::loc_to_pos(
                        adj,
                        self.root_board.x_size,
                        self.nn_x_len,
                        self.nn_y_len,
                    ) as usize;
                    let adj_white_ownership = white_owner_map[adj_pos] as f64;
                    let adj_pla_ownership = if self.root_pla == P_WHITE {
                        adj_white_ownership
                    } else {
                        -adj_white_ownership
                    };
                    if adj_pla_ownership > EXTREME {
                        adj_to_pla_owned = true;
                        break;
                    }
                }
            }
            if opp_owned && !adj_to_pla_owned {
                continue;
            }

            let edge_visits = child_ptr.get_edge_visits();
            let score_mean = child.stats.score_mean_avg.load(Ordering::Acquire);
            let lead = child.stats.lead_avg.load(Ordering::Acquire);
            let utility = child.stats.utility_avg.load(Ordering::Acquire);
            let child_weight = child.stats.get_child_weight(edge_visits);

            if (edge_visits <= 500 && child_weight <= 2.0 * pass_weight.sqrt())
                || child_weight <= 1e-10
            {
                continue;
            }

            if self.root_pla == P_WHITE
                && utility > pass_utility - 0.1
                && score_mean > pass_score_mean - 0.5
                && lead > pass_lead - 0.5
            {
                return true;
            }
            if self.root_pla == P_BLACK
                && utility < pass_utility + 0.1
                && score_mean < pass_score_mean + 0.5
                && lead < pass_lead + 0.5
            {
                return true;
            }
        }
        false
    }

    pub fn interpolate_early(&self, halflife: f64, early_value: f64, value: f64) -> f64 {
        let board_area = (self.root_board.x_size * self.root_board.y_size) as f64;
        let raw_halflives = self.root_history.get_current_turn_number() as f64 / halflife;
        let halflives = raw_halflives * 19.0 / board_area.sqrt();
        value + (early_value - value) * 0.5_f64.powf(halflives)
    }

    pub fn get_self_utility_lcb_and_radius(
        &self,
        parent: &SearchNode,
        child: Option<&SearchNode>,
        edge_visits: i64,
        move_loc: Loc,
        lcb_buf: &mut f64,
        radius_buf: &mut f64,
    ) {
        let utility_range_radius = self.search_params.win_loss_utility_factor
            + self.search_params.static_score_utility_factor
            + self.search_params.dynamic_score_utility_factor;
        *radius_buf = 2.0 * utility_range_radius * self.search_params.lcb_stdevs;
        *lcb_buf = -*radius_buf;

        let child = match child {
            Some(child) => child,
            None => return,
        };

        let child_visits = child.stats.visits.load(Ordering::Acquire);
        let score_mean_avg = child.stats.score_mean_avg.load(Ordering::Acquire);
        let score_mean_sq_avg = child.stats.score_mean_sq_avg.load(Ordering::Acquire);
        let utility_avg = child.stats.utility_avg.load(Ordering::Acquire);
        let mut utility_sq_avg = child.stats.utility_sq_avg.load(Ordering::Acquire);
        let mut weight_sum = child
            .stats
            .get_child_weight_with_child_visits(edge_visits, child_visits);
        let mut weight_sq_sum = child
            .stats
            .get_child_weight_sq_with_child_visits(edge_visits, child_visits);

        if child_visits <= 0 || weight_sum <= 0.0 || weight_sq_sum <= 0.0 {
            return;
        }

        // Effective sample size for weighted data.
        let mut ess = weight_sum * weight_sum / weight_sq_sum;

        // Add a prior with a small weight that the variance is the largest it can be,
        // to behave well at low playouts.
        let prior_weight = weight_sum / (ess * ess * ess);
        utility_sq_avg = utility_sq_avg.max(utility_avg * utility_avg + 1e-8);
        utility_sq_avg = (utility_sq_avg * weight_sum
            + (utility_sq_avg + utility_range_radius * utility_range_radius) * prior_weight)
            / (weight_sum + prior_weight);
        weight_sum += prior_weight;
        weight_sq_sum += prior_weight * prior_weight;

        // Recompute effective sample size now that we have the prior.
        ess = weight_sum * weight_sum / weight_sq_sum;

        let ending_score_bonus = self.get_ending_white_score_bonus(parent, move_loc);
        let utility_diff =
            self.get_score_utility_diff(score_mean_avg, score_mean_sq_avg, ending_score_bonus);
        let utility_with_bonus = utility_avg + utility_diff;
        let self_utility = if parent.next_pla == P_WHITE {
            utility_with_bonus
        } else {
            -utility_with_bonus
        };

        let utility_variance = utility_sq_avg - utility_avg * utility_avg;
        let estimate_stdev = (utility_variance / ess).sqrt();
        let radius = estimate_stdev * self.search_params.lcb_stdevs;

        *lcb_buf = self_utility - radius;
        *radius_buf = radius;
    }

    pub fn get_self_utility_lcb_and_radius_zero_visits(
        &self,
        lcb_buf: &mut f64,
        radius_buf: &mut f64,
    ) {
        let utility_range_radius = self.search_params.win_loss_utility_factor
            + self.search_params.static_score_utility_factor
            + self.search_params.dynamic_score_utility_factor;
        *radius_buf = 2.0 * utility_range_radius * self.search_params.lcb_stdevs;
        *lcb_buf = -*radius_buf;
    }
}

// Mirror handling
impl<'a> Search<'a> {
    /// Updates `mirroring_pla`, `mirror_advantage`, and `mirror_center_symmetry_error`
    /// based on whether the opponent appears to be mirroring our moves.
    fn update_mirroring(&mut self) {
        self.mirroring_pla = C_EMPTY;
        self.mirror_advantage = 0.0;
        self.mirror_center_symmetry_error = 1e10;

        if !self.search_params.anti_mirror {
            return;
        }

        let board = &self.root_board;
        let hist = &self.root_history;
        let mut mirror_count = 0;
        let mut total_count = 0;
        let mut mirror_ewms = 0.0;
        let mut total_ewms = 0.0;
        let mut last_was_mirror = false;

        for i in 1..hist.move_history.len() {
            if hist.move_history[i].pla != self.root_pla {
                last_was_mirror = false;
                if hist.move_history[i].loc
                    == get_mirror_loc(hist.move_history[i - 1].loc, board.x_size, board.y_size)
                {
                    mirror_count += 1;
                    mirror_ewms += 1.0;
                    last_was_mirror = true;
                }
                total_count += 1;
                total_ewms += 1.0;
                mirror_ewms *= 0.75;
                total_ewms *= 0.75;
            }
        }

        if mirror_count as f64 >= 7.0 + 0.5 * total_count as f64
            && mirror_ewms >= 0.45 * total_ewms
            && last_was_mirror
        {
            self.mirroring_pla = get_opp(self.root_pla);

            let mut black_extra_points = 0.0;
            let num_handicap_stones = hist.compute_num_handicap_stones();
            if hist.rules.scoring_rule == ScoringRule::Area {
                if num_handicap_stones > 0 {
                    black_extra_points += (num_handicap_stones - 1) as f64;
                }
                let black_gets_last_move = (board.x_size % 2 == 1 && board.y_size % 2 == 1)
                    == (num_handicap_stones == 0 || num_handicap_stones % 2 == 1);
                if black_gets_last_move {
                    black_extra_points += 1.0;
                }
            }
            if num_handicap_stones > 0
                && hist.rules.white_handicap_bonus_rule == WhiteHandicapBonusRule::N
            {
                black_extra_points -= num_handicap_stones as f64;
            }
            if num_handicap_stones > 0
                && hist.rules.white_handicap_bonus_rule == WhiteHandicapBonusRule::NMinusOne
            {
                black_extra_points -= (num_handicap_stones - 1) as f64;
            }
            self.mirror_advantage = if self.mirroring_pla == P_BLACK {
                black_extra_points - hist.rules.komi as f64
            } else {
                hist.rules.komi as f64 - black_extra_points
            };
        }

        if board.x_size >= 7 && board.y_size >= 7 {
            self.mirror_center_symmetry_error = 0.0;
            let half_x = board.x_size / 2;
            let half_y = board.y_size / 2;
            let mut unmatched_mirror_pla_stones = 0;
            for dy in -3..=3 {
                for dx in -3..=3 {
                    let loc = location::get_loc(half_x + dx, half_y + dy, board.x_size);
                    let mirror_loc = get_mirror_loc(loc, board.x_size, board.y_size);
                    if loc == mirror_loc {
                        continue;
                    }
                    let c0 = board.colors[loc as usize];
                    let c1 = board.colors[mirror_loc as usize];
                    if c0 == get_opp(self.mirroring_pla) && c1 != self.mirroring_pla {
                        self.mirror_center_symmetry_error += 1.0;
                    }
                    if c0 == self.mirroring_pla && c1 == C_EMPTY {
                        unmatched_mirror_pla_stones += 1;
                    }
                }
            }
            if self.mirror_center_symmetry_error > 0.0 {
                self.mirror_center_symmetry_error += 0.2 * unmatched_mirror_pla_stones as f64;
            }
            if self.mirror_center_symmetry_error >= 1.0 {
                self.mirror_center_symmetry_error = 0.5
                    * self.mirror_center_symmetry_error
                    * (1.0 + self.mirror_center_symmetry_error);
            }
        }
    }

    fn is_mirroring_since_search_start(
        &self,
        thread_history: &BoardHistory,
        skip_recent: i32,
    ) -> bool {
        let x_size = thread_history.initial_board.x_size;
        let y_size = thread_history.initial_board.y_size;
        let root_len = self.root_history.move_history.len() as i32;
        let thread_len = thread_history.move_history.len() as i32;
        let mut i = root_len + 1;
        while i + skip_recent < thread_len {
            if thread_history.move_history[i as usize].loc
                != get_mirror_loc(
                    thread_history.move_history[(i - 1) as usize].loc,
                    x_size,
                    y_size,
                )
            {
                return false;
            }
            i += 2;
        }
        true
    }

    fn maybe_apply_anti_mirror_policy(
        &self,
        nn_policy_prob: &mut f32,
        move_loc: Loc,
        policy_probs: &[f32],
        move_pla: Player,
        thread: Option<&SearchThread>,
    ) {
        let thread = match thread {
            Some(t) => t,
            None => return,
        };
        let x_size = thread.board.x_size;
        let y_size = thread.board.y_size;

        let mut weight = 0.0;

        if move_pla == get_opp(self.root_pla) && !thread.history.move_history.is_empty() {
            let prev_loc = thread.history.move_history[thread.history.move_history.len() - 1].loc;
            if prev_loc == PASS_LOC {
                return;
            }
            let mut mirror_loc = get_mirror_loc(prev_loc, x_size, y_size);
            if policy_probs[self.get_pos(mirror_loc) as usize] < 0.0 {
                mirror_loc = PASS_LOC;
            }
            if move_loc == mirror_loc {
                weight = 1.0;
                let center_loc = get_center_loc(x_size, y_size);
                let is_difficult = center_loc != NULL_LOC
                    && thread.board.colors[center_loc as usize] == self.mirroring_pla
                    && self.mirror_advantage >= -0.5;
                if is_difficult {
                    weight *= 3.0;
                }
            }
        } else if move_pla == self.root_pla && move_loc != PASS_LOC {
            if is_central(move_loc, x_size, y_size) {
                weight = 0.3;
            } else {
                if is_near_central(move_loc, x_size, y_size) {
                    weight = 0.05;
                }
                let center_loc = get_center_loc(x_size, y_size);
                if center_loc != NULL_LOC {
                    if self.root_board.colors[center_loc as usize] == get_opp(move_pla) {
                        if thread.board.is_adjacent_to_chain(move_loc, center_loc) {
                            weight = 0.05;
                        } else {
                            let distance_sq =
                                euclidean_distance_squared(move_loc, center_loc, x_size);
                            if distance_sq <= 2 {
                                weight = 0.05;
                            } else if distance_sq <= 4 {
                                weight = 0.03;
                            }
                        }
                    }
                }
            }
        }

        if weight > 0.0 {
            let root_len = self.root_history.move_history.len() as i32;
            let thread_len = thread.history.move_history.len() as i32;
            weight /= 1.0 + ((thread_len - root_len) as f64).sqrt();
            *nn_policy_prob = *nn_policy_prob + (1.0 - *nn_policy_prob) * weight as f32;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn maybe_apply_anti_mirror_forced_explore(
        &self,
        child_utility: &mut f64,
        parent_utility: f64,
        move_loc: Loc,
        policy_probs: &[f32],
        this_child_weight: f64,
        total_child_weight: f64,
        move_pla: Player,
        thread: Option<&SearchThread>,
        parent: &SearchNode,
    ) {
        let thread = match thread {
            Some(t) => t,
            None => return,
        };
        assert!(self.mirroring_pla == get_opp(self.root_pla));

        let x_size = thread.board.x_size;
        let y_size = thread.board.y_size;
        let center_loc = get_center_loc(x_size, y_size);
        let is_difficult = center_loc != NULL_LOC
            && thread.board.colors[center_loc as usize] == self.mirroring_pla
            && self.mirror_advantage >= -0.5;
        let is_root = std::ptr::eq(
            parent,
            self.root_node.as_deref().expect("root node present"),
        );

        if move_pla == self.mirroring_pla && !thread.history.move_history.is_empty() {
            let prev_loc = thread.history.move_history[thread.history.move_history.len() - 1].loc;
            if prev_loc == PASS_LOC {
                return;
            }
            let mut mirror_loc = get_mirror_loc(prev_loc, x_size, y_size);
            if policy_probs[self.get_pos(mirror_loc) as usize] < 0.0 {
                mirror_loc = PASS_LOC;
            }
            if move_loc == mirror_loc {
                let mut proportion_to_dump: f64 = 0.0;
                let mut proportion_to_bias: f64 = 0.0;
                if is_difficult {
                    proportion_to_dump = 0.20;
                    if mirror_loc != PASS_LOC {
                        let dist_sq =
                            euclidean_distance_squared(center_loc, mirror_loc, x_size) as f64;
                        proportion_to_dump = proportion_to_dump.max(
                            1.0 / (0.75 + 0.5 * dist_sq.sqrt())
                                / self.mirror_center_symmetry_error.max(1.0),
                        );
                    }
                    proportion_to_bias = 0.75;
                } else if self.mirror_advantage >= 5.0 {
                    proportion_to_dump = 0.15;
                    proportion_to_bias = 0.50;
                } else if self.mirror_advantage >= -5.0 {
                    proportion_to_dump = 0.10 + self.mirror_advantage;
                    proportion_to_bias = 0.30 + self.mirror_advantage * 4.0;
                } else {
                    proportion_to_dump = 0.05;
                    proportion_to_bias = 0.10;
                }

                if mirror_loc == PASS_LOC {
                    let factor = if move_loc == center_loc {
                        0.35
                    } else {
                        0.35 / self.mirror_center_symmetry_error.max(1.0).sqrt()
                    };
                    proportion_to_dump *= factor;
                }
                if self.mirror_center_symmetry_error >= 1.0 {
                    proportion_to_dump /= self.mirror_center_symmetry_error;
                    proportion_to_bias /= self.mirror_center_symmetry_error;
                }

                if this_child_weight < proportion_to_dump * total_child_weight {
                    *child_utility += if parent.next_pla == P_WHITE {
                        100.0
                    } else {
                        -100.0
                    };
                }
                if this_child_weight < proportion_to_bias * total_child_weight {
                    let bonus = 0.18 * (0.3_f64).max(1.0 - 0.7 * parent_utility * parent_utility);
                    *child_utility += if parent.next_pla == P_WHITE {
                        bonus
                    } else {
                        -bonus
                    };
                }
                if this_child_weight < 0.5 * proportion_to_bias * total_child_weight {
                    let bonus = 0.36 * (0.3_f64).max(1.0 - 0.7 * parent_utility * parent_utility);
                    *child_utility += if parent.next_pla == P_WHITE {
                        bonus
                    } else {
                        -bonus
                    };
                }
            }
        } else if move_pla == self.root_pla && move_loc != PASS_LOC {
            let mut proportion_to_dump = 0.0;
            if is_difficult {
                if thread.board.is_adjacent_to_chain(move_loc, center_loc) {
                    let num_libs = thread.board.get_num_liberties(center_loc);
                    let bonus =
                        0.75 / (1.0 + num_libs as f64) / self.mirror_center_symmetry_error.max(1.0)
                            * (0.3_f64).max(1.0 - 0.7 * parent_utility * parent_utility);
                    *child_utility += if parent.next_pla == P_WHITE {
                        bonus
                    } else {
                        -bonus
                    };
                    proportion_to_dump = 0.10 / num_libs as f64;
                }
                let distance_sq = euclidean_distance_squared(move_loc, center_loc, x_size) as f64;
                if distance_sq <= 2.0 {
                    proportion_to_dump = proportion_to_dump.max(0.010);
                } else if distance_sq <= 4.0 {
                    proportion_to_dump = proportion_to_dump.max(0.005);
                }
            }
            if move_loc == center_loc {
                if is_root {
                    proportion_to_dump = 0.06;
                } else {
                    proportion_to_dump = 0.12;
                }
            }

            let utility_loss = if parent.next_pla == P_WHITE {
                parent_utility - *child_utility
            } else {
                *child_utility - parent_utility
            };
            if utility_loss > 0.0 && utility_loss * proportion_to_dump > 0.03 {
                proportion_to_dump += 0.5 * (0.03 / utility_loss - proportion_to_dump);
            }

            if !thread.history.move_history.is_empty() {
                let prev_loc =
                    thread.history.move_history[thread.history.move_history.len() - 1].loc;
                if prev_loc != NULL_LOC && prev_loc != PASS_LOC {
                    let center_distance_squared =
                        euclidean_distance_squared(center_loc, prev_loc, x_size) as f64;
                    if center_distance_squared <= 16.0 {
                        proportion_to_dump *= 0.900;
                    }
                    if center_distance_squared <= 5.0 {
                        proportion_to_dump *= 0.825;
                    }
                    if center_distance_squared <= 2.0 {
                        proportion_to_dump *= 0.750;
                    }
                }
            }

            if this_child_weight < proportion_to_dump * total_child_weight {
                *child_utility += if parent.next_pla == P_WHITE {
                    100.0
                } else {
                    -100.0
                };
            }
        }
    }

    fn hack_nn_output_for_mirror(&self, result: &mut Arc<NNOutput>) {
        let center_loc = get_center_loc(self.root_board.x_size, self.root_board.y_size);
        if center_loc == NULL_LOC {
            return;
        }
        let center_pos = self.get_pos(center_loc) as usize;
        let output = Arc::make_mut(result);
        let owner_map = match output.white_owner_map.as_ref() {
            Some(m) => m,
            None => return,
        };
        if center_pos >= owner_map.len() {
            return;
        }

        let total_wl_prob = (output.white_win_prob + output.white_loss_prob) as f64;
        let own_scale = if self.mirror_center_symmetry_error <= 0.0 {
            0.7
        } else {
            0.3
        };
        let mut wl =
            (output.white_win_prob - output.white_loss_prob) as f64 / (total_wl_prob + 1e-10);
        wl = wl.min(1.0 - 1e-15).max(-1.0 + 1e-15);
        wl = (wl.atanh() + own_scale * owner_map[center_pos] as f64).tanh();
        let mut white_new_win_prob = 0.5 + 0.5 * wl;
        white_new_win_prob = total_wl_prob * white_new_win_prob;

        output.white_win_prob = white_new_win_prob as f32;
        output.white_loss_prob = (total_wl_prob - white_new_win_prob) as f32;
    }
}

// Recursive graph walking and thread pooling
impl<'a> Search<'a> {
    fn num_additional_threads_to_use_for_tasks(&self) -> i32 {
        (self.search_params.num_threads - 1).max(0)
    }

    /// Spawns the persistent worker thread pool used by the C++ implementation.
    ///
    /// The current Rust port executes parallel tasks via scoped threads in
    /// [`perform_task_with_threads`], so this function is a no-op that keeps the
    /// API surface identical to the original C++. A future slice may revive the
    /// persistent pool if profiling shows thread-spawn overhead matters.
    fn spawn_threads_if_needed(&mut self) {
        let desired = self.num_additional_threads_to_use_for_tasks();
        if self.num_threads_spawned >= desired {
            return;
        }
        self.kill_threads();
        self.num_threads_spawned = desired;
    }

    /// Tears down the persistent worker thread pool.
    ///
    /// Like [`spawn_threads_if_needed`], this is currently a no-op because the
    /// scoped-thread implementation in [`perform_task_with_threads`] does not
    /// keep long-lived workers.
    fn kill_threads(&mut self) {
        self.num_threads_spawned = 0;
    }

    /// Runs `task` on thread 0 and, when `cap_threads` allows, on additional
    /// worker threads. Workers are created on demand with `std::thread::scope`,
    /// which lets the task closure safely borrow non-`'static` data from the
    /// caller while the caller waits for all workers to finish.
    fn perform_task_with_threads(&self, task: &(dyn Fn(i32) + Send + Sync), cap_threads: i32) {
        let num_additional =
            ((cap_threads - 1).min(self.num_additional_threads_to_use_for_tasks())).max(0);
        if num_additional <= 0 {
            task(0);
            return;
        }
        std::thread::scope(|s| {
            for i in 1..=num_additional {
                s.spawn(move || task(i));
            }
            task(0);
        });
    }

    fn maybe_append_shuffled_int_range(
        cap: i32,
        rand: &mut Option<&mut Rand>,
        rand_buf: &mut Vec<i32>,
    ) {
        let r = match rand.as_mut() {
            Some(r) => r,
            None => return,
        };
        if cap <= 0 {
            return;
        }
        let start = rand_buf.len();
        for i in 0..cap {
            rand_buf.push(i);
        }
        for i in 1..cap {
            let j = (r.next_u32() % (i as u32 + 1)) as i32;
            let tmp = rand_buf[start + i as usize];
            rand_buf[start + i as usize] = rand_buf[start + j as usize];
            rand_buf[start + j as usize] = tmp;
        }
    }

    fn apply_recursively_post_order_multithreaded(
        &mut self,
        nodes: &[&SearchNode],
        f: &(dyn Fn(&SearchNode, i32) + Send + Sync),
    ) {
        self.search_node_age += 1;
        let search_node_age = self.search_node_age;
        let num_additional = self.num_additional_threads_to_use_for_tasks();

        let mut rands: Vec<Option<Mutex<Rand>>> = Vec::with_capacity((num_additional + 1) as usize);
        rands.push(None);
        for _ in 1..=num_additional {
            let seed = self.non_search_rand.next_u64();
            rands.push(Some(Mutex::new(Rand::new_from_seed(&format!(
                "postorder {}",
                seed
            )))));
        }
        let mutex_pool = self.mutex_pool.as_ref().unwrap();

        let g = |thread_idx: i32| {
            let mut guard = rands[thread_idx as usize].as_ref().map(|m| m.lock());
            let mut rand: Option<&mut Rand> = guard.as_mut().map(|g| &mut **g);
            let mut node_buf = HashSet::<*const SearchNode>::new();
            let mut rand_buf = Vec::new();
            let start = rand_buf.len();
            Self::maybe_append_shuffled_int_range(nodes.len() as i32, &mut rand, &mut rand_buf);
            for i in 0..nodes.len() as i32 {
                let child_idx = if rand.is_some() {
                    rand_buf[start + i as usize]
                } else {
                    i
                };
                Self::apply_recursively_post_order_multithreaded_helper(
                    nodes[child_idx as usize],
                    thread_idx,
                    &mut rand,
                    &mut node_buf,
                    &mut rand_buf,
                    search_node_age,
                    mutex_pool,
                    f,
                );
            }
        };

        self.perform_task_with_threads(&g, 0x3fff_ffff);
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_recursively_post_order_multithreaded_helper(
        node: &SearchNode,
        thread_idx: i32,
        rand: &mut Option<&mut Rand>,
        node_buf: &mut HashSet<*const SearchNode>,
        rand_buf: &mut Vec<i32>,
        search_node_age: u32,
        mutex_pool: &MutexPool,
        f: &(dyn Fn(&SearchNode, i32) + Send + Sync),
    ) {
        if node.node_age.load(Ordering::Acquire) == search_node_age {
            return;
        }
        let node_ptr = node as *const SearchNode;
        if node_buf.contains(&node_ptr) {
            return;
        }

        let children = node.get_children();
        let num_children = children.iterate_and_count_children() as i32;

        if num_children > 0 {
            let start = rand_buf.len();
            Self::maybe_append_shuffled_int_range(num_children, rand, rand_buf);

            node_buf.insert(node_ptr);
            for i in 0..num_children {
                let child_idx = if rand.is_some() {
                    rand_buf[start + i as usize]
                } else {
                    i
                };
                let child_pointer = children.get(child_idx as usize);
                if let Some(child) = child_pointer.get_if_allocated() {
                    Self::apply_recursively_post_order_multithreaded_helper(
                        child,
                        thread_idx,
                        rand,
                        node_buf,
                        rand_buf,
                        search_node_age,
                        mutex_pool,
                        f,
                    );
                }
            }
            rand_buf.resize(start, 0);
            node_buf.remove(&node_ptr);
        }

        let _guard = mutex_pool.get_with_modulo(node.mutex_idx).lock();
        if node.node_age.load(Ordering::Acquire) == search_node_age {
            return;
        }
        f(node, thread_idx);
        node.node_age.store(search_node_age, Ordering::Release);
    }

    fn apply_recursively_any_order_multithreaded(
        &mut self,
        nodes: &[&SearchNode],
        f: &(dyn Fn(&SearchNode, i32) + Send + Sync),
    ) {
        self.search_node_age += 1;
        let search_node_age = self.search_node_age;
        let num_additional = self.num_additional_threads_to_use_for_tasks();

        let mut rands: Vec<Option<Mutex<Rand>>> = Vec::with_capacity((num_additional + 1) as usize);
        rands.push(None);
        for _ in 1..=num_additional {
            let seed = self.non_search_rand.next_u64();
            rands.push(Some(Mutex::new(Rand::new_from_seed(&format!(
                "anyorder {}",
                seed
            )))));
        }

        let g = |thread_idx: i32| {
            let mut guard = rands[thread_idx as usize].as_ref().map(|m| m.lock());
            let mut rand: Option<&mut Rand> = guard.as_mut().map(|g| &mut **g);
            let mut node_buf = HashSet::<*const SearchNode>::new();
            let mut rand_buf = Vec::new();
            let start = rand_buf.len();
            Self::maybe_append_shuffled_int_range(nodes.len() as i32, &mut rand, &mut rand_buf);
            for i in 0..nodes.len() as i32 {
                let child_idx = if rand.is_some() {
                    rand_buf[start + i as usize]
                } else {
                    i
                };
                Self::apply_recursively_any_order_multithreaded_helper(
                    nodes[child_idx as usize],
                    thread_idx,
                    &mut rand,
                    &mut node_buf,
                    &mut rand_buf,
                    search_node_age,
                    f,
                );
            }
        };

        self.perform_task_with_threads(&g, 0x3fff_ffff);
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_recursively_any_order_multithreaded_helper(
        node: &SearchNode,
        thread_idx: i32,
        rand: &mut Option<&mut Rand>,
        node_buf: &mut HashSet<*const SearchNode>,
        rand_buf: &mut Vec<i32>,
        search_node_age: u32,
        f: &(dyn Fn(&SearchNode, i32) + Send + Sync),
    ) {
        if node.node_age.load(Ordering::Acquire) == search_node_age {
            return;
        }
        let node_ptr = node as *const SearchNode;
        if node_buf.contains(&node_ptr) {
            return;
        }

        let children = node.get_children();
        let num_children = children.iterate_and_count_children() as i32;

        if num_children > 0 {
            let start = rand_buf.len();
            Self::maybe_append_shuffled_int_range(num_children, rand, rand_buf);

            node_buf.insert(node_ptr);
            for i in 0..num_children {
                let child_idx = if rand.is_some() {
                    rand_buf[start + i as usize]
                } else {
                    i
                };
                let child_pointer = children.get(child_idx as usize);
                if let Some(child) = child_pointer.get_if_allocated() {
                    Self::apply_recursively_any_order_multithreaded_helper(
                        child,
                        thread_idx,
                        rand,
                        node_buf,
                        rand_buf,
                        search_node_age,
                        f,
                    );
                }
            }
            rand_buf.resize(start, 0);
            node_buf.remove(&node_ptr);
        }

        let old_age = node.node_age.swap(search_node_age, Ordering::AcqRel);
        if old_age == search_node_age {
            return;
        }
        f(node, thread_idx);
    }

    pub fn enumerate_tree_post_order(&mut self) -> Vec<*mut SearchNode> {
        let root_ptr = self
            .root_node
            .as_deref()
            .map_or(std::ptr::null(), |r| r as *const SearchNode);

        let size_counter = AtomicI64::new(0);
        let size_f = |_node: &SearchNode, _thread_idx: i32| {
            size_counter.fetch_add(1, Ordering::Relaxed);
        };
        if !root_ptr.is_null() {
            self.apply_recursively_post_order_multithreaded(&[unsafe { &*root_ptr }], &size_f);
        }
        let size = size_counter.load(Ordering::Relaxed) as usize;

        let nodes: Vec<AtomicPtr<SearchNode>> = (0..size)
            .map(|_| AtomicPtr::new(std::ptr::null_mut()))
            .collect();
        let index_counter = AtomicI64::new(0);
        let collect_f = |node: &SearchNode, _thread_idx: i32| {
            let index = index_counter.fetch_add(1, Ordering::Relaxed) as usize;
            assert!(index < size);
            nodes[index].store(
                node as *const SearchNode as *mut SearchNode,
                Ordering::Relaxed,
            );
        };
        if !root_ptr.is_null() {
            self.apply_recursively_post_order_multithreaded(&[unsafe { &*root_ptr }], &collect_f);
        }
        assert_eq!(index_counter.load(Ordering::Relaxed) as usize, size);

        nodes.into_iter().map(|p| p.into_inner()).collect()
    }
}

// Time management
impl<'a> Search<'a> {
    fn num_visits_needed_to_be_non_futile(&self, max_visits_move_visits: f64) -> f64 {
        let required_visits = self.search_params.futile_visits_threshold * max_visits_move_visits;
        let chosen_move_temperature = self.interpolate_early(
            self.search_params.chosen_move_temperature_halflife,
            self.search_params.chosen_move_temperature_early,
            self.search_params.chosen_move_temperature,
        );
        if chosen_move_temperature < 1e-3 {
            return required_visits;
        }
        let required_visits_due_to_temp =
            max_visits_move_visits * 0.01_f64.powf(chosen_move_temperature);
        required_visits.min(required_visits_due_to_temp)
    }

    fn compute_upper_bound_visits_left_due_to_time(
        &self,
        root_visits: i64,
        time_used: f64,
        planned_time_limit: f64,
    ) -> f64 {
        if root_visits <= 1 {
            return 1e30;
        }
        let time_thought_so_far = self.effective_search_time_carried_over + time_used;
        let time_left_planned = planned_time_limit - time_used;
        if time_thought_so_far < 0.1 {
            return 1e30;
        }

        let proportion_of_time_thought_left = time_left_planned / time_thought_so_far;
        (proportion_of_time_thought_left * root_visits as f64
            + self.search_params.num_threads as f64
            - 1.0)
            .ceil()
    }

    fn recompute_search_time_limit(
        &self,
        tc: &TimeControls,
        time_used: f64,
        search_factor: f64,
        root_visits: i64,
    ) -> f64 {
        let (tc_min, mut tc_rec, tc_max) = tc
            .get_time(
                &self.root_board,
                &self.root_history,
                self.search_params.lag_buffer,
            )
            .unwrap_or((0.0, 0.0, 0.0));

        tc_rec *= self.search_params.overallocate_time_factor;

        if self.search_params.midgame_time_factor != 1.0 {
            let board_area_scale = (self.root_board.x_size * self.root_board.y_size) as f64 / 361.0;
            let mut presumed_turn_number = self.root_history.get_current_turn_number() as f64;
            if presumed_turn_number < 0.0 {
                presumed_turn_number = 0.0;
            }

            let midgame_weight = if presumed_turn_number
                < self.search_params.midgame_turn_peak_time * board_area_scale
            {
                presumed_turn_number
                    / (self.search_params.midgame_turn_peak_time * board_area_scale)
            } else {
                (-(presumed_turn_number
                    - self.search_params.midgame_turn_peak_time * board_area_scale)
                    / (self.search_params.endgame_turn_time_decay * board_area_scale))
                    .exp()
            };
            let midgame_weight = midgame_weight.clamp(0.0, 1.0);
            tc_rec *= 1.0 + midgame_weight * (self.search_params.midgame_time_factor - 1.0);
        }

        if self.search_params.obvious_moves_time_factor < 1.0 {
            let mut surprise = 0.0;
            let mut search_entropy = 0.0;
            let mut policy_entropy = 0.0;
            let suc = self.get_policy_surprise_and_entropy(
                &mut surprise,
                &mut search_entropy,
                &mut policy_entropy,
            );
            if suc {
                let obviousness_by_entropy = (-policy_entropy
                    / self.search_params.obvious_moves_policy_entropy_tolerance)
                    .exp();
                let obviousness_by_surprise =
                    (-surprise / self.search_params.obvious_moves_policy_surprise_tolerance).exp();
                let obviousness_weight = obviousness_by_entropy.min(obviousness_by_surprise);
                tc_rec *=
                    1.0 + obviousness_weight * (self.search_params.obvious_moves_time_factor - 1.0);
            }
        }

        if tc_rec > 1e-20 {
            let remaining_time_needed = tc_rec - self.effective_search_time_carried_over;
            let remaining_time_needed_factor = remaining_time_needed / tc_rec;
            tc_rec *= (1.0 + (remaining_time_needed_factor * 6.0).exp()).ln() / 6.0;
        }

        tc_rec = tc.round_up_time_limit_if_needed(self.search_params.lag_buffer, time_used, tc_rec);
        if tc_rec > tc_max {
            tc_rec = tc_max;
        }

        if self.search_params.futile_visits_threshold > 0.0 {
            let upper_bound_visits_left_due_to_time =
                self.compute_upper_bound_visits_left_due_to_time(root_visits, time_used, tc_rec);
            if upper_bound_visits_left_due_to_time
                < self.search_params.futile_visits_threshold * root_visits as f64
            {
                let mut locs = Vec::new();
                let mut play_selection_values = Vec::new();
                let mut visit_counts = Vec::new();
                let suc = self.get_play_selection_values_with_counts(
                    &mut locs,
                    &mut play_selection_values,
                    Some(&mut visit_counts),
                    1.0,
                );
                if suc && !play_selection_values.is_empty() {
                    if play_selection_values.len() == visit_counts.len() {
                        let num_moves = play_selection_values.len();
                        let mut max_visits_idx = 0;
                        let mut best_move_idx = 0;
                        for i in 1..num_moves {
                            if play_selection_values[i] > play_selection_values[best_move_idx] {
                                best_move_idx = i;
                            }
                            if visit_counts[i] > visit_counts[max_visits_idx] {
                                max_visits_idx = i;
                            }
                        }
                        if max_visits_idx == best_move_idx {
                            let required_visits = self
                                .num_visits_needed_to_be_non_futile(visit_counts[max_visits_idx]);
                            let mut found_possible_alternative_move = false;
                            for i in 0..num_moves {
                                if i == best_move_idx {
                                    continue;
                                }
                                if visit_counts[i] + upper_bound_visits_left_due_to_time
                                    >= required_visits
                                {
                                    found_possible_alternative_move = true;
                                    break;
                                }
                            }
                            if !found_possible_alternative_move {
                                tc_rec = time_used * (1.0 - 1e-10);
                            }
                        }
                    }
                }
            }
        }

        tc_rec = tc.round_up_time_limit_if_needed(self.search_params.lag_buffer, time_used, tc_rec);
        if tc_rec > tc_max {
            tc_rec = tc_max;
        }

        if tc_rec < tc_min {
            tc_rec = tc_min;
        }
        tc_rec *= search_factor;
        if tc_rec > tc_max {
            tc_rec = tc_max;
        }

        tc_rec
    }
}

// Neural net queries
impl<'a> Search<'a> {
    fn compute_root_nn_evaluation(
        &mut self,
        nn_result_buf: &mut NNResultBuf,
        include_owner_map: bool,
    ) {
        let board = self.root_board.clone();
        let hist = &self.root_history;
        let pla = self.root_pla;
        let is_root = true;
        let skip_cache = false;

        let mut nn_input_params = MiscNNInputParams::default();
        nn_input_params.draw_equivalent_wins_for_white =
            self.search_params.draw_equivalent_wins_for_white;
        nn_input_params.conservative_pass_and_is_root =
            self.search_params.conservative_pass && is_root;
        nn_input_params.enable_passing_hacks = self.search_params.enable_passing_hacks;
        nn_input_params.nn_policy_temperature = self.search_params.nn_policy_temperature;
        nn_input_params.avoid_mytdagger_hack = self.search_params.avoid_mytd_dagger_hack_pla == pla;
        nn_input_params.policy_optimism = self.search_params.root_policy_optimism;
        if self.search_params.playout_doubling_advantage != 0.0 {
            let playout_doubling_advantage_pla = self.get_playout_doubling_advantage_pla();
            nn_input_params.playout_doubling_advantage =
                if get_opp(pla) == playout_doubling_advantage_pla {
                    -self.search_params.playout_doubling_advantage
                } else {
                    self.search_params.playout_doubling_advantage
                };
        }
        if self.search_params.ignore_pre_root_history || self.search_params.ignore_all_history {
            nn_input_params.max_history = 0;
        }

        let evaluator = self
            .nn_evaluator
            .expect("compute_root_nn_evaluation called without an NN evaluator");
        evaluator.evaluate(
            &board,
            hist,
            pla,
            &nn_input_params,
            nn_result_buf,
            skip_cache,
            include_owner_map,
        );
    }

    fn init_node_nn_output(
        &self,
        thread: &mut SearchThread,
        node: &mut SearchNode,
        is_root: bool,
        skip_cache: bool,
        is_re_init: bool,
    ) -> bool {
        let mut include_owner_map = is_root || self.always_include_owner_map;
        let mut anti_mirror_difficult = false;
        if self.search_params.anti_mirror
            && self.mirroring_pla != C_EMPTY
            && self.mirror_advantage >= -0.5
        {
            let center_loc = location::get_center_loc(thread.board.x_size, thread.board.y_size);
            if center_loc != NULL_LOC
                && thread.board.colors[center_loc as usize] == get_opp(self.root_pla)
                && self.is_mirroring_since_search_start(&thread.history, 4)
            {
                include_owner_map = true;
                anti_mirror_difficult = true;
            }
        }

        let mut nn_input_params = MiscNNInputParams::default();
        nn_input_params.draw_equivalent_wins_for_white =
            self.search_params.draw_equivalent_wins_for_white;
        nn_input_params.conservative_pass_and_is_root =
            self.search_params.conservative_pass && is_root;
        nn_input_params.enable_passing_hacks = self.search_params.enable_passing_hacks;
        nn_input_params.nn_policy_temperature = self.search_params.nn_policy_temperature;
        nn_input_params.avoid_mytdagger_hack =
            self.search_params.avoid_mytd_dagger_hack_pla == thread.pla;
        nn_input_params.policy_optimism = if is_root {
            self.search_params.root_policy_optimism
        } else {
            self.search_params.policy_optimism
        };
        if self.search_params.playout_doubling_advantage != 0.0 {
            let playout_doubling_advantage_pla = self.get_playout_doubling_advantage_pla();
            nn_input_params.playout_doubling_advantage =
                if get_opp(thread.pla) == playout_doubling_advantage_pla {
                    -self.search_params.playout_doubling_advantage
                } else {
                    self.search_params.playout_doubling_advantage
                };
        }
        if self.search_params.ignore_all_history {
            nn_input_params.max_history = 0;
        } else if self.search_params.ignore_pre_root_history {
            nn_input_params.max_history = if is_root {
                0
            } else {
                0.max(
                    thread.history.move_history.len() as i32
                        - self.root_history.move_history.len() as i32,
                )
            };
        }

        let evaluator = self
            .nn_evaluator
            .expect("init_node_nn_output called without an NN evaluator");

        let mut result: *mut Arc<NNOutput>;
        let mut human_result: Option<*mut Arc<NNOutput>> = None;

        if is_root && self.search_params.root_num_symmetries_to_sample > 1 {
            const NUM_SYMMETRIES: i32 = 8;
            let num_symmetries_to_sample = self.search_params.root_num_symmetries_to_sample;
            let mut symmetry_indexes: Vec<i32> = (0..NUM_SYMMETRIES).collect();
            let mut outputs = Vec::with_capacity(num_symmetries_to_sample as usize);
            for i in 0..num_symmetries_to_sample {
                let swap_idx = thread.rand.next_i32_range(i, NUM_SYMMETRIES - 1) as usize;
                symmetry_indexes.swap(i as usize, swap_idx);
                let mut iter_params = nn_input_params;
                iter_params.symmetry = symmetry_indexes[i as usize];
                let skip_cache_this_iteration = true;
                evaluator.evaluate(
                    &thread.board,
                    &thread.history,
                    thread.pla,
                    &iter_params,
                    &mut thread.nn_result_buf,
                    skip_cache_this_iteration,
                    include_owner_map,
                );
                outputs.push(
                    (*thread
                        .nn_result_buf
                        .result
                        .take()
                        .expect("evaluate did not produce a result"))
                    .clone(),
                );
            }
            let averaged = Arc::new(NNOutput::average(&outputs));
            result = Box::into_raw(Box::new(averaged));

            if self.needs_human_output_in_tree() || (is_root && self.needs_human_output_at_root()) {
                let human_evaluator = self
                    .human_evaluator
                    .expect("human output requested but no human evaluator");
                let mut outputs = Vec::with_capacity(num_symmetries_to_sample as usize);
                for i in 0..num_symmetries_to_sample {
                    let swap_idx = thread.rand.next_i32_range(i, NUM_SYMMETRIES - 1) as usize;
                    symmetry_indexes.swap(i as usize, swap_idx);
                    let mut iter_params = nn_input_params;
                    iter_params.symmetry = symmetry_indexes[i as usize];
                    let skip_cache_this_iteration = true;
                    human_evaluator.evaluate(
                        &thread.board,
                        &thread.history,
                        thread.pla,
                        &iter_params,
                        &mut thread.nn_result_buf,
                        skip_cache_this_iteration,
                        include_owner_map,
                    );
                    outputs.push(
                        (*thread
                            .nn_result_buf
                            .result
                            .take()
                            .expect("human evaluate did not produce a result"))
                        .clone(),
                    );
                }
                let averaged = Arc::new(NNOutput::average(&outputs));
                human_result = Some(Box::into_raw(Box::new(averaged)));
            }
        } else {
            evaluator.evaluate(
                &thread.board,
                &thread.history,
                thread.pla,
                &nn_input_params,
                &mut thread.nn_result_buf,
                skip_cache,
                include_owner_map,
            );
            result = Box::into_raw(Box::new(
                thread
                    .nn_result_buf
                    .result
                    .take()
                    .expect("evaluate did not produce a result"),
            ));

            if self.needs_human_output_in_tree() || (is_root && self.needs_human_output_at_root()) {
                let human_evaluator = self
                    .human_evaluator
                    .expect("human output requested but no human evaluator");
                human_evaluator.evaluate(
                    &thread.board,
                    &thread.history,
                    thread.pla,
                    &nn_input_params,
                    &mut thread.nn_result_buf,
                    skip_cache,
                    include_owner_map,
                );
                human_result = Some(Box::into_raw(Box::new(
                    thread
                        .nn_result_buf
                        .result
                        .take()
                        .expect("human evaluate did not produce a result"),
                )));
            }
        }

        if anti_mirror_difficult {
            let mut cloned = unsafe { &*result }.clone();
            unsafe {
                drop(Box::from_raw(result));
            }
            self.hack_nn_output_for_mirror(&mut cloned);
            result = Box::into_raw(Box::new(cloned));
        }

        debug_assert!((unsafe { &*result }).noised_policy_probs.is_none());

        let noised_result =
            self.maybe_add_policy_noise_and_temp(thread, is_root, unsafe { &*result });
        if !noised_result.is_null() {
            unsafe {
                drop(Box::from_raw(result));
            }
            result = noised_result;
        }

        node.node_age.store(self.search_node_age, Ordering::Release);

        if is_re_init {
            if let Some(human_ptr) = human_result {
                node.store_human_output(human_ptr, thread);
            }
            node.store_nn_output(result, thread)
        } else {
            if let Some(human_ptr) = human_result {
                if !node.store_human_output_if_null(human_ptr) {
                    unsafe {
                        drop(Box::from_raw(human_ptr));
                    }
                }
            }
            if node.store_nn_output_if_null(result) {
                self.add_current_nn_output_as_leaf_value(node, true);
                true
            } else {
                unsafe {
                    drop(Box::from_raw(result));
                }
                false
            }
        }
    }

    fn maybe_recompute_existing_nn_output(
        &self,
        thread: &mut SearchThread,
        node: &mut SearchNode,
        is_root: bool,
    ) -> bool {
        let mut recompute_happened = false;
        if is_root && node.node_age.load(Ordering::Acquire) != self.search_node_age {
            let old_age = node.node_age.swap(self.search_node_age, Ordering::AcqRel);
            if old_age < self.search_node_age {
                let nn_output = node.get_nn_output();
                let human_output = node.get_human_output();
                debug_assert!(
                    nn_output.is_some(),
                    "maybe_recompute_existing_nn_output called on a node without nn output"
                );

                let nn_output_ref = nn_output.as_ref().unwrap();
                let needs_human = self.needs_human_output_at_root();
                if nn_output_ref.white_owner_map.is_none()
                    || (self.search_params.conservative_pass
                        && thread
                            .history
                            .pass_would_end_game(&thread.board, thread.pla))
                    || self.search_params.root_num_symmetries_to_sample > 1
                    || self.search_params.root_policy_optimism != self.search_params.policy_optimism
                    || (self.search_params.ignore_pre_root_history
                        && !self.search_params.ignore_all_history)
                    || (human_output.is_none() && needs_human)
                {
                    const SKIP_CACHE: bool = false;
                    self.init_node_nn_output(thread, node, is_root, SKIP_CACHE, true);
                    recompute_happened = true;
                } else {
                    let result =
                        self.maybe_add_policy_noise_and_temp(thread, is_root, nn_output_ref);
                    if !result.is_null() {
                        node.store_nn_output(result, thread);
                        recompute_happened = true;
                    }
                }
            }
        }
        recompute_happened
    }

    fn needs_human_output_at_root(&self) -> bool {
        self.human_evaluator.is_some()
            && (self.search_params.human_sl_profile.initialized
                || !self
                    .human_evaluator
                    .expect("needs_human_output_at_root called without human evaluator")
                    .requires_sgf_metadata())
    }

    fn needs_human_output_in_tree(&self) -> bool {
        self.needs_human_output_at_root()
            && (self.search_params.human_sl_pla_explore_prob_weightless > 0.0
                || self.search_params.human_sl_pla_explore_prob_weightful > 0.0
                || self.search_params.human_sl_opp_explore_prob_weightless > 0.0
                || self.search_params.human_sl_opp_explore_prob_weightful > 0.0)
    }
}

// Move selection during search

const TOTALCHILDWEIGHT_PUCT_OFFSET: f64 = 0.01;

fn cpuct_exploration(total_child_weight: f64, search_params: &SearchParams) -> f64 {
    search_params.cpuct_exploration
        + search_params.cpuct_exploration_log
            * ((total_child_weight + search_params.cpuct_exploration_base)
                / search_params.cpuct_exploration_base)
                .ln()
}

fn cpuct_exploration_human(total_child_weight: f64, search_params: &SearchParams) -> f64 {
    search_params.human_sl_cpuct_exploration
        + search_params.human_sl_cpuct_permanent * total_child_weight.sqrt()
}

fn explore_selection_value_raw(
    explore_scaling: f64,
    nn_policy_prob: f64,
    child_weight: f64,
    child_utility: f64,
    pla: Player,
) -> f64 {
    if nn_policy_prob < 0.0 {
        return Search::POLICY_ILLEGAL_SELECTION_VALUE;
    }
    let explore_component = explore_scaling * nn_policy_prob / (1.0 + child_weight);
    let value_component = if pla == P_WHITE {
        child_utility
    } else {
        -child_utility
    };
    explore_component + value_component
}

fn explore_selection_value_inverse_raw(
    explore_selection_value: f64,
    explore_scaling: f64,
    nn_policy_prob: f64,
    child_utility: f64,
    pla: Player,
) -> f64 {
    if nn_policy_prob < 0.0 {
        return 0.0;
    }
    let value_component = if pla == P_WHITE {
        child_utility
    } else {
        -child_utility
    };
    let explore_component = explore_selection_value - value_component;
    let explore_component_scaling = explore_scaling * nn_policy_prob;
    if explore_component <= 0.0 {
        return 1e100;
    }
    let child_weight = explore_component_scaling / explore_component - 1.0;
    child_weight.max(0.0)
}

impl<'a> Search<'a> {
    fn get_explore_scaling(
        &self,
        total_child_weight: f64,
        parent_utility_stdev_factor: f64,
    ) -> f64 {
        cpuct_exploration(total_child_weight, &self.search_params)
            * (total_child_weight + TOTALCHILDWEIGHT_PUCT_OFFSET).sqrt()
            * parent_utility_stdev_factor
    }

    fn get_explore_scaling_human(
        &self,
        total_child_weight: f64,
        _parent_utility_stdev_factor: f64,
    ) -> f64 {
        cpuct_exploration_human(total_child_weight, &self.search_params)
            * (total_child_weight + TOTALCHILDWEIGHT_PUCT_OFFSET).sqrt()
    }

    fn get_explore_selection_value(
        &self,
        node: &SearchNode,
        policy_probs: &[f32],
        move_loc: Loc,
        explore_scaling: f64,
        _total_child_weight: f64,
        edge_visits: i64,
        fpu_value: f64,
        _parent_utility: f64,
        _parent_weight_per_visit: f64,
        _is_during_search: bool,
        _warn_big_weight: bool,
        suppress_pass: bool,
    ) -> f64 {
        if suppress_pass && move_loc == PASS_LOC {
            return Self::POLICY_ILLEGAL_SELECTION_VALUE;
        }
        let move_pos = self.get_pos(move_loc) as usize;
        if move_pos >= policy_probs.len() {
            return Self::POLICY_ILLEGAL_SELECTION_VALUE;
        }
        let nn_policy_prob = policy_probs[move_pos] as f64;

        let child_visits = node.stats.visits.load(Ordering::Acquire);
        let child_weight = if child_visits <= 0 {
            0.0
        } else {
            node.stats.get_child_weight(edge_visits)
        };
        let child_utility = if child_visits <= 0 || child_weight <= 0.0 {
            fpu_value
        } else {
            node.stats.utility_avg.load(Ordering::Acquire)
        };

        // `node` is the child node here; the value is from the parent's perspective.
        let pla = get_opp(node.next_pla);
        explore_selection_value_raw(
            explore_scaling,
            nn_policy_prob,
            child_weight,
            child_utility,
            pla,
        )
    }

    fn get_explore_selection_value_inverse(
        &self,
        explore_selection_value: f64,
        node: &SearchNode,
        policy_probs: &[f32],
        move_loc: Loc,
        explore_scaling: f64,
        _total_child_weight: f64,
        edge_visits: i64,
        fpu_value: f64,
        _parent_utility: f64,
        _parent_weight_per_visit: f64,
        _is_during_search: bool,
        _warn_big_weight: bool,
        _suppress_pass: bool,
    ) -> f64 {
        let move_pos = self.get_pos(move_loc) as usize;
        if move_pos >= policy_probs.len() {
            return 0.0;
        }
        let nn_policy_prob = policy_probs[move_pos] as f64;

        let child_visits = node.stats.visits.load(Ordering::Acquire);
        let child_weight = if child_visits <= 0 {
            0.0
        } else {
            node.stats.get_child_weight(edge_visits)
        };
        let child_utility = if child_visits <= 0 || child_weight <= 0.0 {
            fpu_value
        } else {
            node.stats.utility_avg.load(Ordering::Acquire)
        };

        let pla = get_opp(node.next_pla);
        explore_selection_value_inverse_raw(
            explore_selection_value,
            explore_scaling,
            nn_policy_prob,
            child_utility,
            pla,
        )
    }

    fn maybe_apply_wide_root_noise(
        &self,
        nn_policy_prob: &mut f64,
        child_utility: &mut f64,
        rand: Option<&mut Rand>,
        parent: &SearchNode,
    ) {
        // For very large wideRootNoise, go ahead and also smooth out the policy
        *nn_policy_prob =
            nn_policy_prob.powf(1.0 / (4.0 * self.search_params.wide_root_noise + 1.0));
        if let Some(rand) = rand {
            if rand.next_bool(0.5) {
                let bonus = self.search_params.wide_root_noise * rand.next_gaussian().abs();
                if parent.next_pla == P_WHITE {
                    *child_utility += bonus;
                } else {
                    *child_utility -= bonus;
                }
            }
        }
    }

    fn get_explore_selection_value_of_child(
        &self,
        node: &SearchNode,
        policy_probs: &[f32],
        child: &SearchNode,
        move_loc: Loc,
        explore_scaling: f64,
        total_child_weight: f64,
        edge_visits: i64,
        fpu_value: f64,
        parent_utility: f64,
        parent_weight_per_visit: f64,
        is_during_search: bool,
        is_root: bool,
        _warn_big_weight: bool,
        best_child_weight: f64,
        count_edge_visit: bool,
        lcb_buf: Option<&mut f64>,
        rand: Option<&mut Rand>,
    ) -> f64 {
        let move_pos = self.get_pos(move_loc) as usize;
        if move_pos >= policy_probs.len() {
            return Self::POLICY_ILLEGAL_SELECTION_VALUE;
        }
        let mut nn_policy_prob = policy_probs[move_pos] as f64;
        if nn_policy_prob < 0.0 {
            return Self::POLICY_ILLEGAL_SELECTION_VALUE;
        }

        let child_virtual_losses = child.virtual_losses.load(Ordering::Acquire) as i64;
        let child_visits = child.stats.visits.load(Ordering::Acquire);
        let utility_avg = child.stats.utility_avg.load(Ordering::Acquire);
        // score_mean/score_mean_sq feed only the root-only ending-score bonus,
        // and utility_sq_avg only the optional PUCT-V factor; load them lazily
        // so the default descent touches just one stats cache line per child.
        let mut child_weight = if count_edge_visit {
            child
                .stats
                .get_child_weight_with_child_visits(edge_visits, child_visits)
        } else {
            child.stats.weight_sum.load(Ordering::Acquire)
        };

        let mut child_utility = if child_visits <= 0 || child_weight <= 0.0 {
            fpu_value
        } else {
            let mut u = utility_avg;
            if is_root {
                let ending_score_bonus = self.get_ending_white_score_bonus(node, move_loc);
                if ending_score_bonus != 0.0 {
                    let score_mean_avg = child.stats.score_mean_avg.load(Ordering::Acquire);
                    let score_mean_sq_avg =
                        child.stats.score_mean_sq_avg.load(Ordering::Acquire);
                    u += self.get_score_utility_diff(
                        score_mean_avg,
                        score_mean_sq_avg,
                        ending_score_bonus,
                    );
                }
            }
            u
        };

        // Virtual losses direct threads down different paths.
        if child_virtual_losses > 0 {
            let virtual_loss_weight =
                child_virtual_losses as f64 * self.search_params.num_virtual_losses_per_thread;
            // WU-UCT mode (virtualLossUtilityBlend = 0): keep only the weight
            // inflation — the exploration term p/(1+w) already shrinks for
            // in-flight playouts, while the child utility stays untouched so
            // workers can keep exploiting a clearly best node concurrently.
            // 1.0 (default) is the official KataGo soft blend, bit-identical.
            let virtual_loss_utility_blend = self.search_params.virtual_loss_utility_blend;
            if virtual_loss_utility_blend > 0.0 {
                let utility_radius = self.search_params.win_loss_utility_factor
                    + self.search_params.static_score_utility_factor
                    + self.search_params.dynamic_score_utility_factor;
                let virtual_loss_utility = if node.next_pla == P_WHITE {
                    -utility_radius
                } else {
                    utility_radius
                };
                let virtual_loss_weight_frac =
                    virtual_loss_weight / (virtual_loss_weight + 0.25_f64.max(child_weight));
                child_utility = child_utility
                    + (virtual_loss_utility - child_utility)
                        * virtual_loss_weight_frac
                        * virtual_loss_utility_blend;
            }
            child_weight += virtual_loss_weight;
        }

        if is_during_search && is_root && count_edge_visit {
            // Futile visits pruning is omitted here because it requires the
            // search thread's `upper_bound_visits_left` estimate.

            if self.search_params.root_desired_per_child_visits_coeff > 0.0 {
                if nn_policy_prob > 0.0
                    && child_weight
                        < (nn_policy_prob
                            * total_child_weight
                            * self.search_params.root_desired_per_child_visits_coeff)
                            .sqrt()
                {
                    return Self::EVALUATING_SELECTION_VALUE_PENALTY;
                }
            }

            if self.root_hint_loc != NULL_LOC && move_loc == self.root_hint_loc {
                let average_weight_per_visit =
                    (child_weight + parent_weight_per_visit) / (child_visits as f64 + 1.0);
                let children = node.get_children();
                let capacity = children.get_capacity();
                for i in 0..capacity {
                    let child_ptr = children.get(i);
                    let c = match child_ptr.get_if_allocated() {
                        Some(c) => c,
                        None => break,
                    };
                    let c_edge_visits = child_ptr.get_edge_visits();
                    let c_weight = c.stats.get_child_weight(c_edge_visits);
                    if child_weight + average_weight_per_visit < c_weight * 0.8 {
                        return Self::EVALUATING_SELECTION_VALUE_PENALTY;
                    }
                }
            }

            if self.search_params.wide_root_noise > 0.0 && nn_policy_prob >= 0.0 {
                self.maybe_apply_wide_root_noise(
                    &mut nn_policy_prob,
                    &mut child_utility,
                    rand,
                    node,
                );
            }
        }

        if is_during_search
            && self.search_params.anti_mirror
            && self.mirroring_pla != C_EMPTY
            && nn_policy_prob >= 0.0
            && count_edge_visit
        {
            let mut prob_f = nn_policy_prob as f32;
            self.maybe_apply_anti_mirror_policy(
                &mut prob_f,
                move_loc,
                policy_probs,
                node.next_pla,
                None,
            );
            nn_policy_prob = prob_f as f64;
            self.maybe_apply_anti_mirror_forced_explore(
                &mut child_utility,
                parent_utility,
                move_loc,
                policy_probs,
                child_weight,
                total_child_weight,
                node.next_pla,
                None,
                node,
            );
        }

        if let Some(lcb) = lcb_buf {
            let mut radius = 0.0;
            self.get_self_utility_lcb_and_radius(
                node,
                Some(child),
                edge_visits,
                move_loc,
                lcb,
                &mut radius,
            );
        }

        // PUCT-V style child-level variance-aware exploration (Weichart 2025,
        // arXiv:2512.21648): scale this child's exploration bonus by its
        // empirical utility stdev, normalized against cpuctUtilityStdevPrior
        // so a typical child keeps factor ~1 while volatile subtrees get
        // explored sooner. 0.0 (default) skips this entirely — bit-identical
        // to the baseline. Applies only on the search-descent path, not to
        // reporting / move-selection math.
        let mut effective_explore_scaling = explore_scaling;
        if is_during_search
            && count_edge_visit
            && self.search_params.puct_var_exploration > 0.0
            && child_visits > 0
        {
            let utility_sq_avg = child.stats.utility_sq_avg.load(Ordering::Acquire);
            let variance = (utility_sq_avg - utility_avg * utility_avg).max(0.0);
            let child_stdev = variance.sqrt();
            let factor = 1.0
                + self.search_params.puct_var_exploration
                    * (child_stdev / self.search_params.cpuct_utility_stdev_prior - 1.0);
            effective_explore_scaling *= factor.clamp(0.25, 4.0);
        }

        explore_selection_value_raw(
            effective_explore_scaling,
            nn_policy_prob,
            child_weight,
            child_utility,
            node.next_pla,
        )
    }

    fn get_new_explore_selection_value(
        &self,
        node: &SearchNode,
        policy_probs: &[f32],
        move_loc: Loc,
        explore_scaling: f64,
        _total_child_weight: f64,
        fpu_value: f64,
        _parent_utility: f64,
        _parent_weight_per_visit: f64,
        is_during_search: bool,
        is_root: bool,
        rand: Option<&mut Rand>,
    ) -> f64 {
        let move_pos = self.get_pos(move_loc) as usize;
        if move_pos >= policy_probs.len() {
            return Self::POLICY_ILLEGAL_SELECTION_VALUE;
        }
        let mut nn_policy_prob = policy_probs[move_pos] as f64;
        if nn_policy_prob < 0.0 {
            return Self::POLICY_ILLEGAL_SELECTION_VALUE;
        }

        let mut child_utility = fpu_value;

        if is_during_search && is_root && self.search_params.wide_root_noise > 0.0 {
            self.maybe_apply_wide_root_noise(&mut nn_policy_prob, &mut child_utility, rand, node);
        }

        explore_selection_value_raw(
            explore_scaling,
            nn_policy_prob,
            0.0,
            child_utility,
            node.next_pla,
        )
    }

    fn get_reduced_play_selection_weight(
        &self,
        node: &SearchNode,
        policy_probs: &[f32],
        child: &SearchNode,
        move_loc: Loc,
        explore_scaling: f64,
        edge_visits: i64,
        best_child_explore_selection_value: f64,
    ) -> f64 {
        assert!(
            self.root_node
                .as_deref()
                .map_or(false, |root| std::ptr::eq(node, root)),
            "get_reduced_play_selection_weight is only valid at the root"
        );

        let move_pos = self.get_pos(move_loc) as usize;
        if move_pos >= policy_probs.len() {
            return 0.0;
        }
        let nn_policy_prob = policy_probs[move_pos] as f64;

        let child_visits = child.stats.visits.load(Ordering::Acquire);
        let score_mean_avg = child.stats.score_mean_avg.load(Ordering::Acquire);
        let score_mean_sq_avg = child.stats.score_mean_sq_avg.load(Ordering::Acquire);
        let utility_avg = child.stats.utility_avg.load(Ordering::Acquire);
        let child_weight = child.stats.get_child_weight(edge_visits);

        if child_visits <= 0 || child_weight <= 0.0 {
            return 0.0;
        }

        let ending_score_bonus = self.get_ending_white_score_bonus(node, move_loc);
        let mut child_utility = utility_avg;
        if ending_score_bonus != 0.0 {
            child_utility +=
                self.get_score_utility_diff(score_mean_avg, score_mean_sq_avg, ending_score_bonus);
        }

        let desired_weight = explore_selection_value_inverse_raw(
            best_child_explore_selection_value,
            explore_scaling,
            nn_policy_prob,
            child_utility,
            node.next_pla,
        );

        if child_weight > desired_weight {
            desired_weight
        } else {
            child_weight
        }
    }

    fn get_fpu_value_for_children_assume_visited(
        &self,
        node: &SearchNode,
        pla: Player,
        is_root: bool,
        policy_prob_mass_visited: f64,
        parent_utility: &mut f64,
        parent_weight_per_visit: &mut f64,
        parent_utility_stdev_factor: &mut f64,
    ) -> f64 {
        let visits = node.stats.visits.load(Ordering::Acquire);
        let weight_sum = node.stats.weight_sum.load(Ordering::Acquire);
        let utility_avg = node.stats.utility_avg.load(Ordering::Acquire);
        let utility_sq_avg = node.stats.utility_sq_avg.load(Ordering::Acquire);

        assert!(visits > 0);
        assert!(weight_sum > 0.0);

        *parent_weight_per_visit = weight_sum / visits as f64;
        *parent_utility = utility_avg;

        let variance_prior = self.search_params.cpuct_utility_stdev_prior
            * self.search_params.cpuct_utility_stdev_prior;
        let variance_prior_weight = self.search_params.cpuct_utility_stdev_prior_weight;
        let parent_utility_stdev = if visits <= 0 || weight_sum <= 1.0 {
            self.search_params.cpuct_utility_stdev_prior
        } else {
            let utility_sq = *parent_utility * *parent_utility;
            let mut utility_sq_avg_local = utility_sq_avg;
            if utility_sq_avg_local < utility_sq {
                utility_sq_avg_local = utility_sq;
            }
            (((utility_sq + variance_prior) * variance_prior_weight
                + utility_sq_avg_local * weight_sum)
                / (variance_prior_weight + weight_sum - 1.0)
                - utility_sq)
                .max(0.0)
                .sqrt()
        };

        *parent_utility_stdev_factor = 1.0
            + self.search_params.cpuct_utility_stdev_scale
                * (parent_utility_stdev / self.search_params.cpuct_utility_stdev_prior - 1.0);

        let mut parent_utility_for_fpu = *parent_utility;
        if self.search_params.fpu_parent_weight_by_visited_policy {
            let avg_weight = policy_prob_mass_visited
                .powf(self.search_params.fpu_parent_weight_by_visited_policy_pow)
                .min(1.0);
            if let Some(nn_output) = node.get_nn_output() {
                parent_utility_for_fpu = avg_weight * *parent_utility
                    + (1.0 - avg_weight) * self.get_utility_from_nn(&nn_output);
            }
        } else if self.search_params.fpu_parent_weight > 0.0 {
            if let Some(nn_output) = node.get_nn_output() {
                parent_utility_for_fpu = self.search_params.fpu_parent_weight
                    * self.get_utility_from_nn(&nn_output)
                    + (1.0 - self.search_params.fpu_parent_weight) * *parent_utility;
            }
        }

        let fpu_reduction_max = if is_root {
            self.search_params.root_fpu_reduction_max
        } else {
            self.search_params.fpu_reduction_max
        };
        let fpu_loss_prop = if is_root {
            self.search_params.root_fpu_loss_prop
        } else {
            self.search_params.fpu_loss_prop
        };
        let utility_radius = self.search_params.win_loss_utility_factor
            + self.search_params.static_score_utility_factor
            + self.search_params.dynamic_score_utility_factor;

        let reduction = fpu_reduction_max * policy_prob_mass_visited.sqrt();
        let mut fpu_value = if pla == P_WHITE {
            parent_utility_for_fpu - reduction
        } else {
            parent_utility_for_fpu + reduction
        };
        let loss_value = if pla == P_WHITE {
            -utility_radius
        } else {
            utility_radius
        };
        fpu_value = fpu_value + (loss_value - fpu_value) * fpu_loss_prop;
        fpu_value
    }

    #[allow(clippy::too_many_arguments)]
    fn select_best_child_to_descend(
        &self,
        thread: &mut SearchThread,
        node: &SearchNode,
        node_state: SearchNodeState,
        num_children_found: &mut i32,
        best_child_idx: &mut i32,
        best_child_move_loc: &mut Loc,
        count_edge_visit: &mut bool,
        is_root: bool,
    ) {
        assert_eq!(thread.pla, node.next_pla);

        let mut max_selection_value = Self::POLICY_ILLEGAL_SELECTION_VALUE;
        *best_child_idx = -1;
        *best_child_move_loc = NULL_LOC;
        *count_edge_visit = true;

        let children = node.get_children_with_state(node_state);
        let children_capacity = children.get_capacity();

        let mut policy_prob_mass_visited = 0.0;
        let mut max_child_weight = 0.0;
        let mut total_child_weight = 0.0;
        let mut total_child_edge_visits = 0_i64;

        let nn_output = node
            .get_nn_output()
            .expect("select_best_child_to_descend called on a node without nn output");
        let policy_probs = nn_output.get_policy_probs_maybe_noised();

        for i in 0..children_capacity {
            let child_pointer = children.get(i);
            let child = match child_pointer.get_if_allocated_relaxed() {
                Some(c) => c,
                None => break,
            };
            let move_loc = child_pointer.get_move_loc_relaxed();
            let move_pos = self.get_pos(move_loc) as usize;
            if move_pos >= policy_probs.len() {
                continue;
            }
            let nn_policy_prob = policy_probs[move_pos];
            if nn_policy_prob < 0.0 {
                continue;
            }
            policy_prob_mass_visited += nn_policy_prob as f64;

            let edge_visits = child_pointer.get_edge_visits();
            let child_weight = child.stats.get_child_weight(edge_visits);

            total_child_weight += child_weight;
            if child_weight > max_child_weight {
                max_child_weight = child_weight;
            }
            total_child_edge_visits += edge_visits;
        }

        let mut use_human_sl = false;
        // Only materialized when a human SL policy actually replaces the net
        // policy; the common path avoids the extra Arc refcount round trip.
        let mut active_output: Option<Arc<NNOutput>> = None;
        if let Some(human_eval) = self.human_evaluator {
            if self.search_params.human_sl_profile.initialized
                || !human_eval.requires_sgf_metadata()
            {
                if let Some(human_output) = node.get_human_output() {
                    let (weightless_prob, weightful_prob) = if is_root {
                        (
                            self.search_params.human_sl_root_explore_prob_weightless,
                            self.search_params.human_sl_root_explore_prob_weightful,
                        )
                    } else if thread.pla == self.root_pla {
                        (
                            self.search_params.human_sl_pla_explore_prob_weightless,
                            self.search_params.human_sl_pla_explore_prob_weightful,
                        )
                    } else {
                        (
                            self.search_params.human_sl_opp_explore_prob_weightless,
                            self.search_params.human_sl_opp_explore_prob_weightful,
                        )
                    };

                    let total_human_prob = weightless_prob + weightful_prob;
                    if total_human_prob > 0.0 {
                        let r = thread.rand.next_double();
                        if r < weightless_prob {
                            use_human_sl = true;
                            *count_edge_visit = false;
                        } else if r < total_human_prob {
                            use_human_sl = true;
                        }
                    }

                    if use_human_sl {
                        active_output = Some(human_output);
                        policy_prob_mass_visited = 0.0;
                        let human_policy_probs = active_output
                            .as_ref()
                            .unwrap()
                            .get_policy_probs_maybe_noised();
                        for i in 0..children_capacity {
                            let child_pointer = children.get(i);
                            if child_pointer.get_if_allocated_relaxed().is_none() {
                                break;
                            }
                            let move_loc = child_pointer.get_move_loc_relaxed();
                            let move_pos = self.get_pos(move_loc) as usize;
                            if move_pos >= human_policy_probs.len() {
                                continue;
                            }
                            let nn_policy_prob = human_policy_probs[move_pos];
                            if nn_policy_prob < 0.0 {
                                continue;
                            }
                            policy_prob_mass_visited += nn_policy_prob as f64;
                        }
                    }
                }
            }
        }

        debug_assert!(policy_prob_mass_visited <= 1.0001);

        if !*count_edge_visit {
            total_child_weight = 0.0;
            max_child_weight = 0.0;
            for i in 0..children_capacity {
                let child_pointer = children.get(i);
                let child = match child_pointer.get_if_allocated_relaxed() {
                    Some(c) => c,
                    None => break,
                };
                let child_weight = child.stats.weight_sum.load(Ordering::Acquire);
                total_child_weight += child_weight;
                if child_weight > max_child_weight {
                    max_child_weight = child_weight;
                }
            }
        }

        let mut parent_utility = 0.0;
        let mut parent_weight_per_visit = 0.0;
        let mut parent_utility_stdev_factor = 0.0;
        let fpu_value = self.get_fpu_value_for_children_assume_visited(
            node,
            thread.pla,
            is_root,
            policy_prob_mass_visited,
            &mut parent_utility,
            &mut parent_weight_per_visit,
            &mut parent_utility_stdev_factor,
        );

        let mut poses_with_child_buf = [false; nn_pos::MAX_NN_POLICY_SIZE];
        let anti_mirror = self.search_params.anti_mirror
            && self.mirroring_pla != C_EMPTY
            && self.is_mirroring_since_search_start(&thread.history, 0);

        let explore_scaling = if use_human_sl {
            self.get_explore_scaling_human(total_child_weight, parent_utility_stdev_factor)
        } else {
            self.get_explore_scaling(total_child_weight, parent_utility_stdev_factor)
        };

        *num_children_found = 0;
        let active_policy_probs = match active_output.as_ref() {
            Some(human) => human.get_policy_probs_maybe_noised(),
            None => nn_output.get_policy_probs_maybe_noised(),
        };
        for i in 0..children_capacity {
            let child_pointer = children.get(i);
            let child = match child_pointer.get_if_allocated_relaxed() {
                Some(c) => c,
                None => break,
            };
            *num_children_found += 1;
            let child_edge_visits = child_pointer.get_edge_visits();
            let move_loc = child_pointer.get_move_loc_relaxed();
            let is_during_search = true;
            let selection_value = self.get_explore_selection_value_of_child(
                node,
                active_policy_probs,
                child,
                move_loc,
                explore_scaling,
                total_child_weight,
                child_edge_visits,
                fpu_value,
                parent_utility,
                parent_weight_per_visit,
                is_during_search,
                is_root,
                false,
                max_child_weight,
                *count_edge_visit,
                None,
                Some(&mut thread.rand),
            );
            if selection_value > max_selection_value {
                max_selection_value = selection_value;
                *best_child_idx = i as i32;
                *best_child_move_loc = move_loc;
            }

            let move_pos = self.get_pos(move_loc) as usize;
            if move_pos < poses_with_child_buf.len() {
                poses_with_child_buf[move_pos] = true;
            }
        }

        let avoid_move_until_by_loc = if thread.pla == P_BLACK {
            &self.avoid_move_until_by_loc_black
        } else {
            &self.avoid_move_until_by_loc_white
        };

        if self.search_params.use_eval_cache
            && self.search_params.use_graph_search
            && node.eval_cache_entry.is_some()
            && self.mirroring_pla == C_EMPTY
            && !node.force_non_terminal
        {
            let entry = node.eval_cache_entry.as_ref().unwrap();
            for (&move_loc, eval) in &entry.first_explore_evals {
                let move_pos = self.get_pos(move_loc) as usize;
                if move_pos < poses_with_child_buf.len() && poses_with_child_buf[move_pos] {
                    continue;
                }
                if move_pos >= active_policy_probs.len() {
                    continue;
                }
                let nn_policy_prob = active_policy_probs[move_pos];
                if nn_policy_prob < 0.0 {
                    continue;
                }

                if is_root {
                    debug_assert_eq!(thread.board.pos_hash, self.root_board.pos_hash);
                    debug_assert_eq!(thread.pla, self.root_pla);
                    if !self.is_allowed_root_move(move_loc) {
                        continue;
                    }
                }
                if !avoid_move_until_by_loc.is_empty() {
                    if (move_loc as usize) < avoid_move_until_by_loc.len() {
                        let until_depth = avoid_move_until_by_loc[move_loc as usize];
                        let depth_since_root = thread
                            .history
                            .move_history
                            .len()
                            .saturating_sub(self.root_history.move_history.len());
                        if (depth_since_root as i32) < until_depth {
                            continue;
                        }
                    }
                }

                let cache_avg_utility = self.get_result_utility(eval.avg_win_loss as f64, 0.0)
                    + self.get_score_utility(
                        eval.avg_score_mean as f64,
                        eval.avg_score_mean as f64 * eval.avg_score_mean as f64,
                    );

                let selection_value = self.get_new_explore_selection_value(
                    node,
                    active_policy_probs,
                    move_loc,
                    explore_scaling,
                    total_child_weight,
                    cache_avg_utility,
                    parent_utility,
                    parent_weight_per_visit,
                    true,
                    is_root,
                    Some(&mut thread.rand),
                );
                if selection_value > max_selection_value {
                    max_selection_value = selection_value;
                    *best_child_idx = *num_children_found;
                    *best_child_move_loc = move_loc;
                }
            }
        }

        let mut best_new_move_loc = NULL_LOC;
        let mut best_new_nn_policy_prob = -1.0_f32;
        let policy_size = self.policy_size as i32;

        // At the root with symmetry pruning, the allowed-move set is restricted
        // to one representative per symmetry orbit (isAllowedRootMove ->
        // rootSymDupLoc, mirroring C++ `Search::isAllowedRootMove` in
        // searchhelpers.cpp). If the policy head is not perfectly symmetric,
        // the policy argmax can be a duplicate whose orbit representative has
        // far lower policy - even lower than pass. With very few visits the
        // first playout then goes to that low-policy representative (or pass)
        // and the low-visit chosen move is wrong, whereas C++ with a correctly
        // computed policy explores the policy peak first.
        // So when selecting the *first* new move at the root, we exempt the
        // global policy argmax from the symmetry-duplicate restriction. This
        // is a no-op whenever the argmax is itself a representative (the
        // normal, near-symmetric-policy case) and only kicks in to make the
        // first visit go to the true policy peak.
        let mut sym_argmax_exempt_loc = NULL_LOC;
        if is_root && self.search_params.root_symmetry_pruning {
            let mut best_exempt_prob = -1.0_f32;
            for move_pos in 0..policy_size {
                let move_pos_usize = move_pos as usize;
                if move_pos_usize >= active_policy_probs.len() {
                    continue;
                }
                let mut nn_policy_prob = active_policy_probs[move_pos_usize];
                if nn_policy_prob < 0.0 {
                    continue;
                }
                let move_loc = nn_pos::pos_to_loc(
                    move_pos,
                    thread.board.x_size,
                    thread.board.y_size,
                    self.nn_x_len,
                    self.nn_y_len,
                );
                if move_loc == NULL_LOC {
                    continue;
                }
                // Only the symmetry-duplicate restriction may be bypassed for
                // the policy argmax; all other root restrictions still apply.
                if !(self.is_allowed_root_move(move_loc)
                    || self.root_sym_dup_loc[move_loc as usize])
                {
                    continue;
                }
                if !avoid_move_until_by_loc.is_empty() {
                    if (move_loc as usize) < avoid_move_until_by_loc.len() {
                        let until_depth = avoid_move_until_by_loc[move_loc as usize];
                        let depth_since_root = thread
                            .history
                            .move_history
                            .len()
                            .saturating_sub(self.root_history.move_history.len());
                        if (depth_since_root as i32) < until_depth {
                            continue;
                        }
                    }
                }
                if anti_mirror {
                    let mut prob_f = nn_policy_prob;
                    self.maybe_apply_anti_mirror_policy(
                        &mut prob_f,
                        move_loc,
                        active_policy_probs,
                        node.next_pla,
                        Some(thread),
                    );
                    nn_policy_prob = prob_f;
                }
                if nn_policy_prob > best_exempt_prob {
                    best_exempt_prob = nn_policy_prob;
                    sym_argmax_exempt_loc = move_loc;
                }
            }
        }
        for move_pos in 0..policy_size {
            let move_pos_usize = move_pos as usize;
            if move_pos_usize < poses_with_child_buf.len() && poses_with_child_buf[move_pos_usize] {
                continue;
            }
            if move_pos_usize >= active_policy_probs.len() {
                continue;
            }
            let nn_policy_prob = active_policy_probs[move_pos_usize];
            if nn_policy_prob < 0.0 {
                continue;
            }

            let move_loc = nn_pos::pos_to_loc(
                move_pos,
                thread.board.x_size,
                thread.board.y_size,
                self.nn_x_len,
                self.nn_y_len,
            );
            if move_loc == NULL_LOC {
                continue;
            }

            if is_root {
                debug_assert_eq!(thread.board.pos_hash, self.root_board.pos_hash);
                debug_assert_eq!(thread.pla, self.root_pla);
                // Exempt the global policy argmax from the symmetry-duplicate
                // restriction so the first visit explores the policy peak
                // (see the comment above where `sym_argmax_exempt_loc` is set).
                if move_loc != sym_argmax_exempt_loc && !self.is_allowed_root_move(move_loc) {
                    continue;
                }
            }
            if !avoid_move_until_by_loc.is_empty() {
                if (move_loc as usize) < avoid_move_until_by_loc.len() {
                    let until_depth = avoid_move_until_by_loc[move_loc as usize];
                    let depth_since_root = thread
                        .history
                        .move_history
                        .len()
                        .saturating_sub(self.root_history.move_history.len());
                    if (depth_since_root as i32) < until_depth {
                        continue;
                    }
                }
            }

            if anti_mirror {
                let mut prob_f = nn_policy_prob;
                self.maybe_apply_anti_mirror_policy(
                    &mut prob_f,
                    move_loc,
                    active_policy_probs,
                    node.next_pla,
                    Some(thread),
                );
                if prob_f > best_new_nn_policy_prob {
                    best_new_nn_policy_prob = prob_f;
                    best_new_move_loc = move_loc;
                }
            } else if nn_policy_prob > best_new_nn_policy_prob {
                best_new_nn_policy_prob = nn_policy_prob;
                best_new_move_loc = move_loc;
            }
        }

        if best_new_move_loc != NULL_LOC {
            let selection_value = self.get_new_explore_selection_value(
                node,
                active_policy_probs,
                best_new_move_loc,
                explore_scaling,
                total_child_weight,
                fpu_value,
                parent_utility,
                parent_weight_per_visit,
                true,
                is_root,
                Some(&mut thread.rand),
            );
            if selection_value > max_selection_value {
                max_selection_value = selection_value;
                *best_child_idx = *num_children_found;
                *best_child_move_loc = best_new_move_loc;
            }
        }

        if total_child_edge_visits >= 2
            && self.search_params.enable_more_passing_hacks
            && thread
                .history
                .pass_would_end_phase(&thread.board, thread.pla)
            && avoid_move_until_by_loc.is_empty()
        {
            let mut has_pass_move = false;
            let mut has_non_pass_move = false;
            for i in 0..children_capacity {
                let child_pointer = children.get(i);
                if child_pointer.get_if_allocated_relaxed().is_none() {
                    break;
                }
                let move_loc = child_pointer.get_move_loc_relaxed();
                if move_loc == PASS_LOC {
                    has_pass_move = true;
                } else {
                    has_non_pass_move = true;
                }
            }
            if !has_pass_move
                && *best_child_move_loc != PASS_LOC
                && *best_child_move_loc != NULL_LOC
            {
                *best_child_idx = *num_children_found;
                *best_child_move_loc = PASS_LOC;
                *count_edge_visit = false;
                thread.should_count_playout = false;
            } else if !has_non_pass_move
                && *best_child_move_loc == PASS_LOC
                && best_new_move_loc != PASS_LOC
                && best_new_move_loc != NULL_LOC
            {
                *best_child_idx = *num_children_found;
                *best_child_move_loc = best_new_move_loc;
                *count_edge_visit = false;
                thread.should_count_playout = false;
            }
        }
    }
}

// Update of node values during search
impl<'a> Search<'a> {
    #[allow(clippy::too_many_arguments)]
    fn add_leaf_value(
        &self,
        node: &mut SearchNode,
        win_loss_value: f64,
        no_result_value: f64,
        score_mean: f64,
        score_mean_sq: f64,
        lead: f64,
        weight: f64,
        is_terminal: bool,
        assume_no_existing_weight: bool,
    ) {
        let mut utility = self.get_result_utility(win_loss_value, no_result_value)
            + self.get_score_utility(score_mean, score_mean_sq);

        if self.search_params.subtree_value_bias_factor != 0.0
            && !is_terminal
            && node.subtree_value_bias_table_entry.is_some()
        {
            let entry = node.subtree_value_bias_table_entry.as_ref().unwrap();
            let (delta_utility_sum, weight_sum) = entry.read();
            if weight_sum > 0.001 {
                utility += self.search_params.subtree_value_bias_factor * delta_utility_sum
                    / weight_sum;
            }
        }

        utility += self.get_pattern_bonus(node.pattern_bonus_hash, get_opp(node.next_pla));

        let utility_sq = utility * utility;
        let weight_sq = weight * weight;

        if assume_no_existing_weight {
            while node.stats_lock.swap(true, Ordering::Acquire) {}
            node.stats
                .win_loss_value_avg
                .store(win_loss_value, Ordering::Release);
            node.stats
                .no_result_value_avg
                .store(no_result_value, Ordering::Release);
            node.stats
                .score_mean_avg
                .store(score_mean, Ordering::Release);
            node.stats
                .score_mean_sq_avg
                .store(score_mean_sq, Ordering::Release);
            node.stats.lead_avg.store(lead, Ordering::Release);
            node.stats.utility_avg.store(utility, Ordering::Release);
            node.stats
                .utility_sq_avg
                .store(utility_sq, Ordering::Release);
            node.stats.weight_sq_sum.store(weight_sq, Ordering::Release);
            node.stats.weight_sum.store(weight, Ordering::Release);
            node.stats.visits.fetch_add(1, Ordering::Release);
            node.stats_lock.store(false, Ordering::Release);
        } else {
            while node.stats_lock.swap(true, Ordering::Acquire) {}
            let old_weight_sum = node.stats.weight_sum.load(Ordering::Relaxed);
            let new_weight_sum = old_weight_sum + weight;

            node.stats.win_loss_value_avg.store(
                (node.stats.win_loss_value_avg.load(Ordering::Relaxed) * old_weight_sum
                    + win_loss_value * weight)
                    / new_weight_sum,
                Ordering::Release,
            );
            node.stats.no_result_value_avg.store(
                (node.stats.no_result_value_avg.load(Ordering::Relaxed) * old_weight_sum
                    + no_result_value * weight)
                    / new_weight_sum,
                Ordering::Release,
            );
            node.stats.score_mean_avg.store(
                (node.stats.score_mean_avg.load(Ordering::Relaxed) * old_weight_sum
                    + score_mean * weight)
                    / new_weight_sum,
                Ordering::Release,
            );
            node.stats.score_mean_sq_avg.store(
                (node.stats.score_mean_sq_avg.load(Ordering::Relaxed) * old_weight_sum
                    + score_mean_sq * weight)
                    / new_weight_sum,
                Ordering::Release,
            );
            node.stats.lead_avg.store(
                (node.stats.lead_avg.load(Ordering::Relaxed) * old_weight_sum + lead * weight)
                    / new_weight_sum,
                Ordering::Release,
            );
            node.stats.utility_avg.store(
                (node.stats.utility_avg.load(Ordering::Relaxed) * old_weight_sum
                    + utility * weight)
                    / new_weight_sum,
                Ordering::Release,
            );
            node.stats.utility_sq_avg.store(
                (node.stats.utility_sq_avg.load(Ordering::Relaxed) * old_weight_sum
                    + utility_sq * weight)
                    / new_weight_sum,
                Ordering::Release,
            );
            node.stats.weight_sq_sum.store(
                node.stats.weight_sq_sum.load(Ordering::Relaxed) + weight_sq,
                Ordering::Release,
            );
            node.stats
                .weight_sum
                .store(new_weight_sum, Ordering::Release);
            node.stats.visits.fetch_add(1, Ordering::Release);
            node.stats_lock.store(false, Ordering::Release);
        }
    }

    fn add_current_nn_output_as_leaf_value(
        &self,
        node: &mut SearchNode,
        assume_no_existing_weight: bool,
    ) {
        let nn_output = node
            .get_nn_output()
            .expect("add_current_nn_output_as_leaf_value called on a node without nn output");

        let win_loss_value = nn_output.white_win_prob as f64 - nn_output.white_loss_prob as f64;
        let no_result_value = nn_output.white_no_result_prob as f64;
        let score_mean = nn_output.white_score_mean as f64;
        let score_mean_sq = nn_output.white_score_mean_sq as f64;
        let lead = nn_output.white_lead as f64;

        let is_root_node = self
            .root_node
            .as_deref()
            .map_or(false, |root| std::ptr::eq(root, &*node));
        if self.search_params.use_eval_cache
            && self.search_params.use_graph_search
            && node.eval_cache_entry.is_some()
            && self.mirroring_pla == C_EMPTY
            && !is_root_node
            && !node.force_non_terminal
        {
            let this_node_visits_for_cache = 1;
            if let Some(entry) = &node.eval_cache_entry {
                let mut wl = win_loss_value;
                let mut nr = no_result_value;
                let mut sm = score_mean;
                let mut smsq = score_mean_sq;
                let mut l = lead;
                self.adjust_evals_from_cache_helper(
                    entry,
                    this_node_visits_for_cache,
                    &mut wl,
                    &mut nr,
                    &mut sm,
                    &mut smsq,
                    &mut l,
                    None,
                );
            }
        }

        let weight = self.compute_weight_from_nn_output(Some(&nn_output));
        self.add_leaf_value(
            node,
            win_loss_value,
            no_result_value,
            score_mean,
            score_mean_sq,
            lead,
            weight,
            false,
            assume_no_existing_weight,
        );
    }

    fn compute_weight_from_nn_output(&self, nn_output: Option<&NNOutput>) -> f64 {
        let nn_output = match nn_output {
            Some(o) => o,
            None => return 1.0,
        };
        self.compute_weight_from_nn_output_some(nn_output)
    }

    fn compute_weight_from_nn_output_some(&self, nn_output: &NNOutput) -> f64 {
        if !self.search_params.use_uncertainty {
            return 1.0;
        }
        if let Some(eval) = self.nn_evaluator {
            if !eval.supports_shortterm_error() {
                return 1.0;
            }
        }

        let score_mean = nn_output.white_score_mean as f64;
        let utility_uncertainty_wl =
            self.search_params.win_loss_utility_factor * nn_output.shortterm_winloss_error as f64;
        let utility_uncertainty_score = self.get_approx_score_utility_derivative(score_mean)
            * nn_output.shortterm_score_error as f64;
        let utility_uncertainty = utility_uncertainty_wl + utility_uncertainty_score;

        let powered_uncertainty = if self.search_params.uncertainty_exponent == 1.0 {
            utility_uncertainty
        } else if self.search_params.uncertainty_exponent == 0.5 {
            utility_uncertainty.sqrt()
        } else {
            utility_uncertainty.powf(self.search_params.uncertainty_exponent)
        };

        let baseline_uncertainty =
            self.search_params.uncertainty_coeff / self.search_params.uncertainty_max_weight;
        self.search_params.uncertainty_coeff / (powered_uncertainty + baseline_uncertainty)
    }

    fn update_stats_after_playout(
        &self,
        node: &mut SearchNode,
        thread: &mut SearchThread,
        is_root: bool,
    ) {
        let old_dirty_counter = node.dirty_counter.fetch_add(1, Ordering::AcqRel);
        assert!(old_dirty_counter >= 0);
        if old_dirty_counter > 0 {
            return;
        }
        let mut num_visits_completed = 1;
        loop {
            self.recompute_node_stats(node, thread, num_visits_completed, is_root);
            let old_dirty_counter = node
                .dirty_counter
                .fetch_sub(num_visits_completed, Ordering::AcqRel);
            let new_dirty_counter = old_dirty_counter - num_visits_completed;
            if new_dirty_counter <= 0 {
                assert_eq!(new_dirty_counter, 0);
                break;
            }
            num_visits_completed = new_dirty_counter;
        }
    }

    fn recompute_node_stats(
        &self,
        node: &mut SearchNode,
        thread: &mut SearchThread,
        num_visits_to_add: i32,
        is_root: bool,
    ) {
        thread.stats_buf.clear();
        let children = node.get_children();
        let children_capacity = children.get_capacity();

        let mut orig_total_child_weight = 0.0;
        let mut this_node_visits = 1_i64;
        for i in 0..children_capacity {
            let child_ptr = children.get(i);
            let child = match child_ptr.get_if_allocated() {
                Some(c) => c,
                None => break,
            };
            let move_loc = child_ptr.get_move_loc_relaxed();
            let edge_visits = child_ptr.get_edge_visits_relaxed();
            let stats = NodeStats::from_atomic(&child.stats);
            if stats.visits <= 0 || stats.weight_sum <= 0.0 || edge_visits <= 0 {
                continue;
            }

            let child_utility = stats.utility_avg;
            let self_utility = if node.next_pla == P_WHITE {
                child_utility
            } else {
                -child_utility
            };
            let weight_adjusted = stats.get_child_weight(edge_visits);
            thread.stats_buf.push(MoreNodeStats {
                stats,
                self_utility,
                weight_adjusted,
                prev_move_loc: move_loc,
            });

            orig_total_child_weight += weight_adjusted;
            this_node_visits += edge_visits;
        }
        let num_good_children = thread.stats_buf.len();

        let mut current_total_child_weight = orig_total_child_weight;

        if self.search_params.use_noise_pruning
            && num_good_children > 0
            && !(self.search_params.anti_mirror && self.mirroring_pla != C_EMPTY)
        {
            let nn_output = node
                .get_nn_output()
                .expect("recompute_node_stats called on a node without nn output");
            let policy_probs = nn_output.get_policy_probs_maybe_noised();
            let mut policy_probs_buf = [0.0f64; nn_pos::MAX_NN_POLICY_SIZE];
            for i in 0..num_good_children {
                let pos = self.get_pos(thread.stats_buf[i].prev_move_loc) as usize;
                policy_probs_buf[i] =
                    (policy_probs.get(pos).copied().unwrap_or(0.0f32) as f64).max(1e-30);
            }
            current_total_child_weight = self.prune_noise_weight(
                &mut thread.stats_buf[..num_good_children],
                num_good_children as i32,
                current_total_child_weight,
                &policy_probs_buf[..num_good_children],
            );
        }

        let mut amount_to_subtract = 0.0;
        let mut amount_to_prune = 0.0;
        if is_root && self.search_params.root_noise_enabled && !self.search_params.use_noise_pruning
        {
            let max_child_weight = thread.stats_buf[..num_good_children]
                .iter()
                .map(|s| s.weight_adjusted)
                .fold(0.0, f64::max);
            amount_to_subtract = self
                .search_params
                .chosen_move_subtract
                .min(max_child_weight / 64.0);
            amount_to_prune = self
                .search_params
                .chosen_move_prune
                .min(max_child_weight / 64.0);
        }

        self.downweight_bad_children_and_normalize_weight(
            num_good_children as i32,
            current_total_child_weight,
            current_total_child_weight,
            amount_to_subtract,
            amount_to_prune,
            &mut thread.stats_buf[..num_good_children],
            &mut thread.stdevs_buf,
        );

        let mut win_loss_value_sum = 0.0;
        let mut no_result_value_sum = 0.0;
        let mut score_mean_sum = 0.0;
        let mut score_mean_sq_sum = 0.0;
        let mut lead_sum = 0.0;
        let mut utility_sum = 0.0;
        let mut utility_sq_sum = 0.0;
        let mut weight_sq_sum = 0.0;
        let mut weight_sum = current_total_child_weight;
        for i in 0..num_good_children {
            let stats = &thread.stats_buf[i].stats;
            let desired_weight = thread.stats_buf[i].weight_adjusted;
            let weight_scaling = desired_weight / stats.weight_sum;

            win_loss_value_sum += desired_weight * stats.win_loss_value_avg;
            no_result_value_sum += desired_weight * stats.no_result_value_avg;
            score_mean_sum += desired_weight * stats.score_mean_avg;
            score_mean_sq_sum += desired_weight * stats.score_mean_sq_avg;
            lead_sum += desired_weight * stats.lead_avg;
            utility_sum += desired_weight * stats.utility_avg;
            utility_sq_sum += desired_weight * stats.utility_sq_avg;
            weight_sq_sum += weight_scaling * weight_scaling * stats.weight_sq_sum;
        }

        {
            // Borrow without refcount churn — this runs per node backup.
            // Same lifetime argument as get_ending_white_score_bonus. The
            // scalars are copied out so the borrow ends before the
            // subtree-bias bookkeeping below mutates node fields.
            let (win_prob, loss_prob, no_result_prob, score_mean, score_mean_sq, lead, weight) = {
                let nn_output = unsafe { node.nn_output_ref() }
                    .expect("recompute_node_stats called on a node without nn output");
                (
                    nn_output.white_win_prob as f64,
                    nn_output.white_loss_prob as f64,
                    nn_output.white_no_result_prob as f64,
                    nn_output.white_score_mean as f64,
                    nn_output.white_score_mean_sq as f64,
                    nn_output.white_lead as f64,
                    self.compute_weight_from_nn_output_some(nn_output),
                )
            };
            let mut utility = self.get_result_utility(win_prob - loss_prob, no_result_prob)
                + self.get_score_utility(score_mean, score_mean_sq);

            if self.search_params.subtree_value_bias_factor != 0.0
                && node.subtree_value_bias_table_entry.is_some()
            {
                let entry = node.subtree_value_bias_table_entry.as_ref().unwrap();
                let bias_factor = self.search_params.subtree_value_bias_factor;
                if current_total_child_weight > 1e-10 {
                    let utility_children = utility_sum / current_total_child_weight;
                    let subtree_value_bias_weight = orig_total_child_weight
                        .powf(self.search_params.subtree_value_bias_weight_exponent);
                    let subtree_value_bias_delta_sum =
                        (utility_children - utility) * subtree_value_bias_weight;

                    // Move this node's contribution in the shared entry to the
                    // current (utility_children - utility) delta, atomically.
                    // Mirrors C++ searchupdatehelpers.cpp: the node stores its
                    // last recorded contribution so recompute is idempotent.
                    let (new_entry_delta_utility_sum, new_entry_weight_sum) = entry.update(
                        subtree_value_bias_delta_sum
                            - node.last_subtree_value_bias_delta_sum,
                        subtree_value_bias_weight - node.last_subtree_value_bias_weight,
                    );
                    node.last_subtree_value_bias_delta_sum = subtree_value_bias_delta_sum;
                    node.last_subtree_value_bias_weight = subtree_value_bias_weight;

                    if new_entry_weight_sum > 0.001 {
                        utility += bias_factor * new_entry_delta_utility_sum
                            / new_entry_weight_sum;
                    }
                } else {
                    let (new_entry_delta_utility_sum, new_entry_weight_sum) = entry.read();
                    if new_entry_weight_sum > 0.001 {
                        utility += bias_factor * new_entry_delta_utility_sum
                            / new_entry_weight_sum;
                    }
                }
            }

            let win_loss_value_sum_tail = (win_prob - loss_prob) * weight;
            win_loss_value_sum += win_loss_value_sum_tail;
            no_result_value_sum += no_result_prob * weight;
            score_mean_sum += score_mean * weight;
            score_mean_sq_sum += score_mean_sq * weight;
            lead_sum += lead * weight;
            utility_sum += utility * weight;
            utility_sq_sum += utility * utility * weight;
            weight_sq_sum += weight * weight;
            weight_sum += weight;
        }

        let mut win_loss_value_avg = win_loss_value_sum / weight_sum;
        let mut no_result_value_avg = no_result_value_sum / weight_sum;
        let mut score_mean_avg = score_mean_sum / weight_sum;
        let mut score_mean_sq_avg = score_mean_sq_sum / weight_sum;
        let mut lead_avg = lead_sum / weight_sum;
        let mut utility_avg = utility_sum / weight_sum;
        let mut utility_sq_avg = utility_sq_sum / weight_sum;

        let old_utility_avg = utility_avg;
        utility_avg += self.get_pattern_bonus(node.pattern_bonus_hash, get_opp(node.next_pla));

        let is_root_node = self
            .root_node
            .as_deref()
            .map_or(false, |root| std::ptr::eq(root, &*node));
        if self.search_params.use_eval_cache
            && self.search_params.use_graph_search
            && node.eval_cache_entry.is_some()
            && self.mirroring_pla == C_EMPTY
            && !is_root_node
            && !node.force_non_terminal
        {
            if let Some(entry) = &node.eval_cache_entry {
                self.adjust_evals_from_cache_helper(
                    entry,
                    this_node_visits,
                    &mut win_loss_value_avg,
                    &mut no_result_value_avg,
                    &mut score_mean_avg,
                    &mut score_mean_sq_avg,
                    &mut lead_avg,
                    Some(&mut utility_avg),
                );
            }
        }
        utility_sq_avg =
            utility_sq_avg + (utility_avg * utility_avg - old_utility_avg * old_utility_avg);

        while node.stats_lock.swap(true, Ordering::Acquire) {}
        node.stats
            .win_loss_value_avg
            .store(win_loss_value_avg, Ordering::Release);
        node.stats
            .no_result_value_avg
            .store(no_result_value_avg, Ordering::Release);
        node.stats
            .score_mean_avg
            .store(score_mean_avg, Ordering::Release);
        node.stats
            .score_mean_sq_avg
            .store(score_mean_sq_avg, Ordering::Release);
        node.stats.lead_avg.store(lead_avg, Ordering::Release);
        node.stats.utility_avg.store(utility_avg, Ordering::Release);
        node.stats
            .utility_sq_avg
            .store(utility_sq_avg, Ordering::Release);
        node.stats
            .weight_sq_sum
            .store(weight_sq_sum, Ordering::Release);
        node.stats.weight_sum.store(weight_sum, Ordering::Release);
        node.stats
            .visits
            .fetch_add(num_visits_to_add as i64, Ordering::Release);
        node.stats_lock.store(false, Ordering::Release);
    }

    fn adjust_evals_from_cache_helper(
        &self,
        eval_cache_entry: &Arc<crate::eval_cache::EvalCacheEntry>,
        this_node_visits: i64,
        win_loss_value_avg: &mut f64,
        no_result_value_avg: &mut f64,
        score_mean_avg: &mut f64,
        score_mean_sq_avg: &mut f64,
        lead_avg: &mut f64,
        utility_avg: Option<&mut f64>,
    ) {
        let cache_avg_win_loss = eval_cache_entry.avg_win_loss as f64;
        let cache_avg_no_result = eval_cache_entry.avg_no_result as f64;
        let cache_avg_score_mean = eval_cache_entry.avg_score_mean as f64;
        let cache_avg_lead = eval_cache_entry.avg_lead as f64;
        let mut cache_weight = eval_cache_entry.cache_weight as f64;

        if cache_weight > self.search_params.eval_cache_min_visits as f64 {
            cache_weight = (self.search_params.eval_cache_min_visits as f64 * cache_weight).sqrt();
        }
        let visits_to_cache_ratio = this_node_visits as f64 / cache_weight;
        let cache_frac = 1.0
            / (1.0
                + 3.0
                    * visits_to_cache_ratio
                    * (1.0 + 2.0 * visits_to_cache_ratio * visits_to_cache_ratio));

        *win_loss_value_avg += cache_frac * (cache_avg_win_loss - *win_loss_value_avg);
        *no_result_value_avg += cache_frac * (cache_avg_no_result - *no_result_value_avg);
        let old_score_mean_avg = *score_mean_avg;
        *score_mean_avg += cache_frac * (cache_avg_score_mean - *score_mean_avg);
        *score_mean_sq_avg = (*score_mean_sq_avg - old_score_mean_avg * old_score_mean_avg
            + *score_mean_avg * *score_mean_avg)
            .max(0.0);
        *lead_avg += cache_frac * (cache_avg_lead - *lead_avg);

        if let Some(utility) = utility_avg {
            let cache_score_stdev_sq = (*score_mean_sq_avg - *score_mean_avg * *score_mean_avg
                + cache_avg_score_mean * cache_avg_score_mean)
                .max(0.0);
            let cache_avg_utility = self
                .get_result_utility(cache_avg_win_loss, cache_avg_no_result)
                + self.get_score_utility(cache_avg_score_mean, cache_score_stdev_sq);
            *utility += cache_frac * (cache_avg_utility - *utility);
        }
    }

    fn downweight_bad_children_and_normalize_weight(
        &self,
        num_children: i32,
        mut current_total_weight: f64,
        desired_total_weight: f64,
        amount_to_subtract: f64,
        amount_to_prune: f64,
        stats_buf: &mut [MoreNodeStats],
        stdevs_scratch: &mut Vec<f64>,
    ) {
        let n = num_children as usize;
        if n == 0 || current_total_weight <= 0.0 {
            return;
        }

        if self.search_params.value_weight_exponent == 0.0 || self.mirroring_pla != C_EMPTY {
            for i in 0..n {
                if stats_buf[i].weight_adjusted < amount_to_prune {
                    current_total_weight -= stats_buf[i].weight_adjusted;
                    stats_buf[i].weight_adjusted = 0.0;
                    continue;
                }
                let new_weight = stats_buf[i].weight_adjusted - amount_to_subtract;
                if new_weight <= 0.0 {
                    current_total_weight -= stats_buf[i].weight_adjusted;
                    stats_buf[i].weight_adjusted = 0.0;
                } else {
                    current_total_weight -= amount_to_subtract;
                    stats_buf[i].weight_adjusted = new_weight;
                }
            }

            if (current_total_weight - desired_total_weight).abs() > 1e-50 {
                let factor = desired_total_weight / current_total_weight;
                for i in 0..n {
                    stats_buf[i].weight_adjusted *= factor;
                }
            }
            return;
        }

        let distribution = self
            .value_weight_distribution
            .as_ref()
            .expect("value_weight_distribution required when value_weight_exponent != 0");

        // Thread-owned scratch, resized to the actual child count: no heap
        // alloc in the steady state (capacity is retained across backups) and
        // no fixed 362-slot zeroing (which measurably hurt multithreaded
        // throughput through extra memory traffic).
        stdevs_scratch.clear();
        stdevs_scratch.resize(n, 0.0);
        let mut simple_value_sum = 0.0;
        for i in 0..n {
            let num_visits = stats_buf[i].stats.visits;
            assert!(num_visits >= 0);
            if num_visits == 0 {
                continue;
            }
            let weight = stats_buf[i].weight_adjusted;
            let precision = 1.5 * weight.sqrt();
            const MIN_VARIANCE: f64 = 0.00000001;
            stdevs_scratch[i] = (MIN_VARIANCE + 1.0 / precision).sqrt();
            simple_value_sum += stats_buf[i].self_utility * weight;
        }

        let simple_value = simple_value_sum / current_total_weight;

        let mut total_new_unnorm_weight = 0.0;
        for i in 0..n {
            if stats_buf[i].stats.visits == 0 {
                continue;
            }

            if stats_buf[i].weight_adjusted < amount_to_prune {
                current_total_weight -= stats_buf[i].weight_adjusted;
                stats_buf[i].weight_adjusted = 0.0;
                continue;
            }
            let new_weight = stats_buf[i].weight_adjusted - amount_to_subtract;
            if new_weight <= 0.0 {
                current_total_weight -= stats_buf[i].weight_adjusted;
                stats_buf[i].weight_adjusted = 0.0;
            } else {
                current_total_weight -= amount_to_subtract;
                stats_buf[i].weight_adjusted = new_weight;
            }

            let z = (stats_buf[i].self_utility - simple_value) / stdevs_scratch[i];
            let p = distribution.get_cdf(z) + 0.0001;
            stats_buf[i].weight_adjusted *= p.powf(self.search_params.value_weight_exponent);
            total_new_unnorm_weight += stats_buf[i].weight_adjusted;
        }

        assert!(total_new_unnorm_weight > 0.0);
        let factor = desired_total_weight / total_new_unnorm_weight;
        for i in 0..n {
            stats_buf[i].weight_adjusted *= factor;
        }
    }

    fn prune_noise_weight(
        &self,
        stats_buf: &mut [MoreNodeStats],
        num_children: i32,
        total_child_weight: f64,
        policy_probs_buf: &[f64],
    ) -> f64 {
        let n = num_children as usize;
        if n <= 1 || total_child_weight <= 0.00001 {
            return total_child_weight;
        }

        let mut utility_sum_so_far = 0.0;
        let mut weight_sum_so_far = 0.0;
        let mut raw_policy_sum_so_far = 0.0;
        for i in 0..n {
            let utility = stats_buf[i].self_utility;
            let old_weight = stats_buf[i].weight_adjusted;
            let raw_policy = policy_probs_buf[i];

            let mut new_weight = old_weight;
            if weight_sum_so_far > 0.0 && raw_policy_sum_so_far > 0.0 {
                let avg_utility_so_far = utility_sum_so_far / weight_sum_so_far;
                let utility_gap = avg_utility_so_far - utility;
                if utility_gap > 0.0 {
                    let weight_share_from_raw_policy =
                        weight_sum_so_far * raw_policy / raw_policy_sum_so_far;
                    let lenient_weight_share_from_raw_policy = 2.0 * weight_share_from_raw_policy;
                    if old_weight > lenient_weight_share_from_raw_policy {
                        let excess_weight = old_weight - lenient_weight_share_from_raw_policy;
                        let mut weight_to_subtract = excess_weight
                            * (1.0
                                - (-utility_gap / self.search_params.noise_prune_utility_scale)
                                    .exp());
                        if weight_to_subtract > self.search_params.noise_pruning_cap {
                            weight_to_subtract = self.search_params.noise_pruning_cap;
                        }
                        new_weight = old_weight - weight_to_subtract;
                        stats_buf[i].weight_adjusted = new_weight;
                    }
                }
            }
            utility_sum_so_far += utility * new_weight;
            weight_sum_so_far += new_weight;
            raw_policy_sum_so_far += raw_policy;
        }
        weight_sum_so_far
    }
}

// Allocation, search clearing, and garbage collection
impl<'a> Search<'a> {
    fn create_mutex_idx_for_node(&self, thread: &mut SearchThread) -> u32 {
        let num_mutexes = self
            .mutex_pool
            .as_ref()
            .expect("mutex pool not initialized")
            .num_mutexes();
        thread.rand.next_u32() % num_mutexes
    }

    fn get_eval_cache_key(&self, graph_hash: Hash128) -> Hash128 {
        graph_hash ^ self.eval_cache_params_hash
    }

    fn allocate_or_find_node(
        &self,
        thread: &mut SearchThread,
        next_pla: Player,
        best_child_move_loc: Loc,
        force_non_terminal: bool,
        graph_hash: Hash128,
    ) -> Option<&mut SearchNode> {
        let node_table = self.node_table.as_ref()?;

        let mut child_hash = if self.search_params.use_graph_search {
            let mut h = graph_hash;
            if force_non_terminal {
                h ^= Self::FORCE_NON_TERMINAL_HASH;
            }
            h
        } else {
            thread.board.pos_hash ^ Hash128::new(thread.rand.next_u64(), thread.rand.next_u64())
        };

        let node_table_idx = node_table.get_index(child_hash.hash0);
        let _pool_guard = node_table.mutex_pool.get_with_modulo(node_table_idx).lock();
        let mut map = node_table.entries[node_table_idx as usize].lock();

        loop {
            if let Some(&existing_ptr) = map.get(&child_hash) {
                if existing_ptr.is_null() {
                    // Defensive: treat null as missing.
                } else {
                    let existing = unsafe { &mut *existing_ptr };
                    if existing.next_pla != next_pla {
                        child_hash = thread.board.pos_hash
                            ^ Hash128::new(thread.rand.next_u64(), thread.rand.next_u64());
                        continue;
                    }
                    return Some(existing);
                }
            }

            let mutex_idx = self.create_mutex_idx_for_node(thread);
            let mut child = Box::new(SearchNode::new(
                next_pla,
                force_non_terminal,
                mutex_idx,
                graph_hash,
            ));

            if self.search_params.subtree_value_bias_factor != 0.0
                && self.subtree_value_bias_table.is_some()
            {
                if thread.history.move_history.len() >= 2 {
                    let prev_move_loc =
                        thread.history.move_history[thread.history.move_history.len() - 2].loc;
                    if prev_move_loc != NULL_LOC && best_child_move_loc != PASS_LOC {
                        child.subtree_value_bias_table_entry =
                            Some(self.subtree_value_bias_table.as_ref().unwrap().get(
                                get_opp(thread.pla),
                                prev_move_loc,
                                best_child_move_loc,
                                thread.history.get_recent_board(1),
                            ));
                    }
                }
            }

            if self.search_params.use_eval_cache
                && self.search_params.use_graph_search
                && self.eval_cache.is_some()
                && self.mirroring_pla == C_EMPTY
            {
                let key = self.get_eval_cache_key(child.graph_hash);
                child.eval_cache_entry = self.eval_cache.as_ref().unwrap().find(key);
            }

            if self.pattern_bonus_table.is_some() {
                child.pattern_bonus_hash = self.pattern_bonus_table.as_ref().unwrap().get_hash(
                    get_opp(thread.pla),
                    best_child_move_loc,
                    thread.history.get_recent_board(1),
                );
            }

            let child_ptr = Box::into_raw(child);
            map.insert(child_hash, child_ptr);
            return Some(unsafe { &mut *child_ptr });
        }
    }

    fn clear_old_nn_outputs(&mut self) {
        for ptr in &self.old_nn_outputs_to_clean_up {
            if !ptr.is_null() {
                unsafe {
                    drop(Box::from_raw(*ptr));
                }
            }
        }
        self.old_nn_outputs_to_clean_up.clear();
    }

    fn transfer_old_nn_outputs(&mut self, thread: &mut SearchThread) {
        self.old_nn_outputs_to_clean_up
            .append(&mut thread.old_nn_outputs_to_clean_up);
    }

    fn remove_subtree_value_bias(&self, node: Option<&SearchNode>) {
        // Subtract this node's last recorded contribution from the shared bias
        // entry, mirroring the C++ SearchNode destructor: only the freeProp
        // fraction is withdrawn, and the entry reference is dropped so the
        // table's refcount GC can reclaim it. Must run before the node is
        // freed, on paths where the subtree bias table outlives the node
        // (e.g. subtree reuse on move, stale-node sweeps).
        let node = match node {
            Some(n) => n,
            None => return,
        };
        if let Some(entry) = node.subtree_value_bias_table_entry.as_ref() {
            let free_prop = self.search_params.subtree_value_bias_free_prop;
            entry.update(
                -node.last_subtree_value_bias_delta_sum * free_prop,
                -node.last_subtree_value_bias_weight * free_prop,
            );
        }
    }

    fn delete_all_old_or_all_new_table_nodes_and_subtree_value_bias_multithreaded(
        &mut self,
        old: bool,
    ) {
        let node_table = match self.node_table.as_ref() {
            Some(t) => t,
            None => return,
        };
        let search_node_age = self.search_node_age;

        for map_mutex in &node_table.entries {
            let mut map = map_mutex.lock();
            let mut to_remove = Vec::new();
            for (&hash, &ptr) in map.iter() {
                if ptr.is_null() {
                    continue;
                }
                let node_age = unsafe { &*ptr }.node_age.load(Ordering::Acquire);
                let matches = old == (node_age < search_node_age);
                if matches {
                    self.remove_subtree_value_bias(Some(unsafe { &*ptr }));
                    to_remove.push(hash);
                }
            }
            for hash in to_remove {
                if let Some(ptr) = map.remove(&hash) {
                    if !ptr.is_null() {
                        unsafe {
                            drop(Box::from_raw(ptr));
                        }
                    }
                }
            }
        }
    }

    fn delete_all_table_nodes_multithreaded(&mut self) {
        let node_table = match self.node_table.as_mut() {
            Some(t) => t,
            None => return,
        };

        for map_mutex in &node_table.entries {
            let mut map = map_mutex.lock();
            let mut ptrs: Vec<*mut SearchNode> = Vec::with_capacity(map.len());
            for (_, ptr) in map.drain() {
                if !ptr.is_null() {
                    ptrs.push(ptr);
                }
            }
            drop(map);
            for ptr in ptrs {
                unsafe {
                    let node = &*ptr;
                    if let Some(entry) = node.subtree_value_bias_table_entry.as_ref() {
                        entry.update(
                            -node.last_subtree_value_bias_delta_sum,
                            -node.last_subtree_value_bias_weight,
                        );
                    }
                    drop(Box::from_raw(ptr));
                }
            }
        }
    }
}

// Initialization and core search logic
impl<'a> Drop for Search<'a> {
    fn drop(&mut self) {
        // Mirrors upstream ~Search(): the node table holds non-owning raw
        // pointers, so without this every dropped Search leaks the whole tree.
        self.clear_search();
    }
}

impl<'a> Search<'a> {
    fn compute_root_values(&mut self) {
        let mut root_safe_area = vec![C_EMPTY; MAX_ARR_SIZE as usize].into_boxed_slice();
        let non_pass_alive_stones = false;
        let safe_big_territories = false;
        let unsafe_big_territories = false;
        self.root_board.calculate_area(
            &mut root_safe_area,
            non_pass_alive_stones,
            safe_big_territories,
            unsafe_big_territories,
            self.root_history.rules.multi_stone_suicide_legal,
        );
        self.root_safe_area = Some(root_safe_area);

        let mut expected_score = 0.0;
        let mut found_expected_score_from_tree = false;
        if let Some(root) = self.root_node.as_deref() {
            let num_visits = root.stats.visits.load(Ordering::Acquire);
            let weight_sum = root.stats.weight_sum.load(Ordering::Acquire);
            let score_mean_avg = root.stats.score_mean_avg.load(Ordering::Acquire);
            if num_visits > 0 && weight_sum > 0.0 {
                found_expected_score_from_tree = true;
                expected_score = score_mean_avg;
            }
        }

        if !found_expected_score_from_tree {
            let mut nn_result_buf = NNResultBuf::new();
            let include_owner_map = true;
            self.compute_root_nn_evaluation(&mut nn_result_buf, include_owner_map);
            if let Some(ref arc) = nn_result_buf.result {
                expected_score = arc.white_score_mean as f64;
            }
        }

        self.recent_score_center =
            expected_score * (1.0 - self.search_params.dynamic_score_center_zero_weight);
        let cap = (self.root_board.x_size as f64 * self.root_board.y_size as f64).sqrt()
            * self.search_params.dynamic_score_center_scale;
        if self.recent_score_center > expected_score + cap {
            self.recent_score_center = expected_score + cap;
        }
        if self.recent_score_center < expected_score - cap {
            self.recent_score_center = expected_score - cap;
        }

        let opponent_was_mirroring_pla = self.mirroring_pla;
        self.update_mirroring();
        if opponent_was_mirroring_pla != self.mirroring_pla {
            self.clear_search();
            self.subtree_value_bias_table = None;
        }
    }

    fn recursively_recompute_stats(&mut self, node: &mut SearchNode) {
        struct Frame {
            node: *mut SearchNode,
            next_child: usize,
        }

        // One scratch thread for the whole walk: constructing a SearchThread
        // (MD5+SHA-256 seeding, ~35KB board/history clone) per interior node
        // costs milliseconds-to-seconds on large reused trees, and the stats
        // recompute only needs its stats_buf scratch space.
        let mut scratch_thread = SearchThread::new(0, self);

        let root_ptr = self
            .root_node
            .as_deref()
            .map_or(std::ptr::null_mut(), |root| {
                root as *const SearchNode as *mut SearchNode
            });
        let root_frame = Frame {
            node: node as *mut SearchNode,
            next_child: 0,
        };
        let root_frame_ptr = root_frame.node;
        let mut stack = vec![root_frame];
        let mut on_path = std::collections::HashSet::new();
        let mut completed = std::collections::HashSet::new();
        on_path.insert(root_frame_ptr);

        while let Some(mut frame) = stack.pop() {
            let node_ptr = frame.node;
            let children = unsafe { (*node_ptr).get_children() };
            let children_capacity = children.get_capacity();
            let mut descended = false;

            while frame.next_child < children_capacity {
                let child_index = frame.next_child;
                frame.next_child += 1;
                let child_ptr = children.get(child_index).get_raw_ptr();
                if child_ptr.is_null() {
                    frame.next_child = children_capacity;
                    break;
                }
                if completed.contains(&child_ptr) || on_path.contains(&child_ptr) {
                    continue;
                }

                on_path.insert(child_ptr);
                stack.push(frame);
                stack.push(Frame {
                    node: child_ptr,
                    next_child: 0,
                });
                descended = true;
                break;
            }

            if descended {
                continue;
            }

            let is_root = node_ptr == root_ptr;
            // Upstream drives this walk through applyRecursivelyPostOrder-
            // Mulithreaded, which marks every visited node with the current
            // searchNodeAge; the mark is what keeps the retained subtree alive
            // through the subsequent delete-all-old sweep after root filtering.
            unsafe {
                (*node_ptr)
                    .node_age
                    .store(self.search_node_age, Ordering::Release);
            }
            let found_any_children =
                children_capacity > 0 && !children.get(0).get_raw_ptr().is_null();
            let node = unsafe { &mut *node_ptr };
            if !found_any_children {
                let num_visits = node.stats.visits.load(Ordering::Acquire);
                let weight_sum = node.stats.weight_sum.load(Ordering::Acquire);
                assert!(
                    weight_sum > 0.0 || num_visits == 0,
                    "leaf node with visits but no weight must be the root"
                );
                if weight_sum > 0.0 {
                    let win_loss_value_avg = node.stats.win_loss_value_avg.load(Ordering::Acquire);
                    let no_result_value_avg =
                        node.stats.no_result_value_avg.load(Ordering::Acquire);
                    let score_mean_avg = node.stats.score_mean_avg.load(Ordering::Acquire);
                    let score_mean_sq_avg = node.stats.score_mean_sq_avg.load(Ordering::Acquire);
                    let mut new_utility = self
                        .get_result_utility(win_loss_value_avg, no_result_value_avg)
                        + self.get_score_utility(score_mean_avg, score_mean_sq_avg);
                    new_utility +=
                        self.get_pattern_bonus(node.pattern_bonus_hash, get_opp(node.next_pla));
                    let new_utility_sq = new_utility * new_utility;

                    while node.stats_lock.swap(true, Ordering::Acquire) {}
                    node.stats.utility_avg.store(new_utility, Ordering::Release);
                    node.stats
                        .utility_sq_avg
                        .store(new_utility_sq, Ordering::Release);
                    node.stats_lock.store(false, Ordering::Release);
                }
            } else {
                self.recompute_node_stats(node, &mut scratch_thread, 0, is_root);
            }

            on_path.remove(&node_ptr);
            completed.insert(node_ptr);
        }
    }

    fn recursively_record_eval_cache(&mut self, node: &mut SearchNode) {
        fn recurse(search: &mut Search, node: &mut SearchNode) {
            let node_ptr = node as *mut SearchNode;
            let children = node.get_children();
            let capacity = children.get_capacity();
            for i in 0..capacity {
                let child_pointer = children.get(i);
                let child_ptr = child_pointer.get_raw_ptr();
                if child_ptr.is_null() {
                    break;
                }
                unsafe {
                    recurse(search, &mut *child_ptr);
                }
            }

            let num_visits = node.stats.visits.load(Ordering::Acquire);
            let min_visits = search.search_params.eval_cache_min_visits;
            if num_visits >= min_visits && !node.force_non_terminal {
                let is_root = search
                    .root_node
                    .as_deref()
                    .map_or(false, |root| std::ptr::eq(root, unsafe { &*node_ptr }));
                if let Some(eval_cache) = search.eval_cache.as_ref() {
                    let key = search.get_eval_cache_key(node.graph_hash);
                    eval_cache.update(key, node, min_visits, is_root);
                }
            }
        }

        recurse(self, node);
    }

    fn playout_descend(
        &self,
        thread: &mut SearchThread,
        node: &mut SearchNode,
        is_root: bool,
    ) -> bool {
        if thread.history.is_game_finished && !node.force_non_terminal {
            if let Some(evaluator) = self.nn_evaluator {
                evaluator.wait_for_next_nn_eval_if_any();
            }

            let uncertainty_max_weight = if self.search_params.use_uncertainty
                && self
                    .nn_evaluator
                    .map_or(false, |e| e.supports_shortterm_error())
            {
                self.search_params.uncertainty_max_weight
            } else {
                1.0
            };

            if thread.history.is_no_result {
                self.add_leaf_value(
                    node,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    uncertainty_max_weight,
                    true,
                    false,
                );
            } else {
                let win_loss_value =
                    2.0 * score_value::white_wins_of_winner(
                        thread.history.winner,
                        self.search_params.draw_equivalent_wins_for_white,
                    ) - 1.0;
                let score_mean = score_value::white_score_draw_adjust(
                    f64::from(thread.history.final_white_minus_black_score),
                    self.search_params.draw_equivalent_wins_for_white,
                    &thread.history,
                );
                let score_mean_sq = score_value::white_score_mean_sq_of_score_gridded(
                    f64::from(thread.history.final_white_minus_black_score),
                    self.search_params.draw_equivalent_wins_for_white,
                );
                let lead = score_mean;
                self.add_leaf_value(
                    node,
                    win_loss_value,
                    0.0,
                    score_mean,
                    score_mean_sq,
                    lead,
                    uncertainty_max_weight,
                    true,
                    false,
                );
            }
            return true;
        }

        let mut node_state = node.state.load(Ordering::Acquire);
        if node_state == STATE_UNEVALUATED {
            let suc = self.init_node_nn_output(thread, node, is_root, false, false);
            if !suc {
                thread.should_count_playout = false;
                return false;
            }
            let cas_result = node.state.compare_exchange(
                node_state,
                STATE_EVALUATING,
                Ordering::SeqCst,
                Ordering::SeqCst,
            );
            if cas_result.is_err() {
                thread.should_count_playout = false;
                return false;
            }
            node.initialize_children();
            node.state.store(STATE_EXPANDED0, Ordering::SeqCst);
            return true;
        } else if node_state == STATE_EVALUATING {
            thread.should_count_playout = false;
            return false;
        }

        assert!(node_state >= STATE_EXPANDED0);
        self.maybe_recompute_existing_nn_output(thread, node, is_root);

        let node_ptr = node as *mut SearchNode;
        let mut num_children_found: i32 = 0;
        let mut best_child_idx: i32 = -1;
        let mut best_child_move_loc: Loc = NULL_LOC;
        let mut count_edge_visit: bool = true;
        let mut child_ptr: *mut SearchNode = std::ptr::null_mut();

        loop {
            {
                let this = &*self;
                let node_ref = unsafe { &*node_ptr };
                this.select_best_child_to_descend(
                    thread,
                    node_ref,
                    node_state,
                    &mut num_children_found,
                    &mut best_child_idx,
                    &mut best_child_move_loc,
                    &mut count_edge_visit,
                    is_root,
                );
            }

            if best_child_idx >= 0
                && !thread
                    .history
                    .is_legal(&thread.board, best_child_move_loc, thread.pla)
            {
                let nn_hash = if let Some(nn_output) = node.get_nn_output() {
                    let h = nn_output.nn_hash;
                    if !thread.illegal_move_hashes.contains(&h) {
                        if let Some(logger) = self.logger {
                            logger.write(&format!(
                                "WARNING: Chosen move not legal so about to regenerate nn output, nnhash={}",
                                h
                            ));
                        }
                    }
                    Some(h)
                } else {
                    None
                };

                self.init_node_nn_output(thread, node, is_root, true, true);

                if let Some(h) = nn_hash {
                    if !thread.illegal_move_hashes.contains(&h) {
                        thread.illegal_move_hashes.insert(h);
                        if let Some(logger) = self.logger {
                            logger.write(&format!(
                                "WARNING: Chosen move not legal so regenerated nn output, nnhash={}",
                                h
                            ));
                        }
                    }
                }
                return true;
            }

            if best_child_idx <= -1 {
                self.add_current_nn_output_as_leaf_value(node, false);
                return true;
            }

            let best_child_idx_usize = best_child_idx as usize;

            if best_child_idx >= num_children_found {
                debug_assert_eq!(best_child_idx, num_children_found);
                debug_assert!((best_child_idx as usize) < nn_pos::MAX_NN_POLICY_SIZE);
                let target = (num_children_found + 1) as usize;
                let expanded =
                    node.maybe_expand_children_capacity_for_new_child(&mut node_state, target);
                if !expanded {
                    std::thread::yield_now();
                    node_state = node.state.load(Ordering::Acquire);
                    continue;
                }

                let children = node.get_children_with_state(node_state);
                debug_assert!(children.get_capacity() > best_child_idx_usize);

                let can_force_non_terminal_due_to_friendly_pass = best_child_move_loc == PASS_LOC
                    && thread
                        .history
                        .should_suppress_end_game_from_friendly_pass(&thread.board, thread.pla);

                thread.history.make_board_move_assume_legal(
                    &mut thread.board,
                    best_child_move_loc,
                    thread.pla,
                );
                thread.pla = get_opp(thread.pla);
                if self.search_params.use_graph_search {
                    thread.graph_hash = graph_hash::get_graph_hash(
                        thread.graph_hash,
                        &thread.history,
                        thread.pla,
                        self.search_params.graph_search_rep_bound,
                        self.search_params.draw_equivalent_wins_for_white,
                    );
                }

                let force_non_terminal = best_child_move_loc == PASS_LOC
                    && thread.history.is_game_finished
                    && ((self.search_params.conservative_pass && is_root)
                        || can_force_non_terminal_due_to_friendly_pass);

                let new_child = self
                    .allocate_or_find_node(
                        thread,
                        thread.pla,
                        best_child_move_loc,
                        force_non_terminal,
                        thread.graph_hash,
                    )
                    .expect("allocate_or_find_node returned null");
                new_child.virtual_losses.fetch_add(1, Ordering::Release);

                child_ptr = new_child as *mut SearchNode;
                {
                    let _lock = self
                        .mutex_pool
                        .as_ref()
                        .unwrap()
                        .get_with_modulo(unsafe { &*node_ptr }.mutex_idx)
                        .lock();
                    let children = unsafe { &*node_ptr }.get_children_with_state(node_state);
                    let child_pointer = children.get(best_child_idx_usize);
                    if child_pointer.get_if_allocated_relaxed().is_none() {
                        child_pointer.set_move_loc_relaxed(best_child_move_loc);
                        child_pointer.store(child_ptr);
                    } else {
                        unsafe {
                            (*child_ptr).virtual_losses.fetch_sub(1, Ordering::Release);
                        }
                        thread.should_count_playout = false;
                        return false;
                    }
                }

                if count_edge_visit
                    && self.maybe_catch_up_edge_visits(
                        thread,
                        unsafe { &mut *node_ptr },
                        Some(unsafe { &*child_ptr }),
                        node_state,
                        best_child_idx,
                    )
                {
                    self.update_stats_after_playout(unsafe { &mut *node_ptr }, thread, is_root);
                    unsafe {
                        (*child_ptr).virtual_losses.fetch_sub(1, Ordering::Release);
                    }
                    return true;
                }
            } else {
                let children = unsafe { &*node_ptr }.get_children_with_state(node_state);
                let child_pointer = children.get(best_child_idx_usize);
                let existing_ptr = child_pointer.get_raw_ptr();
                debug_assert!(!existing_ptr.is_null());
                unsafe {
                    (*existing_ptr)
                        .virtual_losses
                        .fetch_add(1, Ordering::Release);
                }

                if count_edge_visit
                    && self.maybe_catch_up_edge_visits(
                        thread,
                        unsafe { &mut *node_ptr },
                        Some(unsafe { &*existing_ptr }),
                        node_state,
                        best_child_idx,
                    )
                {
                    self.update_stats_after_playout(unsafe { &mut *node_ptr }, thread, is_root);
                    unsafe {
                        (*existing_ptr)
                            .virtual_losses
                            .fetch_sub(1, Ordering::Release);
                    }
                    return true;
                }

                thread.history.make_board_move_assume_legal(
                    &mut thread.board,
                    best_child_move_loc,
                    thread.pla,
                );
                thread.pla = get_opp(thread.pla);
                if self.search_params.use_graph_search {
                    thread.graph_hash = graph_hash::get_graph_hash(
                        thread.graph_hash,
                        &thread.history,
                        thread.pla,
                        self.search_params.graph_search_rep_bound,
                        self.search_params.draw_equivalent_wins_for_white,
                    );
                }

                child_ptr = existing_ptr;
            }

            break;
        }

        debug_assert!(!child_ptr.is_null(), "child was selected but not set");
        // Cycle detection is only meaningful when nodes are shared via the
        // graph table; without graph search every node is tree-unique, so skip
        // the hash-set round trip entirely (mirrors C++, which only populates
        // graphPath when graph search dedups a revisit).
        let inserted = if self.search_params.use_graph_search {
            thread.graph_path.insert(child_ptr as *const SearchNode)
        } else {
            true
        };
        if !inserted {
            if count_edge_visit {
                let children = unsafe { &*node_ptr }.get_children_with_state(node_state);
                children.get(best_child_idx as usize).add_edge_visits(1);
                self.update_stats_after_playout(unsafe { &mut *node_ptr }, thread, is_root);
            }
            unsafe {
                (*child_ptr).virtual_losses.fetch_sub(1, Ordering::Release);
            }
            return count_edge_visit;
        }

        let mut should_update_child_ancestors =
            self.playout_descend(thread, unsafe { &mut *child_ptr }, false);
        should_update_child_ancestors = should_update_child_ancestors && count_edge_visit;
        if should_update_child_ancestors {
            node_state = unsafe { &*node_ptr }.state.load(Ordering::Acquire);
            let children = unsafe { &*node_ptr }.get_children_with_state(node_state);
            children.get(best_child_idx as usize).add_edge_visits(1);
            self.update_stats_after_playout(unsafe { &mut *node_ptr }, thread, is_root);
        }
        unsafe {
            (*child_ptr).virtual_losses.fetch_sub(1, Ordering::Release);
        }

        should_update_child_ancestors
    }

    fn maybe_catch_up_edge_visits(
        &self,
        thread: &mut SearchThread,
        node: &mut SearchNode,
        child: Option<&SearchNode>,
        node_state: SearchNodeState,
        best_child_idx: i32,
    ) -> bool {
        let child = match child {
            Some(c) => c,
            None => return false,
        };

        let children = node.get_children_with_state(node_state);
        let child_pointer = children.get(best_child_idx as usize);

        let child_visits = child.stats.visits.load(Ordering::Acquire);
        let mut edge_visits = child_pointer.get_edge_visits();

        if self.search_params.graph_search_catch_up_leak_prob > 0.0
            && edge_visits < child_visits
            && thread
                .rand
                .next_bool(self.search_params.graph_search_catch_up_leak_prob)
        {
            return false;
        }

        const NUM_TO_ADD: i64 = 1;
        loop {
            if edge_visits >= child_visits {
                return false;
            }
            let desired = edge_visits + NUM_TO_ADD;
            if child_pointer.compare_exchange_weak_edge_visits(&mut edge_visits, desired) {
                return true;
            }
        }
    }
}

// Private helpers for search results and analysis
impl<'a> Search<'a> {
    #[allow(clippy::too_many_arguments)]
    fn get_play_selection_values_internal(
        &self,
        node: &SearchNode,
        locs: &mut Vec<Loc>,
        play_selection_values: &mut Vec<f64>,
        mut ret_visit_counts: Option<&mut Vec<f64>>,
        scale_max_to_at_least: f64,
        allow_direct_policy_moves: bool,
        always_compute_lcb: bool,
        never_use_lcb: bool,
        lcb_buf: &mut [f64],
        radius_buf: &mut [f64],
    ) -> bool {
        locs.clear();
        play_selection_values.clear();
        if let Some(ref mut v) = ret_visit_counts {
            v.clear();
        }

        let nn_output = node.get_nn_output();
        let policy_probs: &[f32] = nn_output
            .as_ref()
            .map_or(&[], |o| o.get_policy_probs_maybe_noised());

        let mut total_child_weight = 0.0;
        let suppress_pass = self.should_suppress_pass(Some(node));
        let is_root = self
            .root_node
            .as_deref()
            .map_or(false, |root| std::ptr::eq(node, root));

        let children = node.get_children();
        let children_capacity = children.get_capacity();
        for i in 0..children_capacity {
            let child_ptr = children.get(i);
            let child = match child_ptr.get_if_allocated() {
                Some(c) => c,
                None => break,
            };
            let move_loc = child_ptr.get_move_loc_relaxed();
            let edge_visits = child_ptr.get_edge_visits();
            let child_weight = child.stats.get_child_weight(edge_visits);

            locs.push(move_loc);
            total_child_weight += child_weight;

            let pos = self.get_pos(move_loc) as usize;
            let policy_prob = policy_probs.get(pos).copied().unwrap_or(0.0f32);
            if (suppress_pass && move_loc == PASS_LOC) || policy_prob < 0.0 {
                play_selection_values.push(0.0);
                if let Some(ref mut v) = ret_visit_counts {
                    v.push(0.0);
                }
            } else {
                play_selection_values.push(child_weight);
                if let Some(ref mut v) = ret_visit_counts {
                    v.push(edge_visits as f64);
                }
            }
        }

        let mut num_children = play_selection_values.len();

        // Find the best child before LCB for pruning.
        let mut non_lcb_best_idx = 0;
        let mut non_lcb_best_child_weight = -1e30;
        {
            let mut max_goodness = -1e30;
            for i in 0..num_children {
                let weight = play_selection_values[i];
                let child_ptr = children.get(i);
                let edge_visits = child_ptr.get_edge_visits();
                let move_loc = child_ptr.get_move_loc_relaxed();
                let pos = self.get_pos(move_loc) as usize;
                let policy_prob = policy_probs.get(pos).copied().unwrap_or(0.0f32) as f64;

                let g = weight * (edge_visits as f64 - 1.0).max(0.0)
                    / (edge_visits as f64).max(1.0)
                    + 2.0 * policy_prob;
                if g > max_goodness {
                    max_goodness = g;
                    non_lcb_best_child_weight = weight;
                    non_lcb_best_idx = i;
                }
            }
        }

        // Possibly reduce weight on children that we spent too many visits on in retrospect.
        if is_root && num_children > 0 {
            let best_child_ptr = children.get(non_lcb_best_idx);
            let best_child = best_child_ptr.get_if_allocated().unwrap();
            let best_child_edge_visits = best_child_ptr.get_edge_visits();
            let best_move_loc = best_child_ptr.get_move_loc_relaxed();

            let mut parent_utility = 0.0;
            let mut parent_weight_per_visit = 0.0;
            let mut parent_utility_stdev_factor = 0.0;
            let fpu_value = self.get_fpu_value_for_children_assume_visited(
                node,
                self.root_pla,
                true,
                1.0,
                &mut parent_utility,
                &mut parent_weight_per_visit,
                &mut parent_utility_stdev_factor,
            );

            let explore_scaling =
                self.get_explore_scaling(total_child_weight, parent_utility_stdev_factor);

            assert!(nn_output.is_some());
            let best_child_explore_selection_value = self.get_explore_selection_value_of_child(
                node,
                policy_probs,
                best_child,
                best_move_loc,
                explore_scaling,
                total_child_weight,
                best_child_edge_visits,
                fpu_value,
                parent_utility,
                parent_weight_per_visit,
                false,
                is_root,
                false,
                non_lcb_best_child_weight,
                true,
                None,
                None,
            );

            for i in 0..num_children {
                let child_ptr = children.get(i);
                let child = child_ptr.get_if_allocated().unwrap();
                let move_loc = child_ptr.get_move_loc_relaxed();
                if suppress_pass && move_loc == PASS_LOC {
                    play_selection_values[i] = 0.0;
                    continue;
                }
                if i != non_lcb_best_idx {
                    let edge_visits = child_ptr.get_edge_visits();
                    let reduced = self.get_reduced_play_selection_weight(
                        node,
                        policy_probs,
                        child,
                        move_loc,
                        explore_scaling,
                        edge_visits,
                        best_child_explore_selection_value,
                    );
                    play_selection_values[i] = reduced.ceil();
                }
            }
        }

        // Now compute play selection values taking into account LCB.
        if !never_use_lcb
            && (always_compute_lcb
                || (self.search_params.use_lcb_for_selection && num_children > 0))
        {
            let mut best_lcb = -1e10;
            let mut best_lcb_index = -1i32;
            for i in 0..num_children {
                let child_ptr = children.get(i);
                let child = child_ptr.get_if_allocated();
                let edge_visits = child_ptr.get_edge_visits();
                let move_loc = child_ptr.get_move_loc_relaxed();
                self.get_self_utility_lcb_and_radius(
                    node,
                    child,
                    edge_visits,
                    move_loc,
                    &mut lcb_buf[i],
                    &mut radius_buf[i],
                );

                let weight = play_selection_values[i];
                if weight > 0.0
                    && weight
                        >= self.search_params.min_visit_prop_for_lcb * non_lcb_best_child_weight
                {
                    if lcb_buf[i] > best_lcb {
                        best_lcb = lcb_buf[i];
                        best_lcb_index = i as i32;
                    }
                }
            }

            if self.search_params.use_lcb_for_selection
                && num_children > 0
                && (if self.search_params.use_non_buggy_lcb {
                    best_lcb_index >= 0
                } else {
                    best_lcb_index > 0
                })
            {
                let best_idx = best_lcb_index as usize;
                let mut adjusted_weight = play_selection_values[best_idx];
                for i in 0..num_children {
                    if i != best_idx {
                        let excess_value = best_lcb - lcb_buf[i];
                        if excess_value < 0.0 {
                            continue;
                        }
                        let radius = radius_buf[i];
                        let radius_factor =
                            (radius + excess_value) / (radius + 0.20 * excess_value);
                        let lbound = radius_factor * radius_factor * play_selection_values[i];
                        if lbound > adjusted_weight {
                            adjusted_weight = lbound;
                        }
                    }
                }
                play_selection_values[best_idx] = adjusted_weight;
            }
        }

        // If we have no children, use the policy net directly at the root.
        if num_children == 0 {
            if nn_output.is_none() || !is_root || !allow_direct_policy_moves {
                return false;
            }

            // Same policy-argmax exemption as in select_best_child_to_descend:
            // at zero visits the direct-policy fallback (C++
            // `Search::getPlaySelectionValues`, searchresults.cpp) should still
            // select the policy peak, not the highest-policy symmetry
            // representative, when the peak happens to be a duplicate.
            let mut sym_argmax_exempt_loc = NULL_LOC;
            if self.search_params.root_symmetry_pruning {
                let mut best_exempt_prob = -1.0f32;
                for pos in 0..self.policy_size as usize {
                    let policy_prob = policy_probs[pos];
                    if policy_prob < 0.0 {
                        continue;
                    }
                    let move_loc = nn_pos::pos_to_loc(
                        pos as i32,
                        self.root_board.x_size,
                        self.root_board.y_size,
                        self.nn_x_len,
                        self.nn_y_len,
                    );
                    if move_loc == NULL_LOC {
                        continue;
                    }
                    // Only the symmetry-duplicate restriction may be bypassed
                    // for the policy argmax; all other root restrictions apply.
                    if !(self.is_allowed_root_move(move_loc)
                        || self.root_sym_dup_loc[move_loc as usize])
                    {
                        continue;
                    }
                    if policy_prob > best_exempt_prob {
                        best_exempt_prob = policy_prob;
                        sym_argmax_exempt_loc = move_loc;
                    }
                }
            }

            let mut obey_allowed_root_move = true;
            loop {
                for pos in 0..self.policy_size as usize {
                    let move_loc = nn_pos::pos_to_loc(
                        pos as i32,
                        self.root_board.x_size,
                        self.root_board.y_size,
                        self.nn_x_len,
                        self.nn_y_len,
                    );
                    let mut policy_prob = policy_probs[pos] as f64;
                    if !self.is_okay_raw_policy_move_at_root(
                        move_loc,
                        policy_prob,
                        obey_allowed_root_move,
                    ) && !(move_loc == sym_argmax_exempt_loc
                        && self.root_sym_dup_loc[move_loc as usize])
                    {
                        continue;
                    }
                    if suppress_pass && move_loc == PASS_LOC {
                        policy_prob = 0.0;
                    }
                    locs.push(move_loc);
                    play_selection_values.push(policy_prob);
                    num_children += 1;
                }
                if num_children == 0 && obey_allowed_root_move {
                    obey_allowed_root_move = false;
                    continue;
                }
                break;
            }
        }

        if num_children == 0 {
            return false;
        }

        let mut max_value = 0.0;
        for &v in play_selection_values.iter() {
            if v > max_value {
                max_value = v;
            }
        }

        if max_value <= 1e-50 {
            for i in 0..num_children {
                let pos = self.get_pos(locs[i]) as usize;
                play_selection_values[i] =
                    (policy_probs.get(pos).copied().unwrap_or(0.0f32) as f64).max(0.0);
            }
            max_value = 0.0;
            for &v in play_selection_values.iter() {
                if v > max_value {
                    max_value = v;
                }
            }
            if max_value <= 1e-50 {
                return false;
            }
        }

        assert!(max_value < 1e40);

        let amount_to_subtract = self
            .search_params
            .chosen_move_subtract
            .min(max_value / 64.0);
        let amount_to_prune = self.search_params.chosen_move_prune.min(max_value / 64.0);
        for i in 0..num_children {
            if play_selection_values[i] < amount_to_prune {
                play_selection_values[i] = 0.0;
            } else {
                play_selection_values[i] -= amount_to_subtract;
                if play_selection_values[i] <= 0.0 {
                    play_selection_values[i] = 0.0;
                }
            }
        }

        // Average in human policy.
        if self.human_evaluator.is_some()
            && (self.search_params.human_sl_profile.initialized
                || !self.human_evaluator.unwrap().requires_sgf_metadata())
            && self.search_params.human_sl_chosen_move_prop > 0.0
        {
            if let Some(human_output) = node.get_human_output() {
                let human_probs = human_output.get_policy_probs_maybe_noised();

                if is_root && allow_direct_policy_moves {
                    let mut locs_set: HashSet<Loc> = locs.iter().copied().collect();
                    for pos in 0..self.policy_size as usize {
                        let move_loc = nn_pos::pos_to_loc(
                            pos as i32,
                            self.root_board.x_size,
                            self.root_board.y_size,
                            self.nn_x_len,
                            self.nn_y_len,
                        );
                        let human_prob = human_probs[pos] as f64;
                        if !self.is_okay_raw_policy_move_at_root(move_loc, human_prob, true) {
                            continue;
                        }
                        if locs_set.contains(&move_loc) {
                            continue;
                        }
                        locs.push(move_loc);
                        locs_set.insert(move_loc);
                        play_selection_values.push(0.0);
                        if let Some(ref mut v) = ret_visit_counts {
                            v.push(0.0);
                        }
                        num_children += 1;
                    }
                }

                let mut shifted_policy: HashMap<Loc, f64> = HashMap::new();
                let mut self_utilities: HashMap<Loc, f64> = HashMap::new();
                let mut self_utility_max = -1e10;
                let mut self_utility_sum = 0.0;
                for i in 0..children_capacity {
                    let child_ptr = children.get(i);
                    let child = match child_ptr.get_if_allocated() {
                        Some(c) => c,
                        None => break,
                    };
                    let move_loc = child_ptr.get_move_loc_relaxed();
                    let pos = self.get_pos(move_loc) as usize;
                    let mut human_prob = human_probs.get(pos).copied().unwrap_or(0.0f32) as f64;
                    if (suppress_pass && move_loc == PASS_LOC) || human_prob < 0.0 {
                        human_prob = 0.0;
                    }
                    shifted_policy.insert(move_loc, human_prob);
                    let child_utility = child.stats.utility_avg.load(Ordering::Acquire);
                    let self_utility = if self.root_pla == P_WHITE {
                        child_utility
                    } else {
                        -child_utility
                    };
                    self_utilities.insert(move_loc, self_utility);
                    if self_utility > self_utility_max {
                        self_utility_max = self_utility;
                    }
                    self_utility_sum += self_utility;
                }

                let self_utility_avg = self_utility_sum / self_utilities.len().max(1) as f64;
                let self_utility_max = self_utility_max.max(self_utility_avg);
                for &loc in locs.iter() {
                    if !shifted_policy.contains_key(&loc) {
                        let pos = self.get_pos(loc) as usize;
                        let mut human_prob = human_probs.get(pos).copied().unwrap_or(0.0f32) as f64;
                        if (suppress_pass && loc == PASS_LOC) || human_prob < 0.0 {
                            human_prob = 0.0;
                        }
                        shifted_policy.insert(loc, human_prob);
                        self_utilities.insert(loc, self_utility_avg);
                    }
                }

                for &loc in locs.iter() {
                    let self_utility = self_utilities[&loc];
                    if let Some(p) = shifted_policy.get_mut(&loc) {
                        *p *= ((self_utility - self_utility_max)
                            / self.search_params.human_sl_chosen_move_pikl_lambda)
                            .exp();
                    }
                }

                let shifted_policy_sum: f64 = locs.iter().map(|loc| shifted_policy[loc]).sum();
                if shifted_policy_sum > 0.0 {
                    for &loc in locs.iter() {
                        if let Some(p) = shifted_policy.get_mut(&loc) {
                            *p /= shifted_policy_sum;
                        }
                    }

                    let play_selection_value_sum: f64 = play_selection_values.iter().sum();
                    let play_selection_value_non_pass_sum: f64 = play_selection_values
                        .iter()
                        .zip(locs.iter())
                        .filter(|(_, loc)| **loc != PASS_LOC)
                        .map(|(v, _)| v)
                        .sum();

                    if self.search_params.human_sl_chosen_move_ignore_pass {
                        let shifted_policy_non_pass_sum: f64 = locs
                            .iter()
                            .filter(|loc| **loc != PASS_LOC)
                            .map(|loc| shifted_policy[loc])
                            .sum();
                        if shifted_policy_non_pass_sum > 0.0 {
                            for &loc in locs.iter() {
                                if loc != PASS_LOC {
                                    let new_p = shifted_policy[&loc] / shifted_policy_non_pass_sum
                                        * play_selection_value_non_pass_sum
                                        / play_selection_value_sum;
                                    shifted_policy.insert(loc, new_p);
                                } else {
                                    let new_p = (play_selection_value_sum
                                        - play_selection_value_non_pass_sum)
                                        / play_selection_value_sum;
                                    shifted_policy.insert(loc, new_p);
                                }
                            }
                        }
                    }

                    for i in 0..num_children {
                        play_selection_values[i] += self.search_params.human_sl_chosen_move_prop
                            * (play_selection_value_sum * shifted_policy[&locs[i]]
                                - play_selection_values[i]);
                    }
                }
            }
        }

        max_value = 0.0;
        for &v in play_selection_values.iter() {
            if v > max_value {
                max_value = v;
            }
        }
        assert!(max_value > 0.0);
        if max_value < scale_max_to_at_least {
            let scale = scale_max_to_at_least / max_value;
            for v in play_selection_values.iter_mut() {
                *v *= scale;
            }
        }

        true
    }

    fn is_okay_raw_policy_move_at_root(
        &self,
        move_loc: Loc,
        policy_prob: f64,
        obey_allowed_root_move: bool,
    ) -> bool {
        if !self
            .root_history
            .is_legal(&self.root_board, move_loc, self.root_pla)
            || policy_prob < 0.0
            || (obey_allowed_root_move && !self.is_allowed_root_move(move_loc))
        {
            return false;
        }
        let avoid = if self.root_pla == P_BLACK {
            &self.avoid_move_until_by_loc_black
        } else {
            &self.avoid_move_until_by_loc_white
        };
        if !avoid.is_empty() {
            if (move_loc as usize) >= avoid.len() {
                return false;
            }
            if avoid[move_loc as usize] > 0 {
                return false;
            }
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn get_analysis_data_of_single_child(
        &self,
        child: Option<&SearchNode>,
        edge_visits: i64,
        scratch_locs: &mut Vec<Loc>,
        scratch_values: &mut Vec<f64>,
        move_loc: Loc,
        policy_prob: f64,
        fpu_value: f64,
        parent_utility: f64,
        parent_win_loss_value: f64,
        parent_score_mean: f64,
        parent_score_stdev: f64,
        parent_lead: f64,
        max_pv_depth: i32,
    ) -> AnalysisData {
        let mut child_visits: i64 = 0;
        let mut win_loss_value_avg: f64 = 0.0;
        let mut no_result_value_avg: f64 = 0.0;
        let mut score_mean_avg: f64 = 0.0;
        let mut score_mean_sq_avg: f64 = 0.0;
        let mut lead_avg: f64 = 0.0;
        let mut utility_avg: f64 = 0.0;
        let mut utility_sq_avg: f64 = 0.0;
        let mut weight_sum: f64 = 0.0;
        let mut weight_sq_sum: f64 = 0.0;
        let mut child_weight_sum: f64 = 0.0;

        if let Some(child) = child {
            child_visits = child.stats.visits.load(Ordering::Acquire);
            win_loss_value_avg = child.stats.win_loss_value_avg.load(Ordering::Acquire);
            no_result_value_avg = child.stats.no_result_value_avg.load(Ordering::Acquire);
            score_mean_avg = child.stats.score_mean_avg.load(Ordering::Acquire);
            score_mean_sq_avg = child.stats.score_mean_sq_avg.load(Ordering::Acquire);
            lead_avg = child.stats.lead_avg.load(Ordering::Acquire);
            utility_avg = child.stats.utility_avg.load(Ordering::Acquire);
            utility_sq_avg = child.stats.utility_sq_avg.load(Ordering::Acquire);
            weight_sum = child
                .stats
                .get_child_weight_with_child_visits(edge_visits, child_visits);
            weight_sq_sum = child
                .stats
                .get_child_weight_sq_with_child_visits(edge_visits, child_visits);
            child_weight_sum = child.stats.weight_sum.load(Ordering::Acquire);
        }

        let mut data = AnalysisData::new();
        data.move_loc = move_loc;
        data.num_visits = edge_visits;
        if child_visits <= 0 || child_weight_sum <= 1e-30 {
            data.utility = fpu_value;
            data.score_utility = self.get_score_utility(
                parent_score_mean,
                parent_score_mean * parent_score_mean + parent_score_stdev * parent_score_stdev,
            );
            data.result_utility = fpu_value - data.score_utility;
            data.win_loss_value = if self.search_params.win_loss_utility_factor == 1.0 {
                parent_win_loss_value + (fpu_value - parent_utility)
            } else {
                0.0
            };
            data.no_result_value = 0.0;
            if data.win_loss_value < -1.0 {
                data.win_loss_value = -1.0;
            }
            if data.win_loss_value > 1.0 {
                data.win_loss_value = 1.0;
            }
            data.score_mean = parent_score_mean;
            data.score_stdev = parent_score_stdev;
            data.lead = parent_lead;
            data.ess = 0.0;
            data.weight_sum = 0.0;
            data.weight_sq_sum = 0.0;
            data.utility_sq_avg = data.utility * data.utility;
            data.score_mean_sq_avg =
                parent_score_mean * parent_score_mean + parent_score_stdev * parent_score_stdev;
            data.child_visits = child_visits;
            data.child_weight_sum = child_weight_sum;
        } else {
            data.utility = utility_avg;
            data.result_utility = self.get_result_utility(win_loss_value_avg, no_result_value_avg);
            data.score_utility = self.get_score_utility(score_mean_avg, score_mean_sq_avg);
            data.win_loss_value = win_loss_value_avg;
            data.no_result_value = no_result_value_avg;
            data.score_mean = score_mean_avg;
            data.score_stdev = score_value::get_score_stdev(score_mean_avg, score_mean_sq_avg);
            data.lead = lead_avg;
            data.ess = weight_sum * weight_sum / weight_sq_sum.max(1e-8);
            data.weight_sum = weight_sum;
            data.weight_sq_sum = weight_sq_sum;
            data.utility_sq_avg = utility_sq_avg;
            data.score_mean_sq_avg = score_mean_sq_avg;
            data.child_visits = child_visits;
            data.child_weight_sum = child_weight_sum;
        }

        data.policy_prior = policy_prob;
        data.order = 0;
        data.node = child
            .map(|c| c as *const SearchNode)
            .unwrap_or(std::ptr::null());

        data.pv.clear();
        data.pv.push(move_loc);
        data.pv_visits.clear();
        data.pv_visits.push(child_visits);
        data.pv_edge_visits.clear();
        data.pv_edge_visits.push(edge_visits);
        self.append_pv(
            &mut data.pv,
            &mut data.pv_visits,
            &mut data.pv_edge_visits,
            scratch_locs,
            scratch_values,
            child,
            max_pv_depth,
        );

        data
    }

    fn print_pv_from_buf(&self, out: &mut String, buf: &[Loc]) {
        out.clear();
        let _ = buf;
    }

    fn get_sharp_score_helper(
        &self,
        node: &SearchNode,
        graph_path: &mut HashSet<*const SearchNode>,
        policy_probs_buf: &mut [f64],
        min_prop: f64,
        desired_prop: f64,
        ret: &mut f64,
    ) -> bool {
        let nn_output = node.get_nn_output();

        if desired_prop < min_prop {
            let stats = NodeStats::from_atomic(&node.stats);
            if stats.visits <= 0 {
                return false;
            }
            *ret += stats.score_mean_avg * desired_prop;
            return true;
        }

        let nn_output = match nn_output {
            Some(o) => o,
            None => {
                let stats = NodeStats::from_atomic(&node.stats);
                if stats.visits <= 0 {
                    return false;
                }
                *ret += stats.score_mean_avg * desired_prop;
                return true;
            }
        };

        let children = node.get_children();
        let children_capacity = children.get_capacity();
        if children_capacity == 0 {
            let score_mean = nn_output.white_score_mean as f64;
            *ret += score_mean * desired_prop;
            return true;
        }

        if !graph_path.insert(node as *const SearchNode) {
            let score_mean = nn_output.white_score_mean as f64;
            *ret += score_mean * desired_prop;
            return true;
        }

        let mut stats_buf = Vec::new();
        for i in 0..children_capacity {
            let child_ptr = children.get(i);
            let child = match child_ptr.get_if_allocated() {
                Some(c) => c,
                None => break,
            };
            let edge_visits = child_ptr.get_edge_visits();
            let move_loc = child_ptr.get_move_loc_relaxed();
            let mut stats = MoreNodeStats::default();
            stats.stats = NodeStats::from_atomic(&child.stats);
            stats.self_utility = if node.next_pla == P_WHITE {
                stats.stats.utility_avg
            } else {
                -stats.stats.utility_avg
            };
            stats.weight_adjusted = stats.stats.get_child_weight(edge_visits);
            stats.prev_move_loc = move_loc;
            stats_buf.push(stats);
        }

        let num_children = stats_buf.len() as i32;
        if num_children > 0 {
            let mut total_child_weight = 0.0;
            for s in &stats_buf {
                total_child_weight += s.weight_adjusted;
            }
            let policy_probs = nn_output.get_policy_probs_maybe_noised();
            if self.search_params.use_noise_pruning {
                let stats_buf_len = stats_buf.len();
                for i in 0..stats_buf_len {
                    let pos = self.get_pos(stats_buf[i].prev_move_loc) as usize;
                    policy_probs_buf[i] = (policy_probs[pos] as f64).max(1e-30);
                }
                total_child_weight = self.prune_noise_weight(
                    &mut stats_buf,
                    num_children,
                    total_child_weight,
                    &policy_probs_buf[..stats_buf_len],
                );
            }
            let mut stdevs_scratch = Vec::with_capacity(stats_buf.len());
            self.downweight_bad_children_and_normalize_weight(
                num_children,
                total_child_weight,
                total_child_weight,
                0.0,
                0.0,
                &mut stats_buf,
                &mut stdevs_scratch,
            );
        }

        let mut relative_children_weight_sum = 0.0;
        let mut children_weight_sum = 0.0;
        for s in &stats_buf {
            if s.stats.visits <= 0 {
                continue;
            }
            relative_children_weight_sum +=
                s.weight_adjusted * s.weight_adjusted * s.weight_adjusted;
            children_weight_sum += s.weight_adjusted;
        }

        let mut parent_nn_weight = self.compute_weight_from_nn_output(Some(&*nn_output));
        parent_nn_weight = parent_nn_weight.max(1e-10);
        let desired_prop_from_children =
            desired_prop * children_weight_sum / (children_weight_sum + parent_nn_weight);
        let mut self_prop =
            desired_prop * parent_nn_weight / (children_weight_sum + parent_nn_weight);

        if desired_prop_from_children <= 0.0 || relative_children_weight_sum <= 0.0 {
            self_prop += desired_prop_from_children;
        } else {
            for i in 0..children_capacity {
                let child = children.get(i).get_if_allocated().unwrap();
                let child_weight = stats_buf[i].weight_adjusted;
                let desired_prop_from_child = child_weight * child_weight * child_weight
                    / relative_children_weight_sum
                    * desired_prop_from_children;
                let accumulated = self.get_sharp_score_helper(
                    child,
                    graph_path,
                    policy_probs_buf,
                    min_prop,
                    desired_prop_from_child,
                    ret,
                );
                if !accumulated {
                    self_prop += desired_prop_from_child;
                }
            }
        }

        graph_path.remove(&(node as *const SearchNode));

        let score_mean = nn_output.white_score_mean as f64;
        *ret += score_mean * self_prop;
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn get_shallow_average_shortterm_wl_and_score_error_helper(
        &self,
        node: &SearchNode,
        graph_path: &mut HashSet<*const SearchNode>,
        policy_probs_buf: &mut [f64],
        min_prop: f64,
        desired_prop: f64,
        wl_error: &mut f64,
        score_error: &mut f64,
    ) {
        let nn_output = match node.get_nn_output() {
            Some(o) => o,
            None => {
                // Accumulate nothing. This is correct for terminal nodes and handles
                // nodes that temporarily lack an NN output during multithreading.
                return;
            }
        };

        if desired_prop < min_prop {
            *wl_error += desired_prop * nn_output.shortterm_winloss_error as f64;
            *score_error += desired_prop * nn_output.shortterm_score_error as f64;
            return;
        }

        if !graph_path.insert(node as *const SearchNode) {
            *wl_error += desired_prop * nn_output.shortterm_winloss_error as f64;
            *score_error += desired_prop * nn_output.shortterm_score_error as f64;
            return;
        }

        let children = node.get_children();
        let children_capacity = children.get_capacity();

        let mut stats_buf = Vec::new();
        for i in 0..children_capacity {
            let child_ptr = children.get(i);
            let child = match child_ptr.get_if_allocated() {
                Some(c) => c,
                None => break,
            };
            let edge_visits = child_ptr.get_edge_visits();
            let move_loc = child_ptr.get_move_loc_relaxed();
            let mut stats = MoreNodeStats::default();
            stats.stats = NodeStats::from_atomic(&child.stats);
            stats.self_utility = if node.next_pla == P_WHITE {
                stats.stats.utility_avg
            } else {
                -stats.stats.utility_avg
            };
            stats.weight_adjusted = stats.stats.get_child_weight(edge_visits);
            stats.prev_move_loc = move_loc;
            stats_buf.push(stats);
        }

        let num_children = stats_buf.len() as i32;
        if num_children > 0 {
            let mut total_child_weight = 0.0;
            for s in &stats_buf {
                total_child_weight += s.weight_adjusted;
            }
            let policy_probs = nn_output.get_policy_probs_maybe_noised();
            if self.search_params.use_noise_pruning {
                let stats_buf_len = stats_buf.len();
                for i in 0..stats_buf_len {
                    let pos = self.get_pos(stats_buf[i].prev_move_loc) as usize;
                    policy_probs_buf[i] = (policy_probs[pos] as f64).max(1e-30);
                }
                total_child_weight = self.prune_noise_weight(
                    &mut stats_buf,
                    num_children,
                    total_child_weight,
                    &policy_probs_buf[..stats_buf_len],
                );
            }
            let mut stdevs_scratch = Vec::with_capacity(stats_buf.len());
            self.downweight_bad_children_and_normalize_weight(
                num_children,
                total_child_weight,
                total_child_weight,
                0.0,
                0.0,
                &mut stats_buf,
                &mut stdevs_scratch,
            );
        }

        let mut relative_children_weight_sum = 0.0;
        let mut children_weight_sum = 0.0;
        for s in &stats_buf {
            relative_children_weight_sum += s.weight_adjusted;
            children_weight_sum += s.weight_adjusted;
        }

        let mut parent_nn_weight = self.compute_weight_from_nn_output(Some(&*nn_output));
        parent_nn_weight = parent_nn_weight.max(1e-10);
        let desired_prop_from_children =
            desired_prop * children_weight_sum / (children_weight_sum + parent_nn_weight);
        let mut self_prop =
            desired_prop * parent_nn_weight / (children_weight_sum + parent_nn_weight);

        if desired_prop_from_children <= 0.0 || relative_children_weight_sum <= 0.0 {
            self_prop += desired_prop_from_children;
        } else {
            for i in 0..children_capacity {
                let child = children.get(i).get_if_allocated().unwrap();
                let child_weight = stats_buf[i].weight_adjusted;
                let desired_prop_from_child =
                    child_weight / relative_children_weight_sum * desired_prop_from_children;
                self.get_shallow_average_shortterm_wl_and_score_error_helper(
                    child,
                    graph_path,
                    policy_probs_buf,
                    min_prop,
                    desired_prop_from_child,
                    wl_error,
                    score_error,
                );
            }
        }

        graph_path.remove(&(node as *const SearchNode));

        *wl_error += self_prop * nn_output.shortterm_winloss_error as f64;
        *score_error += self_prop * nn_output.shortterm_score_error as f64;
    }

    fn traverse_tree_for_ownership<A: OwnershipAccumulator>(
        &self,
        min_prop: f64,
        prune_prop: f64,
        desired_prop: f64,
        node: Option<&SearchNode>,
        graph_path: &mut HashSet<*const SearchNode>,
        accumulate: &mut A,
    ) -> bool {
        let node = match node {
            Some(n) => n,
            None => return false,
        };

        let nn_output = match node.get_nn_output() {
            Some(o) => o,
            None => return false,
        };
        let owner_map = match nn_output.white_owner_map.as_ref() {
            Some(m) => m,
            None => return false,
        };

        if desired_prop < min_prop {
            accumulate.accumulate(owner_map, desired_prop);
            return true;
        }

        let children = node.get_children();
        let children_capacity = children.get_capacity();
        if children_capacity == 0 {
            accumulate.accumulate(owner_map, desired_prop);
            return true;
        }

        if !graph_path.insert(node as *const SearchNode) {
            accumulate.accumulate(owner_map, desired_prop);
            return true;
        }

        let parent_nn_weight = self.compute_weight_from_nn_output(Some(&nn_output));
        let self_prop = self.traverse_tree_for_ownership_children(
            min_prop,
            prune_prop,
            desired_prop,
            parent_nn_weight,
            &children,
            children_capacity,
            graph_path,
            accumulate,
        );

        graph_path.remove(&(node as *const SearchNode));
        accumulate.accumulate(owner_map, self_prop);
        true
    }

    fn traverse_tree_for_ownership_children<A: OwnershipAccumulator>(
        &self,
        min_prop: f64,
        prune_prop: f64,
        desired_prop: f64,
        mut parent_nn_weight: f64,
        children: &SearchNodeChildrenReference<'_>,
        children_capacity: usize,
        graph_path: &mut HashSet<*const SearchNode>,
        accumulate: &mut A,
    ) -> f64 {
        let mut child_weights = vec![0.0f64; children_capacity];
        let mut num_children = 0;
        for i in 0..children_capacity {
            let child_pointer = children.get(i);
            let child = match child_pointer.get_if_allocated() {
                Some(c) => c,
                None => break,
            };
            let edge_visits = child_pointer.get_edge_visits();
            child_weights[i] = child.stats.get_child_weight(edge_visits);
            num_children += 1;
        }

        let mut relative_children_weight_sum = 0.0;
        let mut children_weight_sum = 0.0;
        for i in 0..num_children {
            let child_weight = child_weights[i];
            relative_children_weight_sum += child_weight * child_weight;
            children_weight_sum += child_weight;
        }

        parent_nn_weight = parent_nn_weight.max(1e-10);
        let desired_prop_from_children =
            desired_prop * children_weight_sum / (children_weight_sum + parent_nn_weight);
        let mut self_prop =
            desired_prop * parent_nn_weight / (children_weight_sum + parent_nn_weight);

        if desired_prop_from_children <= 0.0 || relative_children_weight_sum <= 0.0 {
            self_prop += desired_prop_from_children;
        } else {
            for i in 0..num_children {
                let child_weight = child_weights[i];
                let child_pointer = children.get(i);
                let child = child_pointer
                    .get_if_allocated()
                    .expect("child should exist");
                let desired_prop_from_child = child_weight * child_weight
                    / relative_children_weight_sum
                    * desired_prop_from_children;
                if desired_prop_from_child < prune_prop {
                    self_prop += desired_prop_from_child;
                } else {
                    let accumulated = self.traverse_tree_for_ownership(
                        min_prop,
                        prune_prop,
                        desired_prop_from_child,
                        Some(child),
                        graph_path,
                        accumulate,
                    );
                    if !accumulated {
                        self_prop += desired_prop_from_child;
                    }
                }
            }
        }

        self_prop
    }

    fn debug_print_children_summary(
        &self,
        out: &mut String,
        node: &SearchNode,
        nn_output: Option<&NNOutput>,
    ) {
        let _ = (out, node, nn_output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::STATE_EXPANDED0;
    use kata_core::config::ConfigParser;
    use kata_core::logger::{Logger, LoggerOptions};
    use kata_game::board::{Board, NULL_LOC, P_BLACK, P_WHITE, PASS_LOC, location};
    use kata_game::rules::Rules;

    fn test_logger() -> Logger {
        Logger::new(LoggerOptions::default(), None)
    }

    fn dummy_evaluator() -> NnEvaluator {
        NnEvaluator::new(
            "dummy".to_string(),
            "dummy.bin".to_string(),
            String::new(),
            Arc::new(test_logger()),
            1,
            19,
            19,
            false,
            false,
            -1,
            0,
            true,
            String::new(),
            kata_nn::backend::Enabled::False,
            0,
            Vec::new(),
            "seed".to_string(),
            false,
            0,
            true,
            &ConfigParser::new(false, false),
        )
    }

    fn search_with_dummy() -> Search<'static> {
        // Safe: we leak the evaluator and logger so their references are static.
        let eval: &'static NnEvaluator = Box::leak(Box::new(dummy_evaluator()));
        let logger: &'static Logger = Box::leak(Box::new(test_logger()));
        Search::new(SearchParams::new(), eval, logger, "test-seed")
    }

    #[test]
    fn test_search_construct_and_getters() {
        let mut search = search_with_dummy();
        assert_eq!(search.get_root_pla(), C_EMPTY);
        assert_eq!(search.get_root_board().x_size, 19);
        assert_eq!(search.get_root_hist().initial_pla, C_EMPTY);
        assert_eq!(search.nn_x_len, 19);
        assert_eq!(search.nn_y_len, 19);
        assert_eq!(search.policy_size, 19 * 19 + 1);
        assert_eq!(search.get_root_visits(), 0);
        assert_eq!(search.get_chosen_move_loc(), NULL_LOC);
        assert_eq!(search.get_playout_doubling_advantage_pla(), C_EMPTY);
    }

    #[test]
    fn test_begin_search_filter_after_reuse_does_not_dangle() {
        // Regression for the begin_search root-filter UAF: after a first
        // search builds a tree, a second begin_search on the SAME root (no
        // make_move) bumps search_node_age, so the retained subtree counts as
        // "old". When avoid-move filtering then triggers the delete-old sweep,
        // the retained subtree must have been re-marked first — otherwise the
        // root's child pointers dangle and the second search reads freed
        // memory (this test would crash or trip heap corruption).
        let mut search = search_with_dummy();
        let mut params = SearchParams::new();
        params.max_visits = 100;
        params.num_threads = 1;
        search.set_params(&params);
        let board = Board::new(19, 19);
        let history = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        search.set_position(P_BLACK, &board, &history);

        // First search: build a tree.
        search.run_whole_search(P_BLACK);
        assert!(search.root_node.is_some());
        assert!(search.get_root_visits() > 0);

        // Second search on the same root WITHOUT make_move — with avoid-move
        // filtering active so any_filtered goes down the delete-old path.
        let mut b_vec = vec![0; kata_game::board::MAX_ARR_SIZE];
        let mut w_vec = vec![0; kata_game::board::MAX_ARR_SIZE];
        b_vec[location::get_loc(3, 3, 19) as usize] = 1;
        search.set_avoid_move_until_by_loc(&b_vec, &w_vec);
        search.run_whole_search(P_BLACK);
        assert!(search.root_node.is_some());
        assert!(search.get_root_visits() > 0);

        // Walk the root children to force dereferences of any retained nodes.
        let root = search.root_node.as_deref().unwrap();
        let children = root.get_children();
        let cap = children.get_capacity();
        let mut num_children = 0;
        for i in 0..cap {
            match children.get(i).get_if_allocated() {
                Some(child) => {
                    let _ = child.stats.visits.load(Ordering::Acquire);
                    num_children += 1;
                }
                None => break,
            }
        }
        assert!(num_children > 0);
    }

    #[test]
    fn test_search_pos_helper() {
        let search = search_with_dummy();
        let loc = location::get_loc(3, 3, search.root_board.x_size);
        let pos = search.get_pos(loc);
        assert_eq!(pos, nn_pos::loc_to_pos(loc, 19, 19, 19));
        assert_eq!(search.get_pos(PASS_LOC), 19 * 19);
    }

    #[test]
    fn test_set_position_and_clear() {
        let mut search = search_with_dummy();
        let board = Board::new(9, 9);
        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        search.set_position(P_BLACK, &board, &hist);
        assert_eq!(search.get_root_pla(), P_BLACK);
        assert_eq!(search.get_root_board().x_size, 9);
        assert_eq!(search.get_root_visits(), 0);
    }

    #[test]
    fn test_setters_do_not_panic() {
        let mut search = search_with_dummy();
        search.set_root_hint_loc(PASS_LOC);
        search.set_avoid_move_until_by_loc(&[1, 2], &[3, 4]);
        search.set_avoid_move_until_rescale_root(true);
        search.set_always_include_owner_map(true);
        search.set_root_symmetry_pruning_only(&[0, 1]);

        let params = SearchParams::for_tests_v1();
        search.set_params_no_clearing(&params);
        assert_eq!(search.search_params.static_score_utility_factor, 0.1);

        search.set_external_pattern_bonus_table(None);
        search.set_external_eval_cache(None);
        search.clear_search();
    }

    #[test]
    fn test_make_move_returns_true_for_legal_false_for_illegal() {
        let mut search = search_with_dummy();
        let loc = location::get_loc(0, 0, 19);
        // (0,0) is a legal empty intersection on a 19x19 board.
        assert!(search.make_move(loc, P_BLACK));
        // The same intersection is now occupied, so a second move is illegal.
        assert!(!search.make_move(loc, P_WHITE));
        // NULL_LOC is always illegal.
        assert!(!search.make_move(NULL_LOC, P_BLACK));
    }

    #[test]
    fn test_search_thread_new() {
        let search = search_with_dummy();
        let thread = SearchThread::new(0, &search);
        assert_eq!(thread.thread_idx, 0);
        assert_eq!(thread.pla, search.root_pla);
        assert!(thread.old_nn_outputs_to_clean_up.is_empty());
    }

    #[test]
    fn test_choose_index_with_temperature() {
        let mut rand = Rand::new_from_seed("temperature-test");
        let probs = [1.0, 2.0, 3.0, 4.0];

        // Low temperature is deterministic argmax.
        let idx = Search::choose_index_with_temperature(&mut rand, &probs, 1e-5, 1.0, None);
        assert_eq!(idx, 3);

        // Finite temperature returns a valid index.
        let idx = Search::choose_index_with_temperature(&mut rand, &probs, 1.0, 1.0, None);
        assert!((idx as usize) < probs.len());
    }

    #[test]
    fn test_compute_dirichlet_alpha_distribution() {
        const POLICY_SIZE: usize = 4;
        let mut policy = [-1.0f32; POLICY_SIZE];
        policy[0] = 0.1;
        policy[1] = 0.2;
        policy[2] = 0.7;
        // policy[3] stays illegal.

        let mut alpha = vec![0.0f64; POLICY_SIZE];
        Search::compute_dirichlet_alpha_distribution(POLICY_SIZE as i32, &policy, &mut alpha);

        let sum: f64 = alpha.iter().filter(|&&a| a > 0.0).sum();
        assert!((sum - 1.0).abs() < 1e-9);
        assert_eq!(alpha[3], 0.0);
    }

    #[test]
    fn test_add_dirichlet_noise_keeps_normalization() {
        const POLICY_SIZE: usize = 4;
        let mut policy = [0.0f32; POLICY_SIZE];
        policy[0] = 0.1;
        policy[1] = 0.2;
        policy[2] = 0.7;

        let mut params = SearchParams::new();
        params.root_noise_enabled = true;
        params.root_dirichlet_noise_total_concentration = 10.83;
        params.root_dirichlet_noise_weight = 0.25;

        let mut rand = Rand::new_from_seed("dirichlet-test");
        Search::add_dirichlet_noise(&params, &mut rand, POLICY_SIZE as i32, &mut policy);

        let sum: f64 = policy.iter().map(|&p| p as f64).sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_interpolate_early() {
        let mut search = search_with_dummy();

        let early = search.interpolate_early(19.0, 2.0, 1.0);
        assert!((early - 2.0).abs() < 1e-9);

        search.root_history.initial_turn_number = 100;
        let late = search.interpolate_early(19.0, 2.0, 1.0);
        assert!(late < early);
        assert!((late - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_score_and_nn_utility_signs() {
        let search = search_with_dummy();

        let positive = search.get_score_utility(10.0, 100.0);
        assert!(positive > 0.0);
        let negative = search.get_score_utility(-10.0, 100.0);
        assert!(negative < 0.0);

        let nn_output = NNOutput {
            white_win_prob: 1.0,
            white_loss_prob: 0.0,
            white_no_result_prob: 0.0,
            white_score_mean: 10.0,
            white_score_mean_sq: 100.0,
            ..NNOutput::default()
        };
        let utility = search.get_utility_from_nn(&nn_output);
        assert!(utility > 0.0);
    }

    #[test]
    fn test_is_allowed_root_move_rejects_avoid_and_null() {
        let mut search = search_with_dummy();
        let board = Board::new(19, 19);
        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        search.set_position(P_BLACK, &board, &hist);

        assert!(!search.is_allowed_root_move(NULL_LOC));

        let loc = location::get_loc(10, 10, board.x_size);
        assert!(search.is_allowed_root_move(PASS_LOC));
        assert!(search.is_allowed_root_move(loc));

        let mut avoid = vec![0; loc as usize + 1];
        avoid[loc as usize] = 1;
        search.set_avoid_move_until_by_loc(&avoid, &[]);
        assert!(!search.is_allowed_root_move(loc));
        assert!(search.is_allowed_root_move(PASS_LOC));
    }

    fn node_with_owner_map(owner_map: Option<Vec<f32>>) -> SearchNode {
        let node = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        let mut nn_output = NNOutput::default();
        nn_output.nn_x_len = 19;
        nn_output.nn_y_len = 19;
        nn_output.white_owner_map = owner_map.map(|v| v.into_boxed_slice());
        node.nn_output.store(
            Box::into_raw(Box::new(Arc::new(nn_output))),
            Ordering::Release,
        );
        node
    }

    #[test]
    fn test_get_ending_white_score_bonus_returns_zero_for_non_root_or_null_loc() {
        let mut search = search_with_dummy();
        search.search_params.root_ending_bonus_points = 0.5;
        search.root_pla = P_WHITE;

        let parent = node_with_owner_map(Some(vec![0.0f32; 19 * 19]));
        let loc = location::get_loc(3, 3, search.root_board.x_size);

        // Non-root parent returns 0 regardless of move_loc.
        assert_eq!(search.get_ending_white_score_bonus(&parent, loc), 0.0);
        assert_eq!(search.get_ending_white_score_bonus(&parent, PASS_LOC), 0.0);

        // NULL_LOC returns 0 even if parent were root.
        assert_eq!(search.get_ending_white_score_bonus(&parent, NULL_LOC), 0.0);

        // Root parent with no NN output also returns 0.
        let root_node = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        search.root_node = Some(Box::new(root_node));
        assert_eq!(
            search.get_ending_white_score_bonus(search.root_node.as_deref().unwrap(), loc),
            0.0
        );
    }

    #[test]
    fn test_should_suppress_pass_returns_false_when_disabled_or_no_root() {
        let mut search = search_with_dummy();
        let parent = node_with_owner_map(Some(vec![0.0f32; 19 * 19]));

        // Disabled by default.
        assert!(!search.should_suppress_pass(Some(&parent)));

        // Enabled but no root node matches.
        search.search_params.fill_dame_before_pass = true;
        assert!(!search.should_suppress_pass(Some(&parent)));

        // Enabled and root node but missing pass child still returns false.
        search.root_node = Some(Box::new(node_with_owner_map(Some(vec![0.0f32; 19 * 19]))));
        assert!(!search.should_suppress_pass(search.root_node.as_deref()));
    }

    #[test]
    fn test_get_self_utility_lcb_and_radius_zero_visits() {
        let search = search_with_dummy();
        let mut lcb = 0.0;
        let mut radius = 0.0;
        search.get_self_utility_lcb_and_radius_zero_visits(&mut lcb, &mut radius);

        let expected_radius = 2.0
            * (search.search_params.win_loss_utility_factor
                + search.search_params.static_score_utility_factor
                + search.search_params.dynamic_score_utility_factor)
            * search.search_params.lcb_stdevs;
        assert!((radius - expected_radius).abs() < 1e-9);
        assert!((lcb - (-expected_radius)).abs() < 1e-9);
    }

    #[test]
    fn test_get_self_utility_lcb_and_radius_with_fabricated_child() {
        let mut search = search_with_dummy();
        search.search_params.root_ending_bonus_points = 0.5;

        let parent = SearchNode::new(P_WHITE, false, 0, Hash128::default());
        parent.nn_output.store(
            Box::into_raw(Box::new(Arc::new(NNOutput::default()))),
            Ordering::Release,
        );
        search.root_node = Some(Box::new(parent));
        let parent_ref = search.root_node.as_deref().unwrap();

        let child = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        child.stats.visits.store(100, Ordering::Release);
        child.stats.score_mean_avg.store(0.0, Ordering::Release);
        child
            .stats
            .score_mean_sq_avg
            .store(100.0, Ordering::Release);
        child.stats.utility_avg.store(0.5, Ordering::Release);
        child.stats.utility_sq_avg.store(0.26, Ordering::Release);
        child.stats.weight_sum.store(100.0, Ordering::Release);
        child.stats.weight_sq_sum.store(10000.0, Ordering::Release);

        let mut lcb = 0.0;
        let mut radius = 0.0;
        search.get_self_utility_lcb_and_radius(
            parent_ref,
            Some(&child),
            100,
            PASS_LOC,
            &mut lcb,
            &mut radius,
        );

        // The radius should be positive and much smaller than the zero-visit radius.
        let mut zero_lcb = 0.0;
        let mut zero_radius = 0.0;
        search.get_self_utility_lcb_and_radius_zero_visits(&mut zero_lcb, &mut zero_radius);
        assert!(radius > 0.0);
        assert!(radius < zero_radius);

        // With utility_avg = 0.5, parent.next_pla = WHITE, and a small positive ending
        // bonus for passing, the LCB should be below the raw utility but not extremely negative.
        assert!(lcb < 0.5);
        assert!(lcb > zero_lcb);
    }

    #[test]
    fn test_cpuct_exploration_and_get_explore_scaling() {
        let search = search_with_dummy();
        let total_child_weight = 100.0;
        let stdev_factor = 1.2;

        let expected_cpuct = cpuct_exploration(total_child_weight, &search.search_params);
        let expected = expected_cpuct
            * (total_child_weight + TOTALCHILDWEIGHT_PUCT_OFFSET).sqrt()
            * stdev_factor;
        let actual = search.get_explore_scaling(total_child_weight, stdev_factor);

        assert!((actual - expected).abs() < 1e-9);

        let human_expected = cpuct_exploration_human(total_child_weight, &search.search_params)
            * (total_child_weight + TOTALCHILDWEIGHT_PUCT_OFFSET).sqrt();
        let human_actual = search.get_explore_scaling_human(total_child_weight, stdev_factor);
        assert!((human_actual - human_expected).abs() < 1e-9);
    }

    #[test]
    fn test_get_fpu_value_for_children_assume_visited() {
        let search = search_with_dummy();

        let node = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        node.stats.visits.store(20, Ordering::Release);
        node.stats.weight_sum.store(15.0, Ordering::Release);
        node.stats.utility_avg.store(0.1, Ordering::Release);
        node.stats.utility_sq_avg.store(0.05, Ordering::Release);

        let mut nn_output = NNOutput::default();
        nn_output.nn_x_len = 19;
        nn_output.nn_y_len = 19;
        node.nn_output.store(
            Box::into_raw(Box::new(Arc::new(nn_output))),
            Ordering::Release,
        );

        let mut parent_utility = 0.0;
        let mut parent_weight_per_visit = 0.0;
        let mut parent_utility_stdev_factor = 0.0;
        let fpu = search.get_fpu_value_for_children_assume_visited(
            &node,
            P_BLACK,
            false,
            0.25,
            &mut parent_utility,
            &mut parent_weight_per_visit,
            &mut parent_utility_stdev_factor,
        );

        assert!((parent_weight_per_visit - 0.75).abs() < 1e-9);
        assert!((parent_utility - 0.1).abs() < 1e-9);
        assert!(parent_utility_stdev_factor > 0.0);
        // For Black, FPU pushes the value upward from the parent utility.
        assert!(fpu > parent_utility);
    }

    #[test]
    fn test_get_explore_selection_value_of_child_with_fabricated_root() {
        let mut search = search_with_dummy();
        search.root_pla = P_BLACK;

        let mut parent = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        parent.initialize_children();
        parent.state.store(STATE_EXPANDED0, Ordering::Release);

        let mut nn_output = NNOutput::default();
        nn_output.nn_x_len = 19;
        nn_output.nn_y_len = 19;
        let loc = location::get_loc(3, 3, search.root_board.x_size);
        let pos = search.get_pos(loc) as usize;
        nn_output.policy_probs[pos] = 0.5;
        nn_output.policy_probs[search.get_pos(PASS_LOC) as usize] = 0.5;
        parent.nn_output.store(
            Box::into_raw(Box::new(Arc::new(nn_output))),
            Ordering::Release,
        );
        search.root_node = Some(Box::new(parent));
        let parent_ref = search.root_node.as_deref().unwrap();

        let child_ptr = Box::into_raw(Box::new(SearchNode::new(
            P_WHITE,
            false,
            0,
            Hash128::default(),
        )));
        unsafe {
            (*child_ptr).stats.visits.store(10, Ordering::Release);
            (*child_ptr).stats.weight_sum.store(10.0, Ordering::Release);
            (*child_ptr).stats.utility_avg.store(0.2, Ordering::Release);
            (*child_ptr)
                .stats
                .score_mean_avg
                .store(0.0, Ordering::Release);
            (*child_ptr)
                .stats
                .score_mean_sq_avg
                .store(100.0, Ordering::Release);
        }

        let children = parent_ref.get_children();
        children.get(0).store(child_ptr);
        children.get(0).set_move_loc(loc);
        children.get(0).set_edge_visits(10);

        let nn_output = parent_ref.get_nn_output().unwrap();
        let policy_probs = nn_output.get_policy_probs_maybe_noised();

        let mut lcb = 0.0;
        let value = search.get_explore_selection_value_of_child(
            parent_ref,
            policy_probs,
            unsafe { &*child_ptr },
            loc,
            1.0,
            10.0,
            10,
            0.0,
            0.1,
            1.0,
            false,
            false,
            false,
            10.0,
            true,
            Some(&mut lcb),
            None,
        );

        // Expected: explore = 1.0 * 0.5 / (1 + 10), value = -0.2 from Black's perspective.
        let expected = 0.5 / 11.0 - 0.2;
        assert!((value - expected).abs() < 1e-9);
        assert!(lcb < 0.2);

        unsafe {
            drop(Box::from_raw(child_ptr));
        }
    }

    #[test]
    fn test_puct_var_exploration_scales_explore_term() {
        // Mirrors test_get_explore_selection_value_of_child_with_fabricated_root
        // but drives the selection path with is_during_search=true so the
        // PUCT-V child-variance factor applies when puct_var_exploration > 0.
        let mut search = search_with_dummy();
        search.root_pla = P_BLACK;

        let mut params = search.search_params.clone();
        params.puct_var_exploration = 1.0;
        search.set_params_no_clearing(&params);

        let mut parent = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        parent.initialize_children();
        parent.state.store(STATE_EXPANDED0, Ordering::Release);

        let mut nn_output = NNOutput::default();
        nn_output.nn_x_len = 19;
        nn_output.nn_y_len = 19;
        let loc = location::get_loc(3, 3, search.root_board.x_size);
        let pos = search.get_pos(loc) as usize;
        nn_output.policy_probs[pos] = 0.5;
        nn_output.policy_probs[search.get_pos(PASS_LOC) as usize] = 0.5;
        parent
            .nn_output
            .store(Box::into_raw(Box::new(Arc::new(nn_output))), Ordering::Release);
        let parent: &'static mut SearchNode = Box::leak(Box::new(parent));

        let mk_child = |utility_sq: f64| {
            let child_ptr = Box::into_raw(Box::new(SearchNode::new(
                P_WHITE,
                false,
                0,
                Hash128::default(),
            )));
            unsafe {
                (*child_ptr).stats.visits.store(10, Ordering::Release);
                (*child_ptr).stats.weight_sum.store(10.0, Ordering::Release);
                (*child_ptr).stats.utility_avg.store(0.2, Ordering::Release);
                (*child_ptr).stats.utility_sq_avg.store(utility_sq, Ordering::Release);
            }
            child_ptr
        };

        let run = |s: &Search, child_ptr: *mut SearchNode| -> f64 {
            s.get_explore_selection_value_of_child(
                parent,
                parent
                    .get_nn_output()
                    .unwrap()
                    .get_policy_probs_maybe_noised(),
                unsafe { &*child_ptr },
                loc,
                1.0,
                10.0,
                10,
                0.0,
                0.1,
                1.0,
                true,
                false,
                false,
                10.0,
                true,
                None,
                None,
            )
        };

        let baseline_explore = 0.5 / 11.0;
        let value_component = -0.2;
        let prior = search.search_params.cpuct_utility_stdev_prior;

        // stdev == prior: factor 1, identical to baseline.
        let c = mk_child(0.2 * 0.2 + prior * prior);
        assert!((run(&search, c) - (baseline_explore + value_component)).abs() < 1e-9);
        unsafe { drop(Box::from_raw(c)) };

        // stdev == 2*prior: factor 2, explore term doubles.
        let c = mk_child(0.2 * 0.2 + (2.0 * prior) * (2.0 * prior));
        assert!((run(&search, c) - (2.0 * baseline_explore + value_component)).abs() < 1e-9);
        unsafe { drop(Box::from_raw(c)) };

        // zero variance: factor clamps at 0.25 instead of collapsing to 0.
        let c = mk_child(0.2 * 0.2);
        assert!((run(&search, c) - (0.25 * baseline_explore + value_component)).abs() < 1e-9);
        unsafe { drop(Box::from_raw(c)) };

        // Disabled (0.0) is bit-identical regardless of variance.
        let mut params = search.search_params.clone();
        params.puct_var_exploration = 0.0;
        search.set_params_no_clearing(&params);
        let c = mk_child(0.2 * 0.2 + (2.0 * prior) * (2.0 * prior));
        assert!((run(&search, c) - (baseline_explore + value_component)).abs() < 1e-12);
        unsafe { drop(Box::from_raw(c)) };
    }

    #[test]
    fn test_virtual_loss_utility_blend() {
        // WU-UCT mode (Liu et al. 2018): virtual loss keeps the child-weight
        // inflation (deflating the exploration term of in-flight playouts) but
        // scales the utility distortion by virtualLossUtilityBlend. Default
        // 1.0 must reproduce the official KataGo soft blend exactly; 0.0 must
        // leave the stored utility untouched.
        let mut search = search_with_dummy();
        search.root_pla = P_BLACK;

        let mut parent = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        parent.initialize_children();
        parent.state.store(STATE_EXPANDED0, Ordering::Release);

        let mut nn_output = NNOutput::default();
        nn_output.nn_x_len = 19;
        nn_output.nn_y_len = 19;
        let loc = location::get_loc(3, 3, search.root_board.x_size);
        let pos = search.get_pos(loc) as usize;
        nn_output.policy_probs[pos] = 0.5;
        nn_output.policy_probs[search.get_pos(PASS_LOC) as usize] = 0.5;
        parent
            .nn_output
            .store(Box::into_raw(Box::new(Arc::new(nn_output))), Ordering::Release);
        let parent: &'static mut SearchNode = Box::leak(Box::new(parent));

        let mk_child = |virtual_losses: i32| {
            let child_ptr = Box::into_raw(Box::new(SearchNode::new(
                P_WHITE,
                false,
                0,
                Hash128::default(),
            )));
            unsafe {
                (*child_ptr).stats.visits.store(10, Ordering::Release);
                (*child_ptr).stats.weight_sum.store(10.0, Ordering::Release);
                (*child_ptr).stats.utility_avg.store(0.2, Ordering::Release);
                (*child_ptr).stats.utility_sq_avg.store(0.04, Ordering::Release);
                (*child_ptr)
                    .virtual_losses
                    .store(virtual_losses, Ordering::Release);
            }
            child_ptr
        };

        let run = |s: &Search, child_ptr: *mut SearchNode| -> f64 {
            s.get_explore_selection_value_of_child(
                parent,
                parent
                    .get_nn_output()
                    .unwrap()
                    .get_policy_probs_maybe_noised(),
                unsafe { &*child_ptr },
                loc,
                1.0,
                10.0,
                10,
                0.0,
                0.1,
                1.0,
                true,
                false,
                false,
                10.0,
                true,
                None,
                None,
            )
        };

        // Analytic expectations. Parent is P_BLACK so the value component
        // negates the stored (white-perspective) utility.
        let vl_w = 1.0 * search.search_params.num_virtual_losses_per_thread;
        let frac = vl_w / (vl_w + 0.25_f64.max(10.0));
        let radius = search.search_params.win_loss_utility_factor
            + search.search_params.static_score_utility_factor
            + search.search_params.dynamic_score_utility_factor;
        let dist_utility = |blend: f64| 0.2 + (radius - 0.2) * frac * blend;
        let explore = |weight: f64| 1.0 * 0.5 / (1.0 + weight);

        // No virtual losses: plain PUCT value.
        let c = mk_child(0);
        assert!((run(&search, c) - (explore(10.0) + -0.2)).abs() < 1e-12);
        unsafe { drop(Box::from_raw(c)) };

        // blend = 1.0 (default): official soft blend, utility pulled toward
        // the loss radius by frac, weight inflated by vl_w.
        let c = mk_child(1);
        assert!((run(&search, c) - (explore(10.0 + vl_w) + -dist_utility(1.0))).abs() < 1e-12);
        unsafe { drop(Box::from_raw(c)) };

        // blend = 0.0 (pure WU-UCT): utility untouched, only the weight
        // inflation deflates the exploration term.
        let mut params = search.search_params.clone();
        params.virtual_loss_utility_blend = 0.0;
        search.set_params_no_clearing(&params);
        let c = mk_child(1);
        assert!((run(&search, c) - (explore(10.0 + vl_w) + -0.2)).abs() < 1e-12);
        unsafe { drop(Box::from_raw(c)) };

        // blend = 0.5: linear interpolation of the utility distortion.
        let mut params = search.search_params.clone();
        params.virtual_loss_utility_blend = 0.5;
        search.set_params_no_clearing(&params);
        let c = mk_child(1);
        assert!(
            (run(&search, c) - (explore(10.0 + vl_w) + -dist_utility(0.5))).abs() < 1e-12,
            "expected {}, got {}",
            explore(10.0 + vl_w) + -dist_utility(0.5),
            run(&search, c)
        );
        unsafe { drop(Box::from_raw(c)) };

        // Sanity: default in fresh params is 1.0 (official semantics).
        assert_eq!(SearchParams::new().virtual_loss_utility_blend, 1.0);
    }

    fn store_output(node: &SearchNode, nn_output: NNOutput) {
        node.nn_output.store(
            Box::into_raw(Box::new(Arc::new(nn_output))),
            Ordering::Release,
        );
    }

    #[test]
    fn test_recompute_stats_handles_deep_tree_without_native_recursion() {
        const DEPTH: usize = 50_000;
        let mut search = search_with_dummy();
        search.root_pla = P_BLACK;

        fn make_node(next_pla: Player, internal: bool) -> Box<SearchNode> {
            let mut node = SearchNode::new(next_pla, false, 0, Hash128::default());
            if internal {
                node.initialize_children();
                node.state.store(STATE_EXPANDED0, Ordering::Release);
            }
            node.stats.visits.store(1, Ordering::Release);
            node.stats.weight_sum.store(1.0, Ordering::Release);
            node.stats.weight_sq_sum.store(1.0, Ordering::Release);
            let mut nn_output = NNOutput::default();
            nn_output.nn_x_len = 19;
            nn_output.nn_y_len = 19;
            store_output(&node, nn_output);
            Box::new(node)
        }

        let mut root = make_node(P_BLACK, true);
        let mut nodes = Vec::with_capacity(DEPTH - 1);
        for i in 0..DEPTH - 1 {
            let next_pla = if i % 2 == 0 { P_WHITE } else { P_BLACK };
            let internal = i + 2 < DEPTH;
            nodes.push(Box::into_raw(make_node(next_pla, internal)));
        }

        let loc = location::get_loc(0, 0, search.root_board.x_size);
        let mut parent: *mut SearchNode = &mut *root;
        for &child in &nodes {
            unsafe {
                let children = (*parent).get_children();
                children.get(0).store(child);
                children.get(0).set_move_loc(loc);
                children.get(0).set_edge_visits(1);
            }
            parent = child;
        }

        search.root_node = Some(root);
        let root_ptr = search.root_node.as_deref_mut().unwrap() as *mut SearchNode;
        unsafe {
            search.recursively_recompute_stats(&mut *root_ptr);
        }
        let root = search.root_node.as_deref().unwrap();
        assert!(root.stats.weight_sum.load(Ordering::Acquire) > 0.0);

        search.root_node = None;
        for ptr in nodes {
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }

    #[test]
    fn test_get_root_visits_with_fabricated_root() {
        let mut search = search_with_dummy();
        let root = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        root.stats.visits.store(42, Ordering::Release);
        search.root_node = Some(Box::new(root));
        assert_eq!(search.get_root_visits(), 42);
    }

    #[test]
    fn test_get_play_selection_values_with_fabricated_root_and_child() {
        let mut search = search_with_dummy();
        search.root_pla = P_BLACK;

        let mut root = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        root.initialize_children();
        root.state.store(STATE_EXPANDED0, Ordering::Release);
        root.stats.visits.store(1, Ordering::Release);
        root.stats.weight_sum.store(1.0, Ordering::Release);
        root.stats.utility_avg.store(0.0, Ordering::Release);

        let mut nn_output = NNOutput::default();
        nn_output.nn_x_len = 19;
        nn_output.nn_y_len = 19;
        let loc = location::get_loc(3, 3, search.root_board.x_size);
        nn_output.policy_probs[search.get_pos(loc) as usize] = 0.5;
        nn_output.policy_probs[search.get_pos(PASS_LOC) as usize] = 0.5;
        store_output(&root, nn_output);

        let child_ptr = Box::into_raw(Box::new(SearchNode::new(
            P_WHITE,
            false,
            0,
            Hash128::default(),
        )));
        unsafe {
            (*child_ptr).stats.visits.store(10, Ordering::Release);
            (*child_ptr).stats.weight_sum.store(10.0, Ordering::Release);
        }
        root.get_children().get(0).store(child_ptr);
        root.get_children().get(0).set_move_loc(loc);
        root.get_children().get(0).set_edge_visits(10);

        search.root_node = Some(Box::new(root));

        let mut locs = Vec::new();
        let mut values = Vec::new();
        assert!(search.get_play_selection_values(&mut locs, &mut values, 0.0));
        assert_eq!(locs, vec![loc]);
        assert!((values[0] - 10.0).abs() < 1e-9);

        unsafe {
            drop(Box::from_raw(child_ptr));
        }
    }

    #[test]
    fn test_append_pv_and_print_pv_follow_best_child() {
        let mut search = search_with_dummy();
        search.root_pla = P_BLACK;

        let mut root = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        root.initialize_children();
        root.state.store(STATE_EXPANDED0, Ordering::Release);
        root.stats.visits.store(1, Ordering::Release);
        root.stats.weight_sum.store(1.0, Ordering::Release);
        root.stats.utility_avg.store(0.0, Ordering::Release);

        let mut nn_output = NNOutput::default();
        nn_output.nn_x_len = 19;
        nn_output.nn_y_len = 19;
        let loc = location::get_loc(3, 3, search.root_board.x_size);
        nn_output.policy_probs[search.get_pos(loc) as usize] = 0.5;
        nn_output.policy_probs[search.get_pos(PASS_LOC) as usize] = 0.5;
        store_output(&root, nn_output);

        let child_ptr = Box::into_raw(Box::new(SearchNode::new(
            P_WHITE,
            false,
            0,
            Hash128::default(),
        )));
        unsafe {
            (*child_ptr).stats.visits.store(10, Ordering::Release);
            (*child_ptr).stats.weight_sum.store(10.0, Ordering::Release);
        }
        root.get_children().get(0).store(child_ptr);
        root.get_children().get(0).set_move_loc(loc);
        root.get_children().get(0).set_edge_visits(10);

        search.root_node = Some(Box::new(root));

        let mut buf = Vec::new();
        let mut visits_buf = Vec::new();
        let mut edge_visits_buf = Vec::new();
        let mut scratch_locs = Vec::new();
        let mut scratch_values = Vec::new();
        search.append_pv(
            &mut buf,
            &mut visits_buf,
            &mut edge_visits_buf,
            &mut scratch_locs,
            &mut scratch_values,
            search.root_node.as_deref(),
            5,
        );

        assert_eq!(buf, vec![loc]);
        assert_eq!(visits_buf, vec![10]);
        assert_eq!(edge_visits_buf, vec![10]);

        let mut out = String::new();
        search.print_pv(&mut out, search.root_node.as_deref(), 5);
        assert_eq!(
            out,
            location::to_string(loc, search.root_board.x_size, search.root_board.y_size)
        );

        unsafe {
            drop(Box::from_raw(child_ptr));
        }
    }

    #[test]
    fn test_get_analysis_data_reads_child_stats_and_sorts() {
        let mut search = search_with_dummy();
        search.root_pla = P_BLACK;

        let mut root = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        root.initialize_children();
        root.state.store(STATE_EXPANDED0, Ordering::Release);
        root.stats.visits.store(1, Ordering::Release);
        root.stats.weight_sum.store(1.0, Ordering::Release);
        root.stats.utility_avg.store(0.0, Ordering::Release);
        root.stats.win_loss_value_avg.store(0.0, Ordering::Release);
        root.stats.score_mean_avg.store(0.0, Ordering::Release);
        root.stats.score_mean_sq_avg.store(0.0, Ordering::Release);
        root.stats.lead_avg.store(0.0, Ordering::Release);

        let mut nn_output = NNOutput::default();
        nn_output.nn_x_len = 19;
        nn_output.nn_y_len = 19;
        let loc = location::get_loc(3, 3, search.root_board.x_size);
        nn_output.policy_probs[search.get_pos(loc) as usize] = 0.5;
        nn_output.policy_probs[search.get_pos(PASS_LOC) as usize] = 0.5;
        store_output(&root, nn_output);

        let child_ptr = Box::into_raw(Box::new(SearchNode::new(
            P_WHITE,
            false,
            0,
            Hash128::default(),
        )));
        unsafe {
            (*child_ptr).stats.visits.store(10, Ordering::Release);
            (*child_ptr).stats.weight_sum.store(10.0, Ordering::Release);
            (*child_ptr).stats.utility_avg.store(0.1, Ordering::Release);
            (*child_ptr)
                .stats
                .utility_sq_avg
                .store(0.01, Ordering::Release);
            (*child_ptr)
                .stats
                .win_loss_value_avg
                .store(0.05, Ordering::Release);
            (*child_ptr)
                .stats
                .score_mean_avg
                .store(0.0, Ordering::Release);
            (*child_ptr)
                .stats
                .score_mean_sq_avg
                .store(0.0, Ordering::Release);
            (*child_ptr).stats.lead_avg.store(0.0, Ordering::Release);
        }
        root.get_children().get(0).store(child_ptr);
        root.get_children().get(0).set_move_loc(loc);
        root.get_children().get(0).set_edge_visits(10);

        search.root_node = Some(Box::new(root));

        let mut buf = Vec::new();
        search.get_analysis_data(&mut buf, 0, false, 10, false);

        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0].move_loc, loc);
        assert_eq!(buf[0].num_visits, 10);
        assert_eq!(buf[0].child_visits, 10);
        assert_eq!(buf[0].child_weight_sum, 10.0);
        assert!((buf[0].utility - 0.1).abs() < 1e-9);
        assert!((buf[0].win_loss_value - 0.05).abs() < 1e-9);
        assert!((buf[0].policy_prior - 0.5).abs() < 1e-9);
        assert_eq!(buf[0].pv, vec![loc]);
        assert_eq!(buf[0].order, 0);

        // A second child-less call with min_moves_to_try_to_get > 1 should add pass from policy.
        let mut buf2 = Vec::new();
        search.get_analysis_data(&mut buf2, 2, false, 10, false);
        assert_eq!(buf2.len(), 2);
        assert_eq!(buf2[0].move_loc, loc);
        assert_eq!(buf2[1].move_loc, PASS_LOC);
        assert_eq!(buf2[1].num_visits, 0);
        assert!((buf2[1].policy_prior - 0.5).abs() < 1e-9);

        unsafe {
            drop(Box::from_raw(child_ptr));
        }
    }

    #[test]
    fn test_print_tree_outputs_root_and_child() {
        let mut search = search_with_dummy();
        search.root_pla = P_BLACK;
        // Avoid the value-weight-distribution path, which is not initialized in tests.
        search.search_params.value_weight_exponent = 0.0;

        let mut root = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        root.initialize_children();
        root.state.store(STATE_EXPANDED0, Ordering::Release);
        root.stats.visits.store(1, Ordering::Release);
        root.stats.weight_sum.store(1.0, Ordering::Release);
        root.stats.utility_avg.store(0.0, Ordering::Release);
        root.stats.win_loss_value_avg.store(0.0, Ordering::Release);
        root.stats.score_mean_avg.store(0.0, Ordering::Release);
        root.stats.score_mean_sq_avg.store(0.0, Ordering::Release);
        root.stats.lead_avg.store(0.0, Ordering::Release);

        let mut nn_output = NNOutput::default();
        nn_output.nn_x_len = 19;
        nn_output.nn_y_len = 19;
        let loc = location::get_loc(3, 3, search.root_board.x_size);
        nn_output.policy_probs[search.get_pos(loc) as usize] = 0.5;
        nn_output.policy_probs[search.get_pos(PASS_LOC) as usize] = 0.5;
        store_output(&root, nn_output);

        let child_ptr = Box::into_raw(Box::new(SearchNode::new(
            P_WHITE,
            false,
            0,
            Hash128::default(),
        )));
        unsafe {
            (*child_ptr).stats.visits.store(10, Ordering::Release);
            (*child_ptr).stats.weight_sum.store(10.0, Ordering::Release);
            (*child_ptr).stats.utility_avg.store(0.1, Ordering::Release);
            (*child_ptr)
                .stats
                .utility_sq_avg
                .store(0.01, Ordering::Release);
            (*child_ptr)
                .stats
                .win_loss_value_avg
                .store(0.05, Ordering::Release);
            (*child_ptr)
                .stats
                .score_mean_avg
                .store(0.0, Ordering::Release);
            (*child_ptr)
                .stats
                .score_mean_sq_avg
                .store(0.0, Ordering::Release);
            (*child_ptr).stats.lead_avg.store(0.0, Ordering::Release);
        }
        root.get_children().get(0).store(child_ptr);
        root.get_children().get(0).set_move_loc(loc);
        root.get_children().get(0).set_edge_visits(10);

        search.root_node = Some(Box::new(root));

        let options = PrintTreeOptions::new();
        let mut out = String::new();
        search.print_tree(&mut out, search.root_node.as_deref(), &options, P_BLACK);

        // Root line should contain the move coordinate and visit count.
        assert!(out.contains("D16"));
        assert!(out.contains("N       1"));
        // Child line should be indented under the move.
        assert!(out.contains("N      10"));

        unsafe {
            drop(Box::from_raw(child_ptr));
        }
    }

    #[test]
    fn test_print_root_policy_map_outputs_policy() {
        let mut search = search_with_dummy();
        search.root_pla = P_BLACK;

        let root = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        let mut nn_output = NNOutput::default();
        nn_output.nn_x_len = 19;
        nn_output.nn_y_len = 19;
        let loc = location::get_loc(3, 3, search.root_board.x_size);
        nn_output.policy_probs[search.get_pos(loc) as usize] = 0.5;
        store_output(&root, nn_output);
        search.root_node = Some(Box::new(root));

        let mut out = String::new();
        search.print_root_policy_map(&mut out);
        // Only the (3,3) cell should have a non-zero value printed as 50.0.
        assert!(out.contains("  50.0"));
    }

    #[test]
    fn test_print_root_ownership_map_outputs_ownership() {
        let mut search = search_with_dummy();
        search.root_pla = P_BLACK;

        let root = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        let mut nn_output = NNOutput::default();
        nn_output.nn_x_len = 19;
        nn_output.nn_y_len = 19;
        let mut owner_map =
            vec![0.0f32; (nn_output.nn_x_len * nn_output.nn_y_len) as usize].into_boxed_slice();
        let loc = location::get_loc(3, 3, search.root_board.x_size);
        let pos = nn_pos::xy_to_pos(3, 3, nn_output.nn_x_len) as usize;
        owner_map[pos] = 0.75;
        nn_output.white_owner_map = Some(owner_map);
        store_output(&root, nn_output);
        search.root_node = Some(Box::new(root));

        let mut out = String::new();
        search.print_root_ownership_map(&mut out, P_BLACK);
        // From Black's perspective positive white ownership becomes negative.
        assert!(out.contains(" -75.0"));
    }

    #[test]
    fn test_print_root_ending_score_value_bonus_outputs_child() {
        let mut search = search_with_dummy();
        search.root_pla = P_BLACK;

        let mut root = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        root.initialize_children();
        root.state.store(STATE_EXPANDED0, Ordering::Release);

        let mut nn_output = NNOutput::default();
        nn_output.nn_x_len = 19;
        nn_output.nn_y_len = 19;
        let owner_map =
            vec![0.0f32; (nn_output.nn_x_len * nn_output.nn_y_len) as usize].into_boxed_slice();
        nn_output.white_owner_map = Some(owner_map);
        store_output(&root, nn_output);

        let loc = location::get_loc(3, 3, search.root_board.x_size);
        let child_ptr = Box::into_raw(Box::new(SearchNode::new(
            P_WHITE,
            false,
            0,
            Hash128::default(),
        )));
        unsafe {
            (*child_ptr).stats.visits.store(5, Ordering::Release);
            (*child_ptr).stats.weight_sum.store(5.0, Ordering::Release);
            (*child_ptr).stats.utility_avg.store(0.1, Ordering::Release);
            (*child_ptr)
                .stats
                .score_mean_avg
                .store(0.0, Ordering::Release);
            (*child_ptr)
                .stats
                .score_mean_sq_avg
                .store(0.0, Ordering::Release);
        }
        root.get_children().get(0).store(child_ptr);
        root.get_children().get(0).set_move_loc(loc);
        root.get_children().get(0).set_edge_visits(5);

        search.root_node = Some(Box::new(root));

        let mut out = String::new();
        search.print_root_ending_score_value_bonus(&mut out);
        assert!(out.contains("D16"));
        assert!(out.contains("visits 5"));
        assert!(out.contains("edgeVisits 5"));

        unsafe {
            drop(Box::from_raw(child_ptr));
        }
    }

    #[test]
    fn test_get_average_tree_ownership_leaf_node() {
        let mut search = search_with_dummy();
        search.root_pla = P_BLACK;
        search.set_always_include_owner_map(true);

        let root = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        root.stats.visits.store(10, Ordering::Release);
        root.stats.weight_sum.store(10.0, Ordering::Release);

        let mut nn_output = NNOutput::default();
        nn_output.nn_x_len = 19;
        nn_output.nn_y_len = 19;
        let mut owner_map =
            vec![0.0f32; (nn_output.nn_x_len * nn_output.nn_y_len) as usize].into_boxed_slice();
        owner_map[nn_pos::xy_to_pos(3, 3, nn_output.nn_x_len) as usize] = 0.5;
        owner_map[nn_pos::xy_to_pos(4, 4, nn_output.nn_x_len) as usize] = -0.25;
        nn_output.white_owner_map = Some(owner_map);
        store_output(&root, nn_output);

        search.root_node = Some(Box::new(root));

        let ownership = search.get_average_tree_ownership(None);
        assert_eq!(ownership.len(), 19 * 19);
        assert!((ownership[nn_pos::xy_to_pos(3, 3, 19) as usize] - 0.5).abs() < 1e-9);
        assert!((ownership[nn_pos::xy_to_pos(4, 4, 19) as usize] - (-0.25)).abs() < 1e-9);
    }

    #[test]
    fn test_get_average_tree_ownership_with_sym_flips_and_rounds() {
        let mut search = search_with_dummy();
        search.root_pla = P_BLACK;
        search.set_always_include_owner_map(true);

        let root = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        root.stats.visits.store(10, Ordering::Release);
        root.stats.weight_sum.store(10.0, Ordering::Release);

        let mut nn_output = NNOutput::default();
        nn_output.nn_x_len = 19;
        nn_output.nn_y_len = 19;
        let mut owner_map =
            vec![0.0f32; (nn_output.nn_x_len * nn_output.nn_y_len) as usize].into_boxed_slice();
        owner_map[nn_pos::xy_to_pos(3, 3, nn_output.nn_x_len) as usize] = 0.5;
        nn_output.white_owner_map = Some(owner_map);
        store_output(&root, nn_output);

        search.root_node = Some(Box::new(root));

        // From Black's perspective the ownership is negated.
        let ownership = search.get_average_tree_ownership_with_sym(P_BLACK, None, 0);
        assert_eq!(ownership.len(), 19 * 19);
        assert!((ownership[nn_pos::xy_to_pos(3, 3, 19) as usize] - (-0.5)).abs() < 1e-9);
    }

    #[test]
    fn test_get_average_and_std_dev_tree_ownership_leaf_node() {
        let mut search = search_with_dummy();
        search.root_pla = P_BLACK;

        let root = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        root.stats.visits.store(10, Ordering::Release);
        root.stats.weight_sum.store(10.0, Ordering::Release);

        let mut nn_output = NNOutput::default();
        nn_output.nn_x_len = 19;
        nn_output.nn_y_len = 19;
        let mut owner_map =
            vec![0.0f32; (nn_output.nn_x_len * nn_output.nn_y_len) as usize].into_boxed_slice();
        owner_map[nn_pos::xy_to_pos(3, 3, nn_output.nn_x_len) as usize] = 0.5;
        nn_output.white_owner_map = Some(owner_map);
        store_output(&root, nn_output);

        search.root_node = Some(Box::new(root));

        let (average, stdev) = search.get_average_and_std_dev_tree_ownership(None);
        assert_eq!(average.len(), 19 * 19);
        assert_eq!(stdev.len(), 19 * 19);
        assert!((average[nn_pos::xy_to_pos(3, 3, 19) as usize] - 0.5).abs() < 1e-9);
        // With only one deterministic value, variance is zero.
        assert!(stdev[nn_pos::xy_to_pos(3, 3, 19) as usize].abs() < 1e-9);
    }

    #[test]
    fn test_get_policy_surprise_and_entropy() {
        let mut search = search_with_dummy();

        // No NN output -> false and zeroed outputs.
        let root_no_nn = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        search.root_node = Some(Box::new(root_no_nn));
        let mut surprise = 1.0;
        let mut search_entropy = 1.0;
        let mut policy_entropy = 1.0;
        assert!(!search.get_policy_surprise_and_entropy(
            &mut surprise,
            &mut search_entropy,
            &mut policy_entropy
        ));
        assert_eq!(surprise, 0.0);
        assert_eq!(search_entropy, 0.0);
        assert_eq!(policy_entropy, 0.0);
        assert_eq!(search.get_policy_surprise(), 0.0);

        // Fabricated root with one heavy child and a policy that differs from the search target.
        search.clear_search();
        search.root_pla = P_BLACK;

        let mut root = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        root.initialize_children();
        root.state.store(STATE_EXPANDED0, Ordering::Release);
        root.stats.visits.store(1, Ordering::Release);
        root.stats.weight_sum.store(1.0, Ordering::Release);
        root.stats.utility_avg.store(0.0, Ordering::Release);

        let mut nn_output = NNOutput::default();
        nn_output.nn_x_len = 19;
        nn_output.nn_y_len = 19;
        let loc = location::get_loc(3, 3, search.root_board.x_size);
        nn_output.policy_probs[search.get_pos(loc) as usize] = 0.5;
        nn_output.policy_probs[search.get_pos(PASS_LOC) as usize] = 0.5;
        store_output(&root, nn_output);

        let child_ptr = Box::into_raw(Box::new(SearchNode::new(
            P_WHITE,
            false,
            0,
            Hash128::default(),
        )));
        unsafe {
            (*child_ptr).stats.visits.store(10, Ordering::Release);
            (*child_ptr).stats.weight_sum.store(10.0, Ordering::Release);
        }
        root.get_children().get(0).store(child_ptr);
        root.get_children().get(0).set_move_loc(loc);
        root.get_children().get(0).set_edge_visits(10);

        search.root_node = Some(Box::new(root));

        let mut surprise = 0.0;
        let mut search_entropy = 0.0;
        let mut policy_entropy = 0.0;
        assert!(search.get_policy_surprise_and_entropy(
            &mut surprise,
            &mut search_entropy,
            &mut policy_entropy
        ));
        assert!(surprise > 0.0);
        assert!(policy_entropy > 0.0);

        unsafe {
            drop(Box::from_raw(child_ptr));
        }
    }

    #[test]
    fn test_get_chosen_move_loc_picks_best_weighted_move() {
        let mut search = search_with_dummy();
        search.root_pla = P_BLACK;

        let mut root = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        root.initialize_children();
        root.state.store(STATE_EXPANDED0, Ordering::Release);
        root.stats.visits.store(1, Ordering::Release);
        root.stats.weight_sum.store(1.0, Ordering::Release);
        root.stats.utility_avg.store(0.0, Ordering::Release);

        let mut nn_output = NNOutput::default();
        nn_output.nn_x_len = 19;
        nn_output.nn_y_len = 19;
        let loc_a = location::get_loc(3, 3, search.root_board.x_size);
        let loc_b = location::get_loc(4, 4, search.root_board.x_size);
        nn_output.policy_probs[search.get_pos(loc_a) as usize] = 0.5;
        nn_output.policy_probs[search.get_pos(loc_b) as usize] = 0.5;
        store_output(&root, nn_output);

        let child_a = Box::into_raw(Box::new(SearchNode::new(
            P_WHITE,
            false,
            0,
            Hash128::default(),
        )));
        unsafe {
            (*child_a).stats.visits.store(10, Ordering::Release);
            (*child_a).stats.weight_sum.store(10.0, Ordering::Release);
        }
        root.get_children().get(0).store(child_a);
        root.get_children().get(0).set_move_loc(loc_a);
        root.get_children().get(0).set_edge_visits(10);

        let child_b = Box::into_raw(Box::new(SearchNode::new(
            P_WHITE,
            false,
            0,
            Hash128::default(),
        )));
        // Leave child_b with zero visits so its selection weight is zero.
        root.get_children().get(1).store(child_b);
        root.get_children().get(1).set_move_loc(loc_b);
        root.get_children().get(1).set_edge_visits(0);

        search.root_node = Some(Box::new(root));

        assert_eq!(search.get_chosen_move_loc(), loc_a);

        unsafe {
            drop(Box::from_raw(child_a));
            drop(Box::from_raw(child_b));
        }
    }

    #[test]
    fn test_get_node_raw_nn_values_reads_output() {
        let mut search = search_with_dummy();
        search.root_pla = P_BLACK;

        let root = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        let mut nn_output = NNOutput::default();
        nn_output.white_win_prob = 0.6;
        nn_output.white_loss_prob = 0.3;
        nn_output.white_no_result_prob = 0.1;
        nn_output.white_score_mean = 2.5;
        nn_output.white_score_mean_sq = 10.0;
        nn_output.white_lead = 2.0;
        store_output(&root, nn_output);

        search.root_node = Some(Box::new(root));

        let mut values = ReportedSearchValues::new();
        assert!(search.get_root_raw_nn_values(&mut values));
        assert!((values.win_value - 0.6).abs() < 1e-7);
        assert!((values.loss_value - 0.3).abs() < 1e-7);
        assert!((values.no_result_value - 0.1).abs() < 1e-7);
        assert!((values.win_loss_value - 0.3).abs() < 1e-7);
        assert!((values.expected_score - 2.5).abs() < 1e-9);
        assert!((values.lead - 2.0).abs() < 1e-9);
        assert_eq!(values.visits, 1);
    }

    #[test]
    fn test_get_root_values_from_fabricated_stats() {
        let mut search = search_with_dummy();
        search.root_pla = P_BLACK;

        let root = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        root.stats.visits.store(5, Ordering::Release);
        root.stats.weight_sum.store(5.0, Ordering::Release);
        root.stats.win_loss_value_avg.store(0.4, Ordering::Release);
        root.stats.no_result_value_avg.store(0.0, Ordering::Release);
        root.stats.score_mean_avg.store(1.0, Ordering::Release);
        root.stats.score_mean_sq_avg.store(2.0, Ordering::Release);
        root.stats.lead_avg.store(0.5, Ordering::Release);
        root.stats.utility_avg.store(0.3, Ordering::Release);

        search.root_node = Some(Box::new(root));

        let mut values = ReportedSearchValues::new();
        assert!(search.get_root_values(&mut values));
        assert_eq!(values.visits, 5);
        assert!((values.weight - 5.0).abs() < 1e-9);
        assert!((values.utility - 0.3).abs() < 1e-9);
        assert!((values.win_loss_value - 0.4).abs() < 1e-9);
        assert!((values.win_value - 0.7).abs() < 1e-9);
        assert!((values.loss_value - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_perform_task_with_threads_runs_on_multiple_threads() {
        let mut search = search_with_dummy();
        search.search_params.num_threads = 4;

        let seen = std::sync::Mutex::new(HashSet::<i32>::new());
        let task = |thread_idx: i32| {
            seen.lock().unwrap().insert(thread_idx);
        };
        search.perform_task_with_threads(&task, 0x3fff_ffff);

        let seen = seen.into_inner().unwrap();
        assert!(seen.contains(&0));
        assert!(
            seen.len() > 1,
            "expected multiple thread indices, got {:?}",
            seen
        );
    }

    #[test]
    fn test_apply_recursively_any_order_multithreaded_marks_all_nodes() {
        let mut search = search_with_dummy();
        search.search_params.num_threads = 2;

        let mut root = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        root.initialize_children();
        root.state.store(STATE_EXPANDED0, Ordering::Release);
        let child = Box::into_raw(Box::new(SearchNode::new(
            P_WHITE,
            false,
            0,
            Hash128::default(),
        )));
        root.get_children().get(0).store(child);
        search.root_node = Some(Box::new(root));

        let root_ptr = search.root_node.as_deref().unwrap() as *const SearchNode;
        let expected_age = search.search_node_age + 1;
        search
            .apply_recursively_any_order_multithreaded(&[unsafe { &*root_ptr }], &|_node, _idx| {});

        assert_eq!(
            unsafe { &*root_ptr }.node_age.load(Ordering::Acquire),
            expected_age
        );
        assert_eq!(
            unsafe { &*child }.node_age.load(Ordering::Acquire),
            expected_age
        );

        unsafe {
            drop(Box::from_raw(child));
        }
    }

    #[test]
    fn test_enumerate_tree_post_order() {
        let mut search = search_with_dummy();

        let mut root = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        root.initialize_children();
        root.state.store(STATE_EXPANDED0, Ordering::Release);
        let child_a = Box::into_raw(Box::new(SearchNode::new(
            P_WHITE,
            false,
            0,
            Hash128::default(),
        )));
        let child_b = Box::into_raw(Box::new(SearchNode::new(
            P_WHITE,
            false,
            0,
            Hash128::default(),
        )));
        root.get_children().get(0).store(child_a);
        root.get_children().get(1).store(child_b);
        search.root_node = Some(Box::new(root));

        let nodes = search.enumerate_tree_post_order();
        assert_eq!(nodes.len(), 3);
        assert!(nodes.contains(&child_a));
        assert!(nodes.contains(&child_b));
        assert!(nodes.contains(
            &(search.root_node.as_deref().unwrap() as *const SearchNode as *mut SearchNode)
        ));

        unsafe {
            drop(Box::from_raw(child_a));
            drop(Box::from_raw(child_b));
        }
    }

    #[test]
    fn test_print_tree_options_builders() {
        let board = Board::new(19, 19);
        let opts = PrintTreeOptions::new()
            .max_depth(3)
            .max_children_to_show(5)
            .min_visits_to_show(10)
            .min_visits_to_expand(2)
            .min_visits_prop_to_show(0.05)
            .min_visits_prop_to_expand(0.10)
            .max_pv_depth(4)
            .print_raw_nn(true)
            .print_sqs(true)
            .print_avg_shortterm_error(true)
            .only_branch(&board, "D4")
            .also_branch(&board, "Q16");

        assert_eq!(opts.max_depth, 3);
        assert_eq!(opts.max_children_to_show, 5);
        assert_eq!(opts.min_visits_to_show, 10);
        assert_eq!(opts.min_visits_to_expand, 2);
        assert!((opts.min_visits_prop_to_show - 0.05).abs() < 1e-9);
        assert!((opts.min_visits_prop_to_expand - 0.10).abs() < 1e-9);
        assert_eq!(opts.max_pv_depth, 4);
        assert!(opts.print_raw_nn);
        assert!(opts.print_sqs);
        assert!(opts.print_avg_shortterm_error);
        assert_eq!(opts.branch.len(), 1);
        assert!(opts.also_branch);
    }

    #[test]
    fn test_update_mirroring_detects_mirror() {
        let mut search = search_with_dummy();
        search.search_params.anti_mirror = true;
        search.set_player_and_clear_history(P_BLACK);

        // Play 14 pairs of Black/White moves where White mirrors Black.
        for i in 0..14 {
            let x = 2 + i % 5;
            let y = 2 + i / 5;
            let black_loc = location::get_loc(x, y, 19);
            let white_loc = get_mirror_loc(black_loc, 19, 19);
            assert!(search.make_move(black_loc, P_BLACK));
            assert!(search.make_move(white_loc, P_WHITE));
        }

        search.update_mirroring();
        assert_eq!(search.mirroring_pla, P_WHITE);
        assert!(search.mirror_advantage.abs() < 100.0);
        assert!(search.mirror_center_symmetry_error < 1e9);
    }

    #[test]
    fn test_update_mirroring_no_mirror_when_disabled() {
        let mut search = search_with_dummy();
        search.search_params.anti_mirror = false;
        search.set_player_and_clear_history(P_BLACK);
        assert!(search.make_move(location::get_loc(3, 3, 19), P_BLACK));
        assert!(search.make_move(location::get_loc(4, 4, 19), P_WHITE));
        search.update_mirroring();
        assert_eq!(search.mirroring_pla, C_EMPTY);
    }

    #[test]
    fn test_hack_nn_output_for_mirror() {
        let mut search = search_with_dummy();
        search.root_pla = P_BLACK;
        search.mirroring_pla = P_WHITE;
        search.mirror_center_symmetry_error = 1.0;

        let mut output = NNOutput::default();
        output.nn_x_len = 19;
        output.nn_y_len = 19;
        output.white_win_prob = 0.6;
        output.white_loss_prob = 0.3;
        output.white_no_result_prob = 0.1;
        output.white_owner_map = Some(vec![0.5f32; 19 * 19].into_boxed_slice());

        let mut arc = Arc::new(output);
        let total_wl_before = arc.white_win_prob + arc.white_loss_prob;
        search.hack_nn_output_for_mirror(&mut arc);
        let total_wl_after = arc.white_win_prob + arc.white_loss_prob;

        assert!((total_wl_after - total_wl_before).abs() < 1e-5);
        assert!((arc.white_win_prob - 0.6).abs() > 1e-6);
    }

    #[test]
    fn test_get_pruned_root_values_combines_children_and_nn() {
        let mut search = search_with_dummy();
        search.root_pla = P_BLACK;

        let mut root = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        root.initialize_children();
        root.state.store(STATE_EXPANDED0, Ordering::Release);
        root.stats.visits.store(1, Ordering::Release);
        root.stats.weight_sum.store(1.0, Ordering::Release);
        root.stats.utility_avg.store(0.0, Ordering::Release);

        let mut nn_output = NNOutput::default();
        nn_output.nn_x_len = 19;
        nn_output.nn_y_len = 19;
        nn_output.white_win_prob = 0.5;
        nn_output.white_loss_prob = 0.5;
        let loc = location::get_loc(3, 3, search.root_board.x_size);
        nn_output.policy_probs[search.get_pos(loc) as usize] = 0.5;
        nn_output.policy_probs[search.get_pos(PASS_LOC) as usize] = 0.5;
        store_output(&root, nn_output);

        let child_ptr = Box::into_raw(Box::new(SearchNode::new(
            P_WHITE,
            false,
            0,
            Hash128::default(),
        )));
        unsafe {
            (*child_ptr).stats.visits.store(10, Ordering::Release);
            (*child_ptr).stats.weight_sum.store(10.0, Ordering::Release);
            (*child_ptr)
                .stats
                .win_loss_value_avg
                .store(0.2, Ordering::Release);
            (*child_ptr)
                .stats
                .no_result_value_avg
                .store(0.0, Ordering::Release);
            (*child_ptr)
                .stats
                .score_mean_avg
                .store(2.0, Ordering::Release);
            (*child_ptr)
                .stats
                .score_mean_sq_avg
                .store(4.0, Ordering::Release);
            (*child_ptr).stats.lead_avg.store(1.5, Ordering::Release);
            (*child_ptr)
                .stats
                .utility_avg
                .store(0.15, Ordering::Release);
        }
        root.get_children().get(0).store(child_ptr);
        root.get_children().get(0).set_move_loc(loc);
        root.get_children().get(0).set_edge_visits(10);

        search.root_node = Some(Box::new(root));

        let mut values = ReportedSearchValues::new();
        assert!(search.get_pruned_root_values(&mut values));
        assert_eq!(values.visits, 1);
        assert!(values.weight > 0.0);

        unsafe {
            drop(Box::from_raw(child_ptr));
        }
    }

    #[test]
    fn test_get_sharp_score_non_root_leaf_uses_nn_score_mean() {
        let mut search = search_with_dummy();
        search.root_pla = P_BLACK;

        let node = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        let mut nn_output = NNOutput::default();
        nn_output.white_score_mean = 7.5;
        store_output(&node, nn_output);
        node.stats.visits.store(5, Ordering::Release);

        let mut ret = 0.0;
        assert!(search.get_sharp_score(&node, &mut ret));
        assert!((ret - 7.5).abs() < 1e-9);
    }

    #[test]
    fn test_get_sharp_score_helper_below_min_prop_uses_stats() {
        let search = search_with_dummy();

        let node = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        let mut nn_output = NNOutput::default();
        nn_output.white_score_mean = 7.0;
        store_output(&node, nn_output);
        node.stats.visits.store(100, Ordering::Release);
        node.stats.score_mean_avg.store(3.0, Ordering::Release);

        let mut graph_path = HashSet::new();
        let mut buf = [0.0f64; nn_pos::MAX_NN_POLICY_SIZE];
        let mut ret = 0.0;
        assert!(search.get_sharp_score_helper(
            &node,
            &mut graph_path,
            &mut buf,
            0.5,
            0.1,
            &mut ret,
        ));
        assert!((ret - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_get_sharp_score_root_weights_children_by_cubed_play_selection() {
        let mut search = search_with_dummy();
        search.root_pla = P_BLACK;
        search.search_params.value_weight_exponent = 0.0;

        let mut root = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        root.initialize_children();
        root.state.store(STATE_EXPANDED0, Ordering::Release);
        root.stats.visits.store(11, Ordering::Release);
        root.stats.weight_sum.store(11.0, Ordering::Release);
        root.stats.utility_avg.store(0.0, Ordering::Release);

        let mut root_nn = NNOutput::default();
        root_nn.nn_x_len = 19;
        root_nn.nn_y_len = 19;
        root_nn.white_score_mean = 10.0;
        root_nn.white_score_mean_sq = 100.0;
        let loc = location::get_loc(3, 3, search.root_board.x_size);
        root_nn.policy_probs[search.get_pos(loc) as usize] = 0.5;
        root_nn.policy_probs[search.get_pos(PASS_LOC) as usize] = 0.5;
        store_output(&root, root_nn);

        let child_ptr = Box::into_raw(Box::new(SearchNode::new(
            P_WHITE,
            false,
            0,
            Hash128::default(),
        )));
        unsafe {
            (*child_ptr).stats.visits.store(10, Ordering::Release);
            (*child_ptr).stats.weight_sum.store(10.0, Ordering::Release);
            (*child_ptr)
                .stats
                .score_mean_avg
                .store(20.0, Ordering::Release);
            (*child_ptr).stats.utility_avg.store(0.0, Ordering::Release);
        }
        let mut child_nn = NNOutput::default();
        child_nn.nn_x_len = 19;
        child_nn.nn_y_len = 19;
        child_nn.white_score_mean = 20.0;
        unsafe {
            store_output(&*child_ptr, child_nn);
        }
        root.get_children().get(0).store(child_ptr);
        root.get_children().get(0).set_move_loc(loc);
        root.get_children().get(0).set_edge_visits(10);

        search.root_node = Some(Box::new(root));

        let root_ref = search.root_node.as_deref().unwrap();
        let mut ret = 0.0;
        assert!(search.get_sharp_score(root_ref, &mut ret));
        let expected = 20.0 * 10.0 / 11.0 + 10.0 / 11.0;
        assert!(
            (ret - expected).abs() < 1e-9,
            "ret={} expected={}",
            ret,
            expected
        );

        unsafe {
            drop(Box::from_raw(child_ptr));
        }
    }

    #[test]
    fn test_get_shallow_average_shortterm_error_unsupported() {
        let mut search = search_with_dummy();
        let root = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        search.root_node = Some(Box::new(root));

        let (wl, score) = search.get_shallow_average_shortterm_wl_and_score_error(None);
        assert!((wl + 1.0).abs() < 1e-9);
        assert!((score + 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_shallow_error_leaf_uses_nn_output() {
        let search = search_with_dummy();

        let node = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        let mut nn_output = NNOutput::default();
        nn_output.shortterm_winloss_error = 0.2;
        nn_output.shortterm_score_error = 1.5;
        store_output(&node, nn_output);

        let mut graph_path = HashSet::new();
        let mut buf = [0.0f64; nn_pos::MAX_NN_POLICY_SIZE];
        let mut wl = 0.0;
        let mut score = 0.0;
        search.get_shallow_average_shortterm_wl_and_score_error_helper(
            &node,
            &mut graph_path,
            &mut buf,
            0.25,
            1.0,
            &mut wl,
            &mut score,
        );
        assert!((wl - 0.2f64).abs() < 1e-6);
        assert!((score - 1.5f64).abs() < 1e-9);
    }

    #[test]
    fn test_get_analysis_json_produces_move_infos_and_root_info() {
        let mut search = search_with_dummy();
        search.root_pla = P_BLACK;
        search.search_params.value_weight_exponent = 0.0;

        let mut root = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        root.initialize_children();
        root.state.store(STATE_EXPANDED0, Ordering::Release);
        root.stats.visits.store(11, Ordering::Release);
        root.stats.weight_sum.store(11.0, Ordering::Release);
        root.stats.win_loss_value_avg.store(0.2, Ordering::Release);
        root.stats.score_mean_avg.store(1.0, Ordering::Release);
        root.stats.score_mean_sq_avg.store(2.0, Ordering::Release);
        root.stats.lead_avg.store(0.5, Ordering::Release);
        root.stats.utility_avg.store(0.1, Ordering::Release);

        let mut root_nn = NNOutput::default();
        root_nn.nn_x_len = 19;
        root_nn.nn_y_len = 19;
        root_nn.white_win_prob = 0.6;
        root_nn.white_loss_prob = 0.3;
        root_nn.white_no_result_prob = 0.1;
        root_nn.white_score_mean = 2.0;
        root_nn.white_score_mean_sq = 5.0;
        root_nn.white_lead = 1.5;
        let loc = location::get_loc(3, 3, search.root_board.x_size);
        root_nn.policy_probs[search.get_pos(loc) as usize] = 0.5;
        root_nn.policy_probs[search.get_pos(PASS_LOC) as usize] = 0.5;
        store_output(&root, root_nn);

        let child_ptr = Box::into_raw(Box::new(SearchNode::new(
            P_WHITE,
            false,
            0,
            Hash128::default(),
        )));
        unsafe {
            (*child_ptr).stats.visits.store(10, Ordering::Release);
            (*child_ptr).stats.weight_sum.store(10.0, Ordering::Release);
            (*child_ptr)
                .stats
                .win_loss_value_avg
                .store(0.15, Ordering::Release);
            (*child_ptr)
                .stats
                .score_mean_avg
                .store(2.0, Ordering::Release);
            (*child_ptr)
                .stats
                .score_mean_sq_avg
                .store(5.0, Ordering::Release);
            (*child_ptr).stats.lead_avg.store(1.0, Ordering::Release);
            (*child_ptr)
                .stats
                .utility_avg
                .store(0.05, Ordering::Release);
        }
        root.get_children().get(0).store(child_ptr);
        root.get_children().get(0).set_move_loc(loc);
        root.get_children().get(0).set_edge_visits(10);

        search.root_node = Some(Box::new(root));

        let mut ret = Value::Null;
        assert!(search.get_analysis_json(
            P_BLACK, 10, false, false, false, false, false, false, false, false, &mut ret,
        ));
        let move_infos = ret.get("moveInfos").unwrap().as_array().unwrap();
        assert_eq!(move_infos.len(), 1);
        assert_eq!(move_infos[0].get("move").unwrap().as_str().unwrap(), "D16");
        assert!(move_infos[0].get("visits").unwrap().as_i64().unwrap() > 0);
        let root_info = ret.get("rootInfo").unwrap().as_object().unwrap();
        assert!(root_info.get("visits").unwrap().as_i64().unwrap() > 0);

        unsafe {
            drop(Box::from_raw(child_ptr));
        }
    }

    #[test]
    fn test_get_analysis_json_includes_policy_when_requested() {
        let mut search = search_with_dummy();
        search.root_pla = P_BLACK;
        search.search_params.value_weight_exponent = 0.0;

        let mut root = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        root.initialize_children();
        root.state.store(STATE_EXPANDED0, Ordering::Release);
        root.stats.visits.store(1, Ordering::Release);
        root.stats.weight_sum.store(1.0, Ordering::Release);
        root.stats.utility_avg.store(0.0, Ordering::Release);

        let mut root_nn = NNOutput::default();
        root_nn.nn_x_len = 19;
        root_nn.nn_y_len = 19;
        root_nn.policy_probs[search.get_pos(PASS_LOC) as usize] = 0.5;
        store_output(&root, root_nn);

        search.root_node = Some(Box::new(root));

        let mut ret = Value::Null;
        assert!(search.get_analysis_json(
            P_BLACK, 10, false, true, false, false, false, false, false, false, &mut ret,
        ));
        let policy = ret.get("policy").unwrap().as_array().unwrap();
        let expected_len = (search.root_board.x_size * search.root_board.y_size + 1) as usize;
        assert_eq!(policy.len(), expected_len);
    }

    #[test]
    fn test_get_analysis_json_includes_ownership_when_requested() {
        let mut search = search_with_dummy();
        search.root_pla = P_BLACK;
        search.search_params.value_weight_exponent = 0.0;
        search.always_include_owner_map = true;

        let root = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        root.stats.visits.store(1, Ordering::Release);
        root.stats.weight_sum.store(1.0, Ordering::Release);
        root.stats.utility_avg.store(0.0, Ordering::Release);

        let mut root_nn = NNOutput::default();
        root_nn.nn_x_len = 19;
        root_nn.nn_y_len = 19;
        root_nn.white_owner_map = Some(vec![0.5f32; 19 * 19].into_boxed_slice());
        store_output(&root, root_nn);

        search.root_node = Some(Box::new(root));

        let mut ret = Value::Null;
        assert!(search.get_analysis_json(
            P_BLACK, 10, false, false, true, false, false, false, false, false, &mut ret,
        ));
        let ownership = ret.get("ownership").unwrap().as_array().unwrap();
        let expected_len = (search.root_board.x_size * search.root_board.y_size) as usize;
        assert_eq!(ownership.len(), expected_len);
    }

    #[test]
    fn test_respawn_threads_resets_spawned_count() {
        let mut search = search_with_dummy();
        search.search_params.num_threads = 4;
        search.respawn_threads();
        assert_eq!(search.num_threads_spawned, 3);

        search.search_params.num_threads = 1;
        search.respawn_threads();
        assert_eq!(search.num_threads_spawned, 0);
    }

    #[test]
    fn test_maybe_recompute_root_nn_output_updates_stale_root_age() {
        let mut search = search_with_dummy();
        search.search_node_age = 5;
        search.root_pla = P_BLACK;
        search.search_params.value_weight_exponent = 0.0;
        search.search_params.root_noise_enabled = false;
        search.search_params.root_policy_temperature = 1.0;
        search.search_params.root_policy_temperature_early = 1.0;
        search.search_params.root_num_symmetries_to_sample = 1;
        search.search_params.conservative_pass = false;
        search.search_params.ignore_pre_root_history = false;
        search.search_params.root_policy_optimism = search.search_params.policy_optimism;

        let root = SearchNode::new(P_BLACK, false, 0, Hash128::default());
        root.stats.visits.store(1, Ordering::Release);
        root.stats.weight_sum.store(1.0, Ordering::Release);
        root.stats.utility_avg.store(0.0, Ordering::Release);

        let mut root_nn = NNOutput::default();
        root_nn.nn_x_len = 19;
        root_nn.nn_y_len = 19;
        root_nn.white_owner_map = Some(vec![0.0f32; 19 * 19].into_boxed_slice());
        store_output(&root, root_nn);

        search.root_node = Some(Box::new(root));
        let root_ptr = search.root_node.as_deref().unwrap() as *const SearchNode;
        unsafe { &*root_ptr }
            .node_age
            .store(search.search_node_age - 1, Ordering::Release);

        search.maybe_recompute_root_nn_output();

        let new_age = unsafe { &*root_ptr }.node_age.load(Ordering::Acquire);
        assert_eq!(new_age, search.search_node_age);
    }
}
