//! KataGo opening book engine.
//!
//! Corresponds to `KataGo/cpp/book/book.h` and `book.cpp`.
//! This crate is a work in progress: it currently exposes the core book hash,
//! symmetry, and node-walking logic used to build and query an opening book.

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(dangerous_implicit_autorefs)]

pub mod css_js;

use kata_core::config::{ConfigError, ConfigParser};
use kata_core::global::IOError;
use kata_core::hash;
use kata_core::hash::Hash128;
use kata_game::board::{
    Board, C_BLACK, C_WHITE, Loc, P_BLACK, P_WHITE, PASS_LOC, Player, location, player_io,
};
use kata_game::graph_hash;
use kata_game::history::BoardHistory;
use kata_game::rules::Rules;
use kata_game::symmetry;
use kata_search::mutex_pool::MutexPool;
use serde_json::{Value, json};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, BufRead, Write};
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign};
use std::path::Path;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering as AtomicOrdering};
use std::thread;

const DRAW_EQUIVALENT_WINS_FOR_WHITE: f64 = 0.5;
const NULL_NODE_IDX: usize = usize::MAX;

/// Hash identifying a book node, combining history and state hashes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BookHash {
    pub history_hash: Hash128,
    pub state_hash: Hash128,
}

/// Result of [`BookHash::get_hash_and_symmetry`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashAndSymmetry {
    /// The canonical hash for the position.
    pub hash: BookHash,
    /// Symmetry to apply so that `hist` aligns with book nodes for its current
    /// position (`hist` space -> node space).
    pub symmetry_to_align: i32,
    /// Symmetries under which the canonical book node is invariant.
    pub symmetries: Vec<i32>,
}

impl BookHash {
    /// Construct a book hash from history and state hashes.
    pub fn new(history_hash: Hash128, state_hash: Hash128) -> Self {
        Self {
            history_hash,
            state_hash,
        }
    }

    /// Get the book hash for `hist`, the symmetry to apply so that `hist`
    /// aligns with book nodes, and the symmetries under which the resulting
    /// book node is invariant.
    ///
    /// `rep_bound` controls when a simple repetition forces the accumulated
    /// history hash to be reset, matching the C++ graph-hash behaviour.
    /// `book_version` selects the hashing scheme: version 2+ uses
    /// `graph_hash::get_state_hash`, older versions use a simpler positional
    /// hash.
    pub fn get_hash_and_symmetry(
        hist: &BoardHistory,
        rep_bound: i32,
        book_version: i32,
    ) -> HashAndSymmetry {
        let board = hist.get_recent_board(0);
        let num_symmetries = if board.x_size != board.y_size {
            symmetry::NUM_SYMMETRIES_WITHOUT_TRANSPOSE
        } else {
            symmetry::NUM_SYMMETRIES
        };

        let mut boards_by_sym: Vec<Board> = Vec::with_capacity(num_symmetries as usize);
        let mut hists_by_sym: Vec<BoardHistory> = Vec::with_capacity(num_symmetries as usize);
        let mut accums: Vec<Hash128> = Vec::with_capacity(num_symmetries as usize);

        for symmetry in 0..num_symmetries {
            let sym_board = symmetry::get_sym_board(&hist.initial_board, symmetry);
            let sym_hist = BoardHistory::new(
                sym_board.clone(),
                hist.initial_pla,
                hist.rules,
                hist.initial_encore_phase,
            );
            boards_by_sym.push(sym_board);
            hists_by_sym.push(sym_hist);
            accums.push(Hash128::default());
        }

        for m in &hist.move_history {
            for symmetry in 0..num_symmetries as usize {
                let move_loc =
                    symmetry::get_sym_loc(m.loc, &boards_by_sym[symmetry], symmetry as i32);
                let move_pla = m.pla;

                let next_hash = if book_version >= 2 {
                    graph_hash::get_state_hash(
                        &hists_by_sym[symmetry],
                        hists_by_sym[symmetry].presumed_next_move_pla,
                        DRAW_EQUIVALENT_WINS_FOR_WHITE,
                    )
                } else {
                    boards_by_sym[symmetry].pos_hash ^ Board::zobrist_player_hash(move_pla as usize)
                };

                accums[symmetry].hash0 = accums[symmetry].hash0.wrapping_add(next_hash.hash0);
                accums[symmetry].hash1 = accums[symmetry].hash1.wrapping_add(next_hash.hash1);
                accums[symmetry].hash0 = hash::split_mix64(accums[symmetry].hash0);
                accums[symmetry].hash1 = hash::nasam(accums[symmetry].hash1);

                hists_by_sym[symmetry].make_board_move_assume_legal(
                    &mut boards_by_sym[symmetry],
                    move_loc,
                    move_pla,
                );

                if boards_by_sym[symmetry].simple_repetition_bound_gt(move_loc, rep_bound) {
                    accums[symmetry] = Hash128::default();
                }
            }
        }

        let mut hashes: Vec<BookHash> = Vec::with_capacity(num_symmetries as usize);
        for symmetry in 0..num_symmetries as usize {
            let state_hash = graph_hash::get_state_hash(
                &hists_by_sym[symmetry],
                hist.presumed_next_move_pla,
                DRAW_EQUIVALENT_WINS_FOR_WHITE,
            );
            let history_hash = accums[symmetry] ^ extra_pos_hash(&boards_by_sym[symmetry]);
            hashes.push(BookHash::new(history_hash, state_hash));
        }

        if book_version >= 2 {
            for h in &mut hashes {
                h.history_hash.hash0 = hash::murmur_mix(h.history_hash.hash0);
                h.history_hash.hash1 = hash::murmur_mix(h.history_hash.hash1);
                h.state_hash.hash0 = hash::murmur_mix(h.state_hash.hash0);
                h.state_hash.hash1 = hash::murmur_mix(h.state_hash.hash1);
            }
        }

        let mut smallest_symmetry = 0;
        let mut smallest_hash = hashes[0];
        for (symmetry, &hash) in hashes
            .iter()
            .enumerate()
            .take(num_symmetries as usize)
            .skip(1)
        {
            if hash < smallest_hash {
                smallest_symmetry = symmetry as i32;
                smallest_hash = hash;
            }
        }

        let mut symmetries = Vec::new();
        for symmetry in 0..num_symmetries {
            let composed = symmetry::compose(smallest_symmetry, symmetry);
            if hashes[composed as usize] == smallest_hash {
                symmetries.push(symmetry);
            }
        }

        HashAndSymmetry {
            hash: smallest_hash,
            symmetry_to_align: smallest_symmetry,
            symmetries,
        }
    }

    /// Serialize as a 64-character hex string: state hash followed by history hash.
    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> String {
        format!("{}{}", self.state_hash, self.history_hash)
    }

    /// Parse the 64-character hex representation produced by [`Self::to_string`].
    pub fn from_string(s: &str) -> Result<Self, kata_core::global::IOError> {
        if s.len() != 64 {
            return Err(kata_core::global::IOError(format!(
                "BookHash::from_string expected 64 hex chars, got {}",
                s.len()
            )));
        }
        let state_hash = Hash128::from_hex_string(&s[..32])?;
        let history_hash = Hash128::from_hex_string(&s[32..])?;
        Ok(Self::new(history_hash, state_hash))
    }
}

impl PartialOrd for BookHash {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BookHash {
    fn cmp(&self, other: &Self) -> Ordering {
        self.state_hash
            .cmp(&other.state_hash)
            .then_with(|| self.history_hash.cmp(&other.history_hash))
    }
}

impl BitXor for BookHash {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self::new(
            self.history_hash ^ rhs.history_hash,
            self.state_hash ^ rhs.state_hash,
        )
    }
}

impl BitXorAssign for BookHash {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.history_hash ^= rhs.history_hash;
        self.state_hash ^= rhs.state_hash;
    }
}

impl BitOr for BookHash {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self::new(
            self.history_hash | rhs.history_hash,
            self.state_hash | rhs.state_hash,
        )
    }
}

impl BitOrAssign for BookHash {
    fn bitor_assign(&mut self, rhs: Self) {
        self.history_hash |= rhs.history_hash;
        self.state_hash |= rhs.state_hash;
    }
}

impl BitAnd for BookHash {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        Self::new(
            self.history_hash & rhs.history_hash,
            self.state_hash & rhs.state_hash,
        )
    }
}

impl BitAndAssign for BookHash {
    fn bitand_assign(&mut self, rhs: Self) {
        self.history_hash &= rhs.history_hash;
        self.state_hash &= rhs.state_hash;
    }
}

fn round_double(x: f64, inv_min_prec: f64) -> f64 {
    (x * inv_min_prec).round() / inv_min_prec
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

fn get_f64(value: &Value, key: &str) -> Result<f64, IOError> {
    value[key]
        .as_f64()
        .ok_or_else(|| IOError(format!("Could not parse f64 for key: {key}")))
}

fn extra_pos_hash(board: &Board) -> Hash128 {
    let mut hash = Hash128::default();
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = location::get_loc(x, y, board.x_size) as usize;
            let color = board.colors[loc] as usize;
            hash ^= Board::zobrist_board_hash2(loc, color);
        }
    }
    hash
}

fn mutex_index_for_hash(hash: BookHash, num_mutexes: u32) -> u32 {
    let xor = hash.history_hash ^ hash.state_hash;
    ((xor.hash0 ^ xor.hash1) % num_mutexes as u64) as u32
}

/// Recompute adjusted visits for a node using raw pointers for multithreaded use.
#[allow(clippy::too_many_arguments)]
unsafe fn recompute_adjusted_visits_raw(
    book: *mut Book,
    node_idx: usize,
    not_in_book_visits: f64,
    not_in_book_max_raw_policy: f64,
    not_in_book_wl: f64,
    not_in_book_score_mean: f64,
    not_in_book_sharp_score_mean: f64,
    not_in_book_score_lcb: f64,
    not_in_book_score_ucb: f64,
) {
    let params = (*book).params;
    let node_ptr = (*book).nodes.as_ptr().add(node_idx);
    let pla_factor = if (*node_ptr).pla == 1 { 1.0 } else { -1.0 };
    let moves_len = (*node_ptr).moves.len();
    let mut sort_indices: Vec<usize> = Vec::with_capacity(moves_len + 1);
    let mut sort_values: Vec<f64> = Vec::with_capacity(moves_len + 1);
    let mut child_adjusted_visits: Vec<f64> = Vec::with_capacity(moves_len + 1);

    for (i, (_, book_move)) in (*node_ptr).moves.iter().enumerate() {
        let child_idx = *(*book)
            .node_idx_by_hash
            .get(&book_move.hash)
            .expect("missing child node");
        let child_values = &(*(*book).nodes.as_ptr().add(child_idx)).recursive_values;
        sort_indices.push(i);
        sort_values.push(Book::get_sorting_value_params(
            &params,
            pla_factor,
            child_values.win_loss_value,
            child_values.score_mean,
            child_values.sharp_score_mean,
            child_values.score_lcb,
            child_values.score_ucb,
            book_move.raw_policy,
        ));
        child_adjusted_visits.push(child_values.adjusted_visits);
    }

    let i = sort_indices.len();
    sort_indices.push(i);
    sort_values.push(Book::get_sorting_value_params(
        &params,
        pla_factor,
        not_in_book_wl,
        not_in_book_score_mean,
        not_in_book_sharp_score_mean,
        not_in_book_score_lcb,
        not_in_book_score_ucb,
        not_in_book_max_raw_policy,
    ));
    child_adjusted_visits.push(not_in_book_visits);
    let num_items = sort_indices.len();

    sort_indices.sort_by(|&a, &b| {
        sort_values[a]
            .partial_cmp(&sort_values[b])
            .unwrap_or(Ordering::Equal)
    });

    let mut wsum = 0.0;
    let mut wvsum = 0.0;
    let mut prev_sorting_value = -1e100;
    let mut caps: Vec<f64> = Vec::with_capacity(num_items);
    for &idx in &sort_indices {
        let time_elapsed = sort_values[idx] - prev_sorting_value;
        prev_sorting_value = sort_values[idx];
        let factor = (-time_elapsed).exp();
        wsum *= factor;
        wvsum *= factor;
        wsum += 1.0;
        wvsum += (1.0 + params.visits_scale * 0.05 + child_adjusted_visits[idx]).ln();
        let ewma_visits = (wvsum / wsum).exp();
        let adjusted_visits_cap = ewma_visits.max(child_adjusted_visits[idx]);
        caps.push(adjusted_visits_cap);
    }

    let mut adjusted_visits = 0.0;
    let mut lowest_cap_so_far: f64 = 1e100;
    for i in (0..num_items).rev() {
        lowest_cap_so_far = lowest_cap_so_far.min(caps[i]);
        adjusted_visits += (4.0 * lowest_cap_so_far + params.visits_scale)
            .min(child_adjusted_visits[sort_indices[i]]);
    }

    let node = &mut (*book).nodes[node_idx];
    node.recursive_values.adjusted_visits = adjusted_visits;
}

/// Recompute minimaxed recursive values for a node using raw pointers.
unsafe fn recompute_node_values_raw(book: *mut Book, node_idx: usize) {
    let params = (*book).params;
    let rules = (*book).initial_rules;
    let node_ptr = (*book).nodes.as_ptr().add(node_idx);
    let mut win_loss_value;
    let mut score_mean;
    let mut sharp_score_mean;
    let mut win_loss_lcb;
    let mut score_lcb;
    let mut score_final_lcb;
    let mut win_loss_ucb;
    let mut score_ucb;
    let mut score_final_ucb;
    let mut weight = 0.0;
    let mut visits = 0.0;

    {
        let values = &(*node_ptr).this_values_not_in_book;
        let score_error = values.get_adjusted_score_error(&rules);
        let win_loss_error = values.get_adjusted_win_loss_error(&rules);
        win_loss_value = values.win_loss_value;
        score_mean = values.score_mean;
        sharp_score_mean = values.sharp_score_mean_raw;
        win_loss_lcb = values.win_loss_value - params.error_factor * win_loss_error;
        score_lcb = values.score_mean - params.error_factor * score_error;
        score_final_lcb = values.score_mean - params.error_factor * values.score_stdev;
        win_loss_ucb = values.win_loss_value + params.error_factor * win_loss_error;
        score_ucb = values.score_mean + params.error_factor * score_error;
        score_final_ucb = values.score_mean + params.error_factor * values.score_stdev;
        weight += values.weight;
        visits += values.visits;

        if score_error > 0.0 {
            if sharp_score_mean > score_ucb {
                score_ucb = sharp_score_mean;
            }
            if sharp_score_mean < score_lcb {
                score_lcb = sharp_score_mean;
            }
        }
        let cap = params.sharp_score_outlier_cap;
        if sharp_score_mean > score_mean + cap {
            sharp_score_mean = score_mean + cap;
        }
        if sharp_score_mean < score_mean - cap {
            sharp_score_mean = score_mean - cap;
        }
    }

    recompute_adjusted_visits_raw(
        book,
        node_idx,
        visits,
        (*node_ptr).this_values_not_in_book.max_policy,
        win_loss_value,
        score_mean,
        sharp_score_mean,
        score_lcb,
        score_ucb,
    );

    let node_ptr = (*book).nodes.as_ptr().add(node_idx);
    for book_move in (*node_ptr).moves.values() {
        let child_idx = *(*book)
            .node_idx_by_hash
            .get(&book_move.hash)
            .expect("missing child node");
        let child_values = &(*(*book).nodes.as_ptr().add(child_idx)).recursive_values;
        if (*node_ptr).pla == 1 {
            win_loss_value = win_loss_value.max(child_values.win_loss_value);
            score_mean = score_mean.max(child_values.score_mean);
            sharp_score_mean = sharp_score_mean.max(child_values.sharp_score_mean);
            win_loss_lcb = win_loss_lcb.max(child_values.win_loss_lcb);
            score_lcb = score_lcb.max(child_values.score_lcb);
            score_final_lcb = score_final_lcb.max(child_values.score_final_lcb);
            win_loss_ucb = win_loss_ucb.max(child_values.win_loss_ucb);
            score_ucb = score_ucb.max(child_values.score_ucb);
            score_final_ucb = score_final_ucb.max(child_values.score_final_ucb);
        } else {
            win_loss_value = win_loss_value.min(child_values.win_loss_value);
            score_mean = score_mean.min(child_values.score_mean);
            sharp_score_mean = sharp_score_mean.min(child_values.sharp_score_mean);
            win_loss_lcb = win_loss_lcb.min(child_values.win_loss_lcb);
            score_lcb = score_lcb.min(child_values.score_lcb);
            score_final_lcb = score_final_lcb.min(child_values.score_final_lcb);
            win_loss_ucb = win_loss_ucb.min(child_values.win_loss_ucb);
            score_ucb = score_ucb.min(child_values.score_ucb);
            score_final_ucb = score_final_ucb.min(child_values.score_final_ucb);
        }
        weight += child_values.weight;
        visits += child_values.visits;
    }

    let values = &mut (*book).nodes[node_idx].recursive_values;
    values.win_loss_value = win_loss_value;
    values.score_mean = score_mean;
    values.sharp_score_mean = sharp_score_mean;
    values.win_loss_lcb = win_loss_lcb;
    values.score_lcb = score_lcb;
    values.score_final_lcb = score_final_lcb;
    values.win_loss_ucb = win_loss_ucb;
    values.score_ucb = score_ucb;
    values.score_final_ucb = score_final_ucb;
    values.weight = weight;
    values.visits = visits;
}

/// Recompute path costs for a node using raw pointers.
#[allow(clippy::too_many_lines)]
unsafe fn recompute_node_cost_raw(book: *mut Book, idx: usize) {
    let params = (*book).params;
    let rules = (*book).initial_rules;
    let nodes = (*book).nodes.as_mut_ptr();
    let node_ptr = nodes.add(idx);
    let pla = (*node_ptr).pla;
    let is_white = pla == P_WHITE;
    let pla_factor = if is_white { 1.0 } else { -1.0 };
    let hash = (*node_ptr).hash;
    let this_values = (*node_ptr).this_values_not_in_book;
    let recursive_values = (*node_ptr).recursive_values;

    #[derive(Debug, Clone, Copy)]
    struct MoveCostData {
        loc: Loc,
        child_idx: usize,
        raw_policy: f64,
        log_raw_policy: f64,
        child_values: RecursiveBookValues,
        child_utility: f64,
        cost_from_root: f64,
        biggest_wl_cost_from_root: f64,
        is_wl_pv: bool,
    }

    fn utility_of(values: &RecursiveBookValues, params: BookParams) -> f64 {
        values.win_loss_value + values.score_mean * params.utility_per_score
    }

    fn boost_log_raw_policy(
        move_data: &[MoveCostData],
        is_white: bool,
        params: BookParams,
        log_raw_policy: f64,
        child_utility: f64,
        raw_policy: f64,
    ) -> f64 {
        let mut boosted = log_raw_policy;
        for other in move_data {
            if other.raw_policy <= raw_policy {
                continue;
            }
            let gain_over_other = if is_white {
                child_utility - other.child_utility
            } else {
                other.child_utility - child_utility
            };
            if gain_over_other <= 0.0 {
                continue;
            }
            let policy_boost_factor = 2.0
                / (1.0 + (-gain_over_other / params.policy_boost_soft_utility_scale).exp())
                - 1.0;
            let policy_boost_factor = 0.1 + 0.9 * policy_boost_factor;
            let p = log_raw_policy + policy_boost_factor * (other.log_raw_policy - log_raw_policy);
            if p > boosted {
                boosted = p;
            }
        }
        boosted
    }

    fn behind_in_visits_bonus(
        move_data: &[MoveCostData],
        is_white: bool,
        params: BookParams,
        best_win_loss: f64,
        child_win_loss: f64,
        adjusted_visits: f64,
    ) -> f64 {
        let mut max_bonus: f64 = 0.0;
        for other in move_data {
            let other_visits = other.child_values.adjusted_visits;
            if other_visits <= 30.0 * adjusted_visits {
                continue;
            }
            let other_child_win_loss = other.child_values.win_loss_value;
            let gain_over_other = if is_white {
                (child_win_loss + pow3(child_win_loss))
                    - (other_child_win_loss + pow3(other_child_win_loss))
            } else {
                (other_child_win_loss + pow3(other_child_win_loss))
                    - (child_win_loss + pow3(child_win_loss))
            };
            if gain_over_other <= -2.0 * params.policy_boost_soft_utility_scale {
                continue;
            }
            let mut this_bonus = (other_visits / (30.0 * adjusted_visits)).log10()
                - 0.40
                    * ((adjusted_visits.max(params.visits_scale_leaves))
                        / params.visits_scale_leaves)
                        .log10();
            if gain_over_other < 0.0 {
                let factor = (gain_over_other + 2.0 * params.policy_boost_soft_utility_scale)
                    / (2.0 * params.policy_boost_soft_utility_scale + 1e-10);
                this_bonus = this_bonus * factor * factor;
            }
            max_bonus = max_bonus.max(this_bonus);
        }
        if max_bonus <= 0.0 {
            return 0.0;
        }
        let gain_over_best = if is_white {
            (child_win_loss + pow3(child_win_loss)) - (best_win_loss + pow3(best_win_loss))
        } else {
            (best_win_loss + pow3(best_win_loss)) - (child_win_loss + pow3(child_win_loss))
        };
        let gain_over_best = gain_over_best.min(0.0);
        let losing_scale = 1.0_f64.min(if is_white {
            child_win_loss + 1.0
        } else {
            1.0 - child_win_loss
        });
        max_bonus
            * (gain_over_best / (3.0 * params.policy_boost_soft_utility_scale)).exp()
            * losing_scale
    }

    // 1. Update minDepthFromRoot, minCostFromRoot, etc. from parents.
    let (min_depth, mut min_cost, mut min_cost_wl_pv, biggest_wl_cost, best_parent_idx) = if idx
        == 0
    {
        (0, 0.0, 0.0, 0.0, 0)
    } else {
        let mut min_depth = i32::MAX;
        let mut min_cost = 1e100;
        let mut min_cost_wl_pv: f64 = 1e100;
        let mut best_biggest_wl_cost = 1e100;
        let mut best_parent_idx = 0usize;
        for (parent_idx, (parent_hash, parent_move_loc)) in (*node_ptr).parents.iter().enumerate() {
            let parent_idx2 = *(*book)
                .node_idx_by_hash
                .get(parent_hash)
                .expect("missing parent node");
            let parent = &(*(*book).nodes.as_ptr().add(parent_idx2));
            let book_move = parent
                .moves
                .get(parent_move_loc)
                .expect("missing parent move");
            let depth = parent.min_depth_from_root + 1;
            let cost = book_move.cost_from_root;
            if cost < min_cost {
                min_cost = cost;
                best_biggest_wl_cost = book_move.biggest_wl_cost_from_root;
                best_parent_idx = parent_idx;
            }
            if book_move.is_wl_pv && parent.min_cost_from_root_wl_pv < min_cost_wl_pv {
                min_cost_wl_pv = parent.min_cost_from_root_wl_pv;
            }
            if depth < min_depth {
                min_depth = depth;
            }
        }
        (
            min_depth,
            min_cost,
            min_cost_wl_pv,
            best_biggest_wl_cost,
            best_parent_idx,
        )
    };

    // 2. Apply user-specified bonuses.
    if let Some(&bonus) = (*book).bonus_by_hash.get(&hash) {
        min_cost -= bonus;
    }
    if let Some(&visits_required) = (*book).visits_required_by_hash.get(&hash) {
        let denom = (visits_required / params.visits_scale.max(1.0))
            .powf(0.1)
            .max(1.0);
        if recursive_values.visits < visits_required
            || recursive_values.adjusted_visits < 0.5 * visits_required / denom
        {
            min_cost -= 500.0;
        }
    }

    // 3. Clamp minCostFromRootWLPV to minCostFromRoot.
    if min_cost < min_cost_wl_pv {
        min_cost_wl_pv = min_cost;
    }

    // Collect child data and find the win/loss PV.
    let mut move_data: Vec<MoveCostData> = Vec::with_capacity((*node_ptr).moves.len());
    let mut pass_policy = 0.0;
    let mut pass_utility = if is_white { -1e100 } else { 1e100 };
    for (&loc, &book_move) in (*node_ptr).moves.iter() {
        let child_idx = *(*book)
            .node_idx_by_hash
            .get(&book_move.hash)
            .expect("missing child node");
        let child_ptr = nodes.add(child_idx);
        let child_values = (*child_ptr).recursive_values;
        let child_utility = utility_of(&child_values, params);
        if loc == PASS_LOC {
            pass_policy = book_move.raw_policy;
            pass_utility = child_utility;
        }
        move_data.push(MoveCostData {
            loc,
            child_idx,
            raw_policy: book_move.raw_policy,
            log_raw_policy: (book_move.raw_policy + 1e-100).ln(),
            child_values,
            child_utility,
            cost_from_root: 0.0,
            biggest_wl_cost_from_root: 0.0,
            is_wl_pv: false,
        });
    }

    let mut best_win_loss_this_perspective = -1e100;
    let mut best_wl_pv_idx: Option<usize> = None;
    for (i, md) in move_data.iter().enumerate() {
        let win_loss_this_perspective = if is_white {
            md.child_values.win_loss_value
        } else {
            -md.child_values.win_loss_value
        };
        if win_loss_this_perspective > best_win_loss_this_perspective {
            best_win_loss_this_perspective = win_loss_this_perspective;
            best_wl_pv_idx = Some(i);
        }
    }
    let mut expansion_is_wl_pv = false;
    {
        let win_loss_this_perspective = if is_white {
            this_values.win_loss_value
        } else {
            -this_values.win_loss_value
        };
        if win_loss_this_perspective > best_win_loss_this_perspective {
            best_win_loss_this_perspective = win_loss_this_perspective;
            best_wl_pv_idx = None;
            expansion_is_wl_pv = true;
        }
    }
    if let Some(i) = best_wl_pv_idx {
        move_data[i].is_wl_pv = true;
    }
    let best_win_loss = if is_white {
        best_win_loss_this_perspective
    } else {
        -best_win_loss_this_perspective
    };

    // Precompute boosted log policies before mutating move_data.
    let mut boosted_log_policies: Vec<f64> = Vec::with_capacity(move_data.len());
    for md in &move_data {
        boosted_log_policies.push(boost_log_raw_policy(
            &move_data,
            is_white,
            params,
            md.log_raw_policy,
            md.child_utility,
            md.raw_policy,
        ));
    }

    // 7. Update costs for each child move.
    let mut smallest_cost_from_ucb = 1e100;
    for (i, md) in move_data.iter_mut().enumerate() {
        let child = &md.child_values;
        let ucb_win_loss_loss = if is_white {
            recursive_values.win_loss_ucb - child.win_loss_ucb
        } else {
            child.win_loss_lcb - recursive_values.win_loss_lcb
        };
        let ucb_win_loss_loss_pow3 = if is_white {
            pow3(recursive_values.win_loss_ucb) - pow3(child.win_loss_ucb)
        } else {
            pow3(child.win_loss_lcb) - pow3(recursive_values.win_loss_lcb)
        };
        let ucb_win_loss_loss_pow7 = if is_white {
            pow7(recursive_values.win_loss_ucb) - pow7(child.win_loss_ucb)
        } else {
            pow7(child.win_loss_lcb) - pow7(recursive_values.win_loss_lcb)
        };
        let mut ucb_score_loss = if is_white {
            recursive_values.score_ucb - child.score_ucb
        } else {
            child.score_lcb - recursive_values.score_lcb
        };
        if ucb_score_loss > params.score_loss_cap {
            ucb_score_loss = params.score_loss_cap;
        }
        let raw_policy = md.raw_policy;
        let boosted_log_raw_policy = boosted_log_policies[i];
        let pass_favored = pass_policy > 0.15
            && pass_policy > raw_policy * 0.8
            && ((is_white && pass_utility > md.child_utility - 0.02)
                || (!is_white && pass_utility < md.child_utility + 0.02));

        let mut cost_from_wl = ucb_win_loss_loss * params.cost_per_ucb_win_loss_loss
            + ucb_win_loss_loss_pow3 * params.cost_per_ucb_win_loss_loss_pow3
            + ucb_win_loss_loss_pow7 * params.cost_per_ucb_win_loss_loss_pow7;
        if cost_from_wl > biggest_wl_cost {
            cost_from_wl -= params.bonus_for_biggest_wl_cost * (cost_from_wl - biggest_wl_cost);
        }
        let cost_from_ucb = cost_from_wl + ucb_score_loss * params.cost_per_ucb_score_loss;

        md.cost_from_root = min_cost
            + params.cost_per_move
            + cost_from_ucb
            + (-boosted_log_raw_policy * params.cost_per_log_policy)
            + if pass_favored {
                params.cost_when_pass_favored
            } else {
                0.0
            };
        md.biggest_wl_cost_from_root = biggest_wl_cost.max(cost_from_wl);
        if cost_from_ucb < smallest_cost_from_ucb {
            smallest_cost_from_ucb = cost_from_ucb;
        }
    }

    // 8. Compute thisNodeExpansionCost.
    let mut this_node_expansion_cost: f64;
    if !(*node_ptr).can_expand {
        this_node_expansion_cost = 1e100;
    } else if (*node_ptr).can_re_expand
        && recursive_values.visits <= params.max_visits_for_re_expansion
    {
        let m = recursive_values.visits / params.max_visits_for_re_expansion.max(1.0);
        this_node_expansion_cost =
            m * params.cost_per_moves_expanded + m * m * params.cost_per_squared_moves_expanded;
        smallest_cost_from_ucb = 0.0;
    } else {
        let score_error = this_values.get_adjusted_score_error(&rules);
        let win_loss_error = this_values.get_adjusted_win_loss_error(&rules);
        let ucb_win_loss_loss = if is_white {
            recursive_values.win_loss_ucb
                - (this_values.win_loss_value + params.error_factor * win_loss_error)
        } else {
            (this_values.win_loss_value - params.error_factor * win_loss_error)
                - recursive_values.win_loss_lcb
        };
        let ucb_win_loss_loss_pow3 = if is_white {
            pow3(recursive_values.win_loss_ucb)
                - pow3(this_values.win_loss_value + params.error_factor * win_loss_error)
        } else {
            pow3(this_values.win_loss_value - params.error_factor * win_loss_error)
                - pow3(recursive_values.win_loss_lcb)
        };
        let ucb_win_loss_loss_pow7 = if is_white {
            pow7(recursive_values.win_loss_ucb)
                - pow7(this_values.win_loss_value + params.error_factor * win_loss_error)
        } else {
            pow7(this_values.win_loss_value - params.error_factor * win_loss_error)
                - pow7(recursive_values.win_loss_lcb)
        };
        let mut ucb_score_loss = if is_white {
            recursive_values.score_ucb
                - (this_values.score_mean + params.error_factor * score_error)
        } else {
            (this_values.score_mean - params.error_factor * score_error)
                - recursive_values.score_lcb
        };
        if ucb_score_loss > params.score_loss_cap {
            ucb_score_loss = params.score_loss_cap;
        }
        let raw_policy = this_values.max_policy;
        let log_raw_policy = (raw_policy + 1e-100).ln();
        let not_in_book_utility =
            this_values.win_loss_value + this_values.score_mean * params.utility_per_score;
        let boosted_log_raw_policy = boost_log_raw_policy(
            &move_data,
            is_white,
            params,
            log_raw_policy,
            not_in_book_utility,
            raw_policy,
        );
        let pass_favored = pass_policy > 0.15
            && pass_policy > raw_policy * 0.8
            && ((is_white && pass_utility > not_in_book_utility - 0.02)
                || (!is_white && pass_utility < not_in_book_utility + 0.02));

        let mut moves_expanded = move_data.len() as f64;
        let mut moves_expanded_cap = 0.5;
        for other in &move_data {
            if moves_expanded_cap >= moves_expanded {
                break;
            }
            let gain_over_other = if is_white {
                not_in_book_utility - other.child_utility
            } else {
                other.child_utility - not_in_book_utility
            };
            let proportion_to_not_count = if gain_over_other <= 0.0 {
                0.0
            } else {
                2.0 / (1.0 + (-gain_over_other / params.policy_boost_soft_utility_scale).exp())
                    - 1.0
            };
            moves_expanded_cap += 1.5 * (1.0 - proportion_to_not_count);
        }
        if moves_expanded > moves_expanded_cap {
            moves_expanded = moves_expanded_cap;
        }
        if moves_expanded > 1.0 / (raw_policy + 1e-30) {
            moves_expanded = 1.0 / (raw_policy + 1e-30);
        }

        let mut cost_from_wl = ucb_win_loss_loss * params.cost_per_ucb_win_loss_loss
            + ucb_win_loss_loss_pow3 * params.cost_per_ucb_win_loss_loss_pow3
            + ucb_win_loss_loss_pow7 * params.cost_per_ucb_win_loss_loss_pow7;
        if cost_from_wl > biggest_wl_cost {
            cost_from_wl -= params.bonus_for_biggest_wl_cost * (cost_from_wl - biggest_wl_cost);
        }
        let cost_from_ucb = cost_from_wl + ucb_score_loss * params.cost_per_ucb_score_loss;

        this_node_expansion_cost = params.cost_per_move
            + cost_from_ucb
            + (-boosted_log_raw_policy * params.cost_per_log_policy)
            + moves_expanded * params.cost_per_moves_expanded
            + moves_expanded * moves_expanded * params.cost_per_squared_moves_expanded
            + if pass_favored {
                params.cost_when_pass_favored
            } else {
                0.0
            };
        if cost_from_ucb < smallest_cost_from_ucb {
            smallest_cost_from_ucb = cost_from_ucb;
        }
    }

    // 9. Replenish move costs based on smallestCostFromUCB.
    if smallest_cost_from_ucb > 1e-100 {
        for md in &mut move_data {
            md.cost_from_root -= 0.8 * smallest_cost_from_ucb;
        }
        this_node_expansion_cost -= 0.8 * smallest_cost_from_ucb;
    }

    // 10. Reduce costs of moves >1.5% better winrate than worse moves.
    for i in 0..move_data.len() {
        let win_loss = if is_white {
            move_data[i].child_values.win_loss_value
        } else {
            -move_data[i].child_values.win_loss_value
        };
        let mut best_other_cost = move_data[i].cost_from_root;
        for other in &move_data {
            if other.cost_from_root < best_other_cost {
                let win_loss_other = if is_white {
                    other.child_values.win_loss_value
                } else {
                    -other.child_values.win_loss_value
                };
                if win_loss > win_loss_other + 0.03 {
                    best_other_cost = other.cost_from_root;
                }
            }
        }
        if best_other_cost < move_data[i].cost_from_root {
            move_data[i].cost_from_root += 0.70 * (best_other_cost - move_data[i].cost_from_root);
        }
    }
    {
        let win_loss = if is_white {
            this_values.win_loss_value
        } else {
            -this_values.win_loss_value
        };
        let mut best_other_cost = this_node_expansion_cost + min_cost;
        for md in &move_data {
            if md.cost_from_root < best_other_cost {
                let win_loss_other = if is_white {
                    md.child_values.win_loss_value
                } else {
                    -md.child_values.win_loss_value
                };
                if win_loss > win_loss_other + 0.03 {
                    best_other_cost = md.cost_from_root;
                }
            }
        }
        if best_other_cost - min_cost < this_node_expansion_cost {
            this_node_expansion_cost +=
                0.70 * (best_other_cost - min_cost - this_node_expansion_cost);
        }
    }

    // 11. Apply bonuses based on errors and WL-PV bonus to moves.
    for md in &mut move_data {
        let child = &md.child_values;
        let win_loss_error =
            (child.win_loss_ucb - child.win_loss_lcb).abs() / params.error_factor / 2.0;
        let score_error = (child.score_ucb - child.score_lcb).abs() / params.error_factor / 2.0;
        let sharp_score_discrepancy = (child.sharp_score_mean - child.score_mean).abs();
        let mut bonus = params.bonus_per_win_loss_error * win_loss_error
            + params.bonus_per_score_error * score_error
            + params.bonus_per_sharp_score_discrepancy * sharp_score_discrepancy;
        let bonus_cap1 = (md.cost_from_root - min_cost) * 0.75;
        if bonus > bonus_cap1 {
            bonus = bonus_cap1;
        }
        md.cost_from_root -= bonus;

        if md.is_wl_pv {
            let wl_pv_bonus_scale =
                (md.cost_from_root - min_cost) * (1.0 - params.bonus_for_wl_pv_final_prop);
            if wl_pv_bonus_scale > 0.0 {
                let factor1 = (1.0 - square(child.win_loss_value)).max(0.0);
                let factor2 = 4.0 * (0.25 - square(0.5 - child.win_loss_value.abs())).max(0.0);
                let wl_pv_bonus = wl_pv_bonus_scale
                    * (factor1 * params.bonus_for_wl_pv1 + factor2 * params.bonus_for_wl_pv2)
                        .tanh();
                md.cost_from_root -= wl_pv_bonus;
            }
        }
    }

    // 12. Apply bonuses to thisNodeExpansionCost.
    {
        let win_loss_error = this_values.get_adjusted_win_loss_error(&rules);
        let score_error = this_values.get_adjusted_score_error(&rules);
        let sharp_score_discrepancy =
            (this_values.sharp_score_mean_raw - this_values.score_mean).abs();
        let moves_expanded = move_data.len() as f64;
        let excess_unexpanded_policy =
            if moves_expanded > 0.0 && this_values.max_policy > 1.0 / moves_expanded {
                this_values.max_policy - 1.0 / moves_expanded
            } else {
                0.0
            };
        let mut bonus = params.bonus_per_win_loss_error * win_loss_error
            + params.bonus_per_score_error * score_error
            + params.bonus_per_sharp_score_discrepancy * sharp_score_discrepancy.min(1.0)
            + params.bonus_per_excess_unexpanded_policy * excess_unexpanded_policy;
        let bonus_cap1 = this_node_expansion_cost * 0.75;
        if bonus > bonus_cap1 {
            bonus = bonus_cap1;
        }
        bonus +=
            params.bonus_per_sharp_score_discrepancy * (sharp_score_discrepancy - 1.0).max(0.0);
        this_node_expansion_cost -= bonus;

        const BEST_WINLOSS_OFFSET: f64 = 0.02;
        let win_loss = if is_white {
            this_values.win_loss_value
        } else {
            -this_values.win_loss_value
        };
        let mut any_other = false;
        let mut best_other_win_loss = 0.0;
        let mut best_other_visits = 0.0;
        let mut total_other_visits = 0.0;
        for md in &move_data {
            let win_loss_other = if is_white {
                md.child_values.win_loss_value
            } else {
                -md.child_values.win_loss_value
            };
            if !any_other || win_loss_other > best_other_win_loss {
                best_other_win_loss = win_loss_other;
                best_other_visits = md.child_values.visits;
                any_other = true;
            }
            total_other_visits += md.child_values.visits;
        }
        if any_other && win_loss > best_other_win_loss {
            let visits_factor = 0.5
                * ((best_other_visits / params.visits_scale.max(1.0))
                    .sqrt()
                    .min(1.0)
                    + (total_other_visits / params.visits_scale.max(1.0))
                        .sqrt()
                        .min(1.0));
            this_node_expansion_cost -= params.bonus_per_unexpanded_best_win_loss
                * (win_loss - best_other_win_loss + BEST_WINLOSS_OFFSET)
                * visits_factor;
        }

        if move_data.len() >= 2 {
            if let Some(i) = best_wl_pv_idx {
                if move_data[i].child_values.visits <= params.max_visits_for_re_expansion {
                    let best_child_idx = move_data[i].child_idx;
                    let best_child_visits = move_data[i].child_values.visits;
                    let mut any_other2 = false;
                    let mut best_other_wl_this_perspective = 0.0;
                    let mut best_other_visits2 = 0.0;
                    let mut total_other_visits2 = 0.0;
                    for md in &move_data {
                        if md.child_idx != best_child_idx {
                            let win_loss_other = if is_white {
                                md.child_values.win_loss_value
                            } else {
                                -md.child_values.win_loss_value
                            };
                            if !any_other2 || win_loss_other > best_other_wl_this_perspective {
                                best_other_wl_this_perspective = win_loss_other;
                                best_other_visits2 = md.child_values.visits;
                                any_other2 = true;
                            }
                            total_other_visits2 += md.child_values.visits;
                        }
                    }
                    if any_other2
                        && best_win_loss_this_perspective > best_other_wl_this_perspective
                        && best_child_visits < best_other_visits2
                    {
                        let visits_factor = 0.5
                            * ((best_other_visits2 / params.visits_scale.max(1.0))
                                .sqrt()
                                .min(1.0)
                                + (total_other_visits2 / params.visits_scale.max(1.0))
                                    .sqrt()
                                    .min(1.0))
                            - (best_child_visits / params.visits_scale.max(1.0))
                                .sqrt()
                                .min(1.0);
                        for md in &mut move_data {
                            if md.child_idx == best_child_idx {
                                md.cost_from_root -= 0.75
                                    * params.bonus_per_unexpanded_best_win_loss
                                    * (best_win_loss_this_perspective
                                        - best_other_wl_this_perspective
                                        + BEST_WINLOSS_OFFSET)
                                    * visits_factor;
                                break;
                            }
                        }
                    }
                }
            }

            // behindInVisitsBonus
            let mut child_bonuses: Vec<f64> = Vec::with_capacity(move_data.len());
            for md in &move_data {
                child_bonuses.push(behind_in_visits_bonus(
                    &move_data,
                    is_white,
                    params,
                    best_win_loss,
                    md.child_values.win_loss_value,
                    md.child_values.adjusted_visits,
                ));
            }
            for (i, md) in move_data.iter_mut().enumerate() {
                md.cost_from_root -= child_bonuses[i] * params.bonus_behind_in_visits_scale;
            }
            let this_bonus = behind_in_visits_bonus(
                &move_data,
                is_white,
                params,
                best_win_loss,
                this_values.win_loss_value,
                this_values.visits,
            );
            this_node_expansion_cost -= this_bonus * params.bonus_behind_in_visits_scale;
        }
    }

    // 14. WL-PV bonus for expansion cost.
    if expansion_is_wl_pv
        || ((*node_ptr).can_re_expand
            && recursive_values.visits <= params.max_visits_for_re_expansion)
    {
        let wl_pv_bonus_scale = this_node_expansion_cost
            + (min_cost - min_cost_wl_pv).max(0.0) * params.bonus_for_wl_pv_final_prop;
        if wl_pv_bonus_scale > 0.0 {
            let factor1 = (1.0 - square(this_values.win_loss_value)).max(0.0);
            let factor2 = 4.0 * (0.25 - square(0.5 - this_values.win_loss_value.abs())).max(0.0);
            let wl_pv_bonus = wl_pv_bonus_scale
                * (factor1 * params.bonus_for_wl_pv1 + factor2 * params.bonus_for_wl_pv2).tanh();
            this_node_expansion_cost -= wl_pv_bonus;
        }
    }

    // 15. Apply earlyBookCostReductionFactor depth scaling.
    let depth_factor = 1.0
        - params.early_book_cost_reduction_factor
            * params.early_book_cost_reduction_lambda.powi(min_depth);
    for md in &mut move_data {
        md.cost_from_root = min_cost + (md.cost_from_root - min_cost) * depth_factor;
    }
    this_node_expansion_cost *= depth_factor;

    // 16. Apply expandBonusByHash and branchRequiredByHash.
    if let Some(&bonus) = (*book).expand_bonus_by_hash.get(&hash) {
        this_node_expansion_cost -= bonus;
    }
    if let Some(&required_branch) = (*book).branch_required_by_hash.get(&hash) {
        if move_data.len() < required_branch as usize {
            this_node_expansion_cost -= 700.0;
        } else {
            let mut child_enough_visits_count = 0;
            for md in &move_data {
                if md.child_values.visits > params.max_visits_for_re_expansion {
                    child_enough_visits_count += 1;
                }
            }
            if child_enough_visits_count < required_branch {
                let mut sort_indices: Vec<usize> = (0..move_data.len()).collect();
                sort_indices.sort_by(|&a, &b| {
                    let va = Book::get_sorting_value_params(
                        &params,
                        pla_factor,
                        move_data[a].child_values.win_loss_value,
                        move_data[a].child_values.score_mean,
                        move_data[a].child_values.sharp_score_mean,
                        move_data[a].child_values.score_lcb,
                        move_data[a].child_values.score_ucb,
                        move_data[a].raw_policy,
                    );
                    let vb = Book::get_sorting_value_params(
                        &params,
                        pla_factor,
                        move_data[b].child_values.win_loss_value,
                        move_data[b].child_values.score_mean,
                        move_data[b].child_values.sharp_score_mean,
                        move_data[b].child_values.score_lcb,
                        move_data[b].child_values.score_ucb,
                        move_data[b].raw_policy,
                    );
                    vb.partial_cmp(&va).unwrap_or(Ordering::Equal)
                });
                let mut num_bonused = 0;
                for &i in &sort_indices {
                    if num_bonused + child_enough_visits_count >= required_branch {
                        break;
                    }
                    if move_data[i].child_values.visits <= params.max_visits_for_re_expansion {
                        num_bonused += 1;
                        move_data[i].cost_from_root -= 200.0;
                    }
                }
            }
        }
    }

    // Write the results back to the node.
    (*node_ptr).min_depth_from_root = min_depth;
    (*node_ptr).min_cost_from_root = min_cost;
    (*node_ptr).min_cost_from_root_wl_pv = min_cost_wl_pv;
    (*node_ptr).biggest_wl_cost_from_root = biggest_wl_cost;
    (*node_ptr).best_parent_idx = best_parent_idx as i64;
    (*node_ptr).expansion_is_wl_pv = expansion_is_wl_pv;
    (*node_ptr).this_node_expansion_cost = this_node_expansion_cost;
    for md in move_data {
        if let Some(book_move) = (*node_ptr).moves.get_mut(&md.loc) {
            book_move.cost_from_root = md.cost_from_root;
            book_move.biggest_wl_cost_from_root = md.biggest_wl_cost_from_root;
            book_move.is_wl_pv = md.is_wl_pv;
        }
    }
}

/// Lock the node and all of its children, then recompute recursive values.
unsafe fn lock_and_recompute_node_values(book: *mut Book, idx: usize, mutex_pool: &MutexPool) {
    let node_ptr = (*book).nodes.as_ptr().add(idx);
    let mut indices = Vec::new();
    indices.push(mutex_index_for_hash(
        (*node_ptr).hash,
        mutex_pool.num_mutexes(),
    ));
    for book_move in (*node_ptr).moves.values() {
        indices.push(mutex_index_for_hash(
            book_move.hash,
            mutex_pool.num_mutexes(),
        ));
    }
    indices.sort_unstable();
    indices.dedup();
    let _guards: Vec<_> = indices.iter().map(|&i| mutex_pool.get(i).lock()).collect();
    recompute_node_values_raw(book, idx);
}

/// Lock the node and all of its parents, then recompute path costs.
unsafe fn lock_and_recompute_node_cost(book: *mut Book, idx: usize, mutex_pool: &MutexPool) {
    let node_ptr = (*book).nodes.as_ptr().add(idx);
    let mut indices = Vec::new();
    indices.push(mutex_index_for_hash(
        (*node_ptr).hash,
        mutex_pool.num_mutexes(),
    ));
    for (parent_hash, _) in &(*node_ptr).parents {
        indices.push(mutex_index_for_hash(*parent_hash, mutex_pool.num_mutexes()));
    }
    indices.sort_unstable();
    indices.dedup();
    let _guards: Vec<_> = indices.iter().map(|&i| mutex_pool.get(i).lock()).collect();
    recompute_node_cost_raw(book, idx);
}

#[allow(dead_code)]
fn square(x: f64) -> f64 {
    x * x
}

#[allow(dead_code)]
fn pow3(x: f64) -> f64 {
    x * x * x
}

#[allow(dead_code)]
fn pow7(x: f64) -> f64 {
    let cube = x * x * x;
    cube * cube * x
}

fn clamp_score_for_sorting(score: f64, win_loss: f64) -> f64 {
    let win_loss = win_loss.clamp(-1.0, 1.0);
    let score_lower_bound = (win_loss - 1.0) / (win_loss + 1.0 + 0.0001) * 2.0;
    let score_upper_bound = -(-win_loss - 1.0) / (-win_loss + 1.0 + 0.0001) * 2.0;
    score.max(score_lower_bound).min(score_upper_bound)
}

/// Tunable parameters that control book construction and cost heuristics.
#[derive(Debug, Clone, Copy)]
pub struct BookParams {
    pub error_factor: f64,
    pub cost_per_move: f64,
    pub cost_per_ucb_win_loss_loss: f64,
    pub cost_per_ucb_win_loss_loss_pow3: f64,
    pub cost_per_ucb_win_loss_loss_pow7: f64,
    pub cost_per_ucb_score_loss: f64,
    pub cost_per_log_policy: f64,
    pub cost_per_moves_expanded: f64,
    pub cost_per_squared_moves_expanded: f64,
    pub cost_when_pass_favored: f64,
    pub bonus_per_win_loss_error: f64,
    pub bonus_per_score_error: f64,
    pub bonus_per_sharp_score_discrepancy: f64,
    pub bonus_per_excess_unexpanded_policy: f64,
    pub bonus_per_unexpanded_best_win_loss: f64,
    pub bonus_for_wl_pv1: f64,
    pub bonus_for_wl_pv2: f64,
    pub bonus_for_wl_pv_final_prop: f64,
    pub bonus_for_biggest_wl_cost: f64,
    pub bonus_behind_in_visits_scale: f64,
    pub score_loss_cap: f64,
    pub early_book_cost_reduction_factor: f64,
    pub early_book_cost_reduction_lambda: f64,
    pub utility_per_score: f64,
    pub policy_boost_soft_utility_scale: f64,
    pub utility_per_policy_for_sorting: f64,
    pub adjusted_visits_wl_scale: f64,
    pub max_visits_for_re_expansion: f64,
    pub visits_scale: f64,
    pub visits_scale_leaves: f64,
    pub sharp_score_outlier_cap: f64,
}

impl Default for BookParams {
    fn default() -> Self {
        Self {
            error_factor: 1.0,
            cost_per_move: 1.0,
            cost_per_ucb_win_loss_loss: 0.0,
            cost_per_ucb_win_loss_loss_pow3: 0.0,
            cost_per_ucb_win_loss_loss_pow7: 0.0,
            cost_per_ucb_score_loss: 0.0,
            cost_per_log_policy: 0.0,
            cost_per_moves_expanded: 1.0,
            cost_per_squared_moves_expanded: 0.0,
            cost_when_pass_favored: 0.0,
            bonus_per_win_loss_error: 0.0,
            bonus_per_score_error: 0.0,
            bonus_per_sharp_score_discrepancy: 0.0,
            bonus_per_excess_unexpanded_policy: 0.0,
            bonus_per_unexpanded_best_win_loss: 0.0,
            bonus_for_wl_pv1: 0.0,
            bonus_for_wl_pv2: 0.0,
            bonus_for_wl_pv_final_prop: 0.5,
            bonus_for_biggest_wl_cost: 0.0,
            bonus_behind_in_visits_scale: 0.0,
            score_loss_cap: 10000.0,
            early_book_cost_reduction_factor: 0.0,
            early_book_cost_reduction_lambda: 0.0,
            utility_per_score: 0.0,
            policy_boost_soft_utility_scale: 1.0,
            utility_per_policy_for_sorting: 0.0,
            adjusted_visits_wl_scale: 0.05,
            max_visits_for_re_expansion: 1000.0,
            visits_scale: 1000.0,
            visits_scale_leaves: 100.0,
            sharp_score_outlier_cap: 10000.0,
        }
    }
}

impl BookParams {
    /// Load book parameters from a `ConfigParser`, matching C++ `BookParams::loadFromCfg`.
    pub fn load_from_cfg(
        cfg: &ConfigParser,
        max_visits: i64,
        max_visits_for_leaves: i64,
    ) -> Result<Self, ConfigError> {
        Ok(Self {
            error_factor: cfg.get_double("errorFactor", 0.01, 100.0)?,
            cost_per_move: cfg.get_double("costPerMove", 0.0, 1000000.0)?,
            cost_per_ucb_win_loss_loss: cfg.get_double("costPerUCBWinLossLoss", 0.0, 1000000.0)?,
            cost_per_ucb_win_loss_loss_pow3: cfg.get_double(
                "costPerUCBWinLossLossPow3",
                0.0,
                1000000.0,
            )?,
            cost_per_ucb_win_loss_loss_pow7: cfg.get_double(
                "costPerUCBWinLossLossPow7",
                0.0,
                1000000.0,
            )?,
            cost_per_ucb_score_loss: cfg.get_double("costPerUCBScoreLoss", 0.0, 1000000.0)?,
            cost_per_log_policy: cfg.get_double("costPerLogPolicy", 0.0, 1000000.0)?,
            cost_per_moves_expanded: cfg.get_double("costPerMovesExpanded", 0.0, 1000000.0)?,
            cost_per_squared_moves_expanded: cfg.get_double(
                "costPerSquaredMovesExpanded",
                0.0,
                1000000.0,
            )?,
            cost_when_pass_favored: cfg.get_double("costWhenPassFavored", 0.0, 1000000.0)?,
            bonus_per_win_loss_error: cfg.get_double("bonusPerWinLossError", 0.0, 1000000.0)?,
            bonus_per_score_error: cfg.get_double("bonusPerScoreError", 0.0, 1000000.0)?,
            bonus_per_sharp_score_discrepancy: cfg.get_double(
                "bonusPerSharpScoreDiscrepancy",
                0.0,
                1000000.0,
            )?,
            bonus_per_excess_unexpanded_policy: cfg.get_double(
                "bonusPerExcessUnexpandedPolicy",
                0.0,
                1000000.0,
            )?,
            bonus_per_unexpanded_best_win_loss: cfg.get_double(
                "bonusPerUnexpandedBestWinLoss",
                0.0,
                1000000.0,
            )?,
            bonus_for_wl_pv1: if cfg.contains("bonusForWLPV1") {
                cfg.get_double("bonusForWLPV1", 0.0, 1000000.0)?
            } else {
                0.0
            },
            bonus_for_wl_pv2: if cfg.contains("bonusForWLPV2") {
                cfg.get_double("bonusForWLPV2", 0.0, 1000000.0)?
            } else {
                0.0
            },
            bonus_for_wl_pv_final_prop: if cfg.contains("bonusForWLPVFinalProp") {
                cfg.get_double("bonusForWLPVFinalProp", 0.0, 1.0)?
            } else {
                0.5
            },
            bonus_for_biggest_wl_cost: if cfg.contains("bonusForBiggestWLCost") {
                cfg.get_double("bonusForBiggestWLCost", 0.0, 1000000.0)?
            } else {
                0.0
            },
            bonus_behind_in_visits_scale: if cfg.contains("bonusBehindInVisitsScale") {
                cfg.get_double("bonusBehindInVisitsScale", 0.0, 1000000.0)?
            } else {
                0.0
            },
            score_loss_cap: cfg.get_double("scoreLossCap", 0.0, 1000000.0)?,
            early_book_cost_reduction_factor: if cfg.contains("earlyBookCostReductionFactor") {
                cfg.get_double("earlyBookCostReductionFactor", 0.0, 1.0)?
            } else {
                0.0
            },
            early_book_cost_reduction_lambda: if cfg.contains("earlyBookCostReductionLambda") {
                cfg.get_double("earlyBookCostReductionLambda", 0.0, 1.0)?
            } else {
                0.5
            },
            utility_per_score: cfg.get_double("utilityPerScore", 0.0, 1000000.0)?,
            policy_boost_soft_utility_scale: cfg.get_double(
                "policyBoostSoftUtilityScale",
                0.0,
                1000000.0,
            )?,
            utility_per_policy_for_sorting: cfg.get_double(
                "utilityPerPolicyForSorting",
                0.0,
                1000000.0,
            )?,
            adjusted_visits_wl_scale: if cfg.contains("adjustedVisitsWLScale") {
                cfg.get_double("adjustedVisitsWLScale", 0.0, 1000000.0)?
            } else {
                0.05
            },
            max_visits_for_re_expansion: if cfg.contains("maxVisitsForReExpansion") {
                cfg.get_double("maxVisitsForReExpansion", 0.0, 1e50)?
            } else {
                0.0
            },
            visits_scale: if cfg.contains("visitsScale") {
                cfg.get_double("visitsScale", f64::MIN, f64::MAX)?
            } else {
                ((max_visits + 1) / 2) as f64
            },
            visits_scale_leaves: if cfg.contains("visitsScaleLeaves") {
                cfg.get_double("visitsScaleLeaves", f64::MIN, f64::MAX)?
            } else {
                max_visits_for_leaves as f64
            },
            sharp_score_outlier_cap: cfg.get_double("sharpScoreOutlierCap", 0.0, 1000000.0)?,
        })
    }
}

/// A move recorded in the book, together with the symmetry needed to align the
/// child node with the parent's orientation.
#[derive(Debug, Clone, Copy, Default)]
pub struct BookMove {
    pub move_loc: Loc,
    pub symmetry_to_align: i32,
    pub hash: BookHash,
    pub raw_policy: f64,
    pub cost_from_root: f64,
    pub is_wl_pv: bool,
    pub biggest_wl_cost_from_root: f64,
}

impl BookMove {
    /// Create a new book move.
    pub fn new(move_loc: Loc, symmetry_to_align: i32, hash: BookHash, raw_policy: f64) -> Self {
        Self {
            move_loc,
            symmetry_to_align,
            hash,
            raw_policy,
            ..Default::default()
        }
    }

    /// Apply a symmetry to this move, returning it in the new orientation.
    pub fn get_sym_book_move(&self, symmetry: i32, x_size: i32, y_size: i32) -> Self {
        Self {
            move_loc: symmetry::get_sym_loc_with_size(self.move_loc, x_size, y_size, symmetry),
            symmetry_to_align: symmetry::compose(
                symmetry::invert(symmetry),
                self.symmetry_to_align,
            ),
            hash: self.hash,
            raw_policy: self.raw_policy,
            cost_from_root: self.cost_from_root,
            is_wl_pv: self.is_wl_pv,
            biggest_wl_cost_from_root: self.biggest_wl_cost_from_root,
        }
    }
}

/// Values recorded for a book node from a search of that position.
#[derive(Debug, Clone, Copy, Default)]
pub struct BookValues {
    pub win_loss_value: f64,
    pub score_mean: f64,
    pub sharp_score_mean_raw: f64,
    pub win_loss_error: f64,
    pub score_error: f64,
    pub score_stdev: f64,
    pub max_policy: f64,
    pub weight: f64,
    pub visits: f64,
    pub sharp_score_mean_clamped: f64,
    pub posterior_policy: f64,
}

impl BookValues {
    /// Adjusted win-loss error, treating negative raw errors as zero.
    pub fn get_adjusted_win_loss_error(&self, _rules: &Rules) -> f64 {
        if self.win_loss_error < 0.0 {
            0.0
        } else {
            self.win_loss_error
        }
    }

    /// Adjusted score error, accounting for territory-scoring integer variance.
    pub fn get_adjusted_score_error(&self, rules: &Rules) -> f64 {
        if self.score_error < 0.0 {
            return 0.0;
        }
        if rules.game_result_will_be_integer() {
            let score_variance = self.score_stdev * self.score_stdev;
            let mut adjusted_score_variance = score_variance - 0.25;
            let floor = score_variance * 0.05;
            if adjusted_score_variance < floor {
                adjusted_score_variance = floor;
            }
            adjusted_score_variance.sqrt().min(self.score_error)
        } else {
            self.score_stdev.min(self.score_error)
        }
    }
}

/// Values propagated through the book via minimax.
#[derive(Debug, Clone, Copy, Default)]
pub struct RecursiveBookValues {
    pub win_loss_value: f64,
    pub score_mean: f64,
    pub sharp_score_mean: f64,
    pub win_loss_lcb: f64,
    pub score_lcb: f64,
    pub score_final_lcb: f64,
    pub win_loss_ucb: f64,
    pub score_ucb: f64,
    pub score_final_ucb: f64,
    pub weight: f64,
    pub visits: f64,
    pub adjusted_visits: f64,
}

/// Internal storage for a book node.
#[derive(Debug)]
#[allow(dead_code)]
pub struct BookNode {
    hash: BookHash,
    pla: Player,
    symmetries: Vec<i32>,

    this_values_not_in_book: BookValues,
    can_expand: bool,
    can_re_expand: bool,
    moves: BTreeMap<Loc, BookMove>,
    parents: Vec<(BookHash, Loc)>,
    best_parent_idx: i64,
    recursive_values: RecursiveBookValues,
    min_depth_from_root: i32,
    min_cost_from_root: f64,
    this_node_expansion_cost: f64,
    min_cost_from_root_wl_pv: f64,
    expansion_is_wl_pv: bool,
    biggest_wl_cost_from_root: f64,
    visited_flag: AtomicI32,
}

impl BookNode {
    /// Create a new book node.
    pub fn new(hash: BookHash, pla: Player, symmetries: Vec<i32>) -> Self {
        Self {
            hash,
            pla,
            symmetries,
            this_values_not_in_book: BookValues::default(),
            can_expand: true,
            can_re_expand: true,
            moves: BTreeMap::new(),
            parents: Vec::new(),
            best_parent_idx: 0,
            recursive_values: RecursiveBookValues::default(),
            min_depth_from_root: 0,
            min_cost_from_root: 0.0,
            this_node_expansion_cost: 0.0,
            min_cost_from_root_wl_pv: 0.0,
            expansion_is_wl_pv: false,
            biggest_wl_cost_from_root: 0.0,
            visited_flag: AtomicI32::new(0),
        }
    }

    /// The player to move at this node.
    pub fn pla(&self) -> Player {
        self.pla
    }

    /// The hash identifying this node.
    pub fn hash(&self) -> BookHash {
        self.hash
    }

    /// Symmetries under which this position is invariant.
    pub fn symmetries(&self) -> &[i32] {
        &self.symmetries
    }

    /// Values based on a search of this node alone, excluding children.
    pub fn this_values_not_in_book(&self) -> &BookValues {
        &self.this_values_not_in_book
    }

    /// Mutable access to [`Self::this_values_not_in_book`].
    pub fn this_values_not_in_book_mut(&mut self) -> &mut BookValues {
        &mut self.this_values_not_in_book
    }

    /// Whether the book construction algorithm may expand more children.
    pub fn can_expand(&self) -> bool {
        self.can_expand
    }

    /// Mutable access to [`Self::can_expand`].
    pub fn can_expand_mut(&mut self) -> &mut bool {
        &mut self.can_expand
    }

    /// Whether this node may be re-expanded during this run.
    pub fn can_re_expand(&self) -> bool {
        self.can_re_expand
    }

    /// Mutable access to [`Self::can_re_expand`].
    pub fn can_re_expand_mut(&mut self) -> &mut bool {
        &mut self.can_re_expand
    }

    /// Values propagated through the book via minimax.
    pub fn recursive_values(&self) -> &RecursiveBookValues {
        &self.recursive_values
    }

    /// Minimum number of moves to reach this node from the root.
    pub fn min_depth_from_root(&self) -> i32 {
        self.min_depth_from_root
    }

    /// Minimum cost to reach this node from the root.
    pub fn min_cost_from_root(&self) -> f64 {
        self.min_cost_from_root
    }

    /// The raw move map for this node (in the node's own orientation).
    pub fn moves(&self) -> &BTreeMap<Loc, BookMove> {
        &self.moves
    }

    /// Mutable access to the raw move map.
    pub fn moves_mut(&mut self) -> &mut BTreeMap<Loc, BookMove> {
        &mut self.moves
    }
}

/// An opening book, storing positions and minimaxed search values.
#[derive(Debug)]
#[allow(dead_code)]
pub struct Book {
    pub book_version: i32,
    pub initial_board: Board,
    pub initial_rules: Rules,
    pub initial_pla: Player,
    pub rep_bound: i32,
    params: BookParams,
    initial_symmetry: i32,
    nodes: Vec<BookNode>,
    node_idx_by_hash: BTreeMap<BookHash, usize>,
    bonus_by_hash: BTreeMap<BookHash, f64>,
    expand_bonus_by_hash: BTreeMap<BookHash, f64>,
    visits_required_by_hash: BTreeMap<BookHash, f64>,
    branch_required_by_hash: BTreeMap<BookHash, i32>,
    next_visited_done_value: i32,
}

impl Book {
    /// Latest supported book version.
    pub const LATEST_BOOK_VERSION: i32 = 2;

    /// Create a new book with the given initial position and parameters.
    pub fn new(
        book_version: i32,
        board: Board,
        rules: Rules,
        initial_pla: Player,
        rep_bound: i32,
        params: BookParams,
    ) -> Self {
        let initial_hist = BoardHistory::new(board.clone(), initial_pla, rules, 0);
        let hash_and_sym = BookHash::get_hash_and_symmetry(&initial_hist, rep_bound, book_version);

        let root = BookNode::new(hash_and_sym.hash, initial_pla, hash_and_sym.symmetries);
        let mut book = Self {
            book_version,
            initial_board: board,
            initial_rules: rules,
            initial_pla,
            rep_bound,
            params,
            initial_symmetry: hash_and_sym.symmetry_to_align,
            nodes: Vec::new(),
            node_idx_by_hash: BTreeMap::new(),
            bonus_by_hash: BTreeMap::new(),
            expand_bonus_by_hash: BTreeMap::new(),
            visits_required_by_hash: BTreeMap::new(),
            branch_required_by_hash: BTreeMap::new(),
            next_visited_done_value: 1,
        };

        let root_hash = root.hash();
        book.nodes.push(root);
        book.node_idx_by_hash.insert(root_hash, 0);
        book
    }

    /// Number of nodes in the book.
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Book parameters.
    pub fn params(&self) -> &BookParams {
        &self.params
    }

    /// Mutable access to book parameters.
    pub fn params_mut(&mut self) -> &mut BookParams {
        &mut self.params
    }

    /// Replace the book parameters.
    pub fn set_params(&mut self, params: BookParams) {
        self.params = params;
    }

    /// Initial history in the book's canonical orientation.
    pub fn get_initial_hist(&self) -> BoardHistory {
        self.get_initial_hist_with_symmetry(0)
    }

    /// Initial history with a symmetry applied to the initial board.
    pub fn get_initial_hist_with_symmetry(&self, symmetry: i32) -> BoardHistory {
        BoardHistory::new(
            symmetry::get_sym_board(&self.initial_board, symmetry),
            self.initial_pla,
            self.initial_rules,
            0,
        )
    }

    /// The root node, in the orientation of the initial board.
    pub fn get_root(&self) -> SymBookNode {
        // `initial_symmetry` maps initial space -> node space; the SymBookNode
        // needs the inverse (node space -> initial space).
        SymBookNode::new(
            0,
            self.nodes[0].hash(),
            self.nodes[0].pla(),
            symmetry::invert(self.initial_symmetry),
        )
    }

    /// Walk down the book following `hist` and return the final node, or a null
    /// node if the history leaves the book.
    pub fn get(&self, hist: &BoardHistory) -> SymBookNode {
        let mut node = self.get_root();
        for m in &hist.move_history {
            node = node.follow(self, m.loc);
            if node.is_null() {
                return node;
            }
        }
        node
    }

    /// Get a node by its book hash, or a null node if not present.
    pub fn get_by_hash(&self, hash: BookHash) -> SymBookNode {
        if let Some(&idx) = self.node_idx_by_hash.get(&hash) {
            SymBookNode::new(idx, self.nodes[idx].hash(), self.nodes[idx].pla(), 0)
        } else {
            SymBookNode::null()
        }
    }

    /// Look up a raw node by hash.
    fn node_by_hash(&self, hash: BookHash) -> Option<&BookNode> {
        self.node_idx_by_hash
            .get(&hash)
            .map(|&idx| &self.nodes[idx])
    }

    /// Mutable access to a raw node by index.
    fn node_mut(&mut self, idx: usize) -> &mut BookNode {
        &mut self.nodes[idx]
    }

    /// Add a new node to the book. Returns `false` if a node with the same hash
    /// already exists.
    fn add_node(&mut self, hash: BookHash, node: BookNode) -> bool {
        if self.node_idx_by_hash.contains_key(&hash) {
            return false;
        }
        let idx = self.nodes.len();
        self.nodes.push(node);
        self.node_idx_by_hash.insert(hash, idx);
        true
    }

    /// Sorting value using an explicit [`BookParams`] reference.
    #[allow(clippy::too_many_arguments)]
    fn get_sorting_value_params(
        params: &BookParams,
        pla_factor: f64,
        win_loss_value: f64,
        score_mean: f64,
        sharp_score_mean_clamped: f64,
        score_lcb: f64,
        score_ucb: f64,
        raw_policy: f64,
    ) -> f64 {
        let score = 0.5 * (sharp_score_mean_clamped + score_mean);
        let clamped_score = clamp_score_for_sorting(score, win_loss_value);
        let clamped_lcb_ucb = clamp_score_for_sorting(
            0.5 * (pla_factor + 1.0) * score_lcb + 0.5 * (1.0 - pla_factor) * score_ucb,
            win_loss_value,
        );
        pla_factor * (win_loss_value + clamped_score * params.utility_per_score * 0.75)
            + pla_factor * clamped_lcb_ucb * 0.25 * params.utility_per_score
            + params.utility_per_policy_for_sorting
                * (0.75 * raw_policy + 0.5 * (raw_policy + 0.0001).log10() / 4.0)
                * (1.0 + win_loss_value * win_loss_value)
    }

    /// Recompute the minimaxed recursive values of a single node from its
    /// not-in-book values and children.
    fn recompute_node_values(&mut self, node_idx: usize) {
        unsafe { recompute_node_values_raw(self, node_idx) }
    }

    /// Recompute the path costs for a single node.
    ///
    /// This mirrors the full cost heuristic in `KataGo/cpp/book/book.cpp`.
    fn recompute_node_cost(&mut self, idx: usize) {
        unsafe { recompute_node_cost_raw(self, idx) }
    }

    /// Mark all nodes that need recomputation by walking up from changed nodes.
    fn collect_dirty_nodes(&self, changed: &[SymBookNode]) -> BTreeSet<BookHash> {
        let mut dirty = BTreeSet::new();
        let mut stack: Vec<BookHash> = changed.iter().map(|n| n.hash()).collect();
        while let Some(hash) = stack.pop() {
            if !dirty.insert(hash) {
                continue;
            }
            if let Some(node) = self.node_by_hash(hash) {
                for (parent_hash, _) in &node.parents {
                    stack.push(*parent_hash);
                }
            }
        }
        dirty
    }

    /// Recompute recursive values for all dirty nodes in post-order, then
    /// recompute costs for the entire book in pre-order.
    pub fn recompute(&mut self, new_and_changed_nodes: &[SymBookNode]) {
        let dirty = self.collect_dirty_nodes(new_and_changed_nodes);
        if dirty.is_empty() {
            return;
        }
        let mut visited = BTreeSet::new();
        self.recompute_post_order(0, &dirty, &mut visited, false);

        let mut visited = BTreeSet::new();
        self.recompute_pre_order(0, &mut visited);
    }

    /// Recompute recursive values and costs for the entire book.
    pub fn recompute_everything(&mut self) {
        let mut visited = BTreeSet::new();
        self.recompute_post_order(0, &BTreeSet::new(), &mut visited, true);
        let mut visited = BTreeSet::new();
        self.recompute_pre_order(0, &mut visited);
    }

    fn recompute_post_order(
        &mut self,
        idx: usize,
        dirty: &BTreeSet<BookHash>,
        visited: &mut BTreeSet<BookHash>,
        all_dirty: bool,
    ) {
        let hash = self.nodes[idx].hash();
        if !visited.insert(hash) {
            return;
        }
        let child_hashes: Vec<BookHash> = self.nodes[idx]
            .moves
            .values()
            .map(|mv| mv.hash)
            .filter(|h| all_dirty || dirty.contains(h))
            .collect();
        for child_hash in child_hashes {
            let child_idx = self.node_idx_by_hash[&child_hash];
            self.recompute_post_order(child_idx, dirty, visited, all_dirty);
        }
        if all_dirty || dirty.contains(&hash) {
            self.recompute_node_values(idx);
        }
    }

    fn recompute_pre_order(&mut self, idx: usize, visited: &mut BTreeSet<BookHash>) {
        let hash = self.nodes[idx].hash();
        if !visited.insert(hash) {
            return;
        }
        self.recompute_node_cost(idx);
        let child_hashes: Vec<BookHash> =
            self.nodes[idx].moves.values().map(|mv| mv.hash).collect();
        for child_hash in child_hashes {
            let child_idx = self.node_idx_by_hash[&child_hash];
            self.recompute_pre_order(child_idx, visited);
        }
    }

    /// Collect node indices in post-order (children before parents).
    fn post_order_indices(&self, dirty: &BTreeSet<BookHash>, all_dirty: bool) -> Vec<usize> {
        let mut order = Vec::new();
        let mut visited = BTreeSet::new();
        self.collect_post_order(0, dirty, &mut visited, all_dirty, &mut order);
        order
    }

    fn collect_post_order(
        &self,
        idx: usize,
        dirty: &BTreeSet<BookHash>,
        visited: &mut BTreeSet<BookHash>,
        all_dirty: bool,
        order: &mut Vec<usize>,
    ) {
        let hash = self.nodes[idx].hash();
        if !visited.insert(hash) {
            return;
        }
        let child_hashes: Vec<BookHash> = self.nodes[idx]
            .moves
            .values()
            .map(|mv| mv.hash)
            .filter(|h| all_dirty || dirty.contains(h))
            .collect();
        for child_hash in child_hashes {
            let child_idx = self.node_idx_by_hash[&child_hash];
            self.collect_post_order(child_idx, dirty, visited, all_dirty, order);
        }
        if all_dirty || dirty.contains(&hash) {
            order.push(idx);
        }
    }

    /// Collect node indices in pre-order (parents before children).
    fn pre_order_indices(&self) -> Vec<usize> {
        let mut order = Vec::new();
        let mut visited = BTreeSet::new();
        self.collect_pre_order(0, &mut visited, &mut order);
        order
    }

    fn collect_pre_order(
        &self,
        idx: usize,
        visited: &mut BTreeSet<BookHash>,
        order: &mut Vec<usize>,
    ) {
        let hash = self.nodes[idx].hash();
        if !visited.insert(hash) {
            return;
        }
        order.push(idx);
        let child_hashes: Vec<BookHash> =
            self.nodes[idx].moves.values().map(|mv| mv.hash).collect();
        for child_hash in child_hashes {
            let child_idx = self.node_idx_by_hash[&child_hash];
            self.collect_pre_order(child_idx, visited, order);
        }
    }

    /// Multithreaded variant of [`Self::recompute`].
    pub fn recompute_multi_threaded(
        &mut self,
        new_and_changed_nodes: &[SymBookNode],
        mutex_pool: &MutexPool,
        num_threads: i32,
    ) {
        if num_threads <= 1 {
            self.recompute(new_and_changed_nodes);
            return;
        }

        let dirty = self.collect_dirty_nodes(new_and_changed_nodes);
        if dirty.is_empty() {
            return;
        }

        let order = self.post_order_indices(&dirty, false);
        let visited_done_value = self.next_visited_done_value;
        self.next_visited_done_value += 1;

        let book_addr = self as *mut Self as usize;
        let counter = AtomicUsize::new(0);
        let order = &order;
        let counter = &counter;
        thread::scope(|s| {
            for _ in 0..num_threads {
                s.spawn(move || {
                    let book = book_addr as *mut Book;
                    loop {
                        let i = counter.fetch_add(1, AtomicOrdering::Relaxed);
                        if i >= order.len() {
                            break;
                        }
                        let idx = order[i];
                        unsafe {
                            lock_and_recompute_node_values(book, idx, mutex_pool);
                            (*book).nodes[idx]
                                .visited_flag
                                .store(visited_done_value, AtomicOrdering::Release);
                        }
                    }
                });
            }
        });
        self.next_visited_done_value += 1;

        let cost_order = self.pre_order_indices();
        let cost_visited_done_value = self.next_visited_done_value;
        self.next_visited_done_value += 1;
        let counter = AtomicUsize::new(0);
        let cost_order = &cost_order;
        let counter = &counter;
        thread::scope(|s| {
            for _ in 0..num_threads {
                s.spawn(move || {
                    let book = book_addr as *mut Book;
                    loop {
                        let i = counter.fetch_add(1, AtomicOrdering::Relaxed);
                        if i >= cost_order.len() {
                            break;
                        }
                        let idx = cost_order[i];
                        unsafe {
                            lock_and_recompute_node_cost(book, idx, mutex_pool);
                            (*book).nodes[idx]
                                .visited_flag
                                .store(cost_visited_done_value, AtomicOrdering::Release);
                        }
                    }
                });
            }
        });
        self.next_visited_done_value += 1;
    }

    /// Multithreaded variant of [`Self::recompute_everything`].
    pub fn recompute_everything_multi_threaded(
        &mut self,
        mutex_pool: &MutexPool,
        num_threads: i32,
    ) {
        if num_threads <= 1 {
            self.recompute_everything();
            return;
        }

        let order = self.post_order_indices(&BTreeSet::new(), true);
        let visited_done_value = self.next_visited_done_value;
        self.next_visited_done_value += 1;

        let book_addr = self as *mut Self as usize;
        let counter = AtomicUsize::new(0);
        let order = &order;
        let counter = &counter;
        thread::scope(|s| {
            for _ in 0..num_threads {
                s.spawn(move || {
                    let book = book_addr as *mut Book;
                    loop {
                        let i = counter.fetch_add(1, AtomicOrdering::Relaxed);
                        if i >= order.len() {
                            break;
                        }
                        let idx = order[i];
                        unsafe {
                            lock_and_recompute_node_values(book, idx, mutex_pool);
                            (*book).nodes[idx]
                                .visited_flag
                                .store(visited_done_value, AtomicOrdering::Release);
                        }
                    }
                });
            }
        });
        self.next_visited_done_value += 1;

        let cost_order = self.pre_order_indices();
        let cost_visited_done_value = self.next_visited_done_value;
        self.next_visited_done_value += 1;
        let counter = AtomicUsize::new(0);
        let cost_order = &cost_order;
        let counter = &counter;
        thread::scope(|s| {
            for _ in 0..num_threads {
                s.spawn(move || {
                    let book = book_addr as *mut Book;
                    loop {
                        let i = counter.fetch_add(1, AtomicOrdering::Relaxed);
                        if i >= cost_order.len() {
                            break;
                        }
                        let idx = cost_order[i];
                        unsafe {
                            lock_and_recompute_node_cost(book, idx, mutex_pool);
                            (*book).nodes[idx]
                                .visited_flag
                                .store(cost_visited_done_value, AtomicOrdering::Release);
                        }
                    }
                });
            }
        });
        self.next_visited_done_value += 1;
    }

    /// Return all nodes in the book.
    pub fn get_all_nodes(&self) -> Vec<SymBookNode> {
        self.nodes
            .iter()
            .map(|node| {
                SymBookNode::new(
                    self.node_idx_by_hash[&node.hash()],
                    node.hash(),
                    node.pla(),
                    0,
                )
            })
            .collect()
    }

    /// Return all leaf nodes with at least `min_visits` total visits.
    pub fn get_all_leaves(&self, min_visits: f64) -> Vec<SymBookNode> {
        self.nodes
            .iter()
            .filter(|node| node.moves.is_empty() && node.recursive_values.visits >= min_visits)
            .map(|node| {
                SymBookNode::new(
                    self.node_idx_by_hash[&node.hash()],
                    node.hash(),
                    node.pla(),
                    0,
                )
            })
            .collect()
    }

    /// Return the next `n` nodes to expand, sorted by total expansion cost.
    pub fn get_next_n_to_expand(&self, n: usize) -> Vec<SymBookNode> {
        let mut candidates: Vec<_> = self
            .nodes
            .iter()
            .filter(|node| node.can_expand)
            .map(|node| {
                let cost = node.min_cost_from_root + node.this_node_expansion_cost;
                (
                    cost,
                    self.node_idx_by_hash[&node.hash()],
                    node.hash(),
                    node.pla(),
                )
            })
            .collect();
        candidates.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.2.cmp(&b.2))
        });
        candidates
            .into_iter()
            .take(n)
            .map(|(_, idx, hash, pla)| SymBookNode::new(idx, hash, pla, 0))
            .collect()
    }

    /// Save the book to `file_name` in the C++ book file format (version 2).
    pub fn save_to_file<P: AsRef<Path>>(&self, file_name: P) -> Result<(), IOError> {
        let mut tmp_name = file_name.as_ref().as_os_str().to_owned();
        tmp_name.push(".tmp");
        let tmp_path = Path::new(&tmp_name);

        let mut out = fs::File::create(tmp_path)
            .map_err(|e| IOError(format!("Could not create book tmp file: {e}")))?;

        let params_dump = json!({
            "version": self.book_version,
            "initialBoard": self.initial_board.to_json(),
            "initialRules": self.initial_rules.to_json_string(),
            "initialPla": player_io::player_to_string(self.initial_pla),
            "repBound": self.rep_bound,
            "errorFactor": self.params.error_factor,
            "costPerMove": self.params.cost_per_move,
            "costPerUCBWinLossLoss": self.params.cost_per_ucb_win_loss_loss,
            "costPerUCBWinLossLossPow3": self.params.cost_per_ucb_win_loss_loss_pow3,
            "costPerUCBWinLossLossPow7": self.params.cost_per_ucb_win_loss_loss_pow7,
            "costPerUCBScoreLoss": self.params.cost_per_ucb_score_loss,
            "costPerLogPolicy": self.params.cost_per_log_policy,
            "costPerMovesExpanded": self.params.cost_per_moves_expanded,
            "costPerSquaredMovesExpanded": self.params.cost_per_squared_moves_expanded,
            "costWhenPassFavored": self.params.cost_when_pass_favored,
            "bonusPerWinLossError": self.params.bonus_per_win_loss_error,
            "bonusPerScoreError": self.params.bonus_per_score_error,
            "bonusPerSharpScoreDiscrepancy": self.params.bonus_per_sharp_score_discrepancy,
            "bonusPerExcessUnexpandedPolicy": self.params.bonus_per_excess_unexpanded_policy,
            "bonusPerUnexpandedBestWinLoss": self.params.bonus_per_unexpanded_best_win_loss,
            "bonusForWLPV1": self.params.bonus_for_wl_pv1,
            "bonusForWLPV2": self.params.bonus_for_wl_pv2,
            "bonusForWLPVFinalProp": self.params.bonus_for_wl_pv_final_prop,
            "bonusForBiggestWLCost": self.params.bonus_for_biggest_wl_cost,
            "bonusBehindInVisitsScale": self.params.bonus_behind_in_visits_scale,
            "scoreLossCap": self.params.score_loss_cap,
            "earlyBookCostReductionFactor": self.params.early_book_cost_reduction_factor,
            "earlyBookCostReductionLambda": self.params.early_book_cost_reduction_lambda,
            "utilityPerScore": self.params.utility_per_score,
            "policyBoostSoftUtilityScale": self.params.policy_boost_soft_utility_scale,
            "utilityPerPolicyForSorting": self.params.utility_per_policy_for_sorting,
            "adjustedVisitsWLScale": self.params.adjusted_visits_wl_scale,
            "maxVisitsForReExpansion": self.params.max_visits_for_re_expansion,
            "visitsScale": self.params.visits_scale,
            "visitsScaleLeaves": self.params.visits_scale_leaves,
            "sharpScoreOutlierCap": self.params.sharp_score_outlier_cap,
            "initialSymmetry": self.initial_symmetry,
        });
        writeln!(
            out,
            "{}",
            serde_json::to_string(&params_dump)
                .map_err(|e| IOError(format!("Could not serialize book params: {e}")))?
        )
        .map_err(|e| IOError(format!("Could not write book params: {e}")))?;

        writeln!(out, "{}", self.nodes.len())
            .map_err(|e| IOError(format!("Could not write book node count: {e}")))?;
        for node in &self.nodes {
            writeln!(out, "{}", node.hash.to_string())
                .map_err(|e| IOError(format!("Could not write book hash: {e}")))?;
        }

        for (node_idx, node) in self.nodes.iter().enumerate() {
            let values = &node.this_values_not_in_book;
            let mut node_data = json!({
                "id": node_idx,
                "pla": player_io::player_to_string_short(node.pla),
                "syms": node.symmetries,
                "wl": round_double(values.win_loss_value, 100_000_000.0),
                "sM": round_double(values.score_mean, 1_000_000.0),
                "ssM": round_double(values.sharp_score_mean_raw, 1_000_000.0),
                "wlE": round_double(values.win_loss_error, 100_000_000.0),
                "sE": round_double(values.score_error, 1_000_000.0),
                "sStd": round_double(values.score_stdev, 1_000_000.0),
                "maxP": values.max_policy,
                "w": round_double(values.weight, 1_000.0),
                "v": values.visits,
                "cEx": node.can_expand,
                "mvs": json!([]),
                "par": json!([]),
            });

            let x_size = self.initial_board.x_size;
            let y_size = self.initial_board.y_size;
            if let Some(mvs) = node_data["mvs"].as_array_mut() {
                for (loc, book_move) in &node.moves {
                    mvs.push(json!({
                        "m": location::to_string(*loc, x_size, y_size),
                        "sym": book_move.symmetry_to_align,
                        "id": self.node_idx_by_hash[&book_move.hash],
                        "rP": book_move.raw_policy,
                    }));
                }
            }
            if let Some(par) = node_data["par"].as_array_mut() {
                for (parent_hash, parent_loc) in &node.parents {
                    par.push(json!({
                        "id": self.node_idx_by_hash[parent_hash],
                        "loc": location::to_string(*parent_loc, x_size, y_size),
                    }));
                }
            }

            writeln!(
                out,
                "{}",
                serde_json::to_string(&node_data)
                    .map_err(|e| IOError(format!("Could not serialize book node: {e}")))?
            )
            .map_err(|e| IOError(format!("Could not write book node: {e}")))?;
        }

        out.flush()
            .map_err(|e| IOError(format!("Could not flush book file: {e}")))?;
        drop(out);

        fs::rename(tmp_path, file_name)
            .map_err(|e| IOError(format!("Could not finalize book file: {e}")))?;
        Ok(())
    }

    /// Load a book from `file_name` in the C++ book file format.
    ///
    /// Currently supports version 2 books.
    pub fn load_from_file<P: AsRef<Path>>(file_name: P) -> Result<Self, IOError> {
        let file = fs::File::open(&file_name)
            .map_err(|e| IOError(format!("Could not open book file: {e}")))?;
        let mut reader = io::BufReader::new(file);
        let mut line = String::new();

        reader
            .read_line(&mut line)
            .map_err(|e| IOError(format!("Could not read book params line: {e}")))?;
        if line.is_empty() {
            return Err(IOError("Could not read book params line".to_string()));
        }
        let params: Value = serde_json::from_str(kata_core::global::trim(&line))
            .map_err(|e| IOError(format!("Could not parse book params: {e}")))?;

        let version = params["version"]
            .as_i64()
            .ok_or_else(|| IOError("Book params missing version".to_string()))?
            as i32;
        if version != 2 {
            return Err(IOError(format!("Unsupported book version: {version}")));
        }

        let initial_board = Board::of_json(&params["initialBoard"])?;
        let initial_rules = Rules::parse_rules(
            params["initialRules"]
                .as_str()
                .ok_or_else(|| IOError("Book params missing initialRules".to_string()))?,
        )?;
        let initial_pla = player_io::parse_player(
            params["initialPla"]
                .as_str()
                .ok_or_else(|| IOError("Book params missing initialPla".to_string()))?,
        )?;
        let rep_bound = params["repBound"]
            .as_i64()
            .ok_or_else(|| IOError("Book params missing repBound".to_string()))?
            as i32;
        let book_params = Self::params_from_json(&params)?;

        let mut book = Self::new(
            version,
            initial_board,
            initial_rules,
            initial_pla,
            rep_bound,
            book_params,
        );

        let saved_initial_symmetry = params["initialSymmetry"]
            .as_i64()
            .ok_or_else(|| IOError("Book params missing initialSymmetry".to_string()))?
            as i32;
        if book.initial_symmetry != saved_initial_symmetry {
            return Err(IOError(
                "Inconsistent initial symmetry with initialization".to_string(),
            ));
        }

        line.clear();
        reader
            .read_line(&mut line)
            .map_err(|e| IOError(format!("Could not read book hash list size: {e}")))?;
        let hash_dict_size = kata_core::global::trim(&line)
            .parse::<usize>()
            .map_err(|e| IOError(format!("Could not parse book hash list size: {e}")))?;
        let mut hash_dict = Vec::with_capacity(hash_dict_size);
        for _ in 0..hash_dict_size {
            line.clear();
            reader
                .read_line(&mut line)
                .map_err(|e| IOError(format!("Could not read book hash: {e}")))?;
            hash_dict.push(BookHash::from_string(kata_core::global::trim(&line))?);
        }

        let lines = reader.lines();
        for line_result in lines {
            let line =
                line_result.map_err(|e| IOError(format!("Could not read book node: {e}")))?;
            let trimmed = kata_core::global::trim(&line);
            if trimmed.is_empty() {
                break;
            }
            let node_data: Value = serde_json::from_str(trimmed)
                .map_err(|e| IOError(format!("Could not parse book node: {e}")))?;

            let node_idx = node_data["id"]
                .as_i64()
                .ok_or_else(|| IOError("Book node missing id".to_string()))?
                as usize;
            let hash = *hash_dict
                .get(node_idx)
                .ok_or_else(|| IOError("Book node id out of range".to_string()))?;
            let pla = player_io::parse_player(
                node_data["pla"]
                    .as_str()
                    .ok_or_else(|| IOError("Book node missing pla".to_string()))?,
            )?;
            let symmetries: Vec<i32> = node_data["syms"]
                .as_array()
                .ok_or_else(|| IOError("Book node missing syms".to_string()))?
                .iter()
                .map(|v| {
                    v.as_i64()
                        .ok_or_else(|| IOError("Book node symmetry not an integer".to_string()))
                        .map(|x| x as i32)
                })
                .collect::<Result<Vec<i32>, IOError>>()?;

            if !book.node_idx_by_hash.contains_key(&hash) {
                book.add_node(hash, BookNode::new(hash, pla, symmetries));
            }
            let idx = book.node_idx_by_hash[&hash];

            {
                let node = &mut book.nodes[idx];
                if node.pla != pla {
                    return Err(IOError("Inconsistent pla for book node".to_string()));
                }

                let values = &mut node.this_values_not_in_book;
                values.win_loss_value = get_f64(&node_data, "wl")?;
                values.score_mean = get_f64(&node_data, "sM")?;
                values.sharp_score_mean_raw = get_f64(&node_data, "ssM")?;
                values.sharp_score_mean_clamped = values.sharp_score_mean_raw;
                values.win_loss_error = get_f64(&node_data, "wlE")?;
                values.score_error = get_f64(&node_data, "sE")?;
                values.score_stdev = get_f64(&node_data, "sStd")?;
                values.max_policy = get_f64(&node_data, "maxP")?;
                values.weight = get_f64(&node_data, "w")?;
                values.visits = get_f64(&node_data, "v")?;

                node.can_expand = node_data["cEx"].as_bool().unwrap_or(true);
                node.can_re_expand = true;

                let x_size = book.initial_board.x_size;
                let y_size = book.initial_board.y_size;

                node.moves.clear();
                if let Some(mvs) = node_data["mvs"].as_array() {
                    for mv in mvs {
                        let move_loc = location::of_string(
                            mv["m"]
                                .as_str()
                                .ok_or_else(|| IOError("Book move missing m".to_string()))?,
                            x_size,
                            y_size,
                        )?;
                        let symmetry_to_align = mv["sym"]
                            .as_i64()
                            .ok_or_else(|| IOError("Book move missing sym".to_string()))?
                            as i32;
                        let child_idx = mv["id"]
                            .as_i64()
                            .ok_or_else(|| IOError("Book move missing id".to_string()))?
                            as usize;
                        let child_hash = *hash_dict
                            .get(child_idx)
                            .ok_or_else(|| IOError("Book move id out of range".to_string()))?;
                        let raw_policy = get_f64(mv, "rP")?;
                        node.moves.insert(
                            move_loc,
                            BookMove::new(move_loc, symmetry_to_align, child_hash, raw_policy),
                        );
                    }
                }

                node.parents.clear();
                if let Some(par) = node_data["par"].as_array() {
                    for parent_data in par {
                        let parent_idx = parent_data["id"]
                            .as_i64()
                            .ok_or_else(|| IOError("Book parent missing id".to_string()))?
                            as usize;
                        let parent_hash = *hash_dict
                            .get(parent_idx)
                            .ok_or_else(|| IOError("Book parent id out of range".to_string()))?;
                        let loc = location::of_string(
                            parent_data["loc"]
                                .as_str()
                                .ok_or_else(|| IOError("Book parent missing loc".to_string()))?,
                            x_size,
                            y_size,
                        )?;
                        node.parents.push((parent_hash, loc));
                    }
                }
            }
        }

        book.recompute_everything();
        Ok(book)
    }

    fn params_from_json(value: &Value) -> Result<BookParams, IOError> {
        let mut params = BookParams::default();
        if let Some(v) = value.get("errorFactor").and_then(|v| v.as_f64()) {
            params.error_factor = v;
        }
        if let Some(v) = value.get("costPerMove").and_then(|v| v.as_f64()) {
            params.cost_per_move = v;
        }
        if let Some(v) = value.get("costPerUCBWinLossLoss").and_then(|v| v.as_f64()) {
            params.cost_per_ucb_win_loss_loss = v;
        }
        if let Some(v) = value
            .get("costPerUCBWinLossLossPow3")
            .and_then(|v| v.as_f64())
        {
            params.cost_per_ucb_win_loss_loss_pow3 = v;
        }
        if let Some(v) = value
            .get("costPerUCBWinLossLossPow7")
            .and_then(|v| v.as_f64())
        {
            params.cost_per_ucb_win_loss_loss_pow7 = v;
        }
        if let Some(v) = value.get("costPerUCBScoreLoss").and_then(|v| v.as_f64()) {
            params.cost_per_ucb_score_loss = v;
        }
        if let Some(v) = value.get("costPerLogPolicy").and_then(|v| v.as_f64()) {
            params.cost_per_log_policy = v;
        }
        if let Some(v) = value.get("costPerMovesExpanded").and_then(|v| v.as_f64()) {
            params.cost_per_moves_expanded = v;
        }
        if let Some(v) = value
            .get("costPerSquaredMovesExpanded")
            .and_then(|v| v.as_f64())
        {
            params.cost_per_squared_moves_expanded = v;
        }
        if let Some(v) = value.get("costWhenPassFavored").and_then(|v| v.as_f64()) {
            params.cost_when_pass_favored = v;
        }
        if let Some(v) = value.get("bonusPerWinLossError").and_then(|v| v.as_f64()) {
            params.bonus_per_win_loss_error = v;
        }
        if let Some(v) = value.get("bonusPerScoreError").and_then(|v| v.as_f64()) {
            params.bonus_per_score_error = v;
        }
        if let Some(v) = value
            .get("bonusPerSharpScoreDiscrepancy")
            .and_then(|v| v.as_f64())
        {
            params.bonus_per_sharp_score_discrepancy = v;
        }
        if let Some(v) = value
            .get("bonusPerExcessUnexpandedPolicy")
            .and_then(|v| v.as_f64())
        {
            params.bonus_per_excess_unexpanded_policy = v;
        }
        if let Some(v) = value
            .get("bonusPerUnexpandedBestWinLoss")
            .and_then(|v| v.as_f64())
        {
            params.bonus_per_unexpanded_best_win_loss = v;
        }
        if let Some(v) = value.get("bonusForWLPV1").and_then(|v| v.as_f64()) {
            params.bonus_for_wl_pv1 = v;
        }
        if let Some(v) = value.get("bonusForWLPV2").and_then(|v| v.as_f64()) {
            params.bonus_for_wl_pv2 = v;
        }
        if let Some(v) = value.get("bonusForWLPVFinalProp").and_then(|v| v.as_f64()) {
            params.bonus_for_wl_pv_final_prop = v;
        }
        if let Some(v) = value.get("bonusForBiggestWLCost").and_then(|v| v.as_f64()) {
            params.bonus_for_biggest_wl_cost = v;
        }
        if let Some(v) = value
            .get("bonusBehindInVisitsScale")
            .and_then(|v| v.as_f64())
        {
            params.bonus_behind_in_visits_scale = v;
        }
        if let Some(v) = value.get("scoreLossCap").and_then(|v| v.as_f64()) {
            params.score_loss_cap = v;
        }
        if let Some(v) = value
            .get("earlyBookCostReductionFactor")
            .and_then(|v| v.as_f64())
        {
            params.early_book_cost_reduction_factor = v;
        }
        if let Some(v) = value
            .get("earlyBookCostReductionLambda")
            .and_then(|v| v.as_f64())
        {
            params.early_book_cost_reduction_lambda = v;
        }
        if let Some(v) = value.get("utilityPerScore").and_then(|v| v.as_f64()) {
            params.utility_per_score = v;
        }
        if let Some(v) = value
            .get("policyBoostSoftUtilityScale")
            .and_then(|v| v.as_f64())
        {
            params.policy_boost_soft_utility_scale = v;
        }
        if let Some(v) = value
            .get("utilityPerPolicyForSorting")
            .and_then(|v| v.as_f64())
        {
            params.utility_per_policy_for_sorting = v;
        }
        if let Some(v) = value.get("adjustedVisitsWLScale").and_then(|v| v.as_f64()) {
            params.adjusted_visits_wl_scale = v;
        }
        if let Some(v) = value
            .get("maxVisitsForReExpansion")
            .and_then(|v| v.as_f64())
        {
            params.max_visits_for_re_expansion = v;
        }
        if let Some(v) = value.get("visitsScale").and_then(|v| v.as_f64()) {
            params.visits_scale = v;
        }
        if let Some(v) = value.get("visitsScaleLeaves").and_then(|v| v.as_f64()) {
            params.visits_scale_leaves = v;
        }
        if let Some(v) = value.get("sharpScoreOutlierCap").and_then(|v| v.as_f64()) {
            params.sharp_score_outlier_cap = v;
        }
        Ok(params)
    }

    /// Export the book to a directory of static HTML pages.
    ///
    /// Returns the number of HTML files written. Nodes with fewer than
    /// `html_min_visits` total visits are skipped, as are nodes past the normal
    /// game end (encore phase > 0).
    pub fn export_to_html_dir<P: AsRef<Path>>(
        &self,
        dir_name: P,
        rules_label: &str,
        rules_link: &str,
        _dev_mode: bool,
        html_min_visits: f64,
    ) -> Result<usize, IOError> {
        if rules_label.contains('"') || rules_label.contains('\n') {
            return Err(IOError(
                "rulesLabel cannot contain quotes or newlines".to_string(),
            ));
        }
        if rules_link.contains('"') || rules_link.contains('\n') {
            return Err(IOError(
                "rulesLink cannot contain quotes or newlines".to_string(),
            ));
        }

        let dir = dir_name.as_ref();
        fs::create_dir_all(dir).map_err(|e| IOError(format!("Could not create html dir: {e}")))?;

        let hex_chars: &[u8] = b"0123456789ABCDEF";
        for i in 0..16 {
            for j in 0..16 {
                let subdir = format!("{}{}", hex_chars[i] as char, hex_chars[j] as char);
                fs::create_dir_all(dir.join(&subdir))
                    .map_err(|e| IOError(format!("Could not create html subdir: {e}")))?;
            }
        }
        fs::create_dir_all(dir.join("root"))
            .map_err(|e| IOError(format!("Could not create html root subdir: {e}")))?;

        fs::write(dir.join("book.css"), css_js::book_css())
            .map_err(|e| IOError(format!("Could not write book.css: {e}")))?;
        fs::write(
            dir.join("book.js"),
            format!(
                "const rulesLabel = \"{}\";\nconst rulesLink = \"{}\";\nconst bSizeX = {};\nconst bSizeY = {};\n{}",
                html_escape(rules_label),
                html_escape(rules_link),
                self.initial_board.x_size,
                self.initial_board.y_size,
                css_js::book_js()
            ),
        )
        .map_err(|e| IOError(format!("Could not write book.js: {e}")))?;

        let mut num_files_written = 0usize;
        for (node_idx, node) in self.nodes.iter().enumerate() {
            if node.recursive_values.visits < html_min_visits {
                continue;
            }
            let sym_node = SymBookNode::new(node_idx, node.hash, node.pla, 0);
            let (hist, _) = match sym_node.get_board_history_reaching_here(self) {
                Some(h) => h,
                None => continue,
            };
            if hist.encore_phase > 0 {
                continue;
            }
            let board = hist.get_recent_board(0);

            let file_path = if node_idx == 0 {
                dir.join("root").join("root.html")
            } else {
                let hash_str = node.hash.to_string();
                let subdir = &hash_str[8..10];
                dir.join(subdir).join(format!("{hash_str}.html"))
            };

            let mut html = String::new();
            html.push_str("<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n");
            html.push_str("<link rel=\"stylesheet\" href=\"../book.css\">\n");
            html.push_str("</head>\n<body>\n");
            html.push_str(&format!("<h1>{}</h1>\n", html_escape(rules_label)));
            if !rules_link.is_empty() {
                html.push_str(&format!(
                    "<p><a href=\"{}\">Rules</a></p>\n",
                    html_escape(rules_link)
                ));
            }
            html.push_str(&format!(
                "<p>Player to move: {}</p>\n",
                if node.pla == P_BLACK {
                    "Black"
                } else {
                    "White"
                }
            ));

            let parent = sym_node.canonical_parent(self);
            if !parent.is_null() {
                let parent_path = Self::relative_html_path(node_idx, &parent, self);
                html.push_str(&format!(
                    "<p class=\"backLink\"><a href=\"{}\">&larr; Parent</a></p>\n",
                    parent_path
                ));
            }

            html.push_str("<table class=\"board\">\n");
            for y in 0..board.y_size {
                html.push_str("<tr>");
                for x in 0..board.x_size {
                    let loc = location::get_loc(x, y, board.x_size);
                    let color = board.colors[loc as usize];
                    let cell = match color {
                        C_BLACK => "<span class=\"blackStone\">X</span>",
                        C_WHITE => "<span class=\"whiteStone\">O</span>",
                        _ => "&middot;",
                    };
                    html.push_str(&format!("<td>{}</td>", cell));
                }
                html.push_str("</tr>\n");
            }
            html.push_str("</table>\n");

            html.push_str("<h2>Moves</h2>\n");
            html.push_str("<table class=\"moveTable\">\n");
            html.push_str("<tr class=\"moveTableHeader\"><th>Move</th><th>Policy</th><th>Win/Loss</th><th>Visits</th></tr>\n");

            let unique_moves = sym_node.get_unique_moves_in_book(self);
            for book_move in unique_moves {
                let child = sym_node.follow(self, book_move.move_loc);
                if child.is_null() {
                    continue;
                }
                let child_idx = child.node_idx();
                let child_node = &self.nodes[child_idx];
                if child_node.recursive_values.visits < html_min_visits {
                    continue;
                }
                let child_path = Self::relative_html_path(node_idx, &child, self);
                let move_label =
                    location::to_string(book_move.move_loc, board.x_size, board.y_size);
                let vals = &child_node.recursive_values;
                html.push_str(&format!(
                    "<tr class=\"moveTableRow\"><td class=\"moveTableCell\"><a href=\"{}\">{}</a></td>\
                     <td class=\"moveTableCell\">{:.4}</td>\
                     <td class=\"moveTableCell\">{:.4}</td>\
                     <td class=\"moveTableCell\">{:.0}</td></tr>\n",
                    child_path,
                    move_label,
                    book_move.raw_policy,
                    vals.win_loss_value,
                    vals.visits
                ));
            }
            html.push_str("</table>\n");
            html.push_str("</body>\n</html>\n");

            fs::write(&file_path, html)
                .map_err(|e| IOError(format!("Could not write html file: {e}")))?;
            num_files_written += 1;
        }

        Ok(num_files_written)
    }

    fn relative_html_path(from_node_idx: usize, to_node: &SymBookNode, book: &Book) -> String {
        let to_idx = to_node.node_idx();
        if to_idx == 0 {
            if from_node_idx == 0 {
                "root/root.html".to_string()
            } else {
                "../root/root.html".to_string()
            }
        } else {
            let hash_str = book.nodes[to_idx].hash.to_string();
            let subdir = &hash_str[8..10];
            if from_node_idx == 0 {
                format!("{subdir}/{hash_str}.html")
            } else {
                format!("../{subdir}/{hash_str}.html")
            }
        }
    }
}

/// A book node viewed under a particular symmetry.
#[derive(Debug, Clone, Copy)]
pub struct SymBookNode {
    node_idx: Option<usize>,
    hash: BookHash,
    pla: Player,
    symmetry_of_node: i32,
    inv_symmetry_of_node: i32,
}

impl SymBookNode {
    /// Create a null/placeholder symmetric book node.
    pub fn null() -> Self {
        Self {
            node_idx: None,
            hash: BookHash::default(),
            pla: 0,
            symmetry_of_node: 0,
            inv_symmetry_of_node: 0,
        }
    }

    fn new(node_idx: usize, hash: BookHash, pla: Player, symmetry_of_node: i32) -> Self {
        Self {
            node_idx: Some(node_idx),
            hash,
            pla,
            symmetry_of_node,
            inv_symmetry_of_node: symmetry::invert(symmetry_of_node),
        }
    }

    /// Whether this node is null.
    pub fn is_null(&self) -> bool {
        self.node_idx.is_none()
    }

    fn node_idx(&self) -> usize {
        self.node_idx.unwrap_or(NULL_NODE_IDX)
    }

    fn node<'a>(&self, book: &'a Book) -> Option<&'a BookNode> {
        self.node_idx.map(|idx| &book.nodes[idx])
    }

    fn node_mut<'a>(&self, book: &'a mut Book) -> Option<&'a mut BookNode> {
        self.node_idx.map(|idx| &mut book.nodes[idx])
    }

    /// Apply an additional symmetry to this view.
    pub fn apply_symmetry(&self, symmetry: i32) -> Self {
        if self.is_null() {
            return Self::null();
        }
        Self::new(
            self.node_idx(),
            self.hash,
            self.pla,
            symmetry::compose(self.symmetry_of_node, symmetry),
        )
    }

    /// The player to move at this node.
    pub fn pla(&self) -> Player {
        self.pla
    }

    /// The hash identifying this node.
    pub fn hash(&self) -> BookHash {
        self.hash
    }

    /// The symmetry applied to the underlying node to obtain this view.
    pub fn symmetry_of_node(&self) -> i32 {
        self.symmetry_of_node
    }

    /// Symmetries under which this viewed position is invariant.
    pub fn get_symmetries(&self, book: &Book) -> Vec<i32> {
        let node = self.node(book).expect("null SymBookNode");
        node.symmetries()
            .iter()
            .map(|&sym| {
                symmetry::compose_three(self.inv_symmetry_of_node, sym, self.symmetry_of_node)
            })
            .collect()
    }

    /// Whether `move_loc` is already recorded as a child of this node.
    pub fn is_move_in_book(&self, book: &Book, move_loc: Loc) -> bool {
        let node = match self.node(book) {
            Some(n) => n,
            None => return false,
        };
        for &sym in node.symmetries() {
            let composed = symmetry::compose(self.inv_symmetry_of_node, sym);
            let loc_in_node = symmetry::get_sym_loc(move_loc, &book.initial_board, composed);
            if node.moves().contains_key(&loc_in_node) {
                return true;
            }
        }
        false
    }

    /// Number of unique child moves stored for this node.
    pub fn num_unique_moves_in_book(&self, book: &Book) -> usize {
        self.node(book).map_or(0, |node| node.moves().len())
    }

    /// All unique child moves, transformed into this view's orientation.
    pub fn get_unique_moves_in_book(&self, book: &Book) -> Vec<BookMove> {
        let node = match self.node(book) {
            Some(n) => n,
            None => return Vec::new(),
        };
        node.moves()
            .values()
            .map(|mv| {
                mv.get_sym_book_move(
                    self.symmetry_of_node,
                    book.initial_board.x_size,
                    book.initial_board.y_size,
                )
            })
            .collect()
    }

    /// Values based on a search of this node alone, excluding children.
    pub fn this_values_not_in_book<'a>(&self, book: &'a Book) -> Option<&'a BookValues> {
        self.node(book).map(|node| &node.this_values_not_in_book)
    }

    /// Mutable access to [`Self::this_values_not_in_book`].
    pub fn this_values_not_in_book_mut<'a>(
        &self,
        book: &'a mut Book,
    ) -> Option<&'a mut BookValues> {
        self.node_mut(book)
            .map(|node| &mut node.this_values_not_in_book)
    }

    /// Whether more children may be expanded at this node.
    pub fn can_expand(&self, book: &Book) -> bool {
        self.node(book).is_some_and(|node| node.can_expand)
    }

    /// Mutable access to [`Self::can_expand`].
    pub fn can_expand_mut<'a>(&self, book: &'a mut Book) -> Option<&'a mut bool> {
        self.node_mut(book).map(|node| &mut node.can_expand)
    }

    /// Whether this node may be re-expanded during this run.
    pub fn can_re_expand(&self, book: &Book) -> bool {
        self.node(book).is_some_and(|node| node.can_re_expand)
    }

    /// Mutable access to [`Self::can_re_expand`].
    pub fn can_re_expand_mut<'a>(&self, book: &'a mut Book) -> Option<&'a mut bool> {
        self.node_mut(book).map(|node| &mut node.can_re_expand)
    }

    /// Values propagated through the book via minimax.
    pub fn recursive_values<'a>(&self, book: &'a Book) -> Option<&'a RecursiveBookValues> {
        self.node(book).map(|node| &node.recursive_values)
    }

    /// Minimum depth from the root.
    pub fn min_depth_from_root(&self, book: &Book) -> i32 {
        self.node(book).map_or(0, |node| node.min_depth_from_root)
    }

    /// Minimum cost from the root.
    pub fn min_cost_from_root(&self, book: &Book) -> f64 {
        self.node(book).map_or(0.0, |node| node.min_cost_from_root)
    }

    /// Total expansion cost (`min_cost_from_root + this_node_expansion_cost`).
    pub fn total_expansion_cost(&self, book: &Book) -> f64 {
        self.node(book).map_or(0.0, |node| {
            node.min_cost_from_root + node.this_node_expansion_cost
        })
    }

    /// The canonical parent of this node, or null if this is the root.
    pub fn canonical_parent(&self, book: &Book) -> Self {
        let node = match self.node(book) {
            Some(n) => n,
            None => return Self::null(),
        };
        if node.parents.is_empty() {
            return Self::null();
        }
        let mut best_parent_idx = node.best_parent_idx as usize;
        if best_parent_idx >= node.parents.len() {
            best_parent_idx = 0;
        }
        let (parent_hash, parent_move) = node.parents[best_parent_idx];
        let parent = match book.node_by_hash(parent_hash) {
            Some(p) => p,
            None => return Self::null(),
        };
        let book_move = match parent.moves().get(&parent_move) {
            Some(mv) => mv,
            None => return Self::null(),
        };
        Self::new(
            book.node_idx_by_hash[&parent_hash],
            parent.hash(),
            parent.pla(),
            symmetry::compose(book_move.symmetry_to_align, self.symmetry_of_node),
        )
    }

    /// Reconstruct a board history that reaches this node by walking up through
    /// the lowest-cost parent links and replaying the moves.
    ///
    /// Returns `None` if the book structure is corrupted or the path is illegal.
    pub fn get_board_history_reaching_here(&self, book: &Book) -> Option<(BoardHistory, Vec<Loc>)> {
        if self.is_null() {
            return None;
        }
        let mut cur_idx = self.node_idx();
        let mut path_parent_indices: Vec<usize> = Vec::new();
        let mut path_moves: Vec<Loc> = Vec::new();

        // Walk from the target node up to the root, collecting parent indices
        // and the move that leads from each parent to its child.
        while cur_idx != 0 {
            let node = &book.nodes[cur_idx];
            let mut best_parent_idx = node.best_parent_idx as usize;
            if best_parent_idx >= node.parents.len() {
                best_parent_idx = 0;
            }
            let (parent_hash, parent_move) = node.parents.get(best_parent_idx)?;
            let parent_idx = *book.node_idx_by_hash.get(parent_hash)?;
            path_parent_indices.push(parent_idx);
            path_moves.push(*parent_move);
            cur_idx = parent_idx;
        }
        path_parent_indices.reverse();
        path_moves.reverse();

        // Compose the symmetry from initial space to the final hist space.
        let mut symmetry_acc = book.initial_symmetry;
        for (parent_idx, parent_move) in path_parent_indices.iter().zip(path_moves.iter()) {
            let parent = &book.nodes[*parent_idx];
            let book_move = parent.moves.get(parent_move)?;
            symmetry_acc = symmetry::compose(symmetry_acc, book_move.symmetry_to_align);
        }
        symmetry_acc = symmetry::compose(symmetry_acc, self.symmetry_of_node);

        let mut hist = book.get_initial_hist_with_symmetry(symmetry_acc);
        let mut board = hist.get_recent_board(0).clone();
        let mut move_history = Vec::new();

        // symmetryPathNodeToHist maps the current parent node space to hist space.
        let mut symmetry_path_node_to_hist =
            symmetry::compose(symmetry::invert(book.initial_symmetry), symmetry_acc);
        for (parent_idx, parent_move) in path_parent_indices.iter().zip(path_moves.iter()) {
            let parent = &book.nodes[*parent_idx];
            let sym_move = symmetry::get_sym_loc(
                *parent_move,
                &book.initial_board,
                symmetry_path_node_to_hist,
            );
            move_history.push(sym_move);
            if board.is_ko_banned(sym_move) || !hist.is_legal_tolerant(&board, sym_move, parent.pla)
            {
                return None;
            }
            hist.make_board_move_assume_legal(&mut board, sym_move, parent.pla);
            let book_move = parent.moves.get(parent_move)?;
            symmetry_path_node_to_hist = symmetry::compose(
                symmetry::invert(book_move.symmetry_to_align),
                symmetry_path_node_to_hist,
            );
        }

        Some((hist, move_history))
    }

    /// Follow an existing book move, returning the child node or null.
    pub fn follow(&self, book: &Book, move_loc: Loc) -> Self {
        let node = match self.node(book) {
            Some(n) => n,
            None => return Self::null(),
        };
        for &sym in node.symmetries() {
            let composed = symmetry::compose(self.inv_symmetry_of_node, sym);
            let loc_in_node = symmetry::get_sym_loc(move_loc, &book.initial_board, composed);
            if let Some(book_move) = node.moves().get(&loc_in_node) {
                let child_hash = book_move.hash;
                let child_idx = match book.node_idx_by_hash.get(&child_hash) {
                    Some(&idx) => idx,
                    None => return Self::null(),
                };
                let child = &book.nodes[child_idx];
                let child_symmetry =
                    symmetry::invert(symmetry::compose(composed, book_move.symmetry_to_align));
                return Self::new(child_idx, child.hash(), child.pla(), child_symmetry);
            }
        }
        Self::null()
    }

    /// Play a move that is already in the book, updating `board`/`hist` and
    /// returning the child node. Returns null if the move is illegal or not in
    /// the book.
    pub fn play_move(
        &self,
        book: &Book,
        board: &mut Board,
        hist: &mut BoardHistory,
        move_loc: Loc,
    ) -> Self {
        let child = self.follow(book, move_loc);
        if child.is_null() {
            return child;
        }
        if !hist.is_legal(board, move_loc, self.pla) {
            return Self::null();
        }
        hist.make_board_move_assume_legal(board, move_loc, self.pla);
        child
    }

    /// Play and add a new move to the book, updating `board`/`hist` and
    /// returning the child node plus a flag indicating whether the child
    /// transposes to an existing node.
    ///
    /// Returns null if the move is illegal or already in the book.
    pub fn play_and_add_move(
        &self,
        book: &mut Book,
        board: &mut Board,
        hist: &mut BoardHistory,
        move_loc: Loc,
        raw_policy: f64,
    ) -> (Self, bool) {
        assert!(!self.is_null());
        assert!(!self.is_move_in_book(book, move_loc));

        if !hist.is_legal(board, move_loc, self.pla) {
            return (Self::null(), false);
        }

        let x_size = book.initial_board.x_size;
        let y_size = book.initial_board.y_size;
        let sym_move =
            symmetry::get_sym_loc_with_size(move_loc, x_size, y_size, self.inv_symmetry_of_node);

        let mut best_loc = sym_move;
        let mut best_symmetry = 0;
        let node = self.node(book).expect("null SymBookNode");
        for &sym in node.symmetries() {
            if sym == 0 {
                continue;
            }
            let sym_loc = symmetry::get_sym_loc_with_size(sym_move, x_size, y_size, sym);
            let (sym_x, sym_y) = (
                location::get_x(sym_loc, x_size),
                location::get_y(sym_loc, x_size),
            );
            let (best_x, best_y) = (
                location::get_x(best_loc, x_size),
                location::get_y(best_loc, x_size),
            );
            if sym_x > best_x || (sym_x == best_x && sym_y < best_y) {
                best_loc = sym_loc;
                best_symmetry = sym;
            }
        }

        hist.make_board_move_assume_legal(board, move_loc, self.pla);
        let child_hash_and_sym =
            BookHash::get_hash_and_symmetry(hist, book.rep_bound, book.book_version);
        let child_hash = child_hash_and_sym.hash;
        let symmetry_to_align_to_child = child_hash_and_sym.symmetry_to_align;

        let child_is_transposing = book.node_idx_by_hash.contains_key(&child_hash);
        if !child_is_transposing {
            let child_pla = hist.presumed_next_move_pla;
            let child_node = BookNode::new(child_hash, child_pla, child_hash_and_sym.symmetries);
            let added = book.add_node(child_hash, child_node);
            assert!(added, "book node hash collision on add");
        }

        let child_idx = book.node_idx_by_hash[&child_hash];
        {
            let parent_idx = self.node_idx();
            book.node_mut(child_idx).parents.push((self.hash, best_loc));
            let new_book_move = BookMove::new(
                best_loc,
                symmetry::compose_three(
                    symmetry::invert(best_symmetry),
                    self.symmetry_of_node,
                    symmetry_to_align_to_child,
                ),
                child_hash,
                raw_policy,
            );
            book.node_mut(parent_idx)
                .moves_mut()
                .insert(best_loc, new_book_move);
        }

        let child_symmetry = symmetry::invert(symmetry_to_align_to_child);
        let child_node = &book.nodes[child_idx];
        (
            Self::new(
                child_idx,
                child_node.hash(),
                child_node.pla(),
                child_symmetry,
            ),
            child_is_transposing,
        )
    }
}

impl Default for SymBookNode {
    fn default() -> Self {
        Self::null()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kata_game::board::{Board, P_BLACK, P_WHITE, location};
    use kata_game::rules::Rules;

    #[test]
    fn book_hash_roundtrip() {
        let h = BookHash::new(Hash128::new(1, 2), Hash128::new(3, 4));
        let node = BookNode::new(h, 0, vec![0, 1]);
        assert_eq!(node.hash(), h);
        assert_eq!(node.pla(), 0);
        assert_eq!(node.symmetries(), &[0, 1]);
    }

    #[test]
    fn sym_book_node_null() {
        assert!(SymBookNode::null().is_null());
    }

    #[test]
    fn book_css_js_are_non_empty() {
        assert!(!css_js::book_css().is_empty());
        assert!(!css_js::book_js().is_empty());
    }

    #[test]
    fn empty_board_has_all_symmetries() {
        let board = Board::new(7, 7);
        let hist = BoardHistory::new(board, P_BLACK, Rules::default(), 0);
        let result = BookHash::get_hash_and_symmetry(&hist, 3, 2);
        assert_eq!(result.symmetry_to_align, 0);
        assert_eq!(result.symmetries.len(), 8);
        for sym in 0..8 {
            assert!(result.symmetries.contains(&sym));
        }
    }

    #[test]
    fn rectangular_board_skips_transpositions() {
        let board = Board::new(5, 7);
        let hist = BoardHistory::new(board, P_BLACK, Rules::default(), 0);
        let result = BookHash::get_hash_and_symmetry(&hist, 3, 2);
        assert_eq!(result.symmetry_to_align, 0);
        assert_eq!(result.symmetries.len(), 4);
        for sym in 0..4 {
            assert!(result.symmetries.contains(&sym));
        }
    }

    #[test]
    fn single_move_breaks_most_symmetries() {
        let mut board = Board::new(7, 7);
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        // A move not on any symmetry axis should have no non-trivial symmetries.
        let b2 = location::get_loc(1, 2, 7);
        hist.make_board_move_assume_legal(&mut board, b2, P_BLACK);

        let result = BookHash::get_hash_and_symmetry(&hist, 3, 2);
        assert_eq!(result.symmetries, vec![0]);
    }

    #[test]
    fn center_stone_keeps_some_symmetries() {
        let mut board = Board::new(7, 7);
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        let center = location::get_loc(3, 3, 7);
        hist.make_board_move_assume_legal(&mut board, center, P_BLACK);

        let result = BookHash::get_hash_and_symmetry(&hist, 3, 2);
        assert!(result.symmetries.contains(&0));
        assert!(result.symmetries.len() > 1);
    }

    #[test]
    fn book_hash_ordering_matches_cpp() {
        let a = BookHash::new(Hash128::new(1, 2), Hash128::new(3, 4));
        let b = BookHash::new(Hash128::new(1, 2), Hash128::new(4, 4));
        assert!(a < b);
        let c = BookHash::new(Hash128::new(2, 2), Hash128::new(3, 4));
        let d = BookHash::new(Hash128::new(1, 2), Hash128::new(3, 4));
        assert!(c > d);
    }

    #[test]
    fn book_root_and_add_move() {
        let board = Board::new(7, 7);
        let rules = Rules::default();
        let mut book = Book::new(2, board.clone(), rules, P_BLACK, 3, BookParams::default());
        assert_eq!(book.size(), 1);

        let root = book.get_root();
        assert!(!root.is_null());
        assert_eq!(root.pla(), P_BLACK);

        let mut hist = book.get_initial_hist();
        let mut board = hist.get_recent_board(0).clone();
        let c3 = location::get_loc(2, 2, 7);
        assert!(!root.is_move_in_book(&book, c3));

        let (child, transposing) =
            root.play_and_add_move(&mut book, &mut board, &mut hist, c3, 0.5);
        assert!(!child.is_null());
        assert!(!transposing);
        assert_eq!(book.size(), 2);
        assert!(root.is_move_in_book(&book, c3));
        assert_eq!(child.pla(), P_WHITE);

        // Following the same move from the root returns an equivalent node.
        let followed = root.follow(&book, c3);
        assert!(!followed.is_null());
        assert_eq!(followed.hash(), child.hash());
    }

    #[test]
    fn book_play_rejects_illegal_move() {
        let board = Board::new(7, 7);
        let rules = Rules::default();
        let book = Book::new(2, board.clone(), rules, P_BLACK, 3, BookParams::default());
        let root = book.get_root();
        let mut hist = book.get_initial_hist();
        let mut board = hist.get_recent_board(0).clone();
        let child = root.play_move(&book, &mut board, &mut hist, location::get_loc(-1, -1, 7));
        assert!(child.is_null());
    }

    #[test]
    fn book_get_by_hist() {
        let board = Board::new(7, 7);
        let rules = Rules::default();
        let mut book = Book::new(2, board.clone(), rules, P_BLACK, 3, BookParams::default());
        let root = book.get_root();
        let mut hist = book.get_initial_hist();
        let mut board = hist.get_recent_board(0).clone();
        let c3 = location::get_loc(2, 2, 7);
        let (child, _) = root.play_and_add_move(&mut book, &mut board, &mut hist, c3, 0.5);

        let looked_up = book.get(&hist);
        assert!(!looked_up.is_null());
        assert_eq!(looked_up.hash(), child.hash());
    }

    #[test]
    fn book_params_load_from_cfg() {
        let cfg_str = r#"
errorFactor = 2.0
costPerMove = 3.0
costPerUCBWinLossLoss = 4.0
costPerUCBWinLossLossPow3 = 0.5
costPerUCBWinLossLossPow7 = 0.7
costPerUCBScoreLoss = 1.5
costPerLogPolicy = 2.5
costPerMovesExpanded = 3.5
costPerSquaredMovesExpanded = 4.5
costWhenPassFavored = 5.5
bonusPerWinLossError = 6.5
bonusPerScoreError = 7.5
bonusPerSharpScoreDiscrepancy = 8.5
bonusPerExcessUnexpandedPolicy = 9.5
bonusPerUnexpandedBestWinLoss = 10.5
bonusForWLPVFinalProp = 0.75
scoreLossCap = 5000.0
utilityPerScore = 11.5
policyBoostSoftUtilityScale = 12.5
utilityPerPolicyForSorting = 13.5
sharpScoreOutlierCap = 14.5
visitsScale = 1234
"#;
        let cfg = ConfigParser::from_str(cfg_str, false, false).unwrap();
        let params = BookParams::load_from_cfg(&cfg, 1000, 500).unwrap();
        assert_eq!(params.error_factor, 2.0);
        assert_eq!(params.cost_per_move, 3.0);
        assert_eq!(params.cost_per_ucb_win_loss_loss, 4.0);
        assert_eq!(params.bonus_for_wl_pv_final_prop, 0.75);
        assert_eq!(params.visits_scale, 1234.0);
        // Defaults for absent keys.
        assert_eq!(params.bonus_for_wl_pv1, 0.0);
        assert_eq!(params.bonus_for_wl_pv2, 0.0);
        assert_eq!(params.bonus_for_biggest_wl_cost, 0.0);
        assert_eq!(params.bonus_behind_in_visits_scale, 0.0);
        assert_eq!(params.early_book_cost_reduction_factor, 0.0);
        assert_eq!(params.early_book_cost_reduction_lambda, 0.5);
        assert_eq!(params.adjusted_visits_wl_scale, 0.05);
        assert_eq!(params.max_visits_for_re_expansion, 0.0);
        assert_eq!(params.visits_scale_leaves, 500.0);

        // visitsScale defaults to (max_visits + 1) / 2 using integer division.
        let required_keys = r#"
errorFactor = 1.0
costPerMove = 1.0
costPerUCBWinLossLoss = 0.0
costPerUCBWinLossLossPow3 = 0.0
costPerUCBWinLossLossPow7 = 0.0
costPerUCBScoreLoss = 0.0
costPerLogPolicy = 0.0
costPerMovesExpanded = 1.0
costPerSquaredMovesExpanded = 0.0
costWhenPassFavored = 0.0
bonusPerWinLossError = 0.0
bonusPerScoreError = 0.0
bonusPerSharpScoreDiscrepancy = 0.0
bonusPerExcessUnexpandedPolicy = 0.0
bonusPerUnexpandedBestWinLoss = 0.0
scoreLossCap = 10000.0
utilityPerScore = 0.0
policyBoostSoftUtilityScale = 1.0
utilityPerPolicyForSorting = 0.0
sharpScoreOutlierCap = 10000.0
"#;
        let cfg_no_visits_scale = ConfigParser::from_str(required_keys, false, false).unwrap();
        let params_default_visits =
            BookParams::load_from_cfg(&cfg_no_visits_scale, 1000, 500).unwrap();
        assert_eq!(params_default_visits.visits_scale, 500.0);
        assert_eq!(params_default_visits.visits_scale_leaves, 500.0);
    }
}
