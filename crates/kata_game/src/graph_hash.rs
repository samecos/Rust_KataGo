//! Graph hash for superko-safe transpositions.
//!
//! Corresponds to `cpp/game/graphhash.h` and `cpp/game/graphhash.cpp`.

use crate::board::{Board, NULL_LOC, Player};
use crate::history::BoardHistory;
use kata_core::hash;
use kata_core::hash::Hash128;

const CONSECPASS_MULT0: u64 = 0x9e3779b97f4a7c15;
const CONSECPASS_MULT1: u64 = 0x853E097C279EBF4E;

/// Hash taking into account all state relevant for move legality, the rules,
/// and some immediate other info like the effect of passing.
/// Does NOT take into account more complex history beyond immediate ko and
/// superko bans.
pub fn get_state_hash(
    hist: &BoardHistory,
    next_player: Player,
    draw_equivalent_wins_for_white: f64,
) -> Hash128 {
    let board = hist.get_recent_board(0);
    let mut hash = BoardHistory::get_situation_rules_and_ko_hash(
        board,
        hist,
        next_player,
        draw_equivalent_wins_for_white,
    );

    if hist.pass_would_end_phase(board, next_player) {
        hash ^= Board::ZOBRIST_PASS_ENDS_PHASE;
    }
    if hist.is_game_finished {
        hash ^= Board::ZOBRIST_GAME_IS_OVER;
    }

    hash.hash0 = hash
        .hash0
        .wrapping_add(CONSECPASS_MULT0.wrapping_mul(hist.consecutive_ending_passes as u64));
    hash.hash1 = hash
        .hash1
        .wrapping_add(CONSECPASS_MULT1.wrapping_mul(hist.consecutive_ending_passes as u64));
    hash
}

/// Call this AFTER making a move, to update a hash suitable for superko-safe
/// transpositions, given the previous graph hash. Will guard against cycles up
/// to `rep_bound` in size and possibly some slightly larger cycles.
pub fn get_graph_hash(
    prev_graph_hash: Hash128,
    hist: &BoardHistory,
    next_player: Player,
    rep_bound: i32,
    draw_equivalent_wins_for_white: f64,
) -> Hash128 {
    let board = hist.get_recent_board(0);
    let prev_move_loc = if hist.move_history.is_empty() {
        NULL_LOC
    } else {
        hist.move_history[hist.move_history.len() - 1].loc
    };

    if prev_move_loc == NULL_LOC || board.simple_repetition_bound_gt(prev_move_loc, rep_bound) {
        get_state_hash(hist, next_player, draw_equivalent_wins_for_white)
    } else {
        let mut new_hash = prev_graph_hash;
        new_hash.hash0 = hash::split_mix64(new_hash.hash0 ^ new_hash.hash1);
        new_hash.hash1 = hash::nasam(new_hash.hash1).wrapping_add(new_hash.hash0);
        let state_hash = get_state_hash(hist, next_player, draw_equivalent_wins_for_white);
        new_hash.hash0 = new_hash.hash0.wrapping_add(state_hash.hash0);
        new_hash.hash1 = new_hash.hash1.wrapping_add(state_hash.hash1);
        new_hash
    }
}

/// Compute graph hash from scratch by replaying the whole history.
pub fn get_graph_hash_from_scratch(
    hist_orig: &BoardHistory,
    next_player: Player,
    rep_bound: i32,
    draw_equivalent_wins_for_white: f64,
) -> Hash128 {
    let mut hist = hist_orig.copy_to_initial();
    let mut board = hist.get_recent_board(0).clone();
    let mut graph_hash = Hash128::default();

    for i in 0..hist_orig.move_history.len() {
        let m = hist_orig.move_history[i];
        graph_hash = get_graph_hash(
            graph_hash,
            &hist,
            m.pla,
            rep_bound,
            draw_equivalent_wins_for_white,
        );
        let prevent_encore = hist_orig.prevent_encore_history()[i];
        let suc =
            hist.make_board_move_tolerant_with_prevent(&mut board, m.loc, m.pla, prevent_encore);
        assert!(suc, "get_graph_hash_from_scratch: failed to replay move");
    }

    assert_eq!(
        BoardHistory::get_situation_rules_and_ko_hash(
            &board,
            &hist,
            next_player,
            draw_equivalent_wins_for_white
        ),
        BoardHistory::get_situation_rules_and_ko_hash(
            hist_orig.get_recent_board(0),
            hist_orig,
            next_player,
            draw_equivalent_wins_for_white
        ),
        "get_graph_hash_from_scratch: replayed state hash mismatch"
    );

    graph_hash = get_graph_hash(
        graph_hash,
        &hist,
        next_player,
        rep_bound,
        draw_equivalent_wins_for_white,
    );
    graph_hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, P_BLACK, P_WHITE, location};
    use crate::rules::Rules;

    #[test]
    fn test_state_hash_changes_with_player() {
        let board = Board::new(5, 5);
        let hist = BoardHistory::new(board, P_BLACK, Rules::default(), 0);
        let h1 = get_state_hash(&hist, P_BLACK, 0.5);
        let h2 = get_state_hash(&hist, P_WHITE, 0.5);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_graph_hash_incremental_matches_from_scratch() {
        let mut board = Board::new(5, 5);
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);

        let c3 = location::get_loc(2, 2, 5);
        let d3 = location::get_loc(3, 2, 5);
        let c2 = location::get_loc(2, 1, 5);

        let mut graph_hash = Hash128::default();
        hist.make_board_move_assume_legal(&mut board, c3, P_BLACK);
        graph_hash = get_graph_hash(graph_hash, &hist, P_WHITE, 3, 0.5);

        hist.make_board_move_assume_legal(&mut board, d3, P_WHITE);
        graph_hash = get_graph_hash(graph_hash, &hist, P_BLACK, 3, 0.5);

        hist.make_board_move_assume_legal(&mut board, c2, P_BLACK);
        graph_hash = get_graph_hash(graph_hash, &hist, P_WHITE, 3, 0.5);

        let from_scratch = get_graph_hash_from_scratch(&hist, P_WHITE, 3, 0.5);
        assert_eq!(graph_hash, from_scratch);
    }

    #[test]
    fn test_graph_hash_changes_after_move() {
        let mut board = Board::new(5, 5);
        let hist_before = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        let h_before = get_state_hash(&hist_before, P_BLACK, 0.5);

        let mut hist_after = hist_before.clone();
        hist_after.make_board_move_assume_legal(&mut board, location::get_loc(2, 2, 5), P_BLACK);
        let h_after = get_state_hash(&hist_after, P_WHITE, 0.5);

        assert_ne!(h_before, h_after);
    }
}
