//! Smoke tests ported from `KataGo/cpp/tests/testnnevalcanary.cpp`.
//!
//! The original canary tests assert exact NN outputs from a real model file.
//! Since the Rust evaluator is currently a skeleton/dummy backend, these tests
//! only verify that `NnEvaluator::evaluate` runs on the same positions and
//! produces structurally valid outputs.

use std::sync::Arc;

use kata_core::config::ConfigParser;
use kata_core::logger::{Logger, LoggerOptions};
use kata_data::sgf::CompactSgf;
use kata_game::board::{Board, Player, location};
use kata_game::history::BoardHistory;
use kata_nn::backend::{Enabled, NNResultBuf};
use kata_nn::eval::NnEvaluator;
use kata_nn::inputs::{MiscNNInputParams, nn_pos};

fn make_dummy_eval(nn_x_len: i32, nn_y_len: i32, require_exact_nn_len: bool) -> NnEvaluator {
    let logger = Arc::new(Logger::new(LoggerOptions::default(), None));
    let cfg = ConfigParser::new(false, false);
    let mut nn_eval = NnEvaluator::new(
        "test-model".to_string(),
        "/dev/null".to_string(),
        String::new(),
        logger,
        16,
        nn_x_len,
        nn_y_len,
        require_exact_nn_len,
        false,
        16,
        12,
        true,
        String::new(),
        Enabled::False,
        1,
        vec![0],
        "canaryTestRandSeed".to_string(),
        false,
        0,
        true,
        &cfg,
    );
    nn_eval.spawn_server_threads();
    nn_eval
}

fn evaluate_position(
    nn_eval: &NnEvaluator,
    sgf_str: &str,
    turn_idx: i64,
    symmetry: i32,
    override_komi: Option<f32>,
) {
    let sgf = CompactSgf::parse(sgf_str).expect("valid SGF");
    let initial_rules = sgf.get_rules_or_fail().expect("rules");
    let mut board = Board::default();
    let mut next_pla: Player = 0;
    let mut hist = BoardHistory::default();
    sgf.setup_board_and_hist_assume_legal(
        &initial_rules,
        &mut board,
        &mut next_pla,
        &mut hist,
        turn_idx,
    )
    .expect("setup board");
    if let Some(komi) = override_komi {
        hist.set_komi(komi);
    }

    let mut buf = NNResultBuf::new();
    let nn_input_params = MiscNNInputParams {
        symmetry,
        ..MiscNNInputParams::default()
    };
    nn_eval.evaluate(
        &board,
        &hist,
        next_pla,
        &nn_input_params,
        &mut buf,
        true,
        true,
    );

    let output = buf.result.expect("evaluate should produce a result");

    // Collect probabilities for legal moves and pass; they should form a
    // distribution summing to 1. Other entries in the MAX_NN_POLICY_SIZE array
    // are placeholder values.
    let mut legal_probs = Vec::new();
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = kata_game::board::location::get_loc(x, y, board.x_size);
            if hist.is_legal(&board, loc, next_pla) {
                let pos =
                    nn_pos::loc_to_pos(loc, board.x_size, nn_eval.nn_x_len(), nn_eval.nn_y_len());
                legal_probs.push(output.policy_probs[pos as usize]);
            }
        }
    }
    if hist.is_legal(&board, kata_game::board::PASS_LOC, next_pla) {
        let pass_pos = nn_pos::loc_to_pos(
            kata_game::board::PASS_LOC,
            board.x_size,
            nn_eval.nn_x_len(),
            nn_eval.nn_y_len(),
        );
        legal_probs.push(output.policy_probs[pass_pos as usize]);
    }

    let policy_sum: f32 = legal_probs.iter().sum();
    assert!(
        (policy_sum - 1.0).abs() < 1e-4,
        "legal move policy should sum to 1, got {policy_sum}"
    );
    for &p in &legal_probs {
        assert!(
            (0.0..=1.0).contains(&p),
            "policy probability {p} out of range"
        );
    }

    // Value probabilities are a valid distribution.
    let value_sum = output.white_win_prob + output.white_loss_prob + output.white_no_result_prob;
    assert!(
        (value_sum - 1.0).abs() < 1e-4,
        "value probabilities should sum to 1, got {value_sum}"
    );

    // Ownership map is present and within [-1, 1].
    let owner_map = output
        .white_owner_map
        .as_ref()
        .expect("owner map should be present");
    assert_eq!(
        owner_map.len(),
        (nn_eval.nn_x_len() * nn_eval.nn_y_len()) as usize
    );
    for &v in owner_map.iter() {
        assert!(
            (-1.0..=1.0).contains(&v),
            "ownership value {v} out of range"
        );
    }

    // Spot-check that we can look up specific locations.
    if board.x_size >= 5 && board.y_size >= 16 {
        let e16 = location::of_string("E16", board.x_size, board.y_size).expect("valid loc");
        let e16_pos = nn_pos::loc_to_pos(e16, board.x_size, nn_eval.nn_x_len(), nn_eval.nn_y_len());
        assert!((-1.0..=1.0).contains(&output.policy_probs[e16_pos as usize]));
    }

    let pass_pos = nn_pos::loc_to_pos(
        kata_game::board::PASS_LOC,
        board.x_size,
        nn_eval.nn_x_len(),
        nn_eval.nn_y_len(),
    );
    assert!((-1.0..=1.0).contains(&output.policy_probs[pass_pos as usize]));
}

#[test]
fn canary_position_18_turns_smoke() {
    let nn_eval = make_dummy_eval(19, 19, false);
    let sgf = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Chinese]SZ[19]KM[7]PW[White]PB[Black];B[pd];W[pp];B[dd];W[dp];B[qn];W[nq];B[cq];W[dq];B[cp];W[do];B[bn];W[cc];B[cd];W[dc];B[ec];W[eb];B[fb];W[fc];B[ed];W[gb];B[db];W[fa];B[cb];W[qo];B[pn];W[nc];B[qj];W[qc];B[qd];W[pc];B[od];W[nd];B[ne];W[me];B[mf];W[nf])";
    for sym in 0..8 {
        evaluate_position(&nn_eval, sgf, 18, sym, None);
    }
}

#[test]
fn canary_position_36_turns_smoke() {
    let nn_eval = make_dummy_eval(19, 19, false);
    let sgf = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Chinese]SZ[19]KM[7]PW[White]PB[Black];B[pd];W[pp];B[dd];W[dp];B[qn];W[nq];B[cq];W[dq];B[cp];W[do];B[bn];W[cc];B[cd];W[dc];B[ec];W[eb];B[fb];W[fc];B[ed];W[gb];B[db];W[fa];B[cb];W[qo];B[pn];W[nc];B[qj];W[qc];B[qd];W[pc];B[od];W[nd];B[ne];W[me];B[mf];W[nf])";
    for sym in 0..8 {
        evaluate_position(&nn_eval, sgf, 36, sym, None);
    }
}

#[test]
fn canary_position_23_turns_smoke() {
    let nn_eval = make_dummy_eval(19, 19, false);
    let sgf = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Chinese]SZ[19]KM[7]PW[White]PB[Black];B[qd];W[dd];B[pp];W[dp];B[cf];W[fc];B[nd];W[nq];B[cq];W[dq];B[cp];W[cn];B[co];W[do];B[bn];W[cm];B[bm];W[cl];B[qn];W[pq];B[qq];W[qr];B[oq])";
    for sym in 0..8 {
        evaluate_position(&nn_eval, sgf, 23, sym, None);
    }
}

#[test]
fn canary_position_23_turns_negative_komi_smoke() {
    let nn_eval = make_dummy_eval(19, 19, false);
    let sgf = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Chinese]SZ[19]KM[7]PW[White]PB[Black];B[qd];W[dd];B[pp];W[dp];B[cf];W[fc];B[nd];W[nq];B[cq];W[dq];B[cp];W[cn];B[co];W[do];B[bn];W[cm];B[bm];W[cl];B[qn];W[pq];B[qq];W[qr];B[oq])";
    for sym in 0..8 {
        evaluate_position(&nn_eval, sgf, 23, sym, Some(-7.0));
    }
}

#[test]
fn canary_position_23_turns_large_komi_smoke() {
    let nn_eval = make_dummy_eval(19, 19, false);
    let sgf = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Chinese]SZ[19]KM[7]PW[White]PB[Black];B[qd];W[dd];B[pp];W[dp];B[cf];W[fc];B[nd];W[nq];B[cq];W[dq];B[cp];W[cn];B[co];W[do];B[bn];W[cm];B[bm];W[cl];B[qn];W[pq];B[qq];W[qr];B[oq])";
    for sym in 0..8 {
        evaluate_position(&nn_eval, sgf, 23, sym, Some(21.0));
    }
}

#[test]
fn canary_rectangular_board_smoke() {
    // Evaluator does not require exact NN length, so rectangular board is allowed.
    let nn_eval = make_dummy_eval(19, 19, false);
    let sgf = "(;FF[4]GM[1]CA[UTF-8]RU[Japanese]KM[6]SZ[16:11];B[md];W[nh];B[dh];W[cd];B[lh];W[li];B[ki])";
    for sym in 0..8 {
        evaluate_position(&nn_eval, sgf, 7, sym, None);
    }
}

#[test]
fn canary_require_exact_nn_len_smoke() {
    // When exact NN length is required, rectangular board should still work if it
    // happens to match one dimension; here we just verify the evaluator with
    // require_exact_nn_len = true still runs on a 19x19 position.
    let nn_eval = make_dummy_eval(19, 19, true);
    let sgf = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Chinese]SZ[19]KM[7]PW[White]PB[Black];B[pd];W[pp];B[dd];W[dp];B[qn];W[nq];B[cq];W[dq];B[cp];W[do];B[bn];W[cc];B[cd];W[dc];B[ec];W[eb];B[fb];W[fc];B[ed];W[gb];B[db];W[fa];B[cb];W[qo];B[pn];W[nc];B[qj];W[qc];B[qd];W[pc];B[od];W[nd];B[ne];W[me];B[mf];W[nf])";
    evaluate_position(&nn_eval, sgf, 18, 0, None);
}
