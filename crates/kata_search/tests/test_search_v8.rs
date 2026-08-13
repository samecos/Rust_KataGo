//! Integration tests ported from `KataGo/cpp/tests/testsearchv8.cpp`.
//!
//! These tests use the dummy neural-net evaluator, so the C++ golden output
//! comparisons are skipped in favour of smoke-test assertions that the APIs do
//! not panic and return plausible results.

mod common;

use common::{
    TestSearchOptions, run_bot_on_position, run_bot_on_sgf, test_logger_and_eval,
    test_logger_and_eval_with_options,
};

use std::collections::HashSet;

use kata_game::board::{Board, Loc, P_BLACK, P_WHITE, PASS_LOC, Player, get_opp, location};
use kata_game::history::BoardHistory;
use kata_game::rules::Rules;
use kata_nn::inputs::nn_pos::{MAX_NN_POLICY_SIZE, xy_to_pos};
use kata_search::async_bot::AsyncBot;
use kata_search::params::SearchParams;
use kata_search::search::{PrintTreeOptions, Search};

fn make_v1_params() -> SearchParams {
    let mut params = SearchParams::for_tests_v1();
    params.value_weight_exponent = 0.0;
    params
}

#[test]
fn exact_vs_masked_9x9_smoke() {
    let (_logger, nn_eval) = test_logger_and_eval(19, 19);
    let (_logger9, nn_eval9) = test_logger_and_eval(9, 9);
    let (_logger9_exact, nn_eval9_exact) = test_logger_and_eval_with_options(9, 9, true, 1);

    let sgf = kata_data::sgf::CompactSgf::parse(
        "(;FF[4]GM[1]SZ[9]HA[0]KM[7]RU[stonescoring];B[ef];W[ed];B[ge])",
    )
    .expect("valid SGF");
    let initial_rules = sgf.get_rules_or_fail().expect("rules");
    let mut board = Board::default();
    let mut next_pla = 0;
    let mut hist = BoardHistory::default();
    sgf.setup_board_and_hist_assume_legal(&initial_rules, &mut board, &mut next_pla, &mut hist, 3)
        .expect("setup board");

    let mut params = make_v1_params();
    params.max_visits = 200;

    let mut bot_a = AsyncBot::new(params.clone(), &nn_eval, &_logger, "test exact again");
    let mut bot_b = AsyncBot::new(params.clone(), &nn_eval9, &_logger9, "test exact again");
    let mut bot_c = AsyncBot::new(
        params.clone(),
        &nn_eval9_exact,
        &_logger9_exact,
        "test exact again",
    );

    run_bot_on_position(
        &mut bot_a,
        board.clone(),
        next_pla,
        hist.clone(),
        TestSearchOptions::default(),
    );
    run_bot_on_position(
        &mut bot_b,
        board.clone(),
        next_pla,
        hist.clone(),
        TestSearchOptions::default(),
    );
    run_bot_on_position(
        &mut bot_c,
        board.clone(),
        next_pla,
        hist.clone(),
        TestSearchOptions::default(),
    );
}

#[test]
fn exact_vs_masked_19x19_smoke() {
    let (logger, nn_eval) = test_logger_and_eval_with_options(19, 19, false, -1);
    let (logger_exact, nn_eval_exact) = test_logger_and_eval_with_options(19, 19, true, -1);

    let sgf = kata_data::sgf::CompactSgf::parse(
        "(;GM[1]FF[4]CA[UTF-8]RU[Japanese]SZ[19]KM[6.5];B[dd];W[qd];B[pq];W[dp];B[oc];W[pe];B[fq];W[jp];B[ph];W[cf];B[ck])",
    )
    .expect("valid SGF");
    let initial_rules = sgf.get_rules_or_fail().expect("rules");
    let mut board = Board::default();
    let mut next_pla = 0;
    let mut hist = BoardHistory::default();
    sgf.setup_board_and_hist_assume_legal(&initial_rules, &mut board, &mut next_pla, &mut hist, 11)
        .expect("setup board");

    let mut params = make_v1_params();
    params.max_visits = 200;

    let mut bot_a = AsyncBot::new(params.clone(), &nn_eval, &logger, "test exact");
    let mut bot_b = AsyncBot::new(params.clone(), &nn_eval_exact, &logger_exact, "test exact");

    run_bot_on_position(
        &mut bot_a,
        board.clone(),
        next_pla,
        hist.clone(),
        TestSearchOptions::default(),
    );
    run_bot_on_position(
        &mut bot_b,
        board.clone(),
        next_pla,
        hist.clone(),
        TestSearchOptions::default(),
    );
}

#[test]
fn symmetry_averaging_19x19_smoke() {
    let (logger, nn_eval) = test_logger_and_eval_with_options(19, 19, false, -1);
    let (logger_exact, nn_eval_exact) = test_logger_and_eval_with_options(19, 19, true, -1);

    // Reset server threads to reseed random symmetry selection.

    let sgf = kata_data::sgf::CompactSgf::parse(
        "(;GM[1]FF[4]CA[UTF-8]RU[Japanese]SZ[19]KM[6.5];B[dd];W[qd];B[od];W[pq];B[dq];W[do];B[eo];W[oe])",
    )
    .expect("valid SGF");
    let initial_rules = sgf.get_rules_or_fail().expect("rules");
    let mut board = Board::default();
    let mut next_pla = 0;
    let mut hist = BoardHistory::default();
    sgf.setup_board_and_hist_assume_legal(&initial_rules, &mut board, &mut next_pla, &mut hist, 8)
        .expect("setup board");

    let mut params = make_v1_params();
    params.root_num_symmetries_to_sample = 8;
    params.max_visits = 200;

    let mut bot_a = AsyncBot::new(params.clone(), &nn_eval, &logger, "test exact");
    let mut bot_b = AsyncBot::new(params.clone(), &nn_eval_exact, &logger_exact, "test exact");

    run_bot_on_position(
        &mut bot_a,
        board.clone(),
        next_pla,
        hist.clone(),
        TestSearchOptions::default(),
    );
    run_bot_on_position(
        &mut bot_b,
        board.clone(),
        next_pla,
        hist.clone(),
        TestSearchOptions::default(),
    );
}

#[test]
fn nn_policy_temperature_smoke() {
    let (logger, nn_eval) = test_logger_and_eval_with_options(19, 19, false, -1);
    let (logger_exact, nn_eval_exact) = test_logger_and_eval_with_options(19, 19, true, -1);

    let sgf = kata_data::sgf::CompactSgf::parse(
        "(;GM[1]FF[4]CA[UTF-8]RU[AGA]SZ[19]KM[7.0];B[dd];W[pd];B[dp];W[pp];B[qc];W[qd];B[pc];W[nc];B[nb])",
    )
    .expect("valid SGF");
    let initial_rules = sgf.get_rules_or_fail().expect("rules");
    let mut board = Board::default();
    let mut next_pla = 0;
    let mut hist = BoardHistory::default();
    sgf.setup_board_and_hist_assume_legal(&initial_rules, &mut board, &mut next_pla, &mut hist, 8)
        .expect("setup board");

    let params_a = {
        let mut p = make_v1_params();
        p.max_visits = 200;
        p
    };
    let params_b = {
        let mut p = make_v1_params();
        p.max_visits = 200;
        p.nn_policy_temperature = 1.5;
        p
    };
    let params_c = {
        let mut p = make_v1_params();
        p.max_visits = 200;
        p.nn_policy_temperature = 0.5;
        p
    };

    let mut bot_a = AsyncBot::new(params_a.clone(), &nn_eval, &logger, "test exact");
    let mut bot_b = AsyncBot::new(params_b.clone(), &nn_eval, &logger, "test exact");
    let mut bot_c = AsyncBot::new(params_c.clone(), &nn_eval, &logger, "test exact");
    let mut bot_a2 = AsyncBot::new(
        params_a.clone(),
        &nn_eval_exact,
        &logger_exact,
        "test exact",
    );
    let mut bot_b2 = AsyncBot::new(
        params_b.clone(),
        &nn_eval_exact,
        &logger_exact,
        "test exact",
    );
    let mut bot_c2 = AsyncBot::new(
        params_c.clone(),
        &nn_eval_exact,
        &logger_exact,
        "test exact",
    );

    let opts = TestSearchOptions {
        print_more: true,
        ..Default::default()
    };

    nn_eval.clear_cache();
    nn_eval.clear_stats();

    run_bot_on_position(&mut bot_a, board.clone(), next_pla, hist.clone(), opts);
    run_bot_on_position(&mut bot_a2, board.clone(), next_pla, hist.clone(), opts);

    run_bot_on_position(&mut bot_b, board.clone(), next_pla, hist.clone(), opts);
    run_bot_on_position(&mut bot_b2, board.clone(), next_pla, hist.clone(), opts);

    run_bot_on_position(&mut bot_c, board.clone(), next_pla, hist.clone(), opts);
    run_bot_on_position(&mut bot_c2, board.clone(), next_pla, hist.clone(), opts);
}

fn assert_move_legal_or_pass(board: &Board, hist: &BoardHistory, loc: Loc, pla: Player) {
    assert!(loc == PASS_LOC || hist.is_legal(board, loc, pla));
}

fn print_search_smoke(
    search: &mut Search<'_>,
    board: &Board,
    hist: &BoardHistory,
    next_pla: Player,
) -> Loc {
    assert!(search.get_root_visits() > 0);
    let move_loc = search.get_chosen_move_loc();
    assert_move_legal_or_pass(board, hist, move_loc, next_pla);

    let options = PrintTreeOptions::new().max_depth(1);
    let root = search.root_node.as_deref().expect("root node should exist");
    let mut out = String::new();
    search.print_tree(&mut out, Some(root), &options, P_WHITE);
    assert!(!out.is_empty());
    move_loc
}

#[test]
fn pda_and_pondering_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);

    let board = Board::parse_board(
        13,
        13,
        ".............
.............
.............
.........x...
.............
.............
.............
.............
.............
..o......x...
.............
.............
.............
",
        '\n',
    )
    .expect("valid board");
    let start_pla = P_WHITE;
    let rules = Rules::get_tromp_taylorish();
    let hist = BoardHistory::new(board.clone(), start_pla, rules, 0);

    let base_params = {
        let mut p = make_v1_params();
        p.max_visits = 400;
        p.max_visits_pondering = 600;
        p.max_playouts = 200;
        p.max_playouts_pondering = 300;
        p
    };

    // Basic search with PDA 1.5, no player.
    {
        let mut params = base_params.clone();
        params.playout_doubling_advantage = 1.5;
        let mut search = Search::new(params, &nn_eval, &logger, "autoSearchRandSeed3");
        let mut board = board.clone();
        let mut hist = hist.clone();
        let mut next_pla = start_pla;
        search.set_position(next_pla, &board, &hist);

        let mv = search.run_whole_search_and_get_move(next_pla);
        print_search_smoke(&mut search, &board, &hist, next_pla);

        search.make_move(mv, next_pla);
        hist.make_board_move_assume_legal(&mut board, mv, next_pla);
        next_pla = get_opp(next_pla);
        let mv = search.run_whole_search_and_get_move(next_pla);
        print_search_smoke(&mut search, &board, &hist, next_pla);

        search.make_move(mv, next_pla);
        hist.make_board_move_assume_legal(&mut board, mv, next_pla);
        next_pla = get_opp(next_pla);
        search.run_whole_search(next_pla);
        print_search_smoke(&mut search, &board, &hist, next_pla);
    }

    // Basic search with PDA 1.5, force black.
    {
        let mut params = base_params.clone();
        params.playout_doubling_advantage = 1.5;
        params.playout_doubling_advantage_pla = P_BLACK;
        let mut search = Search::new(params, &nn_eval, &logger, "autoSearchRandSeed3");
        let mut board = board.clone();
        let mut hist = hist.clone();
        let mut next_pla = start_pla;
        search.set_position(next_pla, &board, &hist);

        let mut mv = search.run_whole_search_and_get_move(next_pla);
        print_search_smoke(&mut search, &board, &hist, next_pla);

        search.make_move(mv, next_pla);
        hist.make_board_move_assume_legal(&mut board, mv, next_pla);
        next_pla = get_opp(next_pla);
        mv = search.run_whole_search_and_get_move(next_pla);
        print_search_smoke(&mut search, &board, &hist, next_pla);

        search.make_move(mv, next_pla);
        hist.make_board_move_assume_legal(&mut board, mv, next_pla);
        next_pla = get_opp(next_pla);
        mv = search.run_whole_search_and_get_move(next_pla);
        print_search_smoke(&mut search, &board, &hist, next_pla);

        search.make_move(mv, next_pla);
        hist.make_board_move_assume_legal(&mut board, mv, next_pla);
        next_pla = get_opp(next_pla);
        mv = search.run_whole_search_and_get_move_pondering(next_pla, true);
        print_search_smoke(&mut search, &board, &hist, next_pla);

        search.make_move(mv, next_pla);
        hist.make_board_move_assume_legal(&mut board, mv, next_pla);
        next_pla = get_opp(next_pla);
        search.run_whole_search(next_pla);
        print_search_smoke(&mut search, &board, &hist, next_pla);
    }

    // Basic search with PDA 1.5, force white.
    {
        let mut params = base_params.clone();
        params.playout_doubling_advantage = 1.5;
        params.playout_doubling_advantage_pla = P_WHITE;
        let mut search = Search::new(params, &nn_eval, &logger, "autoSearchRandSeed3");
        let mut board = board.clone();
        let mut hist = hist.clone();
        let mut next_pla = start_pla;
        search.set_position(next_pla, &board, &hist);

        let mut mv = search.run_whole_search_and_get_move(next_pla);
        print_search_smoke(&mut search, &board, &hist, next_pla);

        search.make_move(mv, next_pla);
        hist.make_board_move_assume_legal(&mut board, mv, next_pla);
        next_pla = get_opp(next_pla);
        mv = search.run_whole_search_and_get_move(next_pla);
        print_search_smoke(&mut search, &board, &hist, next_pla);

        search.make_move(mv, next_pla);
        hist.make_board_move_assume_legal(&mut board, mv, next_pla);
        next_pla = get_opp(next_pla);
        mv = search.run_whole_search_and_get_move(next_pla);
        print_search_smoke(&mut search, &board, &hist, next_pla);

        search.make_move(mv, next_pla);
        hist.make_board_move_assume_legal(&mut board, mv, next_pla);
        next_pla = get_opp(next_pla);
        mv = search.run_whole_search_and_get_move_pondering(next_pla, true);
        print_search_smoke(&mut search, &board, &hist, next_pla);

        search.make_move(mv, next_pla);
        hist.make_board_move_assume_legal(&mut board, mv, next_pla);
        next_pla = get_opp(next_pla);
        mv = search.run_whole_search_and_get_move_pondering(next_pla, true);
        print_search_smoke(&mut search, &board, &hist, next_pla);

        search.make_move(mv, next_pla);
        hist.make_board_move_assume_legal(&mut board, mv, next_pla);
        next_pla = get_opp(next_pla);
        search.run_whole_search(next_pla);
        print_search_smoke(&mut search, &board, &hist, next_pla);
    }

    // Pondering keeps prior tree and PDA.
    {
        let mut params = base_params.clone();
        params.playout_doubling_advantage = 1.5;
        let mut search = Search::new(params, &nn_eval, &logger, "autoSearchRandSeed3");
        let mut board = board.clone();
        let mut hist = hist.clone();
        let mut next_pla = start_pla;
        search.set_position(next_pla, &board, &hist);

        let mut mv = search.run_whole_search_and_get_move(next_pla);
        print_search_smoke(&mut search, &board, &hist, next_pla);

        search.make_move(mv, next_pla);
        hist.make_board_move_assume_legal(&mut board, mv, next_pla);
        next_pla = get_opp(next_pla);
        mv = search.run_whole_search_and_get_move_pondering(next_pla, true);
        print_search_smoke(&mut search, &board, &hist, next_pla);

        search.make_move(mv, next_pla);
        hist.make_board_move_assume_legal(&mut board, mv, next_pla);
        next_pla = get_opp(next_pla);
        search.run_whole_search(next_pla);
        print_search_smoke(&mut search, &board, &hist, next_pla);
    }

    // Two ponderings in a row, then regular search loses tree because player differs.
    {
        let mut params = base_params.clone();
        params.playout_doubling_advantage = 1.5;
        let mut search = Search::new(params, &nn_eval, &logger, "autoSearchRandSeed3");
        let mut board = board.clone();
        let mut hist = hist.clone();
        let mut next_pla = start_pla;
        search.set_position(next_pla, &board, &hist);

        let mut mv = search.run_whole_search_and_get_move(next_pla);
        print_search_smoke(&mut search, &board, &hist, next_pla);

        search.make_move(mv, next_pla);
        hist.make_board_move_assume_legal(&mut board, mv, next_pla);
        next_pla = get_opp(next_pla);
        mv = search.run_whole_search_and_get_move_pondering(next_pla, true);
        print_search_smoke(&mut search, &board, &hist, next_pla);

        search.make_move(mv, next_pla);
        hist.make_board_move_assume_legal(&mut board, mv, next_pla);
        next_pla = get_opp(next_pla);
        mv = search.run_whole_search_and_get_move_pondering(next_pla, true);
        print_search_smoke(&mut search, &board, &hist, next_pla);

        search.make_move(mv, next_pla);
        hist.make_board_move_assume_legal(&mut board, mv, next_pla);
        next_pla = get_opp(next_pla);
        search.run_whole_search(next_pla);
        print_search_smoke(&mut search, &board, &hist, next_pla);
    }

    // Convert ponder to regular search without a move, then set position and ponder again.
    {
        let mut params = base_params.clone();
        params.playout_doubling_advantage = 1.5;
        let mut search = Search::new(params, &nn_eval, &logger, "autoSearchRandSeed3");
        let mut board = board.clone();
        let mut hist = hist.clone();
        let mut next_pla = start_pla;
        search.set_position(next_pla, &board, &hist);

        let mut mv = search.run_whole_search_and_get_move(next_pla);
        print_search_smoke(&mut search, &board, &hist, next_pla);

        search.make_move(mv, next_pla);
        hist.make_board_move_assume_legal(&mut board, mv, next_pla);
        next_pla = get_opp(next_pla);
        mv = search.run_whole_search_and_get_move_pondering(next_pla, true);
        print_search_smoke(&mut search, &board, &hist, next_pla);

        search.make_move(mv, next_pla);
        hist.make_board_move_assume_legal(&mut board, mv, next_pla);
        next_pla = get_opp(next_pla);
        search.run_whole_search_pondering(next_pla, true);
        print_search_smoke(&mut search, &board, &hist, next_pla);

        search.run_whole_search(next_pla);
        print_search_smoke(&mut search, &board, &hist, next_pla);

        nn_eval.clear_cache();
        nn_eval.clear_stats();

        search.set_position(start_pla, &board, &hist);
        next_pla = start_pla;
        search.run_whole_search_and_get_move_pondering(next_pla, true);
        print_search_smoke(&mut search, &board, &hist, next_pla);

        mv = search.run_whole_search_and_get_move(next_pla);
        print_search_smoke(&mut search, &board, &hist, next_pla);

        search.make_move(mv, next_pla);
        hist.make_board_move_assume_legal(&mut board, mv, next_pla);
        next_pla = get_opp(next_pla);
        let d4 = location::of_string("D4", board.x_size, board.y_size).expect("valid loc");
        search.make_move(d4, next_pla);
        hist.make_board_move_assume_legal(&mut board, d4, next_pla);
        next_pla = get_opp(next_pla);
        search.run_whole_search(next_pla);
        print_search_smoke(&mut search, &board, &hist, next_pla);

        nn_eval.clear_cache();
        nn_eval.clear_stats();

        search.set_position(start_pla, &board, &hist);
        next_pla = start_pla;
        mv = search.run_whole_search_and_get_move_pondering(next_pla, true);
        print_search_smoke(&mut search, &board, &hist, next_pla);

        search.make_move(mv, next_pla);
        hist.make_board_move_assume_legal(&mut board, mv, next_pla);
        next_pla = get_opp(next_pla);
        search.run_whole_search(next_pla);
        print_search_smoke(&mut search, &board, &hist, next_pla);
    }
}

#[test]
fn hintloc_t16_o18_19x19_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);

    let board = Board::parse_board(
        19,
        19,
        "...................
............o.oxx..
...x..........ooxo.
...........o..xxo..
..x........oxxxoo.x
..........xxo.oxxo.
............o.o....
..............oxx..
...................
...............x...
..o................
...................
...................
...................
...................
..o.o.........ooo..
.........x...xoxx..
............x.xoo..
.............x.....
",
        '\n',
    )
    .expect("valid board");

    let next_pla = P_BLACK;
    let rules = Rules::parse_rules("Chinese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 400;
    params.root_noise_enabled = true;
    params.root_policy_temperature = 1.2;
    params.root_policy_temperature_early = 1.2;
    params.root_num_symmetries_to_sample = 2;

    {
        let mut bot = AsyncBot::new(params.clone(), &nn_eval, &logger, "hintloc");
        bot.set_root_hint_loc(
            location::of_string("T16", board.x_size, board.y_size).expect("valid loc"),
        );
        run_bot_on_position(
            &mut bot,
            board.clone(),
            next_pla,
            hist.clone(),
            TestSearchOptions::default(),
        );
    }

    {
        let mut bot = AsyncBot::new(params.clone(), &nn_eval, &logger, "hintloc");
        bot.set_root_hint_loc(
            location::of_string("O18", board.x_size, board.y_size).expect("valid loc"),
        );
        run_bot_on_position(
            &mut bot,
            board.clone(),
            next_pla,
            hist.clone(),
            TestSearchOptions::default(),
        );
    }
}

#[test]
fn anti_mirror_white_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);

    let sgf_str = "(;KM[7.5]SZ[19];B[pd];W[dp];B[pp];W[dd];B[cc];W[qq];B[dc];W[pq];B[op];W[ed];B[qp];W[cd];B[ec];W[oq];B[nq];W[fc];B[mp];W[gd];B[rp];W[bd];B[fq];W[nc];B[pi];W[dk];B[fe];W[no];B[cq];W[qc];B[pc];W[dq];B[cp];W[qd];B[do];W[pe];B[oe];W[eo];B[en];W[of];B[nf];W[fn];B[fd];W[np];B[mo];W[ge];B[fo];W[ne];B[od];W[ep];B[gn];W[mf];B[ng];W[fm];B[dn];W[pf];B[ff];W[nn];B[nd];W[fp];B[go];W[me];B[mr];W[gb];B[md];W[gp];B[gm];W[mg];B[mh];W[gl];B[hp];W[ld];B[lc];W[hq];B[fl];W[nh];B[gq];W[mc];B[pb];W[dr];B[hr];W[lb];B[kc];W[iq];B[ir];W[kb];B[hc];W[lq];B[og];W[em];B[hl];W[lh];B[mi];W[gk];B[le];W[ho];B[in];W[kf];B[jq];W[jc];B[pg];W[dm];B[kd];W[ip];B[mq];W[gc];B[bn];W[rf];B[cm];W[qg];B[qh];W[cl];B[rg];W[bm];B[cn];W[qf];B[li];W[hk];B[il];W[kh];B[rh];W[bl];B[bo];W[re];B[ik];W[ki];B[kj];W[ij];B[jj])";
    let default_rules = Rules::get_tromp_taylorish();

    let mut params = make_v1_params();
    params.max_visits = 200;
    params.anti_mirror = true;

    for turn_idx in [24, 32, 124] {
        let mut bot = AsyncBot::new(params.clone(), &nn_eval, &logger, "antimirrortest");
        run_bot_on_sgf(
            &mut bot,
            sgf_str,
            &default_rules,
            turn_idx,
            7.5,
            TestSearchOptions::default(),
        );
    }
}

#[test]
fn anti_mirror_black_negkomi_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);

    let sgf_str = "(;SZ[19]KM[-3.50];B[jj];W[pd];B[dp];W[dd];B[pp];W[cn];B[qf];W[nq];B[fc];W[qn];B[cf];W[df];B[pn];W[pm];B[dg];W[po];B[de];W[fd];B[np];W[mp];B[gd];W[ed];B[op];W[on];B[ef];W[gc];B[mq];W[lq];B[hc];W[jk];B[ji];W[ik];B[ki];W[ij];B[kj];W[gb];B[mr];W[ic];B[kq];W[pf];B[dn];W[do];B[pe];W[qe];B[co];W[lp];B[hd];W[eo];B[oe];W[qg];B[cm];W[bn];B[rf];W[qd];B[cp];W[hb];B[lr];W[bm];B[rg];W[of];B[en];W[fn];B[nf];W[qh];B[cl];W[ck];B[qi];W[ne];B[fo];W[ep];B[od];W[ng];B[fm];W[gn];B[mf];W[mg];B[gm];W[lf];B[hn];W[rh];B[bl];W[me];B[go];W[ii];B[kk];W[kl])";
    let default_rules = Rules::get_tromp_taylorish();

    let mut params = make_v1_params();
    params.max_visits = 200;
    params.anti_mirror = true;

    for turn_idx in [29, 83] {
        let mut bot = AsyncBot::new(params.clone(), &nn_eval, &logger, "antimirrortest");
        run_bot_on_sgf(
            &mut bot,
            sgf_str,
            &default_rules,
            turn_idx,
            -3.5,
            TestSearchOptions::default(),
        );
    }
}

#[test]
fn value_bias_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(9, 9);

    let sgf = kata_data::sgf::CompactSgf::parse(
        "(;GM[1]FF[4]CA[UTF-8]RU[Japanese]SZ[9]KM[0];B[dc];W[ef];B[df];W[de];B[dg];W[eg];B[eh];W[fh];B[ee])",
    )
    .expect("valid SGF");
    let initial_rules = sgf.get_rules_or_fail().expect("rules");
    let mut board = Board::default();
    let mut next_pla = 0;
    let mut hist = BoardHistory::default();
    sgf.setup_board_and_hist_assume_legal(&initial_rules, &mut board, &mut next_pla, &mut hist, 8)
        .expect("setup board");

    let mut params_a = make_v1_params();
    params_a.max_visits = 20;
    params_a.chosen_move_temperature = 0.0;

    let mut params_b = params_a.clone();
    params_b.subtree_value_bias_factor = 0.5;

    let mut params_c = params_b.clone();
    params_c.max_visits = 300;

    let mut bot_a = AsyncBot::new(params_a, &nn_eval, &logger, "valuebias test");
    let mut bot_b = AsyncBot::new(params_b, &nn_eval, &logger, "valuebias test");
    let mut bot_c = AsyncBot::new(params_c, &nn_eval, &logger, "valuebias test");

    let opts = TestSearchOptions {
        print_more_more_more: true,
        num_moves_in_a_row: 3,
        print_after_begun: true,
        ..Default::default()
    };
    run_bot_on_position(&mut bot_a, board.clone(), next_pla, hist.clone(), opts);
    run_bot_on_position(&mut bot_b, board.clone(), next_pla, hist.clone(), opts);

    let opts_c = TestSearchOptions {
        num_moves_in_a_row: 3,
        print_after_begun: true,
        ..Default::default()
    };
    run_bot_on_position(&mut bot_c, board, next_pla, hist, opts_c);
}

fn ending_bonus_points_11x11_board() -> Board {
    Board::parse_board(
        11,
        11,
        ".o.xox.x.o.
xxxxoxxooox
oo.ooxxxxxx
.oo.ox.xooo
oooooxxo.o.
xxxoooxxooo
.x.xoxxxxxx
xxxooox.xx.
oooo.oxx.xx
oxxxooxoooo
.x.o.oxo.x.
",
        '\n',
    )
    .expect("valid board")
}

#[test]
fn ending_bonus_points_area_white_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(11, 11);
    let board = ending_bonus_points_11x11_board();

    let mut params = make_v1_params();
    params.max_visits = 300;
    params.fpu_reduction_max = 0.0;
    params.root_fpu_reduction_max = 0.0;
    params.root_ending_bonus_points = 0.0;
    params.root_policy_temperature = 1.5;
    params.root_policy_temperature_early = 1.5;

    let mut params2 = params.clone();
    params2.root_ending_bonus_points = 0.5;

    let opts = TestSearchOptions {
        print_ending_score_value_bonus: true,
        ..Default::default()
    };

    let next_pla = P_WHITE;
    let rules = Rules::parse_rules("Chinese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut bot_a = AsyncBot::new(
        params,
        &nn_eval,
        &logger,
        "async bot ending bonus points seed",
    );
    run_bot_on_position(&mut bot_a, board.clone(), next_pla, hist.clone(), opts);

    let mut bot_b = AsyncBot::new(
        params2,
        &nn_eval,
        &logger,
        "async bot ending bonus points seed",
    );
    run_bot_on_position(&mut bot_b, board.clone(), next_pla, hist, opts);
}

#[test]
fn ending_bonus_points_area_white_button_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(11, 11);
    let board = ending_bonus_points_11x11_board();

    let mut params = make_v1_params();
    params.max_visits = 300;
    params.fpu_reduction_max = 0.0;
    params.root_fpu_reduction_max = 0.0;
    params.root_ending_bonus_points = 0.0;
    params.root_policy_temperature = 1.5;
    params.root_policy_temperature_early = 1.5;

    let mut params2 = params.clone();
    params2.root_ending_bonus_points = 0.5;

    let opts = TestSearchOptions {
        print_ending_score_value_bonus: true,
        ..Default::default()
    };

    let next_pla = P_WHITE;
    let mut rules = Rules::parse_rules("Chinese").expect("valid rules");
    rules.has_button = true;
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut bot_a = AsyncBot::new(
        params,
        &nn_eval,
        &logger,
        "async bot ending bonus points seed",
    );
    run_bot_on_position(&mut bot_a, board.clone(), next_pla, hist.clone(), opts);

    let mut bot_b = AsyncBot::new(
        params2,
        &nn_eval,
        &logger,
        "async bot ending bonus points seed",
    );
    run_bot_on_position(&mut bot_b, board, next_pla, hist, opts);
}

#[test]
fn ending_bonus_points_area_black_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(11, 11);
    let board = ending_bonus_points_11x11_board();

    let mut params = make_v1_params();
    params.max_visits = 300;
    params.fpu_reduction_max = 0.0;
    params.root_fpu_reduction_max = 0.0;
    params.root_ending_bonus_points = 0.0;
    params.root_policy_temperature = 1.5;
    params.root_policy_temperature_early = 1.5;

    let mut params2 = params.clone();
    params2.root_ending_bonus_points = 0.5;

    let opts = TestSearchOptions {
        print_ending_score_value_bonus: true,
        ..Default::default()
    };

    let next_pla = P_BLACK;
    let rules = Rules::parse_rules("Chinese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut bot_a = AsyncBot::new(
        params,
        &nn_eval,
        &logger,
        "async bot ending bonus points seed",
    );
    run_bot_on_position(&mut bot_a, board.clone(), next_pla, hist.clone(), opts);

    let mut bot_b = AsyncBot::new(
        params2,
        &nn_eval,
        &logger,
        "async bot ending bonus points seed",
    );
    run_bot_on_position(&mut bot_b, board, next_pla, hist, opts);
}

#[test]
fn ending_bonus_points_area_black_button_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(11, 11);
    let board = ending_bonus_points_11x11_board();

    let mut params = make_v1_params();
    params.max_visits = 300;
    params.fpu_reduction_max = 0.0;
    params.root_fpu_reduction_max = 0.0;
    params.root_ending_bonus_points = 0.0;
    params.root_policy_temperature = 1.5;
    params.root_policy_temperature_early = 1.5;

    let mut params2 = params.clone();
    params2.root_ending_bonus_points = 0.5;

    let opts = TestSearchOptions {
        print_ending_score_value_bonus: true,
        ..Default::default()
    };

    let next_pla = P_BLACK;
    let mut rules = Rules::parse_rules("Chinese").expect("valid rules");
    rules.has_button = true;
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut bot_a = AsyncBot::new(
        params,
        &nn_eval,
        &logger,
        "async bot ending bonus points seed",
    );
    run_bot_on_position(&mut bot_a, board.clone(), next_pla, hist.clone(), opts);

    let mut bot_b = AsyncBot::new(
        params2,
        &nn_eval,
        &logger,
        "async bot ending bonus points seed",
    );
    run_bot_on_position(&mut bot_b, board, next_pla, hist, opts);
}

#[test]
fn ending_bonus_points_territory_black_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(11, 11);
    let board = ending_bonus_points_11x11_board();

    let mut params = make_v1_params();
    params.max_visits = 300;
    params.fpu_reduction_max = 0.0;
    params.root_fpu_reduction_max = 0.0;
    params.root_ending_bonus_points = 0.0;
    params.root_policy_temperature = 1.5;
    params.root_policy_temperature_early = 1.5;

    let mut params2 = params.clone();
    params2.root_ending_bonus_points = 0.5;

    let opts = TestSearchOptions {
        print_ending_score_value_bonus: true,
        ..Default::default()
    };

    let next_pla = P_BLACK;
    let rules = Rules::parse_rules("Japanese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut bot_a = AsyncBot::new(
        params,
        &nn_eval,
        &logger,
        "async bot ending bonus points seed",
    );
    run_bot_on_position(&mut bot_a, board.clone(), next_pla, hist.clone(), opts);

    let mut bot_b = AsyncBot::new(
        params2,
        &nn_eval,
        &logger,
        "async bot ending bonus points seed",
    );
    run_bot_on_position(&mut bot_b, board, next_pla, hist, opts);
}

#[test]
fn ending_bonus_points_territory_black_encore2_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(11, 11);
    let board = ending_bonus_points_11x11_board();

    let mut params = make_v1_params();
    params.max_visits = 300;
    params.fpu_reduction_max = 0.0;
    params.root_fpu_reduction_max = 0.0;
    params.root_ending_bonus_points = 0.0;
    params.root_policy_temperature = 1.5;
    params.root_policy_temperature_early = 1.5;

    let mut params2 = params.clone();
    params2.root_ending_bonus_points = 0.5;

    let opts = TestSearchOptions {
        print_ending_score_value_bonus: true,
        ..Default::default()
    };

    let next_pla = P_BLACK;
    let rules = Rules::parse_rules("Japanese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 2);

    let mut bot_a = AsyncBot::new(
        params,
        &nn_eval,
        &logger,
        "async bot ending bonus points seed",
    );
    run_bot_on_position(&mut bot_a, board.clone(), next_pla, hist.clone(), opts);

    let mut bot_b = AsyncBot::new(
        params2,
        &nn_eval,
        &logger,
        "async bot ending bonus points seed",
    );
    run_bot_on_position(&mut bot_b, board, next_pla, hist, opts);
}

#[test]
fn ending_bonus_points_fancy_position_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(11, 11);
    let board = Board::parse_board(
        11,
        11,
        ".x..ox.oxo.
xxxooxxox.o
oooox.xoxxx
xxooo.ooooo
x.xoooo..x.
...oxxxox.x
o.oox.xooxo
ooox..xoooo
xxxx.xxoxxx
.....xoox.o
.....xo.xo.
",
        '\n',
    )
    .expect("valid board");

    let mut params = make_v1_params();
    params.max_visits = 300;
    params.fpu_reduction_max = 0.0;
    params.root_fpu_reduction_max = 0.0;
    params.root_ending_bonus_points = 0.5;
    params.root_policy_temperature = 1.5;
    params.root_policy_temperature_early = 1.5;

    let opts = TestSearchOptions {
        print_ending_score_value_bonus: true,
        ..Default::default()
    };

    let next_pla = P_WHITE;
    let rules = Rules::parse_rules("Chinese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut bot = AsyncBot::new(
        params,
        &nn_eval,
        &logger,
        "async bot ending bonus points seed",
    );
    run_bot_on_position(&mut bot, board, next_pla, hist, opts);
}

fn futile_visits_9x9_board() -> Board {
    Board::parse_board(
        9,
        9,
        ".........
.........
.ox..xo..
.........
..o...x..
.........
..ox..ox.
.........
.........
",
        '\n',
    )
    .expect("valid board")
}

#[test]
fn futile_visits_threshold_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(9, 9);
    let board = futile_visits_9x9_board();

    let next_pla = P_BLACK;
    let rules = Rules::get_tromp_taylorish();
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params_a = make_v1_params();
    params_a.max_visits = 400;

    let mut params_b = params_a.clone();
    params_b.futile_visits_threshold = 0.15;

    let mut params_c = params_b.clone();
    params_c.futile_visits_threshold = 0.4;

    let mut bot_a = AsyncBot::new(params_a, &nn_eval, &logger, "futileVisitsThreshold test");
    let mut bot_b = AsyncBot::new(params_b, &nn_eval, &logger, "futileVisitsThreshold test");
    let mut bot_c = AsyncBot::new(params_c, &nn_eval, &logger, "futileVisitsThreshold test");

    run_bot_on_position(
        &mut bot_a,
        board.clone(),
        next_pla,
        hist.clone(),
        TestSearchOptions::default(),
    );
    run_bot_on_position(
        &mut bot_b,
        board.clone(),
        next_pla,
        hist.clone(),
        TestSearchOptions::default(),
    );
    run_bot_on_position(
        &mut bot_c,
        board,
        next_pla,
        hist,
        TestSearchOptions::default(),
    );
}

#[test]
fn futile_visits_threshold_with_playouts_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(9, 9);
    let board = futile_visits_9x9_board();

    let next_pla = P_BLACK;
    let rules = Rules::get_tromp_taylorish();
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params_a = make_v1_params();
    params_a.max_visits = 10000;
    params_a.max_playouts = 200;

    let mut params_b = params_a.clone();
    params_b.futile_visits_threshold = 0.15;

    let mut params_c = params_b.clone();
    params_c.futile_visits_threshold = 0.4;

    let mut bot_a = AsyncBot::new(params_a, &nn_eval, &logger, "futileVisitsThreshold test");
    let mut bot_b = AsyncBot::new(params_b, &nn_eval, &logger, "futileVisitsThreshold test");
    let mut bot_c = AsyncBot::new(params_c, &nn_eval, &logger, "futileVisitsThreshold test");

    run_bot_on_position(
        &mut bot_a,
        board.clone(),
        next_pla,
        hist.clone(),
        TestSearchOptions::default(),
    );
    run_bot_on_position(
        &mut bot_b,
        board.clone(),
        next_pla,
        hist.clone(),
        TestSearchOptions::default(),
    );
    run_bot_on_position(
        &mut bot_c,
        board,
        next_pla,
        hist,
        TestSearchOptions::default(),
    );
}

fn hintloc_c1_19x19_board() -> Board {
    Board::parse_board(
        19,
        19,
        "...................
...................
.................o.
...x...........x...
...................
...................
...................
...................
...................
...................
...................
...................
...................
...o............x..
...................
.............oo.x..
...o.........oxx...
...................
...................
",
        '\n',
    )
    .expect("valid board")
}

#[test]
fn hintloc_c1_base_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);
    let board = hintloc_c1_19x19_board();
    let next_pla = P_BLACK;
    let rules = Rules::parse_rules("Chinese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 200;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "hintloc");
    bot.set_root_hint_loc(
        location::of_string("C1", board.x_size, board.y_size).expect("valid loc"),
    );
    run_bot_on_position(
        &mut bot,
        board,
        next_pla,
        hist,
        TestSearchOptions::default(),
    );
}

#[test]
fn hintloc_c1_again_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);
    let board = hintloc_c1_19x19_board();
    let next_pla = P_BLACK;
    let rules = Rules::parse_rules("Chinese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 200;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "hintloc");
    bot.set_root_hint_loc(
        location::of_string("C1", board.x_size, board.y_size).expect("valid loc"),
    );
    run_bot_on_position(
        &mut bot,
        board,
        next_pla,
        hist,
        TestSearchOptions::default(),
    );
}

#[test]
fn hintloc_c1_tree_reuse_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);
    let board = hintloc_c1_19x19_board();
    let next_pla = P_BLACK;
    let rules = Rules::parse_rules("Chinese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params_fast = make_v1_params();
    params_fast.max_visits = 5;

    let mut params_slow = make_v1_params();
    params_slow.max_visits = 200;

    let mut bot = AsyncBot::new(params_fast, &nn_eval, &logger, "hintloc");
    run_bot_on_position(
        &mut bot,
        board.clone(),
        next_pla,
        hist.clone(),
        TestSearchOptions::default(),
    );

    bot.set_params_no_clearing(&params_slow);
    bot.set_root_hint_loc(
        location::of_string("C1", board.x_size, board.y_size).expect("valid loc"),
    );
    run_bot_on_position(
        &mut bot,
        board,
        next_pla,
        hist,
        TestSearchOptions {
            ignore_position: true,
            ..Default::default()
        },
    );
}

#[test]
fn hintloc_c1_dirichlet_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);
    let board = hintloc_c1_19x19_board();
    let next_pla = P_BLACK;
    let rules = Rules::parse_rules("Chinese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 200;
    params.root_noise_enabled = true;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "hintloc");
    bot.set_root_hint_loc(
        location::of_string("C1", board.x_size, board.y_size).expect("valid loc"),
    );
    run_bot_on_position(
        &mut bot,
        board,
        next_pla,
        hist,
        TestSearchOptions::default(),
    );
}

#[test]
fn hintloc_c1_dirichlet_tree_reuse_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);
    let board = hintloc_c1_19x19_board();
    let next_pla = P_BLACK;
    let rules = Rules::parse_rules("Chinese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params_fast = make_v1_params();
    params_fast.max_visits = 5;
    params_fast.root_noise_enabled = true;

    let mut params_slow = params_fast.clone();
    params_slow.max_visits = 200;

    let mut bot = AsyncBot::new(params_fast, &nn_eval, &logger, "hintloc");
    run_bot_on_position(
        &mut bot,
        board.clone(),
        next_pla,
        hist.clone(),
        TestSearchOptions::default(),
    );

    bot.set_params_no_clearing(&params_slow);
    bot.set_root_hint_loc(
        location::of_string("C1", board.x_size, board.y_size).expect("valid loc"),
    );
    run_bot_on_position(
        &mut bot,
        board,
        next_pla,
        hist,
        TestSearchOptions {
            ignore_position: true,
            ..Default::default()
        },
    );
}

#[test]
fn hintloc_c1_dirichlet_symmetry_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);
    let board = hintloc_c1_19x19_board();
    let next_pla = P_BLACK;
    let rules = Rules::parse_rules("Chinese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 200;
    params.root_noise_enabled = true;
    params.root_num_symmetries_to_sample = 4;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "hintloc");
    bot.set_root_hint_loc(
        location::of_string("C1", board.x_size, board.y_size).expect("valid loc"),
    );
    run_bot_on_position(
        &mut bot,
        board,
        next_pla,
        hist,
        TestSearchOptions::default(),
    );
}

#[test]
fn hintloc_c1_dirichlet_symmetry_again_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);
    let board = hintloc_c1_19x19_board();
    let next_pla = P_BLACK;
    let rules = Rules::parse_rules("Chinese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 200;
    params.root_noise_enabled = true;
    params.root_num_symmetries_to_sample = 4;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "hintloc");
    bot.set_root_hint_loc(
        location::of_string("C1", board.x_size, board.y_size).expect("valid loc"),
    );
    run_bot_on_position(
        &mut bot,
        board,
        next_pla,
        hist,
        TestSearchOptions::default(),
    );
}

#[test]
fn hintloc_c1_dirichlet_symmetry_tree_reuse_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);
    let board = hintloc_c1_19x19_board();
    let next_pla = P_BLACK;
    let rules = Rules::parse_rules("Chinese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params_fast = make_v1_params();
    params_fast.max_visits = 5;
    params_fast.root_noise_enabled = true;
    params_fast.root_num_symmetries_to_sample = 4;

    let mut params_slow = params_fast.clone();
    params_slow.max_visits = 200;

    let mut bot = AsyncBot::new(params_fast, &nn_eval, &logger, "hintloc");
    run_bot_on_position(
        &mut bot,
        board.clone(),
        next_pla,
        hist.clone(),
        TestSearchOptions::default(),
    );

    bot.set_params_no_clearing(&params_slow);
    bot.set_root_hint_loc(
        location::of_string("C1", board.x_size, board.y_size).expect("valid loc"),
    );
    run_bot_on_position(
        &mut bot,
        board,
        next_pla,
        hist,
        TestSearchOptions {
            ignore_position: true,
            ..Default::default()
        },
    );
}

#[test]
fn hintloc_c1_symmetry_no_noise_seed1_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);
    let board = hintloc_c1_19x19_board();
    let next_pla = P_BLACK;
    let rules = Rules::parse_rules("Chinese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 200;
    params.root_noise_enabled = false;
    params.root_num_symmetries_to_sample = 4;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "abc");
    bot.set_root_hint_loc(
        location::of_string("C1", board.x_size, board.y_size).expect("valid loc"),
    );
    run_bot_on_position(
        &mut bot,
        board,
        next_pla,
        hist,
        TestSearchOptions::default(),
    );
}

#[test]
fn hintloc_c1_symmetry_no_noise_seed2_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);
    let board = hintloc_c1_19x19_board();
    let next_pla = P_BLACK;
    let rules = Rules::parse_rules("Chinese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 200;
    params.root_noise_enabled = false;
    params.root_num_symmetries_to_sample = 4;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "abc2");
    bot.set_root_hint_loc(
        location::of_string("C1", board.x_size, board.y_size).expect("valid loc"),
    );
    run_bot_on_position(
        &mut bot,
        board,
        next_pla,
        hist,
        TestSearchOptions::default(),
    );
}

fn mix_pruning_19x19_board() -> Board {
    Board::parse_board(
        19,
        19,
        "...................
...................
.....x.............
...x............x..
...................
..o................
...................
...................
...................
...................
...................
...................
...................
................x..
...................
..o.........o.o.x..
.....o.......oxx...
...................
...................
",
        '\n',
    )
    .expect("valid board")
}

#[test]
fn mix_pruning_base_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);
    let board = mix_pruning_19x19_board();
    let next_pla = P_WHITE;
    let rules = Rules::parse_rules("Chinese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 500;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "mix");
    run_bot_on_position(
        &mut bot,
        board,
        next_pla,
        hist,
        TestSearchOptions::default(),
    );
}

#[test]
fn mix_pruning_dirichlet_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);
    let board = mix_pruning_19x19_board();
    let next_pla = P_WHITE;
    let rules = Rules::parse_rules("Chinese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 500;
    params.root_noise_enabled = true;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "mix");
    run_bot_on_position(
        &mut bot,
        board,
        next_pla,
        hist,
        TestSearchOptions::default(),
    );
}

#[test]
fn mix_pruning_dirichlet_prune_sub_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);
    let board = mix_pruning_19x19_board();
    let next_pla = P_WHITE;
    let rules = Rules::parse_rules("Chinese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 500;
    params.root_noise_enabled = true;
    params.chosen_move_prune = 10.0;
    params.chosen_move_subtract = 7.0;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "mix");
    run_bot_on_position(
        &mut bot,
        board,
        next_pla,
        hist,
        TestSearchOptions::default(),
    );
}

#[test]
fn mix_pruning_dirichlet_value_weight_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);
    let board = mix_pruning_19x19_board();
    let next_pla = P_WHITE;
    let rules = Rules::parse_rules("Chinese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 500;
    params.root_noise_enabled = true;
    params.value_weight_exponent = 0.8;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "mix");
    run_bot_on_position(
        &mut bot,
        board,
        next_pla,
        hist,
        TestSearchOptions::default(),
    );
}

#[test]
fn mix_pruning_dirichlet_value_weight_prune_sub_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);
    let board = mix_pruning_19x19_board();
    let next_pla = P_WHITE;
    let rules = Rules::parse_rules("Chinese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 500;
    params.root_noise_enabled = true;
    params.chosen_move_prune = 12.0;
    params.chosen_move_subtract = 5.0;
    params.value_weight_exponent = 0.8;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "mix");
    run_bot_on_position(
        &mut bot,
        board,
        next_pla,
        hist,
        TestSearchOptions::default(),
    );
}

#[test]
fn mix_pruning_dirichlet_value_weight_more_visits_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);
    let board = mix_pruning_19x19_board();
    let next_pla = P_WHITE;
    let rules = Rules::parse_rules("Chinese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 2500;
    params.root_noise_enabled = true;
    params.value_weight_exponent = 0.8;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "mix");
    run_bot_on_position(
        &mut bot,
        board,
        next_pla,
        hist,
        TestSearchOptions::default(),
    );
}

#[test]
fn mix_pruning_dirichlet_value_weight_prune_sub_more_visits_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);
    let board = mix_pruning_19x19_board();
    let next_pla = P_WHITE;
    let rules = Rules::parse_rules("Chinese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 2500;
    params.root_noise_enabled = true;
    params.chosen_move_prune = 12.0;
    params.chosen_move_subtract = 5.0;
    params.value_weight_exponent = 0.8;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "mix");
    run_bot_on_position(
        &mut bot,
        board,
        next_pla,
        hist,
        TestSearchOptions::default(),
    );
}

#[test]
fn mix_pruning_dirichlet_value_weight_zero_more_visits_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);
    let board = mix_pruning_19x19_board();
    let next_pla = P_WHITE;
    let rules = Rules::parse_rules("Chinese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 2500;
    params.root_noise_enabled = true;
    params.value_weight_exponent = 0.0;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "mix");
    run_bot_on_position(
        &mut bot,
        board,
        next_pla,
        hist,
        TestSearchOptions::default(),
    );
}

#[test]
fn mix_pruning_dirichlet_value_weight_zero_prune_sub_more_visits_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);
    let board = mix_pruning_19x19_board();
    let next_pla = P_WHITE;
    let rules = Rules::parse_rules("Chinese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 2500;
    params.root_noise_enabled = true;
    params.chosen_move_prune = 12.0;
    params.chosen_move_subtract = 5.0;
    params.value_weight_exponent = 0.0;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "mix");
    run_bot_on_position(
        &mut bot,
        board,
        next_pla,
        hist,
        TestSearchOptions::default(),
    );
}

fn fill_dame_9x10_board() -> Board {
    Board::parse_board(
        9,
        10,
        ".....xoo.
.o...xo.o
.xx..xoox
x....xxxx
oxx...x..
oooxxox..
...ooxxxx
..o.xoox.
...ooo.ox
.......o.
",
        '\n',
    )
    .expect("valid board")
}

#[test]
fn fill_dame_before_pass_base_white_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(9, 10);
    let board = fill_dame_9x10_board();
    let next_pla = P_WHITE;
    let rules = Rules::parse_rules("Japanese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 600;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "fill dame before pass");
    run_bot_on_position(
        &mut bot,
        board,
        next_pla,
        hist,
        TestSearchOptions::default(),
    );
}

#[test]
fn fill_dame_before_pass_enabled_white_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(9, 10);
    let board = fill_dame_9x10_board();
    let next_pla = P_WHITE;
    let rules = Rules::parse_rules("Japanese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 600;
    params.fill_dame_before_pass = true;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "fill dame before pass");
    run_bot_on_position(
        &mut bot,
        board,
        next_pla,
        hist,
        TestSearchOptions::default(),
    );
}

#[test]
fn fill_dame_before_pass_base_black_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(9, 10);
    let board = fill_dame_9x10_board();
    let next_pla = P_BLACK;
    let rules = Rules::parse_rules("Japanese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 600;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "fill dame before pass");
    run_bot_on_position(
        &mut bot,
        board,
        next_pla,
        hist,
        TestSearchOptions::default(),
    );
}

#[test]
fn fill_dame_before_pass_enabled_black_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(9, 10);
    let board = fill_dame_9x10_board();
    let next_pla = P_BLACK;
    let rules = Rules::parse_rules("Japanese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 600;
    params.fill_dame_before_pass = true;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "fill dame before pass");
    run_bot_on_position(
        &mut bot,
        board,
        next_pla,
        hist,
        TestSearchOptions::default(),
    );
}

#[test]
fn conservative_pass_white_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(9, 9);
    let board = Board::parse_board(
        9,
        9,
        "ox.x.xx..
.x.x.x.xx
xxxx.xxx.
.x..x..xx
xxxxxxxxo
xoooxoooo
xo..o.ox.
oooo.o.oo
...ooooo.
",
        '\n',
    )
    .expect("valid board");

    let next_pla = P_WHITE;
    let mut rules = Rules::parse_rules("Chinese").expect("valid rules");
    rules.set_komi(14.0);
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 600;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "conservative pass");
    run_bot_on_position(
        &mut bot,
        board.clone(),
        next_pla,
        hist.clone(),
        TestSearchOptions {
            no_clear_bot: true,
            ..Default::default()
        },
    );

    let search = bot.get_search_stop_and_wait();
    let move_loc = search.get_chosen_move_loc();
    assert_move_legal_or_pass(&board, &hist, move_loc, next_pla);
}

#[test]
fn conservative_pass_black_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(9, 9);
    let board = Board::parse_board(
        9,
        9,
        "ox.x.xx..
.x.x.x.xx
xxxx.xxx.
.x..x..xx
xxxxxxxxo
xoooxoooo
xo..o.ox.
oooo.o.oo
...ooooo.
",
        '\n',
    )
    .expect("valid board");

    let next_pla = P_BLACK;
    let rules = Rules::parse_rules("Chinese").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 600;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "conservative pass");
    run_bot_on_position(
        &mut bot,
        board.clone(),
        next_pla,
        hist.clone(),
        TestSearchOptions {
            no_clear_bot: true,
            ..Default::default()
        },
    );

    let search = bot.get_search_stop_and_wait();
    let move_loc = search.get_chosen_move_loc();
    assert_move_legal_or_pass(&board, &hist, move_loc, next_pla);
}

#[test]
fn basic_graph_search_7x7_fight_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(7, 7);
    let board = Board::parse_board(
        7,
        7,
        ".....o.
.....ox
..ooox.
.xoxxx.
.xxo.x.
..xooox
.......
",
        '\n',
    )
    .expect("valid board");

    let next_pla = P_WHITE;
    let mut rules = Rules::parse_rules("Japanese").expect("valid rules");
    rules.set_komi(8.0);
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 1000;
    params.use_graph_search = true;

    let opts = TestSearchOptions {
        num_moves_in_a_row: 3,
        print_post_order_node_count: true,
        ..Default::default()
    };

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "conservative pass");
    run_bot_on_position(&mut bot, board, next_pla, hist, opts);
}

#[test]
fn friendly_pass_white_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(9, 9);
    let board = Board::parse_board(
        9,
        9,
        ".........
....x.x..
.x.xox...
...x..xxx
xxxxxxooo
xooooo...
o....xo..
.o...o.o.
.........
",
        '\n',
    )
    .expect("valid board");

    let next_pla = P_BLACK;
    let mut rules = Rules::parse_rules("Chinese").expect("valid rules");
    rules.set_komi(4.0);
    let mut board = board.clone();
    let mut hist = BoardHistory::new(board.clone(), next_pla, rules, 0);
    hist.make_board_move_assume_legal(&mut board, PASS_LOC, next_pla);
    let next_pla = P_WHITE;

    let mut params = make_v1_params();
    params.max_visits = 600;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "friendly pass");
    run_bot_on_position(
        &mut bot,
        board.clone(),
        next_pla,
        hist.clone(),
        TestSearchOptions {
            no_clear_bot: true,
            ..Default::default()
        },
    );

    let search = bot.get_search_stop_and_wait();
    let move_loc = search.get_chosen_move_loc();
    assert_move_legal_or_pass(&board, &hist, move_loc, next_pla);
}

#[test]
fn friendly_pass_black_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(9, 9);
    let board = Board::parse_board(
        9,
        9,
        ".........
....x.x..
.x.xox...
......xxx
xxxxxxooo
xooooo...
o....xo..
.o..oo.o.
.........
",
        '\n',
    )
    .expect("valid board");

    let next_pla = P_WHITE;
    let mut rules = Rules::parse_rules("Chinese").expect("valid rules");
    rules.set_komi(7.0);
    let mut board = board.clone();
    let mut hist = BoardHistory::new(board.clone(), next_pla, rules, 0);
    hist.make_board_move_assume_legal(&mut board, PASS_LOC, next_pla);
    let next_pla = P_BLACK;

    let mut params = make_v1_params();
    params.max_visits = 600;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "friendly pass");
    run_bot_on_position(
        &mut bot,
        board.clone(),
        next_pla,
        hist.clone(),
        TestSearchOptions {
            no_clear_bot: true,
            ..Default::default()
        },
    );

    let search = bot.get_search_stop_and_wait();
    let move_loc = search.get_chosen_move_loc();
    assert_move_legal_or_pass(&board, &hist, move_loc, next_pla);
}

#[test]
fn multithreaded_tree_updating_smoke() {
    fn run_test(
        logger: &std::sync::Arc<kata_core::logger::Logger>,
        nn_eval: &kata_nn::eval::NnEvaluator,
        num_visits: i64,
        num_threads: i32,
        subtree_value_bias: bool,
        graph_search: bool,
    ) {
        let board = Board::parse_board(
            19,
            19,
            "...................
...................
...................
...x...........x...
...................
...................
...................
...................
...................
...................
...................
...................
..x.xo.............
...xo..............
.o.xo..............
..ooxo.........o...
.oxxxo.............
.....xo............
...................
",
            '\n',
        )
        .expect("valid board");
        let next_pla_base = P_BLACK;
        let rules = Rules::parse_rules("Japanese").expect("valid rules");
        let hist_base = BoardHistory::new(board.clone(), next_pla_base, rules, 0);

        let mut params = make_v1_params();
        params.max_visits = num_visits;
        if subtree_value_bias {
            params.subtree_value_bias_factor = 0.35;
            params.subtree_value_bias_weight_exponent = 0.8;
            params.subtree_value_bias_free_prop = 0.0;
        }
        if graph_search {
            params.use_graph_search = true;
        }

        nn_eval.clear_cache();
        nn_eval.clear_stats();

        let mut search = Search::new(params, nn_eval, logger, "multithreaded tree updating");
        let mut board = board;
        let mut hist = hist_base;
        let next_pla = next_pla_base;
        search.set_position(next_pla, &board, &hist);
        search.run_whole_search(next_pla);

        let mut params_with_threads = search.search_params.clone();
        params_with_threads.num_threads = num_threads;
        search.set_params_no_clearing(&params_with_threads);

        assert!(search.get_root_visits() > 0);
        let move_loc = location::of_string("E2", board.x_size, board.y_size).expect("valid loc");
        assert_move_legal_or_pass(&board, &hist, move_loc, next_pla);

        let options = PrintTreeOptions::new().max_depth(1);
        let root = search.root_node.as_deref().expect("root node should exist");
        let mut out = String::new();
        search.print_tree(&mut out, Some(root), &options, P_WHITE);
        assert!(!out.is_empty());

        search.make_move(move_loc, next_pla);
        hist.make_board_move_assume_legal(&mut board, move_loc, next_pla);

        nn_eval.clear_cache();
        nn_eval.clear_stats();
    }

    let (logger, nn_eval) = test_logger_and_eval(19, 19);

    run_test(&logger, &nn_eval, 1000, 1, false, false);
    run_test(&logger, &nn_eval, 1000, 1, false, true);
    run_test(&logger, &nn_eval, 1000, 4, false, false);
    run_test(&logger, &nn_eval, 1000, 4, true, false);
    run_test(&logger, &nn_eval, 2000, 1, false, false);
    run_test(&logger, &nn_eval, 2000, 4, false, false);
    run_test(&logger, &nn_eval, 2000, 8, false, false);
    run_test(&logger, &nn_eval, 2000, 8, false, true);
    run_test(&logger, &nn_eval, 2000, 4, true, false);
}

fn pattern_bonus_19x19_board() -> Board {
    Board::parse_board(
        19,
        19,
        "...................
...................
...................
...x...............
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
...o...............
...................
...................
",
        '\n',
    )
    .expect("valid board")
}

#[test]
fn pattern_bonus_main_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);
    let mut board = pattern_bonus_19x19_board();
    let mut next_pla = P_BLACK;
    let mut rules = Rules::parse_rules("Japanese").expect("valid rules");
    rules.set_komi(6.5);
    let mut hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let moves = ["R3", "Q3", "R4", "Q5", "S6", "F17", "C14"];
    for mv_str in moves {
        let mv = location::of_string(mv_str, board.x_size, board.y_size).expect("valid loc");
        hist.make_board_move_assume_legal(&mut board, mv, next_pla);
        next_pla = get_opp(next_pla);
    }

    let mut params0 = make_v1_params();
    params0.max_visits = 1000;
    let mut params1 = params0.clone();
    params1.avoid_repeated_pattern_utility = 0.2;

    let mut bot0 = AsyncBot::new(params0, &nn_eval, &logger, "pattern bonus");
    let mut bot1 = AsyncBot::new(params1.clone(), &nn_eval, &logger, "pattern bonus");
    let mut bot2 = AsyncBot::new(params1, &nn_eval, &logger, "pattern bonus");

    let opts = TestSearchOptions {
        no_clear_bot: true,
        ..Default::default()
    };

    run_bot_on_position(&mut bot0, board.clone(), next_pla, hist.clone(), opts);
    run_bot_on_position(&mut bot1, board.clone(), next_pla, hist.clone(), opts);
    run_bot_on_position(&mut bot2, board.clone(), next_pla, hist.clone(), opts);

    let d18 = location::of_string("D18", board.x_size, board.y_size).expect("valid loc");
    hist.make_board_move_assume_legal(&mut board, d18, next_pla);
    bot0.make_move(d18, next_pla);
    bot1.make_move(d18, next_pla);
    bot2.make_move(d18, next_pla);
    let next_pla_after_d18 = get_opp(next_pla);

    let opts_continue = TestSearchOptions {
        no_clear_bot: true,
        ignore_position: true,
        ..Default::default()
    };
    run_bot_on_position(
        &mut bot0,
        board.clone(),
        next_pla_after_d18,
        hist.clone(),
        opts_continue,
    );
    run_bot_on_position(
        &mut bot1,
        board.clone(),
        next_pla_after_d18,
        hist.clone(),
        opts_continue,
    );

    let m4 = location::of_string("M4", board.x_size, board.y_size).expect("valid loc");
    hist.make_board_move_assume_legal(&mut board, m4, next_pla_after_d18);
    bot0.make_move(m4, next_pla_after_d18);
    bot1.make_move(m4, next_pla_after_d18);
    bot2.make_move(m4, next_pla_after_d18);
    let next_pla_after_m4 = get_opp(next_pla_after_d18);

    run_bot_on_position(
        &mut bot0,
        board.clone(),
        next_pla_after_m4,
        hist.clone(),
        opts_continue,
    );
    run_bot_on_position(
        &mut bot1,
        board.clone(),
        next_pla_after_m4,
        hist.clone(),
        opts_continue,
    );
    run_bot_on_position(
        &mut bot2,
        board.clone(),
        next_pla_after_m4,
        hist,
        opts_continue,
    );
}

fn pattern_bonus_ko_19x19_board() -> Board {
    Board::parse_board(
        19,
        19,
        "...................
...................
...................
...x...............
..x................
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
.....x.............
..o.x..............
..o.x..............
...................
",
        '\n',
    )
    .expect("valid board")
}

#[test]
fn pattern_bonus_does_not_care_about_ko_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);
    let mut board = pattern_bonus_ko_19x19_board();
    let mut next_pla = P_BLACK;
    let mut rules = Rules::parse_rules("Japanese").expect("valid rules");
    rules.set_komi(6.5);
    let mut hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let moves = ["R17", "D15", "C15", "Q17", "C5", "B5"];
    for mv_str in moves {
        let mv = location::of_string(mv_str, board.x_size, board.y_size).expect("valid loc");
        hist.make_board_move_assume_legal(&mut board, mv, next_pla);
        next_pla = get_opp(next_pla);
    }

    let mut params0 = make_v1_params();
    params0.max_visits = 1000;
    let mut params1 = params0.clone();
    params1.avoid_repeated_pattern_utility = 0.2;

    let mut bot0 = AsyncBot::new(params0, &nn_eval, &logger, "pattern bonus");
    let mut bot1 = AsyncBot::new(params1, &nn_eval, &logger, "pattern bonus");

    let opts = TestSearchOptions {
        no_clear_bot: true,
        ..Default::default()
    };

    run_bot_on_position(&mut bot0, board.clone(), next_pla, hist.clone(), opts);
    run_bot_on_position(&mut bot1, board, next_pla, hist, opts);
}

#[test]
fn pattern_bonus_does_not_multi_count_shapes_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);
    let mut board = pattern_bonus_19x19_board();
    let mut next_pla = P_WHITE;
    let mut rules = Rules::parse_rules("Japanese").expect("valid rules");
    rules.set_komi(25.5);
    let mut hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params0 = make_v1_params();
    params0.max_visits = 1000;
    params0.avoid_repeated_pattern_utility = 0.2;

    let mut bot0 = AsyncBot::new(params0, &nn_eval, &logger, "pattern bonus");

    let opts = TestSearchOptions {
        no_clear_bot: false,
        ..Default::default()
    };

    run_bot_on_position(&mut bot0, board.clone(), next_pla, hist.clone(), opts);

    let r14 = location::of_string("R14", board.x_size, board.y_size).expect("valid loc");
    hist.make_board_move_assume_legal(&mut board, r14, next_pla);
    bot0.make_move(r14, next_pla);
    next_pla = get_opp(next_pla);

    let o17 = location::of_string("O17", board.x_size, board.y_size).expect("valid loc");
    hist.make_board_move_assume_legal(&mut board, o17, next_pla);
    bot0.make_move(o17, next_pla);
    next_pla = get_opp(next_pla);

    run_bot_on_position(&mut bot0, board.clone(), next_pla, hist.clone(), opts);

    let f17 = location::of_string("F17", board.x_size, board.y_size).expect("valid loc");
    hist.make_board_move_assume_legal(&mut board, f17, next_pla);
    bot0.make_move(f17, next_pla);
    next_pla = get_opp(next_pla);

    let c14 = location::of_string("C14", board.x_size, board.y_size).expect("valid loc");
    hist.make_board_move_assume_legal(&mut board, c14, next_pla);
    bot0.make_move(c14, next_pla);
    next_pla = get_opp(next_pla);

    run_bot_on_position(&mut bot0, board.clone(), next_pla, hist.clone(), opts);

    let c6 = location::of_string("C6", board.x_size, board.y_size).expect("valid loc");
    hist.make_board_move_assume_legal(&mut board, c6, next_pla);
    bot0.make_move(c6, next_pla);
    next_pla = get_opp(next_pla);

    let f3 = location::of_string("F3", board.x_size, board.y_size).expect("valid loc");
    hist.make_board_move_assume_legal(&mut board, f3, next_pla);
    bot0.make_move(f3, next_pla);
    next_pla = get_opp(next_pla);

    run_bot_on_position(&mut bot0, board, next_pla, hist, opts);
}

#[test]
fn ownership_endgame_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(7, 9);
    let board = Board::parse_board(
        7,
        9,
        "x.ooo.x
xxxxxxx
oooooxx
.o..oo.
ooooooo
.oxxxxx
ooox..o
oxxxxxx
xx.....
",
        '\n',
    )
    .expect("valid board");

    let next_pla = P_WHITE;
    let rules = Rules::get_tromp_taylorish();
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 100;
    params.futile_visits_threshold = 0.4;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "Endgame ownership test");

    let opts = TestSearchOptions {
        print_ownership: true,
        ..Default::default()
    };

    run_bot_on_position(&mut bot, board, next_pla, hist, opts);
}

#[test]
fn sampled_symmetries_repeated_runs_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(15, 15);
    let board = Board::parse_board(
        15,
        15,
        "...............
...............
...x.....x.....
............o..
...o...........
............x..
...............
...x........o..
...............
....o..........
...............
...o........x..
.....x....o....
...............
...............
",
        '\n',
    )
    .expect("valid board");

    let next_pla = P_BLACK;
    let rules = Rules::parse_rules("AGA").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.root_num_symmetries_to_sample = 8;
    params.max_visits = 1;

    let opts = TestSearchOptions {
        print_root_policy: true,
        ..Default::default()
    };

    for seed in ["sample", "sample2", "sample3"] {
        let mut bot = AsyncBot::new(params.clone(), &nn_eval, &logger, seed);
        run_bot_on_position(&mut bot, board.clone(), next_pla, hist.clone(), opts);
    }
}

#[test]
fn sampled_symmetries_distribution_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(15, 15);
    let board = Board::parse_board(
        15,
        15,
        "...............
...............
...x.....x.....
............o..
...o...........
............x..
...............
...x........o..
...............
....o..........
...............
...o........x..
.....x....o....
...............
...............
",
        '\n',
    )
    .expect("valid board");

    let next_pla = P_BLACK;
    let rules = Rules::parse_rules("AGA").expect("valid rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.root_num_symmetries_to_sample = 2;
    params.max_visits = 1;

    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "two root syms");
    bot.set_position(next_pla, &board, &hist);

    let tc = kata_search::time_control::TimeControls::default();
    let mut policy_samples = HashSet::new();
    let mut wl_samples = HashSet::new();

    for _ in 0..500 {
        bot.gen_move_synchronous(next_pla, &tc);
        let search = bot.get_search();
        let mut policy_probs = [0.0f32; MAX_NN_POLICY_SIZE];
        assert!(search.get_policy(&mut policy_probs));
        policy_samples.insert(policy_probs[xy_to_pos(2, 4, search.nn_x_len) as usize].to_bits());
        wl_samples.insert(
            search
                .get_root_values_require_success()
                .win_loss_value
                .to_bits(),
        );
        bot.clear_search();
    }

    assert!(!policy_samples.is_empty());
    assert!(!wl_samples.is_empty());
}

#[test]
fn multithreaded_search_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);
    let board = Board::parse_board(
        19,
        19,
        "...................
...................
...x...........x...
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
..oo...........o...
..xxo..............
...................
...................
...................
...................
",
        '\n',
    )
    .expect("valid board");

    let mut next_pla = P_BLACK;
    let mut rules = Rules::parse_rules("Japanese").expect("valid rules");
    rules.set_komi(8.5);
    let mut hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 2000;
    params.subtree_value_bias_factor = 0.35;
    params.subtree_value_bias_weight_exponent = 0.8;
    params.subtree_value_bias_free_prop = 0.8;
    params.chosen_move_temperature = 0.0;
    params.chosen_move_temperature_early = 0.0;
    params.use_noise_pruning = true;
    params.num_threads = 8;

    let mut search = Search::new(params, &nn_eval, &logger, "multithreaded test");
    let mut board = board;

    for _ in 0..3 {
        search.set_position(next_pla, &board, &hist);
        search.run_whole_search(next_pla);
        let move_loc = search.get_chosen_move_loc();
        assert_move_legal_or_pass(&board, &hist, move_loc, next_pla);
        assert!(search.get_root_visits() > 0);
        let _ = search.get_root_values_require_success();

        search.make_move(move_loc, next_pla);
        hist.make_board_move_assume_legal(&mut board, move_loc, next_pla);
        next_pla = get_opp(next_pla);
    }

    nn_eval.clear_cache();
    nn_eval.clear_stats();
}

#[test]
fn multithreaded_graph_search_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);
    let board = Board::parse_board(
        19,
        19,
        "...................
....x.x......xxoo..
...xox.....x.o.xo..
..x.o..........xo..
..xo...........xx..
..oo............o..
...................
...................
..o................
...................
...................
...................
................x..
...............o...
.o.............ox..
o.oo............x..
xoxxoo........o.x..
.x.................
...................
",
        '\n',
    )
    .expect("valid board");

    let mut next_pla = P_WHITE;
    let mut rules = Rules::parse_rules("Chinese").expect("valid rules");
    rules.set_komi(6.5);
    let mut hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let mut params = make_v1_params();
    params.max_visits = 2000;
    params.use_graph_search = true;
    params.subtree_value_bias_factor = 0.35;
    params.subtree_value_bias_weight_exponent = 0.8;
    params.subtree_value_bias_free_prop = 0.8;
    params.chosen_move_temperature = 0.0;
    params.chosen_move_temperature_early = 0.0;
    params.use_noise_pruning = true;
    params.num_threads = 8;

    let mut search = Search::new(params, &nn_eval, &logger, "multithreaded test");
    let mut board = board;

    for _ in 0..2 {
        search.set_position(next_pla, &board, &hist);
        search.run_whole_search(next_pla);
        let move_loc = search.get_chosen_move_loc();
        assert_move_legal_or_pass(&board, &hist, move_loc, next_pla);
        assert!(search.get_root_visits() > 0);
        let _ = search.get_root_values_require_success();

        search.make_move(move_loc, next_pla);
        hist.make_board_move_assume_legal(&mut board, move_loc, next_pla);
        next_pla = get_opp(next_pla);
    }

    nn_eval.clear_cache();
    nn_eval.clear_stats();
}
