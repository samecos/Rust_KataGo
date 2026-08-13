//! Integration tests ported from `KataGo/cpp/tests/testsearchv9.cpp`.
//!
//! These tests use the dummy neural-net evaluator, so the C++ golden output
//! comparisons are skipped in favour of smoke-test assertions that the APIs do
//! not panic and return plausible results.

mod common;

use common::{TestSearchOptions, run_bot_on_position, run_bot_on_sgf, test_logger_and_eval};

use kata_game::board::{Board, MAX_ARR_SIZE, P_BLACK, P_WHITE, PASS_LOC, Player, location};
use kata_game::history::BoardHistory;
use kata_game::rules::{KoRule, Rules};
use kata_game::symmetry::NUM_SYMMETRIES;
use kata_nn::backend::NNResultBuf;
use kata_nn::inputs::{
    MiscNNInputParams, NNOutput, NUM_FEATURES_GLOBAL_V7, NUM_FEATURES_SPATIAL_V7, fill_row_v7,
};
use kata_search::async_bot::AsyncBot;
use kata_search::params::SearchParams;

const SGF_FLYING_DAGGER: &str = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Japanese]SZ[19]KM[6.50]PW[White]PB[Black];B[pd];W[dd];B[pp];W[dp];B[cq];W[dq];B[cp];W[cn];B[bn];W[bm];B[co];W[dn];B[do];W[eo];B[ep];W[er];B[fp];W[fo];B[go];W[gn];B[gq];W[cr];B[bo];W[cl];B[br];W[gr];B[hr];W[bs];B[bq];W[fr];B[hn];W[ho];B[gp];W[ir];B[hs];W[ap];B[fn];W[en];B[gm];W[nc];B[dl];W[dk];B[em];W[el];B[al];W[bk];B[am];W[ao])";

const SGF_3R_PINCER: &str = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Japanese]SZ[19]KM[6.50]PW[White]PB[Black];B[pd];W[dd];B[pp];W[cp];B[eq];W[hq];B[do];W[dp];B[ep];W[eo];B[dn];W[en];B[bo];W[bp];B[dm];W[em];B[dl];W[dq];B[gp];W[hp];B[er];W[go];B[cr];W[dr];B[ds];W[br];B[bs];W[aq];B[gr];W[hr])";

fn make_v2_params() -> SearchParams {
    let mut params = SearchParams::for_tests_v2();
    params.value_weight_exponent = 0.0;
    params
}

fn full_symmetry_output(
    board: &Board,
    hist: &BoardHistory,
    pla: Player,
    include_owner_map: bool,
    nn_eval: &kata_nn::eval::NnEvaluator,
) -> NNOutput {
    let mut outputs = Vec::with_capacity(NUM_SYMMETRIES as usize);
    for sym in 0..NUM_SYMMETRIES {
        let mut buf = NNResultBuf::new();
        let nn_input_params = MiscNNInputParams {
            symmetry: sym,
            ..MiscNNInputParams::default()
        };
        nn_eval.evaluate(
            board,
            hist,
            pla,
            &nn_input_params,
            &mut buf,
            true,
            include_owner_map,
        );
        let arc = buf
            .result
            .expect("full symmetry evaluation should produce a result");
        outputs.push((*arc).clone());
    }
    NNOutput::average(&outputs)
}

fn evaluate_smoke(
    nn_eval: &kata_nn::eval::NnEvaluator,
    board: &Board,
    hist: &BoardHistory,
    next_pla: Player,
    symmetry: i32,
    include_owner_map: bool,
) -> NNOutput {
    let mut buf = NNResultBuf::new();
    let nn_input_params = MiscNNInputParams {
        symmetry,
        ..MiscNNInputParams::default()
    };
    nn_eval.evaluate(
        board,
        hist,
        next_pla,
        &nn_input_params,
        &mut buf,
        true,
        include_owner_map,
    );
    let arc = buf.result.expect("evaluate should produce a result");
    (*arc).clone()
}

#[test]
fn variety_sgf_positions_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);
    let rules = Rules::parse_rules("Japanese").expect("valid rules");

    let mut params = make_v2_params();
    params.max_visits = 100;
    let mut bot = AsyncBot::new(params.clone(), &nn_eval, &logger, "testSearchSeedV9_0");

    let opts = TestSearchOptions {
        print_play_selection_values: true,
        no_clear_bot: true,
        ..Default::default()
    };

    for i in 15..47 {
        params.use_non_buggy_lcb = i % 3 != 0;
        params.root_noise_enabled = i % 5 == 0;
        params.lcb_stdevs = if i % 7 == 0 { 2.5 } else { 5.0 };
        params.min_visit_prop_for_lcb = if i % 7 == 0 { 0.03 } else { 0.15 };
        params.use_lcb_for_selection = i % 7 != 1;
        params.playout_doubling_advantage = if i % 11 == 0 { 0.75 } else { 0.0 };
        bot.set_params(&params);
        run_bot_on_sgf(&mut bot, SGF_FLYING_DAGGER, &rules, i, 6.5, opts);
    }

    for i in 11..30 {
        params.use_non_buggy_lcb = i % 3 != 0;
        params.root_noise_enabled = i % 5 == 0;
        params.lcb_stdevs = if i % 7 < 4 { 2.5 } else { 5.0 };
        params.min_visit_prop_for_lcb = if i % 7 < 4 { 0.03 } else { 0.15 };
        params.use_lcb_for_selection = i % 7 <= 5;
        params.playout_doubling_advantage = if i % 11 < 3 { 0.75 } else { 0.0 };
        bot.set_params(&params);
        run_bot_on_sgf(&mut bot, SGF_3R_PINCER, &rules, i, 6.5, opts);
    }
}

#[test]
fn pruned_root_values_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);

    let board = Board::parse_board(
        13,
        13,
        ".xoxo.o......
ooox.oxox....
ooxxxxxo.x...
oxxoooxo.....
ooxo.ooxx.x..
ooox.........
xoxx..o.x....
xx...x.o..x..
..x..........
.......o.xx..
..ox..o...oo.
.ox.xxxoo....
.............
",
        '\n',
    )
    .expect("valid board");
    let next_pla = P_WHITE;
    let rules = Rules::parse_rules("Chinese").expect("valid rules");

    let opts = TestSearchOptions {
        print_play_selection_values: true,
        print_root_values: true,
        print_pruned_root_values: true,
        ..Default::default()
    };

    {
        let mut params = make_v2_params();
        params.max_visits = 600;
        params.root_fpu_reduction_max = 0.0;
        let mut bot = AsyncBot::new(params, &nn_eval, &logger, "testSearchSeedV9_1");
        let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);
        run_bot_on_position(&mut bot, board.clone(), next_pla, hist, opts);
    }
    {
        let mut params = make_v2_params();
        params.max_visits = 600;
        params.root_fpu_reduction_max = 0.0;
        params.root_noise_enabled = true;
        let mut bot = AsyncBot::new(params, &nn_eval, &logger, "testSearchSeedV9_2");
        let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);
        run_bot_on_position(&mut bot, board.clone(), next_pla, hist, opts);
    }
}

#[test]
fn conservative_pass_hint_loc_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);

    let board = Board::parse_board(
        9,
        9,
        ".ox.xo.xx
oxx.xooo.
.xo.x.xoo
xxo...xxx
xxxx.....
ooox..xx.
.xox.xxoo
o.ox..oox
.oxx..ox.
",
        '\n',
    )
    .expect("valid board");
    let next_pla = P_WHITE;
    let opts = TestSearchOptions {
        num_moves_in_a_row: 25,
        root_hint_loc: PASS_LOC,
        ..Default::default()
    };

    for conservative_pass in [false, true] {
        for rules_str in ["Chinese", "Japanese"] {
            let rules = Rules::parse_rules(rules_str).expect("valid rules");
            let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);
            let mut params = make_v2_params();
            params.max_visits = 75;
            params.root_fpu_reduction_max = 0.0;
            params.conservative_pass = conservative_pass;
            let mut bot = AsyncBot::new(params, &nn_eval, &logger, "testSearchSeedV9_3");
            run_bot_on_position(&mut bot, board.clone(), next_pla, hist, opts);
        }
    }
}

#[test]
fn ladder_history_spight_rules_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);

    let mut board = Board::parse_board(
        7,
        7,
        "....xo.
....xo.
....xo.
....xoo
.xxxxo.
.xoooox
.xo.xx.
",
        '\n',
    )
    .expect("valid board");
    let next_pla = P_BLACK;

    for conservative_pass in [false, true] {
        let mut rules = Rules::parse_rules("Japanese").expect("valid rules");
        rules.ko_rule = KoRule::Spight;
        let mut hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

        let moves = ["G3", "G1", "G2", "pass", "G3", "G1", "G2", "pass"];
        for mv in moves {
            let loc = location::of_string(mv, board.x_size, board.y_size).expect("valid loc");
            let pla = hist.presumed_next_move_pla;
            hist.make_board_move_assume_legal(&mut board, loc, pla);
        }

        let mut params = make_v2_params();
        params.max_visits = 100;
        params.root_fpu_reduction_max = 0.0;
        params.conservative_pass = conservative_pass;
        let mut bot = AsyncBot::new(params, &nn_eval, &logger, "testSearchSeedV9_4");
        let hist_for_search = BoardHistory::new(board.clone(), next_pla, rules, 0);
        run_bot_on_position(
            &mut bot,
            board.clone(),
            next_pla,
            hist_for_search,
            TestSearchOptions::default(),
        );

        let nn_x_len = 7;
        let nn_y_len = 7;
        let mut row_bin = vec![0.0f32; (NUM_FEATURES_SPATIAL_V7 * nn_x_len * nn_y_len) as usize];
        let mut row_global = vec![0.0f32; NUM_FEATURES_GLOBAL_V7 as usize];
        let nn_input_params = MiscNNInputParams {
            draw_equivalent_wins_for_white: 0.5,
            conservative_pass_and_is_root: conservative_pass,
            ..MiscNNInputParams::default()
        };
        fill_row_v7(
            &board,
            &hist,
            next_pla,
            &nn_input_params,
            nn_x_len,
            nn_y_len,
            false,
            &mut row_bin,
            &mut row_global,
        );

        let mut out = String::new();
        for feature in 14..=16 {
            out.push_str(&format!("Ladder feature {}\n", feature));
            for y in 0..nn_y_len {
                for x in 0..nn_x_len {
                    let idx = (feature * nn_x_len * nn_y_len + y * nn_x_len + x) as usize;
                    out.push_str(&format!("{} ", row_bin[idx]));
                }
                out.push('\n');
            }
        }
        assert!(!out.is_empty());
    }
}

#[test]
fn avoid_move_until_rescale_root_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);

    let rules = Rules::get_tromp_taylorish();
    let board = Board::parse_board(
        9,
        9,
        ".........
.........
.........
....x....
....ox...
....xo...
.........
.........
.........
",
        '\n',
    )
    .expect("valid board");
    let next_pla = P_WHITE;
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let d5 = location::of_string("D5", board.x_size, board.y_size).expect("valid loc");
    let mut avoid = vec![0; MAX_ARR_SIZE];
    avoid[d5 as usize] = 1;

    let mut params = make_v2_params();
    params.max_visits = 200;

    let mut bot_a = AsyncBot::new(
        params.clone(),
        &nn_eval,
        &logger,
        "avoidMoveUntilRescaleRoot test",
    );
    let mut bot_b = AsyncBot::new(
        params.clone(),
        &nn_eval,
        &logger,
        "avoidMoveUntilRescaleRoot test",
    );
    let mut bot_c = AsyncBot::new(
        params.clone(),
        &nn_eval,
        &logger,
        "avoidMoveUntilRescaleRoot test",
    );
    let mut bot_d = AsyncBot::new(
        params.clone(),
        &nn_eval,
        &logger,
        "avoidMoveUntilRescaleRoot test",
    );

    bot_a.set_position(next_pla, &board, &hist);
    bot_b.set_position(next_pla, &board, &hist);
    bot_c.set_position(next_pla, &board, &hist);
    bot_d.set_position(next_pla, &board, &hist);

    bot_b.set_avoid_move_until_rescale_root(true);
    bot_c.set_avoid_move_until_by_loc(&avoid, &avoid);
    bot_d.set_avoid_move_until_rescale_root(true);
    bot_d.set_avoid_move_until_by_loc(&avoid, &avoid);

    let opts = TestSearchOptions {
        ignore_position: true,
        ..Default::default()
    };

    run_bot_on_position(&mut bot_a, board.clone(), next_pla, hist.clone(), opts);
    run_bot_on_position(&mut bot_b, board.clone(), next_pla, hist.clone(), opts);
    run_bot_on_position(&mut bot_c, board.clone(), next_pla, hist.clone(), opts);
    run_bot_on_position(&mut bot_d, board.clone(), next_pla, hist.clone(), opts);
}

#[test]
fn passing_details_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);

    let board = Board::parse_board(
        8,
        8,
        "..ooo...
x.ox.xx.
ooox.x..
.xx..xxo
..xxxoo.
xxxooo..
xoo.....
.o.o....
",
        '\n',
    )
    .expect("valid board");
    let next_pla = P_BLACK;

    let opts = TestSearchOptions {
        root_hint_loc: PASS_LOC,
        print_more: true,
        ..Default::default()
    };

    {
        let mut rules = Rules::parse_rules("Chinese").expect("valid rules");
        rules.friendly_pass_ok = false;
        let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);
        let mut params = make_v2_params();
        params.max_visits = 50;
        params.root_fpu_reduction_max = 0.0;
        let mut bot = AsyncBot::new(params, &nn_eval, &logger, "testSearchSeedV9_5");
        run_bot_on_position(&mut bot, board.clone(), next_pla, hist, opts);
    }
    {
        let mut rules = Rules::parse_rules("Chinese").expect("valid rules");
        rules.friendly_pass_ok = true;
        let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);
        let mut params = make_v2_params();
        params.max_visits = 50;
        params.root_fpu_reduction_max = 0.0;
        let mut bot = AsyncBot::new(params, &nn_eval, &logger, "testSearchSeedV9_6");
        run_bot_on_position(&mut bot, board.clone(), next_pla, hist, opts);
    }
    {
        let mut rules = Rules::parse_rules("Chinese").expect("valid rules");
        rules.friendly_pass_ok = false;
        let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);
        let mut params = make_v2_params();
        params.max_visits = 50;
        params.root_fpu_reduction_max = 0.0;
        params.enable_passing_hacks = true;
        let mut bot = AsyncBot::new(params, &nn_eval, &logger, "testSearchSeedV9_7");
        run_bot_on_position(&mut bot, board.clone(), next_pla, hist, opts);
    }
}

#[test]
fn symmetry_raw_nets_smoke() {
    let (_logger, nn_eval) = test_logger_and_eval(19, 19);

    {
        let mut board = Board::parse_board(
            8,
            8,
            "........
........
.x.xx...
xxxo.x..
xoo.oxxx
xo..ooxo
o.....oo
.o......
",
            '\n',
        )
        .expect("valid board");
        let next_pla = P_WHITE;
        let rules = Rules::parse_rules("Chinese").expect("valid rules");
        let mut hist = BoardHistory::new(board.clone(), next_pla, rules, 0);
        let e5 = location::of_string("E5", board.x_size, board.y_size).expect("valid loc");
        hist.make_board_move_assume_legal(&mut board, e5, P_WHITE);
        let next_pla = kata_game::board::get_opp(next_pla);

        let include_owner_map = true;
        for sym in [0, 1, 2, 4] {
            let out = evaluate_smoke(&nn_eval, &board, &hist, next_pla, sym, include_owner_map);
            assert_eq!(out.nn_x_len, 19);
            assert!(out.white_owner_map.is_some());
        }
        let averaged = full_symmetry_output(&board, &hist, next_pla, include_owner_map, &nn_eval);
        assert!(averaged.white_owner_map.is_some());
    }

    {
        let mut board = Board::parse_board(
            8,
            5,
            ".x.xx...
xxxo.x..
xoo.oxxx
xo..ooxo
oo...ooo
",
            '\n',
        )
        .expect("valid board");
        let next_pla = P_WHITE;
        let rules = Rules::parse_rules("Japanese").expect("valid rules");
        let mut hist = BoardHistory::new(board.clone(), next_pla, rules, 0);
        let e4 = location::of_string("E4", board.x_size, board.y_size).expect("valid loc");
        hist.make_board_move_assume_legal(&mut board, e4, P_WHITE);
        let next_pla = kata_game::board::get_opp(next_pla);

        let include_owner_map = true;
        for sym in [0, 1, 2, 4] {
            let out = evaluate_smoke(&nn_eval, &board, &hist, next_pla, sym, include_owner_map);
            assert_eq!(out.nn_x_len, 19);
            assert!(out.white_owner_map.is_some());
        }
        let averaged = full_symmetry_output(&board, &hist, next_pla, include_owner_map, &nn_eval);
        assert!(averaged.white_owner_map.is_some());
    }

    {
        let mut board = Board::new(19, 19);
        let next_pla = P_BLACK;
        let mut rules = Rules::parse_rules("Japanese").expect("valid rules");
        rules.komi = -8; // C++ stores komi in points; Rust stores half-points.
        let mut hist = BoardHistory::new(board.clone(), next_pla, rules, 0);
        for mv in ["A1", "B2", "C3", "D4", "E5"] {
            let loc = location::of_string(mv, board.x_size, board.y_size).expect("valid loc");
            let pla = hist.presumed_next_move_pla;
            hist.make_board_move_assume_legal(&mut board, loc, pla);
        }
        let next_pla = kata_game::board::get_opp(hist.presumed_next_move_pla);

        let include_owner_map = true;
        let averaged = full_symmetry_output(&board, &hist, next_pla, include_owner_map, &nn_eval);
        assert!(averaged.white_owner_map.is_some());
    }
}

#[test]
fn very_low_visit_discretization_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);

    let mut board = Board::parse_board(
        9,
        9,
        ".........
.........
.........
.........
.........
.........
.........
.........
.........
",
        '\n',
    )
    .expect("valid board");
    let mut rules = Rules::parse_rules("Chinese").expect("valid rules");
    rules.friendly_pass_ok = false;
    let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
    let e5 = location::of_string("E5", board.x_size, board.y_size).expect("valid loc");
    hist.make_board_move_assume_legal(&mut board, e5, P_BLACK);
    let next_pla = P_WHITE;

    let mut params = make_v2_params();
    params.max_visits = 20;
    params.root_symmetry_pruning = true;
    params.root_fpu_reduction_max = 0.0;
    let mut bot = AsyncBot::new(params, &nn_eval, &logger, "testSearchSeedV9_8");
    run_bot_on_position(
        &mut bot,
        board,
        next_pla,
        hist,
        TestSearchOptions::default(),
    );
}

#[test]
fn more_passing_hack_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);

    let mut board = Board::parse_board(
        9,
        9,
        ".o....xo.
.ooooooo.
.xo....ox
ooooooooo
xxxxxxxxx
.o..x....
..ooxxxxx
.oxxxo.oo
.x.......
",
        '\n',
    )
    .expect("valid board");
    let mut rules = Rules::parse_rules("Chinese").expect("valid rules");
    rules.friendly_pass_ok = false;
    rules.komi = 15; // 7.5 in half-points.
    let mut hist = BoardHistory::new(board.clone(), P_WHITE, rules, 0);
    hist.make_board_move_assume_legal(&mut board, PASS_LOC, P_WHITE);
    let next_pla = P_BLACK;

    let opts = TestSearchOptions {
        print_more_more_more: true,
        print_ending_score_value_bonus: true,
        print_play_selection_values: true,
        ..Default::default()
    };

    let mut params = make_v2_params();
    let mut bot = AsyncBot::new(params.clone(), &nn_eval, &logger, "testSearchSeedV9_9");

    for enable_more_passing_hacks in [false, true] {
        params.enable_more_passing_hacks = enable_more_passing_hacks;
        for max_visits in [5, 20, 100] {
            params.max_visits = max_visits;
            bot.set_params(&params);
            run_bot_on_position(&mut bot, board.clone(), next_pla, hist.clone(), opts);
        }
    }
}

#[test]
fn more_passing_hack_2_smoke() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);

    let mut board = Board::parse_board(
        9,
        9,
        ".o....xo.
.ooooooo.
.x.....ox
ooooooooo
xxxxxxxxx
xo..x....
x.ooxxxxx
.oxxxo.ox
.xx.xx..x
",
        '\n',
    )
    .expect("valid board");
    let mut rules = Rules::parse_rules("Chinese").expect("valid rules");
    rules.friendly_pass_ok = false;
    rules.komi = 15; // 7.5 in half-points.
    let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
    hist.make_board_move_assume_legal(&mut board, PASS_LOC, P_BLACK);
    let next_pla = P_WHITE;

    let opts = TestSearchOptions {
        print_more_more_more: true,
        print_ending_score_value_bonus: true,
        print_play_selection_values: true,
        ..Default::default()
    };

    let mut params = make_v2_params();
    let mut bot = AsyncBot::new(params.clone(), &nn_eval, &logger, "testSearchSeedV9_10");

    for enable_more_passing_hacks in [false, true] {
        params.enable_more_passing_hacks = enable_more_passing_hacks;
        for max_visits in [5, 20, 100] {
            params.max_visits = max_visits;
            bot.set_params(&params);
            run_bot_on_position(&mut bot, board.clone(), next_pla, hist.clone(), opts);
        }
    }
}
