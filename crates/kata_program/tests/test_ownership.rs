//! Integration tests ported from `KataGo/cpp/tests/testownership.cpp`.
//!
//! These tests use the dummy neural-net evaluator, so the exact C++ ownership
//! values are skipped in favour of smoke-test assertions that the API returns
//! plausible ownership values.

use std::sync::Arc;

use kata_core::config::ConfigParser;
use kata_core::logger::{Logger, LoggerOptions};
use kata_game::board::{Board, P_BLACK};
use kata_game::history::BoardHistory;
use kata_game::rules::Rules;
use kata_nn::backend::Enabled;
use kata_nn::eval::NnEvaluator;
use kata_program::play_utils::compute_ownership;
use kata_search::params::SearchParams;
use kata_search::search::Search;

fn make_dummy_eval(nn_x_len: i32, nn_y_len: i32) -> NnEvaluator {
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
        false,
        false,
        16,
        12,
        true,
        String::new(),
        Enabled::False,
        1,
        vec![0],
        "ownershipTestRandSeed".to_string(),
        false,
        0,
        true,
        &cfg,
    );
    nn_eval.spawn_server_threads();
    nn_eval
}

fn make_params() -> SearchParams {
    let mut params = SearchParams::for_tests_v1();
    params.value_weight_exponent = 0.0;
    params
}

fn run_on_board(board: &Board, rules: &Rules, nn_eval: &NnEvaluator) -> Vec<f64> {
    let logger = Arc::new(Logger::new(LoggerOptions::default(), None));
    let params = make_params();
    let mut bot = Search::new(params, nn_eval, &logger, "ownershipTest");
    let next_pla = P_BLACK;
    let hist = BoardHistory::new(board.clone(), next_pla, *rules, 0);
    compute_ownership(&mut bot, board, &hist, next_pla, 100)
}

fn assert_ownership_plausible(board: &Board, ownership: &[f64]) {
    assert!(!ownership.is_empty());
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let pos = kata_nn::inputs::nn_pos::xy_to_pos(x, y, board.x_size);
            assert!(
                pos < ownership.len() as i32,
                "ownership vector large enough for board positions"
            );
            let v = ownership[pos as usize];
            assert!(
                (-1.0..=1.0).contains(&v),
                "ownership value {v} at ({x},{y}) out of [-1,1]"
            );
        }
    }
}

#[test]
fn ownership_empty_pattern_smoke() {
    let nn_eval = make_dummy_eval(19, 19);
    let board = Board::parse_board(
        17,
        17,
        ".................
.................
.................
...*....*....*...
.................
.................
.................
.................
...*....*....*...
.................
.................
.................
.................
...*....*....*...
.................
.................
.................\n",
        '\n',
    )
    .expect("valid board");

    let tt_rules = Rules::parse_rules("tromp-taylor").expect("rules");
    let jp_rules = Rules::parse_rules("japanese").expect("rules");

    let ownership_tt = run_on_board(&board, &tt_rules, &nn_eval);
    assert_ownership_plausible(&board, &ownership_tt);

    let ownership_jp = run_on_board(&board, &jp_rules, &nn_eval);
    assert_ownership_plausible(&board, &ownership_jp);
}

#[test]
fn ownership_sparse_position_smoke() {
    let nn_eval = make_dummy_eval(19, 19);
    let board = Board::parse_board(
        17,
        17,
        ".............xo.x
...........o.xoo.
...o...o..o.x.xo.
.oxo..o.*.ox.*xo.
.xoo.....oxx..xx.
.........o.......
...o....o.x..x...
.........ox......
...o....*ox..*...
.........ox......
..o.o...ox...x...
....xo..ox.......
...o.o..ox....x..
...*oxxx*x.x.*...
oooooxox.x...xx..
xxxxxooox...oxo..
.oox.o.ox........\n",
        '\n',
    )
    .expect("valid board");

    let tt_rules = Rules::parse_rules("tromp-taylor").expect("rules");
    let jp_rules = Rules::parse_rules("japanese").expect("rules");

    let ownership_tt = run_on_board(&board, &tt_rules, &nn_eval);
    assert_ownership_plausible(&board, &ownership_tt);

    let ownership_jp = run_on_board(&board, &jp_rules, &nn_eval);
    assert_ownership_plausible(&board, &ownership_jp);
}

#[test]
fn ownership_fighting_position_smoke() {
    let nn_eval = make_dummy_eval(19, 19);
    let board = Board::parse_board(
        17,
        17,
        "x.o.......oxx.xo.
xoox..x..xoox.x.o
xo.x....x.x.oxxxx
.ox*...x*xoo.oooo
oox..xxoxxxox....
.xx.xo.oooo.ox...
....xo......o....
....xo........o..
...x.o..*....*...
....xo......o....
.....xo.o.....o..
..x..x...........
.....x.oooooo....
xxxx..xx*x..xoooo
ooooxx...xoooxxxx
x.o.ox..x.oxxxx.o
.xo.ox....oxo.xo.\n",
        '\n',
    )
    .expect("valid board");

    let tt_rules = Rules::parse_rules("tromp-taylor").expect("rules");
    let jp_rules = Rules::parse_rules("japanese").expect("rules");

    let ownership_tt = run_on_board(&board, &tt_rules, &nn_eval);
    assert_ownership_plausible(&board, &ownership_tt);

    let ownership_jp = run_on_board(&board, &jp_rules, &nn_eval);
    assert_ownership_plausible(&board, &ownership_jp);
}

#[test]
fn ownership_fighting_position_variant_smoke() {
    let nn_eval = make_dummy_eval(19, 19);
    let board = Board::parse_board(
        17,
        17,
        "x.o......xoxx.xo.
xoox..x..xoox.x.o
xo.x....xxxooxxxx
.ox*...x*xoo.oooo
oox..xxoxxxox....
.xx.xoxoooo.ox...
....xoo.....o....
....xo.o......o..
...xxo..*....*...
....xo......o....
.....xo.o.....o..
..x..xxo.........
.....x.oooooo....
xxxx..xxoxo.xoooo
ooooxx..xxoooxxxx
x.o.ox.xx.oxxxx.o
.xo.ox.xoooxo.xo.\n",
        '\n',
    )
    .expect("valid board");

    let tt_rules = Rules::parse_rules("tromp-taylor").expect("rules");
    let jp_rules = Rules::parse_rules("japanese").expect("rules");

    let ownership_tt = run_on_board(&board, &tt_rules, &nn_eval);
    assert_ownership_plausible(&board, &ownership_tt);

    let ownership_jp = run_on_board(&board, &jp_rules, &nn_eval);
    assert_ownership_plausible(&board, &ownership_jp);
}

#[test]
fn ownership_complex_position_smoke() {
    let nn_eval = make_dummy_eval(19, 19);
    let board = Board::parse_board(
        17,
        17,
        "....oxx......xxo.
..oxoox......xoxx
..oo.ox..x.x.xoxx
...*oox.*....xooo
..o.oxxx.....xo..
....oox.xxxxxxox.
ooooox.xooxooo.o.
xoxxxxxo.oo...o..
xx.x..xo*...o*...
..x...xo.....o...
.....xxoooooo.o..
...x.xoo.oxxxo.o.
......xooxx..xoo.
xxxxxxxox.x.xxxoo
xoooooxxx...xoxxo
ooxx.oxox.x....xx
o.xx.ox..........\n",
        '\n',
    )
    .expect("valid board");

    let tt_rules = Rules::parse_rules("tromp-taylor").expect("rules");
    let jp_rules = Rules::parse_rules("japanese").expect("rules");

    let ownership_tt = run_on_board(&board, &tt_rules, &nn_eval);
    assert_ownership_plausible(&board, &ownership_tt);

    let ownership_jp = run_on_board(&board, &jp_rules, &nn_eval);
    assert_ownership_plausible(&board, &ownership_jp);
}
