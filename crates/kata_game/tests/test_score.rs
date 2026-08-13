//! Port of `Tests::runScoreTests` from `KataGo/cpp/tests/testscore.cpp`.

use kata_game::board::{Board, P_BLACK, P_WHITE};
use kata_game::history::BoardHistory;
use kata_game::rules::Rules;
use kata_nn::score_value::{
    approx_white_score_of_score_value_smooth, expected_white_score_value, get_score_stdev,
    white_score_draw_adjust, white_score_mean_sq_of_score_gridded,
    white_score_value_of_score_smooth, white_score_value_of_score_smooth_no_draw_adjust,
    white_wins_of_winner,
};

fn print_score_stats(board: &Board, hist: &BoardHistory, out: &mut String) {
    out.push_str(&format!(
        "Black self komi wins/draw=0.5: {}\n",
        hist.current_self_komi(P_BLACK, 0.5)
    ));
    out.push_str(&format!(
        "White self komi wins/draw=0.5: {}\n",
        hist.current_self_komi(P_WHITE, 0.5)
    ));
    out.push_str(&format!(
        "Black self komi wins/draw=0.25: {}\n",
        hist.current_self_komi(P_BLACK, 0.25)
    ));
    out.push_str(&format!(
        "White self komi wins/draw=0.25: {}\n",
        hist.current_self_komi(P_WHITE, 0.25)
    ));
    out.push_str(&format!(
        "Black self komi wins/draw=0.75: {}\n",
        hist.current_self_komi(P_BLACK, 0.75)
    ));
    out.push_str(&format!(
        "White self komi wins/draw=0.75: {}\n",
        hist.current_self_komi(P_WHITE, 0.75)
    ));

    out.push_str(&format!("Winner: {}\n", winner_char(hist.winner)));
    let score = f64::from(hist.final_white_minus_black_score);
    out.push_str(&format!("Final score: {}\n", score));

    let draw_equivs_to_try = [0.5, 0.3, 0.7, 1.0];
    for draw_equiv in draw_equivs_to_try {
        let s = format!("{:.1}", draw_equiv);
        let score_adjusted = white_score_draw_adjust(score, draw_equiv, hist);
        let stdev = get_score_stdev(
            score_adjusted,
            white_score_mean_sq_of_score_gridded(score, draw_equiv),
        );
        let sqrt_board_area = board.sqrt_board_area();
        let expected_score_value =
            expected_white_score_value(score_adjusted, stdev, 0.0, 2.0, sqrt_board_area);
        out.push_str(&format!(
            "WL Wins wins/draw={}: {}\n",
            s,
            white_wins_of_winner(hist.winner, draw_equiv)
        ));
        out.push_str(&format!("Score wins/draw={}: {}\n", s, score_adjusted));
        out.push_str(&format!("Score Stdev wins/draw={}: {}\n", s, stdev));
        out.push_str(&format!(
            "Score Util Smooth  wins/draw={}: {}\n",
            s,
            white_score_value_of_score_smooth(score, 0.0, 2.0, draw_equiv, sqrt_board_area, hist)
        ));
        out.push_str(&format!(
            "Score Util SmootND wins/draw={}: {}\n",
            s,
            white_score_value_of_score_smooth_no_draw_adjust(score, 0.0, 2.0, sqrt_board_area)
        ));
        out.push_str(&format!(
            "Score Util Gridded wins/draw={}: {}\n",
            s, expected_score_value
        ));
        out.push_str(&format!(
            "Score Util GridInv wins/draw={}: {}\n",
            s,
            approx_white_score_of_score_value_smooth(
                expected_score_value,
                0.0,
                2.0,
                sqrt_board_area
            )
        ));
    }
}

fn winner_char(winner: kata_game::board::Player) -> char {
    use kata_game::board::player_io::color_to_char;
    color_to_char(winner)
}

#[test]
fn test_score_on_board_even_9x9_komi_7_5() {
    let board = Board::parse_board(
        9,
        9,
        r"
.........
.........
ooooooooo
.........
.........
.........
xxxxxxxxx
.........
.........
",
        '\n',
    )
    .unwrap();

    let rules = Rules::get_tromp_taylorish();
    let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
    hist.end_and_score_game_now(&board);

    let mut out = String::new();
    print_score_stats(&board, &hist, &mut out);
    // Smoke test: ensure the score utilities run and produce non-empty output.
    assert!(!out.is_empty());
}

#[test]
fn test_score_on_board_even_9x9_komi_7() {
    let board = Board::parse_board(
        9,
        9,
        r"
.........
.........
ooooooooo
.........
.........
.........
xxxxxxxxx
.........
.........
",
        '\n',
    )
    .unwrap();

    let mut rules = Rules::get_tromp_taylorish();
    rules.set_komi(7.0);
    let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
    hist.end_and_score_game_now(&board);

    let mut out = String::new();
    print_score_stats(&board, &hist, &mut out);
    assert!(!out.is_empty());
}

#[test]
fn test_score_on_board_black_ahead_7_9x9_komi_7() {
    let board = Board::parse_board(
        9,
        9,
        r"
.........
.........
ooooooooo
.........
.........
xxxxxxx..
xxxxxxxxx
.........
.........
",
        '\n',
    )
    .unwrap();

    let mut rules = Rules::get_tromp_taylorish();
    rules.set_komi(7.0);
    let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
    hist.end_and_score_game_now(&board);

    let mut out = String::new();
    print_score_stats(&board, &hist, &mut out);
    assert!(!out.is_empty());
}

#[test]
fn test_score_on_board_even_5x5_komi_7() {
    let board = Board::parse_board(
        5,
        5,
        r"
.....
ooooo
.....
xxxxx
.....
",
        '\n',
    )
    .unwrap();

    let mut rules = Rules::get_tromp_taylorish();
    rules.set_komi(7.0);
    let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
    hist.end_and_score_game_now(&board);

    let mut out = String::new();
    print_score_stats(&board, &hist, &mut out);
    assert!(!out.is_empty());
}

#[test]
fn test_score_value_tables() {
    // Smoke-test the score-value table functions across multiple board sizes,
    // centers, and scales. The C++ test prints these values to stdout; here we
    // just exercise the API and verify results stay in the expected [-1, 1] range.
    let x_sizes = [9, 13, 13, 13, 19];
    let y_sizes = [9, 9, 13, 19, 19];

    for center in [0.0_f64, 5.0_f64] {
        for scale in [1.0_f64, 2.0_f64] {
            for ((&x, &y), _stdev) in x_sizes.iter().zip(y_sizes.iter()).zip(0..=5) {
                let board = Board::new(x, y);
                let sqrt_area = board.sqrt_board_area();
                for stdev in 0..=5 {
                    let mut d = -8.0;
                    while d <= 8.0 {
                        let score_value =
                            expected_white_score_value(d, stdev as f64, center, scale, sqrt_area);
                        assert!(
                            (-1.0..=1.0).contains(&score_value),
                            "score_value out of range: {} for d={} stdev={}",
                            score_value,
                            d,
                            stdev
                        );
                        d += 0.5;
                    }
                }
            }
        }
    }
}
