//! Port of `Tests::runRulesTests` from `KataGo/cpp/tests/testrules.cpp`.
//!
//! Covers area rules, territory rules, ko rules, and game-end scoring.

mod common;

use kata_core::hash::Hash128;
use kata_core::test::expect_lines_match;
use kata_game::board::player_io::{color_to_char, player_to_string};
use kata_game::board::{
    Board, C_BLACK, C_EMPTY, Loc, P_BLACK, P_WHITE, PASS_LOC, Player, location,
};
use kata_game::history::BoardHistory;
use kata_game::rules::{KoRule, Rules, ScoringRule, TaxRule};

fn print_board(out: &mut String, board: &Board) {
    // Note: C++ prints the board's zobrist hash here, but Rust uses a
    // different deterministic zobrist seed, so the hash value would not
    // match. We omit the hash and verify the board contents.
    let x_chars = "ABCDEFGHJKLMNOPQRSTUVWXYZ";
    out.push_str("  ");
    for x in 0..board.x_size {
        out.push(' ');
        out.push(x_chars.chars().nth(x as usize).unwrap());
    }
    out.push('\n');
    for y in 0..board.y_size {
        out.push_str(&format!("{:2} ", board.y_size - y));
        for x in 0..board.x_size {
            let loc = location::get_loc(x, y, board.x_size) as usize;
            out.push(color_to_char(board.colors[loc]));
            if x < board.x_size - 1 {
                out.push(' ');
            }
        }
        out.push('\n');
    }
    out.push_str("\n\n");
}

fn make_move_assert_legal(
    hist: &mut BoardHistory,
    board: &mut Board,
    loc: Loc,
    pla: Player,
    line: u32,
) {
    make_move_assert_legal_with_prevent(hist, board, loc, pla, line, false);
}

fn make_move_assert_legal_with_prevent(
    hist: &mut BoardHistory,
    board: &mut Board,
    loc: Loc,
    pla: Player,
    line: u32,
    prevent_encore: bool,
) {
    let phase_would_end = hist.pass_would_end_phase(board, pla);
    let old_phase = hist.encore_phase;

    assert!(
        hist.is_legal(board, loc, pla),
        "Illegal move on line {}",
        line
    );
    assert!(
        hist.is_legal_tolerant(board, loc, pla),
        "Tolerant illegal move on line {}",
        line
    );
    hist.make_board_move_assume_legal_with_prevent(board, loc, pla, prevent_encore);

    if loc == PASS_LOC {
        let new_phase = hist.encore_phase;
        let actually_ended = new_phase != old_phase || hist.is_game_finished;
        // With territory scoring and prevent_encore, a pass that would normally
        // advance the encore instead leaves the phase unchanged and marks the
        // history as past the normal phase end.
        let would_be_prevented =
            prevent_encore && hist.rules.scoring_rule == ScoringRule::Territory && old_phase < 2;
        let expected_ended = phase_would_end && !would_be_prevented;
        assert!(
            expected_ended == actually_ended,
            "pass_would_end_phase returned different answer than what actually happened after a pass"
        );
    }
}

fn final_score_if_game_ended_now(base_hist: &BoardHistory, base_board: &Board) -> f32 {
    let mut pla = P_BLACK;
    let mut board = base_board.clone();
    let mut hist = base_hist.clone();
    if !hist.move_history.is_empty() {
        pla = kata_game::board::get_opp(hist.move_history[hist.move_history.len() - 1].pla);
    }
    while !hist.is_game_finished {
        hist.make_board_move_assume_legal(&mut board, PASS_LOC, pla);
        pla = kata_game::board::get_opp(pla);
    }

    let score = hist.final_white_minus_black_score;

    hist.end_and_score_game_now(&board);
    assert_eq!(hist.final_white_minus_black_score, score);

    let mut hist2 = base_hist.clone();
    hist2.end_and_score_game_now(base_board);
    assert_eq!(hist2.final_white_minus_black_score, score);

    score
}

fn print_game_result(out: &mut String, hist: &BoardHistory) {
    if !hist.is_game_finished {
        out.push_str("Game is not over\n");
    } else {
        out.push_str(&format!("Winner: {}\n", player_to_string(hist.winner)));
        out.push_str(&format!(
            "W-B Score: {}\n",
            hist.final_white_minus_black_score
        ));
        out.push_str(&format!("isNoResult: {}\n", hist.is_no_result as i32));
        out.push_str(&format!("isResignation: {}\n", hist.is_resignation as i32));
        assert_eq!(
            hist.is_no_result as i32 + hist.is_resignation as i32 + hist.is_scored as i32,
            hist.is_game_finished as i32
        );
    }
}

fn loc_to_string_mach(loc: Loc, x_size: i32) -> String {
    if loc == PASS_LOC {
        return "pass".to_string();
    }
    format!(
        "({},{})",
        location::get_x(loc, x_size),
        location::get_y(loc, x_size)
    )
}

fn print_illegal_moves(out: &mut String, board: &Board, hist: &BoardHistory, pla: Player) {
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = location::get_loc(x, y, board.x_size);
            if board.colors[loc as usize] == C_EMPTY
                && !board.is_illegal_suicide(loc, pla, hist.rules.multi_stone_suicide_legal)
                && !hist.is_legal(board, loc, pla)
            {
                out.push_str(&format!(
                    "Illegal: {} {}\n",
                    loc_to_string_mach(loc, board.x_size),
                    color_to_char(pla)
                ));
            }
            if hist.is_ko_recap_blocked(loc) {
                out.push_str(&format!(
                    "Ko-recap-blocked: {}\n",
                    loc_to_string_mach(loc, board.x_size)
                ));
            }
        }
    }
}

fn print_encore_ko_block(out: &mut String, board: &Board, hist: &BoardHistory) {
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = location::get_loc(x, y, board.x_size);
            if hist.is_ko_recap_blocked(loc) {
                out.push_str(&format!(
                    "Ko recap blocked at {}\n",
                    location::to_string(loc, board.x_size, board.y_size)
                ));
            }
        }
    }
}

#[test]
fn test_rules_area_rules() {
    let mut out = String::new();
    let board = Board::parse_board(
        4,
        4,
        r"
....
....
....
....
",
        '\n',
    )
    .unwrap();
    let rules = Rules {
        ko_rule: KoRule::Positional,
        scoring_rule: ScoringRule::Area,
        komi: 1, // 0.5 in half-points
        multi_stone_suicide_legal: true,
        tax_rule: TaxRule::None,
        ..Rules::default()
    };
    let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
    let mut board = board;
    let xs = board.x_size;

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(1, 1, xs),
        P_BLACK,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(2, 2, xs),
        P_WHITE,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(1, 2, xs),
        P_BLACK,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(2, 1, xs),
        P_WHITE,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(1, 3, xs),
        P_BLACK,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(2, 3, xs),
        P_WHITE,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(1, 0, xs),
        P_BLACK,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(2, 0, xs),
        P_WHITE,
        line!(),
    );
    assert!(!hist.is_game_finished);
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    assert!(!hist.is_game_finished);
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    assert!(hist.is_game_finished);
    assert_eq!(hist.winner, P_WHITE);
    assert_eq!(hist.final_white_minus_black_score, 0.5);
    // Resurrecting the board after game over with another pass
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    assert!(hist.is_game_finished);
    assert_eq!(hist.winner, P_WHITE);
    assert_eq!(hist.final_white_minus_black_score, 0.5);
    // And then some real moves followed by more passes
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(3, 2, xs),
        P_WHITE,
        line!(),
    );
    assert!(!hist.is_game_finished);
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    assert!(!hist.is_game_finished);
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    assert!(hist.is_game_finished);
    assert_eq!(hist.winner, P_WHITE);
    assert_eq!(hist.final_white_minus_black_score, 0.5);
    print_board(&mut out, &board);

    let expected = r#"
   A B C D
 4 . X O .
 3 . X O .
 2 . X O O
 1 . X O .
"#;
    expect_lines_match("Area rules", &out, expected);
}

#[test]
fn test_rules_territory_rules() {
    let mut out = String::new();
    let board = Board::parse_board(
        4,
        4,
        r"
....
....
....
....
",
        '\n',
    )
    .unwrap();
    let rules = Rules {
        ko_rule: KoRule::Positional,
        scoring_rule: ScoringRule::Territory,
        komi: 1, // 0.5 in half-points
        multi_stone_suicide_legal: true,
        tax_rule: TaxRule::Seki,
        ..Rules::default()
    };
    let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
    let mut board = board;
    let xs = board.x_size;

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(1, 1, xs),
        P_BLACK,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(2, 2, xs),
        P_WHITE,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(1, 2, xs),
        P_BLACK,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(2, 1, xs),
        P_WHITE,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(1, 3, xs),
        P_BLACK,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(2, 3, xs),
        P_WHITE,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(1, 0, xs),
        P_BLACK,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(2, 0, xs),
        P_WHITE,
        line!(),
    );
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(3, 2, xs),
        P_WHITE,
        line!(),
    );
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    assert_eq!(hist.encore_phase, 0);
    assert!(!hist.is_game_finished);
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    assert_eq!(hist.encore_phase, 1);
    assert!(!hist.is_game_finished);
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    assert_eq!(hist.encore_phase, 1);
    assert!(!hist.is_game_finished);
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    assert_eq!(hist.encore_phase, 2);
    assert!(!hist.is_game_finished);
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    assert_eq!(hist.encore_phase, 2);
    assert!(!hist.is_game_finished);
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    assert_eq!(hist.encore_phase, 2);
    assert!(hist.is_game_finished);
    assert_eq!(hist.winner, P_BLACK);
    assert_eq!(hist.final_white_minus_black_score, -0.5);
    print_board(&mut out, &board);

    // Resurrecting the board after pass to have black throw in a dead stone, since second encore, should make no difference
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(3, 1, xs),
        P_BLACK,
        line!(),
    );
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    assert_eq!(hist.encore_phase, 2);
    assert!(hist.is_game_finished);
    assert_eq!(hist.winner, P_BLACK);
    assert_eq!(hist.final_white_minus_black_score, -0.5);
    print_board(&mut out, &board);

    // Resurrecting again to have white throw in a junk stone that makes it unclear if black has anything
    // White gets a point for playing, but it's not there second encore, so again no difference
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(0, 1, xs),
        P_WHITE,
        line!(),
    );
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    assert_eq!(hist.encore_phase, 2);
    assert!(hist.is_game_finished);
    assert_eq!(hist.winner, P_WHITE);
    assert_eq!(hist.final_white_minus_black_score, 3.5);
    print_board(&mut out, &board);

    // Resurrecting again to have black solidfy his group and prove it pass-alive
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(0, 2, xs),
        P_BLACK,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(3, 0, xs),
        P_WHITE,
        line!(),
    );
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    // Back to the original result
    assert_eq!(hist.encore_phase, 2);
    assert!(hist.is_game_finished);
    assert_eq!(hist.winner, P_BLACK);
    assert_eq!(hist.final_white_minus_black_score, -0.5);
    print_board(&mut out, &board);

    let expected = r#"
   A B C D
 4 . X O .
 3 . X O .
 2 . X O O
 1 . X O .


   A B C D
 4 . X O .
 3 . X O X
 2 . X O O
 1 . X O .


   A B C D
 4 . X O .
 3 O X O X
 2 . X O O
 1 . X O .


   A B C D
 4 . X O O
 3 O X O .
 2 X X O O
 1 . X O .
"#;
    expect_lines_match("Territory rules", &out, expected);
}

#[test]
fn test_rules_simple_ko() {
    let mut out = String::new();
    let base_board = Board::parse_board(
        6,
        5,
        r"
.o.xxo
oxxxo.
o.x.oo
xx.oo.
oooo.o
",
        '\n',
    )
    .unwrap();
    let base_rules = Rules {
        ko_rule: KoRule::Positional,
        scoring_rule: ScoringRule::Territory,
        komi: 1,
        multi_stone_suicide_legal: false,
        tax_rule: TaxRule::Seki,
        ..Rules::default()
    };

    let mut board = base_board.clone();
    let mut rules = base_rules;
    rules.ko_rule = KoRule::Simple;
    let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
    let xs = board.x_size;

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(5, 1, xs),
        P_BLACK,
        line!(),
    );
    out.push_str("After black ko capture:\n");
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    out.push_str("After black ko capture and one pass:\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);

    assert!(hist.pass_would_end_phase(&board, P_BLACK));
    assert!(!hist.pass_would_end_game(&board, P_BLACK));
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(2, 3, xs),
        P_BLACK,
        line!(),
    );
    assert_eq!(hist.encore_phase, 0);
    assert!(!hist.is_game_finished);

    out.push_str("After black ko capture and one pass and black other move:\n");
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(5, 0, xs),
        P_WHITE,
        line!(),
    );
    out.push_str("White recapture:\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(3, 2, xs),
        P_BLACK,
        line!(),
    );

    out.push_str("Beginning sending two returning one cycle\n");
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(2, 0, xs),
        P_WHITE,
        line!(),
    );
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(0, 0, xs),
        P_BLACK,
        line!(),
    );
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(1, 0, xs),
        P_WHITE,
        line!(),
    );
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(2, 0, xs),
        P_WHITE,
        line!(),
    );
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(0, 0, xs),
        P_BLACK,
        line!(),
    );
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(1, 0, xs),
        P_WHITE,
        line!(),
    );
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    assert_eq!(hist.encore_phase, 0);
    assert!(!hist.is_game_finished);

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    assert_eq!(hist.encore_phase, 1);
    assert!(!hist.is_game_finished);

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    assert_eq!(hist.encore_phase, 2);
    print_game_result(&mut out, &hist);

    let expected = r#"
After black ko capture:
Illegal: (5,0) O
After black ko capture and one pass:
After black ko capture and one pass and black other move:
White recapture:
Illegal: (5,1) X
Beginning sending two returning one cycle
Winner: Black
W-B Score: -1.5
isNoResult: 0
isResignation: 0
"#;
    expect_lines_match("Simple ko rules", &out, expected);
}

#[test]
fn test_rules_positional_ko() {
    let mut out = String::new();
    let base_board = Board::parse_board(
        6,
        5,
        r"
.o.xxo
oxxxo.
o.x.oo
xx.oo.
oooo.o
",
        '\n',
    )
    .unwrap();
    let base_rules = Rules {
        ko_rule: KoRule::Positional,
        scoring_rule: ScoringRule::Territory,
        komi: 1,
        multi_stone_suicide_legal: false,
        tax_rule: TaxRule::Seki,
        ..Rules::default()
    };

    let mut board = base_board.clone();
    let mut rules = base_rules;
    rules.ko_rule = KoRule::Positional;
    let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
    let xs = board.x_size;

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(5, 1, xs),
        P_BLACK,
        line!(),
    );
    out.push_str("After black ko capture:\n");
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    out.push_str("After black ko capture and one pass:\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);

    let mut tmpboard = board.clone();
    let mut tmphist = hist.clone();
    make_move_assert_legal(&mut tmphist, &mut tmpboard, PASS_LOC, P_BLACK, line!());
    assert_eq!(tmphist.encore_phase, 1);
    assert!(!tmphist.is_game_finished);

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(3, 2, xs),
        P_BLACK,
        line!(),
    );
    out.push_str("Beginning sending two returning one cycle\n");

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(2, 0, xs),
        P_WHITE,
        line!(),
    );
    out.push_str("After white sends two?\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(0, 0, xs),
        P_BLACK,
        line!(),
    );
    out.push_str("Can white recapture?\n");
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(5, 0, xs),
        P_WHITE,
        line!(),
    );
    out.push_str("After white recaptures the other ko instead\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    out.push_str("After white recaptures the other ko instead and black passes\n");
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(1, 0, xs),
        P_WHITE,
        line!(),
    );
    out.push_str("After white now returns 1\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    out.push_str("After white now returns 1 and black passes\n");
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(2, 0, xs),
        P_WHITE,
        line!(),
    );
    out.push_str("After white sends 2 again\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    assert_eq!(hist.encore_phase, 0);
    assert!(!hist.is_game_finished);
    assert_eq!(hist.num_consec_valid_turns_this_game, 10);

    assert!(!hist.is_legal(&board, location::get_loc(0, 0, xs), P_BLACK));
    assert!(hist.is_legal_tolerant(&board, location::get_loc(0, 0, xs), P_BLACK));
    hist.make_board_move_assume_legal(&mut board, location::get_loc(0, 0, xs), P_BLACK);
    assert_eq!(hist.num_consec_valid_turns_this_game, 0);

    let expected = r#"
After black ko capture:
Illegal: (5,0) O
After black ko capture and one pass:
Beginning sending two returning one cycle
After white sends two?
Can white recapture?
Illegal: (1,0) O
After white recaptures the other ko instead
Illegal: (5,1) X
After white recaptures the other ko instead and black passes
After white now returns 1
Illegal: (5,1) X
After white now returns 1 and black passes
After white sends 2 again
Illegal: (0,0) X
Illegal: (5,1) X
"#;
    expect_lines_match("Positional ko rules", &out, expected);
}

#[test]
fn test_rules_situational_ko() {
    let mut out = String::new();
    let base_board = Board::parse_board(
        6,
        5,
        r"
.o.xxo
oxxxo.
o.x.oo
xx.oo.
oooo.o
",
        '\n',
    )
    .unwrap();
    let base_rules = Rules {
        ko_rule: KoRule::Positional,
        scoring_rule: ScoringRule::Territory,
        komi: 1,
        multi_stone_suicide_legal: false,
        tax_rule: TaxRule::Seki,
        ..Rules::default()
    };

    let mut board = base_board.clone();
    let mut rules = base_rules;
    rules.ko_rule = KoRule::Situational;
    let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
    let xs = board.x_size;

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(5, 1, xs),
        P_BLACK,
        line!(),
    );
    out.push_str("After black ko capture:\n");
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    out.push_str("After black ko capture and one pass:\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);

    let mut tmpboard = board.clone();
    let mut tmphist = hist.clone();
    make_move_assert_legal(&mut tmphist, &mut tmpboard, PASS_LOC, P_BLACK, line!());
    assert_eq!(tmphist.encore_phase, 1);
    assert!(!tmphist.is_game_finished);

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(3, 2, xs),
        P_BLACK,
        line!(),
    );
    out.push_str("Beginning sending two returning one cycle\n");

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(2, 0, xs),
        P_WHITE,
        line!(),
    );
    out.push_str("After white sends two?\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(0, 0, xs),
        P_BLACK,
        line!(),
    );
    out.push_str("Can white recapture?\n");
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(5, 0, xs),
        P_WHITE,
        line!(),
    );
    out.push_str("After white recaptures the other ko instead\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    out.push_str("After white recaptures the other ko instead and black passes\n");
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(1, 0, xs),
        P_WHITE,
        line!(),
    );
    out.push_str("After white now returns 1\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    out.push_str("After white now returns 1 and black passes\n");
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(2, 0, xs),
        P_WHITE,
        line!(),
    );
    out.push_str("After white sends 2 again\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    assert_eq!(hist.encore_phase, 0);
    assert!(!hist.is_game_finished);

    let expected = r#"
After black ko capture:
Illegal: (5,0) O
After black ko capture and one pass:
Beginning sending two returning one cycle
After white sends two?
Can white recapture?
After white recaptures the other ko instead
Illegal: (5,1) X
After white recaptures the other ko instead and black passes
After white now returns 1
Illegal: (5,1) X
After white now returns 1 and black passes
After white sends 2 again
Illegal: (0,0) X
"#;
    expect_lines_match("Situational ko rules", &out, expected);
}

#[test]
fn test_rules_spight_ko() {
    let mut out = String::new();
    let base_board = Board::parse_board(
        6,
        5,
        r"
.o.xxo
oxxxo.
o.x.oo
xx.oo.
oooo.o
",
        '\n',
    )
    .unwrap();
    let base_rules = Rules {
        ko_rule: KoRule::Positional,
        scoring_rule: ScoringRule::Territory,
        komi: 1,
        multi_stone_suicide_legal: false,
        tax_rule: TaxRule::Seki,
        ..Rules::default()
    };

    let mut board = base_board.clone();
    let suc = board.set_stone(location::get_loc(2, 3, board.x_size), C_BLACK);
    assert!(suc);
    let mut rules = base_rules;
    rules.ko_rule = KoRule::Spight;
    let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
    let xs = board.x_size;

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(5, 1, xs),
        P_BLACK,
        line!(),
    );
    out.push_str("After black ko capture:\n");
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    out.push_str("After black ko capture and one pass:\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);

    let mut tmpboard = board.clone();
    let mut tmphist = hist.clone();
    make_move_assert_legal(&mut tmphist, &mut tmpboard, PASS_LOC, P_BLACK, line!());
    assert_eq!(tmphist.encore_phase, 0);
    assert!(!tmphist.is_game_finished);
    out.push_str("If black were to pass as well??\n");
    print_illegal_moves(&mut out, &tmpboard, &tmphist, P_WHITE);

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(3, 2, xs),
        P_BLACK,
        line!(),
    );
    out.push_str("Beginning sending two returning one cycle\n");

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(2, 0, xs),
        P_WHITE,
        line!(),
    );
    out.push_str("After white sends two?\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(0, 0, xs),
        P_BLACK,
        line!(),
    );
    out.push_str("Can white recapture?\n");
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(5, 0, xs),
        P_WHITE,
        line!(),
    );
    out.push_str("After white recaptures the other ko instead\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    out.push_str("After white recaptures the other ko instead and black passes\n");
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(1, 0, xs),
        P_WHITE,
        line!(),
    );
    out.push_str("After white now returns 1\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    out.push_str("After white now returns 1 and black passes\n");
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(2, 0, xs),
        P_WHITE,
        line!(),
    );
    out.push_str("After white sends 2 again\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(0, 0, xs),
        P_BLACK,
        line!(),
    );
    out.push_str("Can white recapture?\n");
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    out.push_str("After pass\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    assert_eq!(hist.encore_phase, 0);
    assert!(!hist.is_game_finished);

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    out.push_str("After pass\n");
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    assert_eq!(hist.encore_phase, 1);
    assert!(!hist.is_game_finished);

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    assert_eq!(hist.encore_phase, 2);
    print_game_result(&mut out, &hist);

    let expected = r#"
After black ko capture:
Illegal: (5,0) O
After black ko capture and one pass:
If black were to pass as well??
Beginning sending two returning one cycle
After white sends two?
Can white recapture?
Illegal: (1,0) O
After white recaptures the other ko instead
Illegal: (5,1) X
After white recaptures the other ko instead and black passes
After white now returns 1
After white now returns 1 and black passes
After white sends 2 again
Can white recapture?
Illegal: (1,0) O
After pass
After pass
Winner: Black
W-B Score: -2.5
isNoResult: 0
isResignation: 0
"#;
    expect_lines_match("Spight ko rules", &out, expected);
}

#[test]
fn test_rules_triple_ko_encore() {
    let mut out = String::new();
    let board = Board::parse_board(
        7,
        6,
        r"
ooooooo
oxo.o.o
x.xoxox
xxxxxxx
ooooooo
.......
",
        '\n',
    )
    .unwrap();
    let rules = Rules {
        ko_rule: KoRule::Positional,
        scoring_rule: ScoringRule::Territory,
        komi: 1,
        multi_stone_suicide_legal: false,
        tax_rule: TaxRule::Seki,
        ..Rules::default()
    };
    let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
    let mut board = board;
    let xs = board.x_size;

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(3, 1, xs),
        P_BLACK,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(1, 2, xs),
        P_WHITE,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(5, 1, xs),
        P_BLACK,
        line!(),
    );
    // Pass for ko
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(3, 2, xs),
        P_WHITE,
        line!(),
    );
    // Should be a complete capture
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(1, 1, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);

    let expected = r#"
   A B C D E F G
 6 . . . . . . .
 5 . X . X . X .
 4 X . X . X . X
 3 X X X X X X X
 2 O O O O O O O
 1 . . . . . . .


Ko recap blocked at F5
"#;
    expect_lines_match("Triple ko encore", &out, expected);
}

#[test]
fn test_rules_encore_own_throwin_no_clear_ko_recap_block() {
    let mut out = String::new();
    let board = Board::parse_board(
        7,
        6,
        r"
..o....
...o...
.xoxo..
..x.x..
...x...
.......
",
        '\n',
    )
    .unwrap();
    let rules = Rules {
        ko_rule: KoRule::Positional,
        scoring_rule: ScoringRule::Territory,
        komi: 1,
        multi_stone_suicide_legal: false,
        tax_rule: TaxRule::Seki,
        ..Rules::default()
    };
    let mut hist = BoardHistory::new(board.clone(), P_WHITE, rules, 0);
    let mut board = board;
    let xs = board.x_size;

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(3, 3, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(2, 1, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(1, 1, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);

    let expected = r#"
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


Ko recap blocked at D3
   A B C D E F G
 6 . . O . . . .
 5 . . X O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


Ko recap blocked at D3
   A B C D E F G
 6 . . O . . . .
 5 . O . O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


Ko recap blocked at D3
"#;
    expect_lines_match(
        "Encore - own throwin that temporarily breaks the ko shape should not clear the ko recap block",
        &out,
        expected,
    );
}

#[test]
fn test_rules_encore_ko_recap_block_no_stop_non_ko_capture() {
    let mut out = String::new();
    let board = Board::parse_board(
        7,
        6,
        r"
..o....
...o...
.xoxo..
..x.x..
...x...
.......
",
        '\n',
    )
    .unwrap();
    let rules = Rules {
        ko_rule: KoRule::Positional,
        scoring_rule: ScoringRule::Territory,
        komi: 1,
        multi_stone_suicide_legal: false,
        tax_rule: TaxRule::Seki,
        ..Rules::default()
    };
    let mut hist = BoardHistory::new(board.clone(), P_WHITE, rules, 0);
    let mut board = board;
    let xs = board.x_size;

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(3, 3, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(2, 1, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(0, 0, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(3, 2, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);

    let expected = r#"
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


Ko recap blocked at D3
   A B C D E F G
 6 . . O . . . .
 5 . . X O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


Ko recap blocked at D3
   A B C D E F G
 6 O . O . . . .
 5 . . X O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


Ko recap blocked at D3
   A B C D E F G
 6 O . O . . . .
 5 . . X O . . .
 4 . X . X O . .
 3 . . X . X . .
 2 . . . X . . .
 1 . . . . . . .
"#;
    expect_lines_match(
        "Encore - ko recap block does not stop non-ko-capture",
        &out,
        expected,
    );
}

#[test]
fn test_rules_encore_once_only_no_prevent_opponent_fill_ko() {
    let mut out = String::new();
    let board = Board::parse_board(
        7,
        6,
        r"
..o....
...o...
.xoxo..
..x.x..
...x...
.......
",
        '\n',
    )
    .unwrap();
    let rules = Rules {
        ko_rule: KoRule::Positional,
        scoring_rule: ScoringRule::Territory,
        komi: 1,
        multi_stone_suicide_legal: false,
        tax_rule: TaxRule::Seki,
        ..Rules::default()
    };
    let mut hist = BoardHistory::new(board.clone(), P_WHITE, rules, 0);
    let mut board = board;
    let xs = board.x_size;

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(3, 3, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    // Pass for ko
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(3, 2, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    // Pass
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    // Take ko
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(3, 2, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    // Pass
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    // Fill ko
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(3, 3, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);

    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(1, 3, xs),
        P_WHITE,
        line!(),
    );
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(2, 4, xs),
        P_WHITE,
        line!(),
    );
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(3, 5, xs),
        P_WHITE,
        line!(),
    );
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(4, 4, xs),
        P_WHITE,
        line!(),
    );
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(5, 3, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);

    let expected = r#"
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


Ko recap blocked at D3
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O X O . .
 3 . . X . X . .
 2 . . . X . . .
 1 . . . . . . .


Ko recap blocked at D4
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O X O . .
 3 . . X . X . .
 2 . . . X . . .
 1 . . . . . . .


Ko recap blocked at D4
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O X O . .
 3 . . X X X . .
 2 . . . X . . .
 1 . . . . . . .


Ko recap blocked at D4
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O X O . .
 3 . O X X X . .
 2 . . O X O . .
 1 . . . O . . .


Ko recap blocked at D4
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O . O . .
 3 . O . . . O .
 2 . . O . O . .
 1 . . . O . . .
"#;
    expect_lines_match(
        "Encore - once only rule doesn't prevent the opponent moving there (filling ko)",
        &out,
        expected,
    );
}

#[test]
fn test_rules_area_scoring_main_phase() {
    let mut out = String::new();
    let tax_rules = [TaxRule::None, TaxRule::Seki, TaxRule::All];
    for &tax_rule in &tax_rules {
        let board = Board::parse_board(
            7,
            7,
            r"
ox.ooo.
oxxxxxx
ooooooo
.xoxx..
ooox...
x.oxxxx
.xox...
",
            '\n',
        )
        .unwrap();
        let rules = Rules {
            ko_rule: KoRule::Positional,
            scoring_rule: ScoringRule::Area,
            komi: 1,
            multi_stone_suicide_legal: false,
            tax_rule,
            ..Rules::default()
        };
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
        let mut board = board;
        let xs = board.x_size;

        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(5, 3, xs),
            P_BLACK,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(6, 3, xs),
            P_WHITE,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(6, 4, xs),
            P_BLACK,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(5, 4, xs),
            P_WHITE,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(4, 4, xs),
            P_BLACK,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(0, 3, xs),
            P_WHITE,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(6, 6, xs),
            P_BLACK,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        out.push('\n');
    }

    let expected = r#"
Score: -3.5
Score: -4.5
Score: -3.5
Score: -6.5
Score: -6.5
Score: -6.5
Score: -3.5
Score: -3.5

Score: 0.5
Score: -0.5
Score: 0.5
Score: -5.5
Score: -5.5
Score: -5.5
Score: -3.5
Score: -3.5

Score: 0.5
Score: -0.5
Score: 0.5
Score: -3.5
Score: -3.5
Score: -3.5
Score: -1.5
Score: -1.5
"#;
    expect_lines_match("Area scoring in the main phase", &out, expected);
}

#[test]
fn test_rules_territory_scoring_main_phase() {
    let mut out = String::new();
    let tax_rules = [TaxRule::None, TaxRule::Seki, TaxRule::All];
    for &tax_rule in &tax_rules {
        let board = Board::parse_board(
            7,
            7,
            r"
ox.ooo.
oxxxxxx
ooooooo
.xoxx..
ooox...
x.oxxxx
.xox...
",
            '\n',
        )
        .unwrap();
        let rules = Rules {
            ko_rule: KoRule::Positional,
            scoring_rule: ScoringRule::Territory,
            komi: 1,
            multi_stone_suicide_legal: false,
            tax_rule,
            ..Rules::default()
        };
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
        let mut board = board;
        let xs = board.x_size;

        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(5, 3, xs),
            P_BLACK,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(6, 3, xs),
            P_WHITE,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(6, 4, xs),
            P_BLACK,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(5, 4, xs),
            P_WHITE,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(4, 4, xs),
            P_BLACK,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(0, 3, xs),
            P_WHITE,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(6, 6, xs),
            P_BLACK,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        out.push('\n');
    }

    let expected = r#"
Score: -3.5
Score: -3.5
Score: -3.5
Score: -5.5
Score: -6.5
Score: -5.5
Score: -3.5
Score: -2.5

Score: 0.5
Score: 0.5
Score: 0.5
Score: -4.5
Score: -5.5
Score: -4.5
Score: -3.5
Score: -2.5

Score: 0.5
Score: 0.5
Score: 0.5
Score: -2.5
Score: -3.5
Score: -2.5
Score: -1.5
Score: -0.5
"#;
    expect_lines_match("Territory scoring in the main phase", &out, expected);
}

#[test]
fn test_rules_territory_scoring_encore_1() {
    let mut out = String::new();
    let tax_rules = [TaxRule::None, TaxRule::Seki, TaxRule::All];
    for &tax_rule in &tax_rules {
        let board = Board::parse_board(
            7,
            7,
            r"
ox.ooo.
oxxxxxx
ooooooo
.xoxx..
ooox...
x.oxxxx
.xox...
",
            '\n',
        )
        .unwrap();
        let rules = Rules {
            ko_rule: KoRule::Positional,
            scoring_rule: ScoringRule::Territory,
            komi: 1,
            multi_stone_suicide_legal: false,
            tax_rule,
            ..Rules::default()
        };
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
        let mut board = board;
        let xs = board.x_size;

        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(5, 3, xs),
            P_BLACK,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(6, 3, xs),
            P_WHITE,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(6, 4, xs),
            P_BLACK,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(5, 4, xs),
            P_WHITE,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(4, 4, xs),
            P_BLACK,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(0, 3, xs),
            P_WHITE,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(6, 6, xs),
            P_BLACK,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        out.push('\n');
    }

    let expected = r#"
Score: -3.5
Score: -3.5
Score: -3.5
Score: -5.5
Score: -6.5
Score: -5.5
Score: -3.5
Score: -2.5

Score: 0.5
Score: 0.5
Score: 0.5
Score: -4.5
Score: -5.5
Score: -4.5
Score: -3.5
Score: -2.5

Score: 0.5
Score: 0.5
Score: 0.5
Score: -2.5
Score: -3.5
Score: -2.5
Score: -1.5
Score: -0.5
"#;
    expect_lines_match("Territory scoring in encore 1", &out, expected);
}

#[test]
fn test_rules_territory_scoring_encore_2() {
    let mut out = String::new();
    let tax_rules = [TaxRule::None, TaxRule::Seki, TaxRule::All];
    for &tax_rule in &tax_rules {
        let board = Board::parse_board(
            7,
            7,
            r"
ox.ooo.
oxxxxxx
ooooooo
.xoxx..
ooox...
x.oxxxx
.xox...
",
            '\n',
        )
        .unwrap();
        let rules = Rules {
            ko_rule: KoRule::Positional,
            scoring_rule: ScoringRule::Territory,
            komi: 1,
            multi_stone_suicide_legal: false,
            tax_rule,
            ..Rules::default()
        };
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
        let mut board = board;
        let xs = board.x_size;

        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(5, 3, xs),
            P_BLACK,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(6, 3, xs),
            P_WHITE,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(6, 4, xs),
            P_BLACK,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(5, 4, xs),
            P_WHITE,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(4, 4, xs),
            P_BLACK,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(0, 3, xs),
            P_WHITE,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(6, 6, xs),
            P_BLACK,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        out.push('\n');
    }

    let expected = r#"
Score: -3.5
Score: -3.5
Score: -3.5
Score: -5.5
Score: -5.5
Score: -5.5
Score: -3.5
Score: -3.5

Score: 0.5
Score: 0.5
Score: 0.5
Score: -4.5
Score: -4.5
Score: -4.5
Score: -3.5
Score: -3.5

Score: 0.5
Score: 0.5
Score: 0.5
Score: -2.5
Score: -2.5
Score: -2.5
Score: -1.5
Score: -1.5
"#;
    expect_lines_match("Territory scoring in encore 2", &out, expected);
}

#[test]
fn test_rules_fill_seki_liberties_main_phase() {
    let mut out = String::new();
    let tax_rules = [TaxRule::None, TaxRule::Seki, TaxRule::All];
    for &tax_rule in &tax_rules {
        let board = Board::parse_board(
            7,
            7,
            r"
...oxx.
oooox.x
xxxxoxx
o.xoooo
.oxox.o
oxxo.x.
o.xoo.x
",
            '\n',
        )
        .unwrap();
        let rules = Rules {
            ko_rule: KoRule::Positional,
            scoring_rule: ScoringRule::Territory,
            komi: -1,
            multi_stone_suicide_legal: false,
            tax_rule,
            ..Rules::default()
        };
        let mut hist = BoardHistory::new(board.clone(), P_WHITE, rules, 0);
        let mut board = board;
        let xs = board.x_size;

        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(6, 5, xs),
            P_WHITE,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(5, 6, xs),
            P_BLACK,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(0, 4, xs),
            P_WHITE,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(6, 0, xs),
            P_BLACK,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(1, 0, xs),
            P_WHITE,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(4, 5, xs),
            P_BLACK,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(5, 4, xs),
            P_WHITE,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        out.push('\n');
    }

    let expected = r#"
Score: 1.5
Score: 1.5
Score: 1.5
Score: 0.5
Score: 1.5
Score: 0.5
Score: 0.5
Score: 10.5

Score: 0.5
Score: 0.5
Score: 0.5
Score: 0.5
Score: 2.5
Score: 1.5
Score: 1.5
Score: 11.5

Score: 0.5
Score: 0.5
Score: 0.5
Score: 0.5
Score: 0.5
Score: -0.5
Score: -0.5
Score: 7.5
"#;
    expect_lines_match("Fill seki liberties in main phase", &out, expected);
}

#[test]
fn test_rules_fill_seki_liberties_encore_2() {
    let mut out = String::new();
    let tax_rules = [TaxRule::None, TaxRule::Seki, TaxRule::All];
    for &tax_rule in &tax_rules {
        let board = Board::parse_board(
            7,
            7,
            r"
...oxx.
oooox.x
xxxxoxx
o.xoooo
.oxox.o
oxxo.x.
o.xoo.x
",
            '\n',
        )
        .unwrap();
        let rules = Rules {
            ko_rule: KoRule::Positional,
            scoring_rule: ScoringRule::Territory,
            komi: -1,
            multi_stone_suicide_legal: false,
            tax_rule,
            ..Rules::default()
        };
        let mut hist = BoardHistory::new(board.clone(), P_WHITE, rules, 0);
        let mut board = board;
        let xs = board.x_size;

        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(6, 5, xs),
            P_WHITE,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(5, 6, xs),
            P_BLACK,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(0, 4, xs),
            P_WHITE,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(6, 0, xs),
            P_BLACK,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(1, 0, xs),
            P_WHITE,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(4, 5, xs),
            P_BLACK,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(5, 4, xs),
            P_WHITE,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        out.push('\n');
    }

    let expected = r#"
Score: 1.5
Score: 1.5
Score: 1.5
Score: 0.5
Score: 1.5
Score: 1.5
Score: 1.5
Score: 11.5

Score: 0.5
Score: 0.5
Score: 0.5
Score: 0.5
Score: 2.5
Score: 2.5
Score: 2.5
Score: 12.5

Score: 0.5
Score: 0.5
Score: 0.5
Score: 0.5
Score: 0.5
Score: 0.5
Score: 0.5
Score: 8.5
"#;
    expect_lines_match("Fill seki liberties in encore 2", &out, expected);
}

#[test]
fn test_rules_area_scoring_with_button() {
    let mut out = String::new();
    let button_rules = [false, true];
    for &has_button in &button_rules {
        let board = Board::parse_board(
            7,
            7,
            r"
..x.xo.
..xxoo.
...xo..
..xxo..
..x.o..
..xxo..
...xo..
",
            '\n',
        )
        .unwrap();
        let rules = Rules {
            ko_rule: KoRule::Simple,
            scoring_rule: ScoringRule::Area,
            tax_rule: TaxRule::None,
            multi_stone_suicide_legal: false,
            komi: 5,
            has_button,
            ..Rules::default()
        };
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
        let mut board = board;
        let xs = board.x_size;

        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(3, 4, xs),
            P_BLACK,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(3, 0, xs),
            P_WHITE,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(1, 2, xs),
            P_BLACK,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(4, 0, xs),
            P_WHITE,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(
            &mut hist,
            &mut board,
            location::get_loc(6, 2, xs),
            P_WHITE,
            line!(),
        );
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        assert!(hist.is_game_finished);
        out.push_str(&format!(
            "Score: {}\n",
            final_score_if_game_ended_now(&hist, &board)
        ));
        out.push('\n');
    }

    let expected = r#"
Score: -5.5
Score: -6.5
Score: -2.5
Score: -2.5
Score: -2.5
Score: -2.5
Score: -2.5
Score: -2.5
Score: -2.5

Score: -6
Score: -6
Score: -3
Score: -2
Score: -3
Score: -3
Score: -3
Score: -3
Score: -3
"#;
    expect_lines_match("Area scoring with button", &out, expected);
}

#[test]
fn test_rules_pass_for_ko() {
    let mut out = String::new();
    let board = Board::parse_board(
        7,
        7,
        r"
..ox.oo
..oxxxo
...oox.
....oxx
..o.oo.
.......
.......
",
        '\n',
    )
    .unwrap();
    let rules = Rules {
        ko_rule: KoRule::Positional,
        scoring_rule: ScoringRule::Territory,
        komi: 1,
        multi_stone_suicide_legal: false,
        tax_rule: TaxRule::Seki,
        ..Rules::default()
    };
    let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
    let mut board = board;
    let xs = board.x_size;

    let mut hasha: Hash128;
    let mut hashc: Hash128;
    let mut hashd: Hash128;

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    assert_eq!(hist.encore_phase, 1);
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(6, 2, xs),
        P_WHITE,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(4, 0, xs),
        P_BLACK,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(6, 1, xs),
        P_WHITE,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(6, 0, xs),
        P_BLACK,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(5, 0, xs),
        P_WHITE,
        line!(),
    );
    out.push_str("Black can't retake\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(2, 2, xs),
        P_BLACK,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(1, 2, xs),
        P_WHITE,
        line!(),
    );
    out.push_str("Ko threat shouldn't work in the encore\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(0, 6, xs),
        P_WHITE,
        line!(),
    );
    out.push_str("Regular pass shouldn't work in the encore\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    out.push_str("Pass for ko! (Should not affect the board stones)\n");
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(6, 0, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(0, 5, xs),
        P_WHITE,
        line!(),
    );
    hashd = hist.ko_hash_history[hist.ko_hash_history.len() - 1];
    out.push_str("Now black can retake, and white's retake isn't legal\n");
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(6, 0, xs),
        P_BLACK,
        line!(),
    );
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    hasha = hist.ko_hash_history[hist.ko_hash_history.len() - 1];
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(5, 0, xs),
        P_WHITE,
        line!(),
    );
    let hashb: Hash128 = hist.ko_hash_history[hist.ko_hash_history.len() - 1];
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    hashc = hist.ko_hash_history[hist.ko_hash_history.len() - 1];
    assert_ne!(hasha, hashb);
    assert_ne!(hasha, hashc);
    assert_ne!(hashb, hashc);
    out.push_str("White's retake is legal after passing for ko\n");
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(5, 0, xs),
        P_WHITE,
        line!(),
    );
    out.push_str("Black's retake is illegal again\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(6, 0, xs),
        P_BLACK,
        line!(),
    );
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    assert_eq!(hashd, hist.ko_hash_history[hist.ko_hash_history.len() - 1]);
    out.push_str("And is still illegal due to only-once\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(1, 1, xs),
        P_BLACK,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(2, 3, xs),
        P_WHITE,
        line!(),
    );
    out.push_str("But a ko threat fixes that\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(6, 0, xs),
        P_BLACK,
        line!(),
    );
    out.push_str("White illegal now\n");
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    assert_eq!(hist.encore_phase, 1);
    hasha = hist.ko_hash_history[hist.ko_hash_history.len() - 1];
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    hashc = hist.ko_hash_history[hist.ko_hash_history.len() - 1];
    assert_eq!(hist.encore_phase, 2);
    assert_ne!(hasha, hashb);
    assert_ne!(hasha, hashc);
    assert_ne!(hashb, hashc);
    out.push_str("Legal again in second encore\n");
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(5, 0, xs),
        P_WHITE,
        line!(),
    );
    out.push_str("Lastly, try black ko threat one more time\n");
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(1, 0, xs),
        P_BLACK,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(2, 2, xs),
        P_WHITE,
        line!(),
    );
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    out.push_str("And a pass for ko\n");
    hashd = hist.ko_hash_history[hist.ko_hash_history.len() - 1];
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(6, 0, xs),
        P_BLACK,
        line!(),
    );
    let hashe: Hash128 = hist.ko_hash_history[hist.ko_hash_history.len() - 1];
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    let hashf: Hash128 = hist.ko_hash_history[hist.ko_hash_history.len() - 1];
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    out.push_str("And repeat with white\n");
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(6, 0, xs),
        P_BLACK,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(5, 0, xs),
        P_WHITE,
        line!(),
    );
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(5, 0, xs),
        P_WHITE,
        line!(),
    );
    assert_eq!(hashd, hist.ko_hash_history[hist.ko_hash_history.len() - 1]);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(6, 0, xs),
        P_BLACK,
        line!(),
    );
    assert_eq!(hashe, hist.ko_hash_history[hist.ko_hash_history.len() - 1]);
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    assert_eq!(hashf, hist.ko_hash_history[hist.ko_hash_history.len() - 1]);
    out.push_str("And see the only-once for black\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);

    let expected = r#"
Black can't retake
Ko-recap-blocked: (5,0)
Ko threat shouldn't work in the encore
Ko-recap-blocked: (5,0)
Regular pass shouldn't work in the encore
Ko-recap-blocked: (5,0)
Pass for ko! (Should not affect the board stones)
   A B C D E F G
 7 . . O X X O .
 6 . . O X X X O
 5 . O X O O X .
 4 . . . . O X X
 3 . . O . O O .
 2 . . . . . . .
 1 O . . . . . .


Now black can retake, and white's retake isn't legal
Ko-recap-blocked: (6,0)
White's retake is legal after passing for ko
Black's retake is illegal again
Ko-recap-blocked: (5,0)
And is still illegal due to only-once
Illegal: (6,0) X
But a ko threat fixes that
White illegal now
Ko-recap-blocked: (6,0)
Legal again in second encore
Lastly, try black ko threat one more time
Ko-recap-blocked: (5,0)
And a pass for ko
And repeat with white
And see the only-once for black
Illegal: (6,0) X
"#;
    expect_lines_match("Pass for ko", &out, expected);
}

#[test]
fn test_rules_two_step_ko_in_encore() {
    let mut out = String::new();
    let board = Board::parse_board(
        7,
        5,
        r"
x.x....
.xx....
xox....
ooo....
.......
",
        '\n',
    )
    .unwrap();
    let rules = Rules {
        ko_rule: KoRule::Situational,
        scoring_rule: ScoringRule::Territory,
        komi: 1,
        multi_stone_suicide_legal: true,
        tax_rule: TaxRule::Seki,
        ..Rules::default()
    };
    let mut hist = BoardHistory::new(board.clone(), P_WHITE, rules, 0);
    let mut board = board;
    let xs = board.x_size;

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    assert_eq!(hist.encore_phase, 1);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(0, 1, xs),
        P_WHITE,
        line!(),
    );
    out.push_str("After first cap\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(1, 0, xs),
        P_WHITE,
        line!(),
    );
    out.push_str("After second cap\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(0, 0, xs),
        P_BLACK,
        line!(),
    );
    out.push_str("Just after black pass for ko\n");
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    print_board(&mut out, &board);

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    out.push_str("After another white pass\n");
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(0, 0, xs),
        P_BLACK,
        line!(),
    );
    out.push_str("After first cap\n");
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    print_board(&mut out, &board);
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(0, 2, xs),
        P_BLACK,
        line!(),
    );
    out.push_str("After second pass for ko\n");
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    print_board(&mut out, &board);
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(0, 2, xs),
        P_BLACK,
        line!(),
    );
    out.push_str("After second cap\n");
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    print_board(&mut out, &board);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(0, 1, xs),
        P_WHITE,
        line!(),
    );
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    out.push_str("After pass for ko\n");
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    print_board(&mut out, &board);

    let expected = r#"
After first cap
Ko-recap-blocked: (0,1)
After second cap
Ko-recap-blocked: (1,0)
Ko-recap-blocked: (0,1)
Just after black pass for ko
Ko-recap-blocked: (0,1)
   A B C D E F G
 5 . O X . . . .
 4 O X X . . . .
 3 . O X . . . .
 2 O O O . . . .
 1 . . . . . . .


After another white pass
Ko-recap-blocked: (0,1)
After first cap
Ko-recap-blocked: (0,0)
Ko-recap-blocked: (0,1)
   A B C D E F G
 5 X . X . . . .
 4 O X X . . . .
 3 . O X . . . .
 2 O O O . . . .
 1 . . . . . . .


After second pass for ko
Ko-recap-blocked: (0,0)
   A B C D E F G
 5 X . X . . . .
 4 O X X . . . .
 3 . O X . . . .
 2 O O O . . . .
 1 . . . . . . .


After second cap
Ko-recap-blocked: (0,0)
Ko-recap-blocked: (0,2)
   A B C D E F G
 5 X . X . . . .
 4 . X X . . . .
 3 X O X . . . .
 2 O O O . . . .
 1 . . . . . . .


After pass for ko
Ko-recap-blocked: (0,0)
Illegal: (0,1) O
   A B C D E F G
 5 X . X . . . .
 4 . X X . . . .
 3 X O X . . . .
 2 O O O . . . .
 1 . . . . . . .


"#;
    expect_lines_match("Two step ko in encore", &out, expected);
}

#[test]
fn test_rules_throwin_destroys_ko_momentarily_no_clear_ko_recap_block() {
    let mut out = String::new();
    let board = Board::parse_board(
        7,
        5,
        r"
x......
oxx....
.o.....
oo.....
.......
",
        '\n',
    )
    .unwrap();
    let rules = Rules {
        ko_rule: KoRule::Situational,
        scoring_rule: ScoringRule::Territory,
        komi: 1,
        multi_stone_suicide_legal: true,
        tax_rule: TaxRule::Seki,
        ..Rules::default()
    };
    let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
    let mut board = board;
    let xs = board.x_size;

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    assert_eq!(hist.encore_phase, 2);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(0, 2, xs),
        P_BLACK,
        line!(),
    );
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(1, 0, xs),
        P_WHITE,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(2, 0, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);

    let expected = r#"
Ko-recap-blocked: (0,2)
   A B C D E F G
 5 X . X . . . .
 4 . X X . . . .
 3 X O . . . . .
 2 O O . . . . .
 1 . . . . . . .


Ko-recap-blocked: (0,2)
"#;
    expect_lines_match(
        "Throw in that destroys the ko momentarily does not clear ko recap block",
        &out,
        expected,
    );
}

#[test]
fn test_rules_various_komis() {
    let mut out = String::new();
    let board = Board::parse_board(
        7,
        6,
        r"
.......
.......
ooooooo
xxxxxxx
.......
.......
",
        '\n',
    )
    .unwrap();
    let rules = Rules {
        ko_rule: KoRule::Simple,
        scoring_rule: ScoringRule::Area,
        komi: 1,
        multi_stone_suicide_legal: false,
        tax_rule: TaxRule::None,
        ..Rules::default()
    };
    let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
    let mut board = board;

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    assert!(hist.is_game_finished);
    print_game_result(&mut out, &hist);

    hist.set_komi(0.0);
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    assert!(hist.is_game_finished);
    print_game_result(&mut out, &hist);

    hist.set_komi(-0.5);
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    assert!(hist.is_game_finished);
    print_game_result(&mut out, &hist);

    let expected = r#"
Winner: White
W-B Score: 0.5
isNoResult: 0
isResignation: 0
Winner: Empty
W-B Score: 0
isNoResult: 0
isResignation: 0
Winner: Black
W-B Score: -0.5
isNoResult: 0
isResignation: 0
"#;
    expect_lines_match("Various komis", &out, expected);
}

#[test]
fn test_rules_group_tax_seki_scoring() {
    let mut out = String::new();
    let board = Board::parse_board(
        9,
        9,
        r"
.x.xo.o.x
...xooox.
.xxxxxxoo
xoooooxo.
xo.o.oxoo
xoooooxxx
xxxo...oo
.xxxoooo.
.x.xo.o.o
",
        '\n',
    )
    .unwrap();
    let mut rules = Rules {
        ko_rule: KoRule::Positional,
        multi_stone_suicide_legal: false,
        komi: 1,
        ..Rules::default()
    };

    {
        rules.scoring_rule = ScoringRule::Area;
        rules.tax_rule = TaxRule::None;
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
        let mut board = board.clone();
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        assert!(hist.is_game_finished);
        print_game_result(&mut out, &hist);
    }
    {
        rules.scoring_rule = ScoringRule::Area;
        rules.tax_rule = TaxRule::Seki;
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
        let mut board = board.clone();
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        assert!(hist.is_game_finished);
        print_game_result(&mut out, &hist);
    }
    {
        rules.scoring_rule = ScoringRule::Area;
        rules.tax_rule = TaxRule::All;
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
        let mut board = board.clone();
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        assert!(hist.is_game_finished);
        print_game_result(&mut out, &hist);
    }
    {
        rules.scoring_rule = ScoringRule::Territory;
        rules.tax_rule = TaxRule::None;
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
        let mut board = board.clone();
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        assert!(hist.is_game_finished);
        print_game_result(&mut out, &hist);
    }
    {
        rules.scoring_rule = ScoringRule::Territory;
        rules.tax_rule = TaxRule::Seki;
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
        let mut board = board.clone();
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        assert!(hist.is_game_finished);
        print_game_result(&mut out, &hist);
    }
    {
        rules.scoring_rule = ScoringRule::Territory;
        rules.tax_rule = TaxRule::All;
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
        let mut board = board.clone();
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        assert!(hist.is_game_finished);
        print_game_result(&mut out, &hist);
    }

    let expected = r#"
Winner: White
W-B Score: 4.5
isNoResult: 0
isResignation: 0
Winner: White
W-B Score: 6.5
isNoResult: 0
isResignation: 0
Winner: White
W-B Score: 6.5
isNoResult: 0
isResignation: 0
Winner: Black
W-B Score: -1.5
isNoResult: 0
isResignation: 0
Winner: White
W-B Score: 0.5
isNoResult: 0
isResignation: 0
Winner: White
W-B Score: 0.5
isNoResult: 0
isResignation: 0
"#;
    expect_lines_match("GroupTaxSekiScoring", &out, expected);
}

#[test]
fn test_rules_group_tax_seki_scoring_2() {
    let mut out = String::new();
    let board = Board::parse_board(
        9,
        9,
        r"
.x.xo.o.x
...xooox.
.xxxxxxoo
xoooooxo.
xo.o.oxoo
xoooooxxx
xxxoxxxoo
.xxxoooo.
.x.xo.o.o
",
        '\n',
    )
    .unwrap();
    let mut rules = Rules {
        ko_rule: KoRule::Positional,
        multi_stone_suicide_legal: false,
        komi: 1,
        ..Rules::default()
    };

    {
        rules.scoring_rule = ScoringRule::Area;
        rules.tax_rule = TaxRule::None;
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
        let mut board = board.clone();
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        assert!(hist.is_game_finished);
        print_game_result(&mut out, &hist);
    }
    {
        rules.scoring_rule = ScoringRule::Area;
        rules.tax_rule = TaxRule::Seki;
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
        let mut board = board.clone();
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        assert!(hist.is_game_finished);
        print_game_result(&mut out, &hist);
    }
    {
        rules.scoring_rule = ScoringRule::Area;
        rules.tax_rule = TaxRule::All;
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
        let mut board = board.clone();
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        assert!(hist.is_game_finished);
        print_game_result(&mut out, &hist);
    }
    {
        rules.scoring_rule = ScoringRule::Territory;
        rules.tax_rule = TaxRule::None;
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
        let mut board = board.clone();
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        assert!(hist.is_game_finished);
        print_game_result(&mut out, &hist);
    }
    {
        rules.scoring_rule = ScoringRule::Territory;
        rules.tax_rule = TaxRule::Seki;
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
        let mut board = board.clone();
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        assert!(hist.is_game_finished);
        print_game_result(&mut out, &hist);
    }
    {
        rules.scoring_rule = ScoringRule::Territory;
        rules.tax_rule = TaxRule::All;
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
        let mut board = board.clone();
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        assert!(hist.is_game_finished);
        print_game_result(&mut out, &hist);
    }

    let expected = r#"
Winner: White
W-B Score: 1.5
isNoResult: 0
isResignation: 0
Winner: Black
W-B Score: -0.5
isNoResult: 0
isResignation: 0
Winner: Black
W-B Score: -2.5
isNoResult: 0
isResignation: 0
Winner: Black
W-B Score: -1.5
isNoResult: 0
isResignation: 0
Winner: Black
W-B Score: -3.5
isNoResult: 0
isResignation: 0
Winner: Black
W-B Score: -5.5
isNoResult: 0
isResignation: 0
"#;
    expect_lines_match("GroupTaxSekiScoring2", &out, expected);
}

#[test]
fn test_rules_prevent_encore() {
    let mut out = String::new();
    let board = Board::parse_board(
        3,
        3,
        r"
.x.
xxo
.o.
",
        '\n',
    )
    .unwrap();
    let rules = Rules {
        ko_rule: KoRule::Positional,
        scoring_rule: ScoringRule::Territory,
        komi: 1, // 0.5 in half-points
        multi_stone_suicide_legal: false,
        tax_rule: TaxRule::None,
        ..Rules::default()
    };

    {
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
        let mut board = board.clone();
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        hist.print_debug_info(&mut out, &board);
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        hist.print_debug_info(&mut out, &board);
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        hist.print_debug_info(&mut out, &board);
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
        hist.print_debug_info(&mut out, &board);
        make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
        hist.print_debug_info(&mut out, &board);
        hist.make_board_move_assume_legal(&mut board, PASS_LOC, P_BLACK);
        hist.print_debug_info(&mut out, &board);
        out.push('\n');
    }
    {
        out.push_str("-----------------------\n");
        out.push_str("Preventing encore\n");
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
        let mut board = board.clone();
        make_move_assert_legal_with_prevent(
            &mut hist,
            &mut board,
            PASS_LOC,
            P_BLACK,
            line!(),
            true,
        );
        make_move_assert_legal_with_prevent(
            &mut hist,
            &mut board,
            PASS_LOC,
            P_WHITE,
            line!(),
            true,
        );
        hist.print_debug_info(&mut out, &board);
        make_move_assert_legal_with_prevent(
            &mut hist,
            &mut board,
            PASS_LOC,
            P_BLACK,
            line!(),
            true,
        );
        hist.print_debug_info(&mut out, &board);
        make_move_assert_legal_with_prevent(
            &mut hist,
            &mut board,
            PASS_LOC,
            P_WHITE,
            line!(),
            true,
        );
        hist.print_debug_info(&mut out, &board);
    }

    let expected = r#"
   A B C
 3 . X .
 2 X X O
 1 . O .


Initial pla Black
Encore phase 1
Turns this phase 0
Approx valid turns this phase 0
Approx consec valid turns this game 2
Rules koPOSITIONALscoreTERRITORYtaxNONEsui0komi0.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 1
White handicap bonus score 0
Has button 0
Presumed next pla Black
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves pass pass
   A B C
 3 . X .
 2 X X O
 1 . O .


Initial pla Black
Encore phase 1
Turns this phase 1
Approx valid turns this phase 1
Approx consec valid turns this game 3
Rules koPOSITIONALscoreTERRITORYtaxNONEsui0komi0.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 1
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves pass pass pass
   A B C
 3 . X .
 2 X X O
 1 . O .


Initial pla Black
Encore phase 2
Turns this phase 0
Approx valid turns this phase 0
Approx consec valid turns this game 4
Rules koPOSITIONALscoreTERRITORYtaxNONEsui0komi0.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 1
White handicap bonus score 0
Has button 0
Presumed next pla Black
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves pass pass pass pass
   A B C
 3 . X .
 2 X X O
 1 . O .


Initial pla Black
Encore phase 2
Turns this phase 1
Approx valid turns this phase 1
Approx consec valid turns this game 5
Rules koPOSITIONALscoreTERRITORYtaxNONEsui0komi0.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 1
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves pass pass pass pass pass
   A B C
 3 . X .
 2 X X O
 1 . O .


Initial pla Black
Encore phase 2
Turns this phase 2
Approx valid turns this phase 2
Approx consec valid turns this game 6
Rules koPOSITIONALscoreTERRITORYtaxNONEsui0komi0.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 1
White handicap bonus score 0
Has button 0
Presumed next pla Black
Past normal phase end 0
Game result 1 White 0.5 1 0 0
Last moves pass pass pass pass pass pass
   A B C
 3 . X .
 2 X X O
 1 . O .


Initial pla Black
Encore phase 2
Turns this phase 3
Approx valid turns this phase 2
Approx consec valid turns this game 2
Rules koPOSITIONALscoreTERRITORYtaxNONEsui0komi0.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 1
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 1 White 0.5 1 0 0
Last moves pass pass pass pass pass pass pass

-----------------------
Preventing encore
   A B C
 3 . X .
 2 X X O
 1 . O .


Initial pla Black
Encore phase 0
Turns this phase 2
Approx valid turns this phase 1
Approx consec valid turns this game 1
Rules koPOSITIONALscoreTERRITORYtaxNONEsui0komi0.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 1
White handicap bonus score 0
Has button 0
Presumed next pla Black
Past normal phase end 1
Game result 0 Empty 0 0 0 0
Last moves pass pass
   A B C
 3 . X .
 2 X X O
 1 . O .


Initial pla Black
Encore phase 0
Turns this phase 3
Approx valid turns this phase 1
Approx consec valid turns this game 1
Rules koPOSITIONALscoreTERRITORYtaxNONEsui0komi0.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 1
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 1
Game result 0 Empty 0 0 0 0
Last moves pass pass pass
   A B C
 3 . X .
 2 X X O
 1 . O .


Initial pla Black
Encore phase 0
Turns this phase 4
Approx valid turns this phase 1
Approx consec valid turns this game 1
Rules koPOSITIONALscoreTERRITORYtaxNONEsui0komi0.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 1
White handicap bonus score 0
Has button 0
Presumed next pla Black
Past normal phase end 1
Game result 0 Empty 0 0 0 0
Last moves pass pass pass pass
"#;
    expect_lines_match("PreventEncore", &out, expected);
}

#[test]
fn test_rules_double_ko_death_1a() {
    let mut out = String::new();
    let board = Board::parse_board(
        9,
        9,
        r"
   A B C D E F G H J
 9 . . . . X O X . X
 8 . . . . X O O X X
 7 . . X . X X O X O
 6 . . . . O X O O .
 5 . . . O O X X O O
 4 . X X X X . X O .
 3 X X O O O X X O O
 2 O O O X O O X X X
 1 . . . . . O O O O
",
        '\n',
    )
    .unwrap();
    let rules = Rules {
        ko_rule: KoRule::Situational,
        scoring_rule: ScoringRule::Territory,
        tax_rule: TaxRule::None,
        multi_stone_suicide_legal: false,
        komi: 13,
        has_button: false,
        ..Rules::default()
    };
    let mut hist = BoardHistory::new(board.clone(), P_WHITE, rules, 0);
    let mut board = board;
    let xs = board.x_size;

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(8, 3, xs),
        P_BLACK,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(7, 0, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(7, 0, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(8, 3, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    assert!(!hist.is_game_finished);
    hist.print_debug_info(&mut out, &board);

    let expected = r#"
   A B C D E F G H J
 9 . . . . X O . O X
 8 . . . . X O O X X
 7 . . X . X X O X .
 6 . . . . O X O O X
 5 . . . O O X X O O
 4 . X X X X . X O .
 3 X X O O O X X O O
 2 O O O X O O X X X
 1 . . . . . O O O O


Ko recap blocked at H9
Ko recap blocked at J6
   A B C D E F G H J
 9 . . . . X O . O X
 8 . . . . X O O X X
 7 . . X . X X O X .
 6 . . . . O X O O X
 5 . . . O O X X O O
 4 . X X X X . X O .
 3 X X O O O X X O O
 2 O O O X O O X X X
 1 . . . . . O O O O


Ko recap blocked at J6
   A B C D E F G H J
 9 . . . . X O . O X
 8 . . . . X O O X X
 7 . . X . X X O X .
 6 . . . . O X O O X
 5 . . . O O X X O O
 4 . X X X X . X O .
 3 X X O O O X X O O
 2 O O O X O O X X X
 1 . . . . . O O O O


   A B C D E F G H J
 9 . . . . X O . O X
 8 . . . . X O O X X
 7 . . X . X X O X .
 6 . . . . O X O O X
 5 . . . O O X X O O
 4 . X X X X . X O .
 3 X X O O O X X O O
 2 O O O X O O X X X
 1 . . . . . O O O O


Initial pla White
Encore phase 1
Turns this phase 5
Approx valid turns this phase 5
Approx consec valid turns this game 7
Rules koSITUATIONALscoreTERRITORYtaxNONEsui0komi6.5
Ko recap block hash 00000000000000000000000000000000
White bonus score -1
White handicap bonus score 0
Has button 0
Presumed next pla Black
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves pass pass pass J6 H9 H9 J6
"#;
    expect_lines_match("Double ko death 1a", &out, expected);
}

#[test]
fn test_rules_double_ko_death_1b() {
    let mut out = String::new();
    let board = Board::parse_board(
        9,
        9,
        r"
   A B C D E F G H J
 9 . . . . X O X . X
 8 . . . . X O O X X
 7 . . X . X X O X O
 6 . . . . O X O O .
 5 . . . O O X X O O
 4 . X X X X . X O .
 3 X X O O O X X O O
 2 O O O X O O X X X
 1 . . . . . O O O O
",
        '\n',
    )
    .unwrap();
    let rules = Rules {
        ko_rule: KoRule::Situational,
        scoring_rule: ScoringRule::Territory,
        tax_rule: TaxRule::None,
        multi_stone_suicide_legal: false,
        komi: 13,
        has_button: false,
        ..Rules::default()
    };
    let mut hist = BoardHistory::new(board.clone(), P_WHITE, rules, 0);
    let mut board = board;
    let xs = board.x_size;

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(8, 3, xs),
        P_BLACK,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(7, 0, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(7, 0, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(8, 2, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    assert!(!hist.is_game_finished);
    hist.print_debug_info(&mut out, &board);

    let expected = r#"
   A B C D E F G H J
 9 . . . . X O . O X
 8 . . . . X O O X X
 7 . . X . X X O X .
 6 . . . . O X O O X
 5 . . . O O X X O O
 4 . X X X X . X O .
 3 X X O O O X X O O
 2 O O O X O O X X X
 1 . . . . . O O O O


Ko recap blocked at H9
Ko recap blocked at J6
   A B C D E F G H J
 9 . . . . X O . O X
 8 . . . . X O O X X
 7 . . X . X X O X .
 6 . . . . O X O O X
 5 . . . O O X X O O
 4 . X X X X . X O .
 3 X X O O O X X O O
 2 O O O X O O X X X
 1 . . . . . O O O O


Ko recap blocked at J6
   A B C D E F G H J
 9 . . . . X O . O .
 8 . . . . X O O . .
 7 . . X . X X O . O
 6 . . . . O X O O .
 5 . . . O O X X O O
 4 . X X X X . X O .
 3 X X O O O X X O O
 2 O O O X O O X X X
 1 . . . . . O O O O


   A B C D E F G H J
 9 . . . . X O . O .
 8 . . . . X O O . .
 7 . . X . X X O . O
 6 . . . . O X O O .
 5 . . . O O X X O O
 4 . X X X X . X O .
 3 X X O O O X X O O
 2 O O O X O O X X X
 1 . . . . . O O O O


Initial pla White
Encore phase 1
Turns this phase 5
Approx valid turns this phase 5
Approx consec valid turns this game 7
Rules koSITUATIONALscoreTERRITORYtaxNONEsui0komi6.5
Ko recap block hash 00000000000000000000000000000000
White bonus score -2
White handicap bonus score 0
Has button 0
Presumed next pla Black
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves pass pass pass J6 H9 H9 J7
"#;
    expect_lines_match("Double ko death 1b", &out, expected);
}

#[test]
fn test_rules_double_ko_death_1c() {
    let mut out = String::new();
    let board = Board::parse_board(
        9,
        9,
        r"
   A B C D E F G H J
 9 . . . . X O X . X
 8 . . . . X O O X X
 7 . . X . X X O X O
 6 . . . . O X O O .
 5 . . . O O X X O O
 4 . X X X X . X O .
 3 X X O O O X X O O
 2 O O O X O O X X X
 1 . . . . . O O O O
",
        '\n',
    )
    .unwrap();
    let rules = Rules {
        ko_rule: KoRule::Situational,
        scoring_rule: ScoringRule::Territory,
        tax_rule: TaxRule::None,
        multi_stone_suicide_legal: false,
        komi: 13,
        has_button: false,
        ..Rules::default()
    };
    let mut hist = BoardHistory::new(board.clone(), P_WHITE, rules, 0);
    let mut board = board;
    let xs = board.x_size;

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(8, 3, xs),
        P_BLACK,
        line!(),
    );
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(7, 0, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(7, 0, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(8, 3, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(6, 0, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(8, 2, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(8, 3, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);

    let expected = r#"
   A B C D E F G H J
 9 . . . . X O . O X
 8 . . . . X O O X X
 7 . . X . X X O X .
 6 . . . . O X O O X
 5 . . . O O X X O O
 4 . X X X X . X O .
 3 X X O O O X X O O
 2 O O O X O O X X X
 1 . . . . . O O O O


Ko recap blocked at H9
Ko recap blocked at J6
   A B C D E F G H J
 9 . . . . X O . O X
 8 . . . . X O O X X
 7 . . X . X X O X .
 6 . . . . O X O O X
 5 . . . O O X X O O
 4 . X X X X . X O .
 3 X X O O O X X O O
 2 O O O X O O X X X
 1 . . . . . O O O O


Ko recap blocked at J6
   A B C D E F G H J
 9 . . . . X O . O X
 8 . . . . X O O X X
 7 . . X . X X O X .
 6 . . . . O X O O X
 5 . . . O O X X O O
 4 . X X X X . X O .
 3 X X O O O X X O O
 2 O O O X O O X X X
 1 . . . . . O O O O


   A B C D E F G H J
 9 . . . . X O X . X
 8 . . . . X O O X X
 7 . . X . X X O X .
 6 . . . . O X O O X
 5 . . . O O X X O O
 4 . X X X X . X O .
 3 X X O O O X X O O
 2 O O O X O O X X X
 1 . . . . . O O O O


Ko recap blocked at G9
   A B C D E F G H J
 9 . . . . X O X . X
 8 . . . . X O O X X
 7 . . X . X X O X O
 6 . . . . O X O O .
 5 . . . O O X X O O
 4 . X X X X . X O .
 3 X X O O O X X O O
 2 O O O X O O X X X
 1 . . . . . O O O O


Ko recap blocked at G9
Ko recap blocked at J7
   A B C D E F G H J
 9 . . . . X O X . X
 8 . . . . X O O X X
 7 . . X . X X O X O
 6 . . . . O X O O .
 5 . . . O O X X O O
 4 . X X X X . X O .
 3 X X O O O X X O O
 2 O O O X O O X X X
 1 . . . . . O O O O


Ko recap blocked at G9
   A B C D E F G H J
 9 . . . . X O X . X
 8 . . . . X O O X X
 7 . . X . X X O X O
 6 . . . . O X O O .
 5 . . . O O X X O O
 4 . X X X X . X O .
 3 X X O O O X X O O
 2 O O O X O O X X X
 1 . . . . . O O O O


Ko recap blocked at G9
Ko-recap-blocked: (6,0)
Illegal: (8,3) X
"#;
    expect_lines_match("Double ko death 1c", &out, expected);
}

#[test]
fn test_rules_double_ko_death_2a() {
    let mut out = String::new();
    let board = Board::parse_board(
        9,
        9,
        r"
   A B C D E F G H J
 9 . . . X O X . X .
 8 . . . X O O X O X
 7 . . . X X O O . O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .
",
        '\n',
    )
    .unwrap();
    let rules = Rules {
        ko_rule: KoRule::Simple,
        scoring_rule: ScoringRule::Territory,
        tax_rule: TaxRule::All,
        multi_stone_suicide_legal: true,
        komi: 13,
        has_button: false,
        ..Rules::default()
    };
    let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
    let mut board = board;
    let xs = board.x_size;

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(7, 2, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(6, 0, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(8, 3, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(7, 1, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(6, 1, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    assert!(hist.is_legal(&board, location::get_loc(7, 2, xs), P_WHITE));
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(8, 2, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    hist.print_debug_info(&mut out, &board);

    let expected = r#"
   A B C D E F G H J
 9 . . . X O X . X .
 8 . . . X O O X . X
 7 . . . X X O O X O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at H7
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O X . X
 7 . . . X X O O X O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G9
Ko recap blocked at H7
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O X . X
 7 . . . X X O O X .
 6 . . . . X X O O X
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G9
Ko recap blocked at H7
Ko recap blocked at J6
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O . O X
 7 . . . X X O O X .
 6 . . . . X X O O X
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G9
Ko recap blocked at H8
Ko recap blocked at H7
Ko recap blocked at J6
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O . O X
 7 . . . X X O O X .
 6 . . . . X X O O X
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G9
Ko recap blocked at H7
Ko recap blocked at J6
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O . O X
 7 . . . X X O O . O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G9
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O . O X
 7 . . . X X O O . O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Initial pla Black
Encore phase 1
Turns this phase 6
Approx valid turns this phase 6
Approx consec valid turns this game 8
Rules koSIMPLEscoreTERRITORYtaxALLsui1komi6.5
Ko recap block hash F2B074D34150B4B0FC7A68C5A03529A5
White bonus score 3
White handicap bonus score 0
Has button 0
Presumed next pla Black
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves pass pass H7 G9 J6 H8 G8 J7
"#;
    expect_lines_match("Double ko death 2a", &out, expected);
}

#[test]
fn test_rules_double_ko_death_2b() {
    let mut out = String::new();
    let board = Board::parse_board(
        9,
        9,
        r"
   A B C D E F G H J
 9 . . . X O X . X .
 8 . . . X O O X O X
 7 . . . X X O O . O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .
",
        '\n',
    )
    .unwrap();
    let rules = Rules {
        ko_rule: KoRule::Simple,
        scoring_rule: ScoringRule::Territory,
        tax_rule: TaxRule::All,
        multi_stone_suicide_legal: true,
        komi: 13,
        has_button: false,
        ..Rules::default()
    };
    let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
    let mut board = board;
    let xs = board.x_size;

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(7, 2, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(6, 0, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(8, 3, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(7, 1, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(7, 1, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    assert!(hist.is_legal(&board, location::get_loc(7, 2, xs), P_WHITE));
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(8, 3, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(6, 1, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(8, 2, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(5, 0, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(7, 2, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(5, 0, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(7, 1, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    assert!(hist.is_legal(&board, location::get_loc(7, 1, xs), P_BLACK));
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(7, 2, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);

    hist.print_debug_info(&mut out, &board);

    let expected = r#"
   A B C D E F G H J
 9 . . . X O X . X .
 8 . . . X O O X . X
 7 . . . X X O O X O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at H7
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O X . X
 7 . . . X X O O X O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G9
Ko recap blocked at H7
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O X . X
 7 . . . X X O O X .
 6 . . . . X X O O X
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G9
Ko recap blocked at H7
Ko recap blocked at J6
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O . O X
 7 . . . X X O O X .
 6 . . . . X X O O X
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G9
Ko recap blocked at H8
Ko recap blocked at H7
Ko recap blocked at J6
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O . O X
 7 . . . X X O O X .
 6 . . . . X X O O X
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G9
Ko recap blocked at H7
Ko recap blocked at J6
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O . O X
 7 . . . X X O O X .
 6 . . . . X X O O X
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G9
Ko recap blocked at H7
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O X . X
 7 . . . X X O O X .
 6 . . . . X X O O X
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G9
Ko recap blocked at G8
Ko recap blocked at H7
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O X . X
 7 . . . X X O O X O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G9
Ko recap blocked at G8
Ko recap blocked at H7
Ko recap blocked at J7
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O X . X
 7 . . . X X O O X O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G8
Ko recap blocked at H7
Ko recap blocked at J7
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O X . X
 7 . . . X X O O X O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G8
Ko recap blocked at J7
   A B C D E F G H J
 9 . . . X O X . X .
 8 . . . X O O X . X
 7 . . . X X O O X O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at F9
Ko recap blocked at G8
Ko recap blocked at J7
   A B C D E F G H J
 9 . . . X O X . X .
 8 . . . X O O X O X
 7 . . . X X O O . O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at F9
Ko recap blocked at G8
Ko recap blocked at H8
Ko recap blocked at J7
Ko-recap-blocked: (5,0)
Ko-recap-blocked: (6,1)
Ko-recap-blocked: (7,1)
Ko-recap-blocked: (8,2)
   A B C D E F G H J
 9 . . . X O X . X .
 8 . . . X O O X O X
 7 . . . X X O O . O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at F9
Ko recap blocked at G8
Ko recap blocked at J7
   A B C D E F G H J
 9 . . . X O X . X .
 8 . . . X O O X O X
 7 . . . X X O O . O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at F9
Ko recap blocked at G8
Ko recap blocked at J7
Ko-recap-blocked: (5,0)
Ko-recap-blocked: (6,1)
Illegal: (7,2) X
Ko-recap-blocked: (8,2)
   A B C D E F G H J
 9 . . . X O X . X .
 8 . . . X O O X O X
 7 . . . X X O O . O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Initial pla Black
Encore phase 1
Turns this phase 14
Approx valid turns this phase 14
Approx consec valid turns this game 16
Rules koSIMPLEscoreTERRITORYtaxALLsui1komi6.5
Ko recap block hash 880255EB6A5F69E747A8CBA16F7F783B
White bonus score 4
White handicap bonus score 0
Has button 0
Presumed next pla Black
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves pass pass H7 G9 J6 H8 H8 J6 G8 J7 F9 H7 F9 H8 H7 pass 
"#;
    expect_lines_match("Double ko death 2b", &out, expected);
}

#[test]
fn test_rules_double_ko_death_2c() {
    let mut out = String::new();
    let board = Board::parse_board(
        9,
        9,
        r"
   A B C D E F G H J
 9 . . . X O X . X .
 8 . . . X O O X O X
 7 . . . X X O O . O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .
",
        '\n',
    )
    .unwrap();
    let rules = Rules {
        ko_rule: KoRule::Simple,
        scoring_rule: ScoringRule::Territory,
        tax_rule: TaxRule::All,
        multi_stone_suicide_legal: true,
        komi: 13,
        has_button: false,
        ..Rules::default()
    };
    let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
    let mut board = board;
    let xs = board.x_size;

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(7, 2, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(6, 0, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(8, 3, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(7, 1, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    assert!(!hist.is_legal(&board, location::get_loc(6, 0, xs), P_BLACK));
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(7, 1, xs),
        P_BLACK,
        line!(),
    );
    hist.print_debug_info(&mut out, &board);

    let expected = r#"
   A B C D E F G H J
 9 . . . X O X . X .
 8 . . . X O O X . X
 7 . . . X X O O X O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at H7
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O X . X
 7 . . . X X O O X O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G9
Ko recap blocked at H7
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O X . X
 7 . . . X X O O X .
 6 . . . . X X O O X
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G9
Ko recap blocked at H7
Ko recap blocked at J6
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O . O X
 7 . . . X X O O X .
 6 . . . . X X O O X
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G9
Ko recap blocked at H8
Ko recap blocked at H7
Ko recap blocked at J6
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O . O X
 7 . . . X X O O X .
 6 . . . . X X O O X
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Initial pla Black
Encore phase 2
Turns this phase 5
Approx valid turns this phase 5
Approx consec valid turns this game 9
Rules koSIMPLEscoreTERRITORYtaxALLsui1komi6.5
Ko recap block hash 3361ADFB6FEEAE1B61218BE602405A91
White bonus score 4
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves pass pass pass pass H7 G9 J6 H8 H8
"#;
    expect_lines_match("Double ko death 2c", &out, expected);
}

#[test]
fn test_rules_double_ko_death_2d() {
    let mut out = String::new();
    let board = Board::parse_board(
        9,
        9,
        r"
   A B C D E F G H J
 9 . . . X O X . X .
 8 . . . X O O X O X
 7 . . . X X O O . O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .
",
        '\n',
    )
    .unwrap();
    let rules = Rules {
        ko_rule: KoRule::Simple,
        scoring_rule: ScoringRule::Territory,
        tax_rule: TaxRule::All,
        multi_stone_suicide_legal: true,
        komi: 13,
        has_button: false,
        ..Rules::default()
    };
    let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
    let mut board = board;
    let xs = board.x_size;

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(7, 2, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(6, 0, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(8, 3, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(7, 1, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(6, 1, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(7, 2, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(6, 1, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(8, 2, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(5, 0, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    hist.print_debug_info(&mut out, &board);

    let expected = r#"
   A B C D E F G H J
 9 . . . X O X . X .
 8 . . . X O O X . X
 7 . . . X X O O X O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at H7
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O X . X
 7 . . . X X O O X O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G9
Ko recap blocked at H7
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O X . X
 7 . . . X X O O X .
 6 . . . . X X O O X
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G9
Ko recap blocked at H7
Ko recap blocked at J6
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O . O X
 7 . . . X X O O X .
 6 . . . . X X O O X
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G9
Ko recap blocked at H8
Ko recap blocked at H7
Ko recap blocked at J6
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O . O X
 7 . . . X X O O X .
 6 . . . . X X O O X
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G9
Ko recap blocked at H7
Ko recap blocked at J6
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O . O X
 7 . . . X X O O X .
 6 . . . . X X O O X
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G9
Ko recap blocked at J6
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O X . X
 7 . . . X X O O X .
 6 . . . . X X O O X
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G9
Ko recap blocked at G8
Ko recap blocked at J6
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O X . X
 7 . . . X X O O X .
 6 . . . . X X O O X
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G9
Ko recap blocked at G8
   A B C D E F G H J
 9 . . . X . X . X .
 8 . . . X . . X . X
 7 . . . X X . . X .
 6 . . . . X X . . X
 5 . . . . . X X . .
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G8
   A B C D E F G H J
 9 . . . X . X . X .
 8 . . . X . . X . X
 7 . . . X X . . X .
 6 . . . . X X . . X
 5 . . . . . X X . .
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Initial pla Black
Encore phase 1
Turns this phase 9
Approx valid turns this phase 9
Approx consec valid turns this game 11
Rules koSIMPLEscoreTERRITORYtaxALLsui1komi6.5
Ko recap block hash 420E142126FF9F8BB0A4137270E8F8B1
White bonus score 6
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves pass pass H7 G9 J6 H8 G8 H7 G8 J7 F9
"#;
    expect_lines_match("Double ko death 2d", &out, expected);
}

#[test]
fn test_rules_double_ko_death_2e() {
    let mut out = String::new();
    let board = Board::parse_board(
        9,
        9,
        r"
   A B C D E F G H J
 9 . . . X O X . X .
 8 . . . X O O X O X
 7 . . . X X O O . O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .
",
        '\n',
    )
    .unwrap();
    let rules = Rules {
        ko_rule: KoRule::Simple,
        scoring_rule: ScoringRule::Territory,
        tax_rule: TaxRule::All,
        multi_stone_suicide_legal: true,
        komi: 13,
        has_button: false,
        ..Rules::default()
    };
    let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
    let mut board = board;
    let xs = board.x_size;

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(7, 2, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(6, 0, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(5, 0, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(7, 1, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    hist.print_debug_info(&mut out, &board);

    let expected = r#"
   A B C D E F G H J
 9 . . . X O X . X .
 8 . . . X O O X . X
 7 . . . X X O O X O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at H7
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O X . X
 7 . . . X X O O X O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G9
Ko recap blocked at H7
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O X . X
 7 . . . X X O O X O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at H7
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O . O X
 7 . . . X X O O . O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O . O X
 7 . . . X X O O . O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Initial pla Black
Encore phase 2
Turns this phase 4
Approx valid turns this phase 4
Approx consec valid turns this game 8
Rules koSIMPLEscoreTERRITORYtaxALLsui1komi6.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 4
White handicap bonus score 0
Has button 0
Presumed next pla Black
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves pass pass pass pass H7 G9 F9 H8
"#;
    expect_lines_match("Double ko death 2e", &out, expected);
}

#[test]
fn test_rules_double_ko_death_2f() {
    let mut out = String::new();
    let board = Board::parse_board(
        9,
        9,
        r"
   A B C D E F G H J
 9 . . . X O X . X .
 8 . . . X O O X O X
 7 . . . X X O O . O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .
",
        '\n',
    )
    .unwrap();
    let rules = Rules {
        ko_rule: KoRule::Simple,
        scoring_rule: ScoringRule::Territory,
        tax_rule: TaxRule::All,
        multi_stone_suicide_legal: true,
        komi: 13,
        has_button: false,
        ..Rules::default()
    };
    let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
    let mut board = board;
    let xs = board.x_size;

    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_BLACK, line!());
    make_move_assert_legal(&mut hist, &mut board, PASS_LOC, P_WHITE, line!());
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(7, 2, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(6, 0, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(5, 0, xs),
        P_BLACK,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    make_move_assert_legal(
        &mut hist,
        &mut board,
        location::get_loc(7, 2, xs),
        P_WHITE,
        line!(),
    );
    print_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    hist.print_debug_info(&mut out, &board);

    let expected = r#"
   A B C D E F G H J
 9 . . . X O X . X .
 8 . . . X O O X . X
 7 . . . X X O O X O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at H7
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O X . X
 7 . . . X X O O X O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at G9
Ko recap blocked at H7
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O X . X
 7 . . . X X O O X O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Ko recap blocked at H7
   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O X . X
 7 . . . X X O O X O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


   A B C D E F G H J
 9 . . . X O . O X .
 8 . . . X O O X . X
 7 . . . X X O O X O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Initial pla Black
Encore phase 2
Turns this phase 4
Approx valid turns this phase 4
Approx consec valid turns this game 8
Rules koSIMPLEscoreTERRITORYtaxALLsui1komi6.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 4
White handicap bonus score 0
Has button 0
Presumed next pla Black
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves pass pass pass pass H7 G9 F9 H7
"#;
    expect_lines_match("Double ko death 2f", &out, expected);
}
