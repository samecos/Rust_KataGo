//! Port of `Tests::runBoardUndoTest` from `KataGo/cpp/tests/testboardbasic.cpp`.
//!
//! Verifies that `Board::play_move_recorded` and `Board::undo` are inverse
//! operations for long random move sequences on multiple board sizes.

mod common;

use common::boards_seem_equal;
use kata_core::rng::Rand;
use kata_core::test::expect_lines_match;
use kata_game::board::{Board, Loc, NULL_LOC, P_BLACK, P_WHITE, PASS_LOC, Player};

#[test]
fn test_board_undo() {
    let mut regular_move_count = 0i32;
    let mut pass_count = 0i32;
    let mut ko_capture_count = 0i32;
    let mut suicide_count = 0i32;

    let run = |start_board: &Board,
               multi_stone_suicide_legal: bool,
               rand: &mut Rand,
               regular_move_count: &mut i32,
               pass_count: &mut i32,
               ko_capture_count: &mut i32,
               suicide_count: &mut i32| {
        const STEPS: usize = 1000;
        let mut boards: Vec<Board> = Vec::with_capacity(STEPS + 1);
        let mut records: Vec<kata_game::board::MoveRecord> = Vec::with_capacity(STEPS);

        boards.push(start_board.clone());
        for n in 1..=STEPS {
            boards.push(boards[n - 1].clone());
            let mut loc: Loc;
            let mut pla: Player;
            loop {
                pla = if rand.next_u32_bounded(2) == 0 {
                    P_BLACK
                } else {
                    P_WHITE
                };
                // Maximum range of board location values when 19x19.
                let num_locs = (19 + 1) * (19 + 2) + 1;
                loc = rand.next_u32_bounded(num_locs as u32) as Loc;
                if boards[n].is_legal(loc, pla, multi_stone_suicide_legal) {
                    break;
                }
            }

            records.push(boards[n].play_move_recorded(loc, pla));

            if loc == PASS_LOC {
                *pass_count += 1;
            } else if boards[n - 1].is_suicide(loc, pla) {
                *suicide_count += 1;
            } else {
                if boards[n].ko_loc != NULL_LOC {
                    *ko_capture_count += 1;
                }
                *regular_move_count += 1;
            }
        }

        let mut board = boards[STEPS].clone();
        for n in (0..STEPS).rev() {
            let record = records[n];
            board.undo(record);
            assert!(
                boards_seem_equal(&boards[n], &board),
                "undo mismatch at step {}",
                n
            );
        }
    };

    let mut rand = Rand::new_from_seed("runBoardUndoTests");
    run(
        &Board::new(19, 19),
        true,
        &mut rand,
        &mut regular_move_count,
        &mut pass_count,
        &mut ko_capture_count,
        &mut suicide_count,
    );
    run(
        &Board::new(4, 4),
        true,
        &mut rand,
        &mut regular_move_count,
        &mut pass_count,
        &mut ko_capture_count,
        &mut suicide_count,
    );
    run(
        &Board::new(4, 4),
        false,
        &mut rand,
        &mut regular_move_count,
        &mut pass_count,
        &mut ko_capture_count,
        &mut suicide_count,
    );

    let mut out = String::new();
    out.push('\n');
    out.push_str(&format!("regularMoveCount {}\n", regular_move_count));
    out.push_str(&format!("passCount {}\n", pass_count));
    out.push_str(&format!("koCaptureCount {}\n", ko_capture_count));
    out.push_str(&format!("suicideCount {}\n", suicide_count));
    out.push('\n');

    let expected = r#"
regularMoveCount 2446
passCount 475
koCaptureCount 24
suicideCount 79

"#;
    expect_lines_match("Board undo test move counts", &out, expected);
}
