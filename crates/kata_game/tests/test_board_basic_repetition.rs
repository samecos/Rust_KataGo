//! Repetition-bound slice of KataGo's `cpp/tests/testboardbasic.cpp`.
//!
//! Ports the `simpleRepetitionBound` and `simpleRepetitionBoundOpen` cases
//! from `runBoardBasicTests`.

mod common;

use common::boards_seem_equal;
use kata_core::test::expect_lines_match;
use kata_game::board::location::get_loc;
use kata_game::board::{Board, C_EMPTY, P_BLACK, P_WHITE, player_io};

fn append_repetition_board_output(out: &mut String, board: &Board, pla: i8) {
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = get_loc(x, y, board.x_size);
            if board.colors[loc as usize] != C_EMPTY {
                if pla != C_EMPTY {
                    out.push_str(&format!(
                        "   {}",
                        player_io::color_to_char(board.colors[loc as usize])
                    ));
                } else {
                    let mut bound = 0;
                    while bound < 362 {
                        if !board.simple_repetition_bound_gt(loc, bound) {
                            break;
                        }
                        bound += 1;
                    }
                    out.push_str(&format!(" {:>3}", bound));
                    while bound < 362 {
                        assert!(!board.simple_repetition_bound_gt(loc, bound));
                        bound += 1;
                    }
                }
            } else {
                if pla == C_EMPTY {
                    out.push_str("   .");
                } else {
                    let mut board_copy = board.clone();
                    board_copy.clear_simple_ko_loc();
                    let is_multi_stone_suicide_legal = true;
                    let suc = board_copy.play_move(loc, pla, is_multi_stone_suicide_legal);
                    if !suc {
                        out.push_str("   .");
                    } else {
                        let mut bound = 0;
                        while bound < 362 {
                            if !board_copy.simple_repetition_bound_gt(loc, bound) {
                                break;
                            }
                            bound += 1;
                        }
                        out.push_str(&format!(" {:>3}", bound));
                        while bound < 362 {
                            assert!(!board_copy.simple_repetition_bound_gt(loc, bound));
                            bound += 1;
                        }
                    }
                }
            }
        }
        out.push('\n');
    }
    out.push('\n');
}

#[test]
fn test_simple_repetition_bound_open() {
    let board = Board::parse_board(
        19,
        19,
        r#"   A B C D E F G H J K L M N O P Q R S T
19 . . . . . . . . . . . . . . . . . . .
18 . . . . . . . . . . . . . . . . . . .
17 . . . . . . . . . . . . . . . . . . .
16 . . . . . . . . . . . . . . . O . . .
15 . . . . . . . . . . . . . . . . . . .
14 . . . . . . . . . . . . . . . . . . .
13 . . . . . . . . . . . . . . . . . . .
12 . . . . . . . . . . . . . . . . . . .
11 . . . . . . . . . . . . . . . . . . .
10 . . . . . . . . . . . . . . . . . . .
 9 . . . . . . . . . . . . . . . . . . .
 8 . . . . . . . . . . . . . . . . . . .
 7 . . O . . . . . . . . . . . . . . . .
 6 . . X O . . . . . . . . . . . . . . .
 5 . . X . . . . . . . . . . . . . . . .
 4 . . . X . . . . . . . . . . . . . . .
 3 . . . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . .
"#,
        '\n',
    )
    .unwrap();

    let start_board = board.clone();
    let mut out = String::new();

    for pla in [C_EMPTY, P_BLACK, P_WHITE] {
        append_repetition_board_output(&mut out, &board, pla);
    }

    assert!(boards_seem_equal(&board, &start_board));

    let expected = r#"
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   . 356   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   . 356   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   . 357 356   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   . 357   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   . 356   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .

 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355   O 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355   O 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 357   X   O 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 357   X 358 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 358   X 356 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 356 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355

 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 356 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 356   O 356 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 356 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 356 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 356   O 357 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355   X   O 356 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355   X 356 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355   X 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
"#;
    expect_lines_match("simpleRepetitionBoundOpen", &out, expected);
}

#[test]
fn test_simple_repetition_bound() {
    let board = Board::parse_board(
        19,
        19,
        r#"   A B C D E F G H J K L M N O P Q R S T
19 . . X O . X O . X O O . . . . O X . .
18 . . . X X O O X . . X O O . O X X X X
17 . X . . X X X X X X X . . O O O X O .
16 X . . X O X O O O O X X O O X O X X .
15 O X X . O O . O . . X . O X X X O X .
14 . O O O . . X O X X X O O . X X O X O
13 O . . . . O O X O O O O . O X O O O .
12 X O O O O . X X . . O X X . X O . O X
11 . X . X . X . . X . O . X . X X X O .
10 X . . X . . X O X X O O X O . X O . O
 9 . X . O X . X X O . X O . O X . O . .
 8 . X . O . . X O O . X O O X X X O X .
 7 . O X . . O O . . X X O . O X X O O .
 6 X X O X . . . . X . X O O . . X X O .
 5 O . O O O O O X . . . X O O O X O X .
 4 . O . O X X . O X X X X X O X X O . .
 3 . . . O X O O O X . . X O X . X O . .
 2 . . O X X O X X O . X . O X . . O X .
 1 . . O . . X . O O . . X O . . . O . .
"#,
        '\n',
    )
    .unwrap();

    let start_board = board.clone();
    let mut out = String::new();

    for pla in [C_EMPTY, P_BLACK, P_WHITE] {
        append_repetition_board_output(&mut out, &board, pla);
    }

    assert!(boards_seem_equal(&board, &start_board));

    let expected = r#"
   .   .  11   2   .   2   4   .   4   9   9   .   .   .   .   6  15   .   .
   .   .   .  37  37   4   4  37   .   .  37   9   9   .  46  15  15  15  15
   .  11   .   .  37  37  37  37  37  37  37   .   .  46  46  46  15   4   .
  11   .   .  12  11  37   9   9   9   9  37  37  46  46  18  46  15  15   .
   2  13  13   .  11  11   .   9   .   .  37   .  46  18  18  18  25  15   .
   .  11  11  11   .   .   8   9  37  37  37  46  46   .  18  18  25  15   5
   8   .   .   .   .   9   9   9  46  46  46  46   .   5  18  25  25  25   .
   2  30  30  30  30   .   9   9   .   .  46   9   9   .  18  25   .  25   3
   .   7   .  21   .  18   .   .  10   .  46   .   9   .  18  18  18  25   .
  10   .   .  21   .   .  20   3  10  10  46  46   9   6   .  18  21   .  16
   .  10   .  21  15   .  20  20  19   .  25  46   .   6  22   .  21   .   .
   .  10   .  21   .   .  20  19  19   .  25  46  46  22  22  22  21  15   .
   .   4  20   .   .  16  16   .   .  25  25  46   .   4  22  22  21  21   .
   6   6  33  15   .   .   .   .  19   .  25  46  46   .   .  22  22  21   .
  11   .  33  33  33  33  33  19   .   .   .  18  46  46  46  22  25  15   .
   .  11   .  33   8   8   .   6  18  18  18  18  18  46  22  22  25   .   .
   .   .   .  33   8   6   6   6  18   .   .  18  10   8   .  22  25   .   .
   .   .  13   8   8   6   3   3   9   .   7   .  10   8   .   .  25  15   .
   .   .  13   .   .   4   .   9   9   .   .   7  10   .   .   .  25   .   .

  10  11   X   O  39   X   O  41   X   O   O   5   5   5   6   O   X  15  15
  10  11  38   X   X   O   O   X  38  37   X   O   O   5   O   X   X   X   X
  12   X  11  39   X   X   X   X   X   X   X  37   2   O   O   O   X   O  16
   X  15  14   X   O   X   O   O   O   O   X   X   O   O   X   O   X   X  15
   O   X   X  14   O   O   8   O  37  37   X  37   O   X   X   X   O   X  15
   2   O   O   O   6   8   X   O   X   X   X   O   O  18   X   X   O   X   O
   O   6   6   6   6   O   O   X   O   O   O   O   9   O   X   O   O   O   3
   X   O   O   O   O  24   X   X  14   3   O   X   X  25   X   O  18   O   X
  12   X  23   X  25   X  28  15   X  10   O   9   X  25   X   X   X   O   3
   X  13  21   X  22  22   X   O   X   X   O   O   X   O  38   X   O  14   O
  12   X  10   O   X  21   X   X   O  33   X   O   9   O   X  38   O  15  14
  10   X  25   O  15  20   X   O   O  25   X   O   O   X   X   X   O   X  15
   7   O   X  21  14   O   O  14  26   X   X   O   .   O   X   X   O   O  14
   X   X   O   X  15  14  14  20   X  26   X   O   O   2  22   X   X   O  14
   O   6   O   O   O   O   O   X  34  18  39   X   O   O   O   X   O   X  15
   9   O   9   O   X   X  13   O   X   X   X   X   X   O   X   X   O  15  14
   9   9   9   O   X   O   O   O   X  18  19   X   O   X  24   X   O  15  14
   9   9   O   X   X   O   X   X   O   7   X  20   O   X   8  22   O   X  15
   9   9   O   8  10   X   6   O   O   5   8   X   O   8   6   6   O  15  14

  10  10   X   O   3   X   O   4   X   O   O  13   9   5  47   O   X   2   2
  10  10  10   X   X   O   O   X   2   9   X   O   O  48   O   X   X   X   X
  10   X  10  10   X   X   X   X   X   X   X   9  48   O   O   O   X   O   4
   X  10  10   X   O   X   O   O   O   O   X   X   O   O   X   O   X   X   3
   O   X   X  15   O   O  19   O   9   9   X  46   O   X   X   X   O   X   5
  13   O   O   O  15  14   X   O   X   X   X   O   O  49   X   X   O   X   O
   O  36  35  35  32   O   O   X   O   O   O   O  49   O   X   O   O   O  29
   X   O   O   O   O  32   X   X  46  46   O   X   X   5   X   O  25   O   X
   2   X  30   X  30   X   2   3   X  46   O  46   X   6   X   X   X   O  26
   X   5   5   X  14  14   X   O   X   X   O   O   X   O   6   X   O  33   O
   3   X  21   O   X  14   X   X   O  19   X   O  51   O   X  21   O  21  16
   3   X  21   O  21  16   X   O   O  19   X   O   O   X   X   X   O   X  14
   4   O   X  21  16   O   O  21  19   X   X   O  47   O   X   X   O   O  21
   X   X   O   X  33  35  35  14   X   4   X   O   O  47  46   X   X   O  21
   O  35   O   O   O   O   O   X   4   4   4   X   O   O   O   X   O   X  14
  12   O  34   O   X   X  38   O   X   X   X   X   X   O   X   X   O  25  14
   9  11  37   O   X   O   O   O   X   5   5   X   O   X   6   X   O  25  14
   9  13   O   X   X   O   X   X   O   9   X  10   O   X   6  25   O   X  14
   9  13   O  13   2   X  11   O   O   9   5   X   O  10   6  25   O  25  14
"#;
    expect_lines_match("simpleRepetitionBound", &out, expected);
}
