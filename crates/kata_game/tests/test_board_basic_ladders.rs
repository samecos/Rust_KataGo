//! Ladder tests ported from `KataGo/cpp/tests/testboardbasic.cpp`.
//!
//! Corresponds to the ladder section of `runBoardBasicTests` (lines 1030-1603).

mod common;

use common::boards_seem_equal;
use kata_core::test::expect_lines_match;
use kata_game::board::location::get_loc;
use kata_game::board::{Board, C_EMPTY};

fn render_ladder_grid<F: Fn(i32, i32) -> char>(y_size: i32, x_size: i32, f: F) -> String {
    let mut out = String::new();
    out.push('\n');
    for y in 0..y_size {
        for x in 0..x_size {
            out.push(f(x, y));
        }
        out.push('\n');
    }
    out
}

#[test]
fn ladders_1_lib() {
    let board = Board::parse_board(
        9,
        9,
        r#"xo.x..oxo
xoxo..o..
xxo......
..o.x....
xo..xox..
o..ooxo..
.....xo..
xoox..xo.
.xxoo.xxo
"#,
        '\n',
    )
    .unwrap();

    let start_board = board.clone();
    let out = render_ladder_grid(board.y_size, board.x_size, |x, y| {
        let loc = get_loc(x, y, board.x_size);
        if board.colors[loc as usize] != C_EMPTY {
            if board.search_is_ladder_captured(loc, true) {
                '1'
            } else {
                '0'
            }
        } else {
            '.'
        }
    });

    assert!(boards_seem_equal(&board, &start_board));

    let expected = r#"
01.0..010
0100..0..
000......
..0.0....
10..000..
0..0000..
.....00..
0000..00.
.1100.001
"#;
    expect_lines_match("Ladders 1 Lib", &out, expected);
}

#[test]
fn ladders_2_libs() {
    let board = Board::parse_board(
        9,
        9,
        r#"xo.x..oxo
xo.o..o..
xxo......
..o.x....
xo..xo...
...ooxo..
.....xo..
xoox..xo.
.xx.o.xxo
"#,
        '\n',
    )
    .unwrap();

    let start_board = board.clone();
    let out = render_ladder_grid(board.y_size, board.x_size, |x, y| {
        let loc = get_loc(x, y, board.x_size);
        if board.colors[loc as usize] != C_EMPTY {
            if board.search_is_ladder_captured_attacker_first_2_libs(loc).0 {
                '1'
            } else {
                '0'
            }
        } else {
            '.'
        }
    });

    assert!(boards_seem_equal(&board, &start_board));

    let expected = r#"
11.1..000
11.0..0..
110......
..0.0....
10..00...
...0010..
.....10..
1110..01.
.11.0.000

"#;
    expect_lines_match("Ladders 2 Libs", &out, expected);
}

#[test]
fn ladders_ko_1() {
    let board = Board::parse_board(
        18,
        9,
        r#"..................
..................
....ox.......ox...
..xooox....xooox..
..xoxox....xoxox..
..xx.x......x.x...
...ox.......ox....
....o.............
..................
"#,
        '\n',
    )
    .unwrap();

    let start_board = board.clone();
    let out = render_ladder_grid(board.y_size, board.x_size, |x, y| {
        let loc = get_loc(x, y, board.x_size);
        if board.colors[loc as usize] != C_EMPTY {
            if board.search_is_ladder_captured_attacker_first_2_libs(loc).0 {
                '1'
            } else {
                '0'
            }
        } else {
            '.'
        }
    });

    assert!(boards_seem_equal(&board, &start_board));

    let expected = r#"
..................
..................
....10.......00...
..01110....00000..
..01010....00000..
..00.0......0.0...
...00.......00....
....0.............
..................

"#;
    expect_lines_match("LaddersKo-1", &out, expected);
}

#[test]
fn ladders_ko_2() {
    let board = Board::parse_board(
        18,
        9,
        r#".............xo.oo
....x........xxooo
...x.xx.......xxo.
..xoxoxo.......xxo
..xoooxo........xx
...xo......o......
..................
..................
.....x.oo.........
"#,
        '\n',
    )
    .unwrap();
    let board2 = Board::parse_board(
        18,
        9,
        r#"..................
....x.............
...x.xx...........
..xoxoxo..........
..xoooxo..........
...xo......o......
.................x
..................
.....x.oo.........
"#,
        '\n',
    )
    .unwrap();
    let board3 = Board::parse_board(
        18,
        9,
        r#"....xo.......xo...
....xox......xox..
...xo.ox....xo.ox.
...xxoox....xxoox.
..xooox....xooox..
.xo.ox....xo.ox...
..xox......xox....
..xox......xox....
............o.....
"#,
        '\n',
    )
    .unwrap();

    let mut out = String::new();
    out.push('\n');

    for b in [&board, &board2, &board3] {
        let y_size = b.y_size;
        let x_size = b.x_size;
        for y in 0..y_size {
            for x in 0..x_size {
                let loc = get_loc(x, y, x_size);
                if b.colors[loc as usize] != C_EMPTY {
                    if b.search_is_ladder_captured_attacker_first_2_libs(loc).0 {
                        out.push('1');
                    } else {
                        out.push('0');
                    }
                } else {
                    out.push('.');
                }
            }
            out.push('\n');
        }
        out.push('\n');
    }

    assert!(boards_seem_equal(&board, &board));
    assert!(boards_seem_equal(&board2, &board2));
    assert!(boards_seem_equal(&board3, &board3));

    let expected = r#"

.............00.11
....0........00111
...0.00.......001.
..000000.......000
..000000........00
...00......0......
..................
..................
.....0.00.........

..................
....0.............
...0.00...........
..010100..........
..011100..........
...01......0......
.................0
..................
.....0.00.........

....01.......01...
....011......011..
...00.10....00.00.
...00110....00000.
..01110....00000..
.00.10....00.00...
..010......000....
..010......000....
............0.....

"#;
    expect_lines_match("LaddersKo-2", &out, expected);
}

#[test]
fn ladders_multi_ko() {
    let board = Board::parse_board(
        9,
        9,
        r#".xxxxxxo.
xoxox.xo.
ooo.oxoo.
..oooo...
.xxxx....
xx.x.xx..
xoxoxox..
xoooooxx.
o.ooo.ox.
"#,
        '\n',
    )
    .unwrap();

    let out = render_ladder_grid(board.y_size, board.x_size, |x, y| {
        let loc = get_loc(x, y, board.x_size);
        if board.colors[loc as usize] != C_EMPTY {
            if board.search_is_ladder_captured_attacker_first_2_libs(loc).0 {
                '1'
            } else {
                '0'
            }
        } else {
            '.'
        }
    });

    let expected = r#"
.0000000.
00000.00.
000.0000.
..0000...
.0000....
00.0.00..
0000000..
00000000.
0.000.00.
"#;
    expect_lines_match("LaddersMultiKo", &out, expected);
}

#[test]
fn ladders_big_board() {
    let board = Board::parse_board(
        19,
        19,
        r#"   A B C D E F G H J K L M N O P Q R S T
19 . . . . . . . . . . . . . . . . . . .
18 . . . . . . . . . . . . . . . . . . .
17 . . . . . . . . . . . . . . . . . . .
16 . . . X . . . . . . . . . . . X . . .
15 . . . . . . . . . . . . . . . . . . .
14 . . . . . . . . . . . . . . . . . . .
13 . . . . . . . . . . . . . . . . . . .
12 . . . . . . . . . . . . . . . . . . .
11 . . . . . . . . . . . . . . . . . . .
10 . . . . . . . . . . . . . . . . . . .
 9 . . . . . . . . . . . . . . . . . . .
 8 . . . . . . . . . . . . . . . . . . .
 7 . . . . . . . . . . . . . . . . . . .
 6 . . . . . . . . . . . . . . . . . . .
 5 . . . . . . . . . . . . . . . . . . .
 4 . . . . . . . . . . . . . . X O O . .
 3 . . . O . . . . . . . . . . O X X . .
 2 . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . .
"#,
        '\n',
    )
    .unwrap();

    let start_board = board.clone();
    let out = render_ladder_grid(board.y_size, board.x_size, |x, y| {
        let loc = get_loc(x, y, board.x_size);
        if board.colors[loc as usize] != C_EMPTY {
            if board.search_is_ladder_captured_attacker_first_2_libs(loc).0 {
                '1'
            } else {
                '0'
            }
        } else {
            '.'
        }
    });

    assert!(boards_seem_equal(&board, &start_board));

    let expected = r#"
...................
...................
...................
...0...........0...
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
..............000..
...0..........000..
...................
...................
"#;
    expect_lines_match("LaddersBigBoard", &out, expected);
}

#[test]
fn whole_board_ladder() {
    let board = Board::parse_board(
        19,
        19,
        r#"   A B C D E F G H J K L M N O P Q R S T
19 . . . . O . . . . . . . . . . . . . .
18 O . . . . . . . . . . . . . . . . . O
17 . . . . . . . . . . . . . . . . . . O
16 . . . O O X O . . . . . . . . . . . .
15 O . . . O X O O O O . O . . . . . O .
14 O X X X X X X X X X O . . . . . . . .
13 X O O O . X . . . . X . . . . . . . O
12 X . O O O X . . . . X O . . . . . . .
11 X O O O O X . . . . X . . . . . . . .
10 O X X X X X X X X X O . . . . . . . O
 9 O O . . . X O . . O . . . . . . . . .
 8 . O . O . X . O . . . . . . . . . . .
 7 . . . . . . O . . . . . . . . . . . .
 6 . . . . O . . . . X . . . . . . . . .
 5 O . . . . X . . . X . . . X O . . . .
 4 . . . . . X . . . X . . . X . . . . .
 3 . . . . . X . . . X . . . X . . . . .
 2 . . . . O . X X X X X X X O . . . O .
 1 . O . . . . O O O O O O O . O . . . .
"#,
        '\n',
    )
    .unwrap();

    let start_board = board.clone();
    let out = render_ladder_grid(board.y_size, board.x_size, |x, y| {
        let loc = get_loc(x, y, board.x_size);
        if board.colors[loc as usize] != C_EMPTY {
            if board.search_is_ladder_captured_attacker_first_2_libs(loc).0 {
                '1'
            } else {
                '0'
            }
        } else {
            '.'
        }
    });

    assert!(boards_seem_equal(&board, &start_board));

    let expected = r#"
....0..............
0.................0
..................0
...0000............
0...000000.0.....0.
00000000000........
0000.0....0.......0
0.0000....00.......
000000....0........
00000000000.......0
00...00..0.........
.0.0.0.0...........
......0............
....0....0.........
0....0...0...00....
.....0...0...0.....
.....0...0...0.....
....0.00000000...0.
.0....1111111.0....
"#;
    expect_lines_match("WholeBoardLadder", &out, expected);
}

#[test]
fn cubic_ladder_not_far_from_max_node_budget() {
    let board = Board::parse_board(
        19,
        19,
        r#"   A B C D E F G H J K L M N O P Q R S T
19 . . . O O O O O O O O . . . . . O . O
18 X X X . X X X X X . . . O O O O O . O
17 O . . . . . O O . . . . . . . X X X .
16 . . . . . . . O O O . . . . . . O . .
15 O . O . . . . . . . . . . . . . . . .
14 . . O . . . . . . O . . . . . . O . O
13 . . O . . . . . . . X . . . . . O . .
12 . . . . . . . . . O X . . . . . . . .
11 . . . . . . . X . X . O . O . . . . .
10 O X X . . . . . X . O . . . . . . . .
 9 O X . . . . . . X O . . . . . . . . .
 8 . . X O . . . . . X . . . . . . . . .
 7 O . . . X . . . . . . . . . . . . . .
 6 . . . . . . . . . . . . . . . . . . .
 5 . . . . . . . . . . . . . . . . . . .
 4 O . . . . . . . . . . . . . . . O . .
 3 . . . . . . . . . X O . . . . . O . .
 2 . . . O O . . . . X O X . X . . . X .
 1 . . . . . . X . O O O X . X . . O O O
"#,
        '\n',
    )
    .unwrap();

    let start_board = board.clone();
    let out = render_ladder_grid(board.y_size, board.x_size, |x, y| {
        let loc = get_loc(x, y, board.x_size);
        if board.colors[loc as usize] != C_EMPTY {
            if board.search_is_ladder_captured_attacker_first_2_libs(loc).0 {
                '1'
            } else {
                '0'
            }
        } else {
            '.'
        }
    });

    assert!(boards_seem_equal(&board, &start_board));

    let expected = r#"
...00000000.....0.0
000.00000...00000.0
0.....00.......000.
.......000......0..
0.0................
..0......0......0.0
..0.......0.....0..
.........00........
.......0.0.0.0.....
000.....0.0........
00......01.........
..00.....0.........
0...0..............
...................
...................
0...............0..
.........00.....0..
...00....000.0...0.
......0.0000.0..000
"#;
    expect_lines_match("CubicLadder not far from max node budget", &out, expected);
}

#[test]
fn failing_polynomial_ladder_due_to_max_node_budget() {
    let board = Board::parse_board(
        19,
        19,
        r#"   A B C D E F G H J K L M N O P Q R S T
19 X . O O O O X . O O O O X . O O O O O
18 O X X X X O O X X X X O O X X X X X O
17 O X . . X X O X . . X X O X . . . X O
16 O . . . . X O X . . . X O X . . . X O
15 . . . . . . O X . . . . O X . . . X O
14 . . . . . . . . . . . . . . . . . X O
13 . . . . . . . . . . . . . . . X X O O
12 . . . . . . . . . . . . . . . O O O X
11 . . . . . . . . . . . . . . . . X X .
10 . . . . . . . . . . . . . . . . . X O
 9 . . . . . . . . . . . . . . . . . X O
 8 . . . . . . . . . . . . . . . . . X O
 7 . . . . . . . . . . . . . . . X X O O
 6 . . . . . . . . . . . . . . . O O O X
 5 . . . . . . . . . . . . . . . . X X .
 4 . . . . . . . . . . . . . . . . . X O
 3 . . . . . . . . . . . . . . . . . X O
 2 . X . . . X . . . . . . . . . . X . O
 1 . . . . . . . . . . . . . . . O O O .
"#,
        '\n',
    )
    .unwrap();

    let start_board = board.clone();
    let out = render_ladder_grid(board.y_size, board.x_size, |x, y| {
        let loc = get_loc(x, y, board.x_size);
        if board.colors[loc as usize] != C_EMPTY {
            if board.search_is_ladder_captured_attacker_first_2_libs(loc).0 {
                '1'
            } else {
                '0'
            }
        } else {
            '.'
        }
    });

    assert!(boards_seem_equal(&board, &start_board));

    let expected = r#"
0.00000.00000.00000
0000000000000000000
00..0000..0000...00
0....000...000...00
......00....00...00
.................00
...............0000
...............0000
................00.
.................00
.................00
.................00
...............0000
...............0000
................00.
.................00
.................00
.0...0..........0.0
...............000.
"#;
    expect_lines_match(
        "Failing polynomial ladder due to max node budget",
        &out,
        expected,
    );
}
