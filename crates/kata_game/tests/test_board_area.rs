//! Port of `Tests::runBoardAreaTests` from `KataGo/cpp/tests/testboardarea.cpp`.
//!
//! Covers `Board::calculate_area`, rectangular boards, and
//! `Board::is_non_pass_alive_self_connection`. The group-tax scoring and
//! independent-life portions are handled separately.

mod common;

use common::boards_seem_equal;
use kata_core::test::expect_lines_match;
use kata_game::board::player_io::color_to_char;
use kata_game::board::{Board, C_EMPTY, Loc, MAX_ARR_SIZE, P_BLACK, P_WHITE};

fn print_areas(board: &Board, out: &mut String) {
    let safe_big_territories_buf = [false, true, true, true];
    let unsafe_big_territories_buf = [false, false, true, true];
    let non_pass_alive_stones_buf = [false, false, false, true];

    for mode in 0..8 {
        let multi_stone_suicide_legal = mode % 2 == 1;
        let safe_big_territories = safe_big_territories_buf[mode / 2];
        let unsafe_big_territories = unsafe_big_territories_buf[mode / 2];
        let non_pass_alive_stones = non_pass_alive_stones_buf[mode / 2];
        let copy = board.clone();
        let mut result = [C_EMPTY; MAX_ARR_SIZE];
        copy.calculate_area(
            &mut result,
            non_pass_alive_stones,
            safe_big_territories,
            unsafe_big_territories,
            multi_stone_suicide_legal,
        );
        out.push_str(&format!(
            "Safe big territories {} Unsafe big territories {} Non pass alive stones {} Suicide {}\n",
            safe_big_territories as i32,
            unsafe_big_territories as i32,
            non_pass_alive_stones as i32,
            multi_stone_suicide_legal as i32,
        ));
        for y in 0..copy.y_size {
            for x in 0..copy.x_size {
                let loc = kata_game::board::location::get_loc(x, y, copy.x_size) as usize;
                out.push(color_to_char(result[loc]));
            }
            out.push('\n');
        }
        for (i, _) in result.iter().enumerate().take(MAX_ARR_SIZE) {
            if !copy.is_on_board(i as Loc) {
                assert_eq!(result[i], C_EMPTY);
            }
        }
        out.push('\n');
        assert!(boards_seem_equal(&copy, board));
    }
}

#[test]
fn test_board_area_1() {
    let mut out = String::new();
    let board = Board::parse_board(
        9,
        9,
        r"
..o.o.xx.
.oooo.x.x
oo.....xx
.........
.........
x......oo
.xx...o.o
x.x..oo.o
.x.x.o.o.
",
        '\n',
    )
    .unwrap();
    print_areas(&board, &mut out);

    let expected = r#"
Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
......XXX
......XXX
.......XX
.........
.........
.......OO
......OOO
.....OOOO
.....OOOO

Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
......XXX
......XXX
.......XX
.........
.........
.......OO
......OOO
.....OOOO
.....OOOO

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
......XXX
......XXX
.......XX
.........
.........
.......OO
......OOO
.....OOOO
.....OOOO

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
......XXX
......XXX
.......XX
.........
.........
.......OO
......OOO
.....OOOO
.....OOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 0
OO.O..XXX
O.....XXX
.......XX
.........
.........
.......OO
X.....OOO
.X...OOOO
X.X..OOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 1
OO.O..XXX
O.....XXX
.......XX
.........
.........
.......OO
X.....OOO
.X...OOOO
X.X..OOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 0
OOOOO.XXX
OOOOO.XXX
OO.....XX
.........
.........
X......OO
XXX...OOO
XXX..OOOO
XXXX.OOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 1
OOOOO.XXX
OOOOO.XXX
OO.....XX
.........
.........
X......OO
XXX...OOO
XXX..OOOO
XXXX.OOOO

"#;
    expect_lines_match("Area 1", &out, expected);
}

#[test]
fn test_board_area_2() {
    let mut out = String::new();
    let board = Board::parse_board(
        9,
        9,
        r"
x.oooooo.
oox..xx.o
o...xox.o
o...x.x.o
oxxx.xx.o
ox..x...o
o.xox...o
o.xxx...o
.ooooooo.
",
        '\n',
    )
    .unwrap();
    let board2 = Board::parse_board(
        9,
        9,
        r"
..oooooo.
oox..xx.o
o...xox.o
o...x.x.o
oxxx.xx.o
ox..x...o
o.x.x...o
o.xxx...o
.ooooooo.
",
        '\n',
    )
    .unwrap();

    print_areas(&board, &mut out);
    out.push_str("-----\n");
    print_areas(&board2, &mut out);

    let expected = r#"
Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
OOOOOOOOO
OO...XX.O
O...XXX.O
O...XXX.O
OXXXXXX.O
OXXXX...O
O.XXX...O
O.XXX...O
OOOOOOOOO

Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
.........
.........
.........
.........
.........
.........
.........
.........
.........

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
OOOOOOOOO
OO...XX.O
O...XXX.O
O...XXX.O
OXXXXXX.O
OXXXX...O
O.XXX...O
O.XXX...O
OOOOOOOOO

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
.........
.........
.........
.........
.........
.........
.........
.........
.........

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 0
OOOOOOOOO
OO...XX.O
O...XXX.O
O...XXX.O
OXXXXXX.O
OXXXX...O
O.XXX...O
O.XXX...O
OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 1
........O
.........
.........
.........
....X....
.........
.........
.........
O.......O

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 0
OOOOOOOOO
OOX..XX.O
O...XXX.O
O...XXX.O
OXXXXXX.O
OXXXX...O
O.XXX...O
O.XXX...O
OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 1
X.OOOOOOO
OOX..XX.O
O...XOX.O
O...X.X.O
OXXXXXX.O
OX..X...O
O.XOX...O
O.XXX...O
OOOOOOOOO

-----
Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
.........
.........
.........
.........
.........
.........
.........
.........
.........

Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
.........
.........
.........
.........
.........
.........
.........
.........
.........

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
.........
.........
.........
.........
.........
.........
.........
.........
.........

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
.........
.........
.........
.........
.........
.........
.........
.........
.........

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 0
OO......O
.........
.........
.........
....X....
..XX.....
...X.....
.........
O.......O

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 1
OO......O
.........
.........
.........
....X....
..XX.....
...X.....
.........
O.......O

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 0
OOOOOOOOO
OOX..XX.O
O...XOX.O
O...X.X.O
OXXXXXX.O
OXXXX...O
O.XXX...O
O.XXX...O
OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 1
OOOOOOOOO
OOX..XX.O
O...XOX.O
O...X.X.O
OXXXXXX.O
OXXXX...O
O.XXX...O
O.XXX...O
OOOOOOOOO

"#;
    expect_lines_match("Area 2", &out, expected);
}

#[test]
fn test_board_area_3() {
    let mut out = String::new();
    let board = Board::parse_board(
        19,
        19,
        r"
o.o....xx......o..o
.oo.xxx.x......ooo.
xo..x..xx........oo
oo.`xxx..`.....`...
...................
.......xx..........
....xxx.x......xxx.
....xo.xx......x.x.
....xxx........x.x.
xxx`....xxxxx..`x.x
..x.....x...x...xx.
o.xxx...x...x....xx
o.x.x...x...xxx....
..xxx...x...x.x....
xxx.....xxxxxxx....
oo.`.....`.....`...
.o.....oo.oo......o
.oooo.o.o.o.o.oooo.
o..o.o.oo.oo.o.o.o.
",
        '\n',
    )
    .unwrap();
    print_areas(&board, &mut out);

    let expected = r#"
Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
OOO................
OOO................
OO.................
OO.................
...................
.......XX..........
....XXXXX......XXX.
....XXXXX......XXX.
....XXX........XXX.
XXX.............XXX
..X.............XXX
..XXX............XX
..XXX..............
..XXX..............
XXX................
...................
..........OO.......
..........OOO.OOOO.
..........OOOOOOOO.

Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
...................
...................
...................
...................
...................
...................
...............XXX.
...............XXX.
...............XXX.
................XXX
................XXX
.................XX
...................
...................
...................
...................
..........OO.......
..........OOO.OOOO.
..........OOOOOOOO.

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
OOO................
OOO................
OO.................
OO.................
...................
.......XX..........
....XXXXX......XXX.
....XXXXX......XXX.
....XXX........XXX.
XXX.............XXX
..X.............XXX
..XXX............XX
..XXX..............
..XXX..............
XXX................
...................
..........OO.......
..........OOO.OOOO.
..........OOOOOOOO.

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
...................
...................
...................
...................
...................
...................
...............XXX.
...............XXX.
...............XXX.
................XXX
................XXX
.................XX
...................
...................
...................
...................
..........OO.......
..........OOO.OOOO.
..........OOOOOOOO.

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 0
OOO.............OO.
OOO....X..........O
OO...XX............
OO.................
...................
.......XX..........
....XXXXX......XXX.
....XXXXX......XXX.
....XXX........XXX.
XXX.............XXX
..X......XXX....XXX
..XXX....XXX.....XX
..XXX....XXX.......
..XXX....XXX.X.....
XXX................
...................
O.........OO.......
O......O..OOO.OOOOO
.OO.O.O...OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 1
.O..............OO.
.......X..........O
.....XX............
...................
...................
...................
.......X.......XXX.
...............XXX.
...............XXX.
................XXX
.........XXX....XXX
.........XXX.....XX
...X.....XXX.......
.........XXX.X.....
...................
...................
O.........OO.......
O......O..OOO.OOOOO
.OO.O.O...OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 0
OOO....XX......OOOO
OOO.XXXXX......OOOO
OO..XXXXX........OO
OO..XXX............
...................
.......XX..........
....XXXXX......XXX.
....XXXXX......XXX.
....XXX........XXX.
XXX.....XXXXX...XXX
..X.....XXXXX...XXX
O.XXX...XXXXX....XX
O.XXX...XXXXXXX....
..XXX...XXXXXXX....
XXX.....XXXXXXX....
OO.................
OO.....OO.OO......O
OOOOO.OOO.OOO.OOOOO
OOOOOOOOO.OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 1
OOO....XX......OOOO
.OO.XXXXX......OOOO
XO..XXXXX........OO
OO..XXX............
...................
.......XX..........
....XXXXX......XXX.
....XO.XX......XXX.
....XXX........XXX.
XXX.....XXXXX...XXX
..X.....XXXXX...XXX
O.XXX...XXXXX....XX
O.XXX...XXXXXXX....
..XXX...XXXXXXX....
XXX.....XXXXXXX....
OO.................
OO.....OO.OO......O
OOOOO.OOO.OOO.OOOOO
OOOOOOOOO.OOOOOOOOO

"#;
    expect_lines_match("Area 3", &out, expected);
}

#[test]
fn test_board_area_4() {
    let mut out = String::new();
    let board = Board::parse_board(
        19,
        19,
        r"
.x.x.xxxx.xxxx.x.x.
x.xxxx..x.x..xxxx.x
xx.x..x.x.x.x..x.x.
x.x`xx.xx`xx.xx`x.x
.x.xx.xx...xx.xx.x.
.x.x...........x.x.
x.x.............x.x
.x...............x.
...................
xxx`....xxxxx..`...
..x.....x...x.xxxxx
oxxxx...x..xx.x....
o.x.x...x.o.xxx....
..xxx...x...x.x....
xxx.....xxxxxxxxxxx
...`.....`....o`...
ooooo....ooooo.oooo
.o..oo...o...ooo...
o.o.xo...o.o.o.o.oo
",
        '\n',
    )
    .unwrap();
    print_areas(&board, &mut out);

    let expected = r#"
Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
XXXXXXXXX..........
XXXXXX..X..........
XX.X....X..........
X......XX..........
......XX...........
...................
...................
...................
...................
XXX.....XXXXX......
XXX.....XXXXX.XXXXX
XXXXX...XXXXX.X....
XXXXX...XXXXXXX....
XXXXX...XXXXXXX....
XXX.....XXXXXXXXXXX
...................
...................
...................
...................

Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
XXXXXXXXX..........
XXXXXX..X..........
XX.X....X..........
X......XX..........
......XX...........
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
...................
...................

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
XXXXXXXXX..........
XXXXXX..X..........
XX.X....X..........
X......XX..........
......XX...........
...................
...................
...................
...................
XXX.....XXXXX......
XXX.....XXXXX.XXXXX
XXXXX...XXXXX.XXXXX
XXXXX...XXXXXXXXXXX
XXXXX...XXXXXXXXXXX
XXX.....XXXXXXXXXXX
...................
...................
...................
...................

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
XXXXXXXXX..........
XXXXXX..X..........
XX.X....X..........
X......XX..........
......XX...........
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
...................
...................

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 0
XXXXXXXXX.....X.X.X
XXXXXXXXX..XX....X.
XXXXXX.XX..X.XX.X.X
XX.X..XXX...X..X.X.
X.X...XX........X.X
X.X.............X.X
.X...............X.
...................
...................
XXX.....XXXXX......
XXX.....XXXXX.XXXXX
XXXXX...XXXXX.XXXXX
XXXXX...XXXXXXXXXXX
XXXXX...XXXXXXXXXXX
XXX.....XXXXXXXXXXX
...................
..............O....
O.........OOO...OOO
.O........O.O.O.O..

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 1
XXXXXXXXX.....X.X.X
XXXXXXXXX..XX....X.
XXXXXX.XX..X.XX.X.X
XX.X..XXX...X..X.X.
X.X...XX........X.X
X.X.............X.X
.X...............X.
...................
...................
...................
...................
...............XXXX
...X...........XXXX
.............X.XXXX
...................
...................
..............O....
O.........OOO...OOO
.O........O.O.O.O..

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 0
XXXXXXXXX.XXXXXXXXX
XXXXXXXXX.XXXXXXXXX
XXXXXXXXX.XXXXXXXXX
XXXXXXXXX.XXXXXXXXX
XXXXX.XX...XX.XXXXX
XXXX...........XXXX
XXX.............XXX
.X...............X.
...................
XXX.....XXXXX......
XXX.....XXXXX.XXXXX
XXXXX...XXXXX.XXXXX
XXXXX...XXXXXXXXXXX
XXXXX...XXXXXXXXXXX
XXX.....XXXXXXXXXXX
..............O....
OOOOO....OOOOOOOOOO
OO..OO...OOOOOOOOOO
OOO.XO...OOOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 1
XXXXXXXXX.XXXXXXXXX
XXXXXXXXX.XXXXXXXXX
XXXXXXXXX.XXXXXXXXX
XXXXXXXXX.XXXXXXXXX
XXXXX.XX...XX.XXXXX
XXXX...........XXXX
XXX.............XXX
.X...............X.
...................
XXX.....XXXXX......
..X.....X...X.XXXXX
OXXXX...X..XX.XXXXX
O.XXX...X.O.XXXXXXX
..XXX...X...XXXXXXX
XXX.....XXXXXXXXXXX
..............O....
OOOOO....OOOOOOOOOO
OO..OO...OOOOOOOOOO
OOO.XO...OOOOOOOOOO

"#;
    expect_lines_match("Area 4", &out, expected);
}

#[test]
fn test_board_area_5() {
    let mut out = String::new();
    let board = Board::parse_board(
        19,
        13,
        r"
...................
...................
...............xx..
........xxxxxxxx.x.
.....xxxx....x..ox.
..xxx...x..o..xxxx.
..x..xxx........x..
..x.ox..xxxxx...x..
..x.oxxx.oo.xxxxx..
..x.ooo.x..x...x...
..xxx...xxxxxx.x...
....xxxxx.....xx...
...................
",
        '\n',
    )
    .unwrap();
    print_areas(&board, &mut out);

    let expected = r#"
Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
...................
...................
...............XX..
........XXXXXXXXXX.
.....XXXX....XXXXX.
..XXXXXXX.....XXXX.
..XXXXXX........X..
..XXXXXXXXXXX...X..
..XXXXXXXXXXXXXXX..
..XXXXXXXXXXXXXX...
..XXXXXXXXXXXXXX...
....XXXXX.....XX...
...................

Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
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
...................

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXX....XXXXXX
XXXXXXXXX.....XXXXX
XXXXXXXX........XXX
XXXXXXXXXXXXX...XXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
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
...................

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 0
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXX....XXXXXX
XXXXXXXXX.....XXXXX
XXXXXXXX........XXX
XXXXXXXXXXXXX...XXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 1
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXX..XX
XXXXXXXX..........X
XXXXX.............X
XX...XXX..........X
XX...............XX
XX....XX.........XX
XX...............XX
XX..........XXX.XXX
XX............X.XXX
XXXX.....XXXXX..XXX
XXXXXXXXXXXXXXXXXXX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 0
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXX....XXXXXX
XXXXXXXXX..O..XXXXX
XXXXXXXX........XXX
XXXXXXXXXXXXX...XXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 1
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXX.XX
XXXXXXXXX....X..OXX
XXXXXXXXX..O..XXXXX
XXX..XXX........XXX
XXX.OXXXXXXXX...XXX
XXX.OXXX.OO.XXXXXXX
XXX.OOO.X..XXXXXXXX
XXXXX...XXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX

"#;
    expect_lines_match("Area 5", &out, expected);
}

#[test]
fn test_board_area_rect() {
    let mut out = String::new();
    let board = Board::parse_board(
        12,
        3,
        r"
x.ooxxxo.xx.
oo.ox.xoox.x
ooxox.xo.ox.
",
        '\n',
    )
    .unwrap();
    print_areas(&board, &mut out);

    let expected = r#"
Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
OOOO.....XXX
OOOO.....XXX
OOOO......XX

Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
.........XXX
.........XXX
..........XX

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
OOOO.....XXX
OOOO.....XXX
OOOO......XX

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
.........XXX
.........XXX
..........XX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 0
OOOO.....XXX
OOOO.X...XXX
OOOO.X..O.XX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 1
.........XXX
.....X...XXX
.....X..O.XX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 0
OOOOXXXO.XXX
OOOOXXXOOXXX
OOOOXXXOOOXX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 1
X.OOXXXO.XXX
OO.OXXXOOXXX
OOXOXXXOOOXX

"#;
    expect_lines_match("Area Rect", &out, expected);
}

#[test]
fn test_is_non_pass_alive_self_connection() {
    let board = Board::parse_board(
        9,
        9,
        r"
.x.oo.xx.
xoo.oxox.
.o.ooooox
ooo....x.
..o..x.xx
oo.o...x.
...xxxoxx
xxxx.x.x.
o.o.xxoxo
",
        '\n',
    )
    .unwrap();

    let mut result = [C_EMPTY; MAX_ARR_SIZE];
    board.calculate_area(&mut result, false, true, false, false);

    let mut out = String::new();
    out.push_str("\nNonPassAliveSelfConn black\n");
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = kata_game::board::location::get_loc(x, y, board.x_size);
            if board.colors[loc as usize] == C_EMPTY {
                out.push_str(
                    &(board.is_non_pass_alive_self_connection(loc, P_BLACK, &result) as i32)
                        .to_string(),
                );
            } else {
                assert!(!board.is_non_pass_alive_self_connection(loc, P_BLACK, &result));
                out.push('-');
            }
        }
        out.push('\n');
    }

    out.push('\n');
    out.push_str("NonPassAliveSelfConn white\n");
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = kata_game::board::location::get_loc(x, y, board.x_size);
            if board.colors[loc as usize] == C_EMPTY {
                out.push_str(
                    &(board.is_non_pass_alive_self_connection(loc, P_WHITE, &result) as i32)
                        .to_string(),
                );
            } else {
                assert!(!board.is_non_pass_alive_self_connection(loc, P_WHITE, &result));
                out.push('-');
            }
        }
        out.push('\n');
    }
    out.push('\n');

    let expected = r#"

NonPassAliveSelfConn black
0-0--1--0
---0----1
0-0------
---0000-1
00-00-1--
--0-010-0
000------
----0-0-0
-0-0-----

NonPassAliveSelfConn white
0-0--0--0
---0----0
0-0------
---0000-0
11-10-0--
--1-000-0
000------
----0-1-0
-0-0-----

"#;
    expect_lines_match("isNonPassAliveSelfConnection", &out, expected);
}
