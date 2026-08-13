//! Ported from `KataGo/cpp/tests/testboardbasic.cpp`.
//!
//! Covers the "liberties" slice of `runBoardBasicTests`: liberties,
//! liberties-after-move, distance, adjacency, capture, and ko-capture helpers.

mod common;

use kata_core::test::expect_lines_match;
use kata_game::board::location::{distance, euclidean_distance_squared, get_loc};
use kata_game::board::{Board, C_EMPTY, P_BLACK, P_WHITE};

#[test]
fn test_liberties() {
    let board = Board::parse_board(
        9,
        9,
        ".........\n.....x...\n..oo..x..\n..x......\n......xx.\n..x..ox..\n.oxoo.oxx\nxxoo.o.ox\n.x.....oo\n",
        '\n',
    )
    .unwrap();

    let mut out = String::new();
    out.push('\n');
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = get_loc(x, y, board.x_size);
            if board.colors[loc as usize] != C_EMPTY {
                out.push_str(&board.get_num_liberties(loc).to_string());
            } else {
                out.push('.');
            }
        }
        out.push('\n');
    }
    out.push('\n');

    let expected = "\n.........\n.....4...\n..55..4..\n..3......\n......55.\n..3..35..\n.2366.222\n3366.4.22\n.3.....22\n\n";
    expect_lines_match("Liberties", &out, expected);
}

#[test]
fn test_liberties_after_move() {
    let board = Board::parse_board(
        9,
        9,
        ".........\n.....x...\n..oo..x..\n..x......\n......xx.\n..x..ox..\n.oxoo.oxx\nxxoo.o.ox\n.x.....oo\n",
        '\n',
    )
    .unwrap();

    let mut out = String::new();
    out.push('\n');
    out.push_str("After black\n");
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = get_loc(x, y, board.x_size);
            if board.colors[loc as usize] == C_EMPTY {
                out.push_str(
                    &board
                        .get_num_liberties_after_play(loc, P_BLACK, 100)
                        .to_string(),
                );
            } else {
                out.push('-');
            }
        }
        out.push('\n');
    }
    out.push('\n');
    out.push_str("After white\n");
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = get_loc(x, y, board.x_size);
            if board.colors[loc as usize] == C_EMPTY {
                out.push_str(
                    &board
                        .get_num_liberties_after_play(loc, P_WHITE, 100)
                        .to_string(),
                );
            } else {
                out.push('-');
            }
        }
        out.push('\n');
    }
    out.push('\n');

    let expected = "\nAfter black\n233335332\n34336-743\n33--37-63\n35-444863\n346446--6\n34-42--52\n3----0---\n----1-1--\n2-32322--\n\nAfter white\n233332332\n34663-243\n37--72-33\n33-644233\n342444--2\n33-67--12\n2----8---\n----8-4--\n0-56352--\n\n";
    expect_lines_match("Liberties after move", &out, expected);
}

#[test]
fn test_liberties_after_move_capped_at_2() {
    let board = Board::parse_board(
        9,
        9,
        ".........\n.....x...\n..oo..x..\n..x......\n......xx.\n..x..ox..\n.oxoo.oxx\nxxoo.o.ox\n.x.....oo\n",
        '\n',
    )
    .unwrap();

    let mut out = String::new();
    out.push('\n');
    out.push_str("After black\n");
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = get_loc(x, y, board.x_size);
            if board.colors[loc as usize] == C_EMPTY {
                out.push_str(
                    &board
                        .get_num_liberties_after_play(loc, P_BLACK, 2)
                        .to_string(),
                );
            } else {
                out.push('-');
            }
        }
        out.push('\n');
    }
    out.push('\n');
    out.push_str("After white\n");
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = get_loc(x, y, board.x_size);
            if board.colors[loc as usize] == C_EMPTY {
                out.push_str(
                    &board
                        .get_num_liberties_after_play(loc, P_WHITE, 2)
                        .to_string(),
                );
            } else {
                out.push('-');
            }
        }
        out.push('\n');
    }
    out.push('\n');

    let expected = "\nAfter black\n222222222\n22222-222\n22--22-22\n22-222222\n222222--2\n22-22--22\n2----0---\n----1-1--\n2-22222--\n\nAfter white\n222222222\n22222-222\n22--22-22\n22-222222\n222222--2\n22-22--12\n2----2---\n----2-2--\n0-22222--\n\n";
    expect_lines_match("Liberties after move capped at 2", &out, expected);
}

#[test]
fn test_liberties_after_move_capped_at_3() {
    let board = Board::parse_board(
        9,
        9,
        ".........\n.....x...\n..oo..x..\n..x......\n......xx.\n..x..ox..\n.oxoo.oxx\nxxoo.o.ox\n.x.....oo\n",
        '\n',
    )
    .unwrap();

    let mut out = String::new();
    out.push('\n');
    out.push_str("After black\n");
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = get_loc(x, y, board.x_size);
            if board.colors[loc as usize] == C_EMPTY {
                out.push_str(
                    &board
                        .get_num_liberties_after_play(loc, P_BLACK, 3)
                        .to_string(),
                );
            } else {
                out.push('-');
            }
        }
        out.push('\n');
    }
    out.push('\n');
    out.push_str("After white\n");
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = get_loc(x, y, board.x_size);
            if board.colors[loc as usize] == C_EMPTY {
                out.push_str(
                    &board
                        .get_num_liberties_after_play(loc, P_WHITE, 3)
                        .to_string(),
                );
            } else {
                out.push('-');
            }
        }
        out.push('\n');
    }
    out.push('\n');

    let expected = "\nAfter black\n233333332\n33333-333\n33--33-33\n33-333333\n333333--3\n33-32--32\n3----0---\n----1-1--\n2-32322--\n\nAfter white\n233332332\n33333-233\n33--32-33\n33-333233\n332333--2\n33-33--12\n2----3---\n----3-3--\n0-33332--\n\n";
    expect_lines_match("Liberties after move capped at 3", &out, expected);
}

#[test]
fn test_liberties_after_move_2() {
    let board = Board::parse_board(
        9,
        9,
        "x.xx...xx\noooxo.oxo\nxxxxo.ox.\nooooo..ox\n..xxx..o.\n........o\nx......xo\nox....xox\n.ox...xo.\n",
        '\n',
    )
    .unwrap();

    let mut out = String::new();
    out.push('\n');
    out.push_str("After black\n");
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = get_loc(x, y, board.x_size);
            if board.colors[loc as usize] == C_EMPTY {
                out.push_str(
                    &board
                        .get_num_liberties_after_play(loc, P_BLACK, 100)
                        .to_string(),
                );
            } else {
                out.push('-');
            }
        }
        out.push('\n');
    }
    out.push('\n');
    out.push_str("After white\n");
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = get_loc(x, y, board.x_size);
            if board.colors[loc as usize] == C_EMPTY {
                out.push_str(
                    &board
                        .get_num_liberties_after_play(loc, P_WHITE, 100)
                        .to_string(),
                );
            } else {
                out.push('-');
            }
        }
        out.push('\n');
    }
    out.push('\n');

    let expected = "\nAfter black\n-4--232--\n-----2---\n-----2--3\n-----32--\n26---73-1\n34656442-\n-444445--\n--4444---\n2--333--2\n\nAfter white\n-1--634--\n-----8---\n-----7--0\n-----77--\n66---35-3\n24333444-\n-244442--\n--2443---\n0--232--1\n\n";
    expect_lines_match("Liberties after move 2", &out, expected);
}

#[test]
fn test_liberties_after_move_2_capped_at_3() {
    let board = Board::parse_board(
        9,
        9,
        "x.xx...xx\noooxo.oxo\nxxxxo.ox.\nooooo..ox\n..xxx..o.\n........o\nx......xo\nox....xox\n.ox...xo.\n",
        '\n',
    )
    .unwrap();

    let mut out = String::new();
    out.push('\n');
    out.push_str("After black\n");
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = get_loc(x, y, board.x_size);
            if board.colors[loc as usize] == C_EMPTY {
                out.push_str(
                    &board
                        .get_num_liberties_after_play(loc, P_BLACK, 3)
                        .to_string(),
                );
            } else {
                out.push('-');
            }
        }
        out.push('\n');
    }
    out.push('\n');
    out.push_str("After white\n");
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = get_loc(x, y, board.x_size);
            if board.colors[loc as usize] == C_EMPTY {
                out.push_str(
                    &board
                        .get_num_liberties_after_play(loc, P_WHITE, 3)
                        .to_string(),
                );
            } else {
                out.push('-');
            }
        }
        out.push('\n');
    }
    out.push('\n');

    let expected = "\nAfter black\n-3--232--\n-----2---\n-----2--3\n-----32--\n23---33-1\n33333332-\n-333333--\n--3333---\n2--333--2\n\nAfter white\n-1--333--\n-----3---\n-----3--0\n-----33--\n33---33-3\n23333333-\n-233332--\n--2333---\n0--232--1\n\n";
    expect_lines_match("Liberties after move 2 capped at 3", &out, expected);
}

#[test]
fn test_distance() {
    let board = Board::new(17, 12);

    let mut out = String::new();
    let points = [
        (13, 6, 12, 3),
        (13, 6, 12, 4),
        (13, 6, 12, 5),
        (13, 6, 12, 6),
        (13, 6, 12, 7),
        (13, 6, 13, 3),
        (13, 6, 13, 4),
        (13, 6, 13, 5),
        (13, 6, 13, 6),
        (13, 6, 13, 7),
        (13, 6, 14, 3),
        (13, 6, 14, 4),
        (13, 6, 14, 5),
        (13, 6, 14, 6),
        (13, 6, 14, 7),
        (13, 6, 15, 3),
        (13, 6, 15, 4),
        (13, 6, 15, 5),
        (13, 6, 15, 6),
        (13, 6, 15, 7),
        (13, 6, 0, 0),
        (13, 6, 16, 11),
        (13, 6, 0, 11),
        (13, 6, 16, 0),
    ];

    for &(x0, y0, x1, y1) in &points {
        let d = distance(
            get_loc(x0, y0, board.x_size),
            get_loc(x1, y1, board.x_size),
            board.x_size,
        );
        out.push_str(&format!(
            "distance ({},{}) ({},{}) = {}\n",
            x0, y0, x1, y1, d
        ));
    }
    for &(x0, y0, x1, y1) in &points {
        let d = euclidean_distance_squared(
            get_loc(x0, y0, board.x_size),
            get_loc(x1, y1, board.x_size),
            board.x_size,
        );
        out.push_str(&format!(
            "euclideanSq ({},{}) ({},{}) = {}\n",
            x0, y0, x1, y1, d
        ));
    }

    let expected = "distance (13,6) (12,3) = 4\ndistance (13,6) (12,4) = 3\ndistance (13,6) (12,5) = 2\ndistance (13,6) (12,6) = 1\ndistance (13,6) (12,7) = 2\ndistance (13,6) (13,3) = 3\ndistance (13,6) (13,4) = 2\ndistance (13,6) (13,5) = 1\ndistance (13,6) (13,6) = 0\ndistance (13,6) (13,7) = 1\ndistance (13,6) (14,3) = 4\ndistance (13,6) (14,4) = 3\ndistance (13,6) (14,5) = 2\ndistance (13,6) (14,6) = 1\ndistance (13,6) (14,7) = 2\ndistance (13,6) (15,3) = 5\ndistance (13,6) (15,4) = 4\ndistance (13,6) (15,5) = 3\ndistance (13,6) (15,6) = 2\ndistance (13,6) (15,7) = 3\ndistance (13,6) (0,0) = 19\ndistance (13,6) (16,11) = 8\ndistance (13,6) (0,11) = 18\ndistance (13,6) (16,0) = 9\neuclideanSq (13,6) (12,3) = 10\neuclideanSq (13,6) (12,4) = 5\neuclideanSq (13,6) (12,5) = 2\neuclideanSq (13,6) (12,6) = 1\neuclideanSq (13,6) (12,7) = 2\neuclideanSq (13,6) (13,3) = 9\neuclideanSq (13,6) (13,4) = 4\neuclideanSq (13,6) (13,5) = 1\neuclideanSq (13,6) (13,6) = 0\neuclideanSq (13,6) (13,7) = 1\neuclideanSq (13,6) (14,3) = 10\neuclideanSq (13,6) (14,4) = 5\neuclideanSq (13,6) (14,5) = 2\neuclideanSq (13,6) (14,6) = 1\neuclideanSq (13,6) (14,7) = 2\neuclideanSq (13,6) (15,3) = 13\neuclideanSq (13,6) (15,4) = 8\neuclideanSq (13,6) (15,5) = 5\neuclideanSq (13,6) (15,6) = 4\neuclideanSq (13,6) (15,7) = 5\neuclideanSq (13,6) (0,0) = 205\neuclideanSq (13,6) (16,11) = 34\neuclideanSq (13,6) (0,11) = 194\neuclideanSq (13,6) (16,0) = 45\n";
    expect_lines_match("Distance", &out, expected);
}

#[test]
fn test_is_adjacent_to_pla() {
    let board = Board::parse_board(
        9,
        9,
        ".........\n.....x...\n..oo..x..\n..x......\n......xx.\n..x..ox..\n.oxoo.oxx\nxxoo.o.ox\n.x.....oo\n",
        '\n',
    )
    .unwrap();

    let mut out = String::new();
    out.push('\n');
    out.push_str("Adj black\n");
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = get_loc(x, y, board.x_size);
            out.push_str(&(board.is_adjacent_to_pla(loc, P_BLACK) as i32).to_string());
        }
        out.push('\n');
    }
    out.push('\n');
    out.push_str("Adj white\n");
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = get_loc(x, y, board.x_size);
            out.push_str(&(board.is_adjacent_to_pla(loc, P_WHITE) as i32).to_string());
        }
        out.push('\n');
    }
    out.push('\n');

    let expected = "\nAdj black\n000001000\n000010100\n001001010\n010100110\n001001111\n011101111\n111100111\n111000011\n111000001\n\nAdj white\n000000000\n001100000\n011110000\n001100000\n000001000\n010110100\n101111010\n011110111\n001101111\n\n";
    expect_lines_match("IsAdjacentToPla", &out, expected);
}

#[test]
fn test_would_be_capture() {
    let board = Board::parse_board(
        9,
        9,
        ".....oxx.\n..o.o.oox\n.oxo.oxx.\n.o.o..x..\n.xox.x.xo\n..x..oxo.\n....o.oxx\nxo...oxox\n.xo..x.oo\n",
        '\n',
    )
    .unwrap();

    let mut out = String::new();
    out.push('\n');
    out.push_str("WouldBeCapture black\n");
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = get_loc(x, y, board.x_size);
            out.push_str(&(board.would_be_capture(loc, P_BLACK) as i32).to_string());
        }
        out.push('\n');
    }
    out.push('\n');
    out.push_str("WouldBeCapture white\n");
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = get_loc(x, y, board.x_size);
            out.push_str(&(board.would_be_capture(loc, P_WHITE) as i32).to_string());
        }
        out.push('\n');
    }
    out.push('\n');

    let expected = "\nWouldBeCapture black\n000000000\n000001000\n000000000\n001000000\n000000000\n000000001\n000001000\n000000000\n000000100\n\nWouldBeCapture white\n000000001\n000000000\n000000000\n001000000\n000000100\n000000001\n000000000\n000000000\n100000100\n\n";
    expect_lines_match("wouldBeCapture", &out, expected);
}

#[test]
fn test_would_be_ko_capture() {
    let board = Board::parse_board(
        9,
        9,
        ".....oxx.\n..o.o.oox\n.oxo.oxx.\n.o.o..x..\n.xox.x.xo\n..x..oxo.\n....o.oxx\nxo...oxox\n.xo..x.oo\n",
        '\n',
    )
    .unwrap();

    let mut out = String::new();
    out.push('\n');
    out.push_str("WouldBeKo black\n");
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = get_loc(x, y, board.x_size);
            out.push_str(&(board.would_be_ko_capture(loc, P_BLACK) as i32).to_string());
        }
        out.push('\n');
    }
    out.push('\n');
    out.push_str("WouldBeKo white\n");
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = get_loc(x, y, board.x_size);
            out.push_str(&(board.would_be_ko_capture(loc, P_WHITE) as i32).to_string());
        }
        out.push('\n');
    }
    out.push('\n');

    let expected = "\nWouldBeKo black\n000000000\n000000000\n000000000\n000000000\n000000000\n000000000\n000001000\n000000000\n000000000\n\nWouldBeKo white\n000000000\n000000000\n000000000\n000000000\n000000100\n000000000\n000000000\n000000000\n100000000\n\n";
    expect_lines_match("wouldBeKoCapture", &out, expected);
}
