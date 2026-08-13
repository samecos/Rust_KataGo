//! Port of `Tests::runBoardIOTests` from `KataGo/cpp/tests/testboardbasic.cpp`.
//!
//! Covers location parsing across multiple board sizes and board parse/play
//! output formatting.

mod common;

use kata_core::test::expect_lines_match;
use kata_game::board::location::{self, get_x, get_y, to_string};
use kata_game::board::{Board, C_BLACK, C_EMPTY, C_WHITE, P_BLACK, P_WHITE};

/// Format a board the same way C++ `Board::printBoard` does for small boards.
fn format_board(board: &Board) -> String {
    let mut s = String::new();
    s.push_str(&format!("HASH: {}\n", board.pos_hash));

    let show_coords = board.x_size <= 50 && board.y_size <= 50;
    if show_coords {
        s.push_str("  ");
        for x in 0..board.x_size {
            let top = location::get_loc(x, 0, board.x_size);
            let col = to_string(top, board.x_size, board.y_size);
            let letters: String = col.chars().take_while(|c| c.is_alphabetic()).collect();
            s.push(' ');
            s.push_str(&letters);
        }
        s.push('\n');
    }

    for y in 0..board.y_size {
        if show_coords {
            s.push_str(&format!("{:>2} ", board.y_size - y));
        }
        for x in 0..board.x_size {
            let loc = location::get_loc(x, y, board.x_size);
            let c = match board.colors[loc as usize] {
                C_EMPTY => '.',
                C_BLACK => 'X',
                C_WHITE => 'O',
                _ => '?',
            };
            s.push(c);
            if x < board.x_size - 1 {
                s.push(' ');
            }
        }
        s.push('\n');
    }
    s.push('\n');
    s.push('\n');
    s
}

fn test_loc(out: &mut String, s: &str, x_size: i32, y_size: i32) {
    match location::try_of_string(s, x_size, y_size) {
        Some(loc) => {
            let loc_str = to_string(loc, x_size, y_size);
            let x = get_x(loc, x_size);
            let y = get_y(loc, x_size);
            out.push_str(&format!("{} {} x {} y {}\n", s, loc_str, x, y));
        }
        None => {
            out.push_str(&format!("Could not parse board location: {}\n", s));
        }
    }
}

#[test]
fn test_board_io() {
    let mut out = String::new();

    {
        let name = "Location parse test";
        let sizes = [9, 19, 26, 70];
        let test_strings = [
            "A1", "A0", "B2", "b2", "A", "B", "1", "pass", "H9", "I9", "J9", "J10", "K8", "k19",
            "a22", "y1", "z1", "aa1", "AA26", "AZ26", "BC50",
        ];

        for i in 0..sizes.len() {
            for j in 0..sizes.len() {
                if i > j + 1 || j > i + 1 {
                    continue;
                }
                let x_size = sizes[i];
                let y_size = sizes[j];
                out.push_str("----------------------------------\n");
                out.push_str(&format!("{} {}\n", x_size, y_size));
                for s in &test_strings {
                    test_loc(&mut out, s, x_size, y_size);
                }
            }
        }

        let expected = r#"
----------------------------------
9 9
A1 A1 x 0 y 8
Could not parse board location: A0
B2 B2 x 1 y 7
b2 B2 x 1 y 7
Could not parse board location: A
Could not parse board location: B
Could not parse board location: 1
pass pass x 0 y -1
H9 H9 x 7 y 0
Could not parse board location: I9
J9 J9 x 8 y 0
Could not parse board location: J10
Could not parse board location: K8
Could not parse board location: k19
Could not parse board location: a22
Could not parse board location: y1
Could not parse board location: z1
Could not parse board location: aa1
Could not parse board location: AA26
Could not parse board location: AZ26
Could not parse board location: BC50
----------------------------------
9 19
A1 A1 x 0 y 18
Could not parse board location: A0
B2 B2 x 1 y 17
b2 B2 x 1 y 17
Could not parse board location: A
Could not parse board location: B
Could not parse board location: 1
pass pass x 0 y -1
H9 H9 x 7 y 10
Could not parse board location: I9
J9 J9 x 8 y 10
J10 J10 x 8 y 9
Could not parse board location: K8
Could not parse board location: k19
Could not parse board location: a22
Could not parse board location: y1
Could not parse board location: z1
Could not parse board location: aa1
Could not parse board location: AA26
Could not parse board location: AZ26
Could not parse board location: BC50
----------------------------------
19 9
A1 A1 x 0 y 8
Could not parse board location: A0
B2 B2 x 1 y 7
b2 B2 x 1 y 7
Could not parse board location: A
Could not parse board location: B
Could not parse board location: 1
pass pass x 0 y -1
H9 H9 x 7 y 0
Could not parse board location: I9
J9 J9 x 8 y 0
Could not parse board location: J10
K8 K8 x 9 y 1
Could not parse board location: k19
Could not parse board location: a22
Could not parse board location: y1
Could not parse board location: z1
Could not parse board location: aa1
Could not parse board location: AA26
Could not parse board location: AZ26
Could not parse board location: BC50
----------------------------------
19 19
A1 A1 x 0 y 18
Could not parse board location: A0
B2 B2 x 1 y 17
b2 B2 x 1 y 17
Could not parse board location: A
Could not parse board location: B
Could not parse board location: 1
pass pass x 0 y -1
H9 H9 x 7 y 10
Could not parse board location: I9
J9 J9 x 8 y 10
J10 J10 x 8 y 9
K8 K8 x 9 y 11
k19 K19 x 9 y 0
Could not parse board location: a22
Could not parse board location: y1
Could not parse board location: z1
Could not parse board location: aa1
Could not parse board location: AA26
Could not parse board location: AZ26
Could not parse board location: BC50
----------------------------------
19 26
A1 A1 x 0 y 25
Could not parse board location: A0
B2 B2 x 1 y 24
b2 B2 x 1 y 24
Could not parse board location: A
Could not parse board location: B
Could not parse board location: 1
pass pass x 0 y -1
H9 H9 x 7 y 17
Could not parse board location: I9
J9 J9 x 8 y 17
J10 J10 x 8 y 16
K8 K8 x 9 y 18
k19 K19 x 9 y 7
a22 A22 x 0 y 4
Could not parse board location: y1
Could not parse board location: z1
Could not parse board location: aa1
Could not parse board location: AA26
Could not parse board location: AZ26
Could not parse board location: BC50
----------------------------------
26 19
A1 A1 x 0 y 18
Could not parse board location: A0
B2 B2 x 1 y 17
b2 B2 x 1 y 17
Could not parse board location: A
Could not parse board location: B
Could not parse board location: 1
pass pass x 0 y -1
H9 H9 x 7 y 10
Could not parse board location: I9
J9 J9 x 8 y 10
J10 J10 x 8 y 9
K8 K8 x 9 y 11
k19 K19 x 9 y 0
Could not parse board location: a22
y1 Y1 x 23 y 18
z1 Z1 x 24 y 18
aa1 AA1 x 25 y 18
Could not parse board location: AA26
Could not parse board location: AZ26
Could not parse board location: BC50
----------------------------------
26 26
A1 A1 x 0 y 25
Could not parse board location: A0
B2 B2 x 1 y 24
b2 B2 x 1 y 24
Could not parse board location: A
Could not parse board location: B
Could not parse board location: 1
pass pass x 0 y -1
H9 H9 x 7 y 17
Could not parse board location: I9
J9 J9 x 8 y 17
J10 J10 x 8 y 16
K8 K8 x 9 y 18
k19 K19 x 9 y 7
a22 A22 x 0 y 4
y1 Y1 x 23 y 25
z1 Z1 x 24 y 25
aa1 AA1 x 25 y 25
AA26 AA26 x 25 y 0
Could not parse board location: AZ26
Could not parse board location: BC50
----------------------------------
26 70
A1 A1 x 0 y 69
Could not parse board location: A0
B2 B2 x 1 y 68
b2 B2 x 1 y 68
Could not parse board location: A
Could not parse board location: B
Could not parse board location: 1
pass pass x 0 y -1
H9 H9 x 7 y 61
Could not parse board location: I9
J9 J9 x 8 y 61
J10 J10 x 8 y 60
K8 K8 x 9 y 62
k19 K19 x 9 y 51
a22 A22 x 0 y 48
y1 Y1 x 23 y 69
z1 Z1 x 24 y 69
aa1 AA1 x 25 y 69
AA26 AA26 x 25 y 44
Could not parse board location: AZ26
Could not parse board location: BC50
----------------------------------
70 26
A1 A1 x 0 y 25
Could not parse board location: A0
B2 B2 x 1 y 24
b2 B2 x 1 y 24
Could not parse board location: A
Could not parse board location: B
Could not parse board location: 1
pass pass x 0 y -1
H9 H9 x 7 y 17
Could not parse board location: I9
J9 J9 x 8 y 17
J10 J10 x 8 y 16
K8 K8 x 9 y 18
k19 K19 x 9 y 7
a22 A22 x 0 y 4
y1 Y1 x 23 y 25
z1 Z1 x 24 y 25
aa1 AA1 x 25 y 25
AA26 AA26 x 25 y 0
AZ26 AZ26 x 49 y 0
Could not parse board location: BC50
----------------------------------
70 70
A1 A1 x 0 y 69
Could not parse board location: A0
B2 B2 x 1 y 68
b2 B2 x 1 y 68
Could not parse board location: A
Could not parse board location: B
Could not parse board location: 1
pass pass x 0 y -1
H9 H9 x 7 y 61
Could not parse board location: I9
J9 J9 x 8 y 61
J10 J10 x 8 y 60
K8 K8 x 9 y 62
k19 K19 x 9 y 51
a22 A22 x 0 y 48
y1 Y1 x 23 y 69
z1 Z1 x 24 y 69
aa1 AA1 x 25 y 69
AA26 AA26 x 25 y 44
AZ26 AZ26 x 49 y 44
BC50 BC50 x 52 y 20
"#;
        expect_lines_match(name, &out, expected);
        out.clear();
    }

    {
        let name = "Parse test";
        let mut board = Board::parse_board(
            6,
            5,
            "\n ABCDEF\n5......\n4......\n3......\n2......\n1......\n",
            '\n',
        )
        .unwrap();
        let mut board2 = Board::parse_board(
            6,
            5,
            "\n   A B C D E F\n10 . . . . . .\n 9 . . . . . .\n 8 . . . . . .\n 7 . X . . . .\n 6 . . . . . .\n",
            '\n',
        )
        .unwrap();

        let b2 = location::try_of_string("B2", board.x_size, board.y_size).unwrap();
        board.play_move(b2, P_BLACK, true);
        let f1 = location::try_of_string("F1", board2.x_size, board2.y_size).unwrap();
        board2.play_move(f1, P_WHITE, true);

        out.push_str(&format_board(&board));
        out.push_str(&format_board(&board2));

        let expected = r#"
HASH: 044B62A5AFEFB1D31DD9CD3439D42BE4
   A B C D E F
 5 . . . . . .
 4 . . . . . .
 3 . . . . . .
 2 . X . . . .
 1 . . . . . .


HASH: 5428E20877BB47FDF201E12518FA1C0C
   A B C D E F
 5 . . . . . .
 4 . . . . . .
 3 . . . . . .
 2 . X . . . .
 1 . . . . . O
"#;
        expect_lines_match(name, &out, expected);
    }
}
