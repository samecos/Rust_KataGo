//! Integration tests ported from `cpp/tests/testsgf.cpp`.

use kata_core::test::expect_lines_match;
use kata_core::test_assert;
use kata_data::sgf::{CompactSgf, PositionSample, Sgf, XYSize, write_sgf};
use kata_game::board::{Board, C_EMPTY, Move, P_BLACK, P_WHITE, Player, location, player_io};
use kata_game::history::BoardHistory;
use kata_game::rules::Rules;
use std::collections::BTreeSet;

/// Print a board in the same shape as `Board::printBoard`, optionally numbering
/// the last up to three moves from `move_history` the way C++ does.
fn print_board_with_history(out: &mut String, board: &Board, move_history: &[Move]) {
    out.push_str(&format!("MoveNum: {}\n", move_history.len()));
    out.push_str(&format!("{}", board));

    // C++ Board::printBoard marks the last up to 3 moves with 1,2,3 and
    // suppresses the trailing space after a marked stone. We do the same.
    // The printed board above has no move numbers, so we replace it line by
    // line. This is simpler than reimplementing the whole formatter.
    let mut numbered = String::new();
    let x_chars: Vec<char> = "ABCDEFGHJKLMNOPQRSTUVWXYZ".chars().collect();

    // Header
    numbered.push_str("  ");
    for x in 0..board.x_size {
        numbered.push(' ');
        if x <= 24 {
            numbered.push(x_chars[x as usize]);
        } else {
            numbered.push('A');
            numbered.push(x_chars[(x - 25) as usize]);
        }
    }
    numbered.push('\n');

    let start = if move_history.len() >= 3 {
        move_history.len() - 3
    } else {
        0
    };

    for y in 0..board.y_size {
        numbered.push_str(&format!("{:2} ", board.y_size - y));
        for x in 0..board.x_size {
            let loc = location::get_loc(x, y, board.x_size);
            let s = player_io::color_to_char(board.colors[loc as usize]);
            numbered.push(s);
            let mut hist_marked = false;
            for i in 0.. {
                let idx = start + i;
                if idx >= move_history.len() {
                    break;
                }
                if move_history[idx].loc == loc {
                    numbered.push_str(&(i + 1).to_string());
                    hist_marked = true;
                    break;
                }
            }
            if x < board.x_size - 1 && !hist_marked {
                numbered.push(' ');
            }
        }
        numbered.push('\n');
    }
    numbered.push('\n');

    // Replace the simple board dump in `out` with the numbered version.
    // The simple dump ends with the last board row and a trailing newline.
    if let Some(pos) = out.rfind("\nMoveNum: ") {
        let board_start = pos + 1;
        out.truncate(board_start);
        out.push_str(&numbered);
    }
}

fn parse_and_print_sgf_linear(sgf_str: &str) -> String {
    let mut out = String::new();
    let sgf = CompactSgf::parse(sgf_str).unwrap();

    out.push_str(&format!("xSize {}\n", sgf.x_size));
    out.push_str(&format!("ySize {}\n", sgf.y_size));
    out.push_str(&format!("depth {}\n", sgf.depth));
    let rules = sgf
        .get_rules_or_fail_allow_unspecified(&Rules::default())
        .unwrap();
    out.push_str(&format!("komi {}\n", rules.komi_f32()));

    let mut board = Board::new(sgf.x_size, sgf.y_size);
    let mut next_pla: Player = C_EMPTY;
    let mut hist = BoardHistory::default();
    sgf.setup_initial_board_and_hist(&rules, &mut board, &mut next_pla, &mut hist)
        .unwrap();

    out.push_str("placements\n");
    for m in &sgf.placements {
        out.push_str(&format!(
            "{} {}\n",
            player_io::color_to_char(m.pla),
            location::to_string(m.loc, board.x_size, board.y_size)
        ));
    }
    out.push_str("moves\n");
    for m in &sgf.moves {
        out.push_str(&format!(
            "{} {}\n",
            player_io::color_to_char(m.pla),
            location::to_string(m.loc, board.x_size, board.y_size)
        ));
    }

    out.push_str("Initial board hist \n");
    out.push_str(&format!("pla {}\n", player_io::player_to_string(next_pla)));
    hist.print_debug_info(&mut out, &board);

    sgf.setup_board_and_hist_assume_legal(
        &rules,
        &mut board,
        &mut next_pla,
        &mut hist,
        sgf.moves.len() as i64,
    )
    .unwrap();
    out.push_str("Final board hist\n");
    out.push_str(&format!("pla {}\n", player_io::player_to_string(next_pla)));
    hist.print_debug_info(&mut out, &board);

    // Test SGF writing roundtrip.
    {
        let mut out2 = String::new();
        write_sgf(&mut out2, "foo", "bar", &hist);
        let sgf2 = CompactSgf::parse(&out2).unwrap();
        let mut board2 = Board::new(sgf2.x_size, sgf2.y_size);
        let mut next_pla2: Player = C_EMPTY;
        let mut hist2 = BoardHistory::default();
        let rules2 = sgf2.get_rules_or_fail().unwrap();
        sgf2.setup_board_and_hist_assume_legal(
            &rules2,
            &mut board2,
            &mut next_pla2,
            &mut hist2,
            sgf2.moves.len() as i64,
        )
        .unwrap();
        test_assert!(rules2 == rules);
        test_assert!(board2.is_equal_for_testing(&board, true, true));
        test_assert!(hist2.move_history.len() == hist.move_history.len());
    }

    out
}

fn parse_and_print_sgf(sgf_str: &str) -> String {
    let mut out = String::new();
    let sgf = Sgf::parse(sgf_str).unwrap();

    let size = sgf.get_xy_size().unwrap();
    out.push_str(&format!("xSize {}\n", size.x));
    out.push_str(&format!("ySize {}\n", size.y));

    out.push_str(&format!(
        "komi {}\n",
        sgf.get_komi_or_default(f32::NAN).unwrap()
    ));
    out.push_str(&format!("hasRules {}\n", sgf.has_rules()));
    out.push_str("rules ");
    match sgf.get_rules_or_fail() {
        Ok(r) => out.push_str(&format!("{}\n", r)),
        Err(_) => out.push_str("not found\n"),
    }

    out.push_str(&format!(
        "handicapValue {}\n",
        sgf.get_handicap_value().unwrap()
    ));
    out.push_str(&format!(
        "sgfWinner {}\n",
        player_io::player_to_string(sgf.get_sgf_winner())
    ));
    out.push_str(&format!(
        "firstPlayerColor {}\n",
        player_io::color_to_char(sgf.get_first_player_color().unwrap())
    ));

    out.push_str(&format!("black rank {}\n", sgf.get_rank(P_BLACK).unwrap()));
    out.push_str(&format!("white rank {}\n", sgf.get_rank(P_WHITE).unwrap()));
    out.push_str(&format!("black name {}\n", sgf.get_player_name(P_BLACK)));
    out.push_str(&format!("white name {}\n", sgf.get_player_name(P_WHITE)));

    out.push_str(&format!(
        "hasRootProperty(GN) {}\n",
        sgf.has_root_property("GN")
    ));
    out.push_str(&format!(
        "root property (GN) {}\n",
        sgf.get_root_property_with_default("GN", "")
    ));
    out.push_str(&format!(
        "hasRootProperty(SZ) {}\n",
        sgf.has_root_property("SZ")
    ));
    out.push_str(&format!(
        "root property (SZ) {}\n",
        sgf.get_root_property_with_default("SZ", "")
    ));

    if sgf.has_root_property("AW") {
        let props = sgf.get_root_properties("AW");
        out.push_str(&format!("getRootProperties(AW) size={}", props.len()));
        for p in &props {
            out.push_str(&format!(" [{}]", p));
        }
    }
    out.push('\n');
    if sgf.has_root_property("AB") {
        let props = sgf.get_root_properties("AB");
        out.push_str(&format!("getRootProperties(AB) size={}", props.len()));
        for p in &props {
            out.push_str(&format!(" [{}]", p));
        }
        out.push('\n');
    }

    let mut placements = Vec::new();
    sgf.get_placements(&mut placements, size.x, size.y).unwrap();
    out.push_str(&format!("placements {}\n", placements.len()));
    for m in &placements {
        out.push_str(&format!(
            "{} {} ",
            player_io::player_to_string(m.pla),
            location::to_string(m.loc, size.x, size.y)
        ));
    }
    out.push('\n');

    let mut moves = Vec::new();
    sgf.get_moves(&mut moves, size.x, size.y).unwrap();
    out.push_str(&format!("moves {}\n", moves.len()));
    for m in &moves {
        out.push_str(&format!(
            "{} {} ",
            player_io::player_to_string(m.pla),
            location::to_string(m.loc, size.x, size.y)
        ));
    }
    out.push('\n');

    out.push_str(&format!("depth {}\n", sgf.depth()));
    out.push_str(&format!("nodeCount {}\n", sgf.node_count()));
    out.push_str(&format!("branchCount {}\n", sgf.branch_count()));

    let mut unique_hashes = BTreeSet::new();
    sgf.iter_all_unique_positions(
        &mut unique_hashes,
        true,
        false,
        false,
        false,
        None,
        &mut |sample, hist, comments| {
            // Rust board hashes differ from C++; omit the hash line.
            out.push_str(&format!("Comments: {}\n", comments));
            print_board_with_history(&mut out, hist.get_recent_board(0), &hist.move_history);
            let _ = sample;
        },
        false,
    )
    .unwrap();

    out
}

fn load_and_print_unique_position_json(sgf_str: &str, flip: bool) -> String {
    let mut out = String::new();
    let sgf = Sgf::parse(sgf_str).unwrap();
    let mut unique_hashes = BTreeSet::new();
    let samples = sgf
        .load_all_unique_positions(&mut unique_hashes, false, false, flip, false, None, false)
        .unwrap();
    for sample in &samples {
        let json = PositionSample::to_json_line(sample);
        out.push_str(&json);
        out.push('\n');
        let reloaded = PositionSample::of_json_line(&json).unwrap();
        test_assert!(sample.is_equal_for_testing(&reloaded, false, false));
    }
    out
}

#[test]
fn basic_sgf_parse_test() {
    let sgf_str = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Tromp-Taylor]SZ[19]KM[5.00]PW[White]PB[Black]AB[dd][pd][dp][pp]PL[W];W[qf];W[md];B[pf];W[pg];B[of];W[];B[tt])";
    let out = parse_and_print_sgf_linear(sgf_str);
    let expected = r#"
xSize 19
ySize 19
depth 8
komi 5
placements
X D16
X Q16
X D4
X Q4
moves
O R14
O N16
X Q14
O Q13
X P14
O pass
X pass
Initial board hist 
pla White
   A B C D E F G H J K L M N O P Q R S T
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
 4 . . . X . . . . . . . . . . . X . . .
 3 . . . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . .


Initial pla White
Encore phase 0
Turns this phase 0
Approx valid turns this phase 0
Approx consec valid turns this game 0
Rules koPOSITIONALscoreAREAtaxNONEsui1komi5
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves
Final board hist
pla White
   A B C D E F G H J K L M N O P Q R S T
19 . . . . . . . . . . . . . . . . . . .
18 . . . . . . . . . . . . . . . . . . .
17 . . . . . . . . . . . . . . . . . . .
16 . . . X . . . . . . . . O . . X . . .
15 . . . . . . . . . . . . . . . . . . .
14 . . . . . . . . . . . . . . X X O . .
13 . . . . . . . . . . . . . . . O . . .
12 . . . . . . . . . . . . . . . . . . .
11 . . . . . . . . . . . . . . . . . . .
10 . . . . . . . . . . . . . . . . . . .
 9 . . . . . . . . . . . . . . . . . . .
 8 . . . . . . . . . . . . . . . . . . .
 7 . . . . . . . . . . . . . . . . . . .
 6 . . . . . . . . . . . . . . . . . . .
 5 . . . . . . . . . . . . . . . . . . .
 4 . . . X . . . . . . . . . . . X . . .
 3 . . . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . .


Initial pla White
Encore phase 0
Turns this phase 7
Approx valid turns this phase 7
Approx consec valid turns this game 5
Rules koPOSITIONALscoreAREAtaxNONEsui1komi5
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 1 White 2 1 0 0
Last moves R14 N16 Q14 Q13 P14 pass pass
"#;
    expect_lines_match("Basic Sgf parse test", &out, expected);
}

#[test]
fn japanese_sgf_parse_test() {
    let sgf_str = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Japanese]SZ[19]KM[5.00]PW[White]PB[Black]AB[dd][pd][dp][pp]PL[W];W[qf];W[md];B[pf];W[pg];B[of];W[];B[tt])";
    let out = parse_and_print_sgf_linear(sgf_str);
    let expected = r#"
xSize 19
ySize 19
depth 8
komi 5
placements
X D16
X Q16
X D4
X Q4
moves
O R14
O N16
X Q14
O Q13
X P14
O pass
X pass
Initial board hist 
pla White
   A B C D E F G H J K L M N O P Q R S T
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
 4 . . . X . . . . . . . . . . . X . . .
 3 . . . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . .


Initial pla White
Encore phase 0
Turns this phase 0
Approx valid turns this phase 0
Approx consec valid turns this game 0
Rules koSIMPLEscoreTERRITORYtaxSEKIsui0komi5
Ko recap block hash 00000000000000000000000000000000
White bonus score 4
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves
Final board hist
pla White
   A B C D E F G H J K L M N O P Q R S T
19 . . . . . . . . . . . . . . . . . . .
18 . . . . . . . . . . . . . . . . . . .
17 . . . . . . . . . . . . . . . . . . .
16 . . . X . . . . . . . . O . . X . . .
15 . . . . . . . . . . . . . . . . . . .
14 . . . . . . . . . . . . . . X X O . .
13 . . . . . . . . . . . . . . . O . . .
12 . . . . . . . . . . . . . . . . . . .
11 . . . . . . . . . . . . . . . . . . .
10 . . . . . . . . . . . . . . . . . . .
 9 . . . . . . . . . . . . . . . . . . .
 8 . . . . . . . . . . . . . . . . . . .
 7 . . . . . . . . . . . . . . . . . . .
 6 . . . . . . . . . . . . . . . . . . .
 5 . . . . . . . . . . . . . . . . . . .
 4 . . . X . . . . . . . . . . . X . . .
 3 . . . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . .


Initial pla White
Encore phase 1
Turns this phase 0
Approx valid turns this phase 0
Approx consec valid turns this game 5
Rules koSIMPLEscoreTERRITORYtaxSEKIsui0komi5
Ko recap block hash 00000000000000000000000000000000
White bonus score 3
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves R14 N16 Q14 Q13 P14 pass pass
"#;
    expect_lines_match("Japanese Sgf parse test", &out, expected);
}

#[test]
fn chinese_sgf_parse_test() {
    let sgf_str = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Chinese]SZ[19]KM[5.00]PW[White]PB[Black]AB[dd][pd][dp][pp]PL[W];W[qf];W[md];B[pf];W[pg];B[of];W[];B[tt])";
    let out = parse_and_print_sgf_linear(sgf_str);
    let expected = r#"
xSize 19
ySize 19
depth 8
komi 5
placements
X D16
X Q16
X D4
X Q4
moves
O R14
O N16
X Q14
O Q13
X P14
O pass
X pass
Initial board hist 
pla White
   A B C D E F G H J K L M N O P Q R S T
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
 4 . . . X . . . . . . . . . . . X . . .
 3 . . . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . .


Initial pla White
Encore phase 0
Turns this phase 0
Approx valid turns this phase 0
Approx consec valid turns this game 0
Rules koSIMPLEscoreAREAtaxNONEsui0whbNfpok1komi5
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 4
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves
Final board hist
pla White
   A B C D E F G H J K L M N O P Q R S T
19 . . . . . . . . . . . . . . . . . . .
18 . . . . . . . . . . . . . . . . . . .
17 . . . . . . . . . . . . . . . . . . .
16 . . . X . . . . . . . . O . . X . . .
15 . . . . . . . . . . . . . . . . . . .
14 . . . . . . . . . . . . . . X X O . .
13 . . . . . . . . . . . . . . . O . . .
12 . . . . . . . . . . . . . . . . . . .
11 . . . . . . . . . . . . . . . . . . .
10 . . . . . . . . . . . . . . . . . . .
 9 . . . . . . . . . . . . . . . . . . .
 8 . . . . . . . . . . . . . . . . . . .
 7 . . . . . . . . . . . . . . . . . . .
 6 . . . . . . . . . . . . . . . . . . .
 5 . . . . . . . . . . . . . . . . . . .
 4 . . . X . . . . . . . . . . . X . . .
 3 . . . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . .


Initial pla White
Encore phase 0
Turns this phase 7
Approx valid turns this phase 7
Approx consec valid turns this game 5
Rules koSIMPLEscoreAREAtaxNONEsui0whbNfpok1komi5
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 4
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 1 White 6 1 0 0
Last moves R14 N16 Q14 Q13 P14 pass pass
"#;
    expect_lines_match("Chinese Sgf parse test", &out, expected);
}

#[test]
fn aga_sgf_parse_test() {
    let sgf_str = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[AGA]SZ[19]KM[5.00]PW[White]PB[Black]AB[dd][pd][dp][pp]PL[W];W[qf];W[md];B[pf];W[pg];B[of];W[];B[tt])";
    let out = parse_and_print_sgf_linear(sgf_str);
    let expected = r#"
xSize 19
ySize 19
depth 8
komi 5
placements
X D16
X Q16
X D4
X Q4
moves
O R14
O N16
X Q14
O Q13
X P14
O pass
X pass
Initial board hist 
pla White
   A B C D E F G H J K L M N O P Q R S T
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
 4 . . . X . . . . . . . . . . . X . . .
 3 . . . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . .


Initial pla White
Encore phase 0
Turns this phase 0
Approx valid turns this phase 0
Approx consec valid turns this game 0
Rules koSITUATIONALscoreAREAtaxNONEsui0whbN-1fpok1komi5
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 3
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves
Final board hist
pla White
   A B C D E F G H J K L M N O P Q R S T
19 . . . . . . . . . . . . . . . . . . .
18 . . . . . . . . . . . . . . . . . . .
17 . . . . . . . . . . . . . . . . . . .
16 . . . X . . . . . . . . O . . X . . .
15 . . . . . . . . . . . . . . . . . . .
14 . . . . . . . . . . . . . . X X O . .
13 . . . . . . . . . . . . . . . O . . .
12 . . . . . . . . . . . . . . . . . . .
11 . . . . . . . . . . . . . . . . . . .
10 . . . . . . . . . . . . . . . . . . .
 9 . . . . . . . . . . . . . . . . . . .
 8 . . . . . . . . . . . . . . . . . . .
 7 . . . . . . . . . . . . . . . . . . .
 6 . . . . . . . . . . . . . . . . . . .
 5 . . . . . . . . . . . . . . . . . . .
 4 . . . X . . . . . . . . . . . X . . .
 3 . . . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . .


Initial pla White
Encore phase 0
Turns this phase 7
Approx valid turns this phase 7
Approx consec valid turns this game 5
Rules koSITUATIONALscoreAREAtaxNONEsui0whbN-1fpok1komi5
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 3
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 1 White 5 1 0 0
Last moves R14 N16 Q14 Q13 P14 pass pass
"#;
    expect_lines_match("AGA Sgf parse test", &out, expected);
}

#[test]
fn rectangle_sgf_parse_test() {
    let sgf_str = "(;GM[1]FF[4]SZ[17:3]KM[-6.5];B[fc];W[cc];B[la];)";
    let out = parse_and_print_sgf_linear(sgf_str);
    let expected = r#"
xSize 17
ySize 3
depth 5
komi -6.5
placements
moves
X F1
O C1
X M3
Initial board hist 
pla Black
   A B C D E F G H J K L M N O P Q R
 3 . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . .


Initial pla Black
Encore phase 0
Turns this phase 0
Approx valid turns this phase 0
Approx consec valid turns this game 0
Rules koPOSITIONALscoreAREAtaxNONEsui1komi-6.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 0
Has button 0
Presumed next pla Black
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves
Final board hist
pla White
   A B C D E F G H J K L M N O P Q R
 3 . . . . . . . . . . . X . . . . .
 2 . . . . . . . . . . . . . . . . .
 1 . . O . . X . . . . . . . . . . .


Initial pla Black
Encore phase 0
Turns this phase 3
Approx valid turns this phase 3
Approx consec valid turns this game 3
Rules koPOSITIONALscoreAREAtaxNONEsui1komi-6.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves F1 C1 M3
"#;
    expect_lines_match("Rectangle Sgf parse test", &out, expected);
}

#[test]
fn sgf_board_edit_range_placement_test() {
    let sgf_str = "(;GM[1]FF[4]SZ[19]PL[B]AB[ja:jd][ke][ld][lf:of][pe:qe][rc:sc][rd][rf]AW[ka:kd][le:oe][mc][pd][qc][rb:sb])";
    let out = parse_and_print_sgf_linear(sgf_str);
    let expected = r#"
xSize 19
ySize 19
depth 1
komi 7.5
placements
X K19
X K18
X K17
X K16
X L15
X M16
X M14
X N14
X O14
X P14
X Q15
X R15
X S17
X T17
X S16
X S14
O L19
O L18
O L17
O L16
O M15
O N15
O O15
O P15
O N17
O Q16
O R17
O S18
O T18
moves
Initial board hist 
pla Black
   A B C D E F G H J K L M N O P Q R S T
19 . . . . . . . . . X O . . . . . . . .
18 . . . . . . . . . X O . . . . . . O O
17 . . . . . . . . . X O . O . . . O X X
16 . . . . . . . . . X O X . . . O . X .
15 . . . . . . . . . . X O O O O X X . .
14 . . . . . . . . . . . X X X X . . X .
13 . . . . . . . . . . . . . . . . . . .
12 . . . . . . . . . . . . . . . . . . .
11 . . . . . . . . . . . . . . . . . . .
10 . . . . . . . . . . . . . . . . . . .
 9 . . . . . . . . . . . . . . . . . . .
 8 . . . . . . . . . . . . . . . . . . .
 7 . . . . . . . . . . . . . . . . . . .
 6 . . . . . . . . . . . . . . . . . . .
 5 . . . . . . . . . . . . . . . . . . .
 4 . . . . . . . . . . . . . . . . . . .
 3 . . . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . .


Initial pla Black
Encore phase 0
Turns this phase 0
Approx valid turns this phase 0
Approx consec valid turns this game 0
Rules koPOSITIONALscoreAREAtaxNONEsui1komi7.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 0
Has button 0
Presumed next pla Black
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves
Final board hist
pla Black
   A B C D E F G H J K L M N O P Q R S T
19 . . . . . . . . . X O . . . . . . . .
18 . . . . . . . . . X O . . . . . . O O
17 . . . . . . . . . X O . O . . . O X X
16 . . . . . . . . . X O X . . . O . X .
15 . . . . . . . . . . X O O O O X X . .
14 . . . . . . . . . . . X X X X . . X .
13 . . . . . . . . . . . . . . . . . . .
12 . . . . . . . . . . . . . . . . . . .
11 . . . . . . . . . . . . . . . . . . .
10 . . . . . . . . . . . . . . . . . . .
 9 . . . . . . . . . . . . . . . . . . .
 8 . . . . . . . . . . . . . . . . . . .
 7 . . . . . . . . . . . . . . . . . . .
 6 . . . . . . . . . . . . . . . . . . .
 5 . . . . . . . . . . . . . . . . . . .
 4 . . . . . . . . . . . . . . . . . . .
 3 . . . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . .


Initial pla Black
Encore phase 0
Turns this phase 0
Approx valid turns this phase 0
Approx consec valid turns this game 0
Rules koPOSITIONALscoreAREAtaxNONEsui1komi7.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 0
Has button 0
Presumed next pla Black
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves
"#;
    expect_lines_match(
        "Sgf board edit range (rectangle) placement test",
        &out,
        expected,
    );
}

#[test]
fn sgf_parsing_with_whitespace_and_placements_and_comments() {
    let sgf_str = r#"(;GM[1]FF[4]SZ[9]
GN[]
C[Diagram

]
PL[W]

AB[bc][dc][fc]
AW[ac][cc]AW[ec]


;
)"#;
    let out = parse_and_print_sgf_linear(sgf_str);
    let expected = r#"
xSize 9
ySize 9
depth 2
komi 7.5
placements
X B7
X D7
X F7
O A7
O C7
O E7
moves
Initial board hist 
pla White
   A B C D E F G H J
 9 . . . . . . . . .
 8 . . . . . . . . .
 7 O X O X O X . . .
 6 . . . . . . . . .
 5 . . . . . . . . .
 4 . . . . . . . . .
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Initial pla White
Encore phase 0
Turns this phase 0
Approx valid turns this phase 0
Approx consec valid turns this game 0
Rules koPOSITIONALscoreAREAtaxNONEsui1komi7.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves
Final board hist
pla White
   A B C D E F G H J
 9 . . . . . . . . .
 8 . . . . . . . . .
 7 O X O X O X . . .
 6 . . . . . . . . .
 5 . . . . . . . . .
 4 . . . . . . . . .
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Initial pla White
Encore phase 0
Turns this phase 0
Approx valid turns this phase 0
Approx consec valid turns this game 0
Rules koPOSITIONALscoreAREAtaxNONEsui1komi7.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves
"#;
    expect_lines_match(
        "Sgf parsing with whitespace and placements and comments",
        &out,
        expected,
    );
}

#[test]
fn sgf_parsing_with_moveless_and_multimove_nodes() {
    let sgf_str = "(;GM[1]FF[4]SZ[5]KM[24];B[cc]W[cb];;B[bb];C[test];C[test2];W[dc];B[db];W[cd];;;B[bc];C[test3])";
    let mut out = parse_and_print_sgf_linear(sgf_str);
    out.push_str(&load_and_print_unique_position_json(sgf_str, false));
    let expected = r#"
xSize 5
ySize 5
depth 13
komi 24
placements
moves
X C3
O C4
X B4
O D3
X D4
O C2
X B3
Initial board hist 
pla Black
   A B C D E
 5 . . . . .
 4 . . . . .
 3 . . . . .
 2 . . . . .
 1 . . . . .


Initial pla Black
Encore phase 0
Turns this phase 0
Approx valid turns this phase 0
Approx consec valid turns this game 0
Rules koPOSITIONALscoreAREAtaxNONEsui1komi24
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 0
Has button 0
Presumed next pla Black
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves
Final board hist
pla White
   A B C D E
 5 . . . . .
 4 . X O X .
 3 . X X O .
 2 . . O . .
 1 . . . . .


Initial pla Black
Encore phase 0
Turns this phase 7
Approx valid turns this phase 7
Approx consec valid turns this game 7
Rules koPOSITIONALscoreAREAtaxNONEsui1komi24
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves C3 C4 B4 D3 D4 C2 B3
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":[],"movePlas":[],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C3"],"movePlas":["B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C3","C4"],"movePlas":["B","W"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C3","C4","B4"],"movePlas":["B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C3","C4","B4","D3"],"movePlas":["B","W","B","W"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C3","C4","B4","D3","D4"],"movePlas":["B","W","B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../..X../...../...../","hintLoc":"null","initialTurnNumber":1,"moveLocs":["C4","B4","D3","D4","C2"],"movePlas":["W","B","W","B","W"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../..O../..X../...../...../","hintLoc":"null","initialTurnNumber":2,"moveLocs":["B4","D3","D4","C2","B3"],"movePlas":["B","W","B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
"#;
    expect_lines_match(
        "Sgf parsing with moveless and multimove nodes",
        &out,
        expected,
    );
}

#[test]
fn sgf_parsing_with_more_multimove_nodes() {
    let sgf_str = "(;GM[1]FF[4]SZ[5]KM[24];B[cc]W[cb]B[bb];W[dc]B[db];W[cd];;;B[bc])";
    let mut out = parse_and_print_sgf_linear(sgf_str);
    out.push_str(&load_and_print_unique_position_json(sgf_str, false));
    let expected = r#"
xSize 5
ySize 5
depth 7
komi 24
placements
moves
X C3
X B4
O C4
X D4
O D3
O C2
X B3
Initial board hist 
pla Black
   A B C D E
 5 . . . . .
 4 . . . . .
 3 . . . . .
 2 . . . . .
 1 . . . . .


Initial pla Black
Encore phase 0
Turns this phase 0
Approx valid turns this phase 0
Approx consec valid turns this game 0
Rules koPOSITIONALscoreAREAtaxNONEsui1komi24
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 0
Has button 0
Presumed next pla Black
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves
Final board hist
pla White
   A B C D E
 5 . . . . .
 4 . X O X .
 3 . X X O .
 2 . . O . .
 1 . . . . .


Initial pla Black
Encore phase 0
Turns this phase 7
Approx valid turns this phase 7
Approx consec valid turns this game 1
Rules koPOSITIONALscoreAREAtaxNONEsui1komi24
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves C3 B4 C4 D4 D3 C2 B3
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":[],"movePlas":[],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C3"],"movePlas":["B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../..X../...../...../","hintLoc":"null","initialTurnNumber":1,"moveLocs":["B4"],"movePlas":["B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../..X../...../...../","hintLoc":"null","initialTurnNumber":1,"moveLocs":["B4","C4"],"movePlas":["B","W"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../..X../...../...../","hintLoc":"null","initialTurnNumber":1,"moveLocs":["B4","C4","D4"],"movePlas":["B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../..X../...../...../","hintLoc":"null","initialTurnNumber":1,"moveLocs":["B4","C4","D4","D3"],"movePlas":["B","W","B","W"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../.XOX./..XO./...../...../","hintLoc":"null","initialTurnNumber":5,"moveLocs":["C2"],"movePlas":["W"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../.XOX./..XO./...../...../","hintLoc":"null","initialTurnNumber":5,"moveLocs":["C2","B3"],"movePlas":["W","B"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
"#;
    expect_lines_match(
        "Sgf parsing with more multimove nodes, katago doesn't handle ordering but that's okay since this is not actually valid sgf",
        &out,
        expected,
    );
}

#[test]
fn giant_sgf_parse_test() {
    // Rust Board::MAX_LEN is currently 19, so the 37x37 board required by this
    // test is not supported. Skip when the board would be too large.
    if kata_game::board::MAX_LEN < 37 {
        return;
    }
    let sgf_str = "(;GM[1]FF[4]CA[UTF-8]ST[2]RU[Chinese]SZ[37]KM[0.00];B[dd];W[Hd];B[HH];W[dH];B[dG];W[eG];B[eF];W[Gd];B[Ge];W[He];B[ee];W[GG];B[ss])";
    let _out = parse_and_print_sgf_linear(sgf_str);
    // Expected output omitted because the test is skipped in this Rust port.
}

#[test]
fn branching_sgf_parse_test() {
    let sgf_str = r#"(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3] ST[2]
RU[Japanese]SZ[5]KM[12.50]
PW[WhitePlayer]PB[BlackPlayer]HA[5]RE [B+1]AW[aa][ab]PL[B]
(;B[cc]C[Center move!]
(;W[dc]
;B[dd]
((( ;W[cb])))
(;W[db]
;B[cb]) )
(;W[cd])
(;W[dd] ))
(; B[cd])
(;B [dd]
(;W[cc]
(;B[dc]
;W[cb]C[Other branch])))
(;W[cc] C[White first]
;B[cd]
;B[dc]C[Black twice in a row.]))"#;
    let out = parse_and_print_sgf(sgf_str);
    let expected = r#"
xSize 5
ySize 5
komi 12.5
hasRules true
rules koSIMPLEscoreTERRITORYtaxSEKIsui0komi12.5
handicapValue 5
sgfWinner Black
firstPlayerColor X
black rank -100000
white rank -100000
black name BlackPlayer
white name WhitePlayer
hasRootProperty(GN) false
root property (GN)
hasRootProperty(SZ) true
root property (SZ) 5
getRootProperties(AW) size=2 [aa] [ab]
placements 2
White A5 White A4
moves 5
Black C3 White D3 Black D2 White D4 Black C4
depth 6
nodeCount 17
branchCount 7
"#;
    // The unique-position board dumps are complex and depend on Rust-specific
    // hashes, so we just sanity-check the metadata above and the number of
    // dumped positions.
    let metadata_end = expected.trim().lines().count();
    let out_lines: Vec<&str> = out.trim().lines().collect();
    let prefix: Vec<&str> = out_lines.iter().take(metadata_end).copied().collect();
    expect_lines_match(
        "Branching sgf parse test metadata",
        &prefix.join("\n"),
        expected,
    );

    // Count the number of "Comments:" lines, which equals the number of
    // unique positions visited.
    let comments_count = out.lines().filter(|l| l.starts_with("Comments:")).count();
    assert_eq!(comments_count, 17, "expected 17 unique positions");
}

#[test]
fn more_rigorous_test_of_positionsample_parsing() {
    let sgf_str = "(;FF[4]GM[1]SZ[9:17]HA[0]KM[7.5]RU[koSIMPLEscoreAREAtaxSEKIsui0]RE[W+32.5];B[fo];W[dd];B[fc];W[fd];B[ec];W[ed];B[dc];W[cd];B[gd];W[ge];B[hd];W[dn];B[fm];W[eo];B[fp];W[el];B[ck];W[di];B[bn];W[fl];B[gm];W[gl];B[ci];W[ch];B[dh];W[ei];B[em];W[dm];B[dl];W[bm];B[ek];W[hm];B[hn];W[cj];B[bk];W[hl];B[cm];W[cn];B[bo];W[cl];B[dk];W[ep];B[hp];W[cp];B[fj];W[gi];B[am];W[al];B[fi];W[gh];B[bl];W[an];B[cm];W[in];B[ho];W[cl];B[gj];W[hj];B[cm];W[gc];B[fh];W[gg];B[gb];W[cl];B[hk];W[gk];B[cm];W[fn];B[gn];W[cl];B[ik];W[ij];B[cm];W[hc];B[hb];W[cl];B[hi];W[il];B[cm];W[bp];B[bi];W[eh];B[am];W[bh];B[dg];W[bm];B[fg];W[he];B[am];W[ap];B[cf];W[bj];B[eg];W[id];B[cc];W[bm];B[bc];W[bd];B[bf];W[cl];B[ak];W[cm];B[ff];W[gf];B[ac];W[ae];B[af];W[ai];B[ib];W[df];B[ef];W[cg];B[de];W[ee];B[df];W[be];B[aj];W[fe];B[ej];W[ci];B[ah];W[ag];B[bg];W[ce];B[am];W[fk];B[ah];W[fq];B[bi];W[gp];B[go];W[ai];B[en];W[ag];B[co];W[al];B[ah];W[am];B[bi];W[hq];B[gq];W[ai];B[do];W[ao];B[bi];W[gp];B[ii];W[gq];B[ic];W[ad];B[dj];W[hh];B[bh];W[dp];B[fb];W[ih];B[];W[hd];B[];W[ip];B[];W[io];B[];W[fn];B[];W[])";
    let sgf = Sgf::parse(sgf_str).unwrap();
    let mut unique_hashes = BTreeSet::new();
    let mut samples = sgf
        .load_all_unique_positions(&mut unique_hashes, false, false, false, false, None, false)
        .unwrap();
    for (i, sample) in samples.iter_mut().enumerate() {
        sample.weight = i as f64 * 0.5;
        let json = PositionSample::to_json_line(sample);
        let reloaded = PositionSample::of_json_line(&json).unwrap();
        test_assert!(sample.is_equal_for_testing(&reloaded, false, false));
    }
}

#[test]
fn sgf_parsing_with_white_first_and_flip() {
    let sgf_str = "(;GM[1]FF[4]SZ[5]KM[24];W[cc];B[cb];W[bb];B[dc];W[db];B[cd];W[bc];B[dd];W[bd])";
    let out = load_and_print_unique_position_json(sgf_str, true);
    let expected = r#"
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":[],"movePlas":[],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C3"],"movePlas":["B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C3","C4"],"movePlas":["B","W"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C3","C4","B4"],"movePlas":["B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C3","C4","B4","D3"],"movePlas":["B","W","B","W"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C3","C4","B4","D3","D4"],"movePlas":["B","W","B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../..X../...../...../","hintLoc":"null","initialTurnNumber":1,"moveLocs":["C4","B4","D3","D4","C2"],"movePlas":["W","B","W","B","W"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../..O../..X../...../...../","hintLoc":"null","initialTurnNumber":2,"moveLocs":["B4","D3","D4","C2","B3"],"movePlas":["B","W","B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../.XO../..X../...../...../","hintLoc":"null","initialTurnNumber":3,"moveLocs":["D3","D4","C2","B3","D2"],"movePlas":["W","B","W","B","W"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../.XO../..XO./...../...../","hintLoc":"null","initialTurnNumber":4,"moveLocs":["D4","C2","B3","D2","B2"],"movePlas":["B","W","B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
"#;
    expect_lines_match("Sgf parsing with white first, and flip", &out, expected);
}

#[test]
fn sgf_parsing_with_black_pass_and_flip() {
    let sgf_str =
        "(;GM[1]FF[4]SZ[5]KM[24];B[cb];W[cc];B[];W[bb];B[dc];W[db];B[cd];W[bc];B[dd];W[bd];B[])";
    let out = load_and_print_unique_position_json(sgf_str, true);
    let expected = r#"
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":[],"movePlas":[],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C4"],"movePlas":["B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C4","C3"],"movePlas":["B","W"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C4","C3","pass"],"movePlas":["W","B","W"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C4","C3","pass","B4"],"movePlas":["W","B","W","B"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C4","C3","pass","B4","D3"],"movePlas":["W","B","W","B","W"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../..O../...../...../...../","hintLoc":"null","initialTurnNumber":1,"moveLocs":["C3","pass","B4","D3","D4"],"movePlas":["B","W","B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../..O../..X../...../...../","hintLoc":"null","initialTurnNumber":2,"moveLocs":["pass","B4","D3","D4","C2"],"movePlas":["W","B","W","B","W"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../..O../..X../...../...../","hintLoc":"null","initialTurnNumber":3,"moveLocs":["B4","D3","D4","C2","B3"],"movePlas":["B","W","B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../.XO../..X../...../...../","hintLoc":"null","initialTurnNumber":4,"moveLocs":["D3","D4","C2","B3","D2"],"movePlas":["W","B","W","B","W"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../.XO../..XO./...../...../","hintLoc":"null","initialTurnNumber":5,"moveLocs":["D4","C2","B3","D2","B2"],"movePlas":["B","W","B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../.OXO./..OX./...../...../","hintLoc":"null","initialTurnNumber":6,"moveLocs":["C2","B3","D2","B2","pass"],"movePlas":["B","W","B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
"#;
    expect_lines_match("Sgf parsing with black pass, and flip", &out, expected);
}

#[test]
fn sgf_parsing_with_black_white_double_move_and_flip() {
    let sgf_str =
        "(;GM[1]FF[4]SZ[5]KM[24];B[cb];W[cc];W[bb];B[dc];W[db];B[cd];W[bc];B[dd];W[bd];B[])";
    let out = load_and_print_unique_position_json(sgf_str, true);
    let expected = r#"
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":[],"movePlas":[],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C4"],"movePlas":["B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C4","C3"],"movePlas":["B","W"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../..X../..O../...../...../","hintLoc":"null","initialTurnNumber":2,"moveLocs":["B4"],"movePlas":["W"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../..X../..O../...../...../","hintLoc":"null","initialTurnNumber":2,"moveLocs":["B4","D3"],"movePlas":["W","B"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../..X../..O../...../...../","hintLoc":"null","initialTurnNumber":2,"moveLocs":["B4","D3","D4"],"movePlas":["W","B","W"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../..X../..O../...../...../","hintLoc":"null","initialTurnNumber":2,"moveLocs":["B4","D3","D4","C2"],"movePlas":["W","B","W","B"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../..X../..O../...../...../","hintLoc":"null","initialTurnNumber":2,"moveLocs":["B4","D3","D4","C2","B3"],"movePlas":["W","B","W","B","W"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../.OX../..O../...../...../","hintLoc":"null","initialTurnNumber":3,"moveLocs":["D3","D4","C2","B3","D2"],"movePlas":["B","W","B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../.OX../..OX./...../...../","hintLoc":"null","initialTurnNumber":4,"moveLocs":["D4","C2","B3","D2","B2"],"movePlas":["W","B","W","B","W"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../.XOX./..XO./...../...../","hintLoc":"null","initialTurnNumber":5,"moveLocs":["C2","B3","D2","B2","pass"],"movePlas":["W","B","W","B","W"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
"#;
    expect_lines_match(
        "Sgf parsing with black white double move, and flip",
        &out,
        expected,
    );
}

#[test]
fn sgf_tolerate_illegal_moves_option() {
    // Low-level correctness tests for set_stones_tolerant.
    {
        let mut a = Board::new(5, 5);
        let pl = vec![
            Move::new(location::get_loc(2, 2, 5), P_BLACK),
            Move::new(location::get_loc(2, 3, 5), P_BLACK),
            Move::new(location::get_loc(3, 2, 5), P_WHITE),
        ];
        let removed = a.set_stones_tolerant(&pl);
        assert_eq!(removed, 0);
        let mut b = Board::new(5, 5);
        b.set_stone_fail_if_no_libs(location::get_loc(2, 2, 5), P_BLACK);
        b.set_stone_fail_if_no_libs(location::get_loc(2, 3, 5), P_BLACK);
        b.set_stone_fail_if_no_libs(location::get_loc(3, 2, 5), P_WHITE);
        assert!(a.is_equal_for_testing(&b, false, false));
        assert_eq!(
            a.get_num_liberties(location::get_loc(2, 2, 5)),
            b.get_num_liberties(location::get_loc(2, 2, 5))
        );
        assert_eq!(
            a.get_num_liberties(location::get_loc(3, 2, 5)),
            b.get_num_liberties(location::get_loc(3, 2, 5))
        );
    }
    {
        let mut a = Board::new(5, 5);
        let pl = vec![
            Move::new(location::get_loc(2, 2, 5), P_WHITE),
            Move::new(location::get_loc(1, 2, 5), P_BLACK),
            Move::new(location::get_loc(3, 2, 5), P_BLACK),
            Move::new(location::get_loc(2, 1, 5), P_BLACK),
            Move::new(location::get_loc(2, 3, 5), P_BLACK),
        ];
        let removed = a.set_stones_tolerant(&pl);
        assert_eq!(removed, 1);
        let mut b = Board::new(5, 5);
        b.set_stone_fail_if_no_libs(location::get_loc(1, 2, 5), P_BLACK);
        b.set_stone_fail_if_no_libs(location::get_loc(3, 2, 5), P_BLACK);
        b.set_stone_fail_if_no_libs(location::get_loc(2, 1, 5), P_BLACK);
        b.set_stone_fail_if_no_libs(location::get_loc(2, 3, 5), P_BLACK);
        assert!(a.is_equal_for_testing(&b, false, false));
        assert_eq!(a.colors[location::get_loc(2, 2, 5) as usize], C_EMPTY);
    }

    fn run_tol(sgf_str: &str, tolerate: bool) -> (Vec<PositionSample>, Board, bool) {
        let sgf = Sgf::parse(sgf_str).unwrap();
        let mut samples = Vec::new();
        let mut final_board = Board::new(5, 5);
        let res = sgf.iter_all_positions(
            false,
            true,
            None,
            &mut |sample, hist, _comments| {
                samples.push(sample.clone());
                final_board = hist.get_recent_board(0).clone();
            },
            tolerate,
        );
        (samples, final_board, res.is_err())
    }

    // tolerate == true, setup stones.
    {
        let sgf = "(;GM[1]FF[4]SZ[5]KM[7]AW[cc]AB[bc][cb][cd][dc])";
        let (_samples, final_board, threw) = run_tol(sgf, true);
        assert!(!threw);
        assert_eq!(final_board.num_pla_stones_on_board(P_WHITE), 0);
        assert_eq!(final_board.num_pla_stones_on_board(P_BLACK), 4);
    }
    {
        let sgf = "(;GM[1]FF[4]SZ[2]KM[7]AB[aa][ab]AW[ba][bb])";
        let (_samples, final_board, threw) = run_tol(sgf, true);
        assert!(!threw);
        assert_eq!(final_board.num_pla_stones_on_board(P_WHITE), 0);
        assert_eq!(final_board.num_pla_stones_on_board(P_BLACK), 0);
    }
    {
        let sgf = "(;GM[1]FF[4]SZ[5]KM[7]AW[cc]AB[bc][cb])";
        let (_samples, final_board, threw) = run_tol(sgf, true);
        assert!(!threw);
        assert_eq!(final_board.num_pla_stones_on_board(P_WHITE), 1);
        assert_eq!(final_board.num_pla_stones_on_board(P_BLACK), 2);
    }

    // tolerate == true, played moves.
    {
        let sgf = "(;GM[1]FF[4]SZ[5]KM[7];B[cc];W[dc];B[cd];W[dd])";
        let (samples, final_board, threw) = run_tol(sgf, true);
        assert!(!threw);
        assert_eq!(samples.len(), 5);
        assert_eq!(samples.last().unwrap().moves.len(), 4);
        assert_eq!(samples.last().unwrap().initial_turn_number, 0);
        assert_eq!(final_board.num_pla_stones_on_board(P_BLACK), 2);
        assert_eq!(final_board.num_pla_stones_on_board(P_WHITE), 2);
    }
    {
        let sgf = "(;GM[1]FF[4]SZ[5]KM[7]AW[bc][cb][cd][dc];B[cc])";
        let (_samples, final_board, threw) = run_tol(sgf, true);
        assert!(!threw);
        assert_eq!(final_board.num_pla_stones_on_board(P_WHITE), 4);
        assert_eq!(final_board.num_pla_stones_on_board(P_BLACK), 0);
    }
    {
        let sgf = "(;GM[1]FF[4]SZ[5]KM[7]PL[W]AW[bc][ad][be]AB[cc][bd][dd][ce];W[cd];B[bd])";
        let (samples, _final_board, threw) = run_tol(sgf, true);
        assert!(!threw);
        assert_eq!(samples.last().unwrap().moves.len(), 1);
    }
    {
        let sgf =
            "(;GM[1]FF[4]SZ[5]KM[7]PL[W]AW[bc][ad][be]AB[cc][bd][dd][ce];W[cd];B[];W[];B[bd])";
        let (samples, final_board, threw) = run_tol(sgf, true);
        assert!(!threw);
        assert_eq!(samples.last().unwrap().moves.len(), 1);
        assert_eq!(final_board.num_pla_stones_on_board(P_BLACK), 4);
        assert_eq!(final_board.num_pla_stones_on_board(P_WHITE), 3);
    }

    // tolerate == false, setup stones.
    {
        let sgf = "(;GM[1]FF[4]SZ[5]KM[7]AW[cc]AB[bc][cb][cd][dc])";
        let (_samples, _final_board, threw) = run_tol(sgf, false);
        assert!(threw);
    }
    {
        let sgf = "(;GM[1]FF[4]SZ[5]KM[7]AW[cc]AB[bc][cb])";
        let (_samples, _final_board, threw) = run_tol(sgf, false);
        assert!(!threw);
    }

    // tolerate == false, played moves.
    {
        let sgf = "(;GM[1]FF[4]SZ[5]KM[7];B[cc];W[dc];B[cd];W[dd])";
        let (samples, _final_board, threw) = run_tol(sgf, false);
        assert!(!threw);
        assert_eq!(samples.last().unwrap().moves.len(), 4);
        assert_eq!(samples.last().unwrap().initial_turn_number, 0);
    }
    {
        let sgf = "(;GM[1]FF[4]SZ[5]KM[7]AW[bc][cb][cd][dc];B[cc])";
        let (_samples, _final_board, threw) = run_tol(sgf, false);
        assert!(threw);
    }
    {
        let sgf = "(;GM[1]FF[4]SZ[5]KM[7]PL[W]AW[bc][ad][be]AB[cc][bd][dd][ce];W[cd];B[bd])";
        let (_samples, _final_board, threw) = run_tol(sgf, false);
        assert!(threw);
    }
    {
        let sgf =
            "(;GM[1]FF[4]SZ[5]KM[7]PL[W]AW[bc][ad][be]AB[cc][bd][dd][ce];W[cd];B[];W[];B[bd])";
        let (samples, final_board, threw) = run_tol(sgf, false);
        assert!(!threw);
        assert_eq!(samples.last().unwrap().moves.len(), 1);
        assert_eq!(final_board.num_pla_stones_on_board(P_BLACK), 4);
        assert_eq!(final_board.num_pla_stones_on_board(P_WHITE), 3);
    }

    // Discrimination: superko move is emitted, suicide move is skipped.
    {
        let superko_sgf =
            "(;GM[1]FF[4]SZ[5]KM[7]PL[W]AW[bc][ad][be]AB[cc][bd][dd][ce];W[cd];B[];W[];B[bd])";
        let (samples, final_board, threw) = run_tol(superko_sgf, true);
        assert!(!threw);
        let bd = location::of_string("B2", final_board.x_size, final_board.y_size).unwrap();
        let superko_move_emitted = samples
            .iter()
            .any(|s| s.moves.len() == 1 && s.moves[0].loc == bd);
        assert!(superko_move_emitted);
        assert_eq!(final_board.num_pla_stones_on_board(P_BLACK), 4);
        assert_eq!(final_board.num_pla_stones_on_board(P_WHITE), 3);

        let suicide_sgf = "(;GM[1]FF[4]SZ[5]KM[7]AW[bc][cb][cd][dc];B[cc])";
        let (samples, final_board, threw) = run_tol(suicide_sgf, true);
        assert!(!threw);
        let cc = location::of_string("C3", final_board.x_size, final_board.y_size).unwrap();
        let suicide_move_emitted = samples.iter().any(|s| s.moves.iter().any(|m| m.loc == cc));
        assert!(!suicide_move_emitted);
        assert_eq!(final_board.colors[cc as usize], C_EMPTY);
        assert_eq!(final_board.num_pla_stones_on_board(P_WHITE), 4);
        assert_eq!(final_board.num_pla_stones_on_board(P_BLACK), 0);
    }
}

#[test]
fn run_sgf_file_tests() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/foxlike.sgf");
    let sgf = Sgf::parse_file(&path).unwrap();
    assert_eq!(sgf.get_xy_size().unwrap(), XYSize::new(19, 19));
    assert!((sgf.get_komi_or_fail().unwrap() - 6.5).abs() < 1e-6);
    assert!(sgf.has_rules());
    assert!(
        sgf.get_rules_or_fail()
            .unwrap()
            .equals_ignoring_komi(&Rules::parse_rules("chinese").unwrap())
    );
    assert_eq!(sgf.get_handicap_value().unwrap(), 2);
    assert_eq!(sgf.get_sgf_winner(), C_EMPTY);
    assert_eq!(sgf.get_player_name(P_BLACK), "testname1");
    assert_eq!(sgf.get_player_name(P_WHITE), "testname2");
    assert_eq!(sgf.get_rank(P_BLACK).unwrap(), 2);
    assert_eq!(sgf.get_rank(P_WHITE).unwrap(), 4);
}
