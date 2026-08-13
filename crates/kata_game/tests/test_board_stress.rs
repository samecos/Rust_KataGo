//! Ported from `KataGo/cpp/tests/testboardbasic.cpp`.
//!
//! Covers the `runBoardStressTest` slice: randomized play-move legality checks,
//! chain regeneration, and `set_stone*` setup helpers.

mod common;

use kata_core::rng::Rand;
use kata_core::test::expect_lines_match;
use kata_game::board::{
    Board, C_EMPTY, Loc, MAX_ARR_SIZE, Move, NULL_LOC, P_BLACK, P_WHITE, PASS_LOC, Player, get_opp,
    location::get_loc,
};

use common::boards_seem_equal;
use std::collections::HashSet;

/// Shuffle a slice using the same RNG calls as the C++ `rand.shuffle`.
fn shuffle_with_rand<T>(rand: &mut Rand, slice: &mut [T]) {
    for i in 1..slice.len() {
        let r = rand.next_u32_bounded((i + 1) as u32) as usize;
        slice.swap(i, r);
    }
}

#[test]
fn test_board_stress() {
    {
        let mut rand = Rand::new_from_seed("runBoardStressTests");
        const NUM_BOARDS: usize = 4;
        let mut boards: Vec<Board> = vec![
            Board::new(19, 19),
            Board::new(9, 16),
            Board::new(13, 7),
            Board::new(4, 4),
        ];
        let multi_stone_suicide_legal: [bool; NUM_BOARDS] = [false, false, true, false];
        let mut pla = P_BLACK;
        let mut suicide_count = 0i32;
        let mut ko_ban_count = 0i32;
        let mut ko_capture_count = 0i32;
        let mut pass_count = 0i32;
        let mut regular_move_count = 0i32;

        for n in 0..20000 {
            let mut locs: [Loc; NUM_BOARDS] = [NULL_LOC; NUM_BOARDS];
            let mut empty_buf: [Loc; MAX_ARR_SIZE] = [NULL_LOC; MAX_ARR_SIZE];

            if n % 2 == 0 {
                locs[0] = rand.next_i32_range(-10, 500) as Loc;
                locs[1] = rand.next_i32_range(-10, 250) as Loc;
                locs[2] = rand.next_i32_range(-10, 200) as Loc;
                locs[3] = rand.next_i32_range(-10, 50) as Loc;
            } else {
                for i in 0..NUM_BOARDS {
                    let board = &boards[i];
                    let end = get_loc(board.x_size - 1, board.y_size - 1, board.x_size);
                    let mut empty_count: usize = 0;
                    for loc in 0..=end {
                        if board.colors[loc as usize] == C_EMPTY {
                            empty_buf[empty_count] = loc;
                            empty_count += 1;
                        }
                    }
                    assert!(empty_count > 0);
                    locs[i] = empty_buf[rand.next_u32_bounded(empty_count as u32) as usize];
                }
            }

            let mut copies: Vec<Board> = Vec::with_capacity(NUM_BOARDS);
            for board in boards.iter().take(NUM_BOARDS) {
                copies.push(board.clone());
            }

            let mut is_legal: [bool; NUM_BOARDS] = [false; NUM_BOARDS];
            let mut suc: [bool; NUM_BOARDS] = [false; NUM_BOARDS];
            for i in 0..NUM_BOARDS {
                is_legal[i] = boards[i].is_legal(locs[i], pla, multi_stone_suicide_legal[i]);
                assert!(boards_seem_equal(&copies[i], &boards[i]));
                suc[i] = boards[i].play_move(locs[i], pla, multi_stone_suicide_legal[i]);
            }

            for i in 0..NUM_BOARDS {
                assert_eq!(is_legal[i], suc[i]);
                // checkConsistency is not ported in this slice.

                if n % 15 == 0 {
                    let mut regen = boards[i].clone();
                    regen.regen_chains_from_colors();
                    for loc in 0..MAX_ARR_SIZE as Loc {
                        assert_eq!(regen.colors[loc as usize], boards[i].colors[loc as usize]);
                    }
                    // regen.checkConsistency();
                }

                let board = &boards[i];
                let copy = &copies[i];
                let loc = locs[i];
                let mss = multi_stone_suicide_legal[i];

                if !suc[i] {
                    if board.is_on_board(loc) {
                        assert!(boards_seem_equal(copy, board));
                        assert!(
                            loc < 0
                                || loc >= MAX_ARR_SIZE as Loc
                                || board.colors[loc as usize] != C_EMPTY
                                || board.is_illegal_suicide(loc, pla, mss)
                                || board.is_ko_banned(loc)
                        );
                        if board.is_ko_banned(loc) {
                            assert_eq!(board.colors[loc as usize], C_EMPTY);
                            assert!(
                                board.would_be_ko_capture(loc, P_BLACK)
                                    || board.would_be_ko_capture(loc, P_WHITE)
                            );
                            assert!(
                                board.would_be_capture(loc, P_BLACK)
                                    || board.would_be_capture(loc, P_WHITE)
                            );
                            if board.is_adjacent_to_pla(loc, get_opp(pla)) {
                                assert!(board.would_be_ko_capture(loc, pla));
                                assert!(board.would_be_capture(loc, pla));
                            }
                            ko_ban_count += 1;
                        }
                    }
                } else {
                    if loc == PASS_LOC {
                        assert!(boards_seem_equal(copy, board));
                        assert_eq!(board.ko_loc, NULL_LOC);
                        pass_count += 1;
                    } else if copy.is_suicide(loc, pla) {
                        assert_eq!(board.colors[loc as usize], C_EMPTY);
                        assert!(board.is_legal(loc, pla, mss));
                        assert!(mss);
                        assert!(!copy.would_be_capture(loc, pla));
                        suicide_count += 1;
                    } else {
                        assert_eq!(board.colors[loc as usize], pla);
                        let libs = board.get_num_liberties(loc);
                        assert_eq!(libs, copy.get_num_liberties_after_play(loc, pla, 1000));
                        assert_eq!(libs.min(2), copy.get_num_liberties_after_play(loc, pla, 2));
                        assert_eq!(libs.min(4), copy.get_num_liberties_after_play(loc, pla, 4));
                        if board.ko_loc != NULL_LOC {
                            ko_capture_count += 1;
                            assert!(copy.would_be_ko_capture(loc, pla));
                            assert!(copy.would_be_capture(loc, pla));
                        } else {
                            assert!(!copy.would_be_ko_capture(loc, pla));
                        }
                        if !board.is_adjacent_to_pla(loc, get_opp(pla))
                            && copy.is_adjacent_to_pla(loc, get_opp(pla))
                        {
                            assert!(copy.would_be_capture(loc, pla));
                        }
                        if !copy.is_adjacent_to_pla(loc, get_opp(pla)) {
                            assert!(!copy.would_be_capture(loc, pla));
                        }

                        regular_move_count += 1;
                    }
                }
            }

            pla = if rand.next_u32_bounded(2) == 0 {
                get_opp(pla)
            } else {
                pla
            };
        }

        let mut out = String::new();
        out.push('\n');
        out.push_str(&format!("regularMoveCount {}\n", regular_move_count));
        out.push_str(&format!("passCount {}\n", pass_count));
        out.push_str(&format!("koCaptureCount {}\n", ko_capture_count));
        out.push_str(&format!("koBanCount {}\n", ko_ban_count));
        out.push_str(&format!("suicideCount {}\n", suicide_count));
        for board in boards.iter().take(NUM_BOARDS) {
            out.push_str(&format!(
                "Caps {} {}\n",
                board.num_black_captures, board.num_white_captures
            ));
        }

        // Expected counts match the C++ reference after fixing Rand seeding to
        // read SHA-256 digest bytes as big-endian.
        let expected = r#"
regularMoveCount 38017
passCount 273
koCaptureCount 212
koBanCount 45
suicideCount 440
Caps 4753 5024
Caps 4821 4733
Caps 4995 5041
Caps 4420 4335
"#;
        expect_lines_match("Board stress test move counts", &out, expected);
    }

    {
        let mut rand = Rand::new_from_seed("runBoardSetStoneTests");

        let board_to_placements = |b: &Board, rand: &mut Rand| {
            let mut placements = Vec::new();
            for y in 0..b.y_size {
                for x in 0..b.x_size {
                    let loc = get_loc(x, y, b.x_size);
                    placements.push(Move::new(loc, b.colors[loc as usize]));
                }
            }
            shuffle_with_rand(rand, &mut placements);
            placements
        };
        let board_to_non_empty_placements = |b: &Board, rand: &mut Rand| {
            let mut placements = Vec::new();
            for y in 0..b.y_size {
                for x in 0..b.x_size {
                    let loc = get_loc(x, y, b.x_size);
                    if b.colors[loc as usize] != C_EMPTY {
                        placements.push(Move::new(loc, b.colors[loc as usize]));
                    }
                }
            }
            shuffle_with_rand(rand, &mut placements);
            placements
        };

        for _rep in 0..1000 {
            let mut board0 = Board::new(
                5 + rand.next_u32_bounded(14) as i32,
                5 + rand.next_u32_bounded(14) as i32,
            );
            let mut board1 = board0.clone();
            let mut board2 = board0.clone();
            let mut board3 = board0.clone();
            let mut board4 = board0.clone();
            let mut board5 = board0.clone();
            let mut board6 = board0.clone();
            let num_moves1 = rand.next_u32_bounded(1000) as i32;
            let num_moves2 = rand.next_u32_bounded(1000) as i32;

            for _i in 0..num_moves1 {
                let loc = get_loc(
                    rand.next_u32_bounded(board1.x_size as u32) as i32,
                    rand.next_u32_bounded(board1.y_size as u32) as i32,
                    board1.x_size,
                );
                let pla: Player = if rand.next_bool(0.5) {
                    P_BLACK
                } else {
                    P_WHITE
                };
                if board1.is_legal(loc, pla, true) {
                    let suc4 = board4.set_stone_fail_if_no_libs(loc, pla);
                    assert_eq!(
                        suc4,
                        !(board1.would_be_capture(loc, pla) || board1.is_suicide(loc, pla))
                    );
                    if !suc4 {
                        board4.play_move_assume_legal(loc, pla);
                    } else {
                        assert_eq!(board4.colors[loc as usize], pla);
                    }

                    board1.play_move_assume_legal(loc, pla);
                    let suc3 = board3.set_stone(loc, pla);
                    assert!(suc3);
                } else {
                    let old_color = board4.colors[loc as usize];
                    let suc4 = board4.set_stone_fail_if_no_libs(loc, pla);
                    if suc4 {
                        assert_eq!(board4.colors[loc as usize], pla);
                        let suc4_2 = board4.set_stone_fail_if_no_libs(loc, old_color);
                        assert!(suc4_2);
                    }
                }
            }
            // board1.checkConsistency();
            // board3.checkConsistency();
            // board4.checkConsistency();
            assert_eq!(board1.pos_hash, board3.pos_hash);
            assert_eq!(board1.pos_hash, board4.pos_hash);

            let empty_prob = rand.next_double() * 0.5;
            for _i in 0..num_moves2 {
                let loc = get_loc(
                    rand.next_u32_bounded(board2.x_size as u32) as i32,
                    rand.next_u32_bounded(board2.y_size as u32) as i32,
                    board2.x_size,
                );
                let color = if rand.next_bool(empty_prob) {
                    C_EMPTY
                } else if rand.next_bool(0.5) {
                    P_BLACK
                } else {
                    P_WHITE
                };
                let suc2 = board2.set_stone(loc, color);
                assert!(suc2);
                let suc5 = board5.set_stone_fail_if_no_libs(loc, color);
                let suc6 = board6.set_stone_fail_if_no_libs(loc, color);
                if color == C_EMPTY {
                    assert!(suc5);
                    assert!(suc6);
                }
                if !suc5 {
                    board5.set_stone(loc, color);
                } else {
                    assert_eq!(board5.colors[loc as usize], color);
                }
                if suc6 {
                    assert_eq!(board6.colors[loc as usize], color);
                } else {
                    assert_ne!(board6.colors[loc as usize], color);
                }
                // board2.checkConsistency();
                // board5.checkConsistency();
                // board6.checkConsistency();
                assert_eq!(board2.pos_hash, board5.pos_hash);
            }

            let placements = board_to_non_empty_placements(&board1, &mut rand);
            let suc0 = board0.set_stones_fail_if_no_libs(&placements);
            assert!(suc0);
            // board0.checkConsistency();
            assert_eq!(board0.pos_hash, board1.pos_hash);

            let placements = board_to_placements(&board2, &mut rand);
            let suc3 = board3.set_stones_fail_if_no_libs(&placements);
            assert!(suc3);
            // board3.checkConsistency();
            assert_eq!(board3.pos_hash, board2.pos_hash);
        }
    }

    {
        let mut rand = Rand::new_from_seed("runBoardSetStoneTests2");
        for _rep in 0..1000 {
            let mut board = Board::new(
                1 + rand.next_u32_bounded(18) as i32,
                1 + rand.next_u32_bounded(18) as i32,
            );
            let mut placements: Vec<kata_game::board::Move> = Vec::new();
            for _i in 0..1000 {
                let loc = get_loc(
                    rand.next_u32_bounded(board.x_size as u32) as i32,
                    rand.next_u32_bounded(board.y_size as u32) as i32,
                    board.x_size,
                );
                let pla = if rand.next_bool(0.5) {
                    P_BLACK
                } else {
                    P_WHITE
                };
                if board.is_legal(loc, pla, true) {
                    placements.push(Move::new(loc, pla));
                    let any_caps = board.would_be_capture(loc, pla) || board.is_suicide(loc, pla);
                    board.play_move_assume_legal(loc, pla);
                    let mut copy = Board::new(board.x_size, board.y_size);
                    let suc = copy.set_stones_fail_if_no_libs(&placements);
                    assert_eq!(suc, !any_caps);
                    // copy.checkConsistency();
                    if !suc {
                        break;
                    }
                    assert_eq!(board.pos_hash, copy.pos_hash);
                }
            }
        }
    }

    {
        let mut rand = Rand::new_from_seed("runBoardSetStoneTests3");
        for _rep in 0..1000 {
            let mut board = Board::new(
                1 + rand.next_u32_bounded(18) as i32,
                1 + rand.next_u32_bounded(18) as i32,
            );
            for _i in 0..300 {
                let loc = get_loc(
                    rand.next_u32_bounded(board.x_size as u32) as i32,
                    rand.next_u32_bounded(board.y_size as u32) as i32,
                    board.x_size,
                );
                let color = if rand.next_bool(0.25) {
                    C_EMPTY
                } else if rand.next_bool(0.5) {
                    P_BLACK
                } else {
                    P_WHITE
                };
                board.set_stone(loc, color);
            }

            let orig = board.clone();
            let mut prev_placed_locs: HashSet<Loc> = HashSet::new();
            let mut placements: Vec<kata_game::board::Move> = Vec::new();
            for _i in 0..1000 {
                let loc = get_loc(
                    rand.next_u32_bounded(board.x_size as u32) as i32,
                    rand.next_u32_bounded(board.y_size as u32) as i32,
                    board.x_size,
                );
                let color = if rand.next_bool(0.25) {
                    C_EMPTY
                } else if rand.next_bool(0.5) {
                    P_BLACK
                } else {
                    P_WHITE
                };

                placements.push(Move::new(loc, color));
                if prev_placed_locs.contains(&loc) {
                    let mut copy = Board::new(board.x_size, board.y_size);
                    let suc = copy.set_stones_fail_if_no_libs(&placements);
                    assert!(!suc);
                    placements.pop();
                } else {
                    prev_placed_locs.insert(loc);
                    let prev = board.clone();
                    board.set_stone(loc, color);

                    let mut any_caps = false;
                    for y in 0..board.y_size {
                        for x in 0..board.x_size {
                            let l = get_loc(x, y, board.x_size);
                            if l != loc
                                && board.colors[l as usize] == C_EMPTY
                                && prev.colors[l as usize] != C_EMPTY
                            {
                                any_caps = true;
                            }
                        }
                    }

                    let mut copy = orig.clone();
                    let suc = copy.set_stones_fail_if_no_libs(&placements);
                    assert_eq!(suc, !any_caps && board.colors[loc as usize] == color);
                    if !suc {
                        break;
                    }
                }
            }
        }
    }
}
