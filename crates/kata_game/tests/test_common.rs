//! Smoke tests for the common test helpers ported from `cpp/tests/testcommon.cpp`.

mod common;

use common::{boards_seem_equal, get_benchmark_sgf_data, get_multi_game_size9_data};
use kata_game::board::location::get_loc;
use kata_game::board::{Board, P_BLACK};

#[test]
fn test_boards_seem_equal() {
    let b1 = Board::new(9, 9);
    let b2 = Board::new(9, 9);
    assert!(boards_seem_equal(&b1, &b2));

    let mut b3 = Board::new(9, 9);
    let center = get_loc(4, 4, 9);
    b3.play_move(center, P_BLACK, true);
    assert!(!boards_seem_equal(&b1, &b3));

    // Two boards with the same move should be equal again.
    let mut b4 = Board::new(9, 9);
    b4.play_move(center, P_BLACK, true);
    assert!(boards_seem_equal(&b3, &b4));

    // A different move should differ.
    let mut b5 = Board::new(9, 9);
    let corner = get_loc(0, 0, 9);
    b5.play_move(corner, P_BLACK, true);
    assert!(!boards_seem_equal(&b3, &b5));
}

#[test]
fn test_get_benchmark_sgf_data() {
    for size in 7..=19 {
        let sgf = get_benchmark_sgf_data(size);
        assert!(!sgf.is_empty());
        assert!(sgf.contains(&format!("SZ[{}]", size)));
    }
}

#[test]
fn test_get_multi_game_size9_data() {
    let games = get_multi_game_size9_data();
    assert_eq!(games.len(), 30);
    for (i, sgf) in games.iter().enumerate() {
        assert!(!sgf.is_empty(), "game {} is empty", i);
        assert!(sgf.contains("SZ[9]"), "game {} missing SZ[9]", i);
    }
}
