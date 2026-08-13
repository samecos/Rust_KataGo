//! Ported from `KataGo/cpp/tests/testboardbasic.cpp`.
//!
//! Covers the `runBoardHandicapTest` slice: handicap-stone counting and the
//! white handicap bonus under Chinese and AGA rules.

mod common;

use kata_game::board::{Board, P_BLACK, P_WHITE, PASS_LOC, location::get_loc};
use kata_game::history::BoardHistory;
use kata_game::rules::Rules;

#[test]
fn test_board_handicap() {
    {
        let mut board = Board::new(19, 19);
        let x_size = board.x_size;
        let next_pla = P_BLACK;
        let rules = Rules::parse_rules("chinese").unwrap();
        let mut hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

        assert_eq!(hist.compute_num_handicap_stones(), 0);
        assert_eq!(hist.compute_white_handicap_bonus(), 0);
        hist.make_board_move_assume_legal(&mut board, get_loc(3, 3, x_size), P_BLACK);
        assert_eq!(hist.compute_num_handicap_stones(), 0);
        assert_eq!(hist.compute_white_handicap_bonus(), 0);
        hist.make_board_move_assume_legal(&mut board, get_loc(3, 4, x_size), P_BLACK);
        assert_eq!(hist.compute_num_handicap_stones(), 0);
        assert_eq!(hist.compute_white_handicap_bonus(), 0);
        hist.make_board_move_assume_legal(&mut board, get_loc(3, 5, x_size), P_BLACK);
        assert_eq!(hist.compute_num_handicap_stones(), 0);
        assert_eq!(hist.compute_white_handicap_bonus(), 0);
        hist.set_assume_multiple_starting_black_moves_are_handicap(true);
        assert_eq!(hist.compute_num_handicap_stones(), 3);
        assert_eq!(hist.compute_white_handicap_bonus(), 3);
    }

    {
        let mut board = Board::new(19, 19);
        let x_size = board.x_size;
        let next_pla = P_BLACK;
        let rules = Rules::parse_rules("chinese").unwrap();
        let mut hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

        hist.set_assume_multiple_starting_black_moves_are_handicap(true);
        assert_eq!(hist.compute_num_handicap_stones(), 0);
        hist.make_board_move_assume_legal(&mut board, get_loc(3, 3, x_size), P_BLACK);
        assert_eq!(hist.compute_num_handicap_stones(), 0);
        assert_eq!(hist.compute_white_handicap_bonus(), 0);
        hist.make_board_move_assume_legal(&mut board, get_loc(3, 4, x_size), P_BLACK);
        assert_eq!(hist.compute_num_handicap_stones(), 2);
        assert_eq!(hist.compute_white_handicap_bonus(), 2);
        hist.make_board_move_assume_legal(&mut board, get_loc(3, 5, x_size), P_BLACK);
        assert_eq!(hist.compute_num_handicap_stones(), 3);
        assert_eq!(hist.compute_white_handicap_bonus(), 3);
    }

    {
        let mut board = Board::new(19, 19);
        let x_size = board.x_size;
        let next_pla = P_BLACK;
        let rules = Rules::parse_rules("aga").unwrap();
        let mut hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

        hist.set_assume_multiple_starting_black_moves_are_handicap(true);
        assert_eq!(hist.compute_num_handicap_stones(), 0);
        hist.make_board_move_assume_legal(&mut board, get_loc(3, 3, x_size), P_BLACK);
        assert_eq!(hist.compute_num_handicap_stones(), 0);
        assert_eq!(hist.compute_white_handicap_bonus(), 0);
        hist.make_board_move_assume_legal(&mut board, get_loc(3, 4, x_size), P_BLACK);
        assert_eq!(hist.compute_num_handicap_stones(), 2);
        assert_eq!(hist.compute_white_handicap_bonus(), 1);
        hist.make_board_move_assume_legal(&mut board, get_loc(3, 5, x_size), P_BLACK);
        assert_eq!(hist.compute_num_handicap_stones(), 3);
        assert_eq!(hist.compute_white_handicap_bonus(), 2);
    }

    {
        let mut board = Board::new(19, 19);
        let x_size = board.x_size;
        let next_pla = P_BLACK;
        let rules = Rules::parse_rules("aga").unwrap();
        let mut hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

        hist.set_assume_multiple_starting_black_moves_are_handicap(true);
        assert_eq!(hist.compute_num_handicap_stones(), 0);
        hist.make_board_move_assume_legal(&mut board, get_loc(3, 3, x_size), P_BLACK);
        assert_eq!(hist.compute_num_handicap_stones(), 0);
        assert_eq!(hist.compute_white_handicap_bonus(), 0);
        hist.make_board_move_assume_legal(&mut board, PASS_LOC, P_WHITE);
        assert_eq!(hist.compute_num_handicap_stones(), 0);
        assert_eq!(hist.compute_white_handicap_bonus(), 0);
        hist.make_board_move_assume_legal(&mut board, get_loc(3, 4, x_size), P_BLACK);
        assert_eq!(hist.compute_num_handicap_stones(), 2);
        assert_eq!(hist.compute_white_handicap_bonus(), 1);
        hist.make_board_move_assume_legal(&mut board, PASS_LOC, P_WHITE);
        assert_eq!(hist.compute_num_handicap_stones(), 2);
        assert_eq!(hist.compute_white_handicap_bonus(), 1);
        hist.make_board_move_assume_legal(&mut board, get_loc(3, 5, x_size), P_BLACK);
        assert_eq!(hist.compute_num_handicap_stones(), 3);
        assert_eq!(hist.compute_white_handicap_bonus(), 2);
        hist.make_board_move_assume_legal(&mut board, get_loc(3, 6, x_size), P_WHITE);
        assert_eq!(hist.compute_num_handicap_stones(), 3);
        assert_eq!(hist.compute_white_handicap_bonus(), 2);
        hist.make_board_move_assume_legal(&mut board, get_loc(3, 7, x_size), P_BLACK);
        assert_eq!(hist.compute_num_handicap_stones(), 3);
        assert_eq!(hist.compute_white_handicap_bonus(), 2);
    }

    {
        let mut board = Board::new(19, 19);
        let x_size = board.x_size;
        let next_pla = P_BLACK;
        let rules = Rules::parse_rules("chinese").unwrap();
        let mut hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

        hist.set_assume_multiple_starting_black_moves_are_handicap(true);
        assert_eq!(hist.compute_num_handicap_stones(), 0);
        hist.make_board_move_assume_legal(&mut board, get_loc(3, 3, x_size), P_BLACK);
        assert_eq!(hist.compute_num_handicap_stones(), 0);
        assert_eq!(hist.compute_white_handicap_bonus(), 0);
        hist.make_board_move_assume_legal(&mut board, get_loc(3, 4, x_size), P_WHITE);
        assert_eq!(hist.compute_num_handicap_stones(), 0);
        assert_eq!(hist.compute_white_handicap_bonus(), 0);
        hist.make_board_move_assume_legal(&mut board, get_loc(3, 5, x_size), P_BLACK);
        assert_eq!(hist.compute_num_handicap_stones(), 0);
        assert_eq!(hist.compute_white_handicap_bonus(), 0);
        hist.make_board_move_assume_legal(&mut board, PASS_LOC, P_WHITE);
        assert_eq!(hist.compute_num_handicap_stones(), 0);
        assert_eq!(hist.compute_white_handicap_bonus(), 0);
        hist.make_board_move_assume_legal(&mut board, get_loc(3, 6, x_size), P_BLACK);
        assert_eq!(hist.compute_num_handicap_stones(), 0);
        assert_eq!(hist.compute_white_handicap_bonus(), 0);
        hist.make_board_move_assume_legal(&mut board, get_loc(3, 7, x_size), P_BLACK);
        assert_eq!(hist.compute_num_handicap_stones(), 0);
        assert_eq!(hist.compute_white_handicap_bonus(), 0);
    }
}
