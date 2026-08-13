//! Ported from `KataGo/cpp/tests/testboardbasic.cpp`.
//!
//! Covers the `runBoardReplayTest` slice: graph-hash round-tripping and history
//! replay from the initial position.

use kata_core::rng::Rand;
use kata_game::board::{Board, NULL_LOC, P_BLACK, P_WHITE, PASS_LOC, Player, get_opp};
use kata_game::graph_hash::{get_graph_hash, get_graph_hash_from_scratch, get_state_hash};
use kata_game::history::BoardHistory;
use kata_game::rules::{KoRule, Rules, TaxRule};
use kata_program::play_utils::choose_random_legal_move;

#[test]
fn test_board_replay() {
    let mut rand = Rand::new_from_seed("runBoardReplayTest");

    let base = Board::parse_board(
        9,
        5,
        ".xo.o.ooo\nxo.oooooo\n.xooxxxo.\noxxxxx.xo\n.o.xx.xo.\n",
        '\n',
    )
    .unwrap();

    let draw_equivalent_wins_for_white = 0.7;

    for _rep in 0..6000 {
        let mut board = base.clone();
        let mut pla: Player = if rand.next_bool(0.5) {
            P_BLACK
        } else {
            P_WHITE
        };
        let mut initial_encore_phase = if rand.next_bool(0.9) {
            0
        } else if rand.next_bool(0.5) {
            1
        } else {
            2
        };
        let mut rules = if rand.next_bool(0.8) {
            Rules::parse_rules("japanese").unwrap()
        } else {
            Rules::parse_rules("chinese").unwrap()
        };
        if rules.scoring_rule == kata_game::rules::ScoringRule::Area {
            initial_encore_phase = 0;
        }
        if rand.next_bool(0.1) {
            rules.ko_rule = if rand.next_bool(0.5) {
                KoRule::Situational
            } else {
                KoRule::Positional
            };
        }
        if rand.next_bool(0.2) {
            rules.tax_rule = if rand.next_bool(0.5) {
                TaxRule::Seki
            } else if rand.next_bool(0.5) {
                TaxRule::None
            } else {
                TaxRule::All
            };
        }

        let mut hist = BoardHistory::new(board.clone(), pla, rules, initial_encore_phase);
        hist.set_initial_turn_number(rand.next_i32_range(0, 40) as i64);
        hist.set_assume_multiple_starting_black_moves_are_handicap(rand.next_bool(0.5));

        let rep_bound = rand.next_i32_range(1, 5);

        let mut graph_hash =
            get_graph_hash_from_scratch(&hist, pla, rep_bound, draw_equivalent_wins_for_white);

        let mut any_komi_set = false;
        let pass_prob = rand.next_double_range(0.05, 0.80);
        let num_steps = rand.next_i32_range(6, 15);
        for _i in 0..num_steps {
            let tmp_hist = BoardHistory::new(board.clone(), pla, rules, hist.encore_phase);
            let move_loc = if rand.next_bool(pass_prob) {
                PASS_LOC
            } else {
                let mut loc = choose_random_legal_move(&board, &tmp_hist, pla, &mut rand, NULL_LOC);
                // Avoid undefined-behavior-prone checks on a pass move. In C++
                // `isAdjacentToPla` is called unconditionally, but for PASS_LOC it
                // reads out-of-bounds board memory. The test intent is to bias
                // toward capture/adjacent moves, so we simply skip the bias when
                // the sampled move is a pass.
                if loc != PASS_LOC {
                    if !board.would_be_capture(loc, pla) {
                        loc = choose_random_legal_move(&board, &tmp_hist, pla, &mut rand, NULL_LOC);
                    }
                    if loc != PASS_LOC && !board.is_adjacent_to_pla(loc, get_opp(pla)) {
                        loc = choose_random_legal_move(&board, &tmp_hist, pla, &mut rand, NULL_LOC);
                    }
                }
                loc
            };
            let prevent_encore = rand.next_bool(0.5);
            if rand.next_bool(0.5) {
                let suc = hist.make_board_move_tolerant_with_prevent(
                    &mut board,
                    move_loc,
                    pla,
                    prevent_encore,
                );
                assert!(suc);
            } else {
                if hist.is_legal(&board, move_loc, pla) {
                    hist.make_board_move_assume_legal_with_prevent(
                        &mut board,
                        move_loc,
                        pla,
                        prevent_encore,
                    );
                } else {
                    continue;
                }
            }

            pla = get_opp(pla);
            if rand.next_bool(0.1) {
                pla = get_opp(pla);
            }
            if rand.next_bool(0.025) {
                any_komi_set = true;
                let delta = if rand.next_bool(0.5) { -0.5 } else { 0.5 };
                hist.set_komi(hist.rules.komi_f32() + delta);
            }

            graph_hash = get_graph_hash(
                graph_hash,
                &hist,
                pla,
                rep_bound,
                draw_equivalent_wins_for_white,
            );
        }

        let mut hist_copy = hist.copy_to_initial();
        let mut board_copy = hist_copy.get_recent_board(0).clone();
        for i in 0..hist.move_history.len() {
            hist_copy.make_board_move_assume_legal_with_prevent(
                &mut board_copy,
                hist.move_history[i].loc,
                hist.move_history[i].pla,
                hist.prevent_encore_history()[i],
            );
        }
        if rand.next_bool(0.05) {
            hist_copy = hist.clone();
        }

        assert!(board_copy.is_equal_for_testing(&board, true, true));
        assert!(board_copy.is_equal_for_testing(hist_copy.get_recent_board(0), true, true));
        assert!(hist_copy.get_recent_board(0).is_equal_for_testing(
            hist.get_recent_board(0),
            true,
            true
        ));
        assert_eq!(
            BoardHistory::get_situation_rules_and_ko_hash(
                &board_copy,
                &hist_copy,
                pla,
                draw_equivalent_wins_for_white
            ),
            BoardHistory::get_situation_rules_and_ko_hash(
                &board,
                &hist,
                pla,
                draw_equivalent_wins_for_white
            )
        );
        assert_eq!(
            hist_copy.current_self_komi(P_BLACK, draw_equivalent_wins_for_white),
            hist.current_self_komi(P_BLACK, draw_equivalent_wins_for_white)
        );
        assert_eq!(
            hist_copy.current_self_komi(P_WHITE, draw_equivalent_wins_for_white),
            hist.current_self_komi(P_WHITE, draw_equivalent_wins_for_white)
        );
        assert_eq!(hist_copy.initial_turn_number, hist.initial_turn_number);
        assert_eq!(
            hist_copy.presumed_next_move_pla,
            hist.presumed_next_move_pla
        );
        assert_eq!(
            hist_copy.assume_multiple_starting_black_moves_are_handicap,
            hist.assume_multiple_starting_black_moves_are_handicap
        );
        assert_eq!(
            get_state_hash(&hist_copy, pla, draw_equivalent_wins_for_white),
            get_state_hash(&hist, pla, draw_equivalent_wins_for_white)
        );
        assert_eq!(
            get_graph_hash_from_scratch(&hist_copy, pla, rep_bound, draw_equivalent_wins_for_white),
            get_graph_hash_from_scratch(&hist, pla, rep_bound, draw_equivalent_wins_for_white)
        );
        assert!(
            any_komi_set
                || graph_hash
                    == get_graph_hash_from_scratch(
                        &hist,
                        pla,
                        rep_bound,
                        draw_equivalent_wins_for_white
                    )
        );
    }
}
