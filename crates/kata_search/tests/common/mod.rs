//! Common helpers for `kata_search` integration tests ported from C++
//! `tests/testsearchcommon.cpp`.

use std::collections::HashSet;
use std::sync::Arc;

use kata_core::config::ConfigParser;
use kata_core::logger::{Logger, LoggerOptions};
use kata_data::sgf::CompactSgf;
use kata_game::board::{Board, Loc, P_WHITE, PASS_LOC, Player, get_opp};
use kata_game::history::BoardHistory;
use kata_game::rules::Rules;
use kata_nn::backend::Enabled;
use kata_nn::eval::NnEvaluator;
use kata_search::async_bot::AsyncBot;
use kata_search::node::SearchNode;
use kata_search::reported_values::ReportedSearchValues;
use kata_search::search::{PrintTreeOptions, Search};
use kata_search::time_control::TimeControls;

/// Options controlling what extra APIs `run_bot_on_position` exercises.
#[derive(Clone, Copy)]
pub struct TestSearchOptions {
    pub num_moves_in_a_row: i32,
    pub print_root_policy: bool,
    pub print_ownership: bool,
    pub print_ending_score_value_bonus: bool,
    pub print_play_selection_values: bool,
    pub print_root_values: bool,
    pub print_pruned_root_values: bool,
    pub print_sharp_score_and_error: bool,
    pub print_after_begun: bool,
    pub print_post_order_node_count: bool,
    pub print_more: bool,
    pub print_more_more_more: bool,
    pub root_hint_loc: Loc,
    pub no_clear_bot: bool,
    pub ignore_position: bool,
}

impl Default for TestSearchOptions {
    fn default() -> Self {
        Self {
            num_moves_in_a_row: 1,
            print_root_policy: false,
            print_ownership: false,
            print_ending_score_value_bonus: false,
            print_play_selection_values: false,
            print_root_values: false,
            print_pruned_root_values: false,
            print_sharp_score_and_error: false,
            print_after_begun: false,
            print_post_order_node_count: false,
            print_more: false,
            print_more_more_more: false,
            root_hint_loc: PASS_LOC,
            no_clear_bot: false,
            ignore_position: false,
        }
    }
}

/// Build a dummy NNEvaluator that skips real inference and a matching logger.
pub fn test_logger_and_eval(nn_x_len: i32, nn_y_len: i32) -> (Arc<Logger>, NnEvaluator) {
    test_logger_and_eval_with_options(nn_x_len, nn_y_len, false, 0)
}

/// Build a dummy NNEvaluator with control over exact-length requirement and default symmetry.
pub fn test_logger_and_eval_with_options(
    nn_x_len: i32,
    nn_y_len: i32,
    require_exact_nn_len: bool,
    default_symmetry: i32,
) -> (Arc<Logger>, NnEvaluator) {
    let logger = Arc::new(Logger::new(LoggerOptions::default(), None));
    let cfg = ConfigParser::new(false, false);
    let mut nn_eval = NnEvaluator::new(
        "test-model".to_string(),
        "/dev/null".to_string(),
        String::new(),
        logger.clone(),
        16,
        nn_x_len,
        nn_y_len,
        require_exact_nn_len,
        false,
        16,
        12,
        true,
        String::new(),
        Enabled::False,
        1,
        vec![0],
        "runSearchTestsRandSeed".to_string(),
        false,
        default_symmetry,
        true,
        &cfg,
    );
    nn_eval.spawn_server_threads();
    (logger, nn_eval)
}

#[allow(dead_code)]
pub fn count_reachable_nodes(search: &Search<'_>) -> usize {
    let root = search.root_node.as_deref().expect("root node should exist");
    fn visit(node: &SearchNode, visited: &mut HashSet<*const SearchNode>) {
        let ptr = node as *const SearchNode;
        if !visited.insert(ptr) {
            return;
        }
        let children = node.get_children();
        for i in 0..children.get_capacity() {
            if let Some(child) = children.get(i).get_if_allocated() {
                visit(child, visited);
            } else {
                break;
            }
        }
    }
    let mut visited = HashSet::new();
    visit(root, &mut visited);
    visited.len()
}

pub fn run_bot_on_position(
    bot: &mut AsyncBot<'_>,
    mut board: Board,
    mut next_pla: Player,
    mut hist: BoardHistory,
    opts: TestSearchOptions,
) {
    if opts.root_hint_loc != PASS_LOC {
        bot.set_root_hint_loc(opts.root_hint_loc);
    }
    if opts.print_ownership {
        bot.set_always_include_owner_map(true);
    }

    let tc = TimeControls::default();

    if !opts.ignore_position {
        bot.set_position(next_pla, &board, &hist);
    }

    for _ in 0..opts.num_moves_in_a_row {
        let _mv = if opts.print_after_begun {
            bot.gen_move_synchronous_with_factor_and_begun(
                next_pla,
                &tc,
                1.0,
                Some(Box::new(|| {})),
            )
        } else {
            bot.gen_move_synchronous(next_pla, &tc)
        };

        let search = bot.get_search();
        assert!(search.get_root_visits() > 0);

        if opts.print_more_more_more || opts.print_more {
            let options = if opts.print_more_more_more {
                PrintTreeOptions::new().max_depth(20)
            } else {
                PrintTreeOptions::new()
                    .min_visits_prop_to_expand(0.1)
                    .max_depth(2)
            };
            let root = search
                .root_node
                .as_deref()
                .expect("root node should exist for tree print");
            let mut out = String::new();
            search.print_tree(&mut out, Some(root), &options, P_WHITE);
            assert!(!out.is_empty());
        }

        if opts.print_root_policy {
            let mut _out = String::new();
            search.print_root_policy_map(&mut _out);
            assert!(!_out.is_empty());
        }
        if opts.print_ownership {
            let (_ownership, _stdev) = search.get_average_and_std_dev_tree_ownership(None);
            assert!(!_ownership.is_empty());
        }
        if opts.print_ending_score_value_bonus {
            let mut _out = String::new();
            search.print_root_ownership_map(&mut _out, P_WHITE);
            search.print_root_ending_score_value_bonus(&mut _out);
            assert!(!_out.is_empty());
        }
        if opts.print_play_selection_values {
            let mut locs = Vec::new();
            let mut values = Vec::new();
            assert!(search.get_play_selection_values(&mut locs, &mut values, 10.0));
        }
        if opts.print_root_values {
            let mut values = ReportedSearchValues::new();
            assert!(search.get_root_values(&mut values));
        }
        if opts.print_pruned_root_values {
            let mut values = ReportedSearchValues::new();
            assert!(search.get_pruned_root_values(&mut values));
        }
        if opts.print_sharp_score_and_error {
            let root = search.root_node.as_deref().unwrap();
            let mut sharp = 0.0;
            assert!(search.get_sharp_score(root, &mut sharp));
            let (_wl_err, _score_err) =
                search.get_shallow_average_shortterm_wl_and_score_error(Some(root));
        }
        if opts.print_post_order_node_count {
            assert!(count_reachable_nodes(search) > 1);
        }

        if opts.num_moves_in_a_row > 1 {
            let mv = bot.gen_move_synchronous(next_pla, &tc);
            assert!(bot.make_move(mv, next_pla));
            hist.make_board_move_assume_legal(&mut board, mv, next_pla);
            next_pla = get_opp(next_pla);
        }
    }

    if !opts.no_clear_bot {
        bot.clear_search();
    }
}

#[allow(dead_code)]
pub fn run_bot_on_sgf(
    bot: &mut AsyncBot<'_>,
    sgf_str: &str,
    default_rules: &Rules,
    turn_idx: i64,
    override_komi: f32,
    opts: TestSearchOptions,
) {
    let sgf = CompactSgf::parse(sgf_str).expect("valid SGF");
    let mut board = Board::default();
    let mut next_pla = 0;
    let mut hist = BoardHistory::default();
    let initial_rules = sgf
        .get_rules_or_fail_allow_unspecified(default_rules)
        .expect("rules");
    sgf.setup_board_and_hist_assume_legal(
        &initial_rules,
        &mut board,
        &mut next_pla,
        &mut hist,
        turn_idx,
    )
    .expect("setup board");
    hist.set_komi(override_komi);
    run_bot_on_position(bot, board, next_pla, hist, opts);
}
