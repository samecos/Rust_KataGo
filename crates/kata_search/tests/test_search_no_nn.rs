//! Integration tests for kata_search using a dummy neural-net evaluator.
//!
//! These are Rust ports of test blocks from
//! `KataGo/cpp/tests/testsearchnonn.cpp`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use kata_core::config::ConfigParser;
use kata_core::logger::{Logger, LoggerOptions};
use kata_core::rng::Rand;
use kata_game::board::{
    Board, C_EMPTY, Loc, MAX_ARR_SIZE, P_BLACK, P_WHITE, PASS_LOC, get_opp, location,
};
use kata_game::history::BoardHistory;
use kata_game::rules::Rules;
use kata_nn::backend::Enabled;
use kata_nn::eval::NnEvaluator;
use kata_nn::inputs::nn_pos;
use kata_search::node::SearchNode;
use kata_search::params::SearchParams;
use kata_search::reported_values::ReportedSearchValues;
use kata_search::search::Search;

/// Build a dummy NNEvaluator that skips real inference and a matching logger.
fn test_logger_and_eval() -> (Arc<Logger>, NnEvaluator) {
    let logger = Arc::new(Logger::new(LoggerOptions::default(), None));
    let cfg = ConfigParser::new(false, false);
    let nn_eval = NnEvaluator::new(
        "test-model".to_string(),
        "/dev/null".to_string(),
        String::new(),
        logger.clone(),
        1,              // max_batch_size
        19,             // nn_x_len
        19,             // nn_y_len
        false,          // require_exact_nn_len
        false,          // inputs_use_nhwc
        -1,             // nn_cache_size_power_of_two (no cache)
        0,              // nn_mutex_pool_size_power_of_two
        true,           // debug_skip_neural_net
        String::new(),  // home_data_dir_override
        Enabled::False, // using_fp16_mode
        0,              // num_threads
        Vec::new(),     // gpu_idx_by_server_thread
        "test-seed".to_string(),
        false, // do_randomize
        0,     // default_symmetry
        true,  // disable_warmup
        &cfg,
    );
    (logger, nn_eval)
}

/// Sample `search.get_chosen_move_loc()` `n` times and count each location.
fn sample_chosen_moves(search: &mut Search, n: usize) -> HashMap<Loc, usize> {
    let mut counts = HashMap::new();
    for _ in 0..n {
        let loc = search.get_chosen_move_loc();
        *counts.entry(loc).or_insert(0) += 1;
    }
    counts
}

/// Enumerate the search tree in postorder and verify every node is hit exactly
/// once, with all children appearing before their parent.
fn verify_tree_post_order(search: &Search<'_>) -> usize {
    let root = search
        .root_node
        .as_deref()
        .expect("root node should exist after search");

    fn visit(
        node: &SearchNode,
        order: &mut Vec<*const SearchNode>,
        visited: &mut HashSet<*const SearchNode>,
    ) {
        let ptr = node as *const SearchNode;
        if !visited.insert(ptr) {
            return;
        }
        let children = node.get_children();
        let capacity = children.get_capacity();
        for i in 0..capacity {
            if let Some(child) = children.get(i).get_if_allocated() {
                visit(child, order, visited);
            } else {
                break;
            }
        }
        order.push(ptr);
    }

    let mut order = Vec::new();
    let mut visited = HashSet::new();
    visit(root, &mut order, &mut visited);

    let idx_of_node: HashMap<_, _> = order.iter().enumerate().map(|(i, &p)| (p, i)).collect();
    for (i, &node_ptr) in order.iter().enumerate() {
        let node = unsafe { &*node_ptr };
        let children = node.get_children();
        let capacity = children.get_capacity();
        for j in 0..capacity {
            if let Some(child) = children.get(j).get_if_allocated() {
                let child_ptr = child as *const SearchNode;
                let child_idx = idx_of_node
                    .get(&child_ptr)
                    .expect("child should appear in postorder enumeration");
                assert!(
                    *child_idx < i,
                    "child must appear before parent in postorder"
                );
            } else {
                break;
            }
        }
    }

    order.len()
}

/// Count the number of unique nodes reachable from the root, tolerating DAGs
/// that can arise from node-table deduplication.
fn count_reachable_nodes(search: &Search<'_>) -> usize {
    let root = search
        .root_node
        .as_deref()
        .expect("root node should exist after search");

    fn visit(node: &SearchNode, visited: &mut HashSet<*const SearchNode>) {
        let ptr = node as *const SearchNode;
        if !visited.insert(ptr) {
            return;
        }
        let children = node.get_children();
        let capacity = children.get_capacity();
        for i in 0..capacity {
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

/// Build an avoid-move-until vector from a board whose black stones only mark
/// locations to clear (avoid value 0). Returns the board with those stones
/// removed and the avoid vector for both colors.
fn board_with_avoid_moves_from_black_marks(mut board: Board) -> (Board, Vec<i32>) {
    let mut avoid = vec![0; MAX_ARR_SIZE];
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = location::get_loc(x, y, board.x_size);
            if board.colors[loc as usize] == P_BLACK {
                avoid[loc as usize] = 0;
                board.set_stone(loc, C_EMPTY);
            } else {
                avoid[loc as usize] = if y % 2 == 0 { 1 } else { 2 };
            }
        }
    }
    (board, avoid)
}

#[test]
fn basic_search_and_chosen_move_randomization() {
    let (logger, nn_eval) = test_logger_and_eval();

    let params = SearchParams {
        max_visits: 100,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");

    let board = Board::parse_board(
        9,
        9,
        ".........\n\
         .........\n\
         ..x..o...\n\
         .........\n\
         ..x...o..\n\
         ...o.....\n\
         ..o.x.x..\n\
         .........\n\
         .........",
        '\n',
    )
    .expect("parse 9x9 board");
    let next_pla = P_BLACK;
    let hist = BoardHistory::new(board.clone(), next_pla, Rules::get_tromp_taylorish(), 0);

    search.set_position(next_pla, &board, &hist);
    search.run_whole_search(next_pla);

    // Temperature 0 (default): deterministic best move.
    let counts = sample_chosen_moves(&mut search, 10000);
    assert_eq!(
        counts.len(),
        1,
        "chosen move at temperature 0 should be deterministic, got {:?}",
        counts
    );
    assert_eq!(
        counts.values().next().copied().unwrap_or(0),
        10000,
        "all samples should pick the same move"
    );

    // Temperature 1 with early temperature 1: stochastic move selection.
    search.search_params.chosen_move_temperature = 1.0;
    search.search_params.chosen_move_temperature_early = 1.0;
    let counts = sample_chosen_moves(&mut search, 10000);
    assert!(
        counts.len() > 1,
        "chosen move at temperature 1 should be non-deterministic, got {:?}",
        counts
    );
}

#[test]
fn preservation_of_search_tree_across_moves() {
    let (logger, nn_eval) = test_logger_and_eval();

    let params = SearchParams {
        max_visits: 100,
        cpuct_exploration: 2.0,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");

    let board = Board::parse_board(
        7,
        7,
        "..xx...\n\
         xxxxxxx\n\
         .xx..xx\n\
         .xxoooo\n\
         xxxo...\n\
         ooooooo\n\
         ...o...",
        '\n',
    )
    .expect("parse 7x7 board");
    let next_pla = P_BLACK;
    let hist = BoardHistory::new(board.clone(), next_pla, Rules::get_tromp_taylorish(), 0);

    search.set_position(next_pla, &board, &hist);
    search.run_whole_search(next_pla);

    let root = search
        .root_node
        .as_deref()
        .expect("root node should exist after search");
    let children = root.get_children();
    let children_capacity = children.get_capacity();
    assert!(
        children_capacity > 1,
        "root children capacity should be greater than 1"
    );
    let allocated_children = children.iterate_and_count_children();
    assert!(
        allocated_children > 1,
        "root should have more than one allocated child"
    );
    assert!(
        children.get(1).get_if_allocated().is_some(),
        "second root child should be allocated"
    );

    let loc_to_descend = children.get(1).get_move_loc();
    let move_pla = next_pla;
    assert!(
        search.make_move(loc_to_descend, move_pla),
        "make_move should succeed"
    );
    let next_pla = get_opp(move_pla);

    // Verify the root board was actually updated with the move.
    if loc_to_descend != PASS_LOC {
        assert_eq!(
            search.root_board.colors[loc_to_descend as usize], move_pla,
            "stone should appear at the descended location"
        );
    }
    assert_eq!(
        search.get_root_pla(),
        next_pla,
        "root player should switch after a move"
    );

    // Continue the search to complete the visit budget; must not panic.
    search.run_whole_search(next_pla);
}

/// True if any allocated root child is a suicide move for the current root player.
fn has_suicide_root_moves(search: &Search<'_>) -> bool {
    let root = search
        .root_node
        .as_deref()
        .expect("root node should exist after search");
    let children = root.get_children();
    let capacity = children.get_capacity();
    for i in 0..capacity {
        if children.get(i).get_if_allocated().is_none() {
            break;
        }
        let loc = children.get(i).get_move_loc();
        if search.root_board.is_suicide(loc, search.root_pla) {
            return true;
        }
    }
    false
}

/// True if any allocated root child lands in a pass-alive (safe) area.
fn has_pass_alive_root_moves(search: &Search<'_>) -> bool {
    let root = search
        .root_node
        .as_deref()
        .expect("root node should exist after search");
    let children = root.get_children();
    let capacity = children.get_capacity();
    let safe_area = search.root_safe_area.as_deref();
    for i in 0..capacity {
        if children.get(i).get_if_allocated().is_none() {
            break;
        }
        let loc = children.get(i).get_move_loc();
        let area_color = safe_area.map_or(C_EMPTY, |area| area[loc as usize]);
        if area_color != C_EMPTY {
            return true;
        }
    }
    false
}

#[test]
fn pruning_of_search_tree_due_to_root_restrictions() {
    let (logger, nn_eval) = test_logger_and_eval();

    let mut board = Board::parse_board(
        7,
        7,
        "..xx...\n\
         xx.xxxx\n\
         x.xx.xx\n\
         .xxoooo\n\
         xxxo..x\n\
         ooooooo\n\
         o..oo.x",
        '\n',
    )
    .expect("parse 7x7 board");
    let mut next_pla = P_BLACK;
    let rules = Rules::get_tromp_taylorish();
    let mut hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    for m in ["B5", "pass", "C6", "pass", "G7", "pass", "F3", "pass"] {
        let loc = location::of_string(m, board.x_size, board.y_size).expect("valid move");
        hist.make_board_move_assume_legal(&mut board, loc, next_pla);
        next_pla = get_opp(next_pla);
    }

    // Without root-pruning, suicide moves should still appear among the root children.
    {
        let params = SearchParams {
            max_visits: 400,
            value_weight_exponent: 0.0,
            ..SearchParams::default()
        };
        let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed3");
        search.set_position(next_pla, &board, &hist);
        search.run_whole_search(next_pla);

        assert!(
            has_suicide_root_moves(&search),
            "suicide moves should be present without root pruning"
        );
    }

    // With root-pruning enabled, suicide moves should be pruned from the root.
    {
        let params = SearchParams {
            max_visits: 400,
            root_prune_useless_moves: true,
            value_weight_exponent: 0.0,
            ..SearchParams::default()
        };
        let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed3");
        search.set_position(next_pla, &board, &hist);
        search.run_whole_search(next_pla);

        assert!(
            !has_suicide_root_moves(&search),
            "suicide moves should be pruned with root_prune_useless_moves"
        );
    }

    // Progress the game: black fills space while white passes.
    for m in ["A7", "pass", "E7", "pass", "F7"] {
        let loc = location::of_string(m, board.x_size, board.y_size).expect("valid move");
        hist.make_board_move_assume_legal(&mut board, loc, next_pla);
        next_pla = get_opp(next_pla);
    }

    {
        let params = SearchParams {
            max_visits: 400,
            root_prune_useless_moves: true,
            value_weight_exponent: 0.0,
            ..SearchParams::default()
        };
        let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed3");
        search.set_position(next_pla, &board, &hist);
        search.run_whole_search(next_pla);

        // Play the pass forward. The reused tree still contains now-useless moves.
        assert!(
            search.make_move(PASS_LOC, next_pla),
            "make_move(pass) should succeed"
        );
        assert!(
            has_suicide_root_moves(&search),
            "reused tree should still contain suicide moves immediately after make_move"
        );
        assert!(
            has_pass_alive_root_moves(&search),
            "reused tree should still contain pass-alive moves immediately after make_move"
        );

        // Beginning a search recomputes root restrictions and prunes them.
        search.begin_search(false);
        assert!(
            !has_suicide_root_moves(&search),
            "suicide moves should be gone after begin_search"
        );
        assert!(
            !has_pass_alive_root_moves(&search),
            "pass-alive moves should be gone after begin_search"
        );

        // Continue searching; must not panic.
        search.run_whole_search(get_opp(next_pla));
    }
}

#[test]
fn search_tree_update_near_terminal_positions() {
    let (logger, nn_eval) = test_logger_and_eval();

    let board = Board::parse_board(
        7,
        7,
        "x.xx.xx\n\
         xxx.xxx\n\
         xxxxxxx\n\
         xxxxxxx\n\
         ooooooo\n\
         ooooooo\n\
         o..o.oo",
        '\n',
    )
    .expect("parse 7x7 board");
    let next_pla = P_WHITE;
    let mut rules = Rules::get_tromp_taylorish();
    rules.multi_stone_suicide_legal = false;
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let params = SearchParams {
        max_visits: 400,
        dynamic_score_utility_factor: 0.5,
        use_lcb_for_selection: true,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed3");

    search.set_position(next_pla, &board, &hist);
    search.run_whole_search(next_pla);

    // begin_search should be idempotent and not panic.
    search.begin_search(false);

    let b1 = location::of_string("B1", board.x_size, board.y_size).expect("valid coordinate");
    assert!(
        search.make_move(b1, next_pla),
        "make_move(B1) should succeed"
    );

    // Another begin_search after the move should also be stable.
    search.begin_search(false);
}

#[test]
fn pruning_of_search_tree_at_root_due_to_symmetries_empty_board() {
    let (logger, nn_eval) = test_logger_and_eval();

    let params = SearchParams {
        max_visits: 5000,
        cpuct_exploration: 4.0,
        root_symmetry_pruning: true,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");

    let board = Board::parse_board(
        9,
        9,
        ".........\n\
         .........\n\
         .........\n\
         .........\n\
         .........\n\
         .........\n\
         .........\n\
         .........\n\
         .........",
        '\n',
    )
    .expect("parse 9x9 board");
    let next_pla = P_BLACK;
    let hist = BoardHistory::new(board.clone(), next_pla, Rules::get_tromp_taylorish(), 0);

    search.set_position(next_pla, &board, &hist);
    search.run_whole_search(next_pla);

    assert!(
        verify_tree_post_order(&search) > 1,
        "search should produce a nontrivial tree"
    );

    let loc_to_descend =
        location::of_string("H8", board.x_size, board.y_size).expect("valid coordinate");
    assert!(
        search.make_move(loc_to_descend, next_pla),
        "make_move(H8) should succeed"
    );
    let next_pla = get_opp(next_pla);

    assert!(
        verify_tree_post_order(&search) > 1,
        "tree should remain valid after make_move"
    );

    search.begin_search(false);
    assert!(
        verify_tree_post_order(&search) > 1,
        "tree should remain valid after begin_search"
    );

    search.run_whole_search(next_pla);
}

#[test]
fn pruning_of_search_tree_at_root_due_to_symmetries_asymmetric_board() {
    let (logger, nn_eval) = test_logger_and_eval();

    let params = SearchParams {
        max_visits: 2000,
        root_symmetry_pruning: true,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");

    let board = Board::parse_board(
        9,
        9,
        ".........\n\
         .........\n\
         .........\n\
         .........\n\
         .......O.\n\
         .........\n\
         .........\n\
         ...X.....\n\
         .........",
        '\n',
    )
    .expect("parse 9x9 board");
    let next_pla = P_BLACK;
    let hist = BoardHistory::new(board.clone(), next_pla, Rules::get_tromp_taylorish(), 0);

    search.set_position(next_pla, &board, &hist);
    search.run_whole_search(next_pla);

    assert!(
        verify_tree_post_order(&search) > 1,
        "search should produce a nontrivial tree"
    );
}

#[test]
fn pruning_of_search_tree_at_root_due_to_symmetries_diagonal_flips_empty_board() {
    let (logger, nn_eval) = test_logger_and_eval();

    let params = SearchParams {
        max_visits: 5000,
        root_symmetry_pruning: true,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");

    let board = Board::parse_board(
        9,
        9,
        ".........\n\
         .........\n\
         .........\n\
         .........\n\
         .........\n\
         .........\n\
         .........\n\
         .........\n\
         .........",
        '\n',
    )
    .expect("parse 9x9 board");
    let next_pla = P_BLACK;
    let hist = BoardHistory::new(board.clone(), next_pla, Rules::get_tromp_taylorish(), 0);

    search.set_position(next_pla, &board, &hist);
    search.set_root_symmetry_pruning_only(&[0, 3, 4, 7]);
    search.run_whole_search(next_pla);

    assert!(
        verify_tree_post_order(&search) > 1,
        "search should produce a nontrivial tree"
    );
}

#[test]
fn pruning_of_search_tree_at_root_due_to_symmetries_diagonal_flips_one_diagonal() {
    let (logger, nn_eval) = test_logger_and_eval();

    let params = SearchParams {
        max_visits: 5000,
        root_symmetry_pruning: true,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");

    let board = Board::parse_board(
        9,
        9,
        ".........\n\
         .........\n\
         .........\n\
         .........\n\
         ....o....\n\
         .........\n\
         ..x......\n\
         .........\n\
         .........",
        '\n',
    )
    .expect("parse 9x9 board");
    let next_pla = P_BLACK;
    let hist = BoardHistory::new(board.clone(), next_pla, Rules::get_tromp_taylorish(), 0);

    search.set_position(next_pla, &board, &hist);
    search.set_root_symmetry_pruning_only(&[0, 3, 4, 7]);
    search.run_whole_search(next_pla);

    assert!(
        verify_tree_post_order(&search) > 1,
        "search should produce a nontrivial tree"
    );
}

#[test]
fn pruning_of_search_tree_at_root_due_to_symmetries_diagonal_flips_empty_board_with_avoid_moves() {
    let (logger, nn_eval) = test_logger_and_eval();

    let params = SearchParams {
        max_visits: 5000,
        root_symmetry_pruning: true,
        wide_root_noise: 0.05,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");

    let board = Board::parse_board(
        9,
        9,
        ".......xx\n\
         xx.....xx\n\
         .........\n\
         ....x....\n\
         .........\n\
         .........\n\
         .....x...\n\
         xx......x\n\
         .......xx",
        '\n',
    )
    .expect("parse 9x9 board");
    let next_pla = P_BLACK;
    let (board, avoid) = board_with_avoid_moves_from_black_marks(board);
    let hist = BoardHistory::new(board.clone(), next_pla, Rules::get_tromp_taylorish(), 0);

    search.set_position(next_pla, &board, &hist);
    search.set_root_symmetry_pruning_only(&[0, 3, 4, 7]);
    search.set_avoid_move_until_by_loc(&avoid, &avoid);
    search.run_whole_search(next_pla);

    assert!(
        verify_tree_post_order(&search) > 1,
        "search should produce a nontrivial tree"
    );

    let mut json = serde_json::Value::Null;
    let suc = search.get_analysis_json(
        P_WHITE, 2, true, false, false, false, false, false, false, false, &mut json,
    );
    assert!(suc, "get_analysis_json should succeed");
    assert!(
        json.get("moveInfos").is_some(),
        "analysis should contain moveInfos"
    );
}

#[test]
fn pruning_of_search_tree_at_root_no_symmetries_with_avoid_moves() {
    let (logger, nn_eval) = test_logger_and_eval();

    let params = SearchParams {
        max_visits: 5000,
        root_symmetry_pruning: false,
        wide_root_noise: 0.05,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");

    let board = Board::parse_board(
        9,
        9,
        ".......xx\n\
         xx.....xx\n\
         .........\n\
         ....xo...\n\
         ....o....\n\
         .........\n\
         .....x...\n\
         xx......x\n\
         .......xx",
        '\n',
    )
    .expect("parse 9x9 board");
    let next_pla = P_BLACK;
    let (board, avoid) = board_with_avoid_moves_from_black_marks(board);
    let hist = BoardHistory::new(board.clone(), next_pla, Rules::get_tromp_taylorish(), 0);

    search.set_position(next_pla, &board, &hist);
    search.set_root_symmetry_pruning_only(&[0, 3, 4, 7]);
    search.set_avoid_move_until_by_loc(&avoid, &avoid);
    search.run_whole_search(next_pla);

    assert!(
        verify_tree_post_order(&search) > 1,
        "search should produce a nontrivial tree"
    );

    let mut json = serde_json::Value::Null;
    let suc = search.get_analysis_json(
        P_WHITE, 2, true, false, false, false, false, false, false, false, &mut json,
    );
    assert!(suc, "get_analysis_json should succeed");
    assert!(
        json.get("moveInfos").is_some(),
        "analysis should contain moveInfos"
    );
}

#[test]
fn pruning_of_search_tree_at_root_due_to_symmetries_diagonal_flips_one_diagonal_with_avoid_moves() {
    let (logger, nn_eval) = test_logger_and_eval();

    let params = SearchParams {
        max_visits: 5000,
        root_symmetry_pruning: true,
        wide_root_noise: 0.05,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");

    let board = Board::parse_board(
        9,
        9,
        ".......xx\n\
         xx.....xx\n\
         .........\n\
         ....xo...\n\
         ....o....\n\
         .........\n\
         .....x...\n\
         xx......x\n\
         .......xx",
        '\n',
    )
    .expect("parse 9x9 board");
    let next_pla = P_BLACK;
    let (board, avoid) = board_with_avoid_moves_from_black_marks(board);
    let hist = BoardHistory::new(board.clone(), next_pla, Rules::get_tromp_taylorish(), 0);

    search.set_position(next_pla, &board, &hist);
    search.set_root_symmetry_pruning_only(&[0, 3, 4, 7]);
    search.set_avoid_move_until_by_loc(&avoid, &avoid);
    search.run_whole_search(next_pla);

    assert!(
        verify_tree_post_order(&search) > 1,
        "search should produce a nontrivial tree"
    );

    let mut json = serde_json::Value::Null;
    let suc = search.get_analysis_json(
        P_WHITE, 2, true, false, false, false, false, false, false, false, &mut json,
    );
    assert!(suc, "get_analysis_json should succeed");
    assert!(
        json.get("moveInfos").is_some(),
        "analysis should contain moveInfos"
    );
}

#[test]
fn non_square_board_search() {
    let (logger, nn_eval) = test_logger_and_eval();

    let params = SearchParams {
        max_visits: 100,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");

    let board = Board::parse_board(
        7,
        17,
        ".......\n\
         .......\n\
         ..x.o..\n\
         .......\n\
         ...o...\n\
         .......\n\
         .......\n\
         .......\n\
         .......\n\
         .......\n\
         ...x...\n\
         .......\n\
         .......\n\
         ..xx...\n\
         ..oox..\n\
         ....o..\n\
         .......",
        '\n',
    )
    .expect("parse 7x17 board");
    let next_pla = P_BLACK;
    let hist = BoardHistory::new(board.clone(), next_pla, Rules::get_tromp_taylorish(), 0);

    search.set_position(next_pla, &board, &hist);
    search.run_whole_search(next_pla);

    assert!(
        verify_tree_post_order(&search) > 1,
        "search should produce a nontrivial tree"
    );
}

#[test]
fn dirichlet_noise_visualization() {
    let params = SearchParams {
        root_noise_enabled: true,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let mut rand = Rand::new_from_seed("noiseVisualize");

    let mut run = |x_size: i32, y_size: i32| {
        let nn_x_len = 19;
        let nn_y_len = 19;
        let mut sum = 0.0_f32;
        let mut counter = 0;
        let mut policy_probs = vec![-1.0_f32; nn_pos::MAX_NN_POLICY_SIZE];
        for y in 0..y_size {
            for x in 0..x_size {
                let pos = nn_pos::xy_to_pos(x, y, nn_x_len);
                policy_probs[pos as usize] = 0.9_f32.powi(counter);
                sum += policy_probs[pos as usize];
                counter += 1;
            }
        }
        let pass_pos = nn_pos::loc_to_pos(PASS_LOC, x_size, nn_x_len, nn_y_len);
        policy_probs[pass_pos as usize] = 0.9_f32.powi(counter);
        sum += policy_probs[pass_pos as usize];

        for p in &mut policy_probs {
            if *p >= 0.0 {
                *p /= sum;
            }
        }

        let mut noisy = policy_probs.clone();
        Search::add_dirichlet_noise(
            &params,
            &mut rand,
            nn_pos::MAX_NN_POLICY_SIZE as i32,
            &mut noisy,
        );

        let noisy_sum: f32 = noisy.iter().filter(|&&x| x >= 0.0).sum();
        assert!(
            (noisy_sum - 1.0).abs() < 1e-5,
            "noisy policy should remain normalized"
        );
        let changed = noisy
            .iter()
            .zip(&policy_probs)
            .any(|(a, b)| (a - b).abs() > 1e-8);
        assert!(changed, "dirichlet noise should change the policy");
    };

    run(19, 19);
    run(11, 7);
}

#[test]
fn search_tolerates_moving_past_game_end() {
    let (logger, nn_eval) = test_logger_and_eval();

    let params = SearchParams {
        max_visits: 200,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let mut search = Search::new(
        params.clone(),
        &nn_eval,
        logger.as_ref(),
        "autoSearchRandSeed",
    );
    let mut search2 = Search::new(
        params.clone(),
        &nn_eval,
        logger.as_ref(),
        "autoSearchRandSeed",
    );
    let mut search3 = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");

    let mut board = Board::parse_board(
        7,
        7,
        ".x.xo.o\n\
         xxxoooo\n\
         xxxxoo.\n\
         x.xo.oo\n\
         xxxoooo\n\
         xxxxooo\n\
         .xxxooo",
        '\n',
    )
    .expect("parse 7x7 board");
    let mut next_pla = P_WHITE;
    let mut hist = BoardHistory::new(board.clone(), next_pla, Rules::get_tromp_taylorish(), 0);

    search.set_position(next_pla, &board, &hist);
    search2.set_position(next_pla, &board, &hist);
    search3.set_position(next_pla, &board, &hist);

    let c7 = location::of_string("C7", board.x_size, board.y_size).expect("valid coordinate");
    search.make_move(c7, next_pla);
    search2.make_move(c7, next_pla);
    hist.make_board_move_assume_legal(&mut board, c7, next_pla);
    next_pla = get_opp(next_pla);
    search3.set_position(next_pla, &board, &hist);

    search.make_move(PASS_LOC, next_pla);
    search2.make_move(PASS_LOC, next_pla);
    hist.make_board_move_assume_legal(&mut board, PASS_LOC, next_pla);
    next_pla = get_opp(next_pla);
    search3.set_position(next_pla, &board, &hist);

    search2.run_whole_search(next_pla);

    search.make_move(PASS_LOC, next_pla);
    search2.make_move(PASS_LOC, next_pla);
    hist.make_board_move_assume_legal(&mut board, PASS_LOC, next_pla);
    next_pla = get_opp(next_pla);
    search3.set_position(next_pla, &board, &hist);

    assert!(
        hist.is_game_finished,
        "game should be finished after two passes"
    );

    search.run_whole_search(next_pla);
    search2.run_whole_search(next_pla);
    search3.run_whole_search(next_pla);

    let d7 = location::of_string("D7", board.x_size, board.y_size).expect("valid coordinate");
    search.make_move(d7, next_pla);
    search2.make_move(d7, next_pla);
    hist.make_board_move_assume_legal(&mut board, d7, next_pla);
    next_pla = get_opp(next_pla);
    search3.set_position(next_pla, &board, &hist);

    search.run_whole_search(next_pla);
    search2.run_whole_search(next_pla);
    search3.run_whole_search(next_pla);
}

#[test]
fn analysis_json() {
    let (logger, nn_eval) = test_logger_and_eval();

    let params = SearchParams {
        max_visits: 10,
        subtree_value_bias_factor: 0.5,
        chosen_move_temperature: 0.0,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");
    search.set_always_include_owner_map(true);

    let board = Board::parse_board(
        7,
        7,
        ".......\n\
         .......\n\
         .......\n\
         .......\n\
         .......\n\
         .......\n\
         .......",
        '\n',
    )
    .expect("parse 7x7 board");
    let next_pla = P_BLACK;
    let hist = BoardHistory::new(board.clone(), next_pla, Rules::get_tromp_taylorish(), 0);

    search.set_position(next_pla, &board, &hist);
    search.run_whole_search(next_pla);

    let mut json = serde_json::Value::Null;
    let suc = search.get_analysis_json(
        P_WHITE, 2, true, true, true, false, false, false, true, false, &mut json,
    );
    assert!(suc, "get_analysis_json should succeed");
    assert!(
        json.get("moveInfos").is_some(),
        "analysis should contain moveInfos"
    );
    assert!(
        json.get("ownership").is_some(),
        "analysis should contain ownership"
    );
}

#[test]
fn analysis_json_with_moves_ownership_and_stdev() {
    let (logger, nn_eval) = test_logger_and_eval();

    let params = SearchParams {
        max_visits: 4,
        subtree_value_bias_factor: 0.5,
        chosen_move_temperature: 0.0,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");
    search.set_always_include_owner_map(true);

    let board = Board::parse_board(
        7,
        7,
        ".......\n\
         .......\n\
         .......\n\
         .......\n\
         .......\n\
         .......\n\
         .......",
        '\n',
    )
    .expect("parse 7x7 board");
    let next_pla = P_BLACK;
    let hist = BoardHistory::new(board.clone(), next_pla, Rules::get_tromp_taylorish(), 0);

    search.set_position(next_pla, &board, &hist);
    search.run_whole_search(next_pla);

    let mut json = serde_json::Value::Null;
    let suc = search.get_analysis_json(
        P_WHITE, 2, true, false, true, true, true, true, false, false, &mut json,
    );
    assert!(suc, "get_analysis_json should succeed");
    assert!(
        json.get("moveInfos").is_some(),
        "analysis should contain moveInfos"
    );
    assert!(
        json.get("ownership").is_some(),
        "analysis should contain ownership"
    );
}

#[test]
fn analysis_json_with_moves_ownership_and_stdev_and_symmetry() {
    let (logger, nn_eval) = test_logger_and_eval();

    let params = SearchParams {
        max_visits: 4,
        subtree_value_bias_factor: 0.5,
        chosen_move_temperature: 0.0,
        root_symmetry_pruning: true,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");
    search.set_always_include_owner_map(true);

    let board = Board::parse_board(
        7,
        7,
        ".......\n\
         .......\n\
         .......\n\
         .......\n\
         .......\n\
         .......\n\
         .......",
        '\n',
    )
    .expect("parse 7x7 board");
    let next_pla = P_BLACK;
    let hist = BoardHistory::new(board.clone(), next_pla, Rules::get_tromp_taylorish(), 0);

    search.set_position(next_pla, &board, &hist);
    search.run_whole_search(next_pla);

    let mut json = serde_json::Value::Null;
    let suc = search.get_analysis_json(
        P_WHITE, 2, true, false, true, true, true, true, false, false, &mut json,
    );
    assert!(suc, "get_analysis_json should succeed");
    assert!(
        json.get("moveInfos").is_some(),
        "analysis should contain moveInfos"
    );
    assert!(
        json.get("ownership").is_some(),
        "analysis should contain ownership"
    );
}

#[test]
fn analysis_json_2() {
    let (logger, nn_eval) = test_logger_and_eval();

    let params = SearchParams {
        max_visits: 10,
        subtree_value_bias_factor: 0.5,
        chosen_move_temperature: 0.0,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");
    search.set_always_include_owner_map(false);

    let board = Board::parse_board(
        9,
        6,
        ".........\n\
         ooooooooo\n\
         oooxxxooo\n\
         ..xxxxx..\n\
         xxx...xxx\n\
         xxxxxxxxx",
        '\n',
    )
    .expect("parse 9x6 board");
    let next_pla = P_BLACK;
    let hist = BoardHistory::new(board.clone(), next_pla, Rules::get_tromp_taylorish(), 0);

    search.set_position(next_pla, &board, &hist);
    search.run_whole_search(next_pla);

    let mut json = serde_json::Value::Null;
    let suc = search.get_analysis_json(
        P_WHITE, 2, true, true, false, false, false, false, false, false, &mut json,
    );
    assert!(suc, "get_analysis_json should succeed");
    assert!(
        json.get("moveInfos").is_some(),
        "analysis should contain moveInfos"
    );
}

/// Call search-result extraction methods and ensure none panic, even at 0 visits
/// or terminal positions.
fn assert_search_results_methods_do_not_panic(
    search: &Search<'_>,
    allow_direct_policy_moves: bool,
) {
    let mut values = ReportedSearchValues::new();
    let _ = search.get_root_visits();
    let _ = search.get_root_values(&mut values);
    let _ = search.get_pruned_root_values(&mut values);

    if let Some(root) = search.root_node.as_deref() {
        if let Some(pass_child) = search.get_child_for_move(root, PASS_LOC) {
            let _ = search.get_node_values(pass_child, &mut values);
            let _ = search.get_pruned_node_values(pass_child, &mut values);
        }

        let mut locs = Vec::new();
        let mut play_selection_values = Vec::new();
        let _ = search.get_play_selection_values_for_node(
            root,
            &mut locs,
            &mut play_selection_values,
            None,
            1.0,
            allow_direct_policy_moves,
        );
    }

    let mut json = serde_json::Value::Null;
    let _ = search.get_analysis_json(
        P_WHITE, 2, true, true, false, false, false, false, true, false, &mut json,
    );
}

#[test]
fn integrity_of_value_bias_mem_safety_and_updates() {
    let (logger, nn_eval) = test_logger_and_eval();

    let params = SearchParams {
        max_visits: 500,
        subtree_value_bias_factor: 0.5,
        chosen_move_temperature: 0.0,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");

    let board = Board::parse_board(
        7,
        7,
        "x.xxxx.\n\
         xxxooxx\n\
         xxxxox.\n\
         xxx.oxx\n\
         ooxoooo\n\
         o.oo.oo\n\
         .oooooo",
        '\n',
    )
    .expect("parse 7x7 board");
    let mut next_pla = P_BLACK;
    let hist = BoardHistory::new(board.clone(), next_pla, Rules::get_tromp_taylorish(), 0);

    search.set_position(next_pla, &board, &hist);
    search.run_whole_search(next_pla);
    assert!(count_reachable_nodes(&search) > 1);

    for _ in 0..3 {
        let chosen = search.get_chosen_move_loc();
        assert!(
            search.make_move(chosen, next_pla),
            "make_move should succeed"
        );
        next_pla = get_opp(next_pla);
        search.run_whole_search(next_pla);
        assert!(count_reachable_nodes(&search) > 1);
    }
}

#[test]
fn value_bias_with_ko() {
    let (logger, nn_eval) = test_logger_and_eval();

    let params = SearchParams {
        max_visits: 2000,
        subtree_value_bias_factor: 0.8,
        chosen_move_temperature: 0.0,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let mut search = Search::new(params, &nn_eval, logger.as_ref(), "seeeed");

    let board = Board::parse_board(
        14,
        14,
        ".oo.ox.xxxxxxx\n\
         o.oooxxxxxxxxx\n\
         xooxxxxxxxxxxx\n\
         .xxxxxxxxxxxxx\n\
         xxxxxxxooooxxx\n\
         xxxxxxxo.ox.x.\n\
         xxxxxxoooooxxx\n\
         xxxxxxo.oo.oxx\n\
         xxxxxxooooooxx\n\
         oxxxxxxxxxxxx.\n\
         .oooxxxxxxxxxx\n\
         oo.oxxxxxxxxoo\n\
         .ooooooooooo..\n\
         ooooo.oooooooo",
        '\n',
    )
    .expect("parse 14x14 board");
    let mut next_pla = P_BLACK;
    let rules = Rules::parse_rules("japanese").expect("parse japanese rules");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    search.set_position(next_pla, &board, &hist);

    for _ in 0..4 {
        search.run_whole_search(next_pla);
        assert!(count_reachable_nodes(&search) > 1);
        let chosen = search.get_chosen_move_loc();
        assert!(
            search.make_move(chosen, next_pla),
            "make_move should succeed"
        );
        next_pla = get_opp(next_pla);
    }
}

#[test]
fn search_results_at_0_1_2_visits_and_terminal_position() {
    let (logger, nn_eval) = test_logger_and_eval();

    let board = Board::parse_board(
        7,
        7,
        ".......\n\
         .......\n\
         ..O....\n\
         ....O..\n\
         ..X.X..\n\
         .......\n\
         .......",
        '\n',
    )
    .expect("parse 7x7 board");
    let next_pla = P_BLACK;
    let hist = BoardHistory::new(board.clone(), next_pla, Rules::get_tromp_taylorish(), 0);

    // 0 visits, no direct policy moves.
    {
        let params = SearchParams {
            max_visits: 1,
            value_weight_exponent: 0.0,
            ..SearchParams::default()
        };
        let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");
        search.set_position(next_pla, &board, &hist);
        search.begin_search(false);
        assert_search_results_methods_do_not_panic(&search, false);
    }

    // 0 visits, with direct policy moves.
    {
        let params = SearchParams {
            max_visits: 1,
            value_weight_exponent: 0.0,
            ..SearchParams::default()
        };
        let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");
        search.set_position(next_pla, &board, &hist);
        search.begin_search(false);
        assert_search_results_methods_do_not_panic(&search, true);
    }

    // 1 visit.
    {
        let params = SearchParams {
            max_visits: 1,
            value_weight_exponent: 0.0,
            ..SearchParams::default()
        };
        let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");
        search.set_position(next_pla, &board, &hist);
        search.run_whole_search(next_pla);
        assert_search_results_methods_do_not_panic(&search, false);
        assert_search_results_methods_do_not_panic(&search, true);
    }

    // 2 visits.
    {
        let params = SearchParams {
            max_visits: 2,
            value_weight_exponent: 0.0,
            ..SearchParams::default()
        };
        let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");
        search.set_position(next_pla, &board, &hist);
        search.run_whole_search(next_pla);
        assert_search_results_methods_do_not_panic(&search, false);
        assert_search_results_methods_do_not_panic(&search, true);
    }

    // 2 visits at terminal position (two passes).
    {
        let params = SearchParams {
            max_visits: 2,
            value_weight_exponent: 0.0,
            ..SearchParams::default()
        };
        let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");
        search.set_position(next_pla, &board, &hist);
        search.make_move(PASS_LOC, P_BLACK);
        search.make_move(PASS_LOC, P_WHITE);
        search.run_whole_search(next_pla);
        assert_search_results_methods_do_not_panic(&search, false);
        assert_search_results_methods_do_not_panic(&search, true);
    }

    // 1000 visits just before terminal, then pass with tree reuse.
    {
        let params = SearchParams {
            max_visits: 1000,
            value_weight_exponent: 0.0,
            ..SearchParams::default()
        };
        let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");
        search.set_position(next_pla, &board, &hist);
        search.make_move(PASS_LOC, P_BLACK);
        search.set_root_hint_loc(PASS_LOC);
        search.run_whole_search(P_WHITE);
        assert_search_results_methods_do_not_panic(&search, false);
        search.make_move(PASS_LOC, P_WHITE);
        assert_search_results_methods_do_not_panic(&search, false);
    }
}

#[test]
fn coherence_of_search_tree_recursive_walking() {
    let (logger, nn_eval) = test_logger_and_eval();

    let params = SearchParams {
        max_visits: 1000,
        dynamic_score_utility_factor: 3.0,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");

    let board = Board::parse_board(
        7,
        7,
        "x.xx.x.\n\
         xx.x.xx\n\
         xxx..xx\n\
         ...ooo.\n\
         xxxo.o.\n\
         ooooooo\n\
         .o.oo.x",
        '\n',
    )
    .expect("parse 7x7 board");
    let next_pla = P_BLACK;
    let hist = BoardHistory::new(board.clone(), next_pla, Rules::get_tromp_taylorish(), 0);

    search.set_position(next_pla, &board, &hist);
    search.run_whole_search(next_pla);

    let root = search.root_node.as_deref().expect("root node should exist");
    let children = root.get_children();
    assert!(children.get_capacity() > 1);
    assert!(children.iterate_and_count_children() > 1);
    assert!(children.get(1).get_if_allocated().is_some());
    let loc_to_descend = children.get(1).get_move_loc();

    assert!(verify_tree_post_order(&search) > 1);

    assert!(search.make_move(loc_to_descend, next_pla));
    assert!(verify_tree_post_order(&search) > 1);

    search.begin_search(false);
    assert!(verify_tree_post_order(&search) > 1);
}

#[test]
fn avoiding_all_or_almost_all_moves() {
    let (logger, nn_eval) = test_logger_and_eval();

    let params = SearchParams {
        max_visits: 100,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };

    let board = Board::parse_board(
        9,
        5,
        "xx..x..xx\n\
         xxxxxxxxx\n\
         ....oxoxo\n\
         ooooooooo\n\
         oo..o..oo",
        '\n',
    )
    .expect("parse 9x5 board");

    // Avoid all but the first two columns (and pass) for both players.
    {
        let mut search = Search::new(
            params.clone(),
            &nn_eval,
            logger.as_ref(),
            "autoSearchRandSeed1234",
        );
        let mut avoid = vec![0; MAX_ARR_SIZE];
        for y in 0..board.y_size {
            for x in 0..board.x_size {
                let loc = location::get_loc(x, y, board.x_size);
                if x == 0 || x == 1 {
                    avoid[loc as usize] = 0;
                } else {
                    avoid[loc as usize] = 3;
                }
            }
        }
        avoid[PASS_LOC as usize] = 3;

        let next_pla = P_WHITE;
        let hist = BoardHistory::new(board.clone(), next_pla, Rules::get_tromp_taylorish(), 0);
        search.set_position(next_pla, &board, &hist);
        search.set_avoid_move_until_by_loc(&avoid, &avoid);
        search.run_whole_search(next_pla);

        assert!(count_reachable_nodes(&search) > 1);
        let mut json = serde_json::Value::Null;
        assert!(search.get_analysis_json(
            P_WHITE, 2, true, false, false, false, false, false, false, false, &mut json,
        ));
        assert!(json.get("moveInfos").is_some());

        let loc_to_descend =
            location::of_string("B3", board.x_size, board.y_size).expect("valid coordinate");
        assert!(search.make_move(loc_to_descend, next_pla));
        let next_pla = get_opp(next_pla);
        search.run_whole_search(next_pla);
        assert!(count_reachable_nodes(&search) > 1);
    }

    // Avoid all moves (and pass) for both players.
    {
        let mut search = Search::new(
            params.clone(),
            &nn_eval,
            logger.as_ref(),
            "autoSearchRandSeed1235",
        );
        let mut avoid = vec![0; MAX_ARR_SIZE];
        for y in 0..board.y_size {
            for x in 0..board.x_size {
                let loc = location::get_loc(x, y, board.x_size);
                avoid[loc as usize] = 3;
            }
        }
        avoid[PASS_LOC as usize] = 3;

        let next_pla = P_WHITE;
        let hist = BoardHistory::new(board.clone(), next_pla, Rules::get_tromp_taylorish(), 0);
        search.set_position(next_pla, &board, &hist);
        search.set_avoid_move_until_by_loc(&avoid, &avoid);
        search.run_whole_search(next_pla);

        assert!(count_reachable_nodes(&search) >= 1);
        let mut json = serde_json::Value::Null;
        assert!(search.get_analysis_json(
            P_WHITE, 2, true, false, false, false, false, false, false, false, &mut json,
        ));
        assert!(json.get("moveInfos").is_some());
    }

    // Avoid all moves (and pass) for black only.
    {
        let mut search = Search::new(
            params.clone(),
            &nn_eval,
            logger.as_ref(),
            "autoSearchRandSeed1236",
        );
        let mut avoid = vec![0; MAX_ARR_SIZE];
        for y in 0..board.y_size {
            for x in 0..board.x_size {
                let loc = location::get_loc(x, y, board.x_size);
                avoid[loc as usize] = 10;
            }
        }
        avoid[PASS_LOC as usize] = 10;

        let next_pla = P_WHITE;
        let hist = BoardHistory::new(board.clone(), next_pla, Rules::get_tromp_taylorish(), 0);
        search.set_position(next_pla, &board, &hist);
        search.set_avoid_move_until_by_loc(&avoid, &[]);
        search.run_whole_search(next_pla);

        assert!(count_reachable_nodes(&search) > 1);
        let mut json = serde_json::Value::Null;
        assert!(search.get_analysis_json(
            P_WHITE, 2, true, false, false, false, false, false, false, false, &mut json,
        ));
        assert!(json.get("moveInfos").is_some());
    }
}

#[test]
fn graph_search_opening() {
    let (logger, nn_eval) = test_logger_and_eval();

    let params = SearchParams {
        max_visits: 1000,
        subtree_value_bias_factor: 0.5,
        use_graph_search: true,
        chosen_move_temperature: 0.0,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");

    let board = Board::parse_board(
        7,
        7,
        ".......\n\
         .......\n\
         ...o...\n\
         ..ox...\n\
         ..x....\n\
         .......\n\
         .......",
        '\n',
    )
    .expect("parse 7x7 board");
    let mut next_pla = P_BLACK;
    let hist = BoardHistory::new(board.clone(), next_pla, Rules::get_tromp_taylorish(), 0);

    search.set_position(next_pla, &board, &hist);

    search.run_whole_search(next_pla);
    assert!(count_reachable_nodes(&search) > 1);

    for _ in 0..3 {
        let chosen = search.get_chosen_move_loc();
        assert!(search.make_move(chosen, next_pla));
        next_pla = get_opp(next_pla);
        search.run_whole_search(next_pla);
        assert!(count_reachable_nodes(&search) > 1);
    }
}

#[test]
fn graph_search_7x7_big_fight() {
    let (logger, nn_eval) = test_logger_and_eval();

    let params = SearchParams {
        max_visits: 1000,
        subtree_value_bias_factor: 0.5,
        use_graph_search: true,
        chosen_move_temperature: 0.0,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");

    let board = Board::parse_board(
        7,
        7,
        ".....o.\n\
         ...oxox\n\
         ..ooox.\n\
         .xoxxx.\n\
         .xxo.x.\n\
         ..xooox\n\
         .......",
        '\n',
    )
    .expect("parse 7x7 board");
    let mut next_pla = P_WHITE;
    let hist = BoardHistory::new(board.clone(), next_pla, Rules::get_tromp_taylorish(), 0);

    search.set_position(next_pla, &board, &hist);

    search.run_whole_search(next_pla);
    assert!(count_reachable_nodes(&search) > 1);

    for _ in 0..3 {
        let chosen = search.get_chosen_move_loc();
        assert!(search.make_move(chosen, next_pla));
        next_pla = get_opp(next_pla);
        search.run_whole_search(next_pla);
        assert!(count_reachable_nodes(&search) > 1);
    }
}

#[test]
fn graph_search_7x7_endgame_kos() {
    let (logger, nn_eval) = test_logger_and_eval();

    let params = SearchParams {
        max_visits: 1000,
        subtree_value_bias_factor: 0.5,
        use_graph_search: true,
        chosen_move_temperature: 0.0,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");

    let board = Board::parse_board(
        7,
        7,
        ".o.x.x.\n\
         o.oxoxo\n\
         xoxxxo.\n\
         xx.xooo\n\
         xxxxox.\n\
         oxooox.\n\
         .ooox.x",
        '\n',
    )
    .expect("parse 7x7 board");
    let mut next_pla = P_BLACK;
    let hist = BoardHistory::new(
        board.clone(),
        next_pla,
        Rules::parse_rules("japanese").expect("japanese rules"),
        0,
    );

    search.set_position(next_pla, &board, &hist);

    search.run_whole_search(next_pla);
    assert!(count_reachable_nodes(&search) > 1);

    for _ in 0..3 {
        let chosen = search.get_chosen_move_loc();
        assert!(search.make_move(chosen, next_pla));
        next_pla = get_opp(next_pla);
        search.run_whole_search(next_pla);
        assert!(count_reachable_nodes(&search) > 1);
    }
}

fn run_fpu_parent_weight_test(fpu_parent_weight_by_visited_policy: bool, pow: f64) {
    let (logger, nn_eval) = test_logger_and_eval();

    let mut params = SearchParams::for_tests_v1();
    params.max_visits = 1000;
    params.fpu_parent_weight_by_visited_policy = fpu_parent_weight_by_visited_policy;
    params.fpu_parent_weight_by_visited_policy_pow = pow;
    params.value_weight_exponent = 0.0;

    let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeed");

    let board = Board::parse_board(
        7,
        7,
        ".o.x.x.\n\
         o.oxoxo\n\
         xoxxxo.\n\
         xx.xooo\n\
         xxxxox.\n\
         oxooox.\n\
         .ooox.x",
        '\n',
    )
    .expect("parse 7x7 board");
    let next_pla = P_BLACK;
    let hist = BoardHistory::new(
        board.clone(),
        next_pla,
        Rules::parse_rules("japanese").expect("japanese rules"),
        0,
    );

    search.set_position(next_pla, &board, &hist);
    search.run_whole_search(next_pla);

    assert!(count_reachable_nodes(&search) > 1);
}

#[test]
fn fpu_parent_weight_by_visited_policy_false() {
    run_fpu_parent_weight_test(false, 1.0);
}

#[test]
fn fpu_parent_weight_by_visited_policy_1_0() {
    run_fpu_parent_weight_test(true, 1.0);
}

#[test]
fn fpu_parent_weight_by_visited_policy_2_5() {
    run_fpu_parent_weight_test(true, 2.5);
}

#[test]
fn fpu_parent_weight_by_visited_policy_0_5() {
    run_fpu_parent_weight_test(true, 0.5);
}

fn run_policy_optimism_tree_reuse_test(root_policy_optimism: f64, policy_optimism: f64) {
    let (logger, nn_eval) = test_logger_and_eval();

    let mut params = SearchParams::for_tests_v2();
    params.max_visits = 100;
    params.root_policy_optimism = root_policy_optimism;
    params.policy_optimism = policy_optimism;
    params.value_weight_exponent = 0.0;

    let mut params_low_visits = params.clone();
    params_low_visits.max_visits = 8;

    let mut search = Search::new(
        params.clone(),
        &nn_eval,
        logger.as_ref(),
        "autoSearchRandSeeeeeed",
    );

    let board = Board::parse_board(
        5,
        5,
        ".o.o.\n\
         ooooo\n\
         xxoxx\n\
         .xxx.\n\
         x.x.x",
        '\n',
    )
    .expect("parse 5x5 board");
    let mut next_pla = P_BLACK;
    let hist = BoardHistory::new(
        board.clone(),
        next_pla,
        Rules::parse_rules("japanese").expect("japanese rules"),
        0,
    );

    search.set_position(next_pla, &board, &hist);

    search.run_whole_search(next_pla);
    let root_optimism = search
        .root_node
        .as_deref()
        .unwrap()
        .get_nn_output()
        .map(|o| o.policy_optimism_used as f64)
        .unwrap();
    assert!(
        (root_optimism - root_policy_optimism).abs() < 1e-5,
        "root optimism should be {}, got {}",
        root_policy_optimism,
        root_optimism
    );

    let chosen = search.get_chosen_move_loc();
    assert!(search.make_move(chosen, next_pla));
    next_pla = get_opp(next_pla);

    let child_optimism = search
        .root_node
        .as_deref()
        .unwrap()
        .get_nn_output()
        .map(|o| o.policy_optimism_used as f64)
        .unwrap();
    assert!(
        (child_optimism - policy_optimism).abs() < 1e-5,
        "non-root optimism should be {}, got {}",
        policy_optimism,
        child_optimism
    );

    search.run_whole_search(next_pla);
    let root_optimism = search
        .root_node
        .as_deref()
        .unwrap()
        .get_nn_output()
        .map(|o| o.policy_optimism_used as f64)
        .unwrap();
    assert!(
        (root_optimism - root_policy_optimism).abs() < 1e-5,
        "root optimism after re-search should be {}, got {}",
        root_policy_optimism,
        root_optimism
    );

    let chosen = search.get_chosen_move_loc();
    assert!(search.make_move(chosen, next_pla));
    next_pla = get_opp(next_pla);

    let child_optimism = search
        .root_node
        .as_deref()
        .unwrap()
        .get_nn_output()
        .map(|o| o.policy_optimism_used as f64)
        .unwrap();
    assert!(
        (child_optimism - policy_optimism).abs() < 1e-5,
        "non-root optimism after second move should be {}, got {}",
        policy_optimism,
        child_optimism
    );

    search.set_params_no_clearing(&params_low_visits);
    search.run_whole_search(next_pla);
    let root_optimism = search
        .root_node
        .as_deref()
        .unwrap()
        .get_nn_output()
        .map(|o| o.policy_optimism_used as f64)
        .unwrap();
    assert!(
        (root_optimism - root_policy_optimism).abs() < 1e-5,
        "root optimism with low visits should be {}, got {}",
        root_policy_optimism,
        root_optimism
    );
}

#[test]
fn policy_optimism_with_tree_reuse() {
    run_policy_optimism_tree_reuse_test(0.43, 0.71);
}

#[test]
fn policy_optimism_with_tree_reuse_zero_root() {
    run_policy_optimism_tree_reuse_test(0.0, 1.0);
}

#[test]
fn zero_node_search() {
    let (logger, nn_eval) = test_logger_and_eval();

    let mut params = SearchParams::for_tests_v2();
    params.value_weight_exponent = 0.0;

    let mut search = Search::new(params, &nn_eval, logger.as_ref(), "autoSearchRandSeeeeeed");

    let board = Board::parse_board(
        13,
        6,
        ".............\n\
         .............\n\
         .....o.......\n\
         ...x.........\n\
         .............\n\
         .............",
        '\n',
    )
    .expect("parse 13x6 board");
    let next_pla = P_BLACK;
    let hist = BoardHistory::new(
        board.clone(),
        next_pla,
        Rules::parse_rules("japanese").expect("japanese rules"),
        0,
    );

    search.set_position(next_pla, &board, &hist);

    search.run_whole_search_with_stop(next_pla, Some(&|| true));

    assert!(
        search
            .root_node
            .as_deref()
            .unwrap()
            .get_nn_output()
            .is_none(),
        "zero-node search should not evaluate the root"
    );

    // Should still be able to query a chosen move without crashing.
    let _chosen = search.get_chosen_move_loc();
}

#[test]
fn chosen_move_probs_with_temperature() {
    let relative_probs = [40.0f64, 20.0, 160.0, 80.0, 10.0, 10.0];
    let mut rand = Rand::new();

    let mut check = |temp: f64, only_below: f64| {
        let mut buf = vec![0.0; relative_probs.len()];
        let idx = Search::choose_index_with_temperature(
            &mut rand,
            &relative_probs,
            temp,
            only_below,
            Some(&mut buf),
        );
        assert!(idx < relative_probs.len() as u32);
        assert!(buf.iter().all(|&v| v >= 0.0 && v.is_finite()));
        assert!(buf.iter().any(|&v| v > 0.0));
    };

    check(1.0, 1.0);
    check(1.0, 0.2);
    check(0.5, 1.0);
    check(0.5, 0.2);
    check(2.0, 0.2);
    check(100_000.0, 0.2);
    check(0.000_01, 0.2);
}

#[test]
fn eval_cache_keys_depend_on_search_params() {
    let (logger, nn_eval) = test_logger_and_eval();

    let mut params_a = SearchParams::for_tests_v2();
    params_a.value_weight_exponent = 0.0;
    params_a.use_eval_cache = true;
    params_a.eval_cache_min_visits = 5;
    params_a.max_visits = 60;

    let mut params_b = params_a.clone();
    params_b.cpuct_exploration += 0.5;

    let params_a2 = params_a.clone();

    assert_eq!(params_a.get_hash(), params_a2.get_hash());
    assert_ne!(params_a.get_hash(), params_b.get_hash());

    let mut search = Search::new(params_a.clone(), &nn_eval, logger.as_ref(), "evalcacheseed");
    assert!(search.eval_cache.is_some());

    let board = Board::parse_board(
        7,
        7,
        ".......\n\
         ..x.o..\n\
         .......\n\
         ..o.x..\n\
         .......\n\
         .......\n\
         .......",
        '\n',
    )
    .expect("parse 7x7 board");
    let next_pla = P_BLACK;
    let hist = BoardHistory::new(
        board.clone(),
        next_pla,
        Rules::parse_rules("chinese").expect("chinese rules"),
        0,
    );

    search.set_position(next_pla, &board, &hist);
    search.run_whole_search(next_pla);

    let eval_cache = search.eval_cache.as_ref().unwrap();
    let root_graph_hash = search.root_graph_hash;
    let key_a = root_graph_hash ^ params_a.get_hash();
    let key_b = root_graph_hash ^ params_b.get_hash();

    let entry_a = eval_cache.find(key_a).expect("A slot should be populated");
    assert!(eval_cache.find(key_b).is_none(), "B slot should be empty");

    search.set_params(&params_b);
    search.set_position(next_pla, &board, &hist);
    search.run_whole_search(next_pla);

    assert_eq!(search.root_graph_hash, root_graph_hash);
    assert_ne!(key_a, key_b);
    let eval_cache = search.eval_cache.as_ref().unwrap();
    assert!(
        eval_cache.find(key_b).is_some(),
        "B slot should be populated"
    );
    let entry_a_again = eval_cache
        .find(key_a)
        .expect("A slot should still be present");
    assert!(Arc::ptr_eq(&entry_a, &entry_a_again));

    search.set_params(&params_a2);
    search.set_position(next_pla, &board, &hist);
    search.run_whole_search(next_pla);

    assert_eq!(search.root_graph_hash, root_graph_hash);
    let key_a2 = root_graph_hash ^ params_a2.get_hash();
    assert_eq!(key_a2, key_a);
    let eval_cache = search.eval_cache.as_ref().unwrap();
    assert!(eval_cache.find(key_a2).is_some(), "A2 slot should reuse A");
}

#[test]
fn search_params_display_is_non_empty() {
    let default_str = SearchParams::new().to_string();
    assert!(
        default_str.len() > 50,
        "default SearchParams should have a substantial display: {}",
        default_str
    );
    assert!(default_str.contains("win_loss_utility_factor"));

    let v1_str = SearchParams::for_tests_v1().to_string();
    assert!(
        v1_str.len() > 50,
        "for_tests_v1 SearchParams should have a substantial display: {}",
        v1_str
    );
    assert!(v1_str.contains("cpuct_exploration"));
}
