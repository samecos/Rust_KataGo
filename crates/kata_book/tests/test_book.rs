//! Smoke test for `kata_book` recomputation.
//!
//! Corresponds to a subset of `KataGo/cpp/tests/testbook.cpp`: build a small
//! book on a 4×4 board, set node values, and verify that single-threaded and
//! multi-threaded recomputation agree on expansion costs.

use kata_book::{Book, BookParams, BookValues, SymBookNode};
use kata_game::board::{Board, P_BLACK, location};
use kata_game::rules::Rules;
use kata_search::mutex_pool::MutexPool;

fn set_node_values(node: SymBookNode, book: &mut Book, win_loss: f64) {
    let values = BookValues {
        win_loss_value: win_loss,
        score_mean: 0.0,
        sharp_score_mean_raw: 0.0,
        win_loss_error: 0.1,
        score_error: 1.0,
        score_stdev: 1.0,
        max_policy: 0.1,
        weight: 15.0,
        visits: 15.0,
        ..Default::default()
    };
    *node.this_values_not_in_book_mut(book).unwrap() = values;
}

fn total_costs(book: &Book) -> std::collections::BTreeMap<kata_book::BookHash, f64> {
    book.get_all_nodes()
        .into_iter()
        .map(|node| (node.hash(), node.total_expansion_cost(book)))
        .collect()
}

#[test]
fn book_recompute_single_vs_multi_threaded_is_consistent() {
    let initial_board = Board::new(4, 4);
    let rules = Rules::parse_rules("japanese").unwrap();
    let initial_pla = P_BLACK;
    let rep_bound = 9;

    let params = BookParams {
        error_factor: 1.05,
        cost_per_move: 0.53,
        ..Default::default()
    };

    let mut book = Book::new(
        Book::LATEST_BOOK_VERSION,
        initial_board.clone(),
        rules,
        initial_pla,
        rep_bound,
        params,
    );

    let root = book.get_root();
    let mut hist = book.get_initial_hist();
    let mut board = hist.get_recent_board(0).clone();

    // Build a tiny branching book.
    let a1 = location::get_loc(0, 0, 4);
    let b1 = location::get_loc(1, 0, 4);
    let pass = 1; // PASS_LOC

    let (child_a, _) = root.play_and_add_move(&mut book, &mut board, &mut hist, a1, 0.3);
    let mut changed = vec![root, child_a];

    let mut hist_b = book.get_initial_hist();
    let mut board_b = hist_b.get_recent_board(0).clone();
    let (child_b, _) = root.play_and_add_move(&mut book, &mut board_b, &mut hist_b, b1, 0.2);
    changed.push(child_b);

    // Add one more move under child_a.
    let (grandchild, _) = child_a.play_and_add_move(&mut book, &mut board, &mut hist, pass, 0.1);
    changed.push(grandchild);

    // Set per-node values.
    set_node_values(root, &mut book, -0.1);
    set_node_values(child_a, &mut book, 0.2);
    set_node_values(child_b, &mut book, -0.3);
    set_node_values(grandchild, &mut book, 0.4);

    // Single-threaded recompute.
    book.recompute(&changed);
    let single_threaded = total_costs(&book);

    // Multi-threaded recompute currently delegates to the single-threaded path,
    // but we still exercise the API and assert identical results.
    let pool = MutexPool::new(4);
    book.recompute_multi_threaded(&changed, &pool, 4);
    let multi_threaded = total_costs(&book);

    for (hash, cost) in &single_threaded {
        let other = multi_threaded.get(hash).copied().unwrap_or(f64::NAN);
        assert!(
            (cost - other).abs() < 1e-9,
            "cost mismatch for {:?}: single {} vs multi {}",
            hash,
            cost,
            other
        );
    }

    // Full-book recomputation should also agree between single and multi paths.
    book.recompute_everything();
    let everything_single = total_costs(&book);
    book.recompute_everything_multi_threaded(&pool, 4);
    let everything_multi = total_costs(&book);

    for (hash, cost) in &everything_single {
        let other = everything_multi.get(hash).copied().unwrap_or(f64::NAN);
        assert!(
            (cost - other).abs() < 1e-9,
            "full recompute cost mismatch for {:?}: single {} vs multi {}",
            hash,
            cost,
            other
        );
    }

    // Minimax: root is black-to-move, so it picks the best child win-loss
    // value from black's perspective. child_a itself has 0.2, but it is
    // white-to-move, so white chooses the minimum between its own value 0.2
    // and the grandchild value 0.4, giving child_a a minimax value of 0.2.
    // Therefore root takes max(0.2, child_b -0.3) = 0.2.
    let root_values = root.recursive_values(&book).unwrap();
    assert!(
        (root_values.win_loss_value - 0.2).abs() < 1e-9,
        "expected root winLossValue to be 0.2, got {}",
        root_values.win_loss_value
    );
}

#[test]
fn book_save_load_roundtrip() {
    let initial_board = Board::new(5, 5);
    let rules = Rules::parse_rules("japanese").unwrap();
    let initial_pla = P_BLACK;
    let rep_bound = 9;

    let params = BookParams {
        error_factor: 1.05,
        cost_per_move: 0.53,
        ..Default::default()
    };

    let mut book = Book::new(
        Book::LATEST_BOOK_VERSION,
        initial_board.clone(),
        rules,
        initial_pla,
        rep_bound,
        params,
    );

    let root = book.get_root();
    let mut hist = book.get_initial_hist();
    let mut board = hist.get_recent_board(0).clone();

    let c3 = location::get_loc(2, 2, 5);
    let (child, _) = root.play_and_add_move(&mut book, &mut board, &mut hist, c3, 0.4);

    let changed = vec![root, child];
    set_node_values(root, &mut book, -0.1);
    set_node_values(child, &mut book, 0.2);
    book.recompute(&changed);

    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    book.save_to_file(path).unwrap();

    let loaded = Book::load_from_file(path).unwrap();
    assert_eq!(loaded.size(), book.size());

    for node in book.get_all_nodes() {
        let hash = node.hash();
        let original = book.get_by_hash(hash);
        let restored = loaded.get_by_hash(hash);
        assert!(!restored.is_null(), "missing node after load");

        let orig_values = original.this_values_not_in_book(&book).unwrap();
        let rest_values = restored.this_values_not_in_book(&loaded).unwrap();
        assert!(
            (orig_values.win_loss_value - rest_values.win_loss_value).abs() < 1e-9,
            "winLossValue mismatch"
        );
        assert!(
            (orig_values.visits - rest_values.visits).abs() < 1e-9,
            "visits mismatch"
        );

        let orig_recursive = original.recursive_values(&book).unwrap();
        let rest_recursive = restored.recursive_values(&loaded).unwrap();
        assert!(
            (orig_recursive.win_loss_value - rest_recursive.win_loss_value).abs() < 1e-9,
            "recursive winLossValue mismatch"
        );
        assert!(
            (node.total_expansion_cost(&book) - restored.total_expansion_cost(&loaded)).abs()
                < 1e-9,
            "total expansion cost mismatch"
        );
    }
}

#[test]
fn book_html_export_creates_pages() {
    let initial_board = Board::new(5, 5);
    let rules = Rules::parse_rules("japanese").unwrap();
    let initial_pla = P_BLACK;

    let params = BookParams {
        error_factor: 1.05,
        cost_per_move: 0.53,
        ..Default::default()
    };

    let mut book = Book::new(
        Book::LATEST_BOOK_VERSION,
        initial_board.clone(),
        rules,
        initial_pla,
        9,
        params,
    );

    let root = book.get_root();
    let mut hist = book.get_initial_hist();
    let mut board = hist.get_recent_board(0).clone();
    let c3 = location::get_loc(2, 2, 5);
    let (child, _) = root.play_and_add_move(&mut book, &mut board, &mut hist, c3, 0.4);

    set_node_values(root, &mut book, -0.1);
    set_node_values(child, &mut book, 0.2);
    book.recompute(&[root, child]);

    let tmp_dir = tempfile::tempdir().unwrap();
    let count = book
        .export_to_html_dir(tmp_dir.path(), "Japanese Rules", "", false, 0.0)
        .unwrap();
    assert!(count >= 2, "expected at least root and child pages");

    let root_page = tmp_dir.path().join("root").join("root.html");
    assert!(root_page.exists(), "root page missing");
    let root_html = std::fs::read_to_string(&root_page).unwrap();
    assert!(root_html.contains("Japanese Rules"));
    assert!(root_html.contains("C3") || root_html.contains("c3"));

    let child_page = tmp_dir
        .path()
        .join(&child.hash().to_string()[8..10])
        .join(format!("{}.html", child.hash().to_string()));
    assert!(child_page.exists(), "child page missing");
}
