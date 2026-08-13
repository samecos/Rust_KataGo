//! Integration tests ported from `KataGo/cpp/tests/testsearchmisc.cpp`.
//!
//! These tests use the dummy neural-net evaluator, so they are primarily
//! smoke tests that exercise direct `NnEvaluator::evaluate` calls on real
//! SGF positions without crashing.  The C++ originals compare against real
//! model outputs; those golden comparisons are skipped here because no model
//! file is available.

use std::sync::{Arc, Mutex};
use std::thread;

use kata_core::config::ConfigParser;
use kata_core::logger::{Logger, LoggerOptions};
use kata_core::rng::Rand;
use kata_data::sgf::CompactSgf;
use kata_game::board::{Board, P_BLACK, P_WHITE, Player};
use kata_game::history::BoardHistory;
use kata_game::rules::{KoRule, Rules, ScoringRule, TaxRule, WhiteHandicapBonusRule};
use kata_nn::backend::{Enabled, NNResultBuf};
use kata_nn::eval::NnEvaluator;
use kata_nn::inputs::{MiscNNInputParams, nn_pos};

const SGF_GO_SEIGEN: &str = concat!(
    "(;SZ[19]FF[3]PW[Go Seigen]WR[9d]PB[Takagawa Shukaku]BR[8d]DT[1957-09-26]KM[0]RE[W+R];B[qd];W[dc];B[pp];W[cp];B[eq];W[oc];B[ce];W[dh];B[fe];W[gc];B[do];W[co];B[dn];W[cm];B[jq];W[qn];B[pn];W[pm];B[on];W[qq];B[qo];W[or];B[mr];W[mq];B[nr];W[oq];B[lq];W[qm];B[rp];W[rq];B[qg];W[mp];B[lp];W[mo];B[om];W[pk];B[kn];W[mm];B[ok];W[pj];B[mk];W[op];B[dm];W[cl];B[dl];W[dk];B[ek];W[ll];B[cn];W[bn];B[bo];W[bm];B[cq];W[bp];B[oj];W[ph];B[qh];W[oi];B[qi];W[pi];B[mi];W[of];B[ki];W[qc];B[rc];W[qe];B[re];W[pd];B[rd];W",
    "[de];B[df];W[cd];B[ee];W[dd];B[fg];W[hd];B[jl];W[dj];B[bf];W[fj];B[hg];W[dp];B[ep];W[jk];B[il];W[fk];B[ie];W[he];B[hf];W[gm];B[ke];W[fo];B[eo];W[in];B[ho];W[hn];B[fn];W[gn];B[go];W[io];B[ip];W[jp];B[hq];W[qf];B[rf];W[qb];B[ik];W[lr];B[id];W[kr];B[jr];W[bq];B[ib];W[hb];B[cr];W[rj];B[rb];W[kk];B[ij];W[ic];B[jc];W[jb];B[hc];W[iq];B[ir];W[ic];B[kq];W[kc];B[hc];W[nj];B[nk];W[ic];B[oe];W[jd];B[pe];W[pf];B[od];W[pc];B[md];W[mc];B[me];W[ld];B[ng];W[ri];B[rh];W[pg];B[fl];W[je];B[kg];W[be];B[cf];W[bh];B[b",
    "d];W[bc];B[ae];W[kl];B[rn];W[mj];B[lj];W[ni];B[lk];W[mh];B[li];W[mg];B[mf];W[nh];B[jf];W[qj];B[sh];W[rm];B[km];W[if];B[ig];W[dq];B[dr];W[br];B[ci];W[gi];B[ei];W[ej];B[di];W[gl];B[bi];W[cj];B[sq];W[sr];B[so];W[sp];B[fc];W[fb];B[sq];W[lo];B[rr];W[sp];B[ec];W[eb];B[sq];W[ko];B[jn];W[sp];B[nc];W[nb];B[sq];W[nd];B[jo];W[sp];B[qr];W[pq];B[sq];W[ns];B[ks];W[sp];B[bk];W[bj];B[sq];W[ol];B[nl];W[sp];B[aj];W[ck];B[sq];W[nq];B[ls];W[sp];B[gk];W[qp];B[po];W[ro];B[gj];W[eh];B[rp];W[fi];B[sq];W[pl];B[nm];W[sp]",
    ";B[ch];W[ro];B[dg];W[sn];B[ne];W[er];B[fr];W[cs];B[es];W[fh];B[bb];W[cb];B[ac];W[ba];B[cc];W[el];B[fm];W[bc])"
);

const SGF_19X10: &str = "(;FF[4]GM[1]SZ[19:10]HA[0]KM[6]RU[koPOSITIONALscoreAREAtaxNONEsui0]RE[W+2];B[dg];W[cd];B[pg];W[pd];B[ec];W[bg];B[cg];W[bh];B[nc];W[de];B[cf];W[di];B[bf];W[eh];B[eg];W[fd];B[dc];W[cc];B[gb];W[he];B[ee];W[ed];B[ci];W[dd];B[dh];W[hc];B[hh];W[jc];B[kc];W[kb];B[jd];W[lc];B[kd];W[ic];B[oe];W[ld];B[re];W[pe];B[pf];W[od];B[nd];W[ob];B[le];W[rd];B[kf];W[oi];B[ph];W[kh];B[ji];W[mh];B[ki];W[kg];B[jf];W[qf];B[rf];W[qe];B[qg];W[pi];B[qc];W[qb];B[qi];W[mf];B[me];W[nf];B[ng];W[mg];B[li];W[rh];B[rg];W[ri];B[nh];W[ne];B[of];W[ig];B[lh];W[qh];B[ni];W[hg];B[sh];W[ih];B[lg];W[fh];B[rb];W[nb];B[rc];W[qj];B[si];W[oj];B[hi];W[fg];B[rj];W[sj];B[ff];W[gf];B[rj];W[mc];B[md];W[sj];B[cb];W[bb];B[rj];W[ei];B[bi];W[sj];B[eb];W[ca];B[rj];W[be];B[af];W[sj];B[da];W[ba];B[rj];W[ai];B[cj];W[sj];B[hb];W[ib];B[rj];W[pb];B[qi];W[qd];B[sd];W[ii];B[ij];W[ra];B[id];W[gi];B[gj];W[gh];B[fj];W[hj];B[hi];W[hd];B[ce];W[bd];B[if];W[ie];B[je];W[ae];B[ef];W[hf];B[fe];W[ge];B[hj];W[df];B[ah];W[hh];B[sa];W[qa];B[sc];W[sb];B[bc];W[ac];B[sa];W[ag];B[ch];W[sb];B[lb];W[mb];B[sa];W[se];B[sf];W[sb];B[jb];W[ja];B[sa];W[pc];B[se];W[sb];B[fa];W[fc];B[sa];W[dj];B[ah];W[sb];B[fb];W[ha];B[sa];W[ag];B[bg];W[sb];B[jg];W[jh];B[sa];W[lf];B[mi];W[sb];B[jj];W[sa];B[ej];W[fi];B[oc];W[ia];B[lf];W[la];B[nj];W[bc];B[sg];W[db];B[pj];W[ea];B[qj];W[da];B[aj];W[gc];B[oh];W[ga];B[];W[])";

fn test_logger_and_eval(nn_x_len: i32, nn_y_len: i32) -> (Arc<Logger>, NnEvaluator) {
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
        false,
        false,
        16,
        12,
        true,
        String::new(),
        Enabled::False,
        1,
        vec![0],
        "runSearchMiscTestsRandSeed".to_string(),
        false,
        0,
        true,
        &cfg,
    );
    nn_eval.spawn_server_threads();
    (logger, nn_eval)
}

#[test]
fn nn_on_tiny_board_smoke() {
    let (_logger, nn_eval) = test_logger_and_eval(6, 6);
    let board =
        Board::parse_board(5, 5, ".....\n...x.\n..o..\n.xxo.\n.....\n", '\n').expect("valid board");
    let next_pla = P_WHITE;
    let rules = Rules::get_tromp_taylorish();
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let nn_input_params = MiscNNInputParams::default();
    let mut buf = NNResultBuf::new();
    nn_eval.evaluate(
        &board,
        &hist,
        next_pla,
        &nn_input_params,
        &mut buf,
        true,
        true,
    );
    let output = buf.result.expect("evaluate should produce a result");
    assert!(output.white_win_prob >= 0.0 && output.white_win_prob <= 1.0);
    assert!(output.white_loss_prob >= 0.0 && output.white_loss_prob <= 1.0);
    assert!(output.white_owner_map.is_some());
}

#[test]
fn nn_symmetries_smoke() {
    let (_logger, nn_eval) = test_logger_and_eval(13, 13);
    let board = Board::parse_board(
        9,
        13,
        ".........\n.........\n..x.o....\n......o..\n..x......\n.........\n......o..\n.........\n..x......\n....x....\n...xoo...\n.........\n.........\n",
        '\n',
    )
    .expect("valid board");
    let next_pla = P_BLACK;
    let rules = Rules::get_tromp_taylorish();
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    for symmetry in 0..8 {
        nn_eval.set_do_randomize(false);
        nn_eval.set_default_symmetry(symmetry);
        nn_eval.clear_cache();

        let nn_input_params = MiscNNInputParams::default();
        let mut buf = NNResultBuf::new();
        nn_eval.evaluate(
            &board,
            &hist,
            next_pla,
            &nn_input_params,
            &mut buf,
            true,
            true,
        );
        let output = buf.result.expect("evaluate should produce a result");
        assert!(output.white_win_prob >= 0.0 && output.white_win_prob <= 1.0);
    }
}

#[test]
fn nn_on_many_poses_smoke() {
    let (_logger, nn_eval) = test_logger_and_eval(19, 19);
    let sgf = CompactSgf::parse(SGF_GO_SEIGEN).expect("valid SGF");
    let default_rules = Rules::get_tromp_taylorish();

    let mut win_probs = Vec::new();
    let mut score_means = Vec::new();
    let mut policy_probs = Vec::new();

    for turn_idx in 0..sgf.moves.len() as i64 {
        let mut board = Board::default();
        let mut next_pla = 0;
        let mut hist = BoardHistory::default();
        let initial_rules = sgf
            .get_rules_or_fail_allow_unspecified(&default_rules)
            .expect("rules");
        sgf.setup_board_and_hist_assume_legal(
            &initial_rules,
            &mut board,
            &mut next_pla,
            &mut hist,
            turn_idx,
        )
        .expect("setup board");

        let nn_input_params = MiscNNInputParams::default();
        let mut buf = NNResultBuf::new();
        nn_eval.evaluate(
            &board,
            &hist,
            next_pla,
            &nn_input_params,
            &mut buf,
            true,
            true,
        );
        let output = buf.result.expect("evaluate should produce a result");
        win_probs.push(output.white_win_prob);
        score_means.push(output.white_score_mean);
        for y in 0..board.y_size {
            for x in 0..board.x_size {
                let pos = nn_pos::loc_to_pos(
                    kata_game::board::location::get_loc(x, y, board.x_size),
                    board.x_size,
                    nn_eval.nn_x_len(),
                    nn_eval.nn_y_len(),
                );
                policy_probs.push(output.policy_probs[pos as usize]);
            }
        }
    }

    assert_eq!(win_probs.len(), sgf.moves.len());
    assert_eq!(score_means.len(), sgf.moves.len());
    assert!(!policy_probs.is_empty());
}

#[derive(Clone)]
struct NnBatchingTestItem {
    board: Board,
    hist: BoardHistory,
    next_pla: Player,
}

fn random_rules_for_batching(seed: &str, index: usize) -> Rules {
    let mut rand = Rand::new_from_seed(&format!("{seed}_{index}"));
    let mut rules = Rules::get_tromp_taylorish();
    rules.ko_rule = if rand.next_bool(0.5) {
        KoRule::Simple
    } else if rand.next_bool(0.5) {
        KoRule::Positional
    } else {
        KoRule::Situational
    };
    rules.scoring_rule = if rand.next_bool(0.5) {
        ScoringRule::Area
    } else {
        ScoringRule::Territory
    };
    rules.tax_rule = if rand.next_bool(0.5) {
        TaxRule::None
    } else if rand.next_bool(0.5) {
        TaxRule::Seki
    } else {
        TaxRule::All
    };
    rules.multi_stone_suicide_legal = rand.next_bool(0.5);
    rules.has_button = rules.scoring_rule == ScoringRule::Area && rand.next_bool(0.5);
    rules.white_handicap_bonus_rule = if rand.next_bool(0.5) {
        WhiteHandicapBonusRule::Zero
    } else if rand.next_bool(0.5) {
        WhiteHandicapBonusRule::N
    } else {
        WhiteHandicapBonusRule::NMinusOne
    };
    // Komi is stored in half-points.
    rules.komi = 15 + rand.next_i32_range(-10, 10);
    rules
}

fn append_sgf_poses(items: &mut Vec<NnBatchingTestItem>, sgf_str: &str, seed: &str, offset: usize) {
    let sgf = CompactSgf::parse(sgf_str).expect("valid SGF");
    for turn_idx in 0..sgf.moves.len() as i64 {
        let mut board = Board::default();
        let mut next_pla = 0;
        let mut hist = BoardHistory::default();
        let rules = random_rules_for_batching(seed, items.len() + offset);
        sgf.setup_board_and_hist_assume_legal(
            &rules,
            &mut board,
            &mut next_pla,
            &mut hist,
            turn_idx,
        )
        .expect("setup board");
        items.push(NnBatchingTestItem {
            board,
            hist,
            next_pla,
        });
    }
}

fn run_evals_with_shared_nn(
    items: &[NnBatchingTestItem],
    thread_idx: usize,
    num_threads: usize,
    nn_eval: &Arc<Mutex<NnEvaluator>>,
    results: &mut [f64; 4],
) -> [Vec<f64>; 4] {
    let mut policy_results = vec![0.0; items.len()];
    let mut value_results = vec![0.0; items.len()];
    let mut score_results = vec![0.0; items.len()];
    let mut ownership_results = vec![0.0; items.len()];

    let mut rand = Rand::new_from_seed(&format!("runNNBatchingTest{thread_idx}"));
    for (i, item) in items
        .iter()
        .enumerate()
        .skip(thread_idx)
        .step_by(num_threads)
    {
        if rand.next_bool(0.2) {
            thread::yield_now();
        }

        let nn_input_params = MiscNNInputParams {
            draw_equivalent_wins_for_white: rand.next_double(),
            conservative_pass_and_is_root: rand.next_bool(0.5),
            playout_doubling_advantage: rand.next_double_range(-1.0, 1.0),
            symmetry: rand.next_i32_range(0, 7),
            ..MiscNNInputParams::default()
        };

        let mut buf = NNResultBuf::new();
        let board = &item.board;
        {
            let guard = nn_eval.lock().unwrap();
            guard.evaluate(
                board,
                &item.hist,
                item.next_pla,
                &nn_input_params,
                &mut buf,
                true,
                true,
            );
        }
        let output = buf.result.expect("evaluate should produce a result");
        value_results[i] = (output.white_win_prob - output.white_loss_prob) as f64;
        score_results[i] = (output.white_score_mean + output.white_lead) as f64;

        let mut max_policy = 0.0f64;
        for y in 0..board.y_size {
            for x in 0..board.x_size {
                let pos = nn_pos::loc_to_pos(
                    kata_game::board::location::get_loc(x, y, board.x_size),
                    board.x_size,
                    19,
                    19,
                );
                let ownership = output
                    .white_owner_map
                    .as_ref()
                    .map_or(0.0, |m| m[pos as usize]) as f64;
                let policy = output.policy_probs[pos as usize] as f64;
                ownership_results[i] += ownership.abs();
                if policy >= 0.0 && policy > max_policy {
                    max_policy = policy;
                }
            }
        }
        policy_results[i] += max_policy;
    }

    *results = [
        policy_results.iter().sum(),
        value_results.iter().sum(),
        score_results.iter().sum(),
        ownership_results.iter().sum(),
    ];
    [
        policy_results,
        value_results,
        score_results,
        ownership_results,
    ]
}

#[test]
fn nn_batching_consistency_smoke() {
    let (_logger, nn_eval) = test_logger_and_eval(19, 19);
    let nn_eval = Arc::new(Mutex::new(nn_eval));
    nn_eval.lock().unwrap().set_do_randomize(false);

    let mut items: Vec<NnBatchingTestItem> = Vec::new();
    append_sgf_poses(&mut items, SGF_GO_SEIGEN, "batch_a", 0);
    let offset_b = items.len();
    append_sgf_poses(&mut items, SGF_19X10, "batch_b", offset_b);
    // Repeat the same games several times to build up a larger batch.
    let offset_c = items.len();
    append_sgf_poses(&mut items, SGF_GO_SEIGEN, "batch_c", offset_c);
    let offset_d = items.len();
    append_sgf_poses(&mut items, SGF_19X10, "batch_d", offset_d);

    const NUM_THREADS: usize = 4;

    // Single-threaded baseline: run each thread index sequentially in this
    // thread, so every item is evaluated exactly once.
    let mut single_results = [0.0; 4];
    let mut single_per_item: [Vec<f64>; 4] = [
        vec![0.0; items.len()],
        vec![0.0; items.len()],
        vec![0.0; items.len()],
        vec![0.0; items.len()],
    ];
    for thread_idx in 0..NUM_THREADS {
        let mut thread_results = [0.0; 4];
        let thread_per_item = run_evals_with_shared_nn(
            &items,
            thread_idx,
            NUM_THREADS,
            &nn_eval,
            &mut thread_results,
        );
        for i in 0..4 {
            single_results[i] += thread_results[i];
            for (k, v) in thread_per_item[i].iter().enumerate() {
                single_per_item[i][k] += v;
            }
        }
    }

    // Multi-threaded run: each thread index runs concurrently.
    let mut handles = Vec::new();
    for thread_idx in 0..NUM_THREADS {
        let items = items.clone();
        let nn_eval = nn_eval.clone();
        let handle = thread::spawn(move || {
            let mut results = [0.0; 4];
            let per_item =
                run_evals_with_shared_nn(&items, thread_idx, NUM_THREADS, &nn_eval, &mut results);
            (results, per_item)
        });
        handles.push(handle);
    }

    let mut multi_results = [0.0; 4];
    let mut multi_per_item: [Vec<f64>; 4] = [
        vec![0.0; items.len()],
        vec![0.0; items.len()],
        vec![0.0; items.len()],
        vec![0.0; items.len()],
    ];
    for handle in handles {
        let (thread_results, thread_per_item) = handle.join().expect("thread should not panic");
        for i in 0..4 {
            multi_results[i] += thread_results[i];
            for (k, v) in thread_per_item[i].iter().enumerate() {
                multi_per_item[i][k] += v;
            }
        }
    }

    // The dummy evaluator is deterministic, so the totals must match exactly.
    for i in 0..4 {
        assert!(
            (multi_results[i] - single_results[i]).abs() < 1e-9,
            "result {i} totals should match: multi={} single={}",
            multi_results[i],
            single_results[i]
        );
    }

    // Also verify per-item results match the single-threaded baseline.
    for i in 0..items.len() {
        for j in 0..4 {
            assert!(
                (multi_per_item[j][i] - single_per_item[j][i]).abs() < 1e-9,
                "item {i} result {j} should match: multi={} single={}",
                multi_per_item[j][i],
                single_per_item[j][i]
            );
        }
    }
}
