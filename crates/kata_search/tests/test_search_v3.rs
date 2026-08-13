//! Integration tests ported from `KataGo/cpp/tests/testsearchv3.cpp`.
//!
//! These tests use the dummy neural-net evaluator, so they are primarily
//! smoke tests that exercise `AsyncBot` and direct `Search`/`NnEvaluator`
//! APIs on real SGF positions without crashing.  The C++ originals compare
//! against real model outputs; those golden comparisons are skipped here.

mod common;

use std::sync::Arc;

use common::{TestSearchOptions, run_bot_on_position, run_bot_on_sgf, test_logger_and_eval};

use kata_core::logger::{Logger, LoggerOptions};
use kata_game::board::{Board, P_BLACK, P_WHITE, Player, location};
use kata_game::history::BoardHistory;
use kata_game::rules::Rules;
use kata_nn::backend::NNResultBuf;
use kata_nn::inputs::MiscNNInputParams;
use kata_search::async_bot::AsyncBot;
use kata_search::params::SearchParams;
use kata_search::search::PrintTreeOptions;

const SGF_GAME5: &str = "(;FF[4]CA[UTF-8]KM[7.5];B[pp];W[pc];B[cd];W[dq];B[ed];W[pe];B[co];W[cp];B[do];W[fq];B[ck];W[qn];B[qo];W[pn];B[np];W[qj];B[jc];W[lc];B[je];W[lq];B[mq];W[lp];B[ek];W[qq];B[pq];W[ro];B[rp];W[qp];B[po];W[rq];B[rn];W[sp];B[rm];W[ql];B[on];W[om];B[nn];W[nm];B[mn];W[ip];B[mm])";

const SGF_GAME6: &str = "(;FF[4]CA[UTF-8]SZ[11]KM[7.5];B[ci];W[ic];B[ih];W[hi];B[ii];W[ij];B[jj];W[gj];B[ik];W[di];B[hh];W[ch];B[dc];W[cc];B[cb];W[cd];B[eb];W[dd];B[ed];W[ee];B[fd];W[bb];B[ba];W[ab];B[gb];W[je];B[ib];W[jb];B[jc];W[jd];B[hc];W[id];B[dh];W[cg];B[dj];W[ei];B[bi];W[ia];B[hb];W[fg];B[hj];W[eh];B[ej])";

const SGF_GAME7: &str = "(;FF[4]CA[UTF-8]SZ[11]KM[7.5];B[ci];W[ic];B[ih];W[hi];B[ii];W[ij];B[jj];W[gj];B[ik];W[di];B[hh];W[ch];B[dc];W[cc];B[cb];W[cd];B[eb];W[dd];B[ed];W[ee];B[fd];W[bb];B[ba];W[ab];B[gb];W[je];B[ib];W[jb];B[jc];W[jd];B[hc];W[id];B[dh];W[cg];B[dj];W[ei];B[bi];W[ia];B[hb];W[fg];B[hj];W[eh];B[ej];W[fj];B[bh];W[bg];B[fe];W[ef];B[jf];W[kc];B[ke];W[ja];B[if];W[fi];B[gg];W[ek];B[ck];W[bj];B[aj];W[bk];B[ah];W[ag];B[cj];W[he];B[hf];W[hd];B[ff];W[kd];B[kf];W[ha];B[gd];W[ga];B[fa];W[gi];B[hk];W[gh];B[ca];W[gk];B[aa];W[bc];B[ge];W[ig];B[fc];W[ka];B[da];W[jg];B[de];W[ce];B[ak];W[ie];B[dk];W[fk];B[hg];W[dg];B[jh];W[ad])";

const SGF_GAME8: &str = "(;FF[4]CA[UTF-8]SZ[11]KM[7.5];B[ci];W[ic];B[ih];W[hi];B[ii];W[ij];B[jj];W[gj];B[ik];W[di];B[hh];W[ch];B[dc];W[cc];B[cb];W[cd];B[eb];W[dd];B[ed];W[ee];B[fd];W[bb];B[ba];W[ab];B[gb];W[je];B[ib];W[jb];B[jc];W[jd];B[hc];W[id];B[dh];W[cg];B[dj];W[ei];B[bi];W[ia];B[hb];W[fg];B[hj];W[eh];B[ej];W[fj];B[bh];W[bg];B[fe];W[ef];B[jf];W[kc];B[ke];W[ja];B[if];W[fi];B[gg];W[ek];B[ck];W[bj];B[aj];W[bk];B[ah];W[ag];B[cj];W[he];B[hf];W[hd];B[ff];W[kd];B[kf];W[ha];B[gd];W[ga];B[fa];W[gi];B[hk];W[gh];B[ca];W[gk];B[aa];W[bc];B[ge];W[ig];B[fc];W[ka];B[da];W[jg];B[de];W[ce];B[ak];W[hg];B[gf])";

const SGF_GAME9: &str = "(;FF[4]CA[UTF-8]SZ[15]KM[7.5];B[lm];W[lc];B[dm];W[dc];B[me];W[md];B[le];W[jc];B[lk];W[ck];B[cl];W[dk];B[fm];W[de];B[dg];W[fk];B[fg];W[hk];B[hm];W[hg];B[ci];W[cf];B[cg];W[mh];B[kh];W[mj];B[lj];W[mk];B[ml];W[nl];B[nm];W[nk];B[hi];W[gh];B[gi];W[fh];B[fi];W[eh];B[ei];W[dh];B[di];W[ch];B[bh];W[eg];B[bg];W[kg];B[jg];W[kf];B[lh];W[mg];B[lg];W[lf];B[mf];W[ke];B[kd];W[je];B[ld];W[jd];B[mc];W[nd];B[kc];W[lb];B[kb];W[mb];B[ne];W[nc];B[ng];W[nh];B[if];W[jh];B[ih];W[ji];B[li];W[ig];B[ij];W[og];B[ef];W[ff];B[ee];W[df];B[fe];W[gf];B[ec];W[db];B[dd];W[cd];B[ed];W[bf];B[eb];W[he];B[gd];W[hc];B[gc];W[hd];B[gb];W[hb];B[fc];W[fa];B[ea];W[bc];B[ga];W[jj];B[ik];W[jk];B[il];W[jl];B[jm];W[gg];B[ge];W[kl];B[km];W[bl];B[ll];W[cn])";

const SGF_GAME10: &str = "(;GM[1]FF[4]CA[UTF-8]SZ[19]HA[6]KM[0.5]AB[dc][oc][qd][ce][qo][pq];W[cp];B[ep];W[eq];B[fq];W[dq];B[fp];W[dn];B[jq];W[jp];B[ip];W[kq];B[iq];W[kp];B[fm];W[io];B[ho];W[in];B[en];W[dm];B[hn];W[oq];B[op];W[pr];B[pp];W[or];B[qr];W[mq];B[mo];W[qj];B[ql];W[qe];B[rd];W[qg];B[pe];W[ic];B[gc];W[lc];B[ch];W[cj];B[eh];W[ec];B[eb];W[dd];B[ed];W[cc];B[fc];W[db];B[cd];W[ec];B[de];W[dc];B[gb];W[ea];B[fb];W[bb];B[bd];W[ca];B[bc];W[ab];B[ee];W[nc];B[nd];W[ob];B[nb];W[mc];B[pb];W[od];B[pc];W[ne];B[md];W[le];B[oe];W[rl];B[rm];W[rk];B[qm];W[ie];B[me];W[mf];B[nf];W[ld];B[pd];W[ge];B[hd];W[he];B[fd];W[mg];B[id];W[jd];B[hh];W[bi];B[bh];W[ln];B[im];W[jm];B[jl];W[km];B[lo];W[ko];B[il];W[ek];B[dp];W[cq];B[do];W[co];B[fj];W[jh];B[ig];W[jg];B[nm];W[re];B[se];W[rf];B[pj];W[pi];B[oj];W[qk];B[oi];W[ph];B[mb];W[pk];B[ol];W[ok];B[nk];W[nj];B[mj];W[ni];B[mi];W[nh];B[mk];W[er];B[lb];W[kb];B[fr];W[fk];B[ff];W[di];B[ci];W[bj];B[ei];W[dj];B[dh];W[sf];B[jr];W[kr];B[sd];W[qs];B[rr];W[gl];B[gm];W[ib];B[ks];W[ls];B[js];W[np];B[no];W[pl];B[pm];W[if];B[mp];W[mr];B[nq];W[nr];B[gg];W[rs];B[og];W[oh];B[mn];W[ll];B[lh];W[ih];B[hg];W[ml];B[nl];W[gj];B[kl];W[lk];B[gi];W[ej];B[fi];W[hl];B[hj];W[lg];B[gk];W[fl];B[hk];W[em];B[hm];W[sm];B[sn];W[sl];B[sp];W[la];B[kj];W[pf];B[of];W[ii];B[lj];W[lm];B[kh];W[kg];B[fa];W[da];B[jj];W[fs];B[gs];W[es];B[ha];W[ia];B[ij];W[ah];B[ag];W[ai];B[pg];W[qf];B[lp];W[lq];B[hb];W[kk];B[jk];W[ac];B[ad];W[ji];B[ki];W[ka];B[oa];W[ma];B[na];W[sr];B[sq];W[ps];B[ss];W[np];B[sr];W[nq];B[mh];W[ng];B[fe];W[jn];B[mm];W[gr];B[hs];W[fn];B[eo];W[hr];B[is];W[gp];B[go];W[gq];B[hp];W[fo];B[])";

fn evaluate_position_smoke(
    nn_eval: &kata_nn::eval::NnEvaluator,
    board: &Board,
    hist: &BoardHistory,
    next_pla: Player,
) {
    let nn_input_params = MiscNNInputParams::default();
    let mut buf = NNResultBuf::new();
    nn_eval.evaluate(
        board,
        hist,
        next_pla,
        &nn_input_params,
        &mut buf,
        true,
        true,
    );
    let output = buf.result.expect("evaluate should produce a result");
    assert!(output.white_win_prob >= 0.0 && output.white_win_prob <= 1.0);
}

#[test]
fn ownership_and_misc_smoke() {
    let (_logger, nn_eval) = test_logger_and_eval(19, 19);
    let (_logger11, nn_eval11) = test_logger_and_eval(11, 11);
    let (_logger_p, nn_eval_p_temp) = test_logger_and_eval(19, 19);

    // GAME 5
    {
        let sgf = kata_data::sgf::CompactSgf::parse(SGF_GAME5).expect("valid SGF");
        let rules = Rules::get_tromp_taylorish();
        let initial_rules = sgf
            .get_rules_or_fail_allow_unspecified(&rules)
            .expect("rules");
        let mut board = Board::default();
        let mut next_pla = 0;
        let mut hist = BoardHistory::default();
        sgf.setup_board_and_hist_assume_legal(
            &initial_rules,
            &mut board,
            &mut next_pla,
            &mut hist,
            40,
        )
        .expect("setup board");

        evaluate_position_smoke(&nn_eval, &board, &hist, next_pla);

        let nn_input_params = MiscNNInputParams {
            nn_policy_temperature: 1.5,
            ..MiscNNInputParams::default()
        };
        let mut buf = NNResultBuf::new();
        nn_eval_p_temp.evaluate(
            &board,
            &hist,
            next_pla,
            &nn_input_params,
            &mut buf,
            true,
            true,
        );
        assert!(buf.result.is_some());

        let params = SearchParams {
            max_visits: 200,
            value_weight_exponent: 0.0,
            ..SearchParams::default()
        };
        let logger = Arc::new(kata_core::logger::Logger::new(
            kata_core::logger::LoggerOptions::default(),
            None,
        ));
        let mut bot = AsyncBot::new(params, &nn_eval, &logger, "seed");
        let opts = TestSearchOptions {
            print_ownership: true,
            ..Default::default()
        };
        run_bot_on_sgf(&mut bot, SGF_GAME5, &rules, 40, 7.5, opts);
    }

    // GAME 6
    {
        let sgf = kata_data::sgf::CompactSgf::parse(SGF_GAME6).expect("valid SGF");
        let rules = Rules::get_tromp_taylorish();
        let initial_rules = sgf
            .get_rules_or_fail_allow_unspecified(&rules)
            .expect("rules");
        let mut board = Board::default();
        let mut next_pla = 0;
        let mut hist = BoardHistory::default();
        sgf.setup_board_and_hist_assume_legal(
            &initial_rules,
            &mut board,
            &mut next_pla,
            &mut hist,
            43,
        )
        .expect("setup board");

        evaluate_position_smoke(&nn_eval, &board, &hist, next_pla);

        let mut buf11 = NNResultBuf::new();
        let nn_input_params = MiscNNInputParams::default();
        nn_eval11.evaluate(
            &board,
            &hist,
            next_pla,
            &nn_input_params,
            &mut buf11,
            true,
            true,
        );
        let output11 = buf11.result.expect("evaluate should produce a result");
        assert_eq!(output11.nn_x_len, 11);
        assert_eq!(output11.nn_y_len, 11);
    }

    // GAME 7
    {
        let rules = Rules::get_tromp_taylorish();
        let params = SearchParams {
            max_visits: 500,
            fpu_reduction_max: 0.0,
            root_fpu_reduction_max: 0.0,
            value_weight_exponent: 0.0,
            ..SearchParams::default()
        };
        let logger = Arc::new(kata_core::logger::Logger::new(
            kata_core::logger::LoggerOptions::default(),
            None,
        ));
        let mut bot = AsyncBot::new(params.clone(), &nn_eval, &logger, "testSearchSeed0");
        let opts = TestSearchOptions {
            print_ending_score_value_bonus: true,
            no_clear_bot: true,
            ..Default::default()
        };

        let mut params2 = params.clone();
        params2.root_ending_bonus_points = 0.5;
        bot.set_params(&params2);
        run_bot_on_sgf(&mut bot, SGF_GAME7, &rules, 88, 7.5, opts);

        bot.set_params(&params2);
        run_bot_on_sgf(&mut bot, SGF_GAME7, &rules, 89, 7.5, opts);

        bot.set_params(&params);
        run_bot_on_sgf(&mut bot, SGF_GAME7, &rules, 96, 7.5, opts);

        bot.set_params(&params2);
        run_bot_on_sgf(&mut bot, SGF_GAME7, &rules, 96, 7.5, opts);

        bot.clear_search();
    }

    // GAME 8
    {
        let rules = Rules::get_simple_territory();
        let params = SearchParams {
            max_visits: 500,
            fpu_reduction_max: 0.0,
            root_fpu_reduction_max: 0.0,
            value_weight_exponent: 0.0,
            ..SearchParams::default()
        };
        let logger = Arc::new(kata_core::logger::Logger::new(
            kata_core::logger::LoggerOptions::default(),
            None,
        ));
        let mut bot = AsyncBot::new(params.clone(), &nn_eval, &logger, "testSearchSeed1");
        let opts = TestSearchOptions {
            print_ending_score_value_bonus: true,
            no_clear_bot: true,
            ..Default::default()
        };

        bot.set_params(&params);
        run_bot_on_sgf(&mut bot, SGF_GAME8, &rules, 91, 7.5, opts);

        let mut params2 = params.clone();
        params2.root_ending_bonus_points = 0.5;
        bot.set_params(&params2);
        run_bot_on_sgf(&mut bot, SGF_GAME8, &rules, 91, 7.5, opts);

        bot.clear_search();
    }

    // GAME 9
    {
        let rules = Rules::get_simple_territory();
        let params = SearchParams {
            max_visits: 1,
            value_weight_exponent: 0.0,
            ..SearchParams::default()
        };
        let logger = Arc::new(kata_core::logger::Logger::new(
            kata_core::logger::LoggerOptions::default(),
            None,
        ));
        let mut bot = AsyncBot::new(params.clone(), &nn_eval, &logger, "testSearchSeed2");
        let opts = TestSearchOptions {
            print_root_policy: true,
            ..Default::default()
        };
        run_bot_on_sgf(&mut bot, SGF_GAME9, &rules, 114, 6.5, opts);

        let mut test_params = params.clone();
        test_params.root_noise_enabled = true;
        bot.set_params(&test_params);
        run_bot_on_sgf(&mut bot, SGF_GAME9, &rules, 114, 6.5, opts);
        bot.set_params(&params);
        bot.clear_search();
    }
}

#[test]
fn lcb_and_endgame_seki_smoke() {
    let (_logger, nn_eval) = test_logger_and_eval(19, 19);

    let rules = Rules::get_tromp_taylorish();
    let params = SearchParams {
        max_visits: 280,
        static_score_utility_factor: 0.2,
        dynamic_score_utility_factor: 0.3,
        use_lcb_for_selection: true,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let logger = Arc::new(Logger::new(LoggerOptions::default(), None));
    let mut bot = AsyncBot::new(params.clone(), &nn_eval, &logger, "testSearchSeed3");

    let opts = TestSearchOptions {
        no_clear_bot: true,
        ..Default::default()
    };
    run_bot_on_sgf(&mut bot, SGF_GAME10, &rules, 234, 0.5, opts);

    {
        let search = bot.get_search_stop_and_wait();
        let options = PrintTreeOptions::new().max_depth(1);
        search.begin_search(false);
        let mut out = String::new();
        let root = search.root_node.as_deref().unwrap();
        search.print_tree(&mut out, Some(root), &options, P_WHITE);
        assert!(!out.is_empty());

        let o3 = location::of_string("O3", 19, 19).expect("valid loc");
        assert!(bot.make_move(o3, P_WHITE));
        let search = bot.get_search_stop_and_wait();
        if search.root_node.is_none() {
            search.begin_search(false);
        }
        let root = search.root_node.as_deref().unwrap();
        let mut out = String::new();
        search.print_tree(&mut out, Some(root), &options, P_WHITE);
        assert!(!out.is_empty());

        search.begin_search(false);
        let root = search.root_node.as_deref().unwrap();
        let mut out = String::new();
        search.print_tree(&mut out, Some(root), &options, P_WHITE);
        assert!(!out.is_empty());
    }

    bot.clear_search();
}

#[test]
fn non_square_board_smoke() {
    let (_logger, nn_eval) = test_logger_and_eval(19, 19);
    let (_logger11, nn_eval11) = test_logger_and_eval(11, 11);

    let rules = Rules::get_tromp_taylorish();
    let next_pla = P_BLACK;

    let board_a = Board::parse_board(
        7,
        11,
        ".......\n.......\n..x.o..\n.......\n.......\n...xo..\n.......\n..xx...\n..oox..\n....o..\n.......\n",
        '\n',
    )
    .expect("valid board");
    let hist_a = BoardHistory::new(board_a.clone(), next_pla, rules, 0);

    let board_b = Board::parse_board(
        11,
        7,
        "...........\n...........\n..x.o.ox...\n.......ox..\n.......ox..\n...........\n...........\n",
        '\n',
    )
    .expect("valid board");
    let hist_b = BoardHistory::new(board_b.clone(), next_pla, rules, 0);

    let params = SearchParams {
        max_visits: 200,
        dynamic_score_utility_factor: 0.25,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let logger = Arc::new(Logger::new(LoggerOptions::default(), None));

    {
        let mut bot_a = AsyncBot::new(params.clone(), &nn_eval, &logger, "testSearchSeed4");
        let mut bot_b = AsyncBot::new(params.clone(), &nn_eval, &logger, "testSearchSeed5");
        run_bot_on_position(
            &mut bot_a,
            board_a.clone(),
            next_pla,
            hist_a.clone(),
            TestSearchOptions::default(),
        );
        run_bot_on_position(
            &mut bot_b,
            board_b.clone(),
            next_pla,
            hist_b.clone(),
            TestSearchOptions::default(),
        );
    }

    {
        let mut bot_a11 = AsyncBot::new(params.clone(), &nn_eval11, &logger, "testSearchSeed6");
        let mut bot_b11 = AsyncBot::new(params.clone(), &nn_eval11, &logger, "testSearchSeed7");
        run_bot_on_position(
            &mut bot_a11,
            board_a,
            next_pla,
            hist_a,
            TestSearchOptions::default(),
        );
        run_bot_on_position(
            &mut bot_b11,
            board_b,
            next_pla,
            hist_b,
            TestSearchOptions::default(),
        );
    }
}

#[test]
fn multi_stone_suicide_smoke() {
    let (_logger, nn_eval) = test_logger_and_eval(19, 19);

    let seed = "testSearchSeed8";
    for i in 0..2 {
        let mut rules = Rules::get_tromp_taylorish();
        rules.set_komi(0.5);
        if i == 1 {
            rules.multi_stone_suicide_legal = false;
        }

        let next_pla = P_WHITE;
        let mut board = Board::parse_board(
            9,
            9,
            "..ox..xx.\n.ooxxxx.x\no..o..oxo\n.oooooooo\n.xxxxxxxx\n....x.x..\n.x.x.x.oo\n....x.oox\n......ox.\n",
            '\n',
        )
        .expect("valid board");
        let mut hist = BoardHistory::new(board.clone(), next_pla, rules, 0);
        let h8 = location::of_string("H8", 9, 9).expect("valid loc");
        hist.make_board_move_assume_legal(&mut board, h8, next_pla);
        let next_pla = P_BLACK;

        let params = SearchParams {
            max_visits: 200,
            value_weight_exponent: 0.0,
            ..SearchParams::default()
        };
        let logger = Arc::new(Logger::new(LoggerOptions::default(), None));
        let mut bot = AsyncBot::new(params, &nn_eval, &logger, seed);
        run_bot_on_position(
            &mut bot,
            board,
            next_pla,
            hist,
            TestSearchOptions::default(),
        );
    }
}

#[test]
fn conservative_pass_smoke() {
    let (_logger, nn_eval) = test_logger_and_eval(19, 19);

    let seed = "abc";
    let mut rules = Rules::get_tromp_taylorish();
    rules.set_komi(0.0);

    let next_pla = P_BLACK;
    let mut board = Board::parse_board(
        9,
        9,
        ".........\n..x...x..\n.........\nxxxxxxxx.\nooooooooo\n...o.o.o.\nxx.o.o.o.\n.xxo.o.o.\n..xo.o.o.\n",
        '\n',
    )
    .expect("valid board");
    let mut hist = BoardHistory::new(board.clone(), next_pla, rules, 0);
    hist.make_board_move_assume_legal(&mut board, kata_game::board::PASS_LOC, next_pla);
    let next_pla = P_WHITE;

    {
        let params = SearchParams {
            max_visits: 80,
            root_fpu_reduction_max: 0.0,
            root_policy_temperature: 1.5,
            root_policy_temperature_early: 1.5,
            root_noise_enabled: true,
            value_weight_exponent: 0.0,
            ..SearchParams::default()
        };
        let logger = Arc::new(Logger::new(LoggerOptions::default(), None));
        let mut bot = AsyncBot::new(params, &nn_eval, &logger, seed);
        run_bot_on_position(
            &mut bot,
            board.clone(),
            next_pla,
            hist.clone(),
            TestSearchOptions::default(),
        );
    }

    {
        let params = SearchParams {
            max_visits: 80,
            conservative_pass: true,
            root_fpu_reduction_max: 0.0,
            root_policy_temperature: 1.5,
            root_policy_temperature_early: 1.5,
            root_noise_enabled: true,
            value_weight_exponent: 0.0,
            ..SearchParams::default()
        };
        let logger = Arc::new(Logger::new(LoggerOptions::default(), None));
        let mut bot = AsyncBot::new(params, &nn_eval, &logger, seed);
        run_bot_on_position(
            &mut bot,
            board,
            next_pla,
            hist,
            TestSearchOptions::default(),
        );
    }
}

#[test]
fn root_noise_and_temperature_across_moves_smoke() {
    let (_logger, nn_eval) = test_logger_and_eval(19, 19);

    let seed = "testSearchSeed9";
    let mut rules = Rules::get_tromp_taylorish();
    rules.set_komi(5.5);

    let next_pla = P_WHITE;
    let board = Board::parse_board(
        9,
        9,
        ".........\n.........\n....o....\n..x......\n....x.x..\n..xo.....\n.....o...\n.........\n.........\n",
        '\n',
    )
    .expect("valid board");
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    let params = SearchParams {
        max_visits: 200,
        root_policy_temperature: 2.5,
        root_policy_temperature_early: 2.5,
        root_noise_enabled: true,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let logger = Arc::new(Logger::new(LoggerOptions::default(), None));
    let mut bot = AsyncBot::new(params, &nn_eval, &logger, seed);
    bot.set_always_include_owner_map(true);

    let opts = TestSearchOptions {
        no_clear_bot: true,
        ..Default::default()
    };
    run_bot_on_position(&mut bot, board.clone(), next_pla, hist.clone(), opts);

    let d5 = location::of_string("D5", 9, 9).expect("valid loc");
    {
        let search = bot.get_search();
        let options = PrintTreeOptions::new().only_branch(&board, "D5");
        let root = search.root_node.as_deref().unwrap();
        let mut out = String::new();
        search.print_tree(&mut out, Some(root), &options, P_WHITE);
        assert!(!out.is_empty());
    }
    assert!(bot.make_move(d5, next_pla));
    let mut board = board;
    let mut hist = hist;
    hist.make_board_move_assume_legal(&mut board, d5, next_pla);
    let next_pla = kata_game::board::get_opp(next_pla);

    let opts = TestSearchOptions {
        no_clear_bot: true,
        ignore_position: true,
        ..Default::default()
    };
    run_bot_on_position(&mut bot, board, next_pla, hist, opts);

    bot.clear_search();
}

#[test]
fn japanese_rules_endgame_smoke() {
    let (_logger, nn_eval) = test_logger_and_eval(19, 19);

    let seed = "testSearchSeed10";
    let rules = Rules::parse_rules("Japanese").expect("valid rules");

    let next_pla = P_BLACK;
    let mut board = Board::parse_board(
        9,
        7,
        ".........\nooooo.o..\noxxxox...\nxx..xoooo\n..xx.x.xo\n.oox.xxxx\n..x...ox.\n",
        '\n',
    )
    .expect("valid board");
    board.num_white_captures = 3;
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    evaluate_position_smoke(&nn_eval, &board, &hist, next_pla);

    let params = SearchParams {
        max_visits: 200,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let logger = Arc::new(Logger::new(LoggerOptions::default(), None));
    let mut bot = AsyncBot::new(params, &nn_eval, &logger, seed);

    let opts = TestSearchOptions {
        no_clear_bot: true,
        ..Default::default()
    };
    run_bot_on_position(&mut bot, board.clone(), next_pla, hist.clone(), opts);

    {
        let search = bot.get_search();
        let options = PrintTreeOptions::new().only_branch(&board, "G3");
        let root = search.root_node.as_deref().unwrap();
        let mut out = String::new();
        search.print_tree(&mut out, Some(root), &options, P_BLACK);
        assert!(!out.is_empty());
    }

    bot.clear_search();
}

#[test]
fn chinese_rules_endgame_smoke() {
    let (_logger, nn_eval) = test_logger_and_eval(19, 19);

    let seed = "testSearchSeed11";
    let rules = Rules::parse_rules("Chinese").expect("valid rules");

    let next_pla = P_BLACK;
    let mut board = Board::parse_board(
        9,
        7,
        ".........\nooooo.o..\noxxxox...\nxx..xoooo\n..xx.x.xo\n.oox.xxxx\n..x...ox.\n",
        '\n',
    )
    .expect("valid board");
    board.num_white_captures = 3;
    let hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

    evaluate_position_smoke(&nn_eval, &board, &hist, next_pla);

    let params = SearchParams {
        max_visits: 200,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let logger = Arc::new(Logger::new(LoggerOptions::default(), None));
    let mut bot = AsyncBot::new(params, &nn_eval, &logger, seed);

    let opts = TestSearchOptions {
        no_clear_bot: true,
        ..Default::default()
    };
    run_bot_on_position(&mut bot, board.clone(), next_pla, hist.clone(), opts);

    {
        let search = bot.get_search();
        let options = PrintTreeOptions::new().only_branch(&board, "G3");
        let root = search.root_node.as_deref().unwrap();
        let mut out = String::new();
        search.print_tree(&mut out, Some(root), &options, P_BLACK);
        assert!(!out.is_empty());
    }

    bot.clear_search();
}
