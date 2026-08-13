//! Integration tests ported from `KataGo/cpp/tests/testsearch.cpp`.
//!
//! These tests use the dummy neural-net evaluator, so they are primarily
//! smoke tests that exercise `AsyncBot` and `Search` on real SGF positions
//! without crashing, rather than golden-output comparisons against the C++
//! tests (which require a real model).

mod common;

use kata_game::rules::Rules;
use kata_search::async_bot::AsyncBot;
use kata_search::params::SearchParams;

use common::{TestSearchOptions, run_bot_on_sgf, test_logger_and_eval};

const GAME1: &str = concat!(
    "(;SZ[19]FF[3]PW[An Seong-chun]WR[6d]PB[Chen Yaoye]BR[9d]DT[2016-07-02]KM[7.5]RU[Chinese]RE[B+R];B[qd];W[dc];B[pq];W[dp];B[nc];W[po];B[qo];W[qn];B[qp];W[pm];B[nq];W[qi];B[qg];W[oi];B[cn];W[ck];B[fp];W[co];B[dn];W[eo];B[cq];W[dq];B[bo];W[cp];B[bp];W[bq];B[fn];W[bm];B[bn];W[fo];B[go];W[cr];B[en];W[gn];B[ho];W[gm];B[er];W[dr];B[ek];W[di];B[in];W[gk];B[cl];W[dk];B[ej];W[dl];B[el];W[gi];B[fi];W[ch];B[gh];W[hi];B[hh];W[ii];B[eh];W[df];B[ih];W[ji];B[kg];W[fg];B[ff];W[gf];B[eg];W[ef];B[fe];W[ge];B[fd];W[",
    "gg];B[fh];W[gd];B[cg];W[dg];B[dh];W[bg];B[bh];W[cf];B[ci];W[qc];B[pc];W[mp];B[on];W[mn];B[om];W[iq];B[pn];W[ol];B[qm];W[pl];B[rn];W[gq];B[kn];W[jo];B[ko];W[jp];B[jn];W[li];B[mo];W[pb];B[rc];W[oc];B[qb];W[od];B[cg];W[pd];B[dd];W[fc];B[ec];W[eb];B[ed];W[cd];B[fb];W[gc];B[db];W[cc];B[ea];W[gb];B[cb];W[bb];B[be];W[ce];B[bf];W[bd];B[ag];W[ca];B[jc];W[qe];B[ep];W[do];B[gp];W[fr];B[qc];W[nb];B[ib];W[je];B[re];W[kd];B[ba];W[aa];B[lc];W[ha];B[ld];W[le];B[me];W[mb];B[ie];W[id];B[kc];W[if];B[lf];W[ke];B[nd",
    "];W[of];B[jh];W[qf];B[rf];W[pg];B[mh];W[mq];B[mi];W[mj];B[hl];W[kh];B[jf];W[gl];B[lo];W[np];B[nr];W[kq];B[no];W[he];B[mf];W[rg];B[kk];W[jk];B[kj];W[ki];B[kl];W[lj];B[qk];W[ml];B[pa];W[ob];B[hb];W[ga];B[op];W[mr];B[ms];W[ls];B[ns];W[lq];B[pj];W[oj];B[ng];W[qh];B[eq];W[es];B[rj];W[im];B[jj];W[ik];B[jl];W[il];B[hn];W[hm];B[nm];W[mm];B[nl];W[nk];B[sf];W[ri];B[ql];W[ok];B[qj];W[lb];B[hq];W[hr];B[hp])"
);

const GAME2: &str = concat!(
    "(;SZ[19]FF[3]PW[Go Seigen]WR[9d]PB[Takagawa Shukaku]BR[8d]DT[1957-09-26]KM[0]RE[W+R];B[qd];W[dc];B[pp];W[cp];B[eq];W[oc];B[ce];W[dh];B[fe];W[gc];B[do];W[co];B[dn];W[cm];B[jq];W[qn];B[pn];W[pm];B[on];W[qq];B[qo];W[or];B[mr];W[mq];B[nr];W[oq];B[lq];W[qm];B[rp];W[rq];B[qg];W[mp];B[lp];W[mo];B[om];W[pk];B[kn];W[mm];B[ok];W[pj];B[mk];W[op];B[dm];W[cl];B[dl];W[dk];B[ek];W[ll];B[cn];W[bn];B[bo];W[bm];B[cq];W[bp];B[oj];W[ph];B[qh];W[oi];B[qi];W[pi];B[mi];W[of];B[ki];W[qc];B[rc];W[qe];B[re];W[pd];B[rd];W",
    "[de];B[df];W[cd];B[ee];W[dd];B[fg];W[hd];B[jl];W[dj];B[bf];W[fj];B[hg];W[dp];B[ep];W[jk];B[il];W[fk];B[ie];W[he];B[hf];W[gm];B[ke];W[fo];B[eo];W[in];B[ho];W[hn];B[fn];W[gn];B[go];W[io];B[ip];W[jp];B[hq];W[qf];B[rf];W[qb];B[ik];W[lr];B[id];W[kr];B[jr];W[bq];B[ib];W[hb];B[cr];W[rj];B[rb];W[kk];B[ij];W[ic];B[jc];W[jb];B[hc];W[iq];B[ir];W[ic];B[kq];W[kc];B[hc];W[nj];B[nk];W[ic];B[oe];W[jd];B[pe];W[pf];B[od];W[pc];B[md];W[mc];B[me];W[ld];B[ng];W[ri];B[rh];W[pg];B[fl];W[je];B[kg];W[be];B[cf];W[bh];B[b",
    "d];W[bc];B[ae];W[kl];B[rn];W[mj];B[lj];W[ni];B[lk];W[mh];B[li];W[mg];B[mf];W[nh];B[jf];W[qj];B[sh];W[rm];B[km];W[if];B[ig];W[dq];B[dr];W[br];B[ci];W[gi];B[ei];W[ej];B[di];W[gl];B[bi];W[cj];B[sq];W[sr];B[so];W[sp];B[fc];W[fb];B[sq];W[lo];B[rr];W[sp];B[ec];W[eb];B[sq];W[ko];B[jn];W[sp];B[nc];W[nb];B[sq];W[nd];B[jo];W[sp];B[qr];W[pq];B[sq];W[ns];B[ks];W[sp];B[bk];W[bj];B[sq];W[ol];B[nl];W[sp];B[aj];W[ck];B[sq];W[nq];B[ls];W[sp];B[gk];W[qp];B[po];W[ro];B[gj];W[eh];B[rp];W[fi];B[sq];W[pl];B[nm];W[sp]",
    ";B[ch];W[ro];B[dg];W[sn];B[ne];W[er];B[fr];W[cs];B[es];W[fh];B[bb];W[cb];B[ac];W[ba];B[cc];W[el];B[fm];W[bc])"
);

const GAME3: &str = concat!(
    "(;FF[4]GM[1]SZ[19]PB[v49-140-400v-fp16]PW[v49-140-400v-fp16-fpu25]HA[0]KM[7.5]RU[koPOSITIONALscoreAREAsui1]RE[W+0.5];B[qd];W[dp];B[cq];W[dq];B[cp];W[co];B[bo];W[bn];B[cn];W[do];B[bm];W[bp];B[an];W[bq];B[cd];W[qp];B[oq];W[pn];B[nd];W[ec];B[df];W[hc];B[jc];W[cb];B[lq];W[ch];B[cj];W[eh];B[gd];W[gc];B[fd];W[hd];B[gf];W[cl];B[dn];W[el];B[eo];W[fp];B[ej];W[bl];B[bk];W[al];B[cr];W[br];B[fi];W[gl];B[gn];W[gp];B[dk];W[dl];B[fm];W[fl];B[ho];W[iq];B[ip];W[jq];B[jp];W[hp];B[in];W[fh];B[gh];W[gg];B[gi];W[hg]",
    ";B[fg];W[hf];B[eg];W[il];B[ii];W[kl];B[lo];W[jj];B[ql];W[pq];B[op];W[rm];B[ji];W[ki];B[kh];W[li];B[ij];W[gm];B[dc];W[eb];B[fn];W[jk];B[lk];W[ln];B[ll];W[km];B[mn];W[ko];B[kp];W[mo];B[lp];W[mm];B[nn];W[lm];B[on];W[nk];B[qn];W[qm];B[po];W[rn];B[pm];W[ed];B[cf];W[ni];B[rq];W[rp];B[pr];W[qq];B[qr];W[rr];B[rs];W[sr];B[lc];W[rd];B[rc];W[re];B[pd];W[rb];B[qc];W[qg];B[oj];W[ok];B[pk];W[sc];B[qb];W[bc];B[qh];W[ph];B[qi];W[og];B[kr];W[ff];B[ef];W[fe];B[bd];W[rg];B[oi];W[nh];B[pf];W[pg];B[is];W[hr];B[hs];W",
    "[gs];B[gr];W[js];B[fs];W[ir];B[ep];W[eq];B[fq];W[cs];B[er];W[dr];B[fo];W[gq];B[go];W[fr];B[ie];W[he];B[fq];W[hq];B[ib];W[bj];B[bi];W[aj];B[ai];W[ak];B[ci];W[rl];B[ee];W[ge];B[rk];W[ol];B[pl];W[mf];B[nf];W[of];B[ne];W[mg];B[qf];W[rf];B[hb];W[ad];B[jr];W[gs];B[hs];W[es];B[ds];W[fr];B[cc];W[bb];B[fq];W[es];B[ae];W[ac];B[ds];W[fr];B[lh];W[mh];B[fq];W[es];B[qo];W[ro];B[ds];W[fr];B[nj];W[mj];B[fq];W[es];B[bf];W[pi];B[pj];W[ri];B[qj];W[rh];B[gb];W[fb];B[ra];W[sb];B[le];W[me];B[md];W[rj];B[jf];W[sk];B[k",
    "s];W[fr];B[lf];W[if];B[id];W[ic];B[je];W[em];B[en];W[ck];B[ga];W[ek];B[dj];W[is];B[kq];W[gs];B[nm];W[nl];B[hk];W[im];B[jn];W[kn];B[gk];W[fk];B[db];W[da];B[fa];W[ea];B[jm];W[jl];B[ih];W[ig];B[jg];W[lg];B[kg];W[oe];B[od];W[dd];B[fj];W[ce];B[be];W[de];B[hh];W[jo];B[io];W[om];B[ap];W[aq];B[ao];W[qk];B[pn];W[am];B[bn];W[pp];B[rk];W[qe];B[pe];W[qk];B[sd];W[se];B[rk];W[qa];B[pa];W[qk];B[hl];W[hm];B[rk];W[jb];B[kb];W[qk];B[qs];W[rk];B[or];W[oh];B[ss];W[sq];B[ng];W[ik];B[hn];W[dm];B[cm];W[sd];B[qa];W[sa]",
    ";B[kj];W[mk];B[ml];W[mi];B[ja];W[lj];B[dh];W[kk];B[ei];W[fc];B[bh];W[];B[cg];W[];B[no];W[];B[mp];W[];B[])"
);

const GAME4: &str = concat!(
    "(;SZ[19]FF[3]PW[Gu Li]WR[9d]PB[Ke Jie]BR[9d]DT[2015-07-19]KM[7.5]RU[Chinese]RE[B+R];B[qe];W[dd];B[op];W[dp];B[fc];W[cf];B[jd];W[pc];B[nc];W[nd];B[mc];W[pe];B[pf];W[qd];B[re];W[oe];B[rd];W[rc];B[qb];W[qc];B[qj];W[md];B[ld];W[lc];B[lb];W[kc];B[kb];W[qp];B[qq];W[rq];B[pq];W[ro];B[oc];W[pb];B[le];W[kq];B[fq];W[eq];B[fp];W[dn];B[iq];W[ko];B[io];W[mq];B[pm];W[qn];B[mo];W[oo];B[lp];W[np];B[no];W[oq];B[pp];W[po];B[or];W[lq];B[nq];W[mp];B[mr];W[mm];B[cc];W[dc];B[db];W[eb];B[cb];W[ec];B[be];W[bf];B[fb];W[",
    "fa];B[ce];W[de];B[df];W[ae];B[cd];W[fd];B[ad];W[ch];B[ef];W[hc];B[ib];W[gf];B[eh];W[cj];B[ga];W[ea];B[gb];W[hb];B[ha];W[ee];B[ff];W[fe];B[he];W[ge];B[gh];W[ie];B[cl];W[dk];B[cq];W[cp];B[er];W[dr];B[dq];W[ep];B[cr];W[fo];B[go];W[fn];B[fk];W[dl];B[ln];W[jn];B[lm];W[lo];B[ml];W[fr];B[ds];W[hq];B[gr];W[hr];B[ir];W[gp];B[fs];W[ho];B[in];W[im];B[nm];W[jl];B[ii];W[ig];B[of];W[nf];B[mf];W[ng];B[od];W[ne];B[pd];W[qg];B[rg];W[ph];B[qi];W[lh];B[jp];W[kp];B[hm];W[hn];B[qf];W[kj];B[gm];W[gn];B[ik];W[rb];B[jk",
    "];W[oj];B[il];W[jm];B[jr];W[lr];B[ms];W[ip];B[kk];W[jo];B[kg];W[ol];B[nk];W[ok];B[om];W[mj];B[sc];W[qk];B[rk];W[ql];B[sb];W[lk];B[kl];W[af];B[rl];W[gc];B[ia];W[id];B[jc];W[rm];B[ab];W[fl];B[gk];W[bq];B[mg];W[ll];B[km];W[nh];B[nl];W[sl];B[rj];W[rh];B[qh];W[pg];B[sf];W[qm];B[br];W[bp];B[di];W[bi];B[ek];W[dj];B[ej])"
);

#[test]
fn basic_positions() {
    let (logger, nn_eval) = test_logger_and_eval(19, 19);

    let params = SearchParams {
        max_visits: 200,
        value_weight_exponent: 0.0,
        ..SearchParams::default()
    };
    let mut bot = AsyncBot::new(params.clone(), &nn_eval, &logger, "testSearchSeed0");

    let rules = Rules::get_tromp_taylorish();

    // Game 1: ordinary pro game
    for turn_idx in [20, 40, 61, 82, 103] {
        run_bot_on_sgf(
            &mut bot,
            GAME1,
            &rules,
            turn_idx,
            7.5,
            TestSearchOptions::default(),
        );
    }

    // Game 2: another ordinary pro game
    for turn_idx in [23, 38, 65, 80, 115] {
        run_bot_on_sgf(
            &mut bot,
            GAME2,
            &rules,
            turn_idx,
            0.0,
            TestSearchOptions::default(),
        );
    }

    // Game 3: extremely close botvbot game
    for turn_idx in [191, 197, 330] {
        run_bot_on_sgf(
            &mut bot,
            GAME3,
            &rules,
            turn_idx,
            7.5,
            TestSearchOptions::default(),
        );
    }
    {
        let mut draw_params = params.clone();
        draw_params.draw_equivalent_wins_for_white = 0.7;
        bot.set_params(&draw_params);
        run_bot_on_sgf(
            &mut bot,
            GAME3,
            &rules,
            330,
            7.5,
            TestSearchOptions::default(),
        );
        run_bot_on_sgf(
            &mut bot,
            GAME3,
            &rules,
            330,
            7.0,
            TestSearchOptions::default(),
        );
        bot.set_params(&params);
    }
    {
        let opts = TestSearchOptions {
            num_moves_in_a_row: 3,
            print_after_begun: true,
            ..Default::default()
        };
        run_bot_on_sgf(&mut bot, GAME3, &rules, 85, 7.5, opts);
    }
    {
        let mut playout_params = params.clone();
        playout_params.max_playouts = 200;
        playout_params.max_visits = 10_000;
        bot.set_params(&playout_params);
        let opts = TestSearchOptions {
            num_moves_in_a_row: 3,
            print_after_begun: true,
            ..Default::default()
        };
        run_bot_on_sgf(&mut bot, GAME3, &rules, 85, 7.5, opts);
        bot.set_params(&params);
    }

    // Game 4: another pro game, with root noise / temperature variations
    run_bot_on_sgf(
        &mut bot,
        GAME4,
        &rules,
        44,
        7.5,
        TestSearchOptions::default(),
    );
    {
        let mut p = params.clone();
        p.root_noise_enabled = true;
        p.root_fpu_reduction_max = 0.0;
        bot.set_params(&p);
        run_bot_on_sgf(
            &mut bot,
            GAME4,
            &rules,
            44,
            7.5,
            TestSearchOptions::default(),
        );
        bot.set_params(&params);
    }
    {
        let mut p = params.clone();
        p.root_policy_temperature = 1.5;
        p.root_policy_temperature_early = 1.5;
        bot.set_params(&p);
        run_bot_on_sgf(
            &mut bot,
            GAME4,
            &rules,
            44,
            7.5,
            TestSearchOptions::default(),
        );
        bot.set_params(&params);
    }
    {
        let opts = TestSearchOptions {
            print_play_selection_values: true,
            ..Default::default()
        };
        let mut p = params.clone();
        p.root_noise_enabled = true;
        p.max_visits = 400;
        p.root_fpu_reduction_max = 0.0;
        p.root_desired_per_child_visits_coeff = 1.0;
        bot.set_params(&p);
        run_bot_on_sgf(&mut bot, GAME4, &rules, 44, 7.5, opts);
        bot.set_params(&params);
    }
    {
        let opts = TestSearchOptions {
            print_play_selection_values: true,
            ..Default::default()
        };
        let mut p = params.clone();
        p.root_noise_enabled = true;
        p.max_visits = 400;
        p.root_fpu_reduction_max = 0.0;
        p.root_desired_per_child_visits_coeff = 9.0;
        bot.set_params(&p);
        run_bot_on_sgf(&mut bot, GAME4, &rules, 44, 7.5, opts);
        bot.set_params(&params);
    }
}
