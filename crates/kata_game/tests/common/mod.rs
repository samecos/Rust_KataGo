//! Common test helpers ported from `cpp/tests/testcommon.cpp`.
//!
//! Provides board comparison and reusable SGF fixtures for game-logic tests.

#![allow(dead_code)]

use kata_game::board::{Board, MAX_ARR_SIZE};

/// Returns true if two boards have the same colors and capture counts.
///
/// Corresponds to `TestCommon::boardsSeemEqual`.
pub fn boards_seem_equal(b1: &Board, b2: &Board) -> bool {
    for i in 0..MAX_ARR_SIZE {
        if b1.colors[i] != b2.colors[i] {
            return false;
        }
    }
    if b1.num_black_captures != b2.num_black_captures {
        return false;
    }
    if b1.num_white_captures != b2.num_white_captures {
        return false;
    }
    true
}

/// Returns benchmark SGF data for the requested board size.
///
/// Corresponds to `TestCommon::getBenchmarkSGFData`. Supported sizes are 7-19.
pub fn get_benchmark_sgf_data(board_size: i32) -> &'static str {
    match board_size {
        19 => BENCHMARK_SGF_19,
        18 => BENCHMARK_SGF_18,
        17 => BENCHMARK_SGF_17,
        16 => BENCHMARK_SGF_16,
        15 => BENCHMARK_SGF_15,
        14 => BENCHMARK_SGF_14,
        13 => BENCHMARK_SGF_13,
        12 => BENCHMARK_SGF_12,
        11 => BENCHMARK_SGF_11,
        10 => BENCHMARK_SGF_10,
        9 => BENCHMARK_SGF_9,
        8 => BENCHMARK_SGF_8,
        7 => BENCHMARK_SGF_7,
        _ => panic!(
            "get_benchmark_sgf_data: unsupported board size: {}",
            board_size
        ),
    }
}

/// Returns a collection of 9x9 game records covering various rulesets.
///
/// Corresponds to `TestCommon::getMultiGameSize9Data`.
pub fn get_multi_game_size9_data() -> Vec<&'static str> {
    vec![
        MULTI_GAME_SIZE9_0,
        MULTI_GAME_SIZE9_1,
        MULTI_GAME_SIZE9_2,
        MULTI_GAME_SIZE9_3,
        MULTI_GAME_SIZE9_4,
        MULTI_GAME_SIZE9_5,
        MULTI_GAME_SIZE9_6,
        MULTI_GAME_SIZE9_7,
        MULTI_GAME_SIZE9_8,
        MULTI_GAME_SIZE9_9,
        MULTI_GAME_SIZE9_10,
        MULTI_GAME_SIZE9_11,
        MULTI_GAME_SIZE9_12,
        MULTI_GAME_SIZE9_13,
        MULTI_GAME_SIZE9_14,
        MULTI_GAME_SIZE9_15,
        MULTI_GAME_SIZE9_16,
        MULTI_GAME_SIZE9_17,
        MULTI_GAME_SIZE9_18,
        MULTI_GAME_SIZE9_19,
        MULTI_GAME_SIZE9_20,
        MULTI_GAME_SIZE9_21,
        MULTI_GAME_SIZE9_22,
        MULTI_GAME_SIZE9_23,
        MULTI_GAME_SIZE9_24,
        MULTI_GAME_SIZE9_25,
        MULTI_GAME_SIZE9_26,
        MULTI_GAME_SIZE9_27,
        MULTI_GAME_SIZE9_28,
        MULTI_GAME_SIZE9_29,
    ]
}

const BENCHMARK_SGF_19: &str = "(;FF[4]GM[1]SZ[19]HA[0]KM[7.5];B[dd];W[pp];B[dp];W[pd];B[qq];W[pq];B[qp];W[qo];B[ro];W[rn];B[qn];W[po];B[rm];W[rp];B[sn];W[rq];B[nc];W[oc];B[qr];W[rr];B[nd];W[pf];B[re];W[lc];B[jc];W[le];B[qc];W[qd];B[ob];W[pc];B[pb];W[rc];B[qb];W[rd];B[lb];W[mb];B[ld];W[kb];B[kc];W[la];B[mc];W[qi];B[ke];W[cc];B[dc];W[cd];B[cf];W[ce];B[de];W[bf];B[cb];W[bb];B[be];W[db];B[bc];W[ca];B[bd];W[cb];B[bg];W[df];B[cg];W[fc];B[nr];W[nq];B[pg];W[qg];B[of];W[ef];B[mq];W[cq];B[dq];W[cp];B[co];W[bo];B[bn];W[cn];B[do];W[bm];B[bp];W[an];B[bq];W[cr];B[br];W[kq];B[iq];W[ko];B[kr];W[cj];B[lq];W[bi];B[qj];W[ph];B[og];W[rf];B[gc];W[gd];B[hc];W[nb];B[rb];W[sb];B[lb];W[fe];B[ri];W[rh];B[pi];W[qh];B[af];W[lc];B[dn];W[dm];B[em];W[in];B[lb];W[fn];B[lc];W[en];B[fp];W[fm];B[pj];W[hp];B[hq];W[ij];B[di];W[dj];B[jl];W[il];B[jm];W[im];B[jk];W[jj];B[kj];W[ki];B[lj];W[li];B[mi];W[jh];B[pn];W[or];B[np];W[ie];B[dh];W[ei];B[eh];W[fh];B[kn];W[oo];B[mn];W[oh];B[ni];W[oq];B[fg];W[gh];B[hg];W[hh];B[ig];W[ih];B[fb];W[eb];B[gb];W[mj];B[mg];W[kf];B[lf];W[je];B[kg];W[kd];B[jg];W[ln];B[lm];W[lo];B[nn];W[ra];B[na];W[gp];B[gq];W[fo];B[rj];W[oe];B[me];W[mr];B[lr];W[ns];B[sp];W[on];B[om];W[pm];B[nm];W[ip];B[jp];W[jo];B[kp];W[lp];B[jq];W[ap];B[dr];W[ar];B[cs];W[aq];B[sr];W[qs];B[ab];W[ea];B[ec];W[fd];B[fa];W[ee];B[ke];W[oi];B[oj];W[le];B[ik];W[hk];B[ke];W[ep];B[fq];W[le];B[eo];W[ke];B[sq];W[pr];B[qm];W[ml];B[mk];W[lk];B[nj];W[kk];B[mj];W[km];B[kl];W[qa];B[hd];W[he];B[gf];W[ge];B[bh];W[nf];B[ng];W[ne];B[ci];W[ai];B[mf];W[eg];B[gg];W[dg];B[ch];W[si];B[ah];W[bj];B[sj];W[sh];B[ma];W[sc];B[pa];W[id];B[ic];W[ls];B[ks];W[ms];B[sa];W[ra];B[od];W[pe];B[kh];W[lh];B[ji];W[ii];B[mh];W[ji];B[lg];W[mp];B[no];W[jn];B[km];W[as];B[fi];W[ej])";

const BENCHMARK_SGF_18: &str = "(;FF[4]GM[1]SZ[18]HA[0]KM[7.5];B[od];W[do];B[oo];W[dd];B[cp];W[dp];B[co];W[cn];B[bn];W[bm];B[cm];W[dn];B[bl];W[bo];B[am];W[bp];B[cq];W[bq];B[fc];W[pp];B[op];W[po];B[pm];W[pn];B[on];W[qm];B[pq];W[pc];B[qn];W[rn];B[ql];W[qo];B[pl];W[qq];B[pd];W[oc];B[nd];W[mb];B[cf];W[fd];B[gd];W[fe];B[dc];W[cc];B[ec];W[df];B[ed];W[de];B[cb];W[bd];B[bc];W[cd];B[ch];W[bb];B[ba];W[ac];B[lc];W[lb];B[kc];W[mk];B[lm];W[mi];B[lj];W[mj];B[kl];W[mm];B[mn];W[nm];B[om];W[ln];B[kn];W[lo];B[om];W[ko];B[jo];W[jp];B[io];W[ip];B[ho];W[hp];B[kh];W[oi];B[ph];W[oh];B[go];W[gp];B[fp];W[fq];B[fo];W[lq];B[eq];W[dq];B[mp];W[pg];B[og];W[ng];B[of];W[pj];B[qj];W[qi];B[qh];W[qk];B[ri];W[pk];B[rk];W[ki];B[li];W[lh];B[kj];W[ji];B[jj];W[kg];B[jh];W[ii];B[jg];W[kf];B[gq];W[hq];B[hh];W[mg];B[fr];W[lp];B[eh];W[dm];B[nk];W[oj];B[rl];W[dk];B[hf];W[gg];B[gh];W[fg];B[fh];W[db];B[da];W[ca];B[nq];W[ea];B[rp];W[rq];B[rm];W[qr];B[qn];W[cj];B[bj];W[qm];B[qc];W[qb];B[qn];W[pr];B[oq];W[qm];B[qd];W[gl];B[fm];W[gj];B[el];W[dl];B[ek];W[ej];B[fj];W[ij];B[il];W[bi];B[bk];W[ci];B[cl];W[ih];B[ig];W[fk];B[fi];W[ik];B[jk];W[im];B[jl];W[jn];B[hm];W[km];B[ll];W[ml];B[in];W[qn];B[lf];W[le];B[mf];W[lg];B[kb];W[hg];B[je];W[ie];B[he];W[id];B[if];W[gc];B[hd];W[gb];B[ol];W[jd];B[hc];W[kd];B[hb];W[md];B[mc];W[nb];B[me];W[jf];B[rb];W[oa];B[fb];W[cg];B[bh];W[bg];B[dh];W[ah];B[mo];W[ld];B[nf];W[jb];B[jc];W[ka];B[ic];W[ke];B[fl];W[eb];B[ga];W[hr];B[ia];W[ja];B[qa];W[pb];B[gk];W[kn];B[aj];W[ai];B[nj];W[ni];B[jm];W[ao];B[an];W[nn];B[no];W[lk];B[dr];W[cr];B[en];W[er];B[ib];W[la];B[dr];W[mr];B[jq];W[jr];B[kr];W[kq];B[ir];W[iq];B[mq];W[jr];B[er];W[pa];B[rc];W[ra])";

const BENCHMARK_SGF_17: &str = "(;FF[4]GM[1]SZ[17]HA[0]KM[7.5];B[dd];W[nn];B[nd];W[dn];B[oo];W[on];B[no];W[mo];B[mp];W[lp];B[lo];W[mn];B[kp];W[np];B[lq];W[op];B[po];W[pp];B[cn];W[co];B[cm];W[do];B[dm];W[dc];B[cd];W[cc];B[fc];W[ec];B[ed];W[fb];B[bc];W[bb];B[gb];W[fd];B[gc];W[bd];B[pl];W[pm];B[be];W[ac];B[fe];W[nf];B[of];W[og];B[oe];W[ng];B[ld];W[kg];B[ig];W[ke];B[go];W[fn];B[fl];W[fo];B[ji];W[bf];B[cf];W[ae];B[li];W[mj];B[lf];W[ne];B[md];W[od];B[pd];W[oc];B[pc];W[ob];B[pg];W[ph];B[pb];W[lg];B[mf];W[mg];B[kf];W[mb];B[lb];W[kb];B[lc];W[pf];B[pe];W[qg];B[ch];W[la];B[pa];W[jf];B[jg];W[hb];B[jc];W[hc];B[gd];W[ga];B[jb];W[ce];B[hd];W[cg];B[bh];W[ho];B[ip];W[kk];B[mi];W[nj];B[ik];W[hl];B[hk];W[dg];B[dh];W[fg];B[eh];W[ff];B[gp];W[gn];B[hp];W[jo];B[jp];W[bo];B[bn];W[hf];B[if];W[ln];B[io];W[ko];B[hn];W[fp];B[gh];W[ha];B[ib];W[jj];B[ij];W[jm];B[lp];W[il];B[gl];W[jk];B[kh];W[hg];B[hh];W[an];B[am];W[em];B[el];W[ao];B[fq];W[ep];B[ni];W[oi];B[eb];W[fa];B[ic];W[db];B[df];W[de];B[ef];W[eg];B[ee];W[bg];B[nq];W[in];B[pq];W[qo];B[oq];W[pn];B[gq];W[lj];B[ki];W[he];B[ie];W[be];B[ge];W[ah];B[ai];W[ag];B[bj];W[qe];B[qd];W[hm];B[eq];W[dq];B[fh];W[me];B[le];W[qf];B[fm];W[en];B[ia];W[ea];B[nh];W[oh];B[ho];W[qq];B[mq];W[gm];B[qp];W[dl];B[cl];W[qq];B[kj];W[mh];B[qp];W[dk];B[ck];W[qq];B[gk];W[gf];B[qp];W[qb];B[nb];W[qq];B[oj];W[pj];B[qp];W[nc];B[na];W[qq];B[lh];W[gg];B[qp];W[ma];B[oa];W[qq];B[dp];W[cp];B[qp];W[id];B[jd];W[qq];B[cq];W[bq];B[qp];W[bi];B[ci];W[qq];B[qi];W[ok];B[qp];W[bm];B[bl];W[qq];B[qh];W[pg];B[qp];W[iq];B[jq];W[qq];B[pi];W[qj];B[qp];W[qa];B[mc];W[qq];B[da];W[ca];B[qp];W[jh];B[ii];W[qq];B[kn];W[jn];B[qp];W[aj];B[qq];W[bi];B[po];W[ai];B[qn];W[qm])";

const BENCHMARK_SGF_16: &str = "(;FF[4]GM[1]SZ[16]HA[0]KM[7.5];B[mm];W[dd];B[md];W[dm];B[cc];W[cd];B[dc];W[ed];B[fb];W[ck];B[nf];W[nn];B[mn];W[nm];B[nl];W[ol];B[ok];W[nk];B[ml];W[oj];B[om];W[pk];B[on];W[no];B[oo];W[fn];B[bc];W[gd];B[ni];W[mk];B[dj];W[cj];B[dh];W[gh];B[ci];W[cg];B[dk];W[bi];B[dl];W[cl];B[em];W[en];B[hi];W[gi];B[hk];W[hj];B[fm];W[gm];B[fk];W[ij];B[dn];W[cm];B[gn];W[go];B[hn];W[ho];B[in];W[io];B[jn];W[jo];B[do];W[eo];B[ep];W[fo];B[cn];W[kn];B[bm];W[jm];B[gj];W[gl];B[bl];W[bk];B[gk];W[al];B[bn];W[mc];B[nc];W[lc];B[nb];W[me];B[nd];W[ld];B[hc];W[ch];B[di];W[fi];B[jj];W[ii];B[jl];W[mo];B[lo];W[ll];B[ln];W[im];B[hl];W[hm];B[kk];W[lk];B[kh];W[fl];B[ig];W[ih];B[jh];W[ie];B[el];W[lg];B[mh];W[jg];B[kg];W[jf];B[kf];W[ke];B[lf];W[le];B[og];W[bd];B[jd];W[je];B[hd];W[li];B[lj];W[mj];B[kj];W[ac];B[ab];W[ad];B[cb];W[he];B[dg];W[jb];B[am];W[mi];B[bj];W[aj];B[ak];W[lm];B[np];W[al];B[cf];W[bf];B[ak];W[ko];B[mp];W[al];B[kl];W[oi];B[ak];W[gb];B[gc];W[al];B[nh];W[km];B[ak];W[ne];B[oe];W[al];B[hg];W[gg];B[ak];W[fc];B[ec];W[fd];B[hb];W[al];B[hh];W[hf];B[ak];W[eb];B[ga];W[al];B[ik];W[if];B[ak];W[ca];B[ea];W[al];B[mb];W[ak];B[lb];W[lh];B[mf];W[id];B[kb];W[ao];B[bo];W[cp];B[dp];W[bp];B[jc];W[ic];B[ib];W[kc];B[ef];W[ja];B[bg];W[bh];B[be];W[af];B[ce];W[ff];B[ee];W[fe];B[fj];W[ei];B[ej];W[oh];B[ph];W[pi];B[pg];W[kd];B[pm];W[mg];B[ng];W[ae];B[pl];W[ok];B[fp];W[eg];B[df];W[eh];B[gp];W[ia];B[ha])";

const BENCHMARK_SGF_15: &str = "(;FF[4]GM[1]SZ[15]HA[0]KM[7.5];B[dd];W[ll];B[ld];W[dl];B[dm];W[cm];B[em];W[cl];B[im];W[mc];B[lc];W[md];B[le];W[nf];B[el];W[km];B[dj];W[bj];B[mj];W[mg];B[nl];W[mm];B[kk];W[jl];B[nm];W[il];B[hm];W[hl];B[bi];W[cj];B[ci];W[gm];B[gn];W[gl];B[ei];W[fn];B[cn];W[bn];B[en];W[co];B[fm];W[dc];B[ec];W[cd];B[ed];W[cc];B[in];W[ii];B[cf];W[ce];B[df];W[eb];B[fb];W[da];B[ki];W[ig];B[lg];W[gh];B[gj];W[jj];B[kj];W[fj];B[fk];W[gf];B[gi];W[hd];B[hh];W[ih];B[hi];W[hj];B[hg];W[hf];B[if];W[gk];B[fi];W[lh];B[ie];W[kg];B[lf];W[kh];B[gc];W[nn];B[nh];W[mh];B[ni];W[ml];B[mk];W[om];B[ok];W[on];B[ng];W[og];B[lk];W[li];B[kl];W[mf];B[mn];W[lm];B[no];W[ln];B[mo];W[lo];B[jm];W[oo];B[mo];W[jn];B[jo];W[ko];B[kn];W[id];B[mn];W[jc];B[kb];W[jb];B[mb];W[nb];B[je];W[gd];B[na];W[nc];B[ff];W[jd];B[hc];W[ke];B[kf];W[kd];B[gg];W[he];B[jf];W[fe];B[ja];W[ia];B[la];W[ka];B[dn];W[ef];B[de];W[bl];B[ja];W[ib];B[fg];W[fd];B[fc];W[ka];B[jk];W[ik];B[ja];W[kc];B[hb];W[me];B[ee];W[jg];B[lb];W[ka];B[bo];W[ao];B[ja];W[oa];B[ob];W[ka];B[am];W[dk];B[ja];W[bf];B[ha];W[ka];B[an];W[bo];B[ja];W[ek];B[ej];W[ka];B[aj];W[ak];B[ja];W[ai];B[bg];W[be];B[af];W[ka];B[ji];W[ij];B[ja];W[ae];B[ag];W[ka];B[ea];W[ma];B[bb];W[cb];B[db];W[ca];B[bc];W[eb];B[fa];W[ab];B[ac])";

const BENCHMARK_SGF_14: &str = "(;FF[4]GM[1]SZ[14]HA[0]KM[7.5];B[dk];W[kd];B[kk];W[dd];B[dc];W[cc];B[ec];W[cb];B[ic];W[ed];B[fd];W[jc];B[gc];W[ef];B[id];W[le];B[cg];W[bf];B[il];W[cl];B[dl];W[ck];B[cj];W[bj];B[bi];W[ci];B[dj];W[bh];B[bk];W[ai];B[bl];W[cm];B[bm];W[di];B[lg];W[jb];B[lj];W[fe];B[ei];W[eh];B[fi];W[kg];B[kh];W[he];B[ib];W[jg];B[hd];W[jh];B[ja];W[lh];B[kb];W[lc];B[lb];W[mb];B[db];W[hk];B[hi];W[ik];B[jk];W[hl];B[ij];W[jl];B[jm];W[hj];B[ii];W[gi];B[gh];W[gj];B[fh];W[im];B[km];W[em];B[dm];W[fk];B[gm];W[hh];B[fj];W[el];B[gl];W[en];B[gk];W[hm];B[gn];W[cn];B[dn];W[ej];B[bn];W[ek];B[gf];W[jj];B[eg];W[ih];B[dh];W[mi];B[in];W[ji];B[hn];W[ll];B[kl];W[mj];B[mk];W[ml];B[lk];W[nk];B[mm];W[nm];B[lm];W[bg];B[cd];W[ce];B[ge];W[bd];B[ma];W[mc];B[ca];W[ba];B[da];W[mn];B[li];W[mh];B[ki];W[mg];B[bb];W[bc];B[aa];W[je];B[ff];W[ee];B[nl];W[ml];B[if];W[ie];B[hf];W[jf])";

const BENCHMARK_SGF_13: &str = "(;FF[4]GM[1]SZ[13]HA[0]KM[7.5];B[dd];W[jj];B[kk];W[jd];B[kj];W[dj];B[jc];W[dc];B[cc];W[ec];B[ed];W[fc];B[fd];W[ic];B[hc];W[kc];B[gc];W[ch];B[jb];W[ib];B[id];W[ie];B[hd];W[kb];B[je];W[ja];B[kd];W[jc];B[jf];W[jk];B[ji];W[ii];B[fk];W[kl];B[ll];W[il];B[km];W[dk];B[hk];W[jg];B[kg];W[jh];B[ki];W[gj];B[hj];W[gk];B[hl];W[kh];B[if];W[lg];B[gi];W[li];B[jl];W[le];B[bg];W[bh];B[cg];W[dg];B[df];W[gb];B[dh];W[di];B[eg];W[cb];B[bb];W[db];B[bj];W[bi];B[aj];W[ai];B[cl];W[bk];B[dl];W[fj];B[fl];W[fi];B[cj];W[ck];B[bl];W[gh];B[hi];W[eh];B[ig];W[al];B[ek];W[ej];B[fg];W[fh];B[ih];W[kf];B[ba];W[gl];B[gm];W[hb];B[ke];W[ld];B[gg];W[ag];B[bf];W[bc];B[cd];W[ca];B[bd];W[lj];B[lk];W[hh];B[hg];W[mk];B[ml];W[mj];B[dg];W[bm];B[cm];W[ci];B[am];W[ac];B[ad];W[bm];B[af];W[ah];B[am];W[em];B[ak];W[al];B[fm];W[bm];B[ak];W[aj];B[am];W[ab];B[aa];W[bm];B[fb];W[fa];B[am];W[el];B[dm];W[bm];B[hm];W[ak];B[am];W[em];B[el];W[bm];B[ea];W[eb];B[am];W[gd])";

const BENCHMARK_SGF_12: &str = "(;FF[4]GM[1]SZ[12]HA[0]KM[7.5];B[ii];W[dd];B[cc];W[cd];B[dc];W[ed];B[ec];W[di];B[fd];W[jd];B[cg];W[fe];B[gd];W[bc];B[jf];W[hc];B[hd];W[ic];B[eg];W[be];B[fi];W[gf];B[gb];W[fj];B[gj];W[ej];B[gk];W[fh];B[fg];W[if];B[je];W[gi];B[gg];W[hi];B[hh];W[hj];B[ij];W[hk];B[ik];W[il];B[jl];W[hl];B[hb];W[ie];B[id];W[jg];B[kg];W[kh];B[ig];W[jh];B[hf];W[ih];B[he];W[kf];B[ke];W[lg];B[bb];W[ab];B[ba];W[bg];B[bh];W[cf];B[ag];W[bf];B[ch];W[df];B[eh];W[ei];B[gh];W[jk];B[ck];W[le];B[ld];W[lf];B[kd];W[cj];B[bj];W[dk];B[ci];W[cl];B[ee];W[ef];B[ff];W[de];B[fi];W[kl];B[dj];W[bk];B[cj];W[fk];B[ak];W[bl];B[af];W[ae];B[ah];W[fh];B[ge];W[fi];B[ee];W[cb];B[db];W[fe];B[ac];W[ad];B[ee];W[dl];B[aj];W[fe];B[li];W[ki];B[ee];W[ac];B[al];W[fe];B[ek];W[el];B[ee];W[aa];B[ca];W[fe];B[lh];W[lj];B[ee];W[ea];B[da])";

const BENCHMARK_SGF_11: &str = "(;FF[4]GM[1]SZ[11]HA[0]KM[7.5];B[hd];W[dh];B[dc];W[hh];B[cf];W[id];B[ie];W[he];B[hc];W[dd];B[cd];W[ec];B[de];W[ed];B[db];W[ic];B[ib];W[jb];B[hb];W[je];B[if];W[jf];B[ig];W[ge];B[ih];W[fb];B[ja];W[kb];B[jd];W[jc];B[kd];W[ke];B[jg];W[kc];B[eb];W[fc];B[ee];W[fe];B[gg];W[ff];B[fi];W[ej];B[bh];W[hi];B[fg];W[fj];B[ef];W[gd];B[ii];W[cj];B[ci];W[bj];B[gj];W[gi];B[hj];W[di];B[hg];W[ch];B[bg];W[bi];B[eh];W[dg];B[ea];W[gb];B[ai];W[aj];B[ah];W[kg];B[kh];W[ei];B[fh];W[gk];B[hk];W[fk];B[cg];W[eg];B[kf];W[cc];B[bc];W[df];B[ce];W[kg];B[jd];W[kd];B[cb];W[kf];B[hf];W[ga];B[fa];W[ha];B[ia];W[gc];B[gf])";

const BENCHMARK_SGF_10: &str = "(;FF[4]GM[1]SZ[10]HA[0]KM[7.5];B[gc];W[gg];B[dh];W[cd];B[dc];W[cg];B[ch];W[cc];B[he];W[hf];B[ie];W[fh];B[bg];W[gd];B[hc];W[ec];B[db];W[dd];B[eb];W[bf];B[cf];W[dg];B[be];W[bh];B[af];W[bi];B[eg];W[df];B[ef];W[de];B[eh];W[ce];B[ci];W[bf];B[gf];W[hh];B[cf];W[bj];B[cj];W[bf];B[fi];W[gi];B[fg];W[gh];B[if];W[ei];B[cf];W[fe];B[ff];W[bf];B[ah];W[ag];B[ii];W[ih];B[bg];W[hd];B[id];W[ag];B[ej];W[fj];B[bg];W[dj];B[cf];W[fc];B[fb];W[bf];B[di];W[ag];B[ej];W[bg];B[ji];W[dj];B[jh];W[ge];B[hg];W[ig])";

const BENCHMARK_SGF_9: &str = "(;FF[4]GM[1]SZ[9]HA[0]KM[7];B[ef];W[ed];B[ge];W[gc];B[cc];W[cd];B[bd];W[ce];B[be];W[dg];B[cf];W[df];B[de];W[dd];B[ee];W[cg];B[bf];W[cb];B[eg];W[bc];B[bh];W[he];B[hd];W[gf];B[fe];W[hf];B[fc];W[eb];B[gd];W[fh];B[eh];W[hh];B[ac];W[dc];B[fb];W[ab];B[fg];W[gg];B[fi];W[bg];B[dh];W[gh];B[ea];W[da];B[fa];W[ad];B[ch];W[id];B[ic];W[ie];B[gb];W[gi];B[ec];W[hc];B[hb];W[ei];B[db];W[ae];B[ag];W[eb];B[ig];W[db];B[ih];W[ii];B[di];W[ac];B[fi];W[hg];B[ei];W[af];B[ff];W[if];B[fd];W[bb])";

const BENCHMARK_SGF_8: &str = "(;FF[4]GM[1]SZ[8]HA[0]KM[10];B[ee];W[dd];B[ed];W[de];B[df];W[cf];B[dc];W[cc];B[ec];W[ef];B[dg];W[cg];B[eg];W[ff];B[fg];W[cb];B[db];W[fe];B[ce];W[cd];B[be];W[fc];B[fd];W[gd];B[gc];W[da];B[fb];W[bf];B[bd];W[bc];B[ea];W[gg];B[ca];W[ad];B[gf];W[ge];B[hg];W[hf];B[gh];W[ba];B[gf];W[dh];B[he];W[eh];B[fh];W[ch];B[da];W[ae];B[hd])";

const BENCHMARK_SGF_7: &str = "(;FF[4]GM[1]SZ[7]HA[0]KM[9];B[dd];W[ed];B[ee];W[dc];B[cd];W[ec];B[fe];W[cc];B[bc];W[bb];B[bd];W[de];B[fd];W[df];B[ab];W[bf];B[ef];W[fc];B[cb];W[gd];B[dg];W[cg];B[cf];W[ce];B[be];W[eg];B[fg];W[ba];B[dg];W[gc];B[cf];W[db];B[ac];W[ge];B[gf];W[ca];B[aa])";

const MULTI_GAME_SIZE9_0: &str = "(;FF[4]GM[1]SZ[9]PB[bot0]PW[bot1]HA[0]KM[7.5]RU[koPOSITIONALscoreAREAtaxALLsui1];B[ee];W[ec];B[gd];W[dg];B[fg];W[ff];B[ef];W[eg];B[gg];W[cf];B[ed];W[gc];B[dc];W[fc];B[be];W[bf];B[eb];W[hd];B[ge];W[he];B[hf];W[fb];B[db];W[hb];B[ce];W[ae];B[bh];W[ch];B[eh];W[dh];B[fh];W[bd];B[cd];W[bc];B[bb];W[ab];B[cc];W[ah];B[bg];W[bi];B[ag];W[af];B[ai];W[di];B[ci];W[cg];B[df];W[bi];B[ad];W[ac];B[ci];W[if];B[ig];W[bi];B[ga];W[ah];B[fa];W[ha];B[ie];W[ic];B[fd];W[gb];B[ea];W[id];B[ba];W[if];B[ei];W[ie];B[gf];W[ad];)";

const MULTI_GAME_SIZE9_1: &str = "(;FF[4]GM[1]SZ[9]PB[bot1]PW[bot0]HA[0]KM[7.5]RU[koSITUATIONALscoreAREAtaxALLsui0];B[ee];W[eg];B[cf];W[fc];B[dc];W[fe];B[ff];W[ef];B[gf];W[gh];B[hh];W[fh];B[de];W[gd];B[hg];W[he];B[hf];W[cg];B[bg];W[ge];B[dg];W[hi];B[ih];W[dh];B[ch];W[db];B[eb];W[ec];B[cb];W[dd];B[da];W[df];B[ed];W[cg];B[ce];W[if];B[hc];W[gc];B[hb];W[fg];B[fb];W[gg];B[dg];)";

const MULTI_GAME_SIZE9_2: &str = "(;FF[4]GM[1]SZ[9]PB[bot1]PW[bot0]HA[0]KM[6.5]RU[koPOSITIONALscoreTERRITORYtaxNONEsui0];B[ee];W[ec];B[fc];W[fd];B[ed];W[gc];B[fb];W[dc];B[cd];W[gd];B[cc];W[gb];B[fh];W[dh];B[eh];W[cf];B[dg];W[cg];B[ch];W[df];B[di];W[ef];B[fe];W[ff];B[ge];W[gf];B[he];W[hf];B[dd];W[eb];B[be];W[hh];B[bh];W[gh];B[fg];W[bf];B[db];W[fa];B[ae];W[af];B[eg];W[gg];B[ig];W[if];B[gi];W[hi];B[ih];W[ce];B[bd];W[cb];B[bb];W[hd];B[ie];)";

const MULTI_GAME_SIZE9_3: &str = "(;FF[4]GM[1]SZ[9]PB[bot0]PW[bot1]HA[0]KM[7]RU[koPOSITIONALscoreTERRITORYtaxALLsui1];B[ee];W[eg];B[dc];W[gf];B[ff];W[fg];B[ge];W[hf];B[df];W[db];B[eb];W[cc];B[cb];W[dd];B[ec];W[ch];B[cg];W[bh];B[he];W[bb];B[da];W[bd];B[ce];W[ae];B[be];W[cd];B[dh];W[ef];B[fe];W[de];B[dg];W[eh];B[ei];W[gh];B[af];W[bg];B[ad];W[bc];B[fi];W[bf];B[hh];W[cf];)";

const MULTI_GAME_SIZE9_4: &str = "(;FF[4]GM[1]SZ[9]PB[bot1]PW[bot0]HA[0]KM[7]RU[koSITUATIONALscoreAREAtaxNONEsui0];B[df];W[dd];B[fe];W[ec];B[gd];W[fg];B[eh];W[gf];B[fh];W[hd];B[hc];W[he];B[fc];W[gh];B[cc];W[cd];B[eb];W[cf];B[cg];W[dc];B[db];W[cb];B[be];W[bc];B[eg];W[bf];B[id];W[ie];B[hh];W[hi];B[hg];W[gg];B[de];W[gb];B[fb];W[hb];B[ce];W[bd];B[gc];W[ic];B[ib];W[ia];B[ga];W[ih];B[ig];W[ee];B[ef];W[ih];B[ba];W[ca];B[ii];W[ed];B[ff];W[ih];B[bb];W[ab];B[ii];W[ge];B[fa];W[ae];B[ag];W[id];B[ha];W[ib];B[fd];W[af];B[bg];W[ad];B[da];W[aa];B[fi];W[bb];)";

const MULTI_GAME_SIZE9_5: &str = "(;FF[4]GM[1]SZ[9]PB[bot0]PW[bot1]HA[0]KM[7]RU[koSITUATIONALscoreTERRITORYtaxALLsui0];B[ee];W[ce];B[gf];W[dc];B[ec];W[eb];B[dd];W[fc];B[ed];W[dg];B[db];W[fb];B[eh];W[dh];B[be];W[bf];B[bd];W[gd];B[fg];W[cf];B[bh];W[bg];B[ge];W[ch];B[cc];W[he];B[hf];W[ie];B[hc];W[hd];B[hb];W[if];B[hh];W[ig];B[ea];W[ha];B[fa];W[ga];B[da];W[gb];B[ic];W[fd];B[fe];W[hg];B[gh];W[ei];B[fi];W[di];B[ih];W[eg];B[fh];W[ef];B[de];W[df];B[af];W[ag];B[ae];W[ff];B[gg];W[cd];B[bc];W[];B[ia];W[];B[bi];W[ah];B[];W[];B[ib];W[];)";

const MULTI_GAME_SIZE9_6: &str = "(;FF[4]GM[1]SZ[9]PB[bot1]PW[bot0]HA[0]KM[6.5]RU[koSITUATIONALscoreTERRITORYtaxSEKIsui0];B[ee];W[ge];B[cf];W[dc];B[fc];W[cd];B[gf];W[hf];B[gg];W[fe];B[ef];W[ed];B[hd];W[he];B[fd];W[gd];B[gc];W[hc];B[hb];W[id];B[ib];W[eb];B[ic];W[ec];B[gb];W[hg];B[hh];W[fb];B[ff];W[gh];B[ih];W[fh];B[eh];W[hi];B[hd];W[dg];B[ie];W[eg];B[df];W[cg];B[bg];W[bh];B[be];W[ag];B[bc];W[bd];B[ad];W[bb];B[ab];W[cc];B[ac];W[ba];B[bf];W[de];B[ce];W[aa];B[ah];W[ai];B[ae];W[fg];B[dd];W[da];B[ig];W[fa];B[gi];W[ch];)";

const MULTI_GAME_SIZE9_7: &str = "(;FF[4]GM[1]SZ[9]PB[bot0]PW[bot1]HA[0]KM[6]RU[koSIMPLEscoreTERRITORYtaxNONEsui1];B[ee];W[ec];B[dc];W[dd];B[ed];W[cc];B[db];W[fc];B[gd];W[cd];B[gc];W[cb];B[eg];W[cg];B[fb];W[dh];B[cf];W[eh];B[eb];W[df];B[fg];W[gh];B[dg];W[de];B[ch];W[bg];B[bh];W[bf];B[gh];W[gg];B[gf];W[hg];B[hh];W[hf];B[he];W[ff];B[ef];W[ih];B[ig];W[if];B[hi];W[fi];B[di];W[ge];B[hd];W[ah];B[ci];W[da];B[ea];W[ca];B[af];W[be];B[fd];W[bi];B[ai];W[ag];B[fe];W[bi];)";

const MULTI_GAME_SIZE9_8: &str = "(;FF[4]GM[1]SZ[9]PB[bot0]PW[bot1]HA[0]KM[7]RU[koSIMPLEscoreAREAtaxSEKIsui0];B[dd];W[ff];B[fe];W[ge];B[gd];W[gf];B[cf];W[hd];B[eg];W[fd];B[ec];W[fb];B[eb];W[eh];B[dh];W[fh];B[fc];W[gc];B[fg];W[gg];B[dg];W[gb];B[ee];W[ea];B[da];W[fa];B[gg];W[hg];B[db];W[di];B[ci];W[ei];B[ch];W[ef];B[df];W[];B[de];W[];B[hb];W[ha];B[hf];W[hc];B[];W[he];B[];W[ib];B[];W[hg];B[];W[if];B[];W[])";

const MULTI_GAME_SIZE9_9: &str = "(;FF[4]GM[1]SZ[9]PB[bot1]PW[bot0]HA[0]KM[7]RU[koSITUATIONALscoreAREAtaxALLsui1button1];B[ee];W[ce];B[gf];W[ge];B[ff];W[cg];B[ec];W[fd];B[fc];W[gc];B[cd];W[bd];B[gb];W[eg];B[cc];W[hb];B[hc];W[hd];B[gd];W[gh];B[gc];W[hg];B[df];W[cf];B[dg];W[dh];B[ch];W[eh];B[bc];W[hf];B[bh];W[de];B[ef];W[ae];B[bg];W[bf];B[dd];W[bi];B[ci];W[ag];B[ac];W[ah];B[ai];)";

const MULTI_GAME_SIZE9_10: &str = "(;FF[4]GM[1]SZ[9]PB[bot0]PW[bot1]HA[0]KM[6.5]RU[koPOSITIONALscoreTERRITORYtaxNONEsui0];B[ee];W[ec];B[eg];W[cd];B[fc];W[fb];B[fd];W[eh];B[dh];W[fg];B[fh];W[ef];B[dg];W[ff];B[df];W[gh];B[ei];W[ge];B[eb];W[db];B[dc];W[ea];B[cc];W[dd];B[ed];W[eb];B[hd];W[he];B[gc];W[bc];B[hh];W[gi];B[hf];W[hg];B[ie];W[ih];B[hi];W[de];B[if];W[cf];B[bg];W[bf];B[gb];W[ga];B[ha];W[hb];B[fa];W[cg];B[ch];W[bh];B[fi];W[ga];B[cb];W[gg];B[bi];W[ia];B[ic];W[bb];B[ib];W[ah];B[ci];W[ha];B[hc];W[ca];B[fa];W[ha];B[ig];W[ga];B[fe];W[hb];B[gf];W[fa];B[ia];)";

const MULTI_GAME_SIZE9_11: &str = "(;FF[4]GM[1]SZ[9]PB[bot0]PW[bot1]HA[0]KM[6.5]RU[koSITUATIONALscoreTERRITORYtaxSEKIsui0];B[ee];W[ce];B[gf];W[fc];B[dc];W[gd];B[cf];W[bf];B[cg];W[de];B[ef];W[ed];B[dd];W[bd];B[hc];W[hd];B[fb];W[gb];B[ec];W[fd];B[eb];W[cb];B[hb];W[hf];B[hg];W[gc];B[ga];W[he];B[ha];W[fa];B[fe];W[da];B[ea];W[gg];B[gh];W[fg];B[hh];W[fh];B[eg];W[ih];B[eh];W[ic];B[ib];W[cc];B[id];W[hi];B[fi];W[bg];B[bh];)";

const MULTI_GAME_SIZE9_12: &str = "(;FF[4]GM[1]SZ[9]PB[bot0]PW[bot1]HA[0]KM[7.5]RU[koSITUATIONALscoreAREAtaxNONEsui0];B[ee];W[ec];B[cd];W[fg];B[dg];W[df];B[ef];W[eg];B[dh];W[cf];B[cg];W[gf];B[fd];W[bg];B[bh];W[cc];B[de];W[eh];B[bc];W[di];B[ch];W[be];B[ce];W[bf];B[ah];W[gc];B[gd];W[hc];B[hd];W[db];B[bd];W[hf];B[cb];W[dc];B[ba];W[ic];B[da];W[ea];B[ie];W[if];B[fc];W[fb];B[ca];W[hb];B[ci];W[ei];)";

const MULTI_GAME_SIZE9_13: &str = "(;FF[4]GM[1]SZ[9]PB[bot0]PW[bot1]HA[0]KM[7]RU[koSIMPLEscoreAREAtaxSEKIsui0];B[dd];W[ff];B[fe];W[ge];B[gd];W[gf];B[cf];W[hd];B[eg];W[fd];B[ec];W[fb];B[eb];W[eh];B[dh];W[fh];B[fc];W[gc];B[fg];W[gg];B[dg];W[gb];B[ee];W[ef];B[df];W[ea];B[da];W[fa];B[dc];W[ei];B[di];W[gh];B[db];W[gd];B[ed];W[];B[de];W[];B[hb];W[ha];B[hf];W[hc];B[];W[he];B[];W[ib];B[];W[hg];B[];W[if];B[];W[])";

const MULTI_GAME_SIZE9_14: &str = "(;FF[4]GM[1]SZ[9]PB[bot1]PW[bot0]HA[0]KM[7]RU[koSIMPLEscoreAREAtaxSEKIsui1];B[dd];W[ff];B[ef];W[eg];B[dg];W[fg];B[fc];W[cd];B[ce];W[gd];B[fd];W[dc];B[ee];W[dh];B[ge];W[cg];B[bd];W[gc];B[hd];W[hc];B[hf];W[fb];B[eb];W[fa];B[fe];W[id];B[he];W[ec];B[db];W[ha];B[gb];W[ic];B[gh];W[gg];B[hh];W[bf];B[fh];W[cc];B[be];W[bc];B[bb];W[cb];B[bh];W[df];B[cf];W[bg];B[ah];W[hg];B[ig];W[eh];B[ch];W[ag];B[de];W[dg];B[ei];W[bi];B[ac];W[ab];B[aa];W[ca];B[fi];W[ih];B[ii];W[ba];B[di];W[ci];B[gf];W[ai];B[bh];W[ab];B[ae];W[ed];B[if];)";

const MULTI_GAME_SIZE9_15: &str = "(;FF[4]GM[1]SZ[9]PB[bot1]PW[bot0]HA[0]KM[6]RU[koSIMPLEscoreTERRITORYtaxNONEsui1];B[ff];W[dd];B[de];W[ce];B[cf];W[cd];B[gd];W[bf];B[ec];W[cg];B[ed];W[fg];B[df];W[dg];B[bg];W[be];B[gg];W[fh];B[gh];W[ef];B[ee];W[eg];B[cb];W[gf];B[hf];W[fe];B[ge];W[ff];B[fd];W[bb];B[bh];W[ch];B[ei];W[di];B[fi];W[eh];B[gi];W[bi];B[cc];W[dc];B[db];W[he];B[hg];W[hd];B[gc];W[hc];B[hb];W[fb];B[eb];W[bc];B[ba];)";

const MULTI_GAME_SIZE9_16: &str = "(;FF[4]GM[1]SZ[9]PB[bot0]PW[bot1]HA[0]KM[6]RU[koSIMPLEscoreTERRITORYtaxNONEsui0];B[ee];W[ec];B[dc];W[dd];B[ed];W[cc];B[db];W[fc];B[gd];W[cd];B[gc];W[cb];B[dg];W[eh];B[gg];W[cg];B[dh];W[df];B[ch];W[ef];B[eg];W[ff];B[gf];W[bg];B[gb];W[fg];B[fh];W[bh];B[eb];W[gh];B[ei];W[hg];B[hf];W[hi];B[cf];W[bf];B[ig];W[fi];B[gi];W[fb];B[fd];W[fi];B[de];W[ce];B[gi];W[fa];B[fe];W[cf];B[hh];W[da];)";

const MULTI_GAME_SIZE9_17: &str = "(;FF[4]GM[1]SZ[9]PB[bot0]PW[bot1]HA[0]KM[7.5]RU[koPOSITIONALscoreAREAtaxALLsui0button1];B[ee];W[eg];B[dc];W[gf];B[ge];W[ff];B[fe];W[df];B[be];W[he];B[hd];W[hf];B[cg];W[de];B[cd];W[bf];B[bg];W[gc];B[hc];W[cf];B[af];W[fc];B[ed];W[ec];B[dd];W[db];B[cb];W[hb];B[gd];W[da];B[ha];W[ib];B[fa];W[fb];B[ca];W[eb];B[gb];W[ga];B[fh];W[eh];B[gb];W[ae];B[ad];W[ga];B[dh];W[gb];B[ef];W[fg];B[dg];W[di];B[ci];W[ei];B[ie];W[if];B[id];)";

const MULTI_GAME_SIZE9_18: &str = "(;FF[4]GM[1]SZ[9]PB[bot0]PW[bot1]HA[0]KM[6.5]RU[koSITUATIONALscoreTERRITORYtaxSEKIsui1];B[ee];W[ce];B[dc];W[gf];B[eg];W[fd];B[cg];W[bg];B[bh];W[cf];B[bc];W[fe];B[dg];W[ed];B[de];W[dd];B[cd];W[eb];B[db];W[fg];B[eh];W[be];B[bd];W[ch];B[ci];W[ah];B[bi];W[fh];B[fi];W[ei];B[di];W[gi];B[fb];W[ec];B[ea];W[ei];B[ef];W[gb];B[ga];W[ha];B[fa];W[fc];B[da];W[hc];B[ae];W[ff];B[fi];W[bb];B[ei];W[af];B[ab];W[ad];B[ac];W[ae];B[ai];W[ag];B[gh];W[ca];B[cc];W[aa];B[ba];W[hi];B[cb];W[hh];)";

const MULTI_GAME_SIZE9_19: &str = "(;FF[4]GM[1]SZ[9]PB[bot1]PW[bot0]HA[0]KM[6]RU[koPOSITIONALscoreTERRITORYtaxSEKIsui1];B[ee];W[ge];B[gd];W[cd];B[dc];W[hd];B[gc];W[cf];B[df];W[gg];B[cg];W[be];B[fh];W[hc];B[bf];W[ce];B[bg];W[cc];B[cb];W[bb];B[db];W[ba];B[gh];W[hb];B[fg];W[gf];B[gb];W[fd];B[fc];W[ga];B[fa];W[hh];B[hi];W[ed];B[dd];W[ec];B[ha];W[fe];B[de];W[ae];B[hg];W[eb];B[ea];)";

const MULTI_GAME_SIZE9_20: &str = "(;FF[4]GM[1]SZ[9]PB[bot0]PW[bot1]HA[0]KM[7]RU[koSIMPLEscoreAREAtaxSEKIsui0];B[df];W[ff];B[fe];W[ge];B[fd];W[gd];B[gf];W[fg];B[gg];W[gh];B[hh];W[fh];B[fc];W[he];B[cf];W[dh];B[ch];W[gc];B[gb];W[hb];B[fb];W[cd];B[hg];W[dc];B[hi];W[bh];B[cg];W[ci];B[ef];W[eh];B[bg];W[ee];B[ha];W[ib];B[dd];W[ed];B[ec];W[de];B[ce];W[dd];B[db];W[bd];B[bb];W[bi];B[ah];W[be];B[di];W[ei];B[bc];W[eg];B[af];W[gi];B[ad];W[hf];B[ag];)";

const MULTI_GAME_SIZE9_21: &str = "(;FF[4]GM[1]SZ[9]PB[bot0]PW[bot1]HA[0]KM[6]RU[koSITUATIONALscoreTERRITORYtaxSEKIsui1];B[ee];W[ce];B[ge];W[he];B[gd];W[gf];B[cf];W[hd];B[gc];W[df];B[de];W[bf];B[cg];W[cd];B[dc];W[dd];B[ff];W[gg];B[bg];W[ec];B[cc];W[eb];B[bb];W[ed];B[bd];W[be];B[db];W[dg];B[dh];W[ad];B[bc];W[eg];B[fg];W[eh];B[fh];W[ch];B[bh];W[di];B[bi];W[ah];B[fi];W[ei];B[ci];W[ag];B[dh];W[hc];B[hb];W[ea];B[gb];W[ch];B[hh];W[ca];B[dh];W[gh];B[ef];W[gi];B[da];W[ba];B[ac];)";

const MULTI_GAME_SIZE9_22: &str = "(;FF[4]GM[1]SZ[9]PB[bot1]PW[bot0]HA[0]KM[6.5]RU[koPOSITIONALscoreTERRITORYtaxSEKIsui0];B[ee];W[ce];B[cd];W[dd];B[de];W[cc];B[bd];W[cf];B[dg];W[dc];B[cg];W[bc];B[hd];W[ge];B[he];W[gg];B[gd];W[eh];B[ih];W[hg];B[eg];W[dh];B[ch];W[fg];B[di];W[fi];B[ig];W[fe];B[fd];W[hf];B[if];W[ff];B[bg];W[be];B[ci];W[hh];B[ed];W[hi];B[eb];W[ec];)";

const MULTI_GAME_SIZE9_23: &str = "(;FF[4]GM[1]SZ[9]PB[bot0]PW[bot1]HA[0]KM[7]RU[koSIMPLEscoreTERRITORYtaxALLsui0];B[ee];W[ge];B[cd];W[fg];B[ff];W[gf];B[eg];W[fh];B[fd];W[gd];B[fc];W[eh];B[dg];W[bd];B[be];W[cc];B[bc];W[gc];B[gb];W[hb];B[fb];W[ch];B[cg];W[bh];B[bg];W[ah];B[ha];W[ib];B[dh];W[di];B[fe];W[ag];B[af];W[hc];B[ga];W[ia];B[];W[];B[];W[];B[ad];W[];B[dc];W[];B[ef];W[];B[dd];W[];B[cb];W[];B[de];W[];B[ce];W[];B[])";

const MULTI_GAME_SIZE9_24: &str = "(;FF[4]GM[1]SZ[9]PB[bot1]PW[bot0]HA[0]KM[7]RU[koSIMPLEscoreAREAtaxSEKIsui1];B[fd];W[df];B[ef];W[eg];B[fg];W[dg];B[dc];W[fh];B[ce];W[gg];B[ff];W[gf];B[ge];W[ee];B[fe];W[de];B[gh];W[hh];B[cd];W[he];B[hf];W[hg];B[hd];W[ed];B[ec];W[if];B[gc];W[bf];B[cf];W[cg];B[be];W[bg];B[ae];W[id];B[ic];W[ie];B[af];W[ag];B[dd];W[gi];B[];W[])";

const MULTI_GAME_SIZE9_25: &str = "(;FF[4]GM[1]SZ[9]PB[bot0]PW[bot1]HA[0]KM[7]RU[koSIMPLEscoreAREAtaxALLsui0];B[ee];W[eg];B[fc];W[cf];B[df];W[dg];B[ce];W[bf];B[ff];W[fg];B[gf];W[be];B[cc];W[fb];B[gc];W[bc];B[cd];W[bb];B[gg];W[cb];B[db];W[da];B[eb];W[ea];B[bd];W[ad];B[fh];W[eh];B[gh];W[de];B[dd];W[gb];B[hb];W[ha];B[ic];W[ef];B[de];W[ec];B[ei];W[ch];B[dc];W[di];B[fa];W[ga];B[ia];W[ib];B[cg];W[hc];B[dh];W[bg];B[dg];W[he];B[bh];W[ah];B[ci];W[ih];B[ge];W[hd];B[ai];W[bi];B[hf];W[ch];B[ae];W[af];B[bh];W[if];B[ai];W[ac];B[ag];W[ae];B[gd];W[ah];B[ie];W[id];B[ag];W[hh];B[hi];W[ah];B[hg];W[ig];B[ag];W[gi];B[ah];)";

const MULTI_GAME_SIZE9_26: &str = "(;FF[4]GM[1]SZ[9]PB[bot1]PW[bot0]HA[0]KM[6.5]RU[koSITUATIONALscoreTERRITORYtaxSEKIsui0];B[ee];W[ce];B[gf];W[fc];B[dc];W[gd];B[cf];W[bf];B[cg];W[de];B[ef];W[ed];B[dd];W[bd];B[hc];W[hd];B[fb];W[gb];B[ec];W[fd];B[eb];W[cb];B[hb];W[gc];B[ga];W[hf];B[hg];W[he];B[ha];W[fa];B[fe];W[da];B[ea];W[gg];B[fg];W[df];B[gh];W[cc];B[if];)";

const MULTI_GAME_SIZE9_27: &str = "(;FF[4]GM[1]SZ[9]PB[bot0]PW[bot1]HA[0]KM[7]RU[koSIMPLEscoreTERRITORYtaxALLsui0button1];B[ee];W[eg];B[dc];W[gf];B[ge];W[ff];B[fe];W[df];B[be];W[he];B[hd];W[hf];B[cg];W[de];B[cd];W[bf];B[bg];W[gc];B[hc];W[cf];B[af];W[fc];B[ed];W[ec];B[dd];W[db];B[cb];W[hb];B[gd];W[da];B[ha];W[ib];B[fa];W[fb];B[ca];W[eb];B[gb];W[ga];B[fh];W[eh];B[gb];W[ae];B[ad];W[ga];B[dh];W[gb];B[ef];W[fg];B[dg];W[di];B[ci];W[ei];B[ie];W[if];B[id];)";

const MULTI_GAME_SIZE9_28: &str = "(;FF[4]GM[1]SZ[9]PB[bot0]PW[bot1]HA[0]KM[6.5]RU[koSITUATIONALscoreTERRITORYtaxSEKIsui1];B[ee];W[ce];B[dc];W[gf];B[eg];W[fd];B[cg];W[bg];B[bh];W[cf];B[bc];W[fe];B[dg];W[ed];B[de];W[dd];B[cd];W[eb];B[db];W[fg];B[eh];W[be];B[bd];W[ch];B[ci];W[ah];B[bi];W[fh];B[fi];W[ei];B[di];W[gi];B[fb];W[ec];B[ea];W[ei];B[ef];W[gb];B[ga];W[ha];B[fa];W[fc];B[da];W[hc];B[ae];W[ff];B[fi];W[bb];B[ei];W[af];B[ab];W[ad];B[ac];W[ae];B[ai];W[ag];B[gh];W[ca];B[cc];W[aa];B[ba];W[hi];B[cb];W[hh];)";

const MULTI_GAME_SIZE9_29: &str = "(;FF[4]GM[1]SZ[9]PB[bot1]PW[bot0]HA[0]KM[6]RU[koPOSITIONALscoreTERRITORYtaxSEKIsui1];B[ee];W[ge];B[gd];W[cd];B[dc];W[hd];B[gc];W[cf];B[df];W[gg];B[cg];W[be];B[fh];W[hc];B[bf];W[ce];B[bg];W[cc];B[cb];W[bb];B[db];W[ba];B[gh];W[hb];B[fg];W[gf];B[gb];W[fd];B[fc];W[ga];B[fa];W[hh];B[hi];W[ed];B[dd];W[ec];B[ha];W[fe];B[de];W[ae];B[hg];W[eb];B[ea];)";
