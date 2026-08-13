//! Integration tests for NN input feature encoding.
//!
//! Ported from `KataGo/cpp/tests/testnninputs.cpp`. Rather than matching the
//! C++ printed channel snapshots exactly, these tests verify structural
//! properties: expected feature counts, NHWC/NCHW equivalence, hash
//! stability, and a small number of easy-to-check channel semantics.

use kata_data::sgf::CompactSgf;
use kata_game::board::{Board, Move, P_BLACK, P_WHITE, Player, get_opp, location};
use kata_game::history::BoardHistory;
use kata_game::rules::{KoRule, Rules, ScoringRule, TaxRule, WhiteHandicapBonusRule};
use kata_nn::inputs::{
    MiscNNInputParams, NUM_FEATURES_GLOBAL_V3, NUM_FEATURES_GLOBAL_V4, NUM_FEATURES_GLOBAL_V5,
    NUM_FEATURES_GLOBAL_V6, NUM_FEATURES_GLOBAL_V7, NUM_FEATURES_SPATIAL_V3,
    NUM_FEATURES_SPATIAL_V4, NUM_FEATURES_SPATIAL_V5, NUM_FEATURES_SPATIAL_V6,
    NUM_FEATURES_SPATIAL_V7, fill_row_v3, fill_row_v4, fill_row_v5, fill_row_v6, fill_row_v7,
    get_hash,
};

const VERSIONS: &[i32] = &[3, 4, 5, 6, 7];
const VERSIONS_NO_V5: &[i32] = &[3, 4, 6, 7];
const VERSIONS_V6_V7: &[i32] = &[6, 7];

fn num_spatial_features(version: i32) -> usize {
    match version {
        3 => NUM_FEATURES_SPATIAL_V3 as usize,
        4 => NUM_FEATURES_SPATIAL_V4 as usize,
        5 => NUM_FEATURES_SPATIAL_V5 as usize,
        6 => NUM_FEATURES_SPATIAL_V6 as usize,
        7 => NUM_FEATURES_SPATIAL_V7 as usize,
        _ => panic!("unsupported version"),
    }
}

fn num_global_features(version: i32) -> usize {
    match version {
        3 => NUM_FEATURES_GLOBAL_V3 as usize,
        4 => NUM_FEATURES_GLOBAL_V4 as usize,
        5 => NUM_FEATURES_GLOBAL_V5 as usize,
        6 => NUM_FEATURES_GLOBAL_V6 as usize,
        7 => NUM_FEATURES_GLOBAL_V7 as usize,
        _ => panic!("unsupported version"),
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_row_version(
    version: i32,
    board: &Board,
    hist: &BoardHistory,
    next_pla: Player,
    params: &MiscNNInputParams,
    nn_x_len: i32,
    nn_y_len: i32,
    use_nhwc: bool,
    row_bin: &mut [f32],
    row_global: &mut [f32],
) {
    match version {
        3 => fill_row_v3(
            board, hist, next_pla, params, nn_x_len, nn_y_len, use_nhwc, row_bin, row_global,
        ),
        4 => fill_row_v4(
            board, hist, next_pla, params, nn_x_len, nn_y_len, use_nhwc, row_bin, row_global,
        ),
        5 => fill_row_v5(
            board, hist, next_pla, params, nn_x_len, nn_y_len, use_nhwc, row_bin, row_global,
        ),
        6 => fill_row_v6(
            board, hist, next_pla, params, nn_x_len, nn_y_len, use_nhwc, row_bin, row_global,
        ),
        7 => fill_row_v7(
            board, hist, next_pla, params, nn_x_len, nn_y_len, use_nhwc, row_bin, row_global,
        ),
        _ => panic!("unsupported version"),
    }
}

fn nhwc_index(pos: usize, feature: usize, num_features: usize) -> usize {
    pos * num_features + feature
}

fn nchw_index(pos: usize, feature: usize, nn_area: usize) -> usize {
    feature * nn_area + pos
}

fn convert_nhwc_to_nchw(src: &[f32], nn_area: usize, num_features: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; nn_area * num_features];
    for pos in 0..nn_area {
        for f in 0..num_features {
            dst[nchw_index(pos, f, nn_area)] = src[nhwc_index(pos, f, num_features)];
        }
    }
    dst
}

fn assert_layout_equivalence(nhwc: &[f32], nchw: &[f32], nn_area: usize, num_features: usize) {
    assert_eq!(nhwc.len(), nn_area * num_features);
    assert_eq!(nchw.len(), nn_area * num_features);
    let converted = convert_nhwc_to_nchw(nhwc, nn_area, num_features);
    assert_eq!(
        converted, nchw,
        "NHWC and NCHW rows should encode the same spatial features"
    );
}

fn assert_hash_stable(
    board: &Board,
    hist: &BoardHistory,
    next_pla: Player,
    params: &MiscNNInputParams,
) {
    let h1 = get_hash(board, hist, next_pla, params);
    let h2 = get_hash(board, hist, next_pla, params);
    assert_eq!(h1, h2, "get_hash should be stable for identical inputs");
}

fn replay_moves(board: &mut Board, hist: &mut BoardHistory, next_pla: &mut Player, moves: &[Move]) {
    for m in moves {
        assert!(hist.is_legal(board, m.loc, m.pla));
        hist.make_board_move_assume_legal(board, m.loc, m.pla);
        *next_pla = get_opp(m.pla);
    }
}

fn load_sgf_and_replay(sgf_str: &str, default_rules: &Rules) -> (Board, BoardHistory, Player) {
    let sgf = CompactSgf::parse(sgf_str).expect("valid sgf");
    let mut board = Board::new(sgf.x_size, sgf.y_size);
    let mut next_pla = P_BLACK;
    let mut hist = BoardHistory::default();
    let rules = sgf
        .get_rules_or_fail_allow_unspecified(default_rules)
        .expect("rules");
    sgf.setup_initial_board_and_hist(&rules, &mut board, &mut next_pla, &mut hist)
        .expect("setup");
    replay_moves(&mut board, &mut hist, &mut next_pla, &sgf.moves);
    (board, hist, next_pla)
}

fn params_with_draw(draw_equivalent_wins_for_white: f64) -> MiscNNInputParams {
    MiscNNInputParams {
        draw_equivalent_wins_for_white,
        ..MiscNNInputParams::default()
    }
}

struct RowBuffers {
    nhwc: Vec<f32>,
    nchw: Vec<f32>,
    global_nhwc: Vec<f32>,
    global_nchw: Vec<f32>,
}

impl RowBuffers {
    fn new(nn_area: usize, num_bin: usize, num_global: usize) -> Self {
        Self {
            nhwc: vec![0.0f32; nn_area * num_bin],
            nchw: vec![0.0f32; nn_area * num_bin],
            global_nhwc: vec![0.0f32; num_global],
            global_nchw: vec![0.0f32; num_global],
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn fill_both(
        &mut self,
        version: i32,
        board: &Board,
        hist: &BoardHistory,
        next_pla: Player,
        params: &MiscNNInputParams,
        nn_x_len: i32,
        nn_y_len: i32,
    ) {
        fill_row_version(
            version,
            board,
            hist,
            next_pla,
            params,
            nn_x_len,
            nn_y_len,
            true,
            &mut self.nhwc,
            &mut self.global_nhwc,
        );
        fill_row_version(
            version,
            board,
            hist,
            next_pla,
            params,
            nn_x_len,
            nn_y_len,
            false,
            &mut self.nchw,
            &mut self.global_nchw,
        );
    }

    fn assert_equivalent(&self, nn_area: usize, num_bin: usize) {
        assert_layout_equivalence(&self.nhwc, &self.nchw, nn_area, num_bin);
        assert_eq!(
            self.global_nhwc, self.global_nchw,
            "global features should be independent of layout"
        );
    }
}

fn assert_on_board_marker(
    row: &[f32],
    board: &Board,
    nn_x_len: i32,
    nn_y_len: i32,
    num_bin: usize,
) {
    for y in 0..nn_y_len {
        for x in 0..nn_x_len {
            let pos = (y * nn_x_len + x) as usize;
            let on_board = x < board.x_size && y < board.y_size;
            let val = row[nhwc_index(pos, 0, num_bin)];
            if on_board {
                assert_eq!(val, 1.0, "on-board marker should be 1");
            } else {
                assert_eq!(val, 0.0, "padding marker should be 0");
            }
        }
    }
}

fn assert_stone_channels_match_board(
    row: &[f32],
    board: &Board,
    next_pla: Player,
    nn_x_len: i32,
    num_bin: usize,
) {
    let opp = get_opp(next_pla);
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = location::get_loc(x, y, board.x_size);
            let pos = (y * nn_x_len + x) as usize;
            let stone = board.colors[loc as usize];
            let pla_val = row[nhwc_index(pos, 1, num_bin)];
            let opp_val = row[nhwc_index(pos, 2, num_bin)];
            if stone == next_pla {
                assert_eq!(pla_val, 1.0);
                assert_eq!(opp_val, 0.0);
            } else if stone == opp {
                assert_eq!(pla_val, 0.0);
                assert_eq!(opp_val, 1.0);
            } else {
                assert_eq!(pla_val, 0.0);
                assert_eq!(opp_val, 0.0);
            }
        }
    }
}

fn check_basic_scenario(
    _name: &str,
    sgf_str: &str,
    default_rules: &Rules,
    nn_x_len: i32,
    nn_y_len: i32,
    draw_equivalent_wins_for_white: f64,
    versions: &[i32],
) {
    let (board, hist, next_pla) = load_sgf_and_replay(sgf_str, default_rules);
    let nn_area = (nn_x_len * nn_y_len) as usize;
    let params = params_with_draw(draw_equivalent_wins_for_white);

    for &version in versions {
        let num_bin = num_spatial_features(version);
        let num_global = num_global_features(version);
        let mut rows = RowBuffers::new(nn_area, num_bin, num_global);
        rows.fill_both(
            version, &board, &hist, next_pla, &params, nn_x_len, nn_y_len,
        );
        rows.assert_equivalent(nn_area, num_bin);
        assert_hash_stable(&board, &hist, next_pla, &params);
        assert_on_board_marker(&rows.nhwc, &board, nn_x_len, nn_y_len, num_bin);
        if version != 5 {
            assert_stone_channels_match_board(&rows.nhwc, &board, next_pla, nn_x_len, num_bin);
        }
    }
}

#[test]
fn nn_inputs_basic() {
    let sgf = "(;FF[4]KM[7.5];B[pd];W[pq];B[dq];W[dd];B[qo];W[pl];B[qq];W[qr];B[pp];W[rq];B[oq];W[qp];B[pr];W[qq];B[oo];W[ro];B[qn];W[do];B[dl];W[gp];B[eo];W[en];B[fo];W[dp];B[eq];W[cq];B[cr];W[br];B[dn];W[bp];B[cn];W[ep];B[fp];W[fq];B[gq];W[fr];B[gr];W[er];B[hp];W[go];B[fn];W[ho];B[ip];W[io];B[jp];W[jo];B[lp];W[kp];B[kq];W[ko];B[lq];W[ir];B[hq];W[jq];B[jr];W[em];B[gm];W[el];B[hl];W[kl];B[ek];W[fk];B[ej];W[fl];B[fj];W[gk];B[ik];W[gj];B[jj];W[dm];B[lk];W[mm];B[nl];W[nm];B[om];W[ol];B[nk];W[ll];B[kk];W[jl];B[im];W[jk];B[ij];W[kj];B[mk];W[ki];B[ih];W[jh];B[ig];W[jg];B[if];W[oi];B[mi];W[mh];B[lh];W[li];B[nh];W[mj];B[ni];W[nj];B[oj];W[lj];B[ok];W[oh];B[ng];W[pj];B[ji];W[kh];B[jf];W[lg];B[cm];W[cl];B[dk];W[bl];B[bk];W[bn];B[ck];W[bm];B[cc];W[cd];B[dc];W[ec];B[eb];W[fb];B[fc];W[ed];B[gb];W[bc];B[cb];W[cg];B[be];W[bd];B[bg];W[bh];B[cf];W[df];B[ch];W[dg];B[bi];W[qd];B[qc];W[rc];B[rd];W[qe];B[re];W[rb];B[pc];W[qb];B[qf];W[ff];B[sc];W[pb];B[ob];W[oc];B[nc];W[mc];B[lb])";
    check_basic_scenario(
        "basic",
        sgf,
        &Rules::get_tromp_taylorish(),
        19,
        19,
        0.2,
        VERSIONS,
    );
}

#[test]
fn nn_inputs_ko() {
    let sgf = "(;FF[4]KM[0.5];B[rj];W[ri];B[si];W[rh];B[sh];W[sg];B[rk];W[sk];B[sl];W[sj];B[eg];W[fg];B[ff];W[gf];B[fh];W[gh];B[gg];W[hg];B[si];W[fg];B[sh];W[sk];B[gg])";
    check_basic_scenario(
        "ko",
        sgf,
        &Rules::get_tromp_taylorish(),
        19,
        19,
        0.3,
        VERSIONS,
    );
}

#[test]
fn nn_inputs_7x7() {
    let sgf = "(;GM[1]FF[4]CA[UTF-8]ST[2]RU[Tromp-Taylor]SZ[7]HA[3]KM[-4.50]PW[White]PB[Black]AB[fb][bf][ff];W[ed];B[ee];W[de];B[dd];W[ef];B[df];W[fe];B[ce];W[dc];B[ee];W[eg];B[fd];W[de])";
    check_basic_scenario(
        "7x7",
        sgf,
        &Rules::get_tromp_taylorish(),
        7,
        7,
        0.5,
        VERSIONS,
    );
}

#[test]
fn nn_inputs_7x7_embedded_in_9x9() {
    let sgf = "(;GM[1]FF[4]CA[UTF-8]ST[2]RU[Tromp-Taylor]SZ[7]HA[3]KM[-4.50]PW[White]PB[Black]AB[fb][bf][ff];W[ed];B[ee];W[de];B[dd];W[ef];B[df];W[fe];B[ce];W[dc];B[ee];W[eg];B[fd];W[de])";
    check_basic_scenario(
        "7x7-in-9x9",
        sgf,
        &Rules::get_tromp_taylorish(),
        9,
        9,
        0.8,
        VERSIONS,
    );
}

#[test]
fn nn_inputs_area_komi() {
    let board = Board::parse_board(
        7,
        7,
        r".xo.oo.
xxo.xox
ooooooo
xxx..xx
..xoox.
..xxxxx
..xo.ox",
        '\n',
    )
    .unwrap();

    for &version in VERSIONS_NO_V5 {
        let num_bin = num_spatial_features(version);
        let num_global = num_global_features(version);
        let nn_area = 49;

        let mut rules = Rules::get_tromp_taylorish();
        rules.komi = 2;
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
        let params = params_with_draw(0.3);
        let mut rows = RowBuffers::new(nn_area, num_bin, num_global);

        rows.fill_both(version, &board, &hist, P_BLACK, &params, 7, 7);
        rows.assert_equivalent(nn_area, num_bin);
        assert_hash_stable(&board, &hist, P_BLACK, &params);

        hist.clear(board.clone(), P_WHITE, rules, 0);
        rows.fill_both(version, &board, &hist, P_WHITE, &params, 7, 7);
        rows.assert_equivalent(nn_area, num_bin);
        assert_hash_stable(&board, &hist, P_WHITE, &params);

        rules.komi = 1;
        hist.clear(board.clone(), P_BLACK, rules, 0);
        rows.fill_both(version, &board, &hist, P_BLACK, &params, 7, 7);
        rows.assert_equivalent(nn_area, num_bin);
        assert_hash_stable(&board, &hist, P_BLACK, &params);

        // Komi global should differ between komi=2 and komi=1 for the same player.
        rules.komi = 2;
        hist.clear(board.clone(), P_BLACK, rules, 0);
        rows.fill_both(version, &board, &hist, P_BLACK, &params, 7, 7);
        let komi_2 = rows.global_nhwc[5];
        rules.komi = 1;
        hist.clear(board.clone(), P_BLACK, rules, 0);
        rows.fill_both(version, &board, &hist, P_BLACK, &params, 7, 7);
        let komi_1 = rows.global_nhwc[5];
        assert_ne!(komi_2, komi_1, "self-komi global should change with komi");
    }
}

#[test]
fn nn_inputs_rules_variations() {
    let board = Board::new(7, 7);
    let next_pla = P_BLACK;
    let draw = 0.47;

    let rule_sets: Vec<Rules> = [KoRule::Simple, KoRule::Positional, KoRule::Situational]
        .iter()
        .flat_map(|&ko| {
            [ScoringRule::Area, ScoringRule::Territory]
                .iter()
                .flat_map(move |&scoring| {
                    [TaxRule::None, TaxRule::Seki, TaxRule::All]
                        .iter()
                        .map(move |&tax| {
                            Rules::new(
                                ko,
                                scoring,
                                tax,
                                false,
                                false,
                                WhiteHandicapBonusRule::Zero,
                                false,
                                3.0,
                            )
                        })
                })
        })
        .collect();

    for &version in VERSIONS {
        let num_bin = num_spatial_features(version);
        let num_global = num_global_features(version);
        let nn_area = 49;
        let params = params_with_draw(draw);
        let mut rows = RowBuffers::new(nn_area, num_bin, num_global);

        // Verify that at least some rule combinations produce distinct global vectors.
        let mut distinct_globals: Vec<Vec<f32>> = Vec::new();
        for rules in &rule_sets {
            let hist = BoardHistory::new(board.clone(), next_pla, *rules, 0);
            rows.fill_both(version, &board, &hist, next_pla, &params, 7, 7);
            rows.assert_equivalent(nn_area, num_bin);
            if !distinct_globals.iter().any(|g| g == &rows.global_nhwc) {
                distinct_globals.push(rows.global_nhwc.clone());
            }
        }
        assert!(
            distinct_globals.len() > 1,
            "v{}: different rules should produce different global features",
            version
        );
    }
}

#[test]
fn nn_inputs_v6_v7_area_feature_and_komi() {
    let board = Board::parse_board(
        7,
        7,
        r"...oxx.
oooox.x
xxxxoxx
o.xoooo
.oxox.o
..xxxxx
..xo.ox",
        '\n',
    )
    .unwrap();

    let scoring_rules = [
        ScoringRule::Area,
        ScoringRule::Area,
        ScoringRule::Territory,
        ScoringRule::Territory,
    ];
    let tax_rules = [TaxRule::None, TaxRule::Seki, TaxRule::None, TaxRule::Seki];

    for &version in VERSIONS_V6_V7 {
        let num_bin = num_spatial_features(version);
        let num_global = num_global_features(version);
        let nn_area = 49;
        let params = params_with_draw(0.5);
        let mut rows = RowBuffers::new(nn_area, num_bin, num_global);

        for which_rules in 0..4 {
            let rules = Rules::new(
                KoRule::Positional,
                scoring_rules[which_rules],
                tax_rules[which_rules],
                false,
                false,
                WhiteHandicapBonusRule::Zero,
                false,
                6.5,
            );
            let working_board = board.clone();
            let hist = BoardHistory::new(working_board.clone(), P_WHITE, rules, 0);

            rows.fill_both(version, &working_board, &hist, P_WHITE, &params, 7, 7);
            rows.assert_equivalent(nn_area, num_bin);
            assert_hash_stable(&working_board, &hist, P_WHITE, &params);
        }
    }
}

#[test]
fn nn_inputs_v6_v7_pass_history() {
    let locs = location::parse_sequence(
        "pass A1 D1 F1 H1 E1 G1 B1 J1 F1 J1 G1 C1 E1 B1 pass pass H1 J1 E1 G1 pass pass A1 pass D1 C1 B1",
        9,
        1,
    )
    .unwrap();

    for &version in VERSIONS_V6_V7 {
        let mut board = Board::new(9, 1);
        let mut next_pla = P_BLACK;
        let rules = Rules::get_simple_territory();
        let mut hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

        let num_bin = num_spatial_features(version);
        let num_global = num_global_features(version);
        let nn_area = 9;
        let params = params_with_draw(0.5);
        let mut rows = RowBuffers::new(nn_area, num_bin, num_global);

        for &loc in &locs {
            rows.fill_both(version, &board, &hist, next_pla, &params, 9, 1);
            rows.assert_equivalent(nn_area, num_bin);
            assert_hash_stable(&board, &hist, next_pla, &params);

            hist.make_board_move_assume_legal(&mut board, loc, next_pla);
            next_pla = get_opp(next_pla);
        }
    }
}

#[test]
fn nn_inputs_v7_self_komi_handicap_white_bonus() {
    let size = 7;
    let rules_variants = [
        Rules::new(
            KoRule::Positional,
            ScoringRule::Area,
            TaxRule::None,
            false,
            false,
            WhiteHandicapBonusRule::Zero,
            false,
            3.0,
        ),
        Rules::new(
            KoRule::Positional,
            ScoringRule::Area,
            TaxRule::None,
            true,
            false,
            WhiteHandicapBonusRule::NMinusOne,
            false,
            3.0,
        ),
        Rules::new(
            KoRule::Positional,
            ScoringRule::Area,
            TaxRule::None,
            false,
            true,
            WhiteHandicapBonusRule::N,
            false,
            3.0,
        ),
        Rules::new(
            KoRule::Positional,
            ScoringRule::Territory,
            TaxRule::None,
            true,
            false,
            WhiteHandicapBonusRule::NMinusOne,
            false,
            3.0,
        ),
        Rules::new(
            KoRule::Positional,
            ScoringRule::Territory,
            TaxRule::None,
            false,
            true,
            WhiteHandicapBonusRule::N,
            false,
            3.0,
        ),
    ];

    let version = 7;
    let num_bin = num_spatial_features(version);
    let num_global = num_global_features(version);
    let nn_area = (size * size) as usize;
    let params = params_with_draw(0.5);
    let mut rows = RowBuffers::new(nn_area, num_bin, num_global);

    for (i, rules) in rules_variants.iter().enumerate() {
        let mut board = Board::new(size, size);
        let next_pla = P_BLACK;
        let mut hist = BoardHistory::new(board.clone(), next_pla, *rules, 0);
        if i >= 3 {
            hist.set_assume_multiple_starting_black_moves_are_handicap(true);
        }

        let moves = [
            location::get_loc(3, 3, size),
            location::get_loc(3, 2, size),
            location::get_loc(3, 1, size),
        ];
        for loc in moves {
            hist.make_board_move_assume_legal(&mut board, loc, P_BLACK);
            rows.fill_both(version, &board, &hist, next_pla, &params, size, size);
            rows.assert_equivalent(nn_area, num_bin);
            assert_hash_stable(&board, &hist, next_pla, &params);
        }
    }
}
