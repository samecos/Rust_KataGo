//! CPU semantic tests for the shared production output path.
//! Fixtures below are synthetic raw heads; no model output or corpus is read.

use super::*;
use kata_game::board::{Board, P_BLACK, P_WHITE, Player, location};
use kata_game::history::BoardHistory;
use kata_game::rules::{KoRule, Rules, ScoringRule};
use kata_game::symmetry::copy_inputs_with_symmetry;

const AREA: usize = 361;
const POLICY: usize = 362;

struct Heads {
    policy: Vec<f32>,
    value: Vec<f32>,
    misc: Vec<f32>,
    moremisc: Vec<f32>,
    ownership: Vec<f32>,
}

impl Heads {
    fn new(batch: usize) -> Self {
        Self {
            policy: vec![0.0; batch * 6 * POLICY],
            value: vec![0.0; batch * 3],
            misc: vec![0.0; batch * 10],
            moremisc: vec![0.0; batch * 8],
            ownership: vec![0.0; batch * AREA],
        }
    }

    fn raw(&self) -> RawHeads<'_> {
        RawHeads {
            policy: &self.policy,
            value: &self.value,
            misc: &self.misc,
            moremisc: &self.moremisc,
            ownership: &self.ownership,
        }
    }

    fn head_mut(&mut self, head: usize) -> &mut Vec<f32> {
        match head {
            0 => &mut self.policy,
            1 => &mut self.value,
            2 => &mut self.misc,
            3 => &mut self.moremisc,
            4 => &mut self.ownership,
            _ => panic!("unknown test head"),
        }
    }

    fn policy_channel_mut(&mut self, row: usize, channel: usize) -> &mut [f32] {
        let start = (row * 6 + channel) * POLICY;
        &mut self.policy[start..start + POLICY]
    }

    fn decode(&self, batch: usize, row: usize, params: &MiscNNInputParams) -> NNOutput {
        decode_raw_row_v7(self.raw(), batch, row, params, true).unwrap()
    }
}

fn params() -> MiscNNInputParams {
    MiscNNInputParams {
        symmetry: 0,
        ..MiscNNInputParams::default()
    }
}

fn chinese() -> Rules {
    Rules::parse_rules("chinese").unwrap()
}

fn area_without_no_result() -> Rules {
    Rules {
        ko_rule: KoRule::Situational,
        ..chinese()
    }
}

fn process(
    output: &mut NNOutput,
    version: i32,
    pp: ModelPostProcessParams,
    board: &Board,
    rules: Rules,
    next: Player,
    params: &MiscNNInputParams,
) {
    let history = BoardHistory::new(board.clone(), next, rules, 0);
    postprocess_output(
        version,
        pp,
        19,
        19,
        POLICY as i32,
        board,
        &history,
        next,
        params,
        output,
    );
}

fn near(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= tolerance,
        "actual={actual:?}, expected={expected:?}, tolerance={tolerance:?}"
    );
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn decode_uses_each_rows_six_pass_aware_policy_channels_and_exact_head_offsets() {
    let mut heads = Heads::new(2);
    for row in 0..2 {
        for channel in 0..6 {
            heads
                .policy_channel_mut(row, channel)
                .fill(-100.0 - channel as f32);
        }
    }
    heads.policy_channel_mut(0, 0).fill(11.0);
    heads.policy_channel_mut(0, 5).fill(19.0);
    heads.policy_channel_mut(1, 0).fill(2.0);
    heads.policy_channel_mut(1, 5).fill(10.0);
    heads.policy_channel_mut(1, 0)[AREA] = 3.0;
    heads.policy_channel_mut(1, 5)[AREA] = 19.0;
    heads.value[3..6].copy_from_slice(&[2.0, 3.0, 5.0]);
    heads.misc[10..14].copy_from_slice(&[7.0, 11.0, 13.0, 17.0]);
    heads.moremisc[8..10].copy_from_slice(&[19.0, 23.0]);
    heads.ownership[AREA..].fill(29.0);
    let input = MiscNNInputParams {
        policy_optimism: 0.25,
        ..params()
    };
    let output = heads.decode(2, 1, &input);
    assert!(output.policy_probs[..AREA].iter().all(|&x| x == 4.0));
    assert_eq!(output.policy_probs[AREA], 7.0);
    assert_eq!(output.policy_optimism_used, 0.25);
    assert_eq!((output.nn_x_len, output.nn_y_len), (19, 19));
    assert_eq!(
        bits(&[
            output.white_win_prob,
            output.white_loss_prob,
            output.white_no_result_prob,
            output.white_score_mean,
            output.white_score_mean_sq,
            output.white_lead,
            output.var_time_left,
            output.shortterm_winloss_error,
            output.shortterm_score_error,
        ]),
        bits(&[2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0])
    );
    assert_eq!(output.white_owner_map.as_deref(), Some(&[29.0; AREA][..]));
    let without_owner = decode_raw_row_v7(heads.raw(), 2, 1, &input, false).unwrap();
    assert!(without_owner.white_owner_map.is_none());
    assert_eq!(
        bits(&without_owner.policy_probs[..POLICY]),
        bits(&output.policy_probs[..POLICY])
    );
}

#[test]
fn quarter_turn_outputs_undo_the_production_input_transform_once() {
    let original: Vec<f32> = (0..AREA).map(|i| i as f32 - 128.0).collect();
    let owner: Vec<f32> = (0..AREA).map(|i| (i as f32 - 90.0) * 0.125).collect();
    for symmetry in [5, 6] {
        let mut heads = Heads::new(1);
        let mut transformed = vec![0.0; AREA];
        copy_inputs_with_symmetry(&original, &mut transformed, 1, 19, 19, 1, false, symmetry);
        assert_ne!(
            bits(&transformed),
            bits(&original),
            "fixture must expose orientation"
        );
        heads.policy_channel_mut(0, 0)[..AREA].copy_from_slice(&transformed);
        heads.policy_channel_mut(0, 5)[..AREA].copy_from_slice(&transformed);
        heads.policy_channel_mut(0, 0)[AREA] = 17.0;
        heads.policy_channel_mut(0, 5)[AREA] = 33.0;
        copy_inputs_with_symmetry(&owner, &mut heads.ownership, 1, 19, 19, 1, false, symmetry);
        let output = heads.decode(
            1,
            0,
            &MiscNNInputParams {
                symmetry,
                policy_optimism: 0.25,
                ..params()
            },
        );
        assert_eq!(bits(&output.policy_probs[..AREA]), bits(&original));
        assert_eq!(output.policy_probs[AREA], 21.0);
        assert_eq!(
            bits(output.white_owner_map.as_deref().unwrap()),
            bits(&owner)
        );
    }
}

#[test]
fn player_to_move_outputs_flip_only_the_white_perspective_fields() {
    let mut heads = Heads::new(1);
    heads.value.copy_from_slice(&[2.0, -1.0, -2.0]);
    heads.misc[..4].copy_from_slice(&[0.25, 0.5, 0.75, 1.0]);
    heads.moremisc[..2].copy_from_slice(&[0.3, 0.6]);
    heads.ownership[0] = 0.75;
    heads.ownership[1] = -0.5;
    let input = params();
    let board = Board::new(19, 19);
    let mut white = heads.decode(1, 0, &input);
    let mut black = white.clone();
    process(
        &mut white,
        17,
        ModelPostProcessParams::default(),
        &board,
        chinese(),
        P_WHITE,
        &input,
    );
    process(
        &mut black,
        17,
        ModelPostProcessParams::default(),
        &board,
        chinese(),
        P_BLACK,
        &input,
    );
    assert!(white.white_win_prob > white.white_loss_prob);
    assert_eq!(
        white.white_win_prob.to_bits(),
        black.white_loss_prob.to_bits()
    );
    assert_eq!(
        white.white_loss_prob.to_bits(),
        black.white_win_prob.to_bits()
    );
    assert_eq!(
        white.white_no_result_prob.to_bits(),
        black.white_no_result_prob.to_bits()
    );
    assert!(white.white_score_mean > 0.0 && white.white_lead > 0.0);
    assert_eq!(
        white.white_score_mean.to_bits(),
        (-black.white_score_mean).to_bits()
    );
    assert_eq!(white.white_lead.to_bits(), (-black.white_lead).to_bits());
    assert_eq!(
        bits(&[
            white.white_score_mean_sq,
            white.var_time_left,
            white.shortterm_winloss_error,
            white.shortterm_score_error
        ]),
        bits(&[
            black.white_score_mean_sq,
            black.var_time_left,
            black.shortterm_winloss_error,
            black.shortterm_score_error
        ])
    );
    assert_eq!(
        bits(&white.policy_probs[..POLICY]),
        bits(&black.policy_probs[..POLICY])
    );
    for (a, b) in white
        .white_owner_map
        .as_ref()
        .unwrap()
        .iter()
        .zip(black.white_owner_map.as_ref().unwrap().iter())
    {
        assert_eq!(a.to_bits(), (-b).to_bits());
        assert!((-1.0..=1.0).contains(a));
    }
    assert!(white.white_owner_map.as_ref().unwrap()[0] > 0.0);
    assert!(white.white_owner_map.as_ref().unwrap()[1] < 0.0);
}

#[test]
fn chinese_simple_ko_retains_no_result_and_only_non_simple_area_suppresses_it() {
    let input = params();
    let heads = Heads::new(1);
    let board = Board::new(19, 19);
    let chinese_rules = chinese();
    assert_eq!(chinese_rules.ko_rule, KoRule::Simple);
    assert_eq!(chinese_rules.scoring_rule, ScoringRule::Area);
    let mut simple = heads.decode(1, 0, &input);
    process(
        &mut simple,
        17,
        ModelPostProcessParams::default(),
        &board,
        chinese_rules,
        P_WHITE,
        &input,
    );
    near(simple.white_no_result_prob, 1.0 / 3.0, 1e-7);
    let mut area = heads.decode(1, 0, &input);
    process(
        &mut area,
        17,
        ModelPostProcessParams::default(),
        &board,
        area_without_no_result(),
        P_WHITE,
        &input,
    );
    assert_eq!(area.white_no_result_prob, 0.0);
    assert_eq!(area.white_win_prob, 0.5);
    assert_eq!(area.white_loss_prob, 0.5);
    let mut territory = heads.decode(1, 0, &input);
    process(
        &mut territory,
        17,
        ModelPostProcessParams::default(),
        &board,
        Rules {
            scoring_rule: ScoringRule::Territory,
            ..area_without_no_result()
        },
        P_WHITE,
        &input,
    );
    near(territory.white_no_result_prob, 1.0 / 3.0, 1e-7);
}

#[test]
fn score_channel_one_is_stdev_and_model_multipliers_have_distinct_roles() {
    let mut heads = Heads::new(1);
    heads.misc[..4].copy_from_slice(&[2.0, 0.0, 5.0, 50.0]);
    let input = params();
    let board = Board::new(19, 19);
    let pp = ModelPostProcessParams {
        score_mean_multiplier: 3.0,
        score_stdev_multiplier: 4.0,
        lead_multiplier: 7.0,
        variance_time_multiplier: 2.0,
        ..ModelPostProcessParams::default()
    };
    let mut base = heads.decode(1, 0, &input);
    assert_eq!(
        base.white_score_mean_sq, 0.0,
        "raw field carries stdev preactivation"
    );
    process(
        &mut base,
        17,
        pp,
        &board,
        area_without_no_result(),
        P_WHITE,
        &input,
    );
    assert_eq!(base.white_score_mean, 6.0);
    assert_eq!(base.white_lead, 35.0);
    assert_eq!(base.var_time_left, 100.0);
    near(base.white_score_mean_sq, 43.68725, 1e-4);
    let mut wider = heads.decode(1, 0, &input);
    process(
        &mut wider,
        17,
        ModelPostProcessParams {
            score_stdev_multiplier: 8.0,
            ..pp
        },
        &board,
        area_without_no_result(),
        P_WHITE,
        &input,
    );
    assert_eq!(wider.white_score_mean, base.white_score_mean);
    assert_eq!(wider.white_lead, base.white_lead);
    near(
        wider.white_score_mean_sq - 36.0,
        4.0 * (base.white_score_mean_sq - 36.0),
        2e-5,
    );
    let mut scaled = heads.decode(1, 0, &input);
    process(
        &mut scaled,
        17,
        ModelPostProcessParams {
            output_scale_multiplier: 2.0,
            ..pp
        },
        &board,
        area_without_no_result(),
        P_WHITE,
        &input,
    );
    assert_eq!(scaled.white_score_mean, 12.0);
    assert_eq!(scaled.white_lead, 70.0);
    assert_eq!(scaled.var_time_left, 200.0);
}

#[test]
fn uncertainty_uses_model_version_branches_and_preserves_old_sentinels() {
    let heads = Heads::new(1);
    let input = params();
    let board = Board::new(19, 19);
    let pp = ModelPostProcessParams {
        shortterm_value_error_multiplier: 4.0,
        shortterm_score_error_multiplier: 9.0,
        ..ModelPostProcessParams::default()
    };
    // Known semantic values at zero logits; these do not reproduce the implementation.
    for (version, expected_win, expected_score) in [
        (8, -1.0, -1.0),
        (9, 0.6931472, 6.931472),
        (10, 1.6651093, 2.4976637),
        (13, 1.6651093, 2.4976637),
        (14, 1.3862944, 2.0794415),
        (17, 1.3862944, 2.0794415),
    ] {
        let mut output = heads.decode(1, 0, &input);
        process(&mut output, version, pp, &board, chinese(), P_WHITE, &input);
        near(output.shortterm_winloss_error, expected_win, 1e-6);
        near(output.shortterm_score_error, expected_score, 1e-6);
        if version < 9 {
            assert_eq!(output.var_time_left, -1.0);
        } else {
            assert!(output.var_time_left > 0.0);
        }
    }
}

#[test]
fn legal_mask_temperature_scaling_and_pass_hack_share_production_semantics() {
    let input = params();
    let mut board = Board::new(19, 19);
    let occupied = 9 * 19 + 9;
    assert!(board.set_stone(location::get_loc(9, 9, 19), P_BLACK));
    let mut heads = Heads::new(1);
    heads.policy_channel_mut(0, 0)[occupied] = 1000.0;
    heads.policy_channel_mut(0, 0)[0] = 4.0;
    let mut cold = heads.decode(1, 0, &input);
    process(
        &mut cold,
        17,
        ModelPostProcessParams::default(),
        &board,
        chinese(),
        P_WHITE,
        &input,
    );
    assert_eq!(cold.policy_probs[occupied], -1.0);
    near(
        cold.policy_probs[..POLICY]
            .iter()
            .filter(|&&x| x >= 0.0)
            .sum(),
        1.0,
        2e-5,
    );
    let warm_input = MiscNNInputParams {
        nn_policy_temperature: 2.0,
        ..input
    };
    let mut warm = heads.decode(1, 0, &warm_input);
    process(
        &mut warm,
        17,
        ModelPostProcessParams::default(),
        &board,
        chinese(),
        P_WHITE,
        &warm_input,
    );
    assert_eq!(warm.policy_probs[occupied], -1.0);
    assert!(cold.policy_probs[0] > warm.policy_probs[0]);
    assert!(cold.policy_probs[1] < warm.policy_probs[1]);
    let mut equivalent = heads.decode(1, 0, &warm_input);
    process(
        &mut equivalent,
        17,
        ModelPostProcessParams {
            output_scale_multiplier: 2.0,
            ..ModelPostProcessParams::default()
        },
        &board,
        chinese(),
        P_WHITE,
        &warm_input,
    );
    assert_eq!(
        bits(&equivalent.policy_probs[..POLICY]),
        bits(&cold.policy_probs[..POLICY])
    );

    let hack_input = MiscNNInputParams {
        enable_passing_hacks: true,
        ..input
    };
    let mut pass_heads = Heads::new(1);
    pass_heads.policy_channel_mut(0, 0)[AREA] = 10.0;
    let mut plain = pass_heads.decode(1, 0, &input);
    let mut capped = plain.clone();
    process(
        &mut plain,
        17,
        ModelPostProcessParams::default(),
        &board,
        chinese(),
        P_WHITE,
        &input,
    );
    process(
        &mut capped,
        17,
        ModelPostProcessParams::default(),
        &board,
        chinese(),
        P_WHITE,
        &hack_input,
    );
    assert!(plain.policy_probs[AREA] > 0.95);
    near(capped.policy_probs[AREA], 0.95, 1e-6);
    assert_eq!(capped.policy_probs[occupied], -1.0);
    near(
        capped.policy_probs[..POLICY]
            .iter()
            .filter(|&&x| x >= 0.0)
            .sum(),
        1.0,
        2e-5,
    );
    pass_heads.policy_channel_mut(0, 0).fill(-1000.0);
    pass_heads.policy_channel_mut(0, 0)[AREA] = 1000.0;
    let mut underflow = pass_heads.decode(1, 0, &input);
    process(
        &mut underflow,
        17,
        ModelPostProcessParams::default(),
        &board,
        chinese(),
        P_WHITE,
        &hack_input,
    );
    assert_eq!(underflow.policy_probs[AREA], 1.0);
    assert!(
        underflow.policy_probs[..AREA]
            .iter()
            .enumerate()
            .all(|(i, &p)| p == if i == occupied { -1.0 } else { 0.0 })
    );
}

#[test]
fn decoding_rejects_wrong_full_head_shapes_and_nonfinite_values_before_row_selection() {
    let input = params();
    for head in 0..5 {
        for extra in [false, true] {
            let mut heads = Heads::new(2);
            if extra {
                heads.head_mut(head).push(0.0);
            } else {
                heads.head_mut(head).pop();
            }
            assert!(
                decode_raw_row_v7(heads.raw(), 2, 0, &input, true).is_err(),
                "head {head}, extra={extra}"
            );
        }
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut heads = Heads::new(2);
            *heads.head_mut(head).last_mut().unwrap() = invalid;
            // The second row is unselected; all five bound tensors must still be valid.
            assert!(
                decode_raw_row_v7(heads.raw(), 2, 0, &input, true).is_err(),
                "head {head}"
            );
            assert!(
                decode_raw_row_v7(heads.raw(), 2, 0, &input, false).is_err(),
                "head {head}"
            );
        }
    }
    let mut unused_channel = Heads::new(1);
    unused_channel.policy_channel_mut(0, 3)[5] = f32::NAN;
    assert!(decode_raw_row_v7(unused_channel.raw(), 1, 0, &input, true).is_err());
    let valid = Heads::new(2);
    assert!(decode_raw_row_v7(valid.raw(), 0, 0, &input, true).is_err());
    assert!(decode_raw_row_v7(valid.raw(), 2, 2, &input, true).is_err());
    assert!(decode_raw_row_v7(valid.raw(), 65, 0, &input, true).is_err());
    for symmetry in [-1, 8] {
        let invalid = MiscNNInputParams { symmetry, ..input };
        assert!(decode_raw_row_v7(valid.raw(), 2, 0, &invalid, true).is_err());
    }
    for policy_optimism in [-0.01, 1.01, f64::NAN, f64::INFINITY] {
        let invalid = MiscNNInputParams {
            policy_optimism,
            ..input
        };
        assert!(decode_raw_row_v7(valid.raw(), 2, 0, &invalid, true).is_err());
    }
}
