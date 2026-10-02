//! Shared CPU output decoding and evaluator postprocessing.
//! Arithmetic moved from CUDA finish_output and SharedState::postprocess_output.
//! The checked row API does not attest which model/input produced raw tensors.
use crate::backend::NNResultBuf;
use crate::desc::ModelPostProcessParams;
use crate::inputs::{nn_pos, MiscNNInputParams, NNOutput};
use kata_game::board::{Board, Player, P_WHITE};
use kata_game::history::BoardHistory;
use kata_game::rules::{KoRule, ScoringRule};
use kata_game::symmetry::copy_outputs_with_symmetry;

#[derive(Clone, Copy)]
pub struct RawHeads<'a> {
    pub policy: &'a [f32],
    pub value: &'a [f32],
    pub misc: &'a [f32],
    pub moremisc: &'a [f32],
    pub ownership: &'a [f32],
}

pub(crate) fn policy_channel(policy: &[f32], row: usize, channel: usize, area: usize) -> &[f32] {
    let stride = area + 1;
    let offset = (row * 6 + channel) * stride;
    &policy[offset..offset + stride]
}

/// Production CUDA path: preserve its existing shape assumptions and allocations.
/// Physical buffers may contain padded rows; only the first n logical rows decode.
pub(crate) fn decode_raw_outputs(
    nn_x_len: i32, nn_y_len: i32, n: usize,
    input_bufs: &[&mut NNResultBuf], host: RawHeads<'_>, outputs: &mut [&mut NNOutput],
) {
    let policy_area = (nn_x_len * nn_y_len) as usize;
    let mut tmp_policy_base = vec![0.0f32; policy_area];
    let mut tmp_policy_opt = vec![0.0f32; policy_area];
    let mut tmp_ownership = vec![0.0f32; policy_area];
    for i in 0..n {
        let sym_idx = input_bufs[i].symmetry;
        // copy_outputs_with_symmetry already reverses the input
        // transform. Passing invert(sym_idx) reverses it twice for
        // quarter-turn symmetries 5 and 6.
        let base_channel = policy_channel(&host.policy, i, 0, policy_area);
        let opt_channel = policy_channel(&host.policy, i, 5, policy_area);
        let base_src = &base_channel[..policy_area];
        // policy_concat_kernel writes [B,6,S+1]: each channel has its
        // own trailing pass logit. Include it in the channel stride.
        let opt_src = &opt_channel[..policy_area];
        if sym_idx != 0 {
            copy_outputs_with_symmetry(
                base_src,
                &mut tmp_policy_base,
                1,
                nn_y_len,
                nn_x_len,
                sym_idx,
            );
            copy_outputs_with_symmetry(
                opt_src,
                &mut tmp_policy_opt,
                1,
                nn_y_len,
                nn_x_len,
                sym_idx,
            );
        } else {
            tmp_policy_base.copy_from_slice(base_src);
            tmp_policy_opt.copy_from_slice(opt_src);
        }
        let optimism = input_bufs[i].policy_optimism as f32;
        for pos in 0..policy_area {
            outputs[i].policy_probs[pos] = tmp_policy_base[pos]
                + (tmp_policy_opt[pos] - tmp_policy_base[pos]) * optimism;
        }
        let base_pass = base_channel[policy_area];
        let opt_pass = opt_channel[policy_area];
        outputs[i].policy_probs[policy_area] =
            base_pass + (opt_pass - base_pass) * optimism;
        let v_off = i * 3;
        outputs[i].white_win_prob = host.value[v_off];
        outputs[i].white_loss_prob = host.value[v_off + 1];
        outputs[i].white_no_result_prob = host.value[v_off + 2];
        let m_off = i * 10;
        outputs[i].white_score_mean = host.misc[m_off];
        outputs[i].white_score_mean_sq = host.misc[m_off + 1];
        outputs[i].white_lead = host.misc[m_off + 2];
        outputs[i].var_time_left = host.misc[m_off + 3];
        let mm_off = i * 8;
        outputs[i].shortterm_winloss_error = host.moremisc[mm_off];
        outputs[i].shortterm_score_error = host.moremisc[mm_off + 1];
        if input_bufs[i].include_owner_map {
            let o_off = i * policy_area;
            let src = &host.ownership[o_off..o_off + policy_area];
            if sym_idx != 0 {
                copy_outputs_with_symmetry(
                    src,
                    &mut tmp_ownership,
                    1,
                    nn_y_len,
                    nn_x_len,
                    sym_idx,
                );
            } else {
                tmp_ownership.copy_from_slice(src);
            }
            outputs[i].white_owner_map = Some(tmp_ownership.clone().into_boxed_slice());
        }
        outputs[i].nn_x_len = nn_x_len;
        outputs[i].nn_y_len = nn_y_len;
        outputs[i].policy_optimism_used = input_bufs[i].policy_optimism as f32;
    }
}

/// Checked finite 19x19 row for explicit CPU diagnostic consumers.
/// No second input symmetry is applied. The shared decoder reverses output symmetry.
pub fn decode_raw_row_v7(raw: RawHeads<'_>, physical_batch: usize, row: usize,
    params: &MiscNNInputParams, include_ownership: bool) -> Result<NNOutput, String> {
    if !(1..=64).contains(&physical_batch) || row >= physical_batch
        || !(0..=7).contains(&params.symmetry) || !params.policy_optimism.is_finite()
        || !(0.0..=1.0).contains(&params.policy_optimism) {
        return Err("invalid raw output row or symmetry/optimism".into());
    }
    for (head,width) in [(raw.policy,6*362),(raw.value,3),(raw.misc,10),(raw.moremisc,8),(raw.ownership,361)] {
        if head.len()!=physical_batch*width || head.iter().any(|x|!x.is_finite()) {
            return Err("invalid raw output head shape or nonfinite value".into());
        }
    }
    fn head(data: &[f32], row: usize, width: usize) -> &[f32] { &data[row*width..(row+1)*width] }
    let host=RawHeads {policy:head(raw.policy,row,6*362),value:head(raw.value,row,3),misc:head(raw.misc,row,10),
        moremisc:head(raw.moremisc,row,8),ownership:head(raw.ownership,row,361)};
    let mut input=NNResultBuf::default();
    input.symmetry=params.symmetry;input.policy_optimism=params.policy_optimism;input.include_owner_map=include_ownership;
    let mut output=NNOutput::default();
    decode_raw_outputs(19,19,1,&[&mut input],host,&mut [&mut output]);
    Ok(output)
}

/// Shared mathematical operation. Callers supply a real model's parameters and
/// full legal history; Worker exposes a stricter byte/PB-owned wrapper instead.
#[allow(clippy::too_many_arguments)]
pub fn postprocess_output(model_version: i32, pp: ModelPostProcessParams,
    nn_x_len: i32, nn_y_len: i32, policy_size: i32,
    board: &Board, history: &BoardHistory, next_player: Player,
    nn_input_params: &MiscNNInputParams, output: &mut NNOutput) {
    let policy_size = policy_size as usize;
    let x_size = board.x_size;
    let y_size = board.y_size;

    // --- Policy ---------------------------------------------------------
    let policy_output_scaling = pp.output_scale_multiplier
        / nn_input_params.nn_policy_temperature.clamp(1e-6, 1e6);

    let mut is_legal = vec![false; policy_size];
    let mut legal_count = 0usize;
    for i in 0..policy_size {
        let loc = nn_pos::pos_to_loc(
            i as i32,
            x_size,
            y_size,
            nn_x_len,
            nn_y_len,
        );
        is_legal[i] = history.is_legal(board, loc, next_player);
    }
    // TODO(selfplay): the C++ avoidMYTDaggerHack dagger-match ban is not
    // ported here; it only affects selfplay training, not GTP play.

    let mut max_policy = -1e25f32;
    for i in 0..policy_size {
        let v = if is_legal[i] {
            legal_count += 1;
            output.policy_probs[i] * policy_output_scaling
        } else {
            -1e30f32
        };
        output.policy_probs[i] = v;
        if v > max_policy {
            max_policy = v;
        }
    }

    let mut policy_sum = 0.0f32;
    if nn_input_params.enable_passing_hacks {
        // Cap passing prior policy at 95% (19x other moves).
        let max_pass_policy_sum_factor = 19.0f32;
        for i in 0..policy_size - 1 {
            let v = (output.policy_probs[i] - max_policy).exp();
            output.policy_probs[i] = v;
            policy_sum += v;
        }
        let i = policy_size - 1;
        let v = (output.policy_probs[i] - max_policy)
            .exp()
            .min(policy_sum * max_pass_policy_sum_factor)
            .max(1e-20);
        output.policy_probs[i] = v;
        policy_sum += v;
    } else {
        for v in output.policy_probs.iter_mut().take(policy_size) {
            *v = (*v - max_policy).exp();
            policy_sum += *v;
        }
    }

    if policy_sum <= 0.0 {
        // Somehow all legal moves rounded to 0 probability.
        let uniform = 1.0f32 / legal_count.max(1) as f32;
        for i in 0..policy_size {
            output.policy_probs[i] = if is_legal[i] { uniform } else { -1.0 };
        }
    } else {
        for i in 0..policy_size {
            output.policy_probs[i] = if is_legal[i] {
                output.policy_probs[i] / policy_sum
            } else {
                -1.0
            };
        }
    }
    for v in output.policy_probs.iter_mut().skip(policy_size) {
        *v = -1.0f32;
    }
    output.policy_optimism_used = nn_input_params.policy_optimism as f32;

    // --- Value / score (model version >= 4) ------------------------------
    if model_version >= 4 {
        let win_logits = output.white_win_prob as f64 * pp.output_scale_multiplier as f64;
        let loss_logits = output.white_loss_prob as f64 * pp.output_scale_multiplier as f64;
        let mut no_result_logits =
            output.white_no_result_prob as f64 * pp.output_scale_multiplier as f64;
        let score_mean_pre = output.white_score_mean as f64 * pp.output_scale_multiplier as f64;
        let score_stdev_pre =
            output.white_score_mean_sq as f64 * pp.output_scale_multiplier as f64;
        let lead_pre = output.white_lead as f64 * pp.output_scale_multiplier as f64;
        let var_time_pre = output.var_time_left as f64 * pp.output_scale_multiplier as f64;
        let swin_pre =
            output.shortterm_winloss_error as f64 * pp.output_scale_multiplier as f64;
        let sscore_pre =
            output.shortterm_score_error as f64 * pp.output_scale_multiplier as f64;

        if history.rules.ko_rule != KoRule::Simple
            && history.rules.scoring_rule != ScoringRule::Territory
        {
            no_result_logits -= 100000.0;
        }

        let max_logits = win_logits.max(loss_logits).max(no_result_logits);
        let mut win_prob = (win_logits - max_logits).exp();
        let mut loss_prob = (loss_logits - max_logits).exp();
        let mut no_result_prob = (no_result_logits - max_logits).exp();
        if history.rules.ko_rule != KoRule::Simple
            && history.rules.scoring_rule != ScoringRule::Territory
        {
            no_result_prob = 0.0;
        }
        let prob_sum = win_prob + loss_prob + no_result_prob;
        win_prob /= prob_sum;
        loss_prob /= prob_sum;
        no_result_prob /= prob_sum;

        let mut score_mean = score_mean_pre * pp.score_mean_multiplier;
        let score_stdev = softplus(score_stdev_pre) * pp.score_stdev_multiplier;
        let mut score_mean_sq = score_mean * score_mean + score_stdev * score_stdev;
        let mut lead = lead_pre * pp.lead_multiplier;
        let var_time_left = softplus(var_time_pre) * pp.variance_time_multiplier;
        // No-result counts as 0 score for score-value purposes.
        score_mean *= 1.0 - no_result_prob;
        score_mean_sq *= 1.0 - no_result_prob;
        lead *= 1.0 - no_result_prob;

        let (shortterm_winloss_error, shortterm_score_error) = if model_version >= 14 {
            let s1 = softplus(swin_pre * 0.5);
            let s2 = softplus(sscore_pre * 0.5);
            (
                (s1 * s1 * pp.shortterm_value_error_multiplier).sqrt(),
                (s2 * s2 * pp.shortterm_score_error_multiplier).sqrt(),
            )
        } else if model_version >= 10 {
            (
                (softplus(swin_pre) * pp.shortterm_value_error_multiplier).sqrt(),
                (softplus(sscore_pre) * pp.shortterm_score_error_multiplier).sqrt(),
            )
        } else {
            (softplus(swin_pre), softplus(sscore_pre) * 10.0)
        };

        // Flip from player-to-move to white's perspective.
        if next_player == P_WHITE {
            output.white_win_prob = win_prob as f32;
            output.white_loss_prob = loss_prob as f32;
            output.white_no_result_prob = no_result_prob as f32;
            output.white_score_mean = score_mean as f32;
            output.white_score_mean_sq = score_mean_sq as f32;
            output.white_lead = lead as f32;
        } else {
            output.white_win_prob = loss_prob as f32;
            output.white_loss_prob = win_prob as f32;
            output.white_no_result_prob = no_result_prob as f32;
            output.white_score_mean = -(score_mean as f32);
            output.white_score_mean_sq = score_mean_sq as f32;
            output.white_lead = -(lead as f32);
        }
        if model_version >= 9 {
            output.var_time_left = var_time_left as f32;
            output.shortterm_winloss_error = shortterm_winloss_error as f32;
            output.shortterm_score_error = shortterm_score_error as f32;
        } else {
            output.var_time_left = -1.0;
            output.shortterm_winloss_error = -1.0;
            output.shortterm_score_error = -1.0;
        }
    }

    // --- Ownership ---------------------------------------------------------
    if let Some(map) = &mut output.white_owner_map {
        let s = pp.output_scale_multiplier;
        for pos in 0..(nn_x_len * nn_y_len) as usize {
            let y = pos as i32 / nn_x_len;
            let x = pos as i32 % nn_x_len;
            if y >= board.y_size || x >= board.x_size {
                map[pos] = 0.0f32;
            } else {
                // Same as value: flip player-to-move → white and tanh.
                let v = map[pos] * s;
                map[pos] = if next_player == P_WHITE { v.tanh() } else { -v.tanh() };
            }
        }
    }
}

fn softplus(x: f64) -> f64 {
    if x > 40.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[cfg(test)]
#[path = "output_postprocess_tests.rs"]
mod tests;
