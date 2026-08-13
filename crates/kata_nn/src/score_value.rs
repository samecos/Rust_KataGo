//! Score-value utilities for converting expected score to utility.
//!
//! Corresponds to the `ScoreValue` namespace in `cpp/neuralnet/nninputs.h` and
//! `cpp/neuralnet/nninputs.cpp`. The functions needed by the search and by the
//! score tests are ported here.

use std::f64::consts::{FRAC_PI_2, PI};
use std::sync::OnceLock;

use kata_game::board::{C_EMPTY, P_BLACK, P_WHITE, Player};
use kata_game::history::BoardHistory;

use crate::inputs::nn_pos;

const TWO_OVER_PI: f64 = 2.0 / PI;
const STEPS_PER_UNIT: i32 = 10;
const BOUND_STDEVS: i32 = 5;

/// Precomputed table for `expected_white_score_value`.
///
/// The table is flattened as `table[mean_idx * stdev_len + stdev_idx]`.
struct ScoreValueTable {
    data: Vec<f64>,
    mean_radius: i32,
    mean_len: usize,
    stdev_len: usize,
    assumed_b_size: i32,
}

impl ScoreValueTable {
    fn new() -> Self {
        let assumed_b_size = nn_pos::MAX_BOARD_LEN as i32;
        let mean_radius = assumed_b_size * assumed_b_size + nn_pos::EXTRA_SCORE_DISTR_RADIUS;
        let mean_len = (mean_radius * 2) as usize;
        let stdev_len =
            (assumed_b_size * assumed_b_size + nn_pos::EXTRA_SCORE_DISTR_RADIUS) as usize;

        let mut data = vec![0.0f64; mean_len * stdev_len];

        // Precompute normal PDF weights.
        let min_stdev_steps = -BOUND_STDEVS * STEPS_PER_UNIT;
        let max_stdev_steps = BOUND_STDEVS * STEPS_PER_UNIT;
        let mut normal_pdf = vec![0.0f64; (max_stdev_steps - min_stdev_steps + 1) as usize];
        for i in min_stdev_steps..=max_stdev_steps {
            let x_in_stdevs = i as f64 / STEPS_PER_UNIT as f64;
            normal_pdf[(i - min_stdev_steps) as usize] = (-0.5 * x_in_stdevs * x_in_stdevs).exp();
        }

        // Precompute score value at increments of 1/steps_per_unit points.
        let min_sv_steps = -(mean_radius * STEPS_PER_UNIT
            + STEPS_PER_UNIT / 2
            + BOUND_STDEVS * stdev_len as i32 * STEPS_PER_UNIT);
        let max_sv_steps = -min_sv_steps;
        let mut sv_precomp = vec![0.0f64; (max_sv_steps - min_sv_steps + 1) as usize];
        for i in min_sv_steps..=max_sv_steps {
            let mean = i as f64 / STEPS_PER_UNIT as f64;
            sv_precomp[(i - min_sv_steps) as usize] =
                white_score_value_of_score_smooth_no_draw_adjust(
                    mean,
                    0.0,
                    1.0,
                    assumed_b_size as f64,
                );
        }

        // Numerically integrate over a normal distribution for each table cell.
        for mean_idx in 0..mean_len {
            let mean_steps = (mean_idx as i32 - mean_radius) * STEPS_PER_UNIT - STEPS_PER_UNIT / 2;
            for stdev_idx in 0..stdev_len {
                let mut w_sum = 0.0;
                let mut wsv_sum = 0.0;
                for i in min_stdev_steps..=max_stdev_steps {
                    let x_steps = mean_steps + stdev_idx as i32 * i;
                    debug_assert!(
                        x_steps >= min_sv_steps && x_steps <= max_sv_steps,
                        "x_steps {} out of range [{}, {}]",
                        x_steps,
                        min_sv_steps,
                        max_sv_steps
                    );
                    let w = normal_pdf[(i - min_stdev_steps) as usize];
                    let sv = sv_precomp[(x_steps - min_sv_steps) as usize];
                    w_sum += w;
                    wsv_sum += w * sv;
                }
                data[mean_idx * stdev_len + stdev_idx] = wsv_sum / w_sum;
            }
        }

        Self {
            data,
            mean_radius,
            mean_len,
            stdev_len,
            assumed_b_size,
        }
    }
}

fn global_table() -> &'static ScoreValueTable {
    static TABLE: OnceLock<ScoreValueTable> = OnceLock::new();
    TABLE.get_or_init(ScoreValueTable::new)
}

/// The unscaled utility of a deterministic score difference.
pub fn white_score_value_of_score_smooth_no_draw_adjust(
    final_white_minus_black_score: f64,
    center: f64,
    scale: f64,
    sqrt_board_area: f64,
) -> f64 {
    let adjusted_score = final_white_minus_black_score - center;
    adjusted_score.atan2(scale * sqrt_board_area) * TWO_OVER_PI
}

/// The unscaled utility of a deterministic score difference, applying the
/// draw-equivalent komi adjustment from `hist`.
///
/// Mirrors `ScoreValue::whiteScoreValueOfScoreSmooth` in `cpp/neuralnet/nninputs.cpp`.
pub fn white_score_value_of_score_smooth(
    final_white_minus_black_score: f64,
    center: f64,
    scale: f64,
    draw_equivalent_wins_for_white: f64,
    sqrt_board_area: f64,
    hist: &BoardHistory,
) -> f64 {
    let adjusted_score = final_white_minus_black_score
        + f64::from(hist.white_komi_adjustment_for_draws(draw_equivalent_wins_for_white))
        - center;
    adjusted_score.atan2(scale * sqrt_board_area) * TWO_OVER_PI
}

/// Expected unscaled utility of the final score difference, given the mean and
/// standard deviation of a roughly-normal distribution.
///
/// `sqrt_board_area` should be `sqrt(x_size * y_size)`.
pub fn expected_white_score_value(
    white_score_mean: f64,
    white_score_stdev: f64,
    center: f64,
    scale: f64,
    sqrt_board_area: f64,
) -> f64 {
    let table = global_table();
    let scale_factor = table.assumed_b_size as f64 / (scale * sqrt_board_area);

    let mean_scaled = (white_score_mean - center) * scale_factor;
    let stdev_scaled = white_score_stdev * scale_factor;

    let mean_rounded = mean_scaled.round();
    let stdev_floored = stdev_scaled.floor();

    let mut mean_idx0 = mean_rounded as i32 + table.mean_radius;
    let mut mean_idx1 = mean_idx0 + 1;
    if mean_idx0 < 0 {
        mean_idx0 = 0;
        mean_idx1 = 0;
    }
    if mean_idx1 >= table.mean_len as i32 {
        mean_idx0 = table.mean_len as i32 - 1;
        mean_idx1 = mean_idx0;
    }

    let mut stdev_idx0 = stdev_floored as i32;
    let mut stdev_idx1 = stdev_idx0 + 1;
    debug_assert!(stdev_idx0 >= 0);
    if stdev_idx1 >= table.stdev_len as i32 {
        stdev_idx0 = table.stdev_len as i32 - 1;
        stdev_idx1 = stdev_idx0;
    }

    let lambda_mean = mean_scaled - mean_rounded + 0.5;
    let lambda_stdev = stdev_scaled - stdev_floored;

    let a00 = table.data[mean_idx0 as usize * table.stdev_len + stdev_idx0 as usize];
    let a01 = table.data[mean_idx0 as usize * table.stdev_len + stdev_idx1 as usize];
    let a10 = table.data[mean_idx1 as usize * table.stdev_len + stdev_idx0 as usize];
    let a11 = table.data[mean_idx1 as usize * table.stdev_len + stdev_idx1 as usize];

    let b0 = a00 + lambda_stdev * (a01 - a00);
    let b1 = a10 + lambda_stdev * (a11 - a10);
    b0 + lambda_mean * (b1 - b0)
}

/// Compute the standard deviation of score given `E(score)` and `E(score^2)`.
pub fn get_score_stdev(score_mean: f64, score_mean_sq: f64) -> f64 {
    let variance = score_mean_sq - score_mean * score_mean;
    if variance <= 0.0 {
        0.0
    } else {
        variance.sqrt()
    }
}

/// Convert a game winner to the probability that white wins.
pub fn white_wins_of_winner(winner: Player, draw_equivalent_wins_for_white: f64) -> f64 {
    match winner {
        P_WHITE => 1.0,
        P_BLACK => 0.0,
        _ => {
            debug_assert_eq!(winner, C_EMPTY);
            draw_equivalent_wins_for_white
        }
    }
}

/// Adjust a final score by the komi/draw-equivalent term.
pub fn white_score_draw_adjust(
    final_white_minus_black_score: f64,
    draw_equivalent_wins_for_white: f64,
    hist: &BoardHistory,
) -> f64 {
    final_white_minus_black_score
        + f64::from(hist.white_komi_adjustment_for_draws(draw_equivalent_wins_for_white))
}

/// Compute the second moment of the final score around a gridded distribution.
pub fn white_score_mean_sq_of_score_gridded(
    final_white_minus_black_score: f64,
    draw_equivalent_wins_for_white: f64,
) -> f64 {
    debug_assert!(
        (final_white_minus_black_score * 2.0).round() == final_white_minus_black_score * 2.0
    );
    let final_score_is_integer =
        (final_white_minus_black_score.round() as i64) == final_white_minus_black_score as i64;
    if !final_score_is_integer {
        return final_white_minus_black_score * final_white_minus_black_score;
    }

    let lower = final_white_minus_black_score - 0.5;
    let upper = final_white_minus_black_score + 0.5;
    let lower_sq = lower * lower;
    let upper_sq = upper * upper;
    lower_sq + (upper_sq - lower_sq) * draw_equivalent_wins_for_white
}

/// Approximate inverse of `white_score_value_of_score_smooth_no_draw_adjust`.
#[allow(dead_code)]
pub fn approx_white_score_of_score_value_smooth(
    score_value: f64,
    center: f64,
    scale: f64,
    sqrt_board_area: f64,
) -> f64 {
    debug_assert!((-1.0..=1.0).contains(&score_value));
    let unscaled = (score_value * FRAC_PI_2).tan().clamp(-1e6, 1e6);
    unscaled * (scale * sqrt_board_area) + center
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_score_stdev() {
        assert!((get_score_stdev(3.0, 10.0) - 1.0).abs() < 1e-9);
        assert_eq!(get_score_stdev(3.0, 9.0), 0.0);
    }

    #[test]
    fn test_expected_score_value_zero_stdev() {
        // With zero stdev the expected score value should equal the score value
        // at the mean (modulo the table's 0.05-point cell midpoint).
        let sqrt_area = 19.0f64;
        let mean = 2.0;
        let expected = expected_white_score_value(mean, 0.0, 1.0, 2.0, sqrt_area);
        let direct =
            white_score_value_of_score_smooth_no_draw_adjust(mean - 0.05, 2.0, 1.0, sqrt_area);
        assert!((expected - direct).abs() < 0.02);
    }

    #[test]
    fn test_expected_score_value_bounds() {
        let sqrt_area = 19.0f64;
        let v = expected_white_score_value(0.0, 10.0, 0.0, 2.0, sqrt_area);
        assert!(v > -1.0 && v < 1.0);
    }

    #[test]
    fn test_approx_inverse() {
        let sqrt_area = 19.0f64;
        for score in [-10.0, -2.0, 0.0, 3.0, 15.0] {
            let sv = white_score_value_of_score_smooth_no_draw_adjust(score, 0.0, 1.0, sqrt_area);
            let approx = approx_white_score_of_score_value_smooth(sv, 0.0, 1.0, sqrt_area);
            assert!(
                (approx - score).abs() < 0.1,
                "score {} approx {} sv {}",
                score,
                approx,
                sv
            );
        }
    }
}
