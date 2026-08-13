//! Elo rating computation.
//!
//! Corresponds to `cpp/core/elo.h` and `cpp/core/elo.cpp`.

use std::io::Write as IoWrite;

const ELO_PER_LOG_GAMMA: f64 = 173.717_792_761;

/// Win/loss record between two players.
#[derive(Debug, Clone, Copy, Default)]
pub struct WlRecord {
    pub first_wins: f64,
    pub second_wins: f64,
}

impl WlRecord {
    /// Create a new record.
    pub fn new(first_wins: f64, second_wins: f64) -> Self {
        Self {
            first_wins,
            second_wins,
        }
    }
}

/// Return the probability of the first player winning given an Elo difference.
pub fn prob_win(elo_diff: f64) -> f64 {
    let log_gamma_diff = elo_diff / ELO_PER_LOG_GAMMA;
    1.0 / (1.0 + (-log_gamma_diff).exp())
}

fn log_one_plus_exp_x(x: f64) -> f64 {
    if x >= 50.0 {
        50.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[allow(dead_code)]
fn log_one_plus_exp_x_second_derivative(x: f64) -> f64 {
    let half_x = 0.5 * x;
    let denom = half_x.exp() + (-half_x).exp();
    1.0 / (denom * denom)
}

fn log_likelihood_of_wl(elo_first_minus_second: f64, win_record: WlRecord) -> f64 {
    let log_gamma_first_minus_second = elo_first_minus_second / ELO_PER_LOG_GAMMA;
    let log_prob_first_win = -log_one_plus_exp_x(-log_gamma_first_minus_second);
    let log_prob_second_win = -log_one_plus_exp_x(log_gamma_first_minus_second);
    win_record.first_wins * log_prob_first_win + win_record.second_wins * log_prob_second_win
}

#[allow(dead_code)]
fn log_likelihood_of_wl_second_derivative(
    elo_first_minus_second: f64,
    win_record: WlRecord,
) -> f64 {
    let log_gamma_first_minus_second = elo_first_minus_second / ELO_PER_LOG_GAMMA;
    let log_prob_first_win_second_derivative =
        -log_one_plus_exp_x_second_derivative(-log_gamma_first_minus_second);
    let log_prob_second_win_second_derivative =
        -log_one_plus_exp_x_second_derivative(log_gamma_first_minus_second);
    (win_record.first_wins * log_prob_first_win_second_derivative
        + win_record.second_wins * log_prob_second_win_second_derivative)
        / (ELO_PER_LOG_GAMMA * ELO_PER_LOG_GAMMA)
}

fn compute_local_log_likelihood(
    player: usize,
    elos: &[f64],
    win_matrix: &[WlRecord],
    num_players: usize,
    prior_wl: f64,
) -> f64 {
    let mut log_likelihood = 0.0;
    for y in 0..num_players {
        if y == player {
            continue;
        }
        log_likelihood +=
            log_likelihood_of_wl(elos[player] - elos[y], win_matrix[player * num_players + y]);
        log_likelihood +=
            log_likelihood_of_wl(elos[y] - elos[player], win_matrix[y * num_players + player]);
    }
    log_likelihood += log_likelihood_of_wl(elos[player], WlRecord::new(prior_wl, prior_wl));
    log_likelihood
}

#[allow(dead_code)]
fn compute_local_log_likelihood_second_derivative(
    player: usize,
    elos: &[f64],
    win_matrix: &[WlRecord],
    num_players: usize,
    prior_wl: f64,
) -> f64 {
    let mut log_likelihood_second_derivative = 0.0;
    for y in 0..num_players {
        if y == player {
            continue;
        }
        log_likelihood_second_derivative += log_likelihood_of_wl_second_derivative(
            elos[player] - elos[y],
            win_matrix[player * num_players + y],
        );
        log_likelihood_second_derivative += log_likelihood_of_wl_second_derivative(
            elos[y] - elos[player],
            win_matrix[y * num_players + player],
        );
    }
    log_likelihood_second_derivative +=
        log_likelihood_of_wl_second_derivative(elos[player], WlRecord::new(prior_wl, prior_wl));
    log_likelihood_second_derivative
}

/// Compute Elo ratings for `num_players` from a win matrix.
///
/// `win_matrix[a * num_players + b]` is the record `a` has versus `b` when `a`
/// plays first. `prior_wl` is the number of wins and losses against a virtual
/// 0-Elo opponent.
pub fn compute_elos(
    win_matrix: &[WlRecord],
    num_players: usize,
    prior_wl: f64,
    max_iters: i32,
    tolerance: f64,
    mut out: Option<&mut dyn IoWrite>,
) -> Vec<f64> {
    let mut elos = vec![0.0; num_players];
    let mut next_delta = vec![100.0; num_players];

    for i in 0..max_iters {
        let mut max_elo_diff: f64 = 0.0;
        for x in 0..num_players {
            let old_elo = elos[x];
            let hi_elo = old_elo + next_delta[x];
            let lo_elo = old_elo - next_delta[x];

            let likelihood =
                compute_local_log_likelihood(x, &elos, win_matrix, num_players, prior_wl);
            elos[x] = hi_elo;
            let likelihood_hi =
                compute_local_log_likelihood(x, &elos, win_matrix, num_players, prior_wl);
            elos[x] = lo_elo;
            let likelihood_lo =
                compute_local_log_likelihood(x, &elos, win_matrix, num_players, prior_wl);

            if likelihood_hi > likelihood {
                elos[x] = hi_elo;
                next_delta[x] *= 1.1;
            } else if likelihood_lo > likelihood {
                elos[x] = lo_elo;
                next_delta[x] *= 1.1;
            } else {
                elos[x] = old_elo;
                next_delta[x] *= 0.8;
            }

            max_elo_diff = max_elo_diff.max(next_delta[x]);
        }

        if let Some(out) = out.as_mut() {
            if i % 50 == 0 {
                writeln!(out, "Iteration {i} maxEloDiff = {max_elo_diff}").unwrap();
            }
        }

        if max_elo_diff < tolerance {
            break;
        }
    }

    elos
}

/// Approximately compute the standard deviation of all players' Elos, assuming
/// each time that all other player Elos are completely confident.
pub fn compute_approx_elo_stdevs(
    elos: &[f64],
    win_matrix: &[WlRecord],
    num_players: usize,
    prior_wl: f64,
) -> Vec<f64> {
    let radius = 1500;
    let step = 1.0;
    let mut rel_probs = vec![0.0; radius * 2 + 1];
    let mut elo_stdevs = vec![0.0; num_players];
    let mut temp_elos = elos.to_vec();

    for player in 0..num_players {
        let log_likelihood =
            compute_local_log_likelihood(player, &temp_elos, win_matrix, num_players, prior_wl);
        let mut sum_rel_probs = 0.0;
        for (i, rel_prob) in rel_probs.iter_mut().enumerate().take(radius * 2 + 1) {
            let elo = elos[player] + (i as f64 - radius as f64) * step;
            temp_elos[player] = elo;
            let new_log_likelihood =
                compute_local_log_likelihood(player, &temp_elos, win_matrix, num_players, prior_wl);
            *rel_prob = (new_log_likelihood - log_likelihood).exp();
            sum_rel_probs += *rel_prob;
        }

        let mut second_moment_around_elo = 0.0;
        for (i, &rel_prob) in rel_probs.iter().enumerate().take(radius * 2 + 1) {
            let elo = elos[player] + (i as f64 - radius as f64) * step;
            second_moment_around_elo +=
                rel_prob / sum_rel_probs * (elo - elos[player]) * (elo - elos[player]);
        }
        elo_stdevs[player] = second_moment_around_elo.sqrt();

        temp_elos[player] = elos[player];
    }

    elo_stdevs
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::identity_op, clippy::erasing_op)]
mod tests {
    use super::*;

    fn approx_equal(x: f64, y: f64, tolerance: f64) -> bool {
        (x - y).abs() < tolerance
    }

    fn run_test(num_players: usize, prior_wl: f64, max_iters: i32, expected: &[f64]) {
        let win_matrix = vec![WlRecord::default(); num_players * num_players];
        let elos = compute_elos(
            &win_matrix,
            num_players,
            prior_wl,
            max_iters,
            0.000_01,
            None,
        );
        assert_eq!(elos.len(), num_players);
        for (i, &expected_elo) in expected.iter().enumerate() {
            assert!(
                approx_equal(elos[i], expected_elo, 0.01),
                "elo[{i}] = {} expected {expected_elo}",
                elos[i]
            );
        }
    }

    #[test]
    fn test_elo_0() {
        run_test(1, 0.1, 1000, &[0.0]);
    }

    #[test]
    fn test_elo_1() {
        let num_players = 3;
        let mut win_matrix = vec![WlRecord::default(); num_players * num_players];
        win_matrix[1 * num_players + 2] = WlRecord::new(200.0, 0.0);
        win_matrix[2 * num_players + 1] = WlRecord::new(100.0, 0.0);
        let elos = compute_elos(&win_matrix, num_players, 1.0, 1000, 0.000_01, None);
        assert!(approx_equal(elos[0], 0.0, 0.01));
        assert!(approx_equal(elos[1], 59.983_3, 0.01));
        assert!(approx_equal(elos[2], -59.983_4, 0.01));
    }

    #[test]
    fn test_elo_2() {
        let num_players = 3;
        let mut win_matrix = vec![WlRecord::default(); num_players * num_players];
        win_matrix[0 * num_players + 2] = WlRecord::new(0.0, 1.0);
        win_matrix[1 * num_players + 2] = WlRecord::new(5.0, 0.0);
        win_matrix[2 * num_players + 0] = WlRecord::new(0.0, 5.0);
        win_matrix[2 * num_players + 1] = WlRecord::new(1.0, 0.0);
        let elos = compute_elos(&win_matrix, num_players, 1.0, 1000, 0.000_01, None);
        assert!(approx_equal(elos[0], 76.522_8, 0.01));
        assert!(approx_equal(elos[1], 76.522_8, 0.01));
        assert!(approx_equal(elos[2], -161.285, 0.01));
    }

    #[test]
    fn test_elo_3() {
        let num_players = 3;
        let mut win_matrix = vec![WlRecord::default(); num_players * num_players];
        win_matrix[1 * num_players + 2] = WlRecord::new(0.0, 1.0);
        win_matrix[2 * num_players + 0] = WlRecord::new(5.0, 1.0);
        win_matrix[2 * num_players + 1] = WlRecord::new(0.0, 5.0);
        let elos = compute_elos(&win_matrix, num_players, 1.0, 1000, 0.000_01, None);
        assert!(approx_equal(elos[0], -190.849, 0.01));
        assert!(approx_equal(elos[1], 190.849, 0.01));
        assert!(approx_equal(elos[2], 0.0, 0.01));
    }

    #[test]
    fn test_elo_4() {
        let num_players = 3;
        let mut win_matrix = vec![WlRecord::default(); num_players * num_players];
        win_matrix[1 * num_players + 2] = WlRecord::new(0.0, 1.0);
        win_matrix[2 * num_players + 0] = WlRecord::new(5.0, 1.0);
        win_matrix[2 * num_players + 1] = WlRecord::new(0.0, 5.0);
        let elos = compute_elos(&win_matrix, num_players, 0.1, 10_000, 0.000_01, None);
        assert!(approx_equal(elos[0], -266.471, 0.01));
        assert!(approx_equal(elos[1], 266.471, 0.01));
        assert!(approx_equal(elos[2], 0.0, 0.01));
    }

    #[test]
    fn test_elo_5() {
        let num_players = 3;
        let mut win_matrix = vec![WlRecord::default(); num_players * num_players];
        win_matrix[1 * num_players + 2] = WlRecord::new(0.0, 1.0);
        win_matrix[2 * num_players + 0] = WlRecord::new(7.0, 1.0);
        win_matrix[2 * num_players + 1] = WlRecord::new(0.0, 5.0);
        let elos = compute_elos(&win_matrix, num_players, 0.01, 10_000, 0.000_01, None);
        assert!(approx_equal(elos[0], -322.013, 0.01));
        assert!(approx_equal(elos[1], 292.742, 0.01));
        assert!(approx_equal(elos[2], 14.582_8, 0.01));
    }

    #[test]
    fn test_prob_win() {
        assert!(approx_equal(prob_win(0.0), 0.5, 1e-9));
        assert!(prob_win(1000.0) > 0.5);
        assert!(prob_win(-1000.0) < 0.5);
    }
}
