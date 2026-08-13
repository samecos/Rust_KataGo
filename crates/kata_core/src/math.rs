//! Mathematical special functions used by KataGo.
//!
//! Corresponds to `cpp/core/fancymath.h` and `cpp/core/fancymath.cpp`.

pub mod bsearch;

use std::f64::consts::PI;

/// Compute `ln(gamma(x))` for positive `x`.
fn ln_gamma(x: f64) -> f64 {
    libm::lgamma(x)
}

/// Evaluate a continued fraction using the modified Lentz algorithm.
pub fn evaluate_continued_fraction<N, D>(numer: N, denom: D, tolerance: f64, max_terms: i32) -> f64
where
    N: Fn(i32) -> f64,
    D: Fn(i32) -> f64,
{
    let tiny = 1e-300;
    let mut ret = denom(0);
    if ret == 0.0 {
        ret = tiny;
    }
    let mut c = ret;
    let mut d = 0.0;

    for n in 1..max_terms {
        let next_numer = numer(n);
        let next_denom = denom(n);
        d = next_denom + next_numer * d;
        if d == 0.0 {
            d = tiny;
        }
        c = next_denom + next_numer / c;
        if c == 0.0 {
            c = tiny;
        }
        d = 1.0 / d;
        let mult = c * d;
        ret *= mult;
        if (mult - 1.0).abs() <= tolerance {
            break;
        }
    }
    ret
}

/// Textbook continued fraction terms for the incomplete beta function.
fn incomplete_beta_continued_fraction(x: f64, a: f64, b: f64) -> f64 {
    let numer = |n: i32| -> f64 {
        if n % 2 == 0 {
            let m = f64::from(n) / 2.0;
            m * (b - m) * x / (a + 2.0 * m - 1.0) / (a + 2.0 * m)
        } else {
            let m = (f64::from(n) - 1.0) / 2.0;
            -(a + m) * (a + b + m) * x / (a + 2.0 * m) / (a + 2.0 * m + 1.0)
        }
    };
    let denom = |_n: i32| -> f64 { 1.0 };
    evaluate_continued_fraction(numer, denom, 1e-15, 100_000)
}

/// Beta function `B(a, b)`.
pub fn beta(a: f64, b: f64) -> f64 {
    (ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)).exp()
}

/// Natural logarithm of the beta function.
pub fn log_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Incomplete beta function `B(x; a, b)`.
pub fn incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if !((0.0..=1.0).contains(&x) && a > 0.0 && b > 0.0) {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return beta(a, b);
    }
    let logx = x.ln();
    let logy = (1.0 - x).ln();
    if x <= (a + 1.0) / (a + b + 2.0) {
        (logx * a + logy * b).exp() / a / incomplete_beta_continued_fraction(x, a, b)
    } else {
        beta(a, b)
            - (logy * b + logx * a).exp() / b / incomplete_beta_continued_fraction(1.0 - x, b, a)
    }
}

/// Regularized incomplete beta function `I_x(a, b)`.
pub fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if !((0.0..=1.0).contains(&x) && a > 0.0 && b > 0.0) {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let logx = x.ln();
    let logy = (1.0 - x).ln();
    if x <= (a + 1.0) / (a + b + 2.0) {
        (logx * a + logy * b - log_beta(a, b)).exp()
            / a
            / incomplete_beta_continued_fraction(x, a, b)
    } else {
        1.0 - (logy * b + logx * a - log_beta(a, b)).exp()
            / b
            / incomplete_beta_continued_fraction(1.0 - x, b, a)
    }
}

/// Probability density function of Student's t-distribution.
pub fn t_dist_pdf(x: f64, degrees_of_freedom: f64) -> f64 {
    let v = degrees_of_freedom;
    if v <= 0.0 || v.is_nan() {
        return f64::NAN;
    }
    1.0 / (v * PI).sqrt()
        / (ln_gamma(v / 2.0) - ln_gamma((v + 1.0) / 2.0)).exp()
        / (1.0 + x * x / v).powf((v + 1.0) / 2.0)
}

/// Cumulative distribution function of Student's t-distribution.
pub fn t_dist_cdf(x: f64, degrees_of_freedom: f64) -> f64 {
    let v = degrees_of_freedom;
    if v <= 0.0 || v.is_nan() {
        return f64::NAN;
    }
    if x >= 0.0 {
        1.0 - regularized_incomplete_beta(v / (x * x + v), v / 2.0, 0.5) / 2.0
    } else {
        regularized_incomplete_beta(v / (x * x + v), v / 2.0, 0.5) / 2.0
    }
}

/// Probability density function of the Beta distribution.
pub fn beta_pdf(x: f64, a: f64, b: f64) -> f64 {
    if !((0.0..=1.0).contains(&x) && a > 0.0 && b > 0.0) {
        return f64::NAN;
    }
    if x == 0.0 {
        return if a < 1.0 {
            f64::INFINITY
        } else if a > 1.0 {
            0.0
        } else {
            1.0 / beta(a, b)
        };
    }
    if x == 1.0 {
        return if b < 1.0 {
            f64::INFINITY
        } else if b > 1.0 {
            0.0
        } else {
            1.0 / beta(a, b)
        };
    }
    (-log_beta(a, b) + x.ln() * (a - 1.0) + (1.0 - x).ln() * (b - 1.0)).exp()
}

/// Cumulative distribution function of the Beta distribution.
pub fn beta_cdf(x: f64, a: f64, b: f64) -> f64 {
    regularized_incomplete_beta(x, a, b)
}

/// Given `z`, approximate `t` such that `P(T_df > t) == P(N(0,1) > z)`.
pub fn norm_to_t_approx(z: f64, degrees_of_freedom: f64) -> f64 {
    let n = degrees_of_freedom;
    (n * ((z * z * (n - 1.5) / ((n - 1.0) * (n - 1.0))).exp_m1())).sqrt()
}

/// Clamped binary cross-entropy.
pub fn binary_cross_entropy(pred_prob: f64, target_prob: f64, epsilon: f64) -> f64 {
    let mut reverse_prob = 1.0 - pred_prob;
    let mut pred_prob = epsilon * (1.0 - pred_prob) + (1.0 - epsilon) * pred_prob;
    reverse_prob = epsilon * (1.0 - reverse_prob) + (1.0 - epsilon) * reverse_prob;

    if pred_prob < epsilon {
        pred_prob = epsilon;
    }
    if pred_prob > 1.0 - epsilon {
        pred_prob = 1.0 - epsilon;
    }
    if reverse_prob < epsilon {
        reverse_prob = epsilon;
    }
    if reverse_prob > 1.0 - epsilon {
        reverse_prob = 1.0 - epsilon;
    }

    target_prob * (-pred_prob.ln()) + (1.0 - target_prob) * (-reverse_prob.ln())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    fn approx_eq(x: f64, y: f64, tolerance: f64) {
        let max_diff = tolerance * x.abs().max(y.abs().max(1.0));
        assert!(
            (x - y).abs() <= max_diff,
            "Failed approx equal: {:.17} vs {:.17}",
            x,
            y
        );
    }

    #[test]
    fn test_continued_fraction() {
        let x = (1.0 + 5f64.sqrt()) / 2.0;
        let y = evaluate_continued_fraction(|_n| 1.0, |_n| 1.0, 1e-15, 1000);
        approx_eq(x, y, 1e-14);

        let x = 2f64.sqrt();
        let y =
            evaluate_continued_fraction(|_n| 1.0, |n| if n == 0 { 1.0 } else { 2.0 }, 1e-15, 1000);
        approx_eq(x, y, 1e-14);

        let x = std::f64::consts::E;
        let y = evaluate_continued_fraction(
            |_n| 1.0,
            |n| {
                if n == 0 {
                    2.0
                } else if n % 3 == 2 {
                    f64::from(n + 1) / 3.0 * 2.0
                } else {
                    1.0
                }
            },
            1e-15,
            1000,
        );
        approx_eq(x, y, 1e-14);

        let x = PI;
        let y = evaluate_continued_fraction(
            |n| f64::from(n * 2 - 1) * f64::from(n * 2 - 1),
            |n| if n == 0 { 3.0 } else { 6.0 },
            1e-15,
            10_000,
        );
        approx_eq(x, y, 1e-10);

        let y = evaluate_continued_fraction(
            |n| {
                if n == 1 {
                    4.0
                } else {
                    f64::from((n - 1) * (n - 1) * 4 - 1)
                }
            },
            |n| {
                if n == 0 {
                    2.0
                } else if n == 1 {
                    3.0
                } else {
                    4.0
                }
            },
            1e-15,
            10_000,
        );
        approx_eq(x, y, 1e-8);

        let y = evaluate_continued_fraction(
            |n| if n == 1 { 2.0 } else { f64::from(n * (n - 1)) },
            |n| if n == 0 { 2.0 } else { 1.0 },
            1e-15,
            10_000,
        );
        approx_eq(x, y, 1e-3);
    }

    #[test]
    fn test_norm_to_t_approx() {
        approx_eq(norm_to_t_approx(2.0, 2.0), 3.57464854186552161, 1e-14);
        approx_eq(norm_to_t_approx(2.0, 4.0), 2.85498285635306948, 1e-14);
        approx_eq(norm_to_t_approx(2.0, 8.0), 2.36638591905649687, 1e-14);
        approx_eq(norm_to_t_approx(2.0, 16.0), 2.16905959247696289, 1e-14);
        approx_eq(norm_to_t_approx(2.0, 10000.0), 2.0002500310444534, 1e-13);
        approx_eq(norm_to_t_approx(4.0, 2.0), 77.20049205855787022, 1e-14);
        approx_eq(norm_to_t_approx(4.0, 4.0), 18.34694064061386953, 1e-14);
        approx_eq(norm_to_t_approx(4.0, 8.0), 7.66893227341667760, 1e-14);
        approx_eq(norm_to_t_approx(4.0, 16.0), 5.37279049993877056, 1e-14);
        approx_eq(norm_to_t_approx(4.0, 10000.0), 4.00170065227857751, 1e-13);
        approx_eq(
            norm_to_t_approx(8.0, 2.0),
            12566858.01484839618206024,
            1e-14,
        );
        approx_eq(norm_to_t_approx(8.0, 4.0), 14501.91603376931016101, 1e-14);
        approx_eq(norm_to_t_approx(8.0, 8.0), 197.25867566592546609, 1e-14);
        approx_eq(norm_to_t_approx(8.0, 16.0), 31.19831990116452403, 1e-14);
        approx_eq(norm_to_t_approx(8.0, 10000.0), 8.01301804270851292, 1e-13);
    }

    #[test]
    fn test_beta() {
        // a=1 b=1 uniform
        approx_eq(beta_pdf(0.00, 1.0, 1.0), 1.0, 1e-13);
        approx_eq(beta_pdf(0.25, 1.0, 1.0), 1.0, 1e-13);
        approx_eq(beta_pdf(0.50, 1.0, 1.0), 1.0, 1e-13);
        approx_eq(beta_pdf(0.75, 1.0, 1.0), 1.0, 1e-13);
        approx_eq(beta_pdf(1.00, 1.0, 1.0), 1.0, 1e-13);
        approx_eq(beta_cdf(0.00, 1.0, 1.0), 0.0, 1e-13);
        approx_eq(beta_cdf(0.25, 1.0, 1.0), 0.25, 1e-13);
        approx_eq(beta_cdf(0.50, 1.0, 1.0), 0.5, 1e-13);
        approx_eq(beta_cdf(0.75, 1.0, 1.0), 0.75, 1e-13);
        approx_eq(beta_cdf(1.00, 1.0, 1.0), 1.0, 1e-13);

        // a=2 b=1 triangular
        approx_eq(beta_pdf(0.00, 2.0, 1.0), 0.0, 1e-13);
        approx_eq(beta_pdf(0.25, 2.0, 1.0), 0.5, 1e-13);
        approx_eq(beta_pdf(0.50, 2.0, 1.0), 1.0, 1e-13);
        approx_eq(beta_pdf(0.75, 2.0, 1.0), 1.5, 1e-13);
        approx_eq(beta_pdf(1.00, 2.0, 1.0), 2.0, 1e-13);
        approx_eq(beta_cdf(0.00, 2.0, 1.0), 0.0, 1e-13);
        approx_eq(beta_cdf(0.25, 2.0, 1.0), 0.06250000000000001, 1e-13);
        approx_eq(beta_cdf(0.50, 2.0, 1.0), 0.25000000000000006, 1e-13);
        approx_eq(beta_cdf(0.75, 2.0, 1.0), 0.5625, 1e-13);
        approx_eq(beta_cdf(1.00, 2.0, 1.0), 1.0, 1e-13);

        // a=3 b=1 quadratic
        approx_eq(beta_pdf(0.00, 3.0, 1.0), 0.0, 1e-13);
        approx_eq(beta_pdf(0.25, 3.0, 1.0), 0.1875, 1e-13);
        approx_eq(beta_pdf(0.50, 3.0, 1.0), 0.75, 1e-13);
        approx_eq(beta_pdf(0.75, 3.0, 1.0), 1.6875, 1e-13);
        approx_eq(beta_pdf(1.00, 3.0, 1.0), 3.0, 1e-13);
        approx_eq(beta_cdf(0.00, 3.0, 1.0), 0.0, 1e-13);
        approx_eq(beta_cdf(0.25, 3.0, 1.0), 0.01562500000000001, 1e-13);
        approx_eq(beta_cdf(0.50, 3.0, 1.0), 0.125, 1e-13);
        approx_eq(beta_cdf(0.75, 3.0, 1.0), 0.421875, 1e-13);
        approx_eq(beta_cdf(1.00, 3.0, 1.0), 1.0, 1e-13);

        // a=0.5 b=0.5 arcsin
        assert!(beta_pdf(0.00, 0.5, 0.5) >= f64::INFINITY);
        approx_eq(
            beta_pdf(0.10, 0.5, 0.5),
            1.0 / PI / (0.10f64 * (1.0 - 0.10f64)).sqrt(),
            1e-13,
        );
        approx_eq(
            beta_pdf(0.25, 0.5, 0.5),
            1.0 / PI / (0.25f64 * (1.0 - 0.25f64)).sqrt(),
            1e-13,
        );
        approx_eq(
            beta_pdf(0.50, 0.5, 0.5),
            1.0 / PI / (0.50f64 * (1.0 - 0.50f64)).sqrt(),
            1e-13,
        );
        approx_eq(
            beta_pdf(0.75, 0.5, 0.5),
            1.0 / PI / (0.75f64 * (1.0 - 0.75f64)).sqrt(),
            1e-13,
        );
        approx_eq(
            beta_pdf(0.90, 0.5, 0.5),
            1.0 / PI / (0.90f64 * (1.0 - 0.90f64)).sqrt(),
            1e-13,
        );
        assert!(beta_pdf(1.00, 0.5, 0.5) >= f64::INFINITY);
        approx_eq(beta_cdf(0.00, 0.5, 0.5), 0.0, 1e-13);
        approx_eq(
            beta_cdf(0.10, 0.5, 0.5),
            2.0 / PI * (0.10f64.sqrt()).asin(),
            1e-13,
        );
        approx_eq(
            beta_cdf(0.25, 0.5, 0.5),
            2.0 / PI * (0.25f64.sqrt()).asin(),
            1e-13,
        );
        approx_eq(
            beta_cdf(0.50, 0.5, 0.5),
            2.0 / PI * (0.50f64.sqrt()).asin(),
            1e-13,
        );
        approx_eq(
            beta_cdf(0.75, 0.5, 0.5),
            2.0 / PI * (0.75f64.sqrt()).asin(),
            1e-13,
        );
        approx_eq(
            beta_cdf(0.90, 0.5, 0.5),
            2.0 / PI * (0.90f64.sqrt()).asin(),
            1e-13,
        );
        approx_eq(beta_cdf(1.00, 0.5, 0.5), 1.0, 1e-13);

        // extreme values
        approx_eq(beta_pdf(0.00, 0.5e5, 0.5e1), 0.0, 1e-13);
        approx_eq(beta_pdf(0.25, 0.5e5, 0.5e1), 0.0, 1e-13);
        approx_eq(beta_pdf(0.50, 0.5e5, 0.5e1), 0.0, 1e-13);
        approx_eq(beta_pdf(0.75, 0.5e5, 0.5e1), 0.0, 1e-13);
        approx_eq(
            beta_pdf(1.0 - 1e-4, 0.5e5, 0.5e1),
            8773.80701229644182604,
            1e-9,
        );
        approx_eq(beta_pdf(1.00, 0.5e5, 0.5e1), 0.0, 1e-13);
        approx_eq(beta_cdf(0.00, 0.5e5, 0.5e1), 0.0, 1e-13);
        approx_eq(beta_cdf(0.25, 0.5e5, 0.5e1), 0.0, 1e-13);
        approx_eq(beta_cdf(0.50, 0.5e5, 0.5e1), 0.0, 1e-13);
        approx_eq(beta_cdf(0.75, 0.5e5, 0.5e1), 0.0, 1e-13);
        approx_eq(
            beta_cdf(1.0 - 1e-4, 0.5e5, 0.5e1),
            0.44041432429729233,
            1e-9,
        );
        approx_eq(beta_cdf(1.00, 0.5e5, 0.5e1), 1.0, 1e-13);

        approx_eq(beta_pdf(0.00, 0.5e10, 0.5e2), 0.0, 1e-13);
        approx_eq(beta_pdf(0.25, 0.5e10, 0.5e2), 0.0, 1e-13);
        approx_eq(beta_pdf(0.50, 0.5e10, 0.5e2), 0.0, 1e-13);
        approx_eq(beta_pdf(0.75, 0.5e10, 0.5e2), 0.0, 1e-13);
        approx_eq(
            beta_pdf(1.0 - 1e-8, 0.5e10, 0.5e2),
            281620447.51994127035140991,
            1e-4,
        );
        approx_eq(beta_pdf(1.00, 0.5e10, 0.5e2), 0.0, 1e-13);
        approx_eq(beta_cdf(0.00, 0.5e10, 0.5e2), 0.0, 1e-13);
        approx_eq(beta_cdf(0.25, 0.5e10, 0.5e2), 0.0, 1e-13);
        approx_eq(beta_cdf(0.50, 0.5e10, 0.5e2), 0.0, 1e-13);
        approx_eq(beta_cdf(0.75, 0.5e10, 0.5e2), 0.0, 1e-13);
        approx_eq(
            beta_cdf(1.0 - 1e-8, 0.5e10, 0.5e2),
            0.48120008730261921,
            1e-4,
        );
        approx_eq(beta_cdf(1.00, 0.5e10, 0.5e2), 1.0, 1e-13);

        // These hit numerical instability; only verify non-extreme outputs.
        approx_eq(beta_pdf(0.00, 0.5e15, 0.5e3), 0.0, 1e-13);
        approx_eq(beta_pdf(0.25, 0.5e15, 0.5e3), 0.0, 1e-13);
        approx_eq(beta_pdf(0.50, 0.5e15, 0.5e3), 0.0, 1e-13);
        approx_eq(beta_pdf(0.75, 0.5e15, 0.5e3), 0.0, 1e-13);
        approx_eq(beta_pdf(1.00, 0.5e15, 0.5e3), 0.0, 1e-13);
        approx_eq(beta_cdf(0.00, 0.5e15, 0.5e3), 0.0, 1e-13);
        approx_eq(beta_cdf(0.25, 0.5e15, 0.5e3), 0.0, 1e-13);
        approx_eq(beta_cdf(0.50, 0.5e15, 0.5e3), 0.0, 1e-13);
        approx_eq(beta_cdf(0.75, 0.5e15, 0.5e3), 0.0, 1e-13);
        approx_eq(beta_cdf(1.00, 0.5e15, 0.5e3), 1.0, 1e-13);
    }

    #[test]
    fn test_binary_cross_entropy() {
        approx_eq(binary_cross_entropy(0.5, 1.0, 0.001), 2.0f64.ln(), 1e-13);
        approx_eq(binary_cross_entropy(0.5, 0.0, 0.001), 2.0f64.ln(), 1e-13);
        approx_eq(binary_cross_entropy(0.5, 0.7, 0.001), 2.0f64.ln(), 1e-13);
        approx_eq(
            binary_cross_entropy(1.0 / std::f64::consts::E, 1.0, 0.0),
            1.0,
            1e-13,
        );
        approx_eq(
            binary_cross_entropy(1.0 / std::f64::consts::E, 0.0, 0.0),
            1.0 - ((std::f64::consts::E - 1.0).ln()),
            1e-13,
        );
        approx_eq(
            binary_cross_entropy(1.0 / std::f64::consts::E, 0.0, 0.5),
            2.0f64.ln(),
            1e-13,
        );
        approx_eq(
            binary_cross_entropy(0.0, 1.0, 0.25),
            2.0 * 2.0f64.ln(),
            1e-13,
        );
        approx_eq(
            binary_cross_entropy(1.0 / 6.0, 1.0, 0.25),
            3.0f64.ln(),
            1e-13,
        );
        approx_eq(
            binary_cross_entropy(1.0 / 6.0, 0.0, 0.25),
            (3.0f64 / 2.0).ln(),
            1e-13,
        );
        approx_eq(
            binary_cross_entropy(1.0 / 6.0, 0.8, 0.25),
            0.8 * 3.0f64.ln() + 0.2 * (3.0f64 / 2.0).ln(),
            1e-13,
        );
    }

    #[test]
    fn test_t_distribution() {
        let mut out = String::new();
        for df in [1.0, 2.0, 3.4, 12.3] {
            out.push_str(&format!(
                "{} degrees of freedom\n",
                match df as i32 {
                    1 => "1",
                    2 => "2",
                    _ if (df - 3.4f64).abs() < 1e-9 => "3.4",
                    _ => "12.3",
                }
            ));
            for i in 0..41 {
                let x = -6.0 + f64::from(i) * 0.3;
                out.push_str(&format!(
                    "{:.10} {:.10}\n",
                    t_dist_pdf(x, df),
                    t_dist_cdf(x, df)
                ));
            }
        }

        let expected = r"1 degrees of freedom
0.0086029699 0.0525684567
0.0095046248 0.0552812594
0.0105540413 0.0582859834
0.0117848903 0.0616317945
0.0132408439 0.0653793830
0.0149792888 0.0696044873
0.0170767106 0.0744027653
0.0196366370 0.0798966366
0.0228015678 0.0862450611
0.0267712268 0.0936577709
0.0318309886 0.1024163823
0.0383968500 0.1129063157
0.0470872613 0.1256659164
0.0588373172 0.1414630281
0.0750730864 0.1614144672
0.0979415034 0.1871670418
0.1304548714 0.2211420616
0.1758618156 0.2667377084
0.2340513869 0.3279791304
0.2920274185 0.4072264209
0.3183098862 0.5000000000
0.2920274185 0.5927735791
0.2340513869 0.6720208696
0.1758618156 0.7332622916
0.1304548714 0.7788579384
0.0979415034 0.8128329582
0.0750730864 0.8385855328
0.0588373172 0.8585369719
0.0470872613 0.8743340836
0.0383968500 0.8870936843
0.0318309886 0.8975836177
0.0267712268 0.9063422291
0.0228015678 0.9137549389
0.0196366370 0.9201033634
0.0170767106 0.9255972347
0.0149792888 0.9303955127
0.0132408439 0.9346206170
0.0117848903 0.9383682055
0.0105540413 0.9417140166
0.0095046248 0.9447187406
0.0086029699 0.9474315433
2 degrees of freedom
0.0042689848 0.0133357366
0.0049369668 0.0147134410
0.0057491525 0.0163123044
0.0067457515 0.0181813283
0.0079808383 0.0203835398
0.0095280708 0.0230009540
0.0114891467 0.0261416335
0.0140064700 0.0299498710
0.0172823426 0.0346210790
0.0216083012 0.0404238469
0.0274101222 0.0477329831
0.0353164002 0.0570793674
0.0462601906 0.0692251048
0.0616187602 0.0852749346
0.0833687077 0.1068331745
0.1141344118 0.1361965624
0.1567336820 0.1765016804
0.2122953688 0.2315525062
0.2758239639 0.3047166335
0.3309638583 0.3962428304
0.3535533906 0.5000000000
0.3309638583 0.6037571696
0.2758239639 0.6952833665
0.2122953688 0.7684474938
0.1567336820 0.8234983196
0.1141344118 0.8638034376
0.0833687077 0.8931668255
0.0616187602 0.9147250654
0.0462601906 0.9307748952
0.0353164002 0.9429206326
0.0274101222 0.9522670169
0.0216083012 0.9595761531
0.0172823426 0.9653789210
0.0140064700 0.9700501290
0.0114891467 0.9738583665
0.0095280708 0.9769990460
0.0079808383 0.9796164602
0.0067457515 0.9818186717
0.0057491525 0.9836876956
0.0049369668 0.9852865590
0.0042689848 0.9866642634
3.4 degrees of freedom
0.0016926222 0.0032139939
0.0020783090 0.0037772509
0.0025748151 0.0044720255
0.0032207920 0.0053370351
0.0040707697 0.0064248244
0.0052026332 0.0078075721
0.0067290090 0.0095856851
0.0088147440 0.0119006552
0.0117038425 0.0149544744
0.0157608931 0.0190391561
0.0215339658 0.0245817072
0.0298470062 0.0322122030
0.0419258315 0.0428646559
0.0595414562 0.0579191956
0.0850906717 0.0793812260
0.1213794434 0.1100489671
0.1706050692 0.1535122005
0.2318628213 0.2136387203
0.2973298200 0.2930838542
0.3502964915 0.3908009574
0.3710206734 0.5000000000
0.3502964915 0.6091990426
0.2973298200 0.7069161458
0.2318628213 0.7863612797
0.1706050692 0.8464877995
0.1213794434 0.8899510329
0.0850906717 0.9206187740
0.0595414562 0.9420808044
0.0419258315 0.9571353441
0.0298470062 0.9677877970
0.0215339658 0.9754182928
0.0157608931 0.9809608439
0.0117038425 0.9850455256
0.0088147440 0.9880993448
0.0067290090 0.9904143149
0.0052026332 0.9921924279
0.0040707697 0.9935751756
0.0032207920 0.9946629649
0.0025748151 0.9955279745
0.0020783090 0.9962227491
0.0016926222 0.9967860061
12.3 degrees of freedom
0.0000438241 0.0000280358
0.0000723782 0.0000450924
0.0001209837 0.0000734470
0.0002046142 0.0001211477
0.0003499363 0.0002023174
0.0006046408 0.0003419269
0.0010541129 0.0005843634
0.0018507560 0.0010087336
0.0032642231 0.0017558516
0.0057638358 0.0030748159
0.0101446535 0.0054006270
0.0176983420 0.0094766031
0.0303940498 0.0165311964
0.0509528159 0.0284974782
0.0825681085 0.0482098526
0.1279153538 0.0794194234
0.1872189518 0.1263713943
0.2558018507 0.1927011058
0.3226840054 0.2796991312
0.3724237721 0.3845934428
0.3909242313 0.5000000000
0.3724237721 0.6154065572
0.3226840054 0.7203008688
0.2558018507 0.8072988942
0.1872189518 0.8736286057
0.1279153538 0.9205805766
0.0825681085 0.9517901474
0.0509528159 0.9715025218
0.0303940498 0.9834688036
0.0176983420 0.9905233969
0.0101446535 0.9945993730
0.0057638358 0.9969251841
0.0032642231 0.9982441484
0.0018507560 0.9989912664
0.0010541129 0.9994156366
0.0006046408 0.9996580731
0.0003499363 0.9997976826
0.0002046142 0.9998788523
0.0001209837 0.9999265530
0.0000723782 0.9999549076
0.0000438241 0.9999719642
";

        assert_eq!(out, expected);
    }
}
