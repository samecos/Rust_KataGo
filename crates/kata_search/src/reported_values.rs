//! Aggregated values reported to callers from a finished search.
//!
//! Corresponds to `cpp/search/reportedsearchvalues.h` and
//! `cpp/search/reportedsearchvalues.cpp`.
//!
//! The C++ constructor takes `const Search&`. In Rust it is decoupled into
//! explicit parameters so it can be used before `Search` itself is ported.

use std::fmt;

use kata_nn::score_value;

/// Node-level statistics required to build reported values.
#[derive(Debug, Clone, Copy)]
pub struct ReportedSearchStats {
    pub win_loss_value_avg: f64,
    pub no_result_value_avg: f64,
    pub score_mean_avg: f64,
    pub score_mean_sq_avg: f64,
    pub lead_avg: f64,
    pub utility_avg: f64,
    pub total_weight: f64,
    pub total_visits: i64,
}

/// Context values that the C++ `Search` object would provide when constructing
/// `ReportedSearchValues`.
#[derive(Debug, Clone, Copy)]
pub struct ReportedSearchContext {
    /// `sqrt(root_board.x_size * root_board.y_size)`
    pub sqrt_board_area: f64,
    /// `search.recent_score_center`
    pub recent_score_center: f64,
    /// `search.search_params.dynamic_score_center_scale`
    pub dynamic_score_center_scale: f64,
}

/// Final aggregate statistics for a search result, suitable for GTP/reporting.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReportedSearchValues {
    pub win_value: f64,
    pub loss_value: f64,
    pub no_result_value: f64,
    pub static_score_value: f64,
    pub dynamic_score_value: f64,
    pub expected_score: f64,
    pub expected_score_stdev: f64,
    pub lead: f64,
    pub win_loss_value: f64,
    pub utility: f64,
    pub weight: f64,
    pub visits: i64,
}

impl ReportedSearchValues {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build reported values from node statistics and search context.
    ///
    /// All averages are from White's perspective, matching the C++ code.
    pub fn from_stats(stats: &ReportedSearchStats, ctx: &ReportedSearchContext) -> Self {
        let mut win_loss_value = stats.win_loss_value_avg;
        let mut no_result_value = stats.no_result_value_avg;
        let score_mean = stats.score_mean_avg;
        let score_mean_sq = stats.score_mean_sq_avg;
        let score_stdev = score_value::get_score_stdev(score_mean, score_mean_sq);

        let static_score_value = score_value::expected_white_score_value(
            score_mean,
            score_stdev,
            0.0,
            2.0,
            ctx.sqrt_board_area,
        );
        let dynamic_score_value = score_value::expected_white_score_value(
            score_mean,
            score_stdev,
            ctx.recent_score_center,
            ctx.dynamic_score_center_scale,
            ctx.sqrt_board_area,
        );

        // Clamp tiny floating-point errors.
        win_loss_value = win_loss_value.clamp(-1.0, 1.0);
        no_result_value = no_result_value.clamp(0.0, 1.0 - win_loss_value.abs());

        let win_value = (0.5 * (win_loss_value + (1.0 - no_result_value))).clamp(0.0, 1.0);
        let loss_value = (0.5 * (-win_loss_value + (1.0 - no_result_value))).clamp(0.0, 1.0);

        Self {
            win_value,
            loss_value,
            no_result_value,
            static_score_value,
            dynamic_score_value,
            expected_score: score_mean,
            expected_score_stdev: score_stdev,
            lead: stats.lead_avg,
            win_loss_value,
            utility: stats.utility_avg,
            weight: stats.total_weight,
            visits: stats.total_visits,
        }
    }
}

impl fmt::Display for ReportedSearchValues {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "winValue {}", self.win_value)?;
        writeln!(f, "lossValue {}", self.loss_value)?;
        writeln!(f, "noResultValue {}", self.no_result_value)?;
        writeln!(f, "staticScoreValue {}", self.static_score_value)?;
        writeln!(f, "dynamicScoreValue {}", self.dynamic_score_value)?;
        writeln!(f, "expectedScore {}", self.expected_score)?;
        writeln!(f, "expectedScoreStdev {}", self.expected_score_stdev)?;
        writeln!(f, "lead {}", self.lead)?;
        writeln!(f, "winLossValue {}", self.win_loss_value)?;
        writeln!(f, "utility {}", self.utility)?;
        writeln!(f, "weight {}", self.weight)?;
        writeln!(f, "visits {}", self.visits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> ReportedSearchContext {
        ReportedSearchContext {
            sqrt_board_area: 19.0,
            recent_score_center: 0.0,
            dynamic_score_center_scale: 1.0,
        }
    }

    #[test]
    fn test_default_values() {
        let v = ReportedSearchValues::new();
        assert_eq!(v.visits, 0);
        assert_eq!(v.win_value, 0.0);
    }

    #[allow(clippy::too_many_arguments)]
    fn stats(
        win_loss: f64,
        no_result: f64,
        score_mean: f64,
        score_mean_sq: f64,
        lead: f64,
        utility: f64,
        weight: f64,
        visits: i64,
    ) -> ReportedSearchStats {
        ReportedSearchStats {
            win_loss_value_avg: win_loss,
            no_result_value_avg: no_result,
            score_mean_avg: score_mean,
            score_mean_sq_avg: score_mean_sq,
            lead_avg: lead,
            utility_avg: utility,
            total_weight: weight,
            total_visits: visits,
        }
    }

    #[test]
    fn test_from_stats_clamps() {
        // Exaggerated inputs that should be clamped.
        let v = ReportedSearchValues::from_stats(
            &stats(-1.5, 0.9, 0.0, 0.0, 0.0, -0.2, 10.0, 100),
            &ctx(),
        );
        assert_eq!(v.win_loss_value, -1.0);
        assert!(v.no_result_value <= 0.0 + 1e-12);
        assert!(v.win_value >= 0.0);
        assert!(v.loss_value >= 0.0);
        assert!((v.win_value + v.loss_value + v.no_result_value - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_display_contains_fields() {
        let v = ReportedSearchValues::from_stats(
            &stats(0.0, 0.0, 2.0, 5.0, 1.5, 0.1, 20.0, 50),
            &ctx(),
        );
        let s = v.to_string();
        assert!(s.contains("winValue"));
        assert!(s.contains("expectedScoreStdev"));
        assert!(s.contains("visits 50"));
    }
}
