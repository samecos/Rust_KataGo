//! Settings that control how games are played and self-play data is generated.
//!
//! Corresponds to `cpp/program/playsettings.h` and `cpp/program/playsettings.cpp`.

use kata_core::config::ConfigParser;
use kata_core::global::StringError;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct PlaySettings {
    pub init_games_with_policy: bool,
    pub policy_init_area_prop: f64,
    pub start_poses_policy_init_area_prop: f64,
    pub compensate_after_policy_init_prob: f64,
    pub policy_init_gamma_shape: f64,
    pub side_position_prob: f64,

    pub policy_init_area_temperature: f64,
    pub handicap_temperature: f64,

    pub compensate_komi_visits: i32,
    pub flip_komi_prob_when_no_compensate: f64,

    pub estimate_lead_visits: i32,
    pub estimate_lead_prob: f64,

    pub early_fork_game_prob: f64,
    pub early_fork_game_expected_move_prop: f64,
    pub fork_game_prob: f64,
    pub fork_game_min_choices: i32,
    pub early_fork_game_max_choices: i32,
    pub fork_game_max_choices: i32,

    pub seki_fork_hack_prob: f64,
    pub fancy_komi_varying: bool,

    pub cheap_search_prob: f64,
    pub cheap_search_visits: i32,
    pub cheap_search_target_weight: f32,

    pub reduce_visits: bool,
    pub reduce_visits_threshold: f64,
    pub reduce_visits_threshold_lookback: i32,
    pub reduced_visits_min: i32,
    pub reduced_visits_weight: f32,

    pub policy_surprise_data_weight: f64,
    pub value_surprise_data_weight: f64,
    pub scale_data_weight: f64,

    pub record_tree_positions: bool,
    pub record_tree_threshold: i32,
    pub record_tree_target_weight: f32,

    pub no_resolve_target_weights: bool,

    pub allow_resignation: bool,
    pub resign_threshold: f64,
    pub resign_consec_turns: i32,

    pub for_self_play: bool,

    pub handicap_asymmetric_playout_prob: f64,
    pub normal_asymmetric_playout_prob: f64,
    pub max_asymmetric_ratio: f64,
    pub min_asymmetric_compensate_komi_prob: f64,

    pub dynamic_self_komi_bonus_min: f64,
    pub dynamic_self_komi_bonus_max: f64,
    pub dynamic_self_komi_win_loss_min: f64,
    pub dynamic_self_komi_win_loss_max: f64,

    pub record_time_per_move: bool,
}

impl PlaySettings {
    /// Default values for all fields.
    pub fn new() -> Self {
        Self {
            init_games_with_policy: false,
            policy_init_area_prop: 0.0,
            start_poses_policy_init_area_prop: 0.0,
            compensate_after_policy_init_prob: 0.0,
            policy_init_gamma_shape: 1.0,
            side_position_prob: 0.0,

            policy_init_area_temperature: 1.0,
            handicap_temperature: 1.0,

            compensate_komi_visits: 20,
            flip_komi_prob_when_no_compensate: 0.0,

            estimate_lead_visits: 10,
            estimate_lead_prob: 0.0,

            early_fork_game_prob: 0.0,
            early_fork_game_expected_move_prop: 0.0,
            fork_game_prob: 0.0,
            fork_game_min_choices: 1,
            early_fork_game_max_choices: 1,
            fork_game_max_choices: 1,

            seki_fork_hack_prob: 0.0,
            fancy_komi_varying: false,

            cheap_search_prob: 0.0,
            cheap_search_visits: 0,
            cheap_search_target_weight: 0.0,

            reduce_visits: false,
            reduce_visits_threshold: 100.0,
            reduce_visits_threshold_lookback: 1,
            reduced_visits_min: 0,
            reduced_visits_weight: 1.0,

            policy_surprise_data_weight: 0.0,
            value_surprise_data_weight: 0.0,
            scale_data_weight: 1.0,

            record_tree_positions: false,
            record_tree_threshold: 0,
            record_tree_target_weight: 0.0,

            no_resolve_target_weights: false,

            allow_resignation: false,
            resign_threshold: 0.0,
            resign_consec_turns: 1,

            for_self_play: false,

            handicap_asymmetric_playout_prob: 0.0,
            normal_asymmetric_playout_prob: 0.0,
            max_asymmetric_ratio: 2.0,
            min_asymmetric_compensate_komi_prob: 0.0,

            dynamic_self_komi_bonus_min: 0.0,
            dynamic_self_komi_bonus_max: 0.0,
            dynamic_self_komi_win_loss_min: -1.0,
            dynamic_self_komi_win_loss_max: 1.0,

            record_time_per_move: false,
        }
    }

    /// Load settings for a match (two players, no training data).
    pub fn load_for_match(cfg: &ConfigParser) -> Result<Self, StringError> {
        let mut ps = Self::new();
        ps.allow_resignation = cfg.get_bool("allowResignation").map_err(to_string_error)?;
        ps.resign_threshold = cfg
            .get_double("resignThreshold", -1.0, 0.0)
            .map_err(to_string_error)?;
        ps.resign_consec_turns = cfg
            .get_int("resignConsecTurns", 1, 100)
            .map_err(to_string_error)?;
        ps.compensate_komi_visits = if cfg.contains("compensateKomiVisits") {
            cfg.get_int("compensateKomiVisits", 1, 10000)
                .map_err(to_string_error)?
        } else {
            100
        };
        ps.init_games_with_policy = if cfg.contains("initGamesWithPolicy") {
            cfg.get_bool("initGamesWithPolicy")
                .map_err(to_string_error)?
        } else {
            false
        };
        if ps.init_games_with_policy {
            ps.policy_init_area_prop = cfg
                .get_double("policyInitAreaProp", 0.0, 1.0)
                .map_err(to_string_error)?;
            ps.start_poses_policy_init_area_prop = if cfg.contains("startPosesPolicyInitAreaProp") {
                cfg.get_double("startPosesPolicyInitAreaProp", 0.0, 1.0)
                    .map_err(to_string_error)?
            } else {
                0.0
            };
            ps.compensate_after_policy_init_prob = if cfg.contains("compensateAfterPolicyInitProb")
            {
                cfg.get_double("compensateAfterPolicyInitProb", 0.0, 1.0)
                    .map_err(to_string_error)?
            } else {
                1.0
            };
            ps.policy_init_gamma_shape = if cfg.contains("policyInitGammaShape") {
                cfg.get_double("policyInitGammaShape", 0.5, 100.0)
                    .map_err(to_string_error)?
            } else {
                1.0
            };
            ps.policy_init_area_temperature = if cfg.contains("policyInitAreaTemperature") {
                cfg.get_double("policyInitAreaTemperature", 0.1, 5.0)
                    .map_err(to_string_error)?
            } else {
                1.0
            };
        }

        ps.dynamic_self_komi_bonus_min = if cfg.contains("dynamicSelfKomiBonusMin") {
            cfg.get_double("dynamicSelfKomiBonusMin", -100.0, 100.0)
                .map_err(to_string_error)?
        } else {
            0.0
        };
        ps.dynamic_self_komi_bonus_max = if cfg.contains("dynamicSelfKomiBonusMax") {
            cfg.get_double("dynamicSelfKomiBonusMax", -100.0, 100.0)
                .map_err(to_string_error)?
        } else {
            0.0
        };
        ps.dynamic_self_komi_win_loss_min = if cfg.contains("dynamicSelfKomiWinLossMin") {
            cfg.get_double("dynamicSelfKomiWinLossMin", -1.0, 1.0)
                .map_err(to_string_error)?
        } else {
            -1.0
        };
        ps.dynamic_self_komi_win_loss_max = if cfg.contains("dynamicSelfKomiWinLossMax") {
            cfg.get_double("dynamicSelfKomiWinLossMax", -1.0, 1.0)
                .map_err(to_string_error)?
        } else {
            1.0
        };

        if ps.dynamic_self_komi_bonus_min > ps.dynamic_self_komi_bonus_max {
            return Err(StringError {
                message: "dynamicSelfKomiBonusMin > dynamicSelfKomiBonusMax".to_string(),
            });
        }
        if ps.dynamic_self_komi_win_loss_min > ps.dynamic_self_komi_win_loss_max {
            return Err(StringError {
                message: "dynamicSelfKomiWinLossMin > dynamicSelfKomiWinLossMax".to_string(),
            });
        }

        ps.record_time_per_move = true;
        Ok(ps)
    }

    /// Load settings for the gatekeeper command.
    pub fn load_for_gatekeeper(cfg: &ConfigParser) -> Result<Self, StringError> {
        let mut ps = Self::new();
        ps.allow_resignation = cfg.get_bool("allowResignation").map_err(to_string_error)?;
        ps.resign_threshold = cfg
            .get_double("resignThreshold", -1.0, 0.0)
            .map_err(to_string_error)?;
        ps.resign_consec_turns = cfg
            .get_int("resignConsecTurns", 1, 100)
            .map_err(to_string_error)?;
        ps.compensate_komi_visits = if cfg.contains("compensateKomiVisits") {
            cfg.get_int("compensateKomiVisits", 1, 10000)
                .map_err(to_string_error)?
        } else {
            100
        };
        Ok(ps)
    }

    /// Load settings for self-play data generation.
    pub fn load_for_selfplay(
        cfg: &ConfigParser,
        is_distributed: bool,
    ) -> Result<Self, StringError> {
        let mut ps = Self::new();
        ps.init_games_with_policy = cfg
            .get_bool("initGamesWithPolicy")
            .map_err(to_string_error)?;
        ps.policy_init_area_prop = if cfg.contains("policyInitAreaProp") {
            cfg.get_double("policyInitAreaProp", 0.0, 1.0)
                .map_err(to_string_error)?
        } else {
            0.04
        };
        ps.start_poses_policy_init_area_prop = if cfg.contains("startPosesPolicyInitAreaProp") {
            cfg.get_double("startPosesPolicyInitAreaProp", 0.0, 1.0)
                .map_err(to_string_error)?
        } else {
            0.0
        };
        ps.compensate_after_policy_init_prob = cfg
            .get_double("compensateAfterPolicyInitProb", 0.0, 1.0)
            .map_err(to_string_error)?;
        ps.policy_init_gamma_shape = if cfg.contains("policyInitGammaShape") {
            cfg.get_double("policyInitGammaShape", 0.5, 10.0)
                .map_err(to_string_error)?
        } else {
            1.0
        };

        ps.side_position_prob =
            if cfg.contains("forkSidePositionProb") && !cfg.contains("sidePositionProb") {
                cfg.get_double("forkSidePositionProb", 0.0, 1.0)
                    .map_err(to_string_error)?
            } else {
                cfg.get_double("sidePositionProb", 0.0, 1.0)
                    .map_err(to_string_error)?
            };

        ps.policy_init_area_temperature = if cfg.contains("policyInitAreaTemperature") {
            cfg.get_double("policyInitAreaTemperature", 0.1, 5.0)
                .map_err(to_string_error)?
        } else {
            1.0
        };
        ps.handicap_temperature = if cfg.contains("handicapTemperature") {
            cfg.get_double("handicapTemperature", 0.1, 5.0)
                .map_err(to_string_error)?
        } else {
            1.0
        };

        ps.compensate_komi_visits = if cfg.contains("compensateKomiVisits") {
            cfg.get_int("compensateKomiVisits", 1, 10000)
                .map_err(to_string_error)?
        } else {
            20
        };
        ps.flip_komi_prob_when_no_compensate = if cfg.contains("flipKomiProbWhenNoCompensate") {
            cfg.get_double("flipKomiProbWhenNoCompensate", 0.0, 1.0)
                .map_err(to_string_error)?
        } else if is_distributed {
            0.25
        } else {
            0.0
        };
        ps.estimate_lead_visits = if cfg.contains("estimateLeadVisits") {
            cfg.get_int("estimateLeadVisits", 1, 10000)
                .map_err(to_string_error)?
        } else {
            6
        };
        ps.estimate_lead_prob = if cfg.contains("estimateLeadProb") {
            cfg.get_double("estimateLeadProb", 0.0, 1.0)
                .map_err(to_string_error)?
        } else {
            0.0
        };
        ps.fancy_komi_varying = if cfg.contains("fancyKomiVarying") {
            cfg.get_bool("fancyKomiVarying").map_err(to_string_error)?
        } else {
            false
        };

        ps.early_fork_game_prob = cfg
            .get_double("earlyForkGameProb", 0.0, 0.5)
            .map_err(to_string_error)?;
        ps.early_fork_game_expected_move_prop = cfg
            .get_double("earlyForkGameExpectedMoveProp", 0.0, 1.0)
            .map_err(to_string_error)?;
        ps.fork_game_prob = cfg
            .get_double("forkGameProb", 0.0, 0.5)
            .map_err(to_string_error)?;
        ps.fork_game_min_choices = cfg
            .get_int("forkGameMinChoices", 1, 100)
            .map_err(to_string_error)?;
        ps.early_fork_game_max_choices = cfg
            .get_int("earlyForkGameMaxChoices", 1, 100)
            .map_err(to_string_error)?;
        ps.fork_game_max_choices = cfg
            .get_int("forkGameMaxChoices", 1, 100)
            .map_err(to_string_error)?;

        ps.cheap_search_prob = cfg
            .get_double("cheapSearchProb", 0.0, 1.0)
            .map_err(to_string_error)?;
        ps.cheap_search_visits = cfg
            .get_int("cheapSearchVisits", 1, 10_000_000)
            .map_err(to_string_error)?;
        ps.cheap_search_target_weight = cfg
            .get_float("cheapSearchTargetWeight", 0.0, 1.0)
            .map_err(to_string_error)?;
        ps.reduce_visits = cfg.get_bool("reduceVisits").map_err(to_string_error)?;
        ps.reduce_visits_threshold = cfg
            .get_double("reduceVisitsThreshold", 0.0, 0.999_999)
            .map_err(to_string_error)?;
        ps.reduce_visits_threshold_lookback = cfg
            .get_int("reduceVisitsThresholdLookback", 0, 1000)
            .map_err(to_string_error)?;
        ps.reduced_visits_min = cfg
            .get_int("reducedVisitsMin", 1, 10_000_000)
            .map_err(to_string_error)?;
        ps.reduced_visits_weight = cfg
            .get_float("reducedVisitsWeight", 0.0, 1.0)
            .map_err(to_string_error)?;

        ps.policy_surprise_data_weight = cfg
            .get_double("policySurpriseDataWeight", 0.0, 1.0)
            .map_err(to_string_error)?;
        ps.value_surprise_data_weight = cfg
            .get_double("valueSurpriseDataWeight", 0.0, 1.0)
            .map_err(to_string_error)?;
        ps.scale_data_weight = if cfg.contains("scaleDataWeight") {
            cfg.get_double("scaleDataWeight", 0.01, 10.0)
                .map_err(to_string_error)?
        } else {
            1.0
        };

        ps.handicap_asymmetric_playout_prob = cfg
            .get_double("handicapAsymmetricPlayoutProb", 0.0, 1.0)
            .map_err(to_string_error)?;
        ps.normal_asymmetric_playout_prob = cfg
            .get_double("normalAsymmetricPlayoutProb", 0.0, 1.0)
            .map_err(to_string_error)?;
        ps.max_asymmetric_ratio = cfg
            .get_double("maxAsymmetricRatio", 1.0, 100.0)
            .map_err(to_string_error)?;
        ps.min_asymmetric_compensate_komi_prob = cfg
            .get_double("minAsymmetricCompensateKomiProb", 0.0, 1.0)
            .map_err(to_string_error)?;
        ps.seki_fork_hack_prob = if cfg.contains("sekiForkHackProb") {
            cfg.get_double("sekiForkHackProb", 0.0, 1.0)
                .map_err(to_string_error)?
        } else {
            0.0
        };
        ps.for_self_play = true;

        if ps.policy_surprise_data_weight + ps.value_surprise_data_weight > 1.0 {
            return Err(StringError {
                message: "policySurpriseDataWeight + valueSurpriseDataWeight > 1.0".to_string(),
            });
        }

        Ok(ps)
    }
}

fn to_string_error(e: impl std::error::Error) -> StringError {
    StringError {
        message: e.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(s: &str) -> ConfigParser {
        ConfigParser::from_str(s, false, true).unwrap()
    }

    #[test]
    fn test_default_values() {
        let ps = PlaySettings::new();
        assert!(!ps.init_games_with_policy);
        assert_eq!(ps.policy_init_gamma_shape, 1.0);
        assert_eq!(ps.max_asymmetric_ratio, 2.0);
        assert_eq!(ps.dynamic_self_komi_win_loss_min, -1.0);
    }

    #[test]
    fn test_load_for_match() {
        let cfg = cfg("allowResignation = true\nresignThreshold = -0.9\nresignConsecTurns = 3\n");
        let ps = PlaySettings::load_for_match(&cfg).unwrap();
        assert!(ps.allow_resignation);
        assert_eq!(ps.resign_threshold, -0.9);
        assert_eq!(ps.resign_consec_turns, 3);
        assert!(ps.record_time_per_move);
    }

    fn selfplay_cfg(extra: &str) -> ConfigParser {
        let base = "initGamesWithPolicy = false\n\
            compensateAfterPolicyInitProb = 0.5\n\
            sidePositionProb = 0.1\n\
            earlyForkGameProb = 0.0\n\
            earlyForkGameExpectedMoveProp = 0.0\n\
            forkGameProb = 0.0\n\
            forkGameMinChoices = 1\n\
            earlyForkGameMaxChoices = 1\n\
            forkGameMaxChoices = 1\n\
            cheapSearchProb = 0.0\n\
            cheapSearchVisits = 1\n\
            cheapSearchTargetWeight = 0.0\n\
            reduceVisits = false\n\
            reduceVisitsThreshold = 0.5\n\
            reduceVisitsThresholdLookback = 1\n\
            reducedVisitsMin = 1\n\
            reducedVisitsWeight = 1.0\n\
            handicapAsymmetricPlayoutProb = 0.0\n\
            normalAsymmetricPlayoutProb = 0.0\n\
            maxAsymmetricRatio = 2.0\n\
            minAsymmetricCompensateKomiProb = 0.0\n";
        cfg(&format!("{}{}", base, extra))
    }

    #[test]
    fn test_load_for_selfplay_minimal() {
        let cfg = selfplay_cfg("policySurpriseDataWeight = 0.0\nvalueSurpriseDataWeight = 0.0\n");
        let ps = PlaySettings::load_for_selfplay(&cfg, false).unwrap();
        assert!(!ps.init_games_with_policy);
        assert_eq!(ps.compensate_after_policy_init_prob, 0.5);
        assert_eq!(ps.side_position_prob, 0.1);
        assert!(ps.for_self_play);
    }

    #[test]
    fn test_load_for_selfplay_distributed_default() {
        let cfg = selfplay_cfg("policySurpriseDataWeight = 0.0\nvalueSurpriseDataWeight = 0.0\n");
        let ps = PlaySettings::load_for_selfplay(&cfg, true).unwrap();
        assert_eq!(ps.flip_komi_prob_when_no_compensate, 0.25);
    }

    #[test]
    fn test_load_for_selfplay_rejects_excessive_surprise_weight() {
        let cfg = selfplay_cfg("policySurpriseDataWeight = 0.6\nvalueSurpriseDataWeight = 0.5\n");
        assert!(PlaySettings::load_for_selfplay(&cfg, false).is_err());
    }
}
