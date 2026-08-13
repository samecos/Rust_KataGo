//! KataGo program setup helpers.
//!
//! Corresponds to parts of `cpp/program/setup.h` and `cpp/program/setup.cpp`.
//! This slice only covers the mutually-exclusive config key sets needed by
//! command-line override handling; the full `Setup` namespace is left for later
//! slices.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::sync::Arc;

use kata_core::config::ConfigParser;
use kata_core::fs;
use kata_core::global;
use kata_core::global::StringError;
use kata_core::logger::Logger;
use kata_core::rng::Rand;
use kata_data::sgf::PositionSample;
use kata_game::board::{C_EMPTY, Player, player_io};
use kata_game::rules::{KoRule, Rules, ScoringRule, TaxRule, WhiteHandicapBonusRule};
use kata_game::symmetry::NUM_SYMMETRIES;
use kata_nn::backend::Enabled;
use kata_nn::eval::NnEvaluator;
use kata_nn::inputs::nn_pos;
use kata_nn::sgf_meta;
use kata_search::params::SearchParams;
use kata_search::pattern_bonus::PatternBonusTable;

/// Return mutually exclusive config key sets used when applying command-line overrides.
///
/// Mirrors `Setup::getMutexKeySets` in `cpp/program/setup.cpp`.
///
/// If an override specifies `rules`, then the expanded rule keys should be erased
/// from the config, and vice versa.
pub fn get_mutex_key_sets() -> Vec<(BTreeSet<String>, BTreeSet<String>)> {
    let mut rules = BTreeSet::new();
    rules.insert("rules".to_string());

    let mut expanded = BTreeSet::new();
    expanded.insert("koRule".to_string());
    expanded.insert("scoringRule".to_string());
    expanded.insert("multiStoneSuicideLegal".to_string());
    expanded.insert("taxRule".to_string());
    expanded.insert("hasButton".to_string());
    expanded.insert("whiteBonusPerHandicapStone".to_string());
    expanded.insert("friendlyPassOk".to_string());
    expanded.insert("whiteHandicapBonus".to_string());

    vec![(rules, expanded)]
}

/// Which KataGo program mode the parameters are being loaded for.
///
/// Mirrors `Setup::setup_for_t` in `cpp/program/setup.h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SetupFor {
    Gtp,
    Benchmark,
    Match,
    Analysis,
    Other,
    Distributed,
}

/// Maximum number of bot configurations that can be read from a single config file.
///
/// Mirrors `Setup::MAX_BOT_PARAMS_FROM_CFG` in `cpp/program/setup.h`.
pub const MAX_BOT_PARAMS_FROM_CFG: i32 = 4096;

/// Default wide root noise used for analysis mode.
const DEFAULT_ANALYSIS_WIDE_ROOT_NOISE: f64 = 0.04;
/// Default value for ignoring pre-root history in analysis mode.
const DEFAULT_ANALYSIS_IGNORE_PRE_ROOT_HISTORY: bool = true;

fn to_string_error(e: impl std::error::Error) -> StringError {
    StringError {
        message: e.to_string(),
    }
}

fn enabled_from_core(value: kata_core::config::Enabled) -> Enabled {
    match value {
        kata_core::config::Enabled::False => Enabled::False,
        kata_core::config::Enabled::True => Enabled::True,
        kata_core::config::Enabled::Auto => Enabled::Auto,
    }
}

fn parse_player(field: &str, s: &str) -> Result<Player, StringError> {
    player_io::try_parse_player(s).ok_or_else(|| {
        StringError::new(format!(
            "Could not parse player in field {}, should be BLACK or WHITE",
            field
        ))
    })
}

/// Loads search parameters for all bots described by `cfg`.
///
/// Mirrors `Setup::loadParams(ConfigParser&, setup_for_t)`.
pub fn load_params(
    cfg: &ConfigParser,
    setup_for: SetupFor,
) -> Result<Vec<SearchParams>, StringError> {
    load_params_impl(cfg, setup_for, false, false)
}

/// Loads search parameters for all bots described by `cfg`, with a flag for whether
/// a human SL model is available.
///
/// Mirrors `Setup::loadParams(ConfigParser&, setup_for_t, bool)`.
pub fn load_params_with_human(
    cfg: &ConfigParser,
    setup_for: SetupFor,
    has_human_model: bool,
) -> Result<Vec<SearchParams>, StringError> {
    load_params_impl(cfg, setup_for, has_human_model, false)
}

/// Loads search parameters for bots described by `cfg`.
///
/// Mirrors `Setup::loadParams(ConfigParser&, setup_for_t, bool, bool)`.
#[allow(clippy::too_many_arguments)]
pub fn load_params_full(
    cfg: &ConfigParser,
    setup_for: SetupFor,
    has_human_model: bool,
    load_single_config_only: bool,
) -> Result<Vec<SearchParams>, StringError> {
    load_params_impl(cfg, setup_for, has_human_model, load_single_config_only)
}

/// Loads search parameters for a single bot configuration.
///
/// Mirrors `Setup::loadSingleParams(ConfigParser&, setup_for_t)`.
pub fn load_single_params(
    cfg: &ConfigParser,
    setup_for: SetupFor,
) -> Result<SearchParams, StringError> {
    let params_vec = load_params_impl(cfg, setup_for, false, true)?;
    if params_vec.len() != 1 {
        return Err(StringError::new(
            "Config contains parameters for multiple bot configurations, but this KataGo command only supports a single configuration",
        ));
    }
    Ok(params_vec.into_iter().next().expect("length checked"))
}

/// Loads search parameters for a single bot configuration, with a flag for whether
/// a human SL model is available.
///
/// Mirrors `Setup::loadSingleParams(ConfigParser&, setup_for_t, bool)`.
pub fn load_single_params_with_human(
    cfg: &ConfigParser,
    setup_for: SetupFor,
    has_human_model: bool,
) -> Result<SearchParams, StringError> {
    let params_vec = load_params_impl(cfg, setup_for, has_human_model, true)?;
    if params_vec.len() != 1 {
        return Err(StringError::new(
            "Config contains parameters for multiple bot configurations, but this KataGo command only supports a single configuration",
        ));
    }
    Ok(params_vec.into_iter().next().expect("length checked"))
}

fn load_params_impl(
    cfg: &ConfigParser,
    setup_for: SetupFor,
    has_human_model: bool,
    load_single_config_only: bool,
) -> Result<Vec<SearchParams>, StringError> {
    let mut params_vec = Vec::new();

    let mut num_bots = 1;
    if cfg.contains("numBots") {
        num_bots = cfg
            .get_int("numBots", 1, MAX_BOT_PARAMS_FROM_CFG)
            .map_err(to_string_error)?;
    }

    if load_single_config_only && num_bots != 1 {
        return Err(StringError::new(
            "The config for this command cannot have numBots > 0",
        ));
    }

    for i in 0..num_bots {
        let idx_str = if load_single_config_only {
            String::new()
        } else {
            i.to_string()
        };

        let mut params = SearchParams::new();

        let contains_idx = |key: &str| -> bool {
            cfg.contains(&(key.to_string() + &idx_str)) || cfg.contains(key)
        };
        let key_for = |key: &str| -> String {
            if cfg.contains(&(key.to_string() + &idx_str)) {
                key.to_string() + &idx_str
            } else {
                key.to_string()
            }
        };

        params.max_playouts = if contains_idx("maxPlayouts") {
            cfg.get_int64(&key_for("maxPlayouts"), 1, 1_i64 << 50)
                .map_err(to_string_error)?
        } else {
            params.max_playouts
        };
        params.max_visits = if contains_idx("maxVisits") {
            cfg.get_int64(&key_for("maxVisits"), 1, 1_i64 << 50)
                .map_err(to_string_error)?
        } else {
            params.max_visits
        };
        params.max_time = if contains_idx("maxTime") {
            cfg.get_double(&key_for("maxTime"), 0.0, 1.0e20)
                .map_err(to_string_error)?
        } else {
            params.max_time
        };

        params.max_playouts_pondering = if contains_idx("maxPlayoutsPondering") {
            cfg.get_int64(&key_for("maxPlayoutsPondering"), 1, 1_i64 << 50)
                .map_err(to_string_error)?
        } else {
            1_i64 << 50
        };
        params.max_visits_pondering = if contains_idx("maxVisitsPondering") {
            cfg.get_int64(&key_for("maxVisitsPondering"), 1, 1_i64 << 50)
                .map_err(to_string_error)?
        } else {
            1_i64 << 50
        };
        params.max_time_pondering = if contains_idx("maxTimePondering") {
            cfg.get_double(&key_for("maxTimePondering"), 0.0, 1.0e20)
                .map_err(to_string_error)?
        } else {
            1.0e20
        };

        params.lag_buffer = if contains_idx("lagBuffer") {
            cfg.get_double(&key_for("lagBuffer"), 0.0, 3600.0)
                .map_err(to_string_error)?
        } else {
            0.0
        };

        params.search_factor_after_one_pass = if contains_idx("searchFactorAfterOnePass") {
            cfg.get_double(&key_for("searchFactorAfterOnePass"), 0.0, 1.0)
                .map_err(to_string_error)?
        } else {
            params.search_factor_after_one_pass
        };
        params.search_factor_after_two_pass = if contains_idx("searchFactorAfterTwoPass") {
            cfg.get_double(&key_for("searchFactorAfterTwoPass"), 0.0, 1.0)
                .map_err(to_string_error)?
        } else {
            params.search_factor_after_two_pass
        };

        params.num_threads = if contains_idx("numSearchThreads") {
            cfg.get_int(&key_for("numSearchThreads"), 1, 4096)
                .map_err(to_string_error)?
        } else {
            cfg.get_int("numSearchThreads", 1, 4096)
                .map_err(to_string_error)?
        };

        params.min_playouts_per_thread = if contains_idx("minPlayoutsPerThread") {
            cfg.get_double(&key_for("minPlayoutsPerThread"), 0.0, 1.0e20)
                .map_err(to_string_error)?
        } else if cfg.contains("minPlayoutsPerThread") {
            cfg.get_double("minPlayoutsPerThread", 0.0, 1.0e20)
                .map_err(to_string_error)?
        } else {
            match setup_for {
                SetupFor::Analysis | SetupFor::Gtp => 8.0,
                _ => 0.0,
            }
        };

        params.win_loss_utility_factor = if contains_idx("winLossUtilityFactor") {
            cfg.get_double(&key_for("winLossUtilityFactor"), 0.0, 1.0)
                .map_err(to_string_error)?
        } else {
            1.0
        };
        params.static_score_utility_factor = if contains_idx("staticScoreUtilityFactor") {
            cfg.get_double(&key_for("staticScoreUtilityFactor"), 0.0, 1.0)
                .map_err(to_string_error)?
        } else {
            0.1
        };
        params.dynamic_score_utility_factor = if contains_idx("dynamicScoreUtilityFactor") {
            cfg.get_double(&key_for("dynamicScoreUtilityFactor"), 0.0, 1.0)
                .map_err(to_string_error)?
        } else {
            0.3
        };
        params.no_result_utility_for_white = if contains_idx("noResultUtilityForWhite") {
            cfg.get_double(&key_for("noResultUtilityForWhite"), -1.0, 1.0)
                .map_err(to_string_error)?
        } else {
            0.0
        };
        params.draw_equivalent_wins_for_white = if contains_idx("drawEquivalentWinsForWhite") {
            cfg.get_double(&key_for("drawEquivalentWinsForWhite"), 0.0, 1.0)
                .map_err(to_string_error)?
        } else {
            0.5
        };

        params.dynamic_score_center_zero_weight = if contains_idx("dynamicScoreCenterZeroWeight") {
            cfg.get_double(&key_for("dynamicScoreCenterZeroWeight"), 0.0, 1.0)
                .map_err(to_string_error)?
        } else {
            0.20
        };
        params.dynamic_score_center_scale = if contains_idx("dynamicScoreCenterScale") {
            cfg.get_double(&key_for("dynamicScoreCenterScale"), 0.2, 5.0)
                .map_err(to_string_error)?
        } else {
            0.75
        };

        params.cpuct_exploration = if contains_idx("cpuctExploration") {
            cfg.get_double(&key_for("cpuctExploration"), 0.0, 10.0)
                .map_err(to_string_error)?
        } else {
            1.0
        };
        params.cpuct_exploration_log = if contains_idx("cpuctExplorationLog") {
            cfg.get_double(&key_for("cpuctExplorationLog"), 0.0, 10.0)
                .map_err(to_string_error)?
        } else {
            0.45
        };
        params.cpuct_exploration_base = if contains_idx("cpuctExplorationBase") {
            cfg.get_double(&key_for("cpuctExplorationBase"), 10.0, 100_000.0)
                .map_err(to_string_error)?
        } else {
            500.0
        };

        params.cpuct_utility_stdev_prior = if contains_idx("cpuctUtilityStdevPrior") {
            cfg.get_double(&key_for("cpuctUtilityStdevPrior"), 1e-8, 10.0)
                .map_err(to_string_error)?
        } else {
            0.40
        };
        params.cpuct_utility_stdev_prior_weight = if contains_idx("cpuctUtilityStdevPriorWeight") {
            cfg.get_double(&key_for("cpuctUtilityStdevPriorWeight"), 0.0, 100.0)
                .map_err(to_string_error)?
        } else {
            2.0
        };
        params.cpuct_utility_stdev_scale = if contains_idx("cpuctUtilityStdevScale") {
            cfg.get_double(&key_for("cpuctUtilityStdevScale"), 0.0, 1.0)
                .map_err(to_string_error)?
        } else {
            match setup_for {
                SetupFor::Distributed | SetupFor::Other => 0.0,
                _ => 0.85,
            }
        };

        params.fpu_reduction_max = if contains_idx("fpuReductionMax") {
            cfg.get_double(&key_for("fpuReductionMax"), 0.0, 2.0)
                .map_err(to_string_error)?
        } else {
            0.2
        };
        params.fpu_loss_prop = if contains_idx("fpuLossProp") {
            cfg.get_double(&key_for("fpuLossProp"), 0.0, 1.0)
                .map_err(to_string_error)?
        } else {
            0.0
        };
        params.fpu_parent_weight_by_visited_policy =
            if contains_idx("fpuParentWeightByVisitedPolicy") {
                cfg.get_bool(&key_for("fpuParentWeightByVisitedPolicy"))
                    .map_err(to_string_error)?
            } else {
                setup_for != SetupFor::Distributed
            };
        if params.fpu_parent_weight_by_visited_policy {
            params.fpu_parent_weight_by_visited_policy_pow =
                if contains_idx("fpuParentWeightByVisitedPolicyPow") {
                    cfg.get_double(&key_for("fpuParentWeightByVisitedPolicyPow"), 0.0, 5.0)
                        .map_err(to_string_error)?
                } else {
                    2.0
                };
        } else {
            params.fpu_parent_weight = if contains_idx("fpuParentWeight") {
                cfg.get_double(&key_for("fpuParentWeight"), 0.0, 1.0)
                    .map_err(to_string_error)?
            } else {
                0.0
            };
        }

        params.policy_optimism = if contains_idx("policyOptimism") {
            cfg.get_double(&key_for("policyOptimism"), 0.0, 1.0)
                .map_err(to_string_error)?
        } else {
            match setup_for {
                SetupFor::Distributed | SetupFor::Other => 0.0,
                _ => 1.0,
            }
        };

        params.value_weight_exponent = if contains_idx("valueWeightExponent") {
            cfg.get_double(&key_for("valueWeightExponent"), 0.0, 1.0)
                .map_err(to_string_error)?
        } else {
            0.25
        };
        params.use_noise_pruning = if contains_idx("useNoisePruning") {
            cfg.get_bool(&key_for("useNoisePruning"))
                .map_err(to_string_error)?
        } else {
            setup_for != SetupFor::Distributed && setup_for != SetupFor::Other
        };
        params.noise_prune_utility_scale = if contains_idx("noisePruneUtilityScale") {
            cfg.get_double(&key_for("noisePruneUtilityScale"), 0.001, 10.0)
                .map_err(to_string_error)?
        } else {
            0.15
        };
        params.noise_pruning_cap = if contains_idx("noisePruningCap") {
            cfg.get_double(&key_for("noisePruningCap"), 0.0, 1.0e50)
                .map_err(to_string_error)?
        } else {
            1.0e50
        };

        params.use_uncertainty = if contains_idx("useUncertainty") {
            cfg.get_bool(&key_for("useUncertainty"))
                .map_err(to_string_error)?
        } else {
            setup_for != SetupFor::Distributed && setup_for != SetupFor::Other
        };
        params.uncertainty_coeff = if contains_idx("uncertaintyCoeff") {
            cfg.get_double(&key_for("uncertaintyCoeff"), 0.0001, 1.0)
                .map_err(to_string_error)?
        } else {
            0.25
        };
        params.uncertainty_exponent = if contains_idx("uncertaintyExponent") {
            cfg.get_double(&key_for("uncertaintyExponent"), 0.0, 2.0)
                .map_err(to_string_error)?
        } else {
            1.0
        };
        params.uncertainty_max_weight = if contains_idx("uncertaintyMaxWeight") {
            cfg.get_double(&key_for("uncertaintyMaxWeight"), 1.0, 100.0)
                .map_err(to_string_error)?
        } else {
            8.0
        };

        params.use_graph_search = if contains_idx("useGraphSearch") {
            cfg.get_bool(&key_for("useGraphSearch"))
                .map_err(to_string_error)?
        } else {
            setup_for != SetupFor::Distributed
        };
        params.graph_search_rep_bound = if contains_idx("graphSearchRepBound") {
            cfg.get_int(&key_for("graphSearchRepBound"), 3, 50)
                .map_err(to_string_error)?
        } else {
            11
        };
        params.graph_search_catch_up_leak_prob = if contains_idx("graphSearchCatchUpLeakProb") {
            cfg.get_double(&key_for("graphSearchCatchUpLeakProb"), 0.0, 1.0)
                .map_err(to_string_error)?
        } else {
            0.0
        };

        params.root_noise_enabled = if contains_idx("rootNoiseEnabled") {
            cfg.get_bool(&key_for("rootNoiseEnabled"))
                .map_err(to_string_error)?
        } else {
            false
        };
        params.root_dirichlet_noise_total_concentration =
            if contains_idx("rootDirichletNoiseTotalConcentration") {
                cfg.get_double(
                    &key_for("rootDirichletNoiseTotalConcentration"),
                    0.001,
                    10_000.0,
                )
                .map_err(to_string_error)?
            } else {
                10.83
            };
        params.root_dirichlet_noise_weight = if contains_idx("rootDirichletNoiseWeight") {
            cfg.get_double(&key_for("rootDirichletNoiseWeight"), 0.0, 1.0)
                .map_err(to_string_error)?
        } else {
            0.25
        };

        params.root_policy_temperature = if contains_idx("rootPolicyTemperature") {
            cfg.get_double(&key_for("rootPolicyTemperature"), 0.01, 100.0)
                .map_err(to_string_error)?
        } else {
            1.0
        };
        params.root_policy_temperature_early = if contains_idx("rootPolicyTemperatureEarly") {
            cfg.get_double(&key_for("rootPolicyTemperatureEarly"), 0.01, 100.0)
                .map_err(to_string_error)?
        } else {
            params.root_policy_temperature
        };
        params.root_fpu_reduction_max = if contains_idx("rootFpuReductionMax") {
            cfg.get_double(&key_for("rootFpuReductionMax"), 0.0, 2.0)
                .map_err(to_string_error)?
        } else if params.root_noise_enabled {
            0.0
        } else {
            0.1
        };
        params.root_fpu_loss_prop = if contains_idx("rootFpuLossProp") {
            cfg.get_double(&key_for("rootFpuLossProp"), 0.0, 1.0)
                .map_err(to_string_error)?
        } else {
            params.fpu_loss_prop
        };
        params.root_num_symmetries_to_sample = if contains_idx("rootNumSymmetriesToSample") {
            cfg.get_int(&key_for("rootNumSymmetriesToSample"), 1, NUM_SYMMETRIES)
                .map_err(to_string_error)?
        } else {
            1
        };
        params.root_symmetry_pruning = if contains_idx("rootSymmetryPruning") {
            cfg.get_bool(&key_for("rootSymmetryPruning"))
                .map_err(to_string_error)?
        } else {
            matches!(setup_for, SetupFor::Analysis | SetupFor::Gtp)
        };

        params.root_desired_per_child_visits_coeff =
            if contains_idx("rootDesiredPerChildVisitsCoeff") {
                cfg.get_double(&key_for("rootDesiredPerChildVisitsCoeff"), 0.0, 100.0)
                    .map_err(to_string_error)?
            } else {
                0.0
            };

        params.root_policy_optimism = if contains_idx("rootPolicyOptimism") {
            cfg.get_double(&key_for("rootPolicyOptimism"), 0.0, 1.0)
                .map_err(to_string_error)?
        } else {
            match setup_for {
                SetupFor::Distributed | SetupFor::Other => 0.0,
                _ => params.policy_optimism.min(0.2),
            }
        };

        params.chosen_move_temperature = if contains_idx("chosenMoveTemperature") {
            cfg.get_double(&key_for("chosenMoveTemperature"), 0.0, 5.0)
                .map_err(to_string_error)?
        } else {
            0.1
        };
        params.chosen_move_temperature_early = if contains_idx("chosenMoveTemperatureEarly") {
            cfg.get_double(&key_for("chosenMoveTemperatureEarly"), 0.0, 5.0)
                .map_err(to_string_error)?
        } else {
            0.5
        };
        params.chosen_move_temperature_halflife = if contains_idx("chosenMoveTemperatureHalflife") {
            cfg.get_double(&key_for("chosenMoveTemperatureHalflife"), 0.1, 100_000.0)
                .map_err(to_string_error)?
        } else {
            19.0
        };
        params.chosen_move_temperature_only_below_prob =
            if contains_idx("chosenMoveTemperatureOnlyBelowProb") {
                cfg.get_double(&key_for("chosenMoveTemperatureOnlyBelowProb"), 0.0, 1.0)
                    .map_err(to_string_error)?
            } else {
                1.0
            };
        params.chosen_move_subtract = if contains_idx("chosenMoveSubtract") {
            cfg.get_double(&key_for("chosenMoveSubtract"), 0.0, 1.0e10)
                .map_err(to_string_error)?
        } else {
            0.0
        };
        params.chosen_move_prune = if contains_idx("chosenMovePrune") {
            cfg.get_double(&key_for("chosenMovePrune"), 0.0, 1.0e10)
                .map_err(to_string_error)?
        } else {
            1.0
        };

        params.use_lcb_for_selection = if contains_idx("useLcbForSelection") {
            cfg.get_bool(&key_for("useLcbForSelection"))
                .map_err(to_string_error)?
        } else {
            true
        };
        params.lcb_stdevs = if contains_idx("lcbStdevs") {
            cfg.get_double(&key_for("lcbStdevs"), 1.0, 12.0)
                .map_err(to_string_error)?
        } else {
            5.0
        };
        params.min_visit_prop_for_lcb = if contains_idx("minVisitPropForLCB") {
            cfg.get_double(&key_for("minVisitPropForLCB"), 0.0, 1.0)
                .map_err(to_string_error)?
        } else {
            0.15
        };
        params.use_non_buggy_lcb = if contains_idx("useNonBuggyLcb") {
            cfg.get_bool(&key_for("useNonBuggyLcb"))
                .map_err(to_string_error)?
        } else {
            setup_for != SetupFor::Distributed && setup_for != SetupFor::Other
        };

        params.root_ending_bonus_points = if contains_idx("rootEndingBonusPoints") {
            cfg.get_double(&key_for("rootEndingBonusPoints"), -1.0, 1.0)
                .map_err(to_string_error)?
        } else {
            0.5
        };
        params.root_prune_useless_moves = if contains_idx("rootPruneUselessMoves") {
            cfg.get_bool(&key_for("rootPruneUselessMoves"))
                .map_err(to_string_error)?
        } else {
            true
        };
        params.conservative_pass = if contains_idx("conservativePass") {
            cfg.get_bool(&key_for("conservativePass"))
                .map_err(to_string_error)?
        } else {
            false
        };
        params.fill_dame_before_pass = if contains_idx("fillDameBeforePass") {
            cfg.get_bool(&key_for("fillDameBeforePass"))
                .map_err(to_string_error)?
        } else {
            false
        };
        params.avoid_mytd_dagger_hack_pla = C_EMPTY;
        params.wide_root_noise = if contains_idx("wideRootNoise") {
            cfg.get_double(&key_for("wideRootNoise"), 0.0, 5.0)
                .map_err(to_string_error)?
        } else {
            match setup_for {
                SetupFor::Analysis => DEFAULT_ANALYSIS_WIDE_ROOT_NOISE,
                _ => 0.0,
            }
        };

        params.enable_passing_hacks = if contains_idx("enablePassingHacks") {
            cfg.get_bool(&key_for("enablePassingHacks"))
                .map_err(to_string_error)?
        } else {
            matches!(setup_for, SetupFor::Gtp | SetupFor::Analysis)
        };
        params.enable_more_passing_hacks = if contains_idx("enableMorePassingHacks") {
            cfg.get_bool(&key_for("enableMorePassingHacks"))
                .map_err(to_string_error)?
        } else {
            matches!(setup_for, SetupFor::Gtp | SetupFor::Analysis)
        };

        params.playout_doubling_advantage = if contains_idx("playoutDoublingAdvantage") {
            cfg.get_double(&key_for("playoutDoublingAdvantage"), -3.0, 3.0)
                .map_err(to_string_error)?
        } else {
            0.0
        };
        params.playout_doubling_advantage_pla = if contains_idx("playoutDoublingAdvantagePla") {
            parse_player(
                "playoutDoublingAdvantagePla",
                &cfg.get_string(&key_for("playoutDoublingAdvantagePla"))
                    .map_err(to_string_error)?,
            )?
        } else {
            C_EMPTY
        };

        params.avoid_repeated_pattern_utility = if contains_idx("avoidRepeatedPatternUtility") {
            cfg.get_double(&key_for("avoidRepeatedPatternUtility"), -3.0, 3.0)
                .map_err(to_string_error)?
        } else {
            0.0
        };

        params.nn_policy_temperature = if contains_idx("nnPolicyTemperature") {
            cfg.get_float(&key_for("nnPolicyTemperature"), 0.01, 5.0)
                .map_err(to_string_error)?
        } else {
            1.0
        };

        params.anti_mirror = if contains_idx("antiMirror") {
            cfg.get_bool(&key_for("antiMirror"))
                .map_err(to_string_error)?
        } else {
            false
        };

        params.ignore_pre_root_history = if contains_idx("ignorePreRootHistory") {
            cfg.get_bool(&key_for("ignorePreRootHistory"))
                .map_err(to_string_error)?
        } else {
            match setup_for {
                SetupFor::Analysis => DEFAULT_ANALYSIS_IGNORE_PRE_ROOT_HISTORY,
                _ => false,
            }
        };
        params.ignore_all_history = if contains_idx("ignoreAllHistory") {
            cfg.get_bool(&key_for("ignoreAllHistory"))
                .map_err(to_string_error)?
        } else {
            false
        };

        params.subtree_value_bias_factor = if contains_idx("subtreeValueBiasFactor") {
            cfg.get_double(&key_for("subtreeValueBiasFactor"), 0.0, 1.0)
                .map_err(to_string_error)?
        } else {
            0.45
        };
        params.subtree_value_bias_free_prop = if contains_idx("subtreeValueBiasFreeProp") {
            cfg.get_double(&key_for("subtreeValueBiasFreeProp"), 0.0, 1.0)
                .map_err(to_string_error)?
        } else {
            0.8
        };
        params.subtree_value_bias_weight_exponent =
            if contains_idx("subtreeValueBiasWeightExponent") {
                cfg.get_double(&key_for("subtreeValueBiasWeightExponent"), 0.0, 1.0)
                    .map_err(to_string_error)?
            } else {
                0.85
            };

        params.use_eval_cache = if contains_idx("useEvalCache") {
            cfg.get_bool(&key_for("useEvalCache"))
                .map_err(to_string_error)?
        } else {
            false
        };

        params.eval_cache_min_visits = if contains_idx("evalCacheMinVisits") {
            cfg.get_int64(&key_for("evalCacheMinVisits"), 1, 1_i64 << 50)
                .map_err(to_string_error)?
        } else {
            100
        };

        params.node_table_shards_power_of_two = if contains_idx("nodeTableShardsPowerOfTwo") {
            cfg.get_int(&key_for("nodeTableShardsPowerOfTwo"), 8, 24)
                .map_err(to_string_error)?
        } else {
            16
        };
        params.num_virtual_losses_per_thread = if contains_idx("numVirtualLossesPerThread") {
            cfg.get_double(&key_for("numVirtualLossesPerThread"), 0.01, 1000.0)
                .map_err(to_string_error)?
        } else {
            1.0
        };

        params.tree_reuse_carry_over_time_factor = if contains_idx("treeReuseCarryOverTimeFactor") {
            cfg.get_double(&key_for("treeReuseCarryOverTimeFactor"), 0.0, 1.0)
                .map_err(to_string_error)?
        } else {
            0.0
        };
        params.overallocate_time_factor = if contains_idx("overallocateTimeFactor") {
            cfg.get_double(&key_for("overallocateTimeFactor"), 0.01, 100.0)
                .map_err(to_string_error)?
        } else {
            1.0
        };
        params.midgame_time_factor = if contains_idx("midgameTimeFactor") {
            cfg.get_double(&key_for("midgameTimeFactor"), 0.01, 100.0)
                .map_err(to_string_error)?
        } else {
            1.0
        };
        params.midgame_turn_peak_time = if contains_idx("midgameTurnPeakTime") {
            cfg.get_double(&key_for("midgameTurnPeakTime"), 0.0, 1000.0)
                .map_err(to_string_error)?
        } else {
            130.0
        };
        params.endgame_turn_time_decay = if contains_idx("endgameTurnTimeDecay") {
            cfg.get_double(&key_for("endgameTurnTimeDecay"), 0.0, 1000.0)
                .map_err(to_string_error)?
        } else {
            100.0
        };
        params.obvious_moves_time_factor = if contains_idx("obviousMovesTimeFactor") {
            cfg.get_double(&key_for("obviousMovesTimeFactor"), 0.01, 1.0)
                .map_err(to_string_error)?
        } else {
            1.0
        };
        params.obvious_moves_policy_entropy_tolerance =
            if contains_idx("obviousMovesPolicyEntropyTolerance") {
                cfg.get_double(&key_for("obviousMovesPolicyEntropyTolerance"), 0.001, 2.0)
                    .map_err(to_string_error)?
            } else {
                0.30
            };
        params.obvious_moves_policy_surprise_tolerance =
            if contains_idx("obviousMovesPolicySurpriseTolerance") {
                cfg.get_double(&key_for("obviousMovesPolicySurpriseTolerance"), 0.001, 2.0)
                    .map_err(to_string_error)?
            } else {
                0.15
            };
        params.futile_visits_threshold = if contains_idx("futileVisitsThreshold") {
            cfg.get_double(&key_for("futileVisitsThreshold"), 0.01, 1.0)
                .map_err(to_string_error)?
        } else {
            0.0
        };

        if setup_for != SetupFor::Distributed {
            let human_sl_profile_name = if contains_idx("humanSLProfile") {
                cfg.get_string(&key_for("humanSLProfile"))
                    .map_err(to_string_error)?
            } else {
                String::new()
            };
            params.human_sl_profile =
                sgf_meta::get_profile(&human_sl_profile_name).map_err(to_string_error)?;
        }

        let human_sl_keys = [
            "humanSLCpuctExploration",
            "humanSLCpuctPermanent",
            "humanSLRootExploreProbWeightless",
            "humanSLRootExploreProbWeightful",
            "humanSLPlaExploreProbWeightless",
            "humanSLPlaExploreProbWeightful",
            "humanSLOppExploreProbWeightless",
            "humanSLOppExploreProbWeightful",
            "humanSLChosenMoveProp",
            "humanSLChosenMoveIgnorePass",
            "humanSLChosenMovePiklLambda",
        ];
        for &key in &human_sl_keys {
            if contains_idx(key) && !has_human_model {
                return Err(StringError::new(format!(
                    "Provided parameter {} but no human model was specified (e.g -human-model b18c384nbt-humanv0.bin.gz)",
                    key_for(key)
                )));
            }
        }

        params.human_sl_cpuct_exploration = if contains_idx("humanSLCpuctExploration") {
            cfg.get_double(&key_for("humanSLCpuctExploration"), 0.0, 1000.0)
                .map_err(to_string_error)?
        } else {
            1.0
        };
        params.human_sl_cpuct_permanent = if contains_idx("humanSLCpuctPermanent") {
            cfg.get_double(&key_for("humanSLCpuctPermanent"), 0.0, 1000.0)
                .map_err(to_string_error)?
        } else {
            0.0
        };
        params.human_sl_root_explore_prob_weightless =
            if contains_idx("humanSLRootExploreProbWeightless") {
                cfg.get_double(&key_for("humanSLRootExploreProbWeightless"), 0.0, 1.0)
                    .map_err(to_string_error)?
            } else {
                0.0
            };
        params.human_sl_root_explore_prob_weightful =
            if contains_idx("humanSLRootExploreProbWeightful") {
                cfg.get_double(&key_for("humanSLRootExploreProbWeightful"), 0.0, 1.0)
                    .map_err(to_string_error)?
            } else {
                0.0
            };
        params.human_sl_pla_explore_prob_weightless =
            if contains_idx("humanSLPlaExploreProbWeightless") {
                cfg.get_double(&key_for("humanSLPlaExploreProbWeightless"), 0.0, 1.0)
                    .map_err(to_string_error)?
            } else {
                0.0
            };
        params.human_sl_pla_explore_prob_weightful =
            if contains_idx("humanSLPlaExploreProbWeightful") {
                cfg.get_double(&key_for("humanSLPlaExploreProbWeightful"), 0.0, 1.0)
                    .map_err(to_string_error)?
            } else {
                0.0
            };
        params.human_sl_opp_explore_prob_weightless =
            if contains_idx("humanSLOppExploreProbWeightless") {
                cfg.get_double(&key_for("humanSLOppExploreProbWeightless"), 0.0, 1.0)
                    .map_err(to_string_error)?
            } else {
                0.0
            };
        params.human_sl_opp_explore_prob_weightful =
            if contains_idx("humanSLOppExploreProbWeightful") {
                cfg.get_double(&key_for("humanSLOppExploreProbWeightful"), 0.0, 1.0)
                    .map_err(to_string_error)?
            } else {
                0.0
            };
        params.human_sl_chosen_move_prop = if contains_idx("humanSLChosenMoveProp") {
            cfg.get_double(&key_for("humanSLChosenMoveProp"), 0.0, 1.0)
                .map_err(to_string_error)?
        } else {
            0.0
        };
        params.human_sl_chosen_move_ignore_pass = if contains_idx("humanSLChosenMoveIgnorePass") {
            cfg.get_bool(&key_for("humanSLChosenMoveIgnorePass"))
                .map_err(to_string_error)?
        } else {
            false
        };
        params.human_sl_chosen_move_pikl_lambda = if contains_idx("humanSLChosenMovePiklLambda") {
            cfg.get_double(
                &key_for("humanSLChosenMovePiklLambda"),
                0.0,
                1_000_000_000.0,
            )
            .map_err(to_string_error)?
        } else {
            1_000_000_000.0
        };

        if setup_for == SetupFor::Distributed {
            cfg.mark_all_keys_used_with_prefix("mutexPoolSize");
        }

        params_vec.push(params);
    }

    Ok(params_vec)
}

/// Load a single set of game rules from config.
///
/// Mirrors `Setup::loadSingleRules` in `cpp/program/setup.cpp`.
///
/// If `load_komi` is true, the `komi` key is read from the config; otherwise
/// a default komi is left in place.
pub fn load_single_rules(cfg: &ConfigParser, load_komi: bool) -> Result<Rules, StringError> {
    let mut rules = Rules::default();

    if cfg.contains("rules") {
        for key in [
            "koRule",
            "scoringRule",
            "multiStoneSuicideLegal",
            "hasButton",
            "taxRule",
            "whiteHandicapBonus",
            "friendlyPassOk",
            "whiteBonusPerHandicapStone",
        ] {
            if cfg.contains(key) {
                return Err(StringError::new(format!(
                    "Cannot both specify 'rules' and individual rules like {}",
                    key
                )));
            }
        }
        rules = Rules::parse_rules(&cfg.get_string("rules").map_err(to_string_error)?)
            .map_err(to_string_error)?;
    } else {
        let ko_rule_str = cfg
            .get_string_set("koRule", &string_set(&Rules::ko_rule_strings()))
            .map_err(to_string_error)?;
        let scoring_rule_str = cfg
            .get_string_set("scoringRule", &string_set(&Rules::scoring_rule_strings()))
            .map_err(to_string_error)?;
        let multi_stone_suicide_legal = cfg
            .get_bool("multiStoneSuicideLegal")
            .map_err(to_string_error)?;
        let has_button = if cfg.contains("hasButton") {
            cfg.get_bool("hasButton").map_err(to_string_error)?
        } else {
            false
        };

        rules.ko_rule = ko_rule_str
            .parse::<KoRule>()
            .map_err(|e| StringError::new(e.to_string()))?;
        rules.scoring_rule = scoring_rule_str
            .parse::<ScoringRule>()
            .map_err(|e| StringError::new(e.to_string()))?;
        rules.multi_stone_suicide_legal = multi_stone_suicide_legal;
        rules.has_button = has_button;

        if cfg.contains("taxRule") {
            let tax_rule_str = cfg
                .get_string_set("taxRule", &string_set(&Rules::tax_rule_strings()))
                .map_err(to_string_error)?;
            rules.tax_rule = tax_rule_str
                .parse::<TaxRule>()
                .map_err(|e| StringError::new(e.to_string()))?;
        } else {
            rules.tax_rule = if rules.scoring_rule == ScoringRule::Territory {
                TaxRule::Seki
            } else {
                TaxRule::None
            };
        }

        if rules.has_button && rules.scoring_rule != ScoringRule::Area {
            return Err(StringError::new(
                "Config specifies hasButton=true on a scoring system other than AREA",
            ));
        }

        if cfg.contains("whiteBonusPerHandicapStone") && cfg.contains("whiteHandicapBonus") {
            return Err(StringError::new(
                "May specify only one of whiteBonusPerHandicapStone and whiteHandicapBonus in config",
            ));
        } else if cfg.contains("whiteHandicapBonus") {
            let whb_str = cfg
                .get_string_set(
                    "whiteHandicapBonus",
                    &string_set(&Rules::white_handicap_bonus_rule_strings()),
                )
                .map_err(to_string_error)?;
            rules.white_handicap_bonus_rule = whb_str
                .parse::<WhiteHandicapBonusRule>()
                .map_err(|e| StringError::new(e.to_string()))?;
        } else if cfg.contains("whiteBonusPerHandicapStone") {
            let white_bonus_per_handicap_stone = cfg
                .get_int("whiteBonusPerHandicapStone", 0, 1)
                .map_err(to_string_error)?;
            rules.white_handicap_bonus_rule = if white_bonus_per_handicap_stone == 0 {
                WhiteHandicapBonusRule::Zero
            } else {
                WhiteHandicapBonusRule::N
            };
        } else {
            rules.white_handicap_bonus_rule = WhiteHandicapBonusRule::Zero;
        }

        if cfg.contains("friendlyPassOk") {
            rules.friendly_pass_ok = cfg.get_bool("friendlyPassOk").map_err(to_string_error)?;
        }

        // Drop default komi to 6.5 for territory rules, and to 7.0 for button.
        if rules.scoring_rule == ScoringRule::Territory {
            rules.set_komi(6.5);
        } else if rules.has_button {
            rules.set_komi(7.0);
        }
    }

    if load_komi {
        let komi = cfg
            .get_float("komi", Rules::MIN_USER_KOMI, Rules::MAX_USER_KOMI)
            .map_err(to_string_error)?;
        rules.set_komi(komi);
    }

    Ok(rules)
}

/// Load the default board size from config.
///
/// Mirrors `Setup::loadDefaultBoardXYSize` in `cpp/program/setup.cpp`.
///
/// Returns `true` if the config specifies the size. If `false`, the out
/// parameters are left unchanged.
pub fn load_default_board_xy_size(
    cfg: &ConfigParser,
    logger: &Logger,
    default_board_x_size_ret: &mut i32,
    default_board_y_size_ret: &mut i32,
) -> Result<bool, StringError> {
    let max_len = kata_game::board::MAX_LEN as i32;
    let default_board_x_size = if cfg.contains("defaultBoardXSize") {
        cfg.get_int("defaultBoardXSize", 2, max_len)
            .map_err(to_string_error)?
    } else if cfg.contains("defaultBoardSize") {
        cfg.get_int("defaultBoardSize", 2, max_len)
            .map_err(to_string_error)?
    } else {
        -1
    };

    let default_board_y_size = if cfg.contains("defaultBoardYSize") {
        cfg.get_int("defaultBoardYSize", 2, max_len)
            .map_err(to_string_error)?
    } else if cfg.contains("defaultBoardSize") {
        cfg.get_int("defaultBoardSize", 2, max_len)
            .map_err(to_string_error)?
    } else {
        -1
    };

    if (default_board_x_size == -1) != (default_board_y_size == -1) {
        logger.write("Warning: Config specified only one of defaultBoardXSize or defaultBoardYSize and no other board size parameter, ignoring it");
    }

    if default_board_x_size == -1 || default_board_y_size == -1 {
        return Ok(false);
    }

    *default_board_x_size_ret = default_board_x_size;
    *default_board_y_size_ret = default_board_y_size;
    Ok(true)
}

/// Load pattern-bonus tables that penalize repeating moves from user-supplied SGFs.
///
/// Mirrors `Setup::loadAvoidSgfPatternBonusTables` in `cpp/program/setup.cpp`.
pub fn load_avoid_sgf_pattern_bonus_tables(
    cfg: &ConfigParser,
    logger: &Logger,
) -> Result<Vec<Option<PatternBonusTable>>, StringError> {
    let mut num_bots = 1;
    if cfg.contains("numBots") {
        num_bots = cfg
            .get_int("numBots", 1, MAX_BOT_PARAMS_FROM_CFG)
            .map_err(to_string_error)?;
    }

    let mut tables = Vec::new();
    for i in 0..num_bots {
        let idx_str = global::int_to_string(i);

        let mut pattern_bonus_table: Option<PatternBonusTable> = None;
        for j in 1..100_000 {
            let set_str = if j == 1 {
                String::new()
            } else {
                global::int_to_string(j)
            };
            let prefix = format!("avoidSgf{}", set_str);

            let contains = |suffix: &str| -> bool {
                let keys = vec![
                    format!("{}{}{}", prefix, suffix, idx_str),
                    format!("{}{}", prefix, suffix),
                ];
                cfg.contains_any(&keys)
            };
            let find = |suffix: &str| -> Result<String, StringError> {
                let keys = vec![
                    format!("{}{}{}", prefix, suffix, idx_str),
                    format!("{}{}", prefix, suffix),
                ];
                cfg.first_found_or_fail(&keys).map_err(to_string_error)
            };

            if contains("PatternUtility") {
                let penalty = cfg
                    .get_double(&find("PatternUtility")?, -3.0, 3.0)
                    .map_err(to_string_error)?;
                let lambda = if contains("PatternLambda") {
                    cfg.get_double(&find("PatternLambda")?, 0.0, 1.0)
                        .map_err(to_string_error)?
                } else {
                    1.0
                };
                let min_turn_number = if contains("PatternMinTurnNumber") {
                    cfg.get_int(&find("PatternMinTurnNumber")?, 0, 1_000_000)
                        .map_err(to_string_error)? as i64
                } else {
                    0
                };
                let max_files = if contains("PatternMaxFiles") {
                    cfg.get_int(&find("PatternMaxFiles")?, 1, 1_000_000)
                        .map_err(to_string_error)? as usize
                } else {
                    1_000_000
                };
                let allowed_player_names = if contains("PatternAllowedNames") {
                    cfg.get_strings_non_empty_trim(&find("PatternAllowedNames")?)
                        .map_err(to_string_error)?
                } else {
                    Vec::new()
                };
                let sgf_dirs = cfg
                    .get_strings(&find("PatternDirs")?)
                    .map_err(to_string_error)?;

                if pattern_bonus_table.is_none() {
                    pattern_bonus_table = Some(PatternBonusTable::new());
                }
                let log_source = format!("bot {}", idx_str);
                if let Some(table) = pattern_bonus_table.as_ref() {
                    table.avoid_repeated_sgf_moves(
                        &sgf_dirs,
                        penalty,
                        lambda,
                        min_turn_number,
                        max_files,
                        &allowed_player_names,
                        logger,
                        &log_source,
                    );
                }
            }
        }
        tables.push(pattern_bonus_table);
    }
    Ok(tables)
}

/// Save patterns to avoid repeating in the future.
///
/// Mirrors `Setup::saveAutoPatternBonusData` in `cpp/program/setup.cpp`.
///
/// Returns `Ok(true)` if data was saved, `Ok(false)` if there was nothing to save
/// or no output directory configured. Write failures are logged and return `Ok(false)`.
#[allow(clippy::map_entry)]
pub fn save_auto_pattern_bonus_data(
    genmove_samples: &[PositionSample],
    cfg: &ConfigParser,
    logger: &Logger,
    rand: &mut Rand,
) -> Result<bool, StringError> {
    if genmove_samples.is_empty() {
        return Ok(false);
    }
    if !cfg.contains("autoAvoidRepeatDir") {
        return Ok(false);
    }

    let auto_avoid_patterns_dir = cfg
        .get_string("autoAvoidRepeatDir")
        .map_err(to_string_error)?;
    fs::make_dir(&auto_avoid_patterns_dir)?;

    let file_name = format!(
        "{}_poses.txt",
        global::uint64_to_hex_string(rand.next_u64())
    );

    let mut out_by_board_size: BTreeMap<(i32, i32), std::fs::File> = BTreeMap::new();

    for sample in genmove_samples {
        let board_x_size = sample.board.x_size;
        let board_y_size = sample.board.y_size;
        let board_size = (board_x_size, board_y_size);

        let min_turn_number = get_auto_pattern_int_param(
            cfg,
            "autoAvoidRepeatMinTurnNumber",
            board_x_size,
            board_y_size,
            0,
            1_000_000,
        )? as i64;
        let max_turn_number = get_auto_pattern_int_param(
            cfg,
            "autoAvoidRepeatMaxTurnNumber",
            board_x_size,
            board_y_size,
            0,
            1_000_000,
        )? as i64;
        if sample.initial_turn_number < min_turn_number
            || sample.initial_turn_number > max_turn_number
        {
            continue;
        }
        assert!(sample.moves.is_empty());

        if !out_by_board_size.contains_key(&board_size) {
            let sub_dir = format!(
                "{}/{}",
                auto_avoid_patterns_dir,
                board_size_to_str(board_x_size, board_y_size)
            );
            fs::make_dir(&sub_dir)?;
            let file_path = format!("{}/{}", sub_dir, file_name);
            match std::fs::File::create(&file_path) {
                Ok(file) => {
                    out_by_board_size.insert(board_size, file);
                }
                Err(e) => {
                    logger.write(&format!("ERROR: could not open {}: {}", file_path, e));
                    return Ok(false);
                }
            }
        }

        let line = PositionSample::to_json_line(sample);
        if let Some(file) = out_by_board_size.get_mut(&board_size) {
            if writeln!(file, "{}", line).is_err() {
                logger.write("ERROR: could not write to auto avoid pattern file");
                return Ok(false);
            }
        }
    }

    for file in out_by_board_size.values_mut() {
        let _ = file.flush();
    }

    logger.write(&format!(
        "Saved {} avoid poses to {}",
        global::uint64_to_string(genmove_samples.len() as u64),
        auto_avoid_patterns_dir
    ));
    Ok(true)
}

/// Load and prune auto pattern bonus tables from previously saved positions.
///
/// Mirrors `Setup::loadAndPruneAutoPatternBonusTables` in `cpp/program/setup.cpp`.
pub fn load_and_prune_auto_pattern_bonus_tables(
    cfg: &ConfigParser,
    logger: &Logger,
) -> Result<Option<PatternBonusTable>, StringError> {
    let mut pattern_bonus_table: Option<PatternBonusTable> = None;

    if cfg.contains("autoAvoidRepeatDir") {
        let base_dir = cfg
            .get_string("autoAvoidRepeatDir")
            .map_err(to_string_error)?;
        let board_size_dirs = fs::list_files(&base_dir).map_err(to_string_error)?;

        pattern_bonus_table = Some(PatternBonusTable::new());

        for dir_name in board_size_dirs {
            let pieces: Vec<&str> = dir_name.split('x').collect();
            if pieces.len() != 2 {
                continue;
            }
            let board_x_size = global::try_string_to_int(pieces[0]);
            let board_y_size = global::try_string_to_int(pieces[1]);
            let (board_x_size, board_y_size) = match (board_x_size, board_y_size) {
                (Some(x), Some(y)) => (x, y),
                _ => continue,
            };
            let max_len = kata_game::board::MAX_LEN as i32;
            if board_x_size < 2
                || board_x_size > max_len
                || board_y_size < 2
                || board_y_size > max_len
            {
                continue;
            }

            let dir_path = format!("{}/{}", base_dir, dir_name);
            if !fs::is_directory(&dir_path) {
                continue;
            }

            let penalty = get_auto_pattern_double_param(
                cfg,
                "autoAvoidRepeatUtility",
                board_x_size,
                board_y_size,
                -3.0,
                3.0,
            )?;
            let lambda = get_auto_pattern_double_param(
                cfg,
                "autoAvoidRepeatLambda",
                board_x_size,
                board_y_size,
                0.0,
                1.0,
            )?;
            let min_turn_number = get_auto_pattern_int_param(
                cfg,
                "autoAvoidRepeatMinTurnNumber",
                board_x_size,
                board_y_size,
                0,
                1_000_000,
            )? as i64;
            let max_turn_number = get_auto_pattern_int_param(
                cfg,
                "autoAvoidRepeatMaxTurnNumber",
                board_x_size,
                board_y_size,
                0,
                1_000_000,
            )? as i64;
            let max_poses = get_auto_pattern_int64_param(
                cfg,
                "autoAvoidRepeatMaxPoses",
                board_x_size,
                board_y_size,
                0,
                1_000_000_000_000,
            )? as usize;

            let log_source = dir_path.clone();
            if let Some(table) = pattern_bonus_table.as_ref() {
                table.avoid_repeated_pos_moves_and_delete_excess_files(
                    &[dir_path],
                    penalty,
                    lambda,
                    min_turn_number,
                    max_turn_number,
                    max_poses,
                    logger,
                    &log_source,
                );
            }
        }

        cfg.mark_all_keys_used_with_prefix("autoAvoidRepeatUtility");
        cfg.mark_all_keys_used_with_prefix("autoAvoidRepeatLambda");
        cfg.mark_all_keys_used_with_prefix("autoAvoidRepeatMinTurnNumber");
        cfg.mark_all_keys_used_with_prefix("autoAvoidRepeatMaxTurnNumber");
        cfg.mark_all_keys_used_with_prefix("autoAvoidRepeatMaxPoses");
        cfg.mark_all_keys_used_with_prefix("autoAvoidRepeatSaveChunkSize");
    }

    Ok(pattern_bonus_table)
}

/// Parse the `reportAnalysisWinratesAs` config key.
///
/// Mirrors `Setup::parseReportAnalysisWinrates` in `cpp/program/setup.cpp`.
///
/// Returns the requested perspective, or `default_perspective` if the key is not
/// present.
pub fn parse_report_analysis_winrates(
    cfg: &ConfigParser,
    default_perspective: Player,
) -> Result<Player, StringError> {
    if !cfg.contains("reportAnalysisWinratesAs") {
        return Ok(default_perspective);
    }

    let s_orig = cfg
        .get_string("reportAnalysisWinratesAs")
        .map_err(to_string_error)?;
    let s = global::to_lower(&s_orig);
    match s.as_str() {
        "b" | "black" => Ok(kata_game::board::P_BLACK),
        "w" | "white" => Ok(kata_game::board::P_WHITE),
        "sidetomove" => Ok(C_EMPTY),
        _ => Err(StringError::new(format!(
            "Could not parse config value for reportAnalysisWinratesAs: {}",
            s_orig
        ))),
    }
}

/// Warn if a human SL profile is configured but neither model uses SGF metadata.
///
/// Mirrors `Setup::maybeWarnHumanSLParams` in `cpp/program/setup.cpp`.
///
/// Returns `Ok(true)` if a warning was emitted.
pub fn maybe_warn_human_sl_params(
    params: &SearchParams,
    nn_eval: Option<&NnEvaluator>,
    human_eval: Option<&NnEvaluator>,
    out: &mut dyn Write,
    logger: Option<&Logger>,
) -> Result<bool, StringError> {
    if params.human_sl_profile.initialized {
        let has_any_sgf_meta_use = nn_eval.map(|e| e.requires_sgf_metadata()).unwrap_or(false)
            || human_eval
                .map(|e| e.requires_sgf_metadata())
                .unwrap_or(false);
        if !has_any_sgf_meta_use {
            let mut model_names = String::new();
            if let Some(nn) = nn_eval {
                model_names.push_str(nn.model_name());
            }
            if let Some(human) = human_eval {
                if !model_names.is_empty() {
                    model_names.push_str(" and ");
                }
                model_names.push_str(human.model_name());
            }

            let warning = format!(
                "WARNING: humanSLProfile is specified as config param but model(s) don't use it: {}",
                model_names
            );
            if let Some(log) = logger {
                log.write(&warning);
            }
            writeln!(out, "{}", warning).map_err(to_string_error)?;
            return Ok(true);
        }
    }
    Ok(false)
}

/// Convert a `BTreeSet<&'static str>` into `BTreeSet<String>`.
fn string_set(strs: &BTreeSet<&'static str>) -> BTreeSet<String> {
    strs.iter().map(|s| (*s).to_string()).collect()
}

/// Format a board size as "XxY".
fn board_size_to_str(board_x_size: i32, board_y_size: i32) -> String {
    format!("{}x{}", board_x_size, board_y_size)
}

/// Look up an auto-pattern integer parameter, optionally with a board-size suffix.
fn get_auto_pattern_int_param(
    cfg: &ConfigParser,
    param: &str,
    board_x_size: i32,
    board_y_size: i32,
    min: i32,
    max: i32,
) -> Result<i32, StringError> {
    let size_key = format!("{}{}", param, board_size_to_str(board_x_size, board_y_size));
    if cfg.contains(&size_key) {
        return cfg.get_int(&size_key, min, max).map_err(to_string_error);
    }
    if !cfg.contains(param) {
        return Err(StringError::new(format!(
            "{} was not specified in the config",
            param
        )));
    }
    cfg.get_int(param, min, max).map_err(to_string_error)
}

/// Look up an auto-pattern 64-bit integer parameter, optionally with a board-size suffix.
fn get_auto_pattern_int64_param(
    cfg: &ConfigParser,
    param: &str,
    board_x_size: i32,
    board_y_size: i32,
    min: i64,
    max: i64,
) -> Result<i64, StringError> {
    let size_key = format!("{}{}", param, board_size_to_str(board_x_size, board_y_size));
    if cfg.contains(&size_key) {
        return cfg.get_int64(&size_key, min, max).map_err(to_string_error);
    }
    if !cfg.contains(param) {
        return Err(StringError::new(format!(
            "{} was not specified in the config",
            param
        )));
    }
    cfg.get_int64(param, min, max).map_err(to_string_error)
}

/// Look up an auto-pattern floating-point parameter, optionally with a board-size suffix.
fn get_auto_pattern_double_param(
    cfg: &ConfigParser,
    param: &str,
    board_x_size: i32,
    board_y_size: i32,
    min: f64,
    max: f64,
) -> Result<f64, StringError> {
    let size_key = format!("{}{}", param, board_size_to_str(board_x_size, board_y_size));
    if cfg.contains(&size_key) {
        return cfg.get_double(&size_key, min, max).map_err(to_string_error);
    }
    if !cfg.contains(param) {
        return Err(StringError::new(format!(
            "{} was not specified in the config",
            param
        )));
    }
    cfg.get_double(param, min, max).map_err(to_string_error)
}

/// Initialize any global state needed by the neural-net backend.
///
/// Mirrors `Setup::initializeSession` in `cpp/program/setup.cpp`, which calls
/// `NeuralNet::globalInitialize()`. In the Rust port backend-specific global
/// initialization is performed when an evaluator creates its compute context,
/// so this function is intentionally a no-op.
pub fn initialize_session(_cfg: &ConfigParser) {
    // Backend global initialization is handled per-backend during evaluator setup.
}

/// Return the recognized neural-net backend name prefixes.
///
/// Mirrors `Setup::getBackendPrefixes` in `cpp/program/setup.cpp`.
pub fn get_backend_prefixes() -> Vec<String> {
    vec![
        "cuda".to_string(),
        "trt".to_string(),
        "metal".to_string(),
        "opencl".to_string(),
        "eigen".to_string(),
        "dummybackend".to_string(),
    ]
}

/// Select the neural-net backend name from the `nnBackend` config key.
///
/// `model_idx` selects the per-model key `nnBackend{i}`; `None` reads the
/// base `nnBackend` key. Returns `Ok(None)` when the key is absent, and a
/// normalized backend name otherwise ("dummybackend" / "trtbackend" /
/// "cudabackend" / "eigenbackend").
fn select_backend_prefix(
    cfg: &ConfigParser,
    model_idx: Option<usize>,
) -> Result<Option<String>, StringError> {
    let key = match model_idx {
        Some(i) => format!("nnBackend{i}"),
        None => "nnBackend".to_string(),
    };
    if !cfg.contains(&key) {
        return Ok(None);
    }
    let value = cfg.get_string(&key).map_err(to_string_error)?;
    let normalized = match value.as_str() {
        "dummy" | "dummybackend" => "dummybackend".to_string(),
        "trt" | "tensorrt" | "trtbackend" => "trtbackend".to_string(),
        "cuda" | "cudabackend" => "cudabackend".to_string(),
        "eigen" | "cpu" | "eigenbackend" => "eigenbackend".to_string(),
        _ => {
            return Err(StringError::new(format!(
                "Unknown nnBackend value: {value}"
            )))
        }
    };
    Ok(Some(normalized))
}

/// Compute a reasonable default number of Eigen backend threads.
///
/// Mirrors `Setup::computeDefaultEigenBackendThreads` in `cpp/program/setup.cpp`.
///
/// Uses `std::thread::available_parallelism()` to determine the number of cores.
/// If the value cannot be determined, logs a warning via `logger` and assumes 8
/// cores. The result is capped at `expected_concurrent_evals`.
pub fn compute_default_eigen_backend_threads(
    expected_concurrent_evals: i32,
    logger: &Logger,
) -> i32 {
    let num_cores = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or_else(|_| {
            logger.write(
                "WARNING: Unable to determine number of cores for Eigen backend threads, assuming 8",
            );
            8
        });
    expected_concurrent_evals.min(num_cores)
}

/// Initialize a single neural-net evaluator from config.
///
/// Mirrors `Setup::initializeNNEvaluator` in `cpp/program/setup.cpp`.
#[allow(clippy::too_many_arguments)]
pub fn initialize_nn_evaluator(
    nn_model_name: String,
    nn_model_file: String,
    expected_sha256: String,
    cfg: &ConfigParser,
    logger: &Logger,
    seed_rand: &mut Rand,
    expected_concurrent_evals: i32,
    default_nnx_len: i32,
    default_nny_len: i32,
    default_max_batch_size: i32,
    default_require_exact_nn_len: bool,
    disable_fp16: bool,
    setup_for: SetupFor,
) -> Result<NnEvaluator, StringError> {
    let evals = initialize_nn_evaluators(
        vec![nn_model_name],
        vec![nn_model_file],
        vec![expected_sha256],
        cfg,
        logger,
        seed_rand,
        expected_concurrent_evals,
        default_nnx_len,
        default_nny_len,
        default_max_batch_size,
        default_require_exact_nn_len,
        disable_fp16,
        setup_for,
    )?;
    Ok(evals.into_iter().next().expect("initialized one evaluator"))
}

/// Initialize multiple neural-net evaluators from a shared config.
///
/// Mirrors `Setup::initializeNNEvaluators` in `cpp/program/setup.cpp`.
#[allow(clippy::too_many_arguments)]
pub fn initialize_nn_evaluators(
    nn_model_names: Vec<String>,
    nn_model_files: Vec<String>,
    expected_sha256s: Vec<String>,
    cfg: &ConfigParser,
    logger: &Logger,
    seed_rand: &mut Rand,
    expected_concurrent_evals: i32,
    default_nnx_len: i32,
    default_nny_len: i32,
    default_max_batch_size: i32,
    default_require_exact_nn_len: bool,
    disable_fp16: bool,
    setup_for: SetupFor,
) -> Result<Vec<NnEvaluator>, StringError> {
    if nn_model_names.len() != nn_model_files.len() {
        return Err(StringError::new(
            "nn_model_names and nn_model_files must have the same size",
        ));
    }
    if !expected_sha256s.is_empty() && expected_sha256s.len() != nn_model_files.len() {
        return Err(StringError::new(
            "expected_sha256s must be empty or match nn_model_files size",
        ));
    }

    // Backend selection: `nnBackend` config key (per-model override via
    // `nnBackend{i}`). Defaults to the dummy backend.
    let default_backend_prefix = select_backend_prefix(cfg, None)?
        .unwrap_or_else(|| "dummybackend".to_string());

    // Flag keys for other backends as used so that unused-backend config keys
    // do not trigger unused-key warnings.
    for prefix in get_backend_prefixes() {
        if prefix != default_backend_prefix {
            cfg.mark_all_keys_used_with_prefix(&prefix);
        }
    }

    let mut nn_evals = Vec::new();

    for i in 0..nn_model_files.len() {
        let idx_str = i.to_string();
        let nn_model_name = &nn_model_names[i];
        let nn_model_file = &nn_model_files[i];
        let expected_sha256 = if expected_sha256s.is_empty() {
            String::new()
        } else {
            expected_sha256s[i].clone()
        };
        let backend_prefix = select_backend_prefix(cfg, Some(i))?
            .unwrap_or_else(|| default_backend_prefix.clone());

        let debug_skip_neural_net_default = nn_model_file == "/dev/null";
        let debug_skip_neural_net = if setup_for == SetupFor::Distributed {
            debug_skip_neural_net_default
        } else if cfg.contains("debugSkipNeuralNet") {
            cfg.get_bool("debugSkipNeuralNet")
                .map_err(to_string_error)?
        } else {
            debug_skip_neural_net_default
        };

        let mut nn_x_len = default_nnx_len.max(2);
        let mut nn_y_len = default_nny_len.max(2);
        if setup_for != SetupFor::Distributed {
            if cfg.contains(&("maxBoardXSizeForNNBuffer".to_string() + &idx_str)) {
                nn_x_len = cfg
                    .get_int(
                        &("maxBoardXSizeForNNBuffer".to_string() + &idx_str),
                        2,
                        nn_pos::MAX_BOARD_LEN as i32,
                    )
                    .map_err(to_string_error)?;
            } else if cfg.contains("maxBoardXSizeForNNBuffer") {
                nn_x_len = cfg
                    .get_int("maxBoardXSizeForNNBuffer", 2, nn_pos::MAX_BOARD_LEN as i32)
                    .map_err(to_string_error)?;
            } else if cfg.contains(&("maxBoardSizeForNNBuffer".to_string() + &idx_str)) {
                nn_x_len = cfg
                    .get_int(
                        &("maxBoardSizeForNNBuffer".to_string() + &idx_str),
                        2,
                        nn_pos::MAX_BOARD_LEN as i32,
                    )
                    .map_err(to_string_error)?;
            } else if cfg.contains("maxBoardSizeForNNBuffer") {
                nn_x_len = cfg
                    .get_int("maxBoardSizeForNNBuffer", 2, nn_pos::MAX_BOARD_LEN as i32)
                    .map_err(to_string_error)?;
            }

            if cfg.contains(&("maxBoardYSizeForNNBuffer".to_string() + &idx_str)) {
                nn_y_len = cfg
                    .get_int(
                        &("maxBoardYSizeForNNBuffer".to_string() + &idx_str),
                        2,
                        nn_pos::MAX_BOARD_LEN as i32,
                    )
                    .map_err(to_string_error)?;
            } else if cfg.contains("maxBoardYSizeForNNBuffer") {
                nn_y_len = cfg
                    .get_int("maxBoardYSizeForNNBuffer", 2, nn_pos::MAX_BOARD_LEN as i32)
                    .map_err(to_string_error)?;
            } else if cfg.contains(&("maxBoardSizeForNNBuffer".to_string() + &idx_str)) {
                nn_y_len = cfg
                    .get_int(
                        &("maxBoardSizeForNNBuffer".to_string() + &idx_str),
                        2,
                        nn_pos::MAX_BOARD_LEN as i32,
                    )
                    .map_err(to_string_error)?;
            } else if cfg.contains("maxBoardSizeForNNBuffer") {
                nn_y_len = cfg
                    .get_int("maxBoardSizeForNNBuffer", 2, nn_pos::MAX_BOARD_LEN as i32)
                    .map_err(to_string_error)?;
            }
        }

        let mut require_exact_nn_len = default_require_exact_nn_len;
        if setup_for != SetupFor::Distributed {
            if cfg.contains(&("requireMaxBoardSize".to_string() + &idx_str)) {
                require_exact_nn_len = cfg
                    .get_bool(&("requireMaxBoardSize".to_string() + &idx_str))
                    .map_err(to_string_error)?;
            } else if cfg.contains("requireMaxBoardSize") {
                require_exact_nn_len = cfg
                    .get_bool("requireMaxBoardSize")
                    .map_err(to_string_error)?;
            }
        }

        let mut inputs_use_nhwc = true;
        let prefixed_inputs_nhwc_idx = backend_prefix.to_string() + "InputsUseNHWC" + &idx_str;
        let inputs_nhwc_idx = "inputsUseNHWC".to_string() + &idx_str;
        let prefixed_inputs_nhwc = backend_prefix.to_string() + "InputsUseNHWC";
        if cfg.contains(&prefixed_inputs_nhwc_idx) {
            inputs_use_nhwc = cfg
                .get_bool(&prefixed_inputs_nhwc_idx)
                .map_err(to_string_error)?;
        } else if cfg.contains(&inputs_nhwc_idx) {
            inputs_use_nhwc = cfg.get_bool(&inputs_nhwc_idx).map_err(to_string_error)?;
        } else if cfg.contains(&prefixed_inputs_nhwc) {
            inputs_use_nhwc = cfg
                .get_bool(&prefixed_inputs_nhwc)
                .map_err(to_string_error)?;
        } else if cfg.contains("inputsUseNHWC") {
            inputs_use_nhwc = cfg.get_bool("inputsUseNHWC").map_err(to_string_error)?;
        }

        let nn_randomize = if setup_for == SetupFor::Distributed {
            true
        } else if cfg.contains("nnRandomize") {
            cfg.get_bool("nnRandomize").map_err(to_string_error)?
        } else {
            true
        };

        let nn_rand_seed = if setup_for == SetupFor::Distributed {
            seed_rand.next_u64().to_string()
        } else if cfg.contains(&("nnRandSeed".to_string() + &idx_str)) {
            cfg.get_string(&("nnRandSeed".to_string() + &idx_str))
                .map_err(to_string_error)?
        } else if cfg.contains("nnRandSeed") {
            cfg.get_string("nnRandSeed").map_err(to_string_error)?
        } else {
            seed_rand.next_u64().to_string()
        };
        logger.write(&("nnRandSeed".to_string() + &idx_str + " = " + &nn_rand_seed));

        cfg.mark_all_keys_used_with_prefix("numNNServerThreadsPerModel");
        let num_threads = if cfg.contains("numEigenThreadsPerModel") {
            cfg.get_int("numEigenThreadsPerModel", 1, 1024)
                .map_err(to_string_error)?
        } else {
            compute_default_eigen_backend_threads(expected_concurrent_evals, logger)
        };

        let mut gpu_idx_by_server_thread = Vec::new();
        for j in 0..num_threads {
            let thread_idx_str = j.to_string();
            let gpu_idx = if cfg.contains(
                &(backend_prefix.to_string()
                    + "DeviceToUseModel"
                    + &idx_str
                    + "Thread"
                    + &thread_idx_str),
            ) {
                cfg.get_int(
                    &(backend_prefix.to_string()
                        + "DeviceToUseModel"
                        + &idx_str
                        + "Thread"
                        + &thread_idx_str),
                    0,
                    1023,
                )
                .map_err(to_string_error)?
            } else if cfg.contains(
                &(backend_prefix.to_string()
                    + "GpuToUseModel"
                    + &idx_str
                    + "Thread"
                    + &thread_idx_str),
            ) {
                cfg.get_int(
                    &(backend_prefix.to_string()
                        + "GpuToUseModel"
                        + &idx_str
                        + "Thread"
                        + &thread_idx_str),
                    0,
                    1023,
                )
                .map_err(to_string_error)?
            } else if cfg
                .contains(&("deviceToUseModel".to_string() + &idx_str + "Thread" + &thread_idx_str))
            {
                cfg.get_int(
                    &("deviceToUseModel".to_string() + &idx_str + "Thread" + &thread_idx_str),
                    0,
                    1023,
                )
                .map_err(to_string_error)?
            } else if cfg
                .contains(&("gpuToUseModel".to_string() + &idx_str + "Thread" + &thread_idx_str))
            {
                cfg.get_int(
                    &("gpuToUseModel".to_string() + &idx_str + "Thread" + &thread_idx_str),
                    0,
                    1023,
                )
                .map_err(to_string_error)?
            } else if cfg.contains(&(backend_prefix.to_string() + "DeviceToUseModel" + &idx_str)) {
                cfg.get_int(
                    &(backend_prefix.to_string() + "DeviceToUseModel" + &idx_str),
                    0,
                    1023,
                )
                .map_err(to_string_error)?
            } else if cfg.contains(&(backend_prefix.to_string() + "GpuToUseModel" + &idx_str)) {
                cfg.get_int(
                    &(backend_prefix.to_string() + "GpuToUseModel" + &idx_str),
                    0,
                    1023,
                )
                .map_err(to_string_error)?
            } else if cfg.contains(&("deviceToUseModel".to_string() + &idx_str)) {
                cfg.get_int(&("deviceToUseModel".to_string() + &idx_str), 0, 1023)
                    .map_err(to_string_error)?
            } else if cfg.contains(&("gpuToUseModel".to_string() + &idx_str)) {
                cfg.get_int(&("gpuToUseModel".to_string() + &idx_str), 0, 1023)
                    .map_err(to_string_error)?
            } else if cfg
                .contains(&(backend_prefix.to_string() + "DeviceToUseThread" + &thread_idx_str))
            {
                cfg.get_int(
                    &(backend_prefix.to_string() + "DeviceToUseThread" + &thread_idx_str),
                    0,
                    1023,
                )
                .map_err(to_string_error)?
            } else if cfg
                .contains(&(backend_prefix.to_string() + "GpuToUseThread" + &thread_idx_str))
            {
                cfg.get_int(
                    &(backend_prefix.to_string() + "GpuToUseThread" + &thread_idx_str),
                    0,
                    1023,
                )
                .map_err(to_string_error)?
            } else if cfg.contains(&("deviceToUseThread".to_string() + &thread_idx_str)) {
                cfg.get_int(
                    &("deviceToUseThread".to_string() + &thread_idx_str),
                    0,
                    1023,
                )
                .map_err(to_string_error)?
            } else if cfg.contains(&("gpuToUseThread".to_string() + &thread_idx_str)) {
                cfg.get_int(&("gpuToUseThread".to_string() + &thread_idx_str), 0, 1023)
                    .map_err(to_string_error)?
            } else if cfg.contains(&(backend_prefix.to_string() + "DeviceToUse")) {
                cfg.get_int(&(backend_prefix.to_string() + "DeviceToUse"), 0, 1023)
                    .map_err(to_string_error)?
            } else if cfg.contains(&(backend_prefix.to_string() + "GpuToUse")) {
                cfg.get_int(&(backend_prefix.to_string() + "GpuToUse"), 0, 1023)
                    .map_err(to_string_error)?
            } else if cfg.contains("deviceToUse") {
                cfg.get_int("deviceToUse", 0, 1023)
                    .map_err(to_string_error)?
            } else if cfg.contains("gpuToUse") {
                cfg.get_int("gpuToUse", 0, 1023).map_err(to_string_error)?
            } else {
                -1
            };
            gpu_idx_by_server_thread.push(gpu_idx);
        }

        let home_data_dir_override = load_home_data_dir_override(cfg);

        let mut use_fp16_mode = Enabled::Auto;
        let prefixed_use_fp16_idx = backend_prefix.to_string() + "UseFP16-" + &idx_str;
        let use_fp16_idx = "useFP16-".to_string() + &idx_str;
        let prefixed_use_fp16 = backend_prefix.to_string() + "UseFP16";
        if cfg.contains(&prefixed_use_fp16_idx) {
            use_fp16_mode = enabled_from_core(
                cfg.get_enabled(&prefixed_use_fp16_idx)
                    .map_err(to_string_error)?,
            );
        } else if cfg.contains(&use_fp16_idx) {
            use_fp16_mode =
                enabled_from_core(cfg.get_enabled(&use_fp16_idx).map_err(to_string_error)?);
        } else if cfg.contains(&prefixed_use_fp16) {
            use_fp16_mode = enabled_from_core(
                cfg.get_enabled(&prefixed_use_fp16)
                    .map_err(to_string_error)?,
            );
        } else if cfg.contains("useFP16") {
            use_fp16_mode = enabled_from_core(cfg.get_enabled("useFP16").map_err(to_string_error)?);
        }

        let mut forced_symmetry = -1;
        if setup_for != SetupFor::Distributed && cfg.contains("nnForcedSymmetry") {
            forced_symmetry = cfg
                .get_int("nnForcedSymmetry", 0, NUM_SYMMETRIES - 1)
                .map_err(to_string_error)?;
        }

        logger.write(&format!(
            "After dedups: nnModelFile{} = {} useFP16 {}",
            idx_str, nn_model_file, use_fp16_mode
        ));

        let nn_cache_size_power_of_two = if cfg.contains("nnCacheSizePowerOfTwo") {
            cfg.get_int("nnCacheSizePowerOfTwo", -1, 48)
                .map_err(to_string_error)?
        } else {
            match setup_for {
                SetupFor::Gtp => 20,
                SetupFor::Benchmark => 20,
                SetupFor::Distributed => 19,
                SetupFor::Match => 21,
                SetupFor::Analysis => 23,
                SetupFor::Other => cfg
                    .get_int("nnCacheSizePowerOfTwo", -1, 48)
                    .map_err(to_string_error)?,
            }
        };

        let nn_mutex_pool_size_power_of_two = if cfg.contains("nnMutexPoolSizePowerOfTwo") {
            cfg.get_int("nnMutexPoolSizePowerOfTwo", -1, 24)
                .map_err(to_string_error)?
        } else {
            match setup_for {
                SetupFor::Gtp => 16,
                SetupFor::Benchmark => 16,
                SetupFor::Distributed => 16,
                SetupFor::Match => 17,
                SetupFor::Analysis => 17,
                SetupFor::Other => cfg
                    .get_int("nnMutexPoolSizePowerOfTwo", -1, 24)
                    .map_err(to_string_error)?,
            }
        };

        let nn_max_batch_size =
            if setup_for == SetupFor::Benchmark || setup_for == SetupFor::Distributed {
                default_max_batch_size
            } else if default_max_batch_size > 0 {
                if cfg.contains("nnMaxBatchSize") {
                    cfg.get_int("nnMaxBatchSize", 1, 65536)
                        .map_err(to_string_error)?
                } else {
                    default_max_batch_size
                }
            } else {
                cfg.get_int("nnMaxBatchSize", 1, 65536)
                    .map_err(to_string_error)?
            };

        if disable_fp16 {
            use_fp16_mode = Enabled::False;
        }
        let default_symmetry = if forced_symmetry >= 0 {
            forced_symmetry
        } else {
            0
        };
        let do_randomize = forced_symmetry < 0 && nn_randomize;

        let disable_warmup = if cfg.contains("cudaDisableWarmup") {
            cfg.get_bool("cudaDisableWarmup").map_err(to_string_error)?
        } else {
            false
        };

        let mut nn_eval = NnEvaluator::new(
            nn_model_name.clone(),
            nn_model_file.clone(),
            expected_sha256,
            Arc::new(logger.clone()),
            nn_max_batch_size,
            nn_x_len,
            nn_y_len,
            require_exact_nn_len,
            inputs_use_nhwc,
            nn_cache_size_power_of_two,
            nn_mutex_pool_size_power_of_two,
            debug_skip_neural_net,
            home_data_dir_override,
            use_fp16_mode,
            num_threads,
            gpu_idx_by_server_thread,
            nn_rand_seed,
            do_randomize,
            default_symmetry,
            disable_warmup,
            cfg,
        );

        // Wire the selected backend into the evaluator before spawning its
        // server threads (spawn only happens once compute handles exist).
        match backend_prefix.as_str() {
            "dummybackend" => {}
            "trtbackend" => {
                nn_eval.set_backend(Arc::new(kata_nn::backends::trt::TensorRtBackend));
                nn_eval.load_model().map_err(to_string_error)?;
            }
            other => {
                return Err(StringError::new(format!(
                    "Backend '{other}' is not implemented yet"
                )));
            }
        }

        nn_eval.spawn_server_threads();

        nn_evals.push(nn_eval);
    }

    Ok(nn_evals)
}

/// Load an optional override for the KataGo home data directory.
///
/// Mirrors `Setup::loadHomeDataDirOverride` in `cpp/program/setup.cpp`.
///
/// Returns the value of the `homeDataDir` config key if present, otherwise an
/// empty string.
pub fn load_home_data_dir_override(cfg: &ConfigParser) -> String {
    if cfg.contains("homeDataDir") {
        cfg.get_string("homeDataDir")
            .unwrap_or_else(|_| String::new())
    } else {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kata_core::config::ConfigParser;
    use std::io::{self, Write};
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_mutex_key_sets_contains_rules() {
        let mutex = get_mutex_key_sets();
        assert_eq!(mutex.len(), 1);
        let (a, b) = &mutex[0];
        assert!(a.contains("rules"));
        assert!(b.contains("scoringRule"));
        assert!(b.contains("koRule"));
        assert!(b.contains("multiStoneSuicideLegal"));
    }

    #[test]
    fn test_load_single_params_defaults() {
        let cfg = ConfigParser::from_str("numSearchThreads = 1", false, true).unwrap();
        let params = load_single_params(&cfg, SetupFor::Gtp).unwrap();
        assert_eq!(params.num_threads, 1);
        assert_eq!(params.max_visits, 1_i64 << 50);
        assert_eq!(params.min_playouts_per_thread, 8.0);
        assert_eq!(params.static_score_utility_factor, 0.1);
        assert_eq!(params.dynamic_score_utility_factor, 0.3);
        assert!(params.use_lcb_for_selection);
        assert_eq!(params.lcb_stdevs, 5.0);
        assert!(params.root_prune_useless_moves);
        assert_eq!(params.wide_root_noise, 0.0);
        assert!(!params.ignore_pre_root_history);
    }

    #[test]
    fn test_load_single_params_analysis_defaults() {
        let cfg = ConfigParser::from_str("numSearchThreads = 1", false, true).unwrap();
        let params = load_single_params(&cfg, SetupFor::Analysis).unwrap();
        assert_eq!(params.min_playouts_per_thread, 8.0);
        assert_eq!(params.wide_root_noise, DEFAULT_ANALYSIS_WIDE_ROOT_NOISE);
        assert_eq!(
            params.ignore_pre_root_history,
            DEFAULT_ANALYSIS_IGNORE_PRE_ROOT_HISTORY
        );
        assert!(params.root_symmetry_pruning);
    }

    #[test]
    fn test_load_single_params_overrides() {
        let cfg = ConfigParser::from_str(
            "numSearchThreads = 4\nmaxVisits = 100\nwinLossUtilityFactor = 0.5",
            false,
            true,
        )
        .unwrap();
        let params = load_single_params(&cfg, SetupFor::Gtp).unwrap();
        assert_eq!(params.num_threads, 4);
        assert_eq!(params.max_visits, 100);
        assert_eq!(params.win_loss_utility_factor, 0.5);
    }

    #[test]
    fn test_load_params_multiple_bots() {
        let cfg = ConfigParser::from_str(
            "numBots = 2\nnumSearchThreads = 2\nnumSearchThreads1 = 4",
            false,
            true,
        )
        .unwrap();
        let params_vec = load_params(&cfg, SetupFor::Gtp).unwrap();
        assert_eq!(params_vec.len(), 2);
        assert_eq!(params_vec[0].num_threads, 2);
        assert_eq!(params_vec[1].num_threads, 4);
    }

    #[test]
    fn test_load_single_params_rejects_multiple_bots() {
        let cfg = ConfigParser::from_str("numBots = 2\nnumSearchThreads = 1", false, true).unwrap();
        let err = load_single_params(&cfg, SetupFor::Gtp).unwrap_err();
        assert!(err.message.contains("cannot have numBots > 0"));
    }

    #[test]
    fn test_human_sl_params_require_human_model() {
        let cfg = ConfigParser::from_str(
            "numSearchThreads = 1\nhumanSLCpuctExploration = 2.0",
            false,
            true,
        )
        .unwrap();
        let err = load_single_params(&cfg, SetupFor::Gtp).unwrap_err();
        assert!(err.message.contains("no human model was specified"));
    }

    #[test]
    fn test_human_sl_params_allowed_with_human_model() {
        let cfg = ConfigParser::from_str(
            "numSearchThreads = 1\nhumanSLCpuctExploration = 2.0\nhumanSLProfile = proyear_2020",
            false,
            true,
        )
        .unwrap();
        let params = load_single_params_with_human(&cfg, SetupFor::Gtp, true).unwrap();
        assert_eq!(params.human_sl_cpuct_exploration, 2.0);
        assert!(params.human_sl_profile.initialized);
    }

    #[test]
    fn test_load_params_distributed_defaults() {
        let cfg = ConfigParser::from_str("numSearchThreads = 1", false, true).unwrap();
        let params = load_single_params(&cfg, SetupFor::Distributed).unwrap();
        assert_eq!(params.cpuct_utility_stdev_scale, 0.0);
        assert!(!params.use_noise_pruning);
        assert!(!params.use_uncertainty);
        assert!(!params.use_graph_search);
        assert!(!params.use_non_buggy_lcb);
        assert_eq!(params.policy_optimism, 0.0);
        assert_eq!(params.root_policy_optimism, 0.0);
    }

    #[test]
    fn test_parse_player_error() {
        let err = parse_player("testField", "red").unwrap_err();
        assert!(err.message.contains("testField"));
    }

    #[test]
    fn test_get_backend_prefixes() {
        assert_eq!(
            get_backend_prefixes(),
            vec![
                "cuda".to_string(),
                "trt".to_string(),
                "metal".to_string(),
                "opencl".to_string(),
                "eigen".to_string(),
                "dummybackend".to_string(),
            ]
        );
    }

    #[test]
    fn test_compute_default_eigen_backend_threads_bounds() {
        let writer = TestWriter::new();
        let logger = Logger::new(
            kata_core::logger::LoggerOptions {
                log_to_stdout: false,
                log_to_stderr: false,
                log_time: false,
            },
            None,
        );
        logger.add_ostream(writer.clone(), false);

        let cores = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(8);
        let result = compute_default_eigen_backend_threads(1_000_000, &logger);
        assert!(result > 0);
        assert!(result <= 1_000_000);
        assert_eq!(result, cores.min(1_000_000));

        if std::thread::available_parallelism().is_err() {
            assert!(writer.get().contains("WARNING"));
        }
    }

    #[test]
    fn test_compute_default_eigen_backend_threads_respects_expected() {
        let logger = Logger::new(
            kata_core::logger::LoggerOptions {
                log_to_stdout: false,
                log_to_stderr: false,
                log_time: false,
            },
            None,
        );
        let result = compute_default_eigen_backend_threads(2, &logger);
        assert!(result > 0);
        assert!(result <= 2);
    }

    #[test]
    fn test_load_home_data_dir_override_present() {
        let cfg = ConfigParser::from_str("homeDataDir = /path/to/home", false, true).unwrap();
        assert_eq!(load_home_data_dir_override(&cfg), "/path/to/home");
    }

    #[test]
    fn test_load_home_data_dir_override_missing() {
        let cfg = ConfigParser::from_str("numSearchThreads = 1", false, true).unwrap();
        assert_eq!(load_home_data_dir_override(&cfg), "");
    }

    #[test]
    fn test_initialize_nn_evaluator_dummy() {
        let cfg = ConfigParser::from_str("numSearchThreads = 1", false, true).unwrap();
        let logger = Logger::new(
            kata_core::logger::LoggerOptions {
                log_to_stdout: false,
                log_to_stderr: false,
                log_time: false,
            },
            None,
        );
        let mut seed_rand = Rand::new();

        let nn_eval = initialize_nn_evaluator(
            "dummy-model".to_string(),
            "/dev/null".to_string(),
            String::new(),
            &cfg,
            &logger,
            &mut seed_rand,
            1,
            19,
            19,
            8,
            false,
            false,
            SetupFor::Gtp,
        )
        .unwrap();

        assert_eq!(nn_eval.nn_x_len(), 19);
        assert_eq!(nn_eval.nn_y_len(), 19);
    }

    #[test]
    fn test_load_single_rules_from_string() {
        let cfg = ConfigParser::from_str("rules = chinese", false, true).unwrap();
        let rules = load_single_rules(&cfg, false).unwrap();
        assert_eq!(rules.scoring_rule, ScoringRule::Area);
        assert_eq!(rules.ko_rule, KoRule::Simple);
        assert!(!rules.has_button);
    }

    #[test]
    fn test_load_single_rules_individual_keys() {
        let cfg = ConfigParser::from_str(
            "koRule = POSITIONAL\nscoringRule = TERRITORY\nmultiStoneSuicideLegal = false",
            false,
            true,
        )
        .unwrap();
        let rules = load_single_rules(&cfg, false).unwrap();
        assert_eq!(rules.ko_rule, KoRule::Positional);
        assert_eq!(rules.scoring_rule, ScoringRule::Territory);
        assert!(!rules.multi_stone_suicide_legal);
        assert_eq!(rules.tax_rule, TaxRule::Seki);
        assert_eq!(rules.komi_f32(), 6.5);
    }

    #[test]
    fn test_load_single_rules_button() {
        let cfg = ConfigParser::from_str(
            "koRule = POSITIONAL\nscoringRule = AREA\nmultiStoneSuicideLegal = true\nhasButton = true",
            false,
            true,
        )
        .unwrap();
        let rules = load_single_rules(&cfg, false).unwrap();
        assert!(rules.has_button);
        assert_eq!(rules.komi_f32(), 7.0);
    }

    #[test]
    fn test_load_single_rules_rejects_mixed() {
        let cfg =
            ConfigParser::from_str("rules = chinese\nkoRule = POSITIONAL", false, true).unwrap();
        let err = load_single_rules(&cfg, false).unwrap_err();
        assert!(err.message.contains("Cannot both specify 'rules'"));
    }

    #[test]
    fn test_load_single_rules_loads_komi() {
        let cfg = ConfigParser::from_str("rules = chinese\nkomi = 6.5", false, true).unwrap();
        let rules = load_single_rules(&cfg, true).unwrap();
        assert_eq!(rules.komi_f32(), 6.5);
    }

    #[test]
    fn test_load_default_board_xy_size() {
        let cfg = ConfigParser::from_str("defaultBoardSize = 13", false, true).unwrap();
        let logger = Logger::new(
            kata_core::logger::LoggerOptions {
                log_to_stdout: false,
                log_to_stderr: false,
                log_time: false,
            },
            None,
        );
        let mut x = 0;
        let mut y = 0;
        assert!(load_default_board_xy_size(&cfg, &logger, &mut x, &mut y).unwrap());
        assert_eq!(x, 13);
        assert_eq!(y, 13);
    }

    #[test]
    fn test_load_default_board_xy_size_separate() {
        let cfg =
            ConfigParser::from_str("defaultBoardXSize = 9\ndefaultBoardYSize = 13", false, true)
                .unwrap();
        let logger = Logger::new(
            kata_core::logger::LoggerOptions {
                log_to_stdout: false,
                log_to_stderr: false,
                log_time: false,
            },
            None,
        );
        let mut x = 0;
        let mut y = 0;
        assert!(load_default_board_xy_size(&cfg, &logger, &mut x, &mut y).unwrap());
        assert_eq!(x, 9);
        assert_eq!(y, 13);
    }

    #[test]
    fn test_load_default_board_xy_size_missing() {
        let cfg = ConfigParser::from_str("numSearchThreads = 1", false, true).unwrap();
        let logger = Logger::new(
            kata_core::logger::LoggerOptions {
                log_to_stdout: false,
                log_to_stderr: false,
                log_time: false,
            },
            None,
        );
        let mut x = 0;
        let mut y = 0;
        assert!(!load_default_board_xy_size(&cfg, &logger, &mut x, &mut y).unwrap());
        assert_eq!(x, 0);
        assert_eq!(y, 0);
    }

    #[test]
    fn test_load_default_board_xy_size_partial_warns() {
        let cfg = ConfigParser::from_str("defaultBoardXSize = 9", false, true).unwrap();
        let writer = TestWriter::new();
        let logger = Logger::new(
            kata_core::logger::LoggerOptions {
                log_to_stdout: false,
                log_to_stderr: false,
                log_time: false,
            },
            None,
        );
        logger.add_ostream(writer.clone(), false);
        let mut x = 0;
        let mut y = 0;
        assert!(!load_default_board_xy_size(&cfg, &logger, &mut x, &mut y).unwrap());
        assert!(writer.get().contains("Warning"));
    }

    #[test]
    fn test_save_and_load_auto_pattern_bonus_data() {
        use kata_game::board::{Board, P_BLACK, location};

        let temp_dir = std::env::temp_dir().join(format!(
            "katago_auto_pattern_test_{}",
            global::uint64_to_hex_string(Rand::new().next_u64())
        ));
        let _ = std::fs::remove_dir_all(&temp_dir);

        let cfg = ConfigParser::from_str(
            &format!(
                "autoAvoidRepeatDir = {}\n\
                 autoAvoidRepeatUtility = -0.5\n\
                 autoAvoidRepeatLambda = 1.0\n\
                 autoAvoidRepeatMinTurnNumber = 0\n\
                 autoAvoidRepeatMaxTurnNumber = 100\n\
                 autoAvoidRepeatMaxPoses = 1000",
                temp_dir.to_str().unwrap().replace('\\', "/")
            ),
            false,
            true,
        )
        .unwrap();
        let logger = Logger::new(
            kata_core::logger::LoggerOptions {
                log_to_stdout: false,
                log_to_stderr: false,
                log_time: false,
            },
            None,
        );
        let mut rand = Rand::new();

        let mut sample = PositionSample::default();
        sample.board = Board::new(9, 9);
        sample.next_pla = P_BLACK;
        sample.hint_loc = location::get_loc(4, 4, sample.board.x_size);
        sample.initial_turn_number = 10;
        sample.moves = Vec::new();

        assert!(save_auto_pattern_bonus_data(&[sample.clone()], &cfg, &logger, &mut rand).unwrap());

        let table = load_and_prune_auto_pattern_bonus_tables(&cfg, &logger)
            .unwrap()
            .expect("table should be loaded");
        let entry = table.get_for_move(P_BLACK, sample.hint_loc, &sample.board);
        assert!(entry.utility_bonus.abs() > 1e-9);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_load_avoid_sgf_pattern_bonus_tables() {
        use kata_game::board::{Board, P_BLACK, location};

        let sgf_dir = std::env::temp_dir().join(format!(
            "katago_sgf_pattern_test_{}",
            global::uint64_to_hex_string(Rand::new().next_u64())
        ));
        let _ = std::fs::remove_dir_all(&sgf_dir);
        std::fs::create_dir_all(&sgf_dir).unwrap();

        // Minimal SGF: Black plays the center.
        let sgf_content = "(;GM[1]FF[4]SZ[9]PB[Black]PW[White];B[ee];W[ef])";
        let sgf_path = sgf_dir.join("game.sgf");
        std::fs::write(&sgf_path, sgf_content).unwrap();

        let cfg = ConfigParser::from_str(
            &format!(
                "avoidSgfPatternUtility = -0.5\navoidSgfPatternDirs = {}",
                sgf_dir.to_str().unwrap().replace('\\', "/")
            ),
            false,
            true,
        )
        .unwrap();
        let logger = Logger::new(
            kata_core::logger::LoggerOptions {
                log_to_stdout: false,
                log_to_stderr: false,
                log_time: false,
            },
            None,
        );

        let tables = load_avoid_sgf_pattern_bonus_tables(&cfg, &logger).unwrap();
        assert_eq!(tables.len(), 1);
        let table = tables[0].as_ref().expect("table should be loaded");

        let board = Board::new(9, 9);
        let loc = location::get_loc(4, 4, board.x_size);
        let entry = table.get_for_move(P_BLACK, loc, &board);
        assert!(entry.utility_bonus.abs() > 1e-9);

        let _ = std::fs::remove_dir_all(&sgf_dir);
    }

    #[test]
    fn test_maybe_warn_human_sl_params_warns() {
        let cfg = ConfigParser::from_str(
            "numSearchThreads = 1\nhumanSLProfile = proyear_2020",
            false,
            true,
        )
        .unwrap();
        let params = load_single_params(&cfg, SetupFor::Gtp).unwrap();
        assert!(params.human_sl_profile.initialized);

        let logger = Logger::new(
            kata_core::logger::LoggerOptions {
                log_to_stdout: false,
                log_to_stderr: false,
                log_time: false,
            },
            None,
        );
        let mut out = Vec::new();
        let warned = maybe_warn_human_sl_params(
            &params,
            None::<&NnEvaluator>,
            None::<&NnEvaluator>,
            &mut out,
            Some(&logger),
        )
        .unwrap();
        assert!(warned);
        let output = String::from_utf8(out).unwrap();
        assert!(output.contains("humanSLProfile"));
        assert!(output.contains("don't use it"));
    }

    #[test]
    fn test_maybe_warn_human_sl_params_no_profile() {
        let cfg = ConfigParser::from_str("numSearchThreads = 1", false, true).unwrap();
        let params = load_single_params(&cfg, SetupFor::Gtp).unwrap();
        assert!(!params.human_sl_profile.initialized);

        let mut out = Vec::new();
        let warned = maybe_warn_human_sl_params(
            &params,
            None::<&NnEvaluator>,
            None::<&NnEvaluator>,
            &mut out,
            None,
        )
        .unwrap();
        assert!(!warned);
        assert!(out.is_empty());
    }

    #[test]
    fn test_parse_report_analysis_winrates() {
        use kata_game::board::P_BLACK;

        let cfg = ConfigParser::from_str("reportAnalysisWinratesAs = white", false, true).unwrap();
        assert_eq!(
            parse_report_analysis_winrates(&cfg, C_EMPTY).unwrap(),
            kata_game::board::P_WHITE
        );

        let cfg =
            ConfigParser::from_str("reportAnalysisWinratesAs = SIDETOMOVE", false, true).unwrap();
        assert_eq!(
            parse_report_analysis_winrates(&cfg, P_BLACK).unwrap(),
            C_EMPTY
        );

        let cfg = ConfigParser::from_str("numSearchThreads = 1", false, true).unwrap();
        assert_eq!(
            parse_report_analysis_winrates(&cfg, P_BLACK).unwrap(),
            P_BLACK
        );

        let cfg = ConfigParser::from_str("reportAnalysisWinratesAs = red", false, true).unwrap();
        assert!(parse_report_analysis_winrates(&cfg, C_EMPTY).is_err());
    }

    /// A test writer that records bytes to a shared buffer.
    #[derive(Clone)]
    struct TestWriter(Arc<Mutex<Vec<u8>>>);

    impl TestWriter {
        fn new() -> Self {
            Self(Arc::new(Mutex::new(Vec::new())))
        }

        fn get(&self) -> String {
            let bytes = self.0.lock().unwrap().clone();
            String::from_utf8(bytes).unwrap()
        }
    }

    impl Write for TestWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
}
