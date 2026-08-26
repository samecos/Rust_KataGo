//! Search parameters.
//!
//! Corresponds to `cpp/search/searchparams.h` and `cpp/search/searchparams.cpp`.

use kata_core::global::StringError;
use kata_core::hash::{Hash128, sha2};
use kata_game::board::{C_EMPTY, P_BLACK, Player, player_io};
use kata_nn::sgf_meta::SgfMetadata;
use serde_json::{Map, Value, json};
use std::fmt;

/// Full set of parameters controlling MCTS search behavior.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq)]
pub struct SearchParams {
    // Utility function parameters
    pub win_loss_utility_factor: f64,
    pub static_score_utility_factor: f64,
    pub dynamic_score_utility_factor: f64,
    pub dynamic_score_center_zero_weight: f64,
    pub dynamic_score_center_scale: f64,
    pub no_result_utility_for_white: f64,
    pub draw_equivalent_wins_for_white: f64,

    // Search tree exploration parameters
    pub cpuct_exploration: f64,
    pub cpuct_exploration_log: f64,
    pub cpuct_exploration_base: f64,

    pub cpuct_utility_stdev_prior: f64,
    pub cpuct_utility_stdev_prior_weight: f64,
    pub cpuct_utility_stdev_scale: f64,

    pub fpu_reduction_max: f64,
    pub fpu_loss_prop: f64,

    pub fpu_parent_weight_by_visited_policy: bool,
    pub fpu_parent_weight_by_visited_policy_pow: f64,
    pub fpu_parent_weight: f64,

    pub policy_optimism: f64,

    // Tree value aggregation parameters
    pub value_weight_exponent: f64,
    pub use_noise_pruning: bool,
    pub noise_prune_utility_scale: f64,
    pub noise_pruning_cap: f64,

    // Uncertainty weighting
    pub use_uncertainty: bool,
    pub uncertainty_coeff: f64,
    pub uncertainty_exponent: f64,
    pub uncertainty_max_weight: f64,

    // Graph search
    pub use_graph_search: bool,
    pub graph_search_rep_bound: i32,
    pub graph_search_catch_up_leak_prob: f64,

    // Root parameters
    pub root_noise_enabled: bool,
    pub root_dirichlet_noise_total_concentration: f64,
    pub root_dirichlet_noise_weight: f64,

    pub root_policy_temperature: f64,
    pub root_policy_temperature_early: f64,
    pub root_fpu_reduction_max: f64,
    pub root_fpu_loss_prop: f64,
    pub root_num_symmetries_to_sample: i32,
    pub root_symmetry_pruning: bool,
    pub root_desired_per_child_visits_coeff: f64,

    pub root_policy_optimism: f64,

    // Parameters for choosing the move to play
    pub chosen_move_temperature: f64,
    pub chosen_move_temperature_early: f64,
    pub chosen_move_temperature_halflife: f64,
    pub chosen_move_temperature_only_below_prob: f64,
    pub chosen_move_subtract: f64,
    pub chosen_move_prune: f64,

    pub use_lcb_for_selection: bool,
    pub lcb_stdevs: f64,
    pub min_visit_prop_for_lcb: f64,
    pub use_non_buggy_lcb: bool,

    // Mild behavior hackery
    pub root_ending_bonus_points: f64,
    pub root_prune_useless_moves: bool,
    pub conservative_pass: bool,
    pub fill_dame_before_pass: bool,
    pub avoid_mytd_dagger_hack_pla: Player,
    pub wide_root_noise: f64,
    /// PUCT-V style child-level variance-aware exploration scale
    /// (0.0 = off, bit-identical to baseline). See Weichart 2025,
    /// arXiv:2512.21648: exploration bonus scaled by the child's empirical
    /// utility stdev, normalized against `cpuct_utility_stdev_prior`.
    pub puct_var_exploration: f64,
    pub enable_passing_hacks: bool,
    pub enable_more_passing_hacks: bool,

    pub playout_doubling_advantage: f64,
    pub playout_doubling_advantage_pla: Player,

    pub avoid_repeated_pattern_utility: f64,

    pub nn_policy_temperature: f32,
    pub anti_mirror: bool,

    pub ignore_pre_root_history: bool,
    pub ignore_all_history: bool,

    pub subtree_value_bias_factor: f64,
    pub subtree_value_bias_table_num_shards: i32,
    pub subtree_value_bias_free_prop: f64,
    pub subtree_value_bias_weight_exponent: f64,

    pub use_eval_cache: bool,
    pub eval_cache_min_visits: i64,

    // Threading-related
    pub node_table_shards_power_of_two: i32,
    pub num_virtual_losses_per_thread: f64,

    // Asyncbot
    pub num_threads: i32,
    pub min_playouts_per_thread: f64,
    pub max_visits: i64,
    pub max_playouts: i64,
    pub max_time: f64,

    pub max_visits_pondering: i64,
    pub max_playouts_pondering: i64,
    pub max_time_pondering: f64,

    pub lag_buffer: f64,

    // Human-friendliness
    pub search_factor_after_one_pass: f64,
    pub search_factor_after_two_pass: f64,

    // Time control
    pub tree_reuse_carry_over_time_factor: f64,
    pub overallocate_time_factor: f64,
    pub midgame_time_factor: f64,
    pub midgame_turn_peak_time: f64,
    pub endgame_turn_time_decay: f64,
    pub obvious_moves_time_factor: f64,
    pub obvious_moves_policy_entropy_tolerance: f64,
    pub obvious_moves_policy_surprise_tolerance: f64,

    pub futile_visits_threshold: f64,

    // Human SL network
    pub human_sl_profile: SgfMetadata,
    pub human_sl_cpuct_exploration: f64,
    pub human_sl_cpuct_permanent: f64,
    pub human_sl_root_explore_prob_weightless: f64,
    pub human_sl_root_explore_prob_weightful: f64,
    pub human_sl_pla_explore_prob_weightless: f64,
    pub human_sl_pla_explore_prob_weightful: f64,
    pub human_sl_opp_explore_prob_weightless: f64,
    pub human_sl_opp_explore_prob_weightful: f64,

    pub human_sl_chosen_move_prop: f64,
    pub human_sl_chosen_move_ignore_pass: bool,
    pub human_sl_chosen_move_pikl_lambda: f64,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchParams {
    /// Default deterministic search parameters with no noise and no limits.
    pub fn new() -> Self {
        Self {
            win_loss_utility_factor: 1.0,
            static_score_utility_factor: 0.3,
            dynamic_score_utility_factor: 0.0,
            dynamic_score_center_zero_weight: 0.0,
            dynamic_score_center_scale: 1.0,
            no_result_utility_for_white: 0.0,
            draw_equivalent_wins_for_white: 0.5,

            cpuct_exploration: 1.0,
            cpuct_exploration_log: 0.0,
            cpuct_exploration_base: 500.0,
            cpuct_utility_stdev_prior: 0.25,
            cpuct_utility_stdev_prior_weight: 1.0,
            cpuct_utility_stdev_scale: 0.0,

            fpu_reduction_max: 0.2,
            fpu_loss_prop: 0.0,
            fpu_parent_weight_by_visited_policy: false,
            fpu_parent_weight_by_visited_policy_pow: 1.0,
            fpu_parent_weight: 0.0,

            policy_optimism: 0.0,

            value_weight_exponent: 0.5,
            use_noise_pruning: false,
            noise_prune_utility_scale: 0.15,
            noise_pruning_cap: 1e50,

            use_uncertainty: false,
            uncertainty_coeff: 0.2,
            uncertainty_exponent: 1.0,
            uncertainty_max_weight: 8.0,

            use_graph_search: false,
            graph_search_rep_bound: 11,
            graph_search_catch_up_leak_prob: 0.0,

            root_noise_enabled: false,
            root_dirichlet_noise_total_concentration: 10.83,
            root_dirichlet_noise_weight: 0.25,

            root_policy_temperature: 1.0,
            root_policy_temperature_early: 1.0,
            root_fpu_reduction_max: 0.2,
            root_fpu_loss_prop: 0.0,
            root_num_symmetries_to_sample: 1,
            root_symmetry_pruning: false,
            root_desired_per_child_visits_coeff: 0.0,

            root_policy_optimism: 0.0,

            chosen_move_temperature: 0.0,
            chosen_move_temperature_early: 0.0,
            chosen_move_temperature_halflife: 19.0,
            chosen_move_temperature_only_below_prob: 1.0,
            chosen_move_subtract: 0.0,
            chosen_move_prune: 1.0,

            use_lcb_for_selection: false,
            lcb_stdevs: 4.0,
            min_visit_prop_for_lcb: 0.05,
            use_non_buggy_lcb: false,

            root_ending_bonus_points: 0.0,
            root_prune_useless_moves: false,
            conservative_pass: false,
            fill_dame_before_pass: false,
            avoid_mytd_dagger_hack_pla: C_EMPTY,
            wide_root_noise: 0.0,
            puct_var_exploration: 0.0,
            enable_passing_hacks: false,
            enable_more_passing_hacks: false,

            playout_doubling_advantage: 0.0,
            playout_doubling_advantage_pla: C_EMPTY,

            avoid_repeated_pattern_utility: 0.0,

            nn_policy_temperature: 1.0,
            anti_mirror: false,

            ignore_pre_root_history: false,
            ignore_all_history: false,

            subtree_value_bias_factor: 0.0,
            subtree_value_bias_table_num_shards: 65536,
            subtree_value_bias_free_prop: 0.8,
            subtree_value_bias_weight_exponent: 0.5,

            use_eval_cache: false,
            eval_cache_min_visits: 100,

            node_table_shards_power_of_two: 16,
            num_virtual_losses_per_thread: 3.0,

            num_threads: 1,
            min_playouts_per_thread: 0.0,
            max_visits: 1_i64 << 50,
            max_playouts: 1_i64 << 50,
            max_time: 1.0e20,

            max_visits_pondering: 1_i64 << 50,
            max_playouts_pondering: 1_i64 << 50,
            max_time_pondering: 1.0e20,

            lag_buffer: 0.0,

            search_factor_after_one_pass: 1.0,
            search_factor_after_two_pass: 1.0,

            tree_reuse_carry_over_time_factor: 0.0,
            overallocate_time_factor: 1.0,
            midgame_time_factor: 1.0,
            midgame_turn_peak_time: 130.0,
            endgame_turn_time_decay: 100.0,
            obvious_moves_time_factor: 1.0,
            obvious_moves_policy_entropy_tolerance: 0.30,
            obvious_moves_policy_surprise_tolerance: 0.15,

            futile_visits_threshold: 0.0,

            human_sl_profile: SgfMetadata::default(),
            human_sl_cpuct_exploration: 1.0,
            human_sl_cpuct_permanent: 0.0,
            human_sl_root_explore_prob_weightless: 0.0,
            human_sl_root_explore_prob_weightful: 0.0,
            human_sl_pla_explore_prob_weightless: 0.0,
            human_sl_pla_explore_prob_weightful: 0.0,
            human_sl_opp_explore_prob_weightless: 0.0,
            human_sl_opp_explore_prob_weightful: 0.0,

            human_sl_chosen_move_prop: 0.0,
            human_sl_chosen_move_ignore_pass: false,
            human_sl_chosen_move_pikl_lambda: 1_000_000_000.0,
        }
    }

    /// Representative test parameters (as of Jan 2019).
    pub fn for_tests_v1() -> Self {
        let mut params = Self::new();
        params.static_score_utility_factor = 0.1;
        params.dynamic_score_utility_factor = 0.3;
        params.dynamic_score_center_zero_weight = 0.2;
        params.dynamic_score_center_scale = 0.75;
        params.cpuct_exploration = 0.9;
        params.cpuct_exploration_log = 0.4;
        params.root_fpu_reduction_max = 0.1;
        params.root_policy_temperature_early = 1.2;
        params.root_policy_temperature = 1.1;
        params.use_lcb_for_selection = true;
        params.lcb_stdevs = 5.0;
        params.min_visit_prop_for_lcb = 0.15;
        params.root_ending_bonus_points = 0.5;
        params.root_prune_useless_moves = true;
        params.conservative_pass = true;
        params.use_non_buggy_lcb = true;
        params
    }

    /// Representative test parameters (as of Mar 2022).
    pub fn for_tests_v2() -> Self {
        let mut params = Self::for_tests_v1();
        params.use_graph_search = true;
        params.fpu_parent_weight_by_visited_policy = true;
        params.value_weight_exponent = 0.25;
        params.use_noise_pruning = true;
        params.use_uncertainty = true;
        params.uncertainty_coeff = 0.25;
        params.uncertainty_exponent = 1.0;
        params.uncertainty_max_weight = 8.0;
        params.cpuct_utility_stdev_prior = 0.40;
        params.cpuct_utility_stdev_prior_weight = 2.0;
        params.cpuct_utility_stdev_scale = 0.85;
        params.fill_dame_before_pass = true;
        params.subtree_value_bias_factor = 0.45;
        params.subtree_value_bias_free_prop = 0.8;
        params.subtree_value_bias_weight_exponent = 0.85;
        params
    }

    /// A reasonable set of parameters for actual play.
    pub fn basic_decent_params() -> Self {
        let mut params = Self::new();
        params.static_score_utility_factor = 0.1;
        params.dynamic_score_utility_factor = 0.3;
        params.dynamic_score_center_zero_weight = 0.2;
        params.dynamic_score_center_scale = 0.75;
        params.cpuct_exploration = 1.0;
        params.cpuct_exploration_log = 0.45;
        params.root_fpu_reduction_max = 0.1;
        params.root_policy_temperature_early = 1.0;
        params.root_policy_temperature = 1.0;
        params.use_lcb_for_selection = true;
        params.lcb_stdevs = 5.0;
        params.min_visit_prop_for_lcb = 0.20;
        params.root_ending_bonus_points = 0.5;
        params.root_prune_useless_moves = true;
        params.conservative_pass = true;
        params.enable_passing_hacks = true;
        params.use_non_buggy_lcb = true;
        params.use_graph_search = true;
        params.fpu_parent_weight_by_visited_policy = true;
        params.value_weight_exponent = 0.25;
        params.use_noise_pruning = true;
        params.use_uncertainty = true;
        params.uncertainty_coeff = 0.25;
        params.uncertainty_exponent = 1.0;
        params.uncertainty_max_weight = 8.0;
        params.cpuct_utility_stdev_prior = 0.40;
        params.cpuct_utility_stdev_prior_weight = 2.0;
        params.cpuct_utility_stdev_scale = 0.85;
        params.fill_dame_before_pass = true;
        params.subtree_value_bias_factor = 0.45;
        params.subtree_value_bias_free_prop = 0.8;
        params.subtree_value_bias_weight_exponent = 0.85;
        params
    }

    /// Return a JSON object with the parameters that can be changed at runtime.
    pub fn changeable_parameters_to_json(&self) -> Value {
        let mut ret = Map::new();
        ret.insert(
            "winLossUtilityFactor".to_string(),
            json!(self.win_loss_utility_factor),
        );
        ret.insert(
            "staticScoreUtilityFactor".to_string(),
            json!(self.static_score_utility_factor),
        );
        ret.insert(
            "dynamicScoreUtilityFactor".to_string(),
            json!(self.dynamic_score_utility_factor),
        );
        ret.insert(
            "dynamicScoreCenterZeroWeight".to_string(),
            json!(self.dynamic_score_center_zero_weight),
        );
        ret.insert(
            "dynamicScoreCenterScale".to_string(),
            json!(self.dynamic_score_center_scale),
        );
        ret.insert(
            "noResultUtilityForWhite".to_string(),
            json!(self.no_result_utility_for_white),
        );
        ret.insert(
            "drawEquivalentWinsForWhite".to_string(),
            json!(self.draw_equivalent_wins_for_white),
        );

        ret.insert(
            "cpuctExploration".to_string(),
            json!(self.cpuct_exploration),
        );
        ret.insert(
            "cpuctExplorationLog".to_string(),
            json!(self.cpuct_exploration_log),
        );
        ret.insert(
            "cpuctExplorationBase".to_string(),
            json!(self.cpuct_exploration_base),
        );

        ret.insert(
            "cpuctUtilityStdevPrior".to_string(),
            json!(self.cpuct_utility_stdev_prior),
        );
        ret.insert(
            "cpuctUtilityStdevPriorWeight".to_string(),
            json!(self.cpuct_utility_stdev_prior_weight),
        );
        ret.insert(
            "cpuctUtilityStdevScale".to_string(),
            json!(self.cpuct_utility_stdev_scale),
        );

        ret.insert("fpuReductionMax".to_string(), json!(self.fpu_reduction_max));
        ret.insert("fpuLossProp".to_string(), json!(self.fpu_loss_prop));

        ret.insert(
            "fpuParentWeightByVisitedPolicy".to_string(),
            json!(self.fpu_parent_weight_by_visited_policy),
        );
        ret.insert(
            "fpuParentWeightByVisitedPolicyPow".to_string(),
            json!(self.fpu_parent_weight_by_visited_policy_pow),
        );
        ret.insert("fpuParentWeight".to_string(), json!(self.fpu_parent_weight));

        ret.insert("policyOptimism".to_string(), json!(self.policy_optimism));

        ret.insert(
            "valueWeightExponent".to_string(),
            json!(self.value_weight_exponent),
        );
        ret.insert("useNoisePruning".to_string(), json!(self.use_noise_pruning));
        ret.insert(
            "noisePruneUtilityScale".to_string(),
            json!(self.noise_prune_utility_scale),
        );
        ret.insert("noisePruningCap".to_string(), json!(self.noise_pruning_cap));

        ret.insert("useUncertainty".to_string(), json!(self.use_uncertainty));
        ret.insert(
            "uncertaintyCoeff".to_string(),
            json!(self.uncertainty_coeff),
        );
        ret.insert(
            "uncertaintyExponent".to_string(),
            json!(self.uncertainty_exponent),
        );
        ret.insert(
            "uncertaintyMaxWeight".to_string(),
            json!(self.uncertainty_max_weight),
        );

        ret.insert("useGraphSearch".to_string(), json!(self.use_graph_search));
        ret.insert(
            "graphSearchRepBound".to_string(),
            json!(self.graph_search_rep_bound),
        );
        ret.insert(
            "graphSearchCatchUpLeakProb".to_string(),
            json!(self.graph_search_catch_up_leak_prob),
        );

        ret.insert(
            "rootNoiseEnabled".to_string(),
            json!(self.root_noise_enabled),
        );
        ret.insert(
            "rootDirichletNoiseTotalConcentration".to_string(),
            json!(self.root_dirichlet_noise_total_concentration),
        );
        ret.insert(
            "rootDirichletNoiseWeight".to_string(),
            json!(self.root_dirichlet_noise_weight),
        );

        ret.insert(
            "rootPolicyTemperature".to_string(),
            json!(self.root_policy_temperature),
        );
        ret.insert(
            "rootPolicyTemperatureEarly".to_string(),
            json!(self.root_policy_temperature_early),
        );
        ret.insert(
            "rootFpuReductionMax".to_string(),
            json!(self.root_fpu_reduction_max),
        );
        ret.insert(
            "rootFpuLossProp".to_string(),
            json!(self.root_fpu_loss_prop),
        );
        ret.insert(
            "rootNumSymmetriesToSample".to_string(),
            json!(self.root_num_symmetries_to_sample),
        );
        ret.insert(
            "rootSymmetryPruning".to_string(),
            json!(self.root_symmetry_pruning),
        );
        ret.insert(
            "rootDesiredPerChildVisitsCoeff".to_string(),
            json!(self.root_desired_per_child_visits_coeff),
        );

        ret.insert(
            "rootPolicyOptimism".to_string(),
            json!(self.root_policy_optimism),
        );

        ret.insert(
            "chosenMoveTemperature".to_string(),
            json!(self.chosen_move_temperature),
        );
        ret.insert(
            "chosenMoveTemperatureEarly".to_string(),
            json!(self.chosen_move_temperature_early),
        );
        ret.insert(
            "chosenMoveTemperatureHalflife".to_string(),
            json!(self.chosen_move_temperature_halflife),
        );
        ret.insert(
            "chosenMoveTemperatureOnlyBelowProb".to_string(),
            json!(self.chosen_move_temperature_only_below_prob),
        );

        ret.insert(
            "chosenMoveSubtract".to_string(),
            json!(self.chosen_move_subtract),
        );
        ret.insert("chosenMovePrune".to_string(), json!(self.chosen_move_prune));
        ret.insert(
            "useLcbForSelection".to_string(),
            json!(self.use_lcb_for_selection),
        );

        ret.insert("lcbStdevs".to_string(), json!(self.lcb_stdevs));
        ret.insert(
            "minVisitPropForLCB".to_string(),
            json!(self.min_visit_prop_for_lcb),
        );
        ret.insert("useNonBuggyLcb".to_string(), json!(self.use_non_buggy_lcb));
        ret.insert(
            "rootEndingBonusPoints".to_string(),
            json!(self.root_ending_bonus_points),
        );

        ret.insert(
            "rootPruneUselessMoves".to_string(),
            json!(self.root_prune_useless_moves),
        );
        ret.insert(
            "conservativePass".to_string(),
            json!(self.conservative_pass),
        );
        ret.insert(
            "fillDameBeforePass".to_string(),
            json!(self.fill_dame_before_pass),
        );
        ret.insert("wideRootNoise".to_string(), json!(self.wide_root_noise));
        ret.insert(
            "puctVarExploration".to_string(),
            json!(self.puct_var_exploration),
        );
        ret.insert(
            "enablePassingHacks".to_string(),
            json!(self.enable_passing_hacks),
        );
        ret.insert(
            "enableMorePassingHacks".to_string(),
            json!(self.enable_more_passing_hacks),
        );

        ret.insert(
            "playoutDoublingAdvantage".to_string(),
            json!(self.playout_doubling_advantage),
        );
        ret.insert(
            "playoutDoublingAdvantagePla".to_string(),
            json!(player_io::player_to_string_short(
                self.playout_doubling_advantage_pla
            )),
        );

        ret.insert(
            "nnPolicyTemperature".to_string(),
            json!(self.nn_policy_temperature),
        );

        ret.insert(
            "ignorePreRootHistory".to_string(),
            json!(self.ignore_pre_root_history),
        );
        ret.insert(
            "ignoreAllHistory".to_string(),
            json!(self.ignore_all_history),
        );

        ret.insert(
            "subtreeValueBiasFactor".to_string(),
            json!(self.subtree_value_bias_factor),
        );
        ret.insert(
            "subtreeValueBiasTableNumShards".to_string(),
            json!(self.subtree_value_bias_table_num_shards),
        );
        ret.insert(
            "subtreeValueBiasFreeProp".to_string(),
            json!(self.subtree_value_bias_free_prop),
        );
        ret.insert(
            "subtreeValueBiasWeightExponent".to_string(),
            json!(self.subtree_value_bias_weight_exponent),
        );

        ret.insert(
            "numVirtualLossesPerThread".to_string(),
            json!(self.num_virtual_losses_per_thread),
        );

        ret.insert("numSearchThreads".to_string(), json!(self.num_threads));
        ret.insert(
            "minPlayoutsPerThread".to_string(),
            json!(self.min_playouts_per_thread),
        );
        ret.insert("maxVisits".to_string(), json!(self.max_visits));
        ret.insert("maxPlayouts".to_string(), json!(self.max_playouts));
        ret.insert("maxTime".to_string(), json!(self.max_time));

        ret.insert(
            "maxVisitsPondering".to_string(),
            json!(self.max_visits_pondering),
        );
        ret.insert(
            "maxPlayoutsPondering".to_string(),
            json!(self.max_playouts_pondering),
        );
        ret.insert(
            "maxTimePondering".to_string(),
            json!(self.max_time_pondering),
        );

        ret.insert("lagBuffer".to_string(), json!(self.lag_buffer));

        ret.insert(
            "searchFactorAfterOnePass".to_string(),
            json!(self.search_factor_after_one_pass),
        );
        ret.insert(
            "searchFactorAfterTwoPass".to_string(),
            json!(self.search_factor_after_two_pass),
        );

        ret.insert(
            "treeReuseCarryOverTimeFactor".to_string(),
            json!(self.tree_reuse_carry_over_time_factor),
        );
        ret.insert(
            "overallocateTimeFactor".to_string(),
            json!(self.overallocate_time_factor),
        );
        ret.insert(
            "midgameTimeFactor".to_string(),
            json!(self.midgame_time_factor),
        );
        ret.insert(
            "midgameTurnPeakTime".to_string(),
            json!(self.midgame_turn_peak_time),
        );
        ret.insert(
            "endgameTurnTimeDecay".to_string(),
            json!(self.endgame_turn_time_decay),
        );
        ret.insert(
            "obviousMovesTimeFactor".to_string(),
            json!(self.obvious_moves_time_factor),
        );
        ret.insert(
            "obviousMovesPolicyEntropyTolerance".to_string(),
            json!(self.obvious_moves_policy_entropy_tolerance),
        );
        ret.insert(
            "obviousMovesPolicySurpriseTolerance".to_string(),
            json!(self.obvious_moves_policy_surprise_tolerance),
        );

        ret.insert(
            "futileVisitsThreshold".to_string(),
            json!(self.futile_visits_threshold),
        );

        ret.insert(
            "humanSLCpuctExploration".to_string(),
            json!(self.human_sl_cpuct_exploration),
        );
        ret.insert(
            "humanSLCpuctPermanent".to_string(),
            json!(self.human_sl_cpuct_permanent),
        );

        ret.insert(
            "humanSLRootExploreProbWeightless".to_string(),
            json!(self.human_sl_root_explore_prob_weightless),
        );
        ret.insert(
            "humanSLRootExploreProbWeightful".to_string(),
            json!(self.human_sl_root_explore_prob_weightful),
        );
        ret.insert(
            "humanSLPlaExploreProbWeightless".to_string(),
            json!(self.human_sl_pla_explore_prob_weightless),
        );
        ret.insert(
            "humanSLPlaExploreProbWeightful".to_string(),
            json!(self.human_sl_pla_explore_prob_weightful),
        );
        ret.insert(
            "humanSLOppExploreProbWeightless".to_string(),
            json!(self.human_sl_opp_explore_prob_weightless),
        );
        ret.insert(
            "humanSLOppExploreProbWeightful".to_string(),
            json!(self.human_sl_opp_explore_prob_weightful),
        );

        ret.insert(
            "humanSLChosenMoveProp".to_string(),
            json!(self.human_sl_chosen_move_prop),
        );
        ret.insert(
            "humanSLChosenMoveIgnorePass".to_string(),
            json!(self.human_sl_chosen_move_ignore_pass),
        );
        ret.insert(
            "humanSLChosenMovePiklLambda".to_string(),
            json!(self.human_sl_chosen_move_pikl_lambda),
        );

        Value::Object(ret)
    }

    /// Hash of all parameters; unequal parameters hash differently with high probability.
    pub fn get_hash(&self) -> Hash128 {
        let mut ret = self.changeable_parameters_to_json();

        // Parameters deliberately omitted from the changeable JSON.
        ret["avoidMYTDaggerHackPla"] = json!(self.avoid_mytd_dagger_hack_pla as i32);
        ret["avoidRepeatedPatternUtility"] = json!(self.avoid_repeated_pattern_utility);
        ret["antiMirror"] = json!(self.anti_mirror);
        ret["useEvalCache"] = json!(self.use_eval_cache);
        ret["evalCacheMinVisits"] = json!(self.eval_cache_min_visits);
        ret["nodeTableShardsPowerOfTwo"] = json!(self.node_table_shards_power_of_two);

        // `SgfMetadata::get_hash` panics on uninitialized metadata, so only hash when set.
        ret["humanSLProfileInitialized"] = json!(self.human_sl_profile.initialized);
        if self.human_sl_profile.initialized {
            let profile_hash = self.human_sl_profile.get_hash(P_BLACK);
            ret["humanSLProfileHash0"] = json!(profile_hash.hash0);
            ret["humanSLProfileHash1"] = json!(profile_hash.hash1);
        }

        let dumped = ret.to_string();
        let digest = sha2::sha256(dumped.as_bytes());
        let hash0 = u64::from_le_bytes(digest[0..8].try_into().expect("8 bytes"));
        let hash1 = u64::from_le_bytes(digest[8..16].try_into().expect("8 bytes"));
        Hash128::new(hash0, hash1)
    }

    /// Throw if the dynamic parameters differ from `initial` on parameters that must not change
    /// after initialization.
    pub fn fail_if_params_differ_on_unchangeable_parameter(
        initial: &Self,
        dynamic: &Self,
    ) -> Result<(), StringError> {
        if dynamic.node_table_shards_power_of_two != initial.node_table_shards_power_of_two {
            return Err(StringError {
                message: "Cannot change nodeTableShardsPowerOfTwo after initialization".to_string(),
            });
        }
        if dynamic.use_eval_cache != initial.use_eval_cache {
            return Err(StringError {
                message: "Cannot change useEvalCache after initialization".to_string(),
            });
        }
        if dynamic.eval_cache_min_visits != initial.eval_cache_min_visits {
            return Err(StringError {
                message: "Cannot change evalCacheMinVisits after initialization".to_string(),
            });
        }
        Ok(())
    }
}

impl fmt::Display for SearchParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        macro_rules! print_param {
            ($name:ident) => {
                writeln!(f, concat!(stringify!($name), ": {}"), self.$name)?
            };
        }

        print_param!(win_loss_utility_factor);
        print_param!(static_score_utility_factor);
        print_param!(dynamic_score_utility_factor);
        print_param!(dynamic_score_center_zero_weight);
        print_param!(dynamic_score_center_scale);
        print_param!(no_result_utility_for_white);
        print_param!(draw_equivalent_wins_for_white);

        print_param!(cpuct_exploration);
        print_param!(cpuct_exploration_log);
        print_param!(cpuct_exploration_base);

        print_param!(cpuct_utility_stdev_prior);
        print_param!(cpuct_utility_stdev_prior_weight);
        print_param!(cpuct_utility_stdev_scale);

        print_param!(fpu_reduction_max);
        print_param!(fpu_loss_prop);

        print_param!(fpu_parent_weight_by_visited_policy);
        print_param!(fpu_parent_weight_by_visited_policy_pow);
        print_param!(fpu_parent_weight);

        print_param!(policy_optimism);

        print_param!(value_weight_exponent);
        print_param!(use_noise_pruning);
        print_param!(noise_prune_utility_scale);
        print_param!(noise_pruning_cap);

        print_param!(use_uncertainty);
        print_param!(uncertainty_coeff);
        print_param!(uncertainty_exponent);
        print_param!(uncertainty_max_weight);

        print_param!(use_graph_search);
        print_param!(graph_search_rep_bound);
        print_param!(graph_search_catch_up_leak_prob);

        print_param!(root_noise_enabled);
        print_param!(root_dirichlet_noise_total_concentration);
        print_param!(root_dirichlet_noise_weight);

        print_param!(root_policy_temperature);
        print_param!(root_policy_temperature_early);
        print_param!(root_fpu_reduction_max);
        print_param!(root_fpu_loss_prop);
        print_param!(root_num_symmetries_to_sample);
        print_param!(root_symmetry_pruning);

        print_param!(root_desired_per_child_visits_coeff);

        print_param!(root_policy_optimism);

        print_param!(chosen_move_temperature);
        print_param!(chosen_move_temperature_early);
        print_param!(chosen_move_temperature_halflife);
        print_param!(chosen_move_temperature_only_below_prob);
        print_param!(chosen_move_subtract);
        print_param!(chosen_move_prune);

        print_param!(use_lcb_for_selection);
        print_param!(lcb_stdevs);
        print_param!(min_visit_prop_for_lcb);
        print_param!(use_non_buggy_lcb);

        print_param!(root_ending_bonus_points);
        print_param!(root_prune_useless_moves);
        print_param!(conservative_pass);
        print_param!(fill_dame_before_pass);
        writeln!(
            f,
            "avoidMYTDaggerHackPla: {}",
            self.avoid_mytd_dagger_hack_pla as i32
        )?;
        print_param!(wide_root_noise);
        print_param!(puct_var_exploration);
        print_param!(enable_passing_hacks);
        print_param!(enable_more_passing_hacks);

        print_param!(playout_doubling_advantage);
        writeln!(
            f,
            "playoutDoublingAdvantagePla: {}",
            player_io::player_to_string_short(self.playout_doubling_advantage_pla)
        )?;

        print_param!(avoid_repeated_pattern_utility);

        print_param!(nn_policy_temperature);
        print_param!(anti_mirror);

        print_param!(ignore_pre_root_history);
        print_param!(ignore_all_history);

        print_param!(subtree_value_bias_factor);
        print_param!(subtree_value_bias_table_num_shards);
        print_param!(subtree_value_bias_free_prop);
        print_param!(subtree_value_bias_weight_exponent);

        print_param!(use_eval_cache);
        print_param!(eval_cache_min_visits);

        print_param!(node_table_shards_power_of_two);
        print_param!(num_virtual_losses_per_thread);

        print_param!(num_threads);
        print_param!(min_playouts_per_thread);
        print_param!(max_visits);
        print_param!(max_playouts);
        print_param!(max_time);

        print_param!(max_visits_pondering);
        print_param!(max_playouts_pondering);
        print_param!(max_time_pondering);

        print_param!(lag_buffer);

        print_param!(search_factor_after_one_pass);
        print_param!(search_factor_after_two_pass);

        print_param!(tree_reuse_carry_over_time_factor);
        print_param!(overallocate_time_factor);
        print_param!(midgame_time_factor);
        print_param!(midgame_turn_peak_time);
        print_param!(endgame_turn_time_decay);
        print_param!(obvious_moves_time_factor);
        print_param!(obvious_moves_policy_entropy_tolerance);
        print_param!(obvious_moves_policy_surprise_tolerance);

        print_param!(futile_visits_threshold);

        print_param!(human_sl_cpuct_exploration);
        print_param!(human_sl_cpuct_permanent);
        print_param!(human_sl_root_explore_prob_weightless);
        print_param!(human_sl_root_explore_prob_weightful);
        print_param!(human_sl_pla_explore_prob_weightless);
        print_param!(human_sl_pla_explore_prob_weightful);
        print_param!(human_sl_opp_explore_prob_weightless);
        print_param!(human_sl_opp_explore_prob_weightful);
        print_param!(human_sl_chosen_move_prop);
        print_param!(human_sl_chosen_move_ignore_pass);
        print_param!(human_sl_chosen_move_pikl_lambda);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values() {
        let params = SearchParams::new();
        assert_eq!(params.win_loss_utility_factor, 1.0);
        assert_eq!(params.cpuct_exploration, 1.0);
        assert!(!params.use_graph_search);
        assert_eq!(params.max_visits, 1_i64 << 50);
        assert_eq!(params.num_threads, 1);
        assert_eq!(params.avoid_mytd_dagger_hack_pla, C_EMPTY);
        assert_eq!(params.human_sl_chosen_move_pikl_lambda, 1_000_000_000.0);
    }

    #[test]
    fn test_for_tests_v1_changes_defaults() {
        let params = SearchParams::for_tests_v1();
        assert_eq!(params.static_score_utility_factor, 0.1);
        assert!(params.use_lcb_for_selection);
        assert_eq!(params.lcb_stdevs, 5.0);
        assert!(params.conservative_pass);
    }

    #[test]
    fn test_for_tests_v2_enables_graph_search() {
        let params = SearchParams::for_tests_v2();
        assert!(params.use_graph_search);
        assert!(params.use_noise_pruning);
        assert!(params.use_uncertainty);
        assert_eq!(params.subtree_value_bias_factor, 0.45);
    }

    #[test]
    fn test_basic_decent_params_enables_passing_hacks() {
        let params = SearchParams::basic_decent_params();
        assert!(params.enable_passing_hacks);
        assert!(params.use_graph_search);
        assert_eq!(params.min_visit_prop_for_lcb, 0.20);
    }

    #[test]
    fn test_equality_and_inequality() {
        let a = SearchParams::new();
        let b = SearchParams::new();
        assert_eq!(a, b);
        let mut c = SearchParams::new();
        c.cpuct_exploration = 2.0;
        assert_ne!(a, c);
    }

    #[test]
    fn test_get_hash_stable_and_sensitive() {
        let a = SearchParams::new();
        let b = SearchParams::new();
        assert_eq!(a.get_hash(), b.get_hash());

        let mut c = SearchParams::new();
        c.cpuct_exploration = 2.0;
        assert_ne!(a.get_hash(), c.get_hash());
    }

    #[test]
    fn test_fail_if_node_table_shards_changes() {
        let initial = SearchParams::new();
        let mut dynamic = SearchParams::new();
        dynamic.node_table_shards_power_of_two = 8;
        assert!(
            SearchParams::fail_if_params_differ_on_unchangeable_parameter(&initial, &dynamic)
                .is_err()
        );
    }

    #[test]
    fn test_fail_if_use_eval_cache_changes() {
        let initial = SearchParams::new();
        let mut dynamic = SearchParams::new();
        dynamic.use_eval_cache = true;
        assert!(
            SearchParams::fail_if_params_differ_on_unchangeable_parameter(&initial, &dynamic)
                .is_err()
        );
    }

    #[test]
    fn test_fail_if_eval_cache_min_visits_changes() {
        let initial = SearchParams::new();
        let mut dynamic = SearchParams::new();
        dynamic.eval_cache_min_visits = 50;
        assert!(
            SearchParams::fail_if_params_differ_on_unchangeable_parameter(&initial, &dynamic)
                .is_err()
        );
    }

    #[test]
    fn test_changeable_parameters_to_json() {
        let params = SearchParams::for_tests_v1();
        let json = params.changeable_parameters_to_json();
        assert_eq!(json["cpuctExploration"], 0.9);
        assert_eq!(json["numSearchThreads"], 1);
        assert_eq!(json["playoutDoublingAdvantagePla"], "?");
    }

    #[test]
    fn test_display_contains_parameters() {
        let params = SearchParams::new();
        let s = params.to_string();
        assert!(s.contains("win_loss_utility_factor: 1"));
        assert!(s.contains("cpuct_exploration: 1"));
        assert!(s.contains("use_graph_search: false"));
    }
}
