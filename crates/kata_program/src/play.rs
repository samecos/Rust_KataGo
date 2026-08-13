//! Core data structures and game-running logic used for games and self-play.
//!
//! Corresponds to `cpp/program/play.h` and `cpp/program/play.cpp`. The
//! data-structure portions were ported in earlier slices; this slice fills in
//! `GameRunner::runGame`, `Play::runGame`, the fork helpers, and policy-target
//! extraction.

#![allow(
    clippy::all,
    dead_code,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::missing_safety_doc
)]

use kata_core::config::ConfigParser;
use kata_core::global::StringError;
use kata_core::hash::Hash128;
use kata_core::logger::Logger;
use kata_core::rng::Rand;
use kata_core::thread::counter::WaitableFlag;
use kata_data::sgf::PositionSample;
use kata_data::training::{
    ChangedNeuralNet, FinishedGameData, NNRawStats, PolicyTarget, PolicyTargetMove,
    QValueTargetMove, QValueTargets, SidePosition, ValueTargets,
};
use kata_game::board::{
    Board, C_EMPTY, Loc, MAX_ARR_SIZE, MAX_LEN, NULL_LOC, P_BLACK, P_WHITE, Player, get_opp,
    location,
};
use kata_game::history::BoardHistory;
use kata_game::rules::{KoRule, Rules, ScoringRule, TaxRule};
use kata_nn::backend::NNResultBuf;
use kata_nn::eval::NnEvaluator;
use kata_nn::inputs::{MiscNNInputParams, NNOutput, fill_scoring, nn_pos};
use kata_nn::score_value;
use kata_search::node::{SearchChildPointer, SearchNode};
use kata_search::params::SearchParams;
use kata_search::reported_values::ReportedSearchValues;
use kata_search::search::Search;
use std::collections::BTreeSet;
use std::sync::Mutex;
use std::sync::atomic::Ordering;

use crate::play_settings::PlaySettings;
use crate::play_utils::*;

/// A starting position that may be used instead of a fresh empty game.
///
/// Mirrors `InitialPosition` in `cpp/program/play.h`.
#[derive(Clone)]
pub struct InitialPosition {
    pub board: Board,
    pub hist: BoardHistory,
    pub pla: Player,
    pub is_plain_fork: bool,
    pub is_seki_fork: bool,
    pub is_hint_fork: bool,
    pub training_weight: f64,
}

impl InitialPosition {
    /// Construct a new initial position.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        board: Board,
        hist: BoardHistory,
        pla: Player,
        is_plain_fork: bool,
        is_seki_fork: bool,
        is_hint_fork: bool,
        training_weight: f64,
    ) -> Self {
        Self {
            board,
            hist,
            pla,
            is_plain_fork,
            is_seki_fork,
            is_hint_fork,
            training_weight,
        }
    }
}

impl Default for InitialPosition {
    fn default() -> Self {
        Self {
            board: Board::default(),
            hist: BoardHistory::default(),
            pla: C_EMPTY,
            is_plain_fork: false,
            is_seki_fork: false,
            is_hint_fork: false,
            training_weight: 1.0,
        }
    }
}

/// Mutable collection of fork positions that can be sampled during self-play.
///
/// Mirrors `ForkData` in `cpp/program/play.h` and `cpp/program/play.cpp`.
pub struct ForkData {
    inner: Mutex<ForkDataInner>,
}

struct ForkDataInner {
    forks: Vec<InitialPosition>,
    seki_forks: Vec<InitialPosition>,
}

impl ForkData {
    /// Create an empty fork data store.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(ForkDataInner {
                forks: Vec::new(),
                seki_forks: Vec::new(),
            }),
        }
    }

    /// Add a plain fork position.
    pub fn add(&self, pos: InitialPosition) {
        let mut inner = self.inner.lock().expect("ForkData mutex poisoned");
        inner.forks.push(pos);
    }

    /// Remove and return a uniformly random plain fork position, if any.
    pub fn get(&self, rand: &mut Rand) -> Option<InitialPosition> {
        let mut inner = self.inner.lock().expect("ForkData mutex poisoned");
        let len = inner.forks.len();
        if len == 0 {
            return None;
        }
        let idx = rand.next_u32_bounded(len as u32) as usize;
        let last = len - 1;
        inner.forks.swap(idx, last);
        inner.forks.pop()
    }

    /// Add a seki fork position, replacing a random old one if the cap is reached.
    pub fn add_seki(&self, pos: InitialPosition, rand: &mut Rand) {
        let mut inner = self.inner.lock().expect("ForkData mutex poisoned");
        if inner.seki_forks.len() >= 1000 {
            let idx = rand.next_u32_bounded(inner.seki_forks.len() as u32) as usize;
            inner.seki_forks[idx] = pos;
        } else {
            inner.seki_forks.push(pos);
        }
    }

    /// Remove and return a uniformly random seki fork position, if any.
    pub fn get_seki(&self, rand: &mut Rand) -> Option<InitialPosition> {
        let mut inner = self.inner.lock().expect("ForkData mutex poisoned");
        let len = inner.seki_forks.len();
        if len == 0 {
            return None;
        }
        let idx = rand.next_u32_bounded(len as u32) as usize;
        let last = len - 1;
        inner.seki_forks.swap(idx, last);
        inner.seki_forks.pop()
    }

    /// Number of plain forks currently stored.
    pub fn fork_count(&self) -> usize {
        let inner = self.inner.lock().expect("ForkData mutex poisoned");
        inner.forks.len()
    }

    /// Number of seki forks currently stored.
    pub fn seki_fork_count(&self) -> usize {
        let inner = self.inner.lock().expect("ForkData mutex poisoned");
        inner.seki_forks.len()
    }
}

impl Default for ForkData {
    fn default() -> Self {
        Self::new()
    }
}

/// Miscellaneous per-game properties that influence play and training.
///
/// Mirrors `OtherGameProperties` in `cpp/program/play.h`.
#[derive(Debug, Clone)]
pub struct OtherGameProperties {
    pub is_sgf_pos: bool,
    pub is_hint_pos: bool,
    pub allow_policy_init: bool,
    pub is_fork: bool,
    pub is_hint_fork: bool,

    pub hint_turn: i32,
    pub hint_pos_hash: Hash128,
    pub hint_loc: Loc,

    pub training_weight: f64,

    /// These two behave slightly differently than the ones in searchParams - as
    /// properties for the whole game, they make the playouts *actually* vary
    /// instead of only making the neural net think they do.
    pub playout_doubling_advantage: f64,
    pub playout_doubling_advantage_pla: Player,
}

impl Default for OtherGameProperties {
    fn default() -> Self {
        Self {
            is_sgf_pos: false,
            is_hint_pos: false,
            allow_policy_init: true,
            is_fork: false,
            is_hint_fork: false,
            hint_turn: -1,
            hint_pos_hash: Hash128::default(),
            hint_loc: NULL_LOC,
            training_weight: 1.0,
            playout_doubling_advantage: 0.0,
            playout_doubling_advantage_pla: C_EMPTY,
        }
    }
}

fn possible_string_set(possibles: BTreeSet<&'static str>) -> BTreeSet<String> {
    possibles.into_iter().map(|s| s.to_string()).collect()
}

/// Object choosing random initial rules and board sizes for games. Threadsafe.
///
/// Mirrors `GameInitializer` in `cpp/program/play.h` and `cpp/program/play.cpp`.
pub struct GameInitializer {
    create_game_mutex: Mutex<()>,
    rand: Mutex<Rand>,

    allowed_ko_rule_strs: Vec<String>,
    allowed_scoring_rule_strs: Vec<String>,
    allowed_tax_rule_strs: Vec<String>,
    allowed_multi_stone_suicide_legals: Vec<bool>,
    allowed_buttons: Vec<bool>,

    allowed_ko_rules: Vec<KoRule>,
    allowed_scoring_rules: Vec<ScoringRule>,
    allowed_tax_rules: Vec<TaxRule>,

    allowed_b_sizes: Vec<(i32, i32)>,
    allowed_b_size_rel_probs: Vec<f64>,

    komi_mean: f32,
    komi_stdev: f32,
    komi_allow_integer_prob: f64,
    handicap_prob: f64,
    handicap_compensate_komi_prob: f64,
    fork_compensate_komi_prob: f64,
    sgf_compensate_komi_prob: f64,
    komi_big_stdev_prob: f64,
    komi_big_stdev: f32,
    komi_bigger_stdev_prob: f64,
    komi_bigger_stdev: f32,
    handicap_komi_interp_zero_prob: f64,
    sgf_komi_interp_zero_prob: f64,
    komi_auto: bool,

    num_extra_black_fixed: i32,
    no_result_stdev: f64,
    draw_rand_radius: f64,

    start_poses: Vec<PositionSample>,
    start_pos_cum_probs: Vec<f64>,
    start_poses_prob: f64,

    hint_poses: Vec<PositionSample>,
    hint_pos_cum_probs: Vec<f64>,
    hint_poses_prob: f64,

    min_board_x_size: i32,
    min_board_y_size: i32,
    max_board_x_size: i32,
    max_board_y_size: i32,
}

impl GameInitializer {
    /// Construct a new game initializer from config.
    pub fn new(cfg: &ConfigParser, logger: &Logger) -> Result<Self, StringError> {
        let mut init = Self {
            create_game_mutex: Mutex::new(()),
            rand: Mutex::new(Rand::new()),
            allowed_ko_rule_strs: Vec::new(),
            allowed_scoring_rule_strs: Vec::new(),
            allowed_tax_rule_strs: Vec::new(),
            allowed_multi_stone_suicide_legals: Vec::new(),
            allowed_buttons: Vec::new(),
            allowed_ko_rules: Vec::new(),
            allowed_scoring_rules: Vec::new(),
            allowed_tax_rules: Vec::new(),
            allowed_b_sizes: Vec::new(),
            allowed_b_size_rel_probs: Vec::new(),
            komi_mean: 7.5,
            komi_stdev: 0.0,
            komi_allow_integer_prob: 1.0,
            handicap_prob: 0.0,
            handicap_compensate_komi_prob: 0.0,
            fork_compensate_komi_prob: 0.0,
            sgf_compensate_komi_prob: 0.0,
            komi_big_stdev_prob: 0.0,
            komi_big_stdev: 10.0,
            komi_bigger_stdev_prob: 0.0,
            komi_bigger_stdev: 30.0,
            handicap_komi_interp_zero_prob: 0.0,
            sgf_komi_interp_zero_prob: 0.0,
            komi_auto: false,
            num_extra_black_fixed: 0,
            no_result_stdev: 0.0,
            draw_rand_radius: 0.0,
            start_poses: Vec::new(),
            start_pos_cum_probs: Vec::new(),
            start_poses_prob: 0.0,
            hint_poses: Vec::new(),
            hint_pos_cum_probs: Vec::new(),
            hint_poses_prob: 0.0,
            min_board_x_size: 0,
            min_board_y_size: 0,
            max_board_x_size: 0,
            max_board_y_size: 0,
        };
        init.init_shared(cfg, logger)?;
        Ok(init)
    }

    /// Construct a new game initializer with an explicit random seed.
    pub fn new_with_seed(
        cfg: &ConfigParser,
        logger: &Logger,
        rand_seed: &str,
    ) -> Result<Self, StringError> {
        let mut init = Self {
            create_game_mutex: Mutex::new(()),
            rand: Mutex::new(Rand::new_from_seed(rand_seed)),
            allowed_ko_rule_strs: Vec::new(),
            allowed_scoring_rule_strs: Vec::new(),
            allowed_tax_rule_strs: Vec::new(),
            allowed_multi_stone_suicide_legals: Vec::new(),
            allowed_buttons: Vec::new(),
            allowed_ko_rules: Vec::new(),
            allowed_scoring_rules: Vec::new(),
            allowed_tax_rules: Vec::new(),
            allowed_b_sizes: Vec::new(),
            allowed_b_size_rel_probs: Vec::new(),
            komi_mean: 7.5,
            komi_stdev: 0.0,
            komi_allow_integer_prob: 1.0,
            handicap_prob: 0.0,
            handicap_compensate_komi_prob: 0.0,
            fork_compensate_komi_prob: 0.0,
            sgf_compensate_komi_prob: 0.0,
            komi_big_stdev_prob: 0.0,
            komi_big_stdev: 10.0,
            komi_bigger_stdev_prob: 0.0,
            komi_bigger_stdev: 30.0,
            handicap_komi_interp_zero_prob: 0.0,
            sgf_komi_interp_zero_prob: 0.0,
            komi_auto: false,
            num_extra_black_fixed: 0,
            no_result_stdev: 0.0,
            draw_rand_radius: 0.0,
            start_poses: Vec::new(),
            start_pos_cum_probs: Vec::new(),
            start_poses_prob: 0.0,
            hint_poses: Vec::new(),
            hint_pos_cum_probs: Vec::new(),
            hint_poses_prob: 0.0,
            min_board_x_size: 0,
            min_board_y_size: 0,
            max_board_x_size: 0,
            max_board_y_size: 0,
        };
        init.init_shared(cfg, logger)?;
        Ok(init)
    }

    #[allow(clippy::too_many_lines)]
    fn init_shared(&mut self, cfg: &ConfigParser, _logger: &Logger) -> Result<(), StringError> {
        let to_string_err = |e: kata_core::config::ConfigError| StringError::new(e.to_string());

        self.allowed_ko_rule_strs = cfg
            .get_strings_set("koRules", &possible_string_set(Rules::ko_rule_strings()))
            .map_err(to_string_err)?;
        self.allowed_scoring_rule_strs = cfg
            .get_strings_set(
                "scoringRules",
                &possible_string_set(Rules::scoring_rule_strings()),
            )
            .map_err(to_string_err)?;
        self.allowed_tax_rule_strs = cfg
            .get_strings_set("taxRules", &possible_string_set(Rules::tax_rule_strings()))
            .map_err(to_string_err)?;
        self.allowed_multi_stone_suicide_legals = cfg
            .get_bools("multiStoneSuicideLegals")
            .map_err(to_string_err)?;
        self.allowed_buttons = cfg.get_bools("hasButtons").map_err(to_string_err)?;

        self.allowed_ko_rules = self
            .allowed_ko_rule_strs
            .iter()
            .map(|s| {
                s.parse::<KoRule>()
                    .expect("ko rule was validated by get_strings_set")
            })
            .collect();
        self.allowed_scoring_rules = self
            .allowed_scoring_rule_strs
            .iter()
            .map(|s| {
                s.parse::<ScoringRule>()
                    .expect("scoring rule was validated by get_strings_set")
            })
            .collect();
        self.allowed_tax_rules = self
            .allowed_tax_rule_strs
            .iter()
            .map(|s| {
                s.parse::<TaxRule>()
                    .expect("tax rule was validated by get_strings_set")
            })
            .collect();

        if self.allowed_ko_rules.is_empty() {
            return Err(StringError::new(format!(
                "koRules must have at least one value in {}",
                cfg.file_name()
            )));
        }
        if self.allowed_scoring_rules.is_empty() {
            return Err(StringError::new(format!(
                "scoringRules must have at least one value in {}",
                cfg.file_name()
            )));
        }
        if self.allowed_tax_rules.is_empty() {
            return Err(StringError::new(format!(
                "taxRules must have at least one value in {}",
                cfg.file_name()
            )));
        }
        if self.allowed_multi_stone_suicide_legals.is_empty() {
            return Err(StringError::new(format!(
                "multiStoneSuicideLegals must have at least one value in {}",
                cfg.file_name()
            )));
        }
        if self.allowed_buttons.is_empty() {
            return Err(StringError::new(format!(
                "hasButtons must have at least one value in {}",
                cfg.file_name()
            )));
        }

        let has_area_scoring = self.allowed_scoring_rules.contains(&ScoringRule::Area);
        let has_true_button = self.allowed_buttons.contains(&true);
        if !has_area_scoring && has_true_button {
            return Err(StringError::new(format!(
                "If scoringRules does not include AREA, hasButtons must be false in {}",
                cfg.file_name()
            )));
        }

        let b_sizes_present = cfg.contains("bSizes");
        let b_sizes_xy_present = cfg.contains("bSizesXY");
        if b_sizes_present == b_sizes_xy_present {
            return Err(StringError::new(
                "Must specify exactly one of bSizes or bSizesXY".to_string(),
            ));
        }

        if b_sizes_present {
            let allowed_b_edges = cfg
                .get_ints("bSizes", 2, MAX_LEN as i32)
                .map_err(to_string_err)?;
            let allowed_b_edge_rel_probs = cfg
                .get_doubles("bSizeRelProbs", 0.0, 1e100)
                .map_err(to_string_err)?;
            let rel_prob_sum: f64 = allowed_b_edge_rel_probs.iter().sum();
            if rel_prob_sum <= 1e-100 {
                return Err(StringError::new(
                    "bSizeRelProbs must sum to a positive value".to_string(),
                ));
            }
            let allow_rectangle_prob = if cfg.contains("allowRectangleProb") {
                cfg.get_double("allowRectangleProb", 0.0, 1.0)
                    .map_err(to_string_err)?
            } else {
                0.0
            };

            if allowed_b_edges.is_empty() {
                return Err(StringError::new(format!(
                    "bSizes must have at least one value in {}",
                    cfg.file_name()
                )));
            }
            if allowed_b_edges.len() != allowed_b_edge_rel_probs.len() {
                return Err(StringError::new(format!(
                    "bSizes and bSizeRelProbs must have same number of values in {}",
                    cfg.file_name()
                )));
            }

            self.allowed_b_sizes.clear();
            self.allowed_b_size_rel_probs.clear();
            for (i, &x) in allowed_b_edges.iter().enumerate() {
                for (j, &y) in allowed_b_edges.iter().enumerate() {
                    if x == y {
                        self.allowed_b_sizes.push((x, y));
                        self.allowed_b_size_rel_probs.push(
                            (1.0 - allow_rectangle_prob) * allowed_b_edge_rel_probs[i]
                                / rel_prob_sum
                                + allow_rectangle_prob
                                    * allowed_b_edge_rel_probs[i]
                                    * allowed_b_edge_rel_probs[j]
                                    / rel_prob_sum
                                    / rel_prob_sum,
                        );
                    } else if allow_rectangle_prob > 0.0 {
                        self.allowed_b_sizes.push((x, y));
                        self.allowed_b_size_rel_probs.push(
                            allow_rectangle_prob
                                * allowed_b_edge_rel_probs[i]
                                * allowed_b_edge_rel_probs[j]
                                / rel_prob_sum
                                / rel_prob_sum,
                        );
                    }
                }
            }
        } else if b_sizes_xy_present {
            if cfg.contains("allowRectangleProb") {
                return Err(StringError::new(
                    "Cannot specify allowRectangleProb when specifying bSizesXY, please adjust the relative frequency of rectangles yourself".to_string(),
                ));
            }
            self.allowed_b_sizes = cfg
                .get_non_negative_int_dashed_pairs("bSizesXY", 2, MAX_LEN as i32)
                .map_err(to_string_err)?;
            self.allowed_b_size_rel_probs = cfg
                .get_doubles("bSizeRelProbs", 0.0, 1e100)
                .map_err(to_string_err)?;
            let rel_prob_sum: f64 = self.allowed_b_size_rel_probs.iter().sum();
            if rel_prob_sum <= 1e-100 {
                return Err(StringError::new(
                    "bSizeRelProbs must sum to a positive value".to_string(),
                ));
            }
        }

        let has_komi_mean = cfg.contains("komiMean");
        let has_komi_auto =
            cfg.contains("komiAuto") && cfg.get_bool("komiAuto").map_err(to_string_err)?;
        if !has_komi_mean && !has_komi_auto {
            return Err(StringError::new(
                "Must specify either komiMean=<komi value> or komiAuto=True in config".to_string(),
            ));
        }
        if has_komi_mean && has_komi_auto {
            return Err(StringError::new(
                "Must specify only one of komiMean=<komi value> or komiAuto=True in config"
                    .to_string(),
            ));
        }

        self.komi_mean = if cfg.contains("komiMean") {
            cfg.get_float("komiMean", Rules::MIN_USER_KOMI, Rules::MAX_USER_KOMI)
                .map_err(to_string_err)?
        } else {
            7.5
        };
        self.komi_stdev = if cfg.contains("komiStdev") {
            cfg.get_float("komiStdev", 0.0, 60.0)
                .map_err(to_string_err)?
        } else {
            0.0
        };
        self.handicap_prob = if cfg.contains("handicapProb") {
            cfg.get_double("handicapProb", 0.0, 1.0)
                .map_err(to_string_err)?
        } else {
            0.0
        };
        self.handicap_compensate_komi_prob = if cfg.contains("handicapCompensateKomiProb") {
            cfg.get_double("handicapCompensateKomiProb", 0.0, 1.0)
                .map_err(to_string_err)?
        } else {
            0.0
        };
        self.komi_big_stdev_prob = if cfg.contains("komiBigStdevProb") {
            cfg.get_double("komiBigStdevProb", 0.0, 1.0)
                .map_err(to_string_err)?
        } else {
            0.0
        };
        self.komi_big_stdev = if cfg.contains("komiBigStdev") {
            cfg.get_float("komiBigStdev", 0.0, 60.0)
                .map_err(to_string_err)?
        } else {
            10.0
        };
        self.komi_bigger_stdev_prob = if cfg.contains("komiBiggerStdevProb") {
            cfg.get_double("komiBiggerStdevProb", 0.0, 1.0)
                .map_err(to_string_err)?
        } else {
            0.0
        };
        self.komi_bigger_stdev = if cfg.contains("komiBiggerStdev") {
            cfg.get_float("komiBiggerStdev", 0.0, 120.0)
                .map_err(to_string_err)?
        } else {
            30.0
        };
        self.handicap_komi_interp_zero_prob = if cfg.contains("handicapKomiInterpZeroProb") {
            cfg.get_double("handicapKomiInterpZeroProb", 0.0, 1.0)
                .map_err(to_string_err)?
        } else {
            0.0
        };
        self.sgf_komi_interp_zero_prob = if cfg.contains("sgfKomiInterpZeroProb") {
            cfg.get_double("sgfKomiInterpZeroProb", 0.0, 1.0)
                .map_err(to_string_err)?
        } else {
            0.0
        };
        self.komi_auto = if cfg.contains("komiAuto") {
            cfg.get_bool("komiAuto").map_err(to_string_err)?
        } else {
            false
        };

        self.fork_compensate_komi_prob = if cfg.contains("forkCompensateKomiProb") {
            cfg.get_double("forkCompensateKomiProb", 0.0, 1.0)
                .map_err(to_string_err)?
        } else {
            self.handicap_compensate_komi_prob
        };
        self.sgf_compensate_komi_prob = if cfg.contains("sgfCompensateKomiProb") {
            cfg.get_double("sgfCompensateKomiProb", 0.0, 1.0)
                .map_err(to_string_err)?
        } else {
            self.fork_compensate_komi_prob
        };
        self.komi_allow_integer_prob = if cfg.contains("komiAllowIntegerProb") {
            cfg.get_double("komiAllowIntegerProb", 0.0, 1.0)
                .map_err(to_string_err)?
        } else {
            1.0
        };

        self.start_poses_prob = 0.0;
        if cfg.contains("startPosesFromSgfDir") {
            self.start_poses_prob = cfg
                .get_double("startPosesProb", 0.0, 1.0)
                .map_err(to_string_err)?;
            // Loading positions from disk is deferred to a later slice.
            self.start_poses.clear();
            self.start_pos_cum_probs.clear();
            if self.start_poses.is_empty() {
                self.start_poses_prob = 0.0;
            }
        }

        self.hint_poses_prob = 0.0;
        if cfg.contains("hintPosesDir") {
            self.hint_poses_prob = cfg
                .get_double("hintPosesProb", 0.0, 1.0)
                .map_err(to_string_err)?;
            // Loading positions from disk is deferred to a later slice.
            self.hint_poses.clear();
            self.hint_pos_cum_probs.clear();
            if self.hint_poses.is_empty() {
                self.hint_poses_prob = 0.0;
            }
        }

        if self.allowed_b_sizes.is_empty() {
            return Err(StringError::new(format!(
                "bSizes or bSizesXY must have at least one value in {}",
                cfg.file_name()
            )));
        }
        if self.allowed_b_sizes.len() != self.allowed_b_size_rel_probs.len() {
            return Err(StringError::new(format!(
                "bSizes or bSizesXY and bSizeRelProbs must have same number of values in {}",
                cfg.file_name()
            )));
        }

        self.min_board_x_size = self.allowed_b_sizes[0].0;
        self.min_board_y_size = self.allowed_b_sizes[0].1;
        self.max_board_x_size = self.allowed_b_sizes[0].0;
        self.max_board_y_size = self.allowed_b_sizes[0].1;
        for &(x, y) in &self.allowed_b_sizes {
            self.min_board_x_size = self.min_board_x_size.min(x);
            self.min_board_y_size = self.min_board_y_size.min(y);
            self.max_board_x_size = self.max_board_x_size.max(x);
            self.max_board_y_size = self.max_board_y_size.max(y);
        }
        for pos in &self.hint_poses {
            self.min_board_x_size = self.min_board_x_size.min(pos.board.x_size);
            self.min_board_y_size = self.min_board_y_size.min(pos.board.y_size);
            self.max_board_x_size = self.max_board_x_size.max(pos.board.x_size);
            self.max_board_y_size = self.max_board_y_size.max(pos.board.y_size);
        }

        self.no_result_stdev = if cfg.contains("noResultStdev") {
            cfg.get_double("noResultStdev", 0.0, 1.0)
                .map_err(to_string_err)?
        } else {
            0.0
        };
        self.num_extra_black_fixed = if cfg.contains("numExtraBlackFixed") {
            cfg.get_int("numExtraBlackFixed", 1, 18)
                .map_err(to_string_err)?
        } else {
            0
        };
        self.draw_rand_radius = if cfg.contains("drawRandRadius") {
            cfg.get_double("drawRandRadius", 0.0, 1.0)
                .map_err(to_string_err)?
        } else {
            0.0
        };

        Ok(())
    }

    /// Return a copy of the allowed board sizes.
    pub fn get_allowed_b_sizes(&self) -> Vec<(i32, i32)> {
        self.allowed_b_sizes.clone()
    }

    /// Minimum X board size among allowed sizes and hint positions.
    pub fn get_min_board_x_size(&self) -> i32 {
        self.min_board_x_size
    }

    /// Minimum Y board size among allowed sizes and hint positions.
    pub fn get_min_board_y_size(&self) -> i32 {
        self.min_board_y_size
    }

    /// Maximum X board size among allowed sizes and hint positions.
    pub fn get_max_board_x_size(&self) -> i32 {
        self.max_board_x_size
    }

    /// Maximum Y board size among allowed sizes and hint positions.
    pub fn get_max_board_y_size(&self) -> i32 {
        self.max_board_y_size
    }

    /// Return true if the given board size is allowed.
    pub fn is_allowed_b_size(&self, x_size: i32, y_size: i32) -> bool {
        self.allowed_b_sizes.contains(&(x_size, y_size))
    }

    /// Randomize the scoring/tax/button fields of the given rules.
    pub fn randomize_scoring_and_tax_rules(
        &self,
        mut rules: Rules,
        rand_to_use: &mut Rand,
    ) -> Rules {
        rules.scoring_rule = self.allowed_scoring_rules
            [rand_to_use.next_u32_bounded(self.allowed_scoring_rules.len() as u32) as usize];
        rules.tax_rule = self.allowed_tax_rules
            [rand_to_use.next_u32_bounded(self.allowed_tax_rules.len() as u32) as usize];
        rules.has_button = if rules.scoring_rule == ScoringRule::Area {
            self.allowed_buttons
                [rand_to_use.next_u32_bounded(self.allowed_buttons.len() as u32) as usize]
        } else {
            false
        };
        rules
    }

    /// Create a fresh randomized ruleset.
    pub fn create_rules(&self) -> Rules {
        let _lock = self
            .create_game_mutex
            .lock()
            .expect("createGame mutex poisoned");
        let mut rand = self.rand.lock().expect("rand mutex poisoned");
        self.create_rules_unsynchronized(&mut rand)
    }

    fn create_rules_unsynchronized(&self, rand: &mut Rand) -> Rules {
        let mut rules = Rules::default();
        rules.ko_rule = self.allowed_ko_rules
            [rand.next_u32_bounded(self.allowed_ko_rules.len() as u32) as usize];
        rules.scoring_rule = self.allowed_scoring_rules
            [rand.next_u32_bounded(self.allowed_scoring_rules.len() as u32) as usize];
        rules.tax_rule = self.allowed_tax_rules
            [rand.next_u32_bounded(self.allowed_tax_rules.len() as u32) as usize];
        rules.multi_stone_suicide_legal = self.allowed_multi_stone_suicide_legals
            [rand.next_u32_bounded(self.allowed_multi_stone_suicide_legals.len() as u32) as usize];
        rules.has_button = if rules.scoring_rule == ScoringRule::Area {
            self.allowed_buttons[rand.next_u32_bounded(self.allowed_buttons.len() as u32) as usize]
        } else {
            false
        };
        rules
    }

    /// Initialize a new game with randomized rules and board size.
    #[allow(clippy::too_many_arguments)]
    pub fn create_game(
        &self,
        board: &mut Board,
        pla: &mut Player,
        hist: &mut BoardHistory,
        extra_black_and_komi: &mut ExtraBlackAndKomi,
        initial_position: Option<&InitialPosition>,
        _play_settings: &PlaySettings,
        other_game_props: &mut OtherGameProperties,
        _start_pos_sample: Option<&PositionSample>,
    ) {
        let _lock = self
            .create_game_mutex
            .lock()
            .expect("createGame mutex poisoned");
        self.create_game_shared_unsynchronized(
            board,
            pla,
            hist,
            extra_black_and_komi,
            initial_position,
            other_game_props,
        );
        if self.no_result_stdev != 0.0 || self.draw_rand_radius != 0.0 {
            panic!(
                "GameInitializer::createGame called in a mode that doesn't support specifying noResultStdev or drawRandRadius"
            );
        }
    }

    /// Initialize a new game and randomize noise-sensitive search parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn create_game_with_params(
        &self,
        board: &mut Board,
        pla: &mut Player,
        hist: &mut BoardHistory,
        extra_black_and_komi: &mut ExtraBlackAndKomi,
        params: &mut SearchParams,
        initial_position: Option<&InitialPosition>,
        play_settings: &PlaySettings,
        other_game_props: &mut OtherGameProperties,
        start_pos_sample: Option<&PositionSample>,
    ) {
        let _lock = self
            .create_game_mutex
            .lock()
            .expect("createGame mutex poisoned");
        self.create_game_shared_unsynchronized(
            board,
            pla,
            hist,
            extra_black_and_komi,
            initial_position,
            other_game_props,
        );

        let mut rand = self.rand.lock().expect("rand mutex poisoned");
        if self.no_result_stdev > 1e-30 {
            let mean = params.no_result_utility_for_white;
            loop {
                params.no_result_utility_for_white =
                    mean + self.no_result_stdev * rand.next_gaussian_truncated(3.0);
                if (-1.0..=1.0).contains(&params.no_result_utility_for_white) {
                    break;
                }
            }
        }
        if self.draw_rand_radius > 1e-30 {
            let mean = params.draw_equivalent_wins_for_white;
            if !(0.0..=1.0).contains(&mean) {
                panic!(
                    "GameInitializer: params.drawEquivalentWinsForWhite not within [0,1]: {}",
                    mean
                );
            }
            loop {
                params.draw_equivalent_wins_for_white =
                    mean + self.draw_rand_radius * (rand.next_double() * 2.0 - 1.0);
                if (0.0..=1.0).contains(&params.draw_equivalent_wins_for_white) {
                    break;
                }
            }
        }

        // Avoid unused parameter warning while keeping the full signature.
        let _ = (play_settings, start_pos_sample);
    }

    fn create_game_shared_unsynchronized(
        &self,
        board: &mut Board,
        pla: &mut Player,
        hist: &mut BoardHistory,
        extra_black_and_komi: &mut ExtraBlackAndKomi,
        initial_position: Option<&InitialPosition>,
        other_game_props: &mut OtherGameProperties,
    ) {
        let mut rand = self.rand.lock().expect("rand mutex poisoned");

        if let Some(initial) = initial_position {
            *board = initial.board.clone();
            *hist = initial.hist.clone();
            *pla = initial.pla;
            *extra_black_and_komi = ExtraBlackAndKomi {
                allow_integer: true,
                ..ExtraBlackAndKomi::default()
            };
            other_game_props.is_sgf_pos = false;
            other_game_props.is_hint_pos = false;
            other_game_props.allow_policy_init = false;
            other_game_props.is_fork = true;
            other_game_props.is_hint_fork = initial.is_hint_fork;
            other_game_props.hint_loc = NULL_LOC;
            other_game_props.hint_turn = if initial.is_hint_fork {
                hist.move_history.len() as i32
            } else {
                -1
            };
            other_game_props.training_weight = initial.training_weight;
            extra_black_and_komi.make_game_fair = rand.next_bool(self.fork_compensate_komi_prob);
            extra_black_and_komi.make_game_fair_for_empty_board = false;
            extra_black_and_komi.interp_zero = false;
            return;
        }

        let b_size_idx = rand.next_u32_from_probs(&self.allowed_b_size_rel_probs) as usize;
        let rules = self.create_rules_unsynchronized(&mut rand);
        let (x_size, y_size) = self.allowed_b_sizes[b_size_idx];
        *board = Board::new(x_size, y_size);
        *pla = P_BLACK;
        *hist = BoardHistory::new(board.clone(), *pla, rules, 0);
        *extra_black_and_komi = ExtraBlackAndKomi {
            komi_mean: self.komi_mean,
            komi_stdev: self.komi_stdev,
            allow_integer: true,
            ..ExtraBlackAndKomi::default()
        };

        other_game_props.is_sgf_pos = false;
        other_game_props.is_hint_pos = false;
        other_game_props.allow_policy_init = true;
        other_game_props.is_fork = false;
        other_game_props.is_hint_fork = false;
        other_game_props.hint_loc = NULL_LOC;
        other_game_props.hint_turn = -1;
        other_game_props.training_weight = 1.0;
        extra_black_and_komi.make_game_fair = false;
        extra_black_and_komi.make_game_fair_for_empty_board = false;
        extra_black_and_komi.interp_zero = false;
    }
}

/// Specification for one bot in a pairing.
///
/// Mirrors `MatchPairer::BotSpec` in `cpp/program/play.h`.
#[derive(Clone)]
pub struct BotSpec<'a> {
    pub bot_idx: i32,
    pub bot_name: String,
    pub nn_eval: Option<&'a NnEvaluator>,
    pub base_params: SearchParams,
}

/// Object generating evenly distributed pairings between bots.
///
/// Mirrors `MatchPairer` in `cpp/program/play.h` and `cpp/program/play.cpp`.
pub struct MatchPairer<'a> {
    num_bots: i32,
    bot_names: Vec<String>,
    nn_evals: Vec<Option<*mut NnEvaluator>>,
    base_paramss: Vec<SearchParams>,
    matchups_per_round: Vec<(i32, i32)>,
    next_matchups: Vec<(i32, i32)>,
    rand: Rand,
    num_games_started_so_far: i64,
    num_games_total: i64,
    log_games_every: i64,
    get_matchup_mutex: Mutex<()>,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> MatchPairer<'a> {
    /// Create a match pairer from config and bot specifications.
    pub fn new(
        cfg: &ConfigParser,
        num_bots: i32,
        bot_names: Vec<String>,
        nn_evals: Vec<Option<&'a NnEvaluator>>,
        base_paramss: Vec<SearchParams>,
    ) -> Result<Self, StringError> {
        let to_string_err = |e: kata_core::config::ConfigError| StringError::new(e.to_string());

        if num_bots <= 0 {
            return Err(StringError::new(
                "MatchPairer: numBots must be positive".to_string(),
            ));
        }
        if bot_names.len() as i32 != num_bots
            || nn_evals.len() as i32 != num_bots
            || base_paramss.len() as i32 != num_bots
        {
            return Err(StringError::new(
                "MatchPairer: bot specification vectors must have length numBots".to_string(),
            ));
        }

        let matchups_per_round = cfg
            .get_non_negative_int_dashed_pairs("matchupsPerRound", 0, num_bots - 1)
            .map_err(to_string_err)?;
        let num_games_total = cfg
            .get_int64("numGamesTotal", 1, i64::MAX)
            .map_err(to_string_err)?;
        let log_games_every = cfg
            .get_int64("logGamesEvery", 1, 1_000_000)
            .map_err(to_string_err)?;

        if matchups_per_round.is_empty() {
            return Err(StringError::new(
                "MatchPairer: no matchups specified".to_string(),
            ));
        }
        if matchups_per_round.len() > 0xFFFFFF {
            return Err(StringError::new(
                "MatchPairer: too many matchups".to_string(),
            ));
        }

        Ok(Self {
            num_bots,
            bot_names,
            nn_evals: nn_evals
                .into_iter()
                .map(|e| e.map(|p| p as *const NnEvaluator as *mut NnEvaluator))
                .collect(),
            base_paramss,
            matchups_per_round,
            next_matchups: Vec::new(),
            rand: Rand::new(),
            num_games_started_so_far: 0,
            num_games_total,
            log_games_every,
            get_matchup_mutex: Mutex::new(()),
            _marker: std::marker::PhantomData,
        })
    }

    /// Total number of games that this pairer will generate.
    pub fn get_num_games_total_to_generate(&self) -> i64 {
        self.num_games_total
    }

    /// Get the next matchup, logging progress along the way.
    pub fn get_matchup(&mut self, logger: &Logger) -> Option<(BotSpec<'a>, BotSpec<'a>)> {
        let game_number = {
            let _lock = self
                .get_matchup_mutex
                .lock()
                .expect("getMatchup mutex poisoned");

            if self.num_games_started_so_far >= self.num_games_total {
                return None;
            }

            self.num_games_started_so_far += 1;
            self.num_games_started_so_far
        };

        if game_number % self.log_games_every == 0 {
            logger.write(&format!("Started {} games", game_number));
        }
        let log_nn_every = (self.log_games_every * 100).max(1000);
        if game_number % log_nn_every == 0 {
            for nn_eval in self.nn_evals.iter().flatten() {
                let nn_eval = unsafe { &**nn_eval };
                logger.write(nn_eval.model_file_name());
                logger.write(&format!("NN rows: {}", nn_eval.num_rows_processed()));
                logger.write(&format!("NN batches: {}", nn_eval.num_batches_processed()));
                logger.write(&format!(
                    "NN avg batch size: {}",
                    nn_eval.average_processed_batch_size()
                ));
            }
        }

        let matchup = self.get_matchup_pair_unsynchronized();

        let bot_b = BotSpec {
            bot_idx: matchup.0,
            bot_name: self.bot_names[matchup.0 as usize].clone(),
            nn_eval: self.nn_evals[matchup.0 as usize].map(|p| unsafe { &*p }),
            base_params: self.base_paramss[matchup.0 as usize].clone(),
        };
        let bot_w = BotSpec {
            bot_idx: matchup.1,
            bot_name: self.bot_names[matchup.1 as usize].clone(),
            nn_eval: self.nn_evals[matchup.1 as usize].map(|p| unsafe { &*p }),
            base_params: self.base_paramss[matchup.1 as usize].clone(),
        };

        Some((bot_b, bot_w))
    }

    fn get_matchup_pair_unsynchronized(&mut self) -> (i32, i32) {
        if self.next_matchups.is_empty() {
            if self.num_bots == 0 {
                panic!("MatchPairer::getMatchupPairUnsynchronized: no bots to match up");
            }
            self.next_matchups.clear();
            self.next_matchups.extend(&self.matchups_per_round);
            self.rand.shuffle(&mut self.next_matchups);
        }

        self.next_matchups
            .pop()
            .expect("nextMatchups should be non-empty")
    }
}

/// MatchPairer stores pointers to `NnEvaluator`s that are loaded once and then
/// accessed concurrently by multiple game threads. The evaluator itself is
/// shared across threads in other commands (e.g. selfplay), so it is safe to
/// send and share the pairer as long as the internal mutex is held when
/// mutating pairing state.
unsafe impl<'a> Send for MatchPairer<'a> {}
unsafe impl<'a> Sync for MatchPairer<'a> {}

/// Class wrapping parameters needed to run a full game.
///
/// Mirrors `GameRunner` in `cpp/program/play.h` and `cpp/program/play.cpp`.
#[allow(dead_code)]
pub struct GameRunner {
    log_search_info: bool,
    log_moves: bool,
    max_moves_per_game: i32,
    clear_bot_before_search: bool,
    play_settings: PlaySettings,
    game_init: Option<Box<GameInitializer>>,
}

impl GameRunner {
    /// Create a game runner with a freshly seeded initializer.
    pub fn new(
        cfg: &ConfigParser,
        play_settings: PlaySettings,
        logger: &Logger,
    ) -> Result<Self, StringError> {
        let to_string_err = |e: kata_core::config::ConfigError| StringError::new(e.to_string());
        let log_search_info = cfg.get_bool("logSearchInfo").map_err(to_string_err)?;
        let log_moves = cfg.get_bool("logMoves").map_err(to_string_err)?;
        let max_moves_per_game = cfg
            .get_int("maxMovesPerGame", 0, 1 << 30)
            .map_err(to_string_err)?;
        let clear_bot_before_search = if cfg.contains("clearBotBeforeSearch") {
            cfg.get_bool("clearBotBeforeSearch")
                .map_err(to_string_err)?
        } else {
            false
        };
        let game_init = Some(Box::new(GameInitializer::new(cfg, logger)?));

        Ok(Self {
            log_search_info,
            log_moves,
            max_moves_per_game,
            clear_bot_before_search,
            play_settings,
            game_init,
        })
    }

    /// Create a game runner with an explicit initializer seed.
    pub fn new_with_seed(
        cfg: &ConfigParser,
        game_init_rand_seed: &str,
        play_settings: PlaySettings,
        logger: &Logger,
    ) -> Result<Self, StringError> {
        let to_string_err = |e: kata_core::config::ConfigError| StringError::new(e.to_string());
        let log_search_info = cfg.get_bool("logSearchInfo").map_err(to_string_err)?;
        let log_moves = cfg.get_bool("logMoves").map_err(to_string_err)?;
        let max_moves_per_game = cfg
            .get_int("maxMovesPerGame", 0, 1 << 30)
            .map_err(to_string_err)?;
        let clear_bot_before_search = if cfg.contains("clearBotBeforeSearch") {
            cfg.get_bool("clearBotBeforeSearch")
                .map_err(to_string_err)?
        } else {
            false
        };
        let game_init = Some(Box::new(GameInitializer::new_with_seed(
            cfg,
            logger,
            game_init_rand_seed,
        )?));

        Ok(Self {
            log_search_info,
            log_moves,
            max_moves_per_game,
            clear_bot_before_search,
            play_settings,
            game_init,
        })
    }

    /// Run a full game between the two specified bots.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn run_game<'a>(
        &self,
        seed: &str,
        bot_spec_b: &BotSpec<'a>,
        bot_spec_w: &BotSpec<'a>,
        fork_data: Option<&ForkData>,
        start_pos_sample: Option<&PositionSample>,
        logger: &'a Logger,
        should_stop: Box<dyn Fn() -> bool + Send + Sync>,
        should_pause: Option<&WaitableFlag>,
        check_for_new_nn_eval: Box<dyn Fn() -> Option<*mut NnEvaluator> + Send + Sync>,
        after_initialization: Box<dyn Fn(&BotSpec<'a>, *mut Search<'a>) + Send + Sync>,
        on_each_move: Box<
            dyn Fn(&Board, &BoardHistory, Player, Loc, &[f64], &[f64], &[f64], *const Search<'a>)
                + Send
                + Sync,
        >,
    ) -> Option<FinishedGameData> {
        let mut bot_spec_b = bot_spec_b.clone();
        let mut bot_spec_w = bot_spec_w.clone();

        let mut game_rand = Rand::new_from_seed(&format!("{}:forGameRand", seed));

        let game_init = self
            .game_init
            .as_ref()
            .expect("GameRunner used without a GameInitializer");

        let mut initial_position: Option<InitialPosition> = None;
        let mut used_seki_fork_hack_position = false;
        if let Some(fork_data) = fork_data {
            initial_position = fork_data.get(&mut game_rand);

            if initial_position.is_none()
                && self.play_settings.seki_fork_hack_prob > 0.0
                && game_rand.next_bool(self.play_settings.seki_fork_hack_prob)
            {
                initial_position = fork_data.get_seki(&mut game_rand);
                if initial_position.is_some() {
                    used_seki_fork_hack_position = true;
                }
            }
        }

        let mut board = Board::default();
        let mut pla = C_EMPTY;
        let mut hist = BoardHistory::default();
        let mut extra_black_and_komi = ExtraBlackAndKomi::default();
        let mut other_game_props = OtherGameProperties::default();

        if self.play_settings.for_self_play {
            assert_eq!(bot_spec_b.bot_idx, bot_spec_w.bot_idx);
            let mut params = bot_spec_b.base_params.clone();
            game_init.create_game_with_params(
                &mut board,
                &mut pla,
                &mut hist,
                &mut extra_black_and_komi,
                &mut params,
                initial_position.as_ref(),
                &self.play_settings,
                &mut other_game_props,
                start_pos_sample,
            );
            bot_spec_b.base_params = params.clone();
            bot_spec_w.base_params = params;
        } else {
            game_init.create_game(
                &mut board,
                &mut pla,
                &mut hist,
                &mut extra_black_and_komi,
                initial_position.as_ref(),
                &self.play_settings,
                &mut other_game_props,
                start_pos_sample,
            );

            if let Some(nn_eval) = bot_spec_b.nn_eval {
                let mut supported = false;
                nn_eval.supported_rules(hist.rules, &mut supported);
                if !supported {
                    logger.write(&format!(
                        "WARNING: Match is running bot on rules that it does not support: {}",
                        bot_spec_b.bot_name
                    ));
                }
            }
            if let Some(nn_eval) = bot_spec_w.nn_eval {
                let mut supported = false;
                nn_eval.supported_rules(hist.rules, &mut supported);
                if !supported {
                    logger.write(&format!(
                        "WARNING: Match is running bot on rules that it does not support: {}",
                        bot_spec_w.bot_name
                    ));
                }
            }
        }

        let mut clear_bot_before_search_this_game = self.clear_bot_before_search;
        if bot_spec_b.bot_idx == bot_spec_w.bot_idx {
            clear_bot_before_search_this_game = true;
        }

        let do_end_game_if_all_pass_alive = if self.play_settings.for_self_play {
            game_rand.next_bool(0.98)
        } else {
            true
        };

        let nn_eval_b = bot_spec_b
            .nn_eval
            .expect("GameRunner::run_game requires bot B to have an NNEvaluator");
        let bot_b: *mut Search = Box::into_raw(Box::new(Search::new(
            bot_spec_b.base_params.clone(),
            nn_eval_b,
            logger,
            seed,
        )));

        let bot_w: *mut Search = if bot_spec_b.bot_idx == bot_spec_w.bot_idx {
            bot_b
        } else {
            let nn_eval_w = bot_spec_w
                .nn_eval
                .expect("GameRunner::run_game requires bot W to have an NNEvaluator");
            Box::into_raw(Box::new(Search::new(
                bot_spec_w.base_params.clone(),
                nn_eval_w,
                logger,
                &format!("{}@W", seed),
            )))
        };

        if bot_spec_b.bot_idx == bot_spec_w.bot_idx {
            after_initialization(&bot_spec_b, bot_b);
        } else {
            after_initialization(&bot_spec_b, bot_b);
            after_initialization(&bot_spec_w, bot_w);
        }

        let mut finished_game_data = run_game_with_bots(
            &board,
            pla,
            &hist,
            extra_black_and_komi,
            &bot_spec_b,
            &bot_spec_w,
            bot_b,
            bot_w,
            do_end_game_if_all_pass_alive,
            clear_bot_before_search_this_game,
            logger,
            self.log_search_info,
            self.log_moves,
            self.max_moves_per_game,
            &*should_stop,
            should_pause,
            &self.play_settings,
            &other_game_props,
            &mut game_rand,
            &*check_for_new_nn_eval,
            &*on_each_move,
        );

        if let Some(ref mut game_data) = finished_game_data {
            if initial_position.is_some() {
                game_data.used_initial_position = 1;
            }

            if should_stop() {
                if bot_w != bot_b {
                    unsafe {
                        drop(Box::from_raw(bot_w));
                    }
                }
                unsafe {
                    drop(Box::from_raw(bot_b));
                }
                return None;
            }

            let empty_fork_data = ForkData::new();
            let fork_data_ref = fork_data.unwrap_or(&empty_fork_data);
            maybe_fork_game(
                game_data,
                fork_data_ref,
                &self.play_settings,
                &mut game_rand,
                bot_b,
            );
            if !used_seki_fork_hack_position {
                maybe_seki_fork_game(
                    game_data,
                    fork_data_ref,
                    &self.play_settings,
                    Some(game_init.as_ref()),
                    &mut game_rand,
                );
            }
            maybe_hint_fork_game(game_data, fork_data_ref, &other_game_props);
        }

        if bot_w != bot_b {
            unsafe {
                drop(Box::from_raw(bot_w));
            }
        }
        unsafe {
            drop(Box::from_raw(bot_b));
        }

        finished_game_data
    }

    /// Return the owned game initializer, if any.
    pub fn get_game_initializer(&self) -> Option<&GameInitializer> {
        self.game_init.as_deref()
    }
}

impl Drop for GameRunner {
    fn drop(&mut self) {
        // The boxed GameInitializer is dropped automatically; this hook mirrors
        // the explicit destructor in the C++ header.
    }
}

/// Run a game between two bots, creating the `Search` instances internally.
///
/// Mirrors the bot-creating overload of `Play::runGame` in `cpp/program/play.cpp`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn run_game<'a>(
    start_board: &Board,
    pla: Player,
    start_hist: &BoardHistory,
    extra_black_and_komi: ExtraBlackAndKomi,
    bot_spec_b: &BotSpec<'a>,
    bot_spec_w: &BotSpec<'a>,
    search_rand_seed: &str,
    do_end_game_if_all_pass_alive: bool,
    clear_bot_before_search: bool,
    logger: &'a Logger,
    log_search_info: bool,
    log_moves: bool,
    max_moves_per_game: i32,
    should_stop: &(dyn Fn() -> bool + Send + Sync),
    should_pause: Option<&WaitableFlag>,
    play_settings: &PlaySettings,
    other_game_props: &OtherGameProperties,
    game_rand: &mut Rand,
    check_for_new_nn_eval: &(dyn Fn() -> Option<*mut NnEvaluator> + Send + Sync),
    on_each_move: Box<
        dyn Fn(&Board, &BoardHistory, Player, Loc, &[f64], &[f64], &[f64], *const Search<'a>)
            + Send
            + Sync,
    >,
) -> Option<FinishedGameData> {
    let nn_eval_b = bot_spec_b.nn_eval?;
    let bot_b: *mut Search = Box::into_raw(Box::new(Search::new(
        bot_spec_b.base_params.clone(),
        nn_eval_b,
        logger,
        search_rand_seed,
    )));

    let bot_w: *mut Search = if bot_spec_b.bot_idx == bot_spec_w.bot_idx {
        bot_b
    } else {
        let nn_eval_w = bot_spec_w.nn_eval?;
        Box::into_raw(Box::new(Search::new(
            bot_spec_w.base_params.clone(),
            nn_eval_w,
            logger,
            &format!("{}@W", search_rand_seed),
        )))
    };

    let result = run_game_with_bots(
        start_board,
        pla,
        start_hist,
        extra_black_and_komi,
        bot_spec_b,
        bot_spec_w,
        bot_b,
        bot_w,
        do_end_game_if_all_pass_alive,
        clear_bot_before_search,
        logger,
        log_search_info,
        log_moves,
        max_moves_per_game,
        &*should_stop,
        should_pause,
        play_settings,
        other_game_props,
        game_rand,
        &*check_for_new_nn_eval,
        &*on_each_move,
    );

    if bot_w != bot_b {
        unsafe {
            drop(Box::from_raw(bot_w));
        }
    }
    unsafe {
        drop(Box::from_raw(bot_b));
    }

    result
}

/// Sample an integer uniformly from `[min, max]` (inclusive).
fn rand_next_int(rand: &mut Rand, min: i32, max: i32) -> i32 {
    assert!(min <= max);
    min + rand.next_u32_bounded((max - min + 1) as u32) as i32
}

/// Per-move search limits computed from `PlaySettings` and game history.
#[derive(Debug, Clone, Copy)]
struct SearchLimitsThisMove {
    do_alter_visits_playouts: bool,
    num_alter_visits: i64,
    num_alter_playouts: i64,
    clear_bot_before_search_this_move: bool,
    remove_root_noise: bool,
    target_weight: f32,
    playout_doubling_advantage: f64,
    playout_doubling_advantage_pla: Player,
    hint_loc: Loc,
}

fn get_search_limits_this_move(
    to_move_bot: *const Search,
    pla: Player,
    play_settings: &PlaySettings,
    game_rand: &mut Rand,
    historical_mcts_win_loss_values: &[f64],
    clear_bot_before_search: bool,
    other_game_props: &OtherGameProperties,
) -> SearchLimitsThisMove {
    let bot = unsafe { &*to_move_bot };
    let mut do_alter = false;
    let mut num_alter_visits = bot.search_params.max_visits;
    let mut num_alter_playouts = bot.search_params.max_playouts;
    let mut clear_bot_before_search_this_move = clear_bot_before_search;
    let mut remove_root_noise = false;
    let mut target_weight = 1.0f32;
    let mut playout_doubling_advantage = 0.0;
    let mut playout_doubling_advantage_pla = C_EMPTY;
    let mut hint_loc = NULL_LOC;
    let mut cheap_search_prob = play_settings.cheap_search_prob;

    let hist = bot.get_root_hist();
    if other_game_props.hint_loc != NULL_LOC
        && other_game_props.hint_turn == hist.move_history.len() as i32
        && other_game_props.hint_pos_hash == bot.get_root_board().pos_hash
    {
        hint_loc = other_game_props.hint_loc;
        do_alter = true;
        let cap = (1i64 << 50) as f64;
        num_alter_visits = (cap.min(num_alter_visits as f64 * 4.0)).ceil() as i64;
        num_alter_playouts = (cap.min(num_alter_playouts as f64 * 4.0)).ceil() as i64;
    }

    if (other_game_props.hint_loc != NULL_LOC || other_game_props.is_hint_fork)
        && other_game_props.hint_turn + 6 > hist.move_history.len() as i32
    {
        cheap_search_prob *= 0.5;
    }

    if hint_loc == NULL_LOC && cheap_search_prob > 0.0 && game_rand.next_bool(cheap_search_prob) {
        assert!(play_settings.cheap_search_visits > 0);
        assert!(
            play_settings.cheap_search_visits as i64 <= bot.search_params.max_visits
                && play_settings.cheap_search_visits as i64 <= bot.search_params.max_playouts
        );
        do_alter = true;
        num_alter_visits = num_alter_visits.min(play_settings.cheap_search_visits as i64);
        num_alter_playouts = num_alter_playouts.min(play_settings.cheap_search_visits as i64);
        target_weight *= play_settings.cheap_search_target_weight;
        if play_settings.cheap_search_target_weight <= 0.0 {
            clear_bot_before_search_this_move = false;
            remove_root_noise = true;
        }
    } else if hint_loc == NULL_LOC && play_settings.reduce_visits {
        assert!(play_settings.reduced_visits_min > 0);
        assert!(
            play_settings.reduced_visits_min as i64 <= bot.search_params.max_visits
                && play_settings.reduced_visits_min as i64 <= bot.search_params.max_playouts
        );

        if historical_mcts_win_loss_values.len()
            >= play_settings.reduce_visits_threshold_lookback as usize
        {
            let mut min_win_loss_value: f64 = 1e20;
            let mut max_win_loss_value: f64 = -1e20;
            for j in 0..play_settings.reduce_visits_threshold_lookback as usize {
                let idx = historical_mcts_win_loss_values.len() - 1 - j;
                let win_loss_value = historical_mcts_win_loss_values[idx];
                min_win_loss_value = min_win_loss_value.min(win_loss_value);
                max_win_loss_value = max_win_loss_value.max(win_loss_value);
            }
            assert!(play_settings.reduce_visits_threshold >= 0.0);
            let signed_most_extreme = min_win_loss_value.max(-max_win_loss_value);
            assert!(signed_most_extreme <= 1.000001);
            let signed_most_extreme = signed_most_extreme.min(1.0);
            let amount_through = signed_most_extreme - play_settings.reduce_visits_threshold;
            if amount_through > 0.0 {
                let proportion_through =
                    amount_through / (1.0 - play_settings.reduce_visits_threshold);
                assert!((0.0..=1.0).contains(&proportion_through));
                let visit_reduction_prop = proportion_through * proportion_through;
                do_alter = true;
                num_alter_visits = (num_alter_visits as f64
                    + visit_reduction_prop
                        * (play_settings.reduced_visits_min as f64 - num_alter_visits as f64))
                    .round() as i64;
                num_alter_playouts = (num_alter_playouts as f64
                    + visit_reduction_prop
                        * (play_settings.reduced_visits_min as f64 - num_alter_playouts as f64))
                    .round() as i64;
                target_weight = (target_weight as f64
                    + visit_reduction_prop
                        * (play_settings.reduced_visits_weight as f64 - target_weight as f64))
                    as f32;
                num_alter_visits = num_alter_visits.max(play_settings.reduced_visits_min as i64);
                num_alter_playouts =
                    num_alter_playouts.max(play_settings.reduced_visits_min as i64);
            }
        }
    }

    if other_game_props.playout_doubling_advantage != 0.0
        && other_game_props.playout_doubling_advantage_pla != C_EMPTY
    {
        assert!(
            pla == other_game_props.playout_doubling_advantage_pla
                || get_opp(pla) == other_game_props.playout_doubling_advantage_pla
        );
        playout_doubling_advantage = other_game_props.playout_doubling_advantage;
        playout_doubling_advantage_pla = other_game_props.playout_doubling_advantage_pla;

        let factor = 2.0f64.powf(other_game_props.playout_doubling_advantage);
        let factor = if pla == other_game_props.playout_doubling_advantage_pla {
            2.0 * (factor / (factor + 1.0))
        } else {
            2.0 * (1.0 / (factor + 1.0))
        };

        do_alter = true;
        clear_bot_before_search_this_move = true;
        num_alter_visits = (num_alter_visits as f64 * factor).round() as i64;
        num_alter_playouts = (num_alter_playouts as f64 * factor).round() as i64;

        if num_alter_visits < 5 {
            panic!("ERROR: asymmetric playout doubling resulted in fewer than 5 visits");
        }
        if num_alter_playouts < 5 {
            panic!("ERROR: asymmetric playout doubling resulted in fewer than 5 playouts");
        }
    }

    SearchLimitsThisMove {
        do_alter_visits_playouts: do_alter,
        num_alter_visits,
        num_alter_playouts,
        clear_bot_before_search_this_move,
        remove_root_noise,
        target_weight,
        playout_doubling_advantage,
        playout_doubling_advantage_pla,
        hint_loc,
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn run_bot_with_limits(
    to_move_bot: *mut Search,
    pla: Player,
    play_settings: &PlaySettings,
    limits: &SearchLimitsThisMove,
) -> Loc {
    let bot = &mut *to_move_bot;
    if limits.clear_bot_before_search_this_move {
        bot.clear_search();
    }

    let lcb = bot.search_params.use_lcb_for_selection;
    if play_settings.for_self_play {
        bot.search_params.use_lcb_for_selection = false;
    }

    let loc = if limits.do_alter_visits_playouts {
        assert!(limits.num_alter_visits > 0);
        assert!(limits.num_alter_playouts > 0);
        let old_params = bot.search_params.clone();

        bot.search_params.max_visits = limits.num_alter_visits;
        bot.search_params.max_playouts = limits.num_alter_playouts;
        if limits.remove_root_noise {
            bot.search_params.root_noise_enabled = false;
            bot.search_params.root_policy_temperature = 1.0;
            bot.search_params.root_policy_temperature_early = 1.0;
            bot.search_params.root_fpu_loss_prop = bot.search_params.fpu_loss_prop;
            bot.search_params.root_fpu_reduction_max = bot.search_params.fpu_reduction_max;
            bot.search_params.root_desired_per_child_visits_coeff = 0.0;
            bot.search_params.root_num_symmetries_to_sample = 1;
        }
        if limits.playout_doubling_advantage_pla != C_EMPTY {
            bot.search_params.playout_doubling_advantage_pla =
                limits.playout_doubling_advantage_pla;
            bot.search_params.playout_doubling_advantage = limits.playout_doubling_advantage;
        }

        if limits.clear_bot_before_search_this_move
            && bot.search_params.max_visits > 10
            && bot.search_params.max_playouts > 10
        {
            let old_max_visits = bot.search_params.max_visits;
            bot.search_params.max_visits = 10;
            bot.run_whole_search_and_get_move(pla);
            bot.search_params.max_visits = old_max_visits;
        }

        if limits.hint_loc != NULL_LOC {
            assert!(limits.clear_bot_before_search_this_move);
            bot.set_root_hint_loc(limits.hint_loc);
        }

        let loc = bot.run_whole_search_and_get_move(pla);
        bot.set_root_hint_loc(NULL_LOC);
        bot.search_params = old_params;
        loc
    } else {
        assert!(!limits.remove_root_noise);
        bot.run_whole_search_and_get_move(pla)
    };

    if play_settings.for_self_play {
        bot.search_params.use_lcb_for_selection = lcb;
    }

    loc
}

fn extract_value_targets(buf: &mut ValueTargets, to_move_bot: *const Search, node: &SearchNode) {
    let bot = unsafe { &*to_move_bot };
    let mut values = ReportedSearchValues::new();
    let success = bot.get_node_values(node, &mut values);
    assert!(success);
    buf.win = values.win_value as f32;
    buf.loss = values.loss_value as f32;
    buf.no_result = values.no_result_value as f32;
    buf.score = values.expected_score as f32;
}

fn extract_q_value_targets(
    buf: &mut Vec<QValueTargetMove>,
    to_move_bot: *const Search,
    node: &SearchNode,
) {
    let bot = unsafe { &*to_move_bot };
    let children = node.get_children();
    let num_children = children.iterate_and_count_children();
    buf.clear();
    for i in 0..num_children {
        let child_ptr: &SearchChildPointer = children.get(i);
        if let Some(child) = child_ptr.get_if_allocated() {
            let mut values = ReportedSearchValues::new();
            let success = bot.get_node_values(child, &mut values);
            if !success {
                continue;
            }
            if values.visits <= 0 {
                continue;
            }
            let move_loc = child_ptr.get_move_loc();
            buf.push(QValueTargetMove {
                loc: move_loc,
                win_loss: values.win_loss_value as f32,
                score: values.expected_score as f32,
                visits: values.visits,
            });
        }
    }
}

fn compute_nn_raw_stats(
    bot: *const Search,
    board: &Board,
    hist: &BoardHistory,
    pla: Player,
) -> NNRawStats {
    let bot_ref = unsafe { &*bot };
    let nn_eval = bot_ref
        .nn_evaluator
        .expect("computeNNRawStats called with a Search that has no NNEvaluator");
    let mut buf = NNResultBuf::new();
    let mut nn_input_params = MiscNNInputParams::default();
    nn_input_params.draw_equivalent_wins_for_white =
        bot_ref.search_params.draw_equivalent_wins_for_white;
    nn_eval.evaluate(board, hist, pla, &nn_input_params, &mut buf, false, false);
    let nn_output = buf.result.expect("NN evaluation produced no output");

    let white_win_loss = nn_output.white_win_prob - nn_output.white_loss_prob;
    let white_score_mean = nn_output.white_score_mean;
    let mut entropy = 0.0;
    let policy_size = (nn_output.nn_x_len * nn_output.nn_y_len) as usize;
    for pos in 0..policy_size {
        let prob = nn_output.policy_probs[pos];
        if prob >= 1e-30 {
            entropy -= prob as f64 * (prob as f64).ln();
        }
    }

    NNRawStats {
        white_win_loss: f64::from(white_win_loss),
        white_score_mean: f64::from(white_score_mean),
        policy_entropy: entropy,
    }
}

fn choose_random_forking_move(
    nn_output: &NNOutput,
    board: &Board,
    hist: &BoardHistory,
    pla: Player,
    game_rand: &mut Rand,
    ban_move: Loc,
) -> Loc {
    let r = game_rand.next_double();
    if r < 0.70 {
        choose_random_policy_move(nn_output, board, hist, pla, game_rand, 1.0, true, ban_move)
    } else if r < 0.95 {
        choose_random_policy_move(nn_output, board, hist, pla, game_rand, 2.0, true, ban_move)
    } else {
        choose_random_legal_move(board, hist, pla, game_rand, ban_move)
    }
}

fn record_tree_positions(
    game_data: &mut FinishedGameData,
    board: &Board,
    hist: &BoardHistory,
    pla: Player,
    to_move_bot: *const Search,
    min_visits_at_node: i64,
    record_tree_target_weight: f32,
    num_neural_net_changes_so_far: i32,
    locs_buf: &mut Vec<Loc>,
    play_selection_values_buf: &mut Vec<f64>,
    exclude_loc0: Loc,
    exclude_loc1: Loc,
) {
    let bot = unsafe { &*to_move_bot };
    debug_assert_eq!(bot.root_board.pos_hash, board.pos_hash);
    debug_assert_eq!(bot.root_history.move_history.len(), hist.move_history.len());
    debug_assert_eq!(bot.root_pla, pla);
    let root_node = bot
        .root_node
        .as_deref()
        .expect("record_tree_positions called with a Search that has no root node");

    const MAX_DEPTH: i32 = 5;
    record_tree_positions_rec(
        game_data,
        board,
        hist,
        pla,
        to_move_bot,
        root_node,
        0,
        MAX_DEPTH,
        true,
        true,
        min_visits_at_node,
        record_tree_target_weight,
        num_neural_net_changes_so_far,
        locs_buf,
        play_selection_values_buf,
        exclude_loc0,
        exclude_loc1,
    );
}

#[allow(clippy::too_many_arguments)]
fn record_tree_positions_rec(
    game_data: &mut FinishedGameData,
    board: &Board,
    hist: &BoardHistory,
    pla: Player,
    to_move_bot: *const Search,
    node: &SearchNode,
    depth: i32,
    max_depth: i32,
    pla_always_best: bool,
    opp_always_best: bool,
    min_visits_at_node: i64,
    record_tree_target_weight: f32,
    num_neural_net_changes_so_far: i32,
    locs_buf: &mut Vec<Loc>,
    play_selection_values_buf: &mut Vec<f64>,
    exclude_loc0: Loc,
    exclude_loc1: Loc,
) {
    let bot = unsafe { &*to_move_bot };
    let children = node.get_children();
    let num_children = children.iterate_and_count_children();
    if num_children == 0 {
        return;
    }

    let root_node_ptr = bot.root_node.as_deref().map(|r| r as *const SearchNode);
    if pla_always_best && Some(node as *const SearchNode) != root_node_ptr {
        let mut sp = SidePosition {
            board: board.clone(),
            hist: hist.clone(),
            pla,
            num_neural_net_changes_so_far,
            ..Default::default()
        };
        extract_policy_target(
            &mut sp.policy_target,
            to_move_bot,
            node as *const SearchNode,
            locs_buf,
            play_selection_values_buf,
        );
        extract_value_targets(&mut sp.white_value_targets, to_move_bot, node);
        extract_q_value_targets(&mut sp.white_q_value_targets.targets, to_move_bot, node);

        let mut policy_surprise = 0.0;
        let mut search_entropy = 0.0;
        let mut policy_entropy = 0.0;
        let success = bot.get_policy_surprise_and_entropy_for_node(
            &mut policy_surprise,
            &mut search_entropy,
            &mut policy_entropy,
            node,
        );
        assert!(success);

        sp.policy_surprise = policy_surprise;
        sp.search_entropy = search_entropy;
        sp.policy_entropy = policy_entropy;
        sp.nn_raw_stats = compute_nn_raw_stats(to_move_bot, board, hist, pla);
        sp.target_weight = record_tree_target_weight;
        sp.unreduced_num_visits = bot.get_root_visits();
        game_data.side_positions.push(sp);
    }

    if depth >= max_depth {
        return;
    }

    let mut best_child_idx = 0;
    let mut best_child_visits = 0;
    for i in 1..num_children {
        let child_ptr = children.get(i);
        if let Some(child) = child_ptr.get_if_allocated() {
            let visits = child.stats.visits.load(Ordering::Acquire);
            if visits > best_child_visits {
                best_child_visits = visits;
                best_child_idx = i;
            }
        }
    }

    for i in 0..num_children {
        let new_pla_always_best = opp_always_best;
        let new_opp_always_best = pla_always_best && i == best_child_idx;
        if !new_pla_always_best && !new_opp_always_best {
            continue;
        }

        let child_ptr = children.get(i);
        let Some(child) = child_ptr.get_if_allocated() else {
            continue;
        };
        let move_loc = child_ptr.get_move_loc();
        if move_loc == exclude_loc0 || move_loc == exclude_loc1 {
            continue;
        }

        let num_visits = child.stats.visits.load(Ordering::Acquire);
        if num_visits < min_visits_at_node {
            continue;
        }

        if hist.is_legal(board, move_loc, pla) {
            let mut board_copy = board.clone();
            let mut hist_copy = hist.clone();
            hist_copy.make_board_move_assume_legal(&mut board_copy, move_loc, pla);
            let next_pla = get_opp(pla);
            record_tree_positions_rec(
                game_data,
                &board_copy,
                &hist_copy,
                next_pla,
                to_move_bot,
                child,
                depth + 1,
                max_depth,
                new_pla_always_best,
                new_opp_always_best,
                min_visits_at_node,
                record_tree_target_weight,
                num_neural_net_changes_so_far,
                locs_buf,
                play_selection_values_buf,
                NULL_LOC,
                NULL_LOC,
            );
        }
    }
}

/// Run a game between two existing `Search` bots.
///
/// Mirrors the bot-accepting overload of `Play::runGame` in
/// `cpp/program/play.cpp`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn run_game_with_bots<'a>(
    start_board: &Board,
    start_pla: Player,
    start_hist: &BoardHistory,
    extra_black_and_komi: ExtraBlackAndKomi,
    bot_spec_b: &BotSpec<'a>,
    bot_spec_w: &BotSpec<'a>,
    bot_b: *mut Search<'a>,
    bot_w: *mut Search<'a>,
    do_end_game_if_all_pass_alive: bool,
    clear_bot_before_search: bool,
    logger: &'a Logger,
    log_search_info: bool,
    log_moves: bool,
    max_moves_per_game: i32,
    should_stop: &(dyn Fn() -> bool + Send + Sync),
    should_pause: Option<&WaitableFlag>,
    play_settings: &PlaySettings,
    other_game_props: &OtherGameProperties,
    game_rand: &mut Rand,
    check_for_new_nn_eval: &(dyn Fn() -> Option<*mut NnEvaluator> + Send + Sync),
    on_each_move: &(
         dyn Fn(&Board, &BoardHistory, Player, Loc, &[f64], &[f64], &[f64], *const Search<'a>)
             + Send
             + Sync
     ),
) -> Option<FinishedGameData> {
    assert!(
        !(extra_black_and_komi.make_game_fair
            && extra_black_and_komi.make_game_fair_for_empty_board)
    );
    assert!(!(play_settings.for_self_play && !clear_bot_before_search));

    let mut game_data = FinishedGameData::default();
    let mut board = start_board.clone();
    let mut hist = start_hist.clone();
    let mut pla = start_pla;

    if extra_black_and_komi.make_game_fair_for_empty_board {
        let b = Board::new(start_board.x_size, start_board.y_size);
        let mut make_fair_pla = P_BLACK;
        if play_settings.flip_komi_prob_when_no_compensate != 0.0
            && game_rand.next_bool(play_settings.flip_komi_prob_when_no_compensate)
        {
            make_fair_pla = P_WHITE;
        }
        let mut h = BoardHistory::new(
            b.clone(),
            make_fair_pla,
            start_hist.rules,
            start_hist.encore_phase,
        );
        set_komi_without_noise(&extra_black_and_komi, &mut h);
        adjust_komi_to_even(
            unsafe { &mut *bot_b },
            unsafe { &mut *bot_w },
            &b,
            &mut h,
            make_fair_pla,
            play_settings.compensate_komi_visits,
            other_game_props,
            game_rand,
        );
        let mut extra = extra_black_and_komi;
        extra.komi_mean = h.rules.komi as f32 / 2.0;
        set_komi_with_noise(&extra, &mut hist, game_rand);
    }

    if extra_black_and_komi.extra_black > 0 && !hist.is_game_finished {
        let extra_black_temperature = play_settings.handicap_temperature;
        assert!(extra_black_temperature > 0.0 && extra_black_temperature < 10.0);
        play_extra_black(
            unsafe { &mut *bot_b },
            extra_black_and_komi.extra_black,
            &mut board,
            &mut hist,
            extra_black_temperature,
            game_rand,
        );
        assert_eq!(hist.move_history.len(), 0);
    }

    if extra_black_and_komi.make_game_fair {
        set_komi_without_noise(&extra_black_and_komi, &mut hist);
        adjust_komi_to_even(
            unsafe { &mut *bot_b },
            unsafe { &mut *bot_w },
            &board,
            &mut hist,
            pla,
            play_settings.compensate_komi_visits,
            other_game_props,
            game_rand,
        );
        let mut extra = extra_black_and_komi;
        extra.komi_mean = hist.rules.komi as f32 / 2.0;
        set_komi_with_noise(&extra, &mut hist, game_rand);
    } else if (extra_black_and_komi.extra_black > 0 || other_game_props.is_fork)
        && play_settings.fancy_komi_varying
        && game_rand.next_bool(if extra_black_and_komi.extra_black > 0 {
            0.5
        } else {
            0.25
        })
    {
        let orig_komi = hist.rules.komi;
        set_komi_without_noise(&extra_black_and_komi, &mut hist);
        adjust_komi_to_even(
            unsafe { &mut *bot_b },
            unsafe { &mut *bot_w },
            &board,
            &mut hist,
            pla,
            play_settings.compensate_komi_visits,
            other_game_props,
            game_rand,
        );
        let new_komi = hist.rules.komi;
        let mut rand_komi = game_rand.next_double_range(
            f64::from(orig_komi.min(new_komi)),
            f64::from(orig_komi.max(new_komi)),
        );
        rand_komi +=
            0.75 * (board.x_size * board.y_size) as f64 * game_rand.next_gaussian_truncated(2.5);
        let mut extra = extra_black_and_komi;
        extra.komi_mean = rand_komi as f32;
        set_komi_with_noise(&extra, &mut hist, game_rand);
    }

    {
        let bot_b_ref = unsafe { &*bot_b };
        let bot_w_ref = unsafe { &*bot_w };
        if play_settings.fancy_komi_varying
            && bot_b_ref
                .nn_evaluator
                .map_or(false, |e| e.is_neural_net_less())
            && (bot_w == bot_b
                || bot_w_ref
                    .nn_evaluator
                    .map_or(false, |e| e.is_neural_net_less()))
        {
            let rand_komi = hist.rules.komi as f64
                + 1.5
                    * (board.x_size * board.y_size) as f64
                    * game_rand.next_gaussian_truncated(2.5);
            let mut extra = extra_black_and_komi;
            extra.komi_mean = rand_komi as f32;
            set_komi_with_noise(&extra, &mut hist, game_rand);
        }
    }

    game_data.b_name = bot_spec_b.bot_name.clone();
    game_data.w_name = bot_spec_w.bot_name.clone();
    game_data.b_idx = bot_spec_b.bot_idx;
    game_data.w_idx = bot_spec_w.bot_idx;
    game_data.game_hash.hash0 = game_rand.next_u64();
    game_data.game_hash.hash1 = game_rand.next_u64();
    game_data.draw_equivalent_wins_for_white =
        bot_spec_b.base_params.draw_equivalent_wins_for_white;
    game_data.playout_doubling_advantage_pla = other_game_props.playout_doubling_advantage_pla;
    game_data.playout_doubling_advantage = other_game_props.playout_doubling_advantage;
    game_data.num_extra_black = extra_black_and_komi.extra_black;
    game_data.handicap_for_sgf = extra_black_and_komi.extra_black;
    game_data.mode = kata_data::training::mode::NORMAL;
    game_data.began_in_encore_phase = 0;
    game_data.used_initial_position = 0;

    if extra_black_and_komi.extra_black > 0 {
        game_data.mode = kata_data::training::mode::HANDICAP;
    }
    if game_data.playout_doubling_advantage != 0.0 {
        game_data.mode = kata_data::training::mode::ASYM;
    }
    if other_game_props.is_sgf_pos {
        game_data.mode = kata_data::training::mode::SGF_POS;
    }
    if other_game_props.is_hint_pos {
        game_data.mode = kata_data::training::mode::HINT_POS;
    }
    if other_game_props.is_hint_fork {
        game_data.mode = kata_data::training::mode::HINT_FORK;
    } else if other_game_props.is_fork {
        game_data.mode = kata_data::training::mode::FORK;
    }

    let record_full_data = play_settings.for_self_play;

    if play_settings.init_games_with_policy
        && other_game_props.allow_policy_init
        && !hist.is_game_finished
    {
        let proportion_of_board_area = if other_game_props.is_sgf_pos {
            play_settings.start_poses_policy_init_area_prop
        } else {
            play_settings.policy_init_area_prop
        };
        if proportion_of_board_area > 0.0 {
            let old_komi = hist.rules.komi as f32 / 2.0;
            set_komi_with_noise(&extra_black_and_komi, &mut hist, game_rand);
            let policy_init_gamma_shape = play_settings.policy_init_gamma_shape;
            let temperature = play_settings.policy_init_area_temperature;
            assert!(temperature > 0.0 && temperature < 10.0);
            initialize_game_using_policy(
                unsafe { &mut *bot_b },
                unsafe { &mut *bot_w },
                &mut board,
                &mut hist,
                &mut pla,
                game_rand,
                do_end_game_if_all_pass_alive,
                proportion_of_board_area,
                policy_init_gamma_shape,
                temperature,
            );
            hist.set_komi(old_komi);

            let mut should_compensate = play_settings.compensate_after_policy_init_prob > 0.0
                && game_rand.next_bool(play_settings.compensate_after_policy_init_prob);
            if game_data.mode != kata_data::training::mode::NORMAL {
                should_compensate = extra_black_and_komi.make_game_fair;
            }
            if should_compensate {
                adjust_komi_to_even(
                    unsafe { &mut *bot_b },
                    unsafe { &mut *bot_w },
                    &board,
                    &mut hist,
                    pla,
                    play_settings.compensate_komi_visits,
                    other_game_props,
                    game_rand,
                );
                let mut extra = extra_black_and_komi;
                extra.komi_mean = hist.rules.komi as f32 / 2.0;
                set_komi_with_noise(&extra, &mut hist, game_rand);
            }
        }
    }

    if play_settings.for_self_play
        && !other_game_props.is_hint_pos
        && hist.rules.scoring_rule == ScoringRule::Territory
        && hist.encore_phase == 0
        && game_rand.next_bool(0.04)
        && !hist.is_game_finished
    {
        let proportion_of_board_area = 0.25;
        let policy_init_gamma_shape = 1.0 * 0.8 + play_settings.policy_init_gamma_shape * 0.2;
        let temperature = 2.0 / 3.0;
        initialize_game_using_policy(
            unsafe { &mut *bot_b },
            unsafe { &mut *bot_w },
            &mut board,
            &mut hist,
            &mut pla,
            game_rand,
            do_end_game_if_all_pass_alive,
            proportion_of_board_area,
            policy_init_gamma_shape,
            temperature,
        );

        if !hist.is_game_finished {
            adjust_komi_to_even(
                unsafe { &mut *bot_b },
                unsafe { &mut *bot_w },
                &board,
                &mut hist,
                pla,
                play_settings.compensate_komi_visits,
                other_game_props,
                game_rand,
            );
            let mut extra = extra_black_and_komi;
            extra.komi_mean = hist.rules.komi as f32 / 2.0;
            set_komi_with_noise(&extra, &mut hist, game_rand);

            let encore_phase = rand_next_int(game_rand, 1, 2);
            board.clear_simple_ko_loc();
            hist.clear(board.clone(), pla, hist.rules, encore_phase);
            game_data.mode = kata_data::training::mode::CLEANUP_TRAINING;
            game_data.began_in_encore_phase = encore_phase;
            game_data.used_initial_position = 0;
        }
    }

    game_data.start_board = board.clone();
    game_data.start_hist = hist.clone();
    game_data.start_pla = pla;

    unsafe {
        (*bot_b).set_position(pla, &board, &hist);
        if bot_b != bot_w {
            (*bot_w).set_position(pla, &board, &hist);
        }
    }

    let mut locs_buf: Vec<Loc> = Vec::new();
    let mut play_selection_values_buf: Vec<f64> = Vec::new();
    let mut side_positions_to_search: Vec<SidePosition> = Vec::new();
    let mut historical_mcts_win_loss_values: Vec<f64> = Vec::new();
    let mut historical_mcts_leads: Vec<f64> = Vec::new();
    let mut historical_mcts_score_stdevs: Vec<f64> = Vec::new();
    let mut raw_nn_values: Vec<ReportedSearchValues> = Vec::new();

    let timer = kata_core::time::timer::ClockTimer::new();

    for i in 0..max_moves_per_game {
        if do_end_game_if_all_pass_alive {
            hist.end_game_if_all_pass_alive(&board);
        }
        if hist.is_game_finished {
            break;
        }
        if let Some(pause) = should_pause {
            pause.wait_until_false();
        }
        if should_stop() {
            break;
        }

        let to_move_bot = if pla == P_BLACK { bot_b } else { bot_w };

        if play_settings.dynamic_self_komi_bonus_min != 0.0
            || play_settings.dynamic_self_komi_bonus_max != 0.0
        {
            assert!(
                bot_b != bot_w && !record_full_data,
                "Dynamic komi for matches only right now"
            );
            let pla_factor = if pla == P_BLACK { -1.0 } else { 1.0 };
            let bot_ref = unsafe { &*to_move_bot };
            let mut current_komi_bonus =
                pla_factor * ((bot_ref.get_root_hist().rules.komi - hist.rules.komi) as f64 / 2.0);
            if historical_mcts_win_loss_values.len() >= 2 {
                let prev_win_loss = pla_factor
                    * historical_mcts_win_loss_values[historical_mcts_win_loss_values.len() - 2];
                if prev_win_loss < play_settings.dynamic_self_komi_win_loss_min {
                    current_komi_bonus += 0.5;
                }
                if prev_win_loss > play_settings.dynamic_self_komi_win_loss_max {
                    current_komi_bonus -= 0.5;
                }
            }
            current_komi_bonus = current_komi_bonus
                .max(play_settings.dynamic_self_komi_bonus_min)
                .min(play_settings.dynamic_self_komi_bonus_max);
            unsafe {
                (*to_move_bot).set_komi_if_new(
                    pla_factor * current_komi_bonus + hist.rules.komi as f64 / 2.0,
                );
            }
        }

        let limits = get_search_limits_this_move(
            to_move_bot,
            pla,
            play_settings,
            game_rand,
            &historical_mcts_win_loss_values,
            clear_bot_before_search,
            other_game_props,
        );

        let loc = if play_settings.record_time_per_move {
            let t0 = timer.get_seconds();
            let l = unsafe { run_bot_with_limits(to_move_bot, pla, play_settings, &limits) };
            let t1 = timer.get_seconds();
            if pla == P_BLACK {
                game_data.b_time_used += t1 - t0;
            } else {
                game_data.w_time_used += t1 - t0;
            }
            l
        } else {
            unsafe { run_bot_with_limits(to_move_bot, pla, play_settings, &limits) }
        };

        if pla == P_BLACK {
            game_data.b_move_count += 1;
        } else {
            game_data.w_move_count += 1;
        }

        let bot_ref = unsafe { &*to_move_bot };
        if loc == NULL_LOC || !bot_ref.is_legal_strict(loc, pla) {
            logger.write(&format!(
                "Bot returned null location or illegal move!?!\n{:?}\nPla: {:?}\nLoc: {:?}",
                board, pla, loc
            ));
            panic!("Illegal move from bot");
        }
        if log_search_info {
            logger.write(&format!(
                "Search info: turn {} move {:?} visits {}",
                hist.move_history.len(),
                loc,
                bot_ref.get_root_visits()
            ));
        }
        if log_moves {
            logger.write(&format!("Move {} made: {:?}", hist.move_history.len(), loc));
        }

        let root_node = bot_ref.get_root_node().expect("Bot root node was null");
        let mut white_value_targets = ValueTargets::default();
        extract_value_targets(&mut white_value_targets, to_move_bot, root_node);
        game_data
            .white_value_targets_by_turn
            .push(white_value_targets);

        let mut q_targets = Vec::new();
        extract_q_value_targets(&mut q_targets, to_move_bot, root_node);
        game_data
            .white_q_value_targets_by_turn
            .push(QValueTargets { targets: q_targets });

        if !record_full_data {
            let unreduced_num_visits = bot_ref.get_root_visits();
            game_data.policy_targets_by_turn.push(PolicyTarget {
                policy_targets: Vec::new(),
                unreduced_num_visits,
            });
        } else {
            let mut policy_target = Vec::new();
            let unreduced_num_visits = bot_ref.get_root_visits();
            extract_policy_target(
                &mut policy_target,
                to_move_bot,
                root_node as *const SearchNode,
                &mut locs_buf,
                &mut play_selection_values_buf,
            );
            game_data.policy_targets_by_turn.push(PolicyTarget {
                policy_targets: policy_target,
                unreduced_num_visits,
            });
            game_data.nn_raw_stats_by_turn.push(compute_nn_raw_stats(
                to_move_bot,
                &board,
                &hist,
                pla,
            ));
            game_data.target_weight_by_turn.push(limits.target_weight);
            game_data
                .target_weight_by_turn_unrounded
                .push(limits.target_weight);

            let mut policy_surprise = 0.0;
            let mut policy_entropy = 0.0;
            let mut search_entropy = 0.0;
            let success = bot_ref.get_policy_surprise_and_entropy(
                &mut policy_surprise,
                &mut search_entropy,
                &mut policy_entropy,
            );
            assert!(success);
            game_data.policy_surprise_by_turn.push(policy_surprise);
            game_data.policy_entropy_by_turn.push(policy_entropy);
            game_data.search_entropy_by_turn.push(search_entropy);
            raw_nn_values.push(bot_ref.get_root_raw_nn_values_require_success());

            let mut side_position_fork_loc = NULL_LOC;
            if play_settings.side_position_prob > 0.0
                && game_rand.next_bool(play_settings.side_position_prob)
            {
                if let Some(nn_output) = root_node.get_nn_output() {
                    let ban_move = loc;
                    side_position_fork_loc = choose_random_forking_move(
                        &nn_output, &board, &hist, pla, game_rand, ban_move,
                    );
                    if side_position_fork_loc != NULL_LOC {
                        let mut sp = SidePosition {
                            board: board.clone(),
                            hist: hist.clone(),
                            pla,
                            num_neural_net_changes_so_far: game_data.changed_neural_nets.len()
                                as i32,
                            ..Default::default()
                        };
                        sp.hist.make_board_move_assume_legal(
                            &mut sp.board,
                            side_position_fork_loc,
                            sp.pla,
                        );
                        sp.pla = get_opp(sp.pla);
                        if !sp.hist.is_game_finished {
                            side_positions_to_search.push(sp);
                        }
                    }
                }
            }

            if play_settings.record_tree_positions && play_settings.record_tree_target_weight > 0.0
            {
                assert!(play_settings.record_tree_target_weight <= 1.0);
                let num_neural_net_changes_so_far = game_data.changed_neural_nets.len() as i32;
                record_tree_positions(
                    &mut game_data,
                    &board,
                    &hist,
                    pla,
                    to_move_bot,
                    play_settings.record_tree_threshold as i64,
                    play_settings.record_tree_target_weight,
                    num_neural_net_changes_so_far,
                    &mut locs_buf,
                    &mut play_selection_values_buf,
                    loc,
                    side_position_fork_loc,
                );
            }
        }

        if play_settings.allow_resignation || play_settings.reduce_visits {
            let values = bot_ref.get_root_values_require_success();
            historical_mcts_win_loss_values.push(values.win_loss_value);
            historical_mcts_leads.push(values.lead);
            historical_mcts_score_stdevs.push(values.expected_score_stdev);
        }

        on_each_move(
            &board,
            &hist,
            pla,
            loc,
            &historical_mcts_win_loss_values,
            &historical_mcts_leads,
            &historical_mcts_score_stdevs,
            to_move_bot,
        );

        unsafe {
            assert!((*bot_b).make_move(loc, pla));
            if bot_b != bot_w {
                assert!((*bot_w).make_move(loc, pla));
            }
        }

        assert!(hist.is_legal(&board, loc, pla));
        hist.make_board_move_assume_legal(&mut board, loc, pla);

        if play_settings.allow_resignation
            && historical_mcts_win_loss_values.len() >= play_settings.resign_consec_turns as usize
        {
            let min_turn_for_resignation = 1 + board.x_size * board.y_size / 5;
            if i >= min_turn_for_resignation {
                assert!(
                    play_settings.resign_threshold <= 0.0
                        && !play_settings.resign_threshold.is_nan()
                );
                let mut should_resign = true;
                for j in 0..play_settings.resign_consec_turns as usize {
                    let idx = historical_mcts_win_loss_values.len() - 1 - j;
                    let win_loss_value = historical_mcts_win_loss_values[idx];
                    let resign_player_this_turn = if win_loss_value < play_settings.resign_threshold
                    {
                        P_WHITE
                    } else if win_loss_value > -play_settings.resign_threshold {
                        P_BLACK
                    } else {
                        C_EMPTY
                    };
                    if resign_player_this_turn != pla {
                        should_resign = false;
                        break;
                    }
                }
                if should_resign {
                    hist.set_winner_by_resignation(get_opp(pla));
                }
            }
        }

        let next_turn_idx = hist.move_history.len() as i32;
        if game_rand.next_bool(0.1) {
            if let Some(new_nn_eval_ptr) = check_for_new_nn_eval() {
                let new_nn_eval: &NnEvaluator = unsafe { &*new_nn_eval_ptr };
                unsafe {
                    (*bot_b).set_nn_eval(Some(new_nn_eval));
                    if bot_b != bot_w {
                        (*bot_w).set_nn_eval(Some(new_nn_eval));
                    }
                }
                game_data.changed_neural_nets.push(ChangedNeuralNet {
                    name: new_nn_eval.model_name().to_string(),
                    turn_idx: next_turn_idx,
                });
            }
        }

        pla = get_opp(pla);
    }

    game_data.end_hist = hist.clone();
    game_data.hit_turn_limit = !hist.is_game_finished;

    assert_eq!(
        hist.num_consec_valid_turns_this_game,
        hist.move_history.len() as i32,
        "Selfplay got history with not entire game legal"
    );

    {
        let mut hist_copy = hist.clone();
        hist_copy.set_assume_multiple_starting_black_moves_are_handicap(true);
        game_data.handicap_for_sgf = hist_copy.compute_num_handicap_stones();
    }

    if record_full_data {
        assert!(
            !hist.is_resignation,
            "Recording full data currently incompatible with resignation"
        );

        let mut final_value_targets = ValueTargets::default();
        let mut final_full_area = vec![C_EMPTY; MAX_ARR_SIZE];
        let mut final_ownership = vec![C_EMPTY; MAX_ARR_SIZE];
        let mut final_seki_areas = vec![false; MAX_ARR_SIZE];

        if hist.is_game_finished && hist.is_no_result {
            final_value_targets.win = 0.0;
            final_value_targets.loss = 0.0;
            final_value_targets.no_result = 1.0;
            final_value_targets.score = 0.0;
            final_full_area.fill(C_EMPTY);
            final_ownership.fill(C_EMPTY);
            final_seki_areas.fill(false);
        } else {
            hist.end_and_score_game_now(&board);
            final_value_targets.win = score_value::white_wins_of_winner(
                hist.winner,
                game_data.draw_equivalent_wins_for_white,
            ) as f32;
            final_value_targets.loss = 1.0 - final_value_targets.win;
            final_value_targets.no_result = 0.0;
            final_value_targets.score = score_value::white_score_draw_adjust(
                f64::from(hist.final_white_minus_black_score),
                game_data.draw_equivalent_wins_for_white,
                &hist,
            ) as f32;
            final_value_targets.has_lead = true;
            final_value_targets.lead = final_value_targets.score;

            board.calculate_area(
                &mut final_full_area,
                true,
                true,
                true,
                hist.rules.multi_stone_suicide_legal,
            );
            let mut independent_life_area = vec![C_EMPTY; MAX_ARR_SIZE];
            board.calculate_independent_life_area(
                &mut independent_life_area,
                false,
                false,
                hist.rules.multi_stone_suicide_legal,
            );
            for i in 0..MAX_ARR_SIZE {
                final_seki_areas[i] = independent_life_area[i] == C_EMPTY
                    && (final_full_area[i] == P_BLACK || final_full_area[i] == P_WHITE);
            }
        }
        game_data
            .white_value_targets_by_turn
            .push(final_value_targets);

        if other_game_props.hint_loc != NULL_LOC {
            let idx = 1usize.min(game_data.white_value_targets_by_turn.len() - 1);
            game_data.white_value_targets_by_turn[0] =
                game_data.white_value_targets_by_turn[idx].clone();
        }

        let mut final_white_scoring = vec![0.0f32; MAX_ARR_SIZE];
        fill_scoring(
            &board,
            &final_ownership,
            hist.rules.tax_rule == TaxRule::All,
            &mut final_white_scoring,
        );
        game_data.final_full_area = final_full_area;
        game_data.final_ownership = final_ownership;
        game_data.final_seki_areas = final_seki_areas;
        game_data.final_white_scoring = final_white_scoring;
        game_data.has_full_data = true;
    }

    game_data.training_weight = other_game_props.training_weight;
    Some(game_data)
}

/// Replay a finished game up to (but not including) `move_idx`.
fn replay_game_up_to_move(
    finished_game_data: &FinishedGameData,
    move_idx: i32,
    rules: Rules,
    board: &mut Board,
    hist: &mut BoardHistory,
    pla: &mut Player,
) {
    *board = finished_game_data.start_hist.initial_board.clone();
    *pla = finished_game_data.start_hist.initial_pla;

    if rules.scoring_rule == ScoringRule::Area {
        hist.clear(board.clone(), *pla, rules, 0);
    } else {
        hist.clear(
            board.clone(),
            *pla,
            rules,
            finished_game_data.start_hist.initial_encore_phase,
        );
    }

    if finished_game_data.end_hist.move_history.is_empty() {
        return;
    }
    let mut move_idx = move_idx.min(finished_game_data.end_hist.move_history.len() as i32 - 1);
    if move_idx < 0 {
        move_idx = 0;
    }

    for i in 0..move_idx as usize {
        let m = finished_game_data.end_hist.move_history[i];
        if !hist.is_legal(board, m.loc, *pla) {
            if rules == finished_game_data.start_hist.rules && hist.encore_phase == 0 {
                panic!("Illegal move when replaying to fork game?");
            }
            return;
        }
        assert_eq!(m.pla, *pla);
        hist.make_board_move_assume_legal(board, m.loc, *pla);
        *pla = get_opp(*pla);
        if hist.is_game_finished {
            return;
        }
    }
}

fn has_unowned_spot(finished_game_data: &FinishedGameData) -> bool {
    let board = &finished_game_data.start_board;
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = location::get_loc(x, y, board.x_size) as usize;
            if finished_game_data.final_ownership[loc] == C_EMPTY {
                return true;
            }
        }
    }
    false
}

/// Possibly add a forked position from a finished game to `fork_data`.
pub fn maybe_fork_game(
    finished_game_data: &FinishedGameData,
    fork_data: &ForkData,
    play_settings: &PlaySettings,
    game_rand: &mut Rand,
    bot: *mut Search,
) {
    assert_eq!(
        finished_game_data.start_hist.initial_board.pos_hash,
        finished_game_data.end_hist.initial_board.pos_hash
    );
    assert_eq!(
        finished_game_data.start_hist.initial_pla,
        finished_game_data.end_hist.initial_pla
    );

    if finished_game_data.start_hist.encore_phase != 0 {
        return;
    }

    let early_fork = game_rand.next_bool(play_settings.early_fork_game_prob);
    let late_fork = !early_fork
        && play_settings.fork_game_prob > 0.0
        && game_rand.next_bool(play_settings.fork_game_prob);
    if !early_fork && !late_fork {
        return;
    }

    let move_idx = if early_fork {
        (game_rand.next_exponential()
            * (play_settings.early_fork_game_expected_move_prop
                * finished_game_data.start_board.x_size as f64
                * finished_game_data.start_board.y_size as f64))
            .floor() as i32
    } else {
        if finished_game_data.end_hist.move_history.is_empty() {
            0
        } else {
            game_rand.next_u32_bounded(finished_game_data.end_hist.move_history.len() as u32) as i32
        }
    };

    let mut board = Board::default();
    let mut pla = C_EMPTY;
    let mut hist = BoardHistory::default();
    replay_game_up_to_move(
        finished_game_data,
        move_idx,
        finished_game_data.start_hist.rules,
        &mut board,
        &mut hist,
        &mut pla,
    );
    if hist.is_game_finished {
        return;
    }

    assert!(
        play_settings.fork_game_max_choices <= nn_pos::MAX_NN_POLICY_SIZE as i32
            && play_settings.early_fork_game_max_choices <= nn_pos::MAX_NN_POLICY_SIZE as i32
    );
    let max_choices = if early_fork {
        play_settings.early_fork_game_max_choices
    } else {
        play_settings.fork_game_max_choices
    };
    assert!(max_choices >= play_settings.fork_game_min_choices);

    let num_choices = rand_next_int(game_rand, play_settings.fork_game_min_choices, max_choices);
    let mut possible_moves = vec![NULL_LOC; num_choices as usize];
    let num_possible =
        choose_random_legal_moves(&board, &hist, pla, game_rand, &mut possible_moves);
    if num_possible == 0 {
        return;
    }

    let bot_ref = unsafe { &*bot };
    let nn_eval = bot_ref
        .nn_evaluator
        .expect("maybeForkGame called with a Search that has no NNEvaluator");
    let draw_equivalent_wins_for_white = bot_ref.search_params.draw_equivalent_wins_for_white;

    let mut best_move = NULL_LOC;
    let mut best_score = 0.0f32;
    for i in 0..num_possible {
        let loc = possible_moves[i];
        let mut copy = board.clone();
        let mut copy_hist = hist.clone();
        copy_hist.make_board_move_assume_legal(&mut copy, loc, pla);
        let mut buf = NNResultBuf::new();
        let mut nn_input_params = MiscNNInputParams::default();
        nn_input_params.draw_equivalent_wins_for_white = draw_equivalent_wins_for_white;
        nn_eval.evaluate(
            &copy,
            &copy_hist,
            get_opp(pla),
            &nn_input_params,
            &mut buf,
            false,
            false,
        );
        let nn_output = buf.result.expect("NN evaluation produced no output");
        let white_score = nn_output.white_score_mean;
        if best_move == NULL_LOC
            || (pla == P_WHITE && white_score > best_score)
            || (pla == P_BLACK && white_score < best_score)
        {
            best_move = loc;
            best_score = white_score;
        }
    }

    assert!(hist.is_legal(&board, best_move, pla));
    hist.make_board_move_assume_legal(&mut board, best_move, pla);
    pla = get_opp(pla);
    if hist.is_game_finished {
        return;
    }

    fork_data.add(InitialPosition::new(
        board,
        hist,
        pla,
        true,
        false,
        false,
        finished_game_data.training_weight,
    ));
}

/// Possibly add seki fork positions from a finished game to `fork_data`.
pub fn maybe_seki_fork_game(
    finished_game_data: &FinishedGameData,
    fork_data: &ForkData,
    play_settings: &PlaySettings,
    game_init: Option<&GameInitializer>,
    game_rand: &mut Rand,
) {
    if play_settings.seki_fork_hack_prob <= 0.0 {
        return;
    }

    let end_hist = &finished_game_data.end_hist;
    if end_hist.is_game_finished
        && end_hist.is_scored
        && finished_game_data.start_hist.encore_phase < 2
        && has_unowned_spot(finished_game_data)
    {
        for _ in 0..2 {
            let mut move_idx = (end_hist.move_history.len() as f64
                * (1.0 - 0.10 * game_rand.next_exponential())
                - 1.0)
                .floor() as i32;
            if move_idx < 0 {
                move_idx = 0;
            }
            if move_idx > end_hist.move_history.len() as i32 {
                move_idx = end_hist.move_history.len() as i32;
            }

            let mut rules = finished_game_data.start_hist.rules;
            if let Some(init) = game_init {
                rules = init.randomize_scoring_and_tax_rules(rules, game_rand);
            }

            let mut board = Board::default();
            let mut pla = C_EMPTY;
            let mut hist = BoardHistory::default();
            replay_game_up_to_move(
                finished_game_data,
                move_idx,
                rules,
                &mut board,
                &mut hist,
                &mut pla,
            );
            if hist.is_game_finished {
                continue;
            }
            fork_data.add_seki(
                InitialPosition::new(
                    board,
                    hist,
                    pla,
                    false,
                    true,
                    false,
                    finished_game_data.training_weight,
                ),
                game_rand,
            );
        }
    }
}

/// Possibly add a hint fork position from a finished game to `fork_data`.
pub fn maybe_hint_fork_game(
    finished_game_data: &FinishedGameData,
    fork_data: &ForkData,
    other_game_props: &OtherGameProperties,
) {
    if finished_game_data.start_hist.encore_phase != 0 {
        return;
    }

    let hint_fork = other_game_props.hint_loc != NULL_LOC
        && finished_game_data.start_board.pos_hash == other_game_props.hint_pos_hash
        && finished_game_data.start_hist.move_history.len() as i32 == other_game_props.hint_turn
        && finished_game_data.end_hist.move_history.len()
            > finished_game_data.start_hist.move_history.len()
        && finished_game_data.end_hist.move_history
            [finished_game_data.start_hist.move_history.len()]
        .loc != other_game_props.hint_loc;

    if !hint_fork {
        return;
    }

    let mut board = Board::default();
    let mut pla = C_EMPTY;
    let mut hist = BoardHistory::default();
    replay_game_up_to_move(
        finished_game_data,
        finished_game_data.start_hist.move_history.len() as i32,
        finished_game_data.start_hist.rules,
        &mut board,
        &mut hist,
        &mut pla,
    );
    if hist.is_game_finished {
        return;
    }

    assert_eq!(pla, hist.presumed_next_move_pla);
    if !hist.is_legal(&board, other_game_props.hint_loc, pla) {
        return;
    }

    hist.make_board_move_assume_legal(&mut board, other_game_props.hint_loc, pla);
    pla = get_opp(pla);
    if hist.is_game_finished {
        return;
    }

    fork_data.add(InitialPosition::new(
        board,
        hist,
        pla,
        false,
        false,
        true,
        finished_game_data.training_weight,
    ));
}

/// Extract a policy target distribution from a search node.
pub fn extract_policy_target(
    buf: &mut Vec<PolicyTargetMove>,
    to_move_bot: *const Search,
    node: *const SearchNode,
    locs_buf: &mut Vec<Loc>,
    play_selection_values_buf: &mut Vec<f64>,
) {
    assert!(!to_move_bot.is_null());
    assert!(!node.is_null());

    let bot = unsafe { &*to_move_bot };
    assert!(!bot.search_params.root_symmetry_pruning);

    let n = unsafe { &*node };
    let scale_max_to_at_least = 10.0;
    let success = bot.get_play_selection_values_for_node(
        n,
        locs_buf,
        play_selection_values_buf,
        None,
        scale_max_to_at_least,
        false,
    );
    assert!(success);
    assert_eq!(locs_buf.len(), play_selection_values_buf.len());

    let max_value = play_selection_values_buf
        .iter()
        .cloned()
        .fold(0.0, f64::max);
    let factor = if max_value > 30000.0 {
        30000.0 / max_value
    } else {
        1.0
    };

    buf.clear();
    for (&loc, &value) in locs_buf.iter().zip(play_selection_values_buf.iter()) {
        buf.push(PolicyTargetMove {
            loc,
            policy_target: (value * factor).round() as i16,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kata_core::logger::LoggerOptions;
    use kata_game::board::{NULL_LOC, P_BLACK, P_WHITE};
    use kata_game::rules::Rules;
    use kata_nn::backend::Enabled;
    use std::sync::Arc;

    fn sample_position(name: &str) -> InitialPosition {
        let board = Board::new(9, 9);
        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        InitialPosition::new(
            board,
            hist,
            P_BLACK,
            name == "plain",
            name == "seki",
            name == "hint",
            1.0,
        )
    }

    fn test_logger() -> Logger {
        Logger::new(LoggerOptions::default(), None)
    }

    fn cfg(s: &str) -> ConfigParser {
        ConfigParser::from_str(s, false, true).unwrap()
    }

    fn game_init_cfg() -> &'static str {
        "koRules = SIMPLE,POSITIONAL\n\
         scoringRules = AREA\n\
         taxRules = NONE\n\
         multiStoneSuicideLegals = true,false\n\
         hasButtons = true,false\n\
         bSizes = 9,13,19\n\
         bSizeRelProbs = 1.0,1.0,1.0\n\
         komiMean = 7.5\n"
    }

    fn dummy_nn_eval(logger: &Logger) -> NnEvaluator {
        NnEvaluator::new(
            "dummy".to_string(),
            "dummy.bin".to_string(),
            String::new(),
            Arc::new(logger.clone()),
            1,
            7,
            7,
            false,
            false,
            -1,
            0,
            true,
            String::new(),
            Enabled::False,
            1,
            vec![0],
            "seed".to_string(),
            false,
            -1,
            true,
            &ConfigParser::new(false, true),
        )
    }

    fn tiny_search_params() -> SearchParams {
        let mut params = SearchParams::new();
        params.max_visits = 2;
        params.max_playouts = 2;
        params.num_threads = 1;
        params.use_lcb_for_selection = false;
        params.value_weight_exponent = 0.0;
        params
    }

    #[test]
    fn test_initial_position_default() {
        let pos = InitialPosition::default();
        assert_eq!(pos.pla, C_EMPTY);
        assert!(!pos.is_plain_fork);
        assert_eq!(pos.training_weight, 1.0);
    }

    #[test]
    fn test_fork_data_add_and_get() {
        let fork_data = ForkData::new();
        assert!(fork_data.get(&mut Rand::new_from_seed("a")).is_none());

        fork_data.add(sample_position("plain"));
        fork_data.add(sample_position("plain"));
        assert_eq!(fork_data.fork_count(), 2);

        let mut rand = Rand::new_from_seed("b");
        let got = fork_data.get(&mut rand);
        assert!(got.is_some());
        assert_eq!(fork_data.fork_count(), 1);

        let got2 = fork_data.get(&mut rand);
        assert!(got2.is_some());
        assert!(fork_data.get(&mut rand).is_none());
    }

    #[test]
    fn test_fork_data_seki_cap() {
        let fork_data = ForkData::new();
        let mut rand = Rand::new_from_seed("seki");
        for i in 0..1002 {
            let mut pos = sample_position("seki");
            pos.training_weight = i as f64;
            fork_data.add_seki(pos, &mut rand);
        }
        assert_eq!(fork_data.seki_fork_count(), 1000);

        let mut seen = 0;
        while fork_data.get_seki(&mut rand).is_some() {
            seen += 1;
        }
        assert_eq!(seen, 1000);
    }

    #[test]
    fn test_other_game_properties_default() {
        let props = OtherGameProperties::default();
        assert!(props.allow_policy_init);
        assert_eq!(props.training_weight, 1.0);
        assert_eq!(props.playout_doubling_advantage, 0.0);
        assert_eq!(props.hint_turn, -1);
        assert_eq!(props.hint_loc, NULL_LOC);
    }

    #[test]
    fn test_initial_position_preserves_board() {
        let board = Board::new(13, 13);
        let hist = BoardHistory::new(board.clone(), P_WHITE, Rules::default(), 0);
        let pos = InitialPosition::new(board.clone(), hist, P_WHITE, true, false, false, 0.5);
        assert_eq!(pos.board.x_size, 13);
        assert_eq!(pos.pla, P_WHITE);
        assert!(pos.is_plain_fork);
        assert_eq!(pos.training_weight, 0.5);
    }

    #[test]
    fn test_game_initializer_new() {
        let cfg = cfg(game_init_cfg());
        let logger = test_logger();
        let init = GameInitializer::new(&cfg, &logger).unwrap();

        assert!(init.is_allowed_b_size(9, 9));
        assert!(init.is_allowed_b_size(13, 13));
        assert!(init.is_allowed_b_size(19, 19));
        assert!(!init.is_allowed_b_size(9, 13));

        let allowed = init.get_allowed_b_sizes();
        assert_eq!(allowed.len(), 3);

        assert_eq!(init.get_min_board_x_size(), 9);
        assert_eq!(init.get_min_board_y_size(), 9);
        assert_eq!(init.get_max_board_x_size(), 19);
        assert_eq!(init.get_max_board_y_size(), 19);
    }

    #[test]
    fn test_game_initializer_new_with_seed_reproduces() {
        let cfg = cfg(game_init_cfg());
        let logger = test_logger();
        let _init = GameInitializer::new_with_seed(&cfg, &logger, "seed1").unwrap();
    }

    #[test]
    fn test_game_initializer_create_rules() {
        let cfg = cfg(game_init_cfg());
        let logger = test_logger();
        let init = GameInitializer::new(&cfg, &logger).unwrap();

        for _ in 0..20 {
            let rules = init.create_rules();
            assert!(rules.scoring_rule == ScoringRule::Area);
            assert!(rules.tax_rule == TaxRule::None);
            assert!(!rules.has_button || rules.scoring_rule == ScoringRule::Area);
        }
    }

    #[test]
    fn test_game_initializer_create_game_stub() {
        let cfg = cfg(game_init_cfg());
        let logger = test_logger();
        let init = GameInitializer::new(&cfg, &logger).unwrap();

        let mut board = Board::default();
        let mut pla = C_EMPTY;
        let mut hist = BoardHistory::default();
        let mut extra = ExtraBlackAndKomi::default();
        let mut props = OtherGameProperties::default();
        let play_settings = PlaySettings::new();

        init.create_game(
            &mut board,
            &mut pla,
            &mut hist,
            &mut extra,
            None,
            &play_settings,
            &mut props,
            None,
        );

        assert!(board.x_size > 0);
        assert!(board.y_size > 0);
        assert!(pla == P_BLACK);
        assert!(init.is_allowed_b_size(board.x_size, board.y_size));
    }

    #[test]
    fn test_match_pairer_new_and_get_matchup() {
        let cfg = cfg("matchupsPerRound = 0-1\nnumGamesTotal = 4\nlogGamesEvery = 2\n");
        let logger = test_logger();
        let pairer = MatchPairer::new(
            &cfg,
            2,
            vec!["bot0".to_string(), "bot1".to_string()],
            vec![None, None],
            vec![SearchParams::new(), SearchParams::new()],
        )
        .unwrap();

        assert_eq!(pairer.get_num_games_total_to_generate(), 4);

        let mut pairer = pairer;
        for i in 0..4 {
            let (b, w) = pairer.get_matchup(&logger).unwrap();
            assert_eq!(b.bot_idx, 0);
            assert_eq!(w.bot_idx, 1);
            assert_eq!(b.bot_name, "bot0");
            assert_eq!(w.bot_name, "bot1");
            assert!(b.nn_eval.is_none());
            assert!(w.nn_eval.is_none());
            assert_eq!(i as i64 + 1, pairer.num_games_started_so_far);
        }
        assert!(pairer.get_matchup(&logger).is_none());
    }

    #[test]
    fn test_game_runner_new_and_get_initializer() {
        let cfg_str = format!(
            "{}logSearchInfo = false\nlogMoves = false\nmaxMovesPerGame = 1000\nclearBotBeforeSearch = true\n",
            game_init_cfg()
        );
        let cfg = cfg(&cfg_str);
        let logger = test_logger();
        let play_settings = PlaySettings::new();
        let runner = GameRunner::new(&cfg, play_settings, &logger).unwrap();

        let init = runner.get_game_initializer().unwrap();
        assert_eq!(init.get_min_board_x_size(), 9);
        assert_eq!(init.get_max_board_x_size(), 19);
    }

    #[test]
    fn test_game_runner_new_with_seed() {
        let cfg_str = format!(
            "{}logSearchInfo = false\nlogMoves = false\nmaxMovesPerGame = 500\n",
            game_init_cfg()
        );
        let cfg = cfg(&cfg_str);
        let logger = test_logger();
        let play_settings = PlaySettings::new();
        let runner =
            GameRunner::new_with_seed(&cfg, "runner-seed", play_settings, &logger).unwrap();
        assert!(runner.get_game_initializer().is_some());
    }

    #[test]
    fn test_extract_policy_target_after_search() {
        let logger = test_logger();
        let nn_eval = dummy_nn_eval(&logger);
        let mut search = Search::new(tiny_search_params(), &nn_eval, &logger, "search-seed");
        let board = Board::new(5, 5);
        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        search.set_position(P_BLACK, &board, &hist);
        search.run_whole_search(P_BLACK);

        let root_node = search.get_root_node().expect("root node should exist");
        let mut buf = Vec::new();
        let mut locs = Vec::new();
        let mut vals = Vec::new();
        extract_policy_target(
            &mut buf,
            &search as *const Search,
            root_node as *const SearchNode,
            &mut locs,
            &mut vals,
        );

        assert!(!buf.is_empty());
        let sum: i64 = buf.iter().map(|m| m.policy_target as i64).sum();
        assert!(sum > 0);
    }

    #[test]
    fn test_record_tree_positions_populates_side_positions() {
        let logger = test_logger();
        let nn_eval = dummy_nn_eval(&logger);
        let mut params = tiny_search_params();
        params.max_visits = 50;
        params.max_playouts = 50;
        let mut search = Search::new(params, &nn_eval, &logger, "tree-search-seed");
        let board = Board::new(5, 5);
        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        search.set_position(P_BLACK, &board, &hist);
        search.run_whole_search(P_BLACK);

        let mut game_data = FinishedGameData::default();
        let mut locs = Vec::new();
        let mut vals = Vec::new();
        let before = game_data.side_positions.len();
        record_tree_positions(
            &mut game_data,
            &board,
            &hist,
            P_BLACK,
            &search as *const Search,
            1,
            1.0,
            0,
            &mut locs,
            &mut vals,
            NULL_LOC,
            NULL_LOC,
        );
        assert!(
            game_data.side_positions.len() > before,
            "record_tree_positions should add at least one side position"
        );
    }

    #[test]
    fn test_run_game_with_bots_tiny_match() {
        let logger = test_logger();
        let nn_eval = dummy_nn_eval(&logger);
        let params = tiny_search_params();
        let bot_spec_b = BotSpec {
            bot_idx: 0,
            bot_name: "b".to_string(),
            nn_eval: Some(&nn_eval),
            base_params: params.clone(),
        };
        let bot_spec_w = BotSpec {
            bot_idx: 1,
            bot_name: "w".to_string(),
            nn_eval: Some(&nn_eval),
            base_params: params,
        };

        let board = Board::new(5, 5);
        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        let extra = ExtraBlackAndKomi::default();
        let play_settings = PlaySettings::new();
        let other_props = OtherGameProperties::default();
        let mut game_rand = Rand::new_from_seed("tiny-match");

        let result = run_game(
            &board,
            P_BLACK,
            &hist,
            extra,
            &bot_spec_b,
            &bot_spec_w,
            "match-seed",
            true,
            true,
            &logger,
            false,
            false,
            20,
            &|| false,
            None,
            &play_settings,
            &other_props,
            &mut game_rand,
            &|| None,
            Box::new(|_, _, _, _, _, _, _, _| {}),
        );

        assert!(result.is_some());
        let game_data = result.unwrap();
        assert_eq!(game_data.b_name, "b");
        assert_eq!(game_data.w_name, "w");
        assert!(game_data.end_hist.move_history.len() > 0 || game_data.end_hist.is_game_finished);
    }

    #[test]
    fn test_maybe_fork_game_creates_fork() {
        let logger = test_logger();
        let nn_eval = dummy_nn_eval(&logger);
        let mut params = tiny_search_params();
        params.max_visits = 1;
        params.max_playouts = 1;
        let mut search = Search::new(params, &nn_eval, &logger, "fork-search");
        let board = Board::new(5, 5);
        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        search.set_position(P_BLACK, &board, &hist);

        let mut finished = FinishedGameData::default();
        finished.start_board = board.clone();
        finished.start_hist = hist.clone();
        finished.start_pla = P_BLACK;
        finished.end_hist = hist.clone();
        finished.training_weight = 1.0;

        let fork_data = ForkData::new();
        let mut play_settings = PlaySettings::new();
        play_settings.fork_game_prob = 1.0;
        play_settings.fork_game_min_choices = 1;
        play_settings.fork_game_max_choices = 1;
        play_settings.early_fork_game_max_choices = 1;
        let mut game_rand = Rand::new_from_seed("fork-rand");

        maybe_fork_game(
            &finished,
            &fork_data,
            &play_settings,
            &mut game_rand,
            &mut search as *mut Search,
        );
        assert_eq!(fork_data.fork_count(), 1);
    }

    #[test]
    fn test_maybe_hint_fork_game_creates_hint_fork() {
        let logger = test_logger();
        let nn_eval = dummy_nn_eval(&logger);
        let mut params = tiny_search_params();
        params.max_visits = 1;
        params.max_playouts = 1;
        let mut search = Search::new(params, &nn_eval, &logger, "hint-search");
        let mut board = Board::new(5, 5);
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        search.set_position(P_BLACK, &board, &hist);

        let hint_loc = location::get_loc(2, 2, 5);
        let mut other_props = OtherGameProperties::default();
        other_props.hint_loc = hint_loc;
        other_props.hint_turn = 0;
        other_props.hint_pos_hash = board.pos_hash;

        // Play one move that is NOT the hint move.
        hist.make_board_move_assume_legal(&mut board, location::get_loc(0, 0, 5), P_BLACK);

        let mut finished = FinishedGameData::default();
        finished.start_board = Board::new(5, 5);
        finished.start_hist =
            BoardHistory::new(finished.start_board.clone(), P_BLACK, Rules::default(), 0);
        finished.start_pla = P_BLACK;
        finished.end_hist = hist;
        finished.training_weight = 1.0;

        let fork_data = ForkData::new();
        maybe_hint_fork_game(&finished, &fork_data, &other_props);
        assert_eq!(fork_data.fork_count(), 1);
    }

    #[test]
    fn test_maybe_seki_fork_game_creates_seki_fork() {
        let mut finished = FinishedGameData::default();
        let board = Board::new(5, 5);
        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        finished.start_board = board.clone();
        finished.start_hist = hist.clone();
        finished.start_pla = P_BLACK;
        finished.end_hist = hist.clone();
        finished.training_weight = 1.0;
        finished.end_hist.is_game_finished = true;
        finished.end_hist.is_scored = true;
        finished.final_ownership = vec![C_EMPTY; MAX_ARR_SIZE];

        let fork_data = ForkData::new();
        let mut play_settings = PlaySettings::new();
        play_settings.seki_fork_hack_prob = 1.0;
        let mut game_rand = Rand::new_from_seed("seki-fork-rand");

        maybe_seki_fork_game(&finished, &fork_data, &play_settings, None, &mut game_rand);
        assert_eq!(fork_data.seki_fork_count(), 2);
    }
}
