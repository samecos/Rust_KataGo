//! GTP (Go Text Protocol) command loop.
//!
//! Corresponds to `cpp/command/gtp.cpp`. This implementation provides a usable
//! GTP engine built on top of `AsyncBot`, `Search`, and the program setup
//! helpers. A few commands that require large extra infrastructure (exact SGF
//! writing, full debug printouts, etc.) return a polite placeholder response.

#![allow(
    clippy::collapsible_if,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::unused_async,
    dead_code
)]

use std::io::{self, BufRead, Write};
use std::sync::{Arc, Mutex};

use clap::Parser;

use kata_core::command_loop::process_single_command_line;
use kata_core::config::ConfigParser;
use kata_core::global::{self, StringError};
use kata_core::logger::{Logger, LoggerOptions};
use kata_core::rng::Rand;
use kata_core::time::timer::ClockTimer;
use kata_data::sgf::{CompactSgf, write_sgf};
use kata_game::board::{
    Board, C_EMPTY, DEFAULT_LEN, Loc, MAX_ARR_SIZE, MAX_LEN, Move, NULL_LOC, P_BLACK, P_WHITE,
    PASS_LOC, Player, location, player_io,
};
use kata_game::history::BoardHistory;
use kata_game::rules::{Rules, WhiteHandicapBonusRule};
use kata_game::symmetry::NUM_SYMMETRIES;
use kata_nn::backend::Enabled;
use kata_nn::eval::NnEvaluator;
use kata_nn::inputs::{MiscNNInputParams, nn_inputs, nn_pos};
use kata_program::play_utils::{self};
use kata_program::setup::{self, SetupFor};
use kata_search::async_bot::AsyncBot;
use kata_search::params::SearchParams;
use kata_search::search::Search;
use kata_search::time_control::TimeControls;

use crate::cli::CommonArgs;

/// GTP commands advertised by this engine.
const KNOWN_COMMANDS: &[&str] = &[
    "protocol_version",
    "name",
    "version",
    "known_command",
    "list_commands",
    "quit",
    "boardsize",
    "rectangular_boardsize",
    "clear_board",
    "set_position",
    "komi",
    "get_komi",
    "play",
    "undo",
    "kata-get-rules",
    "kata-set-rule",
    "kata-set-rules",
    "kgs-rules",
    "kata-list-params",
    "kata-get-param",
    "kata-get-models",
    "kata-get-params",
    "kata-set-param",
    "kata-set-params",
    "genmove",
    "kata-search",
    "kata-search_cancellable",
    "genmove_debug",
    "kata-search_debug",
    "clear_cache",
    "showboard",
    "fixed_handicap",
    "place_free_handicap",
    "set_free_handicap",
    "time_settings",
    "kgs-time_settings",
    "kata-time_settings",
    "kata-list_time_settings",
    "time_left",
    "kata-debug-print-tc",
    "final_score",
    "final_status_list",
    "loadsgf",
    "printsgf",
    "lz-analyze",
    "kata-analyze",
    "lz-genmove_analyze",
    "kata-genmove_analyze",
    "kata-search_analyze",
    "kata-search_analyze_cancellable",
    "kata-raw-nn",
    "kata-raw-human-nn",
    "cputime",
    "gomill-cpu_time",
    "kata-benchmark",
    "kata-debug-print-tc",
    "debug_moves",
    "kata-list-colors-own-eyes",
    "stop",
];

/// CLI arguments for the `gtp` subcommand.
#[derive(Parser, Debug, Clone)]
struct GtpArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// Force KataGo to report a specific version string.
    #[arg(long = "override-version", value_name = "VERSION")]
    override_version: Option<String>,
}

/// Shared, thread-safe output handle used by the command loop and callbacks.
type OutputHandle = Arc<Mutex<dyn Write + Send + 'static>>;

/// The GTP engine state.
pub struct GtpEngine {
    nn_eval: &'static NnEvaluator,
    human_eval: Option<&'static NnEvaluator>,
    bot: AsyncBot<'static>,
    logger: &'static Logger,
    cfg: ConfigParser,

    model_file: String,
    human_model_file: Option<String>,
    seed_rand: Rand,

    current_rules: Rules,
    genmove_params: SearchParams,
    analysis_params: SearchParams,
    is_genmove_params: bool,

    b_time_controls: TimeControls,
    w_time_controls: TimeControls,

    initial_board: Board,
    initial_pla: Player,
    move_history: Vec<Move>,

    recent_win_loss_values: Vec<f64>,
    last_search_factor: f64,

    perspective: Player,
    analysis_pv_len: i32,
    assume_multiple_starting_black_moves_are_handicap: bool,
    prevent_encore: bool,

    dynamic_playout_doubling_advantage_cap_per_opp_lead: f64,
    static_pda_takes_precedence: bool,
    normal_avoid_repeated_pattern_utility: f64,
    handicap_avoid_repeated_pattern_utility: f64,

    delay_move_scale: f64,
    delay_move_max: f64,

    allow_resignation: bool,
    resign_threshold: f64,
    resign_consec_turns: i32,
    resign_min_score_difference: f64,
    resign_min_moves_per_board_area: f64,

    search_factor_when_winning: f64,
    search_factor_when_winning_threshold: f64,
    ogs_chat_to_stderr: bool,
    log_search_info: bool,
    log_search_info_for_chosen_move: bool,
    cleanup_before_pass: kata_core::config::Enabled,
    friendly_pass: kata_core::config::Enabled,

    genmove_timer: ClockTimer,
    genmove_time_sum: f64,
    genmove_expected_id: i32,
}

/// Arguments controlling a genmove/search command.
#[derive(Debug, Clone, Copy)]
struct GenmoveArgs {
    search_factor_when_winning_threshold: f64,
    search_factor_when_winning: f64,
    cleanup_before_pass: kata_core::config::Enabled,
    friendly_pass: kata_core::config::Enabled,
    ogs_chat_to_stderr: bool,
    allow_resignation: bool,
    resign_threshold: f64,
    resign_consec_turns: i32,
    resign_min_score_difference: f64,
    resign_min_moves_per_board_area: f64,
    log_search_info: bool,
    log_search_info_for_chosen_move: bool,
    debug: bool,
}

/// Arguments controlling an analyze-style command.
#[derive(Debug, Clone)]
struct AnalyzeArgs {
    analyzing: bool,
    lz: bool,
    kata: bool,
    min_moves: i32,
    max_moves: i32,
    show_root_info: bool,
    show_ownership: bool,
    show_ownership_stdev: bool,
    show_moves_ownership: bool,
    show_moves_ownership_stdev: bool,
    show_pv_visits: bool,
    show_pv_edge_visits: bool,
    show_no_result_value: bool,
    seconds_per_report: f64,
    avoid_move_until_by_loc_black: Vec<i32>,
    avoid_move_until_by_loc_white: Vec<i32>,
}

impl Default for AnalyzeArgs {
    fn default() -> Self {
        Self {
            analyzing: false,
            lz: false,
            kata: false,
            min_moves: 0,
            max_moves: 10000000,
            show_root_info: false,
            show_ownership: false,
            show_ownership_stdev: false,
            show_moves_ownership: false,
            show_moves_ownership_stdev: false,
            show_pv_visits: false,
            show_pv_edge_visits: false,
            show_no_result_value: false,
            seconds_per_report: TimeControls::UNLIMITED_TIME_DEFAULT,
            avoid_move_until_by_loc_black: Vec::new(),
            avoid_move_until_by_loc_white: Vec::new(),
        }
    }
}

impl GtpEngine {
    /// Construct a real GTP engine from CLI arguments and config.
    #[allow(clippy::too_many_lines)]
    fn new(
        cfg: ConfigParser,
        model_file: String,
        human_model_file: Option<String>,
        mut seed_rand: Rand,
        logger: &'static Logger,
        override_version: Option<&str>,
    ) -> Result<Self, StringError> {
        let load_komi_from_cfg = false;
        let mut initial_rules = setup::load_single_rules(&cfg, load_komi_from_cfg)?;

        if cfg.contains("ignoreGTPAndForceKomi") {
            let forced = cfg
                .get_float(
                    "ignoreGTPAndForceKomi",
                    Rules::MIN_USER_KOMI,
                    Rules::MAX_USER_KOMI,
                )
                .map_err(to_string_error)?;
            initial_rules.set_komi(forced);
        }

        let has_human_model = human_model_file.is_some();
        let mut initial_genmove_params =
            setup::load_single_params_with_human(&cfg, SetupFor::Gtp, has_human_model)?;
        let mut initial_analysis_params = initial_genmove_params.clone();

        if !cfg.contains("conservativePass") {
            initial_genmove_params.conservative_pass = true;
        }
        if !cfg.contains("fillDameBeforePass") {
            initial_genmove_params.fill_dame_before_pass = true;
        }

        let analysis_wide_root_noise = if cfg.contains("analysisWideRootNoise") {
            cfg.get_double("analysisWideRootNoise", 0.0, 5.0)
                .map_err(to_string_error)?
        } else {
            0.04
        };
        let analysis_ignore_pre_root_history = if cfg.contains("analysisIgnorePreRootHistory") {
            cfg.get_bool("analysisIgnorePreRootHistory")
                .map_err(to_string_error)?
        } else {
            true
        };
        let genmove_anti_mirror = if cfg.contains("genmoveAntiMirror") {
            cfg.get_bool("genmoveAntiMirror").map_err(to_string_error)?
        } else if cfg.contains("antiMirror") {
            cfg.get_bool("antiMirror").map_err(to_string_error)?
        } else {
            true
        };

        initial_genmove_params.anti_mirror = genmove_anti_mirror;
        initial_analysis_params.wide_root_noise = analysis_wide_root_noise;
        initial_analysis_params.ignore_pre_root_history = analysis_ignore_pre_root_history;

        let pondering_enabled = if cfg.contains("ponderingEnabled") {
            cfg.get_bool("ponderingEnabled").unwrap_or(false)
        } else {
            false
        };
        let _ = pondering_enabled;

        let cleanup_before_pass = if cfg.contains("cleanupBeforePass") {
            cfg.get_enabled("cleanupBeforePass")
                .map_err(to_string_error)?
        } else {
            kata_core::config::Enabled::Auto
        };
        let friendly_pass = if cfg.contains("friendlyPass") {
            cfg.get_enabled("friendlyPass").map_err(to_string_error)?
        } else {
            kata_core::config::Enabled::Auto
        };

        let allow_resignation = if cfg.contains("allowResignation") {
            cfg.get_bool("allowResignation").unwrap_or(false)
        } else {
            false
        };
        let resign_threshold = if cfg.contains("allowResignation") {
            cfg.get_double("resignThreshold", -1.0, 0.0)
                .map_err(to_string_error)?
        } else {
            -1.0
        };
        let resign_consec_turns = if cfg.contains("resignConsecTurns") {
            cfg.get_int("resignConsecTurns", 1, 100)
                .map_err(to_string_error)?
        } else {
            3
        };
        let resign_min_score_difference = if cfg.contains("resignMinScoreDifference") {
            cfg.get_double("resignMinScoreDifference", 0.0, 1000.0)
                .map_err(to_string_error)?
        } else {
            -1e10
        };
        let resign_min_moves_per_board_area = if cfg.contains("resignMinMovesPerBoardArea") {
            cfg.get_double("resignMinMovesPerBoardArea", 0.0, 1.0)
                .map_err(to_string_error)?
        } else {
            0.0
        };

        setup::initialize_session(&cfg);

        let search_factor_when_winning = if cfg.contains("searchFactorWhenWinning") {
            cfg.get_double("searchFactorWhenWinning", 0.01, 1.0)
                .map_err(to_string_error)?
        } else {
            1.0
        };
        let search_factor_when_winning_threshold =
            if cfg.contains("searchFactorWhenWinningThreshold") {
                cfg.get_double("searchFactorWhenWinningThreshold", 0.0, 1.0)
                    .map_err(to_string_error)?
            } else {
                1.0
            };
        let ogs_chat_to_stderr = if cfg.contains("ogsChatToStderr") {
            cfg.get_bool("ogsChatToStderr").unwrap_or(false)
        } else {
            false
        };
        let log_search_info = if cfg.contains("logSearchInfo") {
            cfg.get_bool("logSearchInfo").unwrap_or(false)
        } else {
            false
        };
        let log_search_info_for_chosen_move = if cfg.contains("logSearchInfoForChosenMove") {
            cfg.get_bool("logSearchInfoForChosenMove").unwrap_or(false)
        } else {
            false
        };
        let analysis_pv_len = if cfg.contains("analysisPVLen") {
            cfg.get_int("analysisPVLen", 1, 1000).unwrap_or(13)
        } else {
            13
        };
        let assume_multiple_starting_black_moves_are_handicap =
            if cfg.contains("assumeMultipleStartingBlackMovesAreHandicap") {
                cfg.get_bool("assumeMultipleStartingBlackMovesAreHandicap")
                    .unwrap_or(true)
            } else {
                true
            };
        let prevent_encore = if cfg.contains("preventCleanupPhase") {
            cfg.get_bool("preventCleanupPhase").unwrap_or(true)
        } else {
            true
        };
        let dynamic_playout_doubling_advantage_cap_per_opp_lead =
            if cfg.contains("dynamicPlayoutDoublingAdvantageCapPerOppLead") {
                cfg.get_double("dynamicPlayoutDoublingAdvantageCapPerOppLead", 0.0, 0.5)
                    .map_err(to_string_error)?
            } else {
                0.045
            };
        let static_pda_takes_precedence = cfg.contains("playoutDoublingAdvantage")
            && !cfg.contains("dynamicPlayoutDoublingAdvantageCapPerOppLead");

        let normal_avoid_repeated_pattern_utility =
            initial_genmove_params.avoid_repeated_pattern_utility;
        let handicap_avoid_repeated_pattern_utility = if cfg.contains("avoidRepeatedPatternUtility")
        {
            initial_genmove_params.avoid_repeated_pattern_utility
        } else {
            0.005
        };

        let delay_move_scale = if cfg.contains("delayMoveScale") {
            cfg.get_double("delayMoveScale", 0.0, 10000.0)
                .unwrap_or(0.0)
        } else {
            0.0
        };
        let delay_move_max = if cfg.contains("delayMoveMax") {
            cfg.get_double("delayMoveMax", 0.0, 1000000.0)
                .unwrap_or(1000000.0)
        } else {
            1000000.0
        };

        let perspective = setup::parse_report_analysis_winrates(&cfg, C_EMPTY)?;

        let mut default_board_x_size = -1;
        let mut default_board_y_size = -1;
        setup::load_default_board_xy_size(
            &cfg,
            logger,
            &mut default_board_x_size,
            &mut default_board_y_size,
        )?;

        let expected_concurrent_evals = initial_genmove_params
            .num_threads
            .max(initial_analysis_params.num_threads);
        let default_max_batch_size = (expected_concurrent_evals + 3) / 4 * 4;
        let default_max_batch_size = default_max_batch_size.max(8);

        let nn_eval = Box::leak(Box::new(setup::initialize_nn_evaluator(
            model_file.clone(),
            model_file.clone(),
            String::new(),
            &cfg,
            logger,
            &mut Rand::new(),
            expected_concurrent_evals,
            default_board_x_size,
            default_board_y_size,
            default_max_batch_size,
            true,
            false,
            SetupFor::Gtp,
        )?));

        let human_eval = if let Some(human_file) = human_model_file.as_ref() {
            Some(Box::leak(Box::new(setup::initialize_nn_evaluator(
                human_file.clone(),
                human_file.clone(),
                String::new(),
                &cfg,
                logger,
                &mut Rand::new(),
                expected_concurrent_evals,
                default_board_x_size,
                default_board_y_size,
                default_max_batch_size,
                true,
                false,
                SetupFor::Gtp,
            )?)) as &'static NnEvaluator)
        } else {
            None
        };

        let search_rand_seed = if cfg.contains("searchRandSeed") {
            cfg.get_string("searchRandSeed").map_err(to_string_error)?
        } else {
            seed_rand.next_u64().to_string()
        };

        let mut bot = AsyncBot::new_with_human(
            initial_genmove_params.clone(),
            nn_eval,
            human_eval,
            logger,
            &search_rand_seed,
        );

        let board_size = if default_board_x_size > 0 && default_board_y_size > 0 {
            (default_board_x_size, default_board_y_size)
        } else {
            (nn_eval.nn_x_len(), nn_eval.nn_y_len())
        };
        let board = Board::new(board_size.0, board_size.1);
        let pla = P_BLACK;
        let hist = BoardHistory::new(board.clone(), pla, initial_rules, 0);
        bot.set_position(pla, &board, &hist);

        let mut engine = Self {
            nn_eval,
            human_eval,
            bot,
            logger,
            cfg: cfg.clone(),
            model_file,
            human_model_file,
            seed_rand,
            current_rules: initial_rules,
            genmove_params: initial_genmove_params.clone(),
            analysis_params: initial_analysis_params.clone(),
            is_genmove_params: true,
            b_time_controls: TimeControls::new(),
            w_time_controls: TimeControls::new(),
            initial_board: board.clone(),
            initial_pla: pla,
            move_history: Vec::new(),
            recent_win_loss_values: Vec::new(),
            last_search_factor: 1.0,
            perspective,
            analysis_pv_len,
            assume_multiple_starting_black_moves_are_handicap,
            prevent_encore,
            dynamic_playout_doubling_advantage_cap_per_opp_lead,
            static_pda_takes_precedence,
            normal_avoid_repeated_pattern_utility,
            handicap_avoid_repeated_pattern_utility,
            delay_move_scale,
            delay_move_max,
            allow_resignation,
            resign_threshold,
            resign_consec_turns,
            resign_min_score_difference,
            resign_min_moves_per_board_area,
            search_factor_when_winning,
            search_factor_when_winning_threshold,
            ogs_chat_to_stderr,
            log_search_info,
            log_search_info_for_chosen_move,
            cleanup_before_pass,
            friendly_pass,
            genmove_timer: ClockTimer::new(),
            genmove_time_sum: 0.0,
            genmove_expected_id: 0,
        };

        engine.set_or_reset_board_size(default_board_x_size, default_board_y_size)?;

        if !cfg.contains("maxPlayouts") && !cfg.contains("maxVisits") && !cfg.contains("maxTime") {
            let tc = TimeControls::canadian_or_byo_yomi_time(1.0, 5.0, 5, 1);
            engine.b_time_controls = tc.clone();
            engine.w_time_controls = tc;
        }

        if let Some(v) = override_version {
            let _ = v;
        }

        Ok(engine)
    }

    /// Construct a minimal engine suitable for unit tests. Uses a dummy NN
    /// evaluator so that no model file is required.
    fn new_for_tests(x_size: i32, y_size: i32) -> Self {
        let logger: &'static Logger = Box::leak(Box::new(Logger::new(
            LoggerOptions {
                log_to_stdout: false,
                log_to_stderr: false,
                log_time: false,
            },
            None,
        )));

        let mut cfg = ConfigParser::new(false, false);
        cfg.initialize_str(
            "numSearchThreads = 1\nmaxVisits = 10\nmaxPlayouts = 10\nnnCacheSizePowerOfTwo = 10\nnnMutexPoolSizePowerOfTwo = 10\n",
        )
        .unwrap();

        let nn_eval = Box::leak(Box::new(NnEvaluator::new(
            "dummy".to_string(),
            "/dev/null".to_string(),
            String::new(),
            Arc::new(logger.clone()),
            8,
            x_size,
            y_size,
            false,
            false,
            10,
            10,
            true,
            String::new(),
            Enabled::False,
            1,
            Vec::new(),
            "test-seed".to_string(),
            false,
            0,
            true,
            &cfg,
        )));

        let params = SearchParams::new();
        let mut bot = AsyncBot::new_with_human(params.clone(), nn_eval, None, logger, "test-seed");

        let rules = Rules::default();
        let board = Board::new(x_size, y_size);
        let hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
        bot.set_position(P_BLACK, &board, &hist);

        Self {
            nn_eval,
            human_eval: None,
            bot,
            logger,
            cfg,
            model_file: "/dev/null".to_string(),
            human_model_file: None,
            seed_rand: Rand::new(),
            current_rules: rules,
            genmove_params: params.clone(),
            analysis_params: params.clone(),
            is_genmove_params: true,
            b_time_controls: TimeControls::new(),
            w_time_controls: TimeControls::new(),
            initial_board: board.clone(),
            initial_pla: P_BLACK,
            move_history: Vec::new(),
            recent_win_loss_values: Vec::new(),
            last_search_factor: 1.0,
            perspective: C_EMPTY,
            analysis_pv_len: 13,
            assume_multiple_starting_black_moves_are_handicap: true,
            prevent_encore: true,
            dynamic_playout_doubling_advantage_cap_per_opp_lead: 0.045,
            static_pda_takes_precedence: false,
            normal_avoid_repeated_pattern_utility: params.avoid_repeated_pattern_utility,
            handicap_avoid_repeated_pattern_utility: 0.005,
            delay_move_scale: 0.0,
            delay_move_max: 1000000.0,
            allow_resignation: false,
            resign_threshold: -0.9,
            resign_consec_turns: 3,
            resign_min_score_difference: -1e10,
            resign_min_moves_per_board_area: 0.0,
            search_factor_when_winning: 1.0,
            search_factor_when_winning_threshold: 1.0,
            ogs_chat_to_stderr: false,
            log_search_info: false,
            log_search_info_for_chosen_move: false,
            cleanup_before_pass: kata_core::config::Enabled::Auto,
            friendly_pass: kata_core::config::Enabled::Auto,
            genmove_timer: ClockTimer::new(),
            genmove_time_sum: 0.0,
            genmove_expected_id: 0,
        }
    }

    fn stop_and_wait(&mut self) {
        self.genmove_expected_id = (self.genmove_expected_id + 1) & 0x3FFFFFFF;
        self.bot.stop_and_wait();
    }

    fn get_current_rules(&self) -> Rules {
        self.current_rules
    }

    fn set_or_reset_board_size(
        &mut self,
        board_x_size: i32,
        board_y_size: i32,
    ) -> Result<(), StringError> {
        let mut board_x_size = board_x_size;
        let mut board_y_size = board_y_size;
        if board_x_size <= 0 || board_y_size <= 0 {
            board_x_size = DEFAULT_LEN as i32;
            board_y_size = DEFAULT_LEN as i32;
        }

        if board_x_size < 2 || board_y_size < 2 {
            return Err(StringError::new("unacceptable size".to_string()));
        }
        if board_x_size > MAX_LEN as i32 || board_y_size > MAX_LEN as i32 {
            return Err(StringError::new(format!(
                "unacceptable size (Board::MAX_LEN is {}, consider increasing and recompiling)",
                MAX_LEN
            )));
        }

        let expected_concurrent_evals = self
            .genmove_params
            .num_threads
            .max(self.analysis_params.num_threads);
        let default_max_batch_size = ((expected_concurrent_evals + 3) / 4 * 4).max(8);

        if self.nn_eval.nn_x_len() != board_x_size || self.nn_eval.nn_y_len() != board_y_size {
            let new_eval = Box::leak(Box::new(setup::initialize_nn_evaluator(
                self.model_file.clone(),
                self.model_file.clone(),
                String::new(),
                &self.cfg,
                self.logger,
                &mut Rand::new(),
                expected_concurrent_evals,
                board_x_size,
                board_y_size,
                default_max_batch_size,
                true,
                false,
                SetupFor::Gtp,
            )?));
            self.nn_eval = new_eval;
        }

        let search_rand_seed = if self.cfg.contains("searchRandSeed") {
            self.cfg
                .get_string("searchRandSeed")
                .map_err(to_string_error)?
        } else {
            self.seed_rand.next_u64().to_string()
        };

        self.bot = AsyncBot::new_with_human(
            self.genmove_params.clone(),
            self.nn_eval,
            self.human_eval,
            self.logger,
            &search_rand_seed,
        );

        let board = Board::new(board_x_size, board_y_size);
        let pla = P_BLACK;
        let hist = BoardHistory::new(board.clone(), pla, self.current_rules, 0);
        self.set_position_and_rules(pla, board.clone(), hist, board.clone(), pla, Vec::new());
        self.recent_win_loss_values.clear();

        Ok(())
    }

    fn set_position_and_rules(
        &mut self,
        pla: Player,
        board: Board,
        hist: BoardHistory,
        new_initial_board: Board,
        new_initial_pla: Player,
        new_move_history: Vec<Move>,
    ) {
        let mut hist = hist;
        hist.set_assume_multiple_starting_black_moves_are_handicap(
            self.assume_multiple_starting_black_moves_are_handicap,
        );
        self.current_rules = hist.rules;
        self.bot.set_position(pla, &board, &hist);
        self.initial_board = new_initial_board;
        self.initial_pla = new_initial_pla;
        self.move_history = new_move_history;
        self.recent_win_loss_values.clear();
    }

    fn clear_board(&mut self) {
        let new_x_size = self.bot.get_root_board().x_size;
        let new_y_size = self.bot.get_root_board().y_size;
        let board = Board::new(new_x_size, new_y_size);
        let pla = P_BLACK;
        let hist = BoardHistory::new(board.clone(), pla, self.current_rules, 0);
        self.set_position_and_rules(
            pla,
            board,
            hist,
            Board::new(new_x_size, new_y_size),
            pla,
            Vec::new(),
        );
    }

    fn update_komi_if_new(&mut self, new_komi: f64) {
        self.bot.set_komi_if_new(new_komi);
        self.current_rules.set_komi(new_komi as f32);
    }

    fn play(&mut self, loc: Loc, pla: Player) -> bool {
        if self
            .bot
            .make_move_with_prevent(loc, pla, self.prevent_encore)
        {
            self.move_history.push(Move::new(loc, pla));
            true
        } else {
            false
        }
    }

    fn undo(&mut self) -> bool {
        if self.move_history.is_empty() {
            return false;
        }
        let history_copy = self.move_history.clone();
        let undone_board = self.initial_board.clone();
        let mut undone_hist = BoardHistory::new(
            undone_board.clone(),
            self.initial_pla,
            self.current_rules,
            0,
        );
        undone_hist.set_initial_turn_number(self.bot.get_root_hist().initial_turn_number);
        self.set_position_and_rules(
            self.initial_pla,
            undone_board,
            undone_hist,
            self.initial_board.clone(),
            self.initial_pla,
            Vec::new(),
        );
        for mv in history_copy.iter().take(history_copy.len() - 1) {
            assert!(self.play(mv.loc, mv.pla));
        }
        true
    }

    fn set_rules_not_including_komi(&mut self, new_rules: Rules) -> Result<(), StringError> {
        let mut new_rules = new_rules;
        new_rules.set_komi(self.current_rules.komi_f32());

        let history_copy = self.move_history.clone();
        let board = self.initial_board.clone();
        let hist = BoardHistory::new(board.clone(), self.initial_pla, new_rules, 0);
        self.set_position_and_rules(
            self.initial_pla,
            board,
            hist,
            self.initial_board.clone(),
            self.initial_pla,
            Vec::new(),
        );

        for mv in &history_copy {
            if !self.play(mv.loc, mv.pla) {
                return Err(StringError::new(
                    "Could not make the rules change, some earlier moves in the game would now become illegal."
                        .to_string(),
                ));
            }
        }
        Ok(())
    }

    fn clear_cache(&mut self) {
        self.bot.clear_search();
        self.bot.clear_eval_cache();
        self.nn_eval.clear_cache();
        if let Some(he) = self.human_eval {
            he.clear_cache();
        }
    }

    fn handle_loadsgf(&mut self, pieces: &[String]) -> String {
        if pieces.is_empty() || pieces.len() > 2 {
            return "? syntax: loadsgf filename [move_number]".to_string();
        }
        let filename = &pieces[0];
        let turn_idx = if pieces.len() == 2 {
            match pieces[1].parse::<i64>() {
                Ok(n) => n,
                Err(_) => return "? move_number must be an integer".to_string(),
            }
        } else {
            -1
        };

        let compact = match CompactSgf::parse_file(filename) {
            Ok(c) => c,
            Err(e) => return format!("? {}", e.0),
        };

        let rules = match compact.get_rules_or_fail_allow_unspecified(&self.current_rules) {
            Ok(r) => r,
            Err(e) => return format!("? {}", e.0),
        };

        let mut board = Board::new(compact.x_size, compact.y_size);
        let mut pla = P_BLACK;
        let mut hist = BoardHistory::new(board.clone(), pla, rules, 0);
        let idx = if turn_idx < 0 {
            compact.moves.len() as i64
        } else {
            turn_idx
        };
        if idx < 0 || idx > compact.moves.len() as i64 {
            return "? move_number out of range".to_string();
        }

        if let Err(e) =
            compact.setup_board_and_hist_assume_legal(&rules, &mut board, &mut pla, &mut hist, idx)
        {
            return format!("? {}", e.0);
        }

        let move_history = compact.moves[..idx as usize].to_vec();
        self.set_position_and_rules(pla, board.clone(), hist, board, pla, move_history);
        "= ".to_string()
    }

    fn handle_printsgf(&mut self, pieces: &[String]) -> String {
        if pieces.len() > 1 {
            return "? syntax: printsgf [filename]".to_string();
        }

        self.stop_and_wait();
        let hist = self.bot.get_root_hist().clone();
        let mut sgf = String::new();
        write_sgf(&mut sgf, "Black", "White", &hist);

        if pieces.len() == 1 {
            let filename = &pieces[0];
            if let Err(e) = std::fs::write(filename, &sgf) {
                return format!("? could not write SGF file: {e}");
            }
            "= ".to_string()
        } else {
            format!("= {}", sgf)
        }
    }

    fn compute_anticipated_winner_and_score(&mut self) -> (Player, f64) {
        self.stop_and_wait();

        let mut tmp_params = self.genmove_params.clone();
        tmp_params.playout_doubling_advantage = 0.0;
        tmp_params.conservative_pass = true;
        tmp_params.human_sl_chosen_move_prop = 0.0;
        tmp_params.human_sl_root_explore_prob_weightful = 0.0;
        tmp_params.human_sl_root_explore_prob_weightless = 0.0;
        tmp_params.human_sl_pla_explore_prob_weightful = 0.0;
        tmp_params.human_sl_pla_explore_prob_weightless = 0.0;
        tmp_params.human_sl_opp_explore_prob_weightful = 0.0;
        tmp_params.human_sl_opp_explore_prob_weightless = 0.0;
        tmp_params.anti_mirror = false;
        tmp_params.avoid_repeated_pattern_utility = 0.0;
        self.bot.set_params(&tmp_params);

        let old_pla = self.bot.get_root_pla();
        let old_board = self.bot.get_root_board().clone();
        let old_hist = self.bot.get_root_hist().clone();
        let hist = self.bot.get_root_hist();

        let (winner, score) = if hist.is_game_finished
            && ((hist.rules.scoring_rule == kata_game::rules::ScoringRule::Area
                && !hist.rules.friendly_pass_ok)
                || hist.rules.scoring_rule == kata_game::rules::ScoringRule::Territory)
        {
            (hist.winner, f64::from(hist.final_white_minus_black_score))
        } else {
            let lead = 0.0f32;
            let lead = if hist.rules.game_result_will_be_integer() {
                lead.round()
            } else {
                (lead + 0.5).round() - 0.5
            };
            let winner = if lead > 0.0 {
                P_WHITE
            } else if lead < 0.0 {
                P_BLACK
            } else {
                C_EMPTY
            };
            (winner, f64::from(lead))
        };

        self.bot.set_position(old_pla, &old_board, &old_hist);
        self.bot.set_params(&self.genmove_params);
        self.is_genmove_params = true;

        (winner, score)
    }

    fn compute_anticipated_statuses(&mut self) -> Vec<bool> {
        self.stop_and_wait();

        let mut tmp_params = self.genmove_params.clone();
        tmp_params.playout_doubling_advantage = 0.0;
        tmp_params.conservative_pass = true;
        tmp_params.human_sl_chosen_move_prop = 0.0;
        tmp_params.human_sl_root_explore_prob_weightful = 0.0;
        tmp_params.human_sl_root_explore_prob_weightless = 0.0;
        tmp_params.human_sl_pla_explore_prob_weightful = 0.0;
        tmp_params.human_sl_pla_explore_prob_weightless = 0.0;
        tmp_params.human_sl_opp_explore_prob_weightful = 0.0;
        tmp_params.human_sl_opp_explore_prob_weightless = 0.0;
        tmp_params.anti_mirror = false;
        tmp_params.avoid_repeated_pattern_utility = 0.0;
        self.bot.set_params(&tmp_params);

        let old_pla = self.bot.get_root_pla();
        let old_board = self.bot.get_root_board().clone();
        let old_hist = self.bot.get_root_hist().clone();

        let is_alive = play_utils::compute_anticipated_statuses_simple(&old_board, &old_hist);

        self.bot.set_position(old_pla, &old_board, &old_hist);
        self.bot.set_params(&self.genmove_params);
        self.is_genmove_params = true;

        is_alive
    }

    fn raw_nn(&self, which_symmetry: i32, policy_optimism: f64, use_human_model: bool) -> String {
        let nn_eval_to_use = if use_human_model {
            self.human_eval.unwrap_or(self.nn_eval)
        } else {
            self.nn_eval
        };

        let mut out = String::new();
        for symmetry in 0..NUM_SYMMETRIES {
            if which_symmetry != nn_inputs::SYMMETRY_ALL && which_symmetry != symmetry {
                continue;
            }
            let board = self.bot.get_root_board().clone();
            let hist = self.bot.get_root_hist().clone();
            let next_pla = self.bot.get_root_pla();

            let nn_input_params = MiscNNInputParams {
                playout_doubling_advantage: if self.analysis_params.playout_doubling_advantage_pla
                    == C_EMPTY
                    || self.analysis_params.playout_doubling_advantage_pla == next_pla
                {
                    self.analysis_params.playout_doubling_advantage
                } else {
                    -self.analysis_params.playout_doubling_advantage
                },
                symmetry,
                policy_optimism,
                ..Default::default()
            };

            let mut buf = kata_nn::backend::NNResultBuf::new();
            nn_eval_to_use.evaluate(
                &board,
                &hist,
                next_pla,
                &nn_input_params,
                &mut buf,
                true,
                true,
            );
            let Some(nn_output) = buf.result else {
                continue;
            };

            out.push_str(&format!("symmetry {}\n", symmetry));
            out.push_str(&format!("whiteWin {:.6}\n", nn_output.white_win_prob));
            out.push_str(&format!("whiteLoss {:.6}\n", nn_output.white_loss_prob));
            out.push_str(&format!("noResult {:.6}\n", nn_output.white_no_result_prob));
            out.push_str(&format!("whiteLead {:.3}\n", nn_output.white_lead));
            out.push_str(&format!(
                "whiteScoreSelfplay {:.3}\n",
                nn_output.white_score_mean
            ));
            out.push_str(&format!(
                "whiteScoreSelfplaySq {:.3}\n",
                nn_output.white_score_mean_sq
            ));
            out.push_str(&format!("varTimeLeft {:.3}\n", nn_output.var_time_left));
            out.push_str(&format!(
                "shorttermWinlossError {:.3}\n",
                nn_output.shortterm_winloss_error
            ));
            out.push_str(&format!(
                "shorttermScoreError {:.3}\n",
                nn_output.shortterm_score_error
            ));

            out.push_str("policy\n");
            for y in 0..board.y_size {
                for x in 0..board.x_size {
                    let pos = nn_pos::xy_to_pos(x, y, nn_output.nn_x_len);
                    let prob = nn_output.policy_probs[pos as usize];
                    if prob < 0.0 {
                        out.push_str("    NAN ");
                    } else {
                        out.push_str(&format!("{:8.6} ", prob));
                    }
                }
                out.push('\n');
            }
            out.push_str("policyPass ");
            let pass_pos = nn_pos::loc_to_pos(
                PASS_LOC,
                board.x_size,
                nn_output.nn_x_len,
                nn_output.nn_y_len,
            );
            let pass_prob = nn_output.policy_probs[pass_pos as usize];
            if pass_prob < 0.0 {
                out.push_str("    NAN ");
            } else {
                out.push_str(&format!("{:8.6} ", pass_prob));
            }
            out.push('\n');

            out.push_str("whiteOwnership\n");
            if let Some(owner_map) = nn_output.white_owner_map.as_ref() {
                for y in 0..board.y_size {
                    for x in 0..board.x_size {
                        let pos = nn_pos::xy_to_pos(x, y, nn_output.nn_x_len);
                        out.push_str(&format!("{:9.7} ", owner_map[pos as usize]));
                    }
                    out.push('\n');
                }
            }
            out.push('\n');
        }
        out.trim_end().to_string()
    }
}

// ---------------------------------------------------------------------------
// Command dispatch
// ---------------------------------------------------------------------------

impl GtpEngine {
    /// Run the GTP command loop.
    fn run<R: BufRead>(&mut self, input: R, output: OutputHandle) -> io::Result<()> {
        let mut lines = input.lines();
        let mut currently_genmoving = false;
        let mut currently_analyzing = false;

        loop {
            let line = match lines.next() {
                Some(Ok(l)) => l,
                Some(Err(e)) => return Err(e),
                None => break,
            };

            let line = process_single_command_line(&line);

            if currently_analyzing {
                currently_analyzing = false;
                self.stop_and_wait();
                writeln!(output.lock().unwrap())?;
            }
            if currently_genmoving {
                currently_genmoving = false;
                self.stop_and_wait();
            }

            if line.is_empty() {
                continue;
            }

            let (command, pieces, has_id, id) = parse_command_line(&line);
            if command.is_empty() {
                Self::write_response(&output, has_id, id, true, "empty command")?;
                continue;
            }

            let mut response = String::new();
            let mut response_is_error = false;
            let mut suppress_response = false;
            let mut should_quit = false;
            let mut maybe_start_pondering = false;

            match command.as_str() {
                "protocol_version" => response = "2".to_string(),
                "name" => response = "KataGo".to_string(),
                "version" => response = "KataGo-Rust-0.1.0".to_string(),
                "known_command" => {
                    if pieces.len() != 1 {
                        response_is_error = true;
                        response = format!(
                            "Expected single argument for known_command but got '{}'",
                            global::concat_vec(&pieces, " ")
                        );
                    } else {
                        response = if KNOWN_COMMANDS.contains(&pieces[0].as_str()) {
                            "true".to_string()
                        } else {
                            "false".to_string()
                        };
                    }
                }
                "list_commands" => {
                    response = KNOWN_COMMANDS.join("\n");
                }
                "quit" => {
                    should_quit = true;
                    self.logger.write("Quit requested by controller");
                }
                "boardsize" | "rectangular_boardsize" => match parse_board_size(&pieces) {
                    Ok((sx, sy)) => {
                        if let Err(e) = self.set_or_reset_board_size(sx, sy) {
                            response_is_error = true;
                            response = e.message;
                        }
                    }
                    Err(msg) => {
                        response_is_error = true;
                        response = msg;
                    }
                },
                "clear_board" => self.clear_board(),
                "komi" => {
                    if pieces.len() != 1 || global::try_string_to_float(&pieces[0]).is_none() {
                        response_is_error = true;
                        response = format!(
                            "Expected single float argument for komi but got '{}'",
                            global::concat_vec(&pieces, " ")
                        );
                    } else {
                        let new_komi = global::try_string_to_float(&pieces[0]).unwrap();
                        if !new_komi.is_finite()
                            || !(Rules::MIN_USER_KOMI..=Rules::MAX_USER_KOMI).contains(&new_komi)
                        {
                            response_is_error = true;
                            response = "unacceptable komi".to_string();
                        } else if !Rules::komi_is_int_or_half_int(new_komi) {
                            response_is_error = true;
                            response = "komi must be an integer or half-integer".to_string();
                        } else {
                            self.update_komi_if_new(f64::from(new_komi));
                            maybe_start_pondering = !self.move_history.is_empty();
                        }
                    }
                }
                "get_komi" => {
                    response = global::double_to_string(f64::from(self.current_rules.komi_f32()));
                }
                "kata-get-rules" => {
                    if pieces.is_empty() {
                        response = self.current_rules.to_json_no_komi().to_string();
                    } else {
                        response_is_error = true;
                        response = format!(
                            "Expected no arguments for kata-get-rules but got '{}'",
                            global::concat_vec(&pieces, " ")
                        );
                    }
                }
                "kata-set-rules" => {
                    let rest = global::concat_vec(&pieces, " ");
                    match Rules::try_parse_rules_without_komi(&rest, self.current_rules.komi_f32())
                    {
                        Some(new_rules) => {
                            if let Err(e) = self.set_rules_not_including_komi(new_rules) {
                                response_is_error = true;
                                response = e.message;
                            } else {
                                self.logger.write(&format!(
                                    "Changed rules to {}",
                                    new_rules.to_legacy_string_no_komi_maybe_nice()
                                ));
                            }
                        }
                        None => {
                            response_is_error = true;
                            response = format!("Unknown rules '{}'", rest);
                        }
                    }
                }
                "kata-set-rule" => {
                    if pieces.len() != 2 {
                        response_is_error = true;
                        response = format!(
                            "Expected two arguments for kata-set-rule but got '{}'",
                            global::concat_vec(&pieces, " ")
                        );
                    } else {
                        match Rules::update_rules(&pieces[0], &pieces[1], &self.current_rules) {
                            Ok(new_rules) => {
                                if let Err(e) = self.set_rules_not_including_komi(new_rules) {
                                    response_is_error = true;
                                    response = e.message;
                                } else {
                                    self.logger.write(&format!(
                                        "Changed rules to {}",
                                        new_rules.to_legacy_string_no_komi_maybe_nice()
                                    ));
                                }
                            }
                            Err(e) => {
                                response_is_error = true;
                                response = e.0;
                            }
                        }
                    }
                }
                "kgs-rules" => {
                    if pieces.is_empty() {
                        response_is_error = true;
                        response = "Expected one argument kgs-rules".to_string();
                    } else {
                        let s = global::to_lower(global::trim(&pieces[0]));
                        let rules_str = match s.as_str() {
                            "chinese" => Some("chinese-ogs"),
                            "aga" => Some("aga"),
                            "new_zealand" => Some("new_zealand"),
                            "japanese" => Some("japanese"),
                            _ => None,
                        };
                        match rules_str {
                            Some(r) => {
                                let new_rules = Rules::try_parse_rules_without_komi(
                                    r,
                                    self.current_rules.komi_f32(),
                                )
                                .unwrap();
                                if let Err(e) = self.set_rules_not_including_komi(new_rules) {
                                    response_is_error = true;
                                    response = e.message;
                                } else {
                                    self.logger.write(&format!(
                                        "Changed rules to {}",
                                        new_rules.to_legacy_string_no_komi_maybe_nice()
                                    ));
                                }
                            }
                            None => {
                                response_is_error = true;
                                response = format!("Unknown rules '{}'", s);
                            }
                        }
                    }
                }
                "kata-list-params" => {
                    response = "analysisWideRootNoise analysisIgnorePreRootHistory genmoveAntiMirror antiMirror humanSLProfile allowResignation ponderingEnabled delayMoveScale delayMoveMax".to_string();
                }
                "kata-get-param" => {
                    if pieces.len() != 1 {
                        response_is_error = true;
                        response = format!(
                            "Expected one argument for kata-get-param but got '{}'",
                            global::concat_vec(&pieces, " ")
                        );
                    } else {
                        match pieces[0].as_str() {
                            "analysisWideRootNoise" => {
                                response =
                                    global::double_to_string(self.analysis_params.wide_root_noise)
                            }
                            "analysisIgnorePreRootHistory" => {
                                response = global::bool_to_string(
                                    self.analysis_params.ignore_pre_root_history,
                                )
                            }
                            "genmoveAntiMirror" => {
                                response = global::bool_to_string(self.genmove_params.anti_mirror)
                            }
                            "antiMirror" => {
                                response = global::bool_to_string(self.analysis_params.anti_mirror)
                            }
                            "humanSLProfile" => response = String::new(),
                            "allowResignation" => {
                                response = global::bool_to_string(self.allow_resignation)
                            }
                            "ponderingEnabled" => response = "false".to_string(),
                            "delayMoveScale" => {
                                response = global::double_to_string(self.delay_move_scale)
                            }
                            "delayMoveMax" => {
                                response = global::double_to_string(self.delay_move_max)
                            }
                            _ => {
                                response_is_error = true;
                                response = format!("Invalid parameter: {}", pieces[0]);
                            }
                        }
                    }
                }
                "kata-get-models" => {
                    response = "[]".to_string();
                }
                "kata-get-params" => {
                    response = "{}".to_string();
                }
                "kata-set-param" | "kata-set-params" => {
                    response = "? Set param not implemented".to_string();
                    response_is_error = true;
                }
                "time_settings" => match parse_time_settings(&pieces) {
                    Ok(tc) => {
                        self.b_time_controls = tc.clone();
                        self.w_time_controls = tc;
                    }
                    Err(msg) => {
                        response_is_error = true;
                        response = msg;
                    }
                },
                "kgs-time_settings" | "kata-time_settings" => {
                    match parse_kgs_time_settings(&pieces, command == "kata-time_settings") {
                        Ok(Some(tc)) => {
                            self.b_time_controls = tc.clone();
                            self.w_time_controls = tc;
                        }
                        Ok(None) => {}
                        Err(msg) => {
                            response_is_error = true;
                            response = msg;
                        }
                    }
                }
                "kata-list_time_settings" => {
                    response = "none absolute byoyomi canadian fischer fischer-capped".to_string();
                }
                "time_left" => {
                    if pieces.len() != 3
                        || player_io::try_parse_player(&pieces[0]).is_none()
                        || global::try_string_to_double(&pieces[1]).is_none()
                        || global::try_string_to_int(&pieces[2]).is_none()
                    {
                        response_is_error = true;
                        response = format!(
                            "Expected player and float time and int stones for time_left but got '{}'",
                            global::concat_vec(&pieces, " ")
                        );
                    } else {
                        let pla = player_io::try_parse_player(&pieces[0]).unwrap();
                        let time = global::try_string_to_double(&pieces[1]).unwrap();
                        let stones = global::try_string_to_int(&pieces[2]).unwrap();
                        if !time.is_finite()
                            || !(-10.0..=TimeControls::MAX_USER_INPUT_TIME).contains(&time)
                        {
                            response_is_error = true;
                            response = "invalid time".to_string();
                        } else if !(0..=100000).contains(&stones) {
                            response_is_error = true;
                            response = "invalid stones".to_string();
                        } else {
                            let mut tc = if pla == P_BLACK {
                                self.b_time_controls.clone()
                            } else {
                                self.w_time_controls.clone()
                            };
                            if stones > 0 && tc.original_num_periods <= 0 {
                                response_is_error = true;
                                response = "stones left in period is > 0 but the time control used does not have any overtime periods".to_string();
                            } else {
                                if stones == 0 {
                                    tc.main_time_left = time;
                                    tc.in_overtime = false;
                                    tc.num_periods_left_including_current = tc.original_num_periods;
                                    tc.num_stones_left_in_period = 0;
                                    tc.time_left_in_period = 0.0;
                                } else if tc.original_num_periods > 1
                                    && tc.num_stones_per_period == 1
                                {
                                    tc.main_time_left = 0.0;
                                    tc.in_overtime = true;
                                    tc.num_periods_left_including_current =
                                        stones.min(tc.original_num_periods);
                                    tc.num_stones_left_in_period = 1;
                                    tc.time_left_in_period = time;
                                } else {
                                    tc.main_time_left = 0.0;
                                    tc.in_overtime = true;
                                    tc.num_periods_left_including_current = 1;
                                    tc.num_stones_left_in_period =
                                        stones.min(tc.num_stones_per_period);
                                    tc.time_left_in_period = time;
                                }
                                if pla == P_BLACK {
                                    self.b_time_controls = tc;
                                } else {
                                    self.w_time_controls = tc;
                                }
                                maybe_start_pondering = !self.move_history.is_empty();
                            }
                        }
                    }
                }
                "kata-debug-print-tc" => {
                    let black = format!(
                        "Black mainTime={} inOvertime={} periods={} stones={} perPeriod={}",
                        self.b_time_controls.main_time_left,
                        self.b_time_controls.in_overtime,
                        self.b_time_controls.num_periods_left_including_current,
                        self.b_time_controls.num_stones_left_in_period,
                        self.b_time_controls.per_period_time
                    );
                    let white = format!(
                        "White mainTime={} inOvertime={} periods={} stones={} perPeriod={}",
                        self.w_time_controls.main_time_left,
                        self.w_time_controls.in_overtime,
                        self.w_time_controls.num_periods_left_including_current,
                        self.w_time_controls.num_stones_left_in_period,
                        self.w_time_controls.per_period_time
                    );
                    response = format!("{}\n{}", black, white);
                }
                "play" => {
                    if pieces.len() != 2 {
                        response_is_error = true;
                        response = format!(
                            "Expected two arguments for play but got '{}'",
                            global::concat_vec(&pieces, " ")
                        );
                    } else if let Some(pla) = player_io::try_parse_player(&pieces[0]) {
                        if let Some(loc) = location::try_of_string(
                            &pieces[1],
                            self.bot.get_root_board().x_size,
                            self.bot.get_root_board().y_size,
                        ) {
                            if self.play(loc, pla) {
                                maybe_start_pondering = true;
                            } else {
                                response_is_error = true;
                                response = "illegal move".to_string();
                            }
                        } else {
                            response_is_error = true;
                            response = format!("Could not parse vertex: '{}'", pieces[1]);
                        }
                    } else {
                        response_is_error = true;
                        response = format!("Could not parse color: '{}'", pieces[0]);
                    }
                }
                "set_position" => {
                    if pieces.len() % 2 != 0 {
                        response_is_error = true;
                        response = format!(
                            "Expected a space-separated sequence of <COLOR> <VERTEX> pairs but got '{}'",
                            global::concat_vec(&pieces, " ")
                        );
                    } else {
                        let mut stones = Vec::new();
                        let mut ok = true;
                        for chunk in pieces.chunks(2) {
                            let color = &chunk[0];
                            let vertex = &chunk[1];
                            if let Some(pla) = player_io::try_parse_player(color) {
                                if let Some(loc) = location::try_of_string(
                                    vertex,
                                    self.bot.get_root_board().x_size,
                                    self.bot.get_root_board().y_size,
                                ) {
                                    if loc == PASS_LOC {
                                        response_is_error = true;
                                        response = format!("Could not parse vertex: '{}'", vertex);
                                        ok = false;
                                        break;
                                    }
                                    stones.push(Move::new(loc, pla));
                                } else {
                                    response_is_error = true;
                                    response = format!("Could not parse vertex: '{}'", vertex);
                                    ok = false;
                                    break;
                                }
                            } else {
                                response_is_error = true;
                                response = format!("Could not parse color: '{}'", color);
                                ok = false;
                                break;
                            }
                        }
                        if ok {
                            let mut board = Board::new(
                                self.bot.get_root_board().x_size,
                                self.bot.get_root_board().y_size,
                            );
                            if board.set_stones_fail_if_no_libs(&stones) {
                                let pla = P_BLACK;
                                let mut hist =
                                    BoardHistory::new(board.clone(), pla, self.current_rules, 0);
                                hist.set_initial_turn_number(board.num_stones_on_board() as i64);
                                self.set_position_and_rules(
                                    pla,
                                    board.clone(),
                                    hist,
                                    board.clone(),
                                    pla,
                                    Vec::new(),
                                );
                            } else {
                                response_is_error = true;
                                response = "Illegal stone placements - overlapping stones or stones with no liberties?".to_string();
                            }
                        }
                    }
                }
                "undo" => {
                    if !self.undo() {
                        response_is_error = true;
                        response = "cannot undo".to_string();
                    }
                }
                "genmove"
                | "genmove_debug"
                | "kata-search"
                | "kata-search_cancellable"
                | "kata-search_debug" => {
                    if pieces.len() != 1 {
                        response_is_error = true;
                        response = format!(
                            "Expected one argument for {} but got '{}'",
                            command,
                            global::concat_vec(&pieces, " ")
                        );
                    } else if let Some(pla) = player_io::try_parse_player(&pieces[0]) {
                        let debug = command == "genmove_debug" || command == "kata-search_debug";
                        let play_chosen_move = command == "genmove" || command == "genmove_debug";
                        let gargs = self.build_genmove_args(debug);

                        let (move_str, move_loc) =
                            self.gen_move(pla, &gargs, &AnalyzeArgs::default());
                        Self::write_response(&output, has_id, id, false, &move_str)?;
                        suppress_response = true;

                        if play_chosen_move && move_loc != NULL_LOC {
                            self.play(move_loc, pla);
                            maybe_start_pondering = true;
                        }
                    } else {
                        response_is_error = true;
                        response = format!("Could not parse color: '{}'", pieces[0]);
                    }
                }
                "genmove_analyze"
                | "lz-genmove_analyze"
                | "kata-genmove_analyze"
                | "kata-search_analyze"
                | "kata-search_analyze_cancellable" => {
                    let mut pla = self.bot.get_root_pla();
                    let (args, parse_failed) = parse_analyze_command(&command, &pieces, &mut pla);
                    if parse_failed {
                        response_is_error = true;
                        response = format!(
                            "Could not parse genmove_analyze arguments or arguments out of range: '{}'",
                            global::concat_vec(&pieces, " ")
                        );
                    } else {
                        let play_chosen_move = command == "genmove_analyze"
                            || command == "lz-genmove_analyze"
                            || command == "kata-genmove_analyze";
                        let gargs = self.build_genmove_args(false);

                        Self::write_response(&output, has_id, id, false, "")?;
                        suppress_response = true;

                        let out = Arc::clone(&output);
                        let mut analyze_out = String::new();
                        let (move_str, move_loc) =
                            self.gen_move_analyze(pla, &gargs, &args, &mut analyze_out);
                        if !analyze_out.is_empty() {
                            write!(out.lock().unwrap(), "{}", analyze_out)?;
                        }
                        let final_response = if play_chosen_move {
                            format!("play {}", move_str)
                        } else {
                            move_str
                        };
                        writeln!(output.lock().unwrap(), "{}", final_response)?;
                        writeln!(output.lock().unwrap())?;

                        if play_chosen_move && move_loc != NULL_LOC {
                            self.play(move_loc, pla);
                            maybe_start_pondering = true;
                        }
                    }
                }
                "analyze" | "lz-analyze" | "kata-analyze" => {
                    let mut pla = self.bot.get_root_pla();
                    let (args, parse_failed) = parse_analyze_command(&command, &pieces, &mut pla);
                    if parse_failed {
                        response_is_error = true;
                        response = format!(
                            "Could not parse analyze arguments or arguments out of range: '{}'",
                            global::concat_vec(&pieces, " ")
                        );
                    } else {
                        Self::write_response(&output, has_id, id, false, "")?;
                        suppress_response = true;
                        let mut analyze_out = String::new();
                        self.run_analyze(pla, &args, &mut analyze_out);
                        if !analyze_out.is_empty() {
                            write!(output.lock().unwrap(), "{}", analyze_out)?;
                        }
                        writeln!(output.lock().unwrap())?;
                        currently_analyzing = true;
                    }
                }
                "clear_cache" => self.clear_cache(),
                "showboard" => {
                    response = self.showboard();
                }
                "fixed_handicap" => {
                    if pieces.len() != 1 || global::try_string_to_int(&pieces[0]).is_none() {
                        response_is_error = true;
                        response = format!(
                            "Expected one argument for fixed_handicap but got '{}'",
                            global::concat_vec(&pieces, " ")
                        );
                    } else {
                        let n = global::try_string_to_int(&pieces[0]).unwrap();
                        if n < 2 {
                            response_is_error = true;
                            response =
                                format!("Number of handicap stones less than 2: '{}'", pieces[0]);
                        } else if !self.bot.get_root_board().is_empty() {
                            response_is_error = true;
                            response = "Board is not empty".to_string();
                        } else {
                            let mut board = Board::new(
                                self.bot.get_root_board().x_size,
                                self.bot.get_root_board().y_size,
                            );
                            match play_utils::place_fixed_handicap(&mut board, n) {
                                Ok(()) => {
                                    let mut hist = BoardHistory::new(
                                        board.clone(),
                                        P_WHITE,
                                        self.current_rules,
                                        0,
                                    );
                                    hist.set_assume_multiple_starting_black_moves_are_handicap(
                                        self.assume_multiple_starting_black_moves_are_handicap,
                                    );
                                    hist.set_initial_turn_number(board.num_stones_on_board() as i64);
                                    self.set_position_and_rules(
                                        P_WHITE,
                                        board.clone(),
                                        hist,
                                        board.clone(),
                                        P_WHITE,
                                        Vec::new(),
                                    );
                                    response = self.handicap_stones_response(&board);
                                }
                                Err(e) => {
                                    response_is_error = true;
                                    response = format!("{}, try place_free_handicap", e.message);
                                }
                            }
                        }
                    }
                }
                "place_free_handicap" => {
                    if pieces.len() != 1 || global::try_string_to_int(&pieces[0]).is_none() {
                        response_is_error = true;
                        response = format!(
                            "Expected one argument for place_free_handicap but got '{}'",
                            global::concat_vec(&pieces, " ")
                        );
                    } else {
                        let n = global::try_string_to_int(&pieces[0]).unwrap();
                        if n < 2 {
                            response_is_error = true;
                            response =
                                format!("Number of handicap stones less than 2: '{}'", pieces[0]);
                        } else if !self.bot.get_root_board().is_empty() {
                            response_is_error = true;
                            response = "Board is not empty".to_string();
                        } else {
                            let mut board = Board::new(
                                self.bot.get_root_board().x_size,
                                self.bot.get_root_board().y_size,
                            );
                            let mut hist =
                                BoardHistory::new(board.clone(), P_BLACK, self.current_rules, 0);
                            let mut rand = Rand::new();
                            play_utils::play_extra_black(
                                self.bot.get_search_stop_and_wait(),
                                n,
                                &mut board,
                                &mut hist,
                                0.25,
                                &mut rand,
                            );
                            let mut new_hist =
                                BoardHistory::new(board.clone(), P_WHITE, self.current_rules, 0);
                            new_hist.set_assume_multiple_starting_black_moves_are_handicap(
                                self.assume_multiple_starting_black_moves_are_handicap,
                            );
                            new_hist.set_initial_turn_number(board.num_stones_on_board() as i64);
                            self.set_position_and_rules(
                                P_WHITE,
                                board.clone(),
                                new_hist,
                                board.clone(),
                                P_WHITE,
                                Vec::new(),
                            );
                            response = self.handicap_stones_response(&board);
                        }
                    }
                }
                "set_free_handicap" => {
                    if !self.bot.get_root_board().is_empty() {
                        response_is_error = true;
                        response = "Board is not empty".to_string();
                    } else {
                        let mut board = Board::new(
                            self.bot.get_root_board().x_size,
                            self.bot.get_root_board().y_size,
                        );
                        let mut stones = Vec::new();
                        let mut ok = true;
                        for s in &pieces {
                            if let Some(loc) =
                                location::try_of_string(s, board.x_size, board.y_size)
                            {
                                if loc == PASS_LOC {
                                    response_is_error = true;
                                    response = format!("Invalid handicap location: {}", s);
                                    ok = false;
                                    break;
                                }
                                stones.push(Move::new(loc, P_BLACK));
                            } else {
                                response_is_error = true;
                                response = format!("Invalid handicap location: {}", s);
                                ok = false;
                                break;
                            }
                        }
                        if ok {
                            if board.set_stones_fail_if_no_libs(&stones) {
                                let mut hist = BoardHistory::new(
                                    board.clone(),
                                    P_WHITE,
                                    self.current_rules,
                                    0,
                                );
                                hist.set_initial_turn_number(board.num_stones_on_board() as i64);
                                self.set_position_and_rules(
                                    P_WHITE,
                                    board.clone(),
                                    hist,
                                    board.clone(),
                                    P_WHITE,
                                    Vec::new(),
                                );
                            } else {
                                response_is_error = true;
                                response = "Handicap placement is invalid".to_string();
                            }
                        }
                    }
                }
                "final_score" => {
                    let (winner, score) = self.compute_anticipated_winner_and_score();
                    response = match winner {
                        C_EMPTY => "0".to_string(),
                        P_BLACK => format!("B+{:.1}", -score),
                        P_WHITE => format!("W+{:.1}", score),
                        _ => "0".to_string(),
                    };
                }
                "final_status_list" => {
                    if pieces.len() != 1 {
                        response_is_error = true;
                        response = format!(
                            "Expected one argument for final_status_list but got '{}'",
                            global::concat_vec(&pieces, " ")
                        );
                    } else {
                        let status_mode = match pieces[0].as_str() {
                            "alive" => Some(0),
                            "seki" => Some(1),
                            "dead" => Some(2),
                            _ => None,
                        };
                        if let Some(mode) = status_mode {
                            let is_alive = self.compute_anticipated_statuses();
                            let board = self.bot.get_root_board().clone();
                            let mut locs = Vec::new();
                            for y in 0..board.y_size {
                                for x in 0..board.x_size {
                                    let loc = location::get_loc(x, y, board.x_size);
                                    let color = board.colors[loc as usize];
                                    if color != C_EMPTY {
                                        if (mode == 0 && is_alive[loc as usize])
                                            || (mode == 2 && !is_alive[loc as usize])
                                        {
                                            locs.push(loc);
                                        }
                                    }
                                }
                            }
                            response = locs
                                .iter()
                                .map(|&loc| location::to_string(loc, board.x_size, board.y_size))
                                .collect::<Vec<_>>()
                                .join(" ");
                        } else {
                            response_is_error = true;
                            response =
                                "Argument to final_status_list must be 'alive' or 'seki' or 'dead'"
                                    .to_string();
                        }
                    }
                }
                "loadsgf" => {
                    response = self.handle_loadsgf(&pieces);
                    response_is_error = response.starts_with('?');
                }
                "printsgf" => {
                    response = self.handle_printsgf(&pieces);
                    response_is_error = response.starts_with('?');
                }
                "kata-raw-nn" => {
                    response = self.handle_raw_nn(&pieces, false);
                    if response.starts_with('?') {
                        response_is_error = true;
                    }
                }
                "kata-raw-human-nn" => {
                    if self.human_eval.is_none() {
                        response_is_error = true;
                        response = "Cannot run kata-raw-human-nn, -human-model was not provided"
                            .to_string();
                    } else {
                        response = self.handle_raw_nn(&pieces, true);
                        if response.starts_with('?') {
                            response_is_error = true;
                        }
                    }
                }
                "debug_moves" => {
                    response = "? debug_moves not implemented".to_string();
                    response_is_error = true;
                }
                "cputime" | "gomill-cpu_time" => {
                    response = global::double_to_string(self.genmove_time_sum);
                }
                "kata-benchmark" => {
                    response = "? kata-benchmark not implemented".to_string();
                    response_is_error = true;
                }
                "kata-list-colors-own-eyes" => {
                    response = "? kata-list-colors-own-eyes not implemented".to_string();
                    response_is_error = true;
                }
                "stop" => {
                    self.stop_and_wait();
                }
                _ => {
                    response_is_error = true;
                    response = "unknown command".to_string();
                }
            }

            if !suppress_response {
                Self::write_response(&output, has_id, id, response_is_error, &response)?;
            }

            if should_quit {
                break;
            }

            let _ = maybe_start_pondering;
        }

        if currently_analyzing {
            self.stop_and_wait();
            writeln!(output.lock().unwrap())?;
        }
        if currently_genmoving {
            self.stop_and_wait();
        }

        Ok(())
    }

    fn write_response(
        output: &OutputHandle,
        has_id: bool,
        id: i32,
        is_error: bool,
        response: &str,
    ) -> io::Result<()> {
        let mut out = output.lock().unwrap();
        if is_error {
            write!(out, "? ")?;
        } else {
            write!(out, "= ")?;
        }
        if has_id {
            write!(out, "{} ", id)?;
        }
        writeln!(out, "{}", response)?;
        writeln!(out)?;
        Ok(())
    }

    fn showboard(&self) -> String {
        let board = self.bot.get_root_board();
        let hist = self.bot.get_root_hist();
        let mut out = String::new();
        out.push_str(&format!(
            "Root player: {}  Next player: {}  Rules: {}  Komi: {}\n",
            player_io::player_to_string(self.initial_pla),
            player_io::player_to_string(hist.presumed_next_move_pla),
            hist.rules.to_legacy_string_no_komi_maybe_nice(),
            hist.rules.komi_f32()
        ));
        out.push_str("  ");
        for x in 0..board.x_size {
            out.push_str(&showboard_column_letter(x));
            out.push(' ');
        }
        out.push('\n');
        for y in 0..board.y_size {
            out.push_str(&format!("{:2} ", board.y_size - y));
            for x in 0..board.x_size {
                let loc = location::get_loc(x, y, board.x_size) as usize;
                out.push(player_io::color_to_char(board.colors[loc]));
                out.push(' ');
            }
            out.push('\n');
        }
        out.trim_end().to_string()
    }

    fn handicap_stones_response(&self, board: &Board) -> String {
        let mut locs = Vec::new();
        for y in 0..board.y_size {
            for x in 0..board.x_size {
                let loc = location::get_loc(x, y, board.x_size);
                if board.colors[loc as usize] != C_EMPTY {
                    locs.push(loc);
                }
            }
        }
        locs.iter()
            .map(|&loc| location::to_string(loc, board.x_size, board.y_size))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn build_genmove_args(&self, debug: bool) -> GenmoveArgs {
        GenmoveArgs {
            search_factor_when_winning_threshold: self.search_factor_when_winning_threshold,
            search_factor_when_winning: self.search_factor_when_winning,
            cleanup_before_pass: self.cleanup_before_pass,
            friendly_pass: self.friendly_pass,
            ogs_chat_to_stderr: self.ogs_chat_to_stderr,
            allow_resignation: self.allow_resignation,
            resign_threshold: self.resign_threshold,
            resign_consec_turns: self.resign_consec_turns,
            resign_min_score_difference: self.resign_min_score_difference,
            resign_min_moves_per_board_area: self.resign_min_moves_per_board_area,
            log_search_info: self.log_search_info,
            log_search_info_for_chosen_move: self.log_search_info_for_chosen_move,
            debug,
        }
    }

    fn gen_move(&mut self, pla: Player, gargs: &GenmoveArgs, _args: &AnalyzeArgs) -> (String, Loc) {
        self.launch_gen_move(pla, gargs);
        self.bot.wait_for_search_to_end();
        let search = self.bot.get_search_stop_and_wait();
        let mut move_loc = search.get_chosen_move_loc();
        let board_x_size = search.root_board.x_size;
        let board_y_size = search.root_board.y_size;

        if move_loc == NULL_LOC || !search.is_legal_strict(move_loc, pla) {
            self.logger.write(&format!(
                "genmove null location or illegal move\n{:?}\nPla: {}\nMoveLoc: {}",
                search.root_board,
                player_io::player_to_string(pla),
                location::to_string(move_loc, board_x_size, board_y_size)
            ));
            return ("pass".to_string(), PASS_LOC);
        }

        let values = search.get_root_values_require_success();
        self.recent_win_loss_values.push(values.win_loss_value);

        let mut resigned = false;
        if gargs.allow_resignation {
            resigned = self.should_resign(pla, values.lead);
        }

        let time_taken = self.genmove_timer.get_seconds();
        self.genmove_time_sum += time_taken;

        if gargs.log_search_info || gargs.debug {
            self.logger.write(&format!(
                "genmove {} visits {} winrate {:.2}% scoreLead {:.2}",
                location::to_string(move_loc, board_x_size, board_y_size),
                values.visits,
                50.0 * (1.0 + values.win_loss_value),
                values.lead
            ));
        }

        move_loc = self.maybe_friendly_pass(gargs, pla, move_loc);
        move_loc = self.maybe_cleanup_before_pass(gargs, pla, move_loc);

        let response = if resigned {
            "resign".to_string()
        } else {
            location::to_string(move_loc, board_x_size, board_y_size)
        };

        let move_to_play = if resigned { NULL_LOC } else { move_loc };
        (response, move_to_play)
    }

    fn gen_move_analyze(
        &mut self,
        pla: Player,
        gargs: &GenmoveArgs,
        args: &AnalyzeArgs,
        out: &mut String,
    ) -> (String, Loc) {
        let (move_str, move_loc) = self.gen_move(pla, gargs, args);
        let search = self.bot.get_search();
        let analyze_line =
            Self::format_analyze_data(search, args, self.analysis_pv_len, self.perspective);
        if !analyze_line.is_empty() {
            out.push_str(&analyze_line);
            out.push('\n');
        }
        (move_str, move_loc)
    }

    fn run_analyze(&mut self, pla: Player, args: &AnalyzeArgs, out: &mut String) {
        if self.is_genmove_params {
            self.bot.set_params(&self.analysis_params);
            self.is_genmove_params = false;
        }
        self.bot.set_avoid_move_until_by_loc(
            &args.avoid_move_until_by_loc_black,
            &args.avoid_move_until_by_loc_white,
        );
        self.bot.set_always_include_owner_map(
            args.show_ownership
                || args.show_ownership_stdev
                || args.show_moves_ownership
                || args.show_moves_ownership_stdev,
        );

        let tc = TimeControls::new();
        self.bot.gen_move_synchronous(pla, &tc);
        self.bot.wait_for_search_to_end();
        let search = self.bot.get_search_stop_and_wait();
        let analyze_line =
            Self::format_analyze_data(search, args, self.analysis_pv_len, self.perspective);
        if !analyze_line.is_empty() {
            out.push_str(&analyze_line);
            out.push('\n');
        }
    }

    fn launch_gen_move(&mut self, pla: Player, _gargs: &GenmoveArgs) {
        self.genmove_timer.reset();
        self.nn_eval.clear_cache();
        if let Some(he) = self.human_eval {
            he.clear_cache();
        }

        if !self.is_genmove_params {
            self.bot.set_params(&self.genmove_params);
            self.is_genmove_params = true;
        }

        let mut params_to_use = self.genmove_params.clone();
        if !self.static_pda_takes_precedence {
            let desired_dynamic_pda = match params_to_use.playout_doubling_advantage_pla {
                P_WHITE => self.desired_dynamic_pda_for_white(),
                P_BLACK => -self.desired_dynamic_pda_for_white(),
                _ => {
                    if pla == P_WHITE {
                        self.desired_dynamic_pda_for_white()
                    } else {
                        -self.desired_dynamic_pda_for_white()
                    }
                }
            };
            params_to_use.playout_doubling_advantage = desired_dynamic_pda;
        }

        let initial_opp_advantage =
            self.initial_black_advantage() * if pla == P_WHITE { 1.0 } else { -1.0 };
        let board_scaling = self.board_size_scaling();
        let threshold = (4.0 / board_scaling).max(2.0);
        let avoid_repeated_pattern_utility = if initial_opp_advantage > threshold {
            self.handicap_avoid_repeated_pattern_utility
        } else {
            self.normal_avoid_repeated_pattern_utility
        };
        params_to_use.avoid_repeated_pattern_utility = avoid_repeated_pattern_utility;

        if params_to_use != *self.bot.get_params() {
            self.bot.set_params(&params_to_use);
        }

        self.last_search_factor = play_utils::get_search_factor(
            self.search_factor_when_winning_threshold,
            self.search_factor_when_winning,
            &self.genmove_params,
            &self.recent_win_loss_values,
            pla,
        );

        self.genmove_expected_id = (self.genmove_expected_id + 1) & 0x3FFFFFFF;
    }

    fn desired_dynamic_pda_for_white(&self) -> f64 {
        0.0
    }

    fn initial_black_advantage(&self) -> f64 {
        let hist = self.bot.get_root_hist();
        let mut hist_copy = hist.clone();
        hist_copy.set_assume_multiple_starting_black_moves_are_handicap(true);
        let handicap_stones = hist_copy.compute_num_handicap_stones();
        if handicap_stones <= 1 {
            return 7.0 - f64::from(hist.rules.komi_f32());
        }
        let extra_black_stones = handicap_stones - 1;
        let stone_value = if hist.rules.scoring_rule == kata_game::rules::ScoringRule::Area {
            15.0
        } else {
            14.0
        };
        let mut white_handicap_bonus = 0.0;
        if hist.rules.white_handicap_bonus_rule == WhiteHandicapBonusRule::N {
            white_handicap_bonus += f64::from(handicap_stones);
        } else if hist.rules.white_handicap_bonus_rule == WhiteHandicapBonusRule::NMinusOne {
            white_handicap_bonus += f64::from(handicap_stones - 1);
        }
        stone_value * f64::from(extra_black_stones)
            + (7.0 - f64::from(hist.rules.komi_f32()) - white_handicap_bonus)
    }

    fn board_size_scaling(&self) -> f64 {
        let board = self.bot.get_root_board();
        ((19.0 * 19.0) / (board.x_size as f64 * board.y_size as f64)).powf(0.75)
    }

    fn should_resign(&self, _pla: Player, _lead: f64) -> bool {
        if !self.allow_resignation
            || self.recent_win_loss_values.len() < self.resign_consec_turns as usize
        {
            return false;
        }
        let start = self.recent_win_loss_values.len() - self.resign_consec_turns as usize;
        for &wl in &self.recent_win_loss_values[start..] {
            if wl > self.resign_threshold {
                return false;
            }
        }
        true
    }

    fn maybe_friendly_pass(&mut self, _gargs: &GenmoveArgs, _pla: Player, move_loc: Loc) -> Loc {
        move_loc
    }

    fn maybe_cleanup_before_pass(
        &mut self,
        _gargs: &GenmoveArgs,
        _pla: Player,
        move_loc: Loc,
    ) -> Loc {
        move_loc
    }

    fn format_analyze_data(
        search: &Search,
        args: &AnalyzeArgs,
        analysis_pv_len: i32,
        perspective: Player,
    ) -> String {
        let mut buf = Vec::new();
        search.get_analysis_data(&mut buf, args.min_moves, false, analysis_pv_len, true);
        buf.retain(|d| d.child_visits > 0);
        buf.truncate(args.max_moves as usize);
        if buf.is_empty() {
            return String::new();
        }

        let board = &search.root_board;
        let mut out = String::new();
        for (i, data) in buf.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            let mut winrate = 0.5 * (1.0 + data.win_loss_value);
            let mut utility = data.utility;
            let mut score_mean = data.score_mean;
            let mut lead = data.lead;
            let mut lcb = data.lcb;
            if perspective == P_BLACK
                || (perspective != P_BLACK && perspective != P_WHITE && search.root_pla == P_BLACK)
            {
                winrate = 1.0 - winrate;
                utility = -utility;
                score_mean = -score_mean;
                lead = -lead;
                lcb = -lcb;
            }
            out.push_str("info");
            out.push_str(&format!(
                " move {} visits {} edgeVisits {} utility {} winrate {} scoreMean {} scoreStdev {} scoreLead {} scoreSelfplay {} prior {} lcb {} utilityLcb {} weight {} order {} pv ",
                location::to_string(data.move_loc, board.x_size, board.y_size),
                data.child_visits,
                data.num_visits,
                utility,
                winrate,
                lead,
                data.score_stdev,
                lead,
                score_mean,
                data.policy_prior,
                lcb,
                lcb,
                data.child_weight_sum,
                data.order
            ));
            data.write_pv(&mut out, board);
        }
        out
    }

    fn handle_raw_nn(&self, pieces: &[String], use_human_model: bool) -> String {
        let mut which_symmetry = nn_inputs::SYMMETRY_ALL;
        let mut parsed = false;
        if pieces.len() == 1 {
            let s = global::to_lower(global::trim(&pieces[0]));
            if s == "all" {
                parsed = true;
            } else if let Some(n) = global::try_string_to_int(&s) {
                if (0..NUM_SYMMETRIES).contains(&n) {
                    which_symmetry = n;
                    parsed = true;
                }
            }
        }
        if !parsed {
            return format!(
                "? Expected one argument 'all' or symmetry index [0-{}] for kata-raw-nn but got '{}'",
                NUM_SYMMETRIES - 1,
                global::concat_vec(pieces, " ")
            );
        }
        self.raw_nn(
            which_symmetry,
            self.genmove_params.root_policy_optimism,
            use_human_model,
        )
    }
}

// ---------------------------------------------------------------------------
// Free-function helpers
// ---------------------------------------------------------------------------

fn to_string_error(e: impl std::error::Error) -> StringError {
    StringError::new(e.to_string())
}

fn showboard_column_letter(x: i32) -> String {
    if x <= 24 {
        let mut c = x;
        if c >= 8 {
            c += 1;
        }
        ((b'A' + c as u8) as char).to_string()
    } else {
        format!(
            "{}{}",
            showboard_column_letter(x / 25 - 1),
            showboard_column_letter(x % 25)
        )
    }
}

fn parse_command_line(line: &str) -> (String, Vec<String>, bool, i32) {
    let mut line = line.to_string();
    let mut has_id = false;
    let mut id = 0;

    let digit_prefix_len = line.chars().take_while(|c| c.is_ascii_digit()).count();
    if digit_prefix_len > 0 {
        has_id = true;
        if let Ok(parsed) = global::parse_digits_range(&line, 0, digit_prefix_len) {
            id = parsed;
        }
        line = line[digit_prefix_len..].to_string();
    }

    line = global::trim(&line).to_string();
    let pieces: Vec<String> = global::split_by(&line, ' ')
        .into_iter()
        .map(|s| global::trim(&s).to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if pieces.is_empty() {
        return (String::new(), Vec::new(), has_id, id);
    }
    let command = pieces[0].clone();
    let args = pieces.into_iter().skip(1).collect();
    (command, args, has_id, id)
}

fn parse_board_size(pieces: &[String]) -> Result<(i32, i32), String> {
    if pieces.len() == 1 {
        if pieces[0].contains(':') {
            let parts: Vec<&str> = pieces[0].split(':').collect();
            if parts.len() == 2 {
                if let (Some(x), Some(y)) = (
                    global::try_string_to_int(parts[0]),
                    global::try_string_to_int(parts[1]),
                ) {
                    return Ok((x, y));
                }
            }
        } else if let Some(n) = global::try_string_to_int(&pieces[0]) {
            return Ok((n, n));
        }
    } else if pieces.len() == 2 {
        if let (Some(x), Some(y)) = (
            global::try_string_to_int(&pieces[0]),
            global::try_string_to_int(&pieces[1]),
        ) {
            return Ok((x, y));
        }
    }
    Err(format!(
        "Expected int argument for boardsize or pair of ints but got '{}'",
        global::concat_vec(pieces, " ")
    ))
}

fn parse_time_settings(pieces: &[String]) -> Result<TimeControls, String> {
    if pieces.len() != 3 {
        return Err(format!(
            "Expected three arguments for time_settings but got '{}'",
            global::concat_vec(pieces, " ")
        ));
    }
    let main_time = global::try_string_to_double(&pieces[0])
        .filter(|t| t.is_finite() && *t >= 0.0 && *t <= TimeControls::MAX_USER_INPUT_TIME)
        .ok_or_else(|| format!("Invalid main time: {}", pieces[0]))?;
    let byo_yomi_time = global::try_string_to_double(&pieces[1])
        .filter(|t| t.is_finite() && *t >= 0.0 && *t <= TimeControls::MAX_USER_INPUT_TIME)
        .ok_or_else(|| format!("Invalid byo-yomi time: {}", pieces[1]))?;
    let byo_yomi_stones = global::try_string_to_int(&pieces[2])
        .filter(|n| *n >= 0 && *n <= 1000000)
        .ok_or_else(|| format!("Invalid byo-yomi stones: {}", pieces[2]))?;

    Ok(if byo_yomi_stones == 0 && byo_yomi_time > 0.0 {
        TimeControls::new()
    } else if byo_yomi_stones == 0 {
        TimeControls::absolute_time(main_time)
    } else {
        TimeControls::canadian_or_byo_yomi_time(main_time, byo_yomi_time, 1, byo_yomi_stones)
    })
}

fn parse_kgs_time_settings(
    pieces: &[String],
    allow_fischer: bool,
) -> Result<Option<TimeControls>, String> {
    if pieces.is_empty() {
        return Err(format!(
            "Expected time control type but got '{}'",
            global::concat_vec(pieces, " ")
        ));
    }
    let what = global::to_lower(global::trim(&pieces[0]));
    match what.as_str() {
        "none" => Ok(Some(TimeControls::new())),
        "absolute" => {
            if pieces.len() < 2 {
                return Err("Expected main time for absolute".to_string());
            }
            let main_time = parse_time_piece(&pieces[1], "main time")?;
            Ok(Some(TimeControls::absolute_time(main_time)))
        }
        "canadian" => {
            if pieces.len() < 4 {
                return Err("Expected main time, period time, and stones for canadian".to_string());
            }
            let main_time = parse_time_piece(&pieces[1], "main time")?;
            let period_time = parse_time_piece(&pieces[2], "period time")?;
            let stones = parse_int_piece(&pieces[3], "stones")?;
            Ok(Some(if stones == 0 && period_time > 0.0 {
                TimeControls::new()
            } else if stones == 0 {
                TimeControls::absolute_time(main_time)
            } else {
                TimeControls::canadian_or_byo_yomi_time(main_time, period_time, 1, stones)
            }))
        }
        "byoyomi" => {
            if pieces.len() < 4 {
                return Err("Expected main time, period time, and periods for byoyomi".to_string());
            }
            let main_time = parse_time_piece(&pieces[1], "main time")?;
            let period_time = parse_time_piece(&pieces[2], "period time")?;
            let periods = parse_int_piece(&pieces[3], "periods")?;
            Ok(Some(if periods == 0 {
                TimeControls::absolute_time(main_time)
            } else {
                TimeControls::canadian_or_byo_yomi_time(main_time, period_time, periods, 1)
            }))
        }
        "fischer" if allow_fischer => {
            if pieces.len() < 3 {
                return Err("Expected main time and increment for fischer".to_string());
            }
            let main_time = parse_time_piece(&pieces[1], "main time")?;
            let increment = parse_time_piece(&pieces[2], "increment")?;
            Ok(Some(TimeControls::fischer_time(main_time, increment)))
        }
        "fischer-capped" if allow_fischer => {
            if pieces.len() < 5 {
                return Err("Expected main time, increment, main time limit, and max time per move for fischer-capped".to_string());
            }
            let main_time = parse_time_piece(&pieces[1], "main time")?;
            let increment = parse_time_piece(&pieces[2], "increment")?;
            let mut main_limit = parse_time_piece_allow_negative(&pieces[3], "main time limit")?;
            let mut max_per_move =
                parse_time_piece_allow_negative(&pieces[4], "max time per move")?;
            if main_limit < 0.0 {
                main_limit = TimeControls::MAX_USER_INPUT_TIME;
            }
            if max_per_move < 0.0 {
                max_per_move = TimeControls::MAX_USER_INPUT_TIME;
            }
            Ok(Some(
                TimeControls::fischer_capped_time(main_time, increment, main_limit, max_per_move)
                    .map_err(|e| e.message)?,
            ))
        }
        _ => Err(format!(
            "Expected 'none', 'absolute', 'byoyomi', 'canadian'{} as first argument",
            if allow_fischer {
                ", 'fischer', or 'fischer-capped'"
            } else {
                ""
            }
        )),
    }
}

fn parse_time_piece(s: &str, description: &str) -> Result<f64, String> {
    global::try_string_to_double(s)
        .filter(|t| t.is_finite() && *t >= 0.0 && *t <= TimeControls::MAX_USER_INPUT_TIME)
        .ok_or_else(|| format!("Invalid {}: {}", description, s))
}

fn parse_time_piece_allow_negative(s: &str, description: &str) -> Result<f64, String> {
    global::try_string_to_double(s)
        .filter(|t| {
            t.is_finite()
                && *t >= -TimeControls::MAX_USER_INPUT_TIME
                && *t <= TimeControls::MAX_USER_INPUT_TIME
        })
        .ok_or_else(|| format!("Invalid {}: {}", description, s))
}

fn parse_int_piece(s: &str, description: &str) -> Result<i32, String> {
    global::try_string_to_int(s)
        .filter(|n| *n >= 0 && *n <= 1000000)
        .ok_or_else(|| format!("Invalid {}: {}", description, s))
}

fn parse_analyze_command(
    command: &str,
    pieces: &[String],
    pla: &mut Player,
) -> (AnalyzeArgs, bool) {
    let is_lz = command == "lz-analyze" || command == "lz-genmove_analyze";
    let is_kata = command.starts_with("kata-") && command.contains("analyze");

    let mut args = AnalyzeArgs {
        analyzing: true,
        lz: is_lz,
        kata: is_kata,
        ..Default::default()
    };

    let mut idx = 0;
    if let Some(p) = pieces.get(idx).and_then(|s| player_io::try_parse_player(s)) {
        *pla = p;
        idx += 1;
    }

    if let Some(s) = pieces.get(idx) {
        if let Some(interval) = global::try_string_to_double(s)
            .filter(|v| v.is_finite() && *v >= 0.0 && *v < TimeControls::MAX_USER_INPUT_TIME)
        {
            args.seconds_per_report = interval * 0.01;
            idx += 1;
        }
    }

    while idx < pieces.len() {
        let key = &pieces[idx];
        idx += 1;
        if idx >= pieces.len() {
            return (args, true);
        }
        let value = &pieces[idx];
        idx += 1;

        match key.as_str() {
            "interval" => {
                if let Some(v) = global::try_string_to_double(value).filter(|v| {
                    v.is_finite() && *v >= 0.0 && *v < TimeControls::MAX_USER_INPUT_TIME
                }) {
                    args.seconds_per_report = v * 0.01;
                } else {
                    return (args, true);
                }
            }
            "avoid" | "allow" => {
                if idx + 1 >= pieces.len() {
                    return (args, true);
                }
                let moves_str = &pieces[idx];
                idx += 1;
                let until_str = &pieces[idx];
                idx += 1;
                let until = global::try_string_to_int(until_str).filter(|n| *n >= 1);
                if until.is_none() {
                    return (args, true);
                }
                let avoid_pla = player_io::try_parse_player(value);
                if avoid_pla.is_none() {
                    return (args, true);
                }
                let avoid_pla = avoid_pla.unwrap();
                let target = if avoid_pla == P_BLACK {
                    &mut args.avoid_move_until_by_loc_black
                } else {
                    &mut args.avoid_move_until_by_loc_white
                };
                target.resize(MAX_ARR_SIZE, 0);
                if key == "allow" {
                    target.fill(until.unwrap());
                }
                for mv in global::split_by(moves_str, ',') {
                    let mv = global::trim(&mv).to_string();
                    if mv.is_empty() {
                        continue;
                    }
                    if let Some(loc) = location::try_of_string(&mv, 19, 19) {
                        if key == "allow" {
                            target[loc as usize] = 0;
                        } else {
                            target[loc as usize] = until.unwrap();
                        }
                    } else {
                        return (args, true);
                    }
                }
            }
            "minmoves" => {
                if let Some(v) =
                    global::try_string_to_int(value).filter(|n| *n >= 0 && *n < 1000000000)
                {
                    args.min_moves = v;
                } else {
                    return (args, true);
                }
            }
            "maxmoves" => {
                if let Some(v) =
                    global::try_string_to_int(value).filter(|n| *n >= 0 && *n < 1000000000)
                {
                    args.max_moves = v;
                } else {
                    return (args, true);
                }
            }
            "rootInfo" if is_kata => {
                if let Ok(b) = global::string_to_bool(value) {
                    args.show_root_info = b;
                } else {
                    return (args, true);
                }
            }
            "ownership" if is_kata => {
                if let Ok(b) = global::string_to_bool(value) {
                    args.show_ownership = b;
                } else {
                    return (args, true);
                }
            }
            "ownershipStdev" if is_kata => {
                if let Ok(b) = global::string_to_bool(value) {
                    args.show_ownership_stdev = b;
                } else {
                    return (args, true);
                }
            }
            "movesOwnership" if is_kata => {
                if let Ok(b) = global::string_to_bool(value) {
                    args.show_moves_ownership = b;
                } else {
                    return (args, true);
                }
            }
            "movesOwnershipStdev" if is_kata => {
                if let Ok(b) = global::string_to_bool(value) {
                    args.show_moves_ownership_stdev = b;
                } else {
                    return (args, true);
                }
            }
            "pvVisits" if is_kata => {
                if let Ok(b) = global::string_to_bool(value) {
                    args.show_pv_visits = b;
                } else {
                    return (args, true);
                }
            }
            "pvEdgeVisits" if is_kata => {
                if let Ok(b) = global::string_to_bool(value) {
                    args.show_pv_edge_visits = b;
                } else {
                    return (args, true);
                }
            }
            "noResultValue" if is_kata => {
                if let Ok(b) = global::string_to_bool(value) {
                    args.show_no_result_value = b;
                } else {
                    return (args, true);
                }
            }
            _ => return (args, true),
        }
    }

    (args, false)
}

/// Public entry point for the `gtp` subcommand.
pub fn gtp(args: &[String]) -> i32 {
    let parsed =
        match GtpArgs::try_parse_from(std::iter::once(&"gtp".to_string()).chain(args.iter())) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Error: {}", e);
                return 1;
            }
        };

    let cfg = match parsed.common.get_config("gtp_example.cfg") {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {}", e.message);
            return 1;
        }
    };

    let model_file = match parsed.common.get_model_file() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error: {}", e.message);
            return 1;
        }
    };
    let human_model_file = parsed.common.get_human_model_file();

    let logger: &'static Logger = Box::leak(Box::new(Logger::new(
        LoggerOptions {
            log_to_stdout: false,
            log_to_stderr: true,
            log_time: false,
        },
        None,
    )));

    logger.write("GTP Engine starting...");
    let seed_rand = Rand::new();

    let mut engine = match GtpEngine::new(
        cfg,
        model_file,
        human_model_file,
        seed_rand,
        logger,
        parsed.override_version.as_deref(),
    ) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Error: {}", e.message);
            return 1;
        }
    };

    let output: OutputHandle = Arc::new(Mutex::new(std::io::stdout()));
    if let Err(e) = engine.run(std::io::stdin().lock(), output) {
        eprintln!("I/O error: {}", e);
        return 1;
    }
    0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn run_session(input: &str) -> String {
        let mut engine = GtpEngine::new_for_tests(9, 9);
        let output: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
        let out_clone: OutputHandle = output.clone() as OutputHandle;
        engine
            .run(Cursor::new(input.as_bytes()), out_clone)
            .unwrap();
        let bytes = output.lock().unwrap().clone();
        String::from_utf8(bytes).unwrap()
    }

    #[test]
    fn test_parse_command_line_with_id() {
        let (cmd, pieces, has_id, id) = parse_command_line("123 genmove black");
        assert_eq!(cmd, "genmove");
        assert_eq!(pieces, vec!["black"]);
        assert!(has_id);
        assert_eq!(id, 123);
    }

    #[test]
    fn test_parse_command_line_no_id() {
        let (cmd, pieces, has_id, id) = parse_command_line("  play B Q16  ");
        assert_eq!(cmd, "play");
        assert_eq!(pieces, vec!["B", "Q16"]);
        assert!(!has_id);
        assert_eq!(id, 0);
    }

    #[test]
    fn test_parse_board_size_square() {
        assert_eq!(parse_board_size(&["19".to_string()]).unwrap(), (19, 19));
    }

    #[test]
    fn test_parse_board_size_rectangular() {
        assert_eq!(parse_board_size(&["19:13".to_string()]).unwrap(), (19, 13));
        assert_eq!(
            parse_board_size(&["19".to_string(), "13".to_string()]).unwrap(),
            (19, 13)
        );
    }

    #[test]
    fn test_known_and_list_commands() {
        let out = run_session("list_commands\nknown_command genmove\nknown_command foobar\n");
        assert!(out.contains("genmove"));
        assert!(out.contains("= true"));
        assert!(out.contains("= false"));
    }

    #[test]
    fn test_basic_session() {
        let out = run_session(
            "boardsize 5\nclear_board\nkomi 7.5\nplay B C3\nplay W D3\nshowboard\nundo\nshowboard\n",
        );
        assert!(out.contains("= "), "responses should be prefixed with =");
        assert!(out.contains("C3") || out.contains("D3") || out.contains("X") || out.contains("O"));
    }

    #[test]
    fn test_genmove_returns_legal_move() {
        let out = run_session("boardsize 5\nclear_board\ngenmove black\n");
        assert!(out.contains("= "));
        // With the dummy evaluator the bot should return a GTP vertex.
        let line = out
            .lines()
            .find(|l| l.starts_with("= ") && !l.trim().is_empty())
            .unwrap();
        assert!(!line.contains('?'));
    }

    #[test]
    fn test_final_score_and_status() {
        let out = run_session(
            "boardsize 5\nclear_board\nplay B C3\nplay W D3\nfinal_score\nfinal_status_list alive\n",
        );
        assert!(out.contains("= "));
        // Score stub returns 0 for unfinished games.
        assert!(out.contains("= 0") || out.contains("= B+") || out.contains("= W+"));
    }

    #[test]
    fn test_time_settings() {
        let out = run_session("time_settings 60 15 1\nkgs-time_settings byoyomi 60 10 5\n");
        assert!(!out.contains('?'));
    }

    #[test]
    fn test_rules_commands() {
        let out = run_session("kata-get-rules\nkata-set-rule ko simple\n");
        assert!(out.contains("ko"));
        assert!(!out.contains('?'));
    }

    #[test]
    fn test_invalid_command_is_error() {
        let out = run_session("not_a_command\n");
        assert!(out.contains("?"));
        assert!(out.contains("unknown command"));
    }

    #[test]
    fn test_quit() {
        let out = run_session("quit\n");
        // No error response; engine simply exits.
        assert!(!out.contains('?'));
    }

    #[test]
    fn test_showboard_has_grid() {
        let out = run_session("boardsize 3\nclear_board\nshowboard\n");
        assert!(out.contains('.'));
        assert!(out.contains('A') || out.contains('B') || out.contains('C'));
    }

    #[test]
    fn test_printsgf_outputs_sgf() {
        let out = run_session("boardsize 3\nclear_board\nprintsgf\n");
        assert!(!out.contains('?'));
        assert!(out.contains("FF[4]") && out.contains("SZ[3]"));
    }

    #[test]
    fn test_loadsgf_sets_position() {
        let sgf = "(;FF[4]GM[1]SZ[3];B[aa];W[bb])";
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), sgf).unwrap();
        let path = tmp.path().to_str().unwrap();
        let out = run_session(&format!(
            "boardsize 3\nclear_board\nloadsgf {path}\nshowboard\n"
        ));
        assert!(!out.contains('?'), "output: {}", out);
        assert!(out.contains('X') && out.contains('O'), "output: {}", out);
    }
}
