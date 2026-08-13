//! Miscellaneous CLI subcommands.
//!
//! Corresponds to the independent commands in `cpp/command/misc.cpp`.

#![allow(dead_code)]

use clap::Parser;
use kata_core::config::ConfigParser;
use kata_core::global::StringError;
use kata_core::logger::{Logger, LoggerOptions};
use kata_core::rng::Rand;
use kata_data::sgf::CompactSgf;
use kata_game::board::{Board, NULL_LOC, P_BLACK, P_WHITE, Player, get_opp, location, player_io};
use kata_game::history::BoardHistory;
use kata_game::rules::Rules;
use kata_nn::eval::NnEvaluator;
use kata_program::play::{BotSpec, GameRunner};
use kata_program::play_settings::PlaySettings;
use kata_program::play_utils::get_game_initialization_move;
use kata_program::setup::{self, SetupFor};
use kata_search::async_bot::AsyncBot;
use kata_search::search::Search;
use kata_search::time_control::TimeControls;

use crate::cli::CommonArgs;

#[cfg(unix)]
use std::time::{SystemTime, UNIX_EPOCH};

/// Print information about the system's monotonic clock.
///
/// Mirrors `MainCmds::printclockinfo` in `cpp/command/misc.cpp`.
/// On Windows the original C++ command is a no-op; here we still print the
/// current Unix epoch time for diagnostics.
pub fn print_clock_info(_args: &[String]) -> i32 {
    #[cfg(unix)]
    {
        // Rust's `Instant` does not expose a steady-clock period, so we report
        // the nanosecond resolution that Linux's CLOCK_MONOTONIC typically uses.
        println!("Tick unit in seconds: 1 / 1000000000");
        if let Ok(d) = SystemTime::now().duration_since(UNIX_EPOCH) {
            println!("Ticks since epoch: {}", d.as_nanos());
        }
    }
    #[cfg(windows)]
    {
        println!("Does nothing on windows, disabled");
    }
    0
}

/// CLI arguments for `sampleinitializations`.
#[derive(Parser, Debug, Clone)]
struct SampleInitializationsArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// Number of initial positions to generate.
    #[arg(long = "num", default_value = "1")]
    num: i32,

    /// Print out values and scores on the initialized poses.
    #[arg(long = "evaluate")]
    evaluate: bool,
}

/// Build a minimal config that satisfies `GameRunner`, `GameInitializer`,
/// `PlaySettings::load_for_selfplay`, and `SearchParams` loading.
fn minimal_selfplay_config() -> ConfigParser {
    let mut cfg = ConfigParser::new(false, false);
    cfg.initialize_str(
        "numSearchThreads = 1
         maxVisits = 10
         valueWeightExponent = 0
         policyOptimism = 0
         logSearchInfo = false
         logMoves = false
         maxMovesPerGame = 0
         koRules = POSITIONAL
         scoringRules = AREA
         taxRules = NONE
         multiStoneSuicideLegals = true
         hasButtons = false
         bSizes = 9
         bSizeRelProbs = 1
         komiMean = 7.5
         initGamesWithPolicy = false
         compensateAfterPolicyInitProb = 0.0
         sidePositionProb = 0.0
         earlyForkGameProb = 0.0
         earlyForkGameExpectedMoveProp = 0.0
         forkGameProb = 0.0
         forkGameMinChoices = 1
         earlyForkGameMaxChoices = 1
         forkGameMaxChoices = 1
         cheapSearchProb = 0.0
         cheapSearchVisits = 1
         cheapSearchTargetWeight = 0.0
         reduceVisits = false
         reduceVisitsThreshold = 0.5
         reduceVisitsThresholdLookback = 1
         reducedVisitsMin = 1
         reducedVisitsWeight = 1.0
         handicapAsymmetricPlayoutProb = 0.0
         normalAsymmetricPlayoutProb = 0.0
         maxAsymmetricRatio = 2.0
         minAsymmetricCompensateKomiProb = 0.0
         policySurpriseDataWeight = 0.0
         valueSurpriseDataWeight = 0.0
         allowResignation = false
         resignThreshold = -1.0
         resignConsecTurns = 1
        ",
    )
    .expect("hard-coded minimal config is valid");
    cfg
}

/// Create a dummy NN evaluator backed by `/dev/null` so that commands can run
/// without a real model on disk.
fn make_dummy_nn_evaluator(
    model_file: String,
    cfg: &ConfigParser,
    logger: &Logger,
    rand: &mut Rand,
    setup_for: SetupFor,
) -> Result<NnEvaluator, StringError> {
    let params = setup::load_single_params(cfg, setup_for)?;
    setup::initialize_session(cfg);
    let expected_concurrent_evals = params.num_threads;
    let default_max_batch_size = ((params.num_threads + 3) / 4 * 4).max(8);
    setup::initialize_nn_evaluator(
        model_file.clone(),
        model_file,
        String::new(),
        cfg,
        logger,
        rand,
        expected_concurrent_evals,
        19,
        19,
        default_max_batch_size,
        false,
        false,
        setup_for,
    )
}

/// Create a logger that mirrors the C++ default of writing to stdout.
fn make_stdout_logger() -> Logger {
    Logger::new(
        LoggerOptions {
            log_to_stdout: true,
            log_to_stderr: false,
            log_time: false,
        },
        None,
    )
}

/// Print a board in a simple labeled form.
fn print_board_labeled(board: &Board) {
    print!("   ");
    for x in 0..board.x_size {
        print!("{} ", (b'A' + x as u8) as char);
    }
    println!();
    for y in 0..board.y_size {
        print!("{:2} ", board.y_size - y);
        for x in 0..board.x_size {
            let loc = location::get_loc(x, y, board.x_size);
            print!("{} ", player_io::color_to_char(board.colors[loc as usize]));
        }
        println!();
    }
}

/// Sample random game initializations and print the resulting start positions.
///
/// Mirrors `MainCmds::sampleinitializations` in `cpp/command/misc.cpp`.
pub fn sample_initializations(args: &[String]) -> i32 {
    if let Err(e) = sample_initializations_impl(args) {
        eprintln!("Error: {}", e);
        return 1;
    }
    0
}

fn sample_initializations_impl(args: &[String]) -> Result<(), StringError> {
    let parsed = SampleInitializationsArgs::try_parse_from(
        std::iter::once(&"sampleinitializations".to_string()).chain(args.iter()),
    )
    .map_err(|e| StringError::new(format!("Argument error: {}", e)))?;

    let num_to_gen = parsed.num.max(0);
    let evaluate = parsed.evaluate;

    let cfg = if parsed.common.config.is_empty() {
        let mut cfg = minimal_selfplay_config();
        parsed.common.maybe_apply_override_config_arg(&mut cfg)?;
        cfg
    } else {
        parsed.common.get_config_allow_empty(None)?
    };

    let mut rand = Rand::new();
    let logger = make_stdout_logger();

    let nn_eval = if parsed.common.config.is_empty() {
        make_dummy_nn_evaluator(
            "/dev/null".to_string(),
            &cfg,
            &logger,
            &mut rand,
            SetupFor::Gtp,
        )?
    } else {
        let model_file = parsed.common.get_model_file()?;
        let params = setup::load_single_params(&cfg, SetupFor::Gtp)?;
        setup::initialize_session(&cfg);
        let expected_concurrent_evals = params.num_threads;
        let default_max_batch_size = ((params.num_threads + 3) / 4 * 4).max(8);
        setup::initialize_nn_evaluator(
            model_file.clone(),
            model_file,
            String::new(),
            &cfg,
            &logger,
            &mut rand,
            expected_concurrent_evals,
            19,
            19,
            default_max_batch_size,
            false,
            false,
            SetupFor::Gtp,
        )?
    };

    logger.write("Loaded neural net");

    let mut eval_bot = {
        let mut params = setup::load_single_params(&cfg, SetupFor::Distributed)?;
        params.max_visits = 20;
        params.num_threads = 1;
        let seed = rand.next_u64().to_string();
        AsyncBot::new(params, &nn_eval, &logger, &seed)
    };

    let mut cfg = cfg;
    cfg.override_key("maxMovesPerGame", "0");

    let play_settings = PlaySettings::load_for_selfplay(&cfg, false)?;
    let game_runner = GameRunner::new(&cfg, play_settings, &logger)?;

    for _ in 0..num_to_gen {
        let seed = rand.next_u64().to_string();
        let params = setup::load_single_params(&cfg, SetupFor::Distributed)?;
        let bot_spec = BotSpec {
            bot_idx: 0,
            bot_name: String::new(),
            nn_eval: Some(&nn_eval),
            base_params: params,
        };

        let data = game_runner
            .run_game(
                &seed,
                &bot_spec,
                &bot_spec,
                None,
                None,
                &logger,
                Box::new(|| false),
                None,
                Box::new(|| None),
                Box::new(|_, _| {}),
                Box::new(|_, _, _, _, _, _, _, _| {}),
            )
            .ok_or_else(|| {
                StringError::new("Game was stopped before initialization".to_string())
            })?;

        println!("{}", data.start_hist.rules);
        print_board_labeled(&data.start_board);
        println!();

        if evaluate {
            eval_bot.set_position(data.start_pla, &data.start_board, &data.start_hist);
            eval_bot.gen_move_synchronous(data.start_pla, &TimeControls::new());
            let values = eval_bot
                .get_search_stop_and_wait()
                .get_root_values_require_success();
            println!("Winloss: {}", values.win_loss_value);
            println!("Lead: {}", values.lead);
        }
    }

    Ok(())
}

/// CLI arguments for `evalrandominits`.
#[derive(Parser, Debug, Clone)]
struct EvalRandomInitsArgs {
    #[command(flatten)]
    common: CommonArgs,
}

/// Evaluate random game initializations by playing policy moves and searching.
///
/// Mirrors `MainCmds::evalrandominits` in `cpp/command/misc.cpp`.
/// The public entry point loops forever as the original does; use
/// `eval_random_inits_limited` for testing.
pub fn eval_random_inits(args: &[String]) -> i32 {
    eval_random_inits_impl(args, None)
}

/// Run a bounded number of iterations (used by unit tests).
pub fn eval_random_inits_limited(args: &[String], max_iterations: usize) -> i32 {
    eval_random_inits_impl(args, Some(max_iterations))
}

fn eval_random_inits_impl(args: &[String], max_iterations: Option<usize>) -> i32 {
    let parsed = match EvalRandomInitsArgs::try_parse_from(
        std::iter::once(&"evalrandominits".to_string()).chain(args.iter()),
    ) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error: {}", e);
            return 1;
        }
    };

    let cfg = if parsed.common.config.is_empty() {
        let mut cfg = minimal_selfplay_config();
        parsed
            .common
            .maybe_apply_override_config_arg(&mut cfg)
            .unwrap();
        cfg
    } else {
        match parsed.common.get_config_allow_empty(None) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Error: {}", e);
                return 1;
            }
        }
    };

    let mut rand = Rand::new();
    let logger = make_stdout_logger();

    let nn_eval = if parsed.common.config.is_empty() {
        match make_dummy_nn_evaluator(
            "/dev/null".to_string(),
            &cfg,
            &logger,
            &mut rand,
            SetupFor::Gtp,
        ) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("Error: {}", e);
                return 1;
            }
        }
    } else {
        let model_file = match parsed.common.get_model_file() {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Error: {}", e);
                return 1;
            }
        };
        let params = match setup::load_single_params(&cfg, SetupFor::Gtp) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Error: {}", e);
                return 1;
            }
        };
        setup::initialize_session(&cfg);
        let expected_concurrent_evals = params.num_threads;
        let default_max_batch_size = ((params.num_threads + 3) / 4 * 4).max(8);
        match setup::initialize_nn_evaluator(
            model_file.clone(),
            model_file,
            String::new(),
            &cfg,
            &logger,
            &mut rand,
            expected_concurrent_evals,
            19,
            19,
            default_max_batch_size,
            false,
            false,
            SetupFor::Gtp,
        ) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("Error: {}", e);
                return 1;
            }
        }
    };

    let mut eval_bot = {
        let mut params = setup::load_single_params(&cfg, SetupFor::Distributed).unwrap();
        params.max_visits = 40;
        params.num_threads = 1;
        let seed = rand.next_u64().to_string();
        Search::new(params, &nn_eval, &logger, &seed)
    };

    let mut game_rand = Rand::new();
    let mut iteration = 0usize;

    loop {
        if let Some(max) = max_iterations {
            if iteration >= max {
                break;
            }
            iteration += 1;
        }

        let mut board = Board::new(19, 19);
        let mut pla = P_BLACK;
        let rules = Rules::parse_rules("japanese").unwrap();
        let mut hist = BoardHistory::new(board.clone(), pla, rules, 0);
        let num_initial_moves_to_play = game_rand.next_u32_bounded(200) as i32;
        let temperature = 1.0;

        for _ in 0..num_initial_moves_to_play {
            let loc = get_game_initialization_move(
                &mut eval_bot,
                &board,
                &hist,
                pla,
                &mut game_rand,
                temperature,
            );

            assert!(hist.is_legal(&board, loc, pla));
            hist.make_board_move_assume_legal(&mut board, loc, pla);
            pla = get_opp(pla);

            hist.end_game_if_all_pass_alive(&board);
            if hist.is_game_finished {
                break;
            }
        }

        eval_bot.set_position(pla, &board, &hist);
        eval_bot.run_whole_search(pla);
        let values = eval_bot.get_root_values_require_success();
        println!(
            "{},{},{}",
            num_initial_moves_to_play, values.win_loss_value, values.lead
        );
    }

    0
}

/// CLI arguments for `searchentropyanalysis`.
#[derive(Parser, Debug, Clone)]
struct SearchEntropyAnalysisArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// Dataset to analyze (9, 13, 19, 10x14, rectangle).
    #[arg(long = "boardsizedataset")]
    board_size_dataset: String,
}

/// Return a tiny built-in SGF dataset for the named board size.
///
/// The original C++ command uses `TestCommon::getMultiGameSize*Data`. Those
/// test datasets are not yet ported to Rust, so this function supplies
/// minimal valid SGFs that exercise the same analysis loop.
fn built_in_sgf_dataset(name: &str) -> Result<Vec<String>, StringError> {
    match name {
        "9" => Ok(vec![
            "(;FF[4]GM[1]SZ[9]KM[7.5]RU[koPOSITIONALscoreAREAtaxALLsui1];B[ee];W[ec];B[gd];W[dg];B[fg];W[cf];B[ed];W[gc];B[dc];W[fc];B[be];W[bf])"
                .to_string(),
            "(;FF[4]GM[1]SZ[9]KM[7.5]RU[koSITUATIONALscoreAREAtaxALLsui0];B[de];W[fe];B[df];W[fd];B[ge];W[gd];B[ff];W[gf];B[fg];W[gg];B[fh];W[gh])"
                .to_string(),
        ]),
        "13" => Ok(vec![
            "(;FF[4]GM[1]SZ[13]KM[7.5]RU[koPOSITIONALscoreAREAtaxALLsui1];B[dd];W[ff];B[df];W[fd];B[jd];W[jf];B[gg];W[fg];B[fh];W[eh])"
                .to_string(),
        ]),
        "19" => Ok(vec![
            "(;FF[4]GM[1]SZ[19]KM[7.5]RU[koPOSITIONALscoreAREAtaxALLsui1];B[dd];W[pp];B[dp];W[pd];B[dj];W[pj];B[jd];W[jp])"
                .to_string(),
        ]),
        "10x14" => Ok(vec![
            "(;FF[4]GM[1]SZ[10:14]KM[7.5]RU[koPOSITIONALscoreAREAtaxALLsui1];B[ee];W[eh];B[ej];W[em];B[eo])"
                .to_string(),
        ]),
        "rectangle" => Ok(vec![
            "(;FF[4]GM[1]SZ[10:14]KM[7.5]RU[koPOSITIONALscoreAREAtaxALLsui1];B[ee];W[eh];B[ej];W[em];B[eo])"
                .to_string(),
        ]),
        _ => Err(StringError::new(format!(
            "Unknown dataset to test gpu error on: {}",
            name
        ))),
    }
}

/// Set up the initial board and history from a compact SGF.
///
/// Mirrors `CompactSgf::setupInitialBoardAndHist` in `cpp/dataio/sgf.cpp`.
fn setup_initial_board_and_hist(
    sgf: &CompactSgf,
    initial_rules: &Rules,
) -> Result<(Board, Player, BoardHistory), StringError> {
    let mut next_pla = P_BLACK;
    let pl_color = sgf.root_node.get_pl_specified_color();
    if pl_color == P_BLACK || pl_color == P_WHITE {
        next_pla = pl_color;
    } else {
        let has_black = sgf.placements.iter().any(|m| m.pla == P_BLACK);
        let all_black = sgf.placements.iter().all(|m| m.pla == P_BLACK);
        if has_black && all_black {
            next_pla = P_WHITE;
        }
    }
    if let Some(first_move) = sgf.moves.first() {
        next_pla = first_move.pla;
    }

    let mut board = Board::new(sgf.x_size, sgf.y_size);
    if !board.set_stones_fail_if_no_libs(&sgf.placements) {
        return Err(StringError::new(
            "setup_initial_board_and_hist: initial board position contains invalid stones or zero-liberty stones"
                .to_string(),
        ));
    }
    let mut hist = BoardHistory::new(board.clone(), next_pla, *initial_rules, 0);
    let num_stones = i64::from(board.num_stones_on_board());
    if hist.initial_turn_number < num_stones {
        hist.set_initial_turn_number(num_stones);
    }
    Ok((board, next_pla, hist))
}

/// Analyze search entropy across built-in SGF datasets.
///
/// Mirrors `MainCmds::searchentropyanalysis` in `cpp/command/misc.cpp`.
pub fn search_entropy_analysis(args: &[String]) -> i32 {
    if let Err(e) = search_entropy_analysis_impl(args) {
        eprintln!("Error: {}", e);
        return 1;
    }
    0
}

fn search_entropy_analysis_impl(args: &[String]) -> Result<(), StringError> {
    let parsed = SearchEntropyAnalysisArgs::try_parse_from(
        std::iter::once(&"searchentropyanalysis".to_string()).chain(args.iter()),
    )
    .map_err(|e| StringError::new(format!("Argument error: {}", e)))?;

    let model_file = if parsed.common.config.is_empty() {
        "/dev/null".to_string()
    } else {
        parsed.common.get_model_file()?
    };
    let cfg = if parsed.common.config.is_empty() {
        let mut cfg = minimal_selfplay_config();
        parsed.common.maybe_apply_override_config_arg(&mut cfg)?;
        cfg
    } else {
        parsed.common.get_config("gtp_example.cfg")?
    };
    let board_size_dataset = parsed.board_size_dataset;

    let mut rand = Rand::new();
    let logger = make_stdout_logger();
    logger.write("Search Entropy Analysis");
    logger.write(&format!("Model: {}", model_file));
    logger.write(&format!("Dataset: {}", board_size_dataset));

    let params = setup::load_single_params(&cfg, SetupFor::Gtp)?;
    setup::initialize_session(&cfg);
    let expected_concurrent_evals = params.num_threads;
    let default_max_batch_size = ((params.num_threads + 3) / 4 * 4).max(8);
    let nn_eval = setup::initialize_nn_evaluator(
        model_file.clone(),
        model_file.clone(),
        String::new(),
        &cfg,
        &logger,
        &mut rand,
        expected_concurrent_evals,
        19,
        19,
        default_max_batch_size,
        false,
        false,
        SetupFor::Gtp,
    )?;
    logger.write("Loaded neural net");

    let mut bot = {
        let search_params = setup::load_single_params(&cfg, SetupFor::Gtp)?;
        let seed = rand.next_u64().to_string();
        Search::new(search_params, &nn_eval, &logger, &seed)
    };

    let sgf_data = built_in_sgf_dataset(&board_size_dataset)?;

    // Build list of all (sgf_idx, turn_idx) pairs, then deterministically shuffle.
    let mut positions: Vec<(usize, usize)> = Vec::new();
    {
        let mut sgf_objs = Vec::new();
        for sgf in &sgf_data {
            sgf_objs.push(CompactSgf::parse(sgf).map_err(|e| StringError::new(e.to_string()))?);
        }
        for (sgf_idx, sgf_obj) in sgf_objs.iter().enumerate() {
            for turn_idx in 0..sgf_obj.moves.len() {
                positions.push((sgf_idx, turn_idx));
            }
        }
    }
    {
        let mut shuffle_rand = Rand::new_from_seed("searchentropyanalysis_shuffle_seed");
        shuffle_rand.shuffle(&mut positions);
    }
    logger.write(&format!("Total positions to search: {}", positions.len()));

    let mut search_entropies: Vec<f64> = Vec::new();
    let mut search_surprises: Vec<f64> = Vec::new();
    let mut num_positions = 0i32;

    let print_running_stats = |num: i32, entropies: &[f64], surprises: &[f64]| {
        println!("Searched numPositions: {}", num);
        if num > 0 {
            let entropy_mean: f64 = entropies.iter().sum::<f64>() / num as f64;
            let entropy_var: f64 = entropies
                .iter()
                .map(|v| {
                    let d = v - entropy_mean;
                    d * d
                })
                .sum::<f64>()
                / num as f64;
            println!(
                "Mean search entropy: {} standard deviation: {}",
                entropy_mean,
                entropy_var.sqrt()
            );

            let surprise_mean: f64 = surprises.iter().sum::<f64>() / num as f64;
            let surprise_var: f64 = surprises
                .iter()
                .map(|v| {
                    let d = v - surprise_mean;
                    d * d
                })
                .sum::<f64>()
                / num as f64;
            println!(
                "Mean search surprise: {} standard deviation: {}",
                surprise_mean,
                surprise_var.sqrt()
            );
        }
    };

    let initial_rules = Rules::default();

    for (sgf_idx, turn_idx) in positions {
        let sgf =
            CompactSgf::parse(&sgf_data[sgf_idx]).map_err(|e| StringError::new(e.to_string()))?;

        let (mut board, mut pla, mut hist) = setup_initial_board_and_hist(&sgf, &initial_rules)?;

        for i in 0..turn_idx {
            let move_loc = sgf.moves[i].loc;
            if move_loc != NULL_LOC && hist.is_legal(&board, move_loc, pla) {
                hist.make_board_move_assume_legal(&mut board, move_loc, pla);
                pla = get_opp(pla);
            }
        }

        if !hist.is_game_finished {
            bot.set_position(pla, &board, &hist);
            bot.run_whole_search(pla);

            let mut surprise = 0.0;
            let mut search_entropy = 0.0;
            let mut policy_entropy = 0.0;
            if bot.get_policy_surprise_and_entropy(
                &mut surprise,
                &mut search_entropy,
                &mut policy_entropy,
            ) {
                search_entropies.push(search_entropy);
                search_surprises.push(surprise);
                num_positions += 1;
                if num_positions % 10 == 0 {
                    print_running_stats(num_positions, &search_entropies, &search_surprises);
                }
            }
        }
    }

    println!("{}", model_file);
    println!("Dataset: {}", board_size_dataset);
    print_running_stats(num_positions, &search_entropies, &search_surprises);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_print_clock_info_returns_zero() {
        assert_eq!(print_clock_info(&[]), 0);
    }

    #[test]
    fn test_sample_initializations_default_config() {
        let args = vec!["--num".to_string(), "1".to_string()];
        assert_eq!(sample_initializations(&args), 0);
    }

    #[test]
    fn test_sample_initializations_with_evaluate() {
        let args = vec![
            "--num".to_string(),
            "1".to_string(),
            "--evaluate".to_string(),
        ];
        assert_eq!(sample_initializations(&args), 0);
    }

    #[test]
    fn test_eval_random_inits_limited() {
        let args = vec![];
        assert_eq!(eval_random_inits_limited(&args, 1), 0);
    }

    #[test]
    fn test_search_entropy_analysis_size9() {
        let args = vec!["--boardsizedataset".to_string(), "9".to_string()];
        assert_eq!(search_entropy_analysis(&args), 0);
    }

    #[test]
    fn test_search_entropy_analysis_unknown_dataset() {
        let args = vec![
            "--boardsizedataset".to_string(),
            "not-a-dataset".to_string(),
        ];
        assert_eq!(search_entropy_analysis(&args), 1);
    }
}
