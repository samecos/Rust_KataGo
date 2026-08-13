//! Gatekeeper command: test candidate neural nets against an accepted baseline.
//!
//! Corresponds to `MainCmds::gatekeeper` in `cpp/command/gatekeeper.cpp`.
//!
//! This Rust implementation is a simplified single-threaded gatekeeping loop. It
//! loads the latest candidate model from `--test-models-dir`, the latest accepted
//! model from `--accepted-models-dir`, plays `numGamesPerGating` games
//! alternating colors, and moves the candidate to `--accepted-models-dir` or
//! `--rejected-models-dir` based on the required win proportion.
//!
//! The original C++ implementation supports concurrent game threads, per-model
//! data write threads, SGF output, and signal handling. Those are omitted here
//! because the underlying Rust infrastructure (notably SGF writing) is not yet
//! ported.

#![allow(
    dead_code,
    clippy::collapsible_if,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

use std::sync::Arc;

use clap::Parser;

use kata_core::global::StringError;
use kata_core::logger::{Logger, LoggerOptions};
use kata_core::rng::Rand;
use kata_core::time::get_compact_date_time_string;
use kata_data::model_loader::{self, ModelInfo};
use kata_data::training::FinishedGameData;
use kata_game::board::{P_BLACK, P_WHITE};
use kata_nn::eval::NnEvaluator;
use kata_program::play::{BotSpec, GameRunner};
use kata_program::play_settings::PlaySettings;
use kata_program::setup::{self, SetupFor};

use crate::cli::CommonArgs;

/// CLI arguments for the `gatekeeper` subcommand.
#[derive(Parser, Debug, Clone)]
struct GatekeeperArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// Directory to poll and load candidate models from.
    #[arg(long = "test-models-dir", value_name = "DIR", required = true)]
    test_models_dir: String,

    /// Directory to output SGF files and logs to.
    #[arg(long = "sgf-output-dir", value_name = "DIR", required = true)]
    sgf_output_dir: String,

    /// Directory to write accepted models to.
    #[arg(long = "accepted-models-dir", value_name = "DIR", required = true)]
    accepted_models_dir: String,

    /// Directory to write rejected models to.
    #[arg(long = "rejected-models-dir", value_name = "DIR", required = true)]
    rejected_models_dir: String,

    /// Directory where self-play data will be produced if a model passes.
    #[arg(long = "selfplay-dir", value_name = "DIR")]
    selfplay_dir: Option<String>,

    /// Required win proportion for the candidate to be accepted.
    #[arg(
        long = "required-candidate-win-prop",
        value_name = "PROP",
        default_value_t = 0.5
    )]
    required_candidate_win_prop: f64,

    /// Test older models than the latest accepted model instead of auto-rejecting.
    #[arg(long = "no-autoreject-old-models")]
    no_autoreject_old_models: bool,

    /// Terminate instead of waiting for a new net to test.
    #[arg(long = "quit-if-no-nets-to-test")]
    quit_if_no_nets_to_test: bool,
}

/// Public CLI entry point.
pub fn gatekeeper(args: &[String]) -> i32 {
    match gatekeeper_impl(args) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {}", e);
            1
        }
    }
}

fn gatekeeper_impl(args: &[String]) -> Result<(), StringError> {
    let parsed = GatekeeperArgs::try_parse_from(
        std::iter::once(&"gatekeeper".to_string()).chain(args.iter()),
    )
    .map_err(|e| StringError::new(format!("Argument error: {}", e)))?;

    for dir in [
        &parsed.test_models_dir,
        &parsed.sgf_output_dir,
        &parsed.accepted_models_dir,
        &parsed.rejected_models_dir,
    ] {
        if dir.is_empty() {
            return Err(StringError::new("Empty directory specified".to_string()));
        }
        std::fs::create_dir_all(dir)
            .map_err(|e| StringError::new(format!("Could not create dir {}: {}", dir, e)))?;
    }
    if let Some(ref selfplay_dir) = parsed.selfplay_dir {
        if !selfplay_dir.is_empty() {
            std::fs::create_dir_all(selfplay_dir).map_err(|e| {
                StringError::new(format!(
                    "Could not create selfplay dir {}: {}",
                    selfplay_dir, e
                ))
            })?;
        }
    }

    let mut seed_rand = Rand::new();
    let log_file = format!(
        "{}/log{}-{}.log",
        parsed.sgf_output_dir,
        get_compact_date_time_string(),
        kata_core::global::uint64_to_hex_string(seed_rand.next_u64())
    );
    let logger = Arc::new(Logger::new(
        LoggerOptions {
            log_to_stdout: true,
            log_to_stderr: true,
            log_time: true,
        },
        None,
    ));
    logger.add_file(&log_file, true);

    logger.write("Gatekeeper Engine starting...");
    logger.write(&format!(
        "Required candidate win prop: {}",
        parsed.required_candidate_win_prop
    ));

    let cfg = Arc::new(
        parsed
            .common
            .get_config_allow_empty(None)
            .map_err(|e| StringError::new(format!("Could not load config: {}", e)))?,
    );
    parsed.common.log_overrides(&logger);

    let num_game_threads = cfg
        .get_int("numGameThreads", 1, 16384)
        .map_err(|e| StringError::new(e.to_string()))?;
    let num_games_per_gating = cfg
        .get_int64("numGamesPerGating", 1, 1_i64 << 24)
        .map_err(|e| StringError::new(e.to_string()))?;

    let play_settings = PlaySettings::load_for_gatekeeper(&cfg)
        .map_err(|e| StringError::new(format!("Could not load play settings: {}", e)))?;
    let game_runner = Arc::new(
        GameRunner::new(&cfg, play_settings, &logger)
            .map_err(|e| StringError::new(format!("Could not create GameRunner: {}", e)))?,
    );

    let game_initializer = game_runner
        .get_game_initializer()
        .expect("GameRunner should have a GameInitializer");
    let min_board_x_size_used = game_initializer.get_min_board_x_size();
    let min_board_y_size_used = game_initializer.get_min_board_y_size();
    let max_board_x_size_used = game_initializer.get_max_board_x_size();
    let max_board_y_size_used = game_initializer.get_max_board_y_size();

    let base_params = setup::load_single_params(&cfg, SetupFor::Other)
        .map_err(|e| StringError::new(format!("Could not load search params: {}", e)))?;

    setup::initialize_session(&cfg);

    logger.write(&format!(
        "Loaded all config stuff, watching for new neural nets in {}",
        parsed.test_models_dir
    ));
    if !logger.is_logging_to_stdout() {
        println!(
            "Loaded all config stuff, watching for new neural nets in {}",
            parsed.test_models_dir
        );
    }

    let candidate = match find_latest_model(&parsed.test_models_dir, &logger)? {
        Some(info) => info,
        None => {
            if parsed.quit_if_no_nets_to_test {
                logger.write("No nets to test, quitting");
                return Ok(());
            }
            logger.write("No candidate nets found, sleeping is not implemented in this slice");
            return Ok(());
        }
    };

    if candidate.model_file == "/dev/null" {
        logger.write("Skipping /dev/null candidate");
        return Ok(());
    }

    logger.write(&format!(
        "Found new candidate neural net {}",
        candidate.model_name
    ));

    let accepted = match find_latest_model(&parsed.accepted_models_dir, &logger)? {
        Some(info) => info,
        None => {
            return Err(StringError::new(format!(
                "No accepted model found in {}",
                parsed.accepted_models_dir
            )));
        }
    };

    if accepted.model_time > candidate.model_time && !parsed.no_autoreject_old_models {
        logger.write(&format!(
            "Rejecting {} automatically since older than best accepted model",
            candidate.model_name
        ));
        move_model(
            &candidate,
            &parsed.test_models_dir,
            &parsed.rejected_models_dir,
            &logger,
        )?;
        return Ok(());
    }

    let expected_concurrent_evals = cfg
        .get_int("numSearchThreads", 1, 16384)
        .map_err(|e| StringError::new(e.to_string()))?
        * num_game_threads;
    let default_max_batch_size = -1;
    let default_require_exact_nn_len = min_board_x_size_used == max_board_x_size_used
        && min_board_y_size_used == max_board_y_size_used;
    let disable_fp16 = false;
    let expected_sha256 = String::new();

    let accepted_eval = setup::initialize_nn_evaluator(
        accepted.model_name.clone(),
        accepted.model_file.clone(),
        expected_sha256.clone(),
        &cfg,
        &logger,
        &mut seed_rand,
        expected_concurrent_evals,
        max_board_x_size_used,
        max_board_y_size_used,
        default_max_batch_size,
        default_require_exact_nn_len,
        disable_fp16,
        SetupFor::Other,
    )
    .map_err(|e| StringError::new(format!("Could not initialize accepted nn: {}", e)))?;
    logger.write(&format!(
        "Loaded accepted neural net {} from: {}",
        accepted.model_name, accepted.model_file
    ));

    let candidate_eval = setup::initialize_nn_evaluator(
        candidate.model_name.clone(),
        candidate.model_file.clone(),
        expected_sha256,
        &cfg,
        &logger,
        &mut seed_rand,
        expected_concurrent_evals,
        max_board_x_size_used,
        max_board_y_size_used,
        default_max_batch_size,
        default_require_exact_nn_len,
        disable_fp16,
        SetupFor::Other,
    )
    .map_err(|e| StringError::new(format!("Could not initialize candidate nn: {}", e)))?;
    logger.write(&format!(
        "Loaded candidate neural net {} from: {}",
        candidate.model_name, candidate.model_file
    ));

    // Leak the evaluators so that the GameRunner can hold static references.
    let accepted_eval: &'static NnEvaluator = Box::leak(Box::new(accepted_eval));
    let candidate_eval: &'static NnEvaluator = Box::leak(Box::new(candidate_eval));

    cfg.warn_unused_keys(&mut std::io::stdout(), Some(&logger))
        .map_err(|e| StringError::new(e.to_string()))?;

    let game_seed_base = kata_core::global::uint64_to_hex_string(seed_rand.next_u64());
    let mut this_loop_seed_rand = Rand::new();

    let mut num_games_tallied = 0i64;
    let mut baseline_win_points = 0.0;
    let mut candidate_win_points = 0.0;

    for game_idx in 0..num_games_per_gating {
        let baseline_is_black = game_idx % 2 == 0;
        let (bot_b, bot_w) = if baseline_is_black {
            (
                BotSpec {
                    bot_idx: 0,
                    bot_name: accepted.model_name.clone(),
                    nn_eval: Some(accepted_eval),
                    base_params: base_params.clone(),
                },
                BotSpec {
                    bot_idx: 1,
                    bot_name: candidate.model_name.clone(),
                    nn_eval: Some(candidate_eval),
                    base_params: base_params.clone(),
                },
            )
        } else {
            (
                BotSpec {
                    bot_idx: 1,
                    bot_name: candidate.model_name.clone(),
                    nn_eval: Some(candidate_eval),
                    base_params: base_params.clone(),
                },
                BotSpec {
                    bot_idx: 0,
                    bot_name: accepted.model_name.clone(),
                    nn_eval: Some(accepted_eval),
                    base_params: base_params.clone(),
                },
            )
        };

        let seed = format!(
            "{}:{}",
            game_seed_base,
            kata_core::global::uint64_to_hex_string(this_loop_seed_rand.next_u64())
        );
        let logger_ref = &*logger;
        let game_data = game_runner.run_game(
            &seed,
            &bot_b,
            &bot_w,
            None,
            None,
            logger_ref,
            Box::new(|| false),
            None,
            Box::new(|| None),
            Box::new(|_bot_spec, _search| {}),
            Box::new(|_board, _hist, _pla, _loc, _before, _after, _weights, _search| {}),
        );

        let data = match game_data {
            Some(d) => d,
            None => {
                logger.write(&format!("Game {} did not produce data, stopping", game_idx));
                break;
            }
        };

        let (black_points, white_points) = game_points(&data);
        baseline_win_points += if data.b_idx == 0 {
            black_points
        } else {
            white_points
        };
        candidate_win_points += if data.b_idx == 1 {
            black_points
        } else {
            white_points
        };
        num_games_tallied += 1;

        logger.write(&format!(
            "Game {}: baseline {:.2} candidate {:.2}",
            game_idx, baseline_win_points, candidate_win_points
        ));
    }

    let required = parsed.required_candidate_win_prop * num_games_tallied as f64;
    // Candidate wins ties.
    if candidate_win_points + 1e-10 < required {
        logger.write(&format!(
            "Candidate lost match, score {:.3} to {:.3} in {} games, rejecting candidate {}",
            candidate_win_points, baseline_win_points, num_games_tallied, candidate.model_name
        ));
        move_model(
            &candidate,
            &parsed.test_models_dir,
            &parsed.rejected_models_dir,
            &logger,
        )?;
    } else {
        logger.write(&format!(
            "Candidate won match, score {:.3} to {:.3} in {} games, accepting candidate {}",
            candidate_win_points, baseline_win_points, num_games_tallied, candidate.model_name
        ));

        if let Some(ref selfplay_dir) = parsed.selfplay_dir {
            if !selfplay_dir.is_empty() {
                let candidate_dir = format!("{}/{}", selfplay_dir, candidate.model_name);
                std::fs::create_dir_all(&candidate_dir).ok();
                std::fs::create_dir_all(format!("{}/sgfs", candidate_dir)).ok();
                std::fs::create_dir_all(format!("{}/tdata", candidate_dir)).ok();
                std::fs::create_dir_all(format!("{}/vadata", candidate_dir)).ok();
            }
        }

        move_model(
            &candidate,
            &parsed.test_models_dir,
            &parsed.accepted_models_dir,
            &logger,
        )?;
    }

    logger.write("All cleaned up, quitting");
    Ok(())
}

fn find_latest_model(dir: &str, logger: &Logger) -> Result<Option<ModelInfo>, StringError> {
    model_loader::find_latest_model(dir, logger)
        .map_err(|e| StringError::new(format!("Could not find latest model in {}: {}", dir, e.0)))
}

fn game_points(data: &FinishedGameData) -> (f64, f64) {
    if data.end_hist.is_game_finished && data.end_hist.is_no_result {
        let white_points = data.draw_equivalent_wins_for_white;
        return (1.0 - white_points, white_points);
    }

    let mut hist = data.end_hist.clone();
    if !hist.is_game_finished {
        let board = hist.get_recent_board(0).clone();
        hist.end_and_score_game_now(&board);
    }

    if hist.winner == P_BLACK || hist.final_white_minus_black_score < 0.0 {
        (1.0, 0.0)
    } else if hist.winner == P_WHITE || hist.final_white_minus_black_score > 0.0 {
        (0.0, 1.0)
    } else {
        (0.5, 0.5)
    }
}

fn move_model(
    model: &ModelInfo,
    test_models_dir: &str,
    into_dir: &str,
    logger: &Logger,
) -> Result<(), StringError> {
    std::fs::create_dir_all(into_dir)
        .map_err(|e| StringError::new(format!("Could not create dir {}: {}", into_dir, e)))?;

    let test_dir_canon = kata_core::fs::weakly_canonical(test_models_dir);
    let model_dir_canon = kata_core::fs::weakly_canonical(&model.model_dir);

    let dest = format!("{}/{}", into_dir, model.model_name);

    if model_dir_canon == test_dir_canon {
        logger.write(&format!("Moving {} to {}", model.model_file, dest));
        kata_core::fs::rename(&model.model_file, &dest).map_err(|e| {
            StringError::new(format!(
                "Could not rename {} to {}: {}",
                model.model_file, dest, e.0
            ))
        })?;
    } else if model_dir_canon.starts_with(&test_dir_canon) {
        logger.write(&format!("Moving {} to {}", model.model_dir, dest));
        kata_core::fs::rename(&model.model_dir, &dest).map_err(|e| {
            StringError::new(format!(
                "Could not rename {} to {}: {}",
                model.model_dir, dest, e.0
            ))
        })?;
    } else {
        return Err(StringError::new(format!(
            "Model {} does not appear to be a subdir of {}",
            model.model_dir, test_models_dir
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gatekeeper_cfg(num_games: i64) -> String {
        format!(
            "koRules = SIMPLE,POSITIONAL\n\
             scoringRules = AREA\n\
             taxRules = NONE\n\
             multiStoneSuicideLegals = true,false\n\
             hasButtons = true,false\n\
             bSizes = 7\n\
             bSizeRelProbs = 1.0\n\
             komiMean = 7.5\n\
             numGameThreads = 1\n\
             numGamesPerGating = {}\n\
             numSearchThreads = 1\n\
             nnMaxBatchSize = 8\n\
             nnCacheSizePowerOfTwo = 20\n\
             nnMutexPoolSizePowerOfTwo = 16\n\
             maxVisits = 2\n\
             maxPlayouts = 2\n\
             logSearchInfo = false\n\
             logMoves = false\n\
             maxMovesPerGame = 30\n\
             clearBotBeforeSearch = true\n\
             valueWeightExponent = 0\n\
             allowResignation = false\n\
             resignThreshold = -0.9\n\
             resignConsecTurns = 1\n",
            num_games
        )
    }

    fn unique_tmp_dir(prefix: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{}_{}_{}", prefix, std::process::id(), nanos))
    }

    fn touch(path: &std::path::Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"").unwrap();
    }

    #[test]
    fn test_gatekeeper_parses_args() {
        let args = GatekeeperArgs::parse_from([
            "gatekeeper",
            "--config",
            "cfg.cfg",
            "--test-models-dir",
            "test",
            "--sgf-output-dir",
            "sgfs",
            "--accepted-models-dir",
            "accepted",
            "--rejected-models-dir",
            "rejected",
        ]);
        assert_eq!(args.test_models_dir, "test");
        assert_eq!(args.required_candidate_win_prop, 0.5);
    }

    #[test]
    fn test_gatekeeper_runs_match_and_accepts() {
        let tmp = unique_tmp_dir("katago_gatekeeper_run");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let test_dir = tmp.join("test");
        let accepted_dir = tmp.join("accepted");
        let rejected_dir = tmp.join("rejected");
        let sgf_dir = tmp.join("sgfs");
        let cfg_path = tmp.join("gatekeeper.cfg");

        std::fs::create_dir_all(&test_dir).unwrap();
        std::fs::create_dir_all(&accepted_dir).unwrap();
        std::fs::create_dir_all(&rejected_dir).unwrap();
        std::fs::create_dir_all(&sgf_dir).unwrap();

        // Accepted baseline model.
        touch(&accepted_dir.join("accepted.bin.gz"));
        // Candidate model.
        touch(&test_dir.join("candidate.bin.gz"));

        std::fs::write(&cfg_path, gatekeeper_cfg(2)).unwrap();

        let args = vec![
            "--config".to_string(),
            cfg_path.to_string_lossy().to_string(),
            "--test-models-dir".to_string(),
            test_dir.to_string_lossy().to_string(),
            "--sgf-output-dir".to_string(),
            sgf_dir.to_string_lossy().to_string(),
            "--accepted-models-dir".to_string(),
            accepted_dir.to_string_lossy().to_string(),
            "--rejected-models-dir".to_string(),
            rejected_dir.to_string_lossy().to_string(),
            "--required-candidate-win-prop".to_string(),
            "0.0".to_string(),
        ];

        let result = gatekeeper_impl(&args);
        assert!(result.is_ok(), "gatekeeper failed: {:?}", result);
        assert!(
            accepted_dir.join("candidate.bin.gz").exists(),
            "candidate should have been accepted"
        );
    }
}
