//! Self-play command implementation.
//!
//! Corresponds to `MainCmds::selfplay` in `cpp/command/selfplay.cpp`.
//! This slice implements the basic single-process self-play loop: load a config,
//! poll a models directory for the latest neural net, play games via
//! [`GameRunner`], and enqueue finished games to a [`SelfplayManager`] for
//! training-data writing.
//!
//! Advanced features from the C++ original (distributed self-play, model
//! gating, mid-game net switching, and signal handling) are left as stubs or
//! omitted because the underlying Rust infrastructure is not yet ported.

#![allow(
    dead_code,
    clippy::collapsible_if,
    clippy::mixed_case_hex_literals,
    clippy::obfuscated_if_else,
    clippy::needless_borrow,
    clippy::needless_range_loop,
    clippy::redundant_pattern_matching,
    clippy::clone_on_copy,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::significant_drop_tightening
)]

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use clap::Parser;

use kata_core::config::ConfigParser;
use kata_core::global::{StringError, try_string_to_int64, uint64_to_hex_string};
use kata_core::logger::{Logger, LoggerOptions};
use kata_core::rng::Rand;
use kata_core::time::get_compact_date_time_string;
use kata_data::model_loader;
use kata_data::training::TrainingDataWriter;
use kata_game::board::MAX_LEN;
use kata_nn::eval::NnEvaluator;
use kata_nn::version;
use kata_program::play::{BotSpec, GameRunner};
use kata_program::play_settings::PlaySettings;
use kata_program::selfplay_manager::SelfplayManager;
use kata_program::setup::{self, SetupFor};

use crate::cli::CommonArgs;

/// CLI arguments for the `selfplay` subcommand.
#[derive(Parser, Debug, Clone)]
struct SelfplayArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// Directory to poll and load models from.
    #[arg(long = "models-dir", value_name = "DIR", required = true)]
    models_dir: String,

    /// Directory to output training data and SGFs to.
    #[arg(long = "output-dir", value_name = "DIR", required = true)]
    output_dir: String,

    /// Terminate after this many games.
    #[arg(long = "max-games-total", value_name = "NGAMES")]
    max_games_total: Option<String>,
}

/// Public CLI entry point.
pub fn selfplay(args: &[String]) -> i32 {
    match selfplay_impl(args) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {}", e);
            1
        }
    }
}

fn selfplay_impl(args: &[String]) -> Result<(), StringError> {
    let parsed = parse_args(args)?;

    let models_dir = parsed.models_dir;
    let output_dir = parsed.output_dir;
    if models_dir.is_empty() {
        return Err(StringError::new(
            "Empty directory specified for --models-dir",
        ));
    }
    if output_dir.is_empty() {
        return Err(StringError::new(
            "Empty directory specified for --output-dir",
        ));
    }

    let max_games_total = match parsed.max_games_total {
        Some(s) => try_string_to_int64(&s)
            .filter(|&n| n > 0)
            .ok_or_else(|| StringError::new("-max-games-total must be a positive integer"))?,
        None => i64::MAX,
    };

    std::fs::create_dir_all(&output_dir)
        .map_err(|e| StringError::new(format!("Could not create output dir: {}", e)))?;
    std::fs::create_dir_all(&models_dir)
        .map_err(|e| StringError::new(format!("Could not create models dir: {}", e)))?;

    let mut seed_rand = Rand::new();
    let log_file = format!(
        "{}/log{}-{}.log",
        output_dir,
        get_compact_date_time_string(),
        uint64_to_hex_string(seed_rand.next_u64())
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

    logger.write("Self Play Engine starting...");

    let cfg = Arc::new(parsed.common.get_config("")?);
    parsed.common.log_overrides(&logger);

    // Load runner settings.
    let num_game_threads = cfg
        .get_int("numGameThreads", 1, 16384)
        .map_err(|e| StringError::new(e.to_string()))?;
    let game_seed_base = uint64_to_hex_string(seed_rand.next_u64());

    let data_board_len = cfg
        .get_int("dataBoardLen", 3, MAX_LEN as i32)
        .map_err(|e| StringError::new(e.to_string()))?;
    let inputs_version = if cfg.contains("inputsVersion") {
        cfg.get_int("inputsVersion", 0, 10000)
            .map_err(|e| StringError::new(e.to_string()))?
    } else {
        version::get_inputs_version(version::DEFAULT_MODEL_VERSION).unwrap_or(7)
    };
    let max_data_queue_size = cfg
        .get_int("maxDataQueueSize", 1, 1000000)
        .map_err(|e| StringError::new(e.to_string()))? as usize;
    let max_rows_per_train_file = cfg
        .get_int("maxRowsPerTrainFile", 1, 100000000)
        .map_err(|e| StringError::new(e.to_string()))?;
    let first_file_rand_min_prop = cfg
        .get_double("firstFileRandMinProp", 0.0, 1.0)
        .map_err(|e| StringError::new(e.to_string()))?;
    let log_games_every = cfg
        .get_int64("logGamesEvery", 1, 1000000)
        .map_err(|e| StringError::new(e.to_string()))?;
    let switch_nets_mid_game = cfg
        .get_bool("switchNetsMidGame")
        .map_err(|e| StringError::new(e.to_string()))?;

    let base_params = setup::load_single_params(&cfg, SetupFor::Other)
        .map_err(|e| StringError::new(format!("Could not load search params: {}", e)))?;

    let play_settings = PlaySettings::load_for_selfplay(&cfg, false)
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

    let manager = SelfplayManager::new(
        max_data_queue_size,
        Some(Arc::clone(&logger)),
        log_games_every,
        true,
    );

    // Load the initial neural net.
    let loaded = try_load_latest_model(
        &cfg,
        &logger,
        &manager,
        &models_dir,
        &output_dir,
        inputs_version,
        max_rows_per_train_file,
        first_file_rand_min_prop,
        data_board_len,
        num_game_threads,
        min_board_x_size_used,
        max_board_x_size_used,
        min_board_y_size_used,
        max_board_y_size_used,
        None,
    )?;
    if !loaded {
        return Err(StringError::new(
            "Either could not load latest neural net or access/write appropriate directories",
        ));
    }

    cfg.warn_unused_keys(&mut std::io::stdout(), Some(&logger))
        .map_err(|e| StringError::new(format!("Could not warn unused keys: {}", e)))?;

    setup::initialize_session(&cfg);

    logger.write("Loaded all config stuff, starting self play");
    if !logger.is_logging_to_stdout() {
        println!("Loaded all config stuff, starting self play");
    }

    let should_stop = Arc::new(AtomicBool::new(false));
    let stop_pair: Arc<(Mutex<bool>, Condvar)> = Arc::new((Mutex::new(false), Condvar::new()));
    let num_games_started = Arc::new(AtomicI64::new(0));

    // Spawn game-loop threads.
    let mut game_handles = Vec::with_capacity(num_game_threads as usize);
    for thread_idx in 0..num_game_threads {
        let logger = Arc::clone(&logger);
        let manager = Arc::clone(&manager);
        let game_runner = Arc::clone(&game_runner);
        let should_stop = Arc::clone(&should_stop);
        let num_games_started = Arc::clone(&num_games_started);
        let base_params = base_params.clone();
        let game_seed_base = game_seed_base.clone();

        let handle = thread::spawn(move || -> Result<(), StringError> {
            let mut this_loop_seed_rand = Rand::new();
            let mut prev_model_name = String::new();
            loop {
                if should_stop.load(Ordering::Relaxed) {
                    break;
                }
                let nn_eval = manager.acquire_latest().ok_or_else(|| {
                    StringError::new("Game loop thread could not acquire latest model")
                })?;
                let model_name = unsafe { &*nn_eval }.model_name().to_string();
                if prev_model_name != model_name {
                    prev_model_name = model_name.clone();
                    logger.write(&format!(
                        "Game loop thread {} starting game on new neural net: {}",
                        thread_idx, prev_model_name
                    ));
                }

                let game_idx = num_games_started.fetch_add(1, Ordering::AcqRel);
                if game_idx >= max_games_total {
                    manager.release_by_nn_eval(nn_eval);
                    break;
                }
                manager.count_one_game_started(nn_eval).map_err(|e| {
                    StringError::new(format!("count_one_game_started failed: {}", e.message))
                })?;

                let bot_spec = BotSpec {
                    bot_idx: 0,
                    bot_name: model_name,
                    nn_eval: Some(unsafe { &*nn_eval }),
                    base_params: base_params.clone(),
                };
                let seed = format!(
                    "{}:{}",
                    game_seed_base,
                    uint64_to_hex_string(this_loop_seed_rand.next_u64())
                );

                let should_stop_for_game = Arc::clone(&should_stop);
                let check_for_new_nn_eval: Box<dyn Fn() -> Option<*mut NnEvaluator> + Send + Sync> =
                    if switch_nets_mid_game {
                        let manager = Arc::clone(&manager);
                        let current = nn_eval as usize;
                        Box::new(move || {
                            let latest = manager.acquire_latest()?;
                            if std::ptr::eq(latest, current as *const NnEvaluator) {
                                manager.release_by_nn_eval(latest);
                                None
                            } else {
                                // The game runner takes ownership of the new eval
                                // and will release the old one when it is done.
                                Some(latest as *mut NnEvaluator)
                            }
                        })
                    } else {
                        Box::new(|| None)
                    };

                let logger_ref = &*logger;
                let game_data = game_runner.run_game(
                    &seed,
                    &bot_spec,
                    &bot_spec,
                    None,
                    None,
                    logger_ref,
                    Box::new(move || should_stop_for_game.load(Ordering::Relaxed)),
                    None,
                    check_for_new_nn_eval,
                    Box::new(|_bot_spec, _search| {}),
                    Box::new(|_board, _hist, _pla, _loc, _before, _after, _weights, _search| {}),
                );

                let should_continue = game_data.is_some();
                if let Some(game_data) = game_data {
                    manager
                        .enqueue_data_to_write_by_nn_eval(nn_eval, game_data)
                        .map_err(|e| {
                            StringError::new(format!("enqueue_data_to_write failed: {}", e.message))
                        })?;
                }
                manager.release_by_nn_eval(nn_eval);

                if !should_continue {
                    break;
                }
            }
            logger.write(&format!("Game loop thread {} terminating", thread_idx));
            Ok(())
        });
        game_handles.push(handle);
    }

    // Spawn the model-polling thread.
    let model_load_handle = {
        let logger = Arc::clone(&logger);
        let manager = Arc::clone(&manager);
        let cfg = Arc::clone(&cfg);
        let should_stop = Arc::clone(&should_stop);
        let stop_pair = Arc::clone(&stop_pair);
        thread::spawn(move || -> Result<(), StringError> {
            logger.write("Model loading loop thread starting");
            loop {
                if should_stop.load(Ordering::Relaxed) {
                    break;
                }
                let last_net_name = manager.get_latest_model_name().ok();
                let _ = try_load_latest_model(
                    &cfg,
                    &logger,
                    &manager,
                    &models_dir,
                    &output_dir,
                    inputs_version,
                    max_rows_per_train_file,
                    first_file_rand_min_prop,
                    data_board_len,
                    num_game_threads,
                    min_board_x_size_used,
                    max_board_x_size_used,
                    min_board_y_size_used,
                    max_board_y_size_used,
                    last_net_name.as_deref(),
                );

                if should_stop.load(Ordering::Relaxed) {
                    break;
                }
                let (lock, cvar) = &*stop_pair;
                let mut stopped = lock.lock().unwrap();
                while !*stopped {
                    let result = cvar.wait_timeout(stopped, Duration::from_secs(20));
                    stopped = result.unwrap().0;
                }
            }
            logger.write("Model loading loop thread terminating");
            Ok(())
        })
    };

    // Wait for game threads, propagating any errors.
    let mut first_error: Option<StringError> = None;
    for handle in game_handles {
        match handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) if first_error.is_none() => first_error = Some(e),
            Err(_) if first_error.is_none() => {
                first_error = Some(StringError::new("Game loop thread panicked"));
            }
            _ => {}
        }
    }

    // Signal the model-loading thread to shut down and wait for it.
    should_stop.store(true, Ordering::Relaxed);
    {
        let (lock, cvar) = &*stop_pair;
        let mut stopped = lock.lock().unwrap();
        *stopped = true;
        cvar.notify_all();
    }
    if let Err(_) = model_load_handle.join() {
        if first_error.is_none() {
            first_error = Some(StringError::new("Model loading thread panicked"));
        }
    }

    if let Some(e) = first_error {
        return Err(e);
    }

    logger.write("All cleaned up, quitting");
    Ok(())
}

fn parse_args(args: &[String]) -> Result<SelfplayArgs, StringError> {
    SelfplayArgs::try_parse_from(std::iter::once(&"selfplay".to_string()).chain(args.iter()))
        .map_err(|e| StringError::new(format!("Argument error: {}", e)))
}

#[allow(clippy::too_many_arguments)]
fn try_load_latest_model(
    cfg: &ConfigParser,
    logger: &Logger,
    manager: &SelfplayManager,
    models_dir: &str,
    output_dir: &str,
    inputs_version: i32,
    max_rows_per_file: i32,
    first_file_rand_min_prop: f64,
    data_board_len: i32,
    num_game_threads: i32,
    min_board_x_size_used: i32,
    max_board_x_size_used: i32,
    min_board_y_size_used: i32,
    max_board_y_size_used: i32,
    last_net_name: Option<&str>,
) -> Result<bool, StringError> {
    let model_info = match model_loader::find_latest_model(models_dir, logger)
        .map_err(|e| StringError::new(format!("Error finding latest model: {}", e.0)))?
    {
        Some(m) => m,
        None => return Ok(false),
    };

    if model_info.model_file == "/dev/null" {
        return Ok(false);
    }

    if last_net_name
        .map(|name| name == model_info.model_name)
        .unwrap_or(false)
    {
        return Ok(false);
    }

    logger.write(&format!("Found new neural net {}", model_info.model_name));

    let num_search_threads = cfg
        .get_int("numSearchThreads", 1, 4096)
        .map_err(|e| StringError::new(e.to_string()))?;
    let expected_concurrent_evals = num_search_threads * num_game_threads;
    let default_require_exact_nn_len = min_board_x_size_used == max_board_x_size_used
        && min_board_y_size_used == max_board_y_size_used;
    let default_max_batch_size = -1;
    let disable_fp16 = false;
    let expected_sha256 = "";

    let mut rand = Rand::new();
    let nn_eval = setup::initialize_nn_evaluator(
        model_info.model_name.clone(),
        model_info.model_file.clone(),
        expected_sha256.to_string(),
        cfg,
        logger,
        &mut rand,
        expected_concurrent_evals,
        max_board_x_size_used,
        max_board_y_size_used,
        default_max_batch_size,
        default_require_exact_nn_len,
        disable_fp16,
        SetupFor::Other,
    )
    .map_err(|e| StringError::new(format!("Could not initialize NN evaluator: {}", e.message)))?;

    logger.write(&format!(
        "Loaded latest neural net {} from: {}",
        model_info.model_name, model_info.model_file
    ));

    let model_output_dir = PathBuf::from(output_dir).join(&model_info.model_name);
    let sgf_output_dir = model_output_dir.join("sgfs");
    let tdata_output_dir = model_output_dir.join("tdata");

    std::fs::create_dir_all(&model_output_dir)
        .map_err(|e| StringError::new(format!("Could not create model output dir: {}", e)))?;
    std::fs::create_dir_all(&sgf_output_dir)
        .map_err(|e| StringError::new(format!("Could not create sgf output dir: {}", e)))?;
    std::fs::create_dir_all(&tdata_output_dir)
        .map_err(|e| StringError::new(format!("Could not create tdata output dir: {}", e)))?;

    let cfg_path = model_output_dir.join(format!(
        "selfplay-{}.cfg",
        uint64_to_hex_string(rand.next_u64())
    ));
    std::fs::write(&cfg_path, cfg.contents())
        .map_err(|e| StringError::new(format!("Could not write selfplay cfg: {}", e)))?;

    let tdata_writer = TrainingDataWriter::new(
        tdata_output_dir.to_str().unwrap_or(""),
        inputs_version,
        max_rows_per_file,
        first_file_rand_min_prop,
        data_board_len,
        data_board_len,
        &uint64_to_hex_string(rand.next_u64()),
    )
    .map_err(|e| StringError::new(format!("Could not create TrainingDataWriter: {}", e.0)))?;

    let sgf_out: Option<Box<dyn Write + Send + Sync>> = if !sgf_output_dir.as_os_str().is_empty() {
        let sgf_path =
            sgf_output_dir.join(format!("{}.sgfs", uint64_to_hex_string(rand.next_u64())));
        let file = std::fs::File::create(&sgf_path)
            .map_err(|e| StringError::new(format!("Could not create SGF output file: {}", e)))?;
        Some(Box::new(file) as Box<dyn Write + Send + Sync>)
    } else {
        None
    };

    manager
        .load_model_and_start_data_writing(nn_eval, tdata_writer, sgf_out)
        .map_err(|e| {
            StringError::new(format!("Could not load model into manager: {}", e.message))
        })?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_text() -> &'static str {
        "koRules = SIMPLE,POSITIONAL\n\
         scoringRules = AREA\n\
         taxRules = NONE\n\
         multiStoneSuicideLegals = true,false\n\
         hasButtons = true,false\n\
         bSizes = 7\n\
         bSizeRelProbs = 1.0\n\
         komiMean = 7.5\n\
         numGameThreads = 1\n\
         dataBoardLen = 7\n\
         maxDataQueueSize = 10\n\
         maxRowsPerTrainFile = 1\n\
         firstFileRandMinProp = 1.0\n\
         logGamesEvery = 1\n\
         switchNetsMidGame = false\n\
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
         initGamesWithPolicy = false\n\
         compensateAfterPolicyInitProb = 0.0\n\
         sidePositionProb = 0.0\n\
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
         minAsymmetricCompensateKomiProb = 0.0\n\
         policySurpriseDataWeight = 0.0\n\
         valueSurpriseDataWeight = 0.0\n\
         valueWeightExponent = 0\n"
    }

    #[test]
    fn test_parse_args() {
        let args = SelfplayArgs::parse_from([
            "selfplay",
            "--config",
            "cfg.cfg",
            "--models-dir",
            "models",
            "--output-dir",
            "out",
            "--max-games-total",
            "5",
        ]);
        assert_eq!(args.models_dir, "models");
        assert_eq!(args.output_dir, "out");
        assert_eq!(args.max_games_total, Some("5".to_string()));
    }

    fn unique_tmp_dir(prefix: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{}_{}_{}", prefix, std::process::id(), nanos))
    }

    #[test]
    fn test_invalid_max_games_total() {
        let tmp = unique_tmp_dir("katago_selfplay_invalid");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let cfg_path = tmp.join("cfg.cfg");
        std::fs::write(&cfg_path, cfg_text()).unwrap();
        let models_dir = tmp.join("models");
        std::fs::create_dir_all(&models_dir).unwrap();
        std::fs::write(models_dir.join("model"), b"").unwrap();
        let output_dir = tmp.join("out");

        let args = vec![
            "--config".to_string(),
            cfg_path.to_str().unwrap().to_string(),
            "--models-dir".to_string(),
            models_dir.to_str().unwrap().to_string(),
            "--output-dir".to_string(),
            output_dir.to_str().unwrap().to_string(),
            "--max-games-total".to_string(),
            "abc".to_string(),
        ];
        assert!(selfplay_impl(&args).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_selfplay_runs_and_writes_data() {
        let tmp = unique_tmp_dir("katago_selfplay_run");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let cfg_path = tmp.join("cfg.cfg");
        std::fs::write(&cfg_path, cfg_text()).unwrap();
        let models_dir = tmp.join("models");
        std::fs::create_dir_all(&models_dir).unwrap();
        let model_file = models_dir.join("model.bin.gz");
        std::fs::write(&model_file, b"").unwrap();
        assert!(model_file.exists());
        let output_dir = tmp.join("out");

        let args = vec![
            "--config".to_string(),
            cfg_path.to_str().unwrap().to_string(),
            "--models-dir".to_string(),
            models_dir.to_str().unwrap().to_string(),
            "--output-dir".to_string(),
            output_dir.to_str().unwrap().to_string(),
            "--max-games-total".to_string(),
            "2".to_string(),
        ];
        selfplay_impl(&args).expect("selfplay should run successfully");

        // `model.bin.gz` is a generic model name, so the output directory is named
        // after its parent directory ("models").
        let model_output_dir = output_dir.join("models");
        assert!(model_output_dir.exists(), "model output dir should exist");
        let tdata_dir = model_output_dir.join("tdata");
        assert!(tdata_dir.exists(), "tdata dir should exist");
        let entries: Vec<_> = std::fs::read_dir(&tdata_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert!(
            !entries.is_empty(),
            "tdata dir should contain at least one training data file"
        );
        let has_npz = entries.iter().any(|e| {
            e.path()
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s == "npz")
                .unwrap_or(false)
        });
        assert!(has_npz, "tdata dir should contain a .npz file");

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
