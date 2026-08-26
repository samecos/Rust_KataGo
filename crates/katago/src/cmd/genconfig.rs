//! `genconfig` command implementation.
//!
//! Interactively generates and performance-tunes a GTP config file.
//! Corresponds to `MainCmds::genconfig` in `cpp/command/benchmark.cpp`.
//!
//! Deviations from upstream, both forced by this port's runtime backend
//! selection and single-consumer NN batching design:
//! - An extra prompt asks which `nnBackend` to use (cuda/trt/dummy) and the
//!   choice is written into the generated config; upstream has no such key
//!   because its backend is fixed at compile time.
//! - The "2 NN server threads per GPU" extra tuning test is omitted:
//!   `NnEvaluator::set_num_threads` is a no-op here and the evaluator
//!   deliberately runs a single batching consumer per model.

use std::fs as std_fs;
use std::io::{BufRead, Write};

use clap::Parser;

use kata_core::config::ConfigParser;
use kata_core::fs;
use kata_core::global::{self, StringError};
use kata_core::logger::{Logger, LoggerOptions};
use kata_data::sgf::CompactSgf;
use kata_game::rules::Rules;
use kata_nn::backend::Backend;
use kata_nn::eval::NnEvaluator;
use kata_program::gtp_config::make_config;
use kata_program::setup::{self, SetupFor};

use crate::cli::CommonArgs;
use crate::cmd::benchmark::{
    self, BenchmarkResults, DEFAULT_SECONDS_PER_GAME_MOVE,
};

/// Require at least this much speedup before writing `nnMaxBatchSize` into the
/// generated config, so run-to-run noise doesn't flip the recommendation.
const EXTRA_TUNE_MIN_SPEEDUP_FACTOR: f64 = 1.03;

/// CLI arguments for the `genconfig` subcommand.
///
/// Mirrors upstream: only `-model` and `-output` (upstream `KataGoCommandLine`
/// for genconfig adds no config arguments).
#[derive(Parser, Debug, Clone)]
struct GenConfigArgs {
    /// Neural net model file.
    #[arg(long = "model", value_name = "FILE")]
    model: Option<String>,

    /// Path to write new config.
    #[arg(long = "output", value_name = "FILE", default_value = "gtp.cfg")]
    output: String,
}

/// Config values accumulated from the interactive session, rendered into the
/// final config file.
struct GenConfigState {
    rules: Rules,
    max_visits: i64,
    max_playouts: i64,
    max_time: f64,
    max_ponder_time: f64,
    device_idxs: Vec<i32>,
    nn_cache_size_power_of_two: i32,
    nn_mutex_pool_size_power_of_two: i32,
    num_search_threads: i32,
    nn_max_batch_size: i32,
    backend: String,
}

impl GenConfigState {
    /// Defaults matching upstream `MainCmds::genconfig` initial values.
    fn new() -> Self {
        Self {
            rules: Rules::default(),
            max_visits: 1_i64 << 50,
            max_playouts: 1_i64 << 50,
            max_time: 1e20,
            max_ponder_time: -1.0,
            device_idxs: Vec::new(),
            nn_cache_size_power_of_two: 20,
            nn_mutex_pool_size_power_of_two: 16,
            num_search_threads: 6,
            nn_max_batch_size: -1,
            backend: "dummybackend".to_string(),
        }
    }

    fn contents(&self) -> String {
        let contents = make_config(
            &self.rules,
            self.max_visits,
            self.max_playouts,
            self.max_time,
            self.max_ponder_time,
            &self.device_idxs,
            // The Rust evaluator runs a single batching consumer per model by
            // design, so this is always 1 (upstream also tests 2 here).
            1,
            self.nn_max_batch_size,
            self.nn_cache_size_power_of_two,
            self.nn_mutex_pool_size_power_of_two,
            self.num_search_threads,
        );
        insert_backend_line(contents, &self.backend)
    }
}

/// Public CLI entry point.
pub fn genconfig(args: &[String]) -> i32 {
    let stdin = std::io::stdin();
    let mut stdin_lock = stdin.lock();
    let mut out = std::io::stdout();
    match genconfig_impl(args, &mut stdin_lock, &mut out) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {}", e);
            1
        }
    }
}

/// Internal testable entry point.
fn genconfig_impl(
    args: &[String],
    input: &mut dyn BufRead,
    out: &mut dyn Write,
) -> Result<(), StringError> {
    let parsed = parse_args(args)?;

    let model_file = {
        let common = CommonArgs {
            model: parsed.model.clone(),
            ..CommonArgs::default()
        };
        common.get_model_file()?
    };
    let model_file_is_default = parsed.model.as_deref().unwrap_or("").is_empty();
    let output_file = parsed.output;

    // Overwrite confirmation.
    if fs::exists(&output_file) {
        let overwrite = prompt_and_parse_input(
            input,
            out,
            &format!(
                "File {} already exists, okay to overwrite it with an entirely new config (y/n)?\n",
                output_file
            ),
            parse_yn,
        )?;
        if !overwrite {
            writeln!(out, "Please provide an alternate file path to output the generated config to via '-output NEWFILEPATH'")
                .map_err(io_err)?;
            return Ok(());
        }
    }

    let sgf = CompactSgf::parse(benchmark::get_benchmark_sgf_data(19))
        .map_err(|e| StringError::new(format!("Could not parse built-in SGF: {}", e)))?;

    // Config values accumulated from the interactive session.
    let mut state = GenConfigState::new();

    banner(out, "RULES")?;

    {
        let prompt = "What rules should KataGo use by default for play and analysis?\n\
                      (chinese, japanese, korean, tromp-taylor, aga, chinese-ogs, new-zealand, bga, stone-scoring, aga-button):\n";
        state.rules = prompt_and_parse_input(input, out, prompt, parse_rules_line)?;
    }

    banner(out, "SEARCH LIMITS")?;

    let use_search_limit = prompt_and_parse_input(
        input,
        out,
        "When playing games, KataGo will always obey the time controls given by the GUI/tournament/match/online server.\n\
         But you can specify an additional limit to make KataGo move much faster. This does NOT affect analysis/review,\n\
         only affects playing games. Add a limit? (y/n) (default n):\n",
        parse_optional_yn_default_no,
    )?;

    if !use_search_limit {
        prompt_and_continue(
            input,
            out,
            "NOTE: No limits configured for KataGo. KataGo will obey time controls provided by the GUI or server or match script\n\
             but if they don't specify any, when playing games KataGo may think forever without moving. (press enter to continue)\n",
        )?;
    } else {
        let what_limit = prompt_and_parse_input(
            input,
            out,
            "What to limit per move? Visits, playouts, or seconds?:\n",
            parse_search_limit_kind,
        )?;
        match what_limit.as_str() {
            "visits" => {
                state.max_visits = prompt_and_parse_input(
                    input,
                    out,
                    "Specify max number of visits/move when playing games (doesn't affect analysis), leave blank for default (500):\n",
                    parse_visits_limit,
                )?;
            }
            "playouts" => {
                state.max_playouts = prompt_and_parse_input(
                    input,
                    out,
                    "Specify max number of playouts/move when playing games (doesn't affect analysis), leave blank for default (300):\n",
                    parse_playouts_limit,
                )?;
            }
            "seconds" => {
                state.max_time = prompt_and_parse_input(
                    input,
                    out,
                    "Specify max time/move in seconds when playing games (doesn't affect analysis). Leave blank for default (10):\n",
                    parse_time_limit,
                )?;
            }
            _ => unreachable!("parse_search_limit_kind returns a known kind"),
        }
    }

    let use_ponder = prompt_and_parse_input(
        input,
        out,
        "When playing games, KataGo can optionally ponder during the opponent's turn. This gives faster/stronger play\n\
         in real games but should NOT be enabled if you are running tests with fixed limits (pondering may exceed those\n\
         limits), or to avoid stealing the opponent's compute time when testing two bots on the same machine.\n\
         Enable pondering? (y/n, default n):",
        parse_optional_yn_default_no,
    )?;

    if use_ponder {
        state.max_ponder_time = prompt_and_parse_input(
            input,
            out,
            "Specify max num seconds KataGo should ponder during the opponent's turn. Leave blank for no limit:\n",
            parse_ponder_time,
        )?;
    }

    banner(out, "GPUS AND RAM")?;

    // Rust-port-specific: pick the runtime backend (upstream compiles exactly
    // one backend in, so it has no such question).
    {
        let available = available_backends();
        let default_backend = if available.contains(&"cudabackend") {
            "cudabackend"
        } else if available.contains(&"trtbackend") {
            "trtbackend"
        } else {
            "dummybackend"
        };
        let prompt = format!(
            "Which neural net backend should this Rust port of KataGo use? ({}), leave blank for default ({}):\n",
            available.join(", "),
            default_backend
        );
        state.backend = prompt_and_parse_input(input, out, &prompt, |line| {
            parse_backend_choice(line, &available, default_backend)
        })?;
    }

    {
        writeln!(out, "Finding available GPU-like devices...")
            .map_err(io_err)?;
        print_backend_devices(&state.backend);
        writeln!(out).map_err(io_err)?;

        state.device_idxs = prompt_and_parse_input(
            input,
            out,
            "Specify devices/GPUs to use (for example \"0,1,2\" to use devices 0, 1, and 2). Leave blank for a default SINGLE-GPU config:\n",
            parse_device_idxs,
        )?;
    }

    {
        let (cache_pow, mutex_pow) = prompt_and_parse_input(
            input,
            out,
            "By default, KataGo will cache up to about 3GB of positions in memory (RAM), in addition to\n\
             whatever the current search is using. Specify a different max in GB or leave blank for default:\n",
            parse_ram_limit,
        )?;
        state.nn_cache_size_power_of_two = cache_pow;
        state.nn_mutex_pool_size_power_of_two = mutex_pow;
    }

    banner(out, "PERFORMANCE TUNING")?;

    let mut skip_thread_tuning = false;
    if fs::exists(&output_file) {
        let old_config_num_search_threads = read_num_search_threads(&output_file);
        match old_config_num_search_threads {
            Err(_) => {
                writeln!(out, "NOTE: Overwritten config does not specify numSearchThreads or otherwise could not be parsed.")
                    .map_err(io_err)?;
                writeln!(out, "Beginning performance tuning to set this.")
                    .map_err(io_err)?;
            }
            Ok(old_threads) => {
                skip_thread_tuning = prompt_and_parse_input(
                    input,
                    out,
                    &format!(
                        "Actually {} already exists, can skip performance tuning if desired and just use\n\
                         the number of threads ({}) already in that config (all other settings will still be overwritten).\n\
                         Skip performance tuning (y/n)?\n",
                        output_file, old_threads
                    ),
                    parse_yn,
                )?;
                if skip_thread_tuning {
                    state.num_search_threads = old_threads;
                }
            }
        }
    }

    let write_config_file = |contents: &str| -> Result<(), StringError> {
        std_fs::write(&output_file, contents)
            .map_err(|e| StringError::new(format!("Could not write {}: {}", output_file, e)))
    };

    if !skip_thread_tuning {
        let max_visits_from_user = prompt_and_parse_input(
            input,
            out,
            "Specify number of visits to use test/tune performance with, leave blank for default based on GPU speed.\n\
             Use large number for more accurate results, small if your GPU is old and this is taking forever:\n",
            parse_tuning_visits,
        )?;

        let seconds_per_game_move = prompt_and_parse_input(
            input,
            out,
            &format!(
                "Specify number of seconds/move to optimize performance for (default {}), leave blank for default:\n",
                global::double_to_string(DEFAULT_SECONDS_PER_GAME_MOVE)
            ),
            parse_tuning_seconds,
        )?;

        let config_file_contents = state.contents();
        let cfg = ConfigParser::from_str(&config_file_contents, false, true)
            .map_err(|e| StringError::new(format!("Could not parse generated config: {}", e)))?;

        let logger: &'static Logger = Box::leak(Box::new(Logger::new(
            LoggerOptions {
                log_to_stdout: true,
                log_to_stderr: true,
                log_time: true,
            },
            None,
        )));
        logger.write("Loading model and initializing benchmark...");

        let mut params = setup::load_single_params(&cfg, SetupFor::Benchmark)
            .map_err(|e| StringError::new(format!("Could not load params: {}", e)))?;
        // Like the benchmark command: tune with a fixed visit count and no
        // friendly-move shortcuts.
        params.max_visits = benchmark::DEFAULT_MAX_VISITS;
        params.max_playouts = benchmark::DEFAULT_MAX_VISITS;
        params.max_time = 1e20;
        params.search_factor_after_one_pass = 1.0;
        params.search_factor_after_two_pass = 1.0;

        setup::initialize_session(&cfg);

        // Tune with the batch sizing the backend derives on its own per thread
        // count: the generated config omits nnMaxBatchSize unless the extra
        // tuning below measures that half batches are faster.
        let get_desired_batch_size = |_: i32, nn_eval: &NnEvaluator| nn_eval.max_batch_size();

        let mut nn_eval: Option<&'static NnEvaluator> = None;
        let mut max_threads_for_current_nneval: i32 = -1;
        // The closure only uses the warm-start shape of the params, which is
        // not affected by the later max-visits adjustments, so a snapshot
        // avoids holding a borrow of `params` across those assignments.
        let params_for_realloc = params.clone();
        let reallocate_nneval =
            |max_num_threads: i32,
             nn_eval: &mut Option<&'static NnEvaluator>,
             max_threads: &mut i32| {
                if *max_threads >= max_num_threads {
                    return;
                }
                let new_eval = benchmark::create_nneval(
                    max_num_threads,
                    &sgf,
                    &model_file,
                    logger,
                    &cfg,
                    &params_for_realloc,
                );
                *nn_eval = Some(new_eval);
                *max_threads = max_num_threads;
            };

        writeln!(out).map_err(io_err)?;

        let max_visits: i64;
        if max_visits_from_user > 0 {
            max_visits = max_visits_from_user;
            reallocate_nneval(
                benchmark::TERNARY_SEARCH_INITIAL_MAX,
                &mut nn_eval,
                &mut max_threads_for_current_nneval,
            );
        } else {
            writeln!(out, "Running quick initial benchmark at 16 threads!")
                .map_err(io_err)?;
            reallocate_nneval(
                std::cmp::max(16, benchmark::TERNARY_SEARCH_INITIAL_MAX),
                &mut nn_eval,
                &mut max_threads_for_current_nneval,
            );
            let nn_eval_ref = nn_eval.unwrap();
            let results = benchmark::do_fixed_tune_threads(
                &params,
                &sgf,
                3,
                nn_eval_ref,
                logger,
                seconds_per_game_move,
                &[16],
                false,
                &get_desired_batch_size,
                out,
            )?;
            let visits_per_second =
                results[0].total_visits as f64 / (results[0].total_seconds + 0.00001);
            // Make tests use about 2 seconds each.
            let mut v = (2.0 * visits_per_second / 100.0).round() as i64 * 100;
            v = v.clamp(200, 10000);
            max_visits = v;
        }

        params.max_visits = max_visits;
        params.max_playouts = max_visits;

        let num_positions_per_game = 10;

        banner(out, "TUNING NOW")?;
        writeln!(out, "Tuning using {} visits.", max_visits).map_err(io_err)?;

        let nn_eval_ref = nn_eval.unwrap();
        let results = benchmark::do_auto_tune_threads(
            &params,
            &sgf,
            num_positions_per_game,
            nn_eval_ref,
            logger,
            seconds_per_game_move,
            &reallocate_nneval,
            &get_desired_batch_size,
            out,
        )?;

        benchmark::print_elo_comparison(&results, seconds_per_game_move, out)?;
        let mut best_idx = 0usize;
        for i in 1..results.len() {
            if results[i].compute_elo_effect(seconds_per_game_move)
                > results[best_idx].compute_elo_effect(seconds_per_game_move)
            {
                best_idx = i;
            }
        }
        writeln!(out, "Using {} numSearchThreads!", results[best_idx].num_threads)
            .map_err(io_err)?;
        state.num_search_threads = results[best_idx].num_threads;

        // Write the config as tuned so far, so the interactive answers are not
        // lost if the extra test below dies. The final write overwrites this.
        write_config_file(&state.contents())?;

        // Extra test at the final thread count: half batch size. (Upstream
        // also tests 2 NN server threads per GPU on CUDA/ROCM; omitted here,
        // see the module docs.)
        let best_threads = results[best_idx].num_threads;
        let half_batch_size = (best_threads + 1) / 2;
        if best_threads >= 2 && half_batch_size < best_threads {
            writeln!(out).map_err(io_err)?;
            writeln!(
                out,
                "Running additional tests of a few other settings at numSearchThreads = {}.",
                best_threads
            )
            .map_err(io_err)?;

            let mut this_params = params.clone();
            this_params.num_threads = best_threads;

            writeln!(out, "Re-measuring the current recommendation as a baseline:")
                .map_err(io_err)?;
            nn_eval_ref.set_current_batch_size(nn_eval_ref.max_batch_size());
            let baseline = benchmark::benchmark_search_on_positions_and_print(
                this_params.clone(),
                &sgf,
                num_positions_per_game,
                nn_eval_ref,
                None,
                seconds_per_game_move,
                false,
                logger,
                out,
            )?;

            writeln!(
                out,
                "Testing a max batch size of {}, half the search threads, which can pipeline better on some GPUs:",
                half_batch_size
            )
            .map_err(io_err)?;
            nn_eval_ref.set_current_batch_size(half_batch_size);
            let half_result = benchmark::benchmark_search_on_positions_and_print(
                this_params,
                &sgf,
                num_positions_per_game,
                nn_eval_ref,
                Some(&baseline),
                seconds_per_game_move,
                true,
                logger,
                out,
            )?;

            let speedup = visits_per_second(&half_result) / (visits_per_second(&baseline) + 0.00001);
            if speedup >= EXTRA_TUNE_MIN_SPEEDUP_FACTOR {
                writeln!(out, "Half batch size was {:.1}% faster, will recommend it.", 100.0 * (speedup - 1.0))
                    .map_err(io_err)?;
                state.nn_max_batch_size = half_batch_size;
                writeln!(
                    out,
                    "Using nnMaxBatchSize = {} in the generated config since it measured faster.",
                    half_batch_size
                )
                .map_err(io_err)?;
            } else {
                writeln!(out, "Half batch size was not at least {:.0}% faster (measured {:+.1}%), keeping the default.",
                    100.0 * (EXTRA_TUNE_MIN_SPEEDUP_FACTOR - 1.0), 100.0 * (speedup - 1.0))
                    .map_err(io_err)?;
                nn_eval_ref.set_current_batch_size(nn_eval_ref.max_batch_size());
            }
            writeln!(out).map_err(io_err)?;
        }
    }

    let config_file_contents = state.contents();

    writeln!(out).map_err(io_err)?;
    banner(out, "DONE")?;
    writeln!(out).map_err(io_err)?;
    writeln!(out, "Writing new config file to {}", output_file).map_err(io_err)?;
    write_config_file(&config_file_contents)?;

    writeln!(out, "You should be now able to run KataGo with this config via something like:")
        .map_err(io_err)?;
    if model_file_is_default {
        writeln!(out, "katago-rs gtp -config '{}'", output_file).map_err(io_err)?;
    } else {
        writeln!(
            out,
            "katago-rs gtp -model '{}' -config '{}'",
            model_file, output_file
        )
        .map_err(io_err)?;
    }
    writeln!(out).map_err(io_err)?;

    writeln!(out, "Feel free to look at and edit the above config file further by hand in a txt editor.")
        .map_err(io_err)?;
    writeln!(out, "For more detailed notes about performance and what options in the config do, see:")
        .map_err(io_err)?;
    writeln!(out, "https://github.com/lightvector/KataGo/blob/master/cpp/configs/gtp_example.cfg")
        .map_err(io_err)?;
    writeln!(out).map_err(io_err)?;

    Ok(())
}

fn parse_args(args: &[String]) -> Result<GenConfigArgs, StringError> {
    GenConfigArgs::try_parse_from(std::iter::once(&"genconfig".to_string()).chain(args.iter()))
        .map_err(|e| StringError::new(format!("Argument error: {}", e)))
}

fn io_err(e: std::io::Error) -> StringError {
    StringError::new(e.to_string())
}

fn banner(out: &mut dyn Write, title: &str) -> Result<(), StringError> {
    writeln!(out).map_err(io_err)?;
    writeln!(out, "=========================================================================")
        .map_err(io_err)?;
    writeln!(out, "{}", title).map_err(io_err)?;
    Ok(())
}

/// Prompt on `out`, read one line from `input`, and loop until `parse`
/// accepts it. Fails if stdin is closed.
fn prompt_and_parse_input<T, F>(
    input: &mut dyn BufRead,
    out: &mut dyn Write,
    prompt: &str,
    parse: F,
) -> Result<T, StringError>
where
    F: Fn(&str) -> Result<T, StringError>,
{
    loop {
        write!(out, "{}", prompt).map_err(io_err)?;
        out.flush().map_err(io_err)?;
        let mut line = String::new();
        match input.read_line(&mut line) {
            Ok(0) => {
                return Err(StringError::new(
                    "Stdin was closed - failing and not generating a config".to_string(),
                ))
            }
            Ok(_) => {}
            Err(e) => return Err(io_err(e)),
        }
        match parse(global::trim(&line)) {
            Ok(value) => return Ok(value),
            Err(err) => {
                let what = global::trim(&err.message);
                if !what.is_empty() {
                    writeln!(out, "{}", what).map_err(io_err)?;
                }
            }
        }
    }
}

/// Prompt that accepts any single line (used for "press enter to continue").
fn prompt_and_continue(
    input: &mut dyn BufRead,
    out: &mut dyn Write,
    prompt: &str,
) -> Result<(), StringError> {
    prompt_and_parse_input(input, out, prompt, |_| Ok(()))
}

fn parse_yn(line: &str) -> Result<bool, StringError> {
    let s = global::to_lower(line);
    match s.as_str() {
        "yes" | "y" => Ok(true),
        "no" | "n" => Ok(false),
        _ => Err(StringError::new("Please answer y or n".to_string())),
    }
}

fn parse_optional_yn_default_no(line: &str) -> Result<bool, StringError> {
    if line.is_empty() {
        Ok(false)
    } else {
        parse_yn(line)
    }
}

fn parse_rules_line(line: &str) -> Result<Rules, StringError> {
    Rules::parse_rules(line).map_err(|e| StringError::new(e.to_string()))
}

fn parse_search_limit_kind(line: &str) -> Result<String, StringError> {
    let s = global::to_lower(line);
    match s.as_str() {
        "visits" | "visit" => Ok("visits".to_string()),
        "playouts" | "playout" => Ok("playouts".to_string()),
        "seconds" | "second" => Ok("seconds".to_string()),
        _ => Err(StringError::new(
            "Please specify one of \"visits\" or \"playouts\" or '\"seconds\"".to_string(),
        )),
    }
}

fn parse_visits_limit(line: &str) -> Result<i64, StringError> {
    if line.is_empty() {
        return Ok(500);
    }
    let v = global::string_to_int64(line)
        .map_err(|_| StringError::new("Please enter a valid integer".to_string()))?;
    if !(1..=1_000_000_000).contains(&v) {
        return Err(StringError::new("Must be between 1 and 1000000000".to_string()));
    }
    Ok(v)
}

fn parse_playouts_limit(line: &str) -> Result<i64, StringError> {
    if line.is_empty() {
        return Ok(300);
    }
    let v = global::string_to_int64(line)
        .map_err(|_| StringError::new("Please enter a valid integer".to_string()))?;
    if !(1..=1_000_000_000).contains(&v) {
        return Err(StringError::new("Must be between 1 and 1000000000".to_string()));
    }
    Ok(v)
}

fn parse_time_limit(line: &str) -> Result<f64, StringError> {
    if line.is_empty() {
        return Ok(10.0);
    }
    let v = global::string_to_double(line)
        .map_err(|_| StringError::new("Please enter a valid number".to_string()))?;
    if !v.is_finite() || v <= 0.0 || v >= 1.0e20 {
        return Err(StringError::new("Must positive and less than 1e20".to_string()));
    }
    Ok(v)
}

fn parse_ponder_time(line: &str) -> Result<f64, StringError> {
    if line.is_empty() {
        return Ok(1.0e20);
    }
    let v = global::string_to_double(line)
        .map_err(|_| StringError::new("Please enter a valid number".to_string()))?;
    if !v.is_finite() || v <= 0.0 || v >= 1.0e20 {
        return Err(StringError::new("Must positive and less than 1e20".to_string()));
    }
    Ok(v)
}

fn parse_device_idxs(line: &str) -> Result<Vec<i32>, StringError> {
    if line.is_empty() {
        return Ok(Vec::new());
    }
    let mut idxs = Vec::new();
    for piece in line.split(',') {
        let piece = global::trim(piece);
        let idx = global::string_to_int(piece)
            .map_err(|_| StringError::new(format!("Invalid device idx: {}", piece)))?;
        if !(0..=10000).contains(&idx) {
            return Err(StringError::new(format!("Invalid device idx: {}", idx)));
        }
        idxs.push(idx);
    }
    Ok(idxs)
}

/// Parse a RAM limit in GB (optionally suffixed "gb"), returning
/// `(nnCacheSizePowerOfTwo, nnMutexPoolSizePowerOfTwo)`.
///
/// Mirrors the sizing loop in upstream `MainCmds::genconfig`: each cache entry
/// costs about 3000 bytes, and the cache stores 2 entries per slot.
fn parse_ram_limit(line: &str) -> Result<(i32, i32), StringError> {
    let mut s = global::to_lower(line);
    if s.ends_with("gb") {
        s = s[..s.len() - 2].to_string();
    }
    let s = global::trim(&s);
    let approx_gb_limit = if s.is_empty() {
        3.0
    } else {
        let v = global::string_to_double(s)
            .map_err(|_| StringError::new("Please enter a valid number".to_string()))?;
        if !v.is_finite() || v <= 0.0 || v >= 1_000_000.0 {
            return Err(StringError::new("Must positive and less than 1000000".to_string()));
        }
        v
    };
    Ok(compute_nn_cache_sizes(approx_gb_limit))
}

pub(crate) fn compute_nn_cache_sizes(approx_gb_limit: f64) -> (i32, i32) {
    let approx_gb_limit = approx_gb_limit * 1.00001;
    let mut cache_pow: i32 = 10; // Never set below this size
    while cache_pow < 48 {
        let mem_usage = (2.0f64).powi(cache_pow) * 3000.0;
        if mem_usage * 2.0 > approx_gb_limit * 1_073_741_824.0 {
            break;
        }
        cache_pow += 1;
    }
    let mutex_pow = (cache_pow - 4).clamp(10, 24);
    (cache_pow, mutex_pow)
}

fn parse_tuning_visits(line: &str) -> Result<i64, StringError> {
    if line.is_empty() {
        return Ok(-1);
    }
    let v = global::string_to_int64(line)
        .map_err(|_| StringError::new("Please enter a valid integer".to_string()))?;
    if !(1..=1_000_000_000).contains(&v) {
        return Err(StringError::new("Must be between 1 and 1000000000".to_string()));
    }
    Ok(v)
}

fn parse_tuning_seconds(line: &str) -> Result<f64, StringError> {
    if line.is_empty() {
        return Ok(DEFAULT_SECONDS_PER_GAME_MOVE);
    }
    let v = global::string_to_double(line)
        .map_err(|_| StringError::new("Please enter a valid number".to_string()))?;
    if !v.is_finite() || v <= 0.0 || v > 1_000_000.0 {
        return Err(StringError::new("Must be between 0 and 1000000".to_string()));
    }
    Ok(v)
}

/// Backend names selectable in this build, in prompt order.
fn available_backends() -> Vec<&'static str> {
    let mut backends = Vec::new();
    if cfg!(feature = "cuda") {
        backends.push("cudabackend");
    }
    backends.push("trtbackend");
    backends.push("dummybackend");
    backends
}

fn parse_backend_choice(
    line: &str,
    available: &[&str],
    default_backend: &str,
) -> Result<String, StringError> {
    if line.is_empty() {
        return Ok(default_backend.to_string());
    }
    let s = global::to_lower(line);
    let normalized = match s.as_str() {
        "cuda" | "cudabackend" => "cudabackend",
        "trt" | "tensorrt" | "trtbackend" => "trtbackend",
        "dummy" | "dummybackend" => "dummybackend",
        _ => {
            return Err(StringError::new(format!(
                "Please answer one of: {}",
                available.join(", ")
            )))
        }
    };
    if !available.contains(&normalized) {
        return Err(StringError::new(format!(
            "{} is not available in this build (available: {})",
            normalized,
            available.join(", ")
        )));
    }
    Ok(normalized.to_string())
}

/// Print the device list of the chosen backend, like upstream
/// `NeuralNet::printDevices()`.
fn print_backend_devices(backend: &str) {
    match backend {
        "cudabackend" => {
            #[cfg(feature = "cuda")]
            kata_nn::backends::cuda::CudaBackend.print_devices();
        }
        "trtbackend" => {
            kata_nn::backends::trt::TensorRtBackend.print_devices();
        }
        _ => {}
    }
}

/// Insert the `nnBackend` key into the generated config's GPU section.
///
/// Upstream configs have no `nnBackend` (the backend is compiled in); this
/// port selects it at runtime, so the generated config records the choice the
/// tuning session actually measured.
fn insert_backend_line(contents: String, backend: &str) -> String {
    let anchor = "# This section configures GPU settings.\n";
    let idx = contents
        .find(anchor)
        .expect("GTP config template GPU settings anchor not found");
    let mut out = String::with_capacity(contents.len() + 64);
    out.push_str(&contents[..idx + anchor.len()]);
    out.push_str(&format!(
        "\n# Neural net backend used by this Rust port of KataGo.\nnnBackend = {}\n",
        backend
    ));
    out.push_str(&contents[idx + anchor.len()..]);
    out
}

/// Read `numSearchThreads` from an existing config file, failing like upstream
/// when the file cannot be parsed or the key is out of range.
fn read_num_search_threads(path: &str) -> Result<i32, StringError> {
    let contents = std_fs::read_to_string(path)
        .map_err(|e| StringError::new(format!("Could not read {}: {}", path, e)))?;
    let cfg = ConfigParser::from_str(&contents, false, false)
        .map_err(|e| StringError::new(format!("Could not parse {}: {}", path, e)))?;
    cfg.get_int("numSearchThreads", 1, 4096)
        .map_err(|e| StringError::new(e.to_string()))
}

fn visits_per_second(result: &BenchmarkResults) -> f64 {
    result.total_visits as f64 / (result.total_seconds + 0.00001)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_yn() {
        assert!(parse_yn("y").unwrap());
        assert!(parse_yn("YES").unwrap());
        assert!(!parse_yn("n").unwrap());
        assert!(!parse_yn("No").unwrap());
        assert!(parse_yn("maybe").is_err());
    }

    #[test]
    fn test_parse_optional_yn_default_no() {
        assert!(!parse_optional_yn_default_no("").unwrap());
        assert!(parse_optional_yn_default_no("y").unwrap());
        assert!(parse_optional_yn_default_no("what").is_err());
    }

    #[test]
    fn test_parse_rules_line() {
        assert!(parse_rules_line("chinese").is_ok());
        assert!(parse_rules_line("tromp-taylor").is_ok());
        assert!(parse_rules_line("notaruleset").is_err());
    }

    #[test]
    fn test_parse_search_limit_kind() {
        assert_eq!(parse_search_limit_kind("visits").unwrap(), "visits");
        assert_eq!(parse_search_limit_kind("playout").unwrap(), "playouts");
        assert_eq!(parse_search_limit_kind("Seconds").unwrap(), "seconds");
        assert!(parse_search_limit_kind("hours").is_err());
    }

    #[test]
    fn test_parse_limits() {
        assert_eq!(parse_visits_limit("").unwrap(), 500);
        assert_eq!(parse_visits_limit("1234").unwrap(), 1234);
        assert!(parse_visits_limit("0").is_err());
        assert_eq!(parse_playouts_limit("").unwrap(), 300);
        assert_eq!(parse_time_limit("").unwrap(), 10.0);
        assert!(parse_time_limit("-1").is_err());
        assert_eq!(parse_ponder_time("").unwrap(), 1.0e20);
        assert_eq!(parse_ponder_time("60").unwrap(), 60.0);
    }

    #[test]
    fn test_parse_device_idxs() {
        assert!(parse_device_idxs("").unwrap().is_empty());
        assert_eq!(parse_device_idxs("0").unwrap(), vec![0]);
        assert_eq!(parse_device_idxs("0,1,2").unwrap(), vec![0, 1, 2]);
        assert_eq!(parse_device_idxs(" 0 , 1 ").unwrap(), vec![0, 1]);
        assert!(parse_device_idxs("x").is_err());
        assert!(parse_device_idxs("-1").is_err());
        assert!(parse_device_idxs("20000").is_err());
    }

    #[test]
    fn test_compute_nn_cache_sizes() {
        // Default 3 GB: 2 * 2^20 * 3000 B ~= 6.3 GB does not fit, 2^19 does.
        assert_eq!(compute_nn_cache_sizes(3.0), (20, 16));
        // Tiny limit floors at 2^10.
        assert_eq!(compute_nn_cache_sizes(0.001), (10, 10));
        // Huge limit: 999999 GB allows 2^38 slots; the mutex pool caps at 2^24.
        assert_eq!(compute_nn_cache_sizes(999999.0), (38, 24));
        // 48 GB: the first size whose doubled cost (2 * 2^p * 3000 B) exceeds
        // 5.15e10 bytes is p = 24 (cost 1.0e11); p = 23 costs 5.03e10 and fits.
        assert_eq!(compute_nn_cache_sizes(48.0), (24, 20));
    }

    #[test]
    fn test_parse_ram_limit() {
        assert_eq!(parse_ram_limit("").unwrap(), (20, 16));
        assert_eq!(parse_ram_limit("3gb").unwrap(), (20, 16));
        assert_eq!(parse_ram_limit("3 GB").unwrap(), (20, 16));
        assert!(parse_ram_limit("abc").is_err());
        assert!(parse_ram_limit("-1").is_err());
    }

    #[test]
    fn test_parse_tuning_inputs() {
        assert_eq!(parse_tuning_visits("").unwrap(), -1);
        assert_eq!(parse_tuning_visits("800").unwrap(), 800);
        assert!(parse_tuning_visits("0").is_err());
        assert_eq!(parse_tuning_seconds("").unwrap(), 5.0);
        assert_eq!(parse_tuning_seconds("12.5").unwrap(), 12.5);
        assert!(parse_tuning_seconds("0").is_err());
    }

    #[test]
    fn test_parse_backend_choice() {
        let available = ["trtbackend", "dummybackend"];
        assert_eq!(
            parse_backend_choice("", &available, "trtbackend").unwrap(),
            "trtbackend"
        );
        assert_eq!(
            parse_backend_choice("cuda", &available, "trtbackend").unwrap_err().message,
            "cudabackend is not available in this build (available: trtbackend, dummybackend)"
        );
        assert_eq!(
            parse_backend_choice("dummy", &available, "trtbackend").unwrap(),
            "dummybackend"
        );
        assert!(parse_backend_choice("eigen", &available, "trtbackend").is_err());
    }

    #[test]
    fn test_insert_backend_line() {
        let contents = "before\n# This section configures GPU settings.\nafter\n";
        let out = insert_backend_line(contents.to_string(), "cudabackend");
        assert!(out.contains("# This section configures GPU settings.\n\n# Neural net backend used by this Rust port of KataGo.\nnnBackend = cudabackend\n"));
        assert!(out.starts_with("before\n"));
        assert!(out.ends_with("after\n"));
    }

    #[test]
    fn test_read_num_search_threads() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("old.cfg");
        std_fs::write(&path, "numSearchThreads = 8\n").unwrap();
        assert_eq!(read_num_search_threads(path.to_str().unwrap()).unwrap(), 8);

        std_fs::write(&path, "nothing relevant\n").unwrap();
        assert!(read_num_search_threads(path.to_str().unwrap()).is_err());
    }

    // End-to-end over the skip-tuning path: all prompts answered from a
    // scripted input, existing config has numSearchThreads, no NN evaluator
    // is created.
    #[test]
    fn test_genconfig_skip_tuning_writes_config() {
        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("gtp.cfg");
        std_fs::write(&out_path, "numSearchThreads = 8\n").unwrap();

        let input = "\
y
chinese
n

n
dummy

3gb
y
";
        let mut reader = input.as_bytes();
        let mut writer = Vec::new();
        genconfig_impl(
            &[
                "--model".to_string(),
                "some_model.bin.gz".to_string(),
                "--output".to_string(),
                out_path.to_str().unwrap().to_string(),
            ],
            &mut reader,
            &mut writer,
        )
        .unwrap();

        let contents = std_fs::read_to_string(&out_path).unwrap();
        assert!(contents.contains("koRule = SIMPLE"));
        assert!(contents.contains("scoringRule = AREA"));
        assert!(contents.contains("numSearchThreads = 8"));
        assert!(contents.contains("nnBackend = dummybackend"));
        assert!(contents.contains("nnCacheSizePowerOfTwo = 20"));
        assert!(contents.contains("nnMutexPoolSizePowerOfTwo = 16"));
        assert!(contents.contains("# maxVisits = 500"));
        assert!(contents.contains("ponderingEnabled = false"));
        assert!(!contents.contains("$$"));

        let output = String::from_utf8(writer).unwrap();
        assert!(output.contains("Skip performance tuning (y/n)?"));
        assert!(output.contains("Writing new config file to"));
    }

    #[test]
    fn test_genconfig_refuses_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("gtp.cfg");
        std_fs::write(&out_path, "numSearchThreads = 8\n").unwrap();

        let input = "n\n";
        let mut reader = input.as_bytes();
        let mut writer = Vec::new();
        genconfig_impl(
            &[
                "--model".to_string(),
                "some_model.bin.gz".to_string(),
                "--output".to_string(),
                out_path.to_str().unwrap().to_string(),
            ],
            &mut reader,
            &mut writer,
        )
        .unwrap();

        // File untouched.
        assert_eq!(
            std_fs::read_to_string(&out_path).unwrap(),
            "numSearchThreads = 8\n"
        );
        let output = String::from_utf8(writer).unwrap();
        assert!(output.contains("Please provide an alternate file path"));
    }

    #[test]
    fn test_genconfig_stdin_closed() {
        let mut reader: &[u8] = b"";
        let mut writer = Vec::new();
        let err = genconfig_impl(
            &[
                "--model".to_string(),
                "some_model.bin.gz".to_string(),
                "--output".to_string(),
                "unused.cfg".to_string(),
            ],
            &mut reader,
            &mut writer,
        )
        .unwrap_err();
        assert!(err.message.contains("Stdin was closed"));
    }

    #[test]
    fn test_genconfig_reprompts_on_bad_rules() {
        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("gtp.cfg");

        // Bad rules answer is re-prompted, then "japanese" is accepted. The
        // remaining answers walk through to the first tuning prompt, where
        // stdin runs out and the command fails cleanly without writing the
        // config.
        let input = "notarules\njapanese\nn\n\nn\ndummy\n\n3\n";
        let mut reader = input.as_bytes();
        let mut writer = Vec::new();
        let err = genconfig_impl(
            &[
                "--model".to_string(),
                "some_model.bin.gz".to_string(),
                "--output".to_string(),
                out_path.to_str().unwrap().to_string(),
            ],
            &mut reader,
            &mut writer,
        )
        .unwrap_err();
        assert!(err.message.contains("Stdin was closed"));

        let output = String::from_utf8(writer).unwrap();
        assert_eq!(
            output.matches("What rules should KataGo use").count(),
            2,
            "bad rules answer should be re-prompted"
        );
        assert!(!out_path.exists());
    }
}
