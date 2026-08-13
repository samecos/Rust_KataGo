//! JSON-based analysis engine.
//!
//! Corresponds to `MainCmds::analysis` in `cpp/command/analysis.cpp`.

#![allow(
    dead_code,
    clippy::collapsible_if,
    clippy::mixed_case_hex_literals,
    clippy::obfuscated_if_else,
    clippy::needless_borrow,
    clippy::needless_range_loop,
    clippy::redundant_locals,
    clippy::redundant_pattern_matching,
    clippy::clone_on_copy,
    clippy::too_many_arguments
)]

use std::collections::{HashMap, HashSet};
use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};

use clap::Parser;
use serde_json::{Map, Value};

use kata_core::config::ConfigParser;
use kata_core::global::StringError;
use kata_core::logger::{Logger, LoggerOptions};
use kata_core::rng::Rand;
use kata_core::thread::queue::{ThreadSafePriorityQueue, ThreadSafeQueue};
use kata_game::board::{
    Board, C_EMPTY, Loc, MAX_ARR_SIZE, MAX_LEN, Move, NULL_LOC, P_BLACK, P_WHITE, PASS_LOC, Player,
    location, player_io,
};
use kata_game::history::BoardHistory;
use kata_game::rules::{Rules, WhiteHandicapBonusRule};
use kata_nn::eval::NnEvaluator;
use kata_program::setup::{self, SetupFor};
use kata_search::async_bot::AsyncBot;
use kata_search::eval_cache::EvalCacheTable;
use kata_search::params::SearchParams;
use kata_search::pattern_bonus::PatternBonusTable;
use kata_search::search::Search;
use kata_search::time_control::TimeControls;

use crate::cli::CommonArgs;

const STATUS_IN_QUEUE: i32 = -1;
const STATUS_POPPED: i32 = -2;
const STATUS_TERMINATED: i32 = -3;

/// Wrapper around a bot raw pointer so it can be sent between threads.
#[derive(Clone, Copy)]
struct BotPtr(usize);

impl BotPtr {
    fn new(bot: *mut AsyncBot<'static>) -> Self {
        Self(bot as usize)
    }

    fn as_ptr(self) -> *mut AsyncBot<'static> {
        self.0 as *mut AsyncBot<'static>
    }
}

unsafe impl Send for BotPtr {}
unsafe impl Sync for BotPtr {}

/// A single position that the analysis engine has been asked to evaluate.
struct AnalyzeRequest {
    internal_id: i64,
    id: String,
    turn_number: i32,
    priority: i64,

    board: Board,
    hist: BoardHistory,
    next_pla: Player,

    params: SearchParams,
    perspective: Player,
    analysis_pv_len: i32,
    include_ownership: bool,
    include_ownership_stdev: bool,
    include_moves_ownership: bool,
    include_moves_ownership_stdev: bool,
    include_policy: bool,
    include_pv_visits: bool,
    include_no_result_value: bool,

    report_during_search: bool,
    report_during_search_every: f64,
    first_report_during_search_after: f64,

    avoid_move_until_by_loc_black: Vec<i32>,
    avoid_move_until_by_loc_white: Vec<i32>,

    /// Atomic request lifecycle status.
    status: AtomicI32,
}

impl AnalyzeRequest {
    fn new() -> Self {
        Self {
            internal_id: 0,
            id: String::new(),
            turn_number: 0,
            priority: 0,
            board: Board::new(19, 19),
            hist: BoardHistory::new(Board::new(19, 19), C_EMPTY, Rules::default(), 0),
            next_pla: C_EMPTY,
            params: SearchParams::new(),
            perspective: C_EMPTY,
            analysis_pv_len: 15,
            include_ownership: false,
            include_ownership_stdev: false,
            include_moves_ownership: false,
            include_moves_ownership_stdev: false,
            include_policy: false,
            include_pv_visits: false,
            include_no_result_value: false,
            report_during_search: false,
            report_during_search_every: 1e30,
            first_report_during_search_after: 1e30,
            avoid_move_until_by_loc_black: Vec::new(),
            avoid_move_until_by_loc_white: Vec::new(),
            status: AtomicI32::new(STATUS_IN_QUEUE),
        }
    }
}

/// CLI arguments for the `analysis` subcommand.
#[derive(Parser, Debug, Clone)]
struct AnalysisArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// Analyze up to this many positions in parallel.
    #[arg(long = "analysis-threads", value_name = "THREADS")]
    analysis_threads: Option<i32>,

    /// When stdin is closed, quit quickly without waiting for queued tasks.
    #[arg(long = "quit-without-waiting")]
    quit_without_waiting: bool,
}

/// Public CLI entry point.
pub fn analysis(args: &[String]) -> i32 {
    match analysis_impl(args, io::stdin().lock()) {
        Ok(lines) => {
            for line in lines {
                println!("{}", line);
            }
            0
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            1
        }
    }
}

/// Internal testable entry point.
///
/// Runs the analysis engine on the provided input and returns all JSON lines
/// that would have been written to stdout.
fn analysis_impl<R: BufRead>(args: &[String], input: R) -> Result<Vec<String>, StringError> {
    let parsed =
        AnalysisArgs::try_parse_from(std::iter::once(&"analysis".to_string()).chain(args.iter()))
            .map_err(|e| StringError::new(format!("Argument error: {}", e)))?;

    let mut cfg = if parsed.common.config.is_empty() {
        let mut cfg = minimal_analysis_config();
        parsed.common.maybe_apply_override_config_arg(&mut cfg)?;
        cfg
    } else {
        parsed.common.get_config("analysis_example.cfg")?
    };

    let _ = cfg.apply_alias("numSearchThreadsPerAnalysisThread", "numSearchThreads");

    let num_analysis_threads_cmdline = parsed.analysis_threads.unwrap_or(0);
    let num_analysis_threads_cmdline_specified = parsed.analysis_threads.is_some();

    if cfg.contains("numAnalysisThreads") && num_analysis_threads_cmdline_specified {
        return Err(StringError::new(format!(
            "When specifying numAnalysisThreads in the config ({}), it is redundant and disallowed to also specify it via -analysis-threads",
            cfg.file_name()
        )));
    }

    let num_analysis_threads = if num_analysis_threads_cmdline_specified {
        num_analysis_threads_cmdline
    } else if cfg.contains("numAnalysisThreads") {
        cfg.get_int("numAnalysisThreads", 1, 16384)
            .map_err(|e| StringError::new(format!("Invalid value for numAnalysisThreads: {}", e)))?
    } else {
        1
    };
    if num_analysis_threads <= 0 || num_analysis_threads > 16384 {
        return Err(StringError::new(format!(
            "Invalid value for numAnalysisThreads: {}",
            num_analysis_threads
        )));
    }

    let for_deterministic_testing = cfg
        .contains("forDeterministicTesting")
        .then(|| cfg.get_bool("forDeterministicTesting").unwrap_or(false))
        .unwrap_or(false);

    let mut seed_rand = Rand::new();
    if for_deterministic_testing {
        seed_rand.init_from_seed("forDeterministicTesting");
    }

    let logger: &'static Logger = Box::leak(Box::new(Logger::new(
        LoggerOptions {
            log_to_stdout: false,
            log_to_stderr: true,
            log_time: false,
        },
        None,
    )));
    let log_to_stderr = logger.is_logging_to_stderr();

    let log_all_requests = cfg
        .contains("logAllRequests")
        .then(|| cfg.get_bool("logAllRequests").unwrap_or(false))
        .unwrap_or(false);
    let log_all_responses = cfg
        .contains("logAllResponses")
        .then(|| cfg.get_bool("logAllResponses").unwrap_or(false))
        .unwrap_or(false);
    let log_errors_and_warnings = cfg
        .contains("logErrorsAndWarnings")
        .then(|| cfg.get_bool("logErrorsAndWarnings").unwrap_or(false))
        .unwrap_or(true);
    let log_search_info = cfg
        .contains("logSearchInfo")
        .then(|| cfg.get_bool("logSearchInfo").unwrap_or(false))
        .unwrap_or(false);

    let warn_unused_fields = cfg
        .contains("warnUnusedFields")
        .then(|| cfg.get_bool("warnUnusedFields").unwrap_or(false))
        .unwrap_or(true);

    let human_model_file = parsed.common.get_human_model_file();

    let load_params = |config: &ConfigParser,
                       params: &mut SearchParams,
                       perspective: &mut Player,
                       default_perspective: Player| {
        let has_human_model = human_model_file.is_some();
        *params = setup::load_single_params_with_human(config, SetupFor::Analysis, has_human_model)
            .map_err(|e| StringError::new(format!("Could not load params: {}", e)))?;
        *perspective = setup::parse_report_analysis_winrates(config, default_perspective)
            .map_err(|e| StringError::new(format!("Could not parse perspective: {}", e)))?;
        if !config.contains("conservativePass") {
            params.conservative_pass = true;
        }
        Ok::<(), StringError>(())
    };

    let mut default_params = SearchParams::new();
    let mut default_perspective = C_EMPTY;
    load_params(&cfg, &mut default_params, &mut default_perspective, C_EMPTY)?;

    let pattern_bonus_tables = setup::load_avoid_sgf_pattern_bonus_tables(&cfg, &logger)
        .map_err(|e| StringError::new(format!("Could not load pattern bonus tables: {}", e)))?;
    let pattern_bonus_table: Option<Box<PatternBonusTable>> = if pattern_bonus_tables.is_empty() {
        None
    } else {
        pattern_bonus_tables
            .into_iter()
            .next()
            .flatten()
            .map(Box::new)
    };

    let analysis_pv_len = cfg
        .contains("analysisPVLen")
        .then(|| {
            cfg.get_int("analysisPVLen", 1, 100)
                .map_err(|e| StringError::new(format!("Invalid analysisPVLen: {}", e)))
        })
        .unwrap_or(Ok(15))?;
    let assume_multiple_starting_black_moves_are_handicap = cfg
        .contains("assumeMultipleStartingBlackMovesAreHandicap")
        .then(|| {
            cfg.get_bool("assumeMultipleStartingBlackMovesAreHandicap")
                .unwrap_or(true)
        })
        .unwrap_or(true);
    let prevent_encore = cfg
        .contains("preventCleanupPhase")
        .then(|| cfg.get_bool("preventCleanupPhase").unwrap_or(true))
        .unwrap_or(true);

    setup::initialize_session(&cfg);
    let expected_concurrent_evals = num_analysis_threads * default_params.num_threads;
    let default_max_batch_size = -1;

    let model_file = if parsed.common.config.is_empty() {
        "/dev/null".to_string()
    } else {
        parsed.common.get_model_file()?
    };

    let nn_eval = setup::initialize_nn_evaluator(
        model_file.clone(),
        model_file.clone(),
        String::new(),
        &cfg,
        &logger,
        &mut seed_rand,
        expected_concurrent_evals,
        kata_nn::inputs::nn_pos::MAX_BOARD_LEN as i32,
        kata_nn::inputs::nn_pos::MAX_BOARD_LEN as i32,
        default_max_batch_size,
        false,
        false,
        SetupFor::Analysis,
    )
    .map_err(|e| StringError::new(format!("Could not initialize neural net: {}", e)))?;

    let mut human_eval: Option<NnEvaluator> = None;
    if let Some(human_model_file) = &human_model_file {
        human_eval = Some(
            setup::initialize_nn_evaluator(
                human_model_file.clone(),
                human_model_file.clone(),
                String::new(),
                &cfg,
                &logger,
                &mut seed_rand,
                expected_concurrent_evals,
                kata_nn::inputs::nn_pos::MAX_BOARD_LEN as i32,
                kata_nn::inputs::nn_pos::MAX_BOARD_LEN as i32,
                default_max_batch_size,
                false,
                false,
                SetupFor::Analysis,
            )
            .map_err(|e| StringError::new(format!("Could not initialize human model: {}", e)))?,
        );
        if let Some(ref he) = human_eval {
            if !he.requires_sgf_metadata() {
                let warning = "WARNING: Human model was not trained from SGF metadata to vary by rank! Did you pass the wrong model for -human-model?";
                logger.write(warning);
                if !log_to_stderr {
                    eprintln!("{}", warning);
                }
            }
        }
    }

    let _ = cfg.warn_unused_keys(&mut std::io::stderr(), Some(logger));
    {
        let mut out = Vec::new();
        setup::maybe_warn_human_sl_params(
            &default_params,
            Some(&nn_eval),
            human_eval.as_ref(),
            &mut out,
            Some(&logger),
        )
        .map_err(|e| StringError::new(format!("Human SL param warning failed: {}", e)))?;
        if !out.is_empty() && !log_to_stderr {
            std::io::stderr().write_all(&out).unwrap();
        }
    }

    logger.write("Analysis Engine starting...");
    logger.write(&get_version_string());
    if !log_to_stderr {
        eprintln!("{}", get_version_string());
    }

    logger.write(&format!("Loaded config {}", cfg.file_name()));
    logger.write(&format!("Loaded model {}", model_file));
    parsed.common.log_overrides(&logger);

    if human_eval.is_some()
        && !cfg.contains("humanSLProfile")
        && human_eval.as_ref().unwrap().requires_sgf_metadata()
    {
        let warning = "Warning: Provided -human-model but humanSLProfile is not yet set. The human SL model will only be used on queries that provide humanSLProfile in overrideSettings.";
        logger.write(warning);
        if !logger.is_logging_to_stderr() {
            eprintln!("{}", warning);
        }
    }

    // Leak the evaluators so that worker threads can hold &'static references
    // to them. This matches the C++ design where the evaluators outlive all
    // worker threads and are deleted only at process exit.
    let nn_eval: &'static NnEvaluator = Box::leak(Box::new(nn_eval));
    let human_eval: Option<&'static NnEvaluator> = human_eval.map(|e| {
        let leaked: &'static NnEvaluator = Box::leak(Box::new(e));
        leaked
    });

    let eval_cache: Option<Arc<EvalCacheTable>> = if default_params.use_eval_cache {
        Some(Arc::new(EvalCacheTable::new(
            default_params.subtree_value_bias_table_num_shards as u32,
        )))
    } else {
        None
    };

    let expected_keys: HashSet<String> = EXPECTED_KEYS.iter().map(|s| s.to_string()).collect();

    let to_write_queue = Arc::new(ThreadSafeQueue::<String>::new());
    let to_analyze_queue =
        Arc::new(ThreadSafePriorityQueue::<(i64, i64), Arc<AnalyzeRequest>>::new());
    let open_requests = Arc::new(Mutex::new(HashMap::<i64, Arc<AnalyzeRequest>>::new()));

    let output_lines: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    // Build one bot per analysis thread in the main thread. Each bot is passed
    // to its worker thread as a raw pointer so that we do not need `AsyncBot`
    // to implement `Send`; the C++ design assumes one bot per analysis thread.
    let mut bots_vec: Vec<BotPtr> = Vec::with_capacity(num_analysis_threads as usize);
    for _ in 0..num_analysis_threads {
        let search_rand_seed = format!(
            "{}{}",
            kata_core::global::uint64_to_hex_string(seed_rand.next_u64()),
            kata_core::global::uint64_to_hex_string(seed_rand.next_u64())
        );
        let mut bot = AsyncBot::new_with_human(
            default_params.clone(),
            nn_eval,
            human_eval,
            logger,
            &search_rand_seed,
        );
        bot.set_copy_of_external_pattern_bonus_table(&pattern_bonus_table);
        bot.set_external_eval_cache(eval_cache.clone());
        bots_vec.push(BotPtr::new(Box::into_raw(Box::new(bot))));
    }
    let bots = Arc::new(bots_vec);

    logger.write(&format!(
        "Analyzing up to {} positions at a time in parallel",
        num_analysis_threads
    ));
    logger.write("Started, ready to begin handling requests");
    if !log_to_stderr {
        eprintln!("Started, ready to begin handling requests");
    }

    let mut handles: Vec<std::thread::JoinHandle<()>> = Vec::new();

    // Write thread.
    let write_queue = Arc::clone(&to_write_queue);
    let write_lines = Arc::clone(&output_lines);
    handles.push(std::thread::spawn(move || {
        let mut buf = String::new();
        while write_queue.wait_pop(&mut buf) {
            write_lines.lock().unwrap().push(buf.clone());
        }
    }));

    // Analysis threads.
    for thread_idx in 0..num_analysis_threads {
        let analyze_queue = Arc::clone(&to_analyze_queue);
        let write_queue = Arc::clone(&to_write_queue);
        let open_reqs = Arc::clone(&open_requests);
        let bots = Arc::clone(&bots);
        let logger = logger;
        let bot_ptr = bots[thread_idx as usize].clone();
        let handle = std::thread::spawn(move || {
            analysis_loop(
                thread_idx,
                bot_ptr,
                analyze_queue,
                write_queue,
                open_reqs,
                logger,
                log_search_info,
                prevent_encore,
            );
        });
        handles.push(handle);
    }

    let result = run_request_loop(
        input,
        &to_write_queue,
        &to_analyze_queue,
        &open_requests,
        &bots,
        &logger,
        log_all_requests,
        log_all_responses,
        log_errors_and_warnings,
        warn_unused_fields,
        &expected_keys,
        &default_params,
        default_perspective,
        analysis_pv_len,
        assume_multiple_starting_black_moves_are_handicap,
        prevent_encore,
        nn_eval,
        human_eval,
    );

    if parsed.quit_without_waiting {
        to_write_queue.set_read_only();
        to_analyze_queue.set_read_only();
        for i in 0..bots.len() {
            unsafe { (*bots[i].as_ptr()).stop_without_wait() };
        }
        for i in 0..bots.len() {
            unsafe { (*bots[i].as_ptr()).set_killed() };
        }
    } else {
        to_analyze_queue.set_read_only();
        // Wait for analysis threads to drain before stopping the write thread.
        let analysis_handles: Vec<_> = handles.drain(1..).collect();
        for handle in analysis_handles {
            let _ = handle.join();
        }
        to_write_queue.set_read_only();
    }

    // Join the write thread (and analysis threads in the quit-without-waiting case).
    for handle in handles {
        let _ = handle.join();
    }

    logger.write(nn_eval.model_file_name());
    logger.write(&format!("NN rows: {}", nn_eval.num_rows_processed()));
    logger.write(&format!("NN batches: {}", nn_eval.num_batches_processed()));
    logger.write(&format!(
        "NN avg batch size: {}",
        nn_eval.average_processed_batch_size()
    ));
    if let Some(he) = human_eval {
        logger.write(he.model_file_name());
        logger.write(&format!("NN rows: {}", he.num_rows_processed()));
        logger.write(&format!("NN batches: {}", he.num_batches_processed()));
        logger.write(&format!(
            "NN avg batch size: {}",
            he.average_processed_batch_size()
        ));
    }
    logger.write("All cleaned up, quitting");

    result?;

    let lines = Arc::try_unwrap(output_lines)
        .map_err(|_| StringError::new("output_lines still referenced".to_string()))?
        .into_inner()
        .map_err(|_| StringError::new("output_lines mutex poisoned".to_string()))?;
    Ok(lines)
}

fn get_version_string() -> String {
    // Placeholder matching the C++ help string format.
    "katago-rs analysis engine".to_string()
}

fn minimal_analysis_config() -> ConfigParser {
    ConfigParser::from_str(
        "numSearchThreads = 1
         maxVisits = 10
         nnMaxBatchSize = 8
         valueWeightExponent = 0
        ",
        false,
        false,
    )
    .expect("hard-coded minimal config is valid")
}

const EXPECTED_KEYS: &[&str] = &[
    "id",
    "action",
    "terminateId",
    "turnNumbers",
    "boardXSize",
    "boardYSize",
    "initialStones",
    "moves",
    "initialPlayer",
    "analyzeTurns",
    "priorities",
    "rules",
    "komi",
    "whiteHandicapBonus",
    "overrideSettings",
    "maxVisits",
    "analysisPVLen",
    "rootFpuReductionMax",
    "rootPolicyTemperature",
    "includeMovesOwnership",
    "includeMovesOwnershipStdev",
    "includeOwnership",
    "includeOwnershipStdev",
    "includePolicy",
    "includePVVisits",
    "includeNoResultValue",
    "reportDuringSearchEvery",
    "firstReportDuringSearchAfter",
    "priority",
    "allowMoves",
    "avoidMoves",
];

fn push_to_write(queue: &ThreadSafeQueue<String>, message: String) {
    let _ = queue.force_push(message);
}

fn report_error(
    queue: &ThreadSafeQueue<String>,
    logger: &Logger,
    log_errors_and_warnings: bool,
    s: &str,
) {
    let mut ret = Map::new();
    ret.insert("error".to_string(), Value::String(s.to_string()));
    let msg = serde_json::to_string(&ret).unwrap();
    push_to_write(queue, msg.clone());
    if log_errors_and_warnings {
        logger.write(&format!("Error: {}", msg));
    }
}

fn report_error_for_id(
    queue: &ThreadSafeQueue<String>,
    logger: &Logger,
    log_errors_and_warnings: bool,
    id: &str,
    field: &str,
    s: &str,
) {
    let mut ret = Map::new();
    ret.insert("id".to_string(), Value::String(id.to_string()));
    ret.insert("field".to_string(), Value::String(field.to_string()));
    ret.insert("error".to_string(), Value::String(s.to_string()));
    let msg = serde_json::to_string(&ret).unwrap();
    push_to_write(queue, msg.clone());
    if log_errors_and_warnings {
        logger.write(&format!("Error: {}", msg));
    }
}

fn report_warning_for_id(
    queue: &ThreadSafeQueue<String>,
    logger: &Logger,
    log_errors_and_warnings: bool,
    id: &str,
    field: &str,
    s: &str,
) {
    let mut ret = Map::new();
    ret.insert("id".to_string(), Value::String(id.to_string()));
    ret.insert("field".to_string(), Value::String(field.to_string()));
    ret.insert("warning".to_string(), Value::String(s.to_string()));
    let msg = serde_json::to_string(&ret).unwrap();
    push_to_write(queue, msg.clone());
    if log_errors_and_warnings {
        logger.write(&format!("Warning: {}", msg));
    }
}

fn report_no_analysis(queue: &ThreadSafeQueue<String>, request: &AnalyzeRequest) {
    let mut ret = Map::new();
    ret.insert("id".to_string(), Value::String(request.id.clone()));
    ret.insert(
        "turnNumber".to_string(),
        Value::Number(request.turn_number.into()),
    );
    ret.insert("isDuringSearch".to_string(), Value::Bool(false));
    ret.insert("noResults".to_string(), Value::Bool(true));
    push_to_write(queue, serde_json::to_string(&ret).unwrap());
}

fn report_analysis(
    queue: &ThreadSafeQueue<String>,
    request: &AnalyzeRequest,
    search: &Search,
    is_during_search: bool,
    prevent_encore: bool,
) -> bool {
    let mut value = Value::Object(Map::new());

    let success = search.get_analysis_json(
        request.perspective,
        request.analysis_pv_len,
        prevent_encore,
        request.include_policy,
        request.include_ownership,
        request.include_ownership_stdev,
        request.include_moves_ownership,
        request.include_moves_ownership_stdev,
        request.include_pv_visits,
        request.include_no_result_value,
        &mut value,
    );

    if success {
        if let Value::Object(ref mut m) = value {
            m.insert("id".to_string(), Value::String(request.id.clone()));
            m.insert(
                "turnNumber".to_string(),
                Value::Number(request.turn_number.into()),
            );
            m.insert("isDuringSearch".to_string(), Value::Bool(is_during_search));
        }
        push_to_write(queue, serde_json::to_string(&value).unwrap());
    }
    success
}

fn terminate_request(
    request: &Arc<AnalyzeRequest>,
    bots: &Arc<Vec<BotPtr>>,
    queue: &ThreadSafeQueue<String>,
) {
    let prev_status = request.status.swap(STATUS_TERMINATED, Ordering::AcqRel);
    match prev_status {
        STATUS_TERMINATED => {}
        STATUS_IN_QUEUE => {
            report_no_analysis(queue, request);
        }
        STATUS_POPPED => {}
        thread_idx => {
            if thread_idx >= 0 && (thread_idx as usize) < bots.len() {
                unsafe { (*bots[thread_idx as usize].as_ptr()).stop_without_wait() };
            }
        }
    }
}

fn analysis_loop(
    thread_idx: i32,
    bot_ptr: BotPtr,
    to_analyze_queue: Arc<ThreadSafePriorityQueue<(i64, i64), Arc<AnalyzeRequest>>>,
    to_write_queue: Arc<ThreadSafeQueue<String>>,
    open_requests: Arc<Mutex<HashMap<i64, Arc<AnalyzeRequest>>>>,
    logger: &'static Logger,
    log_search_info: bool,
    prevent_encore: bool,
) {
    // Safety: this thread exclusively owns `bot_ptr` until it reclaims the box
    // at the end of the function. The main thread never dereferences this pointer.
    let bot = unsafe { &mut *bot_ptr.as_ptr() };

    let mut item: ((i64, i64), Arc<AnalyzeRequest>) = ((0, 0), Arc::new(AnalyzeRequest::new()));
    while to_analyze_queue.wait_pop(&mut item) {
        let request = Arc::clone(&item.1);
        let expected = STATUS_IN_QUEUE;
        if request
            .status
            .compare_exchange(expected, STATUS_POPPED, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            // Already terminated; nothing to do. Remove from open requests.
            open_requests.lock().unwrap().remove(&request.internal_id);
            continue;
        }

        {
            bot.set_position(request.next_pla, &request.board, &request.hist);
            bot.set_always_include_owner_map(
                request.include_ownership
                    || request.include_ownership_stdev
                    || request.include_moves_ownership
                    || request.include_moves_ownership_stdev,
            );
            bot.set_params(&request.params);
            bot.set_avoid_move_until_by_loc(
                &request.avoid_move_until_by_loc_black,
                &request.avoid_move_until_by_loc_white,
            );

            let pla = request.next_pla;
            let search_factor = 1.0;

            let request_for_begun = Arc::clone(&request);
            let bot_for_begun = bot_ptr.clone();
            let on_search_begun = Box::new(move || {
                let expected2 = STATUS_POPPED;
                if request_for_begun
                    .status
                    .compare_exchange(expected2, thread_idx, Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
                {
                    // Safety: the analysis thread still owns the bot and the
                    // callback only runs while a search is active.
                    unsafe { (*bot_for_begun.as_ptr()).stop_without_wait() };
                }
            });

            if request.report_during_search {
                let request_for_callback = Arc::clone(&request);
                let queue_for_callback = to_write_queue.clone();
                let callback = Box::new(move |search: &Search| {
                    report_analysis(
                        queue_for_callback.as_ref(),
                        &request_for_callback,
                        search,
                        true,
                        prevent_encore,
                    );
                });
                bot.gen_move_synchronous_analyze_with_begun(
                    pla,
                    &TimeControls::new(),
                    search_factor,
                    request.report_during_search_every,
                    request.first_report_during_search_after,
                    callback,
                    Some(on_search_begun),
                );
            } else {
                bot.gen_move_synchronous_with_factor_and_begun(
                    pla,
                    &TimeControls::new(),
                    search_factor,
                    Some(on_search_begun),
                );
            }

            if log_search_info {
                // Minimal search info logging.
                let visits = bot
                    .get_search()
                    .root_node
                    .as_ref()
                    .map(|n| n.stats.visits.load(Ordering::Relaxed))
                    .unwrap_or(0);
                logger.write(&format!("Search info: visits={}", visits));
            }

            let search = bot.get_search_stop_and_wait();
            let analysis_written = report_analysis(
                to_write_queue.as_ref(),
                &request,
                search,
                false,
                prevent_encore,
            );
            if !analysis_written {
                if request.status.load(Ordering::Acquire) == STATUS_TERMINATED {
                    report_no_analysis(to_write_queue.as_ref(), &request);
                } else {
                    logger.write("Note: Search quitting due to no visits - this is normal and possible when shutting down but a bug under any other situation.");
                }
            }

            bot.clear_search();
        }

        open_requests.lock().unwrap().remove(&request.internal_id);
    }

    // Reclaim ownership so the bot is dropped when this thread exits.
    let _ = unsafe { Box::from_raw(bot_ptr.as_ptr()) };
}

#[allow(clippy::too_many_arguments)]
fn run_request_loop<R: BufRead>(
    input: R,
    to_write_queue: &Arc<ThreadSafeQueue<String>>,
    to_analyze_queue: &Arc<ThreadSafePriorityQueue<(i64, i64), Arc<AnalyzeRequest>>>,
    open_requests: &Arc<Mutex<HashMap<i64, Arc<AnalyzeRequest>>>>,
    bots: &Arc<Vec<BotPtr>>,
    logger: &Logger,
    log_all_requests: bool,
    _log_all_responses: bool,
    log_errors_and_warnings: bool,
    warn_unused_fields: bool,
    expected_keys: &HashSet<String>,
    default_params: &SearchParams,
    default_perspective: Player,
    default_analysis_pv_len: i32,
    assume_multiple_starting_black_moves_are_handicap: bool,
    prevent_encore: bool,
    nn_eval: &NnEvaluator,
    human_eval: Option<&NnEvaluator>,
) -> Result<(), StringError> {
    let push_write = |s: String| push_to_write(to_write_queue, s);
    let report_err = |s: &str| {
        report_error(to_write_queue, logger, log_errors_and_warnings, s);
    };

    let cfg_for_overrides = {
        let mut c = ConfigParser::new(false, false);
        c.initialize_map(std::collections::BTreeMap::new());
        c
    };

    let mut num_requests_so_far: i64 = 0;
    let mut internal_id_counter: i64 = 0;

    for line_result in input.lines() {
        let line = match line_result {
            Ok(l) => kata_core::global::trim(&l).to_string(),
            Err(e) => {
                report_err(&format!("Error reading stdin: {}", e));
                continue;
            }
        };
        if line.is_empty() {
            continue;
        }

        if log_all_requests {
            logger.write(&format!("Request: {}", line));
        }

        let input_json: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                report_err(&format!(
                    "{} - could not parse input line as json request: {}",
                    e, line
                ));
                continue;
            }
        };

        if !input_json.is_object() {
            report_err(&format!(
                "Request line was valid json but was not an object, ignoring: {}",
                input_json
            ));
            continue;
        }
        let input_obj = input_json.as_object().unwrap();

        let id = match input_obj.get("id") {
            Some(Value::String(s)) => s.clone(),
            _ => {
                report_err("Request must have a string \"id\" field");
                continue;
            }
        };

        // Special actions.
        if let Some(Value::String(action)) = input_obj.get("action") {
            match action.as_str() {
                "query_version" => {
                    let mut ret = input_json.clone();
                    if let Value::Object(ref mut m) = ret {
                        m.insert("version".to_string(), Value::String(get_version_string()));
                        m.insert("git_hash".to_string(), Value::String(String::new()));
                    }
                    push_write(ret.to_string());
                }
                "query_models" => {
                    let mut ret = input_json.clone();
                    if let Value::Object(ref mut m) = ret {
                        let mut models = Vec::new();
                        models.push(nn_eval_model_info(nn_eval));
                        if let Some(he) = human_eval {
                            models.push(nn_eval_model_info(he));
                        }
                        m.insert("models".to_string(), Value::Array(models));
                    }
                    push_write(ret.to_string());
                }
                "clear_cache" => {
                    nn_eval.clear_cache();
                    if let Some(he) = human_eval {
                        he.clear_cache();
                    }
                    push_write(input_json.to_string());
                }
                "terminate" => {
                    let terminate_id = match input_obj.get("terminateId") {
                        Some(Value::String(s)) => s.clone(),
                        _ => {
                            report_error_for_id(
                                to_write_queue,
                                logger,
                                log_errors_and_warnings,
                                &id,
                                "terminateId",
                                "Requests for a terminate action must have a string \"terminateId\" field",
                            );
                            continue;
                        }
                    };
                    let turn_numbers = match parse_turn_numbers(input_obj.get("turnNumbers")) {
                        Ok(v) => v,
                        Err(msg) => {
                            report_error_for_id(
                                to_write_queue,
                                logger,
                                log_errors_and_warnings,
                                &id,
                                "turnNumbers",
                                &msg,
                            );
                            continue;
                        }
                    };
                    {
                        let open = open_requests.lock().unwrap();
                        for request in open.values() {
                            if request.id == terminate_id
                                && (turn_numbers.is_none()
                                    || turn_numbers
                                        .as_ref()
                                        .unwrap()
                                        .contains(&request.turn_number))
                            {
                                terminate_request(request, bots, to_write_queue);
                            }
                        }
                    }
                    push_write(input_json.to_string());
                }
                "terminate_all" => {
                    let turn_numbers = match parse_turn_numbers(input_obj.get("turnNumbers")) {
                        Ok(v) => v,
                        Err(msg) => {
                            report_error_for_id(
                                to_write_queue,
                                logger,
                                log_errors_and_warnings,
                                &id,
                                "turnNumbers",
                                &msg,
                            );
                            continue;
                        }
                    };
                    {
                        let open = open_requests.lock().unwrap();
                        for request in open.values() {
                            if turn_numbers.is_none()
                                || turn_numbers
                                    .as_ref()
                                    .unwrap()
                                    .contains(&request.turn_number)
                            {
                                terminate_request(request, bots, to_write_queue);
                            }
                        }
                    }
                    push_write(input_json.to_string());
                }
                // "analyze" is not a special action: it falls through to the
                // same query parsing as a request with no "action" field at
                // all. (C++ has no such branch either — requests without an
                // action are analyze queries, analysis.cpp:507-614.)
                "analyze" => {}
                _ => {
                    report_error(
                        to_write_queue,
                        logger,
                        log_errors_and_warnings,
                        "'action' field must be 'query_version' or 'query_models' or 'clear_cache' or 'terminate' or 'terminate_all' or 'analyze'",
                    );
                }
            }
            // Every special action is fully handled above; "analyze" alone
            // continues into the query parsing below.
            if action.as_str() != "analyze" {
                continue;
            }
        }

        // Parse a query.
        let mut rbase = AnalyzeRequest::new();
        rbase.id = id.clone();
        rbase.params = default_params.clone();
        rbase.perspective = default_perspective;
        rbase.analysis_pv_len = default_analysis_pv_len;

        let parse_integer = |dict: &Map<String, Value>,
                             field: &str,
                             buf: &mut i64,
                             min: i64,
                             max: i64,
                             error_message: &str|
         -> bool {
            match dict.get(field) {
                Some(Value::Number(n)) if n.is_i64() => {
                    let x = n.as_i64().unwrap();
                    if x < min || x > max {
                        report_error_for_id(
                            to_write_queue,
                            logger,
                            log_errors_and_warnings,
                            &rbase.id,
                            field,
                            error_message,
                        );
                        return false;
                    }
                    *buf = x;
                    true
                }
                _ => {
                    report_error_for_id(
                        to_write_queue,
                        logger,
                        log_errors_and_warnings,
                        &rbase.id,
                        field,
                        error_message,
                    );
                    false
                }
            }
        };

        let parse_double = |dict: &Map<String, Value>,
                            field: &str,
                            buf: &mut f64,
                            min: f64,
                            max: f64,
                            error_message: &str|
         -> bool {
            match dict.get(field) {
                Some(Value::Number(n)) => {
                    let x = n.as_f64().unwrap();
                    if !x.is_finite() || x < min || x > max {
                        report_error_for_id(
                            to_write_queue,
                            logger,
                            log_errors_and_warnings,
                            &rbase.id,
                            field,
                            error_message,
                        );
                        return false;
                    }
                    *buf = x;
                    true
                }
                _ => {
                    report_error_for_id(
                        to_write_queue,
                        logger,
                        log_errors_and_warnings,
                        &rbase.id,
                        field,
                        error_message,
                    );
                    false
                }
            }
        };

        let parse_boolean =
            |dict: &Map<String, Value>, field: &str, buf: &mut bool, error_message: &str| -> bool {
                match dict.get(field) {
                    Some(Value::Bool(b)) => {
                        *buf = *b;
                        true
                    }
                    _ => {
                        report_error_for_id(
                            to_write_queue,
                            logger,
                            log_errors_and_warnings,
                            &rbase.id,
                            field,
                            error_message,
                        );
                        false
                    }
                }
            };

        let parse_player = |dict: &Map<String, Value>, field: &str, buf: &mut Player| -> bool {
            *buf = C_EMPTY;
            if let Some(Value::String(s)) = dict.get(field) {
                if let Some(p) = player_io::try_parse_player(s) {
                    *buf = p;
                }
            }
            if *buf != P_BLACK && *buf != P_WHITE {
                report_error_for_id(
                    to_write_queue,
                    logger,
                    log_errors_and_warnings,
                    &rbase.id,
                    field,
                    "Must be \"b\" or \"w\"",
                );
                false
            } else {
                true
            }
        };

        let board_x_size: i32;
        let board_y_size: i32;
        {
            let mut x_buf = 0i64;
            let mut y_buf = 0i64;
            let board_size_error = format!("Must provide an integer from 2 to {}", MAX_LEN);
            if input_obj.get("boardXSize").is_none() {
                report_error_for_id(
                    to_write_queue,
                    logger,
                    log_errors_and_warnings,
                    &rbase.id,
                    "boardXSize",
                    &board_size_error,
                );
                continue;
            }
            if input_obj.get("boardYSize").is_none() {
                report_error_for_id(
                    to_write_queue,
                    logger,
                    log_errors_and_warnings,
                    &rbase.id,
                    "boardYSize",
                    &board_size_error,
                );
                continue;
            }
            if !parse_integer(
                input_obj,
                "boardXSize",
                &mut x_buf,
                2,
                MAX_LEN as i64,
                &board_size_error,
            ) {
                continue;
            }
            if !parse_integer(
                input_obj,
                "boardYSize",
                &mut y_buf,
                2,
                MAX_LEN as i64,
                &board_size_error,
            ) {
                continue;
            }
            board_x_size = x_buf as i32;
            board_y_size = y_buf as i32;
        }

        let parse_board_locs = |dict: &Map<String, Value>,
                                field: &str,
                                buf: &mut Vec<Loc>,
                                allow_pass: bool|
         -> bool {
            buf.clear();
            let Some(Value::Array(arr)) = dict.get(field) else {
                report_error_for_id(
                    to_write_queue,
                    logger,
                    log_errors_and_warnings,
                    &rbase.id,
                    field,
                    "Must be an array of GTP board vertices",
                );
                return false;
            };
            for elt in arr {
                let Value::String(s) = elt else {
                    report_error_for_id(
                        to_write_queue,
                        logger,
                        log_errors_and_warnings,
                        &rbase.id,
                        field,
                        "Must be an array of GTP board vertices",
                    );
                    return false;
                };
                let Some(loc) = location::try_of_string(s, board_x_size, board_y_size) else {
                    report_error_for_id(
                        to_write_queue,
                        logger,
                        log_errors_and_warnings,
                        &rbase.id,
                        field,
                        &format!("Could not parse board location: {}", s),
                    );
                    return false;
                };
                if (!allow_pass && loc == PASS_LOC) || loc == NULL_LOC {
                    report_error_for_id(
                        to_write_queue,
                        logger,
                        log_errors_and_warnings,
                        &rbase.id,
                        field,
                        &format!("Could not parse board location: {}", s),
                    );
                    return false;
                }
                buf.push(loc);
            }
            true
        };

        let parse_board_moves = |dict: &Map<String, Value>,
                                 field: &str,
                                 buf: &mut Vec<Move>,
                                 allow_pass: bool|
         -> bool {
            buf.clear();
            let Some(Value::Array(arr)) = dict.get(field) else {
                report_error_for_id(
                    to_write_queue,
                    logger,
                    log_errors_and_warnings,
                    &rbase.id,
                    field,
                    "Must be an array of pairs of the form: [\"b\" or \"w\", GTP board vertex]",
                );
                return false;
            };
            for elt in arr {
                let Value::Array(pair) = elt else {
                    report_error_for_id(
                        to_write_queue,
                        logger,
                        log_errors_and_warnings,
                        &rbase.id,
                        field,
                        "Must be an array of pairs of the form: [\"b\" or \"w\", GTP board vertex]",
                    );
                    return false;
                };
                if pair.len() != 2 {
                    report_error_for_id(
                        to_write_queue,
                        logger,
                        log_errors_and_warnings,
                        &rbase.id,
                        field,
                        "Must be an array of pairs of the form: [\"b\" or \"w\", GTP board vertex]",
                    );
                    return false;
                }
                let (Value::String(s0), Value::String(s1)) = (&pair[0], &pair[1]) else {
                    report_error_for_id(
                        to_write_queue,
                        logger,
                        log_errors_and_warnings,
                        &rbase.id,
                        field,
                        "Must be an array of pairs of the form: [\"b\" or \"w\", GTP board vertex]",
                    );
                    return false;
                };
                let Some(pla) = player_io::try_parse_player(s0) else {
                    report_error_for_id(
                        to_write_queue,
                        logger,
                        log_errors_and_warnings,
                        &rbase.id,
                        field,
                        &format!("Could not parse player: {}", s0),
                    );
                    return false;
                };
                let Some(loc) = location::try_of_string(s1, board_x_size, board_y_size) else {
                    report_error_for_id(
                        to_write_queue,
                        logger,
                        log_errors_and_warnings,
                        &rbase.id,
                        field,
                        &format!("Could not parse board location: {}", s1),
                    );
                    return false;
                };
                if (!allow_pass && loc == PASS_LOC) || loc == NULL_LOC {
                    report_error_for_id(
                        to_write_queue,
                        logger,
                        log_errors_and_warnings,
                        &rbase.id,
                        field,
                        &format!("Could not parse board location: {}", s1),
                    );
                    return false;
                }
                buf.push(Move::new(loc, pla));
            }
            true
        };

        let mut placements = Vec::new();
        if let Some(_) = input_obj.get("initialStones") {
            if !parse_board_moves(input_obj, "initialStones", &mut placements, false) {
                continue;
            }
        }
        let mut move_history = Vec::new();
        if let Some(_) = input_obj.get("moves") {
            if !parse_board_moves(input_obj, "moves", &mut move_history, true) {
                continue;
            }
        } else {
            report_error_for_id(
                to_write_queue,
                logger,
                log_errors_and_warnings,
                &rbase.id,
                "moves",
                "Must specify an array of [player,location] pairs",
            );
            continue;
        }

        let mut initial_player = C_EMPTY;
        if input_obj.get("initialPlayer").is_some() {
            if !parse_player(input_obj, "initialPlayer", &mut initial_player) {
                continue;
            }
        }

        let mut should_analyze = vec![false; move_history.len() + 1];
        if let Some(v) = input_obj.get("analyzeTurns") {
            match parse_int_array(v) {
                Ok(turns) => {
                    let mut failed = false;
                    for turn_number in turns {
                        if turn_number < 0 || turn_number >= should_analyze.len() as i32 {
                            report_error_for_id(
                                to_write_queue,
                                logger,
                                log_errors_and_warnings,
                                &rbase.id,
                                "analyzeTurns",
                                &format!("Invalid turn number: {}", turn_number),
                            );
                            failed = true;
                            break;
                        }
                        should_analyze[turn_number as usize] = true;
                    }
                    if failed {
                        continue;
                    }
                }
                Err(_) => {
                    report_error_for_id(
                        to_write_queue,
                        logger,
                        log_errors_and_warnings,
                        &rbase.id,
                        "analyzeTurns",
                        "Must specify an array of integers indicating turns to analyze",
                    );
                    continue;
                }
            }
        } else {
            let last = should_analyze.len() - 1;
            should_analyze[last] = true;
        }

        let mut priorities: HashMap<i32, i64> = HashMap::new();
        if let Some(v) = input_obj.get("priorities") {
            let priorities_vec = match parse_i64_array(v) {
                Ok(v) => v,
                Err(_) => {
                    report_error_for_id(
                        to_write_queue,
                        logger,
                        log_errors_and_warnings,
                        &rbase.id,
                        "priorities",
                        "Must specify an array of integers indicating priorities",
                    );
                    continue;
                }
            };
            if input_obj.get("analyzeTurns").is_none() {
                report_error_for_id(
                    to_write_queue,
                    logger,
                    log_errors_and_warnings,
                    &rbase.id,
                    "priorities",
                    "Can only specify when also specifying analyzeTurns",
                );
                continue;
            }
            let analyze_turns = match parse_int_array(input_obj.get("analyzeTurns").unwrap()) {
                Ok(v) => v,
                Err(_) => {
                    report_error_for_id(
                        to_write_queue,
                        logger,
                        log_errors_and_warnings,
                        &rbase.id,
                        "priorities",
                        " analyzeTurns must be valid",
                    );
                    continue;
                }
            };
            if priorities_vec.len() != analyze_turns.len() {
                report_error_for_id(
                    to_write_queue,
                    logger,
                    log_errors_and_warnings,
                    &rbase.id,
                    "priorities",
                    "Must be of matching length to analyzeTurns",
                );
                continue;
            }
            let mut failed = false;
            for (i, priority) in priorities_vec.iter().enumerate() {
                if *priority < -(0x3FFF_ffff_FFFF_ffffi64) || *priority > 0x3FFF_ffff_FFFF_ffffi64 {
                    report_error_for_id(
                        to_write_queue,
                        logger,
                        log_errors_and_warnings,
                        &rbase.id,
                        "priorities",
                        &format!("Invalid priority: {}", priority),
                    );
                    failed = true;
                    break;
                }
                priorities.insert(analyze_turns[i], *priority);
            }
            if failed {
                priorities.clear();
                continue;
            }
        }

        let mut rules: Rules = match input_obj.get("rules") {
            Some(Value::String(s)) => match Rules::try_parse_rules(s) {
                Some(r) => r,
                None => {
                    report_error_for_id(
                        to_write_queue,
                        logger,
                        log_errors_and_warnings,
                        &rbase.id,
                        "rules",
                        &format!("Could not parse rules: {}", s),
                    );
                    continue;
                }
            },
            Some(Value::Object(m)) => {
                let mut r = Rules::default();
                for (k, v) in m {
                    let value_str = match v {
                        Value::String(s) => s.clone(),
                        _ => v.to_string(),
                    };
                    match Rules::update_rules(k, &value_str, &r) {
                        Ok(updated) => r = updated,
                        Err(e) => {
                            report_error_for_id(
                                to_write_queue,
                                logger,
                                log_errors_and_warnings,
                                &rbase.id,
                                "rules",
                                &format!("Could not parse rules: {}={} ({e})", k, value_str),
                            );
                            continue;
                        }
                    }
                }
                r
            }
            _ => {
                report_error_for_id(
                    to_write_queue,
                    logger,
                    log_errors_and_warnings,
                    &rbase.id,
                    "rules",
                    "Must specify rules string, such as \"chinese\" or \"tromp-taylor\", or a JSON object with detailed rules parameters.",
                );
                continue;
            }
        };

        if input_obj.get("komi").is_some() {
            let mut komi = 0.0;
            const KOMI_MSG: &str = "Must be a integer or half-integer from -150.0 to 150.0";
            if !parse_double(
                input_obj,
                "komi",
                &mut komi,
                Rules::MIN_USER_KOMI as f64,
                Rules::MAX_USER_KOMI as f64,
                KOMI_MSG,
            ) {
                continue;
            }
            rules.set_komi(komi as f32);
            if !Rules::komi_is_int_or_half_int(rules.komi_f32()) {
                report_error_for_id(
                    to_write_queue,
                    logger,
                    log_errors_and_warnings,
                    &rbase.id,
                    "komi",
                    KOMI_MSG,
                );
                continue;
            }
        }

        if let Some(v) = input_obj.get("whiteHandicapBonus") {
            let Value::String(s) = v else {
                report_error_for_id(
                    to_write_queue,
                    logger,
                    log_errors_and_warnings,
                    &rbase.id,
                    "whiteHandicapBonus",
                    "Must be a string",
                );
                continue;
            };
            match s.parse::<WhiteHandicapBonusRule>() {
                Ok(rule) => rules.white_handicap_bonus_rule = rule,
                Err(e) => {
                    report_error_for_id(
                        to_write_queue,
                        logger,
                        log_errors_and_warnings,
                        &rbase.id,
                        "whiteHandicapBonus",
                        &e.0,
                    );
                    continue;
                }
            }
        }

        if let Some(v) = input_obj.get("overrideSettings") {
            let Value::Object(settings) = v else {
                report_error_for_id(
                    to_write_queue,
                    logger,
                    log_errors_and_warnings,
                    &rbase.id,
                    "overrideSettings",
                    "Must be an object",
                );
                continue;
            };
            if !settings.is_empty() {
                let mut override_settings: std::collections::BTreeMap<String, String> =
                    std::collections::BTreeMap::new();
                for (k, val) in settings {
                    override_settings.insert(
                        k.clone(),
                        match val {
                            Value::String(s) => s.clone(),
                            _ => val.to_string(),
                        },
                    );
                }
                let mut local_cfg = cfg_for_overrides.clone();
                local_cfg.mark_all_keys_used_with_prefix("");
                local_cfg.override_keys_map(&override_settings);
                match setup::load_single_params_with_human(
                    &local_cfg,
                    SetupFor::Analysis,
                    human_eval.is_some(),
                ) {
                    Ok(mut p) => {
                        if let Err(e) =
                            SearchParams::fail_if_params_differ_on_unchangeable_parameter(
                                default_params,
                                &p,
                            )
                        {
                            report_error_for_id(
                                to_write_queue,
                                logger,
                                log_errors_and_warnings,
                                &rbase.id,
                                "overrideSettings",
                                &format!("Could not set settings: {}", e),
                            );
                            continue;
                        }
                        if !local_cfg.contains("conservativePass") {
                            p.conservative_pass = true;
                        }
                        let mut out = Vec::new();
                        if let Err(e) = setup::maybe_warn_human_sl_params(
                            &p,
                            Some(nn_eval),
                            human_eval,
                            &mut out,
                            None,
                        ) {
                            report_error_for_id(
                                to_write_queue,
                                logger,
                                log_errors_and_warnings,
                                &rbase.id,
                                "overrideSettings",
                                &format!("Could not set settings: {}", e),
                            );
                            continue;
                        }
                        rbase.params = p;
                        rbase.perspective = match setup::parse_report_analysis_winrates(
                            &local_cfg,
                            default_perspective,
                        ) {
                            Ok(p) => p,
                            Err(e) => {
                                report_error_for_id(
                                    to_write_queue,
                                    logger,
                                    log_errors_and_warnings,
                                    &rbase.id,
                                    "overrideSettings",
                                    &format!("Could not set settings: {}", e),
                                );
                                continue;
                            }
                        };
                        let unused_keys = local_cfg.unused_keys();
                        if !unused_keys.is_empty() {
                            report_warning_for_id(
                                to_write_queue,
                                logger,
                                log_errors_and_warnings,
                                &rbase.id,
                                "overrideSettings",
                                &format!(
                                    "Unknown config params: {}",
                                    kata_core::global::concat_vec(&unused_keys, ",")
                                ),
                            );
                        }
                    }
                    Err(e) => {
                        report_error_for_id(
                            to_write_queue,
                            logger,
                            log_errors_and_warnings,
                            &rbase.id,
                            "overrideSettings",
                            &format!("Could not set settings: {}", e),
                        );
                        continue;
                    }
                }
            }
        }

        if input_obj.get("maxVisits").is_some() {
            let mut buf = 0i64;
            if !parse_integer(
                input_obj,
                "maxVisits",
                &mut buf,
                1,
                1i64 << 50,
                "Must be an integer from 1 to 2^50",
            ) {
                continue;
            }
            rbase.params.max_visits = buf;
        }

        if input_obj.get("analysisPVLen").is_some() {
            let mut buf = 0i64;
            if !parse_integer(
                input_obj,
                "analysisPVLen",
                &mut buf,
                1,
                1000,
                "Must be an integer from 1 to 1000",
            ) {
                continue;
            }
            rbase.analysis_pv_len = buf as i32;
        }

        if input_obj.get("rootFpuReductionMax").is_some() {
            let mut buf = 0.0;
            if !parse_double(
                input_obj,
                "rootFpuReductionMax",
                &mut buf,
                0.0,
                2.0,
                "Must be a number from 0.0 to 2.0",
            ) {
                continue;
            }
            rbase.params.root_fpu_reduction_max = buf;
        }

        if input_obj.get("rootPolicyTemperature").is_some() {
            let mut buf = 0.0;
            if !parse_double(
                input_obj,
                "rootPolicyTemperature",
                &mut buf,
                0.01,
                100.0,
                "Must be a number from 0.01 to 100.0",
            ) {
                continue;
            }
            rbase.params.root_policy_temperature = buf;
            rbase.params.root_policy_temperature_early = buf;
        }

        if input_obj.get("includeMovesOwnership").is_some() {
            let mut buf = false;
            if !parse_boolean(
                input_obj,
                "includeMovesOwnership",
                &mut buf,
                "Must be a boolean",
            ) {
                continue;
            }
            rbase.include_moves_ownership = buf;
        }
        if input_obj.get("includeMovesOwnershipStdev").is_some() {
            let mut buf = false;
            if !parse_boolean(
                input_obj,
                "includeMovesOwnershipStdev",
                &mut buf,
                "Must be a boolean",
            ) {
                continue;
            }
            rbase.include_moves_ownership_stdev = buf;
        }
        if input_obj.get("includeOwnership").is_some() {
            let mut buf = false;
            if !parse_boolean(input_obj, "includeOwnership", &mut buf, "Must be a boolean") {
                continue;
            }
            rbase.include_ownership = buf;
        }
        if input_obj.get("includeOwnershipStdev").is_some() {
            let mut buf = false;
            if !parse_boolean(
                input_obj,
                "includeOwnershipStdev",
                &mut buf,
                "Must be a boolean",
            ) {
                continue;
            }
            rbase.include_ownership_stdev = buf;
        }
        if input_obj.get("includePolicy").is_some() {
            let mut buf = false;
            if !parse_boolean(input_obj, "includePolicy", &mut buf, "Must be a boolean") {
                continue;
            }
            rbase.include_policy = buf;
        }
        if input_obj.get("includePVVisits").is_some() {
            let mut buf = false;
            if !parse_boolean(input_obj, "includePVVisits", &mut buf, "Must be a boolean") {
                continue;
            }
            rbase.include_pv_visits = buf;
        }
        if input_obj.get("includeNoResultValue").is_some() {
            let mut buf = false;
            if !parse_boolean(
                input_obj,
                "includeNoResultValue",
                &mut buf,
                "Must be a boolean",
            ) {
                continue;
            }
            rbase.include_no_result_value = buf;
        }

        if input_obj.get("reportDuringSearchEvery").is_some() {
            let mut buf = 0.0;
            if !parse_double(
                input_obj,
                "reportDuringSearchEvery",
                &mut buf,
                0.001,
                1_000_000.0,
                "Must be number of seconds from 0.001 to 1000000.0",
            ) {
                continue;
            }
            rbase.report_during_search_every = buf;
            rbase.report_during_search = true;
            rbase.first_report_during_search_after = buf;
        }
        if input_obj.get("firstReportDuringSearchAfter").is_some() {
            let mut buf = 0.0;
            if !parse_double(
                input_obj,
                "firstReportDuringSearchAfter",
                &mut buf,
                0.001,
                1_000_000.0,
                "Must be number of seconds from 0.001 to 1000000.0",
            ) {
                continue;
            }
            rbase.first_report_during_search_after = buf;
            rbase.report_during_search = true;
        }

        if input_obj.get("priority").is_some() {
            if input_obj.get("priorities").is_some() {
                report_error_for_id(
                    to_write_queue,
                    logger,
                    log_errors_and_warnings,
                    &rbase.id,
                    "priority",
                    "Cannot specify both priority and priorities",
                );
                continue;
            }
            let mut buf = 0i64;
            if !parse_integer(
                input_obj,
                "priority",
                &mut buf,
                -(0x3FFF_ffff_FFFF_ffffi64),
                0x3FFF_ffff_FFFF_ffffi64,
                "Must be a number between -2^62 and 2^62",
            ) {
                continue;
            }
            rbase.priority = buf;
        }

        let has_allow_moves = input_obj.get("allowMoves").is_some();
        let has_avoid_moves = input_obj.get("avoidMoves").is_some();
        if has_allow_moves || has_avoid_moves {
            if has_allow_moves && has_avoid_moves {
                report_error_for_id(
                    to_write_queue,
                    logger,
                    log_errors_and_warnings,
                    &rbase.id,
                    "allowMoves",
                    "Cannot specify both allowMoves and avoidMoves",
                );
                continue;
            }
            let field = if has_allow_moves {
                "allowMoves"
            } else {
                "avoidMoves"
            };
            let Some(Value::Array(avoid_params_list)) = input_obj.get(field) else {
                report_error_for_id(
                    to_write_queue,
                    logger,
                    log_errors_and_warnings,
                    &rbase.id,
                    field,
                    "Must be a list of dicts with subfields 'player', 'moves', 'untilDepth'",
                );
                continue;
            };
            if has_allow_moves && avoid_params_list.len() > 1 {
                report_error_for_id(
                    to_write_queue,
                    logger,
                    log_errors_and_warnings,
                    &rbase.id,
                    field,
                    "Currently allowMoves only allows one entry",
                );
                continue;
            }

            let mut failed = false;
            for avoid_params in avoid_params_list {
                let Value::Object(params_obj) = avoid_params else {
                    report_error_for_id(
                        to_write_queue,
                        logger,
                        log_errors_and_warnings,
                        &rbase.id,
                        field,
                        "Must be a list of dicts with subfields 'player', 'moves', 'untilDepth'",
                    );
                    failed = true;
                    break;
                };
                if params_obj.get("moves").is_none()
                    || params_obj.get("untilDepth").is_none()
                    || params_obj.get("player").is_none()
                {
                    report_error_for_id(
                        to_write_queue,
                        logger,
                        log_errors_and_warnings,
                        &rbase.id,
                        field,
                        "Must be a list of dicts with subfields 'player', 'moves', 'untilDepth'",
                    );
                    failed = true;
                    break;
                }
                let mut avoid_pla = C_EMPTY;
                if !parse_player(params_obj, "player", &mut avoid_pla) {
                    failed = true;
                    break;
                }
                let mut parsed_locs = Vec::new();
                if !parse_board_locs(params_obj, "moves", &mut parsed_locs, true) {
                    failed = true;
                    break;
                }
                let mut until_depth = 0i64;
                if !parse_integer(
                    params_obj,
                    "untilDepth",
                    &mut until_depth,
                    1,
                    1_000_000_000,
                    "Must be a positive integer",
                ) {
                    failed = true;
                    break;
                }
                let avoid_move_until_by_loc = if avoid_pla == P_BLACK {
                    &mut rbase.avoid_move_until_by_loc_black
                } else {
                    &mut rbase.avoid_move_until_by_loc_white
                };
                avoid_move_until_by_loc.resize(MAX_ARR_SIZE, until_depth as i32);
                if has_allow_moves {
                    for loc in 0..MAX_ARR_SIZE {
                        avoid_move_until_by_loc[loc] = until_depth as i32;
                    }
                    for loc in parsed_locs {
                        avoid_move_until_by_loc[loc as usize] = 0;
                    }
                } else {
                    for loc in parsed_locs {
                        avoid_move_until_by_loc[loc as usize] = until_depth as i32;
                    }
                }
            }
            if failed {
                continue;
            }
        }

        let mut board = Board::new(board_x_size, board_y_size);
        for placement in &placements {
            board.set_stone(placement.loc, placement.pla);
        }

        if initial_player == C_EMPTY {
            if !move_history.is_empty() {
                initial_player = move_history[0].pla;
            } else if BoardHistory::num_handicap_stones_on_board(&board) > 0 {
                initial_player = P_WHITE;
            } else {
                initial_player = P_BLACK;
            }
        }

        let mut supported = false;
        let supported_rules = nn_eval.supported_rules(rules, &mut supported);
        if !supported {
            let warning = format!(
                "Rules {:?} not supported by neural net, using {:?} instead",
                rules, supported_rules
            );
            report_warning_for_id(
                to_write_queue,
                logger,
                log_errors_and_warnings,
                &rbase.id,
                "rules",
                &warning,
            );
            rules = supported_rules;
        }

        let mut next_pla = initial_player;
        let mut hist = BoardHistory::new(board.clone(), next_pla, rules, 0);
        hist.set_assume_multiple_starting_black_moves_are_handicap(
            assume_multiple_starting_black_moves_are_handicap,
        );

        if warn_unused_fields {
            for key in input_obj.keys() {
                if !expected_keys.contains(key) {
                    report_warning_for_id(
                        to_write_queue,
                        logger,
                        log_errors_and_warnings,
                        &rbase.id,
                        key,
                        "Unexpected or unused field, do you have a typo? (set warnUnusedFields=false in the config to disable this warning)",
                    );
                }
            }
        }

        // Build and enqueue requests, replaying moves and detecting illegal ones.
        let mut new_requests: Vec<Arc<AnalyzeRequest>> = Vec::new();
        let mut found_illegal_move = false;
        for turn_number in 0..=move_history.len() {
            if should_analyze[turn_number] {
                let priority = if !priorities.is_empty() {
                    *priorities
                        .get(&(turn_number as i32))
                        .unwrap_or(&rbase.priority)
                } else {
                    rbase.priority
                };

                let mut new_request = AnalyzeRequest::new();
                new_request.internal_id = internal_id_counter;
                internal_id_counter += 1;
                new_request.id = rbase.id.clone();
                new_request.turn_number = turn_number as i32;
                new_request.board = board.clone();
                new_request.hist = hist.clone();
                new_request.next_pla = next_pla;
                new_request.params = rbase.params.clone();
                new_request.perspective = rbase.perspective;
                new_request.analysis_pv_len = rbase.analysis_pv_len;
                new_request.include_ownership = rbase.include_ownership;
                new_request.include_ownership_stdev = rbase.include_ownership_stdev;
                new_request.include_moves_ownership = rbase.include_moves_ownership;
                new_request.include_moves_ownership_stdev = rbase.include_moves_ownership_stdev;
                new_request.include_policy = rbase.include_policy;
                new_request.include_pv_visits = rbase.include_pv_visits;
                new_request.include_no_result_value = rbase.include_no_result_value;
                new_request.report_during_search = rbase.report_during_search;
                new_request.report_during_search_every = rbase.report_during_search_every;
                new_request.first_report_during_search_after =
                    rbase.first_report_during_search_after;
                new_request.priority = priority;
                new_request.avoid_move_until_by_loc_black =
                    rbase.avoid_move_until_by_loc_black.clone();
                new_request.avoid_move_until_by_loc_white =
                    rbase.avoid_move_until_by_loc_white.clone();
                new_request.status = AtomicI32::new(STATUS_IN_QUEUE);
                new_requests.push(Arc::new(new_request));
            }
            if turn_number >= move_history.len() {
                break;
            }

            let move_pla = move_history[turn_number].pla;
            let move_loc = move_history[turn_number].loc;
            if move_pla != next_pla {
                board.clear_simple_ko_loc();
                hist.clear(board.clone(), move_pla, rules, hist.encore_phase);
                hist.set_assume_multiple_starting_black_moves_are_handicap(
                    assume_multiple_starting_black_moves_are_handicap,
                );
            }

            if !hist.make_board_move_tolerant_with_prevent(
                &mut board,
                move_loc,
                move_pla,
                prevent_encore,
            ) {
                report_error_for_id(
                    to_write_queue,
                    logger,
                    log_errors_and_warnings,
                    &rbase.id,
                    "moves",
                    &format!(
                        "Illegal move {}: {}",
                        turn_number,
                        location::to_string(move_loc, board.x_size, board.y_size)
                    ),
                );
                found_illegal_move = true;
                break;
            }
            next_pla = kata_game::board::get_opp(move_pla);
        }

        if found_illegal_move {
            new_requests.clear();
            continue;
        }

        {
            let mut open = open_requests.lock().unwrap();
            for req in &new_requests {
                open.insert(req.internal_id, Arc::clone(req));
            }
        }
        for req in &new_requests {
            let priority_key = (req.priority, -num_requests_so_far);
            to_analyze_queue.force_push(priority_key, Arc::clone(req));
            num_requests_so_far += 1;
        }
    }

    Ok(())
}

fn nn_eval_model_info(nn_eval: &NnEvaluator) -> Value {
    let mut m = Map::new();
    m.insert(
        "name".to_string(),
        Value::String(nn_eval.model_name().to_string()),
    );
    m.insert(
        "internalName".to_string(),
        Value::String(nn_eval.internal_model_name().to_string()),
    );
    m.insert(
        "maxBatchSize".to_string(),
        Value::Number(nn_eval.max_batch_size().into()),
    );
    m.insert(
        "usesHumanSLProfile".to_string(),
        Value::Bool(nn_eval.requires_sgf_metadata()),
    );
    m.insert(
        "version".to_string(),
        Value::Number(nn_eval.model_version().into()),
    );
    m.insert(
        "usingFP16".to_string(),
        Value::String(nn_eval.using_fp16_mode().to_string()),
    );
    Value::Object(m)
}

fn parse_turn_numbers(v: Option<&Value>) -> Result<Option<HashSet<i32>>, String> {
    match v {
        None => Ok(None),
        Some(value) => {
            let arr = parse_int_array(value)?;
            Ok(Some(arr.into_iter().collect()))
        }
    }
}

fn parse_int_array(v: &Value) -> Result<Vec<i32>, String> {
    let Value::Array(arr) = v else {
        return Err("Must be an array of integers".to_string());
    };
    let mut out = Vec::with_capacity(arr.len());
    for elt in arr {
        match elt {
            Value::Number(n) if n.is_i64() => out.push(n.as_i64().unwrap() as i32),
            _ => return Err("Must be an array of integers".to_string()),
        }
    }
    Ok(out)
}

fn parse_i64_array(v: &Value) -> Result<Vec<i64>, String> {
    let Value::Array(arr) = v else {
        return Err("Must be an array of integers".to_string());
    };
    let mut out = Vec::with_capacity(arr.len());
    for elt in arr {
        match elt {
            Value::Number(n) if n.is_i64() => out.push(n.as_i64().unwrap()),
            _ => return Err("Must be an array of integers".to_string()),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analysis_quit_action() {
        let input = r#"{"id":"foo","action":"query_version"}
{"id":"bar","action":"query_models"}
{"id":"baz","action":"clear_cache"}
{"id":"q","action":"terminate_all"}"#;
        let lines = analysis_impl(&[], input.as_bytes()).unwrap();
        assert!(!lines.is_empty());
        // All action requests are echoed back.
        assert_eq!(lines.len(), 4);
        for line in &lines {
            assert!(serde_json::from_str::<Value>(line).is_ok());
        }
    }

    #[test]
    fn test_analysis_simple_query() {
        let input = r#"{"id":"test","boardXSize":9,"boardYSize":9,"rules":"tromp-taylor","moves":[["b","d4"],["w","e4"]],"maxVisits":5}"#;
        let lines = analysis_impl(&[], input.as_bytes()).unwrap();
        assert!(!lines.is_empty(), "expected at least one response line");
        let value: Value = serde_json::from_str(&lines[lines.len() - 1]).unwrap();
        let obj = value.as_object().unwrap();
        assert_eq!(obj.get("id").unwrap().as_str().unwrap(), "test");
        assert_eq!(obj.get("turnNumber").unwrap().as_i64().unwrap(), 2);
        assert!(obj.get("moveInfos").is_some());
    }

    #[test]
    fn test_analysis_priority_ordering() {
        // Use two analysis threads so both queries can run concurrently. The
        // higher-priority query uses far fewer visits, so it should finish first
        // even if the lower-priority query is enqueued slightly earlier.
        let args = vec!["--analysis-threads".to_string(), "2".to_string()];
        let input = r#"{"id":"low","boardXSize":9,"boardYSize":9,"rules":"tromp-taylor","moves":[["b","d4"]],"maxVisits":100,"priority":10}
{"id":"high","boardXSize":9,"boardYSize":9,"rules":"tromp-taylor","moves":[["b","d4"]],"maxVisits":3,"priority":100}"#;
        let lines = analysis_impl(&args, input.as_bytes()).unwrap();
        let analysis_lines: Vec<&String> = lines
            .iter()
            .filter(|l| {
                let v: Value = serde_json::from_str(l).unwrap();
                v.get("moveInfos").is_some()
            })
            .collect();
        assert_eq!(analysis_lines.len(), 2);
        let first: Value = serde_json::from_str(analysis_lines[0]).unwrap();
        assert_eq!(first.get("id").unwrap().as_str().unwrap(), "high");
        let second: Value = serde_json::from_str(analysis_lines[1]).unwrap();
        assert_eq!(second.get("id").unwrap().as_str().unwrap(), "low");
    }

    #[test]
    fn test_analysis_terminate_in_queue() {
        let input = r#"{"id":"term","boardXSize":9,"boardYSize":9,"rules":"tromp-taylor","moves":[["b","d4"]],"maxVisits":1000}
{"id":"control","action":"terminate","terminateId":"term"}"#;
        let lines = analysis_impl(&[], input.as_bytes()).unwrap();
        // The terminate control echoes back; the terminated query may emit noResults.
        assert!(!lines.is_empty());
        let mut saw_terminate_response = false;
        let mut saw_no_results = false;
        for line in &lines {
            let v: Value = serde_json::from_str(line).unwrap();
            if let Some(id) = v.get("id").and_then(|x| x.as_str()) {
                if id == "control" {
                    saw_terminate_response = true;
                }
                if id == "term" && v.get("noResults").is_some() {
                    saw_no_results = true;
                }
            }
        }
        assert!(saw_terminate_response);
        assert!(saw_no_results);
    }

    #[test]
    fn test_analysis_parse_error() {
        let input = "not json";
        let lines = analysis_impl(&[], input.as_bytes()).unwrap();
        assert_eq!(lines.len(), 1);
        let v: Value = serde_json::from_str(&lines[0]).unwrap();
        assert!(v.get("error").is_some());
    }

    #[test]
    fn test_analysis_missing_id() {
        let input = r#"{"boardXSize":9,"boardYSize":9,"rules":"tromp-taylor","moves":[]}"#;
        let lines = analysis_impl(&[], input.as_bytes()).unwrap();
        assert_eq!(lines.len(), 1);
        let v: Value = serde_json::from_str(&lines[0]).unwrap();
        assert!(v.get("error").is_some());
    }

    #[test]
    fn test_analysis_analyze_action() {
        // An explicit "action":"analyze" request must be accepted and treated
        // exactly like a query without an action field: one result line per
        // requested turn, each carrying id/turnNumber/isDuringSearch.
        let input = r#"{"id":"a","action":"analyze","boardXSize":9,"boardYSize":9,"rules":"tromp-taylor","moves":[["b","d4"],["w","e4"]],"analyzeTurns":[0,1,2],"maxVisits":5}"#;
        let lines = analysis_impl(&[], input.as_bytes()).unwrap();
        let results: Vec<Value> = lines
            .iter()
            .map(|l| serde_json::from_str::<Value>(l).unwrap())
            .filter(|v| {
                v.get("id").and_then(|x| x.as_str()) == Some("a")
                    && v.get("warning").is_none()
                    && v.get("moveInfos").is_some()
            })
            .collect();
        assert_eq!(results.len(), 3, "one result per analyzed turn");
        let mut turn_numbers: Vec<i64> = results
            .iter()
            .map(|v| v.get("turnNumber").unwrap().as_i64().unwrap())
            .collect();
        turn_numbers.sort();
        assert_eq!(turn_numbers, vec![0, 1, 2]);
        for v in &results {
            assert!(
                v.get("moveInfos").is_some(),
                "expected real analysis in result: {}",
                v
            );
            assert_eq!(v.get("isDuringSearch"), Some(&Value::Bool(false)));
        }
    }

    #[test]
    fn test_analysis_analyze_action_sequence() {
        // query_version + analyze + terminate: version line, an analyze result
        // line (real analysis or noResults if terminated early), and the
        // terminate echo.
        let input = r#"{"id":"v","action":"query_version"}
{"id":"a","action":"analyze","boardXSize":9,"boardYSize":9,"rules":"tromp-taylor","moves":[["b","d4"]],"maxVisits":3}
{"id":"t","action":"terminate","terminateId":"a"}"#;
        let lines = analysis_impl(&[], input.as_bytes()).unwrap();
        let values: Vec<Value> = lines
            .iter()
            .map(|l| serde_json::from_str::<Value>(l).unwrap())
            .collect();
        assert!(
            values.iter().any(|v| v.get("version").is_some()),
            "query_version response"
        );
        assert!(
            values.iter().any(|v| {
                v.get("id").and_then(|x| x.as_str()) == Some("a")
                    && (v.get("moveInfos").is_some() || v.get("noResults").is_some())
                    && v.get("isDuringSearch") == Some(&Value::Bool(false))
            }),
            "analyze final result line"
        );
        assert!(
            values.iter().any(|v| {
                v.get("id").and_then(|x| x.as_str()) == Some("t")
                    && v.get("action").and_then(|x| x.as_str()) == Some("terminate")
            }),
            "terminate echo"
        );
    }

    #[test]
    fn test_analysis_analyze_action_errors() {
        // Missing "moves" and an unparseable move coordinate must both produce
        // per-id error lines, not a crash.
        let input = r#"{"id":"e1","action":"analyze","boardXSize":9,"boardYSize":9,"rules":"tromp-taylor"}
{"id":"e2","action":"analyze","boardXSize":9,"boardYSize":9,"rules":"tromp-taylor","moves":[["b","z99"]]}"#;
        let lines = analysis_impl(&[], input.as_bytes()).unwrap();
        let values: Vec<Value> = lines
            .iter()
            .map(|l| serde_json::from_str::<Value>(l).unwrap())
            .collect();
        assert_eq!(values.len(), 2);
        let e1 = &values[0];
        assert_eq!(e1.get("id").unwrap().as_str().unwrap(), "e1");
        assert_eq!(e1.get("field").unwrap().as_str().unwrap(), "moves");
        assert!(e1.get("error").is_some());
        let e2 = &values[1];
        assert_eq!(e2.get("id").unwrap().as_str().unwrap(), "e2");
        assert_eq!(e2.get("field").unwrap().as_str().unwrap(), "moves");
        assert!(e2.get("error").is_some());
    }
}
