//! Match command implementation.
//!
//! Corresponds to `MainCmds::match` in `cpp/command/match.cpp`.
//! Runs a match or tournament between two or more bots, each potentially using
//! a different neural-net model and search settings.

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

use std::collections::BTreeMap;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use clap::Parser;

use kata_core::config::ConfigParser;
use kata_core::global::{StringError, int_to_string, uint64_to_hex_string};
use kata_core::logger::{Logger, LoggerOptions};
use kata_core::rng::Rand;
use kata_data::sgf::write_sgf_loc;
use kata_game::board::{C_EMPTY, P_BLACK, P_WHITE};
use kata_nn::eval::NnEvaluator;
use kata_program::play::{BotSpec, GameRunner, MatchPairer};
use kata_program::play_settings::PlaySettings;
use kata_program::setup::{self, SetupFor};
use kata_search::pattern_bonus::PatternBonusTable;
use kata_search::search::Search;

use crate::cli::CommonArgs;

/// CLI arguments for the `match` subcommand.
#[derive(Parser, Debug, Clone)]
struct MatchArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// Log file to output to.
    #[arg(long = "log-file", value_name = "FILE")]
    log_file: Option<String>,

    /// Directory to output SGF files.
    #[arg(long = "sgf-output-dir", value_name = "DIR")]
    sgf_output_dir: Option<String>,
}

/// Result of a single finished game, used for the final summary.
#[derive(Debug, Clone)]
struct GameResult {
    b_name: String,
    w_name: String,
    winner: String,
}

/// Public CLI entry point.
pub fn match_cmd(args: &[String]) -> i32 {
    match match_impl(args) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {}", e);
            1
        }
    }
}

fn match_impl(args: &[String]) -> Result<(), StringError> {
    let parsed = parse_args(args)?;
    let mut cfg = parsed.common.get_config("match_example.cfg")?;

    let log_file = parsed.log_file.unwrap_or_default();
    let sgf_output_dir = parsed.sgf_output_dir.unwrap_or_default();

    let logger: &'static Logger = Box::leak(Box::new(Logger::new(
        LoggerOptions {
            log_to_stdout: true,
            log_to_stderr: true,
            log_time: true,
        },
        None,
    )));
    if !log_file.is_empty() {
        logger.add_file(&log_file, true);
    }

    parsed.common.log_overrides(logger);

    logger.write("Match Engine starting...");

    let paramss = setup::load_params(&cfg, SetupFor::Match)
        .map_err(|e| StringError::new(format!("Could not load search params: {}", e)))?;
    if paramss.is_empty() {
        return Err(StringError::new("No bots loaded from config".to_string()));
    }
    let num_bots = paramss.len() as i32;

    let matchups_per_round = compute_matchups(&cfg, num_bots)?;
    if matchups_per_round.is_empty() {
        return Err(StringError::new(
            "No matchups to play; check numBots and includeBots/extraPairs config".to_string(),
        ));
    }

    let (bot_names, nn_model_files_by_bot) = load_bot_names_and_models(&cfg, num_bots)?;

    let mut bot_is_used = vec![false; num_bots as usize];
    for (b, w) in &matchups_per_round {
        bot_is_used[*b as usize] = true;
        bot_is_used[*w as usize] = true;
    }

    let mut nn_model_files: Vec<String> = Vec::new();
    let mut which_nn_model = vec![0; num_bots as usize];
    for i in 0..num_bots as usize {
        if !bot_is_used[i] {
            continue;
        }
        let desired = &nn_model_files_by_bot[i];
        if let Some(idx) = nn_model_files.iter().position(|f| f == desired) {
            which_nn_model[i] = idx as i32;
        } else {
            which_nn_model[i] = nn_model_files.len() as i32;
            nn_model_files.push(desired.clone());
        }
    }

    let num_game_threads = cfg
        .get_int("numGameThreads", 1, 16384)
        .map_err(|e| StringError::new(e.to_string()))?;
    let mut seed_rand = Rand::new();
    let game_seed_base = uint64_to_hex_string(seed_rand.next_u64());

    let max_bot_threads = paramss.iter().map(|p| p.num_threads).max().unwrap_or(1);
    let expected_concurrent_evals = max_bot_threads * num_game_threads;

    let play_settings = PlaySettings::load_for_match(&cfg)
        .map_err(|e| StringError::new(format!("Could not load play settings: {}", e)))?;
    let game_runner = Arc::new(
        GameRunner::new(&cfg, play_settings, logger)
            .map_err(|e| StringError::new(format!("Could not create GameRunner: {}", e)))?,
    );

    let game_initializer = game_runner
        .get_game_initializer()
        .expect("GameRunner should have a GameInitializer");
    let min_board_x_size_used = game_initializer.get_min_board_x_size();
    let min_board_y_size_used = game_initializer.get_min_board_y_size();
    let max_board_x_size_used = game_initializer.get_max_board_x_size();
    let max_board_y_size_used = game_initializer.get_max_board_y_size();

    setup::initialize_session(&cfg);

    let default_max_batch_size = -1;
    let default_require_exact_nn_len = min_board_x_size_used == max_board_x_size_used
        && min_board_y_size_used == max_board_y_size_used;
    let disable_fp16 = false;

    let nn_evals: Vec<&'static NnEvaluator> = setup::initialize_nn_evaluators(
        nn_model_files.clone(),
        nn_model_files,
        Vec::new(),
        &cfg,
        logger,
        &mut seed_rand,
        expected_concurrent_evals,
        max_board_x_size_used,
        max_board_y_size_used,
        default_max_batch_size,
        default_require_exact_nn_len,
        disable_fp16,
        SetupFor::Match,
    )
    .map_err(|e| StringError::new(format!("Could not initialize NN evaluators: {}", e)))?
    .into_iter()
    .map(|eval| Box::leak(Box::new(eval)) as &'static NnEvaluator)
    .collect();

    logger.write("Loaded neural net");

    let nn_evals_by_bot: Vec<Option<&'static NnEvaluator>> = (0..num_bots)
        .map(|i| {
            if bot_is_used[i as usize] {
                Some(nn_evals[which_nn_model[i as usize] as usize])
            } else {
                None
            }
        })
        .collect();

    let pattern_bonus_tables = setup::load_avoid_sgf_pattern_bonus_tables(&cfg, logger)
        .map_err(|e| StringError::new(format!("Could not load pattern bonus tables: {}", e)))?;
    assert_eq!(pattern_bonus_tables.len() as i32, num_bots);
    let pattern_bonus_tables: Arc<Vec<Option<Box<PatternBonusTable>>>> = Arc::new(
        pattern_bonus_tables
            .into_iter()
            .map(|t| t.map(Box::new))
            .collect(),
    );

    // The Rust MatchPairer reads its pairings from the config, unlike the C++
    // constructor that takes them as an explicit argument. Compute the C++-style
    // matchups and inject them before constructing the pairer.
    let matchups_str = matchups_per_round
        .iter()
        .map(|(a, b)| format!("{}-{}", a, b))
        .collect::<Vec<_>>()
        .join(",");
    cfg.override_key("matchupsPerRound", &matchups_str);
    cfg.mark_key_used("matchupsPerRound");

    if !cfg.contains("logGamesEvery") {
        cfg.override_key("logGamesEvery", "1");
        cfg.mark_key_used("logGamesEvery");
    }

    let match_pairer = Arc::new(Mutex::new(
        MatchPairer::new(
            &cfg,
            num_bots,
            bot_names.clone(),
            nn_evals_by_bot.clone(),
            paramss.clone(),
        )
        .map_err(|e| StringError::new(format!("Could not create MatchPairer: {}", e)))?,
    ));

    cfg.warn_unused_keys(&mut std::io::stdout(), Some(logger))
        .map_err(|e| StringError::new(format!("Could not warn unused keys: {}", e)))?;
    for i in 0..num_bots {
        if bot_is_used[i as usize] {
            setup::maybe_warn_human_sl_params(
                &paramss[i as usize],
                nn_evals_by_bot[i as usize],
                None,
                &mut std::io::stdout(),
                Some(logger),
            )
            .map_err(|e| StringError::new(format!("Could not warn human SL params: {}", e)))?;
        }
    }

    logger.write("Loaded all config stuff, starting matches");
    if !logger.is_logging_to_stdout() {
        println!("Loaded all config stuff, starting matches");
    }

    if !sgf_output_dir.is_empty() {
        std::fs::create_dir_all(&sgf_output_dir)
            .map_err(|e| StringError::new(format!("Could not create SGF output dir: {}", e)))?;
    }

    let should_stop = Arc::new(AtomicBool::new(false));
    let game_count = Arc::new(AtomicI64::new(0));
    let time_used_by_bot = Arc::new(Mutex::new(BTreeMap::<String, f64>::new()));
    let moves_by_bot = Arc::new(Mutex::new(BTreeMap::<String, f64>::new()));
    let wins_by_bot = Arc::new(Mutex::new(BTreeMap::<String, i64>::new()));
    let game_results = Arc::new(Mutex::new(Vec::<GameResult>::new()));

    let mut handles = Vec::with_capacity(num_game_threads as usize);
    let mut hash_rand = Rand::new();
    for _ in 0..num_game_threads {
        let thread_hash = hash_rand.next_u64();
        let game_runner = Arc::clone(&game_runner);
        let match_pairer = Arc::clone(&match_pairer);
        let pattern_bonus_tables = Arc::clone(&pattern_bonus_tables);
        let should_stop = Arc::clone(&should_stop);
        let game_count = Arc::clone(&game_count);
        let time_used_by_bot = Arc::clone(&time_used_by_bot);
        let moves_by_bot = Arc::clone(&moves_by_bot);
        let wins_by_bot = Arc::clone(&wins_by_bot);
        let game_results = Arc::clone(&game_results);
        let sgf_output_dir = sgf_output_dir.clone();
        let game_seed_base = game_seed_base.clone();

        handles.push(thread::spawn(move || -> Result<(), StringError> {
            let mut sgf_out: Option<Box<dyn Write + Send>> = None;
            if !sgf_output_dir.is_empty() {
                let path = format!(
                    "{}/{}.sgfs",
                    sgf_output_dir,
                    uint64_to_hex_string(thread_hash)
                );
                let file = std::fs::File::create(&path).map_err(|e| {
                    StringError::new(format!("Could not create SGF output file {}: {}", path, e))
                })?;
                sgf_out = Some(Box::new(file));
            }

            let mut this_loop_seed_rand = Rand::new();
            loop {
                if should_stop.load(Ordering::Relaxed) {
                    break;
                }

                let maybe_matchup = {
                    let mut pairer = match_pairer.lock().map_err(|e| {
                        StringError::new(format!("MatchPairer mutex poisoned: {}", e))
                    })?;
                    pairer.get_matchup(logger)
                };
                let (bot_spec_b, bot_spec_w) = match maybe_matchup {
                    Some(m) => m,
                    None => break,
                };

                let seed = format!(
                    "{}:{}",
                    game_seed_base,
                    uint64_to_hex_string(this_loop_seed_rand.next_u64())
                );

                let tables = Arc::clone(&pattern_bonus_tables);
                let after_initialization: Box<
                    dyn Fn(&BotSpec<'static>, *mut Search<'static>) + Send + Sync,
                > = Box::new(move |spec, search| unsafe {
                    (*search)
                        .set_copy_of_external_pattern_bonus_table(&tables[spec.bot_idx as usize]);
                });

                let should_stop_for_game = Arc::clone(&should_stop);
                let should_stop_func: Box<dyn Fn() -> bool + Send + Sync> =
                    Box::new(move || should_stop_for_game.load(Ordering::Relaxed));

                let game_data = game_runner.run_game(
                    &seed,
                    &bot_spec_b,
                    &bot_spec_w,
                    None,
                    None,
                    logger,
                    should_stop_func,
                    None,
                    Box::new(|| None),
                    after_initialization,
                    Box::new(|_board, _hist, _pla, _loc, _before, _after, _weights, _search| {}),
                );

                let should_continue = game_data.is_some();
                if let Some(game_data) = game_data {
                    if let Some(ref mut out) = sgf_out {
                        let _ = write_sgf_for_game(out, &game_data);
                    }

                    {
                        let mut time_used = time_used_by_bot.lock().map_err(|e| {
                            StringError::new(format!("Stats mutex poisoned: {}", e))
                        })?;
                        let mut moves = moves_by_bot.lock().map_err(|e| {
                            StringError::new(format!("Stats mutex poisoned: {}", e))
                        })?;
                        let mut wins = wins_by_bot.lock().map_err(|e| {
                            StringError::new(format!("Stats mutex poisoned: {}", e))
                        })?;
                        let mut results = game_results.lock().map_err(|e| {
                            StringError::new(format!("Stats mutex poisoned: {}", e))
                        })?;

                        let count = game_count.fetch_add(1, Ordering::AcqRel) + 1;
                        *time_used.entry(game_data.b_name.clone()).or_insert(0.0) +=
                            game_data.b_time_used;
                        *time_used.entry(game_data.w_name.clone()).or_insert(0.0) +=
                            game_data.w_time_used;
                        *moves.entry(game_data.b_name.clone()).or_insert(0.0) +=
                            game_data.b_move_count as f64;
                        *moves.entry(game_data.w_name.clone()).or_insert(0.0) +=
                            game_data.w_move_count as f64;

                        let winner = determine_winner_string(&game_data.end_hist);
                        if winner == "B" {
                            *wins.entry(game_data.b_name.clone()).or_insert(0) += 1;
                        } else if winner == "W" {
                            *wins.entry(game_data.w_name.clone()).or_insert(0) += 1;
                        }

                        results.push(GameResult {
                            b_name: game_data.b_name.clone(),
                            w_name: game_data.w_name.clone(),
                            winner: winner.to_string(),
                        });

                        let mut x = count;
                        while x % 2 == 0 && x > 1 {
                            x /= 2;
                        }
                        if x == 1 || x == 3 || x == 5 {
                            for (name, total_time) in time_used.iter() {
                                let move_count = moves.get(name).copied().unwrap_or(0.0);
                                if move_count > 0.0 {
                                    logger.write(&format!(
                                        "Avg move time used by {} {} {} moves",
                                        name,
                                        total_time / move_count,
                                        move_count
                                    ));
                                }
                            }
                        }
                    }
                }

                if should_stop.load(Ordering::Relaxed) {
                    break;
                }
                if !should_continue {
                    break;
                }
            }

            if let Some(mut out) = sgf_out {
                let _ = out.flush();
            }
            logger.write("Match loop thread terminating");
            Ok(())
        }));
    }

    let mut first_error: Option<StringError> = None;
    for handle in handles {
        match handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) if first_error.is_none() => first_error = Some(e),
            Err(_) if first_error.is_none() => {
                first_error = Some(StringError::new("Match loop thread panicked".to_string()));
            }
            _ => {}
        }
    }

    if let Some(e) = first_error {
        return Err(e);
    }

    print_summary(
        game_count.load(Ordering::Acquire),
        &wins_by_bot.lock().unwrap(),
        &game_results.lock().unwrap(),
        &bot_names,
    );

    for nn_eval in &nn_evals {
        logger.write(nn_eval.model_file_name());
        logger.write(&format!("NN rows: {}", nn_eval.num_rows_processed()));
        logger.write(&format!("NN batches: {}", nn_eval.num_batches_processed()));
        logger.write(&format!(
            "NN avg batch size: {}",
            nn_eval.average_processed_batch_size()
        ));
    }

    logger.write("All cleaned up, quitting");
    Ok(())
}

fn parse_args(args: &[String]) -> Result<MatchArgs, StringError> {
    MatchArgs::try_parse_from(std::iter::once(&"match".to_string()).chain(args.iter()))
        .map_err(|e| StringError::new(format!("Argument error: {}", e)))
}

fn get_logger_placeholder() -> Result<Logger, StringError> {
    Ok(Logger::new(
        LoggerOptions {
            log_to_stdout: false,
            log_to_stderr: false,
            log_time: false,
        },
        None,
    ))
}

fn compute_matchups(cfg: &ConfigParser, num_bots: i32) -> Result<Vec<(i32, i32)>, StringError> {
    let to_string_err = |e: kata_core::config::ConfigError| StringError::new(e.to_string());

    let mut include_bot = vec![false; num_bots as usize];
    if cfg.contains("includeBots") {
        let include_bot_idxs = cfg
            .get_ints("includeBots", 0, setup::MAX_BOT_PARAMS_FROM_CFG)
            .map_err(to_string_err)?;
        for i in 0..num_bots {
            if include_bot_idxs.contains(&i) {
                include_bot[i as usize] = true;
            }
        }
    } else {
        for i in 0..num_bots {
            include_bot[i as usize] = true;
        }
    }

    let mut secondary_bot_idxs = Vec::new();
    if cfg.contains("secondaryBots") {
        secondary_bot_idxs = cfg
            .get_ints("secondaryBots", 0, setup::MAX_BOT_PARAMS_FROM_CFG)
            .map_err(to_string_err)?;
    }
    for &idx in &secondary_bot_idxs {
        if idx < 0 || idx >= num_bots {
            return Err(StringError::new(format!(
                "secondaryBots value {} is out of range, numBots is {}",
                idx, num_bots
            )));
        }
    }

    let mut matchups = Vec::new();
    for i in 0..num_bots {
        if !include_bot[i as usize] {
            continue;
        }
        for j in 0..num_bots {
            if !include_bot[j as usize] {
                continue;
            }
            if i < j && !(secondary_bot_idxs.contains(&i) && secondary_bot_idxs.contains(&j)) {
                matchups.push((i, j));
                matchups.push((j, i));
            }
        }
    }

    if cfg.contains("extraPairs") {
        let pairs = cfg
            .get_non_negative_int_dashed_pairs("extraPairs", 0, num_bots - 1)
            .map_err(to_string_err)?;
        let one_sided = cfg.get_bool("extraPairsAreOneSidedBW").unwrap_or(false);
        for (p0, p1) in pairs {
            if one_sided {
                matchups.push((p0, p1));
            } else {
                matchups.push((p0, p1));
                matchups.push((p1, p0));
            }
        }
    }

    Ok(matchups)
}

fn load_bot_names_and_models(
    cfg: &ConfigParser,
    num_bots: i32,
) -> Result<(Vec<String>, Vec<String>), StringError> {
    let to_string_err = |e: kata_core::config::ConfigError| StringError::new(e.to_string());

    let mut bot_names = Vec::with_capacity(num_bots as usize);
    let mut nn_model_files = Vec::with_capacity(num_bots as usize);

    for i in 0..num_bots {
        let idx_str = int_to_string(i);

        let name = if cfg.contains(&("botName".to_string() + &idx_str)) {
            cfg.get_string(&("botName".to_string() + &idx_str))
                .map_err(to_string_err)?
        } else if num_bots == 1 {
            cfg.get_string("botName").map_err(to_string_err)?
        } else {
            return Err(StringError::new(
                "If more than one bot, must specify botName0, botName1,... individually"
                    .to_string(),
            ));
        };

        let model = if cfg.contains(&("nnModelFile".to_string() + &idx_str)) {
            cfg.get_string(&("nnModelFile".to_string() + &idx_str))
                .map_err(to_string_err)?
        } else {
            cfg.get_string("nnModelFile").map_err(to_string_err)?
        };

        bot_names.push(name);
        nn_model_files.push(model);
    }

    Ok((bot_names, nn_model_files))
}

fn determine_winner_string(end_hist: &kata_game::history::BoardHistory) -> &'static str {
    if end_hist.is_no_result {
        "None"
    } else if end_hist.winner == P_BLACK {
        "B"
    } else if end_hist.winner == P_WHITE {
        "W"
    } else {
        "None"
    }
}

fn write_sgf_for_game(
    out: &mut dyn Write,
    game_data: &kata_data::training::FinishedGameData,
) -> Result<(), StringError> {
    let hist = &game_data.end_hist;
    let x_size = hist.initial_board.x_size;
    let y_size = hist.initial_board.y_size;
    let komi = hist.rules.komi as f32 / 2.0;
    let result = format_sgf_result(hist);

    write!(
        out,
        "(;FF[4]GM[1]SZ[{}]KM[{}]PB[{}]PW[{}]RE[{}]",
        x_size, komi, game_data.b_name, game_data.w_name, result
    )
    .map_err(|e| StringError::new(format!("SGF write failed: {}", e)))?;

    for m in &hist.move_history {
        let color = if m.pla == P_BLACK { "B" } else { "W" };
        let coord = write_sgf_loc(m.loc, x_size, y_size)
            .map_err(|e| StringError::new(format!("SGF coord failed: {}", e)))?;
        write!(out, ";{}[{}]", color, coord)
            .map_err(|e| StringError::new(format!("SGF write failed: {}", e)))?;
    }

    writeln!(out, ")").map_err(|e| StringError::new(format!("SGF write failed: {}", e)))?;
    Ok(())
}

fn format_sgf_result(end_hist: &kata_game::history::BoardHistory) -> String {
    if end_hist.is_no_result {
        "Void".to_string()
    } else if end_hist.is_resignation {
        if end_hist.winner == P_BLACK {
            "B+R".to_string()
        } else {
            "W+R".to_string()
        }
    } else if end_hist.winner == C_EMPTY {
        "0".to_string()
    } else {
        let score = end_hist.final_white_minus_black_score.abs();
        if end_hist.winner == P_BLACK {
            format!("B+{}", score)
        } else {
            format!("W+{}", score)
        }
    }
}

fn print_summary(
    total_games: i64,
    wins_by_bot: &BTreeMap<String, i64>,
    game_results: &[GameResult],
    bot_names: &[String],
) {
    println!("Match finished: {} games played", total_games);
    if total_games == 0 {
        return;
    }

    println!("Wins by bot:");
    for name in bot_names {
        let wins = wins_by_bot.get(name).copied().unwrap_or(0);
        let losses = game_results
            .iter()
            .filter(|r| r.b_name == *name || r.w_name == *name)
            .count() as i64
            - wins;
        println!("  {}: {} wins, {} losses", name, wins, losses);
    }

    let mut head_to_head: BTreeMap<(String, String), (i64, i64)> = BTreeMap::new();
    for r in game_results {
        let key = (r.b_name.clone(), r.w_name.clone());
        let entry = head_to_head.entry(key).or_insert((0, 0));
        if r.winner == "B" {
            entry.0 += 1;
        } else if r.winner == "W" {
            entry.1 += 1;
        }
    }
    if !head_to_head.is_empty() {
        println!("Head-to-head (black wins / white wins):");
        for ((b, w), (bwins, wwins)) in head_to_head {
            println!("  {} vs {} (black): {} - {}", b, w, bwins, wwins);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn match_cfg(model0: &str, model1: &str, num_games: i64) -> String {
        format!(
            "koRules = SIMPLE,POSITIONAL\n\
             scoringRules = AREA\n\
             taxRules = NONE\n\
             multiStoneSuicideLegals = true,false\n\
             hasButtons = true,false\n\
             bSizes = 7\n\
             bSizeRelProbs = 1.0\n\
             komiMean = 7.5\n\
             numBots = 2\n\
             botName0 = bot0\n\
             botName1 = bot1\n\
             nnModelFile0 = {}\n\
             nnModelFile1 = {}\n\
             numGameThreads = 1\n\
             numGamesTotal = {}\n\
             logGamesEvery = 1\n\
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
            model0, model1, num_games
        )
    }

    fn unique_tmp_dir(prefix: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{}_{}_{}", prefix, std::process::id(), nanos))
    }

    #[test]
    fn test_parse_args() {
        let args = MatchArgs::parse_from([
            "match",
            "--config",
            "cfg.cfg",
            "--log-file",
            "log.log",
            "--sgf-output-dir",
            "sgfs",
        ]);
        assert_eq!(args.common.config, vec!["cfg.cfg"]);
        assert_eq!(args.log_file, Some("log.log".to_string()));
        assert_eq!(args.sgf_output_dir, Some("sgfs".to_string()));
    }

    #[test]
    fn test_compute_matchups_default() {
        let cfg = ConfigParser::from_str(
            "numBots = 2\nnumGamesTotal = 10\nlogGamesEvery = 1\n",
            false,
            false,
        )
        .unwrap();
        let matchups = compute_matchups(&cfg, 2).unwrap();
        assert_eq!(matchups, vec![(0, 1), (1, 0)]);
    }

    #[test]
    fn test_compute_matchups_include_and_extra() {
        let cfg = ConfigParser::from_str(
            "numBots = 3\nincludeBots = 0,2\nextraPairs = 0-2\nextraPairsAreOneSidedBW = true\nnumGamesTotal = 10\nlogGamesEvery = 1\n",
            false,
            false,
        )
        .unwrap();
        let matchups = compute_matchups(&cfg, 3).unwrap();
        // Bots 0 and 2 included produce (0,2) and (2,0); the one-sided extra
        // pair adds another (0,2).
        assert_eq!(matchups, vec![(0, 2), (2, 0), (0, 2)]);
    }

    #[test]
    fn test_match_runs_and_reports() {
        let tmp = unique_tmp_dir("katago_match_run");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let model0 = tmp.join("b.bin.gz");
        let model1 = tmp.join("w.bin.gz");
        std::fs::write(&model0, b"").unwrap();
        std::fs::write(&model1, b"").unwrap();

        let cfg_path = tmp.join("match.cfg");
        let cfg_text = match_cfg(model0.to_str().unwrap(), model1.to_str().unwrap(), 2);
        std::fs::write(&cfg_path, cfg_text).unwrap();

        let sgf_dir = tmp.join("sgfs");

        let args = vec![
            "--config".to_string(),
            cfg_path.to_str().unwrap().to_string(),
            "--sgf-output-dir".to_string(),
            sgf_dir.to_str().unwrap().to_string(),
        ];
        match_impl(&args).expect("match should run successfully");

        assert!(sgf_dir.exists(), "SGF output dir should exist");
        let sgf_files: Vec<_> = std::fs::read_dir(&sgf_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert!(
            !sgf_files.is_empty(),
            "SGF output dir should contain at least one file"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
