//! Run a search on a position from an SGF file, for debugging.
//!
//! Corresponds to `MainCmds::evalsgf` in `cpp/command/evalsgf.cpp`.

#![allow(
    clippy::collapsible_if,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    dead_code
)]

use std::collections::HashMap;
use std::io::Write;

use clap::Parser;

use kata_core::config::ConfigParser;
use kata_core::global::{self, StringError};
use kata_core::logger::{Logger, LoggerOptions};
use kata_core::rng::Rand;
use kata_core::time::timer::ClockTimer;
use kata_data::numpy::{NumpyBuffer, ZipFile};
use kata_data::sgf::CompactSgf;
use kata_game::board::{
    Board, C_EMPTY, MAX_ARR_SIZE, Move, P_BLACK, P_WHITE, PASS_LOC, Player, get_opp, location,
    player_io,
};
use kata_game::history::BoardHistory;
use kata_game::rules::Rules;
use kata_nn::backend::NNResultBuf;
use kata_nn::eval::NnEvaluator;
use kata_nn::inputs::{MiscNNInputParams, fill_row_v7, nn_pos};
use kata_nn::version;
use kata_program::setup::{self, SetupFor};
use kata_search::async_bot::AsyncBot;
use kata_search::node::SearchNode;
use kata_search::params::SearchParams;
use kata_search::search::{PrintTreeOptions, Search};
use kata_search::time_control::TimeControls;

use crate::cli::CommonArgs;

const MAX_VISITS_DEFAULT: i64 = 10;

/// CLI arguments for the `evalsgf` subcommand.
#[derive(Parser, Debug, Clone)]
struct EvalSgfArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// SGF file to analyze.
    sgf_file: String,

    /// SGF move number to analyze.
    #[arg(short = 'm', long = "move-num", value_name = "MOVENUM")]
    move_num: i32,

    /// End SGF move number range to analyze, inclusive.
    #[arg(long = "move-num-end", value_name = "MOVENUM")]
    move_num_end: Option<i32>,

    /// Move branch in search tree to print.
    #[arg(long = "print-branch", alias = "print", value_name = "MOVE MOVE ...")]
    print_branch: Option<String>,

    /// Extra moves to force-play before doing search.
    #[arg(long = "extra-moves", alias = "extra", value_name = "MOVE MOVE ...")]
    extra_moves: Option<String>,

    /// Avoid moves in search.
    #[arg(long = "avoid-moves", value_name = "MOVE MOVE ...")]
    avoid_moves: Option<String>,

    /// Hint loc.
    #[arg(long = "hint-loc", value_name = "MOVE")]
    hint_loc: Option<String>,

    /// Set the number of visits.
    #[arg(short = 'v', long = "visits", value_name = "VISITS")]
    visits: Option<i64>,

    /// Set the number of threads.
    #[arg(short = 't', long = "threads", value_name = "THREADS")]
    threads: Option<i32>,

    /// Artificially set komi.
    #[arg(long = "override-komi", value_name = "KOMI")]
    override_komi: Option<f32>,

    /// Artificially set rules.
    #[arg(long = "override-rules", value_name = "RULES")]
    override_rules: Option<String>,

    /// Print ownership.
    #[arg(long = "print-ownership")]
    print_ownership: bool,

    /// Print root nn values.
    #[arg(long = "print-root-nn-values")]
    print_root_nn_values: bool,

    /// Print policy.
    #[arg(long = "print-policy")]
    print_policy: bool,

    /// Print log policy.
    #[arg(long = "print-log-policy")]
    print_log_policy: bool,

    /// Print dirichlet shape.
    #[arg(long = "print-dirichlet-shape")]
    print_dirichlet_shape: bool,

    /// Print score now.
    #[arg(long = "print-score-now")]
    print_score_now: bool,

    /// Print root ending bonus now.
    #[arg(long = "print-root-ending-bonus")]
    print_root_ending_bonus: bool,

    /// Compute and print lead.
    #[arg(long = "print-lead")]
    print_lead: bool,

    /// Compute and print avgShorttermError.
    #[arg(long = "print-avg-shortterm-error")]
    print_avg_shortterm_error: bool,

    /// Compute and print sharp weighted score.
    #[arg(long = "print-sharp-score")]
    print_sharp_score: bool,

    /// Print graph structure of the search.
    #[arg(long = "print-graph")]
    print_graph: bool,

    /// Print analysis json of the search.
    #[arg(long = "print-json")]
    print_json: bool,

    /// How deep to print.
    #[arg(long = "print-max-depth", value_name = "DEPTH", default_value_t = 1)]
    print_max_depth: i32,

    /// Perform single raw neural net eval.
    #[arg(long = "raw-nn")]
    raw_nn: bool,

    /// Dump the nn input tensor to npz file.
    #[arg(long = "dump-npz-input-to", value_name = "NPZFILE")]
    dump_npz_input_to: Option<String>,
}

/// Public CLI entry point.
pub fn evalsgf(args: &[String]) -> i32 {
    let mut out = std::io::stdout();
    match evalsgf_impl(args, &mut out) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {}", e);
            1
        }
    }
}

/// Internal testable entry point.
fn evalsgf_impl(args: &[String], out: &mut dyn Write) -> Result<(), StringError> {
    let parsed =
        EvalSgfArgs::try_parse_from(std::iter::once(&"evalsgf".to_string()).chain(args.iter()))
            .map_err(|e| StringError::new(format!("Argument error: {}", e)))?;

    let cfg = if parsed.common.config.is_empty() {
        let mut cfg = minimal_evalsgf_config();
        parsed.common.maybe_apply_override_config_arg(&mut cfg)?;
        cfg
    } else {
        parsed.common.get_config("gtp_example.cfg")?
    };

    let model_file = if parsed.common.config.is_empty() {
        "/dev/null".to_string()
    } else {
        parsed.common.get_model_file()?
    };
    let human_model_file = parsed.common.get_human_model_file();

    let log_to_stdout_default = true;
    let logger: &'static Logger = Box::leak(Box::new(Logger::new(
        LoggerOptions {
            log_to_stdout: log_to_stdout_default,
            log_to_stderr: true,
            log_time: true,
        },
        None,
    )));

    let default_rules = Rules::get_tromp_taylorish();
    let perspective = setup::parse_report_analysis_winrates(&cfg, P_BLACK)
        .map_err(|e| StringError::new(format!("Could not parse perspective: {}", e)))?;

    let sgf = CompactSgf::parse_file(&parsed.sgf_file)
        .map_err(|e| StringError::new(format!("Could not load SGF file: {}", e.0)))?;

    let mut board = Board::default();
    let mut next_pla = P_BLACK;
    let mut hist = BoardHistory::default();

    let extra_moves_str = parsed.extra_moves.unwrap_or_default();

    let set_up_board_using_rules = |board: &mut Board,
                                    next_pla: &mut Player,
                                    hist: &mut BoardHistory,
                                    initial_rules: &Rules,
                                    move_num: i32|
     -> Result<(), StringError> {
        setup_initial_board_and_hist(&sgf, initial_rules, board, next_pla, hist)?;

        if let Some(komi) = parsed.override_komi {
            if komi > board.x_size as f32 * board.y_size as f32 + nn_pos::KOMI_CLIP_RADIUS
                || komi < -(board.x_size as f32 * board.y_size as f32) - nn_pos::KOMI_CLIP_RADIUS
            {
                return Err(StringError::new(
                    "Invalid komi, too much greater than the area of the board".to_string(),
                ));
            }
            hist.set_komi(komi);
        }

        if move_num < 0 {
            return Err(StringError::new(format!(
                "Move num {} requested but must be non-negative",
                move_num
            )));
        }
        if move_num > sgf.moves.len() as i32 {
            return Err(StringError::new(format!(
                "Move num {} requested but sgf has only {}",
                move_num,
                sgf.moves.len()
            )));
        }

        for i in 0..move_num as usize {
            let m = sgf.moves[i];
            hist.make_board_move_tolerant(board, m.loc, m.pla);
            // Tolerant play ignores illegal moves; update next_pla based on the move color.
            *next_pla = get_opp(m.pla);
        }

        if !extra_moves_str.is_empty() {
            let extra_locs = location::parse_sequence(&extra_moves_str, board.x_size, board.y_size)
                .map_err(|e| StringError::new(format!("Could not parse extra moves: {}", e.0)))?;
            for loc in extra_locs {
                if !hist.is_legal(board, loc, *next_pla) {
                    eprintln!("{}", board.to_string_simple('\n'));
                    eprintln!(
                        "Extra illegal move for {}: {}",
                        player_io::color_to_char(*next_pla),
                        location::to_string(loc, board.x_size, board.y_size)
                    );
                    return Err(StringError::new("Illegal extra move".to_string()));
                }
                hist.make_board_move_assume_legal(board, loc, *next_pla);
                *next_pla = get_opp(*next_pla);
            }
        }

        Ok(())
    };

    let mut initial_rules = sgf
        .get_rules_or_fail_allow_unspecified(&default_rules)
        .map_err(|e| StringError::new(format!("Could not parse SGF rules: {}", e.0)))?;
    if let Some(override_rules) = &parsed.override_rules {
        initial_rules = Rules::parse_rules(override_rules)
            .map_err(|e| StringError::new(format!("Could not parse override rules: {}", e.0)))?;
    }

    // Set up once now for error catching.
    set_up_board_using_rules(
        &mut board,
        &mut next_pla,
        &mut hist,
        &initial_rules,
        parsed.move_num,
    )?;

    let mut options = PrintTreeOptions::new().max_depth(parsed.print_max_depth);
    if let Some(branch) = &parsed.print_branch {
        options = options.only_branch(&board, branch);
    }
    options.print_avg_shortterm_error = parsed.print_avg_shortterm_error;

    logger.write("Engine starting...");

    let has_human_model = human_model_file.is_some();
    let mut params = setup::load_single_params_with_human(&cfg, SetupFor::Gtp, has_human_model)
        .map_err(|e| StringError::new(format!("Could not load params: {}", e)))?;

    if let Some(max_visits) = parsed.visits {
        if max_visits < -1 || max_visits == 0 {
            return Err(StringError::new("maxVisits: invalid value".to_string()));
        }
        if max_visits == -1 {
            logger.write(&format!(
                "No max visits specified on cmdline, using defaults in {}",
                cfg.file_name()
            ));
        } else {
            params.max_visits = max_visits;
            params.max_playouts = max_visits;
        }
    }

    if let Some(num_threads) = parsed.threads {
        if num_threads < -1 || num_threads == 0 {
            return Err(StringError::new("numThreads: invalid value".to_string()));
        }
        if num_threads == -1 {
            logger.write(&format!(
                "No num threads specified on cmdline, using defaults in {}",
                cfg.file_name()
            ));
        } else {
            params.num_threads = num_threads;
        }
    }

    let mut seed_rand = Rand::new();
    let search_rand_seed = if cfg.contains("searchRandSeed") {
        cfg.get_string("searchRandSeed").unwrap_or_default()
    } else {
        global::uint64_to_string(seed_rand.next_u64())
    };

    setup::initialize_session(&cfg);
    let expected_concurrent_evals = params.num_threads;
    let default_max_batch_size = std::cmp::max(8, ((params.num_threads + 3) / 4) * 4);
    let default_require_exact_nn_len = true;
    let disable_fp16 = false;
    let expected_sha256 = String::new();

    let nn_eval = setup::initialize_nn_evaluator(
        model_file.clone(),
        model_file.clone(),
        expected_sha256.clone(),
        &cfg,
        logger,
        &mut seed_rand,
        expected_concurrent_evals,
        board.x_size,
        board.y_size,
        default_max_batch_size,
        default_require_exact_nn_len,
        disable_fp16,
        SetupFor::Gtp,
    )
    .map_err(|e| StringError::new(format!("Could not initialize neural net: {}", e)))?;

    let mut human_eval: Option<NnEvaluator> = None;
    if let Some(human_model_file) = &human_model_file {
        human_eval = Some(
            setup::initialize_nn_evaluator(
                human_model_file.clone(),
                human_model_file.clone(),
                expected_sha256,
                &cfg,
                logger,
                &mut seed_rand,
                expected_concurrent_evals,
                board.x_size,
                board.y_size,
                default_max_batch_size,
                default_require_exact_nn_len,
                disable_fp16,
                SetupFor::Gtp,
            )
            .map_err(|e| StringError::new(format!("Could not initialize human model: {}", e)))?,
        );
    }

    logger.write("Loaded neural net");

    {
        let mut supported = false;
        let supported_rules = nn_eval.supported_rules(initial_rules, &mut supported);
        if !supported {
            writeln!(
                out,
                "Warning: Rules {} from sgf not supported by neural net, using {} instead",
                initial_rules, supported_rules
            )
            .map_err(|e| StringError::new(e.to_string()))?;
            initial_rules = supported_rules;
        }
    }

    let nn_eval: &'static NnEvaluator = Box::leak(Box::new(nn_eval));
    let human_eval: Option<&'static NnEvaluator> =
        human_eval.map(|e| Box::leak(Box::new(e)) as &'static NnEvaluator);

    let move_num_start = parsed.move_num;
    let mut move_num_end = parsed.move_num_end.unwrap_or(-1);
    if move_num_end < move_num_start {
        move_num_end = move_num_start;
    }

    for move_num in move_num_start..=move_num_end {
        set_up_board_using_rules(
            &mut board,
            &mut next_pla,
            &mut hist,
            &initial_rules,
            move_num,
        )?;

        cfg.warn_unused_keys(&mut std::io::stderr(), Some(logger))
            .map_err(|e| StringError::new(e.to_string()))?;
        {
            let mut warn_out = Vec::new();
            setup::maybe_warn_human_sl_params(
                &params,
                Some(nn_eval),
                human_eval,
                &mut warn_out,
                Some(logger),
            )
            .map_err(|e| StringError::new(format!("Human SL param warning failed: {}", e)))?;
            if !warn_out.is_empty() {
                out.write_all(&warn_out)
                    .map_err(|e| StringError::new(e.to_string()))?;
            }
        }

        if parsed.raw_nn {
            let mut buf = NNResultBuf::new();
            let nn_input_params = MiscNNInputParams {
                draw_equivalent_wins_for_white: params.draw_equivalent_wins_for_white,
                ..MiscNNInputParams::default()
            };
            nn_eval.evaluate(
                &board,
                &hist,
                next_pla,
                &nn_input_params,
                &mut buf,
                true,
                true,
            );

            writeln!(out, "Rules: {}", hist.rules).map_err(string_error)?;
            writeln!(out, "Encore phase {}", hist.encore_phase).map_err(string_error)?;
            writeln!(out, "{}", board_to_string(&board, None)).map_err(string_error)?;
            if let Some(output) = buf.result.as_ref() {
                debug_print_nn_output(out, output, &board)?;
            }

            if let Some(human_eval) = human_eval {
                let mut buf = NNResultBuf::new();
                human_eval.evaluate(
                    &board,
                    &hist,
                    next_pla,
                    &nn_input_params,
                    &mut buf,
                    true,
                    true,
                );
                if let Some(output) = buf.result.as_ref() {
                    debug_print_nn_output(out, output, &board)?;
                }
            }
            continue;
        }

        let mut bot = AsyncBot::new_with_human(
            params.clone(),
            nn_eval,
            human_eval,
            logger,
            &search_rand_seed,
        );
        bot.set_position(next_pla, &board, &hist);
        if let Some(hint) = &parsed.hint_loc {
            let loc = location::of_string(hint, board.x_size, board.y_size)
                .map_err(|e| StringError::new(format!("Could not parse hint loc: {}", e.0)))?;
            bot.set_root_hint_loc(loc);
        }

        if let Some(avoid) = &parsed.avoid_moves {
            let avoid_locs = location::parse_sequence(avoid, board.x_size, board.y_size)
                .map_err(|e| StringError::new(format!("Could not parse avoid moves: {}", e.0)))?;
            let mut avoid_black = vec![0; MAX_ARR_SIZE];
            let mut avoid_white = vec![0; MAX_ARR_SIZE];
            for loc in avoid_locs {
                avoid_black[loc as usize] = 1;
                avoid_white[loc as usize] = 1;
            }
            bot.set_avoid_move_until_by_loc(&avoid_black, &avoid_white);
        }

        let root_pla = bot.get_search_stop_and_wait().root_pla;
        let mut sout = String::new();
        sout.push_str(&format!("Rules: {}\n", hist.rules));
        sout.push_str(&format!("Encore phase {}\n", hist.encore_phase));
        sout.push_str(&board_to_string(&board, Some(&hist.move_history)));

        if !options.branch.is_empty() {
            let mut copy = board.clone();
            let mut copy_hist = hist.clone();
            let mut pla = next_pla;
            for &loc in &options.branch {
                if !copy_hist.is_legal(&copy, loc, pla) {
                    eprintln!("{}", board.to_string_simple('\n'));
                    eprintln!(
                        "Branch Illegal move for {}: {}",
                        player_io::color_to_char(pla),
                        location::to_string(loc, board.x_size, board.y_size)
                    );
                    return Err(StringError::new("Illegal branch move".to_string()));
                }
                copy_hist.make_board_move_assume_legal(&mut copy, loc, pla);
                pla = get_opp(pla);
            }
            sout.push_str(&board_to_string(&copy, Some(&copy_hist.move_history)));
        }

        sout.push('\n');
        logger.write(&sout);
        sout.clear();

        let timer = ClockTimer::new();
        nn_eval.clear_stats();
        if let Some(human_eval) = human_eval {
            human_eval.clear_stats();
        }
        let _loc = bot.gen_move_synchronous(root_pla, &TimeControls::new());

        let search = bot.get_search_stop_and_wait();

        if parsed.print_ownership {
            sout.push_str("Ownership map (ROOT position):\n");
            search.print_root_ownership_map(&mut sout, perspective);
        }

        if parsed.print_root_nn_values {
            if let Some(nn_output) = search.root_node.as_ref().and_then(|n| n.get_nn_output()) {
                writeln!(out, "White win: {}", nn_output.white_win_prob).map_err(string_error)?;
                writeln!(out, "White loss: {}", nn_output.white_loss_prob).map_err(string_error)?;
                writeln!(out, "White noresult: {}", nn_output.white_no_result_prob)
                    .map_err(string_error)?;
                writeln!(out, "White score mean {}", nn_output.white_score_mean)
                    .map_err(string_error)?;
                let mean = nn_output.white_score_mean as f64;
                let mean_sq = nn_output.white_score_mean_sq as f64;
                let stdev = (mean_sq - mean * mean).max(0.0).sqrt();
                writeln!(out, "White score stdev {}", stdev).map_err(string_error)?;
                writeln!(out, "Var time left {}", nn_output.var_time_left).map_err(string_error)?;
                writeln!(
                    out,
                    "Shortterm winloss error {}",
                    nn_output.shortterm_winloss_error
                )
                .map_err(string_error)?;
                writeln!(
                    out,
                    "Shortterm score error {}",
                    nn_output.shortterm_score_error
                )
                .map_err(string_error)?;
            }
        }

        if parsed.print_sharp_score {
            if let Some(root) = search.root_node.as_ref() {
                let mut ret = 0.0;
                let suc = search.get_sharp_score(root, &mut ret);
                if suc {
                    writeln!(out, "White sharp score {}", ret).map_err(string_error)?;
                }
            }
        }

        if parsed.print_policy {
            if let Some(nn_output) = search.root_node.as_ref().and_then(|n| n.get_nn_output()) {
                writeln!(out, "Root policy: ").map_err(string_error)?;
                print_policy_map(
                    out,
                    &board,
                    nn_output.get_policy_probs_maybe_noised(),
                    nn_output.nn_x_len,
                    nn_output.nn_y_len,
                    false,
                )?;
            }
            if let Some(human_output) = search.root_node.as_ref().and_then(|n| n.get_human_output())
            {
                writeln!(out, "Root human policy: ").map_err(string_error)?;
                print_policy_map(
                    out,
                    &board,
                    human_output.get_policy_probs_maybe_noised(),
                    human_output.nn_x_len,
                    human_output.nn_y_len,
                    false,
                )?;
            }
        }

        if parsed.print_log_policy {
            if let Some(nn_output) = search.root_node.as_ref().and_then(|n| n.get_nn_output()) {
                writeln!(out, "Root policy: ").map_err(string_error)?;
                print_policy_map(
                    out,
                    &board,
                    nn_output.get_policy_probs_maybe_noised(),
                    nn_output.nn_x_len,
                    nn_output.nn_y_len,
                    true,
                )?;
            }
            if let Some(human_output) = search.root_node.as_ref().and_then(|n| n.get_human_output())
            {
                writeln!(out, "Root human policy: ").map_err(string_error)?;
                print_policy_map(
                    out,
                    &board,
                    human_output.get_policy_probs_maybe_noised(),
                    human_output.nn_x_len,
                    human_output.nn_y_len,
                    true,
                )?;
            }
        }

        if parsed.print_dirichlet_shape {
            if let Some(nn_output) = search.root_node.as_ref().and_then(|n| n.get_nn_output()) {
                let policy_size = nn_output.nn_x_len * nn_output.nn_y_len;
                let mut alpha_distr = [0.0f64; nn_pos::MAX_NN_POLICY_SIZE];
                Search::compute_dirichlet_alpha_distribution(
                    policy_size,
                    nn_output.get_policy_probs_maybe_noised(),
                    &mut alpha_distr,
                );
                writeln!(out, "Dirichlet alphas with 10.83 total concentration: ")
                    .map_err(string_error)?;
                print_alpha_map(
                    out,
                    &board,
                    &alpha_distr,
                    nn_output.nn_x_len,
                    nn_output.nn_y_len,
                )?;
            }
        }

        if parsed.print_score_now {
            sout.push_str("Score now (ROOT position):\n");
            let mut copy_hist = hist.clone();
            copy_hist.end_and_score_game_now(&board);
            let mut area = [C_EMPTY; MAX_ARR_SIZE];
            board.calculate_area(
                &mut area,
                true,
                true,
                true,
                hist.rules.multi_stone_suicide_legal,
            );
            for y in 0..board.y_size {
                for x in 0..board.x_size {
                    let l = location::get_loc(x, y, board.x_size) as usize;
                    sout.push(player_io::color_to_char(area[l]));
                }
                sout.push('\n');
            }
            sout.push('\n');
            sout.push_str(&format!("Komi: {}\n", copy_hist.rules.komi_f32()));
            sout.push_str(&format!("WBonus: {}\n", copy_hist.white_bonus_score));
            sout.push_str(&format!("Final: {}\n", game_result_string(&copy_hist)));
        }

        if parsed.print_root_ending_bonus {
            sout.push_str("Ending bonus (ROOT position)\n");
            search.print_root_ending_score_value_bonus(&mut sout);
        }

        sout.push_str(&format!("Time taken: {}\n", timer.get_seconds()));
        sout.push_str(&format!("Root visits: {}\n", search.get_root_visits()));
        sout.push_str(&format!("NN rows: {}\n", nn_eval.num_rows_processed()));
        sout.push_str(&format!(
            "NN batches: {}\n",
            nn_eval.num_batches_processed()
        ));
        sout.push_str(&format!(
            "NN avg batch size: {}\n",
            nn_eval.average_processed_batch_size()
        ));

        let node_count = {
            let temp_nodes = search.enumerate_tree_post_order();
            temp_nodes.len()
        };
        sout.push_str(&format!("True number of tree nodes: {}\n", node_count));
        sout.push_str("PV: ");
        search.print_pv(&mut sout, search.root_node.as_deref(), 25);
        sout.push('\n');
        sout.push_str("Tree:\n");
        search.print_tree(
            &mut sout,
            search.root_node.as_deref(),
            &options,
            perspective,
        );
        logger.write(&sout);
        sout.clear();

        if parsed.print_lead {
            // The Rust stub for compute_lead always returns 0, matching the current port.
            writeln!(out, "LEAD: {}", 0.0).map_err(string_error)?;
        }

        if parsed.print_json {
            let mut ret = serde_json::Value::Object(serde_json::Map::new());
            let suc = search.get_analysis_json(
                perspective,
                7,
                false,
                parsed.print_policy,
                parsed.print_ownership,
                false,
                false,
                false,
                true,
                false,
                &mut ret,
            );
            if suc {
                writeln!(out, "{}", ret).map_err(string_error)?;
            }
        }

        if let Some(dump_path) = &parsed.dump_npz_input_to {
            dump_nn_input_to_npz(out, &board, &hist, next_pla, &params, nn_eval, dump_path)?;
        }

        let nodes = bot.get_search_stop_and_wait().enumerate_tree_post_order();
        if parsed.print_graph {
            print_graph(out, &nodes)?;
        }

        bot.clear_search();
    }

    Ok(())
}

fn minimal_evalsgf_config() -> ConfigParser {
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

fn string_error<E: std::fmt::Display>(e: E) -> StringError {
    StringError::new(e.to_string())
}

fn setup_initial_board_and_hist(
    sgf: &CompactSgf,
    rules: &Rules,
    board: &mut Board,
    next_pla: &mut Player,
    hist: &mut BoardHistory,
) -> Result<(), StringError> {
    *board = Board::new(sgf.x_size, sgf.y_size);
    *next_pla = sgf.root_node.get_pl_specified_color();
    if *next_pla != P_BLACK && *next_pla != P_WHITE {
        *next_pla = P_BLACK;
    }
    for placement in &sgf.placements {
        let color = placement.pla;
        board.set_stone(placement.loc, color);
    }
    *hist = BoardHistory::new(board.clone(), *next_pla, *rules, 0);
    Ok(())
}

fn board_to_string(board: &Board, _move_history: Option<&[Move]>) -> String {
    // A compact board representation with coordinates.
    let mut s = String::new();
    s.push_str("   ");
    for x in 0..board.x_size {
        s.push(column_letter(x));
        s.push(' ');
    }
    s.push('\n');
    for y in 0..board.y_size {
        let row_label = board.y_size - y;
        s.push_str(&format!("{:2} ", row_label));
        for x in 0..board.x_size {
            let loc = location::get_loc(x, y, board.x_size) as usize;
            s.push(player_io::color_to_char(board.colors[loc]));
            s.push(' ');
        }
        s.push_str(&format!("{:2}\n", row_label));
    }
    s.push_str("   ");
    for x in 0..board.x_size {
        s.push(column_letter(x));
        s.push(' ');
    }
    s.push('\n');
    s
}

fn column_letter(x: i32) -> char {
    // GTP columns skip 'I'.
    let mut c = x;
    if c >= 8 {
        c += 1;
    }
    (b'A' + c as u8) as char
}

fn print_policy_map(
    out: &mut dyn Write,
    board: &Board,
    policy_probs: &[f32],
    nn_x_len: i32,
    nn_y_len: i32,
    log_scale: bool,
) -> Result<(), StringError> {
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let pos = nn_pos::xy_to_pos(x, y, nn_x_len);
            let prob = policy_probs[pos as usize];
            if prob < 0.0 {
                write!(out, "  -   ").map_err(string_error)?;
            } else {
                let value = if log_scale {
                    prob.ln() as f64
                } else {
                    prob as f64 * 100.0
                };
                write!(out, "{:6.2} ", value).map_err(string_error)?;
            }
        }
        writeln!(out).map_err(string_error)?;
    }
    let pass_pos = nn_pos::loc_to_pos(PASS_LOC, board.x_size, nn_x_len, nn_y_len);
    let pass_prob = policy_probs[pass_pos as usize];
    if pass_prob < 0.0 {
        writeln!(out, "Pass   -").map_err(string_error)?;
    } else {
        let value = if log_scale {
            pass_prob.ln() as f64
        } else {
            pass_prob as f64 * 100.0
        };
        writeln!(out, "Pass {:6.2}", value).map_err(string_error)?;
    }
    Ok(())
}

fn print_alpha_map(
    out: &mut dyn Write,
    board: &Board,
    alpha_distr: &[f64],
    nn_x_len: i32,
    nn_y_len: i32,
) -> Result<(), StringError> {
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let pos = nn_pos::xy_to_pos(x, y, nn_x_len);
            let alpha = alpha_distr[pos as usize];
            if alpha < 0.0 {
                write!(out, "  -    ").map_err(string_error)?;
            } else {
                write!(out, "{:7.4} ", alpha * 10.83).map_err(string_error)?;
            }
        }
        writeln!(out).map_err(string_error)?;
    }
    let pass_pos = nn_pos::loc_to_pos(PASS_LOC, board.x_size, nn_x_len, nn_y_len);
    let alpha = alpha_distr[pass_pos as usize];
    if alpha < 0.0 {
        writeln!(out, "Pass   -").map_err(string_error)?;
    } else {
        writeln!(out, "Pass {:7.2}", alpha * 10.83).map_err(string_error)?;
    }
    Ok(())
}

fn game_result_string(hist: &BoardHistory) -> String {
    if hist.is_resignation {
        format!("{}+R", if hist.winner == P_BLACK { "B" } else { "W" })
    } else if hist.is_no_result {
        "Void".to_string()
    } else {
        let score = hist.final_white_minus_black_score;
        if score > 0.0 {
            format!("W+{:.1}", score)
        } else if score < 0.0 {
            format!("B+{:.1}", -score)
        } else {
            "0".to_string()
        }
    }
}

fn print_graph(out: &mut dyn Write, nodes: &[*mut SearchNode]) -> Result<(), StringError> {
    let mut ordered = nodes.to_vec();
    ordered.reverse();
    let mut idx_of_node: HashMap<usize, usize> = HashMap::new();
    for (node_idx, &ptr) in ordered.iter().enumerate() {
        idx_of_node.insert(ptr as usize, node_idx);
    }

    for (node_idx, &ptr) in ordered.iter().enumerate() {
        // Safety: the search is stopped and no other thread accesses the tree.
        let node = unsafe { &*ptr };
        let children = node.get_children();
        let capacity = children.get_capacity();
        for i in 0..capacity {
            let child_ptr = children.get(i).get_raw_ptr();
            if child_ptr.is_null() {
                break;
            }
            if let Some(&child_idx) = idx_of_node.get(&(child_ptr as usize)) {
                writeln!(out, "{} -> {}", node_idx, child_idx).map_err(string_error)?;
            }
        }
    }
    writeln!(out).map_err(string_error)
}

fn debug_print_nn_output(
    out: &mut dyn Write,
    output: &kata_nn::inputs::NNOutput,
    board: &Board,
) -> Result<(), StringError> {
    writeln!(
        out,
        "nnHash {} nnXLen {} nnYLen {}",
        output.nn_hash, output.nn_x_len, output.nn_y_len
    )
    .map_err(string_error)?;
    writeln!(
        out,
        "whiteWinProb {} whiteLossProb {} whiteNoResultProb {}",
        output.white_win_prob, output.white_loss_prob, output.white_no_result_prob
    )
    .map_err(string_error)?;
    writeln!(
        out,
        "whiteScoreMean {} whiteScoreMeanSq {} whiteLead {}",
        output.white_score_mean, output.white_score_mean_sq, output.white_lead
    )
    .map_err(string_error)?;
    writeln!(
        out,
        "policyOptimismUsed {} varTimeLeft {} shorttermWinlossError {} shorttermScoreError {}",
        output.policy_optimism_used,
        output.var_time_left,
        output.shortterm_winloss_error,
        output.shortterm_score_error
    )
    .map_err(string_error)?;
    writeln!(out, "Policy map:").map_err(string_error)?;
    print_policy_map(
        out,
        board,
        output.get_policy_probs_maybe_noised(),
        output.nn_x_len,
        output.nn_y_len,
        false,
    )
}

fn dump_nn_input_to_npz(
    out: &mut dyn Write,
    board: &Board,
    hist: &BoardHistory,
    next_pla: Player,
    params: &SearchParams,
    nn_eval: &NnEvaluator,
    dump_path: &str,
) -> Result<(), StringError> {
    let inputs_use_nhwc = false;
    let nn_x_len = nn_eval.nn_x_len();
    let nn_y_len = nn_eval.nn_y_len();
    let model_version = nn_eval.model_version();
    let num_spatial_features = version::get_num_spatial_features(model_version).map_err(|e| {
        StringError::new(format!(
            "Model version {} does not support NN input dumping: {}",
            model_version, e
        ))
    })?;
    let num_global_features = version::get_num_global_features(model_version).map_err(|e| {
        StringError::new(format!(
            "Model version {} does not support NN input dumping: {}",
            model_version, e
        ))
    })?;

    let mut binary_input_nchw = NumpyBuffer::<f32>::new(vec![
        1,
        num_spatial_features as i64,
        nn_x_len as i64,
        nn_y_len as i64,
    ])
    .map_err(|e| StringError::new(format!("Could not create binary input buffer: {}", e.0)))?;
    let mut global_input_nc = NumpyBuffer::<f32>::new(vec![1, num_global_features as i64])
        .map_err(|e| StringError::new(format!("Could not create global input buffer: {}", e.0)))?;

    let nn_input_params = MiscNNInputParams {
        symmetry: 0,
        policy_optimism: params.root_policy_optimism,
        ..MiscNNInputParams::default()
    };

    let header_elems = kata_data::numpy::TOTAL_HEADER_BYTES / std::mem::size_of::<f32>();
    fill_row_v7(
        board,
        hist,
        next_pla,
        &nn_input_params,
        nn_x_len,
        nn_y_len,
        inputs_use_nhwc,
        &mut binary_input_nchw.data[header_elems..],
        &mut global_input_nc.data[header_elems..],
    );

    let mut zip_file = ZipFile::new(dump_path)
        .map_err(|e| StringError::new(format!("Could not create npz file: {}", e.0)))?;
    let binary_bytes = binary_input_nchw
        .prepare_header_with_num_rows(1)
        .map_err(|e| StringError::new(format!("Could not prepare binary input header: {}", e.0)))?;
    zip_file
        .write_buffer("binaryInputNCHW", &binary_bytes)
        .map_err(|e| StringError::new(format!("Could not write binary input: {}", e.0)))?;
    let global_bytes = global_input_nc
        .prepare_header_with_num_rows(1)
        .map_err(|e| StringError::new(format!("Could not prepare global input header: {}", e.0)))?;
    zip_file
        .write_buffer("globalInputNC", &global_bytes)
        .map_err(|e| StringError::new(format!("Could not write global input: {}", e.0)))?;
    zip_file
        .close()
        .map_err(|e| StringError::new(format!("Could not close npz file: {}", e.0)))?;
    writeln!(out, "Wrote to {}", dump_path).map_err(string_error)?;

    let mut buf = NNResultBuf::new();
    nn_eval.evaluate(
        board,
        hist,
        next_pla,
        &nn_input_params,
        &mut buf,
        true,
        true,
    );
    if let Some(output) = buf.result.as_ref() {
        debug_print_nn_output(out, output, board)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evalsgf_arg_parsing() {
        let args = EvalSgfArgs::parse_from([
            "evalsgf",
            "--model",
            "/dev/null",
            "--move-num",
            "0",
            "game.sgf",
        ]);
        assert_eq!(args.sgf_file, "game.sgf");
        assert_eq!(args.move_num, 0);
        assert!(!args.print_ownership);
    }

    #[test]
    fn test_evalsgf_runs_on_minimal_sgf() {
        let tmp = std::env::temp_dir();
        let sgf_path = tmp.join("katago_evalsgf_test.sgf");
        std::fs::write(
            &sgf_path,
            "(;GM[1]FF[4]SZ[9]KM[7]RU[chinese];B[de];W[ee];B[ed];W[fd])",
        )
        .unwrap();

        let args = vec![
            "--model".to_string(),
            "/dev/null".to_string(),
            "--move-num".to_string(),
            "2".to_string(),
            sgf_path.to_string_lossy().to_string(),
        ];
        let mut out = Vec::new();
        let result = evalsgf_impl(&args, &mut out);
        let output = String::from_utf8_lossy(&out);
        assert!(
            result.is_ok(),
            "evalsgf failed: {}\noutput:\n{}",
            result.unwrap_err(),
            output
        );
        assert!(
            output.contains("Warning:"),
            "expected warning output, got:\n{}",
            output
        );
    }
}
