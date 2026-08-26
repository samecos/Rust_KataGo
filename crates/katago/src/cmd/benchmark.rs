//! Benchmark command implementation.
//!
//! Corresponds to `MainCmds::benchmark` in `cpp/command/benchmark.cpp`.

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

use std::collections::BTreeMap;
use std::io::Write;

use clap::Parser;

use kata_core::config::ConfigParser;
use kata_core::global::StringError;
use kata_core::logger::{Logger, LoggerOptions};
use kata_core::rng::Rand;
use kata_core::time::timer::ClockTimer;
use kata_data::sgf::CompactSgf;
use kata_game::board::{Board, Move, P_BLACK, Player, get_opp, location};
use kata_game::history::BoardHistory;
use kata_game::rules::Rules;
use kata_nn::eval::NnEvaluator;
use kata_program::setup::{self, SetupFor};
use kata_search::params::SearchParams;
use kata_search::search::Search;

use crate::cli::CommonArgs;

pub(crate) const DEFAULT_MAX_VISITS: i64 = 800;
pub(crate) const DEFAULT_SECONDS_PER_GAME_MOVE: f64 = 5.0;
pub(crate) const TERNARY_SEARCH_INITIAL_MAX: i32 = 32;
const MIN_BENCHMARK_SGF_DATA_SIZE: i32 = 9;
const MAX_BENCHMARK_SGF_DATA_SIZE: i32 = 19;
const DEFAULT_BENCHMARK_SGF_DATA_SIZE: i32 = 19;

/// CLI arguments for the `benchmark` subcommand.
#[derive(Parser, Debug, Clone)]
struct BenchmarkArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// How many visits to use per search.
    #[arg(short = 'v', long = "visits", value_name = "VISITS", default_value_t = DEFAULT_MAX_VISITS)]
    visits: i64,

    /// Test these many threads, comma-separated, e.g. '4,8,12,16'.
    #[arg(short = 't', long = "threads", value_name = "THREADS")]
    threads: Option<String>,

    /// How many positions to sample from a game.
    #[arg(
        short = 'n',
        long = "numpositions",
        value_name = "NUM",
        default_value_t = 10
    )]
    num_positions: i32,

    /// Optional game to sample positions from.
    #[arg(long = "sgf", value_name = "FILE")]
    sgf: Option<String>,

    /// Size of board to benchmark on.
    #[arg(long = "boardsize", value_name = "SIZE")]
    board_size: Option<i32>,

    /// Automatically search for the optimal number of threads.
    #[arg(short = 's', long = "tune")]
    tune: bool,

    /// Set max batch size to this fixed value.
    #[arg(long = "fixed-batch-size", value_name = "NUM")]
    fixed_batch_size: Option<i32>,

    /// Set max batch size to half of the number of threads.
    #[arg(long = "half-batch-size")]
    half_batch_size: bool,

    /// Typical amount of time per move spent while playing, in seconds.
    #[arg(short = 'i', long = "time", value_name = "SECONDS", default_value_t = DEFAULT_SECONDS_PER_GAME_MOVE)]
    time: f64,
}

/// Results collected for a single benchmark configuration.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BenchmarkResults {
    pub(crate) num_threads: i32,
    total_positions_searched: i32,
    total_positions: i32,
    pub(crate) total_visits: i64,
    pub(crate) total_seconds: f64,
    num_nn_evals: i64,
    num_nn_batches: i64,
    avg_batch_size: f64,
}

impl BenchmarkResults {
    fn new(num_threads: i32, total_positions: i32) -> Self {
        Self {
            num_threads,
            total_positions_searched: 0,
            total_positions,
            total_visits: 0,
            total_seconds: 0.0,
            num_nn_evals: 0,
            num_nn_batches: 0,
            avg_batch_size: 0.0,
        }
    }

    fn as_string_not_done(&self) -> String {
        format!(
            "numSearchThreads = {:2}: {} / {} positions, visits/s = {:.2} ({:.1} secs)",
            self.num_threads,
            self.total_positions_searched,
            self.total_positions,
            self.total_visits as f64 / (self.total_seconds + 0.00001),
            self.total_seconds
        )
    }

    fn as_string(&self) -> String {
        format!(
            "numSearchThreads = {:2}: {} / {} positions, visits/s = {:.2}, \
             nnEvals/s = {:.2}, nnBatches/s = {:.2}, avgBatchSize = {:.2} ({:.1} secs)",
            self.num_threads,
            self.total_positions_searched,
            self.total_positions,
            self.total_visits as f64 / (self.total_seconds + 0.00001),
            self.num_nn_evals as f64 / (self.total_seconds + 0.00001),
            self.num_nn_batches as f64 / (self.total_seconds + 0.00001),
            self.avg_batch_size,
            self.total_seconds
        )
    }

    fn as_string_with_elo(
        &self,
        baseline: Option<&BenchmarkResults>,
        seconds_per_game_move: f64,
    ) -> String {
        let mut s = format!(
            "numSearchThreads = {:2}: {} / {} positions, visits/s = {:.2}, \
             nnEvals/s = {:.2}, nnBatches/s = {:.2}, avgBatchSize = {:.2} ({:.1} secs)",
            self.num_threads,
            self.total_positions_searched,
            self.total_positions,
            self.total_visits as f64 / (self.total_seconds + 0.00001),
            self.num_nn_evals as f64 / (self.total_seconds + 0.00001),
            self.num_nn_batches as f64 / (self.total_seconds + 0.00001),
            self.avg_batch_size,
            self.total_seconds
        );

        if let Some(baseline) = baseline {
            let diff = self.compute_elo_effect(seconds_per_game_move)
                - baseline.compute_elo_effect(seconds_per_game_move);
            s.push_str(&format!(" (EloDiff {:+.0})", diff));
        } else {
            s.push_str(" (EloDiff baseline)");
        }
        s
    }

    pub(crate) fn compute_elo_effect(&self, seconds_per_game_move: f64) -> f64 {
        const ELO_GAIN_PER_DOUBLING: f64 = 250.0;

        let compute_elo_cost = |base_visits: f64| {
            self.num_threads as f64 * 7.0 * (1600.0 / (800.0 + base_visits)).powf(0.85)
        };

        let visits_per_second = self.total_visits as f64 / (self.total_seconds + 0.00001);
        let gain =
            ELO_GAIN_PER_DOUBLING * (visits_per_second * seconds_per_game_move / 800.0).log2();
        let visits_per_move = visits_per_second * seconds_per_game_move;
        let cost = compute_elo_cost(visits_per_move);
        gain - cost
    }
}

/// Public CLI entry point.
pub fn benchmark(args: &[String]) -> i32 {
    let mut out = std::io::stdout();
    match benchmark_impl(args, &mut out) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {}", e);
            1
        }
    }
}

/// Internal testable entry point.
fn benchmark_impl(args: &[String], out: &mut dyn Write) -> Result<(), StringError> {
    let parsed = parse_args(args)?;
    validate_args(&parsed)?;

    let mut cfg = if parsed.common.config.is_empty() {
        let mut cfg = minimal_benchmark_config();
        parsed.common.maybe_apply_override_config_arg(&mut cfg)?;
        cfg
    } else {
        parsed.common.get_config("gtp_example.cfg")?
    };

    let model_file = parsed.common.get_model_file()?;

    let log_to_stdout_default = true;
    let logger: &'static Logger = Box::leak(Box::new(Logger::new(
        LoggerOptions {
            log_to_stdout: log_to_stdout_default,
            log_to_stderr: true,
            log_time: true,
        },
        None,
    )));

    logger.write("Loading model and initializing benchmark...");

    let sgf = load_sgf(&parsed, &mut cfg, logger)?;

    let mut params = setup::load_single_params(&cfg, SetupFor::Benchmark)
        .map_err(|e| StringError::new(format!("Could not load params: {}", e)))?;
    params.max_visits = parsed.visits;
    params.max_playouts = parsed.visits;
    params.max_time = 1e20;
    params.search_factor_after_one_pass = 1.0;
    params.search_factor_after_two_pass = 1.0;

    setup::initialize_session(&cfg);

    if cfg.contains("nnMaxBatchSize") {
        let msg = if let Some(fixed) = parsed.fixed_batch_size {
            format!(
                "WARNING: Your nnMaxBatchSize is hardcoded to {}, ignoring it and assuming it is {}, for this benchmark.",
                cfg.get_string("nnMaxBatchSize").unwrap_or_default(),
                fixed
            )
        } else if parsed.half_batch_size {
            format!(
                "WARNING: Your nnMaxBatchSize is hardcoded to {}, ignoring it and assuming it is = threads/2, for this benchmark.",
                cfg.get_string("nnMaxBatchSize").unwrap_or_default()
            )
        } else {
            format!(
                "WARNING: Your nnMaxBatchSize is hardcoded to {}, ignoring it and assuming it is >= threads, for this benchmark.",
                cfg.get_string("nnMaxBatchSize").unwrap_or_default()
            )
        };
        writeln!(out, "{}", msg).map_err(|e| StringError::new(e.to_string()))?;
    }

    let mut nn_eval: Option<&'static NnEvaluator> = None;
    let mut max_threads_for_current_nneval: i32 = -1;

    let reallocate_nneval = |max_num_threads: i32,
                             nn_eval: &mut Option<&'static NnEvaluator>,
                             max_threads: &mut i32| {
        if *max_threads >= max_num_threads {
            return;
        }
        let batch_size_limit = if let Some(fixed) = parsed.fixed_batch_size {
            fixed
        } else if parsed.half_batch_size {
            (max_num_threads + 1) / 2
        } else {
            max_num_threads
        };
        let new_eval = create_nneval(batch_size_limit, &sgf, &model_file, logger, &cfg, &params);
        *nn_eval = Some(new_eval);
        *max_threads = max_num_threads;
    };

    let get_desired_batch_size = |current_num_threads: i32, nn_eval: &NnEvaluator| -> i32 {
        if let Some(fixed) = parsed.fixed_batch_size {
            fixed
        } else if parsed.half_batch_size {
            (current_num_threads + 1) / 2
        } else {
            nn_eval.max_batch_size()
        }
    };

    let auto_tune_threads = parsed.tune || parsed.threads.is_none();
    let num_threads_to_test: Vec<i32> = if !auto_tune_threads {
        parse_thread_list(parsed.threads.as_ref().unwrap())?
    } else {
        Vec::new()
    };

    if !auto_tune_threads {
        let max_threads = *num_threads_to_test.iter().max().unwrap_or(&1);
        reallocate_nneval(
            max_threads,
            &mut nn_eval,
            &mut max_threads_for_current_nneval,
        );
    } else {
        reallocate_nneval(
            TERNARY_SEARCH_INITIAL_MAX,
            &mut nn_eval,
            &mut max_threads_for_current_nneval,
        );
    }

    let nn_eval = nn_eval
        .ok_or_else(|| StringError::new("Neural net evaluator was not initialized".to_string()))?;

    logger.write(&format!("Loaded config {}", cfg.file_name()));
    logger.write(&format!("Loaded model {}", model_file));

    writeln!(out).map_err(|e| StringError::new(e.to_string()))?;
    writeln!(out, "Testing using {} visits.", parsed.visits)
        .map_err(|e| StringError::new(e.to_string()))?;
    if parsed.visits == DEFAULT_MAX_VISITS {
        writeln!(out, "  If you have a good GPU, you might increase this using \"-visits N\" to get more accurate results.").map_err(|e| StringError::new(e.to_string()))?;
        writeln!(out, "  If you have a weak GPU and this is taking forever, you can decrease it instead to finish the benchmark faster.").map_err(|e| StringError::new(e.to_string()))?;
    }
    writeln!(out).map_err(|e| StringError::new(e.to_string()))?;

    writeln!(
        out,
        "Your GTP config is currently set to use numSearchThreads = {}",
        params.num_threads
    )
    .map_err(|e| StringError::new(e.to_string()))?;

    let results = if !auto_tune_threads {
        do_fixed_tune_threads(
            &params,
            &sgf,
            parsed.num_positions,
            nn_eval,
            logger,
            parsed.time,
            &num_threads_to_test,
            true,
            &get_desired_batch_size,
            out,
        )?
    } else {
        do_auto_tune_threads(
            &params,
            &sgf,
            parsed.num_positions,
            nn_eval,
            logger,
            parsed.time,
            &reallocate_nneval,
            &get_desired_batch_size,
            out,
        )?
    };

    if num_threads_to_test.len() > 1 || auto_tune_threads {
        print_elo_comparison(&results, parsed.time, out)?;

        writeln!(
            out,
            "If you care about performance, you may want to edit numSearchThreads in {} based on the above results!",
            cfg.file_name()
        )
        .map_err(|e| StringError::new(e.to_string()))?;

        if cfg.contains("nnMaxBatchSize") {
            writeln!(
                out,
                "WARNING: Your nnMaxBatchSize is hardcoded to {}, recommend deleting it and using the default (which this benchmark assumes)",
                cfg.get_string("nnMaxBatchSize").unwrap_or_default()
            )
            .map_err(|e| StringError::new(e.to_string()))?;
        }

        writeln!(out, "If you intend to do much longer searches, configure the seconds per game move you expect with the '-time' flag and benchmark again.").map_err(|e| StringError::new(e.to_string()))?;
        writeln!(out, "If you intend to do short or fixed-visit searches, use lower numSearchThreads for better strength, high threads will weaken strength.").map_err(|e| StringError::new(e.to_string()))?;
        writeln!(
            out,
            "If interested see also other notes about performance and mem usage in the top of {}",
            cfg.file_name()
        )
        .map_err(|e| StringError::new(e.to_string()))?;
        writeln!(out).map_err(|e| StringError::new(e.to_string()))?;
    }

    Ok(())
}

fn parse_args(args: &[String]) -> Result<BenchmarkArgs, StringError> {
    BenchmarkArgs::try_parse_from(std::iter::once(&"benchmark".to_string()).chain(args.iter()))
        .map_err(|e| StringError::new(format!("Argument error: {}", e)))
}

fn validate_args(args: &BenchmarkArgs) -> Result<(), StringError> {
    if args.board_size.is_some() && args.sgf.is_some() {
        return Err(StringError::new(
            "Cannot specify both -sgf and -boardsize at the same time".to_string(),
        ));
    }
    if let Some(board_size) = args.board_size {
        if !(MIN_BENCHMARK_SGF_DATA_SIZE..=MAX_BENCHMARK_SGF_DATA_SIZE).contains(&board_size) {
            return Err(StringError::new(format!(
                "Board size to test: invalid value {}",
                board_size
            )));
        }
    }
    if args.visits <= 1 || args.visits >= 1_000_000_000 {
        return Err(StringError::new(format!(
            "Number of visits to use: invalid value {}",
            args.visits
        )));
    }
    if args.num_positions <= 0 || args.num_positions > 100_000 {
        return Err(StringError::new(format!(
            "Number of positions per game to use: invalid value {}",
            args.num_positions
        )));
    }
    if args.time <= 0.0 || args.time > 1_000_000.0 {
        return Err(StringError::new(format!(
            "Number of seconds per game move to assume: invalid value {}",
            args.time
        )));
    }
    if args.threads.is_some() && args.tune {
        return Err(StringError::new(
            "Cannot both automatically tune threads and specify fixed exact numbers of threads to test".to_string(),
        ));
    }
    if let Some(fixed) = args.fixed_batch_size {
        if fixed <= 0 || fixed > 65536 {
            return Err(StringError::new(
                "Invalid value for fixed batch size".to_string(),
            ));
        }
    }
    if args.fixed_batch_size.is_some() && args.half_batch_size {
        return Err(StringError::new(
            "Cannot specify both fixed batch size and use half batch size".to_string(),
        ));
    }
    Ok(())
}

fn minimal_benchmark_config() -> ConfigParser {
    ConfigParser::from_str(
        "numSearchThreads = 1
         nnMaxBatchSize = 8
         valueWeightExponent = 0
        ",
        false,
        false,
    )
    .expect("hard-coded minimal config is valid")
}

fn load_sgf(
    parsed: &BenchmarkArgs,
    cfg: &mut ConfigParser,
    logger: &Logger,
) -> Result<CompactSgf, StringError> {
    if let Some(sgf_file) = &parsed.sgf {
        CompactSgf::parse_file(sgf_file)
            .map_err(|e| StringError::new(format!("Could not load SGF file: {}", e)))
    } else {
        let board_size = if let Some(size) = parsed.board_size {
            size
        } else {
            let mut default_board_x_size = DEFAULT_BENCHMARK_SGF_DATA_SIZE;
            let mut default_board_y_size = DEFAULT_BENCHMARK_SGF_DATA_SIZE;
            let _ = setup::load_default_board_xy_size(
                cfg,
                logger,
                &mut default_board_x_size,
                &mut default_board_y_size,
            );
            std::cmp::max(default_board_x_size, default_board_y_size)
        };
        logger.write(&format!(
            "Testing with default positions for board size: {}",
            board_size
        ));
        let sgf_data = get_benchmark_sgf_data(board_size);
        CompactSgf::parse(sgf_data)
            .map_err(|e| StringError::new(format!("Could not parse built-in SGF: {}", e)))
    }
}

/// Return the built-in benchmark SGF for the requested board size.
///
/// These strings are taken from `TestCommon::getBenchmarkSGFData` in
/// `KataGo/cpp/tests/testcommon.cpp` to match the original C++ behavior.
pub(crate) fn get_benchmark_sgf_data(board_size: i32) -> &'static str {
    match board_size {
        9 => {
            "(;FF[4]GM[1]SZ[9]HA[0]KM[7];B[ef];W[ed];B[ge];W[gc];B[cc];W[cd];B[bd];W[ce];B[be];W[dg];B[cf];W[df];B[de];W[dd];B[ee];W[cg];B[bf];W[cb];B[eg];W[bc];B[bh];W[he];B[hd];W[gf];B[fe];W[hf];B[fc];W[eb];B[gd];W[fh];B[eh];W[hh];B[ac];W[dc];B[fb];W[ab];B[fg];W[gg];B[fi];W[bg];B[dh];W[gh];B[ea];W[da];B[fa];W[ad];B[ch];W[id];B[ic];W[ie];B[gb];W[gi];B[ec];W[hc];B[hb];W[ei];B[db];W[ae];B[ag];W[eb];B[ig];W[db];B[ih];W[ii];B[di];W[ac];B[fi];W[hg];B[ei];W[af];B[ff];W[if];B[fd];W[bb])"
        }
        13 => {
            "(;FF[4]GM[1]SZ[13]HA[0]KM[7.5];B[dd];W[jj];B[kk];W[jd];B[kj];W[dj];B[jc];W[dc];B[cc];W[ec];B[ed];W[fc];B[fd];W[ic];B[hc];W[kc];B[gc];W[ch];B[jb];W[ib];B[id];W[ie];B[hd];W[kb];B[je];W[ja];B[kd];W[jc];B[jf];W[jk];B[ji];W[ii];B[fk];W[kl];B[ll];W[il];B[km];W[dk];B[hk];W[jg];B[kg];W[jh];B[ki];W[gj];B[hj];W[gk];B[hl];W[kh];B[if];W[lg];B[gi];W[li];B[jl];W[le];B[bg];W[bh];B[cg];W[dg];B[df];W[cb];B[dh];W[di];B[eg];W[db];B[bj];W[bi];B[aj];W[ai];B[cl];W[bk];B[dl];W[fj];B[fl];W[fi];B[cj];W[ck];B[bl];W[gh];B[hi];W[eh];B[ig];W[al];B[ek];W[ej];B[fg];W[fh];B[ih];W[kf];B[ba];W[gl];B[gm];W[hb];B[ke];W[ld];B[gg];W[ag];B[bf];W[bc];B[cd];W[ca];B[bd];W[lj];B[lk];W[hh];B[hg];W[mk];B[ml];W[mj];B[dg];W[bm];B[cm];W[ci];B[am];W[ac];B[ad];W[bm];B[af];W[ah];B[am];W[em];B[ak];W[al];B[fm];W[bm];B[ak];W[aj];B[am];W[ab];B[aa];W[bm];B[fb];W[fa];B[am];W[el];B[dm];W[bm];B[hm];W[ak];B[am];W[em];B[el];W[bm];B[ea];W[eb];B[am];W[gd])"
        }
        19 => {
            "(;FF[4]GM[1]SZ[19]HA[0]KM[7.5];B[dd];W[pp];B[dp];W[pd];B[qq];W[pq];B[qp];W[qo];B[ro];W[rn];B[qn];W[po];B[rm];W[rp];B[sn];W[rq];B[nc];W[oc];B[qr];W[rr];B[nd];W[pf];B[re];W[lc];B[jc];W[le];B[qc];W[qd];B[ob];W[pc];B[pb];W[rc];B[qb];W[rd];B[lb];W[mb];B[ld];W[kb];B[kc];W[la];B[mc];W[qi];B[ke];W[cc];B[dc];W[cd];B[cf];W[ce];B[de];W[bf];B[cb];W[bb];B[be];W[db];B[bc];W[ca];B[bd];W[cb];B[bg];W[df];B[cg];W[fc];B[nr];W[nq];B[pg];W[qg];B[of];W[ef];B[mq];W[cq];B[dq];W[cp];B[co];W[bo];B[bn];W[cn];B[do];W[bm];B[bp];W[an];B[bq];W[cr];B[br];W[kq];B[iq];W[ko];B[kr];W[cj];B[lq];W[bi];B[qj];W[ph];B[og];W[rf];B[gc];W[gd];B[hc];W[nb];B[rb];W[sb];B[lb];W[fe];B[ri];W[rh];B[pi];W[qh];B[af];W[lc];B[dn];W[dm];B[em];W[in];B[lb];W[fn];B[lc];W[en];B[fp];W[fm];B[pj];W[hp];B[hq];W[ij];B[di];W[dj];B[jl];W[il];B[jm];W[im];B[jk];W[jj];B[kj];W[ki];B[lj];W[li];B[mi];W[jh];B[pn];W[or];B[np];W[ie];B[dh];W[ei];B[eh];W[fh];B[kn];W[oo];B[mn];W[oh];B[ni];W[oq];B[fg];W[gh];B[hg];W[hh];B[ig];W[ih];B[fb];W[eb];B[gb];W[mj];B[mg];W[kf];B[lf];W[je];B[kg];W[kd];B[jg];W[ln];B[lm];W[lo];B[nn];W[ra];B[na];W[gp];B[gq];W[fo];B[rj];W[oe];B[me];W[mr];B[lr];W[ns];B[sp];W[on];B[om];W[pm];B[nm];W[ip];B[jp];W[jo];B[kp];W[lp];B[jq];W[ap];B[dr];W[ar];B[cs];W[aq];B[sr];W[qs];B[ab];W[ea];B[ec];W[fd];B[fa];W[ee];B[ke];W[oi];B[oj];W[le];B[ik];W[hk];B[ke];W[ep];B[fq];W[le];B[eo];W[ke];B[sq];W[pr];B[qm];W[ml];B[mk];W[lk];B[nj];W[kk];B[mj];W[km];B[kl];W[qa];B[hd];W[he];B[gf];W[ge];B[bh];W[nf];B[ng];W[ne];B[ci];W[ai];B[mf];W[eg];B[gg];W[dg];B[ch];W[si];B[ah];W[bj];B[sj];W[sh];B[ma];W[sc];B[pa];W[id];B[ic];W[ls];B[ks];W[ms];B[sa];W[ra];B[od];W[pe];B[kh];W[lh];B[ji];W[ii];B[mh];W[ji];B[lg];W[mp];B[no];W[jn];B[km];W[as];B[fi];W[ej])"
        }
        _ => get_benchmark_sgf_data(DEFAULT_BENCHMARK_SGF_DATA_SIZE),
    }
}

fn parse_thread_list(s: &str) -> Result<Vec<i32>, StringError> {
    let mut result = Vec::new();
    for piece in s.split(',') {
        let piece = kata_core::global::trim(piece);
        if piece.is_empty() {
            continue;
        }
        let threads = kata_core::global::try_string_to_int(piece).ok_or_else(|| {
            StringError::new(format!(
                "Number of threads to use: invalid value: {}",
                piece
            ))
        })?;
        if threads <= 0 || threads > 4096 {
            return Err(StringError::new(format!(
                "Number of threads to use: invalid value: {}",
                piece
            )));
        }
        result.push(threads);
    }
    if result.is_empty() {
        return Err(StringError::new(
            "Must specify at least one valid value for -threads".to_string(),
        ));
    }
    Ok(result)
}

fn setup_initial_board_and_hist(
    sgf: &CompactSgf,
    initial_rules: &Rules,
) -> Result<(Board, Player, BoardHistory), StringError> {
    let next_pla = if let Some(first_move) = sgf.moves.first() {
        first_move.pla
    } else {
        P_BLACK
    };

    let mut board = Board::new(sgf.x_size, sgf.y_size);
    if !board.set_stones_fail_if_no_libs(&sgf.placements) {
        return Err(StringError::new(
            "setup_initial_board_and_hist: initial board position contains invalid stones or zero-liberty stones".to_string(),
        ));
    }

    let mut hist = BoardHistory::new(board.clone(), next_pla, *initial_rules, 0);
    if hist.initial_turn_number < board.num_stones_on_board() as i64 {
        hist.set_initial_turn_number(board.num_stones_on_board() as i64);
    }

    Ok((board, next_pla, hist))
}

pub(crate) fn create_nneval(
    max_num_threads: i32,
    sgf: &CompactSgf,
    model_file: &str,
    logger: &Logger,
    cfg: &ConfigParser,
    params: &SearchParams,
) -> &'static NnEvaluator {
    let expected_concurrent_evals = max_num_threads;
    let default_max_batch_size = ((max_num_threads + 3) / 4 * 4).max(8);

    let mut seed_rand = Rand::new();

    let nn_eval = setup::initialize_nn_evaluator(
        model_file.to_string(),
        model_file.to_string(),
        String::new(),
        cfg,
        logger,
        &mut seed_rand,
        expected_concurrent_evals,
        sgf.x_size,
        sgf.y_size,
        default_max_batch_size,
        true,
        false,
        SetupFor::Benchmark,
    )
    .expect("Failed to initialize NN evaluator");

    let nn_eval: &'static NnEvaluator = Box::leak(Box::new(nn_eval));

    warm_start_nneval(sgf, logger, params, nn_eval, &mut seed_rand);

    nn_eval
}

fn warm_start_nneval(
    sgf: &CompactSgf,
    logger: &Logger,
    params: &SearchParams,
    nn_eval: &NnEvaluator,
    seed_rand: &mut Rand,
) {
    let (board, next_pla, hist) = setup_initial_board_and_hist(sgf, &Rules::default())
        .expect("Failed to set up initial board for warm start");

    let mut this_params = params.clone();
    this_params.num_threads = 1;
    this_params.max_visits = 5;
    this_params.max_playouts = 5;
    this_params.max_time = 1e20;

    let mut bot = Search::new(
        this_params,
        nn_eval,
        logger,
        &seed_rand.next_u64().to_string(),
    );
    bot.set_position(next_pla, &board, &hist);
    bot.run_whole_search(next_pla);
}

fn set_num_threads(
    params: &mut SearchParams,
    nn_eval: &NnEvaluator,
    num_threads: i32,
    desired_batch_size: i32,
) {
    params.num_threads = num_threads;
    nn_eval.set_current_batch_size(desired_batch_size);
}

pub(crate) fn do_fixed_tune_threads(
    params: &SearchParams,
    sgf: &CompactSgf,
    num_positions_per_game: i32,
    nn_eval: &NnEvaluator,
    logger: &Logger,
    seconds_per_game_move: f64,
    num_threads_to_test: &[i32],
    print_elo: bool,
    get_desired_batch_size: &dyn Fn(i32, &NnEvaluator) -> i32,
    out: &mut dyn Write,
) -> Result<Vec<BenchmarkResults>, StringError> {
    if num_threads_to_test.len() > 1 {
        writeln!(
            out,
            "Testing different numbers of threads (board size {}x{}):",
            sgf.x_size, sgf.y_size
        )
        .map_err(|e| StringError::new(e.to_string()))?;
    } else {
        writeln!(out, "Testing (board size {}x{}):", sgf.x_size, sgf.y_size)
            .map_err(|e| StringError::new(e.to_string()))?;
    }

    let mut results = Vec::new();
    for (i, &num_threads) in num_threads_to_test.iter().enumerate() {
        let baseline = if i == 0 { None } else { results.first() };
        let mut this_params = params.clone();
        let desired_batch_size = get_desired_batch_size(num_threads, nn_eval);
        set_num_threads(&mut this_params, nn_eval, num_threads, desired_batch_size);
        let result = benchmark_search_on_positions_and_print(
            this_params,
            sgf,
            num_positions_per_game,
            nn_eval,
            baseline,
            seconds_per_game_move,
            print_elo,
            logger,
            out,
        )?;
        results.push(result);
    }
    writeln!(out).map_err(|e| StringError::new(e.to_string()))?;
    Ok(results)
}

pub(crate) fn do_auto_tune_threads(
    params: &SearchParams,
    sgf: &CompactSgf,
    num_positions_per_game: i32,
    nn_eval: &'static NnEvaluator,
    logger: &Logger,
    seconds_per_game_move: f64,
    reallocate_nneval: &dyn Fn(i32, &mut Option<&'static NnEvaluator>, &mut i32),
    get_desired_batch_size: &dyn Fn(i32, &NnEvaluator) -> i32,
    out: &mut dyn Write,
) -> Result<Vec<BenchmarkResults>, StringError> {
    writeln!(
        out,
        "Automatically trying different numbers of threads to home in on the best (board size {}x{}):",
        sgf.x_size, sgf.y_size
    )
    .map_err(|e| StringError::new(e.to_string()))?;
    writeln!(out).map_err(|e| StringError::new(e.to_string()))?;

    let mut possible_numbers_of_threads: Vec<i32> = Vec::new();
    let mut twopow: i32 = 1;
    for _ in 0..20 {
        possible_numbers_of_threads.push(twopow);
        possible_numbers_of_threads.push(twopow * 3);
        possible_numbers_of_threads.push(twopow * 5);
        twopow *= 2;
    }
    possible_numbers_of_threads.sort_unstable();

    let mut ternary_search_min = nn_eval.num_gpus().max(1);
    let mut ternary_search_max =
        (TERNARY_SEARCH_INITIAL_MAX as f64 * 0.5 * (1.0 + nn_eval.num_gpus() as f64)).round()
            as i32;
    if ternary_search_max < ternary_search_min * 4 {
        ternary_search_max = ternary_search_min * 4;
    }

    let mut result_cache: BTreeMap<i32, BenchmarkResults> = BTreeMap::new();
    let mut nn_eval_opt: Option<&'static NnEvaluator> = Some(nn_eval);
    let mut max_threads_for_current_nneval: i32 = ternary_search_max;

    loop {
        reallocate_nneval(
            ternary_search_max,
            &mut nn_eval_opt,
            &mut max_threads_for_current_nneval,
        );
        let nn_eval = nn_eval_opt.unwrap();
        writeln!(out).map_err(|e| StringError::new(e.to_string()))?;

        let mut start = 0usize;
        let mut end = possible_numbers_of_threads.len().saturating_sub(1);
        for (i, &n) in possible_numbers_of_threads.iter().enumerate() {
            if n < ternary_search_min {
                start = i + 1;
            }
            if n > ternary_search_max {
                end = i.saturating_sub(1);
                break;
            }
        }
        if start > end {
            start = end;
        }

        write!(out, "Possible numbers of threads to test: ")
            .map_err(|e| StringError::new(e.to_string()))?;
        for i in start..=end {
            write!(out, "{}, ", possible_numbers_of_threads[i])
                .map_err(|e| StringError::new(e.to_string()))?;
        }
        writeln!(out).map_err(|e| StringError::new(e.to_string()))?;
        writeln!(out).map_err(|e| StringError::new(e.to_string()))?;

        while start <= end {
            let first_mid = start + (end - start) / 3;
            let second_mid = end - (end - start) / 3;

            let effect1 = get_or_run_result(
                possible_numbers_of_threads[first_mid],
                params,
                sgf,
                num_positions_per_game,
                nn_eval,
                logger,
                seconds_per_game_move,
                get_desired_batch_size,
                &mut result_cache,
                out,
            )?
            .compute_elo_effect(seconds_per_game_move);
            let effect2 = get_or_run_result(
                possible_numbers_of_threads[second_mid],
                params,
                sgf,
                num_positions_per_game,
                nn_eval,
                logger,
                seconds_per_game_move,
                get_desired_batch_size,
                &mut result_cache,
                out,
            )?
            .compute_elo_effect(seconds_per_game_move);

            if effect1 < effect2 {
                start = first_mid + 1;
            } else {
                end = second_mid.saturating_sub(1);
            }
        }

        let mut best_elo = f64::NEG_INFINITY;
        let mut best_threads = 0;
        let mut results: Vec<BenchmarkResults> = Vec::new();
        for result in result_cache.values() {
            let elo = result.compute_elo_effect(seconds_per_game_move);
            results.push(*result);
            if elo > best_elo {
                best_threads = result.num_threads;
                best_elo = elo;
            }
        }

        if 3 * best_threads > 2 * ternary_search_max && ternary_search_max < 5000 {
            ternary_search_min = ternary_search_max / 2;
            ternary_search_max = ternary_search_max * 2 + 32;
            writeln!(out).map_err(|e| StringError::new(e.to_string()))?;
            writeln!(out, "Optimal number of threads is fairly high, increasing the search limit and trying again.").map_err(|e| StringError::new(e.to_string()))?;
            writeln!(out).map_err(|e| StringError::new(e.to_string()))?;
            continue;
        }

        results.sort_by(|a, b| {
            b.compute_elo_effect(seconds_per_game_move)
                .partial_cmp(&a.compute_elo_effect(seconds_per_game_move))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        writeln!(out).map_err(|e| StringError::new(e.to_string()))?;
        writeln!(out, "Ordered summary of results:")
            .map_err(|e| StringError::new(e.to_string()))?;
        writeln!(out).map_err(|e| StringError::new(e.to_string()))?;
        for (i, result) in results.iter().enumerate() {
            let baseline = if i == 0 { None } else { results.first() };
            writeln!(
                out,
                "{}",
                result.as_string_with_elo(baseline, seconds_per_game_move)
            )
            .map_err(|e| StringError::new(e.to_string()))?;
        }
        writeln!(out).map_err(|e| StringError::new(e.to_string()))?;
        return Ok(results);
    }
}

fn get_or_run_result(
    num_threads: i32,
    params: &SearchParams,
    sgf: &CompactSgf,
    num_positions_per_game: i32,
    nn_eval: &NnEvaluator,
    logger: &Logger,
    seconds_per_game_move: f64,
    get_desired_batch_size: &dyn Fn(i32, &NnEvaluator) -> i32,
    result_cache: &mut BTreeMap<i32, BenchmarkResults>,
    out: &mut dyn Write,
) -> Result<BenchmarkResults, StringError> {
    if let Some(result) = result_cache.get(&num_threads) {
        return Ok(*result);
    }
    let mut this_params = params.clone();
    let desired_batch_size = get_desired_batch_size(num_threads, nn_eval);
    set_num_threads(&mut this_params, nn_eval, num_threads, desired_batch_size);
    let result = benchmark_search_on_positions_and_print(
        this_params,
        sgf,
        num_positions_per_game,
        nn_eval,
        None,
        seconds_per_game_move,
        false,
        logger,
        out,
    )?;
    result_cache.insert(num_threads, result);
    Ok(result)
}

pub(crate) fn benchmark_search_on_positions_and_print(
    params: SearchParams,
    sgf: &CompactSgf,
    num_positions_to_use: i32,
    nn_eval: &NnEvaluator,
    baseline: Option<&BenchmarkResults>,
    seconds_per_game_move: f64,
    print_elo: bool,
    logger: &Logger,
    out: &mut dyn Write,
) -> Result<BenchmarkResults, StringError> {
    let mut moves = sgf.moves.clone();
    if moves.len() > 0xFFFF {
        moves.resize(0xFFFF, Move::new(0, P_BLACK));
    }

    let mut pos_seed = "benchmarkPosSeed|".to_string();
    for m in &moves {
        pos_seed.push_str(&format!("{}|", m.loc));
    }

    let mut possible_position_idxs: Vec<usize> = (0..moves.len()).collect();
    {
        let mut pos_rand = Rand::new_from_seed(&pos_seed);
        for i in (2..possible_position_idxs.len()).rev() {
            let r = pos_rand.next_u32_bounded((i + 1) as u32) as usize;
            possible_position_idxs.swap(i, r);
        }
        if possible_position_idxs.len() > num_positions_to_use as usize {
            possible_position_idxs.resize(num_positions_to_use as usize, 0);
        }
    }
    possible_position_idxs.sort_unstable();

    let mut result = BenchmarkResults::new(params.num_threads, possible_position_idxs.len() as i32);

    nn_eval.clear_cache();
    nn_eval.clear_stats();

    let mut seed_rand = Rand::new();
    let mut bot = Search::new(params, nn_eval, logger, &seed_rand.next_u64().to_string());

    let mut initial_rules = Rules::get_tromp_taylorish();
    initial_rules.komi = sgf
        .get_rules_or_fail_allow_unspecified(&initial_rules)
        .map_err(|e| StringError::new(format!("Could not get SGF rules: {}", e)))?
        .komi;

    let (mut board, mut next_pla, mut hist) = setup_initial_board_and_hist(sgf, &initial_rules)?;
    let mut move_num = 0usize;

    for &next_idx in &possible_position_idxs {
        write!(out, "\r{}      ", result.as_string_not_done())
            .map_err(|e| StringError::new(e.to_string()))?;
        out.flush().map_err(|e| StringError::new(e.to_string()))?;

        while move_num < moves.len() && move_num < next_idx {
            let m = moves[move_num];
            if !hist.make_board_move_tolerant(&mut board, m.loc, m.pla) {
                return Err(StringError::new(format!(
                    "SGF Illegal move {} for {} at {}",
                    move_num + 1,
                    if m.pla == P_BLACK { "Black" } else { "White" },
                    location::to_string(m.loc, board.x_size, board.y_size)
                )));
            }
            next_pla = get_opp(m.pla);
            move_num += 1;
        }

        bot.clear_search();
        bot.set_position(next_pla, &board, &hist);
        nn_eval.clear_cache();

        let timer = ClockTimer::new();
        bot.run_whole_search(next_pla);
        let seconds = timer.get_seconds();

        result.total_positions_searched += 1;
        result.total_seconds += seconds;
        result.total_visits += bot.get_root_visits();
    }

    result.num_nn_evals = nn_eval.num_rows_processed() as i64;
    result.num_nn_batches = nn_eval.num_batches_processed() as i64;
    result.avg_batch_size = nn_eval.average_processed_batch_size();

    if print_elo {
        writeln!(
            out,
            "\r{}",
            result.as_string_with_elo(baseline, seconds_per_game_move)
        )
        .map_err(|e| StringError::new(e.to_string()))?;
    } else {
        writeln!(out, "\r{}", result.as_string()).map_err(|e| StringError::new(e.to_string()))?;
    }

    Ok(result)
}

pub(crate) fn print_elo_comparison(
    results: &[BenchmarkResults],
    seconds_per_game_move: f64,
    out: &mut dyn Write,
) -> Result<(), StringError> {
    let mut best_idx = 0usize;
    for i in 1..results.len() {
        if results[i].compute_elo_effect(seconds_per_game_move)
            > results[best_idx].compute_elo_effect(seconds_per_game_move)
        {
            best_idx = i;
        }
    }

    writeln!(out).map_err(|e| StringError::new(e.to_string()))?;
    writeln!(
        out,
        "Based on some test data, each speed doubling gains perhaps ~250 Elo by searching deeper."
    )
    .map_err(|e| StringError::new(e.to_string()))?;
    writeln!(out, "Based on some test data, each thread costs perhaps 7 Elo if using 800 visits, and 2 Elo if using 5000 visits (by making MCTS worse).").map_err(|e| StringError::new(e.to_string()))?;
    writeln!(
        out,
        "So APPROXIMATELY based on this benchmark, if you intend to do a {} second search:",
        seconds_per_game_move
    )
    .map_err(|e| StringError::new(e.to_string()))?;
    for (i, result) in results.iter().enumerate() {
        let elo_effect = result.compute_elo_effect(seconds_per_game_move)
            - results[0].compute_elo_effect(seconds_per_game_move);
        write!(out, "numSearchThreads = {:2}: ", result.num_threads)
            .map_err(|e| StringError::new(e.to_string()))?;
        if i == 0 {
            writeln!(
                out,
                "(baseline){}",
                if i == best_idx { " (recommended)" } else { "" }
            )
            .map_err(|e| StringError::new(e.to_string()))?;
        } else {
            writeln!(
                out,
                "{:+5.0} Elo{}",
                elo_effect,
                if i == best_idx { " (recommended)" } else { "" }
            )
            .map_err(|e| StringError::new(e.to_string()))?;
        }
    }
    writeln!(out).map_err(|e| StringError::new(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_args_defaults() {
        let args = BenchmarkArgs::parse_from([
            "benchmark",
            "--model",
            "/dev/null",
            "--visits",
            "10",
            "--threads",
            "1",
        ]);
        assert_eq!(args.visits, 10);
        assert_eq!(args.threads, Some("1".to_string()));
        assert_eq!(args.num_positions, 10);
        assert!(!args.tune);
        assert!(!args.half_batch_size);
        assert_eq!(args.time, 5.0);
    }

    #[test]
    fn test_validate_args_rejects_sgf_and_boardsize() {
        let args = BenchmarkArgs::parse_from([
            "benchmark",
            "--model",
            "/dev/null",
            "--sgf",
            "game.sgf",
            "--boardsize",
            "19",
        ]);
        assert!(validate_args(&args).is_err());
    }

    #[test]
    fn test_validate_args_rejects_invalid_visits() {
        let args =
            BenchmarkArgs::parse_from(["benchmark", "--model", "/dev/null", "--visits", "1"]);
        assert!(validate_args(&args).is_err());
    }

    #[test]
    fn test_parse_thread_list() {
        assert_eq!(parse_thread_list("1,2,4").unwrap(), vec![1, 2, 4]);
        assert_eq!(parse_thread_list(" 8 ").unwrap(), vec![8]);
        assert!(parse_thread_list("").is_err());
        assert!(parse_thread_list("0").is_err());
        assert!(parse_thread_list("4097").is_err());
    }

    #[test]
    fn test_benchmark_results_formatting() {
        let mut r = BenchmarkResults::new(4, 5);
        r.total_positions_searched = 5;
        r.total_visits = 100;
        r.total_seconds = 2.0;
        r.num_nn_evals = 50;
        r.num_nn_batches = 10;
        r.avg_batch_size = 5.0;

        let s = r.as_string();
        assert!(s.contains("numSearchThreads =  4"));
        assert!(s.contains("5 / 5 positions"));
        assert!(s.contains("visits/s = 50.00"));
        assert!(s.contains("nnEvals/s = 25.00"));

        let with_elo = r.as_string_with_elo(None, 5.0);
        assert!(with_elo.contains("(EloDiff baseline)"));
    }

    #[test]
    fn test_tiny_benchmark_runs() {
        let args = vec![
            "--model".to_string(),
            "/dev/null".to_string(),
            "--visits".to_string(),
            "2".to_string(),
            "--threads".to_string(),
            "1".to_string(),
            "--numpositions".to_string(),
            "2".to_string(),
            "--boardsize".to_string(),
            "9".to_string(),
        ];
        let mut output: Vec<u8> = Vec::new();
        let result = benchmark_impl(&args, &mut output);
        assert!(result.is_ok(), "benchmark failed: {:?}", result);

        let s = String::from_utf8(output).expect("output is valid utf-8");
        assert!(
            s.contains("numSearchThreads =  1"),
            "missing thread report: {}",
            s
        );
        assert!(
            s.contains("visits/s = ") || s.contains("visits/s ="),
            "missing visits/s: {}",
            s
        );
        assert!(s.contains("positions"), "missing positions: {}", s);
    }
}
