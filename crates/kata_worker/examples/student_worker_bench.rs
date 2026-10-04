//! Isolated Worker Evaluator throughput/latency measurement; no RPC or search.
//! --prepare is CPU-only. --run loads the fixed real student CUDA model.
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Barrier, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use kata_core::config::ConfigParser;
use kata_core::rng::Rand;
use kata_game::board::{Board, P_BLACK, get_opp, location};
use kata_game::history::BoardHistory;
use kata_game::rules::Rules;
use kata_worker::{Evaluator, evaluator::Engine, wire};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const MODEL_SHA: &str = "fb3b77e796db158d3fb6a836291f7efa0fb33bc5b7ac0f0f5f9e45ae8061721a";
const HISTORY_LENGTHS: [usize; 8] = [0, 1, 4, 8, 16, 32, 64, 128];

#[derive(Clone, Serialize, Deserialize)]
struct Position {
    index: usize,
    moves: Vec<(u32, i32)>,
    next_player: u32,
}

#[derive(Serialize, Deserialize)]
struct Corpus {
    schema: String,
    seed: String,
    positions: Vec<Position>,
}

#[derive(Clone, Serialize, Deserialize)]
struct Case {
    id: String,
    batch: usize,
    concurrency: usize,
    server_threads: usize,
    warmup_ms: u64,
    duration_ms: u64,
    repetitions: usize,
}

fn write_fresh(path: &str, value: &impl Serialize) -> Result<()> {
    let mut file = fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(&serde_json::to_vec_pretty(value)?)?;
    file.write_all(b"\n")?;
    Ok(())
}

fn sha_file(path: impl AsRef<Path>) -> Result<String> {
    Ok(hex::encode(Sha256::digest(fs::read(path)?)))
}

fn validate(corpus: &Corpus, cases: &[Case]) -> Result<()> {
    ensure!(corpus.schema == "rustgo_student_worker_corpus_v1" && corpus.seed == "student-worker-20261004-v1");
    ensure!(corpus.positions.len() == 64);
    for (index, position) in corpus.positions.iter().enumerate() {
        ensure!(position.index == index && position.moves.len() == HISTORY_LENGTHS[index % 8]);
        let mut board = Board::new(19, 19);
        let mut rules = Rules::parse_rules("chinese")?;
        rules.set_komi(7.5);
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
        hist.set_assume_multiple_starting_black_moves_are_handicap(false);
        let mut next = P_BLACK;
        for &(color, vertex) in &position.moves {
            ensure!(color == next as u32 && (0..361).contains(&vertex) && !hist.is_game_finished);
            let loc = location::get_loc(vertex % 19, vertex / 19, 19);
            ensure!(hist.is_legal(&board, loc, next), "illegal fixture {index}");
            hist.make_board_move_assume_legal(&mut board, loc, next);
            next = get_opp(next);
        }
        ensure!(!hist.is_game_finished && next as u32 == position.next_player);
    }
    ensure!(!cases.is_empty() && cases.len() <= 2);
    for case in cases {
        ensure!(!case.id.is_empty() && case.server_threads == 1);
        ensure!(matches!((case.batch, case.concurrency), (8, 8) | (8, 64) | (16, 64) | (32, 64)));
        ensure!(case.warmup_ms == 2000 && matches!(case.duration_ms, 3000 | 5000) && case.repetitions == 3);
    }
    Ok(())
}

fn prepare() -> Result<Corpus> {
    let mut positions = Vec::new();
    for index in 0..64 {
        let mut board = Board::new(19, 19);
        let mut rules = Rules::parse_rules("chinese")?;
        rules.set_komi(7.5);
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
        hist.set_assume_multiple_starting_black_moves_are_handicap(false);
        let mut next = P_BLACK;
        let mut rand = Rand::new_from_seed(&format!("student-worker-20261004-v1-{index}"));
        let mut moves = Vec::new();
        for _ in 0..HISTORY_LENGTHS[index % 8] {
            let mut legal = Vec::new();
            for vertex in 0..361 {
                let loc = location::get_loc(vertex % 19, vertex / 19, 19);
                if hist.is_legal(&board, loc, next) {
                    legal.push(vertex);
                }
            }
            ensure!(!legal.is_empty());
            let vertex = legal[rand.next_u64() as usize % legal.len()];
            hist.make_board_move_assume_legal(&mut board, location::get_loc(vertex % 19, vertex / 19, 19), next);
            moves.push((next as u32, vertex));
            next = get_opp(next);
        }
        positions.push(Position { index, moves, next_player: next as u32 });
    }
    Ok(Corpus { schema: "rustgo_student_worker_corpus_v1".into(), seed: "student-worker-20261004-v1".into(), positions })
}

fn templates(corpus: &Corpus) -> Vec<wire::EvalRequest> {
    corpus.positions.iter().map(|position| wire::EvalRequest {
        task_id: position.index as u64 + 1,
        generation: 1,
        session_id: "student-worker-bench".into(),
        model_sha256: MODEL_SHA.into(),
        position: Some(wire::Position {
            board_size: 19, komi: 7.5, rules: "chinese".into(), initial_player: 1,
            next_player: position.next_player as i32,
            moves: position.moves.iter().map(|&(color, vertex)| wire::Move { color: color as i32, vertex }).collect(),
            ..Default::default()
        }),
        parameters: Some(wire::EvalParameters {
            symmetry: (position.index % 8) as i32, policy_temperature: 1.0,
            policy_optimism: 0.75, draw_equivalent_wins_for_white: 0.5,
            include_ownership: true, max_history: 1000, skip_cache: true,
            ..Default::default()
        }),
        ..Default::default()
    }).collect()
}

fn settle(engine: &Engine, expected_rows: u64) -> Result<(u64, u64)> {
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let (rows, batches) = engine.stats();
        if rows == expected_rows { return Ok((rows, batches)); }
        ensure!(rows < expected_rows && Instant::now() < deadline, "NN row counter {rows}, expected {expected_rows}");
        std::thread::yield_now();
    }
}

fn distribution(mut values: Vec<u64>) -> Value {
    values.sort_unstable();
    let n = values.len();
    let percentile = |p: f64| values[((n - 1) as f64 * p).ceil() as usize] as f64 / 1000.0;
    json!({"samples":n,"unit":"microseconds","minimum":values[0] as f64/1000.0,
           "mean":values.iter().map(|&v| v as f64/1000.0).sum::<f64>()/n as f64,
           "p50":percentile(0.5),"p95":percentile(0.95),"p99":percentile(0.99),
           "maximum":values[n-1] as f64/1000.0})
}

fn phase(engine: &Arc<Engine>, requests: &[wire::EvalRequest], concurrency: usize, ms: u64) -> Result<Value> {
    let before = engine.stats();
    let barrier = Arc::new(Barrier::new(concurrency + 1));
    let deadline = Arc::new(OnceLock::<Instant>::new());
    let start = Instant::now();
    let (thread_results, elapsed) = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for thread_index in 0..concurrency {
            let engine = Arc::clone(engine);
            let barrier = Arc::clone(&barrier);
            let deadline = Arc::clone(&deadline);
            handles.push(scope.spawn(move || -> Result<_> {
                let mut latencies = Vec::new();
                let mut context = Vec::new();
                let mut evaluator = Vec::new();
                let mut corpus_counts = vec![0u64; requests.len()];
                let mut checksum = 0.0f64;
                barrier.wait();
                let end = *deadline.get().expect("phase deadline published before barrier");
                let mut iteration = 0;
                while Instant::now() < end {
                    let index = (iteration * concurrency + thread_index) % requests.len();
                    let t0 = Instant::now();
                    let report = engine.evaluate(&requests[index]);
                    let ns = t0.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                    let output = report.result.map_err(|e| anyhow::anyhow!("{}: {}", e.code, e.message))?;
                    ensure!(output.policy.len() == 362 && output.ownership.len() == 361 && output.white_score_mean.is_finite());
                    checksum += output.white_score_mean + output.white_win_prob;
                    latencies.push(ns);
                    context.push(report.context_us.unwrap_or(0) * 1000);
                    evaluator.push(report.evaluator_us.unwrap_or(0) * 1000);
                    corpus_counts[index] += 1;
                    iteration += 1;
                }
                Ok((latencies, context, evaluator, corpus_counts, checksum))
            }));
        }
        let timed_start = Instant::now();
        deadline.set(timed_start + Duration::from_millis(ms)).unwrap();
        barrier.wait();
        let results = handles.into_iter().map(|handle| handle.join().map_err(|_| anyhow::anyhow!("measurement client panicked"))?).collect::<Result<Vec<_>>>();
        (results, timed_start.elapsed().as_secs_f64())
    });
    let mut latency = Vec::new();
    let mut context = Vec::new();
    let mut evaluator = Vec::new();
    let mut counts = vec![0u64; requests.len()];
    let mut checksum = 0.0;
    for (mut a, mut b, mut c, calls, sum) in thread_results? {
        latency.append(&mut a); context.append(&mut b); evaluator.append(&mut c);
        for (total, value) in counts.iter_mut().zip(calls) { *total += value; }
        checksum += sum;
    }
    ensure!(!latency.is_empty() && checksum.is_finite());
    let rows = latency.len() as u64;
    let after = settle(engine, before.0 + rows)?;
    let nn_rows = after.0 - before.0;
    let batches = after.1 - before.1;
    ensure!(nn_rows == rows && batches > 0, "cached/skipped inference or missing batches");
    Ok(json!({"nominal_duration_ms":ms,"completed_wall_seconds":elapsed,
       "phase_total_seconds_with_counter_settle":start.elapsed().as_secs_f64(),
       "requests":rows,"nn_rows":nn_rows,"nn_batches":batches,
       "actual_average_batch":nn_rows as f64/batches as f64,"throughput_requests_per_second":rows as f64/elapsed,
       "request_latency":distribution(latency),"context_replay_latency":distribution(context),
       "nn_queue_forward_decode_latency":distribution(evaluator),"corpus_request_counts":counts,
       "output_checksum":checksum,"counter_settle_outside_timed_region":true}))
}

fn run(model: &str, fixture_path: &str, cases_path: &str, output: &str) -> Result<()> {
    ensure!(!Path::new(output).exists(), "fresh result path required");
    ensure!(sha_file(model)? == MODEL_SHA);
    let corpus: Corpus = serde_json::from_slice(&fs::read(fixture_path)?)?;
    let cases: Vec<Case> = serde_json::from_slice(&fs::read(cases_path)?)?;
    validate(&corpus, &cases)?;
    let requests = templates(&corpus);
    let mut reports = Vec::new();
    for case in cases {
        let cfg_string = format!("nnBackend=cudabackend\ninputsUseNHWC=false\nrequireMaxBoardSize=true\nmaxBoardSizeForNNBuffer=19\nnumNNServerThreadsPerModel={}\nnnMaxBatchSize={}\nnnCacheSizePowerOfTwo=-1\nnnMutexPoolSizePowerOfTwo=-1\nnnRandSeed=student-worker-bench\n", case.server_threads, case.batch);
        let cfg = ConfigParser::from_str(&cfg_string, false, false)?;
        let engine = Arc::new(Engine::load(model, Some(MODEL_SHA), &cfg, case.concurrency as u32, false)?);
        ensure!(engine.metadata().model_version == 8 && engine.metadata().execution_profile_id.starts_with("rustgo-student-v1:"));
        let warmup = phase(&engine, &requests, case.concurrency, case.warmup_ms)?;
        let mut windows = Vec::new();
        for _ in 0..case.repetitions { windows.push(phase(&engine, &requests, case.concurrency, case.duration_ms)?); }
        let meta = engine.metadata();
        reports.push(json!({"case":case,"config":cfg_string,"model_sha256":meta.model_sha256,
            "execution_profile_id":meta.execution_profile_id,"backend_info":meta.backend_info,
            "engine_commit":meta.engine_commit,"warmup":warmup,"measurement_windows":windows}));
    }
    ensure!(sha_file(model)? == MODEL_SHA, "model changed during measurement");
    write_fresh(output, &json!({"schema":"rustgo_student_worker_bench_v1","status":"passed",
        "model_sha256":MODEL_SHA,"fixtures_sha256":sha_file(fixture_path)?,"cases_sha256":sha_file(cases_path)?,
        "cases":reports,"skip_cache":true,"nn_cache_power_of_two":-1,
        "scope":"synchronous original Worker Evaluator context replay, NN queue/forward/postprocess and response decode; no protobuf serialization/RPC/search",
        "timing":"warmup excluded; all requests admitted before fixed deadline completed and included in wall/latency; counters settled outside timed interval",
        "batch_scope":"actual dispatched rows and batch count/mean; no per-batch histogram API exposed",
        "server_started":false,"production_worker_contacted":false}))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("--prepare") if args.len() == 3 => {
            let corpus = prepare()?;
            let check = Case { id:"cpu-validation".into(), batch:8, concurrency:8, server_threads:1, warmup_ms:2000, duration_ms:3000, repetitions:3 };
            validate(&corpus, &[check])?;
            write_fresh(&args[2], &corpus)?;
            println!("CPU-only: 64 deterministic legal histories, no Engine/CUDA loaded");
            Ok(())
        }
        Some("--validate") if args.len() == 4 => {
            let corpus: Corpus = serde_json::from_slice(&fs::read(&args[2])?)?;
            let cases: Vec<Case> = serde_json::from_slice(&fs::read(&args[3])?)?;
            validate(&corpus, &cases)?;
            println!("CPU-only: corpus and cases valid, no Engine/CUDA loaded");
            Ok(())
        }
        Some("--run") if args.len() == 6 => run(&args[2], &args[3], &args[4], &args[5]),
        _ => anyhow::bail!("usage: student_worker_bench --prepare <fixtures.json> | --validate <fixtures.json> <cases.json> | --run <model> <fixtures.json> <cases.json> <result.json>"),
    }
}
