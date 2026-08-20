//! 纯前向 NN 吞吐基准 —— 对齐 KataGo C++ `benchmarknn` 口径:
//! 固定物理 batch、batch 打满,统计 nnEvals/s。
//!
//! 两种模式:
//! - `eval`(默认):B 个 worker 线程反复 `NnEvaluator::evaluate(skip_cache=true)`,
//!   走完整生产栈(特征编码 + 凑批 + submit/finish 流水线 + per-size CUDA graph),
//!   与 fork benchmarknn(经 NNEvaluator)同口径。
//! - `direct`:直连 `CudaModel::apply`(无 graph/无调度),kernel 下界对照。
//!
//! 用法:
//!   katago-rs nnbench --model D:/code/b11fix.onnx --batch 12
//!   katago-rs nnbench --model ... --batch 4,8,12,16,24,32 --iterations 400
//!   katago-rs nnbench ... --mode direct
//!   katago-rs nnbench ... --mode direct --handles 1,2 --json

use clap::Parser;

use crate::cli::CommonArgs;

#[derive(Parser, Debug)]
struct NnBenchArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// 逗号分隔的物理 batch 列表,如 "4,8,12,16"。
    #[arg(long, default_value = "12")]
    batch: String,

    /// 计时迭代次数(每 worker)。
    #[arg(long, default_value_t = 400)]
    iterations: usize,

    /// worker 线程数(默认 2×最大 batch;高周转并发才凑得满批,
    /// 与搜索多线程同构)。
    #[arg(long)]
    workers: Option<usize>,

    /// 预热迭代次数(每 worker,吸收 kernel/graph 首建)。
    #[arg(long, default_value_t = 60)]
    warmup: usize,

    /// eval = NnEvaluator 生产栈;direct = CudaModel::apply 直连。
    #[arg(long, default_value = "eval")]
    mode: String,

    /// direct/kernel 模式的并发 CUDA handle 数。每个 handle 拥有独立流、
    /// 输入和工作区；eval 模式只接受 1，以免把测量工具混入生产调度。
    #[arg(long, default_value = "1")]
    handles: String,

    /// 输出机器可解析的 JSON（仅 direct/kernel 模式）。
    #[arg(long)]
    json: bool,
}

pub fn nnbench(args: &[String]) -> i32 {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = args;
        eprintln!("nnbench 需要 CUDA 后端:cargo build -p katago --features cuda");
        1
    }
    #[cfg(feature = "cuda")]
    {
        match nnbench_impl(args) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("Error: {e}");
                1
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn nnbench_impl(args: &[String]) -> Result<(), String> {
    let parsed =
        NnBenchArgs::try_parse_from(std::iter::once(&"nnbench".to_string()).chain(args.iter()))
            .map_err(|e| format!("Argument error: {e}"))?;

    let batches: Vec<usize> = parsed
        .batch
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<_, _>>()
        .map_err(|e| format!("invalid batch list '{}': {e}", parsed.batch))?;
    if batches.is_empty() {
        return Err("batch list is empty".into());
    }
    if parsed.iterations == 0 {
        return Err("iterations must be >0".into());
    }

    let handles: Vec<usize> = parsed
        .handles
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<_, _>>()
        .map_err(|e| format!("invalid handles list '{}': {e}", parsed.handles))?;
    if handles.is_empty() || handles.iter().any(|&h| h == 0) {
        return Err("handles must contain one or more positive integers".into());
    }
    let mut unique_handles = handles.clone();
    unique_handles.sort_unstable();
    unique_handles.dedup();
    if unique_handles.len() != handles.len() {
        return Err("handles must not contain duplicates".into());
    }

    match parsed.mode.as_str() {
        "eval" if handles != [1] => {
            Err("--handles is only supported by direct/kernel; eval requires --handles 1".into())
        }
        "eval" if parsed.json => Err("--json is only supported by direct/kernel".into()),
        "eval" => eval_sweep(&parsed, &batches),
        "direct" => direct_sweep(&parsed, &batches, false, &handles),
        "kernel" => direct_sweep(&parsed, &batches, true, &handles),
        m => Err(format!("unknown mode '{m}' (eval|direct|kernel)")),
    }
}

#[cfg(feature = "cuda")]
fn eval_sweep(parsed: &NnBenchArgs, batches: &[usize]) -> Result<(), String> {
    use std::time::Instant;

    use kata_core::config::ConfigParser;
    use kata_core::logger::{Logger, LoggerOptions};
    use kata_core::rng::Rand;
    use kata_nn::backend::NNResultBuf;
    use kata_nn::eval::NnEvaluator;
    use kata_nn::inputs::MiscNNInputParams;
    use kata_program::setup::{self, SetupFor};

    let model_file = parsed.common.get_model_file().map_err(|e| e.to_string())?;

    let cfg = if parsed.common.config.is_empty() {
        let mut cfg = ConfigParser::from_str(
            "numSearchThreads = 1\nnnBackend = cudabackend\nvalueWeightExponent = 0\n",
            false,
            false,
        )
        .map_err(|e| e.to_string())?;
        parsed
            .common
            .maybe_apply_override_config_arg(&mut cfg)
            .map_err(|e| e.to_string())?;
        cfg
    } else {
        parsed
            .common
            .get_config("gtp_example.cfg")
            .map_err(|e| e.to_string())?
    };
    if cfg.get_string("nnBackend").unwrap_or_default() != "cudabackend" {
        return Err("eval 模式需要 nnBackend=cudabackend(用 --override-config 指定)".into());
    }

    let logger: &'static Logger = Box::leak(Box::new(Logger::new(
        LoggerOptions {
            log_to_stdout: false,
            log_to_stderr: false,
            log_time: true,
        },
        None,
    )));

    let max_b = *batches.iter().max().unwrap() as i32;
    // 初始化按全程最大并发开额度;每个 batch 点的 worker 数在循环内
    // 另算(见下)。
    let init_workers = parsed
        .workers
        .unwrap_or_else(|| (max_b as usize * 2).max(16));
    let mut seed_rand = Rand::new();
    let nn_eval: &'static NnEvaluator = Box::leak(Box::new(
        setup::initialize_nn_evaluator(
            model_file.clone(),
            model_file.clone(),
            String::new(),
            &cfg,
            logger,
            &mut seed_rand,
            init_workers as i32,
            19,
            19,
            init_workers.max(max_b as usize) as i32,
            true,
            false,
            SetupFor::Benchmark,
        )
        .map_err(|e| format!("init nn evaluator: {e}"))?,
    ));

    println!(
        "nnbench(eval): model={} device={} iterations={} warmup={}",
        model_file,
        nn_eval.num_gpus().max(1),
        parsed.iterations,
        parsed.warmup
    );
    println!("batch | perBatchMs | nnEvals/s");

    let params = MiscNNInputParams::default();
    for &b in batches {
        nn_eval.set_current_batch_size(b as i32);
        nn_eval.clear_cache();
        nn_eval.clear_stats();

        // worker 数必须随 batch 点走(默认 2b):serve_pipelined 的 target
        // 是发射阈值而非上限,发射时取走整个 filling  vec;若 W/3 > b,
        // 在途满 2 期间 filling 会积过 target,avgBatch 被抬到 ~W/3 而
        // 不再是固定物理 B。2b 时 W/3 < b,阈值发射主导,批≈b。
        let workers = parsed.workers.unwrap_or_else(|| (b * 2).max(2));

        // workers 个确定性局面(与 dump_nn_io_cuda 的 make_position 同源)。
        let positions: Vec<_> = (0..workers).map(bench_position).collect();
        let run_rounds = |rounds: usize| -> Result<(), String> {
            // &NnEvaluator 跨线程与搜索同款(search.rs 的 `unsafe impl Send
            // for SearchThread`):evaluate 客户端路径只经 Arc<SharedState>
            // 与锁保护字段,compute handle 仅 server 线程触碰。
            struct EvalShared<'a>(&'a NnEvaluator);
            unsafe impl Send for EvalShared<'_> {}
            std::thread::scope(|s| {
                for (_i, (board, hist, next)) in positions.iter().enumerate() {
                    let nn = EvalShared(nn_eval);
                    let params = &params;
                    s.spawn(move || {
                        // 先绑定整个 newtype,防止 2021 精确捕获绕过
                        // EvalShared 的 unsafe Send(直接 nn.0 会按字段捕获
                        // &NnEvaluator 本身)。
                        let nn = nn;
                        let nn = nn.0;
                        let mut buf = NNResultBuf::new();
                        for _ in 0..rounds {
                            nn.evaluate(board, hist, *next, params, &mut buf, true, false);
                        }
                    });
                }
            });
            Ok(())
        };

        run_rounds(parsed.warmup)?;
        nn_eval.clear_cache();
        nn_eval.clear_stats();

        let t0 = Instant::now();
        run_rounds(parsed.iterations)?;
        let secs = t0.elapsed().as_secs_f64();

        let rows = workers * parsed.iterations;
        let evals_per_s = rows as f64 / secs;
        let per_batch_ms = secs * 1000.0 / parsed.iterations as f64;
        let avg_b = nn_eval.average_processed_batch_size();
        println!(
            "{b:5} | {per_batch_ms:10.3} | {evals_per_s:9.1}   (workers {workers}, avgBatch {avg_b:.2})",
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn direct_sweep(
    parsed: &NnBenchArgs,
    batches: &[usize],
    kernel_only: bool,
    handles: &[usize],
) -> Result<(), String> {
    use std::collections::BTreeMap;
    use std::sync::{Arc, Barrier};
    use std::time::Instant;

    use kata_nn::backends::cuda::CudaRuntime;
    use kata_nn::backends::cuda_exec::{CudaModel, CudaWorkspace};
    use kata_nn::onnx_parser::parse_layer_graph;
    use serde::Serialize;

    let model_file = parsed.common.get_model_file().map_err(|e| e.to_string())?;
    let bytes = std::fs::read(&model_file).map_err(|e| format!("read model: {e}"))?;
    let rt = CudaRuntime::new().map_err(|e| format!("CUDA runtime: {e}"))?;
    let graph = parse_layer_graph(&bytes).map_err(|e| format!("parse layer graph: {e}"))?;
    let stream = rt.device.default_stream();
    let model = Arc::new(
        CudaModel::load(&graph, &rt, &stream).map_err(|e| format!("load CudaModel: {e}"))?,
    );
    rt.device.synchronize().map_err(|e| format!("sync: {e}"))?;
    let rt_shared = Arc::new(rt);

    #[derive(Debug, Serialize)]
    struct HandleSample {
        handle: usize,
        iterations: usize,
        elapsed_ms: f64,
        per_batch_ms: f64,
        nn_evals_per_s: f64,
    }
    #[derive(Debug, Serialize)]
    struct BatchResult {
        batch: usize,
        handles: Vec<HandleSample>,
        wall_elapsed_ms: f64,
        wall_nn_evals_per_s: f64,
    }
    #[derive(Debug, Serialize)]
    struct JsonReport {
        schema: u32,
        command: &'static str,
        mode: String,
        kernel_only: bool,
        model: String,
        device_target: String,
        revision: String,
        build: kata_nn::tactic_plan::BackendBuildFingerprint,
        tactics: BTreeMap<String, Option<String>>,
        warmup: usize,
        iterations: usize,
        requested_batches: Vec<usize>,
        handles: Vec<usize>,
        results: Vec<BatchResult>,
    }

    if !parsed.json {
        println!(
            "nnbench({}): model={} handles={} iterations={} warmup={}",
            if kernel_only { "kernel" } else { "direct" },
            model_file,
            handles
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(","),
            parsed.iterations,
            parsed.warmup
        );
        println!("batch | wallBatchMs | wall nnEvals/s | per-handle nnEvals/s");
    }

    let mut results = Vec::with_capacity(batches.len());

    for &b in batches {
        // 每个 handle 都复制一份 host 输入；设备输入、workspace 和 stream
        // 在对应线程中独立创建，避免跨流复用可变缓冲。
        let (spatial, global) = batch_inputs(b);
        let barrier = Arc::new(Barrier::new(handles.len() + 1));
        let done_barrier = Arc::new(Barrier::new(handles.len() + 1));
        let (mut samples, wall_elapsed_ms) = std::thread::scope(|scope| -> Result<_, String> {
            let mut scope_results = Vec::with_capacity(handles.len());
            for (_handle_idx, &handle_id) in handles.iter().enumerate() {
                let barrier = barrier.clone();
                let done_barrier = done_barrier.clone();
                let rt = rt_shared.clone();
                let model = model.clone();
                let spatial = spatial.clone();
                let global = global.clone();
                scope_results.push(scope.spawn(move || -> Result<HandleSample, String> {
                    // Keep the barrier outside the fallible setup body. Any
                    // allocation/warmup failure must still release peers.
                    let setup_result = (|| -> Result<_, String> {
                        let stream = rt
                            .device
                            .new_stream()
                            .map_err(|e| format!("handle {handle_id} stream: {e}"))?;
                        let mut d_spatial = unsafe { stream.alloc(spatial.len()) }
                            .map_err(|e| format!("handle {handle_id} alloc spatial: {e}"))?;
                        let mut d_global = unsafe { stream.alloc(global.len()) }
                            .map_err(|e| format!("handle {handle_id} alloc global: {e}"))?;
                        let mut ws = CudaWorkspace::new(&stream, &model, b)
                            .map_err(|e| format!("handle {handle_id} workspace: {e}"))?;
                        if kernel_only {
                            // Kernel-only mode excludes transfers from the timed
                            // loop, but still needs deterministic valid inputs.
                            stream
                                .memcpy_htod(&spatial, &mut d_spatial)
                                .map_err(|e| format!("handle {handle_id} init spatial: {e}"))?;
                            stream
                                .memcpy_htod(&global, &mut d_global)
                                .map_err(|e| format!("handle {handle_id} init global: {e}"))?;
                        }
                        let mut run_once = || -> Result<(), String> {
                            if !kernel_only {
                                stream
                                    .memcpy_htod(&spatial, &mut d_spatial)
                                    .map_err(|e| format!("handle {handle_id} htod spatial: {e}"))?;
                                stream
                                    .memcpy_htod(&global, &mut d_global)
                                    .map_err(|e| format!("handle {handle_id} htod global: {e}"))?;
                            }
                            model
                                .apply(&rt, &stream, &mut ws, &d_spatial, &d_global)
                                .map_err(|e| format!("handle {handle_id} apply: {e}"))?;
                            if !kernel_only {
                                ws.to_host(&stream)
                                    .map_err(|e| format!("handle {handle_id} dtoh: {e}"))?;
                            }
                            Ok(())
                        };
                        for _ in 0..parsed.warmup {
                            run_once()?;
                        }
                        stream
                            .synchronize()
                            .map_err(|e| format!("handle {handle_id} warmup sync: {e}"))?;
                        Ok((stream, d_spatial, d_global, ws))
                    })();
                    barrier.wait();

                    let worker_result =
                        setup_result.and_then(|(stream, mut d_spatial, mut d_global, mut ws)| {
                            let mut run_once = || -> Result<(), String> {
                                if !kernel_only {
                                    stream.memcpy_htod(&spatial, &mut d_spatial).map_err(|e| {
                                        format!("handle {handle_id} htod spatial: {e}")
                                    })?;
                                    stream.memcpy_htod(&global, &mut d_global).map_err(|e| {
                                        format!("handle {handle_id} htod global: {e}")
                                    })?;
                                }
                                model
                                    .apply(&rt, &stream, &mut ws, &d_spatial, &d_global)
                                    .map_err(|e| format!("handle {handle_id} apply: {e}"))?;
                                if !kernel_only {
                                    ws.to_host(&stream)
                                        .map_err(|e| format!("handle {handle_id} dtoh: {e}"))?;
                                }
                                Ok(())
                            };
                            let t0 = Instant::now();
                            for _ in 0..parsed.iterations {
                                run_once()?;
                            }
                            stream
                                .synchronize()
                                .map_err(|e| format!("handle {handle_id} final sync: {e}"))?;
                            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                            let per_batch_ms = elapsed_ms / parsed.iterations as f64;
                            Ok(HandleSample {
                                handle: handle_id,
                                iterations: parsed.iterations,
                                elapsed_ms,
                                per_batch_ms,
                                nn_evals_per_s: b as f64 * 1000.0 / per_batch_ms,
                            })
                        });
                    // Exclude workspace/buffer destruction from the wall clock.
                    done_barrier.wait();
                    worker_result
                }));
            }
            barrier.wait();
            // Start after setup barrier release so workspace allocation and
            // warmup are not charged to the steady-state wall throughput.
            let wall_start = Instant::now();
            done_barrier.wait();
            let wall_elapsed_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
            let mut samples = Vec::with_capacity(handles.len());
            for result in scope_results {
                let joined = result
                    .join()
                    .map_err(|_| "nnbench worker panicked".to_string())?;
                samples.push(joined?);
            }
            Ok((samples, wall_elapsed_ms))
        })?;
        let wall_nn_evals_per_s =
            handles.len() as f64 * b as f64 * parsed.iterations as f64 / (wall_elapsed_ms / 1000.0);
        samples.sort_by_key(|sample| sample.handle);
        if !parsed.json {
            println!(
                "{b:5} | {wall_elapsed_ms:12.3} | {wall_nn_evals_per_s:14.1} | {}",
                samples
                    .iter()
                    .map(|sample| format!("h{}={:.1}", sample.handle, sample.nn_evals_per_s))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
        results.push(BatchResult {
            batch: b,
            handles: samples,
            wall_elapsed_ms,
            wall_nn_evals_per_s,
        });
    }

    if parsed.json {
        let mut tactics = BTreeMap::new();
        for &key in kata_nn::tactic_plan::ALLOWED_TACTIC_KEYS {
            tactics.insert(key.to_string(), kata_nn::tactic_plan::tactic_var(key).ok());
        }
        let report = JsonReport {
            schema: 1,
            command: "nnbench",
            mode: parsed.mode.clone(),
            kernel_only,
            model: model_file,
            device_target: rt_shared.target_id.clone(),
            revision: source_revision(),
            build: kata_nn::backends::cuda::backend_build_fingerprint(),
            tactics,
            warmup: parsed.warmup,
            iterations: parsed.iterations,
            requested_batches: batches.to_vec(),
            handles: handles.to_vec(),
            results,
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&report).map_err(|e| format!("json: {e}"))?
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn source_revision() -> String {
    if let Some(revision) = option_env!("KATAGO_GIT_REVISION") {
        return revision.to_string();
    }
    std::process::Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|revision| revision.trim().to_string())
        .filter(|revision| !revision.is_empty())
        .unwrap_or_else(|| format!("rust-{}", env!("CARGO_PKG_VERSION")))
}

/// 确定性局面 i:空盘 + i*7%80 步伪随机合法落子
/// (与 dump_nn_io_cuda 测试的 make_position 同源)。
#[cfg(feature = "cuda")]
fn bench_position(
    i: usize,
) -> (
    kata_game::board::Board,
    kata_game::history::BoardHistory,
    kata_game::board::Player,
) {
    use kata_game::board::{Board, P_BLACK, get_opp, location};
    use kata_game::history::BoardHistory;
    use kata_game::rules::Rules;

    let mut board = Board::new(19, 19);
    let mut hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
    let mut next_player = P_BLACK;
    let mut rand = kata_core::rng::Rand::new_from_seed(&format!("nnbench-{i}"));
    let num_moves = i * 7 % 80;
    for _ in 0..num_moves {
        let mut legal = Vec::new();
        for y in 0..19 {
            for x in 0..19 {
                let loc = location::get_loc(x, y, 19);
                if hist.is_legal(&board, loc, next_player) {
                    legal.push(loc);
                }
            }
        }
        if legal.is_empty() {
            break;
        }
        let loc = legal[rand.next_u64() as usize % legal.len()];
        hist.make_board_move_assume_legal(&mut board, loc, next_player);
        next_player = get_opp(next_player);
    }
    (board, hist, next_player)
}

/// B 个确定性局面的特征拼接(spatial [B,22,19,19] NCHW f32 + global [B,19]),
/// 供 direct 模式一次性整批提交。
#[cfg(feature = "cuda")]
fn batch_inputs(b: usize) -> (Vec<f32>, Vec<f32>) {
    use kata_nn::inputs::{MiscNNInputParams, fill_row_v7};

    let mut spatial = vec![0.0f32; b * 22 * 19 * 19];
    let mut global = vec![0.0f32; b * 19];
    let params = MiscNNInputParams::default();
    for i in 0..b {
        let (board, hist, next) = bench_position(i);
        let s_off = i * 22 * 19 * 19;
        let g_off = i * 19;
        fill_row_v7(
            &board,
            &hist,
            next,
            &params,
            19,
            19,
            false, // NCHW(cuda 后端强制 NCHW)
            &mut spatial[s_off..s_off + 22 * 19 * 19],
            &mut global[g_off..g_off + 19],
        );
    }
    (spatial, global)
}
