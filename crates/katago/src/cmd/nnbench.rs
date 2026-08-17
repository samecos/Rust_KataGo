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
    let parsed = NnBenchArgs::try_parse_from(
        std::iter::once(&"nnbench".to_string()).chain(args.iter()),
    )
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
    if parsed.iterations == 0 || parsed.warmup < 2 {
        return Err("iterations must be >0 and warmup >= 2".into());
    }

    match parsed.mode.as_str() {
        "eval" => eval_sweep(&parsed, &batches),
        "direct" => direct_sweep(&parsed, &batches, false),
        "kernel" => direct_sweep(&parsed, &batches, true),
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
    let init_workers = parsed.workers.unwrap_or_else(|| (max_b as usize * 2).max(16));
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
fn direct_sweep(parsed: &NnBenchArgs, batches: &[usize], kernel_only: bool) -> Result<(), String> {
    use std::time::Instant;

    use kata_nn::backends::cuda::CudaRuntime;
    use kata_nn::backends::cuda_exec::{CudaModel, CudaWorkspace};
    use kata_nn::onnx_parser::parse_layer_graph;

    let model_file = parsed.common.get_model_file().map_err(|e| e.to_string())?;
    let bytes = std::fs::read(&model_file).map_err(|e| format!("read model: {e}"))?;
    let rt = CudaRuntime::new().map_err(|e| format!("CUDA runtime: {e}"))?;
    let graph = parse_layer_graph(&bytes).map_err(|e| format!("parse layer graph: {e}"))?;
    let stream = rt.device.default_stream();
    let model = CudaModel::load(&graph, &rt, &stream).map_err(|e| format!("load CudaModel: {e}"))?;
    rt.device.synchronize().map_err(|e| format!("sync: {e}"))?;

    println!("nnbench(direct): model={model_file} iterations={} warmup={}",
        parsed.iterations, parsed.warmup);
    println!("batch | perBatchMs | nnEvals/s");

    for &b in batches {
        // 输入构造一次(内容与耗时无关),之后每轮重新 H2D(与生产请求路径一致)。
        let (spatial, global) = batch_inputs(b);
        let mut d_spatial =
            unsafe { stream.alloc(spatial.len()) }.map_err(|e| format!("alloc spatial: {e}"))?;
        let mut d_global =
            unsafe { stream.alloc(global.len()) }.map_err(|e| format!("alloc global: {e}"))?;
        let mut ws = CudaWorkspace::new(&stream, &model, b).map_err(|e| format!("workspace: {e}"))?;

        let mut run_once = |ws: &mut CudaWorkspace| -> Result<(), String> {
            if !kernel_only {
                stream
                    .memcpy_htod(&spatial, &mut d_spatial)
                    .map_err(|e| format!("htod spatial: {e}"))?;
                stream
                    .memcpy_htod(&global, &mut d_global)
                    .map_err(|e| format!("htod global: {e}"))?;
            }
            model
                .apply(&rt, &stream, ws, &d_spatial, &d_global)
                .map_err(|e| format!("apply: {e}"))?;
            if !kernel_only {
                ws.to_host(&stream).map_err(|e| format!("dtoh: {e}"))?;
            }
            Ok(())
        };

        for _ in 0..parsed.warmup {
            run_once(&mut ws)?;
        }
        stream.synchronize().map_err(|e| format!("warmup sync: {e}"))?;

        // 每轮 to_host 为同步 D2H,循环 wall time 即 GPU 时间线(含 H2D/前向/D2H)。
        let t0 = Instant::now();
        for _ in 0..parsed.iterations {
            run_once(&mut ws)?;
        }
        stream.synchronize().map_err(|e| format!("final sync: {e}"))?;
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        let per_batch_ms = ms / parsed.iterations as f32;
        let evals_per_s = b as f32 * 1000.0 / per_batch_ms;
        println!("{b:5} | {per_batch_ms:10.3} | {evals_per_s:9.1}",);
    }
    Ok(())
}

/// 确定性局面 i:空盘 + i*7%80 步伪随机合法落子
/// (与 dump_nn_io_cuda 测试的 make_position 同源)。
#[cfg(feature = "cuda")]
fn bench_position(i: usize) -> (kata_game::board::Board, kata_game::history::BoardHistory, kata_game::board::Player) {
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
