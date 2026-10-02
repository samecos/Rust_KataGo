//! 纯前向 NN 吞吐基准 —— 对齐 KataGo C++ `benchmarknn` 口径:
//! 固定物理 batch、batch 打满,统计 nnEvals/s。
//!
//! 三种模式:
//! - `eval`(默认):worker 线程反复 `NnEvaluator::evaluate(skip_cache=true)`,
//!   走完整生产栈(特征编码 + 凑批 + submit/finish 流水线 + per-size CUDA graph),
//!   报告真实后端 rows/batches；C++ benchmarknn 则跳过特征编码与请求调度。
//! - `direct`:直连 `CudaModel::apply`(无 graph/无调度),kernel 下界对照。
//! - `kernel`:设备输入常驻,计时纯前向(无 graph/拷贝/调度)。
//!   MXFP8 additionally snapshots each forward's status on-device: wall time
//!   includes these copies, while each CUDA end event precedes its snapshot.
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

    /// worker 线程数(默认 2×当前 batch;高周转并发才凑得满批,
    /// 与搜索多线程同构)。
    #[arg(long)]
    workers: Option<usize>,

    /// 预热迭代次数(每 worker,吸收 kernel/graph 首建)。
    #[arg(long, default_value_t = 60)]
    warmup: usize,

    /// eval = NnEvaluator 生产栈;direct = CudaModel::apply 含拷贝;kernel = 纯前向。
    #[arg(long, default_value = "eval")]
    mode: String,

    /// direct/kernel 输入：positions 为原有确定性局面；empty 对齐 Fork
    /// benchmarknn 的空盘、黑行、TrompTaylorish 7.5 目，整批重复同一行。
    #[arg(long, default_value = "positions", value_parser = ["positions", "empty"])]
    input: String,

    /// direct/kernel 模式的并发 CUDA handle 数。每个 handle 拥有独立流、
    /// 输入和工作区；eval 模式只接受 1，以免把测量工具混入生产调度。
    #[arg(long, default_value = "1")]
    handles: String,

    /// 输出机器可解析的 JSON（仅 direct/kernel 模式）。
    #[arg(long)]
    json: bool,

    /// kernel 模式可额外记录每次前向的 CUDA event 时间，对齐 Fork 的
    /// 各流中位数口径；墙钟吞吐仍独立报告。事件仅在循环结束后同步。
    #[arg(long, default_value = "wall", value_parser = ["wall", "cuda-event"])]
    timing: String,
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

#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchCudaBackend {
    Fp16,
    Int8,
    Quantized,
}

/// Reject incompatible precision options before reading a model or touching
/// CUDA. In particular, direct/kernel cannot authenticate an evaluator profile.
#[cfg(any(feature = "cuda", test))]
fn benchmark_backend(
    cfg: &kata_core::config::ConfigParser,
    direct: bool,
) -> Result<BenchCudaBackend, String> {
    let key = if cfg.contains("nnBackend0") {
        "nnBackend0"
    } else {
        "nnBackend"
    };
    let name = if cfg.contains(key) {
        cfg.get_string(key).map_err(|e| e.to_string())?
    } else if !direct {
        return Err("eval mode requires an explicitly configured CUDA backend".into());
    } else {
        "cudabackend".into()
    };
    let backend = match name.as_str() {
        "cuda" | "cudabackend" => BenchCudaBackend::Fp16,
        "cudaint8" | "cudaint8backend" => BenchCudaBackend::Int8,
        "cudaquant" | "cudaquantbackend" => BenchCudaBackend::Quantized,
        _ => {
            return Err("nnbench requires cudabackend, cudaint8backend or cudaquantbackend".into());
        }
    };
    if backend != BenchCudaBackend::Int8
        && (cfg.contains("cudaInt8Scope") || cfg.contains("cudaInt8MinFfnWidth"))
    {
        return Err("cudaInt8Scope/cudaInt8MinFfnWidth requires nnBackend=cudaint8backend; cudaquantbackend selects layers in cudaQuantPlan".into());
    }
    if backend != BenchCudaBackend::Quantized
        && (cfg.contains("cudaQuantPlan") || cfg.contains("cudaQuantExpectedProfile"))
    {
        return Err(
            "cudaQuantPlan/cudaQuantExpectedProfile requires nnBackend=cudaquantbackend".into(),
        );
    }
    if direct && cfg.contains("cudaTacticPlan") {
        return Err(
            "direct/kernel modes do not install cudaTacticPlan; use eval for plan benchmarks"
                .into(),
        );
    }
    if backend == BenchCudaBackend::Quantized {
        if cfg.contains("cudaTacticPlan") {
            return Err(
                "cudaquantbackend uses cudaQuantPlan and cannot reuse cudaTacticPlan".into(),
            );
        }
        if cfg.contains("cudaQuantPlan")
            && cfg
                .get_string("cudaQuantPlan")
                .map_err(|e| e.to_string())?
                .trim()
                .is_empty()
        {
            return Err("cudaQuantPlan must not be empty; omit it for the FP16 base recipe".into());
        }
        if direct && cfg.contains("cudaQuantExpectedProfile") {
            return Err("direct/kernel modes cannot verify cudaQuantExpectedProfile; use --mode eval for evaluator execution-profile verification".into());
        }
    }
    Ok(backend)
}

#[cfg(any(feature = "cuda", test))]
const QUANTIZED_STATUS_WORDS: usize = 2;

#[cfg(any(feature = "cuda", test))]
fn quantized_status_ring_words(iterations: usize) -> Result<usize, String> {
    iterations
        .checked_mul(QUANTIZED_STATUS_WORDS)
        .filter(|&words| words.checked_mul(std::mem::size_of::<u32>()).is_some())
        .ok_or_else(|| "quantized benchmark status ring size overflow".to_string())
}

/// Validate every saved forward, not only the last workspace status after its
/// predecessors have been reset. This helper is also testable without CUDA.
#[cfg(any(feature = "cuda", test))]
fn validate_quantized_status_ring(
    statuses: &[u32],
    iterations: usize,
    mut validate: impl FnMut(&[u32]) -> Result<(), String>,
) -> Result<(), String> {
    let expected = quantized_status_ring_words(iterations)?;
    if statuses.len() != expected {
        return Err(format!(
            "quantized benchmark status ring has {} words, expected {expected}",
            statuses.len()
        ));
    }
    for (iteration, status) in statuses.chunks_exact(QUANTIZED_STATUS_WORDS).enumerate() {
        validate(status).map_err(|error| {
            format!(
                "quantized benchmark timed iteration {}: {error}",
                iteration + 1
            )
        })?;
    }
    Ok(())
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
    if batches.iter().any(|&batch| !(1..=65536).contains(&batch)) {
        return Err("batch sizes must be in 1..=65536".into());
    }
    if parsed.workers == Some(0) {
        return Err("workers must be positive".into());
    }
    if parsed.iterations == 0 {
        return Err("iterations must be >0".into());
    }
    if parsed.timing == "cuda-event" && parsed.mode != "kernel" {
        return Err("--timing cuda-event requires --mode kernel".into());
    }
    if parsed.input != "positions" && !matches!(parsed.mode.as_str(), "direct" | "kernel") {
        return Err("--input empty requires --mode direct or --mode kernel".into());
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
    benchmark_backend(&cfg, false)?;

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
            max_b,
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
    if let Some(profile) = nn_eval.inference_profile_id() {
        println!("[cuda-quant] inference_profile={profile} validation=unverified");
    }
    println!("batch | perBatchMs | nnEvals/s | rows | batches | avgBatch");

    let params = MiscNNInputParams::default();
    for &b in batches {
        nn_eval.set_current_batch_size(b as i32);
        nn_eval.clear_cache();
        nn_eval.clear_stats();

        // The physical allocation is the largest requested batch, independent
        // of client concurrency. At smaller sweep points, report the actual
        // average as dispatch may exceed its target up to that physical cap.
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

        // Results awaken their callers immediately before the server updates
        // counters. Settle those counters outside the timed region so the last
        // warmup batch cannot leak into the next measurement.
        let completed_stats = |expected_rows: u64| -> Result<(u64, u64), String> {
            let deadline = Instant::now() + std::time::Duration::from_secs(1);
            loop {
                let rows = nn_eval.num_rows_processed();
                if rows == expected_rows {
                    return Ok((rows, nn_eval.num_batches_processed()));
                }
                if rows > expected_rows || Instant::now() >= deadline {
                    return Err(format!(
                        "NN counters recorded {rows} rows, expected {expected_rows}"
                    ));
                }
                std::thread::yield_now();
            }
        };
        run_rounds(parsed.warmup)?;
        completed_stats(workers as u64 * parsed.warmup as u64)?;
        nn_eval.clear_cache();
        nn_eval.clear_stats();

        let t0 = Instant::now();
        run_rounds(parsed.iterations)?;
        let secs = t0.elapsed().as_secs_f64();

        let (rows, completed_batches) = completed_stats(workers as u64 * parsed.iterations as u64)?;
        if completed_batches == 0 {
            return Err("NN benchmark completed without any recorded backend batches".into());
        }
        let evals_per_s = rows as f64 / secs;
        let per_batch_ms = secs * 1000.0 / completed_batches as f64;
        let avg_b = rows as f64 / completed_batches as f64;
        println!(
            "{b:5} | {per_batch_ms:10.3} | {evals_per_s:9.1} | {rows} | {completed_batches} | {avg_b:.2}   (workers {workers}, physicalMax {max_b})",
        );
    }
    Ok(())
}

/// A setup/timed phase must release its peers even if a backend panics. The
/// caller propagates the error after the barrier, never resuming failed work.
#[cfg(any(feature = "cuda", test))]
fn run_benchmark_phase<T>(
    handle_id: usize,
    phase: &str,
    barrier: &std::sync::Barrier,
    run: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).unwrap_or_else(|payload| {
            let message = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("non-string panic");
            Err(format!("handle {handle_id} {phase} panicked: {message}"))
        });
    barrier.wait();
    result
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
    use serde::Serialize;

    let model_file = parsed.common.get_model_file().map_err(|e| e.to_string())?;
    let cfg = if parsed.common.config.is_empty() {
        let mut cfg =
            kata_core::config::ConfigParser::from_str("nnBackend=cudabackend\n", false, false)
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
    let backend = benchmark_backend(&cfg, true)?;
    let int8_scope = if backend == BenchCudaBackend::Int8 {
        Some(kata_nn::backends::int8::Int8Scope::parse(
            &if cfg.contains("cudaInt8Scope") {
                cfg.get_string("cudaInt8Scope").map_err(|e| e.to_string())?
            } else {
                "ffn".into()
            },
        )?)
    } else {
        None
    };
    let int8_min_ffn_width = if cfg.contains("cudaInt8MinFfnWidth") {
        kata_nn::backends::int8::parse_min_ffn_width(
            &cfg.get_string("cudaInt8MinFfnWidth")
                .map_err(|e| e.to_string())?,
        )?
    } else {
        0
    };
    let bytes = std::fs::read(&model_file).map_err(|e| format!("read model: {e}"))?;
    let model_sha256 = kata_core::hash::sha2::sha256_hex(&bytes);
    if backend != BenchCudaBackend::Quantized {
        kata_nn::tactic_plan::validate_outproj_model_sha(&model_sha256)?;
    }
    let graph = direct_layer_graph(&model_file, &bytes)?;
    let rt = CudaRuntime::new().map_err(|e| format!("CUDA runtime: {e}"))?;
    if backend == BenchCudaBackend::Quantized {
        kata_nn::backends::cuda::validate_quantized_runtime(&rt)?;
        kata_nn::backends::cuda::validate_quantized_model_source(&model_sha256)?;
    }
    rt.validate_residual_algo_request()
        .map_err(|e| format!("CUDA residual tactic: {e}"))?;
    let recipe = if backend == BenchCudaBackend::Quantized {
        Some(if cfg.contains("cudaQuantPlan") {
            let path = cfg.get_string("cudaQuantPlan").map_err(|e| e.to_string())?;
            kata_nn::quantization_plan::load_and_resolve(
                std::path::Path::new(&path),
                &graph,
                &model_sha256,
            )?
        } else {
            kata_nn::quantization_plan::fp16_recipe(&graph, &model_sha256)?
        })
    } else {
        None
    };
    let stream = rt.device.default_stream();
    let model = Arc::new(
        if let Some(recipe) = &recipe {
            CudaModel::load_quantized(&graph, &rt, &stream, recipe)
        } else {
            match int8_scope {
                Some(scope) => {
                    CudaModel::load_int8_selective(&graph, &rt, &stream, scope, int8_min_ffn_width)
                }
                None => CudaModel::load(&graph, &rt, &stream),
            }
        }
        .map_err(|e| format!("load CudaModel: {e}"))?,
    );
    rt.device.synchronize().map_err(|e| format!("sync: {e}"))?;
    if recipe.is_some() {
        kata_nn::backends::cuda::ensure_requested_cuda_capabilities(&model)?;
    }
    let rt_shared = Arc::new(rt);

    #[derive(Debug, Serialize)]
    struct QuantizedStatusValidation {
        warmup_forwards: usize,
        timed_forwards: usize,
        device_snapshot_words: usize,
        timing_boundary: &'static str,
    }
    #[derive(Debug, Serialize)]
    struct HandleSample {
        handle: usize,
        iterations: usize,
        elapsed_ms: f64,
        per_batch_ms: f64,
        nn_evals_per_s: f64,
        cuda_event_ms: Option<Vec<f32>>,
        cuda_event_median_ms: Option<f64>,
        cuda_event_median_nn_evals_per_s: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        quantized_status_validation: Option<QuantizedStatusValidation>,
    }
    #[derive(Debug, Serialize)]
    struct BatchResult {
        batch: usize,
        spatial_input_sha256: String,
        global_input_sha256: String,
        handles: Vec<HandleSample>,
        wall_elapsed_ms: f64,
        wall_nn_evals_per_s: f64,
        sum_median_nn_evals_per_s: Option<f64>,
    }
    #[derive(Debug, Serialize)]
    struct JsonReport {
        schema: u32,
        precision: &'static str,
        int8_scope: Option<&'static str>,
        int8_min_ffn_width: usize,
        int8_ffn_layers: usize,
        quantization_version: Option<&'static str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mxfp8_quantization_version: Option<&'static str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        recipe_sha256: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        graph_sha256: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        quantization_semantics_version: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        identity_scope: Option<&'static str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        validation: Option<&'static str>,
        command: &'static str,
        mode: String,
        kernel_only: bool,
        timing: String,
        model: String,
        model_sha256: String,
        input: String,
        input_description: serde_json::Value,
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
        if let Some(recipe) = &recipe {
            println!(
                "[cuda-quant] model_sha256={} graph_sha256={} recipe_sha256={} identity_scope=precision-recipe-only validation=unverified",
                model_sha256, recipe.graph_sha256, recipe.recipe_sha256
            );
        }
        println!(
            "nnbench({}): model={} handles={} iterations={} warmup={} input={}",
            if kernel_only { "kernel" } else { "direct" },
            model_file,
            handles
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(","),
            parsed.iterations,
            parsed.warmup,
            parsed.input
        );
        println!("batch | wallBatchMs | wall nnEvals/s | per-handle nnEvals/s");
    }

    let mut results = Vec::with_capacity(batches.len());

    for &b in batches {
        // 每个 handle 都复制一份 host 输入；设备输入、workspace 和 stream
        // 在对应线程中独立创建，避免跨流复用可变缓冲。
        let (spatial, global) = batch_inputs(b, &parsed.input);
        let spatial_input_sha256 = f32_input_sha256(&spatial);
        let global_input_sha256 = f32_input_sha256(&global);
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
                scope_results.push(scope.spawn(
                    move || -> Result<(HandleSample, Instant, Instant), String> {
                        // Keep the barrier outside the fallible setup body. Any
                        // allocation/warmup failure must still release peers.
                        let setup_result = run_benchmark_phase(
                            handle_id,
                            "setup",
                            &barrier,
                            || -> Result<_, String> {
                                let stream = rt
                                    .device
                                    .new_stream()
                                    .map_err(|e| format!("handle {handle_id} stream: {e}"))?;
                                let mut d_spatial = unsafe { stream.alloc(spatial.len()) }
                                    .map_err(|e| {
                                        format!("handle {handle_id} alloc spatial: {e}")
                                    })?;
                                let mut d_global = unsafe { stream.alloc(global.len()) }
                                    .map_err(|e| format!("handle {handle_id} alloc global: {e}"))?;
                                let mut ws = CudaWorkspace::new_with_runtime(
                                    rt.clone(), &stream, &model, b,
                                )
                                    .map_err(|e| format!("handle {handle_id} workspace: {e}"))?;
                                if kernel_only {
                                    // Kernel-only mode excludes transfers from the timed
                                    // loop, but still needs deterministic valid inputs.
                                    stream.memcpy_htod(&spatial, &mut d_spatial).map_err(|e| {
                                        format!("handle {handle_id} init spatial: {e}")
                                    })?;
                                    stream.memcpy_htod(&global, &mut d_global).map_err(|e| {
                                        format!("handle {handle_id} init global: {e}")
                                    })?;
                                }
                                let mut run_once = |ws: &mut CudaWorkspace| -> Result<(), String> {
                                    if !kernel_only {
                                        stream.memcpy_htod(&spatial, &mut d_spatial).map_err(
                                            |e| format!("handle {handle_id} htod spatial: {e}"),
                                        )?;
                                        stream.memcpy_htod(&global, &mut d_global).map_err(
                                            |e| format!("handle {handle_id} htod global: {e}"),
                                        )?;
                                    }
                                    model
                                        .apply(&rt, &stream, ws, &d_spatial, &d_global)
                                        .map_err(|e| format!("handle {handle_id} apply: {e}"))?;
                                    if !kernel_only {
                                        ws.to_host(&stream)
                                            .map_err(|e| format!("handle {handle_id} dtoh: {e}"))?;
                                    }
                                    Ok(())
                                };
                                for iteration in 0..parsed.warmup {
                                    run_once(&mut ws)?;
                                    // Preparation may synchronize. Check this forward
                                    // before the next apply clears its sticky status.
                                    // Non-MXFP8 workspaces make this a no-op.
                                    ws.complete_quantized_warmup().map_err(|error| {
                                        format!("handle {handle_id} warmup iteration {}: {error}", iteration + 1)
                                    })?;
                                }
                                stream
                                    .synchronize()
                                    .map_err(|e| format!("handle {handle_id} warmup sync: {e}"))?;
                                let status_ring = if kernel_only && ws.has_mxfp8() {
                                    let words = quantized_status_ring_words(parsed.iterations)?;
                                    Some(stream.alloc_zeros::<u32>(words).map_err(|error| {
                                        format!("handle {handle_id} allocate quantized status ring: {error}")
                                    })?)
                                } else {
                                    None
                                };
                                // Match Fork's per-iteration CUDA events: allocate all
                                // pairs before timing, enqueue the full loop, then sync
                                // once. Never turn this into a sync after every apply.
                                let events = if parsed.timing == "cuda-event" {
                                    let flags =
                                        Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
                                    let mut pairs = Vec::with_capacity(parsed.iterations);
                                    for _ in 0..parsed.iterations {
                                        let begin = rt
                                            .device
                                            .new_event(flags)
                                            .map_err(|e| format!("create begin event: {e}"))?;
                                        let end = rt
                                            .device
                                            .new_event(flags)
                                            .map_err(|e| format!("create end event: {e}"))?;
                                        pairs.push((begin, end));
                                    }
                                    Some(pairs)
                                } else {
                                    None
                                };
                                Ok((stream, d_spatial, d_global, ws, events, status_ring))
                            },
                        );

                        let (stream, mut d_spatial, mut d_global, mut ws, events, mut status_ring) =
                            match setup_result {
                                Ok(value) => value,
                                Err(error) => {
                                    done_barrier.wait();
                                    return Err(error);
                                }
                            };
                        let has_mxfp8 = ws.has_mxfp8();
                        let mut run_once = |ws: &mut CudaWorkspace| -> Result<(), String> {
                            if !kernel_only {
                                stream
                                    .memcpy_htod(&spatial, &mut d_spatial)
                                    .map_err(|e| format!("handle {handle_id} htod spatial: {e}"))?;
                                stream
                                    .memcpy_htod(&global, &mut d_global)
                                    .map_err(|e| format!("handle {handle_id} htod global: {e}"))?;
                            }
                            model
                                .apply(&rt, &stream, ws, &d_spatial, &d_global)
                                .map_err(|e| format!("handle {handle_id} apply: {e}"))?;
                            if !kernel_only {
                                ws.to_host(&stream)
                                    .map_err(|e| format!("handle {handle_id} dtoh: {e}"))?;
                            }
                            Ok(())
                        };
                        let timed_result = run_benchmark_phase(
                            handle_id,
                            "timed",
                            &done_barrier,
                            || -> Result<(Instant, Instant), String> {
                                let started = Instant::now();
                                for i in 0..parsed.iterations {
                                    if let Some(pairs) = &events {
                                        pairs[i]
                                            .0
                                            .record(&stream)
                                            .map_err(|e| format!("record begin event: {e}"))?;
                                    }
                                    run_once(&mut ws)?;
                                    if let Some(pairs) = &events {
                                        pairs[i]
                                            .1
                                            .record(&stream)
                                            .map_err(|e| format!("record end event: {e}"))?;
                                    }
                                    if let Some(ring) = status_ring.as_mut() {
                                        let status = ws.quantized_status().ok_or_else(|| {
                                            format!("handle {handle_id} MXFP8 forward has no status buffer")
                                        })?;
                                        if status.len() != QUANTIZED_STATUS_WORDS {
                                            return Err(format!("handle {handle_id} MXFP8 status word count changed"));
                                        }
                                        let start = i * QUANTIZED_STATUS_WORDS;
                                        let mut snapshot = ring.slice_mut(start..start + QUANTIZED_STATUS_WORDS);
                                        // The event excludes this D2D snapshot, but
                                        // wall time includes it. The same stream
                                        // copies before the next forward reset.
                                        stream.memcpy_dtod(status, &mut snapshot).map_err(|error| {
                                            format!("handle {handle_id} snapshot timed iteration {}: {error}", i + 1)
                                        })?;
                                    }
                                }
                                stream
                                    .synchronize()
                                    .map_err(|e| format!("handle {handle_id} final sync: {e}"))?;
                                Ok((started, Instant::now()))
                            },
                        );
                        // The phase barrier is reached on errors and panics too.
                        // A failed phase returns immediately without using this
                        // runtime again. Event reads/destruction happen only after
                        // the measured endpoints were captured and all peers stop.
                        let (started, ended) = timed_result?;
                        if let Some(ring) = status_ring.as_ref() {
                            // The complete stream (including snapshots) has ended.
                            // One readback here leaves the timed kernel loop free
                            // of host synchronization, allocation and validation.
                            let statuses = stream.clone_dtoh(ring).map_err(|error| {
                                format!("handle {handle_id} read quantized status ring: {error}")
                            })?;
                            validate_quantized_status_ring(
                                &statuses, parsed.iterations, CudaWorkspace::validate_quantized_status,
                            ).map_err(|error| format!("handle {handle_id}: {error}"))?;
                        }
                        let elapsed_ms = ended.duration_since(started).as_secs_f64() * 1000.0;
                        let per_batch_ms = elapsed_ms / parsed.iterations as f64;
                        let cuda_event_ms = events
                            .as_ref()
                            .map(|pairs| {
                                pairs
                                    .iter()
                                    .map(|(begin, end)| {
                                        let ms = begin
                                            .elapsed_ms(end)
                                            .map_err(|e| format!("read elapsed event: {e}"))?;
                                        if !ms.is_finite() || ms <= 0.0 {
                                            return Err(format!("invalid CUDA event time {ms}"));
                                        }
                                        Ok(ms)
                                    })
                                    .collect::<Result<Vec<_>, String>>()
                            })
                            .transpose()?;
                        let cuda_event_median_ms = cuda_event_ms.as_ref().map(|values| {
                            let mut sorted = values.clone();
                            sorted.sort_by(f32::total_cmp);
                            let n = sorted.len();
                            (sorted[(n - 1) / 2] as f64 + sorted[n / 2] as f64) * 0.5
                        });
                        Ok((
                            HandleSample {
                                handle: handle_id,
                                iterations: parsed.iterations,
                                elapsed_ms,
                                per_batch_ms,
                                nn_evals_per_s: b as f64 * 1000.0 / per_batch_ms,
                                cuda_event_ms,
                                cuda_event_median_ms,
                                cuda_event_median_nn_evals_per_s: cuda_event_median_ms
                                    .map(|ms| b as f64 * 1000.0 / ms),
                                quantized_status_validation: has_mxfp8.then(|| QuantizedStatusValidation {
                                    warmup_forwards: parsed.warmup,
                                    timed_forwards: parsed.iterations,
                                    device_snapshot_words: status_ring.as_ref().map_or(0, |ring| ring.len()),
                                    timing_boundary: if kernel_only {
                                        "each CUDA end event precedes its D2D status snapshot; wall includes snapshots and final sync; complete ring DTOH/validation follows measured endpoints"
                                    } else {
                                        "each direct to_host validates that forward's status inside measured wall time"
                                    },
                                }),
                            },
                            started,
                            ended,
                        ))
                    },
                ));
            }
            barrier.wait();
            done_barrier.wait();
            let mut timed_samples = Vec::with_capacity(handles.len());
            for result in scope_results {
                let joined = result
                    .join()
                    .map_err(|_| "nnbench worker panicked".to_string())?;
                timed_samples.push(joined?);
            }
            // Use actual worker endpoints: a worker can start before the main
            // thread resumes at the ready barrier, and can extract/destroy
            // events before the main thread resumes at the done barrier.
            // Neither scheduling race changes this measured forward envelope.
            let started = timed_samples
                .iter()
                .map(|(_, start, _)| *start)
                .min()
                .ok_or_else(|| "nnbench produced no handle samples".to_string())?;
            let ended = timed_samples.iter().map(|(_, _, end)| *end).max().unwrap();
            let wall_elapsed_ms = ended.duration_since(started).as_secs_f64() * 1000.0;
            let samples = timed_samples
                .into_iter()
                .map(|(sample, _, _)| sample)
                .collect::<Vec<_>>();
            Ok((samples, wall_elapsed_ms))
        })?;
        let wall_nn_evals_per_s =
            handles.len() as f64 * b as f64 * parsed.iterations as f64 / (wall_elapsed_ms / 1000.0);
        samples.sort_by_key(|sample| sample.handle);
        let sum_median_nn_evals_per_s = if parsed.timing == "cuda-event" {
            Some(
                samples
                    .iter()
                    .map(|sample| sample.cuda_event_median_nn_evals_per_s.unwrap())
                    .sum(),
            )
        } else {
            None
        };
        if !parsed.json {
            let wall_per_batch_ms = wall_elapsed_ms / (handles.len() * parsed.iterations) as f64;
            println!(
                "{b:5} | {wall_per_batch_ms:12.3} | {wall_nn_evals_per_s:14.1} | {}",
                samples
                    .iter()
                    .map(|sample| format!("h{}={:.1}", sample.handle, sample.nn_evals_per_s))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            if let Some(rate) = sum_median_nn_evals_per_s {
                println!(
                    "      CUDA event median-rate sum: {rate:.1} nnEvals/s (separate from wall throughput)"
                );
            }
            if samples
                .iter()
                .any(|sample| sample.quantized_status_validation.is_some())
            {
                println!(
                    "      MXFP8 status: every warmup/timed forward verified; {}",
                    if kernel_only {
                        "wall includes D2D snapshots; CUDA events exclude snapshots; ring readback follows timing"
                    } else {
                        "direct to_host status validation is included in wall time"
                    }
                );
            }
        }
        results.push(BatchResult {
            batch: b,
            spatial_input_sha256,
            global_input_sha256,
            handles: samples,
            wall_elapsed_ms,
            wall_nn_evals_per_s,
            sum_median_nn_evals_per_s,
        });
    }

    if parsed.json {
        let mut tactics = BTreeMap::new();
        for key in kata_nn::tactic_plan::ALLOWED_TACTIC_KEYS
            .iter()
            .copied()
            .chain(["KATAGO_CUDA_INT8_GEMM_TUNE", "KATAGO_CUDA_INT8_RMS_FUSION"])
        {
            tactics.insert(key.to_string(), kata_nn::tactic_plan::tactic_var(key).ok());
        }
        let uses_int8 = int8_scope.is_some()
            || recipe
                .as_ref()
                .is_some_and(|recipe| recipe.layers.iter().any(|layer| layer.any_int8()));
        let report = JsonReport {
            schema: 1,
            precision: if recipe.is_some() {
                "per-projection-recipe"
            } else if uses_int8 {
                "W8A8-mixed"
            } else {
                "FP16-mixed"
            },
            int8_scope: int8_scope.map(|s| s.name()),
            int8_min_ffn_width,
            int8_ffn_layers: model.int8_ffn_count(),
            quantization_version: uses_int8
                .then_some(kata_nn::backends::int8::QUANTIZATION_VERSION),
            mxfp8_quantization_version: model.has_mxfp8()
                .then_some(kata_nn::backends::mxfp8::QUANTIZATION_VERSION),
            recipe_sha256: recipe.as_ref().map(|recipe| recipe.recipe_sha256.clone()),
            graph_sha256: recipe.as_ref().map(|recipe| recipe.graph_sha256.clone()),
            quantization_semantics_version: recipe
                .as_ref()
                .map(|_| if model.has_mxfp8() {
                    kata_nn::quantization_plan::MXFP8_QUANTIZATION_SEMANTICS_VERSION
                } else {
                    kata_nn::quantization_plan::QUANTIZATION_SEMANTICS_VERSION
                }),
            identity_scope: recipe
                .as_ref()
                .map(|_| "precision recipe only; not an evaluator inference profile"),
            validation: recipe.as_ref().map(|_| "unverified"),
            command: "nnbench",
            mode: parsed.mode.clone(),
            kernel_only,
            timing: parsed.timing.clone(),
            model: model_file,
            model_sha256,
            input: parsed.input.clone(),
            input_description: benchmark_input_description(&parsed.input),
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
fn direct_layer_graph(
    model_file: &str,
    bytes: &[u8],
) -> Result<kata_nn::onnx_parser::LayerGraph, String> {
    let lower_file = model_file.to_ascii_lowercase();
    if lower_file.ends_with(".onnx") {
        return kata_nn::onnx_parser::parse_layer_graph(bytes)
            .map_err(|e| format!("parse ONNX layer graph: {e}"));
    }
    let compressed = lower_file.ends_with(".gz");
    let inner = lower_file.strip_suffix(".gz").unwrap_or(&lower_file);
    let binary = if inner.ends_with(".bin") {
        true
    } else if inner.ends_with(".txt") {
        false
    } else {
        return Err("direct/kernel model must be .onnx, .bin[.gz], or .txt[.gz]".into());
    };
    let desc = kata_nn::model_parser::load_model_from_bytes(bytes, binary, compressed)
        .map_err(|e| format!("parse native model: {e}"))?;
    kata_nn::native_model::lower_model(&desc).map_err(|e| format!("lower native CUDA model: {e}"))
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
fn batch_inputs(b: usize, input: &str) -> (Vec<f32>, Vec<f32>) {
    use kata_nn::inputs::{MiscNNInputParams, fill_row_v7};

    let mut spatial = vec![0.0f32; b * 22 * 19 * 19];
    let mut global = vec![0.0f32; b * 19];
    let mut params = MiscNNInputParams::default();
    if input == "empty" {
        // Fork fills default MiscNNInputParams, then sets NNResultBuf.symmetry
        // and policyOptimism to zero before getOutput. Direct input has no
        // evaluator transform, so explicitly record the same effective values.
        params.symmetry = 0;
        params.policy_optimism = 0.0;
    }
    for i in 0..b {
        let s_off = i * 22 * 19 * 19;
        let g_off = i * 19;
        if input == "empty" && i > 0 {
            spatial.copy_within(0..22 * 19 * 19, s_off);
            global.copy_within(0..19, g_off);
            continue;
        }
        let (board, hist, next) = if input == "empty" {
            use kata_game::board::{Board, P_BLACK};
            use kata_game::history::BoardHistory;
            use kata_game::rules::Rules;
            let board = Board::new(19, 19);
            // Explicit preset, verified against cpp/game/rules.cpp:85. Do not
            // couple this control to future changes in Rules::default(). Fork's
            // model pass-alive preference cannot change areas on an empty board.
            let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::get_tromp_taylorish(), 0);
            (board, hist, P_BLACK)
        } else {
            bench_position(i)
        };
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

#[cfg(feature = "cuda")]
fn f32_input_sha256(values: &[f32]) -> String {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    kata_core::hash::sha2::sha256_hex(&bytes)
}

#[cfg(feature = "cuda")]
fn benchmark_input_description(input: &str) -> serde_json::Value {
    use kata_game::rules::Rules;
    use kata_nn::inputs::MiscNNInputParams;
    let params = MiscNNInputParams::default();
    serde_json::json!({
        "kind": input,
        "board": [19, 19],
        "inputs_version": 7,
        "spatial": "B,22,19,19 NCHW f32",
        "global": "B,19 f32",
        "input_hash_encoding": "SHA256 of contiguous little-endian f32 bytes, before device conversion",
        "position_generation": if input == "empty" {
            "one empty board repeated across the physical batch; Black to play"
        } else {
            "row i: legal random moves using seed nnbench-i, move count i*7%80, alternating Black/White"
        },
        "rules": if input == "empty" { Rules::get_tromp_taylorish().to_json() } else { Rules::default().to_json() },
        "encore_phase": 0,
        "effective_symmetry": 0,
        "policy_optimism": params.policy_optimism,
        "misc_params": {
            "draw_equivalent_wins_for_white": params.draw_equivalent_wins_for_white,
            "conservative_pass_and_is_root": params.conservative_pass_and_is_root,
            "enable_passing_hacks": params.enable_passing_hacks,
            "playout_doubling_advantage": params.playout_doubling_advantage,
            "nn_policy_temperature": params.nn_policy_temperature,
            "avoid_mytdagger_hack": params.avoid_mytdagger_hack,
            "max_history": params.max_history
        },
        "fork_comparison": if input == "empty" {
            "same semantic empty-board control as Fork benchmarknn; cross-implementation feature-byte equality is not asserted"
        } else {
            "different positions from Fork benchmarknn, which repeats one empty board"
        }
    })
}

#[cfg(test)]
mod benchmark_phase_tests {
    use super::run_benchmark_phase;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier, mpsc};
    use std::time::Duration;

    #[test]
    fn quantized_status_history_keeps_early_failure_after_a_valid_last_forward() {
        let statuses = [1, 7, 0, u32::MAX, 0, u32::MAX];
        let error = super::validate_quantized_status_ring(&statuses, 3, |status| {
            if status[0] == 0 && status[1] == u32::MAX {
                Ok(())
            } else {
                Err(format!("flags={} layer={}", status[0], status[1]))
            }
        })
        .unwrap_err();
        assert!(error.contains("timed iteration 1"), "{error}");
        assert!(error.contains("flags=1 layer=7"), "{error}");
        // A bad later projection/forward must also be attributed correctly.
        let statuses = [0, u32::MAX, 4, 11, 0, u32::MAX];
        let error = super::validate_quantized_status_ring(&statuses, 3, |status| {
            if status[0] == 0 {
                Ok(())
            } else {
                Err("nonfinite output".into())
            }
        })
        .unwrap_err();
        assert!(error.contains("timed iteration 2"), "{error}");
    }

    #[test]
    fn quantized_status_history_requires_every_snapshot_and_checked_allocation_size() {
        let statuses = [0, u32::MAX, 0, u32::MAX, 0, u32::MAX];
        let mut checked = 0;
        super::validate_quantized_status_ring(&statuses, 3, |status| {
            assert_eq!(status, &[0, u32::MAX]);
            checked += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(checked, 3);
        for invalid in [&statuses[..4], &statuses[..5], &statuses[..]] {
            let iterations = if invalid.len() == 6 { 2 } else { 3 };
            assert!(
                super::validate_quantized_status_ring(invalid, iterations, |_| {
                    panic!("incomplete or oversized rings must fail before checking any snapshot")
                })
                .is_err()
            );
        }
        assert_eq!(super::quantized_status_ring_words(400).unwrap(), 800);
        assert!(super::quantized_status_ring_words(usize::MAX).is_err());
        assert!(super::quantized_status_ring_words(usize::MAX / 2).is_err());
    }

    #[test]
    fn precision_options_cannot_silently_select_a_different_benchmark_backend() {
        use super::{BenchCudaBackend, benchmark_backend};
        use kata_core::config::ConfigParser;
        let config = |body: &str| ConfigParser::from_str(body, false, false).unwrap();
        // Setup defaults to dummybackend when a user-supplied config omits the
        // backend; eval must reject that rather than label it a CUDA result.
        assert!(benchmark_backend(&config("numSearchThreads=1\n"), false).is_err());
        assert_eq!(
            benchmark_backend(&config("numSearchThreads=1\n"), true).unwrap(),
            BenchCudaBackend::Fp16
        );
        // Both aliases and model-zero precedence must agree with evaluator
        // setup. No model is read and no runtime is initialized by admission.
        for name in ["cudaquant", "cudaquantbackend"] {
            let cfg = config(&format!(
                "nnBackend=cudabackend\nnnBackend0={name}\ncudaQuantPlan=not-opened-by-admission.json\n"
            ));
            assert_eq!(
                benchmark_backend(&cfg, false).unwrap(),
                BenchCudaBackend::Quantized
            );
            assert_eq!(
                benchmark_backend(&cfg, true).unwrap(),
                BenchCudaBackend::Quantized
            );
        }
        for name in ["cuda", "cudabackend", "cudaint8", "cudaint8backend"] {
            let cfg = config(&format!("nnBackend={name}\ncudaQuantPlan=recipe.json\n"));
            for direct in [false, true] {
                assert!(
                    benchmark_backend(&cfg, direct)
                        .unwrap_err()
                        .contains("requires nnBackend=cudaquantbackend")
                );
            }
        }
        for option in [
            "cudaInt8Scope=ffn",
            "cudaInt8MinFfnWidth=384",
            "cudaTacticPlan=fp16.json",
        ] {
            let cfg = config(&format!("nnBackend=cudaquantbackend\n{option}\n"));
            assert!(benchmark_backend(&cfg, false).is_err(), "{option}");
            assert!(benchmark_backend(&cfg, true).is_err(), "{option}");
        }
        let cfg = config("nnBackend=cudaquantbackend\ncudaQuantPlan=\"  \"\n");
        assert!(
            benchmark_backend(&cfg, true)
                .unwrap_err()
                .contains("must not be empty")
        );
        let cfg = config(
            "nnBackend=cudaquantbackend\nnnBackend0=cudabackend\ncudaQuantPlan=recipe.json\n",
        );
        assert!(benchmark_backend(&cfg, true).is_err());
    }

    #[test]
    fn evaluator_identity_cannot_be_claimed_by_a_direct_benchmark() {
        use super::{BenchCudaBackend, benchmark_backend};
        use kata_core::config::ConfigParser;
        let cfg = ConfigParser::from_str(
            "nnBackend=cudaquantbackend\ncudaQuantExpectedProfile=expected-execution\n",
            false,
            false,
        )
        .unwrap();
        assert_eq!(
            benchmark_backend(&cfg, false).unwrap(),
            BenchCudaBackend::Quantized
        );
        let err = benchmark_backend(&cfg, true).unwrap_err();
        assert!(
            err.contains("cannot verify cudaQuantExpectedProfile"),
            "{err}"
        );
        assert!(err.contains("--mode eval"), "{err}");
        // Existing explicitly selected INT8 options remain admitted.
        let cfg = ConfigParser::from_str(
            "nnBackend=cudaint8backend\ncudaInt8Scope=transformer\ncudaInt8MinFfnWidth=384\n",
            false,
            false,
        )
        .unwrap();
        assert_eq!(
            benchmark_backend(&cfg, true).unwrap(),
            BenchCudaBackend::Int8
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn empty_input_is_repeated_black_tromp_taylor_with_explicit_komi() {
        let row_spatial = 22 * 19 * 19;
        let (spatial, global) = super::batch_inputs(14, "empty");
        // This is a real board feature vector, not an all-zero tensor. The
        // always-on board mask and signed komi/rule features match Fork V7.
        assert!(spatial[..19 * 19].iter().all(|&value| value == 1.0));
        assert_eq!(global[5], -7.5 / 20.0);
        assert_eq!(&global[6..9], &[1.0, 0.5, 1.0]);
        assert_eq!(global[9], 0.0); // Area scoring, not territory.
        for row in 1..14 {
            assert_eq!(
                &spatial[..row_spatial],
                &spatial[row * row_spatial..(row + 1) * row_spatial]
            );
            assert_eq!(&global[..19], &global[row * 19..(row + 1) * 19]);
        }
        let (positions, _) = super::batch_inputs(2, "positions");
        assert_ne!(
            &positions[row_spatial..],
            &spatial[row_spatial..2 * row_spatial]
        );
        let description = super::benchmark_input_description("empty");
        assert_eq!(description["effective_symmetry"], 0);
        assert_eq!(description["policy_optimism"], 0.0);
        assert_eq!(description["rules"]["komi"], 7.5);
    }

    #[test]
    fn setup_and_timed_failures_release_peers_without_resuming_failed_work() {
        for failed_phase in ["setup", "timed"] {
            for panic_instead_of_error in [false, true] {
                let ready = Arc::new(Barrier::new(2));
                let done = Arc::new(Barrier::new(2));
                let failed_worker_timed_calls = Arc::new(AtomicUsize::new(0));
                let (tx, rx) = mpsc::channel();
                let mut workers = Vec::new();
                for id in 0..2 {
                    let ready = ready.clone();
                    let done = done.clone();
                    let tx = tx.clone();
                    let timed_calls = failed_worker_timed_calls.clone();
                    workers.push(std::thread::spawn(move || {
                        let phase = |name: &str| -> Result<(), String> {
                            if id == 0 && name == "timed" {
                                timed_calls.fetch_add(1, Ordering::SeqCst);
                            }
                            if id == 0 && name == failed_phase {
                                if panic_instead_of_error {
                                    panic!("injected {name} failure");
                                }
                                return Err(format!("injected {name} error"));
                            }
                            Ok(())
                        };
                        let setup = run_benchmark_phase(id, "setup", &ready, || phase("setup"));
                        let result = match setup {
                            Ok(()) => run_benchmark_phase(id, "timed", &done, || phase("timed")),
                            Err(error) => {
                                done.wait();
                                Err(error)
                            }
                        };
                        tx.send((id, result)).unwrap();
                    }));
                }
                drop(tx);
                for _ in 0..2 {
                    // A regression must fail the test instead of hanging the
                    // test harness forever at a barrier or thread join.
                    let (id, result) = rx
                        .recv_timeout(Duration::from_secs(5))
                        .expect("a failed benchmark phase left a peer blocked");
                    if id == 0 {
                        let error = result.expect_err("injected failure escaped");
                        assert!(error.contains(failed_phase), "{error}");
                        if panic_instead_of_error {
                            assert!(error.contains("panicked"), "{error}");
                        }
                    } else {
                        result.expect("healthy peer should finish");
                    }
                }
                for worker in workers {
                    worker.join().expect("panic must become a phase error");
                }
                assert_eq!(
                    failed_worker_timed_calls.load(Ordering::SeqCst),
                    usize::from(failed_phase == "timed")
                );
            }
        }
    }
}
