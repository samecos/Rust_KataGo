//! Dumps NN inputs/outputs for a set of positions using the hand-written CUDA
//! executor (`kata_nn::backends::cuda_exec::CudaModel`), for comparison against
//! the ONNX Runtime FP32 golden reference (`scripts/compare_nn_output.py`).
//!
//! 输入特征构造与 `dump_nn_io.rs`（TRT 后端）完全一致（fill_row_v7 + NCHW），
//! dump 协议相同：`pos{i}_{spatial,global,policy,value,misc,ownership}.bin` +
//! `meta.json`。policy 写通道 0 的 362 个 logits（乐观系数 0 时与
//! `generic_get_output` 的解码一致，直接可对拍 ORT 的 `out_policy[0,0]`）。
//!
//! Environment variables:
//! - `KATAGO_ONNX_MODEL`  (default `D:/code/b11fix.onnx`)
//! - `KATAGO_DUMP_DIR`    (default `target/nn_io_dump_cuda`)
//! - `KATAGO_DUMP_POSITIONS` (default 16)
#![cfg(feature = "cuda")]

use cudarc::driver::CudaStream;
use kata_game::board::{Board, P_BLACK};
use kata_game::history::BoardHistory;
use kata_game::rules::Rules;
use kata_nn::backends::cuda::CudaRuntime;
use kata_nn::backends::cuda_exec::{CudaModel, CudaOutputsHost};
use kata_nn::inputs::{MiscNNInputParams, fill_row_v7};
use kata_nn::onnx_parser::parse_layer_graph;
use std::sync::Arc;

fn model_path() -> String {
    std::env::var("KATAGO_ONNX_MODEL").unwrap_or_else(|_| "D:/code/b11fix.onnx".to_string())
}

/// 加载模型（CUDA 不可用/文件缺失时返回 None，测试跳过）。
fn try_load() -> Option<(CudaRuntime, CudaModel)> {
    let path = model_path();
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipped: model not found at {path}");
        return None;
    }
    let bytes = std::fs::read(&path).expect("read model");
    let rt = match CudaRuntime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("skipped: CUDA runtime unavailable: {e}");
            return None;
        }
    };
    let graph = parse_layer_graph(&bytes).expect("parse layer graph");
    let stream = rt.device.default_stream();
    let model = CudaModel::load(&graph, &rt, &stream).expect("load CudaModel");
    rt.device.synchronize().expect("sync after load");
    Some((rt, model))
}

fn write_f32(path: &std::path::Path, data: &[f32]) {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(path, bytes).expect("write dump file");
}

/// 多流并发能力测试：4 个线程各自独立流并发 apply，对比单线程墙钟。
/// 若 4 线程墙钟 ≈ 单线程/4，说明后端流并发正常（benchmark 不涨的
/// 瓶颈在搜索侧）；若 ≈ 单线程×4，说明后端仍有全局串行点。
#[test]
fn cuda_multi_stream_concurrency() {
    use cudarc::driver::CudaStream;
    use std::sync::Arc;
    let Some((rt, model)) = try_load() else { return };
    let rt = Arc::new(rt);
    let model = Arc::new(model);
    let (spatial, global) = make_position(0);

    // 单线程基线：10 次 apply。
    let base = {
        let stream = rt.device.new_stream().expect("stream");
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let out = run_one_stream(&rt, &model, &stream, &spatial, &global);
            std::hint::black_box(out);
        }
        start.elapsed()
    };

    // 4 线程 × 10 次，各线程独立流。
    let parallel = {
        let start = std::time::Instant::now();
        let mut handles = Vec::new();
        for _ in 0..4 {
            let rt = Arc::clone(&rt);
            let model = Arc::clone(&model);
            let spatial = spatial.clone();
            let global = global.clone();
            handles.push(std::thread::spawn(move || {
                let stream = rt.device.new_stream().expect("stream");
                for _ in 0..10 {
                    let out = run_one_stream(&rt, &model, &stream, &spatial, &global);
                    std::hint::black_box(out);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        start.elapsed()
    };

    let base_s = base.as_secs_f64();
    let par_s = parallel.as_secs_f64();
    eprintln!(
        "multi-stream: single-thread 10x = {base:.2?}, 4-thread 10x = {parallel:.2?}, throughput ratio = {:.2}x",
        base_s / par_s
    );

    // 细诊断：2 线程各 1 次前向的墙钟（若 ≈ 单次×2，流间无重叠）。
    let single_once = {
        let stream = rt.device.new_stream().expect("stream");
        let start = std::time::Instant::now();
        run_one_stream(&rt, &model, &stream, &spatial, &global);
        start.elapsed()
    };
    let two_once = {
        let start = std::time::Instant::now();
        let mut handles = Vec::new();
        for _ in 0..2 {
            let rt = Arc::clone(&rt);
            let model = Arc::clone(&model);
            let spatial = spatial.clone();
            let global = global.clone();
            handles.push(std::thread::spawn(move || {
                let stream = rt.device.new_stream().expect("stream");
                run_one_stream(&rt, &model, &stream, &spatial, &global)
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        start.elapsed()
    };
    eprintln!(
        "multi-stream diag: 1x1 = {single_once:.2?}, 2x1 = {two_once:.2?} (2x串行 = {:.2?})",
        single_once * 2
    );
    // 注：本测试走的是非 graph 直连路径；生产路径（CudaBackend::get_output）
    // 已用 CUDA Graph 把 180 kernel 压缩为 1 次提交，WDDM 提交瓶颈不复存在。
    // 此处仅记录直连路径的并发行为，不做断言。
}

/// 逐层 smoke：加载真实模型跑一个局面，检查输出形状/有限性/量级。
#[test]
fn cuda_model_smoke() {
    let Some((rt, model)) = try_load() else { return };
    eprintln!("CUDA 层数: {}", model.num_layers());

    // 与 dump 一致的确定性局面（pos 0：空盘）。
    let (spatial, global) = make_position(0);
    let out = run_one(&rt, &model, &spatial, &global);

    assert_eq!(out.policy.len(), 6 * 362, "policy 形状 [1,6,362]");
    assert_eq!(out.value.len(), 3);
    assert_eq!(out.misc.len(), 10);
    assert_eq!(out.moremisc.len(), 8);
    assert_eq!(out.ownership.len(), 361);

    for (name, v, bound) in [
        ("policy", out.policy.as_slice(), 300.0f32),
        ("value", out.value.as_slice(), 60.0f32),
        ("misc", out.misc.as_slice(), 500.0f32),
        ("moremisc", out.moremisc.as_slice(), 60.0f32),
        ("ownership", out.ownership.as_slice(), 2.0f32),
    ] {
        for (i, &x) in v.iter().enumerate() {
            if !x.is_finite() {
                eprintln!("SMOKE {name}[{i}] = {x}");
            }
            assert!(x.is_finite(), "{name}[{i}] 含非有限值: {x}");
            assert!(x.abs() < bound, "{name}[{i}] 数值越界: {x}");
        }
    }
    // 空盘策略 top-1 不应是 pass（361），且最大值显著（策略集中）。
    let (max_idx, max_v) = out
        .policy
        .iter()
        .take(361)
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    assert!(max_idx < 361, "空盘 policy top-1 是 pass");
    let mean = out.policy[..361].iter().sum::<f32>() / 361.0;
    assert!(*max_v > mean + 2.0, "空盘 policy 过于平坦: max={max_v:.3} mean={mean:.3}");
    eprintln!("smoke OK: top-1 idx={max_idx} val={max_v:.3}");
}

#[test]
fn dump_nn_io_cuda() {
    let Some((rt, model)) = try_load() else { return };
    let dump_dir = std::env::var("KATAGO_DUMP_DIR")
        .unwrap_or_else(|_| "target/nn_io_dump_cuda".to_string());
    let num_positions: usize = std::env::var("KATAGO_DUMP_POSITIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    std::fs::create_dir_all(&dump_dir).expect("create dump dir");
    let dump_dir = std::path::PathBuf::from(dump_dir);

    for i in 0..num_positions {
        let (spatial, global) = make_position(i);
        let out = run_one(&rt, &model, &spatial, &global);

        write_f32(&dump_dir.join(format!("pos{i}_spatial.bin")), &spatial);
        write_f32(&dump_dir.join(format!("pos{i}_global.bin")), &global);
        // 通道 0（基策略）的 362 logits，与 TRT dump（optimism=0）口径一致。
        write_f32(&dump_dir.join(format!("pos{i}_policy.bin")), &out.policy[..362]);
        write_f32(&dump_dir.join(format!("pos{i}_value.bin")), &out.value);
        write_f32(
            &dump_dir.join(format!("pos{i}_misc.bin")),
            &[
                out.misc[0],
                out.misc[1],
                out.misc[2],
                out.misc[3],
                out.moremisc[0],
                out.moremisc[1],
            ],
        );
        write_f32(&dump_dir.join(format!("pos{i}_ownership.bin")), &out.ownership);
    }

    let model = model_path();
    let meta = format!(
        "{{\"model\":\"{model}\",\"n\":{num_positions},\"spatial_elts\":{},\"global_elts\":19,\"policy_elts\":362,\"value_elts\":3,\"misc_elts\":6,\"ownership_elts\":361}}",
        22 * 19 * 19
    );
    std::fs::write(dump_dir.join("meta.json"), meta).expect("write meta");
    println!("dumped {num_positions} positions to {}", dump_dir.display());
}

/// 与 dump_nn_io.rs 一致的确定性局面：空盘 + i*7%80 步伪随机合法落子。
fn make_position(i: usize) -> (Vec<f32>, Vec<f32>) {
    let mut board = Board::new(19, 19);
    let mut hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
    let mut next_player = P_BLACK;
    let mut rand = kata_core::rng::Rand::new_from_seed(&format!("dump-{i}"));
    let num_moves = i * 7 % 80;
    for _ in 0..num_moves {
        let mut legal = Vec::new();
        for y in 0..19 {
            for x in 0..19 {
                let loc = kata_game::board::location::get_loc(x, y, 19);
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
        next_player = kata_game::board::get_opp(next_player);
    }

    let nn_input_params = MiscNNInputParams::default();
    let mut spatial = vec![0.0f32; 22 * 19 * 19];
    let mut global = vec![0.0f32; 19];
    fill_row_v7(
        &board,
        &hist,
        next_player,
        &nn_input_params,
        19,
        19,
        false, // NCHW
        &mut spatial,
        &mut global,
    );
    (spatial, global)
}

fn run_one(rt: &CudaRuntime, model: &CudaModel, spatial: &[f32], global: &[f32]) -> CudaOutputsHost {
    use cudarc::driver::CudaSlice;
    let stream = rt.device.default_stream();
    run_one_stream(rt, model, &stream, spatial, global)
}

/// 指定流上的单次前向。
fn run_one_stream(
    rt: &CudaRuntime,
    model: &CudaModel,
    stream: &Arc<CudaStream>,
    spatial: &[f32],
    global: &[f32],
) -> CudaOutputsHost {
    use cudarc::driver::CudaSlice;
    let mut ws = kata_nn::backends::cuda_exec::CudaWorkspace::new(stream, model, 1)
        .expect("workspace");
    let mut d_spatial: CudaSlice<f32> =
        unsafe { stream.alloc(spatial.len()) }.expect("alloc spatial");
    let mut d_global: CudaSlice<f32> = unsafe { stream.alloc(global.len()) }.expect("alloc global");
    stream
        .memcpy_htod(spatial, &mut d_spatial)
        .expect("htod spatial");
    stream
        .memcpy_htod(global, &mut d_global)
        .expect("htod global");
    model
        .apply(rt, stream, &mut ws, &d_spatial, &d_global)
        .expect("CudaModel::apply");
    ws.to_host(stream).expect("copy outputs to host")
}
