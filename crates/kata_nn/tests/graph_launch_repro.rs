//! CUDA graph 捕获/发射回归测试(graph exec 被后续捕获作废 bug)。
//!
//! 背景(2026-08-24 修复,见 cuda.rs pre-capture warm 注释):进程内首次
//! graph 捕获若没有先直连跑一次 forward,其 exec 会被同流后续捕获作废,
//! `cuGraphLaunch` 恒报 `CUDA_ERROR_INVALID_VALUE`(重 upload 无效;驱动
//! lazy module loading × stream capture 交互)。生产修复 = 捕获前无条件
//! warm apply。本测试默认按修复后语义断言全部 launch OK;设
//! `REPRO_NO_WARM=1` 可复现 bug(供驱动行为研究,此时断言关闭)。
//!
//! 环境变量:
//! - `KATAGO_ONNX_MODEL`(默认 `D:/code/b11fix.onnx`),缺失时跳过
//! - `REPRO_NO_WARM=1` 跳过 warm,复现 INVALID_VALUE(诊断用)
//! - `REPRO_BATCHES=1,8` 覆盖测试的 batch 档位
#![cfg(feature = "cuda")]

use kata_nn::backends::cuda::CudaRuntime;
use kata_nn::backends::cuda_exec::CudaModel;
use kata_nn::onnx_parser::parse_layer_graph;

fn load_model() -> Option<(CudaRuntime, CudaModel)> {
    let path =
        std::env::var("KATAGO_ONNX_MODEL").unwrap_or_else(|_| "D:/code/b11fix.onnx".to_string());
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipped: model not found at {path}");
        return None;
    }
    let bytes = std::fs::read(&path).expect("read model");
    let rt = CudaRuntime::new().expect("runtime");
    let graph = parse_layer_graph(&bytes).expect("parse");
    let stream = rt.device.new_stream().expect("stream");
    let model = CudaModel::load(&graph, &rt, &stream).expect("load");
    rt.device.synchronize().expect("sync");
    Some((rt, model))
}

fn flags() -> cudarc::driver::sys::CUgraphInstantiate_flags {
    // 与生产 graph_instantiate_flags() 一致。
    cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY
}

/// 复刻 NNEvaluator warmup 序列(生产拓扑):B1 两 slot → B8 两 slot,
/// 同流捕获、slot0 捕获后立即首发。断言(默认 warm 开启)所有 launch OK。
#[test]
fn graph_warmup_launch_regression() {
    let Some((rt, model)) = load_model() else {
        return;
    };
    let stream = rt.device.new_stream().expect("stream");
    let no_warm = std::env::var("REPRO_NO_WARM").is_ok();
    let assert_ok = !no_warm;

    let batches: Vec<usize> = if let Ok(v) = std::env::var("REPRO_BATCHES") {
        v.split(',').filter_map(|x| x.parse().ok()).collect()
    } else {
        vec![1, 8]
    };

    for phys_batch in batches {
        let n_spatial = phys_batch * 22 * 19 * 19;
        let n_global = phys_batch * 19;
        let mut d_spatial: cudarc::driver::CudaSlice<f32> =
            unsafe { stream.alloc(n_spatial) }.expect("alloc");
        let mut d_global: cudarc::driver::CudaSlice<f32> =
            unsafe { stream.alloc(n_global) }.expect("alloc");
        let zeros = vec![0.0f32; n_spatial];
        let zeros_g = vec![0.0f32; n_global];
        stream.memcpy_htod(&zeros, &mut d_spatial).expect("htod");
        stream.memcpy_htod(&zeros_g, &mut d_global).expect("htod");

        // 生产语义:同一 ws 双 slot 共享(st.ws)。
        let mut ws = kata_nn::backends::cuda_exec::CudaWorkspace::new(&stream, &model, phys_batch)
            .expect("workspace");

        // 修复核心:捕获前直连 warm(生产已无条件执行;此处由开关控制)。
        if !no_warm {
            model
                .apply(&rt, &stream, &mut ws, &d_spatial, &d_global)
                .expect("pre-capture warm apply");
            rt.device.synchronize().expect("warm sync");
        }

        let mut graphs = Vec::new();
        for slot in 0..2 {
            rt.device.synchronize().expect("pre-capture sync");
            stream
                .begin_capture(
                    cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                )
                .expect("begin capture");
            model
                .apply(&rt, &stream, &mut ws, &d_spatial, &d_global)
                .expect("apply in capture");
            let g = stream
                .end_capture(flags())
                .expect("end capture")
                .expect("graph built");
            g.upload().expect("upload");
            graphs.push(g);
            // 生产语义:slot0 捕获完立刻发射一次(warmup get_output)。
            if slot == 0 {
                let r = graphs[0].launch();
                rt.device.synchronize().expect("sync");
                if assert_ok {
                    r.expect("B{phys_batch} slot0 first launch");
                } else {
                    eprintln!("B{phys_batch} slot0 first launch: {r:?}");
                }
            }
        }
        for (round, g) in graphs.iter().enumerate() {
            let r = g.launch();
            rt.device.synchronize().expect("sync");
            if assert_ok {
                r.unwrap_or_else(|e| panic!("B{phys_batch} post-warm launch {round}: {e}"));
            } else {
                eprintln!("B{phys_batch} post-warm launch {round}: {r:?}");
            }
        }
    }
}
