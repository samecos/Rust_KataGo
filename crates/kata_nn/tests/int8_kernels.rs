//! Independent CPU integer oracle, pruned tails, RNE and graph replay.
#![cfg(feature = "cuda")]

use kata_nn::backends::{
    cuda::{CudaRuntime, f16_to_f32_bits, f32_to_f16_bits},
    cuda_exec::set_capturing,
    int8::{Int8Weight, Int8Workspace, QuantizedWeights},
};

#[test]
fn fused_rms_quantization_matches_original_boundaries_and_graph() {
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    let rt = CudaRuntime::new().unwrap();
    let stream = rt.device.new_stream().unwrap();
    for (rows, mid, hidden) in [
        (1, 384, 1152),
        (361, 512, 16),
        (1025, 384, 472),
        (2888, 512, 1160),
    ] {
        let values: Vec<f32> = (0..rows * mid)
            .map(|i| {
                let v = ((i * 127 + i / mid) % 997) as f32 / 191.0 - 2.5;
                match (i / mid) % 7 {
                    0 => 0.0,
                    1 => v * 1e-18,
                    2 => v * 1e12,
                    _ => v,
                }
            })
            .collect();
        let mut input = stream.clone_htod(&values).unwrap();
        let gamma = stream
            .clone_htod(
                &(0..mid)
                    .map(|i| (i % 101) as f32 / 37.0 - 1.0)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let mut normed = stream.alloc_zeros::<u16>(rows * mid).unwrap();
        let mut reference_q = stream.alloc_zeros::<i8>(rows * mid).unwrap();
        let mut reference_s = stream.alloc_zeros::<f32>(rows).unwrap();
        let mut fused_q = stream.alloc_zeros::<i8>(rows * mid).unwrap();
        let mut fused_s = stream.alloc_zeros::<f32>(rows).unwrap();
        let eps = 1e-5f32;
        let rms = rt
            .get_func(if mid == 384 {
                "rms_norm_f32_w4_kernel"
            } else {
                "rms_norm_f32_w4_generic_kernel"
            })
            .unwrap();
        let quant = rt.get_func("int8_quantize_rows_kernel").unwrap();
        let fused = rt
            .get_func(if mid == 384 {
                "int8_rms_quantize384_kernel"
            } else {
                "int8_rms_quantize512_kernel"
            })
            .unwrap();
        let cfg = LaunchConfig {
            grid_dim: (rows.div_ceil(4) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        // Also verify that nonfinite rows retain NaN scales and zero integers.
        for nonfinite in [false, true] {
            let mut x = values.clone();
            if nonfinite {
                x[0] = f32::INFINITY;
                if rows > 1 {
                    x[mid] = f32::NAN;
                }
            }
            stream.memcpy_htod(&x, &mut input).unwrap();
            unsafe {
                stream
                    .launch_builder(&rms)
                    .arg(&input)
                    .arg(&gamma)
                    .arg(&mut normed)
                    .arg(&eps)
                    .arg(&(mid as i32))
                    .arg(&(rows as i32))
                    .launch(cfg)
                    .unwrap();
                stream
                    .launch_builder(&quant)
                    .arg(&normed)
                    .arg(&mut reference_q)
                    .arg(&mut reference_s)
                    .arg(&(rows as i32))
                    .arg(&(mid as i32))
                    .arg(&(mid as i32))
                    .arg(&(mid as i32))
                    .launch(LaunchConfig {
                        grid_dim: (rows as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
                stream
                    .launch_builder(&fused)
                    .arg(&input)
                    .arg(&gamma)
                    .arg(&mut fused_q)
                    .arg(&mut fused_s)
                    .arg(&eps)
                    .arg(&(rows as i32))
                    .launch(cfg)
                    .unwrap();
            }
            assert_eq!(
                stream.clone_dtoh(&reference_q).unwrap(),
                stream.clone_dtoh(&fused_q).unwrap(),
                "RMS integers {rows}/{mid}"
            );
            for (a, b) in stream
                .clone_dtoh(&reference_s)
                .unwrap()
                .into_iter()
                .zip(stream.clone_dtoh(&fused_s).unwrap())
            {
                assert!(
                    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                    "RMS scale {a} != {b}"
                );
            }
        }
        let weights = |n| {
            (0..n)
                .map(|i| ((i * 7) % 61) as f32 / 123.0 - 0.25)
                .collect::<Vec<_>>()
        };
        let dual =
            Int8Weight::upload(&stream, &weights(2 * hidden * mid), 2 * hidden, mid).unwrap();
        let down = Int8Weight::upload(&stream, &weights(mid * hidden), mid, hidden).unwrap();
        let mut ws =
            Int8Workspace::new(&rt, &stream, rows, hidden.max(mid), (2 * hidden).max(mid)).unwrap();
        let mut residual = stream.clone_htod(&values).unwrap();
        // Warm GEMM descriptors before graph capture, then replay changed input.
        ws.ffn_rms(&rt, &stream, &gamma, eps, &dual, &down, &mut residual, rows)
            .unwrap();
        stream.synchronize().unwrap();
        stream
            .begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .unwrap();
        set_capturing(true);
        let result = ws.ffn_rms(&rt, &stream, &gamma, eps, &dual, &down, &mut residual, rows);
        set_capturing(false);
        result.unwrap();
        let graph = stream
            .end_capture(kata_nn::backends::cuda::graph_instantiate_flags())
            .unwrap()
            .unwrap();
        let changed: Vec<f32> = values.iter().map(|x| x * 0.75 + 0.0625).collect();
        stream.memcpy_htod(&changed, &mut residual).unwrap();
        graph.launch().unwrap();
        let replay = stream.clone_dtoh(&residual).unwrap();
        stream.memcpy_htod(&changed, &mut residual).unwrap();
        unsafe {
            stream
                .launch_builder(&rms)
                .arg(&residual)
                .arg(&gamma)
                .arg(&mut normed)
                .arg(&eps)
                .arg(&(mid as i32))
                .arg(&(rows as i32))
                .launch(cfg)
                .unwrap();
        }
        ws.ffn(&rt, &stream, &normed, &dual, &down, &mut residual, rows)
            .unwrap();
        assert_eq!(
            replay,
            stream.clone_dtoh(&residual).unwrap(),
            "RMS FFN graph {rows}/{mid}/{hidden}"
        );
        let other_stream = rt.device.new_stream().unwrap();
        assert!(
            ws.ffn_rms(
                &rt,
                &other_stream,
                &gamma,
                eps,
                &dual,
                &down,
                &mut residual,
                rows
            )
            .is_err()
        );
        assert!(
            ws.ffn_rms(&rt, &stream, &gamma, eps, &dual, &down, &mut residual, 0)
                .is_err()
        );
    }
}

#[test]
fn weight_axes_zero_channels_and_rounding() {
    let mut w = vec![0.0; 4 * 8];
    w[8..16].copy_from_slice(&[127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, -127.0]);
    w[16..24].fill(0.001);
    w[24..32].fill(-2.0);
    let q = QuantizedWeights::new(&w, 4, 8).unwrap();
    assert_eq!(q.kp, 16);
    assert_eq!(q.scales[0], 1.0);
    assert_eq!(&q.values[16..24], &[127, 0, 2, 2, 0, -2, -2, -127]);
    assert_eq!(&q.values[32..40], &[127; 8]);
    for row in 0..4 {
        assert!(
            q.values[row * 16 + 8..row * 16 + 16]
                .iter()
                .all(|&x| x == 0)
        );
    }
    w[0] = f32::NAN;
    assert!(QuantizedWeights::new(&w, 4, 8).is_err());
    assert!(QuantizedWeights::new(&[], 0, 0).is_err());
}

#[test]
fn integer_dot_products_pruned_tails_and_dynamic_graph_scales() {
    let rt = CudaRuntime::new().expect("CUDA required");
    let stream = rt.device.new_stream().unwrap();
    for (rows, n, k) in [
        (1usize, 16usize, 8usize),
        (3, 48, 24),
        (65, 72, 48),
        (1025, 48, 24),
        (3, 944, 512),
        (2, 512, 472),
        (3, 1536, 512),
        (4, 2304, 384),
    ] {
        let stride = k.div_ceil(16) * 16;
        let mut host = vec![f32_to_f16_bits(31.0); rows * stride];
        for row in 0..rows {
            for col in 0..k {
                host[row * stride + col] = f32_to_f16_bits(if row == 0 {
                    0.0
                } else {
                    ((row * 13 + col * 7) % 113) as f32 / 19.0 - 3.0
                });
            }
        }
        let weights: Vec<f32> = (0..n * k)
            .map(|i| {
                if i < k {
                    0.0
                } else {
                    ((i * 17 + i / k) % 127) as f32 / 127.0 - 0.5
                }
            })
            .collect();
        let cpu = QuantizedWeights::new(&weights, n, k).unwrap();
        let weight = Int8Weight::upload(&stream, &weights, n, k).unwrap();
        let mut input = stream.clone_htod(&host).unwrap();
        let mut ws = Int8Workspace::new(&rt, &stream, rows, stride, n).unwrap();
        let mut output = stream.alloc_zeros::<u16>(rows * n).unwrap();
        let initial: Vec<f32> = (0..rows * n).map(|i| (i % 17) as f32 / 13.0).collect();
        let mut residual = stream.clone_htod(&initial).unwrap();
        ws.project_half(&rt, &stream, &input, stride, &weight, &mut output, rows)
            .unwrap();
        ws.project_residual(&rt, &stream, &input, stride, &weight, &mut residual, rows)
            .unwrap();
        let actual_dot = stream.clone_dtoh(&ws.dots).unwrap();
        let actual_half = stream.clone_dtoh(&output).unwrap();
        let actual_residual = stream.clone_dtoh(&residual).unwrap();
        let actual_q = stream.clone_dtoh(&ws.quantized).unwrap();
        for row in 0..rows {
            let x: Vec<f32> = host[row * stride..row * stride + k]
                .iter()
                .map(|&x| f16_to_f32_bits(x))
                .collect();
            let max = x.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            let scale = if max == 0.0 { 1.0 } else { max / 127.0 };
            let q: Vec<i32> = x
                .iter()
                .map(|&x| (x / scale).round_ties_even().clamp(-127.0, 127.0) as i32)
                .collect();
            for col in k..stride {
                assert_eq!(actual_q[row * stride + col], 0);
            }
            for col in 0..n {
                let dot: i32 = (0..k)
                    .map(|j| q[j] * cpu.values[col * cpu.kp + j] as i32)
                    .sum();
                let i = row * n + col;
                assert_eq!(
                    actual_dot[i], dot,
                    "integer oracle {rows}/{n}/{k} row={row} col={col}"
                );
                let expected = (dot as f32 * scale) * cpu.scales[col];
                assert_eq!(actual_half[i], f32_to_f16_bits(expected));
                assert_eq!(
                    actual_residual[i].to_bits(),
                    (initial[i] + expected).to_bits()
                );
            }
        }
        // Capture uses previously prepared descriptors. Replay must recompute
        // the row scales from new input values, including nonzero first row.
        stream.synchronize().unwrap();
        stream
            .begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .unwrap();
        set_capturing(true);
        let result = ws.project_half(&rt, &stream, &input, stride, &weight, &mut output, rows);
        set_capturing(false);
        result.unwrap();
        let graph = stream
            .end_capture(kata_nn::backends::cuda::graph_instantiate_flags())
            .unwrap()
            .unwrap();
        host.iter_mut().for_each(|v| *v = f32_to_f16_bits(0.125));
        stream.memcpy_htod(&host, &mut input).unwrap();
        graph.launch().unwrap();
        let replay = stream.clone_dtoh(&output).unwrap();
        ws.project_half(&rt, &stream, &input, stride, &weight, &mut output, rows)
            .unwrap();
        assert_eq!(replay, stream.clone_dtoh(&output).unwrap());
        assert_ne!(actual_half, replay, "graph retained stale inputs/scales");
    }
}

#[test]
fn fused_ffn_matches_unfused_half_boundaries_and_graph_replay() {
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    let rt = CudaRuntime::new().unwrap();
    let stream = rt.device.new_stream().unwrap();
    // Include the smallest pruned FFN, 8-but-not-16 tails and full B11/B15.
    for (rows, mid, hidden) in [
        (1, 512, 16),
        (1025, 512, 128),
        (1025, 512, 512),
        (1025, 512, 520),
        (1025, 512, 472),
        (3, 512, 1160),
        (3, 384, 1152),
        (3, 512, 1536),
    ] {
        let x: Vec<u16> = (0..rows * mid)
            .map(|i| {
                f32_to_f16_bits(if i < mid {
                    0.0
                } else {
                    (i % 97) as f32 / 43.0 - 1.0
                })
            })
            .collect();
        let mut input = stream.clone_htod(&x).unwrap();
        let weights = |n| {
            (0..n)
                .map(|i| ((i * 7) % 61) as f32 / 123.0 - 0.25)
                .collect::<Vec<_>>()
        };
        let dual =
            Int8Weight::upload(&stream, &weights(2 * hidden * mid), 2 * hidden, mid).unwrap();
        let down = Int8Weight::upload(&stream, &weights(mid * hidden), mid, hidden).unwrap();
        let mut ws = Int8Workspace::new(&rt, &stream, rows, 3 * mid, 6 * mid).unwrap();
        let mut temporary = stream.alloc_zeros::<u16>(rows * 2 * hidden).unwrap();
        let mut activated = stream.alloc_zeros::<u16>(rows * down.kp).unwrap();
        let mut reference = stream.clone_htod(&vec![0.125f32; rows * mid]).unwrap();
        let mut fused = stream.clone_htod(&vec![0.125f32; rows * mid]).unwrap();
        ws.project_half(&rt, &stream, &input, mid, &dual, &mut temporary, rows)
            .unwrap();
        let swiglu = rt.get_func("swiglu_dual_padded_kernel").unwrap();
        unsafe {
            stream
                .launch_builder(&swiglu)
                .arg(&temporary)
                .arg(&mut activated)
                .arg(&(rows as i32))
                .arg(&(hidden as i32))
                .arg(&(down.kp as i32))
                .launch(LaunchConfig::for_num_elems((rows * down.kp) as u32))
                .unwrap();
        }
        ws.project_residual(
            &rt,
            &stream,
            &activated,
            down.kp,
            &down,
            &mut reference,
            rows,
        )
        .unwrap();
        let expected_q = stream.clone_dtoh(&ws.quantized).unwrap();
        let expected_scale = stream.clone_dtoh(&ws.scales).unwrap();
        ws.ffn(&rt, &stream, &input, &dual, &down, &mut fused, rows)
            .unwrap();
        assert_eq!(
            stream.clone_dtoh(&fused).unwrap(),
            stream.clone_dtoh(&reference).unwrap(),
            "fused FFN {rows}/{mid}/{hidden}"
        );
        let q = stream.clone_dtoh(&ws.quantized).unwrap();
        assert_eq!(q[..rows * down.kp], expected_q[..rows * down.kp]);
        assert_eq!(stream.clone_dtoh(&ws.scales).unwrap(), expected_scale);
        stream.synchronize().unwrap();
        stream
            .begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .unwrap();
        set_capturing(true);
        let result = ws.ffn(&rt, &stream, &input, &dual, &down, &mut fused, rows);
        set_capturing(false);
        result.unwrap();
        let graph = stream
            .end_capture(kata_nn::backends::cuda::graph_instantiate_flags())
            .unwrap()
            .unwrap();
        stream
            .memcpy_htod(&vec![f32_to_f16_bits(0.25); rows * mid], &mut input)
            .unwrap();
        stream
            .memcpy_htod(&vec![0.125f32; rows * mid], &mut fused)
            .unwrap();
        graph.launch().unwrap();
        let replay = stream.clone_dtoh(&fused).unwrap();
        stream
            .memcpy_htod(&vec![0.125f32; rows * mid], &mut fused)
            .unwrap();
        ws.ffn(&rt, &stream, &input, &dual, &down, &mut fused, rows)
            .unwrap();
        assert_eq!(replay, stream.clone_dtoh(&fused).unwrap());
    }
}
