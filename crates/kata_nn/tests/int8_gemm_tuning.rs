//! Opt-in tuning must preserve CPU INT32 results, cached plans across weights,
//! dynamic graph inputs, and the owning stream's output after scratch tuning.
#![cfg(feature = "cuda")]

use cudarc::driver::{CudaStream, sys};
use kata_nn::backends::{
    cuda::{CudaRuntime, f32_to_f16_bits, graph_instantiate_flags},
    cuda_exec::set_capturing,
    int8::{Int8Weight, Int8Workspace, QuantizedWeights},
};
use std::sync::Arc;

fn input_code(row: usize, col: usize, variant: usize) -> i32 {
    if col == 0 {
        127
    } else {
        (((row % 16) * 31 + col * 17 + variant * 53) % 255) as i32 - 127
    }
}

fn weight_code(row: usize, col: usize, variant: usize) -> i32 {
    if col == 0 {
        -127
    } else {
        (((row % 16) * 43 + col * 7 + variant * 101) % 255) as i32 - 127
    }
}

fn check_cpu_dots(
    stream: &Arc<CudaStream>,
    ws: &Int8Workspace,
    rows: usize,
    n: usize,
    k: usize,
    input_variant: usize,
    weight_variant: usize,
) {
    // Exactly representable scales make the independent CPU integer oracle
    // simple. Sixteen distinct rows/channels repeat to bound CPU test cost.
    let oracle: Vec<i32> = (0..256)
        .map(|i| {
            (0..k)
                .map(|j| {
                    input_code(i / 16, j, input_variant) * weight_code(i % 16, j, weight_variant)
                })
                .sum()
        })
        .collect();
    let dots = stream.clone_dtoh(&ws.dots).unwrap();
    for row in 0..rows {
        for col in 0..n {
            assert_eq!(
                dots[row * n + col],
                oracle[(row % 16) * 16 + col % 16],
                "M={rows} N={n} K={k} row={row} col={col}"
            );
        }
    }
    let q = stream.clone_dtoh(&ws.quantized).unwrap();
    let kp = k.div_ceil(16) * 16;
    for row in 0..rows {
        for col in 0..kp {
            assert_eq!(
                q[row * kp + col] as i32,
                if col < k {
                    input_code(row, col, input_variant)
                } else {
                    0
                }
            );
        }
    }
    for scale in stream.clone_dtoh(&ws.scales).unwrap().iter().take(rows) {
        assert_eq!(scale.to_bits(), (1.0f32 / 64.0).to_bits());
    }
}

#[test]
#[ignore = "explicit GPU tuning test; set KATAGO_CUDA_INT8_GEMM_TUNE=1"]
fn tuned_integer_gemm_matches_cpu_and_preserves_dynamic_graphs() {
    assert_eq!(
        kata_nn::tactic_plan::tactic_var("KATAGO_CUDA_INT8_GEMM_TUNE").unwrap(),
        "1"
    );
    let rt = CudaRuntime::new().unwrap();
    let stream = rt.device.new_stream().unwrap();
    let other_stream = rt.device.new_stream().unwrap();
    for (rows, n, k) in [
        (361usize, 2304usize, 384usize),
        (2888, 2320, 512),
        (361, 512, 472),
    ] {
        let stride = k.div_ceil(16) * 16;
        let host_input = |variant| {
            (0..(rows + 1) * stride)
                .map(|i| {
                    f32_to_f16_bits(if i % stride < k {
                        input_code(i / stride, i % stride, variant) as f32 / 64.0
                    } else {
                        31.0
                    })
                })
                .collect::<Vec<_>>()
        };
        let weights = |variant| {
            (0..n * k)
                .map(|i| weight_code(i / k, i % k, variant) as f32 / 256.0)
                .collect::<Vec<_>>()
        };
        let host_weights = weights(0);
        let cpu_weights = QuantizedWeights::new(&host_weights, n, k).unwrap();
        for row in 0..n {
            for col in 0..k {
                assert_eq!(
                    cpu_weights.values[row * stride + col] as i32,
                    weight_code(row, col, 0)
                );
            }
        }
        let weight = Int8Weight::upload(&stream, &host_weights, n, k).unwrap();
        let second_weight = Int8Weight::upload(&stream, &weights(1), n, k).unwrap();
        let mut input = stream.clone_htod(&host_input(0)).unwrap();
        let mut output = stream.alloc_zeros::<u16>((rows + 1) * n).unwrap();
        let mut ws = Int8Workspace::new(&rt, &stream, rows + 1, stride, n).unwrap();
        ws.project_half(&rt, &stream, &input, stride, &weight, &mut output, rows)
            .unwrap();
        check_cpu_dots(&stream, &ws, rows, n, k, 0, 0);
        stream.synchronize().unwrap();
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .unwrap();
        set_capturing(true);
        let result = ws.project_half(&rt, &stream, &input, stride, &weight, &mut output, rows);
        set_capturing(false);
        result.unwrap();
        let graph = stream
            .end_capture(graph_instantiate_flags())
            .unwrap()
            .unwrap();
        stream.memcpy_htod(&host_input(1), &mut input).unwrap();
        graph.launch().unwrap();
        check_cpu_dots(&stream, &ws, rows, n, k, 1, 0);
        // A cached shape is also used by other model layers with new weights.
        ws.project_half(
            &rt,
            &stream,
            &input,
            stride,
            &second_weight,
            &mut output,
            rows,
        )
        .unwrap();
        check_cpu_dots(&stream, &ws, rows, n, k, 1, 1);
        assert!(
            ws.project_half(
                &rt,
                &other_stream,
                &input,
                stride,
                &weight,
                &mut output,
                rows
            )
            .unwrap_err()
            .contains("another CUDA stream")
        );

        // Check the driver's actual capture state without the executor's TLS
        // flag: a cold shape must never start nested capture or host copies.
        stream.synchronize().unwrap();
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .unwrap();
        let cold = ws.project_half(&rt, &stream, &input, stride, &weight, &mut output, rows + 1);
        assert!(cold.unwrap_err().contains("warmed before graph capture"));
        drop(stream.end_capture(graph_instantiate_flags()).unwrap());
        if rows == 361 && n == 2304 {
            // An existing inference graph must also survive later tuning of
            // another shape, which captures many private candidate graphs.
            ws.project_half(&rt, &stream, &input, stride, &weight, &mut output, rows + 1)
                .unwrap();
            check_cpu_dots(&stream, &ws, rows + 1, n, k, 1, 0);
        }
        // Cold capture rejection and new-shape tuning preserve the old graph.
        stream.memcpy_htod(&host_input(0), &mut input).unwrap();
        graph.launch().unwrap();
        check_cpu_dots(&stream, &ws, rows, n, k, 0, 0);
    }
}
