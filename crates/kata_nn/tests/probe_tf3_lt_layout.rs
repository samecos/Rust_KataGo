//! Opt-in K384 layout diagnostic through the production cuBLASLt entry points.
//! KATAGO_RUN_LT_LAYOUT_PROBE=1 cargo test -p kata_nn --test probe_tf3_lt_layout
//!   --features cuda --release -- --nocapture
//! Explicit TN/NN arguments let both layouts share one runtime and algorithm
//! cache, with identical half operands and unchanged FP32 compute/output types.
//! Allocation/transfers and CPU references are outside the CUDA event interval.
//! This is a candidate gate; full TF3/ONNX accuracy and Worker ABBA remain required.
#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use kata_nn::backends::cuda::{
    CublasLtWeightLayout, CudaRuntime, f16_to_f32_bits, f32_to_f16_bits,
};

const WARMUP: usize = 30;
const ITERATIONS: usize = 120;
const SAMPLES: usize = 96;

#[derive(Clone, Copy, Debug)]
enum Output {
    Half,
    Residual,
    Float,
}

fn half_inputs(len: usize, mut seed: u64) -> Vec<u16> {
    (0..len)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let unit = (seed >> 40) as f32 / 16_777_216.0;
            f32_to_f16_bits((unit - 0.5) * 0.25)
        })
        .collect()
}

fn upload(stream: &Arc<CudaStream>, values: &[u16]) -> CudaSlice<u16> {
    let mut data = stream.alloc_zeros(values.len()).expect("allocate operand");
    stream
        .memcpy_htod(values, &mut data)
        .expect("upload operand");
    data
}

struct Problem {
    m: usize,
    n: usize,
    k: usize,
    inputs: CudaSlice<u16>,
    tn: CudaSlice<u16>,
    nn: CudaSlice<u16>,
    out_half: CudaSlice<u16>,
    out_float: CudaSlice<f32>,
}

impl Problem {
    fn run(
        &mut self,
        rt: &CudaRuntime,
        stream: &Arc<CudaStream>,
        layout: CublasLtWeightLayout,
        output: Output,
    ) {
        let weights = match layout {
            CublasLtWeightLayout::Tn => &self.tn,
            CublasLtWeightLayout::Nn => &self.nn,
        };
        let ran = match output {
            Output::Half => rt.cublaslt_gemm_f16out_with_layout(
                stream,
                &self.inputs,
                weights,
                &mut self.out_half,
                self.m,
                self.n,
                self.k,
                layout,
            ),
            Output::Residual | Output::Float => rt.cublaslt_gemm_with_layout(
                stream,
                &self.inputs,
                weights,
                &mut self.out_float,
                self.m,
                self.n,
                self.k,
                if matches!(output, Output::Residual) {
                    1.0
                } else {
                    0.0
                },
                layout,
            ),
        }
        .expect("production cuBLASLt launch");
        assert!(
            ran,
            "cuBLASLt candidate must run; a hand-written fallback is not a probe result"
        );
    }

    fn read(&self, stream: &Arc<CudaStream>, output: Output) -> Vec<f32> {
        match output {
            Output::Half => {
                let mut host = vec![0u16; self.m * self.n];
                stream
                    .memcpy_dtoh(&self.out_half, &mut host)
                    .expect("read half output");
                host.into_iter().map(f16_to_f32_bits).collect()
            }
            _ => {
                let mut host = vec![0.0f32; self.m * self.n];
                stream
                    .memcpy_dtoh(&self.out_float, &mut host)
                    .expect("read float output");
                host
            }
        }
    }

    fn measure(
        &mut self,
        rt: &CudaRuntime,
        stream: &Arc<CudaStream>,
        layout: CublasLtWeightLayout,
        output: Output,
    ) -> f32 {
        for _ in 0..WARMUP {
            self.run(rt, stream, layout, output);
        }
        stream.synchronize().expect("warmup sync");
        let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
        let begin = rt.device.new_event(flags).expect("begin event");
        let end = rt.device.new_event(flags).expect("end event");
        begin.record(stream).expect("record begin");
        for _ in 0..ITERATIONS {
            self.run(rt, stream, layout, output);
        }
        end.record(stream).expect("record end");
        end.synchronize().expect("finish event");
        begin.elapsed_ms(&end).expect("event elapsed") * 1000.0 / ITERATIONS as f32
    }
}

#[test]
fn probe_tf3_lt_layout() {
    if std::env::var("KATAGO_RUN_LT_LAYOUT_PROBE").as_deref() != Ok("1") {
        eprintln!("skipped: set KATAGO_RUN_LT_LAYOUT_PROBE=1 for the CUDA Lt layout probe");
        return;
    }
    let rt = CudaRuntime::new().expect("CUDA runtime");
    assert!(rt.cublaslt_handle().is_some(), "cuBLASLt required");
    let stream = rt.device.new_stream().expect("probe stream");
    println!(
        "K384 Lt layout probe: FP32 compute; identical half operands; event time includes production descriptor/launch pacing; {WARMUP} warmups, {ITERATIONS} iterations"
    );

    // B4/B8 bracket the conditional nn_k384_b8 candidate's launch threshold;
    // both layouts are still measured explicitly here, not selected by env.
    for m in [361, 4 * 361, 8 * 361, 16 * 361] {
        // outproj, bottleneck up, QKV, and unfused gate/up. DualFFN continues
        // to use its original weights and is deliberately outside this probe.
        for n in [384, 768, 1152, 2304] {
            let k = 384;
            let inputs = half_inputs(m * k, 123 + n as u64);
            let tn = half_inputs(n * k, 456 + m as u64);
            let mut nn = vec![0u16; n * k];
            for row in 0..n {
                for col in 0..k {
                    nn[col * n + row] = tn[row * k + col];
                }
            }
            let residual: Vec<f32> = (0..m * n)
                .map(|i| ((i % 31) as f32 - 15.0) * 0.03125)
                .collect();
            let mut problem = Problem {
                m,
                n,
                k,
                inputs: upload(&stream, &inputs),
                tn: upload(&stream, &tn),
                nn: upload(&stream, &nn),
                out_half: stream.alloc_zeros(m * n).expect("allocate half output"),
                out_float: stream.alloc_zeros(m * n).expect("allocate float output"),
            };
            // Revisit the shape in both directions and interleave output types.
            // This also detects algorithm-cache collisions between FP16 output,
            // FP32 beta=1 output, and the two layouts at identical M/N/K.
            for round in 0..2 {
                for output in [Output::Half, Output::Residual, Output::Float] {
                    let mut previous: Option<Vec<f32>> = None;
                    let layouts = if round == 0 {
                        [CublasLtWeightLayout::Tn, CublasLtWeightLayout::Nn]
                    } else {
                        [CublasLtWeightLayout::Nn, CublasLtWeightLayout::Tn]
                    };
                    for layout in layouts {
                        // Reset before the correctness launch so beta=1 checks
                        // the exact original residual, independent of timing.
                        stream
                            .memcpy_htod(&residual, &mut problem.out_float)
                            .expect("initialize residual");
                        problem.run(&rt, &stream, layout, output);
                        let actual = problem.read(&stream, output);
                        assert!(actual.iter().all(|x| x.is_finite()), "nonfinite output");
                        let tolerance = if matches!(output, Output::Half) {
                            3e-4
                        } else {
                            3e-5
                        };
                        let mut max_error = 0.0f64;
                        for sample in 0..SAMPLES {
                            let row = (sample * 3571 + 17) % m;
                            let col = (sample * 1877 + 23) % n;
                            let index = row * n + col;
                            let mut reference = if matches!(output, Output::Residual) {
                                residual[index] as f64
                            } else {
                                0.0
                            };
                            for reduction in 0..k {
                                reference += f16_to_f32_bits(inputs[row * k + reduction]) as f64
                                    * f16_to_f32_bits(tn[col * k + reduction]) as f64;
                            }
                            let error = (actual[index] as f64 - reference).abs();
                            max_error = max_error.max(error);
                            assert!(
                                error <= tolerance,
                                "M={m} N={n} {layout:?} {output:?}: sample {sample}, got {}, ref {reference}, error {error}",
                                actual[index]
                            );
                        }
                        if let Some(other) = previous.as_ref() {
                            let max_difference = actual
                                .iter()
                                .zip(other)
                                .map(|(a, b)| (*a - *b).abs())
                                .fold(0.0f32, f32::max);
                            assert!(
                                max_difference as f64 <= tolerance * 2.0,
                                "M={m} N={n} {output:?}: layout difference {max_difference}"
                            );
                        }
                        previous = Some(actual);
                        let us = problem.measure(&rt, &stream, layout, output);
                        println!(
                            "M={m} N={n} K={k} round={round} layout={layout:?} output={output:?} us={us:.3} sampled_max_error={max_error:.8}"
                        );
                    }
                }
            }
        }
    }
}
