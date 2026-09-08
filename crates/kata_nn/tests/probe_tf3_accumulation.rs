//! Opt-in diagnostic, not a production tactic or an accuracy acceptance test.
//! Run with KATAGO_RUN_ACCUM_PROBE=1 and --features cuda --release -- --nocapture.
//! Identical half inputs isolate output type, compute type, and weight layout.
//! CUDA events exclude allocations, transfers, initialization, and CPU reference.
//! A compute-type contract does not prove the kernel's internal instructions;
//! correlate these measurements with Nsight kernel names/SASS before attributing
//! a hardware throughput difference to FP16 versus FP32 accumulation.
#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::cublas::sys;
use cudarc::driver::{CudaContext, CudaStream, DevicePtr, DevicePtrMut};
use kata_nn::backends::cuda::{f16_to_f32_bits, f32_to_f16_bits};

const M: usize = 16 * 361;
const WARMUP: usize = 50;
const ITERATIONS: usize = 200;
const SAMPLES: usize = 256;

fn check(status: sys::cublasStatus_t, operation: &str) {
    assert_eq!(
        status,
        sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
        "{operation}"
    );
}

struct Handle(sys::cublasHandle_t);

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cublasDestroy_v2(self.0);
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Compute32Out32,
    Compute32Out16,
    Compute16Out16,
}

#[derive(Clone, Copy, Debug)]
enum Layout {
    // C++: weights [K,N] row-major = column-major [N,K], OP_N.
    CppNn,
    // Rust: weights [N,K] row-major = column-major [K,N], OP_T.
    RustTn,
}

fn half_inputs(len: usize, mut seed: u64) -> Vec<u16> {
    (0..len)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let unit = ((seed >> 40) as u32) as f32 / 16_777_216.0;
            f32_to_f16_bits((unit - 0.5) * 0.25)
        })
        .collect()
}

struct Problem {
    n: usize,
    k: usize,
    layout: Layout,
    weights: u64,
    inputs: u64,
    output16: u64,
    output32: u64,
}

fn measure(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    handle: &Handle,
    problem: &Problem,
    mode: Mode,
) -> f32 {
    let alpha32 = 1.0f32;
    let beta32 = 0.0f32;
    let alpha16 = f32_to_f16_bits(1.0);
    let beta16 = f32_to_f16_bits(0.0);
    let (compute_type, alpha, beta) = match mode {
        Mode::Compute16Out16 => (
            sys::cublasComputeType_t::CUBLAS_COMPUTE_16F,
            &alpha16 as *const u16 as *const c_void,
            &beta16 as *const u16 as *const c_void,
        ),
        _ => (
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            &alpha32 as *const f32 as *const c_void,
            &beta32 as *const f32 as *const c_void,
        ),
    };
    let (output, output_type) = match mode {
        Mode::Compute32Out32 => (problem.output32, sys::cudaDataType_t::CUDA_R_32F),
        _ => (problem.output16, sys::cudaDataType_t::CUDA_R_16F),
    };
    let (trans_w, ld_w) = match problem.layout {
        Layout::CppNn => (sys::cublasOperation_t::CUBLAS_OP_N, problem.n),
        Layout::RustTn => (sys::cublasOperation_t::CUBLAS_OP_T, problem.k),
    };
    let run = || unsafe {
        // C_cm[N,M] = W_cm[N,K] @ X_cm[K,M]. beta=0 makes each
        // iteration independent; no output initialization is necessary.
        check(
            sys::cublasGemmEx(
                handle.0,
                trans_w,
                sys::cublasOperation_t::CUBLAS_OP_N,
                problem.n as i32,
                M as i32,
                problem.k as i32,
                alpha,
                problem.weights as *const c_void,
                sys::cudaDataType_t::CUDA_R_16F,
                ld_w as i32,
                problem.inputs as *const c_void,
                sys::cudaDataType_t::CUDA_R_16F,
                problem.k as i32,
                beta,
                output as *mut c_void,
                output_type,
                problem.n as i32,
                compute_type,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            ),
            "cublasGemmEx",
        );
    };
    for _ in 0..WARMUP {
        run();
    }
    stream.synchronize().expect("warmup synchronization");
    let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    let begin = context.new_event(flags).expect("begin event");
    let end = context.new_event(flags).expect("end event");
    begin.record(stream).expect("record begin");
    for _ in 0..ITERATIONS {
        run();
    }
    end.record(stream).expect("record end");
    end.synchronize().expect("synchronize end event");
    begin.elapsed_ms(&end).expect("elapsed events") / ITERATIONS as f32
}

#[test]
fn probe_tf3_accumulation() {
    if std::env::var("KATAGO_RUN_ACCUM_PROBE").as_deref() != Ok("1") {
        eprintln!("skipped: set KATAGO_RUN_ACCUM_PROBE=1 to run CUDA accumulation diagnostic");
        return;
    }
    // Explicit opt-in should fail loudly if CUDA/cuBLAS is unavailable.
    let context = CudaContext::new(0).expect("CUDA device 0");
    let stream = context.default_stream();
    let mut raw_handle = std::ptr::null_mut();
    unsafe {
        check(
            sys::cublasCreate_v2(&mut raw_handle),
            "create cuBLAS handle",
        );
    }
    let handle = Handle(raw_handle);
    let mut version = 0;
    unsafe {
        check(
            sys::cublasSetStream_v2(handle.0, stream.cu_stream() as _),
            "set stream",
        );
        check(
            sys::cublasSetPointerMode_v2(
                handle.0,
                sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST,
            ),
            "host scalar pointers",
        );
        check(
            sys::cublasSetMathMode(handle.0, sys::cublasMath_t::CUBLAS_DEFAULT_MATH),
            "default math",
        );
        check(
            sys::cublasGetVersion_v2(handle.0, &mut version),
            "cuBLAS version",
        );
    }
    println!(
        "TF3 accumulation diagnostic: cuBLAS={version}, M={M}, warmup={WARMUP}, iterations={ITERATIONS}, alpha=1,beta=0; GPU event times, no transfers; CPU references use actual half inputs"
    );

    for (name, n, k) in [
        ("out_projection", 384, 384),
        ("qkv_or_single_ffn", 1152, 384),
        ("packed_dual_ffn", 2304, 384),
        ("ffn_down", 384, 1152),
    ] {
        let x = half_inputs(M * k, 0x5446_3301 ^ k as u64);
        let w = half_inputs(n * k, 0x5446_3302 ^ n as u64);
        let mut w_nn = vec![0u16; n * k];
        for o in 0..n {
            for i in 0..k {
                w_nn[i * n + o] = w[o * k + i];
            }
        }
        let indices: Vec<usize> = (0..SAMPLES)
            .map(|i| ((i * 3571) % M) * n + (i * 1193) % n)
            .collect();
        let reference: Vec<f64> = indices
            .iter()
            .map(|&index| {
                let row = index / n;
                let col = index % n;
                (0..k)
                    .map(|i| {
                        f16_to_f32_bits(x[row * k + i]) as f64
                            * f16_to_f32_bits(w[col * k + i]) as f64
                    })
                    .sum()
            })
            .collect();
        let d_x = stream.clone_htod(&x).expect("upload inputs");
        let d_w_tn = stream.clone_htod(&w).expect("upload TN weights");
        let d_w_nn = stream.clone_htod(&w_nn).expect("upload NN weights");
        let mut d_c16 = unsafe { stream.alloc::<u16>(M * n) }.expect("allocate half output");
        let mut d_c32 = unsafe { stream.alloc::<f32>(M * n) }.expect("allocate float output");
        stream.synchronize().expect("uploads complete");
        for layout in [Layout::CppNn, Layout::RustTn] {
            let modes = [
                Mode::Compute32Out32,
                Mode::Compute32Out16,
                Mode::Compute16Out16,
            ];
            // Forward/reverse order exposes clock drift without changing inputs.
            for round in 0..2 {
                for slot in 0..modes.len() {
                    let mode = modes[if round == 0 {
                        slot
                    } else {
                        modes.len() - 1 - slot
                    }];
                    let ms = {
                        let (inputs, _input_guard) = d_x.device_ptr(&stream);
                        let weights_slice = match layout {
                            Layout::CppNn => &d_w_nn,
                            Layout::RustTn => &d_w_tn,
                        };
                        let (weights, _weight_guard) = weights_slice.device_ptr(&stream);
                        let (output16, _output16_guard) = d_c16.device_ptr_mut(&stream);
                        let (output32, _output32_guard) = d_c32.device_ptr_mut(&stream);
                        measure(
                            &context,
                            &stream,
                            &handle,
                            &Problem {
                                n,
                                k,
                                layout,
                                weights,
                                inputs,
                                output16,
                                output32,
                            },
                            mode,
                        )
                    };
                    let actual: Vec<f64> = match mode {
                        Mode::Compute32Out32 => {
                            let out = stream.memcpy_dtov(&d_c32).expect("read float output");
                            indices.iter().map(|&i| out[i] as f64).collect()
                        }
                        _ => {
                            let out = stream.memcpy_dtov(&d_c16).expect("read half output");
                            indices
                                .iter()
                                .map(|&i| f16_to_f32_bits(out[i]) as f64)
                                .collect()
                        }
                    };
                    assert!(
                        actual.iter().all(|value| value.is_finite()),
                        "non-finite output"
                    );
                    let max_abs = actual
                        .iter()
                        .zip(&reference)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f64, f64::max);
                    let rmse = (actual
                        .iter()
                        .zip(&reference)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        / SAMPLES as f64)
                        .sqrt();
                    let tflops = 2.0 * M as f64 * n as f64 * k as f64 / (ms as f64 * 1.0e9);
                    println!(
                        "{name} M={M} N={n} K={k} layout={layout:?} mode={mode:?} round={round} ms={ms:.6} TFLOPs={tflops:.3} sampleMaxAbs={max_abs:.9} sampleRmse={rmse:.9}"
                    );
                }
            }
        }
    }
}
