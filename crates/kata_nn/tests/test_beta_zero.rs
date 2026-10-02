//! Regression for rejected-forward NaN/Inf remaining in a reused GEMM output.
//! Opt in with KATAGO_TEST_BETA_ZERO=1. Calls the embedded production kernels;
//! no test kernel, alternate compiler flags, model, or cuBLAS fallback is used.
//! beta=+/-0 discards C and retains `alpha*acc + +0`; this can normalize a
//! signed zero compared with the former `alpha*acc + beta*negative_C` formula.
#![cfg(feature = "cuda")]

use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg};
use kata_nn::backends::cuda::{CudaRuntime, f16_to_f32_bits, f32_to_f16_bits};
use std::sync::{Arc, Mutex};

static GPU_LOCK: Mutex<()> = Mutex::new(());

fn enabled() -> bool {
    let yes = std::env::var("KATAGO_TEST_BETA_ZERO").as_deref() == Ok("1");
    if !yes {
        eprintln!("skipped beta-zero GPU regression; set KATAGO_TEST_BETA_ZERO=1");
    }
    yes
}

fn upload<T: DeviceRepr>(stream: &Arc<CudaStream>, host: &[T]) -> CudaSlice<T> {
    let device = stream.clone_htod(host).unwrap();
    stream.synchronize().unwrap(); // Keep pageable setup input alive until copied.
    device
}

fn replace<T: DeviceRepr>(stream: &Arc<CudaStream>, device: &mut CudaSlice<T>, host: &[T]) {
    stream.memcpy_htod(host, device).unwrap();
    stream.synchronize().unwrap();
}

fn download<T: DeviceRepr + Default + Clone>(
    stream: &Arc<CudaStream>,
    device: &CudaSlice<T>,
) -> Vec<T> {
    let mut host = vec![T::default(); device.len()];
    stream.memcpy_dtoh(device, &mut host).unwrap();
    stream.synchronize().unwrap();
    host
}

fn poison(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| match i % 6 {
            0 => f32::from_bits(0x7fc0_1234),
            1 => f32::INFINITY,
            2 => f32::NEG_INFINITY,
            3 => -0.0,
            4 => -17.0,
            _ => 0.0,
        })
        .collect()
}

fn bits_equal(label: &str, got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len());
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "{label} index={i}: {g:?} vs {e:?}"
        );
    }
}

// Small dyadic inputs keep every product, partial sum, and reference addition
// exactly representable in FP32; this oracle does not hide reduction errors.
fn inputs(m: usize, n: usize, k: usize) -> (Vec<u16>, Vec<u16>, Vec<f32>) {
    let a: Vec<u16> = (0..m * k)
        .map(|i| f32_to_f16_bits(((i * 7 % 9) as i32 - 4) as f32 / 8.0))
        .collect();
    let b: Vec<u16> = (0..n * k)
        .map(|i| f32_to_f16_bits(((i * 3 % 11) as i32 - 5) as f32 / 8.0))
        .collect();
    let mut product = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f64;
            for inner in 0..k {
                acc += f16_to_f32_bits(a[row * k + inner]) as f64
                    * f16_to_f32_bits(b[col * k + inner]) as f64;
            }
            product[row * n + col] = acc as f32;
        }
    }
    (a, b, product)
}

#[derive(Clone, Copy)]
struct Kernel {
    name: &'static str,
    tile: (usize, usize),
    threads: u32,
}

impl Kernel {
    fn config(self, m: usize, n: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (
                m.div_ceil(self.tile.0) as u32,
                n.div_ceil(self.tile.1) as u32,
                1,
            ),
            block_dim: (self.threads, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn gemm(
    rt: &CudaRuntime,
    stream: &Arc<CudaStream>,
    kernel: Kernel,
    a: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    c: &mut CudaSlice<f32>,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    let f = rt.get_func(kernel.name).expect("production GEMM entry");
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(b)
            .arg(c)
            .arg(&(m as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .arg(&alpha)
            .arg(&beta)
            .launch(kernel.config(m, n))
            .unwrap();
    }
}

#[test]
fn ordinary_tiles_overwrite_nonfinite_c_and_preserve_nonzero_beta() {
    if !enabled() {
        return;
    }
    let _guard = GPU_LOCK.lock().unwrap();
    let rt = CudaRuntime::new().expect("explicit beta-zero CUDA test");
    let stream = rt.device.new_stream().unwrap();
    for kernel in [
        Kernel {
            name: "hgemm_m16n8k16_kernel",
            tile: (64, 64),
            threads: 128,
        },
        Kernel {
            name: "hgemm_v2_kernel",
            tile: (128, 128),
            threads: 256,
        },
        Kernel {
            name: "hgemm_t64_kernel",
            tile: (64, 64),
            threads: 256,
        },
        Kernel {
            name: "hgemm_t32_kernel",
            tile: (32, 32),
            threads: 128,
        },
        Kernel {
            name: "hgemm_t64n32_kernel",
            tile: (64, 32),
            threads: 256,
        },
    ] {
        // Covers vector/scalar stores, column guards, row/tile tails, and K tail.
        for n in [1usize, 131, 136] {
            let (m, k) = (137usize, 80usize);
            let (a, b, product) = inputs(m, n, k);
            let a = upload(&stream, &a);
            let b = upload(&stream, &b);
            let mut c = upload(&stream, &vec![0.0f32; m * n]);
            for alpha in [1.0f32, -0.5, -0.0] {
                replace(&stream, &mut c, &vec![0.0f32; m * n]);
                gemm(&rt, &stream, kernel, &a, &b, &mut c, m, n, k, alpha, 0.0);
                let fresh = download(&stream, &c);
                let expected: Vec<f32> = product.iter().map(|&x| alpha * x + 0.0).collect();
                bits_equal("independent dyadic oracle", &fresh, &expected);
                for beta in [0.0f32, -0.0] {
                    replace(&stream, &mut c, &poison(m * n));
                    gemm(&rt, &stream, kernel, &a, &b, &mut c, m, n, k, alpha, beta);
                    bits_equal(kernel.name, &download(&stream, &c), &fresh);
                }
            }
            let old: Vec<f32> = (0..m * n)
                .map(|i| (i as i32 % 17 - 8) as f32 / 8.0)
                .collect();
            for beta in [1.0f32, -0.5] {
                replace(&stream, &mut c, &old);
                gemm(&rt, &stream, kernel, &a, &b, &mut c, m, n, k, 1.0, beta);
                let expected: Vec<f32> = product
                    .iter()
                    .zip(&old)
                    .map(|(&x, &y)| x + beta * y)
                    .collect();
                bits_equal(
                    "nonzero beta dyadic oracle",
                    &download(&stream, &c),
                    &expected,
                );
            }
            let bad = poison(m * n);
            replace(&stream, &mut c, &bad);
            gemm(&rt, &stream, kernel, &a, &b, &mut c, m, n, k, 1.0, 1.0);
            for (i, &got) in download(&stream, &c).iter().enumerate() {
                let expected = product[i] + bad[i];
                if expected.is_nan() {
                    assert!(got.is_nan(), "{} beta1 NaN {i}", kernel.name);
                } else {
                    assert_eq!(
                        got.to_bits(),
                        expected.to_bits(),
                        "{} beta1 {i}",
                        kernel.name
                    );
                }
            }
            eprintln!(
                "[beta-zero] kernel={} M={m} N={n} K={k} CPU-exact/fresh/poison/beta1=PASS",
                kernel.name
            );
        }
    }
}

#[test]
fn fused_gates_and_swiglu_discard_old_c_only_for_zero_beta() {
    if !enabled() {
        return;
    }
    let _guard = GPU_LOCK.lock().unwrap();
    let rt = CudaRuntime::new().expect("explicit beta-zero CUDA test");
    let stream = rt.device.new_stream().unwrap();
    let (m, n, k) = (137usize, 136usize, 80usize); // fused gate contract requires even N
    let (a, b, product) = inputs(m, n, k);
    let a = upload(&stream, &a);
    let b = upload(&stream, &b);
    let scale = upload(
        &stream,
        &(0..n)
            .map(|i| 0.5 + (i % 3) as f32 / 4.0)
            .collect::<Vec<_>>(),
    );
    let bias = upload(
        &stream,
        &(0..n)
            .map(|i| (i as i32 % 7 - 3) as f32 / 8.0)
            .collect::<Vec<_>>(),
    );
    for name in [
        "hgemm_v2_gatesilu_f32_kernel",
        "hgemm_v2_gatesilu_f16_kernel",
    ] {
        let f = rt.get_func(name).unwrap();
        let mut c = upload(&stream, &vec![0.0f32; m * n]);
        let mut out = upload(&stream, &vec![0u16; m * n]);
        let launch = |c: &mut CudaSlice<f32>, out: &mut CudaSlice<u16>, beta: f32| unsafe {
            stream
                .launch_builder(&f)
                .arg(&a)
                .arg(&b)
                .arg(c)
                .arg(out)
                .arg(&scale)
                .arg(&bias)
                .arg(&(m as i32))
                .arg(&(n as i32))
                .arg(&(k as i32))
                .arg(&1.0f32)
                .arg(&beta)
                .launch(
                    Kernel {
                        name,
                        tile: (128, 128),
                        threads: 256,
                    }
                    .config(m, n),
                )
                .unwrap();
        };
        launch(&mut c, &mut out, 0.0);
        let fresh = download(&stream, &c);
        let fresh_half = download(&stream, &out);
        assert!(fresh.iter().all(|x| x.is_finite()));
        if name.contains("f16") {
            bits_equal("fused gate residual oracle", &fresh, &product);
        }
        for beta in [0.0f32, -0.0] {
            replace(&stream, &mut c, &poison(m * n));
            replace(&stream, &mut out, &vec![0x7e00u16; m * n]);
            launch(&mut c, &mut out, beta);
            bits_equal(name, &download(&stream, &c), &fresh);
            if name.contains("f16") {
                assert_eq!(download(&stream, &out), fresh_half, "{name} half boundary");
            }
        }
        replace(&stream, &mut c, &vec![f32::NAN; m * n]);
        launch(&mut c, &mut out, 1.0);
        assert!(
            download(&stream, &c).iter().all(|x| x.is_nan()),
            "{name} beta1 must retain NaN"
        );
        if name.contains("f16") {
            assert!(
                download(&stream, &out)
                    .iter()
                    // Inspect IEEE binary16 directly: the legacy finite-value
                    // conversion helper maps all exponent-31 values to Inf.
                    .all(|&x| x & 0x7c00 == 0x7c00 && x & 0x03ff != 0)
            );
        }
        eprintln!("[beta-zero] kernel={name} fresh/poison/beta1=PASS");
    }
    let kernel = Kernel {
        name: "hgemm_t64_swiglu_residual_kernel",
        tile: (64, 64),
        threads: 256,
    };
    let (dual, _, _) = inputs(m, n, 2 * k);
    let dual = upload(&stream, &dual);
    let mut c = upload(&stream, &vec![0.0f32; m * n]);
    gemm(&rt, &stream, kernel, &dual, &b, &mut c, m, n, k, 1.0, 0.0);
    let fresh = download(&stream, &c);
    assert!(fresh.iter().all(|x| x.is_finite()));
    for beta in [0.0f32, -0.0] {
        replace(&stream, &mut c, &poison(m * n));
        gemm(&rt, &stream, kernel, &dual, &b, &mut c, m, n, k, 1.0, beta);
        bits_equal(kernel.name, &download(&stream, &c), &fresh);
    }
    replace(&stream, &mut c, &vec![f32::NAN; m * n]);
    gemm(&rt, &stream, kernel, &dual, &b, &mut c, m, n, k, 1.0, 1.0);
    assert!(download(&stream, &c).iter().all(|x| x.is_nan()));
    eprintln!("[beta-zero] kernel={} fresh/poison/beta1=PASS", kernel.name);
}

#[test]
fn splitk_and_fused_rms_reducers_overwrite_poisoned_residual() {
    if !enabled() {
        return;
    }
    let _guard = GPU_LOCK.lock().unwrap();
    let rt = CudaRuntime::new().expect("explicit beta-zero CUDA test");
    let stream = rt.device.new_stream().unwrap();
    let (m, n, k, splits) = (7usize, 384usize, 128usize, 2usize);
    let (a, b, product) = inputs(m, n, k);
    let a = upload(&stream, &a);
    let b = upload(&stream, &b);
    let mut cp = upload(&stream, &vec![f32::NAN; splits * m * n]);
    let partial = rt.get_func("hgemm_t64_splitk2_partial_kernel").unwrap();
    unsafe {
        stream
            .launch_builder(&partial)
            .arg(&a)
            .arg(&b)
            .arg(&mut cp)
            .arg(&(m as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .launch(LaunchConfig {
                grid_dim: (m.div_ceil(64) as u32, n.div_ceil(64) as u32, 2),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
    }
    let scale = upload(&stream, &vec![1.0f32; n]);
    for fused in [false, true] {
        let name = if fused {
            "rms_norm_splitk_kernel"
        } else {
            "splitk_reduce_kernel"
        };
        let f = rt.get_func(name).unwrap();
        let mut c = upload(&stream, &vec![0.0f32; m * n]);
        let mut y = upload(&stream, &vec![0u16; m * n]);
        let launch = |c: &mut CudaSlice<f32>, y: &mut CudaSlice<u16>, beta: f32| unsafe {
            if fused {
                stream
                    .launch_builder(&f)
                    .arg(c)
                    .arg(&cp)
                    .arg(&scale)
                    .arg(y)
                    .arg(&beta)
                    .arg(&1e-6f32)
                    .arg(&(n as i32))
                    .arg(&(m as i32))
                    .arg(&(splits as i32))
                    .launch(LaunchConfig {
                        grid_dim: (m.div_ceil(4) as u32, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            } else {
                stream
                    .launch_builder(&f)
                    .arg(c)
                    .arg(&cp)
                    .arg(&(m as i32))
                    .arg(&(n as i32))
                    .arg(&(splits as i32))
                    .arg(&beta)
                    .launch(LaunchConfig::for_num_elems((m * n) as u32))
                    .unwrap();
            }
        };
        launch(&mut c, &mut y, 0.0);
        let fresh = download(&stream, &c);
        let fresh_half = download(&stream, &y);
        bits_equal("splitK independent dyadic oracle", &fresh, &product);
        if fused {
            assert!(fresh_half.iter().all(|&x| f16_to_f32_bits(x).is_finite()));
        }
        for beta in [0.0f32, -0.0] {
            replace(&stream, &mut c, &poison(m * n));
            replace(&stream, &mut y, &vec![0x7e00u16; m * n]);
            launch(&mut c, &mut y, beta);
            bits_equal(name, &download(&stream, &c), &fresh);
            if fused {
                assert_eq!(download(&stream, &y), fresh_half, "{name} half boundary");
            }
        }
        replace(&stream, &mut c, &vec![f32::NAN; m * n]);
        launch(&mut c, &mut y, 1.0);
        assert!(
            download(&stream, &c).iter().all(|x| x.is_nan()),
            "{name} beta1 must retain NaN"
        );
        if fused {
            assert!(
                download(&stream, &y)
                    .iter()
                    .all(|&x| x & 0x7c00 == 0x7c00 && x & 0x03ff != 0)
            );
        }
        eprintln!("[beta-zero] kernel={name} CPU-exact/fresh/poison/beta1=PASS");
    }
}
