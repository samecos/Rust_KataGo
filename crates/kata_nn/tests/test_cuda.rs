//! CUDA 后端冒烟测试：验证 build.rs → nvcc → PTX 嵌入 → cudarc 加载 → 启动
//! 全链路，以及手写 PTX mma m16n8k16 GEMM 的数值正确性。
//! 需要 `cuda` feature 与可用的 CUDA GPU，否则跳过。
#![cfg(feature = "cuda")]

/// 所有 CUDA 测试共享同一 primary context；capture 与并发 CUDA 活动
/// 互斥（WDDM/CUDA13 限制），故测试全局串行。
static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());


#[test]
fn cuda_pipeline_smoke_f32_add() {
    let _guard = TEST_LOCK.lock().unwrap();
    let Ok(rt) = kata_nn::backends::cuda::CudaRuntime::new() else {
        eprintln!("skipped: CUDA runtime unavailable");
        return;
    };
    eprintln!("CUDA target: {}", rt.target_id);

    let n = 1024usize;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
    let out = rt.f32_add(&a, &b).expect("f32_add_kernel launch");

    for i in 0..n {
        let expect = a[i] + b[i];
        assert!(
            (out[i] - expect).abs() < 1e-5,
            "mismatch at {i}: got {}, expected {expect}",
            out[i]
        );
    }
    eprintln!("f32_add_kernel OK (n={n})");
}

#[test]
fn cuda_attention_row_vs_cpu_reference() {
    let _guard = TEST_LOCK.lock().unwrap();
    let Ok(rt) = kata_nn::backends::cuda::CudaRuntime::new() else {
        eprintln!("skipped: CUDA runtime unavailable");
        return;
    };
    use kata_nn::backends::cuda::{f16_to_f32_bits, f32_to_f16_bits};

    let (s, d) = (64usize, 32usize);
    let mut rng = kata_core::rng::Rand::new_from_seed("attention-test");
    let rand_f = |rng: &mut kata_core::rng::Rand| -> f32 {
        (rng.next_u64() as f64 / u64::MAX as f64 - 0.5) as f32 * 2.0
    };
    let mut q = Vec::with_capacity(s * d);
    let mut k = Vec::with_capacity(s * d);
    let mut v = Vec::with_capacity(s * d);
    for _ in 0..s * d {
        q.push(rand_f(&mut rng));
        k.push(rand_f(&mut rng));
        v.push(rand_f(&mut rng));
    }

    let out = rt
        .attention_row(&q, &k, &v, s, d)
        .expect("attention launch");

    // CPU 参考（f16 舍入输入，fp32 计算）
    let q16: Vec<f32> = q.iter().map(|&x| f16_to_f32_bits(f32_to_f16_bits(x))).collect();
    let k16: Vec<f32> = k.iter().map(|&x| f16_to_f32_bits(f32_to_f16_bits(x))).collect();
    let v16: Vec<f32> = v.iter().map(|&x| f16_to_f32_bits(f32_to_f16_bits(x))).collect();
    let scale = 1.0 / (d as f32).sqrt();
    let mut max_abs = 0.0f32;
    let mut worst = (0usize, 0.0f32, 0.0f32);
    for i in 0..s {
        // softmax
        let mut scores = vec![0.0f32; s];
        let mut m = f32::NEG_INFINITY;
        for j in 0..s {
            let mut dot = 0.0f32;
            for dd in 0..d {
                dot += q16[i * d + dd] * k16[j * d + dd];
            }
            scores[j] = dot * scale;
            m = m.max(scores[j]);
        }
        let mut sum = 0.0f32;
        for j in 0..s {
            scores[j] = (scores[j] - m).exp();
            sum += scores[j];
        }
        for dd in 0..d {
            let mut acc = 0.0f32;
            for j in 0..s {
                acc += scores[j] / sum * v16[j * d + dd];
            }
            let got = out[i * d + dd];
            let err = (got - acc).abs();
            if err > max_abs {
                max_abs = err;
                worst = (i * d + dd, got, acc);
            }
        }
    }
    eprintln!("attention max_abs err = {max_abs:.3e} at idx {} got {} expect {}", worst.0, worst.1, worst.2);
    for dd in 0..4 {
        eprintln!("  row0[{dd}]: got {} expect {}", out[dd], {
            let mut scores = vec![0.0f32; s];
            let mut m = f32::NEG_INFINITY;
            for j in 0..s {
                let mut dot = 0.0f32;
                for ddd in 0..d { dot += q16[ddd] * k16[j * d + ddd]; }
                scores[j] = dot * scale;
                m = m.max(scores[j]);
            }
            let mut sum = 0.0f32;
            for j in 0..s { scores[j] = (scores[j] - m).exp(); sum += scores[j]; }
            let mut acc = 0.0f32;
            for j in 0..s { acc += scores[j] / sum * v16[j * d + dd]; }
            acc
        });
    }
    eprintln!("attention max_abs err = {max_abs:.3e}");
    assert!(max_abs < 0.02, "attention 与 CPU 参考偏差过大: {max_abs:.3e}");
    eprintln!("attention_row_kernel OK");
}

#[test]
fn cuda_hgemm_m16n8k16_vs_cpu_reference() {
    let _guard = TEST_LOCK.lock().unwrap();
    let Ok(rt) = kata_nn::backends::cuda::CudaRuntime::new() else {
        eprintln!("skipped: CUDA runtime unavailable");
        return;
    };

    use kata_nn::backends::cuda::f32_to_f16_bits;
    fn f16_to_f32(bits: u16) -> f32 {
        let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0f32 };
        let e = ((bits >> 10) & 0x1f) as i32;
        let m = (bits & 0x3ff) as u32;
        if e == 0 {
            if m == 0 {
                return 0.0 * sign;
            }
            sign * (m as f32) * 2f32.powi(-24)
        } else if e == 0x1f {
            f32::INFINITY * sign
        } else {
            sign * (1.0 + (m as f32) / 1024.0) * 2f32.powi(e - 15)
        }
    }

    // 三个规模：最小 mma、常规、非 64 对齐（走边界守卫）。
    for (m, n, k) in [(16usize, 8usize, 16usize), (128, 96, 64), (100, 50, 128)] {
        let mut rng = kata_core::rng::Rand::new_from_seed(&format!("hgemm-{m}x{n}x{k}"));
        let rand_f = |rng: &mut kata_core::rng::Rand| -> f32 {
            (rng.next_u64() as f64 / u64::MAX as f64 - 0.5) as f32 * 2.0
        };
        let mut a = Vec::with_capacity(m * k);
        let mut b = Vec::with_capacity(n * k);
        for _ in 0..m * k {
            a.push(rand_f(&mut rng));
        }
        for _ in 0..n * k {
            b.push(rand_f(&mut rng));
        }

        // CPU 参考：输入先按 f16 舍入，f64 累加。
        let a16: Vec<f32> = a.iter().map(|&x| f16_to_f32(f32_to_f16_bits(x))).collect();
        let b16: Vec<f32> = b.iter().map(|&x| f16_to_f32(f32_to_f16_bits(x))).collect();
        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f64;
                for kk in 0..k {
                    s += a16[i * k + kk] as f64 * b16[j * k + kk] as f64;
                }
                c_ref[i * n + j] = s as f32;
            }
        }

        // beta=1 残差路径：C 初值非零。
        let mut c = vec![0.5f32; m * n];
        rt.hgemm_m16n8k16(&a, &b, &mut c, m, n, k, 1.0, 1.0)
            .expect("hgemm launch");

        let mut max_abs = 0.0f32;
        for i in 0..m * n {
            let err = (c[i] - (c_ref[i] + 0.5)).abs();
            max_abs = max_abs.max(err);
        }
        eprintln!("hgemm {m}x{n}x{k}: max_abs err = {max_abs:.3e}");
        assert!(
            max_abs < 0.05,
            "FP16 mma GEMM {m}x{n}x{k} 与 CPU 参考偏差过大: {max_abs:.3e}"
        );
    }
    eprintln!("hgemm_m16n8k16_kernel OK");
}

#[test]
fn cuda_graph_minimal_capture() {
    let _guard = TEST_LOCK.lock().unwrap();
    // capture 与同 context 的并发 CUDA 活动互斥（WDDM/CUDA13 限制，
    // 与生产后端 get_output 的全局锁同理）。

    let Ok(rt) = kata_nn::backends::cuda::CudaRuntime::new() else {
        eprintln!("skipped: CUDA runtime unavailable");
        return;
    };
    use cudarc::driver::PushKernelArg;
    let f = rt.get_func("f32_add_kernel").expect("kernel");
    let stream = rt.device.new_stream().expect("stream");
    let n = 1024usize;
    let mut a: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(n) }.expect("alloc a");
    let mut b: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(n) }.expect("alloc b");
    let mut out: cudarc::driver::CudaSlice<f32> =
        stream.alloc_zeros(n).expect("alloc out");
    let a_h: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_h: Vec<f32> = (0..n).map(|i| 1.0f32).collect();
    stream.memcpy_htod(&a_h, &mut a).expect("htod a");
    stream.memcpy_htod(&b_h, &mut b).expect("htod b");

    use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
    stream.synchronize().expect("presync");
    rt.device.synchronize().expect("dev presync");
    stream
        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)
        .expect("begin");
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(n as u32);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(&a)
            .arg(&b)
            .arg(&mut out)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .expect("launch");
    let flags: CUgraphInstantiate_flags = unsafe { std::mem::transmute(0u32) };
    let graph = stream.end_capture(flags).expect("end").expect("graph");
    graph.upload().expect("upload");
    graph.launch().expect("graph launch");
    stream.synchronize().expect("sync");
    let mut out_h = vec![0.0f32; n];
    stream.memcpy_dtoh(&out, &mut out_h).expect("dtoh");
    stream.synchronize().expect("sync2");
    eprintln!("minimal graph out[0]={} out[5]={} out[1023]={}",
        out_h[0], out_h[5], out_h[1023]);
    assert!((out_h[0] - 1.0).abs() < 1e-4, "out[0] should be 1.0");
    assert!((out_h[5] - 6.0).abs() < 1e-4, "out[5] should be 6.0");
}

#[test]
fn cuda_graph_launch_sync_latency() {
    let _guard = TEST_LOCK.lock().unwrap();
    // capture 与同 context 的并发 CUDA 活动互斥（WDDM/CUDA13 限制，
    // 与生产后端 get_output 的全局锁同理）。

    let Ok(rt) = kata_nn::backends::cuda::CudaRuntime::new() else {
        eprintln!("skipped: CUDA runtime unavailable");
        return;
    };
    use cudarc::driver::PushKernelArg;
    let f = rt.get_func("f32_add_kernel").expect("kernel");
    let stream = rt.device.new_stream().expect("stream");
    let n = 1024usize;
    let mut a: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(n) }.expect("a");
    let mut b: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(n) }.expect("b");
    let mut out: cudarc::driver::CudaSlice<f32> = stream.alloc_zeros(n).expect("out");
    use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
    stream.synchronize().expect("presync");
    stream
        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)
        .expect("begin");
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(n as u32);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(&a)
            .arg(&b)
            .arg(&mut out)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .expect("launch");
    let flags: CUgraphInstantiate_flags = unsafe { std::mem::transmute(0u32) };
    let graph = stream.end_capture(flags).expect("end").expect("graph");
    graph.upload().expect("upload");
    // 热身
    for _ in 0..10 {
        graph.launch().expect("l");
        stream.synchronize().expect("s");
    }
    let t0 = std::time::Instant::now();
    let iters = 1000u32;
    for _ in 0..iters {
        graph.launch().expect("l");
        stream.synchronize().expect("s");
    }
    let dt = t0.elapsed();
    eprintln!(
        "graph launch+sync latency: {:.1} µs/iter ({} iters, total {dt:.2?})",
        dt.as_secs_f64() * 1e6 / iters as f64,
        iters
    );
}
