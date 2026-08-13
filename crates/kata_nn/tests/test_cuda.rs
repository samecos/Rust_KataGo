//! CUDA 后端冒烟测试：验证 build.rs → nvcc → PTX 嵌入 → cudarc 加载 → 启动
//! 全链路，以及手写 PTX mma m16n8k16 GEMM 的数值正确性。
//! 需要 `cuda` feature 与可用的 CUDA GPU，否则跳过。
#![cfg(feature = "cuda")]

#[test]
fn cuda_pipeline_smoke_f32_add() {
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
fn cuda_hgemm_m16n8k16_vs_cpu_reference() {
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
