//! CUDA 后端冒烟测试：验证 build.rs → nvcc → PTX 嵌入 → cudarc 加载 → 启动
//! 全链路。需要 `cuda` feature 与可用的 CUDA GPU，否则跳过。
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
