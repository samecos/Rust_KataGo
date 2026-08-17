//! cuBLASLt heuristic 探针(H3 验证):拉取生产 GEMM 形状的全部 heuristic
//! 候选,逐个实测计时,判断 sm120 上 cuBLASLt 13.x 候选池里是否存在显著
//! 更快的(疑 nvjet/tcgen05)算法。
//! 第二个测试(C0):经典 cuBLAS(cublasGemmEx)同形状计时——CUDA 12.8+ 的
//! 经典 cublas 在 sm120 上对部分 FP16 形状会派发 nvjet(tcgen05)kernel,
//! 与 Lt 候选池(已证全是 mma.sync 遗留路径)对照。
//!
//! 运行:`cargo test -p kata_nn --test probe_cublaslt_algos --features cuda --release -- --nocapture`
//! 配合 `CUBLASLT_LOG_LEVEL=5 CUBLASLT_LOG_FILE=<f>` 可把每个候选执行时的
//! algo 描述符(tile/stages/customOption)与排名对上。
#![cfg(feature = "cuda")]

use cudarc::cublaslt::sys;
use cudarc::driver::{DevicePtr, DevicePtrMut};

struct Shape {
    name: &'static str,
    m: u64,
    n: u64,
    k: u64,
    f16out: bool,
}

unsafe fn make_desc_and_layouts(s: &Shape) -> (
    sys::cublasLtMatmulDesc_t,
    sys::cublasLtMatrixLayout_t,
    sys::cublasLtMatrixLayout_t,
    sys::cublasLtMatrixLayout_t,
) {
    let mut desc: sys::cublasLtMatmulDesc_t = std::ptr::null_mut();
    sys::cublasLtMatmulDescCreate(
        &mut desc,
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        sys::cudaDataType_t::CUDA_R_32F,
    );
    let transa: u32 = 1; // CUBLAS_OP_T
    let transb: u32 = 0; // CUBLAS_OP_N
    sys::cublasLtMatmulDescSetAttribute(
        desc,
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
        &transa as *const _ as *const _,
        std::mem::size_of_val(&transa),
    );
    sys::cublasLtMatmulDescSetAttribute(
        desc,
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
        &transb as *const _ as *const _,
        std::mem::size_of_val(&transb),
    );
    let cdtype = if s.f16out {
        sys::cudaDataType_t::CUDA_R_16F
    } else {
        sys::cudaDataType_t::CUDA_R_32F
    };
    let mut a_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut b_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut c_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
    sys::cublasLtMatrixLayoutCreate(&mut a_lay, sys::cudaDataType_t::CUDA_R_16F, s.k, s.n, s.k as i64);
    sys::cublasLtMatrixLayoutCreate(&mut b_lay, sys::cudaDataType_t::CUDA_R_16F, s.k, s.m, s.k as i64);
    sys::cublasLtMatrixLayoutCreate(&mut c_lay, cdtype, s.n, s.m, s.n as i64);
    (desc, a_lay, b_lay, c_lay)
}

#[test]
fn probe_cublaslt_heuristic_pool() {
    let Ok(rt) = kata_nn::backends::cuda::CudaRuntime::new() else {
        eprintln!("skipped: CUDA runtime unavailable");
        return;
    };
    let Some(handle) = rt.cublaslt_handle() else {
        eprintln!("skipped: cublaslt unavailable");
        return;
    };
    let ws_len = rt.cublaslt_workspace_len();
    let stream = rt.device.default_stream();
    let (ws_ptr, _ws_guard) = rt.cublaslt_workspace_ptr(&stream);

    let shapes = [
        Shape { name: "ffn_up", m: 5776, n: 2304, k: 384, f16out: true },
        Shape { name: "qkv", m: 5776, n: 1152, k: 384, f16out: true },
        Shape { name: "ffn_down", m: 5776, n: 384, k: 2304, f16out: false },
    ];

    for s in &shapes {
        unsafe {
            let (desc, a_lay, b_lay, c_lay) = make_desc_and_layouts(s);
            let mut pref: sys::cublasLtMatmulPreference_t = std::ptr::null_mut();
            sys::cublasLtMatmulPreferenceCreate(&mut pref);
            sys::cublasLtMatmulPreferenceSetAttribute(
                pref,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws_len as *const _ as *const _,
                std::mem::size_of_val(&ws_len),
            );
            const REQ: usize = 64;
            let mut heur: Vec<sys::cublasLtMatmulHeuristicResult_t> =
                vec![std::mem::zeroed(); REQ];
            let mut cnt = 0i32;
            let r = sys::cublasLtMatmulAlgoGetHeuristic(
                handle, desc, a_lay, b_lay, c_lay, c_lay, pref,
                REQ as i32, heur.as_mut_ptr(), &mut cnt,
            );
            sys::cublasLtMatmulPreferenceDestroy(pref);
            println!(
                "== {} M={} N={} K={} {}: {:?}, {} results",
                s.name, s.m, s.n, s.k, if s.f16out { "f16" } else { "f32" }, r, cnt
            );
            if r != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                sys::cublasLtMatmulDescDestroy(desc);
                sys::cublasLtMatrixLayoutDestroy(a_lay);
                sys::cublasLtMatrixLayoutDestroy(b_lay);
                sys::cublasLtMatrixLayoutDestroy(c_lay);
                continue;
            }

            // 缓冲:A_cm=B 内存 [k,n](权重);B_cm=A 内存 [k,m](激活);C [n,m]
            let mut d_b = stream.alloc::<u16>((s.k * s.n) as usize).expect("alloc w");
            let mut d_a = stream.alloc::<u16>((s.k * s.m) as usize).expect("alloc x");
            let c_elems = if s.f16out { (s.n * s.m) as usize } else { 0 };
            let mut d_c16 = stream.alloc::<u16>(c_elems.max(1)).expect("alloc c16");
            let mut d_c32 = stream.alloc::<f32>((s.n * s.m) as usize).expect("alloc c32");
            // 填充非零内容,避免 denormal/零捷径影响计时
            let ones16 = vec![0x3C00u16; (s.k * s.n).max(s.k * s.m) as usize];
            stream.memcpy_htod(&ones16[..(s.k * s.n) as usize], &mut d_b).expect("htod w");
            stream.memcpy_htod(&ones16[..(s.k * s.m) as usize], &mut d_a).expect("htod x");

            let alpha = 1.0f32;
            let beta = 0.0f32;
            for (i, h) in heur.iter().take(cnt as usize).enumerate() {
                if h.workspaceSize > ws_len {
                    println!("  #{i:2} skip (ws {} > {ws_len})", h.workspaceSize);
                    continue;
                }
                let mut run_once = || {
                    let (b_ptr, _g1) = d_b.device_ptr(&stream);
                    let (a_ptr, _g2) = d_a.device_ptr(&stream);
                    let rr = if s.f16out {
                        let (c_ptr, _g3) = d_c16.device_ptr_mut(&stream);
                        sys::cublasLtMatmul(
                            handle, desc,
                            &alpha as *const _ as *const _,
                            b_ptr as *const _, a_lay,
                            a_ptr as *const _, b_lay,
                            &beta as *const _ as *const _,
                            c_ptr as *const _, c_lay,
                            c_ptr as *mut _, c_lay,
                            &h.algo,
                            ws_ptr as *mut _, ws_len,
                            stream.cu_stream() as _,
                        )
                    } else {
                        let (c_ptr, _g3) = d_c32.device_ptr_mut(&stream);
                        sys::cublasLtMatmul(
                            handle, desc,
                            &alpha as *const _ as *const _,
                            b_ptr as *const _, a_lay,
                            a_ptr as *const _, b_lay,
                            &beta as *const _ as *const _,
                            c_ptr as *const _, c_lay,
                            c_ptr as *mut _, c_lay,
                            &h.algo,
                            ws_ptr as *mut _, ws_len,
                            stream.cu_stream() as _,
                        )
                    };
                    rr
                };
                // warmup(也触发 JIT/自动调优)
                let mut bad = None;
                for _ in 0..5 {
                    let rr = run_once();
                    if rr != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                        bad = Some(rr);
                        break;
                    }
                }
                if let Some(rr) = bad {
                    println!("  #{i:2} exec failed: {rr:?}");
                    continue;
                }
                stream.synchronize().expect("warmup sync");
                let t0 = std::time::Instant::now();
                for _ in 0..50 {
                    run_once();
                }
                stream.synchronize().expect("timed sync");
                let ms = t0.elapsed().as_secs_f64() * 1000.0 / 50.0;
                println!("  #{i:2} {ms:8.4} ms  waves={:.1} ws={}", h.wavesCount, h.workspaceSize);
            }
            sys::cublasLtMatmulDescDestroy(desc);
            sys::cublasLtMatrixLayoutDestroy(a_lay);
            sys::cublasLtMatrixLayoutDestroy(b_lay);
            sys::cublasLtMatrixLayoutDestroy(c_lay);
        }
    }
}

/// C0:经典 cuBLAS(cublasGemmEx)在 sm120 上同形状计时对照。
/// 列主序映射与 Lt 路径一致:C_cm[N,M] = W[K,N]^T @ X[K,M]。
#[test]
fn probe_classic_cublas_nvjet() {
    use cudarc::cublas::sys as csys;

    let Ok(rt) = kata_nn::backends::cuda::CudaRuntime::new() else {
        eprintln!("skipped: CUDA runtime unavailable");
        return;
    };
    let stream = rt.device.default_stream();
    let shapes = [
        Shape { name: "ffn_up", m: 5776, n: 2304, k: 384, f16out: true },
        Shape { name: "qkv", m: 5776, n: 1152, k: 384, f16out: true },
        Shape { name: "ffn_down", m: 5776, n: 384, k: 2304, f16out: false },
    ];
    unsafe {
        let mut handle: csys::cublasHandle_t = std::ptr::null_mut();
        let r = csys::cublasCreate_v2(&mut handle);
        assert_eq!(r, csys::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        csys::cublasSetStream_v2(handle, stream.cu_stream() as _);
        for s in &shapes {
            let mut d_w = stream.alloc::<u16>((s.k * s.n) as usize).expect("alloc w");
            let mut d_x = stream.alloc::<u16>((s.k * s.m) as usize).expect("alloc x");
            let mut d_c16 = stream.alloc::<u16>((s.n * s.m).max(1) as usize).expect("alloc c16");
            let mut d_c32 = stream.alloc::<f32>((s.n * s.m) as usize).expect("alloc c32");
            let ones = vec![0x3C00u16; (s.k * s.n).max(s.k * s.m) as usize];
            stream.memcpy_htod(&ones[..(s.k * s.n) as usize], &mut d_w).expect("htod w");
            stream.memcpy_htod(&ones[..(s.k * s.m) as usize], &mut d_x).expect("htod x");
            let alpha = 1.0f32;
            let beta = 0.0f32;
            let ctype = if s.f16out { csys::cudaDataType_t::CUDA_R_16F } else { csys::cudaDataType_t::CUDA_R_32F };
            let mut run_once = || {
                let (w_ptr, _g1) = d_w.device_ptr(&stream);
                let (x_ptr, _g2) = d_x.device_ptr(&stream);
                let c_ptr = if s.f16out {
                    let (p, _g3) = d_c16.device_ptr_mut(&stream);
                    p
                } else {
                    let (p, _g3) = d_c32.device_ptr_mut(&stream);
                    p
                };
                csys::cublasGemmEx(
                    handle,
                    csys::cublasOperation_t::CUBLAS_OP_T,
                    csys::cublasOperation_t::CUBLAS_OP_N,
                    s.n as i32, // m(列主序行数)= N
                    s.m as i32, // n(列主序列数)= M
                    s.k as i32,
                    &alpha as *const _ as *const _,
                    w_ptr as *const _, csys::cudaDataType_t::CUDA_R_16F, s.k as i32,
                    x_ptr as *const _, csys::cudaDataType_t::CUDA_R_16F, s.k as i32,
                    &beta as *const _ as *const _,
                    c_ptr as *mut _, ctype, s.n as i32,
                    csys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    csys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
            };
            let rr = run_once();
            if rr != csys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                println!("== {}: classic cublas exec failed: {rr:?}", s.name);
                continue;
            }
            for _ in 0..5 { run_once(); }
            stream.synchronize().expect("warmup sync");
            let t0 = std::time::Instant::now();
            for _ in 0..50 { run_once(); }
            stream.synchronize().expect("timed sync");
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / 50.0;
            println!("== {} M={} N={} K={} {}: classic cublas {ms:.4} ms(Lt 最优对照:ffn_up 0.1303 / qkv 0.0619 / ffn_down 0.1112)",
                s.name, s.m, s.n, s.k, if s.f16out { "f16" } else { "f32" });
        }
        csys::cublasDestroy_v2(handle);
    }
}
