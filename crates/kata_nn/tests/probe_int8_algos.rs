//! Offline INT8 heuristic-pool probe. Every candidate must match a CPU INT32
//! oracle before graph/event timing. This does not select production tactics.
#![cfg(feature = "cuda")]
use cudarc::cublaslt::sys;
use cudarc::driver::{DevicePtr, DevicePtrMut};
use kata_nn::backends::cuda::CudaRuntime;

fn check(s: sys::cublasStatus_t) {
    assert_eq!(s, sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
}

#[test]
#[ignore = "offline GPU timing; run explicitly with --ignored --nocapture"]
fn probe_int8_heuristic_pool() {
    const REPEATS: usize = 512;
    let rt = CudaRuntime::new().unwrap();
    let stream = rt.device.new_stream().unwrap();
    let handle = rt.cublaslt_handle().unwrap();
    let ws_len = rt.cublaslt_workspace_len();
    let (ws_ptr, _guard) = rt.cublaslt_workspace_ptr(&stream);
    for m in [361usize, 2888] {
        // B11 up/down and B15 smallest, 472-tail, largest pruned FFNs.
        for (n, k) in [
            (2304usize, 384usize),
            (384, 1152),
            (32, 512),
            (512, 16),
            (944, 512),
            (512, 480),
            (2320, 512),
            (512, 1168),
        ] {
            unsafe {
                let mut desc = std::ptr::null_mut();
                check(sys::cublasLtMatmulDescCreate(
                    &mut desc,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32I,
                    sys::cudaDataType_t::CUDA_R_32I,
                ));
                let transpose = 1u32;
                check(sys::cublasLtMatmulDescSetAttribute(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                    &transpose as *const _ as _,
                    4,
                ));
                let mut a = std::ptr::null_mut();
                let mut b = std::ptr::null_mut();
                let mut c = std::ptr::null_mut();
                check(sys::cublasLtMatrixLayoutCreate(
                    &mut a,
                    sys::cudaDataType_t::CUDA_R_8I,
                    k as u64,
                    n as u64,
                    k as i64,
                ));
                check(sys::cublasLtMatrixLayoutCreate(
                    &mut b,
                    sys::cudaDataType_t::CUDA_R_8I,
                    k as u64,
                    m as u64,
                    k as i64,
                ));
                check(sys::cublasLtMatrixLayoutCreate(
                    &mut c,
                    sys::cudaDataType_t::CUDA_R_32I,
                    n as u64,
                    m as u64,
                    n as i64,
                ));
                let mut pref = std::ptr::null_mut();
                check(sys::cublasLtMatmulPreferenceCreate(&mut pref));
                check(sys::cublasLtMatmulPreferenceSetAttribute(pref, sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_len as *const _ as _, std::mem::size_of_val(&ws_len)));
                let mut heur = vec![std::mem::zeroed::<sys::cublasLtMatmulHeuristicResult_t>(); 16];
                let mut count = 0;
                let mut production = std::mem::zeroed::<sys::cublasLtMatmulHeuristicResult_t>();
                let mut production_count = 0;
                check(sys::cublasLtMatmulAlgoGetHeuristic(
                    handle,
                    desc,
                    a,
                    b,
                    c,
                    c,
                    pref,
                    1,
                    &mut production,
                    &mut production_count,
                ));
                assert_eq!(production_count, 1);
                check(sys::cublasLtMatmulAlgoGetHeuristic(
                    handle,
                    desc,
                    a,
                    b,
                    c,
                    c,
                    pref,
                    heur.len() as i32,
                    heur.as_mut_ptr(),
                    &mut count,
                ));
                check(sys::cublasLtMatmulPreferenceDestroy(pref));
                assert!(count > 0);
                // Query the production baseline separately. Opaque algorithm
                // bytes may differ between heuristic requests; never assume
                // the first element of a larger pool is the production route.
                heur.truncate(count as usize);
                heur.insert(0, production);
                let weight = |row, col| ((row * 13 + col * 7) % 255) as i32 - 127;
                let input = |row, col| ((row * 3 + col * 17) % 255) as i32 - 127;
                let weights: Vec<i8> = (0..n * k)
                    .map(|i| weight((i / k) % 16, i % k) as i8)
                    .collect();
                let inputs: Vec<i8> = (0..m * k)
                    .map(|i| input((i / k) % 16, i % k) as i8)
                    .collect();
                let oracle: Vec<i32> = (0..256)
                    .map(|i| (0..k).map(|j| input(i / 16, j) * weight(i % 16, j)).sum())
                    .collect();
                let dw = stream.clone_htod(&weights).unwrap();
                let dx = stream.clone_htod(&inputs).unwrap();
                let mut dy = stream.alloc_zeros::<i32>(m * n).unwrap();
                // rank 0 = production query, rank 1.. = pool rank 0..
                for (rank, h) in heur.iter().enumerate() {
                    if h.state != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                        || h.workspaceSize > ws_len
                    {
                        continue;
                    }
                    let mut run = || {
                        let (pw, _w) = dw.device_ptr(&stream);
                        let (px, _x) = dx.device_ptr(&stream);
                        let (py, _y) = dy.device_ptr_mut(&stream);
                        let alpha = 1i32;
                        let beta = 0i32;
                        sys::cublasLtMatmul(
                            handle,
                            desc,
                            &alpha as *const _ as _,
                            pw as _,
                            a,
                            px as _,
                            b,
                            &beta as *const _ as _,
                            py as _,
                            c,
                            py as _,
                            c,
                            &h.algo,
                            ws_ptr as _,
                            ws_len,
                            stream.cu_stream() as _,
                        )
                    };
                    let status = run();
                    if status != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                        println!(
                            "INT8_PROBE {{\"m\":{m},\"n\":{n},\"k\":{k},\"rank\":{rank},\"status\":\"{status:?}\"}}"
                        );
                        continue;
                    }
                    for _ in 0..3 {
                        check(run());
                    }
                    stream.synchronize().unwrap();
                    stream.begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED).unwrap();
                    for _ in 0..REPEATS {
                        check(run());
                    }
                    let graph = stream
                        .end_capture(kata_nn::backends::cuda::graph_instantiate_flags())
                        .unwrap()
                        .unwrap();
                    // Full output, not a single spot check, against a nonzero
                    // independent CPU oracle (includes both signs and extremes).
                    for (i, actual) in stream.clone_dtoh(&dy).unwrap().into_iter().enumerate() {
                        assert_eq!(
                            actual,
                            oracle[((i / n) % 16) * 16 + i % n % 16],
                            "rank={rank} index={i}"
                        );
                    }
                    graph.launch().unwrap();
                    stream.synchronize().unwrap();
                    let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
                    let start = rt.device.new_event(flags).unwrap();
                    let end = rt.device.new_event(flags).unwrap();
                    let mut samples = Vec::new();
                    for _ in 0..5 {
                        start.record(&stream).unwrap();
                        graph.launch().unwrap();
                        end.record(&stream).unwrap();
                        end.synchronize().unwrap();
                        samples.push(start.elapsed_ms(&end).unwrap() * 1000.0 / REPEATS as f32);
                    }
                    samples.sort_by(f32::total_cmp);
                    println!(
                        "INT8_PROBE {}",
                        serde_json::json!({"m":m,"n":n,"k":k,"rank":rank,"status":"PASS","median_us":samples[2],"samples_us":samples,"workspace":h.workspaceSize})
                    );
                }
                check(sys::cublasLtMatmulDescDestroy(desc));
                for p in [a, b, c] {
                    check(sys::cublasLtMatrixLayoutDestroy(p));
                }
            }
        }
    }
}
