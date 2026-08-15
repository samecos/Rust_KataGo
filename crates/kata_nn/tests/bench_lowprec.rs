//! 低精度 GEMM 微基准（混合精度调研 Stage-1）：
//! 对比 trunk 三类 GEMM 形状在 FP16 / FP8(E4M3) / INT8 下的
//! cuBLASLt 吞吐与精度。SM120 (RTX 5070 Ti) 目标。
//!
//! 运行：
//!   cargo test -p kata_nn --features cuda --release \
//!     --test bench_lowprec -- --ignored --nocapture
//!
//! 布局约定与 cuda.rs 的 cublaslt_exec 相同（列主序映射：
//! A_cm=B [k,n] ld=k OP_T；B_cm=A [k,m] ld=k OP_N；C [n,m] ld=n），
//! 即逻辑计算 C[m,n] = A[m,k] @ B[n,k]^T。
#![cfg(feature = "cuda")]
#![allow(unsafe_op_in_unsafe_fn)]

use cudarc::cublaslt::sys;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut, PushKernelArg};
use kata_nn::backends::cuda::CudaRuntime;
use std::sync::Arc;

/// FP8 E4M3 编码（1s4e3m，bias 7，RNE 近似，饱和到 448；NaN 不处理）。
fn f32_to_e4m3(x: f32) -> u8 {
    if x == 0.0 || !x.is_finite() {
        return if x.is_nan() { 0x7f } else { 0 };
    }
    let sign = if x < 0.0 { 0x80u8 } else { 0 };
    let ax = x.abs().min(448.0);
    // 规格化: ax = m * 2^e, m in [1,2)
    let mut e = ax.log2().floor() as i32;
    let mut m = ax / 2f32.powi(e);
    if m >= 2.0 {
        m /= 2.0;
        e += 1;
    }
    if e < -6 {
        // subnormal: value = mantissa * 2^-9
        let mant = (ax / 2f32.powi(-9)).round() as u32;
        return sign | (mant.min(7) as u8);
    }
    if e > 8 {
        return sign | 0x7e; // clamp to 448
    }
    let exp = (e + 7) as u32;
    let mant = ((m - 1.0) * 8.0).round() as u32;
    let (exp, mant) = if mant == 8 { (exp + 1, 0) } else { (exp, mant) };
    if exp > 15 {
        return sign | 0x7e;
    }
    sign | ((exp as u8) << 3) | (mant as u8)
}

/// E4M3 解码。
fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0xf) as i32;
    let mant = (b & 0x7) as f32;
    if exp == 0 {
        return sign * mant * 2f32.powi(-9);
    }
    if exp == 15 && mant == 7.0 {
        return f32::NAN;
    }
    sign * (1.0 + mant / 8.0) * 2f32.powi(exp - 7)
}

struct LtGemm {
    desc: sys::cublasLtMatmulDesc_t,
    a_lay: sys::cublasLtMatrixLayout_t,
    b_lay: sys::cublasLtMatrixLayout_t,
    c_lay: sys::cublasLtMatrixLayout_t,
    algo: sys::cublasLtMatmulAlgo_t,
}

impl Drop for LtGemm {
    fn drop(&mut self) {
        unsafe {
            sys::cublasLtMatmulDescDestroy(self.desc);
            sys::cublasLtMatrixLayoutDestroy(self.a_lay);
            sys::cublasLtMatrixLayoutDestroy(self.b_lay);
            sys::cublasLtMatrixLayoutDestroy(self.c_lay);
        }
    }
}

/// 建一个 cublasLt GEMM 计划（启发式选算法一次，后续复用）。
/// a_type/b_type/c_type: A/B/C 元素类型；compute: 计算类型。
/// fp8_scales: Some((a_scale_dev, b_scale_dev)) 时挂 scale 指针。
#[allow(clippy::too_many_arguments)]
unsafe fn make_plan(
    rt: &CudaRuntime,
    stream: &Arc<CudaStream>,
    m: usize,
    n: usize,
    k: usize,
    a_type: sys::cudaDataType_t,
    b_type: sys::cudaDataType_t,
    c_type: sys::cudaDataType_t,
    compute: sys::cublasComputeType_t,
    fp8_scales: Option<(&CudaSlice<f32>, &CudaSlice<f32>)>,
) -> Result<LtGemm, String> {
    let st = rt.cublaslt_handle().ok_or("cublasLt unavailable")?;
    let mut desc: sys::cublasLtMatmulDesc_t = std::ptr::null_mut();
    sys::cublasLtMatmulDescCreate(&mut desc, compute, sys::cudaDataType_t::CUDA_R_32F);
    let transa: u32 = 1; // OP_T
    let transb: u32 = 0; // OP_N
    sys::cublasLtMatmulDescSetAttribute(
        desc,
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
        &transa as *const _ as *const _,
        4,
    );
    sys::cublasLtMatmulDescSetAttribute(
        desc,
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
        &transb as *const _ as *const _,
        4,
    );
    if let Some((a_scale, b_scale)) = fp8_scales {
        let (a_ptr, _g) = a_scale.device_ptr(stream);
        let (b_ptr, _g) = b_scale.device_ptr(stream);
        sys::cublasLtMatmulDescSetAttribute(
            desc,
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            &a_ptr as *const _ as *const _,
            8,
        );
        sys::cublasLtMatmulDescSetAttribute(
            desc,
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            &b_ptr as *const _ as *const _,
            8,
        );
    }
    let mut a_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut b_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut c_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
    sys::cublasLtMatrixLayoutCreate(&mut a_lay, b_type, k as u64, n as u64, k as i64);
    sys::cublasLtMatrixLayoutCreate(&mut b_lay, a_type, k as u64, m as u64, k as i64);
    sys::cublasLtMatrixLayoutCreate(&mut c_lay, c_type, n as u64, m as u64, n as i64);
    let mut pref: sys::cublasLtMatmulPreference_t = std::ptr::null_mut();
    sys::cublasLtMatmulPreferenceCreate(&mut pref);
    let ws_size = rt.cublaslt_workspace_len();
    sys::cublasLtMatmulPreferenceSetAttribute(
        pref,
        sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &ws_size as *const _ as *const _,
        8,
    );
    let mut heur: sys::cublasLtMatmulHeuristicResult_t = std::mem::zeroed();
    let mut cnt = 0i32;
    let r = sys::cublasLtMatmulAlgoGetHeuristic(
        st, desc, a_lay, b_lay, c_lay, c_lay, pref, 1, &mut heur, &mut cnt,
    );
    sys::cublasLtMatmulPreferenceDestroy(pref);
    if r != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS || cnt == 0 {
        sys::cublasLtMatmulDescDestroy(desc);
        sys::cublasLtMatrixLayoutDestroy(a_lay);
        sys::cublasLtMatrixLayoutDestroy(b_lay);
        sys::cublasLtMatrixLayoutDestroy(c_lay);
        return Err(format!("heuristic failed: {r:?} cnt={cnt}"));
    }
    Ok(LtGemm {
        desc,
        a_lay,
        b_lay,
        c_lay,
        algo: heur.algo,
    })
}

/// 执行计划。A/B/C 原始设备指针（类型由 plan 的布局决定）。
unsafe fn run_plan(
    rt: &CudaRuntime,
    stream: &Arc<CudaStream>,
    plan: &LtGemm,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
) -> Result<(), String> {
    let st = rt.cublaslt_handle().ok_or("cublasLt unavailable")?;
    let alpha = 1.0f32;
    let beta = 0.0f32;
    let (ws_ptr, _g) = rt.cublaslt_workspace_ptr(stream);
    let r = sys::cublasLtMatmul(
        st,
        plan.desc,
        &alpha as *const _ as *const _,
        b_ptr as *const _,
        plan.a_lay,
        a_ptr as *const _,
        plan.b_lay,
        &beta as *const _ as *const _,
        c_ptr as *const _,
        plan.c_lay,
        c_ptr as *mut _,
        plan.c_lay,
        &plan.algo,
        ws_ptr as *mut _,
        rt.cublaslt_workspace_len(),
        stream.cu_stream() as _,
    );
    if r != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(format!("matmul failed: {r:?}"));
    }
    Ok(())
}

/// f32 → f16 bits（RNE，含 subnormal；本测试数据范围足够）。
fn half_from_f32(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 255 {
        return sign | 0x7c00; // inf/nan → inf
    }
    let e = exp - 127 + 15;
    if e <= 0 {
        // subnormal f16 或 0
        if e < -10 {
            return sign;
        }
        let m = (mant | 0x0080_0000) >> ((14 - e) as u32);
        // RNE
        let rem = (mant | 0x0080_0000) & ((1u32 << ((14 - e) as u32)) - 1);
        let halfway = 1u32 << ((13 - e) as u32);
        let mut m = m;
        if rem > halfway || (rem == halfway && m & 1 == 1) {
            m += 1;
        }
        return sign | (m as u16);
    }
    if e >= 31 {
        return sign | 0x7c00; // 饱和到 inf
    }
    let mut h = sign | ((e as u16) << 10) | ((mant >> 13) as u16);
    // RNE
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h = h.wrapping_add(1);
    }
    h
}

/// f16 bits → f32。
fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as f32;
    if exp == 0 {
        return sign * mant * 2f32.powi(-24);
    }
    if exp == 31 {
        return if mant == 0.0 { sign * f32::INFINITY } else { f32::NAN };
    }
    sign * (1.0 + mant / 1024.0) * 2f32.powi(exp - 15)
}

/// 简单 LCG（可复现）。
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (1u32 << 31) as f32
    }
    /// 近似正态（中心极限,8 样本）。
    fn next_norm(&mut self) -> f32 {
        let s: f32 = (0..8).map(|_| self.next_f32()).sum();
        (s - 4.0) * 0.5
    }
}

#[allow(clippy::too_many_arguments)]
fn bench_one(
    rt: &CudaRuntime,
    stream: &Arc<CudaStream>,
    m: usize,
    n: usize,
    k: usize,
    iters: usize,
) -> Result<(), String> {
    unsafe {
        // --- host 数据（真实分布尺度: 权重/激活均 ~N(0,0.05..0.2)） ---
        let mut rng = Rng(0x1234_5678_9abc_def0 ^ ((m * 131 + n * 7 + k) as u64));
        let a_host: Vec<f32> = (0..m * k).map(|_| rng.next_norm() * 0.15).collect();
        let b_host: Vec<f32> = (0..n * k).map(|_| rng.next_norm() * 0.05).collect();

        // --- FP16 设备缓冲 ---
        let a16_h: Vec<u16> = a_host.iter().map(|&x| half_from_f32(x)).collect();
        let b16_h: Vec<u16> = b_host.iter().map(|&x| half_from_f32(x)).collect();
        let a16 = stream.memcpy_stod(&a16_h).map_err(|e| format!("stod a16: {e}"))?;
        let b16 = stream.memcpy_stod(&b16_h).map_err(|e| e.to_string())?;
        let mut c16: CudaSlice<u16> = stream.alloc_zeros(m * n).map_err(|e| format!("alloc c16: {e}"))?;

        // --- FP8 设备缓冲（per-tensor scale=amax/448） ---
        let a_amax = a_host.iter().fold(0.0f32, |p, &x| p.max(x.abs()));
        let b_amax = b_host.iter().fold(0.0f32, |p, &x| p.max(x.abs()));
        let a_scale = a_amax / 448.0;
        let b_scale = b_amax / 448.0;
        let a8_h: Vec<u8> = a_host.iter().map(|&x| f32_to_e4m3(x / a_scale)).collect();
        let b8_h: Vec<u8> = b_host.iter().map(|&x| f32_to_e4m3(x / b_scale)).collect();
        let a8 = stream.memcpy_stod(&a8_h).map_err(|e| format!("stod a8: {e}"))?;
        let b8 = stream.memcpy_stod(&b8_h).map_err(|e| e.to_string())?;
        let mut c8: CudaSlice<u16> = stream.alloc_zeros(m * n).map_err(|e| e.to_string())?;
        // cublasLt FP8 的 scale 语义: D = A*inv_a_scale... 实际为
        // D = (A_descale)(B_descale) * (a_scale_ptr * b_scale_ptr)。
        // 传 a_scale*b_scale 合成一个不行(两指针分开),分别传 a_scale/b_scale。
        let a_scale_dev = stream
            .memcpy_stod(&[a_scale])
            .map_err(|e| e.to_string())?;
        let b_scale_dev = stream
            .memcpy_stod(&[b_scale])
            .map_err(|e| e.to_string())?;

        // --- INT8 设备缓冲（per-tensor scale=amax/127） ---
        let a8i_h: Vec<i8> = a_host
            .iter()
            .map(|&x| (x / (a_amax / 127.0)).round().clamp(-127.0, 127.0) as i8)
            .collect();
        let b8i_h: Vec<i8> = b_host
            .iter()
            .map(|&x| (x / (b_amax / 127.0)).round().clamp(-127.0, 127.0) as i8)
            .collect();
        let a8i = stream.memcpy_stod(&a8i_h).map_err(|e| e.to_string())?;
        let b8i = stream.memcpy_stod(&b8i_h).map_err(|e| e.to_string())?;
        let mut c32i: CudaSlice<i32> = stream.alloc_zeros(m * n).map_err(|e| e.to_string())?;

        // --- 计划 ---
        let plan16 = make_plan(
            rt, stream, m, n, k,
            sys::cudaDataType_t::CUDA_R_16F,
            sys::cudaDataType_t::CUDA_R_16F,
            sys::cudaDataType_t::CUDA_R_16F,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            None,
        )?;
        let plan8 = make_plan(
            rt, stream, m, n, k,
            sys::cudaDataType_t::CUDA_R_8F_E4M3,
            sys::cudaDataType_t::CUDA_R_8F_E4M3,
            sys::cudaDataType_t::CUDA_R_16F,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            Some((&a_scale_dev, &b_scale_dev)),
        );
        let plan8i = make_plan(
            rt, stream, m, n, k,
            sys::cudaDataType_t::CUDA_R_8I,
            sys::cudaDataType_t::CUDA_R_8I,
            sys::cudaDataType_t::CUDA_R_32I,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32I,
            None,
        );

        // --- 计时闭包 ---
        let time_plan = |plan: &LtGemm, ap: u64, bp: u64, cp: u64| -> Result<f32, String> {
            for _ in 0..10 {
                run_plan(rt, stream, plan, ap, bp, cp)?;
            }
            stream.synchronize().map_err(|e| format!("warmup sync: {e}"))?;
            let ev0 = rt
                .device
                .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|e| format!("ev0 create: {e}"))?;
            let ev1 = rt
                .device
                .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|e| format!("ev1 create: {e}"))?;
            ev0.record(stream).map_err(|e| format!("ev0 record: {e}"))?;
            for _ in 0..iters {
                run_plan(rt, stream, plan, ap, bp, cp)?;
            }
            ev1.record(stream).map_err(|e| format!("ev1 record: {e}"))?;
            stream.synchronize().map_err(|e| format!("final sync: {e}"))?;
            let ms = ev0.elapsed_ms(&ev1).map_err(|e| format!("elapsed: {e}"))?;
            Ok(ms * 1000.0 / iters as f32) // us/iter
        };

        // 取原始设备指针（守卫立即释放；slice 本体活到函数尾，指针有效）。
        let (a16p, b16p, c16p, a8p, b8p, c8p, a8ip, b8ip, c32ip) = {
            let (a16p, _g) = a16.device_ptr(stream);
            let (b16p, _g) = b16.device_ptr(stream);
            let (c16p, _g) = c16.device_ptr_mut(stream);
            let (a8p, _g) = a8.device_ptr(stream);
            let (b8p, _g) = b8.device_ptr(stream);
            let (c8p, _g) = c8.device_ptr_mut(stream);
            let (a8ip, _g) = a8i.device_ptr(stream);
            let (b8ip, _g) = b8i.device_ptr(stream);
            let (c32ip, _g) = c32i.device_ptr_mut(stream);
            (a16p, b16p, c16p, a8p, b8p, c8p, a8ip, b8ip, c32ip)
        };
        let t16 = time_plan(&plan16, a16p, b16p, c16p)?;

        let t8 = match &plan8 {
            Ok(p) => {
                let t = time_plan(p, a8p, b8p, c8p)?;
                format!("{t:8.2}")
            }
            Err(e) => format!("FAIL({e})"),
        };

        let t8i = match &plan8i {
            Ok(p) => {
                let t = time_plan(p, a8ip, b8ip, c32ip)?;
                format!("{t:8.2}")
            }
            Err(e) => format!("FAIL({e})"),
        };

        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let tf16 = flops / (t16 as f64 * 1e-6) / 1e12;
        eprintln!(
            "M={m:6} N={n:5} K={k:5} | fp16 {t16:8.2}us ({tf16:6.1} TF) | fp8 {t8}us | int8 {t8i}us"
        );

        // --- 精度抽查（仅最大 M） ---
        if m >= 2888 {
            if let Ok(p) = &plan8 {
                run_plan(rt, stream, &plan16, a16p, b16p, c16p)?;
                run_plan(rt, stream, p, a8p, b8p, c8p)?;
                stream.synchronize().map_err(|e| e.to_string())?;
                let mut c16_h = vec![0u16; m * n];
                let mut c8_h = vec![0u16; m * n];
                stream.memcpy_dtoh(&c16, &mut c16_h).map_err(|e| e.to_string())?;
                stream.memcpy_dtoh(&c8, &mut c8_h).map_err(|e| e.to_string())?;
                let mut num = 0.0f64;
                let mut den = 0.0f64;
                let mut max_rel = 0.0f32;
                for i in 0..m * n {
                    let x = half_to_f32(c16_h[i]) as f64;
                    let y = half_to_f32(c8_h[i]) as f64;
                    num += (x - y) * (x - y);
                    den += x * x;
                    let r = ((x - y).abs() / (x.abs() + 1e-3)) as f32;
                    if r > max_rel {
                        max_rel = r;
                    }
                }
                let rel_l2 = (num / den.max(1e-12)).sqrt();
                eprintln!(
                    "    fp8 vs fp16: rel_l2={rel_l2:.4e} max_rel={max_rel:.3} (a_scale={a_scale:.5} b_scale={b_scale:.5})"
                );
            }
        }
        Ok(())
    }
}

#[test]
#[ignore = "micro-benchmark, run on demand"]
fn bench_gemm_lowprec() {
    let Ok(rt) = CudaRuntime::new() else {
        eprintln!("skipped: CUDA runtime unavailable");
        return;
    };
    let stream = rt.device.new_stream().expect("stream");
    eprintln!("=== low-precision GEMM micro-bench (us/iter) ===");
    // trunk 三类形状 × 实用 batch 档
    for (n, k) in [(384usize, 384usize), (1152, 384), (384, 1152)] {
        for b in [1usize, 4, 8, 16] {
            let m = b * 361;
            if let Err(e) = bench_one(&rt, &stream, m, n, k, 200) {
                eprintln!("M={m} N={n} K={k}: {e}");
            }
        }
    }
}

/// INT8 (imma) 接受性+速度探针：A_cm=权重 OP_T→COL4_4R2_8C，
/// B_cm=激活 OP_N→COL32，C=int32。只测速度（布局内容不影响耗时）。
#[test]
#[ignore = "int8 layout probe"]
fn bench_int8_layouts() {
    let Ok(rt) = CudaRuntime::new() else {
        eprintln!("skipped: CUDA runtime unavailable");
        return;
    };
    let stream = rt.device.new_stream().expect("stream");
    eprintln!("=== int8 imma layout probe (us/iter) ===");
    use sys::cublasLtOrder_t as O;
    let combos = [
        (true, O::CUBLASLT_ORDER_COL, O::CUBLASLT_ORDER_COL, O::CUBLASLT_ORDER_COL),
        (true, O::CUBLASLT_ORDER_COL32, O::CUBLASLT_ORDER_COL32, O::CUBLASLT_ORDER_COL32),
        (true, O::CUBLASLT_ORDER_COL4_4R2_8C, O::CUBLASLT_ORDER_COL32, O::CUBLASLT_ORDER_COL32),
        (true, O::CUBLASLT_ORDER_COL4_4R2_8C, O::CUBLASLT_ORDER_COL32, O::CUBLASLT_ORDER_COL),
        (true, O::CUBLASLT_ORDER_ROW, O::CUBLASLT_ORDER_ROW, O::CUBLASLT_ORDER_ROW),
        (false, O::CUBLASLT_ORDER_COL4_4R2_8C, O::CUBLASLT_ORDER_COL32, O::CUBLASLT_ORDER_COL32),
    ];
    for (sf, oa, ob, oc) in combos {
        let r = unsafe { probe_int8(&rt, &stream, 2888, 1152, 384, 200, sf, oa, ob, oc) };
        match r {
            Ok((us, note)) => {
                let tf = 2.0f64 * 2888.0 * 1152.0 * 384.0 / (us as f64 * 1e-6) / 1e12;
                eprintln!("ACCEPT {note}: {us:8.2}us ({tf:6.1} TOPS)");
            }
            Err(e) => eprintln!("reject sf={sf} {oa:?}/{ob:?}/{oc:?}: {e}"),
        }
    }
}

#[cfg(feature = "cuda")]
unsafe fn probe_int8(
    rt: &CudaRuntime,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    m: usize,
    n: usize,
    k: usize,
    iters: usize,
    scale_f32: bool,
    order_a: sys::cublasLtOrder_t,
    order_b: sys::cublasLtOrder_t,
    order_c: sys::cublasLtOrder_t,
) -> Result<(f32, String), String> {
    use cudarc::driver::DevicePtrMut;
    let st = rt.cublaslt_handle().ok_or("no cublaslt")?;
    let a_sz = n * k; // 权重（逻辑 [n,k]）
    let b_sz = m * k; // 激活（逻辑 [m,k]）
    let c_sz = m * n;
    let a8: CudaSlice<i8> = stream.alloc_zeros(a_sz).map_err(|e| e.to_string())?;
    let b8: CudaSlice<i8> = stream.alloc_zeros(b_sz).map_err(|e| e.to_string())?;
    let mut c32: CudaSlice<i32> = stream.alloc_zeros(c_sz).map_err(|e| e.to_string())?;

    let mut desc: sys::cublasLtMatmulDesc_t = std::ptr::null_mut();
    let scale_ty = if scale_f32 {
        sys::cudaDataType_t::CUDA_R_32F
    } else {
        sys::cudaDataType_t::CUDA_R_32I
    };
    sys::cublasLtMatmulDescCreate(
        &mut desc,
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32I,
        scale_ty,
    );
    let transa: u32 = 1; // OP_T（A_cm=权重）
    let transb: u32 = 0; // OP_N（B_cm=激活）
    sys::cublasLtMatmulDescSetAttribute(
        desc,
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
        &transa as *const _ as *const _,
        4,
    );
    sys::cublasLtMatmulDescSetAttribute(
        desc,
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
        &transb as *const _ as *const _,
        4,
    );
    let mut a_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut b_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut c_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
    sys::cublasLtMatrixLayoutCreate(
        &mut a_lay,
        sys::cudaDataType_t::CUDA_R_8I,
        k as u64,
        n as u64,
        k as i64,
    );
    sys::cublasLtMatrixLayoutCreate(
        &mut b_lay,
        sys::cudaDataType_t::CUDA_R_8I,
        k as u64,
        m as u64,
        k as i64,
    );
    sys::cublasLtMatrixLayoutCreate(
        &mut c_lay,
        sys::cudaDataType_t::CUDA_R_32I,
        n as u64,
        m as u64,
        n as i64,
    );

    sys::cublasLtMatrixLayoutSetAttribute(
        a_lay,
        sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_ORDER,
        &order_a as *const _ as *const _,
        4,
    );
    sys::cublasLtMatrixLayoutSetAttribute(
        b_lay,
        sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_ORDER,
        &order_b as *const _ as *const _,
        4,
    );
    sys::cublasLtMatrixLayoutSetAttribute(
        c_lay,
        sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_ORDER,
        &order_c as *const _ as *const _,
        4,
    );
    let mut pref: sys::cublasLtMatmulPreference_t = std::ptr::null_mut();
    sys::cublasLtMatmulPreferenceCreate(&mut pref);
    let ws_len = rt.cublaslt_workspace_len();
    sys::cublasLtMatmulPreferenceSetAttribute(
        pref,
        sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &ws_len as *const _ as *const _,
        8,
    );
    let mut heur: sys::cublasLtMatmulHeuristicResult_t = std::mem::zeroed();
    let mut cnt = 0i32;
    let r = sys::cublasLtMatmulAlgoGetHeuristic(
        st, desc, a_lay, b_lay, c_lay, c_lay, pref, 1, &mut heur, &mut cnt,
    );
    if r != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS || cnt == 0 {
        sys::cublasLtMatmulDescDestroy(desc);
        sys::cublasLtMatrixLayoutDestroy(a_lay);
        sys::cublasLtMatrixLayoutDestroy(b_lay);
        sys::cublasLtMatrixLayoutDestroy(c_lay);
        sys::cublasLtMatmulPreferenceDestroy(pref);
        return Err(format!("heuristic: {r:?} cnt={cnt}"));
    }
    let (a8p, b8p, c32p, ws_ptr) = {
        let (a8p, _g) = a8.device_ptr(stream);
        let (b8p, _g) = b8.device_ptr(stream);
        let (c32p, _g) = c32.device_ptr_mut(stream);
        let (ws_ptr, _gw) = rt.cublaslt_workspace_ptr(stream);
        (a8p, b8p, c32p, ws_ptr)
    };
    let alpha_i = 1i32;
    let beta_i = 0i32;
    let alpha_f = 1.0f32;
    let beta_f = 0.0f32;
    let (ap, bp) = if scale_f32 {
        (
            &alpha_f as *const f32 as *const std::ffi::c_void,
            &beta_f as *const f32 as *const std::ffi::c_void,
        )
    } else {
        (
            &alpha_i as *const i32 as *const std::ffi::c_void,
            &beta_i as *const i32 as *const std::ffi::c_void,
        )
    };
    let mut run = || -> Result<(), String> {
        let r = sys::cublasLtMatmul(
            st,
            desc,
            ap,
            a8p as *const _,
            a_lay,
            b8p as *const _,
            b_lay,
            bp,
            c32p as *const _,
            c_lay,
            c32p as *mut _,
            c_lay,
            &heur.algo,
            ws_ptr as *mut _,
            ws_len,
            stream.cu_stream() as _,
        );
        if r != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(format!("matmul: {r:?}"));
        }
        Ok(())
    };
    for _ in 0..10 {
        run()?;
    }
    stream.synchronize().map_err(|e| e.to_string())?;
    let ev0 = rt
        .device
        .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|e| e.to_string())?;
    let ev1 = rt
        .device
        .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|e| e.to_string())?;
    ev0.record(stream).map_err(|e| e.to_string())?;
    for _ in 0..iters {
        run()?;
    }
    ev1.record(stream).map_err(|e| e.to_string())?;
    stream.synchronize().map_err(|e| e.to_string())?;
    let ms = ev0.elapsed_ms(&ev1).map_err(|e| e.to_string())?;
    sys::cublasLtMatmulDescDestroy(desc);
    sys::cublasLtMatrixLayoutDestroy(a_lay);
    sys::cublasLtMatrixLayoutDestroy(b_lay);
    sys::cublasLtMatrixLayoutDestroy(c_lay);
    sys::cublasLtMatmulPreferenceDestroy(pref);
    Ok((ms * 1000.0 / iters as f32, format!("scale_f32={scale_f32} ord={order_a:?}/{order_b:?}/{order_c:?}")))
}

// ---------------------------------------------------------------------------
// igemm_t64_kernel（手写 INT8 imma）正确性 + 速度验证
// ---------------------------------------------------------------------------

/// host 参考：dot(i32) × sa × sb → f16 bits。
fn igemm_cpu_ref(
    a: &[i8], b: &[i8], sa: &[f32], sb: &[f32],
    m: usize, n: usize, k: usize,
) -> Vec<u16> {
    let mut out = vec![0u16; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i32;
            for t in 0..k {
                acc += a[i * k + t] as i32 * b[j * k + t] as i32;
            }
            out[i * n + j] = half_from_f32(acc as f32 * sa[i] * sb[j]);
        }
    }
    out
}

fn igemm_run(
    rt: &CudaRuntime,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    a: &CudaSlice<i8>, b: &CudaSlice<i8>,
    sa: &CudaSlice<f32>, sb: &CudaSlice<f32>,
    c: &mut CudaSlice<u16>,
    m: usize, n: usize, k: usize,
) -> Result<(), String> {
    let f = rt.get_func("igemm_t64_kernel")?;
    let grid = (m.div_ceil(64) as u32, n.div_ceil(64) as u32, 1u32);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: grid,
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(b)
            .arg(sa)
            .arg(sb)
            .arg(c)
            .arg(&(m as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| format!("igemm launch: {e}"))
}

#[test]
#[ignore = "int8 imma kernel validation + bench"]
fn igemm_t64_validate_and_bench() {
    let Ok(rt) = CudaRuntime::new() else {
        eprintln!("skipped: CUDA runtime unavailable");
        return;
    };
    let stream = rt.device.new_stream().expect("stream");

    // ---- 正确性（奇数 M 测边界） ----
    let (m, n, k) = (131usize, 384usize, 384usize);
    let mut rng = Rng(42);
    let a_h: Vec<i8> = (0..m * k).map(|_| (rng.next_f32() * 255.0 - 127.5) as i8).collect();
    let b_h: Vec<i8> = (0..n * k).map(|_| (rng.next_f32() * 255.0 - 127.5) as i8).collect();
    let sa_h: Vec<f32> = (0..m).map(|_| 0.001 + rng.next_f32() * 0.004).collect();
    let sb_h: Vec<f32> = (0..n).map(|_| 0.0005 + rng.next_f32() * 0.002).collect();
    unsafe {
        let a = stream.memcpy_stod(&a_h).map_err(|e| e.to_string()).unwrap();
        let b = stream.memcpy_stod(&b_h).map_err(|e| e.to_string()).unwrap();
        let sa = stream.memcpy_stod(&sa_h).map_err(|e| e.to_string()).unwrap();
        let sb = stream.memcpy_stod(&sb_h).map_err(|e| e.to_string()).unwrap();
        let mut c: CudaSlice<u16> = stream.alloc_zeros(m * n).unwrap();
        igemm_run(&rt, &stream, &a, &b, &sa, &sb, &mut c, m, n, k).unwrap();
        stream.synchronize().unwrap();
        let mut c_h = vec![0u16; m * n];
        stream.memcpy_dtoh(&c, &mut c_h).unwrap();
        let refr = igemm_cpu_ref(&a_h, &b_h, &sa_h, &sb_h, m, n, k);
        let mut bad = 0;
        let mut max_rel = 0.0f32;
        for i in 0..m * n {
            let x = half_to_f32(c_h[i]);
            let y = half_to_f32(refr[i]);
            let rel = (x - y).abs() / y.abs().max(1e-2);
            if rel > max_rel {
                max_rel = rel;
            }
            if rel > 2e-3 {
                bad += 1;
                if bad < 5 {
                    eprintln!("  mismatch at flat {i}: got {x}, want {y} (rel {rel:.2e})");
                }
            }
        }
        eprintln!("igemm correctness: bad={bad}/{} max_rel={max_rel:.2e}", m * n);
        assert_eq!(bad, 0, "igemm kernel mismatch");
    }

    // ---- 速度（vs 已测 fp16 cuBLASLt 参考值同跑复测） ----
    eprintln!("=== igemm_t64 vs cublasLt-fp16 (us/iter) ===");
    for (n, k) in [(384usize, 384usize), (1152, 384), (384, 1152)] {
        for bb in [1usize, 4, 8, 16] {
            let m = bb * 361;
            unsafe {
                let a_h = vec![3i8; m * k];
                let b_h = vec![2i8; n * k];
                let sa_h = vec![0.001f32; m];
                let sb_h = vec![0.0005f32; n];
                let a = stream.memcpy_stod(&a_h).unwrap();
                let b = stream.memcpy_stod(&b_h).unwrap();
                let sa = stream.memcpy_stod(&sa_h).unwrap();
                let sb = stream.memcpy_stod(&sb_h).unwrap();
                let mut c: CudaSlice<u16> = stream.alloc_zeros(m * n).unwrap();
                // fp16 对照
                let a16 = stream.memcpy_stod(&vec![0x2c66u16; m * k]).unwrap();
                let b16 = stream.memcpy_stod(&vec![0x2c66u16; n * k]).unwrap();
                let mut c16: CudaSlice<u16> = stream.alloc_zeros(m * n).unwrap();

                let iters = 200;
                // warmup
                for _ in 0..10 {
                    igemm_run(&rt, &stream, &a, &b, &sa, &sb, &mut c, m, n, k).unwrap();
                }
                stream.synchronize().unwrap();
                let ev0 = rt.device.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                let ev1 = rt.device.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                ev0.record(&stream).unwrap();
                for _ in 0..iters {
                    igemm_run(&rt, &stream, &a, &b, &sa, &sb, &mut c, m, n, k).unwrap();
                }
                ev1.record(&stream).unwrap();
                stream.synchronize().unwrap();
                let t_i8 = ev0.elapsed_ms(&ev1).unwrap() * 1000.0 / iters as f32;

                let ok = rt.cublaslt_gemm_f16out(&stream, &a16, &b16, &mut c16, m, n, k).unwrap();
                assert!(ok);
                for _ in 0..10 {
                    rt.cublaslt_gemm_f16out(&stream, &a16, &b16, &mut c16, m, n, k).unwrap();
                }
                stream.synchronize().unwrap();
                let ev2 = rt.device.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                let ev3 = rt.device.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                ev2.record(&stream).unwrap();
                for _ in 0..iters {
                    rt.cublaslt_gemm_f16out(&stream, &a16, &b16, &mut c16, m, n, k).unwrap();
                }
                ev3.record(&stream).unwrap();
                stream.synchronize().unwrap();
                let t_f16 = ev2.elapsed_ms(&ev3).unwrap() * 1000.0 / iters as f32;

                let tops = 2.0 * m as f64 * n as f64 * k as f64 / (t_i8 as f64 * 1e-6) / 1e12;
                eprintln!(
                    "M={m:6} N={n:5} K={k:5} | igemm {t_i8:8.2}us ({tops:6.1} TOPS) | fp16lt {t_f16:8.2}us | speedup {:.2}x",
                    t_f16 / t_i8
                );
            }
        }
    }
}
