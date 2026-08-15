//! E4M3 缩放策略精度实验（混合精度调研 Stage-1.5）：
//! 同一 GEMM 在三种缩放策略下的 rel_l2 误差对比：
//!   A) per-tensor（单标定 scale）——微基准已测 ~3.7e-2
//!   B) per-row 激活 × per-channel 权重（目标 ≤5e-3）
//!   C) per-row 激活 × per-tensor 权重（折中）
//! 输出 FP32 原始点积，host 端反量化后对比 FP16 参考。
//!
//! 运行：cargo test -p kata_nn --features cuda --release \
//!   --test bench_lowprec_scale -- --ignored --nocapture
#![cfg(feature = "cuda")]
#![allow(unsafe_op_in_unsafe_fn)]

use cudarc::cublaslt::sys;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use kata_nn::backends::cuda::CudaRuntime;

fn f32_to_e4m3(x: f32) -> u8 {
    if x == 0.0 || !x.is_finite() {
        return if x.is_nan() { 0x7f } else { 0 };
    }
    let sign = if x < 0.0 { 0x80u8 } else { 0 };
    let ax = x.abs().min(448.0);
    let mut e = ax.log2().floor() as i32;
    let mut m = ax / 2f32.powi(e);
    if m >= 2.0 {
        m /= 2.0;
        e += 1;
    }
    if e < -6 {
        let mant = (ax / 2f32.powi(-9)).round() as u32;
        return sign | (mant.min(7) as u8);
    }
    if e > 8 {
        return sign | 0x7e;
    }
    let exp = (e + 7) as u32;
    let mant = ((m - 1.0) * 8.0).round() as u32;
    let (exp, mant) = if mant == 8 { (exp + 1, 0) } else { (exp, mant) };
    if exp > 15 {
        return sign | 0x7e;
    }
    sign | ((exp as u8) << 3) | (mant as u8)
}

fn half_from_f32(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 255 {
        return sign | 0x7c00;
    }
    let e = exp - 127 + 15;
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (mant | 0x0080_0000) >> ((14 - e) as u32);
        let rem = (mant | 0x0080_0000) & ((1u32 << ((14 - e) as u32)) - 1);
        let halfway = 1u32 << ((13 - e) as u32);
        let mut m = m;
        if rem > halfway || (rem == halfway && m & 1 == 1) {
            m += 1;
        }
        return sign | (m as u16);
    }
    if e >= 31 {
        return sign | 0x7c00;
    }
    let mut h = sign | ((e as u16) << 10) | ((mant >> 13) as u16);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h = h.wrapping_add(1);
    }
    h
}

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

struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (1u32 << 31) as f32
    }
    fn next_norm(&mut self) -> f32 {
        let s: f32 = (0..8).map(|_| self.next_f32()).sum();
        (s - 4.0) * 0.5
    }
}

/// 跑一次 E4M3 GEMM（A[M,K] row scales sa，B[N,K] row scales sb；
/// GEMM 用 scale=1.0，输出 f32 原始点积后 host 反量化），
/// 返回与 FP16 参考的 rel_l2。
#[allow(clippy::too_many_arguments)]
fn quant_rel_l2(
    rt: &CudaRuntime,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    per_row_a: bool,
    per_chan_b: bool,
) -> Result<f64, String> {
    unsafe {
        // scales
        let sa: Vec<f32> = (0..m)
            .map(|i| {
                let amax = if per_row_a {
                    (0..k).fold(0.0f32, |p, j| p.max(a[i * k + j].abs()))
                } else {
                    a.iter().fold(0.0f32, |p, &x| p.max(x.abs()))
                };
                (amax / 448.0).max(1e-12)
            })
            .collect();
        let sb: Vec<f32> = (0..n)
            .map(|j| {
                let amax = if per_chan_b {
                    (0..k).fold(0.0f32, |p, t| p.max(b[j * k + t].abs()))
                } else {
                    b.iter().fold(0.0f32, |p, &x| p.max(x.abs()))
                };
                (amax / 448.0).max(1e-12)
            })
            .collect();
        let a_q: Vec<u8> = (0..m * k)
            .map(|idx| f32_to_e4m3(a[idx] / sa[idx / k]))
            .collect();
        let b_q: Vec<u8> = (0..n * k)
            .map(|idx| f32_to_e4m3(b[idx] / sb[idx / k]))
            .collect();
        let a16: Vec<u16> = a.iter().map(|&x| half_from_f32(x)).collect();
        let b16: Vec<u16> = b.iter().map(|&x| half_from_f32(x)).collect();

        let a8 = stream.memcpy_stod(&a_q).map_err(|e| e.to_string())?;
        let b8 = stream.memcpy_stod(&b_q).map_err(|e| e.to_string())?;
        let a16d = stream.memcpy_stod(&a16).map_err(|e| e.to_string())?;
        let b16d = stream.memcpy_stod(&b16).map_err(|e| e.to_string())?;
        let mut c32: CudaSlice<f32> = stream.alloc_zeros(m * n).map_err(|e| e.to_string())?;
        let mut c16: CudaSlice<u16> = stream.alloc_zeros(m * n).map_err(|e| e.to_string())?;
        let one = stream.memcpy_stod(&[1.0f32]).map_err(|e| e.to_string())?;

        let st = rt.cublaslt_handle().ok_or("no cublaslt")?;
        let mut desc: sys::cublasLtMatmulDesc_t = std::ptr::null_mut();
        sys::cublasLtMatmulDescCreate(
            &mut desc,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cudaDataType_t::CUDA_R_32F,
        );
        let transa: u32 = 1;
        let transb: u32 = 0;
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
        let (one_ptr, _g) = one.device_ptr(stream);
        for attr in [
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        ] {
            sys::cublasLtMatmulDescSetAttribute(desc, attr, &one_ptr as *const _ as *const _, 8);
        }
        let mut a_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut b_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut c_lay: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
        sys::cublasLtMatrixLayoutCreate(
            &mut a_lay,
            sys::cudaDataType_t::CUDA_R_8F_E4M3,
            k as u64,
            n as u64,
            k as i64,
        );
        sys::cublasLtMatrixLayoutCreate(
            &mut b_lay,
            sys::cudaDataType_t::CUDA_R_8F_E4M3,
            k as u64,
            m as u64,
            k as i64,
        );
        sys::cublasLtMatrixLayoutCreate(
            &mut c_lay,
            sys::cudaDataType_t::CUDA_R_32F,
            n as u64,
            m as u64,
            n as i64,
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
            return Err(format!("fp8 heuristic: {r:?}"));
        }
        let (a8p, b8p, c32p, ws_ptr) = {
            let (a8p, _g1) = a8.device_ptr(stream);
            let (b8p, _g2) = b8.device_ptr(stream);
            let (c32p, _g3) = c32.device_ptr_mut(stream);
            let (ws_ptr, _gw) = rt.cublaslt_workspace_ptr(stream);
            (a8p, b8p, c32p, ws_ptr)
        };
        let alpha = 1.0f32;
        let beta = 0.0f32;
        let r = sys::cublasLtMatmul(
            st,
            desc,
            &alpha as *const _ as *const _,
            b8p as *const _,
            a_lay,
            a8p as *const _,
            b_lay,
            &beta as *const _ as *const _,
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
            return Err(format!("fp8 matmul: {r:?}"));
        }

        // FP16 参考（生产路径同款）
        let ok = rt.cublaslt_gemm_f16out(stream, &a16d, &b16d, &mut c16, m, n, k)?;
        if !ok {
            return Err("fp16 gemm unavailable".into());
        }
        stream.synchronize().map_err(|e| e.to_string())?;

        let mut c32_h = vec![0.0f32; m * n];
        let mut c16_h = vec![0u16; m * n];
        stream.memcpy_dtoh(&c32, &mut c32_h).map_err(|e| e.to_string())?;
        stream.memcpy_dtoh(&c16, &mut c16_h).map_err(|e| e.to_string())?;

        sys::cublasLtMatmulDescDestroy(desc);
        sys::cublasLtMatrixLayoutDestroy(a_lay);
        sys::cublasLtMatrixLayoutDestroy(b_lay);
        sys::cublasLtMatrixLayoutDestroy(c_lay);
        sys::cublasLtMatmulPreferenceDestroy(pref);

        // host 反量化 + rel_l2
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let raw = c32_h[i * n + j] as f64;
                let deq = raw * sa[i] as f64 * sb[j] as f64;
                let refr = half_to_f32(c16_h[i * n + j]) as f64;
                num += (deq - refr) * (deq - refr);
                den += refr * refr;
            }
        }
        Ok((num / den.max(1e-12)).sqrt())
    }
}

#[test]
#[ignore = "accuracy experiment, run on demand"]
fn fp8_scale_strategy_accuracy() {
    let Ok(rt) = CudaRuntime::new() else {
        eprintln!("skipped: CUDA runtime unavailable");
        return;
    };
    let stream = rt.device.new_stream().expect("stream");
    let (m, n, k) = (2888usize, 1152usize, 384usize);
    let mut rng = Rng(0xdead_beef_cafe_f00d);
    // 模拟真实分布：权重小尺度正态; 激活含少量 outlier 通道
    // （transformer 激活的通道间幅度差异是 per-tensor 量化的主要误差源）。
    let a: Vec<f32> = (0..m * k)
        .map(|idx| {
            let ch = idx % k;
            let chan_gain = if ch % 37 == 0 { 6.0 } else { 1.0 }; // outlier 通道
            rng.next_norm() * 0.12 * chan_gain
        })
        .collect();
    let b: Vec<f32> = (0..n * k).map(|_| rng.next_norm() * 0.05).collect();

    for (name, pra, pcb) in [
        ("per-tensor A × per-tensor B", false, false),
        ("per-row    A × per-tensor B", true, false),
        ("per-tensor A × per-chan   B", false, true),
        ("per-row    A × per-chan   B", true, true),
    ] {
        match quant_rel_l2(&rt, &stream, &a, &b, m, n, k, pra, pcb) {
            Ok(r) => eprintln!("{name}: rel_l2 = {r:.4e}"),
            Err(e) => eprintln!("{name}: FAIL {e}"),
        }
    }
}
