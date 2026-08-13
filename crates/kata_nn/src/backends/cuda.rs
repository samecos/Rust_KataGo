//! Hand-written CUDA backend (M3 骨架：kernel 加载与基础运行时）。
//!
//! kernel 由 `build.rs` 按 `configs/sm-targets.json` 编译为 PTX 并嵌入二进制
//! （`CUDA_TARGETS` 表）；运行时按设备 compute capability 选择最优目标。
//!
//! 对应 KataGo 官方 `cpp/neuralnet/cudabackend.cpp`，并参考 KataGomo_fork 的
//! SM120 优化路径（plan 驱动、fail-closed 见 M4）。

#[cfg(feature = "cuda")]
mod imp {
    use cudarc::driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg,
    };
    use cudarc::driver::sys::CUdevice_attribute;
    use std::sync::Arc;

    // 由 build.rs 生成的 PTX 表：`(target_id, compute_capability, [(kernel_name, ptx_bytes)])`。
    include!(concat!(env!("OUT_DIR"), "/cuda_kernels.rs"));

    /// 按设备 compute capability 选择编译入的最优 SM 目标。
    fn select_target(dev: &Arc<CudaContext>) -> Result<(&'static str, &'static [(&'static str, &'static [u8])]), String> {
        let major = dev
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| format!("failed to query compute capability: {e}"))?;
        let minor = dev
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| format!("failed to query compute capability: {e}"))?;
        let cap = format!("{major}.{minor}");
        let _ = &cap;
        // 精确匹配优先，否则选数值上不大于设备能力的最新目标。
        for (id, target_cap, kernels) in CUDA_TARGETS {
            if *target_cap == cap {
                return Ok((id, kernels));
            }
        }
        let mut best: Option<(&str, &str, &[(&str, &[u8])])> = None;
        for (id, target_cap, kernels) in CUDA_TARGETS {
            let compatible = target_cap
                .parse::<f32>()
                .map(|t| t <= major as f32 + minor as f32 / 10.0)
                .unwrap_or(false);
            if compatible {
                let better = best
                    .map(|(_, bcap, _)| target_cap.parse::<f32>().unwrap_or(0.0) > bcap.parse::<f32>().unwrap_or(0.0))
                    .unwrap_or(true);
                if better {
                    best = Some((id, target_cap, kernels));
                }
            }
        }
        match best {
            Some((id, _, kernels)) => Ok((id, kernels)),
            None => Err(format!("no compiled SM target compatible with device capability {cap}")),
        }
    }

    /// CUDA 运行时：设备 + 按 SM 目标加载的 kernel 模块。
    pub struct CudaRuntime {
        pub device: Arc<CudaContext>,
        pub target_id: String,
        modules: Vec<(String, Arc<CudaModule>)>,
    }

    impl CudaRuntime {
        /// 初始化设备 0 并加载最优 SM 目标的全部 kernel（fail-closed）。
        pub fn new() -> Result<Self, String> {
            let device = CudaContext::new(0).map_err(|e| format!("CUDA device 0 init failed: {e}"))?;
            let (target_id, kernels) = select_target(&device)?;
            let mut modules = Vec::new();
            for (name, ptx) in kernels {
                let ptx = cudarc::nvrtc::Ptx::from_src(
                    std::str::from_utf8(ptx).map_err(|_| "PTX not UTF-8".to_string())?.to_string(),
                );
                let module = device
                    .load_module(ptx)
                    .map_err(|e| format!("failed to load PTX for kernel {name}: {e}"))?;
                modules.push((name.to_string(), module));
            }
            Ok(Self {
                device,
                target_id: target_id.to_string(),
                modules,
            })
        }

        /// 按名字查找 kernel 函数。
        pub fn get_func(&self, name: &str) -> Result<CudaFunction, String> {
            for (kname, module) in &self.modules {
                if let Ok(f) = module.load_function(name) {
                    let _ = kname;
                    return Ok(f);
                }
            }
            Err(format!("kernel {name} not found"))
        }

        /// 冒烟/自检：`out[i] = a[i] + b[i]`。
        pub fn f32_add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
            assert_eq!(a.len(), b.len());
            let n = a.len();
            let f = self.get_func("f32_add_kernel")?;
            let stream = self.device.default_stream();
            let mut d_a: CudaSlice<f32> =
                unsafe { stream.alloc(n) }.map_err(|e| e.to_string())?;
            let mut d_b: CudaSlice<f32> =
                unsafe { stream.alloc(n) }.map_err(|e| e.to_string())?;
            stream
                .memcpy_htod(a, &mut d_a)
                .map_err(|e| e.to_string())?;
            stream
                .memcpy_htod(b, &mut d_b)
                .map_err(|e| e.to_string())?;
            let mut d_out: CudaSlice<f32> = stream.alloc_zeros(n).map_err(|e| e.to_string())?;
            let cfg = LaunchConfig::for_num_elems(n as u32);
            unsafe {
                stream
                    .launch_builder(&f)
                    .arg(&d_a)
                    .arg(&d_b)
                    .arg(&mut d_out)
                    .arg(&(n as i32))
                    .launch(cfg)
            }
            .map_err(|e| format!("launch failed: {e}"))?;
            let mut out = vec![0.0f32; n];
            stream
                .memcpy_dtoh(&d_out, &mut out)
                .map_err(|e| e.to_string())?;
            Ok(out)
        }

        /// FP16 张量核 GEMM：`C[M,N] = alpha * A[M,K] * B[N,K]^T + beta * C`。
        ///
        /// `a`/`b` 为 f32 主机数据（主机侧转 half），`c` 就地读写；K 必须是
        /// 16 的倍数（调用方负责 padding）。对应 `cuda-kernels/gemm.cu` 的
        /// `hgemm_m16n8k16_kernel`（内联 PTX mma.sync m16n8k16）。
        #[allow(clippy::too_many_arguments)]
        pub fn hgemm_m16n8k16(
            &self,
            a: &[f32],
            b: &[f32],
            c: &mut [f32],
            m: usize,
            n: usize,
            k: usize,
            alpha: f32,
            beta: f32,
        ) -> Result<(), String> {
            assert_eq!(a.len(), m * k, "A 尺寸不符");
            assert_eq!(b.len(), n * k, "B 尺寸不符");
            assert_eq!(c.len(), m * n, "C 尺寸不符");
            assert_eq!(k % 16, 0, "K 必须是 16 的倍数");

            let a_half: Vec<u16> = a.iter().map(|&x| f32_to_f16_bits(x)).collect();
            let b_half: Vec<u16> = b.iter().map(|&x| f32_to_f16_bits(x)).collect();

            let f = self.get_func("hgemm_m16n8k16_kernel")?;
            let stream = self.device.default_stream();
            let mut d_a: CudaSlice<u16> =
                unsafe { stream.alloc(a_half.len()) }.map_err(|e| e.to_string())?;
            let mut d_b: CudaSlice<u16> =
                unsafe { stream.alloc(b_half.len()) }.map_err(|e| e.to_string())?;
            let mut d_c: CudaSlice<f32> =
                unsafe { stream.alloc(c.len()) }.map_err(|e| e.to_string())?;
            stream
                .memcpy_htod(a_half.as_slice(), &mut d_a)
                .map_err(|e| e.to_string())?;
            stream
                .memcpy_htod(b_half.as_slice(), &mut d_b)
                .map_err(|e| e.to_string())?;
            stream
                .memcpy_htod(c, &mut d_c)
                .map_err(|e| e.to_string())?;

            let grid = (
                m.div_ceil(64) as u32,
                n.div_ceil(64) as u32,
                1u32,
            );
            let block = (4 * 32) as u32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: grid,
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(&f)
                    .arg(&d_a)
                    .arg(&d_b)
                    .arg(&mut d_c)
                    .arg(&(m as i32))
                    .arg(&(n as i32))
                    .arg(&(k as i32))
                    .arg(&alpha)
                    .arg(&beta)
                    .launch(cfg)
            }
            .map_err(|e| format!("hgemm launch failed: {e}"))?;

            stream
                .memcpy_dtoh(&d_c, c)
                .map_err(|e| e.to_string())?;
            Ok(())
        }
    }

    /// f32 → f16 位模式（round-to-nearest-even，与 CUDA `__float2half` 一致）。
    pub fn f32_to_f16_bits(x: f32) -> u16 {
        let b = x.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32;
        let mant = b & 0x7fffff;
        if exp == 0xff {
            return if mant != 0 { sign | 0x7e00 } else { sign | 0x7c00 };
        }
        let e = exp - 127 + 15;
        if e >= 0x1f {
            return sign | 0x7c00;
        }
        if e <= 0 {
            if e < -10 {
                return sign;
            }
            let m = (mant | 0x800000) >> (14 - e);
            let m = (m + 1) >> 1;
            return sign | (m as u16);
        }
        let m = mant >> 13;
        let round_bits = mant & 0x1fff;
        let mut m = m as u16;
        if round_bits > 0x1000 || (round_bits == 0x1000 && (m & 1) == 1) {
            m += 1;
            if m == 0x400 {
                if e + 1 >= 0x1f {
                    return sign | 0x7c00;
                }
                return sign | (((e + 1) as u16) << 10);
            }
        }
        sign | ((e as u16) << 10) | m
    }
}

/// CUDA 后端入口（无 feature 时的占位实现）。
#[cfg(not(feature = "cuda"))]
pub struct CudaRuntime;

#[cfg(not(feature = "cuda"))]
impl CudaRuntime {
    pub fn new() -> Result<Self, String> {
        Err("CUDA backend not enabled (enable the 'cuda' Cargo feature)".to_string())
    }
}

#[cfg(feature = "cuda")]
pub use imp::{CudaRuntime, f32_to_f16_bits};
