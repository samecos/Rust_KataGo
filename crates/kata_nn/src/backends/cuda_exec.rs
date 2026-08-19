//! 手写 CUDA 推理执行器：把 [`crate::onnx_parser::LayerGraph`] 逐层对接
//! `cuda-kernels/` 里的手写 kernel，跑通 b11fix.onnx（19 路）的完整前向。
//!
//! ## 数值与布局纪律
//!
//! - 所有权重（GEMM 用）在 `load` 期 f32 → f16（round-to-nearest-even）上传设备；
//!   K 维 pad 到 16 的倍数（pad 列恒零）。RoPE 表与逐通道参数（scale/bias/BN）
//!   元素量小，保留 f32 精度。
//! - 中间激活一律 NHWC 行主序 `[B*S, C]`，存 f16；GEMM 输出 f32，进入下一层
//!   前转 f16（`f32_to_half_kernel` / 带激活的融合 kernel）。残差加回时
//!   f16 残差流转 f32 与 GEMM 输出相加再舍入回 f16（对齐官方 half 存储边界）。
//! - **GEMM A 缓冲的列 stride 必须等于该 GEMM 的 K**（`hgemm` 内核按单 stride
//!   同时读 A/B）。因此各 GEMM 的 A 源要么是精确尺寸的专用缓冲（cols=208、
//!   p96/p192、pooled/vec 的紧凑前缀），要么是 stride 恰好等于 K 的流缓冲
//!   （768/384/1152）；不存在"零填充列"策略。
//! - 首层 conv 为 3x3 same（pad=1，导出图如此）→ im2col `[B*S, 208]`（K=198
//!   pad 16 对齐）+ `hgemm`；全局输入 `[B,19] @ W[768,19]^T` 逐 token 广播
//!   加在 conv 输出上，再一起过门控 SiLU（严格按 IR 的 Add→gate 顺序）。
//! - Attention：qkv GEMM → RoPE（半维 (16,2) 偶/奇下标对，cos/sin 表
//!   `[S, H*D/2]` f32）→ `[B*H,S,D]` → `attention_row_v3_kernel`（score 缩放
//!   传 `qk_scale*qk_scale`，等价于导出图 q/k 各乘 1/∜d）→ 拼回 → 输出投影
//!   → 残差。掩码恒零（19 路 on-board 恒 1，解析器已校验），不传掩码。
//! - 策略/价值头按 IR 注释逐步执行；×on_board 掩码乘法恒 1 省略，落点屏蔽
//!   5000 的 `(1-on_board)` 因子恒 0（结构保留在 `policy_concat_kernel` 的
//!   penalty 参数里，调用方传 0.0）。
//!
//! ## 与 `Backend` trait 的关系
//!
//! 本文件交付 `CudaModel::load` + `CudaModel::apply` + 数值对拍（
//! `tests/dump_nn_io_cuda.rs`）。`crate::backend::Backend` trait 的对接
//! 已实现（2026-08-14，`cuda.rs::backend_impl::CudaBackend`）：load_model_file
//! → onnx_parser → CudaModel::load；get_output → apply + v15 解码 + 对称性。
//! 调试开关：`KATAGO_CUDA_DEBUG_LAYER=<i>` 逐层 dump、`KATAGO_CUDA_PROFILE=1`
//! 逐层耗时、`KATAGO_CUDA_DUMP_INPUT=<dir>` 输入 dump（见 cuda.rs）。

use crate::backends::cuda::{f16_to_f32_bits, f32_to_f16_bits, CudaRuntime};
use crate::onnx_parser::{Layer, LayerGraph, Tensor, TensorData};
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::cell::RefCell;
use std::sync::Arc;

/// cudarc 0.19 的流方法定义在 `Arc<CudaStream>` 上（`self: &Arc<Self>`）。
type StreamRef = Arc<CudaStream>;

/// 当前线程活跃的 CUDA 流（apply 入口设置，各 kernel 辅助函数从这里取）。
///
/// 每线程独立的 non-blocking 流使多个 server 线程可并行提交推理；
/// 未设置时回退到 legacy 默认流（单算子测试路径）。
thread_local! {
    static ACTIVE_STREAM: RefCell<Option<Arc<CudaStream>>> = const { RefCell::new(None) };
    /// CUDA Graph capture 进行中：跳过 profile 事件与 debug 的 host 回拷
    /// （两者都会破坏 capture 或使 graph 重放产生悬垂 host 缓冲）。
    static CAPTURING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// 取当前活跃流，未设置则回退 legacy 默认流。
fn active_stream(rt: &CudaRuntime) -> Arc<CudaStream> {
    ACTIVE_STREAM.with(|s| {
        s.borrow()
            .as_ref()
            .cloned()
            .unwrap_or_else(|| rt.device.default_stream())
    })
}

/// 取当前线程活跃流的 Arc 克隆(C1 计时选算法需要 Arc 才能分配 scratch
/// 并在同一流上计时,保证与真实工作同流有序);非 apply 语境返回 None。
pub(crate) fn active_stream_clone() -> Option<Arc<CudaStream>> {
    ACTIVE_STREAM.with(|s| s.borrow().as_ref().cloned())
}

/// capture 模式是否开启（apply 内部据此跳过不可重放的操作）。
pub fn capturing() -> bool {
    CAPTURING.with(|c| c.get())
}

/// 设置当前线程的 capture 模式（CUDA Graph 捕获前后调用）。
pub fn set_capturing(v: bool) {
    CAPTURING.with(|c| c.set(v));
}

// ---------------------------------------------------------------------------
// 设备侧权重缓冲
// ---------------------------------------------------------------------------

/// GEMM 权重（设备侧）：out-first `[n, kp]` f16 行主序，K 已 pad 到 16 的倍数。
pub struct WeightBuf {
    pub data: CudaSlice<u16>,
    pub n: usize,
    pub k: usize,
    pub kp: usize,
}

/// f32 参数向量（设备侧，长度 `len`）。
pub struct ParamBuf {
    pub data: CudaSlice<f32>,
    pub len: usize,
}

/// 每层上传到设备的权重/参数。
enum LayerBuf {
    InitialConv {
        w: WeightBuf,
        gw: CudaSlice<f32>,
        g: usize,
        gate_scale: ParamBuf,
        gate_bias: ParamBuf,
    },
    Linear {
        w: WeightBuf,
        bias: Option<ParamBuf>,
        act_silu: bool,
        residual_add: bool,
    },
    RmsNorm {
        scale: ParamBuf,
        eps: f32,
        channels: usize,
    },
    Attention {
        qkv: WeightBuf,
        out: WeightBuf,
        cos: CudaSlice<f32>,
        sin: CudaSlice<f32>,
        qk_scale: f32,
        h: usize,
        d: usize,
        s: usize,
    },
    Ffn {
        dual: WeightBuf,
        down: WeightBuf,
        hidden: usize,
    },
    GateSilu {
        scale: ParamBuf,
        bias: ParamBuf,
        channels: usize,
    },
    TrunkFinal {
        mean: ParamBuf,
        std: ParamBuf,
        gamma: ParamBuf,
        beta: ParamBuf,
        channels: usize,
    },
    PolicyHead {
        /// 合并的 conv1p/conv1g [192, 768]（G4 头融合，-1 kernel）。
        conv1pg: WeightBuf,
        g_bias: ParamBuf,
        g_matmul: WeightBuf,
        pass1: WeightBuf,
        pass_b1: ParamBuf,
        pass2: WeightBuf,
        bias2: ParamBuf,
        conv2p: WeightBuf,
        mask_scale: f32,
    },
    ValueHead {
        conv1: WeightBuf,
        bias1: ParamBuf,
        l2: WeightBuf,
        l2_bias: ParamBuf,
        /// 合并的 value/misc/moremisc 输出权重 [21, 192]（G4 头融合）。
        vall_w: WeightBuf,
        /// 合并的 bias [21]。
        vall_b: ParamBuf,
        own_w: WeightBuf,
        mask_scale: f32,
        mask_quad: f32,
    },
}

// ---------------------------------------------------------------------------
// CudaModel
// ---------------------------------------------------------------------------

/// 已加载的 CUDA 模型：全部层权重驻留设备，`apply` 前向执行。
pub struct CudaModel {
    layers: Vec<LayerBuf>,
    seq_len: usize,
    trunk: usize,
    mid: usize,
    num_heads: usize,
    head_dim: usize,
    /// B2 dual-FFN(CUTLASS DualGemm)主机句柄;build 无 CUTLASS 时None。
    #[cfg(katago_dualffn)]
    dual_ffn: Option<DualFfnState>,
}

/// B2:CUTLASS DualGemm 的 FFI 句柄(内部状态非线程安全,经 Mutex 串行)。
#[cfg(katago_dualffn)]
pub struct DualFfnState(std::sync::Mutex<*mut std::ffi::c_void>);
#[cfg(katago_dualffn)]
unsafe impl Send for DualFfnState {}
#[cfg(katago_dualffn)]
unsafe impl Sync for DualFfnState {}
#[cfg(katago_dualffn)]
impl Drop for DualFfnState {
    fn drop(&mut self) {
        let h = self.0.lock().unwrap();
        if !h.is_null() {
            unsafe { dual_ffn_ffi::katago_dual_ffn_destroy(*h) };
        }
    }
}

#[cfg(katago_dualffn)]
mod dual_ffn_ffi {
    use std::ffi::c_void;
    unsafe extern "C" {
        pub fn katago_dual_ffn_create() -> *mut c_void;
        pub fn katago_dual_ffn_destroy(h: *mut c_void);
        pub fn katago_dual_ffn_exec(
            h: *mut c_void,
            input: *const c_void,
            gate: *const c_void,
            up: *const c_void,
            output: *mut c_void,
            tokens: i32,
            stream: usize,
        ) -> i32;
    }
}

/// B2 开关(KATAGO_CUDA_DUALFFN=1):dual FFN + SwiGLU epilogue 一步出
/// act1152,省 act2304 往返与独立 swiglu kernel。
pub(crate) fn dual_ffn_enabled() -> bool {
    crate::tactic_plan::tactic_var("KATAGO_CUDA_DUALFFN").as_deref() == Ok("1")
}

/// B2 执行入口:成功返回 true(act1152 已写好);tactic 关/不可用返回
/// false(调用方走 hgemm_f16 + swiglu_dual 现有路径);执行错误返回 Err
/// (fail-closed,与 tactic 路径约定一致)。
#[cfg(katago_dualffn)]
fn dual_ffn_run(
    state: &DualFfnState,
    rt: &CudaRuntime,
    input: &CudaSlice<u16>,
    dual_w: &WeightBuf,
    out: &mut CudaSlice<u16>,
    m: usize,
) -> Result<bool, String> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    if !dual_ffn_enabled() {
        return Ok(false);
    }
    if dual_w.n != 2304 || dual_w.k != 384 || dual_w.kp != 384 {
        return Err(format!(
            "dual_ffn 权重形状不支持:n={} k={} kp={}(要求 2304/384/384)",
            dual_w.n, dual_w.k, dual_w.kp
        ));
    }
    let stream = active_stream(rt);
    let (in_ptr, _g1) = input.device_ptr(&stream);
    let (w_ptr, _g2) = dual_w.data.device_ptr(&stream);
    let (out_ptr, _g3) = out.device_ptr_mut(&stream);
    // packed [2304, 384] 行主序:前 1152 行 gate,后 1152 行 up(零拷贝切片)。
    let gate_ptr = w_ptr;
    let up_ptr = w_ptr + (1152 * 384 * 2) as u64;
    let h = state.0.lock().unwrap();
    let r = unsafe {
        dual_ffn_ffi::katago_dual_ffn_exec(
            *h,
            in_ptr as *const std::ffi::c_void,
            gate_ptr as *const std::ffi::c_void,
            up_ptr as *const std::ffi::c_void,
            out_ptr as *mut std::ffi::c_void,
            m as i32,
            stream.cu_stream() as usize,
        )
    };
    if r == 0 {
        Ok(true)
    } else {
        Err(format!("dual_ffn exec failed: {r}"))
    }
}

impl CudaModel {
    /// 把层图全部权重上传设备（f32 → f16，GEMM K pad 16）。
    /// 在 `stream` 上执行（调用方负责与其他流间的同步）。
    pub fn load(graph: &LayerGraph, rt: &CudaRuntime, stream: &Arc<CudaStream>) -> Result<Self, String> {
        let stream = stream.clone();
        let mut layers = Vec::with_capacity(graph.layers.len());
        for layer in &graph.layers {
            layers.push(upload_layer(&stream, layer)?);
        }
        Ok(Self {
            layers,
            seq_len: graph.board_size * graph.board_size,
            trunk: graph.trunk_channels,
            mid: graph.mid_channels,
            num_heads: graph.num_heads,
            head_dim: graph.head_dim,
            #[cfg(katago_dualffn)]
            dual_ffn: {
                let h = unsafe { dual_ffn_ffi::katago_dual_ffn_create() };
                if h.is_null() {
                    None
                } else {
                    Some(DualFfnState(std::sync::Mutex::new(h)))
                }
            },
        })
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// 前向推理：`spatial` [B,22,19,19] NCHW f32、`global` [B,19] f32。
    ///
    /// 输出为模型原始张量（未 softmax、未后处理），写入 `ws` 的 out_* 缓冲，
    /// 与 ONNX 逐位对齐口径。全部 kernel 在 `stream` 上提交
    /// （线程内设置活跃流供辅助函数取用）。
    pub fn apply(
        &self,
        rt: &CudaRuntime,
        stream: &Arc<CudaStream>,
        ws: &mut CudaWorkspace,
        spatial: &CudaSlice<f32>,
        global: &CudaSlice<f32>,
    ) -> Result<(), String> {
        ACTIVE_STREAM.with(|s| *s.borrow_mut() = Some(stream.clone()));
        let batch = ws.batch;
        let s = self.seq_len;
        let m = batch * s;
        let h = self.num_heads;
        let d = self.head_dim;
        let trunk = self.trunk;
        let mid = self.mid;
        assert_eq!((mid, h, d, s), (384, 12, 32, 361), "执行器仅支持 b11 19 路架构");
        let stream = active_stream(rt);

        // --- 工作区（预分配复用；batch 由 ws 决定） ------------------------
        let cols = &mut ws.cols;
        let act768 = &mut ws.act768;
        let gated768 = &mut ws.gated768;
        let act384 = &mut ws.act384;
        let act384f16 = &mut ws.act384f16;
        let normed = &mut ws.normed;
        let act1152 = &mut ws.act1152;
        let act2304 = &mut ws.act2304;
        let gemm_out = &mut ws.gemm_out;
        let qbuf = &mut ws.qbuf;
        let kbuf = &mut ws.kbuf;
        let vbuf = &mut ws.vbuf;
        let p96f = &mut ws.p96f;
        let p192 = &mut ws.p192;
        let conv1p = &mut ws.conv1p;
        let pooled = &mut ws.pooled;
        let vec_a = &mut ws.vec_a;
        let vec_b = &mut ws.vec_b;
        let vall = &mut ws.vall;
        let conv1pg = &mut ws.conv1pg;
        let c_partial = &mut ws.c_partial;
        let pass_logits = &mut ws.pass_logits;
        let out_policy = &mut ws.out_policy;
        let out_value = &mut ws.out_value;
        let out_misc = &mut ws.out_misc;
        let out_moremisc = &mut ws.out_moremisc;
        let out_ownership = &mut ws.out_ownership;

        // --- 逐层执行 ----------------------------------------------------
        // per-layer 剖析（KATAGO_CUDA_PROFILE=1）：每层一对事件计时。
        // capture 模式下跳过（事件同步会破坏 graph capture）。
        let profiling = !capturing() && std::env::var("KATAGO_CUDA_PROFILE").is_ok();
        let mut layer_times: Vec<(i64, &'static str, f32)> = Vec::new();
        let (ev_start, ev_end) = if profiling {
            let f = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
            (
                Some(rt.device.new_event(f).map_err(|e| format!("event: {e}"))?),
                Some(rt.device.new_event(f).map_err(|e| format!("event: {e}"))?),
            )
        } else {
            (None, None)
        };
        if profiling {
            let _ = ev_start.as_ref().unwrap().record(&stream);
        }
        im2col(rt, spatial, cols, batch, s)?;
        if profiling {
            let _ = ev_end.as_ref().unwrap().record(&stream);
            let ms = ev_start
                .as_ref()
                .unwrap()
                .elapsed_ms(ev_end.as_ref().unwrap())
                .unwrap_or(-1.0);
            layer_times.push((-1, "im2col", ms));
        }
        // split-K 状态：Some(beta) 表示上一个 GEMM 是 split-K partial，
        // 待下一个 RmsNorm 融合求和。KATAGO_CUDA_SPLITK=1 启用。
        let splitk_on = crate::tactic_plan::tactic_enabled("KATAGO_CUDA_SPLITK");
        let mut pending_splitk_beta: Option<f32> = None;
        let mut li = 0usize;
        while li < self.layers.len() {
            let lb = &self.layers[li];
            let layer_tag: &'static str = match lb {
                LayerBuf::InitialConv { .. } => "InitialConv",
                LayerBuf::Linear { .. } => "Linear",
                LayerBuf::RmsNorm { .. } => "RmsNorm",
                LayerBuf::Attention { .. } => "Attention",
                LayerBuf::Ffn { .. } => "Ffn",
                LayerBuf::GateSilu { .. } => "GateSilu",
                LayerBuf::TrunkFinal { .. } => "TrunkFinal",
                LayerBuf::PolicyHead { .. } => "PolicyHead",
                LayerBuf::ValueHead { .. } => "ValueHead",
            };
            if profiling {
                let _ = ev_start.as_ref().unwrap().record(&stream);
            }
            // 融合前瞻：返回 true 表示已把下一层（GateSilu）一并消费。
            // KATAGO_CUDA_FUSION=none|up|down|all。默认 none：融合 epilogue
            // 走手写 GEMM，cuBLASLt（默认开）下慢于「cuBLASLt + 独立
            // elementwise」（ABBA 2026-08-15：t=1 302→374、t=8 621→678
            // nnEvals/s）；纯手写 GEMM（CUBLASLT=0）时融合仍是收益，
            // 供 autotune 按组合实测选择。
            let fusion_mode = crate::tactic_plan::tactic_var("KATAGO_CUDA_FUSION")
                .unwrap_or_else(|_| "none".to_string());
            let fuse_up = fusion_mode == "all" || fusion_mode == "up";
            let fuse_down = fusion_mode == "all" || fusion_mode == "down";
            let skip_next = match lb {
                LayerBuf::InitialConv { w, gw, g, gate_scale, gate_bias } => {
                    // raw 768 流（块残差加的是 gate 前的值）+ 异位门控 SiLU
                    // 融合：conv_bias_gate epilogue 同时写 f32 残差与 f16 gated。
                    hgemm(rt, cols, w, gemm_out, m)?;
                    conv_bias_gate(
                        rt, gemm_out, global, gw, act768, gated768,
                        &gate_scale.data, &gate_bias.data, batch, s, trunk, *g,
                    )?;
                    false
                }
                // 1x1 线性。本模型两种：768→384 下投影（写 act384）与
                // 384→768 上投影（残差加回 act768，IR 的块残差 Add）。
                // match_linear 恒产 bias=None/act=None/residual=false。
                LayerBuf::Linear { w, bias, act_silu, residual_add } => {
                    assert!(
                        bias.is_none() && !act_silu && !residual_add,
                        "Linear 的 bias/act/residual 组合本执行器未实现（IR 恒为 None/false/false）"
                    );
                    if w.k == trunk && w.n == mid {
                        // Linear down（K=768, N=384, beta=0）：split-K partial
                        //（reduce 由后续 RmsNorm 融合）。KATAGO_CUDA_SPLITK=1。
                        if splitk_on {
                            hgemm_splitk2_partial(rt, gated768, w, c_partial, m)?;
                            pending_splitk_beta = Some(0.0);
                        } else {
                            hgemm(rt, gated768, w, act384, m)?;
                        }
                        false
                    } else if w.k == mid && w.n == trunk {
                        // 前瞻：后一层是 GateSilu(768) 时融合（up 残差 GEMM
                        // epilogue 同时写 f32 残差与 f16 gated 流）。
                        if fuse_up
                            && let Some(LayerBuf::GateSilu { scale, bias, channels }) =
                                self.layers.get(li + 1)
                        {
                            if *channels == trunk {
                                // act384f16 已由块尾 Ffn down 的 gatesilu
                                // epilogue 写好（f16 双写），直接读
                                hgemm_residual_gatesilu_f16(
                                    rt,
                                    act384f16,
                                    w,
                                    act768,
                                    gated768,
                                    &scale.data,
                                    &bias.data,
                                    m,
                                )?;
                                true
                            } else {
                                f32_to_f16(rt, act384, act384f16, m * mid)?;
                                // 上投影 GEMM beta=1 直接残差进 act768（省独立残差 kernel）
                                hgemm_residual(rt, act384f16, w, act768, m)?;
                                false
                            }
                        } else {
                            // 未融合（FUSION=none/down）时独立 GateSilu(384)
                            // 原地写 act384 f32，此处须转换出 f16 流
                            // （融合路径由 down 的 gatesilu epilogue 双写省掉本步）。
                            f32_to_f16(rt, act384, act384f16, m * mid)?;
                            hgemm_residual(rt, act384f16, w, act768, m)?;
                            false
                        }
                    } else {
                        return Err(format!("Linear 形状不支持: k={} n={}", w.k, w.n));
                    }
                }
                LayerBuf::RmsNorm { scale, eps, channels } => {
                    assert_eq!(*channels, mid, "RMSNorm 仅出现在 384 维流");
                    if let Some(beta) = pending_splitk_beta.take() {
                        // split-K partial 求和 + 残差 + RMSNorm 融合
                        rms_norm_splitk(
                            rt, act384, c_partial, &scale.data, normed, beta,
                            *eps, *channels, m, 2,
                        )?;
                    } else {
                        rms_norm_f32(rt, act384, normed, &scale.data, *eps, *channels, m)?;
                    }
                    false
                }
                LayerBuf::Attention { qkv, out, cos, sin, qk_scale, h: lh, d: ld, s: ls } => {
                    let (lh, ld, ls) = (*lh, *ld, *ls);
                    assert_eq!((lh, ld, ls), (h, d, s), "attention 结构参数不符");
                    let sub_profile = profiling;
                    let mut sub_times: Vec<(&'static str, f32)> = Vec::new();
                    macro_rules! timed {
                        ($name:expr, $body:expr) => {
                            if sub_profile {
                                let _ = ev_start.as_ref().unwrap().record(&stream);
                            }
                            $body?;
                            if sub_profile {
                                let _ = ev_end.as_ref().unwrap().record(&stream);
                                let ms = ev_start
                                    .as_ref()
                                    .unwrap()
                                    .elapsed_ms(ev_end.as_ref().unwrap())
                                    .unwrap_or(-1.0);
                                sub_times.push(($name, ms));
                            }
                        };
                    }
                    let scale = qk_scale * qk_scale; // 导出图 q、k 各乘 1/∜d
                    // FA2：qkv GEMM f16 packed([M,1152] 复用 act1152)→
                    // attention 内部加载时做 RoPE（省独立 rope kernel）。
                    // v3 回退：f32 GEMM + 独立 rope 拆分 qbuf/kbuf/vbuf。
                    let use_v3 = crate::tactic_plan::tactic_var("KATAGO_CUDA_ATTN").as_deref() == Ok("v3");
                    if use_v3 {
                        timed!("qkv", hgemm(rt, normed, qkv, gemm_out, m));
                        timed!(
                            "rope",
                            qkv_rope(
                                rt, gemm_out, cos, sin, qbuf, kbuf, vbuf, batch, lh, ls,
                                ld,
                            )
                        );
                        timed!("attn", attention_row(rt, qbuf, kbuf, vbuf, normed, ls, ld, batch * lh, scale, lh));
                    } else {
                        timed!("qkv", hgemm_f16(rt, normed, qkv, act1152, m));
                        timed!("attn", attention_fa2(rt, act1152, cos, sin, normed, ls, ld, batch * lh, scale, lh));
                    }
                    // 输出投影 beta=1 残差进 act384。split-K partial（reduce
                    // 由后续 RmsNorm 融合）。KATAGO_CUDA_SPLITK=1 启用。
                    if splitk_on {
                        hgemm_splitk2_partial(rt, normed, out, c_partial, m)?;
                        pending_splitk_beta = Some(1.0);
                    } else {
                        timed!("outproj", hgemm_residual(rt, normed, out, act384, m));
                    }
                    if sub_profile {
                        let total: f32 = sub_times.iter().map(|t| t.1.max(0.0)).sum();
                        eprintln!("[cuda-profile] attention l{li} total {total:.3} ms: {}",
                            sub_times.iter().map(|(n, t)| format!("{n}={t:.3}")).collect::<Vec<_>>().join(" "));
                    }
                    false
                }
                LayerBuf::Ffn { dual, down, hidden } => {
                    let hidden = *hidden;
                    assert_eq!(hidden, 3 * mid, "FFN 隐层维度不符");
                    // B2(KATAGO_CUDA_DUALFFN=1):CUTLASS DualGemm + SwiGLU
                    // epilogue 一步出 act1152,省 act2304 往返 + swiglu kernel。
                    // 数值与两 kernel 路径逐位一致(中间 half 舍入在 epilogue
                    // 内复刻)。不可用/未启用时走现有路径。
                    #[cfg(katago_dualffn)]
                    let dual_done = match &self.dual_ffn {
                        Some(st) => dual_ffn_run(st, rt, normed, dual, act1152, m)?,
                        None => false,
                    };
                    #[cfg(not(katago_dualffn))]
                    let dual_done = false;
                    if !dual_done {
                        // dual FFN:gate|up 单 GEMM(N=2304,f16 直出)→ dual SwiGLU
                        hgemm_f16(rt, normed, dual, act2304, m)?;
                        swiglu_dual(rt, act2304, act1152, m, hidden)?;
                    }
                    // 前瞻：后一层是 GateSilu(384) 时融合（down 残差 GEMM
                    // epilogue 直接算 silu(affine)，省独立 gate kernel）。
                    if fuse_down
                        && let Some(LayerBuf::GateSilu { scale, bias, channels }) =
                            self.layers.get(li + 1)
                    {
                        if *channels == mid {
                            // down 残差 + GateSilu384 融合，epilogue 同时写
                            // act384f16（供 up GEMM 直接读，省 f32_to_f16）
                            hgemm_residual_gatesilu_f32(
                                rt, act1152, down, act384, act384f16,
                                &scale.data, &bias.data, m,
                            )?;
                            true
                        } else {
                            hgemm_residual(rt, act1152, down, act384, m)?;
                            false
                        }
                    } else {
                        hgemm_residual(rt, act1152, down, act384, m)?;
                        false
                    }
                }
                LayerBuf::GateSilu { scale, bias, channels } => {
                    // 384 维（块尾）与 768 维（块边界）两种门。
                    // 通常已被上一层的融合前瞻消费；独立路径保留兜底。
                    if *channels == mid {
                        gate_silu_f32(rt, act384, &scale.data, &bias.data, m * mid, *channels)?;
                    } else if *channels == trunk {
                        gate_silu_out(
                            rt,
                            act768,
                            &scale.data,
                            &bias.data,
                            gated768,
                            m * trunk,
                            *channels,
                        )?;
                    } else {
                        return Err(format!("GateSilu 通道数 {channels} 不支持"));
                    }
                    false
                }
                LayerBuf::TrunkFinal { mean, std, gamma, beta, channels } => {
                    assert_eq!(*channels, trunk, "TrunkFinal 仅支持 768 维");
                    bn_silu(
                        rt, act768, &mean.data, &std.data, &gamma.data, &beta.data,
                        gated768, m * trunk, *channels,
                    )?;
                    false
                }
                LayerBuf::PolicyHead {
                    conv1pg: c1pg,
                    g_bias,
                    g_matmul,
                    pass1,
                    pass_b1,
                    pass2,
                    bias2,
                    conv2p,
                    mask_scale,
                } => {
                    // conv1p|conv1g 合并 GEMM（N=192，G4）→ conv1pg [M,192] f32
                    hgemm(rt, gated768, c1pg, conv1pg, m)?;
                    // conv1g 部分（后 96 列）→ BN+silu → p192
                    add_bias_silu_f16_ld(rt, conv1pg, 96, 192, &g_bias.data, p192, m * 96, 96)?;
                    // 池化：[mean, mean*scale, max]（f32；小头链路保持 f32 精度）
                    pool_mean_max(rt, p192, pooled, batch, s, 96, *mask_scale)?;
                    // pass 分支融合（G4）：pooled@pass1+silu → @pass2（-2 kernel）
                    policy_pass_fused(rt, pooled, pass1, &pass_b1.data, pass2, pass_logits, batch)?;
                    // g 分支：Gemm → +conv1p → +bias2 → SiLU → conv2p
                    sgemm(rt, pooled, g_matmul, vec_b, batch)?;
                    // conv1p 部分（前 96 列）
                    policy_g_ld(rt, conv1pg, 0, 192, vec_b, &bias2.data, p96f, batch, s, 96)?;
                    sgemm(rt, p96f, conv2p, gemm_out, m)?;
                    // 拼接 [B,6,362]；penalty=0（19 路无非法落点，结构保留）
                    policy_concat(rt, gemm_out, pass_logits, out_policy, batch, s, 6, 0.0)?;
                    false
                }
                LayerBuf::ValueHead {
                    conv1,
                    bias1,
                    l2,
                    l2_bias,
                    vall_w,
                    vall_b,
                    own_w,
                    mask_scale,
                    mask_quad,
                } => {
                    // conv1 + bias + SiLU → v_act [M,192]
                    hgemm(rt, gated768, conv1, gemm_out, m)?;
                    add_bias_silu_f16(rt, gemm_out, &bias1.data, p192, m * 192, 192)?;
                    // 池化：[mean, mean*scale, mean*quad]（f32）
                    pool_mean3(rt, p192, pooled, batch, s, 192, *mask_scale, *mask_quad)?;
                    // linear2+silu+输出合并+bias 拆分（G4 单 kernel 写 3 目标）
                    value_fc_fused(
                        rt, pooled, l2, &l2_bias.data, vall_w, &vall_b.data,
                        out_value, out_misc, out_moremisc, batch,
                    )?;
                    // ownership 1x1（×mask 恒 1，省略；v_act 为 f16 激活流）
                    hgemm(rt, p192, own_w, out_ownership, m)?;
                    false
                }
            };
            if profiling {
                let _ = ev_end.as_ref().unwrap().record(&stream);
                let ms = ev_start
                    .as_ref()
                    .unwrap()
                    .elapsed_ms(ev_end.as_ref().unwrap())
                    .unwrap_or(-1.0);
                layer_times.push((li as i64, layer_tag, ms));
            }
            li += 1 + skip_next as usize;
            // --- 临时调试钩子：KATAGO_CUDA_DEBUG_LAYER=<i> 时 dump 两路流 ---
            // capture 模式下跳过（host 回拷不可重放）。
            if !capturing() {
            if let Ok(spec) = std::env::var("KATAGO_CUDA_DEBUG_LAYER") {
                if spec == li.to_string() {
                    let tag = match lb {
                        LayerBuf::InitialConv { .. } => "InitialConv",
                        LayerBuf::Linear { .. } => "Linear",
                        LayerBuf::RmsNorm { .. } => "RmsNorm",
                        LayerBuf::Attention { .. } => "Attention",
                        LayerBuf::Ffn { .. } => "Ffn",
                        LayerBuf::GateSilu { channels, .. } => {
                            if *channels == 768 { "GateSilu768" } else { "GateSilu384" }
                        }
                        LayerBuf::TrunkFinal { .. } => "TrunkFinal",
                        LayerBuf::PolicyHead { .. } => "PolicyHead",
                        LayerBuf::ValueHead { .. } => "ValueHead",
                    };
                    eprintln!("DEBUG layer {li} = {tag}");
                    let w = |name: &str, d: &CudaSlice<u16>| -> Result<(), String> {
                        let mut host = vec![0u16; d.len()];
                        stream
                            .memcpy_dtoh(d, &mut host)
                            .map_err(|e| e.to_string())?;
                        let f32v: Vec<f32> = host.iter().map(|&b| f16_to_f32_bits(b)).collect();
                        let bytes: Vec<u8> = f32v.iter().flat_map(|f| f.to_le_bytes()).collect();
                        std::fs::write(format!("target/cuda_debug_l{li}_{name}.bin"), bytes)
                            .map_err(|e| e.to_string())?;
                        Ok(())
                    };
                    w("act384f16", act384f16)?;
                    {
                        // act384 为 f32 流，走 f32 导出
                        let mut host = vec![0.0f32; act384.len()];
                        stream
                            .memcpy_dtoh(act384, &mut host)
                            .map_err(|e| e.to_string())?;
                        let bytes: Vec<u8> =
                            host.iter().flat_map(|f| f.to_le_bytes()).collect();
                        std::fs::write(format!("target/cuda_debug_l{li}_act384.bin"), bytes)
                            .map_err(|e| e.to_string())?;
                    }
                    {
                        // act768 为 f32 流，走 f32 导出
                        let mut host = vec![0.0f32; act768.len()];
                        stream
                            .memcpy_dtoh(act768, &mut host)
                            .map_err(|e| e.to_string())?;
                        let bytes: Vec<u8> =
                            host.iter().flat_map(|f| f.to_le_bytes()).collect();
                        std::fs::write(format!("target/cuda_debug_l{li}_act768.bin"), bytes)
                            .map_err(|e| e.to_string())?;
                    }
                    for (name, d) in [
                        ("normed", &*normed),
                        ("act1152", &*act1152),
                        ("act2304", &*act2304),
                        ("gated768", &*gated768),
                    ] {
                        let mut host = vec![0u16; d.len()];
                        stream
                            .memcpy_dtoh(d, &mut host)
                            .map_err(|e| e.to_string())?;
                        let f32v: Vec<f32> = host.iter().map(|&b| f16_to_f32_bits(b)).collect();
                        let bytes: Vec<u8> = f32v.iter().flat_map(|f| f.to_le_bytes()).collect();
                        std::fs::write(format!("target/cuda_debug_l{li}_{name}.bin"), bytes)
                            .map_err(|e| e.to_string())?;
                    }
                    // 全部 dump 的 memcpy 排队后统一同步（消除 host 读竞态）。
                    let _ = stream.synchronize();
                    if li == self.layers.len() - 1 {
                        let wf = |name: &str, d: &CudaSlice<f32>| -> Result<(), String> {
                            let mut host = vec![0.0f32; d.len()];
                            stream
                                .memcpy_dtoh(d, &mut host)
                                .map_err(|e| e.to_string())?;
                            let bytes: Vec<u8> =
                                host.iter().flat_map(|f| f.to_le_bytes()).collect();
                            std::fs::write(format!("target/cuda_debug_l{li}_{name}.bin"), bytes)
                                .map_err(|e| e.to_string())?;
                            Ok(())
                        };
                        wf("pooled", pooled)?;
                        wf("vec_a", vec_a)?;
                    }
                }
            }
            } // !capturing
        }

        if profiling {
            let total: f32 = layer_times.iter().map(|t| t.2.max(0.0)).sum();
            eprintln!("[cuda-profile] total {total:.3} ms across {} kernels:",
                layer_times.len());
            for (li, tag, ms) in &layer_times {
                eprintln!("  {li:3} {tag:12} {ms:9.3} ms");
            }
        }

        ACTIVE_STREAM.with(|s| *s.borrow_mut() = None);
        Ok(())
    }
}

fn upload_layer(stream: &StreamRef, layer: &Layer) -> Result<LayerBuf, String> {
    Ok(match layer {
        Layer::InitialConv(l) => LayerBuf::InitialConv {
            w: upload_weight(
                stream,
                &l.weight,
                l.out_channels,
                l.weight.numel() / l.out_channels,
            )?,
            gw: upload_f32(
                stream,
                l.global_weight.f32_data(),
                l.out_channels,
                l.global_weight.numel() / l.out_channels,
            )?,
            g: l.global_weight.numel() / l.out_channels,
            gate_scale: upload_param(stream, &l.gate_scale, l.out_channels)?,
            gate_bias: upload_param(stream, &l.gate_bias, l.out_channels)?,
        },
        Layer::Linear(l) => LayerBuf::Linear {
            w: upload_weight(stream, &l.weight, l.n, l.k)?,
            bias: match &l.bias {
                Some(b) => Some(upload_param(stream, b, l.n)?),
                None => None,
            },
            act_silu: l.act.is_some(),
            residual_add: l.residual_add,
        },
        Layer::RmsNorm(l) => LayerBuf::RmsNorm {
            scale: upload_param(stream, &l.scale, l.channels)?,
            eps: l.eps,
            channels: l.channels,
        },
        Layer::Attention(l) => LayerBuf::Attention {
            qkv: upload_weight(
                stream,
                &l.qkv_weight,
                3 * l.num_heads * l.head_dim,
                l.qkv_weight.numel() / (3 * l.num_heads * l.head_dim),
            )?,
            out: upload_weight(
                stream,
                &l.out_weight,
                l.num_heads * l.head_dim,
                l.out_weight.numel() / (l.num_heads * l.head_dim),
            )?,
            cos: upload_f32(
                stream,
                l.rope_cos.f32_data(),
                l.seq_len,
                l.num_heads * l.head_dim / 2,
            )?,
            sin: upload_f32(
                stream,
                l.rope_sin.f32_data(),
                l.seq_len,
                l.num_heads * l.head_dim / 2,
            )?,
            qk_scale: l.qk_scale,
            h: l.num_heads,
            d: l.head_dim,
            s: l.seq_len,
        },
        Layer::Ffn(l) => LayerBuf::Ffn {
            // dual FFN：gate/up 合并为 [2H, K] 单 GEMM（前 H 行 gate、后 H 行 up），
            // N=2304 的 grid 更大，SM 占用更好（fork G1 dual_ffn 认证路径）。
            dual: upload_weight_concat2(
                stream,
                &l.gate_weight,
                &l.up_weight,
                l.hidden,
                l.up_weight.numel() / l.hidden,
            )?,
            down: upload_weight(
                stream,
                &l.down_weight,
                l.down_weight.numel() / l.hidden,
                l.hidden,
            )?,
            hidden: l.hidden,
        },
        Layer::GateSilu(l) => LayerBuf::GateSilu {
            scale: upload_param(stream, &l.scale, l.channels)?,
            bias: upload_param(stream, &l.bias, l.channels)?,
            channels: l.channels,
        },
        Layer::TrunkFinal(l) => LayerBuf::TrunkFinal {
            mean: upload_param(stream, &l.mean, l.channels)?,
            std: upload_param(stream, &l.std, l.channels)?,
            gamma: upload_param(stream, &l.gamma, l.channels)?,
            beta: upload_param(stream, &l.beta, l.channels)?,
            channels: l.channels,
        },
        Layer::PolicyHead(l) => LayerBuf::PolicyHead {
            conv1pg: upload_weight_concat2(
                stream,
                &l.conv1p_weight,
                &l.conv1g_weight,
                96,
                768,
            )?,
            g_bias: upload_param(stream, &l.g_bias, 96)?,
            g_matmul: upload_weight(stream, &l.g_matmul, 96, 288)?,
            pass1: upload_weight(stream, &l.pass_matmul1, 96, 288)?,
            pass_b1: upload_param(stream, &l.pass_bias1, 96)?,
            pass2: upload_weight(stream, &l.pass_matmul2, 6, 96)?,
            bias2: upload_param(stream, &l.bias2, 96)?,
            conv2p: upload_weight(stream, &l.conv2p_weight, 6, 96)?,
            mask_scale: l.mask_scale,
        },
        Layer::ValueHead(l) => LayerBuf::ValueHead {
            conv1: upload_weight(stream, &l.conv1_weight, 192, 768)?,
            bias1: upload_param(stream, &l.bias1, 192)?,
            l2: upload_weight(stream, &l.linear2_weight, 192, 576)?,
            l2_bias: upload_param(stream, &l.linear2_bias, 192)?,
            // value/misc/moremisc 三输出合并 [21, 192] + bias [21]
            vall_w: upload_weight_concat3(
                stream,
                &l.value_matmul,
                &l.misc_matmul,
                &l.moremisc_matmul,
                &[3, 10, 8],
                192,
            )?,
            vall_b: upload_param_concat3(
                stream,
                &l.value_bias,
                &l.misc_bias,
                &l.moremisc_bias,
            )?,
            own_w: upload_weight(stream, &l.ownership_conv, 1, 192)?,
            mask_scale: l.mask_scale,
            mask_quad: l.mask_quad,
        },
    })
}

// ---------------------------------------------------------------------------
// 上传辅助
// ---------------------------------------------------------------------------

fn upload_weight(stream: &StreamRef, t: &Tensor, n: usize, k: usize) -> Result<WeightBuf, String> {
    let data = match &t.data {
        TensorData::F32(d) => d,
        TensorData::I64(_) => panic!("层图权重必须是 F32（I64 仅用于形状常量）"),
    };
    assert_eq!(data.len(), n * k, "权重形状与 IR 声明不符: {:?}", t.dims);
    let kp = k.div_ceil(16) * 16;
    let mut host = vec![0u16; n * kp];
    for i in 0..n {
        for j in 0..k {
            host[i * kp + j] = f32_to_f16_bits(data[i * k + j]);
        }
    }
    let mut dev: CudaSlice<u16> = unsafe { stream.alloc(n * kp) }.map_err(|e| e.to_string())?;
    stream
        .memcpy_htod(host.as_slice(), &mut dev)
        .map_err(|e| e.to_string())?;
    Ok(WeightBuf { data: dev, n, k, kp })
}

/// 两个权重行拼接上传：`[t1; t2]` → `[2*n_each, k]` f16（dual FFN 用）。
fn upload_weight_concat2(
    stream: &StreamRef,
    t1: &Tensor,
    t2: &Tensor,
    n_each: usize,
    k: usize,
) -> Result<WeightBuf, String> {
    let (d1, d2) = match (&t1.data, &t2.data) {
        (TensorData::F32(a), TensorData::F32(b)) => (a, b),
        _ => panic!("层图权重必须是 F32"),
    };
    assert_eq!(d1.len(), n_each * k, "t1 形状与 IR 声明不符");
    assert_eq!(d2.len(), n_each * k, "t2 形状与 IR 声明不符");
    let kp = k.div_ceil(16) * 16;
    let mut host = vec![0u16; 2 * n_each * kp];
    for i in 0..n_each {
        for j in 0..k {
            host[i * kp + j] = f32_to_f16_bits(d1[i * k + j]);
            host[(n_each + i) * kp + j] = f32_to_f16_bits(d2[i * k + j]);
        }
    }
    let mut dev: CudaSlice<u16> = unsafe { stream.alloc(2 * n_each * kp) }.map_err(|e| e.to_string())?;
    stream
        .memcpy_htod(host.as_slice(), &mut dev)
        .map_err(|e| e.to_string())?;
    Ok(WeightBuf { data: dev, n: 2 * n_each, k, kp })
}

/// 三个权重行拼接上传（G4 ValueHead 输出合并）：[t1; t2; t3] → [Σn, k]。
fn upload_weight_concat3(
    stream: &StreamRef,
    t1: &Tensor,
    t2: &Tensor,
    t3: &Tensor,
    ns: &[usize; 3],
    k: usize,
) -> Result<WeightBuf, String> {
    let (d1, d2, d3) = match (&t1.data, &t2.data, &t3.data) {
        (TensorData::F32(a), TensorData::F32(b), TensorData::F32(c)) => (a, b, c),
        _ => panic!("层图权重必须是 F32"),
    };
    assert_eq!(d1.len(), ns[0] * k);
    assert_eq!(d2.len(), ns[1] * k);
    assert_eq!(d3.len(), ns[2] * k);
    let kp = k.div_ceil(16) * 16;
    let total_n = ns[0] + ns[1] + ns[2];
    let mut host = vec![0u16; total_n * kp];
    for (ti, (d, n0)) in [(d1, ns[0]), (d2, ns[1]), (d3, ns[2])].iter().enumerate() {
        let base: usize = ns[..ti].iter().sum();
        for i in 0..*n0 {
            for j in 0..k {
                host[(base + i) * kp + j] = f32_to_f16_bits(d[i * k + j]);
            }
        }
    }
    let mut dev: CudaSlice<u16> = unsafe { stream.alloc(total_n * kp) }.map_err(|e| e.to_string())?;
    stream
        .memcpy_htod(host.as_slice(), &mut dev)
        .map_err(|e| e.to_string())?;
    Ok(WeightBuf { data: dev, n: total_n, k, kp })
}

/// 三个 f32 参数向量拼接上传。
fn upload_param_concat3(
    stream: &StreamRef,
    t1: &Tensor,
    t2: &Tensor,
    t3: &Tensor,
) -> Result<ParamBuf, String> {
    let (d1, d2, d3) = match (&t1.data, &t2.data, &t3.data) {
        (TensorData::F32(a), TensorData::F32(b), TensorData::F32(c)) => (a, b, c),
        _ => panic!("层图参数必须是 F32"),
    };
    let mut host = Vec::with_capacity(d1.len() + d2.len() + d3.len());
    host.extend_from_slice(d1);
    host.extend_from_slice(d2);
    host.extend_from_slice(d3);
    let mut dev: CudaSlice<f32> = unsafe { stream.alloc(host.len()) }.map_err(|e| e.to_string())?;
    stream.memcpy_htod(host.as_slice(), &mut dev).map_err(|e| e.to_string())?;
    Ok(ParamBuf { data: dev, len: host.len() })
}

fn upload_param(stream: &StreamRef, t: &Tensor, len: usize) -> Result<ParamBuf, String> {
    let data = match &t.data {
        TensorData::F32(d) => d,
        TensorData::I64(_) => panic!("层图参数必须是 F32"),
    };
    assert_eq!(data.len(), len, "参数形状与 IR 声明不符: {:?}", t.dims);
    let mut dev: CudaSlice<f32> = unsafe { stream.alloc(len) }.map_err(|e| e.to_string())?;
    stream.memcpy_htod(data, &mut dev).map_err(|e| e.to_string())?;
    Ok(ParamBuf { data: dev, len })
}

/// 上传 f32 向量（`rows × stride` 行主序读取源数据）。
fn upload_f32(
    stream: &StreamRef,
    data: &[f32],
    rows: usize,
    stride: usize,
) -> Result<CudaSlice<f32>, String> {
    assert_eq!(data.len(), rows * stride, "f32 上传尺寸不符");
    let mut dev: CudaSlice<f32> =
        unsafe { stream.alloc(rows * stride) }.map_err(|e| e.to_string())?;
    stream.memcpy_htod(data, &mut dev).map_err(|e| e.to_string())?;
    Ok(dev)
}

fn zeros16(stream: &StreamRef, n: usize) -> Result<CudaSlice<u16>, String> {
    stream.alloc_zeros(n).map_err(|e| e.to_string())
}

fn zeros32(stream: &StreamRef, n: usize) -> Result<CudaSlice<f32>, String> {
    stream.alloc_zeros(n).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// kernel 启动包装（全部直接走设备内存，不回拷）
// ---------------------------------------------------------------------------

/// `C[M,N] = A[M,K] @ B[N,K]^T`（f16 张量核；A 的列 stride 必须为 `b.kp`）。
/// M 较小时用 tile 64×64（grid 是 128-tile 的 4 倍，缓解小 batch 的
/// grid-starved SM 空闲）；M 大时用 128×128×32 双缓冲流水。
fn hgemm(
    rt: &CudaRuntime,
    a: &CudaSlice<u16>,
    b: &WeightBuf,
    c: &mut CudaSlice<f32>,
    m: usize,
) -> Result<(), String> {
    // tile 选择：N≤512 且 M 小 → t32(grid 是 t64 的 4 倍,解 N=384 的 starved);
    // M<1024 → t64;大 → v2(128)。
    let use_t32 = crate::tactic_plan::tactic_enabled("KATAGO_CUDA_T32") && m < 1024 && b.n <= 512;
    let use_t64n32 = crate::tactic_plan::tactic_enabled("KATAGO_CUDA_T64N32") && m < 1024 && b.n <= 512;
    let f = if use_t32 {
        rt.get_func("hgemm_t32_kernel")?
    } else if use_t64n32 {
        rt.get_func("hgemm_t64n32_kernel")?
    } else if m < 1024 {
        rt.get_func("hgemm_t64_kernel")?
    } else {
        rt.get_func("hgemm_v2_kernel")?
    };
    let stream = active_stream(rt);
    // cuBLASLt 旁路(KATAGO_CUDA_CUBLASLT=1,仅非 pad 的 f32 输出 GEMM)。
    if crate::tactic_plan::tactic_var("KATAGO_CUDA_CUBLASLT").as_deref() != Ok("0") && b.kp == b.k {
        if rt.cublaslt_gemm(&stream, a, &b.data, c, m, b.n, b.k, 0.0)? {
            return Ok(());
        }
    }
    let (tile_m, tile_n, kname) = if use_t32 { (32usize, 32usize, "hgemm_t32") } else if use_t64n32 { (64usize, 32usize, "hgemm_t64n32") } else if m < 1024 { (64usize, 64usize, "hgemm_t64") } else { (128usize, 128usize, "hgemm_v2") };
    let grid = (m.div_ceil(tile_m) as u32, b.n.div_ceil(tile_n) as u32, 1u32);
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: (if use_t32 { 128 } else { 256 }, 1, 1),
        shared_mem_bytes: 0,
    };
    let (alpha, beta) = (1.0f32, 0.0f32);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(&b.data)
            .arg(c)
            .arg(&(m as i32))
            .arg(&(b.n as i32))
            .arg(&(b.kp as i32))
            .arg(&alpha)
            .arg(&beta)
            .launch(cfg)
    }
    .map_err(|e| format!("{kname} launch failed: {e}"))?;
    Ok(())
}

/// `C[M,N] = A[M,K] @ B[N,K]^T` 的 f16 输出变体（GEMM epilogue 直接
/// `__float2half_rn`，取代 f32 GEMM + `f32_to_f16` 两次 launch）。
fn hgemm_f16(
    rt: &CudaRuntime,
    a: &CudaSlice<u16>,
    b: &WeightBuf,
    c: &mut CudaSlice<u16>,
    m: usize,
) -> Result<(), String> {
    // cuBLASLt f16 输出旁路（KATAGO_CUDA_CUBLASLT=1）。
    if crate::tactic_plan::tactic_var("KATAGO_CUDA_CUBLASLT").as_deref() != Ok("0") && b.kp == b.k {
        let stream0 = active_stream(rt);
        if rt.cublaslt_gemm_f16out(&stream0, a, &b.data, c, m, b.n, b.k)? {
            return Ok(());
        }
    }
    let use_t32 = crate::tactic_plan::tactic_enabled("KATAGO_CUDA_T32") && m < 1024 && b.n <= 512;
    let f = if use_t32 {
        rt.get_func("hgemm_t32_f16out_kernel")?
    } else if m < 1024 {
        rt.get_func("hgemm_t64_f16out_kernel")?
    } else {
        rt.get_func("hgemm_v2_f16out_kernel")?
    };
    let stream = active_stream(rt);
    let (tile, kname) = if use_t32 { (32usize, "hgemm_t32_f16out") } else if m < 1024 { (64usize, "hgemm_t64_f16out") } else { (128usize, "hgemm_v2_f16out") };
    let grid = (m.div_ceil(tile) as u32, b.n.div_ceil(tile) as u32, 1u32);
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: (if use_t32 { 128 } else { 256 }, 1, 1),
        shared_mem_bytes: 0,
    };
    let (alpha, beta) = (1.0f32, 0.0f32);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(&b.data)
            .arg(c)
            .arg(&(m as i32))
            .arg(&(b.n as i32))
            .arg(&(b.kp as i32))
            .arg(&alpha)
            .arg(&beta)
            .launch(cfg)
    }
    .map_err(|e| format!("{kname} launch failed: {e}"))?;
    Ok(())
}

/// `C[M,N] = A[M,K] @ B[N,K]^T + C`（beta=1 原位残差：epilogue 先读 C 再写 D，
/// C 与 D 同指针，省去独立的残差加 kernel）。
fn hgemm_residual(
    rt: &CudaRuntime,
    a: &CudaSlice<u16>,
    b: &WeightBuf,
    c: &mut CudaSlice<f32>,
    m: usize,
) -> Result<(), String> {
    let use_t32 = crate::tactic_plan::tactic_enabled("KATAGO_CUDA_T32") && m < 1024 && b.n <= 512;
    let use_t64n32 = crate::tactic_plan::tactic_enabled("KATAGO_CUDA_T64N32") && m < 1024 && b.n <= 512;
    let f = if use_t32 {
        rt.get_func("hgemm_t32_kernel")?
    } else if use_t64n32 {
        rt.get_func("hgemm_t64n32_kernel")?
    } else if m < 1024 {
        rt.get_func("hgemm_t64_kernel")?
    } else {
        rt.get_func("hgemm_v2_kernel")?
    };
    let stream = active_stream(rt);
    // cuBLASLt 旁路(KATAGO_CUDA_CUBLASLT=1,beta=1 残差)。
    if crate::tactic_plan::tactic_var("KATAGO_CUDA_CUBLASLT").as_deref() != Ok("0") && b.kp == b.k {
        if rt.cublaslt_gemm(&stream, a, &b.data, c, m, b.n, b.k, 1.0)? {
            return Ok(());
        }
    }
    let (tile_m, tile_n, kname) = if use_t32 { (32usize, 32usize, "hgemm_t32") } else if use_t64n32 { (64usize, 32usize, "hgemm_t64n32") } else if m < 1024 { (64usize, 64usize, "hgemm_t64") } else { (128usize, 128usize, "hgemm_v2") };
    let grid = (m.div_ceil(tile_m) as u32, b.n.div_ceil(tile_n) as u32, 1u32);
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: (if use_t32 { 128 } else { 256 }, 1, 1),
        shared_mem_bytes: 0,
    };
    let (alpha, beta) = (1.0f32, 1.0f32);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(&b.data)
            .arg(c)
            .arg(&(m as i32))
            .arg(&(b.n as i32))
            .arg(&(b.kp as i32))
            .arg(&alpha)
            .arg(&beta)
            .launch(cfg)
    }
    .map_err(|e| format!("{kname} residual launch failed: {e}"))?;
    Ok(())
}

/// split-K partial（K/2 部分和写 Cp，无 reduce——reduce 由后续
/// RmsNorm 融合完成）。确定性两段式的前半。
fn hgemm_splitk2_partial(
    rt: &CudaRuntime,
    a: &CudaSlice<u16>,
    b: &WeightBuf,
    cp: &mut CudaSlice<f32>,
    m: usize,
) -> Result<(), String> {
    let fp = rt.get_func("hgemm_t64_splitk2_partial_kernel")?;
    let stream = active_stream(rt);
    let grid = (m.div_ceil(64) as u32, b.n.div_ceil(64) as u32, 2u32);
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let alpha = 1.0f32;
    unsafe {
        stream
            .launch_builder(&fp)
            .arg(a)
            .arg(&b.data)
            .arg(cp)
            .arg(&(m as i32))
            .arg(&(b.n as i32))
            .arg(&(b.kp as i32))
            .arg(&alpha)
            .launch(cfg)
    }
    .map_err(|e| format!("splitk partial launch failed: {e}"))?;
    Ok(())
}

/// down 门融合（G3）：`C = silu((A@B^T + beta*C) * scale + bias)`，f32 原位。
/// 融合 Ffn down 残差 GEMM 与块尾 GateSilu(384)，省 1 kernel/块。
#[allow(clippy::too_many_arguments)]
fn hgemm_residual_gatesilu_f32(
    rt: &CudaRuntime,
    a: &CudaSlice<u16>,
    b: &WeightBuf,
    c: &mut CudaSlice<f32>,
    out16: &mut CudaSlice<u16>,
    scale: &CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    m: usize,
) -> Result<(), String> {
    let f = rt.get_func("hgemm_v2_gatesilu_f32_kernel")?;
    let stream = active_stream(rt);
    let grid = (m.div_ceil(128) as u32, b.n.div_ceil(128) as u32, 1u32);
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (alpha, beta) = (1.0f32, 1.0f32);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(&b.data)
            .arg(c)
            .arg(out16)
            .arg(scale)
            .arg(bias)
            .arg(&(m as i32))
            .arg(&(b.n as i32))
            .arg(&(b.kp as i32))
            .arg(&alpha)
            .arg(&beta)
            .launch(cfg)
    }
    .map_err(|e| format!("hgemm gatesilu f32 launch failed: {e}"))?;
    Ok(())
}

/// up 门融合（G3）：C 保留 f32 残差（silu 前，供下一块残差累积），
/// `out = silu((A@B^T + beta*C) * scale + bias)` 写 f16 gated 流。
/// 融合 Linear up 残差 GEMM 与块边界 GateSilu(768)，省 1 kernel/块。
#[allow(clippy::too_many_arguments)]
fn hgemm_residual_gatesilu_f16(
    rt: &CudaRuntime,
    a: &CudaSlice<u16>,
    b: &WeightBuf,
    c: &mut CudaSlice<f32>,
    out: &mut CudaSlice<u16>,
    scale: &CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    m: usize,
) -> Result<(), String> {
    let f = rt.get_func("hgemm_v2_gatesilu_f16_kernel")?;
    let stream = active_stream(rt);
    let grid = (m.div_ceil(128) as u32, b.n.div_ceil(128) as u32, 1u32);
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (alpha, beta) = (1.0f32, 1.0f32);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(&b.data)
            .arg(c)
            .arg(out)
            .arg(scale)
            .arg(bias)
            .arg(&(m as i32))
            .arg(&(b.n as i32))
            .arg(&(b.kp as i32))
            .arg(&alpha)
            .arg(&beta)
            .launch(cfg)
    }
    .map_err(|e| format!("hgemm gatesilu f16 launch failed: {e}"))?;
    Ok(())
}

/// 小头 GEMM：`C[M,N] = A[M,K] @ B[N,K]^T`，A f32、B f16（权重）、C f32。
/// 一线程一输出元素；仅用于池化向量等小张量（A 的 stride 必须等于 `b.k`）。
fn sgemm(
    rt: &CudaRuntime,
    a: &CudaSlice<f32>,
    b: &WeightBuf,
    c: &mut CudaSlice<f32>,
    m: usize,
) -> Result<(), String> {
    let f = rt.get_func("sgemm_f16b_kernel")?;
    let stream = active_stream(rt);
    let n = m * b.n;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(&b.data)
            .arg(c)
            .arg(&(m as i32))
            .arg(&(b.n as i32))
            .arg(&(b.k as i32))
            .launch(LaunchConfig::for_num_elems(n as u32))
    }
    .map_err(|e| format!("sgemm launch failed: {e}"))?;
    Ok(())
}

fn im2col(
    rt: &CudaRuntime,
    spatial: &CudaSlice<f32>,
    cols: &mut CudaSlice<u16>,
    batch: usize,
    s: usize,
) -> Result<(), String> {
    const W: i32 = 19;
    const C: i32 = 22;
    const K: i32 = C * 9;
    const KP: i32 = 208;
    let f = rt.get_func("im2col_f16_kernel")?;
    let stream = active_stream(rt);
    let n = batch * s * K as usize;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(spatial)
            .arg(cols)
            .arg(&(batch as i32))
            .arg(&(s as i32))
            .arg(&W)
            .arg(&C)
            .arg(&K)
            .arg(&KP)
            .launch(LaunchConfig::for_num_elems(n as u32))
    }
    .map_err(|e| format!("im2col launch failed: {e}"))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn conv_bias_gate(
    rt: &CudaRuntime,
    conv: &CudaSlice<f32>,
    global: &CudaSlice<f32>,
    gw: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    gated: &mut CudaSlice<u16>,
    scale: &CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    batch: usize,
    s: usize,
    c: usize,
    g: usize,
) -> Result<(), String> {
    // 融合版：conv+global 加和 → out f32（残差保留）+ gated f16（silu）。
    let f = rt.get_func("conv_bias_gate_silu_kernel")?;
    let stream = active_stream(rt);
    let n = batch * s * c;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(conv)
            .arg(global)
            .arg(gw)
            .arg(out)
            .arg(gated)
            .arg(scale)
            .arg(bias)
            .arg(&(batch as i32))
            .arg(&(s as i32))
            .arg(&(c as i32))
            .arg(&(g as i32))
            .launch(LaunchConfig::for_num_elems(n as u32))
    }
    .map_err(|e| format!("conv_bias_gate launch failed: {e}"))?;
    Ok(())
}

/// 异位门控 SiLU（f32 入 → f16 出）。
fn gate_silu_out(
    rt: &CudaRuntime,
    x: &CudaSlice<f32>,
    scale: &CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    out: &mut CudaSlice<u16>,
    n: usize,
    c: usize,
) -> Result<(), String> {
    let f = rt.get_func("gate_silu_out_f16_kernel")?;
    let stream = active_stream(rt);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(x)
            .arg(scale)
            .arg(bias)
            .arg(out)
            .arg(&(n as i32))
            .arg(&(c as i32))
            .launch(LaunchConfig::for_num_elems(n as u32))
    }
    .map_err(|e| format!("gate_silu_out launch failed: {e}"))?;
    Ok(())
}

/// 原位门控 SiLU（f16 流）。
fn gate_silu(
    rt: &CudaRuntime,
    x: &mut CudaSlice<u16>,
    scale: &CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    n: usize,
    c: usize,
) -> Result<(), String> {
    let f = rt.get_func("gate_silu_f16_kernel")?;
    let stream = active_stream(rt);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(&mut *x)
            .arg(scale)
            .arg(bias)
            .arg(&(n as i32))
            .arg(&(c as i32))
            .launch(LaunchConfig::for_num_elems(n as u32))
    }
    .map_err(|e| format!("gate_silu launch failed: {e}"))?;
    Ok(())
}

fn add_bias_silu_f16(
    rt: &CudaRuntime,
    x: &CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    out: &mut CudaSlice<u16>,
    n: usize,
    c: usize,
) -> Result<(), String> {
    let f = rt.get_func("add_bias_silu_f16_kernel")?;
    let stream = active_stream(rt);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(x)
            .arg(bias)
            .arg(out)
            .arg(&(n as i32))
            .arg(&(c as i32))
            .launch(LaunchConfig::for_num_elems(n as u32))
    }
    .map_err(|e| format!("add_bias_silu_f16 launch failed: {e}"))?;
    Ok(())
}

/// PolicyHead pass 分支融合：pooled@pass1+silu → @pass2 → pass_logits。
fn policy_pass_fused(
    rt: &CudaRuntime,
    pooled: &CudaSlice<f32>,
    pass1: &WeightBuf,
    pass_b1: &CudaSlice<f32>,
    pass2: &WeightBuf,
    out: &mut CudaSlice<f32>,
    batch: usize,
) -> Result<(), String> {
    let f = rt.get_func("policy_pass_fused_kernel")?;
    let stream = active_stream(rt);
    let cfg = LaunchConfig {
        grid_dim: (batch as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&f)
            .arg(pooled)
            .arg(&pass1.data)
            .arg(pass_b1)
            .arg(&pass2.data)
            .arg(out)
            .arg(&(batch as i32))
            .launch(cfg)
    }
    .map_err(|e| format!("policy_pass_fused launch failed: {e}"))?;
    Ok(())
}

/// ValueHead FC 融合：pooled@l2+silu → @vall_w → vall [B,21]。
fn value_fc_fused(
    rt: &CudaRuntime,
    pooled: &CudaSlice<f32>,
    l2: &WeightBuf,
    l2_bias: &CudaSlice<f32>,
    vall_w: &WeightBuf,
    vall_b: &CudaSlice<f32>,
    out0: &mut CudaSlice<f32>,
    out1: &mut CudaSlice<f32>,
    out2: &mut CudaSlice<f32>,
    batch: usize,
) -> Result<(), String> {
    let f = rt.get_func("value_fc_fused_kernel")?;
    let stream = active_stream(rt);
    let cfg = LaunchConfig {
        grid_dim: (batch as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&f)
            .arg(pooled)
            .arg(&l2.data)
            .arg(l2_bias)
            .arg(&vall_w.data)
            .arg(vall_b)
            .arg(out0)
            .arg(out1)
            .arg(out2)
            .arg(&(batch as i32))
            .launch(cfg)
    }
    .map_err(|e| format!("value_fc_fused launch failed: {e}"))?;
    Ok(())
}

/// 带输入 leading-dim/offset 的 add_bias_silu（G4 合并 GEMM 输出子段读取）。
#[allow(clippy::too_many_arguments)]
fn add_bias_silu_f16_ld(
    rt: &CudaRuntime,
    x: &CudaSlice<f32>,
    in_off: usize,
    in_ld: usize,
    bias: &CudaSlice<f32>,
    out: &mut CudaSlice<u16>,
    n: usize,
    c: usize,
) -> Result<(), String> {
    let f = rt.get_func("add_bias_silu_f16_ld_kernel")?;
    let stream = active_stream(rt);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(x)
            .arg(bias)
            .arg(out)
            .arg(&(n as i32))
            .arg(&(c as i32))
            .arg(&(in_ld as i32))
            .arg(&(in_off as i32))
            .launch(LaunchConfig::for_num_elems(n as u32))
    }
    .map_err(|e| format!("add_bias_silu_f16_ld launch failed: {e}"))?;
    Ok(())
}

/// f32 就地 bias + SiLU。
fn bias_silu_f32(
    rt: &CudaRuntime,
    x: &mut CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    n: usize,
    c: usize,
) -> Result<(), String> {
    let f = rt.get_func("bias_silu_f32_kernel")?;
    let stream = active_stream(rt);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(&mut *x)
            .arg(bias)
            .arg(&(n as i32))
            .arg(&(c as i32))
            .launch(LaunchConfig::for_num_elems(n as u32))
    }
    .map_err(|e| format!("bias_silu_f32 launch failed: {e}"))?;
    Ok(())
}

/// f32 就地 + 逐通道 bias（无激活）。
fn f32_bias_add(
    rt: &CudaRuntime,
    x: &mut CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    n: usize,
    c: usize,
) -> Result<(), String> {
    let f = rt.get_func("f32_bias_add_kernel")?;
    let stream = active_stream(rt);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(&mut *x)
            .arg(bias)
            .arg(&(n as i32))
            .arg(&(c as i32))
            .launch(LaunchConfig::for_num_elems(n as u32))
    }
    .map_err(|e| format!("f32_bias_add launch failed: {e}"))?;
    Ok(())
}

/// 合并 bias_add + 拆分写（G4 ValueHead）：in [B,21] + bias[21] →
/// out0 [B,3] / out1 [B,10] / out2 [B,8]。
#[allow(clippy::too_many_arguments)]
fn f32_bias_add_split(
    rt: &CudaRuntime,
    x: &CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    out0: &mut CudaSlice<f32>,
    out1: &mut CudaSlice<f32>,
    out2: &mut CudaSlice<f32>,
    n0: usize,
    n1: usize,
    n2: usize,
    batch: usize,
) -> Result<(), String> {
    let f = rt.get_func("f32_bias_add_split_kernel")?;
    let stream = active_stream(rt);
    let n = batch * (n0 + n1 + n2);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(x)
            .arg(bias)
            .arg(out0)
            .arg(out1)
            .arg(out2)
            .arg(&(n0 as i32))
            .arg(&(n1 as i32))
            .arg(&(n2 as i32))
            .arg(&(batch as i32))
            .launch(LaunchConfig::for_num_elems(n as u32))
    }
    .map_err(|e| format!("f32_bias_add_split launch failed: {e}"))?;
    Ok(())
}


fn f32_to_f16(
    rt: &CudaRuntime,
    x: &CudaSlice<f32>,
    out: &mut CudaSlice<u16>,
    n: usize,
) -> Result<(), String> {
    let f = rt.get_func("f32_to_half_kernel")?;
    let stream = active_stream(rt);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(x)
            .arg(out)
            .arg(&(n as i32))
            .launch(LaunchConfig::for_num_elems(n as u32))
    }
    .map_err(|e| format!("f32_to_half launch failed: {e}"))?;
    Ok(())
}

/// f32 原位累加：x[i] += a[i]（768 上投影残差用：f32 raw 流 + f32 GEMM 输出）。
fn f32_add_inplace(
    rt: &CudaRuntime,
    x: &mut CudaSlice<f32>,
    a: &CudaSlice<f32>,
    n: usize,
) -> Result<(), String> {
    let f = rt.get_func("f32_add_inplace_kernel")?;
    let stream = active_stream(rt);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(&mut *x)
            .arg(a)
            .arg(&(n as i32))
            .launch(LaunchConfig::for_num_elems(n as u32))
    }
    .map_err(|e| format!("f32_add_inplace launch failed: {e}"))?;
    Ok(())
}

/// 残差：out[i] = a[i] + f16(b[i])（b == out 原位）。
fn f32_add_f16(
    rt: &CudaRuntime,
    a: &CudaSlice<f32>,
    b_out: &mut CudaSlice<u16>,
    n: usize,
) -> Result<(), String> {
    let f = rt.get_func("f32_add_f16_kernel")?;
    let stream = active_stream(rt);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(&mut *b_out)
            .arg(&(n as i32))
            .launch(LaunchConfig::for_num_elems(n as u32))
    }
    .map_err(|e| format!("f32_add_f16 launch failed: {e}"))?;
    Ok(())
}

/// split-K 求和 + RMSNorm 融合：x = x*beta + ΣCp（split-K partial 合并），
/// 写回 x（残差流）+ RMSNorm 写 y。省 split-K 的独立 reduce kernel。
#[allow(clippy::too_many_arguments)]
fn rms_norm_splitk(
    rt: &CudaRuntime,
    x: &mut CudaSlice<f32>,
    cp: &CudaSlice<f32>,
    scale: &CudaSlice<f32>,
    out: &mut CudaSlice<u16>,
    beta: f32,
    eps: f32,
    ncols: usize,
    rows: usize,
    splits: usize,
) -> Result<(), String> {
    let f = rt.get_func("rms_norm_splitk_kernel")?;
    let stream = active_stream(rt);
    let cfg = LaunchConfig {
        grid_dim: (rows.div_ceil(4) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&f)
            .arg(x)
            .arg(cp)
            .arg(scale)
            .arg(out)
            .arg(&beta)
            .arg(&eps)
            .arg(&(ncols as i32))
            .arg(&(rows as i32))
            .arg(&(splits as i32))
            .launch(cfg)
    }
    .map_err(|e| format!("rms_norm_splitk launch failed: {e}"))?;
    Ok(())
}

/// RMSNorm（f32 入 → f16 出，供张量核 GEMM）。warp4-vec8：4 行/块、
/// 零 smem 纯 shfl（fork G3 认证路径），与旧单行版同公式。
fn rms_norm_f32(
    rt: &CudaRuntime,
    x: &CudaSlice<f32>,
    out: &mut CudaSlice<u16>,
    scale: &CudaSlice<f32>,
    eps: f32,
    ncols: usize,
    rows: usize,
) -> Result<(), String> {
    // KATAGO_CUDA_RMS=v1 回退旧单行版(诊断 warp4 写坏 act384 的嫌疑)。
    let use_v1 = crate::tactic_plan::tactic_var("KATAGO_CUDA_RMS").as_deref() == Ok("v1");
    let (f, cfg) = if use_v1 {
        (
            rt.get_func("rms_norm_f32_kernel")?,
            LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            },
        )
    } else {
        (
            rt.get_func("rms_norm_f32_w4_kernel")?,
            LaunchConfig {
                grid_dim: (rows.div_ceil(4) as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            },
        )
    };
    let stream = active_stream(rt);
    if use_v1 {
        unsafe {
            stream
                .launch_builder(&f)
                .arg(x)
                .arg(scale)
                .arg(out)
                .arg(&eps)
                .arg(&(ncols as i32))
                .launch(cfg)
        }
        .map_err(|e| format!("rms_norm_f32 launch failed: {e}"))?;
    } else {
        unsafe {
            stream
                .launch_builder(&f)
                .arg(x)
                .arg(scale)
                .arg(out)
                .arg(&eps)
                .arg(&(ncols as i32))
                .arg(&(rows as i32))
                .launch(cfg)
        }
        .map_err(|e| format!("rms_norm_f32 launch failed: {e}"))?;
    }
    Ok(())
}

/// 原位门控 SiLU（f32 流）。
fn gate_silu_f32(
    rt: &CudaRuntime,
    x: &mut CudaSlice<f32>,
    scale: &CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    n: usize,
    c: usize,
) -> Result<(), String> {
    let f = rt.get_func("gate_silu_f32_kernel")?;
    let stream = active_stream(rt);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(&mut *x)
            .arg(scale)
            .arg(bias)
            .arg(&(n as i32))
            .arg(&(c as i32))
            .launch(LaunchConfig::for_num_elems(n as u32))
    }
    .map_err(|e| format!("gate_silu_f32 launch failed: {e}"))?;
    Ok(())
}

/// dual FFN SwiGLU：x [M, 2H]（行内 [gate H | up H]）→ out [M, H] = up * silu(gate)。
fn swiglu_dual(
    rt: &CudaRuntime,
    x: &CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    m: usize,
    hidden: usize,
) -> Result<(), String> {
    let f = rt.get_func("swiglu_dual_kernel")?;
    let stream = active_stream(rt);
    let n = m * hidden;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(x)
            .arg(out)
            .arg(&(m as i32))
            .arg(&(hidden as i32))
            .launch(LaunchConfig::for_num_elems(n as u32))
    }
    .map_err(|e| format!("swiglu_dual launch failed: {e}"))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn qkv_rope(
    rt: &CudaRuntime,
    qkv: &CudaSlice<f32>,
    cos: &CudaSlice<f32>,
    sin: &CudaSlice<f32>,
    q: &mut CudaSlice<u16>,
    k: &mut CudaSlice<u16>,
    v: &mut CudaSlice<u16>,
    batch: usize,
    h: usize,
    s: usize,
    d: usize,
) -> Result<(), String> {
    let f = rt.get_func("qkv_rope_kernel")?;
    let stream = active_stream(rt);
    let cfg = LaunchConfig {
        grid_dim: (s as u32, (batch * h) as u32, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&f)
            .arg(qkv)
            .arg(cos)
            .arg(sin)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(&(batch as i32))
            .arg(&(h as i32))
            .arg(&(s as i32))
            .arg(&(d as i32))
            .launch(cfg)
    }
    .map_err(|e| format!("qkv_rope launch failed: {e}"))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn attention_row(
    rt: &CudaRuntime,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    s: usize,
    d: usize,
    bh: usize,
    scale: f32,
    heads: usize,
) -> Result<(), String> {
    assert!(s <= 512, "attention_row 要求 S <= 512");
    assert_eq!(d, 32, "attention 要求 D=32（smem/打包布局按 D=32 编译期定）");
    // 默认 FA2（tensor core QK + 标量 PV，对拍已验证）；
    // KATAGO_CUDA_ATTN=v3 回退旧 kernel（数值对照兜底）。
    let use_v3 = crate::tactic_plan::tactic_var("KATAGO_CUDA_ATTN").as_deref() == Ok("v3");
    // v3（warp-per-row 标量点积，数值已验证）；
    // 输出直接写 merge 后布局 [B*S, H*D]（省 attn_merge 独立 kernel）。
    let f = if use_v3 {
        rt.get_func("attention_row_v3_kernel")?
    } else {
        rt.get_func("attention_fa2_kernel")?
    };
    let stream = active_stream(rt);
    let cfg = if use_v3 {
        LaunchConfig {
            grid_dim: (((s + 7) / 8) as u32, bh as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    } else {
        LaunchConfig {
            grid_dim: (s.div_ceil(128) as u32, bh as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    };
    unsafe {
        stream
            .launch_builder(&f)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(out)
            .arg(&(s as i32))
            .arg(&(d as i32))
            .arg(&scale)
            .arg(&(heads as i32))
            .launch(cfg)
    }
    .map_err(|e| format!("attention launch failed: {e}"))?;
    Ok(())
}

/// FA2 attention：输入 packed qkv [M,1152] f16 + RoPE 表，内部加载时
/// 旋转 Q/K（省独立 rope kernel 与 qbuf/kbuf/vbuf 缓冲）。
#[allow(clippy::too_many_arguments)]
fn attention_fa2(
    rt: &CudaRuntime,
    qkv: &CudaSlice<u16>,
    rope_cos: &CudaSlice<f32>,
    rope_sin: &CudaSlice<f32>,
    out: &mut CudaSlice<u16>,
    s: usize,
    d: usize,
    bh: usize,
    scale: f32,
    heads: usize,
) -> Result<(), String> {
    assert!(s <= 512, "attention_fa2 要求 S <= 512");
    assert_eq!(d, 32, "attention_fa2 要求 D=32");
    let f = rt.get_func("attention_fa2_kernel")?;
    let stream = active_stream(rt);
    let cfg = LaunchConfig {
        grid_dim: (s.div_ceil(128) as u32, bh as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&f)
            .arg(qkv)
            .arg(rope_cos)
            .arg(rope_sin)
            .arg(out)
            .arg(&(s as i32))
            .arg(&(d as i32))
            .arg(&scale)
            .arg(&(heads as i32))
            .launch(cfg)
    }
    .map_err(|e| format!("attention_fa2 launch failed: {e}"))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn bn_silu(
    rt: &CudaRuntime,
    x: &CudaSlice<f32>,
    mean: &CudaSlice<f32>,
    std: &CudaSlice<f32>,
    gamma: &CudaSlice<f32>,
    beta: &CudaSlice<f32>,
    out: &mut CudaSlice<u16>,
    n: usize,
    c: usize,
) -> Result<(), String> {
    let f = rt.get_func("bn_silu_f16_kernel")?;
    let stream = active_stream(rt);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(x)
            .arg(mean)
            .arg(std)
            .arg(gamma)
            .arg(beta)
            .arg(out)
            .arg(&(n as i32))
            .arg(&(c as i32))
            .launch(LaunchConfig::for_num_elems(n as u32))
    }
    .map_err(|e| format!("bn_silu launch failed: {e}"))?;
    Ok(())
}

fn pool_mean_max(
    rt: &CudaRuntime,
    x: &CudaSlice<u16>,
    out: &mut CudaSlice<f32>,
    batch: usize,
    s: usize,
    c: usize,
    scale: f32,
) -> Result<(), String> {
    let f = rt.get_func("pool_mean_max_kernel")?;
    let stream = active_stream(rt);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(x)
            .arg(out)
            .arg(&(batch as i32))
            .arg(&(s as i32))
            .arg(&(c as i32))
            .arg(&scale)
            .launch(LaunchConfig::for_num_elems((batch * c) as u32))
    }
    .map_err(|e| format!("pool_mean_max launch failed: {e}"))?;
    Ok(())
}

fn pool_mean3(
    rt: &CudaRuntime,
    x: &CudaSlice<u16>,
    out: &mut CudaSlice<f32>,
    batch: usize,
    s: usize,
    c: usize,
    scale: f32,
    quad: f32,
) -> Result<(), String> {
    let f = rt.get_func("pool_mean3_kernel")?;
    let stream = active_stream(rt);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(x)
            .arg(out)
            .arg(&(batch as i32))
            .arg(&(s as i32))
            .arg(&(c as i32))
            .arg(&scale)
            .arg(&quad)
            .launch(LaunchConfig::for_num_elems((batch * c) as u32))
    }
    .map_err(|e| format!("pool_mean3 launch failed: {e}"))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn policy_g(
    rt: &CudaRuntime,
    conv1p: &CudaSlice<f32>,
    gproj: &CudaSlice<f32>,
    bias2: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    batch: usize,
    s: usize,
    c: usize,
) -> Result<(), String> {
    let f = rt.get_func("policy_g_kernel")?;
    let stream = active_stream(rt);
    let n = batch * s * c;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(conv1p)
            .arg(gproj)
            .arg(bias2)
            .arg(out)
            .arg(&(batch as i32))
            .arg(&(s as i32))
            .arg(&(c as i32))
            .launch(LaunchConfig::for_num_elems(n as u32))
    }
    .map_err(|e| format!("policy_g launch failed: {e}"))?;
    Ok(())
}

/// 带 conv1p leading-dim/offset 的 policy_g（G4 合并 GEMM 输出子段读取）。
#[allow(clippy::too_many_arguments)]
fn policy_g_ld(
    rt: &CudaRuntime,
    conv1p: &CudaSlice<f32>,
    c1_off: usize,
    c1_ld: usize,
    gproj: &CudaSlice<f32>,
    bias2: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    batch: usize,
    s: usize,
    c: usize,
) -> Result<(), String> {
    let f = rt.get_func("policy_g_ld_kernel")?;
    let stream = active_stream(rt);
    let n = batch * s * c;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(conv1p)
            .arg(gproj)
            .arg(bias2)
            .arg(out)
            .arg(&(batch as i32))
            .arg(&(s as i32))
            .arg(&(c as i32))
            .arg(&(c1_ld as i32))
            .arg(&(c1_off as i32))
            .launch(LaunchConfig::for_num_elems(n as u32))
    }
    .map_err(|e| format!("policy_g_ld launch failed: {e}"))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn policy_concat(
    rt: &CudaRuntime,
    moves: &CudaSlice<f32>,
    pass: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    batch: usize,
    s: usize,
    c: usize,
    penalty: f32,
) -> Result<(), String> {
    let f = rt.get_func("policy_concat_kernel")?;
    let stream = active_stream(rt);
    let n = batch * c * (s + 1);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(moves)
            .arg(pass)
            .arg(out)
            .arg(&(batch as i32))
            .arg(&(s as i32))
            .arg(&(c as i32))
            .arg(&penalty)
            .launch(LaunchConfig::for_num_elems(n as u32))
    }
    .map_err(|e| format!("policy_concat launch failed: {e}"))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// 工作区与输出
// ---------------------------------------------------------------------------

/// 前向推理工作区：全部中间/输出缓冲一次性预分配，多次前向复用。
///
/// 消除每次 apply 的 20+ 次设备内存分配（WDDM 下 malloc 虽为
/// stream-ordered，但配合 CUDA Graph capture 需预分配固定地址）。
pub struct CudaWorkspace {
    pub cols: CudaSlice<u16>,   // im2col [M,208]
    pub act768: CudaSlice<f32>, // raw 768 流（f32 残差累积）
    pub gated768: CudaSlice<u16>,
    pub act384: CudaSlice<f32>,
    pub act384f16: CudaSlice<u16>,
    pub normed: CudaSlice<u16>,
    pub act1152: CudaSlice<u16>,
    /// dual FFN 的 gate|up 拼接输出 [M, 2H]（行内 [gate H | up H]）。
    pub act2304: CudaSlice<u16>,
    pub gemm_out: CudaSlice<f32>,
    pub qbuf: CudaSlice<u16>,
    pub kbuf: CudaSlice<u16>,
    pub vbuf: CudaSlice<u16>,
    pub attn: CudaSlice<u16>,
    pub p96f: CudaSlice<f32>,
    pub p192: CudaSlice<u16>,
    pub conv1p: CudaSlice<f32>,
    pub pooled: CudaSlice<f32>,
    pub vec_a: CudaSlice<f32>,
    pub vec_b: CudaSlice<f32>,
    /// ValueHead 合并输出 [B,21]（value|misc|moremisc，G4 头融合）。
    pub vall: CudaSlice<f32>,
    /// PolicyHead 合并的 conv1p|conv1g 输出 [M,192] f32。
    pub conv1pg: CudaSlice<f32>,
    /// split-K partial 缓冲 [2, M, 384] f32（N=384 的 starved GEMM 拆 K=2）。
    pub c_partial: CudaSlice<f32>,
    pub pass_logits: CudaSlice<f32>,
    /// `[B, 6, 362]` 策略 logits（通道 0 基策略、通道 5 乐观策略，pass 位 361）。
    pub out_policy: CudaSlice<f32>,
    /// `[B, 3]` 价值 logits。
    pub out_value: CudaSlice<f32>,
    /// `[B, 10]`（scoreMean/scoreMeanSq/lead/varTimeLeft/…）。
    pub out_misc: CudaSlice<f32>,
    /// `[B, 8]`（shortterm winloss/score error/…）。
    pub out_moremisc: CudaSlice<f32>,
    /// `[B, 361]` 所有权。
    pub out_ownership: CudaSlice<f32>,
    batch: usize,
}

impl CudaWorkspace {
    /// 按模型形状与 batch 一次性预分配全部工作区。
    pub fn new(stream: &Arc<CudaStream>, model: &CudaModel, batch: usize) -> Result<Self, String> {
        let s = model.seq_len;
        let m = batch * s;
        let trunk = model.trunk;
        let mid = model.mid;
        let h = model.num_heads;
        let d = model.head_dim;
        let qkv_elts = batch * h * s * d;
        Ok(Self {
            cols: zeros16(stream, m * 208)?,
            act768: zeros32(stream, m * trunk)?,
            gated768: zeros16(stream, m * trunk)?,
            act384: zeros32(stream, m * mid)?,
            act384f16: zeros16(stream, m * mid)?,
            normed: zeros16(stream, m * mid)?,
            act1152: zeros16(stream, m * (3 * mid))?,
            act2304: zeros16(stream, m * (6 * mid))?,
            gemm_out: zeros32(stream, m * (3 * mid))?,
            qbuf: zeros16(stream, qkv_elts)?,
            kbuf: zeros16(stream, qkv_elts)?,
            vbuf: zeros16(stream, qkv_elts)?,
            attn: zeros16(stream, qkv_elts)?,
            p96f: zeros32(stream, m * 96)?,
            p192: zeros16(stream, m * 192)?,
            conv1p: zeros32(stream, m * 96)?,
            pooled: zeros32(stream, batch * 576)?,
            vec_a: zeros32(stream, batch * 192)?,
            vec_b: zeros32(stream, batch * 96)?,
            vall: zeros32(stream, batch * 21)?,
            conv1pg: zeros32(stream, m * 192)?,
            c_partial: zeros32(stream, 2 * m * 384)?,
            pass_logits: zeros32(stream, batch * 6)?,
            out_policy: zeros32(stream, batch * 6 * (s + 1))?,
            out_value: zeros32(stream, batch * 3)?,
            out_misc: zeros32(stream, batch * 10)?,
            out_moremisc: zeros32(stream, batch * 8)?,
            out_ownership: zeros32(stream, m)?,
            batch,
        })
    }

    pub fn batch(&self) -> usize {
        self.batch
    }
}

/// 主机侧输出（对拍/测试用）。
#[derive(Debug)]
pub struct CudaOutputsHost {
    pub policy: Vec<f32>,
    pub value: Vec<f32>,
    pub misc: Vec<f32>,
    pub moremisc: Vec<f32>,
    pub ownership: Vec<f32>,
}

impl CudaWorkspace {
    pub fn to_host(&self, stream: &Arc<CudaStream>) -> Result<CudaOutputsHost, String> {
        let stream = stream.clone();
        let get = |d: &CudaSlice<f32>| -> Result<Vec<f32>, String> {
            let mut v = vec![0.0f32; d.len()];
            stream
                .memcpy_dtoh(d, &mut v)
                .map_err(|e| format!("memcpy_dtoh failed: {e}"))?;
            Ok(v)
        };
        Ok(CudaOutputsHost {
            policy: get(&self.out_policy)?,
            value: get(&self.out_value)?,
            misc: get(&self.out_misc)?,
            moremisc: get(&self.out_moremisc)?,
            ownership: get(&self.out_ownership)?,
        })
    }
}
