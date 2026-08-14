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
        up: WeightBuf,
        gate: WeightBuf,
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
        conv1p: WeightBuf,
        conv1g: WeightBuf,
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
        value_w: WeightBuf,
        value_b: ParamBuf,
        misc_w: WeightBuf,
        misc_b: ParamBuf,
        moremisc_w: WeightBuf,
        moremisc_b: ParamBuf,
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
        let act1152b = &mut ws.act1152b;
        let gemm_out = &mut ws.gemm_out;
        let qbuf = &mut ws.qbuf;
        let kbuf = &mut ws.kbuf;
        let vbuf = &mut ws.vbuf;
        let attn = &mut ws.attn;
        let p96f = &mut ws.p96f;
        let p192 = &mut ws.p192;
        let conv1p = &mut ws.conv1p;
        let pooled = &mut ws.pooled;
        let vec_a = &mut ws.vec_a;
        let vec_b = &mut ws.vec_b;
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
        for (li, lb) in self.layers.iter().enumerate() {
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
            match lb {
                LayerBuf::InitialConv { w, gw, g, gate_scale, gate_bias } => {
                    // raw 768 流（块残差加的是 gate 前的值）→ 异位门控 SiLU
                    hgemm(rt, cols, w, gemm_out, m)?;
                    conv_bias_gate(rt, gemm_out, global, gw, act768, batch, s, trunk, *g)?;
                    gate_silu_out(
                        rt,
                        act768,
                        &gate_scale.data,
                        &gate_bias.data,
                        gated768,
                        m * trunk,
                        trunk,
                    )?;
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
                        hgemm(rt, gated768, w, act384, m)?;
                    } else if w.k == mid && w.n == trunk {
                        f32_to_f16(rt, act384, act384f16, m * mid)?;
                        // 上投影 GEMM beta=1 直接残差进 act768（省独立残差 kernel）
                        hgemm_residual(rt, act384f16, w, act768, m)?;
                    } else {
                        return Err(format!("Linear 形状不支持: k={} n={}", w.k, w.n));
                    }
                }
                LayerBuf::RmsNorm { scale, eps, channels } => {
                    assert_eq!(*channels, mid, "RMSNorm 仅出现在 384 维流");
                    rms_norm_f32(rt, act384, normed, &scale.data, *eps, *channels, m)?;
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
                    timed!("qkv", hgemm(rt, normed, qkv, gemm_out, m));
                    timed!(
                        "rope",
                        qkv_rope(
                            rt, gemm_out, cos, sin, qbuf, kbuf, vbuf, batch, lh, ls,
                            ld,
                        )
                    );
                    let scale = qk_scale * qk_scale; // 导出图 q、k 各乘 1/∜d
                    timed!("attn", attention_row(rt, qbuf, kbuf, vbuf, attn, ls, ld, batch * lh, scale));
                    timed!("merge", attn_merge(rt, attn, normed, batch, lh, ls, ld));
                    // 输出投影 beta=1 直接残差进 act384（省独立残差 kernel）
                    timed!("outproj", hgemm_residual(rt, normed, out, act384, m));
                    if sub_profile {
                        let total: f32 = sub_times.iter().map(|t| t.1.max(0.0)).sum();
                        eprintln!("[cuda-profile] attention l{li} total {total:.3} ms: {}",
                            sub_times.iter().map(|(n, t)| format!("{n}={t:.3}")).collect::<Vec<_>>().join(" "));
                    }
                }
                LayerBuf::Ffn { up, gate, down, hidden } => {
                    let hidden = *hidden;
                    assert_eq!(hidden, 3 * mid, "FFN 隐层维度不符");
                    // gate/up 支路：GEMM epilogue 直接 f16 输出（省 2 次转换 launch）
                    hgemm_f16(rt, normed, gate, act1152, m)?;
                    hgemm_f16(rt, normed, up, act1152b, m)?;
                    // silu(gate) * up → 下投影 beta=1 直接残差进 act384
                    swiglu(rt, act1152b, act1152, m * hidden)?;
                    hgemm_residual(rt, act1152, down, act384, m)?;
                }
                LayerBuf::GateSilu { scale, bias, channels } => {
                    // 384 维（块尾）与 768 维（块边界）两种门
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
                }
                LayerBuf::TrunkFinal { mean, std, gamma, beta, channels } => {
                    assert_eq!(*channels, trunk, "TrunkFinal 仅支持 768 维");
                    bn_silu(
                        rt, act768, &mean.data, &std.data, &gamma.data, &beta.data,
                        gated768, m * trunk, *channels,
                    )?;
                }
                LayerBuf::PolicyHead {
                    conv1p: c1p,
                    conv1g: c1g,
                    g_bias,
                    g_matmul,
                    pass1,
                    pass_b1,
                    pass2,
                    bias2,
                    conv2p,
                    mask_scale,
                } => {
                    // conv1p / conv1g（1x1，A=gated768 f16 激活流，K=768）
                    hgemm(rt, gated768, c1p, conv1p, m)?;
                    hgemm(rt, gated768, c1g, gemm_out, m)?;
                    add_bias_silu_f16(rt, gemm_out, &g_bias.data, p192, m * 96, 96)?;
                    // 池化：[mean, mean*scale, max]（f32；小头链路保持 f32 精度）
                    pool_mean_max(rt, p192, pooled, batch, s, 96, *mask_scale)?;
                    // pass 分支：Gemm+bias → SiLU → Gemm（f32×f16 混合小 GEMM）
                    sgemm(rt, pooled, pass1, vec_a, batch)?;
                    bias_silu_f32(rt, vec_a, &pass_b1.data, batch * 96, 96)?;
                    sgemm(rt, vec_a, pass2, pass_logits, batch)?;
                    // g 分支：Gemm → +conv1p → +bias2 → SiLU → conv2p
                    sgemm(rt, pooled, g_matmul, vec_b, batch)?;
                    policy_g(rt, conv1p, vec_b, &bias2.data, p96f, batch, s, 96)?;
                    sgemm(rt, p96f, conv2p, gemm_out, m)?;
                    // 拼接 [B,6,362]；penalty=0（19 路无非法落点，结构保留）
                    policy_concat(rt, gemm_out, pass_logits, out_policy, batch, s, 6, 0.0)?;
                }
                LayerBuf::ValueHead {
                    conv1,
                    bias1,
                    l2,
                    l2_bias,
                    value_w,
                    value_b,
                    misc_w,
                    misc_b,
                    moremisc_w,
                    moremisc_b,
                    own_w,
                    mask_scale,
                    mask_quad,
                } => {
                    // conv1 + bias + SiLU → v_act [M,192]
                    hgemm(rt, gated768, conv1, gemm_out, m)?;
                    add_bias_silu_f16(rt, gemm_out, &bias1.data, p192, m * 192, 192)?;
                    // 池化：[mean, mean*scale, mean*quad]（f32）
                    pool_mean3(rt, p192, pooled, batch, s, 192, *mask_scale, *mask_quad)?;
                    // linear2 + bias + SiLU（f32×f16 混合小 GEMM）
                    sgemm(rt, pooled, l2, vec_a, batch)?;
                    bias_silu_f32(rt, vec_a, &l2_bias.data, batch * 192, 192)?;
                    // 三个输出 Gemm（+bias）
                    sgemm(rt, vec_a, value_w, out_value, batch)?;
                    f32_bias_add(rt, out_value, &value_b.data, batch * 3, 3)?;
                    sgemm(rt, vec_a, misc_w, out_misc, batch)?;
                    f32_bias_add(rt, out_misc, &misc_b.data, batch * 10, 10)?;
                    sgemm(rt, vec_a, moremisc_w, out_moremisc, batch)?;
                    f32_bias_add(rt, out_moremisc, &moremisc_b.data, batch * 8, 8)?;
                    // ownership 1x1（×mask 恒 1，省略；v_act 为 f16 激活流）
                    hgemm(rt, p192, own_w, out_ownership, m)?;
                }
            }
            if profiling {
                let _ = ev_end.as_ref().unwrap().record(&stream);
                let ms = ev_start
                    .as_ref()
                    .unwrap()
                    .elapsed_ms(ev_end.as_ref().unwrap())
                    .unwrap_or(-1.0);
                layer_times.push((li as i64, layer_tag, ms));
            }
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
                        ("act1152b", &*act1152b),
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
            up: upload_weight(stream, &l.up_weight, l.hidden, l.up_weight.numel() / l.hidden)?,
            gate: upload_weight(
                stream,
                &l.gate_weight,
                l.hidden,
                l.gate_weight.numel() / l.hidden,
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
            conv1p: upload_weight(stream, &l.conv1p_weight, 96, 768)?,
            conv1g: upload_weight(stream, &l.conv1g_weight, 96, 768)?,
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
            value_w: upload_weight(stream, &l.value_matmul, 3, 192)?,
            value_b: upload_param(stream, &l.value_bias, 3)?,
            misc_w: upload_weight(stream, &l.misc_matmul, 10, 192)?,
            misc_b: upload_param(stream, &l.misc_bias, 10)?,
            moremisc_w: upload_weight(stream, &l.moremisc_matmul, 8, 192)?,
            moremisc_b: upload_param(stream, &l.moremisc_bias, 8)?,
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
/// v2：tile 128×128×32、smem 双缓冲 + cp.async 流水（gemm_v2.cu）。
fn hgemm(
    rt: &CudaRuntime,
    a: &CudaSlice<u16>,
    b: &WeightBuf,
    c: &mut CudaSlice<f32>,
    m: usize,
) -> Result<(), String> {
    let f = rt.get_func("hgemm_v2_kernel")?;
    let stream = active_stream(rt);
    let grid = (m.div_ceil(128) as u32, b.n.div_ceil(128) as u32, 1u32);
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: (256, 1, 1),
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
    .map_err(|e| format!("hgemm launch failed: {e}"))?;
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
    let f = rt.get_func("hgemm_v2_f16out_kernel")?;
    let stream = active_stream(rt);
    let grid = (m.div_ceil(128) as u32, b.n.div_ceil(128) as u32, 1u32);
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: (256, 1, 1),
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
    .map_err(|e| format!("hgemm_f16 launch failed: {e}"))?;
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
    let f = rt.get_func("hgemm_v2_kernel")?;
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
            .arg(&(m as i32))
            .arg(&(b.n as i32))
            .arg(&(b.kp as i32))
            .arg(&alpha)
            .arg(&beta)
            .launch(cfg)
    }
    .map_err(|e| format!("hgemm_residual launch failed: {e}"))?;
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
    batch: usize,
    s: usize,
    c: usize,
    g: usize,
) -> Result<(), String> {
    let f = rt.get_func("conv_bias_gate_kernel")?;
    let stream = active_stream(rt);
    let n = batch * s * c;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(conv)
            .arg(global)
            .arg(gw)
            .arg(out)
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

/// RMSNorm（f32 入 → f16 出，供张量核 GEMM）。
fn rms_norm_f32(
    rt: &CudaRuntime,
    x: &CudaSlice<f32>,
    out: &mut CudaSlice<u16>,
    scale: &CudaSlice<f32>,
    eps: f32,
    ncols: usize,
    rows: usize,
) -> Result<(), String> {
    let f = rt.get_func("rms_norm_f32_kernel")?;
    let stream = active_stream(rt);
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
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

fn swiglu(
    rt: &CudaRuntime,
    up: &CudaSlice<u16>,
    gate_out: &mut CudaSlice<u16>,
    n: usize,
) -> Result<(), String> {
    let f = rt.get_func("swiglu_f16_kernel")?;
    let stream = active_stream(rt);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(up)
            .arg(&mut *gate_out)
            .arg(&(n as i32))
            .launch(LaunchConfig::for_num_elems(n as u32))
    }
    .map_err(|e| format!("swiglu launch failed: {e}"))?;
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
) -> Result<(), String> {
    assert!(s <= 512, "attention_row 要求 S <= 512");
    assert_eq!(d, 32, "attention_row v3 要求 D=32（smem/打包布局按 D=32 编译期定）");
    // v3（warp-per-row + K/V 64 键分块 cp.async 双缓冲 + 128B 打包 smem）：
    // v1 每行一块、512 线程两遍 18 次 syncthreads 树归约且 p·V 仅 32 线程串行；
    // v2 整段 K/V 46KB smem 只 4 块/SM、串行依赖链长（ABBA 40 < 47 被证伪）。
    // v1/v2 保留在 attention.cu 作参考。数值：FP32 全程 + RNE 舍回 half。
    let f = rt.get_func("attention_row_v3_kernel")?;
    let stream = active_stream(rt);
    let cfg = LaunchConfig {
        grid_dim: (((s + 7) / 8) as u32, bh as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
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
            .launch(cfg)
    }
    .map_err(|e| format!("attention_row launch failed: {e}"))?;
    Ok(())
}

fn attn_merge(
    rt: &CudaRuntime,
    attn: &CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    batch: usize,
    h: usize,
    s: usize,
    d: usize,
) -> Result<(), String> {
    let f = rt.get_func("attn_merge_kernel")?;
    let stream = active_stream(rt);
    let n = batch * s * h * d;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(attn)
            .arg(out)
            .arg(&(batch as i32))
            .arg(&(h as i32))
            .arg(&(s as i32))
            .arg(&(d as i32))
            .launch(LaunchConfig::for_num_elems(n as u32))
    }
    .map_err(|e| format!("attn_merge launch failed: {e}"))?;
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
    pub act1152b: CudaSlice<u16>,
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
            act1152b: zeros16(stream, m * (3 * mid))?,
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
