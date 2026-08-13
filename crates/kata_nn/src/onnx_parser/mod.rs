//! KataGo ONNX 模型 → 手写 CUDA 后端可执行的"层图"解析器。
//!
//! 流水线分三步（对应三个子模块）：
//!
//! 1. [`load_onnx`] 用 prost 解码模型，产出 [`OnnxGraph`]（initializer 常量表、
//!    拓扑序节点、输入/输出名）。
//! 2. [`fold::fold_graph`] 做常量折叠 + 符号形状传播，产出 [`SimplifiedGraph`]
//!    （布局类算子与 RoPE 的 Sin/Cos 已在加载期求值，剩下来的只有数值节点和
//!    少量动态布局节点）。
//! 3. [`match::match_layers`] 在简化图上做模式匹配，产出 [`LayerGraph`]。
//!
//! 目标模型为 KataGo 导出的 19 路 b11 系列 ONNX（本目录只支持该结构）：
//! trunk C768 / mid C384 / FFN 隐层 1152 / 12 头 × 32 维 / 序列长 361，
//! 11 个块 × 每块 6 个子层（3 attention + 3 FFN），每子层前一个 RMSNorm。
//!
//! 所有权重以 f32 保存、行主序、**以 out 为第一维**（即 ONNX 里 `[k, n]` 的
//! MatMul 权重存成 `[n, k]`）；attention 的 Q/K/V 三个投影矩阵在解析期拼成
//! 宽矩阵 `[3*h*d, k]`，RoPE 表拼成 cos/sin 两个 `[361, h*d/2]` 张量。
//!
//! 注意：模型里唯一的 `Where(Equal(输入 plane 0, 0), -inf, 0)` 掩码链来自输入
//! 的 on-board 平面（本引擎对 19 路恒填 1，掩码恒为零）。解析器识别该结构并
//! 校验其语义；若发现真实（非恒零）掩码则报错——本项目不支持带掩码模型。

mod fold;
mod r#match;

pub use fold::{SimplifiedGraph, SimpleNode, SymDim, Tensor, TensorData, fold_graph};
pub use r#match::match_layers;

use std::collections::HashMap;

use prost::Message;

use crate::onnx_proto::{GraphProto, ModelProto};

// ---------------------------------------------------------------------------
// 原始图
// ---------------------------------------------------------------------------

/// 原始 ONNX 图的轻量视图（prost 解码后）。
#[derive(Debug, Clone)]
pub struct RawNode {
    pub op: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub orig_index: usize,
}

/// prost 解码后的执行图：initializer 常量表 + 拓扑序节点 + I/O 名。
pub struct OnnxGraph {
    /// 原始 GraphProto（含所有权重字节）。
    pub graph: GraphProto,
    /// 所有 initializer（name → 张量）。
    pub initializers: HashMap<String, Tensor>,
    /// 拓扑序节点列表。
    pub nodes: Vec<RawNode>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    /// 全部 FLOAT initializer 的元素总数（≈ 参数量）。
    pub total_param_elts: usize,
}

/// 解码 ONNX 模型字节。
pub fn load_onnx(bytes: &[u8]) -> Result<OnnxGraph, String> {
    let model = ModelProto::decode(bytes).map_err(|e| format!("ONNX protobuf 解码失败: {e}"))?;
    let graph = model
        .graph
        .ok_or_else(|| "ONNX 模型没有 graph".to_string())?;
    load_onnx_graph(graph)
}

/// 从已解码的 GraphProto 构建 [`OnnxGraph`]。
pub fn load_onnx_graph(graph: GraphProto) -> Result<OnnxGraph, String> {
    let mut initializers = HashMap::new();
    let mut total_param_elts = 0usize;
    for t in &graph.initializer {
        let name = t.name.clone().unwrap_or_default();
        if name.is_empty() {
            continue;
        }
        let tensor = fold::decode_tensor_proto(t)?;
        if tensor.is_f32() {
            total_param_elts += tensor.numel();
        }
        initializers.insert(name, tensor);
    }
    let nodes: Vec<RawNode> = graph
        .node
        .iter()
        .enumerate()
        .map(|(i, n)| RawNode {
            op: n.op_type.clone().unwrap_or_default(),
            inputs: n.input.clone(),
            outputs: n.output.clone(),
            orig_index: i,
        })
        .collect();
    let inputs: Vec<String> = graph
        .input
        .iter()
        .filter_map(|vi| vi.name.clone())
        .collect();
    let outputs: Vec<String> = graph
        .output
        .iter()
        .filter_map(|vi| vi.name.clone())
        .collect();
    Ok(OnnxGraph {
        graph,
        initializers,
        nodes,
        inputs,
        outputs,
        total_param_elts,
    })
}

// ---------------------------------------------------------------------------
// 层图
// ---------------------------------------------------------------------------

/// 激活函数标记。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act {
    /// `x * sigmoid(x)`
    Silu,
}

/// 初始卷积 + 全局偏置 + 门控 SiLU（trunk 入口）。
#[derive(Debug, Clone)]
pub struct InitialConvLayer {
    /// 3x3 卷积权重 `[768, 22, 3, 3]`（NCHW，out-first）。
    pub weight: Tensor,
    /// 全局输入线性变换权重 `[768, 19]`（`input_global @ Wᵀ` 广播加到卷积输出上）。
    pub global_weight: Tensor,
    /// 门控 SiLU 的逐通道缩放 `[768]`（`silu(x*scale+bias)` 里的 scale）。
    pub gate_scale: Tensor,
    /// 门控 SiLU 的逐通道偏置 `[768]`。
    pub gate_bias: Tensor,
    pub out_channels: usize,
}

/// RMSNorm：`x / sqrt(mean(x²) + eps) * scale`，沿最后一维。
#[derive(Debug, Clone)]
pub struct RmsNormLayer {
    pub scale: Tensor,
    pub channels: usize,
    pub eps: f32,
}

/// 线性层（1x1 卷积在序列化后等价于 MatMul）。`weight` 为 out-first `[n, k]`。
#[derive(Debug, Clone)]
pub struct MatMulLayer {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    /// 输出/输入宽（与 weight 形状一致）。
    pub n: usize,
    pub k: usize,
    pub act: Option<Act>,
    /// 输出是否加回残差输入。
    pub residual_add: bool,
}

/// 多头自注意力。
///
/// 计算语义（与导出 ONNX 逐位一致）：
/// `qkv = x @ qkv_weightᵀ`（`[B,361,3*h*d]`，列拼 Q|K|V）→ 拆 3 段 →
/// 每段 `[B,361,12,32]` → 末维拆 (16,2) 两半做 RoPE（a=偶下标半，b=奇下标半）：
/// `rot(a,b) = (a*cos - b*sin, a*sin + b*cos)` → 转置成 `[B,12,361,32]` →
/// `logits = (q*qk_scale) @ (k*qk_scale)ᵀ`（导出把 1/√d 拆成 q、k 各乘
/// `qk_scale = 1/∜d`）→ `+ 0`（无掩码）→ Softmax(axis=-1) → `@ v` →
/// 拼回 `[B,361,384]` → 输出投影 → `+ x`。
#[derive(Debug, Clone)]
pub struct AttentionLayer {
    pub num_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    /// Q|K|V 宽投影，out-first `[3*h*d, k]`。
    pub qkv_weight: Tensor,
    /// 输出投影，out-first `[h*d, k]`。
    pub out_weight: Tensor,
    /// RoPE cos 表 `[seq_len, h*d/2]`（`[361, 192]`）。
    pub rope_cos: Tensor,
    /// RoPE sin 表 `[seq_len, h*d/2]`。
    pub rope_sin: Tensor,
    /// q、k 各自的缩放系数（`1/∜(head_dim)`），乘积即 1/√d。
    pub qk_scale: f32,
    pub residual_add: bool,
}

/// SwiGLU FFN：`down(silu(gate(x)) * up(x)) + x`。
#[derive(Debug, Clone)]
pub struct FfnLayer {
    /// out-first `[hidden, k]`。
    pub up_weight: Tensor,
    /// out-first `[hidden, k]`。
    pub gate_weight: Tensor,
    /// out-first `[k, hidden]`。
    pub down_weight: Tensor,
    pub hidden: usize,
    pub residual_add: bool,
}

/// 逐通道门控 SiLU：`x * sigmoid(x*scale + bias)`（块边界处的 "normactconv" 门）。
#[derive(Debug, Clone)]
pub struct GateSiluLayer {
    pub scale: Tensor,
    pub bias: Tensor,
    pub channels: usize,
}

/// trunk 末端的 BN（eval 模式仿射）+ 普通 SiLU：`silu((x-mean)/std*gamma + beta)`。
#[derive(Debug, Clone)]
pub struct TrunkFinalLayer {
    pub mean: Tensor,
    pub std: Tensor,
    pub gamma: Tensor,
    pub beta: Tensor,
    pub channels: usize,
}

/// 策略头。
///
/// 结构（对应导出图）：`conv1p`（96 通道）与 `conv1g`（96 通道，+biasg+SiLU）两路
/// 1x1 卷积 → g 路做池化（mean、mean*mask_scale、通道 max 拼成 `[B,288]`）→
/// `pass` 分支（Gemm+SiLU+Gemm → 6 个 pass logits）与 `g` 分支（Gemm → 加回
/// conv1p → +bias2 → SiLU → conv2p → 6×361 落点 logits）→ 拼接成 `[B,6,362]`。
#[derive(Debug, Clone)]
pub struct PolicyHeadLayer {
    /// `[96, 768]`（1x1 卷积，out-first）。
    pub conv1p_weight: Tensor,
    /// `[96, 768]`。
    pub conv1g_weight: Tensor,
    /// `[96]`。
    pub g_bias: Tensor,
    /// 池化后全局分支投影 `[96, 288]`（无 bias）。
    pub g_matmul: Tensor,
    /// pass 分支第一层 `[96, 288]`。
    pub pass_matmul1: Tensor,
    /// pass 分支 bias `[96]`。
    pub pass_bias1: Tensor,
    /// pass 分支第二层 `[6, 96]`（无 bias）。
    pub pass_matmul2: Tensor,
    /// g 分支逐通道 bias `[96]`。
    pub bias2: Tensor,
    /// 最终 1x1 卷积 `[6, 96]`。
    pub conv2p_weight: Tensor,
    /// 激活一律为 SiLU（导出图如此）。
    pub act_silu: bool,
    /// 池化里 mean 通道的缩放系数（19 路 = 0.5，即 `w*0.1-1.4`）。
    pub mask_scale: f32,
}

/// 价值/所有权头。
///
/// 结构：`conv1`（192 通道，+bias1+SiLU）→ 池化（mean、mean*mask_scale、
/// mean*mask_quad 拼成 `[B,576]`）→ `linear2`+SiLU → 三个 Gemm 分别产出
/// value/misc/moremisc；ownership 是 v 激活上的 1x1 卷积。
#[derive(Debug, Clone)]
pub struct ValueHeadLayer {
    /// `[192, 768]`。
    pub conv1_weight: Tensor,
    /// `[192]`。
    pub bias1: Tensor,
    /// `[192, 576]`。
    pub linear2_weight: Tensor,
    /// `[192]`。
    pub linear2_bias: Tensor,
    /// `[3, 192]`。
    pub value_matmul: Tensor,
    /// `[3]`。
    pub value_bias: Tensor,
    /// `[10, 192]`。
    pub misc_matmul: Tensor,
    /// `[10]`。
    pub misc_bias: Tensor,
    /// `[8, 192]`。
    pub moremisc_matmul: Tensor,
    /// `[8]`。
    pub moremisc_bias: Tensor,
    /// `[1, 192]`。
    pub ownership_conv: Tensor,
    pub act_silu: bool,
    /// 池化里 mean 通道的缩放系数（0.5）。
    pub mask_scale: f32,
    /// 池化里 mean 通道的二次系数（0.15）。
    pub mask_quad: f32,
}

/// 层图里的一个层。
#[derive(Debug, Clone)]
pub enum Layer {
    InitialConv(InitialConvLayer),
    /// 1x1 卷积下/上投影（MatMul 形式）。
    Linear(MatMulLayer),
    RmsNorm(RmsNormLayer),
    Attention(AttentionLayer),
    Ffn(FfnLayer),
    GateSilu(GateSiluLayer),
    TrunkFinal(TrunkFinalLayer),
    PolicyHead(PolicyHeadLayer),
    ValueHead(ValueHeadLayer),
}

/// 解析产物：可供 CUDA 后端逐层执行的层图。
pub struct LayerGraph {
    /// 按执行顺序排列的层。
    pub layers: Vec<Layer>,
    pub num_spatial_inputs: usize,
    pub num_global_inputs: usize,
    pub board_size: usize,
    pub trunk_channels: usize,
    pub mid_channels: usize,
    pub num_blocks: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    /// 所有权重元素总数（f32 个数）。
    pub total_params: usize,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
}

impl LayerGraph {
    pub fn num_attention_layers(&self) -> usize {
        self.layers.iter().filter(|l| matches!(l, Layer::Attention(_))).count()
    }

    pub fn num_rmsnorm_layers(&self) -> usize {
        self.layers.iter().filter(|l| matches!(l, Layer::RmsNorm(_))).count()
    }

    pub fn num_ffn_layers(&self) -> usize {
        self.layers.iter().filter(|l| matches!(l, Layer::Ffn(_))).count()
    }

    /// 统计层图内全部权重/常量张量的元素数（应等于 initializer 总元素数）。
    pub fn layer_param_elts(&self) -> usize {
        fn t(t: &Tensor) -> usize {
            t.numel()
        }
        self.layers
            .iter()
            .map(|l| match l {
                Layer::InitialConv(x) => {
                    t(&x.weight) + t(&x.global_weight) + t(&x.gate_scale) + t(&x.gate_bias)
                }
                Layer::Linear(x) => {
                    t(&x.weight) + x.bias.as_ref().map(|b| t(b)).unwrap_or(0)
                }
                Layer::RmsNorm(x) => t(&x.scale),
                Layer::Attention(x) => {
                    t(&x.qkv_weight)
                        + t(&x.out_weight)
                        + t(&x.rope_cos)
                        + t(&x.rope_sin)
                }
                Layer::Ffn(x) => t(&x.up_weight) + t(&x.gate_weight) + t(&x.down_weight),
                Layer::GateSilu(x) => t(&x.scale) + t(&x.bias),
                Layer::TrunkFinal(x) => t(&x.mean) + t(&x.std) + t(&x.gamma) + t(&x.beta),
                Layer::PolicyHead(x) => {
                    t(&x.conv1p_weight)
                        + t(&x.conv1g_weight)
                        + t(&x.g_bias)
                        + t(&x.g_matmul)
                        + t(&x.pass_matmul1)
                        + t(&x.pass_bias1)
                        + t(&x.pass_matmul2)
                        + t(&x.bias2)
                        + t(&x.conv2p_weight)
                }
                Layer::ValueHead(x) => {
                    t(&x.conv1_weight)
                        + t(&x.bias1)
                        + t(&x.linear2_weight)
                        + t(&x.linear2_bias)
                        + t(&x.value_matmul)
                        + t(&x.value_bias)
                        + t(&x.misc_matmul)
                        + t(&x.misc_bias)
                        + t(&x.moremisc_matmul)
                        + t(&x.moremisc_bias)
                        + t(&x.ownership_conv)
                }
            })
            .sum()
    }
}

/// 便捷入口：字节 → 层图（加载 → 折叠 → 匹配 → 校验）。
pub fn parse_layer_graph(bytes: &[u8]) -> Result<LayerGraph, String> {
    let graph = load_onnx(bytes)?;
    build_layer_graph(&graph)
}

/// 从已加载的 [`OnnxGraph`] 构建层图。
pub fn build_layer_graph(graph: &OnnxGraph) -> Result<LayerGraph, String> {
    let simplified = fold_graph(&graph.graph)?;
    match_layers(&simplified, &graph.inputs, &graph.outputs)
}
