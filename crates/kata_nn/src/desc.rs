//! Neural-network model data descriptors.
//!
//! Corresponds to `cpp/neuralnet/desc.h` and `cpp/neuralnet/desc.cpp`.
//! This module ports the descriptor structs and their parameter-count helpers;
//! model-file parsing and the `istream` constructors are intentionally omitted.

use crate::activations::ACTIVATION_RELU;

/// Trunk final normalization kind: standard BatchNorm / BiasMask.
pub const TRUNK_NORM_KIND_STANDARD: i32 = 0;
/// Trunk final normalization kind: RMSNorm.
pub const TRUNK_NORM_KIND_RMSNORM: i32 = 1;

/// Ordinary residual block.
pub const ORDINARY_BLOCK_KIND: i32 = 0;
/// Global-pooling residual block.
pub const GLOBAL_POOLING_BLOCK_KIND: i32 = 2;
/// Nested-bottleneck residual block.
pub const NESTED_BOTTLENECK_BLOCK_KIND: i32 = 3;
/// Transformer attention block.
pub const TRANSFORMER_ATTENTION_BLOCK_KIND: i32 = 4;
/// Transformer feed-forward block.
pub const TRANSFORMER_FFN_BLOCK_KIND: i32 = 5;

/// Convolution layer descriptor.
#[derive(Debug, Default, Clone)]
pub struct ConvLayerDesc {
    pub name: String,
    pub conv_y_size: i32,
    pub conv_x_size: i32,
    pub in_channels: i32,
    pub out_channels: i32,
    pub dilation_y: i32,
    pub dilation_x: i32,
    /// `outC x inC x H x W` in row-major order (W has least stride, outC greatest).
    pub weights: Vec<f32>,
}

impl ConvLayerDesc {
    /// 1x1 = 0, 3x3 = 1, 5x5 = 2, ...
    pub fn get_spatial_conv_depth(&self) -> f64 {
        (self.conv_y_size + self.conv_x_size - 2) as f64 / 4.0
    }

    pub fn get_num_parameters(&self) -> i64 {
        self.weights.len() as i64
    }
}

/// Batch-normalization layer descriptor.
#[derive(Debug, Clone)]
pub struct BatchNormLayerDesc {
    pub name: String,
    pub num_channels: i32,
    pub epsilon: f32,
    pub has_scale: bool,
    pub has_bias: bool,
    pub mean: Vec<f32>,
    pub variance: Vec<f32>,
    pub scale: Vec<f32>,
    pub bias: Vec<f32>,
    pub merged_scale: Vec<f32>,
    pub merged_bias: Vec<f32>,
}

impl BatchNormLayerDesc {
    pub fn get_num_parameters(&self) -> i64 {
        // Count learnable scale and bias; mean/variance are running statistics.
        (if self.has_scale {
            self.num_channels as i64
        } else {
            0
        }) + (if self.has_bias {
            self.num_channels as i64
        } else {
            0
        })
    }
}

impl Default for BatchNormLayerDesc {
    fn default() -> Self {
        Self {
            name: String::new(),
            num_channels: 0,
            epsilon: 0.001f32,
            has_scale: false,
            has_bias: false,
            mean: Vec::new(),
            variance: Vec::new(),
            scale: Vec::new(),
            bias: Vec::new(),
            merged_scale: Vec::new(),
            merged_bias: Vec::new(),
        }
    }
}

/// Activation layer descriptor.
#[derive(Debug, Clone)]
pub struct ActivationLayerDesc {
    pub name: String,
    pub activation: i32,
}

impl Default for ActivationLayerDesc {
    fn default() -> Self {
        Self {
            name: String::new(),
            activation: ACTIVATION_RELU,
        }
    }
}

/// Matrix-multiplication layer descriptor.
#[derive(Debug, Default, Clone)]
pub struct MatMulLayerDesc {
    pub name: String,
    pub in_channels: i32,
    pub out_channels: i32,
    /// `inC x outC`.
    pub weights: Vec<f32>,
}

impl MatMulLayerDesc {
    pub fn get_num_parameters(&self) -> i64 {
        self.weights.len() as i64
    }
}

/// Matrix-bias layer descriptor.
#[derive(Debug, Default, Clone)]
pub struct MatBiasLayerDesc {
    pub name: String,
    pub num_channels: i32,
    pub weights: Vec<f32>,
}

impl MatBiasLayerDesc {
    pub fn get_num_parameters(&self) -> i64 {
        self.weights.len() as i64
    }
}

/// Ordinary residual block descriptor.
#[derive(Debug, Default, Clone)]
pub struct ResidualBlockDesc {
    pub name: String,
    pub pre_bn: BatchNormLayerDesc,
    pub pre_activation: ActivationLayerDesc,
    pub regular_conv: ConvLayerDesc,
    pub mid_bn: BatchNormLayerDesc,
    pub mid_activation: ActivationLayerDesc,
    pub final_conv: ConvLayerDesc,
}

impl ResidualBlockDesc {
    pub fn iter_conv_layers(&self, f: &mut dyn FnMut(&ConvLayerDesc)) {
        f(&self.regular_conv);
        f(&self.final_conv);
    }

    pub fn get_spatial_conv_depth(&self) -> f64 {
        self.regular_conv.get_spatial_conv_depth() + self.final_conv.get_spatial_conv_depth()
    }

    pub fn get_num_parameters(&self) -> i64 {
        self.pre_bn.get_num_parameters()
            + self.regular_conv.get_num_parameters()
            + self.mid_bn.get_num_parameters()
            + self.final_conv.get_num_parameters()
    }
}

/// Global-pooling residual block descriptor.
#[derive(Debug, Clone)]
pub struct GlobalPoolingResidualBlockDesc {
    pub name: String,
    pub model_version: i32,
    pub pre_bn: BatchNormLayerDesc,
    pub pre_activation: ActivationLayerDesc,
    pub regular_conv: ConvLayerDesc,
    pub gpool_conv: ConvLayerDesc,
    pub gpool_bn: BatchNormLayerDesc,
    pub gpool_activation: ActivationLayerDesc,
    pub gpool_to_bias_mul: MatMulLayerDesc,
    pub mid_bn: BatchNormLayerDesc,
    pub mid_activation: ActivationLayerDesc,
    pub final_conv: ConvLayerDesc,
}

impl GlobalPoolingResidualBlockDesc {
    pub fn iter_conv_layers(&self, f: &mut dyn FnMut(&ConvLayerDesc)) {
        f(&self.regular_conv);
        f(&self.gpool_conv);
        f(&self.final_conv);
    }

    pub fn get_spatial_conv_depth(&self) -> f64 {
        self.regular_conv.get_spatial_conv_depth() + self.final_conv.get_spatial_conv_depth()
    }

    pub fn get_num_parameters(&self) -> i64 {
        self.pre_bn.get_num_parameters()
            + self.regular_conv.get_num_parameters()
            + self.gpool_conv.get_num_parameters()
            + self.gpool_bn.get_num_parameters()
            + self.gpool_to_bias_mul.get_num_parameters()
            + self.mid_bn.get_num_parameters()
            + self.final_conv.get_num_parameters()
    }
}

impl Default for GlobalPoolingResidualBlockDesc {
    fn default() -> Self {
        Self {
            name: String::new(),
            model_version: -1,
            pre_bn: BatchNormLayerDesc::default(),
            pre_activation: ActivationLayerDesc::default(),
            regular_conv: ConvLayerDesc::default(),
            gpool_conv: ConvLayerDesc::default(),
            gpool_bn: BatchNormLayerDesc::default(),
            gpool_activation: ActivationLayerDesc::default(),
            gpool_to_bias_mul: MatMulLayerDesc::default(),
            mid_bn: BatchNormLayerDesc::default(),
            mid_activation: ActivationLayerDesc::default(),
            final_conv: ConvLayerDesc::default(),
        }
    }
}

/// One block inside a heterogeneous residual stack.
///
/// Mirrors C++ runtime-polymorphic block storage; the large variant sizes are expected.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum BlockDesc {
    Ordinary(ResidualBlockDesc),
    GlobalPooling(GlobalPoolingResidualBlockDesc),
    NestedBottleneck(Box<NestedBottleneckResidualBlockDesc>),
    TransformerAttention(TransformerAttentionDesc),
    TransformerFfn(TransformerFFNDesc),
}

impl BlockDesc {
    fn iter_conv_layers(&self, f: &mut dyn FnMut(&ConvLayerDesc)) {
        match self {
            BlockDesc::Ordinary(b) => b.iter_conv_layers(f),
            BlockDesc::GlobalPooling(b) => b.iter_conv_layers(f),
            BlockDesc::NestedBottleneck(b) => b.iter_conv_layers(f),
            BlockDesc::TransformerAttention(_) | BlockDesc::TransformerFfn(_) => {
                // No conv layers in transformer blocks.
            }
        }
    }

    fn get_spatial_conv_depth(&self) -> f64 {
        match self {
            BlockDesc::Ordinary(b) => b.get_spatial_conv_depth(),
            BlockDesc::GlobalPooling(b) => b.get_spatial_conv_depth(),
            BlockDesc::NestedBottleneck(b) => b.get_spatial_conv_depth(),
            // Transformer blocks don't technically contribute spatial conv depth but in practice
            // we count it as 2 for things that want a crude idea of model size.
            BlockDesc::TransformerAttention(_) | BlockDesc::TransformerFfn(_) => 2.0,
        }
    }

    fn get_num_parameters(&self) -> i64 {
        match self {
            BlockDesc::Ordinary(b) => b.get_num_parameters(),
            BlockDesc::GlobalPooling(b) => b.get_num_parameters(),
            BlockDesc::NestedBottleneck(b) => b.get_num_parameters(),
            BlockDesc::TransformerAttention(b) => b.get_num_parameters(),
            BlockDesc::TransformerFfn(b) => b.get_num_parameters(),
        }
    }

    fn is_transformer(&self) -> bool {
        matches!(
            self,
            BlockDesc::TransformerAttention(_) | BlockDesc::TransformerFfn(_)
        )
    }

    fn has_any_transformer_blocks(&self) -> bool {
        if self.is_transformer() {
            return true;
        }
        if let BlockDesc::NestedBottleneck(b) = self {
            return b.has_any_transformer_blocks();
        }
        false
    }
}

/// Nested-bottleneck residual block descriptor.
#[derive(Debug, Default, Clone)]
pub struct NestedBottleneckResidualBlockDesc {
    pub name: String,
    pub num_blocks: i32,
    pub pre_bn: BatchNormLayerDesc,
    pub pre_activation: ActivationLayerDesc,
    pub pre_conv: ConvLayerDesc,
    pub blocks: Vec<BlockDesc>,
    pub post_bn: BatchNormLayerDesc,
    pub post_activation: ActivationLayerDesc,
    pub post_conv: ConvLayerDesc,
}

impl NestedBottleneckResidualBlockDesc {
    pub fn iter_conv_layers(&self, f: &mut dyn FnMut(&ConvLayerDesc)) {
        f(&self.pre_conv);
        for block in &self.blocks {
            block.iter_conv_layers(f);
        }
        f(&self.post_conv);
    }

    pub fn get_spatial_conv_depth(&self) -> f64 {
        let mut depth = self.pre_conv.get_spatial_conv_depth();
        for block in &self.blocks {
            depth += block.get_spatial_conv_depth();
        }
        depth += self.post_conv.get_spatial_conv_depth();
        depth
    }

    pub fn get_num_parameters(&self) -> i64 {
        let mut num_parameters = 0i64;
        num_parameters += self.pre_bn.get_num_parameters();
        num_parameters += self.pre_conv.get_num_parameters();
        for block in &self.blocks {
            num_parameters += block.get_num_parameters();
        }
        num_parameters += self.post_bn.get_num_parameters();
        num_parameters += self.post_conv.get_num_parameters();
        num_parameters
    }

    /// True if this block contains any transformer block, including ones nested inside further
    /// nested-bottleneck children.
    pub fn has_any_transformer_blocks(&self) -> bool {
        self.blocks
            .iter()
            .any(BlockDesc::has_any_transformer_blocks)
    }
}

/// RMSNorm layer descriptor.
#[derive(Debug, Default, Clone)]
pub struct RMSNormLayerDesc {
    pub name: String,
    pub num_channels: i32,
    pub epsilon: f32,
    pub spatial: bool,
    /// 0 if not grouped.
    pub cgroup_size: i32,
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
}

impl RMSNormLayerDesc {
    pub fn get_num_parameters(&self) -> i64 {
        (self.gamma.len() + self.beta.len()) as i64
    }
}

/// Lightweight RMSNorm used inside transformer blocks.
#[derive(Debug, Default, Clone)]
pub struct TransformerRMSNormDesc {
    pub name: String,
    pub num_channels: i32,
    pub epsilon: f32,
    pub weight: Vec<f32>,
}

impl TransformerRMSNormDesc {
    pub fn get_num_parameters(&self) -> i64 {
        self.weight.len() as i64
    }
}

/// Transformer multi-head attention descriptor.
#[derive(Debug, Default, Clone)]
pub struct TransformerAttentionDesc {
    pub name: String,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub q_head_dim: i32,
    pub v_head_dim: i32,
    pub use_rope: bool,
    pub learnable_rope: bool,
    pub pre_ln: TransformerRMSNormDesc,
    pub q_proj: MatMulLayerDesc,
    pub k_proj: MatMulLayerDesc,
    pub v_proj: MatMulLayerDesc,
    pub out_proj: MatMulLayerDesc,
    /// For learnable RoPE: `(numKVHeads, numPairs, 2)` flattened.
    pub rope_num_kv_heads: i32,
    pub rope_num_pairs: i32,
    pub rope_freqs: Vec<f32>,
    /// For non-learnable RoPE.
    pub rope_theta: f32,
}

impl TransformerAttentionDesc {
    pub fn get_num_parameters(&self) -> i64 {
        self.pre_ln.get_num_parameters()
            + self.q_proj.get_num_parameters()
            + self.k_proj.get_num_parameters()
            + self.v_proj.get_num_parameters()
            + self.out_proj.get_num_parameters()
            + self.rope_freqs.len() as i64
    }
}

/// Transformer feed-forward network descriptor.
#[derive(Debug, Default, Clone)]
pub struct TransformerFFNDesc {
    pub name: String,
    pub num_channels: i32,
    pub ffn_channels: i32,
    pub use_swi_glu: bool,
    pub pre_ln: TransformerRMSNormDesc,
    pub linear1: MatMulLayerDesc,
    pub linear_gate: MatMulLayerDesc,
    pub linear2: MatMulLayerDesc,
}

impl TransformerFFNDesc {
    pub fn get_num_parameters(&self) -> i64 {
        self.pre_ln.get_num_parameters()
            + self.linear1.get_num_parameters()
            + self.linear_gate.get_num_parameters()
            + self.linear2.get_num_parameters()
    }
}

/// SGF metadata encoder descriptor.
#[derive(Debug, Default, Clone)]
pub struct SGFMetadataEncoderDesc {
    pub name: String,
    pub meta_encoder_version: i32,
    pub num_input_meta_channels: i32,
    pub mul1: MatMulLayerDesc,
    pub bias1: MatBiasLayerDesc,
    pub act1: ActivationLayerDesc,
    pub mul2: MatMulLayerDesc,
    pub bias2: MatBiasLayerDesc,
    pub act2: ActivationLayerDesc,
    pub mul3: MatMulLayerDesc,
}

impl SGFMetadataEncoderDesc {
    pub fn get_num_parameters(&self) -> i64 {
        self.mul1.get_num_parameters()
            + self.bias1.get_num_parameters()
            + self.mul2.get_num_parameters()
            + self.bias2.get_num_parameters()
            + self.mul3.get_num_parameters()
    }
}

/// Trunk descriptor.
#[derive(Debug, Clone)]
pub struct TrunkDesc {
    pub name: String,
    pub model_version: i32,
    pub num_blocks: i32,
    pub trunk_num_channels: i32,
    pub mid_num_channels: i32,
    pub regular_num_channels: i32,
    pub gpool_num_channels: i32,
    pub meta_encoder_version: i32,
    pub trunk_norm_kind: i32,
    pub initial_conv: ConvLayerDesc,
    pub initial_mat_mul: MatMulLayerDesc,
    pub sgf_metadata_encoder: SGFMetadataEncoderDesc,
    pub blocks: Vec<BlockDesc>,
    pub trunk_tip_bn: BatchNormLayerDesc,
    pub trunk_tip_rms_norm: RMSNormLayerDesc,
    pub trunk_tip_activation: ActivationLayerDesc,
}

impl TrunkDesc {
    pub fn iter_conv_layers(&self, f: &mut dyn FnMut(&ConvLayerDesc)) {
        f(&self.initial_conv);
        for block in &self.blocks {
            block.iter_conv_layers(f);
        }
    }

    pub fn get_spatial_conv_depth(&self) -> f64 {
        let mut depth = self.initial_conv.get_spatial_conv_depth();
        for block in &self.blocks {
            depth += block.get_spatial_conv_depth();
        }
        depth
    }

    pub fn get_num_parameters(&self) -> i64 {
        let mut num_parameters = 0i64;
        num_parameters += self.initial_conv.get_num_parameters();
        num_parameters += self.initial_mat_mul.get_num_parameters();
        if self.meta_encoder_version > 0 {
            num_parameters += self.sgf_metadata_encoder.get_num_parameters();
        }
        for block in &self.blocks {
            num_parameters += block.get_num_parameters();
        }
        // Whichever trunk tip norm is unused has empty parameter vectors, so summing both is safe.
        num_parameters += self.trunk_tip_bn.get_num_parameters();
        num_parameters += self.trunk_tip_rms_norm.get_num_parameters();
        num_parameters
    }

    /// True if any block in the trunk is a transformer block, including ones nested inside
    /// nested-bottleneck blocks.
    pub fn has_any_transformer_blocks(&self) -> bool {
        self.blocks
            .iter()
            .any(BlockDesc::has_any_transformer_blocks)
    }
}

impl Default for TrunkDesc {
    fn default() -> Self {
        Self {
            name: String::new(),
            model_version: -1,
            num_blocks: 0,
            trunk_num_channels: 0,
            mid_num_channels: 0,
            regular_num_channels: 0,
            gpool_num_channels: 0,
            meta_encoder_version: 0,
            trunk_norm_kind: TRUNK_NORM_KIND_STANDARD,
            initial_conv: ConvLayerDesc::default(),
            initial_mat_mul: MatMulLayerDesc::default(),
            sgf_metadata_encoder: SGFMetadataEncoderDesc::default(),
            blocks: Vec::new(),
            trunk_tip_bn: BatchNormLayerDesc::default(),
            trunk_tip_rms_norm: RMSNormLayerDesc::default(),
            trunk_tip_activation: ActivationLayerDesc::default(),
        }
    }
}

/// Policy head descriptor.
#[derive(Debug, Clone)]
pub struct PolicyHeadDesc {
    pub name: String,
    pub model_version: i32,
    pub policy_out_channels: i32,
    pub p1_conv: ConvLayerDesc,
    pub g1_conv: ConvLayerDesc,
    pub g1_bn: BatchNormLayerDesc,
    pub g1_activation: ActivationLayerDesc,
    pub gpool_to_bias_mul: MatMulLayerDesc,
    pub p1_bn: BatchNormLayerDesc,
    pub p1_activation: ActivationLayerDesc,
    pub p2_conv: ConvLayerDesc,
    pub gpool_to_pass_mul: MatMulLayerDesc,
    pub gpool_to_pass_bias: MatBiasLayerDesc,
    pub pass_activation: ActivationLayerDesc,
    pub gpool_to_pass_mul2: MatMulLayerDesc,
}

impl PolicyHeadDesc {
    pub fn iter_conv_layers(&self, f: &mut dyn FnMut(&ConvLayerDesc)) {
        f(&self.p1_conv);
        f(&self.g1_conv);
        f(&self.p2_conv);
    }

    pub fn get_num_parameters(&self) -> i64 {
        self.p1_conv.get_num_parameters()
            + self.g1_conv.get_num_parameters()
            + self.g1_bn.get_num_parameters()
            + self.gpool_to_bias_mul.get_num_parameters()
            + self.p1_bn.get_num_parameters()
            + self.p2_conv.get_num_parameters()
            + self.gpool_to_pass_mul.get_num_parameters()
            + self.gpool_to_pass_bias.get_num_parameters()
            + self.gpool_to_pass_mul2.get_num_parameters()
    }
}

impl Default for PolicyHeadDesc {
    fn default() -> Self {
        Self {
            name: String::new(),
            model_version: -1,
            policy_out_channels: 0,
            p1_conv: ConvLayerDesc::default(),
            g1_conv: ConvLayerDesc::default(),
            g1_bn: BatchNormLayerDesc::default(),
            g1_activation: ActivationLayerDesc::default(),
            gpool_to_bias_mul: MatMulLayerDesc::default(),
            p1_bn: BatchNormLayerDesc::default(),
            p1_activation: ActivationLayerDesc::default(),
            p2_conv: ConvLayerDesc::default(),
            gpool_to_pass_mul: MatMulLayerDesc::default(),
            gpool_to_pass_bias: MatBiasLayerDesc::default(),
            pass_activation: ActivationLayerDesc::default(),
            gpool_to_pass_mul2: MatMulLayerDesc::default(),
        }
    }
}

/// Value head descriptor.
#[derive(Debug, Clone)]
pub struct ValueHeadDesc {
    pub name: String,
    pub model_version: i32,
    pub v1_conv: ConvLayerDesc,
    pub v1_bn: BatchNormLayerDesc,
    pub v1_activation: ActivationLayerDesc,
    pub v2_mul: MatMulLayerDesc,
    pub v2_bias: MatBiasLayerDesc,
    pub v2_activation: ActivationLayerDesc,
    pub v3_mul: MatMulLayerDesc,
    pub v3_bias: MatBiasLayerDesc,
    pub sv3_mul: MatMulLayerDesc,
    pub sv3_bias: MatBiasLayerDesc,
    pub v_ownership_conv: ConvLayerDesc,
}

impl ValueHeadDesc {
    pub fn iter_conv_layers(&self, f: &mut dyn FnMut(&ConvLayerDesc)) {
        f(&self.v1_conv);
        f(&self.v_ownership_conv);
    }

    pub fn get_num_parameters(&self) -> i64 {
        self.v1_conv.get_num_parameters()
            + self.v1_bn.get_num_parameters()
            + self.v2_mul.get_num_parameters()
            + self.v2_bias.get_num_parameters()
            + self.v3_mul.get_num_parameters()
            + self.v3_bias.get_num_parameters()
            + self.sv3_mul.get_num_parameters()
            + self.sv3_bias.get_num_parameters()
            + self.v_ownership_conv.get_num_parameters()
    }
}

impl Default for ValueHeadDesc {
    fn default() -> Self {
        Self {
            name: String::new(),
            model_version: -1,
            v1_conv: ConvLayerDesc::default(),
            v1_bn: BatchNormLayerDesc::default(),
            v1_activation: ActivationLayerDesc::default(),
            v2_mul: MatMulLayerDesc::default(),
            v2_bias: MatBiasLayerDesc::default(),
            v2_activation: ActivationLayerDesc::default(),
            v3_mul: MatMulLayerDesc::default(),
            v3_bias: MatBiasLayerDesc::default(),
            sv3_mul: MatMulLayerDesc::default(),
            sv3_bias: MatBiasLayerDesc::default(),
            v_ownership_conv: ConvLayerDesc::default(),
        }
    }
}

/// Model post-processing parameters.
#[derive(Debug, Clone, Copy)]
pub struct ModelPostProcessParams {
    pub td_score_multiplier: f64,
    pub score_mean_multiplier: f64,
    pub score_stdev_multiplier: f64,
    pub lead_multiplier: f64,
    pub variance_time_multiplier: f64,
    pub shortterm_value_error_multiplier: f64,
    pub shortterm_score_error_multiplier: f64,
    pub output_scale_multiplier: f32,
}

impl Default for ModelPostProcessParams {
    fn default() -> Self {
        Self {
            td_score_multiplier: 20.0,
            score_mean_multiplier: 20.0,
            score_stdev_multiplier: 20.0,
            lead_multiplier: 20.0,
            variance_time_multiplier: 40.0,
            shortterm_value_error_multiplier: 0.25,
            shortterm_score_error_multiplier: 30.0,
            output_scale_multiplier: 1.0f32,
        }
    }
}

/// Top-level model descriptor.
#[derive(Debug, Clone)]
pub struct ModelDesc {
    pub name: String,
    pub sha256: String,
    pub model_version: i32,
    pub num_input_channels: i32,
    pub num_input_global_channels: i32,
    pub num_input_meta_channels: i32,
    pub num_policy_channels: i32,
    pub num_value_channels: i32,
    pub num_score_value_channels: i32,
    pub num_ownership_channels: i32,
    pub meta_encoder_version: i32,
    /// Compute pass-alive features as if multi-stone suicide were legal.
    pub prefer_pass_alive_under_suicide_rules: bool,
    /// Exclude empty territory points adjacent to chains in atari (rules v3).
    pub prefer_exclude_territory_adjacent_to_atari: bool,
    pub post_process_params: ModelPostProcessParams,
    pub trunk: TrunkDesc,
    pub policy_head: PolicyHeadDesc,
    pub value_head: ValueHeadDesc,
}

impl ModelDesc {
    pub fn iter_conv_layers(&self, f: &mut dyn FnMut(&ConvLayerDesc)) {
        self.trunk.iter_conv_layers(f);
        self.policy_head.iter_conv_layers(f);
        self.value_head.iter_conv_layers(f);
    }

    /// Maximum number of input/output channels among conv layers with the given kernel size.
    pub fn max_conv_channels(&self, conv_x_size: i32, conv_y_size: i32) -> i32 {
        let mut c = 0i32;
        self.iter_conv_layers(&mut |desc: &ConvLayerDesc| {
            if desc.conv_x_size == conv_x_size && desc.conv_y_size == conv_y_size {
                if desc.in_channels > c {
                    c = desc.in_channels;
                }
                if desc.out_channels > c {
                    c = desc.out_channels;
                }
            }
        });
        c
    }

    pub fn get_trunk_spatial_conv_depth(&self) -> f64 {
        self.trunk.get_spatial_conv_depth()
    }

    pub fn get_num_parameters(&self) -> i64 {
        self.trunk.get_num_parameters()
            + self.policy_head.get_num_parameters()
            + self.value_head.get_num_parameters()
    }

    /// True if the model's trunk contains any transformer block.
    pub fn has_any_transformer_blocks(&self) -> bool {
        self.trunk.has_any_transformer_blocks()
    }

    /// Short human-readable summary of the model architecture kind and parameter count.
    pub fn get_short_info_string(&self) -> String {
        let is_transformer = self.has_any_transformer_blocks();
        let is_nbt = self
            .trunk
            .blocks
            .iter()
            .any(|b| matches!(b, BlockDesc::NestedBottleneck(_)));
        let kind = if is_nbt {
            if is_transformer {
                "nbt transformer"
            } else {
                "nbt convnet"
            }
        } else if is_transformer {
            "transformer"
        } else {
            "convnet"
        };
        format!("{}, {} params", kind, self.get_num_parameters())
    }
}

impl Default for ModelDesc {
    fn default() -> Self {
        Self {
            name: String::new(),
            sha256: String::new(),
            model_version: -1,
            num_input_channels: 0,
            num_input_global_channels: 0,
            num_input_meta_channels: 0,
            num_policy_channels: 0,
            num_value_channels: 0,
            num_score_value_channels: 0,
            num_ownership_channels: 0,
            meta_encoder_version: 0,
            prefer_pass_alive_under_suicide_rules: false,
            prefer_exclude_territory_adjacent_to_atari: false,
            post_process_params: ModelPostProcessParams::default(),
            trunk: TrunkDesc::default(),
            policy_head: PolicyHeadDesc::default(),
            value_head: ValueHeadDesc::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_structs_have_zero_parameters() {
        assert_eq!(ConvLayerDesc::default().get_num_parameters(), 0);
        assert_eq!(BatchNormLayerDesc::default().get_num_parameters(), 0);
        assert_eq!(MatMulLayerDesc::default().get_num_parameters(), 0);
        assert_eq!(MatBiasLayerDesc::default().get_num_parameters(), 0);
        assert_eq!(ResidualBlockDesc::default().get_num_parameters(), 0);
        assert_eq!(
            GlobalPoolingResidualBlockDesc::default().get_num_parameters(),
            0
        );
        assert_eq!(
            NestedBottleneckResidualBlockDesc::default().get_num_parameters(),
            0
        );
        assert_eq!(RMSNormLayerDesc::default().get_num_parameters(), 0);
        assert_eq!(TransformerRMSNormDesc::default().get_num_parameters(), 0);
        assert_eq!(TransformerAttentionDesc::default().get_num_parameters(), 0);
        assert_eq!(TransformerFFNDesc::default().get_num_parameters(), 0);
        assert_eq!(SGFMetadataEncoderDesc::default().get_num_parameters(), 0);
        assert_eq!(TrunkDesc::default().get_num_parameters(), 0);
        assert_eq!(PolicyHeadDesc::default().get_num_parameters(), 0);
        assert_eq!(ValueHeadDesc::default().get_num_parameters(), 0);
        assert_eq!(ModelDesc::default().get_num_parameters(), 0);
    }

    #[test]
    fn test_conv_parameter_count() {
        let conv = ConvLayerDesc {
            conv_y_size: 3,
            conv_x_size: 3,
            in_channels: 4,
            out_channels: 8,
            weights: vec![0.0f32; 3 * 3 * 4 * 8],
            ..Default::default()
        };
        assert_eq!(conv.get_num_parameters(), 288);
        assert!((conv.get_spatial_conv_depth() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_mat_mul_parameter_count() {
        let mm = MatMulLayerDesc {
            in_channels: 16,
            out_channels: 32,
            weights: vec![0.0f32; 16 * 32],
            ..Default::default()
        };
        assert_eq!(mm.get_num_parameters(), 512);
    }

    #[test]
    fn test_mat_bias_parameter_count() {
        let mb = MatBiasLayerDesc {
            num_channels: 64,
            weights: vec![0.0f32; 64],
            ..Default::default()
        };
        assert_eq!(mb.get_num_parameters(), 64);
    }

    #[test]
    fn test_batch_norm_parameter_count() {
        let bn = BatchNormLayerDesc {
            num_channels: 32,
            has_scale: true,
            has_bias: true,
            ..Default::default()
        };
        assert_eq!(bn.get_num_parameters(), 64);

        let bn = BatchNormLayerDesc {
            num_channels: 32,
            has_scale: false,
            has_bias: true,
            ..Default::default()
        };
        assert_eq!(bn.get_num_parameters(), 32);

        let bn = BatchNormLayerDesc {
            num_channels: 32,
            has_scale: false,
            has_bias: false,
            ..Default::default()
        };
        assert_eq!(bn.get_num_parameters(), 0);
    }

    #[test]
    fn test_block_kind_constants() {
        assert_eq!(ORDINARY_BLOCK_KIND, 0);
        assert_eq!(GLOBAL_POOLING_BLOCK_KIND, 2);
        assert_eq!(NESTED_BOTTLENECK_BLOCK_KIND, 3);
        assert_eq!(TRANSFORMER_ATTENTION_BLOCK_KIND, 4);
        assert_eq!(TRANSFORMER_FFN_BLOCK_KIND, 5);
        assert_eq!(TRUNK_NORM_KIND_STANDARD, 0);
        assert_eq!(TRUNK_NORM_KIND_RMSNORM, 1);
    }

    #[test]
    fn test_has_any_transformer_blocks() {
        let mut trunk = TrunkDesc::default();
        assert!(!trunk.has_any_transformer_blocks());

        trunk
            .blocks
            .push(BlockDesc::Ordinary(ResidualBlockDesc::default()));
        assert!(!trunk.has_any_transformer_blocks());

        trunk.blocks.push(BlockDesc::TransformerAttention(
            TransformerAttentionDesc::default(),
        ));
        assert!(trunk.has_any_transformer_blocks());

        let mut nested = NestedBottleneckResidualBlockDesc::default();
        assert!(!nested.has_any_transformer_blocks());
        nested
            .blocks
            .push(BlockDesc::TransformerFfn(TransformerFFNDesc::default()));
        assert!(nested.has_any_transformer_blocks());

        // Transformer nested inside a nested-bottleneck inside the trunk.
        let mut trunk2 = TrunkDesc::default();
        trunk2
            .blocks
            .push(BlockDesc::NestedBottleneck(Box::new(nested)));
        assert!(trunk2.has_any_transformer_blocks());
    }

    #[test]
    fn test_get_short_info_string() {
        let mut model = ModelDesc::default();
        assert_eq!(model.get_short_info_string(), "convnet, 0 params");

        model
            .trunk
            .blocks
            .push(BlockDesc::NestedBottleneck(Box::default()));
        assert_eq!(model.get_short_info_string(), "nbt convnet, 0 params");

        model.trunk.blocks.push(BlockDesc::TransformerAttention(
            TransformerAttentionDesc::default(),
        ));
        assert_eq!(model.get_short_info_string(), "nbt transformer, 0 params");
    }
}
