//! Lower the supported native TF3 descriptor directly to the CUDA layer graph.
//!
//! Native matrices are in-first; the CUDA executor uses out-first matrices.
//! No ONNX export, model hash replacement, or activation substitution occurs.
//! Native 19x19 nested TF3 transformers use model-provided trunk/head/FFN widths.
//! D=32 attention, equal Q/K/V heads and the existing 96/192 heads remain required.

use crate::activations::ACTIVATION_SILU;
use crate::desc::*;
use crate::onnx_parser::{
    AttentionLayer, FfnLayer, GateSiluLayer, InitialConvLayer, Layer, LayerGraph, MatMulLayer,
    PolicyHeadLayer, RmsNormLayer, Tensor, TensorData, TrunkFinalLayer, ValueHeadLayer,
};

const DIM: usize = 32;
const SIDE: usize = 19;
const SEQ: usize = SIDE * SIDE;

/// Convert a parsed native v17 TF3 model into the existing execution graph.
/// Unsupported architecture or activation combinations fail before GPU upload.
pub fn lower_model(desc: &ModelDesc) -> Result<LayerGraph, String> {
    ensure(
        desc.model_version == 17
            && desc.num_input_channels == 22
            && desc.num_input_global_channels == 19
            && desc.num_input_meta_channels == 0
            && desc.meta_encoder_version == 0
            && desc.num_value_channels == 3
            && desc.num_score_value_channels == 6
            && desc.num_ownership_channels == 1,
        "native CUDA requires v17, 22 spatial/19 global inputs, no metadata, and value/score/ownership 3/6/1",
    )?;
    let trunk = &desc.trunk;
    let trunk_channels = trunk.trunk_num_channels as usize;
    let mid = trunk.mid_num_channels as usize;
    let heads = mid / DIM;
    ensure(
        trunk.model_version == 17
            && (32..=4096).contains(&trunk.trunk_num_channels)
            && trunk_channels % 16 == 0
            && (32..=2048).contains(&trunk.mid_num_channels)
            && mid % DIM == 0
            && trunk_channels > mid
            && (1..=64).contains(&trunk.num_blocks)
            && trunk.blocks.len() == trunk.num_blocks as usize
            && trunk.meta_encoder_version == 0
            && trunk.trunk_norm_kind == TRUNK_NORM_KIND_STANDARD,
        "native CUDA requires 1..64 nested blocks, aligned trunk/mid channels, D32 attention and standard trunk-tip BN",
    )?;
    let blocks: Vec<&NestedBottleneckResidualBlockDesc> = trunk
        .blocks
        .iter()
        .map(|block| match block {
            BlockDesc::NestedBottleneck(block) => Ok(block.as_ref()),
            _ => Err("native CUDA requires only nested bottleneck trunk blocks".to_owned()),
        })
        .collect::<Result<_, _>>()?;
    let mut layers = Vec::new();
    for block in &blocks {
        ensure(
            block.num_blocks > 0
                && block.num_blocks % 2 == 0
                && block.blocks.len() == block.num_blocks as usize,
            "each native nested block must contain complete attention/FFN pairs",
        )?;
        silu(&block.pre_activation)?;
        silu(&block.post_activation)?;
        bn_affine(&block.pre_bn, trunk_channels)?;
        bn_affine(&block.post_bn, mid)?;
    }
    let (scale, bias) = bn_affine(&blocks[0].pre_bn, trunk_channels)?;
    layers.push(Layer::InitialConv(InitialConvLayer {
        weight: conv(&trunk.initial_conv, trunk_channels, 22, 3)?,
        global_weight: matmul(&trunk.initial_mat_mul, trunk_channels, 19)?,
        gate_scale: vector(scale),
        gate_bias: vector(bias),
        out_channels: trunk_channels,
    }));
    for (index, block) in blocks.iter().enumerate() {
        layers.push(Layer::Linear(linear_conv(
            &block.pre_conv,
            mid,
            trunk_channels,
        )?));
        for pair in block.blocks.chunks_exact(2) {
            let attention = match &pair[0] {
                BlockDesc::TransformerAttention(layer) => layer,
                _ => return Err("native nested block must alternate attention then FFN".into()),
            };
            let ffn = match &pair[1] {
                BlockDesc::TransformerFfn(layer) => layer,
                _ => return Err("native nested block must alternate attention then FFN".into()),
            };
            layers.push(Layer::RmsNorm(rms(&attention.pre_ln, mid)?));
            layers.push(Layer::Attention(lower_attention(attention, mid)?));
            layers.push(Layer::RmsNorm(rms(&ffn.pre_ln, mid)?));
            let hidden = ffn.ffn_channels as usize;
            ensure(
                ffn.num_channels == mid as i32
                    && ffn.ffn_channels > 0
                    && hidden <= 3 * mid
                    && hidden % 8 == 0
                    && ffn.use_swi_glu,
                "native CUDA requires SwiGLU FFN with matching mid channels and 8-aligned positive hidden width <= 3*mid",
            )?;
            // C++ uses silu(linear1(x)) * linearGate(x), despite the names.
            layers.push(Layer::Ffn(FfnLayer {
                gate_weight: matmul(&ffn.linear1, hidden, mid)?,
                up_weight: matmul(&ffn.linear_gate, hidden, mid)?,
                down_weight: matmul(&ffn.linear2, mid, hidden)?,
                hidden,
                residual_add: true,
            }));
        }
        layers.push(Layer::GateSilu(gate(&block.post_bn, mid)?));
        // Existing Linear-up execution adds to the separate raw trunk.
        layers.push(Layer::Linear(linear_conv(
            &block.post_conv,
            trunk_channels,
            mid,
        )?));
        if let Some(next) = blocks.get(index + 1) {
            layers.push(Layer::GateSilu(gate(&next.pre_bn, trunk_channels)?));
        }
    }
    silu(&trunk.trunk_tip_activation)?;
    let (scale, bias) = bn_affine(&trunk.trunk_tip_bn, trunk_channels)?;
    layers.push(Layer::TrunkFinal(TrunkFinalLayer {
        mean: vector(vec![0.0; trunk_channels]),
        std: vector(vec![1.0; trunk_channels]),
        gamma: vector(scale),
        beta: vector(bias),
        channels: trunk_channels,
    }));
    layers.push(Layer::PolicyHead(lower_policy(desc, trunk_channels)?));
    layers.push(Layer::ValueHead(lower_value(
        &desc.value_head,
        trunk_channels,
    )?));
    let mut graph = LayerGraph {
        layers,
        num_spatial_inputs: 22,
        num_global_inputs: 19,
        board_size: SIDE,
        trunk_channels,
        mid_channels: mid,
        num_blocks: trunk.blocks.len(),
        num_heads: heads,
        head_dim: DIM,
        // This is the lowered graph's storage count, not the native parameter
        // count: RoPE tables and zero-padded compatibility outputs are expanded.
        total_params: 0,
        scalar_params: 0,
        input_names: vec!["input_spatial".into(), "input_global".into()],
        output_names: vec![
            "policy".into(),
            "value".into(),
            "misc".into(),
            "moremisc".into(),
            "ownership".into(),
        ],
    };
    graph.total_params = graph.layer_param_elts();
    Ok(graph)
}

fn lower_attention(desc: &TransformerAttentionDesc, mid: usize) -> Result<AttentionLayer, String> {
    let heads = mid / DIM;
    ensure(
        desc.num_heads == heads as i32
            && desc.num_kv_heads == heads as i32
            && desc.q_head_dim == DIM as i32
            && desc.v_head_dim == DIM as i32
            && desc.use_rope
            && desc.learnable_rope,
        "native CUDA requires matching Q/K/V head counts, width 32 and learned 2D RoPE",
    )?;
    let mut packed = Vec::with_capacity(3 * mid * mid);
    for projection in [&desc.q_proj, &desc.k_proj, &desc.v_proj] {
        packed.extend_from_slice(matmul(projection, mid, mid)?.f32_data());
    }
    ensure(
        desc.rope_num_kv_heads == heads as i32,
        "RoPE head count must match attention",
    )?;
    let (cos, sin) = rope_tables(desc)?;
    Ok(AttentionLayer {
        num_heads: heads,
        head_dim: DIM,
        seq_len: SEQ,
        qkv_weight: tensor(&[3 * mid, mid], packed),
        out_weight: matmul(&desc.out_proj, mid, mid)?,
        rope_cos: tensor(&[SEQ, mid / 2], cos),
        rope_sin: tensor(&[SEQ, mid / 2], sin),
        qk_scale: 1.0 / (DIM as f32).sqrt().sqrt(),
        residual_add: true,
    })
}

fn rope_tables(desc: &TransformerAttentionDesc) -> Result<(Vec<f32>, Vec<f32>), String> {
    let heads = desc.rope_num_kv_heads as usize;
    let pairs = DIM / 2;
    ensure(
        (1..=64).contains(&desc.rope_num_kv_heads) && desc.rope_num_pairs == pairs as i32,
        "native RoPE frequency shape must be [heads,16,2]",
    )?;
    finite_len(&desc.rope_freqs, heads * pairs * 2, "RoPE frequencies")?;
    let mut cos = vec![0.0; SEQ * heads * pairs];
    let mut sin = vec![0.0; SEQ * heads * pairs];
    for y in 0..SIDE {
        for x in 0..SIDE {
            for head in 0..heads {
                for pair in 0..pairs {
                    let frequency = (head * pairs + pair) * 2;
                    // Match desc.cpp::computeRopeCosSin's f32 arithmetic and
                    // interleaved channel pairs (2p, 2p+1), then transpose from
                    // its [head,pair,position] table to executor [position,head,pair].
                    let angle = x as f32 * desc.rope_freqs[frequency]
                        + y as f32 * desc.rope_freqs[frequency + 1];
                    let index = ((y * SIDE + x) * heads + head) * pairs + pair;
                    cos[index] = angle.cos();
                    sin[index] = angle.sin();
                }
            }
        }
    }
    Ok((cos, sin))
}

fn lower_policy(desc: &ModelDesc, trunk: usize) -> Result<PolicyHeadLayer, String> {
    let head = &desc.policy_head;
    let channels = head.policy_out_channels as usize;
    ensure(
        head.model_version == 17
            && matches!(channels, 2 | 4)
            && desc.num_policy_channels == channels as i32,
        "native v17 policy requires two or four output channels",
    )?;
    for act in [
        &head.g1_activation,
        &head.p1_activation,
        &head.pass_activation,
    ] {
        silu(act)?;
    }
    let (g_scale, g_bias) = bn_affine(&head.g1_bn, 96)?;
    let (p_scale, p_bias) = bn_affine(&head.p1_bn, 96)?;
    ensure(
        g_scale.iter().chain(&p_scale).all(|&scale| scale == 1.0),
        "native CUDA policy head requires unit BN scales; folding nonunit scales changes FP16 storage boundaries",
    )?;
    let p2 = conv(&head.p2_conv, channels, 96, 1)?;
    let pass2 = matmul(&head.gpool_to_pass_mul2, channels, 96)?;
    Ok(PolicyHeadLayer {
        conv1p_weight: scale_rows(conv(&head.p1_conv, 96, trunk, 1)?, &p_scale)?,
        conv1g_weight: scale_rows(conv(&head.g1_conv, 96, trunk, 1)?, &g_scale)?,
        g_bias: vector(g_bias),
        g_matmul: scale_rows(matmul(&head.gpool_to_bias_mul, 96, 288)?, &p_scale)?,
        pass_matmul1: matmul(&head.gpool_to_pass_mul, 96, 288)?,
        pass_bias1: bias(&head.gpool_to_pass_bias, 96)?,
        // The executor stores six ONNX channels. Native channels 0 and 1 are
        // base/optimistic policy, mapped to executor 0 and 5. Native Q auxiliary
        // channels (when present) are not part of the Worker NNOutput contract.
        pass_matmul2: remap_rows(&pass2, 6, &[(0, 0), (1, 5)])?,
        bias2: vector(p_bias),
        conv2p_weight: remap_rows(&p2, 6, &[(0, 0), (1, 5)])?,
        act_silu: true,
        mask_scale: 0.5,
    })
}

fn lower_value(head: &ValueHeadDesc, trunk: usize) -> Result<ValueHeadLayer, String> {
    ensure(head.model_version == 17, "native value head must be v17")?;
    silu(&head.v1_activation)?;
    silu(&head.v2_activation)?;
    let (scale, v_bias) = bn_affine(&head.v1_bn, 192)?;
    ensure(
        scale.iter().all(|&scale| scale == 1.0),
        "native CUDA value head requires unit BN scales; folding nonunit scales changes FP16 storage boundaries",
    )?;
    let score = matmul(&head.sv3_mul, 6, 192)?;
    let score_bias = bias(&head.sv3_bias, 6)?;
    Ok(ValueHeadLayer {
        conv1_weight: scale_rows(conv(&head.v1_conv, 192, trunk, 1)?, &scale)?,
        bias1: vector(v_bias),
        linear2_weight: matmul(&head.v2_mul, 192, 576)?,
        linear2_bias: bias(&head.v2_bias, 192)?,
        value_matmul: matmul(&head.v3_mul, 3, 192)?,
        value_bias: bias(&head.v3_bias, 3)?,
        // Native score channels: mean, stdev, lead, variance-time,
        // shortterm winloss error, shortterm score error.
        misc_matmul: remap_rows(&score, 10, &[(0, 0), (1, 1), (2, 2), (3, 3)])?,
        misc_bias: remap_rows(&score_bias, 10, &[(0, 0), (1, 1), (2, 2), (3, 3)])?,
        moremisc_matmul: remap_rows(&score, 8, &[(4, 0), (5, 1)])?,
        moremisc_bias: remap_rows(&score_bias, 8, &[(4, 0), (5, 1)])?,
        ownership_conv: conv(&head.v_ownership_conv, 1, 192, 1)?,
        act_silu: true,
        mask_scale: 0.5,
        mask_quad: 0.15,
    })
}

fn rms(desc: &TransformerRMSNormDesc, mid: usize) -> Result<RmsNormLayer, String> {
    ensure(
        desc.num_channels == mid as i32 && desc.epsilon.is_finite() && desc.epsilon > 0.0,
        "native transformer RMSNorm requires matching channels and positive finite epsilon",
    )?;
    finite_len(&desc.weight, mid, &desc.name)?;
    Ok(RmsNormLayer {
        scale: vector(desc.weight.clone()),
        channels: mid,
        eps: desc.epsilon,
    })
}

fn gate(desc: &BatchNormLayerDesc, channels: usize) -> Result<GateSiluLayer, String> {
    let (scale, bias) = bn_affine(desc, channels)?;
    Ok(GateSiluLayer {
        scale: vector(scale),
        bias: vector(bias),
        channels,
    })
}

fn bn_affine(desc: &BatchNormLayerDesc, channels: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
    ensure(
        desc.num_channels == channels as i32,
        &format!("{}: BN width must be {channels}", desc.name),
    )?;
    finite_len(
        &desc.merged_scale,
        channels,
        &format!("{} merged scale", desc.name),
    )?;
    finite_len(
        &desc.merged_bias,
        channels,
        &format!("{} merged bias", desc.name),
    )?;
    Ok((desc.merged_scale.clone(), desc.merged_bias.clone()))
}

fn linear_conv(desc: &ConvLayerDesc, n: usize, k: usize) -> Result<MatMulLayer, String> {
    Ok(MatMulLayer {
        weight: conv(desc, n, k, 1)?,
        bias: None,
        n,
        k,
        act: None,
        residual_add: false,
    })
}

fn conv(desc: &ConvLayerDesc, n: usize, k: usize, kernel: usize) -> Result<Tensor, String> {
    ensure(
        desc.out_channels == n as i32
            && desc.in_channels == k as i32
            && desc.conv_x_size == kernel as i32
            && desc.conv_y_size == kernel as i32
            && desc.dilation_x == 1
            && desc.dilation_y == 1,
        &format!(
            "{}: expected convolution [{n},{k},{kernel},{kernel}] with dilation 1",
            desc.name
        ),
    )?;
    finite_len(&desc.weights, n * k * kernel * kernel, &desc.name)?;
    Ok(if kernel == 1 {
        tensor(&[n, k], desc.weights.clone())
    } else {
        tensor(&[n, k, kernel, kernel], desc.weights.clone())
    })
}

fn matmul(desc: &MatMulLayerDesc, n: usize, k: usize) -> Result<Tensor, String> {
    ensure(
        desc.out_channels == n as i32 && desc.in_channels == k as i32,
        &format!("{}: expected matrix in={k}, out={n}", desc.name),
    )?;
    finite_len(&desc.weights, n * k, &desc.name)?;
    let mut out = vec![0.0; n * k];
    for input in 0..k {
        for output in 0..n {
            out[output * k + input] = desc.weights[input * n + output];
        }
    }
    Ok(tensor(&[n, k], out))
}

fn bias(desc: &MatBiasLayerDesc, channels: usize) -> Result<Tensor, String> {
    ensure(
        desc.num_channels == channels as i32,
        &format!("{}: bias width must be {channels}", desc.name),
    )?;
    finite_len(&desc.weights, channels, &desc.name)?;
    Ok(vector(desc.weights.clone()))
}

fn scale_rows(mut weights: Tensor, scale: &[f32]) -> Result<Tensor, String> {
    ensure(
        weights.dims.len() == 2 && weights.dims[0] == scale.len() as i64,
        "BN folding matrix width mismatch",
    )?;
    let k = weights.dims[1] as usize;
    let TensorData::F32(data) = &mut weights.data else {
        return Err("BN folding expects f32 weights".into());
    };
    for (row, scale) in data.chunks_exact_mut(k).zip(scale) {
        for value in row {
            *value *= scale;
        }
    }
    finite_len(data, data.len(), "BN-folded weights")?;
    Ok(weights)
}

fn remap_rows(source: &Tensor, rows: usize, mapping: &[(usize, usize)]) -> Result<Tensor, String> {
    ensure(
        matches!(source.dims.len(), 1 | 2),
        "head remapping requires a vector or matrix",
    )?;
    let source_rows = source.dims[0] as usize;
    let width = source.numel() / source_rows;
    let mut data = vec![0.0; rows * width];
    for &(from, to) in mapping {
        ensure(
            from < source_rows && to < rows,
            "head channel remapping out of bounds",
        )?;
        data[to * width..(to + 1) * width]
            .copy_from_slice(&source.f32_data()[from * width..(from + 1) * width]);
    }
    let dims = if source.dims.len() == 1 {
        vec![rows]
    } else {
        vec![rows, width]
    };
    Ok(tensor(&dims, data))
}

fn silu(desc: &ActivationLayerDesc) -> Result<(), String> {
    ensure(
        desc.activation == ACTIVATION_SILU,
        &format!(
            "{}: native CUDA requires SiLU, got activation {}",
            desc.name, desc.activation
        ),
    )
}

fn finite_len(values: &[f32], length: usize, name: &str) -> Result<(), String> {
    ensure(
        values.len() == length && values.iter().all(|value| value.is_finite()),
        &format!(
            "{name}: expected {length} finite f32 values, got {}",
            values.len()
        ),
    )
}

fn ensure(condition: bool, message: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(format!("native CUDA model: {message}"))
    }
}

fn tensor(dims: &[usize], values: Vec<f32>) -> Tensor {
    Tensor {
        dims: dims.iter().map(|&d| d as i64).collect(),
        data: TensorData::F32(values),
    }
}

fn vector(values: Vec<f32>) -> Tensor {
    tensor(&[values.len()], values)
}

#[cfg(test)]
mod tests {
    use super::*;
    const HEADS: usize = 12;
    const MID: usize = 384;

    #[test]
    fn native_matrix_transposes_and_head_channels_keep_identity() {
        let source = matmul(
            &MatMulLayerDesc {
                in_channels: 3,
                out_channels: 2,
                weights: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                ..Default::default()
            },
            2,
            3,
        )
        .unwrap();
        assert_eq!(source.f32_data(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
        let packed = remap_rows(&source, 6, &[(0, 0), (1, 5)]).unwrap();
        assert_eq!(&packed.f32_data()[..3], &[1.0, 3.0, 5.0]);
        assert_eq!(&packed.f32_data()[15..], &[2.0, 4.0, 6.0]);
        assert!(packed.f32_data()[3..15].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn rope_uses_xy_frequencies_and_position_major_interleaved_pairs() {
        let mut desc = TransformerAttentionDesc {
            rope_num_kv_heads: HEADS as i32,
            rope_num_pairs: (DIM / 2) as i32,
            rope_freqs: vec![0.0; MID],
            ..Default::default()
        };
        let frequency = (3 * (DIM / 2) + 5) * 2;
        desc.rope_freqs[frequency] = 0.2;
        desc.rope_freqs[frequency + 1] = -0.3;
        let (cos, sin) = rope_tables(&desc).unwrap();
        let index = ((2 * SIDE + 7) * HEADS + 3) * (DIM / 2) + 5;
        let angle: f32 = 7.0 * 0.2 + 2.0 * -0.3;
        assert_eq!(cos[index], angle.cos());
        assert_eq!(sin[index], angle.sin());
        assert_eq!(cos[0], 1.0);
        assert_eq!(sin[0], 0.0);
    }

    #[test]
    fn mish_is_not_silently_relabelled_silu() {
        assert!(
            silu(&ActivationLayerDesc {
                name: "mish".into(),
                activation: crate::activations::ACTIVATION_MISH,
            })
            .is_err()
        );
        assert!(lower_model(&ModelDesc::default()).is_err());
    }
}
