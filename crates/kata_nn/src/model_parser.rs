//! Model file parser for KataGo `.bin.gz` / `.txt.gz` / `.txt` model files.

use std::io::Read;

use flate2::bufread::GzDecoder;
use sha2::{Digest, Sha256};

use kata_core::global::StringError;

use crate::activations::{ACTIVATION_IDENTITY, ACTIVATION_MISH, ACTIVATION_RELU, ACTIVATION_SILU};
use crate::desc::*;

#[derive(Debug, thiserror::Error)]
pub enum ModelParseError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("string error: {0}")]
    StringError(String),
    #[error("parse: {0}")]
    Parse(String),
}

impl From<StringError> for ModelParseError {
    fn from(e: StringError) -> Self {
        ModelParseError::StringError(format!("{e}"))
    }
}

struct Parser<'a> {
    buf: &'a [u8],
    pos: usize,
    binary: bool,
}

impl<'a> Parser<'a> {
    fn new(buf: &'a [u8], binary: bool) -> Self {
        Self {
            buf,
            pos: 0,
            binary,
        }
    }

    /// Read a float32 as text (all scalar fields are text even in .bin files).
    fn read_f32(&mut self) -> Result<f32, ModelParseError> {
        let t = self.read_token()?;
        let value = t
            .parse::<f32>()
            .map_err(|_| ModelParseError::Parse(format!("bad f32: {t}")))?;
        require(value.is_finite(), "nonfinite model float")?;
        Ok(value)
    }

    fn read_f64(&mut self) -> Result<f64, ModelParseError> {
        let t = self.read_token()?;
        let value = t
            .parse::<f64>()
            .map_err(|_| ModelParseError::Parse(format!("bad f64: {t}")))?;
        require(value.is_finite(), "nonfinite model scalar")?;
        Ok(value)
    }

    /// Read an i32 as text (all scalar fields are text even in .bin files).
    fn read_i32(&mut self) -> Result<i32, ModelParseError> {
        let t = self.read_token()?;
        t.parse::<i32>()
            .map_err(|_| ModelParseError::Parse(format!("bad i32: {t}")))
    }

    fn read_bool(&mut self) -> Result<bool, ModelParseError> {
        match self.read_i32()? {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(ModelParseError::Parse(format!("invalid boolean {value}"))),
        }
    }

    fn reserved(&mut self, count: usize, context: &str) -> Result<(), ModelParseError> {
        for _ in 0..count {
            require(
                self.read_i32()? == 0,
                &format!("unsupported {context} option"),
            )?;
        }
        Ok(())
    }

    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn read_token(&mut self) -> Result<String, ModelParseError> {
        while self.pos < self.buf.len() && self.buf[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
        let start = self.pos;
        while self.pos < self.buf.len() && !self.buf[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
        if start == self.pos {
            return Err(ModelParseError::Parse("eof reading token".into()));
        }
        std::str::from_utf8(&self.buf[start..self.pos])
            .map(|s| s.to_string())
            .map_err(|_| ModelParseError::Parse("invalid utf8".into()))
    }

    /// Read the `@BIN@` marker then raw binary float data (for .bin/.bin.gz).
    fn floats(&mut self, n: usize) -> Result<Vec<f32>, ModelParseError> {
        if !self.binary {
            require(n <= self.remaining(), "truncated text weight array")?;
            // Text format: read floats as whitespace-delimited text.
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(self.read_f32()?);
            }
            Ok(v)
        } else {
            // Binary format: skip whitespace, read "@BIN@" marker, then raw floats.
            self.skip_to_bin_marker()?;
            let bytes = n
                .checked_mul(4)
                .ok_or_else(|| ModelParseError::Parse("weight array overflow".into()))?;
            require(bytes <= self.remaining(), "truncated binary weight array")?;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(self.read_f32_binary()?);
            }
            Ok(v)
        }
    }

    /// Read a single f32 as raw binary (little-endian).
    fn read_f32_binary(&mut self) -> Result<f32, ModelParseError> {
        if self.remaining() < 4 {
            return Err(ModelParseError::Parse("eof reading f32".into()));
        }
        let v = f32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        require(v.is_finite(), "nonfinite binary weight")?;
        Ok(v)
    }

    /// Skip to the `@BIN@` marker in the binary stream.
    fn skip_to_bin_marker(&mut self) -> Result<(), ModelParseError> {
        let mut chars_before_at = 0;
        while self.pos < self.buf.len() {
            if self.buf[self.pos] == b'@' {
                break;
            }
            if !self.buf[self.pos].is_ascii_whitespace() {
                return Err(ModelParseError::Parse(
                    "non-whitespace before @BIN@ marker".into(),
                ));
            }
            self.pos += 1;
            chars_before_at += 1;
            if chars_before_at > 100 {
                return Err(ModelParseError::Parse(
                    "could not find @BIN@ marker (too many chars)".into(),
                ));
            }
        }
        if self.remaining() < 4 {
            return Err(ModelParseError::Parse("eof before @BIN@".into()));
        }
        if &self.buf[self.pos..self.pos + 4] != b"@BIN" {
            return Err(ModelParseError::Parse(format!(
                "expected @BIN@, got {:?}",
                &self.buf[self.pos..self.pos + 4]
            )));
        }
        self.pos += 4;
        if self.remaining() < 1 || self.buf[self.pos] != b'@' {
            return Err(ModelParseError::Parse(
                "expected trailing @ after @BIN".into(),
            ));
        }
        self.pos += 1;
        Ok(())
    }
}

fn require(condition: bool, message: &str) -> Result<(), ModelParseError> {
    if condition {
        Ok(())
    } else {
        Err(ModelParseError::Parse(message.into()))
    }
}

fn weight_count(dims: &[i32]) -> Result<usize, ModelParseError> {
    dims.iter().try_fold(1usize, |size, &dim| {
        require(dim > 0, "layer dimensions must be positive")?;
        size.checked_mul(dim as usize)
            .ok_or_else(|| ModelParseError::Parse("layer dimensions overflow".into()))
    })
}

fn conv(p: &mut Parser) -> Result<ConvLayerDesc, ModelParseError> {
    let name = p.read_token()?;
    let conv_y_size = p.read_i32()?;
    let conv_x_size = p.read_i32()?;
    let in_channels = p.read_i32()?;
    let out_channels = p.read_i32()?;
    let dilation_y = p.read_i32()?;
    let dilation_x = p.read_i32()?;
    // Compute weight count from dims (not read from stream).
    let n = weight_count(&[conv_y_size, conv_x_size, in_channels, out_channels])?;
    require(
        conv_y_size % 2 == 1 && conv_x_size % 2 == 1,
        "convolution filters must be odd",
    )?;
    require(
        dilation_y > 0 && dilation_x > 0,
        "convolution dilation must be positive",
    )?;
    let source = p.floats(n)?;
    // Native file: Y,X,IC,OC. ModelDesc/CUDA/ONNX: OC,IC,Y,X.
    let (h, w, ic, oc) = (
        conv_y_size as usize,
        conv_x_size as usize,
        in_channels as usize,
        out_channels as usize,
    );
    let mut weights = vec![0.0; n];
    for y in 0..h {
        for x in 0..w {
            for i in 0..ic {
                for o in 0..oc {
                    weights[((o * ic + i) * h + y) * w + x] =
                        source[((y * w + x) * ic + i) * oc + o];
                }
            }
        }
    }
    Ok(ConvLayerDesc {
        name,
        conv_y_size,
        conv_x_size,
        in_channels,
        out_channels,
        dilation_y,
        dilation_x,
        weights,
    })
}

fn bn(p: &mut Parser) -> Result<BatchNormLayerDesc, ModelParseError> {
    let name = p.read_token()?;
    let num_channels = p.read_i32()?;
    let epsilon = p.read_f32()?;
    let has_scale = p.read_bool()?;
    let has_bias = p.read_bool()?;
    weight_count(&[num_channels])?;
    require(epsilon > 0.0, "batch norm epsilon must be positive")?;

    let mean = p.floats(num_channels as usize)?;
    let variance = p.floats(num_channels as usize)?;

    let scale = if has_scale {
        p.floats(num_channels as usize)?
    } else {
        vec![1.0f32; num_channels as usize]
    };
    let bias = if has_bias {
        p.floats(num_channels as usize)?
    } else {
        vec![0.0f32; num_channels as usize]
    };

    let mut merged_scale = Vec::with_capacity(num_channels as usize);
    let mut merged_bias = Vec::with_capacity(num_channels as usize);
    for i in 0..num_channels as usize {
        require(variance[i] + epsilon > 0.0, "invalid batch norm variance")?;
        let scale_value = scale[i] / (variance[i] + epsilon).sqrt();
        let bias_value = bias[i] - scale_value * mean[i];
        require(
            scale_value.is_finite() && bias_value.is_finite(),
            "nonfinite merged batch norm affine",
        )?;
        merged_scale.push(scale_value);
        merged_bias.push(bias_value);
    }
    Ok(BatchNormLayerDesc {
        name,
        num_channels,
        epsilon,
        has_scale,
        has_bias,
        mean,
        variance,
        scale,
        bias,
        merged_scale,
        merged_bias,
    })
}

fn act(p: &mut Parser, model_version: i32) -> Result<ActivationLayerDesc, ModelParseError> {
    let name = p.read_token()?;
    let activation = if model_version >= 11 {
        let kind = p.read_token()?;
        match kind.as_str() {
            "ACTIVATION_IDENTITY" => ACTIVATION_IDENTITY,
            "ACTIVATION_RELU" => ACTIVATION_RELU,
            "ACTIVATION_MISH" => ACTIVATION_MISH,
            "ACTIVATION_SILU" => ACTIVATION_SILU,
            _ => {
                return Err(ModelParseError::Parse(format!(
                    "unknown activation: {kind}"
                )));
            }
        }
    } else {
        ACTIVATION_RELU
    };
    Ok(ActivationLayerDesc { name, activation })
}

fn mm(p: &mut Parser) -> Result<MatMulLayerDesc, ModelParseError> {
    let name = p.read_token()?;
    let in_channels = p.read_i32()?;
    let out_channels = p.read_i32()?;
    let n = weight_count(&[in_channels, out_channels])?;
    let weights = p.floats(n)?;
    Ok(MatMulLayerDesc {
        name,
        in_channels,
        out_channels,
        weights,
    })
}

fn mb(p: &mut Parser) -> Result<MatBiasLayerDesc, ModelParseError> {
    let name = p.read_token()?;
    let num_channels = p.read_i32()?;
    weight_count(&[num_channels])?;
    let weights = p.floats(num_channels as usize)?;
    Ok(MatBiasLayerDesc {
        name,
        num_channels,
        weights,
    })
}

fn residual(p: &mut Parser, model_version: i32) -> Result<ResidualBlockDesc, ModelParseError> {
    Ok(ResidualBlockDesc {
        name: p.read_token()?,
        pre_bn: bn(p)?,
        pre_activation: act(p, model_version)?,
        regular_conv: conv(p)?,
        mid_bn: bn(p)?,
        mid_activation: act(p, model_version)?,
        final_conv: conv(p)?,
    })
}

const ORDINARY: i32 = 1;
const GLOBAL_POOLING: i32 = 2;
const BLOCK_KIND_NESTED_BOTTLENECK: i32 = 3;

fn block(p: &mut Parser, model_version: i32) -> Result<BlockDesc, ModelParseError> {
    // Block kind is stored as a string: "ordinary_block", "gpool_block",
    // "nested_bottleneck_block", "transformer_attention_block", "transformer_ffn_block".
    let kind_str = p.read_token()?;
    let kind = match kind_str.as_str() {
        "ordinary_block" => ORDINARY,
        "gpool_block" => GLOBAL_POOLING,
        "nested_bottleneck_block" => BLOCK_KIND_NESTED_BOTTLENECK,
        "transformer_attention_block" => TRANSFORMER_ATTENTION_BLOCK_KIND,
        "transformer_ffn_block" => TRANSFORMER_FFN_BLOCK_KIND,
        _ => {
            return Err(ModelParseError::Parse(format!(
                "unsupported block kind: {kind_str}"
            )));
        }
    };
    match kind {
        ORDINARY => Ok(BlockDesc::Ordinary(residual(p, model_version)?)),
        GLOBAL_POOLING => Ok(BlockDesc::GlobalPooling(GlobalPoolingResidualBlockDesc {
            name: p.read_token()?,
            model_version,
            pre_bn: bn(p)?,
            pre_activation: act(p, model_version)?,
            regular_conv: conv(p)?,
            gpool_conv: conv(p)?,
            gpool_bn: bn(p)?,
            gpool_activation: act(p, model_version)?,
            gpool_to_bias_mul: mm(p)?,
            mid_bn: bn(p)?,
            mid_activation: act(p, model_version)?,
            final_conv: conv(p)?,
        })),
        BLOCK_KIND_NESTED_BOTTLENECK => {
            let name = p.read_token()?;
            let num_blocks = p.read_i32()?;
            require(
                num_blocks > 0 && num_blocks as usize <= p.remaining(),
                "invalid nested block count",
            )?;
            let pre_bn = bn(p)?;
            let pre_activation = act(p, model_version)?;
            let pre_conv = conv(p)?;
            let mut blocks = Vec::new();
            for _ in 0..num_blocks {
                blocks.push(block(p, model_version)?);
            }
            let post_bn = bn(p)?;
            let post_activation = act(p, model_version)?;
            let post_conv = conv(p)?;
            Ok(BlockDesc::NestedBottleneck(Box::new(
                NestedBottleneckResidualBlockDesc {
                    name,
                    num_blocks,
                    pre_bn,
                    pre_activation,
                    pre_conv,
                    blocks,
                    post_bn,
                    post_activation,
                    post_conv,
                },
            )))
        }
        TRANSFORMER_ATTENTION_BLOCK_KIND => Ok(BlockDesc::TransformerAttention(attention(p)?)),
        TRANSFORMER_FFN_BLOCK_KIND => Ok(BlockDesc::TransformerFfn(ffn(p)?)),
        _ => Err(ModelParseError::Parse(format!(
            "unsupported block kind {kind}"
        ))),
    }
}

fn rms(p: &mut Parser) -> Result<RMSNormLayerDesc, ModelParseError> {
    let name = p.read_token()?;
    let num_channels = p.read_i32()?;
    let epsilon = p.read_f32()?;
    let spatial = p.read_bool()?;
    let cgroup_size = p.read_i32()?;
    weight_count(&[num_channels])?;
    require(epsilon > 0.0 && epsilon <= 1.0, "invalid RMSNorm epsilon")?;
    require(cgroup_size == 0, "grouped spatial RMSNorm is unsupported")?;
    let gamma = p.floats(num_channels as usize)?;
    let beta = p.floats(num_channels as usize)?;

    Ok(RMSNormLayerDesc {
        name,
        num_channels,
        epsilon,
        spatial,
        cgroup_size,
        gamma,
        beta,
    })
}

fn transformer_rms(p: &mut Parser) -> Result<TransformerRMSNormDesc, ModelParseError> {
    let name = p.read_token()?;
    let num_channels = p.read_i32()?;
    let epsilon = p.read_f32()?;
    weight_count(&[num_channels])?;
    require(
        epsilon > 0.0 && epsilon <= 1.0,
        "invalid transformer RMSNorm epsilon",
    )?;
    let weight = p.floats(num_channels as usize)?;
    Ok(TransformerRMSNormDesc {
        name,
        num_channels,
        epsilon,
        weight,
    })
}

fn attention(p: &mut Parser) -> Result<TransformerAttentionDesc, ModelParseError> {
    let name = p.read_token()?;
    let num_heads = p.read_i32()?;
    let num_kv_heads = p.read_i32()?;
    let q_head_dim = p.read_i32()?;
    let v_head_dim = p.read_i32()?;
    let use_rope = p.read_bool()?;
    let learnable_rope = p.read_bool()?;
    weight_count(&[num_heads, num_kv_heads, q_head_dim, v_head_dim])?;
    require(
        num_heads % num_kv_heads == 0,
        "attention heads must be divisible by KV heads",
    )?;
    require(
        !use_rope || q_head_dim % 2 == 0,
        "RoPE requires an even Q head dimension",
    )?;
    let pre_ln = transformer_rms(p)?;
    let q_proj = mm(p)?;
    let k_proj = mm(p)?;
    let v_proj = mm(p)?;
    let out_proj = mm(p)?;
    let channels = pre_ln.num_channels;
    require(
        q_proj.in_channels == channels
            && k_proj.in_channels == channels
            && v_proj.in_channels == channels
            && out_proj.out_channels == channels,
        "attention projection input/output channels do not match RMSNorm",
    )?;
    require(
        q_proj.out_channels as i64 == num_heads as i64 * q_head_dim as i64
            && k_proj.out_channels as i64 == num_kv_heads as i64 * q_head_dim as i64
            && v_proj.out_channels as i64 == num_kv_heads as i64 * v_head_dim as i64
            && out_proj.in_channels as i64 == num_heads as i64 * v_head_dim as i64,
        "attention projection dimensions do not match head counts",
    )?;
    let mut desc = TransformerAttentionDesc {
        name,
        num_heads,
        num_kv_heads,
        q_head_dim,
        v_head_dim,
        use_rope,
        learnable_rope,
        pre_ln,
        q_proj,
        k_proj,
        v_proj,
        out_proj,
        ..Default::default()
    };
    if use_rope {
        let _rope_name = p.read_token()?;
        if learnable_rope {
            desc.rope_num_kv_heads = p.read_i32()?;
            desc.rope_num_pairs = p.read_i32()?;
            let last_dim = p.read_i32()?;
            require(
                desc.rope_num_kv_heads == num_kv_heads
                    && desc.rope_num_pairs == q_head_dim / 2
                    && last_dim == 2,
                "learnable RoPE shape does not match attention",
            )?;
            desc.rope_freqs = p.floats(weight_count(&[num_kv_heads, q_head_dim / 2, 2])?)?;
        } else {
            desc.rope_theta = p.read_f32()?;
            require(desc.rope_theta > 0.0, "RoPE theta must be positive")?;
        }
    }
    Ok(desc)
}

fn ffn(p: &mut Parser) -> Result<TransformerFFNDesc, ModelParseError> {
    let name = p.read_token()?;
    let num_channels = p.read_i32()?;
    let ffn_channels = p.read_i32()?;
    let use_swi_glu = p.read_bool()?;
    weight_count(&[num_channels, ffn_channels])?;
    let pre_ln = transformer_rms(p)?;
    let linear1 = mm(p)?;
    let linear_gate = if use_swi_glu {
        mm(p)?
    } else {
        MatMulLayerDesc::default()
    };
    let linear2 = mm(p)?;
    require(
        pre_ln.num_channels == num_channels
            && linear1.in_channels == num_channels
            && linear1.out_channels == ffn_channels
            && linear2.in_channels == ffn_channels
            && linear2.out_channels == num_channels,
        "FFN layer channel mismatch",
    )?;
    require(
        !use_swi_glu
            || (linear_gate.in_channels == num_channels
                && linear_gate.out_channels == ffn_channels),
        "SwiGLU gate shape mismatch",
    )?;
    Ok(TransformerFFNDesc {
        name,
        num_channels,
        ffn_channels,
        use_swi_glu,
        pre_ln,
        linear1,
        linear_gate,
        linear2,
    })
}

fn trunk(
    p: &mut Parser,
    model_version: i32,
    meta_encoder_version: i32,
) -> Result<TrunkDesc, ModelParseError> {
    let name = p.read_token()?;
    // model_version is from outer model, not read from stream.
    let num_blocks = p.read_i32()?;
    let trunk_num_channels = p.read_i32()?;
    let mid_num_channels = p.read_i32()?;
    let regular_num_channels = p.read_i32()?;
    let _dilated_num_channels = p.read_i32()?; // unused
    let gpool_num_channels = p.read_i32()?;
    weight_count(&[
        trunk_num_channels,
        mid_num_channels,
        regular_num_channels,
        gpool_num_channels,
    ])?;
    require(
        num_blocks > 0 && num_blocks as usize <= p.remaining(),
        "invalid trunk block count",
    )?;
    // meta_encoder_version is from outer model, not read from stream.
    // trunk_norm_kind and 5 unused model options (modelVersion >= 15).
    let trunk_norm_kind = if model_version >= 15 {
        let v = p.read_i32()?;
        // 5 unused model options (must be 0).
        for _ in 0..5 {
            let unused = p.read_i32()?;
            if unused != 0 {
                return Err(ModelParseError::Parse(format!(
                    "unsupported trunk option {unused}"
                )));
            }
        }
        v
    } else {
        0
    };
    require(
        trunk_norm_kind == TRUNK_NORM_KIND_STANDARD || trunk_norm_kind == TRUNK_NORM_KIND_RMSNORM,
        "unsupported trunk normalization kind",
    )?;

    let initial_conv = conv(p)?;
    let initial_mat_mul = mm(p)?;

    let sgf_metadata_encoder = if meta_encoder_version > 0 {
        SGFMetadataEncoderDesc {
            name: p.read_token()?,
            meta_encoder_version,
            num_input_meta_channels: p.read_i32()?,
            mul1: mm(p)?,
            bias1: mb(p)?,
            act1: act(p, model_version)?,
            mul2: mm(p)?,
            bias2: mb(p)?,
            act2: act(p, model_version)?,
            mul3: mm(p)?,
        }
    } else {
        SGFMetadataEncoderDesc::default()
    };

    let mut blocks = Vec::new();
    for _ in 0..num_blocks {
        blocks.push(block(p, model_version)?);
    }

    let (trunk_tip_bn, trunk_tip_rms_norm) = if trunk_norm_kind == TRUNK_NORM_KIND_STANDARD {
        (bn(p)?, RMSNormLayerDesc::default())
    } else {
        (BatchNormLayerDesc::default(), rms(p)?)
    };
    let trunk_tip_activation = act(p, model_version)?;

    Ok(TrunkDesc {
        name,
        model_version,
        num_blocks,
        trunk_num_channels,
        mid_num_channels,
        regular_num_channels,
        gpool_num_channels,
        meta_encoder_version,
        trunk_norm_kind,
        initial_conv,
        initial_mat_mul,
        sgf_metadata_encoder,
        blocks,
        trunk_tip_bn,
        trunk_tip_rms_norm,
        trunk_tip_activation,
    })
}

fn policy_head(p: &mut Parser, model_version: i32) -> Result<PolicyHeadDesc, ModelParseError> {
    let name = p.read_token()?;
    let policy_out_channels = if model_version >= 17 {
        let channels = p.read_i32()?;
        require(
            channels == 2 || channels == 4,
            "invalid policy output channels",
        )?;
        channels
    } else if model_version == 16 {
        4
    } else if model_version >= 12 {
        2
    } else {
        1
    };
    if model_version >= 17 {
        p.reserved(3, "policy")?;
    }
    let p1_conv = conv(p)?;
    let g1_conv = conv(p)?;
    let g1_bn = bn(p)?;
    let g1_activation = act(p, model_version)?;
    let gpool_to_bias_mul = mm(p)?;
    let p1_bn = bn(p)?;
    let p1_activation = act(p, model_version)?;
    let p2_conv = conv(p)?;
    let gpool_to_pass_mul = mm(p)?;
    let (gpool_to_pass_bias, pass_activation, gpool_to_pass_mul2) = if model_version >= 15 {
        (mb(p)?, act(p, model_version)?, mm(p)?)
    } else {
        (
            MatBiasLayerDesc::default(),
            ActivationLayerDesc::default(),
            MatMulLayerDesc::default(),
        )
    };

    Ok(PolicyHeadDesc {
        name,
        model_version,
        policy_out_channels,
        p1_conv,
        g1_conv,
        g1_bn,
        g1_activation,
        gpool_to_bias_mul,
        p1_bn,
        p1_activation,
        p2_conv,
        gpool_to_pass_mul,
        gpool_to_pass_bias,
        pass_activation,
        gpool_to_pass_mul2,
    })
}

fn value_head(p: &mut Parser, model_version: i32) -> Result<ValueHeadDesc, ModelParseError> {
    let name = p.read_token()?;
    if model_version >= 17 {
        p.reserved(3, "value")?;
    }
    let v1_conv = conv(p)?;
    let v1_bn = bn(p)?;
    let v1_activation = act(p, model_version)?;
    let v2_mul = mm(p)?;
    let v2_bias = mb(p)?;
    let v2_activation = act(p, model_version)?;
    let v3_mul = mm(p)?;
    let v3_bias = mb(p)?;
    let sv3_mul = mm(p)?;
    let sv3_bias = mb(p)?;
    let v_ownership_conv = conv(p)?;
    Ok(ValueHeadDesc {
        name,
        model_version,
        v1_conv,
        v1_bn,
        v1_activation,
        v2_mul,
        v2_bias,
        v2_activation,
        v3_mul,
        v3_bias,
        sv3_mul,
        sv3_bias,
        v_ownership_conv,
    })
}

// ==========================================================================
// Public API
// ==========================================================================

fn same_channels(actual: i32, expected: i32, label: &str) -> Result<(), ModelParseError> {
    require(
        actual == expected,
        &format!("{label}: got {actual} channels, expected {expected}"),
    )
}

fn validate_blocks(blocks: &[BlockDesc], channels: i32) -> Result<(), ModelParseError> {
    for block in blocks {
        match block {
            BlockDesc::Ordinary(b) => {
                same_channels(b.pre_bn.num_channels, channels, &b.pre_bn.name)?;
                same_channels(b.regular_conv.in_channels, channels, &b.regular_conv.name)?;
                same_channels(
                    b.mid_bn.num_channels,
                    b.regular_conv.out_channels,
                    &b.mid_bn.name,
                )?;
                same_channels(
                    b.final_conv.in_channels,
                    b.mid_bn.num_channels,
                    &b.final_conv.name,
                )?;
                same_channels(b.final_conv.out_channels, channels, &b.final_conv.name)?;
            }
            BlockDesc::GlobalPooling(b) => {
                same_channels(b.pre_bn.num_channels, channels, &b.pre_bn.name)?;
                same_channels(b.regular_conv.in_channels, channels, &b.regular_conv.name)?;
                same_channels(b.gpool_conv.in_channels, channels, &b.gpool_conv.name)?;
                same_channels(
                    b.gpool_bn.num_channels,
                    b.gpool_conv.out_channels,
                    &b.gpool_bn.name,
                )?;
                same_channels(
                    b.gpool_to_bias_mul.in_channels,
                    3 * b.gpool_bn.num_channels,
                    &b.gpool_to_bias_mul.name,
                )?;
                same_channels(
                    b.gpool_to_bias_mul.out_channels,
                    b.regular_conv.out_channels,
                    &b.gpool_to_bias_mul.name,
                )?;
                same_channels(
                    b.mid_bn.num_channels,
                    b.regular_conv.out_channels,
                    &b.mid_bn.name,
                )?;
                same_channels(
                    b.final_conv.in_channels,
                    b.mid_bn.num_channels,
                    &b.final_conv.name,
                )?;
                same_channels(b.final_conv.out_channels, channels, &b.final_conv.name)?;
            }
            BlockDesc::NestedBottleneck(b) => {
                same_channels(b.pre_bn.num_channels, channels, &b.pre_bn.name)?;
                same_channels(b.pre_conv.in_channels, channels, &b.pre_conv.name)?;
                validate_blocks(&b.blocks, b.pre_conv.out_channels)?;
                same_channels(
                    b.post_bn.num_channels,
                    b.pre_conv.out_channels,
                    &b.post_bn.name,
                )?;
                same_channels(
                    b.post_conv.in_channels,
                    b.post_bn.num_channels,
                    &b.post_conv.name,
                )?;
                same_channels(b.post_conv.out_channels, channels, &b.post_conv.name)?;
            }
            BlockDesc::TransformerAttention(b) => {
                same_channels(b.pre_ln.num_channels, channels, &b.name)?
            }
            BlockDesc::TransformerFfn(b) => same_channels(b.num_channels, channels, &b.name)?,
        }
    }
    Ok(())
}

fn validate_model(model: &ModelDesc) -> Result<(), ModelParseError> {
    let t = &model.trunk;
    let channels = t.trunk_num_channels;
    same_channels(
        t.initial_conv.in_channels,
        model.num_input_channels,
        &t.initial_conv.name,
    )?;
    same_channels(t.initial_conv.out_channels, channels, &t.initial_conv.name)?;
    same_channels(
        t.initial_mat_mul.in_channels,
        model.num_input_global_channels,
        &t.initial_mat_mul.name,
    )?;
    same_channels(
        t.initial_mat_mul.out_channels,
        channels,
        &t.initial_mat_mul.name,
    )?;
    if t.meta_encoder_version > 0 {
        let e = &t.sgf_metadata_encoder;
        same_channels(
            e.num_input_meta_channels,
            model.num_input_meta_channels,
            &e.name,
        )?;
        same_channels(
            e.mul1.in_channels,
            model.num_input_meta_channels,
            &e.mul1.name,
        )?;
        same_channels(e.bias1.num_channels, e.mul1.out_channels, &e.bias1.name)?;
        same_channels(e.mul2.in_channels, e.mul1.out_channels, &e.mul2.name)?;
        same_channels(e.bias2.num_channels, e.mul2.out_channels, &e.bias2.name)?;
        same_channels(e.mul3.in_channels, e.mul2.out_channels, &e.mul3.name)?;
        same_channels(e.mul3.out_channels, channels, &e.mul3.name)?;
    }
    validate_blocks(&t.blocks, channels)?;
    if t.trunk_norm_kind == TRUNK_NORM_KIND_STANDARD {
        same_channels(t.trunk_tip_bn.num_channels, channels, &t.trunk_tip_bn.name)?;
    } else {
        same_channels(
            t.trunk_tip_rms_norm.num_channels,
            channels,
            &t.trunk_tip_rms_norm.name,
        )?;
    }
    let p = &model.policy_head;
    same_channels(p.p1_conv.in_channels, channels, &p.p1_conv.name)?;
    same_channels(p.g1_conv.in_channels, channels, &p.g1_conv.name)?;
    same_channels(p.p1_bn.num_channels, p.p1_conv.out_channels, &p.p1_bn.name)?;
    same_channels(p.g1_bn.num_channels, p.g1_conv.out_channels, &p.g1_bn.name)?;
    same_channels(
        p.gpool_to_bias_mul.in_channels,
        p.g1_bn.num_channels * 3,
        &p.gpool_to_bias_mul.name,
    )?;
    same_channels(
        p.gpool_to_bias_mul.out_channels,
        p.p1_bn.num_channels,
        &p.gpool_to_bias_mul.name,
    )?;
    same_channels(p.p2_conv.in_channels, p.p1_bn.num_channels, &p.p2_conv.name)?;
    same_channels(
        p.p2_conv.out_channels,
        p.policy_out_channels,
        &p.p2_conv.name,
    )?;
    same_channels(
        p.gpool_to_pass_mul.in_channels,
        p.g1_bn.num_channels * 3,
        &p.gpool_to_pass_mul.name,
    )?;
    if model.model_version >= 15 {
        same_channels(
            p.gpool_to_pass_mul.out_channels,
            p.p1_conv.out_channels,
            &p.gpool_to_pass_mul.name,
        )?;
        same_channels(
            p.gpool_to_pass_bias.num_channels,
            p.gpool_to_pass_mul.out_channels,
            &p.gpool_to_pass_bias.name,
        )?;
        same_channels(
            p.gpool_to_pass_mul2.in_channels,
            p.gpool_to_pass_mul.out_channels,
            &p.gpool_to_pass_mul2.name,
        )?;
        same_channels(
            p.gpool_to_pass_mul2.out_channels,
            p.policy_out_channels,
            &p.gpool_to_pass_mul2.name,
        )?;
    } else {
        same_channels(
            p.gpool_to_pass_mul.out_channels,
            p.policy_out_channels,
            &p.gpool_to_pass_mul.name,
        )?;
    }
    let v = &model.value_head;
    same_channels(v.v1_conv.in_channels, channels, &v.v1_conv.name)?;
    same_channels(v.v1_bn.num_channels, v.v1_conv.out_channels, &v.v1_bn.name)?;
    same_channels(
        v.v2_mul.in_channels,
        3 * v.v1_bn.num_channels,
        &v.v2_mul.name,
    )?;
    same_channels(
        v.v2_bias.num_channels,
        v.v2_mul.out_channels,
        &v.v2_bias.name,
    )?;
    same_channels(v.v3_mul.in_channels, v.v2_mul.out_channels, &v.v3_mul.name)?;
    same_channels(v.v3_mul.out_channels, 3, &v.v3_mul.name)?;
    same_channels(v.v3_bias.num_channels, 3, &v.v3_bias.name)?;
    let score_channels = if model.model_version >= 9 {
        6
    } else if model.model_version >= 8 {
        4
    } else if model.model_version >= 4 {
        2
    } else {
        1
    };
    same_channels(
        v.sv3_mul.in_channels,
        v.v2_mul.out_channels,
        &v.sv3_mul.name,
    )?;
    same_channels(v.sv3_mul.out_channels, score_channels, &v.sv3_mul.name)?;
    same_channels(v.sv3_bias.num_channels, score_channels, &v.sv3_bias.name)?;
    same_channels(
        v.v_ownership_conv.in_channels,
        v.v1_conv.out_channels,
        &v.v_ownership_conv.name,
    )?;
    same_channels(v.v_ownership_conv.out_channels, 1, &v.v_ownership_conv.name)?;
    Ok(())
}

pub fn parse_model(
    data: &[u8],
    sha256: &str,
    binary_floats: bool,
) -> Result<ModelDesc, ModelParseError> {
    let mut p = Parser::new(data, binary_floats);

    let name = p.read_token()?;
    let model_version = p.read_i32()?;

    if name.len() > 96 {
        return Err(ModelParseError::Parse("model name too long".into()));
    }
    for c in name.chars() {
        if !c.is_alphanumeric() && c != '_' && c != '-' {
            return Err(ModelParseError::Parse(format!(
                "invalid char in model name: {c}"
            )));
        }
    }
    if !(3..=crate::version::LATEST_MODEL_VERSION_IMPLEMENTED).contains(&model_version) {
        return Err(ModelParseError::Parse(format!(
            "unsupported model version {model_version}"
        )));
    }

    let num_input_channels = p.read_i32()?;
    let num_input_global_channels = p.read_i32()?;
    weight_count(&[num_input_channels, num_input_global_channels])?;

    let post_process_params = if model_version >= 13 {
        ModelPostProcessParams {
            td_score_multiplier: p.read_f64()?,
            score_mean_multiplier: p.read_f64()?,
            score_stdev_multiplier: p.read_f64()?,
            lead_multiplier: p.read_f64()?,
            variance_time_multiplier: p.read_f64()?,
            shortterm_value_error_multiplier: p.read_f64()?,
            shortterm_score_error_multiplier: p.read_f64()?,
            output_scale_multiplier: 1.0,
        }
    } else {
        ModelPostProcessParams::default()
    };
    require(
        [
            post_process_params.td_score_multiplier,
            post_process_params.score_mean_multiplier,
            post_process_params.score_stdev_multiplier,
            post_process_params.lead_multiplier,
            post_process_params.variance_time_multiplier,
            post_process_params.shortterm_value_error_multiplier,
            post_process_params.shortterm_score_error_multiplier,
        ]
        .iter()
        .all(|v| *v > 0.0),
        "postprocessing multipliers must be positive",
    )?;

    let (
        meta_encoder_version,
        prefer_pass_alive_under_suicide_rules,
        prefer_exclude_territory_adjacent_to_atari,
    ) = if model_version >= 15 {
        let v = p.read_i32()?;
        if !(0..=1).contains(&v) {
            return Err(ModelParseError::Parse(format!(
                "unsupported metaEncoderVersion {v}"
            )));
        }
        let prefer_pass = p.read_bool()?;
        let prefer_atari = p.read_bool()?;
        p.reserved(5, "model")?;
        (v, prefer_pass, prefer_atari)
    } else {
        (0, false, false)
    };

    let num_input_meta_channels = crate::version::get_num_input_meta_channels(meta_encoder_version)
        .map_err(ModelParseError::Parse)?;

    let trunk = trunk(&mut p, model_version, meta_encoder_version)?;
    let policy_head = policy_head(&mut p, model_version)?;
    let value_head = value_head(&mut p, model_version)?;
    require(
        p.buf[p.pos..].iter().all(u8::is_ascii_whitespace),
        "unexpected trailing model data",
    )?;

    let desc = ModelDesc {
        name,
        sha256: sha256.into(),
        model_version,
        num_input_channels,
        num_input_global_channels,
        num_input_meta_channels,
        num_policy_channels: policy_head.policy_out_channels,
        num_value_channels: value_head.v3_mul.out_channels,
        num_score_value_channels: value_head.sv3_mul.out_channels,
        num_ownership_channels: value_head.v_ownership_conv.out_channels,
        meta_encoder_version,
        prefer_pass_alive_under_suicide_rules,
        prefer_exclude_territory_adjacent_to_atari,
        post_process_params,
        trunk,
        policy_head,
        value_head,
    };
    validate_model(&desc)?;
    Ok(desc)
}

pub fn load_model_file(path: &str) -> Result<ModelDesc, ModelParseError> {
    let mut file = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    let lower = path.to_lowercase();
    let (binary, compressed) = if lower.ends_with(".bin.gz") {
        (true, true)
    } else if lower.ends_with(".txt.gz") || lower.ends_with(".gz") {
        (false, true)
    } else if lower.ends_with(".bin") {
        (true, false)
    } else if lower.ends_with(".txt") {
        (false, false)
    } else {
        return Err(ModelParseError::Parse(format!(
            "unsupported model file extension: {path}"
        )));
    };

    load_model_from_bytes(&buf, binary, compressed)
}

/// Parse the same native file bytes that the caller authenticated. The returned
/// SHA256 always identifies these original bytes, including gzip compression.
/// This API never opens a file or substitutes a neighboring ONNX model.
pub fn load_model_from_bytes(
    bytes: &[u8],
    binary: bool,
    compressed: bool,
) -> Result<ModelDesc, ModelParseError> {
    let sha256 = hex::encode(Sha256::digest(bytes));
    if compressed {
        let mut decoder = GzDecoder::new(bytes);
        let mut data = Vec::new();
        decoder.read_to_end(&mut data)?;
        require(
            decoder.into_inner().is_empty(),
            "unexpected bytes after gzip model",
        )?;
        parse_model(&data, &sha256, binary)
    } else {
        parse_model(bytes, &sha256, binary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A complete tiny native model, with each version's original on-disk
    /// headers and head widths. Text and binary variants carry identical weights.
    struct Fixture {
        data: Vec<u8>,
        binary: bool,
        version: i32,
    }

    impl Fixture {
        fn line(&mut self, value: impl std::fmt::Display) {
            writeln!(self.data, "{value}").unwrap();
        }

        fn weights(&mut self, values: &[f32]) {
            if self.binary {
                self.data.extend_from_slice(b"@BIN@");
                for value in values {
                    self.data.extend_from_slice(&value.to_le_bytes());
                }
                self.data.push(b'\n');
            } else {
                for value in values {
                    self.line(value);
                }
            }
        }

        fn conv(&mut self, name: &str, input: i32, output: i32) {
            self.line(format!("{name}\n1\n1\n{input}\n{output}\n1\n1"));
            self.weights(&vec![0.125; (input * output) as usize]);
        }

        fn mm(&mut self, name: &str, input: i32, output: i32) {
            self.line(format!("{name}\n{input}\n{output}"));
            self.weights(&vec![0.25; (input * output) as usize]);
        }

        fn bias(&mut self, name: &str, channels: i32) {
            self.line(format!("{name}\n{channels}"));
            self.weights(&vec![0.0; channels as usize]);
        }

        fn bn(&mut self, name: &str) {
            self.line(format!("{name}\n1\n1\n0\n0"));
            self.weights(&[0.0]);
            self.weights(&[0.0]);
        }

        fn act(&mut self) {
            self.line("activation");
            if self.version >= 11 {
                self.line("ACTIVATION_MISH");
            }
        }

        fn model(version: i32, binary: bool) -> Vec<u8> {
            let mut f = Self {
                data: Vec::new(),
                binary,
                version,
            };
            f.line(format!("tiny-native\n{version}\n22\n19"));
            if version >= 13 {
                f.line("20\n20\n20\n20\n40\n0.25\n150");
            }
            // Two live preference slots are true, the remaining slots reserved.
            if version >= 15 {
                f.line("0\n1\n1\n0\n0\n0\n0\n0");
            }
            f.line("trunk\n1\n1\n1\n1\n1\n1");
            if version >= 15 {
                f.line("0\n0\n0\n0\n0\n0");
            }
            f.conv("initial", 22, 1);
            f.mm("global", 19, 1);
            f.line("ordinary_block\nresidual");
            f.bn("pre");
            f.act();
            f.conv("regular", 1, 1);
            f.bn("mid");
            f.act();
            f.conv("final", 1, 1);
            f.bn("tip");
            f.act();
            f.line("policy");
            let policy_channels = if version >= 16 {
                4
            } else if version >= 12 {
                2
            } else {
                1
            };
            if version >= 17 {
                f.line(format!("{policy_channels}\n0\n0\n0"));
            }
            f.conv("p1", 1, 1);
            f.conv("g1", 1, 1);
            f.bn("g1bn");
            f.act();
            f.mm("gpool_bias", 3, 1);
            f.bn("p1bn");
            f.act();
            f.conv("p2", 1, policy_channels);
            if version >= 15 {
                f.mm("pass1", 3, 1);
                f.bias("passbias", 1);
                f.act();
                f.mm("pass2", 1, policy_channels);
            } else {
                f.mm("pass", 3, policy_channels);
            }
            f.line("value");
            if version >= 17 {
                f.line("0\n0\n0");
            }
            f.conv("v1", 1, 1);
            f.bn("v1bn");
            f.act();
            f.mm("v2", 3, 1);
            f.bias("v2bias", 1);
            f.act();
            f.mm("v3", 1, 3);
            f.bias("v3bias", 3);
            let score_channels = if version >= 9 {
                6
            } else if version >= 8 {
                4
            } else if version >= 4 {
                2
            } else {
                1
            };
            f.mm("score", 1, score_channels);
            f.bias("scorebias", score_channels);
            f.conv("owner", 1, 1);
            f.data
        }
    }

    #[test]
    fn native_versions_preserve_headers_weights_preferences_and_real_sha256() {
        for version in 3..=17 {
            for binary in [false, true] {
                let bytes = Fixture::model(version, binary);
                let model = load_model_from_bytes(&bytes, binary, false).unwrap();
                assert_eq!(model.model_version, version);
                assert_eq!(model.sha256, hex::encode(Sha256::digest(&bytes)));
                assert_eq!(model.prefer_pass_alive_under_suicide_rules, version >= 15);
                assert_eq!(
                    model.prefer_exclude_territory_adjacent_to_atari,
                    version >= 15
                );
                assert_eq!(
                    model.trunk.trunk_tip_activation.activation,
                    if version >= 11 {
                        ACTIVATION_MISH
                    } else {
                        ACTIVATION_RELU
                    }
                );
                assert_eq!(model.trunk.initial_conv.weights, vec![0.125; 22]);
                assert_eq!(model.trunk.trunk_tip_bn.merged_scale, [1.0]);
            }
        }
    }

    #[test]
    fn compressed_identity_is_sha256_of_original_gzip_bytes() {
        let bytes = Fixture::model(17, true);
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&bytes).unwrap();
        let gzip = encoder.finish().unwrap();
        let model = load_model_from_bytes(&gzip, true, true).unwrap();
        assert_eq!(model.sha256, hex::encode(Sha256::digest(&gzip)));
        assert_ne!(model.sha256, hex::encode(Sha256::digest(&bytes)));
        let mut appended = gzip;
        appended.extend_from_slice(b"trailing compressed content");
        assert!(load_model_from_bytes(&appended, true, true).is_err());
    }

    #[test]
    fn convolution_file_order_is_transposed_into_descriptor_order() {
        let mut f = Fixture {
            data: Vec::new(),
            binary: true,
            version: 17,
        };
        f.line("conv\n1\n3\n2\n2\n1\n1");
        f.weights(&(0..12).map(|i| i as f32).collect::<Vec<_>>());
        let parsed = conv(&mut Parser::new(&f.data, true)).unwrap();
        assert_eq!(
            parsed.weights,
            [0.0, 4.0, 8.0, 2.0, 6.0, 10.0, 1.0, 5.0, 9.0, 3.0, 7.0, 11.0]
        );
    }

    #[test]
    fn rejects_incomplete_nonfinite_wrong_shape_and_trailing_data() {
        let valid = Fixture::model(17, true);
        let mut trailing = valid.clone();
        trailing.extend_from_slice(b"extra-layer");
        assert!(load_model_from_bytes(&trailing, true, false).is_err());
        assert!(load_model_from_bytes(&valid[..valid.len() - 3], true, false).is_err());
        let marker = valid.windows(5).position(|v| v == b"@BIN@").unwrap();
        let mut nonfinite = valid.clone();
        nonfinite[marker + 5..marker + 9].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(load_model_from_bytes(&nonfinite, true, false).is_err());
        let text = String::from_utf8(Fixture::model(17, false)).unwrap();
        let wrong_channels = text.replacen("initial\n1\n1\n22\n1", "initial\n1\n1\n21\n1", 1);
        assert!(load_model_from_bytes(wrong_channels.as_bytes(), false, false).is_err());
        assert!(conv(&mut Parser::new(b"bad\n-1\n3\n1\n1\n1\n1\n", true)).is_err());
        assert!(act(&mut Parser::new(b"act\nACTIVATION_UNKNOWN\n", false), 17).is_err());
    }

    #[test]
    fn rmsnorm_layout_and_silu_activation_are_distinct_from_mish() {
        let mut f = Fixture {
            data: Vec::new(),
            binary: true,
            version: 17,
        };
        f.line("rms\n2\n0.001\n1\n0");
        f.weights(&[1.0, 2.0]);
        f.weights(&[3.0, 4.0]);
        let parsed = rms(&mut Parser::new(&f.data, true)).unwrap();
        assert!(parsed.spatial);
        assert_eq!(parsed.gamma, [1.0, 2.0]);
        assert_eq!(parsed.beta, [3.0, 4.0]);
        assert_eq!(
            act(&mut Parser::new(b"act\nACTIVATION_SILU\n", false), 17)
                .unwrap()
                .activation,
            ACTIVATION_SILU
        );
        assert_eq!(
            act(&mut Parser::new(b"act\nACTIVATION_MISH\n", false), 17)
                .unwrap()
                .activation,
            ACTIVATION_MISH
        );
    }
}
