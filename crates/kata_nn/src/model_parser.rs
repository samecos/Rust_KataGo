//! Model file parser for KataGo `.bin.gz` / `.txt.gz` / `.txt` model files.

use std::io::Read;

use flate2::read::GzDecoder;

use kata_core::global::StringError;

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

struct Parser {
    buf: Vec<u8>,
    pos: usize,
    binary: bool,
}

impl Parser {
    fn new(buf: Vec<u8>, binary: bool) -> Self {
        Self { buf, pos: 0, binary }
    }

    /// Read a float32 as text (all scalar fields are text even in .bin files).
    fn read_f32(&mut self) -> Result<f32, ModelParseError> {
        let t = self.read_token()?;
        t.parse::<f32>()
            .map_err(|_| ModelParseError::Parse(format!("bad f32: {t}")))
    }

    /// Read an i32 as text (all scalar fields are text even in .bin files).
    fn read_i32(&mut self) -> Result<i32, ModelParseError> {
        let t = self.read_token()?;
        t.parse::<i32>().map_err(|_| {
            eprintln!(
                "PARSE ERROR: bad i32: {t} (position {}, remaining {}, last 100 bytes: {})",
                self.pos,
                self.remaining(),
                String::from_utf8_lossy(&self.buf[self.pos.saturating_sub(100)..self.pos.min(self.buf.len())])
            );
            ModelParseError::Parse(format!("bad i32: {t}"))
        })
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
            // Text format: read floats as whitespace-delimited text.
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(self.read_f32()?);
            }
            Ok(v)
        } else {
            // Binary format: skip whitespace, read "@BIN@" marker, then raw floats.
            self.skip_to_bin_marker()?;
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
        Ok(v)
    }

    /// Skip to the `@BIN@` marker in the binary stream.
    fn skip_to_bin_marker(&mut self) -> Result<(), ModelParseError> {
        let mut chars_before_at = 0;
        while self.pos < self.buf.len() {
            if self.buf[self.pos] == b'@' {
                break;
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
            return Err(ModelParseError::Parse("expected trailing @ after @BIN".into()));
        }
        self.pos += 1;
        Ok(())
    }
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
    let n = (conv_y_size * conv_x_size * in_channels * out_channels) as usize;
    let weights = p.floats(n)?;
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
    let has_scale = p.read_i32()? != 0;
    let has_bias = p.read_i32()? != 0;

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
        merged_scale: Vec::new(),
        merged_bias: Vec::new(),
    })
}

fn act(p: &mut Parser, model_version: i32) -> Result<ActivationLayerDesc, ModelParseError> {
    let name = p.read_token()?;
    let activation = if model_version >= 11 {
        let kind = p.read_token()?;
        match kind.as_str() {
            "ACTIVATION_IDENTITY" => 0,
            "ACTIVATION_RELU" => 1,
            "ACTIVATION_MISH" => 3,
            "ACTIVATION_SILU" => 3,
            _ => {
                return Err(ModelParseError::Parse(format!(
                    "unknown activation: {kind}"
                )))
            }
        }
    } else {
        1 // ACTIVATION_RELU for older versions
    };
    eprintln!("act: name={name}, kind={activation}");
    Ok(ActivationLayerDesc { name, activation })
}

fn mm(p: &mut Parser) -> Result<MatMulLayerDesc, ModelParseError> {
    let name = p.read_token()?;
    let in_channels = p.read_i32()?;
    let out_channels = p.read_i32()?;
    let n = (in_channels * out_channels) as usize;
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
        _ => {
            return Err(ModelParseError::Parse(format!(
                "unsupported block kind: {kind_str}"
            )))
        }
    };
    match kind {
        ORDINARY => Ok(BlockDesc::Ordinary(residual(p, model_version)?)),
        GLOBAL_POOLING => Ok(BlockDesc::GlobalPooling(GlobalPoolingResidualBlockDesc {
            name: p.read_token()?,
            model_version: 0,
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
        _ => Err(ModelParseError::Parse(format!(
            "unsupported block kind {kind}"
        ))),
    }
}

fn rms(p: &mut Parser) -> Result<RMSNormLayerDesc, ModelParseError> {
    let name = p.read_token()?;
    let _kind = p.read_i32()?;
    let num_channels = p.read_i32()?;
    let epsilon = p.read_f32()?;
    let has_scale = p.read_i32()? != 0;
    let has_bias = p.read_i32()? != 0;

    let gamma = if has_scale {
        p.floats(num_channels as usize)?
    } else {
        vec![1.0f32; num_channels as usize]
    };
    let beta = if has_bias {
        p.floats(num_channels as usize)?
    } else {
        vec![0.0f32; num_channels as usize]
    };

    Ok(RMSNormLayerDesc {
        name,
        num_channels,
        epsilon,
        spatial: false,
        cgroup_size: 0,
        gamma,
        beta,
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

    let trunk_tip_bn = bn(p)?;
    let trunk_tip_rms_norm = rms(p)?;
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

fn policy_head(p: &mut Parser) -> Result<PolicyHeadDesc, ModelParseError> {
    let name = p.read_token()?;
    let model_version = p.read_i32()?;
    let policy_out_channels = p.read_i32()?;
    let p1_conv = conv(p)?;
    let g1_conv = conv(p)?;
    let g1_bn = bn(p)?;
    let g1_activation = act(p, model_version)?;
    let gpool_to_bias_mul = mm(p)?;
    let p1_bn = bn(p)?;
    let p1_activation = act(p, model_version)?;
    let p2_conv = conv(p)?;
    let gpool_to_pass_mul = mm(p)?;
    let gpool_to_pass_bias = mb(p)?;
    let pass_activation = act(p, model_version)?;
    let gpool_to_pass_mul2 = if model_version >= 15 {
        mm(p)?
    } else {
        MatMulLayerDesc::default()
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

fn value_head(p: &mut Parser) -> Result<ValueHeadDesc, ModelParseError> {
    let name = p.read_token()?;
    let model_version = p.read_i32()?;
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

pub fn parse_model(
    data: &[u8],
    sha256: &str,
    binary_floats: bool,
) -> Result<ModelDesc, ModelParseError> {
    let mut p = Parser::new(data.to_vec(), binary_floats);

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
    if !(3..=15).contains(&model_version) {
        return Err(ModelParseError::Parse(format!(
            "unsupported model version {model_version}"
        )));
    }

    let num_input_channels = p.read_i32()?;
    let num_input_global_channels = p.read_i32()?;

    let post_process_params = if model_version >= 13 {
        ModelPostProcessParams {
            td_score_multiplier: p.read_f32()? as f64,
            score_mean_multiplier: p.read_f32()? as f64,
            score_stdev_multiplier: p.read_f32()? as f64,
            lead_multiplier: p.read_f32()? as f64,
            variance_time_multiplier: p.read_f32()? as f64,
            shortterm_value_error_multiplier: p.read_f32()? as f64,
            shortterm_score_error_multiplier: p.read_f32()? as f64,
            output_scale_multiplier: 1.0,
        }
    } else {
        ModelPostProcessParams::default()
    };

    let meta_encoder_version = if model_version >= 15 {
        let v = p.read_i32()?;
        if v > 1 {
            return Err(ModelParseError::Parse(format!(
                "unsupported metaEncoderVersion {v}"
            )));
        }
        for _ in 0..7 {
            let unused = p.read_i32()?;
            if unused != 0 {
                return Err(ModelParseError::Parse(format!(
                    "unsupported model option {unused}"
                )));
            }
        }
        v
    } else {
        0
    };

    let num_input_meta_channels = if meta_encoder_version > 0 { 4 } else { 0 };

    let trunk = trunk(&mut p, model_version, meta_encoder_version)?;
    let policy_head = policy_head(&mut p)?;
    let value_head = value_head(&mut p)?;

    Ok(ModelDesc {
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
        post_process_params,
        trunk,
        policy_head,
        value_head,
    })
}

pub fn load_model_file(path: &str) -> Result<ModelDesc, ModelParseError> {
    let mut file = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    let lower = path.to_lowercase();
    let sha256 = format!("{:016x}", compute_sha256(&buf));
    let (binary_floats, data) = if lower.ends_with(".bin.gz") {
        let mut d = GzDecoder::new(&buf[..]);
        let mut out = Vec::new();
        d.read_to_end(&mut out)?;
        (true, out)
    } else if lower.ends_with(".txt.gz") || lower.ends_with(".gz") {
        let mut d = GzDecoder::new(&buf[..]);
        let mut out = Vec::new();
        d.read_to_end(&mut out)?;
        (false, out)
    } else if lower.ends_with(".bin") {
        (true, buf)
    } else if lower.ends_with(".txt") {
        (false, buf)
    } else {
        return Err(ModelParseError::Parse(format!(
            "unsupported model file extension: {path}"
        )));
    };

    parse_model(&data, &sha256, binary_floats)
}

fn compute_sha256(data: &[u8]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    data.hash(&mut h);
    h.finish()
}
