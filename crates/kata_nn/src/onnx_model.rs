//! Minimal parser for KataGo-exported ONNX models (e.g. `b11fix.onnx`).
//!
//! KataGo's exported models carry custom metadata (`modelVersion`, `name`,
//! `num_spatial_inputs`, `num_global_inputs`, `pos_len`, ...) that is enough
//! to reconstruct the [`ModelDesc`] the engine needs, without parsing the
//! full computation graph. The I/O tensor specs are used by the generic
//! TensorRT shim path.

use prost::Message;

use crate::desc::ModelDesc;
use crate::onnx_proto::{self, type_proto, ModelProto, ValueInfoProto};

/// Specification of a single I/O tensor.
#[derive(Debug, Clone)]
pub struct OnnxTensorSpec {
    pub name: String,
    /// Declared dims; a symbolic batch dim is represented as `-1`.
    pub dims: Vec<i64>,
    /// ONNX `TensorProto.DataType` element type (1 = FLOAT).
    pub elem_type: i32,
}

impl OnnxTensorSpec {
    /// Number of elements for one batch element.
    pub fn single_elts(&self) -> i64 {
        self.dims.iter().skip(1).filter(|&&d| d >= 0).product()
    }
}

/// Parsed metadata of an exported KataGo ONNX model.
#[derive(Debug, Clone)]
pub struct OnnxModelInfo {
    pub model_desc: ModelDesc,
    pub inputs: Vec<OnnxTensorSpec>,
    pub outputs: Vec<OnnxTensorSpec>,
}

fn parse_spec(vi: &ValueInfoProto) -> OnnxTensorSpec {
    let name = vi.name.clone().unwrap_or_default();
    let mut dims = Vec::new();
    let mut elem_type = 0i32;
    if let Some(tp) = &vi.r#type {
        if let Some(type_proto::Value::TensorType(tensor)) = &tp.value {
            elem_type = tensor.elem_type.unwrap_or(0);
            if let Some(shape) = &tensor.shape {
                dims = shape
                    .dim
                    .iter()
                    .map(|d| match &d.value {
                        Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(x)) => *x,
                        _ => -1, // symbolic dim (batch) or unset
                    })
                    .collect();
            }
        }
    }
    OnnxTensorSpec {
        name,
        dims,
        elem_type,
    }
}

/// Parse the model metadata and I/O specs from serialized ONNX bytes.
pub fn parse_onnx_model(bytes: &[u8]) -> Result<OnnxModelInfo, String> {
    let model = ModelProto::decode(bytes).map_err(|e| format!("ONNX protobuf decode failed: {e}"))?;

    let mut name = String::new();
    let mut model_version: i32 = -1;
    let mut num_spatial: i32 = -1;
    let mut num_global: i32 = -1;
    for e in &model.metadata_props {
        let key = e.key.as_deref().unwrap_or("");
        let value = e.value.as_deref().unwrap_or("");
        match key {
            "name" => name = value.to_string(),
            "modelVersion" => model_version = value.parse().unwrap_or(-1),
            "num_spatial_inputs" => num_spatial = value.parse().unwrap_or(-1),
            "num_global_inputs" => num_global = value.parse().unwrap_or(-1),
            _ => {}
        }
    }

    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| "ONNX model has no graph".to_string())?;
    let inputs: Vec<OnnxTensorSpec> = graph.input.iter().map(parse_spec).collect();
    let outputs: Vec<OnnxTensorSpec> = graph.output.iter().map(parse_spec).collect();

    // Locate the spatial / global inputs by role: the 4-D input is spatial,
    // the 2-D input is global.
    let (spatial, global): (Option<&OnnxTensorSpec>, Option<&OnnxTensorSpec>) = (
        inputs.iter().find(|t| t.dims.len() >= 4),
        inputs.iter().find(|t| t.dims.len() == 2),
    );
    let nn_x_len = spatial.and_then(|t| t.dims.last().copied()).unwrap_or(-1) as i32;
    let nn_y_len = spatial
        .and_then(|t| t.dims.get(t.dims.len().saturating_sub(2)).copied())
        .unwrap_or(-1) as i32;
    let num_input_channels = spatial
        .and_then(|t| t.dims.get(t.dims.len().saturating_sub(3)).copied())
        .map(|d| if d > 0 { d } else { num_spatial as i64 })
        .unwrap_or(num_spatial as i64) as i32;
    let num_input_global_channels = global
        .and_then(|t| t.dims.get(1).copied())
        .map(|d| if d > 0 { d } else { num_global as i64 })
        .unwrap_or(num_global as i64) as i32;

    if model_version < 0 {
        return Err(format!(
            "ONNX model metadata lacks a valid modelVersion (got {model_version})"
        ));
    }
    // Only the standard 19x19 board is supported.
    if nn_x_len != 19 || nn_y_len != 19 {
        return Err(format!(
            "ONNX model spatial size must be 19x19 (got {nn_x_len}x{nn_y_len})"
        ));
    }

    // Engine-side channel semantics: the exported 6-channel policy head is
    // consumed as regular + short-term-optimistic logits (2 channels), and
    // the score/value head as misc[0:4] + moremisc[0:2] (6 channels).
    let model_desc = ModelDesc {
        name: if name.is_empty() {
            "unknown".to_string()
        } else {
            name
        },
        model_version,
        num_input_channels,
        num_input_global_channels,
        num_input_meta_channels: 0,
        num_policy_channels: 2,
        num_value_channels: 3,
        num_score_value_channels: 6,
        num_ownership_channels: 1,
        ..ModelDesc::default()
    };

    Ok(OnnxModelInfo {
        model_desc,
        inputs,
        outputs,
    })
}
