//! Portable, explicitly versioned weights for the research Go student networks.
//! The magic identifies this format independently of the filename extension.

use crate::desc::{ModelDesc, ModelPostProcessParams};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const MAGIC: &[u8; 16] = b"RustGoStudentV1\n";
pub const MAX_FILE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudentManifest {
    pub format_version: u32,
    pub input_version: u32,
    pub variant: String,
    pub width: usize,
    pub blocks: usize,
    pub score_scale: f64,
    pub tensors: Vec<TensorSpec>,
}

#[derive(Debug, Clone)]
pub struct StudentTensor {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct StudentModel {
    pub manifest: StudentManifest,
    pub tensors: BTreeMap<String, StudentTensor>,
}

pub fn is_student_model(bytes: &[u8]) -> bool {
    bytes.starts_with(MAGIC)
}

pub fn expected_shapes(width: usize, blocks: usize) -> BTreeMap<String, Vec<usize>> {
    let mut shapes = BTreeMap::new();
    let mut add = |name: &str, shape: Vec<usize>, out: usize| {
        shapes.insert(format!("{name}.weight"), shape);
        shapes.insert(format!("{name}.bias"), vec![out]);
    };
    add("stem", vec![width, 22, 3, 3], width);
    add("global_to_stem", vec![width, 19], width);
    for i in 0..blocks {
        for conv in ["conv1", "conv2"] {
            add(
                &format!("blocks.{i}.{conv}"),
                vec![width, width, 3, 3],
                width,
            );
        }
    }
    add("policy_board", vec![1, width, 1, 1], 1);
    add("policy_pass", vec![1, width + 19], 1);
    add("value_features", vec![64, width + 19], 64);
    add("value", vec![3, 64], 3);
    add("score_head", vec![1, 64], 1);
    add("ownership", vec![1, width, 1, 1], 1);
    shapes
}

impl StudentModel {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        if !is_student_model(bytes) || bytes.len() < 20 || bytes.len() > MAX_FILE_BYTES {
            return Err("invalid or oversized RustGo student model".into());
        }
        let header_len = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
        if header_len == 0 || header_len > 65536 || header_len > bytes.len() - 20 {
            return Err("invalid student manifest length".into());
        }
        let manifest: StudentManifest = serde_json::from_slice(&bytes[20..20 + header_len])
            .map_err(|e| format!("student manifest: {e}"))?;
        let spec = match manifest.variant.as_str() {
            "compact" => (48, 4),
            "dense" => (64, 6),
            "large" => (96, 10),
            _ => return Err("unsupported student variant".into()),
        };
        if manifest.format_version != 1
            || manifest.input_version != 7
            || (manifest.width, manifest.blocks) != spec
            || manifest.score_scale != 20.0
        {
            return Err("unsupported student architecture or input/score contract".into());
        }
        let mut expected = expected_shapes(manifest.width, manifest.blocks);
        if manifest.tensors.len() != expected.len() {
            return Err("student tensor count mismatch".into());
        }
        let payload = &bytes[20 + header_len..];
        let mut offset = 0usize;
        let mut tensors = BTreeMap::new();
        for tensor in &manifest.tensors {
            let shape = expected
                .remove(&tensor.name)
                .ok_or_else(|| format!("unknown or duplicate student tensor {}", tensor.name))?;
            let count: usize = shape.iter().product();
            if tensor.shape != shape || tensor.length != count || tensor.offset != offset {
                return Err(format!("student tensor layout mismatch: {}", tensor.name));
            }
            let end = offset
                .checked_add(count * 4)
                .ok_or("student tensor size overflow")?;
            let raw = payload.get(offset..end).ok_or("truncated student tensor")?;
            let values: Vec<f32> = raw
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            if values.iter().any(|v| !v.is_finite()) {
                return Err(format!("nonfinite student tensor {}", tensor.name));
            }
            tensors.insert(tensor.name.clone(), StudentTensor { shape, values });
            offset = end;
        }
        if !expected.is_empty() || offset != payload.len() {
            return Err("student payload has missing tensors or trailing bytes".into());
        }
        Ok(Self { manifest, tensors })
    }

    pub fn model_desc(&self, sha256: String) -> ModelDesc {
        // v8 supplies the same v7 features and marks the untrained uncertainty
        // and time heads unavailable (-1) in the existing postprocessor.
        ModelDesc {
            name: format!("rustgo-student-v1-{}", self.manifest.variant),
            sha256,
            model_version: 8,
            num_input_channels: 22,
            num_input_global_channels: 19,
            num_policy_channels: 1,
            num_value_channels: 3,
            num_score_value_channels: 2,
            num_ownership_channels: 1,
            post_process_params: ModelPostProcessParams {
                score_mean_multiplier: self.manifest.score_scale,
                score_stdev_multiplier: 0.0,
                lead_multiplier: self.manifest.score_scale,
                ..ModelPostProcessParams::default()
            },
            ..ModelDesc::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Vec<u8> {
        fixture_for("compact", 48, 4)
    }
    fn fixture_for(variant: &str, width: usize, blocks: usize) -> Vec<u8> {
        let mut offset = 0;
        let tensors = expected_shapes(width, blocks)
            .into_iter()
            .map(|(name, shape)| {
                let length = shape.iter().product::<usize>();
                let spec = TensorSpec {
                    name,
                    shape,
                    length,
                    offset,
                };
                offset += length * 4;
                spec
            })
            .collect();
        let m = StudentManifest {
            format_version: 1,
            input_version: 7,
            variant: variant.into(),
            width,
            blocks,
            score_scale: 20.0,
            tensors,
        };
        let header = serde_json::to_vec(&m).unwrap();
        let mut out = MAGIC.to_vec();
        out.extend_from_slice(&(header.len() as u32).to_le_bytes());
        out.extend(header);
        out.resize(out.len() + offset, 0);
        out
    }
    #[test]
    fn validates_large_and_rejects_mislabeled_dimensions() {
        let bytes = fixture_for("large", 96, 10);
        let parsed = StudentModel::parse(&bytes).unwrap();
        assert_eq!(parsed.tensors.len(), 56);
        assert_eq!(parsed.tensors["blocks.9.conv2.weight"].shape, vec![96,96,3,3]);
        assert!(StudentModel::parse(&fixture_for("large", 64, 6)).is_err());
        assert!(StudentModel::parse(&fixture_for("dense", 96, 10)).is_err());
    }
    #[test]
    fn validates_complete_model_and_descriptor() {
        let model = StudentModel::parse(&fixture()).unwrap();
        let desc = model.model_desc("hash".into());
        assert_eq!(
            crate::version::get_inputs_version(desc.model_version).unwrap(),
            7
        );
        assert_eq!(desc.post_process_params.score_stdev_multiplier, 0.0);
        assert_eq!(model.tensors.len(), 32);
    }
    #[test]
    fn rejects_truncated_trailing_and_nonfinite_payload() {
        let mut bytes = fixture();
        assert!(StudentModel::parse(&bytes[..bytes.len() - 1]).is_err());
        bytes.extend([0]);
        assert!(StudentModel::parse(&bytes).is_err());
        bytes.pop();
        let start = 20 + u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
        bytes[start..start + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(StudentModel::parse(&bytes).is_err());
    }
    #[test]
    fn rejects_architecture_and_duplicate_tensors_before_upload() {
        let bytes = fixture();
        let len = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
        let mut m: StudentManifest = serde_json::from_slice(&bytes[20..20 + len]).unwrap();
        let payload = &bytes[20 + len..];
        let encode = |m: &StudentManifest| {
            let h = serde_json::to_vec(m).unwrap();
            let mut b = MAGIC.to_vec();
            b.extend_from_slice(&(h.len() as u32).to_le_bytes());
            b.extend(h);
            b.extend(payload);
            b
        };
        m.width = 64;
        assert!(StudentModel::parse(&encode(&m)).is_err());
        m.width = 48;
        m.tensors[1].name = m.tensors[0].name.clone();
        assert!(StudentModel::parse(&encode(&m)).is_err());
    }
}
