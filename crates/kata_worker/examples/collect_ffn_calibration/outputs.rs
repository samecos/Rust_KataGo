//! Chunk full raw heads and exact production Worker PBs; no copied NN math.
use super::{
    artifacts::{Source, hash, write_json, write_new},
    inputs::LoadedInput,
};
use anyhow::{Result, ensure};
use kata_nn::output_postprocess::RawHeads;
use kata_worker::{evaluator::output_postprocessing::NativeModelOutputContract, wire};
use prost::Message;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

pub const HEADS: [(&str, usize); 5] = [
    ("policy", 2172),
    ("value", 3),
    ("misc", 10),
    ("moremisc", 8),
    ("ownership", 361),
];
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PbBits {
    pub policy_f32: Vec<u32>,
    pub ownership_f32: Vec<u32>,
    pub scalars_f64: [u64; 9],
    pub has_shortterm_error: bool,
}
impl PbBits {
    pub fn from_output(o: &wire::NnOutput) -> Result<Self> {
        let scalars = [
            o.white_win_prob,
            o.white_loss_prob,
            o.white_no_result_prob,
            o.white_score_mean,
            o.white_score_mean_sq,
            o.white_lead,
            o.var_time_left,
            o.shortterm_winloss_error,
            o.shortterm_score_error,
        ];
        ensure!(
            o.policy.len() == 362 && [0, 361].contains(&o.ownership.len()),
            "PB shape mismatch"
        );
        ensure!(
            o.policy.iter().chain(&o.ownership).all(|x| x.is_finite())
                && scalars.iter().all(|x| x.is_finite()),
            "nonfinite PB"
        );
        Ok(Self {
            policy_f32: o.policy.iter().map(|x| x.to_bits()).collect(),
            ownership_f32: o.ownership.iter().map(|x| x.to_bits()).collect(),
            scalars_f64: scalars.map(f64::to_bits),
            has_shortterm_error: o.has_shortterm_error,
        })
    }
}
pub fn validate_raw(raw: RawHeads<'_>, batch: usize) -> Result<()> {
    ensure!((1..=8).contains(&batch), "raw batch outside 1..8");
    for ((name, width), head) in
        HEADS
            .iter()
            .zip([raw.policy, raw.value, raw.misc, raw.moremisc, raw.ownership])
    {
        ensure!(
            head.len() == batch * width && head.iter().all(|v| v.is_finite()),
            "wrong/nonfinite complete raw head {name}"
        );
    }
    Ok(())
}
fn append_blob(blob: &mut Vec<u8>, bytes: &[u8]) -> Value {
    let offset = blob.len();
    blob.extend_from_slice(bytes);
    json!({"offset":offset,"bytes":bytes.len(),"sha256":hash(bytes)})
}

pub struct Chunks {
    root: PathBuf,
    limit: usize,
    raw: Vec<u8>,
    pb: Vec<u8>,
    entries: Vec<Value>,
    pub committed: Vec<Source>,
    pub artifacts: Vec<Source>,
    pub rows: usize,
    pub inputs: usize,
}
impl Chunks {
    pub fn new(root: &Path, limit: usize) -> Result<Self> {
        ensure!((1..=128).contains(&limit), "invalid chunk limit");
        Ok(Self {
            root: root.into(),
            limit,
            raw: Vec::new(),
            pb: Vec::new(),
            entries: Vec::new(),
            committed: Vec::new(),
            artifacts: Vec::new(),
            rows: 0,
            inputs: 0,
        })
    }
    pub fn append(
        &mut self,
        input_index: usize,
        input: &LoadedInput,
        raw: RawHeads<'_>,
        contract: &NativeModelOutputContract,
    ) -> Result<()> {
        let batch = input.descriptor.physical_batch;
        ensure!(input.rows.len() == batch, "raw logical row count differs");
        validate_raw(raw, batch)?;
        let mut head_slices = Vec::new();
        let mut row_bytes: Vec<Vec<u8>> =
            (0..batch).map(|_| Vec::with_capacity(2554 * 4)).collect();
        for ((name, width), head) in
            HEADS
                .iter()
                .zip([raw.policy, raw.value, raw.misc, raw.moremisc, raw.ownership])
        {
            let bytes: Vec<_> = head.iter().flat_map(|v| v.to_le_bytes()).collect();
            let slice = append_blob(&mut self.raw, &bytes);
            head_slices.push(json!({"head":name,"row_width":width,"slice":slice}));
            for (row, bytes) in row_bytes.iter_mut().enumerate() {
                bytes.extend(
                    head[row * width..(row + 1) * width]
                        .iter()
                        .flat_map(|v| v.to_le_bytes()),
                );
            }
        }
        let mut rows = Vec::new();
        for (row_index, row) in input.rows.iter().enumerate() {
            let prepared = contract
                .prepare_request_pb(
                    &row.request_pb,
                    &row.descriptor.pb_sha256,
                    &row.descriptor.row_feature_sha256,
                )
                .map_err(|e| anyhow::anyhow!("{}: {}", e.code, e.message))?;
            let output = contract
                .postprocess_raw_row(&prepared, raw, batch, row_index)
                .map_err(|e| anyhow::anyhow!("{}: {}", e.code, e.message))?;
            let bits = PbBits::from_output(&output)?;
            let bytes = output.encode_to_vec();
            // Proto3 elides default scalar values, including -0.0. Preserve
            // both the shared API's bits and the exact wire-decoded bits.
            let wire_bits = PbBits::from_output(&wire::NnOutput::decode(bytes.as_slice())?)?;
            let slice = append_blob(&mut self.pb, &bytes);
            rows.push(json!({"row":row_index,"identity":row.descriptor,"raw_row_sha256":hash(&row_bytes[row_index]),"pb":slice,
                "bits_stage":"shared-output-before-protobuf-default-elision","bits":bits,"wire_bits":wire_bits}));
        }
        self.rows += batch;
        self.inputs += 1;
        self.entries.push(json!({"input_index":input_index,"input":input.descriptor,"tensor_sha256":input.tensor_sha256,
            "heads":head_slices,"rows":rows}));
        if self.entries.len() == self.limit {
            self.flush()?;
        }
        Ok(())
    }
    pub fn flush(&mut self) -> Result<()> {
        if self.entries.is_empty() {
            return Ok(());
        }
        let index = self.committed.len();
        let raw = write_new(
            &self.root,
            &format!("chunk-{index:04}-raw.f32le"),
            &self.raw,
        )?;
        let pb = write_new(&self.root, &format!("chunk-{index:04}-pb.bin"), &self.pb)?;
        let metadata = write_json(
            &self.root,
            &format!("chunk-{index:04}.json"),
            &json!({
            "schema":"rustgo-ffn-calibration-output-chunk-v1","raw":raw,"pb":pb,"inputs":self.entries}),
        )?;
        self.artifacts.extend([raw, pb, metadata.clone()]);
        self.committed.push(metadata);
        self.raw.clear();
        self.pb.clear();
        self.entries.clear();
        Ok(())
    }
    pub fn recheck(&self) -> Result<()> {
        ensure!(self.entries.is_empty(), "unflushed outputs");
        for source in &self.artifacts {
            source.recheck()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scalar_bits_do_not_narrow_to_f32_and_preserve_signed_zero() {
        let mut o = wire::NnOutput {
            policy: vec![0.0; 362],
            ..Default::default()
        };
        o.white_score_mean = 1.0 + f64::EPSILON;
        o.white_lead = -0.0;
        let b = PbBits::from_output(&o).unwrap();
        assert_ne!(
            b.scalars_f64[3],
            (o.white_score_mean as f32 as f64).to_bits()
        );
        assert_eq!(b.scalars_f64[5], (-0.0f64).to_bits());
        let decoded = wire::NnOutput::decode(o.encode_to_vec().as_slice()).unwrap();
        let wire_bits = PbBits::from_output(&decoded).unwrap();
        assert_eq!(wire_bits.scalars_f64[3], b.scalars_f64[3]);
        assert_eq!(wire_bits.scalars_f64[5], 0.0f64.to_bits());
        assert_eq!(decoded.white_lead, o.white_lead);
        o.shortterm_score_error = f64::NAN;
        assert!(PbBits::from_output(&o).is_err());
    }
    #[test]
    fn complete_head_check_rejects_nonselected_row_and_unused_channel_nan() {
        let mut heads: [Vec<f32>; 5] = std::array::from_fn(|i| vec![0.; HEADS[i].1 * 3]);
        let raw = |h: &[Vec<f32>; 5]| {
            validate_raw(
                RawHeads {
                    policy: &h[0],
                    value: &h[1],
                    misc: &h[2],
                    moremisc: &h[3],
                    ownership: &h[4],
                },
                3,
            )
        };
        assert!(raw(&heads).is_ok());
        heads[0][2172 + 1000] = f32::NAN;
        assert!(raw(&heads).is_err());
        heads[0][2172 + 1000] = 0.;
        heads[3].pop();
        assert!(raw(&heads).is_err());
    }
}
