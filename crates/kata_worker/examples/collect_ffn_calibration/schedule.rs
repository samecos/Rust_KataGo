//! An explicit per-recipe call list; no inferred budget, sampling, retries or tuning.
use super::{
    artifacts::{Source, hash, valid_sha},
    inputs::InputDescriptor,
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostBatch {
    pub physical_batch: usize,
    pub input_indices: Vec<usize>,
    pub warmup: usize,
    pub measurements: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub schema: String,
    pub mode: String,
    pub model: Source,
    pub binary: bool,
    pub compressed: bool,
    pub graph_sha256: String,
    pub proposal: Source,
    pub proposal_object_sha256: String,
    pub recipe: Source,
    pub resolved_recipe_sha256: String,
    pub group_ids: Vec<String>,
    pub provenance: Source,
    pub numeric_input_indices: Vec<usize>,
    pub cost: Vec<CostBatch>,
    pub expected_forwards: usize,
    pub expected_numeric_rows: usize,
    pub chunk_inputs: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedCall {
    pub input_index: usize,
    pub phase: String,
    pub iteration: usize,
    pub physical_batch: usize,
    pub tensor_sha256: String,
}

impl Plan {
    pub fn validate(
        &self,
        inputs: &[InputDescriptor],
        coverage: &str,
        selected: usize,
    ) -> Result<Vec<ExpectedCall>> {
        ensure!(
            self.schema == "rustgo-ffn-calibration-collection-plan-v1",
            "unknown collection schema"
        );
        ensure!(
            ["complete-calibration", "bounded-diagnostic"].contains(&self.mode.as_str()),
            "unknown collection mode"
        );
        ensure!(
            valid_sha(&self.graph_sha256)
                && valid_sha(&self.resolved_recipe_sha256)
                && valid_sha(&self.proposal_object_sha256),
            "invalid identity SHA"
        );
        ensure!(
            (1..=128).contains(&self.chunk_inputs),
            "chunk limit outside 1..128"
        );
        ensure!(
            !self.numeric_input_indices.is_empty() && self.numeric_input_indices.len() <= 2044,
            "numeric input list outside 1..2044"
        );
        let mut seen = BTreeSet::new();
        let mut calls = Vec::new();
        let mut rows = 0usize;
        for &index in &self.numeric_input_indices {
            ensure!(seen.insert(index), "duplicate numeric input");
            let input = inputs
                .get(index)
                .context("numeric input index outside corpus")?;
            rows += input.physical_batch;
            calls.push(call(index, input, "numeric", 0));
        }
        if self.mode == "complete-calibration" {
            ensure!(
                coverage == "complete-calibration" && selected == 2044,
                "complete collection requires full corpus"
            );
            let exact: Vec<_> = inputs
                .iter()
                .enumerate()
                .filter(|(_, i)| i.target_batch == 1 && i.physical_batch == 1)
                .map(|(n, _)| n)
                .collect();
            ensure!(
                exact.len() == 2044 && self.numeric_input_indices == exact,
                "full numeric stage requires exact ordered B1 coverage"
            );
        }
        ensure!(
            rows == self.expected_numeric_rows,
            "numeric row budget mismatch"
        );
        let mut batches = BTreeSet::new();
        for cost in &self.cost {
            ensure!(
                [1, 3, 8].contains(&cost.physical_batch) && batches.insert(cost.physical_batch),
                "duplicate/unsupported cost batch"
            );
            ensure!(
                (1..=32).contains(&cost.warmup)
                    && (1..=64).contains(&cost.measurements)
                    && (1..=128).contains(&cost.input_indices.len()),
                "invalid cost W/M/sample count"
            );
            let mut indices = BTreeSet::new();
            for &index in &cost.input_indices {
                ensure!(indices.insert(index), "duplicate cost input");
                let input = inputs
                    .get(index)
                    .context("cost input index outside corpus")?;
                ensure!(
                    input.target_batch == cost.physical_batch
                        && input.physical_batch == cost.physical_batch,
                    "tail/other packing cannot stand in for complete physical cost batch"
                );
            }
            let first = cost.input_indices[0];
            for iteration in 0..cost.warmup {
                calls.push(call(first, &inputs[first], "cost-warmup", iteration));
            }
            for &index in &cost.input_indices {
                calls.push(call(index, &inputs[index], "cost-before", 0));
                for iteration in 0..cost.measurements {
                    calls.push(call(index, &inputs[index], "cost-measure", iteration));
                }
                calls.push(call(index, &inputs[index], "cost-after", 0));
            }
        }
        ensure!(
            calls.len() == self.expected_forwards,
            "explicit forward budget differs from call list"
        );
        Ok(calls)
    }
    pub fn verify_proposal(&self, recipe_bytes: &[u8]) -> Result<()> {
        let raw = self.proposal.read(16 * 1024 * 1024)?;
        let mut proposal: Value = serde_json::from_slice(&raw)?;
        ensure!(
            proposal["schema"] == "rustgo-single-ffn-calibration-recipe-proposal-v1"
                && proposal["model_sha256"] == self.model.sha256
                && proposal["graph_sha256"] == self.graph_sha256,
            "proposal model/graph/schema differs"
        );
        let binding = &proposal["input_binding"];
        ensure!(
            binding["schema"] == "rustgo-ffn-ablation-input-binding-v1"
                && binding["model_sha256"] == self.model.sha256
                && binding["object_encoding"] == "json-utf8-compact-sorted-object-v1"
                && binding["calibration"]["split"] == "calibration"
                && binding["calibration"]["request_count"] == 2044
                && binding["calibration"]["corpus_manifest_sha256"]
                    == "c7b80205d26d5dd7c7cf38dceb8b88bcc8b3b452b764f7a83113d156fc88c842"
                && binding["calibration"]["request_file_sha256"]
                    == "c3d3ee4b4f75fadc613e343beaf60c60d95879bb89e6d516cfa5b4278650238f"
                && binding["calibration"]["request_set_sha256"]
                    == "13c90ca4c3d3783c6c183c23415ae7f6c2ba23dfd800f8d50d39a4efcadd78a9"
                && binding["calibration"]["unique_games"] == 32,
            "proposal calibration identity differs"
        );
        let declared = proposal
            .as_object_mut()
            .context("proposal must be object")?
            .remove("proposal_sha256")
            .context("missing proposal object SHA")?;
        ensure!(
            declared == self.proposal_object_sha256
                && hash(&serde_json::to_vec(&proposal)?) == self.proposal_object_sha256,
            "proposal object SHA differs"
        );
        let candidates = proposal["recipes"]
            .as_array()
            .context("proposal has no recipes")?;
        let matches: Vec<_> = candidates
            .iter()
            .filter(|c| c["recipe_sha256"] == self.resolved_recipe_sha256)
            .collect();
        ensure!(matches.len() == 1, "proposal recipe missing/duplicated");
        let candidate = matches[0];
        let recipe: Value = serde_json::from_slice(recipe_bytes)?;
        ensure!(
            candidate["recipe"] == recipe
                && candidate["group_ids"] == serde_json::to_value(&self.group_ids)?,
            "proposal recipe/group differs"
        );
        ensure!(
            self.group_ids.len() <= 1,
            "only reference or single-FFN ablation accepted"
        );
        ensure!(
            candidate["role"]
                == if self.group_ids.is_empty() {
                    "FP16_REFERENCE"
                } else {
                    "SINGLE_FFN_INT8_ABLATION"
                },
            "proposal role differs"
        );
        let expected: BTreeSet<_> = self
            .group_ids
            .iter()
            .flat_map(|id| [format!("{id}.dual"), format!("{id}.down")])
            .collect();
        let mut actual = BTreeSet::new();
        for p in recipe["projections"]
            .as_array()
            .context("missing projections")?
        {
            let precision = p["precision"].as_str().context("missing precision")?;
            ensure!(
                ["fp16", "int8"].contains(&precision),
                "unsupported ablation precision"
            );
            if precision == "int8" {
                ensure!(
                    actual.insert(p["id"].as_str().context("projection ID")?.to_owned()),
                    "duplicate INT8 projection"
                );
            }
        }
        ensure!(
            actual == expected,
            "ablation must quantize exactly both target FFN projections"
        );
        Ok(())
    }
}
fn call(index: usize, input: &InputDescriptor, phase: &str, iteration: usize) -> ExpectedCall {
    ExpectedCall {
        input_index: index,
        phase: phase.into(),
        iteration,
        physical_batch: input.physical_batch,
        tensor_sha256: input.tensor_sha256.clone(),
    }
}

#[cfg(test)]
#[path = "schedule_tests.rs"]
mod tests;
