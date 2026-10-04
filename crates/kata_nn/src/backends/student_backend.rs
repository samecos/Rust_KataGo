//! Engine adapter for student weights, selected by the existing CUDA backend.

use super::student_cuda::{StudentCudaModel, StudentOutputs};
use crate::backend::*;
use crate::desc::ModelDesc;
use crate::student_model::StudentModel;
use kata_core::{config::Config, logger::Logger};
use kata_game::symmetry::{copy_inputs_with_symmetry, copy_outputs_with_symmetry};
use std::{
    any::Any,
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

pub struct StudentLoadedModel {
    desc: ModelDesc,
    parsed: Arc<StudentModel>,
    profile: Mutex<Option<String>>,
}
impl LoadedModel for StudentLoadedModel {
    fn model_desc(&self) -> &ModelDesc {
        &self.desc
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn execution_profile_id(&self) -> Option<String> {
        self.profile.lock().ok()?.clone()
    }
}

pub fn load(bytes: &[u8], sha: String) -> Result<Box<dyn LoadedModel>, NeuralNetError> {
    let parsed = StudentModel::parse(bytes).map_err(NeuralNetError)?;
    Ok(Box::new(StudentLoadedModel {
        desc: parsed.model_desc(sha),
        parsed: Arc::new(parsed),
        profile: Mutex::new(None),
    }))
}

pub struct StudentContext {
    models: BTreeMap<usize, Arc<StudentCudaModel>>,
    sha: String,
    max_batch: i32,
}
impl ComputeContext for StudentContext {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub fn context(
    model: &StudentLoadedModel,
    gpu_idxs: &[i32],
    logger: &Logger,
    x: i32,
    y: i32,
    cfg: &Config,
) -> Result<Box<dyn ComputeContext>, NeuralNetError> {
    if (x, y) != (19, 19) {
        return Err(NeuralNetError("student CUDA requires 19x19".into()));
    }
    for key in [
        "cudaTacticPlan",
        "cudaQuantPlan",
        "cudaQuantExpectedProfile",
        "cudaInt8Scope",
        "cudaInt8MinFfnWidth",
    ] {
        if cfg.contains(key) {
            return Err(NeuralNetError(format!(
                "student CUDA cannot use TF3 option {key}"
            )));
        }
    }
    let max_batch = if cfg.contains("nnMaxBatchSize") {
        cfg.get_int("nnMaxBatchSize", 1, 65536)
            .map_err(|e| NeuralNetError(e.to_string()))?
    } else {
        16
    };
    if cfg.contains("rustgoResolvedInputsUseNHWC")
        && cfg
            .get_bool("rustgoResolvedInputsUseNHWC")
            .map_err(|e| NeuralNetError(e.to_string()))?
    {
        return Err(NeuralNetError("student CUDA requires NCHW inputs".into()));
    }
    let server_threads = if cfg.contains("numNNServerThreadsPerModel") {
        cfg.get_int("numNNServerThreadsPerModel", 1, 65536)
            .map_err(|e| NeuralNetError(e.to_string()))?
    } else {
        1
    };
    let exact_len = if cfg.contains("rustgoResolvedRequireExactNNLen") {
        cfg.get_bool("rustgoResolvedRequireExactNNLen")
            .map_err(|e| NeuralNetError(e.to_string()))?
    } else {
        true
    };
    let mut models = BTreeMap::new();
    let indices: Vec<usize> = if gpu_idxs.is_empty() {
        vec![0]
    } else {
        gpu_idxs
            .iter()
            .map(|v| if *v < 0 { 0 } else { *v as usize })
            .collect()
    };
    let mut facts = Vec::new();
    for idx in indices {
        if models.contains_key(&idx) {
            continue;
        }
        let uploaded =
            Arc::new(StudentCudaModel::load(&model.parsed, idx).map_err(NeuralNetError)?);
        facts.push(uploaded.execution_facts().map_err(NeuralNetError)?);
        models.insert(idx, uploaded);
    }
    use sha2::{Digest, Sha256};
    let exe = std::env::current_exe().map_err(|e| NeuralNetError(e.to_string()))?;
    let exe_sha = crate::tactic_plan::sha256_file(&exe).map_err(NeuralNetError)?;
    let contract = serde_json::json!({"format":"rustgo-student-execution-v1",
        "model_sha256":model.desc.sha256,"executable_sha256":exe_sha,
        "backend_build":super::cuda::backend_build_fingerprint(),"devices":facts,
        "max_batch":max_batch,"server_threads":server_threads,"require_exact_nn_len":exact_len,
        "layout":"NCHW","input_version":7,"score_scale":20.0,
        "adapter":"v8-no-uncertainty-zero-score-variance-v1"});
    let profile = format!(
        "rustgo-student-v1:{}",
        hex::encode(Sha256::digest(
            serde_json::to_vec(&contract).map_err(|e| NeuralNetError(e.to_string()))?
        ))
    );
    let mut state = model
        .profile
        .lock()
        .map_err(|_| NeuralNetError("student profile lock poisoned".into()))?;
    if state.as_ref().is_some_and(|v| v != &profile) {
        return Err(NeuralNetError(
            "student execution configuration changed; reload model".into(),
        ));
    }
    *state = Some(profile.clone());
    logger.write(&format!(
        "[cuda-student] variant={} precision=FP32 input=v7 execution-profile={}\n",
        model.parsed.manifest.variant, profile
    ));
    Ok(Box::new(StudentContext {
        models,
        sha: model.desc.sha256.clone(),
        max_batch,
    }))
}

struct PendingBatch {
    raw: StudentOutputs,
    rows: Vec<(i32, bool, u64)>,
}
#[derive(Default)]
struct Pending {
    next: usize,
    outputs: BTreeMap<usize, PendingBatch>,
}
pub struct StudentHandle {
    model: Arc<StudentCudaModel>,
    max_batch: i32,
    pending: Mutex<Pending>,
}
impl ComputeHandle for StudentHandle {
    fn is_using_fp16(&self) -> bool {
        false
    }
    fn set_is_warmup(&mut self, _: bool) -> bool {
        false
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
pub fn handle(
    ctx: &StudentContext,
    loaded: &dyn LoadedModel,
    max_batch: i32,
    nhwc: bool,
    gpu_idx: i32,
) -> Result<Box<dyn ComputeHandle>, NeuralNetError> {
    if nhwc
        || max_batch != ctx.max_batch
        || loaded.model_desc().sha256 != ctx.sha
        || !loaded.as_any().is::<StudentLoadedModel>()
    {
        return Err(NeuralNetError(
            "student handle must match loaded model, batch capacity and NCHW layout".into(),
        ));
    }
    let idx = if gpu_idx < 0 { 0 } else { gpu_idx as usize };
    let model = ctx
        .models
        .get(&idx)
        .ok_or_else(|| NeuralNetError("student GPU not in compute context".into()))?;
    Ok(Box::new(StudentHandle {
        model: model.clone(),
        max_batch,
        pending: Mutex::new(Pending::default()),
    }))
}

fn evaluate(
    h: &StudentHandle,
    n: i32,
    inputs: &[&mut NNResultBuf],
) -> Result<StudentOutputs, NeuralNetError> {
    if n < 1 || n > h.max_batch || inputs.len() != n as usize {
        return Err(NeuralNetError("invalid student batch size".into()));
    }
    let mut spatial = vec![0f32; n as usize * 22 * 361];
    let mut global = vec![0f32; n as usize * 19];
    for (i, input) in inputs.iter().enumerate() {
        if input.row_spatial_buf.len() != 22 * 361
            || input.row_global_buf.len() != 19
            || !(0..=7).contains(&input.symmetry)
            || input
                .row_spatial_buf
                .iter()
                .chain(input.row_global_buf.iter())
                .any(|v| !v.is_finite())
        {
            return Err(NeuralNetError(
                "invalid student input features or symmetry".into(),
            ));
        }
        copy_inputs_with_symmetry(
            &input.row_spatial_buf,
            &mut spatial[i * 22 * 361..(i + 1) * 22 * 361],
            1,
            19,
            19,
            22,
            false,
            input.symmetry,
        );
        global[i * 19..(i + 1) * 19].copy_from_slice(&input.row_global_buf);
    }
    h.model
        .apply(&spatial, &global, n as usize)
        .map_err(NeuralNetError)
}

pub fn decode(
    raw: &StudentOutputs,
    n: i32,
    inputs: &[&mut NNResultBuf],
    outputs: &mut [&mut NNOutput],
) -> Result<(), NeuralNetError> {
    if n < 1 || inputs.len() != n as usize || outputs.len() != n as usize {
        return Err(NeuralNetError("invalid student output row count".into()));
    }
    for (data, size) in [
        (&raw.policy_logits, 362),
        (&raw.value_logits, 3),
        (&raw.score, 1),
        (&raw.ownership_logits, 361),
    ] {
        if data.len() != n as usize * size || data.iter().any(|v| !v.is_finite()) {
            return Err(NeuralNetError(
                "invalid student output shape or nonfinite value".into(),
            ));
        }
    }
    if inputs
        .iter()
        .any(|r| !(0..=7).contains(&r.symmetry) || !r.policy_optimism.is_finite())
    {
        return Err(NeuralNetError("invalid student output row metadata".into()));
    }
    for (i, out) in outputs.iter_mut().enumerate() {
        let sym = inputs[i].symmetry;
        if !(0..=7).contains(&sym) {
            return Err(NeuralNetError("invalid student output symmetry".into()));
        }
        **out = NNOutput::default();
        copy_outputs_with_symmetry(
            &raw.policy_logits[i * 362..i * 362 + 361],
            &mut out.policy_probs[..361],
            1,
            19,
            19,
            sym,
        );
        out.policy_probs[361] = raw.policy_logits[i * 362 + 361];
        out.white_win_prob = raw.value_logits[i * 3];
        out.white_loss_prob = raw.value_logits[i * 3 + 1];
        out.white_no_result_prob = raw.value_logits[i * 3 + 2];
        out.white_score_mean = raw.score[i];
        out.white_lead = raw.score[i];
        if inputs[i].include_owner_map {
            let mut own = vec![0f32; 361];
            copy_outputs_with_symmetry(
                &raw.ownership_logits[i * 361..(i + 1) * 361],
                &mut own,
                1,
                19,
                19,
                sym,
            );
            out.white_owner_map = Some(own.into_boxed_slice());
        }
        out.nn_x_len = 19;
        out.nn_y_len = 19;
        out.policy_optimism_used = inputs[i].policy_optimism as f32;
    }
    Ok(())
}

pub fn get_output(
    h: &StudentHandle,
    n: i32,
    inputs: &[&mut NNResultBuf],
    outputs: &mut [&mut NNOutput],
) -> Result<(), NeuralNetError> {
    if n == 0 && inputs.is_empty() && outputs.is_empty() {
        return Ok(());
    }
    decode(&evaluate(h, n, inputs)?, n, inputs, outputs)
}

// Existing evaluator pipeline can submit twice before finishing. Student
// submissions execute synchronously; tokens own complete independent results.
pub fn submit(
    h: &StudentHandle,
    n: i32,
    inputs: &[&mut NNResultBuf],
) -> Result<usize, NeuralNetError> {
    let mut pending = h
        .pending
        .lock()
        .map_err(|_| NeuralNetError("student pending lock poisoned".into()))?;
    if pending.outputs.len() >= 2 {
        return Err(NeuralNetError(
            "student pipeline has two unfinished batches".into(),
        ));
    }
    let raw = evaluate(h, n, inputs)?;
    pending.next = pending
        .next
        .checked_add(1)
        .ok_or_else(|| NeuralNetError("student token overflow".into()))?;
    let rows = inputs
        .iter()
        .map(|r| (r.symmetry, r.include_owner_map, r.policy_optimism.to_bits()))
        .collect();
    let token = pending.next;
    pending.outputs.insert(token, PendingBatch { raw, rows });
    Ok(token)
}
pub fn finish(
    h: &StudentHandle,
    token: usize,
    n: i32,
    inputs: &[&mut NNResultBuf],
    outputs: &mut [&mut NNOutput],
) -> Result<(), NeuralNetError> {
    let mut pending = h
        .pending
        .lock()
        .map_err(|_| NeuralNetError("student pending lock poisoned".into()))?;
    let batch = pending
        .outputs
        .get(&token)
        .ok_or_else(|| NeuralNetError("unknown or consumed student token".into()))?;
    let rows: Vec<_> = inputs
        .iter()
        .map(|r| (r.symmetry, r.include_owner_map, r.policy_optimism.to_bits()))
        .collect();
    if rows != batch.rows {
        return Err(NeuralNetError(
            "student token row metadata differs from submission".into(),
        ));
    }
    decode(&batch.raw, n, inputs, outputs)?;
    pending.outputs.remove(&token);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn restores_all_symmetries_and_preserves_pass_and_raw_heads() {
        for sym in 0..8 {
            let original: Vec<f32> = (0..361).map(|i| i as f32 / 361.0).collect();
            let mut transformed = vec![0f32; 361];
            copy_inputs_with_symmetry(&original, &mut transformed, 1, 19, 19, 1, false, sym);
            let mut policy = transformed.clone();
            policy.push(2.0);
            let raw = StudentOutputs {
                policy_logits: policy,
                value_logits: vec![1.0, 2.0, 3.0],
                score: vec![0.5],
                ownership_logits: transformed,
            };
            let mut input = NNResultBuf::default();
            input.symmetry = sym;
            input.include_owner_map = true;
            let mut out = NNOutput::default();
            decode(&raw, 1, &[&mut input], &mut [&mut out]).unwrap();
            assert_eq!(&out.policy_probs[..361], original.as_slice());
            assert_eq!(out.policy_probs[361], 2.0);
            assert_eq!(out.white_owner_map.unwrap().as_ref(), original.as_slice());
            assert_eq!(out.white_score_mean, 0.5);
        }
    }
}
