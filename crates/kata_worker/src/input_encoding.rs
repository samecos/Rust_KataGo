//! CPU-only v7/NCHW encoding of the exact context accepted by the Worker.
//! No model is loaded: the caller must separately verify its model input contract.
use super::prepare_context;
use crate::{EvalFailure, wire};
use kata_game::{
    board::location,
    symmetry::{copy_inputs_with_symmetry, invert},
};
use kata_nn::inputs::{
    MiscNNInputParams, NUM_FEATURES_GLOBAL_V7, NUM_FEATURES_SPATIAL_V7, fill_row_v7,
};
use sha2::{Digest, Sha256};

pub const SPATIAL_FLOATS: usize = 22 * 361;
pub const GLOBAL_FLOATS: usize = 19;
pub const ROW_FEATURE_DOMAIN: &[u8] = b"rustgo-worker-v7-nchw19-row-f32le-v1\0";

#[derive(Debug)]
pub struct PreparedEvidence {
    pub rules_json: String,
    pub board_colors_yx: Vec<i32>,
    pub next_player: i32,
    pub replayed_moves: usize,
    pub is_game_finished: bool,
    pub is_no_result: bool,
    pub encore_phase: i32,
    pub params: MiscNNInputParams,
    pub skip_cache: bool,
    pub include_ownership: bool,
}

#[derive(Debug)]
pub struct EncodedWorkerRow {
    pub pre_spatial: Vec<f32>,
    pub spatial: Vec<f32>,
    pub global: Vec<f32>,
    pub evidence: PreparedEvidence,
}

fn invalid(message: &str) -> EvalFailure {
    EvalFailure::new("INVALID_CONTEXT", message)
}

pub fn f32le(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Fixed v7 shape and domain bind the exact f32 bits, including signed zero.
pub fn row_feature_sha256(spatial: &[f32], global: &[f32]) -> Result<String, EvalFailure> {
    if spatial.len() != SPATIAL_FLOATS
        || global.len() != GLOBAL_FLOATS
        || spatial.iter().chain(global).any(|x| !x.is_finite())
    {
        return Err(invalid("invalid v7 input feature shape/nonfinite value"));
    }
    let mut digest = Sha256::new();
    digest.update(ROW_FEATURE_DOMAIN);
    digest.update(f32le(spatial));
    digest.update(f32le(global));
    Ok(hex::encode(digest.finalize()))
}

/// Uses the production Worker's strict replay and parameter conversions.
/// Applies the same spatial-only symmetry used by CUDA upload exactly once.
/// This produces input features, not model compatibility or numerical evidence.
pub fn encode_request_v7_cpu(
    request: &wire::EvalRequest,
    expected_model_sha256: &str,
) -> Result<EncodedWorkerRow, EvalFailure> {
    if expected_model_sha256.len() != 64
        || !expected_model_sha256.bytes().all(|b| b.is_ascii_hexdigit())
        || !request
            .model_sha256
            .eq_ignore_ascii_case(expected_model_sha256)
    {
        return Err(invalid(
            "request model_sha256 differs from expected model identity",
        ));
    }
    let context = prepare_context(request)?;
    encode_prepared_v7(&context)
}

/// Shared strict Prepared context; never a public caller-supplied evidence object.
pub(super) fn encode_prepared_v7(context: &super::Prepared) -> Result<EncodedWorkerRow, EvalFailure> {
    if NUM_FEATURES_SPATIAL_V7 != 22 || NUM_FEATURES_GLOBAL_V7 != 19 {
        return Err(invalid("v7 feature contract changed"));
    }
    let mut pre_spatial = vec![0.0; SPATIAL_FLOATS];
    let mut global = vec![0.0; GLOBAL_FLOATS];
    fill_row_v7(
        &context.board,
        &context.history,
        context.next,
        &context.params,
        19,
        19,
        false,
        &mut pre_spatial,
        &mut global,
    );
    let mut spatial = vec![0.0; SPATIAL_FLOATS];
    copy_inputs_with_symmetry(
        &pre_spatial,
        &mut spatial,
        1,
        19,
        19,
        22,
        false,
        context.params.symmetry,
    );
    let mut restored = vec![0.0; SPATIAL_FLOATS];
    copy_inputs_with_symmetry(
        &spatial,
        &mut restored,
        1,
        19,
        19,
        22,
        false,
        invert(context.params.symmetry),
    );
    if f32le(&restored) != f32le(&pre_spatial) {
        return Err(invalid(
            "spatial symmetry inverse did not preserve feature bits",
        ));
    }
    row_feature_sha256(&pre_spatial, &global)?;
    row_feature_sha256(&spatial, &global)?;
    Ok(EncodedWorkerRow {
        pre_spatial,
        spatial,
        global,
        evidence: PreparedEvidence {
            rules_json: context.history.rules.to_json().to_string(),
            board_colors_yx: (0..361)
                .map(|i| {
                    context.board.colors[location::get_loc(i % 19, i / 19, 19) as usize] as i32
                })
                .collect(),
            next_player: context.next as i32,
            replayed_moves: context.history.move_history.len(),
            is_game_finished: context.history.is_game_finished,
            is_no_result: context.history.is_no_result,
            encore_phase: context.history.encore_phase,
            params: context.params,
            skip_cache: context.skip_cache,
            include_ownership: context.include_ownership,
        },
    })
}

#[cfg(test)]
#[path = "input_encoding_tests.rs"]
mod tests;
