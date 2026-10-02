//! Explicit CPU bridge for previously captured raw CUDA heads.
//! Does not execute a model, prove raw tensor provenance, or produce a certificate.
//! Model contract and request context cannot be caller-authored evidence structs.
use super::{Prepared, convert_output, invalid_context, prepare_context};
use crate::{EvalFailure, wire};
use kata_nn::desc::ModelPostProcessParams;
use kata_nn::inputs::get_hash;
use kata_nn::output_postprocess::{RawHeads, decode_raw_row_v7, postprocess_output};
use prost::Message;
use sha2::{Digest, Sha256};

/// Derived only by parsing the exact native model bytes and validating its lower.
/// Fields and construction from descriptors are deliberately private.
pub struct NativeModelOutputContract {
    model_sha256: String,
    graph_sha256: String,
    model_version: i32,
    postprocess: ModelPostProcessParams,
    contract_sha256: String,
}

/// Strict PB replay and encoded-feature identity. No public field or constructor.
/// In particular, a board-color summary cannot stand in for BoardHistory.
pub struct PreparedOutputRequest {
    context: Prepared,
    model_contract_sha256: String,
    request_pb_sha256: String,
    row_feature_sha256: String,
}

fn digest(raw: &[u8]) -> String {
    hex::encode(Sha256::digest(raw))
}
fn contract_digest(model: &str, graph: &str, version: i32, pp: ModelPostProcessParams) -> String {
    let mut h = Sha256::new();
    h.update(b"rustgo-native-output-postprocess-contract-v1\0");
    h.update(model.as_bytes());
    h.update(graph.as_bytes());
    h.update(version.to_le_bytes());
    // Bind every actual parameter including the currently unused TD multiplier.
    for value in [
        pp.td_score_multiplier,
        pp.score_mean_multiplier,
        pp.score_stdev_multiplier,
        pp.lead_multiplier,
        pp.variance_time_multiplier,
        pp.shortterm_value_error_multiplier,
        pp.shortterm_score_error_multiplier,
    ] {
        h.update(value.to_bits().to_le_bytes());
    }
    h.update(pp.output_scale_multiplier.to_bits().to_le_bytes());
    hex::encode(h.finalize())
}

impl NativeModelOutputContract {
    /// CPU-only; parses and lowers the same exact native bytes as the CUDA loader.
    /// Binary/compressed flags describe the supplied bytes, never a neighboring file.
    pub fn from_native_model_bytes(
        bytes: &[u8],
        expected_model_sha256: &str,
        binary: bool,
        compressed: bool,
    ) -> Result<Self, EvalFailure> {
        let model_sha256 = digest(bytes);
        if !model_sha256.eq_ignore_ascii_case(expected_model_sha256) {
            return Err(invalid_context("native output contract model SHA mismatch"));
        }
        let desc = kata_nn::model_parser::load_model_from_bytes(bytes, binary, compressed)
            .map_err(|e| invalid_context(format!("native output contract parse: {e}")))?;
        let graph = kata_nn::native_model::lower_model(&desc)
            .map_err(|e| invalid_context(format!("native output contract lower: {e}")))?;
        // lower_model already checks v17, 22/19, no metadata and native head layout.
        let graph_sha256 =
            kata_nn::quantization_plan::graph_sha256(&graph).map_err(invalid_context)?;
        if desc.sha256 != model_sha256 {
            return Err(invalid_context("native parser source identity differs"));
        }
        let model_version = desc.model_version;
        let postprocess = desc.post_process_params;
        let contract_sha256 =
            contract_digest(&model_sha256, &graph_sha256, model_version, postprocess);
        Ok(Self {
            model_sha256,
            graph_sha256,
            model_version,
            postprocess,
            contract_sha256,
        })
    }

    pub fn model_sha256(&self) -> &str {
        &self.model_sha256
    }
    pub fn graph_sha256(&self) -> &str {
        &self.graph_sha256
    }
    pub fn model_version(&self) -> i32 {
        self.model_version
    }
    pub fn contract_sha256(&self) -> &str {
        &self.contract_sha256
    }

    /// Reconstruct full legal history via production prepare_context once.
    /// Re-encode using the same production input wrapper body and require the
    /// already-frozen exact PB and post-symmetry row digests supplied by the
    /// artifact reader. Features do not encode every postprocess-only parameter.
    pub fn prepare_request_pb(
        &self,
        request_pb: &[u8],
        expected_request_pb_sha256: &str,
        expected_row_feature_sha256: &str,
    ) -> Result<PreparedOutputRequest, EvalFailure> {
        let request_pb_sha256 = digest(request_pb);
        if request_pb_sha256 != expected_request_pb_sha256 {
            return Err(invalid_context(
                "raw output request differs from frozen PB bytes",
            ));
        }
        let request = wire::EvalRequest::decode(request_pb)
            .map_err(|e| invalid_context(format!("raw output request protobuf: {e}")))?;
        if !request
            .model_sha256
            .eq_ignore_ascii_case(&self.model_sha256)
        {
            return Err(invalid_context(
                "raw output request model differs from parsed model",
            ));
        }
        let context = prepare_context(&request)?;
        let encoded = super::input_encoding::encode_prepared_v7(&context)?;
        let row_feature_sha256 =
            super::input_encoding::row_feature_sha256(&encoded.spatial, &encoded.global)?;
        if row_feature_sha256 != expected_row_feature_sha256 {
            return Err(invalid_context(
                "raw output request differs from frozen feature row",
            ));
        }
        Ok(PreparedOutputRequest {
            context,
            model_contract_sha256: self.contract_sha256.clone(),
            request_pb_sha256,
            row_feature_sha256,
        })
    }

    /// CPU decode -> unchanged evaluator math -> unchanged Worker validator/PB.
    /// Caller must bind batch/row/raw-head artifacts to this prepared request;
    /// this method cannot attest that a GPU actually produced the supplied bytes.
    pub fn postprocess_raw_row(
        &self,
        request: &PreparedOutputRequest,
        raw: RawHeads<'_>,
        physical_batch: usize,
        row: usize,
    ) -> Result<wire::NnOutput, EvalFailure> {
        if request.model_contract_sha256 != self.contract_sha256 {
            return Err(invalid_context(
                "prepared request belongs to another model contract",
            ));
        }
        let operation = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let context = &request.context;
            let mut output = decode_raw_row_v7(
                raw,
                physical_batch,
                row,
                &context.params,
                context.include_ownership,
            )
            .map_err(|e| EvalFailure::new("INVALID_OUTPUT", e))?;
            postprocess_output(
                self.model_version,
                self.postprocess,
                19,
                19,
                362,
                &context.board,
                &context.history,
                context.next,
                &context.params,
                &mut output,
            );
            // Matches the fresh, cache-free evaluator finalization. PB does not
            // serialize nn_hash, but keep the intermediate NNOutput truthful.
            output.nn_hash = get_hash(
                &context.board,
                &context.history,
                context.next,
                &context.params,
            );
            convert_output(&output, context.include_ownership, self.model_version >= 9)
        }));
        match operation {
            Ok(value) => value,
            Err(panic) => Err(EvalFailure::new(
                "INVALID_OUTPUT",
                super::panic_message(panic),
            )),
        }
    }
}

impl PreparedOutputRequest {
    pub fn request_pb_sha256(&self) -> &str {
        &self.request_pb_sha256
    }
    pub fn row_feature_sha256(&self) -> &str {
        &self.row_feature_sha256
    }
    pub fn model_contract_sha256(&self) -> &str {
        &self.model_contract_sha256
    }
}

#[cfg(test)]
#[path = "output_postprocessing_tests.rs"]
mod tests;
