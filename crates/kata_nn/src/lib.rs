//! KataGo neural network abstraction and backends.

pub mod activations;
pub mod backend;
pub mod backends;
pub mod debug;
pub mod desc;
pub mod eval;
pub mod inputs;
pub mod model_parser;
pub mod onnx_builder;
pub mod onnx_model;
pub mod onnx_parser;
pub mod score_value;
pub mod sgf_meta;
pub mod tactic_plan;
pub mod version;

/// Prost-generated ONNX protobuf types (from the vendored `onnx.proto`).
#[allow(clippy::all)]
pub(crate) mod onnx_proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}
