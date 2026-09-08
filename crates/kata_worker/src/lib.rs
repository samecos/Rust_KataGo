//! Go Server's pure NN evaluation worker. Search stays on the server.

pub mod client;
pub mod evaluator;

#[allow(clippy::large_enum_variant)]
pub mod wire {
    tonic::include_proto!("goeval.v1");
}

pub const PROTOCOL_VERSION: u32 = 1;
pub const INPUT_PROFILE: &str = "katago-eval-v1";

/// Honest engine provenance and capabilities advertised after model loading.
#[derive(Clone, Debug)]
pub struct Metadata {
    pub model_sha256: String,
    pub model_version: u32,
    pub engine_commit: String,
    pub backend_info: String,
    pub supports_shortterm_error: bool,
    pub default_always_compute_pass_alive: bool,
    pub default_exclude_territory_adjacent_to_atari: bool,
}

#[derive(Clone, Debug)]
pub struct EvalFailure {
    pub code: String,
    pub message: String,
}

impl EvalFailure {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

pub struct EvaluationReport {
    pub result: Result<wire::NnOutput, EvalFailure>,
    pub context_us: Option<u64>,
    pub evaluator_us: Option<u64>,
}

/// Synchronous NN work is run outside the async gRPC event loop.
pub trait Evaluator: Send + Sync + 'static {
    fn metadata(&self) -> &Metadata;
    fn evaluate(&self, request: &wire::EvalRequest) -> EvaluationReport;
    fn stats(&self) -> (u64, u64);
}
