//! Placeholder for `KataGo/cpp/tests/testnn.cpp`.
//!
//! The original file contains two independent test suites:
//!
//! 1. `runNNLayerTests` — exercises backend-specific layer test hooks
//!    (`NeuralNet::testEvaluateConv`, `testEvaluateBatchNorm`,
//!    `testEvaluateResidualBlock`, `testEvaluateGlobalPoolingResidualBlock`).
//!    These require a real NN backend (Eigen/CUDA/OpenCL/etc.) and are not
//!    supported by the current skeleton/dummy backend.
//!
//! 2. `runNNSymmetryTests` — verifies `SymmetryHelpers::copyInputsWithSymmetry`
//!    and `copyOutputsWithSymmetry`. This is already covered by
//!    `kata_game/tests/test_symmetries.rs`.
//!
//! Keeping this placeholder so the mapping entry has a corresponding Rust test
//! file.

#[test]
#[ignore = "requires real NN backend layer test hooks (not yet ported)"]
fn nn_layer_tests_reject_reason_is_documented() {
    // When a real backend is wired in, implement the four layer test blocks
    // (1x1/3x3/5x5 convolution, batch norm with/without mask, basic residual
    // block, global-pooling residual block) against the expected tensors from
    // the original C++ source.
    panic!("NN layer tests are not yet implementable without a real NN backend");
}
