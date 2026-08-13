//! Placeholder for `KataGo/cpp/tests/tinymodel.h` and `tinymodel.cpp`.
//!
//! The C++ tiny-model test decodes base64-encoded real neural-network model
//! files, writes them to disk, initializes a live NN evaluator (CUDA/OpenCL/
//! Eigen), and asserts exact numeric policy/value/ownership outputs.
//!
//! The Rust port currently only has a skeleton/dummy NN backend that produces
//! uniform policies and neutral values. A real backend and model-file parser are
//! required before this test can run meaningfully. Keeping this placeholder so
//! the mapping entry has a corresponding Rust test file.

#[test]
#[ignore = "requires real NN backend and model parser (not yet ported)"]
fn tiny_model_reject_reason_is_documented() {
    // When a real backend is wired in, this test should:
    // 1. Decode `tinyModelBase64Part0..Part6` into a `.bin.gz` model file.
    // 2. Initialize an `NnEvaluator` with that model on a 19x19 board.
    // 3. Run inference with symmetry 6 and assert policy/value/ownership
    //    against the expected tables in the original C++ source.
    // 4. Repeat for `tinyMishModelBase64` on 19x19 (symmetry 7) and 13x6
    //    (symmetry 1).
    panic!("tiny model test is not yet implementable without a real NN backend");
}
