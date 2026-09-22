//! Model-level metamorphic test: restoring zero FFN channels to a full B15
//! width must preserve W8A8 inference. No transformed model is published.
#![cfg(feature = "cuda")]
use kata_game::{
    board::{Board, P_BLACK},
    history::BoardHistory,
    rules::Rules,
};
use kata_nn::{
    backends::{
        cuda::CudaRuntime,
        cuda_exec::{CudaModel, CudaOutputsHost, CudaWorkspace},
        int8::Int8Scope,
    },
    inputs::{MiscNNInputParams, fill_row_v7},
    native_model::lower_model,
    onnx_parser::{Layer, Tensor, TensorData},
};

#[test]
fn pruned_and_zero_expanded_b15_have_identical_int8_outputs() {
    let Some(dir) = std::env::var_os("KATAGO_TEST_MODEL_DIR") else {
        return;
    };
    let file = std::path::PathBuf::from(dir).join("b15-ffn-pruned-a8.bin.gz");
    if !file.is_file() {
        eprintln!("skipped missing {}", file.display());
        return;
    }
    let desc =
        kata_nn::model_parser::load_model_from_bytes(&std::fs::read(&file).unwrap(), true, true)
            .unwrap();
    let mut graph = lower_model(&desc).unwrap();
    drop(desc);
    let rt = CudaRuntime::new().unwrap();
    let stream = rt.device.new_stream().unwrap();
    let board = Board::new(19, 19);
    let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
    let mut spatial = vec![0.0; 22 * 361];
    let mut global = vec![0.0; 19];
    fill_row_v7(
        &board,
        &hist,
        P_BLACK,
        &MiscNNInputParams::default(),
        19,
        19,
        false,
        &mut spatial,
        &mut global,
    );
    let spatial = stream.clone_htod(&spatial).unwrap();
    let global = stream.clone_htod(&global).unwrap();
    let evaluate = |graph: &kata_nn::onnx_parser::LayerGraph| -> CudaOutputsHost {
        let model = CudaModel::load_int8(graph, &rt, &stream, Int8Scope::Transformer).unwrap();
        let mut workspace = CudaWorkspace::new(&stream, &model, 1).unwrap();
        model
            .apply(&rt, &stream, &mut workspace, &spatial, &global)
            .unwrap();
        workspace.to_host(&stream).unwrap()
    };
    let pruned = evaluate(&graph);
    let mid = graph.mid_channels;
    let dense = 3 * mid;
    let mut count = 0;
    for layer in &mut graph.layers {
        if let Layer::Ffn(ffn) = layer {
            let hidden = ffn.hidden;
            for weight in [&mut ffn.gate_weight, &mut ffn.up_weight] {
                let mut values = weight.f32_data().to_vec();
                values.resize(dense * mid, 0.0);
                *weight = Tensor {
                    dims: vec![dense as i64, mid as i64],
                    data: TensorData::F32(values),
                };
            }
            let mut down = vec![0.0; mid * dense];
            for row in 0..mid {
                down[row * dense..row * dense + hidden]
                    .copy_from_slice(&ffn.down_weight.f32_data()[row * hidden..(row + 1) * hidden]);
            }
            ffn.down_weight = Tensor {
                dims: vec![mid as i64, dense as i64],
                data: TensorData::F32(down),
            };
            ffn.hidden = dense;
            count += 1;
        }
    }
    assert_eq!((count, mid, dense), (45, 512, 1536));
    let expanded = evaluate(&graph);
    for (name, a, b) in [
        ("policy", pruned.policy, expanded.policy),
        ("value", pruned.value, expanded.value),
        ("misc", pruned.misc, expanded.misc),
        ("moremisc", pruned.moremisc, expanded.moremisc),
        ("ownership", pruned.ownership, expanded.ownership),
    ] {
        assert!(a.iter().all(|v| v.is_finite()), "{name} nonfinite");
        assert_eq!(
            a, b,
            "{name}: zero channel restoration changed INT8 outputs"
        );
    }
}
