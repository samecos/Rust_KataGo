//! Fixed-physical-batch raw output evidence for the INT8 RMS fusion toggle.
//! Run in separate processes with KATAGO_CUDA_INT8_RMS_FUSION=0/1 and distinct
//! KATAGO_INT8_RAW_DUMP files, then compare `cases` exactly. Not an FP32 oracle.
#![cfg(feature = "cuda")]
use kata_game::{
    board::{Board, P_BLACK},
    history::BoardHistory,
    rules::Rules,
};
use kata_nn::{
    backends::{
        cuda::CudaRuntime,
        cuda_exec::{CudaModel, CudaWorkspace},
        int8::Int8Scope,
    },
    inputs::{MiscNNInputParams, fill_row_v7},
    native_model::lower_model,
};
use std::io::Write;

#[test]
fn dump_int8_fixed_batches() {
    let Some(destination) = std::env::var_os("KATAGO_INT8_RAW_DUMP") else {
        eprintln!("SKIP: KATAGO_INT8_RAW_DUMP not set");
        return;
    };
    let directory = std::path::PathBuf::from(
        std::env::var_os("KATAGO_TEST_MODEL_DIR").expect("model directory required"),
    );
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .unwrap();
    let rt = CudaRuntime::new().unwrap();
    let stream = rt.device.new_stream().unwrap();
    let mut spatial = Vec::new();
    let mut global = Vec::new();
    for position in 0..8 {
        let mut board = Board::new(19, 19);
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        let mut player = P_BLACK;
        let mut rand = kata_core::rng::Rand::new_from_seed(&format!("int8-rms-{position}"));
        for _ in 0..position * 9 {
            let legal: Vec<_> = (0..361)
                .map(|i| kata_game::board::location::get_loc(i % 19, i / 19, 19))
                .filter(|&loc| hist.is_legal(&board, loc, player))
                .collect();
            if legal.is_empty() {
                break;
            }
            let loc = legal[rand.next_u64() as usize % legal.len()];
            hist.make_board_move_assume_legal(&mut board, loc, player);
            player = kata_game::board::get_opp(player);
        }
        let mut s = vec![0.0; 22 * 361];
        let mut g = vec![0.0; 19];
        fill_row_v7(
            &board,
            &hist,
            player,
            &MiscNNInputParams::default(),
            19,
            19,
            false,
            &mut s,
            &mut g,
        );
        spatial.extend(s);
        global.extend(g);
    }
    let mut cases = Vec::new();
    for (file, min_width) in [
        ("b11c768h12nbt3tflrs-fson-silu.bin.gz", 0),
        ("b15-ffn-pruned-a8.bin.gz", 0),
        ("b15-ffn-pruned-a8.bin.gz", 384),
    ] {
        let path = directory.join(file);
        let bytes = std::fs::read(&path).unwrap();
        let desc = kata_nn::model_parser::load_model_from_bytes(&bytes, true, true).unwrap();
        let graph = lower_model(&desc).unwrap();
        let model = CudaModel::load_int8_selective(&graph, &rt, &stream, Int8Scope::Ffn, min_width)
            .unwrap();
        for batch in [1, 8] {
            let mut ws = CudaWorkspace::new(&stream, &model, batch).unwrap();
            let s = stream.clone_htod(&spatial[..batch * 22 * 361]).unwrap();
            let g = stream.clone_htod(&global[..batch * 19]).unwrap();
            model.apply(&rt, &stream, &mut ws, &s, &g).unwrap();
            let raw = ws.to_host(&stream).unwrap();
            let mut fields = serde_json::Map::new();
            for (name, values) in [
                ("policy", raw.policy),
                ("value", raw.value),
                ("misc", raw.misc),
                ("moremisc", raw.moremisc),
                ("ownership", raw.ownership),
            ] {
                assert!(
                    values.iter().all(|v| v.is_finite()),
                    "{file}/{batch}/{name}"
                );
                fields.insert(
                    name.into(),
                    serde_json::json!(values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()),
                );
            }
            cases.push(serde_json::json!({"model":file, "model_sha256":kata_nn::tactic_plan::sha256_file(&path).unwrap(), "min_width":min_width, "batch":batch, "output_bits":fields}));
        }
    }
    let report = serde_json::json!({"status":"PASS_RAW_DUMP_NOT_FP32_CERTIFICATION", "rms_fusion":std::env::var("KATAGO_CUDA_INT8_RMS_FUSION").unwrap_or_else(|_| "auto".into()), "cases":cases});
    output
        .write_all(&serde_json::to_vec(&report).unwrap())
        .unwrap();
    eprintln!("PASS_RAW_DUMP_NOT_FP32_CERTIFICATION: 3 profiles x B1/B8, all 5 output tensors");
}
