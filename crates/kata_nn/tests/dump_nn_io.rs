//! Dumps NN inputs/outputs for a set of positions using the TensorRT backend,
//! for comparison against the ONNX Runtime FP32 golden reference
//! (`scripts/compare_nn_output.py`).
//!
//! Environment variables:
//! - `KATAGO_ONNX_MODEL`  (default `D:/code/b11fix.onnx`)
//! - `KATAGO_DUMP_DIR`    (default `target/nn_io_dump`)
//! - `KATAGO_DUMP_POSITIONS` (default 16)
#![cfg(all(feature = "trt", trt_shim_available))]

use kata_core::logger::{Logger, LoggerOptions};
use kata_game::board::{Board, P_BLACK};
use kata_game::history::BoardHistory;
use kata_game::rules::Rules;
use kata_nn::backend::{Backend, Enabled, NNResultBuf, NNOutput};
use kata_nn::backends::trt::TensorRtBackend;
use kata_nn::inputs::{MiscNNInputParams, fill_row_v7};

fn write_f32(path: &std::path::Path, data: &[f32]) {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(path, bytes).expect("write dump file");
}

#[test]
fn dump_nn_io() {
    let model = std::env::var("KATAGO_ONNX_MODEL")
        .unwrap_or_else(|_| "D:/code/b11fix.onnx".to_string());
    let dump_dir = std::env::var("KATAGO_DUMP_DIR")
        .unwrap_or_else(|_| "target/nn_io_dump".to_string());
    let num_positions: usize = std::env::var("KATAGO_DUMP_POSITIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    if !std::path::Path::new(&model).exists() {
        eprintln!("skipped: model not found at {model}");
        return;
    }

    let logger = Logger::new(
        LoggerOptions {
            log_to_stdout: false,
            log_to_stderr: true,
            log_time: false,
        },
        None,
    );
    let cfg = kata_core::config::ConfigParser::from_str(
        "nnMaxBatchSize = 4",
        false,
        true,
    )
    .unwrap();

    let backend = TensorRtBackend;
    let loaded = backend
        .load_model_file(&model, "")
        .expect("load model");
    let ctx = backend
        .create_compute_context(&[0], &logger, 19, 19, "", Enabled::Auto, &*loaded, &cfg)
        .expect("create compute context");
    let handle = backend
        .create_compute_handle(&*ctx, &*loaded, &logger, 4, true, false, 0, 0)
        .expect("create compute handle");
    let buffers = backend
        .create_input_buffers(&*loaded, 4, 19, 19)
        .expect("create input buffers");

    std::fs::create_dir_all(&dump_dir).expect("create dump dir");
    let dump_dir = std::path::PathBuf::from(dump_dir);

    let nn_input_params = MiscNNInputParams::default();

    for i in 0..num_positions {
        // Deterministic pseudo-random position: empty board + i random-ish legal moves.
        let mut board = Board::new(19, 19);
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        let mut next_player = P_BLACK;
        let mut rand = kata_core::rng::Rand::new_from_seed(&format!("dump-{i}"));
        let num_moves = i * 7 % 80;
        for _ in 0..num_moves {
            // Collect legal moves, pick one deterministically.
            let mut legal = Vec::new();
            for y in 0..19 {
                for x in 0..19 {
                    let loc = kata_game::board::location::get_loc(x, y, 19);
                    if hist.is_legal(&board, loc, next_player) {
                        legal.push(loc);
                    }
                }
            }
            if legal.is_empty() {
                break;
            }
            let loc = legal[rand.next_u64() as usize % legal.len()];
            hist.make_board_move_assume_legal(&mut board, loc, next_player);
            next_player = kata_game::board::get_opp(next_player);
        }

        let mut spatial = vec![0.0f32; 22 * 19 * 19];
        let mut global = vec![0.0f32; 19];
        fill_row_v7(
            &board,
            &hist,
            next_player,
            &nn_input_params,
            19,
            19,
            false, // NCHW
            &mut spatial,
            &mut global,
        );

        let mut result_buf = NNResultBuf::new();
        result_buf.row_spatial_buf = spatial;
        result_buf.row_global_buf = global;
        result_buf.symmetry = 0;
        result_buf.policy_optimism = 0.0;
        result_buf.include_owner_map = true;
        result_buf.board_x_size_for_server = 19;
        result_buf.board_y_size_for_server = 19;

        let mut output = NNOutput::default();
        backend
            .get_output(
                &*handle,
                &*buffers,
                1,
                &mut [&mut result_buf],
                &mut [&mut output],
            )
            .expect("get_output");

        write_f32(&dump_dir.join(format!("pos{i}_spatial.bin")), &result_buf.row_spatial_buf);
        write_f32(&dump_dir.join(format!("pos{i}_global.bin")), &result_buf.row_global_buf);
        write_f32(&dump_dir.join(format!("pos{i}_policy.bin")), &output.policy_probs);
        write_f32(
            &dump_dir.join(format!("pos{i}_value.bin")),
            &[
                output.white_win_prob,
                output.white_loss_prob,
                output.white_no_result_prob,
            ],
        );
        write_f32(
            &dump_dir.join(format!("pos{i}_misc.bin")),
            &[
                output.white_score_mean,
                output.white_score_mean_sq,
                output.white_lead,
                output.var_time_left,
                output.shortterm_winloss_error,
                output.shortterm_score_error,
            ],
        );
        if let Some(map) = &output.white_owner_map {
            write_f32(&dump_dir.join(format!("pos{i}_ownership.bin")), map);
        }
    }

    let meta = format!(
        "{{\"model\":\"{model}\",\"n\":{num_positions},\"spatial_elts\":{},\"global_elts\":19,\"policy_elts\":362,\"value_elts\":3,\"misc_elts\":6,\"ownership_elts\":361}}",
        22 * 19 * 19
    );
    std::fs::write(dump_dir.join("meta.json"), meta).expect("write meta");
    println!("dumped {num_positions} positions to {}", dump_dir.display());
}
