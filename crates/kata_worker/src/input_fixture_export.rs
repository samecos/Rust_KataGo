//! Opt-in CPU-only replay of the two frozen G3 Worker failures.
//!
//! This child module intentionally calls private `prepare_context` instead of
//! recreating protocol rules/history handling. It never constructs an Engine,
//! NnEvaluator, CUDA runtime or model. No production API is exposed.
//!
//! Root-scheduled command (no CUDA feature needed):
//! KATAGO_G3_EXPORT_INPUTS=1 cargo test -p kata_worker --lib --no-default-features
//!   evaluator::input_fixture_export::export_g3_worker_input_fixtures_cpu
//!   -- --exact --nocapture
//!
//! Optional KATAGO_G3_INPUT_REQUEST_DIR overrides the source directory but the
//! exact frozen PB hashes remain mandatory. KATAGO_G3_INPUT_OUTPUT_DIR must be
//! a fresh directory below this workspace's target, with an existing parent.

use super::prepare_context;
use crate::wire;
use anyhow::{Context, Result, bail, ensure};
use kata_game::board::location;
use kata_game::symmetry::{copy_inputs_with_symmetry, invert};
use kata_nn::inputs::{NUM_FEATURES_GLOBAL_V7, NUM_FEATURES_SPATIAL_V7, fill_row_v7};
use prost::Message;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const MODEL_SHA: &str = "1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6";
const REQUEST_DIR: &str =
    "target/fork-parity-20260908/g3-worker-b14-vs-b16-r1/numeric-r1/b16-s2-window32/requests";
const OUTPUT_DIR: &str = "target/fork-parity-20260908/g3-worker-input-fixtures-r1";
const CASES: [(u64, i32, &str); 2] = [
    (
        11,
        2,
        "d93435970300d3f50e60df033f87e2fd33e3a7f41c7ffef1768622c03e9e0107",
    ),
    (
        14,
        5,
        "e21017a181b6a00310907aa7bd7e4fb53bb854de0e1a027b61d2ba9e6630737a",
    ),
];

fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn file_hash(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            return Ok(hex::encode(hash.finalize()));
        }
        hash.update(&buf[..n]);
    }
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("refuse overwrite or cannot create {}", path.display()))?;
    file.write_all(bytes)?;
    Ok(())
}

fn json_new(path: &Path, value: &Value) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    write_new(path, &bytes)
}

fn tensor_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn tensor(dir: &Path, name: &str, values: &[f32], shape: &[usize], layout: &str) -> Result<Value> {
    ensure!(
        shape.iter().product::<usize>() == values.len(),
        "tensor shape mismatch"
    );
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite input tensor"
    );
    let bytes = tensor_bytes(values);
    write_new(&dir.join(name), &bytes)?;
    Ok(
        json!({"file": name, "sha256": hash(&bytes), "bytes": bytes.len(),
              "elements": values.len(), "dtype": "float32", "byte_order": "little-endian",
              "shape": shape, "layout": layout}),
    )
}

fn full_request(request: &wire::EvalRequest) -> Value {
    let position = request
        .position
        .as_ref()
        .expect("prepare_context checked position");
    let p = request
        .parameters
        .as_ref()
        .expect("prepare_context checked parameters");
    json!({
        "task_id": request.task_id, "generation": request.generation,
        "session_id": request.session_id, "input_hash_hex": hex::encode(&request.input_hash),
        "model_sha256": request.model_sha256, "lease_ms": request.lease_ms,
        "position": {
            "board_size": position.board_size, "komi": position.komi, "rules": position.rules,
            "initial_player": position.initial_player, "next_player": position.next_player,
            "moves": position.moves.iter().map(|m| json!({"color": m.color, "vertex": m.vertex})).collect::<Vec<_>>(),
            "initial_stones": position.initial_stones.iter().map(|s| json!({"color": s.color, "vertex": s.vertex})).collect::<Vec<_>>()
        },
        "parameters": {
            "symmetry": p.symmetry, "policy_temperature": p.policy_temperature,
            "policy_optimism": p.policy_optimism, "draw_equivalent_wins_for_white": p.draw_equivalent_wins_for_white,
            "playout_doubling_advantage": p.playout_doubling_advantage, "include_ownership": p.include_ownership,
            "max_history": p.max_history, "conservative_pass": p.conservative_pass,
            "enable_passing_hacks": p.enable_passing_hacks, "always_compute_pass_alive": p.always_compute_pass_alive,
            "exclude_territory_adjacent_to_atari": p.exclude_territory_adjacent_to_atari,
            "avoid_mytdagger_hack": p.avoid_mytdagger_hack, "skip_cache": p.skip_cache,
            "allow_terminal_search_history": p.allow_terminal_search_history, "force_non_terminal": p.force_non_terminal
        }
    })
}

fn export_one(
    source: &Path,
    output: &Path,
    task_id: u64,
    symmetry: i32,
    expected_sha: &str,
) -> Result<Value> {
    let bytes = fs::read(source)?;
    ensure!(
        hash(&bytes) == expected_sha,
        "frozen request bytes changed: {}",
        source.display()
    );
    let request = wire::EvalRequest::decode(bytes.as_slice())
        .context("decode original EvalRequest protobuf")?;
    ensure!(request.task_id == task_id, "unexpected task identity");
    ensure!(
        request.model_sha256 == MODEL_SHA,
        "unexpected request model identity"
    );
    let context =
        prepare_context(&request).map_err(|e| anyhow::anyhow!("{}: {}", e.code, e.message))?;
    ensure!(
        context.params.symmetry == symmetry,
        "unexpected request symmetry"
    );
    ensure!(
        NUM_FEATURES_SPATIAL_V7 == 22 && NUM_FEATURES_GLOBAL_V7 == 19,
        "input profile dimensions changed"
    );
    let mut pre_spatial = vec![0.0f32; 22 * 19 * 19];
    let mut pre_global = vec![0.0f32; 19];
    fill_row_v7(
        &context.board,
        &context.history,
        context.next,
        &context.params,
        19,
        19,
        false,
        &mut pre_spatial,
        &mut pre_global,
    );
    let mut post_spatial = vec![0.0f32; pre_spatial.len()];
    copy_inputs_with_symmetry(
        &pre_spatial,
        &mut post_spatial,
        1,
        19,
        19,
        22,
        false,
        symmetry,
    );
    // cuda::get_output copies global features unchanged and transforms only spatial.
    let post_global = pre_global.clone();
    let mut restored = vec![0.0f32; pre_spatial.len()];
    copy_inputs_with_symmetry(
        &post_spatial,
        &mut restored,
        1,
        19,
        19,
        22,
        false,
        invert(symmetry),
    );
    ensure!(
        tensor_bytes(&restored) == tensor_bytes(&pre_spatial),
        "spatial symmetry roundtrip changed bits"
    );

    let dir = output.join(format!("task-{task_id:03}-sym-{symmetry}"));
    fs::create_dir(&dir)?;
    write_new(&dir.join("request.pb"), &bytes)?;
    let mut pre_combined = tensor_bytes(&pre_spatial);
    pre_combined.extend(tensor_bytes(&pre_global));
    let mut post_combined = tensor_bytes(&post_spatial);
    post_combined.extend(tensor_bytes(&post_global));
    let board_colors: Vec<_> = (0..361)
        .map(|i| context.board.colors[location::get_loc(i % 19, i / 19, 19) as usize])
        .collect();
    let p = context.params;
    let manifest = json!({
        "schema": 1, "status": "CPU_INPUT_FIXTURE_ONLY_NOT_NUMERICAL_CERTIFICATION",
        "source_request": {"path": source, "sha256": expected_sha, "bytes": bytes.len(), "copy": "request.pb"},
        "request": full_request(&request),
        "input_hash_scope": "Opaque service semantic hash echoed from protobuf; NEVER the SHA of the actual feature tensors below",
        "prepared_context": {
            "replay": "kata_worker::evaluator::prepare_context (same private production function)",
            "next_player": context.next, "rules": context.history.rules.to_json(),
            "board_colors_yx": board_colors, "board_pos_hash_debug": format!("{:?}", context.board.pos_hash),
            "replayed_moves": context.history.move_history.len(), "presumed_next_move_pla": context.history.presumed_next_move_pla,
            "is_game_finished": context.history.is_game_finished, "is_no_result": context.history.is_no_result,
            "encore_phase": context.history.encore_phase, "skip_cache": context.skip_cache, "include_ownership": context.include_ownership,
            "nn_input_params": {
                "symmetry": p.symmetry, "nn_policy_temperature_f32": p.nn_policy_temperature,
                "nn_policy_temperature_f32_bits": format!("{:08x}", p.nn_policy_temperature.to_bits()),
                "policy_optimism": p.policy_optimism, "draw_equivalent_wins_for_white": p.draw_equivalent_wins_for_white,
                "playout_doubling_advantage": p.playout_doubling_advantage, "max_history": p.max_history,
                "conservative_pass_and_is_root": p.conservative_pass_and_is_root,
                "enable_passing_hacks": p.enable_passing_hacks, "avoid_mytdagger_hack": p.avoid_mytdagger_hack
            }
        },
        "feature_contract": {"inputs_version": 7, "batch": 1, "use_nhwc": false, "symmetry": symmetry,
            "spatial_generator": "kata_nn::inputs::fill_row_v7", "spatial_transform": "kata_game::symmetry::copy_inputs_with_symmetry",
            "global_transform": "identity copy", "symmetry_inverse_bitwise_roundtrip": true,
            "spatial_pre_post_bit_differences": pre_spatial.iter().zip(&post_spatial).filter(|(a, b)| a.to_bits() != b.to_bits()).count(),
            "half_conversion": false, "model_loaded": false, "gpu_called": false,
            "batch_scope": "one actual request row, no padding; later fixed-batch probes must declare their replication/packing"},
        "tensors": {
            "pre_spatial": tensor(&dir, "pre_spatial.f32le", &pre_spatial, &[1, 22, 19, 19], "NCHW")?,
            "pre_global": tensor(&dir, "pre_global.f32le", &pre_global, &[1, 19], "NC")?,
            "post_spatial": tensor(&dir, "post_spatial.f32le", &post_spatial, &[1, 22, 19, 19], "NCHW")?,
            "post_global": tensor(&dir, "post_global.f32le", &post_global, &[1, 19], "NC")?
        },
        "actual_tensor_pair_sha256": {"concatenation": "spatial float32LE bytes then global float32LE bytes; lengths fixed by tensor descriptors",
            "pre_symmetry": hash(&pre_combined), "post_symmetry": hash(&post_combined)}
    });
    json_new(&dir.join("manifest.json"), &manifest)?;
    ensure!(
        file_hash(source)? == expected_sha,
        "source request changed during export"
    );
    Ok(
        json!({"task_id": task_id, "symmetry": symmetry, "directory": dir.file_name().unwrap().to_string_lossy(),
              "manifest_sha256": file_hash(&dir.join("manifest.json"))?, "actual_tensor_pair_sha256": manifest["actual_tensor_pair_sha256"]}),
    )
}

#[test]
fn export_g3_worker_input_fixtures_cpu() -> Result<()> {
    match std::env::var("KATAGO_G3_EXPORT_INPUTS").as_deref() {
        Err(_) | Ok("0") => {
            eprintln!("SKIP: set KATAGO_G3_EXPORT_INPUTS=1 for frozen CPU input export");
            return Ok(());
        }
        Ok("1") => {}
        Ok(_) => bail!("KATAGO_G3_EXPORT_INPUTS must be 0 or 1"),
    }
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let resolve = |key: &str, default: &str| {
        let path = std::env::var_os(key)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(default));
        if path.is_absolute() {
            path
        } else {
            workspace.join(path)
        }
    };
    let request_dir = resolve("KATAGO_G3_INPUT_REQUEST_DIR", REQUEST_DIR);
    let sources: Vec<_> = CASES
        .iter()
        .map(|(task, _, _)| request_dir.join(format!("{task:03}.pb")))
        .collect();
    if sources.iter().any(|path| !path.is_file()) {
        eprintln!(
            "SKIP: frozen G3 request fixtures missing below {}",
            request_dir.display()
        );
        return Ok(());
    }
    let output = resolve("KATAGO_G3_INPUT_OUTPUT_DIR", OUTPUT_DIR);
    ensure!(
        !output.exists(),
        "refuse overwrite of existing fixture directory {}",
        output.display()
    );
    let parent = output
        .parent()
        .context("output must have a parent")?
        .canonicalize()?;
    ensure!(
        parent.starts_with(workspace.join("target").canonicalize()?),
        "output must stay below workspace target"
    );
    // Validate all original identities before creating any output directory.
    for (source, (_, _, expected_sha)) in sources.iter().zip(CASES.iter()) {
        ensure!(
            file_hash(source)? == *expected_sha,
            "frozen request SHA mismatch: {}",
            source.display()
        );
    }
    fs::create_dir(&output)?;
    let mut cases = Vec::new();
    for (source, (task, symmetry, sha)) in sources.iter().zip(CASES.iter()) {
        cases.push(export_one(source, &output, *task, *symmetry, sha)?);
    }
    let exe = std::env::current_exe()?;
    let manifest = json!({"schema": 1, "status": "PASS_CPU_INPUT_FREEZE", "platform": std::env::consts::OS,
        "model_sha256_from_requests": MODEL_SHA, "model_loaded": false, "gpu_called": false,
        "test_executable": {"path": exe, "sha256": file_hash(&exe)?},
        "sources_at_compile_time": {
            "evaluator.rs": hash(include_bytes!("evaluator.rs")),
            "input_fixture_export.rs": hash(include_bytes!("input_fixture_export.rs")),
            "nn/inputs.rs": hash(include_bytes!("../../kata_nn/src/inputs.rs")),
            "nn/eval.rs": hash(include_bytes!("../../kata_nn/src/eval.rs")),
            "game/board.rs": hash(include_bytes!("../../kata_game/src/board.rs")),
            "game/history.rs": hash(include_bytes!("../../kata_game/src/history.rs")),
            "game/rules.rs": hash(include_bytes!("../../kata_game/src/rules.rs")),
            "game/symmetry.rs": hash(include_bytes!("../../kata_game/src/symmetry.rs")),
            "worker.proto": hash(include_bytes!("../proto/worker.proto"))
        },
        "scope": "CPU feature reconstruction only, no C++ oracle, GPU math, batch or performance certification",
        "cases": cases});
    json_new(&output.join("manifest.json"), &manifest)?;
    eprintln!(
        "PASS_CPU_INPUT_FREEZE: {}",
        output.join("manifest.json").display()
    );
    Ok(())
}
