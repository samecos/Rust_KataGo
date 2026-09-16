//! Explicit physical-batch/native-TF3 full-output diagnostic; never a C++ oracle.
//!
//! KATAGO_RUN_L2_INNER_DIAG=1 enables each test in a separate process. Then model/GPU/configuration
//! errors fail loudly. KATAGO_TEST_MODEL_DIR must contain the SHA-bound model.
//! Fixed KATAGO_DUMP_BATCH=14 and KATAGO_DUMP_LANES=2; KATAGO_DUMP_L2_INNER=0|1 required,
//! KATAGO_DUMP_DIR=<new or empty directory>. Exactly the same 16 deterministic
//! positions are evaluated on every lane; final batches repeat the last position.
//!
//! Each lane has independent input buffers, stream, executor workspace and Lt
//! workspace. Scoped host threads enqueue apply on those streams, followed by
//! one device synchronization before any downloads. This checks independent
//! lane execution, not a performance/overlap claim or Worker batching policy.
//!
//! Raw FP32 little-endian files retain all policy 6x362, value 3, misc 10,
//! moremisc 8, ownership 361 elements. Real positions use laneN/posI_*.bin;
//! padding also has its own files and meta row mapping. B1 files are an internal
//! consistency reference only; the separate C++ FP32 full-model gate is required.
#![cfg(feature = "cuda")]

#[path = "l2_inner_diag/l2_cache.rs"]
mod l2_cache;
#[path = "l2_inner_diag/l2_forward.rs"]
mod l2_forward;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use kata_game::board::{Board, P_BLACK};
use kata_game::history::BoardHistory;
use kata_game::rules::Rules;
use kata_nn::backends::cuda::{CudaRuntime, backend_build_fingerprint, device_fingerprint};
use kata_nn::backends::cuda_exec::{CudaModel, CudaOutputsHost, CudaWorkspace};
use kata_nn::inputs::{MiscNNInputParams, fill_row_v7};
use kata_nn::model_parser::load_model_from_bytes;
use kata_nn::native_model::lower_model;
use kata_nn::tactic_plan::{installed_plan_id, sha256_file, tactic_var};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const MODEL: &str = "kata1-tf3-b11c768-s11001M-d5973M.bin.gz";
const MODEL_SHA256: &str = "1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6";
const POSITIONS: usize = 16;
const SPATIAL: usize = 22 * 361;
const GLOBAL: usize = 19;
const POLICY: usize = 6 * 362;

struct Position {
    spatial: Vec<f32>,
    global: Vec<f32>,
    sha256: String,
}

// Keep the position/seed/move/feature construction identical to dump_nn_io_cuda.
fn make_position(i: usize) -> Position {
    let mut board = Board::new(19, 19);
    let mut hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
    let mut next_player = P_BLACK;
    let mut rand = kata_core::rng::Rand::new_from_seed(&format!("dump-{i}"));
    for _ in 0..i * 7 % 80 {
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
    let mut spatial = vec![0.0f32; SPATIAL];
    let mut global = vec![0.0f32; GLOBAL];
    fill_row_v7(
        &board,
        &hist,
        next_player,
        &MiscNNInputParams::default(),
        19,
        19,
        false,
        &mut spatial,
        &mut global,
    );
    let sha256 = input_hash(&spatial, &global);
    Position {
        spatial,
        global,
        sha256,
    }
}

fn input_hash(spatial: &[f32], global: &[f32]) -> String {
    let mut hash = Sha256::new();
    for &value in spatial.iter().chain(global) {
        assert!(value.is_finite(), "nonfinite generated input");
        hash.update(value.to_le_bytes());
    }
    hex::encode(hash.finalize())
}

fn write_f32(path: &Path, data: &[f32]) -> String {
    assert!(
        data.iter().all(|x| x.is_finite()),
        "nonfinite values for {}",
        path.display()
    );
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, &bytes).expect("write complete raw FP32 dump");
    hex::encode(Sha256::digest(&bytes))
}

fn write_meta(directory: &Path, meta: &Value) {
    std::fs::write(
        directory.join("meta.json"),
        serde_json::to_vec_pretty(meta).unwrap(),
    )
    .expect("write dump metadata");
}

fn integer_env(name: &str, default: usize, accepted: &[usize]) -> usize {
    let value = match std::env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .unwrap_or_else(|_| panic!("{name} must be an integer")),
        Err(std::env::VarError::NotPresent) => default,
        Err(error) => panic!("{name}: {error}"),
    };
    assert!(
        accepted.contains(&value),
        "{name}={value}; expected one of {accepted:?}"
    );
    value
}

fn selected_tactics() -> BTreeMap<String, Value> {
    // Use the production resolver for every decision; do not bypass a plan.
    let mut selection = BTreeMap::new();
    for (key, default, allowed) in [
        ("KATAGO_CUDA_CUBLASLT", "1", &["0", "1"][..]),
        ("KATAGO_CUDA_DUALFFN", "0", &["0", "1"][..]),
        ("KATAGO_CUDA_SPLITK", "0", &["0", "1"][..]),
        ("KATAGO_CUDA_STEM_MAP3D", "0", &["0", "1"][..]),
        ("KATAGO_CUDA_T32", "0", &["0", "1"][..]),
        ("KATAGO_CUDA_T64N32", "0", &["0", "1"][..]),
        (
            "KATAGO_CUDA_FUSION",
            "none",
            &["none", "up", "down", "all"][..],
        ),
        (
            "KATAGO_CUDA_CUBLASLT_RANK",
            "heuristic",
            &["heuristic", "time"][..],
        ),
        (
            "KATAGO_CUDA_RESIDUAL_ALGO",
            "heuristic",
            &["heuristic", "tf3_5070ti_r1"][..],
        ),
        (
            "KATAGO_CUDA_GEMM_LAYOUT",
            "tn",
            &["tn", "nn_k384", "nn_k384_b8", "nn_k384_b16"][..],
        ),
        (
            "KATAGO_CUDA_ATTN_TILE",
            "q128",
            &["q128", "q64", "q64-serial"][..],
        ),
        (
            "KATAGO_CUDA_ATTN",
            "fa2",
            &["fa2", "v3", "fa4-strict-b14-r1"][..],
        ),
        ("KATAGO_CUDA_RMS", "warp4", &["v1"][..]),
        ("KATAGO_CUDA_NOGRAPH", "0", &["0", "1"][..]),
        ("KATAGO_CUDA_PADBATCH", "0", &["0", "1"][..]),
        ("KATAGO_CUDA_NOPIPELINE", "0", &["0", "1"][..]),
    ] {
        let configured = match tactic_var(key) {
            Ok(value) => {
                assert!(allowed.contains(&value.as_str()), "invalid {key}={value:?}");
                Some(value)
            }
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => panic!("invalid tactic {key}: {error}"),
        };
        selection.insert(
            key.to_string(),
            json!({
                "resolved":configured.as_deref().unwrap_or(default),
                "configured":configured,
                "resolver":"tactic_plan::tactic_var; plan > environment > branch default"
            }),
        );
    }
    selection
}

struct Lane {
    stream: Arc<CudaStream>,
    workspace: CudaWorkspace,
    spatial: CudaSlice<f32>,
    global: CudaSlice<f32>,
}

fn validate_outputs(out: &CudaOutputsHost, batch: usize) {
    for (name, values, width) in [
        ("policy", out.policy.as_slice(), POLICY),
        ("value", out.value.as_slice(), 3),
        ("misc", out.misc.as_slice(), 10),
        ("moremisc", out.moremisc.as_slice(), 8),
        ("ownership", out.ownership.as_slice(), 361),
    ] {
        assert_eq!(
            values.len(),
            batch * width,
            "raw {name} physical batch shape"
        );
        for (index, value) in values.iter().enumerate() {
            assert!(value.is_finite(), "nonfinite {name}[{index}]={value}");
        }
    }
}

#[test]
fn dump_l2_inner_native() {
    match std::env::var("KATAGO_RUN_L2_INNER_DIAG") {
        Err(std::env::VarError::NotPresent) => {
            eprintln!("skipped: set KATAGO_RUN_L2_INNER_DIAG=1 in an exclusive GPU window");
            return;
        }
        Ok(value) if value == "1" => {}
        value => panic!("KATAGO_RUN_L2_INNER_DIAG must be 1 when set: {value:?}"),
    }
    let batch = integer_env("KATAGO_DUMP_BATCH", 14, &[14]);
    let lane_count = integer_env("KATAGO_DUMP_LANES", 2, &[2]);
    integer_env("KATAGO_DUMP_POSITIONS", POSITIONS, &[POSITIONS]);
    let model_dir = std::env::var_os("KATAGO_TEST_MODEL_DIR")
        .expect("explicit native TF3 dump requires KATAGO_TEST_MODEL_DIR");
    let model_path = PathBuf::from(model_dir).join(MODEL);
    let bytes = std::fs::read(&model_path).expect("read required native TF3 model");
    assert_eq!(
        hex::encode(Sha256::digest(&bytes)),
        MODEL_SHA256,
        "native compressed model SHA"
    );
    let desc = load_model_from_bytes(&bytes, true, true).expect("parse native TF3");
    assert_eq!(desc.sha256, MODEL_SHA256, "native parser model identity");
    let graph = lower_model(&desc).expect("lower native TF3 without ONNX substitution");
    assert_eq!(
        (graph.board_size, graph.trunk_channels, graph.mid_channels),
        (19, 768, 384)
    );
    assert_eq!(
        (graph.num_blocks, graph.num_heads, graph.head_dim),
        (11, 12, 32)
    );

    let directory = std::env::var_os("KATAGO_DUMP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
                "../../target/native_tf3_cuda_b{batch}_l{lane_count}"
            ))
        });
    std::fs::create_dir_all(&directory).expect("create native dump directory");
    assert!(
        std::fs::read_dir(&directory)
            .expect("read dump directory")
            .next()
            .is_none(),
        "dump directory must be empty; keep previous evidence in {}",
        directory.display()
    );
    let executable = std::env::current_exe().expect("test executable path");
    let selection = selected_tactics();
    require_current_tactics(&selection);
    let l2_requested = l2_enabled();
    let positions: Vec<_> = (0..POSITIONS).map(make_position).collect();
    let mut fixture_hash = Sha256::new();
    for position in &positions {
        for &value in position.spatial.iter().chain(&position.global) {
            fixture_hash.update(value.to_le_bytes());
        }
    }
    let rounds = POSITIONS.div_ceil(batch);
    let mut meta = json!({
        "schema":1,"status":"PREPARING_NOT_COMPLETE","backend":"native_tf3_cuda_direct_apply",
        "model_path":model_path,"model_sha256":MODEL_SHA256,"model_name":desc.name,
        "model_version":desc.model_version,"test_executable":executable,
        "test_executable_sha256":sha256_file(&executable).expect("hash test executable"),
        "dump_test_source_sha256":hex::encode(Sha256::digest(include_bytes!("l2_inner_diag.rs"))),
        "l2_cache_source_sha256":hex::encode(Sha256::digest(include_bytes!("l2_inner_diag/l2_cache.rs"))),
        "l2_forward_source_sha256":hex::encode(Sha256::digest(include_bytes!("l2_inner_diag/l2_forward.rs"))),
        "physical_batch":batch,"lanes":lane_count,"positions_per_lane":POSITIONS,
        "batches_per_lane":rounds,"physical_rows":rounds*batch*lane_count,
        "real_rows":POSITIONS*lane_count,"padded_rows":(rounds*batch-POSITIONS)*lane_count,
        "tail_padding":"repeat_last","input_layout":"FP32 NCHW [22,19,19], global[19]",
        "byte_order":"little_endian","position_recipe":"dump_nn_io_cuda::make_position: dump-{i}, i*7%80 legal moves, fill_row_v7, default Rules/MiscNNInputParams",
        "input_hash_order":"position order, each spatial FP32 LE bytes followed by global FP32 LE bytes",
        "input_sha256":hex::encode(fixture_hash.finalize()),
        "position_input_sha256":positions.iter().map(|p|p.sha256.clone()).collect::<Vec<_>>(),
        "output_widths":{"policy":POLICY,"value":3,"misc":10,"moremisc":8,"ownership":361},
        "output_scope":"raw tensors; all six policy channels including pass; no softmax/postprocess",
        "selected_tactics":selection,"installed_plan_id":installed_plan_id(),
        "graph_execution":false,"worker_batching":false,"padding_tactic_applied":false,
        "acceptance":"finite/shape checks only; external B1 consistency and C++ FP32 gates remain required",
        "is_cpp_golden_reference":false,"numeric_tolerances_added":false,
        "execution_marker_verification":"external [cuda-tactic] log verification required; test-only act384 cache window, no arithmetic or workspace data modification",
        "lane_resources":[],"rows":[]
    });
    write_meta(&directory, &meta);

    let rt = CudaRuntime::new().expect("explicit native TF3 dump requires working CUDA");
    let build = backend_build_fingerprint();
    let selected = |key: &str| selection[key]["resolved"].as_str().unwrap();
    let use_lt = selected("KATAGO_CUDA_CUBLASLT") == "1";
    if use_lt {
        assert!(
            rt.cublaslt_handle().is_some(),
            "requested cuBLASLt unavailable"
        );
    }
    if selected("KATAGO_CUDA_DUALFFN") == "1" {
        assert!(
            build.capabilities.dual_ffn,
            "requested DualFFN was not compiled"
        );
    }
    let strict_attention = selected("KATAGO_CUDA_ATTN") == "fa4-strict-b14-r1";
    let attention = if strict_attention && batch == 14 {
        "fa4-strict-b14-r1"
    } else if selected("KATAGO_CUDA_ATTN") == "v3" {
        "v3"
    } else {
        selected("KATAGO_CUDA_ATTN_TILE")
    };
    if attention == "q64" {
        assert!(
            build.capabilities.attention_q64,
            "requested q64 was not compiled"
        );
    } else if attention == "q64-serial" {
        assert!(
            build.capabilities.attention_q64_serial,
            "requested q64 serial was not compiled"
        );
    }
    rt.validate_residual_algo_request()
        .expect("validate actual Lt residual target");
    let device = device_fingerprint(&rt.device).expect("actual CUDA device fingerprint");
    meta["device_fingerprint"] = json!({
        "gpu_name":device.gpu_name,"compute_capability":device.compute_capability,
        "sm_count":device.sm_count,"l2_cache_bytes":device.l2_cache_bytes
    });
    meta["backend_build_fingerprint"] = serde_json::to_value(&build).unwrap();
    let layout = selected("KATAGO_CUDA_GEMM_LAYOUT");
    let k384_layout = match layout {
        "tn" => "tn",
        "nn_k384_b8" if batch < 8 => "tn",
        "nn_k384_b16" if batch < 16 => "tn",
        _ => "nn",
    };
    meta["effective_tactics"] = json!({
        "attention":attention,"k384_weight_layout":k384_layout,
        "stem_global_map":if selected("KATAGO_CUDA_STEM_MAP3D")=="1" {"map3d"} else {"flat"},
        "strict_attention_requested":strict_attention,
        "strict_attention_fallback":strict_attention && batch != 14,
        "cublaslt_requested_and_available":use_lt,
        "dualffn_requested_and_compiled":selected("KATAGO_CUDA_DUALFFN")=="1" && build.capabilities.dual_ffn,
        "residual_selection":selected("KATAGO_CUDA_RESIDUAL_ALGO"),
        "scope":"resolved decisions and capability/target checks; actual branch launches require external tactic markers",
        "direct_apply_ignores_backend_only_tactics":if strict_attention {
            vec!["KATAGO_CUDA_PADBATCH","KATAGO_CUDA_NOPIPELINE"]
        } else {
            vec!["KATAGO_CUDA_NOGRAPH","KATAGO_CUDA_PADBATCH","KATAGO_CUDA_NOPIPELINE"]
        },
        "graph_scope":if strict_attention {"NOGRAPH=1 contract enforced; direct apply performs no graph capture"} else {"direct apply performs no graph capture"}
    });
    let load_stream = rt.device.new_stream().expect("model load stream");
    let model = CudaModel::load(&graph, &rt, &load_stream).expect("upload native CUDA graph");
    rt.device
        .synchronize()
        .expect("model visible on all lane streams");
    let mut lanes = Vec::new();
    let mut stream_ids = Vec::new();
    let mut workspace_ptrs = Vec::new();
    let mut lt_ptrs = Vec::new();
    for lane_index in 0..lane_count {
        let stream = rt.device.new_stream().expect("independent lane stream");
        let workspace =
            CudaWorkspace::new(&stream, &model, batch).expect("physical-batch workspace");
        assert_eq!(workspace.batch(), batch);
        let spatial = stream
            .alloc_zeros::<f32>(batch * SPATIAL)
            .expect("lane spatial");
        let global = stream
            .alloc_zeros::<f32>(batch * GLOBAL)
            .expect("lane global");
        let stream_id = stream.cu_stream() as usize;
        let (workspace_ptr, _) = workspace.out_policy.device_ptr(&stream);
        let (spatial_ptr, _) = spatial.device_ptr(&stream);
        let (global_ptr, _) = global.device_ptr(&stream);
        assert!(
            !stream_ids.contains(&stream_id),
            "lanes must have distinct streams"
        );
        assert!(
            !workspace_ptrs.contains(&workspace_ptr),
            "lanes must have distinct workspaces"
        );
        stream_ids.push(stream_id);
        workspace_ptrs.push(workspace_ptr);
        let (lt_ptr, lt_bytes) = if use_lt {
            // The pointer helper's second tuple slot is reserved (currently 0),
            // not its allocation size. Query the public size accessor instead.
            let (pointer, _) = rt.cublaslt_workspace_ptr(&stream);
            (pointer, rt.cublaslt_workspace_len())
        } else {
            (0, 0)
        };
        if use_lt {
            assert!(lt_ptr != 0 && lt_bytes > 0, "lane Lt workspace allocation");
            assert!(
                !lt_ptrs.contains(&lt_ptr),
                "lanes must have distinct Lt workspaces"
            );
            lt_ptrs.push(lt_ptr);
        }
        meta["lane_resources"].as_array_mut().unwrap().push(json!({
            "lane":lane_index,"stream_id":stream_id,"workspace_batch":workspace.batch(),
            "output_policy_device_ptr":workspace_ptr,"spatial_device_ptr":spatial_ptr,
            "global_device_ptr":global_ptr,"cublaslt_workspace_device_ptr":lt_ptr,
            "cublaslt_workspace_bytes":lt_bytes
        }));
        std::fs::create_dir(directory.join(format!("lane{lane_index}")))
            .expect("lane dump directory");
        lanes.push(Lane {
            stream,
            workspace,
            spatial,
            global,
        });
    }
    let mut cache_scope = l2_cache::CacheScope::begin(&rt, &lanes, l2_requested)
        .expect("initialize fixed two-stream L2 diagnostic");
    meta["l2_diagnostic"] = cache_scope.metadata();
    meta["status"] = json!("RUNNING_NOT_COMPLETE");
    meta["submission"] = json!({"host_threads_per_round":lane_count,
        "streams":"independent non-default streams; shared read-only model/runtime",
        "synchronization":"all scoped apply submissions joined, then one device sync before any output download",
        "gpu_overlap_measured":false});
    write_meta(&directory, &meta);

    for round in 0..rounds {
        let first = round * batch;
        let real_rows = batch.min(POSITIONS - first);
        let position_indices: Vec<_> = (0..batch)
            .map(|row| first + row.min(real_rows - 1))
            .collect();
        let spatial: Vec<_> = position_indices
            .iter()
            .flat_map(|&i| positions[i].spatial.iter().copied())
            .collect();
        let global: Vec<_> = position_indices
            .iter()
            .flat_map(|&i| positions[i].global.iter().copied())
            .collect();
        for lane in &mut lanes {
            lane.stream
                .memcpy_htod(&spatial, &mut lane.spatial)
                .expect("upload physical spatial rows");
            lane.stream
                .memcpy_htod(&global, &mut lane.global)
                .expect("upload physical global rows");
        }
        rt.device.synchronize().expect("all lane inputs ready");
        std::thread::scope(|scope| {
            let jobs: Vec<_> = lanes
                .iter_mut()
                .map(|lane| {
                    let rt = &rt;
                    let model = &model;
                    scope.spawn(move || {
                        rt.device
                            .bind_to_thread()
                            .map_err(|e| format!("bind lane CUDA context: {e}"))?;
                        model.apply(
                            rt,
                            &lane.stream,
                            &mut lane.workspace,
                            &lane.spatial,
                            &lane.global,
                        )
                    })
                })
                .collect();
            for job in jobs {
                job.join()
                    .expect("lane submission panicked")
                    .expect("native physical-batch apply failed");
            }
        });
        rt.device
            .synchronize()
            .expect("all concurrent lane forwards complete");
        for (lane_index, lane) in lanes.iter().enumerate() {
            let out = lane
                .workspace
                .to_host(&lane.stream)
                .expect("download raw lane outputs");
            lane.stream.synchronize().expect("lane downloads complete");
            validate_outputs(&out, batch);
            for (row, &position) in position_indices.iter().enumerate() {
                let padding = row >= real_rows;
                let prefix = if padding {
                    format!("lane{lane_index}/pad_batch{round}_row{row}")
                } else {
                    format!("lane{lane_index}/pos{position}")
                };
                let row_spatial = &spatial[row * SPATIAL..(row + 1) * SPATIAL];
                let row_global = &global[row * GLOBAL..(row + 1) * GLOBAL];
                let row_input_hash = input_hash(row_spatial, row_global);
                assert_eq!(
                    row_input_hash, positions[position].sha256,
                    "row input identity"
                );
                let mut files = BTreeMap::new();
                for (name, values) in [
                    ("spatial", row_spatial),
                    ("global", row_global),
                    ("policy", &out.policy[row * POLICY..(row + 1) * POLICY]),
                    ("value", &out.value[row * 3..(row + 1) * 3]),
                    ("misc", &out.misc[row * 10..(row + 1) * 10]),
                    ("moremisc", &out.moremisc[row * 8..(row + 1) * 8]),
                    ("ownership", &out.ownership[row * 361..(row + 1) * 361]),
                ] {
                    let relative_path = format!("{prefix}_{name}.bin");
                    let sha = write_f32(&directory.join(&relative_path), values);
                    files.insert(
                        name,
                        json!({"path":relative_path,"elements":values.len(),"sha256":sha}),
                    );
                }
                meta["rows"].as_array_mut().unwrap().push(json!({
                    "lane":lane_index,"batch_index":round,"physical_row":row,
                    "position":position,"padding":padding,"input_sha256":row_input_hash,"files":files
                }));
            }
        }
        meta["completed_rounds"] = json!(round + 1);
        write_meta(&directory, &meta);
    }
    assert_eq!(
        meta["rows"].as_array().unwrap().len(),
        rounds * batch * lane_count
    );
    match cache_scope.finish() {
        Ok(value) => meta["l2_diagnostic"] = value,
        Err(error) => {
            meta["l2_diagnostic"] = cache_scope.metadata();
            meta["status"] = json!("FAILED_L2_CLEANUP");
            write_meta(&directory, &meta);
            panic!("L2 cleanup: {error}");
        }
    }
    meta["status"] = json!("COMPLETE_FINITE_RAW_OUTPUTS_EXTERNAL_NUMERIC_GATES_PENDING");
    meta["all_raw_outputs_finite"] = json!(true);
    write_meta(&directory, &meta);
    println!(
        "NATIVE_TF3_DUMP physical_batch={batch} lanes={lane_count} positions_per_lane={POSITIONS} path={}",
        directory.display()
    );
}

fn l2_enabled() -> bool {
    match std::env::var("KATAGO_DUMP_L2_INNER") {
        Ok(value) if value == "0" => false,
        Ok(value) if value == "1" => true,
        value => panic!("KATAGO_DUMP_L2_INNER must explicitly be 0 or 1: {value:?}"),
    }
}

fn require_current_tactics(selection: &BTreeMap<String, Value>) {
    for (key, expected) in [
        ("KATAGO_CUDA_CUBLASLT", "1"),
        ("KATAGO_CUDA_DUALFFN", "1"),
        ("KATAGO_CUDA_SPLITK", "0"),
        ("KATAGO_CUDA_STEM_MAP3D", "1"),
        ("KATAGO_CUDA_T32", "0"),
        ("KATAGO_CUDA_T64N32", "0"),
        ("KATAGO_CUDA_FUSION", "none"),
        ("KATAGO_CUDA_CUBLASLT_RANK", "heuristic"),
        ("KATAGO_CUDA_RESIDUAL_ALGO", "tf3_5070ti_r1"),
        ("KATAGO_CUDA_GEMM_LAYOUT", "tn"),
        ("KATAGO_CUDA_ATTN_TILE", "q64-serial"),
        ("KATAGO_CUDA_ATTN", "fa4-strict-b14-r1"),
        ("KATAGO_CUDA_RMS", "warp4"),
        ("KATAGO_CUDA_NOGRAPH", "1"),
        ("KATAGO_CUDA_PADBATCH", "0"),
        ("KATAGO_CUDA_NOPIPELINE", "0"),
    ] {
        assert_eq!(
            selection[key]["resolved"].as_str(),
            Some(expected),
            "fixed L2 diagnostic tactic {key}"
        );
    }
    for key in ["KATAGO_CUDA_PROFILE", "KATAGO_CUDA_DEBUG_LAYER"] {
        assert!(
            std::env::var_os(key).is_none(),
            "{key} must be absent in this diagnostic"
        );
    }
}

#[test]
fn fixed_l2_inner_forward() {
    l2_forward::fixed_l2_inner_forward();
}
