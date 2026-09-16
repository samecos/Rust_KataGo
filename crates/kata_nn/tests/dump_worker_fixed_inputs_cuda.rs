//! Frozen Worker-request inputs through one native TF3 CUDA lane at B1/B14/B16.
//!
//! Opt in: KATAGO_RUN_WORKER_FIXED_INPUT_DUMP=1, KATAGO_TEST_MODEL_DIR=<models>.
//! Required baseline overrides: KATAGO_CUDA_DUALFFN=1,
//! KATAGO_CUDA_ATTN_TILE=q64-serial, KATAGO_CUDA_NOGRAPH=1. Other tactics must
//! resolve to the original G3 TN/heuristic baseline. No plan is loaded here.
//! Optional KATAGO_WORKER_INPUT_FIXTURE_DIR selects the CPU-frozen fixtures;
//! KATAGO_WORKER_FIXED_DUMP_DIR selects a fresh directory below workspace target.
//!
//! This is raw-output evidence, not a C++ golden, accuracy gate or benchmark.
//! It does not replay requests or apply another symmetry. Every physical row is
//! the exact already-transformed feature row read from the frozen CPU fixture.
//! Build with `--features cuda`; a feature-disabled binary has zero such tests
//! and must not be reported as a successful export. Require one test and the
//! final PASS_RAW_DUMP_COMPLETE_NOT_CERTIFIED metadata/marker.
#![cfg(feature = "cuda")]

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use kata_nn::backends::cuda::{CudaRuntime, backend_build_fingerprint, device_fingerprint};
use kata_nn::backends::cuda_exec::{CudaModel, CudaWorkspace};
use kata_nn::model_parser::load_model_from_bytes;
use kata_nn::native_model::lower_model;
use kata_nn::tactic_plan::{installed_plan_id, sha256_file, tactic_var};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const MODEL: &str = "kata1-tf3-b11c768-s11001M-d5973M.bin.gz";
const MODEL_SHA: &str = "1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6";
const INPUT_SHA: &str = "f9caf09c4fbf32b48736dda5531a3292bdacccef07db6a9b974678c09ab22911";
const SPATIAL: usize = 22 * 361;
const GLOBAL: usize = 19;
const POLICY: usize = 6 * 362;

fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn bytes_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn write_new(path: &Path, bytes: &[u8]) {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap_or_else(|e| panic!("refuse overwrite/cannot create {}: {e}", path.display()));
    file.write_all(bytes).expect("write complete evidence file");
}

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).expect("read required frozen manifest"))
        .expect("parse frozen JSON")
}

fn write_meta(dir: &Path, meta: &Value) {
    // Only this run's incremental metadata is replaced; its directory was fresh.
    fs::write(
        dir.join("meta.json"),
        serde_json::to_vec_pretty(meta).unwrap(),
    )
    .expect("write diagnostic metadata");
}

fn resolve(workspace: &Path, env: &str, default: &str) -> PathBuf {
    let path = std::env::var_os(env)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default));
    if path.is_absolute() {
        path
    } else {
        workspace.join(path)
    }
}

fn frozen_tensor(
    dir: &Path,
    description: &Value,
    file: &str,
    shape: &[usize],
    layout: &str,
) -> Vec<u8> {
    assert_eq!(description["file"], file, "registered tensor filename");
    assert_eq!(
        description["shape"],
        json!(shape),
        "registered tensor shape"
    );
    assert_eq!(description["layout"], layout);
    assert_eq!(description["dtype"], "float32");
    assert_eq!(description["byte_order"], "little-endian");
    let bytes = fs::read(dir.join(file)).expect("read frozen FP32 tensor");
    assert_eq!(bytes.len(), shape.iter().product::<usize>() * 4);
    assert_eq!(description["bytes"], json!(bytes.len()));
    assert_eq!(
        description["sha256"],
        hash(&bytes),
        "actual tensor SHA differs from manifest"
    );
    bytes
}

fn load_fixtures(dir: &Path) -> (Vec<f32>, Vec<f32>, Value) {
    let manifest = read_json(&dir.join("manifest.json"));
    assert_eq!(manifest["status"], "PASS_CPU_INPUT_FREEZE");
    assert_eq!(manifest["model_sha256_from_requests"], MODEL_SHA);
    assert_eq!(manifest["gpu_called"], false);
    let cases = manifest["cases"].as_array().expect("frozen case array");
    assert_eq!(cases.len(), 2);
    let mut input: Option<(Vec<u8>, Vec<u8>)> = None;
    let mut mapping = Vec::new();
    for (task, symmetry, dirname, request_sha) in [
        (
            11u64,
            2,
            "task-011-sym-2",
            "d93435970300d3f50e60df033f87e2fd33e3a7f41c7ffef1768622c03e9e0107",
        ),
        (
            14u64,
            5,
            "task-014-sym-5",
            "e21017a181b6a00310907aa7bd7e4fb53bb854de0e1a027b61d2ba9e6630737a",
        ),
    ] {
        let registered: Vec<_> = cases.iter().filter(|c| c["task_id"] == task).collect();
        assert_eq!(registered.len(), 1, "unique task mapping");
        let registered = registered[0];
        assert_eq!(registered["directory"], dirname);
        assert_eq!(registered["symmetry"], symmetry);
        let child = dir.join(dirname);
        let case_bytes = fs::read(child.join("manifest.json")).expect("case manifest");
        assert_eq!(registered["manifest_sha256"], hash(&case_bytes));
        let case: Value = serde_json::from_slice(&case_bytes).unwrap();
        assert_eq!(case["request"]["task_id"], task);
        assert_eq!(case["request"]["parameters"]["symmetry"], symmetry);
        assert_eq!(case["request"]["model_sha256"], MODEL_SHA);
        assert_eq!(case["feature_contract"]["symmetry"], symmetry);
        assert_eq!(case["feature_contract"]["inputs_version"], 7);
        assert_eq!(case["feature_contract"]["use_nhwc"], false);
        assert_eq!(case["source_request"]["sha256"], request_sha);
        assert_eq!(
            hash(&fs::read(child.join("request.pb")).unwrap()),
            request_sha
        );
        let spatial = frozen_tensor(
            &child,
            &case["tensors"]["post_spatial"],
            "post_spatial.f32le",
            &[1, 22, 19, 19],
            "NCHW",
        );
        let global = frozen_tensor(
            &child,
            &case["tensors"]["post_global"],
            "post_global.f32le",
            &[1, 19],
            "NC",
        );
        let mut pair = spatial.clone();
        pair.extend_from_slice(&global);
        assert_eq!(
            hash(&pair),
            INPUT_SHA,
            "exact G3 post-symmetry tensor identity"
        );
        assert_eq!(
            case["actual_tensor_pair_sha256"]["post_symmetry"],
            INPUT_SHA
        );
        if let Some((previous_spatial, previous_global)) = &input {
            assert_eq!(
                &spatial, previous_spatial,
                "dedup requires actual spatial byte equality"
            );
            assert_eq!(
                &global, previous_global,
                "dedup requires actual global byte equality"
            );
        } else {
            input = Some((spatial, global));
        }
        mapping.push(json!({"task_id": task, "symmetry": symmetry, "inverse_symmetry": if symmetry == 5 { 6 } else { symmetry },
            "case_manifest": child.join("manifest.json"), "case_manifest_sha256": hash(&case_bytes),
            "request_sha256": request_sha, "request": case["request"], "unique_input_index": 0,
            "actual_post_tensor_pair_sha256": INPUT_SHA}));
    }
    let (spatial, global) = input.unwrap();
    let decode = |bytes: &[u8]| -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|b| {
                let value = f32::from_le_bytes(b.try_into().unwrap());
                assert!(value.is_finite(), "nonfinite frozen input");
                value
            })
            .collect()
    };
    (
        decode(&spatial),
        decode(&global),
        json!({"directory": dir,
        "manifest_sha256": sha256_file(&dir.join("manifest.json")).unwrap(),
        "source_platform": manifest["platform"], "unique_inputs": 1, "case_mapping": mapping,
        "deduplication": "both spatial/global byte arrays compared exactly, not semantic input_hash",
        "orientation": "post-symmetry backend input; no additional transform is performed"}),
    )
}

fn baseline_tactics() -> BTreeMap<String, Value> {
    assert!(
        installed_plan_id().is_none(),
        "this standalone test does not install a plan"
    );
    let mut selection = BTreeMap::new();
    for (key, default, expected) in [
        ("KATAGO_CUDA_CUBLASLT", "1", "1"),
        ("KATAGO_CUDA_DUALFFN", "0", "1"),
        ("KATAGO_CUDA_CUBLASLT_RANK", "heuristic", "heuristic"),
        ("KATAGO_CUDA_RESIDUAL_ALGO", "heuristic", "heuristic"),
        ("KATAGO_CUDA_GEMM_LAYOUT", "tn", "tn"),
        ("KATAGO_CUDA_ATTN", "fa2", "fa2"),
        ("KATAGO_CUDA_ATTN_TILE", "q128", "q64-serial"),
        ("KATAGO_CUDA_NOGRAPH", "0", "1"),
        ("KATAGO_CUDA_SPLITK", "0", "0"),
        ("KATAGO_CUDA_T32", "0", "0"),
        ("KATAGO_CUDA_T64N32", "0", "0"),
        ("KATAGO_CUDA_FUSION", "none", "none"),
        ("KATAGO_CUDA_RMS", "warp4", "warp4"),
        ("KATAGO_CUDA_PADBATCH", "0", "0"),
        ("KATAGO_CUDA_NOPIPELINE", "0", "0"),
    ] {
        let configured = match tactic_var(key) {
            Ok(v) => Some(v),
            Err(std::env::VarError::NotPresent) => None,
            Err(e) => panic!("invalid {key}: {e}"),
        };
        let effective = configured.as_deref().unwrap_or(default);
        assert_eq!(
            effective, expected,
            "fixed-input G3 baseline requires {key}={expected}"
        );
        selection.insert(
            key.to_string(),
            json!({"configured": configured, "resolved": effective,
            "resolver": "tactic_plan::tactic_var then production branch default"}),
        );
    }
    for key in [
        "KATAGO_CUDA_PROFILE",
        "KATAGO_CUDA_DUMP_INPUT",
        "KATAGO_CUDA_DEBUG_LAYER",
    ] {
        assert!(
            std::env::var_os(key).is_none(),
            "clear {key}; initial probe has no layer tracing"
        );
    }
    selection
}

fn tensor_output(dir: &Path, name: &str, values: &[f32], shape: &[usize]) -> Value {
    assert_eq!(values.len(), shape.iter().product::<usize>());
    let bytes = bytes_f32(values);
    write_new(&dir.join(name), &bytes);
    json!({"file": name, "sha256": hash(&bytes), "shape": shape, "elements": values.len(),
           "bytes": bytes.len(), "dtype": "float32", "byte_order": "little-endian",
           "raw_bits_preserved": true, "finite": values.iter().all(|v| v.is_finite())})
}

fn raw_policy_top4(values: &[f32]) -> Vec<Value> {
    (0..6).map(|channel| {
        let logits = &values[channel * 362..(channel + 1) * 362];
        let mut indices: Vec<_> = (0..362).collect();
        indices.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        json!({"channel": channel, "scope": "raw backend orientation; no legality mask, softmax or optimism mixing",
            "top4": indices[..4].iter().map(|&index| json!({"index": index, "value": logits[index],
                "bits_hex": format!("{:08x}", logits[index].to_bits())})).collect::<Vec<_>>()})
    }).collect()
}

#[test]
fn dump_worker_fixed_inputs_cuda() {
    match std::env::var("KATAGO_RUN_WORKER_FIXED_INPUT_DUMP").as_deref() {
        Err(std::env::VarError::NotPresent) => {
            eprintln!("SKIP: set KATAGO_RUN_WORKER_FIXED_INPUT_DUMP=1 in an exclusive GPU window");
            return;
        }
        Ok("1") => {}
        value => panic!("KATAGO_RUN_WORKER_FIXED_INPUT_DUMP must be 1 when set: {value:?}"),
    }
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let fixtures = resolve(
        &workspace,
        "KATAGO_WORKER_INPUT_FIXTURE_DIR",
        "target/fork-parity-20260908/g3-worker-input-fixtures-r1",
    );
    assert!(
        fixtures.join("manifest.json").is_file(),
        "opted-in run requires the actual frozen CPU fixtures; refusing silent skip"
    );
    let (spatial_row, global_row, fixture_meta) = load_fixtures(&fixtures);
    assert_eq!((spatial_row.len(), global_row.len()), (SPATIAL, GLOBAL));
    let selection = baseline_tactics();
    let model_path = PathBuf::from(
        std::env::var_os("KATAGO_TEST_MODEL_DIR")
            .expect("explicit dump requires KATAGO_TEST_MODEL_DIR"),
    )
    .join(MODEL);
    let model_bytes = fs::read(&model_path).expect("read required native TF3 model");
    assert_eq!(hash(&model_bytes), MODEL_SHA);
    let desc = load_model_from_bytes(&model_bytes, true, true).expect("parse native TF3");
    assert_eq!(desc.sha256, MODEL_SHA);
    let graph = lower_model(&desc).expect("lower native TF3 without ONNX substitution");
    assert_eq!(
        (
            graph.board_size,
            graph.trunk_channels,
            graph.mid_channels,
            graph.num_blocks,
            graph.num_heads,
            graph.head_dim
        ),
        (19, 768, 384, 11, 12, 32)
    );
    let dir = resolve(
        &workspace,
        "KATAGO_WORKER_FIXED_DUMP_DIR",
        "target/fork-parity-20260908/g3-fixed-input-cuda-r1",
    );
    assert!(
        !dir.exists(),
        "fresh output directory required: {}",
        dir.display()
    );
    assert!(
        dir.parent()
            .unwrap()
            .canonicalize()
            .unwrap()
            .starts_with(workspace.join("target").canonicalize().unwrap()),
        "output must stay below workspace target"
    );
    fs::create_dir(&dir).unwrap();
    let exe = std::env::current_exe().unwrap();
    let mut meta = json!({"schema": 1, "status": "PREPARING_NOT_COMPLETE",
        "model_path": model_path, "model_sha256": MODEL_SHA, "model_version": desc.model_version,
        "test_executable": exe, "test_executable_sha256": sha256_file(&exe).unwrap(),
        "dump_test_source_sha256": hash(include_bytes!("dump_worker_fixed_inputs_cuda.rs")),
        "platform": std::env::consts::OS, "fixtures": fixture_meta, "actual_input_pair_sha256": INPUT_SHA,
        "selected_tactics": selection, "installed_plan_id": installed_plan_id(), "plan_loading_test": false,
        "registered_batch_order": [1, 14, 16], "lanes": 1, "physical_rows": 31,
        "submission": "one stream, one synchronous direct apply per batch; no graph, warmup, scheduler or timing",
        "replication": "every physical row is identical spatial/global FP32 bytes; no backend padding and no repeated symmetry",
        "policy_layout": "[batch,6,362], including each channel's pass at index361",
        "output_widths": {"policy": POLICY,"value": 3,"misc": 10,"moremisc": 8,"ownership": 361},
        "scope": "complete raw output evidence only; no C++ golden, acceptance tolerances or performance claim",
        "is_cpp_golden_reference": false, "is_performance_measurement": false,
        "execution_marker_verification": "external [cuda-tactic] log verification required",
        "batches": []});
    write_meta(&dir, &meta);
    let rt = CudaRuntime::new().expect("opted-in diagnostic requires working CUDA");
    let build = backend_build_fingerprint();
    assert_eq!(build.fp16_encoding_revision, Some(1));
    assert_eq!(build.cublaslt_version, Some(130600));
    assert!(build.capabilities.dual_ffn && build.capabilities.attention_q64_serial);
    assert!(rt.cublaslt_handle().is_some(), "requested Lt unavailable");
    rt.validate_residual_algo_request()
        .expect("validate heuristic residual request");
    let device = device_fingerprint(&rt.device).unwrap();
    assert_eq!(device.gpu_name, "NVIDIA GeForce RTX 5070 Ti");
    assert_eq!(device.compute_capability, "12.0");
    assert_eq!((device.sm_count, device.l2_cache_bytes), (70, 50331648));
    meta["backend_build_fingerprint"] = serde_json::to_value(&build).unwrap();
    meta["device_fingerprint"] = json!({"gpu_name": device.gpu_name, "compute_capability": device.compute_capability,
        "sm_count": device.sm_count, "l2_cache_bytes": device.l2_cache_bytes});
    let stream = rt.device.new_stream().expect("single lane stream");
    let model = CudaModel::load(&graph, &rt, &stream).expect("load native model onto lane stream");
    stream.synchronize().expect("model upload complete");
    meta["stream_id"] = json!(stream.cu_stream() as usize);
    meta["status"] = json!("RUNNING_NOT_COMPLETE");
    write_meta(&dir, &meta);
    let mut all_finite = true;
    for batch in [1usize, 14, 16] {
        let batch_dir = dir.join(format!("b{batch}"));
        fs::create_dir(&batch_dir).unwrap();
        let spatial = spatial_row.repeat(batch);
        let global = global_row.repeat(batch);
        let spatial_file = tensor_output(
            &batch_dir,
            "input_spatial.f32le",
            &spatial,
            &[batch, 22, 19, 19],
        );
        let global_file = tensor_output(&batch_dir, "input_global.f32le", &global, &[batch, 19]);
        let mut gpu_spatial = stream.alloc_zeros::<f32>(spatial.len()).unwrap();
        let mut gpu_global = stream.alloc_zeros::<f32>(global.len()).unwrap();
        stream.memcpy_htod(&spatial, &mut gpu_spatial).unwrap();
        stream.memcpy_htod(&global, &mut gpu_global).unwrap();
        let mut scratch =
            CudaWorkspace::new(&stream, &model, batch).expect("physical batch workspace");
        assert_eq!(scratch.batch(), batch);
        stream.synchronize().expect("fixed inputs uploaded");
        eprintln!(
            "[worker-fixed-input] phase=before_apply physical_batch={batch} lane=0 rows={batch} input_pair_sha256={INPUT_SHA} layout=nchw symmetry=already_applied"
        );
        model
            .apply(&rt, &stream, &mut scratch, &gpu_spatial, &gpu_global)
            .expect("fixed-input native TF3 forward");
        stream.synchronize().expect("fixed-input forward complete");
        let out = scratch
            .to_host(&stream)
            .expect("download complete raw outputs");
        stream.synchronize().expect("output download complete");
        let heads = [
            ("policy", out.policy.as_slice(), POLICY),
            ("value", out.value.as_slice(), 3),
            ("misc", out.misc.as_slice(), 10),
            ("moremisc", out.moremisc.as_slice(), 8),
            ("ownership", out.ownership.as_slice(), 361),
        ];
        let mut batch_heads = BTreeMap::new();
        for (name, values, width) in heads {
            assert_eq!(values.len(), batch * width, "full {name} shape");
            all_finite &= values.iter().all(|v| v.is_finite());
            batch_heads.insert(
                name,
                tensor_output(
                    &batch_dir,
                    &format!("{name}.f32le"),
                    values,
                    &[batch, width],
                ),
            );
        }
        let mut rows = Vec::new();
        for row in 0..batch {
            let row_dir = batch_dir.join(format!("row{row:02}"));
            fs::create_dir(&row_dir).unwrap();
            let mut files = BTreeMap::new();
            let mut row0_equal = BTreeMap::new();
            for (name, values, width) in heads {
                let actual = &values[row * width..(row + 1) * width];
                files.insert(
                    name,
                    tensor_output(&row_dir, &format!("{name}.f32le"), actual, &[width]),
                );
                row0_equal.insert(name, bytes_f32(actual) == bytes_f32(&values[..width]));
            }
            rows.push(json!({"physical_row": row, "lane": 0, "unique_input_index": 0, "maps_to_tasks": [11,14],
                "actual_input_pair_sha256": INPUT_SHA, "files": files, "bitwise_equal_to_row0": row0_equal,
                "raw_policy_channel_top4": raw_policy_top4(&out.policy[row * POLICY..(row + 1) * POLICY])}));
        }
        meta["batches"].as_array_mut().unwrap().push(json!({"physical_batch": batch, "lane": 0,
            "directory": format!("b{batch}"), "input_spatial": spatial_file, "input_global": global_file,
            "heads": batch_heads, "rows": rows}));
        write_meta(&dir, &meta);
        eprintln!(
            "[worker-fixed-input] phase=complete physical_batch={batch} lane=0 rows={batch} heads=policy6_value3_misc10_moremisc8_ownership361 input_pair_sha256={INPUT_SHA}"
        );
    }
    assert_eq!(
        sha256_file(&model_path).unwrap(),
        MODEL_SHA,
        "model changed during dump"
    );
    meta["status"] = json!(if all_finite {
        "PASS_RAW_DUMP_COMPLETE_NOT_CERTIFIED"
    } else {
        "FAIL_NONFINITE_RAW_BITS_PRESERVED"
    });
    write_meta(&dir, &meta);
    assert!(all_finite, "nonfinite outputs retained in raw dump");
    eprintln!(
        "PASS_RAW_DUMP_COMPLETE_NOT_CERTIFIED: {}",
        dir.join("meta.json").display()
    );
}
