//! Opt-in, whole-network first-down tensor injection diagnostic.
//! Uses only public LayerGraph/CudaModel/CudaWorkspace APIs. No production edit.
//! Full versus split controls must match every output bit. Frozen r4 B1 down
//! output is repeated at B14; this is an intervention, not an algorithm rollout.
//! KATAGO_RUN_WORKER_FIRST_DOWN_INJECTION=1 enables this WSL-only diagnostic.
//! KATAGO_WORKER_FIRST_DOWN_INJECTION_DIR must be a fresh directory under target.
//! No Worker scheduler, C++ golden, 128-request gate or performance certification.
#![cfg(feature = "cuda")]
use cudarc::driver::{CudaStream, DevicePtr};
use kata_nn::onnx_parser::Layer;
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use kata_nn::backends::cuda::{backend_build_fingerprint, device_fingerprint, CudaRuntime};
use kata_nn::backends::cuda_exec::{CudaModel, CudaOutputsHost, CudaWorkspace};
use kata_nn::model_parser::load_model_from_bytes;
use kata_nn::native_model::lower_model;
use kata_nn::tactic_plan::{installed_plan_id, sha256_file, tactic_var};
use serde_json::{json, Value};
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

fn write_report(dir: &Path, meta: &Value) {
    // Only this run's incremental metadata is replaced; its directory was fresh.
    fs::write(
        dir.join("report.json"),
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

const FIXTURE_SHA: &str = "408ca276bf1e87acc39ba7d386a363eed65536f723915cf48212d22fd8e0ec80";
const R4: &str = "target/fork-parity-20260908/g3-linear-down-lt-replay-r1/cross-shape-r4";
const R4_REPORT_SHA: &str = "88325ab54253ceae91bd52df5a61a69a3d87ffcf8d579df368a5b600ef167969";
const R4_BINARY_SHA: &str = "2b98d44b0e6156798a487ad33188c100e9db992a2f39ab8bd40d1b67899443aa";
const PREFIX_SHA: &str = "a16ab88cbf023926140a3bb9c352e8006ac119c47978b056fc315dc7fb6fa849";

fn record(path: &Path) -> Value {
    json!({"path":path,"sha256":sha256_file(path).unwrap()})
}

fn bound(path: &Path, expected: &str, bytes: usize) -> Vec<u8> {
    let raw = fs::read(path).expect("read frozen raw tensor");
    assert_eq!(raw.len(), bytes, "frozen tensor byte count");
    assert_eq!(hash(&raw), expected, "frozen tensor SHA");
    raw
}

fn decode(raw: &[u8]) -> Vec<f32> {
    assert_eq!(raw.len() % 4, 0);
    raw.chunks_exact(4)
        .map(|b| {
            let f = f32::from_le_bytes(b.try_into().unwrap());
            assert!(f.is_finite(), "nonfinite frozen tensor");
            f
        })
        .collect()
}

fn export(dir: &Path, file: &str, values: &[f32], shape: &[usize]) -> Value {
    assert_eq!(values.len(), shape.iter().product::<usize>());
    let raw = bytes_f32(values);
    write_new(&dir.join(file), &raw);
    json!({"file":file,"sha256":hash(&raw),"bytes":raw.len(),"elements":values.len(),
        "dtype":"float32","byte_order":"little-endian","shape":shape,
        "finite":values.iter().all(|f|f.is_finite()),"raw_bits_preserved":true})
}

fn heads(out: &CudaOutputsHost) -> [(&'static str, &[f32], usize); 5] {
    [
        ("policy", &out.policy, POLICY),
        ("value", &out.value, 3),
        ("misc", &out.misc, 10),
        ("moremisc", &out.moremisc, 8),
        ("ownership", &out.ownership, 361),
    ]
}

fn compare(a: &CudaOutputsHost, b: &CudaOutputsHost) -> Value {
    let mut fields = serde_json::Map::new();
    let mut all_equal = true;
    for ((name, x, _), (other, y, _)) in heads(a).into_iter().zip(heads(b)) {
        assert_eq!(name, other);
        assert_eq!(x.len(), y.len());
        let mismatches = x
            .iter()
            .zip(y)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        all_equal &= mismatches == 0;
        fields.insert(
            name.into(),
            json!({"elements":x.len(),"bit_mismatches":mismatches,
            "all_equal":mismatches==0}),
        );
    }
    json!({"all_equal":all_equal,"heads":fields,"scope":"all physical rows and all five raw heads, FP32 bits"})
}

struct FrozenInjection {
    b1: Vec<f32>,
    original: BTreeMap<usize, Vec<u8>>,
    gated: BTreeMap<usize, Vec<u8>>,
    report: Value,
    provenance: Value,
}

fn frozen_injection(root: &Path) -> FrozenInjection {
    let dir = root.join(R4);
    let report_path = dir.join("run-wsl-r1/raw/report.json");
    assert_eq!(sha256_file(&report_path).unwrap(), R4_REPORT_SHA);
    let report = read_json(&report_path);
    assert_eq!(
        report["status"],
        "COMPLETE_CROSS_SHAPE_DIAGNOSTIC_NOT_CERTIFIED"
    );
    assert_eq!(report["controls_bitwise_pass"], true);
    assert_eq!(report["diagnostic_complete"], true);
    assert_eq!(report["model"]["sha256"], MODEL_SHA);
    assert_eq!(report["test_executable"]["sha256"], R4_BINARY_SHA);
    let archived_binary = dir.join("passed-r4-snapshot/probe-passed-r4-wsl");
    assert_eq!(sha256_file(&archived_binary).unwrap(), R4_BINARY_SHA);
    assert_eq!(&fs::read(&archived_binary).unwrap()[..4], b"\x7fELF");
    let prefix_dir = root.join("target/fork-parity-20260908/g3-prefix-trace-wsl-r1");
    let prefix_path = prefix_dir.join("meta.json");
    assert_eq!(sha256_file(&prefix_path).unwrap(), PREFIX_SHA);
    assert_eq!(report["reference"]["sha256"], PREFIX_SHA);
    let prefix = read_json(&prefix_path);
    assert_eq!(prefix["actual_input_pair_sha256"], INPUT_SHA);
    assert_eq!(prefix["model_sha256"], MODEL_SHA);
    let mut original = BTreeMap::new();
    let mut gated = BTreeMap::new();
    let mut source_records = Vec::new();
    for batch in [1usize, 14] {
        let runs: Vec<_> = report["runs"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["selection_batch"] == batch && r["execution_batch"] == batch)
            .collect();
        assert_eq!(runs.len(), 1);
        let run = runs[0];
        assert_eq!(run["status"], "CONTROL_EXECUTED");
        assert_eq!(run["lt"]["status"], "PASS_REPLAY_EXECUTED");
        assert_eq!(run["lt"]["executed"], true);
        assert_eq!(run["operands_unchanged"], true);
        let file = dir.join(format!(
            "run-wsl-r1/raw/select-b{batch}-execute-b{batch}/output.f32le"
        ));
        let bytes = bound(
            &file,
            run["output"]["sha256"].as_str().unwrap(),
            batch * 361 * 384 * 4,
        );
        original.insert(batch, bytes);
        let input_file = prefix_dir.join(format!("prefix2-b{batch}/gated768.f16le"));
        gated.insert(
            batch,
            bound(
                &input_file,
                run["input"]["sha256"].as_str().unwrap(),
                batch * 361 * 768 * 2,
            ),
        );
        source_records.push(
            json!({"physical_batch":batch,"output":record(&file),"input":record(&input_file)}),
        );
    }
    assert_eq!(gated[&14], gated[&1].repeat(14));
    let cross: Vec<_> = report["runs"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["selection_batch"] == 1 && r["execution_batch"] == 14)
        .collect();
    assert_eq!(cross.len(), 1);
    let cross = cross[0];
    assert_eq!(cross["status"], "CROSS_EXECUTED_DESCRIPTIVE");
    assert_eq!(cross["lt"]["status"], "PASS_REPLAY_EXECUTED");
    assert_eq!(cross["vs_r2_b1_every_board"]["bitwise_equal"], true);
    let cross_path = dir.join("run-wsl-r1/raw/select-b1-execute-b14/output.f32le");
    let cross_bytes = bound(
        &cross_path,
        cross["output"]["sha256"].as_str().unwrap(),
        14 * 361 * 384 * 4,
    );
    assert_eq!(
        cross_bytes,
        original[&1].repeat(14),
        "actual r4 cross bytes equal repeated B1 source"
    );
    let b1 = decode(&original[&1]);
    let provenance = json!({"raw_report":record(&report_path),"archived_executed_binary":record(&archived_binary),
        "prefix_meta":record(&prefix_path),"self_controls":source_records,"cross_b1_to_b14":record(&cross_path),
        "cross_b1_to_b14_equals_repeated_b1_all_bits":true,
        "injection_source":"r4 select-b1-execute-b1/output.f32le, repeated per physical board",
        "scope":"frozen tensor intervention; no new Lt algorithm launch or private cache access"});
    FrozenInjection {
        b1,
        original,
        gated,
        report,
        provenance,
    }
}

#[allow(clippy::too_many_arguments)]
fn run_one(
    root: &Path,
    mode: &str,
    batch: usize,
    rt: &CudaRuntime,
    stream: &Arc<CudaStream>,
    full: &CudaModel,
    prefix: &CudaModel,
    suffix: &CudaModel,
    spatial_row: &[f32],
    global_row: &[f32],
    frozen: &FrozenInjection,
) -> (CudaOutputsHost, Value) {
    let subdir = format!("{mode}-b{batch}");
    fs::create_dir(root.join(&subdir)).unwrap();
    let sp = spatial_row.repeat(batch);
    let gl = global_row.repeat(batch);
    let gpu_sp = stream.clone_htod(&sp).unwrap();
    let gpu_gl = stream.clone_htod(&gl).unwrap();
    let before_sp = stream.clone_dtoh(&gpu_sp).unwrap();
    let before_gl = stream.clone_dtoh(&gpu_gl).unwrap();
    assert_eq!(bytes_f32(&before_sp), bytes_f32(&sp));
    assert_eq!(bytes_f32(&before_gl), bytes_f32(&gl));
    let mut inputs = json!({"before":{
        "spatial":export(root,&format!("{subdir}/input-spatial-before.f32le"),&before_sp,&[batch,22,19,19]),
        "global":export(root,&format!("{subdir}/input-global-before.f32le"),&before_gl,&[batch,19])},
        "per_board_post_symmetry_pair_sha256":INPUT_SHA});
    let mut ws = CudaWorkspace::new(stream, full, batch).unwrap();
    assert_eq!(ws.batch(), batch);
    let ws_ptr = ws.act384.device_ptr(stream).0;
    let mut item = json!({"mode":mode,"physical_batch":batch,"stream_id":stream.cu_stream() as usize,
        "act384_device_pointer":format!("0x{ws_ptr:016x}"),"same_workspace_across_segments":true,
        "same_stream_across_segments":true,"graph_ranges":if mode=="full" {json!([[0,full.num_layers()]])} else {json!([[0,2],[2,full.num_layers()]])}});
    eprintln!(
        "[first-down-injection] phase=before_apply mode={mode} physical_batch={batch} lane=0"
    );
    if mode == "full" {
        full.apply(rt, stream, &mut ws, &gpu_sp, &gpu_gl)
            .expect("complete frozen production graph");
    } else {
        prefix
            .apply(rt, stream, &mut ws, &gpu_sp, &gpu_gl)
            .expect("InitialConv plus first down prefix");
        stream.synchronize().unwrap();
        let old = stream.clone_dtoh(&ws.act384).unwrap();
        let actual_gated = stream.clone_dtoh(&ws.gated768).unwrap();
        let gated_bytes: Vec<u8> = actual_gated.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            gated_bytes, frozen.gated[&batch],
            "current prefix operand matches r4 source exactly"
        );
        assert_eq!(
            bytes_f32(&old),
            frozen.original[&batch],
            "current first down matches r4 self control"
        );
        item["prefix_act384"] = export(
            root,
            &format!("{subdir}/prefix-act384.f32le"),
            &old,
            &[batch, 361, 384],
        );
        item["prefix_gated768_sha256"] = json!(hash(&gated_bytes));
        item["prefix_matches_r4_self_control"] = json!(true);
        item["prefix_reference_sha256"] = json!(hash(&frozen.original[&batch]));
        item["prefix_bit_mismatches"] = json!(0);
        if mode == "injected" {
            let injected = frozen.b1.repeat(batch);
            let source = export(
                root,
                &format!("{subdir}/injection-source.f32le"),
                &injected,
                &[batch, 361, 384],
            );
            stream.memcpy_htod(&injected, &mut ws.act384).unwrap();
            stream.synchronize().unwrap();
            let actual = stream.clone_dtoh(&ws.act384).unwrap();
            assert_eq!(
                bytes_f32(&actual),
                bytes_f32(&injected),
                "exact injection roundtrip"
            );
            item["injection"] = json!({"source":source,
                "dtoh":export(root,&format!("{subdir}/injection-dtoh.f32le"),&actual,&[batch,361,384]),
                "roundtrip_all_bits_equal":true,"repeated_b1_source":true});
        } else {
            assert_eq!(mode, "split");
        }
        assert_eq!(ws.act384.device_ptr(stream).0, ws_ptr);
        // apply recomputes im2col into cols only. At this split boundary all
        // required live state is act768/gated768/act384; SPLITK is disabled.
        suffix
            .apply(rt, stream, &mut ws, &gpu_sp, &gpu_gl)
            .expect("suffix starting at first RMSNorm");
    }
    stream.synchronize().unwrap();
    let out = ws.to_host(stream).expect("all five raw heads");
    stream.synchronize().unwrap();
    let after_sp = stream.clone_dtoh(&gpu_sp).unwrap();
    let after_gl = stream.clone_dtoh(&gpu_gl).unwrap();
    assert_eq!(bytes_f32(&after_sp), bytes_f32(&before_sp));
    assert_eq!(bytes_f32(&after_gl), bytes_f32(&before_gl));
    inputs["after"] = json!({
        "spatial":export(root,&format!("{subdir}/input-spatial-after.f32le"),&after_sp,&[batch,22,19,19]),
        "global":export(root,&format!("{subdir}/input-global-after.f32le"),&after_gl,&[batch,19])});
    inputs["all_bits_unchanged"] = json!(true);
    item["inputs"] = inputs;
    item["input_unchanged"] = json!(true);
    let mut files = serde_json::Map::new();
    for (name, values, width) in heads(&out) {
        let shape = if name == "policy" {
            vec![batch, 6, 362]
        } else {
            vec![batch, width]
        };
        files.insert(
            name.into(),
            export(root, &format!("{subdir}/{name}.f32le"), values, &shape),
        );
    }
    item["files"] = json!(files);
    item["all_finite"] = json!(heads(&out)
        .iter()
        .all(|(_, values, _)| values.iter().all(|v| v.is_finite())));
    item["status"] = json!("RAW_RUN_COMPLETE");
    eprintln!("[first-down-injection] phase=complete mode={mode} physical_batch={batch} lane=0 input_unchanged=1");
    (out, item)
}

fn execute(
    root: &Path,
    dir: &Path,
    sp: &[f32],
    gl: &[f32],
    frozen: &FrozenInjection,
    report: &mut Value,
) {
    let model_path =
        PathBuf::from(std::env::var_os("KATAGO_TEST_MODEL_DIR").expect("required model directory"))
            .join(MODEL);
    let bytes = fs::read(&model_path).unwrap();
    assert_eq!(hash(&bytes), MODEL_SHA);
    let model_desc = load_model_from_bytes(&bytes, true, true).unwrap();
    assert_eq!(model_desc.sha256, MODEL_SHA);
    let mut graph = lower_model(&model_desc).unwrap();
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
    assert!(matches!(graph.layers[0], Layer::InitialConv(_)));
    let Layer::Linear(down) = &graph.layers[1] else {
        panic!("first down layer required")
    };
    assert_eq!((down.n, down.k), (384, 768));
    assert!(down.bias.is_none() && down.act.is_none() && !down.residual_add);
    assert!(matches!(graph.layers[2], Layer::RmsNorm(_)));
    report["model"] = record(&model_path);
    report["model_sha256"] = json!(MODEL_SHA);
    let rt = CudaRuntime::new().unwrap();
    let build = serde_json::to_value(backend_build_fingerprint()).unwrap();
    assert_eq!(build, frozen.report["backend_build_fingerprint"]);
    assert!(rt.cublaslt_handle().is_some());
    rt.validate_residual_algo_request().unwrap();
    let device = device_fingerprint(&rt.device).unwrap();
    let device = json!({"gpu_name":device.gpu_name,"compute_capability":device.compute_capability,
        "sm_count":device.sm_count,"l2_cache_bytes":device.l2_cache_bytes});
    assert_eq!(device, frozen.report["device_fingerprint"]);
    report["backend_build_fingerprint"] = build;
    report["device_fingerprint"] = device;
    let stream = rt.device.new_stream().unwrap();
    let full = CudaModel::load(&graph, &rt, &stream).unwrap();
    let layers = std::mem::take(&mut graph.layers);
    graph.layers = layers[..2].to_vec();
    let prefix = CudaModel::load(&graph, &rt, &stream).unwrap();
    graph.layers = layers[2..].to_vec();
    let suffix = CudaModel::load(&graph, &rt, &stream).unwrap();
    assert_eq!(full.num_layers(), prefix.num_layers() + suffix.num_layers());
    assert_eq!(prefix.num_layers(), 2);
    stream.synchronize().unwrap();
    report["layer_counts"] =
        json!({"full":full.num_layers(),"prefix":prefix.num_layers(),"suffix":suffix.num_layers()});
    report["stream_id"] = json!(stream.cu_stream() as usize);
    report["status"] = json!("RUNNING_NOT_COMPLETE");
    write_report(dir, report);
    let mut outputs = BTreeMap::new();
    for batch in [1usize, 14] {
        for mode in ["full", "split", "injected"] {
            let (out, item) = run_one(
                dir, mode, batch, &rt, &stream, &full, &prefix, &suffix, sp, gl, frozen,
            );
            report["runs"].as_array_mut().unwrap().push(item);
            write_report(dir, report);
            assert_eq!(
                report["runs"].as_array().unwrap().last().unwrap()["all_finite"],
                true
            );
            if mode != "full" {
                let result = compare(outputs.get(&(batch, "full")).unwrap(), &out);
                let key = format!("full_vs_{mode}_b{batch}");
                report["controls"][&key] = result;
                write_report(dir, report);
                if mode == "split" || batch == 1 {
                    assert_eq!(
                        report["controls"][&key]["all_equal"], true,
                        "mandatory split/B1 identity control"
                    );
                } else {
                    report["controls"][&key]["scope"] =
                        json!("B14 injection effect, descriptive only; equality not required");
                }
            }
            outputs.insert((batch, mode), out);
        }
    }
    report["all_raw_outputs_finite"] = json!(true);
    report["status"] = json!("PASS_SPLIT_AND_B1_INJECTION_CONTROLS_NOT_CERTIFIED");
    report["workspace"] = json!(root);
    write_report(dir, report);
}

#[test]
fn probe_worker_first_down_injection_cuda() {
    match std::env::var("KATAGO_RUN_WORKER_FIRST_DOWN_INJECTION").as_deref() {
        Err(std::env::VarError::NotPresent) => {
            eprintln!("SKIP: set KATAGO_RUN_WORKER_FIRST_DOWN_INJECTION=1");
            return;
        }
        Ok("1") => {}
        other => panic!("invalid explicit opt-in {other:?}"),
    }
    assert_eq!(
        std::env::consts::OS,
        "linux",
        "frozen r4 WSL platform required"
    );
    let kernel = fs::read_to_string("/proc/sys/kernel/osrelease")
        .unwrap()
        .to_lowercase();
    assert!(kernel.contains("microsoft"), "WSL runtime required");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let fixture = resolve(
        &root,
        "KATAGO_WORKER_INPUT_FIXTURE_DIR",
        "target/fork-parity-20260908/g3-worker-input-fixtures-wsl-r1",
    );
    assert_eq!(
        sha256_file(&fixture.join("manifest.json")).unwrap(),
        FIXTURE_SHA
    );
    let (sp, gl, input_provenance) = load_fixtures(&fixture);
    assert_eq!((sp.len(), gl.len()), (SPATIAL, GLOBAL));
    let frozen = frozen_injection(&root);
    let tactics = baseline_tactics();
    for (key, value) in &tactics {
        assert_eq!(value["resolved"], frozen.report["selected_tactics"][key]);
    }
    let dir = resolve(
        &root,
        "KATAGO_WORKER_FIRST_DOWN_INJECTION_DIR",
        "target/fork-parity-20260908/g3-first-down-injection-r1/raw-wsl-r1",
    );
    assert!(!dir.exists(), "fresh output directory required");
    assert!(dir
        .parent()
        .unwrap()
        .canonicalize()
        .unwrap()
        .starts_with(root.join("target").canonicalize().unwrap()));
    fs::create_dir(&dir).unwrap();
    let exe = std::env::current_exe().unwrap();
    let mut report = json!({"schema":1,"status":"PREPARING_NOT_COMPLETE","scope":"first-down tensor intervention through public split graph; two real request identities sharing one exact post-symmetry input, not 128 Worker or C++ golden certification",
        "test_executable":record(&exe),"test_source_sha256":hash(include_bytes!("probe_worker_first_down_injection_cuda.rs")),
        "platform":std::env::consts::OS,"kernel_release":kernel.trim(),"input_provenance":input_provenance,
        "actual_input_pair_sha256":INPUT_SHA,"r4_provenance":frozen.provenance,"selected_tactics":tactics,
        "registered_order":[["full",1],["split",1],["injected",1],["full",14],["split",14],["injected",14]],
        "lanes":1,"no_warmup":true,"production_certified":false,"is_cpp_golden_gate":false,"is_performance_measurement":false,
        "split":"prefix layers[..2], optional act384 overwrite, suffix layers[2..], same workspace/stream; suffix im2col overwrites only cols",
        "output_widths":{"policy":POLICY,"value":3,"misc":10,"moremisc":8,"ownership":361},
        "runs":[],"controls":{}});
    write_report(&dir, &report);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        execute(&root, &dir, &sp, &gl, &frozen, &mut report)
    }));
    if let Err(panic) = result {
        report["status"] = json!("FAILED_NOT_CERTIFIED");
        report["failure"] = json!(panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("non-string panic"));
        write_report(&dir, &report);
        std::panic::resume_unwind(panic);
    }
    eprintln!("[first-down-injection] status=PASS_SPLIT_AND_B1_INJECTION_CONTROLS_NOT_CERTIFIED raw_report={}",dir.join("report.json").display());
}
