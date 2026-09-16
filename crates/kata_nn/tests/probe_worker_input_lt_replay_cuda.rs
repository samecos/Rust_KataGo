//! Opt-in first-Linear diagnostic using frozen real half inputs, not a benchmark.
//! Observes an independent REQ8 heuristic replay, never the private model cache.
#![cfg(feature = "cuda")]

use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};

use kata_nn::backends::cuda::{
    CublasLtWeightLayout, CudaRuntime, backend_build_fingerprint, device_fingerprint,
    f16_to_f32_bits, f32_to_f16_bits,
};
use kata_nn::model_parser::load_model_from_bytes;
use kata_nn::native_model::lower_model;
use kata_nn::onnx_parser::Layer;
use kata_nn::tactic_plan::{installed_plan_id, sha256_file, tactic_var};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[path = "support/lt_replay_metadata.rs"]
mod lt_metadata;

const MODEL: &str = "kata1-tf3-b11c768-s11001M-d5973M.bin.gz";
const MODEL_SHA: &str = "1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6";
const PREFIX_SHA: &str = "a16ab88cbf023926140a3bb9c352e8006ac119c47978b056fc315dc7fb6fa849";
const WEIGHT_SHA: &str = "ae0ea2e05d5d453e053a77df577db581acf59c4a37ad34bd5598e34849f8434c";
const N: usize = 384;
const K: usize = 768;
const R2_REPORT_SHA: &str = "6ad3d2d194dcaeb995f75127275728351a4e1bd3e46e4b435ed9c359b33401a4";

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .collect()
}
fn half_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn record(path: &Path) -> Value {
    json!({"path":path,"sha256":sha256_file(path).unwrap()})
}
fn new_bytes(path: &Path, bytes: &[u8]) -> Value {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap()
        .write_all(bytes)
        .unwrap();
    record(path)
}
fn save(dir: &Path, report: &Value) {
    fs::write(
        dir.join("report.json"),
        serde_json::to_vec_pretty(report).unwrap(),
    )
    .unwrap();
}
fn resolve(root: &Path, key: &str, default: &str) -> PathBuf {
    let p = std::env::var_os(key)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default));
    if p.is_absolute() { p } else { root.join(p) }
}
fn bytes_bound(path: &Path, expected: &str, count: usize) -> Vec<u8> {
    let bytes = fs::read(path).unwrap();
    assert_eq!(bytes.len(), count, "raw length: {}", path.display());
    assert_eq!(hash(&bytes), expected, "raw SHA: {}", path.display());
    bytes
}
fn read_half(bytes: &[u8]) -> Vec<u16> {
    assert_eq!(bytes.len() % 2, 0);
    bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn read_float(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn comparison(a: &[f32], b: &[f32]) -> Value {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().chain(b).all(|x| x.is_finite()));
    let mismatches = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    let max_abs = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0.0f64, f64::max);
    json!({"elements":a.len(),"raw_bit_mismatches":mismatches,"max_abs":max_abs,"bitwise_equal":mismatches==0})
}
fn fp64_samples(
    inputs: &[u16],
    weights: &[u16],
    public: &[f32],
    explicit: &[f32],
    prefix: &[f32],
    batch: usize,
) -> Value {
    let mut indices = BTreeSet::new();
    // Always include the already observed worst output: token 0, channel 203,
    // and its corresponding position in every physical board repetition.
    for board in 0..batch {
        indices.insert(board * 361 * N + 203);
    }
    for i in 0..96usize {
        indices.insert(i * (public.len() - 1) / 95);
    }
    let mut samples = Vec::new();
    let mut max_abs = 0.0f64;
    for index in indices {
        let row = index / N;
        let channel = index % N;
        let mut reference = 0.0f64;
        // Independent FP64 arithmetic on raw uploaded half operands. No FP32
        // activation, residual, bias, or invented postprocessing enters here.
        for k in 0..K {
            reference += f16_to_f32_bits(inputs[row * K + k]) as f64
                * f16_to_f32_bits(weights[channel * K + k]) as f64;
        }
        assert!(reference.is_finite());
        let error = (public[index] as f64 - reference).abs();
        max_abs = max_abs.max(error);
        samples.push(json!({"flat_index":index,"token_row":row,"physical_board":row/361,
            "channel":channel,"is_registered_worst_channel":index%(361*N)==203,
            "reference_fp64":reference,"reference_rounded_fp32_bits":(reference as f32).to_bits(),
            "public_fp32_bits":public[index].to_bits(),"explicit_fp32_bits":explicit[index].to_bits(),
            "prefix_fp32_bits":prefix[index].to_bits(),"public_signed_error":public[index] as f64-reference,
            "public_abs_error":error}));
    }
    json!({"count":samples.len(),"max_public_abs_error":max_abs,"samples":samples,
        "scope":"descriptive FP64 dot-product audit; not a new model accuracy gate or a C++ golden comparison"})
}

fn repeated_comparison(a: &[f32], b: &[f32]) -> Value {
    let count = a.len().max(b.len());
    assert_eq!(count % a.len(), 0);
    assert_eq!(count % b.len(), 0);
    let mut mismatches = 0usize;
    let mut nonfinite = 0usize;
    let mut max_abs = 0.0f64;
    for i in 0..count {
        let (x, y) = (a[i % a.len()], b[i % b.len()]);
        mismatches += usize::from(x.to_bits() != y.to_bits());
        if x.is_finite() && y.is_finite() {
            max_abs = max_abs.max((x as f64 - y as f64).abs());
        } else {
            nonfinite += 1;
        }
    }
    json!({"elements":count,"candidate_elements":a.len(),"reference_elements":b.len(),
        "raw_bit_mismatches":mismatches,"nonfinite_pairs":nonfinite,"max_finite_abs":max_abs,
        "bitwise_equal":mismatches==0,"scope":"compare every physical board; repeat B1 when paired with B14; descriptive"})
}

fn cross_samples(inputs: &[u16], weights: &[u16], actual: &[f32], batch: usize) -> Value {
    let mut indices = BTreeSet::new();
    for board in 0..batch {
        indices.insert(board * 361 * N + 203);
    }
    for i in 0..96usize {
        indices.insert(i * (actual.len() - 1) / 95);
    }
    let samples:Vec<Value>=indices.into_iter().map(|index| {
        let (row,channel)=(index/N,index%N);
        let mut reference=0.0f64;
        for k in 0..K { reference+=f16_to_f32_bits(inputs[row*K+k]) as f64 * f16_to_f32_bits(weights[channel*K+k]) as f64; }
        assert!(reference.is_finite());
        json!({"flat_index":index,"token_row":row,"channel":channel,"reference_fp64":reference,
            "actual_fp32_bits":actual[index].to_bits(),"actual_finite":actual[index].is_finite(),
            "actual_signed_error":actual[index].is_finite().then_some(actual[index] as f64-reference)})
    }).collect();
    json!({"count":samples.len(),"samples":samples,"scope":"descriptive only; no cross-output accuracy acceptance threshold"})
}

#[allow(clippy::too_many_arguments)]
fn cross_shape(
    rt: &CudaRuntime,
    stream: &Arc<CudaStream>,
    gpu_weights: &CudaSlice<u16>,
    uploaded: &[u16],
    reference_dir: &Path,
    output: &Path,
    reference: &Value,
    r2_dir: &Path,
    r2: &Value,
    report: &mut Value,
) {
    let selections = [
        lt_metadata::select(rt, stream, 361, N, K).unwrap(),
        lt_metadata::select(rt, stream, 5054, N, K).unwrap(),
    ];
    report["selections"] = json!([selections[0].metadata, selections[1].metadata]);
    let mut operands = Vec::new();
    let mut baselines = Vec::new();
    let mut input_paths = Vec::new();
    for batch in [1usize, 14] {
        let p = reference["prefixes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["prefix_layers"] == 2 && p["physical_batch"] == batch)
            .unwrap();
        let e = &p["exported_tensors"]["gated768"];
        let path = reference_dir
            .join(p["directory"].as_str().unwrap())
            .join(e["file"].as_str().unwrap());
        operands.push(read_half(&bytes_bound(
            &path,
            e["sha256"].as_str().unwrap(),
            batch * 361 * K * 2,
        )));
        input_paths.push(path);
        let prior = r2["runs"]
            .as_array()
            .unwrap()
            .iter()
            .find(|x| x["physical_batch"] == batch)
            .unwrap();
        baselines.push(read_float(&bytes_bound(
            &r2_dir.join(format!("b{batch}/public.f32le")),
            prior["public_output"]["sha256"].as_str().unwrap(),
            batch * 361 * N * 4,
        )));
    }
    assert_eq!(operands[1], operands[0].repeat(14));
    save(output, report);
    // Controls first, followed by both cross-shape assignments. The same two
    // original algo objects are retained; selection never runs for a target.
    for (selection_index, target_index) in [(0usize, 0usize), (1, 1), (0, 1), (1, 0)] {
        let source_batch = [1usize, 14][selection_index];
        let target_batch = [1usize, 14][target_index];
        let is_control = selection_index == target_index;
        let inputs = &operands[target_index];
        let gpu_inputs = stream.clone_htod(inputs).unwrap();
        assert_eq!(stream.clone_dtoh(&gpu_inputs).unwrap(), *inputs);
        let mut gpu_out = stream.alloc_zeros::<f32>(target_batch * 361 * N).unwrap();
        let lt = lt_metadata::execute_selected(
            rt,
            stream,
            &gpu_inputs,
            gpu_weights,
            &mut gpu_out,
            target_batch * 361,
            N,
            K,
            &selections[selection_index],
        )
        .unwrap();
        let mut item = json!({"selection_batch":source_batch,"execution_batch":target_batch,"control":is_control,
            "input":record(&input_paths[target_index]),"lt":lt});
        if item["lt"]["executed"] == true && item["lt"]["status"] != "PASS_REPLAY_EXECUTED" {
            item["status"] = json!("EXECUTION_FAILED");
            report["runs"].as_array_mut().unwrap().push(item);
            save(output, report);
            panic!("checked algorithm launch/completion failed; no further CUDA work");
        }
        if item["lt"]["executed"] == true {
            stream.synchronize().unwrap();
            let actual = stream.clone_dtoh(&gpu_out).unwrap();
            assert_eq!(stream.clone_dtoh(&gpu_inputs).unwrap(), *inputs);
            assert_eq!(stream.clone_dtoh(gpu_weights).unwrap(), uploaded);
            let dir = output.join(format!("select-b{source_batch}-execute-b{target_batch}"));
            fs::create_dir(&dir).unwrap();
            item["output"] = new_bytes(&dir.join("output.f32le"), &f32_bytes(&actual));
            item["operands_unchanged"] = json!(true);
            item["vs_r2_b1_every_board"] = repeated_comparison(&actual, &baselines[0]);
            item["vs_r2_b14_every_board"] = repeated_comparison(&actual, &baselines[1]);
            item["fp64"] = cross_samples(inputs, uploaded, &actual, target_batch);
            item["status"] = json!(if is_control {
                "CONTROL_EXECUTED"
            } else {
                "CROSS_EXECUTED_DESCRIPTIVE"
            });
        } else {
            assert_eq!(item["lt"]["status"], "NOT_SUPPORTED");
            item["status"] = json!("NOT_SUPPORTED_NOT_EXECUTED");
        }
        report["runs"].as_array_mut().unwrap().push(item.clone());
        save(output, report);
        if is_control {
            assert_eq!(item["lt"]["executed"], true, "self control must execute");
            let comparison = if target_batch == 1 {
                "vs_r2_b1_every_board"
            } else {
                "vs_r2_b14_every_board"
            };
            assert_eq!(
                item[comparison]["bitwise_equal"], true,
                "self control must fully reproduce r2"
            );
        }
    }
}

fn run(root: &Path, reference_dir: &Path, output: &Path, reference: &Value, report: &mut Value) {
    let mode =
        std::env::var("KATAGO_LT_REPLAY_MODE").unwrap_or_else(|_| "same_shape_r2".to_string());
    assert!(
        mode == "same_shape_r2" || mode == "cross_shape_r3",
        "unknown replay mode"
    );
    let r2_dir =
        root.join("target/fork-parity-20260908/g3-linear-down-lt-replay-r1/run-wsl-r2/raw");
    let r2 = if mode == "cross_shape_r3" {
        let bytes = fs::read(r2_dir.join("report.json")).unwrap();
        assert_eq!(hash(&bytes), R2_REPORT_SHA);
        let prior: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(prior["status"], "PASS_INDEPENDENT_LT_REPLAY_NOT_CERTIFIED");
        let binary = root
            .join("target/fork-parity-20260908/g3-linear-down-lt-replay-r1/probe-passed-r2-wsl");
        assert_eq!(
            sha256_file(&binary).unwrap(),
            prior["test_executable"]["sha256"]
        );
        report["r2_reference"] = record(&r2_dir.join("report.json"));
        report["r2_archived_binary"] = record(&binary);
        Some(prior)
    } else {
        None
    };
    report["mode"] = json!(mode);
    if mode == "cross_shape_r3" {
        report["registered_assignments"] = json!([[1, 1], [14, 14], [1, 14], [14, 1]]);
        report["scope"] = json!(
            "independent cross-shape algo replay; own-shape controls must reproduce r2; cross outputs are descriptive, not a model accuracy or performance gate"
        );
    }
    // The current prefix test can advance independently. Bind the exact r1
    // source and executable that produced this reference, before any CUDA work.
    let archive = root.join("target/fork-parity-20260908/g3-prefix-trace-build-r1");
    let receipt_path = archive.join("receipt.json");
    let receipt_bytes = fs::read(&receipt_path).unwrap();
    assert_eq!(
        hash(&receipt_bytes),
        "d77ec71f904b4fc5030b7ebbc17868e9c07fdaf4078db69d9cc90840094b3b07"
    );
    let receipt: Value = serde_json::from_slice(&receipt_bytes).unwrap();
    assert_eq!(receipt["status"], "PASS_FROZEN_PREFIX_DIAGNOSTIC_ONLY");
    assert_eq!(receipt["meta"]["sha256"], PREFIX_SHA);
    let source_path = archive.join("trace_worker_input_prefix_cuda.rs");
    let binary_path = archive.join("prefix-trace-wsl");
    assert_eq!(
        sha256_file(&source_path).unwrap(),
        receipt["source"]["sha256"]
    );
    assert_eq!(receipt["source"]["sha256"], reference["test_source_sha256"]);
    assert_eq!(
        sha256_file(&binary_path).unwrap(),
        receipt["binary"]["sha256"]
    );
    assert_eq!(
        receipt["binary"]["sha256"],
        reference["test_executable_sha256"]
    );
    report["prefix_provenance"] = json!({"receipt":record(&receipt_path),"source":record(&source_path),"binary":record(&binary_path)});
    save(output, report);
    assert!(
        installed_plan_id().is_none(),
        "this independent test loads no plan"
    );
    let selected = reference["selected_tactics"].as_object().unwrap();
    for (key, value) in selected {
        assert_eq!(
            tactic_var(key).unwrap(),
            value.as_str().unwrap(),
            "exact replay tactic {key}"
        );
    }
    assert_eq!(selected["KATAGO_CUDA_CUBLASLT"], "1");
    assert_eq!(selected["KATAGO_CUDA_CUBLASLT_RANK"], "heuristic");
    assert_eq!(selected["KATAGO_CUDA_GEMM_LAYOUT"], "tn");
    assert_eq!(selected["KATAGO_CUDA_RESIDUAL_ALGO"], "heuristic");
    assert_eq!(selected["KATAGO_CUDA_NOGRAPH"], "1");
    for key in [
        "KATAGO_CUDA_PROFILE",
        "KATAGO_CUDA_DEBUG_LAYER",
        "KATAGO_CUDA_DUMP_INPUT",
    ] {
        assert!(
            std::env::var_os(key).is_none(),
            "unregistered diagnostic variable {key}"
        );
    }
    let model_path =
        PathBuf::from(std::env::var_os("KATAGO_TEST_MODEL_DIR").expect("model directory required"))
            .join(MODEL);
    let bytes = fs::read(&model_path).unwrap();
    assert_eq!(hash(&bytes), MODEL_SHA);
    let desc = load_model_from_bytes(&bytes, true, true).unwrap();
    let graph = lower_model(&desc).unwrap();
    let Layer::Linear(down) = &graph.layers[1] else {
        panic!("first down is not Linear")
    };
    assert_eq!((down.n, down.k), (N, K));
    assert!(down.bias.is_none() && down.act.is_none() && !down.residual_add);
    let f32_weights = down.weight.f32_data();
    assert_eq!(f32_weights.len(), N * K);
    let source_bytes = f32_bytes(f32_weights);
    assert_eq!(hash(&source_bytes), WEIGHT_SHA);
    assert_eq!(
        reference["layers"][1]["weights"]["down"]["source_fp32_sha256"],
        WEIGHT_SHA
    );
    // K is already aligned to 16; production upload_weight performs just this
    // RNE conversion, with no padding or transpose for TN [N,K].
    assert_eq!(K % 16, 0);
    let host_weights: Vec<u16> = f32_weights.iter().copied().map(f32_to_f16_bits).collect();
    assert!(host_weights.iter().all(|x| f16_to_f32_bits(*x).is_finite()));
    report["model"] = record(&model_path);
    report["source_weights"] = new_bytes(&output.join("down-source.f32le"), &source_bytes);
    report["host_rne1_weights"] =
        new_bytes(&output.join("down-host.f16le"), &half_bytes(&host_weights));
    save(output, report);

    let rt = CudaRuntime::new().expect("explicit replay requires CUDA");
    let build = backend_build_fingerprint();
    assert_eq!(build.fp16_encoding_revision, Some(1));
    assert_eq!(
        serde_json::to_value(&build).unwrap(),
        reference["backend_build_fingerprint"]
    );
    assert_eq!(rt.cublaslt_workspace_len(), 32 * 1024 * 1024);
    let device = device_fingerprint(&rt.device).unwrap();
    let device_json = json!({"gpu_name":device.gpu_name,"compute_capability":device.compute_capability,
        "sm_count":device.sm_count,"l2_cache_bytes":device.l2_cache_bytes});
    assert_eq!(device_json, reference["device_fingerprint"]);
    report["backend_build_fingerprint"] = serde_json::to_value(build).unwrap();
    report["device_fingerprint"] = device_json;
    let stream = rt.device.new_stream().unwrap();
    let gpu_weights = stream.clone_htod(&host_weights).unwrap();
    let uploaded = stream.clone_dtoh(&gpu_weights).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(
        uploaded, host_weights,
        "actual replay device weights differ after upload"
    );
    report["uploaded_device_weights"] =
        new_bytes(&output.join("down-device.f16le"), &half_bytes(&uploaded));
    report["uploaded_device_weight_identity"] = json!(
        "full u16 dtoh equals RNE1 host buffer; independent replay allocation, not CudaModel private weights"
    );
    report["stream_id"] = json!(stream.cu_stream() as usize);
    if let Some(prior) = r2 {
        assert_eq!(
            hash(&half_bytes(&uploaded)),
            prior["uploaded_device_weights"]["sha256"]
        );
        cross_shape(
            &rt,
            &stream,
            &gpu_weights,
            &uploaded,
            reference_dir,
            output,
            reference,
            &r2_dir,
            &prior,
            report,
        );
        return;
    }
    let mut b1_input: Option<Vec<u16>> = None;
    let mut all_bitwise = true;
    for batch in [1usize, 14] {
        let p = reference["prefixes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["prefix_layers"] == 2 && p["physical_batch"] == batch)
            .unwrap();
        let directory = reference_dir.join(p["directory"].as_str().unwrap());
        let input_desc = &p["exported_tensors"]["gated768"];
        let prefix_desc = &p["exported_tensors"]["act384"];
        assert_eq!(input_desc["shape"], json!([batch, 361, K]));
        assert_eq!(prefix_desc["shape"], json!([batch, 361, N]));
        let input_path = directory.join(input_desc["file"].as_str().unwrap());
        let prefix_path = directory.join(prefix_desc["file"].as_str().unwrap());
        let input_bytes = bytes_bound(
            &input_path,
            input_desc["sha256"].as_str().unwrap(),
            batch * 361 * K * 2,
        );
        let prefix_bytes = bytes_bound(
            &prefix_path,
            prefix_desc["sha256"].as_str().unwrap(),
            batch * 361 * N * 4,
        );
        let inputs = read_half(&input_bytes);
        let prefix = read_float(&prefix_bytes);
        assert!(inputs.iter().all(|x| f16_to_f32_bits(*x).is_finite()));
        if batch == 1 {
            b1_input = Some(inputs.clone());
        } else {
            assert_eq!(
                inputs,
                b1_input.as_ref().unwrap().repeat(batch),
                "B14 input is not bitwise repeated B1"
            );
        }
        let m = batch * 361;
        let gpu_inputs = stream.clone_htod(&inputs).unwrap();
        assert_eq!(
            stream.clone_dtoh(&gpu_inputs).unwrap(),
            inputs,
            "actual input upload differs"
        );
        let mut public_out = stream.alloc_zeros::<f32>(m * N).unwrap();
        let mut explicit_out = stream.alloc_zeros::<f32>(m * N).unwrap();
        assert!(
            rt.cublaslt_gemm_with_layout(
                &stream,
                &gpu_inputs,
                &gpu_weights,
                &mut public_out,
                m,
                N,
                K,
                0.0,
                CublasLtWeightLayout::Tn
            )
            .unwrap(),
            "public production API did not execute Lt"
        );
        stream.synchronize().unwrap();
        let public = stream.clone_dtoh(&public_out).unwrap();
        let metadata = lt_metadata::replay(
            &rt,
            &stream,
            &gpu_inputs,
            &gpu_weights,
            &mut explicit_out,
            m,
            N,
            K,
        )
        .unwrap();
        stream.synchronize().unwrap();
        let explicit = stream.clone_dtoh(&explicit_out).unwrap();
        assert_eq!(
            stream.clone_dtoh(&gpu_inputs).unwrap(),
            inputs,
            "replay modified input"
        );
        assert_eq!(
            stream.clone_dtoh(&gpu_weights).unwrap(),
            uploaded,
            "replay modified weights"
        );
        let c1 = comparison(&public, &prefix);
        let c2 = comparison(&explicit, &public);
        all_bitwise &= c1["bitwise_equal"] == true && c2["bitwise_equal"] == true;
        let case_dir = output.join(format!("b{batch}"));
        fs::create_dir(&case_dir).unwrap();
        let row = json!({"physical_batch":batch,"m":m,"n":N,"k":K,"alpha":1,"beta":0,"output":"FP32",
            "input":record(&input_path),"prefix_output":record(&prefix_path),"actual_input_u16_roundtrip":"PASS",
            "operands_unchanged_after_both_launches":true,
            "public_output":new_bytes(&case_dir.join("public.f32le"),&f32_bytes(&public)),
            "explicit_output":new_bytes(&case_dir.join("explicit.f32le"),&f32_bytes(&explicit)),
            "public_vs_prefix":c1,"explicit_vs_public":c2,"lt":metadata,
            "fp64":fp64_samples(&inputs,&uploaded,&public,&explicit,&prefix,batch)});
        report["runs"].as_array_mut().unwrap().push(row);
        save(output, report);
    }
    assert!(
        all_bitwise,
        "replay differs: preserved both batch results; no accuracy gate relaxed"
    );
    assert_eq!(
        sha256_file(&reference_dir.join("meta.json")).unwrap(),
        PREFIX_SHA
    );
}

#[test]
fn probe_worker_input_lt_replay_cuda() {
    match std::env::var("KATAGO_RUN_WORKER_LT_REPLAY").as_deref() {
        Err(std::env::VarError::NotPresent) => {
            eprintln!("SKIP: explicit KATAGO_RUN_WORKER_LT_REPLAY=1 required");
            return;
        }
        Ok("1") => {}
        other => panic!("invalid KATAGO_RUN_WORKER_LT_REPLAY: {other:?}"),
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let reference_dir = resolve(
        &root,
        "KATAGO_LT_REPLAY_REFERENCE_DIR",
        "target/fork-parity-20260908/g3-prefix-trace-wsl-r1",
    );
    let reference_bytes = fs::read(reference_dir.join("meta.json")).unwrap();
    assert_eq!(hash(&reference_bytes), PREFIX_SHA);
    let reference: Value = serde_json::from_slice(&reference_bytes).unwrap();
    assert_eq!(
        reference["status"],
        "PASS_PREFIX_TRACE_COMPLETE_NOT_CERTIFIED"
    );
    assert_eq!(reference["model_sha256"], MODEL_SHA);
    let output = resolve(
        &root,
        "KATAGO_LT_REPLAY_DUMP_DIR",
        "target/fork-parity-20260908/g3-linear-down-lt-replay-r1/run-wsl-r1/raw",
    );
    assert!(!output.exists(), "new output directory required");
    assert!(
        output
            .parent()
            .unwrap()
            .canonicalize()
            .unwrap()
            .starts_with(root.join("target").canonicalize().unwrap())
    );
    fs::create_dir(&output).unwrap();
    let executable = std::env::current_exe().unwrap();
    let mut report = json!({"schema":1,"status":"PREPARING_NOT_COMPLETE","pass":false,
        "scope":"independent first-Linear Lt replay; not intercepted model weights/cache, not a full-network accuracy or performance certification",
        "reference":record(&reference_dir.join("meta.json")),"test_executable":record(&executable),
        "test_source_sha256":hash(include_bytes!("probe_worker_input_lt_replay_cuda.rs")),
        "helper_source_sha256":hash(include_bytes!("support/lt_replay_metadata.rs")),
        "precision_source_sha256":hash(include_bytes!("support/lt_replay_precision.rs")),
        "registered_batches":[1,14],"selected_tactics":reference["selected_tactics"],"runs":[]});
    save(&output, &report);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run(&root, &reference_dir, &output, &reference, &mut report)
    }));
    match outcome {
        Ok(()) => {
            report["status"] = json!(if report["mode"] == "cross_shape_r3" {
                "COMPLETE_CROSS_SHAPE_DIAGNOSTIC_NOT_CERTIFIED"
            } else {
                "PASS_INDEPENDENT_LT_REPLAY_NOT_CERTIFIED"
            });
            if report["mode"] == "cross_shape_r3" {
                report.as_object_mut().unwrap().remove("pass");
                report["diagnostic_complete"] = json!(true);
                report["controls_bitwise_pass"] = json!(true);
            } else {
                report["pass"] = json!(true);
            }
            save(&output, &report);
        }
        Err(error) => {
            report["status"] = json!("FAIL_REPLAY_EVIDENCE_PRESERVED");
            save(&output, &report);
            std::panic::resume_unwind(error);
        }
    }
}
