//! Three public-API graph prefixes locate the fixed-input B1/B14 divergence.
//!
//! Opt in with KATAGO_RUN_WORKER_PREFIX_TRACE=1 and KATAGO_TEST_MODEL_DIR.
//! Use the same explicit DUALFFN=1 / ATTN_TILE=q64-serial / NOGRAPH=1 baseline
//! as dump_worker_fixed_inputs_cuda; other choices must remain TN/heuristic.
//! KATAGO_WORKER_PREFIX_REFERENCE_DIR selects that completed full-output dump;
//! KATAGO_WORKER_PREFIX_DUMP_DIR must select a fresh target directory.
//!
//! Parse/lower once. Run graph.layers[..1] (InitialConv), [..2] (first Linear
//! down), and [..3] (first RMSNorm) at physical batches 1 and 14. These three
//! branches have no lookahead/skip when FUSION=none and SPLITK=0. This is a
//! truncated-graph diagnostic, not an observed full-network intermediate dump.
//! Only written active ranges are exported; half buffers retain raw u16 bits.
//! Revision2 uses a fresh output directory; the complete revision1 source and
//! binary remain frozen in target/.../g3-prefix-trace-build-r1.
#![cfg(feature = "cuda")]

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use kata_nn::backends::cuda::{
    CudaRuntime, backend_build_fingerprint, device_fingerprint, f16_to_f32_bits,
};
use kata_nn::backends::cuda_exec::{CudaModel, CudaWorkspace};
use kata_nn::model_parser::load_model_from_bytes;
use kata_nn::native_model::lower_model;
use kata_nn::onnx_parser::{Layer, Tensor};
use kata_nn::tactic_plan::{installed_plan_id, sha256_file, tactic_var};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const MODEL: &str = "kata1-tf3-b11c768-s11001M-d5973M.bin.gz";
const MODEL_SHA: &str = "1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6";
const INPUT_SHA: &str = "f9caf09c4fbf32b48736dda5531a3292bdacccef07db6a9b974678c09ab22911";

fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn json_file(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).expect("read diagnostic contract")).unwrap()
}
fn fresh_file(path: &Path, bytes: &[u8]) {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap_or_else(|e| panic!("refuse overwrite/cannot create {}: {e}", path.display()))
        .write_all(bytes)
        .unwrap();
}
fn metadata(dir: &Path, meta: &Value) {
    fs::write(
        dir.join("meta.json"),
        serde_json::to_vec_pretty(meta).unwrap(),
    )
    .unwrap();
}
fn resolve(workspace: &Path, key: &str, default: &str) -> PathBuf {
    let path = std::env::var_os(key)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default));
    if path.is_absolute() {
        path
    } else {
        workspace.join(path)
    }
}
fn weight(t: &Tensor) -> Value {
    json!({"shape":t.dims,"source_fp32_sha256":hash(&f32_bytes(t.f32_data())),"elements":t.numel(),
           "scope":"lowered source FP32 tensor; not an assertion about padded/transposed uploaded weight bytes"})
}

fn tactics(reference: &Value) -> BTreeMap<String, String> {
    assert!(
        installed_plan_id().is_none(),
        "no plan is loaded by this test"
    );
    let mut result = BTreeMap::new();
    for (key, default, expected) in [
        ("KATAGO_CUDA_CUBLASLT", "1", "1"),
        ("KATAGO_CUDA_DUALFFN", "0", "1"),
        ("KATAGO_CUDA_ATTN_TILE", "q128", "q64-serial"),
        ("KATAGO_CUDA_ATTN", "fa2", "fa2"),
        ("KATAGO_CUDA_GEMM_LAYOUT", "tn", "tn"),
        ("KATAGO_CUDA_NOGRAPH", "0", "1"),
        ("KATAGO_CUDA_CUBLASLT_RANK", "heuristic", "heuristic"),
        ("KATAGO_CUDA_RESIDUAL_ALGO", "heuristic", "heuristic"),
        ("KATAGO_CUDA_SPLITK", "0", "0"),
        ("KATAGO_CUDA_FUSION", "none", "none"),
        ("KATAGO_CUDA_T32", "0", "0"),
        ("KATAGO_CUDA_T64N32", "0", "0"),
        ("KATAGO_CUDA_RMS", "warp4", "warp4"),
        ("KATAGO_CUDA_PADBATCH", "0", "0"),
        ("KATAGO_CUDA_NOPIPELINE", "0", "0"),
    ] {
        let value = match tactic_var(key) {
            Ok(v) => v,
            Err(std::env::VarError::NotPresent) => default.to_string(),
            Err(e) => panic!("{key}: {e}"),
        };
        assert_eq!(value, expected, "prefix diagnostic baseline {key}");
        assert_eq!(
            reference["selected_tactics"][key]["resolved"], value,
            "full-output reference tactic identity"
        );
        result.insert(key.to_string(), value);
    }
    for key in [
        "KATAGO_CUDA_PROFILE",
        "KATAGO_CUDA_DEBUG_LAYER",
        "KATAGO_CUDA_DUMP_INPUT",
    ] {
        assert!(
            std::env::var_os(key).is_none(),
            "clear {key}; this test performs its own synchronized exports"
        );
    }
    result
}

fn download_f32(stream: &Arc<CudaStream>, src: &CudaSlice<f32>, count: usize) -> Vec<f32> {
    assert!(count <= src.len());
    let mut host = vec![0f32; count];
    stream.memcpy_dtoh(&src.slice(..count), &mut host).unwrap();
    stream
        .synchronize()
        .expect("download complete before host reads");
    host
}
fn download_f16(stream: &Arc<CudaStream>, src: &CudaSlice<u16>, count: usize) -> Vec<u16> {
    assert!(count <= src.len());
    let mut host = vec![0u16; count];
    stream.memcpy_dtoh(&src.slice(..count), &mut host).unwrap();
    stream
        .synchronize()
        .expect("raw half download complete before host reads");
    host
}

struct Export {
    raw: Vec<u8>,
    values: Vec<f32>,
    element_bytes: usize,
    description: Value,
}
fn export(
    dir: &Path,
    name: &str,
    batch: usize,
    channels: usize,
    raw: Vec<u8>,
    values: Vec<f32>,
    element_bytes: usize,
) -> Export {
    assert_eq!(values.len(), batch * 361 * channels);
    assert_eq!(raw.len(), values.len() * element_bytes);
    let file = format!(
        "{name}.{}le",
        if element_bytes == 2 { "f16" } else { "f32" }
    );
    fresh_file(&dir.join(&file), &raw);
    let expanded = f32_bytes(&values);
    let expanded_file = if element_bytes == 2 {
        format!("{name}_expanded.f32le")
    } else {
        file.clone()
    };
    if element_bytes == 2 {
        fresh_file(&dir.join(&expanded_file), &expanded);
    }
    let row_bytes = 361 * channels * element_bytes;
    let row_sha: Vec<_> = raw.chunks_exact(row_bytes).map(hash).collect();
    let description = json!({"file":file,"sha256":hash(&raw),"dtype":if element_bytes==2 {"float16"} else {"float32"},
        "byte_order":"little-endian","shape":[batch,361,channels],"elements":values.len(),"bytes":raw.len(),
        "raw_ieee_bits_preserved":true,"physical_row_sha256":row_sha,
        "expanded_fp32":{"file":expanded_file,"sha256":hash(&expanded),"bytes":expanded.len(),
            "scope":if element_bytes==2 {"diagnostic exact finite half-to-float expansion; raw16 identity resides in the separate f16 file"} else {"same raw FP32 file"}},
        "all_finite":values.iter().all(|v|v.is_finite())});
    assert!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite {name}; raw bytes saved"
    );
    Export {
        raw,
        values,
        element_bytes,
        description,
    }
}

fn compare(candidate: &Export, baseline: &Export, batch: usize) -> Value {
    assert_eq!(candidate.element_bytes, baseline.element_bytes);
    assert_eq!(candidate.values.len(), baseline.values.len() * batch);
    let width = baseline.values.len();
    let elem = candidate.element_bytes;
    let mut bit_differences = 0usize;
    let mut max_abs = 0f64;
    let mut sum_sq = 0f64;
    let mut worst = 0usize;
    for i in 0..candidate.values.len() {
        let j = i % width;
        bit_differences += usize::from(
            candidate.raw[i * elem..(i + 1) * elem] != baseline.raw[j * elem..(j + 1) * elem],
        );
        let diff = (candidate.values[i] as f64 - baseline.values[j] as f64).abs();
        sum_sq += diff * diff;
        if diff > max_abs {
            max_abs = diff;
            worst = i;
        }
    }
    json!({"elements":candidate.values.len(),"raw_bit_mismatches":bit_differences,"max_abs":max_abs,
        "rmse":(sum_sq/candidate.values.len() as f64).sqrt(),"worst_flat_index":worst,
        "candidate_value":candidate.values[worst],"baseline_value":baseline.values[worst%width],
        "scope":"candidate all physical rows compared with B1; descriptive, no accuracy threshold"})
}

#[test]
fn trace_worker_input_prefix_cuda() {
    match std::env::var("KATAGO_RUN_WORKER_PREFIX_TRACE").as_deref() {
        Err(std::env::VarError::NotPresent) => {
            eprintln!("SKIP: set KATAGO_RUN_WORKER_PREFIX_TRACE=1 in an exclusive GPU window");
            return;
        }
        Ok("1") => {}
        value => panic!("KATAGO_RUN_WORKER_PREFIX_TRACE must be1 when set: {value:?}"),
    }
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let reference_dir = resolve(
        &workspace,
        "KATAGO_WORKER_PREFIX_REFERENCE_DIR",
        "target/fork-parity-20260908/g3-fixed-input-cuda-wsl-r1",
    );
    let reference = json_file(&reference_dir.join("meta.json"));
    assert_eq!(reference["status"], "PASS_RAW_DUMP_COMPLETE_NOT_CERTIFIED");
    assert_eq!(reference["model_sha256"], MODEL_SHA);
    assert_eq!(reference["actual_input_pair_sha256"], INPUT_SHA);
    let selection = tactics(&reference);
    let first = &reference["batches"][0];
    assert_eq!(first["physical_batch"], 1);
    assert_eq!(first["directory"], "b1");
    let read_input = |name: &str, count: usize| {
        let desc = &first[name];
        let filename = desc["file"].as_str().unwrap();
        assert_eq!(Path::new(filename).file_name().unwrap(), filename);
        let bytes = fs::read(reference_dir.join("b1").join(filename)).unwrap();
        assert_eq!(desc["sha256"], hash(&bytes));
        assert_eq!(bytes.len(), count * 4);
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect();
        assert!(values.iter().all(|x| x.is_finite()));
        (bytes, values)
    };
    let (sp_bytes, spatial) = read_input("input_spatial", 22 * 361);
    let (gl_bytes, global) = read_input("input_global", 19);
    let mut input_pair = sp_bytes;
    input_pair.extend_from_slice(&gl_bytes);
    assert_eq!(hash(&input_pair), INPUT_SHA);
    let model_path =
        PathBuf::from(std::env::var_os("KATAGO_TEST_MODEL_DIR").expect("model directory required"))
            .join(MODEL);
    let bytes = fs::read(&model_path).unwrap();
    assert_eq!(hash(&bytes), MODEL_SHA);
    let desc = load_model_from_bytes(&bytes, true, true).expect("parse one native model");
    assert_eq!(desc.sha256, MODEL_SHA);
    let mut graph = lower_model(&desc).expect("lower once");
    assert_eq!(
        (
            graph.board_size,
            graph.trunk_channels,
            graph.mid_channels,
            graph.num_heads,
            graph.head_dim
        ),
        (19, 768, 384, 12, 32)
    );
    let original_layers = std::mem::take(&mut graph.layers);
    let original_layer_count = original_layers.len();
    let Layer::InitialConv(conv) = &original_layers[0] else {
        panic!("prefix0 is not InitialConv")
    };
    assert_eq!(conv.out_channels, 768);
    let Layer::Linear(down) = &original_layers[1] else {
        panic!("prefix1 is not first Linear down")
    };
    assert_eq!((down.n, down.k), (384, 768));
    assert!(down.bias.is_none() && down.act.is_none() && !down.residual_add);
    let Layer::RmsNorm(rms) = &original_layers[2] else {
        panic!("prefix2 is not first RMSNorm")
    };
    assert_eq!(rms.channels, 384);
    assert_eq!(rms.scale.numel(), 384);
    assert!(rms.eps.is_finite() && rms.eps > 0.0);
    let rms_scale_bytes = f32_bytes(rms.scale.f32_data());
    let layers = json!([
        {"zero_based_index":0,"post_layer_boundary":1,"kind":"InitialConv","operation":"im2col half -> GEMM FP32 -> global bias/gate; act768 FP32 and gated768 half",
         "weights":{"conv":weight(&conv.weight),"global":weight(&conv.global_weight),"gate_scale":weight(&conv.gate_scale),"gate_bias":weight(&conv.gate_bias)}},
        {"zero_based_index":1,"post_layer_boundary":2,"kind":"LinearDown","n":384,"k":768,"beta":0,"output":"act384 FP32","weights":{"down":weight(&down.weight)}},
        {"zero_based_index":2,"post_layer_boundary":3,"kind":"RmsNorm","channels":384,"epsilon":rms.eps,"epsilon_fp32_bits":format!("0x{:08x}",rms.eps.to_bits()),
         "operation":"act384 FP32 read only -> FP32 sum/square/rsqrt/scale -> normed half; SPLITK=0; kernel rms_norm_f32_w4_kernel",
         "weights":{"scale":weight(&rms.scale)},"scale_file":{"file":"rms_scale.f32le","sha256":hash(&rms_scale_bytes),"shape":[384],"dtype":"float32","byte_order":"little-endian","bytes":rms_scale_bytes.len(),
             "scope":"lowered FP32 scale, uploaded without dtype conversion by production upload_param; not a direct private model device allocation readback"}}]);
    let dir = resolve(
        &workspace,
        "KATAGO_WORKER_PREFIX_DUMP_DIR",
        "target/fork-parity-20260908/g3-prefix-trace-wsl-r2",
    );
    assert!(!dir.exists(), "fresh trace directory required");
    assert!(
        dir.parent()
            .unwrap()
            .canonicalize()
            .unwrap()
            .starts_with(workspace.join("target").canonicalize().unwrap())
    );
    fs::create_dir(&dir).unwrap();
    fresh_file(&dir.join("rms_scale.f32le"), &rms_scale_bytes);
    let exe = std::env::current_exe().unwrap();
    let mut meta = json!({"schema":1,"diagnostic_revision":2,"status":"PREPARING_NOT_COMPLETE","scope":"public-API truncated graph localization; not a full-network intermediate observation, C++ golden or performance test",
        "reference_meta":{"path":reference_dir.join("meta.json"),"sha256":sha256_file(&reference_dir.join("meta.json")).unwrap()},
        "model_path":model_path,"model_sha256":MODEL_SHA,"actual_input_pair_sha256":INPUT_SHA,
        "test_executable":exe,"test_executable_sha256":sha256_file(&exe).unwrap(),
        "test_source_sha256":hash(include_bytes!("trace_worker_input_prefix_cuda.rs")),
        "selected_tactics":selection,"original_layer_count":original_layer_count,"layers":layers,
        "registered_order":[[1,1],[1,14],[2,1],[2,14],[3,1],[3,14]],"lanes":1,"prefixes":[],"comparisons":{},
        "constraints":"FUSION=none/SPLITK=0; first three production branches have no lookahead/skip; no DEBUG_LAYER hook; no unwritten scratch compared; RMSNorm does not write act384",
        "raw_half_identity":"u16 little-endian files preserved separately from diagnostic FP32 expansions"});
    metadata(&dir, &meta);
    let rt = CudaRuntime::new().expect("explicit prefix trace requires CUDA");
    let build = backend_build_fingerprint();
    assert_eq!(
        serde_json::to_value(&build).unwrap(),
        reference["backend_build_fingerprint"],
        "same compiled backend as full-output reference"
    );
    assert_eq!(build.fp16_encoding_revision, Some(1));
    assert!(rt.cublaslt_handle().is_some());
    let device = device_fingerprint(&rt.device).unwrap();
    let device_json = json!({"gpu_name":device.gpu_name,"compute_capability":device.compute_capability,"sm_count":device.sm_count,"l2_cache_bytes":device.l2_cache_bytes});
    assert_eq!(
        device_json, reference["device_fingerprint"],
        "same actual GPU target as full-output reference"
    );
    meta["backend_build_fingerprint"] = serde_json::to_value(&build).unwrap();
    meta["device_fingerprint"] = device_json;
    let stream = rt.device.new_stream().unwrap();
    meta["stream_id"] = json!(stream.cu_stream() as usize);
    let mut outputs: BTreeMap<(usize, usize), BTreeMap<&str, Export>> = BTreeMap::new();
    for prefix in [1usize, 2, 3] {
        graph.layers = original_layers[..prefix].to_vec();
        let model =
            CudaModel::load(&graph, &rt, &stream).expect("load only requested prefix weights");
        assert_eq!(model.num_layers(), prefix);
        stream.synchronize().unwrap();
        for batch in [1usize, 14] {
            let run_dir = dir.join(format!("prefix{prefix}-b{batch}"));
            fs::create_dir(&run_dir).unwrap();
            let sp = spatial.repeat(batch);
            let gl = global.repeat(batch);
            let mut gpu_sp = stream.alloc_zeros::<f32>(sp.len()).unwrap();
            let mut gpu_gl = stream.alloc_zeros::<f32>(gl.len()).unwrap();
            stream.memcpy_htod(&sp, &mut gpu_sp).unwrap();
            stream.memcpy_htod(&gl, &mut gpu_gl).unwrap();
            let mut ws = CudaWorkspace::new(&stream, &model, batch).unwrap();
            assert_eq!(ws.batch(), batch);
            stream.synchronize().unwrap();
            eprintln!(
                "[worker-prefix] phase=before_apply prefix_layers={prefix} physical_batch={batch} lane=0 input_pair_sha256={INPUT_SHA}"
            );
            model
                .apply(&rt, &stream, &mut ws, &gpu_sp, &gpu_gl)
                .expect("prefix forward");
            stream.synchronize().unwrap();
            let m = batch * 361;
            let mut tensors = BTreeMap::new();
            for (name, src, channels) in [("cols", &ws.cols, 208), ("gated768", &ws.gated768, 768)]
            {
                let bits = download_f16(&stream, src, m * channels);
                let values = bits.iter().map(|&x| f16_to_f32_bits(x)).collect();
                tensors.insert(
                    name,
                    export(
                        &run_dir,
                        name,
                        batch,
                        channels,
                        bits.iter().flat_map(|x| x.to_le_bytes()).collect(),
                        values,
                        2,
                    ),
                );
            }
            // GEMM output allocation is M*1152 but InitialConv writes a dense
            // M*768 prefix. Never interpret the unwritten tail as a row stride.
            for (name, src, channels) in [
                ("initial_gemm_out", &ws.gemm_out, 768),
                ("act768", &ws.act768, 768),
            ] {
                let values = download_f32(&stream, src, m * channels);
                let raw = f32_bytes(&values);
                tensors.insert(
                    name,
                    export(&run_dir, name, batch, channels, raw, values, 4),
                );
            }
            if prefix >= 2 {
                let values = download_f32(&stream, &ws.act384, m * 384);
                let raw = f32_bytes(&values);
                tensors.insert(
                    "act384",
                    export(&run_dir, "act384", batch, 384, raw, values, 4),
                );
            }
            if prefix == 3 {
                let bits = download_f16(&stream, &ws.normed, m * 384);
                let values = bits.iter().map(|&x| f16_to_f32_bits(x)).collect();
                tensors.insert(
                    "normed",
                    export(
                        &run_dir,
                        "normed",
                        batch,
                        384,
                        bits.iter().flat_map(|x| x.to_le_bytes()).collect(),
                        values,
                        2,
                    ),
                );
            }
            let descriptions: BTreeMap<_, _> = tensors
                .iter()
                .map(|(name, value)| (*name, value.description.clone()))
                .collect();
            meta["prefixes"].as_array_mut().unwrap().push(json!({"prefix_layers":prefix,"physical_batch":batch,"lane":0,
                "directory":format!("prefix{prefix}-b{batch}"),"workspace_batch":ws.batch(),"m_tokens":m,
                "allocated_elements":{"cols":ws.cols.len(),"gated768":ws.gated768.len(),"act768":ws.act768.len(),"act384":ws.act384.len(),"normed":ws.normed.len(),"gemm_out":ws.gemm_out.len()},
                "exported_tensors":descriptions,"source_row_replication":"exact post-symmetry frozen spatial/global row repeated batch times"}));
            outputs.insert((prefix, batch), tensors);
            metadata(&dir, &meta);
            eprintln!(
                "[worker-prefix] phase=complete prefix_layers={prefix} physical_batch={batch} lane=0 active_buffers_saved=1 raw_f16_preserved=1"
            );
        }
    }
    let mut comparisons = BTreeMap::new();
    for prefix in [1usize, 2, 3] {
        let a = &outputs[&(prefix, 1)];
        let b = &outputs[&(prefix, 14)];
        let mut stats = BTreeMap::new();
        for (name, candidate) in b {
            stats.insert(*name, compare(candidate, &a[name], 14));
        }
        comparisons.insert(format!("prefix{prefix}-b14-vs-b1"), json!(stats));
    }
    let mut self_consistency = BTreeMap::new();
    let mut rms_input_consistency = BTreeMap::new();
    let mut consistent = true;
    for batch in [1usize, 14] {
        let a = &outputs[&(1, batch)];
        let b = &outputs[&(2, batch)];
        let mut stats = BTreeMap::new();
        for name in ["cols", "initial_gemm_out", "act768", "gated768"] {
            let same = a[name].raw == b[name].raw;
            consistent &= same;
            stats.insert(name, same);
        }
        self_consistency.insert(batch.to_string(), stats);
        let c = &outputs[&(3, batch)];
        let mut rms_stats = BTreeMap::new();
        for name in ["cols", "initial_gemm_out", "act768", "gated768", "act384"] {
            let same = b[name].raw == c[name].raw;
            consistent &= same;
            rms_stats.insert(name, same);
        }
        rms_input_consistency.insert(batch.to_string(), rms_stats);
    }
    meta["comparisons"] = json!(comparisons);
    meta["initial_buffers_cross_prefix_bitwise_equal"] = json!(self_consistency);
    meta["prefix2_to_prefix3_live_inputs_bitwise_equal"] = json!(rms_input_consistency);
    meta["status"] = json!(if consistent {
        "PASS_PREFIX_TRACE_COMPLETE_NOT_CERTIFIED"
    } else {
        "FAIL_PREFIX_SELF_CONSISTENCY_RAW_SAVED"
    });
    metadata(&dir, &meta);
    assert!(
        consistent,
        "cross-prefix buffers changed: initial1/2 or RMS input2/3; raw files saved"
    );
    eprintln!(
        "PASS_PREFIX_TRACE_COMPLETE_NOT_CERTIFIED: {}",
        dir.join("meta.json").display()
    );
}
