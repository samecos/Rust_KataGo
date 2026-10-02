//! Migration correctness for model-bound FP16/INT8 recipes, not an accuracy gate.
//!
//! Uses KATAGO_TEST_MODEL_DIR/{b11c768h12nbt3tflrs-fson-silu.bin.gz,
//! b15-ffn-pruned-a8.bin.gz}. KATAGO_QUANT_TEST_B11_MODEL and
//! KATAGO_QUANT_TEST_B15_MODEL optionally select explicit model files.
//! KATAGO_QUANT_TEST_ONNX_MODEL optionally adds a supported ONNX fixture.
//! Missing fixtures skip individually, before any GPU initialization.
//!
//! Common baseline tactics are required, with no installed FP16 plan. This
//! test does not mutate environment variables. Run in a clean process, e.g.:
//! cargo test -p kata_nn --test test_quantized_backend --features cuda --release -- --nocapture
//!
//! Every loaded model runs B1/B8 with A -> B -> A input updates at the same
//! device addresses. Graph replays must equal fresh direct outputs bitwise.
//! Legacy-vs-recipe comparisons also use raw f32 bits across all five outputs.
//! Mixed projection coverage checks routing/finite outputs and graph replay;
//! it does not compare mixed precision against an FP32 accuracy reference.
//! The separate KATAGO_TEST_MXFP8_BACKEND=1 gate adds full-FFN/full-transformer
//! MXFP8 and three-precision routing, with whole-graph invalid-input recovery.
#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaStream, sys};
use kata_game::{
    board::{Board, P_BLACK, get_opp, location},
    history::BoardHistory,
    rules::Rules,
};
use kata_nn::{
    backends::{
        cuda::CudaRuntime,
        cuda_exec::{CudaModel, CudaOutputsHost, CudaWorkspace, set_capturing},
        int8::Int8Scope,
    },
    inputs::{MiscNNInputParams, fill_row_v7},
    onnx_parser::{Layer, LayerGraph},
    quantization_plan::{Precision, resolve_recipe, template_recipe},
};

const FIELDS: [&str; 5] = ["policy", "value", "misc", "moremisc", "ownership"];
type OutputBits = [Vec<u32>; 5];
// These integration tests share the device's primary context. A context-wide
// completion barrier in one test invalidates another test's active capture.
// Serialize complete cases even when Rust's test harness uses multiple threads.
static GPU_LOCK: Mutex<()> = Mutex::new(());

fn test_batches() -> Vec<usize> {
    let batches = std::env::var("KATAGO_QUANT_TEST_BATCHES")
        .unwrap_or_else(|_| "1,8".into())
        .split(',')
        .map(|v| v.parse::<usize>().expect("integer test batch"))
        .collect::<Vec<_>>();
    assert!(!batches.is_empty() && batches.iter().all(|b| (1..=8).contains(b)));
    batches
}

struct Fixture {
    label: &'static str,
    path: PathBuf,
}

fn fixtures() -> Vec<Fixture> {
    let directory = std::env::var_os("KATAGO_TEST_MODEL_DIR").map(PathBuf::from);
    let mut fixtures = Vec::new();
    for (label, variable, filename) in [
        (
            "B11",
            "KATAGO_QUANT_TEST_B11_MODEL",
            "b11c768h12nbt3tflrs-fson-silu.bin.gz",
        ),
        (
            "B15-pruned",
            "KATAGO_QUANT_TEST_B15_MODEL",
            "b15-ffn-pruned-a8.bin.gz",
        ),
    ] {
        let path = std::env::var_os(variable)
            .map(PathBuf::from)
            .or_else(|| directory.as_ref().map(|d| d.join(filename)));
        match path {
            Some(path) if path.is_file() => fixtures.push(Fixture { label, path }),
            Some(path) => eprintln!("SKIP {label}: missing model {}", path.display()),
            None => eprintln!("SKIP {label}: set KATAGO_TEST_MODEL_DIR or {variable}"),
        }
    }
    if let Some(path) = std::env::var_os("KATAGO_QUANT_TEST_ONNX_MODEL").map(PathBuf::from) {
        assert!(
            path.is_file(),
            "explicit ONNX fixture missing: {}",
            path.display()
        );
        fixtures.push(Fixture {
            label: "ONNX",
            path,
        });
    }
    fixtures
}

fn require_common_baseline() {
    assert!(
        kata_nn::tactic_plan::installed_plan_id().is_none(),
        "migration comparison requires no installed FP16 tactic plan"
    );
    for (key, expected) in [
        ("KATAGO_CUDA_DUALFFN", "0"),
        ("KATAGO_CUDA_SPLITK", "0"),
        ("KATAGO_CUDA_FUSION", "none"),
        ("KATAGO_CUDA_GEMM_LAYOUT", "tn"),
        ("KATAGO_CUDA_RESIDUAL_ALGO", "heuristic"),
        ("KATAGO_CUDA_ATTN", "fa2"),
        ("KATAGO_CUDA_QKV_IMMUTABLE_R1", "0"),
        ("KATAGO_CUDA_QKV_CLASSIC_N128_R1", "0"),
        ("KATAGO_CUDA_OUTPROJ_CLASSIC_N128_R1", "0"),
        ("KATAGO_CUDA_FFN_COMPACT_R1", "0"),
        ("KATAGO_CUDA_INT8_GEMM_TUNE", "0"),
        ("KATAGO_CUDA_PROFILE", "0"),
    ] {
        match kata_nn::tactic_plan::tactic_var(key) {
            Ok(actual) => assert_eq!(actual, expected, "unset {key} or use {expected}"),
            Err(std::env::VarError::NotPresent) => {}
            Err(error) => panic!("invalid {key}: {error}"),
        }
    }
    assert!(
        std::env::var_os("KATAGO_CUDA_DEBUG_LAYER").is_none(),
        "unset KATAGO_CUDA_DEBUG_LAYER for graph replay checks"
    );
}

fn load_graph(path: &Path) -> (LayerGraph, String) {
    let bytes = std::fs::read(path).expect("read model fixture");
    let sha256 = kata_core::hash::sha2::sha256_hex(&bytes);
    let lower = path.to_string_lossy().to_ascii_lowercase();
    let graph = if lower.ends_with(".onnx") {
        kata_nn::onnx_model::parse_onnx_model(&bytes).expect("parse ONNX metadata");
        kata_nn::onnx_parser::parse_layer_graph(&bytes).expect("parse ONNX graph")
    } else {
        let compressed = lower.ends_with(".gz");
        let inner = lower.strip_suffix(".gz").unwrap_or(&lower);
        assert!(inner.ends_with(".bin") || inner.ends_with(".txt"));
        let desc = kata_nn::model_parser::load_model_from_bytes(
            &bytes,
            inner.ends_with(".bin"),
            compressed,
        )
        .expect("parse native fixture");
        kata_nn::native_model::lower_model(&desc).expect("lower native fixture")
    };
    (graph, sha256)
}

fn input_variant(variant: usize) -> (Vec<f32>, Vec<f32>) {
    let mut spatial = Vec::with_capacity(8 * 22 * 361);
    let mut global = Vec::with_capacity(8 * 19);
    for row in 0..8 {
        let mut board = Board::new(19, 19);
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        let mut player = P_BLACK;
        let mut rand =
            kata_core::rng::Rand::new_from_seed(&format!("quant-recipe-{variant}-{row}"));
        for _ in 0..variant * 17 + row * 9 {
            let legal: Vec<_> = (0..361)
                .map(|i| location::get_loc(i % 19, i / 19, 19))
                .filter(|&loc| hist.is_legal(&board, loc, player))
                .collect();
            if legal.is_empty() {
                break;
            }
            hist.make_board_move_assume_legal(
                &mut board,
                legal[rand.next_u64() as usize % legal.len()],
                player,
            );
            player = get_opp(player);
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
    (spatial, global)
}

fn output_bits(output: CudaOutputsHost, context: &str) -> OutputBits {
    let values = [
        output.policy,
        output.value,
        output.misc,
        output.moremisc,
        output.ownership,
    ];
    for (field, values) in FIELDS.iter().zip(&values) {
        assert!(!values.is_empty(), "{context}/{field}: missing output");
        assert!(
            values.iter().all(|v| v.is_finite()),
            "{context}/{field}: nonfinite output"
        );
    }
    values.map(|values| values.into_iter().map(f32::to_bits).collect())
}

fn assert_bits_equal(expected: &OutputBits, actual: &OutputBits, context: &str) {
    for (field, (expected, actual)) in FIELDS.iter().zip(expected.iter().zip(actual)) {
        assert_eq!(
            expected.len(),
            actual.len(),
            "{context}/{field}: output length"
        );
        if let Some((index, (&a, &b))) = expected
            .iter()
            .zip(actual)
            .enumerate()
            .find(|(_, (a, b))| a != b)
        {
            panic!(
                "{context}/{field}[{index}]: expected bits {a:#010x} ({}), actual {b:#010x} ({})",
                f32::from_bits(a),
                f32::from_bits(b)
            );
        }
    }
}

struct CaptureMode;
impl Drop for CaptureMode {
    fn drop(&mut self) {
        set_capturing(false);
    }
}

/// Returns direct A/B outputs at B1 and B8 after checking graph replay A/B/A.
fn exercise_model(
    rt: &Arc<CudaRuntime>,
    stream: &Arc<CudaStream>,
    model: &CudaModel,
    label: &str,
) -> Vec<OutputBits> {
    let inputs = [input_variant(0), input_variant(1)];
    assert_ne!(inputs[0].0, inputs[1].0, "input variants must differ");
    let mut outputs = Vec::new();
    for batch in test_batches() {
        let mut spatial = stream.clone_htod(&inputs[0].0[..batch * 22 * 361]).unwrap();
        let mut global = stream.clone_htod(&inputs[0].1[..batch * 19]).unwrap();
        let mut ws =
            CudaWorkspace::new_with_runtime(rt.clone(), stream, model, batch).expect("workspace");
        // Keep the reference buffers separate: a direct call must not refresh
        // graph outputs/scratch and hide a missing or stale captured operation.
        let mut reference_ws = CudaWorkspace::new_with_runtime(rt.clone(), stream, model, batch)
            .expect("reference workspace");
        model
            .apply(rt, stream, &mut reference_ws, &spatial, &global)
            .expect("reference warm");
        reference_ws.complete_quantized_warmup().expect("validate independent reference warmup");
        model
            .apply(rt, stream, &mut ws, &spatial, &global)
            .expect("pre-capture warm");
        ws.complete_quantized_warmup()
            .expect("validate warmup status");
        let graph_resources = ws
            .quantized_graph_resources()
            .expect("retain graph resources");
        rt.device.synchronize().expect("warm synchronize");
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .expect("begin graph capture");
        let applied = {
            set_capturing(true);
            let _capture = CaptureMode;
            model.apply(rt, stream, &mut ws, &spatial, &global)
        };
        let captured = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY,
        );
        applied.expect("apply during graph capture");
        let graph = captured
            .expect("end graph capture")
            .expect("captured graph");
        graph.upload().expect("upload graph");
        let mut first = None;
        for (round, variant) in [0, 1, 0].into_iter().enumerate() {
            stream
                .memcpy_htod(&inputs[variant].0[..batch * 22 * 361], &mut spatial)
                .unwrap();
            stream
                .memcpy_htod(&inputs[variant].1[..batch * 19], &mut global)
                .unwrap();
            let context = format!("{label}/B{batch}/input{variant}/round{round}");
            model
                .apply(rt, stream, &mut reference_ws, &spatial, &global)
                .expect("fresh direct apply");
            let direct = output_bits(
                reference_ws.to_host(stream).expect("direct outputs"),
                &context,
            );
            graph.launch().expect("graph replay");
            let replay = output_bits(ws.to_host(stream).expect("replayed outputs"), &context);
            assert_bits_equal(&direct, &replay, &format!("{context}/direct-vs-graph"));
            if round == 0 {
                first = Some(direct.clone());
            } else if round == 1 {
                assert!(
                    first.as_ref().unwrap() != &direct,
                    "{context}: changed input produced identical outputs; stale-input check is ineffective"
                );
            } else {
                assert_bits_equal(
                    first.as_ref().unwrap(),
                    &direct,
                    &format!("{context}/return-to-A"),
                );
            }
            if round < 2 {
                outputs.push(direct);
            }
        }
        rt.device
            .synchronize()
            .expect("finish batch before dropping graph buffers");
        if model.has_mxfp8() {
            // A captured graph cannot cancel subsequent nodes after a NaN.
            // Its sticky status must reject the whole output, then recover on A.
            let bad = vec![f32::NAN; batch * 22 * 361];
            stream.memcpy_htod(&bad, &mut spatial).unwrap();
            graph.launch().expect("invalid-input graph submission");
            let rejected = ws
                .to_host(stream)
                .expect_err("nonfinite MXFP8 forward must be rejected");
            assert!(rejected.contains("MXFP8 forward rejected"), "{rejected}");
            stream
                .memcpy_htod(&inputs[0].0[..batch * 22 * 361], &mut spatial)
                .unwrap();
            stream
                .memcpy_htod(&inputs[0].1[..batch * 19], &mut global)
                .unwrap();
            graph.launch().expect("legal replay after rejected forward");
            let recovered = output_bits(ws.to_host(stream).expect("recovered outputs"), label);
            assert_bits_equal(first.as_ref().unwrap(), &recovered, "full-graph recovery");
        }
        drop(graph);
        drop(graph_resources);
    }
    outputs
}

#[test]
fn recipes_preserve_legacy_bits_and_refresh_graph_inputs() {
    let _guard = GPU_LOCK.lock().unwrap();
    let fixtures = fixtures();
    if fixtures.is_empty() {
        return;
    }
    require_common_baseline();
    let rt = Arc::new(CudaRuntime::new().expect("CUDA runtime"));
    let stream = rt.device.new_stream().expect("CUDA stream");
    for fixture in fixtures {
        let (graph, model_sha256) = load_graph(&fixture.path);
        if fixture.label != "ONNX" {
            let expected_blocks = if fixture.label == "B11" { 11 } else { 15 };
            assert_eq!(
                graph.num_blocks, expected_blocks,
                "wrong {} fixture",
                fixture.label
            );
        }
        let ffn_count = graph
            .layers
            .iter()
            .filter(|l| matches!(l, Layer::Ffn(_)))
            .count();
        assert!(ffn_count >= 2);
        if fixture.label == "B15-pruned" {
            assert!(
                graph
                    .layers
                    .iter()
                    .any(|l| matches!(l, Layer::Ffn(f) if f.hidden < 3 * graph.mid_channels)),
                "B15 fixture must contain pruned FFNs"
            );
        }
        for (precision, scope) in [
            ("all-FP16", None),
            ("all-FFN-INT8", Some(Int8Scope::Ffn)),
            ("all-transformer-INT8", Some(Int8Scope::Transformer)),
        ] {
            let label = format!("{}/{precision}", fixture.label);
            let mut template = template_recipe(&graph, &model_sha256).expect("FP16 template");
            for projection in &mut template.projections {
                if scope == Some(Int8Scope::Transformer)
                    || (scope == Some(Int8Scope::Ffn) && projection.id.contains(".ffn."))
                {
                    projection.precision = Precision::Int8;
                }
            }
            let recipe = resolve_recipe(&template, &graph, &model_sha256).expect("resolve recipe");
            let legacy = if let Some(scope) = scope {
                CudaModel::load_int8(&graph, &rt, &stream, scope)
            } else {
                CudaModel::load(&graph, &rt, &stream)
            }
            .expect("load legacy model");
            let expected = exercise_model(&rt, &stream, &legacy, &format!("{label}/legacy"));
            drop(legacy);
            let unified = CudaModel::load_quantized(&graph, &rt, &stream, &recipe)
                .expect("load recipe model");
            assert_eq!(
                unified.quantization_recipe_id(),
                Some(recipe.recipe_sha256.as_str())
            );
            assert_eq!(
                unified.int8_ffn_count(),
                if scope.is_some() { ffn_count } else { 0 }
            );
            let actual = exercise_model(&rt, &stream, &unified, &format!("{label}/recipe"));
            assert_eq!(expected.len(), actual.len());
            for (case, (a, b)) in expected.iter().zip(&actual).enumerate() {
                let batch = test_batches()[case / 2];
                assert_bits_equal(
                    a,
                    b,
                    &format!("{label}/B{batch}/input{}/legacy-vs-recipe", case % 2),
                );
            }
            eprintln!(
                "PASS_RAW_BITS_NOT_ACCURACY_CERTIFICATION: {label} model_sha256={model_sha256}, batches={:?}, A/B, all five outputs, graph A/B/A",
                test_batches()
            );
        }

        let mut mixed = template_recipe(&graph, &model_sha256).unwrap();
        let mut selected = 0;
        for projection in &mut mixed.projections {
            if matches!(
                projection.id.as_str(),
                "trunk.block00.pair00.ffn.dual"
                    | "trunk.block00.pair01.ffn.down"
                    | "trunk.block00.pair00.attention.qkv"
                    | "trunk.block00.pair01.attention.out"
            ) {
                projection.precision = Precision::Int8;
                selected += 1;
            }
        }
        assert_eq!(
            selected, 4,
            "fixture must cover independent dual/down/QKV/out projections"
        );
        let recipe = resolve_recipe(&mixed, &graph, &model_sha256).unwrap();
        let model = CudaModel::load_quantized(&graph, &rt, &stream, &recipe)
            .expect("load independently mixed projections");
        assert_eq!(
            model.quantization_recipe_id(),
            Some(recipe.recipe_sha256.as_str())
        );
        assert_eq!(model.int8_ffn_count(), 2);
        exercise_model(
            &rt,
            &stream,
            &model,
            &format!("{}/mixed-projection-smoke", fixture.label),
        );
        eprintln!(
            "PASS_MIXED_ROUTING_SMOKE_NOT_ACCURACY_CERTIFICATION: {} batches={:?}, finite outputs, graph A/B/A",
            fixture.label,
            test_batches()
        );
    }
}

/// Whole-graph execution gate only. Representative SGF holdout accuracy and
/// end-to-end performance remain separate; random legal positions are not a
/// replacement for either. Explicit opt-in prevents accidental experiment runs.
#[test]
fn mxfp8_recipes_refresh_graph_and_reject_invalid_forward() {
    let _guard = GPU_LOCK.lock().unwrap();
    if std::env::var("KATAGO_TEST_MXFP8_BACKEND").as_deref() != Ok("1") {
        eprintln!("SKIP whole-graph MXFP8: set KATAGO_TEST_MXFP8_BACKEND=1");
        return;
    }
    let fixtures = fixtures();
    assert!(
        !fixtures.is_empty(),
        "explicit MXFP8 test requires model fixtures"
    );
    require_common_baseline();
    let rt = Arc::new(CudaRuntime::new().expect("CUDA runtime"));
    let stream = rt.device.new_stream().expect("CUDA stream");
    let report_dir = std::env::var_os("KATAGO_MXFP8_GRAPH_REPORT_DIR").map(PathBuf::from);
    if let Some(directory) = &report_dir {
        std::fs::create_dir(directory).expect("new report directory required");
    }
    for fixture in fixtures {
        let (graph, model_sha256) = load_graph(&fixture.path);
        for case in [
            "all-ffn-mxfp8",
            "all-transformer-mxfp8",
            "three-precision-mixed",
        ] {
            let mut recipe = template_recipe(&graph, &model_sha256).unwrap();
            recipe.version = kata_nn::quantization_plan::MXFP8_RECIPE_VERSION;
            recipe.quantization_semantics_version =
                kata_nn::quantization_plan::MXFP8_QUANTIZATION_SEMANTICS_VERSION;
            for projection in &mut recipe.projections {
                projection.precision = match case {
                    "all-ffn-mxfp8" if projection.id.contains(".ffn.") => Precision::Mxfp8,
                    "all-transformer-mxfp8" => Precision::Mxfp8,
                    "three-precision-mixed" => match projection.id.as_str() {
                        "trunk.block00.pair00.ffn.dual"
                        | "trunk.block00.pair01.ffn.down"
                        | "trunk.block00.pair00.attention.qkv"
                        | "trunk.block00.pair01.attention.out" => Precision::Mxfp8,
                        "trunk.block00.pair00.ffn.down"
                        | "trunk.block00.pair01.ffn.dual"
                        | "trunk.block00.pair00.attention.out"
                        | "trunk.block00.pair01.attention.qkv" => Precision::Int8,
                        _ => Precision::Fp16,
                    },
                    _ => Precision::Fp16,
                };
            }
            let resolved =
                resolve_recipe(&recipe, &graph, &model_sha256).expect("resolve MXFP8 recipe");
            let model = CudaModel::load_quantized(&graph, &rt, &stream, &resolved)
                .expect("load MXFP8 graph");
            assert!(model.has_mxfp8());
            assert!(
                CudaWorkspace::new(&stream, &model, 1).is_err(),
                "old workspace API must not silently omit MXFP8 preparation"
            );
            let label = format!("{}/{case}", fixture.label);
            let outputs = exercise_model(&rt, &stream, &model, &label);
            if let Some(directory) = &report_dir {
                let value = serde_json::json!({
                    "schema": "rustgo-mxfp8-graph-test-v1",
                    "status": "PASS_EXECUTION_NOT_ACCURACY_CERTIFICATION",
                    "model_sha256": model_sha256,
                    "recipe_sha256": resolved.recipe_sha256,
                    "batches": test_batches(),
                    "graph_aba_bitwise": true,
                    "nonfinite_rejection_and_recovery": true,
                    "raw_output_bits": outputs,
                    "production_certified": false
                });
                let name = format!("{}-{case}", fixture.label);
                std::fs::write(
                    directory.join(format!("{name}.report.json")),
                    serde_json::to_vec_pretty(&value).unwrap(),
                )
                .unwrap();
                std::fs::write(
                    directory.join(format!("{name}.recipe.json")),
                    serde_json::to_vec_pretty(&recipe).unwrap(),
                )
                .unwrap();
            }
            eprintln!(
                "PASS_MXFP8_GRAPH_NOT_ACCURACY_CERTIFICATION: {label} model={model_sha256} recipe={} batches={:?}",
                resolved.recipe_sha256,
                test_batches()
            );
        }
    }
}

// A dedicated opt-in bounds this diagnostic to one fixture; the existing
// whole-graph matrix still covers all fixtures and physical batches.
#[test]
fn mxfp8_final_heads_reject_late_overflow_and_recover_after_corruption() {
    let _guard = GPU_LOCK.lock().unwrap();
    use cudarc::driver::{CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};
    use kata_nn::backends::mxfp8::{FINAL_OUTPUT_LAYER, STATUS_OUTPUT_NONFINITE};
    use kata_nn::onnx_parser::{Tensor, TensorData};
    if std::env::var("KATAGO_TEST_MXFP8_FINAL_HEADS").as_deref() != Ok("1") {
        eprintln!("SKIP late-head regression: set KATAGO_TEST_MXFP8_FINAL_HEADS=1");
        return;
    }
    require_common_baseline();
    let fixture = fixtures().into_iter().next().expect("late-head test needs a real model fixture");
    let (mut source_graph, model_sha256) = load_graph(&fixture.path);
    let rt = Arc::new(CudaRuntime::new().unwrap());
    let stream = rt.device.new_stream().unwrap();
    let load = |graph: &LayerGraph| {
        // This deliberately transformed graph is a test-only synthetic model,
        // not the fixture file's certified identity. Bind its changed source
        // tensors as well as the original fixture SHA, then rebuild the recipe.
        let mut identity = b"rustgo-test-only-late-head-synthetic-v1\0".to_vec();
        identity.extend_from_slice(model_sha256.as_bytes());
        for layer in &graph.layers {
            if let Layer::ValueHead(head) = layer {
                for tensor in [&head.conv1_weight, &head.bias1, &head.linear2_weight,
                    &head.linear2_bias, &head.value_matmul, &head.value_bias,
                    &head.misc_matmul, &head.misc_bias, &head.moremisc_matmul,
                    &head.moremisc_bias, &head.ownership_conv] {
                    identity.extend_from_slice(&(tensor.numel() as u64).to_le_bytes());
                    for value in tensor.f32_data() { identity.extend_from_slice(&value.to_bits().to_le_bytes()); }
                }
            }
        }
        let synthetic_sha = kata_core::hash::sha2::sha256_hex(&identity);
        let mut recipe = template_recipe(graph, &synthetic_sha).unwrap();
        recipe.version = kata_nn::quantization_plan::MXFP8_RECIPE_VERSION;
        recipe.quantization_semantics_version = kata_nn::quantization_plan::MXFP8_QUANTIZATION_SEMANTICS_VERSION;
        recipe.projections.iter_mut().rev().find(|p| p.id.ends_with(".ffn.down"))
            .expect("last FFN down projection").precision = Precision::Mxfp8;
        let resolved = resolve_recipe(&recipe, graph, &synthetic_sha).unwrap();
        eprintln!("[late-head-test] synthetic_model_sha256={synthetic_sha} production_certified=false");
        CudaModel::load_quantized(graph, &rt, &stream, &resolved).unwrap()
    };
    let inputs = input_variant(0);
    let spatial = stream.clone_htod(&inputs.0[..22*361]).unwrap();
    let global = stream.clone_htod(&inputs.1[..19]).unwrap();
    stream.synchronize().unwrap();
    // Source tensors are all finite. Only the final FP32 FC computes MAX*2,
    // after the last MX projection. No kernel or status flag is mocked.
    let original_head = source_graph.layers.iter().find_map(|l| if let Layer::ValueHead(h)=l {Some(h.clone())} else {None}).unwrap();
    let fill = |tensor: &mut Tensor, first: f32| {
        let mut data = vec![0.0f32; tensor.numel()]; data[0] = first;
        tensor.data = TensorData::F32(data);
    };
    for layer in &mut source_graph.layers {
        if let Layer::ValueHead(head) = layer {
            fill(&mut head.linear2_weight, 0.0);
            fill(&mut head.linear2_bias, f32::MAX);
            fill(&mut head.value_matmul, 2.0);
            fill(&mut head.value_bias, 0.0);
            fill(&mut head.misc_matmul, 0.0);
            fill(&mut head.misc_bias, 0.0);
            fill(&mut head.moremisc_matmul, 0.0);
            fill(&mut head.moremisc_bias, 0.0);
        }
    }
    {
        let bad_model = load(&source_graph);
        let mut bad_ws = CudaWorkspace::new_with_runtime(rt.clone(), &stream, &bad_model, 1).unwrap();
        bad_model.apply(&rt,&stream,&mut bad_ws,&spatial,&global).unwrap();
        let error = bad_ws.to_host(&stream).expect_err("final FP32 overflow must reject the batch");
        assert!(error.contains("final_raw_heads"),"{error}");
        let status = stream.clone_dtoh(bad_ws.quantized_status().unwrap()).unwrap();
        let value = stream.clone_dtoh(&bad_ws.out_value).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(status,[STATUS_OUTPUT_NONFINITE,FINAL_OUTPUT_LAYER]);
        assert_eq!(value[0], f32::INFINITY, "deterministic final FC overflow");
        assert!(bad_ws.complete_quantized_warmup().is_err(),"late failure must not authorize capture");
    }
    for layer in &mut source_graph.layers {
        if let Layer::ValueHead(head) = layer { *head = original_head.clone(); }
    }
    let model = load(&source_graph);
    let mut ws = CudaWorkspace::new_with_runtime(rt.clone(), &stream, &model, 1).unwrap();
    let mut reference = CudaWorkspace::new_with_runtime(rt.clone(), &stream, &model, 1).unwrap();
    model.apply(&rt,&stream,&mut ws,&spatial,&global).unwrap();
    ws.complete_quantized_warmup().unwrap();
    let resources = ws.quantized_graph_resources().unwrap();
    let add = rt.get_func("f32_add_kernel").unwrap();
    fn head(workspace: &CudaWorkspace, field: usize) -> &CudaSlice<f32> {
        match field {0=>&workspace.out_policy,1=>&workspace.out_value,2=>&workspace.out_misc,3=>&workspace.out_moremisc,_=>&workspace.out_ownership}
    }
    let inject = |workspace: &CudaWorkspace, field: usize, delta: &CudaSlice<f32>| {
        let target = head(workspace,field);
        let (pointer,_guard) = target.device_ptr(&stream);
        // Production add kernel, same owner stream, after all model projections.
        unsafe { stream.launch_builder(&add).arg(&pointer).arg(delta).arg(&pointer)
            .arg(&(target.len() as i32)).launch(LaunchConfig::for_num_elems(target.len() as u32)).unwrap(); }
        workspace.enqueue_quantized_final_output_check().unwrap();
    };
    let mut bad_snapshot = unsafe {rt.device.alloc_pinned::<u32>(2)}.unwrap();
    let mut good_snapshot = unsafe {rt.device.alloc_pinned::<u32>(2)}.unwrap();
    for field in 0..5 {
        let count = head(&ws,field).len();
        let zeros = vec![0.0f32;count];
        let mut delta = stream.clone_htod(&zeros).unwrap();
        stream.synchronize().unwrap();
        model.apply(&rt,&stream,&mut reference,&spatial,&global).unwrap();
        inject(&reference,field,&delta);
        let fresh = output_bits(reference.to_host(&stream).unwrap(),"late-head fresh direct");
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED).unwrap();
        {
            set_capturing(true); let _capture = CaptureMode;
            model.apply(&rt,&stream,&mut ws,&spatial,&global).unwrap();
            inject(&ws,field,&delta);
        }
        let graph = stream.end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY).unwrap().unwrap();
        graph.upload().unwrap();
        graph.launch().unwrap();
        assert_bits_equal(&fresh,&output_bits(ws.to_host(&stream).unwrap(),"initial valid head"),FIELDS[field]);
        for invalid in [f32::NAN,f32::INFINITY] {
            let mut corruption = zeros.clone(); corruption[count-1] = invalid;
            stream.memcpy_htod(&corruption,&mut delta).unwrap();
            graph.launch().unwrap();
            ws.enqueue_quantized_status_copy(&mut bad_snapshot).unwrap();
            let bad_done = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)).unwrap();
            let error=ws.to_host(&stream).expect_err("late head corruption must reject outputs");
            assert!(error.contains("final_raw_heads"),"{error}");
            // Recovery is a fresh full replay. Its reset must not erase this
            // submission's already captured pinned status snapshot.
            stream.memcpy_htod(&zeros,&mut delta).unwrap();
            graph.launch().unwrap();
            ws.enqueue_quantized_status_copy(&mut good_snapshot).unwrap();
            let good_done=stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)).unwrap();
            good_done.synchronize().unwrap(); bad_done.synchronize().unwrap();
            assert_eq!(bad_snapshot.as_slice().unwrap(),[STATUS_OUTPUT_NONFINITE,FINAL_OUTPUT_LAYER]);
            CudaWorkspace::validate_quantized_status(good_snapshot.as_slice().unwrap()).unwrap();
            assert_bits_equal(&fresh,&output_bits(ws.to_host(&stream).unwrap(),"late-head recovery"),FIELDS[field]);
        }
        drop(graph);
        eprintln!("PASS_LATE_HEAD_FINITE_CHECK: {} Graph A-bad-A and pinned isolation",FIELDS[field]);
    }
    drop(resources);
}
