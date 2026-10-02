//! Same mixed recipe, different FP16 execution tactics: a G1 regression, not
//! quantization accuracy or performance certification. No FP32/FP16 model is
//! used as a reference for the INT8 recipe.
//!
//! Set KATAGO_TEST_MODEL_DIR (native b11c768h12nbt3tflrs-fson-silu.bin.gz)
//! or KATAGO_QUANT_TEST_B11_MODEL. Missing fixtures skip before GPU creation.
//! Six isolated processes cover baseline, DualFFN, fusion up/down/all, split-K.
//! Each runs B1/B8 with graph A -> B -> A against independent direct scratch.
//! DualFFN must preserve raw bits; fusion/split-K use the original five-head
//! absolute gates and 100% top-1, applied to all six policy channels here.
//! Graph/direct and repeated A always require raw-bit equality.
//!
//! KATAGO_QUANT_FAST_PATH_REPORT_DIR optionally selects an artifact directory;
//! otherwise logs and bit-exact JSON outputs stay under kata_nn/target.
#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use cudarc::driver::{CudaStream, sys};
use kata_game::{
    board::{Board, P_BLACK, get_opp, location},
    history::BoardHistory,
    rules::Rules,
};
use kata_nn::{
    backends::{
        cuda::{CudaRuntime, ensure_requested_cuda_capabilities, validate_quantized_runtime},
        cuda_exec::{CudaModel, CudaOutputsHost, CudaWorkspace, set_capturing},
    },
    inputs::{MiscNNInputParams, fill_row_v7},
    onnx_parser::Layer,
    quantization_plan::{Precision, resolve_recipe, template_recipe},
};
use serde::{Deserialize, Serialize};

const TEST_NAME: &str = "mixed_recipe_preserves_fast_paths_and_graph_replay";
const CHILD_MODE: &str = "RUSTGO_QUANT_FAST_PATH_CHILD_MODE";
const CHILD_OUTPUT: &str = "RUSTGO_QUANT_FAST_PATH_CHILD_OUTPUT";
const MODES: [&str; 6] = [
    "baseline",
    "dualffn",
    "fusion-up",
    "fusion-down",
    "fusion-all",
    "splitk",
];
const FIELDS: [&str; 5] = ["policy", "value", "misc", "moremisc", "ownership"];
const WIDTHS: [usize; 5] = [6 * 362, 3, 10, 8, 361];
const GATES: [f64; 5] = [0.05, 0.025, 0.01, 0.01, 0.005];
type OutputBits = [Vec<u32>; 5];

#[derive(Serialize, Deserialize)]
struct Case {
    batch: usize,
    variant: usize,
    outputs: OutputBits,
}

#[derive(Serialize, Deserialize)]
struct Report {
    mode: String,
    model_sha256: String,
    recipe_sha256: String,
    cases: Vec<Case>,
}

fn fixture() -> Option<PathBuf> {
    let path = std::env::var_os("KATAGO_QUANT_TEST_B11_MODEL")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("KATAGO_TEST_MODEL_DIR")
                .map(|dir| PathBuf::from(dir).join("b11c768h12nbt3tflrs-fson-silu.bin.gz"))
        });
    match path {
        Some(path) if path.is_file() => Some(path),
        Some(path) => {
            eprintln!(
                "SKIP mixed fast paths: missing B11 model {}",
                path.display()
            );
            None
        }
        None => {
            eprintln!(
                "SKIP mixed fast paths: set KATAGO_TEST_MODEL_DIR or KATAGO_QUANT_TEST_B11_MODEL"
            );
            None
        }
    }
}

fn input_variant(variant: usize) -> (Vec<f32>, Vec<f32>) {
    let mut spatial = Vec::with_capacity(8 * 22 * 361);
    let mut global = Vec::with_capacity(8 * 19);
    for row in 0..8 {
        let mut board = Board::new(19, 19);
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        let mut player = P_BLACK;
        let mut rand =
            kata_core::rng::Rand::new_from_seed(&format!("quant-fast-path-{variant}-{row}"));
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

fn output_bits(output: CudaOutputsHost, batch: usize, context: &str) -> OutputBits {
    let values = [
        output.policy,
        output.value,
        output.misc,
        output.moremisc,
        output.ownership,
    ];
    for ((field, width), values) in FIELDS.iter().zip(WIDTHS).zip(&values) {
        assert_eq!(
            values.len(),
            batch * width,
            "{context}/{field}: output shape"
        );
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
                "{context}/{field}[{index}]: expected {a:#010x} ({}), actual {b:#010x} ({})",
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

fn exercise_model(
    rt: &CudaRuntime,
    stream: &Arc<CudaStream>,
    model: &CudaModel,
    mode: &str,
) -> Vec<Case> {
    let inputs = [input_variant(0), input_variant(1)];
    assert_ne!(inputs[0].0, inputs[1].0);
    let mut cases = Vec::new();
    for batch in [1, 8] {
        let mut spatial = stream.clone_htod(&inputs[0].0[..batch * 22 * 361]).unwrap();
        let mut global = stream.clone_htod(&inputs[0].1[..batch * 19]).unwrap();
        let mut ws = CudaWorkspace::new(stream, model, batch).expect("graph workspace");
        // A direct call must not refresh graph scratch or output buffers and
        // thereby conceal a missing/stale operation in the captured graph.
        let mut direct_ws = CudaWorkspace::new(stream, model, batch).expect("direct workspace");
        model
            .apply(rt, stream, &mut direct_ws, &spatial, &global)
            .expect("direct warm");
        model
            .apply(rt, stream, &mut ws, &spatial, &global)
            .expect("pre-capture warm");
        rt.device.synchronize().expect("warm synchronize");
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .expect("begin capture");
        let applied = {
            set_capturing(true);
            let _capture = CaptureMode;
            model.apply(rt, stream, &mut ws, &spatial, &global)
        };
        let captured = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY,
        );
        applied.expect("captured apply");
        let graph = captured.expect("end capture").expect("captured graph");
        graph.upload().expect("upload graph");
        let mut first = None;
        for (round, variant) in [0, 1, 0].into_iter().enumerate() {
            stream
                .memcpy_htod(&inputs[variant].0[..batch * 22 * 361], &mut spatial)
                .unwrap();
            stream
                .memcpy_htod(&inputs[variant].1[..batch * 19], &mut global)
                .unwrap();
            let context = format!("{mode}/B{batch}/input{variant}/round{round}");
            model
                .apply(rt, stream, &mut direct_ws, &spatial, &global)
                .expect("fresh direct apply");
            let direct = output_bits(
                direct_ws.to_host(stream).expect("direct outputs"),
                batch,
                &context,
            );
            graph.launch().expect("replay graph");
            let replay = output_bits(ws.to_host(stream).expect("graph outputs"), batch, &context);
            assert_bits_equal(&direct, &replay, &format!("{context}/direct-vs-graph"));
            if round == 0 {
                first = Some(direct.clone());
            } else if round == 1 {
                assert_ne!(
                    first.as_ref().unwrap(),
                    &direct,
                    "{context}: stale-input check needs differing outputs"
                );
            } else {
                assert_bits_equal(
                    first.as_ref().unwrap(),
                    &direct,
                    &format!("{context}/return-to-A"),
                );
            }
            if round < 2 {
                cases.push(Case {
                    batch,
                    variant,
                    outputs: direct,
                });
            }
        }
        rt.device
            .synchronize()
            .expect("finish batch before dropping graph buffers");
        eprintln!("[quant-fast-path-test] mode={mode} batch={batch} direct_graph_aba=bitwise_pass");
    }
    cases
}

fn run_child(mode: &str, model_path: &Path) {
    assert!(MODES.contains(&mode), "unknown child mode {mode}");
    assert!(kata_nn::tactic_plan::installed_plan_id().is_none());
    let bytes = std::fs::read(model_path).expect("read native B11");
    let name = model_path.to_string_lossy().to_ascii_lowercase();
    assert!(
        name.ends_with(".bin.gz"),
        "this regression requires native B11 .bin.gz"
    );
    let model_sha256 = kata_core::hash::sha2::sha256_hex(&bytes);
    let desc =
        kata_nn::model_parser::load_model_from_bytes(&bytes, true, true).expect("parse native B11");
    let graph = kata_nn::native_model::lower_model(&desc).expect("lower native B11");
    assert_eq!(
        (graph.num_blocks, graph.mid_channels),
        (11, 384),
        "wrong B11 fixture"
    );
    let mut template = template_recipe(&graph, &model_sha256).expect("recipe template");
    // The second pair's INT8 down also exercises FUSION=up after a producer
    // that does not write the reusable FP16 residual buffer.
    let selected = [
        "trunk.block00.pair00.ffn.dual",
        "trunk.block00.pair01.ffn.down",
        "trunk.block00.pair00.attention.qkv",
        "trunk.block00.pair01.attention.out",
    ];
    let mut changed = 0;
    for projection in &mut template.projections {
        if selected.contains(&projection.id.as_str()) {
            projection.precision = Precision::Int8;
            changed += 1;
        }
    }
    assert_eq!(
        changed,
        selected.len(),
        "missing independent precision slots"
    );
    let recipe = resolve_recipe(&template, &graph, &model_sha256).expect("resolve mixed recipe");
    assert!(
        graph.layers.iter().zip(&recipe.layers).any(
            |(layer, p)| matches!(layer, Layer::Ffn(f) if f.hidden == 1152)
                && p.ffn_dual == Precision::Fp16
        ),
        "fixture must retain eligible FP16 DualFFN"
    );
    let rt = CudaRuntime::new().expect("CUDA runtime");
    validate_quantized_runtime(&rt).expect("freeze and validate real recipe tactics");
    let stream = rt.device.new_stream().expect("CUDA stream");
    let model =
        CudaModel::load_quantized(&graph, &rt, &stream, &recipe).expect("load mixed recipe");
    assert_eq!(model.int8_ffn_count(), 2);
    assert_eq!(
        model.quantization_recipe_id(),
        Some(recipe.recipe_sha256.as_str())
    );
    // This invokes the actual CUTLASS runtime probe and enables its execution
    // path. Merely setting the environment variable does not do so.
    ensure_requested_cuda_capabilities(&model).expect("requested tactics must be available");
    let cases = exercise_model(&rt, &stream, &model, mode);
    let report = Report {
        mode: mode.into(),
        model_sha256,
        recipe_sha256: recipe.recipe_sha256,
        cases,
    };
    let output = std::env::var_os(CHILD_OUTPUT).expect("child output path");
    std::fs::write(PathBuf::from(output), serde_json::to_vec(&report).unwrap())
        .expect("write child report");
}

fn assert_launches(log: &str, mode: &str) {
    let has = |marker: &str| log.lines().any(|line| line.contains(marker));
    assert!(
        has("[cuda-precision] stage=launched"),
        "{mode}: no recipe execution evidence"
    );
    for marker in [
        "op=ffn dual=int8 down=fp16",
        "op=ffn dual=fp16 down=int8",
        "op=attention qkv=int8 out=fp16",
        "op=attention qkv=fp16 out=int8",
    ] {
        assert!(has(marker), "{mode}: missing actual mixed route {marker}");
    }
    let dual = mode == "dualffn";
    assert_eq!(
        has("name=dual_ffn launch=fused effective=1"),
        dual,
        "{mode}: actual DualFFN launch"
    );
    if dual {
        assert!(
            has("name=dual_ffn requested=1 compiled=1 probe=pass handle=ready effective=1"),
            "DualFFN must pass the production capability probe"
        );
    }
    for (marker, expected) in [
        (
            "name=fusion launch=up",
            matches!(mode, "fusion-up" | "fusion-all"),
        ),
        (
            "name=fusion launch=down",
            matches!(mode, "fusion-down" | "fusion-all"),
        ),
        ("name=splitk launch=on", mode == "splitk"),
    ] {
        assert_eq!(
            has(marker),
            expected,
            "{mode}: actual launch marker {marker}"
        );
    }
    for batch in [1, 8] {
        assert!(
            has(&format!(
                "mode={mode} batch={batch} direct_graph_aba=bitwise_pass"
            )),
            "{mode}: missing B{batch} graph verification"
        );
    }
}

fn argmax(values: &[u32]) -> usize {
    let mut best = 0;
    for i in 1..values.len() {
        if f32::from_bits(values[i]) > f32::from_bits(values[best]) {
            best = i;
        }
    }
    best
}

fn compare_reports(reference: &Report, actual: &Report) {
    assert_eq!(
        reference.model_sha256, actual.model_sha256,
        "different model bytes"
    );
    assert_eq!(
        reference.recipe_sha256, actual.recipe_sha256,
        "different precision recipe"
    );
    assert_eq!(reference.cases.len(), 4);
    assert_eq!(actual.cases.len(), reference.cases.len());
    let mut maxima = [0.0_f64; 5];
    for (expected, actual_case) in reference.cases.iter().zip(&actual.cases) {
        assert_eq!(
            (expected.batch, expected.variant),
            (actual_case.batch, actual_case.variant)
        );
        let context = format!(
            "{}/B{}/input{}",
            actual.mode, expected.batch, expected.variant
        );
        if actual.mode == "dualffn" {
            // cuda_exec and the DualFFN contract promise the same half-rounding
            // boundaries and bitwise results, not merely an accuracy tolerance.
            assert_bits_equal(&expected.outputs, &actual_case.outputs, &context);
        }
        for field in 0..FIELDS.len() {
            assert_eq!(
                expected.outputs[field].len(),
                expected.batch * WIDTHS[field]
            );
            assert_eq!(
                actual_case.outputs[field].len(),
                expected.outputs[field].len()
            );
            for (index, (&a, &b)) in expected.outputs[field]
                .iter()
                .zip(&actual_case.outputs[field])
                .enumerate()
            {
                let (a, b) = (f32::from_bits(a) as f64, f32::from_bits(b) as f64);
                assert!(
                    a.is_finite() && b.is_finite(),
                    "{context}/{}[{index}]: nonfinite",
                    FIELDS[field]
                );
                let error = (a - b).abs();
                maxima[field] = maxima[field].max(error);
                assert!(
                    error <= GATES[field],
                    "{context}/{}[{index}]: error {error} exceeds original gate {} (baseline={a}, actual={b})",
                    FIELDS[field],
                    GATES[field]
                );
            }
        }
        for (channel, (a, b)) in expected.outputs[0]
            .chunks_exact(362)
            .zip(actual_case.outputs[0].chunks_exact(362))
            .enumerate()
        {
            assert_eq!(
                argmax(a),
                argmax(b),
                "{context}: policy row={} channel={} top-1 changed",
                channel / 6,
                channel % 6
            );
        }
    }
    eprintln!(
        "PASS_SAME_RECIPE_FAST_PATH_NOT_ACCURACY_CERTIFICATION: mode={} B1/B8 A/B maxima={maxima:?} bitwise_required={}",
        actual.mode,
        actual.mode == "dualffn"
    );
}

#[test]
fn mixed_recipe_preserves_fast_paths_and_graph_replay() {
    if let Ok(mode) = std::env::var(CHILD_MODE) {
        run_child(
            &mode,
            &fixture().expect("parent-selected fixture disappeared"),
        );
        return;
    }
    let Some(model_path) = fixture() else {
        return;
    };
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::var_os("KATAGO_QUANT_FAST_PATH_REPORT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/test_quantized_fast_paths")
        })
        .join(format!("{}-{nonce}", std::process::id()));
    std::fs::create_dir_all(&directory).expect("create report directory");
    eprintln!("mixed fast-path artifacts: {}", directory.display());
    let mut baseline = None;
    for mode in MODES {
        let output_path = directory.join(format!("{mode}.json"));
        let mut command = Command::new(std::env::current_exe().unwrap());
        command.args(["--exact", TEST_NAME, "--nocapture", "--test-threads=1"]);
        // Clean the child's environment only; no unsafe process-wide mutation
        // or inherited plan/profile/debug switch can contaminate a comparison.
        for (key, _) in std::env::vars_os() {
            if key.to_string_lossy().starts_with("KATAGO_CUDA_") {
                command.env_remove(key);
            }
        }
        for key in kata_nn::tactic_plan::ALLOWED_TACTIC_KEYS {
            command.env_remove(key);
        }
        command
            .env(CHILD_MODE, mode)
            .env(CHILD_OUTPUT, &output_path)
            .env("KATAGO_QUANT_TEST_B11_MODEL", &model_path)
            .env("KATAGO_CUDA_CUBLASLT", "1")
            .env("KATAGO_CUDA_GEMM_LAYOUT", "tn")
            .env(
                "KATAGO_CUDA_DUALFFN",
                if mode == "dualffn" { "1" } else { "0" },
            )
            .env(
                "KATAGO_CUDA_SPLITK",
                if mode == "splitk" { "1" } else { "0" },
            )
            .env(
                "KATAGO_CUDA_FUSION",
                mode.strip_prefix("fusion-").unwrap_or("none"),
            )
            .env("KATAGO_CUDA_ATTN", "fa2")
            .env("KATAGO_CUDA_ATTN_TILE", "q128")
            .env("KATAGO_CUDA_INT8_GEMM_TUNE", "0");
        let result = command.output().expect("run isolated tactic process");
        let log = format!(
            "stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        std::fs::write(directory.join(format!("{mode}.log")), &log).expect("save child log");
        assert!(
            result.status.success(),
            "{mode} failed; artifacts={}\n{log}",
            directory.display()
        );
        assert_launches(&log, mode);
        let report: Report =
            serde_json::from_slice(&std::fs::read(&output_path).expect("read child report"))
                .expect("parse child report");
        assert_eq!(report.mode, mode);
        if mode == "baseline" {
            baseline = Some(report);
        } else {
            compare_reports(baseline.as_ref().unwrap(), &report);
        }
    }
}
