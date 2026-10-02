//! Per-submission MXFP8 status through the public Backend pipeline API.
//!
//! Explicitly opt in with KATAGO_TEST_MXFP8_PIPELINE=1 and supply fixtures via
//! KATAGO_TEST_MODEL_DIR, KATAGO_QUANT_TEST_B11_MODEL,
//! KATAGO_QUANT_TEST_B15_MODEL or KATAGO_QUANT_TEST_ONNX_MODEL. Native defaults
//! match test_quantized_backend. This test launches separate Graph and direct
//! child processes, so frozen tactic state is never changed in one process.
//!
//! At physical B1/B8, warm both slots, submit NaN(slot 0), submit legal(slot 1),
//! wait until slot 1 completed, then finish slot 0 (must reject without output
//! mutation) and slot 1 (must match a separate handle's legal output bitwise).
//! Finally release the loaded model/context/reference, submit both slots on the
//! last owning handle, and drop it without finish to exercise cancellation.
//! These are backend logits/ownership, not postprocessed search probabilities.
//! This is an execution regression, not an accuracy or performance certificate.
#![cfg(feature = "cuda")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use kata_core::{
    config::Config,
    logger::{Logger, LoggerOptions},
};
use kata_game::{
    board::{get_opp, location, Board, P_BLACK},
    history::BoardHistory,
    rules::Rules,
};
use kata_nn::{
    backend::{Backend, ComputeHandle, Enabled, InputBuffers, NNOutput, NNResultBuf},
    backends::cuda::CudaQuantBackend,
    inputs::{fill_row_v7, MiscNNInputParams},
    onnx_parser::{Layer, LayerGraph},
    quantization_plan::{
        resolve_recipe, template_recipe, Precision, MXFP8_QUANTIZATION_SEMANTICS_VERSION,
        MXFP8_RECIPE_VERSION,
    },
};

const TEST_NAME: &str = "mxfp8_pipeline_preserves_each_submission_status";
const CHILD_MODE: &str = "KATAGO_MXFP8_PIPELINE_CHILD_MODE";
const MAX_BATCH: usize = 8;

struct Fixture {
    label: &'static str,
    path: PathBuf,
}

fn fixtures() -> Vec<Fixture> {
    let directory = std::env::var_os("KATAGO_TEST_MODEL_DIR").map(PathBuf::from);
    let mut result = Vec::new();
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
        let explicit = std::env::var_os(variable).map(PathBuf::from);
        if let Some(path) = &explicit {
            assert!(
                path.is_file(),
                "explicit {variable} fixture missing: {}",
                path.display()
            );
        }
        let path = explicit.or_else(|| directory.as_ref().map(|d| d.join(filename)));
        match path {
            Some(path) if path.is_file() => result.push(Fixture { label, path }),
            _ => eprintln!("SKIP {label}: set KATAGO_TEST_MODEL_DIR or {variable}"),
        }
    }
    if let Some(path) = std::env::var_os("KATAGO_QUANT_TEST_ONNX_MODEL").map(PathBuf::from) {
        assert!(
            path.is_file(),
            "explicit ONNX fixture missing: {}",
            path.display()
        );
        result.push(Fixture {
            label: "ONNX",
            path,
        });
    }
    assert!(
        !result.is_empty(),
        "MXFP8 pipeline opt-in requires at least one model fixture"
    );
    result
}

fn load_graph(path: &Path) -> (LayerGraph, String) {
    let bytes = std::fs::read(path).expect("read model fixture");
    let sha = kata_core::hash::sha2::sha256_hex(&bytes);
    let lower = path.to_string_lossy().to_ascii_lowercase();
    let graph = if lower.ends_with(".onnx") {
        kata_nn::onnx_model::parse_onnx_model(&bytes).expect("ONNX metadata");
        kata_nn::onnx_parser::parse_layer_graph(&bytes).expect("ONNX layer graph")
    } else {
        let compressed = lower.ends_with(".gz");
        let inner = lower.strip_suffix(".gz").unwrap_or(&lower);
        assert!(inner.ends_with(".bin") || inner.ends_with(".txt"));
        let desc = kata_nn::model_parser::load_model_from_bytes(
            &bytes,
            inner.ends_with(".bin"),
            compressed,
        )
        .expect("native model parse");
        kata_nn::native_model::lower_model(&desc).expect("native model lowering")
    };
    (graph, sha)
}

/// Own only this newly-created directory and its one recipe, never user files.
struct RecipeFile {
    directory: PathBuf,
    path: PathBuf,
}
impl Drop for RecipeFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
        let _ = std::fs::remove_dir(&self.directory);
    }
}

fn recipe_file(graph: &LayerGraph, sha: &str, label: &str, mode: &str) -> RecipeFile {
    let mut recipe = template_recipe(graph, sha).expect("model-bound FP16 template");
    recipe.version = MXFP8_RECIPE_VERSION;
    recipe.quantization_semantics_version = MXFP8_QUANTIZATION_SEMANTICS_VERSION;
    let mut selected = 0;
    for projection in &mut recipe.projections {
        if projection.id.contains(".ffn.") {
            projection.precision = Precision::Mxfp8;
            selected += 1;
        }
    }
    assert!(selected > 0, "fixture must have MXFP8 FFN projections");
    resolve_recipe(&recipe, graph, sha).expect("resolve actual graph shapes");
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir().join(format!(
        "rustgo-mxfp8-pipeline-{}-{nonce}-{label}-{mode}",
        std::process::id(),
    ));
    std::fs::create_dir(&directory).expect("new temporary recipe directory");
    let file = RecipeFile {
        path: directory.join("recipe.json"),
        directory,
    };
    std::fs::write(&file.path, serde_json::to_vec_pretty(&recipe).unwrap())
        .expect("write test recipe");
    file
}

fn legal_rows(batch: usize, variant: usize) -> Vec<NNResultBuf> {
    (0..batch)
        .map(|row| {
            let mut board = Board::new(19, 19);
            let mut hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
            let mut player = P_BLACK;
            let mut random =
                kata_core::rng::Rand::new_from_seed(&format!("mxfp8-pipeline-{variant}-{row}"));
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
                    legal[random.next_u64() as usize % legal.len()],
                    player,
                );
                player = get_opp(player);
            }
            let mut input = NNResultBuf {
                include_owner_map: true,
                board_x_size_for_server: 19,
                board_y_size_for_server: 19,
                row_spatial_buf: vec![0.0; 22 * 361],
                row_global_buf: vec![0.0; 19],
                symmetry: 0,
                policy_optimism: 0.0,
                ..NNResultBuf::default()
            };
            fill_row_v7(
                &board,
                &hist,
                player,
                &MiscNNInputParams::default(),
                19,
                19,
                false,
                &mut input.row_spatial_buf,
                &mut input.row_global_buf,
            );
            input
        })
        .collect()
}

#[derive(Debug, PartialEq, Eq)]
struct OutputSnapshot {
    hash: [u64; 2],
    dimensions: [i32; 2],
    values: Vec<u32>,
    ownership: Option<Vec<u32>>,
    noised_policy: Option<Vec<u32>>,
}

fn snapshot(output: &NNOutput) -> OutputSnapshot {
    let values = [
        output.white_win_prob,
        output.white_loss_prob,
        output.white_no_result_prob,
        output.white_score_mean,
        output.white_score_mean_sq,
        output.white_lead,
        output.var_time_left,
        output.shortterm_winloss_error,
        output.shortterm_score_error,
        output.policy_optimism_used,
    ]
    .into_iter()
    .chain(output.policy_probs.iter().copied())
    .map(f32::to_bits)
    .collect();
    OutputSnapshot {
        hash: [output.nn_hash.hash0, output.nn_hash.hash1],
        dimensions: [output.nn_x_len, output.nn_y_len],
        values,
        ownership: output
            .white_owner_map
            .as_ref()
            .map(|v| v.iter().copied().map(f32::to_bits).collect()),
        noised_policy: output
            .noised_policy_probs
            .as_ref()
            .map(|v| v.iter().copied().map(f32::to_bits).collect()),
    }
}

fn finite_snapshots(outputs: &[NNOutput]) -> Vec<OutputSnapshot> {
    outputs
        .iter()
        .map(|output| {
            let result = snapshot(output);
            assert_eq!(result.dimensions, [19, 19]);
            let owner = result.ownership.as_ref().expect("ownership output");
            assert_eq!(owner.len(), 361);
            assert!(
                result
                    .values
                    .iter()
                    .chain(owner)
                    .all(|v| f32::from_bits(*v).is_finite()),
                "nonfinite backend output"
            );
            result
        })
        .collect()
}

fn evaluate(
    backend: &dyn Backend,
    handle: &dyn ComputeHandle,
    buffers: &dyn InputBuffers,
    rows: &mut [NNResultBuf],
) -> Vec<NNOutput> {
    let mut outputs = vec![NNOutput::default(); rows.len()];
    backend
        .get_output(
            handle,
            buffers,
            rows.len() as i32,
            &mut rows.iter_mut().collect::<Vec<_>>(),
            &mut outputs.iter_mut().collect::<Vec<_>>(),
        )
        .expect("independent legal forward");
    finite_snapshots(&outputs);
    outputs
}

fn run_fixture(fixture: &Fixture, mode: &str) {
    let (graph, sha) = load_graph(&fixture.path);
    if fixture.label != "ONNX" {
        assert_eq!(
            graph.num_blocks,
            if fixture.label == "B11" { 11 } else { 15 }
        );
    }
    if fixture.label == "B15-pruned" {
        assert!(graph
            .layers
            .iter()
            .any(|l| matches!(l, Layer::Ffn(f) if f.hidden < 3 * graph.mid_channels)));
    }
    let recipe = recipe_file(&graph, &sha, fixture.label, mode);
    drop(graph);
    let backend: &dyn Backend = &CudaQuantBackend;
    let logger = Logger::new(LoggerOptions::default(), None);
    let cfg = Config::from_map(BTreeMap::from([
        ("nnBackend".into(), "cudaquantbackend".into()),
        (
            "cudaQuantPlan".into(),
            recipe.path.to_str().expect("Unicode recipe path").into(),
        ),
        ("nnMaxBatchSize".into(), MAX_BATCH.to_string()),
        ("numNNServerThreadsPerModel".into(), "2".into()),
        ("nnUseFP16".into(), "true".into()),
        ("nnUseNHWC".into(), "false".into()),
    ]));
    backend.global_initialize();
    let model = backend
        .load_model_file(fixture.path.to_str().expect("Unicode model path"), &sha)
        .expect("backend model load");
    let context = backend
        .create_compute_context(&[0], &logger, 19, 19, "", Enabled::True, &*model, &cfg)
        .expect("quantized context");
    let profile = model
        .inference_profile_id()
        .expect("actual loaded recipe profile");
    assert!(
        backend.supports_async_pipeline(),
        "unset KATAGO_CUDA_NOPIPELINE"
    );
    let reference = backend
        .create_compute_handle(
            &*context,
            &*model,
            &logger,
            MAX_BATCH as i32,
            true,
            false,
            0,
            0,
        )
        .expect("independent reference handle");
    let pipeline = backend
        .create_compute_handle(
            &*context,
            &*model,
            &logger,
            MAX_BATCH as i32,
            true,
            false,
            0,
            1,
        )
        .expect("pipeline handle");
    let buffers = backend
        .create_input_buffers(&*model, MAX_BATCH as i32, 19, 19)
        .expect("backend input buffers");
    for batch in [1, MAX_BATCH] {
        let expected = finite_snapshots(&evaluate(
            backend,
            &*reference,
            &*buffers,
            &mut legal_rows(batch, 1),
        ));
        // Valid warmup creates/captures this physical batch before injecting NaN.
        // Two completed submissions return the production round-robin to slot 0.
        for _ in 0..2 {
            evaluate(backend, &*pipeline, &*buffers, &mut legal_rows(batch, 0));
        }
        let mut bad = legal_rows(batch, 1);
        for row in &mut bad {
            row.row_spatial_buf.fill(f32::NAN);
        }
        let mut good = legal_rows(batch, 1);
        let bad_token = backend
            .submit_output(
                &*pipeline,
                &*buffers,
                batch as i32,
                &mut bad.iter_mut().collect::<Vec<_>>(),
            )
            .expect("submit invalid slot");
        let good_token = backend
            .submit_output(
                &*pipeline,
                &*buffers,
                batch as i32,
                &mut good.iter_mut().collect::<Vec<_>>(),
            )
            .expect("submit valid slot before finishing invalid slot");
        assert_ne!(
            bad_token, good_token,
            "two outstanding submissions need distinct slots"
        );
        // Wait for the later completion without consuming either slot. If finish
        // reads the shared status instead of its pinned snapshot, bad now passes.
        let deadline = Instant::now() + Duration::from_secs(120);
        while !backend.query_output_done(&*pipeline, good_token) {
            assert!(Instant::now() < deadline, "valid slot completion timed out");
            std::thread::sleep(Duration::from_millis(1));
        }
        let mut rejected_outputs = vec![NNOutput::default(); batch];
        for output in &mut rejected_outputs {
            output.nn_x_len = -1;
            output.nn_y_len = -1;
            output.white_win_prob = -12345.0;
            output.policy_probs.fill(-12345.0);
            output.white_owner_map = Some(vec![-12345.0; 361].into_boxed_slice());
        }
        let untouched = rejected_outputs.iter().map(snapshot).collect::<Vec<_>>();
        let rejected = backend.finish_output(
            &*pipeline,
            &*buffers,
            bad_token,
            batch as i32,
            &mut bad.iter_mut().collect::<Vec<_>>(),
            &mut rejected_outputs.iter_mut().collect::<Vec<_>>(),
        );
        // Drain the legal slot even when the rejection assertion would fail.
        let mut legal_outputs = vec![NNOutput::default(); batch];
        let accepted = backend.finish_output(
            &*pipeline,
            &*buffers,
            good_token,
            batch as i32,
            &mut good.iter_mut().collect::<Vec<_>>(),
            &mut legal_outputs.iter_mut().collect::<Vec<_>>(),
        );
        let error =
            rejected.expect_err("later valid reset must not erase the invalid slot's status");
        assert!(
            error.0.contains("MXFP8 forward rejected"),
            "unexpected failure: {error}"
        );
        assert_eq!(
            untouched,
            rejected_outputs.iter().map(snapshot).collect::<Vec<_>>(),
            "rejected batch must not publish or partially decode output"
        );
        accepted.expect("later valid slot must remain usable");
        assert_eq!(
            expected,
            finite_snapshots(&legal_outputs),
            "{}/{mode}/physical B{batch}: valid slot differs from fresh independent handle",
            fixture.label
        );
        eprintln!(
            "PASS_EXECUTION_ONLY model={} mode={mode} physical_batch={batch} bad_rejected=true good_bitwise=true profile={profile}",
            fixture.label
        );
    }
    // Hold only an independent reference to the primary CUDA context so errors
    // from the actual last Runtime/Model owner's teardown remain observable.
    // This observer owns neither CudaRuntime nor model/graph/weight resources.
    let observer = cudarc::driver::CudaContext::new(0).expect("teardown observer context");
    drop(reference);
    drop(context);
    drop(model);
    let mut cancelled_a = legal_rows(MAX_BATCH, 1);
    let mut cancelled_b = legal_rows(MAX_BATCH, 2);
    let token_a = backend
        .submit_output(
            &*pipeline,
            &*buffers,
            MAX_BATCH as i32,
            &mut cancelled_a.iter_mut().collect::<Vec<_>>(),
        )
        .expect("submit first cancelled slot on last model owner");
    let token_b = backend
        .submit_output(
            &*pipeline,
            &*buffers,
            MAX_BATCH as i32,
            &mut cancelled_b.iter_mut().collect::<Vec<_>>(),
        )
        .expect("submit second cancelled slot on last model owner");
    assert_ne!(token_a, token_b);
    // Readiness is diagnostic, not a timing assertion: completion before the
    // host reaches Drop is legal. In either case neither slot is ever finished.
    let pending_when_dropped = !backend.query_output_done(&*pipeline, token_b);
    drop(pipeline);
    observer
        .synchronize()
        .expect("last-owner teardown must not free resources used by pending inference");
    eprintln!(
        "PASS_LIFETIME_ONLY model={} mode={mode} physical_batch={MAX_BATCH} last_model_owner=true submit_without_finish=true pending_when_dropped={pending_when_dropped}",
        fixture.label
    );
    drop(buffers);
    backend.global_cleanup();
}

#[test]
fn mxfp8_pipeline_preserves_each_submission_status() {
    if std::env::var("KATAGO_TEST_MXFP8_PIPELINE").as_deref() != Ok("1") {
        eprintln!("SKIP: set KATAGO_TEST_MXFP8_PIPELINE=1 for real MXFP8 pipeline regression");
        return;
    }
    if let Ok(mode) = std::env::var(CHILD_MODE) {
        assert!(mode == "graph" || mode == "direct", "invalid child mode");
        assert_eq!(
            std::env::var("KATAGO_CUDA_NOGRAPH").as_deref(),
            Ok(if mode == "direct" { "1" } else { "0" })
        );
        assert!(kata_nn::tactic_plan::installed_plan_id().is_none());
        for fixture in fixtures() {
            run_fixture(&fixture, &mode);
        }
        return;
    }
    // Validate opt-in fixtures before launching any child or initializing CUDA.
    fixtures();
    for (mode, nograph, launch_marker) in [
        (
            "graph",
            "0",
            "name=graph requested=graph launch=graph effective=graph",
        ),
        (
            "direct",
            "1",
            "name=graph requested=direct launch=direct effective=direct fallback=0",
        ),
    ] {
        let output = Command::new(std::env::current_exe().expect("current test executable"))
            .args(["--exact", TEST_NAME, "--nocapture", "--test-threads=1"])
            .env(CHILD_MODE, mode)
            .env("KATAGO_CUDA_NOGRAPH", nograph)
            .output()
            .expect("spawn isolated pipeline mode");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        print!("{stdout}");
        eprint!("{stderr}");
        assert!(
            output.status.success(),
            "MXFP8 pipeline {mode} child failed: {}",
            output.status
        );
        assert!(
            stderr.contains(launch_marker),
            "{mode}: actual backend launch marker missing"
        );
        assert!(stderr.contains("physical_batch=1 bad_rejected=true good_bitwise=true"));
        assert!(stderr.contains("physical_batch=8 bad_rejected=true good_bitwise=true"));
        assert!(stderr.contains("last_model_owner=true submit_without_finish=true"));
        assert!(
            !stderr.contains("[cuda-handle-drop] owner-stream synchronization failed"),
            "{mode}: destructor could not synchronize its outstanding submissions"
        );
    }
}
