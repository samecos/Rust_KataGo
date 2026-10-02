//! Whole-FFN cost diagnostics on explicit encoded inputs. Never an ABBA result.
//! CPU fixture: cargo run -p kata_nn --example ffn_group_cost -- --make-smoke-input NEW_DIR
//! GPU: --model MODEL --recipe RECIPE --inputs INPUT_MANIFEST --output NEW_DIR
//!      [--warmup 3] [--iterations 8], compiled with --features cuda.
//! Input files are exact little-endian f32 NCHW tensors; their hashes, physical
//! batch and provenance are preserved. This tool does not certify corpus splits.

use serde::{Deserialize, Serialize};
#[cfg(feature = "cuda")]
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    io::Write,
    path::{Component, Path, PathBuf},
};

type Result<T> = std::result::Result<T, String>;
const SCHEMA: &str = "rustgo-encoded-group-cost-input-v1";
const MAX_BYTES: usize = 64 * 22 * 361 * 4;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FileBinding {
    path: String,
    bytes: usize,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Input {
    id: String,
    physical_batch: usize,
    spatial: FileBinding,
    global: FileBinding,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    // Provenance is descriptive, not a declaration that the tensors were
    // independently verified against a corpus or Worker encoder.
    purpose: String,
    provenance: String,
    inputs: Vec<Input>,
}

#[derive(Debug)]
struct Encoded {
    metadata: Input,
    spatial: Vec<f32>,
    global: Vec<f32>,
    label: String,
}

fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn require(ok: bool, why: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(why.into()) }
}
fn read_bound(root: &Path, binding: &FileBinding, expected_floats: usize) -> Result<Vec<f32>> {
    let path = Path::new(&binding.path);
    require(
        !binding.path.is_empty() && path.components().all(|c| matches!(c, Component::Normal(_))),
        "input tensor path must be relative without traversal",
    )?;
    let expected = expected_floats
        .checked_mul(4)
        .ok_or("input size overflow")?;
    require(
        expected <= MAX_BYTES && binding.bytes == expected,
        "input byte count differs from physical shape",
    )?;
    let full = root.join(path);
    let len = fs::metadata(&full).map_err(|e| e.to_string())?.len();
    require(
        len == expected as u64,
        "input tensor file length differs from binding",
    )?;
    let raw = fs::read(full).map_err(|e| e.to_string())?;
    require(
        raw.len() == expected && hash(&raw) == binding.sha256,
        "input tensor SHA256/length mismatch",
    )?;
    let values: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    require(
        values.iter().all(|v| v.is_finite()),
        "input tensor contains NaN/Inf",
    )?;
    Ok(values)
}

fn load_inputs(path: &Path) -> Result<(Manifest, String, Vec<Encoded>)> {
    require(
        fs::metadata(path).map_err(|e| e.to_string())?.len() <= 1024 * 1024,
        "input manifest too large",
    )?;
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let manifest: Manifest = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    require(manifest.schema == SCHEMA, "unknown encoded input schema")?;
    require(
        matches!(
            manifest.purpose.as_str(),
            "synthetic-diagnostic" | "calibration"
        ),
        "group cost input purpose must be synthetic-diagnostic or calibration",
    )?;
    require(
        !manifest.provenance.is_empty() && manifest.provenance.len() <= 4096,
        "missing/oversized provenance",
    )?;
    require(
        !manifest.inputs.is_empty() && manifest.inputs.len() <= 128,
        "input count outside 1..128",
    )?;
    let mut names = BTreeSet::new();
    let mut total_bytes = 0usize;
    let mut result = Vec::new();
    for input in &manifest.inputs {
        require(
            !input.id.is_empty() && input.id.len() <= 128 && names.insert(input.id.clone()),
            "duplicate/invalid input ID",
        )?;
        require(
            (1..=64).contains(&input.physical_batch),
            "physical batch outside 1..64",
        )?;
        total_bytes = total_bytes
            .checked_add(input.physical_batch * (22 * 361 + 19) * 4)
            .ok_or("input sum overflow")?;
        require(
            total_bytes <= 128 * 1024 * 1024,
            "input bundle exceeds 128 MiB",
        )?;
        let root = path.parent().ok_or("input manifest has no parent")?;
        let spatial = read_bound(root, &input.spatial, input.physical_batch * 22 * 361)?;
        let global = read_bound(root, &input.global, input.physical_batch * 19)?;
        require(
            spatial
                .chunks_exact(22 * 361)
                .all(|row| row[..361].iter().all(|v| *v == 1.0)),
            "this backend requires the complete 19x19 on-board plane",
        )?;
        let mut digest = Sha256::new();
        digest.update(b"rustgo-encoded-group-cost-tensors-v1\0");
        digest.update((input.physical_batch as u64).to_le_bytes());
        for value in spatial.iter().chain(&global) {
            digest.update(value.to_le_bytes());
        }
        result.push(Encoded {
            metadata: input.clone(),
            spatial,
            global,
            label: hex::encode(digest.finalize()),
        });
    }
    Ok((manifest, hash(&bytes), result))
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    file.write_all(bytes).map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())
}
fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    write_new(
        path,
        &serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
}
fn tensor(root: &Path, name: &str, values: &[f32]) -> Result<FileBinding> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_new(&root.join(name), &bytes)?;
    Ok(FileBinding {
        path: name.into(),
        bytes: bytes.len(),
        sha256: hash(&bytes),
    })
}

fn smoke_inputs(root: &Path) -> Result<()> {
    use kata_game::{
        board::{Board, P_BLACK, get_opp, location},
        history::BoardHistory,
        rules::Rules,
    };
    use kata_nn::inputs::{MiscNNInputParams, fill_row_v7};
    fs::create_dir(root).map_err(|e| e.to_string())?;
    let mut spatial = Vec::new();
    let mut global = Vec::new();
    for row in 0..8 {
        let mut board = Board::new(19, 19);
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        let mut player = P_BLACK;
        let mut rand = kata_core::rng::Rand::new_from_seed(&format!("group-cost-smoke-v1-{row}"));
        for _ in 0..17 + row * 11 {
            let legal: Vec<_> = (0..361)
                .map(|i| location::get_loc(i % 19, i / 19, 19))
                .filter(|&loc| hist.is_legal(&board, loc, player))
                .collect();
            require(
                !legal.is_empty(),
                "unexpected exhausted synthetic legal positions",
            )?;
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
    let mut inputs = Vec::new();
    // B3 crosses the INT8 1024-row quantizer dispatch boundary; B8 exercises
    // wider physical-batch cache entries. These are diagnostic shapes only.
    for batch in [1, 3, 8] {
        inputs.push(Input {
            id: format!("synthetic-b{batch}"),
            physical_batch: batch,
            spatial: tensor(
                root,
                &format!("b{batch}-spatial.f32le"),
                &spatial[..batch * 22 * 361],
            )?,
            global: tensor(
                root,
                &format!("b{batch}-global.f32le"),
                &global[..batch * 19],
            )?,
        });
    }
    write_json(&root.join("inputs.json"), &Manifest { schema: SCHEMA.into(), purpose: "synthetic-diagnostic".into(),
        provenance: "Deterministic legal synthetic positions encoded with fill_row_v7; no corpus/holdout coverage claim".into(), inputs })
}

#[derive(Debug)]
struct Options {
    model: PathBuf,
    recipe: PathBuf,
    inputs: PathBuf,
    output: PathBuf,
    warmup: usize,
    iterations: usize,
}
fn parse(args: Vec<String>) -> Result<Options> {
    require(args.len() % 2 == 0, "arguments require --name value pairs")?;
    let mut values = std::collections::BTreeMap::new();
    for pair in args.chunks_exact(2) {
        require(
            [
                "--model",
                "--recipe",
                "--inputs",
                "--output",
                "--warmup",
                "--iterations",
            ]
            .contains(&pair[0].as_str()),
            "unknown argument",
        )?;
        require(
            values.insert(pair[0].clone(), pair[1].clone()).is_none(),
            "duplicate argument",
        )?;
    }
    let path = |name: &str| -> Result<PathBuf> {
        Ok(values
            .get(name)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| format!("missing {name}"))?
            .into())
    };
    let number = |name: &str, default: usize| -> Result<usize> {
        values.get(name).map_or(Ok(default), |s| {
            s.parse().map_err(|_| format!("invalid {name}"))
        })
    };
    let result = Options {
        model: path("--model")?,
        recipe: path("--recipe")?,
        inputs: path("--inputs")?,
        output: path("--output")?,
        warmup: number("--warmup", 3)?,
        iterations: number("--iterations", 8)?,
    };
    require(
        (2..=16).contains(&result.warmup) && (1..=128).contains(&result.iterations),
        "warmup/iterations outside finite bounds",
    )?;
    Ok(result)
}

fn native_format(path: &Path) -> Result<Option<(bool, bool)>> {
    let lower = path.to_string_lossy().to_ascii_lowercase();
    if lower.ends_with(".onnx") {
        return Ok(None);
    }
    let compressed = lower.ends_with(".gz");
    let inner = lower.strip_suffix(".gz").unwrap_or(&lower);
    let binary = if inner.ends_with(".bin") {
        true
    } else if inner.ends_with(".txt") {
        false
    } else {
        return Err("model must be .onnx, .bin[.gz], or .txt[.gz]".into());
    };
    Ok(Some((binary, compressed)))
}

#[cfg(feature = "cuda")]
fn run(options: &Options) -> Result<()> {
    use kata_nn::backends::{
        cuda::{CudaRuntime, ensure_requested_cuda_capabilities, validate_quantized_runtime},
        cuda_exec::{CudaModel, CudaOutputsHost, CudaWorkspace},
        group_cost::Phase,
    };
    use kata_nn::quantization_plan::{PrecisionRecipe, resolve_recipe};
    fn bits(output: CudaOutputsHost, batch: usize) -> Result<[Vec<u32>; 5]> {
        let values = [
            output.policy,
            output.value,
            output.misc,
            output.moremisc,
            output.ownership,
        ];
        for (v, width) in values.iter().zip([6 * 362, 3, 10, 8, 361]) {
            require(
                v.len() == batch * width && v.iter().all(|x| x.is_finite()),
                "invalid raw output shape/nonfinite",
            )?;
        }
        Ok(values.map(|v| v.into_iter().map(f32::to_bits).collect()))
    }
    let (manifest, manifest_sha, inputs) = load_inputs(&options.inputs)?;
    let model_bytes = fs::read(&options.model).map_err(|e| e.to_string())?;
    let model_sha = hash(&model_bytes);
    let graph = match native_format(&options.model)? {
        None => kata_nn::onnx_parser::parse_layer_graph(&model_bytes)?,
        Some((binary, compressed)) => {
            let desc =
                kata_nn::model_parser::load_model_from_bytes(&model_bytes, binary, compressed)
                    .map_err(|e| e.to_string())?;
            kata_nn::native_model::lower_model(&desc)?
        }
    };
    let recipe_bytes = fs::read(&options.recipe).map_err(|e| e.to_string())?;
    let source: PrecisionRecipe =
        serde_json::from_slice(&recipe_bytes).map_err(|e| e.to_string())?;
    let recipe = resolve_recipe(&source, &graph, &model_sha)?;
    fs::create_dir(&options.output).map_err(|e| e.to_string())?;
    write_json(
        &options.output.join("intent.json"),
        &json!({"schema":"rustgo-ffn-cost-diagnostic-v1",
        "purpose": manifest.purpose, "provenance": manifest.provenance, "input_manifest_sha256":manifest_sha,
        "inputs":manifest.inputs, "model_sha256":model_sha, "source_recipe_sha256":hash(&recipe_bytes),
        "resolved_recipe_sha256":recipe.recipe_sha256, "warmup":options.warmup,"iterations":options.iterations,
        "scope":"DIRECT_DIAGNOSTIC; device FFN intervals exclude input transfer and output serialization; no ABBA or accuracy certification"}),
    )?;
    let rt = CudaRuntime::new()?;
    validate_quantized_runtime(&rt)?;
    let stream = rt.device.new_stream().map_err(|e| e.to_string())?;
    let model = CudaModel::load_quantized(&graph, &rt, &stream, &recipe)?;
    ensure_requested_cuda_capabilities(&model)?;
    let mut checks = Vec::new();
    for (index, input) in inputs.iter().enumerate() {
        let batch = input.metadata.physical_batch;
        let spatial = stream
            .clone_htod(&input.spatial)
            .map_err(|e| e.to_string())?;
        let global = stream
            .clone_htod(&input.global)
            .map_err(|e| e.to_string())?;
        let mut ws = CudaWorkspace::new(&stream, &model, batch)?;
        let mut recorder = model.prepare_group_cost(&rt, &stream, &mut ws, &graph, &recipe)?;
        // First diagnostic observes setup/cache misses; later warmup must be
        // setup-free. Do not prewarm and conceal these markers.
        for iteration in 0..options.warmup {
            model.apply_with_group_cost(
                &rt,
                &stream,
                &mut ws,
                &spatial,
                &global,
                &mut recorder,
                Phase::Warmup,
                &input.label,
            )?;
            write_json(
                &options
                    .output
                    .join(format!("input-{index:03}-warmup-{iteration:03}.json")),
                &recorder.drain()?,
            )?;
        }
        require(
            recorder.ready_to_measure(),
            "bounded warmup did not establish a setup-free route",
        )?;
        model.apply(&rt, &stream, &mut ws, &spatial, &global)?;
        let expected = bits(ws.to_host(&stream)?, batch)?;
        let mut hashes = Vec::new();
        for (head, values) in expected.iter().enumerate() {
            let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let name = format!("input-{index:03}-head-{head}.f32le");
            write_new(&options.output.join(&name), &raw)?;
            hashes.push(json!({"path":name,"bytes":raw.len(),"sha256":hash(&raw)}));
        }
        for iteration in 0..options.iterations {
            model.apply_with_group_cost(
                &rt,
                &stream,
                &mut ws,
                &spatial,
                &global,
                &mut recorder,
                Phase::Measure,
                &input.label,
            )?;
            let sample = recorder.drain()?;
            let actual = bits(ws.to_host(&stream)?, batch)?;
            let actual_hashes: Vec<_> = actual
                .iter()
                .map(|values| {
                    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                    json!({"bytes":bytes.len(),"sha256":hash(&bytes)})
                })
                .collect();
            let identical = actual == expected;
            write_json(
                &options
                    .output
                    .join(format!("input-{index:03}-measure-{iteration:03}.json")),
                &json!({"sample":sample,"raw_output_heads":actual_hashes,"bitwise_matches_reference":identical}),
            )?;
            require(
                identical,
                "instrumented output differs bitwise from uninstrumented apply",
            )?;
        }
        model.apply(&rt, &stream, &mut ws, &spatial, &global)?;
        require(
            bits(ws.to_host(&stream)?, batch)? == expected,
            "post-diagnostic uninstrumented output changed",
        )?;
        checks.push(json!({"input_id":input.metadata.id,"input_tensor_sha256":input.label,"physical_batch":batch,
            "measured_output_checks":options.iterations,"raw_reference_heads":hashes,"before_during_after_bitwise":"PASS"}));
    }
    write_json(
        &options.output.join("result.json"),
        &json!({"status":"DIRECT_DIAGNOSTIC_COMPLETE",
        "performance_adoption":false,"quantization_accuracy_certified":false,"checks":checks}),
    )
}

#[cfg(not(feature = "cuda"))]
fn run(_options: &Options) -> Result<()> {
    Err("GPU diagnostics require --features cuda".into())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = if args.len() == 2 && args[0] == "--make-smoke-input" {
        smoke_inputs(Path::new(&args[1]))
    } else {
        parse(args).and_then(|options| run(&options))
    };
    if let Err(error) = result {
        eprintln!("ffn_group_cost: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            Self(std::env::temp_dir().join(format!(
                    "rustgo-group-cost-{}-{}",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                )))
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn encoded_smoke_roundtrip_and_content_corruption_rejected() {
        let temp = Temp::new();
        smoke_inputs(&temp.0).unwrap();
        let (manifest, _, inputs) = load_inputs(&temp.0.join("inputs.json")).unwrap();
        assert_eq!(
            inputs
                .iter()
                .map(|x| x.metadata.physical_batch)
                .collect::<Vec<_>>(),
            [1, 3, 8]
        );
        assert_eq!(inputs[2].spatial.len(), 8 * 22 * 361);
        assert_eq!(inputs[2].global.len(), 8 * 19);
        assert_ne!(inputs[0].label, inputs[1].label);
        let path = temp.0.join(&manifest.inputs[0].global.path);
        let mut bytes = fs::read(&path).unwrap();
        bytes[0] ^= 1;
        fs::write(path, bytes).unwrap();
        assert!(
            load_inputs(&temp.0.join("inputs.json"))
                .unwrap_err()
                .contains("SHA256")
        );
    }
    #[test]
    fn malformed_or_nonfinite_tensor_rejected() {
        let temp = Temp::new();
        fs::create_dir(&temp.0).unwrap();
        let binding = tensor(&temp.0, "nan.f32le", &[f32::NAN]).unwrap();
        assert!(
            read_bound(&temp.0, &binding, 1)
                .unwrap_err()
                .contains("NaN")
        );
        assert!(
            read_bound(&temp.0, &binding, 2)
                .unwrap_err()
                .contains("byte count")
        );
        let traversal = FileBinding {
            path: "../nan.f32le".into(),
            ..binding
        };
        assert!(
            read_bound(&temp.0, &traversal, 1)
                .unwrap_err()
                .contains("relative")
        );
    }
    #[test]
    fn holdout_and_duplicate_shapes_fail_before_gpu() {
        let temp = Temp::new();
        smoke_inputs(&temp.0).unwrap();
        let path = temp.0.join("inputs.json");
        let mut manifest: Manifest = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        manifest.purpose = "holdout".into();
        fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(load_inputs(&path).unwrap_err().contains("purpose"));
        manifest.purpose = "calibration".into();
        manifest.inputs[1].id = manifest.inputs[0].id.clone();
        fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(load_inputs(&path).unwrap_err().contains("duplicate"));
    }
    #[test]
    fn cli_rejects_unbounded_or_duplicate_arguments() {
        let base: Vec<String> = [
            "--model", "m", "--recipe", "r", "--inputs", "i", "--output", "o",
        ]
        .map(str::to_owned)
        .into();
        assert_eq!(parse(base.clone()).unwrap().warmup, 3);
        for extra in [
            ["--iterations", "0"],
            ["--iterations", "129"],
            ["--warmup", "1"],
            ["--model", "x"],
        ] {
            let mut args = base.clone();
            args.extend(extra.map(str::to_owned));
            assert!(parse(args).is_err());
        }
    }

    #[test]
    fn model_formats_match_supported_loader_suffixes() {
        for (name, expected) in [
            ("m.ONNX", None),
            ("m.bin", Some((true, false))),
            ("m.BIN.GZ", Some((true, true))),
            ("m.txt", Some((false, false))),
            ("m.TXT.GZ", Some((false, true))),
        ] {
            assert_eq!(native_format(Path::new(name)).unwrap(), expected);
        }
        for name in ["m", "m.gz", "m.onnx.gz", "m.bin.zip"] {
            assert!(native_format(Path::new(name)).is_err());
        }
    }

    /// Explicitly run with --ignored and KATAGO_TEST_MODEL_DIR. The rejected
    /// calls must return before the forward loop; model uploads are real GPU
    /// work. This is an API-boundary test, not a cost/performance measurement.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a native B11 model and CUDA"]
    fn gpu_rejects_foreign_workspace_and_short_input_before_forward() {
        use kata_nn::backends::{
            cuda::CudaRuntime,
            cuda_exec::{CudaModel, CudaWorkspace},
            group_cost::Phase,
        };
        use kata_nn::quantization_plan::{resolve_recipe, template_recipe};
        let directory =
            std::env::var_os("KATAGO_TEST_MODEL_DIR").expect("explicit model directory required");
        let bytes = fs::read(PathBuf::from(directory).join("b11c768h12nbt3tflrs-fson-silu.bin.gz"))
            .unwrap();
        let desc = kata_nn::model_parser::load_model_from_bytes(&bytes, true, true).unwrap();
        let graph = kata_nn::native_model::lower_model(&desc).unwrap();
        let model_sha = hash(&bytes);
        let recipe = resolve_recipe(
            &template_recipe(&graph, &model_sha).unwrap(),
            &graph,
            &model_sha,
        )
        .unwrap();
        let rt = CudaRuntime::new().unwrap();
        let stream = rt.device.new_stream().unwrap();
        let first = CudaModel::load_quantized(&graph, &rt, &stream, &recipe).unwrap();
        let second = CudaModel::load_quantized(&graph, &rt, &stream, &recipe).unwrap();
        let mut ws = CudaWorkspace::new(&stream, &first, 1).unwrap();
        let error = match second.prepare_group_cost(&rt, &stream, &mut ws, &graph, &recipe) {
            Ok(_) => panic!("foreign creation model accepted"),
            Err(e) => e,
        };
        assert!(error.contains("another model"));
        let mut recorder = first
            .prepare_group_cost(&rt, &stream, &mut ws, &graph, &recipe)
            .unwrap();
        let short = stream.clone_htod(&[0.0f32]).unwrap();
        let global = stream.clone_htod(&[0.0f32; 19]).unwrap();
        let label = "a".repeat(64);
        let error = first
            .apply_with_group_cost(
                &rt,
                &stream,
                &mut ws,
                &short,
                &global,
                &mut recorder,
                Phase::Warmup,
                &label,
            )
            .unwrap_err();
        assert!(error.contains("input lengths"));
        let error = second
            .apply_with_group_cost(
                &rt,
                &stream,
                &mut ws,
                &short,
                &global,
                &mut recorder,
                Phase::Warmup,
                &label,
            )
            .unwrap_err();
        assert!(error.contains("another model"));
        assert!(!recorder.ready_to_measure());
        let other_rt = CudaRuntime::new().unwrap();
        let other_stream = other_rt.device.new_stream().unwrap();
        assert!(CudaWorkspace::new(&other_stream, &first, 1).is_err());
        let error = match first.prepare_group_cost(&other_rt, &stream, &mut ws, &graph, &recipe) {
            Ok(_) => panic!("foreign context accepted"),
            Err(e) => e,
        };
        assert!(error.contains("context"));
        rt.device.synchronize().unwrap();
    }
}
