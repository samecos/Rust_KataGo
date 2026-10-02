//! CPU-only inspection of models accepted by the CUDA layer-graph loader.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use kata_nn::onnx_parser::LayerGraph;
use kata_nn::quantization_plan::{graph_manifest, resolve_recipe_identity, template_recipe};

#[derive(Debug, Parser)]
#[command(
    about = "Inspect model shapes and write an uncalibrated FP16 recipe without a GPU",
    after_help = "Writes model-manifest.json and fp16-recipe.json to a NEW output directory.\nWith --recipe, also writes recipe-identity.json and an exact source-recipe.json copy.\nSupported models: .onnx, .bin[.gz], .txt[.gz]. No calibration, accuracy certification, or performance measurement is performed."
)]
struct InspectArgs {
    /// Exact source model file; its original bytes determine the SHA-256.
    #[arg(long, value_name = "FILE")]
    model: PathBuf,
    /// New output directory; existing directories and artifacts are never replaced.
    #[arg(long, value_name = "DIR")]
    output: PathBuf,
    /// Validate and export a recipe's complete canonical identity for this model.
    #[arg(long, value_name = "FILE")]
    recipe: Option<PathBuf>,
}

pub fn quant_inspect(args: &[String]) -> i32 {
    let parsed = match InspectArgs::try_parse_from(
        std::iter::once("quant-inspect").chain(args.iter().map(String::as_str)),
    ) {
        Ok(parsed) => parsed,
        Err(error) => {
            let code = error.exit_code();
            let _ = error.print();
            return code;
        }
    };
    match run(parsed) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("quant-inspect: {error:#}");
            1
        }
    }
}

fn run(args: InspectArgs) -> Result<()> {
    if args
        .output
        .try_exists()
        .with_context(|| format!("inspect output directory {}", args.output.display()))?
    {
        bail!(
            "output directory already exists: {}; choose a new directory",
            args.output.display()
        );
    }
    let bytes =
        fs::read(&args.model).with_context(|| format!("read model {}", args.model.display()))?;
    // Hash the same bytes passed to the parser, including gzip headers when
    // compressed. Do not hash the lowered graph as a substitute model identity.
    let model_sha256 = kata_core::hash::sha2::sha256_hex(&bytes);
    let graph = parse_graph(&args.model, &bytes)?;
    let manifest = graph_manifest(&graph, &model_sha256)
        .map_err(anyhow::Error::msg)
        .context("build model manifest")?;
    let recipe = template_recipe(&graph, &model_sha256)
        .map_err(anyhow::Error::msg)
        .context("build FP16 recipe template")?;
    let manifest_json = serde_json::to_vec_pretty(&manifest).context("serialize model manifest")?;
    let recipe_json = serde_json::to_vec_pretty(&recipe).context("serialize FP16 recipe")?;
    // Read, validate and serialize every optional artifact before reserving the
    // output directory. The source hash and frozen copy use these exact bytes.
    let identity = args
        .recipe
        .as_deref()
        .map(|path| -> Result<_> {
            let source = fs::read(path)
                .with_context(|| format!("read precision recipe {}", path.display()))?;
            let identity = resolve_recipe_identity(&source, &graph, &model_sha256)
                .map_err(anyhow::Error::msg)
                .context("resolve precision recipe identity")?;
            // Serialize the typed envelope directly; Value maps would reorder
            // canonical fields and make independent SHA recomputation fail.
            let json = serde_json::to_vec_pretty(&identity)
                .context("serialize precision recipe identity")?;
            Ok((source, json, identity.recipe_sha256))
        })
        .transpose()?;
    publish_artifacts(
        &args.output,
        &manifest_json,
        &recipe_json,
        identity
            .as_ref()
            .map(|(source, json, _)| (source.as_slice(), json.as_slice())),
    )?;
    println!("model-sha256={model_sha256}");
    println!(
        "blocks={} trunk={} mid={} attention-heads={}",
        graph.num_blocks, graph.trunk_channels, graph.mid_channels, graph.num_heads
    );
    println!(
        "manifest={}",
        args.output.join("model-manifest.json").display()
    );
    println!(
        "fp16-recipe={}",
        args.output.join("fp16-recipe.json").display()
    );
    if let Some((_, _, recipe_sha256)) = identity {
        println!("recipe-sha256={recipe_sha256}");
        println!(
            "recipe-identity={}",
            args.output.join("recipe-identity.json").display()
        );
        println!(
            "source-recipe={}",
            args.output.join("source-recipe.json").display()
        );
    }
    println!(
        "CPU inspection only: FP16 template is not calibrated or certified; no GPU inference or performance test was run."
    );
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
enum ModelFormat {
    Onnx,
    Native { binary: bool, compressed: bool },
}

fn model_format(model: &Path) -> Result<ModelFormat> {
    let lower = model.to_string_lossy().to_ascii_lowercase();
    if lower.ends_with(".onnx") {
        return Ok(ModelFormat::Onnx);
    }
    let compressed = lower.ends_with(".gz");
    let inner = lower.strip_suffix(".gz").unwrap_or(&lower);
    let binary = if inner.ends_with(".bin") {
        true
    } else if inner.ends_with(".txt") {
        false
    } else {
        bail!("model must be .onnx, .bin[.gz], or .txt[.gz]");
    };
    Ok(ModelFormat::Native { binary, compressed })
}

fn parse_graph(model: &Path, bytes: &[u8]) -> Result<LayerGraph> {
    match model_format(model)? {
        ModelFormat::Onnx => {
            // Match both validations in CudaBackend::load_model_file.
            kata_nn::onnx_model::parse_onnx_model(bytes)
                .map_err(anyhow::Error::msg)
                .context("parse ONNX metadata")?;
            kata_nn::onnx_parser::parse_layer_graph(bytes)
                .map_err(anyhow::Error::msg)
                .context("parse ONNX layer graph")
        }
        ModelFormat::Native { binary, compressed } => {
            let desc = kata_nn::model_parser::load_model_from_bytes(bytes, binary, compressed)
                .context("parse native model")?;
            kata_nn::native_model::lower_model(&desc)
                .map_err(anyhow::Error::msg)
                .context("native model is unsupported by the CUDA layer graph")
        }
    }
}

fn publish_artifacts(
    output: &Path,
    manifest: &[u8],
    recipe: &[u8],
    identity: Option<(&[u8], &[u8])>,
) -> Result<()> {
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .with_context(|| format!("create output parent {}", parent.display()))?;
    }
    // Reserve a fresh directory before writing either artifact. create_dir is
    // also the race-safe check for an output created after run's early check.
    fs::create_dir(output).with_context(|| {
        format!(
            "create NEW output directory {}; existing output is never overwritten",
            output.display()
        )
    })?;
    publish_new_file(output, "model-manifest.json", manifest)?;
    publish_new_file(output, "fp16-recipe.json", recipe)?;
    if let Some((source, identity_json)) = identity {
        publish_new_bytes(output, "source-recipe.json", source, false)?;
        publish_new_file(output, "recipe-identity.json", identity_json)?;
    }
    Ok(())
}

fn publish_new_file(output: &Path, name: &str, bytes: &[u8]) -> Result<()> {
    publish_new_bytes(output, name, bytes, true)
}

fn publish_new_bytes(output: &Path, name: &str, bytes: &[u8], append_lf: bool) -> Result<()> {
    let temporary = output.join(format!(".{name}.tmp"));
    let target = output.join(name);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .with_context(|| format!("create temporary artifact {}", temporary.display()))?;
    let result = (|| -> Result<()> {
        file.write_all(bytes).context("write artifact")?;
        if append_lf {
            file.write_all(b"\n").context("finish artifact JSON")?;
        }
        file.sync_all().context("flush artifact")?;
        Ok(())
    })();
    drop(file);
    let result = result.and_then(|()| {
        // Unlike rename on Unix, hard_link publishes the completed file
        // atomically and refuses an existing destination on every platform.
        fs::hard_link(&temporary, &target).with_context(|| {
            format!(
                "publish new artifact {} without replacing existing files",
                target.display()
            )
        })
    });
    let cleanup = fs::remove_file(&temporary);
    result?;
    cleanup.with_context(|| format!("remove temporary artifact {}", temporary.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_succeeds_without_model_or_cuda() {
        assert_eq!(quant_inspect(&["--help".to_string()]), 0);
        assert!(InspectArgs::try_parse_from(["quant-inspect", "--model", "m.bin"]).is_err());
    }

    #[test]
    fn native_format_matches_loader_extensions() {
        for (name, binary, compressed) in [
            ("model.BIN", true, false),
            ("model.bin.GZ", true, true),
            ("model.TXT", false, false),
            ("model.txt.gz", false, true),
        ] {
            assert_eq!(
                model_format(Path::new(name)).unwrap(),
                ModelFormat::Native { binary, compressed }
            );
        }
        assert_eq!(
            model_format(Path::new("model.ONNX")).unwrap(),
            ModelFormat::Onnx
        );
        for name in ["model.onnx.gz", "model.gz", "model.pt"] {
            assert!(model_format(Path::new(name)).is_err());
        }
        assert!(parse_graph(Path::new("broken.onnx"), b"not a model").is_err());
        assert!(parse_graph(Path::new("broken.bin.gz"), b"not gzip").is_err());
    }

    #[test]
    fn artifacts_are_complete_and_existing_output_is_untouched() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("new-output");
        publish_artifacts(&output, b"{\"manifest\":1}", b"{\"recipe\":1}", None).unwrap();
        let manifest = output.join("model-manifest.json");
        let recipe = output.join("fp16-recipe.json");
        assert_eq!(fs::read(&manifest).unwrap(), b"{\"manifest\":1}\n");
        assert_eq!(fs::read(&recipe).unwrap(), b"{\"recipe\":1}\n");
        assert_eq!(fs::read_dir(&output).unwrap().count(), 2);
        assert!(publish_artifacts(&output, b"changed", b"changed", None).is_err());
        assert!(publish_new_file(&output, "model-manifest.json", b"changed").is_err());
        assert_eq!(fs::read(&manifest).unwrap(), b"{\"manifest\":1}\n");
        assert_eq!(fs::read(&recipe).unwrap(), b"{\"recipe\":1}\n");
        assert_eq!(fs::read_dir(&output).unwrap().count(), 2);
    }

    #[test]
    fn optional_identity_preserves_source_bytes_and_refuses_overwrite() {
        let parsed = InspectArgs::try_parse_from([
            "quant-inspect",
            "--model",
            "m.bin",
            "--output",
            "out",
            "--recipe",
            "r.json",
        ])
        .unwrap();
        assert_eq!(parsed.recipe.as_deref(), Some(Path::new("r.json")));
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("with-identity");
        // Whitespace, CRLF and the absence of a final LF belong to the source
        // file identity and must survive the optional export byte for byte.
        let source = b"{\r\n  \"projections\": []\r\n}";
        let identity = b"{\"canonical_envelope\":{\"schema\":\"example\"}}";
        publish_artifacts(&output, b"{}", b"{}", Some((source, identity))).unwrap();
        assert_eq!(fs::read(output.join("source-recipe.json")).unwrap(), source);
        assert_eq!(
            fs::read(output.join("recipe-identity.json")).unwrap(),
            [identity.as_slice(), b"\n"].concat()
        );
        assert_eq!(fs::read(output.join("fp16-recipe.json")).unwrap(), b"{}\n");
        assert_eq!(fs::read_dir(&output).unwrap().count(), 4);
        assert!(publish_new_bytes(&output, "source-recipe.json", b"changed", false).is_err());
        assert!(publish_artifacts(&output, b"{}", b"{}", Some((b"changed", b"{}"))).is_err());
        assert_eq!(fs::read(output.join("source-recipe.json")).unwrap(), source);
        assert_eq!(fs::read_dir(&output).unwrap().count(), 4);
    }
}
