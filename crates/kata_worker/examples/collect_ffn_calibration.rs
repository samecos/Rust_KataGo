//! Finite calibration collection for one native model and one precision recipe.
//! --mode preflight|collect --plan ABS_JSON --plan-sha256 SHA --output FRESH_ABS_DIR
//! Preflight is CPU-only. Collect requires --features cuda and an explicit plan.
//! It does not select recipes, certify accuracy/performance, or contact a Worker.
#[path = "collect_ffn_calibration/artifacts.rs"]
mod artifacts;
#[path = "collect_ffn_calibration/inputs.rs"]
mod inputs;
#[path = "collect_ffn_calibration/outputs.rs"]
mod outputs;
#[path = "collect_ffn_calibration/runtime.rs"]
mod runtime;
#[path = "collect_ffn_calibration/schedule.rs"]
mod schedule;

use anyhow::{Context, Result, ensure};
use artifacts::{Source, commit_json, write_json};
use inputs::Corpus;
use kata_worker::evaluator::output_postprocessing::NativeModelOutputContract;
use schedule::{ExpectedCall, Plan};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::PathBuf};

struct Options {
    mode: String,
    plan: PathBuf,
    plan_sha: String,
    output: PathBuf,
}
fn options() -> Result<Options> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 8,
        "usage: --mode preflight|collect --plan ABS_JSON --plan-sha256 SHA --output FRESH_ABS_DIR"
    );
    let mut pairs = BTreeMap::new();
    for p in args.chunks_exact(2) {
        let key = p[0].to_str().context("non-UTF8 option")?;
        ensure!(
            ["--mode", "--plan", "--plan-sha256", "--output"].contains(&key)
                && pairs.insert(key, p[1].clone()).is_none(),
            "unknown/duplicate option"
        );
    }
    let mode = pairs["--mode"]
        .to_str()
        .context("mode encoding")?
        .to_owned();
    ensure!(
        ["preflight", "collect"].contains(&mode.as_str()),
        "unknown mode"
    );
    Ok(Options {
        mode,
        plan: PathBuf::from(&pairs["--plan"]),
        plan_sha: pairs["--plan-sha256"]
            .to_str()
            .context("SHA encoding")?
            .to_owned(),
        output: PathBuf::from(&pairs["--output"]),
    })
}
fn compiled_sources() -> Value {
    let mut map = serde_json::Map::new();
    for (name, raw) in [
        (
            "collect_ffn_calibration.rs",
            include_bytes!("collect_ffn_calibration.rs").as_slice(),
        ),
        (
            "artifacts.rs",
            include_bytes!("collect_ffn_calibration/artifacts.rs").as_slice(),
        ),
        (
            "schedule.rs",
            include_bytes!("collect_ffn_calibration/schedule.rs").as_slice(),
        ),
        (
            "inputs.rs",
            include_bytes!("collect_ffn_calibration/inputs.rs").as_slice(),
        ),
        (
            "outputs.rs",
            include_bytes!("collect_ffn_calibration/outputs.rs").as_slice(),
        ),
        (
            "input_encoding.rs",
            include_bytes!("../src/input_encoding.rs").as_slice(),
        ),
        (
            "output_postprocessing.rs",
            include_bytes!("../src/output_postprocessing.rs").as_slice(),
        ),
        (
            "output_postprocess.rs",
            include_bytes!("../../kata_nn/src/output_postprocess.rs").as_slice(),
        ),
        (
            "worker_evaluator.rs",
            include_bytes!("../src/evaluator.rs").as_slice(),
        ),
        (
            "nn_eval.rs",
            include_bytes!("../../kata_nn/src/eval.rs").as_slice(),
        ),
    ] {
        map.insert(name.into(), json!(artifacts::hash(raw)));
    }
    #[cfg(feature = "cuda")]
    map.insert(
        "runtime.rs".into(),
        json!(artifacts::hash(include_bytes!(
            "collect_ffn_calibration/runtime.rs"
        ))),
    );
    Value::Object(map)
}

fn check_callback(call: &ExpectedCall, event: &Value) -> Result<()> {
    ensure!(
        event["phase"] == call.phase
            && event["iteration"] == call.iteration
            && event["physical_batch"] == call.physical_batch
            && event["tensor_sha256"] == call.tensor_sha256,
        "runtime call differs from frozen schedule: {event}"
    );
    Ok(())
}

fn body(options: &Options) -> Result<()> {
    ensure!(
        options.plan.is_absolute() && artifacts::valid_sha(&options.plan_sha),
        "absolute plan and lowercase SHA required"
    );
    let plan_source = Source::capture(&options.plan)?;
    ensure!(plan_source.sha256 == options.plan_sha, "plan SHA mismatch");
    let plan: Plan = serde_json::from_slice(&plan_source.read(4 * 1024 * 1024)?)?;
    let exe = Source::capture(&std::env::current_exe()?)?;
    let recipe_bytes = plan.recipe.read(4 * 1024 * 1024)?;
    plan.verify_proposal(&recipe_bytes)?;
    plan.provenance.read(4 * 1024 * 1024)?;
    println!("Validating calibration provenance and exact PB/feature joins");
    let corpus = Corpus::load(
        &plan.provenance.path,
        &plan.provenance.sha256,
        &plan.model.sha256,
    )?;
    let calls = plan.validate(
        corpus.inputs(),
        corpus.coverage_mode(),
        corpus.selected_count(),
    )?;
    let model_bytes = plan.model.read(2 * 1024 * 1024 * 1024)?;
    {
        let desc = kata_nn::model_parser::load_model_from_bytes(
            &model_bytes,
            plan.binary,
            plan.compressed,
        )
        .map_err(anyhow::Error::msg)?;
        let graph = kata_nn::native_model::lower_model(&desc).map_err(anyhow::Error::msg)?;
        let source: kata_nn::quantization_plan::PrecisionRecipe =
            serde_json::from_slice(&recipe_bytes)?;
        let recipe =
            kata_nn::quantization_plan::resolve_recipe(&source, &graph, &plan.model.sha256)
                .map_err(anyhow::Error::msg)?;
        ensure!(
            desc.sha256 == plan.model.sha256
                && recipe.graph_sha256 == plan.graph_sha256
                && recipe.recipe_sha256 == plan.resolved_recipe_sha256,
            "real model/graph/resolved recipe differs"
        );
    }
    let contract = NativeModelOutputContract::from_native_model_bytes(
        &model_bytes,
        &plan.model.sha256,
        plan.binary,
        plan.compressed,
    )
    .map_err(|e| anyhow::anyhow!("{}: {}", e.code, e.message))?;
    ensure!(
        contract.graph_sha256() == plan.graph_sha256,
        "shared output graph differs"
    );
    let sources = [
        plan_source,
        exe,
        plan.model.clone(),
        plan.recipe.clone(),
        plan.proposal.clone(),
        plan.provenance.clone(),
    ];
    let intent = write_json(
        &options.output,
        "intent.json",
        &json!({"schema":"rustgo-ffn-calibration-attempt-v1",
        "mode":options.mode,"pid":std::process::id(),"plan":plan,"sources":sources,"calls":calls,
        "compiled_sources":compiled_sources(),"output_contract_sha256":contract.contract_sha256(),
        "planned_uploads":if options.mode=="collect"{1}else{0},"retry_allowed":false,
        "scope":"calibration numeric and fixed-sample group cost; not selection, holdout, ABBA or certification"}),
    )?;
    if options.mode == "preflight" {
        let indices: std::collections::BTreeSet<_> = calls.iter().map(|c| c.input_index).collect();
        let mut loaded_rows = 0usize;
        for index in indices {
            let input = corpus.load_input(index)?;
            for row in input.rows {
                contract
                    .prepare_request_pb(
                        &row.request_pb,
                        &row.descriptor.pb_sha256,
                        &row.descriptor.row_feature_sha256,
                    )
                    .map_err(|e| anyhow::anyhow!("{}: {}", e.code, e.message))?;
                loaded_rows += 1;
            }
        }
        corpus.recheck()?;
        for source in sources.iter().chain([&intent]) {
            source.recheck()?;
        }
        commit_json(
            &options.output,
            "result.json",
            &json!({"status":"PASS_CPU_PREFLIGHT_ONLY","gpu_forwards":0,
            "gpu_model_uploads":0,"planned_forwards":calls.len(),"validated_selected_rows":loaded_rows,
            "output_contract_sha256":contract.contract_sha256(),"sources":sources,"intent":intent,"sources_unchanged":true}),
        )?;
        return Ok(());
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (recipe_bytes, model_bytes);
        anyhow::bail!("collect requires the cuda feature");
    }
    #[cfg(feature = "cuda")]
    {
        let mut journal =
            artifacts::Journal::new(&options.output.join("forward-attempts.jsonl"), calls.len())?;
        // Load attempt is persistent before the one actual upload. No retry on failure.
        let load_attempt = write_json(
            &options.output,
            "model-load-attempt.json",
            &json!({"model":plan.model,"recipe":plan.recipe,"ordinal":0}),
        )?;
        let (mut runtime, metadata) = runtime::Runtime::load(runtime::ModelSpec {
            model_bytes: &model_bytes,
            binary: plan.binary,
            compressed: plan.compressed,
            model_sha256: &plan.model.sha256,
            graph_sha256: &plan.graph_sha256,
            recipe_bytes: &recipe_bytes,
            source_recipe_sha256: &plan.recipe.sha256,
            resolved_recipe_sha256: &plan.resolved_recipe_sha256,
        })
        .map_err(anyhow::Error::msg)?;
        let runtime_source = write_json(&options.output, "runtime.json", &metadata)?;
        let mut next = 0usize;
        let mut consume = |event: &Value| -> runtime::Result<()> {
            let call = calls.get(next).ok_or("unexpected extra forward")?;
            check_callback(call, event).map_err(|e| e.to_string())?;
            journal
                .consume(&json!({"expected":call,"runtime":event}))
                .map_err(|e| e.to_string())?;
            next += 1;
            Ok(())
        };
        let mut chunks = outputs::Chunks::new(&options.output, plan.chunk_inputs)?;
        for &index in &plan.numeric_input_indices {
            let input = corpus.load_input(index)?;
            let heads = runtime
                .numeric(
                    &input.spatial,
                    &input.global,
                    input.descriptor.physical_batch,
                    &mut consume,
                )
                .map_err(anyhow::Error::msg)?;
            chunks.append(index, &input, heads.as_raw(), &contract)?;
        }
        chunks.flush()?;
        let mut cost_sources = Vec::new();
        for cost in &plan.cost {
            let first = corpus.load_input(cost.input_indices[0])?;
            let warm = runtime
                .warm_cost(
                    &first.spatial,
                    &first.global,
                    cost.physical_batch,
                    cost.warmup,
                    &mut consume,
                )
                .map_err(anyhow::Error::msg)?;
            cost_sources.push(write_json(
                &options.output,
                &format!("cost-b{}-warm.json", cost.physical_batch),
                &json!({"input":first.descriptor,"runtime":warm}),
            )?);
            for &index in &cost.input_indices {
                let input = corpus.load_input(index)?;
                let (heads, measurement) = runtime
                    .measure_cost(
                        &input.spatial,
                        &input.global,
                        cost.physical_batch,
                        cost.measurements,
                        &mut consume,
                    )
                    .map_err(anyhow::Error::msg)?;
                outputs::validate_raw(heads.as_raw(), cost.physical_batch)?;
                cost_sources.push(write_json(&options.output,&format!("cost-b{}-input{index:04}.json",cost.physical_batch),&json!({
                    "input_index":index,"input":input.descriptor,"tensor_sha256":input.tensor_sha256,"runtime":measurement}))?);
            }
        }
        drop(consume);
        ensure!(
            next == calls.len()
                && journal.consumed == calls.len()
                && chunks.rows == plan.expected_numeric_rows
                && chunks.inputs == plan.numeric_input_indices.len(),
            "incomplete collection counts"
        );
        runtime.recheck().map_err(anyhow::Error::msg)?;
        corpus.recheck()?;
        chunks.recheck()?;
        for source in
            sources
                .iter()
                .chain(&cost_sources)
                .chain([&intent, &load_attempt, &runtime_source])
        {
            source.recheck()?;
        }
        let journal_source = journal.seal()?;
        commit_json(
            &options.output,
            "result.json",
            &json!({"status":"CAPTURE_COMPLETE_METRICS_PENDING",
            "model_sha256":plan.model.sha256,"graph_sha256":plan.graph_sha256,"recipe_sha256":plan.resolved_recipe_sha256,
            "output_contract_sha256":contract.contract_sha256(),"numeric_inputs":chunks.inputs,"numeric_rows":chunks.rows,
            "consumed_forwards":next,"gpu_model_uploads":1,"chunks":chunks.committed,"cost":cost_sources,
            "sources":sources,"intent":intent,"load_attempt":load_attempt,"runtime":runtime_source,
            "journal":journal_source,"sources_unchanged":true,
            "scope":"raw collection only; no accuracy or performance acceptance"}),
        )?;
    }
    Ok(())
}
fn run(options: Options) -> Result<()> {
    ensure!(
        options.output.is_absolute() && !options.output.exists(),
        "fresh absolute output directory required"
    );
    fs::create_dir(&options.output)?;
    let result = body(&options);
    if let Err(error) = &result {
        let _ = write_json(
            &options.output,
            "FAILED.json",
            &json!({"status":"FAIL_STOPPED_NO_RETRY","error":format!("{error:#}"),
            "mode":options.mode,"consumption":"see model-load-attempt.json and forward-attempts.jsonl; any uncertain attempt remains consumed"}),
        );
    }
    result
}
fn main() -> Result<()> {
    let options = options()?;
    std::thread::Builder::new()
        .name("ffn-calibration-collector".into())
        .stack_size(256 * 1024 * 1024)
        .spawn(move || run(options))?
        .join()
        .map_err(|_| anyhow::anyhow!("collector thread panicked; no retry"))?
}
