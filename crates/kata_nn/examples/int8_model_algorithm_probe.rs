//! Finite actual-model INT8 algorithm record/restore/Graph diagnostic.
//! Five raw heads compare within the SAME loaded quantized model/recipe only.
//! No FP16/FP32 accuracy, performance, per-operation trace, or deployment claim.
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
};
type Result<T> = std::result::Result<T, String>;
const PURPOSE: &str = "ACTUAL_MODEL_PARTIAL_INT8_CACHE_DIAGNOSTIC_NOT_ACCURACY_OR_PERFORMANCE";
const B11: &str = "1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6";
const B15: &str = "3f216ee88226ee49ca826eaa5f7e1e9982ff48a96625914bfc4e5ad20ee095a7";
const SMOKE: &str = "8be0a3c91be5f8fd9e60be9e50d55fa755b521b8e9dc4584062bab417d5e6503";
const HEADS: [(&str, usize); 5] = [
    ("policy", 6 * 362),
    ("value", 3),
    ("misc", 10),
    ("moremisc", 8),
    ("ownership", 361),
];
fn expected_environment() -> BTreeMap<String, String> {
    [
        ("KATAGO_CUDA_SPLITK", "0"),
        ("KATAGO_CUDA_INT8_GEMM_TUNE", "0"),
        ("KATAGO_CUDA_CUBLASLT_RANK", "heuristic"),
        ("KATAGO_CUDA_FFN_COMPACT_R1", "0"),
        ("KATAGO_CUDA_CUBLASLT", "1"),
        ("KATAGO_CUDA_DUALFFN", "0"),
        ("KATAGO_CUDA_FUSION", "none"),
        ("KATAGO_CUDA_GEMM_LAYOUT", "tn"),
        ("KATAGO_CUDA_INT8_RMS_FUSION", "1"),
    ]
    .into_iter()
    .map(|(k, v)| (k.into(), v.into()))
    .collect()
}
fn require(ok: bool, why: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(why.into()) }
}
fn hash(raw: &[u8]) -> String {
    hex::encode(Sha256::digest(raw))
}
fn save(path: &Path, raw: &[u8]) -> Result<String> {
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    file.write_all(raw)
        .and_then(|_| file.sync_all())
        .map_err(|e| e.to_string())?;
    Ok(hash(raw))
}
fn save_json(path: &Path, value: &impl Serialize) -> Result<String> {
    save(
        path,
        &serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
}
fn commit_result(path: &Path, value: &impl Serialize) -> Result<()> {
    let raw = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    let pending = path.with_extension("json.pending");
    save(&pending, &raw)?;
    fs::hard_link(&pending, path).map_err(|e| e.to_string())
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    logical_name: String,
    path: PathBuf,
    bytes: u64,
    sha256: String,
}
impl Source {
    fn read(&self) -> Result<Vec<u8>> {
        let raw = fs::read(&self.path).map_err(|e| e.to_string())?;
        require(
            raw.len() as u64 == self.bytes && hash(&raw) == self.sha256,
            &format!("bound file changed: {}", self.logical_name),
        )?;
        Ok(raw)
    }
    fn verify(&self) -> Result<()> {
        self.read().map(|_| ())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    family: String,
    model: Source,
    recipe: Source,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Smoke {
    batch: usize,
    spatial: Source,
    global: Source,
}
fn require_frozen_smoke(input: &Smoke) -> Result<()> {
    let (spatial, global) = match input.batch {
        1 => (
            "137f93ed0b5fe407fe20c5805429dddb478efb7478d4b85c42eabd50a4545ef1",
            "80643b217bb1565fc103796973ccd282b942a9863550ceeb1cec0687cc31762a",
        ),
        3 => (
            "43ff703d03e877d4f33664f536ef852ed4f9bf3c48221e3da3946ec6b7703a7b",
            "201d1b1d1e7e7cf5ead24f19a14cd0a2d4669d0c58e5cb9728a8003392e712c1",
        ),
        _ => return Err("unregistered smoke batch".into()),
    };
    require(
        input.spatial.sha256 == spatial
            && input.global.sha256 == global
            && input.spatial.bytes == (input.batch * 22 * 361 * 4) as u64
            && input.global.bytes == (input.batch * 19 * 4) as u64,
        "not the frozen B1/B3 tensor bytes",
    )
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Provenance {
    schema: String,
    purpose: String,
    sources: Vec<Source>,
    header: Source,
    static_cudart: Source,
    smoke_manifest: Source,
    inputs: Vec<Smoke>,
    cases: Vec<Case>,
    environment: BTreeMap<String, String>,
}
impl Provenance {
    fn verify(&self) -> Result<()> {
        require(
            self.schema == "rustgo-model-int8-probe-provenance-v1" && self.purpose == PURPOSE,
            "wrong provenance scope",
        )?;
        require(
            (2..=4).contains(&self.cases.len()),
            "requires 2..4 explicitly bound cases",
        )?;
        require(
            !self.sources.is_empty() && self.sources.len() <= 64,
            "invalid source inventory",
        )?;
        let mut previous = "";
        for source in &self.sources {
            require(
                source.logical_name.as_str() > previous
                    && !source.logical_name.contains(['\\', ':'])
                    && !source.logical_name.starts_with('/')
                    && !source
                        .logical_name
                        .split('/')
                        .any(|p| p.is_empty() || p == ".."),
                "source names not unique/sorted/relative",
            )?;
            source.verify()?;
            previous = &source.logical_name;
        }
        for required in [
            "crates/kata_nn/examples/int8_model_algorithm_probe.rs",
            "crates/kata_nn/examples/int8_model_algorithm_probe/native_support.rs",
            "crates/kata_nn/src/backends/int8_model_plan.rs",
            "Cargo.lock",
        ] {
            require(
                self.sources.iter().any(|x| x.logical_name == required),
                "missing actual build source",
            )?;
        }
        self.header.verify()?;
        self.static_cudart.verify()?;
        require(
            self.smoke_manifest.sha256 == SMOKE,
            "not the frozen GroupCost smoke manifest",
        )?;
        self.smoke_manifest.verify()?;
        require(
            self.inputs.iter().map(|x| x.batch).collect::<Vec<_>>() == vec![1, 3],
            "requires exactly frozen B1/B3",
        )?;
        for s in &self.inputs {
            require_frozen_smoke(s)?;
            s.spatial.verify()?;
            s.global.verify()?;
        }
        let mut identities = BTreeSet::new();
        let mut families = BTreeSet::new();
        for case in &self.cases {
            require(
                !case.id.is_empty()
                    && case.id.len() <= 64
                    && case
                        .id
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'-'),
                "invalid case ID",
            )?;
            require(
                identities.insert((case.model.sha256.as_str(), case.recipe.sha256.as_str())),
                "duplicate model/recipe attempt",
            )?;
            require(
                match case.family.as_str() {
                    "b11" => case.model.sha256 == B11,
                    "b15" => case.model.sha256 == B15,
                    _ => false,
                },
                "family does not match actual frozen model bytes",
            )?;
            families.insert(case.family.as_str());
            case.model.verify()?;
            case.recipe.verify()?;
        }
        require(
            families == BTreeSet::from(["b11", "b15"]),
            "both actual B11 and B15 required",
        )?;
        require(
            self.cases
                .iter()
                .map(|c| c.id.as_str())
                .collect::<BTreeSet<_>>()
                .len()
                == self.cases.len(),
            "duplicate case IDs",
        )?;
        require(
            self.environment == expected_environment(),
            "environment differs from finite reviewed policy",
        )?;
        Ok(())
    }
}
fn decoded_f32(source: &Source, count: usize) -> Result<Vec<f32>> {
    let raw = source.read()?;
    require(raw.len() == count * 4, "wrong frozen tensor dimensions")?;
    let data: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    require(data.iter().all(|x| x.is_finite()), "nonfinite input tensor")?;
    Ok(data)
}
fn raw_bits(heads: [Vec<f32>; 5], batch: usize) -> Result<[Vec<u32>; 5]> {
    for (head, (_, width)) in heads.iter().zip(HEADS) {
        require(
            head.len() == batch * width && head.iter().all(|v| v.is_finite()),
            "unwritten/nonfinite or wrong-sized output head",
        )?;
    }
    Ok(heads.map(|head| head.into_iter().map(f32::to_bits).collect()))
}
fn save_heads(out: &Path, stage: &str, bits: &[Vec<u32>; 5]) -> Result<Value> {
    let mut files = Vec::new();
    for (data, (name, _)) in bits.iter().zip(HEADS) {
        let path = format!("{stage}-{name}.f32le");
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        files.push(json!({"head":name,"path":path,"bytes":raw.len(),"sha256":save(&out.join(&path),&raw)?}));
    }
    Ok(json!(files))
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
#[path = "int8_model_algorithm_probe/native_support.rs"]
mod native_support;

#[cfg(all(feature = "cuda", target_os = "windows"))]
mod gpu {
    use super::*;
    use cudarc::driver::{CudaSlice, CudaStream, sys as ds};
    use kata_nn::{
        backends::{
            cuda::{self, CudaRuntime},
            cuda_exec::{self, CudaModel, CudaOutputsHost, CudaWorkspace},
            int8, int8_algorithm_plan as plan,
        },
        onnx_parser::LayerGraph,
        quantization_plan::{self, Precision, PrecisionRecipe, ResolvedRecipe},
    };
    use std::sync::Arc;
    struct Parsed {
        case: Case,
        graph: LayerGraph,
        recipe: ResolvedRecipe,
        manifest: Value,
        recipe_identity: Vec<u8>,
        all_ffn_int8: bool,
    }
    #[derive(Default, Serialize)]
    struct Counts {
        host_apply_calls: usize,
        graph_replays: usize,
    }
    impl Counts {
        fn apply(&mut self) -> Result<()> {
            require(
                self.host_apply_calls < 32,
                "finite host-apply budget exceeded",
            )?;
            self.host_apply_calls += 1;
            Ok(())
        }
        fn replay(&mut self) -> Result<()> {
            require(
                self.graph_replays < 8,
                "finite graph-replay budget exceeded",
            )?;
            self.graph_replays += 1;
            Ok(())
        }
    }
    fn parse_cases(provenance: &Provenance) -> Result<Vec<Parsed>> {
        let mut result = Vec::new();
        let mut full = BTreeSet::new();
        let mut canonical = BTreeSet::new();
        for case in &provenance.cases {
            let model_raw = case.model.read()?;
            let desc = kata_nn::model_parser::load_model_from_bytes(&model_raw, true, true)
                .map_err(|e| e.to_string())?;
            let graph = kata_nn::native_model::lower_model(&desc)?;
            let source: PrecisionRecipe =
                serde_json::from_slice(&case.recipe.read()?).map_err(|e| e.to_string())?;
            let recipe = quantization_plan::resolve_recipe(&source, &graph, &case.model.sha256)?;
            require(
                canonical.insert((case.model.sha256.clone(), recipe.recipe_sha256.clone())),
                "duplicate resolved model/recipe attempt",
            )?;
            let identity = quantization_plan::resolve_recipe_identity(
                &case.recipe.read()?,
                &graph,
                &case.model.sha256,
            )?;
            require(
                identity.recipe_sha256 == recipe.recipe_sha256,
                "resolved recipe export differs",
            )?;
            let recipe_identity =
                serde_json::to_vec_pretty(&identity).map_err(|e| e.to_string())?;
            require(
                !recipe.layers.iter().any(|x| x.any_mxfp8()),
                "this diagnostic accepts only FP16/INT8 recipes",
            )?;
            let manifest = quantization_plan::graph_manifest(&graph, &case.model.sha256)?;
            let mut ffn = 0;
            let mut all = true;
            for p in &manifest.projections {
                let precision = &recipe.layers[p.layer_index];
                let actual = if p.id.ends_with(".ffn.dual") {
                    ffn += 1;
                    precision.ffn_dual
                } else if p.id.ends_with(".ffn.down") {
                    ffn += 1;
                    precision.ffn_down
                } else if p.id.ends_with(".attention.qkv") {
                    precision.attention_qkv
                } else if p.id.ends_with(".attention.out") {
                    precision.attention_out
                } else {
                    return Err("unsupported projection in model manifest".into());
                };
                all &= if p.id.contains(".ffn.") {
                    actual == Precision::Int8
                } else {
                    actual == Precision::Fp16
                };
            }
            require(
                ffn > 0 && recipe.layers.iter().any(|x| x.any_int8()),
                "recipe has no actual INT8 projections",
            )?;
            if all {
                full.insert(case.family.clone());
            }
            result.push(Parsed {
                case: case.clone(),
                graph,
                recipe,
                manifest: serde_json::to_value(manifest).map_err(|e| e.to_string())?,
                recipe_identity,
                all_ffn_int8: all,
            });
        }
        require(
            full == BTreeSet::from(["b11".into(), "b15".into()]),
            "each model needs an all-FFN INT8 / FP16 attention case",
        )?;
        Ok(result)
    }
    fn binding(
        rt: &CudaRuntime,
        provenance: &Provenance,
        parsed: &Parsed,
        out: &Path,
        policy: String,
    ) -> Result<(plan::Binding, Source, Source)> {
        let (lt, blas) = native_support::loaded_libraries()?;
        let libraries = save_json(
            &out.join("libraries.json"),
            &json!({"cublaslt":lt,"cublas":blas,
            "loaded_symbol_resolution":"actual culib symbol -> GetModuleHandleExW FROM_ADDRESS",
            "header":provenance.header,"static_cudart":provenance.static_cudart,
            "static_archive_scope":"build input, not a loaded DLL or proof of embedded object contents"}),
        )?;
        let build = save_json(
            &out.join("backend-build.json"),
            &cuda::backend_build_fingerprint(),
        )?;
        let source = save_json(&out.join("source-evidence.json"), &provenance.sources)?;
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        let exe_sha = hash(&fs::read(&exe).map_err(|e| e.to_string())?);
        let u = hex::encode(
            rt.device
                .uuid()
                .map_err(|e| e.to_string())?
                .bytes
                .map(|b| b as u8),
        );
        let mut driver = 0;
        unsafe {
            ds::cuDriverGetVersion(&mut driver)
                .result()
                .map_err(|e| e.to_string())?;
        }
        let (major, minor) = rt.device.compute_capability().map_err(|e| e.to_string())?;
        let fp = cuda::device_fingerprint(&rt.device)?;
        save_json(
            &out.join("device.json"),
            &json!({"gpu_name":fp.gpu_name,"compute_capability":fp.compute_capability,
            "sm_count":fp.sm_count,"l2_cache_bytes":fp.l2_cache_bytes,"driver_api":driver,"executable":exe,"executable_sha256":exe_sha}),
        )?;
        let binding = plan::Binding {
            model_sha256: parsed.case.model.sha256.clone(),
            graph_sha256: parsed.recipe.graph_sha256.clone(),
            recipe_sha256: parsed.recipe.recipe_sha256.clone(),
            execution_policy_sha256: policy,
            executable_sha256: exe_sha,
            host_source_sha256: source,
            kernel_build_sha256: build,
            device_fingerprint_sha256: int8::algorithm_device_fingerprint_sha256(rt)?,
            gpu_uuid: format!(
                "GPU-{}-{}-{}-{}-{}",
                &u[..8],
                &u[8..12],
                &u[12..16],
                &u[16..20],
                &u[20..]
            ),
            compute_capability: [major as u32, minor as u32],
            sm_count: fp.sm_count,
            driver_api_version: driver,
            cublaslt_version: unsafe { cudarc::cublaslt::sys::cublasLtGetVersion() } as u64,
            cublaslt_binary_sha256: lt.sha256.clone(),
            cublas_binary_sha256: blas.sha256.clone(),
            cuda_runtime_provenance_sha256: libraries,
            host_abi: format!("{}-{}-msvc", std::env::consts::ARCH, std::env::consts::OS),
            pointer_width_bits: usize::BITS,
            little_endian: cfg!(target_endian = "little"),
            algo_size_bytes: std::mem::size_of::<cudarc::cublaslt::sys::cublasLtMatmulAlgo_t>()
                as u32,
            algo_alignment_bytes: std::mem::align_of::<cudarc::cublaslt::sys::cublasLtMatmulAlgo_t>(
            ) as u32,
            sources: provenance
                .sources
                .iter()
                .map(|s| plan::SourceIdentity {
                    logical_name: s.logical_name.clone(),
                    sha256: s.sha256.clone(),
                })
                .collect(),
        };
        binding.validate()?;
        save_json(&out.join("binding.json"), &binding)?;
        Ok((binding, lt, blas))
    }
    fn fill_nan(stream: &Arc<CudaStream>, ws: &mut CudaWorkspace) -> Result<()> {
        let mut outputs = [
            &mut ws.out_policy,
            &mut ws.out_value,
            &mut ws.out_misc,
            &mut ws.out_moremisc,
            &mut ws.out_ownership,
        ];
        let host: Vec<Vec<f32>> = outputs
            .iter()
            .map(|x| vec![f32::from_bits(0x7fc00000); x.len()])
            .collect();
        for (dst, src) in outputs.iter_mut().zip(&host) {
            stream.memcpy_htod(src, *dst).map_err(|e| e.to_string())?;
        }
        stream.synchronize().map_err(|e| e.to_string())
    }
    fn read_heads(
        stream: &Arc<CudaStream>,
        ws: &CudaWorkspace,
        batch: usize,
    ) -> Result<[Vec<u32>; 5]> {
        let CudaOutputsHost {
            policy,
            value,
            misc,
            moremisc,
            ownership,
        } = ws.to_host(stream)?;
        stream.synchronize().map_err(|e| e.to_string())?;
        raw_bits([policy, value, misc, moremisc, ownership], batch)
    }
    fn complete<T>(
        model: &CudaModel,
        rt: &CudaRuntime,
        stream: &Arc<CudaStream>,
        ws: &mut CudaWorkspace,
        stage: &str,
        operation: Result<T>,
    ) -> Result<T> {
        let original = operation.as_ref().err().cloned();
        model
            .complete_int8_algorithm_operation(rt, stream, ws, operation)
            .map_err(|e| format!("{stage}: original={original:?}; completion={e}"))
    }
    fn forward(
        model: &CudaModel,
        rt: &CudaRuntime,
        stream: &Arc<CudaStream>,
        ws: &mut CudaWorkspace,
        inputs: &(CudaSlice<f32>, CudaSlice<f32>),
        batch: usize,
        stage: &str,
        counts: &mut Counts,
    ) -> Result<[Vec<u32>; 5]> {
        let operation = (|| {
            fill_nan(stream, ws)?;
            counts.apply()?;
            model.apply(rt, stream, ws, &inputs.0, &inputs.1)?;
            stream.synchronize().map_err(|e| e.to_string())?;
            read_heads(stream, ws, batch)
        })();
        complete(model, rt, stream, ws, stage, operation)
    }
    struct Capturing;
    impl Drop for Capturing {
        fn drop(&mut self) {
            cuda_exec::set_capturing(false);
        }
    }
    pub(super) fn run(out: &Path, provenance: &Provenance) -> Result<Value> {
        let environment: BTreeMap<String, String> = std::env::vars()
            .filter(|(k, _)| k.starts_with("KATAGO_CUDA_"))
            .collect();
        require(
            environment == provenance.environment,
            "runtime CUDA environment differs from frozen policy",
        )?;
        require(
            !std::env::vars()
                .any(|(k, _)| k.starts_with("KATAGO_") && !k.starts_with("KATAGO_CUDA_")),
            "unbound KATAGO environment",
        )?;
        let parsed = parse_cases(provenance)?; // CPU parse/identity/coverage before CUDA init.
        let mut host = Vec::new();
        for input in &provenance.inputs {
            let spatial = decoded_f32(&input.spatial, input.batch * 22 * 361)?;
            let global = decoded_f32(&input.global, input.batch * 19)?;
            require(
                spatial
                    .chunks_exact(22 * 361)
                    .all(|r| r[..361].iter().all(|&x| x == 1.0)),
                "invalid on-board plane",
            )?;
            host.push((input.batch, spatial, global));
        }
        let rt = CudaRuntime::new()?;
        cuda::validate_quantized_runtime(&rt)?;
        let mut counts = Counts::default();
        let mut results = Vec::new();
        for parsed in &parsed {
            let case_out = out.join(&parsed.case.id);
            fs::create_dir(&case_out).map_err(|e| e.to_string())?;
            save_json(&case_out.join("graph-manifest.json"), &parsed.manifest)?;
            save(
                &case_out.join("source-recipe.json"),
                &parsed.case.recipe.read()?,
            )?;
            save(
                &case_out.join("recipe-identity.json"),
                &parsed.recipe_identity,
            )?;
            let policy = save_json(
                &case_out.join("execution-policy.json"),
                &json!({"purpose":PURPOSE,"environment":environment,
                "inputs":provenance.inputs,"batches":[1,3],"ordinary_baseline_then_fresh_record_then_other_stream_restore":true,
                "raw_output_prefill":"f32 NaN 0x7fc00000 at every stage","graph_replays_per_batch":1,"warm_restored_forwards":1,
                "maximum_host_apply_calls":32,"maximum_graph_replays":8,"runtime_tuning":false}),
            )?;
            let (binding, lt, blas) = binding(&rt, provenance, parsed, &case_out, policy)?;
            let streams = [
                rt.device.new_stream().map_err(|e| e.to_string())?,
                rt.device.new_stream().map_err(|e| e.to_string())?,
            ];
            require(
                !Arc::ptr_eq(&streams[0], &streams[1])
                    && streams[0].cu_stream() != streams[1].cu_stream(),
                "restore requires a distinct live stream",
            )?;
            let model = CudaModel::load_quantized(&parsed.graph, &rt, &streams[0], &parsed.recipe)?;
            cuda::ensure_requested_cuda_capabilities(&model)?;
            rt.device.synchronize().map_err(|e| e.to_string())?; // model weights ready on the other stream
            for (batch, spatial, global) in &host {
                let batch_out = case_out.join(format!("b{batch}"));
                fs::create_dir(&batch_out).map_err(|e| e.to_string())?;
                let s0 = &streams[0];
                let s1 = &streams[1];
                let input0 = (
                    s0.clone_htod(spatial).map_err(|e| e.to_string())?,
                    s0.clone_htod(global).map_err(|e| e.to_string())?,
                );
                let input1 = (
                    s1.clone_htod(spatial).map_err(|e| e.to_string())?,
                    s1.clone_htod(global).map_err(|e| e.to_string())?,
                );
                let inventory =
                    model.int8_algorithm_inventory(&parsed.graph, &parsed.recipe, *batch)?;
                save_json(
                    &batch_out.join("stable-operation-directory.json"),
                    &inventory.projections,
                )?;
                save_json(
                    &batch_out.join("deduplicated-shape-directory.json"),
                    &inventory.shapes,
                )?;
                save_json(&batch_out.join("inventory.json"), &inventory)?;
                let mut ordinary = CudaWorkspace::new(s0, &model, *batch)?;
                let expected = forward(
                    &model,
                    &rt,
                    s0,
                    &mut ordinary,
                    &input0,
                    *batch,
                    "ordinary",
                    &mut counts,
                )?;
                let ordinary_heads = save_heads(&batch_out, "ordinary", &expected)?;
                let mut recording = CudaWorkspace::new(s0, &model, *batch)?;
                require(
                    model.begin_int8_algorithm_recording(
                        &rt,
                        s0,
                        &mut recording,
                        &parsed.graph,
                        &parsed.recipe,
                        binding.clone(),
                    )? == inventory,
                    "record inventory differs",
                )?;
                let recorded = forward(
                    &model,
                    &rt,
                    s0,
                    &mut recording,
                    &input0,
                    *batch,
                    "record",
                    &mut counts,
                )?;
                let recorded_heads = save_heads(&batch_out, "record", &recorded)?;
                require(
                    recorded == expected,
                    "record forward differs bitwise from ordinary workspace",
                )?;
                let plan = model.export_int8_algorithm_plan(&rt, s0, &mut recording)?;
                let plan_sha = save(&batch_out.join("record.plan.json"), &plan)?;
                let mut restored = CudaWorkspace::new(s1, &model, *batch)?;
                require(
                    model.install_int8_algorithm_plan(
                        &rt,
                        s1,
                        &mut restored,
                        &parsed.graph,
                        &parsed.recipe,
                        &binding,
                        &plan,
                        &plan_sha,
                    )? == inventory,
                    "restore inventory differs",
                )?;
                let actual = forward(
                    &model,
                    &rt,
                    s1,
                    &mut restored,
                    &input1,
                    *batch,
                    "restore-warm",
                    &mut counts,
                )?;
                let restored_heads = save_heads(&batch_out, "restore", &actual)?;
                require(
                    actual == expected,
                    "restore forward differs bitwise from ordinary workspace",
                )?;
                let restored_plan = model.export_int8_algorithm_plan(&rt, s1, &mut restored)?;
                save(&batch_out.join("restored.plan.json"), &restored_plan)?;
                require(
                    restored_plan == plan,
                    "restored export is not byte-identical",
                )?;
                // Warmed all authoritative logical shapes and the other model paths.
                let prefill = fill_nan(s1, &mut restored);
                complete(&model, &rt, s1, &mut restored, "graph-prefill", prefill)?;
                let capture = native_support::capture(s1, || {
                    require(!cuda_exec::capturing(), "nested model capture")?;
                    cuda_exec::set_capturing(true);
                    let _guard = Capturing;
                    counts.apply()?;
                    model.apply(&rt, s1, &mut restored, &input1.0, &input1.1)
                });
                let graph = complete(&model, &rt, s1, &mut restored, "graph-capture", capture)?;
                let replay = (|| {
                    counts.replay()?;
                    graph.launch()?;
                    s1.synchronize().map_err(|e| e.to_string())?;
                    read_heads(s1, &restored, *batch)
                })();
                let replayed = complete(
                    &model,
                    &rt,
                    s1,
                    &mut restored,
                    "graph-launch-sync-copy",
                    replay,
                )?;
                let graph_heads = save_heads(&batch_out, "graph", &replayed)?;
                require(
                    replayed == expected,
                    "Graph replay differs bitwise from ordinary workspace",
                )?;
                let graph_plan = model.export_int8_algorithm_plan(&rt, s1, &mut restored)?;
                save(&batch_out.join("post-graph.plan.json"), &graph_plan)?;
                require(graph_plan == plan, "post-Graph export changed")?;
                drop(graph); // completed replay, graph destroyed before workspace/model
                results.push(json!({"case":parsed.case.id,"family":parsed.case.family,"batch":batch,
                    "all_ffn_int8_case":parsed.all_ffn_int8,"model_sha256":parsed.case.model.sha256,
                    "graph_sha256":parsed.recipe.graph_sha256,"recipe_sha256":parsed.recipe.recipe_sha256,
                    "stable_operation_count":inventory.projections.len(),"deduplicated_shape_count":inventory.shapes.len(),
                    "directory_scope":"static loaded-projection directory plus shape-cache observations; not per-operation dynamic tracking",
                    "plan_sha256":plan_sha,"record_restore_post_graph_plan_bytes_equal":true,"graph_replays":1,
                    "raw_five_heads_bitwise_equal":true,"ordinary_heads":ordinary_heads,"record_heads":recorded_heads,
                    "restore_heads":restored_heads,"graph_heads":graph_heads,"fresh_workspaces":3,"restore_on_distinct_stream":true}));
            }
            lt.verify()?;
            blas.verify()?;
        }
        require(
            counts.host_apply_calls == provenance.cases.len() * 8
                && counts.graph_replays == provenance.cases.len() * 2,
            "finite operation inventory incomplete",
        )?;
        provenance.verify()?;
        Ok(
            json!({"status":"PASS","purpose":PURPOSE,"checks":results,"counts":counts,
            "scope":"full-forward intra-recipe bit equality; partial INT8 LT algorithm cache only",
            "per_operation_execution_trace":false,"fp16_or_fp32_accuracy_certified":false,"performance_adoption":false}),
        )
    }
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    require(
        args.len() == 6
            && args[0] == "--provenance"
            && args[2] == "--provenance-sha256"
            && args[4] == "--output",
        "usage: --provenance FILE --provenance-sha256 SHA --output NEW_DIR",
    )?;
    let raw = fs::read(&args[1]).map_err(|e| e.to_string())?;
    require(hash(&raw) == args[3], "provenance external SHA mismatch")?;
    let provenance: Provenance = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    provenance.verify()?;
    let out = PathBuf::from(&args[5]);
    fs::create_dir(&out).map_err(|e| e.to_string())?;
    save_json(
        &out.join("intent.json"),
        &json!({"purpose":PURPOSE,"provenance_sha256":args[3],"cases":provenance.cases,
        "input_source":"frozen synthetic GroupCost B1/B3","retries":0,"holdout_outputs_read":0,"performance_measurement":false}),
    )?;
    #[cfg(all(feature = "cuda", target_os = "windows"))]
    let result = gpu::run(&out, &provenance);
    #[cfg(not(all(feature = "cuda", target_os = "windows")))]
    let result: Result<Value> = Err("requires Windows and --features cuda".into());
    match result {
        Ok(report) => {
            commit_result(&out.join("result.json"), &report)?;
            Ok(())
        }
        Err(error) => {
            commit_result(
                &out.join("result.json"),
                &json!({"status":"FAIL","purpose":PURPOSE,"error":error}),
            )?;
            Err(error)
        }
    }
}

#[cfg(test)]
#[path = "int8_model_algorithm_probe/tests.rs"]
mod tests;
