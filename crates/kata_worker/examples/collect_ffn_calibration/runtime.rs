//! Single-upload direct CUDA adapter for the explicit calibration collector.
//! No scheduling, file publication, implicit warmup, retry, Graph or autotuning.
//! The caller must durably reserve each callback before returning Ok(()).

use kata_nn::output_postprocess::RawHeads;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub type Result<T> = std::result::Result<T, String>;
pub const HEADS: [(&str, usize); 5] = [
    ("policy", 6 * 362),
    ("value", 3),
    ("misc", 10),
    ("moremisc", 8),
    ("ownership", 361),
];

/// Expected identities come from the caller's immutable plan, never filenames.
pub struct ModelSpec<'a> {
    pub model_bytes: &'a [u8],
    pub binary: bool,
    pub compressed: bool,
    pub model_sha256: &'a str,
    pub graph_sha256: &'a str,
    pub recipe_bytes: &'a [u8],
    pub source_recipe_sha256: &'a str,
    pub resolved_recipe_sha256: &'a str,
}

#[derive(Debug)]
pub struct Heads {
    pub policy: Vec<f32>,
    pub value: Vec<f32>,
    pub misc: Vec<f32>,
    pub moremisc: Vec<f32>,
    pub ownership: Vec<f32>,
}
impl Heads {
    pub fn as_raw(&self) -> RawHeads<'_> {
        RawHeads {
            policy: &self.policy,
            value: &self.value,
            misc: &self.misc,
            moremisc: &self.moremisc,
            ownership: &self.ownership,
        }
    }
    pub fn as_slices(&self) -> [&[f32]; 5] {
        [
            &self.policy,
            &self.value,
            &self.misc,
            &self.moremisc,
            &self.ownership,
        ]
    }
    pub fn validate(&self, batch: usize) -> Result<()> {
        batch_lengths(batch)?;
        for ((name, width), values) in HEADS.iter().zip(self.as_slices()) {
            require(
                values.len() == batch * width && values.iter().all(|x| x.is_finite()),
                &format!("wrong shape, unwritten or nonfinite head: {name}"),
            )?;
        }
        Ok(())
    }
    pub fn bitwise_eq(&self, other: &Self) -> bool {
        self.as_slices()
            .into_iter()
            .zip(other.as_slices())
            .all(|(a, b)| {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
            })
    }
    pub fn hashes(&self) -> Value {
        Value::Array(HEADS.iter().zip(self.as_slices()).map(|((name, _), values)| {
            let mut digest = Sha256::new();
            for value in values { digest.update(value.to_bits().to_le_bytes()); }
            json!({"head":name,"bytes":values.len()*4,"sha256":hex::encode(digest.finalize())})
        }).collect())
    }
}

fn require(ok: bool, why: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(why.to_owned()) }
}
fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn valid_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|x| x.is_ascii_digit() || (b'a'..=b'f').contains(&x))
}
fn batch_lengths(batch: usize) -> Result<(usize, usize)> {
    require(
        (1..=64).contains(&batch),
        "physical batch must be in 1..=64",
    )?;
    Ok((
        batch.checked_mul(22 * 361).ok_or("spatial size overflow")?,
        batch.checked_mul(19).ok_or("global size overflow")?,
    ))
}
/// Same tensor domain as the actual GroupCost input reader; labels are host
/// provenance, not a claim that a device tensor has been independently hashed.
pub fn input_label(spatial: &[f32], global: &[f32], batch: usize) -> Result<String> {
    let (s, g) = batch_lengths(batch)?;
    require(
        spatial.len() == s && global.len() == g,
        "input lengths differ from exact physical batch",
    )?;
    require(
        spatial.iter().chain(global).all(|x| x.is_finite()),
        "input contains NaN/Inf",
    )?;
    require(
        spatial
            .chunks_exact(22 * 361)
            .all(|r| r[..361].iter().all(|x| x.to_bits() == 1.0f32.to_bits())),
        "requires complete 19x19 on-board plane",
    )?;
    let mut digest = Sha256::new();
    digest.update(b"rustgo-encoded-group-cost-tensors-v1\0");
    digest.update((batch as u64).to_le_bytes());
    for value in spatial.iter().chain(global) {
        digest.update(value.to_bits().to_le_bytes());
    }
    Ok(hex::encode(digest.finalize()))
}

/// Chosen collector policy, not a statement that GroupCost requires DUALFFN=0.
pub fn expected_environment() -> BTreeMap<String, String> {
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
        ("KATAGO_CUDA_NOGRAPH", "1"),
        ("KATAGO_CUDA_NOPIPELINE", "1"),
        ("KATAGO_CUDA_PADBATCH", "0"),
        ("KATAGO_CUDA_BATCH_TRACE", "1"),
    ]
    .into_iter()
    .map(|(k, v)| (k.into(), v.into()))
    .collect()
}
pub fn check_environment() -> Result<BTreeMap<String, String>> {
    let expected = expected_environment();
    let mut actual = BTreeMap::new();
    for (key, value) in std::env::vars_os() {
        // Windows environment names are case-insensitive. Reject lower/mixed
        // case aliases as unbound entries instead of accidentally ignoring them.
        if key
            .to_string_lossy()
            .to_ascii_uppercase()
            .starts_with("KATAGO_CUDA_")
        {
            actual.insert(
                key.into_string()
                    .map_err(|_| "non-Unicode CUDA environment key")?,
                value
                    .into_string()
                    .map_err(|_| "non-Unicode CUDA environment value")?,
            );
        }
    }
    require(
        actual == expected,
        "CUDA environment must equal the collector's explicit 13-key policy",
    )?;
    require(
        kata_nn::tactic_plan::installed_plan_id().is_none(),
        "legacy tactic plan is forbidden",
    )?;
    for (key, expected_value) in &expected {
        require(
            kata_nn::tactic_plan::tactic_var(key).ok().as_ref() == Some(expected_value),
            &format!("effective tactic differs from collector policy: {key}"),
        )?;
    }
    Ok(expected)
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use cudarc::driver::{CudaSlice, CudaStream};
    use kata_nn::{
        backends::{
            cuda::{self, CudaRuntime},
            cuda_exec::{CudaModel, CudaWorkspace},
            group_cost::{GroupCostRecorder, Phase},
        },
        onnx_parser::LayerGraph,
        quantization_plan::{self, Precision, PrecisionRecipe, ResolvedRecipe},
    };
    use serde::Serialize;
    use std::{
        fs,
        io::Read,
        path::{Path, PathBuf},
        sync::Arc,
    };

    #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
    struct Source {
        path: PathBuf,
        bytes: u64,
        sha256: String,
    }
    fn source(path: &Path) -> Result<Source> {
        let path = path.canonicalize().map_err(|e| e.to_string())?;
        let mut file = fs::File::open(&path).map_err(|e| e.to_string())?;
        let before = file.metadata().map_err(|e| e.to_string())?;
        let mut digest = Sha256::new();
        let mut buffer = vec![0u8; 1024 * 1024];
        let mut bytes = 0u64;
        loop {
            let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            digest.update(&buffer[..n]);
            bytes += n as u64;
        }
        let after = file.metadata().map_err(|e| e.to_string())?;
        require(
            before.len() == bytes
                && after.len() == bytes
                && before.modified().map_err(|e| e.to_string())?
                    == after.modified().map_err(|e| e.to_string())?,
            "file changed while hashing source",
        )?;
        Ok(Source {
            path,
            bytes,
            sha256: hex::encode(digest.finalize()),
        })
    }

    // Adapted from the already exercised fixed-output probe: resolve the actual
    // cudarc library export address, not a guessed installation or Rust wrapper.
    #[cfg(windows)]
    fn loaded_libraries() -> Result<[Source; 2]> {
        use std::ffi::c_void;
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetModuleHandleExW(flags: u32, address: *const u16, module: *mut *mut c_void)
            -> i32;
            fn GetModuleFileNameW(module: *mut c_void, buffer: *mut u16, size: u32) -> u32;
        }
        fn module(address: *const c_void) -> Result<Source> {
            let mut handle = std::ptr::null_mut();
            require(
                unsafe { GetModuleHandleExW(6, address.cast(), &mut handle) } != 0,
                "GetModuleHandleExW failed",
            )?;
            let mut text = vec![0u16; 32768];
            let n = unsafe { GetModuleFileNameW(handle, text.as_mut_ptr(), text.len() as u32) }
                as usize;
            require(
                n > 0 && n < text.len(),
                "GetModuleFileNameW failed/truncated",
            )?;
            source(Path::new(
                &String::from_utf16(&text[..n]).map_err(|e| e.to_string())?,
            ))
        }
        unsafe {
            let lt = cudarc::cublaslt::sys::culib()
                .get::<unsafe extern "C" fn() -> usize>(b"cublasLtGetVersion\0")
                .map_err(|e| e.to_string())?;
            let blas = cudarc::cublas::sys::culib()
                .get::<unsafe extern "C" fn(
                    cudarc::cublas::sys::cublasHandle_t,
                    *mut i32,
                ) -> cudarc::cublas::sys::cublasStatus_t>(b"cublasGetVersion_v2\0")
                .map_err(|e| e.to_string())?;
            Ok([
                module(*lt as *const c_void)?,
                module(*blas as *const c_void)?,
            ])
        }
    }
    #[cfg(not(windows))]
    fn loaded_libraries() -> Result<[Source; 2]> {
        Err("this finite collector requires Windows actual DLL-source binding".into())
    }

    struct Parsed {
        graph: LayerGraph,
        recipe: ResolvedRecipe,
        groups: Vec<String>,
        metadata: Value,
    }
    fn parse(spec: &ModelSpec<'_>) -> Result<Parsed> {
        for value in [
            spec.model_sha256,
            spec.graph_sha256,
            spec.source_recipe_sha256,
            spec.resolved_recipe_sha256,
        ] {
            require(
                valid_sha(value),
                "expected identity is not canonical SHA256",
            )?;
        }
        require(
            hash(spec.model_bytes) == spec.model_sha256,
            "native model bytes SHA mismatch",
        )?;
        require(
            hash(spec.recipe_bytes) == spec.source_recipe_sha256,
            "recipe source bytes SHA mismatch",
        )?;
        let desc = kata_nn::model_parser::load_model_from_bytes(
            spec.model_bytes,
            spec.binary,
            spec.compressed,
        )
        .map_err(|e| e.to_string())?;
        let inputs_version = kata_nn::version::get_inputs_version(desc.model_version)?;
        require(
            desc.sha256 == spec.model_sha256
                && inputs_version == 7
                && desc.num_input_channels == 22
                && desc.num_input_global_channels == 19
                && desc.num_input_meta_channels == 0,
            "native model input/source contract differs from v7/22/19/meta0",
        )?;
        let graph = kata_nn::native_model::lower_model(&desc)?;
        require(
            graph.board_size == 19
                && graph.num_spatial_inputs == 22
                && graph.num_global_inputs == 19
                && quantization_plan::graph_sha256(&graph)? == spec.graph_sha256,
            "native lowered graph shape or SHA differs",
        )?;
        let original: PrecisionRecipe =
            serde_json::from_slice(spec.recipe_bytes).map_err(|e| e.to_string())?;
        let recipe = quantization_plan::resolve_recipe(&original, &graph, spec.model_sha256)?;
        require(
            recipe.recipe_sha256 == spec.resolved_recipe_sha256,
            "resolved recipe SHA mismatch",
        )?;
        require(
            recipe.layers.iter().all(|layer| {
                !layer.any_mxfp8()
                    && layer.attention_qkv == Precision::Fp16
                    && layer.attention_out == Precision::Fp16
            }),
            "collector supports only FP16/INT8 FFN recipes with FP16 attention",
        )?;
        let manifest = quantization_plan::graph_manifest(&graph, spec.model_sha256)?;
        let groups: Vec<_> = manifest
            .projections
            .iter()
            .filter_map(|p| {
                p.id.strip_suffix(".ffn.dual")
                    .map(|prefix| format!("{prefix}.ffn"))
            })
            .collect();
        require(
            !groups.is_empty() && groups.len() == graph.num_ffn_layers(),
            "FFN manifest coverage differs",
        )?;
        let precisions: Vec<_> = manifest
            .projections
            .iter()
            .map(|p| {
                let layer = recipe.layers[p.layer_index];
                let precision = if p.id.ends_with(".ffn.dual") {
                    layer.ffn_dual
                } else if p.id.ends_with(".ffn.down") {
                    layer.ffn_down
                } else if p.id.ends_with(".attention.qkv") {
                    layer.attention_qkv
                } else {
                    layer.attention_out
                };
                json!({"projection":p,"precision":precision})
            })
            .collect();
        let metadata = json!({"model_sha256":spec.model_sha256,"model_bytes":spec.model_bytes.len(),
            "binary":spec.binary,"compressed":spec.compressed,"model_version":desc.model_version,
            "inputs_version":inputs_version,"spatial_channels":22,"global_channels":19,"meta_channels":0,
            "board_size":19,"graph_sha256":spec.graph_sha256,"source_recipe_sha256":spec.source_recipe_sha256,
            "resolved_recipe_sha256":recipe.recipe_sha256,"projections":precisions,"ffn_group_ids":groups});
        Ok(Parsed {
            graph,
            recipe,
            groups,
            metadata,
        })
    }

    struct Slot {
        id: u64,
        workspace: CudaWorkspace,
        spatial: CudaSlice<f32>,
        global: CudaSlice<f32>,
        nan: [Vec<f32>; 5],
        recorder: Option<GroupCostRecorder>,
        warmup: Option<usize>,
        group_slot: Option<u64>,
        sequence: u64,
    }
    impl Slot {
        fn new(
            id: u64,
            batch: usize,
            rt: &CudaRuntime,
            stream: &Arc<CudaStream>,
            model: &CudaModel,
            graph: &LayerGraph,
            recipe: &ResolvedRecipe,
            cost: bool,
        ) -> Result<Self> {
            let (s, g) = batch_lengths(batch)?;
            let mut workspace = CudaWorkspace::new(stream, model, batch)?;
            let recorder = if cost {
                Some(model.prepare_group_cost(rt, stream, &mut workspace, graph, recipe)?)
            } else {
                None
            };
            let nan = HEADS.map(|(_, width)| vec![f32::from_bits(0x7fc00000); batch * width]);
            Ok(Self {
                id,
                workspace,
                spatial: stream.alloc_zeros::<f32>(s).map_err(|e| e.to_string())?,
                global: stream.alloc_zeros::<f32>(g).map_err(|e| e.to_string())?,
                nan,
                recorder,
                warmup: None,
                group_slot: None,
                sequence: 0,
            })
        }
        fn copy_and_poison_outputs(
            &mut self,
            stream: &Arc<CudaStream>,
            spatial: &[f32],
            global: &[f32],
        ) -> Result<()> {
            stream
                .memcpy_htod(spatial, &mut self.spatial)
                .map_err(|e| e.to_string())?;
            stream
                .memcpy_htod(global, &mut self.global)
                .map_err(|e| e.to_string())?;
            let targets = [
                &mut self.workspace.out_policy,
                &mut self.workspace.out_value,
                &mut self.workspace.out_misc,
                &mut self.workspace.out_moremisc,
                &mut self.workspace.out_ownership,
            ];
            for (target, values) in targets.into_iter().zip(&self.nan) {
                require(
                    target.len() == values.len(),
                    "workspace output allocation differs from physical shape",
                )?;
                stream
                    .memcpy_htod(values, target)
                    .map_err(|e| e.to_string())?;
            }
            stream.synchronize().map_err(|e| e.to_string())
        }
    }

    /// Owns one uploaded recipe and one explicit stream. Numeric and cost slots
    /// are separate, persistent per B; all public operation errors are sticky.
    pub struct Runtime {
        numeric_slots: BTreeMap<usize, Slot>,
        cost_slots: BTreeMap<usize, Slot>,
        model: CudaModel,
        stream: Arc<CudaStream>,
        rt: CudaRuntime,
        graph: LayerGraph,
        recipe: ResolvedRecipe,
        groups: Vec<String>,
        libraries: [Source; 2],
        executable: Source,
        poisoned: std::cell::Cell<bool>,
        next_slot: u64,
        attempts: u64,
    }
    impl Runtime {
        pub fn load(spec: ModelSpec<'_>) -> Result<(Self, Value)> {
            let environment = check_environment()?;
            let parsed = parse(&spec)?;
            let rt = CudaRuntime::new()?;
            cuda::validate_quantized_runtime(&rt)?;
            let stream = rt.device.new_stream().map_err(|e| e.to_string())?;
            let model = CudaModel::load_quantized(&parsed.graph, &rt, &stream, &parsed.recipe)?;
            cuda::ensure_requested_cuda_capabilities(&model)?;
            require(
                model.quantization_recipe_id() == Some(parsed.recipe.recipe_sha256.as_str()),
                "uploaded model recipe identity differs",
            )?;
            stream.synchronize().map_err(|e| e.to_string())?;
            let libraries = loaded_libraries()?;
            let executable = source(&std::env::current_exe().map_err(|e| e.to_string())?)?;
            let device = cuda::device_fingerprint(&rt.device)?;
            let uuid = rt.device.uuid().map_err(|e| e.to_string())?;
            let uuid_bytes: Vec<_> = uuid.bytes.iter().map(|x| *x as u8).collect();
            let mut driver_version = 0;
            require(
                unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut driver_version) }
                    == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS,
                "driver identity query failed",
            )?;
            let metadata = json!({"schema":"rustgo-ffn-calibration-runtime-v1","native":parsed.metadata,
                "device":{"gpu_name":device.gpu_name,"compute_capability":device.compute_capability,
                    "sm_count":device.sm_count,"l2_cache_bytes":device.l2_cache_bytes,"uuid_hex":hex::encode(uuid_bytes)},
                "driver_version":driver_version,"backend_build":cuda::backend_build_fingerprint(),
                "effective_tactics":environment,"stream_handle":format!("{:p}",stream.cu_stream()),
                "context_process_address":format!("{:p}",Arc::as_ptr(&rt.device)),"executable":executable,
                "library_order":["cublaslt","cublas"],"libraries":libraries,
                "library_resolution":"actual DLL symbols via GetModuleHandleExW FROM_ADDRESS",
                "model_uploads":1,"implicit_warmup":0,"graph_capture":0,"graph_replay":0,"automatic_retries":0,
                "raw_heads_prefilled_nan_each_forward":true,
                "scope":"same-policy calibration/FFN-cost observations; FP16 reference is not the strongest old FP16 control"});
            Ok((
                Self {
                    numeric_slots: BTreeMap::new(),
                    cost_slots: BTreeMap::new(),
                    model,
                    stream,
                    rt,
                    graph: parsed.graph,
                    recipe: parsed.recipe,
                    groups: parsed.groups,
                    libraries,
                    executable,
                    poisoned: std::cell::Cell::new(false),
                    next_slot: 1,
                    attempts: 0,
                },
                metadata,
            ))
        }
        fn begin(&self) -> Result<()> {
            require(
                !self.poisoned.replace(true),
                "runtime is poisoned; retry/reuse is forbidden",
            )
        }
        fn finish<T>(&self, result: Result<T>) -> Result<T> {
            if result.is_ok() {
                self.poisoned.set(false);
            }
            result
        }
        fn take_slot(&mut self, batch: usize, cost: bool) -> Result<Slot> {
            let slots = if cost {
                &mut self.cost_slots
            } else {
                &mut self.numeric_slots
            };
            if let Some(slot) = slots.remove(&batch) {
                return Ok(slot);
            }
            let id = self.next_slot;
            self.next_slot = id
                .checked_add(1)
                .ok_or("collector slot identity exhausted")?;
            Slot::new(
                id,
                batch,
                &self.rt,
                &self.stream,
                &self.model,
                &self.graph,
                &self.recipe,
                cost,
            )
        }
        fn forward(
            &mut self,
            slot: &mut Slot,
            spatial: &[f32],
            global: &[f32],
            batch: usize,
            phase: &'static str,
            iteration: usize,
            label: &str,
            diagnostic: Option<Phase>,
            before_apply: &mut dyn FnMut(&Value) -> Result<()>,
        ) -> Result<(Heads, Option<Value>)> {
            check_environment()?;
            require(
                slot.workspace.batch() == batch,
                "workspace physical batch mismatch",
            )?;
            slot.copy_and_poison_outputs(&self.stream, spatial, global)?;
            self.attempts = self
                .attempts
                .checked_add(1)
                .ok_or("forward counter exhausted")?;
            // This callback is the final fallible host step before apply. A
            // failed reservation or failed forward poisons the entire runtime.
            before_apply(
                &json!({"phase":phase,"iteration":iteration,"physical_batch":batch,
                "runtime_attempt":self.attempts,"collector_slot":slot.id,
                "workspace_kind":if slot.recorder.is_some(){"cost"}else{"numeric"},
                "stream_handle":format!("{:p}",self.stream.cu_stream()),
                "group_workspace_slot":slot.group_slot,"tensor_sha256":label,
                "model_sha256":self.recipe.model_sha256,"graph_sha256":self.recipe.graph_sha256,
                "recipe_sha256":self.recipe.recipe_sha256}),
            )?;
            if let Some(phase) = diagnostic {
                let recorder = slot.recorder.as_mut().ok_or("diagnostic recorder absent")?;
                self.model.apply_with_group_cost(
                    &self.rt,
                    &self.stream,
                    &mut slot.workspace,
                    &slot.spatial,
                    &slot.global,
                    recorder,
                    phase,
                    label,
                )?;
            } else {
                self.model.apply(
                    &self.rt,
                    &self.stream,
                    &mut slot.workspace,
                    &slot.spatial,
                    &slot.global,
                )?;
            }
            self.stream.synchronize().map_err(|e| e.to_string())?;
            let output = slot.workspace.to_host(&self.stream)?;
            self.stream.synchronize().map_err(|e| e.to_string())?;
            let heads = Heads {
                policy: output.policy,
                value: output.value,
                misc: output.misc,
                moremisc: output.moremisc,
                ownership: output.ownership,
            };
            heads.validate(batch)?;
            let sample = if let Some(phase) = diagnostic {
                let sample = slot
                    .recorder
                    .as_mut()
                    .ok_or("diagnostic recorder absent")?
                    .drain()?;
                let value = serde_json::to_value(sample).map_err(|e| e.to_string())?;
                validate_sample(
                    &value,
                    &self.groups,
                    &self.recipe,
                    batch,
                    label,
                    phase,
                    slot,
                )?;
                Some(value)
            } else {
                None
            };
            Ok((heads, sample))
        }
        pub fn numeric(
            &mut self,
            spatial: &[f32],
            global: &[f32],
            batch: usize,
            before_apply: &mut dyn FnMut(&Value) -> Result<()>,
        ) -> Result<Heads> {
            self.begin()?;
            let result = (|| {
                let label = input_label(spatial, global, batch)?;
                let mut slot = self.take_slot(batch, false)?;
                let (heads, _) = self.forward(
                    &mut slot,
                    spatial,
                    global,
                    batch,
                    "numeric",
                    0,
                    &label,
                    None,
                    before_apply,
                )?;
                self.numeric_slots.insert(batch, slot);
                Ok(heads)
            })();
            self.finish(result)
        }
        /// Exactly W warm forwards on the caller's first frozen cost tensor.
        /// A second warm call for a B is rejected, even with identical input.
        pub fn warm_cost(
            &mut self,
            spatial: &[f32],
            global: &[f32],
            batch: usize,
            warmup: usize,
            before_apply: &mut dyn FnMut(&Value) -> Result<()>,
        ) -> Result<Value> {
            self.begin()?;
            let result = (|| {
                require((1..=64).contains(&warmup), "warmup count must be in 1..=64")?;
                let label = input_label(spatial, global, batch)?;
                require(
                    !self.cost_slots.contains_key(&batch),
                    "cost workspace for B already warmed; no extra warmup",
                )?;
                let mut slot = self.take_slot(batch, true)?;
                let mut samples = Vec::with_capacity(warmup);
                for iteration in 0..warmup {
                    let (heads, sample) = self.forward(
                        &mut slot,
                        spatial,
                        global,
                        batch,
                        "cost-warmup",
                        iteration,
                        &label,
                        Some(Phase::Warmup),
                        before_apply,
                    )?;
                    samples.push(json!({"sample":sample,"raw_output_heads":heads.hashes()}));
                }
                require(
                    slot.recorder
                        .as_ref()
                        .is_some_and(GroupCostRecorder::ready_to_measure),
                    "fixed warmup failed to establish a setup-free route; no extra warmup permitted",
                )?;
                slot.warmup = Some(warmup);
                let result = json!({"physical_batch":batch,"collector_slot":slot.id,"group_workspace_slot":slot.group_slot,
                    "warmup":warmup,"input_tensor_sha256":label,"samples":samples,"ready_to_measure":true});
                self.cost_slots.insert(batch, slot);
                Ok(result)
            })();
            self.finish(result)
        }
        /// Ordinary-before + exactly M measured forwards + ordinary-after.
        /// Host copies, NaN initialization and serialization lie outside the
        /// GroupCost event intervals. Only raw bit identity is asserted here.
        pub fn measure_cost(
            &mut self,
            spatial: &[f32],
            global: &[f32],
            batch: usize,
            iterations: usize,
            before_apply: &mut dyn FnMut(&Value) -> Result<()>,
        ) -> Result<(Heads, Value)> {
            self.begin()?;
            let result = (|| {
                require(
                    (1..=128).contains(&iterations),
                    "measure count must be in 1..=128",
                )?;
                let label = input_label(spatial, global, batch)?;
                let mut slot = self
                    .cost_slots
                    .remove(&batch)
                    .ok_or("cost B has no completed fixed warmup")?;
                require(
                    slot.warmup.is_some()
                        && slot
                            .recorder
                            .as_ref()
                            .is_some_and(GroupCostRecorder::ready_to_measure),
                    "cost recorder is not ready after fixed warmup",
                )?;
                let (expected, _) = self.forward(
                    &mut slot,
                    spatial,
                    global,
                    batch,
                    "cost-before",
                    0,
                    &label,
                    None,
                    before_apply,
                )?;
                let mut samples = Vec::with_capacity(iterations);
                for iteration in 0..iterations {
                    let (actual, sample) = self.forward(
                        &mut slot,
                        spatial,
                        global,
                        batch,
                        "cost-measure",
                        iteration,
                        &label,
                        Some(Phase::Measure),
                        before_apply,
                    )?;
                    require(
                        expected.bitwise_eq(&actual),
                        "measured five-head raw bits differ from ordinary-before",
                    )?;
                    samples.push(json!({"sample":sample,"raw_output_heads":actual.hashes(),"bitwise_matches_before":true}));
                }
                let (after, _) = self.forward(
                    &mut slot,
                    spatial,
                    global,
                    batch,
                    "cost-after",
                    0,
                    &label,
                    None,
                    before_apply,
                )?;
                require(
                    expected.bitwise_eq(&after),
                    "ordinary-after five-head raw bits differ from ordinary-before",
                )?;
                let metadata = json!({"physical_batch":batch,"collector_slot":slot.id,"group_workspace_slot":slot.group_slot,
                    "warmup_once_for_batch":slot.warmup,"input_tensor_sha256":label,"measurements":iterations,
                    "ordinary_before_heads":expected.hashes(),"samples":samples,"ordinary_after_heads":after.hashes(),
                    "all_five_heads_bitwise_equal":true,"forward_attempts_this_input":iterations+2,
                    "scope":"complete FFN event intervals; no ABBA/end-to-end/accuracy certification"});
                self.cost_slots.insert(batch, slot);
                Ok((expected, metadata))
            })();
            self.finish(result)
        }
        /// Must pass before the caller publishes its complete marker. The
        /// caller separately rechecks its model/recipe/input source files.
        pub fn recheck(&self) -> Result<()> {
            self.begin()?;
            let result = (|| {
                check_environment()?;
                cuda::validate_quantized_runtime(&self.rt)?;
                require(
                    self.model.quantization_recipe_id() == Some(self.recipe.recipe_sha256.as_str()),
                    "actual uploaded recipe changed",
                )?;
                self.stream.synchronize().map_err(|e| e.to_string())?;
                require(
                    loaded_libraries()? == self.libraries,
                    "actual loaded DLL source changed",
                )?;
                require(
                    source(&self.executable.path)? == self.executable,
                    "collector executable source changed",
                )
            })();
            self.finish(result)
        }
    }

    fn validate_sample(
        value: &Value,
        groups: &[String],
        recipe: &ResolvedRecipe,
        batch: usize,
        label: &str,
        phase: Phase,
        slot: &mut Slot,
    ) -> Result<()> {
        let phase_name = match phase {
            Phase::Warmup => "warmup",
            Phase::Measure => "measure",
        };
        require(
            value["physical_batch"].as_u64() == Some(batch as u64)
                && value["activation_rows"].as_u64() == Some((batch * 361) as u64)
                && value["input_label_sha256"].as_str() == Some(label)
                && value["phase"].as_str() == Some(phase_name),
            "GroupCost sample shape/input/phase mismatch",
        )?;
        require(
            value["binding"]["model_sha256"].as_str() == Some(recipe.model_sha256.as_str())
                && value["binding"]["graph_sha256"].as_str() == Some(recipe.graph_sha256.as_str())
                && value["binding"]["recipe_sha256"].as_str()
                    == Some(recipe.recipe_sha256.as_str()),
            "GroupCost sample source binding mismatch",
        )?;
        let id = value["workspace_slot"]
            .as_u64()
            .filter(|x| *x > 0)
            .ok_or("missing GroupCost workspace identity")?;
        require(
            slot.group_slot.is_none_or(|expected| expected == id),
            "GroupCost workspace changed",
        )?;
        slot.group_slot = Some(id);
        slot.sequence = slot
            .sequence
            .checked_add(1)
            .ok_or("GroupCost sequence exhausted")?;
        require(
            value["sequence"].as_u64() == Some(slot.sequence),
            "GroupCost sequence missing/reused",
        )?;
        require(
            phase != Phase::Measure || value["setup_observed"] == false,
            "setup during measured sample",
        )?;
        let observed = value["groups"]
            .as_array()
            .ok_or("GroupCost groups absent")?;
        require(
            observed.len() == groups.len(),
            "incomplete GroupCost group coverage",
        )?;
        for (entry, expected) in observed.iter().zip(groups) {
            require(
                entry["group"]["id"].as_str() == Some(expected.as_str()),
                "GroupCost group order/identity changed",
            )?;
            let bits: u32 = entry["elapsed_ms_bits"]
                .as_u64()
                .ok_or("missing elapsed bits")?
                .try_into()
                .map_err(|_| "elapsed bits overflow")?;
            let elapsed = f32::from_bits(bits);
            require(
                elapsed.is_finite()
                    && elapsed >= 0.0
                    && entry["elapsed_ms"]
                        .as_f64()
                        .is_some_and(|x| (x as f32).to_bits() == bits),
                "invalid GroupCost elapsed f32 bits",
            )?;
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub use gpu::Runtime;

#[cfg(test)]
#[path = "runtime_tests.rs"]
mod tests;
