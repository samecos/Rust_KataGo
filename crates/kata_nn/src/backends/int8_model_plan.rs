//! Explicit model adapter for the partial INT8 LT cache, not a full model plan.
//! No configuration or normal loading path installs a session automatically.
use super::{CudaModel, CudaWorkspace, CudaRuntime, CudaStream, LayerBuf, ProjectionWeight};
use crate::backends::int8_algorithm_plan::{self as plan, Binding, ProblemKey};
use crate::onnx_parser::LayerGraph;
use crate::quantization_plan::{self, Precision, ResolvedRecipe};
use serde::Serialize;
use std::{collections::BTreeSet, sync::Arc};

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Int8ModelProjection {
    pub id: String,
    pub shape: ProblemKey,
}

/// Stable projection coverage plus the deduplicated actual physical-shape list.
/// FFN dual N is the actual 2*hidden GEMM output, unlike manifest logical N.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Int8ModelInventory {
    pub model_sha256: String,
    pub graph_sha256: String,
    pub recipe_sha256: String,
    pub physical_batch: usize,
    pub projections: Vec<Int8ModelProjection>,
    pub shapes: Vec<ProblemKey>,
}

fn require(ok: bool, message: &str) -> Result<(), String> {
    if ok { Ok(()) } else { Err(format!("model INT8 cache: {message}")) }
}

fn describe(weight: &ProjectionWeight) -> (Precision, usize, usize, usize) {
    match weight {
        ProjectionWeight::Fp16(w) => (Precision::Fp16, w.n, w.k, w.kp),
        ProjectionWeight::Int8(w) => (Precision::Int8, w.n, w.k, w.kp),
        // The caller rejects MXFP8 before traversing any uploaded projection.
        ProjectionWeight::Mxfp8(_) => (Precision::Mxfp8, 0, 0, 0),
    }
}

impl CudaModel {
    /// Derive authoritative INT8 shapes from the actual uploaded model. A caller
    /// cannot obtain partial coverage by supplying a smaller shape directory.
    pub fn int8_algorithm_inventory(&self, graph: &LayerGraph, recipe: &ResolvedRecipe,
        physical_batch: usize) -> Result<Int8ModelInventory, String> {
        quantization_plan::validate_resolved_recipe(graph, recipe)?;
        require(self.quantization_recipe_id() == Some(recipe.recipe_sha256.as_str()),
            "recipe differs from the actual loaded unified model")?;
        require(!self.has_mxfp8() && self.ffn_compact.is_none(),
            "this explicit adapter supports FP16/INT8 graphs without compact artifacts")?;
        require(self.seq_len == 361 && graph.layers.len() == self.layers.len(),
            "graph/sequence differs from the uploaded model")?;
        let rows = physical_batch.checked_mul(self.seq_len).ok_or("INT8 model row overflow")?;
        require(physical_batch > 0, "physical batch must be positive")?;
        let manifest = quantization_plan::graph_manifest(graph, &recipe.model_sha256)?;
        let mut projections = Vec::new();
        let mut shapes = BTreeSet::new();
        for projection in &manifest.projections {
            let precision = &recipe.layers[projection.layer_index];
            let layer = &self.layers[projection.layer_index];
            let (expected_precision, actual, multiplier) = if projection.id.ends_with(".ffn.dual") {
                let actual = match layer {
                    LayerBuf::Ffn { dual, .. } => describe(dual),
                    LayerBuf::Int8Ffn { dual, .. } => (Precision::Int8, dual.n, dual.k, dual.kp),
                    _ => return Err("INT8 directory FFN dual addresses a different uploaded layer".into()),
                };
                (precision.ffn_dual, actual, 2usize)
            } else if projection.id.ends_with(".ffn.down") {
                let actual = match layer {
                    LayerBuf::Ffn { down, .. } => describe(down),
                    LayerBuf::Int8Ffn { down, .. } => (Precision::Int8, down.n, down.k, down.kp),
                    _ => return Err("INT8 directory FFN down addresses a different uploaded layer".into()),
                };
                (precision.ffn_down, actual, 1)
            } else if projection.id.ends_with(".attention.qkv") {
                let actual = match layer {
                    LayerBuf::Attention { qkv, .. } => describe(qkv),
                    LayerBuf::Int8Attention { qkv, .. } => (Precision::Int8, qkv.n, qkv.k, qkv.kp),
                    _ => return Err("INT8 directory QKV addresses a different uploaded layer".into()),
                };
                (precision.attention_qkv, actual, 1)
            } else if projection.id.ends_with(".attention.out") {
                let actual = match layer {
                    LayerBuf::Attention { out, .. } => describe(out),
                    LayerBuf::Int8Attention { out, .. } => (Precision::Int8, out.n, out.k, out.kp),
                    _ => return Err("INT8 directory attention out addresses a different uploaded layer".into()),
                };
                (precision.attention_out, actual, 1)
            } else {
                return Err("INT8 directory contains an unsupported projection kind".into());
            };
            let (actual_precision, n, k, kp) = actual;
            require(actual_precision == expected_precision
                && projection.n.checked_mul(multiplier) == Some(n) && projection.k == k,
                "uploaded projection precision/shape differs from its validated recipe")?;
            if actual_precision == Precision::Int8 {
                let shape = ProblemKey::new(rows, n, k, kp)?;
                shapes.insert(shape);
                projections.push(Int8ModelProjection { id: projection.id.clone(), shape });
            }
        }
        let actual_count: usize = self.layers.iter().map(|layer| match layer {
            LayerBuf::Int8Ffn { .. } | LayerBuf::Int8Attention { .. } => 2,
            LayerBuf::Ffn { dual: a, down: b, .. } | LayerBuf::Attention { qkv: a, out: b, .. } =>
                usize::from(matches!(a, ProjectionWeight::Int8(_))) + usize::from(matches!(b, ProjectionWeight::Int8(_))),
            _ => 0,
        }).sum();
        require(actual_count > 0 && projections.len() == actual_count,
            "directory does not cover every uploaded INT8 projection")?;
        projections.sort_by(|a, b| a.id.cmp(&b.id));
        let shapes: Vec<_> = shapes.into_iter().collect();
        plan::validate_inventory(&shapes)?;
        Ok(Int8ModelInventory { model_sha256: recipe.model_sha256.clone(),
            graph_sha256: recipe.graph_sha256.clone(), recipe_sha256: recipe.recipe_sha256.clone(),
            physical_batch, projections, shapes })
    }

    fn validate_int8_cache_owner(&self, rt: &CudaRuntime, stream: &Arc<CudaStream>,
        ws: &CudaWorkspace) -> Result<(), String> {
        super::validate_workspace_model(self.model_instance_id, ws.model_instance_id)?;
        require(Arc::ptr_eq(&self.owner_context, &rt.device)
            && Arc::ptr_eq(&self.owner_context, stream.context()), "runtime/stream differs from model context")?;
        require(ws.int8.is_some(), "model workspace has no INT8 operations")
    }

    fn model_int8_directory(&self, rt: &CudaRuntime, stream: &Arc<CudaStream>,
        ws: &CudaWorkspace, graph: &LayerGraph, recipe: &ResolvedRecipe,
        binding: &Binding) -> Result<Int8ModelInventory, String> {
        self.validate_int8_cache_owner(rt, stream, ws)?;
        let inventory = self.int8_algorithm_inventory(graph, recipe, ws.batch)?;
        require(binding.model_sha256 == inventory.model_sha256
            && binding.graph_sha256 == inventory.graph_sha256
            && binding.recipe_sha256 == inventory.recipe_sha256,
            "plan model/graph/recipe binding differs from the uploaded model")?;
        // Other provenance fields are still supplied by the authoritative loader;
        // the leaf adapter checks actual device/LT/executable and buffer owners.
        Ok(inventory)
    }

    /// Explicit setup only. Normal model loading never invokes this method.
    pub fn begin_int8_algorithm_recording(&self, rt: &CudaRuntime, stream: &Arc<CudaStream>,
        ws: &mut CudaWorkspace, graph: &LayerGraph, recipe: &ResolvedRecipe,
        binding: Binding) -> Result<Int8ModelInventory, String> {
        let inventory = self.model_int8_directory(rt, stream, ws, graph, recipe, &binding)?;
        ws.int8.as_mut().ok_or("missing INT8 workspace")?
            .begin_algorithm_recording(rt, stream, binding, inventory.shapes.clone())?;
        Ok(inventory)
    }

    pub fn install_int8_algorithm_plan(&self, rt: &CudaRuntime, stream: &Arc<CudaStream>,
        ws: &mut CudaWorkspace, graph: &LayerGraph, recipe: &ResolvedRecipe,
        binding: &Binding, bytes: &[u8], expected_sha256: &str) -> Result<Int8ModelInventory, String> {
        let inventory = self.model_int8_directory(rt, stream, ws, graph, recipe, binding)?;
        ws.int8.as_mut().ok_or("missing INT8 workspace")?
            .install_algorithm_plan(rt, stream, bytes, expected_sha256, binding, &inventory.shapes)?;
        Ok(inventory)
    }

    pub fn export_int8_algorithm_plan(&self, rt: &CudaRuntime, stream: &Arc<CudaStream>,
        ws: &mut CudaWorkspace) -> Result<Vec<u8>, String> {
        let result = (|| {
            self.validate_int8_cache_owner(rt, stream, ws)?;
            ws.int8.as_mut().ok_or("missing INT8 workspace")?.export_algorithm_plan(stream)
        })();
        ws.int8_algorithm_forward_exit(result)
    }

    /// External Graph/copy consumers can report their operation result through
    /// this boundary. They must actually call it; it does not intercept CUDA APIs.
    pub fn complete_int8_algorithm_operation<T>(&self, rt: &CudaRuntime,
        stream: &Arc<CudaStream>, ws: &mut CudaWorkspace, operation: Result<T, String>) -> Result<T, String> {
        let result = self.validate_int8_cache_owner(rt, stream, ws)
            .and_then(|_| ws.int8_algorithm_forward_entry(rt, stream)).and(operation);
        ws.int8_algorithm_forward_exit(result)
    }
}
