//! Model-bound, immutable per-projection precision recipes.
//!
//! This is a CPU-only description of FP16, symmetric INT8 and experimental
//! MXFP8 execution semantics, not an accuracy/performance certificate. Unlisted
//! projections stay FP16. IDs come from the nested block grammar, never from
//! caller-supplied vector indices. Both parsers lower to this same grammar.
//!
//! Version 1 INT8 means signed W8A8, static absmax scales per output channel,
//! dynamic absmax scales per token, and INT32 GEMM accumulation. Activations,
//! normalization, attention core and output heads retain existing semantics.
//! Legacy (recipe=1, semantics=1) identities are frozen. MXFP8 requires explicit
//! (2,2) and a separate canonical envelope binding its complete format contract
//! and original FP32 tensor bits. Future MXFP8 semantic changes require a new
//! contract/recipe version; they must never alter the v1 canonical path.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::onnx_parser::{Layer, LayerGraph, Tensor, TensorData};

pub const RECIPE_SCHEMA: &str = "rustgo-precision-recipe";
pub const RECIPE_VERSION: u32 = 1;
pub const QUANTIZATION_SEMANTICS_VERSION: u32 = 1;
pub const MXFP8_RECIPE_VERSION: u32 = 2;
pub const MXFP8_QUANTIZATION_SEMANTICS_VERSION: u32 = 2;
/// Shared with the production GPU module; it must not maintain a second copy.
pub const MXFP8_QUANTIZATION_VERSION: &str =
    "mxfp8-e4m3-rne-ue8m0-ceil-fp32-block32-half-boundary-v1";
// Domain separation also makes a graph-description format change explicit.
const GRAPH_SCHEMA: &str = "rustgo-layer-graph-v1";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    #[default]
    Fp16,
    Int8,
    Mxfp8,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LayerPrecision {
    pub ffn_dual: Precision,
    pub ffn_down: Precision,
    pub attention_qkv: Precision,
    pub attention_out: Precision,
}

impl LayerPrecision {
    pub fn any_int8(&self) -> bool {
        [
            self.ffn_dual,
            self.ffn_down,
            self.attention_qkv,
            self.attention_out,
        ]
        .contains(&Precision::Int8)
    }

    pub fn any_mxfp8(&self) -> bool {
        [
            self.ffn_dual,
            self.ffn_down,
            self.attention_qkv,
            self.attention_out,
        ]
        .contains(&Precision::Mxfp8)
    }

    pub fn any_quantized(&self) -> bool {
        [
            self.ffn_dual,
            self.ffn_down,
            self.attention_qkv,
            self.attention_out,
        ]
        .iter()
        .any(|&precision| precision != Precision::Fp16)
    }
}

/// Fixed, CPU-visible numerical contract, included verbatim in every v2 hash.
/// These fields are descriptive constants, not user-adjustable kernel options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Mxfp8Semantics {
    pub schema: &'static str,
    pub revision: u32,
    pub quantization_version: &'static str,
    pub value_format: &'static str,
    pub block_size: u32,
    pub weight_source: &'static str,
    pub activation_source: &'static str,
    pub scale_format: &'static str,
    pub scale_rounding: &'static str,
    pub value_rounding: &'static str,
    pub zero_block: &'static str,
    pub padding: &'static str,
    pub scale_layout: &'static str,
    pub gemm_layout: &'static str,
    pub compute: &'static str,
    pub fast_accum: bool,
    pub outputs: &'static str,
    pub algorithm_selection: &'static str,
    pub invalid_output: &'static str,
}

pub fn mxfp8_semantics() -> Mxfp8Semantics {
    Mxfp8Semantics {
        schema: "rustgo-mxfp8-semantics-v1",
        revision: 1,
        quantization_version: MXFP8_QUANTIZATION_VERSION,
        value_format: "e4m3fn-rne-satfinite",
        block_size: 32,
        weight_source: "lowered-model-f32-tensor-before-any-half-conversion",
        activation_source: "existing-f16-storage-boundary",
        scale_format: "ue8m0",
        scale_rounding: "fp32-rne(absmax/448)-then-positive-infinity-satfinite",
        value_rounding: "fp32-rne(x*pow2(127-scale-byte))-then-e4m3-rne-satfinite",
        zero_block: "q=positive-zero;scale=one;nonzero-block-preserves-negative-zero",
        padding: "K-to32-q=positive-zero;scale-outer-to128-inner-to4-unused-bytes=0",
        scale_layout: "cublaslt-128x4-swizzled-v1",
        gemm_layout: "TN:row-major-X-times-row-major-W-transpose;alpha-f32-one",
        compute: "fp32-accumulation",
        fast_accum: false,
        outputs: "ffn-dual,attention-qkv=f16-beta0;ffn-down,attention-out=f32-beta1",
        algorithm_selection:
            "first-of32-heuristics-with-AlgorithmCheck-TensorCore-E4M3-FP32-no-timing-v1",
        invalid_output: "sticky-nan-inf-nonfinite-output;discard-entire-forward",
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedRecipe {
    /// Exactly one entry per graph layer, including FP16-only layers.
    pub layers: Vec<LayerPrecision>,
    /// Canonical resolved identity: independent of whitespace/entry order and
    /// omission vs explicit FP16, but binds every reachable precision choice.
    pub recipe_sha256: String,
    pub graph_sha256: String,
    pub model_sha256: String,
}

/// Portable recipe JSON. All fields are mandatory; unknown/duplicate JSON
/// fields and unsupported enum values fail during deserialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrecisionRecipe {
    pub schema: String,
    pub version: u32,
    pub quantization_semantics_version: u32,
    pub model_sha256: String,
    pub graph_sha256: String,
    pub projections: Vec<ProjectionPrecision>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionPrecision {
    pub id: String,
    /// GEMM output width; FFN dual describes each gate/up matrix (not 2*n).
    pub expected_n: usize,
    pub expected_k: usize,
    pub precision: Precision,
}

/// Serialize directly. Do not convert the canonical envelope through
/// `serde_json::Value`: its sorted map order would change the canonical bytes.
#[derive(Debug, Serialize)]
pub struct RecipeIdentityExport {
    pub schema: &'static str,
    pub version: u32,
    pub model_sha256: String,
    pub graph_sha256: String,
    pub source_recipe_file: &'static str,
    pub source_recipe_file_sha256: String,
    pub recipe_sha256: String,
    pub canonical_encoding: &'static str,
    pub canonical_envelope: CanonicalRecipeIdentity,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum CanonicalRecipeIdentity {
    V1(PrecisionRecipe),
    V2(Mxfp8ResolvedIdentity),
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphManifest {
    pub schema: &'static str,
    pub model_sha256: String,
    pub graph_sha256: String,
    pub projections: Vec<ProjectionManifest>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProjectionManifest {
    pub id: String,
    /// Diagnostic only; recipes cannot address a layer by this index.
    pub layer_index: usize,
    pub n: usize,
    pub k: usize,
    #[serde(skip)]
    kind: ProjectionKind,
}

#[derive(Debug, Clone, Copy)]
enum ProjectionKind {
    FfnDual,
    FfnDown,
    AttentionQkv,
    AttentionOut,
}

impl ProjectionKind {
    fn slot(self, layer: &mut LayerPrecision) -> &mut Precision {
        match self {
            Self::FfnDual => &mut layer.ffn_dual,
            Self::FfnDown => &mut layer.ffn_down,
            Self::AttentionQkv => &mut layer.attention_qkv,
            Self::AttentionOut => &mut layer.attention_out,
        }
    }
}

pub fn load_and_resolve(
    path: &Path,
    graph: &LayerGraph,
    model_sha256: &str,
) -> Result<ResolvedRecipe, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("read precision recipe {}: {e}", path.display()))?;
    let recipe: PrecisionRecipe = serde_json::from_slice(&bytes)
        .map_err(|e| format!("parse precision recipe {}: {e}", path.display()))?;
    resolve_recipe(&recipe, graph, model_sha256)
}

/// Resolve and export the exact canonical object used by the Rust loader. The
/// same input bytes are parsed and source-hashed, and the original model bytes
/// must have produced both `graph` and `model_sha256` at the caller's boundary.
/// CPU resolution alone does not certify GPU support, accuracy or performance.
pub fn resolve_recipe_identity(
    source_recipe_bytes: &[u8],
    graph: &LayerGraph,
    model_sha256: &str,
) -> Result<RecipeIdentityExport, String> {
    let recipe: PrecisionRecipe = serde_json::from_slice(source_recipe_bytes)
        .map_err(|e| format!("parse precision recipe for identity export: {e}"))?;
    let manifest = graph_manifest(graph, model_sha256)?;
    let (resolved, canonical_envelope) =
        resolve_canonical_with_manifest(&recipe, graph, &manifest)?;
    Ok(RecipeIdentityExport {
        schema: "rustgo-recipe-identity-export",
        version: 1,
        model_sha256: resolved.model_sha256,
        graph_sha256: resolved.graph_sha256,
        source_recipe_file: "source-recipe.json",
        source_recipe_file_sha256: hex::encode(Sha256::digest(source_recipe_bytes)),
        recipe_sha256: resolved.recipe_sha256,
        canonical_encoding: "serde-json-compact-ordered-v1",
        canonical_envelope,
    })
}

pub fn fp16_recipe(graph: &LayerGraph, model_sha256: &str) -> Result<ResolvedRecipe, String> {
    let manifest = graph_manifest(graph, model_sha256)?;
    resolve_with_manifest(&template_from_manifest(&manifest), graph, &manifest)
}

/// Describes all selectable projections and the exact graph identity. This
/// validates the nested transformer grammar before assigning stable IDs.
pub fn graph_manifest(graph: &LayerGraph, model_sha256: &str) -> Result<GraphManifest, String> {
    check_hash("model_sha256", model_sha256)?;
    let projections = scan_projections(graph)?;
    let graph_sha256 = hash_json(&graph_description(graph)?)?;
    Ok(GraphManifest {
        schema: GRAPH_SCHEMA,
        model_sha256: model_sha256.to_owned(),
        graph_sha256,
        projections,
    })
}

/// Recompute the CPU graph identity at an execution boundary. This does not
/// certify numerical accuracy; it prevents a resolved recipe being paired with
/// a structurally or semantically different graph.
pub fn graph_sha256(graph: &LayerGraph) -> Result<String, String> {
    scan_projections(graph)?;
    hash_json(&graph_description(graph)?)
}

/// Returns an editable, all-FP16 JSON template with every supported projection.
/// Serialize with `serde_json::to_string_pretty`; change selected precisions.
pub fn template_recipe(graph: &LayerGraph, model_sha256: &str) -> Result<PrecisionRecipe, String> {
    Ok(template_from_manifest(&graph_manifest(
        graph,
        model_sha256,
    )?))
}

/// Explicit experimental v2 recipe. All unselected projections are FP16; this
/// CPU helper does not claim GPU support, calibration, or numerical validation.
/// Further INT8 selections may be applied before calling `resolve_recipe`.
pub fn mxfp8_recipe(
    graph: &LayerGraph,
    model_sha256: &str,
    projection_ids: &[&str],
) -> Result<PrecisionRecipe, String> {
    if projection_ids.is_empty() {
        return Err("MXFP8 v2 recipe requires at least one explicit projection".into());
    }
    let mut requested = BTreeSet::new();
    for &id in projection_ids {
        if !requested.insert(id) {
            return Err(format!("duplicate MXFP8 projection ID {id}"));
        }
    }
    let mut recipe = template_recipe(graph, model_sha256)?;
    recipe.version = MXFP8_RECIPE_VERSION;
    recipe.quantization_semantics_version = MXFP8_QUANTIZATION_SEMANTICS_VERSION;
    for entry in &mut recipe.projections {
        if requested.remove(entry.id.as_str()) {
            entry.precision = Precision::Mxfp8;
        }
    }
    if !requested.is_empty() {
        return Err(format!("unknown MXFP8 projection IDs: {requested:?}"));
    }
    resolve_recipe(&recipe, graph, model_sha256)?;
    Ok(recipe)
}

fn template_from_manifest(manifest: &GraphManifest) -> PrecisionRecipe {
    PrecisionRecipe {
        schema: RECIPE_SCHEMA.into(),
        version: RECIPE_VERSION,
        quantization_semantics_version: QUANTIZATION_SEMANTICS_VERSION,
        model_sha256: manifest.model_sha256.clone(),
        graph_sha256: manifest.graph_sha256.clone(),
        projections: manifest
            .projections
            .iter()
            .map(|p| ProjectionPrecision {
                id: p.id.clone(),
                expected_n: p.n,
                expected_k: p.k,
                precision: Precision::Fp16,
            })
            .collect(),
    }
}

pub fn resolve_recipe(
    recipe: &PrecisionRecipe,
    graph: &LayerGraph,
    model_sha256: &str,
) -> Result<ResolvedRecipe, String> {
    resolve_with_manifest(recipe, graph, &graph_manifest(graph, model_sha256)?)
}

/// Validate a resolved value again at an execution boundary. The public fields
/// allow callers to accidentally change a precision after resolution, so graph
/// identity and layer count alone are insufficient: rebuild the canonical
/// recipe from all supported slots and compare its complete identity and slots.
/// The file loader remains responsible for binding `model_sha256` to the exact
/// source bytes that produced `graph`.
pub fn validate_resolved_recipe(graph: &LayerGraph, recipe: &ResolvedRecipe) -> Result<(), String> {
    if recipe.layers.len() != graph.layers.len() {
        return Err("resolved precision recipe layer count does not match the graph".into());
    }
    check_hash("resolved recipe_sha256", &recipe.recipe_sha256)?;
    check_hash("resolved graph_sha256", &recipe.graph_sha256)?;
    let manifest = graph_manifest(graph, &recipe.model_sha256)?;
    if recipe.graph_sha256 != manifest.graph_sha256 {
        return Err("resolved precision recipe graph_sha256 mismatch".into());
    }
    let mut source = template_from_manifest(&manifest);
    for (entry, projection) in source.projections.iter_mut().zip(&manifest.projections) {
        let mut layer = recipe.layers[projection.layer_index];
        entry.precision = *projection.kind.slot(&mut layer);
    }
    if source
        .projections
        .iter()
        .any(|entry| entry.precision == Precision::Mxfp8)
    {
        source.version = MXFP8_RECIPE_VERSION;
        source.quantization_semantics_version = MXFP8_QUANTIZATION_SEMANTICS_VERSION;
    }
    let canonical = resolve_with_manifest(&source, graph, &manifest)?;
    // This also rejects INT8 in irrelevant slots (e.g. attention_qkv on an FFN,
    // or any precision on a head) even when all supported projection hashes fit.
    if recipe.layers != canonical.layers {
        return Err("resolved precision recipe contains unsupported projection assignments".into());
    }
    if recipe.recipe_sha256 != canonical.recipe_sha256 {
        return Err("resolved precision recipe canonical recipe_sha256 mismatch".into());
    }
    Ok(())
}

fn resolve_with_manifest(
    recipe: &PrecisionRecipe,
    graph: &LayerGraph,
    manifest: &GraphManifest,
) -> Result<ResolvedRecipe, String> {
    resolve_canonical_with_manifest(recipe, graph, manifest).map(|(resolved, _)| resolved)
}

fn resolve_canonical_with_manifest(
    recipe: &PrecisionRecipe,
    graph: &LayerGraph,
    manifest: &GraphManifest,
) -> Result<(ResolvedRecipe, CanonicalRecipeIdentity), String> {
    let uses_mxfp8 = recipe
        .projections
        .iter()
        .any(|p| p.precision == Precision::Mxfp8);
    let versions = if uses_mxfp8 {
        (MXFP8_RECIPE_VERSION, MXFP8_QUANTIZATION_SEMANTICS_VERSION)
    } else {
        (RECIPE_VERSION, QUANTIZATION_SEMANTICS_VERSION)
    };
    if recipe.schema != RECIPE_SCHEMA
        || (recipe.version, recipe.quantization_semantics_version) != versions
    {
        return Err(
            "unsupported precision recipe schema/version/quantization semantics version: FP16/INT8 require (1,1), recipes containing MXFP8 require (2,2)".into(),
        );
    }
    check_hash("recipe model_sha256", &recipe.model_sha256)?;
    check_hash("recipe graph_sha256", &recipe.graph_sha256)?;
    if recipe.model_sha256 != manifest.model_sha256 {
        return Err("precision recipe model_sha256 mismatch".into());
    }
    if recipe.graph_sha256 != manifest.graph_sha256 {
        return Err("precision recipe graph_sha256 mismatch".into());
    }
    let by_id: BTreeMap<_, _> = manifest
        .projections
        .iter()
        .map(|p| (p.id.as_str(), p))
        .collect();
    let mut seen = BTreeSet::new();
    let mut layers = vec![LayerPrecision::default(); graph.layers.len()];
    for entry in &recipe.projections {
        if !seen.insert(entry.id.as_str()) {
            return Err(format!("duplicate precision projection ID {}", entry.id));
        }
        let projection = by_id
            .get(entry.id.as_str())
            .ok_or_else(|| format!("unknown precision projection ID {}", entry.id))?;
        if (entry.expected_n, entry.expected_k) != (projection.n, projection.k) {
            return Err(format!(
                "precision projection {} shape mismatch: expected n/k {}/{}, graph {}/{}",
                entry.id, entry.expected_n, entry.expected_k, projection.n, projection.k
            ));
        }
        *projection.kind.slot(&mut layers[projection.layer_index]) = entry.precision;
    }
    // Canonicalize ALL projections, including implicit FP16, in stable ID order.
    // Integer scalars and length-delimited JSON avoid ambiguous concatenation.
    let canonical_projections: Vec<_> = by_id
        .values()
        .map(|p| ProjectionPrecision {
            id: p.id.clone(),
            expected_n: p.n,
            expected_k: p.k,
            precision: *p.kind.slot(&mut layers[p.layer_index]),
        })
        .collect();
    let canonical = PrecisionRecipe {
        projections: canonical_projections,
        ..recipe.clone()
    };
    // Do not wrap, extend, reorder, or change this v1 serialization. Existing
    // FP16/INT8 recipe hashes are a compatibility contract independent of G3.
    let canonical_envelope = if uses_mxfp8 {
        let source_f32_projections = mxfp8_sources(graph, &canonical, &by_id)?;
        CanonicalRecipeIdentity::V2(Mxfp8ResolvedIdentity {
            schema: "rustgo-resolved-precision-recipe-v2",
            recipe: canonical,
            mxfp8_semantics: mxfp8_semantics(),
            source_f32_projections,
        })
    } else {
        CanonicalRecipeIdentity::V1(canonical)
    };
    let recipe_sha256 = hash_json(&canonical_envelope)?;
    Ok((
        ResolvedRecipe {
            layers,
            recipe_sha256,
            graph_sha256: manifest.graph_sha256.clone(),
            model_sha256: manifest.model_sha256.clone(),
        },
        canonical_envelope,
    ))
}

/// Public only as the typed v2 export variant; construct via resolution so its
/// canonical projection order, source hashes and semantics cannot diverge.
#[derive(Debug, Serialize)]
pub struct Mxfp8ResolvedIdentity {
    schema: &'static str,
    recipe: PrecisionRecipe,
    mxfp8_semantics: Mxfp8Semantics,
    source_f32_projections: Vec<Mxfp8SourceProjection>,
}

#[derive(Debug, Serialize)]
struct Mxfp8SourceProjection {
    id: String,
    /// FFN dual is gate followed by up: actual GEMM N is twice expected_n.
    gemm_n: usize,
    logical_k: usize,
    source_stride: usize,
    packing: &'static str,
    source_f32_sha256: String,
}

fn mxfp8_sources(
    graph: &LayerGraph,
    canonical: &PrecisionRecipe,
    by_id: &BTreeMap<&str, &ProjectionManifest>,
) -> Result<Vec<Mxfp8SourceProjection>, String> {
    canonical
        .projections
        .iter()
        .filter(|entry| entry.precision == Precision::Mxfp8)
        .map(|entry| {
            let projection = by_id
                .get(entry.id.as_str())
                .ok_or_else(|| format!("unknown MXFP8 source projection {}", entry.id))?;
            let (parts, gemm_n, packing): (Vec<(&str, &Tensor)>, usize, &'static str) =
                match (&graph.layers[projection.layer_index], projection.kind) {
                    (Layer::Ffn(f), ProjectionKind::FfnDual) => (
                        vec![("gate", &f.gate_weight), ("up", &f.up_weight)],
                        projection
                            .n
                            .checked_mul(2)
                            .ok_or("MXFP8 dual width overflow")?,
                        "row-major-gate-then-up-v1",
                    ),
                    (Layer::Ffn(f), ProjectionKind::FfnDown) => {
                        (vec![("down", &f.down_weight)], projection.n, "row-major-v1")
                    }
                    (Layer::Attention(a), ProjectionKind::AttentionQkv) => {
                        (vec![("qkv", &a.qkv_weight)], projection.n, "row-major-v1")
                    }
                    (Layer::Attention(a), ProjectionKind::AttentionOut) => {
                        (vec![("out", &a.out_weight)], projection.n, "row-major-v1")
                    }
                    _ => {
                        return Err(format!(
                            "MXFP8 source projection {} is not supported",
                            entry.id
                        ))
                    }
                };
            let mut digest = Sha256::new();
            // Length-framed domain, ID, shape, part names and exact little-endian
            // IEEE bits disallow concatenation collisions and hidden half casts.
            let mut bytes = |value: &[u8]| {
                digest.update((value.len() as u64).to_le_bytes());
                digest.update(value);
            };
            bytes(b"rustgo-mxfp8-source-f32-v1");
            bytes(entry.id.as_bytes());
            bytes(packing.as_bytes());
            drop(bytes);
            digest.update((gemm_n as u64).to_le_bytes());
            digest.update((projection.k as u64).to_le_bytes());
            digest.update((parts.len() as u64).to_le_bytes());
            for (role, tensor) in parts {
                digest.update((role.len() as u64).to_le_bytes());
                digest.update(role.as_bytes());
                digest.update((tensor.dims.len() as u64).to_le_bytes());
                for dimension in &tensor.dims {
                    digest.update(dimension.to_le_bytes());
                }
                let TensorData::F32(values) = &tensor.data else {
                    return Err(format!("MXFP8 source {} must be original F32", entry.id));
                };
                digest.update((values.len() as u64).to_le_bytes());
                for value in values {
                    if !value.is_finite() {
                        return Err(format!(
                            "MXFP8 source {} contains NaN or infinity",
                            entry.id
                        ));
                    }
                    digest.update(value.to_bits().to_le_bytes());
                }
            }
            Ok(Mxfp8SourceProjection {
                id: entry.id.clone(),
                gemm_n,
                logical_k: projection.k,
                source_stride: projection.k,
                packing,
                source_f32_sha256: hex::encode(digest.finalize()),
            })
        })
        .collect()
}

fn check_hash(name: &str, value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(format!(
            "{name} must be 64 lowercase hexadecimal SHA-256 characters"
        ));
    }
    Ok(())
}

fn hash_json(value: &impl Serialize) -> Result<String, String> {
    let bytes =
        serde_json::to_vec(value).map_err(|e| format!("serialize precision identity: {e}"))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn shape(tensor: &Tensor, dims: &[usize], label: &str) -> Result<(), String> {
    let expected: Vec<_> = dims.iter().map(|&n| n as i64).collect();
    if tensor.dims != expected {
        return Err(format!(
            "{label} has shape {:?}, expected {expected:?}",
            tensor.dims
        ));
    }
    Ok(())
}

/// Strict grammar prevents a missing/surplus pair from silently renumbering a
/// different kind of projection. Graph SHA additionally binds complete order.
fn scan_projections(graph: &LayerGraph) -> Result<Vec<ProjectionManifest>, String> {
    let trunk = graph.trunk_channels;
    let mid = graph.mid_channels;
    if graph.board_size != 19 || mid == 0 || trunk <= mid || graph.num_blocks == 0 {
        return Err("precision recipes require a supported 19x19 nested transformer graph".into());
    }
    let mut cursor = 0;
    match graph.layers.get(cursor) {
        Some(Layer::InitialConv(x)) if x.out_channels == trunk => cursor += 1,
        _ => return Err("precision graph must begin with trunk InitialConv".into()),
    }
    let mut result = Vec::new();
    for block in 0..graph.num_blocks {
        match graph.layers.get(cursor) {
            Some(Layer::Linear(x)) if x.k == trunk && x.n == mid => {
                shape(&x.weight, &[mid, trunk], "block down projection")?;
                cursor += 1;
            }
            _ => {
                return Err(format!(
                    "block {block}: expected trunk-to-mid Linear at layer {cursor}"
                ));
            }
        }
        let mut pairs = 0;
        while matches!(graph.layers.get(cursor), Some(Layer::RmsNorm(_))) {
            let prefix = format!("trunk.block{block:02}.pair{pairs:02}");
            require_rms(graph, cursor, mid)?;
            cursor += 1;
            let Some(Layer::Attention(att)) = graph.layers.get(cursor) else {
                return Err(format!("{prefix}: expected Attention at layer {cursor}"));
            };
            if att.num_heads.checked_mul(att.head_dim) != Some(mid)
                || att.num_heads != graph.num_heads
                || att.head_dim != graph.head_dim
                || att.seq_len != 361
            {
                return Err(format!(
                    "{prefix}: attention shape metadata does not match graph"
                ));
            }
            shape(&att.qkv_weight, &[3 * mid, mid], "attention QKV")?;
            shape(&att.out_weight, &[mid, mid], "attention out")?;
            for (suffix, kind, n) in [
                ("attention.qkv", ProjectionKind::AttentionQkv, 3 * mid),
                ("attention.out", ProjectionKind::AttentionOut, mid),
            ] {
                result.push(ProjectionManifest {
                    id: format!("{prefix}.{suffix}"),
                    layer_index: cursor,
                    n,
                    k: mid,
                    kind,
                });
            }
            cursor += 1;
            require_rms(graph, cursor, mid)?;
            cursor += 1;
            let Some(Layer::Ffn(ffn)) = graph.layers.get(cursor) else {
                return Err(format!("{prefix}: expected Ffn at layer {cursor}"));
            };
            if ffn.hidden == 0 {
                return Err(format!("{prefix}: zero FFN width"));
            }
            shape(&ffn.up_weight, &[ffn.hidden, mid], "FFN up")?;
            shape(&ffn.gate_weight, &[ffn.hidden, mid], "FFN gate")?;
            shape(&ffn.down_weight, &[mid, ffn.hidden], "FFN down")?;
            result.push(ProjectionManifest {
                id: format!("{prefix}.ffn.dual"),
                layer_index: cursor,
                n: ffn.hidden,
                k: mid,
                kind: ProjectionKind::FfnDual,
            });
            result.push(ProjectionManifest {
                id: format!("{prefix}.ffn.down"),
                layer_index: cursor,
                n: mid,
                k: ffn.hidden,
                kind: ProjectionKind::FfnDown,
            });
            cursor += 1;
            pairs += 1;
        }
        if pairs == 0 {
            return Err(format!("block {block} contains no attention/FFN pairs"));
        }
        require_gate(graph, cursor, mid)?;
        cursor += 1;
        match graph.layers.get(cursor) {
            Some(Layer::Linear(x)) if x.k == mid && x.n == trunk => {
                shape(&x.weight, &[trunk, mid], "block up projection")?;
                cursor += 1;
            }
            _ => {
                return Err(format!(
                    "block {block}: expected mid-to-trunk Linear at layer {cursor}"
                ));
            }
        }
        if block + 1 < graph.num_blocks {
            require_gate(graph, cursor, trunk)?;
            cursor += 1;
        }
    }
    if !matches!(graph.layers.get(cursor), Some(Layer::TrunkFinal(x)) if x.channels == trunk)
        || !matches!(graph.layers.get(cursor + 1), Some(Layer::PolicyHead(_)))
        || !matches!(graph.layers.get(cursor + 2), Some(Layer::ValueHead(_)))
        || graph.layers.len() != cursor + 3
    {
        return Err(format!(
            "precision graph expects exactly TrunkFinal/PolicyHead/ValueHead after layer {cursor}"
        ));
    }
    Ok(result)
}

fn require_rms(graph: &LayerGraph, index: usize, mid: usize) -> Result<(), String> {
    match graph.layers.get(index) {
        Some(Layer::RmsNorm(x)) if x.channels == mid => shape(&x.scale, &[mid], "RMS scale"),
        _ => Err(format!("expected mid-channel RmsNorm at layer {index}")),
    }
}

fn require_gate(graph: &LayerGraph, index: usize, channels: usize) -> Result<(), String> {
    match graph.layers.get(index) {
        Some(Layer::GateSilu(x)) if x.channels == channels => Ok(()),
        _ => Err(format!(
            "expected {channels}-channel GateSilu at layer {index}"
        )),
    }
}

// Exhaustive destructuring deliberately catches newly introduced graph/layer
// fields at compile time: a semantics field must never escape the identity.
// Tensor contents derive from model_sha256; all tensor shapes, types and lengths
// are included. Scalar values use IEEE bits (including signed zero), not lossy
// decimal formatting. Non-finite semantic scalars are rejected.
fn graph_description(graph: &LayerGraph) -> Result<Value, String> {
    let LayerGraph {
        layers,
        num_spatial_inputs,
        num_global_inputs,
        board_size,
        trunk_channels,
        mid_channels,
        num_blocks,
        num_heads,
        head_dim,
        total_params,
        scalar_params,
        input_names,
        output_names,
    } = graph;
    let layers = layers
        .iter()
        .map(layer_description)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({ "schema": GRAPH_SCHEMA, "layers": layers,
        "num_spatial_inputs": num_spatial_inputs, "num_global_inputs": num_global_inputs,
        "board_size": board_size, "trunk_channels": trunk_channels, "mid_channels": mid_channels,
        "num_blocks": num_blocks, "num_heads": num_heads, "head_dim": head_dim,
        "total_params": total_params, "scalar_params": scalar_params,
        "input_names": input_names, "output_names": output_names }))
}

fn tensor_description(t: &Tensor) -> Result<Value, String> {
    let Tensor { dims, data } = t;
    let mut elements = 1usize;
    for &dimension in dims {
        let dimension = usize::try_from(dimension)
            .ok()
            .filter(|&n| n > 0)
            .ok_or_else(|| format!("precision graph tensor has invalid static shape {dims:?}"))?;
        elements = elements
            .checked_mul(dimension)
            .ok_or_else(|| "precision graph tensor shape overflow".to_owned())?;
    }
    let (dtype, len) = match data {
        TensorData::F32(values) => ("f32", values.len()),
        TensorData::I64(values) => ("i64", values.len()),
    };
    if len != elements || dtype != "f32" {
        return Err(format!(
            "precision graph tensor {dims:?} has invalid {dtype} storage length {len} (expected {elements} f32)"
        ));
    }
    Ok(json!({ "dims": dims, "dtype": dtype, "elements": len }))
}

fn finite_bits(value: f32) -> Result<u32, String> {
    if !value.is_finite() {
        return Err("precision graph contains non-finite semantic scalar".into());
    }
    Ok(value.to_bits())
}

fn layer_description(layer: &Layer) -> Result<Value, String> {
    use crate::onnx_parser::*;
    let t = tensor_description;
    let f = finite_bits;
    Ok(match layer {
        Layer::InitialConv(InitialConvLayer {
            weight,
            global_weight,
            gate_scale,
            gate_bias,
            out_channels,
        }) => {
            json!({ "kind": "initial_conv", "weight": t(weight)?, "global_weight": t(global_weight)?,
                "gate_scale": t(gate_scale)?, "gate_bias": t(gate_bias)?, "out_channels": out_channels })
        }
        Layer::Linear(MatMulLayer {
            weight,
            bias,
            n,
            k,
            act,
            residual_add,
        }) => {
            let activation = act.map(|a| match a {
                Act::Silu => "silu",
            });
            json!({ "kind": "linear", "weight": t(weight)?, "bias": bias.as_ref().map(t).transpose()?,
                "n": n, "k": k, "act": activation, "residual_add": residual_add })
        }
        Layer::RmsNorm(RmsNormLayer {
            scale,
            channels,
            eps,
        }) => {
            json!({ "kind": "rms_norm", "scale": t(scale)?, "channels": channels, "eps_bits": f(*eps)? })
        }
        Layer::Attention(AttentionLayer {
            num_heads,
            head_dim,
            seq_len,
            qkv_weight,
            out_weight,
            rope_cos,
            rope_sin,
            qk_scale,
            residual_add,
        }) => {
            json!({ "kind": "attention", "num_heads": num_heads, "head_dim": head_dim, "seq_len": seq_len,
                "qkv_weight": t(qkv_weight)?, "out_weight": t(out_weight)?, "rope_cos": t(rope_cos)?,
                "rope_sin": t(rope_sin)?, "qk_scale_bits": f(*qk_scale)?, "residual_add": residual_add })
        }
        Layer::Ffn(FfnLayer {
            up_weight,
            gate_weight,
            down_weight,
            hidden,
            residual_add,
        }) => json!({ "kind": "ffn", "up_weight": t(up_weight)?, "gate_weight": t(gate_weight)?,
                "down_weight": t(down_weight)?, "hidden": hidden, "residual_add": residual_add }),
        Layer::GateSilu(GateSiluLayer {
            scale,
            bias,
            channels,
        }) => {
            json!({ "kind": "gate_silu", "scale": t(scale)?, "bias": t(bias)?, "channels": channels })
        }
        Layer::TrunkFinal(TrunkFinalLayer {
            mean,
            std,
            gamma,
            beta,
            channels,
        }) => json!({ "kind": "trunk_final", "mean": t(mean)?, "std": t(std)?, "gamma": t(gamma)?,
                "beta": t(beta)?, "channels": channels }),
        Layer::PolicyHead(PolicyHeadLayer {
            conv1p_weight,
            conv1g_weight,
            g_bias,
            g_matmul,
            pass_matmul1,
            pass_bias1,
            pass_matmul2,
            bias2,
            conv2p_weight,
            act_silu,
            mask_scale,
        }) => {
            json!({ "kind": "policy_head", "conv1p_weight": t(conv1p_weight)?, "conv1g_weight": t(conv1g_weight)?,
                "g_bias": t(g_bias)?, "g_matmul": t(g_matmul)?, "pass_matmul1": t(pass_matmul1)?,
                "pass_bias1": t(pass_bias1)?, "pass_matmul2": t(pass_matmul2)?, "bias2": t(bias2)?,
                "conv2p_weight": t(conv2p_weight)?, "act_silu": act_silu, "mask_scale_bits": f(*mask_scale)? })
        }
        Layer::ValueHead(ValueHeadLayer {
            conv1_weight,
            bias1,
            linear2_weight,
            linear2_bias,
            value_matmul,
            value_bias,
            misc_matmul,
            misc_bias,
            moremisc_matmul,
            moremisc_bias,
            ownership_conv,
            act_silu,
            mask_scale,
            mask_quad,
        }) => json!({ "kind": "value_head", "conv1_weight": t(conv1_weight)?, "bias1": t(bias1)?,
                "linear2_weight": t(linear2_weight)?, "linear2_bias": t(linear2_bias)?,
                "value_matmul": t(value_matmul)?, "value_bias": t(value_bias)?, "misc_matmul": t(misc_matmul)?,
                "misc_bias": t(misc_bias)?, "moremisc_matmul": t(moremisc_matmul)?, "moremisc_bias": t(moremisc_bias)?,
                "ownership_conv": t(ownership_conv)?, "act_silu": act_silu,
                "mask_scale_bits": f(*mask_scale)?, "mask_quad_bits": f(*mask_quad)? }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx_parser::*;

    const MODEL: &str = "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef";

    fn tensor(dims: &[usize]) -> Tensor {
        Tensor {
            dims: dims.iter().map(|&n| n as i64).collect(),
            data: TensorData::F32(vec![0.; dims.iter().product()]),
        }
    }

    fn fixture() -> LayerGraph {
        let (trunk, mid) = (64, 32);
        let linear = |n, k| {
            Layer::Linear(MatMulLayer {
                weight: tensor(&[n, k]),
                bias: None,
                n,
                k,
                act: None,
                residual_add: false,
            })
        };
        let gate = |channels| {
            Layer::GateSilu(GateSiluLayer {
                scale: tensor(&[channels]),
                bias: tensor(&[channels]),
                channels,
            })
        };
        let mut layers = vec![Layer::InitialConv(InitialConvLayer {
            weight: tensor(&[trunk, 22, 3, 3]),
            global_weight: tensor(&[trunk, 19]),
            gate_scale: tensor(&[trunk]),
            gate_bias: tensor(&[trunk]),
            out_channels: trunk,
        })];
        // Different pair counts/hidden widths catch assumptions about 3 pairs
        // per block or uniform unpruned FFNs. This is intentionally CPU-only.
        for block in 0..2 {
            layers.push(linear(mid, trunk));
            for pair in 0..=block {
                let hidden = 8 * (pair + 1);
                layers.push(Layer::RmsNorm(RmsNormLayer {
                    scale: tensor(&[mid]),
                    channels: mid,
                    eps: 1e-5,
                }));
                layers.push(Layer::Attention(AttentionLayer {
                    num_heads: 1,
                    head_dim: 32,
                    seq_len: 361,
                    qkv_weight: tensor(&[3 * mid, mid]),
                    out_weight: tensor(&[mid, mid]),
                    rope_cos: tensor(&[361, mid / 2]),
                    rope_sin: tensor(&[361, mid / 2]),
                    qk_scale: 0.4204482,
                    residual_add: true,
                }));
                layers.push(Layer::RmsNorm(RmsNormLayer {
                    scale: tensor(&[mid]),
                    channels: mid,
                    eps: 1e-5,
                }));
                layers.push(Layer::Ffn(FfnLayer {
                    up_weight: tensor(&[hidden, mid]),
                    gate_weight: tensor(&[hidden, mid]),
                    down_weight: tensor(&[mid, hidden]),
                    hidden,
                    residual_add: true,
                }));
            }
            layers.push(gate(mid));
            layers.push(linear(trunk, mid));
            if block == 0 {
                layers.push(gate(trunk));
            }
        }
        layers.push(Layer::TrunkFinal(TrunkFinalLayer {
            mean: tensor(&[trunk]),
            std: tensor(&[trunk]),
            gamma: tensor(&[trunk]),
            beta: tensor(&[trunk]),
            channels: trunk,
        }));
        layers.push(Layer::PolicyHead(PolicyHeadLayer {
            conv1p_weight: tensor(&[96, trunk]),
            conv1g_weight: tensor(&[96, trunk]),
            g_bias: tensor(&[96]),
            g_matmul: tensor(&[96, 288]),
            pass_matmul1: tensor(&[96, 288]),
            pass_bias1: tensor(&[96]),
            pass_matmul2: tensor(&[6, 96]),
            bias2: tensor(&[96]),
            conv2p_weight: tensor(&[6, 96]),
            act_silu: true,
            mask_scale: 0.5,
        }));
        layers.push(Layer::ValueHead(ValueHeadLayer {
            conv1_weight: tensor(&[192, trunk]),
            bias1: tensor(&[192]),
            linear2_weight: tensor(&[192, 576]),
            linear2_bias: tensor(&[192]),
            value_matmul: tensor(&[3, 192]),
            value_bias: tensor(&[3]),
            misc_matmul: tensor(&[10, 192]),
            misc_bias: tensor(&[10]),
            moremisc_matmul: tensor(&[8, 192]),
            moremisc_bias: tensor(&[8]),
            ownership_conv: tensor(&[1, 192]),
            act_silu: true,
            mask_scale: 0.5,
            mask_quad: 0.15,
        }));
        let mut graph = LayerGraph {
            layers,
            num_spatial_inputs: 22,
            num_global_inputs: 19,
            board_size: 19,
            trunk_channels: trunk,
            mid_channels: mid,
            num_blocks: 2,
            num_heads: 1,
            head_dim: 32,
            total_params: 0,
            scalar_params: 0,
            input_names: vec!["spatial".into(), "global".into()],
            output_names: vec![
                "policy".into(),
                "value".into(),
                "misc".into(),
                "moremisc".into(),
                "ownership".into(),
            ],
        };
        graph.total_params = graph.layer_param_elts();
        graph
    }

    #[test]
    fn stable_structural_ids_resolve_independent_projections() {
        let graph = fixture();
        let manifest = graph_manifest(&graph, MODEL).unwrap();
        assert_eq!(manifest.projections.len(), 12);
        let mut recipe = template_recipe(&graph, MODEL).unwrap();
        for entry in &mut recipe.projections {
            if entry.id == "trunk.block01.pair01.ffn.down"
                || entry.id == "trunk.block00.pair00.attention.qkv"
            {
                entry.precision = Precision::Int8;
            }
        }
        let resolved = resolve_recipe(&recipe, &graph, MODEL).unwrap();
        assert_eq!(resolved.layers.len(), graph.layers.len());
        assert_eq!(resolved.layers.iter().filter(|l| l.any_int8()).count(), 2);
        for projection in &manifest.projections {
            let mut layer = resolved.layers[projection.layer_index];
            let precision = *projection.kind.slot(&mut layer);
            assert_eq!(
                precision,
                recipe
                    .projections
                    .iter()
                    .find(|p| p.id == projection.id)
                    .unwrap()
                    .precision
            );
        }
        let down = manifest
            .projections
            .iter()
            .find(|p| p.id == "trunk.block01.pair01.ffn.down")
            .unwrap();
        assert_eq!((down.n, down.k), (32, 16));
    }

    #[test]
    fn canonical_identity_binds_precision_model_and_not_json_format() {
        let graph = fixture();
        let baseline = fp16_recipe(&graph, MODEL).unwrap();
        let mut recipe = template_recipe(&graph, MODEL).unwrap();
        assert_eq!(
            resolve_recipe(&recipe, &graph, MODEL)
                .unwrap()
                .recipe_sha256,
            baseline.recipe_sha256
        );
        recipe.projections.reverse();
        recipe.projections.truncate(2);
        assert_eq!(
            resolve_recipe(&recipe, &graph, MODEL)
                .unwrap()
                .recipe_sha256,
            baseline.recipe_sha256
        );
        recipe.projections[0].precision = Precision::Int8;
        assert_ne!(
            resolve_recipe(&recipe, &graph, MODEL)
                .unwrap()
                .recipe_sha256,
            baseline.recipe_sha256
        );
        assert_ne!(
            fp16_recipe(&graph, &"f".repeat(64)).unwrap().recipe_sha256,
            baseline.recipe_sha256
        );
        let pretty = serde_json::to_string_pretty(&recipe).unwrap();
        let decoded: PrecisionRecipe = serde_json::from_str(&pretty).unwrap();
        assert_eq!(
            resolve_recipe(&recipe, &graph, MODEL)
                .unwrap()
                .recipe_sha256,
            resolve_recipe(&decoded, &graph, MODEL)
                .unwrap()
                .recipe_sha256
        );
    }

    #[test]
    fn resolved_identity_rejects_public_field_mutation() {
        let graph = fixture();
        let baseline = fp16_recipe(&graph, MODEL).unwrap();
        validate_resolved_recipe(&graph, &baseline).unwrap();
        let changes: Vec<Box<dyn Fn(&mut ResolvedRecipe)>> = vec![
            // Legitimate projection, stale canonical identity.
            Box::new(|r| r.layers[5].ffn_down = Precision::Int8),
            Box::new(|r| r.layers[3].attention_qkv = Precision::Int8),
            // Slots that are not represented by this operator's projections.
            Box::new(|r| r.layers[5].attention_qkv = Precision::Int8),
            Box::new(|r| r.layers[0].ffn_dual = Precision::Int8),
            Box::new(|r| {
                r.layers.pop();
            }),
            Box::new(|r| r.model_sha256 = "f".repeat(64)),
            Box::new(|r| r.model_sha256 = "invalid".into()),
            Box::new(|r| r.graph_sha256 = "f".repeat(64)),
            Box::new(|r| r.graph_sha256 = "invalid".into()),
            Box::new(|r| r.recipe_sha256 = "f".repeat(64)),
            Box::new(|r| r.recipe_sha256 = "invalid".into()),
        ];
        for change in changes {
            let mut changed = baseline.clone();
            change(&mut changed);
            assert!(
                validate_resolved_recipe(&graph, &changed).is_err(),
                "accepted {changed:?}"
            );
        }
        // A changed selection is valid only with the corresponding new identity.
        let mut source = template_recipe(&graph, MODEL).unwrap();
        source.projections[0].precision = Precision::Int8;
        let mixed = resolve_recipe(&source, &graph, MODEL).unwrap();
        validate_resolved_recipe(&graph, &mixed).unwrap();
        let mut stale = mixed.clone();
        stale.layers[3].attention_qkv = Precision::Fp16;
        assert!(validate_resolved_recipe(&graph, &stale).is_err());
    }

    #[test]
    fn rejects_unknown_duplicate_and_unsupported_json_fields() {
        let recipe = template_recipe(&fixture(), MODEL).unwrap();
        let text = serde_json::to_string(&recipe).unwrap();
        for invalid in [
            text.replacen('{', "{\"extra\":0,", 1),
            text.replacen('{', "{\"version\":1,", 1),
            text.replacen("\"expected_n\":", "\"index\":0,\"expected_n\":", 1),
            text.replacen("\"expected_n\":", "\"expected_n\":96,\"expected_n\":", 1),
            text.replacen("\"fp16\"", "\"fp8\"", 1),
            text.replacen("\"fp16\"", "\"nvfp4\"", 1),
        ] {
            assert!(
                serde_json::from_str::<PrecisionRecipe>(&invalid).is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn rejects_mismatched_identity_shape_and_projection() {
        let graph = fixture();
        let recipe = template_recipe(&graph, MODEL).unwrap();
        let checks: Vec<Box<dyn Fn(&mut PrecisionRecipe)>> = vec![
            Box::new(|r| r.schema.push_str("-future")),
            Box::new(|r| r.version += 1),
            Box::new(|r| r.quantization_semantics_version += 1),
            Box::new(|r| r.model_sha256 = "f".repeat(64)),
            Box::new(|r| r.model_sha256 = "A".repeat(64)),
            Box::new(|r| r.graph_sha256 = "0".repeat(64)),
            Box::new(|r| r.graph_sha256 = "not-a-hash".into()),
            Box::new(|r| r.projections[0].expected_n += 1),
            Box::new(|r| r.projections[0].expected_k += 1),
            Box::new(|r| r.projections[0].id = "trunk.block00.pair00.attention.down".into()),
            Box::new(|r| r.projections[0].id = "3".into()),
            Box::new(|r| r.projections.push(r.projections[0].clone())),
        ];
        for change in checks {
            let mut invalid = recipe.clone();
            change(&mut invalid);
            assert!(
                resolve_recipe(&invalid, &graph, MODEL).is_err(),
                "accepted {invalid:?}"
            );
        }
    }

    #[test]
    fn graph_identity_covers_semantic_scalars_order_and_tensor_metadata() {
        let graph = fixture();
        let original = template_recipe(&graph, MODEL).unwrap();
        let changes: Vec<Box<dyn Fn(&mut LayerGraph)>> = vec![
            Box::new(|g| {
                if let Layer::RmsNorm(x) = &mut g.layers[2] {
                    x.eps *= 2.;
                }
            }),
            Box::new(|g| {
                if let Layer::Attention(x) = &mut g.layers[3] {
                    x.qk_scale *= 2.;
                }
            }),
            Box::new(|g| {
                if let Layer::Attention(x) = &mut g.layers[3] {
                    x.residual_add = false;
                }
            }),
            Box::new(|g| {
                if let Layer::Ffn(x) = &mut g.layers[5] {
                    x.residual_add = false;
                }
            }),
            Box::new(|g| {
                if let Layer::Linear(x) = &mut g.layers[1] {
                    x.act = Some(Act::Silu);
                }
            }),
            Box::new(|g| {
                if let Layer::Linear(x) = &mut g.layers[1] {
                    x.bias = Some(tensor(&[32]));
                }
            }),
            Box::new(|g| {
                for l in &mut g.layers {
                    if let Layer::PolicyHead(x) = l {
                        x.mask_scale += 0.01;
                    }
                }
            }),
            Box::new(|g| {
                for l in &mut g.layers {
                    if let Layer::ValueHead(x) = l {
                        x.mask_quad += 0.01;
                    }
                }
            }),
            Box::new(|g| {
                for l in &mut g.layers {
                    if let Layer::ValueHead(x) = l {
                        x.act_silu = false;
                    }
                }
            }),
            Box::new(|g| g.layers.swap(3, 5)),
            Box::new(|g| {
                if let Layer::Attention(x) = &mut g.layers[3] {
                    x.rope_cos.dims = vec![5776];
                }
            }),
            Box::new(|g| g.input_names.reverse()),
            Box::new(|g| g.output_names.reverse()),
            Box::new(|g| g.scalar_params += 1),
        ];
        for change in changes {
            let mut changed = fixture();
            change(&mut changed);
            assert!(resolve_recipe(&original, &changed, MODEL).is_err());
        }
    }

    #[test]
    fn rejects_malformed_graph_before_assigning_ids() {
        let changes: Vec<Box<dyn Fn(&mut LayerGraph)>> = vec![
            Box::new(|g| g.num_blocks += 1),
            Box::new(|g| g.layers.insert(5, g.layers[5].clone())),
            Box::new(|g| {
                g.layers.remove(4);
            }),
            Box::new(|g| {
                g.layers.pop();
            }),
            Box::new(|g| {
                if let Layer::Ffn(x) = &mut g.layers[5] {
                    x.gate_weight = tensor(&[7, 32]);
                }
            }),
            Box::new(|g| {
                if let Layer::RmsNorm(x) = &mut g.layers[2] {
                    x.eps = f32::NAN;
                }
            }),
            Box::new(|g| {
                if let Layer::RmsNorm(x) = &mut g.layers[2] {
                    x.scale.data = TensorData::I64(vec![0; 32]);
                }
            }),
            Box::new(|g| {
                if let Layer::Attention(x) = &mut g.layers[3] {
                    x.out_weight.data = TensorData::F32(vec![0.]);
                }
            }),
        ];
        for change in changes {
            let mut invalid = fixture();
            change(&mut invalid);
            assert!(fp16_recipe(&invalid, MODEL).is_err());
        }
    }

    #[test]
    fn file_load_is_fail_closed() {
        let graph = fixture();
        let recipe = template_recipe(&graph, MODEL).unwrap();
        let path = std::env::temp_dir().join(format!(
            "rustgo-quantization-plan-test-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, serde_json::to_vec_pretty(&recipe).unwrap()).unwrap();
        assert_eq!(
            load_and_resolve(&path, &graph, MODEL)
                .unwrap()
                .recipe_sha256,
            fp16_recipe(&graph, MODEL).unwrap().recipe_sha256
        );
        std::fs::write(&path, b"{\"schema\": \"truncated").unwrap();
        assert!(load_and_resolve(&path, &graph, MODEL).is_err());
        std::fs::remove_file(&path).unwrap();
        assert!(load_and_resolve(&path, &graph, MODEL).is_err());
    }

    #[test]
    fn actual_b11_b15_v1_recipe_sha_goldens_remain_unchanged() {
        // Snapshots of quant-inspect manifests and validated Rust worker traces.
        // No model files/CUDA are required. Graph/model digests remain the exact
        // recorded source identities; expected_n for dual is one gate/up width.
        let cases: Vec<(usize, Vec<usize>, &str, &str, &str, &str)> = vec![
            (
                384,
                vec![1152; 33],
                "1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6",
                "4c26313631d8795cecc72471d5612280515235ec1490104f0aca5f4f43613d5f",
                "60840fb46738825d4f6ae931bb7816ab158ab94e533548540d890a954145d748",
                "21d6e318945f0f821042e6735fe56a16c5e227da006a59aa86277c2ed3115d65",
            ),
            (
                512,
                vec![
                    240, 48, 472, 72, 16, 16, 48, 48, 568, 16, 64, 72, 960, 368, 392, 48, 104, 88,
                    16, 24, 40, 48, 160, 184, 304, 400, 656, 184, 256, 288, 1088, 992, 1160, 40,
                    96, 88, 112, 96, 24, 264, 296, 304, 424, 760, 728,
                ],
                "3f216ee88226ee49ca826eaa5f7e1e9982ff48a96625914bfc4e5ad20ee095a7",
                "6f0bb7abf426bf331cee6007490d221acc6f0b3b150d22a87c952d1777577ea1",
                "f1873567d659b3ccf30aadf7ae6c9c6615d1af4801c6cc334a82bfe1d5564d29",
                "1e6b21363131a191eb08e40499119f1470725bffbb22600010eea7b0f8c46487",
            ),
        ];
        for (mid, widths, model, graph, fp16_sha, int8_sha) in cases {
            for (int8, expected) in [(false, fp16_sha), (true, int8_sha)] {
                let mut projections = Vec::new();
                for (i, hidden) in widths.iter().copied().enumerate() {
                    for (suffix, n, k) in [
                        ("attention.out", mid, mid),
                        ("attention.qkv", 3 * mid, mid),
                        ("ffn.down", mid, hidden),
                        ("ffn.dual", hidden, mid),
                    ] {
                        projections.push(ProjectionPrecision {
                            id: format!("trunk.block{:02}.pair{:02}.{suffix}", i / 3, i % 3),
                            expected_n: n,
                            expected_k: k,
                            precision: if int8 && suffix.starts_with("ffn.") {
                                Precision::Int8
                            } else {
                                Precision::Fp16
                            },
                        });
                    }
                }
                let snapshot = PrecisionRecipe {
                    schema: RECIPE_SCHEMA.into(),
                    version: RECIPE_VERSION,
                    quantization_semantics_version: QUANTIZATION_SEMANTICS_VERSION,
                    model_sha256: model.into(),
                    graph_sha256: graph.into(),
                    projections,
                };
                assert_eq!(
                    hash_json(&CanonicalRecipeIdentity::V1(snapshot)).unwrap(),
                    expected
                );
            }
        }
    }

    #[test]
    fn mxfp8_requires_explicit_v2_and_preserves_implicit_fp16() {
        let graph = fixture();
        let mut recipe = mxfp8_recipe(
            &graph,
            MODEL,
            &[
                "trunk.block00.pair00.ffn.dual",
                "trunk.block01.pair01.attention.qkv",
            ],
        )
        .unwrap();
        assert_eq!(
            (recipe.version, recipe.quantization_semantics_version),
            (2, 2)
        );
        recipe
            .projections
            .iter_mut()
            .find(|p| p.id == "trunk.block00.pair00.ffn.down")
            .unwrap()
            .precision = Precision::Int8;
        let resolved = resolve_recipe(&recipe, &graph, MODEL).unwrap();
        assert_eq!(resolved.layers.iter().filter(|p| p.any_int8()).count(), 1);
        assert_eq!(resolved.layers.iter().filter(|p| p.any_mxfp8()).count(), 2);
        assert_eq!(
            resolved.layers.iter().filter(|p| p.any_quantized()).count(),
            2
        );
        validate_resolved_recipe(&graph, &resolved).unwrap();
        recipe
            .projections
            .retain(|p| p.precision != Precision::Fp16);
        recipe.projections.reverse();
        let decoded: PrecisionRecipe =
            serde_json::from_str(&serde_json::to_string_pretty(&recipe).unwrap()).unwrap();
        assert_eq!(
            resolve_recipe(&decoded, &graph, MODEL)
                .unwrap()
                .recipe_sha256,
            resolved.recipe_sha256
        );
        assert_eq!(resolved.layers[3].attention_qkv, Precision::Fp16);
        assert_eq!(resolved.layers[5].attention_out, Precision::Fp16);
        for (version, semantics) in [(1, 1), (1, 2), (2, 1), (3, 2), (2, 3)] {
            let mut invalid = recipe.clone();
            invalid.version = version;
            invalid.quantization_semantics_version = semantics;
            assert!(resolve_recipe(&invalid, &graph, MODEL).is_err());
        }
        let mut no_mxfp8 = template_recipe(&graph, MODEL).unwrap();
        no_mxfp8.version = 2;
        no_mxfp8.quantization_semantics_version = 2;
        assert!(resolve_recipe(&no_mxfp8, &graph, MODEL).is_err());
        no_mxfp8.projections[0].precision = Precision::Int8;
        assert!(resolve_recipe(&no_mxfp8, &graph, MODEL).is_err());
        assert!(mxfp8_recipe(&graph, MODEL, &[]).is_err());
        assert!(mxfp8_recipe(&graph, MODEL, &["5"]).is_err());
        assert!(mxfp8_recipe(
            &graph,
            MODEL,
            &[
                "trunk.block00.pair00.ffn.dual",
                "trunk.block00.pair00.ffn.dual"
            ]
        )
        .is_err());
    }

    #[test]
    fn mxfp8_identity_binds_original_f32_bits_and_packing() {
        let source_graph = || {
            let mut graph = fixture();
            if let Layer::Ffn(f) = &mut graph.layers[5] {
                if let TensorData::F32(values) = &mut f.gate_weight.data {
                    values[0] = f32::from_bits(1.0625f32.to_bits() + 1);
                }
            }
            graph
        };
        let graph = source_graph();
        let recipe = mxfp8_recipe(&graph, MODEL, &["trunk.block00.pair00.ffn.dual"]).unwrap();
        let resolved = resolve_recipe(&recipe, &graph, MODEL).unwrap();
        let original_graph_sha = graph_sha256(&graph).unwrap();
        let changes: Vec<Box<dyn Fn(&mut LayerGraph)>> = vec![
            // Early half rounding loses this FP32 bit before E4M3 ties-even.
            Box::new(|g| {
                if let Layer::Ffn(f) = &mut g.layers[5] {
                    if let TensorData::F32(values) = &mut f.gate_weight.data {
                        values[0] = 1.0625;
                    }
                }
            }),
            Box::new(|g| {
                if let Layer::Ffn(f) = &mut g.layers[5] {
                    if let TensorData::F32(values) = &mut f.up_weight.data {
                        values[0] = 0.25;
                    }
                }
            }),
            Box::new(|g| {
                if let Layer::Ffn(f) = &mut g.layers[5] {
                    std::mem::swap(&mut f.gate_weight, &mut f.up_weight);
                }
            }),
        ];
        for change in changes {
            let mut changed = source_graph();
            change(&mut changed);
            assert_eq!(
                graph_sha256(&changed).unwrap(),
                original_graph_sha,
                "do not change old structural graph identity"
            );
            assert_ne!(
                resolve_recipe(&recipe, &changed, MODEL)
                    .unwrap()
                    .recipe_sha256,
                resolved.recipe_sha256
            );
            assert!(validate_resolved_recipe(&changed, &resolved).is_err());
        }
        for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut invalid = source_graph();
            if let Layer::Ffn(f) = &mut invalid.layers[5] {
                if let TensorData::F32(values) = &mut f.gate_weight.data {
                    values[0] = nonfinite;
                }
            }
            assert!(resolve_recipe(&recipe, &invalid, MODEL).is_err());
        }
        let mut changed = resolved.clone();
        changed.layers[5].ffn_dual = Precision::Fp16;
        assert!(validate_resolved_recipe(&graph, &changed).is_err());
        let mut changed = resolved.clone();
        changed.layers[0].ffn_down = Precision::Mxfp8;
        assert!(validate_resolved_recipe(&graph, &changed).is_err());
    }

    #[test]
    fn mxfp8_canonical_envelope_contains_versioned_contract_and_source_hashes() {
        let graph = fixture();
        let recipe = mxfp8_recipe(
            &graph,
            MODEL,
            &[
                "trunk.block00.pair00.ffn.dual",
                "trunk.block00.pair00.attention.out",
            ],
        )
        .unwrap();
        let manifest = graph_manifest(&graph, MODEL).unwrap();
        let by_id = manifest
            .projections
            .iter()
            .map(|p| (p.id.as_str(), p))
            .collect();
        let sources = mxfp8_sources(&graph, &recipe, &by_id).unwrap();
        let dual = sources
            .iter()
            .find(|source| source.id.ends_with("ffn.dual"))
            .unwrap();
        assert_eq!(
            (dual.gemm_n, dual.logical_k, dual.source_stride),
            (16, 32, 32)
        );
        assert_eq!(dual.packing, "row-major-gate-then-up-v1");
        for source in &sources {
            check_hash("source", &source.source_f32_sha256).unwrap();
        }
        let semantics = mxfp8_semantics();
        assert_eq!(semantics.revision, 1);
        assert_eq!(semantics.quantization_version, MXFP8_QUANTIZATION_VERSION);
        assert_eq!(semantics.block_size, 32);
        assert_eq!(semantics.compute, "fp32-accumulation");
        assert!(!semantics.fast_accum);
        let envelope = Mxfp8ResolvedIdentity {
            schema: "rustgo-resolved-precision-recipe-v2",
            recipe: recipe.clone(),
            mxfp8_semantics: semantics,
            source_f32_projections: sources,
        };
        let first = hash_json(&envelope).unwrap();
        let mut changed = envelope;
        changed.mxfp8_semantics.fast_accum = true;
        assert_ne!(hash_json(&changed).unwrap(), first);
        // The wire format has no adjustable semantics fields: v2 maps to this
        // exact contract, and changing it requires a new supported version.
        let text = serde_json::to_string(&recipe).unwrap();
        assert!(serde_json::from_str::<PrecisionRecipe>(&text.replacen(
            '{',
            "{\"mxfp8_semantics\":{},",
            1
        ))
        .is_err());
        assert!(serde_json::from_str::<PrecisionRecipe>(&text.replacen(
            "\"mxfp8\"",
            "\"mxfp8-v2\"",
            1
        ))
        .is_err());
    }

    #[test]
    fn exported_identity_reuses_canonical_loader_bytes_and_completes_fp16() {
        let graph = fixture();
        let projection_count = graph_manifest(&graph, MODEL).unwrap().projections.len();
        for precision in [Precision::Fp16, Precision::Int8, Precision::Mxfp8] {
            let mut recipe = if precision == Precision::Mxfp8 {
                mxfp8_recipe(&graph, MODEL, &["trunk.block00.pair00.ffn.dual"]).unwrap()
            } else {
                template_recipe(&graph, MODEL).unwrap()
            };
            if precision == Precision::Int8 {
                recipe.projections[0].precision = Precision::Int8;
            }
            let resolved = resolve_recipe(&recipe, &graph, MODEL).unwrap();
            let full_source = serde_json::to_vec(&recipe).unwrap();
            let full = resolve_recipe_identity(&full_source, &graph, MODEL).unwrap();
            recipe
                .projections
                .retain(|p| p.precision != Precision::Fp16);
            recipe.projections.reverse();
            let sparse_source = serde_json::to_vec_pretty(&recipe).unwrap();
            let sparse = resolve_recipe_identity(&sparse_source, &graph, MODEL).unwrap();
            assert_ne!(
                full.source_recipe_file_sha256,
                sparse.source_recipe_file_sha256
            );
            assert_eq!(full.recipe_sha256, sparse.recipe_sha256);
            assert_eq!(sparse.recipe_sha256, resolved.recipe_sha256);
            assert_eq!(sparse.model_sha256, MODEL);
            assert_eq!(sparse.graph_sha256, resolved.graph_sha256);
            assert_eq!(sparse.schema, "rustgo-recipe-identity-export");
            assert_eq!(sparse.version, 1);
            assert_eq!(sparse.source_recipe_file, "source-recipe.json");
            assert_eq!(sparse.canonical_encoding, "serde-json-compact-ordered-v1");
            assert_eq!(
                sparse.source_recipe_file_sha256,
                hex::encode(Sha256::digest(&sparse_source))
            );
            let canonical = serde_json::to_vec(&sparse.canonical_envelope).unwrap();
            assert_eq!(
                hex::encode(Sha256::digest(&canonical)),
                sparse.recipe_sha256
            );
            let canonical_recipe = match &sparse.canonical_envelope {
                CanonicalRecipeIdentity::V1(recipe) => {
                    assert_ne!(precision, Precision::Mxfp8);
                    recipe
                }
                CanonicalRecipeIdentity::V2(envelope) => {
                    assert_eq!(precision, Precision::Mxfp8);
                    assert_eq!(envelope.source_f32_projections.len(), 1);
                    assert_eq!(
                        envelope.mxfp8_semantics.quantization_version,
                        MXFP8_QUANTIZATION_VERSION
                    );
                    &envelope.recipe
                }
            };
            assert_eq!(canonical_recipe.projections.len(), projection_count);
            assert!(canonical_recipe
                .projections
                .windows(2)
                .all(|w| w[0].id < w[1].id));
            // The embedded compact object must have exactly the bytes hashed
            // by the loader, with neither an enum tag nor sorted Value keys.
            let wrapper_json = serde_json::to_string(&sparse).unwrap();
            let envelope_json = String::from_utf8(canonical).unwrap();
            assert!(wrapper_json.ends_with(&format!("\"canonical_envelope\":{envelope_json}}}")));
            assert!(envelope_json.starts_with("{\"schema\":"));
        }
    }

    #[test]
    fn exported_identity_rejects_untrusted_recipe_bytes_before_export() {
        let graph = fixture();
        let recipe = mxfp8_recipe(&graph, MODEL, &["trunk.block00.pair00.ffn.dual"]).unwrap();
        let valid = serde_json::to_string(&recipe).unwrap();
        for source in [
            "{".to_string(),
            valid.replacen('{', "{\"version\":2,", 1),
            valid.replacen('{', "{\"canonical_envelope\":{},", 1),
            valid.replacen("\"mxfp8\"", "\"mxfp8-unversioned\"", 1),
        ] {
            assert!(resolve_recipe_identity(source.as_bytes(), &graph, MODEL).is_err());
        }
        let mut variants = Vec::new();
        let mut changed = recipe.clone();
        changed.model_sha256 = "f".repeat(64);
        variants.push(changed);
        let mut changed = recipe.clone();
        changed.graph_sha256 = "e".repeat(64);
        variants.push(changed);
        let mut changed = recipe.clone();
        changed.projections[0].expected_k += 1;
        variants.push(changed);
        let mut changed = recipe.clone();
        changed.projections.push(changed.projections[0].clone());
        variants.push(changed);
        let mut changed = recipe.clone();
        changed.version = 1;
        variants.push(changed);
        let mut changed = recipe;
        changed.projections.clear();
        variants.push(changed);
        for source in variants {
            assert!(
                resolve_recipe_identity(&serde_json::to_vec(&source).unwrap(), &graph, MODEL)
                    .is_err()
            );
        }
    }

    #[test]
    fn exported_mxfp8_identity_contains_actual_original_f32_source_hashes() {
        let mut graph = fixture();
        let source = serde_json::to_vec(
            &mxfp8_recipe(&graph, MODEL, &["trunk.block00.pair00.ffn.dual"]).unwrap(),
        )
        .unwrap();
        let before = resolve_recipe_identity(&source, &graph, MODEL).unwrap();
        if let Layer::Ffn(layer) = &mut graph.layers[5] {
            if let TensorData::F32(weights) = &mut layer.gate_weight.data {
                weights[0] = f32::from_bits(1);
            }
        }
        let after = resolve_recipe_identity(&source, &graph, MODEL).unwrap();
        assert_eq!(before.graph_sha256, after.graph_sha256);
        assert_eq!(
            before.source_recipe_file_sha256,
            after.source_recipe_file_sha256
        );
        assert_ne!(before.recipe_sha256, after.recipe_sha256);
        match (&before.canonical_envelope, &after.canonical_envelope) {
            (CanonicalRecipeIdentity::V2(a), CanonicalRecipeIdentity::V2(b)) => {
                assert_ne!(
                    a.source_f32_projections[0].source_f32_sha256,
                    b.source_f32_projections[0].source_f32_sha256
                );
                assert_eq!(
                    b.source_f32_projections[0].packing,
                    "row-major-gate-then-up-v1"
                );
                assert_eq!(
                    (
                        b.source_f32_projections[0].gemm_n,
                        b.source_f32_projections[0].logical_k
                    ),
                    (16, 32)
                );
            }
            _ => panic!("MXFP8 export must use its versioned envelope"),
        }
        if let Layer::Ffn(layer) = &mut graph.layers[5] {
            if let TensorData::F32(weights) = &mut layer.gate_weight.data {
                weights[0] = f32::NAN;
            }
        }
        assert!(resolve_recipe_identity(&source, &graph, MODEL).is_err());
    }
}
