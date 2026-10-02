//! Versioned persistence for the INT8 cuBLASLt workspace cache only.
//!
//! This is deliberately not a complete execution plan: it has no FP16/custom
//! kernels, operation-ID map, numeric certificate, performance result, or
//! model-level readiness claim. The caller supplies an authoritative, exact
//! shape inventory and binding. Runtime code separately reads actual LT
//! descriptors/algorithm attributes and checks actual launch addresses.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const SCHEMA: &str = "rustgo-int8-lt-cache-v1";
pub const SCOPE: &str = "partial-int8-lt-cache-only";
pub const ENCODING: &str = "serde-struct-order-sorted-inventory-native-algo-hex-v1";
pub const MAX_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_ENTRIES: usize = 4096;
pub type Result<T> = std::result::Result<T, String>;

fn require(ok: bool, msg: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(format!("INT8 algorithm plan: {msg}")) }
}
pub fn sha256(bytes: &[u8]) -> String { hex::encode(Sha256::digest(bytes)) }
fn sha(value: &str) -> Result<()> {
    require(value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)), "invalid lowercase SHA256")
}
fn text(value: &str) -> Result<()> {
    require(!value.is_empty() && value.len() <= 256 && !value.chars().any(char::is_control), "invalid identity text")
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceIdentity {
    pub logical_name: String,
    pub sha256: String,
}

/// All fields are strict. Except LT version/ABI, CC/SM count and driver API
/// version (checked by runtime code), provenance is supplied by the model
/// loader and must come from its actual executable/model/library evidence.
/// A self-declared binding is not independent device/library attestation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub model_sha256: String,
    pub graph_sha256: String,
    pub recipe_sha256: String,
    pub execution_policy_sha256: String,
    pub executable_sha256: String,
    pub host_source_sha256: String,
    pub kernel_build_sha256: String,
    pub device_fingerprint_sha256: String,
    pub gpu_uuid: String,
    pub compute_capability: [u32; 2],
    pub sm_count: u32,
    pub driver_api_version: i32,
    pub cublaslt_version: u64,
    pub cublaslt_binary_sha256: String,
    pub cublas_binary_sha256: String,
    /// Handles static or dynamic CUDA runtime provenance without inventing a DLL.
    pub cuda_runtime_provenance_sha256: String,
    pub host_abi: String,
    pub pointer_width_bits: u32,
    pub little_endian: bool,
    pub algo_size_bytes: u32,
    pub algo_alignment_bytes: u32,
    /// Sorted immutable source names and contents; never live process addresses.
    pub sources: Vec<SourceIdentity>,
}

impl Binding {
    pub fn validate(&self) -> Result<()> {
        for value in [&self.model_sha256, &self.graph_sha256, &self.recipe_sha256,
            &self.execution_policy_sha256, &self.executable_sha256, &self.host_source_sha256,
            &self.kernel_build_sha256, &self.device_fingerprint_sha256,
            &self.cublaslt_binary_sha256, &self.cublas_binary_sha256,
            &self.cuda_runtime_provenance_sha256] { sha(value)?; }
        text(&self.gpu_uuid)?; text(&self.host_abi)?;
        require(self.compute_capability == [12, 0] && self.sm_count > 0 && self.sm_count <= i32::MAX as u32, "this revision requires SM120")?;
        require(self.driver_api_version > 0 && self.cublaslt_version >= 130000, "invalid runtime versions")?;
        require(self.pointer_width_bits == 64 && self.algo_size_bytes == 64 && self.algo_alignment_bytes == 8,
            "unsupported native ABI")?;
        require(!self.sources.is_empty() && self.sources.len() <= 64, "missing or excessive source identities")?;
        let mut previous: Option<&str> = None;
        for source in &self.sources {
            text(&source.logical_name)?; sha(&source.sha256)?;
            require(!source.logical_name.contains(['\\', ':']) && !source.logical_name.starts_with('/')
                && !source.logical_name.split('/').any(|x| x == ".." || x.is_empty()), "source name must be logical and relative")?;
            require(previous.is_none_or(|p| p < source.logical_name.as_str()), "source identities must be unique and sorted")?;
            previous = Some(&source.logical_name);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProblemKey {
    pub physical_batch: u32,
    pub rows: u64,
    pub n: u64,
    pub logical_k: u64,
    pub padded_k: u64,
}
impl ProblemKey {
    pub fn new(rows: usize, n: usize, logical_k: usize, padded_k: usize) -> Result<Self> {
        let key = Self { physical_batch: u32::try_from(rows / 361).map_err(|_| "physical batch overflow")?,
            rows: rows as u64, n: n as u64, logical_k: logical_k as u64, padded_k: padded_k as u64 };
        key.validate()?; Ok(key)
    }
    pub fn validate(&self) -> Result<()> {
        require(self.physical_batch > 0 && self.rows == u64::from(self.physical_batch) * 361
            && self.rows <= i32::MAX as u64, "invalid physical batch/token rows")?;
        require(self.n > 0 && self.n <= i32::MAX as u64 && self.n % 4 == 0
            && self.logical_k > 0 && self.logical_k <= 8192
            && self.padded_k == self.logical_k.div_ceil(16) * 16, "invalid INT8 dimensions/padding")?;
        require(self.rows.checked_mul(self.n).is_some_and(|v| v <= u32::MAX as u64), "output indexing overflow")
    }
    pub fn cache_key(&self) -> (usize, usize, usize) { (self.rows as usize, self.n as usize, self.padded_k as usize) }
}

pub fn validate_inventory(inventory: &[ProblemKey]) -> Result<()> {
    require(!inventory.is_empty() && inventory.len() <= MAX_ENTRIES, "empty or excessive inventory")?;
    for (i, key) in inventory.iter().enumerate() {
        key.validate()?;
        require(i == 0 || inventory[i-1] < *key, "inventory must be sorted and unique")?;
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layout {
    pub data_type: u32,
    pub order: i32,
    pub rows: u64,
    pub columns: u64,
    pub leading_dimension: i64,
    pub batch_count: i32,
    pub batch_stride: i64,
    pub plane_offset: i64,
    pub batch_mode: u32,
}
impl Layout {
    fn plain(data_type: u32, rows: u64, columns: u64) -> Self {
        Self { data_type, order: 0, rows, columns, leading_dimension: rows as i64,
            batch_count: 1, batch_stride: 0, plane_offset: 0, batch_mode: 0 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Descriptors {
    pub compute_type: i32,
    pub scale_type: i32,
    pub pointer_mode: i32,
    pub transpose_a: i32,
    pub transpose_b: i32,
    pub transpose_c: i32,
    pub fill_mode: i32,
    pub epilogue: u32,
    pub sm_count_target: i32,
    pub fast_accum: i8,
    pub scale_modes: [i32; 4],
    /// All supported auxiliary/bias/scale/amax pointers were queried as NULL.
    /// No device pointer values are persisted.
    pub auxiliary_pointers_null: bool,
    pub alpha: i32,
    pub beta: i32,
    pub c_d_alias: bool,
    pub layouts: [Layout; 4],
}
impl Descriptors {
    pub fn expected(key: ProblemKey) -> Self {
        // Values are CUDA native enum ABI values, not alternative datatype IDs.
        Self { compute_type: 72, scale_type: 10, pointer_mode: 0,
            transpose_a: 1, transpose_b: 0, transpose_c: 0, fill_mode: 2,
            epilogue: 1, sm_count_target: 0, fast_accum: 0, scale_modes: [0; 4],
            auxiliary_pointers_null: true, alpha: 1, beta: 0, c_d_alias: true,
            layouts: [Layout::plain(3, key.padded_k, key.n), Layout::plain(3, key.padded_k, key.rows),
                Layout::plain(10, key.n, key.rows), Layout::plain(10, key.n, key.rows)] }
    }
}

/// Optional newer fields retain explicit NOT_SUPPORTED status. INVALID_VALUE
/// is not treated as unsupported: it can mean a wrong buffer ABI size.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", content = "value", rename_all = "snake_case")]
pub enum Queried<T> { Value(T), NotSupported }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlgorithmAttributes {
    pub algorithm_id: i32,
    pub tile_id: u32,
    pub stages_id: u32,
    pub split_k: i32,
    pub reduction_scheme: u32,
    pub cta_swizzling: u32,
    pub custom_option: u32,
    pub inner_shape_id: Queried<u16>,
    pub cluster_shape_id: Queried<u16>,
    pub numerical_implementation_flags: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Implementation {
    pub descriptors: Descriptors,
    /// Exact native cublasLtMatmulAlgo_t bytes, 64 bytes encoded as lowercase hex.
    pub opaque_algorithm_hex: String,
    pub attributes: AlgorithmAttributes,
    pub minimum_alignment_bytes: [u32; 4],
    pub required_workspace_bytes: u64,
    pub provided_workspace_bytes: u64,
    pub workspace_alignment_bytes: u32,
    pub workspace_ownership: String,
}
impl Implementation {
    pub fn validate(&self, key: ProblemKey) -> Result<()> {
        key.validate()?;
        require(self.descriptors == Descriptors::expected(key), "descriptor problem differs from this INT8 runtime contract")?;
        require(self.opaque_algorithm_hex.len() == 128 && self.opaque_algorithm_hex.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)), "invalid opaque algorithm encoding")?;
        require(self.attributes.algorithm_id >= 0 && self.attributes.split_k >= 0, "invalid algorithm attributes")?;
        require(self.required_workspace_bytes <= self.provided_workspace_bytes
            && self.provided_workspace_bytes > 0 && self.workspace_alignment_bytes == 256
            && self.workspace_ownership == "runtime-per-stream-exclusive", "workspace contract mismatch")?;
        for (layout, &alignment) in self.descriptors.layouts.iter().zip(&self.minimum_alignment_bytes) {
            require(alignment.is_power_of_two(), "invalid algorithm alignment")?;
            let bytes = if layout.data_type == 3 { 1 } else { 4 };
            require((layout.leading_dimension as u64 * bytes) % u64::from(alignment) == 0,
                "layout leading dimension violates algorithm alignment")?;
        }
        Ok(())
    }
    /// Check actual subview addresses on every launch, including cache hits and
    /// graph capture. This must not be replaced by allocation-base guarantees.
    pub fn check_addresses(&self, addresses: [u64; 4], workspace: u64, workspace_len: usize) -> Result<()> {
        for (&address, &alignment) in addresses.iter().zip(&self.minimum_alignment_bytes) {
            require(alignment.is_power_of_two() && address != 0 && address % u64::from(alignment) == 0,
                "actual matrix address violates algorithm alignment")?;
        }
        require(addresses[2] == addresses[3], "C/D must alias in this runtime")?;
        require(workspace != 0 && workspace % 256 == 0
            && workspace_len as u64 == self.provided_workspace_bytes
            && workspace_len as u64 >= self.required_workspace_bytes, "actual workspace differs from frozen contract")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry { pub key: ProblemKey, pub implementation: Implementation }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub schema: String,
    pub scope: String,
    pub encoding: String,
    pub binding: Binding,
    pub inventory: Vec<ProblemKey>,
    pub entries: Vec<Entry>,
}
impl Plan {
    pub fn validate(&self) -> Result<()> {
        require(self.schema == SCHEMA && self.scope == SCOPE && self.encoding == ENCODING, "schema/scope/encoding mismatch")?;
        self.binding.validate()?; validate_inventory(&self.inventory)?;
        require(self.entries.len() == self.inventory.len(), "incomplete or extra entries")?;
        let mut shared: BTreeMap<(usize, usize, usize), &Implementation> = BTreeMap::new();
        for (entry, key) in self.entries.iter().zip(&self.inventory) {
            require(entry.key == *key, "entry coverage/order mismatch")?;
            entry.implementation.validate(*key)?;
            if let Some(previous) = shared.insert(key.cache_key(), &entry.implementation) {
                require(previous == &entry.implementation, "shared physical cache key has conflicting algorithms")?;
            }
        }
        Ok(())
    }
    pub fn canonical_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(|e| e.to_string())?;
        require(bytes.len() <= MAX_BYTES, "document too large")?;
        Ok(bytes)
    }
    pub fn parse_exact(bytes: &[u8], expected_sha: &str, binding: &Binding, inventory: &[ProblemKey]) -> Result<Self> {
        sha(expected_sha)?; binding.validate()?; validate_inventory(inventory)?;
        require(bytes.len() <= MAX_BYTES && sha256(bytes) == expected_sha, "document size/SHA mismatch")?;
        let plan: Self = serde_json::from_slice(bytes).map_err(|e| format!("INT8 plan JSON: {e}"))?;
        plan.validate()?;
        require(&plan.binding == binding, "expected binding mismatch")?;
        require(plan.inventory == inventory, "expected inventory mismatch")?;
        require(plan.canonical_bytes()? == bytes, "document is not canonical")?;
        Ok(plan)
    }
}

/// Per-workspace state, never shared between streams or installed over a live
/// cache. Runtime uses it before any heuristic/restore and after launch success.
pub(crate) struct Session {
    binding: Binding,
    inventory: Vec<ProblemKey>,
    expected: Option<BTreeMap<ProblemKey, Implementation>>,
    observed: BTreeMap<ProblemKey, Implementation>,
    poisoned: bool,
}
impl Session {
    pub(crate) fn record(binding: Binding, inventory: Vec<ProblemKey>) -> Result<Self> {
        binding.validate()?; validate_inventory(&inventory)?;
        Ok(Self { binding, inventory, expected: None, observed: BTreeMap::new(), poisoned: false })
    }
    pub(crate) fn restore(plan: Plan) -> Result<Self> {
        plan.validate()?;
        Ok(Self { binding: plan.binding, inventory: plan.inventory,
            expected: Some(plan.entries.into_iter().map(|e| (e.key, e.implementation)).collect()),
            observed: BTreeMap::new(), poisoned: false })
    }
    pub(crate) fn expected(&self, key: ProblemKey) -> Result<Option<&Implementation>> {
        require(!self.poisoned, "workspace plan session was poisoned by an earlier failure")?;
        require(self.inventory.binary_search(&key).is_ok(), "shape missing from authoritative inventory; no heuristic fallback")?;
        Ok(self.expected.as_ref().map(|entries| &entries[&key]))
    }
    pub(crate) fn has_observed(&self, key: ProblemKey) -> bool { self.observed.contains_key(&key) }
    pub(crate) fn observe(&mut self, key: ProblemKey, implementation: &Implementation) -> Result<()> {
        if let Some(expected) = self.expected(key)? { require(expected == implementation, "restored implementation mismatch")?; }
        implementation.validate(key)?;
        if let Some(previous) = self.observed.get(&key) { require(previous == implementation, "algorithm changed within one session")?; }
        else { self.observed.insert(key, implementation.clone()); }
        Ok(())
    }
    pub(crate) fn ensure_healthy(&self) -> Result<()> {
        require(!self.poisoned, "workspace plan session was poisoned by an earlier failure")
    }
    pub(crate) fn complete_operation<T>(&mut self, result: Result<T>) -> Result<T> {
        match result {
            Err(error) => { self.poison(); Err(error) }
            Ok(value) => { self.ensure_healthy()?; Ok(value) }
        }
    }
    pub(crate) fn poison(&mut self) { self.poisoned = true; }
    pub(crate) fn export(&self) -> Result<Plan> {
        require(!self.poisoned, "workspace plan session was poisoned by an earlier failure")?;
        require(self.observed.keys().copied().collect::<BTreeSet<_>>() == self.inventory.iter().copied().collect(),
            "not all authoritative shapes have completed a launch")?;
        let plan = Plan { schema: SCHEMA.into(), scope: SCOPE.into(), encoding: ENCODING.into(),
            binding: self.binding.clone(), inventory: self.inventory.clone(),
            entries: self.observed.iter().map(|(&key, implementation)| Entry { key, implementation: implementation.clone() }).collect() };
        plan.validate()?; Ok(plan)
    }
}

/// Strict owner-object checks are stronger than raw CUDA handles: every
/// context has a default stream with handle zero. These live Arc references
/// never enter the serialized plan.
pub(crate) fn validate_owner_identity<S, C>(
    owner: &std::sync::Arc<S>, incoming: &std::sync::Arc<S>,
    owner_context: &std::sync::Arc<C>, incoming_context: &std::sync::Arc<C>,
    buffer_owners: [&std::sync::Arc<S>; 3], buffer_contexts: [&std::sync::Arc<C>; 3],
) -> Result<()> {
    require(std::sync::Arc::ptr_eq(owner, incoming), "workspace stream owner object mismatch")?;
    require_same_context(owner_context, incoming_context)?;
    for (buffer_owner, buffer_context) in buffer_owners.into_iter().zip(buffer_contexts) {
        require(std::sync::Arc::ptr_eq(owner, buffer_owner), "workspace scratch buffer stream owner mismatch")?;
        require_same_context(owner_context, buffer_context)?;
    }
    Ok(())
}

pub(crate) fn require_same_context<C>(owner: &std::sync::Arc<C>, actual: &std::sync::Arc<C>) -> Result<()> {
    require(std::sync::Arc::ptr_eq(owner, actual), "actual buffer/runtime context differs from workspace owner")
}

#[cfg(test)]
#[path = "int8_algorithm_plan_tests.rs"]
mod tests;
