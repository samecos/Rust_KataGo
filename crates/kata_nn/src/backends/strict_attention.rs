//! Independently fingerprinted, opt-in B14 SM120 attention candidate.
//!
//! The embedded strict r2 CUBIN and exact QK-only RoPE PTX are immutable inputs;
//! this module never invokes a compiler or reads a runtime artifact path. Their
//! standalone evidence is not whole-model certification. Keep the ordinary
//! q64-serial fallback and all existing half/FP32 boundaries unchanged.

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    LaunchConfig, PushKernelArg, sys,
};
use cudarc::nvrtc::Ptx;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, OnceLock};

const CANDIDATE: &str = "fa4-strict-b14-r1";
const BATCH: usize = 14;
const SEQUENCE: usize = 361;
const HEADS: usize = 12;
const HEAD_DIM: usize = 32;
const WIDTH: usize = HEADS * HEAD_DIM;
const PACKED_ELEMENTS: usize = BATCH * SEQUENCE * WIDTH * 3;
const OUTPUT_ELEMENTS: usize = BATCH * SEQUENCE * WIDTH;
const COEFFICIENT_ELEMENTS: usize = SEQUENCE * WIDTH / 2;
const ATTENTION_SYMBOL: &str = "kernel_cutlass_kernel_strict_flash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16o3613212141152132415872_tensorptrf16gmemalign16o3613212141152132415872_tensorptrf1_0";
const ROPE_SYMBOL: &str = "probe_rope_no_vcopy";
const ATTENTION_SHA256: &str = "90341b6d5c54a69e983a7e7d8b62956748785ad44c3e4e1b2734b92f3da46749";
const ROPE_SHA256: &str = "3a5448f8cb639fca97bf13ce17b66f12bdc83c4875b8f146447e7c861f33191a";
const ABI_SHA256: &str = "8c8ef96685ce5dceceb1aaf0d5c1b8aa30183defd1affbd08c983c4ff5f152ae";
const ATTENTION_BYTES: &[u8] =
    include_bytes!("../../cuda-aot/fa4-strict-b14-r1/attention.sm120.cubin");
const ROPE_BYTES: &[u8] =
    include_bytes!("../../cuda-aot/fa4-strict-b14-r1/rope_no_vcopy.sm120.ptx");
const ABI_BYTES: &[u8] = include_bytes!("../../cuda-aot/fa4-strict-b14-r1/abi.json");
const FAST_DIVMOD_ONE: [u32; 3] = [1, 1, 0];
const _: () = assert!(std::mem::size_of::<[u32; 3]>() == 12);
const _: () = assert!(std::mem::align_of::<[u32; 3]>() == 4);

fn bundle_digest(files: &[(&str, &[u8])]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"Rust_KataGo strict-attention AOT\0");
    hash.update(1u32.to_le_bytes());
    for (name, bytes) in files {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    format!("sha256:{}", hex::encode(hash.finalize()))
}

/// GPU-free identity of the actual embedded bytes, independent of the legacy
/// core kernel build ID. Invalid embedded assets advertise no capability. Only
/// immutable runtime assets and ABI enter the identity, not mutable evidence.
pub(crate) fn fingerprint() -> Option<String> {
    static ID: OnceLock<Option<String>> = OnceLock::new();
    ID.get_or_init(|| {
        validate_assets().ok()?;
        Some(bundle_digest(&[
            ("attention.sm120.cubin", ATTENTION_BYTES),
            ("rope_no_vcopy.sm120.ptx", ROPE_BYTES),
            ("abi.json", ABI_BYTES),
        ]))
    })
    .clone()
}

fn verify_hash(label: &str, bytes: &[u8], expected: &str) -> Result<(), String> {
    let actual = hex::encode(Sha256::digest(bytes));
    if actual != expected {
        return Err(format!(
            "{CANDIDATE} {label} SHA256 mismatch: expected {expected}, actual {actual}"
        ));
    }
    Ok(())
}

fn expect_json(value: &Value, pointer: &str, expected: Value) -> Result<(), String> {
    if value.pointer(pointer) != Some(&expected) {
        return Err(format!(
            "{CANDIDATE} ABI mismatch at {pointer}: expected {expected}, actual {:?}",
            value.pointer(pointer)
        ));
    }
    Ok(())
}

fn validate_abi(abi: &Value) -> Result<(), String> {
    for (pointer, expected) in [
        ("/schema", json!(1)),
        ("/revision", json!(1)),
        ("/candidate", json!(CANDIDATE)),
        ("/target/compute_capability", json!([12, 0])),
        ("/attention/kernel", json!(ATTENTION_SYMBOL)),
        ("/attention/sha256", json!(ATTENTION_SHA256)),
        ("/attention/parameter_bytes", json!([8, 8, 8, 8, 4, 12])),
        ("/attention/parameter_alignments", json!([8, 8, 8, 8, 4, 4])),
        (
            "/attention/parameter_5_bytes_hex",
            json!("010000000100000000000000"),
        ),
        ("/attention/grid", json!([3, 12, 14])),
        ("/attention/block", json!([128, 1, 1])),
        ("/attention/dynamic_shared_memory_bytes", json!(20480)),
        ("/attention/qkv_shape_bshd", json!([14, 361, 12, 32])),
        (
            "/attention/qkv_stride_elements",
            json!([415872, 1152, 32, 1]),
        ),
        (
            "/attention/output_stride_elements",
            json!([138624, 384, 32, 1]),
        ),
        (
            "/attention/qkv_pointer_offsets_bytes",
            json!([0, 768, 1536]),
        ),
        (
            "/attention/qkv_pointer_allocations",
            json!([
                "rotated_scratch",
                "rotated_scratch",
                "original_packed_input"
            ]),
        ),
        ("/attention/pointer_min_alignment_bytes", json!(16)),
        ("/attention/runtime_shape_stride_arguments", json!(false)),
        ("/attention/qk_accumulator", json!("float32")),
        ("/attention/pv_accumulator", json!("float32")),
        ("/rope_helper/kernel", json!(ROPE_SYMBOL)),
        ("/rope_helper/sha256", json!(ROPE_SHA256)),
        ("/rope_helper/parameter_bytes", json!([8, 8, 8, 8])),
        ("/rope_helper/parameter_alignments", json!([8, 8, 8, 8])),
        ("/rope_helper/grid", json!([948, 1, 1])),
        ("/rope_helper/block", json!([1024, 1, 1])),
        ("/rope_helper/dynamic_shared_memory_bytes", json!(0)),
        ("/rope_helper/pairs", json!(BATCH * COEFFICIENT_ELEMENTS)),
        ("/rope_helper/coefficient_shape", json!([361, 192])),
        ("/numeric_contract/fp16_encoding_revision", json!(1)),
        ("/numeric_contract/fast_math", json!(false)),
    ] {
        expect_json(abi, pointer, expected)?;
    }
    Ok(())
}

fn validate_assets() -> Result<(), String> {
    verify_hash("attention CUBIN", ATTENTION_BYTES, ATTENTION_SHA256)?;
    verify_hash("QK-only RoPE PTX", ROPE_BYTES, ROPE_SHA256)?;
    verify_hash("ABI document", ABI_BYTES, ABI_SHA256)?;
    let abi: Value = serde_json::from_slice(ABI_BYTES)
        .map_err(|e| format!("{CANDIDATE} invalid ABI JSON: {e}"))?;
    validate_abi(&abi)?;
    Ok(())
}

pub(crate) fn validate_shape(
    sequence: usize,
    heads: usize,
    head_dim: usize,
    width: usize,
) -> Result<(), String> {
    if (sequence, heads, head_dim, width) != (SEQUENCE, HEADS, HEAD_DIM, WIDTH) {
        return Err(format!(
            "{CANDIDATE} requires S361/H12/D32/width384, actual S{sequence}/H{heads}/D{head_dim}/width{width}"
        ));
    }
    Ok(())
}

/// The AOT has no dynamic batch argument. Every other nonempty physical batch
/// uses the explicitly required q64-serial kernel; no additional padding occurs.
pub(crate) fn uses_aot(batch: usize) -> Result<bool, String> {
    if batch == 0 {
        return Err(format!("{CANDIDATE} cannot apply an empty physical batch"));
    }
    Ok(batch == BATCH)
}

fn check_lengths(raw: usize, scratch: usize, output: usize) -> Result<(), String> {
    if raw != PACKED_ELEMENTS || scratch < PACKED_ELEMENTS || output != OUTPUT_ELEMENTS {
        return Err(format!(
            "{CANDIDATE} packed buffer lengths mismatch: raw={raw}, scratch={scratch}, output={output}"
        ));
    }
    Ok(())
}

fn disjoint(a: u64, a_bytes: usize, b: u64, b_bytes: usize) -> bool {
    match (a.checked_add(a_bytes as u64), b.checked_add(b_bytes as u64)) {
        (Some(a_end), Some(b_end)) => a_end <= b || b_end <= a,
        _ => false,
    }
}

pub(crate) struct StrictAttention {
    // Retain both modules for the runtime's full lifetime, including all lanes.
    _attention_module: Arc<CudaModule>,
    _rope_module: Arc<CudaModule>,
    attention: CudaFunction,
    rope: CudaFunction,
    artifact: String,
    reported_batches: Mutex<BTreeSet<usize>>,
}

impl StrictAttention {
    pub(crate) fn load(device: &Arc<CudaContext>) -> Result<Self, String> {
        validate_assets()?;
        let major = device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| format!("{CANDIDATE} compute capability query failed: {e}"))?;
        let minor = device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| format!("{CANDIDATE} compute capability query failed: {e}"))?;
        if (major, minor) != (12, 0) {
            return Err(format!(
                "{CANDIDATE} requires SM120, actual {major}.{minor}"
            ));
        }
        let attention_module = device
            .load_module(Ptx::from_binary(ATTENTION_BYTES.to_vec()))
            .map_err(|e| format!("{CANDIDATE} CUBIN load failed: {e}"))?;
        let attention = attention_module
            .load_function(ATTENTION_SYMBOL)
            .map_err(|e| format!("{CANDIDATE} attention symbol missing: {e}"))?;
        let rope_source = std::str::from_utf8(ROPE_BYTES)
            .map_err(|e| format!("{CANDIDATE} RoPE PTX is not UTF-8: {e}"))?;
        let rope_module = device
            .load_module(Ptx::from_src(rope_source))
            .map_err(|e| format!("{CANDIDATE} RoPE PTX load failed: {e}"))?;
        let rope = rope_module
            .load_function(ROPE_SYMBOL)
            .map_err(|e| format!("{CANDIDATE} RoPE symbol missing: {e}"))?;
        for (name, function, threads) in [("attention", &attention, 128), ("RoPE", &rope, 1024)] {
            let version = function
                .binary_version()
                .map_err(|e| format!("{CANDIDATE} {name} binary-version query failed: {e}"))?;
            let maximum = function
                .max_threads_per_block()
                .map_err(|e| format!("{CANDIDATE} {name} thread-limit query failed: {e}"))?;
            if version != 120 || maximum < threads {
                return Err(format!(
                    "{CANDIDATE} {name} loaded function scope mismatch: binary={version}, max_threads={maximum}"
                ));
            }
        }
        Ok(Self {
            _attention_module: attention_module,
            _rope_module: rope_module,
            attention,
            rope,
            artifact: fingerprint().expect("embedded strict attention fingerprint"),
            reported_batches: Mutex::new(BTreeSet::new()),
        })
    }

    pub(crate) fn launch_rope(
        &self,
        stream: &Arc<CudaStream>,
        raw: &CudaSlice<u16>,
        cos: &CudaSlice<f32>,
        sin: &CudaSlice<f32>,
        scratch: &mut CudaSlice<u16>,
    ) -> Result<(), String> {
        check_lengths(raw.len(), scratch.len(), OUTPUT_ELEMENTS)?;
        if cos.len() != COEFFICIENT_ELEMENTS || sin.len() != COEFFICIENT_ELEMENTS {
            return Err(format!("{CANDIDATE} requires FP32 RoPE cache [361,192]"));
        }
        let (raw_ptr, _raw_guard) = raw.device_ptr(stream);
        let (scratch_ptr, _scratch_guard) = scratch.device_ptr_mut(stream);
        if !disjoint(
            raw_ptr,
            PACKED_ELEMENTS * 2,
            scratch_ptr,
            PACKED_ELEMENTS * 2,
        ) {
            return Err(format!(
                "{CANDIDATE} RoPE input and scratch must be disjoint"
            ));
        }
        // This is the measured 1024-thread linear helper. The original packed
        // V slots in scratch remain unwritten and are never supplied to AOT.
        unsafe {
            stream
                .launch_builder(&self.rope)
                .arg(&raw_ptr)
                .arg(cos)
                .arg(sin)
                .arg(&scratch_ptr)
                .launch(LaunchConfig {
                    grid_dim: (948, 1, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| format!("{CANDIDATE} exact QK-only RoPE launch failed: {e}"))?;
        Ok(())
    }

    pub(crate) fn launch_attention(
        &self,
        stream: &Arc<CudaStream>,
        scratch: &CudaSlice<u16>,
        raw: &CudaSlice<u16>,
        output: &mut CudaSlice<u16>,
        scale: f32,
    ) -> Result<(), String> {
        check_lengths(raw.len(), scratch.len(), output.len())?;
        if !scale.is_finite() || scale <= 0.0 {
            return Err(format!(
                "{CANDIDATE} requires a finite positive natural FP32 scale"
            ));
        }
        let (q, _q_guard) = scratch.device_ptr(stream);
        let (raw_ptr, _raw_guard) = raw.device_ptr(stream);
        let (out, _out_guard) = output.device_ptr_mut(stream);
        if [q, raw_ptr, out].iter().any(|p| p % 16 != 0)
            || !disjoint(q, PACKED_ELEMENTS * 2, raw_ptr, PACKED_ELEMENTS * 2)
            || !disjoint(q, PACKED_ELEMENTS * 2, out, OUTPUT_ELEMENTS * 2)
            || !disjoint(raw_ptr, PACKED_ELEMENTS * 2, out, OUTPUT_ELEMENTS * 2)
        {
            return Err(format!(
                "{CANDIDATE} requires aligned, disjoint input/scratch/output buffers"
            ));
        }
        let k = q + (WIDTH * 2) as u64;
        let v = raw_ptr + (WIDTH * 4) as u64;
        // Exactly six by-value Driver parameters. No shape descriptor, LOG2_E
        // conversion, V copy, or new half conversion is inserted here.
        unsafe {
            stream
                .launch_builder(&self.attention)
                .arg(&q)
                .arg(&k)
                .arg(&v)
                .arg(&out)
                .arg(&scale)
                .arg(&FAST_DIVMOD_ONE)
                .launch(LaunchConfig {
                    grid_dim: (3, HEADS as u32, BATCH as u32),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 20480,
                })
        }
        .map_err(|e| format!("{CANDIDATE} strict AOT attention launch failed: {e}"))?;
        Ok(())
    }

    pub(crate) fn launch_attention_rotated_packed(
        &self,
        stream: &Arc<CudaStream>,
        packed: &CudaSlice<u16>,
        output: &mut CudaSlice<u16>,
        scale: f32,
    ) -> Result<(), String> {
        check_lengths(packed.len(), packed.len(), output.len())?;
        if !scale.is_finite() || scale <= 0.0 {
            return Err(format!(
                "{CANDIDATE} requires a finite positive natural FP32 scale"
            ));
        }
        let (q, _q_guard) = packed.device_ptr(stream);
        let raw_ptr = q;
        let (out, _out_guard) = output.device_ptr_mut(stream);
        if [q, raw_ptr, out].iter().any(|p| p % 16 != 0)
            || !disjoint(q, PACKED_ELEMENTS * 2, out, OUTPUT_ELEMENTS * 2)
            || !disjoint(raw_ptr, PACKED_ELEMENTS * 2, out, OUTPUT_ELEMENTS * 2)
        {
            return Err(format!(
                "{CANDIDATE} requires aligned packed input and disjoint output"
            ));
        }
        let k = q + (WIDTH * 2) as u64;
        let v = raw_ptr + (WIDTH * 4) as u64;
        // Exactly six by-value Driver parameters. No shape descriptor, LOG2_E
        // conversion, V copy, or new half conversion is inserted here.
        unsafe {
            stream
                .launch_builder(&self.attention)
                .arg(&q)
                .arg(&k)
                .arg(&v)
                .arg(&out)
                .arg(&scale)
                .arg(&FAST_DIVMOD_ONE)
                .launch(LaunchConfig {
                    grid_dim: (3, HEADS as u32, BATCH as u32),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 20480,
                })
        }
        .map_err(|e| format!("{CANDIDATE} strict AOT attention launch failed: {e}"))?;
        Ok(())
    }

    pub(crate) fn report_scope(&self, batch: usize) {
        if !self.reported_batches.lock().unwrap().insert(batch) {
            return;
        }
        if batch == BATCH {
            eprintln!(
                "[cuda-tactic] name=attention requested={CANDIDATE} launch=strict-aot effective={CANDIDATE} physical_batch={batch} rope=exact-qk-only v=original-packed abi=1 artifact={} identity=matched production_certified=0",
                self.artifact
            );
        } else {
            eprintln!(
                "[cuda-tactic] name=attention requested={CANDIDATE} launch=fa2 effective=fa2 tile=q64-serial physical_batch={batch} fallback=q64-serial reason=physical-batch-not-14 artifact={} identity=matched production_certified=0",
                self.artifact
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_strict_attention_assets_and_abi_are_pinned() {
        validate_assets().unwrap();
        let mut abi: Value = serde_json::from_slice(ABI_BYTES).unwrap();
        abi["attention"]["parameter_bytes"] = json!([48, 48, 48, 48, 4, 4, 4, 4, 12]);
        assert!(validate_abi(&abi).unwrap_err().contains("parameter_bytes"));
        let changed = [ATTENTION_BYTES[0] ^ 1];
        assert!(verify_hash("changed", &changed, ATTENTION_SHA256).is_err());
        assert_ne!(
            bundle_digest(&[("a", b"bc")]),
            bundle_digest(&[("ab", b"c")])
        );
        assert_ne!(
            bundle_digest(&[("a", b"bc")]),
            bundle_digest(&[("a", b"bd")])
        );
    }

    #[test]
    fn strict_attention_scope_uses_only_physical_b14() {
        validate_shape(361, 12, 32, 384).unwrap();
        for shape in [(360, 12, 32, 384), (361, 6, 64, 384), (361, 12, 32, 768)] {
            assert!(validate_shape(shape.0, shape.1, shape.2, shape.3).is_err());
        }
        for batch in 1..=64 {
            assert_eq!(uses_aot(batch).unwrap(), batch == 14);
        }
        assert!(uses_aot(0).is_err());
        check_lengths(14 * 361 * 1152, 14 * 361 * 2304, 14 * 361 * 384).unwrap();
        assert!(check_lengths(16 * 361 * 1152, PACKED_ELEMENTS, OUTPUT_ELEMENTS).is_err());
        assert!(check_lengths(PACKED_ELEMENTS, PACKED_ELEMENTS - 1, OUTPUT_ELEMENTS).is_err());
        assert!(disjoint(16, 16, 32, 16));
        assert!(!disjoint(16, 17, 32, 16));
        assert!(!disjoint(u64::MAX - 3, 8, 16, 16));
        assert_eq!(
            FAST_DIVMOD_ONE.map(u32::to_le_bytes).concat(),
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        );
    }
}
