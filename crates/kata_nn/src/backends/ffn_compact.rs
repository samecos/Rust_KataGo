//! Opt-in compaction after the existing RN-half weight boundary.
//! Only common whole-zero channels are removed. Kept values are copied as bits.
use std::collections::{BTreeSet, HashMap};
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use sha2::{Digest, Sha256};
use crate::onnx_parser::{FfnLayer, Layer, LayerGraph};

const MANIFEST: &[u8] = include_bytes!("ffn_compact_manifest.json");
#[derive(serde::Deserialize)]
struct Expected {
    hidden: usize,
    dense: [String; 3],
    packed: [String; 3],
    mapping: String,
}
fn expected() -> &'static Vec<Expected> {
    static VALUE: OnceLock<Vec<Expected>> = OnceLock::new();
    VALUE.get_or_init(|| serde_json::from_slice(MANIFEST).expect("compiled compact manifest"))
}
fn hash_half(values: &[u16]) -> String {
    let mut h = Sha256::new();
    for value in values { h.update(value.to_le_bytes()); }
    hex::encode(h.finalize())
}
fn hash_mapping(values: &[usize]) -> String {
    let mut h = Sha256::new();
    for &value in values { h.update((value as u32).to_le_bytes()); }
    hex::encode(h.finalize())
}
pub(crate) fn fingerprint() -> Option<String> {
    if !cfg!(all(katago_dualffn, windows, target_pointer_width = "64")) { return None; }
    static ID: OnceLock<String> = OnceLock::new();
    Some(ID.get_or_init(|| {
        let mut h = Sha256::new();
        h.update(b"exact-existing-half-ffn-channel-compaction-r1\0");
        for bytes in [MANIFEST, include_bytes!("ffn_compact.rs").as_slice(),
            include_bytes!("cuda_exec.rs").as_slice(), include_bytes!("cuda.rs").as_slice(),
            include_bytes!("../tactic_plan.rs").as_slice(),
            include_bytes!("../../cuda-host/dual_ffn_cutlass.cu").as_slice()] {
            h.update((bytes.len() as u64).to_le_bytes()); h.update(bytes);
        }
        format!("sha256:{}", hex::encode(h.finalize()))
    }).clone())
}

pub(crate) struct Packed {
    pub hidden: usize,
    pub dual: Vec<u16>,
    pub down: Vec<u16>,
    pub mapping: String,
}
fn pack_bits(gate: &[u16], up: &[u16], down: &[u16], hidden: usize, channels: usize) -> Result<(Packed, Vec<usize>), String> {
    if hidden == 0 || hidden % 16 != 0 || channels == 0
        || [gate.len(), up.len(), down.len()].iter().any(|&n| n != hidden * channels) {
        return Err("compact FFN weight shape mismatch".into());
    }
    if gate.iter().chain(up).chain(down).any(|&v| (v & 0x7fff) >= 0x7c00) {
        return Err("compact FFN requires finite half weights".into());
    }
    let mut kept = Vec::new();
    let mut zero = Vec::new();
    for i in 0..hidden {
        let g = gate[i * channels..(i + 1) * channels].iter().all(|&v| v & 0x7fff == 0);
        let u = up[i * channels..(i + 1) * channels].iter().all(|&v| v & 0x7fff == 0);
        let d = (0..channels).all(|j| down[j * hidden + i] & 0x7fff == 0);
        if g != u || g != d { return Err("compact FFN zero masks differ".into()); }
        if g { zero.push(i); } else { kept.push(i); }
    }
    let width = kept.len().max(1).div_ceil(16) * 16;
    kept.extend(zero.into_iter().take(width - kept.len()));
    kept.sort_unstable();
    if kept.len() != width || width >= hidden { return Err("compact FFN has no aligned reduction".into()); }
    let mut dual = Vec::with_capacity(2 * width * channels);
    for source in [gate, up] { for &i in &kept { dual.extend_from_slice(&source[i * channels..(i + 1) * channels]); } }
    let mut packed_down = Vec::with_capacity(width * channels);
    for j in 0..channels { for &i in &kept { packed_down.push(down[j * hidden + i]); } }
    let mapping = hash_mapping(&kept);
    Ok((Packed { hidden: width, dual, down: packed_down, mapping }, kept))
}
pub(crate) fn pack(index: usize, layer: &FfnLayer, encode: fn(f32) -> u16) -> Result<Packed, String> {
    if layer.hidden != 1152 || !layer.residual_add { return Err("compact FFN requires original H1152 residual layer".into()); }
    let e = expected().get(index).ok_or("compact FFN unexpected layer index")?;
    let half = [&layer.gate_weight, &layer.up_weight, &layer.down_weight].map(|t| t.f32_data().iter().copied().map(encode).collect::<Vec<_>>());
    for (values, sha) in half.iter().zip(&e.dense) {
        if values.len() != 1152 * 384 || hash_half(values) != *sha {
            return Err(format!("compact FFN layer {index} original half identity mismatch"));
        }
    }
    let (p, _) = pack_bits(&half[0], &half[1], &half[2], 1152, 384)?;
    let parts = [&p.dual[..p.hidden * 384], &p.dual[p.hidden * 384..], p.down.as_slice()];
    if p.hidden != e.hidden || p.mapping != e.mapping || parts.iter().zip(&e.packed).any(|(values, sha)| hash_half(values) != *sha) {
        return Err(format!("compact FFN layer {index} packed identity mismatch"));
    }
    eprintln!("[cuda-tactic] name=ffn_compact stage=upload layer={index} original_hidden=1152 hidden={} removed={} mapping_sha={} identity=matched", p.hidden, 1152 - p.hidden, p.mapping);
    Ok(p)
}

#[cfg(katago_dualffn)]
unsafe extern "C" {
    fn katago_compact_ffn_create() -> *mut c_void;
    fn katago_compact_ffn_destroy(handle: *mut c_void);
    fn katago_compact_ffn_exec(handle: *mut c_void, input: *const c_void, gate: *const c_void,
        up: *const c_void, output: *mut c_void, tokens: i32, hidden: i32, stream: usize) -> i32;
}
pub(crate) struct FfnCompact {
    handles: Mutex<HashMap<usize, *mut c_void>>,
    reported: Mutex<BTreeSet<(usize, usize)>>,
}
// Each stream uses its own host handle. The caller serializes submissions on
// each stream, as for the existing DualFFN handle pool; distinct streams may run concurrently.
unsafe impl Send for FfnCompact {}
unsafe impl Sync for FfnCompact {}
impl Drop for FfnCompact {
    fn drop(&mut self) {
        #[cfg(katago_dualffn)]
        for &handle in self.handles.lock().unwrap().values() {
            if !handle.is_null() { unsafe { katago_compact_ffn_destroy(handle); } }
        }
    }
}
impl FfnCompact {
    pub(crate) fn load(graph: &LayerGraph, rt: &crate::backends::cuda::CudaRuntime) -> Result<Self, String> {
        if fingerprint().is_none() { return Err("compact FFN requires compiled Windows x64 CUTLASS artifact".into()); }
        crate::tactic_plan::validate_tf3_residual_target(&crate::backends::cuda::device_fingerprint(&rt.device)?,
            Some(unsafe { cudarc::cublaslt::sys::cublasLtGetVersion() } as u64))?;
        if graph.board_size != 19 || graph.mid_channels != 384 || graph.trunk_channels != 768
            || graph.layers.iter().filter(|l| matches!(l, Layer::Ffn(_))).count() != 33 || expected().len() != 33 {
            return Err("compact FFN requires the bound 33-layer native TF3 geometry".into());
        }
        Ok(Self { handles: Mutex::new(HashMap::new()), reported: Mutex::new(BTreeSet::new()) })
    }
    pub(crate) fn launch(&self, stream: &Arc<CudaStream>, input: &CudaSlice<u16>, dual: &CudaSlice<u16>,
        output: &mut CudaSlice<u16>, tokens: usize, hidden: usize) -> Result<bool, String> {
        if tokens == 0 || tokens > 5776 || tokens % 361 != 0 || hidden < 16 || hidden >= 1152 || hidden % 16 != 0
            || input.len() < tokens * 384 || dual.len() != 2 * hidden * 384 || output.len() < tokens * hidden {
            return Err("compact FFN launch shape mismatch".into());
        }
        #[cfg(not(katago_dualffn))]
        return Err("compact FFN CUTLASS capability missing".into());
        #[cfg(katago_dualffn)] {
            let key = stream.cu_stream() as usize;
            let handle = *self.handles.lock().unwrap().entry(key).or_insert_with(|| unsafe { katago_compact_ffn_create() });
            if handle.is_null() { return Err("compact FFN handle creation failed".into()); }
            let (a, _a) = input.device_ptr(stream); let (w, _w) = dual.device_ptr(stream); let (y, _y) = output.device_ptr_mut(stream);
            if [a, w, y].iter().any(|p| p % 16 != 0) { return Err("compact FFN unaligned device pointer".into()); }
            let code = unsafe { katago_compact_ffn_exec(handle, a as *const c_void, w as *const c_void,
                (w + (hidden * 384 * 2) as u64) as *const c_void, y as *mut c_void, tokens as i32, hidden as i32, key) };
            if code != 0 { return Err(format!("compact FFN execution failed: {code}")); }
            if self.reported.lock().unwrap().insert((tokens, hidden)) {
                eprintln!("[cuda-tactic] name=ffn_compact stage=execute effective=1 tokens={tokens} physical_batch={} hidden={hidden} kernel=original-dualgemm-fp32 cache_key=tokens+hidden", tokens / 361);
            }
            Ok(true)
        }
    }
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn preserve_signed_zeros_order_and_smallest_half() {
        let mut g = vec![0x8000; 32 * 2]; let mut u = g.clone(); let mut d = g.clone();
        g[2] = 1; u[2] = 0x8001; d[1] = 1;
        let (p, indices) = pack_bits(&g, &u, &d, 32, 2).unwrap();
        assert_eq!(indices, (0..16).collect::<Vec<_>>()); assert_eq!(p.hidden, 16);
        assert_eq!(&p.dual[..32], &g[..32]); assert_eq!(&p.dual[32..], &u[..32]);
        assert_eq!(&p.down[..16], &d[..16]); assert_eq!(&p.down[16..], &d[32..48]);
        d[1] = 0; assert!(pack_bits(&g, &u, &d, 32, 2).is_err());
        g[2] = 0x7c00; assert!(pack_bits(&g, &u, &d, 32, 2).is_err());
        assert!(pack_bits(&[], &[], &[], 32, 2).is_err());
    }
    #[test] fn all_bound_operator_packings_match() {
        let Some(root) = std::env::var_os("KATAGO_TEST_COMPACT_FIXTURES") else { return; };
        for (index, e) in expected().iter().enumerate() {
            let path = std::path::Path::new(&root).join(format!("layer-{index:02}"));
            let read = |name: &str| { let bytes = std::fs::read(path.join(name)).unwrap(); bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect::<Vec<_>>() };
            let raw = ["gate", "up", "down"].map(|s| read(&format!("dense-{s}.f16le")));
            for (values, sha) in raw.iter().zip(&e.dense) { assert_eq!(hash_half(values), *sha); }
            let (p, _) = pack_bits(&raw[0], &raw[1], &raw[2], 1152, 384).unwrap();
            assert_eq!(p.hidden, e.hidden); assert_eq!(p.mapping, e.mapping);
            assert_eq!(p.dual, [read("gate.f16le"), read("up.f16le")].concat()); assert_eq!(p.down, read("down.f16le"));
        }
        println!("COMPACT_PACKING_PASS layers=33 original_values=43794432 retained_hidden=21712");
    }
}
