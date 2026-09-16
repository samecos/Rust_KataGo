//! Fixed TF3 B14 output projection with FP32 residual storage and accumulation.
use std::sync::{Arc, OnceLock};
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig, PushKernelArg, sys};
use cudarc::nvrtc::Ptx;
use sha2::{Digest, Sha256};
use crate::onnx_parser::{Layer, LayerGraph};
const CUBIN: &[u8] = include_bytes!("../../cuda-aot/outproj-classic-n128-b14-r1/kernel.sm120.cubin");
const TEMPLATE: &[u8; 368] = include_bytes!("../../cuda-aot/outproj-classic-n128-b14-r1/params-template.bin");
const ABI: &[u8] = include_bytes!("../../cuda-aot/outproj-classic-n128-b14-r1/abi.json");
const CUBIN_SHA: &str = "07ce7eb15cbc59e813d0cdea7acb3ad0b344b6036bab31443e34b89f55baf375";
const TEMPLATE_SHA: &str = "8f9ab3ff2ed3ee75f51c7c6e2f8e4b5565aba8c009b33ada7715bf215393169c";
const ABI_SHA: &str = "3ce5f29dd256d61ad8254b80f6d6d5f284d6e16f03070132a8a212fc9fadc1b8";
const ROWS: usize = 5054;
const CHANNELS: usize = 384;
fn check(bytes: &[u8], expected: &str) -> Result<(), String> {
    if hex::encode(Sha256::digest(bytes)) != expected { return Err("outproj N128 artifact hash mismatch".into()); } Ok(())
}
fn validate_assets() -> Result<(), String> {
    check(CUBIN, CUBIN_SHA)?; check(TEMPLATE, TEMPLATE_SHA)?; check(ABI, ABI_SHA)?;
    let abi: serde_json::Value = serde_json::from_slice(ABI).map_err(|e| e.to_string())?;
    if abi["parameter_size"] != 368 || abi["parameter_alignment"] != 8
        || abi["pointer_offsets"] != serde_json::json!({"input":64,"weight":112,"source":192,"output":272})
        || abi["shape"] != serde_json::json!([5054,384,384]) || abi["grid"] != serde_json::json!([40,3,1])
        || abi["threads"] != 128 || abi["shared_bytes"] != 49152 || abi["alpha"] != 1 || abi["beta"] != 1 {
        return Err("outproj N128 ABI mismatch".into());
    }
    for offset in [64,112,192,272] { if TEMPLATE[offset..offset+8] != [0;8] { return Err("outproj template contains an address".into()); } }
    Ok(())
}
pub(crate) fn fingerprint() -> Option<String> {
    if !cfg!(all(windows, target_pointer_width="64")) { return None; }
    static ID: OnceLock<Option<String>> = OnceLock::new();
    ID.get_or_init(|| {
        validate_assets().ok()?;
        let mut hash = Sha256::new(); hash.update(b"outproj-classic-tn-n128-fp32-host-abi-r1\0");
        for bytes in [CUBIN, TEMPLATE.as_slice(), ABI, include_bytes!("outproj_n128.rs").as_slice(), include_bytes!("cuda_exec.rs").as_slice(), include_bytes!("cuda.rs").as_slice(), include_bytes!("../tactic_plan.rs").as_slice()] {
            hash.update((bytes.len() as u64).to_le_bytes()); hash.update(bytes);
        }
        Some(format!("sha256:{}", hex::encode(hash.finalize())))
    }).clone()
}
#[repr(C, align(8))]
struct Parameters([u8;368]);
// Compiler-exported, pointer-free Rust representation of one by-value C++ Params.
unsafe impl DeviceRepr for Parameters {}
fn parameters(input:u64, weight:u64, output:u64) -> Parameters {
    let mut bytes=*TEMPLATE;
    for (offset,pointer) in [(64,input),(112,weight),(192,output),(272,output)] { bytes[offset..offset+8].copy_from_slice(&pointer.to_le_bytes()); }
    Parameters(bytes)
}
fn disjoint(a:u64, alen:usize, b:u64, blen:usize) -> bool {
    a.checked_add(alen as u64).is_some_and(|end| end<=b) || b.checked_add(blen as u64).is_some_and(|end| end<=a)
}
pub(crate) struct OutprojN128 { _module:Arc<CudaModule>, function:CudaFunction }
impl OutprojN128 {
    pub(crate) fn load(graph:&LayerGraph, device:&Arc<CudaContext>) -> Result<Self,String> {
        if fingerprint().is_none() { return Err("outproj N128 requires the compiled Windows x64 artifact".into()); }
        crate::tactic_plan::validate_strict_attention_target(&super::device_fingerprint(device)?)?;
        if (graph.board_size,graph.mid_channels,graph.num_heads,graph.head_dim)!=(19,384,12,32) { return Err("outproj N128 requires TF3 S361/C384/H12/D32".into()); }
        let layers=graph.layers.iter().filter_map(|l| if let Layer::Attention(a)=l {Some(a)} else {None}).collect::<Vec<_>>();
        if layers.len()!=33 || layers.iter().any(|a| !a.residual_add || a.out_weight.dims!=vec![384,384] || (a.num_heads,a.head_dim,a.seq_len)!=(12,32,361)) { return Err("outproj N128 requires 33 TF3 residual output projections".into()); }
        let module=device.load_module(Ptx::from_binary(CUBIN.to_vec())).map_err(|e|format!("outproj CUBIN load: {e}"))?;
        let function=module.load_function("outproj_classic_tn_n128").map_err(|e|e.to_string())?;
        function.set_attribute(sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,49152).map_err(|e|e.to_string())?;
        if function.binary_version().map_err(|e|e.to_string())?!=120 || function.num_regs().map_err(|e|e.to_string())?!=226 || function.local_size_bytes().map_err(|e|e.to_string())?!=0 || function.max_threads_per_block().map_err(|e|e.to_string())?<128 { return Err("outproj N128 loaded resources mismatch".into()); }
        Ok(Self { _module:module, function })
    }
    pub(crate) fn launch(&self,stream:&Arc<CudaStream>,input:&CudaSlice<u16>,weight:&CudaSlice<u16>,output:&mut CudaSlice<f32>) -> Result<(),String> {
        if input.len()<ROWS*CHANNELS || weight.len()!=CHANNELS*CHANNELS || output.len()<ROWS*CHANNELS { return Err("outproj N128 requires B14/N384/K384 complete buffers".into()); }
        let (a,_a)=input.device_ptr(stream); let (b,_b)=weight.device_ptr(stream); let (y,_y)=output.device_ptr_mut(stream);
        if [a,b,y].iter().any(|p| *p==0 || p%16!=0) || !disjoint(a,ROWS*CHANNELS*2,y,ROWS*CHANNELS*4) || !disjoint(b,CHANNELS*CHANNELS*2,y,ROWS*CHANNELS*4) { return Err("outproj N128 needs aligned disjoint input/weight and FP32 output".into()); }
        let params=parameters(a,b,y);
        unsafe { stream.launch_builder(&self.function).arg(&params).launch(LaunchConfig { grid_dim:(40,3,1), block_dim:(128,1,1), shared_mem_bytes:49152 }) }.map_err(|e|format!("outproj N128 launch: {e}"))?;
        Ok(())
    }
}
pub(crate) fn report(batch:usize,requested:bool) {
    static SEEN:[OnceLock<()>;16]=[const {OnceLock::new()};16];
    if !(1..=16).contains(&batch) {return;}
    SEEN[batch-1].get_or_init(|| {
        let effective=requested && batch==14;
        eprintln!("[cuda-tactic] name=outproj_classic_n128 requested={} effective={} physical_batch={} launch={} grid={} block=128 output=f32 identity=matched",u8::from(requested),u8::from(effective),batch,if effective {"classic-tn-n128"} else {"existing"},if effective {"40x3"} else {"existing"});
    });
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn assets_and_parameter_layout() {
        validate_assets().unwrap(); assert_eq!(std::mem::size_of::<Parameters>(),368); assert_eq!(std::mem::align_of::<Parameters>(),8);
        let p=parameters(0x1110,0x2220,0x3330); let mut expected=*TEMPLATE;
        for (o,v) in [(64,0x1110u64),(112,0x2220),(192,0x3330),(272,0x3330)] { expected[o..o+8].copy_from_slice(&v.to_le_bytes()); }
        assert_eq!(p.0,expected);let mut corrupt=CUBIN.to_vec();corrupt[0]^=1;assert!(check(&corrupt,CUBIN_SHA).is_err());
    }
    #[test] fn aliasing_and_overflow_are_rejected() { assert!(!disjoint(16,1024,32,1024));assert!(disjoint(16,1024,1040,1024));assert!(!disjoint(u64::MAX-4,16,u64::MAX-2,16)); }
}
