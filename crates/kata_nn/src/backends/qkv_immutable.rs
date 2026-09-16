//! Exact B14 CUTLASS QKV/RoPE CUBIN with a compiler-exported parameter image.
use std::collections::BTreeSet;
use std::sync::{Arc,Mutex,OnceLock};
use cudarc::driver::{CudaContext,CudaFunction,CudaModule,CudaSlice,CudaStream,DevicePtr,DevicePtrMut,DeviceRepr,LaunchConfig,PushKernelArg,sys};
use cudarc::nvrtc::Ptx;
use sha2::{Digest,Sha256};
const CUBIN:&[u8]=include_bytes!("../../cuda-aot/qkv-immutable-params-b14-r1/kernel.sm120.cubin");
const TEMPLATE:&[u8;400]=include_bytes!("../../cuda-aot/qkv-immutable-params-b14-r1/params-template.bin");
const ABI:&[u8]=include_bytes!("../../cuda-aot/qkv-immutable-params-b14-r1/abi.json");
const CUBIN_SHA:&str="faf6f3136db74a02b75c99c993ef344f44ec521c9f95eaca69e361830f76fbf2";
const TEMPLATE_SHA:&str="108fe87b8aaff0057016dfd37350cad9990b5ce17ffad013f504ae8550cb0cd1";
const ABI_SHA:&str="66bd94815619849643cdb6f9a472fd3730f5bf1d77b34f522b626eaa4e3d1b11";
const SYMBOL:&str="_Z18immutable_qkv_ropeILb1EEvN7cutlass4gemm6kernel4GemmINS1_11threadblock13MmaMultistageINS1_9GemmShapeILi128ELi64ELi32EEENS0_9transform11threadblock28PredicatedTileAccessIteratorINS0_11MatrixShapeILi128ELi32EEENS0_6half_tENS0_6layout8RowMajorELi1ENS8_29PitchLinearWarpRakedThreadMapINS0_16PitchLinearShapeILi32ELi128EEELi128ENSH_ILi4ELi8EEELi8EEENS0_5ArrayISD_Li8ELb0EEELb0ENSE_9NoPermuteEEENS9_25RegularTileAccessIteratorISC_SD_NSE_37RowMajorTensorOpMultiplicandCrosswiseILi16ELi32EEELi0ESK_Li16EEELNS0_4arch14CacheOperation4KindE1ENSA_INSB_ILi32ELi64EEESD_NSE_11ColumnMajorELi0ENSG_INSH_ILi32ELi64EEELi128ESJ_Li8EEESM_Lb0ESN_EENSP_ISW_SD_NSE_40ColumnMajorTensorOpMultiplicandCrosswiseILi16ELi32EEELi1ESZ_Li16EEELSV_1EfSF_NS4_9MmaPolicyINS1_4warp11MmaTensorOpINS6_ILi64ELi32ELi32EEESD_SR_SD_S12_fSF_NS15_17MmaTensorOpPolicyINST_3MmaINS6_ILi16ELi8ELi16EEELi32ESD_SF_SD_SX_fSF_NST_13OpMultiplyAddEEENSB_ILi1ELi1EEEEELi1ELb0EbEENSB_ILi0ELi0EEES1G_Li1EEELi3ELNS1_23SharedMemoryClearOptionE0EbEENS0_8epilogue11threadblock8EpilogueIS7_S1F_Li1E17RopeStoreIteratorINS1L_22PredicatedTileIteratorINS1L_26OutputTileOptimalThreadMapINS1L_15OutputTileShapeILi64ELi8ELi2ELi1ELi1EEENS1Q_ILi1ELi8ELi1ELi1ELi8EEELi128ELi8ELi16EEESD_Lb0ESN_Lb0EEEENS1K_4warp24FragmentIteratorTensorOpIS17_S1A_fNSL_IfLi4ELb1EEESF_EENS1W_25TileIteratorTensorOpMixedIS17_S1A_fLi32ELi16ELi8ELi8ELb0EEENS1L_23SharedLoadIteratorMixedINS1T_18CompactedThreadMapEfLi32ELi16ELi8ELi8ELb0EEENS1K_6thread17LinearCombinationISD_Li8EffLNS25_9ScaleType4KindE0ELNS0_15FloatRoundStyleE2ESD_EENSB_ILi0ELi8EEELi2ELi1EEENS4_30GemmIdentityThreadblockSwizzleILi1EEELb0EE6ParamsE";
const N128_CUBIN:&[u8]=include_bytes!("../../cuda-aot/qkv-classic-n128-b14-r1/kernel.sm120.cubin");
const N128_SHA:&str="f4138e27b303a8321082441a22312e8888f2847ba84ad11c920a5e9f03d4db89";
const N128_SYMBOL:&str="classic_n128_qkv_rope";
const N128_TEMPLATE:&[u8;400]=include_bytes!("../../cuda-aot/qkv-classic-n128-b14-r1/params-template.bin");
const N128_ABI:&[u8]=include_bytes!("../../cuda-aot/qkv-classic-n128-b14-r1/abi.json");
const N128_TEMPLATE_SHA:&str="9672dc7c9d68e8ea3b2074042f563e7c4b5bd7b5723c69c0784bbd90fac0cf41";
const N128_ABI_SHA:&str="fe0b0e6847203428dcb5d0fdc3332eab60788017f24bdb71e2c90b7c69d12978";
const PACKED:usize=5054*1152;
fn check(bytes:&[u8],expected:&str)->Result<(),String>{
    if hex::encode(Sha256::digest(bytes))!=expected {return Err("QKV epilogue artifact hash mismatch".into());}Ok(())
}
fn validate_assets()->Result<(),String>{
    check(CUBIN,CUBIN_SHA)?;check(N128_CUBIN,N128_SHA)?;check(N128_TEMPLATE,N128_TEMPLATE_SHA)?;check(N128_ABI,N128_ABI_SHA)?;check(TEMPLATE,TEMPLATE_SHA)?;check(ABI,ABI_SHA)?;
    let v:serde_json::Value=serde_json::from_slice(ABI).map_err(|e|e.to_string())?;
    if v["parameter_size"]!=400 || v["parameter_alignment"]!=8 || v["pointer_offsets"]!=serde_json::json!({"input":64,"weight":112,"source":208,"output":304,"cos":288,"sin":296}) {return Err("QKV epilogue ABI mismatch".into());}
    for offset in [64,112,208,304,288,296] {if TEMPLATE[offset..offset+8]!=[0;8]{return Err("QKV epilogue template contains an address".into());}}
    let n:serde_json::Value=serde_json::from_slice(N128_ABI).map_err(|e|e.to_string())?;
    if n["parameter_size"]!=400 || n["parameter_alignment"]!=8 || n["pointer_offsets"]!=v["pointer_offsets"] || n["grid"]!=serde_json::json!([40,9,1]) || n["shape"]!=serde_json::json!([5054,1152,384]) || n["shared_bytes"]!=49152 || n["threads"]!=128 {return Err("QKV N128 ABI or launch geometry mismatch".into());}
    for offset in [64,112,208,304,288,296] {if N128_TEMPLATE[offset..offset+8]!=[0;8]{return Err("QKV N128 template contains an address".into());}}
    Ok(())
}
pub(crate) fn fingerprint()->Option<String>{
    if !cfg!(all(windows,target_pointer_width="64")){return None;}
    static ID:OnceLock<Option<String>>=OnceLock::new();
    ID.get_or_init(||{validate_assets().ok()?;let mut h=Sha256::new();h.update(b"qkv-immutable-params-b14-host-abi-r1\0");
        for b in [CUBIN,N128_CUBIN,N128_TEMPLATE.as_slice(),N128_ABI,TEMPLATE.as_slice(),ABI,include_bytes!("qkv_immutable.rs").as_slice(),include_bytes!("strict_attention.rs").as_slice(),include_bytes!("cuda_exec.rs").as_slice()]{h.update((b.len() as u64).to_le_bytes());h.update(b);}
        Some(format!("sha256:{}",hex::encode(h.finalize())))
    }).clone()
}
#[repr(C,align(8))]
struct Parameters([u8;400]);
// The C++ exporter and PTX agree on all 400 bytes, alignment and six pointer
// offsets. No Rust pointer/reference or destructor appears in this by-value ABI.
unsafe impl DeviceRepr for Parameters {}
fn parameters(n128:bool,input:u64,weight:u64,output:u64,cos:u64,sin:u64)->Parameters{
    let mut bytes=if n128 {*N128_TEMPLATE}else{*TEMPLATE};
    for (offset,pointer) in [(64,input),(112,weight),(208,output),(304,output),(288,cos),(296,sin)]{bytes[offset..offset+8].copy_from_slice(&pointer.to_le_bytes());}
    Parameters(bytes)
}
fn disjoint(a:u64,alen:usize,b:u64,blen:usize)->bool{a.checked_add(alen as u64).is_some_and(|end|end<=b)||b.checked_add(blen as u64).is_some_and(|end|end<=a)}
pub(crate) struct QkvImmutable{_module:Arc<CudaModule>,function:CudaFunction,n128:bool,reported:Mutex<BTreeSet<usize>>}
impl QkvImmutable{
    pub(crate) fn load(device:&Arc<CudaContext>)->Result<Self,String>{
        if fingerprint().is_none(){return Err("QKV epilogue requires the compiled Windows x64 artifact".into());}
        crate::tactic_plan::validate_strict_attention_target(&crate::backends::cuda::device_fingerprint(device)?)?;
        let n128=crate::tactic_plan::qkv_n128_requested()?;
        let code=if n128 {N128_CUBIN} else {CUBIN};
        let symbol=if n128 {N128_SYMBOL} else {SYMBOL};
        let module=device.load_module(Ptx::from_binary(code.to_vec())).map_err(|e|format!("QKV CUBIN load: {e}"))?;
        let function=module.load_function(symbol).map_err(|e|format!("QKV symbol: {e}"))?;
        function.set_attribute(sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,if n128 {49152}else{36864}).map_err(|e|e.to_string())?;
        if function.binary_version().map_err(|e|e.to_string())?!=120 || function.num_regs().map_err(|e|e.to_string())?!=if n128 {226}else{146} || function.local_size_bytes().map_err(|e|e.to_string())?!=0 || function.max_threads_per_block().map_err(|e|e.to_string())?<128{return Err("QKV loaded kernel resources differ from measured artifact".into());}
        Ok(Self{_module:module,function,n128,reported:Mutex::new(BTreeSet::new())})
    }
    pub(crate) fn validate_selection(&self)->Result<(),String>{
        if self.n128!=crate::tactic_plan::qkv_n128_requested()?{return Err("QKV n128 selection changed after model preparation".into());}Ok(())
    }
    pub(crate) fn launch(&self,stream:&Arc<CudaStream>,input:&CudaSlice<u16>,weight:&CudaSlice<u16>,out:&mut CudaSlice<u16>,cos:&CudaSlice<f32>,sin:&CudaSlice<f32>)->Result<(),String>{
        if input.len()<5054*384 || weight.len()!=1152*384 || out.len()<PACKED || cos.len()!=361*192 || sin.len()!=361*192{return Err("QKV epilogue requires exact B14/N1152/K384 weights and RoPE tables".into());}
        let (a,_a)=input.device_ptr(stream);let (b,_b)=weight.device_ptr(stream);let (y,_y)=out.device_ptr_mut(stream);let (co,_co)=cos.device_ptr(stream);let (sn,_sn)=sin.device_ptr(stream);
        if [a,b,y].iter().any(|p|p%16!=0) || !disjoint(a,5054*384*2,y,PACKED*2) || !disjoint(b,1152*384*2,y,PACKED*2) || !disjoint(co,361*192*4,y,PACKED*2) || !disjoint(sn,361*192*4,y,PACKED*2){return Err("QKV epilogue buffers require alignment and a disjoint output".into());}
        let params=parameters(self.n128,a,b,y,co,sn);
        unsafe{stream.launch_builder(&self.function).arg(&params).launch(LaunchConfig{grid_dim:(40,if self.n128 {9}else{18},1),block_dim:(128,1,1),shared_mem_bytes:if self.n128 {49152}else{36864}})}.map_err(|e|format!("QKV epilogue launch: {e}"))?;
        Ok(())
    }
    pub(crate) fn report_scope(&self,batch:usize){if self.reported.lock().unwrap().insert(batch){
        eprintln!("[cuda-tactic] name=qkv_classic_n128 requested={} effective={} physical_batch={} launch={} artifact={} identity=matched",u8::from(self.n128),u8::from(self.n128 && batch==14),batch,if batch!=14 {"existing"}else if self.n128 {"n128-register-rope"}else {"immutable-register-rope"},fingerprint().unwrap());
        if batch==14 {eprintln!("[cuda-tactic] name=qkv_immutable requested=1 effective=1 launch=immutable-register-rope physical_batch=14 grid=40x{} block=128 abi=1 artifact={} identity=matched",if self.n128 {9}else{18},fingerprint().unwrap());}
        else {eprintln!("[cuda-tactic] name=qkv_immutable requested=1 effective=0 launch=existing physical_batch={batch} reason=physical-batch-not-14");}
    }}
}
#[cfg(test)]mod tests{
    use super::*;
    #[test]fn exact_assets_and_parameter_patch(){validate_assets().unwrap();assert_eq!(std::mem::size_of::<Parameters>(),400);assert_eq!(std::mem::align_of::<Parameters>(),8);for n128 in [false,true] {let p=parameters(n128,0x1110,0x2220,0x3330,0x4440,0x5550);let mut expected=if n128 {*N128_TEMPLATE}else{*TEMPLATE};for (o,v) in [(64,0x1110u64),(112,0x2220),(208,0x3330),(304,0x3330),(288,0x4440),(296,0x5550)]{expected[o..o+8].copy_from_slice(&v.to_le_bytes());}assert_eq!(p.0,expected);}assert_ne!(TEMPLATE,N128_TEMPLATE);let mut changed=CUBIN.to_vec();changed[0]^=1;assert!(check(&changed,CUBIN_SHA).is_err());}
    #[test]fn overlapping_packed_buffers_are_refused(){assert!(!disjoint(16,1024,32,1024));assert!(disjoint(16,1024,1040,1024));assert!(!disjoint(u64::MAX-4,16,u64::MAX-2,16));}
}
