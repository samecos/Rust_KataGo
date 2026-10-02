//! Synthetic, finite INT8 cache diagnostic. No B11/B15 model or performance claim.
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::{Path, PathBuf}};
type Result<T> = std::result::Result<T, String>;
const PURPOSE: &str = "SYNTHETIC_PARTIAL_INT8_CACHE_DIAGNOSTIC_NOT_MODEL_OR_PERFORMANCE";
fn hash(bytes: &[u8]) -> String { hex::encode(Sha256::digest(bytes)) }
fn require(ok: bool, message: &str) -> Result<()> { if ok { Ok(()) } else { Err(message.into()) } }
fn save(path: &Path, bytes: &[u8]) -> Result<String> {
    let mut f = fs::OpenOptions::new().write(true).create_new(true).open(path).map_err(|e| e.to_string())?;
    f.write_all(bytes).map_err(|e| e.to_string())?;
    f.sync_all().map_err(|e| e.to_string())?;
    Ok(hash(bytes))
}
fn save_json(path: &Path, value: &impl Serialize) -> Result<String> {
    save(path, &serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?)
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Source { logical_name: String, path: PathBuf, bytes: u64, sha256: String }
impl Source {
    fn verify(&self) -> Result<()> {
        let b = fs::read(&self.path).map_err(|e| e.to_string())?;
        require(b.len() as u64 == self.bytes && hash(&b) == self.sha256, &format!("source changed: {}", self.logical_name))
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Provenance { schema: String, purpose: String, build_mode:String, sources: Vec<Source>, header: Source, static_cudart: Source }
impl Provenance {
    fn verify(&self) -> Result<()> {
        require(self.schema == "rustgo-synthetic-int8-probe-provenance-v1" && self.purpose == PURPOSE, "provenance schema/purpose mismatch")?;
        require(matches!(self.build_mode.as_str(),"repository-example"|"standalone"),"invalid provenance build mode")?;
        let required=if self.build_mode=="standalone" {"synthetic-probe/Cargo.lock"} else {"crates/kata_nn/examples/int8_algorithm_probe.rs"};
        require(self.sources.iter().any(|s|s.logical_name==required),"actual build lock/example evidence missing")?;
        require(!self.sources.is_empty() && self.sources.len() <= 60, "source inventory invalid")?;
        let mut last = "";
        for s in &self.sources {
            require(s.logical_name.as_str() > last && !s.logical_name.contains(['\\', ':'])
                && !s.logical_name.starts_with('/') && !s.logical_name.split('/').any(|x| x == ".." || x.is_empty()), "invalid source order/path")?;
            s.verify()?; last = &s.logical_name;
        }
        self.header.verify()?; self.static_cudart.verify()
    }
}
#[derive(Clone, Copy, Serialize)]
struct Shape { batch: usize, rows: usize, n: usize, k: usize, kp: usize, stride: usize }
fn shapes() -> Vec<Shape> {
    [1usize, 3].into_iter().flat_map(|batch| [(384usize,1152usize), (512,1160), (512,1168)].into_iter()
        .map(move |(n,k)| Shape { batch, rows: batch * 361, n, k, kp: k.div_ceil(16) * 16, stride: k.div_ceil(16) * 16 })).collect()
}
fn input_value(slot: usize, row: usize, col: usize, k: usize) -> f32 {
    // The two logical aliases have identical useful columns. K1160 padding is
    // deliberately nonzero, while K1168 extra columns are genuine zero input.
    if col >= 1160 { return if k == 1160 { 13.0 } else { 0.0 }; }
    (((row * 29 + col * 17 + slot * 43) % 251) as i32 - 125) as f32 / 128.0
}
fn weight_value(row: usize, col: usize) -> f32 {
    if col >= 1160 { 0.0 } else { (((row * 11 + col * 7) % 127) as i32 - 63) as f32 / 1024.0 }
}
fn u16_bytes(values: &[u16]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }
fn f32_bytes(values: &[f32]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }

#[cfg(all(feature="cuda", target_os="windows"))]
mod gpu {
    use super::*;
    use std::{ffi::c_void, sync::Arc};
    use cudarc::driver::{CudaContext, CudaSlice, CudaStream, sys as ds};
    use kata_nn::backends::{cuda::{self, CudaRuntime}, int8::{self, Int8Kernels, Int8Weight, Int8Workspace, QuantizedWeights}, int8_algorithm_plan as plan};
    thread_local! { static PROJECTION_CALLS:std::cell::Cell<usize> = const {std::cell::Cell::new(0)}; }
    fn projection_attempt()->Result<()> {
        PROJECTION_CALLS.with(|calls| {let next=calls.get()+1; require(next<=80,"finite projection-call budget exceeded")?; calls.set(next); Ok(())})
    }

    #[link(name="kernel32")]
    unsafe extern "system" {
        fn GetModuleHandleExW(flags: u32, address: *const u16, module: *mut *mut c_void) -> i32;
        fn GetModuleFileNameW(module: *mut c_void, buffer: *mut u16, size: u32) -> u32;
    }
    fn module_source(name: &str, address: *const c_void) -> Result<Source> {
        let mut module = std::ptr::null_mut();
        // FROM_ADDRESS | UNCHANGED_REFCOUNT: the real DLL symbol, never the Rust wrapper.
        require(unsafe { GetModuleHandleExW(6, address.cast(), &mut module) } != 0, "GetModuleHandleExW failed")?;
        let mut path = vec![0u16; 32768];
        let n = unsafe { GetModuleFileNameW(module, path.as_mut_ptr(), path.len() as u32) } as usize;
        require(n > 0 && n < path.len(), "GetModuleFileNameW failed/truncated")?;
        let path = PathBuf::from(String::from_utf16(&path[..n]).map_err(|e| e.to_string())?);
        let raw = fs::read(&path).map_err(|e| e.to_string())?;
        Ok(Source { logical_name: name.into(), path, bytes: raw.len() as u64, sha256: hash(&raw) })
    }
    fn loaded_libraries() -> Result<(Source, Source)> {
        unsafe {
            let lt = cudarc::cublaslt::sys::culib().get::<unsafe extern "C" fn()->usize>(b"cublasLtGetVersion\0").map_err(|e|e.to_string())?;
            let blas = cudarc::cublas::sys::culib().get::<unsafe extern "C" fn(cudarc::cublas::sys::cublasHandle_t,*mut i32)->cudarc::cublas::sys::cublasStatus_t>(b"cublasGetVersion_v2\0").map_err(|e|e.to_string())?;
            Ok((module_source("loaded/cublasLt", *lt as *const c_void)?, module_source("loaded/cublas", *blas as *const c_void)?))
        }
    }
    struct Fixture { shape: Shape, input: Vec<u16>, weights: Vec<f32> }
    struct DeviceFixture { shape: Shape, input: CudaSlice<u16>, weight: Int8Weight, output: CudaSlice<u16> }
    fn fixtures(slot: usize, out: &Path) -> Result<Vec<Fixture>> {
        let mut fixtures = vec![];
        let mut manifest = vec![];
        for (i, s) in shapes().into_iter().enumerate() {
            let input = (0..s.rows).flat_map(|row| (0..s.stride).map(move |col| cuda::f32_to_f16_bits(input_value(slot,row,col,s.k)))).collect::<Vec<_>>();
            let weights = (0..s.n).flat_map(|row| (0..s.k).map(move |col| weight_value(row,col))).collect::<Vec<_>>();
            require(input.iter().any(|&v|v != 0) && weights.iter().any(|&v| v != 0.0), "degenerate fixture")?;
            require(input.iter().all(|&v|cuda::f16_to_f32_bits(v).is_finite()), "nonfinite fixture")?;
            let q = QuantizedWeights::new(&weights,s.n,s.k)?;
            let input_name = format!("slot{slot}-shape{i}-input.u16le");
            let weight_name = format!("slot{slot}-shape{i}-weights.f32le");
            let quant_name = format!("slot{slot}-shape{i}-weights.i8");
            let scale_name = format!("slot{slot}-shape{i}-scales.f32le");
            manifest.push(json!({"shape":s,
                "input":{"path":input_name,"sha256":save(&out.join(&input_name),&u16_bytes(&input))?},
                "weights":{"path":weight_name,"sha256":save(&out.join(&weight_name),&f32_bytes(&weights))?},
                "quantized":{"path":quant_name,"sha256":save(&out.join(&quant_name),&q.values.iter().map(|&v|v as u8).collect::<Vec<_>>())?},
                "scales":{"path":scale_name,"sha256":save(&out.join(&scale_name),&f32_bytes(&q.scales))?}}));
            fixtures.push(Fixture { shape:s,input,weights });
        }
        save_json(&out.join(format!("slot{slot}-fixtures.json")), &json!({"purpose":PURPOSE,"slot":slot,"fixtures":manifest}))?;
        Ok(fixtures)
    }
    fn upload(stream: &Arc<CudaStream>, fixture: &Fixture) -> Result<DeviceFixture> {
        let s = fixture.shape;
        Ok(DeviceFixture { shape:s, input:stream.clone_htod(&fixture.input).map_err(|e|e.to_string())?,
            weight:Int8Weight::upload(stream,&fixture.weights,s.n,s.k)?,
            output:stream.alloc_zeros(s.rows*s.n).map_err(|e|e.to_string())? })
    }
    fn workspace(kernels: &Int8Kernels, stream: &Arc<CudaStream>) -> Result<Int8Workspace> {
        Int8Workspace::from_kernels(kernels,stream,1083,1168,512)
    }
    fn project(rt: &CudaRuntime, stream: &Arc<CudaStream>, ws: &mut Int8Workspace, f: &mut DeviceFixture) -> Result<()> {
        projection_attempt()?;
        ws.project_half(rt,stream,&f.input,f.shape.stride,&f.weight,&mut f.output,f.shape.rows)
    }
    fn poison_outputs(stream:&Arc<CudaStream>, fixtures:&mut [DeviceFixture])->Result<()> {
        for f in fixtures {
            stream.memcpy_htod(&vec![0x7e00u16;f.output.len()],&mut f.output).map_err(|e|e.to_string())?;
        }
        stream.synchronize().map_err(|e|e.to_string())
    }
    fn output(stream:&Arc<CudaStream>, f:&DeviceFixture) -> Result<Vec<u16>> {
        let result = stream.clone_dtoh(&f.output).map_err(|e|e.to_string())?;
        require(result.iter().all(|&x|cuda::f16_to_f32_bits(x).is_finite()) && result.iter().any(|&x| x & 0x7fff != 0), "nonfinite/zero projection")?;
        Ok(result)
    }
    fn write_outputs(out:&Path, slot:usize, stage:&str, values:&[Vec<u16>], expected:Option<&[Vec<u16>]>, events:&mut Vec<Value>) -> Result<()> {
        for (i,value) in values.iter().enumerate() {
            let name = format!("slot{slot}-{stage}-shape{i}.u16le");
            let digest = save(&out.join(&name),&u16_bytes(value))?;
            if let Some(expected) = expected { require(*value == expected[i], &format!("slot{slot}/{stage}/shape{i} output differs"))?; }
            events.push(json!({"kind":"output","slot":slot,"stage":stage,"shape_index":i,"path":name,"sha256":digest,"elements":value.len(),"bitwise_equal":expected.map(|_|true)}));
        }
        // Same useful data, different logical K and nonzero ignored padding.
        require(values[1] == values[2] && values[4] == values[5], "logical alias/padding outputs differ")
    }
    fn expect_error<T>(name:&str, result:Result<T>, needle:&str, events:&mut Vec<Value>) -> Result<()> {
        let error = result.err().ok_or_else(||format!("negative {name} unexpectedly succeeded"))?;
        require(error.contains(needle), &format!("negative {name} wrong rejection: {error}"))?;
        events.push(json!({"kind":"expected_rejection","name":name,"error":error})); Ok(())
    }
    fn assert_poison(rt:&CudaRuntime, stream:&Arc<CudaStream>, ws:&mut Int8Workspace, f:&mut DeviceFixture, name:&str, events:&mut Vec<Value>) -> Result<()> {
        expect_error(&format!("{name}/valid-operation-after-error"),project(rt,stream,ws,f),"poisoned",events)?;
        expect_error(&format!("{name}/export-after-error"),ws.export_algorithm_plan(stream),"poisoned",events)
    }
    struct ProbeGraph { graph:ds::CUgraph, exec:ds::CUgraphExec, stream:Arc<CudaStream> }
    impl Drop for ProbeGraph {
        fn drop(&mut self) { unsafe {
            let _=self.stream.context().bind_to_thread();
            if !self.exec.is_null(){let _=cudarc::driver::result::graph::exec_destroy(self.exec);}
            if !self.graph.is_null(){let _=cudarc::driver::result::graph::destroy(self.graph);}
        }}
    }
    impl ProbeGraph {
        fn launch(&self)->Result<()> { self.stream.context().bind_to_thread().map_err(|e|e.to_string())?;
            unsafe { cudarc::driver::result::graph::launch(self.exec,self.stream.cu_stream()).map_err(|e|e.to_string()) }
        }
    }
    struct CaptureCleanup { stream:Arc<CudaStream>, active:bool }
    impl Drop for CaptureCleanup {
        fn drop(&mut self) {if self.active {unsafe {
            if let Ok(g)=cudarc::driver::result::stream::end_capture(self.stream.cu_stream()) {
                if !g.is_null(){let _=cudarc::driver::result::graph::destroy(g);}
            }
        }}}
    }
    fn capture(stream:&Arc<CudaStream>, operation:impl FnOnce()->Result<()>) -> Result<ProbeGraph> {
        stream.begin_capture(ds::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED).map_err(|e|e.to_string())?;
        let mut cleanup=CaptureCleanup {stream:stream.clone(),active:true};
        let result = operation();
        // Raw end first: do not instantiate a failed capture, and do not leak
        // the graph when instantiate fails (cudarc 0.19.9 safe API can leak it).
        let ended=unsafe {cudarc::driver::result::stream::end_capture(stream.cu_stream())};
        cleanup.active=false;
        let raw=ended.map_err(|e|format!("CAPTURE_CLEANUP_FAILED: {e}"))?;
        let mut graph=ProbeGraph {graph:raw,exec:std::ptr::null_mut(),stream:stream.clone()};
        require(stream.capture_status().map_err(|e|e.to_string())?==ds::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE,"CAPTURE_CLEANUP_FAILED: still capturing")?;
        result?;
        require(!raw.is_null(),"empty captured graph")?;
        graph.exec=unsafe {cudarc::driver::result::graph::instantiate(raw,cuda::graph_instantiate_flags()).map_err(|e|e.to_string())?};
        Ok(graph)
    }
    fn foreign_context() -> Result<Arc<CudaContext>> {
        // cudarc's convenience new_non_primary is cfg-excluded for CUDA 13.3.
        let mut device = 0;
        unsafe { ds::cuDeviceGet(&mut device,0).result().map_err(|e|e.to_string())?; }
        let mut raw = std::ptr::null_mut();
        unsafe {
            ds::cuCtxCreate_v4(&mut raw,std::ptr::null_mut(),0,device).result().map_err(|e|e.to_string())?;
            CudaContext::from_raw_context(0,device,raw).map_err(|e|e.to_string())
        }
    }
    pub(super) fn run(out:&Path, provenance:&Provenance, events:&mut Vec<Value>) -> Result<Value> {
        let all_env: std::collections::BTreeMap<String,String> = std::env::vars().filter(|(k,_)| k.starts_with("KATAGO_")).collect();
        require(all_env.get("KATAGO_CUDA_INT8_GEMM_TUNE").is_none_or(|v|v=="0"),"probe prohibits GEMM tune")?;
        for key in ["KATAGO_CUDA_PROFILE","KATAGO_CUDA_DEBUG_LAYER","KATAGO_CUDA_DUMP_INPUT"] { require(!all_env.contains_key(key),"probe prohibits profiling/debug/dump environment")?; }
        let host = [fixtures(0,out)?,fixtures(1,out)?];
        let model_sha = save_json(&out.join("synthetic-model.json"),&json!({"purpose":PURPOSE,"fixture_manifests":[hash(&fs::read(out.join("slot0-fixtures.json")).map_err(|e|e.to_string())?),hash(&fs::read(out.join("slot1-fixtures.json")).map_err(|e|e.to_string())?)]}))?;
        let graph_sha = save_json(&out.join("synthetic-graph.json"),&json!({"purpose":PURPOSE,"operations":"six independent row-major half to INT8 projections, no model graph","shapes":shapes()}))?;
        let recipe_sha = save_json(&out.join("synthetic-recipe.json"),&json!({"purpose":PURPOSE,"weights":"symmetric per-output INT8","activation":"dynamic per-row INT8","result":"half","tuning":false}))?;
        let policy_sha = save_json(&out.join("execution-policy.json"),&json!({"purpose":PURPOSE,"slots":2,"batches":[1,3],"stages":["legacy","record","restore","graph-replay-once"],"environment":all_env,"max_projection_calls_including_capture_and_rejected":80,"performance_measurement":false}))?;
        let rt = CudaRuntime::new()?;
        let kernels = Int8Kernels::load(&rt)?;
        let (lt_library,blas_library) = loaded_libraries()?;
        let library_evidence = json!({"cublaslt":lt_library,"cublas":blas_library,"loaded_symbol_resolution":"cudarc culib exact symbol -> GetModuleHandleExW FROM_ADDRESS", "static_cudart":provenance.static_cudart,"static_cudart_scope":"build input archive, not a loaded DLL or independent embedded object attestation; projection uses driver/LT", "header":provenance.header});
        let runtime_provenance = save_json(&out.join("libraries.json"),&library_evidence)?;
        let build_sha = save_json(&out.join("backend-build.json"),&cuda::backend_build_fingerprint())?;
        let source_sha = save_json(&out.join("source-evidence.json"),&provenance.sources)?;
        let exe = std::env::current_exe().map_err(|e|e.to_string())?;
        let executable_sha = hash(&fs::read(&exe).map_err(|e|e.to_string())?);
        let uuid = rt.device.uuid().map_err(|e|e.to_string())?;
        let u = hex::encode(uuid.bytes.map(|v|v as u8));
        let mut driver = 0;
        unsafe { ds::cuDriverGetVersion(&mut driver).result().map_err(|e|e.to_string())?; }
        let (major,minor) = rt.device.compute_capability().map_err(|e|e.to_string())?;
        let fp = cuda::device_fingerprint(&rt.device)?;
        save_json(&out.join("device.json"),&json!({"gpu_name":fp.gpu_name,"compute_capability":fp.compute_capability,"sm_count":fp.sm_count,"l2_cache_bytes":fp.l2_cache_bytes,"driver_api":driver,"executable":exe,"executable_sha256":executable_sha}))?;
        let binding = plan::Binding { model_sha256:model_sha,graph_sha256:graph_sha,recipe_sha256:recipe_sha,
            execution_policy_sha256:policy_sha,executable_sha256:executable_sha,host_source_sha256:source_sha,
            kernel_build_sha256:build_sha,device_fingerprint_sha256:int8::algorithm_device_fingerprint_sha256(&rt)?,
            gpu_uuid:format!("GPU-{}-{}-{}-{}-{}",&u[..8],&u[8..12],&u[12..16],&u[16..20],&u[20..]),
            compute_capability:[major as u32,minor as u32],sm_count:fp.sm_count,driver_api_version:driver,
            cublaslt_version:unsafe { cudarc::cublaslt::sys::cublasLtGetVersion() } as u64,
            cublaslt_binary_sha256:lt_library.sha256.clone(),cublas_binary_sha256:blas_library.sha256.clone(),
            cuda_runtime_provenance_sha256:runtime_provenance,host_abi:format!("{}-{}-msvc",std::env::consts::ARCH,std::env::consts::OS),
            pointer_width_bits:usize::BITS,little_endian:cfg!(target_endian="little"),
            algo_size_bytes:std::mem::size_of::<cudarc::cublaslt::sys::cublasLtMatmulAlgo_t>() as u32,
            algo_alignment_bytes:std::mem::align_of::<cudarc::cublaslt::sys::cublasLtMatmulAlgo_t>() as u32,
            sources:provenance.sources.iter().map(|s|plan::SourceIdentity{logical_name:s.logical_name.clone(),sha256:s.sha256.clone()}).collect() };
        binding.validate()?; save_json(&out.join("binding.json"),&binding)?;
        let inventory = shapes().into_iter().map(|s|plan::ProblemKey::new(s.rows,s.n,s.k,s.kp)).collect::<Result<Vec<_>>>()?;
        plan::validate_inventory(&inventory)?;
        let streams = [rt.device.new_stream().map_err(|e|e.to_string())?,rt.device.new_stream().map_err(|e|e.to_string())?];
        require(streams[0].cu_stream()!=streams[1].cu_stream(),"slots unexpectedly share stream")?;
        let mut devices = [host[0].iter().map(|f|upload(&streams[0],f)).collect::<Result<Vec<_>>>()?,host[1].iter().map(|f|upload(&streams[1],f)).collect::<Result<Vec<_>>>()?];
        rt.device.synchronize().map_err(|e|e.to_string())?;
        let mut reference_tables: Vec<Vec<u8>> = vec![];
        let mut restored_slots = vec![];
        let mut references = vec![];
        for slot in 0..2 {
            let stream = &streams[slot]; let data = &mut devices[slot];
            let mut legacy = workspace(&kernels,stream)?;
            poison_outputs(stream,data)?;
            for f in data.iter_mut() { project(&rt,stream,&mut legacy,f)?; }
            let baseline = data.iter().map(|f|output(stream,f)).collect::<Result<Vec<_>>>()?;
            write_outputs(out,slot,"legacy",&baseline,None,events)?;
            let mut recording = workspace(&kernels,stream)?;
            recording.begin_algorithm_recording(&rt,stream,binding.clone(),inventory.clone())?;
            poison_outputs(stream,data)?;
            for f in data.iter_mut() { project(&rt,stream,&mut recording,f)?; }
            let recorded = data.iter().map(|f|output(stream,f)).collect::<Result<Vec<_>>>()?;
            write_outputs(out,slot,"record",&recorded,Some(&baseline),events)?;
            let bytes = recording.export_algorithm_plan(stream)?;
            let sha = save(&out.join(format!("slot{slot}-record.plan.json")),&bytes)?;
            let mut restored = workspace(&kernels,stream)?;
            // Slot 1 restores slot 0's artifact too: no raw stream address is portable identity.
            let source_bytes = if slot == 0 { &bytes } else { &reference_tables[0] };
            require(bytes==*source_bytes,"slot record tables differ; cannot claim same-table cross-slot roundtrip")?;
            restored.install_algorithm_plan(&rt,stream,source_bytes,&hash(source_bytes),&binding,&inventory)?;
            poison_outputs(stream,data)?;
            for f in data.iter_mut() { project(&rt,stream,&mut restored,f)?; }
            let restored_values = data.iter().map(|f|output(stream,f)).collect::<Result<Vec<_>>>()?;
            write_outputs(out,slot,"restore",&restored_values,Some(&baseline),events)?;
            let restored_bytes = restored.export_algorithm_plan(stream)?;
            require(restored_bytes==*source_bytes,"restored algorithm bytes changed")?;
            save(&out.join(format!("slot{slot}-restored.plan.json")),&restored_bytes)?;
            events.push(json!({"kind":"algorithm_roundtrip","slot":slot,"record_sha256":sha,"restore_sha256":hash(&restored_bytes),"restored_from_slot":0,"same_as_installed":true}));
            reference_tables.push(bytes); restored_slots.push(restored); references.push(baseline);
        }
        require(references[0]!=references[1],"slots have identical outputs; isolation fixture ineffective")?;
        // Both live slots are fully warmed before any capture. Capture and replay
        // are serialized; this is two-slot ownership, not overlap certification.
        for slot in 0..2 {
            let stream=&streams[slot]; let data=&mut devices[slot]; let ws=&mut restored_slots[slot];
            rt.device.synchronize().map_err(|e|e.to_string())?;
            let graph=capture(stream,|| {for f in data.iter_mut(){project(&rt,stream,ws,f)?;} Ok(())})?;
            poison_outputs(stream,data)?;
            graph.launch().map_err(|e|e.to_string())?;
            stream.synchronize().map_err(|e|e.to_string())?;
            let values=data.iter().map(|f|output(stream,f)).collect::<Result<Vec<_>>>()?;
            write_outputs(out,slot,"graph",&values,Some(&references[slot]),events)?;
            require(ws.export_algorithm_plan(stream)?==reference_tables[0],"post-graph table changed")?;
            events.push(json!({"kind":"graph","slot":slot,"replays":1,"output_prefill":"0x7e00 NaN","all_six_logical_shapes_warmed":true}));
        }
        let stream=&streams[0]; let bytes=&reference_tables[0]; let sha=hash(bytes);
        // Every destructive negative owns a separate new workspace.
        let fresh=||workspace(&kernels,stream);
        let install=|ws:&mut Int8Workspace|ws.install_algorithm_plan(&rt,stream,bytes,&sha,&binding,&inventory);
        let mut bad=fresh()?;
        expect_error("wrong-sha",bad.install_algorithm_plan(&rt,stream,bytes,&"0".repeat(64),&binding,&inventory),"SHA",events)?;
        let mut wrong=binding.clone(); wrong.recipe_sha256=hash(b"wrong synthetic recipe");
        let mut bad=fresh()?;
        expect_error("wrong-binding",bad.install_algorithm_plan(&rt,stream,bytes,&sha,&wrong,&inventory),"binding",events)?;
        let mut bad=fresh()?;
        expect_error("wrong-inventory",bad.install_algorithm_plan(&rt,stream,bytes,&sha,&binding,&inventory[..5]),"inventory",events)?;
        let mut bad=fresh()?;
        project(&rt,stream,&mut bad,&mut devices[0][0])?;
        expect_error("install-after-legacy-cache",install(&mut bad),"fresh workspace",events)?;
        let mut bad=fresh()?; install(&mut bad)?;
        expect_error("incomplete-coverage-export",bad.export_algorithm_plan(stream),"not all authoritative",events)?;
        assert_poison(&rt,stream,&mut bad,&mut devices[0][0],"incomplete-coverage",events)?;
        let mut bad=fresh()?;
        bad.begin_algorithm_recording(&rt,stream,binding.clone(),inventory[..5].to_vec())?;
        expect_error("missing-authoritative-shape",project(&rt,stream,&mut bad,&mut devices[0][5]),"missing",events)?;
        assert_poison(&rt,stream,&mut bad,&mut devices[0][0],"missing-shape",events)?;
        let mut bad=fresh()?; install(&mut bad)?;
        for f in devices[0].iter_mut(){project(&rt,stream,&mut bad,f)?;}
        require(bad.export_algorithm_plan(stream)?==*bytes,"negative precondition complete coverage differs")?;
        let short=stream.alloc_zeros::<u16>(1).map_err(|e|e.to_string())?;
        let f=&mut devices[0][0];
        projection_attempt()?;
        expect_error("short-input-after-complete",bad.project_half(&rt,stream,&short,f.shape.stride,&f.weight,&mut f.output,f.shape.rows),"buffer/stride mismatch",events)?;
        assert_poison(&rt,stream,&mut bad,&mut devices[0][0],"short-input",events)?;
        let mut bad=fresh()?; install(&mut bad)?;
        expect_error("foreign-stream-owner",project(&rt,&streams[1],&mut bad,&mut devices[0][0]),"owner",events)?;
        assert_poison(&rt,stream,&mut bad,&mut devices[0][0],"foreign-stream",events)?;
        let foreign=foreign_context()?; require(!foreign.is_primary(),"foreign test did not create a real non-primary context")?;
        let foreign_stream=foreign.new_stream().map_err(|e|e.to_string())?;
        let foreign_input=foreign_stream.clone_htod(&host[0][0].input).map_err(|e|e.to_string())?;
        foreign_stream.synchronize().map_err(|e|e.to_string())?;
        rt.device.bind_to_thread().map_err(|e|e.to_string())?;
        let mut bad=fresh()?; install(&mut bad)?;
        let f=&mut devices[0][0];
        projection_attempt()?;
        expect_error("foreign-context-input",bad.project_half(&rt,stream,&foreign_input,f.shape.stride,&f.weight,&mut f.output,f.shape.rows),"context",events)?;
        assert_poison(&rt,stream,&mut bad,&mut devices[0][0],"foreign-context",events)?;
        // The physical plan for logical K1160 is warm, but K1168 is unobserved.
        let mut bad=fresh()?; install(&mut bad)?;
        project(&rt,stream,&mut bad,&mut devices[0][1])?;
        rt.device.synchronize().map_err(|e|e.to_string())?;
        expect_error("capture-first-logical-alias",capture(stream,||project(&rt,stream,&mut bad,&mut devices[0][2])),"outside capture",events)?;
        assert_poison(&rt,stream,&mut bad,&mut devices[0][0],"capture-first-alias",events)?;
        rt.device.synchronize().map_err(|e|e.to_string())?;
        provenance.verify()?; lt_library.verify()?; blas_library.verify()?;
        require(hash(&fs::read(std::env::current_exe().map_err(|e|e.to_string())?).map_err(|e|e.to_string())?)==binding.executable_sha256,"executable changed")?;
        let calls=PROJECTION_CALLS.with(|calls|calls.get());
        require(calls==67,"unexpected diagnostic projection call inventory")?;
        Ok(json!({"purpose":PURPOSE,"slots":2,"logical_shapes_per_slot":6,"physical_cache_shapes_per_slot":4,"graph_replays":2,
            "normal_executed_projections":48,"public_projection_call_attempts":calls,"captured_host_projection_calls":13,
            "negative_tests":"see events","source_library_executable_unchanged":true,
            "not_verified":["whole model plan","B11/B15 weights or accuracy","overlapping stream execution","external raw graph error poison","performance","Tensor Core instruction profiler"]}))
    }
}
fn main() {
    if let Err(error)=entry(){eprintln!("{error}");std::process::exit(1);}
}
fn entry()->Result<()> {
    let args=std::env::args().skip(1).collect::<Vec<_>>();
    require(args.len()==6 && args[0]=="--provenance" && args[2]=="--provenance-sha256" && args[4]=="--output", "usage: --provenance FILE --provenance-sha256 SHA --output NEW_DIR")?;
    let raw=fs::read(&args[1]).map_err(|e|e.to_string())?;
    require(hash(&raw)==args[3],"provenance external SHA mismatch")?;
    let provenance:Provenance=serde_json::from_slice(&raw).map_err(|e|e.to_string())?;
    provenance.verify()?;
    let out=PathBuf::from(&args[5]);fs::create_dir(&out).map_err(|e|e.to_string())?;
    save_json(&out.join("intent.json"),&json!({"purpose":PURPOSE,"provenance":args[1],"provenance_sha256":args[3],"batches":[1,3],"slots":2,"shapes":[[384,1152],[512,1160],[512,1168]],"retries":0,"max_projection_calls_including_capture_and_rejected":80,"performance_measurement":false}))?;
    let mut events:Vec<Value>=vec![];
    #[cfg(all(feature="cuda",target_os="windows"))]
    let result=gpu::run(&out,&provenance,&mut events);
    #[cfg(not(all(feature="cuda",target_os="windows")))]
    let result:Result<Value>=Err("probe requires Windows --features cuda; no GPU operation performed".into());
    save_json(&out.join("events.json"),&events)?;
    save_json(&out.join("result.json"),&json!({"status":if result.is_ok(){"PASS"}else{"FAIL"},"purpose":PURPOSE,"summary":result.as_ref().ok(),"error":result.as_ref().err()}))?;
    result.map(|_|())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn finite_inventory_has_six_logical_and_four_physical_shapes(){
        let s=shapes(); assert_eq!(s.len(),6);
        let keys=s.iter().map(|s|(s.rows,s.n,s.kp)).collect::<std::collections::BTreeSet<_>>(); assert_eq!(keys.len(),4);
        assert_eq!(s.iter().map(|s|s.batch).collect::<Vec<_>>(),vec![1,1,1,3,3,3]);
    }
    #[test] fn alias_fixture_distinguishes_padding_and_slots(){
        for col in 0..1160 {assert_eq!(input_value(0,7,col,1160),input_value(0,7,col,1168));}
        for col in 1160..1168 {assert_ne!(input_value(0,7,col,1160),0.0);assert_eq!(input_value(0,7,col,1168),0.0);assert_eq!(weight_value(7,col),0.0);}
        assert_ne!(input_value(0,7,3,1160),input_value(1,7,3,1160));
    }
}
