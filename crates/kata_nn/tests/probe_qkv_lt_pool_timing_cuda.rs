//! Bounded Windows QKV pool timing: screen, or separately CPU-admitted winner ABBA.
//! All candidates come from one independent query; the public API control has its
//! own ordinary query/cache. The separate CPU FP64/Lt gate must admit any timing.
#![cfg(feature = "cuda")]

#[path = "support/qkv_lt_pool_timing.rs"]
mod metadata;
#[path = "support/qkv_lt_baseline.rs"]
mod operands;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use std::sync::Arc;
use std::time::Instant;
use kata_nn::backends::cuda::{CudaRuntime, backend_build_fingerprint, device_fingerprint};
use kata_nn::tactic_plan::{installed_plan_id, sha256_file, tactic_var};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const FINGERPRINT_SHA: &str = "17f0b3b99aed726467316f033df63debef2dd1c8a3a2b389f81eb3e404f6714c";
const MODEL_SHA: &str = "1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6";

fn hash(bytes: &[u8]) -> String { hex::encode(Sha256::digest(bytes)) }
fn half_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn finite(values: &[u16]) -> bool {
    values.iter().all(|x| x & 0x7c00 != 0x7c00)
}
fn record(path: &Path) -> Value {
    json!({"path":path,"sha256":sha256_file(path).unwrap(),"bytes":fs::metadata(path).unwrap().len()})
}
fn raw(path: &Path, bytes: &[u8]) -> Value {
    OpenOptions::new().write(true).create_new(true).open(path).unwrap()
        .write_all(bytes).unwrap();
    json!({"path":path,"sha256":hash(bytes),"bytes":bytes.len()})
}
fn save(dir: &Path, report: &Value) {
    fs::write(dir.join("report.json"), serde_json::to_vec_pretty(report).unwrap()).unwrap();
}

#[cfg(windows)]
fn loaded_libraries() -> Value {
    use std::ffi::c_void;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetModuleHandleW(name: *const u16) -> *mut c_void;
        fn GetModuleFileNameW(module: *mut c_void, file: *mut u16, size: u32) -> u32;
    }
    let name: Vec<u16> = "cublasLt64_13.dll".encode_utf16().chain(Some(0)).collect();
    let module = unsafe { GetModuleHandleW(name.as_ptr()) };
    assert!(!module.is_null(), "the actual loaded CUDA 13 cuBLASLt DLL is required");
    let mut buffer = vec![0u16; 32768];
    let count = unsafe { GetModuleFileNameW(module, buffer.as_mut_ptr(), buffer.len() as u32) };
    assert!(count > 0 && (count as usize) < buffer.len(), "actual Lt DLL path query failed/truncated");
    let path = PathBuf::from(String::from_utf16(&buffer[..count as usize]).unwrap());
    assert!(path.is_absolute() && path.is_file(), "actual loaded Lt DLL path must resolve");
    json!({"cublasLt":record(&path),"source":"GetModuleHandleW + GetModuleFileNameW on the loaded process module",
        "version":unsafe{cudarc::cublaslt::sys::cublasLtGetVersion()},"platform":"windows"})
}
#[cfg(not(windows))]
fn loaded_libraries() -> Value { panic!("this bounded candidate-pool registration targets Windows") }

const NUMERIC_ANALYSIS_SHA: &str = "95b1247e9644e8c1f6d44318c87f21c707a8be6fa480bf9c86db8247aba0c518";
const NUMERIC_RAW_SHA: &str = "fb34e812a0e6a668a21373118565cc999a29a928dc66f4981da2516692e9d95f";
const ACTIVE: usize = operands::M * operands::N;
const WARMUP: usize = 80;

fn local(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("/mnt/d/") { PathBuf::from(format!("D:/{rest}")) }
    else { PathBuf::from(path) }
}
fn bound(entry: &Value) -> Vec<u8> {
    let path = local(entry["path"].as_str().expect("bound path"));
    let data = fs::read(&path).unwrap();
    assert_eq!(hash(&data), entry["sha256"].as_str().unwrap(), "bound hash {}",path.display());
    if let Some(bytes) = entry["bytes"].as_u64() { assert_eq!(data.len() as u64,bytes); }
    data
}
fn half_values(bytes: &[u8]) -> Vec<u16> {
    assert_eq!(bytes.len()%2,0);
    bytes.chunks_exact(2).map(|v|u16::from_le_bytes(v.try_into().unwrap())).collect()
}

struct Admission {
    value: Value,
    numeric: Value,
    expected: Vec<u16>,
    order: Vec<usize>,
    iterations: usize,
}

fn admission(path: &Path) -> Admission {
    let a: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    assert_eq!(a["schema"],1);
    assert_eq!(a["source_hashes"]["test"],hash(include_bytes!("probe_qkv_lt_pool_timing_cuda.rs")));
    assert_eq!(a["source_hashes"]["metadata"],hash(include_bytes!("support/qkv_lt_pool_timing.rs")));
    assert_eq!(a["source_hashes"]["operands"],hash(include_bytes!("support/qkv_lt_baseline.rs")));
    assert_eq!(a["source_hashes"]["precision"],hash(include_bytes!("support/lt_replay_precision.rs")));
    let binary_path = local(a["binary"]["path"].as_str().unwrap()).canonicalize().unwrap();
    assert_eq!(binary_path,std::env::current_exe().unwrap().canonicalize().unwrap());
    assert_eq!(sha256_file(&binary_path).unwrap(),a["binary"]["sha256"]);
    assert_eq!(a["numeric_analysis"]["sha256"],NUMERIC_ANALYSIS_SHA);
    assert_eq!(a["numeric_raw"]["sha256"],NUMERIC_RAW_SHA);
    let analysis: Value = serde_json::from_slice(&bound(&a["numeric_analysis"])).unwrap();
    let numeric: Value = serde_json::from_slice(&bound(&a["numeric_raw"])).unwrap();
    assert_eq!(analysis["status"],"COMPLETE_CPU_VERIFIED_QKV_POOL_NUMERIC_DIAGNOSTIC");
    assert_eq!(analysis["controls_pass"],true);
    assert_eq!(analysis["numeric_admitted_original_indices"],json!([0,1,2,3,4,5,6,7]));
    assert_eq!(analysis["raw_report"]["sha256"],NUMERIC_RAW_SHA);
    assert_eq!(numeric["status"],"COMPLETE_QKV_LT_POOL_PENDING_CPU");
    assert_eq!(numeric["test_source_sha256"],"6f07bb9d8943d3296412ab1e9bd525267a9f2b61595ef9ac6fed43d8a827320d");
    assert_eq!(numeric["metadata_source_sha256"],"f714f115131f5d2e7bbefec96fc2af3dfccc19a1c2110fe5a1a5d704ed5a5e93");
    assert_eq!(numeric["operand_source_sha256"],hash(include_bytes!("support/qkv_lt_baseline.rs")));
    assert_eq!(numeric["precision_source_sha256"],hash(include_bytes!("support/lt_replay_precision.rs")));
    assert_eq!(numeric["fingerprint"]["sha256"],FINGERPRINT_SHA);
    let expected_bytes = bound(&numeric["public"]["output"]);
    assert_eq!(expected_bytes.len(),ACTIVE*2);
    let expected = half_values(&expected_bytes);
    assert!(finite(&expected));
    for index in 0..8 {
        let c = &numeric["candidates"][index];
        assert_eq!(c["original_index"],index);
        assert_eq!(analysis["candidates"][index]["original_index"],index);
        assert_eq!(analysis["candidates"][index]["numeric_admitted"],true);
        assert_eq!(analysis["candidates"][index]["comparison"]["operator_gate"]["pass_all"],true);
        assert_eq!(analysis["candidates"][index]["comparison"]["operator_gate"]["strict_max_abs_limit"],0.01);
        assert_eq!(c["status"],"COMPLETE_FINITE_REPEAT_PENDING_CPU");
        assert_eq!(c["repeat_bits_equal"],true);
        assert_eq!(c["vs_public"]["all_bits_equal"],true);
        assert_eq!(c["runs"].as_array().unwrap().len(),2);
        for run in c["runs"].as_array().unwrap() {
            assert_eq!(run["execution"]["status"],"PASS_REPLAY_EXECUTED");
            assert_eq!(run["all_finite"],true);
            assert_eq!(run["operands_unchanged"],true);
            assert_eq!(bound(&run["output"]),expected_bytes,"numeric output evidence drift");
        }
    }
    let (order,iterations) = match a["mode"].as_str() {
        Some("screen") => {
            assert_eq!(a["status"],"ADMITTED_QKV_POOL_SCREEN");
            assert!(a["winner_original_index"].is_null());
            ((0..8).chain((0..8).rev()).collect(),300)
        }
        Some("abba") => {
            assert_eq!(a["status"],"ADMITTED_QKV_POOL_ABBA");
            let winner=a["winner_original_index"].as_u64().expect("CPU-registered unique winner") as usize;
            assert!((1..8).contains(&winner));
            let screen: Value=serde_json::from_slice(&bound(&a["screen_analysis"])).unwrap();
            assert_eq!(screen["status"],"SCREEN_WINNER_REQUIRES_INDEPENDENT_ABBA");
            assert_eq!(screen["mode"],"screen");
            assert_eq!(screen["controls_pass"],true);
            assert_eq!(screen["winner_original_index"],winner);
            assert_eq!(screen["numeric_report"]["sha256"],NUMERIC_ANALYSIS_SHA);
            let candidates=screen["candidates"].as_array().expect("complete screen candidate decisions");
            assert_eq!(candidates.len(),8);
            for (index,candidate) in candidates.iter().enumerate() {
                assert_eq!(candidate["original_index"],index);
            }
            let candidate=&candidates[winner];
            let decision=&a["screen_decision"];
            assert_eq!(decision["numeric_and_controls_pass"],true);
            assert_eq!(decision["stable"],true);
            assert_eq!(candidate["stable"],decision["stable"]);
            assert_eq!(candidate["gain_fraction"],decision["gain_fraction"]);
            let gain=decision["gain_fraction"].as_f64().unwrap();
            let spread=decision["maximum_relative_spread"].as_f64().unwrap();
            let baseline_spread=candidates[0]["relative_spread"].as_f64().unwrap();
            let candidate_spread=candidate["relative_spread"].as_f64().unwrap();
            assert!(baseline_spread.is_finite() && baseline_spread>=0.0);
            assert!(candidate_spread.is_finite() && candidate_spread>=0.0);
            assert_eq!(spread,baseline_spread.max(candidate_spread));
            assert!(gain.is_finite() && gain>=0.01 && spread.is_finite() && (0.0..=0.05).contains(&spread));
            ([0,winner,winner,0].repeat(3),1000)
        }
        _ => panic!("only separately admitted screen or ABBA mode is supported"),
    };
    Admission{value:a,numeric,expected,order,iterations}
}

fn reset(stream: &Arc<CudaStream>, output: &mut CudaSlice<u16>, initial: &[u16]) -> String {
    stream.memcpy_htod(initial,output).unwrap();
    let actual=stream.clone_dtoh(output).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(actual,initial,"actual full NaN/tail initialization");
    hash(&half_bytes(&actual))
}

fn submit_once(rt: &CudaRuntime, stream: &Arc<CudaStream>, gpu: &operands::DeviceOperands,
    output: &mut CudaSlice<u16>, prepared: &metadata::PreparedLt) {
    let (ip,ig)=gpu.input.device_ptr(stream);
    let (wp,wg)=gpu.weights.device_ptr(stream);
    let (op,og)=output.device_ptr_mut(stream);
    let (workspace,_)=rt.cublaslt_workspace_ptr(stream);
    assert_eq!([ip,wp,op,workspace],prepared.pointers());
    assert!(prepared.identity_unchanged());
    let result=unsafe{prepared.submit()};
    let completion=stream.synchronize();
    drop((ig,wg,og));
    result.unwrap();
    completion.unwrap();
    assert!(prepared.identity_unchanged());
}

fn checkpoint(dir: &Path, stem: &str, stream: &Arc<CudaStream>, gpu: &operands::DeviceOperands,
    host: &operands::HostOperands, output: &CudaSlice<u16>, expected: &[u16]) -> Value {
    let all=stream.clone_dtoh(output).unwrap();
    let input=stream.clone_dtoh(&gpu.input).unwrap();
    let weights=stream.clone_dtoh(&gpu.weights).unwrap();
    stream.synchronize().unwrap();
    let guard: Vec<u16>=(0..256).map(|i|0x5000 ^ i).collect();
    let result=json!({"output":raw(&dir.join(format!("{stem}-output.f16le")),&half_bytes(&all[..ACTIVE])),
        "guard_after":raw(&dir.join(format!("{stem}-guard-after.f16le")),&half_bytes(&all[ACTIVE..])),
        "elements":ACTIVE,"all_bits_equal":all[..ACTIVE]==*expected,"all_finite":finite(&all[..ACTIVE]),
        "operands_unchanged":input==host.input && weights==host.weights,
        "input_after_sha256":hash(&half_bytes(&input)),"weights_after_sha256":hash(&half_bytes(&weights))});
    // Save even failed checks before unwinding into the enclosing report handler.
    fs::write(dir.join(format!("{stem}-check.json")),serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    assert_eq!(&all[ACTIVE..],guard.as_slice(),"full output tail guard");
    assert_eq!(&all[..ACTIVE],expected,"complete output differs from admitted numeric bytes");
    assert!(finite(&all[..ACTIVE]));
    assert_eq!(input,host.input);
    assert_eq!(weights,host.weights);
    result
}

#[test]
fn probe_qkv_lt_pool_timing_cuda() {
    match std::env::var("KATAGO_RUN_QKV_LT_POOL_TIMING").as_deref() {
        Err(std::env::VarError::NotPresent) => { eprintln!("SKIP: set KATAGO_RUN_QKV_LT_POOL_TIMING=1");return; }
        Ok("1") => {}
        other => panic!("invalid explicit timing opt-in {other:?}"),
    }
    assert_eq!(std::env::consts::OS,"windows");
    let root=PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap();
    let dir=PathBuf::from(std::env::var_os("KATAGO_QKV_LT_POOL_TIMING_DIR").expect("fresh output"));
    assert!(!dir.exists() && dir.parent().unwrap().canonicalize().unwrap()
        .starts_with(root.join("target").canonicalize().unwrap()));
    fs::create_dir(&dir).unwrap();
    let admission_path=PathBuf::from(std::env::var_os("KATAGO_QKV_POOL_TIMING_ADMISSION").expect("CPU admission required"));
    let model=PathBuf::from(std::env::var_os("KATAGO_TEST_MODEL_DIR").expect("model directory"))
        .join("kata1-tf3-b11c768-s11001M-d5973M.bin.gz");
    let mut report=json!({"schema":1,"status":"PREPARING_NOT_COMPLETE","production_certified":false,
        "timing_performed":false,"phases":[],"prepared_candidates":[],"pre_checks":[],
        "scope":"one bounded pool screen OR separately admitted unique-winner ABBA; no whole-network certification",
        "test_executable":record(&std::env::current_exe().unwrap()),
        "test_source_sha256":hash(include_bytes!("probe_qkv_lt_pool_timing_cuda.rs")),
        "metadata_source_sha256":hash(include_bytes!("support/qkv_lt_pool_timing.rs")),
        "operand_source_sha256":hash(include_bytes!("support/qkv_lt_baseline.rs")),
        "precision_source_sha256":hash(include_bytes!("support/lt_replay_precision.rs"))});
    save(&dir,&report);
    let outcome=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let admitted=admission(&admission_path); // CPU identities and numeric bytes before CUDA.
        report["admission"]=record(&admission_path);
        report["admission_value"]=admitted.value.clone();
        report["numeric_analysis"]=admitted.value["numeric_analysis"].clone();
        report["numeric_raw"]=admitted.value["numeric_raw"].clone();
        report["mode"]=admitted.value["mode"].clone();
        report["order"]=json!(admitted.order);
        report["warmup"]=json!(WARMUP);
        report["iterations"]=json!(admitted.iterations);
        report["shape"]=json!({"m":operands::M,"n":operands::N,"k":operands::K,"batch":14});
        report["registration"]=json!({"maximum_relative_spread":0.05,"minimum_gain_fraction":0.01,
            "screen_candidates":8,"maximum_screen_winners":1,"screen_phases":16,"screen_event_samples":4800,
            "abba_phases":12,"abba_event_samples":12000,"stop_sync_every_iteration":true,
            "timing_path":"both arms prepared original-object Lt enqueue only",
            "statistics":"all samples retained; CPU computes phase medians/geometric means and gates"});
        assert_eq!(sha256_file(&model).unwrap(),MODEL_SHA);
        report["model"]=record(&model);
        assert!(installed_plan_id().is_none());
        let mut tactics=serde_json::Map::new();
        for (key,value) in [("KATAGO_CUDA_CUBLASLT","1"),("KATAGO_CUDA_CUBLASLT_RANK","heuristic"),("KATAGO_CUDA_GEMM_LAYOUT","tn")] {
            let actual=tactic_var(key).expect("explicit registered tactic");
            assert_eq!(actual,value);tactics.insert(key.into(),json!(actual));
        }
        report["tactics"]=Value::Object(tactics);
        let host=operands::load_operands(&root,&model).unwrap();
        report["operand_provenance"]=host.provenance.clone();
        let rt=CudaRuntime::new().expect("explicit timing CUDA runtime");
        assert!(!kata_nn::backends::cuda_exec::capturing());
        let build=serde_json::to_value(backend_build_fingerprint()).unwrap();
        assert_eq!(build,admitted.numeric["backend_build_fingerprint"]);
        report["backend_build_fingerprint"]=build;
        let device=device_fingerprint(&rt.device).unwrap();
        let device=json!({"gpu_name":device.gpu_name,"compute_capability":device.compute_capability,
            "sm_count":device.sm_count,"l2_cache_bytes":device.l2_cache_bytes});
        assert_eq!(device,admitted.numeric["device_fingerprint"]);
        report["device_fingerprint"]=device;
        let libraries=loaded_libraries();
        assert_eq!(libraries,admitted.numeric["loaded_libraries"],"actual DLL path/hash/version differs");
        report["loaded_libraries"]=libraries;
        let stream=rt.device.new_stream().unwrap();
        report["stream"]=json!(format!("0x{:016x}",stream.cu_stream() as usize));
        let gpu=operands::DeviceOperands{input:stream.clone_htod(&host.input).unwrap(),
            weights:stream.clone_htod(&host.weights).unwrap()};
        assert_eq!(stream.clone_dtoh(&gpu.input).unwrap(),host.input);
        assert_eq!(stream.clone_dtoh(&gpu.weights).unwrap(),host.weights);
        stream.synchronize().unwrap();
        let mut initial=vec![0x7e00u16;ACTIVE];
        initial.extend((0..256).map(|i|0x5000 ^ i as u16));
        let mut output=stream.clone_htod(&initial).unwrap();
        let initial_sha=reset(&stream,&mut output,&initial);
        report["initial_output"]=raw(&dir.join("initial-output.f16le"),&half_bytes(&initial));
        assert_eq!(report["initial_output"]["sha256"],initial_sha);
        operands::run_lt_baseline(&rt,&stream,&gpu,&mut output).unwrap();
        report["public_check"]=checkpoint(&dir,"public",&stream,&gpu,&host,&output,&admitted.expected);
        save(&dir,&report);
        let pool=metadata::select_pool(&rt,&stream,operands::M,operands::N,operands::K).unwrap();
        report["pool_query"]=pool.metadata.clone();
        assert_eq!(pool.candidates.len(),8);
        // Check the entire original pool in the new process, including entries
        // that ABBA will not execute. Never mask any opaque byte.
        for i in 0..8 {
            assert_eq!(pool.metadata["unfiltered"][i]["opaque"],admitted.numeric["pool_query"]["unfiltered"][i]["opaque"]);
            assert_eq!(pool.metadata["unfiltered"][i]["original_index"],i);
            let actual=&pool.candidates[i].metadata["selected"];
            let expected=&admitted.numeric["candidates"][i]["metadata"]["selected"];
            for key in ["opaque","nine_config_attributes","numerical_impl_flags","minimum_alignment"] {
                assert_eq!(actual[key],expected[key],"candidate {i} {key} differs from numeric admission");
            }
        }
        let mut selected=admitted.order.clone();selected.sort_unstable();selected.dedup();
        let mut prepared=Vec::new();
        for index in selected {
            let selection=pool.candidates[index].selection.as_ref().expect("admitted candidate metadata");
            let object=metadata::prepare_selected(&rt,&stream,&gpu.input,&gpu.weights,&mut output,selection).unwrap();
            report["prepared_candidates"].as_array_mut().unwrap().push(json!({"original_index":index,"metadata":object.metadata}));
            reset(&stream,&mut output,&initial);
            submit_once(&rt,&stream,&gpu,&mut output,&object);
            let check=checkpoint(&dir,&format!("pre-{index:02}"),&stream,&gpu,&host,&output,&admitted.expected);
            report["pre_checks"].as_array_mut().unwrap().push(json!({"original_index":index,"check":check}));
            prepared.push((index,object));
            save(&dir,&report);
        }
        report["device_pointers"]=json!({"input":format!("0x{:016x}",prepared[0].1.pointers()[0]),
            "weights":format!("0x{:016x}",prepared[0].1.pointers()[1]),"output":format!("0x{:016x}",prepared[0].1.pointers()[2]),
            "workspace":format!("0x{:016x}",prepared[0].1.pointers()[3]),"same_for_all_phases":true});
        let flags=Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
        let events: Vec<_>=(0..admitted.iterations).map(|_|(rt.device.new_event(flags).unwrap(),rt.device.new_event(flags).unwrap())).collect();
        report["timing_performed"]=json!(true);
        for (phase,&index) in admitted.order.iter().enumerate() {
            let object=&prepared.iter().find(|(i,_)|*i==index).unwrap().1;
            let actual_initial_sha=reset(&stream,&mut output,&initial);
            assert_eq!(actual_initial_sha,initial_sha);
            let (completed,wall_seconds,samples,measured_error,completion_error)={
                // Access guards enclose warmup AND every measured stop sync.
                let (ip,ig)=gpu.input.device_ptr(&stream);
                let (wp,wg)=gpu.weights.device_ptr(&stream);
                let (op,og)=output.device_ptr_mut(&stream);
                let (workspace,_)=rt.cublaslt_workspace_ptr(&stream);
                assert_eq!([ip,wp,op,workspace],object.pointers());
                assert!(object.identity_unchanged());
                let warmed=(||->Result<(),String>{for _ in 0..WARMUP {unsafe{object.submit()}?;}Ok(())})();
                let warm_completion=stream.synchronize();
                warmed.unwrap();warm_completion.unwrap();
                eprintln!("[qkv-lt-pool-timing] mode={} phase={phase} original_index={index} warmup=80 iterations={} stop_sync=every_iteration",
                    admitted.value["mode"].as_str().unwrap(),admitted.iterations);
                let mut completed=0usize;
                let started=Instant::now();
                let measured=(||->Result<(),String>{
                    for (start,stop) in &events {
                        start.record(&stream).map_err(|e|e.to_string())?;
                        unsafe{object.submit()}?;
                        stop.record(&stream).map_err(|e|e.to_string())?;
                        stop.synchronize().map_err(|e|e.to_string())?;
                        completed+=1;
                    }
                    Ok(())
                })();
                let wall=started.elapsed().as_secs_f64();
                let completion=stream.synchronize();
                // Ensure completion on failure before pointer guards can drop.
                let measured_error=measured.err();
                let completion_error=completion.err().map(|e|e.to_string());
                drop((ig,wg,og));
                let samples: Vec<f64>=events[..completed].iter().map(|(a,b)|a.elapsed_ms(b).unwrap() as f64).collect();
                (completed,wall,samples,measured_error,completion_error)
            };
            let arm=if index==0 {"baseline"} else {"candidate"};
            let round=if admitted.value["mode"]=="screen" {phase/8} else {phase/4};
            report["phases"].as_array_mut().unwrap().push(json!({"phase":phase,"round":round,
                "original_index":index,"arm":arm,"warmup":WARMUP,"requested_iterations":admitted.iterations,
                "completed_iterations":completed,"cuda_event_ms":samples,"wall_seconds":wall_seconds,
                "stop_sync_every_iteration":true,"initial_output_sha256":actual_initial_sha,
                "measured_enqueue_error":measured_error,"completion_error":completion_error,
                "identity_unchanged":object.identity_unchanged()}));
            save(&dir,&report);
            assert!(measured_error.is_none() && completion_error.is_none(),"measured phase failed");
            assert_eq!(completed,admitted.iterations);
            assert!(samples.iter().all(|v|v.is_finite() && *v>0.0));
            assert!(wall_seconds.is_finite() && wall_seconds>0.0);
            assert!(object.identity_unchanged());
            report["phases"][phase]["post_check"]=checkpoint(&dir,&format!("phase-{phase:02}-idx-{index:02}"),
                &stream,&gpu,&host,&output,&admitted.expected);
            save(&dir,&report);
        }
        stream.synchronize().unwrap();
        report["final_stream_synchronized"]=json!(true);
        assert_eq!(sha256_file(&admission_path).unwrap(),report["admission"]["sha256"]);
        // Numeric records, binary, source identities, and all bound reference
        // output bytes are checked again after the last GPU operation.
        let repeated=admission(&admission_path);
        assert_eq!(repeated.value,admitted.value);
        assert_eq!(repeated.numeric,admitted.numeric);
        assert_eq!(loaded_libraries(),report["loaded_libraries"]);
        report["admission_revalidated_after_timing"]=json!(true);
        report["status"]=json!("COMPLETE_QKV_LT_POOL_TIMING_PENDING_CPU");
        save(&dir,&report);
    }));
    if let Err(panic)=outcome {
        report["status"]=json!("FAILED_QKV_LT_POOL_TIMING_EVIDENCE_PRESERVED");
        report["failure"]=json!(panic.downcast_ref::<String>().map(String::as_str)
            .or_else(||panic.downcast_ref::<&str>().copied()).unwrap_or("non-string panic"));
        save(&dir,&report);
        std::panic::resume_unwind(panic);
    }
}
