//! Opt-in diagnostics for hash-bound SM120 FA4 FP32-accumulator CUBINs.
//! No production tactic/build changes. Root owns the exclusive GPU run window.
//!
//! Run with KATAGO_RUN_FORK_FA4_PROBE=1 and --features cuda --release --nocapture.
//! KATAGO_FORK_FA4_VARIANT=original (default) uses fa4-reference; `strict` uses
//! target/fork-parity-20260908/fa4-strict-reference-r2 with its own device ABI.
//! KATAGO_FORK_FA4_DIR may override the selected artifact directory.
//! KATAGO_FORK_FA4_ITERATIONS defaults to 100; KATAGO_FORK_FA4_CASES to 3.
//! The copied artifact directories contain source, licenses, hashes and offline
//! ABI evidence. The original FP32 name describes QK/PV MMA accumulation only:
//! Original FA4 uses exp2 approximation and half P. The strict export changes
//! softmax arithmetic and scale ABI; it remains an unverified experiment until
//! its semantic audit and numerical/performance gates complete. This isolated
//! attention probe cannot grant whole-model or production certification.
#![cfg(feature = "cuda")]

use std::{path::PathBuf, sync::Arc, time::Instant};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig,
    PushKernelArg, sys,
};
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};
use kata_nn::backends::cuda::{CudaRuntime, f16_to_f32_bits, f32_to_f16_bits};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const B: usize = 14;
const S: usize = 361;
const H: usize = 12;
const D: usize = 32;
const C: usize = H * D;
const N: usize = B * S * C;
const ORIGINAL_CUBIN_SHA256: &str =
    "98eeb4a4bc6d20f664c6ad78d37c5f2e98128c1d1b31ceb08ed9b9a24f221043";
const ORIGINAL_KERNEL: &str = "kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0";
const STRICT_CUBIN_SHA256: &str =
    "90341b6d5c54a69e983a7e7d8b62956748785ad44c3e4e1b2734b92f3da46749";
const STRICT_KERNEL: &str = "kernel_cutlass_kernel_strict_flash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16o3613212141152132415872_tensorptrf16gmemalign16o3613212141152132415872_tensorptrf1_0";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fa4Variant {
    Original,
    Strict,
}

impl Fa4Variant {
    fn from_env() -> Self {
        match std::env::var("KATAGO_FORK_FA4_VARIANT") {
            Err(std::env::VarError::NotPresent) => Self::Original,
            Ok(value) if value == "original" => Self::Original,
            Ok(value) if value == "strict" => Self::Strict,
            other => panic!("KATAGO_FORK_FA4_VARIANT must be original or strict: {other:?}"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Original => "original",
            Self::Strict => "strict",
        }
    }

    fn directory(self) -> &'static str {
        match self {
            Self::Original => "fa4-reference",
            Self::Strict => "fa4-strict-reference-r2",
        }
    }

    fn cubin_sha256(self) -> &'static str {
        match self {
            Self::Original => ORIGINAL_CUBIN_SHA256,
            Self::Strict => STRICT_CUBIN_SHA256,
        }
    }

    fn kernel(self) -> &'static str {
        match self {
            Self::Original => ORIGINAL_KERNEL,
            Self::Strict => STRICT_KERNEL,
        }
    }

    fn candidate_id(self) -> &'static str {
        match self {
            Self::Original => "fa4-b14-s361-h12-d32-tm128-tn96-s1-fp32",
            Self::Strict => "fa4-strict-r2-b14-s361-h12-d32-tm128-tn96-s1-fp32",
        }
    }

    fn abi(self) -> Value {
        match self {
            Self::Original => json!({"parameterBytes":[48,48,48,48,4,4,4,4,12],
                "scale":"natural scale multiplied by LOG2_E"}),
            Self::Strict => json!({"parameterBytes":[8,8,8,8,4,12],
                "scale":"natural FP32 scale, no LOG2_E conversion",
                "evidence":"probe-abi-manifest.json; r2 PTX/header/MLIR/host-object disassembly"}),
        }
    }
}

/// Original export's device arguments, distinct from its generated C descriptor.
/// Verified by trace_abi.cpp with stubbed CUDA functions, then against PTX.
#[repr(C)]
#[derive(Clone, Copy)]
struct TensorArg {
    ptr: u64,
    shape: [i32; 4],  // S,D,H,B
    stride: [i64; 3], // row,head,batch; dim stride is statically one
}
// SAFETY: plain 48-byte, 8-aligned CUDA by-value parameter with no Rust references.
unsafe impl DeviceRepr for TensorArg {}
const _: () = assert!(std::mem::size_of::<TensorArg>() == 48);

fn descriptor(ptr: u64, row_stride: usize) -> TensorArg {
    TensorArg {
        ptr,
        shape: [S as i32, D as i32, H as i32, B as i32],
        stride: [row_stride as i64, D as i64, (S * row_stride) as i64],
    }
}

fn launch_fork_original(
    stream: &CudaStream,
    f: &CudaFunction,
    input: u64,
    output: u64,
    scale: f32,
) {
    let q = descriptor(input, 3 * C);
    let k = descriptor(input + (C * 2) as u64, 3 * C);
    let v = descriptor(input + (C * 4) as u64, 3 * C);
    let o = descriptor(output, C);
    let scale_log2 = scale * std::f32::consts::LOG2_E;
    // The final scheduler tuple's last two bytes are padding. PTX does not read
    // arguments 5..8 in this fixed noncausal artifact; initialize padding to zero.
    let scheduler = [1u32, 1, 0];
    unsafe {
        stream
            .launch_builder(f)
            .arg(&q)
            .arg(&k)
            .arg(&v)
            .arg(&o)
            .arg(&scale_log2)
            .arg(&3i32)
            .arg(&(H as i32))
            .arg(&(B as i32))
            .arg(&scheduler)
            .launch(LaunchConfig {
                grid_dim: (3, H as u32, B as u32),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 20480,
            })
            .expect("fork FA4 Driver launch");
    }
}

fn launch_fork(
    stream: &CudaStream,
    f: &CudaFunction,
    input: u64,
    output: u64,
    scale: f32,
    variant: Fa4Variant,
) {
    if variant == Fa4Variant::Original {
        return launch_fork_original(stream, f, input, output, scale);
    }
    // Strict r2's static fake tensors erase all runtime shape/stride fields:
    // PTX parameters 0..3 are 8-byte pointers, 4 is f32, 5 is a 12-byte tuple.
    // The generated header, clean MLIR and host .o at 0x3c0..0x4ec agree.
    // Host .o writes u32(1),u32(1),u16(0); zero the trailing two padding bytes.
    // The fixed noncausal kernel has no ld.param for this fast_divmod(1) tuple.
    let k = input + (C * 2) as u64;
    let v = input + (C * 4) as u64;
    let fast_divmod_one = [1u32, 1, 0];
    unsafe {
        stream
            .launch_builder(f)
            .arg(&input)
            .arg(&k)
            .arg(&v)
            .arg(&output)
            .arg(&scale) // Strict softmax consumes the natural scale unchanged.
            .arg(&fast_divmod_one)
            .launch(LaunchConfig {
                grid_dim: (3, H as u32, B as u32),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 20480,
            })
            .expect("strict r2 FA4 Driver launch");
    }
}

fn launch_rust(
    stream: &CudaStream,
    f: &CudaFunction,
    input: u64,
    cos: u64,
    sin: u64,
    output: u64,
    scale: f32,
    rows: usize,
    threads: u32,
) {
    unsafe {
        stream
            .launch_builder(f)
            .arg(&input)
            .arg(&cos)
            .arg(&sin)
            .arg(&output)
            .arg(&(S as i32))
            .arg(&(D as i32))
            .arg(&scale)
            .arg(&(H as i32))
            .launch(LaunchConfig {
                grid_dim: (S.div_ceil(rows) as u32, (B * H) as u32, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            })
            .expect("Rust attention launch");
    }
}

// Test-only helpers: an independent, unblocked FP32 softmax reference and a
// standalone batch-shared RoPE. Inline half conversion avoids CUDA header paths.
// Reference explicitly rounds each FP32 multiply/add and never rounds P to half.
const REFERENCE_SOURCE: &str = r#"
__device__ float h2f(unsigned short h) {
    float v; asm("cvt.f32.f16 %0, %1;" : "=f"(v) : "h"(h)); return v;
}
__device__ unsigned short f2h(float v) {
    unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v)); return h;
}
extern "C" __global__ void probe_rope(
    const unsigned short* x, const float* co, const float* sn, unsigned short* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= 14 * 361 * 192) return;
    int row = i / 192, pair = i % 192, r = (row % 361) * 192 + pair;
    int offset = row * 1152 + pair * 2;
    float c = co[r], s = sn[r];
    #pragma unroll
    for(int qk=0;qk<2;qk++) {
        int p = offset + qk * 384;
        float a=h2f(x[p]), b=h2f(x[p+1]);
        y[p]=f2h(a*c-b*s); y[p+1]=f2h(a*s+b*c);
    }
    y[offset+768]=x[offset+768]; y[offset+769]=x[offset+769];
}
extern "C" __global__ void probe_reference(
    const unsigned short* x, float* out, float scale) {
    int qi=blockIdx.x, bh=blockIdx.y, b=bh/12, h=bh%12, t=threadIdx.x;
    __shared__ float scores[361];
    int qoff=(b*361+qi)*1152+h*32;
    for(int ki=t;ki<361;ki+=blockDim.x) {
        int koff=(b*361+ki)*1152+384+h*32;
        float dot=0;
        #pragma unroll
        for(int d=0;d<32;d++) dot=__fadd_rn(dot,__fmul_rn(h2f(x[qoff+d]),h2f(x[koff+d])));
        scores[ki]=__fmul_rn(dot,scale);
    }
    __syncthreads();
    if(t==0) {
        float mx=-3.402823466e38F, sum=0;
        for(int i=0;i<361;i++) mx=fmaxf(mx,scores[i]);
        for(int i=0;i<361;i++) { scores[i]=expf(scores[i]-mx); sum=__fadd_rn(sum,scores[i]); }
        for(int i=0;i<361;i++) scores[i]=__fdiv_rn(scores[i],sum);
    }
    __syncthreads();
    if(t<32) {
        float acc=0;
        for(int ki=0;ki<361;ki++) {
            int voff=(b*361+ki)*1152+768+h*32+t;
            acc=__fadd_rn(acc,__fmul_rn(scores[ki],h2f(x[voff])));
        }
        out[(b*361+qi)*384+h*32+t]=acc;
    }
}
"#;

fn measure(
    device: &Arc<CudaContext>,
    stream: &CudaStream,
    label: &str,
    iterations: usize,
    mut launch: impl FnMut(),
) -> Value {
    for _ in 0..32 {
        launch();
    }
    stream.synchronize().expect("timing warmup complete");
    let events: Vec<_> = (0..iterations)
        .map(|_| {
            (
                device
                    .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .expect("start event"),
                device
                    .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .expect("end event"),
            )
        })
        .collect();
    let wall_start = Instant::now();
    // Match fork benchmarkOutput: enqueue all event pairs, synchronize once.
    for (start, end) in &events {
        start.record(stream).expect("record start");
        launch();
        end.record(stream).expect("record end");
    }
    stream.synchronize().expect("timed queue complete");
    let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
    let mut times: Vec<f64> = events
        .iter()
        .map(|(a, b)| a.elapsed_ms(b).expect("elapsed event") as f64)
        .collect();
    times.sort_by(f64::total_cmp);
    let median = if iterations % 2 == 1 {
        times[iterations / 2]
    } else {
        (times[iterations / 2 - 1] + times[iterations / 2]) * 0.5
    };
    json!({"candidate":label,"iterations":iterations,"cudaEventMedianMs":median,
        "cudaEventMeanMs":times.iter().sum::<f64>()/iterations as f64,
        "wallMeanMs":wall_ms/iterations as f64})
}

fn compare_half(reference: &[f32], output: &[u16]) -> Value {
    let mut max_abs = 0.0f32;
    let mut square_sum = 0.0f64;
    let mut different = 0usize;
    for (&r, &bits) in reference.iter().zip(output) {
        let v = f16_to_f32_bits(bits);
        assert!(r.is_finite() && v.is_finite(), "nonfinite attention output");
        let error = (r - v).abs();
        max_abs = max_abs.max(error);
        square_sum += (error as f64).powi(2);
        different += usize::from(f32_to_f16_bits(r) != bits);
    }
    let rmse = (square_sum / reference.len() as f64).sqrt();
    json!({"maxAbs":max_abs,"rmse":rmse,"roundedReferenceHalfBitMismatches":different,
        "elements":reference.len()})
}

fn verify_reference_cpu(rotated: &[u16], reference: &[f32], scale: f32) {
    let mut worst = 0.0f32;
    for sample in 0..64 {
        let bh = sample * 37 % (B * H);
        let (b, h, qi) = (bh / H, bh % H, sample * 53 % S);
        let mut p = [0.0f32; S];
        let mut max = f32::NEG_INFINITY;
        for (ki, score) in p.iter_mut().enumerate() {
            let mut dot = 0.0f32;
            for d in 0..D {
                dot += f16_to_f32_bits(rotated[(b * S + qi) * 3 * C + h * D + d])
                    * f16_to_f32_bits(rotated[(b * S + ki) * 3 * C + C + h * D + d]);
            }
            *score = dot * scale;
            max = max.max(*score);
        }
        let mut sum = 0.0f32;
        for v in &mut p {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in &mut p {
            *v /= sum;
        }
        for d in 0..D {
            let mut value = 0.0f32;
            for (ki, &probability) in p.iter().enumerate() {
                value += probability
                    * f16_to_f32_bits(rotated[(b * S + ki) * 3 * C + 2 * C + h * D + d]);
            }
            worst = worst.max((value - reference[(b * S + qi) * C + h * D + d]).abs());
        }
    }
    assert!(
        worst <= 3e-5,
        "independent GPU/CPU FP32 reference mismatch: {worst}"
    );
    println!("CPU reference sampledRows=64 maxAbs={worst:.9}");
}

#[test]
fn fork_fa4_fp32_attention_probe() {
    if std::env::var("KATAGO_RUN_FORK_FA4_PROBE").as_deref() != Ok("1") {
        eprintln!("skipped: set KATAGO_RUN_FORK_FA4_PROBE=1 in an exclusive GPU window");
        return;
    }
    let variant = Fa4Variant::from_env();
    let directory = std::env::var_os("KATAGO_FORK_FA4_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/fork-parity-20260908")
                .join(variant.directory())
        });
    let paths: Vec<_> = std::fs::read_dir(&directory)
        .expect("copied fork artifact directory")
        .map(|p| p.expect("artifact entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "cubin"))
        .collect();
    assert_eq!(
        paths.len(),
        1,
        "expected the single hash-bound B14 FP32 CUBIN"
    );
    let binary = std::fs::read(&paths[0]).expect("read CUBIN");
    assert_eq!(
        hex::encode(Sha256::digest(&binary)),
        variant.cubin_sha256(),
        "CUBIN identity changed"
    );
    let iterations: usize = std::env::var("KATAGO_FORK_FA4_ITERATIONS")
        .unwrap_or_else(|_| "100".into())
        .parse()
        .expect("iterations integer");
    let cases: usize = std::env::var("KATAGO_FORK_FA4_CASES")
        .unwrap_or_else(|_| "3".into())
        .parse()
        .expect("cases integer");
    assert!((2..=10000).contains(&iterations) && (1..=3).contains(&cases));
    let rt = CudaRuntime::new().expect("CUDA runtime");
    assert_eq!(
        rt.device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .unwrap(),
        12
    );
    assert_eq!(
        rt.device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .unwrap(),
        0
    );
    let stream = rt.device.new_stream().expect("probe stream");
    // SAFETY: all probe buffers are used on this one stream and explicitly
    // synchronized before download/destruction; no work uses another stream.
    unsafe {
        rt.device.disable_event_tracking();
    }
    let module = rt
        .device
        .load_module(Ptx::from_binary(binary))
        .expect("load SM120 CUBIN on native Driver");
    let fa4 = module.load_function(variant.kernel()).expect("CUBIN entry");
    let q128 = rt.get_func("attention_fa2_kernel").expect("q128 kernel");
    let q64 = rt
        .get_func("attention_fa2_q64_serial_kernel")
        .expect("q64 serial kernel");
    let helpers = rt
        .device
        .load_module(
            compile_ptx_with_opts(
                REFERENCE_SOURCE,
                CompileOptions {
                    arch: Some("compute_120"),
                    fmad: Some(true),
                    ftz: Some(false),
                    prec_div: Some(true),
                    prec_sqrt: Some(true),
                    use_fast_math: Some(false),
                    ..Default::default()
                },
            )
            .expect("compile independent test reference"),
        )
        .expect("load reference module");
    let rope = helpers
        .load_function("probe_rope")
        .expect("standalone RoPE");
    let reference_function = helpers
        .load_function("probe_reference")
        .expect("reference function");
    let qk_scale = 1.0 / (D as f32).sqrt().sqrt();
    let scale = qk_scale * qk_scale;
    let mut report = json!({"schema":1,"artifactSha256":variant.cubin_sha256(),"batch":B,
        "heads":H,"sequence":S,"headDim":D,"scale":scale,"handles":1,
        "candidateId":variant.candidate_id(),"variant":variant.name(),
        "deviceAbi":variant.abi(),"kernelSymbol":variant.kernel(),
        "grid":[3,H,B],"block":[128,1,1],"dynamicSharedMemoryBytes":20480,
        "productionCertified":false,"fullModelVerified":false,
        "scope":"attention diagnostic only; production all-head gates still required",
        "cases":[]});
    let mut all_numeric_guards_pass = true;
    for seed in 0..cases {
        let mut state = 0x4b41_5441_4641_3400u64 ^ seed as u64;
        let mut random = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 40) as u32) as f32 / 16777216.0 - 0.5
        };
        let amplitude = [0.5f32, 2.0, 6.0][seed];
        let input: Vec<u16> = (0..3 * N)
            .map(|_| f32_to_f16_bits(random() * amplitude))
            .collect();
        let frequencies: Vec<_> = (0..H * 16).map(|_| (random(), random())).collect();
        let mut cos = Vec::with_capacity(S * H * 16);
        let mut sin = Vec::with_capacity(S * H * 16);
        for pos in 0..S {
            for &(fx, fy) in &frequencies {
                let angle = (pos % 19) as f32 * fx + (pos / 19) as f32 * fy;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        let d_input = stream.clone_htod(&input).expect("input");
        let d_cos = stream.clone_htod(&cos).expect("cos");
        let d_sin = stream.clone_htod(&sin).expect("sin");
        let mut d_rotated = stream.alloc_zeros::<u16>(3 * N).expect("rotated QKV");
        let mut d_q128 = stream.alloc_zeros::<u16>(N).expect("q128 output");
        let mut d_q64 = stream.alloc_zeros::<u16>(N).expect("q64 output");
        let mut d_fa4 = stream.alloc_zeros::<u16>(N).expect("FA4 output");
        let mut d_reference = stream.alloc_zeros::<f32>(N).expect("FP32 reference");
        let (input_ptr, _) = d_input.device_ptr(&stream);
        let (cos_ptr, _) = d_cos.device_ptr(&stream);
        let (sin_ptr, _) = d_sin.device_ptr(&stream);
        let (rotated_ptr, _) = d_rotated.device_ptr_mut(&stream);
        let (q128_ptr, _) = d_q128.device_ptr_mut(&stream);
        let (q64_ptr, _) = d_q64.device_ptr_mut(&stream);
        let (fa4_ptr, _) = d_fa4.device_ptr_mut(&stream);
        let (reference_ptr, _) = d_reference.device_ptr_mut(&stream);
        stream.synchronize().expect("uploads complete");
        let launch_rope = || unsafe {
            stream
                .launch_builder(&rope)
                .arg(&input_ptr)
                .arg(&cos_ptr)
                .arg(&sin_ptr)
                .arg(&rotated_ptr)
                .launch(LaunchConfig::for_num_elems((B * S * H * 16) as u32))
                .expect("RoPE launch");
        };
        launch_rope();
        unsafe {
            stream
                .launch_builder(&reference_function)
                .arg(&rotated_ptr)
                .arg(&reference_ptr)
                .arg(&scale)
                .launch(LaunchConfig {
                    grid_dim: (S as u32, (B * H) as u32, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
                .expect("FP32 reference launch");
        }
        launch_rust(
            &stream, &q128, input_ptr, cos_ptr, sin_ptr, q128_ptr, scale, 128, 256,
        );
        launch_rust(
            &stream, &q64, input_ptr, cos_ptr, sin_ptr, q64_ptr, scale, 64, 128,
        );
        launch_fork(&stream, &fa4, rotated_ptr, fa4_ptr, scale, variant);
        stream.synchronize().expect("correctness kernels complete");
        let reference = stream.clone_dtoh(&d_reference).expect("reference download");
        let rotated = stream.clone_dtoh(&d_rotated).expect("RoPE download");
        verify_reference_cpu(&rotated, &reference, scale);
        let out128 = stream.clone_dtoh(&d_q128).expect("q128 download");
        let out64 = stream.clone_dtoh(&d_q64).expect("q64 download");
        let outfa4 = stream.clone_dtoh(&d_fa4).expect("FA4 download");
        let serial_mismatches = out128.iter().zip(&out64).filter(|(a, b)| a != b).count();
        assert_eq!(
            serial_mismatches, 0,
            "established q128/q64 serial bitwise gate"
        );
        let numerical = json!({"q128":compare_half(&reference,&out128),
            "q64serial":compare_half(&reference,&out64),"forkFa4":compare_half(&reference,&outfa4),
            "forkVsQ128HalfBitMismatches":outfa4.iter().zip(&out128).filter(|(a,b)|a!=b).count()});
        println!(
            "FA4_NUMERIC variant={} seed={seed} amplitude={amplitude} {numerical}",
            variant.name()
        );
        // A diagnostic guard, stricter than existing attention-row smoke (.02).
        // It does not replace any native TF3 full-model output/Top1 gate.
        let numeric_guards_pass = ["q128", "q64serial", "forkFa4"].iter().all(|name| {
            numerical[*name]["maxAbs"].as_f64().unwrap() <= 0.005
                && numerical[*name]["rmse"].as_f64().unwrap() <= 0.0005
        });
        all_numeric_guards_pass &= numeric_guards_pass;
        let mut timings = Vec::new();
        // The first pair exposes the cost difference (Rust includes RoPE).
        // The second pair includes RoPE on both paths and is the fair comparison.
        for timing_case in [0, 2, 2, 0, 1, 3, 3, 1] {
            let label = [
                "rustQ128WithRope",
                "rustQ64SerialWithRope",
                "forkFa4PreRotated",
                "standaloneRopePlusForkFa4",
            ][timing_case];
            let timing = measure(
                &rt.device,
                &stream,
                label,
                iterations,
                || match timing_case {
                    0 => launch_rust(
                        &stream, &q128, input_ptr, cos_ptr, sin_ptr, q128_ptr, scale, 128, 256,
                    ),
                    1 => launch_rust(
                        &stream, &q64, input_ptr, cos_ptr, sin_ptr, q64_ptr, scale, 64, 128,
                    ),
                    2 => launch_fork(&stream, &fa4, rotated_ptr, fa4_ptr, scale, variant),
                    _ => {
                        launch_rope();
                        launch_fork(&stream, &fa4, rotated_ptr, fa4_ptr, scale, variant);
                    }
                },
            );
            println!("FA4_TIMING variant={} seed={seed} {timing}", variant.name());
            timings.push(timing);
        }
        report["cases"]
            .as_array_mut()
            .unwrap()
            .push(json!({"seed":seed,"amplitude":amplitude,
            "numeric":numerical,"numericGuardsPass":numeric_guards_pass,"timings":timings}));
        stream.synchronize().expect("all buffers safe to release");
    }
    let report_path = directory.join("probe-fork-fa4-report.json");
    std::fs::write(&report_path, serde_json::to_vec_pretty(&report).unwrap())
        .expect("write probe report");
    println!("FA4_REPORT {}", report_path.display());
    assert!(
        all_numeric_guards_pass,
        "attention diagnostic numeric guard failed; retained full report"
    );
}
