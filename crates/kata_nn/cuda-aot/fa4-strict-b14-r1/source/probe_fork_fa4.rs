//! Opt-in diagnostics for hash-bound SM120 FA4 FP32-accumulator CUBINs.
//! No production tactic/build changes. Root owns the exclusive GPU run window.
//!
//! Run with KATAGO_RUN_FORK_FA4_PROBE=1 and --features cuda --release --nocapture.
//! KATAGO_FORK_FA4_VARIANT=original (default) uses fa4-reference; `strict` uses
//! target/fork-parity-20260908/fa4-strict-reference-r2 with its own device ABI.
//! `strict-rope` uses fa4-strict-rope-smem with raw QKV and cached FP32 cos/sin.
//! KATAGO_FORK_FA4_DIR may override the selected artifact directory.
//! KATAGO_FORK_FA4_ITERATIONS defaults to 100; KATAGO_FORK_FA4_CASES to 3.
//! Separate `fork_fa4_batch_unrolled_rope_probe` requires variant=strict,
//! KATAGO_FORK_ROPE_LAYOUT=batch-unrolled-b14 and a fresh
//! KATAGO_FORK_ROPE_REPORT_DIR. Existing test/default helpers remain unchanged.
//! The copied artifact directories contain source, licenses, hashes and offline
//! ABI evidence. The original FP32 name describes QK/PV MMA accumulation only:
//! Original FA4 uses exp2 approximation and half P. The strict export changes
//! softmax arithmetic and scale ABI; it remains an unverified experiment until
//! its semantic audit and numerical/performance gates complete. This isolated
//! attention probe cannot grant whole-model or production certification.
#![cfg(feature = "cuda")]

use std::{path::PathBuf, sync::Arc, time::Instant};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr,
    LaunchConfig, PushKernelArg, sys,
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
const STRICT_ROPE_CUBIN_SHA256: &str =
    "0af1b2cf4b03e96baa1af2e9987f9b87842c87665006c739f1d02267d4c27464";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fa4Variant {
    Original,
    Strict,
    StrictRope,
}

impl Fa4Variant {
    fn from_env() -> Self {
        match std::env::var("KATAGO_FORK_FA4_VARIANT") {
            Err(std::env::VarError::NotPresent) => Self::Original,
            Ok(value) if value == "original" => Self::Original,
            Ok(value) if value == "strict" => Self::Strict,
            Ok(value) if value == "strict-rope" => Self::StrictRope,
            other => {
                panic!("KATAGO_FORK_FA4_VARIANT must be original, strict or strict-rope: {other:?}")
            }
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Original => "original",
            Self::Strict => "strict",
            Self::StrictRope => "strict-rope",
        }
    }

    fn directory(self) -> &'static str {
        match self {
            Self::Original => "fa4-reference",
            Self::Strict => "fa4-strict-reference-r2",
            Self::StrictRope => "fa4-strict-rope-smem",
        }
    }

    fn cubin_sha256(self) -> &'static str {
        match self {
            Self::Original => ORIGINAL_CUBIN_SHA256,
            Self::Strict => STRICT_CUBIN_SHA256,
            Self::StrictRope => STRICT_ROPE_CUBIN_SHA256,
        }
    }

    fn kernel(self) -> &'static str {
        match self {
            Self::Original => ORIGINAL_KERNEL,
            // Truncated generated symbols collide; the distinct CUBIN hash is
            // mandatory because the strict-rope device ABI has two extra args.
            Self::Strict | Self::StrictRope => STRICT_KERNEL,
        }
    }

    fn candidate_id(self) -> &'static str {
        match self {
            Self::Original => "fa4-b14-s361-h12-d32-tm128-tn96-s1-fp32",
            Self::Strict => "fa4-strict-r2-b14-s361-h12-d32-tm128-tn96-s1-fp32",
            Self::StrictRope => "fa4-strict-rope-smem-b14-s361-h12-d32-tm128-tn96-s1-fp32",
        }
    }

    fn abi(self) -> Value {
        match self {
            Self::Original => json!({"parameterBytes":[48,48,48,48,4,4,4,4,12],
                "scale":"natural scale multiplied by LOG2_E"}),
            Self::Strict => json!({"parameterBytes":[8,8,8,8,4,12],
                "scale":"natural FP32 scale, no LOG2_E conversion",
                "evidence":"probe-abi-manifest.json; r2 PTX/header/MLIR/host-object disassembly"}),
            Self::StrictRope => json!({"parameterBytes":[8,8,8,8,8,8,4,12],
                "parameters":["rawQ","rawK","V","O","cos","sin","naturalScale","fastDivmodOne"],
                "scale":"natural FP32 scale, no LOG2_E conversion",
                "ropeCacheShape":[S,H,D/2],"ropeCacheStrides":[H*D/2,D/2,1],
                "ropeCacheType":"FP32, shared across batch, same allocation as Rust attention",
                "registersPerThread":255,"staticSharedMemoryBytes":1024,"stackBytes":24,
                "evidence":"probe-abi-manifest.json; strict-rope PTX/header/host-object disassembly"}),
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
    cos: u64,
    sin: u64,
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
    if variant == Fa4Variant::StrictRope {
        // Static export: Q,K,V,O,cos,sin pointers; natural f32 scale; scheduler.
        // Host .o 0x3c0..0x514 supplies these eight arguments in this order.
        unsafe {
            stream
                .launch_builder(f)
                .arg(&input)
                .arg(&k)
                .arg(&v)
                .arg(&output)
                .arg(&cos)
                .arg(&sin)
                .arg(&scale)
                .arg(&fast_divmod_one)
                .launch(LaunchConfig {
                    grid_dim: (3, H as u32, B as u32),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 20480,
                })
                .expect("strict shared-memory RoPE FA4 Driver launch");
        }
        return;
    }
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

// Only appended for strict-rope. This independent helper exposes the contracted
// FP32-to-half boundary for a complete CPU half-bit check without modifying AOT.
// The AOT's internal shared memory is not observable here: its instruction audit
// and final attention reference gate remain separately required.
const STRICT_ROPE_REFERENCE_SOURCE: &str = r#"
extern "C" __global__ void probe_rope_contract(
    const unsigned short* x, const float* co, const float* sn, unsigned short* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= 14 * 361 * 192) return;
    int row=i/192, pair=i%192, r=(row%361)*192+pair;
    int offset=row*1152+pair*2;
    float c=co[r], s=sn[r];
    #pragma unroll
    for(int qk=0;qk<2;qk++) {
        int p=offset+qk*384;
        float a=h2f(x[p]), b=h2f(x[p+1]), ac, bs, as, u, v;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(ac) : "f"(a), "f"(c));
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(bs) : "f"(b), "f"(s));
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(as) : "f"(a), "f"(s));
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(u) : "f"(ac), "f"(bs));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(v) : "f"(b), "f"(c), "f"(as));
        y[p]=f2h(u); y[p+1]=f2h(v);
    }
    y[offset+768]=x[offset+768]; y[offset+769]=x[offset+769];
}
"#;

// Isolated G2 candidate: only move the batch dimension into this compile-time
// unrolled loop. Arithmetic, packed strides and V copies match the existing
// exact linear helper above. Coefficients are FP32 cached values, not sincosf.
const BATCH_UNROLLED_ROPE_SOURCE: &str = r#"
extern "C" __global__ void probe_rope_batch_unrolled_b14(
    const unsigned short* x, const float* co, const float* sn, unsigned short* y) {
    int xy=blockIdx.x, pair=threadIdx.x, r=xy*192+pair;
    float c=co[r], s=sn[r];
    #pragma unroll
    for(int n=0;n<14;n++) {
        int offset=(n*361+xy)*1152+pair*2;
        #pragma unroll
        for(int qk=0;qk<2;qk++) {
            int p=offset+qk*384;
            float a=h2f(x[p]), b=h2f(x[p+1]), ac, bs, as, u, v;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(ac) : "f"(a), "f"(c));
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(bs) : "f"(b), "f"(s));
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(as) : "f"(a), "f"(s));
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(u) : "f"(ac), "f"(bs));
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(v) : "f"(b), "f"(c), "f"(as));
            y[p]=f2h(u); y[p+1]=f2h(v);
        }
        y[offset+768]=x[offset+768]; y[offset+769]=x[offset+769];
    }
}
"#;

// Independent test oracle. Do not use the production f32_to_f16_bits here:
// its subnormal branch shifts the significand twice (cuda.rs:1095-1096),
// approximately halving values below 2^-14. Production/input generation stay
// unchanged in this diagnostic; this function fixes only the new boundary oracle.
fn reference_half_rne(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent_bits = (bits >> 23) & 0xff;
    let fraction = bits & 0x7fffff;
    if exponent_bits == 0xff {
        return sign | if fraction == 0 { 0x7c00 } else { 0x7e00 };
    }
    let exponent = exponent_bits as i32 - 127;
    if exponent < -25 {
        return sign;
    }
    if exponent > 15 {
        return sign | 0x7c00;
    }
    let significand = fraction | 0x800000;
    let shift = if exponent < -14 { -exponent - 1 } else { 13 } as u32;
    let quotient = significand >> shift;
    let remainder = significand & ((1u32 << shift) - 1);
    let midpoint = 1u32 << (shift - 1);
    let rounded =
        quotient + u32::from(remainder > midpoint || (remainder == midpoint && quotient & 1 != 0));
    let magnitude = if exponent < -14 {
        rounded
    } else {
        // Including the implicit bit allows rounding carry to propagate into
        // the exponent, including the finite-to-infinity boundary.
        (((exponent + 14) as u32) << 10) + rounded
    };
    sign | magnitude as u16
}

fn verify_half_rounding_oracle() {
    // Exhaust every finite half value and each adjacent finite-half midpoint,
    // checking the midpoint's two neighboring FP32 values and both signs.
    for low in 0u16..=0x7bff {
        let value = f16_to_f32_bits(low);
        assert_eq!(reference_half_rne(value), low);
        assert_eq!(reference_half_rne(-value), low | 0x8000);
        if low < 0x7bff {
            let high = f16_to_f32_bits(low + 1);
            let middle = ((value as f64 + high as f64) * 0.5) as f32;
            let expected_tie = if low & 1 == 0 { low } else { low + 1 };
            for (v, expected) in [
                (f32::from_bits(middle.to_bits() - 1), low),
                (middle, expected_tie),
                (f32::from_bits(middle.to_bits() + 1), low + 1),
            ] {
                assert_eq!(reference_half_rne(v), expected);
                assert_eq!(reference_half_rne(-v), expected | 0x8000);
            }
        }
    }
    for (value, expected) in [
        (65519.0, 0x7bff),
        (65520.0, 0x7c00),
        (f32::INFINITY, 0x7c00),
    ] {
        assert_eq!(reference_half_rne(value), expected);
        assert_eq!(reference_half_rne(-value), expected | 0x8000);
    }
    assert_eq!(reference_half_rne(f32::NAN) & 0x7fff, 0x7e00);
}

fn verify_rotation_half_cpu(input: &[u16], cos: &[f32], sin: &[f32], rotated: &[u16]) -> Value {
    assert_eq!(input.len(), 3 * N);
    assert_eq!(rotated.len(), input.len());
    let mut qk_mismatches = 0usize;
    let mut v_mismatches = 0usize;
    let mut first_mismatch = None;
    for row in 0..B * S {
        for pair in 0..C / 2 {
            let cache = (row % S) * (C / 2) + pair;
            let offset = row * 3 * C + pair * 2;
            for qk in 0..2 {
                let p = offset + qk * C;
                let a = f16_to_f32_bits(input[p]);
                let b = f16_to_f32_bits(input[p + 1]);
                // Rust FP operations keep separate products; only mul_add fuses.
                let ac = a * cos[cache];
                let bs = b * sin[cache];
                let a_sin = a * sin[cache];
                let expected = [
                    reference_half_rne(ac - bs),
                    reference_half_rne(b.mul_add(cos[cache], a_sin)),
                ];
                for d in 0..2 {
                    if rotated[p + d] != expected[d] {
                        qk_mismatches += 1;
                        first_mismatch.get_or_insert(json!({"element":p+d,
                            "expectedHalfBits":expected[d],"actualHalfBits":rotated[p+d]}));
                    }
                }
            }
            for d in 0..2 {
                let p = offset + 2 * C + d;
                v_mismatches += usize::from(input[p] != rotated[p]);
            }
        }
    }
    json!({"scope":"independent GPU helper vs CPU FP32/RNE contract; AOT internal shared memory is not dumped",
        "halfOracle":"test-local IEEE RNE; exhaustive finite-half roundtrip and midpoint-neighbor self-check",
        "qkHalfElements":2*N,"qkHalfBitMismatches":qk_mismatches,
        "vElements":N,"vCopyBitMismatches":v_mismatches,"firstMismatch":first_mismatch,
        "pass":qk_mismatches==0 && v_mismatches==0})
}

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
    if variant == Fa4Variant::StrictRope {
        verify_half_rounding_oracle();
    }
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
    let helper_source = if variant == Fa4Variant::StrictRope {
        format!("{REFERENCE_SOURCE}\n{STRICT_ROPE_REFERENCE_SOURCE}")
    } else {
        REFERENCE_SOURCE.to_owned()
    };
    let helpers = rt
        .device
        .load_module(
            compile_ptx_with_opts(
                &helper_source,
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
        .load_function(if variant == Fa4Variant::StrictRope {
            "probe_rope_contract"
        } else {
            "probe_rope"
        })
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
        let candidate_input = if variant == Fa4Variant::StrictRope {
            input_ptr
        } else {
            rotated_ptr
        };
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
        // Needed once to form the independent reference. Strict-rope candidate
        // launches always read raw input and never invoke this standalone helper.
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
        launch_fork(
            &stream,
            &fa4,
            candidate_input,
            fa4_ptr,
            scale,
            variant,
            cos_ptr,
            sin_ptr,
        );
        stream.synchronize().expect("correctness kernels complete");
        let reference = stream.clone_dtoh(&d_reference).expect("reference download");
        let rotated = stream.clone_dtoh(&d_rotated).expect("RoPE download");
        let rotation_boundary = if variant == Fa4Variant::StrictRope {
            verify_rotation_half_cpu(&input, &cos, &sin, &rotated)
        } else {
            Value::Null
        };
        verify_reference_cpu(&rotated, &reference, scale);
        let out128 = stream.clone_dtoh(&d_q128).expect("q128 download");
        let out64 = stream.clone_dtoh(&d_q64).expect("q64 download");
        let outfa4 = stream.clone_dtoh(&d_fa4).expect("FA4 download");
        let serial_mismatches = out128.iter().zip(&out64).filter(|(a, b)| a != b).count();
        assert_eq!(
            serial_mismatches, 0,
            "established q128/q64 serial bitwise gate"
        );
        let mut numerical = json!({"q128":compare_half(&reference,&out128),
            "q64serial":compare_half(&reference,&out64),"forkFa4":compare_half(&reference,&outfa4),
            "forkVsQ128HalfBitMismatches":outfa4.iter().zip(&out128).filter(|(a,b)|a!=b).count()});
        if variant == Fa4Variant::StrictRope {
            numerical["rotationHalfBoundary"] = rotation_boundary.clone();
        }
        println!(
            "FA4_NUMERIC variant={} seed={seed} amplitude={amplitude} {numerical}",
            variant.name()
        );
        // A diagnostic guard, stricter than existing attention-row smoke (.02).
        // It does not replace any native TF3 full-model output/Top1 gate.
        let numeric_guards_pass = ["q128", "q64serial", "forkFa4"].iter().all(|name| {
            numerical[*name]["maxAbs"].as_f64().unwrap() <= 0.005
                && numerical[*name]["rmse"].as_f64().unwrap() <= 0.0005
        }) && (variant != Fa4Variant::StrictRope
            || rotation_boundary["pass"] == true);
        all_numeric_guards_pass &= numeric_guards_pass;
        let mut timings = Vec::new();
        // The first pair exposes the cost difference (Rust includes RoPE).
        // The second pair includes RoPE on both paths and is the fair comparison.
        let timing_order = if variant == Fa4Variant::StrictRope {
            [0, 2, 2, 0, 1, 2, 2, 1]
        } else {
            [0, 2, 2, 0, 1, 3, 3, 1]
        };
        for timing_case in timing_order {
            // Preserve both established variants' diagnostics. Fused candidate
            // timing includes its internal RoPE, with no external RoPE launch.
            if variant == Fa4Variant::StrictRope && !numeric_guards_pass {
                break;
            }
            let label = if variant == Fa4Variant::StrictRope && timing_case == 2 {
                "strictFa4FusedRope"
            } else {
                [
                    "rustQ128WithRope",
                    "rustQ64SerialWithRope",
                    "forkFa4PreRotated",
                    "standaloneRopePlusForkFa4",
                ][timing_case]
            };
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
                    2 => launch_fork(
                        &stream,
                        &fa4,
                        candidate_input,
                        fa4_ptr,
                        scale,
                        variant,
                        cos_ptr,
                        sin_ptr,
                    ),
                    _ => {
                        launch_rope();
                        launch_fork(
                            &stream,
                            &fa4,
                            rotated_ptr,
                            fa4_ptr,
                            scale,
                            variant,
                            cos_ptr,
                            sin_ptr,
                        );
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

// This experiment deliberately uses a separate entry, fresh report directory,
// and the exact linear control. The established diagnostic above is unchanged.
struct RopeBatchCase {
    input: u64,
    cos: u64,
    sin: u64,
    linear: u64,
    unrolled: u64,
    q64_output: u64,
    linear_output: u64,
    unrolled_output: u64,
    _half_buffers: Vec<CudaSlice<u16>>,
    _coefficient_buffers: Vec<CudaSlice<f32>>,
}

fn launch_rope_batch_helper(
    stream: &CudaStream,
    function: &CudaFunction,
    case: &RopeBatchCase,
    unrolled: bool,
) {
    let output = if unrolled { case.unrolled } else { case.linear };
    let config = if unrolled {
        LaunchConfig {
            grid_dim: (S as u32, 1, 1),
            block_dim: ((H * D / 2) as u32, 1, 1),
            shared_mem_bytes: 0,
        }
    } else {
        LaunchConfig::for_num_elems((B * S * H * 16) as u32)
    };
    unsafe {
        stream
            .launch_builder(function)
            .arg(&case.input)
            .arg(&case.cos)
            .arg(&case.sin)
            .arg(&output)
            .launch(config)
            .expect("standalone exact RoPE helper launch");
    }
}

fn run_rope_batch_experiment(directory: &std::path::Path, iterations: usize, report: &mut Value) {
    verify_half_rounding_oracle();
    let artifact = std::env::var_os("KATAGO_FORK_FA4_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/fork-parity-20260908/fa4-strict-reference-r2")
        });
    let cubins: Vec<_> = std::fs::read_dir(&artifact)
        .expect("strict r2 artifact directory")
        .map(|entry| entry.expect("artifact entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "cubin"))
        .collect();
    assert_eq!(cubins.len(), 1, "exactly one strict r2 CUBIN required");
    let binary = std::fs::read(&cubins[0]).expect("read strict r2 CUBIN");
    assert_eq!(hex::encode(Sha256::digest(&binary)), STRICT_CUBIN_SHA256);
    report["artifactPath"] = json!(cubins[0]);
    report["backendBuild"] = json!(kata_nn::backends::cuda::backend_build_fingerprint());
    assert_eq!(report["backendBuild"]["fp16_encoding_revision"], 1);
    let executable = std::env::current_exe().expect("current probe executable");
    report["testExecutable"] = json!(executable);
    report["testExecutableSha256"] = json!(hex::encode(Sha256::digest(
        std::fs::read(&executable).expect("probe executable bytes")
    )));
    let helper_source =
        format!("{REFERENCE_SOURCE}\n{STRICT_ROPE_REFERENCE_SOURCE}\n{BATCH_UNROLLED_ROPE_SOURCE}");
    std::fs::write(directory.join("helpers.cu"), &helper_source).expect("record helper source");
    report["helperSourceSha256"] = json!(hex::encode(Sha256::digest(helper_source.as_bytes())));
    let ptx = compile_ptx_with_opts(
        &helper_source,
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
    .expect("compile exact RoPE experiment helpers");
    let ptx_source = ptx.to_src();
    std::fs::write(directory.join("helpers.ptx"), &ptx_source)
        .expect("record generated helper PTX");
    report["helperPtxSha256"] = json!(hex::encode(Sha256::digest(ptx_source.as_bytes())));
    let rt = CudaRuntime::new().expect("CUDA runtime");
    let major = rt
        .device
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .expect("compute capability major");
    let minor = rt
        .device
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .expect("compute capability minor");
    assert_eq!((major, minor), (12, 0), "SM120 artifact requires SM120");
    report["device"] = json!({"name":rt.device.name().expect("device name"),
        "ordinal":rt.device.ordinal(),"computeCapability":[major,minor]});
    let stream = rt.device.new_stream().expect("experiment stream");
    // All buffers stay alive until the single stream's final synchronization.
    unsafe { rt.device.disable_event_tracking() };
    let module = rt
        .device
        .load_module(Ptx::from_binary(binary))
        .expect("load unchanged strict r2");
    let fa4 = module
        .load_function(STRICT_KERNEL)
        .expect("strict r2 entry");
    let helpers = rt.device.load_module(ptx).expect("load exact helper PTX");
    let linear = helpers
        .load_function("probe_rope_contract")
        .expect("exact linear control");
    let unrolled = helpers
        .load_function("probe_rope_batch_unrolled_b14")
        .expect("B14 unrolled candidate");
    let reference_function = helpers
        .load_function("probe_reference")
        .expect("FP32 reference");
    let q128 = rt.get_func("attention_fa2_kernel").expect("q128 entry");
    let q64 = rt
        .get_func("attention_fa2_q64_serial_kernel")
        .expect("q64 serial entry");
    let qk_scale = 1.0 / (D as f32).sqrt().sqrt();
    let scale = qk_scale * qk_scale;
    let mut retained = Vec::new();
    let mut all_numeric_pass = true;
    // All three numerical cases finish before any timing begins.
    for seed in 0..3 {
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
        let input_bytes: Vec<_> = input.iter().flat_map(|value| value.to_le_bytes()).collect();
        let input_sha = hex::encode(Sha256::digest(&input_bytes));
        let d_input = stream.clone_htod(&input).expect("input upload");
        let d_cos = stream.clone_htod(&cos).expect("cos upload");
        let d_sin = stream.clone_htod(&sin).expect("sin upload");
        let mut d_linear = stream
            .alloc_zeros::<u16>(3 * N)
            .expect("linear rotated QKV");
        let mut d_unrolled = stream
            .alloc_zeros::<u16>(3 * N)
            .expect("unrolled rotated QKV");
        let mut d_out_linear = stream.alloc_zeros::<u16>(N).expect("linear FA4 output");
        let mut d_out_unrolled = stream.alloc_zeros::<u16>(N).expect("unrolled FA4 output");
        let mut d_q128 = stream.alloc_zeros::<u16>(N).expect("q128 output");
        let mut d_q64 = stream.alloc_zeros::<u16>(N).expect("q64 output");
        let mut d_reference = stream.alloc_zeros::<f32>(N).expect("FP32 reference");
        let (input_ptr, _) = d_input.device_ptr(&stream);
        let (cos_ptr, _) = d_cos.device_ptr(&stream);
        let (sin_ptr, _) = d_sin.device_ptr(&stream);
        let (linear_ptr, _) = d_linear.device_ptr_mut(&stream);
        let (unrolled_ptr, _) = d_unrolled.device_ptr_mut(&stream);
        let (out_linear_ptr, _) = d_out_linear.device_ptr_mut(&stream);
        let (out_unrolled_ptr, _) = d_out_unrolled.device_ptr_mut(&stream);
        let (q128_ptr, _) = d_q128.device_ptr_mut(&stream);
        let (q64_ptr, _) = d_q64.device_ptr_mut(&stream);
        let (reference_ptr, _) = d_reference.device_ptr_mut(&stream);
        let case = RopeBatchCase {
            input: input_ptr,
            cos: cos_ptr,
            sin: sin_ptr,
            linear: linear_ptr,
            unrolled: unrolled_ptr,
            q64_output: q64_ptr,
            linear_output: out_linear_ptr,
            unrolled_output: out_unrolled_ptr,
            _half_buffers: Vec::new(),
            _coefficient_buffers: Vec::new(),
        };
        stream.synchronize().expect("uploads complete");
        launch_rope_batch_helper(&stream, &linear, &case, false);
        launch_rope_batch_helper(&stream, &unrolled, &case, true);
        unsafe {
            stream
                .launch_builder(&reference_function)
                .arg(&linear_ptr)
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
        launch_fork(
            &stream,
            &fa4,
            linear_ptr,
            out_linear_ptr,
            scale,
            Fa4Variant::Strict,
            cos_ptr,
            sin_ptr,
        );
        launch_fork(
            &stream,
            &fa4,
            unrolled_ptr,
            out_unrolled_ptr,
            scale,
            Fa4Variant::Strict,
            cos_ptr,
            sin_ptr,
        );
        stream.synchronize().expect("all numeric kernels complete");
        let rotated_linear = stream.clone_dtoh(&d_linear).expect("linear QKV download");
        let rotated_unrolled = stream
            .clone_dtoh(&d_unrolled)
            .expect("unrolled QKV download");
        let reference = stream.clone_dtoh(&d_reference).expect("reference download");
        let out_linear = stream
            .clone_dtoh(&d_out_linear)
            .expect("linear FA4 download");
        let out_unrolled = stream
            .clone_dtoh(&d_out_unrolled)
            .expect("unrolled FA4 download");
        let out128 = stream.clone_dtoh(&d_q128).expect("q128 download");
        let out64 = stream.clone_dtoh(&d_q64).expect("q64 download");
        verify_reference_cpu(&rotated_linear, &reference, scale);
        let count_differences =
            |a: &[u16], b: &[u16]| a.iter().zip(b).filter(|(a, b)| a != b).count();
        let numeric = json!({
            "linearRotation":verify_rotation_half_cpu(&input,&cos,&sin,&rotated_linear),
            "unrolledRotation":verify_rotation_half_cpu(&input,&cos,&sin,&rotated_unrolled),
            "rotatedQkvHalfBitMismatches":count_differences(&rotated_linear,&rotated_unrolled),
            "fa4LinearVsUnrolledHalfBitMismatches":count_differences(&out_linear,&out_unrolled),
            "q128VsQ64SerialHalfBitMismatches":count_differences(&out128,&out64),
            "q128":compare_half(&reference,&out128),"q64serial":compare_half(&reference,&out64),
            "fa4Linear":compare_half(&reference,&out_linear),"fa4Unrolled":compare_half(&reference,&out_unrolled)});
        let passed = numeric["linearRotation"]["pass"] == true
            && numeric["unrolledRotation"]["pass"] == true
            && numeric["rotatedQkvHalfBitMismatches"] == 0
            && numeric["fa4LinearVsUnrolledHalfBitMismatches"] == 0
            && numeric["q128VsQ64SerialHalfBitMismatches"] == 0
            && ["q128", "q64serial", "fa4Linear", "fa4Unrolled"]
                .iter()
                .all(|key| {
                    numeric[*key]["maxAbs"].as_f64().unwrap() <= 0.005
                        && numeric[*key]["rmse"].as_f64().unwrap() <= 0.0005
                });
        all_numeric_pass &= passed;
        println!("ROPE_BATCH_NUMERIC seed={seed} amplitude={amplitude} pass={passed} {numeric}");
        report["cases"]
            .as_array_mut()
            .unwrap()
            .push(json!({"seed":seed,"amplitude":amplitude,
            "inputSha256":input_sha,"numeric":numeric,"numericGuardsPass":passed,"timings":[]}));
        retained.push(RopeBatchCase {
            _half_buffers: vec![
                d_input,
                d_linear,
                d_unrolled,
                d_out_linear,
                d_out_unrolled,
                d_q128,
                d_q64,
            ],
            _coefficient_buffers: vec![d_cos, d_sin],
            ..case
        });
    }
    report["allNumericGuardsPass"] = json!(all_numeric_pass);
    if !all_numeric_pass {
        report["status"] = json!("NUMERIC_REJECTED_NO_TIMING");
        return;
    }
    for (seed, case) in retained.iter().enumerate() {
        // ABBA groups: helper-only diagnosis; same FA4 core with two helpers;
        // full candidate including RoPE against the actual q64-serial baseline.
        for (index, timing_case) in [0, 1, 1, 0, 2, 3, 3, 2, 4, 3, 3, 4].into_iter().enumerate() {
            let label = [
                "linearExactRopeOnly",
                "batchUnrolledExactRopeOnly",
                "linearExactRopePlusStrictFa4",
                "batchUnrolledExactRopePlusStrictFa4",
                "rustQ64SerialWithRope",
            ][timing_case];
            let mut timing = measure(
                &rt.device,
                &stream,
                label,
                iterations,
                || match timing_case {
                    0 => launch_rope_batch_helper(&stream, &linear, case, false),
                    1 => launch_rope_batch_helper(&stream, &unrolled, case, true),
                    2 | 3 => {
                        let use_unrolled = timing_case == 3;
                        launch_rope_batch_helper(
                            &stream,
                            if use_unrolled { &unrolled } else { &linear },
                            case,
                            use_unrolled,
                        );
                        launch_fork(
                            &stream,
                            &fa4,
                            if use_unrolled {
                                case.unrolled
                            } else {
                                case.linear
                            },
                            if use_unrolled {
                                case.unrolled_output
                            } else {
                                case.linear_output
                            },
                            scale,
                            Fa4Variant::Strict,
                            case.cos,
                            case.sin,
                        );
                    }
                    _ => launch_rust(
                        &stream,
                        &q64,
                        case.input,
                        case.cos,
                        case.sin,
                        case.q64_output,
                        scale,
                        64,
                        128,
                    ),
                },
            );
            timing["abbaGroup"] =
                json!(["helper_only", "same_fa4_core", "candidate_vs_q64"][index / 4]);
            timing["orderIndex"] = json!(index % 4);
            println!("ROPE_BATCH_TIMING seed={seed} {timing}");
            report["cases"][seed]["timings"]
                .as_array_mut()
                .unwrap()
                .push(timing);
        }
    }
    stream
        .synchronize()
        .expect("all experiment buffers safe to release");
    report["status"] = json!("NUMERIC_PASS_TIMING_COMPLETE_UNINTERPRETED");
}

#[test]
fn fork_fa4_batch_unrolled_rope_probe() {
    if std::env::var("KATAGO_RUN_FORK_FA4_PROBE").as_deref() != Ok("1") {
        eprintln!("skipped: opt-in/exclusive GPU window required for RoPE batch experiment");
        return;
    }
    match std::env::var("KATAGO_FORK_ROPE_LAYOUT") {
        Err(std::env::VarError::NotPresent) => {
            eprintln!("skipped: set KATAGO_FORK_ROPE_LAYOUT=batch-unrolled-b14 explicitly");
            return;
        }
        Ok(value) if value == "batch-unrolled-b14" => {}
        other => panic!("KATAGO_FORK_ROPE_LAYOUT must be batch-unrolled-b14: {other:?}"),
    }
    assert!(
        Fa4Variant::from_env() == Fa4Variant::Strict,
        "this experiment requires KATAGO_FORK_FA4_VARIANT=strict"
    );
    let cases: usize = std::env::var("KATAGO_FORK_FA4_CASES")
        .unwrap_or_else(|_| "3".into())
        .parse()
        .expect("cases integer");
    assert_eq!(cases, 3, "fixed experiment requires all amplitudes 0.5/2/6");
    let iterations: usize = std::env::var("KATAGO_FORK_FA4_ITERATIONS")
        .unwrap_or_else(|_| "200".into())
        .parse()
        .expect("iterations integer");
    assert!(
        (200..=10000).contains(&iterations),
        "require 200..10000 iterations per ABBA phase"
    );
    let directory = PathBuf::from(
        std::env::var_os("KATAGO_FORK_ROPE_REPORT_DIR").expect("fresh report directory required"),
    );
    if let Some(parent) = directory.parent() {
        std::fs::create_dir_all(parent).expect("report parent directory");
    }
    std::fs::create_dir(&directory).expect("refuse overwrite: report directory must be new");
    let mut report = json!({"schema":1,"status":"STARTED_NOT_VERIFIED",
        "experiment":"standalone-rope-b14-compile-time-batch-unroll-r1",
        "artifactSha256":STRICT_CUBIN_SHA256,"deviceAbi":Fa4Variant::Strict.abi(),
        "kernelSymbol":STRICT_KERNEL,"batch":B,"heads":H,"sequence":S,"headDim":D,"handles":1,
        "grid":[3,H,B],"block":[128,1,1],"dynamicSharedMemoryBytes":20480,
        "unrolledHelperGrid":[S,1,1],"unrolledHelperBlock":[H*D/2,1,1],
        "control":"existing exact linear probe_rope_contract; not implicit-FMA legacy probe_rope",
        "arithmetic":"same explicit FP32 mul/sub/FMA and half RN; FP32 cached cos/sin; V copies retained",
        "sourceSha256":hex::encode(Sha256::digest(include_bytes!("probe_fork_fa4.rs"))),
        "controlHelperSourceSha256":hex::encode(Sha256::digest(STRICT_ROPE_REFERENCE_SOURCE.as_bytes())),
        "candidateHelperSourceSha256":hex::encode(Sha256::digest(BATCH_UNROLLED_ROPE_SOURCE.as_bytes())),
        "warmupPerPhase":32,"iterationsPerPhase":iterations,"maximumRelativeSpread":0.05,
        "minimumImprovement":0.01,"productionCertified":false,"fullModelVerified":false,
        "timingScope":"single-stream CUDA event queue; each ABBA group has two independent phase medians per arm",
        "numericScope":"complete QKV half-bit and attention oracle; all 3 amplitudes pass before any timing",
        "allNumericGuardsPass":false,"cases":[]});
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_rope_batch_experiment(&directory, iterations, &mut report);
    }));
    if let Err(payload) = &outcome {
        let error = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("non-string panic");
        report["status"] = json!("FAILED_EVIDENCE_RETAINED");
        report["error"] = json!(error);
    }
    let path = directory.join("report.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&report).unwrap())
        .expect("write fresh experiment report");
    println!("ROPE_BATCH_REPORT {}", path.display());
    if let Err(payload) = outcome {
        std::panic::resume_unwind(payload);
    }
    assert!(
        report["allNumericGuardsPass"] == true,
        "numerical failure; report retained and no timing performed"
    );
}

// Independent G2 factor: retain exact linear scheduling and Q/K arithmetic,
// omit only V loads/stores. The strict AOT receives V from the original input.
const LINEAR_NO_VCOPY_ROPE_SOURCE: &str = r#"
extern "C" __global__ void probe_rope_no_vcopy(
    const unsigned short* x, const float* co, const float* sn, unsigned short* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= 14 * 361 * 192) return;
    int row=i/192, pair=i%192, r=(row%361)*192+pair;
    int offset=row*1152+pair*2;
    float c=co[r], s=sn[r];
    #pragma unroll
    for(int qk=0;qk<2;qk++) {
        int p=offset+qk*384;
        float a=h2f(x[p]), b=h2f(x[p+1]), ac, bs, as, u, v;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(ac) : "f"(a), "f"(c));
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(bs) : "f"(b), "f"(s));
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(as) : "f"(a), "f"(s));
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(u) : "f"(ac), "f"(bs));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(v) : "f"(b), "f"(c), "f"(as));
        y[p]=f2h(u); y[p+1]=f2h(v);
    }
}
"#;

// Strict r2 ABI is unchanged: only the V pointer names the original allocation.
// Q/K still use the original packed row/head/batch strides in rotated scratch.
fn launch_fork_strict_raw_v(
    stream: &CudaStream,
    f: &CudaFunction,
    rotated: u64,
    raw_input: u64,
    output: u64,
    scale: f32,
) {
    let k = rotated + (C * 2) as u64;
    let v = raw_input + (C * 4) as u64;
    let fast_divmod_one = [1u32, 1, 0];
    unsafe {
        stream
            .launch_builder(f)
            .arg(&rotated)
            .arg(&k)
            .arg(&v)
            .arg(&output)
            .arg(&scale)
            .arg(&fast_divmod_one)
            .launch(LaunchConfig {
                grid_dim: (3, H as u32, B as u32),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 20480,
            })
            .expect("strict r2 FA4 raw V Driver launch");
    }
}

const NO_VCOPY_SENTINEL: u16 = 0x7e35;

fn verify_no_vcopy_half_cpu(
    input: &[u16],
    cos: &[f32],
    sin: &[f32],
    linear: &[u16],
    no_vcopy: &[u16],
    input_after: &[u16],
) -> Value {
    for buffer in [input, linear, no_vcopy, input_after] {
        assert_eq!(buffer.len(), 3 * N, "complete packed QKV allocation");
    }
    // Reuse only the full independent Q/K oracle result. Its V-copy guard is
    // intentionally irrelevant here: the candidate must leave V unwritten.
    let qk_oracle = verify_rotation_half_cpu(input, cos, sin, no_vcopy);
    let mut qk_control_mismatches = 0usize;
    let mut v_source_mismatches = 0usize;
    let mut v_sentinel_writes = 0usize;
    let mut original_v_mismatches = 0usize;
    for row in 0..B * S {
        let base = row * 3 * C;
        for d in 0..2 * C {
            qk_control_mismatches += usize::from(linear[base + d] != no_vcopy[base + d]);
        }
        for d in 2 * C..3 * C {
            let i = base + d;
            v_source_mismatches += usize::from(input_after[i] != linear[i]);
            v_sentinel_writes += usize::from(no_vcopy[i] != NO_VCOPY_SENTINEL);
            original_v_mismatches += usize::from(input[i] != input_after[i]);
        }
    }
    let input_mismatches = input
        .iter()
        .zip(input_after)
        .filter(|(a, b)| a != b)
        .count();
    let pass = qk_oracle["qkHalfBitMismatches"] == 0
        && qk_control_mismatches == 0
        && v_source_mismatches == 0
        && v_sentinel_writes == 0
        && original_v_mismatches == 0
        && input_mismatches == 0;
    json!({"scope":"all Q/K bits vs independent CPU FP32/RNE; all V source/canary and original input bits",
        "qkHalfElements":2*N,"qkHalfBitMismatches":qk_oracle["qkHalfBitMismatches"],
        "qkFirstMismatch":qk_oracle["firstMismatch"],
        "qkControlHalfBitMismatches":qk_control_mismatches,
        "vElements":N,"rawVSourceVsControlHalfBitMismatches":v_source_mismatches,
        "vSentinelHalfBits":NO_VCOPY_SENTINEL,"vSentinelUnexpectedWrites":v_sentinel_writes,
        "originalVHalfBitMismatches":original_v_mismatches,
        "originalInputHalfElements":3*N,"originalInputHalfBitMismatches":input_mismatches,"pass":pass})
}

struct RopeNoVcopyCase {
    input: u64,
    cos: u64,
    sin: u64,
    linear: u64,
    no_vcopy: u64,
    q64_output: u64,
    linear_output: u64,
    no_vcopy_output: u64,
    _half_buffers: Vec<CudaSlice<u16>>,
    _coefficient_buffers: Vec<CudaSlice<f32>>,
}

fn launch_rope_no_vcopy_helper(
    stream: &CudaStream,
    function: &CudaFunction,
    case: &RopeNoVcopyCase,
    no_vcopy: bool,
) {
    let output = if no_vcopy { case.no_vcopy } else { case.linear };
    // The existing exact linear control uses cudarc's 1024-thread default.
    // Both arms retain it; changing block size would be a second factor.
    let config = LaunchConfig::for_num_elems((B * S * H * 16) as u32);
    unsafe {
        stream
            .launch_builder(function)
            .arg(&case.input)
            .arg(&case.cos)
            .arg(&case.sin)
            .arg(&output)
            .launch(config)
            .expect("standalone exact RoPE helper launch");
    }
}

fn run_rope_no_vcopy_experiment(
    directory: &std::path::Path,
    iterations: usize,
    report: &mut Value,
) {
    verify_half_rounding_oracle();
    let artifact = std::env::var_os("KATAGO_FORK_FA4_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/fork-parity-20260908/fa4-strict-reference-r2")
        });
    let cubins: Vec<_> = std::fs::read_dir(&artifact)
        .expect("strict r2 artifact directory")
        .map(|entry| entry.expect("artifact entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "cubin"))
        .collect();
    assert_eq!(cubins.len(), 1, "exactly one strict r2 CUBIN required");
    let binary = std::fs::read(&cubins[0]).expect("read strict r2 CUBIN");
    assert_eq!(hex::encode(Sha256::digest(&binary)), STRICT_CUBIN_SHA256);
    report["artifactPath"] = json!(cubins[0]);
    report["backendBuild"] = json!(kata_nn::backends::cuda::backend_build_fingerprint());
    assert_eq!(report["backendBuild"]["fp16_encoding_revision"], 1);
    let executable = std::env::current_exe().expect("current probe executable");
    report["testExecutable"] = json!(executable);
    report["testExecutableSha256"] = json!(hex::encode(Sha256::digest(
        std::fs::read(&executable).expect("probe executable bytes")
    )));
    let helper_source = format!(
        "{REFERENCE_SOURCE}\n{STRICT_ROPE_REFERENCE_SOURCE}\n{LINEAR_NO_VCOPY_ROPE_SOURCE}"
    );
    std::fs::write(directory.join("helpers.cu"), &helper_source).expect("record helper source");
    report["helperSourceSha256"] = json!(hex::encode(Sha256::digest(helper_source.as_bytes())));
    let ptx = compile_ptx_with_opts(
        &helper_source,
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
    .expect("compile exact RoPE experiment helpers");
    let ptx_source = ptx.to_src();
    std::fs::write(directory.join("helpers.ptx"), &ptx_source)
        .expect("record generated helper PTX");
    report["helperPtxSha256"] = json!(hex::encode(Sha256::digest(ptx_source.as_bytes())));
    let rt = CudaRuntime::new().expect("CUDA runtime");
    let major = rt
        .device
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .expect("compute capability major");
    let minor = rt
        .device
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .expect("compute capability minor");
    assert_eq!((major, minor), (12, 0), "SM120 artifact requires SM120");
    report["device"] = json!({"name":rt.device.name().expect("device name"),
        "ordinal":rt.device.ordinal(),"computeCapability":[major,minor]});
    let stream = rt.device.new_stream().expect("experiment stream");
    // All buffers stay alive until the single stream's final synchronization.
    unsafe { rt.device.disable_event_tracking() };
    let module = rt
        .device
        .load_module(Ptx::from_binary(binary))
        .expect("load unchanged strict r2");
    let fa4 = module
        .load_function(STRICT_KERNEL)
        .expect("strict r2 entry");
    let helpers = rt.device.load_module(ptx).expect("load exact helper PTX");
    let linear = helpers
        .load_function("probe_rope_contract")
        .expect("exact linear control");
    let no_vcopy = helpers
        .load_function("probe_rope_no_vcopy")
        .expect("exact linear QK-only candidate");
    let reference_function = helpers
        .load_function("probe_reference")
        .expect("FP32 reference");
    let q128 = rt.get_func("attention_fa2_kernel").expect("q128 entry");
    let q64 = rt
        .get_func("attention_fa2_q64_serial_kernel")
        .expect("q64 serial entry");
    let qk_scale = 1.0 / (D as f32).sqrt().sqrt();
    let scale = qk_scale * qk_scale;
    let mut retained = Vec::new();
    let mut all_numeric_pass = true;
    // All three numerical cases finish before any timing begins.
    for seed in 0..3 {
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
        let input_bytes: Vec<_> = input.iter().flat_map(|value| value.to_le_bytes()).collect();
        let input_sha = hex::encode(Sha256::digest(&input_bytes));
        let d_input = stream.clone_htod(&input).expect("input upload");
        let d_cos = stream.clone_htod(&cos).expect("cos upload");
        let d_sin = stream.clone_htod(&sin).expect("sin upload");
        let mut d_linear = stream
            .alloc_zeros::<u16>(3 * N)
            .expect("linear rotated QKV");
        // Dedicated probe scratch: NaN half sentinels make any V write visible.
        // The candidate AOT must obtain V from d_input, never this region.
        let mut d_no_vcopy = stream
            .clone_htod(&vec![NO_VCOPY_SENTINEL; 3 * N])
            .expect("QK-only scratch with unwritten V sentinels");
        let mut d_out_linear = stream.alloc_zeros::<u16>(N).expect("linear FA4 output");
        let mut d_out_no_vcopy = stream.alloc_zeros::<u16>(N).expect("no_vcopy FA4 output");
        let mut d_q128 = stream.alloc_zeros::<u16>(N).expect("q128 output");
        let mut d_q64 = stream.alloc_zeros::<u16>(N).expect("q64 output");
        let mut d_reference = stream.alloc_zeros::<f32>(N).expect("FP32 reference");
        let (input_ptr, _) = d_input.device_ptr(&stream);
        let (cos_ptr, _) = d_cos.device_ptr(&stream);
        let (sin_ptr, _) = d_sin.device_ptr(&stream);
        let (linear_ptr, _) = d_linear.device_ptr_mut(&stream);
        let (no_vcopy_ptr, _) = d_no_vcopy.device_ptr_mut(&stream);
        let (out_linear_ptr, _) = d_out_linear.device_ptr_mut(&stream);
        let (out_no_vcopy_ptr, _) = d_out_no_vcopy.device_ptr_mut(&stream);
        let (q128_ptr, _) = d_q128.device_ptr_mut(&stream);
        let (q64_ptr, _) = d_q64.device_ptr_mut(&stream);
        let (reference_ptr, _) = d_reference.device_ptr_mut(&stream);
        let case = RopeNoVcopyCase {
            input: input_ptr,
            cos: cos_ptr,
            sin: sin_ptr,
            linear: linear_ptr,
            no_vcopy: no_vcopy_ptr,
            q64_output: q64_ptr,
            linear_output: out_linear_ptr,
            no_vcopy_output: out_no_vcopy_ptr,
            _half_buffers: Vec::new(),
            _coefficient_buffers: Vec::new(),
        };
        stream.synchronize().expect("uploads complete");
        launch_rope_no_vcopy_helper(&stream, &linear, &case, false);
        launch_rope_no_vcopy_helper(&stream, &no_vcopy, &case, true);
        unsafe {
            stream
                .launch_builder(&reference_function)
                .arg(&linear_ptr)
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
        launch_fork(
            &stream,
            &fa4,
            linear_ptr,
            out_linear_ptr,
            scale,
            Fa4Variant::Strict,
            cos_ptr,
            sin_ptr,
        );
        launch_fork_strict_raw_v(
            &stream,
            &fa4,
            no_vcopy_ptr,
            input_ptr,
            out_no_vcopy_ptr,
            scale,
        );
        stream.synchronize().expect("all numeric kernels complete");
        let rotated_linear = stream.clone_dtoh(&d_linear).expect("linear QKV download");
        let rotated_no_vcopy = stream
            .clone_dtoh(&d_no_vcopy)
            .expect("no_vcopy QKV download");
        let reference = stream.clone_dtoh(&d_reference).expect("reference download");
        let out_linear = stream
            .clone_dtoh(&d_out_linear)
            .expect("linear FA4 download");
        let out_no_vcopy = stream
            .clone_dtoh(&d_out_no_vcopy)
            .expect("no_vcopy FA4 download");
        let out128 = stream.clone_dtoh(&d_q128).expect("q128 download");
        let out64 = stream.clone_dtoh(&d_q64).expect("q64 download");
        let input_after = stream
            .clone_dtoh(&d_input)
            .expect("original input after candidate");
        verify_reference_cpu(&rotated_linear, &reference, scale);
        let count_differences =
            |a: &[u16], b: &[u16]| a.iter().zip(b).filter(|(a, b)| a != b).count();
        let numeric = json!({
            "linearRotation":verify_rotation_half_cpu(&input,&cos,&sin,&rotated_linear),
            "noVcopyBoundary":verify_no_vcopy_half_cpu(&input,&cos,&sin,&rotated_linear,&rotated_no_vcopy,&input_after),
            "fa4LinearVsNoVcopyHalfBitMismatches":count_differences(&out_linear,&out_no_vcopy),
            "q128VsQ64SerialHalfBitMismatches":count_differences(&out128,&out64),
            "q128":compare_half(&reference,&out128),"q64serial":compare_half(&reference,&out64),
            "fa4Linear":compare_half(&reference,&out_linear),"fa4NoVcopy":compare_half(&reference,&out_no_vcopy)});
        let passed = numeric["linearRotation"]["pass"] == true
            && numeric["noVcopyBoundary"]["pass"] == true
            && numeric["fa4LinearVsNoVcopyHalfBitMismatches"] == 0
            && numeric["q128VsQ64SerialHalfBitMismatches"] == 0
            && ["q128", "q64serial", "fa4Linear", "fa4NoVcopy"]
                .iter()
                .all(|key| {
                    numeric[*key]["maxAbs"].as_f64().unwrap() <= 0.005
                        && numeric[*key]["rmse"].as_f64().unwrap() <= 0.0005
                });
        all_numeric_pass &= passed;
        println!("ROPE_NO_VCOPY_NUMERIC seed={seed} amplitude={amplitude} pass={passed} {numeric}");
        report["cases"]
            .as_array_mut()
            .unwrap()
            .push(json!({"seed":seed,"amplitude":amplitude,
            "inputSha256":input_sha,"numeric":numeric,"numericGuardsPass":passed,"timings":[]}));
        retained.push(RopeNoVcopyCase {
            _half_buffers: vec![
                d_input,
                d_linear,
                d_no_vcopy,
                d_out_linear,
                d_out_no_vcopy,
                d_q128,
                d_q64,
            ],
            _coefficient_buffers: vec![d_cos, d_sin],
            ..case
        });
    }
    report["allNumericGuardsPass"] = json!(all_numeric_pass);
    if !all_numeric_pass {
        report["status"] = json!("NUMERIC_REJECTED_NO_TIMING");
        return;
    }
    for (seed, case) in retained.iter().enumerate() {
        // ABBA groups: helper-only diagnosis; same FA4 core with two helpers;
        // full candidate including RoPE against the actual q64-serial baseline.
        for (index, timing_case) in [0, 1, 1, 0, 2, 3, 3, 2, 4, 3, 3, 4].into_iter().enumerate() {
            let label = [
                "linearExactRopeOnly",
                "linearNoVcopyExactRopeOnly",
                "linearExactRopePlusStrictFa4",
                "linearNoVcopyExactRopePlusStrictFa4",
                "rustQ64SerialWithRope",
            ][timing_case];
            let mut timing =
                measure_no_vcopy(
                    &rt.device,
                    &stream,
                    label,
                    iterations,
                    || match timing_case {
                        0 => launch_rope_no_vcopy_helper(&stream, &linear, case, false),
                        1 => launch_rope_no_vcopy_helper(&stream, &no_vcopy, case, true),
                        2 | 3 => {
                            let use_no_vcopy = timing_case == 3;
                            launch_rope_no_vcopy_helper(
                                &stream,
                                if use_no_vcopy { &no_vcopy } else { &linear },
                                case,
                                use_no_vcopy,
                            );
                            if use_no_vcopy {
                                launch_fork_strict_raw_v(
                                    &stream,
                                    &fa4,
                                    case.no_vcopy,
                                    case.input,
                                    case.no_vcopy_output,
                                    scale,
                                );
                            } else {
                                launch_fork(
                                    &stream,
                                    &fa4,
                                    case.linear,
                                    case.linear_output,
                                    scale,
                                    Fa4Variant::Strict,
                                    case.cos,
                                    case.sin,
                                );
                            }
                        }
                        _ => launch_rust(
                            &stream,
                            &q64,
                            case.input,
                            case.cos,
                            case.sin,
                            case.q64_output,
                            scale,
                            64,
                            128,
                        ),
                    },
                );
            timing["abbaGroup"] =
                json!(["helper_only", "same_fa4_core", "candidate_vs_q64"][index / 4]);
            timing["orderIndex"] = json!(index % 4);
            println!("ROPE_NO_VCOPY_TIMING seed={seed} {timing}");
            report["cases"][seed]["timings"]
                .as_array_mut()
                .unwrap()
                .push(timing);
        }
        // Outside all event measurements: verify repeated launches preserved
        // original input and never wrote the scratch V region either.
        let input_after = stream
            .clone_dtoh(&case._half_buffers[0])
            .expect("post-timing input");
        let scratch_after = stream
            .clone_dtoh(&case._half_buffers[2])
            .expect("post-timing scratch");
        let input_bytes: Vec<_> = input_after.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_sha = hex::encode(Sha256::digest(&input_bytes));
        let sentinel_writes = (0..B * S)
            .flat_map(|row| (2 * C..3 * C).map(move |d| row * 3 * C + d))
            .filter(|&i| scratch_after[i] != NO_VCOPY_SENTINEL)
            .count();
        let passed = report["cases"][seed]["inputSha256"].as_str() == Some(input_sha.as_str())
            && sentinel_writes == 0;
        let post = json!({"inputSha256":input_sha,"vSentinelUnexpectedWrites":sentinel_writes,"pass":passed});
        println!("ROPE_NO_VCOPY_POST_TIMING seed={seed} {post}");
        report["cases"][seed]["postTimingGuards"] = post;
        assert!(passed, "post-timing input/scratch boundary changed");
    }
    stream
        .synchronize()
        .expect("all experiment buffers safe to release");
    report["status"] = json!("NUMERIC_PASS_TIMING_COMPLETE_UNINTERPRETED");
}

#[test]
fn fork_fa4_no_vcopy_rope_probe() {
    if std::env::var("KATAGO_RUN_FORK_FA4_PROBE").as_deref() != Ok("1") {
        eprintln!("skipped: opt-in/exclusive GPU window required for RoPE batch experiment");
        return;
    }
    match std::env::var("KATAGO_FORK_ROPE_LAYOUT") {
        Err(std::env::VarError::NotPresent) => {
            eprintln!("skipped: set KATAGO_FORK_ROPE_LAYOUT=linear-no-vcopy explicitly");
            return;
        }
        Ok(value) if value == "linear-no-vcopy" => {}
        other => panic!("KATAGO_FORK_ROPE_LAYOUT must be linear-no-vcopy: {other:?}"),
    }
    assert!(
        Fa4Variant::from_env() == Fa4Variant::Strict,
        "this experiment requires KATAGO_FORK_FA4_VARIANT=strict"
    );
    let cases: usize = std::env::var("KATAGO_FORK_FA4_CASES")
        .unwrap_or_else(|_| "3".into())
        .parse()
        .expect("cases integer");
    assert_eq!(cases, 3, "fixed experiment requires all amplitudes 0.5/2/6");
    let iterations: usize = std::env::var("KATAGO_FORK_FA4_ITERATIONS")
        .unwrap_or_else(|_| "200".into())
        .parse()
        .expect("iterations integer");
    assert!(
        (200..=10000).contains(&iterations),
        "require 200..10000 iterations per ABBA phase"
    );
    let directory = PathBuf::from(
        std::env::var_os("KATAGO_FORK_ROPE_REPORT_DIR").expect("fresh report directory required"),
    );
    if let Some(parent) = directory.parent() {
        std::fs::create_dir_all(parent).expect("report parent directory");
    }
    std::fs::create_dir(&directory).expect("refuse overwrite: report directory must be new");
    let mut report = json!({"schema":1,"status":"STARTED_NOT_VERIFIED",
        "experiment":"standalone-rope-linear-no-vcopy-b14-r1",
        "artifactSha256":STRICT_CUBIN_SHA256,"deviceAbi":Fa4Variant::Strict.abi(),
        "kernelSymbol":STRICT_KERNEL,"batch":B,"heads":H,"sequence":S,"headDim":D,"handles":1,
        "grid":[3,H,B],"block":[128,1,1],"dynamicSharedMemoryBytes":20480,
        "controlHelperGrid":[(B*S*H*16).div_ceil(1024),1,1],"controlHelperBlock":[1024,1,1],
        "candidateHelperGrid":[(B*S*H*16).div_ceil(1024),1,1],"candidateHelperBlock":[1024,1,1],
        "vPointerControl":"rotated scratch + 768 half elements",
        "vPointerCandidate":"original packed input + 768 half elements; same fixed strides",
        "vSentinelHalfBits":NO_VCOPY_SENTINEL,
        "control":"existing exact linear probe_rope_contract; not implicit-FMA legacy probe_rope",
        "arithmetic":"same explicit FP32 mul/sub/FMA and half RN; FP32 cached cos/sin; only candidate V copy removed",
        "sourceSha256":hex::encode(Sha256::digest(include_bytes!("probe_fork_fa4.rs"))),
        "controlHelperSourceSha256":hex::encode(Sha256::digest(STRICT_ROPE_REFERENCE_SOURCE.as_bytes())),
        "candidateHelperSourceSha256":hex::encode(Sha256::digest(LINEAR_NO_VCOPY_ROPE_SOURCE.as_bytes())),
        "warmupPerPhase":32,"iterationsPerPhase":iterations,"maximumRelativeSpread":0.05,
        "minimumImprovement":0.01,"phaseMetric":"cudaEventMedianMs",
        "armAggregation":"geometric mean of two phase medians",
        "relativeSpreadFormula":"max/min-1","rateRatioFormula":"control geomean / candidate geomean",
        "productionCertified":false,"fullModelVerified":false,
        "timingScope":"single-stream CUDA event queue; each ABBA group has two independent phase medians per arm",
        "numericScope":"complete QK CPU/RNE bits, V source/sentinel and original input bits, attention oracle; all 3 amplitudes pass before timing",
        "allNumericGuardsPass":false,"cases":[]});
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_rope_no_vcopy_experiment(&directory, iterations, &mut report);
    }));
    if let Err(payload) = &outcome {
        let error = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("non-string panic");
        report["status"] = json!("FAILED_EVIDENCE_RETAINED");
        report["error"] = json!(error);
    }
    let path = directory.join("report.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&report).unwrap())
        .expect("write fresh experiment report");
    println!("ROPE_NO_VCOPY_REPORT {}", path.display());
    if let Err(payload) = outcome {
        std::panic::resume_unwind(payload);
    }
    assert!(
        report["allNumericGuardsPass"] == true,
        "numerical failure; report retained and no timing performed"
    );
}

// New experiment retains every event duration so its registered phase median
// can be independently recomputed. Existing experiment timing stays unchanged.
fn measure_no_vcopy(
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
    let raw_times = times.clone();
    times.sort_by(f64::total_cmp);
    let median = if iterations % 2 == 1 {
        times[iterations / 2]
    } else {
        (times[iterations / 2 - 1] + times[iterations / 2]) * 0.5
    };
    json!({"candidate":label,"iterations":iterations,"cudaEventMedianMs":median,
        "cudaEventSamplesMs":raw_times,"sampleOrder":"original event enqueue order",
        "cudaEventMeanMs":times.iter().sum::<f64>()/iterations as f64,
        "wallMeanMs":wall_ms/iterations as f64})
}
