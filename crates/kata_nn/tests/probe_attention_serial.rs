//! Opt-in bitwise diagnostic for q64-serial versus the established q128 kernel.
//! Set KATAGO_RUN_ATTN_PROBE=1; root controls the exclusive GPU test window.
#![cfg(feature = "cuda")]

use cudarc::driver::{LaunchConfig, PushKernelArg};
use kata_nn::backends::cuda::{CudaRuntime, f16_to_f32_bits, f32_to_f16_bits};

#[test]
fn q64_serial_matches_q128_half_output() {
    if std::env::var("KATAGO_RUN_ATTN_PROBE").as_deref() != Ok("1") {
        eprintln!("skipped: set KATAGO_RUN_ATTN_PROBE=1 for attention diagnostic");
        return;
    }
    let rt = CudaRuntime::new().expect("CUDA runtime");
    let stream = rt.device.default_stream();
    let s = 361usize;
    let heads = 12usize;
    let d = 32usize;
    // Match native lower + executor, which square the per-Q/K scale.
    let qk_scale = 1.0 / (d as f32).sqrt().sqrt();
    let scale = qk_scale * qk_scale;
    for seed in 0..3u64 {
        let mut state = 0x5446_3352_4f50_4500 ^ seed;
        let mut random = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 40) as u32) as f32 / 16_777_216.0 - 0.5
        };
        // Learned-style 2D RoPE: f32 tables, FP32 rotation, then half Q/K.
        let frequencies: Vec<(f32, f32)> = (0..heads * 16).map(|_| (random(), random())).collect();
        let mut cos = Vec::with_capacity(s * heads * 16);
        let mut sin = Vec::with_capacity(s * heads * 16);
        for pos in 0..s {
            for &(fx, fy) in &frequencies {
                let angle = (pos % 19) as f32 * fx + (pos / 19) as f32 * fy;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        let d_cos = stream.clone_htod(&cos).expect("cos upload");
        let d_sin = stream.clone_htod(&sin).expect("sin upload");
        for batch in [1usize, 2, 4, 8, 16] {
            let amplitude = [0.5, 2.0, 6.0][seed as usize];
            let qkv: Vec<u16> = (0..batch * s * heads * d * 3)
                .map(|_| f32_to_f16_bits(random() * amplitude))
                .collect();
            let d_qkv = stream.clone_htod(&qkv).expect("QKV upload");
            let mut results = Vec::new();
            for (name, rows, threads) in [
                ("attention_fa2_kernel", 128usize, 256u32),
                ("attention_fa2_q64_kernel", 64, 128),
                ("attention_fa2_q64_serial_kernel", 64, 128),
            ] {
                let f = rt.get_func(name).expect("attention entry");
                let mut output = unsafe { stream.alloc::<u16>(batch * s * heads * d) }
                    .expect("attention output");
                unsafe {
                    stream
                        .launch_builder(&f)
                        .arg(&d_qkv)
                        .arg(&d_cos)
                        .arg(&d_sin)
                        .arg(&mut output)
                        .arg(&(s as i32))
                        .arg(&(d as i32))
                        .arg(&scale)
                        .arg(&(heads as i32))
                        .launch(LaunchConfig {
                            grid_dim: (s.div_ceil(rows) as u32, (batch * heads) as u32, 1),
                            block_dim: (threads, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .expect("attention launch");
                }
                stream.synchronize().expect("attention completion");
                results.push(stream.clone_dtoh(&output).expect("attention download"));
            }
            for (index, name) in [(1usize, "q64"), (2, "q64-serial")] {
                let mut different = 0usize;
                let mut max_abs = 0.0f32;
                for (&a, &b) in results[0].iter().zip(&results[index]) {
                    let af = f16_to_f32_bits(a);
                    let bf = f16_to_f32_bits(b);
                    assert!(af.is_finite() && bf.is_finite());
                    different += usize::from(a != b);
                    max_abs = max_abs.max((af - bf).abs());
                }
                println!(
                    "seed={seed} amplitude={amplitude} batch={batch} candidate={name} halfBitMismatches={different}/{} maxAbs={max_abs:.9}",
                    results[0].len()
                );
                if index == 2 {
                    assert_eq!(
                        different, 0,
                        "q64-serial differs from q128 at seed={seed}, batch={batch}, maxAbs={max_abs}"
                    );
                }
            }
        }
    }
}
