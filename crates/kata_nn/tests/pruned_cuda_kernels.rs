//! Numeric and tail-padding tests for native pruned model kernels.
#![cfg(feature = "cuda")]
use cudarc::driver::{LaunchConfig, PushKernelArg};
use kata_nn::backends::cuda::{CudaRuntime, f16_to_f32_bits, f32_to_f16_bits};

#[test]
fn padded_swiglu_and_variable_width_rms_match_fp32_reference() {
    let rt = CudaRuntime::new().expect("CUDA required for this kernel gate");
    let stream = rt.device.default_stream();
    let rows = 7usize; // partial final warp4 block
    for hidden in [8usize, 24, 48, 72, 392, 1160] {
        let padded = hidden.div_ceil(16) * 16;
        let host: Vec<u16> = (0..rows * 2 * hidden)
            .map(|i| f32_to_f16_bits((i as i32 % 71 - 35) as f32 / 13.0))
            .collect();
        let x = stream.clone_htod(&host).unwrap();
        // Dirty output padding must be overwritten on every launch.
        let mut y = stream.clone_htod(&vec![0x7e00u16; rows * padded]).unwrap();
        let f = rt.get_func("swiglu_dual_padded_kernel").unwrap();
        unsafe {
            stream
                .launch_builder(&f)
                .arg(&x)
                .arg(&mut y)
                .arg(&(rows as i32))
                .arg(&(hidden as i32))
                .arg(&(padded as i32))
                .launch(LaunchConfig::for_num_elems((rows * padded) as u32))
                .unwrap();
        }
        let actual = stream.clone_dtoh(&y).unwrap();
        for row in 0..rows {
            for col in 0..padded {
                let got = actual[row * padded + col];
                if col >= hidden {
                    assert_eq!(got, 0, "padding {hidden}/{row}/{col}");
                    continue;
                }
                let g = f16_to_f32_bits(host[row * 2 * hidden + col]);
                let u = f16_to_f32_bits(host[row * 2 * hidden + hidden + col]);
                let expected = u * (g / (1.0 + (-g).exp()));
                assert!((f16_to_f32_bits(got) - expected).abs() < 0.001 * expected.abs().max(1.0));
            }
        }
    }
    for width in [32usize, 384, 512, 1024] {
        let host: Vec<f32> = (0..rows * width)
            .map(|i| {
                if i < width {
                    0.0
                } else {
                    (i as i32 % 97 - 48) as f32 / 17.0
                }
            })
            .collect();
        let scales: Vec<f32> = (0..width).map(|i| 0.5 + (i % 19) as f32 / 19.0).collect();
        let x = stream.clone_htod(&host).unwrap();
        let scale = stream.clone_htod(&scales).unwrap();
        let mut y = stream.alloc_zeros::<u16>(rows * width).unwrap();
        let f = rt.get_func("rms_norm_f32_w4_generic_kernel").unwrap();
        unsafe {
            stream
                .launch_builder(&f)
                .arg(&x)
                .arg(&scale)
                .arg(&mut y)
                .arg(&1e-6f32)
                .arg(&(width as i32))
                .arg(&(rows as i32))
                .launch(LaunchConfig {
                    grid_dim: (rows.div_ceil(4) as u32, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        let actual = stream.clone_dtoh(&y).unwrap();
        for row in 0..rows {
            let sq: f64 = host[row * width..(row + 1) * width]
                .iter()
                .map(|&x| (x as f64).powi(2))
                .sum();
            let rms = (sq / width as f64 + 1e-6).sqrt();
            for col in 0..width {
                let expected = (host[row * width + col] as f64 / rms * scales[col] as f64) as f32;
                let got = f16_to_f32_bits(actual[row * width + col]);
                assert!(
                    (got - expected).abs() < 0.001 * expected.abs().max(1.0),
                    "RMS {width}/{row}/{col}: {got} vs {expected}"
                );
            }
        }
    }
    variable_head_attention(&rt);
}

fn variable_head_attention(rt: &CudaRuntime) {
    let stream = rt.device.default_stream();
    // Cross query/key tile boundaries and batch boundaries with nonuniform RoPE.
    let (batch, seq, heads, dim) = (2usize, 137usize, 16usize, 32usize);
    let mid = heads * dim;
    let packed: Vec<_> = (0..batch * seq * 3 * mid)
        .map(|i| f32_to_f16_bits(((i * 17 + i / 7) % 101) as f32 / 100.0 - 0.5))
        .collect();
    let angles: Vec<_> = (0..seq * heads * 16)
        .map(|i| (i % 257) as f32 * 0.021)
        .collect();
    let cos: Vec<_> = angles.iter().map(|x| x.cos()).collect();
    let sin: Vec<_> = angles.iter().map(|x| x.sin()).collect();
    let mut q = vec![0.0f32; batch * seq * mid];
    let mut k = q.clone();
    for b in 0..batch {
        for s in 0..seq {
            for h in 0..heads {
                for d in 0..dim {
                    let r = s * mid / 2 + h * 16 + d / 2;
                    for (segment, dst) in [(0, &mut q), (1, &mut k)] {
                        let pos = (b * seq + s) * 3 * mid + segment * mid + h * dim + (d / 2) * 2;
                        let a = f16_to_f32_bits(packed[pos]);
                        let z = f16_to_f32_bits(packed[pos + 1]);
                        // GPU compiler contracts the final multiply-add of the rotation.
                        let rotated = if d % 2 == 0 {
                            a.mul_add(cos[r], -z * sin[r])
                        } else {
                            a.mul_add(sin[r], z * cos[r])
                        };
                        dst[(b * seq + s) * mid + h * dim + d] =
                            f16_to_f32_bits(f32_to_f16_bits(rotated));
                    }
                }
            }
        }
    }
    let scale = 1.0f32 / (dim as f32).sqrt();
    let mut expected = vec![0.0f32; batch * seq * mid];
    for b in 0..batch {
        for h in 0..heads {
            for row in 0..seq {
                let mut scores = vec![0.0f32; seq];
                for col in 0..seq {
                    for d in 0..dim {
                        scores[col] += q[(b * seq + row) * mid + h * dim + d]
                            * k[(b * seq + col) * mid + h * dim + d];
                    }
                    scores[col] *= scale;
                }
                let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = scores.iter().map(|&x| (x - mx).exp()).sum();
                for d in 0..dim {
                    expected[(b * seq + row) * mid + h * dim + d] = scores
                        .iter()
                        .enumerate()
                        .map(|(col, &s)| {
                            (s - mx).exp() / sum
                                * f16_to_f32_bits(
                                    packed[(b * seq + col) * 3 * mid + 2 * mid + h * dim + d],
                                )
                        })
                        .sum();
                }
            }
        }
    }
    let x = stream.clone_htod(&packed).unwrap();
    let co = stream.clone_htod(&cos).unwrap();
    let sn = stream.clone_htod(&sin).unwrap();
    for (name, tile, threads) in [
        ("attention_fa2_generic_kernel", 128usize, 256),
        ("attention_fa2_q64_generic_kernel", 64, 128),
        ("attention_fa2_q64_serial_generic_kernel", 64, 128),
    ] {
        let mut y = stream.alloc_zeros::<u16>(expected.len()).unwrap();
        let f = rt.get_func(name).unwrap();
        unsafe {
            stream
                .launch_builder(&f)
                .arg(&x)
                .arg(&co)
                .arg(&sn)
                .arg(&mut y)
                .arg(&(seq as i32))
                .arg(&(dim as i32))
                .arg(&scale)
                .arg(&(heads as i32))
                .launch(LaunchConfig {
                    grid_dim: (seq.div_ceil(tile) as u32, (batch * heads) as u32, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        let actual = stream.clone_dtoh(&y).unwrap();
        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(&a, &b)| (f16_to_f32_bits(a) - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error < 0.001, "{name}: max abs error {max_error}");
        eprintln!("{name}: max abs error {max_error}");
    }
}
