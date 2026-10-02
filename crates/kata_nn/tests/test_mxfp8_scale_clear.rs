//! Isolated comparison of separate and fused MXFP8 activation scale clearing.
//!
//! CPU mapping/codec gates run without CUDA. The GPU parent requires
//! KATAGO_TEST_MXFP8=1 and launches two child processes with frozen flag 0/1.
//! Optional KATAGO_MXFP8_SCALE_CLEAR_EVIDENCE_DIR names a NEW output directory.
//! Full allocation bytes, numerical output bits, and status evidence are retained.
//! This is an operator regression gate, not an accuracy/performance certificate.
#![allow(dead_code)] // CPU-only builds retain the independent GPU oracle.

const ROWS: [usize; 6] = [1, 17, 127, 128, 129, 361];
const KS: [usize; 9] = [31, 32, 33, 72, 168, 344, 384, 512, 1152];
const MAX_K: usize = 1152;

fn packed_offset(row: usize, block: usize, blocks4: usize) -> usize {
    ((row / 128) * (blocks4 / 4) + block / 4) * 512
        + (row % 32) * 16
        + ((row % 128) / 32) * 4
        + block % 4
}

#[test]
fn scale_inverse_partitions_entire_shared_capacity() {
    use std::collections::HashSet;
    // The independent oracle is the set of forward-mapped logical coordinates.
    // Include both an exact allocation and the much larger reused allocation.
    for rows in ROWS {
        for k in KS {
            let blocks = k.div_ceil(32);
            let blocks4 = blocks.div_ceil(4) * 4;
            for max_k in [k, MAX_K] {
                let capacity = rows.div_ceil(128) * 128 * max_k.div_ceil(128) * 4;
                let mut active = HashSet::new();
                for row in 0..rows {
                    for block in 0..blocks {
                        let offset = packed_offset(row, block, blocks4);
                        assert!(offset < capacity);
                        assert!(active.insert(offset), "logical writers must be unique");
                    }
                }
                let mut inverse_coordinates = HashSet::new();
                let mut active_count = 0;
                for offset in 0..capacity {
                    let tile = offset / 512;
                    let local = offset % 512;
                    let quads = blocks4 / 4;
                    // Mirrors only the proposed addressing predicate, not the oracle set.
                    let row = (tile / quads) * 128 + ((local >> 2) & 3) * 32 + (local >> 4);
                    let block = (tile % quads) * 4 + (local & 3);
                    assert!(inverse_coordinates.insert((row, block)));
                    assert_eq!(packed_offset(row, block, blocks4), offset);
                    let is_active = row < rows && block < blocks;
                    assert_eq!(
                        is_active,
                        active.contains(&offset),
                        "rows={rows} K={k} maxK={max_k} offset={offset}"
                    );
                    active_count += usize::from(is_active);
                }
                assert_eq!(active_count, rows * blocks);
                assert_eq!(inverse_coordinates.len(), capacity);
                // A clear write belongs exclusively to the inactive complement.
                assert_eq!(capacity - active_count + active.len(), capacity);
            }
        }
    }
    // Cover the device predicate's u32 indexing limit without allocating a giant set.
    for offset in [u32::MAX as usize - 1, u32::MAX as usize - 511] {
        for quads in [1usize, 64] {
            let tile = offset / 512;
            let local = offset % 512;
            let row = (tile / quads) * 128 + ((local >> 2) & 3) * 32 + (local >> 4);
            let block = (tile % quads) * 4 + (local & 3);
            assert!(row <= 1usize << 30);
            assert!(block < quads * 4);
            assert_eq!(packed_offset(row, block, quads * 4), offset);
        }
    }
}

fn half_decode(bits: u16) -> f32 {
    let exponent = (bits >> 10) & 31;
    let mantissa = bits & 1023;
    let magnitude = match exponent {
        0 => (mantissa as f64 * 2f64.powi(-24)) as f32,
        31 if mantissa == 0 => f32::INFINITY,
        31 => f32::NAN,
        e => ((1024 + mantissa) as f64 * 2f64.powi(e as i32 - 25)) as f32,
    };
    if bits & 0x8000 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

// Independent IEEE binary16 RNE, including gradual underflow.
fn half_encode(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 255) as i32;
    let mut mantissa = bits & 0x7fffff;
    if exponent == 255 {
        return sign | 0x7c00 | if mantissa == 0 { 0 } else { 0x200 };
    }
    let mut e = exponent - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        mantissa |= 0x800000;
        let shift = (14 - e) as u32;
        let mut rounded = mantissa >> shift;
        let remainder = mantissa & ((1 << shift) - 1);
        let midpoint = 1 << (shift - 1);
        if remainder > midpoint || (remainder == midpoint && rounded & 1 != 0) {
            rounded += 1;
        }
        return sign | rounded as u16;
    }
    let mut rounded = mantissa >> 13;
    let remainder = mantissa & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && rounded & 1 != 0) {
        rounded += 1;
        if rounded == 0x400 {
            rounded = 0;
            e += 1;
        }
    }
    sign | ((e as u16) << 10) | rounded as u16
}

fn fp8_decode(bits: u8) -> f64 {
    let e = (bits >> 3) & 15;
    let m = bits & 7;
    assert!(e != 15 || m != 7);
    let magnitude = if e == 0 {
        m as f64 * 2f64.powi(-9)
    } else {
        (8 + m) as f64 * 2f64.powi(e as i32 - 10)
    };
    if bits & 128 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

fn fp8_encode(value: f32) -> u8 {
    assert!(value.is_finite());
    // Search the independently decoded finite codebook, not CUDA conversion bits.
    static CODEBOOK: std::sync::OnceLock<[f64; 127]> = std::sync::OnceLock::new();
    let codes = CODEBOOK.get_or_init(|| std::array::from_fn(|i| fp8_decode(i as u8)));
    let magnitude = value.abs() as f64;
    let upper = codes.partition_point(|&v| v < magnitude);
    let code = if upper == 0 {
        0
    } else if upper == codes.len() {
        126
    } else {
        let lo = magnitude - codes[upper - 1];
        let hi = codes[upper] - magnitude;
        if lo < hi || (lo == hi && (upper - 1) % 2 == 0) {
            upper - 1
        } else {
            upper
        }
    };
    code as u8 | if value.is_sign_negative() { 128 } else { 0 }
}

fn scale_encode(ratio: f32) -> u8 {
    assert!(ratio.is_finite() && ratio >= 0.0);
    for exponent in -127..=127 {
        if 2f64.powi(exponent) >= ratio as f64 {
            return (exponent + 127) as u8;
        }
    }
    254
}

struct Packed {
    values: Vec<u8>,
    scales: Vec<u8>,
    logical_scales: Vec<u8>,
    rows: usize,
    k: usize,
    kp: usize,
}

fn quantize(source: &[f32], rows: usize, k: usize, stride: usize) -> Packed {
    let blocks = k.div_ceil(32);
    let blocks4 = blocks.div_ceil(4) * 4;
    let kp = blocks * 32;
    let mut out = Packed {
        values: vec![0; rows * kp],
        scales: vec![0; rows.div_ceil(128) * 128 * blocks4],
        logical_scales: vec![0; rows * blocks],
        rows,
        k,
        kp,
    };
    for row in 0..rows {
        for block in 0..blocks {
            let range = block * 32..((block + 1) * 32).min(k);
            let mut maximum = 0.0f32;
            for col in range.clone() {
                let value = source[row * stride + col];
                assert!(value.is_finite());
                maximum = maximum.max(value.abs());
            }
            let scale = if maximum == 0.0 {
                127
            } else {
                scale_encode(maximum / 448.0)
            };
            out.logical_scales[row * blocks + block] = scale;
            out.scales[packed_offset(row, block, blocks4)] = scale;
            for col in range {
                let scaled =
                    (source[row * stride + col] as f64 * 2f64.powi(127 - scale as i32)) as f32;
                out.values[row * kp + col] = if maximum == 0.0 {
                    0
                } else {
                    fp8_encode(scaled)
                };
            }
        }
    }
    out
}

impl Packed {
    fn value(&self, row: usize, col: usize) -> f64 {
        fp8_decode(self.values[row * self.kp + col])
            * 2f64.powi(self.logical_scales[row * self.k.div_ceil(32) + col / 32] as i32 - 127)
    }
}

#[test]
fn independent_codec_covers_rne_and_signed_zero() {
    for code in 0u8..=126 {
        let decoded = fp8_decode(code) as f32;
        assert_eq!(fp8_encode(decoded), code);
        assert_eq!(fp8_encode(-decoded), code | 128);
        if code < 126 {
            let midpoint = ((fp8_decode(code) + fp8_decode(code + 1)) * 0.5) as f32;
            assert_eq!(
                fp8_encode(midpoint),
                if code % 2 == 0 { code } else { code + 1 }
            );
            assert_eq!(fp8_encode(f32::from_bits(midpoint.to_bits() - 1)), code);
            assert_eq!(fp8_encode(f32::from_bits(midpoint.to_bits() + 1)), code + 1);
        }
    }
    for bits in 0u16..=u16::MAX {
        if bits & 0x7c00 != 0x7c00 {
            assert_eq!(half_encode(half_decode(bits)), bits);
        }
    }
    assert_eq!(scale_encode(1.0), 127);
    assert_eq!(scale_encode(f32::from_bits(1.0f32.to_bits() + 1)), 128);
    assert_eq!(scale_encode(0.0), 0);
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr};
    use kata_nn::backends::{
        cuda::CudaRuntime,
        mxfp8::{
            validate_completed_status, Mxfp8Kernels, Mxfp8Output, Mxfp8Weight, Mxfp8Workspace,
            PreparedProjectionId, STATUS_WORDS,
        },
    };
    use std::{
        fs,
        io::{BufWriter, Write},
        path::{Path, PathBuf},
        process::Command,
        sync::Arc,
        time::{SystemTime, UNIX_EPOCH},
    };

    const TEST: &str = "gpu::separate_and_fused_are_bitwise_identical";
    const CHILD: &str = "KATAGO_MXFP8_SCALE_CLEAR_CHILD";
    const FLAG: &str = "KATAGO_CUDA_MXFP8_SCALE_CLEAR_FUSION";
    const EVIDENCE: &str = "KATAGO_MXFP8_SCALE_CLEAR_EVIDENCE_DIR";
    const GUARD: usize = 128;
    const HALF_GUARD: u16 = 0x3555;
    const FLOAT_GUARD: u32 = 0x4a123456;

    fn upload<T: DeviceRepr>(stream: &Arc<CudaStream>, source: &[T]) -> CudaSlice<T> {
        let out = stream.clone_htod(source).unwrap();
        stream.synchronize().unwrap();
        out
    }
    fn download<T: DeviceRepr + Default + Clone>(
        stream: &Arc<CudaStream>,
        source: &CudaSlice<T>,
    ) -> Vec<T> {
        let mut out = vec![T::default(); source.len()];
        stream.memcpy_dtoh(source, &mut out).unwrap();
        stream.synchronize().unwrap();
        out
    }
    fn status(stream: &Arc<CudaStream>, ws: &Mxfp8Workspace) -> Vec<u32> {
        let out = download(stream, ws.status());
        assert_eq!(out.len(), STATUS_WORDS);
        out
    }
    fn valid(stream: &Arc<CudaStream>, ws: &Mxfp8Workspace) {
        validate_completed_status(&status(stream, ws)).unwrap();
    }

    struct Evidence(BufWriter<fs::File>);
    impl Evidence {
        fn new(path: &Path) -> Self {
            Self(BufWriter::new(
                fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)
                    .unwrap(),
            ))
        }
        fn bytes(&mut self, label: &str, bytes: &[u8]) {
            self.0
                .write_all(&(label.len() as u64).to_le_bytes())
                .unwrap();
            self.0.write_all(label.as_bytes()).unwrap();
            self.0
                .write_all(&(bytes.len() as u64).to_le_bytes())
                .unwrap();
            self.0.write_all(bytes).unwrap();
        }
        fn words(&mut self, label: &str, values: &[u32]) {
            self.bytes(
                label,
                &values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
        }
        fn finish(mut self) {
            self.0.flush().unwrap();
            self.0.get_ref().sync_all().unwrap();
        }
    }

    enum Output {
        Half(CudaSlice<u16>),
        Float(CudaSlice<f32>),
    }
    impl Output {
        fn new(stream: &Arc<CudaStream>, count: usize, mode: Mxfp8Output) -> Self {
            if matches!(mode, Mxfp8Output::Half) {
                let mut values = vec![HALF_GUARD; count + GUARD];
                values[..count].fill(0);
                Self::Half(upload(stream, &values))
            } else {
                let mut values = vec![f32::from_bits(FLOAT_GUARD); count + GUARD];
                values[..count].fill(0.0);
                Self::Float(upload(stream, &values))
            }
        }
        fn project(
            &mut self,
            ws: &mut Mxfp8Workspace,
            id: PreparedProjectionId,
            layer: u32,
            input: &CudaSlice<u16>,
            stride: usize,
        ) {
            match self {
                Self::Half(out) => ws.project_half(id, layer, input, stride, out).unwrap(),
                Self::Float(out) => ws.project_f32(id, layer, input, stride, out).unwrap(),
            }
        }
        fn restore(&mut self, stream: &Arc<CudaStream>, residual: &CudaSlice<f32>) {
            match self {
                Self::Float(out) => stream.memcpy_dtod(residual, out).unwrap(),
                Self::Half(_) => panic!("only FloatResidual consumes C"),
            }
        }
        fn bits(&self, stream: &Arc<CudaStream>) -> Vec<u32> {
            match self {
                Self::Half(out) => {
                    let values = download(stream, out);
                    let count = values.len() - GUARD;
                    assert!(values[count..].iter().all(|&v| v == HALF_GUARD));
                    values[..count].iter().map(|&v| u32::from(v)).collect()
                }
                Self::Float(out) => {
                    let values = download(stream, out);
                    let count = values.len() - GUARD;
                    assert!(values[count..].iter().all(|&v| v.to_bits() == FLOAT_GUARD));
                    values[..count].iter().map(|v| v.to_bits()).collect()
                }
            }
        }
    }

    fn half_source(rows: usize, k: usize, stride: usize, seed: usize, edges: bool) -> Vec<u16> {
        let mut result = vec![0x7e00; rows * stride]; // NaN stride padding must not be read.
        for row in 0..rows {
            for col in 0..k {
                let code = (row * 37 + col * 17 + seed * 13) % 9;
                let exponent = ((row + col / 32 + seed) % 4) as i32 - 4;
                let value = (code as f32 - 4.0) * 2f32.powi(exponent);
                result[row * stride + col] = if edges && row == 0 && col < 32 {
                    if col % 2 == 0 {
                        0
                    } else {
                        0x8000
                    }
                } else if edges && row == 1 {
                    [0x0001, 0x8001, 0x03ff, 0x7bff, 0xfbff, 0x0000][col % 6]
                } else {
                    half_encode(value)
                };
            }
        }
        result
    }

    fn check_allocation(
        stream: &Arc<CudaStream>,
        ws: &Mxfp8Workspace,
        oracle: &Packed,
        expected_data: &mut [u8],
        evidence: &mut Evidence,
        label: &str,
    ) {
        // Inactive FP8 data is retained, unlike scales, so model its full history.
        expected_data[..oracle.values.len()].copy_from_slice(&oracle.values);
        let actual_data = download(stream, ws.quantized_data());
        assert_eq!(
            actual_data.as_slice(),
            &*expected_data,
            "{label}: whole FP8 allocation"
        );
        let actual_scales = download(stream, ws.packed_scales());
        let mut expected_scales = vec![0; actual_scales.len()];
        expected_scales[..oracle.scales.len()].copy_from_slice(&oracle.scales);
        assert_eq!(
            actual_scales, expected_scales,
            "{label}: whole scale allocation and tail"
        );
        evidence.bytes(&format!("{label}/fp8"), &actual_data);
        evidence.bytes(&format!("{label}/scales"), &actual_scales);
    }

    fn shape_switches(
        rt: &Arc<CudaRuntime>,
        stream: &Arc<CudaStream>,
        kernels: &Mxfp8Kernels,
        fused: bool,
        evidence: &mut Evidence,
    ) {
        for rows in ROWS {
            let mut ws =
                Mxfp8Workspace::new(rt.clone(), stream.clone(), kernels.clone(), rows, MAX_K)
                    .unwrap();
            assert_eq!(ws.scale_clear_fusion_enabled(), fused);
            let mut expected_data = vec![0; rows * MAX_K];
            let mut ids = Vec::new();
            for k in KS {
                let weight =
                    Mxfp8Weight::upload_from_f32(rt, stream, kernels, &vec![0.0; 8 * k], 8, k, k)
                        .unwrap();
                ids.push(ws.prepare_projection(weight, Mxfp8Output::Float).unwrap());
            }
            let mut sequence = vec![KS.len() - 1];
            for index in 0..KS.len() - 1 {
                sequence.extend([index, KS.len() - 1]);
            }
            for (step, index) in sequence.into_iter().enumerate() {
                let k = KS[index];
                let stride = k + 7;
                let source = half_source(rows, k, stride, step + 3, true);
                let host: Vec<_> = source.iter().copied().map(half_decode).collect();
                let oracle = quantize(&host, rows, k, stride);
                let device = upload(stream, &source);
                let mut output = Output::new(stream, rows * 8, Mxfp8Output::Float);
                ws.begin_forward().unwrap();
                output.project(&mut ws, ids[index], 5, &device, stride);
                valid(stream, &ws);
                let label = format!("shape/rows={rows}/step={step}/k={k}");
                check_allocation(stream, &ws, &oracle, &mut expected_data, evidence, &label);
                let bits = output.bits(stream);
                assert!(bits.iter().all(|&v| f32::from_bits(v) == 0.0));
                evidence.words(&format!("{label}/output"), &bits);
                evidence.words(&format!("{label}/status"), &status(stream, &ws));
                assert_eq!(download(stream, &device), source);
            }
        }
    }

    fn reference(x: &Packed, w: &Packed, residual: &[f32], half: bool) -> Vec<u32> {
        assert_eq!(x.k, w.k);
        let mut out = Vec::with_capacity(x.rows * w.rows);
        for row in 0..x.rows {
            for n in 0..w.rows {
                let mut sum = residual[row * w.rows + n] as f64;
                for k in 0..x.k {
                    sum += x.value(row, k) * w.value(n, k);
                }
                // Chosen dyadic inputs/products fit exactly in FP32 throughout reduction.
                assert_eq!((sum as f32) as f64, sum);
                out.push(if half {
                    u32::from(half_encode(sum as f32))
                } else {
                    (sum as f32).to_bits()
                });
            }
        }
        out
    }

    fn outputs_and_graph(
        rt: &Arc<CudaRuntime>,
        stream: &Arc<CudaStream>,
        kernels: &Mxfp8Kernels,
        evidence: &mut Evidence,
    ) {
        for mode in [
            Mxfp8Output::Half,
            Mxfp8Output::Float,
            Mxfp8Output::FloatResidual,
        ] {
            let (rows, n, k, stride) = (17, 8, 72, 79);
            let weight_source: Vec<_> = half_source(n, k, k + 5, 11, false)
                .into_iter()
                .map(half_decode)
                .collect();
            let w = quantize(&weight_source, n, k, k + 5);
            let weight =
                Mxfp8Weight::upload_from_f32(rt, stream, kernels, &weight_source, n, k, k + 5)
                    .unwrap();
            assert_eq!(download(stream, weight.quantized_data()), w.values);
            assert_eq!(download(stream, weight.packed_scales()), w.scales);
            let sources = [
                half_source(rows, k, stride, 5, false),
                half_source(rows, k, stride, 6, false),
            ];
            let packed: Vec<_> = sources
                .iter()
                .map(|s| {
                    quantize(
                        &s.iter().copied().map(half_decode).collect::<Vec<_>>(),
                        rows,
                        k,
                        stride,
                    )
                })
                .collect();
            assert_ne!(packed[0].values, packed[1].values);
            assert_ne!(packed[0].scales, packed[1].scales);
            let d_sources: Vec<_> = sources.iter().map(|s| upload(stream, s)).collect();
            let residual: Vec<_> = (0..rows * n)
                .map(|i| {
                    if matches!(mode, Mxfp8Output::FloatResidual) {
                        (i as i32 % 11 - 5) as f32 * 0.125
                    } else {
                        0.0
                    }
                })
                .collect();
            let mut guarded_residual = residual.clone();
            guarded_residual.extend(vec![f32::from_bits(FLOAT_GUARD); GUARD]);
            let d_residual = upload(stream, &guarded_residual);
            let expected: Vec<_> = packed
                .iter()
                .map(|x| reference(x, &w, &residual, matches!(mode, Mxfp8Output::Half)))
                .collect();
            let mut ws =
                Mxfp8Workspace::new(rt.clone(), stream.clone(), kernels.clone(), rows, MAX_K)
                    .unwrap();
            let id = ws.prepare_projection(weight.clone(), mode).unwrap();
            let resources = ws.graph_resources().unwrap();
            let mut input = upload(stream, &sources[0]);
            let mut output = Output::new(stream, rows * n, mode);
            let mut history = vec![0; rows * MAX_K];
            ws.begin_forward().unwrap();
            if matches!(mode, Mxfp8Output::FloatResidual) {
                output.restore(stream, &d_residual);
            }
            output.project(&mut ws, id, 9, &input, stride);
            ws.complete_warmup().unwrap();
            assert_eq!(
                output.bits(stream),
                expected[0],
                "direct decoded CPU oracle {mode:?}"
            );
            stream
                .begin_capture(
                    cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                )
                .unwrap();
            ws.begin_forward().unwrap();
            if matches!(mode, Mxfp8Output::FloatResidual) {
                output.restore(stream, &d_residual);
            }
            output.project(&mut ws, id, 9, &input, stride);
            let graph = stream.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY).unwrap().expect("nonempty graph");
            graph.upload().unwrap();
            let mut first = None;
            for (step, variant) in [0, 1, 0].into_iter().enumerate() {
                // Fresh workspace, plan, activation buffer and output for every independent direct reference.
                let mut direct_ws =
                    Mxfp8Workspace::new(rt.clone(), stream.clone(), kernels.clone(), rows, MAX_K)
                        .unwrap();
                let direct_id = direct_ws.prepare_projection(weight.clone(), mode).unwrap();
                let direct_input = upload(stream, &sources[variant]);
                let mut direct_output = Output::new(stream, rows * n, mode);
                direct_ws.begin_forward().unwrap();
                if matches!(mode, Mxfp8Output::FloatResidual) {
                    direct_output.restore(stream, &d_residual);
                }
                direct_output.project(&mut direct_ws, direct_id, 9, &direct_input, stride);
                valid(stream, &direct_ws);
                let direct_bits = direct_output.bits(stream);
                assert_eq!(
                    direct_bits, expected[variant],
                    "fresh direct CPU oracle {mode:?}"
                );
                stream.memcpy_dtod(&d_sources[variant], &mut input).unwrap();
                graph.launch().unwrap();
                valid(stream, &ws);
                let bits = output.bits(stream);
                assert_eq!(bits, direct_bits, "{mode:?} Graph/fresh direct");
                if variant == 0 {
                    if let Some(first) = &first {
                        assert_eq!(&bits, first, "A/B/A returns A exactly");
                    } else {
                        first = Some(bits.clone());
                    }
                }
                let label = format!("graph/{mode:?}/step={step}");
                check_allocation(
                    stream,
                    &ws,
                    &packed[variant],
                    &mut history,
                    evidence,
                    &label,
                );
                evidence.words(&format!("{label}/output"), &bits);
                evidence.words(&format!("{label}/status"), &status(stream, &ws));
            }
            let mut bad = sources[0].clone();
            bad[stride + 7] = 0x7e00;
            let d_bad = upload(stream, &bad);
            stream.memcpy_dtod(&d_bad, &mut input).unwrap();
            graph.launch().unwrap();
            let failure = status(stream, &ws);
            assert_ne!(failure[0] & 1, 0);
            assert_eq!(failure[1], 9);
            assert!(validate_completed_status(&failure).is_err());
            evidence.words(&format!("graph/{mode:?}/bad"), &failure);
            stream.memcpy_dtod(&d_sources[0], &mut input).unwrap();
            graph.launch().unwrap();
            valid(stream, &ws);
            assert_eq!(
                output.bits(stream),
                first.unwrap(),
                "Graph recovers after invalid input"
            );
            evidence.words(&format!("graph/{mode:?}/recovered"), &output.bits(stream));
            assert_eq!(download(stream, weight.quantized_data()), w.values);
            assert_eq!(download(stream, weight.packed_scales()), w.scales);
            drop(graph);
            drop(resources);
        }
    }

    fn sticky_and_recovery(
        rt: &Arc<CudaRuntime>,
        stream: &Arc<CudaStream>,
        kernels: &Mxfp8Kernels,
        evidence: &mut Evidence,
    ) {
        let (rows, n, k, stride) = (17, 8, 72, 79);
        let weight =
            Mxfp8Weight::upload_from_f32(rt, stream, kernels, &vec![0.25; n * k], n, k, k).unwrap();
        let mut ws =
            Mxfp8Workspace::new(rt.clone(), stream.clone(), kernels.clone(), rows, MAX_K).unwrap();
        let id = ws.prepare_projection(weight, Mxfp8Output::Float).unwrap();
        let good = half_source(rows, k, stride, 7, false);
        let device = upload(stream, &good);
        let mut output = Output::new(stream, rows * n, Mxfp8Output::Float);
        ws.begin_forward().unwrap();
        output.project(&mut ws, id, 2, &device, stride);
        valid(stream, &ws);
        let reference = output.bits(stream);
        for bits in [0x7e00, 0x7c00, 0xfc00] {
            let mut source = good.clone();
            source[stride + 5] = bits;
            let bad = upload(stream, &source);
            ws.begin_forward().unwrap();
            output.project(&mut ws, id, 2, &device, stride);
            output.project(&mut ws, id, 17, &bad, stride);
            output.project(&mut ws, id, 23, &device, stride);
            let failure = status(stream, &ws);
            assert_ne!(failure[0] & if bits == 0x7e00 { 1 } else { 2 }, 0);
            assert_eq!(failure[1], 17, "non-first bad layer stays sticky");
            assert!(validate_completed_status(&failure).is_err());
            evidence.words(&format!("sticky/{bits:x}/bad"), &failure);
            ws.begin_forward().unwrap();
            output.project(&mut ws, id, 2, &device, stride);
            valid(stream, &ws);
            assert_eq!(output.bits(stream), reference);
            evidence.words(
                &format!("sticky/{bits:x}/recovered-status"),
                &status(stream, &ws),
            );
            evidence.words(
                &format!("sticky/{bits:x}/recovered-output"),
                &output.bits(stream),
            );
        }
    }

    fn child(fused: bool, directory: &Path) {
        let rt = Arc::new(CudaRuntime::new().expect("explicit MXFP8 opt-in requires CUDA"));
        let stream = rt.device.new_stream().unwrap();
        let kernels =
            Mxfp8Kernels::load(&rt).expect("explicit MXFP8 opt-in requires supported SM120/Lt");
        assert_eq!(kernels.scale_clear_fusion_enabled(), fused);
        let mut evidence =
            Evidence::new(&directory.join(format!("flag-{}.bin", usize::from(fused))));
        evidence.bytes("format", b"mxfp8-scale-clear-raw-evidence-v1");
        shape_switches(&rt, &stream, &kernels, fused, &mut evidence);
        outputs_and_graph(&rt, &stream, &kernels, &mut evidence);
        sticky_and_recovery(&rt, &stream, &kernels, &mut evidence);
        stream.synchronize().unwrap();
        evidence.finish();
        eprintln!("[scale-clear-test] completed flag={} shape_forwards=102 output_modes=3 graph_aba_modes=3", usize::from(fused));
    }

    #[test]
    fn separate_and_fused_are_bitwise_identical() {
        if std::env::var("KATAGO_TEST_MXFP8").ok().as_deref() != Some("1") {
            eprintln!("SKIP MXFP8 scale-clear GPU gate: set KATAGO_TEST_MXFP8=1");
            return;
        }
        if let Ok(mode) = std::env::var(CHILD) {
            assert!(mode == "0" || mode == "1");
            assert_eq!(std::env::var(FLAG).unwrap(), mode);
            child(
                mode == "1",
                &PathBuf::from(std::env::var_os(EVIDENCE).unwrap()),
            );
            return;
        }
        let directory = std::env::var_os(EVIDENCE)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let nanos = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos();
                std::env::temp_dir().join(format!(
                    "rustgo-mxfp8-scale-clear-{}-{nanos}",
                    std::process::id()
                ))
            });
        fs::create_dir(&directory).expect("evidence output must be a NEW directory");
        for mode in ["0", "1"] {
            let result = Command::new(std::env::current_exe().unwrap())
                .args(["--exact", TEST, "--nocapture", "--test-threads=1"])
                .env(CHILD, mode)
                .env(FLAG, mode)
                .env(EVIDENCE, &directory)
                .output()
                .expect("launch isolated scale-clear child");
            fs::write(
                directory.join(format!("flag-{mode}.stdout.txt")),
                &result.stdout,
            )
            .unwrap();
            fs::write(
                directory.join(format!("flag-{mode}.stderr.txt")),
                &result.stderr,
            )
            .unwrap();
            assert!(
                result.status.success(),
                "flag={mode} child failed; inspect {}\n{}\n{}",
                directory.display(),
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
            let text = format!(
                "{}\n{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
            let route = if mode == "1" { "fused" } else { "separate" };
            let marker = format!("[cuda-tactic] name=mxfp8_scale_clear_fusion requested={mode} launch={route} effective={mode}");
            assert!(
                text.contains(&marker),
                "actual route marker missing: {marker}"
            );
            assert!(text.contains(&format!("[scale-clear-test] completed flag={mode}")));
        }
        let separate = fs::read(directory.join("flag-0.bin")).unwrap();
        let fused = fs::read(directory.join("flag-1.bin")).unwrap();
        assert!(!separate.is_empty());
        assert_eq!(
            separate.len(),
            fused.len(),
            "framed evidence lengths differ"
        );
        if let Some(offset) = separate.iter().zip(&fused).position(|(a, b)| a != b) {
            panic!(
                "flag 0/1 raw evidence differs at byte {offset}: {} vs {}; inspect {}",
                separate[offset],
                fused[offset],
                directory.display()
            );
        }
        eprintln!(
            "[scale-clear-test] bitwise PASS {} bytes per route; evidence={}",
            separate.len(),
            directory.display()
        );
    }
}
