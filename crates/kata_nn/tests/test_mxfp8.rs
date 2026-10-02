//! Independent MXFP8 operator gates. No model files or accuracy/performance claim.
//! CPU codec tests always run. GPU tests require both `--features cuda` and
//! `KATAGO_TEST_MXFP8=1`; explicitly requested GPU failures must never skip.
//! The arithmetic bounds and Graph bitwise gate match experiments/mxfp8_probe.md.

#![allow(dead_code)] // CPU-only builds retain the oracle used by GPU tests.

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

// Independent IEEE binary16 RNE; no production conversion helper is called.
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
    assert!(e != 15 || m != 7, "oracle cannot decode FP8 NaN");
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

// Enumerate the finite codebook, independently of the CUDA intrinsic/bit tricks.
fn fp8_encode(value: f32) -> Result<u8, &'static str> {
    if !value.is_finite() {
        return Err("nonfinite input");
    }
    let magnitude = value.abs() as f64;
    let mut best = 0u8;
    let mut distance = f64::INFINITY;
    for code in 0u8..=126 {
        let d = (fp8_decode(code) - magnitude).abs();
        if d < distance || (d == distance && code & 1 == 0) {
            best = code;
            distance = d;
        }
    }
    Ok(best | if value.is_sign_negative() { 128 } else { 0 })
}

fn scale_encode(ratio: f32) -> u8 {
    assert!(ratio.is_finite() && ratio >= 0.0);
    // ratio has already undergone the required FP32 amax/448 division.
    for e in -127..=127 {
        if 2f64.powi(e) >= ratio as f64 {
            return (e + 127) as u8;
        }
    }
    254
}

fn next_up(value: f32) -> f32 {
    if value == f32::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f32::from_bits(1);
    }
    f32::from_bits(if value > 0.0 {
        value.to_bits() + 1
    } else {
        value.to_bits() - 1
    })
}

fn next_down(value: f32) -> f32 {
    -next_up(-value)
}

struct Packed {
    rows: usize,
    k: usize,
    kp: usize,
    blocks: usize,
    values: Vec<u8>,
    logical_scales: Vec<u8>,
    scales: Vec<u8>,
}

fn packed_offset(row: usize, block: usize, blocks4: usize) -> usize {
    ((row / 128) * (blocks4 / 4) + block / 4) * 512
        + (row % 32) * 16
        + ((row % 128) / 32) * 4
        + block % 4
}

fn quantize(source: &[f32], rows: usize, k: usize, stride: usize) -> Result<Packed, &'static str> {
    assert!(rows > 0 && k > 0 && stride >= k && source.len() >= rows * stride);
    let blocks = k.div_ceil(32);
    let kp = blocks * 32;
    let blocks4 = blocks.div_ceil(4) * 4;
    let mut result = Packed {
        rows,
        k,
        kp,
        blocks,
        values: vec![0; rows * kp],
        logical_scales: vec![0; rows * blocks],
        scales: vec![0; rows.div_ceil(128) * 128 * blocks4],
    };
    for row in 0..rows {
        for block in 0..blocks {
            let range = block * 32..((block + 1) * 32).min(k);
            let mut maximum = 0.0f32;
            for col in range.clone() {
                let value = source[row * stride + col];
                if !value.is_finite() {
                    return Err("nonfinite input");
                }
                maximum = maximum.max(value.abs());
            }
            let scale = if maximum == 0.0 {
                127
            } else {
                scale_encode(maximum / 448.0)
            };
            result.logical_scales[row * blocks + block] = scale;
            result.scales[packed_offset(row, block, blocks4)] = scale;
            for col in range {
                let scaled =
                    (source[row * stride + col] as f64 * 2f64.powi(127 - scale as i32)) as f32;
                result.values[row * kp + col] = if maximum == 0.0 {
                    0
                } else {
                    fp8_encode(scaled)?
                };
            }
        }
    }
    Ok(result)
}

impl Packed {
    fn decoded(&self) -> Vec<f64> {
        (0..self.rows * self.k)
            .map(|i| {
                let row = i / self.k;
                let col = i % self.k;
                // Deliberately use logical scales, never the GPU's swizzled layout.
                fp8_decode(self.values[row * self.kp + col])
                    * 2f64.powi(self.logical_scales[row * self.blocks + col / 32] as i32 - 127)
            })
            .collect()
    }
}

fn mix(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846ca68b);
    value ^ (value >> 16)
}

fn input(rows: usize, k: usize, stride: usize, seed: u32, half: bool, exact: bool) -> Vec<f32> {
    let mut result = vec![f32::NAN; rows * stride];
    for row in 0..rows {
        for col in 0..k {
            let noise = mix(seed
                .wrapping_add((row as u32).wrapping_mul(65537))
                .wrapping_add((col as u32).wrapping_mul(257)));
            let exponent = if exact {
                (mix(seed ^ row as u32 ^ (col / 32) as u32) % 4) as i32 - 4
            } else {
                (mix(seed ^ (row as u32).wrapping_mul(131) ^ ((col / 32) as u32).wrapping_mul(977))
                    % 13) as i32
                    - 8
            };
            let fraction = if exact {
                (noise % 9) as f32 - 4.0
            } else {
                ((noise % 20001) as i32 - 10000) as f32 / 10000.0
            };
            let mut value = fraction * 2f32.powi(exponent);
            if row == 0 && col < 32 {
                value = if col & 1 == 0 { 0.0 } else { -0.0 };
            }
            result[row * stride + col] = if half {
                half_decode(half_encode(value))
            } else {
                value
            };
        }
    }
    result
}

#[test]
fn independent_codecs_cover_rounding_and_finite_boundaries() {
    for code in 0u16..0x7c00 {
        assert_eq!(half_encode(half_decode(code)), code);
        assert_eq!(half_encode(half_decode(code | 0x8000)), code | 0x8000);
    }
    assert_eq!(half_encode(1.00048828125), 0x3c00);
    assert_eq!(half_encode(1.00146484375), 0x3c02);
    assert_eq!(half_encode(2f32.powi(-25)), 0);
    assert_eq!(half_encode(next_up(2f32.powi(-25))), 1);
    assert_eq!(fp8_encode(-0.0).unwrap(), 0x80);
    assert_eq!(fp8_encode(500.0).unwrap(), 0x7e);
    assert_eq!(fp8_encode(-500.0).unwrap(), 0xfe);
    for code in 0u8..=126 {
        assert_eq!(fp8_encode(fp8_decode(code) as f32).unwrap(), code);
        if code < 126 {
            let midpoint = ((fp8_decode(code) + fp8_decode(code + 1)) / 2.0) as f32;
            assert_eq!(fp8_encode(next_down(midpoint)).unwrap(), code);
            assert_eq!(fp8_encode(next_up(midpoint)).unwrap(), code + 1);
            assert_eq!(
                fp8_encode(midpoint).unwrap(),
                if code & 1 == 0 { code } else { code + 1 }
            );
        }
    }
    for exponent in [-127, -126, -120, -20, -1, 0, 1, 20, 120, 127] {
        let power = 2f64.powi(exponent) as f32;
        assert_eq!(scale_encode(power), (exponent + 127) as u8);
        assert_eq!(
            scale_encode(next_up(power)),
            (exponent + 128).min(254) as u8
        );
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(fp8_encode(value).is_err());
    }
}

#[test]
fn independent_layout_covers_padding_and_source_precision() {
    let (rows, k, stride) = (129, 344, 357);
    let source = input(rows, k, stride, 29, false, false);
    let packed = quantize(&source, rows, k, stride).unwrap();
    let blocks4 = packed.blocks.div_ceil(4) * 4;
    let mut visited = vec![false; packed.scales.len()];
    for row in 0..rows {
        assert!(packed.values[row * packed.kp + k..(row + 1) * packed.kp]
            .iter()
            .all(|&v| v == 0));
        for block in 0..packed.blocks {
            let offset = packed_offset(row, block, blocks4);
            assert!(!visited[offset]);
            visited[offset] = true;
        }
    }
    for (used, byte) in visited.iter().zip(&packed.scales) {
        if !used {
            assert_eq!(*byte, 0);
        }
    }
    assert_eq!(packed.logical_scales[0], 127);
    assert!(packed.values[..32].iter().all(|&v| v == 0));
    let mut edge = vec![1.0f32; 32];
    edge[0] = 448.0;
    edge[1] = next_up(1.0625);
    let rounded: Vec<f32> = edge.iter().map(|&v| half_decode(half_encode(v))).collect();
    assert_ne!(
        quantize(&edge, 1, 32, 32).unwrap().values,
        quantize(&rounded, 1, 32, 32).unwrap().values,
        "weight test must distinguish original FP32 from prior half rounding"
    );
}

struct Reference {
    exact: Vec<f64>,
    sum_abs: Vec<f64>,
    quantization_error: f64,
}

fn reference(
    weight: &Packed,
    activation: &Packed,
    original_weight: &[f32],
    weight_stride: usize,
    original_input: &[f32],
    input_stride: usize,
    residual: &[f32],
    exact: bool,
) -> Reference {
    let w = weight.decoded();
    let x = activation.decoded();
    let mut result = Reference {
        exact: vec![],
        sum_abs: vec![],
        quantization_error: 0.0,
    };
    for row in 0..activation.rows {
        for channel in 0..weight.rows {
            let mut sum = 0.0f64;
            let mut absolute = 0.0f64;
            let mut original = 0.0f64;
            for col in 0..weight.k {
                let product = w[channel * weight.k + col] * x[row * weight.k + col];
                sum += product;
                absolute += product.abs();
                original += original_weight[channel * weight_stride + col] as f64
                    * original_input[row * input_stride + col] as f64;
                if exact {
                    assert_eq!(
                        sum, sum as f32 as f64,
                        "dyadic reference loses FP32 exactness"
                    );
                }
            }
            let r = residual[row * weight.rows + channel] as f64;
            result.quantization_error = result.quantization_error.max((sum - original).abs());
            sum += r;
            if exact {
                assert_eq!(sum, sum as f32 as f64);
            }
            result.exact.push(sum);
            result.sum_abs.push(absolute + r.abs());
        }
    }
    result
}

fn check_output(
    label: &str,
    actual_bits: &[u32],
    half: bool,
    k: usize,
    reference: &Reference,
    exact: bool,
) {
    assert_eq!(actual_bits.len(), reference.exact.len());
    let operations = (k + 2) as f64;
    let u = 2f64.powi(-24);
    let du = 2f64.powi(-53);
    let gamma = operations * u / (1.0 - operations * u) + operations * du / (1.0 - operations * du);
    let mut max_error = 0.0f64;
    for (i, &bits) in actual_bits.iter().enumerate() {
        let actual = if half {
            half_decode(bits as u16)
        } else {
            f32::from_bits(bits)
        } as f64;
        assert!(actual.is_finite(), "{label} nonfinite output at {i}");
        if exact {
            let expected = if half {
                half_encode(reference.exact[i] as f32) as u32
            } else {
                (reference.exact[i] as f32).to_bits()
            };
            assert_eq!(bits, expected, "{label} dyadic bitwise mismatch at {i}");
        } else {
            let bound = gamma * reference.sum_abs[i];
            let (mut low, mut high) = (reference.exact[i] - bound, reference.exact[i] + bound);
            if half {
                low = half_decode(half_encode(next_down(low as f32))) as f64;
                high = half_decode(half_encode(next_up(high as f32))) as f64;
            }
            assert!(low.is_finite() && high.is_finite());
            assert!(
                actual >= low && actual <= high,
                "{label} index={i} actual={actual} decoded64={} allowed=[{low},{high}]",
                reference.exact[i]
            );
        }
        max_error = max_error.max((actual - reference.exact[i]).abs());
    }
    eprintln!("[mxfp8-test] {label} exact={exact} max_error_vs_decoded64={max_error} separate_quantization_error={}", reference.quantization_error);
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, PinnedHostSlice};
    use kata_nn::backends::cuda::CudaRuntime;
    use kata_nn::backends::mxfp8::{
        validate_completed_status, Mxfp8Kernels, Mxfp8Output, Mxfp8Weight, Mxfp8Workspace,
        PreparedProjectionId, STATUS_WORDS,
    };
    use std::sync::{Arc, Mutex};

    static GPU_LOCK: Mutex<()> = Mutex::new(());

    fn enabled() -> bool {
        let enabled = std::env::var("KATAGO_TEST_MXFP8").as_deref() == Ok("1");
        if !enabled {
            eprintln!("skipped MXFP8 GPU test; set KATAGO_TEST_MXFP8=1 explicitly");
        }
        enabled
    }

    fn setup() -> (Arc<CudaRuntime>, Arc<CudaStream>, Mxfp8Kernels) {
        let rt = Arc::new(CudaRuntime::new().expect("explicit MXFP8 runtime request"));
        let stream = rt.device.new_stream().expect("stream");
        let kernels = Mxfp8Kernels::load(&rt).expect("MXFP8 capability must fail closed");
        (rt, stream, kernels)
    }

    fn download<T: DeviceRepr + Default + Clone>(
        stream: &Arc<CudaStream>,
        source: &CudaSlice<T>,
    ) -> Vec<T> {
        let mut host = vec![T::default(); source.len()];
        stream.memcpy_dtoh(source, &mut host).unwrap();
        stream.synchronize().unwrap();
        host
    }

    fn upload<T: DeviceRepr>(stream: &Arc<CudaStream>, source: &[T]) -> CudaSlice<T> {
        // cudarc's pageable HostSlice does not retain the caller's allocation.
        // Tests synchronise this setup copy before any temporary host vector drops.
        let device = stream.clone_htod(source).unwrap();
        stream.synchronize().unwrap();
        device
    }

    fn pin(rt: &CudaRuntime) -> PinnedHostSlice<u32> {
        unsafe { rt.device.alloc_pinned(STATUS_WORDS) }.unwrap()
    }

    fn valid(stream: &Arc<CudaStream>, ws: &Mxfp8Workspace) {
        let status = download(stream, ws.status());
        validate_completed_status(&status).expect("completed forward status");
    }

    fn check_pack(
        stream: &Arc<CudaStream>,
        data: &CudaSlice<u8>,
        scales: &CudaSlice<u8>,
        expected: &Packed,
    ) {
        let values = download(stream, data);
        let scale_bytes = download(stream, scales);
        assert_eq!(
            &values[..expected.values.len()],
            expected.values.as_slice(),
            "FP8 bytes/padding"
        );
        assert_eq!(
            &scale_bytes[..expected.scales.len()],
            expected.scales.as_slice(),
            "packed scale bytes/padding"
        );
        assert!(
            scale_bytes[expected.scales.len()..].iter().all(|&v| v == 0),
            "shared scale scratch tail must clear"
        );
    }

    enum Output {
        Half(CudaSlice<u16>),
        Float(CudaSlice<f32>),
    }

    const OUTPUT_GUARD: usize = 128;
    const HALF_CANARY: u16 = 0x3555;
    const FLOAT_CANARY: u32 = 0x4a123456;

    impl Output {
        fn new(stream: &Arc<CudaStream>, count: usize, mode: Mxfp8Output) -> Self {
            match mode {
                Mxfp8Output::Half => {
                    let mut source = vec![HALF_CANARY; count + OUTPUT_GUARD];
                    source[..count].fill(0);
                    Self::Half(upload(stream, &source))
                }
                _ => {
                    let mut source = vec![f32::from_bits(FLOAT_CANARY); count + OUTPUT_GUARD];
                    source[..count].fill(0.0);
                    Self::Float(upload(stream, &source))
                }
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
            if let Self::Float(out) = self {
                stream.memcpy_dtod(residual, out).unwrap();
            }
        }
        fn bits(&self, stream: &Arc<CudaStream>) -> Vec<u32> {
            match self {
                Self::Half(out) => {
                    let mut host = download(stream, out);
                    let count = host.len() - OUTPUT_GUARD;
                    assert!(
                        host[count..].iter().all(|&v| v == HALF_CANARY),
                        "half output tail guard"
                    );
                    host.truncate(count);
                    host.into_iter().map(u32::from).collect()
                }
                Self::Float(out) => {
                    let mut host = download(stream, out);
                    let count = host.len() - OUTPUT_GUARD;
                    assert!(
                        host[count..].iter().all(|&v| v.to_bits() == FLOAT_CANARY),
                        "float output tail guard"
                    );
                    host.truncate(count);
                    host.into_iter().map(f32::to_bits).collect()
                }
            }
        }
    }

    #[test]
    fn original_fp32_weights_and_half_activation_bytes() {
        if !enabled() {
            return;
        }
        let _guard = GPU_LOCK.lock().unwrap();
        let (rt, stream, kernels) = setup();
        // Every finite E4M3 midpoint and its neighbouring FP32 values, of both signs.
        // Each row's 448 anchor fixes S=1, independently of the tested conversion.
        let mut boundaries = Vec::new();
        for code in 0u8..126 {
            let midpoint = ((fp8_decode(code) + fp8_decode(code + 1)) * 0.5) as f32;
            for value in [next_down(midpoint), midpoint, next_up(midpoint)] {
                boundaries.extend([value, -value]);
            }
        }
        let mut conversion_source = vec![f32::NAN; boundaries.len() * 39];
        for (row, &value) in boundaries.iter().enumerate() {
            conversion_source[row * 39..row * 39 + 32].fill(0.0);
            conversion_source[row * 39] = 448.0;
            conversion_source[row * 39 + 1] = value;
        }
        let conversion_weight = Mxfp8Weight::upload_from_f32(
            &rt,
            &stream,
            &kernels,
            &conversion_source,
            boundaries.len(),
            32,
            39,
        )
        .unwrap();
        check_pack(
            &stream,
            conversion_weight.quantized_data(),
            conversion_weight.packed_scales(),
            &quantize(&conversion_source, boundaries.len(), 32, 39).unwrap(),
        );
        // Scale ceil boundaries and FP32 subnormals; upper cases remain executable finite weights.
        let mut maxima = vec![0.0, f32::from_bits(1), f32::MIN_POSITIVE];
        for exponent in [-127, -126, -100, -20, 0, 20, 100, 117] {
            let maximum = (448.0f64 * 2f64.powi(exponent)) as f32;
            maxima.extend([next_down(maximum), maximum, next_up(maximum)]);
        }
        let mut scale_source = vec![f32::NAN; maxima.len() * 37];
        for (row, &maximum) in maxima.iter().enumerate() {
            scale_source[row * 37..row * 37 + 32].fill(maximum);
            scale_source[row * 37 + 1] = -maximum;
        }
        let scale_weight = Mxfp8Weight::upload_from_f32(
            &rt,
            &stream,
            &kernels,
            &scale_source,
            maxima.len(),
            32,
            37,
        )
        .unwrap();
        check_pack(
            &stream,
            scale_weight.quantized_data(),
            scale_weight.packed_scales(),
            &quantize(&scale_source, maxima.len(), 32, 37).unwrap(),
        );
        // Adjacent FP32 values around an E4M3 midpoint differ after premature half rounding.
        let (n, k, stride) = (24, 65, 79);
        let mut original = input(n, k, stride, 37, false, false);
        original[0] = 448.0;
        original[1] = next_up(1.0625);
        original[2] = next_down(1.1875);
        original[3] = -0.0;
        original[stride] = f32::from_bits(1);
        let rounded: Vec<_> = original
            .iter()
            .map(|&v| half_decode(half_encode(v)))
            .collect();
        let expected = quantize(&original, n, k, stride).unwrap();
        assert_ne!(
            expected.values,
            quantize(&rounded, n, k, stride).unwrap().values
        );
        let weight =
            Mxfp8Weight::upload_from_f32(&rt, &stream, &kernels, &original, n, k, stride).unwrap();
        assert_eq!((weight.n(), weight.k(), weight.padded_k()), (n, k, 96));
        check_pack(
            &stream,
            weight.quantized_data(),
            weight.packed_scales(),
            &expected,
        );
        // Finite extreme encodings that dequantize beyond FP32 are not executable weights.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
            let mut values = vec![0.0f32; 24 * 32];
            values[35] = bad;
            assert!(
                Mxfp8Weight::upload_from_f32(&rt, &stream, &kernels, &values, 24, 32, 32).is_err(),
                "must reject unsafe weight {bad}"
            );
        }
        // Zero weights isolate quantization even for half activation max/subnormal edges.
        let mut ws =
            Mxfp8Workspace::new(rt.clone(), stream.clone(), kernels.clone(), 129, 344).unwrap();
        for (case, k) in [31usize, 32, 33, 65, 344, 31].into_iter().enumerate() {
            let stride = k + 7;
            let zero =
                Mxfp8Weight::upload_from_f32(&rt, &stream, &kernels, &vec![0.0; 24 * k], 24, k, k)
                    .unwrap();
            let id = ws.prepare_projection(zero, Mxfp8Output::Float).unwrap();
            let mut source = input(129, k, stride, 41 + case as u32, true, false);
            if k >= 32 {
                for col in 0..k {
                    source[stride + col] =
                        half_decode([0x0001, 0x8001, 0x03ff, 0x7bff, 0xfbff, 0][col % 6]);
                }
            }
            let half: Vec<_> = source.iter().map(|&v| half_encode(v)).collect();
            let device = upload(&stream, &half);
            let mut out = Output::new(&stream, 129 * 24, Mxfp8Output::Float);
            ws.begin_forward().unwrap();
            out.project(&mut ws, id, case as u32, &device, stride);
            valid(&stream, &ws);
            check_pack(
                &stream,
                ws.quantized_data(),
                ws.packed_scales(),
                &quantize(&source, 129, k, stride).unwrap(),
            );
            assert_eq!(
                download(&stream, &device),
                half,
                "strided source must remain read-only"
            );
            assert!(out
                .bits(&stream)
                .iter()
                .all(|&bits| f32::from_bits(bits) == 0.0));
        }
        // Later work must not modify static packed weights or their padding.
        check_pack(
            &stream,
            weight.quantized_data(),
            weight.packed_scales(),
            &expected,
        );
    }

    fn gemm_case(
        rt: &Arc<CudaRuntime>,
        stream: &Arc<CudaStream>,
        kernels: &Mxfp8Kernels,
        rows: usize,
        n: usize,
        k: usize,
        mode: Mxfp8Output,
        exact: bool,
        seed: u32,
    ) {
        let weight_stride = k + 13;
        let stride = k + 7;
        let weights = input(n, k, weight_stride, seed, false, exact);
        let activation = input(rows, k, stride, seed + 2, true, exact);
        let w = quantize(&weights, n, k, weight_stride).unwrap();
        let x = quantize(&activation, rows, k, stride).unwrap();
        let residual: Vec<_> = (0..rows * n)
            .map(|i| {
                if matches!(mode, Mxfp8Output::FloatResidual) {
                    ((i % 17) as f32 - 8.0) * 0.25
                } else {
                    0.0
                }
            })
            .collect();
        let expected = reference(
            &w,
            &x,
            &weights,
            weight_stride,
            &activation,
            stride,
            &residual,
            exact,
        );
        let weight =
            Mxfp8Weight::upload_from_f32(rt, stream, kernels, &weights, n, k, weight_stride)
                .unwrap();
        check_pack(stream, weight.quantized_data(), weight.packed_scales(), &w);
        let mut ws =
            Mxfp8Workspace::new(rt.clone(), stream.clone(), kernels.clone(), rows, k).unwrap();
        let id = ws.prepare_projection(weight.clone(), mode).unwrap();
        let info = ws.plan_info(id).unwrap();
        assert_eq!(
            (info.rows, info.n, info.k, info.padded_k),
            (rows, n, k, k.div_ceil(32) * 32)
        );
        eprintln!(
            "[mxfp8-test] shape=({rows},{n},{k}) mode={mode:?} algo={} flags={:#x}",
            info.algorithm_id, info.numerical_flags
        );
        let half: Vec<_> = activation.iter().map(|&v| half_encode(v)).collect();
        let device = upload(stream, &half);
        let d_residual = upload(stream, &residual);
        let mut out = Output::new(stream, rows * n, mode);
        ws.begin_forward().unwrap();
        if matches!(mode, Mxfp8Output::FloatResidual) {
            out.restore(stream, &d_residual);
        }
        out.project(&mut ws, id, 7, &device, stride);
        valid(stream, &ws);
        check_pack(stream, ws.quantized_data(), ws.packed_scales(), &x);
        check_output(
            &format!("({rows},{n},{k})/{mode:?}"),
            &out.bits(stream),
            matches!(mode, Mxfp8Output::Half),
            k,
            &expected,
            exact,
        );
        check_pack(stream, weight.quantized_data(), weight.packed_scales(), &w);
        assert_eq!(download(stream, &device), half);
    }

    #[test]
    fn six_dyadic_shapes_bitwise_and_random_decoded_fp64_bound() {
        if !enabled() {
            return;
        }
        let _guard = GPU_LOCK.lock().unwrap();
        let (rt, stream, kernels) = setup();
        for (rows, n, k, mode) in [
            (361, 136, 160, Mxfp8Output::Half),
            (361, 384, 384, Mxfp8Output::Half),
            (17, 384, 344, Mxfp8Output::Half),
            (129, 24, 72, Mxfp8Output::Half),
            (361, 136, 160, Mxfp8Output::Float),
            (361, 136, 160, Mxfp8Output::FloatResidual),
        ] {
            gemm_case(&rt, &stream, &kernels, rows, n, k, mode, true, 43);
        }
        for (rows, n, k, mode) in [
            (361, 136, 344, Mxfp8Output::Half),
            (17, 24, 72, Mxfp8Output::Float),
            (129, 24, 168, Mxfp8Output::FloatResidual),
        ] {
            gemm_case(&rt, &stream, &kernels, rows, n, k, mode, false, 47);
        }
    }

    #[test]
    fn same_shape_weight_scale_binding_and_shared_scratch_switches() {
        if !enabled() {
            return;
        }
        let _guard = GPU_LOCK.lock().unwrap();
        let (rt, stream, kernels) = setup();
        let rows = 129;
        let mut ws =
            Mxfp8Workspace::new(rt.clone(), stream.clone(), kernels.clone(), rows, 344).unwrap();
        let mut sources = Vec::new();
        let mut weights = Vec::new();
        let mut ids = Vec::new();
        for (index, k) in [344usize, 344, 72, 168].into_iter().enumerate() {
            let mut source = input(24, k, k + 13, 53, false, false);
            if index == 1 {
                for row in 0..24 {
                    for col in 0..k {
                        source[row * (k + 13) + col] *= 8.0;
                    }
                }
            }
            let weight =
                Mxfp8Weight::upload_from_f32(&rt, &stream, &kernels, &source, 24, k, k + 13)
                    .unwrap();
            ids.push(
                ws.prepare_projection(weight.clone(), Mxfp8Output::Float)
                    .unwrap(),
            );
            weights.push(weight);
            sources.push(source);
        }
        assert_ne!(
            ws.plan_info(ids[0]).unwrap().scale_a,
            ws.plan_info(ids[1]).unwrap().scale_a
        );
        assert_eq!(
            ws.plan_info(ids[0]).unwrap().scale_b,
            ws.plan_info(ids[1]).unwrap().scale_b
        );
        assert_ne!(
            download(&stream, weights[0].packed_scales()),
            download(&stream, weights[1].packed_scales())
        );
        let mut first = None;
        for index in [0, 1, 2, 3, 1, 2, 0] {
            let k = weights[index].k();
            let stride = k + 7;
            let activation = input(rows, k, stride, 59, true, false);
            let x = quantize(&activation, rows, k, stride).unwrap();
            let w = quantize(&sources[index], 24, k, k + 13).unwrap();
            let reference = reference(
                &w,
                &x,
                &sources[index],
                k + 13,
                &activation,
                stride,
                &vec![0.0; rows * 24],
                false,
            );
            let half: Vec<_> = activation.iter().map(|&v| half_encode(v)).collect();
            let device = upload(&stream, &half);
            let mut out = Output::new(&stream, rows * 24, Mxfp8Output::Float);
            ws.begin_forward().unwrap();
            out.project(&mut ws, ids[index], 10 + index as u32, &device, stride);
            valid(&stream, &ws);
            check_pack(&stream, ws.quantized_data(), ws.packed_scales(), &x);
            let bits = out.bits(&stream);
            check_output("shared-scratch", &bits, false, k, &reference, false);
            if index == 0 {
                if let Some(old) = &first {
                    assert_eq!(&bits, old, "return to large K must reproduce exact bytes");
                } else {
                    first = Some(bits);
                }
            }
            check_pack(
                &stream,
                weights[index].quantized_data(),
                weights[index].packed_scales(),
                &w,
            );
        }
    }

    #[test]
    fn sticky_nonfirst_layer_slot_snapshots_and_invalid_forward_recovery() {
        if !enabled() {
            return;
        }
        let _guard = GPU_LOCK.lock().unwrap();
        let (rt, stream, kernels) = setup();
        let (rows, n, k, stride) = (17, 24, 72, 79);
        let weight =
            Mxfp8Weight::upload_from_f32(&rt, &stream, &kernels, &vec![1.0; n * k], n, k, k)
                .unwrap();
        let mut ws =
            Mxfp8Workspace::new(rt.clone(), stream.clone(), kernels.clone(), rows, k).unwrap();
        let id = ws
            .prepare_projection(weight.clone(), Mxfp8Output::Float)
            .unwrap();
        let half_id = ws
            .prepare_projection(weight.clone(), Mxfp8Output::Half)
            .unwrap();
        let source = input(rows, k, stride, 61, true, true);
        let good: Vec<_> = source.iter().map(|&v| half_encode(v)).collect();
        let mut nan = good.clone();
        nan[stride + 3] = 0x7e00;
        let mut inf = good.clone();
        inf[2 * stride + 5] = 0xfc00;
        let d_good = upload(&stream, &good);
        let d_nan = upload(&stream, &nan);
        let d_inf = upload(&stream, &inf);
        let mut out = upload(&stream, &vec![0.0f32; rows * n]);
        let mut half_out = upload(&stream, &vec![0u16; rows * n]);
        let mut slots = [pin(&rt), pin(&rt)];
        let done = [
            rt.device.new_event(None).unwrap(),
            rt.device.new_event(None).unwrap(),
        ];
        ws.begin_forward().unwrap();
        ws.project_f32(id, 3, &d_good, stride, &mut out).unwrap();
        ws.project_f32(id, 17, &d_nan, stride, &mut out).unwrap();
        ws.project_f32(id, 23, &d_inf, stride, &mut out).unwrap();
        ws.project_f32(id, 31, &d_good, stride, &mut out).unwrap();
        ws.enqueue_status_copy(&mut slots[0]).unwrap();
        done[0].record(&stream).unwrap();
        // Submit the next slot before examining the first slot's status.
        ws.begin_forward().unwrap();
        ws.project_f32(id, 3, &d_good, stride, &mut out).unwrap();
        ws.enqueue_status_copy(&mut slots[1]).unwrap();
        done[1].record(&stream).unwrap();
        done[0].synchronize().unwrap();
        let failed = slots[0].as_slice().unwrap();
        assert_eq!(
            failed[0] & 3,
            3,
            "later valid layer must not clear NaN/Inf bits"
        );
        assert_eq!(failed[1], 17, "first bad layer is sticky");
        assert!(validate_completed_status(failed).is_err());
        done[1].synchronize().unwrap();
        validate_completed_status(slots[1].as_slice().unwrap()).unwrap();
        valid(&stream, &ws);
        let recovered = download(&stream, &out);
        // Reverse the status order: the first successful slot must remain valid.
        ws.begin_forward().unwrap();
        ws.project_f32(id, 5, &d_good, stride, &mut out).unwrap();
        ws.enqueue_status_copy(&mut slots[0]).unwrap();
        done[0].record(&stream).unwrap();
        ws.begin_forward().unwrap();
        ws.project_f32(id, 19, &d_nan, stride, &mut out).unwrap();
        ws.enqueue_status_copy(&mut slots[1]).unwrap();
        done[1].record(&stream).unwrap();
        done[0].synchronize().unwrap();
        validate_completed_status(slots[0].as_slice().unwrap()).unwrap();
        done[1].synchronize().unwrap();
        assert!(validate_completed_status(slots[1].as_slice().unwrap()).is_err());
        // Finite inputs can still overflow half GEMM storage; the whole forward fails.
        let d_large = upload(&stream, &vec![half_encode(65504.0); rows * stride]);
        ws.begin_forward().unwrap();
        ws.project_half(half_id, 42, &d_large, stride, &mut half_out)
            .unwrap();
        ws.project_f32(id, 43, &d_good, stride, &mut out).unwrap();
        let overflow = download(&stream, ws.status());
        assert_ne!(overflow[0], 0, "finite-input half overflow must fail");
        assert_eq!(overflow[1], 42);
        assert!(validate_completed_status(&overflow).is_err());
        // Host misuse is rejected instead of launching a different prepared plan.
        assert!(ws
            .project_f32(half_id, 1, &d_good, stride, &mut out)
            .is_err());
        assert!(ws
            .project_half(id, 1, &d_good, stride, &mut half_out)
            .is_err());
        assert!(ws.project_f32(id, 1, &d_good, k - 1, &mut out).is_err());
        let short = upload(&stream, &vec![0u16; k - 1]);
        assert!(ws.project_f32(id, 1, &short, k, &mut out).is_err());
        let mut small_out = upload(&stream, &vec![0.0f32; rows * n - 1]);
        assert!(ws
            .project_f32(id, 1, &d_good, stride, &mut small_out)
            .is_err());
        let mut other =
            Mxfp8Workspace::new(rt.clone(), stream.clone(), kernels.clone(), 33, k).unwrap();
        let other_id = other
            .prepare_projection(weight, Mxfp8Output::Float)
            .unwrap();
        assert!(ws
            .project_f32(other_id, 1, &d_good, stride, &mut out)
            .is_err());
        // A physical row/batch change has its own scratch/status and must not reset an old snapshot.
        let other_input = upload(&stream, &vec![half_encode(0.25); 33 * k]);
        let mut other_out = upload(&stream, &vec![0.0f32; 33 * n]);
        other.begin_forward().unwrap();
        other
            .project_f32(other_id, 1, &other_input, k, &mut other_out)
            .unwrap();
        valid(&stream, &other);
        assert!(validate_completed_status(slots[1].as_slice().unwrap()).is_err());
        ws.begin_forward().unwrap();
        ws.project_f32(id, 3, &d_good, stride, &mut out).unwrap();
        valid(&stream, &ws);
        assert_eq!(
            download(&stream, &out)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            recovered.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn captured_a_b_a_matches_independent_fresh_direct_bitwise() {
        if !enabled() {
            return;
        }
        let _guard = GPU_LOCK.lock().unwrap();
        let (rt, stream, kernels) = setup();
        for (rows, n, k, mode) in [
            (361, 136, 344, Mxfp8Output::Half),
            (129, 24, 168, Mxfp8Output::FloatResidual),
        ] {
            let stride = k + 7;
            let weight_stride = k + 13;
            let weights = input(n, k, weight_stride, 67, false, false);
            let w = quantize(&weights, n, k, weight_stride).unwrap();
            let weight =
                Mxfp8Weight::upload_from_f32(&rt, &stream, &kernels, &weights, n, k, weight_stride)
                    .unwrap();
            let sources = [
                input(rows, k, stride, 71, true, false),
                input(rows, k, stride, 73, true, false),
            ];
            let quantized = [
                quantize(&sources[0], rows, k, stride).unwrap(),
                quantize(&sources[1], rows, k, stride).unwrap(),
            ];
            assert_ne!(
                quantized[0].values, quantized[1].values,
                "A/B must change FP8 data"
            );
            assert_ne!(
                quantized[0].scales, quantized[1].scales,
                "A/B must change scales"
            );
            let half_sources: Vec<Vec<_>> = sources
                .iter()
                .map(|source| source.iter().map(|&v| half_encode(v)).collect())
                .collect();
            let d_sources: Vec<_> = half_sources
                .iter()
                .map(|source| upload(&stream, source))
                .collect();
            let residual: Vec<_> = (0..rows * n)
                .map(|i| {
                    if matches!(mode, Mxfp8Output::FloatResidual) {
                        ((i % 23) as f32 - 11.0) * 0.125
                    } else {
                        0.0
                    }
                })
                .collect();
            let expected = [
                reference(
                    &w,
                    &quantized[0],
                    &weights,
                    weight_stride,
                    &sources[0],
                    stride,
                    &residual,
                    false,
                ),
                reference(
                    &w,
                    &quantized[1],
                    &weights,
                    weight_stride,
                    &sources[1],
                    stride,
                    &residual,
                    false,
                ),
            ];
            let d_residual = upload(&stream, &residual);
            let mut captured_ws =
                Mxfp8Workspace::new(rt.clone(), stream.clone(), kernels.clone(), rows, k).unwrap();
            let mut direct_ws =
                Mxfp8Workspace::new(rt.clone(), stream.clone(), kernels.clone(), rows, k).unwrap();
            let captured_id = captured_ws
                .prepare_projection(weight.clone(), mode)
                .unwrap();
            let direct_id = direct_ws.prepare_projection(weight.clone(), mode).unwrap();
            let graph_resources = captured_ws.graph_resources().unwrap();
            let captured_scales = (
                captured_ws.plan_info(captured_id).unwrap().scale_a,
                captured_ws.plan_info(captured_id).unwrap().scale_b,
            );
            let direct_scales = (
                direct_ws.plan_info(direct_id).unwrap().scale_a,
                direct_ws.plan_info(direct_id).unwrap().scale_b,
            );
            assert_eq!(
                captured_scales.0, direct_scales.0,
                "static weight may be shared read-only"
            );
            assert_ne!(
                captured_scales.1, direct_scales.1,
                "direct activation scale storage must be independent"
            );
            let mut captured_input = upload(&stream, &half_sources[0]);
            let mut direct_input = upload(&stream, &half_sources[0]);
            let mut captured_out = Output::new(&stream, rows * n, mode);
            let mut direct_out = Output::new(&stream, rows * n, mode);
            let mut captured_status = pin(&rt);
            let mut direct_status = pin(&rt);
            // Warm both complete pipelines outside capture: lazy allocation/loading cannot leak into Graph.
            captured_ws.begin_forward().unwrap();
            if matches!(mode, Mxfp8Output::FloatResidual) {
                captured_out.restore(&stream, &d_residual);
            }
            captured_out.project(&mut captured_ws, captured_id, 9, &captured_input, stride);
            captured_ws.complete_warmup().unwrap();
            valid(&stream, &captured_ws);
            direct_ws.begin_forward().unwrap();
            if matches!(mode, Mxfp8Output::FloatResidual) {
                direct_out.restore(&stream, &d_residual);
            }
            direct_out.project(&mut direct_ws, direct_id, 9, &direct_input, stride);
            direct_ws.complete_warmup().unwrap();
            valid(&stream, &direct_ws);
            stream.synchronize().unwrap();
            stream
                .begin_capture(
                    cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                )
                .unwrap();
            captured_ws.begin_forward().unwrap();
            // Test residual restoration is explicit; the production operator must consume the current C.
            if matches!(mode, Mxfp8Output::FloatResidual) {
                captured_out.restore(&stream, &d_residual);
            }
            captured_out.project(&mut captured_ws, captured_id, 9, &captured_input, stride);
            let graph = stream.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY)
                .unwrap().expect("nonempty MXFP8 Graph");
            graph.upload().unwrap();
            let mut first_a = None;
            let mut first_b = None;
            for variant in [0, 1, 0] {
                stream
                    .memcpy_dtod(&d_sources[variant], &mut direct_input)
                    .unwrap();
                direct_ws.begin_forward().unwrap();
                if matches!(mode, Mxfp8Output::FloatResidual) {
                    direct_out.restore(&stream, &d_residual);
                }
                direct_out.project(&mut direct_ws, direct_id, 9, &direct_input, stride);
                direct_ws.enqueue_status_copy(&mut direct_status).unwrap();
                stream
                    .memcpy_dtod(&d_sources[variant], &mut captured_input)
                    .unwrap();
                graph.launch().unwrap();
                captured_ws
                    .enqueue_status_copy(&mut captured_status)
                    .unwrap();
                stream.synchronize().unwrap();
                validate_completed_status(direct_status.as_slice().unwrap()).unwrap();
                validate_completed_status(captured_status.as_slice().unwrap()).unwrap();
                check_pack(
                    &stream,
                    captured_ws.quantized_data(),
                    captured_ws.packed_scales(),
                    &quantized[variant],
                );
                check_pack(
                    &stream,
                    direct_ws.quantized_data(),
                    direct_ws.packed_scales(),
                    &quantized[variant],
                );
                let direct = direct_out.bits(&stream);
                let captured = captured_out.bits(&stream);
                check_output(
                    "Graph/fresh-direct",
                    &direct,
                    matches!(mode, Mxfp8Output::Half),
                    k,
                    &expected[variant],
                    false,
                );
                assert_eq!(
                    captured, direct,
                    "Graph replay must match independent fresh direct bitwise"
                );
                if variant == 0 {
                    if let Some(first) = &first_a {
                        assert_eq!(&captured, first, "A-B-A must restore A exactly");
                    } else {
                        first_a = Some(captured);
                    }
                } else {
                    first_b = Some(captured);
                }
                assert_eq!(
                    captured_scales,
                    (
                        captured_ws.plan_info(captured_id).unwrap().scale_a,
                        captured_ws.plan_info(captured_id).unwrap().scale_b
                    )
                );
            }
            // The captured error clear must run on each replay, without hiding this replay's invalid input.
            let mut bad = half_sources[0].clone();
            bad[stride + 7] = 0x7e00;
            let d_bad = upload(&stream, &bad);
            stream.memcpy_dtod(&d_bad, &mut captured_input).unwrap();
            graph.launch().unwrap();
            captured_ws
                .enqueue_status_copy(&mut captured_status)
                .unwrap();
            stream.synchronize().unwrap();
            assert_ne!(captured_status.as_slice().unwrap()[0] & 1, 0);
            assert!(validate_completed_status(captured_status.as_slice().unwrap()).is_err());
            // Invalid Graph output is deliberately discarded, not compared or exposed as successful output.
            stream
                .memcpy_dtod(&d_sources[0], &mut captured_input)
                .unwrap();
            graph.launch().unwrap();
            captured_ws
                .enqueue_status_copy(&mut captured_status)
                .unwrap();
            stream.synchronize().unwrap();
            validate_completed_status(captured_status.as_slice().unwrap()).unwrap();
            assert_eq!(
                captured_out.bits(&stream),
                *first_a.as_ref().unwrap(),
                "valid replay after rejection must recover"
            );
            check_pack(&stream, weight.quantized_data(), weight.packed_scales(), &w);
            assert_eq!(
                download(&stream, &d_residual)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                residual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "read-only residual source changed"
            );
            // The keepalive token must retain plans/weights/scratch even after wrappers drop.
            assert_ne!(
                first_a, first_b,
                "lifetime replay must observably change output"
            );
            drop(captured_ws);
            drop(direct_ws);
            drop(weight);
            stream
                .memcpy_dtod(&d_sources[1], &mut captured_input)
                .unwrap();
            graph.launch().unwrap();
            stream.synchronize().unwrap();
            assert_eq!(
                captured_out.bits(&stream),
                first_b.unwrap(),
                "Graph resources survive wrapper destruction"
            );
            // Caller-owned input/output buffers remain alive; Graph drops before its keepalive.
            drop(graph);
            drop(graph_resources);
        }
    }
}
