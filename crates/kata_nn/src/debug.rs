//! Debug printing utilities for neural net tensors.
//!
//! Corresponds to `cpp/neuralnet/debugprint.h` and `cpp/neuralnet/debugprint.cpp`.
//! All functions operate on host-side float data; backends are responsible for
//! copying device data to host before calling these.

use std::io::{self, Write};

/// Compute summary statistics over `data`, skipping elements where `mask`
/// is present and zero.
fn compute_stats(data: &[f32], mask: Option<&[f32]>) -> (usize, f32, f32, f64, f64) {
    let mut valid_count = 0usize;
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;

    for (i, &v) in data.iter().enumerate() {
        if let Some(m) = mask {
            if m.get(i).copied().unwrap_or(0.0f32) == 0.0f32 {
                continue;
            }
        }
        valid_count += 1;
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
        let vd = f64::from(v);
        sum += vd;
        sum_sq += vd * vd;
    }

    if valid_count == 0 {
        min_val = 0.0f32;
        max_val = 0.0f32;
    }

    (valid_count, min_val, max_val, sum, sum_sq)
}

fn write_stats_line<W: Write>(
    writer: &mut W,
    name: &str,
    shape_str: &str,
    stats: (usize, f32, f32, f64, f64),
    data: &[f32],
    num_to_print: usize,
) -> io::Result<()> {
    let (valid_count, min_val, max_val, sum, sum_sq) = stats;
    let mean = if valid_count > 0 {
        sum / valid_count as f64
    } else {
        0.0
    };
    let rms = if valid_count > 0 {
        (sum_sq / valid_count as f64).sqrt()
    } else {
        0.0
    };

    write!(
        writer,
        "DEBUG {} {} valid={} min={:.6e} max={:.6e} mean={:.6e} rms={:.6e}",
        name, shape_str, valid_count, min_val, max_val, mean, rms
    )?;

    let n = num_to_print.min(data.len());
    if n > 0 {
        write!(writer, " first{}=", n)?;
        for &v in &data[..n] {
            write!(writer, " {:.6e}", v)?;
        }
    }
    writeln!(writer)
}

/// Build a flat mask of length `total_size` from a spatial mask of shape
/// `[n_size, spatial_size]`. The returned vector contains `1.0` for elements
/// whose spatial position is valid and `0.0` otherwise.
fn expand_spatial_mask(
    dim_order: &str,
    dim0: usize,
    dim1: usize,
    dim2: usize,
    n_size: usize,
    spatial_size: usize,
    mask: &[f32],
) -> Vec<f32> {
    let total_size = dim0 * dim1 * dim2;
    let mut flat_mask = vec![0.0f32; total_size];

    if total_size == 0 || n_size * spatial_size == 0 {
        return flat_mask;
    }
    let c_size = total_size / (n_size * spatial_size);

    if dim_order.len() >= 3 && dim_order.as_bytes()[1] == b'S' {
        // NSC ordering: dim0=N, dim1=S, dim2=C
        for n in 0..n_size {
            for s in 0..spatial_size {
                let m = mask[n * spatial_size + s];
                for c in 0..c_size {
                    flat_mask[(n * spatial_size + s) * c_size + c] = m;
                }
            }
        }
    } else if dim_order.len() >= 3 && dim_order.as_bytes()[1] == b'C' {
        // NCS ordering: dim0=N, dim1=C, dim2=S
        for n in 0..n_size {
            for c in 0..c_size {
                for s in 0..spatial_size {
                    flat_mask[(n * c_size + c) * spatial_size + s] = mask[n * spatial_size + s];
                }
            }
        }
    } else {
        // Unknown ordering: treat all positions as valid.
        flat_mask.fill(1.0f32);
    }

    flat_mask
}

/// Print a one-line summary of a flat buffer to stderr.
///
/// `mask`, if given, must have the same length as `data`; zero entries are
/// skipped. The first `num_to_print` raw values are appended to the line.
pub fn print_summary(name: &str, data: &[f32], mask: Option<&[f32]>, num_to_print: usize) {
    let stats = compute_stats(data, mask);
    let shape_str = format!("[{}]", data.len());
    let mut stderr = io::stderr().lock();
    let _ = write_stats_line(&mut stderr, name, &shape_str, stats, data, num_to_print);
}

#[allow(clippy::too_many_arguments)]
/// Print a one-line summary of a 3-D tensor to stderr.
///
/// `dim_order` is a label such as `"NCS"` or `"NSC"` and only affects how
/// `mask` is expanded. Statistics are computed over all valid spatial
/// positions regardless of ordering.
pub fn print_3d_summary(
    name: &str,
    data: &[f32],
    dim0: usize,
    dim1: usize,
    dim2: usize,
    dim_order: &str,
    n_size: usize,
    spatial_size: usize,
    mask: Option<&[f32]>,
    num_to_print: usize,
) {
    let total_size = dim0 * dim1 * dim2;
    assert_eq!(
        data.len(),
        total_size,
        "data length does not match declared 3-D shape"
    );

    let stats = if let Some(m) = mask {
        let flat_mask = expand_spatial_mask(dim_order, dim0, dim1, dim2, n_size, spatial_size, m);
        compute_stats(data, Some(&flat_mask))
    } else {
        compute_stats(data, None)
    };

    let shape_str = format!("[{} {}x{}x{}]", dim_order, dim0, dim1, dim2);
    let mut stderr = io::stderr().lock();
    let _ = write_stats_line(&mut stderr, name, &shape_str, stats, data, num_to_print);
}

/// Print a one-line summary of a 2-D `[n_size, c_size]` tensor to stderr.
pub fn print_2d_summary(
    name: &str,
    data: &[f32],
    n_size: usize,
    c_size: usize,
    num_to_print: usize,
) {
    let total_size = n_size * c_size;
    assert_eq!(
        data.len(),
        total_size,
        "data length does not match declared 2-D shape"
    );

    let stats = compute_stats(data, None);
    let shape_str = format!("[NC {}x{}]", n_size, c_size);
    let mut stderr = io::stderr().lock();
    let _ = write_stats_line(&mut stderr, name, &shape_str, stats, data, num_to_print);
}

/// Verbosely dump every element of a flat buffer to stderr.
pub fn print_verbose(name: &str, data: &[f32]) {
    let mut stderr = io::stderr().lock();
    let _ = writeln!(
        &mut stderr,
        "========================================================="
    );
    let _ = writeln!(&mut stderr, "{} [{}]", name, data.len());
    for (i, &v) in data.iter().enumerate() {
        let _ = write!(&mut stderr, "{:.8e} ", v);
        if (i + 1) % 16 == 0 {
            let _ = writeln!(&mut stderr);
        }
    }
    if data.len() % 16 != 0 {
        let _ = writeln!(&mut stderr);
    }
    let _ = writeln!(
        &mut stderr,
        "========================================================="
    );
}

/// Verbosely dump every element of a 3-D tensor to stderr.
pub fn print_3d_verbose(
    name: &str,
    data: &[f32],
    dim0: usize,
    dim1: usize,
    dim2: usize,
    dim_order: &str,
) {
    let total_size = dim0 * dim1 * dim2;
    assert_eq!(
        data.len(),
        total_size,
        "data length does not match declared 3-D shape"
    );

    let mut stderr = io::stderr().lock();
    let _ = writeln!(
        &mut stderr,
        "========================================================="
    );
    let _ = writeln!(
        &mut stderr,
        "{} [{} {}x{}x{}]",
        name, dim_order, dim0, dim1, dim2
    );
    let mut i = 0usize;
    for d0 in 0..dim0 {
        let _ = writeln!(
            &mut stderr,
            "-({}={})--------------------",
            dim_order.chars().next().unwrap_or('?'),
            d0
        );
        for _ in 0..dim1 {
            for _ in 0..dim2 {
                let _ = write!(&mut stderr, "{:.8e} ", data[i]);
                i += 1;
            }
            let _ = writeln!(&mut stderr);
        }
        let _ = writeln!(&mut stderr);
    }
    let _ = writeln!(
        &mut stderr,
        "========================================================="
    );
}

/// Verbosely dump every element of a 2-D `[n_size, c_size]` tensor to stderr.
pub fn print_2d_verbose(name: &str, data: &[f32], n_size: usize, c_size: usize) {
    let total_size = n_size * c_size;
    assert_eq!(
        data.len(),
        total_size,
        "data length does not match declared 2-D shape"
    );

    let mut stderr = io::stderr().lock();
    let _ = writeln!(
        &mut stderr,
        "========================================================="
    );
    let _ = writeln!(&mut stderr, "{} [NC {}x{}]", name, n_size, c_size);
    let mut i = 0usize;
    for n in 0..n_size {
        let _ = writeln!(&mut stderr, "-(n={})--------------------", n);
        for _ in 0..c_size {
            let _ = write!(&mut stderr, "{:.8e} ", data[i]);
            i += 1;
        }
        let _ = writeln!(&mut stderr);
    }
    let _ = writeln!(&mut stderr);
    let _ = writeln!(
        &mut stderr,
        "========================================================="
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_summary_line(line: &str) -> (usize, f64, f64, f64, f64) {
        // valid=... min=... max=... mean=... rms=...
        let valid = extract_after_usize(line, "valid=");
        let min = extract_after_f64(line, "min=");
        let max = extract_after_f64(line, "max=");
        let mean = extract_after_f64(line, "mean=");
        let rms = extract_after_f64(line, "rms=");
        (valid, min, max, mean, rms)
    }

    fn extract_after_usize(line: &str, key: &str) -> usize {
        let start = line
            .find(key)
            .unwrap_or_else(|| panic!("missing {} in {}", key, line));
        let rest = &line[start + key.len()..];
        let end = rest.find(' ').unwrap_or(rest.len());
        rest[..end].trim().parse::<usize>().unwrap_or_else(|_| {
            let alt = rest[..end]
                .trim()
                .split('=')
                .next()
                .unwrap_or(rest[..end].trim());
            alt.parse::<usize>()
                .unwrap_or_else(|_| panic!("cannot parse {} from {}", key, line))
        })
    }

    fn extract_after_f64(line: &str, key: &str) -> f64 {
        let start = line
            .find(key)
            .unwrap_or_else(|| panic!("missing {} in {}", key, line));
        let rest = &line[start + key.len()..];
        let end = rest.find(' ').unwrap_or(rest.len());
        rest[..end].trim().parse::<f64>().unwrap_or_else(|_| {
            // The value may be followed by " firstN=" with no space before first=,
            // so if parsing fails, try trimming at '='.
            let alt = rest[..end]
                .trim()
                .split('=')
                .next()
                .unwrap_or(rest[..end].trim());
            alt.parse::<f64>()
                .unwrap_or_else(|_| panic!("cannot parse {} from {}", key, line))
        })
    }

    #[test]
    fn test_compute_stats_basic() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let (count, min, max, sum, sum_sq) = compute_stats(&data, None);
        assert_eq!(count, 4);
        assert!((min - 1.0).abs() < 1e-6);
        assert!((max - 4.0).abs() < 1e-6);
        assert!((sum - 10.0).abs() < 1e-6);
        assert!((sum_sq - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_stats_with_mask() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mask = vec![1.0f32, 0.0, 1.0, 0.0];
        let (count, min, max, sum, _sum_sq) = compute_stats(&data, Some(&mask));
        assert_eq!(count, 2);
        assert!((min - 1.0).abs() < 1e-6);
        assert!((max - 3.0).abs() < 1e-6);
        assert!((sum - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_stats_empty() {
        let data: Vec<f32> = vec![];
        let (count, min, max, sum, sum_sq) = compute_stats(&data, None);
        assert_eq!(count, 0);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
        assert_eq!(sum, 0.0);
        assert_eq!(sum_sq, 0.0);
    }

    #[test]
    fn test_format_summary_line() {
        let data = vec![1.0f32, 2.0, 3.0];
        let stats = compute_stats(&data, None);
        let mut buf: Vec<u8> = Vec::new();
        write_stats_line(&mut buf, "foo", "[3]", stats, &data, 2).unwrap();
        let line = String::from_utf8(buf).unwrap();
        assert!(line.starts_with("DEBUG foo [3] "));
        let (count, min, max, mean, rms) = parse_summary_line(&line);
        assert_eq!(count, 3);
        assert!((min - 1.0).abs() < 1e-5);
        assert!((max - 3.0).abs() < 1e-5);
        assert!((mean - 2.0).abs() < 1e-5);
        assert!((rms - ((14.0f64 / 3.0).sqrt())).abs() < 1e-5);
        assert!(line.contains(" first2="));
    }

    #[test]
    fn test_expand_spatial_mask_ncs() {
        // N=1, C=2, S=3 => total 6.
        let mask = vec![1.0f32, 0.0, 1.0]; // spatial positions 0 and 2 valid
        let flat = expand_spatial_mask("NCS", 1, 2, 3, 1, 3, &mask);
        // Layout: [(n*c + c)*s + s]
        // c0: s0=1, s1=0, s2=1
        // c1: s0=1, s1=0, s2=1
        let expected = vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        assert_eq!(flat, expected);
    }

    #[test]
    fn test_expand_spatial_mask_nsc() {
        // N=1, S=3, C=2 => total 6.
        let mask = vec![1.0f32, 0.0, 1.0];
        let flat = expand_spatial_mask("NSC", 1, 3, 2, 1, 3, &mask);
        // Layout: [(n*s + s)*c + c]
        // s0: c0=1, c1=1
        // s1: c0=0, c1=0
        // s2: c0=1, c1=1
        let expected = vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        assert_eq!(flat, expected);
    }

    #[test]
    fn test_expand_spatial_mask_unknown_order_defaults_valid() {
        let mask = vec![0.0f32];
        let flat = expand_spatial_mask("XYZ", 1, 1, 1, 1, 1, &mask);
        assert_eq!(flat, vec![1.0]);
    }
}
