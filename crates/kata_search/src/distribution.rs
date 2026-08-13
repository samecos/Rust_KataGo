//! Tabulated PDF/CDF lookup with linear interpolation.
//!
//! Corresponds to `cpp/search/distributiontable.h` and `cpp/search/distributiontable.cpp`.

/// Precomputed lookup table for a one-dimensional probability distribution.
pub struct DistributionTable {
    pdf_table: Vec<f64>,
    cdf_table: Vec<f64>,
    size: usize,
    min_z: f64,
    max_z: f64,
}

impl DistributionTable {
    /// Build a lookup table by sampling `pdf` and `cdf` on a uniform grid.
    pub fn new(
        pdf: impl Fn(f64) -> f64,
        cdf: impl Fn(f64) -> f64,
        min_z: f64,
        max_z: f64,
        size: usize,
    ) -> Self {
        assert!(size >= 2, "DistributionTable size must be at least 2");
        assert!(
            max_z > min_z,
            "DistributionTable maxZ must be greater than minZ"
        );

        let mut pdf_table = vec![0.0; size];
        let mut cdf_table = vec![0.0; size];

        for i in 0..size {
            if i == 0 {
                pdf_table[i] = 0.0;
                cdf_table[i] = 0.0;
            } else if i == size - 1 {
                pdf_table[i] = 0.0;
                cdf_table[i] = 1.0;
            } else {
                let z = min_z + i as f64 * (max_z - min_z) / (size as f64 - 1.0);
                pdf_table[i] = pdf(z);
                cdf_table[i] = cdf(z);
            }
        }

        Self {
            pdf_table,
            cdf_table,
            size,
            min_z,
            max_z,
        }
    }

    /// Look up both the PDF and CDF at `z`.
    pub fn get_pdf_cdf(&self, z: f64) -> (f64, f64) {
        let d = (self.size - 1) as f64 * (z - self.min_z) / (self.max_z - self.min_z);
        if d <= 0.0 {
            return (0.0, 0.0);
        }
        let idx = d as usize;
        if idx >= self.size - 1 {
            return (0.0, 1.0);
        }
        let lambda = d - idx as f64;
        let pdf = self.pdf_table[idx] + lambda * (self.pdf_table[idx + 1] - self.pdf_table[idx]);
        let cdf = self.cdf_table[idx] + lambda * (self.cdf_table[idx + 1] - self.cdf_table[idx]);
        (pdf, cdf)
    }

    /// Look up the PDF at `z`.
    pub fn get_pdf(&self, z: f64) -> f64 {
        let d = (self.size - 1) as f64 * (z - self.min_z) / (self.max_z - self.min_z);
        if d <= 0.0 {
            return 0.0;
        }
        let idx = d as usize;
        if idx >= self.size - 1 {
            return 0.0;
        }
        let lambda = d - idx as f64;
        self.pdf_table[idx] + lambda * (self.pdf_table[idx + 1] - self.pdf_table[idx])
    }

    /// Look up the CDF at `z`.
    pub fn get_cdf(&self, z: f64) -> f64 {
        let d = (self.size - 1) as f64 * (z - self.min_z) / (self.max_z - self.min_z);
        if d <= 0.0 {
            return 0.0;
        }
        let idx = d as usize;
        if idx >= self.size - 1 {
            return 1.0;
        }
        let lambda = d - idx as f64;
        self.cdf_table[idx] + lambda * (self.cdf_table[idx + 1] - self.cdf_table[idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_table(size: usize) -> DistributionTable {
        DistributionTable::new(|_z| 1.0, |z| z, 0.0, 1.0, size)
    }

    #[test]
    fn test_endpoints() {
        let table = uniform_table(5);
        assert_eq!(table.get_pdf_cdf(0.0), (0.0, 0.0));
        assert_eq!(table.get_pdf_cdf(-0.1), (0.0, 0.0));
        assert_eq!(table.get_pdf_cdf(1.0), (0.0, 1.0));
        assert_eq!(table.get_pdf_cdf(1.1), (0.0, 1.0));
    }

    #[test]
    fn test_uniform_midpoint() {
        let table = uniform_table(5);
        // Sample points are 0, 0.25, 0.5, 0.75, 1. Interior values are exact.
        let (pdf, cdf) = table.get_pdf_cdf(0.5);
        assert!((pdf - 1.0).abs() < 1e-12);
        assert!((cdf - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_uniform_interpolated() {
        let table = uniform_table(5);
        let (pdf, cdf) = table.get_pdf_cdf(0.375);
        assert!((pdf - 1.0).abs() < 1e-12);
        assert!((cdf - 0.375).abs() < 1e-12);
    }

    #[test]
    fn test_linear_cdf() {
        // cdf(z) = z^2 on [0,1] -> pdf(z) = 2z.
        let table = DistributionTable::new(|z| 2.0 * z, |z| z * z, 0.0, 1.0, 5);
        let (_, cdf) = table.get_pdf_cdf(0.5);
        assert!((cdf - 0.25).abs() < 1e-12);
        let (pdf, _) = table.get_pdf_cdf(0.5);
        assert!((pdf - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_get_pdf_and_cdf_match() {
        let table = uniform_table(5);
        for z in [0.1, 0.33, 0.66, 0.9] {
            let (pdf, cdf) = table.get_pdf_cdf(z);
            assert!((pdf - table.get_pdf(z)).abs() < 1e-12);
            assert!((cdf - table.get_cdf(z)).abs() < 1e-12);
        }
    }

    #[test]
    #[should_panic]
    fn test_rejects_single_bucket() {
        DistributionTable::new(|_z| 1.0, |z| z, 0.0, 1.0, 1);
    }
}
