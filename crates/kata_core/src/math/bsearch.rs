//! Binary search utilities.
//!
//! Corresponds to `cpp/core/bsearch.h` and `cpp/core/bsearch.cpp`.

/// Find the first index `i` in the sorted slice `arr` within `[low, high)`
/// where `arr[i] > x`. Returns `high` if no such index exists.
///
/// `arr` must be sorted in ascending order.
pub fn find_first_gt(arr: &[f64], x: f64, low: usize, high: usize) -> usize {
    if low >= high {
        return high;
    }
    let mid = (low + high) / 2;
    if arr[mid] > x {
        find_first_gt(arr, x, low, mid)
    } else {
        find_first_gt(arr, x, mid + 1, high)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_first_gt() {
        let arr: Vec<f64> = (0..13).map(|i| i as f64).collect();
        for i in 0..arr.len() {
            assert_eq!(find_first_gt(&arr, i as f64, 0, arr.len()), i + 1);
            assert_eq!(find_first_gt(&arr, i as f64 + 0.7, 0, arr.len()), i + 1);
            assert_eq!(find_first_gt(&arr, i as f64 + 0.99, 0, arr.len()), i + 1);
            assert_eq!(find_first_gt(&arr, i as f64 - 0.01, 0, arr.len()), i);
        }
    }

    #[test]
    fn test_find_first_gt_empty() {
        let arr: [f64; 0] = [];
        assert_eq!(find_first_gt(&arr, 0.0, 0, 0), 0);
    }
}
