//! Hash utilities and 128-bit hash type.
//!
//! Corresponds to `cpp/core/hash.h` and `cpp/core/hash.cpp`.

pub mod md5;
pub mod sha2;

use std::fmt;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign};

/// 128-bit hash value, stored as two little-endian `u64` halves.
///
/// Printing `hash1` followed by `hash0` yields the standard 32-digit hex form.
#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq)]
pub struct Hash128 {
    pub hash0: u64,
    pub hash1: u64,
}

impl Hash128 {
    pub const fn new(hash0: u64, hash1: u64) -> Self {
        Self { hash0, hash1 }
    }

    /// Parse a 32-digit hex string into a `Hash128`.
    pub fn from_hex_string(s: &str) -> Result<Self, crate::global::IOError> {
        if s.len() != 32 {
            return Err(crate::global::IOError(format!(
                "Hash128::ofString expected 32 hex chars, got {}",
                s.len()
            )));
        }
        let h1 = u64::from_str_radix(&s[..16], 16)
            .map_err(|_| crate::global::IOError(format!("Hash128::ofString invalid hex: {}", s)))?;
        let h0 = u64::from_str_radix(&s[16..], 16)
            .map_err(|_| crate::global::IOError(format!("Hash128::ofString invalid hex: {}", s)))?;
        Ok(Self {
            hash0: h0,
            hash1: h1,
        })
    }
}

impl fmt::Display for Hash128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016X}{:016X}", self.hash1, self.hash0)
    }
}

impl PartialOrd for Hash128 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Hash128 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.hash1
            .cmp(&other.hash1)
            .then_with(|| self.hash0.cmp(&other.hash0))
    }
}

impl BitXor for Hash128 {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self {
            hash0: self.hash0 ^ rhs.hash0,
            hash1: self.hash1 ^ rhs.hash1,
        }
    }
}

impl BitXorAssign for Hash128 {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.hash0 ^= rhs.hash0;
        self.hash1 ^= rhs.hash1;
    }
}

impl BitOr for Hash128 {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self {
            hash0: self.hash0 | rhs.hash0,
            hash1: self.hash1 | rhs.hash1,
        }
    }
}

impl BitOrAssign for Hash128 {
    fn bitor_assign(&mut self, rhs: Self) {
        self.hash0 |= rhs.hash0;
        self.hash1 |= rhs.hash1;
    }
}

impl BitAnd for Hash128 {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        Self {
            hash0: self.hash0 & rhs.hash0,
            hash1: self.hash1 & rhs.hash1,
        }
    }
}

impl BitAndAssign for Hash128 {
    fn bitand_assign(&mut self, rhs: Self) {
        self.hash0 &= rhs.hash0;
        self.hash1 &= rhs.hash1;
    }
}

/// Simple string hash (FNV-1a variant).
pub fn simple_hash_str(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Combine two `u32` values into a `u64`.
pub fn combine(hi: u32, lo: u32) -> u64 {
    (u64::from(hi) << 32) | u64::from(lo)
}

/// Extract high 32 bits.
pub fn high_bits(x: u64) -> u32 {
    (x >> 32) as u32
}

/// Extract low 32 bits.
pub fn low_bits(x: u64) -> u32 {
    x as u32
}

/// A simple 64-bit linear congruential generator step.
pub fn basic_l_cong(x: u64) -> u64 {
    x.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

/// An alternate LCG step.
pub fn basic_l_cong2(x: u64) -> u64 {
    x.wrapping_mul(2862933555777941757).wrapping_add(3037000493)
}

/// MurmurHash 64-bit finalizer.
pub fn murmur_mix(x: u64) -> u64 {
    let mut x = x;
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

/// SplitMix64 next function.
pub fn split_mix64(x: u64) -> u64 {
    let mut x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// rrmxmx bit mixer.
pub fn rrmxmx(x: u64) -> u64 {
    let mut x = x;
    x ^= x.rotate_right(49) ^ x.rotate_right(24);
    x = x.wrapping_mul(0x9fb21c651e98df25);
    x ^= x >> 28;
    x = x.wrapping_mul(0x9fb21c651e98df25);
    x ^ x >> 28
}

/// NASAM bit mixer.
pub fn nasam(x: u64) -> u64 {
    let mut x = x;
    x ^= x >> 23;
    x = x.wrapping_mul(0x2127599bf4325c37);
    x ^= x >> 47;
    x
}

/// Jenkins mix on three 32-bit values.
pub fn jenkins_mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *a = a.wrapping_sub(*c);
    *a ^= c.rotate_left(4);
    *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a);
    *b ^= a.rotate_left(6);
    *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b);
    *c ^= b.rotate_left(8);
    *b = b.wrapping_add(*a);
    *a = a.wrapping_sub(*c);
    *a ^= c.rotate_left(16);
    *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a);
    *b ^= a.rotate_left(19);
    *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b);
    *c ^= b.rotate_left(4);
    *b = b.wrapping_add(*a);
}

/// Single-call Jenkins mix.
pub fn jenkins_mix_single(mut a: u32, mut b: u32, mut c: u32) -> u32 {
    jenkins_mix(&mut a, &mut b, &mut c);
    c
}

/// Return a float in `[0, 1)` deterministically based on a seeded hash.
pub fn seeded_hash_float(str: &str, seed: &str) -> f64 {
    let h1 = simple_hash_str(str);
    let h2 = simple_hash_str(seed);
    let combined = h1 ^ split_mix64(h2);
    // Treat top 53 bits as a mantissa for a double in [0,1).
    let mantissa = combined >> 11;
    mantissa as f64 / ((1u64 << 53) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash128_display() {
        let h = Hash128::new(0x1234, 0x5678);
        assert_eq!(format!("{}", h), "00000000000056780000000000001234");
    }

    #[test]
    fn test_hash128_roundtrip() {
        let original = Hash128::new(0xDEADBEEFCAFEBABE, 0x1122334455667788);
        let s = format!("{}", original);
        let parsed = Hash128::from_hex_string(&s).unwrap();
        assert_eq!(original, parsed);
    }

    #[test]
    fn test_bitwise_ops() {
        let a = Hash128::new(0xFFFF, 0);
        let b = Hash128::new(0x0F0F, 0);
        assert_eq!((a ^ b).hash0, 0xF0F0);
        assert_eq!((a & b).hash0, 0x0F0F);
        assert_eq!((a | b).hash0, 0xFFFF);
    }

    #[test]
    fn test_ordering() {
        let a = Hash128::new(0, 0);
        let b = Hash128::new(0, 1);
        let c = Hash128::new(1, 0);
        assert!(a < b);
        assert!(a < c);
        assert!(c < b);
    }
}
