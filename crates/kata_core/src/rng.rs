//! Deterministic random number generator.
//!
//! Corresponds to `cpp/core/rand.h` and `cpp/core/rand.cpp`.
//! Combines a PCG32 and an XorShift1024* generator, matching the original
//! seeding, integer, floating point, and shuffle APIs.

use crate::global;
use crate::hash::md5::md5;
use crate::hash::sha2::{sha256, sha256_hex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const XORMULT_LEN: usize = 16;
const XORMULT_MASK: usize = XORMULT_LEN - 1;

static INIT_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// xorshift1024* generator (period 2^1024 - 1).
struct XorShift1024Mult {
    a: [u64; XORMULT_LEN],
    a_idx: usize,
}

impl XorShift1024Mult {
    fn new(init_a: &[u64; XORMULT_LEN]) -> Self {
        Self {
            a: *init_a,
            a_idx: 0,
        }
    }

    fn next_u32(&mut self) -> u32 {
        let a0 = self.a[self.a_idx];
        self.a_idx = (self.a_idx + 1) & XORMULT_MASK;
        let a1 = self.a[self.a_idx];
        let a1 = a1 ^ (a1 << 31);
        let a1 = a1 ^ (a1 >> 11);
        let a0 = a0 ^ (a0 >> 30);
        self.a[self.a_idx] = a0 ^ a1;
        let result = self.a[self.a_idx].wrapping_mul(1181783497276652981u64);
        (result >> 32) as u32
    }
}

/// PCG32 generator (period 2^64).
struct Pcg32 {
    s: u64,
}

impl Pcg32 {
    fn new(state: u64) -> Self {
        Self { s: state }
    }

    fn next_u32(&mut self) -> u32 {
        self.s = self
            .s
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(1442695040888963407u64);
        let x = (((self.s >> 18) ^ self.s) >> 27) as u32;
        let rot = (self.s >> 59) as u32;
        if rot == 0 { x } else { x.rotate_right(rot) }
    }
}

/// KataGo's combined random number generator.
pub struct Rand {
    xorm: XorShift1024Mult,
    pcg32: Pcg32,
    has_gaussian: bool,
    stored_gaussian: f64,
    init_seed: String,
    num_calls: u64,
}

impl Default for Rand {
    fn default() -> Self {
        Self::new()
    }
}

impl Rand {
    /// Initialize with a seed derived from system entropy and unique identifiers.
    pub fn new() -> Self {
        let mut r = Self {
            xorm: XorShift1024Mult::new(&[0; XORMULT_LEN]),
            pcg32: Pcg32::new(0),
            has_gaussian: false,
            stored_gaussian: 0.0,
            init_seed: String::new(),
            num_calls: 0,
        };
        r.init();
        r
    }

    /// Initialize with the provided string seed.
    pub fn new_from_seed(seed: &str) -> Self {
        let mut r = Self {
            xorm: XorShift1024Mult::new(&[0; XORMULT_LEN]),
            pcg32: Pcg32::new(0),
            has_gaussian: false,
            stored_gaussian: 0.0,
            init_seed: String::new(),
            num_calls: 0,
        };
        r.init_from_seed(seed);
        r
    }

    /// Initialize with the provided integer seed.
    pub fn new_from_u64(seed: u64) -> Self {
        Self::new_from_seed(&global::uint64_to_hex_string(seed))
    }

    /// Reinitialize from system entropy.
    pub fn init(&mut self) {
        let x = INIT_COUNTER.fetch_add(1, Ordering::SeqCst);

        let time0 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let precision_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);

        let mut s = String::new();
        s.push_str(&global::int_to_string(x as i32));
        s.push_str(&global::uint64_to_hex_string(time0));
        s.push_str(&global::int64_to_string(precision_time));

        s.push('|');
        s.push_str(&global::int64_to_string(std::process::id() as i64));
        s.push('|');
        let hostname = std::env::var("COMPUTERNAME")
            .or_else(|_| std::env::var("HOSTNAME"))
            .unwrap_or_default();
        s.push_str(&hostname);

        s.push('|');
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        std::thread::current().id().hash(&mut hasher);
        s.push_str(&global::uint64_to_hex_string(hasher.finish()));

        {
            let stack_val = 0usize;
            let heap_val = Box::new(0usize);
            s.push('|');
            s.push_str(&global::uint64_to_hex_string(
                &stack_val as *const _ as usize as u64,
            ));
            s.push_str(&global::uint64_to_hex_string(
                &*heap_val as *const _ as usize as u64,
            ));
        }

        let hash = sha256_hex(s.as_bytes());
        self.init_from_seed(&hash);
    }

    /// Reinitialize from a string seed.
    pub fn init_from_seed(&mut self, seed: &str) {
        self.init_seed = seed.to_string();

        let md5_hash = md5(seed.as_bytes());
        let mut s = String::new();
        s.push('|');
        s.push_str(&md5_hash[0].to_string());
        s.push('|');
        s.push_str(seed);

        let mut counter: usize = 0;
        let mut next_hash_idx = 4;
        let mut hash = [0u64; 4];

        let mut next_value = || -> u64 {
            loop {
                if next_hash_idx >= 4 {
                    let tmp = format!("{}{}", counter, s);
                    counter = counter.wrapping_add(37);
                    let digest = sha256(tmp.as_bytes());
                    for i in 0..4 {
                        hash[i] = u64::from_be_bytes([
                            digest[i * 8],
                            digest[i * 8 + 1],
                            digest[i * 8 + 2],
                            digest[i * 8 + 3],
                            digest[i * 8 + 4],
                            digest[i * 8 + 5],
                            digest[i * 8 + 6],
                            digest[i * 8 + 7],
                        ]);
                    }
                    next_hash_idx = 0;
                }
                let v = hash[next_hash_idx];
                next_hash_idx += 1;
                if v != 0 {
                    return v;
                }
            }
        };

        let mut init_a = [0u64; XORMULT_LEN];
        for item in init_a.iter_mut() {
            *item = next_value();
        }
        self.xorm = XorShift1024Mult::new(&init_a);
        self.pcg32 = Pcg32::new(next_value());
        self.has_gaussian = false;
        self.stored_gaussian = 0.0;
        self.num_calls = 0;
    }

    /// Reinitialize from an integer seed.
    pub fn init_from_u64(&mut self, seed: u64) {
        self.init_from_seed(&global::uint64_to_hex_string(seed));
    }

    /// Return the seed used for initialization.
    pub fn seed(&self) -> &str {
        &self.init_seed
    }

    /// Return the number of calls made to the generator.
    pub fn num_calls(&self) -> u64 {
        self.num_calls
    }

    /// Random `u32` in [0, 2^32).
    pub fn next_u32(&mut self) -> u32 {
        self.num_calls += 1;
        self.pcg32.next_u32().wrapping_add(self.xorm.next_u32())
    }

    /// Random `u32` in [0, n).
    pub fn next_u32_bounded(&mut self, n: u32) -> u32 {
        assert!(n > 0);
        loop {
            let bits = self.next_u32();
            let val = bits % n;
            if bits.wrapping_sub(val).wrapping_add(n - 1) >= bits.wrapping_sub(val) {
                return val;
            }
        }
    }

    /// Random index according to the given integer frequency distribution.
    pub fn next_u32_from_freqs(&mut self, freqs: &[i32]) -> u32 {
        assert!(!freqs.is_empty());
        let mut sum: i64 = 0;
        for &f in freqs {
            assert!(f >= 0);
            sum += i64::from(f);
        }
        assert!(sum > 0);
        let mut r = self.next_u64_bounded(sum as u64) as i64;
        for (i, &f) in freqs.iter().enumerate() {
            r -= i64::from(f);
            if r < 0 {
                return i as u32;
            }
        }
        freqs.len() as u32 - 1
    }

    /// Random index according to the given unnormalized probability distribution.
    pub fn next_u32_from_probs(&mut self, rel_probs: &[f64]) -> u32 {
        assert!(!rel_probs.is_empty());
        let mut sum = 0.0;
        for &p in rel_probs {
            assert!(p >= 0.0);
            sum += p;
        }
        assert!(sum > 0.0);
        let d = self.next_double_up_to(sum);
        let mut acc = 0.0;
        for (i, &p) in rel_probs.iter().enumerate() {
            acc += p;
            if acc > d {
                return i as u32;
            }
        }
        rel_probs.len() as u32 - 1
    }

    /// Random index according to a cumulative probability distribution.
    pub fn next_index_cumulative(&mut self, cum_rel_probs: &[f64]) -> usize {
        assert!(!cum_rel_probs.is_empty());
        let sum = cum_rel_probs[cum_rel_probs.len() - 1];
        let d = self.next_double_up_to(sum);
        match cum_rel_probs.binary_search_by(|v| v.partial_cmp(&d).unwrap()) {
            Ok(i) => i,
            Err(i) => i.min(cum_rel_probs.len() - 1),
        }
    }

    /// Random `i32` in [-2^31, 2^31).
    pub fn next_i32(&mut self) -> i32 {
        self.next_u32() as i32
    }

    /// Random `i32` in [a, b].
    pub fn next_i32_range(&mut self, a: i32, b: i32) -> i32 {
        assert!(b >= a);
        let max = (b as u32).wrapping_sub(a as u32).wrapping_add(1);
        if max == 0 {
            self.next_i32()
        } else {
            (self.next_u32_bounded(max).wrapping_add(a as u32)) as i32
        }
    }

    /// Random `u64` in [0, 2^64).
    pub fn next_u64(&mut self) -> u64 {
        let lower = self.next_u32() as u64;
        let upper = (self.next_u32() as u64) << 32;
        lower | upper
    }

    /// Random `u64` in [0, n).
    pub fn next_u64_bounded(&mut self, n: u64) -> u64 {
        assert!(n > 0);
        loop {
            let bits = self.next_u64();
            let val = bits % n;
            if bits.wrapping_sub(val).wrapping_add(n - 1) >= bits.wrapping_sub(val) {
                return val;
            }
        }
    }

    /// Return true with probability `prob`.
    pub fn next_bool(&mut self, prob: f64) -> bool {
        self.next_double() < prob
    }

    /// Random `f64` in [0, 1).
    pub fn next_double(&mut self) -> f64 {
        loop {
            let bits = self.next_u64() & ((1u64 << 53) - 1);
            let x = bits as f64 / (1u64 << 53) as f64;
            if (0.0..1.0).contains(&x) {
                return x;
            }
        }
    }

    /// Random `f64` in [0, n).
    pub fn next_double_up_to(&mut self, n: f64) -> f64 {
        assert!(n >= 0.0);
        self.next_double() * n
    }

    /// Random `f64` in [a, b).
    pub fn next_double_range(&mut self, a: f64, b: f64) -> f64 {
        assert!(b >= a);
        a + self.next_double_up_to(b - a)
    }

    /// Standard normal variate.
    pub fn next_gaussian(&mut self) -> f64 {
        if self.has_gaussian {
            self.has_gaussian = false;
            return self.stored_gaussian;
        }
        loop {
            let v1 = self.next_double() * 2.0 - 1.0;
            let v2 = self.next_double() * 2.0 - 1.0;
            let s = v1 * v1 + v2 * v2;
            if s < 1.0 && s != 0.0 {
                let multiplier = (-2.0 * s.ln() / s).sqrt();
                self.stored_gaussian = v2 * multiplier;
                self.has_gaussian = true;
                return v1 * multiplier;
            }
        }
    }

    /// Standard normal variate, redrawn until it falls within [-bound, bound].
    pub fn next_gaussian_truncated(&mut self, bound: f64) -> f64 {
        assert!(bound >= 0.1);
        let mut d = self.next_gaussian();
        while d < -bound || d > bound {
            d = self.next_gaussian();
        }
        d
    }

    /// Exponential variate with mean 1.
    pub fn next_exponential(&mut self) -> f64 {
        let mut r = 0.0;
        while r <= 1e-17 {
            r = self.next_double();
        }
        -r.ln()
    }

    /// Logistic variate with mean 0 and scale 1.
    pub fn next_logistic(&mut self) -> f64 {
        let mut num = 0.0;
        let mut denom = 0.0;
        while num <= 0.0 || denom <= 0.0 {
            num = self.next_double();
            denom = 1.0 - num;
        }
        (num / denom).ln()
    }

    /// Gamma variate with shape `a` and scale 1.
    pub fn next_gamma(&mut self, a: f64) -> f64 {
        assert!(a > 0.0, "Rand::next_gamma: invalid value for a: {}", a);
        if a <= 1.0 {
            let r = self.next_gamma(a + 1.0);
            let inva = 1.0 / a;
            let scale = if inva == 0.0 {
                1.0
            } else {
                self.next_double().powf(inva)
            };
            return r * scale;
        }

        let d = a - 1.0 / 3.0;
        let c = (1.0 / 3.0) / d.sqrt();
        loop {
            let x = self.next_gaussian();
            let vtmp = 1.0 + c * x;
            if vtmp <= 0.0 {
                continue;
            }
            let v = vtmp * vtmp * vtmp;
            let u = self.next_double();
            let xx = x * x;
            if u < 1.0 - 0.0331 * xx * xx {
                return d * v;
            }
            if u == 0.0 || u.ln() < 0.5 * xx + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }

    /// Fill `buf` with a random permutation of 0..n.
    pub fn fill_shuffled_u32_range(&mut self, n: u32, buf: &mut [u32]) {
        assert!((n as usize) <= buf.len());
        for i in 0..n {
            buf[i as usize] = i;
        }
        for i in 1..n {
            let r = self.next_u64_bounded((i + 1) as u64) as u32;
            buf.swap(i as usize, r as usize);
        }
    }

    /// Shuffle a slice in place.
    pub fn shuffle<T>(&mut self, vec: &mut [T]) {
        for i in 1..vec.len() {
            let r = self.next_u64_bounded((i + 1) as u64) as usize;
            vec.swap(i, r);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pcg32_only() {
        let mut pcg = Pcg32::new(123);
        let expected: [u32; 16] = [
            0xb376_6cbd,
            0x65fd_d305,
            0x2a3b_9b9c,
            0x09a2_dee9,
            0x1a86_aabc,
            0x36a9_8234,
            0x82e6_e2b4,
            0x10c0_77e5,
            0x2975_5fc7,
            0xf7fa_7b5c,
            0x1cb7_ae7d,
            0xcce0_e3d9,
            0x065e_c08b,
            0x505d_1cdb,
            0x8b77_8f3c,
            0xdb72_f217,
        ];
        for (i, &exp) in expected.iter().enumerate() {
            let got = pcg.next_u32();
            assert_eq!(
                got, exp,
                "pcg mismatch at {}: got {:08x}, expected {:08x}",
                i, got, exp
            );
        }
    }

    #[test]
    fn test_xorshift_only() {
        let init_a: [u64; 16] = [
            15148282349006049087,
            3601266951833665894,
            16929445066801446424,
            13475938501103070154,
            15713138009143754412,
            4148159782736716337,
            16035594834001032141,
            5555591070439871209,
            4101130512537511022,
            12821547636792886909,
            9050874162294428797,
            6187760405891629771,
            10053646276519763308,
            2219782655280501359,
            3719698449347562208,
            5421263376768154227,
        ];
        let mut xorm = XorShift1024Mult::new(&init_a);
        let expected: [u32; 32] = [
            0x7497_46d1,
            0x9242_ca14,
            0x98db_98a1,
            0x1348_e491,
            0xde60_e668,
            0x77e3_7a69,
            0xeb51_a9d3,
            0xd44b_4727,
            0x3418_95b0,
            0xc7b1_b3f4,
            0xe7ef_0529,
            0x8e72_ea7e,
            0x5855_da19,
            0xfffc_d2b2,
            0xa684_e430,
            0xb76a_7e0d,
            0x5af3_820e,
            0x320b_0699,
            0xdbb8_5ee0,
            0xc1dc_d25c,
            0x4b39_5e3e,
            0x4007_756f,
            0x76a0_c667,
            0xaa60_41f6,
            0x756f_94bb,
            0x3952_7d1b,
            0x6e12_32ef,
            0xb302_7668,
            0x776e_a832,
            0x35a0_ed1b,
            0x1f2f_0268,
            0xadd5_9669,
        ];
        for (i, &exp) in expected.iter().enumerate() {
            let got = xorm.next_u32();
            assert_eq!(
                got, exp,
                "xorm mismatch at {}: got {:08x}, expected {:08x}",
                i, got, exp
            );
        }
    }

    #[test]
    fn test_simple_next_uint() {
        let mut rand = Rand::new_from_seed("abc");
        let expected: [u32; 24] = [
            0x1C6B_83BD,
            0xFB76_77DB,
            0x6986_88D5,
            0xA3CD_21C3,
            0xD0AD_5B77,
            0x8F88_9E6E,
            0x2285_2278,
            0xD71A_114D,
            0x295E_F301,
            0xAA0C_CA48,
            0x0B72_71BB,
            0x4FE7_98FB,
            0x26B4_DD4B,
            0x78B7_7C1B,
            0x231C_4DFB,
            0x17FB_87C6,
            0x9CC2_3870,
            0x1C2C_2CF7,
            0x62D5_1240,
            0xF1D1_A7FF,
            0x44C4_5C0A,
            0xF93A_CFCE,
            0x42B1_D236,
            0xC106_9B75,
        ];
        for (i, &exp) in expected.iter().enumerate() {
            let got = rand.next_u32();
            assert_eq!(
                got, exp,
                "next_u32 mismatch at index {}: got {:08x}, expected {:08x}",
                i, got, exp
            );
        }
    }

    #[test]
    fn test_next_u32_bounded() {
        let mut rand = Rand::new_from_seed("abc");
        let expected: [u32; 16] = [5, 26, 0, 17, 3, 10, 24, 11, 16, 0, 26, 17, 25, 16, 16, 25];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(rand.next_u32_bounded(27), exp, "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_next_i32() {
        let mut rand = Rand::new_from_seed("abc");
        let expected: [i32; 16] = [
            476_808_125,
            -76_122_149,
            1_770_424_533,
            -1_546_837_565,
            -793_945_225,
            -1_886_871_954,
            579_150_456,
            -686_157_491,
            694_088_449,
            -1_442_002_360,
            192_049_595,
            1_340_578_043,
            649_387_339,
            2_025_290_779,
            589_057_531,
            402_360_262,
        ];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(rand.next_i32(), exp, "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_next_double() {
        let mut rand = Rand::new_from_seed("abc");
        let expected: [f64; 16] = [
            0.702131, 0.410372, 0.26934, 0.814612, 0.399693, 0.237424, 0.7339, 0.860324, 0.380489,
            0.551758, 0.837867, 0.206477, 0.935634, 0.657769, 0.1208, 0.142558,
        ];
        for (i, &exp) in expected.iter().enumerate() {
            let got = (rand.next_double() * 1_000_000.0).round() / 1_000_000.0;
            assert!(
                (got - exp).abs() < 1e-6,
                "next_double mismatch at index {}: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    fn test_fill_shuffled_u32_range() {
        let mut rand = Rand::new_from_seed("shuffletest!");
        let mut buf = [99u32; 32];
        rand.fill_shuffled_u32_range(16, &mut buf[8..]);
        let expected: [u32; 16] = [3, 7, 13, 14, 4, 6, 2, 0, 8, 11, 10, 1, 5, 9, 15, 12];
        assert_eq!(buf[8..24], expected);
    }
}
