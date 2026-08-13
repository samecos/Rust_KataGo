//! Minimal MD5 implementation.
//!
//! Corresponds to `cpp/core/md5.h` and `cpp/core/md5.cpp`.
//! Only the core 128-bit digest is implemented, matching the C++ `MD5::get` API.

/// Compute the MD5 digest of `data` and return it as four little-endian `u32` words.
pub fn md5(data: &[u8]) -> [u32; 4] {
    let mut state = [0x6745_2301u32, 0xEFCD_AB89, 0x98BA_DCFE, 0x1032_5476];
    let mut buffer = [0u8; 128];
    let mut buffer_len: usize = 0;
    let mut total_len: u64 = 0;

    for &byte in data {
        buffer[buffer_len] = byte;
        buffer_len += 1;
        total_len += 1;
        if buffer_len == 64 {
            process_block(&mut state, &buffer[..64]);
            buffer_len = 0;
        }
    }

    // Padding.
    let mut padding = [0u8; 64];
    padding[0] = 0x80;
    let bit_len = total_len * 8;
    let pad_len = if buffer_len < 56 {
        56 - buffer_len
    } else {
        120 - buffer_len
    };
    buffer[buffer_len..buffer_len + pad_len].copy_from_slice(&padding[..pad_len]);
    buffer_len += pad_len;

    // Append length in bits as little-endian u64.
    let len_bytes = bit_len.to_le_bytes();
    if buffer_len == 56 {
        buffer[56..64].copy_from_slice(&len_bytes);
        process_block(&mut state, &buffer[..64]);
    } else {
        buffer[120..128].copy_from_slice(&len_bytes);
        process_block(&mut state, &buffer[0..64]);
        process_block(&mut state, &buffer[64..128]);
    }

    state
}

fn process_block(state: &mut [u32; 4], block: &[u8]) {
    let mut w = [0u32; 16];
    for i in 0..16 {
        w[i] = u32::from_le_bytes([
            block[i * 4],
            block[i * 4 + 1],
            block[i * 4 + 2],
            block[i * 4 + 3],
        ]);
    }

    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];

    let s: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let k: [u32; 64] = [
        0xd76a_a478,
        0xe8c7_b756,
        0x2420_70db,
        0xc1bd_ceee,
        0xf57c_0faf,
        0x4787_c62a,
        0xa830_4613,
        0xfd46_9501,
        0x6980_98d8,
        0x8b44_f7af,
        0xffff_5bb1,
        0x895c_d7be,
        0x6b90_1122,
        0xfd98_7193,
        0xa679_438e,
        0x49b4_0821,
        0xf61e_2562,
        0xc040_b340,
        0x265e_5a51,
        0xe9b6_c7aa,
        0xd62f_105d,
        0x0244_1453,
        0xd8a1_e681,
        0xe7d3_fbc8,
        0x21e1_cde6,
        0xc337_07d6,
        0xf4d5_0d87,
        0x455a_14ed,
        0xa9e3_e905,
        0xfcef_a3f8,
        0x676f_02d9,
        0x8d2a_4c8a,
        0xfffa_3942,
        0x8771_f681,
        0x6d9d_6122,
        0xfde5_380c,
        0xa4be_ea44,
        0x4bde_cfa9,
        0xf6bb_4b60,
        0xbebf_bc70,
        0x289b_7ec6,
        0xeaa1_27fa,
        0xd4ef_3085,
        0x0488_1d05,
        0xd9d4_d039,
        0xe6db_99e5,
        0x1fa2_7cf8,
        0xc4ac_5665,
        0xf429_2244,
        0x432a_ff97,
        0xab94_23a7,
        0xfc93_a039,
        0x655b_59c3,
        0x8f0c_cc92,
        0xffef_f47d,
        0x8584_5dd1,
        0x6fa8_7e4f,
        0xfe2c_e6e0,
        0xa301_4314,
        0x4e08_11a1,
        0xf753_7e82,
        0xbd3a_f235,
        0x2ad7_d2bb,
        0xeb86_d391,
    ];

    for i in 0..64 {
        let (f, g): (u32, usize);
        if i < 16 {
            f = (b & c) | ((!b) & d);
            g = i;
        } else if i < 32 {
            f = (d & b) | ((!d) & c);
            g = (5 * i + 1) % 16;
        } else if i < 48 {
            f = b ^ c ^ d;
            g = (3 * i + 5) % 16;
        } else {
            f = c ^ (b | (!d));
            g = (7 * i) % 16;
        }

        let temp = d;
        d = c;
        c = b;
        b = b.wrapping_add(
            a.wrapping_add(f)
                .wrapping_add(k[i])
                .wrapping_add(w[g])
                .rotate_left(s[i]),
        );
        a = temp;
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_md5_empty() {
        assert_eq!(
            md5(b""),
            [0xd98c_1dd4, 0x04b2_008f, 0x9809_80e9, 0x7e42_f8ec]
        );
    }

    #[test]
    fn test_md5_quick_brown_fox() {
        assert_eq!(
            md5(b"The quick brown fox jumps over the lazy dog."),
            [0xc209_d9e4, 0x1cfb_d090, 0xadff_68a0, 0xd0cb_22df]
        );
    }

    #[test]
    fn test_md5_long_input() {
        // Two-block input, verifying padding across block boundaries.
        assert_eq!(
            md5(&[b'a'; 100]),
            [0xc92c_a936, 0xa20f_9e4a, 0x8b5f_621f, 0xdf7a_00fb]
        );
    }
}
