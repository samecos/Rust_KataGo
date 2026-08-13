//! Base64 encoding and decoding.
//!
//! Corresponds to `cpp/core/base64.h` and `cpp/core/base64.cpp`.

use crate::global::StringError;

const BASE64_CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

const DECODE_TABLE: [i8; 128] = [
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, -1, -1, 63,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1, -1, -1, -1, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, -1, -1, -1, -1, -1,
];

/// Encode a byte slice as standard Base64.
pub fn encode(input: &[u8]) -> String {
    let size = input.len();
    let mut result = String::with_capacity(size.div_ceil(3) * 4);
    let mut i = 0;
    while i < size {
        let c0 = input[i];
        let c1 = if i + 1 < size { input[i + 1] } else { 0 };
        let c2 = if i + 2 < size { input[i + 2] } else { 0 };

        let e0 = (c0 >> 2) as usize;
        let e1 = (((c0 & 0x3) << 4) | (c1 >> 4)) as usize;
        let e2 = (((c1 & 0xf) << 2) | (c2 >> 6)) as usize;
        let e3 = (c2 & 0x3f) as usize;

        result.push(BASE64_CHARS[e0] as char);
        result.push(BASE64_CHARS[e1] as char);
        result.push(BASE64_CHARS[e2] as char);
        result.push(BASE64_CHARS[e3] as char);
        i += 3;
    }

    let excess = size - (size / 3 * 3);
    if excess == 2 {
        result.pop();
        result.push('=');
    } else if excess == 1 {
        result.pop();
        result.pop();
        result.push_str("==");
    }
    result
}

/// Decode a standard Base64 string into a byte vector.
pub fn decode(input: &str) -> Result<Vec<u8>, StringError> {
    let size = input.len();
    let mut result = Vec::with_capacity(size.div_ceil(4) * 3);

    let mut carry_num_bits = 0i32;
    let mut carry = 0i32;
    let mut i = 0;
    for c in input.bytes() {
        if c == b'=' {
            break;
        }
        if !(b'+'..=b'z').contains(&c) {
            return Err(StringError::new(format!(
                "Base64::decode: invalid character {}",
                c as char
            )));
        }
        let d = DECODE_TABLE[c as usize];
        if d < 0 {
            return Err(StringError::new(format!(
                "Base64::decode: invalid character {}",
                c as char
            )));
        }

        carry_num_bits += 6;
        carry = (carry << 6) | i32::from(d);
        if carry_num_bits >= 8 {
            let extracted = carry >> (carry_num_bits - 8);
            result.push(extracted as u8);
            carry ^= extracted << (carry_num_bits - 8);
            carry_num_bits -= 8;
        }
        i += 1;
    }

    for c in input.bytes().skip(i) {
        if c != b'=' {
            return Err(StringError::new(
                "Base64::decode: string contains other characters after '='".to_string(),
            ));
        }
    }

    if carry != 0 {
        return Err(StringError::new(
            "Base64::decode: unexpected end of decode, carry is nonzero".to_string(),
        ));
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn safe_print(s: &[u8]) -> String {
        let mut out = String::new();
        for &b in s {
            if b.is_ascii_graphic() || b == b' ' {
                out.push(b as char);
            } else {
                // Match C++ `(int)(char)b`, which is signed for values >= 0x80.
                out.push_str(&format!("({})", b as i8));
            }
        }
        out
    }

    fn run_test(s: &[u8]) -> String {
        let encoded = encode(s);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, s);
        format!("{} : {}\n", safe_print(s), encoded)
    }

    fn run_decode(s: &str) -> String {
        match decode(s) {
            Ok(decoded) => {
                let encoded = encode(&decoded);
                let decoded2 = decode(&encoded).unwrap();
                assert_eq!(decoded, decoded2);
                format!(
                    "{} -> {}\n",
                    safe_print(&s.bytes().collect::<Vec<_>>()),
                    safe_print(&decoded)
                )
            }
            Err(e) => format!(
                "{} error: {}\n",
                safe_print(&s.bytes().collect::<Vec<_>>()),
                e.message
            ),
        }
    }

    #[test]
    fn test_base64() {
        let mut out = String::new();
        out.push_str(&run_test(b""));
        out.push_str(&run_test(b"pleasure."));
        out.push_str(&run_test(b"leasure."));
        out.push_str(&run_test(b"easure."));
        out.push_str(&run_test(b"asure."));
        out.push_str(&run_test(b"sure."));
        out.push_str(&run_test(b"ure."));
        out.push_str(&run_test(b"re."));
        out.push_str(&run_test(b"e."));
        out.push_str(&run_test(b"."));
        out.push_str(&run_test(&[0xFF]));
        out.push_str(&run_test(&[0xFF, 0x01]));
        out.push_str(&run_test(&[0xFF, 0x01, 0xFF]));
        out.push_str(&run_test(&[0xFF, 0x01, 0xFF, 0xFF]));
        out.push_str(&run_test(&[0xFF, 0x01, 0xFF, 0xFF, 0x01]));
        out.push_str(&run_test(&[0xFF, 0x01, 0xFF, 0xFF, 0x01, 0xFF]));
        out.push_str(&run_test(&[0xFF, 0x01, 0xFF, 0xFF, 0x01, 0xFF, 0xFF]));
        out.push_str(&run_test(&[0xFF, 0x01, 0xFF, 0xFF, 0x01, 0xFF, 0xFF, 0xFF]));
        out.push_str(&run_test(&[
            0xFF, 0x01, 0xFF, 0xFF, 0x01, 0xFF, 0xFF, 0xFF, 0x01,
        ]));
        out.push_str(&run_test(&[0, 100, 0, 100]));

        out.push_str(&run_decode("YWJjZGVm"));
        out.push_str(&run_decode("YWJjZGVm="));
        out.push_str(&run_decode("YWJjZGVm=="));
        out.push_str(&run_decode("YWJjZGVm==="));
        out.push_str(&run_decode("YWJjZGU"));
        out.push_str(&run_decode("YWJjZGU="));
        out.push_str(&run_decode("YWJjZGU=="));
        out.push_str(&run_decode("YWJjZGU==="));
        out.push_str(&run_decode("YWJjZA"));
        out.push_str(&run_decode("YWJjZA="));
        out.push_str(&run_decode("YWJjZA=="));
        out.push_str(&run_decode("YWJjZA==="));

        let lorem = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";
        out.push_str(&run_test(lorem.as_bytes()));

        out.push_str(&run_decode("YWJjZB="));
        out.push_str(&run_decode("YWJjZC="));
        out.push_str(&run_decode("YWJjZD="));
        out.push_str(&run_decode("YWJjZE="));
        out.push_str(&run_decode("YWJjZF="));
        out.push_str(&run_decode("YWJjZG="));
        out.push_str(&run_decode("YWJjZH="));
        out.push_str(&run_decode("YWJjZI="));
        out.push_str(&run_decode("YWJjZJ="));
        out.push_str(&run_decode("YWJjZK="));
        out.push_str(&run_decode("YWJjZL="));
        out.push_str(&run_decode("YWJjZM="));
        out.push_str(&run_decode("YWJjZN="));
        out.push_str(&run_decode("YWJjZO="));
        out.push_str(&run_decode("YWJjZP="));
        out.push_str(&run_decode("YWJjZQ="));
        out.push_str(&run_decode("YWJjZR="));

        out.push_str(&run_decode("YWJj!"));
        out.push_str(&run_decode("YWJjZGVm\n"));

        let expected = r" : 
pleasure. : cGxlYXN1cmUu
leasure. : bGVhc3VyZS4=
easure. : ZWFzdXJlLg==
asure. : YXN1cmUu
sure. : c3VyZS4=
ure. : dXJlLg==
re. : cmUu
e. : ZS4=
. : Lg==
(-1) : /w==
(-1)(1) : /wE=
(-1)(1)(-1) : /wH/
(-1)(1)(-1)(-1) : /wH//w==
(-1)(1)(-1)(-1)(1) : /wH//wE=
(-1)(1)(-1)(-1)(1)(-1) : /wH//wH/
(-1)(1)(-1)(-1)(1)(-1)(-1) : /wH//wH//w==
(-1)(1)(-1)(-1)(1)(-1)(-1)(-1) : /wH//wH///8=
(-1)(1)(-1)(-1)(1)(-1)(-1)(-1)(1) : /wH//wH///8B
(0)d(0)d : AGQAZA==
YWJjZGVm -> abcdef
YWJjZGVm= -> abcdef
YWJjZGVm== -> abcdef
YWJjZGVm=== -> abcdef
YWJjZGU -> abcde
YWJjZGU= -> abcde
YWJjZGU== -> abcde
YWJjZGU=== -> abcde
YWJjZA -> abcd
YWJjZA= -> abcd
YWJjZA== -> abcd
YWJjZA=== -> abcd
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. : TG9yZW0gaXBzdW0gZG9sb3Igc2l0IGFtZXQsIGNvbnNlY3RldHVyIGFkaXBpc2NpbmcgZWxpdCwgc2VkIGRvIGVpdXNtb2QgdGVtcG9yIGluY2lkaWR1bnQgdXQgbGFib3JlIGV0IGRvbG9yZSBtYWduYSBhbGlxdWEuIFV0IGVuaW0gYWQgbWluaW0gdmVuaWFtLCBxdWlzIG5vc3RydWQgZXhlcmNpdGF0aW9uIHVsbGFtY28gbGFib3JpcyBuaXNpIHV0IGFsaXF1aXAgZXggZWEgY29tbW9kbyBjb25zZXF1YXQuIER1aXMgYXV0ZSBpcnVyZSBkb2xvciBpbiByZXByZWhlbmRlcml0IGluIHZvbHVwdGF0ZSB2ZWxpdCBlc3NlIGNpbGx1bSBkb2xvcmUgZXUgZnVnaWF0IG51bGxhIHBhcmlhdHVyLiBFeGNlcHRldXIgc2ludCBvY2NhZWNhdCBjdXBpZGF0YXQgbm9uIHByb2lkZW50LCBzdW50IGluIGN1bHBhIHF1aSBvZmZpY2lhIGRlc2VydW50IG1vbGxpdCBhbmltIGlkIGVzdCBsYWJvcnVtLg==
YWJjZB= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZC= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZD= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZE= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZF= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZG= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZH= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZI= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZJ= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZK= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZL= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZM= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZN= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZO= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZP= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJjZQ= -> abce
YWJjZR= error: Base64::decode: unexpected end of decode, carry is nonzero
YWJj! error: Base64::decode: invalid character !
YWJjZGVm(10) error: Base64::decode: invalid character
";
        crate::test::expect_lines_match("base64 tests", &out, expected);
    }
}
