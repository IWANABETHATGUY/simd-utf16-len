//! SIMD-accelerated UTF-16 length calculation from UTF-8 bytes.
//!
//! Formula: `utf16_len = byte_length - continuation_bytes + four_byte_leaders`
//!
//! Where:
//! - continuation bytes: `(byte & 0xC0) == 0x80`
//! - four-byte leaders: `byte >= 0xF0`

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
mod ascii;

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm32;

#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
)))]
mod scalar;

/// Whether a `lanes`-byte load at `ptr` may read past the end of the input:
/// true when it stays within one 4 KiB page, so it can't fault. This saves
/// short inputs a copy into a zeroed buffer, as in napi-rs/json-escape-simd.
/// Debug builds and Miri always copy, since they would flag the read.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
fn can_overread(ptr: *const u8, lanes: usize) -> bool {
    cfg!(all(
        any(target_os = "linux", target_os = "macos"),
        not(debug_assertions),
        not(miri)
    )) && (ptr as usize & 4095) + lanes <= 4096
}

#[cfg(target_arch = "x86_64")]
pub use x86_64::utf16_len;

#[cfg(target_arch = "aarch64")]
pub use aarch64::utf16_len;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub use wasm32::utf16_len;

#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
)))]
pub use scalar::utf16_len;

/// The kernels behind `utf16_len`, for this crate's tests and benchmarks.
/// Not part of the public API.
#[doc(hidden)]
pub mod __kernels {
    /// One kernel, including the ASCII prefix scan that runs before it.
    pub struct Kernel {
        pub name: &'static str,
        pub utf16_len: fn(&str) -> usize,
    }

    /// Every kernel this CPU supports. `utf16_len` runs the last one.
    pub fn available() -> Vec<Kernel> {
        #[cfg(target_arch = "x86_64")]
        {
            crate::x86_64::kernels()
        }
        #[cfg(target_arch = "aarch64")]
        {
            vec![Kernel {
                name: "neon",
                utf16_len: crate::aarch64::utf16_len,
            }]
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            vec![Kernel {
                name: "simd128",
                utf16_len: crate::wasm32::utf16_len,
            }]
        }
        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128"),
        )))]
        {
            vec![Kernel {
                name: "scalar",
                utf16_len: crate::scalar::utf16_len,
            }]
        }
    }
}

#[cfg(test)]
mod tests {
    /// `crate::utf16_len`, after checking that every kernel this CPU supports
    /// agrees with it, so each test below covers all of them.
    #[track_caller]
    fn utf16_len(s: &str) -> usize {
        let result = super::utf16_len(s);
        for kernel in super::__kernels::available() {
            assert_eq!(
                (kernel.utf16_len)(s),
                result,
                "{} kernel disagrees on {} bytes",
                kernel.name,
                s.len()
            );
        }
        result
    }

    // CI sets this where a kernel must run, so an emulator or runner that hides
    // a CPU feature fails here instead of silently skipping that kernel.
    #[test]
    fn expected_kernels_are_available() {
        let Ok(expected) = std::env::var("SIMD_UTF16_LEN_EXPECT_KERNELS") else {
            return;
        };
        let available: Vec<_> = super::__kernels::available()
            .iter()
            .map(|kernel| kernel.name)
            .collect();
        for name in expected.split(',') {
            assert!(
                available.contains(&name),
                "{name} kernel is not available; found {available:?}"
            );
        }
    }

    /// Reference implementation using the standard library.
    fn reference(s: &str) -> usize {
        s.encode_utf16().count()
    }

    #[test]
    fn empty() {
        assert_eq!(utf16_len(""), reference(""));
    }

    #[test]
    fn ascii_only() {
        assert_eq!(utf16_len("hello"), reference("hello"));
        // Include both ends of the ASCII range and unaligned slice starts.
        let bytes: Vec<u8> = (0..272).map(|i| (i % 128) as u8).collect();
        let input = String::from_utf8(bytes).unwrap();
        for offset in 0..16 {
            for len in 0..=256 {
                let s = &input[offset..offset + len];
                assert_eq!(utf16_len(s), len, "offset: {offset}, len: {len}");
            }
        }
    }

    #[test]
    fn two_byte_chars() {
        // Latin, Cyrillic, etc.
        let s = "café résumé";
        assert_eq!(utf16_len(s), reference(s));
    }

    #[test]
    fn three_byte_chars() {
        // CJK characters (U+4E00..U+9FFF)
        let s = "你好世界";
        assert_eq!(utf16_len(s), reference(s));
    }

    #[test]
    fn four_byte_chars() {
        // Emoji / supplementary plane (surrogate pairs in UTF-16)
        let s = "😀🎉🚀💯";
        assert_eq!(utf16_len(s), reference(s));
    }

    #[test]
    fn mixed() {
        let s = "Hello, 世界! 🌍🌎🌏 café";
        assert_eq!(utf16_len(s), reference(s));
    }

    #[test]
    fn single_char_boundaries() {
        // One character of each UTF-8 width
        for c in ['a', 'é', '中', '🦀'] {
            let s = String::from(c);
            assert_eq!(utf16_len(&s), reference(&s), "char: {c}");
        }
    }

    #[test]
    fn longer_than_simd_width() {
        // Ensure the SIMD loop and scalar tail both work (> 16 bytes).
        let s = "abcdefghijklmnopqrstuvwxyz";
        assert_eq!(utf16_len(s), reference(s));

        let s = "αβγδεζηθικλμνξοπρστυφχψω";
        assert_eq!(utf16_len(s), reference(s));

        let s = "你好世界你好世界你好世界你好世界";
        assert_eq!(utf16_len(s), reference(s));

        let s = "🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀";
        assert_eq!(utf16_len(s), reference(s));
    }

    #[test]
    fn repeated_pattern_large() {
        // Stress test: exceed the 255-iteration batch boundary (255 * 16 = 4080 bytes).
        let s = "a".repeat(5000);
        assert_eq!(utf16_len(&s), reference(&s));

        let s = "🦀".repeat(1500); // 1500 * 4 = 6000 bytes
        assert_eq!(utf16_len(&s), reference(&s));
    }

    #[test]
    fn all_byte_widths_interleaved() {
        // Repeating pattern of 1+2+3+4 byte chars to test alignment variations.
        let pattern = "aé中🦀";
        let s = pattern.repeat(100);
        assert_eq!(utf16_len(&s), reference(&s));
    }

    #[test]
    fn non_ascii_after_ascii_prefix() {
        for prefix_len in (0..=129).chain([2031, 2032, 2033, 4079, 4080, 4081, 4095, 4096, 4097]) {
            for suffix in [
                "é",
                "中",
                "🦀",
                "é中🦀",
                "\u{7ff}\u{800}\u{ffff}\u{10000}\u{10ffff}",
            ] {
                for tail_len in [0, 1, 15, 16, 63, 64, 65] {
                    // Exercise aligned word loads and overlapping tails from
                    // every possible 16-byte slice alignment.
                    for offset in 0..16 {
                        let storage =
                            "a".repeat(offset + prefix_len) + suffix + &"a".repeat(tail_len);
                        let s = &storage[offset..];
                        assert_eq!(
                            utf16_len(s),
                            reference(s),
                            "offset: {offset}, prefix_len: {prefix_len}, tail_len: {tail_len}, suffix: {suffix}"
                        );
                    }
                }
            }
        }
    }
}
