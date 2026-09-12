//! A/B benchmark: the current `utf16_len` vs the previous implementation
//! (vendored from commit 852d444) in the same process, so shared-runner
//! noise cancels out.

use std::hint::black_box;
use std::time::{Duration, Instant};

use simd_utf16_len::utf16_len;

const ASCII: &str = "The quick brown fox jumps over the lazy dog. This is a longer sentence to provide more data for benchmarking purposes, with various words and punctuation marks included.";
const CJK: &str = "这是一段中文测试文本，用于测试UTF-8编码中多字节字符的处理性能。日本語のテキストも含まれています。한국어 텍스트도 포함되어 있습니다。";
const EMOJI: &str = "Hello 🌍🌍🌍! Flags: 🇺🇸🇬🇧🇯🇵🇨🇳 Family: 👨\u{200d}👩\u{200d}👧\u{200d}👦 Skin: 👋🏻👋🏼👋🏽👋🏾👋🏿 Fun: 🎉🎊🎈🎁🎄🎃";
const MIXED: &str = "Hello, 世界! 🌍 Привет мир! こんにちは世界！Héllo wörld! 你好世界！안녕하세요 세계! مرحبا بالعالم";

const WARMUP_ITERS: u32 = 1_000;
const BENCH_ITERS: u32 = 10_000;
const SAMPLE_ROUNDS: usize = 10;

fn bench<F: Fn() -> usize>(f: F) -> Duration {
    for _ in 0..WARMUP_ITERS {
        black_box(f());
    }
    let mut samples = Vec::with_capacity(SAMPLE_ROUNDS);
    for _ in 0..SAMPLE_ROUNDS {
        let start = Instant::now();
        for _ in 0..BENCH_ITERS {
            black_box(f());
        }
        samples.push(start.elapsed());
    }
    samples.sort();
    samples[SAMPLE_ROUNDS / 2]
}

fn main() {
    println!("| OS | {} {} |", std::env::consts::OS, std::env::consts::ARCH);

    let cjk_large = CJK.repeat(64);
    let emoji_large = EMOJI.repeat(64);
    let ascii_large = ASCII.repeat(64);
    let inputs: &[(&str, &str)] = &[
        ("ascii", ASCII),
        ("cjk", CJK),
        ("emoji", EMOJI),
        ("mixed", MIXED),
        ("ascii_large", &ascii_large),
        ("cjk_large", &cjk_large),
        ("emoji_large", &emoji_large),
    ];

    // Interleave old/new per input and take the best-of-3 ratio to further
    // damp frequency scaling.
    println!("| Input | Bytes | old (ns/iter) | new (ns/iter) | old/new |");
    println!("|-------|------:|--------------:|--------------:|--------:|");
    for &(name, input) in inputs {
        let mut ratios = Vec::new();
        let mut old_best = f64::MAX;
        let mut new_best = f64::MAX;
        for _ in 0..3 {
            let old_ns = bench(|| baseline::utf16_len(black_box(input))).as_nanos() as f64
                / BENCH_ITERS as f64;
            let new_ns =
                bench(|| utf16_len(black_box(input))).as_nanos() as f64 / BENCH_ITERS as f64;
            old_best = old_best.min(old_ns);
            new_best = new_best.min(new_ns);
            ratios.push(old_ns / new_ns);
        }
        ratios.sort_by(f64::total_cmp);
        println!(
            "| {:<5} | {:>5} | {:>13.1} | {:>13.1} | {:>6.2}x |",
            name,
            input.len(),
            old_best,
            new_best,
            ratios[1],
        );
    }
}

/// The previous implementation, vendored verbatim from commit 852d444
/// (`src/ascii.rs` x86_64/aarch64 paths + `src/x86_64.rs` / `src/aarch64.rs`
/// + the `utf16_len_tail` helper from `src/lib.rs`).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod baseline {
    pub fn utf16_len(s: &str) -> usize {
        let bytes = s.as_bytes();
        let start = ascii_prefix_len(bytes);
        if start == bytes.len() {
            start
        } else {
            // SAFETY: bytes comes from a valid str, and start is a verified ASCII prefix.
            unsafe { utf16_len_non_ascii(bytes, start) }
        }
    }

    /// Count the tail after skipping continuation bytes at `i`.
    ///
    /// # Safety
    /// `bytes` must be valid UTF-8, and `i <= bytes.len()`.
    #[inline(always)]
    unsafe fn utf16_len_tail(bytes: &[u8], i: usize) -> usize {
        let mut tail_start = i;
        // SAFETY: the length check guards each byte access.
        while tail_start < bytes.len()
            && (unsafe { *bytes.get_unchecked(tail_start) } & 0xC0) == 0x80
        {
            tail_start += 1;
        }
        // SAFETY: bytes is valid UTF-8, and tail_start <= bytes.len() is a char boundary.
        let tail = unsafe { std::str::from_utf8_unchecked(bytes.get_unchecked(tail_start..)) };
        tail.encode_utf16().count()
    }

    #[inline(always)]
    fn ascii_prefix_len(bytes: &[u8]) -> usize {
        const USIZE_SIZE: usize = size_of::<usize>();
        const NONASCII_MASK: usize = usize::MAX / 255 * 0x80;

        if bytes.len() < 64 {
            let (chunks, remainder) = bytes.as_chunks::<USIZE_SIZE>();
            for chunk in chunks {
                let word = usize::from_ne_bytes(*chunk);
                if (word & NONASCII_MASK) != 0 {
                    // SAFETY: chunk starts within the same allocation as bytes.
                    return unsafe { chunk.as_ptr().offset_from_unsigned(bytes.as_ptr()) };
                }
            }
            return if remainder.iter().all(|b| b.is_ascii()) {
                bytes.len()
            } else {
                bytes.len() - remainder.len()
            };
        }

        #[cfg(target_arch = "x86_64")]
        {
            ascii_prefix_len_sse2(bytes)
        }
        #[cfg(target_arch = "aarch64")]
        {
            ascii_prefix_len_neon(bytes)
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn ascii_prefix_len_sse2(bytes: &[u8]) -> usize {
        use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_movemask_epi8, _mm_or_si128};

        let (chunks, rest) = bytes.as_chunks::<64>();
        for chunk in chunks {
            let ptr = chunk.as_ptr();
            // SAFETY: chunk is 64 bytes. SSE2 is baseline on x86_64.
            let mask = unsafe {
                let a1 = _mm_loadu_si128(ptr as *const __m128i);
                let a2 = _mm_loadu_si128(ptr.add(16) as *const __m128i);
                let b1 = _mm_loadu_si128(ptr.add(32) as *const __m128i);
                let b2 = _mm_loadu_si128(ptr.add(48) as *const __m128i);
                let combined = _mm_or_si128(_mm_or_si128(a1, a2), _mm_or_si128(b1, b2));
                _mm_movemask_epi8(combined)
            };
            if mask != 0 {
                // SAFETY: chunk starts within the same allocation as bytes.
                return unsafe { ptr.offset_from_unsigned(bytes.as_ptr()) };
            }
        }

        if rest.is_ascii() {
            bytes.len()
        } else {
            bytes.len() - rest.len()
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn ascii_prefix_len_neon(bytes: &[u8]) -> usize {
        use std::arch::aarch64::{vld1q_u8, vmaxvq_u8, vorrq_u8};

        let (chunks, rest) = bytes.as_chunks::<64>();
        for chunk in chunks {
            let ptr = chunk.as_ptr();
            // SAFETY: chunk is 64 bytes. NEON is baseline on aarch64, and these
            // vector loads do not require alignment.
            let max = unsafe {
                let a1 = vld1q_u8(ptr);
                let a2 = vld1q_u8(ptr.add(16));
                let b1 = vld1q_u8(ptr.add(32));
                let b2 = vld1q_u8(ptr.add(48));
                let combined = vorrq_u8(vorrq_u8(a1, a2), vorrq_u8(b1, b2));
                vmaxvq_u8(combined)
            };
            if max >= 128 {
                // SAFETY: chunk starts within the same allocation as bytes.
                return unsafe { ptr.offset_from_unsigned(bytes.as_ptr()) };
            }
        }

        let (vectors, rest) = rest.as_chunks::<16>();
        for vector in vectors {
            // SAFETY: vector contains 16 bytes, and the load is unaligned.
            let max = unsafe { vmaxvq_u8(vld1q_u8(vector.as_ptr())) };
            if max >= 128 {
                // SAFETY: vector starts within the same allocation as bytes.
                return unsafe { vector.as_ptr().offset_from_unsigned(bytes.as_ptr()) };
            }
        }

        if rest.is_ascii() {
            bytes.len()
        } else {
            bytes.len() - rest.len()
        }
    }

    /// Old x86_64 kernel: SSE2, 16 bytes at a time.
    ///
    /// # Safety
    /// `bytes` must be valid UTF-8, with `i <= bytes.len()` and an ASCII prefix `bytes[..i]`.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn utf16_len_non_ascii(bytes: &[u8], mut i: usize) -> usize {
        use std::arch::x86_64::*;

        let len = bytes.len();
        let mut count = i;

        // SAFETY: SSE2 is always available on x86_64, and every load is guarded by
        // `i + 16 <= len`.
        unsafe {
            let cont_max = _mm_set1_epi8(0xBF_u8 as i8);
            let four_mask = _mm_set1_epi8(0xF0_u8 as i8);
            let zero = _mm_setzero_si128();

            while i + 16 <= len {
                let batch = ((len - i) / 16).min(255);
                let mut leader_acc = zero;
                let mut four_acc = zero;
                for _ in 0..batch {
                    let chunk = _mm_loadu_si128(bytes.as_ptr().add(i) as *const __m128i);
                    let is_leader = _mm_cmpgt_epi8(chunk, cont_max);
                    let is_four = _mm_cmpeq_epi8(_mm_and_si128(chunk, four_mask), four_mask);
                    leader_acc = _mm_sub_epi8(leader_acc, is_leader);
                    four_acc = _mm_sub_epi8(four_acc, is_four);
                    i += 16;
                }
                let sad =
                    _mm_add_epi64(_mm_sad_epu8(leader_acc, zero), _mm_sad_epu8(four_acc, zero));
                let sum = _mm_add_epi64(sad, _mm_srli_si128::<8>(sad));
                count += _mm_cvtsi128_si64(sum) as usize;
            }
        }

        if i == len {
            return count;
        }

        // SAFETY: bytes is valid UTF-8, and the SIMD loop maintains i <= len.
        count + unsafe { utf16_len_tail(bytes, i) }
    }

    /// Old aarch64 kernel: NEON, 16 bytes at a time.
    ///
    /// # Safety
    /// `bytes` must be valid UTF-8, with `i <= bytes.len()` and an ASCII prefix `bytes[..i]`.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn utf16_len_non_ascii(bytes: &[u8], mut i: usize) -> usize {
        use std::arch::aarch64::*;

        let len = bytes.len();

        let mut continuation_count: usize = 0;
        let mut four_byte_count: usize = 0;

        // SAFETY: NEON is always available on aarch64.
        unsafe {
            let cont_mask = vdupq_n_u8(0xC0);
            let cont_val = vdupq_n_u8(0x80);
            let four_threshold = vdupq_n_u8(0xEF);
            let one = vdupq_n_u8(1);

            while i + 16 <= len {
                let batch = ((len - i) / 16).min(255);
                let mut cont_acc = vdupq_n_u8(0);
                let mut four_acc = vdupq_n_u8(0);

                for _ in 0..batch {
                    let chunk = vld1q_u8(bytes.as_ptr().add(i));

                    let masked = vandq_u8(chunk, cont_mask);
                    let is_cont = vceqq_u8(masked, cont_val);
                    cont_acc = vsubq_u8(cont_acc, is_cont);

                    let sub = vqsubq_u8(chunk, four_threshold);
                    let is_four = vminq_u8(sub, one);
                    four_acc = vaddq_u8(four_acc, is_four);

                    i += 16;
                }

                continuation_count += vaddlvq_u8(cont_acc) as usize;
                four_byte_count += vaddlvq_u8(four_acc) as usize;
            }

            // SAFETY: bytes is valid UTF-8, and the SIMD loop maintains i <= len.
            i - continuation_count + four_byte_count + utf16_len_tail(bytes, i)
        }
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod baseline {
    pub fn utf16_len(s: &str) -> usize {
        s.encode_utf16().count()
    }
}
