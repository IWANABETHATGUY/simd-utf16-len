//! x86_64 SIMD UTF-16 length calculation.
//!
//! Runtime dispatch, as in napi-rs/json-escape-simd: AVX-512BW (only with the
//! `avx512` feature) -> AVX2 -> SSE2, which is baseline on x86_64. Every
//! kernel walks a pointer cursor with a remaining-byte count, counts four
//! unrolled vectors per iteration into four byte-lane accumulators, folds the
//! leftover vectors and the tail into the same accumulators, and sums them
//! once with `psadbw`. AVX2 and AVX-512 look up each byte's units by its high
//! nibble with `pshufb`; SSE2 has no byte shuffle, so it compares. Inputs with
//! fewer than `WIDE_MIN` bytes after their ASCII prefix stay on the SSE2
//! kernel inlined into the dispatch: a wider kernel's call, upper-register
//! cleanup, and wider reduction cost more than they save there.

use std::arch::x86_64::*;

/// A lane gains at most 2 per vector, so after this many iterations the four
/// merged accumulators hold at most 4 * 2 * 30 = 240, which leaves room for
/// 3 leftover vectors and the tail: 240 + 3 * 2 + 2 = 248 < 256.
const MAX_BATCH: usize = 30;

/// Below this many bytes after the ASCII prefix, the inlined SSE2 kernel
/// beats calling the AVX2 kernel, as measured by the kernel sweep on Zen 3
/// and Granite Rapids.
const WIDE_MIN: usize = 64;

/// Below this many bytes after the ASCII prefix, the AVX2 kernel beats the
/// AVX-512 kernel, whose 64-byte vectors and wider reduction only pay off
/// on longer inputs.
#[cfg(feature = "avx512")]
const AVX512_MIN: usize = 256;

/// Compute the number of UTF-16 code units for UTF-8 string.
#[inline]
pub fn utf16_len(s: &str) -> usize {
    let bytes = s.as_bytes();
    let start = crate::ascii::ascii_prefix_len(bytes);
    if start == bytes.len() {
        start
    } else {
        non_ascii(bytes, start)
    }
}

/// Counts the bytes after the ASCII prefix. Out of line, like
/// json-escape-simd's dispatch, so callers that inline the ASCII scan above
/// stay small. Short inputs run the SSE2 kernel here with no further call,
/// and this path calls nothing else, so it saves no registers; longer inputs
/// tail-call `wide`.
#[inline(never)]
fn non_ascii(bytes: &[u8], start: usize) -> usize {
    if bytes.len() - start >= WIDE_MIN {
        return wide(bytes, start);
    }
    // SAFETY: bytes comes from a valid str, start is a verified ASCII prefix,
    // and SSE2 is baseline on x86_64.
    unsafe { utf16_len_sse2(bytes, start) }
}

/// Runs the widest kernel this CPU supports on an input of at least
/// `WIDE_MIN` bytes after the ASCII prefix.
#[inline(never)]
fn wide(bytes: &[u8], start: usize) -> usize {
    // SAFETY: bytes comes from a valid str, start is a verified ASCII prefix,
    // SSE2 is baseline on x86_64, and each wider kernel runs only when the
    // CPU supports its features.
    unsafe {
        #[cfg(feature = "avx512")]
        {
            if bytes.len() - start >= AVX512_MIN && is_x86_feature_detected!("avx512bw") {
                return utf16_len_avx512(bytes, start);
            }
        }
        if is_x86_feature_detected!("avx2") {
            utf16_len_avx2(bytes, start)
        } else {
            utf16_len_sse2(bytes, start)
        }
    }
}

/// SSE2, then AVX2 and AVX-512 when this CPU has them (AVX-512 only with the
/// `avx512` feature), in the order `utf16_len` prefers them for long inputs.
pub(crate) fn kernels() -> Vec<crate::__kernels::Kernel> {
    use crate::__kernels::Kernel;
    let mut kernels = vec![Kernel {
        name: "sse2",
        utf16_len: sse2,
    }];
    if is_x86_feature_detected!("avx2") {
        kernels.push(Kernel {
            name: "avx2",
            utf16_len: avx2,
        });
    }
    #[cfg(feature = "avx512")]
    {
        if is_x86_feature_detected!("avx512bw") {
            kernels.push(Kernel {
                name: "avx512",
                utf16_len: avx512,
            });
        }
    }
    kernels
}

/// The ASCII prefix scan, then `kernel` on the rest, like `utf16_len`.
#[inline(always)]
fn with_kernel(s: &str, kernel: impl FnOnce(&[u8], usize) -> usize) -> usize {
    let bytes = s.as_bytes();
    let start = crate::ascii::ascii_prefix_len(bytes);
    if start == bytes.len() {
        start
    } else {
        kernel(bytes, start)
    }
}

fn sse2(s: &str) -> usize {
    // SAFETY: SSE2 is baseline on x86_64, and with_kernel passes a valid str's
    // bytes with a verified ASCII prefix.
    with_kernel(s, |bytes, start| unsafe { utf16_len_sse2(bytes, start) })
}

fn avx2(s: &str) -> usize {
    // SAFETY: `kernels` only lists this after detecting AVX2.
    with_kernel(s, |bytes, start| unsafe { utf16_len_avx2(bytes, start) })
}

#[cfg(feature = "avx512")]
fn avx512(s: &str) -> usize {
    // SAFETY: `kernels` only lists this after detecting AVX-512BW.
    with_kernel(s, |bytes, start| unsafe { utf16_len_avx512(bytes, start) })
}

/// The vector holding the last `nb` bytes of `bytes`, and the mask of the
/// lanes among them that are still uncounted, as `$load` produces them.
///
/// It is the vector that ends at the input's end when the input holds at
/// least `$lanes` bytes: only its last `nb` lanes are uncounted, and byte-wise
/// counting needs no UTF-8 boundary care. A shorter input reads a full vector
/// forward from its start when that stays within its page, or else the vector
/// that ends at its end, which then starts before the input but within the
/// same page, so neither load can fault. Debug builds and Miri copy the input
/// into a zeroed placeholder instead.
///
/// Must be expanded inside an `unsafe` block, with `$sptr` pointing at the
/// last `nb` bytes of `bytes`, `0 < nb < $lanes`, and `$lanes <= 64`.
macro_rules! tail_vector {
    ($bytes:expr, $sptr:expr, $nb:expr, $lanes:expr, $load:expr) => {{
        let (bytes, sptr, nb): (&[u8], *const u8, usize) = ($bytes, $sptr, $nb);
        let len = bytes.len();
        if len >= $lanes || (crate::OVERREAD && !crate::fits_in_page(sptr, $lanes)) {
            (
                $load(bytes.as_ptr().wrapping_add(len).wrapping_sub($lanes)),
                $load(crate::keep_last($lanes, nb)),
            )
        } else if crate::OVERREAD {
            ($load(sptr), $load(crate::keep_first(nb)))
        } else {
            let mut placeholder = [0u8; $lanes];
            std::ptr::copy_nonoverlapping(sptr, placeholder.as_mut_ptr(), nb);
            ($load(placeholder.as_ptr()), $load(crate::keep_first(nb)))
        }
    }};
}

/// SSE2 kernel, `#[inline(always)]` so short inputs run it inside `non_ascii`
/// with no call.
///
/// # Safety
/// `bytes` must be valid UTF-8 with an ASCII prefix of `start` bytes.
#[inline(always)]
unsafe fn utf16_len_sse2(bytes: &[u8], start: usize) -> usize {
    const LANES: usize = 16;
    const CHUNK: usize = LANES * 4;

    // SAFETY: SSE2 is baseline on x86_64. Each full-vector load stays within
    // the input, the mask table, or the placeholder, except the short-input
    // load, which stays within the input's page.
    unsafe {
        let mut sptr = bytes.as_ptr().add(start);
        let mut nb = bytes.len() - start;

        let zero = _mm_setzero_si128();
        let cont_max = _mm_set1_epi8(0xBF_u8 as i8);
        let four = _mm_set1_epi8(0xF0_u8 as i8);

        let load = |p: *const u8| _mm_loadu_si128(p as *const __m128i);
        // Minus the units each byte contributes: -1 for a leader (any byte
        // above the continuation range, compared as signed bytes), and -1
        // more for a four-byte leader.
        macro_rules! neg_units {
            ($v:expr) => {{
                let v = $v;
                _mm_add_epi8(
                    _mm_cmpgt_epi8(v, cont_max),
                    _mm_cmpeq_epi8(_mm_and_si128(v, four), four),
                )
            }};
        }
        macro_rules! merge {
            ($a0:expr, $a1:expr, $a2:expr, $a3:expr) => {
                _mm_add_epi8(_mm_add_epi8($a0, $a1), _mm_add_epi8($a2, $a3))
            };
        }

        // Sums of full batches, as two u64 lanes.
        let mut total = zero;
        let (mut a0, mut a1, mut a2, mut a3) = (zero, zero, zero, zero);
        while nb >= CHUNK {
            let batch = (nb / CHUNK).min(MAX_BATCH);
            for _ in 0..batch {
                a0 = _mm_sub_epi8(a0, neg_units!(load(sptr)));
                a1 = _mm_sub_epi8(a1, neg_units!(load(sptr.add(LANES))));
                a2 = _mm_sub_epi8(a2, neg_units!(load(sptr.add(LANES * 2))));
                a3 = _mm_sub_epi8(a3, neg_units!(load(sptr.add(LANES * 3))));
                sptr = sptr.add(CHUNK);
            }
            nb -= batch * CHUNK;
            if nb >= CHUNK {
                // Another batch follows: sum now so the lanes can't overflow.
                total = _mm_add_epi64(total, _mm_sad_epu8(merge!(a0, a1, a2, a3), zero));
                (a0, a1, a2, a3) = (zero, zero, zero, zero);
            }
        }
        while nb >= LANES {
            a0 = _mm_sub_epi8(a0, neg_units!(load(sptr)));
            sptr = sptr.add(LANES);
            nb -= LANES;
        }
        if nb > 0 {
            let (v, keep) = tail_vector!(bytes, sptr, nb, LANES, load);
            a1 = _mm_sub_epi8(a1, _mm_and_si128(neg_units!(v), keep));
        }

        total = _mm_add_epi64(total, _mm_sad_epu8(merge!(a0, a1, a2, a3), zero));
        total = _mm_add_epi64(total, _mm_srli_si128::<8>(total));
        start + _mm_cvtsi128_si64(total) as usize
    }
}

/// AVX2 kernel. Each byte's units come from a `pshufb` lookup of its high
/// nibble. There is no byte shift, so it shifts 16-bit lanes and masks off
/// the bits that bled in from the neighbor byte, as json-escape-simd's
/// nibble-table classifier does.
///
/// # Safety
/// The CPU must support AVX2, and `bytes` must be valid UTF-8 with an ASCII
/// prefix of `start` bytes.
#[target_feature(enable = "avx2")]
unsafe fn utf16_len_avx2(bytes: &[u8], start: usize) -> usize {
    const LANES: usize = 32;
    const CHUNK: usize = LANES * 4;

    // SAFETY: the caller checked AVX2. Each full-vector load stays within the
    // input, the mask table, or the placeholder, except the short-input load,
    // which stays within the input's page.
    unsafe {
        let mut sptr = bytes.as_ptr().add(start);
        let mut nb = bytes.len() - start;

        let zero = _mm256_setzero_si256();
        let nibble = _mm256_set1_epi8(0x0F);
        let table = _mm256_broadcastsi128_si256(_mm_loadu_si128(
            crate::UNITS_BY_HIGH_NIBBLE.as_ptr() as *const __m128i,
        ));

        let load = |p: *const u8| _mm256_loadu_si256(p as *const __m256i);
        macro_rules! units {
            ($v:expr) => {
                _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16::<4>($v), nibble))
            };
        }
        macro_rules! merge {
            ($a0:expr, $a1:expr, $a2:expr, $a3:expr) => {
                _mm256_add_epi8(_mm256_add_epi8($a0, $a1), _mm256_add_epi8($a2, $a3))
            };
        }

        // Sums of full batches, as four u64 lanes.
        let mut total = zero;
        let (mut a0, mut a1, mut a2, mut a3) = (zero, zero, zero, zero);
        while nb >= CHUNK {
            let batch = (nb / CHUNK).min(MAX_BATCH);
            for _ in 0..batch {
                a0 = _mm256_add_epi8(a0, units!(load(sptr)));
                a1 = _mm256_add_epi8(a1, units!(load(sptr.add(LANES))));
                a2 = _mm256_add_epi8(a2, units!(load(sptr.add(LANES * 2))));
                a3 = _mm256_add_epi8(a3, units!(load(sptr.add(LANES * 3))));
                sptr = sptr.add(CHUNK);
            }
            nb -= batch * CHUNK;
            if nb >= CHUNK {
                // Another batch follows: sum now so the lanes can't overflow.
                total = _mm256_add_epi64(total, _mm256_sad_epu8(merge!(a0, a1, a2, a3), zero));
                (a0, a1, a2, a3) = (zero, zero, zero, zero);
            }
        }
        while nb >= LANES {
            a0 = _mm256_add_epi8(a0, units!(load(sptr)));
            sptr = sptr.add(LANES);
            nb -= LANES;
        }
        if nb > 0 {
            let (v, keep) = tail_vector!(bytes, sptr, nb, LANES, load);
            a1 = _mm256_add_epi8(a1, _mm256_and_si256(units!(v), keep));
        }

        total = _mm256_add_epi64(total, _mm256_sad_epu8(merge!(a0, a1, a2, a3), zero));
        let sum = _mm_add_epi64(
            _mm256_castsi256_si128(total),
            _mm256_extracti128_si256::<1>(total),
        );
        let sum = _mm_add_epi64(sum, _mm_srli_si128::<8>(sum));
        start + _mm_cvtsi128_si64(sum) as usize
    }
}

/// AVX-512BW kernel: the same lookup on 64-byte vectors, and a masked load
/// for the tail, which suppresses faults on the masked-off lanes, so it needs
/// neither a copy nor an over-read.
///
/// # Safety
/// The CPU must support AVX-512BW, and `bytes` must be valid UTF-8 with an
/// ASCII prefix of `start` bytes.
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn utf16_len_avx512(bytes: &[u8], start: usize) -> usize {
    const LANES: usize = 64;
    const CHUNK: usize = LANES * 4;

    // SAFETY: the caller checked AVX-512BW. Each full-vector load stays
    // within the input, and the masked load reads only the last nb bytes.
    unsafe {
        let mut sptr = bytes.as_ptr().add(start);
        let mut nb = bytes.len() - start;

        let zero = _mm512_setzero_si512();
        let nibble = _mm512_set1_epi8(0x0F);
        let cont = _mm512_set1_epi8(0x80_u8 as i8);
        let table = _mm512_broadcast_i32x4(_mm_loadu_si128(
            crate::UNITS_BY_HIGH_NIBBLE.as_ptr() as *const __m128i
        ));

        let load = |p: *const u8| _mm512_loadu_si512(p as *const __m512i);
        macro_rules! units {
            ($v:expr) => {
                _mm512_shuffle_epi8(table, _mm512_and_si512(_mm512_srli_epi16::<4>($v), nibble))
            };
        }
        macro_rules! merge {
            ($a0:expr, $a1:expr, $a2:expr, $a3:expr) => {
                _mm512_add_epi8(_mm512_add_epi8($a0, $a1), _mm512_add_epi8($a2, $a3))
            };
        }

        // Sums of full batches, as eight u64 lanes.
        let mut total = zero;
        let (mut a0, mut a1, mut a2, mut a3) = (zero, zero, zero, zero);
        while nb >= CHUNK {
            let batch = (nb / CHUNK).min(MAX_BATCH);
            for _ in 0..batch {
                a0 = _mm512_add_epi8(a0, units!(load(sptr)));
                a1 = _mm512_add_epi8(a1, units!(load(sptr.add(LANES))));
                a2 = _mm512_add_epi8(a2, units!(load(sptr.add(LANES * 2))));
                a3 = _mm512_add_epi8(a3, units!(load(sptr.add(LANES * 3))));
                sptr = sptr.add(CHUNK);
            }
            nb -= batch * CHUNK;
            if nb >= CHUNK {
                // Another batch follows: sum now so the lanes can't overflow.
                total = _mm512_add_epi64(total, _mm512_sad_epu8(merge!(a0, a1, a2, a3), zero));
                (a0, a1, a2, a3) = (zero, zero, zero, zero);
            }
        }
        while nb >= LANES {
            a0 = _mm512_add_epi8(a0, units!(load(sptr)));
            sptr = sptr.add(LANES);
            nb -= LANES;
        }
        if nb > 0 {
            // Masked-off lanes take a continuation byte, which counts nothing.
            let keep: __mmask64 = (1u64 << nb) - 1;
            a1 = _mm512_add_epi8(
                a1,
                units!(_mm512_mask_loadu_epi8(cont, keep, sptr as *const i8)),
            );
        }

        total = _mm512_add_epi64(total, _mm512_sad_epu8(merge!(a0, a1, a2, a3), zero));
        start + _mm512_reduce_add_epi64(total) as usize
    }
}
