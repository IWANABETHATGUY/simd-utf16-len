//! x86_64 SIMD UTF-16 length calculation.
//!
//! Runtime dispatch, as in napi-rs/json-escape-simd: AVX-512BW (only with the
//! `avx512` feature) -> AVX2 -> SSE2 (baseline on x86_64). Each kernel counts
//! 4 unrolled vectors per iteration through a pointer cursor into `u8` lane
//! accumulators, folds leftover vectors and a masked tail vector into the same
//! accumulators, and reduces them once at the end. Nothing counts bits with
//! `count_ones`: POPCNT isn't baseline, so it would expand to a long software
//! sequence.

use std::arch::x86_64::*;

/// Accumulator lanes gain at most 2 per vector (a leader plus a four-byte
/// leader), so 4 accumulators merged into one stay within `u8` after 30
/// iterations, with room for 3 leftover vectors and the tail:
/// 4 * 2 * 30 + 3 * 2 + 2 = 248.
const MAX_BATCH: usize = 30;

/// Lane indices, compared with the byte count to keep only the tail lanes that
/// hold uncounted bytes.
static LANE_INDEX: [u8; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31,
];

/// Compute the number of UTF-16 code units for UTF-8 string.
pub fn utf16_len(s: &str) -> usize {
    let bytes = s.as_bytes();
    let start = crate::ascii::ascii_prefix_len(bytes);
    if start == bytes.len() {
        start
    } else {
        non_ascii(bytes, start)
    }
}

/// Below this many bytes after the ASCII prefix, the SSE2 kernel inlined into
/// `non_ascii` beats calling a wider kernel: the call, the `vzeroupper`, and
/// reducing a wider sum cost more than the wider vectors save.
const WIDE_MIN: usize = 256;

/// Runs the best kernel after the ASCII prefix. Kept out of line, like
/// napi-rs/json-escape-simd's dispatch, so the feature checks don't grow the
/// callers that inline the ASCII scan above.
#[inline]
fn non_ascii(bytes: &[u8], start: usize) -> usize {
    // SAFETY: bytes comes from a valid str, start is a verified ASCII prefix,
    // and each kernel only runs when the CPU supports its features.
    unsafe {
        if bytes.len() - start < WIDE_MIN {
            return utf16_len_sse2_short(bytes, start);
        }
        {
            #[cfg(feature = "avx512")]
            {
                if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vl") {
                    return utf16_len_avx512(bytes, start);
                }
            }
            if is_x86_feature_detected!("avx2") {
                return utf16_len_avx2(bytes, start);
            }
        }
        utf16_len_sse2(bytes, start)
    }
}

/// SSE2, then AVX2 and AVX-512 when this CPU has them (AVX-512 only with the
/// `avx512` feature). `utf16_len` runs the last one from `WIDE_MIN` bytes on.
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
        if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vl") {
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
    // SAFETY: `kernels` only lists this after detecting AVX-512BW and VL.
    with_kernel(s, |bytes, start| unsafe { utf16_len_avx512(bytes, start) })
}

/// AVX-512BW kernel: compares produce mask registers, and masked adds count
/// them straight into the accumulators.
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn utf16_len_avx512(bytes: &[u8], start: usize) -> usize {
    const LANES: usize = 64;
    const CHUNK: usize = LANES * 4;

    unsafe {
        let len = bytes.len();
        let mut sptr = bytes.as_ptr().add(start);
        let mut nb = len - start;

        let zero = _mm512_setzero_si512();
        let one = _mm512_set1_epi8(1);
        let cont_max = _mm512_set1_epi8(0xBF_u8 as i8);
        let four_min = _mm512_set1_epi8(0xF0_u8 as i8);

        // Adds 1 in each `$keep` lane holding a leader (not a continuation
        // byte), and 1 more for a four-byte leader.
        macro_rules! count {
            ($acc:expr, $v:expr, $keep:expr) => {{
                let v = $v;
                let leader = _mm512_cmpgt_epi8_mask(v, cont_max) & $keep;
                let four = _mm512_cmpge_epu8_mask(v, four_min) & $keep;
                $acc = _mm512_mask_add_epi8($acc, leader, $acc, one);
                $acc = _mm512_mask_add_epi8($acc, four, $acc, one);
            }};
        }
        macro_rules! fold {
            ($acc:expr) => {
                _mm512_sad_epu8(
                    _mm512_add_epi8(
                        _mm512_add_epi8($acc[0], $acc[1]),
                        _mm512_add_epi8($acc[2], $acc[3]),
                    ),
                    zero,
                )
            };
        }

        let mut total = zero;
        let mut acc = [zero; 4];
        while nb >= CHUNK {
            let batch = (nb / CHUNK).min(MAX_BATCH);
            for _ in 0..batch {
                for (j, a) in acc.iter_mut().enumerate() {
                    count!(
                        *a,
                        _mm512_loadu_si512(sptr.add(LANES * j) as *const __m512i),
                        u64::MAX
                    );
                }
                sptr = sptr.add(CHUNK);
            }
            nb -= batch * CHUNK;
            if nb >= CHUNK {
                // Another batch follows: fold now so the u8 lanes can't overflow.
                total = _mm512_add_epi64(total, fold!(acc));
                acc = [zero; 4];
            }
        }
        while nb >= LANES {
            count!(acc[0], _mm512_loadu_si512(sptr as *const __m512i), u64::MAX);
            sptr = sptr.add(LANES);
            nb -= LANES;
        }
        if nb > 0 {
            // Fault-suppressing masked load: no copy and no over-read.
            let keep: __mmask64 = (1u64 << nb) - 1;
            count!(
                acc[1],
                _mm512_maskz_loadu_epi8(keep, sptr as *const i8),
                keep
            );
        }

        total = _mm512_add_epi64(total, fold!(acc));
        start + _mm512_reduce_add_epi64(total) as usize
    }
}

#[target_feature(enable = "avx2")]
unsafe fn utf16_len_avx2(bytes: &[u8], start: usize) -> usize {
    const LANES: usize = 32;
    const CHUNK: usize = LANES * 4;

    unsafe {
        let len = bytes.len();
        let mut sptr = bytes.as_ptr().add(start);
        let mut nb = len - start;

        let zero = _mm256_setzero_si256();
        let cont_max = _mm256_set1_epi8(0xBF_u8 as i8);
        let four_mask = _mm256_set1_epi8(0xF0_u8 as i8);

        macro_rules! load {
            ($p:expr) => {
                _mm256_loadu_si256($p as *const __m256i)
            };
        }
        // 0, -1, or -2 per lane: -1 for a leader (not a continuation byte),
        // and -1 more for a four-byte leader. Subtracting it counts units.
        macro_rules! units {
            ($v:expr) => {{
                let v = $v;
                _mm256_add_epi8(
                    _mm256_cmpgt_epi8(v, cont_max),
                    _mm256_cmpeq_epi8(_mm256_and_si256(v, four_mask), four_mask),
                )
            }};
        }
        macro_rules! fold {
            ($acc:expr) => {
                _mm256_sad_epu8(
                    _mm256_add_epi8(
                        _mm256_add_epi8($acc[0], $acc[1]),
                        _mm256_add_epi8($acc[2], $acc[3]),
                    ),
                    zero,
                )
            };
        }

        let mut total = zero;
        let mut acc = [zero; 4];
        while nb >= CHUNK {
            let batch = (nb / CHUNK).min(MAX_BATCH);
            for _ in 0..batch {
                for (j, a) in acc.iter_mut().enumerate() {
                    *a = _mm256_sub_epi8(*a, units!(load!(sptr.add(LANES * j))));
                }
                sptr = sptr.add(CHUNK);
            }
            nb -= batch * CHUNK;
            if nb >= CHUNK {
                // Another batch follows: fold now so the u8 lanes can't overflow.
                total = _mm256_add_epi64(total, fold!(acc));
                acc = [zero; 4];
            }
        }
        while nb >= LANES {
            acc[0] = _mm256_sub_epi8(acc[0], units!(load!(sptr)));
            sptr = sptr.add(LANES);
            nb -= LANES;
        }
        if nb > 0 {
            let index = load!(LANE_INDEX.as_ptr());
            let (v, keep) = if len >= LANES {
                // Overlapping load of the last vector: only its last nb lanes
                // are uncounted. Byte-wise counting needs no UTF-8 boundary care.
                let v = load!(bytes.as_ptr().add(len - LANES));
                let keep = _mm256_cmpgt_epi8(index, _mm256_set1_epi8((LANES - nb - 1) as i8));
                (v, keep)
            } else {
                let v = if crate::can_overread(sptr, LANES) {
                    load!(sptr)
                } else {
                    let mut placeholder = [0u8; LANES];
                    std::ptr::copy_nonoverlapping(sptr, placeholder.as_mut_ptr(), nb);
                    load!(placeholder.as_ptr())
                };
                (v, _mm256_cmpgt_epi8(_mm256_set1_epi8(nb as i8), index))
            };
            acc[1] = _mm256_sub_epi8(acc[1], _mm256_and_si256(units!(v), keep));
        }

        total = _mm256_add_epi64(total, fold!(acc));
        let sum = _mm_add_epi64(
            _mm256_castsi256_si128(total),
            _mm256_extracti128_si256::<1>(total),
        );
        start + (_mm_cvtsi128_si64(sum) + _mm_cvtsi128_si64(_mm_unpackhi_epi64(sum, sum))) as usize
    }
}

/// `#[inline]` so the dispatch can inline this path for short inputs.
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn utf16_len_sse2(bytes: &[u8], start: usize) -> usize {
    const LANES: usize = 16;
    const CHUNK: usize = LANES * 4;

    unsafe {
        let len = bytes.len();
        let mut sptr = bytes.as_ptr().add(start);
        let mut nb = len - start;

        let zero = _mm_setzero_si128();
        let cont_max = _mm_set1_epi8(0xBF_u8 as i8);
        let four_mask = _mm_set1_epi8(0xF0_u8 as i8);

        macro_rules! load {
            ($p:expr) => {
                _mm_loadu_si128($p as *const __m128i)
            };
        }
        // 0, -1, or -2 per lane: -1 for a leader (not a continuation byte),
        // and -1 more for a four-byte leader. Subtracting it counts units.
        macro_rules! units {
            ($v:expr) => {{
                let v = $v;
                _mm_add_epi8(
                    _mm_cmpgt_epi8(v, cont_max),
                    _mm_cmpeq_epi8(_mm_and_si128(v, four_mask), four_mask),
                )
            }};
        }
        macro_rules! fold {
            ($acc:expr) => {
                _mm_sad_epu8(
                    _mm_add_epi8(
                        _mm_add_epi8($acc[0], $acc[1]),
                        _mm_add_epi8($acc[2], $acc[3]),
                    ),
                    zero,
                )
            };
        }

        let mut total = zero;
        let mut acc = [zero; 4];
        while nb >= CHUNK {
            let batch = (nb / CHUNK).min(MAX_BATCH);
            for _ in 0..batch {
                for (j, a) in acc.iter_mut().enumerate() {
                    *a = _mm_sub_epi8(*a, units!(load!(sptr.add(LANES * j))));
                }
                sptr = sptr.add(CHUNK);
            }
            nb -= batch * CHUNK;
            if nb >= CHUNK {
                // Another batch follows: fold now so the u8 lanes can't overflow.
                total = _mm_add_epi64(total, fold!(acc));
                acc = [zero; 4];
            }
        }
        while nb >= LANES {
            acc[0] = _mm_sub_epi8(acc[0], units!(load!(sptr)));
            sptr = sptr.add(LANES);
            nb -= LANES;
        }
        if nb > 0 {
            let index = load!(LANE_INDEX.as_ptr());
            let (v, keep) = if len >= LANES {
                // Overlapping load of the last vector: only its last nb lanes
                // are uncounted. Byte-wise counting needs no UTF-8 boundary care.
                let v = load!(bytes.as_ptr().add(len - LANES));
                let keep = _mm_cmpgt_epi8(index, _mm_set1_epi8((LANES - nb - 1) as i8));
                (v, keep)
            } else {
                let v = if crate::can_overread(sptr, LANES) {
                    load!(sptr)
                } else {
                    let mut placeholder = [0u8; LANES];
                    std::ptr::copy_nonoverlapping(sptr, placeholder.as_mut_ptr(), nb);
                    load!(placeholder.as_ptr())
                };
                (v, _mm_cmpgt_epi8(_mm_set1_epi8(nb as i8), index))
            };
            acc[1] = _mm_sub_epi8(acc[1], _mm_and_si128(units!(v), keep));
        }

        total = _mm_add_epi64(total, fold!(acc));
        start
            + (_mm_cvtsi128_si64(total) + _mm_cvtsi128_si64(_mm_unpackhi_epi64(total, total)))
                as usize
    }
}

/// Experiment: `main`'s simple loop for inputs under `WIDE_MIN` bytes, with
/// separate leader and four-byte accumulators and the in-register tail.
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn utf16_len_sse2_short(bytes: &[u8], start: usize) -> usize {
    const LANES: usize = 16;

    unsafe {
        let len = bytes.len();
        let mut sptr = bytes.as_ptr().add(start);
        let mut nb = len - start;

        let zero = _mm_setzero_si128();
        let cont_max = _mm_set1_epi8(0xBF_u8 as i8);
        let four_mask = _mm_set1_epi8(0xF0_u8 as i8);
        let mut leader_acc = zero;
        let mut four_acc = zero;

        // Under WIDE_MIN bytes there are at most 16 vectors, so u8 lanes can't overflow.
        while nb >= LANES {
            let v = _mm_loadu_si128(sptr as *const __m128i);
            leader_acc = _mm_sub_epi8(leader_acc, _mm_cmpgt_epi8(v, cont_max));
            four_acc = _mm_sub_epi8(
                four_acc,
                _mm_cmpeq_epi8(_mm_and_si128(v, four_mask), four_mask),
            );
            sptr = sptr.add(LANES);
            nb -= LANES;
        }
        if nb > 0 {
            let index = _mm_loadu_si128(LANE_INDEX.as_ptr() as *const __m128i);
            let (v, keep) = if len >= LANES {
                let v = _mm_loadu_si128(bytes.as_ptr().add(len - LANES) as *const __m128i);
                (
                    v,
                    _mm_cmpgt_epi8(index, _mm_set1_epi8((LANES - nb - 1) as i8)),
                )
            } else {
                let v = if crate::can_overread(sptr, LANES) {
                    _mm_loadu_si128(sptr as *const __m128i)
                } else {
                    let mut placeholder = [0u8; LANES];
                    std::ptr::copy_nonoverlapping(sptr, placeholder.as_mut_ptr(), nb);
                    _mm_loadu_si128(placeholder.as_ptr() as *const __m128i)
                };
                (v, _mm_cmpgt_epi8(_mm_set1_epi8(nb as i8), index))
            };
            leader_acc = _mm_sub_epi8(leader_acc, _mm_and_si128(_mm_cmpgt_epi8(v, cont_max), keep));
            four_acc = _mm_sub_epi8(
                four_acc,
                _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(v, four_mask), four_mask), keep),
            );
        }

        let sad = _mm_add_epi64(_mm_sad_epu8(leader_acc, zero), _mm_sad_epu8(four_acc, zero));
        start + (_mm_cvtsi128_si64(sad) + _mm_cvtsi128_si64(_mm_unpackhi_epi64(sad, sad))) as usize
    }
}
