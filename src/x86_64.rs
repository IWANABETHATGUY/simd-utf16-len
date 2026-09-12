//! x86_64 SIMD UTF-16 length calculation.
//!
//! Runtime dispatch (mirroring napi-rs/escape-simd): AVX-512BW -> AVX2 -> SSE2
//! (baseline on x86_64). Each kernel processes 4 unrolled vectors per
//! iteration through a pointer cursor, and finishes with an in-register tail
//! (overlapping last-vector load, or a stack placeholder for short inputs)
//! instead of a scalar fallback.

use std::arch::x86_64::*;

/// Compute the number of UTF-16 code units for UTF-8 string.
pub fn utf16_len(s: &str) -> usize {
    let bytes = s.as_bytes();
    let start = crate::ascii::ascii_prefix_len(bytes);
    if start == bytes.len() {
        return start;
    }
    // SAFETY: bytes comes from a valid str, and start is a verified ASCII prefix.
    unsafe {
        if is_x86_feature_detected!("avx512bw") {
            utf16_len_avx512(bytes, start)
        } else if is_x86_feature_detected!("avx2") {
            utf16_len_avx2(bytes, start)
        } else {
            utf16_len_sse2(bytes, start)
        }
    }
}

/// UTF-16 units contributed by the last `nb` bytes of `bytes`, given the
/// per-lane leader/four-leader bitmasks of the vector holding them.
///
/// When `len >= LANES` the tail is covered by an overlapping load of the last
/// `LANES` bytes, so only lanes `LANES - nb..` are new; otherwise the `nb`
/// bytes sit in a zeroed stack placeholder and only lanes `..nb` count.
/// Byte-wise counting needs no UTF-8 boundary care: ignored lanes are simply
/// not counted, wherever their character starts.
///
/// Must be expanded inside an `unsafe` block: the overlapping load stays in
/// bounds because `len >= LANES`, the placeholder copy reads the `nb < LANES`
/// bytes remaining at `sptr`, and the placeholder is fully initialized for
/// `LANES` bytes.
macro_rules! tail_count {
    ($bytes:expr, $sptr:expr, $nb:expr, $lanes:expr, $load:expr, $masks:expr) => {{
        let bytes = $bytes;
        let nb = $nb;
        let lanes = $lanes;
        let (leader_bits, four_bits) = if bytes.len() >= lanes {
            let v = $load(bytes.as_ptr().add(bytes.len() - lanes));
            let (leader, four) = $masks(v);
            let shift = (lanes - nb) as u32;
            (leader >> shift, four >> shift)
        } else {
            let mut placeholder = [0u8; 64];
            std::ptr::copy_nonoverlapping($sptr, placeholder.as_mut_ptr(), nb);
            let v = $load(placeholder.as_ptr());
            let (leader, four) = $masks(v);
            let keep = (1u64 << nb) - 1;
            (leader & keep, four & keep)
        };
        (leader_bits.count_ones() + four_bits.count_ones()) as usize
    }};
}

/// AVX-512BW kernel: byte compares produce 64-bit mask registers directly, so
/// counting is a plain popcount — no accumulator batching needed.
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn utf16_len_avx512(bytes: &[u8], start: usize) -> usize {
    const LANES: usize = 64;
    const CHUNK: usize = LANES * 4;

    unsafe {
        let len = bytes.len();
        let mut sptr = bytes.as_ptr().add(start);
        let mut nb = len - start;
        let mut count = start;

        let cont_mask = _mm512_set1_epi8(0xC0_u8 as i8);
        let cont_val = _mm512_set1_epi8(0x80_u8 as i8);
        let four_val = _mm512_set1_epi8(0xF0_u8 as i8);

        // Continuation-byte count of one vector; masked-off lanes load as
        // zero, which is neither a continuation byte nor a four-byte leader.
        macro_rules! cont {
            ($v:expr) => {
                _mm512_cmpeq_epi8_mask(_mm512_and_si512($v, cont_mask), cont_val).count_ones()
                    as usize
            };
        }
        macro_rules! four {
            ($v:expr) => {
                _mm512_cmpge_epu8_mask($v, four_val).count_ones() as usize
            };
        }

        // 4 independent load+count chains per iteration (256 bytes).
        while nb >= CHUNK {
            let v1 = _mm512_loadu_si512(sptr as *const __m512i);
            let v2 = _mm512_loadu_si512(sptr.add(LANES) as *const __m512i);
            let v3 = _mm512_loadu_si512(sptr.add(LANES * 2) as *const __m512i);
            let v4 = _mm512_loadu_si512(sptr.add(LANES * 3) as *const __m512i);
            count += CHUNK - (cont!(v1) + cont!(v2) + cont!(v3) + cont!(v4))
                + (four!(v1) + four!(v2) + four!(v3) + four!(v4));
            sptr = sptr.add(CHUNK);
            nb -= CHUNK;
        }

        while nb >= LANES {
            let v = _mm512_loadu_si512(sptr as *const __m512i);
            count += LANES - cont!(v) + four!(v);
            sptr = sptr.add(LANES);
            nb -= LANES;
        }

        if nb > 0 {
            // Fault-suppressing masked load: no scalar tail, no overread.
            let k: __mmask64 = (1u64 << nb) - 1;
            let v = _mm512_maskz_loadu_epi8(k, sptr as *const i8);
            count += nb - cont!(v) + four!(v);
        }

        count
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
        let mut count = start;

        let cont_max = _mm256_set1_epi8(0xBF_u8 as i8);
        let four_mask = _mm256_set1_epi8(0xF0_u8 as i8);
        let zero = _mm256_setzero_si256();

        macro_rules! leader {
            ($v:expr) => {
                _mm256_cmpgt_epi8($v, cont_max)
            };
        }
        macro_rules! four {
            ($v:expr) => {
                _mm256_cmpeq_epi8(_mm256_and_si256($v, four_mask), four_mask)
            };
        }
        macro_rules! leader_bits {
            ($v:expr) => {
                _mm256_movemask_epi8(leader!($v)) as u32
            };
        }
        macro_rules! four_bits {
            ($v:expr) => {
                _mm256_movemask_epi8(four!($v)) as u32
            };
        }
        // Horizontal byte sum of one accumulator via SAD.
        macro_rules! sad_sum {
            ($acc:expr) => {{
                let sad = _mm256_sad_epu8($acc, zero);
                (_mm256_extract_epi64::<0>(sad)
                    + _mm256_extract_epi64::<1>(sad)
                    + _mm256_extract_epi64::<2>(sad)
                    + _mm256_extract_epi64::<3>(sad)) as usize
            }};
        }

        // u8 lane accumulators overflow after 255 increments; batches are
        // capped at 63 iterations so each lane stays <= 63 and the 4
        // accumulators can be merged (<= 252) before one horizontal sum.
        while nb >= CHUNK {
            let batch = (nb / CHUNK).min(63);
            let mut leader_acc = [zero; 4];
            let mut four_acc = [zero; 4];
            for _ in 0..batch {
                for (j, acc) in leader_acc.iter_mut().zip(four_acc.iter_mut()).enumerate() {
                    let v = _mm256_loadu_si256(sptr.add(LANES * j) as *const __m256i);
                    *acc.0 = _mm256_sub_epi8(*acc.0, leader!(v));
                    *acc.1 = _mm256_sub_epi8(*acc.1, four!(v));
                }
                sptr = sptr.add(CHUNK);
            }
            let leader_total = _mm256_add_epi8(
                _mm256_add_epi8(leader_acc[0], leader_acc[1]),
                _mm256_add_epi8(leader_acc[2], leader_acc[3]),
            );
            let four_total = _mm256_add_epi8(
                _mm256_add_epi8(four_acc[0], four_acc[1]),
                _mm256_add_epi8(four_acc[2], four_acc[3]),
            );
            count += sad_sum!(leader_total) + sad_sum!(four_total);
            nb -= batch * CHUNK;
        }

        while nb >= LANES {
            let v = _mm256_loadu_si256(sptr as *const __m256i);
            count += (leader_bits!(v).count_ones() + four_bits!(v).count_ones()) as usize;
            sptr = sptr.add(LANES);
            nb -= LANES;
        }

        if nb > 0 {
            count += tail_count!(
                bytes,
                sptr,
                nb,
                LANES,
                |p: *const u8| _mm256_loadu_si256(p as *const __m256i),
                |v: __m256i| (leader_bits!(v) as u64, four_bits!(v) as u64)
            );
        }

        count
    }
}

#[target_feature(enable = "sse2")]
unsafe fn utf16_len_sse2(bytes: &[u8], start: usize) -> usize {
    const LANES: usize = 16;
    const CHUNK: usize = LANES * 4;

    unsafe {
        let len = bytes.len();
        let mut sptr = bytes.as_ptr().add(start);
        let mut nb = len - start;
        let mut count = start;

        let cont_max = _mm_set1_epi8(0xBF_u8 as i8);
        let four_mask = _mm_set1_epi8(0xF0_u8 as i8);
        let zero = _mm_setzero_si128();

        macro_rules! leader {
            ($v:expr) => {
                _mm_cmpgt_epi8($v, cont_max)
            };
        }
        macro_rules! four {
            ($v:expr) => {
                _mm_cmpeq_epi8(_mm_and_si128($v, four_mask), four_mask)
            };
        }
        macro_rules! leader_bits {
            ($v:expr) => {
                _mm_movemask_epi8(leader!($v)) as u32
            };
        }
        macro_rules! four_bits {
            ($v:expr) => {
                _mm_movemask_epi8(four!($v)) as u32
            };
        }
        macro_rules! sad_sum {
            ($acc:expr) => {{
                let sad = _mm_sad_epu8($acc, zero);
                (_mm_cvtsi128_si64(sad) + _mm_cvtsi128_si64(_mm_srli_si128::<8>(sad))) as usize
            }};
        }

        // u8 lane accumulators overflow after 255 increments; batches are
        // capped at 63 iterations so each lane stays <= 63 and the 4
        // accumulators can be merged (<= 252) before one horizontal sum.
        while nb >= CHUNK {
            let batch = (nb / CHUNK).min(63);
            let mut leader_acc = [zero; 4];
            let mut four_acc = [zero; 4];
            for _ in 0..batch {
                for (j, acc) in leader_acc.iter_mut().zip(four_acc.iter_mut()).enumerate() {
                    let v = _mm_loadu_si128(sptr.add(LANES * j) as *const __m128i);
                    *acc.0 = _mm_sub_epi8(*acc.0, leader!(v));
                    *acc.1 = _mm_sub_epi8(*acc.1, four!(v));
                }
                sptr = sptr.add(CHUNK);
            }
            let leader_total = _mm_add_epi8(
                _mm_add_epi8(leader_acc[0], leader_acc[1]),
                _mm_add_epi8(leader_acc[2], leader_acc[3]),
            );
            let four_total = _mm_add_epi8(
                _mm_add_epi8(four_acc[0], four_acc[1]),
                _mm_add_epi8(four_acc[2], four_acc[3]),
            );
            count += sad_sum!(leader_total) + sad_sum!(four_total);
            nb -= batch * CHUNK;
        }

        while nb >= LANES {
            let v = _mm_loadu_si128(sptr as *const __m128i);
            count += (leader_bits!(v).count_ones() + four_bits!(v).count_ones()) as usize;
            sptr = sptr.add(LANES);
            nb -= LANES;
        }

        if nb > 0 {
            count += tail_count!(
                bytes,
                sptr,
                nb,
                LANES,
                |p: *const u8| _mm_loadu_si128(p as *const __m128i),
                |v: __m128i| (leader_bits!(v) as u64, four_bits!(v) as u64)
            );
        }

        count
    }
}
