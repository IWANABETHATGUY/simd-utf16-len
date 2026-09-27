//! x86_64 SIMD UTF-16 length calculation.
//!
//! Uses SSE2 (16 bytes at a time, always available on x86_64).

use std::arch::x86_64::*;

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

/// Counts the bytes after the ASCII prefix. Out of line, so callers that
/// inline the ASCII scan above stay small.
#[inline(never)]
fn non_ascii(bytes: &[u8], start: usize) -> usize {
    // SAFETY: bytes comes from a valid str, and start is a verified ASCII prefix.
    unsafe { utf16_len_sse2(bytes, start) }
}

/// The kernels `utf16_len` can run on this CPU: SSE2, which is baseline.
pub(crate) fn kernels() -> Vec<crate::__kernels::Kernel> {
    vec![crate::__kernels::Kernel {
        name: "sse2",
        utf16_len,
    }]
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

/// SSE2 kernel, `#[inline(always)]` so it runs inside `non_ascii` with no
/// further call.
///
/// # Safety
/// `bytes` must be valid UTF-8 with an ASCII prefix of `start` bytes.
#[inline(always)]
unsafe fn utf16_len_sse2(bytes: &[u8], start: usize) -> usize {
    const LANES: usize = 16;
    /// A lane of either accumulator gains at most 1 per vector, so this many
    /// vectors and the tail fit before a sum.
    const MAX_BATCH: usize = 254;

    // SAFETY: SSE2 is baseline on x86_64. Each full-vector load stays within
    // the input, the mask table, or the placeholder, except the short-input
    // load, which stays within the input's page.
    unsafe {
        let mut sptr = bytes.as_ptr().add(start);
        let mut nb = bytes.len() - start;

        let zero = _mm_setzero_si128();
        let cont_max = _mm_set1_epi8(0xBF_u8 as i8);
        let four_mask = _mm_set1_epi8(0xF0_u8 as i8);
        let load = |p: *const u8| _mm_loadu_si128(p as *const __m128i);

        // ASCII bytes and UTF-8 leaders contribute one unit, with one extra
        // unit for four-byte leaders. Independent accumulators avoid a serial
        // dependency between the two subtractions.
        let (mut leader_acc, mut four_acc) = (zero, zero);
        macro_rules! count {
            ($v:expr, $keep:expr) => {{
                let v = $v;
                let is_leader = _mm_cmpgt_epi8(v, cont_max);
                let is_four = _mm_cmpeq_epi8(_mm_and_si128(v, four_mask), four_mask);
                leader_acc = _mm_sub_epi8(leader_acc, $keep(is_leader));
                four_acc = _mm_sub_epi8(four_acc, $keep(is_four));
            }};
        }
        macro_rules! sum {
            () => {{
                let sad =
                    _mm_add_epi64(_mm_sad_epu8(leader_acc, zero), _mm_sad_epu8(four_acc, zero));
                _mm_cvtsi128_si64(_mm_add_epi64(sad, _mm_srli_si128::<8>(sad))) as usize
            }};
        }

        let all = |mask| mask;
        let mut total = 0;
        while nb >= LANES * MAX_BATCH {
            for _ in 0..MAX_BATCH {
                count!(load(sptr), all);
                sptr = sptr.add(LANES);
            }
            nb -= LANES * MAX_BATCH;
            // More follows: sum now so the lanes can't overflow.
            total += sum!();
            (leader_acc, four_acc) = (zero, zero);
        }
        // Fewer than MAX_BATCH vectors remain.
        while nb >= LANES {
            count!(load(sptr), all);
            sptr = sptr.add(LANES);
            nb -= LANES;
        }
        if nb > 0 {
            let (v, keep) = tail_vector!(bytes, sptr, nb, LANES, load);
            count!(v, |mask| _mm_and_si128(mask, keep));
        }

        start + total + sum!()
    }
}
