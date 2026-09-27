//! x86_64 SIMD UTF-16 length calculation.
//!
//! Uses SSE2 (16 bytes at a time, always available on x86_64).

use std::arch::x86_64::*;

/// Compute the number of UTF-16 code units for UTF-8 string.
///
/// Not inlined into callers yet: inlining it, and the SSE2 kernel with it,
/// measured `mixed` and `cjk` 10 to 40% slower on Linux on Zen 3 with the
/// same instructions, so the kernel's placement decides its speed. The
/// kernels get restructured first; a later change inlines this. The ASCII
/// scan and its early return are all this function holds; the rest is a
/// tail call into `non_ascii`.
pub fn utf16_len(s: &str) -> usize {
    let bytes = s.as_bytes();
    let start = crate::ascii::ascii_prefix_len(bytes);
    if start == bytes.len() {
        start
    } else {
        non_ascii(bytes, start)
    }
}

/// Counts the bytes after the ASCII prefix. Kept out of line and reached by
/// a tail call, so `utf16_len` itself saves no registers: with the kernel
/// inlined into it, every call paid its prologue and epilogue, ASCII input
/// included, and the epilogue's position moved with the kernel's size, which
/// cost `ascii` 7 to 10% on Intel and Zen 4.
#[inline(never)]
fn non_ascii(bytes: &[u8], start: usize) -> usize {
    // SAFETY: bytes comes from a valid str, and start is a verified ASCII
    // prefix shorter than it.
    unsafe { utf16_len_sse2(bytes, start) }
}

/// The kernels `utf16_len` can run on this CPU: SSE2, which is baseline.
pub(crate) fn kernels() -> Vec<crate::__kernels::Kernel> {
    vec![crate::__kernels::Kernel {
        name: "sse2",
        utf16_len,
    }]
}

/// SSE2 kernel, `#[inline(always)]` so it runs inside `non_ascii` with no
/// further call.
///
/// # Safety
/// `bytes` must be valid UTF-8 with an ASCII prefix of `start` bytes, and
/// `start < bytes.len()`.
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
        let len = bytes.len();
        let mut sptr = bytes.as_ptr().add(start);
        let mut nb = len - start;

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

        if len < LANES {
            // The whole input is shorter than a vector.
            let (v, keep) = short_vector!(sptr, nb, LANES, load);
            count!(v, |mask| _mm_and_si128(mask, keep));
            return start + sum!();
        }

        // Besides the accumulators, only the cursor, the remaining count, and
        // the running total stay live across the loop, so `utf16_len` needs
        // no callee-saved registers, which every call would pay for, ASCII
        // input included.
        let all = |mask| mask;
        let mut total = start;
        loop {
            let batch = (nb / LANES).min(MAX_BATCH);
            for _ in 0..batch {
                count!(load(sptr), all);
                sptr = sptr.add(LANES);
            }
            nb -= batch * LANES;
            if nb < LANES {
                break;
            }
            // More full vectors follow: sum now so the lanes can't overflow.
            total += sum!();
            (leader_acc, four_acc) = (zero, zero);
        }
        if nb > 0 {
            // The vector that ends at the input's end, which holds at least a
            // vector: only its last nb lanes are uncounted, and byte-wise
            // counting needs no UTF-8 boundary care.
            let v = load(sptr.add(nb).sub(LANES));
            let keep = load(crate::keep_last(LANES, nb));
            count!(v, |mask| _mm_and_si128(mask, keep));
        }
        total + sum!()
    }
}
