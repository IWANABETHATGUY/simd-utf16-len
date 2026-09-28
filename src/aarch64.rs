//! NEON-based UTF-16 length calculation (always available on aarch64).

use std::arch::aarch64::*;

/// Compute the number of UTF-16 code units for UTF-8 string using NEON.
#[inline]
pub fn utf16_len(s: &str) -> usize {
    let bytes = s.as_bytes();
    let start = crate::ascii::ascii_prefix_len(bytes);
    if start == bytes.len() {
        start
    } else {
        // SAFETY: bytes comes from a valid str, and start is a verified ASCII
        // prefix shorter than it.
        unsafe { utf16_len_neon(bytes, start) }
    }
}

/// A function of its own, but always inlined: a caller that inlines
/// `utf16_len` gets the whole count without a call, and one the inliner
/// turns down calls `utf16_len` as before.
///
/// # Safety
/// `bytes` must be valid UTF-8 with an ASCII prefix of `start` bytes, and
/// `start < bytes.len()`.
#[inline(always)]
unsafe fn utf16_len_neon(bytes: &[u8], start: usize) -> usize {
    const LANES: usize = 16;
    /// A lane of either accumulator gains at most 1 per vector, so this many
    /// vectors and the tail fit before a sum.
    const MAX_BATCH: usize = 254;

    // SAFETY: NEON is baseline on aarch64. Each full-vector load stays within
    // the input, the mask table, or the placeholder, except the short-input
    // load, which stays within the input's page.
    unsafe {
        let len = bytes.len();
        let mut sptr = bytes.as_ptr().add(start);
        let mut nb = len - start;

        let zero = vdupq_n_u8(0);
        let cont_mask = vdupq_n_u8(0xC0);
        let cont_val = vdupq_n_u8(0x80);
        let four_threshold = vdupq_n_u8(0xEF);
        let one = vdupq_n_u8(1);

        // Every byte contributes one unit, except continuation bytes, which
        // contribute none, and four-byte leaders, which contribute two: count
        // both kinds and adjust the byte count at the end.
        let (mut cont_acc, mut four_acc) = (zero, zero);
        macro_rules! count {
            ($v:expr, $keep:expr) => {{
                let v = $v;
                // Continuation bytes: (byte & 0xC0) == 0x80, as 0xFF lanes.
                let is_cont = vceqq_u8(vandq_u8(v, cont_mask), cont_val);
                // Four-byte leaders (byte >= 0xF0): saturating subtract 0xEF
                // gives non-zero only for them, then clamp to 1 with min.
                let is_four = vminq_u8(vqsubq_u8(v, four_threshold), one);
                cont_acc = vsubq_u8(cont_acc, $keep(is_cont));
                four_acc = vaddq_u8(four_acc, $keep(is_four));
            }};
        }

        if len < LANES {
            // The whole input is shorter than a vector.
            let (v, keep) = short_vector!(sptr, nb, LANES, vld1q_u8);
            count!(v, |mask| vandq_u8(mask, keep));
            return len - vaddlvq_u8(cont_acc) as usize + vaddlvq_u8(four_acc) as usize;
        }

        // Full vectors in batches, as the compiler unrolls a counted loop;
        // besides the accumulators, only the cursor, the remaining count, and
        // the running sums stay live across it.
        let all = |mask| mask;
        let (mut continuations, mut fours) = (0, 0);
        loop {
            let batch = (nb / LANES).min(MAX_BATCH);
            for _ in 0..batch {
                count!(vld1q_u8(sptr), all);
                sptr = sptr.add(LANES);
            }
            nb -= batch * LANES;
            if nb < LANES {
                break;
            }
            // More full vectors follow: sum now so the lanes can't overflow.
            continuations += vaddlvq_u8(cont_acc) as usize;
            fours += vaddlvq_u8(four_acc) as usize;
            (cont_acc, four_acc) = (zero, zero);
        }
        if nb > 0 {
            // The vector that ends at the input's end, which holds at least a
            // vector: only its last nb lanes are uncounted, and byte-wise
            // counting needs no UTF-8 boundary care.
            let v = vld1q_u8(sptr.add(nb).sub(LANES));
            let keep = vld1q_u8(crate::keep_last(LANES, nb));
            count!(v, |mask| vandq_u8(mask, keep));
        }
        continuations += vaddlvq_u8(cont_acc) as usize;
        fours += vaddlvq_u8(four_acc) as usize;

        len - continuations + fours
    }
}
