//! NEON-based UTF-16 length calculation (always available on aarch64).
//!
//! Shaped like napi-rs/json-escape-simd's kernels: a pointer cursor with a
//! remaining-byte count, four unrolled vectors per iteration counted into
//! four byte-lane accumulators, leftover vectors and an in-register tail
//! folded into the same accumulators, and one horizontal sum. Inputs of
//! fewer than four vectors skip the accumulators and count into one.

use std::arch::aarch64::*;

/// Iterations of four vectors between two sums of the byte-lane
/// accumulators. A lane gains at most 2 per vector, so the merged
/// accumulators hold at most 4 * 2 * 30 = 240 after a batch, which leaves
/// room for 3 leftover vectors and the tail: 240 + 3 * 2 + 2 = 248 < 256.
const MAX_BATCH: usize = 30;

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
    const CHUNK: usize = LANES * 4;

    // SAFETY: NEON is baseline on aarch64. Each full-vector load stays within
    // the input, the mask table, or the placeholder, except the short-input
    // load, which stays within the input's page.
    unsafe {
        let len = bytes.len();
        let mut sptr = bytes.as_ptr().add(start);
        let mut nb = len - start;

        let zero = vdupq_n_u8(0);
        let cont_max = vdupq_n_s8(0xBF_u8 as i8);
        let four_min = vdupq_n_u8(0xF0);

        // Minus the units each byte contributes: 0xFF (-1) for a leader (any
        // byte above the continuation range, compared as signed bytes), and
        // -1 more for a four-byte leader.
        macro_rules! neg_units {
            ($v:expr) => {{
                let v = $v;
                vaddq_u8(
                    vcgtq_s8(vreinterpretq_s8_u8(v), cont_max),
                    vcgeq_u8(v, four_min),
                )
            }};
        }
        // Minus the units of the last nb bytes, from the vector that ends at
        // the input's end, which holds at least a vector: only its last nb
        // lanes are uncounted, and byte-wise counting needs no UTF-8 boundary
        // care.
        macro_rules! neg_tail_units {
            () => {{
                let v = vld1q_u8(sptr.add(nb).sub(LANES));
                vandq_u8(neg_units!(v), vld1q_u8(crate::keep_last(LANES, nb)))
            }};
        }

        if len < LANES {
            // The whole input is shorter than a vector.
            let (v, keep) = short_vector!(sptr, nb, LANES, vld1q_u8);
            let acc = vsubq_u8(zero, vandq_u8(neg_units!(v), keep));
            return start + vaddlvq_u8(acc) as usize;
        }

        if nb < CHUNK {
            // Fewer than four vectors: one accumulator, one sum.
            let mut acc = zero;
            while nb >= LANES {
                acc = vsubq_u8(acc, neg_units!(vld1q_u8(sptr)));
                sptr = sptr.add(LANES);
                nb -= LANES;
            }
            if nb > 0 {
                acc = vsubq_u8(acc, neg_tail_units!());
            }
            return start + vaddlvq_u8(acc) as usize;
        }

        macro_rules! merge {
            ($a0:expr, $a1:expr, $a2:expr, $a3:expr) => {
                vaddq_u8(vaddq_u8($a0, $a1), vaddq_u8($a2, $a3))
            };
        }

        // Sums of full batches, widened to u32 lanes without a horizontal add.
        let mut total = vdupq_n_u32(0);
        let (mut a0, mut a1, mut a2, mut a3) = (zero, zero, zero, zero);
        macro_rules! chunk {
            () => {
                a0 = vsubq_u8(a0, neg_units!(vld1q_u8(sptr)));
                a1 = vsubq_u8(a1, neg_units!(vld1q_u8(sptr.add(LANES))));
                a2 = vsubq_u8(a2, neg_units!(vld1q_u8(sptr.add(LANES * 2))));
                a3 = vsubq_u8(a3, neg_units!(vld1q_u8(sptr.add(LANES * 3))));
                sptr = sptr.add(CHUNK);
            };
        }

        while nb >= CHUNK * MAX_BATCH {
            for _ in 0..MAX_BATCH {
                chunk!();
            }
            nb -= CHUNK * MAX_BATCH;
            // More follows: widen now so the lanes can't overflow.
            total = vpadalq_u16(total, vpaddlq_u8(merge!(a0, a1, a2, a3)));
            (a0, a1, a2, a3) = (zero, zero, zero, zero);
        }
        // Fewer than MAX_BATCH iterations remain.
        while nb >= CHUNK {
            chunk!();
            nb -= CHUNK;
        }
        while nb >= LANES {
            a0 = vsubq_u8(a0, neg_units!(vld1q_u8(sptr)));
            sptr = sptr.add(LANES);
            nb -= LANES;
        }
        if nb > 0 {
            a1 = vsubq_u8(a1, neg_tail_units!());
        }

        start + vaddlvq_u8(merge!(a0, a1, a2, a3)) as usize + vaddvq_u32(total) as usize
    }
}
