//! NEON-based UTF-16 length calculation (always available on aarch64).
//!
//! Same shape as the x86_64 kernels, from napi-rs/json-escape-simd: a pointer
//! cursor with a remaining-byte count, four unrolled vectors per iteration
//! counted into four byte-lane accumulators, leftover vectors and an
//! in-register tail folded into the same accumulators, and one horizontal
//! sum. Each byte's units come from a `tbl` lookup of its high nibble, as in
//! json-escape-simd's nibble-table classifier; NEON shifts bytes directly, so
//! the index needs no masking.

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
        // SAFETY: bytes comes from a valid str, and start is a verified ASCII prefix.
        unsafe { utf16_len_neon(bytes, start) }
    }
}

/// Out of line, so callers that inline the ASCII scan above stay small.
///
/// # Safety
/// `bytes` must be valid UTF-8 with an ASCII prefix of `start` bytes.
#[inline(never)]
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
        let table = vld1q_u8(crate::UNITS_BY_HIGH_NIBBLE.as_ptr());

        macro_rules! units {
            ($v:expr) => {
                vqtbl1q_u8(table, vshrq_n_u8::<4>($v))
            };
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
                a0 = vaddq_u8(a0, units!(vld1q_u8(sptr)));
                a1 = vaddq_u8(a1, units!(vld1q_u8(sptr.add(LANES))));
                a2 = vaddq_u8(a2, units!(vld1q_u8(sptr.add(LANES * 2))));
                a3 = vaddq_u8(a3, units!(vld1q_u8(sptr.add(LANES * 3))));
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
            a0 = vaddq_u8(a0, units!(vld1q_u8(sptr)));
            sptr = sptr.add(LANES);
            nb -= LANES;
        }
        if nb > 0 {
            // The vector that ends at the input's end, when the input holds a
            // whole vector: only its last nb lanes are uncounted, and
            // byte-wise counting needs no UTF-8 boundary care. A shorter
            // input reads a full vector forward from its start when that
            // stays within its page, or else the vector that ends at its
            // end, which starts before the input but within the same page,
            // so neither load can fault. Debug builds and Miri copy into a
            // zeroed placeholder instead.
            let (v, keep) =
                if len >= LANES || (crate::OVERREAD && !crate::fits_in_page(sptr, LANES)) {
                    (
                        vld1q_u8(bytes.as_ptr().wrapping_add(len).wrapping_sub(LANES)),
                        vld1q_u8(crate::keep_last(LANES, nb)),
                    )
                } else if crate::OVERREAD {
                    (vld1q_u8(sptr), vld1q_u8(crate::keep_first(nb)))
                } else {
                    let mut placeholder = [0u8; LANES];
                    std::ptr::copy_nonoverlapping(sptr, placeholder.as_mut_ptr(), nb);
                    (
                        vld1q_u8(placeholder.as_ptr()),
                        vld1q_u8(crate::keep_first(nb)),
                    )
                };
            a1 = vaddq_u8(a1, vandq_u8(units!(v), keep));
        }

        start + vaddlvq_u8(merge!(a0, a1, a2, a3)) as usize + vaddvq_u32(total) as usize
    }
}
