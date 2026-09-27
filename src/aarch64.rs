//! NEON-based UTF-16 length calculation (always available on aarch64).
//!
//! Same structure as the x86_64 kernels, from napi-rs/json-escape-simd: a
//! pointer cursor and 4 unrolled vectors per iteration, counted into `u8`
//! lane accumulators. Leftover vectors and an in-register tail (overlapping
//! last-vector load, or for short inputs a page-safe over-read or stack
//! placeholder) fold into the same accumulators, which are summed once.

use std::arch::aarch64::*;

/// Lane indices 0..16, used to keep only the not-yet-counted lanes of an
/// overlapping or placeholder tail vector.
const LANE_INDEX: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// Accumulator lanes gain at most 2 per vector (a leader plus a four-byte
/// leader), so 4 accumulators merged into one stay within `u8` after 30
/// iterations, with room for 3 leftover vectors and the tail:
/// 4 * 2 * 30 + 3 * 2 + 2 = 248.
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

#[target_feature(enable = "neon")]
unsafe fn utf16_len_neon(bytes: &[u8], start: usize) -> usize {
    const LANES: usize = 16;
    const CHUNK: usize = LANES * 4;

    unsafe {
        let len = bytes.len();
        let mut sptr = bytes.as_ptr().add(start);
        let mut nb = len - start;

        let zero = vdupq_n_u8(0);
        let cont_max = vdupq_n_s8(0xBF_u8 as i8);
        let four_min = vdupq_n_u8(0xF0);

        // 0, 0xFF, or 0xFE per lane: 0xFF (-1) for a leader (not a
        // continuation byte), and -1 more for a four-byte leader. Subtracting
        // it counts units.
        macro_rules! units {
            ($v:expr) => {{
                let v = $v;
                vaddq_u8(
                    vcgtq_s8(vreinterpretq_s8_u8(v), cont_max),
                    vcgeq_u8(v, four_min),
                )
            }};
        }
        macro_rules! merge {
            ($acc:expr) => {
                vaddq_u8(vaddq_u8($acc[0], $acc[1]), vaddq_u8($acc[2], $acc[3]))
            };
        }

        // Sums of full batches, widened to u32 lanes without a horizontal add.
        let mut total = vdupq_n_u32(0);
        let mut acc = [zero; 4];
        while nb >= CHUNK {
            let batch = (nb / CHUNK).min(MAX_BATCH);
            for _ in 0..batch {
                for (j, a) in acc.iter_mut().enumerate() {
                    *a = vsubq_u8(*a, units!(vld1q_u8(sptr.add(LANES * j))));
                }
                sptr = sptr.add(CHUNK);
            }
            nb -= batch * CHUNK;
            if nb >= CHUNK {
                // Another batch follows: widen now so the u8 lanes can't overflow.
                total = vpadalq_u16(total, vpaddlq_u8(merge!(acc)));
                acc = [zero; 4];
            }
        }
        while nb >= LANES {
            acc[0] = vsubq_u8(acc[0], units!(vld1q_u8(sptr)));
            sptr = sptr.add(LANES);
            nb -= LANES;
        }
        if nb > 0 {
            // Load the vector that ends at the input's end: only its last nb lanes
            // are uncounted. Byte-wise counting needs no UTF-8 boundary care. For
            // inputs shorter than a vector, this starts before the input, and runs
            // only near the end of a page, where a forward load could fault; the
            // bytes it reads stay within the pages the input touches. Otherwise
            // short inputs load forward past their end, within the page, or copy
            // into a zeroed buffer in debug builds and Miri.
            let index = vld1q_u8(LANE_INDEX.as_ptr());
            let (v, keep) =
                if len >= LANES || (crate::OVERREAD && !crate::fits_in_page(sptr, LANES)) {
                    let v = vld1q_u8(bytes.as_ptr().wrapping_add(len).wrapping_sub(LANES));
                    (v, vcgeq_u8(index, vdupq_n_u8((LANES - nb) as u8)))
                } else {
                    let v = if crate::OVERREAD {
                        vld1q_u8(sptr)
                    } else {
                        let mut placeholder = [0u8; LANES];
                        std::ptr::copy_nonoverlapping(sptr, placeholder.as_mut_ptr(), nb);
                        vld1q_u8(placeholder.as_ptr())
                    };
                    (v, vcltq_u8(index, vdupq_n_u8(nb as u8)))
                };
            acc[1] = vsubq_u8(acc[1], vandq_u8(units!(v), keep));
        }

        start + vaddlvq_u8(merge!(acc)) as usize + vaddlvq_u32(total) as usize
    }
}
