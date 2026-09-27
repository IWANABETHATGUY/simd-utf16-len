//! NEON-based UTF-16 length calculation (always available on aarch64).
//!
//! Same structure as the x86_64 kernels from napi-rs/escape-simd: a pointer
//! cursor, 4 unrolled vectors per iteration with batched u8 accumulators,
//! then an in-register tail (overlapping last-vector load, or for short
//! inputs a page-safe over-read or stack placeholder) instead of a scalar
//! fallback.

use std::arch::aarch64::*;

/// Lane indices 0..16, used to keep only the not-yet-counted lanes of an
/// overlapping or placeholder tail vector.
const LANE_INDEX: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

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
        let mut count = start;

        let cont_mask = vdupq_n_u8(0xC0);
        let cont_val = vdupq_n_u8(0x80);
        let four_mask = vdupq_n_u8(0xF0);

        macro_rules! cont {
            ($v:expr) => {
                vceqq_u8(vandq_u8($v, cont_mask), cont_val)
            };
        }
        macro_rules! four {
            ($v:expr) => {
                vceqq_u8(vandq_u8($v, four_mask), four_mask)
            };
        }

        // u8 lane accumulators overflow after 255 increments; batches are
        // capped at 63 iterations so each lane stays <= 63 and the 4
        // accumulators can be merged (<= 252) before one horizontal sum.
        while nb >= CHUNK {
            let batch = (nb / CHUNK).min(63);
            let mut cont_acc = [vdupq_n_u8(0); 4];
            let mut four_acc = [vdupq_n_u8(0); 4];
            for _ in 0..batch {
                for (j, acc) in cont_acc.iter_mut().zip(four_acc.iter_mut()).enumerate() {
                    let v = vld1q_u8(sptr.add(LANES * j));
                    *acc.0 = vsubq_u8(*acc.0, cont!(v));
                    *acc.1 = vsubq_u8(*acc.1, four!(v));
                }
                sptr = sptr.add(CHUNK);
            }
            let cont_total = vaddq_u8(
                vaddq_u8(cont_acc[0], cont_acc[1]),
                vaddq_u8(cont_acc[2], cont_acc[3]),
            );
            let four_total = vaddq_u8(
                vaddq_u8(four_acc[0], four_acc[1]),
                vaddq_u8(four_acc[2], four_acc[3]),
            );
            count +=
                batch * CHUNK - vaddlvq_u8(cont_total) as usize + vaddlvq_u8(four_total) as usize;
            nb -= batch * CHUNK;
        }

        // Up to 3 leftover vectors: accumulate in vector registers and extract
        // once, instead of paying a vshrn+popcnt dependency chain per vector.
        if nb >= LANES {
            let mut cont_acc = vdupq_n_u8(0);
            let mut four_acc = vdupq_n_u8(0);
            let mut vectors = 0usize;
            while nb >= LANES {
                let v = vld1q_u8(sptr);
                cont_acc = vsubq_u8(cont_acc, cont!(v));
                four_acc = vsubq_u8(four_acc, four!(v));
                vectors += 1;
                sptr = sptr.add(LANES);
                nb -= LANES;
            }
            count +=
                vectors * LANES - vaddlvq_u8(cont_acc) as usize + vaddlvq_u8(four_acc) as usize;
        }

        if nb > 0 {
            // Cover the tail with an overlapping load of the last LANES bytes
            // (only lanes LANES - nb.. are new). When the whole input is
            // shorter than a vector (only lanes ..nb exist), load past its end
            // if that stays within the page, else from a zeroed stack
            // placeholder. Byte-wise counting needs no UTF-8 boundary care:
            // ignored lanes are simply not counted.
            let (v, keep) = if len >= LANES {
                let v = vld1q_u8(bytes.as_ptr().add(len - LANES));
                let keep = vcgeq_u8(
                    vld1q_u8(LANE_INDEX.as_ptr()),
                    vdupq_n_u8((LANES - nb) as u8),
                );
                (v, keep)
            } else {
                let v = if crate::can_overread(sptr, LANES) {
                    vld1q_u8(sptr)
                } else {
                    let mut placeholder = [0u8; LANES];
                    std::ptr::copy_nonoverlapping(sptr, placeholder.as_mut_ptr(), nb);
                    vld1q_u8(placeholder.as_ptr())
                };
                let keep = vcltq_u8(vld1q_u8(LANE_INDEX.as_ptr()), vdupq_n_u8(nb as u8));
                (v, keep)
            };
            // UTF-16 units per kept lane: 1 - is_cont + is_four (0 elsewhere);
            // one horizontal sum for the whole tail. `keep01` turns the 0xFF
            // compare masks into 0/1 values for the arithmetic.
            let keep01 = vandq_u8(keep, vdupq_n_u8(1));
            let contrib = vsubq_u8(
                vaddq_u8(keep01, vandq_u8(four!(v), keep01)),
                vandq_u8(cont!(v), keep01),
            );
            count += vaddlvq_u8(contrib) as usize;
        }

        count
    }
}
