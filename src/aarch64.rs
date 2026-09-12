//! NEON-based UTF-16 length calculation (always available on aarch64).
//!
//! Same structure as the x86_64 kernels from napi-rs/escape-simd: a pointer
//! cursor, 4 unrolled vectors per iteration with batched u8 accumulators,
//! then an in-register tail (overlapping last-vector load, or a stack
//! placeholder for short inputs) instead of a scalar fallback.

use std::arch::aarch64::*;

/// Lane indices 0..16, used to keep only the not-yet-counted lanes of an
/// overlapping or placeholder tail vector.
const LANE_INDEX: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// Compute the number of UTF-16 code units for UTF-8 string using NEON.
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

/// Lane count of a 0x00/0xFF mask vector, via the vshrn bitmask trick from
/// escape-simd's `bits.rs` (one nibble per lane) plus popcount. Counting is
/// order-insensitive, so no endianness normalization is needed.
#[inline(always)]
unsafe fn mask_count(m: uint8x16_t) -> usize {
    unsafe {
        let sr4 = vshrn_n_u16(vreinterpretq_u16_u8(m), 4);
        (vget_lane_u64(vreinterpret_u64_u8(sr4), 0).count_ones() >> 2) as usize
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
            count += batch * CHUNK - vaddlvq_u8(cont_total) as usize
                + vaddlvq_u8(four_total) as usize;
            nb -= batch * CHUNK;
        }

        while nb >= LANES {
            let v = vld1q_u8(sptr);
            count += LANES - mask_count(cont!(v)) + mask_count(four!(v));
            sptr = sptr.add(LANES);
            nb -= LANES;
        }

        if nb > 0 {
            // Cover the tail with an overlapping load of the last LANES bytes
            // (only lanes LANES - nb.. are new), or with a zeroed stack
            // placeholder when the whole input is shorter than a vector
            // (only lanes ..nb exist). Byte-wise counting needs no UTF-8
            // boundary care: ignored lanes are simply not counted.
            let (v, keep) = if len >= LANES {
                let v = vld1q_u8(bytes.as_ptr().add(len - LANES));
                let keep = vcgeq_u8(vld1q_u8(LANE_INDEX.as_ptr()), vdupq_n_u8((LANES - nb) as u8));
                (v, keep)
            } else {
                let mut placeholder = [0u8; LANES];
                std::ptr::copy_nonoverlapping(sptr, placeholder.as_mut_ptr(), nb);
                let v = vld1q_u8(placeholder.as_ptr());
                let keep = vcltq_u8(vld1q_u8(LANE_INDEX.as_ptr()), vdupq_n_u8(nb as u8));
                (v, keep)
            };
            let cont = mask_count(vandq_u8(cont!(v), keep));
            let four = mask_count(vandq_u8(four!(v), keep));
            count += nb - cont + four;
        }

        count
    }
}
