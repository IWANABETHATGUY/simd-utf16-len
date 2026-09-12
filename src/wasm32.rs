//! WASM SIMD128-based UTF-16 length calculation.
//!
//! Same structure as the x86_64/aarch64 kernels: a pointer cursor, 4 unrolled
//! vectors per iteration with batched u8 accumulators, then an in-register
//! tail via `u8x16_bitmask` instead of a scalar fallback.

use std::arch::wasm32::*;

/// Compute the number of UTF-16 code units for UTF-8 string using WASM SIMD128.
pub fn utf16_len(s: &str) -> usize {
    const LANES: usize = 16;
    const CHUNK: usize = LANES * 4;

    let bytes = s.as_bytes();
    let len = bytes.len();
    let start = crate::ascii::ascii_prefix_len(bytes);
    if start == len {
        return start;
    }

    // SAFETY: start <= len was verified above.
    let mut sptr = unsafe { bytes.as_ptr().add(start) };
    let mut nb = len - start;
    let mut count = start;

    let cont_mask = u8x16_splat(0xC0);
    let cont_val = u8x16_splat(0x80);
    let four_mask = u8x16_splat(0xF0);

    macro_rules! cont {
        ($v:expr) => {
            u8x16_eq(v128_and($v, cont_mask), cont_val)
        };
    }
    macro_rules! four {
        ($v:expr) => {
            u8x16_eq(v128_and($v, four_mask), four_mask)
        };
    }

    // u8 lane accumulators overflow after 255 increments; batches are capped
    // at 63 iterations so each lane stays <= 63 and the 4 accumulators can be
    // merged (<= 252) before one horizontal sum.
    while nb >= CHUNK {
        let batch = (nb / CHUNK).min(63);
        let mut cont_acc = [u8x16_splat(0); 4];
        let mut four_acc = [u8x16_splat(0); 4];
        for _ in 0..batch {
            for (j, acc) in cont_acc.iter_mut().zip(four_acc.iter_mut()).enumerate() {
                // SAFETY: nb >= CHUNK, so all 4 vector loads stay in bounds.
                let v = unsafe { v128_load(sptr.add(LANES * j) as *const v128) };
                *acc.0 = u8x16_sub(*acc.0, cont!(v));
                *acc.1 = u8x16_sub(*acc.1, four!(v));
            }
            // SAFETY: see above.
            sptr = unsafe { sptr.add(CHUNK) };
        }
        let cont_total = u8x16_add(
            u8x16_add(cont_acc[0], cont_acc[1]),
            u8x16_add(cont_acc[2], cont_acc[3]),
        );
        let four_total = u8x16_add(
            u8x16_add(four_acc[0], four_acc[1]),
            u8x16_add(four_acc[2], four_acc[3]),
        );
        count += batch * CHUNK - horizontal_sum_u8(cont_total) + horizontal_sum_u8(four_total);
        nb -= batch * CHUNK;
    }

    while nb >= LANES {
        // SAFETY: nb >= LANES, so the load stays in bounds.
        let v = unsafe { v128_load(sptr as *const v128) };
        count += LANES - (u8x16_bitmask(cont!(v)).count_ones() as usize)
            + (u8x16_bitmask(four!(v)).count_ones() as usize);
        // SAFETY: see above.
        sptr = unsafe { sptr.add(LANES) };
        nb -= LANES;
    }

    if nb > 0 {
        // Cover the tail with an overlapping load of the last LANES bytes
        // (only lanes LANES - nb.. are new), or with a zeroed stack
        // placeholder when the whole input is shorter than a vector (only
        // lanes ..nb exist). Byte-wise counting needs no UTF-8 boundary care.
        let (cont_bits, four_bits) = if len >= LANES {
            // SAFETY: len >= LANES, so the last-vector load stays in bounds.
            let v = unsafe { v128_load(bytes.as_ptr().add(len - LANES) as *const v128) };
            let shift = (LANES - nb) as u32;
            (
                u32::from(u8x16_bitmask(cont!(v))) >> shift,
                u32::from(u8x16_bitmask(four!(v))) >> shift,
            )
        } else {
            let mut placeholder = [0u8; LANES];
            // SAFETY: nb < LANES, and sptr has nb readable bytes.
            unsafe { std::ptr::copy_nonoverlapping(sptr, placeholder.as_mut_ptr(), nb) };
            let v = unsafe { v128_load(placeholder.as_ptr() as *const v128) };
            let keep = (1u32 << nb) - 1;
            (
                u32::from(u8x16_bitmask(cont!(v))) & keep,
                u32::from(u8x16_bitmask(four!(v))) & keep,
            )
        };
        count += nb - cont_bits.count_ones() as usize + four_bits.count_ones() as usize;
    }

    count
}

/// Horizontal sum of all u8 lanes in a v128 register.
#[inline(always)]
fn horizontal_sum_u8(v: v128) -> usize {
    // u8x16 -> i16x8 (pairwise add adjacent u8 lanes)
    let pairs = i16x8_extadd_pairwise_u8x16(v);
    // i16x8 -> i32x4 (pairwise add adjacent i16 lanes)
    let quads = i32x4_extadd_pairwise_i16x8(pairs);
    // Sum the 4 i32 lanes.
    (i32x4_extract_lane::<0>(quads)
        + i32x4_extract_lane::<1>(quads)
        + i32x4_extract_lane::<2>(quads)
        + i32x4_extract_lane::<3>(quads)) as usize
}
