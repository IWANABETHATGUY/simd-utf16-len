//! WASM SIMD128-based UTF-16 length calculation.

use std::arch::wasm32::*;

/// Compute the number of UTF-16 code units for UTF-8 string using WASM SIMD128.
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

/// Counts the bytes after the ASCII prefix. A function of its own, but always
/// inlined: a caller that inlines `utf16_len` gets the whole count without a
/// call, and one the inliner turns down calls `utf16_len` as before.
#[inline(always)]
fn non_ascii(bytes: &[u8], start: usize) -> usize {
    const LANES: usize = 16;
    /// A lane of either accumulator gains at most 1 per vector, so this many
    /// vectors and the tail fit before a sum.
    const MAX_BATCH: usize = 254;

    let len = bytes.len();
    // SAFETY: start < len is a verified ASCII prefix length, and each
    // full-vector load stays within the input, the placeholder, or the mask
    // table.
    unsafe {
        let mut sptr = bytes.as_ptr().add(start);
        let mut nb = len - start;

        let zero = u8x16_splat(0);
        let cont_mask = u8x16_splat(0xC0);
        let cont_val = u8x16_splat(0x80);
        let four_threshold = u8x16_splat(0xEF);
        let ones = u8x16_splat(1);
        let load = |p: *const u8| v128_load(p as *const v128);

        // Every byte contributes one unit, except continuation bytes, which
        // contribute none, and four-byte leaders, which contribute two: count
        // both kinds and adjust the byte count at the end.
        let (mut cont_acc, mut four_acc) = (zero, zero);
        macro_rules! count {
            ($v:expr, $keep:expr) => {{
                let v = $v;
                // Continuation bytes: (byte & 0xC0) == 0x80, as 0xFF lanes.
                let is_cont = u8x16_eq(v128_and(v, cont_mask), cont_val);
                // Four-byte leaders (byte >= 0xF0): saturating subtract 0xEF
                // gives non-zero only for them, then clamp to 1 with min.
                let is_four = u8x16_min(u8x16_sub_sat(v, four_threshold), ones);
                cont_acc = u8x16_sub(cont_acc, $keep(is_cont));
                four_acc = u8x16_add(four_acc, $keep(is_four));
            }};
        }

        if len < LANES {
            // The whole input is shorter than a vector: copied into a zeroed
            // placeholder, as json-escape-simd does outside Linux and macOS.
            let (v, keep) = short_vector!(sptr, nb, LANES, load);
            count!(v, |mask| v128_and(mask, keep));
            return len - horizontal_sum_u8(cont_acc) + horizontal_sum_u8(four_acc);
        }

        // Full vectors in batches, as the compiler unrolls a counted loop;
        // besides the accumulators, only the cursor, the remaining count, and
        // the running sums stay live across it.
        let all = |mask| mask;
        let (mut continuations, mut fours) = (0, 0);
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
            continuations += horizontal_sum_u8(cont_acc);
            fours += horizontal_sum_u8(four_acc);
            (cont_acc, four_acc) = (zero, zero);
        }
        if nb > 0 {
            // The vector that ends at the input's end, which holds at least a
            // vector: only its last nb lanes are uncounted, and byte-wise
            // counting needs no UTF-8 boundary care.
            let v = load(sptr.add(nb).sub(LANES));
            let keep = load(crate::keep_last(LANES, nb));
            count!(v, |mask| v128_and(mask, keep));
        }
        continuations += horizontal_sum_u8(cont_acc);
        fours += horizontal_sum_u8(four_acc);

        len - continuations + fours
    }
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
