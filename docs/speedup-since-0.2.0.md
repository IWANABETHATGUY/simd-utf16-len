# Speedup since 0.2.0

How much faster `utf16_len` on `main` (commit `180439b`, 2026-09-28) is than the upstream 0.2.0 release (commit `852d444`), measured by the crate's Perf A/B on GitHub-hosted runners.

## Method

[PR #39](https://github.com/IWANABETHATGUY/simd-utf16-len/pull/39), never to be merged, carries the `src/` of 0.2.0 with today's benchmarks, harness, and workflows, so its Perf A/B measures `main` as the base and 0.2.0 as the head, on the same sixteen inputs, in the same build. Each input's two sides run back to back in fresh processes, alternating which goes first, seven times; the harness reports the median time per call of each side. The speedup below is 0.2.0's median time divided by `main`'s. Three runs sampled five CPU and OS combinations: runs [36438011025](https://github.com/IWANABETHATGUY/simd-utf16-len/actions/runs/36438011025), [36438016000](https://github.com/IWANABETHATGUY/simd-utf16-len/actions/runs/36438016000), and [36438010008](https://github.com/IWANABETHATGUY/simd-utf16-len/actions/runs/36438010008), with Rust 1.98.1.

The macOS runner is an M1 virtual machine whose timings vary by tens of percent between runs, so its column gives the range over the three runs. The x86 columns come from one run each; the two Linux Zen 3 samples agreed to the tenth of a nanosecond.

## Summary

| Input | Bytes | Linux, Zen 3 (EPYC 7763) | Linux, Emerald Rapids (Xeon 8573C) | Windows, Zen 3 (EPYC 7763) | Windows, Emerald Rapids (Xeon 8573C) | Windows, Zen 4 (EPYC 9V45) | macOS, Apple M1 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ascii | 169 | 1.7x | 1.6x | 2.1x | 2.1x | 1.7x | 1.2 to 1.4x |
| cjk | 194 | 1.6x | 1.8x | 1.4x | 1.4x | 1.5x | 1.5 to 1.6x |
| emoji | 170 | 2.6x | 2.9x | 1.9x | 2.1x | 2.0x | 1.9 to 2.5x |
| mixed | 144 | 1.5x | 1.6x | 1.3x | 1.2x | 1.2x | 1.3 to 1.4x |
| cjk_tail3 | 195 | 1.7x | 2.1x | 1.5x | 1.5x | 1.7x | 1.6 to 1.8x |
| cjk_tail15 | 207 | 2.4x | 2.9x | 2.1x | 2.1x | 2.2x | 2.4 to 3.1x |
| ascii_large | 10816 | 1.9x | 1.6x | 1.8x | 1.5x | 1.4x | 0.9 to 1.1x |
| cjk_large | 12416 | 2.5x | 2.6x | 2.5x | 2.7x | 3.5x | 1.7 to 2.1x |
| emoji_large | 10880 | 2.5x | 2.5x | 2.5x | 2.6x | 3.5x | 1.7 to 2.0x |
| mixed_large | 9216 | 2.5x | 2.7x | 2.5x | 2.5x | 3.5x | 2.0x |
| early_non_ascii | 10818 | 2.5x | 2.5x | 2.5x | 2.5x | 3.5x | 1.9 to 2.0x |
| late_non_ascii | 10818 | 2.9x | 2.3x | 1.9x | 1.7x | 1.5x | 1.1 to 1.4x |
| ascii_tiny | 11 | 1.0x | 1.0x | 1.4x | 1.1x | 1.8x | 1.2x |
| utf8_tiny | 13 | 3.7x | 3.7x | 2.8x | 3.0x | 2.4x | 5.5 to 7.5x |
| ascii_short | 44 | 1.0x | 1.0x | 1.2x | 1.2x | 1.0x | 1.0 to 1.2x |
| utf8_short | 53 | 1.6x | 2.1x | 1.5x | 1.6x | 1.4x | 2.9 to 3.1x |

## Readings

- The long non-ASCII inputs gain the most on x86: 2.5x on Zen 3 and Emerald Rapids, 3.5x on Zen 4, from the AVX2 kernel's nibble-table lookup and the ASCII skip in front of it.
- The 13-byte non-ASCII input gains the most in relative terms, 2.4 to 7.5x, from the in-register tail and the SSE2 and NEON short paths inlined into the dispatch.
- Pure ASCII under 64 bytes is unchanged on Linux: those inputs still take the standard library's word-at-a-time path. The Windows gains there come from the entry being inlined into callers on Windows and macOS.
- macOS gains less on the long inputs, since 0.2.0's NEON code was already closer to the M1's limits.

## Per-job tables

Time per call in nanoseconds, median of 7 runs; speedup is the 0.2.0 column divided by the `main` column.

### Linux x86_64, Intel Xeon Platinum 8573C (Emerald Rapids), run 36438011025

| Input | Bytes | `main` ns | 0.2.0 ns | Speedup |
|:--|--:|--:|--:|--:|
| ascii | 169 | 2.8 | 4.5 | 1.61x |
| cjk | 194 | 6.4 | 11.5 | 1.80x |
| emoji | 170 | 5.8 | 16.8 | 2.90x |
| mixed | 144 | 5.2 | 8.3 | 1.60x |
| cjk_tail3 | 195 | 6.1 | 12.6 | 2.07x |
| cjk_tail15 | 207 | 6.4 | 18.3 | 2.86x |
| ascii_large | 10816 | 80.5 | 131.3 | 1.63x |
| cjk_large | 12416 | 154.6 | 408.8 | 2.64x |
| emoji_large | 10880 | 144.6 | 357.5 | 2.47x |
| mixed_large | 9216 | 114.9 | 309.6 | 2.69x |
| early_non_ascii | 10818 | 144.6 | 361.7 | 2.50x |
| late_non_ascii | 10818 | 59.4 | 133.6 | 2.25x |
| ascii_tiny | 11 | 2.3 | 2.2 | 0.96x |
| utf8_tiny | 13 | 3.1 | 11.5 | 3.71x |
| ascii_short | 44 | 4.4 | 4.5 | 1.02x |
| utf8_short | 53 | 4.9 | 10.4 | 2.12x |

### Windows x86_64, AMD EPYC 7763 (Zen 3), run 36438011025

| Input | Bytes | `main` ns | 0.2.0 ns | Speedup |
|:--|--:|--:|--:|--:|
| ascii | 169 | 2.8 | 6.0 | 2.14x |
| cjk | 194 | 8.8 | 12.2 | 1.39x |
| emoji | 170 | 8.3 | 15.7 | 1.89x |
| mixed | 144 | 7.2 | 9.5 | 1.32x |
| cjk_tail3 | 195 | 8.8 | 13.2 | 1.50x |
| cjk_tail15 | 207 | 8.8 | 18.2 | 2.07x |
| ascii_large | 10816 | 91.3 | 167.7 | 1.84x |
| cjk_large | 12416 | 158.2 | 396.2 | 2.50x |
| emoji_large | 10880 | 139.8 | 346.0 | 2.47x |
| mixed_large | 9216 | 119.2 | 297.8 | 2.50x |
| early_non_ascii | 10818 | 139.8 | 346.8 | 2.48x |
| late_non_ascii | 10818 | 88.6 | 171.0 | 1.93x |
| ascii_tiny | 11 | 2.2 | 3.1 | 1.41x |
| utf8_tiny | 13 | 4.4 | 12.4 | 2.82x |
| ascii_short | 44 | 3.8 | 4.7 | 1.24x |
| utf8_short | 53 | 6.3 | 9.3 | 1.48x |

### macOS aarch64, Apple M1 (virtual), run 36438011025

| Input | Bytes | `main` ns | 0.2.0 ns | Speedup |
|:--|--:|--:|--:|--:|
| ascii | 169 | 5.0 | 6.2 | 1.24x |
| cjk | 194 | 6.8 | 10.6 | 1.56x |
| emoji | 170 | 8.1 | 16.5 | 2.04x |
| mixed | 144 | 6.3 | 8.5 | 1.35x |
| cjk_tail3 | 195 | 6.6 | 11.7 | 1.77x |
| cjk_tail15 | 207 | 6.7 | 20.8 | 3.10x |
| ascii_large | 10816 | 177.9 | 169.3 | 0.95x |
| cjk_large | 12416 | 358.2 | 592.6 | 1.65x |
| emoji_large | 10880 | 256.7 | 524.7 | 2.04x |
| mixed_large | 9216 | 211.0 | 426.6 | 2.02x |
| early_non_ascii | 10818 | 223.0 | 444.7 | 1.99x |
| late_non_ascii | 10818 | 132.4 | 185.7 | 1.40x |
| ascii_tiny | 11 | 2.8 | 3.3 | 1.18x |
| utf8_tiny | 13 | 2.2 | 12.1 | 5.50x |
| ascii_short | 44 | 5.3 | 5.1 | 0.96x |
| utf8_short | 53 | 5.2 | 14.9 | 2.87x |

### Linux x86_64, AMD EPYC 7763 (Zen 3), run 36438016000

| Input | Bytes | `main` ns | 0.2.0 ns | Speedup |
|:--|--:|--:|--:|--:|
| ascii | 169 | 3.4 | 5.6 | 1.65x |
| cjk | 194 | 7.5 | 11.8 | 1.57x |
| emoji | 170 | 7.2 | 18.6 | 2.58x |
| mixed | 144 | 5.9 | 8.7 | 1.47x |
| cjk_tail3 | 195 | 7.5 | 12.8 | 1.71x |
| cjk_tail15 | 207 | 7.5 | 17.9 | 2.39x |
| ascii_large | 10816 | 86.4 | 166.3 | 1.92x |
| cjk_large | 12416 | 156.5 | 392.9 | 2.51x |
| emoji_large | 10880 | 137.9 | 342.8 | 2.49x |
| mixed_large | 9216 | 118.1 | 294.5 | 2.49x |
| early_non_ascii | 10818 | 137.8 | 342.5 | 2.49x |
| late_non_ascii | 10818 | 58.9 | 169.0 | 2.87x |
| ascii_tiny | 11 | 2.8 | 2.8 | 1.00x |
| utf8_tiny | 13 | 3.4 | 12.6 | 3.71x |
| ascii_short | 44 | 4.4 | 4.4 | 1.00x |
| utf8_short | 53 | 6.2 | 9.6 | 1.55x |

### Windows x86_64, Intel Xeon Platinum 8573C (Emerald Rapids), run 36438016000

| Input | Bytes | `main` ns | 0.2.0 ns | Speedup |
|:--|--:|--:|--:|--:|
| ascii | 169 | 2.5 | 5.2 | 2.08x |
| cjk | 194 | 8.4 | 11.6 | 1.38x |
| emoji | 170 | 7.7 | 16.0 | 2.08x |
| mixed | 144 | 6.3 | 7.8 | 1.24x |
| cjk_tail3 | 195 | 8.3 | 12.2 | 1.47x |
| cjk_tail15 | 207 | 8.3 | 17.8 | 2.14x |
| ascii_large | 10816 | 84.5 | 124.7 | 1.48x |
| cjk_large | 12416 | 164.3 | 448.6 | 2.73x |
| emoji_large | 10880 | 152.1 | 393.0 | 2.58x |
| mixed_large | 9216 | 125.6 | 318.9 | 2.54x |
| early_non_ascii | 10818 | 157.0 | 385.0 | 2.45x |
| late_non_ascii | 10818 | 81.0 | 134.6 | 1.66x |
| ascii_tiny | 11 | 2.2 | 2.4 | 1.09x |
| utf8_tiny | 13 | 3.9 | 11.5 | 2.95x |
| ascii_short | 44 | 3.9 | 4.5 | 1.15x |
| utf8_short | 53 | 5.7 | 9.2 | 1.61x |

### macOS aarch64, Apple M1 (virtual), run 36438016000

| Input | Bytes | `main` ns | 0.2.0 ns | Speedup |
|:--|--:|--:|--:|--:|
| ascii | 169 | 4.8 | 6.5 | 1.35x |
| cjk | 194 | 7.8 | 12.2 | 1.56x |
| emoji | 170 | 7.9 | 19.5 | 2.47x |
| mixed | 144 | 7.7 | 10.3 | 1.34x |
| cjk_tail3 | 195 | 8.5 | 14.2 | 1.67x |
| cjk_tail15 | 207 | 8.9 | 21.0 | 2.36x |
| ascii_large | 10816 | 159.1 | 160.4 | 1.01x |
| cjk_large | 12416 | 300.6 | 578.6 | 1.92x |
| emoji_large | 10880 | 266.7 | 464.2 | 1.74x |
| mixed_large | 9216 | 217.6 | 440.4 | 2.02x |
| early_non_ascii | 10818 | 262.1 | 487.4 | 1.86x |
| late_non_ascii | 10818 | 145.0 | 163.8 | 1.13x |
| ascii_tiny | 11 | 3.1 | 3.7 | 1.19x |
| utf8_tiny | 13 | 2.2 | 16.4 | 7.45x |
| ascii_short | 44 | 5.1 | 6.3 | 1.24x |
| utf8_short | 53 | 4.4 | 13.8 | 3.14x |

### Linux x86_64, AMD EPYC 7763 (Zen 3), run 36438010008

| Input | Bytes | `main` ns | 0.2.0 ns | Speedup |
|:--|--:|--:|--:|--:|
| ascii | 169 | 3.4 | 5.6 | 1.65x |
| cjk | 194 | 7.5 | 11.8 | 1.57x |
| emoji | 170 | 7.2 | 18.5 | 2.57x |
| mixed | 144 | 5.9 | 8.7 | 1.47x |
| cjk_tail3 | 195 | 7.5 | 12.8 | 1.71x |
| cjk_tail15 | 207 | 7.5 | 17.9 | 2.39x |
| ascii_large | 10816 | 86.4 | 166.3 | 1.92x |
| cjk_large | 12416 | 156.5 | 393.0 | 2.51x |
| emoji_large | 10880 | 138.0 | 342.9 | 2.48x |
| mixed_large | 9216 | 118.2 | 294.7 | 2.49x |
| early_non_ascii | 10818 | 137.8 | 342.4 | 2.48x |
| late_non_ascii | 10818 | 58.9 | 169.1 | 2.87x |
| ascii_tiny | 11 | 2.8 | 2.8 | 1.00x |
| utf8_tiny | 13 | 3.4 | 12.8 | 3.76x |
| ascii_short | 44 | 4.4 | 4.4 | 1.00x |
| utf8_short | 53 | 6.2 | 9.6 | 1.55x |

### Windows x86_64, AMD EPYC 9V45 (Zen 4), run 36438010008

| Input | Bytes | `main` ns | 0.2.0 ns | Speedup |
|:--|--:|--:|--:|--:|
| ascii | 169 | 1.6 | 2.7 | 1.69x |
| cjk | 194 | 4.8 | 7.3 | 1.52x |
| emoji | 170 | 4.5 | 8.8 | 1.96x |
| mixed | 144 | 4.3 | 5.1 | 1.19x |
| cjk_tail3 | 195 | 4.7 | 8.0 | 1.70x |
| cjk_tail15 | 207 | 4.8 | 10.6 | 2.21x |
| ascii_large | 10816 | 60.4 | 87.3 | 1.45x |
| cjk_large | 12416 | 102.7 | 356.7 | 3.47x |
| emoji_large | 10880 | 90.6 | 319.0 | 3.52x |
| mixed_large | 9216 | 76.8 | 270.6 | 3.52x |
| early_non_ascii | 10818 | 90.7 | 321.5 | 3.54x |
| late_non_ascii | 10818 | 59.2 | 90.5 | 1.53x |
| ascii_tiny | 11 | 1.2 | 2.1 | 1.75x |
| utf8_tiny | 13 | 2.2 | 5.3 | 2.41x |
| ascii_short | 44 | 2.1 | 2.1 | 1.00x |
| utf8_short | 53 | 3.1 | 4.3 | 1.39x |

### macOS aarch64, Apple M1 (virtual), run 36438010008

| Input | Bytes | `main` ns | 0.2.0 ns | Speedup |
|:--|--:|--:|--:|--:|
| ascii | 169 | 3.8 | 5.1 | 1.34x |
| cjk | 194 | 6.3 | 9.4 | 1.49x |
| emoji | 170 | 7.2 | 13.8 | 1.92x |
| mixed | 144 | 6.3 | 8.6 | 1.37x |
| cjk_tail3 | 195 | 6.3 | 10.0 | 1.59x |
| cjk_tail15 | 207 | 6.4 | 15.6 | 2.44x |
| ascii_large | 10816 | 124.8 | 139.1 | 1.11x |
| cjk_large | 12416 | 231.9 | 492.3 | 2.12x |
| emoji_large | 10880 | 201.6 | 395.5 | 1.96x |
| mixed_large | 9216 | 166.6 | 331.3 | 1.99x |
| early_non_ascii | 10818 | 199.3 | 392.9 | 1.97x |
| late_non_ascii | 10818 | 117.8 | 128.8 | 1.09x |
| ascii_tiny | 11 | 2.1 | 2.5 | 1.19x |
| utf8_tiny | 13 | 1.8 | 11.3 | 6.28x |
| ascii_short | 44 | 4.2 | 4.4 | 1.05x |
| utf8_short | 53 | 3.6 | 11.3 | 3.14x |
