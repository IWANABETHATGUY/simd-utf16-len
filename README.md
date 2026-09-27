# simd-utf16-len

SIMD-accelerated UTF-16 length calculation from UTF-8 strings, with dedicated ASCII fast paths and no runtime dependencies.

## Usage

```rust
use simd_utf16_len::utf16_len;

assert_eq!(utf16_len("Hello"), 5);
assert_eq!(utf16_len("Hello, 世界! 🌍"), 13);
```

The result counts UTF-16 code units. Characters outside the Basic Multilingual Plane, such as `🌍`, contribute two units.

## How it works

Computing the UTF-16 length of a UTF-8 string doesn't require actually encoding it. The length can be derived directly from byte patterns:

```text
utf16_len = byte_length - continuation_bytes + four_byte_leaders
```

Where:

- **Continuation bytes** (`(byte & 0xC0) == 0x80`) don't produce UTF-16 code units
- **Four-byte leaders** (`byte >= 0xF0`) produce surrogate pairs (2 UTF-16 code units instead of 1)

The SIMD implementations first scan for an ASCII prefix. Entirely ASCII strings return their byte length; otherwise, the verified prefix contributes its byte length and the remaining bytes are counted using 16-byte SIMD vectors. The ASCII scans follow Rust's standard-library strategy at commit [`4aa1fbc`](https://github.com/rust-lang/rust/blob/4aa1fbcf467cf38ce58abfa8eb9213a789c5381c/library/core/src/slice/ascii.rs): x86_64 uses 64-byte SSE2 blocks, and aarch64 uses 64-byte NEON blocks with a 16-byte vector tail. Both use word-sized checks below 64 bytes. wasm32 uses aligned `usize` loads between unaligned first and last words. The adaptation returns the verified prefix length instead of a boolean; it uses the same block sizes, loads, and tail checks.

Call `utf16_len(s)` directly when the ASCII status is unknown. If the caller already guarantees or caches that a string is ASCII, `s.len()` remains an O(1) operation and avoids scanning altogether.

## Platform support

| Architecture | SIMD | Instruction set |
|-------------|------|-----------------|
| x86_64 | SSE2 | Available by default on this architecture |
| aarch64 | NEON | Available by default on this architecture |
| wasm32 | simd128 | Requires `target_feature = "simd128"` |
| Other | — | Falls back to `encode_utf16().count()` |

## Benchmarks

The [Perf A/B workflow](.github/workflows/perf-ab.yml) also compares `utf16_len` with this standard-library baseline, which skips counting for ASCII input:

```rust
fn std_guard_len(s: &str) -> usize {
    if s.is_ascii() {
        s.len()
    } else {
        s.encode_utf16().count()
    }
}
```

The speedups below are from [run 36311375947](https://github.com/IWANABETHATGUY/simd-utf16-len/actions/runs/36311375947) on **2026-09-27**, at commit `ace7793`, using Rust **1.98.1**. Speedup is the baseline's time divided by `utf16_len`'s time, taken as the median of 7 runs in fresh processes, so above 1x means `utf16_len` is faster. The job summaries also list the time per call. The runners were an AMD EPYC 7763 ([x86_64 job](https://github.com/IWANABETHATGUY/simd-utf16-len/actions/runs/36311375947/job/108597838579)), a Neoverse-N2 ([aarch64 job](https://github.com/IWANABETHATGUY/simd-utf16-len/actions/runs/36311375947/job/108597838670)), and an Apple M1 (Virtual) ([macOS job](https://github.com/IWANABETHATGUY/simd-utf16-len/actions/runs/36311375947/job/108597838445)).

| Input | Bytes | Linux x86_64 | Linux aarch64 | macOS aarch64 |
|:------|------:|-------------:|--------------:|--------------:|
| ascii_tiny | 11 | 1.0x | 0.7x | 0.8x |
| utf8_tiny | 13 | 0.7x | 0.8x | 0.9x |
| ascii_short | 44 | 0.7x | 0.8x | 0.9x |
| utf8_short | 53 | 2.7x | 3.0x | 3.1x |
| ascii | 169 | 6.6x | 1.7x | 2.1x |
| cjk | 194 | 7.9x | 8.3x | 9.9x |
| emoji | 170 | 6.4x | 6.1x | 7.4x |
| mixed | 144 | 9.8x | 6.9x | 11.6x |
| cjk_tail3 | 195 | 7.4x | 7.5x | 8.0x |
| cjk_tail15 | 207 | 5.5x | 5.4x | 5.8x |
| ascii_large | 10816 | 0.7x | 2.1x | 4.1x |
| cjk_large | 12416 | 15.4x | 12.3x | 13.0x |
| emoji_large | 10880 | 19.2x | 12.0x | 16.5x |
| mixed_large | 9216 | 18.0x | 11.5x | 17.5x |
| early_non_ascii | 10818 | 21.6x | 15.6x | 15.6x |
| late_non_ascii | 10818 | 40.5x | 35.7x | 49.9x |

The baseline ties or wins on the inputs of 44 bytes or less (`ascii_tiny`, `utf8_tiny`, and `ascii_short`). On x86_64, its `is_ascii` also beats this crate's ASCII scan on the 10,816-byte ASCII input. Results depend on input length, character distribution, CPU, and compiler, so these ratios don't promise a speedup for every string or platform.

### Reproduce locally

```sh
scripts/perf-ab.sh HEAD
```

With `HEAD` as the base, the first table shows the no-change spread on your machine, and the second compares your working tree with the standard library.

### Base-vs-PR check

The [Perf A/B workflow](.github/workflows/perf-ab.yml) builds `utf16_len` from the base commit and from the change into one binary, then times them in alternating batches on the same machine, so runner noise affects both sides equally. It repeats this in 7 fresh processes and uses the median, because where the code lands in memory can shift one process's result for short inputs. It runs for pull requests and pushes to `main` on Linux x86_64, Windows x86_64, and macOS aarch64, and writes the median change per input to the job summary. It fails when an input's median is more than 5% slower and every run agrees that it's slower. To accept an intended slowdown, label the pull request `perf-regression-accepted`.

Run the same comparison locally against any ref:

```sh
scripts/perf-ab.sh main
```

### CodSpeed regression tracking

The separate [CodSpeed workflow](.github/workflows/codspeed.yml) runs the [benchmark suite](benches/utf16_len.rs) in **Simulation** mode by default for pushes, pull requests, and manual runs. Its first 9 cases cover long ASCII (10,816 bytes), CJK, emoji, and mixed text; they compare SIMD with `encode_utf16().count()` and include the ASCII guard for the ASCII input. The long ASCII fixture has a separate benchmark identity from the historical 169-byte fixture, so changing the input size is not reported as a code regression; Unicode benchmark identities remain unchanged.

The `code_path` group adds 12 cases that time `utf16_len` alone on inputs chosen by the code path they reach: under 16 bytes, under 64 bytes, 3 or 15 bytes left after the last 16-byte vector, text longer than one 4,080-byte batch, and long ASCII with one non-ASCII character at the start or end. All benchmark inputs live in [`benches/inputs.rs`](benches/inputs.rs), which the base-vs-PR check also uses.

Use the [CodSpeed dashboard](https://app.codspeed.io/SyMind/simd-utf16-len) to track changes across commits and inspect flamegraphs. Simulation results represent modeled execution costs and are distinct from the native timings above. The workflow also supports **Walltime** mode through its manual `mode` input to measure actual elapsed time.

The [CI workflow](.github/workflows/ci.yml) runs `cargo test` on Linux, macOS, and Windows to check correctness. It also runs the unit tests for wasm32 under wasmtime, both with `simd128` and with the scalar fallback.

## License

MIT
