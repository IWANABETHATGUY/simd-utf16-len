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

The SIMD implementations first scan for an ASCII prefix. Entirely ASCII strings return their byte length; otherwise, the verified prefix contributes its byte length and the remaining bytes are counted with SIMD kernels modeled on [napi-rs/json-escape-simd](https://github.com/napi-rs/json-escape-simd): a pointer cursor with a remaining-byte count, four unrolled vectors per iteration on AVX2, AVX-512, NEON, and simd128 counted into byte-lane accumulators, and one horizontal sum per call. AVX2, AVX-512, NEON, and simd128 look up each byte's UTF-16 units by its high nibble with a byte shuffle, the way json-escape-simd's nibble-table classifier works; SSE2 has no byte shuffle, so it compares. The last bytes come from an overlapping load of the final vector, masked with a static lane table, or from a fault-suppressing masked load under AVX-512. Inputs shorter than one vector are read with one full-vector load that stays within their memory page, past their end or, at the end of a page, past their start, in release builds on Linux, macOS, and Windows, instead of copying into a buffer. On x86_64, inputs with fewer than 64 bytes after the prefix stay on the SSE2 kernel inlined into the dispatch, since a wider kernel's call and reduction cost more than they save there, and the AVX-512 kernel takes over from 256 bytes. The ASCII scans follow Rust's standard-library strategy at commit [`4aa1fbc`](https://github.com/rust-lang/rust/blob/4aa1fbcf467cf38ce58abfa8eb9213a789c5381c/library/core/src/slice/ascii.rs): x86_64 uses 64-byte SSE2 blocks, and aarch64 uses 64-byte NEON blocks with a 16-byte vector tail. Both use word-sized checks below 64 bytes. wasm32 uses aligned `usize` loads between unaligned first and last words. The adaptation returns the verified prefix length instead of a boolean; it uses the same block sizes, loads, and tail checks.

Call `utf16_len(s)` directly when the ASCII status is unknown. If the caller already guarantees or caches that a string is ASCII, `s.len()` remains an O(1) operation and avoids scanning altogether.

## Platform support

| Architecture | SIMD | Instruction set |
|-------------|------|-----------------|
| x86_64 | AVX2 / SSE2, or AVX-512BW with the `avx512` feature | Runtime dispatch; SSE2 available by default on this architecture |
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

The speedups below are from [run 36421345289](https://github.com/IWANABETHATGUY/simd-utf16-len/actions/runs/36421345289) on **2026-09-28**, at commit `cc0add1`, using Rust **1.98.1**. Speedup is the baseline's time divided by `utf16_len`'s time, taken as the median of 7 runs in fresh processes, so above 1x means `utf16_len` is faster. The job summaries also list the time per call. The runners were an Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz ([Linux x86_64 job](https://github.com/IWANABETHATGUY/simd-utf16-len/actions/runs/36421345289/job/108924568086)), an AMD EPYC 9V45 96-Core Processor ([Windows x86_64 job](https://github.com/IWANABETHATGUY/simd-utf16-len/actions/runs/36421345289/job/108924568049)), an Apple M1 (Virtual) ([macOS aarch64 job](https://github.com/IWANABETHATGUY/simd-utf16-len/actions/runs/36421345289/job/108924567797)).

| Input | Bytes | Linux x86_64 | Windows x86_64 | macOS aarch64 |
|:------|------:|-------------:|---------------:|--------------:|
| ascii | 169 | 3.4x | 5.3x | 2.2x |
| cjk | 194 | 11.1x | 11.0x | 14.1x |
| emoji | 170 | 11.3x | 11.8x | 15.1x |
| mixed | 144 | 11.8x | 8.4x | 15.8x |
| cjk_tail3 | 195 | 11.1x | 13.1x | 13.1x |
| cjk_tail15 | 207 | 12.5x | 13.7x | 13.8x |
| ascii_large | 10816 | 0.7x | 1.0x | 4.1x |
| cjk_large | 12416 | 37.0x | 31.0x | 24.3x |
| emoji_large | 10880 | 39.3x | 35.5x | 30.9x |
| mixed_large | 9216 | 41.7x | 28.4x | 35.1x |
| early_non_ascii | 10818 | 42.7x | 31.0x | 29.9x |
| late_non_ascii | 10818 | 30.9x | 31.7x | 51.7x |

On x86_64, the baseline's `is_ascii` beats this crate's ASCII scan on the 10,816-byte ASCII input. Results depend on input length, character distribution, CPU, and compiler, so these ratios don't promise a speedup for every string or platform.

### Reproduce locally

```sh
scripts/perf-ab.sh HEAD
```

With `HEAD` as the base, the first table shows the no-change spread on your machine, and the second compares your working tree with the standard library.

### Base-vs-PR check

The [Perf A/B workflow](.github/workflows/perf-ab.yml) builds `utf16_len` from the base commit and from the change into two binaries, each by the same harness from the same directory, so identical code sits at identical addresses in both. It times each input's two sides back to back in fresh processes, alternating which goes first, repeats this 7 times, and uses the median. It runs for pull requests and pushes to `main` on Linux x86_64, Windows x86_64, and macOS aarch64, and writes the median change per input to the job summary. It fails when an input's median is more than 5% slower and every run is more than 1% slower. To accept an intended slowdown, label the pull request `perf-regression-accepted`.

Run the same comparison locally against any ref:

```sh
scripts/perf-ab.sh main
```

### CodSpeed regression tracking

The separate [CodSpeed workflow](.github/workflows/codspeed.yml) runs the [benchmark suite](benches/utf16_len.rs) in **Simulation** mode by default for pushes, pull requests, and manual runs. Its first 9 cases cover long ASCII (10,816 bytes), CJK, emoji, and mixed text; they compare SIMD with `encode_utf16().count()` and include the ASCII guard for the ASCII input. The long ASCII fixture has a separate benchmark identity from the historical 169-byte fixture, so changing the input size is not reported as a code regression; Unicode benchmark identities remain unchanged.

The `code_path` group adds 12 cases that time `utf16_len` alone on inputs chosen by the code path they reach: under 16 bytes, under 64 bytes, 3 or 15 bytes left after the last 16-byte vector, text longer than one accumulator batch, and long ASCII with one non-ASCII character at the start or end. The `kernel` group times each kernel the machine supports on the non-ASCII inputs, since ASCII input returns before any kernel runs. All benchmark inputs live in [`benches/inputs.rs`](benches/inputs.rs), which the base-vs-PR check also uses.

Use the [CodSpeed dashboard](https://app.codspeed.io/SyMind/simd-utf16-len) to track changes across commits and inspect flamegraphs. Simulation results represent modeled execution costs and are distinct from the native timings above. The workflow also supports **Walltime** mode through its manual `mode` input to measure actual elapsed time.

### Kernel sweep

The [Kernel sweep workflow](.github/workflows/kernels.yml) times every kernel a runner supports, and the `utf16_len` dispatch, on non-ASCII inputs from 13 bytes to 16 KiB, to show where a wider kernel starts to pay off. It runs on demand. Locally:

```sh
cargo run --release --manifest-path perf/kernels/Cargo.toml --features avx512
```

The [CI workflow](.github/workflows/ci.yml) runs `cargo test` on Linux, macOS, and Windows to check correctness. It also runs the unit tests for wasm32 under wasmtime, both with `simd128` and with the scalar fallback. Every test checks each kernel the CPU supports, not only the one `utf16_len` picks. Most runners don't expose AVX-512, so one job runs the tests with the `avx512` feature under Intel's Software Development Emulator as a Sapphire Rapids CPU and fails if any x86_64 kernel is missing. The tests also run in release builds, where short inputs read past their end instead of being copied.

## License

MIT
