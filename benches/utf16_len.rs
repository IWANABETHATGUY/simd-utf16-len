mod inputs;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use inputs::{ASCII, CJK, EMOJI, MIXED};
use simd_utf16_len::utf16_len;

#[inline]
fn ascii_guard_len(s: &str) -> usize {
    if s.is_ascii() {
        s.len()
    } else {
        s.encode_utf16().count()
    }
}

// The long ASCII fixture has its own benchmark identity: the historical
// bench_inputs::ascii used 169 bytes, while this fixture uses 10,816 bytes.
fn bench_ascii(c: &mut Criterion) {
    let ascii = ASCII.repeat(64);
    let input = ascii.as_str();
    let mut group = c.benchmark_group("utf16_len");
    group.bench_function(BenchmarkId::new("ascii", "simd"), |b| {
        b.iter(|| utf16_len(black_box(input)));
    });
    group.bench_function(BenchmarkId::new("ascii", "encode_utf16"), |b| {
        b.iter(|| black_box(input).encode_utf16().count());
    });
    group.bench_function(BenchmarkId::new("ascii", "is_ascii"), |b| {
        b.iter(|| ascii_guard_len(black_box(input)));
    });
    group.finish();
}

fn bench_inputs(c: &mut Criterion) {
    let inputs: &[(&str, &str)] = &[("cjk", CJK), ("emoji", EMOJI), ("mixed", MIXED)];

    let mut group = c.benchmark_group("utf16_len");
    for &(name, input) in inputs {
        group.bench_function(BenchmarkId::new(name, "simd"), |b| {
            b.iter(|| utf16_len(black_box(input)));
        });
        group.bench_function(BenchmarkId::new(name, "encode_utf16"), |b| {
            b.iter(|| black_box(input).encode_utf16().count());
        });
    }
    group.finish();
}

// Every other shared input, timing `utf16_len` alone. Inputs benchmarked above
// keep their original identities there, so CodSpeed history continues.
fn bench_code_paths(c: &mut Criterion) {
    let covered = ["cjk", "emoji", "mixed", "ascii_large"];
    let mut group = c.benchmark_group("code_path");
    for (name, input) in inputs::all() {
        if covered.contains(&name) {
            continue;
        }
        group.bench_function(BenchmarkId::new(name, "simd"), |b| {
            b.iter(|| utf16_len(black_box(input.as_str())));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_ascii, bench_inputs, bench_code_paths);
criterion_main!(benches);
