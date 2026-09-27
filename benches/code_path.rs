//! Times `utf16_len` alone on every shared input that `utf16_len.rs` doesn't
//! already cover. Kept in its own binary so adding cases here doesn't move the
//! code and data of the original benchmarks, which CodSpeed's cold-cache
//! simulation would report as a regression.

mod inputs;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use simd_utf16_len::utf16_len;

fn bench_code_paths(c: &mut Criterion) {
    // Benchmarked in utf16_len.rs under their original identities.
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

criterion_group!(benches, bench_code_paths);
criterion_main!(benches);
