use criterion::{criterion_group, criterion_main, Criterion};

use aoc2023::{default_input, ALL_SOLUTIONS};

pub fn criterion_benchmark(c: &mut Criterion) {
    for (i, day) in ALL_SOLUTIONS.iter().enumerate() {
        c.bench_function(&format!("day{}", i + 1), |b| {
            let input = default_input(i + 1);
            b.iter(|| day(&input))
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
