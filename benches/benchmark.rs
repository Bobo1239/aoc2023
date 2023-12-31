use criterion::{criterion_group, criterion_main, Criterion};

use aoc2023::{read_day_input, ALL_SOLUTIONS};

pub fn criterion_benchmark(c: &mut Criterion) {
    rayon::ThreadPoolBuilder::new().build_global().unwrap();
    for (i, day) in ALL_SOLUTIONS.iter().enumerate() {
        c.bench_function(&format!("day{}", i + 1), |b| {
            let input = read_day_input(i + 1);
            b.iter(|| day(&input))
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
