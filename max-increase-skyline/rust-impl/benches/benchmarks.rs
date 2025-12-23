//! Criterion benchmarks for all implementations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;

// Include the modules
mod lib {
    include!("../src/lib.rs");
}

mod parallel {
    include!("../src/parallel.rs");
}

use lib::{max_increase_single_threaded, max_increase_optimized};
use parallel::{max_increase_parallel, max_increase_parallel_chunked};

fn generate_grid(n: usize) -> Vec<Vec<i32>> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| (0..n).map(|_| rng.gen_range(0..100)).collect())
        .collect()
}

fn bench_single_threaded(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_threaded");
    
    for size in [64, 128, 256, 512].iter() {
        let grid = generate_grid(*size);
        
        group.throughput(Throughput::Elements((*size * *size) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("baseline", size),
            &grid,
            |b, g| b.iter(|| max_increase_single_threaded(black_box(g))),
        );
        
        group.bench_with_input(
            BenchmarkId::new("optimized", size),
            &grid,
            |b, g| b.iter(|| max_increase_optimized(black_box(g))),
        );
    }
    
    group.finish();
}

fn bench_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel");
    
    for size in [64, 128, 256, 512, 1024].iter() {
        let grid = generate_grid(*size);
        
        group.throughput(Throughput::Elements((*size * *size) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("rayon", size),
            &grid,
            |b, g| b.iter(|| max_increase_parallel(black_box(g))),
        );
        
        group.bench_with_input(
            BenchmarkId::new("rayon_chunked", size),
            &grid,
            |b, g| b.iter(|| max_increase_parallel_chunked(black_box(g))),
        );
    }
    
    group.finish();
}

fn bench_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison");
    
    let size = 512;
    let grid = generate_grid(size);
    
    group.throughput(Throughput::Elements((size * size) as u64));
    
    group.bench_function("single_threaded", |b| {
        b.iter(|| max_increase_single_threaded(black_box(&grid)))
    });
    
    group.bench_function("optimized", |b| {
        b.iter(|| max_increase_optimized(black_box(&grid)))
    });
    
    group.bench_function("parallel", |b| {
        b.iter(|| max_increase_parallel(black_box(&grid)))
    });
    
    group.bench_function("parallel_chunked", |b| {
        b.iter(|| max_increase_parallel_chunked(black_box(&grid)))
    });
    
    group.finish();
}

criterion_group!(benches, bench_single_threaded, bench_parallel, bench_comparison);
criterion_main!(benches);
