//! Parallel multi-threaded implementation demo

mod parallel {
    include!("../parallel.rs");
}

use parallel::{max_increase_parallel, max_increase_parallel_chunked};
use rayon::ThreadPoolBuilder;

fn main() {
    let grid = vec![
        vec![3, 0, 8, 4],
        vec![2, 4, 5, 7],
        vec![9, 2, 6, 3],
        vec![0, 3, 1, 0],
    ];

    println!("===========================================");
    println!("  Multi-Threaded Parallel Implementation");
    println!("===========================================");
    println!();
    
    let num_threads = rayon::current_num_threads();
    println!("Using {} threads", num_threads);
    println!();
    
    println!("Input Grid:");
    for row in &grid {
        println!("  {:?}", row);
    }
    println!();

    let result1 = max_increase_parallel(&grid);
    println!("Parallel result: {}", result1);

    let result2 = max_increase_parallel_chunked(&grid);
    println!("Parallel chunked result: {}", result2);
    
    println!();
    println!("Expected: 35");
    println!("✓ Test passed: {}", result1 == 35 && result2 == 35);
    
    // Test with larger grid
    println!();
    println!("Testing with larger grid (512x512)...");
    let large_grid: Vec<Vec<i32>> = (0..512)
        .map(|i| (0..512).map(|j| (i * 37 + j * 17) % 100).collect())
        .collect();
    
    let start = std::time::Instant::now();
    let large_result = max_increase_parallel(&large_grid);
    let duration = start.elapsed();
    
    println!("Result: {}", large_result);
    println!("Time: {:?}", duration);
}
