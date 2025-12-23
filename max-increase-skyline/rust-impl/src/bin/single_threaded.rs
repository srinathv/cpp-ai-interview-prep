//! Single-threaded implementation demo

use skyline::{max_increase_single_threaded, max_increase_optimized};

fn main() {
    let grid = vec![
        vec![3, 0, 8, 4],
        vec![2, 4, 5, 7],
        vec![9, 2, 6, 3],
        vec![0, 3, 1, 0],
    ];

    println!("===========================================");
    println!("  Single-Threaded Implementation");
    println!("===========================================");
    println!();
    
    println!("Input Grid:");
    for row in &grid {
        println!("  {:?}", row);
    }
    println!();

    let result1 = max_increase_single_threaded(&grid);
    println!("Baseline result: {}", result1);

    let result2 = max_increase_optimized(&grid);
    println!("Optimized result: {}", result2);
    
    println!();
    println!("Expected: 35");
    println!("✓ Test passed: {}", result1 == 35 && result2 == 35);
}
