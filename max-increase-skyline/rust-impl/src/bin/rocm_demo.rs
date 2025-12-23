//! ROCm GPU implementation demo

mod rocm_gpu {
    include!("../rocm_gpu.rs");
}

use rocm_gpu::max_increase_rocm_simplified;

fn main() {
    let grid = vec![
        vec![3, 0, 8, 4],
        vec![2, 4, 5, 7],
        vec![9, 2, 6, 3],
        vec![0, 3, 1, 0],
    ];

    println!("===========================================");
    println!("  ROCm/HIP GPU Implementation");
    println!("===========================================");
    println!();
    
    println!("Input Grid:");
    for row in &grid {
        println!("  {:?}", row);
    }
    println!();

    match max_increase_rocm_simplified(&grid) {
        Ok(result) => {
            println!("ROCm GPU result: {}", result);
            println!();
            println!("Expected: 35");
            println!("✓ Test passed: {}", result == 35);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!("Make sure ROCm is properly installed and an AMD GPU is available.");
        }
    }
}
