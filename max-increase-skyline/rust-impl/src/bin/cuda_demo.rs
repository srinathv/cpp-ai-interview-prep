//! CUDA GPU implementation demo

mod cuda_gpu {
    include!("../cuda_gpu.rs");
}

use cuda_gpu::max_increase_cuda;

fn main() {
    let grid = vec![
        vec![3, 0, 8, 4],
        vec![2, 4, 5, 7],
        vec![9, 2, 6, 3],
        vec![0, 3, 1, 0],
    ];

    println!("===========================================");
    println!("  CUDA GPU Implementation");
    println!("===========================================");
    println!();
    
    println!("Input Grid:");
    for row in &grid {
        println!("  {:?}", row);
    }
    println!();

    match max_increase_cuda(&grid) {
        Ok(result) => {
            println!("CUDA GPU result: {}", result);
            println!();
            println!("Expected: 35");
            println!("✓ Test passed: {}", result == 35);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!("Make sure CUDA is properly installed and a GPU is available.");
        }
    }
}
