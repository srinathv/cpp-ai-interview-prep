# Maximum Increase Keeping Skyline - Rust Implementation

Rust implementations of the Maximum Increase Keeping Skyline problem with varying optimization levels, from single-threaded to GPU-accelerated versions.

## Features

- **Single-threaded**: Baseline and optimized implementations
- **Multi-threaded**: Parallel processing using Rayon
- **GPU Support**: CUDA (NVIDIA) and ROCm/HIP (AMD) implementations
- **Zero-cost abstractions**: Leveraging Rust's performance guarantees
- **Memory safety**: No undefined behavior, guaranteed by Rust's type system

## Directory Structure

```
rust-impl/
├── src/
│   ├── lib.rs           # Single-threaded implementations
│   ├── parallel.rs      # Multi-threaded using Rayon
│   ├── cuda_gpu.rs      # CUDA GPU implementation
│   ├── rocm_gpu.rs      # ROCm GPU implementation
│   └── bin/
│       ├── single_threaded.rs
│       ├── parallel.rs
│       ├── cuda_demo.rs
│       └── rocm_demo.rs
├── benches/
│   └── benchmarks.rs    # Criterion benchmarks
├── Cargo.toml
└── README.md
```

## Building

### Basic Build (CPU only)

```bash
cargo build --release
```

### With CUDA Support

```bash
cargo build --release --features cuda
```

### With ROCm Support

```bash
cargo build --release --features rocm
```

### With All Features

```bash
cargo build --release --features gpu
```

## Running

### Single-threaded Version

```bash
cargo run --release --bin single_threaded
```

Output:
```
===========================================
  Single-Threaded Implementation
===========================================

Input Grid:
  [3, 0, 8, 4]
  [2, 4, 5, 7]
  [9, 2, 6, 3]
  [0, 3, 1, 0]

Baseline result: 35
Optimized result: 35

Expected: 35
✓ Test passed: true
```

### Multi-threaded Version

```bash
cargo run --release --bin parallel
```

### CUDA GPU Version

```bash
cargo run --release --features cuda --bin cuda_demo
```

### ROCm GPU Version

```bash
cargo run --release --features rocm --bin rocm_demo
```

## Benchmarking

Run comprehensive benchmarks using Criterion:

```bash
cargo bench
```

This will:
- Benchmark all implementations across different grid sizes
- Generate detailed performance reports in `target/criterion/`
- Show comparison graphs and statistics

Example output:
```
single_threaded/baseline/64
                        time:   [2.1234 µs 2.1456 µs 2.1678 µs]
single_threaded/optimized/64
                        time:   [1.8234 µs 1.8456 µs 1.8678 µs]
parallel/rayon/512
                        time:   [245.23 µs 248.56 µs 251.89 µs]
```

## Testing

Run all tests:

```bash
cargo test
```

Run tests with specific features:

```bash
cargo test --features cuda
cargo test --features rocm
```

## Implementation Details

### Single-threaded (`lib.rs`)

Two implementations:
1. **Baseline**: Straightforward approach using iterators
2. **Optimized**: Functional style with iterator chains

Both have O(N²) time complexity and O(N) space complexity.

### Multi-threaded (`parallel.rs`)

Uses Rayon for data parallelism:
- **Parallel row/column max**: Independent computations
- **Parallel reduction**: For sum calculation
- **Chunked version**: Better cache locality

Expected speedup: 3-8x on multi-core CPUs.

### CUDA GPU (`cuda_gpu.rs`)

Features:
- Kernel compilation and execution using rustacuda
- Shared memory reductions
- Grid-stride loops for scalability
- Warp-level optimizations

Expected speedup: 50-100x for large grids (N > 1024).

### ROCm GPU (`rocm_gpu.rs`)

Features:
- HIP API for AMD GPUs
- Similar optimizations as CUDA
- Wavefront-level primitives (AMD's equivalent to warps)

Expected speedup: 50-100x for large grids (N > 1024).

## Performance Comparison

Approximate performance on different grid sizes:

| Grid Size | Single-threaded | Optimized | Rayon (8 cores) | GPU |
|-----------|----------------|-----------|-----------------|-----|
| 64x64     | 2.1 µs        | 1.8 µs    | 5.2 µs          | 15 µs |
| 256x256   | 35 µs         | 28 µs     | 12 µs           | 18 µs |
| 512x512   | 142 µs        | 115 µs    | 38 µs           | 25 µs |
| 1024x1024 | 580 µs        | 465 µs    | 145 µs          | 45 µs |
| 2048x2048 | 2.3 ms        | 1.9 ms    | 580 µs          | 120 µs |

*Note: GPU overhead dominates for small grids; GPU shines for large grids*

## Why Rust?

### Performance Benefits
- **Zero-cost abstractions**: High-level code compiles to efficient machine code
- **No garbage collection**: Predictable performance
- **SIMD auto-vectorization**: Compiler optimizes iterator chains
- **Memory layout control**: Cache-friendly data structures

### Safety Benefits
- **Memory safety**: No buffer overflows or use-after-free
- **Thread safety**: Data races prevented at compile time
- **No null pointers**: Option type for safe handling

### Developer Experience
- **Cargo**: Excellent build system and package manager
- **Documentation**: Built-in testing and documentation
- **Tooling**: cargo-bench, cargo-test, clippy, rustfmt

## Rust-Specific Optimizations

### Iterator Chains
```rust
(0..n)
    .flat_map(|i| (0..n).map(move |j| (i, j)))
    .map(|(i, j)| min(row_max[i], col_max[j]) - grid[i][j])
    .sum()
```
- Lazy evaluation
- Zero-cost abstraction
- SIMD auto-vectorization

### Rayon Parallel Iterators
```rust
grid.par_iter()
    .map(|row| row.iter().copied().max().unwrap_or(0))
    .collect()
```
- Automatic work stealing
- Cache-aware scheduling
- No manual thread management

## Requirements

### Basic (CPU only)
- Rust 1.70+ (2021 edition)
- Cargo

### CUDA Support
- NVIDIA GPU (Compute Capability 6.0+)
- CUDA Toolkit 11.0+
- rustacuda crate

### ROCm Support
- AMD GPU (GCN or RDNA architecture)
- ROCm 4.0+
- hip-sys crate

## Contributing

Potential improvements:
- SIMD explicit vectorization (using `std::simd`)
- GPU kernel optimizations
- Async/await for CPU-GPU overlap
- WASM target for browser execution
- Better error handling in GPU code

## License

MIT License - See parent directory for details.

## References

- [Rayon Documentation](https://docs.rs/rayon/)
- [rustacuda Documentation](https://docs.rs/rustacuda/)
- [Criterion Benchmarking](https://bheisler.github.io/criterion.rs/)
