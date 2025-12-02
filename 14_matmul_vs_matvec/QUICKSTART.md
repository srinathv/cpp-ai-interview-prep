# Quick Start Guide

## Overview

This performance comparison suite benchmarks matrix-matrix multiplication (Mat*Mat) vs matrix-vector multiplication (Mat*Vec) across three parallelization strategies:

1. **CPU Single-Threaded** - Baseline performance
2. **CPU Multi-Threaded (OpenMP)** - Thread scaling analysis
3. **CUDA GPU** - GPU acceleration comparison

## Prerequisites

### All Platforms
- C++ compiler with C++17 support (g++, clang++)

### OpenMP (Multi-threaded)
- **Linux**: Usually included with g++
- **macOS**: `brew install libomp`
- **Windows**: Visual Studio includes OpenMP

### CUDA (GPU)
- NVIDIA GPU
- CUDA Toolkit installed
- `nvcc` compiler available

## Build Everything

```bash
./build_all.sh
```

This will attempt to build all three versions. It will skip CUDA if `nvcc` is not available.

## Run Benchmarks

### Individual Benchmarks

```bash
# CPU single-threaded
cd cpu && ./comparison

# OpenMP multi-threaded
cd multithreaded && ./comparison_omp

# CUDA GPU
cd cuda && ./comparison
```

### Run All Benchmarks

```bash
./run_benchmarks.sh
```

Results will be saved to `results/` directory with timestamps.

## What to Expect

### CPU Single-Threaded
- Baseline performance for both operations
- Shows O(n³) vs O(n²) complexity difference
- Demonstrates memory-bound (Mat*Vec) vs compute-bound (Mat*Mat) behavior

**Sample Output:**
```
Size: 512
--------------------------------------------------
Mat*Mat (512x512 * 512x512): 125.342 ms, 2.145 GFLOPS
Mat*Vec (512x512 * 512): 0.245 ms, 1.073 GFLOPS
Ops ratio (MatMul/MatVec): 512x
```

### OpenMP Multi-Threaded
- Thread scaling comparison (1, 2, 4, 8 threads)
- Shows Mat*Mat scales better than Mat*Vec
- Demonstrates overhead impact on Mat*Vec

**Sample Output:**
```
========== Size: 1024 ==========

--- 4 Thread(s) ---
Mat*Mat (1024x1024 * 1024x1024) [4 threads]: 312.456 ms, 6.891 GFLOPS
Mat*Vec (1024x1024 * 1024) [4 threads]: 0.523 ms, 4.012 GFLOPS
```

### CUDA GPU
- Both compute-only and total (with transfers) timing
- Shows when GPU acceleration is beneficial
- Demonstrates transfer overhead impact

**Sample Output:**
```
========== Size: 2048 ==========
Mat*Mat (2048x2048 * 2048x2048):
  Compute: 42.123 ms, 408.234 GFLOPS
  Total:   89.456 ms, 192.345 GFLOPS (with transfers)

Mat*Vec (2048x2048 * 2048):
  Compute: 0.234 ms, 71.234 GFLOPS
  Total:   12.345 ms, 1.352 GFLOPS (with transfers)
```

## Understanding Results

### Key Metrics

1. **Time (ms)**: Execution time in milliseconds
2. **GFLOPS**: Giga floating-point operations per second (higher is better)
3. **Ops Ratio**: How many more operations Mat*Mat performs vs Mat*Vec

### What to Look For

**Single-Threaded:**
- Mat*Mat should have higher GFLOPS (better cache utilization)
- Mat*Vec is memory-bound (lower GFLOPS)
- Time ratio should be less than operations ratio

**Multi-Threaded:**
- Mat*Mat scales well with threads
- Mat*Vec scaling limited by overhead and memory bandwidth
- Efficiency decreases as thread count increases

**CUDA:**
- Compute-only shows kernel performance
- Total time includes memory transfers
- Mat*Mat: compute dominates for large sizes
- Mat*Vec: transfers can dominate

## Performance Analysis

See [ANALYSIS.md](ANALYSIS.md) for detailed performance characteristics including:
- Computational complexity analysis
- Memory access patterns
- Cache behavior
- Thread scaling theory
- GPU performance characteristics
- When to use CPU vs GPU
- Roofline model analysis

## Troubleshooting

### OpenMP Build Fails
```bash
# macOS
brew install libomp

# Then rebuild
cd multithreaded && make clean && make
```

### CUDA Build Fails
- Verify CUDA toolkit installed: `nvcc --version`
- Check GPU compute capability and update Makefile `-arch` flag
- Common values: `sm_70` (Volta), `sm_80` (Ampere), `sm_86` (RTX 30xx)

### Low Performance
- Ensure compiler optimizations enabled (`-O3`)
- Close other applications
- For GPU: check GPU utilization with `nvidia-smi`

## Next Steps

1. Run all benchmarks on your system
2. Compare results with theoretical predictions in ANALYSIS.md
3. Experiment with different sizes
4. Try different thread counts (edit comparison_omp.cpp)
5. Implement optimizations:
   - Cache blocking for CPU mat*mat
   - Shared memory tiling for CUDA mat*mat
   - SIMD vectorization for mat*vec

## Interview Preparation

Key points to understand:
1. Why Mat*Mat has O(n³) complexity vs Mat*Vec O(n²)
2. Arithmetic intensity difference (n/12 vs 0.25 FLOPs/byte)
3. Why Mat*Mat scales better with parallelization
4. When GPU acceleration is beneficial
5. Memory bandwidth vs compute bottlenecks
6. Thread overhead impact
