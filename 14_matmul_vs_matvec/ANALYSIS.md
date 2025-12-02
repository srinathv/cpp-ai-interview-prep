# Performance Analysis: Matrix-Matrix vs Matrix-Vector Multiplication

## Overview

This document provides a comprehensive analysis of the performance characteristics of matrix-matrix multiplication (Mat*Mat) versus matrix-vector multiplication (Mat*Vec) across different parallelization strategies.

## Computational Complexity

### Operations Count

For square n×n matrices:

**Matrix-Matrix (C = A × B):**
- Multiply-add operations: n³
- Total FLOPs: 2n³
- Complexity: O(n³)

**Matrix-Vector (y = A × x):**
- Multiply-add operations: n²
- Total FLOPs: 2n²
- Complexity: O(n²)

**Operations Ratio:** Mat*Mat performs **n times** more operations than Mat*Vec

### Scaling Behavior

| Size (n) | Mat*Mat FLOPs | Mat*Vec FLOPs | Ratio |
|----------|---------------|---------------|-------|
| 128      | 4.2M          | 32.8K         | 128x  |
| 256      | 33.6M         | 131K          | 256x  |
| 512      | 268M          | 524K          | 512x  |
| 1024     | 2.15B         | 2.1M          | 1024x |
| 2048     | 17.2B         | 8.4M          | 2048x |

## Memory Access Patterns

### Data Movement

**Matrix-Matrix:**
- Read: 2n² elements (matrices A and B)
- Write: n² elements (matrix C)
- Total: 3n² memory operations
- Bytes (double precision): 24n² bytes
- Arithmetic Intensity: (2n³)/(24n²) = **n/12 FLOPs/byte**

**Matrix-Vector:**
- Read: n² + n elements (matrix A and vector x)
- Write: n elements (vector y)
- Total: n² + 2n memory operations
- Bytes (double precision): 8n² + 16n bytes ≈ 8n² bytes
- Arithmetic Intensity: (2n²)/(8n²) = **0.25 FLOPs/byte**

### Key Insight
Mat*Mat has **~4n** higher arithmetic intensity than Mat*Vec for size n, making it much more compute-bound rather than memory-bound.

## Cache Behavior

### Mat*Mat Characteristics
- **Temporal Locality**: Each output element C[i][j] accumulates n products
- **Spatial Locality**: Sequential access to rows of A
- **Reuse**: Matrix B elements reused across multiple output rows
- **Cache Efficiency**: Benefits significantly from blocking/tiling
- **Working Set**: Can fit sub-blocks in cache

### Mat*Vec Characteristics
- **Temporal Locality**: Limited - each matrix element used once
- **Spatial Locality**: Good for sequential row access
- **Reuse**: Vector x reused n times (fits in cache)
- **Cache Efficiency**: Single-pass through matrix
- **Working Set**: Vector fits in L1, matrix in L3/DRAM

## Single-Threaded CPU Performance

### Expected Behavior

**Mat*Mat:**
- Limited by: Cache misses, memory bandwidth for large n
- Peak Performance: ~1-10 GFLOPS (depends on cache efficiency)
- Scales: O(n³) time complexity

**Mat*Vec:**
- Limited by: Memory bandwidth (memory-bound)
- Peak Performance: ~0.1-1 GFLOPS
- Scales: O(n²) time complexity
- Bottleneck: Reading matrix from DRAM

### Performance Ratio

Expected time ratio (Mat*Mat / Mat*Vec) is typically **< n** because:
1. Mat*Vec is more memory-bound (lower GFLOPS)
2. Mat*Mat benefits from cache reuse
3. Modern CPUs have good cache hierarchies

## Multi-Threaded CPU Performance (OpenMP)

### Thread Scaling Characteristics

**Mat*Mat:**
- **Parallelism**: n² independent output elements
- **Load Balance**: Excellent with `collapse(2)` directive
- **Scaling**: Near-linear for p threads (small p)
- **Bottleneck**: Memory bandwidth saturation at high thread count
- **Speedup**: ~p× for p ≤ physical cores

**Mat*Vec:**
- **Parallelism**: n independent output elements
- **Load Balance**: Good with static scheduling
- **Scaling**: Sub-linear, overhead-limited
- **Bottleneck**: Memory bandwidth, thread overhead
- **Speedup**: ~0.5p to 0.8p× (overhead dependent)

### Thread Overhead Impact

| Threads | Mat*Mat Efficiency | Mat*Vec Efficiency |
|---------|-------------------|-------------------|
| 1       | 100%              | 100%              |
| 2       | 95-98%            | 80-90%            |
| 4       | 90-95%            | 60-75%            |
| 8       | 75-85%            | 40-60%            |

Mat*Vec suffers more from overhead because:
1. Less work per thread (O(n) vs O(n²))
2. Thread creation overhead is more significant
3. Memory bandwidth saturates earlier

## GPU/CUDA Performance

### Parallelization Strategy

**Mat*Mat:**
- **Kernel Config**: 2D grid of 16×16 thread blocks
- **Threads**: n² threads for n×n output
- **Work per Thread**: O(n) operations (dot product)
- **Memory Access**: Global memory, non-coalesced for B
- **Occupancy**: High (many threads)

**Mat*Vec:**
- **Kernel Config**: 1D grid of 256-thread blocks
- **Threads**: n threads for n-dimensional output
- **Work per Thread**: O(n) operations (dot product)
- **Memory Access**: Coalesced for A, broadcast for x
- **Occupancy**: Lower (fewer threads)

### GPU Performance Characteristics

**Mat*Mat:**
- **Compute/Memory Ratio**: Favorable (n/12 FLOPs/byte)
- **Transfer Overhead**: Amortized for large n
- **Peak Performance**: 100-1000+ GFLOPS (GPU dependent)
- **Bottleneck**: Memory bandwidth (naive), compute (optimized)
- **GPU Advantage**: Significant for n > 512

**Mat*Vec:**
- **Compute/Memory Ratio**: Poor (0.25 FLOPs/byte)
- **Transfer Overhead**: Can dominate total time
- **Peak Performance**: 10-100 GFLOPS
- **Bottleneck**: Memory bandwidth
- **GPU Advantage**: Minimal for n < 2048

### Memory Transfer Impact

Transfer time for double precision:

| Size | Mat*Mat Transfer | Mat*Vec Transfer | Compute Time Advantage |
|------|-----------------|------------------|----------------------|
| 512  | ~6 MB           | ~2 MB            | Mat*Mat >> Mat*Vec   |
| 1024 | ~24 MB          | ~8 MB            | Mat*Mat >> Mat*Vec   |
| 2048 | ~96 MB          | ~32 MB           | Mat*Mat >> Mat*Vec   |
| 4096 | ~384 MB         | ~128 MB          | Mat*Mat >> Mat*Vec   |

For Mat*Mat, compute time grows as O(n³), overwhelming transfer time O(n²).
For Mat*Vec, compute time is O(n²), similar to transfer time.

## When to Use GPU

### Matrix-Matrix Multiplication
✅ **Use GPU when:**
- n ≥ 512 (compute dominates transfer)
- Data already on GPU (no transfer cost)
- Need to perform multiple operations

❌ **Use CPU when:**
- n < 256 (transfer overhead dominates)
- One-time computation with cold data
- CPU already saturated with other work

### Matrix-Vector Multiplication
✅ **Use GPU when:**
- n ≥ 2048 (parallelism benefits)
- Data already on GPU
- Part of larger GPU pipeline

❌ **Use CPU when:**
- n < 2048 (CPU competitive or faster)
- Cold data (transfer overhead)
- Memory bandwidth limited

## Roofline Model Analysis

### Mat*Mat on CPU
```
Arithmetic Intensity = n/12 FLOPs/byte
For n = 1024: AI = 85 FLOPs/byte (compute-bound)
Expected: ~80% of peak compute performance
```

### Mat*Vec on CPU
```
Arithmetic Intensity = 0.25 FLOPs/byte
Memory-bound: limited by ~50 GB/s bandwidth
Expected: ~1-5% of peak compute performance
```

### Mat*Mat on GPU
```
Arithmetic Intensity = n/12 FLOPs/byte
For n = 1024: AI = 85 FLOPs/byte
Naive: ~10-20% of peak (memory limited)
Optimized (tiled): ~40-80% of peak
```

### Mat*Vec on GPU
```
Arithmetic Intensity = 0.25 FLOPs/byte
Memory-bound: limited by ~500 GB/s bandwidth (modern GPU)
Expected: ~1-10% of peak compute performance
```

## Summary and Recommendations

### Key Takeaways

1. **Complexity Matters**: Mat*Mat is O(n³) vs Mat*Vec O(n²)
   - n times more operations for size n

2. **Memory Intensity Differs**: Mat*Mat is compute-bound, Mat*Vec is memory-bound
   - Mat*Mat: ~n/12 FLOPs/byte
   - Mat*Vec: ~0.25 FLOPs/byte

3. **Parallelization Benefits**: Mat*Mat scales better with threads/GPU
   - More work per thread
   - Better amortization of overhead

4. **GPU Acceleration**: Mat*Mat benefits more from GPU
   - Higher compute intensity
   - Transfer overhead amortized by O(n³) compute

### Best Practices

**For Mat*Mat:**
- Use multi-threading for n ≥ 256
- Use GPU for n ≥ 512
- Consider optimized libraries (BLAS, cuBLAS)
- Implement cache-blocking for CPU

**For Mat*Vec:**
- Use multi-threading for n ≥ 1024
- Use GPU only when data already on GPU or n ≥ 2048
- Consider vectorization (SIMD)
- Optimize memory layout for cache

**Library Recommendations:**
- CPU: Intel MKL, OpenBLAS, Eigen
- GPU: cuBLAS, rocBLAS, cuSPARSE (for sparse)
