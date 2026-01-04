# Multi-Threaded OpenMP Implementation

This directory contains OpenMP parallelized implementations of matrix-matrix and matrix-vector multiplication for thread scaling analysis.

## Files

- `comparison_omp.cpp` - OpenMP parallelized implementations with multi-thread benchmarking
- `Makefile` - Build configuration with OpenMP support

## Parallelization Strategy

### Matrix-Matrix Multiplication
```cpp
#pragma omp parallel for collapse(2)
for i in 0..M:
  for j in 0..N:
    for k in 0..K:
      C[i][j] += A[i][k] * B[k][j]
```
- Parallelizes outer two loops (i, j)
- Each thread computes independent output elements
- Good load balancing with `collapse(2)`

### Matrix-Vector Multiplication
```cpp
#pragma omp parallel for
for i in 0..M:
  for j in 0..N:
    y[i] += A[i][j] * x[j]
```
- Parallelizes outer loop only
- Each thread computes independent output elements
- Less work per thread compared to mat*mat

## Build and Run

```bash
make
./comparison_omp
```

## Thread Scaling Behavior

### Matrix-Matrix
- **Strong Scaling**: O(n³) work scales well across threads
- **Compute-Bound**: High arithmetic intensity
- **Cache Effects**: Each thread operates on different rows
- **Overhead**: Thread creation overhead amortized over many operations

### Matrix-Vector
- **Weak Scaling**: O(n²) work provides less parallelism
- **Memory-Bound**: Limited by memory bandwidth
- **Cache Effects**: Vector reuse across all threads (cache-friendly)
- **Overhead**: Thread overhead more significant relative to work

## Expected Scaling

For n×n matrices with p threads:

**Mat*Mat:**
- Ideal speedup: ~p (for large n)
- Limited by: Memory bandwidth, cache coherency

**Mat*Vec:**
- Ideal speedup: ~p (diminishes for small n)
- Limited by: Memory bandwidth, thread overhead

## Analysis Questions

1. At what matrix size does multi-threading become beneficial?
2. How does speedup compare between mat*mat and mat*vec?
3. What is the parallel efficiency (speedup/threads)?
4. When does memory bandwidth become the bottleneck?
5. How does cache coherency impact performance?

## Optimization Opportunities

Not implemented here, but possible improvements:
- Block/tile matrices for better cache utilization
- SIMD vectorization within threads
- Non-uniform memory access (NUMA) awareness
- Thread affinity control
