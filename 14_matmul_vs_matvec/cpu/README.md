# CPU Single-Threaded Implementation

This directory contains single-threaded CPU implementations of matrix-matrix and matrix-vector multiplication for direct performance comparison.

## Files

- `comparison.cpp` - Side-by-side implementations with benchmarking
- `Makefile` - Build configuration

## Implementations

### Matrix-Matrix Multiplication (O(n³))
```cpp
void matmul(A[M][K], B[K][N], C[M][N])
  for i in 0..M:
    for j in 0..N:
      for k in 0..K:
        C[i][j] += A[i][k] * B[k][j]
```

### Matrix-Vector Multiplication (O(n²))
```cpp
void matvec(A[M][N], x[N], y[M])
  for i in 0..M:
    for j in 0..N:
      y[i] += A[i][j] * x[j]
```

## Build and Run

```bash
make
./comparison
```

## Expected Output

The program benchmarks both operations across multiple sizes (128, 256, 512, 1024, 2048) and reports:
- Execution time (ms)
- Performance (GFLOPS)
- Operations ratio

## Performance Characteristics

### Computational Complexity
- Mat*Mat: 2·M·N·K FLOPs
- Mat*Vec: 2·M·N FLOPs
- Ratio: K times more operations (K for square matrices)

### Memory Access
- Mat*Mat: Better data reuse for output elements
- Mat*Vec: Single pass through matrix, vector reused

### Cache Behavior
- Mat*Mat: Can benefit from blocking/tiling (not implemented here)
- Mat*Vec: Limited data reuse, more memory-bound

## Analysis Questions

1. How does GFLOPS scale with problem size?
2. At what size does performance plateau?
3. What is the memory bandwidth bottleneck?
4. How does the operations ratio compare to the time ratio?
