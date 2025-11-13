# Matrix-Vector Multiplication

Matrix-vector product: y = A × x where A is M×N, x is N×1, y is M×1

Simpler than GEMM but still important for optimization practice.

## Complexity Levels

1. **Naive**: Each thread computes one output element (row-wise)
2. **Coalesced**: Optimized memory access pattern
3. **Reduction**: Using parallel reduction per row

Expected performance: Limited by memory bandwidth (~100-300 GB/s)

## Files

- CUDA: `cuda/01_naive_matvec.cu`, `cuda/02_optimized_matvec.cu`
- HIP: `hip/01_naive_matvec.hip.cpp`, `hip/02_optimized_matvec.hip.cpp`
