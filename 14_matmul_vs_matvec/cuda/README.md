# CUDA GPU Implementation

This directory contains CUDA implementations of matrix-matrix and matrix-vector multiplication for GPU performance comparison.

## Files

- `comparison.cu` - CUDA kernel implementations with benchmarking
- `Makefile` - CUDA compilation configuration

## CUDA Implementations

### Matrix-Matrix Multiplication Kernel
```cuda
__global__ void matmul_kernel(A, B, C, M, K, N) {
    row = blockIdx.y * blockDim.y + threadIdx.y
    col = blockIdx.x * blockDim.x + threadIdx.x
    
    if (row < M && col < N):
        sum = 0
        for k in 0..K:
            sum += A[row][k] * B[k][col]
        C[row][col] = sum
}
```
- **Thread Layout**: 2D grid of 16×16 thread blocks
- **Parallelism**: Each thread computes one output element
- **Memory Access**: Non-coalesced for matrix B (column access)

### Matrix-Vector Multiplication Kernel
```cuda
__global__ void matvec_kernel(A, x, y, M, N) {
    row = blockIdx.x * blockDim.x + threadIdx.x
    
    if (row < M):
        sum = 0
        for j in 0..N:
            sum += A[row][j] * x[j]
        y[row] = sum
}
```
- **Thread Layout**: 1D grid of 256-thread blocks
- **Parallelism**: Each thread computes one output element
- **Memory Access**: Coalesced for matrix A, broadcast for vector x

## Build and Run

```bash
# Adjust -arch flag based on your GPU
# sm_70 = Volta, sm_75 = Turing, sm_80 = Ampere, sm_86 = RTX 30xx
make
./comparison
```

## Performance Metrics

The benchmark reports two timings:
1. **Compute**: Kernel execution time only
2. **Total**: Including host-device memory transfers

## GPU Performance Characteristics

### Matrix-Matrix Multiplication
- **High Parallelism**: n² independent output elements for n×n matrices
- **Compute Intensity**: O(n) operations per output element
- **Memory Pattern**: Global memory reads, limited reuse
- **Optimization Potential**: Shared memory tiling, coalescing

### Matrix-Vector Multiplication
- **Lower Parallelism**: Only n output elements
- **Compute Intensity**: O(n) operations per output element
- **Memory Pattern**: Vector broadcast benefits from cache
- **Bottleneck**: Memory bandwidth limited

## Expected Performance

### When GPU Wins
- Large matrix sizes (>1024×1024)
- Mat*Mat with high FLOP/byte ratio
- When data already on GPU (compute-only timing)

### When CPU Competitive
- Small sizes (<256×256)
- Mat*Vec with memory-bound workload
- When including transfer overhead

## Memory Transfer Impact

For n×n matrices:
- **Mat*Mat**: Transfer 2n² + n² = 3n² doubles
- **Mat*Vec**: Transfer n² + n + n = n² + 2n doubles

Transfer time often dominates for small n, especially in Mat*Vec.

## Analysis Questions

1. At what size does GPU outperform CPU?
2. How much does memory transfer overhead impact performance?
3. What is the achieved occupancy for each kernel?
4. How does arithmetic intensity differ between operations?
5. What percentage of peak FLOPS is achieved?

## Optimization Opportunities

Not implemented here, but possible improvements:
- **Shared memory tiling** for mat*mat to improve data reuse
- **Coalesced memory access** through transposed layouts
- **Warp-level reductions** for mat*vec
- **Tensor Cores** for mixed-precision operations (Volta+)
- **Persistent kernels** to amortize launch overhead
- **cuBLAS library** for production-optimized kernels

## Comparison with cuBLAS

For reference, cuBLAS provides:
- `cublasDgemm()` for matrix-matrix
- `cublasDgemv()` for matrix-vector

These are highly optimized and should outperform these naive kernels significantly.
