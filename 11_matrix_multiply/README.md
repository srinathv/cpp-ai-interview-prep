# Matrix-Matrix Multiplication (GEMM)

General Matrix-Matrix multiplication (GEMM): C = A × B

This is the most important kernel in deep learning and scientific computing!

## Why GEMM Matters

- **Deep Learning**: 90%+ of training/inference time
- **BLAS**: Level 3 operation (O(N³) compute, O(N²) memory)
- **Optimization Challenge**: Achieving high FLOPS utilization

## Complexity Progression

### Level 1: Naive (❌ ~50 GFLOPS)
- Each thread computes one output element
- **Problem**: No data reuse, poor memory bandwidth utilization
- Arithmetic intensity: O(N) / O(N²) = O(1/N) - memory bound

### Level 2: Tiled with Shared Memory (✓ ~500 GFLOPS)
- Tile matrices into shared memory
- **Optimization**: Reuse data across threads in a block
- Arithmetic intensity: O(TILE_SIZE)

### Level 3: Optimized Tiling (✓✓ ~1-2 TFLOPS)
- Larger tiles, vectorized loads
- Thread-level tiling (each thread computes multiple outputs)
- **Optimization**: Better register usage, reduced shared memory traffic

### Level 4: Production (✓✓✓ ~5-15 TFLOPS)
- Warp-level primitives (tensor cores on modern GPUs)
- Double buffering, async copies
- **Note**: This is what cuBLAS/rocBLAS does

## Performance Metrics

```
FLOPS = 2 × M × N × K / time
       (2× for multiply-add operations)

Roofline: min(Peak FLOPS, Bandwidth × Arithmetic Intensity)
```

## Files

### CUDA
```
cuda/
├── 01_naive_matmul.cu              # Baseline O(1/N) arithmetic intensity
├── 02_tiled_matmul.cu              # Shared memory tiling
├── 03_optimized_matmul.cu          # Thread-level tiling + vectorization
└── matmul_utils.h                  # Common utilities
```

### HIP
```
hip/
├── 01_naive_matmul.hip.cpp         # HIP naive version
├── 02_tiled_matmul.hip.cpp         # HIP tiled version
├── 03_optimized_matmul.hip.cpp     # HIP optimized version
└── matmul_utils.h                  # Common utilities
```

## Building

### CUDA
```bash
cd cuda
nvcc -O3 -arch=sm_80 01_naive_matmul.cu -o naive
nvcc -O3 -arch=sm_80 02_tiled_matmul.cu -o tiled
nvcc -O3 -arch=sm_80 03_optimized_matmul.cu -o optimized
```

### HIP
```bash
cd hip
hipcc -O3 01_naive_matmul.hip.cpp -o naive
hipcc -O3 02_tiled_matmul.hip.cpp -o tiled
hipcc -O3 03_optimized_matmul.hip.cpp -o optimized
```

## Expected Performance

On NVIDIA A100 (19.5 TFLOPS FP32):

| Implementation | GFLOPS | % Peak | Notes |
|----------------|--------|--------|-------|
| Naive | ~50 | 0.3% | Memory bound |
| Tiled | ~500 | 2.5% | Better data reuse |
| Optimized | ~1500 | 7.7% | Good for learning |
| cuBLAS | ~15000 | 77% | Production use |

On AMD MI250X (47.9 TFLOPS FP32):

| Implementation | GFLOPS | % Peak | Notes |
|----------------|--------|--------|-------|
| Naive | ~40 | 0.08% | Memory bound |
| Tiled | ~600 | 1.25% | Shared memory helps |
| Optimized | ~2000 | 4.2% | Educational |
| rocBLAS | ~20000 | 42% | Production use |

## Key Concepts

### Arithmetic Intensity
```
AI = FLOPS / Bytes Accessed
Naive:     2MNK / (4MN + 4NK + 4MK) ≈ O(1/N)
Tiled:     2MNK / (4MN/T + 4NK/T + 4MK) ≈ O(T)
Optimized: Even higher with register blocking
```

### Memory Hierarchy
```
Global Memory: ~1 TB/s, high latency
Shared Memory: ~10 TB/s, low latency  
Registers:     ~50 TB/s, lowest latency
```

### Optimization Strategies
1. **Tiling**: Break into smaller blocks
2. **Data Reuse**: Load once, use many times
3. **Coalescing**: Ensure aligned, contiguous access
4. **Occupancy**: Balance threads/blocks/shared memory

## Interview Questions

1. **Why is naive GEMM slow?**
   - Poor arithmetic intensity (memory bound)
   - Each element loaded once from global memory
   
2. **How does tiling help?**
   - Loads tiles into shared memory
   - Threads reuse the cached data
   - Increases arithmetic intensity

3. **What's the theoretical peak?**
   - Limited by either compute or memory bandwidth
   - Use roofline model to analyze

4. **Why not write your own GEMM?**
   - Vendor libraries (cuBLAS/rocBLAS) are highly optimized
   - Use tensor cores on modern GPUs
   - But understanding helps optimize custom kernels!

5. **CUDA vs HIP for GEMM?**
   - Algorithm is identical
   - Performance depends on hardware
   - Use vendor libraries in production

## Real-World Usage

```cpp
// In production, always use optimized libraries:

// CUDA
#include <cublas_v2.h>
cublasSgemm(handle, ...);

// HIP
#include <hipblas.h>
hipblasSgemm(handle, ...);
```

## Next Steps

After mastering GEMM:
- Tensor cores (Ampere+) / Matrix cores (CDNA)
- Mixed precision (FP16, BF16, TF32)
- Fused operations (GEMM + activation)
- Batched GEMM
