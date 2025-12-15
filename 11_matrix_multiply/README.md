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

### Level 4: PTX Assembly Enhanced (✓✓✓ ~2-4 TFLOPS)
- Inline PTX for FMA, prefetching, cache control
- Warp-level primitives
- Double buffering with async copy (SM 80+)

### Level 5: Production (✓✓✓✓ ~5-15 TFLOPS)
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

### CUDA (GPU)
```
cuda/
├── 01_naive_matmul.cu              # Baseline O(1/N) arithmetic intensity
├── 02_tiled_matmul.cu              # Shared memory tiling
├── 04_asm_optimized_matmul.cu      # PTX assembly enhancements
│   ├── PTX FMA (fma.rn.f32)
│   ├── Prefetching (prefetch.global.L1/L2)
│   ├── Cache hints (ld.global.nc)
│   ├── Vector loads (ld.global.v4.f32)
│   ├── Async copy (cp.async) for SM 80+
│   └── Thread-tiled with register blocking
└── matmul_utils.h                  # Common utilities
```

### HIP (AMD GPU)
```
hip/
├── 01_naive_matmul.hip.cpp         # HIP naive version
├── 02_tiled_matmul.hip.cpp         # HIP tiled version
├── 03_optimized_matmul.hip.cpp     # HIP optimized version
└── matmul_utils.h                  # Common utilities
```

### x86 CPU
```
x86/
├── dense_matmul.cpp                # Dense matrix optimizations
│   ├── Naive triple loop
│   ├── Loop reordering (ikj)
│   ├── Cache blocking/tiling
│   ├── AVX2 SIMD (8 floats)
│   ├── AVX2 + blocking + unrolling
│   ├── Register blocking (6x16)
│   ├── Inline assembly microkernel
│   └── AVX-512 (16 floats, if available)
│
└── sparse_matmul.cpp               # Sparse matrix optimizations
    ├── Storage formats: CSR, CSC, BCSR, ELL
    ├── SpMV implementations
    │   ├── CSR naive, OpenMP, unrolled
    │   ├── CSC column-wise
    │   ├── BCSR 4x4 vectorized
    │   └── ELL SIMD with gather
    ├── SpMM (sparse × dense)
    └── Dense vs Sparse comparison
```

## Building

### CUDA
```bash
cd cuda
nvcc -O3 -arch=sm_80 01_naive_matmul.cu -o naive
nvcc -O3 -arch=sm_80 02_tiled_matmul.cu -o tiled
nvcc -O3 -arch=sm_80 04_asm_optimized_matmul.cu -o asm_optimized
```

### HIP
```bash
cd hip
hipcc -O3 01_naive_matmul.hip.cpp -o naive
hipcc -O3 02_tiled_matmul.hip.cpp -o tiled
hipcc -O3 03_optimized_matmul.hip.cpp -o optimized
```

### x86
```bash
cd x86
# Dense matrix multiply
g++ -O3 -march=native -mavx2 -mfma dense_matmul.cpp -o dense_matmul
# Add -mavx512f for AVX-512 support

# Sparse matrix multiply
g++ -O3 -march=native -mavx2 -fopenmp sparse_matmul.cpp -o sparse_matmul
```

## Expected Performance

### NVIDIA A100 (19.5 TFLOPS FP32)

| Implementation | GFLOPS | % Peak | Notes |
|----------------|--------|--------|-------|
| Naive | ~50 | 0.3% | Memory bound |
| Tiled | ~500 | 2.5% | Better data reuse |
| PTX Optimized | ~2000 | 10% | Assembly enhancements |
| cuBLAS | ~15000 | 77% | Production use |

### AMD MI250X (47.9 TFLOPS FP32)

| Implementation | GFLOPS | % Peak | Notes |
|----------------|--------|--------|-------|
| Naive | ~40 | 0.08% | Memory bound |
| Tiled | ~600 | 1.25% | Shared memory helps |
| Optimized | ~2000 | 4.2% | Educational |
| rocBLAS | ~20000 | 42% | Production use |

### x86 CPU (e.g., Intel i9, ~1 TFLOPS peak)

| Implementation | GFLOPS | Notes |
|----------------|--------|-------|
| Naive (ijk) | ~2 | Cache thrashing |
| Loop reorder (ikj) | ~10 | Better access pattern |
| Blocked | ~30 | Cache-friendly |
| AVX2 | ~100 | 8-wide SIMD |
| AVX2 + Blocked | ~200 | Combined optimizations |
| Register Blocked | ~300 | Maximum register reuse |
| MKL/OpenBLAS | ~800 | Production use |

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
GPU:
  Global Memory: ~1 TB/s, high latency
  Shared Memory: ~10 TB/s, low latency
  Registers:     ~50 TB/s, lowest latency

CPU:
  DRAM:    ~50 GB/s
  L3:      ~200 GB/s
  L2:      ~500 GB/s
  L1:      ~1 TB/s
  Registers: ~10 TB/s
```

### Dense vs Sparse Strategies

| Aspect | Dense | Sparse |
|--------|-------|--------|
| Storage | O(n²) | O(nnz) |
| Access Pattern | Regular | Irregular |
| SIMD | Easy | Difficult (gathers) |
| Cache | Streaming | Random |
| Bottleneck | Compute | Memory |
| Key Optimization | Blocking | Format selection |

## Interview Questions

### GPU (CUDA/HIP)

1. **Why is naive GEMM slow?**
   - Poor arithmetic intensity (memory bound)
   - Each element loaded once from global memory

2. **How does tiling help?**
   - Loads tiles into shared memory
   - Threads reuse the cached data
   - Increases arithmetic intensity

3. **When should you use inline PTX?**
   - Architecture-specific features (tensor cores, async copy)
   - Prefetching and cache hints not exposed in CUDA C++
   - Usually NOT needed - nvcc is very good

4. **PTX vs SASS?**
   - PTX: Virtual ISA, portable across GPU generations
   - SASS: Native GPU assembly, architecture-specific
   - PTX compiled to SASS by driver/ptxas

### CPU (x86)

5. **Why does loop order matter?**
   - Memory access patterns determine cache efficiency
   - ijk: B has stride-1, but A reloaded for each j
   - ikj: Both B and C have stride-1, A hoisted

6. **How to choose block size?**
   - 3 blocks should fit in cache
   - L1 (32KB): B ~= 52
   - L2 (256KB): B ~= 147
   - Typically use 32-128 for L2 blocking

7. **AVX2 vs AVX-512 tradeoffs?**
   - AVX-512 pros: 2x wider, 2x more registers
   - AVX-512 cons: May cause frequency throttling
   - Measure on target hardware!

### Sparse

8. **Why is SpMV memory-bound?**
   - Arithmetic intensity = 2 FLOP / 12 bytes ≈ 0.17
   - Cannot reuse data - each element accessed once

9. **How to choose sparse format?**
   - CSR: General purpose, most common
   - BCSR: Natural block structure, SIMD-friendly
   - ELL: GPU-friendly, uniform row lengths
   - COO: Construction only

## Real-World Usage

```cpp
// In production, always use optimized libraries:

// CUDA
#include <cublas_v2.h>
cublasSgemm(handle, ...);

// HIP
#include <hipblas.h>
hipblasSgemm(handle, ...);

// x86 CPU
#include <mkl.h>
cblas_sgemm(CblasRowMajor, ...);

// Sparse
#include <mkl_spblas.h>
mkl_sparse_s_mv(...);
```

## Next Steps

After mastering GEMM:
- Tensor cores (Ampere+) / Matrix cores (CDNA)
- Mixed precision (FP16, BF16, TF32)
- Fused operations (GEMM + activation)
- Batched GEMM
- Sparse GEMM libraries (cuSPARSE, MKL Sparse)
