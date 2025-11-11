# Matrix Transpose on GPU

Progressive complexity examples of matrix transpose optimization on GPUs, showing both CUDA and HIP implementations.

## Why Transpose Matters

Matrix transpose is a fundamental operation in:
- Linear algebra libraries (BLAS)
- Deep learning frameworks (tensor operations)
- Scientific computing (FFT, data reorganization)
- Image processing (rotation, filtering)

**Challenge**: Transpose involves non-coalesced memory access, making it a perfect case study for GPU optimization.

## Complexity Progression

### Level 1: Naive Transpose (❌ Slow)
- Simple row/column swap
- **Problem**: Uncoalesced writes cause ~10x slowdown
- **Performance**: ~10-20 GB/s on modern GPUs

### Level 2: Coalesced Transpose (✓ Better)
- Read coalesced, write coalesced using shared memory
- **Solution**: Tile the matrix, use shared memory as intermediate
- **Performance**: ~100-200 GB/s

### Level 3: Bank Conflict Free (✓✓ Best)
- Adds padding to avoid shared memory bank conflicts
- **Solution**: Pad shared memory tiles
- **Performance**: ~300-400 GB/s (near memory bandwidth)

## Files

### CUDA Implementations
```
cuda/
├── 01_naive_transpose.cu           # Baseline (slow)
├── 02_coalesced_transpose.cu       # Shared memory optimization
├── 03_bank_conflict_free.cu        # Padding for bank conflicts
└── transpose_utils.h               # Common utilities
```

### HIP Implementations
```
hip/
├── 01_naive_transpose.hip.cpp      # HIP version of naive
├── 02_coalesced_transpose.hip.cpp  # HIP version with shared memory
├── 03_bank_conflict_free.hip.cpp   # HIP version optimized
└── transpose_utils.h               # Common utilities
```

## Key Concepts

### Memory Coalescing
```
Coalesced:     [T0 T1 T2 T3] read consecutive addresses
               ✓ Single memory transaction

Uncoalesced:   [T0 T1 T2 T3] read strided addresses
               ✗ Multiple memory transactions
```

### Shared Memory Banks
- Shared memory divided into 32 banks
- Simultaneous access to same bank = conflict
- Solution: Pad arrays to avoid conflicts

### Performance Metrics
```
Effective Bandwidth = 2 × sizeof(matrix) / time
                     (factor of 2 for read + write)
```

## Building

### CUDA
```bash
cd cuda
nvcc -O3 -arch=sm_80 01_naive_transpose.cu -o naive
nvcc -O3 -arch=sm_80 02_coalesced_transpose.cu -o coalesced
nvcc -O3 -arch=sm_80 03_bank_conflict_free.cu -o optimized

./naive
./coalesced
./optimized
```

### HIP
```bash
cd hip
hipcc -O3 01_naive_transpose.hip.cpp -o naive
hipcc -O3 02_coalesced_transpose.hip.cpp -o coalesced
hipcc -O3 03_bank_conflict_free.hip.cpp -o optimized

./naive
./coalesced
./optimized
```

## Expected Results

On modern GPUs (NVIDIA A100 or AMD MI250):

| Implementation | Bandwidth | Speedup | Notes |
|----------------|-----------|---------|-------|
| Naive | ~15 GB/s | 1x | Baseline |
| Coalesced | ~150 GB/s | 10x | Major improvement |
| Bank Conflict Free | ~350 GB/s | 23x | Near optimal |

## Interview Questions

1. **Why is transpose slow on GPU?**
   - Non-coalesced memory access pattern
   
2. **How do you optimize transpose?**
   - Use shared memory tiling
   - Ensure coalesced reads and writes
   
3. **What are bank conflicts?**
   - Multiple threads accessing same shared memory bank
   - Solution: Padding
   
4. **CUDA vs HIP for transpose?**
   - Same algorithm works for both
   - HIP allows portability across vendors

5. **Real-world usage?**
   - cuBLAS/hipBLAS for production
   - Understanding helps optimize custom kernels

## References

- [CUDA C Best Practices Guide - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory)
- [Efficient Matrix Transpose in CUDA](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)

## Next Steps

After mastering transpose, explore:
- Matrix multiplication with shared memory
- Reduction patterns
- Scan (prefix sum)
- Convolution
