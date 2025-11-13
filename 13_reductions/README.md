# Parallel Reductions

Reduce an array to a single value (sum, max, min, etc.)

Classic parallel algorithm with multiple optimization levels.

## Why Reductions Matter

- Fundamental building block in many algorithms
- Norm calculations, dot products, finding extrema
- Used in machine learning (loss calculation, gradient norms)

## Complexity Progression

### Level 1: Naive (~50 GB/s)
- Each thread processes one element
- Heavy atomic contention on result

### Level 2: Shared Memory (~200 GB/s)
- Tree-based reduction within block
- Reduces atomic contention

### Level 3: Optimized (~400 GB/s)
- Warp-level primitives (__shfl_down)
- Sequential addressing
- Multiple elements per thread

### Level 4: Production
- Cooperative groups
- Multi-stage reductions
- What libraries like CUB/Thrust use

## Files

### CUDA
```
cuda/
├── 01_naive_reduction.cu          # Atomic operations
├── 02_shared_reduction.cu         # Tree reduction in shared memory
├── 03_warp_reduction.cu           # Warp shuffle primitives
└── reduction_utils.h
```

### HIP
```
hip/
├── 01_naive_reduction.hip.cpp
├── 02_shared_reduction.hip.cpp
├── 03_warp_reduction.hip.cpp
└── reduction_utils.h
```

## Key Concepts

### Tree Reduction
```
[1 2 3 4 5 6 7 8]
 └─┴─ └─┴─ └─┴─ └─┴─
  [3    7   11   15]
   └────┴─   └────┴─
     [10      26]
      └────────┴─
         [36]
```

### Warp Shuffle
```cpp
// Modern way (no shared memory needed!)
int val = data[threadIdx.x];
val += __shfl_down_sync(0xffffffff, val, 16);
val += __shfl_down_sync(0xffffffff, val, 8);
val += __shfl_down_sync(0xffffffff, val, 4);
val += __shfl_down_sync(0xffffffff, val, 2);
val += __shfl_down_sync(0xffffffff, val, 1);
// Thread 0 now has sum of all 32 threads
```

## Performance

| Algorithm | Bandwidth | Notes |
|-----------|-----------|-------|
| Naive atomic | ~50 GB/s | Bottlenecked by atomic ops |
| Shared memory | ~200 GB/s | Better parallelism |
| Warp shuffle | ~400 GB/s | Near optimal |
| CUB/Thrust | ~500 GB/s | Production ready |

## Interview Questions

1. **Why not use a single atomic add?**
   - Serializes all updates, very slow

2. **How does tree reduction help?**
   - Logarithmic depth: O(log N) steps
   - Parallel at each level

3. **What are warp shuffles?**
   - Direct register-to-register communication
   - Faster than shared memory

4. **When to use reductions?**
   - Sum/mean/variance calculations
   - Finding min/max
   - Dot products
   - Any commutative, associative operation

## Real-World Usage

```cpp
// Use optimized libraries in production:
#include <thrust/reduce.h>
float sum = thrust::reduce(data, data + N);

// Or CUB for more control:
#include <cub/cub.cuh>
cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, N);
```
