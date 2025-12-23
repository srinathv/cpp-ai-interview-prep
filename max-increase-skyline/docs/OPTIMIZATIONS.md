# Optimization Techniques Applied

## 1. Baseline Implementation

**File:** `src/baseline.cpp`

The baseline implementation is a straightforward solution that:
- Uses three separate loops to compute row max, column max, and sum
- Time Complexity: O(N²)
- Space Complexity: O(N)

**Characteristics:**
- Simple and readable
- No optimization flags
- Serves as reference for comparison

---

## 2. CPU Optimizations

**File:** `src/cpu_optimized.cpp`

### Cache-Friendly Memory Access
- Process data row-by-row to maximize cache hits
- Minimize cache misses by maintaining spatial locality

### Compiler Hints
```cpp
#pragma GCC ivdep  // Ignore vector dependencies for auto-vectorization
```

### Loop Optimizations
- Reduced branching by using `std::max` instead of if statements
- Hoisted invariant computations out of inner loops
- Better register utilization

### SIMD (AVX2) Version
- Processes 8 integers at once using 256-bit SIMD registers
- Uses intrinsics for explicit vectorization:
  - `_mm256_loadu_si256` - Load 8 integers
  - `_mm256_max_epi32` - Parallel max operation
  - Horizontal reduction for final max value

**Expected Speedup:** 2-3x over baseline

---

## 3. OpenMP Parallelization

**File:** `src/openmp_parallel.cpp`

### Parallel Strategies

#### Row Max Calculation
```cpp
#pragma omp parallel for schedule(static)
```
- Each row is independent
- Static scheduling for balanced load
- No synchronization needed

#### Column Max Calculation
```cpp
#pragma omp parallel for schedule(static)
```
- Each column is independent
- Note: Column access is cache-unfriendly but parallelizable

#### Sum Reduction
```cpp
#pragma omp parallel for reduction(+:totalSum)
```
- Built-in reduction clause for thread-safe summation
- Each thread maintains local sum, combined at end

### Advanced: Nested Parallelism
For very large grids (N > 512):
```cpp
#pragma omp parallel sections
```
- Row and column max computed concurrently
- Two-level parallelism

**Expected Speedup:** 4-12x over baseline (depends on core count)

---

## 4. CUDA GPU Acceleration

**File:** `src/cuda_gpu.cu`

### Memory Management
- Flatten 2D grid to 1D for coalesced memory access
- Minimize host-device transfers

### Kernel Optimizations

#### Row/Column Max Kernels
- One thread per row/column
- Simple parallel pattern
- Maximizes GPU utilization

#### Sum Reduction Kernel
**Shared Memory Reduction:**
```cuda
__shared__ int sharedSum[256];
```
- Fast on-chip memory for inter-thread communication
- Tree-based reduction pattern

**Warp Shuffle Optimization:**
```cuda
__shfl_down_sync(0xffffffff, localSum, offset)
```
- Warp-level primitives (no shared memory needed)
- 32 threads cooperate within warp
- Faster than shared memory for small reductions

### Grid-Stride Loop
```cuda
for (int i = idx; i < totalElements; i += blockDim.x * gridDim.x)
```
- Handles any grid size
- Better work distribution

**Expected Speedup:** 50-100x over baseline for large grids

---

## 5. ROCm/HIP GPU Acceleration

**File:** `src/rocm_gpu.cpp`

### HIP API (AMD GPUs)
HIP provides portability between AMD and NVIDIA:
- Same kernel code as CUDA with minor differences
- Different launch syntax: `hipLaunchKernelGGL`

### AMD-Specific Optimizations

#### Wavefront vs Warp
- AMD GPUs use 64-wide wavefronts (vs NVIDIA's 32-wide warps)
- Adjusted shuffle reduction for 64 threads

```cpp
for (int offset = 32; offset > 0; offset >>= 1) {
    localSum += __shfl_down(localSum, offset);
}
```

### Memory Coalescing
- Same principles as CUDA
- AMD GCN architecture benefits from coalesced access

**Expected Speedup:** 50-100x over baseline (similar to CUDA)

---

## Optimization Summary

| Technique | Implementation | Speedup | Best For |
|-----------|---------------|---------|----------|
| Cache Locality | CPU Optimized | 2-3x | Small-medium grids |
| SIMD/AVX2 | CPU Optimized | 2-4x | Row processing |
| OpenMP | OpenMP Parallel | 4-12x | Multi-core CPUs |
| GPU Parallelism | CUDA/ROCm | 50-100x | Large grids (N>1024) |
| Shared Memory | CUDA/ROCm | 2-3x | Within GPU kernel |
| Warp Shuffle | CUDA/ROCm | 1.5-2x | Small reductions |

---

## When to Use Each Implementation

### Baseline
- Small grids (N < 100)
- Quick prototyping
- Verification reference

### CPU Optimized
- Medium grids (100 < N < 1000)
- No multi-threading support
- Single-core performance critical

### OpenMP
- Large grids (N > 500)
- Multi-core CPUs available
- No GPU available

### CUDA
- Very large grids (N > 2000)
- NVIDIA GPU available
- Maximum performance needed

### ROCm
- Very large grids (N > 2000)
- AMD GPU available
- Maximum performance needed

---

## Compilation Flags

### CPU Optimizations
```bash
-O3              # Maximum optimization
-march=native    # Target current CPU architecture
-mtune=native    # Tune for current CPU
-mavx2           # Enable AVX2 instructions
-mfma            # Enable FMA instructions
```

### OpenMP
```bash
-fopenmp         # Enable OpenMP support
```

### CUDA
```bash
-O3              # Optimization level
--gpu-architecture=sm_XX  # Target GPU compute capability
```

### ROCm
```bash
-O3              # Optimization level
```

---

## Profiling and Analysis Tools

### CPU
- `perf` - Linux performance counters
- `valgrind --tool=cachegrind` - Cache analysis
- Intel VTune / AMD uProf

### GPU
- `nvprof` / `nsys` - NVIDIA profiler
- `rocprof` - AMD profiler
- Nsight Compute / Radeon GPU Profiler

---

## Further Optimizations (Advanced)

1. **Memory Alignment**: Align data to cache line boundaries (64 bytes)
2. **Prefetching**: Manual prefetch for predictable access patterns
3. **Tile-Based Processing**: Process in cache-sized tiles
4. **Persistent Kernels**: GPU kernels that stay resident
5. **Asynchronous Execution**: Overlap computation and data transfer
