# Trigonometric Function Fusion: cos(x), sin(x), and sum

## Testing Status

| Platform | Status | Notes |
|----------|--------|-------|
| OpenMP (CPU) | Verified | Tested on macOS aarch64 (Apple Silicon) with g++-13 |
| CUDA | Not tested | Requires NVIDIA GPU + CUDA toolkit |
| HIP/ROCm | Not tested | Requires AMD GPU + ROCm stack |

## Overview

This directory demonstrates kernel fusion optimization for computing:
- cos(x) for all elements
- sin(x) for all elements
- sum of all elements

## Mathematical Background

### Trigonometric Identities Used

1. Pythagorean Identity: sin^2(x) + cos^2(x) = 1
2. Euler's Formula: e^(ix) = cos(x) + i*sin(x)

### Why Fusion Matters

Without fusion (3 separate kernels):
- Kernel 1: Load x, compute cos(x), store
- Kernel 2: Load x, compute sin(x), store
- Kernel 3: Load values, reduce to sum

With fusion (1 kernel):
- Load x ONCE, compute both cos(x) and sin(x), reduce, store

Memory bandwidth savings: 3x fewer global memory accesses!

### Simultaneous sin/cos Computation

Most CPUs and GPUs have instructions to compute sin and cos together:
- x86: fsincos instruction
- CUDA/HIP: sincosf() intrinsic
- This is faster than calling sin() and cos() separately

## Optimization Techniques

### Level 1: Naive
- Separate kernels for each operation
- Multiple memory passes
- No fusion

### Level 2: Optimized
- Fused cos/sin computation using sincosf()
- Combined memory access
- Basic parallel reduction for sum

### Level 3: Super Optimized
- Warp/wave shuffle-based reduction
- Shared memory tiling
- Coalesced memory access
- Single pass for all operations

## Parallel Reduction for Sum

### Basic Reduction
Each block reduces its portion, then atomic add to global result.

### Shuffle-based Reduction (GPU)
Uses warp/wave shuffles for register-to-register communication:

    for (offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }

Benefits:
- No shared memory needed for warp-level reduction
- No synchronization within warp
- Lower latency than shared memory

### Tree Reduction Pattern
          [a0 a1 a2 a3 a4 a5 a6 a7]
                    |
          [a0+a4  a1+a5  a2+a6  a3+a7]
                    |
          [a0+a2+a4+a6  a1+a3+a5+a7]
                    |
          [a0+a1+a2+a3+a4+a5+a6+a7]

## Directory Structure

    trig-fusion/
    ├── README.md
    ├── cuda/
    │   ├── naive.cu           # Separate kernels
    │   ├── optimized.cu       # Fused with sincosf
    │   └── super_optimized.cu # + warp shuffles
    ├── hip/
    │   ├── naive.hip
    │   ├── optimized.hip
    │   └── super_optimized.hip
    └── openmp/
        ├── naive.cpp
        ├── optimized.cpp
        └── super_optimized.cpp

## Compilation

### OpenMP (CPU)
    g++ -O3 -fopenmp -march=native naive.cpp -o naive
    g++ -O3 -fopenmp -march=native optimized.cpp -o optimized
    g++ -O3 -fopenmp -march=native super_optimized.cpp -o super_optimized

### CUDA (requires NVIDIA GPU)
    nvcc -O3 -arch=sm_80 --use_fast_math naive.cu -o naive
    nvcc -O3 -arch=sm_80 --use_fast_math optimized.cu -o optimized
    nvcc -O3 -arch=sm_80 --use_fast_math super_optimized.cu -o super_optimized

### HIP (requires AMD GPU)
    hipcc -O3 --use_fast_math naive.hip -o naive
    hipcc -O3 --use_fast_math optimized.hip -o optimized
    hipcc -O3 --use_fast_math super_optimized.hip -o super_optimized

## Performance Notes

| Version | Memory Passes | Kernel Launches | Uses sincosf |
|---------|---------------|-----------------|--------------|
| Naive   | 5             | 3               | No           |
| Optimized | 3           | 1               | Yes          |
| Super   | 2             | 1               | Yes          |
