# Softmax Optimization for AI/ML

## Testing Status

| Platform | Status | Notes |
|----------|--------|-------|
| OpenMP (CPU) | Verified | Tested on macOS aarch64 (Apple Silicon) with g++-13 |
| CUDA | Not tested | Requires NVIDIA GPU + CUDA toolkit |
| HIP/ROCm | Not tested | Requires AMD GPU + ROCm stack |

## Mathematical Foundation

### Standard Softmax Definition

The softmax function transforms a vector of real numbers into a probability distribution:

    softmax(x_i) = exp(x_i) / sum(exp(x_j)) for j = 1 to N

### The Numerical Stability Problem

Problem: For large values of x, exp(x) overflows (e.g., exp(1000) = infinity)

### Stable Softmax Derivation

Key Insight: Softmax is shift-invariant!

Proof:
    softmax(x - c)_i = exp(x_i - c) / SUM_j exp(x_j - c)
                     = exp(x_i) * exp(-c) / [exp(-c) * SUM_j exp(x_j)]
                     = exp(x_i) / SUM_j exp(x_j)
                     = softmax(x)_i

Solution: Subtract max(x) before computing exp:
    c = max(x)
    softmax(x)_i = exp(x_i - c) / SUM_j exp(x_j - c)

Now all exponents are <= 0, so exp values are in (0, 1].

### Online Softmax (Flash Attention Optimization)

Traditional approach requires 3 passes:
1. Pass 1: Find max(x)
2. Pass 2: Compute sum of exp(x_i - max)
3. Pass 3: Compute softmax values

Online Softmax reduces to 2 passes using incremental updates:
    m_new = max(m_old, x_i)
    d_new = d_old * exp(m_old - m_new) + exp(x_i - m_new)

## Optimization Levels

### Level 1: Naive - Simple loop, no numerical stability
### Level 2: Stable - Subtracts max, 3 passes
### Level 3: Online/Fused - Single pass max+sum, shuffle reductions

## Directory Structure

    softmax-optimization/
    ├── README.md
    ├── cuda/
    │   ├── naive.cu
    │   ├── optimized.cu
    │   └── super_optimized.cu
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
    nvcc -O3 -arch=sm_80 naive.cu -o naive
    nvcc -O3 -arch=sm_80 optimized.cu -o optimized
    nvcc -O3 -arch=sm_80 super_optimized.cu -o super_optimized

### HIP (requires AMD GPU)
    hipcc -O3 naive.hip -o naive
    hipcc -O3 optimized.hip -o optimized
    hipcc -O3 super_optimized.hip -o super_optimized
