# GPU Programming with HIP

HIP (Heterogeneous-compute Interface for Portability) - AMD's GPU programming model compatible with both NVIDIA and AMD GPUs.

## CUDA vs HIP Quick Reference

### Basic Syntax Comparison

| Concept | CUDA | HIP |
|---------|------|-----|
| Include | `#include <cuda_runtime.h>` | `#include <hip/hip_runtime.h>` |
| Error type | `cudaError_t` | `hipError_t` |
| Success | `cudaSuccess` | `hipSuccess` |
| Memory alloc | `cudaMalloc` | `hipMalloc` |
| Memory copy | `cudaMemcpy` | `hipMemcpy` |
| Synchronize | `cudaDeviceSynchronize` | `hipDeviceSynchronize` |
| Kernel qualifier | `__global__` | `__global__` |
| Launch | `kernel<<<g,b>>>` | `hipLaunchKernelGGL(kernel, g, b, ...)` |

### Key Differences

1. **Portability**: HIP runs on both NVIDIA (via CUDA backend) and AMD (via ROCm)
2. **Kernel Launch**: HIP uses macro (also supports CUDA-style syntax)
3. **Tool Chain**: 
   - CUDA: `nvcc`
   - HIP: `hipcc` (wraps nvcc or amdclang++)

## Example: Vector Add in Both

### CUDA Version
```cuda
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

// Launch
vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
cudaDeviceSynchronize();
```

### HIP Version  
```cpp
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

// Launch (explicit)
hipLaunchKernelGGL(vectorAdd, blocks, threads, 0, 0, d_a, d_b, d_c, n);
hipDeviceSynchronize();

// Or CUDA-style (also works)
vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
hipDeviceSynchronize();
```

## Converting CUDA to HIP

AMD provides `hipify-perl` tool:
```bash
hipify-perl program.cu > program.hip.cpp
```

Most CUDA code converts automatically by replacing:
- `cuda` → `hip`
- `CUDA` → `HIP`

## When to Use What

### Use CUDA when:
- Targeting only NVIDIA GPUs
- Using NVIDIA-specific libraries (cuDNN, cuBLAS)
- Mature ecosystem needed

### Use HIP when:
- Need portability across vendors
- Targeting AMD GPUs (MI series)
- Want single source for multi-vendor

## ROCm Ecosystem (AMD)

- **ROCm**: AMD's GPU software platform
- **hipBLAS**: BLAS for HIP
- **rocFFT**: FFT library
- **MIOpen**: Deep learning primitives
- **rocPRIM**: Parallel primitives

## Interview Topics

- Explain CUDA vs HIP differences
- Why would you choose HIP over CUDA?
- How to write portable GPU code?
- ROCm vs CUDA ecosystem comparison
- Performance portability considerations

## Building HIP Programs

```bash
# Compile HIP code
hipcc -o program program.hip.cpp

# For NVIDIA backend
hipcc --amdgpu-target=nvptx64 -o program program.hip.cpp

# For AMD backend
hipcc --amdgpu-target=gfx908 -o program program.hip.cpp
```

## Coming Soon

Examples will be added showing:
- Basic HIP programs
- Memory management
- Parallel patterns
- Performance optimization
- CUDA to HIP porting
