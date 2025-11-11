# CUDA Basics

Fundamental CUDA programming concepts for GPU acceleration.

## Topics

1. **hello_cuda.cu** - First CUDA program, kernel launch
2. **vector_add.cu** - Basic parallel vector addition
3. **memory_management.cu** - cudaMalloc, cudaMemcpy, cudaFree
4. **thread_hierarchy.cu** - Blocks, threads, grid dimensions
5. **shared_memory.cu** - Using shared memory for optimization
6. **reduction.cu** - Parallel reduction pattern
7. **matrix_multiply.cu** - Simple matrix multiplication

## CUDA Basics Concepts

### Kernel Launch Syntax
```cuda
kernel<<<gridSize, blockSize>>>(args);
```

### Thread Indexing
```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

### Memory Hierarchy
- **Global Memory**: Accessible by all threads (slow)
- **Shared Memory**: Per-block fast memory
- **Registers**: Per-thread fastest memory
- **Constant Memory**: Read-only cached memory

## Building CUDA Programs

```bash
# Compile CUDA code
nvcc -o program program.cu

# With optimization
nvcc -O3 -o program program.cu

# Specify compute capability (e.g., for A100)
nvcc -arch=sm_80 -o program program.cu
```

## Key Interview Topics

- Explain CUDA thread hierarchy (grid, block, thread)
- What is warp divergence?
- Difference between global and shared memory
- How to optimize memory access patterns (coalescing)
- When to use shared memory
- Basic parallel patterns (map, reduce, scan)

## Common Pitfalls

1. **Forgetting cudaDeviceSynchronize()** - Kernels are asynchronous
2. **Not checking CUDA errors** - Always check return values
3. **Inefficient memory access** - Non-coalesced reads/writes
4. **Warp divergence** - Branching within warps
5. **Shared memory bank conflicts** - Simultaneous access to same bank

## Useful Commands

```bash
# Check CUDA devices
nvidia-smi

# Get device properties
nvcc deviceQuery.cu -o deviceQuery && ./deviceQuery

# Profile CUDA code
nvprof ./program
```
