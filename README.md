# C++ AI/ML Interview Prep

A comprehensive collection of C++ examples for AI/ML library developer interviews, covering modern C++, GPU programming (CUDA/HIP), parallel computing, and performance optimization.

## Overview

This repository provides hands-on examples and detailed explanations for topics commonly encountered in technical interviews for roles like ROCm, CUDA library developers, HPC engineers, and AI/ML infrastructure positions.

**All examples are verified to compile and run** (where hardware permits).

## Repository Structure

Organized by topic with consistent naming (kebab-case, no number prefixes):

```
cpp-ai-interview-prep/
├── cpp-fundamentals/          # Modern C++ (C++17/20/23) ✓
├── data-structures/           # Essential data structures ✓
├── algorithms/                # Common algorithms and patterns ✓
├── memory-management/         # Smart pointers, RAII, move semantics ✓
├── parallel-programming/      # Threading, async/futures ✓
├── ai-ml-concepts/            # Neural networks, linear algebra ✓
├── virtual-vs-template/       # Runtime vs compile-time polymorphism ✓
│
├── gpu-hip/                   # AMD ROCm/HIP programming ⚠️
├── cuda-basics/               # NVIDIA CUDA fundamentals ⚠️
│
├── transpose/                 # GPU matrix transpose optimizations ⚠️
├── matrix-multiply/           # GPU matrix multiplication ⚠️
├── matrix-vector/             # GPU matrix-vector operations ⚠️
├── reductions/                # GPU reduction algorithms ⚠️
├── matmul-vs-matvec/          # Performance comparison (CPU/GPU) ✓
├── dgemm-gpu-techniques/      # Double-precision GEMM optimizations ⚠️
├── memory-allocators/         # Custom memory allocators benchmark ✓
│
├── dictionary-search/         # Prefix search algorithms ✓
├── heap-vs-stack/             # Memory allocation patterns ✓
├── max-increase-skyline/      # Optimization problem example ✓
└── segfault-examples/         # Common memory errors ✓
```

**Legend:**
- ✓ = Tested and working on macOS ARM
- ⚠️ = Requires specific hardware (NVIDIA GPU, AMD GPU)

## Quick Start

```bash
# Clone the repository
git clone git@codeberg.org:srinathv/cpp-ai-interview-prep.git
cd cpp-ai-interview-prep

# Test C++ fundamentals
cd cpp-fundamentals
g++ -std=c++20 auto_and_decltype.cpp -o test && ./test
g++ -std=c++20 lambdas.cpp -o test && ./test

# Test data structures
cd ../data-structures
g++ -std=c++20 binary_tree.cpp -o test && ./test

# Test performance comparison
cd ../matmul-vs-matvec/cpu
make && make run

# Test memory allocators
cd ../../memory-allocators/benchmarks
make && ./basic_bench
```

## Core Topics

### 1. C++ Fundamentals ✓
**Modern C++ features essential for high-performance computing**

- **Auto and decltype**: Type deduction
- **Lambdas**: Inline functions and closures
- **Move semantics**: Zero-copy optimization
- **Smart pointers**: RAII and ownership
- **Templates**: Generic programming
- **Ranges**: Modern STL algorithms (C++20)
- **Concepts**: Compile-time constraints (C++20)
- **Coroutines**: Async programming (C++20)

**Files:** `cpp-fundamentals/`

### 2. Virtual Functions vs Templates ✓
**Runtime vs compile-time polymorphism deep dive**

- How vtables work (memory layout, call mechanism)
- Template instantiation process
- Performance comparison (3x faster with templates)
- Memory overhead analysis (8 bytes vptr vs none)
- When to use each approach
- Assembly-level explanations

**Files:** `virtual-vs-template/`

### 3. Data Structures ✓
**Fundamental building blocks**

- Binary trees (BST traversal, validation)
- Graphs (BFS, DFS, island counting)
- Hash tables (collision handling)
- Linked lists (reversal, cycles)
- Vectors (dynamic arrays, custom implementation)

**Files:** `data-structures/`

### 4. Algorithms ✓
**Essential algorithmic patterns**

- **Dynamic Programming**: Fibonacci, LCS, LIS, knapsack
- **Searching**: Linear, binary (recursive/iterative)
- **Sorting**: Bubble, quick, merge, heap

**Files:** `algorithms/`

### 5. Memory Management ✓
**Resource ownership and lifetime**

- **Smart Pointers**: `unique_ptr`, `shared_ptr`, `weak_ptr`
- **RAII**: Resource acquisition patterns
- **Move Semantics**: Efficient transfers
- Stack vs heap allocation
- Use-after-free, buffer overflow examples
- Memory allocator benchmarks

**Files:** `memory-management/`, `heap-vs-stack/`, `segfault-examples/`, `memory-allocators/`

### 6. Parallel Programming ✓
**Concurrent execution patterns**

- **Threading**: `std::thread`, mutexes, condition variables
- **Async**: `std::async`, `std::future`, `std::promise`
- **Shared futures**: Multiple readers
- Thread-safe counters (atomic vs mutex)

**Files:** `parallel-programming/`

### 7. AI/ML Concepts ✓
**Mathematical foundations**

- **Linear Regression**: Gradient descent implementation
- **Logistic Regression**: Binary classification
- **Neural Networks**: Backpropagation, XOR problem
- **Matrix Operations**: Multiplication, transpose, sigmoid
- Custom implementations (no external libraries)

**Files:** `ai-ml-concepts/`

### 8. GPU Programming (CUDA/HIP) ⚠️

**CUDA (NVIDIA)**
- Kernel launch syntax: `<<<blocks, threads>>>`
- Thread hierarchy: grids, blocks, warps
- Memory: global, shared, constant
- Synchronization: `__syncthreads()`
- Warp-level primitives

**HIP (AMD ROCm)**
- Portable GPU code (AMD + NVIDIA)
- `hipLaunchKernelGGL()` syntax
- Wavefront vs warp (64 vs 32 threads)
- ROCm ecosystem

**Files:** `cuda-basics/`, `gpu-hip/`

### 9. GPU Optimizations ⚠️

**Matrix Transpose**: Coalesced memory access, bank conflict avoidance  
**Matrix Multiply**: Tiled algorithms, shared memory  
**Reductions**: Warp shuffles, atomic operations  
**DGEMM**: Double-precision GEMM techniques

**Files:** `transpose/`, `matrix-multiply/`, `reductions/`, `dgemm-gpu-techniques/`

### 10. Specialized Examples ✓

- **Dictionary Search**: Prefix matching, trie structures
- **Max Increase Skyline**: Optimization problem
- **MatMul vs MatVec**: CPU performance comparison

**Files:** `dictionary-search/`, `max-increase-skyline/`, `matmul-vs-matvec/`

## Performance Insights

### Virtual vs Template (from benchmarks)
```
Virtual functions:
  Time: 96 ms
  Memory: 8 bytes vptr overhead per object
  Mechanism: 2 pointer dereferences per call

Templates:
  Time: 0 ms (compiler optimized)
  Memory: No overhead
  Mechanism: Direct calls, fully inlined
```

### Memory Allocators
```
System malloc:
  Small (64B): 33M ops/sec
  Medium (4KB): 2.6M ops/sec
  Large (1MB): 1.3M ops/sec
```

### Matrix Operations (CPU)
```
Size 1024:
  Mat*Mat: 921 ms (2.3 GFLOPS)
  Mat*Vec: 0.46 ms (4.5 GFLOPS)
  Ops ratio: 1024x
```

## Building Examples

### C++ Examples
```bash
# Basic compilation
g++ -std=c++20 -O2 example.cpp -o example

# With threading
g++ -std=c++20 -pthread example.cpp -o example

# With optimization
g++ -std=c++20 -O3 -march=native example.cpp -o example
```

### CUDA Examples (requires NVIDIA GPU)
```bash
nvcc -o example example.cu
nvcc -O3 -arch=sm_80 example.cu  # RTX 30/40 series
nvcc -O3 -arch=sm_75 example.cu  # RTX 20 series
```

### HIP Examples (requires AMD GPU)
```bash
hipcc -o example example.hip.cpp
hipcc -O3 example.hip.cpp
```

### Using Makefiles
```bash
cd <directory>
make        # Build all
make run    # Build and run
make clean  # Clean binaries
```

## Study Plans

### Quick Interview Prep (3-5 hours)
1. **C++ Fundamentals** (1 hour)
   - Auto, lambdas, move semantics
   - Smart pointers

2. **Virtual vs Template** (30 min)
   - Vtable mechanism
   - Performance trade-offs

3. **Data Structures** (1 hour)
   - Trees, graphs, hash tables

4. **Algorithms** (1 hour)
   - DP, searching, sorting

5. **Memory & Performance** (1 hour)
   - RAII, allocators
   - Parallel programming basics

### Week-Long Deep Dive
- **Days 1-2**: Modern C++ (fundamentals + polymorphism)
- **Day 3**: Data structures and algorithms
- **Day 4**: Memory management and debugging
- **Day 5**: Parallel programming (CPU)
- **Days 6-7**: GPU programming (if applicable)

### Targeted Prep by Role

**AI/ML Library Developer (ROCm/CUDA)**
- Focus: `gpu-hip/`, `cuda-basics/`, `matrix-multiply/`, `ai-ml-concepts/`
- Key topics: Kernel optimization, memory coalescing, shared memory

**HPC Engineer**
- Focus: `parallel-programming/`, `memory-allocators/`, `matmul-vs-matvec/`
- Key topics: Threading, NUMA, cache optimization

**C++ Library Developer**
- Focus: `cpp-fundamentals/`, `virtual-vs-template/`, `memory-management/`
- Key topics: Templates, move semantics, zero-cost abstractions

## Interview Topics Covered

### Conceptual Questions
✓ How do vtables work in C++?  
✓ What's the difference between virtual functions and templates?  
✓ Explain move semantics  
✓ When should you use smart pointers?  
✓ How does CUDA memory hierarchy work?  
✓ What are warp divergence and bank conflicts?

### Coding Questions
✓ Implement binary tree traversal  
✓ Write a thread-safe counter  
✓ Optimize matrix multiplication  
✓ Implement smart pointer (RAII)  
✓ Write CUDA kernel for vector addition

### Performance Questions
✓ Why are templates faster than virtual functions?  
✓ How to avoid cache misses in matrix operations?  
✓ What's the overhead of std::function vs lambdas?  
✓ GPU memory coalescing techniques

## Resources

### Official Documentation
- [C++ Reference](https://en.cppreference.com/) - Modern C++ features
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)

### Books
- *Effective Modern C++* - Scott Meyers
- *C++ Concurrency in Action* - Anthony Williams
- *Programming Massively Parallel Processors* - Kirk & Hwu

### Practice
- [LeetCode](https://leetcode.com/) - Algorithm practice
- [HackerRank C++](https://www.hackerrank.com/domains/cpp)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Needed:**
- More GPU optimization examples
- Additional AI/ML algorithms
- Real interview questions from companies
- Performance profiling examples

## License

MIT

---

**Note**: This repository is actively maintained. Star/watch for updates!
