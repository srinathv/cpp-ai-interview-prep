# C++ AI Interview Prep

A growing collection of C++ and CUDA/HIP examples for AI/ML library developer interviews, specifically targeting roles like ROCm-DS/hipDF.

## Repository Structure

This repository is organized by topic, with each directory containing focused examples and explanations:

```
cpp-ai-interview-prep/
├── 01_cpp_fundamentals/       # Modern C++ features (C++17/20/23) ✓
├── 02_data_structures/        # Essential data structures
├── 03_algorithms/             # Common algorithms and patterns
├── 04_memory_management/      # Smart pointers, RAII, allocation
├── 05_parallel_programming/   # Threading, OpenMP, async
├── 06_gpu_hip/                # GPU programming with HIP/ROCm ✓
├── 07_ai_ml_concepts/         # DataFrames, tensors, performance
├── 08_cuda_basics/            # CUDA fundamentals ✓
└── 09_cuda_advanced/          # Advanced CUDA patterns
```

## Quick Start

```bash
# Clone the repository
git clone git@github.com:srinathv/cpp-ai-interview-prep.git
cd cpp-ai-interview-prep

# Try C++ examples
cd 01_cpp_fundamentals
g++ -std=c++20 auto_and_decltype.cpp && ./a.out

# Try CUDA examples (requires CUDA toolkit)
cd ../08_cuda_basics
nvcc hello_cuda.cu && ./a.out
nvcc vector_add.cu && ./a.out
```

## Topics Covered

### 1. C++ Fundamentals ✓
- Modern C++ features (auto, lambdas, ranges)
- Move semantics and perfect forwarding
- Templates and metaprogramming
- STL containers and algorithms
- **Examples**: auto_and_decltype.cpp, lambdas.cpp

### 2. Data Structures
- Arrays, vectors, linked lists
- Hash tables and maps
- Trees and graphs
- Custom data structures

### 3. Algorithms
- Sorting and searching
- Dynamic programming
- Graph algorithms
- String manipulation

### 4. Memory Management
- Smart pointers (unique_ptr, shared_ptr)
- RAII patterns
- Custom allocators
- Memory pools

### 5. Parallel Programming
- std::thread and std::async
- OpenMP directives
- Thread-safe data structures
- Lock-free programming

### 6. GPU/HIP Programming ✓
- HIP basics and kernel launch
- CUDA vs HIP comparison
- Memory management (host/device)
- Parallel patterns on GPU
- ROCm ecosystem overview

### 7. AI/ML Concepts
- DataFrame operations
- Tensor manipulation
- Performance optimization
- Numerical computing patterns

### 8. CUDA Basics ✓
- Hello CUDA and device queries
- Vector addition (parallel map)
- Matrix multiplication (2D indexing)
- Memory management patterns
- Thread hierarchy and indexing
- **Examples**: hello_cuda.cu, vector_add.cu, matrix_multiply.cu

### 9. CUDA Advanced
- Shared memory optimization
- Reduction patterns
- Warp-level primitives
- Memory coalescing
- Performance tuning

## Study Plan

### 3-Day Interview Prep
1. **Day 1**: C++ fundamentals, CUDA basics, GPU concepts
2. **Day 2**: Data structures, algorithms, memory management
3. **Day 3**: Parallel programming, HIP/ROCm, AI/ML patterns

### Week-Long Deep Dive
- **Days 1-2**: Master modern C++ and STL
- **Days 3-4**: GPU programming (CUDA/HIP)
- **Days 5-6**: Algorithms and optimization
- **Day 7**: AI/ML specific patterns, practice problems

### Ongoing Learning
- Add new examples as you learn
- Practice LeetCode problems
- Contribute real-world patterns

## Building Examples

### C++ Examples
```bash
g++ -std=c++20 -O2 example.cpp -o example
./example
```

### CUDA Examples
```bash
nvcc -o example example.cu
./example

# With optimization
nvcc -O3 -arch=sm_80 -o example example.cu
```

### HIP Examples (coming soon)
```bash
hipcc -o example example.hip.cpp
./example
```

## Contributing

This is a living repository! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

Topics needed:
- [ ] LeetCode-style problems
- [ ] HIP example implementations
- [ ] Shared memory CUDA examples
- [ ] Real interview questions
- [ ] Performance optimization patterns

## Resources

### GPU Programming
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)

### Modern C++
- [C++17/20 Features](https://en.cppreference.com/)
- [Modern C++ Design Patterns](https://refactoring.guru/design-patterns/cpp)

### Practice
- [LeetCode](https://leetcode.com/)
- [HackerRank C++](https://www.hackerrank.com/domains/cpp)

## License

MIT
