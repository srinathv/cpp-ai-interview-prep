# C++ AI Interview Prep

A growing collection of C++ examples and practice problems for AI/ML library developer interviews, specifically targeting roles like ROCm-DS/hipDF.

## Repository Structure

This repository is organized by topic, with each directory containing focused examples and explanations:

```
cpp-ai-interview-prep/
├── 01_cpp_fundamentals/       # Modern C++ features (C++17/20/23)
├── 02_data_structures/        # Essential data structures
├── 03_algorithms/             # Common algorithms and patterns
├── 04_memory_management/      # Smart pointers, RAII, allocation
├── 05_parallel_programming/   # Threading, OpenMP, async
├── 06_gpu_hip/                # GPU programming with HIP/ROCm
└── 07_ai_ml_concepts/         # DataFrames, tensors, performance
```

## Quick Start

```bash
# Clone the repository
git clone git@github.com:srinathv/cpp-ai-interview-prep.git
cd cpp-ai-interview-prep

# Build all examples (coming soon)
make all

# Or explore individual topics
cd 01_cpp_fundamentals
```

## Topics Covered

### 1. C++ Fundamentals
- Modern C++ features (auto, lambdas, ranges)
- Move semantics and perfect forwarding
- Templates and metaprogramming
- STL containers and algorithms

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

### 6. GPU/HIP Programming
- HIP basics and kernel launch
- Memory management (host/device)
- Parallel patterns on GPU
- ROCm ecosystem overview

### 7. AI/ML Concepts
- DataFrame operations
- Tensor manipulation
- Performance optimization
- Numerical computing patterns

## Study Plan

### 3-Day Interview Prep
1. **Day 1**: Focus on C++ fundamentals and data structures
2. **Day 2**: Algorithms and memory management
3. **Day 3**: Parallel programming and AI/ML concepts

### Ongoing Learning
- Add new examples as you learn
- Practice LeetCode problems
- Contribute real-world patterns

## Contributing

This is a living repository! Feel free to:
- Add new examples
- Improve existing code
- Add explanations and comments
- Share interview experiences

## Resources

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [Modern C++ Features](https://en.cppreference.com/)
- [LeetCode](https://leetcode.com/)

## License

MIT
