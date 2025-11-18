# Dictionary Search - Parallel Computing Implementations

A comprehensive C++ interview coding example demonstrating dictionary prefix search implemented with multiple parallel computing approaches.

## Overview

This project showcases a dictionary search application that finds words matching 2-3 letter input prefixes. The same algorithm is implemented using five different approaches:

1. **Serial** - Traditional single-threaded C++ with STL containers
2. **Pthreads** - Multi-threaded CPU parallelism using POSIX threads
3. **OpenMPI** - Distributed computing across multiple processes
4. **CUDA** - GPU-accelerated search for NVIDIA GPUs
5. **HIP/ROCm** - GPU-accelerated search for AMD GPUs

## Dictionary

The program uses a dictionary of 20 technical words:
- algorithm, binary, compiler, database, encryption
- framework, gateway, hashmap, interface, java
- kernel, library, memory, network, object
- protocol, query, runtime, socket, thread

## Features Demonstrated

### Core C++ Concepts
- ✅ **std::map** - Prefix mapping for efficient lookups
- ✅ **for loops** - Traditional iteration patterns
- ✅ **STL containers** - vectors, maps, strings
- ✅ **String manipulation** - substr, transform, case-insensitive comparison

### Parallel Computing
- ✅ **Pthreads** - POSIX threads for shared-memory parallelism
- ✅ **OpenMPI** - Message Passing Interface for distributed computing
- ✅ **CUDA** - NVIDIA GPU programming
- ✅ **HIP** - AMD GPU programming (ROCm)

### Software Engineering
- ✅ **Performance measurement** - Timing with chrono
- ✅ **Modular design** - Separate implementations
- ✅ **Cross-platform builds** - CMake configuration
- ✅ **Error handling** - Robust error checking

## Project Structure

```
dictionary-search/
├── CMakeLists.txt              # Main build configuration
├── README.md                   # This file
├── serial/
│   ├── CMakeLists.txt
│   └── dictionary_search.cpp   # Serial implementation
├── pthreads/
│   ├── CMakeLists.txt
│   └── dictionary_search_pthreads.cpp
├── mpi/
│   ├── CMakeLists.txt
│   └── dictionary_search_mpi.cpp
├── cuda/
│   ├── CMakeLists.txt
│   └── dictionary_search.cu    # CUDA kernel implementation
└── hip/
    ├── CMakeLists.txt
    └── dictionary_search.cpp    # HIP kernel implementation
```

## Building

### Prerequisites

**All versions:**
- CMake 3.18 or higher
- C++17 compatible compiler (g++, clang++)

**Pthreads version:**
- pthread library (usually included with Linux/macOS)

**MPI version:**
- OpenMPI or MPICH installation
```bash
# Ubuntu/Debian
sudo apt-get install libopenmpi-dev openmpi-bin

# macOS
brew install open-mpi

# Fedora/RHEL
sudo dnf install openmpi openmpi-devel
```

**CUDA version:**
- NVIDIA CUDA Toolkit 11.0 or higher
- NVIDIA GPU with compute capability 3.5+
```bash
# Download from: https://developer.nvidia.com/cuda-downloads
```

**HIP version:**
- AMD ROCm 4.0 or higher
- AMD GPU (Radeon Instinct, Radeon VII, or newer)
```bash
# Ubuntu 20.04/22.04
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-hip-sdk
```

### Build Instructions

```bash
# Create build directory
mkdir build && cd build

# Configure (CMake will detect available features)
cmake ..

# Build all available versions
cmake --build .

# Or build specific version
cmake --build . --target dictionary_search_serial
cmake --build . --target dictionary_search_pthreads
cmake --build . --target dictionary_search_mpi
cmake --build . --target dictionary_search_cuda
cmake --build . --target dictionary_search_hip
```

### Building Individual Versions

**Serial version:**
```bash
cd serial
g++ -std=c++17 -O3 dictionary_search.cpp -o dictionary_search_serial
```

**Pthreads version:**
```bash
cd pthreads
g++ -std=c++17 -O3 -pthread dictionary_search_pthreads.cpp -o dictionary_search_pthreads
```

**MPI version:**
```bash
cd mpi
mpic++ -std=c++17 -O3 dictionary_search_mpi.cpp -o dictionary_search_mpi
```

**CUDA version:**
```bash
cd cuda
nvcc -std=c++17 -O3 dictionary_search.cu -o dictionary_search_cuda
```

**HIP version:**
```bash
cd hip
hipcc -std=c++17 -O3 dictionary_search.cpp -o dictionary_search_hip
```

## Running

### Serial Version
```bash
./serial/dictionary_search_serial
```

**Output:**
- Displays all dictionary words
- Builds prefix map
- Runs automated searches for common prefixes
- Provides interactive search prompt

**Features:**
- Uses `std::map<string, vector<string>>` for prefix storage
- Traditional `for` loops for iteration
- Case-insensitive prefix matching
- Performance timing for each search

### Pthreads Version
```bash
./pthreads/dictionary_search_pthreads
```

**Output:**
- Shows thread work distribution (4 threads by default)
- Parallel prefix map building
- Multi-threaded search execution
- Performance comparison

**Features:**
- Creates 4 worker threads
- Round-robin word distribution
- Thread-safe result aggregation
- Mutex-protected output

### MPI Version
```bash
# Run with 4 processes
mpirun -np 4 ./mpi/dictionary_search_mpi

# Run with different process counts
mpirun -np 2 ./mpi/dictionary_search_mpi
mpirun -np 8 ./mpi/dictionary_search_mpi
```

**Output:**
- Process rank and size information
- Word distribution across processes
- Distributed search results
- Inter-process communication timing

**Features:**
- Distributes dictionary across MPI ranks
- Each process builds local prefix map
- MPI_Reduce for result aggregation
- MPI_Barrier for synchronization

### CUDA Version
```bash
./cuda/dictionary_search_cuda
```

**Output:**
- GPU device information (name, compute capability)
- Dictionary and kernel parameters
- GPU-accelerated search results
- Transfer overhead timing

**Features:**
- Parallel string matching on GPU
- Flat memory layout for coalesced access
- Thread-per-word mapping
- Case-insensitive comparison in kernel

### HIP Version
```bash
./hip/dictionary_search_hip
```

**Output:**
- AMD GPU information (name, compute units, memory)
- Dictionary search results
- ROCm platform details
- Performance metrics

**Features:**
- Compatible with AMD GPUs
- Same algorithm as CUDA version
- HIP runtime API
- Portable across AMD GPU architectures

## Algorithm Explanation

### Prefix Map Building

All implementations use a prefix map approach:

```cpp
map<string, vector<string>> prefixMap;

// For each word in dictionary
for (word : dictionary) {
    // Generate 2-letter prefix
    prefix2 = word.substr(0, 2);
    prefixMap[prefix2].push_back(word);

    // Generate 3-letter prefix
    prefix3 = word.substr(0, 3);
    prefixMap[prefix3].push_back(word);
}
```

**Example:**
- "algorithm" → maps to prefixes "al" and "alg"
- "binary" → maps to prefixes "bi" and "bin"

### Search Process

1. Convert input prefix to lowercase
2. Look up prefix in map: `O(log n)` complexity
3. Return matching words vector

### Parallelization Strategies

**Pthreads:**
- Divide dictionary into chunks
- Each thread processes its chunk
- Merge results from all threads

**MPI:**
- Distribute words across processes
- Each process searches locally
- Gather results to root process

**GPU (CUDA/HIP):**
- One thread per dictionary word
- Parallel prefix comparison
- Results copied back to host

## Performance Considerations

### Small Dictionary Overhead

With only 20 words, parallel/GPU versions may be slower than serial due to:
- Thread creation overhead
- Memory transfer latency (GPU)
- Inter-process communication (MPI)

**This is intentional for educational purposes!**

### When Each Approach Shines

**Serial:**
- Small datasets (< 1000 words)
- Simple requirements
- Single machine

**Pthreads:**
- Medium datasets (1K-100K words)
- Shared memory systems
- Low-latency requirements

**MPI:**
- Large datasets (> 1M words)
- Distributed systems
- Multiple machines

**GPU (CUDA/HIP):**
- Very large datasets (> 1M words)
- Massively parallel searches
- High-throughput requirements

## Interview Discussion Points

### Technical Topics Covered

1. **Data Structures:**
   - When to use std::map vs std::unordered_map
   - Trade-offs between different container types
   - Memory layout for GPU transfers

2. **Parallelization:**
   - Shared memory vs distributed memory
   - Thread synchronization and race conditions
   - GPU thread hierarchy (blocks, threads)

3. **Performance:**
   - Big-O complexity analysis
   - Amdahl's Law and speedup limits
   - Cache effects and memory access patterns

4. **Design Patterns:**
   - Data parallelism
   - Map-reduce pattern
   - Producer-consumer pattern (threads)

5. **C++ Best Practices:**
   - RAII and resource management
   - Move semantics
   - Const correctness
   - Template usage

### Questions You Might Be Asked

1. **Why use a prefix map instead of linear search?**
   - O(log n) lookup vs O(n) for each search
   - Pre-processing trades space for time

2. **How would you scale this to millions of words?**
   - Trie data structure
   - Distributed hash tables
   - GPU with multiple kernels

3. **What are the bottlenecks in each implementation?**
   - Serial: CPU computation time
   - Pthreads: Thread creation overhead
   - MPI: Network communication
   - GPU: Memory transfer bandwidth

4. **How would you handle Unicode/UTF-8?**
   - Wide character support
   - Locale-aware comparison
   - ICU library integration

5. **What about memory usage?**
   - Duplicate storage in prefix map
   - Trade-offs: time vs space
   - Compression techniques

## Extending the Example

### Easy Extensions
- Increase dictionary size (load from file)
- Support variable-length prefixes
- Add fuzzy matching (Levenshtein distance)
- Case-sensitive option

### Medium Extensions
- Trie data structure implementation
- Benchmark harness with graphs
- REST API wrapper
- Multi-threaded file loading

### Advanced Extensions
- Distributed hash table (DHT)
- Multiple GPU support
- Async/await pattern
- Real-time streaming search

## Common Pitfalls & Solutions

### Pthreads
**Pitfall:** Race conditions in result collection
**Solution:** Use mutex locks or atomic operations

### MPI
**Pitfall:** Deadlocks from improper barriers
**Solution:** Ensure all processes hit synchronization points

### CUDA
**Pitfall:** Out-of-bounds memory access
**Solution:** Always check `idx < size` in kernel

### HIP
**Pitfall:** Incorrect memory management
**Solution:** Match every malloc with free, use RAII

## Benchmarking Tips

For accurate performance measurement:

1. **Warm up the code:**
   - Run searches once before timing
   - Eliminates cold-start effects

2. **Multiple runs:**
   - Average over 100+ iterations
   - Report median and standard deviation

3. **Isolate components:**
   - Time only the search, not I/O
   - Separate build time from search time

4. **Control variables:**
   - Fix CPU frequency (disable turbo boost)
   - Close background applications
   - Use consistent test data

## Resources

### C++ & STL
- [cppreference.com](https://en.cppreference.com/)
- Effective C++ by Scott Meyers

### Parallel Programming
- [OpenMP Tutorial](https://www.openmp.org/)
- [MPI Tutorial](https://mpitutorial.com/)

### GPU Programming
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [HIP Programming Guide](https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html)

### Books
- "Programming Massively Parallel Processors" by Kirk & Hwu
- "An Introduction to Parallel Programming" by Pacheco
- "Parallel Programming in C with MPI and OpenMP" by Quinn

## License

This example code is provided for educational purposes. Feel free to use and modify for interview preparation and learning.

## Author Notes

This example demonstrates multiple parallel computing paradigms in a single, understandable problem. While the dictionary size is small for demonstration purposes, the patterns and techniques scale to production systems with millions of entries.

Key interview strengths this showcases:
- ✅ Understanding of multiple parallel paradigms
- ✅ Performance-conscious programming
- ✅ Cross-platform development
- ✅ Modern C++ practices
- ✅ Build system configuration (CMake)
- ✅ Clear documentation and code organization

Good luck with your interview preparation! 🚀
