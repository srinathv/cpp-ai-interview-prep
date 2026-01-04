# Memory Allocators Comparison

A comprehensive guide and benchmark suite comparing different memory allocator implementations across operating systems.

## Overview

Memory allocators are critical components that manage heap memory allocation and deallocation. Different implementations offer varying trade-offs between speed, memory efficiency, fragmentation, thread-safety, and security.

## Allocators Covered

### 1. **glibc malloc (ptmalloc2)** - Default on Linux
- **Developer**: GNU Project
- **Used by**: Most Linux systems by default
- **Best for**: General-purpose applications

### 2. **tcmalloc** - Google's Thread-Caching Malloc
- **Developer**: Google
- **Used by**: Google services, Chrome, many high-performance applications
- **Best for**: Multi-threaded applications, high-throughput scenarios

### 3. **jemalloc** - Facebook's Allocator
- **Developer**: Jason Evans (FreeBSD), Facebook
- **Used by**: FreeBSD default, Firefox, Redis, MariaDB
- **Best for**: Multi-threaded applications, low fragmentation

### 4. **mimalloc** - Microsoft's Allocator
- **Developer**: Microsoft Research
- **Used by**: Microsoft cloud services
- **Best for**: Security-critical applications, low fragmentation

### 5. **Windows Heap API**
- **Developer**: Microsoft
- **Used by**: Windows applications by default
- **Best for**: Windows-native applications

## Quick Comparison

| Feature | glibc malloc | tcmalloc | jemalloc | mimalloc |
|---------|-------------|----------|----------|----------|
| **Speed (single)** | Good | Excellent | Very Good | Excellent |
| **Speed (multi)** | Moderate | Excellent | Excellent | Excellent |
| **Memory Overhead** | Moderate | Higher | Low | Low |
| **Fragmentation** | Moderate | Good | Excellent | Excellent |
| **Security** | Basic | Basic | Basic | Advanced |
| **Startup Memory** | ~2MB | ~13MB | ~9MB | ~4MB |

## Directory Structure

```
15_memory_allocators/
├── README.md                  # This file
├── COMPARISON.md              # Detailed comparison and analysis
├── INSTALL.md                 # Installation guide for each allocator
├── benchmarks/
│   ├── basic_bench.cpp        # Basic allocation/deallocation
│   ├── multithreaded_bench.cpp # Thread contention testing
│   ├── fragmentation_bench.cpp # Memory fragmentation analysis
│   └── realistic_bench.cpp    # Real-world workload simulation
├── examples/
│   ├── using_tcmalloc.cpp     # Example using tcmalloc
│   ├── using_jemalloc.cpp     # Example using jemalloc
│   ├── using_mimalloc.cpp     # Example using mimalloc
│   └── allocator_switch.cpp   # Runtime allocator switching
└── docs/
    ├── internals.md           # How each allocator works
    ├── tuning.md              # Performance tuning guide
    └── debugging.md           # Debugging memory issues
```

## Key Differences

### Performance Characteristics

**Single-threaded:**
- mimalloc and tcmalloc lead in throughput
- glibc malloc is competitive for small allocations

**Multi-threaded:**
- tcmalloc and jemalloc excel with >8 cores
- 4x performance improvement over glibc malloc on 32-core systems
- Per-thread caching reduces lock contention

### Memory Efficiency

**Low fragmentation:**
1. jemalloc (best)
2. mimalloc
3. tcmalloc
4. glibc malloc

**Startup memory:**
1. glibc malloc (~2MB)
2. mimalloc (~4MB)
3. jemalloc (~9MB)
4. tcmalloc (~13MB)

### Use Case Recommendations

**Web Servers (high concurrency):**
- 1st choice: tcmalloc or jemalloc
- Reduces memory usage by 4GB+ on heavily loaded systems

**Long-running Services:**
- 1st choice: mimalloc or jemalloc
- Better fragmentation resistance over time

**Security-critical Applications:**
- 1st choice: mimalloc (with secure mode)
- Guard pages, randomization, encrypted free lists

**Database Systems:**
- 1st choice: jemalloc
- Used by Redis, MariaDB, PostgreSQL builds
- Excellent multi-thread performance

**Default/General Purpose:**
- glibc malloc sufficient for most applications
- Zero configuration needed

## Quick Start

### Install Allocators

```bash
# Ubuntu/Debian
sudo apt-get install libtcmalloc-minimal4 libgoogle-perftools-dev
sudo apt-get install libjemalloc-dev

# macOS
brew install gperftools
brew install jemalloc
brew install mimalloc

# Build mimalloc from source
git clone https://github.com/microsoft/mimalloc
cd mimalloc && mkdir build && cd build
cmake .. && make && sudo make install
```

### Use Allocator

```bash
# Link at compile time
g++ -o app app.cpp -ltcmalloc
g++ -o app app.cpp -ljemalloc
g++ -o app app.cpp -lmimalloc

# Use via LD_PRELOAD (Linux)
LD_PRELOAD=/usr/lib/libtcmalloc.so ./app
LD_PRELOAD=/usr/lib/libjemalloc.so ./app

# Use via DYLD_INSERT_LIBRARIES (macOS)
DYLD_INSERT_LIBRARIES=/usr/local/lib/libtcmalloc.dylib ./app
```

### Run Benchmarks

```bash
cd benchmarks
make
./basic_bench
./multithreaded_bench
./fragmentation_bench
```

## Performance Impact Examples

### MySQL/MariaDB
- With jemalloc: 20-30% better performance
- 4GB less memory usage under load

### Redis
- With jemalloc: Default choice
- Reduced fragmentation in long-running instances

### Chrome Browser
- With tcmalloc: Faster page loading
- Better multi-tab performance

## When to Switch Allocators

Consider switching from default allocator when:

1. **Multi-threaded application with >8 cores** → tcmalloc or jemalloc
2. **Long-running service with fragmentation** → mimalloc or jemalloc
3. **Security-critical application** → mimalloc with secure mode
4. **High allocation rate** → tcmalloc
5. **Memory-constrained environment** → jemalloc

## Further Reading

- [COMPARISON.md](COMPARISON.md) - Detailed technical comparison
- [INSTALL.md](INSTALL.md) - Platform-specific installation
- [docs/internals.md](docs/internals.md) - How allocators work
- [docs/tuning.md](docs/tuning.md) - Performance optimization

## References

- [tcmalloc Documentation](https://github.com/google/tcmalloc)
- [jemalloc Documentation](http://jemalloc.net/)
- [mimalloc Repository](https://github.com/microsoft/mimalloc)
- [glibc malloc internals](https://sourceware.org/glibc/wiki/MallocInternals)
