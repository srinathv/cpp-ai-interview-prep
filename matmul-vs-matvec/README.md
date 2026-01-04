# Matrix-Matrix vs Matrix-Vector Performance Analysis

This directory contains a comprehensive performance comparison between matrix-matrix multiplication and matrix-vector multiplication, implemented across different parallelization strategies.

## Directory Structure

```
14_matmul_vs_matvec/
├── cpu/                    # Single-threaded CPU implementations
├── multithreaded/          # Multi-threaded CPU implementations (OpenMP)
├── cuda/                   # CUDA GPU implementations
└── README.md               # This file
```

## Performance Analysis Objectives

1. **Computational Complexity Comparison**
   - Mat*Mat: O(n³) operations
   - Mat*Vec: O(n²) operations

2. **Memory Access Patterns**
   - Cache efficiency differences
   - Memory bandwidth utilization
   - Data reuse opportunities

3. **Parallelization Efficiency**
   - CPU multi-threading scalability
   - GPU kernel performance
   - Load balancing characteristics

## Implementations

### CPU Single-Threaded (cpu/)
Baseline implementations for both operations to establish single-core performance characteristics.

### Multi-threaded (multithreaded/)
OpenMP parallelized versions comparing:
- Thread scalability
- Overhead analysis
- Cache effects with multiple threads

### CUDA (cuda/)
GPU implementations comparing:
- Kernel launch overhead
- Memory transfer costs
- Compute throughput
- Occupancy differences

## Build and Run

See individual directories for specific build instructions.

## Key Insights

This analysis helps understand:
- When GPU acceleration becomes beneficial
- Memory vs compute bound characteristics
- Optimal parallelization strategies for different problem sizes
