# Interview Guide: Maximum Increase Keeping Skyline

## Problem Statement

A city has a 2D array `grid` where `grid[i][j]` represents the height of a building at position (i,j). 

Buildings can increase in height as long as the skyline remains unchanged when viewed from:
- North (top)
- South (bottom)  
- East (right)
- West (left)

Return the total height increase possible.

---

## Understanding the Problem

### Key Insights

1. **Skyline Constraint**: The maximum height in each row and column defines the skyline
2. **Maximum Possible Height**: For building at (i,j), max height = min(rowMax[i], colMax[j])
3. **Increase**: Difference between max possible and current height

### Example Walkthrough

```
Input:
3 0 8 4
2 4 5 7
9 2 6 3
0 3 1 0

Row Max: [8, 7, 9, 3]
Col Max: [9, 4, 8, 7]

Position (0,0): min(8,9) = 8, increase = 8-3 = 5
Position (0,1): min(8,4) = 4, increase = 4-0 = 4
...
Total = 35
```

---

## Solution Approach

### Algorithm Steps

1. Calculate maximum height for each row
2. Calculate maximum height for each column
3. For each cell, compute: min(rowMax[i], colMax[j]) - grid[i][j]
4. Sum all increases

### Time/Space Complexity

- Time: O(N²)
- Space: O(N)

---

## Common Interview Questions

### Q1: Can you optimize the space complexity?

**A:** Not really - we need O(N) space for row/column maximums. We could compute them on-the-fly but that would increase time complexity to O(N³).

### Q2: What if the grid is not square (M x N)?

**A:** Algorithm remains the same, just track M row maxes and N column maxes.

```cpp
vector<int> rowMax(M);
vector<int> colMax(N);
```

### Q3: How would you handle very large grids?

**A:** Several approaches:
1. **Multi-threading**: Parallelize with OpenMP (see `openmp_parallel.cpp`)
2. **GPU acceleration**: CUDA/ROCm for massive parallelism
3. **Memory mapping**: For grids too large for RAM
4. **Streaming**: Process in chunks if constraints allow

### Q4: Can you solve this in one pass?

**A:** No - we need row/column maxes first before calculating increases. However, we can:
- Compute row and column maxes in parallel
- Use SIMD to speed up max finding

### Q5: How would you test this solution?

**A:** Test cases:
1. **Given example**: Verify output = 35
2. **All same height**: Should return 0
3. **Single row/column**: Edge case
4. **Already at max**: Grid where each cell is already at min(rowMax, colMax)
5. **Large random grid**: Performance test

---

## Follow-up Optimization Questions

### CPU Optimizations

**Q: How can you optimize for modern CPUs?**

**A: Several techniques:**

1. **Cache-friendly access**: Process row-by-row
2. **SIMD vectorization**: Use AVX2 for parallel comparisons
3. **Loop unrolling**: Reduce loop overhead
4. **Compiler hints**: Use `#pragma` directives

See `cpu_optimized.cpp` for implementation.

### Parallel Optimizations

**Q: How would you parallelize this?**

**A: Independent operations can be parallelized:**

1. **Row maxes**: Each row independent → parallelize
2. **Column maxes**: Each column independent → parallelize  
3. **Sum calculation**: Use parallel reduction

```cpp
#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < N*N; ++i) {
    // compute increase
}
```

See `openmp_parallel.cpp`.

### GPU Optimizations

**Q: When would GPU acceleration help? How would you implement it?**

**A: GPUs excel for large grids (N > 1000):**

**Benefits:**
- Thousands of parallel threads
- High memory bandwidth
- Specialized for data-parallel tasks

**Implementation:**
1. Flatten 2D grid for coalesced access
2. Launch kernels for row max, col max, sum
3. Use shared memory for reductions
4. Warp shuffle for efficient communication

See `cuda_gpu.cu` and `rocm_gpu.cpp`.

---

## Code Quality Discussion Points

### Edge Cases

```cpp
// Empty grid
if (grid.empty() || grid[0].empty()) return 0;

// Single cell
if (N == 1) return 0;

// Non-square grid
int M = grid.size();
int N = grid[0].size();
```

### Error Handling

```cpp
// Validate input
for (auto& row : grid) {
    if (row.size() != N) {
        throw invalid_argument("Grid must be rectangular");
    }
}
```

### Const Correctness

```cpp
int maxIncrease(const vector<vector<int>>& grid) {
    // Don't modify input
}
```

---

## Performance Analysis

### Bottlenecks

1. **Column access**: Cache-unfriendly due to memory layout
2. **Three passes**: Could overlap computation
3. **Memory bandwidth**: For large grids, limited by RAM speed

### Profiling

**What to measure:**
- Cache miss rate
- CPU utilization
- Memory bandwidth
- Thread scaling

**Tools:**
- `perf stat` for cache statistics
- `time` for wall-clock time
- Custom benchmarking (see `benchmark.cpp`)

---

## Real-World Considerations

### Production Code

1. **Input validation**: Check for valid heights (non-negative)
2. **Integer overflow**: Use `long long` for very large grids
3. **Memory limits**: Validate grid size before allocation
4. **Thread safety**: Consider if called concurrently

### API Design

```cpp
class SkylineCalculator {
public:
    // Different implementations
    int compute(const Grid& grid, Strategy strategy = Strategy::AUTO);
    
    enum Strategy {
        BASELINE,
        CPU_OPTIMIZED,
        OPENMP,
        GPU
    };
};
```

---

## Common Mistakes to Avoid

1. **Modifying input grid**: Should be const
2. **Integer overflow**: In sum calculation
3. **Off-by-one errors**: In loop bounds
4. **Forgetting edge cases**: Empty, single element
5. **Inefficient column access**: Consider transpose for better cache performance

---

## Advanced Topics

### Cache Optimization

**Q: How does cache affect performance?**

**A:** 
- Row access: Sequential (good cache locality)
- Column access: Strided (poor cache locality)
- Solution: Transpose grid or use cache blocking

### NUMA Awareness

For very large grids on NUMA systems:
```cpp
#pragma omp parallel for num_threads(N) proc_bind(spread)
```

### GPU Memory Management

**Q: How to minimize GPU memory transfers?**

**A:**
- Keep data on GPU if multiple operations
- Use pinned memory for faster transfers
- Asynchronous transfers with computation overlap

---

## Interview Tips

1. **Start simple**: Write baseline first, then optimize
2. **Explain trade-offs**: Time vs space, complexity vs readability
3. **Ask clarifying questions**: Grid size? Constraints? Performance requirements?
4. **Test incrementally**: Verify each step
5. **Discuss alternatives**: Show you considered multiple approaches
