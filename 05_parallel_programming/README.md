# Parallel Programming

Modern C++ parallel programming techniques and patterns.

## Topics Covered

### 1. Thread Basics
- **threads_basics.cpp** - Thread creation, synchronization, mutex
- `std::thread`, `std::mutex`, `std::lock_guard`
- Race conditions and thread safety
- Join vs detach

### 2. OpenMP
- **openmp_basics.cpp** - OpenMP directives for parallelism
- Parallel for loops, reductions, sections
- Critical sections and atomic operations
- Matrix operations parallelization

### 3. Async and Futures
- **async_futures.cpp** - `std::async`, `std::future`, `std::promise`
- Asynchronous task execution
- Shared futures for multiple waiters
- Exception handling in async tasks

## Compilation

### For OpenMP programs:
```bash
g++ -fopenmp openmp_basics.cpp -o openmp_basics
clang++ -fopenmp openmp_basics.cpp -o openmp_basics
```

### For thread programs:
```bash
g++ -std=c++17 -pthread threads_basics.cpp -o threads_basics
```

## Parallelization Patterns

| Pattern | Use Case | Tool |
|---------|----------|------|
| Data parallelism | Independent operations on array elements | OpenMP parallel for |
| Task parallelism | Independent tasks | std::async |
| Pipeline | Sequential stages with parallelism | std::future chains |
| Reduction | Combining results | OpenMP reduction |

## Synchronization Primitives

| Primitive | Purpose | Overhead |
|-----------|---------|----------|
| `std::mutex` | Mutual exclusion | Medium |
| `std::lock_guard` | RAII mutex wrapper | Medium |
| `std::atomic` | Lock-free operations | Low |
| `#pragma omp critical` | OpenMP critical section | Medium |
| `#pragma omp atomic` | OpenMP atomic op | Low |

## Best Practices

1. **Minimize shared state** - Reduce contention
2. **Use lock-free when possible** - `std::atomic` for simple operations
3. **RAII for locks** - `std::lock_guard`, `std::unique_lock`
4. **Match thread count to hardware** - `std::thread::hardware_concurrency()`
5. **Avoid false sharing** - Align data to cache lines
6. **Profile before optimizing** - Measure speedup

## Common Pitfalls

- Race conditions from unsynchronized access
- Deadlocks from circular lock dependencies
- False sharing degrading performance
- Too many threads (oversubscription)
- Not joining threads (resource leaks)

## Interview Questions

- Explain race conditions and how to prevent them
- When to use `std::async` vs `std::thread`?
- What is false sharing?
- How does OpenMP parallelize loops?
- Explain deadlock and how to avoid it
- What is the difference between `std::future` and `std::shared_future`?
