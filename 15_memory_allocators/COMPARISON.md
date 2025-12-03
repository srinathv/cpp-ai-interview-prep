# Memory Allocator Detailed Comparison

This document provides an in-depth technical comparison of modern memory allocators.

## Architecture Overview

### glibc malloc (ptmalloc2)

**Design Philosophy:** General-purpose, stable, compatible

**Key Features:**
- Arena-based allocation (one main arena + per-thread arenas)
- Bins for different size classes
- Fast bins for small, frequently used sizes
- Coalescing of adjacent free chunks

**Internal Structure:**
```
Main Arena
├── Fast bins (16-80 bytes, LIFO)
├── Small bins (< 512 bytes, FIFO)
├── Large bins (>= 512 bytes, sorted)
└── Unsorted bin (temporary staging)

Per-thread Arenas (created on contention)
├── Similar bin structure
└── Reduces lock contention
```

**Allocation Strategy:**
- Small allocations (<= 64 bytes): Fast bins
- Medium allocations: Small bins
- Large allocations: Large bins or mmap
- Very large (>= 128KB): Direct mmap

**Performance Characteristics:**
- Single-threaded: Good
- Multi-threaded (< 8 cores): Moderate
- Multi-threaded (>= 8 cores): Poor (lock contention)
- Fragmentation: Moderate

### tcmalloc (Thread-Caching Malloc)

**Design Philosophy:** Optimize for multi-threaded performance

**Key Features:**
- Per-thread cache (no locks for most operations)
- Size classes (90+ classes from 8 to 256KB)
- Central free list for cache refills
- Page heap for large allocations

**Internal Structure:**
```
Thread Cache (per thread, no locks)
├── Free lists per size class
└── Max cache size configurable

Central Cache (shared, locks required)
├── Transfer batches to thread caches
└── Organized by size classes

Page Heap
├── Manages spans of pages
└── Uses radix tree for fast lookup
```

**Allocation Strategy:**
- Small (<= 256KB): Thread cache → Central cache → Page heap
- Large (> 256KB): Direct from page heap
- No locks in common case (thread cache hit)

**Performance Characteristics:**
- Single-threaded: Excellent
- Multi-threaded: Excellent (near-linear scaling)
- Memory overhead: Higher (per-thread caches)
- Fragmentation: Good
- Startup memory: ~13MB

### jemalloc

**Design Philosophy:** Minimize fragmentation, optimize for multi-core

**Key Features:**
- Multiple arenas (4x CPU cores by default)
- Size classes with 25% spacing
- Thread-local caching
- Chunk-based allocation (2MB or 4MB chunks)
- Extensive statistics and profiling

**Internal Structure:**
```
Arenas (multiple, reduces contention)
├── Bins for small allocations
│   ├── Size classes: 8, 16, 32, ..., 14KB
│   └── Thread cache per size class
├── Runs for medium allocations (large pages)
└── Chunks for backing storage

Thread Cache
├── Small allocation cache
└── Periodically flushes to arena
```

**Allocation Strategy:**
- Tiny (<= 8 bytes): Specialized handling
- Small (<= 14KB): Thread cache → Arena bins
- Large (> 14KB, < 4MB): Arena large allocation
- Huge (>= 4MB): Direct chunk allocation

**Performance Characteristics:**
- Single-threaded: Very Good
- Multi-threaded: Excellent
- Memory overhead: Low
- Fragmentation: Excellent (best in class)
- Startup memory: ~9MB

### mimalloc

**Design Philosophy:** Security + Performance

**Key Features:**
- Per-thread heaps (sharded heaps)
- Free list sharding
- Randomization and security features
- Delayed freeing
- Fast path optimization

**Internal Structure:**
```
Thread Heaps (one per thread)
├── Small object pages (<= 1024 bytes)
│   ├── Free list per size class
│   └── Bitmaps for allocation tracking
├── Medium pages (1KB - 512KB)
└── Large allocations (> 512KB, direct mapping)

Segment Cache
├── Reuses memory segments
└── Reduces system calls
```

**Allocation Strategy:**
- Small (<= 1KB): Thread heap small pages
- Medium (1KB - 512KB): Thread heap medium pages
- Large (> 512KB): Direct OS allocation
- Free list sharding reduces false sharing

**Performance Characteristics:**
- Single-threaded: Excellent
- Multi-threaded: Excellent
- Memory overhead: Low
- Fragmentation: Excellent
- Startup memory: ~4MB
- Security: Best (optional secure mode)

## Detailed Performance Comparison

### Allocation Speed

**Small allocations (64 bytes, 1M operations):**
```
mimalloc:     ~50-60 ns/op   (fastest)
tcmalloc:     ~60-70 ns/op
jemalloc:     ~70-80 ns/op
glibc malloc: ~80-100 ns/op
```

**Medium allocations (4KB, 100K operations):**
```
tcmalloc:     ~200-250 ns/op (fastest)
mimalloc:     ~220-270 ns/op
jemalloc:     ~250-300 ns/op
glibc malloc: ~300-400 ns/op
```

**Large allocations (1MB, 10K operations):**
```
tcmalloc:     ~5-8 µs/op
jemalloc:     ~6-9 µs/op
mimalloc:     ~7-10 µs/op
glibc malloc: ~10-15 µs/op
```

### Multi-threaded Scaling (8 cores, small allocations)

```
Speedup vs single-threaded:

tcmalloc:     7.5x (94% efficiency)
jemalloc:     7.2x (90% efficiency)
mimalloc:     7.0x (88% efficiency)
glibc malloc: 4.5x (56% efficiency)
```

### Memory Overhead

**Per-allocation overhead:**
```
glibc malloc: 8-16 bytes
jemalloc:     8-16 bytes
tcmalloc:     8-16 bytes
mimalloc:     0-8 bytes (optimized)
```

**Metadata overhead (for 1GB allocated):**
```
glibc malloc: ~20-30 MB
tcmalloc:     ~25-35 MB
jemalloc:     ~15-25 MB (best)
mimalloc:     ~18-28 MB
```

### Fragmentation Resistance

**After 1M random alloc/free operations:**
```
jemalloc:     1.05x memory usage (best)
mimalloc:     1.08x
tcmalloc:     1.12x
glibc malloc: 1.25x (worst)
```

## Platform Differences

### Linux
- **Default**: glibc malloc (ptmalloc2)
- **Easy integration**: LD_PRELOAD or link-time
- **Best performance**: tcmalloc or jemalloc for multi-threaded

### macOS
- **Default**: System malloc (based on magazine malloc)
- **Integration**: DYLD_INSERT_LIBRARIES or link-time
- **Good default**: macOS malloc is already optimized

### Windows
- **Default**: Windows Heap API (NT Heap)
- **Integration**: Link-time replacement or source modification
- **Alternatives**: mimalloc works well on Windows

### FreeBSD
- **Default**: jemalloc
- **Already optimized**: No need to change

## Use Case Recommendations

### High-Concurrency Web Servers
**Recommendation: tcmalloc or jemalloc**

Reasons:
- Excellent multi-threaded scaling
- Low lock contention
- Proven in production (Google, Facebook)

Example: Nginx with jemalloc can handle 20-30% more requests

### Long-Running Daemons/Services
**Recommendation: jemalloc or mimalloc**

Reasons:
- Superior fragmentation resistance
- Memory usage stays stable over time
- Used by Redis, PostgreSQL, systemd

### Real-Time Applications
**Recommendation: Custom pool or mimalloc**

Reasons:
- Predictable latency
- Fast path optimization
- Minimal jitter

### Security-Critical Applications
**Recommendation: mimalloc (secure mode)**

Features:
- Guard pages detect overflow
- Randomized allocation
- Encrypted free lists
- Use-after-free detection

Cost: ~10% performance overhead

### Database Systems
**Recommendation: jemalloc**

Reasons:
- Used by Redis, MariaDB, MongoDB
- Excellent for mixed workloads
- Low fragmentation with varied sizes

### Memory-Constrained Embedded
**Recommendation: glibc malloc or custom**

Reasons:
- Lower startup memory
- Smaller code size
- Adequate for single-threaded

## Configuration and Tuning

### tcmalloc Environment Variables

```bash
# Set max thread cache size (default 2MB)
TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES=4194304

# Aggressive memory return to OS
TCMALLOC_RELEASE_RATE=10

# Sample memory allocations for profiling
TCMALLOC_SAMPLE_PARAMETER=524288
```

### jemalloc Environment Variables

```bash
# Number of arenas (default: 4x cores)
MALLOC_CONF="narenas:16"

# Dirty page purging
MALLOC_CONF="dirty_decay_ms:5000,muzzy_decay_ms:10000"

# Enable profiling
MALLOC_CONF="prof:true,prof_prefix:jeprof.out"
```

### mimalloc Environment Variables

```bash
# Show statistics on exit
MIMALLOC_SHOW_STATS=1

# Verbose output
MIMALLOC_VERBOSE=1

# Page reset time
MIMALLOC_RESET_DELAY=100

# Secure mode
MIMALLOC_SECURE=1
```

## Migration Checklist

When switching allocators:

1. **Benchmark first**
   - Measure current performance
   - Identify bottlenecks
   - Set performance goals

2. **Test with LD_PRELOAD** (Linux)
   ```bash
   LD_PRELOAD=/usr/lib/libtcmalloc.so ./app
   ```

3. **Monitor metrics**
   - RSS memory usage
   - Allocation rate
   - Thread scaling
   - Latency percentiles

4. **Test in staging**
   - Run for extended period
   - Check for memory leaks
   - Verify correctness

5. **Deploy gradually**
   - Canary deployment
   - Monitor production metrics
   - Have rollback plan

## Common Pitfalls

### 1. Using tcmalloc with fork()
- Issue: Child processes inherit full thread cache
- Solution: Set smaller TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES

### 2. jemalloc with many threads
- Issue: Too many arenas consume memory
- Solution: Reduce narenas

### 3. Mixing allocators
- Issue: Allocate with one, free with another = crash
- Solution: Use one allocator consistently

### 4. Not tuning for workload
- Issue: Default settings may not be optimal
- Solution: Profile and tune environment variables

## Debugging Tools

### tcmalloc
```bash
# Heap profiler
HEAPPROFILE=/tmp/profile ./app
google-pprof --text ./app /tmp/profile.0001.heap

# CPU profiler
CPUPROFILE=/tmp/prof.out ./app
google-pprof --text ./app /tmp/prof.out
```

### jemalloc
```bash
# Enable profiling
MALLOC_CONF=prof:true,prof_prefix:jeprof ./app

# Analyze with jeprof
jeprof --show_bytes ./app jeprof.*.heap
```

### mimalloc
```bash
# Show statistics
MIMALLOC_SHOW_STATS=1 ./app

# Verbose debugging
MIMALLOC_VERBOSE=2 ./app
```

## Summary

| When to use | Allocator |
|-------------|-----------|
| Multi-threaded (>8 cores) | tcmalloc or jemalloc |
| Long-running service | jemalloc or mimalloc |
| Security-critical | mimalloc (secure mode) |
| Database workload | jemalloc |
| Low fragmentation | jemalloc |
| Windows native | mimalloc |
| Default/simple | glibc malloc |
| Maximum performance | tcmalloc |

**Key Takeaway:** The default allocator is sufficient for most applications. Switch to an alternative when profiling shows allocation bottlenecks, multi-threaded contention, or fragmentation issues.
