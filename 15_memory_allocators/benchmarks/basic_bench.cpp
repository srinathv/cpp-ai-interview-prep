#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <iomanip>

using namespace std;
using namespace std::chrono;

// Benchmark basic allocation/deallocation patterns
class BasicBenchmark {
public:
    void run_all() {
        cout << "=== Basic Memory Allocator Benchmark ===" << endl;
        cout << "Testing basic allocation/deallocation patterns\n" << endl;
        
        benchmark_small_allocations();
        benchmark_medium_allocations();
        benchmark_large_allocations();
        benchmark_mixed_sizes();
        benchmark_alloc_free_pattern();
    }
    
private:
    void benchmark_small_allocations() {
        const int iterations = 1000000;
        const size_t size = 64; // Small allocations (64 bytes)
        
        auto start = high_resolution_clock::now();
        
        vector<void*> ptrs;
        ptrs.reserve(iterations);
        
        // Allocate
        for (int i = 0; i < iterations; i++) {
            ptrs.push_back(malloc(size));
        }
        
        // Free
        for (void* ptr : ptrs) {
            free(ptr);
        }
        
        auto end = high_resolution_clock::now();
        double ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        
        cout << "Small allocations (64 bytes):" << endl;
        cout << "  Iterations: " << iterations << endl;
        cout << "  Time: " << fixed << setprecision(3) << ms << " ms" << endl;
        cout << "  Ops/sec: " << (iterations / (ms / 1000.0)) << endl << endl;
    }
    
    void benchmark_medium_allocations() {
        const int iterations = 100000;
        const size_t size = 4096; // Medium allocations (4KB)
        
        auto start = high_resolution_clock::now();
        
        vector<void*> ptrs;
        ptrs.reserve(iterations);
        
        for (int i = 0; i < iterations; i++) {
            ptrs.push_back(malloc(size));
        }
        
        for (void* ptr : ptrs) {
            free(ptr);
        }
        
        auto end = high_resolution_clock::now();
        double ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        
        cout << "Medium allocations (4KB):" << endl;
        cout << "  Iterations: " << iterations << endl;
        cout << "  Time: " << fixed << setprecision(3) << ms << " ms" << endl;
        cout << "  Ops/sec: " << (iterations / (ms / 1000.0)) << endl << endl;
    }
    
    void benchmark_large_allocations() {
        const int iterations = 10000;
        const size_t size = 1024 * 1024; // Large allocations (1MB)
        
        auto start = high_resolution_clock::now();
        
        vector<void*> ptrs;
        ptrs.reserve(iterations);
        
        for (int i = 0; i < iterations; i++) {
            ptrs.push_back(malloc(size));
        }
        
        for (void* ptr : ptrs) {
            free(ptr);
        }
        
        auto end = high_resolution_clock::now();
        double ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        
        cout << "Large allocations (1MB):" << endl;
        cout << "  Iterations: " << iterations << endl;
        cout << "  Time: " << fixed << setprecision(3) << ms << " ms" << endl;
        cout << "  Ops/sec: " << (iterations / (ms / 1000.0)) << endl << endl;
    }
    
    void benchmark_mixed_sizes() {
        const int iterations = 100000;
        size_t sizes[] = {16, 64, 256, 1024, 4096, 16384};
        
        auto start = high_resolution_clock::now();
        
        vector<void*> ptrs;
        ptrs.reserve(iterations);
        
        for (int i = 0; i < iterations; i++) {
            size_t size = sizes[i % 6];
            ptrs.push_back(malloc(size));
        }
        
        for (void* ptr : ptrs) {
            free(ptr);
        }
        
        auto end = high_resolution_clock::now();
        double ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        
        cout << "Mixed size allocations (16B-16KB):" << endl;
        cout << "  Iterations: " << iterations << endl;
        cout << "  Time: " << fixed << setprecision(3) << ms << " ms" << endl;
        cout << "  Ops/sec: " << (iterations / (ms / 1000.0)) << endl << endl;
    }
    
    void benchmark_alloc_free_pattern() {
        const int iterations = 500000;
        const size_t size = 128;
        
        auto start = high_resolution_clock::now();
        
        // Interleaved alloc/free pattern (simulates real workloads)
        for (int i = 0; i < iterations; i++) {
            void* ptr1 = malloc(size);
            void* ptr2 = malloc(size);
            free(ptr1);
            void* ptr3 = malloc(size);
            free(ptr2);
            free(ptr3);
        }
        
        auto end = high_resolution_clock::now();
        double ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        
        cout << "Interleaved alloc/free pattern:" << endl;
        cout << "  Iterations: " << iterations << endl;
        cout << "  Time: " << fixed << setprecision(3) << ms << " ms" << endl;
        cout << "  Ops/sec: " << (iterations / (ms / 1000.0)) << endl << endl;
    }
};

int main() {
    // Print which allocator is being used
    cout << "Allocator: ";
    
#ifdef USE_TCMALLOC
    cout << "tcmalloc" << endl;
#elif defined(USE_JEMALLOC)
    cout << "jemalloc" << endl;
#elif defined(USE_MIMALLOC)
    cout << "mimalloc" << endl;
#else
    cout << "system default (glibc malloc on Linux, msvcrt on Windows)" << endl;
#endif
    
    cout << endl;
    
    BasicBenchmark bench;
    bench.run_all();
    
    cout << "=== Benchmark Complete ===" << endl;
    cout << "\nTo test with different allocators:" << endl;
    cout << "  Linux:   LD_PRELOAD=/usr/lib/libtcmalloc.so ./basic_bench" << endl;
    cout << "  macOS:   DYLD_INSERT_LIBRARIES=/usr/local/lib/libtcmalloc.dylib ./basic_bench" << endl;
    
    return 0;
}
