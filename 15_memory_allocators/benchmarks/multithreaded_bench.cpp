#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <atomic>

using namespace std;
using namespace std::chrono;

atomic<long long> total_ops{0};

void worker_thread(int thread_id, int iterations, size_t alloc_size) {
    vector<void*> local_ptrs;
    local_ptrs.reserve(iterations);
    
    // Each thread does its own allocations
    for (int i = 0; i < iterations; i++) {
        local_ptrs.push_back(malloc(alloc_size));
    }
    
    // Free them
    for (void* ptr : local_ptrs) {
        free(ptr);
    }
    
    total_ops += iterations;
}

void benchmark_threaded(int num_threads, int iterations_per_thread, size_t alloc_size, const string& desc) {
    total_ops = 0;
    
    auto start = high_resolution_clock::now();
    
    vector<thread> threads;
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(worker_thread, i, iterations_per_thread, alloc_size);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    auto end = high_resolution_clock::now();
    double ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    
    cout << desc << " [" << num_threads << " threads]:" << endl;
    cout << "  Total ops: " << total_ops.load() << endl;
    cout << "  Time: " << fixed << setprecision(3) << ms << " ms" << endl;
    cout << "  Throughput: " << (total_ops.load() / (ms / 1000.0)) << " ops/sec" << endl;
    cout << "  Per-thread: " << (iterations_per_thread / (ms / 1000.0)) << " ops/sec" << endl << endl;
}

void contention_test(int num_threads, int iterations) {
    cout << "=== Thread Contention Test ===" << endl;
    cout << "All threads allocating/freeing from shared pool\n" << endl;
    
    total_ops = 0;
    auto start = high_resolution_clock::now();
    
    vector<thread> threads;
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back([iterations]() {
            for (int j = 0; j < iterations; j++) {
                // Immediate alloc/free creates contention
                void* ptr = malloc(256);
                free(ptr);
                total_ops++;
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    auto end = high_resolution_clock::now();
    double ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    
    cout << "Contention test [" << num_threads << " threads]:" << endl;
    cout << "  Total ops: " << total_ops.load() << endl;
    cout << "  Time: " << fixed << setprecision(3) << ms << " ms" << endl;
    cout << "  Throughput: " << (total_ops.load() / (ms / 1000.0)) << " ops/sec" << endl << endl;
}

int main() {
    cout << "=== Multi-Threaded Allocator Benchmark ===" << endl;
    
    cout << "Allocator: ";
#ifdef USE_TCMALLOC
    cout << "tcmalloc" << endl;
#elif defined(USE_JEMALLOC)
    cout << "jemalloc" << endl;
#elif defined(USE_MIMALLOC)
    cout << "mimalloc" << endl;
#else
    cout << "system default" << endl;
#endif
    
    cout << "\nHardware threads: " << thread::hardware_concurrency() << endl << endl;
    
    // Test with different thread counts
    vector<int> thread_counts = {1, 2, 4, 8};
    int max_threads = thread::hardware_concurrency();
    
    // Filter to available threads
    thread_counts.erase(
        remove_if(thread_counts.begin(), thread_counts.end(),
                  [max_threads](int t) { return t > max_threads; }),
        thread_counts.end()
    );
    
    // Small allocations (64 bytes)
    cout << "=== Small Allocations (64 bytes) ===" << endl;
    for (int threads : thread_counts) {
        benchmark_threaded(threads, 100000, 64, "Small allocations");
    }
    
    // Medium allocations (4KB)
    cout << "=== Medium Allocations (4KB) ===" << endl;
    for (int threads : thread_counts) {
        benchmark_threaded(threads, 10000, 4096, "Medium allocations");
    }
    
    // Large allocations (1MB)
    cout << "=== Large Allocations (1MB) ===" << endl;
    for (int threads : thread_counts) {
        benchmark_threaded(threads, 1000, 1024*1024, "Large allocations");
    }
    
    // Contention test
    for (int threads : thread_counts) {
        contention_test(threads, 50000);
    }
    
    cout << "=== Benchmark Complete ===" << endl;
    cout << "\nKey Observations:" << endl;
    cout << "- tcmalloc/jemalloc should show near-linear scaling" << endl;
    cout << "- glibc malloc may show contention with many threads" << endl;
    cout << "- Per-thread caching reduces lock contention" << endl;
    
    return 0;
}
