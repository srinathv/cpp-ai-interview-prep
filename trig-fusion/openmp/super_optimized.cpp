/**
 * Super Optimized OpenMP Trig Fusion
 * 
 * All optimizations:
 * 1. Cache blocking for L1/L2 efficiency
 * 2. SIMD vectorization with explicit hints
 * 3. Thread-local accumulation (reduces contention)
 * 4. Memory prefetching hints
 * 5. Aligned memory access
 * 
 * Cache Blocking Explanation:
 * ===========================
 * Process data in chunks that fit in L1 cache (typically 32KB)
 * This ensures data is loaded once and reused for all operations
 * 
 * For float (4 bytes), L1-friendly block size: ~4096 elements
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <omp.h>

// Aligned allocation for SIMD
template<typename T>
T* aligned_alloc_array(size_t n, size_t alignment = 64) {
    void* ptr = nullptr;
    posix_memalign(&ptr, alignment, n * sizeof(T));
    return static_cast<T*>(ptr);
}

#ifdef __GLIBC__
extern "C" void sincosf(float x, float* sin_val, float* cos_val);
#else
inline void sincosf(float x, float* sin_val, float* cos_val) {
    *sin_val = std::sin(x);
    *cos_val = std::cos(x);
}
#endif

void trig_super_optimized(const float* __restrict__ input,
                          float* __restrict__ cos_out,
                          float* __restrict__ sin_out,
                          float& sum, int n) {
    const int BLOCK_SIZE = 256;  // L1 cache friendly
    
    float total_sum = 0.0f;
    
    #pragma omp parallel reduction(+:total_sum)
    {
        float local_sum = 0.0f;
        
        #pragma omp for schedule(static) nowait
        for (int block_start = 0; block_start < n; block_start += BLOCK_SIZE) {
            int block_end = std::min(block_start + BLOCK_SIZE, n);
            
            // Prefetch next block (compiler hint)
            #ifdef __GNUC__
            if (block_start + BLOCK_SIZE < n) {
                __builtin_prefetch(&input[block_start + BLOCK_SIZE], 0, 3);
            }
            #endif
            
            // Process block with SIMD
            #pragma omp simd reduction(+:local_sum)
            for (int i = block_start; i < block_end; i++) {
                float x = input[i];
                
                float s, c;
                sincosf(x, &s, &c);
                
                sin_out[i] = s;
                cos_out[i] = c;
                local_sum += x;
            }
        }
        
        total_sum += local_sum;
    }
    
    sum = total_sum;
}

// Alternative: Explicit SIMD with intrinsics concept
// (shown for educational purposes - actual intrinsics would be architecture-specific)
void trig_manual_simd_concept(const float* input, float* cos_out, float* sin_out,
                               float& sum, int n) {
    /*
     * Conceptual SIMD approach (x86 AVX2 would use _mm256_* intrinsics):
     * 
     * 1. Load 8 floats at once: __m256 x = _mm256_load_ps(&input[i])
     * 2. Compute sin/cos using vector math library (e.g., Intel SVML)
     * 3. Store 8 results: _mm256_store_ps(&cos_out[i], cos_vec)
     * 4. Horizontal sum for reduction: _mm256_hadd_ps()
     * 
     * This achieves 8x throughput for trig operations!
     */
    
    // Fallback to scalar for portability
    sum = 0.0f;
    
    #pragma omp parallel for simd reduction(+:sum) aligned(input, cos_out, sin_out: 64)
    for (int i = 0; i < n; i++) {
        float x = input[i];
        float s, c;
        sincosf(x, &s, &c);
        sin_out[i] = s;
        cos_out[i] = c;
        sum += x;
    }
}

int main() {
    const int N = 1024;
    
    // Use aligned memory for optimal SIMD performance
    float* input = aligned_alloc_array<float>(N);
    float* cos_out = aligned_alloc_array<float>(N);
    float* sin_out = aligned_alloc_array<float>(N);
    float sum1, sum2;
    
    srand(42);
    for (int i = 0; i < N; i++) {
        input[i] = static_cast<float>(rand() % 628) / 100.0f;
    }
    
    // Warmup
    trig_super_optimized(input, cos_out, sin_out, sum1, N);
    trig_manual_simd_concept(input, cos_out, sin_out, sum2, N);
    
    // Benchmark blocked version
    double start1 = omp_get_wtime();
    for (int iter = 0; iter < 10000; iter++) {
        trig_super_optimized(input, cos_out, sin_out, sum1, N);
    }
    double end1 = omp_get_wtime();
    
    // Benchmark SIMD version
    double start2 = omp_get_wtime();
    for (int iter = 0; iter < 10000; iter++) {
        trig_manual_simd_concept(input, cos_out, sin_out, sum2, N);
    }
    double end2 = omp_get_wtime();
    
    float identity_error = 0.0f;
    for (int i = 0; i < N; i++) {
        float val = sin_out[i] * sin_out[i] + cos_out[i] * cos_out[i];
        identity_error += std::abs(val - 1.0f);
    }
    
    std::cout << "=== Super Optimized OpenMP Trig ===\n\n";
    
    std::cout << "Optimizations applied:\n";
    std::cout << "  1. Cache blocking (L1 friendly)\n";
    std::cout << "  2. SIMD vectorization hints\n";
    std::cout << "  3. Aligned memory (64-byte)\n";
    std::cout << "  4. Prefetch hints\n";
    std::cout << "  5. Thread-local reduction\n\n";
    
    std::cout << "Results:\n";
    std::cout << "First 5 cos(x): ";
    for (int i = 0; i < 5; i++) std::cout << cos_out[i] << " ";
    std::cout << "\nFirst 5 sin(x): ";
    for (int i = 0; i < 5; i++) std::cout << sin_out[i] << " ";
    std::cout << "\nSum: " << sum1 << "\n";
    std::cout << "Identity error: " << identity_error / N << "\n\n";
    
    std::cout << "Performance:\n";
    std::cout << "  Blocked version: " << (end1 - start1) / 10000 * 1e6 << " us\n";
    std::cout << "  SIMD version:    " << (end2 - start2) / 10000 * 1e6 << " us\n";
    
    free(input);
    free(cos_out);
    free(sin_out);
    
    return 0;
}
