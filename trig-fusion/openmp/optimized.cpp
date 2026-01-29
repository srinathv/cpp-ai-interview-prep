/**
 * Optimized OpenMP Trig Fusion - Single Loop with sincos
 * 
 * Optimizations:
 * 1. Single loop for all operations (cache friendly)
 * 2. Uses sincos() to compute both at once (GNU extension)
 * 3. SIMD hints for vectorization
 * 
 * sincos() Explanation:
 * - Computes sin and cos simultaneously
 * - Shares intermediate calculations
 * - Available via GNU C library or compiler intrinsics
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <omp.h>

// Use sincosf if available (GNU extension)
#ifdef __GLIBC__
extern "C" void sincosf(float x, float* sin_val, float* cos_val);
#else
inline void sincosf(float x, float* sin_val, float* cos_val) {
    *sin_val = std::sin(x);
    *cos_val = std::cos(x);
}
#endif

void trig_optimized(const float* input, float* cos_out, float* sin_out,
                    float& sum, int n) {
    sum = 0.0f;
    
    // Single fused loop
    #pragma omp parallel for reduction(+:sum)
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
    std::vector<float> input(N);
    std::vector<float> cos_out(N);
    std::vector<float> sin_out(N);
    float sum;
    
    srand(42);
    for (int i = 0; i < N; i++) {
        input[i] = static_cast<float>(rand() % 628) / 100.0f;
    }
    
    // Warmup
    trig_optimized(input.data(), cos_out.data(), sin_out.data(), sum, N);
    
    double start = omp_get_wtime();
    for (int iter = 0; iter < 10000; iter++) {
        trig_optimized(input.data(), cos_out.data(), sin_out.data(), sum, N);
    }
    double end = omp_get_wtime();
    
    float identity_error = 0.0f;
    for (int i = 0; i < N; i++) {
        float val = sin_out[i] * sin_out[i] + cos_out[i] * cos_out[i];
        identity_error += std::abs(val - 1.0f);
    }
    
    std::cout << "=== Optimized OpenMP Trig (Fused Loop) ===\n";
    std::cout << "Optimizations: sincosf(), single loop, reduction\n\n";
    std::cout << "First 5 cos(x): ";
    for (int i = 0; i < 5; i++) std::cout << cos_out[i] << " ";
    std::cout << "\nFirst 5 sin(x): ";
    for (int i = 0; i < 5; i++) std::cout << sin_out[i] << " ";
    std::cout << "\nSum: " << sum << "\n";
    std::cout << "Identity error: " << identity_error / N << "\n";
    std::cout << "Time per call: " << (end - start) / 10000 * 1e6 << " us\n";
    
    return 0;
}
