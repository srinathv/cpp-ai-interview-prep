/**
 * Naive OpenMP Trig Fusion - Separate Loops
 * 
 * Problems:
 * 1. Multiple passes over data (cache unfriendly)
 * 2. Computes sin and cos separately
 * 3. No SIMD optimization hints
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <omp.h>

void trig_naive(const float* input, float* cos_out, float* sin_out,
                float& sum, int n) {
    // Pass 1: Compute cos
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        cos_out[i] = std::cos(input[i]);
    }
    
    // Pass 2: Compute sin
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        sin_out[i] = std::sin(input[i]);
    }
    
    // Pass 3: Compute sum
    sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += input[i];
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
    trig_naive(input.data(), cos_out.data(), sin_out.data(), sum, N);
    
    double start = omp_get_wtime();
    for (int iter = 0; iter < 10000; iter++) {
        trig_naive(input.data(), cos_out.data(), sin_out.data(), sum, N);
    }
    double end = omp_get_wtime();
    
    // Verify with identity
    float identity_error = 0.0f;
    for (int i = 0; i < N; i++) {
        float val = sin_out[i] * sin_out[i] + cos_out[i] * cos_out[i];
        identity_error += std::abs(val - 1.0f);
    }
    
    std::cout << "=== Naive OpenMP Trig (Separate Loops) ===\n";
    std::cout << "Problems: 3 separate loops, no sincosf\n\n";
    std::cout << "First 5 cos(x): ";
    for (int i = 0; i < 5; i++) std::cout << cos_out[i] << " ";
    std::cout << "\nFirst 5 sin(x): ";
    for (int i = 0; i < 5; i++) std::cout << sin_out[i] << " ";
    std::cout << "\nSum: " << sum << "\n";
    std::cout << "Identity error: " << identity_error / N << "\n";
    std::cout << "Time per call: " << (end - start) / 10000 * 1e6 << " us\n";
    
    return 0;
}
