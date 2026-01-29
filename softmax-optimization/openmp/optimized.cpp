/**
 * Optimized OpenMP Softmax - Numerically Stable
 * 
 * Key Optimization: Subtract max(x) before exp to prevent overflow
 * 
 * Mathematical Derivation:
 * ========================
 * softmax(x)_i = exp(x_i - max) / sum(exp(x_j - max))
 * 
 * This works because softmax is shift-invariant:
 * exp(x_i - c) / sum(exp(x_j - c)) 
 *   = exp(x_i)exp(-c) / [exp(-c) * sum(exp(x_j))]
 *   = exp(x_i) / sum(exp(x_j))
 * 
 * By choosing c = max(x), all exponents become <= 0
 * So exp values are in (0, 1], preventing overflow
 * 
 * Uses 3 passes: max, sum, normalize
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <omp.h>

void softmax_stable(const float* input, float* output, int n) {
    // Pass 1: Find maximum
    float max_val = -std::numeric_limits<float>::infinity();
    
    #pragma omp parallel for reduction(max:max_val)
    for (int i = 0; i < n; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    // Pass 2: Compute exp(x - max) and sum
    float sum = 0.0f;
    
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        output[i] = std::exp(input[i] - max_val);  // STABLE: exponent <= 0
        sum += output[i];
    }
    
    // Pass 3: Normalize
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        output[i] /= sum;
    }
}

int main() {
    const int N = 1024;
    std::vector<float> input(N);
    std::vector<float> output(N);
    
    // Large values - stable version handles this!
    srand(42);
    for (int i = 0; i < N; i++) {
        input[i] = static_cast<float>(rand() % 1000);  // Range [0, 1000]
    }
    
    double start = omp_get_wtime();
    for (int iter = 0; iter < 10000; iter++) {
        softmax_stable(input.data(), output.data(), N);
    }
    double end = omp_get_wtime();
    
    float sum = 0.0f;
    for (int i = 0; i < N; i++) sum += output[i];
    
    std::cout << "Stable OpenMP Softmax\n";
    std::cout << "Input range: [0, 1000] - would overflow naive version!\n";
    std::cout << "First 5 outputs: ";
    for (int i = 0; i < 5; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << "\nSum: " << sum << " (should be 1.0)\n";
    std::cout << "Time per call: " << (end - start) / 10000 * 1e6 << " us\n";
    
    return 0;
}
