/**
 * Naive OpenMP Softmax Implementation
 * 
 * Problems:
 * 1. No numerical stability - exp(large_value) overflows
 * 2. No vectorization hints
 * 3. Multiple passes over data (cache unfriendly)
 * 
 * Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j))
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <omp.h>

void softmax_naive(const float* input, float* output, int n) {
    float sum = 0.0f;
    
    // Pass 1: Compute exp and sum (DANGER: can overflow!)
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        output[i] = std::exp(input[i]);  // Can overflow for large input!
        sum += output[i];
    }
    
    // Pass 2: Normalize
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        output[i] /= sum;
    }
}

int main() {
    const int N = 1024;
    std::vector<float> input(N);
    std::vector<float> output(N);
    
    // Small values to avoid overflow
    srand(42);
    for (int i = 0; i < N; i++) {
        input[i] = static_cast<float>(rand() % 10) - 5.0f;
    }
    
    double start = omp_get_wtime();
    for (int iter = 0; iter < 10000; iter++) {
        softmax_naive(input.data(), output.data(), N);
    }
    double end = omp_get_wtime();
    
    // Verify sum = 1
    float sum = 0.0f;
    for (int i = 0; i < N; i++) sum += output[i];
    
    std::cout << "Naive OpenMP Softmax\n";
    std::cout << "First 5 outputs: ";
    for (int i = 0; i < 5; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << "\nSum: " << sum << " (should be 1.0)\n";
    std::cout << "Time per call: " << (end - start) / 10000 * 1e6 << " us\n";
    
    // Demonstrate overflow problem
    std::cout << "\n=== Overflow Problem Demo ===\n";
    std::cout << "exp(100) = " << std::exp(100.0f) << "\n";
    std::cout << "exp(500) = " << std::exp(500.0f) << " (inf!)\n";
    
    return 0;
}
