/**
 * Super Optimized OpenMP Softmax
 * 
 * Key Optimizations:
 * 1. Online algorithm: Compute max and sum in a SINGLE pass
 * 2. SIMD vectorization hints
 * 3. Cache-friendly memory access patterns
 * 4. Custom reduction for online softmax
 * 
 * Online Softmax Algorithm:
 * =========================
 * Process elements incrementally, maintaining running max and sum:
 * 
 *   m_new = max(m_old, x_i)
 *   d_new = d_old * exp(m_old - m_new) + exp(x_i - m_new)
 * 
 * Parallel Merge:
 * When combining two partial results with different maxes:
 *   m_combined = max(m_a, m_b)
 *   d_combined = d_a * exp(m_a - m_combined) + d_b * exp(m_b - m_combined)
 * 
 * This reduces passes from 3 to 2!
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <omp.h>

// Structure to hold online softmax state
struct OnlineSoftmaxState {
    float max_val;
    float sum_exp;
    
    OnlineSoftmaxState() 
        : max_val(-std::numeric_limits<float>::infinity()), sum_exp(0.0f) {}
    
    OnlineSoftmaxState(float m, float s) : max_val(m), sum_exp(s) {}
    
    // Merge two states with different maxes
    void merge(const OnlineSoftmaxState& other) {
        // Handle empty states
        if (other.sum_exp == 0.0f) return;
        if (sum_exp == 0.0f) {
            max_val = other.max_val;
            sum_exp = other.sum_exp;
            return;
        }
        
        if (other.max_val > max_val) {
            // Other has larger max, scale our sum down
            sum_exp = sum_exp * std::exp(max_val - other.max_val) + other.sum_exp;
            max_val = other.max_val;
        } else {
            // We have larger or equal max, scale other's sum down
            sum_exp = sum_exp + other.sum_exp * std::exp(other.max_val - max_val);
        }
    }
    
    // Add a single element
    void add(float x) {
        if (sum_exp == 0.0f) {
            max_val = x;
            sum_exp = 1.0f;
        } else if (x > max_val) {
            sum_exp = sum_exp * std::exp(max_val - x) + 1.0f;
            max_val = x;
        } else {
            sum_exp += std::exp(x - max_val);
        }
    }
};

// OpenMP custom reduction declaration
#pragma omp declare reduction(online_merge : OnlineSoftmaxState : \
    omp_out.merge(omp_in)) \
    initializer(omp_priv = OnlineSoftmaxState())

void softmax_online(const float* input, float* output, int n) {
    OnlineSoftmaxState state;
    
    // Pass 1: Compute max and sum in ONE pass using online algorithm
    #pragma omp parallel for reduction(online_merge:state) schedule(static)
    for (int i = 0; i < n; i++) {
        state.add(input[i]);
    }
    
    const float max_val = state.max_val;
    const float sum_exp = state.sum_exp;
    
    // Pass 2: Compute final softmax values (vectorizable)
    #pragma omp parallel for simd schedule(static)
    for (int i = 0; i < n; i++) {
        output[i] = std::exp(input[i] - max_val) / sum_exp;
    }
}

// Even more optimized: blocked version for better cache utilization
void softmax_online_blocked(const float* input, float* output, int n) {
    const int BLOCK_SIZE = 256;  // L1 cache friendly
    
    OnlineSoftmaxState global_state;
    
    // Pass 1: Block-wise online reduction
    #pragma omp parallel
    {
        OnlineSoftmaxState local_state;
        
        #pragma omp for schedule(static) nowait
        for (int block_start = 0; block_start < n; block_start += BLOCK_SIZE) {
            int block_end = std::min(block_start + BLOCK_SIZE, n);
            
            // Process block
            for (int i = block_start; i < block_end; i++) {
                local_state.add(input[i]);
            }
        }
        
        // Merge into global state
        #pragma omp critical
        {
            global_state.merge(local_state);
        }
    }
    
    const float max_val = global_state.max_val;
    const float sum_exp = global_state.sum_exp;
    
    // Pass 2: Apply softmax (vectorized)
    #pragma omp parallel for simd schedule(static)
    for (int i = 0; i < n; i++) {
        output[i] = std::exp(input[i] - max_val) / sum_exp;
    }
}

int main() {
    const int N = 1024;
    std::vector<float> input(N);
    std::vector<float> output1(N);
    std::vector<float> output2(N);
    
    srand(42);
    for (int i = 0; i < N; i++) {
        input[i] = static_cast<float>(rand() % 1000);
    }
    
    // Warmup
    softmax_online(input.data(), output1.data(), N);
    softmax_online_blocked(input.data(), output2.data(), N);
    
    // Benchmark online version
    double start1 = omp_get_wtime();
    for (int iter = 0; iter < 10000; iter++) {
        softmax_online(input.data(), output1.data(), N);
    }
    double end1 = omp_get_wtime();
    
    // Benchmark blocked version
    double start2 = omp_get_wtime();
    for (int iter = 0; iter < 10000; iter++) {
        softmax_online_blocked(input.data(), output2.data(), N);
    }
    double end2 = omp_get_wtime();
    
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int i = 0; i < N; i++) {
        sum1 += output1[i];
        sum2 += output2[i];
    }
    
    std::cout << "=== Super Optimized OpenMP Softmax ===\n\n";
    
    std::cout << "Optimizations applied:\n";
    std::cout << "  1. Online algorithm: max+sum in single pass\n";
    std::cout << "  2. Custom OpenMP reduction for online merge\n";
    std::cout << "  3. SIMD vectorization hints\n";
    std::cout << "  4. Cache-blocking for L1 efficiency\n\n";
    
    std::cout << "Online Version:\n";
    std::cout << "  First 5: ";
    for (int i = 0; i < 5; i++) std::cout << output1[i] << " ";
    std::cout << "\n  Sum: " << sum1 << " (should be 1.0)\n";
    std::cout << "  Time: " << (end1 - start1) / 10000 * 1e6 << " us\n\n";
    
    std::cout << "Blocked Version:\n";
    std::cout << "  First 5: ";
    for (int i = 0; i < 5; i++) std::cout << output2[i] << " ";
    std::cout << "\n  Sum: " << sum2 << " (should be 1.0)\n";
    std::cout << "  Time: " << (end2 - start2) / 10000 * 1e6 << " us\n";
    
    return 0;
}
