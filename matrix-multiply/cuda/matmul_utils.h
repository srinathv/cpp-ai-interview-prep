#ifndef MATMUL_UTILS_H
#define MATMUL_UTILS_H

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CPU matrix multiply for verification
void matmulCPU(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * K + j];
            }
            C[i * K + j] = sum;
        }
    }
}

// Verify results
bool verifyMatmul(const float *gpu_result, const float *cpu_result, int size) {
    const float epsilon = 1e-3f;  // Relaxed for accumulated floating point errors
    int errors = 0;
    for (int i = 0; i < size && errors < 10; i++) {
        float diff = fabs(gpu_result[i] - cpu_result[i]);
        if (diff > epsilon) {
            if (errors == 0) {
                std::cerr << "First mismatches:\n";
            }
            std::cerr << "  Index " << i << ": GPU=" << gpu_result[i] 
                      << " CPU=" << cpu_result[i] << " diff=" << diff << "\n";
            errors++;
        }
    }
    return errors == 0;
}

// Calculate GFLOPS
double calculateGFLOPS(int M, int N, int K, float time_ms) {
    double flops = 2.0 * M * N * K;  // 2x for multiply-add
    double time_s = time_ms / 1000.0;
    return (flops / time_s) / 1e9;
}

// Initialize matrix with random values
void initMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

#endif // MATMUL_UTILS_H
