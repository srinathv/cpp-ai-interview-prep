/**
 * Naive Matrix Multiplication - CUDA
 * C = A × B where A is M×N, B is N×K, C is M×K
 * 
 * Each thread computes one element of C
 * Problem: No data reuse - very memory bound!
 * 
 * Expected: ~50-100 GFLOPS (very poor for modern GPUs)
 */

#include <iostream>
#include <cuda_runtime.h>
#include "matmul_utils.h"

// Naive kernel: each thread computes one output element
__global__ void matmulNaive(const float *A, const float *B, float *C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < K) {
        float sum = 0.0f;
        // Each thread does N multiply-adds
        // Loads N elements from A and N elements from B
        // But each element is loaded by multiple threads (no reuse!)
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

int main() {
    std::cout << "=== Naive Matrix Multiplication (CUDA) ===\n\n";
    
    // Matrix dimensions (keep small for verification, but need large for timing)
    const int M = 1024, N = 1024, K = 1024;
    
    size_t bytes_A = M * N * sizeof(float);
    size_t bytes_B = N * K * sizeof(float);
    size_t bytes_C = M * K * sizeof(float);
    
    std::cout << "Matrix dimensions:\n";
    std::cout << "  A: " << M << " × " << N << "\n";
    std::cout << "  B: " << N << " × " << K << "\n";
    std::cout << "  C: " << M << " × " << K << "\n";
    std::cout << "  Total memory: " << (bytes_A + bytes_B + bytes_C) / (1024*1024) << " MB\n\n";
    
    // Allocate host memory
    float *h_A = new float[M * N];
    float *h_B = new float[N * K];
    float *h_C = new float[M * K];
    float *h_C_ref = new float[M * K];
    
    // Initialize with random data
    initMatrix(h_A, M * N);
    initMatrix(h_B, N * K);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    
    // Launch configuration
    dim3 blockDim(16, 16);  // 256 threads per block
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);
    
    std::cout << "Launch config:\n";
    std::cout << "  Block: " << blockDim.x << " × " << blockDim.y << "\n";
    std::cout << "  Grid: " << gridDim.x << " × " << gridDim.y << "\n\n";
    
    // Warm up
    matmulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int iterations = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        matmulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
    time_ms /= iterations;
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    
    // Verify (use small subset)
    std::cout << "Verifying result...\n";
    matmulCPU(h_A, h_B, h_C_ref, M, N, K);
    bool correct = verifyMatmul(h_C, h_C_ref, M * K);
    
    // Performance
    double gflops = calculateGFLOPS(M, N, K, time_ms);
    
    std::cout << "\nResults:\n";
    std::cout << "  Time: " << time_ms << " ms\n";
    std::cout << "  Performance: " << gflops << " GFLOPS\n";
    std::cout << "  Verification: " << (correct ? "PASSED ✓" : "FAILED ✗") << "\n\n";
    
    std::cout << "Problem Analysis:\n";
    std::cout << "  Total operations: " << 2L * M * N * K << " FLOPs\n";
    std::cout << "  Memory accesses: " << M * K * N << " reads from A\n";
    std::cout << "                   " << M * K * N << " reads from B\n";
    std::cout << "  ✗ Each element loaded multiple times (no reuse)\n";
    std::cout << "  ✗ Arithmetic intensity: O(1/N) - memory bound\n\n";
    
    std::cout << "Solution: Use shared memory tiling (see 02_tiled_matmul.cu)\n";
    
    // Cleanup
    delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_C_ref;
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}
