/**
 * Matrix Multiplication in CUDA
 * 
 * Basic GPU matrix multiply: C = A * B
 * Shows 2D thread indexing and memory access patterns
 * 
 * Key concepts:
 * - 2D grid and block dimensions
 * - Row-major matrix storage
 * - Memory coalescing importance
 * - Shared memory optimization (see advanced examples)
 */

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

// Simple matrix multiply kernel (naive implementation)
__global__ void matrixMulNaive(const float *A, const float *B, float *C,
                                int M, int N, int K) {
    // Calculate row and column for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

// CPU version for verification
void matrixMulCPU(const float *A, const float *B, float *C,
                  int M, int N, int K) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < K; col++) {
            float sum = 0.0f;
            for (int i = 0; i < N; i++) {
                sum += A[row * N + i] * B[i * K + col];
            }
            C[row * K + col] = sum;
        }
    }
}

int main() {
    std::cout << "=== CUDA Matrix Multiplication ===\n\n";
    
    // Matrix dimensions: C(M x K) = A(M x N) * B(N x K)
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    
    size_t bytes_A = M * N * sizeof(float);
    size_t bytes_B = N * K * sizeof(float);
    size_t bytes_C = M * K * sizeof(float);
    
    // Allocate host memory
    float *h_A = new float[M * N];
    float *h_B = new float[N * K];
    float *h_C = new float[M * K];
    float *h_C_cpu = new float[M * K];
    
    // Initialize matrices
    for (int i = 0; i < M * N; i++) h_A[i] = float(rand()) / RAND_MAX;
    for (int i = 0; i < N * K; i++) h_B[i] = float(rand()) / RAND_MAX;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    
    // Launch configuration
    dim3 blockDim(16, 16);  // 16x16 = 256 threads per block
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);
    
    std::cout << "Matrix dimensions:\n";
    std::cout << "  A: " << M << " x " << N << "\n";
    std::cout << "  B: " << N << " x " << K << "\n";
    std::cout << "  C: " << M << " x " << K << "\n\n";
    
    std::cout << "Launch configuration:\n";
    std::cout << "  Block: " << blockDim.x << " x " << blockDim.y << "\n";
    std::cout << "  Grid: " << gridDim.x << " x " << gridDim.y << "\n\n";
    
    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // GPU execution
    CUDA_CHECK(cudaEventRecord(start));
    matrixMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    
    // CPU execution (small subset for timing)
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matrixMulCPU(h_A, h_B, h_C_cpu, M, N, K);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    
    // Verify
    bool correct = true;
    float max_error = 0.0f;
    for (int i = 0; i < M * K && correct; i++) {
        float error = std::abs(h_C[i] - h_C_cpu[i]);
        max_error = std::max(max_error, error);
        if (error > 1e-2) {
            correct = false;
        }
    }
    
    // Performance metrics
    double gflops = (2.0 * M * N * K) / (gpu_time * 1e6);  // Giga FLOPS
    
    std::cout << "Results:\n";
    std::cout << "  GPU time: " << gpu_time << " ms\n";
    std::cout << "  CPU time: " << cpu_time << " ms\n";
    std::cout << "  Speedup: " << cpu_time / gpu_time << "x\n";
    std::cout << "  GPU Performance: " << gflops << " GFLOPS\n";
    std::cout << "  Verification: " << (correct ? "PASSED" : "FAILED") << "\n";
    std::cout << "  Max error: " << max_error << "\n\n";
    
    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_cpu;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    std::cout << "Key interview points:\n";
    std::cout << "1. 2D indexing: row = blockIdx.y * blockDim.y + threadIdx.y\n";
    std::cout << "2. Row-major storage: A[row][col] = A[row * N + col]\n";
    std::cout << "3. This is naive - can be optimized with shared memory\n";
    std::cout << "4. Memory access pattern affects performance\n";
    std::cout << "5. Standard block size: 16x16 or 32x32\n";
    
    return 0;
}
