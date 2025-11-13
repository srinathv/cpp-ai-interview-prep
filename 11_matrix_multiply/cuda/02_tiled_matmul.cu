/**
 * Tiled Matrix Multiplication with Shared Memory - CUDA
 * 
 * Key optimization: Load tiles into shared memory, reuse across threads
 * Arithmetic intensity: O(TILE_SIZE) vs O(1/N) for naive
 * 
 * Expected: ~500-1000 GFLOPS (10x improvement!)
 */

#include <iostream>
#include <cuda_runtime.h>
#include "matmul_utils.h"

#define TILE_SIZE 32

__global__ void matmulTiled(const float *A, const float *B, float *C,
                            int M, int N, int K) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile from A
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from B
        int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row < N && col < K) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * K + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial product for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

int main() {
    std::cout << "=== Tiled Matrix Multiplication (CUDA) ===\n\n";
    
    const int M = 2048, N = 2048, K = 2048;
    
    size_t bytes_A = M * N * sizeof(float);
    size_t bytes_B = N * K * sizeof(float);
    size_t bytes_C = M * K * sizeof(float);
    
    std::cout << "Matrix dimensions: " << M << " × " << N << " × " << K << "\n";
    std::cout << "Tile size: " << TILE_SIZE << " × " << TILE_SIZE << "\n\n";
    
    float *h_A = new float[M * N];
    float *h_B = new float[N * K];
    float *h_C = new float[M * K];
    
    initMatrix(h_A, M * N);
    initMatrix(h_B, N * K);
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((K + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // Warm up
    matmulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int iterations = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        matmulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
    time_ms /= iterations;
    
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    
    double gflops = calculateGFLOPS(M, N, K, time_ms);
    
    std::cout << "Results:\n";
    std::cout << "  Time: " << time_ms << " ms\n";
    std::cout << "  Performance: " << gflops << " GFLOPS\n\n";
    
    std::cout << "Optimization Summary:\n";
    std::cout << "  ✓ Load tiles into shared memory\n";
    std::cout << "  ✓ Each element reused " << TILE_SIZE << " times\n";
    std::cout << "  ✓ Arithmetic intensity: O(" << TILE_SIZE << ")\n";
    std::cout << "  → Result: ~10x faster than naive!\n";
    
    delete[] h_A; delete[] h_B; delete[] h_C;
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}
