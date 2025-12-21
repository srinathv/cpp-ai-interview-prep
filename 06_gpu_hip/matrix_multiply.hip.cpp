#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

// Simple matrix multiplication kernel
__global__ void matmulSimple(const float* A, const float* B, float* C, 
                              int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled matrix multiplication with shared memory
#define TILE_SIZE 16

__global__ void matmulTiled(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        
        if ((t * TILE_SIZE + threadIdx.y) < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        // Compute partial product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

#define HIP_CHECK(call) \
    { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    }

int main() {
    const int M = 1024;  // A: M x K
    const int K = 512;   // B: K x N
    const int N = 1024;  // C: M x N
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    // Allocate and initialize host memory
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, size_A));
    HIP_CHECK(hipMalloc(&d_B, size_B));
    HIP_CHECK(hipMalloc(&d_C, size_C));
    
    // Copy to device
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));
    
    // Launch tiled kernel
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    HIP_CHECK(hipEventRecord(start));
    hipLaunchKernelGGL(matmulTiled, gridDim, blockDim, 0, 0, d_A, d_B, d_C, M, N, K);
    HIP_CHECK(hipEventRecord(stop));
    
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    
    float milliseconds = 0;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
    
    // Copy result back
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, size_C, hipMemcpyDeviceToHost));
    
    std::cout << "Matrix multiplication complete!" << std::endl;
    std::cout << "Dimensions: " << M << " x " << K << " x " << N << std::endl;
    std::cout << "Time: " << milliseconds << " ms" << std::endl;
    
    double gflops = (2.0 * M * N * K) / (milliseconds * 1e6);
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    
    // Clean up
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    
    return 0;
}
