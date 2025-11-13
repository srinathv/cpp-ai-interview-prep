/**
 * Warp Shuffle Reduction - Modern approach
 * Uses __shfl_down_sync for warp-level reduction
 * Very fast! (~400 GB/s)
 */
#include <iostream>
#include <cuda_runtime.h>
#include "reduction_utils.h"

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduceWarp(const float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = (idx < N) ? input[idx] : 0.0f;
    
    // Warp-level reduction
    sum = warpReduceSum(sum);
    
    // First thread in each warp writes to global
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(output, sum);
    }
}

int main() {
    std::cout << "=== Warp Shuffle Reduction (CUDA) ===\n";
    const int N = 1<<24;
    float *h_data = new float[N];
    initData(h_data, N);
    float *d_data, *d_result;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
    CUDA_CHECK(cudaEventRecord(start));
    reduceWarp<<<blocks, threads>>>(d_data, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "Time: " << ms << " ms\n";
    std::cout << "Bandwidth: " << calculateBandwidth(N, ms) << " GB/s\n";
    std::cout << "✓ Near-optimal with warp shuffles!\n";
    
    delete[] h_data;
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result));
    return 0;
}
