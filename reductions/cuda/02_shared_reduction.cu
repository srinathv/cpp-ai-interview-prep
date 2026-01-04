/**
 * Shared Memory Tree Reduction
 * Much better! (~200 GB/s)
 */
#include <iostream>
#include <cuda_runtime.h>
#include "reduction_utils.h"

__global__ void reduceShared(const float *input, float *output, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load to shared memory
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Tree reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

int main() {
    std::cout << "=== Shared Memory Reduction (CUDA) ===\n";
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
    reduceShared<<<blocks, threads>>>(d_data, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "Time: " << ms << " ms\n";
    std::cout << "Bandwidth: " << calculateBandwidth(N, ms) << " GB/s\n";
    std::cout << "✓ Much better with shared memory!\n";
    
    delete[] h_data;
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result));
    return 0;
}
