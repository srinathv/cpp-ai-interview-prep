/**
 * Naive Reduction - Heavy atomic contention
 * Each thread atomically adds its element
 * Very slow! (~50 GB/s)
 */
#include <iostream>
#include <cuda_runtime.h>
#include "reduction_utils.h"

__global__ void reduceNaive(const float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(output, input[idx]);  // Heavy contention!
    }
}

int main() {
    std::cout << "=== Naive Reduction (CUDA) ===\n";
    const int N = 1<<24;  // 16M elements
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
    reduceNaive<<<blocks, threads>>>(d_data, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    float expected = reduceCPU(h_data, N);
    
    std::cout << "Result: " << result << " (expected: " << expected << ")\n";
    std::cout << "Time: " << ms << " ms\n";
    std::cout << "Bandwidth: " << calculateBandwidth(N, ms) << " GB/s\n";
    std::cout << "✗ Bottlenecked by atomic contention!\n";
    
    delete[] h_data;
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result));
    return 0;
}
