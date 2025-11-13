#include <iostream>
#include <hip/hip_runtime.h>
#include "reduction_utils.h"

__global__ void reduceNaive(const float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) atomicAdd(output, input[idx]);
}

int main() {
    std::cout << "=== Naive Reduction (HIP) ===\n";
    const int N = 1<<24;
    float *h_data = new float[N]; initData(h_data, N);
    float *d_data, *d_result;
    HIP_CHECK(hipMalloc(&d_data, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_result, sizeof(float)));
    HIP_CHECK(hipMemcpy(d_data, h_data, N * sizeof(float), hipMemcpyHostToDevice));
    int threads = 256, blocks = (N + threads - 1) / threads;
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipMemset(d_result, 0, sizeof(float)));
    HIP_CHECK(hipEventRecord(start));
    hipLaunchKernelGGL(reduceNaive, dim3(blocks), dim3(threads), 0, 0, d_data, d_result, N);
    HIP_CHECK(hipEventRecord(stop)); HIP_CHECK(hipEventSynchronize(stop));
    float ms; HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    float result; HIP_CHECK(hipMemcpy(&result, d_result, sizeof(float), hipMemcpyDeviceToHost));
    std::cout << "Time: " << ms << " ms, Bandwidth: " << calculateBandwidth(N, ms) << " GB/s\n";
    delete[] h_data; HIP_CHECK(hipFree(d_data)); HIP_CHECK(hipFree(d_result));
    return 0;
}
