#include <iostream>
#include <hip/hip_runtime.h>
#include "reduction_utils.h"

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down(val, offset);
    return val;
}

__global__ void reduceWarp(const float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = (idx < N) ? input[idx] : 0.0f;
    sum = warpReduceSum(sum);
    if ((threadIdx.x & 31) == 0) atomicAdd(output, sum);
}

int main() {
    std::cout << "=== Warp Shuffle Reduction (HIP) ===\n";
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
    hipLaunchKernelGGL(reduceWarp, dim3(blocks), dim3(threads), 0, 0, d_data, d_result, N);
    HIP_CHECK(hipEventRecord(stop)); HIP_CHECK(hipEventSynchronize(stop));
    float ms; HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    std::cout << "Time: " << ms << " ms, Bandwidth: " << calculateBandwidth(N, ms) << " GB/s\n";
    delete[] h_data; HIP_CHECK(hipFree(d_data)); HIP_CHECK(hipFree(d_result));
    return 0;
}
