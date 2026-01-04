/**
 * Naive Matrix Transpose - HIP/ROCm
 * 
 * This is the HIP version of the naive CUDA transpose.
 * Nearly identical to CUDA, just using HIP API calls.
 * 
 * Runs on both AMD and NVIDIA GPUs!
 */

#include <iostream>
#include <hip/hip_runtime.h>
#include "transpose_utils.h"

// Naive transpose kernel (identical to CUDA version!)
__global__ void transposeNaive(const float *input, float *output, 
                               int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        output[x * height + y] = input[y * width + x];
    }
}

int main() {
    std::cout << "=== Naive Matrix Transpose (HIP/ROCm) ===\n\n";
    
    const int WIDTH = 4096;
    const int HEIGHT = 4096;
    const size_t bytes = WIDTH * HEIGHT * sizeof(float);
    
    std::cout << "Matrix size: " << WIDTH << " x " << HEIGHT << "\n";
    std::cout << "Memory: " << bytes / (1024*1024) << " MB\n\n";
    
    // Allocate host memory
    float *h_input = new float[WIDTH * HEIGHT];
    float *h_output = new float[WIDTH * HEIGHT];
    float *h_reference = new float[WIDTH * HEIGHT];
    
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_input[i] = static_cast<float>(i);
    }
    
    // Allocate device memory (hipMalloc instead of cudaMalloc)
    float *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input, bytes));
    HIP_CHECK(hipMalloc(&d_output, bytes));
    
    // Copy to device (hipMemcpy instead of cudaMemcpy)
    HIP_CHECK(hipMemcpy(d_input, h_input, bytes, hipMemcpyHostToDevice));
    
    // Launch configuration (same as CUDA)
    dim3 blockDim(32, 32);
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x,
                 (HEIGHT + blockDim.y - 1) / blockDim.y);
    
    std::cout << "Launch config:\n";
    std::cout << "  Block: " << blockDim.x << " x " << blockDim.y << "\n";
    std::cout << "  Grid: " << gridDim.x << " x " << gridDim.y << "\n\n";
    
    // Warm up
    hipLaunchKernelGGL(transposeNaive, gridDim, blockDim, 0, 0,
                       d_input, d_output, WIDTH, HEIGHT);
    HIP_CHECK(hipDeviceSynchronize());
    
    // Timing (hipEvent instead of cudaEvent)
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    const int num_iterations = 100;
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        hipLaunchKernelGGL(transposeNaive, gridDim, blockDim, 0, 0,
                           d_input, d_output, WIDTH, HEIGHT);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float time_ms;
    HIP_CHECK(hipEventElapsedTime(&time_ms, start, stop));
    time_ms /= num_iterations;
    
    // Copy result back
    HIP_CHECK(hipMemcpy(h_output, d_output, bytes, hipMemcpyDeviceToHost));
    
    // Verify
    transposeCPU(h_input, h_reference, WIDTH, HEIGHT);
    bool correct = verifyTranspose(h_output, h_reference, WIDTH * HEIGHT);
    
    // Performance
    double bandwidth = calculateBandwidth(WIDTH, HEIGHT, time_ms);
    
    std::cout << "Results:\n";
    std::cout << "  Time: " << time_ms << " ms\n";
    std::cout << "  Bandwidth: " << bandwidth << " GB/s\n";
    std::cout << "  Verification: " << (correct ? "PASSED ✓" : "FAILED ✗") << "\n\n";
    
    std::cout << "Note: This is HIP code - runs on AMD ROCm or NVIDIA CUDA!\n";
    std::cout << "Same algorithm, portable across vendors.\n";
    
    // Cleanup
    delete[] h_input;
    delete[] h_output;
    delete[] h_reference;
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    
    return 0;
}
