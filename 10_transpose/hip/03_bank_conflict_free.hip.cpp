/**
 * Bank Conflict-Free Transpose - HIP/ROCm
 * Final optimized version with padding
 */

#include <iostream>
#include <hip/hip_runtime.h>
#include "transpose_utils.h"

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transposeBankConflictFree(const float *input, float *output,
                                          int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 padding!
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x < width && (y + i) < height) {
            tile[threadIdx.y + i][threadIdx.x] = input[(y + i) * width + x];
        }
    }
    
    __syncthreads();
    
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x < height && (y + i) < width) {
            output[(y + i) * height + x] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

int main() {
    std::cout << "=== Bank Conflict-Free Transpose (HIP) ===\n\n";
    
    const int WIDTH = 4096, HEIGHT = 4096;
    const size_t bytes = WIDTH * HEIGHT * sizeof(float);
    
    float *h_input = new float[WIDTH * HEIGHT];
    float *h_output = new float[WIDTH * HEIGHT];
    float *h_reference = new float[WIDTH * HEIGHT];
    
    for (int i = 0; i < WIDTH * HEIGHT; i++) h_input[i] = float(i);
    
    float *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input, bytes));
    HIP_CHECK(hipMalloc(&d_output, bytes));
    HIP_CHECK(hipMemcpy(d_input, h_input, bytes, hipMemcpyHostToDevice));
    
    dim3 blockDim(TILE_DIM, BLOCK_ROWS);
    dim3 gridDim((WIDTH + TILE_DIM - 1) / TILE_DIM,
                 (HEIGHT + TILE_DIM - 1) / TILE_DIM);
    
    hipLaunchKernelGGL(transposeBankConflictFree, gridDim, blockDim, 0, 0,
                       d_input, d_output, WIDTH, HEIGHT);
    HIP_CHECK(hipDeviceSynchronize());
    
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    const int iterations = 100;
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        hipLaunchKernelGGL(transposeBankConflictFree, gridDim, blockDim, 0, 0,
                           d_input, d_output, WIDTH, HEIGHT);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float time_ms;
    HIP_CHECK(hipEventElapsedTime(&time_ms, start, stop));
    time_ms /= iterations;
    
    HIP_CHECK(hipMemcpy(h_output, d_output, bytes, hipMemcpyDeviceToHost));
    
    transposeCPU(h_input, h_reference, WIDTH, HEIGHT);
    bool correct = verifyTranspose(h_output, h_reference, WIDTH * HEIGHT);
    double bandwidth = calculateBandwidth(WIDTH, HEIGHT, time_ms);
    
    std::cout << "Time: " << time_ms << " ms\n";
    std::cout << "Bandwidth: " << bandwidth << " GB/s\n";
    std::cout << "Verification: " << (correct ? "PASSED ✓" : "FAILED ✗") << "\n";
    std::cout << "\n✓ Optimal performance with HIP!\n";
    
    delete[] h_input; delete[] h_output; delete[] h_reference;
    HIP_CHECK(hipFree(d_input)); HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipEventDestroy(start)); HIP_CHECK(hipEventDestroy(stop));
    
    return 0;
}
