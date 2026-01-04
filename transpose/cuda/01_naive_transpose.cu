/**
 * Naive Matrix Transpose - CUDA
 * 
 * Problem: Uncoalesced memory writes cause poor performance
 * 
 * Memory access pattern:
 * - Reads: Coalesced (consecutive threads read consecutive addresses)
 * - Writes: Uncoalesced (consecutive threads write strided addresses)
 * 
 * Expected performance: ~10-20 GB/s (very slow!)
 */

#include <iostream>
#include <cuda_runtime.h>
#include "transpose_utils.h"

// Naive transpose kernel
// Each thread transposes one element
__global__ void transposeNaive(const float *input, float *output, 
                               int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Read is coalesced: input[y * width + x]
        // Write is NOT coalesced: output[x * height + y]
        // This causes major performance issues!
        output[x * height + y] = input[y * width + x];
    }
}

int main() {
    std::cout << "=== Naive Matrix Transpose (CUDA) ===\n\n";
    
    // Matrix dimensions (must be large to see performance difference)
    const int WIDTH = 4096;
    const int HEIGHT = 4096;
    const size_t bytes = WIDTH * HEIGHT * sizeof(float);
    
    std::cout << "Matrix size: " << WIDTH << " x " << HEIGHT << "\n";
    std::cout << "Memory: " << bytes / (1024*1024) << " MB\n\n";
    
    // Allocate host memory
    float *h_input = new float[WIDTH * HEIGHT];
    float *h_output = new float[WIDTH * HEIGHT];
    float *h_reference = new float[WIDTH * HEIGHT];
    
    // Initialize input matrix
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_input[i] = static_cast<float>(i);
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    
    // Launch configuration
    dim3 blockDim(32, 32);  // 1024 threads per block
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x,
                 (HEIGHT + blockDim.y - 1) / blockDim.y);
    
    std::cout << "Launch config:\n";
    std::cout << "  Block: " << blockDim.x << " x " << blockDim.y << "\n";
    std::cout << "  Grid: " << gridDim.x << " x " << gridDim.y << "\n\n";
    
    // Warm up
    transposeNaive<<<gridDim, blockDim>>>(d_input, d_output, WIDTH, HEIGHT);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int num_iterations = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        transposeNaive<<<gridDim, blockDim>>>(d_input, d_output, WIDTH, HEIGHT);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
    time_ms /= num_iterations;  // Average time
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    
    // Verify with CPU
    transposeCPU(h_input, h_reference, WIDTH, HEIGHT);
    bool correct = verifyTranspose(h_output, h_reference, WIDTH * HEIGHT);
    
    // Performance metrics
    double bandwidth = calculateBandwidth(WIDTH, HEIGHT, time_ms);
    
    std::cout << "Results:\n";
    std::cout << "  Time: " << time_ms << " ms\n";
    std::cout << "  Bandwidth: " << bandwidth << " GB/s\n";
    std::cout << "  Verification: " << (correct ? "PASSED ✓" : "FAILED ✗") << "\n\n";
    
    std::cout << "Problem Analysis:\n";
    std::cout << "  ✓ Reads are coalesced (consecutive threads, consecutive memory)\n";
    std::cout << "  ✗ Writes are uncoalesced (consecutive threads, strided memory)\n";
    std::cout << "  → Result: Poor performance (~10-20 GB/s)\n\n";
    
    std::cout << "Solution: Use shared memory (see 02_coalesced_transpose.cu)\n";
    
    // Cleanup
    delete[] h_input;
    delete[] h_output;
    delete[] h_reference;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}
