/**
 * Bank Conflict-Free Matrix Transpose - CUDA
 * 
 * Final optimization: Eliminate shared memory bank conflicts
 * 
 * Problem: When transposing, threads in a warp access the same bank
 * Solution: Pad shared memory array by 1 element
 * 
 * Shared memory banks:
 * - 32 banks on most GPUs
 * - 32-bit words per bank
 * - Simultaneous access to same bank = conflict
 * 
 * Expected performance: ~300-400 GB/s (near memory bandwidth!)
 */

#include <iostream>
#include <cuda_runtime.h>
#include "transpose_utils.h"

#define TILE_DIM 32
#define BLOCK_ROWS 8

// Optimized transpose with bank conflict resolution
__global__ void transposeBankConflictFree(const float *input, float *output,
                                          int width, int height) {
    // Shared memory with padding to avoid bank conflicts
    // TILE_DIM + 1 eliminates conflicts when reading transposed data
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 is the key!
    
    // Global position (reading)
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load tile with coalesced reads
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x < width && (y + i) < height) {
            tile[threadIdx.y + i][threadIdx.x] = input[(y + i) * width + x];
        }
    }
    
    __syncthreads();
    
    // Global position (writing) - transposed
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // Write tile with coalesced writes
    // The +1 padding ensures no bank conflicts here!
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x < height && (y + i) < width) {
            output[(y + i) * height + x] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

int main() {
    std::cout << "=== Bank Conflict-Free Matrix Transpose (CUDA) ===\n\n";
    
    // Matrix dimensions
    const int WIDTH = 4096;
    const int HEIGHT = 4096;
    const size_t bytes = WIDTH * HEIGHT * sizeof(float);
    
    std::cout << "Matrix size: " << WIDTH << " x " << HEIGHT << "\n";
    std::cout << "Memory: " << bytes / (1024*1024) << " MB\n";
    std::cout << "Tile size: " << TILE_DIM << " x " << TILE_DIM << "\n";
    std::cout << "Shared memory padding: +1 (bank conflict avoidance)\n\n";
    
    // Allocate host memory
    float *h_input = new float[WIDTH * HEIGHT];
    float *h_output = new float[WIDTH * HEIGHT];
    float *h_reference = new float[WIDTH * HEIGHT];
    
    // Initialize
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_input[i] = static_cast<float>(i);
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    
    // Launch configuration
    dim3 blockDim(TILE_DIM, BLOCK_ROWS);
    dim3 gridDim((WIDTH + TILE_DIM - 1) / TILE_DIM,
                 (HEIGHT + TILE_DIM - 1) / TILE_DIM);
    
    std::cout << "Launch config:\n";
    std::cout << "  Block: " << blockDim.x << " x " << blockDim.y << "\n";
    std::cout << "  Grid: " << gridDim.x << " x " << gridDim.y << "\n\n";
    
    // Warm up
    transposeBankConflictFree<<<gridDim, blockDim>>>(d_input, d_output, WIDTH, HEIGHT);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int num_iterations = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        transposeBankConflictFree<<<gridDim, blockDim>>>(d_input, d_output, WIDTH, HEIGHT);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
    time_ms /= num_iterations;
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    
    // Verify
    transposeCPU(h_input, h_reference, WIDTH, HEIGHT);
    bool correct = verifyTranspose(h_output, h_reference, WIDTH * HEIGHT);
    
    // Performance
    double bandwidth = calculateBandwidth(WIDTH, HEIGHT, time_ms);
    
    // Get theoretical peak bandwidth
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    double peak_bandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    double efficiency = (bandwidth / peak_bandwidth) * 100.0;
    
    std::cout << "Results:\n";
    std::cout << "  Time: " << time_ms << " ms\n";
    std::cout << "  Bandwidth: " << bandwidth << " GB/s\n";
    std::cout << "  Peak device bandwidth: " << peak_bandwidth << " GB/s\n";
    std::cout << "  Efficiency: " << efficiency << "%\n";
    std::cout << "  Verification: " << (correct ? "PASSED ✓" : "FAILED ✗") << "\n\n";
    
    std::cout << "Final Optimization Summary:\n";
    std::cout << "  ✓ Coalesced global reads\n";
    std::cout << "  ✓ Coalesced global writes\n";
    std::cout << "  ✓ No bank conflicts (padding)\n";
    std::cout << "  → Result: Near-optimal performance!\n\n";
    
    std::cout << "Key Insight:\n";
    std::cout << "  The +1 padding breaks the stride that causes bank conflicts.\n";
    std::cout << "  Without padding: tile[0][0], tile[1][0], tile[2][0]... same bank\n";
    std::cout << "  With padding: Different banks due to offset!\n\n";
    
    std::cout << "Interview Answer:\n";
    std::cout << "  'To optimize transpose, use shared memory tiling for coalesced\n";
    std::cout << "   memory access, and add +1 padding to avoid bank conflicts.'\n";
    
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
