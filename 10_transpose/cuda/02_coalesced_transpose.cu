/**
 * Coalesced Matrix Transpose with Shared Memory - CUDA
 * 
 * Solution: Use shared memory to enable coalesced reads AND writes
 * 
 * Strategy:
 * 1. Load tile from global memory (coalesced read)
 * 2. Store tile to shared memory
 * 3. Read from shared memory (transposed)
 * 4. Write to global memory (coalesced write)
 * 
 * Expected performance: ~150-250 GB/s (10x improvement!)
 */

#include <iostream>
#include <cuda_runtime.h>
#include "transpose_utils.h"

#define TILE_DIM 32
#define BLOCK_ROWS 8  // To avoid shared memory bank conflicts partially

// Optimized transpose using shared memory
__global__ void transposeCoalesced(const float *input, float *output,
                                   int width, int height) {
    // Shared memory tile
    __shared__ float tile[TILE_DIM][TILE_DIM];
    
    // Global position (reading)
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load tile into shared memory with coalesced reads
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x < width && (y + i) < height) {
            // Coalesced read from global memory
            tile[threadIdx.y + i][threadIdx.x] = input[(y + i) * width + x];
        }
    }
    
    // Wait for all threads to finish loading
    __syncthreads();
    
    // Global position (writing) - transposed
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // Write tile to global memory with coalesced writes
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x < height && (y + i) < width) {
            // Coalesced write to global memory
            // Note: Reading from tile in transposed manner
            output[(y + i) * height + x] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

int main() {
    std::cout << "=== Coalesced Matrix Transpose with Shared Memory (CUDA) ===\n\n";
    
    // Matrix dimensions
    const int WIDTH = 4096;
    const int HEIGHT = 4096;
    const size_t bytes = WIDTH * HEIGHT * sizeof(float);
    
    std::cout << "Matrix size: " << WIDTH << " x " << HEIGHT << "\n";
    std::cout << "Memory: " << bytes / (1024*1024) << " MB\n";
    std::cout << "Tile size: " << TILE_DIM << " x " << TILE_DIM << "\n\n";
    
    // Allocate host memory
    float *h_input = new float[WIDTH * HEIGHT];
    float *h_output = new float[WIDTH * HEIGHT];
    float *h_reference = new float[WIDTH * HEIGHT];
    
    // Initialize input
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
    transposeCoalesced<<<gridDim, blockDim>>>(d_input, d_output, WIDTH, HEIGHT);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int num_iterations = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        transposeCoalesced<<<gridDim, blockDim>>>(d_input, d_output, WIDTH, HEIGHT);
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
    
    std::cout << "Results:\n";
    std::cout << "  Time: " << time_ms << " ms\n";
    std::cout << "  Bandwidth: " << bandwidth << " GB/s\n";
    std::cout << "  Verification: " << (correct ? "PASSED ✓" : "FAILED ✗") << "\n\n";
    
    std::cout << "Optimization Analysis:\n";
    std::cout << "  ✓ Global reads are coalesced\n";
    std::cout << "  ✓ Global writes are coalesced (via shared memory)\n";
    std::cout << "  ✓ Shared memory acts as transpose buffer\n";
    std::cout << "  → Result: ~10x faster than naive!\n\n";
    
    std::cout << "Remaining issue: Shared memory bank conflicts\n";
    std::cout << "Solution: Pad shared memory (see 03_bank_conflict_free.cu)\n";
    
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
