/**
 * Vector Addition in CUDA
 * 
 * Classic first parallel program - add two vectors element-wise
 * 
 * Key concepts:
 * - Parallel map operation
 * - Thread indexing and bounds checking
 * - Memory transfer patterns
 * - Grid/block sizing
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check - important for when N is not a multiple of block size
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// CPU version for comparison
void vectorAddCPU(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    std::cout << "=== CUDA Vector Addition ===\n\n";
    
    const int N = 1000000;  // 1 million elements
    const size_t bytes = N * sizeof(float);
    
    // Allocate host memory
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];
    float *h_c_cpu = new float[N];
    
    // Initialize vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = float(i);
        h_b[i] = float(i * 2);
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    
    // Launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    std::cout << "Vector size: " << N << " elements\n";
    std::cout << "Threads per block: " << threadsPerBlock << "\n";
    std::cout << "Blocks per grid: " << blocksPerGrid << "\n\n";
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Time GPU execution
    CUDA_CHECK(cudaEventRecord(start));
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    
    // Time CPU execution for comparison
    auto cpu_start = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_a, h_b, h_c_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    
    // Verify results
    bool correct = true;
    float max_error = 0.0f;
    for (int i = 0; i < N; i++) {
        float error = std::abs(h_c[i] - h_c_cpu[i]);
        max_error = std::max(max_error, error);
        if (error > 1e-5) {
            correct = false;
            break;
        }
    }
    
    std::cout << "Results:\n";
    std::cout << "  GPU time: " << gpu_time << " ms\n";
    std::cout << "  CPU time: " << cpu_time << " ms\n";
    std::cout << "  Speedup: " << cpu_time / gpu_time << "x\n";
    std::cout << "  Verification: " << (correct ? "PASSED" : "FAILED") << "\n";
    std::cout << "  Max error: " << max_error << "\n\n";
    
    // Sample output
    std::cout << "Sample values:\n";
    std::cout << "  h_a[0] = " << h_a[0] << ", h_b[0] = " << h_b[0] 
              << " -> h_c[0] = " << h_c[0] << "\n";
    std::cout << "  h_a[999] = " << h_a[999] << ", h_b[999] = " << h_b[999] 
              << " -> h_c[999] = " << h_c[999] << "\n\n";
    
    // Clean up
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_c_cpu;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    std::cout << "Key interview points:\n";
    std::cout << "1. Thread index: blockIdx.x * blockDim.x + threadIdx.x\n";
    std::cout << "2. Always check bounds: if (idx < n)\n";
    std::cout << "3. Grid size: (N + blockSize - 1) / blockSize\n";
    std::cout << "4. Memory pattern: H2D, compute, D2H\n";
    std::cout << "5. Use events for accurate timing\n";
    
    return 0;
}
