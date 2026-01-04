/**
 * Hello CUDA - First CUDA Program
 * 
 * Key concepts:
 * - __global__ keyword for kernel functions
 * - Kernel launch syntax: kernel<<<gridSize, blockSize>>>()
 * - Thread indexing with threadIdx and blockIdx
 * - Device synchronization
 */

#include <iostream>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Simple kernel that prints from GPU
__global__ void hello_from_gpu() {
    printf("Hello from GPU thread %d in block %d!\n", 
           threadIdx.x, blockIdx.x);
}

// Kernel that adds a value to each element
__global__ void add_kernel(int *data, int n, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        data[idx] += value;
    }
}

int main() {
    std::cout << "=== CUDA Hello World ===\n\n";
    
    // Example 1: Query device properties
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    std::cout << "Found " << deviceCount << " CUDA device(s)\n\n";
    
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        
        std::cout << "Device 0: " << prop.name << "\n";
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024*1024) << " MB\n";
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Warp size: " << prop.warpSize << "\n\n";
    }
    
    // Example 2: Launch a simple kernel
    std::cout << "Launching hello kernel with 2 blocks, 4 threads each:\n";
    hello_from_gpu<<<2, 4>>>();
    CUDA_CHECK(cudaDeviceSynchronize());  // Wait for GPU to finish
    std::cout << "\n";
    
    // Example 3: Simple data manipulation
    const int N = 10;
    int h_data[N];
    
    // Initialize host data
    std::cout << "Initial data: ";
    for (int i = 0; i < N; i++) {
        h_data[i] = i;
        std::cout << h_data[i] << " ";
    }
    std::cout << "\n";
    
    // Allocate device memory
    int *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(int), 
                          cudaMemcpyHostToDevice));
    
    // Launch kernel to add 10 to each element
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(int), 
                          cudaMemcpyDeviceToHost));
    
    // Print result
    std::cout << "After adding 10: ";
    for (int i = 0; i < N; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << "\n\n";
    
    // Clean up
    CUDA_CHECK(cudaFree(d_data));
    
    std::cout << "Key takeaways:\n";
    std::cout << "1. Use __global__ for GPU kernels\n";
    std::cout << "2. Launch with <<<blocks, threads>>>\n";
    std::cout << "3. Calculate thread index: blockIdx.x * blockDim.x + threadIdx.x\n";
    std::cout << "4. Always check CUDA errors!\n";
    std::cout << "5. Synchronize when needed\n";
    
    return 0;
}
