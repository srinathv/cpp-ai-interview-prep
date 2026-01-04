#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

// HIP kernel for vector addition
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Error checking macro
#define HIP_CHECK(call) \
    { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    }

int main() {
    const int N = 1000000;
    const size_t bytes = N * sizeof(float);
    
    // Allocate host memory
    std::vector<float> h_a(N, 1.0f);
    std::vector<float> h_b(N, 2.0f);
    std::vector<float> h_c(N, 0.0f);
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));
    
    // Copy data to device
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    hipLaunchKernelGGL(vectorAdd, dim3(numBlocks), dim3(blockSize), 0, 0, 
                       d_a, d_b, d_c, N);
    
    // Or use CUDA-style syntax (also works in HIP):
    // vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
    
    HIP_CHECK(hipGetLastError());
    
    // Copy result back to host
    HIP_CHECK(hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost));
    
    // Verify result
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (std::abs(h_c[i] - 3.0f) > 1e-5) {
            success = false;
            std::cout << "Error at index " << i << ": " << h_c[i] << std::endl;
            break;
        }
    }
    
    if (success) {
        std::cout << "Vector addition successful!" << std::endl;
        std::cout << "Result: " << h_c[0] << " (expected 3.0)" << std::endl;
    }
    
    // Clean up
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    
    return 0;
}
