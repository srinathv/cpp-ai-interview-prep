#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#define HIP_CHECK(call) \
    { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    }

// Coalesced memory access (good)
__global__ void coalescedAccess(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f; // Sequential access
    }
}

// Strided memory access (bad - but shown for comparison)
__global__ void stridedAccess(float* data, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f; // Non-coalesced
    }
}

// Shared memory reduction
__global__ void reduceSum(const float* input, float* output, int n) {
    __shared__ float shared[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    shared[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

// Constant memory example
__constant__ float constData[256];

__global__ void useConstantMemory(float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = constData[idx % 256];
    }
}

// Pinned (page-locked) memory demo
void demonstratePinnedMemory() {
    std::cout << "\n=== Pinned Memory ===" << std::endl;
    
    const int N = 10000000;
    const size_t bytes = N * sizeof(float);
    
    // Regular pageable memory
    std::vector<float> pageable(N, 1.0f);
    
    // Pinned memory
    float* pinned;
    HIP_CHECK(hipHostMalloc(&pinned, bytes));
    for (int i = 0; i < N; ++i) pinned[i] = 1.0f;
    
    float *d_data;
    HIP_CHECK(hipMalloc(&d_data, bytes));
    
    // Time pageable transfer
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    HIP_CHECK(hipEventRecord(start));
    HIP_CHECK(hipMemcpy(d_data, pageable.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float pageableTime;
    HIP_CHECK(hipEventElapsedTime(&pageableTime, start, stop));
    
    // Time pinned transfer
    HIP_CHECK(hipEventRecord(start));
    HIP_CHECK(hipMemcpy(d_data, pinned, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float pinnedTime;
    HIP_CHECK(hipEventElapsedTime(&pinnedTime, start, stop));
    
    std::cout << "Pageable memory transfer: " << pageableTime << " ms" << std::endl;
    std::cout << "Pinned memory transfer: " << pinnedTime << " ms" << std::endl;
    std::cout << "Speedup: " << pageableTime / pinnedTime << "x" << std::endl;
    
    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipHostFree(pinned));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
}

// Unified memory example
void demonstrateUnifiedMemory() {
    std::cout << "\n=== Unified Memory ===" << std::endl;
    
    const int N = 1000000;
    float *unified;
    
    // Allocate unified memory
    HIP_CHECK(hipMallocManaged(&unified, N * sizeof(float)));
    
    // Initialize on host
    for (int i = 0; i < N; ++i) {
        unified[i] = static_cast<float>(i);
    }
    
    // Use on device
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(coalescedAccess, dim3(numBlocks), dim3(blockSize), 0, 0,
                       unified, N);
    
    HIP_CHECK(hipDeviceSynchronize());
    
    // Access on host (automatically migrated back)
    std::cout << "First element: " << unified[0] << std::endl;
    std::cout << "Last element: " << unified[N-1] << std::endl;
    
    HIP_CHECK(hipFree(unified));
}

int main() {
    std::cout << "=== GPU Memory Patterns ===" << std::endl;
    
    // Query device properties
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Constant memory: " << prop.totalConstMem / 1024 << " KB" << std::endl;
    
    demonstratePinnedMemory();
    demonstrateUnifiedMemory();
    
    // Reduction example
    std::cout << "\n=== Reduction ===" << std::endl;
    const int N = 1000000;
    std::vector<float> h_input(N, 1.0f);
    
    float *d_input, *d_output;
    int numBlocks = (N + 255) / 256;
    
    HIP_CHECK(hipMalloc(&d_input, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, numBlocks * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), N * sizeof(float), hipMemcpyHostToDevice));
    
    hipLaunchKernelGGL(reduceSum, dim3(numBlocks), dim3(256), 0, 0,
                       d_input, d_output, N);
    
    std::vector<float> h_output(numBlocks);
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, numBlocks * sizeof(float), hipMemcpyDeviceToHost));
    
    float sum = 0.0f;
    for (float val : h_output) sum += val;
    std::cout << "Sum: " << sum << " (expected " << N << ")" << std::endl;
    
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    
    return 0;
}
