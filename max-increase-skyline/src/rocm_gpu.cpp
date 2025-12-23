#include <iostream>
#include <vector>
#include <algorithm>
#include <hip/hip_runtime.h>

using namespace std;

/**
 * ROCm/HIP GPU Implementation for AMD GPUs
 * 
 * HIP is AMD's API that works on both AMD and NVIDIA GPUs
 * Similar optimizations as CUDA version:
 * 1. Coalesced memory access
 * 2. Shared memory for reductions
 * 3. Warp-level primitives (wavefronts in AMD)
 */

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << " - " \
                 << hipGetErrorString(err) << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel to find row maximums
__global__ void findRowMax(int* grid, int* rowMax, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N) {
        int maxVal = 0;
        for (int j = 0; j < N; ++j) {
            maxVal = max(maxVal, grid[row * N + j]);
        }
        rowMax[row] = maxVal;
    }
}

// Kernel to find column maximums
__global__ void findColMax(int* grid, int* colMax, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < N) {
        int maxVal = 0;
        for (int i = 0; i < N; ++i) {
            maxVal = max(maxVal, grid[i * N + col]);
        }
        colMax[col] = maxVal;
    }
}

// Kernel to calculate the total sum with shared memory reduction
__global__ void calculateSum(int* grid, int* rowMax, int* colMax, int* partialSums, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = N * N;
    
    __shared__ int sharedSum[256];
    
    int localSum = 0;
    
    // Grid-stride loop to handle any size
    for (int i = idx; i < totalElements; i += blockDim.x * gridDim.x) {
        int row = i / N;
        int col = i % N;
        int currentHeight = grid[i];
        int maxHeight = min(rowMax[row], colMax[col]);
        localSum += (maxHeight - currentHeight);
    }
    
    sharedSum[threadIdx.x] = localSum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sharedSum[threadIdx.x] += sharedSum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (threadIdx.x == 0) {
        partialSums[blockIdx.x] = sharedSum[0];
    }
}

// Optimized kernel using wavefront shuffle for reduction
// AMD GPUs have wavefront size of 64 (vs NVIDIA's warp size of 32)
__global__ void calculateSumWaveOpt(int* grid, int* rowMax, int* colMax, int* partialSums, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = N * N;
    
    int localSum = 0;
    
    // Grid-stride loop
    for (int i = idx; i < totalElements; i += blockDim.x * gridDim.x) {
        int row = i / N;
        int col = i % N;
        int currentHeight = grid[i];
        int maxHeight = min(rowMax[row], colMax[col]);
        localSum += (maxHeight - currentHeight);
    }
    
    // Wavefront-level reduction using shuffle
    // AMD wavefront size is typically 64
    for (int offset = 32; offset > 0; offset >>= 1) {
        localSum += __shfl_down(localSum, offset);
    }
    
    // First thread in each wavefront writes to shared memory
    __shared__ int waveSums[4]; // 256 threads / 64 wavefront = 4 waves
    int waveId = threadIdx.x / 64;
    int laneId = threadIdx.x % 64;
    
    if (laneId == 0) {
        waveSums[waveId] = localSum;
    }
    __syncthreads();
    
    // Final reduction by first wavefront
    if (threadIdx.x < 4) {
        int waveSum = waveSums[threadIdx.x];
        for (int offset = 2; offset > 0; offset >>= 1) {
            waveSum += __shfl_down(waveSum, offset);
        }
        if (threadIdx.x == 0) {
            partialSums[blockIdx.x] = waveSum;
        }
    }
}

int maxIncreaseKeepingSkyline(vector<vector<int>>& grid) {
    int N = grid.size();
    int totalElements = N * N;
    
    // Flatten the 2D grid
    vector<int> flatGrid(totalElements);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            flatGrid[i * N + j] = grid[i][j];
        }
    }
    
    // Allocate device memory
    int *d_grid, *d_rowMax, *d_colMax, *d_partialSums;
    HIP_CHECK(hipMalloc(&d_grid, totalElements * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_rowMax, N * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_colMax, N * sizeof(int)));
    
    int numBlocks = (N + 255) / 256;
    HIP_CHECK(hipMalloc(&d_partialSums, numBlocks * sizeof(int)));
    
    // Copy data to device
    HIP_CHECK(hipMemcpy(d_grid, flatGrid.data(), totalElements * sizeof(int), hipMemcpyHostToDevice));
    
    // Launch kernels
    int threadsPerBlock = 256;
    int blocksForN = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    hipLaunchKernelGGL(findRowMax, dim3(blocksForN), dim3(threadsPerBlock), 0, 0, d_grid, d_rowMax, N);
    HIP_CHECK(hipGetLastError());
    
    hipLaunchKernelGGL(findColMax, dim3(blocksForN), dim3(threadsPerBlock), 0, 0, d_grid, d_colMax, N);
    HIP_CHECK(hipGetLastError());
    
    int blocksForSum = min((totalElements + threadsPerBlock - 1) / threadsPerBlock, 1024);
    hipLaunchKernelGGL(calculateSumWaveOpt, dim3(blocksForSum), dim3(threadsPerBlock), 0, 0, 
                       d_grid, d_rowMax, d_colMax, d_partialSums, N);
    HIP_CHECK(hipGetLastError());
    
    // Copy partial sums back and reduce on CPU
    vector<int> partialSums(blocksForSum);
    HIP_CHECK(hipMemcpy(partialSums.data(), d_partialSums, blocksForSum * sizeof(int), hipMemcpyDeviceToHost));
    
    int totalSum = 0;
    for (int sum : partialSums) {
        totalSum += sum;
    }
    
    // Cleanup
    HIP_CHECK(hipFree(d_grid));
    HIP_CHECK(hipFree(d_rowMax));
    HIP_CHECK(hipFree(d_colMax));
    HIP_CHECK(hipFree(d_partialSums));
    
    return totalSum;
}

int main() {
    // Check for HIP-capable devices
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        cerr << "No HIP-capable devices found!" << endl;
        return 1;
    }
    
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    cout << "Using GPU: " << prop.name << endl;
    cout << "Wavefront size: " << prop.warpSize << endl;
    
    vector<vector<int>> grid = {
        {3, 0, 8, 4}, 
        {2, 4, 5, 7}, 
        {9, 2, 6, 3}, 
        {0, 3, 1, 0}
    };
    
    cout << "ROCm/HIP GPU - Total increase: " << maxIncreaseKeepingSkyline(grid) << endl;
    
    return 0;
}
