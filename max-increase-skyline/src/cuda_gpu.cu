#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

using namespace std;

/**
 * CUDA GPU Implementation
 * 
 * GPU Optimizations:
 * 1. Coalesced memory access patterns
 * 2. Shared memory for reductions
 * 3. Warp-level primitives for efficiency
 * 4. Multiple kernel launches for different phases
 */

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                 << cudaGetErrorString(err) << endl; \
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

// Kernel to calculate the total sum
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

// Optimized kernel using warp shuffle for reduction
__global__ void calculateSumWarpOpt(int* grid, int* rowMax, int* colMax, int* partialSums, int N) {
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
    
    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        localSum += __shfl_down_sync(0xffffffff, localSum, offset);
    }
    
    // First thread in each warp writes to shared memory
    __shared__ int warpSums[8]; // Assumes 256 threads = 8 warps
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    
    if (laneId == 0) {
        warpSums[warpId] = localSum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (threadIdx.x < 8) {
        int warpSum = warpSums[threadIdx.x];
        for (int offset = 4; offset > 0; offset >>= 1) {
            warpSum += __shfl_down_sync(0xff, warpSum, offset);
        }
        if (threadIdx.x == 0) {
            partialSums[blockIdx.x] = warpSum;
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
    CUDA_CHECK(cudaMalloc(&d_grid, totalElements * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rowMax, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colMax, N * sizeof(int)));
    
    int numBlocks = (N + 255) / 256;
    CUDA_CHECK(cudaMalloc(&d_partialSums, numBlocks * sizeof(int)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_grid, flatGrid.data(), totalElements * sizeof(int), cudaMemcpyHostToDevice));
    
    // Launch kernels
    int threadsPerBlock = 256;
    int blocksForN = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    findRowMax<<<blocksForN, threadsPerBlock>>>(d_grid, d_rowMax, N);
    CUDA_CHECK(cudaGetLastError());
    
    findColMax<<<blocksForN, threadsPerBlock>>>(d_grid, d_colMax, N);
    CUDA_CHECK(cudaGetLastError());
    
    int blocksForSum = min((totalElements + threadsPerBlock - 1) / threadsPerBlock, 1024);
    calculateSumWarpOpt<<<blocksForSum, threadsPerBlock>>>(d_grid, d_rowMax, d_colMax, d_partialSums, N);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy partial sums back and reduce on CPU
    vector<int> partialSums(blocksForSum);
    CUDA_CHECK(cudaMemcpy(partialSums.data(), d_partialSums, blocksForSum * sizeof(int), cudaMemcpyDeviceToHost));
    
    int totalSum = 0;
    for (int sum : partialSums) {
        totalSum += sum;
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_grid));
    CUDA_CHECK(cudaFree(d_rowMax));
    CUDA_CHECK(cudaFree(d_colMax));
    CUDA_CHECK(cudaFree(d_partialSums));
    
    return totalSum;
}

int main() {
    vector<vector<int>> grid = {
        {3, 0, 8, 4}, 
        {2, 4, 5, 7}, 
        {9, 2, 6, 3}, 
        {0, 3, 1, 0}
    };
    
    cout << "CUDA GPU - Total increase: " << maxIncreaseKeepingSkyline(grid) << endl;
    
    return 0;
}
