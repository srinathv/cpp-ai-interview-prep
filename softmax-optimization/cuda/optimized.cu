/**
 * Optimized CUDA Softmax - Numerically Stable
 * 
 * Key Optimization: Subtract max(x) before exp to prevent overflow
 * 
 * Mathematical Derivation:
 * softmax(x)_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
 * 
 * This is equivalent because:
 * exp(x_i - c) / sum(exp(x_j - c)) = exp(x_i)*exp(-c) / [exp(-c)*sum(exp(x_j))]
 *                                  = exp(x_i) / sum(exp(x_j))
 * 
 * Still uses 3 passes: max, sum, normalize
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define BLOCK_SIZE 256
#define N 1024

// Pass 1: Find maximum value (parallel reduction)
__global__ void find_max(const float* input, float* block_maxes, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load with -inf for out-of-bounds
    sdata[tid] = (idx < n) ? input[idx] : -FLT_MAX;
    __syncthreads();
    
    // Optimized reduction: sequential addressing (no bank conflicts)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        block_maxes[blockIdx.x] = sdata[0];
    }
}

// Final max reduction across blocks
__global__ void final_max(float* block_maxes, int num_blocks) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    sdata[tid] = (tid < num_blocks) ? block_maxes[tid] : -FLT_MAX;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        block_maxes[0] = sdata[0];  // Store final max in first element
    }
}

// Pass 2: Compute exp(x - max) and sum
__global__ void exp_and_sum(const float* input, float* output, 
                            const float* max_val, float* block_sums, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float max_v = *max_val;
    float val = 0.0f;
    
    if (idx < n) {
        val = expf(input[idx] - max_v);  // STABLE: exponent is <= 0
        output[idx] = val;
    }
    
    sdata[tid] = val;
    __syncthreads();
    
    // Sum reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

// Final sum reduction
__global__ void final_sum(float* block_sums, int num_blocks) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    sdata[tid] = (tid < num_blocks) ? block_sums[tid] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        block_sums[0] = sdata[0];
    }
}

// Pass 3: Normalize
__global__ void normalize(float* values, const float* sum, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        values[idx] /= *sum;
    }
}

void softmax_stable(float* d_input, float* d_output, 
                    float* d_workspace, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    float* d_block_vals = d_workspace;  // Reuse for max and sum
    
    // Pass 1: Find max
    find_max<<<blocks, BLOCK_SIZE>>>(d_input, d_block_vals, n);
    final_max<<<1, BLOCK_SIZE>>>(d_block_vals, blocks);
    
    // Pass 2: exp(x - max) and sum
    exp_and_sum<<<blocks, BLOCK_SIZE>>>(d_input, d_output, 
                                         d_block_vals, d_block_vals + 1, n);
    final_sum<<<1, BLOCK_SIZE>>>(d_block_vals + 1, blocks);
    
    // Pass 3: Normalize
    normalize<<<blocks, BLOCK_SIZE>>>(d_output, d_block_vals + 1, n);
}

int main() {
    float *h_input, *h_output;
    float *d_input, *d_output, *d_workspace;
    
    h_input = (float*)malloc(N * sizeof(float));
    h_output = (float*)malloc(N * sizeof(float));
    
    // Test with LARGE values - stable version handles this!
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 1000);  // Range [0, 1000]
    }
    
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_workspace, (blocks + 1) * sizeof(float));
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    softmax_stable(d_input, d_output, d_workspace, N);
    
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify sum = 1
    float sum = 0.0f;
    for (int i = 0; i < N; i++) sum += h_output[i];
    
    printf("Stable Softmax Results:\n");
    printf("Input range: [0, 1000] - would overflow naive version!\n");
    printf("First 5 outputs: ");
    for (int i = 0; i < 5; i++) printf("%.6e ", h_output[i]);
    printf("\nSum of outputs: %.6f (should be 1.0)\n", sum);
    
    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output); cudaFree(d_workspace);
    
    return 0;
}
