/**
 * Optimized HIP Softmax - Numerically Stable
 * 
 * Key Optimization: Subtract max(x) before exp to prevent overflow
 * 
 * Mathematical Property (Shift Invariance):
 * softmax(x - c) = softmax(x) for any constant c
 * 
 * Proof:
 * exp(x_i - c) / sum(exp(x_j - c)) = exp(x_i)exp(-c) / [exp(-c)sum(exp(x_j))]
 *                                  = exp(x_i) / sum(exp(x_j))
 * 
 * By choosing c = max(x), all exponents become <= 0
 * So exp values are in (0, 1], preventing overflow
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define BLOCK_SIZE 256
#define N 1024

// Pass 1: Find maximum (parallel reduction)
__global__ void find_max(const float* input, float* block_maxes, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : -FLT_MAX;
    __syncthreads();
    
    // Sequential addressing reduction (no bank conflicts)
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
        block_maxes[0] = sdata[0];
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
        val = expf(input[idx] - max_v);  // STABLE: exponent <= 0
        output[idx] = val;
    }
    
    sdata[tid] = val;
    __syncthreads();
    
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
    float* d_block_vals = d_workspace;
    
    // Pass 1: Find max
    hipLaunchKernelGGL(find_max, dim3(blocks), dim3(BLOCK_SIZE),
                       0, 0, d_input, d_block_vals, n);
    hipLaunchKernelGGL(final_max, dim3(1), dim3(BLOCK_SIZE),
                       0, 0, d_block_vals, blocks);
    
    // Pass 2: exp and sum
    hipLaunchKernelGGL(exp_and_sum, dim3(blocks), dim3(BLOCK_SIZE),
                       0, 0, d_input, d_output, d_block_vals, 
                       d_block_vals + 1, n);
    hipLaunchKernelGGL(final_sum, dim3(1), dim3(BLOCK_SIZE),
                       0, 0, d_block_vals + 1, blocks);
    
    // Pass 3: Normalize
    hipLaunchKernelGGL(normalize, dim3(blocks), dim3(BLOCK_SIZE),
                       0, 0, d_output, d_block_vals + 1, n);
}

int main() {
    float *h_input, *h_output;
    float *d_input, *d_output, *d_workspace;
    
    h_input = (float*)malloc(N * sizeof(float));
    h_output = (float*)malloc(N * sizeof(float));
    
    // Large values - stable version handles this
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 1000);
    }
    
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_output, N * sizeof(float));
    hipMalloc(&d_workspace, (blocks + 1) * sizeof(float));
    
    hipMemcpy(d_input, h_input, N * sizeof(float), hipMemcpyHostToDevice);
    
    softmax_stable(d_input, d_output, d_workspace, N);
    
    hipMemcpy(h_output, d_output, N * sizeof(float), hipMemcpyDeviceToHost);
    
    float sum = 0.0f;
    for (int i = 0; i < N; i++) sum += h_output[i];
    
    printf("Stable HIP Softmax Results:\n");
    printf("Input range: [0, 1000] - would overflow naive version!\n");
    printf("First 5 outputs: ");
    for (int i = 0; i < 5; i++) printf("%.6e ", h_output[i]);
    printf("\nSum of outputs: %.6f (should be 1.0)\n", sum);
    
    free(h_input); free(h_output);
    hipFree(d_input); hipFree(d_output); hipFree(d_workspace);
    
    return 0;
}
