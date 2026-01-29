/**
 * Naive HIP Softmax Implementation
 * 
 * Problems with this approach:
 * 1. No numerical stability - exp(large_value) overflows
 * 2. Multiple kernel launches (inefficient)
 * 3. Divergent branching in reduction
 * 
 * Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j))
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 256
#define N 1024

__global__ void compute_exp(const float* input, float* exp_values, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        exp_values[idx] = expf(input[idx]);  // DANGER: Can overflow!
    }
}

__global__ void sum_reduce_naive(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Naive reduction with divergent branching (inefficient)
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

__global__ void normalize(float* exp_values, const float* sum, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        exp_values[idx] /= *sum;
    }
}

void softmax_naive(float* d_input, float* d_output, float* d_sum, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    hipMemset(d_sum, 0, sizeof(float));
    
    hipLaunchKernelGGL(compute_exp, dim3(blocks), dim3(BLOCK_SIZE), 
                       0, 0, d_input, d_output, n);
    hipLaunchKernelGGL(sum_reduce_naive, dim3(blocks), dim3(BLOCK_SIZE), 
                       0, 0, d_output, d_sum, n);
    hipLaunchKernelGGL(normalize, dim3(blocks), dim3(BLOCK_SIZE), 
                       0, 0, d_output, d_sum, n);
}

int main() {
    float *h_input, *h_output;
    float *d_input, *d_output, *d_sum;
    
    h_input = (float*)malloc(N * sizeof(float));
    h_output = (float*)malloc(N * sizeof(float));
    
    // Small values to avoid overflow
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 10) - 5.0f;
    }
    
    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_output, N * sizeof(float));
    hipMalloc(&d_sum, sizeof(float));
    
    hipMemcpy(d_input, h_input, N * sizeof(float), hipMemcpyHostToDevice);
    
    softmax_naive(d_input, d_output, d_sum, N);
    
    hipMemcpy(h_output, d_output, N * sizeof(float), hipMemcpyDeviceToHost);
    
    printf("Naive HIP Softmax - First 5 outputs: ");
    for (int i = 0; i < 5; i++) printf("%.6f ", h_output[i]);
    printf("\n");
    
    printf("\n=== Overflow Problem Demo ===\n");
    printf("exp(100) = %e\n", expf(100.0f));
    printf("exp(500) = %e (inf!)\n", expf(500.0f));
    
    free(h_input); free(h_output);
    hipFree(d_input); hipFree(d_output); hipFree(d_sum);
    
    return 0;
}
