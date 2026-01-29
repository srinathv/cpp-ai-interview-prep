/**
 * Optimized CUDA Trig Fusion - Single Fused Kernel
 * 
 * Optimizations:
 * 1. Single kernel launch (reduces overhead)
 * 2. Uses sincosf() intrinsic (computes both at once)
 * 3. Single memory load per element
 * 4. Fused reduction with trig computation
 * 
 * sincosf() Explanation:
 * - Hardware can compute sin and cos together efficiently
 * - Both share intermediate computations
 * - ~1.5x faster than separate sin() and cos() calls
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 256
#define N 1024

// Fused kernel: compute sin, cos, and partial sum in one pass
__global__ void trig_fused(const float* input, float* cos_out, float* sin_out,
                           float* partial_sums, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_sum = 0.0f;
    
    if (idx < n) {
        float x = input[idx];  // Single memory load!
        
        float s, c;
        sincosf(x, &s, &c);  // Compute both at once
        
        sin_out[idx] = s;
        cos_out[idx] = c;
        local_sum = x;
    }
    
    // Reduction in shared memory
    sdata[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Second kernel: reduce partial sums
__global__ void final_sum(float* partial_sums, float* result, int num_blocks) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    sdata[tid] = (tid < num_blocks) ? partial_sums[tid] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *result = sdata[0];
    }
}

void trig_optimized(float* d_input, float* d_cos, float* d_sin,
                    float* d_partial, float* d_sum, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Single fused kernel for trig + partial sums
    trig_fused<<<blocks, BLOCK_SIZE>>>(d_input, d_cos, d_sin, d_partial, n);
    
    // Final reduction
    final_sum<<<1, BLOCK_SIZE>>>(d_partial, d_sum, blocks);
}

int main() {
    float *h_input, *h_cos, *h_sin;
    float h_sum;
    float *d_input, *d_cos, *d_sin, *d_partial, *d_sum;
    
    h_input = (float*)malloc(N * sizeof(float));
    h_cos = (float*)malloc(N * sizeof(float));
    h_sin = (float*)malloc(N * sizeof(float));
    
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 628) / 100.0f;
    }
    
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_cos, N * sizeof(float));
    cudaMalloc(&d_sin, N * sizeof(float));
    cudaMalloc(&d_partial, blocks * sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Warmup
    trig_optimized(d_input, d_cos, d_sin, d_partial, d_sum, N);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        trig_optimized(d_input, d_cos, d_sin, d_partial, d_sum, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaMemcpy(h_cos, d_cos, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sin, d_sin, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    
    float identity_error = 0.0f;
    for (int i = 0; i < N; i++) {
        float val = h_sin[i] * h_sin[i] + h_cos[i] * h_cos[i];
        identity_error += fabsf(val - 1.0f);
    }
    
    printf("=== Optimized CUDA Trig (Fused Kernel) ===\n");
    printf("Optimizations:\n");
    printf("  1. Single kernel for sin/cos computation\n");
    printf("  2. sincosf() intrinsic\n");
    printf("  3. Single memory load per element\n\n");
    
    printf("First 5 inputs:  ");
    for (int i = 0; i < 5; i++) printf("%.4f ", h_input[i]);
    printf("\nFirst 5 cos(x):  ");
    for (int i = 0; i < 5; i++) printf("%.4f ", h_cos[i]);
    printf("\nFirst 5 sin(x):  ");
    for (int i = 0; i < 5; i++) printf("%.4f ", h_sin[i]);
    printf("\n\nSum of inputs: %.4f\n", h_sum);
    printf("Identity error: %e\n", identity_error / N);
    printf("Avg time per call: %.4f us\n", ms);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_input); free(h_cos); free(h_sin);
    cudaFree(d_input); cudaFree(d_cos); cudaFree(d_sin);
    cudaFree(d_partial); cudaFree(d_sum);
    
    return 0;
}
