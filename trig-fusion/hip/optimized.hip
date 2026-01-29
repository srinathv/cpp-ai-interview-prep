/**
 * Optimized HIP Trig Fusion - Single Fused Kernel
 * 
 * Optimizations:
 * 1. Single kernel launch
 * 2. Uses sincosf() to compute sin and cos together
 * 3. Single memory load per element
 * 4. Fused reduction
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 256
#define N 1024

__global__ void trig_fused(const float* input, float* cos_out, float* sin_out,
                           float* partial_sums, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_sum = 0.0f;
    
    if (idx < n) {
        float x = input[idx];
        
        float s, c;
        sincosf(x, &s, &c);  // HIP also has sincosf
        
        sin_out[idx] = s;
        cos_out[idx] = c;
        local_sum = x;
    }
    
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
    
    hipLaunchKernelGGL(trig_fused, dim3(blocks), dim3(BLOCK_SIZE),
                       0, 0, d_input, d_cos, d_sin, d_partial, n);
    hipLaunchKernelGGL(final_sum, dim3(1), dim3(BLOCK_SIZE),
                       0, 0, d_partial, d_sum, blocks);
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
    
    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_cos, N * sizeof(float));
    hipMalloc(&d_sin, N * sizeof(float));
    hipMalloc(&d_partial, blocks * sizeof(float));
    hipMalloc(&d_sum, sizeof(float));
    
    hipMemcpy(d_input, h_input, N * sizeof(float), hipMemcpyHostToDevice);
    
    // Warmup
    trig_optimized(d_input, d_cos, d_sin, d_partial, d_sum, N);
    hipDeviceSynchronize();
    
    // Benchmark
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    
    hipEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        trig_optimized(d_input, d_cos, d_sin, d_partial, d_sum, N);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    
    float ms;
    hipEventElapsedTime(&ms, start, stop);
    
    hipMemcpy(h_cos, d_cos, N * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_sin, d_sin, N * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(&h_sum, d_sum, sizeof(float), hipMemcpyDeviceToHost);
    
    float identity_error = 0.0f;
    for (int i = 0; i < N; i++) {
        float val = h_sin[i] * h_sin[i] + h_cos[i] * h_cos[i];
        identity_error += fabsf(val - 1.0f);
    }
    
    printf("=== Optimized HIP Trig (Fused Kernel) ===\n");
    printf("Optimizations: sincosf(), single kernel, fused reduction\n\n");
    printf("First 5 cos(x):  ");
    for (int i = 0; i < 5; i++) printf("%.4f ", h_cos[i]);
    printf("\nFirst 5 sin(x):  ");
    for (int i = 0; i < 5; i++) printf("%.4f ", h_sin[i]);
    printf("\nSum: %.4f\n", h_sum);
    printf("Identity error: %e\n", identity_error / N);
    printf("Avg time: %.4f us\n", ms);
    
    hipEventDestroy(start);
    hipEventDestroy(stop);
    free(h_input); free(h_cos); free(h_sin);
    hipFree(d_input); hipFree(d_cos); hipFree(d_sin);
    hipFree(d_partial); hipFree(d_sum);
    
    return 0;
}
