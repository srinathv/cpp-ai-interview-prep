/**
 * Naive HIP Trig Fusion - Separate Kernels
 * 
 * Problems:
 * 1. Multiple kernel launches (overhead)
 * 2. Multiple memory passes (bandwidth inefficient)
 * 3. Computes sin and cos separately
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 256
#define N 1024

__global__ void compute_cos(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = cosf(input[idx]);
    }
}

__global__ void compute_sin(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = sinf(input[idx]);
    }
}

__global__ void sum_reduce(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

void trig_naive(float* d_input, float* d_cos, float* d_sin,
                float* d_sum, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    hipMemset(d_sum, 0, sizeof(float));
    
    hipLaunchKernelGGL(compute_cos, dim3(blocks), dim3(BLOCK_SIZE),
                       0, 0, d_input, d_cos, n);
    hipLaunchKernelGGL(compute_sin, dim3(blocks), dim3(BLOCK_SIZE),
                       0, 0, d_input, d_sin, n);
    hipLaunchKernelGGL(sum_reduce, dim3(blocks), dim3(BLOCK_SIZE),
                       0, 0, d_input, d_sum, n);
}

int main() {
    float *h_input, *h_cos, *h_sin;
    float h_sum;
    float *d_input, *d_cos, *d_sin, *d_sum;
    
    h_input = (float*)malloc(N * sizeof(float));
    h_cos = (float*)malloc(N * sizeof(float));
    h_sin = (float*)malloc(N * sizeof(float));
    
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 628) / 100.0f;
    }
    
    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_cos, N * sizeof(float));
    hipMalloc(&d_sin, N * sizeof(float));
    hipMalloc(&d_sum, sizeof(float));
    
    hipMemcpy(d_input, h_input, N * sizeof(float), hipMemcpyHostToDevice);
    
    // Warmup
    trig_naive(d_input, d_cos, d_sin, d_sum, N);
    hipDeviceSynchronize();
    
    // Benchmark
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    
    hipEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        trig_naive(d_input, d_cos, d_sin, d_sum, N);
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
    
    printf("=== Naive HIP Trig (Separate Kernels) ===\n");
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
    hipFree(d_input); hipFree(d_cos); hipFree(d_sin); hipFree(d_sum);
    
    return 0;
}
