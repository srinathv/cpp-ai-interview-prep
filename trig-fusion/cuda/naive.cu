/**
 * Naive CUDA Trig Fusion - Separate Kernels
 * 
 * Problems:
 * 1. Multiple kernel launches (overhead)
 * 2. Multiple memory passes (bandwidth inefficient)
 * 3. Computes sin and cos separately (could use sincosf)
 * 
 * Operations:
 * - Compute cos(x) for all elements
 * - Compute sin(x) for all elements
 * - Compute sum of original elements
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 256
#define N 1024

// Kernel 1: Compute cos(x) - separate memory pass
__global__ void compute_cos(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = cosf(input[idx]);
    }
}

// Kernel 2: Compute sin(x) - another memory pass
__global__ void compute_sin(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = sinf(input[idx]);
    }
}

// Kernel 3: Sum reduction - yet another memory pass
__global__ void sum_reduce(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Basic reduction (could be optimized)
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
    
    // Reset sum
    cudaMemset(d_sum, 0, sizeof(float));
    
    // Three separate kernels - inefficient!
    compute_cos<<<blocks, BLOCK_SIZE>>>(d_input, d_cos, n);
    compute_sin<<<blocks, BLOCK_SIZE>>>(d_input, d_sin, n);
    sum_reduce<<<blocks, BLOCK_SIZE>>>(d_input, d_sum, n);
}

int main() {
    float *h_input, *h_cos, *h_sin;
    float h_sum;
    float *d_input, *d_cos, *d_sin, *d_sum;
    
    h_input = (float*)malloc(N * sizeof(float));
    h_cos = (float*)malloc(N * sizeof(float));
    h_sin = (float*)malloc(N * sizeof(float));
    
    // Initialize with random angles
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 628) / 100.0f;  // 0 to 2*pi
    }
    
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_cos, N * sizeof(float));
    cudaMalloc(&d_sin, N * sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Warmup
    trig_naive(d_input, d_cos, d_sin, d_sum, N);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        trig_naive(d_input, d_cos, d_sin, d_sum, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaMemcpy(h_cos, d_cos, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sin, d_sin, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify using sin^2 + cos^2 = 1
    float identity_error = 0.0f;
    for (int i = 0; i < N; i++) {
        float val = h_sin[i] * h_sin[i] + h_cos[i] * h_cos[i];
        identity_error += fabsf(val - 1.0f);
    }
    
    printf("=== Naive CUDA Trig (Separate Kernels) ===\n");
    printf("Problems: 3 kernel launches, 5 memory passes\n\n");
    printf("First 5 inputs:  ");
    for (int i = 0; i < 5; i++) printf("%.4f ", h_input[i]);
    printf("\nFirst 5 cos(x):  ");
    for (int i = 0; i < 5; i++) printf("%.4f ", h_cos[i]);
    printf("\nFirst 5 sin(x):  ");
    for (int i = 0; i < 5; i++) printf("%.4f ", h_sin[i]);
    printf("\n\nSum of inputs: %.4f\n", h_sum);
    printf("Identity error (sin^2+cos^2=1): %e\n", identity_error / N);
    printf("Avg time per call: %.4f us\n", ms);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_input); free(h_cos); free(h_sin);
    cudaFree(d_input); cudaFree(d_cos); cudaFree(d_sin); cudaFree(d_sum);
    
    return 0;
}
