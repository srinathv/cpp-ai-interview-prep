/**
 * Super Optimized HIP Softmax - Online Algorithm with Wave Shuffles
 * 
 * Key Optimizations:
 * 1. Online softmax: Compute max and sum in a SINGLE pass
 * 2. Wave shuffle intrinsics (HIP equivalent of warp shuffles)
 * 3. Fused operations to minimize memory traffic
 * 
 * Online Softmax Algorithm:
 * =========================
 * Instead of: max -> sum -> normalize (3 passes)
 * We do: max+sum together -> normalize (2 passes)
 * 
 * The key insight is that we can incrementally update the sum
 * when the running maximum changes:
 * 
 *   m_new = max(m_old, x_i)
 *   d_new = d_old * exp(m_old - m_new) + exp(x_i - m_new)
 * 
 * When combining two partial results (a and b):
 *   m_combined = max(m_a, m_b)
 *   d_combined = d_a * exp(m_a - m_combined) + d_b * exp(m_b - m_combined)
 * 
 * HIP Wave Operations:
 * ====================
 * AMD GPUs use "wavefronts" (64 threads) instead of NVIDIA warps (32)
 * __shfl_down() works the same way as CUDA
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define WAVE_SIZE 64  // AMD wavefront size (use 32 for NVIDIA via HIP)
#define BLOCK_SIZE 256
#define WAVES_PER_BLOCK (BLOCK_SIZE / WAVE_SIZE)
#define N 1024

// Wave-level online softmax reduction
__device__ void wave_online_softmax(float& max_val, float& sum_val) {
    // Reduce across the wavefront using shuffles
    for (int offset = WAVE_SIZE / 2; offset > 0; offset /= 2) {
        float other_max = __shfl_down(max_val, offset);
        float other_sum = __shfl_down(sum_val, offset);
        
        // Online merge: correct sums to common max
        float new_max = fmaxf(max_val, other_max);
        sum_val = sum_val * expf(max_val - new_max) 
                + other_sum * expf(other_max - new_max);
        max_val = new_max;
    }
}

// Single pass: compute max and sum together
__global__ void online_softmax_pass1(const float* input,
                                     float* block_maxes,
                                     float* block_sums,
                                     int n) {
    __shared__ float s_max[WAVES_PER_BLOCK];
    __shared__ float s_sum[WAVES_PER_BLOCK];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % WAVE_SIZE;
    int wave_id = tid / WAVE_SIZE;
    
    // Each thread loads and initializes
    float my_max = (idx < n) ? input[idx] : -FLT_MAX;
    float my_sum = (idx < n) ? 1.0f : 0.0f;  // exp(0) = 1 (relative to self)
    
    // Wave-level reduction
    wave_online_softmax(my_max, my_sum);
    
    // First thread of each wave writes to shared memory
    if (lane == 0) {
        s_max[wave_id] = my_max;
        s_sum[wave_id] = my_sum;
    }
    __syncthreads();
    
    // Final reduction across waves (first wave only)
    if (wave_id == 0) {
        my_max = (lane < WAVES_PER_BLOCK) ? s_max[lane] : -FLT_MAX;
        my_sum = (lane < WAVES_PER_BLOCK) ? s_sum[lane] : 0.0f;
        
        wave_online_softmax(my_max, my_sum);
        
        if (lane == 0) {
            block_maxes[blockIdx.x] = my_max;
            block_sums[blockIdx.x] = my_sum;
        }
    }
}

// Reduce across blocks
__global__ void reduce_blocks(float* block_maxes, float* block_sums,
                              int num_blocks) {
    __shared__ float s_max[WAVES_PER_BLOCK];
    __shared__ float s_sum[WAVES_PER_BLOCK];
    
    int tid = threadIdx.x;
    int lane = tid % WAVE_SIZE;
    int wave_id = tid / WAVE_SIZE;
    
    float my_max = (tid < num_blocks) ? block_maxes[tid] : -FLT_MAX;
    float my_sum = (tid < num_blocks) ? block_sums[tid] : 0.0f;
    
    wave_online_softmax(my_max, my_sum);
    
    if (lane == 0) {
        s_max[wave_id] = my_max;
        s_sum[wave_id] = my_sum;
    }
    __syncthreads();
    
    if (wave_id == 0) {
        my_max = (lane < WAVES_PER_BLOCK) ? s_max[lane] : -FLT_MAX;
        my_sum = (lane < WAVES_PER_BLOCK) ? s_sum[lane] : 0.0f;
        
        wave_online_softmax(my_max, my_sum);
        
        if (lane == 0) {
            block_maxes[0] = my_max;
            block_sums[0] = my_sum;
        }
    }
}

// Apply final softmax
__global__ void apply_softmax(const float* input, float* output,
                              const float* global_max, const float* global_sum,
                              int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float max_v = *global_max;
        float sum_v = *global_sum;
        output[idx] = expf(input[idx] - max_v) / sum_v;
    }
}

void softmax_online(float* d_input, float* d_output,
                    float* d_maxes, float* d_sums, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Pass 1: Online max+sum
    hipLaunchKernelGGL(online_softmax_pass1, dim3(blocks), dim3(BLOCK_SIZE),
                       0, 0, d_input, d_maxes, d_sums, n);
    
    // Reduce across blocks
    hipLaunchKernelGGL(reduce_blocks, dim3(1), dim3(BLOCK_SIZE),
                       0, 0, d_maxes, d_sums, blocks);
    
    // Pass 2: Apply softmax
    hipLaunchKernelGGL(apply_softmax, dim3(blocks), dim3(BLOCK_SIZE),
                       0, 0, d_input, d_output, d_maxes, d_sums, n);
}

int main() {
    float *h_input, *h_output;
    float *d_input, *d_output, *d_maxes, *d_sums;
    
    h_input = (float*)malloc(N * sizeof(float));
    h_output = (float*)malloc(N * sizeof(float));
    
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 1000);
    }
    
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_output, N * sizeof(float));
    hipMalloc(&d_maxes, blocks * sizeof(float));
    hipMalloc(&d_sums, blocks * sizeof(float));
    
    hipMemcpy(d_input, h_input, N * sizeof(float), hipMemcpyHostToDevice);
    
    // Warmup
    softmax_online(d_input, d_output, d_maxes, d_sums, N);
    hipDeviceSynchronize();
    
    // Benchmark
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    
    hipEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        softmax_online(d_input, d_output, d_maxes, d_sums, N);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    
    float ms;
    hipEventElapsedTime(&ms, start, stop);
    
    hipMemcpy(h_output, d_output, N * sizeof(float), hipMemcpyDeviceToHost);
    
    float sum = 0.0f;
    for (int i = 0; i < N; i++) sum += h_output[i];
    
    printf("=== Super Optimized HIP Softmax ===\n");
    printf("Optimizations:\n");
    printf("  1. Online algorithm: max+sum in single pass\n");
    printf("  2. Wave shuffles: register-to-register communication\n");
    printf("  3. Fused kernels: reduced memory traffic\n");
    printf("\nResults:\n");
    printf("  First 5 outputs: ");
    for (int i = 0; i < 5; i++) printf("%.6e ", h_output[i]);
    printf("\n  Sum: %.6f (should be 1.0)\n", sum);
    printf("  Avg time per call: %.4f us\n", ms);
    
    hipEventDestroy(start);
    hipEventDestroy(stop);
    free(h_input); free(h_output);
    hipFree(d_input); hipFree(d_output);
    hipFree(d_maxes); hipFree(d_sums);
    
    return 0;
}
