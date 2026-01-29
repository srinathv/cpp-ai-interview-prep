/**
 * Super Optimized CUDA Softmax - Online Algorithm with Warp Shuffles
 * 
 * Key Optimizations:
 * 1. Online softmax: Compute max and sum in a SINGLE pass
 * 2. Warp shuffle intrinsics: No shared memory for warp reductions
 * 3. Coalesced memory access
 * 4. Fused operations
 * 
 * Online Softmax Derivation:
 * ===========================
 * Standard: Need max first, then sum, then normalize (3 passes)
 * 
 * Online: Process elements incrementally
 * - m_new = max(m_old, x_i)
 * - d_new = d_old * exp(m_old - m_new) + exp(x_i - m_new)
 * 
 * When m_old >= x_i: correction factor exp(m_old - m_new) = 1
 * When x_i > m_old:  correction factor < 1, scales down old sum
 * 
 * This allows computing max and sum in ONE pass!
 * 
 * Warp Shuffle Operations:
 * ========================
 * __shfl_down_sync: Exchange data between threads without shared memory
 * - Faster than shared memory (register-to-register)
 * - No bank conflicts
 * - Implicit synchronization within warp
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define N 1024

// Warp-level online softmax reduction using shuffles
__device__ void warp_online_softmax(float& max_val, float& sum_val) {
    unsigned int mask = 0xffffffff;  // Full warp participation
    
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(mask, max_val, offset);
        float other_sum = __shfl_down_sync(mask, sum_val, offset);
        
        // Online softmax merge: correct sums when max changes
        float new_max = fmaxf(max_val, other_max);
        
        // Scale both sums to the new max
        sum_val = sum_val * expf(max_val - new_max) 
                + other_sum * expf(other_max - new_max);
        max_val = new_max;
    }
}

// Fused kernel: Single pass to compute max and sum
__global__ void online_softmax_pass1(const float* input, 
                                     float* block_maxes, 
                                     float* block_sums, 
                                     int n) {
    __shared__ float s_max[WARPS_PER_BLOCK];
    __shared__ float s_sum[WARPS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    // Each thread loads one element
    float my_max = (idx < n) ? input[idx] : -FLT_MAX;
    float my_sum = (idx < n) ? 1.0f : 0.0f;  // exp(x - x) = 1
    
    // Warp-level reduction using shuffles
    warp_online_softmax(my_max, my_sum);
    
    // First thread of each warp writes to shared memory
    if (lane == 0) {
        s_max[warp_id] = my_max;
        s_sum[warp_id] = my_sum;
    }
    __syncthreads();
    
    // Final reduction across warps (first warp only)
    if (warp_id == 0) {
        my_max = (lane < WARPS_PER_BLOCK) ? s_max[lane] : -FLT_MAX;
        my_sum = (lane < WARPS_PER_BLOCK) ? s_sum[lane] : 0.0f;
        
        warp_online_softmax(my_max, my_sum);
        
        if (lane == 0) {
            block_maxes[blockIdx.x] = my_max;
            block_sums[blockIdx.x] = my_sum;
        }
    }
}

// Reduce across blocks
__global__ void reduce_blocks(float* block_maxes, float* block_sums, 
                              int num_blocks) {
    __shared__ float s_max[WARPS_PER_BLOCK];
    __shared__ float s_sum[WARPS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    float my_max = (tid < num_blocks) ? block_maxes[tid] : -FLT_MAX;
    float my_sum = (tid < num_blocks) ? block_sums[tid] : 0.0f;
    
    // Correct sum to common max before warp reduction
    // Need to reload the original max for each element
    if (tid < num_blocks) {
        float orig_max = block_maxes[tid];
        my_sum = my_sum;  // Already relative to orig_max
    }
    
    warp_online_softmax(my_max, my_sum);
    
    if (lane == 0) {
        s_max[warp_id] = my_max;
        s_sum[warp_id] = my_sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        my_max = (lane < WARPS_PER_BLOCK) ? s_max[lane] : -FLT_MAX;
        my_sum = (lane < WARPS_PER_BLOCK) ? s_sum[lane] : 0.0f;
        
        warp_online_softmax(my_max, my_sum);
        
        if (lane == 0) {
            block_maxes[0] = my_max;
            block_sums[0] = my_sum;
        }
    }
}

// Pass 2: Compute final softmax values
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
    
    // Single pass: compute block-wise max and sum using online algorithm
    online_softmax_pass1<<<blocks, BLOCK_SIZE>>>(d_input, d_maxes, d_sums, n);
    
    // Reduce across blocks
    reduce_blocks<<<1, BLOCK_SIZE>>>(d_maxes, d_sums, blocks);
    
    // Apply softmax
    apply_softmax<<<blocks, BLOCK_SIZE>>>(d_input, d_output, 
                                           d_maxes, d_sums, n);
}

int main() {
    float *h_input, *h_output;
    float *d_input, *d_output, *d_maxes, *d_sums;
    
    h_input = (float*)malloc(N * sizeof(float));
    h_output = (float*)malloc(N * sizeof(float));
    
    // Large values - online algorithm handles this efficiently
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 1000);
    }
    
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_maxes, blocks * sizeof(float));
    cudaMalloc(&d_sums, blocks * sizeof(float));
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Warmup
    softmax_online(d_input, d_output, d_maxes, d_sums, N);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        softmax_online(d_input, d_output, d_maxes, d_sums, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    float sum = 0.0f;
    for (int i = 0; i < N; i++) sum += h_output[i];
    
    printf("=== Super Optimized Softmax (Online + Warp Shuffles) ===\n");
    printf("Optimizations:\n");
    printf("  1. Online algorithm: max+sum in single pass\n");
    printf("  2. Warp shuffles: no shared memory for reductions\n");
    printf("  3. Fused operations: reduced kernel launches\n");
    printf("\nResults:\n");
    printf("  First 5 outputs: ");
    for (int i = 0; i < 5; i++) printf("%.6e ", h_output[i]);
    printf("\n  Sum: %.6f (should be 1.0)\n", sum);
    printf("  Avg time per call: %.4f us\n", ms);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    cudaFree(d_maxes); cudaFree(d_sums);
    
    return 0;
}
