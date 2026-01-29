/**
 * Super Optimized CUDA Trig Fusion
 * 
 * All optimizations combined:
 * 1. Fused sin/cos with sincosf() intrinsic
 * 2. Warp shuffle-based reduction (no shared memory for warp)
 * 3. Grid-stride loop for arbitrary sizes
 * 4. Coalesced memory access
 * 5. Single kernel launch
 * 
 * Warp Shuffle Reduction:
 * =======================
 * Instead of shared memory + __syncthreads():
 *   __shfl_down_sync(mask, val, offset)
 * 
 * Benefits:
 * - Register-to-register: lower latency
 * - No bank conflicts
 * - Implicit synchronization within warp
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define N 1024

// Warp-level sum reduction using shuffles
__device__ __forceinline__ float warp_reduce_sum(float val) {
    unsigned int mask = 0xffffffff;
    
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

// Super optimized: fused trig + shuffle reduction
__global__ void trig_super_fused(const float* __restrict__ input,
                                  float* __restrict__ cos_out,
                                  float* __restrict__ sin_out,
                                  float* __restrict__ block_sums,
                                  int n) {
    __shared__ float warp_sums[WARPS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    float local_sum = 0.0f;
    
    // Grid-stride loop for handling large arrays
    for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
        float x = input[i];  // Coalesced load
        
        float s, c;
        sincosf(x, &s, &c);  // Simultaneous sin/cos
        
        sin_out[i] = s;  // Coalesced store
        cos_out[i] = c;  // Coalesced store
        
        local_sum += x;
    }
    
    // Warp-level reduction using shuffles (no shared memory!)
    local_sum = warp_reduce_sum(local_sum);
    
    // First thread of each warp writes to shared memory
    if (lane == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Final reduction across warps (first warp only)
    if (warp_id == 0) {
        local_sum = (lane < WARPS_PER_BLOCK) ? warp_sums[lane] : 0.0f;
        local_sum = warp_reduce_sum(local_sum);
        
        if (lane == 0) {
            block_sums[blockIdx.x] = local_sum;
        }
    }
}

// Final reduction kernel (also uses shuffles)
__global__ void final_reduce(float* block_sums, float* result, int num_blocks) {
    __shared__ float warp_sums[WARPS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    float val = (tid < num_blocks) ? block_sums[tid] : 0.0f;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        val = (lane < WARPS_PER_BLOCK) ? warp_sums[lane] : 0.0f;
        val = warp_reduce_sum(val);
        
        if (lane == 0) {
            *result = val;
        }
    }
}

void trig_super_optimized(float* d_input, float* d_cos, float* d_sin,
                          float* d_block_sums, float* d_sum, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Single fused kernel with shuffle reductions
    trig_super_fused<<<blocks, BLOCK_SIZE>>>(d_input, d_cos, d_sin, 
                                              d_block_sums, n);
    
    // Final reduction
    final_reduce<<<1, BLOCK_SIZE>>>(d_block_sums, d_sum, blocks);
}

int main() {
    float *h_input, *h_cos, *h_sin;
    float h_sum;
    float *d_input, *d_cos, *d_sin, *d_block_sums, *d_sum;
    
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
    cudaMalloc(&d_block_sums, blocks * sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Warmup
    trig_super_optimized(d_input, d_cos, d_sin, d_block_sums, d_sum, N);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        trig_super_optimized(d_input, d_cos, d_sin, d_block_sums, d_sum, N);
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
    
    printf("=== Super Optimized CUDA Trig ===\n");
    printf("Optimizations:\n");
    printf("  1. Fused sin/cos with sincosf()\n");
    printf("  2. Warp shuffle reductions (__shfl_down_sync)\n");
    printf("  3. Grid-stride loop for scalability\n");
    printf("  4. Coalesced memory access\n");
    printf("  5. Minimal shared memory usage\n\n");
    
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
    cudaFree(d_block_sums); cudaFree(d_sum);
    
    return 0;
}
