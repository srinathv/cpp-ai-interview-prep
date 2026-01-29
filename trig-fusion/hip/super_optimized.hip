/**
 * Super Optimized HIP Trig Fusion
 * 
 * All optimizations:
 * 1. Fused sin/cos with sincosf()
 * 2. Wave shuffle-based reduction (AMD wavefront = 64 threads)
 * 3. Grid-stride loop for scalability
 * 4. Coalesced memory access
 * 5. Minimal shared memory usage
 * 
 * Wave Shuffle on AMD:
 * ====================
 * AMD GPUs use wavefronts of 64 threads (vs NVIDIA's 32-thread warps)
 * __shfl_down() provides register-to-register communication
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WAVE_SIZE 64  // AMD wavefront size
#define BLOCK_SIZE 256
#define WAVES_PER_BLOCK (BLOCK_SIZE / WAVE_SIZE)
#define N 1024

// Wave-level sum reduction using shuffles
__device__ __forceinline__ float wave_reduce_sum(float val) {
    for (int offset = WAVE_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__global__ void trig_super_fused(const float* __restrict__ input,
                                  float* __restrict__ cos_out,
                                  float* __restrict__ sin_out,
                                  float* __restrict__ block_sums,
                                  int n) {
    __shared__ float wave_sums[WAVES_PER_BLOCK];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % WAVE_SIZE;
    int wave_id = tid / WAVE_SIZE;
    
    float local_sum = 0.0f;
    
    // Grid-stride loop
    for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
        float x = input[i];
        
        float s, c;
        sincosf(x, &s, &c);
        
        sin_out[i] = s;
        cos_out[i] = c;
        
        local_sum += x;
    }
    
    // Wave-level reduction using shuffles
    local_sum = wave_reduce_sum(local_sum);
    
    if (lane == 0) {
        wave_sums[wave_id] = local_sum;
    }
    __syncthreads();
    
    // Final reduction across waves
    if (wave_id == 0) {
        local_sum = (lane < WAVES_PER_BLOCK) ? wave_sums[lane] : 0.0f;
        local_sum = wave_reduce_sum(local_sum);
        
        if (lane == 0) {
            block_sums[blockIdx.x] = local_sum;
        }
    }
}

__global__ void final_reduce(float* block_sums, float* result, int num_blocks) {
    __shared__ float wave_sums[WAVES_PER_BLOCK];
    
    int tid = threadIdx.x;
    int lane = tid % WAVE_SIZE;
    int wave_id = tid / WAVE_SIZE;
    
    float val = (tid < num_blocks) ? block_sums[tid] : 0.0f;
    
    val = wave_reduce_sum(val);
    
    if (lane == 0) {
        wave_sums[wave_id] = val;
    }
    __syncthreads();
    
    if (wave_id == 0) {
        val = (lane < WAVES_PER_BLOCK) ? wave_sums[lane] : 0.0f;
        val = wave_reduce_sum(val);
        
        if (lane == 0) {
            *result = val;
        }
    }
}

void trig_super_optimized(float* d_input, float* d_cos, float* d_sin,
                          float* d_block_sums, float* d_sum, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    hipLaunchKernelGGL(trig_super_fused, dim3(blocks), dim3(BLOCK_SIZE),
                       0, 0, d_input, d_cos, d_sin, d_block_sums, n);
    hipLaunchKernelGGL(final_reduce, dim3(1), dim3(BLOCK_SIZE),
                       0, 0, d_block_sums, d_sum, blocks);
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
    
    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_cos, N * sizeof(float));
    hipMalloc(&d_sin, N * sizeof(float));
    hipMalloc(&d_block_sums, blocks * sizeof(float));
    hipMalloc(&d_sum, sizeof(float));
    
    hipMemcpy(d_input, h_input, N * sizeof(float), hipMemcpyHostToDevice);
    
    // Warmup
    trig_super_optimized(d_input, d_cos, d_sin, d_block_sums, d_sum, N);
    hipDeviceSynchronize();
    
    // Benchmark
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    
    hipEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        trig_super_optimized(d_input, d_cos, d_sin, d_block_sums, d_sum, N);
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
    
    printf("=== Super Optimized HIP Trig ===\n");
    printf("Optimizations:\n");
    printf("  1. sincosf() intrinsic\n");
    printf("  2. Wave shuffle reductions (64-thread wavefront)\n");
    printf("  3. Grid-stride loop\n");
    printf("  4. Coalesced memory access\n\n");
    
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
    hipFree(d_block_sums); hipFree(d_sum);
    
    return 0;
}
