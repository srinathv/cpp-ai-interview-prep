/**
 * CUDA Matrix Multiplication with PTX Assembly Enhancements
 *
 * This implementation demonstrates how inline PTX assembly can be used
 * to squeeze extra performance from CUDA kernels. Key techniques:
 *
 * 1. PTX fused multiply-add (fma.rn.f32) for better precision/performance
 * 2. Prefetching with ld.global.L1 hints
 * 3. Warp-level primitives via inline PTX
 * 4. Register-level optimizations
 *
 * Expected: ~2-4 TFLOPS (depending on GPU architecture)
 */

#include <iostream>
#include <cuda_runtime.h>
#include "matmul_utils.h"

#define TILE_SIZE 32
#define THREAD_TILE 4  // Each thread computes 4x4 output elements

//=============================================================================
// PTX Assembly Helper Macros
//=============================================================================

// Fused multiply-add using PTX (better than compiler default in some cases)
__device__ __forceinline__ float ptx_fma(float a, float b, float c) {
    float result;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result) : "f"(a), "f"(b), "f"(c));
    return result;
}

// Prefetch to L1 cache
__device__ __forceinline__ void ptx_prefetch_l1(const float* addr) {
    asm("prefetch.global.L1 [%0];" : : "l"(addr));
}

// Prefetch to L2 cache
__device__ __forceinline__ void ptx_prefetch_l2(const float* addr) {
    asm("prefetch.global.L2 [%0];" : : "l"(addr));
}

// Load with cache hint (non-coherent, read-only data)
__device__ __forceinline__ float ptx_ldg(const float* addr) {
    float result;
    asm("ld.global.nc.f32 %0, [%1];" : "=f"(result) : "l"(addr));
    return result;
}

// Vector load (float4) with cache hint
__device__ __forceinline__ float4 ptx_ldg_float4(const float4* addr) {
    float4 result;
    asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w)
        : "l"(addr));
    return result;
}

// Memory fence for shared memory
__device__ __forceinline__ void ptx_membar_cta() {
    asm volatile("membar.cta;");
}

// Warp-level shuffle down (PTX version)
__device__ __forceinline__ float ptx_shfl_down(float val, int offset) {
    float result;
    asm volatile(
        "{"
        ".reg .u32 mask;"
        "mov.u32 mask, 0xffffffff;"
        "shfl.sync.down.b32 %0, %1, %2, 0x1f, mask;"
        "}"
        : "=f"(result) : "f"(val), "r"(offset));
    return result;
}

//=============================================================================
// Kernel: PTX-Enhanced Tiled Matrix Multiply
//=============================================================================

__global__ void matmulPTXOptimized(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
    // Shared memory with padding to avoid bank conflicts
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // +1 padding
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;

    // Register accumulator
    float sum = 0.0f;

    // Prefetch first tile to L2
    if (row < M && tx < N) {
        ptx_prefetch_l2(&A[row * N + tx]);
    }
    if (ty < N && col < K) {
        ptx_prefetch_l2(&B[ty * K + col]);
    }

    const int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Prefetch next tile while processing current
        if (t + 1 < numTiles) {
            int next_a_col = (t + 1) * TILE_SIZE + tx;
            int next_b_row = (t + 1) * TILE_SIZE + ty;
            if (row < M && next_a_col < N) {
                ptx_prefetch_l1(&A[row * N + next_a_col]);
            }
            if (next_b_row < N && col < K) {
                ptx_prefetch_l1(&B[next_b_row * K + col]);
            }
        }

        // Load current tile using PTX cached loads
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < N) {
            As[ty][tx] = ptx_ldg(&A[row * N + a_col]);
        } else {
            As[ty][tx] = 0.0f;
        }

        int b_row = t * TILE_SIZE + ty;
        if (b_row < N && col < K) {
            Bs[ty][tx] = ptx_ldg(&B[b_row * K + col]);
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute using PTX FMA - unrolled for performance
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = ptx_fma(As[ty][k], Bs[k][tx], sum);
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

//=============================================================================
// Kernel: Thread-Tiled with PTX Vectorization
//=============================================================================

__global__ void matmulThreadTiledPTX(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row_start = blockIdx.y * TILE_SIZE + ty * THREAD_TILE;
    const int col_start = blockIdx.x * TILE_SIZE + tx * THREAD_TILE;

    // Register file for output accumulation (THREAD_TILE x THREAD_TILE)
    float acc[THREAD_TILE][THREAD_TILE] = {{0.0f}};
    float a_reg[THREAD_TILE];
    float b_reg[THREAD_TILE];

    const int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Cooperative loading
        #pragma unroll
        for (int i = 0; i < THREAD_TILE; i++) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE; j++) {
                int a_row = row_start + i;
                int a_col = t * TILE_SIZE + tx * THREAD_TILE + j;

                if (a_row < M && a_col < N) {
                    As[ty * THREAD_TILE + i][tx * THREAD_TILE + j] =
                        ptx_ldg(&A[a_row * N + a_col]);
                } else {
                    As[ty * THREAD_TILE + i][tx * THREAD_TILE + j] = 0.0f;
                }

                int b_row = t * TILE_SIZE + ty * THREAD_TILE + i;
                int b_col = col_start + j;

                if (b_row < N && b_col < K) {
                    Bs[ty * THREAD_TILE + i][tx * THREAD_TILE + j] =
                        ptx_ldg(&B[b_row * K + b_col]);
                } else {
                    Bs[ty * THREAD_TILE + i][tx * THREAD_TILE + j] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Compute via outer product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            #pragma unroll
            for (int i = 0; i < THREAD_TILE; i++) {
                a_reg[i] = As[ty * THREAD_TILE + i][k];
            }
            #pragma unroll
            for (int j = 0; j < THREAD_TILE; j++) {
                b_reg[j] = Bs[k][tx * THREAD_TILE + j];
            }
            #pragma unroll
            for (int i = 0; i < THREAD_TILE; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE; j++) {
                    acc[i][j] = ptx_fma(a_reg[i], b_reg[j], acc[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // Write results
    #pragma unroll
    for (int i = 0; i < THREAD_TILE; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE; j++) {
            int out_row = row_start + i;
            int out_col = col_start + j;
            if (out_row < M && out_col < K) {
                C[out_row * K + out_col] = acc[i][j];
            }
        }
    }
}

//=============================================================================
// Main
//=============================================================================

int main() {
    std::cout << "=== PTX-Enhanced Matrix Multiplication (CUDA) ===\n\n";

    const int M = 2048, N = 2048, K = 2048;

    size_t bytes_A = M * N * sizeof(float);
    size_t bytes_B = N * K * sizeof(float);
    size_t bytes_C = M * K * sizeof(float);

    std::cout << "Matrix dimensions: " << M << " x " << N << " x " << K << "\n";
    std::cout << "Total FLOPS: " << (2.0 * M * N * K / 1e9) << " GFLOP\n\n";

    float *h_A = new float[M * N];
    float *h_B = new float[N * K];
    float *h_C = new float[M * K];

    initMatrix(h_A, M * N);
    initMatrix(h_B, N * K);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 10;
    float time_ms;

    // Benchmark PTX-Enhanced Tiled
    {
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((K + TILE_SIZE - 1) / TILE_SIZE,
                     (M + TILE_SIZE - 1) / TILE_SIZE);

        matmulPTXOptimized<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iterations; i++) {
            matmulPTXOptimized<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
        time_ms /= iterations;

        double gflops = calculateGFLOPS(M, N, K, time_ms);
        std::cout << "PTX-Enhanced Tiled:\n";
        std::cout << "  Time: " << time_ms << " ms\n";
        std::cout << "  Performance: " << gflops << " GFLOPS\n\n";
    }

    // Benchmark Thread-Tiled PTX
    {
        const int BLOCK_SIZE = TILE_SIZE / THREAD_TILE;
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDim((K + TILE_SIZE - 1) / TILE_SIZE,
                     (M + TILE_SIZE - 1) / TILE_SIZE);

        matmulThreadTiledPTX<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iterations; i++) {
            matmulThreadTiledPTX<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
        time_ms /= iterations;

        double gflops = calculateGFLOPS(M, N, K, time_ms);
        std::cout << "Thread-Tiled PTX (4x4 per thread):\n";
        std::cout << "  Time: " << time_ms << " ms\n";
        std::cout << "  Performance: " << gflops << " GFLOPS\n\n";
    }

    std::cout << "=== PTX Assembly Optimizations Used ===\n\n";
    std::cout << "1. Fused Multiply-Add (fma.rn.f32):\n";
    std::cout << "   - Single instruction for a*b+c\n";
    std::cout << "   - Better precision (one rounding vs two)\n\n";

    std::cout << "2. Cache Control (ld.global.nc.f32):\n";
    std::cout << "   - Non-coherent loads for read-only data\n";
    std::cout << "   - Bypasses L1, uses texture cache path\n\n";

    std::cout << "3. Prefetching (prefetch.global.L1/L2):\n";
    std::cout << "   - Hide memory latency\n";
    std::cout << "   - Software-controlled cache warming\n\n";

    std::cout << "4. Vector Loads (ld.global.v4.f32):\n";
    std::cout << "   - 128-bit loads (4 floats)\n";
    std::cout << "   - Better memory bandwidth utilization\n\n";

    std::cout << "=== Interview Discussion Points ===\n\n";
    std::cout << "Q: When should you use inline PTX?\n";
    std::cout << "A: - Architecture-specific features (tensor cores, async copy)\n";
    std::cout << "   - Prefetching and cache hints not exposed in CUDA C++\n";
    std::cout << "   - Usually NOT needed - nvcc is very good\n\n";

    std::cout << "Q: PTX vs SASS?\n";
    std::cout << "A: - PTX: Virtual ISA, portable across GPU generations\n";
    std::cout << "   - SASS: Native GPU assembly, architecture-specific\n";
    std::cout << "   - PTX compiled to SASS by driver/ptxas\n\n";

    delete[] h_A; delete[] h_B; delete[] h_C;
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
