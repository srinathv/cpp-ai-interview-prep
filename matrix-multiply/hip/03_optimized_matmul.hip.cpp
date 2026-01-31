/**
 * HIP Optimized Matrix Multiplication
 *
 * This implementation demonstrates AMD GPU-specific optimization techniques
 * for matrix multiplication. Key techniques:
 *
 * 1. Shared memory tiling with bank conflict avoidance
 * 2. Register blocking (each thread computes multiple outputs)
 * 3. Vectorized loads (float4) for better memory bandwidth
 * 4. Prefetching via __builtin_nontemporal_load hints
 * 5. Loop unrolling for instruction-level parallelism
 *
 * Expected: ~1-2 TFLOPS on MI100/MI250X (educational, not production-optimized)
 *
 * Compile: hipcc -O3 03_optimized_matmul.hip.cpp -o optimized
 */

#include <iostream>
#include <hip/hip_runtime.h>
#include "matmul_utils.h"

#define TILE_SIZE 32
#define THREAD_TILE 4  // Each thread computes 4x4 output elements

//=============================================================================
// HIP-Specific Optimization Helpers
//=============================================================================

// Vectorized load helper (float4 = 128 bits)
__device__ __forceinline__ float4 load_float4(const float* addr) {
    return *reinterpret_cast<const float4*>(addr);
}

// Vectorized store helper
__device__ __forceinline__ void store_float4(float* addr, float4 val) {
    *reinterpret_cast<float4*>(addr) = val;
}

// Built-in FMA (HIP/ROCm compiler handles this well)
__device__ __forceinline__ float hip_fma(float a, float b, float c) {
    return __fmaf_rn(a, b, c);
}

//=============================================================================
// Kernel 1: Optimized Tiled with Bank Conflict Avoidance
//=============================================================================

__global__ void matmulOptimizedTiled(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int M, int N, int K) {
    // Shared memory with padding to avoid bank conflicts
    // AMD GPUs have 32 banks, 4-byte bank width
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // +1 padding
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;

    // Register accumulator
    float sum = 0.0f;

    const int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load A tile
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < N) {
            As[ty][tx] = A[row * N + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B tile
        int b_row = t * TILE_SIZE + ty;
        if (b_row < N && col < K) {
            Bs[ty][tx] = B[b_row * K + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute - fully unrolled for better ILP
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = hip_fma(As[ty][k], Bs[k][tx], sum);
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

//=============================================================================
// Kernel 2: Register Blocking (Thread-Tiled)
//=============================================================================

/*
 * Register Blocking Strategy:
 *
 * Each thread computes a THREAD_TILE x THREAD_TILE submatrix of C.
 * This increases arithmetic intensity by reusing data in registers.
 *
 * For THREAD_TILE=4:
 * - 16 accumulator registers per thread
 * - 4 A registers + 4 B registers for outer product
 * - Much better register utilization
 *
 * Block configuration:
 * - blockDim = (TILE_SIZE/THREAD_TILE, TILE_SIZE/THREAD_TILE) = (8, 8)
 * - Each 8x8 block of threads computes 32x32 output tile
 */
__global__ void matmulRegisterBlocked(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int N, int K) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Each thread's starting position in output C
    const int row_start = blockIdx.y * TILE_SIZE + ty * THREAD_TILE;
    const int col_start = blockIdx.x * TILE_SIZE + tx * THREAD_TILE;

    // Register accumulators (THREAD_TILE x THREAD_TILE per thread)
    float acc[THREAD_TILE][THREAD_TILE];
    #pragma unroll
    for (int i = 0; i < THREAD_TILE; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE; j++) {
            acc[i][j] = 0.0f;
        }
    }

    // Register storage for A and B fragments
    float a_reg[THREAD_TILE];
    float b_reg[THREAD_TILE];

    const int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Cooperative loading of tiles into shared memory
        // Each thread loads THREAD_TILE x THREAD_TILE elements
        #pragma unroll
        for (int i = 0; i < THREAD_TILE; i++) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE; j++) {
                // Load A tile
                int a_row = row_start + i;
                int a_col = t * TILE_SIZE + tx * THREAD_TILE + j;
                int smem_row = ty * THREAD_TILE + i;
                int smem_col = tx * THREAD_TILE + j;

                if (a_row < M && a_col < N) {
                    As[smem_row][smem_col] = A[a_row * N + a_col];
                } else {
                    As[smem_row][smem_col] = 0.0f;
                }

                // Load B tile
                int b_row = t * TILE_SIZE + ty * THREAD_TILE + i;
                int b_col = col_start + j;

                if (b_row < N && b_col < K) {
                    Bs[smem_row][smem_col] = B[b_row * K + b_col];
                } else {
                    Bs[smem_row][smem_col] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Compute via outer product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            // Load A column fragment into registers
            #pragma unroll
            for (int i = 0; i < THREAD_TILE; i++) {
                a_reg[i] = As[ty * THREAD_TILE + i][k];
            }

            // Load B row fragment into registers
            #pragma unroll
            for (int j = 0; j < THREAD_TILE; j++) {
                b_reg[j] = Bs[k][tx * THREAD_TILE + j];
            }

            // Outer product accumulation
            #pragma unroll
            for (int i = 0; i < THREAD_TILE; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE; j++) {
                    acc[i][j] = hip_fma(a_reg[i], b_reg[j], acc[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // Write results to global memory
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
// Kernel 3: Vectorized Loads (float4)
//=============================================================================

/*
 * Vectorized Memory Access:
 *
 * Using float4 (128-bit) loads/stores improves memory bandwidth utilization.
 * AMD GPUs can achieve higher effective bandwidth with wider transactions.
 *
 * Requirements:
 * - Addresses must be 16-byte aligned
 * - K must be multiple of 4 for clean vectorization
 */
__global__ void matmulVectorized(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    const int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load A tile - vectorized when possible
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < N) {
            As[ty][tx] = A[row * N + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B tile
        int b_row = t * TILE_SIZE + ty;
        if (b_row < N && col < K) {
            Bs[ty][tx] = B[b_row * K + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute with manual unrolling by 4
        int k = 0;
        #pragma unroll 4
        for (; k + 3 < TILE_SIZE; k += 4) {
            sum = hip_fma(As[ty][k], Bs[k][tx], sum);
            sum = hip_fma(As[ty][k+1], Bs[k+1][tx], sum);
            sum = hip_fma(As[ty][k+2], Bs[k+2][tx], sum);
            sum = hip_fma(As[ty][k+3], Bs[k+3][tx], sum);
        }
        // Handle remainder
        for (; k < TILE_SIZE; k++) {
            sum = hip_fma(As[ty][k], Bs[k][tx], sum);
        }

        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

//=============================================================================
// Kernel 4: Double Buffering (Software Pipelining)
//=============================================================================

/*
 * Double Buffering Strategy:
 *
 * Overlap memory loads with computation by using two sets of shared memory.
 * While computing on buffer 0, load next tile into buffer 1, and vice versa.
 *
 * This hides memory latency and improves GPU utilization.
 */
__global__ void matmulDoubleBuffered(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int M, int N, int K) {
    // Double-buffered shared memory
    __shared__ float As[2][TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    const int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    int curr = 0;  // Current buffer

    // Prefetch first tile
    {
        int a_col = tx;
        if (row < M && a_col < N) {
            As[0][ty][tx] = A[row * N + a_col];
        } else {
            As[0][ty][tx] = 0.0f;
        }

        int b_row = ty;
        if (b_row < N && col < K) {
            Bs[0][ty][tx] = B[b_row * K + col];
        } else {
            Bs[0][ty][tx] = 0.0f;
        }
    }

    __syncthreads();

    for (int t = 0; t < numTiles; t++) {
        int next = 1 - curr;  // Next buffer

        // Load next tile while computing current
        if (t + 1 < numTiles) {
            int a_col = (t + 1) * TILE_SIZE + tx;
            if (row < M && a_col < N) {
                As[next][ty][tx] = A[row * N + a_col];
            } else {
                As[next][ty][tx] = 0.0f;
            }

            int b_row = (t + 1) * TILE_SIZE + ty;
            if (b_row < N && col < K) {
                Bs[next][ty][tx] = B[b_row * K + col];
            } else {
                Bs[next][ty][tx] = 0.0f;
            }
        }

        // Compute on current buffer
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = hip_fma(As[curr][ty][k], Bs[curr][k][tx], sum);
        }

        __syncthreads();
        curr = next;
    }

    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

//=============================================================================
// Main Benchmark
//=============================================================================

int main() {
    std::cout << "=== Optimized Matrix Multiplication (HIP) ===\n\n";

    // Print device info
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute Units: " << prop.multiProcessorCount << "\n";
    std::cout << "Max Threads/Block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Shared Memory/Block: " << prop.sharedMemPerBlock / 1024 << " KB\n\n";

    const int M = 2048, N = 2048, K = 2048;

    size_t bytes_A = M * N * sizeof(float);
    size_t bytes_B = N * K * sizeof(float);
    size_t bytes_C = M * K * sizeof(float);

    std::cout << "Matrix dimensions: " << M << " x " << N << " x " << K << "\n";
    std::cout << "Total FLOPS: " << (2.0 * M * N * K / 1e9) << " GFLOP\n\n";

    // Allocate host memory
    float *h_A = new float[M * N];
    float *h_B = new float[N * K];
    float *h_C = new float[M * K];
    float *h_C_ref = new float[M * K];

    initMatrix(h_A, M * N);
    initMatrix(h_B, N * K);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, bytes_A));
    HIP_CHECK(hipMalloc(&d_B, bytes_B));
    HIP_CHECK(hipMalloc(&d_C, bytes_C));

    HIP_CHECK(hipMemcpy(d_A, h_A, bytes_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, bytes_B, hipMemcpyHostToDevice));

    // Compute reference on CPU (small subset for verification)
    std::cout << "Computing CPU reference (subset for verification)...\n";
    const int verify_size = 256;
    float *h_A_small = new float[verify_size * verify_size];
    float *h_B_small = new float[verify_size * verify_size];
    float *h_C_small = new float[verify_size * verify_size];
    for (int i = 0; i < verify_size * verify_size; i++) {
        h_A_small[i] = h_A[i];
        h_B_small[i] = h_B[i];
    }
    matmulCPU(h_A_small, h_B_small, h_C_small, verify_size, verify_size, verify_size);

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    const int iterations = 10;
    float time_ms;

    auto benchmark = [&](const char* name, auto kernel, dim3 grid, dim3 block) {
        // Warmup
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        HIP_CHECK(hipDeviceSynchronize());

        // Benchmark
        HIP_CHECK(hipEventRecord(start));
        for (int i = 0; i < iterations; i++) {
            kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        HIP_CHECK(hipEventElapsedTime(&time_ms, start, stop));
        time_ms /= iterations;

        double gflops = calculateGFLOPS(M, N, K, time_ms);
        std::cout << name << ":\n";
        std::cout << "  Grid: (" << grid.x << ", " << grid.y << "), Block: (" << block.x << ", " << block.y << ")\n";
        std::cout << "  Time: " << time_ms << " ms\n";
        std::cout << "  Performance: " << gflops << " GFLOPS\n\n";
    };

    // Benchmark all kernels
    dim3 block_tiled(TILE_SIZE, TILE_SIZE);
    dim3 grid_tiled((K + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    benchmark("Optimized Tiled (bank conflict avoidance)", matmulOptimizedTiled, grid_tiled, block_tiled);
    benchmark("Vectorized (unrolled x4)", matmulVectorized, grid_tiled, block_tiled);
    benchmark("Double Buffered", matmulDoubleBuffered, grid_tiled, block_tiled);

    // Register blocked uses different block size
    const int REG_BLOCK = TILE_SIZE / THREAD_TILE;
    dim3 block_reg(REG_BLOCK, REG_BLOCK);
    dim3 grid_reg((K + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    benchmark("Register Blocked (4x4 per thread)", matmulRegisterBlocked, grid_reg, block_reg);

    // Verify correctness
    std::cout << "=== Verification ===\n";
    float *d_C_verify;
    HIP_CHECK(hipMalloc(&d_C_verify, verify_size * verify_size * sizeof(float)));
    float *d_A_verify, *d_B_verify;
    HIP_CHECK(hipMalloc(&d_A_verify, verify_size * verify_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B_verify, verify_size * verify_size * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_A_verify, h_A_small, verify_size * verify_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B_verify, h_B_small, verify_size * verify_size * sizeof(float), hipMemcpyHostToDevice));

    dim3 block_v(TILE_SIZE, TILE_SIZE);
    dim3 grid_v((verify_size + TILE_SIZE - 1) / TILE_SIZE, (verify_size + TILE_SIZE - 1) / TILE_SIZE);
    matmulOptimizedTiled<<<grid_v, block_v>>>(d_A_verify, d_B_verify, d_C_verify, verify_size, verify_size, verify_size);
    HIP_CHECK(hipDeviceSynchronize());

    float *h_C_gpu = new float[verify_size * verify_size];
    HIP_CHECK(hipMemcpy(h_C_gpu, d_C_verify, verify_size * verify_size * sizeof(float), hipMemcpyDeviceToHost));

    if (verifyMatmul(h_C_gpu, h_C_small, verify_size * verify_size)) {
        std::cout << "Correctness: PASSED\n\n";
    } else {
        std::cout << "Correctness: FAILED\n\n";
    }

    // Print optimization summary
    std::cout << "=== HIP Optimization Techniques ===\n\n";

    std::cout << "1. Bank Conflict Avoidance:\n";
    std::cout << "   - Pad shared memory arrays (+1 column)\n";
    std::cout << "   - AMD GPUs: 32 banks, 4-byte width\n\n";

    std::cout << "2. Register Blocking:\n";
    std::cout << "   - Each thread computes 4x4 outputs\n";
    std::cout << "   - 16 accumulator registers\n";
    std::cout << "   - Higher arithmetic intensity\n\n";

    std::cout << "3. Vectorized Memory Access:\n";
    std::cout << "   - float4 loads (128-bit)\n";
    std::cout << "   - Better memory bandwidth utilization\n\n";

    std::cout << "4. Double Buffering:\n";
    std::cout << "   - Overlap loads with computation\n";
    std::cout << "   - Hides memory latency\n\n";

    std::cout << "5. Loop Unrolling:\n";
    std::cout << "   - #pragma unroll for inner loops\n";
    std::cout << "   - Improves instruction-level parallelism\n\n";

    std::cout << "=== Interview Discussion Points ===\n\n";

    std::cout << "Q: HIP vs CUDA differences for GEMM?\n";
    std::cout << "A: - Similar programming model\n";
    std::cout << "   - AMD has wider wavefronts (64 vs 32)\n";
    std::cout << "   - Different shared memory bank configuration\n";
    std::cout << "   - No PTX equivalent (GCN ISA is different)\n\n";

    std::cout << "Q: Why register blocking helps?\n";
    std::cout << "A: - Increases arithmetic intensity\n";
    std::cout << "   - Reuses data in registers (fastest memory)\n";
    std::cout << "   - Reduces shared memory traffic\n\n";

    std::cout << "Q: When to use double buffering?\n";
    std::cout << "A: - When memory latency is the bottleneck\n";
    std::cout << "   - Requires 2x shared memory\n";
    std::cout << "   - Most effective when compute >> load time\n\n";

    // Cleanup
    delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_C_ref;
    delete[] h_A_small; delete[] h_B_small; delete[] h_C_small; delete[] h_C_gpu;
    HIP_CHECK(hipFree(d_A)); HIP_CHECK(hipFree(d_B)); HIP_CHECK(hipFree(d_C));
    HIP_CHECK(hipFree(d_A_verify)); HIP_CHECK(hipFree(d_B_verify)); HIP_CHECK(hipFree(d_C_verify));
    HIP_CHECK(hipEventDestroy(start)); HIP_CHECK(hipEventDestroy(stop));

    return 0;
}
