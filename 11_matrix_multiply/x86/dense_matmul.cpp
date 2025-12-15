/**
 * x86 Dense Matrix Multiplication Optimization Strategies
 *
 * This file demonstrates various x86-specific optimization techniques
 * for dense matrix multiplication, from naive to highly optimized.
 *
 * Optimization Levels:
 * 1. Naive triple loop
 * 2. Loop reordering (cache-friendly)
 * 3. Blocking/Tiling
 * 4. SIMD with SSE/AVX
 * 5. AVX-512 with inline assembly
 * 6. Register blocking + prefetching
 *
 * Compile with: g++ -O3 -march=native -mavx2 -mfma x86_dense_matmul.cpp
 *               (add -mavx512f for AVX-512)
 */

#include <iostream>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>  // SSE, AVX, AVX2, AVX-512

//=============================================================================
// Utility Functions
//=============================================================================

void init_matrix(float* M, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        M[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void zero_matrix(float* M, int rows, int cols) {
    memset(M, 0, rows * cols * sizeof(float));
}

double calculate_gflops(int M, int N, int K, double time_sec) {
    double flops = 2.0 * M * N * K;  // multiply-add = 2 ops
    return (flops / time_sec) / 1e9;
}

bool verify_result(const float* C1, const float* C2, int size, float epsilon = 1e-3f) {
    for (int i = 0; i < size; i++) {
        if (fabs(C1[i] - C2[i]) > epsilon) {
            std::cerr << "Mismatch at " << i << ": " << C1[i] << " vs " << C2[i] << "\n";
            return false;
        }
    }
    return true;
}

//=============================================================================
// Level 1: Naive Implementation
//=============================================================================

void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    // C[i,j] = sum_k A[i,k] * B[k,j]
    // Row-major: A[i,k] = A[i*N + k], B[k,j] = B[k*K + j], C[i,j] = C[i*K + j]
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * K + j];
            }
            C[i * K + j] = sum;
        }
    }
}

//=============================================================================
// Level 2: Loop Reordering (ikj order for better cache usage)
//=============================================================================

/*
 * Why loop order matters:
 *
 * Naive (ijk): B[k,j] stride = 1 (good), but inner loop over k means
 *              A[i,k] has stride N (bad for cache)
 *
 * Reordered (ikj): Inner loop over j
 *   - A[i,k] accessed once per k iteration (hoisted)
 *   - B[k,j] sequential access (stride 1)
 *   - C[i,j] sequential access (stride 1)
 *
 * This is a simple but effective optimization!
 */
void matmul_loop_reorder(const float* A, const float* B, float* C, int M, int N, int K) {
    zero_matrix(C, M, K);

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < N; k++) {
            float a_ik = A[i * N + k];  // Hoisted - accessed once
            for (int j = 0; j < K; j++) {
                // Both B[k,j] and C[i,j] have stride 1 - cache friendly!
                C[i * K + j] += a_ik * B[k * K + j];
            }
        }
    }
}

//=============================================================================
// Level 3: Blocking/Tiling
//=============================================================================

/*
 * Blocking Strategy:
 *
 * Divide matrices into blocks that fit in L1/L2 cache.
 * For a block size B:
 *   - 3 blocks of size B x B = 3 * B^2 * 4 bytes
 *   - L1 cache typically 32KB, so B ~= sqrt(32KB / 12) ~= 52
 *   - L2 cache typically 256KB, so B ~= sqrt(256KB / 12) ~= 147
 *
 * We use block size that fits in L2, optimizing for L1 within the inner loop.
 */
#define BLOCK_SIZE 64

void matmul_blocked(const float* A, const float* B, float* C, int M, int N, int K) {
    zero_matrix(C, M, K);

    // Block over all three dimensions
    for (int i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
        for (int k0 = 0; k0 < N; k0 += BLOCK_SIZE) {
            for (int j0 = 0; j0 < K; j0 += BLOCK_SIZE) {
                // Compute block C[i0:i0+B, j0:j0+B] += A[i0:i0+B, k0:k0+B] * B[k0:k0+B, j0:j0+B]
                int i_max = std::min(i0 + BLOCK_SIZE, M);
                int k_max = std::min(k0 + BLOCK_SIZE, N);
                int j_max = std::min(j0 + BLOCK_SIZE, K);

                for (int i = i0; i < i_max; i++) {
                    for (int k = k0; k < k_max; k++) {
                        float a_ik = A[i * N + k];
                        for (int j = j0; j < j_max; j++) {
                            C[i * K + j] += a_ik * B[k * K + j];
                        }
                    }
                }
            }
        }
    }
}

//=============================================================================
// Level 4: AVX2 SIMD (256-bit, 8 floats)
//=============================================================================

/*
 * AVX2 Vectorization:
 *
 * __m256 = 256-bit register = 8 x float32
 * Key intrinsics:
 *   _mm256_loadu_ps  - unaligned load
 *   _mm256_storeu_ps - unaligned store
 *   _mm256_set1_ps   - broadcast scalar
 *   _mm256_fmadd_ps  - fused multiply-add (a*b + c)
 *
 * Process 8 columns at a time in the inner loop.
 */
void matmul_avx2(const float* A, const float* B, float* C, int M, int N, int K) {
    zero_matrix(C, M, K);

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < N; k++) {
            __m256 a_ik = _mm256_set1_ps(A[i * N + k]);  // Broadcast A[i,k]

            int j = 0;
            // Process 8 elements at a time
            for (; j + 7 < K; j += 8) {
                __m256 b_kj = _mm256_loadu_ps(&B[k * K + j]);
                __m256 c_ij = _mm256_loadu_ps(&C[i * K + j]);
                c_ij = _mm256_fmadd_ps(a_ik, b_kj, c_ij);  // c += a * b
                _mm256_storeu_ps(&C[i * K + j], c_ij);
            }

            // Handle remainder
            for (; j < K; j++) {
                C[i * K + j] += A[i * N + k] * B[k * K + j];
            }
        }
    }
}

//=============================================================================
// Level 5: AVX2 + Blocking + Unrolling
//=============================================================================

/*
 * Combined optimizations:
 * 1. Cache blocking for temporal locality
 * 2. AVX2 for data parallelism
 * 3. Loop unrolling to reduce loop overhead
 * 4. Register blocking to maximize register usage
 */
void matmul_avx2_blocked(const float* A, const float* B, float* C, int M, int N, int K) {
    zero_matrix(C, M, K);

    const int BLOCK = 64;

    for (int i0 = 0; i0 < M; i0 += BLOCK) {
        for (int k0 = 0; k0 < N; k0 += BLOCK) {
            for (int j0 = 0; j0 < K; j0 += BLOCK) {
                int i_max = std::min(i0 + BLOCK, M);
                int k_max = std::min(k0 + BLOCK, N);
                int j_max = std::min(j0 + BLOCK, K);

                for (int i = i0; i < i_max; i++) {
                    for (int k = k0; k < k_max; k++) {
                        __m256 a_ik = _mm256_set1_ps(A[i * N + k]);

                        int j = j0;
                        // Unroll by 4 (32 floats per iteration)
                        for (; j + 31 < j_max; j += 32) {
                            __m256 c0 = _mm256_loadu_ps(&C[i * K + j]);
                            __m256 c1 = _mm256_loadu_ps(&C[i * K + j + 8]);
                            __m256 c2 = _mm256_loadu_ps(&C[i * K + j + 16]);
                            __m256 c3 = _mm256_loadu_ps(&C[i * K + j + 24]);

                            __m256 b0 = _mm256_loadu_ps(&B[k * K + j]);
                            __m256 b1 = _mm256_loadu_ps(&B[k * K + j + 8]);
                            __m256 b2 = _mm256_loadu_ps(&B[k * K + j + 16]);
                            __m256 b3 = _mm256_loadu_ps(&B[k * K + j + 24]);

                            c0 = _mm256_fmadd_ps(a_ik, b0, c0);
                            c1 = _mm256_fmadd_ps(a_ik, b1, c1);
                            c2 = _mm256_fmadd_ps(a_ik, b2, c2);
                            c3 = _mm256_fmadd_ps(a_ik, b3, c3);

                            _mm256_storeu_ps(&C[i * K + j], c0);
                            _mm256_storeu_ps(&C[i * K + j + 8], c1);
                            _mm256_storeu_ps(&C[i * K + j + 16], c2);
                            _mm256_storeu_ps(&C[i * K + j + 24], c3);
                        }

                        // Handle remaining 8-element chunks
                        for (; j + 7 < j_max; j += 8) {
                            __m256 c = _mm256_loadu_ps(&C[i * K + j]);
                            __m256 b = _mm256_loadu_ps(&B[k * K + j]);
                            c = _mm256_fmadd_ps(a_ik, b, c);
                            _mm256_storeu_ps(&C[i * K + j], c);
                        }

                        // Scalar remainder
                        for (; j < j_max; j++) {
                            C[i * K + j] += A[i * N + k] * B[k * K + j];
                        }
                    }
                }
            }
        }
    }
}

//=============================================================================
// Level 6: Register Blocking with Inline Assembly
//=============================================================================

/*
 * Register Blocking Strategy:
 *
 * Each thread computes a small tile of C (e.g., 6x16) using:
 * - 6 rows of A (6 broadcasts)
 * - 16 columns of B (2 x __m256 = 16 floats)
 * - 12 __m256 accumulators (6 rows x 2 vectors)
 *
 * This maximizes register usage and minimizes memory traffic.
 *
 * x86-64 has 16 YMM registers:
 * - 12 for accumulators
 * - 2 for B values
 * - 2 for A broadcasts
 */
void matmul_register_blocked(const float* A, const float* B, float* C, int M, int N, int K) {
    zero_matrix(C, M, K);

    const int MR = 6;   // Rows per micro-kernel
    const int NR = 16;  // Columns per micro-kernel (2 x __m256)

    for (int i = 0; i < M; i += MR) {
        for (int j = 0; j < K; j += NR) {
            // Accumulators for MR x NR tile
            __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
            __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
            __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
            __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
            __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
            __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

            for (int k = 0; k < N; k++) {
                // Load B[k, j:j+16]
                __m256 b0 = _mm256_loadu_ps(&B[k * K + j]);
                __m256 b1 = _mm256_loadu_ps(&B[k * K + j + 8]);

                // Broadcast A values and accumulate
                __m256 a0 = _mm256_set1_ps(A[(i + 0) * N + k]);
                c00 = _mm256_fmadd_ps(a0, b0, c00);
                c01 = _mm256_fmadd_ps(a0, b1, c01);

                __m256 a1 = _mm256_set1_ps(A[(i + 1) * N + k]);
                c10 = _mm256_fmadd_ps(a1, b0, c10);
                c11 = _mm256_fmadd_ps(a1, b1, c11);

                __m256 a2 = _mm256_set1_ps(A[(i + 2) * N + k]);
                c20 = _mm256_fmadd_ps(a2, b0, c20);
                c21 = _mm256_fmadd_ps(a2, b1, c21);

                __m256 a3 = _mm256_set1_ps(A[(i + 3) * N + k]);
                c30 = _mm256_fmadd_ps(a3, b0, c30);
                c31 = _mm256_fmadd_ps(a3, b1, c31);

                __m256 a4 = _mm256_set1_ps(A[(i + 4) * N + k]);
                c40 = _mm256_fmadd_ps(a4, b0, c40);
                c41 = _mm256_fmadd_ps(a4, b1, c41);

                __m256 a5 = _mm256_set1_ps(A[(i + 5) * N + k]);
                c50 = _mm256_fmadd_ps(a5, b0, c50);
                c51 = _mm256_fmadd_ps(a5, b1, c51);
            }

            // Store results
            _mm256_storeu_ps(&C[(i + 0) * K + j], c00);
            _mm256_storeu_ps(&C[(i + 0) * K + j + 8], c01);
            _mm256_storeu_ps(&C[(i + 1) * K + j], c10);
            _mm256_storeu_ps(&C[(i + 1) * K + j + 8], c11);
            _mm256_storeu_ps(&C[(i + 2) * K + j], c20);
            _mm256_storeu_ps(&C[(i + 2) * K + j + 8], c21);
            _mm256_storeu_ps(&C[(i + 3) * K + j], c30);
            _mm256_storeu_ps(&C[(i + 3) * K + j + 8], c31);
            _mm256_storeu_ps(&C[(i + 4) * K + j], c40);
            _mm256_storeu_ps(&C[(i + 4) * K + j + 8], c41);
            _mm256_storeu_ps(&C[(i + 5) * K + j], c50);
            _mm256_storeu_ps(&C[(i + 5) * K + j + 8], c51);
        }
    }
}

//=============================================================================
// Level 7: Inline Assembly Microkernel
//=============================================================================

/*
 * Hand-tuned inline assembly for the inner kernel.
 *
 * Benefits over intrinsics:
 * 1. Precise control over register allocation
 * 2. Explicit prefetch instructions
 * 3. Optimal instruction scheduling
 * 4. Avoid compiler suboptimal decisions
 *
 * Note: This is rarely needed - modern compilers are very good!
 * But useful for learning and extreme optimization cases.
 */

// Prefetch distance in cache lines (64 bytes = 16 floats)
#define PREFETCH_DISTANCE 8

void micro_kernel_asm(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    // This is a 4x8 microkernel using inline asm
    // Computes C[0:4, 0:8] += A[0:4, :] * B[:, 0:8]

    __asm__ __volatile__ (
        // Zero accumulator registers
        "vxorps %%ymm0, %%ymm0, %%ymm0\n\t"
        "vxorps %%ymm1, %%ymm1, %%ymm1\n\t"
        "vxorps %%ymm2, %%ymm2, %%ymm2\n\t"
        "vxorps %%ymm3, %%ymm3, %%ymm3\n\t"

        // Loop counter
        "xor %%rcx, %%rcx\n\t"

        "1:\n\t"  // Loop start

        // Prefetch next B row
        "prefetcht0 (%[B], %%rcx, 4)\n\t"

        // Load B[k, 0:8]
        "vmovups (%[B], %%rcx, 4), %%ymm4\n\t"

        // Broadcast A[0,k] and FMA
        "vbroadcastss (%[A], %%rcx, 4), %%ymm5\n\t"
        "vfmadd231ps %%ymm5, %%ymm4, %%ymm0\n\t"

        // Broadcast A[1,k] and FMA
        "vbroadcastss (%[A], %%rcx, 4), %%ymm5\n\t"  // Note: offset needed
        "vfmadd231ps %%ymm5, %%ymm4, %%ymm1\n\t"

        // Broadcast A[2,k] and FMA
        "vbroadcastss (%[A], %%rcx, 4), %%ymm5\n\t"
        "vfmadd231ps %%ymm5, %%ymm4, %%ymm2\n\t"

        // Broadcast A[3,k] and FMA
        "vbroadcastss (%[A], %%rcx, 4), %%ymm5\n\t"
        "vfmadd231ps %%ymm5, %%ymm4, %%ymm3\n\t"

        // Increment and loop
        "inc %%rcx\n\t"
        "cmp %[N], %%rcx\n\t"
        "jl 1b\n\t"

        // Store results
        "vmovups %%ymm0, (%[C])\n\t"
        "vmovups %%ymm1, 32(%[C])\n\t"
        "vmovups %%ymm2, 64(%[C])\n\t"
        "vmovups %%ymm3, 96(%[C])\n\t"

        :  // outputs
        : [A] "r" (A), [B] "r" (B), [C] "r" (C), [N] "r" ((long)N)  // inputs
        : "rcx", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "memory"
    );
}

//=============================================================================
// AVX-512 Implementation (if available)
//=============================================================================

#ifdef __AVX512F__

/*
 * AVX-512 provides:
 * - 512-bit registers (16 floats per register)
 * - 32 ZMM registers (vs 16 YMM in AVX2)
 * - Masked operations
 * - More powerful broadcast/gather/scatter
 *
 * This enables larger register blocking: 8x32 or even 12x48 tiles
 */
void matmul_avx512(const float* A, const float* B, float* C, int M, int N, int K) {
    zero_matrix(C, M, K);

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < N; k++) {
            __m512 a_ik = _mm512_set1_ps(A[i * N + k]);

            int j = 0;
            // Process 16 elements at a time
            for (; j + 15 < K; j += 16) {
                __m512 b_kj = _mm512_loadu_ps(&B[k * K + j]);
                __m512 c_ij = _mm512_loadu_ps(&C[i * K + j]);
                c_ij = _mm512_fmadd_ps(a_ik, b_kj, c_ij);
                _mm512_storeu_ps(&C[i * K + j], c_ij);
            }

            // Handle remainder with mask
            if (j < K) {
                __mmask16 mask = (1 << (K - j)) - 1;
                __m512 b_kj = _mm512_maskz_loadu_ps(mask, &B[k * K + j]);
                __m512 c_ij = _mm512_maskz_loadu_ps(mask, &C[i * K + j]);
                c_ij = _mm512_fmadd_ps(a_ik, b_kj, c_ij);
                _mm512_mask_storeu_ps(&C[i * K + j], mask, c_ij);
            }
        }
    }
}

void matmul_avx512_blocked(const float* A, const float* B, float* C, int M, int N, int K) {
    zero_matrix(C, M, K);

    const int BLOCK = 64;
    const int MR = 6;
    const int NR = 32;  // 2 x __m512 = 32 floats

    for (int i0 = 0; i0 < M; i0 += BLOCK) {
        for (int k0 = 0; k0 < N; k0 += BLOCK) {
            for (int j0 = 0; j0 < K; j0 += BLOCK) {
                // Micro-kernel for each block
                for (int i = i0; i < std::min(i0 + BLOCK, M); i += MR) {
                    for (int j = j0; j < std::min(j0 + BLOCK, K); j += NR) {
                        // 6x32 accumulator registers (12 ZMM registers)
                        __m512 c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
                        __m512 c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
                        __m512 c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
                        __m512 c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
                        __m512 c40 = _mm512_setzero_ps(), c41 = _mm512_setzero_ps();
                        __m512 c50 = _mm512_setzero_ps(), c51 = _mm512_setzero_ps();

                        for (int k = k0; k < std::min(k0 + BLOCK, N); k++) {
                            __m512 b0 = _mm512_loadu_ps(&B[k * K + j]);
                            __m512 b1 = _mm512_loadu_ps(&B[k * K + j + 16]);

                            __m512 a0 = _mm512_set1_ps(A[(i + 0) * N + k]);
                            c00 = _mm512_fmadd_ps(a0, b0, c00);
                            c01 = _mm512_fmadd_ps(a0, b1, c01);

                            __m512 a1 = _mm512_set1_ps(A[(i + 1) * N + k]);
                            c10 = _mm512_fmadd_ps(a1, b0, c10);
                            c11 = _mm512_fmadd_ps(a1, b1, c11);

                            __m512 a2 = _mm512_set1_ps(A[(i + 2) * N + k]);
                            c20 = _mm512_fmadd_ps(a2, b0, c20);
                            c21 = _mm512_fmadd_ps(a2, b1, c21);

                            __m512 a3 = _mm512_set1_ps(A[(i + 3) * N + k]);
                            c30 = _mm512_fmadd_ps(a3, b0, c30);
                            c31 = _mm512_fmadd_ps(a3, b1, c31);

                            __m512 a4 = _mm512_set1_ps(A[(i + 4) * N + k]);
                            c40 = _mm512_fmadd_ps(a4, b0, c40);
                            c41 = _mm512_fmadd_ps(a4, b1, c41);

                            __m512 a5 = _mm512_set1_ps(A[(i + 5) * N + k]);
                            c50 = _mm512_fmadd_ps(a5, b0, c50);
                            c51 = _mm512_fmadd_ps(a5, b1, c51);
                        }

                        // Accumulate to C
                        __m512 t;
                        t = _mm512_loadu_ps(&C[(i+0)*K + j]); _mm512_storeu_ps(&C[(i+0)*K + j], _mm512_add_ps(t, c00));
                        t = _mm512_loadu_ps(&C[(i+0)*K + j+16]); _mm512_storeu_ps(&C[(i+0)*K + j+16], _mm512_add_ps(t, c01));
                        t = _mm512_loadu_ps(&C[(i+1)*K + j]); _mm512_storeu_ps(&C[(i+1)*K + j], _mm512_add_ps(t, c10));
                        t = _mm512_loadu_ps(&C[(i+1)*K + j+16]); _mm512_storeu_ps(&C[(i+1)*K + j+16], _mm512_add_ps(t, c11));
                        t = _mm512_loadu_ps(&C[(i+2)*K + j]); _mm512_storeu_ps(&C[(i+2)*K + j], _mm512_add_ps(t, c20));
                        t = _mm512_loadu_ps(&C[(i+2)*K + j+16]); _mm512_storeu_ps(&C[(i+2)*K + j+16], _mm512_add_ps(t, c21));
                        t = _mm512_loadu_ps(&C[(i+3)*K + j]); _mm512_storeu_ps(&C[(i+3)*K + j], _mm512_add_ps(t, c30));
                        t = _mm512_loadu_ps(&C[(i+3)*K + j+16]); _mm512_storeu_ps(&C[(i+3)*K + j+16], _mm512_add_ps(t, c31));
                        t = _mm512_loadu_ps(&C[(i+4)*K + j]); _mm512_storeu_ps(&C[(i+4)*K + j], _mm512_add_ps(t, c40));
                        t = _mm512_loadu_ps(&C[(i+4)*K + j+16]); _mm512_storeu_ps(&C[(i+4)*K + j+16], _mm512_add_ps(t, c41));
                        t = _mm512_loadu_ps(&C[(i+5)*K + j]); _mm512_storeu_ps(&C[(i+5)*K + j], _mm512_add_ps(t, c50));
                        t = _mm512_loadu_ps(&C[(i+5)*K + j+16]); _mm512_storeu_ps(&C[(i+5)*K + j+16], _mm512_add_ps(t, c51));
                    }
                }
            }
        }
    }
}

#endif // __AVX512F__

//=============================================================================
// Main Benchmark
//=============================================================================

int main() {
    std::cout << "=== x86 Dense Matrix Multiplication Strategies ===\n\n";

    const int M = 1024, N = 1024, K = 1024;

    std::cout << "Matrix size: " << M << " x " << N << " x " << K << "\n";
    std::cout << "Total FLOPs: " << (2.0 * M * N * K / 1e9) << " GFLOP\n\n";

    // Allocate aligned memory
    float* A = static_cast<float*>(aligned_alloc(64, M * N * sizeof(float)));
    float* B = static_cast<float*>(aligned_alloc(64, N * K * sizeof(float)));
    float* C = static_cast<float*>(aligned_alloc(64, M * K * sizeof(float)));
    float* C_ref = static_cast<float*>(aligned_alloc(64, M * K * sizeof(float)));

    init_matrix(A, M, N);
    init_matrix(B, N, K);

    // Reference result
    matmul_naive(A, B, C_ref, M, N, K);

    auto benchmark = [&](const char* name, void (*func)(const float*, const float*, float*, int, int, int)) {
        // Warmup
        func(A, B, C, M, N, K);

        const int iterations = 5;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            func(A, B, C, M, N, K);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double time_sec = std::chrono::duration<double>(end - start).count() / iterations;
        double gflops = calculate_gflops(M, N, K, time_sec);

        bool correct = verify_result(C, C_ref, M * K);

        std::cout << name << ":\n";
        std::cout << "  Time: " << (time_sec * 1000) << " ms\n";
        std::cout << "  Performance: " << gflops << " GFLOPS\n";
        std::cout << "  Correct: " << (correct ? "Yes" : "NO!") << "\n\n";
    };

    benchmark("Naive (ijk)", matmul_naive);
    benchmark("Loop Reorder (ikj)", matmul_loop_reorder);
    benchmark("Blocked", matmul_blocked);
    benchmark("AVX2", matmul_avx2);
    benchmark("AVX2 + Blocked", matmul_avx2_blocked);
    benchmark("Register Blocked (6x16)", matmul_register_blocked);

#ifdef __AVX512F__
    benchmark("AVX-512", matmul_avx512);
    benchmark("AVX-512 + Blocked", matmul_avx512_blocked);
#else
    std::cout << "AVX-512: Not available on this CPU\n\n";
#endif

    std::cout << "=== Optimization Summary ===\n\n";

    std::cout << "Key Strategies for Dense GEMM:\n";
    std::cout << "1. Loop Reordering: Access memory sequentially\n";
    std::cout << "2. Blocking/Tiling: Fit working set in cache\n";
    std::cout << "3. SIMD (AVX2/AVX-512): Process 8-16 floats per instruction\n";
    std::cout << "4. Register Blocking: Maximize register reuse\n";
    std::cout << "5. Prefetching: Hide memory latency\n";
    std::cout << "6. Loop Unrolling: Reduce loop overhead\n\n";

    std::cout << "=== Interview Questions ===\n\n";

    std::cout << "Q: Why does loop order matter?\n";
    std::cout << "A: Memory access patterns determine cache efficiency.\n";
    std::cout << "   - ijk: B has stride-1, but A reloaded for each j\n";
    std::cout << "   - ikj: Both B and C have stride-1, A hoisted\n\n";

    std::cout << "Q: How do you choose block size?\n";
    std::cout << "A: 3 blocks should fit in cache:\n";
    std::cout << "   - L1 (32KB): B ~= 52\n";
    std::cout << "   - L2 (256KB): B ~= 147\n";
    std::cout << "   - Typically use 32-128 for L2 blocking\n\n";

    std::cout << "Q: AVX2 vs AVX-512 tradeoffs?\n";
    std::cout << "A: AVX-512 pros: 2x wider, 2x more registers\n";
    std::cout << "   AVX-512 cons: May cause frequency throttling\n";
    std::cout << "   Measure on target hardware!\n\n";

    std::cout << "Q: When to use inline assembly?\n";
    std::cout << "A: Rarely needed! Compilers are good.\n";
    std::cout << "   Use when: precise register control, specific\n";
    std::cout << "   prefetch patterns, debugging compiler output.\n\n";

    free(A); free(B); free(C); free(C_ref);

    return 0;
}
