/**
 * x86 Sparse Matrix Multiplication Optimization Strategies
 *
 * Sparse matrices have fundamentally different optimization strategies than dense:
 * - Storage format matters: CSR, CSC, COO, BCSR, ELL, etc.
 * - Indirect addressing kills vectorization
 * - Memory bandwidth bound (low arithmetic intensity)
 * - Irregular access patterns hurt cache
 *
 * This file covers:
 * 1. Sparse storage formats
 * 2. SpMV (Sparse Matrix-Vector) optimizations
 * 3. SpMM (Sparse Matrix-Dense Matrix) optimizations
 * 4. SIMD strategies for sparse
 * 5. Block sparse formats
 *
 * Compile: g++ -O3 -march=native -mavx2 -fopenmp sparse_matmul.cpp
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include <omp.h>

//=============================================================================
// Sparse Matrix Formats
//=============================================================================

/*
 * CSR (Compressed Sparse Row) - Most common format
 *
 * For matrix:
 *   [1 0 2]
 *   [0 3 0]
 *   [4 0 5]
 *
 * values:     [1, 2, 3, 4, 5]          - non-zero values
 * col_idx:    [0, 2, 1, 0, 2]          - column indices
 * row_ptr:    [0, 2, 3, 5]             - row start pointers
 *
 * Pros: Efficient row access, good for SpMV
 * Cons: Inserting elements is O(nnz), column access is expensive
 */
struct CSRMatrix {
    std::vector<float> values;
    std::vector<int> col_idx;
    std::vector<int> row_ptr;
    int rows, cols, nnz;

    CSRMatrix() : rows(0), cols(0), nnz(0) {}

    void from_dense(const float* dense, int m, int n, float threshold = 0.0f) {
        rows = m;
        cols = n;
        row_ptr.resize(m + 1);
        row_ptr[0] = 0;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float v = dense[i * n + j];
                if (fabs(v) > threshold) {
                    values.push_back(v);
                    col_idx.push_back(j);
                }
            }
            row_ptr[i + 1] = values.size();
        }
        nnz = values.size();
    }
};

/*
 * CSC (Compressed Sparse Column)
 *
 * Same as CSR but column-major. Useful for SpMV with column vector.
 */
struct CSCMatrix {
    std::vector<float> values;
    std::vector<int> row_idx;
    std::vector<int> col_ptr;
    int rows, cols, nnz;

    void from_csr(const CSRMatrix& csr) {
        rows = csr.rows;
        cols = csr.cols;
        nnz = csr.nnz;

        col_ptr.resize(cols + 1, 0);
        values.resize(nnz);
        row_idx.resize(nnz);

        // Count elements per column
        for (int i = 0; i < nnz; i++) {
            col_ptr[csr.col_idx[i] + 1]++;
        }

        // Cumulative sum
        for (int j = 1; j <= cols; j++) {
            col_ptr[j] += col_ptr[j - 1];
        }

        // Fill in values
        std::vector<int> next(cols);
        std::copy(col_ptr.begin(), col_ptr.end() - 1, next.begin());

        for (int i = 0; i < rows; i++) {
            for (int k = csr.row_ptr[i]; k < csr.row_ptr[i + 1]; k++) {
                int j = csr.col_idx[k];
                int dest = next[j]++;
                values[dest] = csr.values[k];
                row_idx[dest] = i;
            }
        }
    }
};

/*
 * BCSR (Block Compressed Sparse Row)
 *
 * Groups non-zeros into dense blocks (e.g., 4x4).
 * Much better for SIMD - can vectorize within blocks!
 *
 * Pros: SIMD-friendly, reduced indexing overhead
 * Cons: May have zeros within blocks (fill-in)
 */
struct BCSRMatrix {
    std::vector<float> values;      // Block values (row-major within block)
    std::vector<int> col_idx;       // Block column indices
    std::vector<int> row_ptr;       // Block row pointers
    int rows, cols, nnz;            // Original dimensions
    int block_rows, block_cols;     // Number of blocks
    int br, bc;                     // Block size

    void from_dense(const float* dense, int m, int n, int block_r, int block_c) {
        br = block_r;
        bc = block_c;
        rows = m;
        cols = n;
        block_rows = (m + br - 1) / br;
        block_cols = (n + bc - 1) / bc;

        row_ptr.resize(block_rows + 1);
        row_ptr[0] = 0;

        for (int bi = 0; bi < block_rows; bi++) {
            for (int bj = 0; bj < block_cols; bj++) {
                // Check if block has any non-zeros
                bool has_nonzero = false;
                for (int i = 0; i < br && !has_nonzero; i++) {
                    for (int j = 0; j < bc && !has_nonzero; j++) {
                        int gi = bi * br + i;
                        int gj = bj * bc + j;
                        if (gi < m && gj < n && dense[gi * n + gj] != 0.0f) {
                            has_nonzero = true;
                        }
                    }
                }

                if (has_nonzero) {
                    col_idx.push_back(bj);
                    // Store block values
                    for (int i = 0; i < br; i++) {
                        for (int j = 0; j < bc; j++) {
                            int gi = bi * br + i;
                            int gj = bj * bc + j;
                            float v = (gi < m && gj < n) ? dense[gi * n + gj] : 0.0f;
                            values.push_back(v);
                        }
                    }
                }
            }
            row_ptr[bi + 1] = col_idx.size();
        }
        nnz = values.size() / (br * bc);
    }
};

/*
 * ELL (ELLPACK) Format
 *
 * Pads each row to same length (max nnz per row).
 * Stored in column-major for coalesced GPU access.
 *
 * Pros: Regular access pattern, good for GPUs
 * Cons: Wasted space if row lengths vary significantly
 */
struct ELLMatrix {
    std::vector<float> values;      // Column-major, padded
    std::vector<int> col_idx;       // Column-major, padded
    int rows, cols;
    int max_nnz_per_row;

    void from_csr(const CSRMatrix& csr) {
        rows = csr.rows;
        cols = csr.cols;

        // Find max nnz per row
        max_nnz_per_row = 0;
        for (int i = 0; i < rows; i++) {
            int row_nnz = csr.row_ptr[i + 1] - csr.row_ptr[i];
            max_nnz_per_row = std::max(max_nnz_per_row, row_nnz);
        }

        // Allocate padded arrays (column-major)
        values.resize(rows * max_nnz_per_row, 0.0f);
        col_idx.resize(rows * max_nnz_per_row, 0);

        // Fill in values
        for (int i = 0; i < rows; i++) {
            int k = 0;
            for (int j = csr.row_ptr[i]; j < csr.row_ptr[i + 1]; j++, k++) {
                values[k * rows + i] = csr.values[j];
                col_idx[k * rows + i] = csr.col_idx[j];
            }
        }
    }
};

//=============================================================================
// SpMV: Sparse Matrix-Vector Multiplication
//=============================================================================

/*
 * Naive CSR SpMV: y = A * x
 *
 * For each row:
 *   y[i] = sum_j A[i,j] * x[j]
 *
 * Problem: Indirect indexing (x[col_idx[k]]) prevents vectorization
 */
void spmv_csr_naive(const CSRMatrix& A, const float* x, float* y) {
    for (int i = 0; i < A.rows; i++) {
        float sum = 0.0f;
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; k++) {
            sum += A.values[k] * x[A.col_idx[k]];
        }
        y[i] = sum;
    }
}

/*
 * CSR SpMV with OpenMP parallelization
 *
 * Each row is independent - embarrassingly parallel!
 */
void spmv_csr_omp(const CSRMatrix& A, const float* x, float* y) {
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < A.rows; i++) {
        float sum = 0.0f;
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; k++) {
            sum += A.values[k] * x[A.col_idx[k]];
        }
        y[i] = sum;
    }
}

/*
 * CSR SpMV with manual unrolling
 *
 * Unroll inner loop to reduce loop overhead and enable ILP.
 */
void spmv_csr_unrolled(const CSRMatrix& A, const float* x, float* y) {
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < A.rows; i++) {
        int start = A.row_ptr[i];
        int end = A.row_ptr[i + 1];
        int len = end - start;

        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

        int k = start;
        // Unroll by 4
        for (; k + 3 < end; k += 4) {
            sum0 += A.values[k + 0] * x[A.col_idx[k + 0]];
            sum1 += A.values[k + 1] * x[A.col_idx[k + 1]];
            sum2 += A.values[k + 2] * x[A.col_idx[k + 2]];
            sum3 += A.values[k + 3] * x[A.col_idx[k + 3]];
        }

        // Remainder
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; k < end; k++) {
            sum += A.values[k] * x[A.col_idx[k]];
        }

        y[i] = sum;
    }
}

/*
 * CSC SpMV (column-wise): y = A * x
 *
 * For each column:
 *   y += A[:,j] * x[j]
 *
 * Different access pattern - may be better for some matrices.
 */
void spmv_csc(const CSCMatrix& A, const float* x, float* y) {
    memset(y, 0, A.rows * sizeof(float));

    for (int j = 0; j < A.cols; j++) {
        float xj = x[j];
        for (int k = A.col_ptr[j]; k < A.col_ptr[j + 1]; k++) {
            y[A.row_idx[k]] += A.values[k] * xj;
        }
    }
}

/*
 * BCSR SpMV - Vectorizable within blocks!
 *
 * For 4x4 blocks, we can use AVX to process the block.
 */
void spmv_bcsr_4x4(const BCSRMatrix& A, const float* x, float* y) {
    memset(y, 0, A.rows * sizeof(float));

    for (int bi = 0; bi < A.block_rows; bi++) {
        int row_start = bi * 4;

        // Accumulator for this block row
        __m128 y_acc = _mm_setzero_ps();

        for (int bk = A.row_ptr[bi]; bk < A.row_ptr[bi + 1]; bk++) {
            int bj = A.col_idx[bk];
            int col_start = bj * 4;

            // Load x vector chunk (4 elements)
            __m128 x_vec = _mm_loadu_ps(&x[col_start]);

            // Load block (4x4 = 16 floats)
            const float* block = &A.values[bk * 16];

            // Multiply each row of block by x_vec and accumulate
            // Row 0
            __m128 row0 = _mm_loadu_ps(&block[0]);
            __m128 prod0 = _mm_mul_ps(row0, x_vec);
            // Horizontal sum
            prod0 = _mm_hadd_ps(prod0, prod0);
            prod0 = _mm_hadd_ps(prod0, prod0);

            // Row 1
            __m128 row1 = _mm_loadu_ps(&block[4]);
            __m128 prod1 = _mm_mul_ps(row1, x_vec);
            prod1 = _mm_hadd_ps(prod1, prod1);
            prod1 = _mm_hadd_ps(prod1, prod1);

            // Row 2
            __m128 row2 = _mm_loadu_ps(&block[8]);
            __m128 prod2 = _mm_mul_ps(row2, x_vec);
            prod2 = _mm_hadd_ps(prod2, prod2);
            prod2 = _mm_hadd_ps(prod2, prod2);

            // Row 3
            __m128 row3 = _mm_loadu_ps(&block[12]);
            __m128 prod3 = _mm_mul_ps(row3, x_vec);
            prod3 = _mm_hadd_ps(prod3, prod3);
            prod3 = _mm_hadd_ps(prod3, prod3);

            // Combine results
            __m128 result = _mm_set_ps(
                _mm_cvtss_f32(prod3),
                _mm_cvtss_f32(prod2),
                _mm_cvtss_f32(prod1),
                _mm_cvtss_f32(prod0)
            );

            y_acc = _mm_add_ps(y_acc, result);
        }

        // Store accumulated result
        if (row_start + 4 <= A.rows) {
            _mm_storeu_ps(&y[row_start], y_acc);
        } else {
            // Handle boundary
            float temp[4];
            _mm_storeu_ps(temp, y_acc);
            for (int i = 0; i < A.rows - row_start; i++) {
                y[row_start + i] = temp[i];
            }
        }
    }
}

/*
 * ELL SpMV - Regular access pattern, SIMD-friendly
 *
 * Process multiple rows simultaneously with SIMD.
 */
void spmv_ell_simd(const ELLMatrix& A, const float* x, float* y) {
    // Process 8 rows at a time with AVX
    int i = 0;
    for (; i + 7 < A.rows; i += 8) {
        __m256 sum = _mm256_setzero_ps();

        for (int k = 0; k < A.max_nnz_per_row; k++) {
            // Load 8 values (contiguous in column-major)
            __m256 vals = _mm256_loadu_ps(&A.values[k * A.rows + i]);

            // Gather 8 x values (indirect access)
            // Note: Gather is expensive! This is where ELL loses to CSR
            __m256i indices = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(&A.col_idx[k * A.rows + i]));
            __m256 x_vals = _mm256_i32gather_ps(x, indices, 4);

            sum = _mm256_fmadd_ps(vals, x_vals, sum);
        }

        _mm256_storeu_ps(&y[i], sum);
    }

    // Remainder
    for (; i < A.rows; i++) {
        float sum = 0.0f;
        for (int k = 0; k < A.max_nnz_per_row; k++) {
            int idx = k * A.rows + i;
            sum += A.values[idx] * x[A.col_idx[idx]];
        }
        y[i] = sum;
    }
}

//=============================================================================
// SpMM: Sparse Matrix x Dense Matrix
//=============================================================================

/*
 * CSR SpMM: C = A * B where A is sparse, B is dense
 *
 * C[i,j] = sum_k A[i,k] * B[k,j]
 *
 * Can vectorize over the j dimension (dense columns)!
 */
void spmm_csr_dense(const CSRMatrix& A, const float* B, float* C,
                    int B_cols) {
    memset(C, 0, A.rows * B_cols * sizeof(float));

    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < A.rows; i++) {
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; k++) {
            float a_ik = A.values[k];
            int col_k = A.col_idx[k];

            // Vectorized over B columns
            int j = 0;
            __m256 a_vec = _mm256_set1_ps(a_ik);

            for (; j + 7 < B_cols; j += 8) {
                __m256 b_kj = _mm256_loadu_ps(&B[col_k * B_cols + j]);
                __m256 c_ij = _mm256_loadu_ps(&C[i * B_cols + j]);
                c_ij = _mm256_fmadd_ps(a_vec, b_kj, c_ij);
                _mm256_storeu_ps(&C[i * B_cols + j], c_ij);
            }

            // Remainder
            for (; j < B_cols; j++) {
                C[i * B_cols + j] += a_ik * B[col_k * B_cols + j];
            }
        }
    }
}

/*
 * Row-major SpMM with register blocking
 *
 * Process multiple B columns at once for better register usage.
 */
void spmm_csr_blocked(const CSRMatrix& A, const float* B, float* C,
                      int B_cols) {
    const int NR = 32;  // Process 32 B columns per iteration

    memset(C, 0, A.rows * B_cols * sizeof(float));

    #pragma omp parallel for schedule(dynamic, 8)
    for (int i = 0; i < A.rows; i++) {
        for (int j0 = 0; j0 < B_cols; j0 += NR) {
            int j_max = std::min(j0 + NR, B_cols);

            // Accumulators
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();

            for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; k++) {
                float a_ik = A.values[k];
                int col_k = A.col_idx[k];
                __m256 a_vec = _mm256_set1_ps(a_ik);

                if (j_max - j0 >= 32) {
                    __m256 b0 = _mm256_loadu_ps(&B[col_k * B_cols + j0]);
                    __m256 b1 = _mm256_loadu_ps(&B[col_k * B_cols + j0 + 8]);
                    __m256 b2 = _mm256_loadu_ps(&B[col_k * B_cols + j0 + 16]);
                    __m256 b3 = _mm256_loadu_ps(&B[col_k * B_cols + j0 + 24]);

                    c0 = _mm256_fmadd_ps(a_vec, b0, c0);
                    c1 = _mm256_fmadd_ps(a_vec, b1, c1);
                    c2 = _mm256_fmadd_ps(a_vec, b2, c2);
                    c3 = _mm256_fmadd_ps(a_vec, b3, c3);
                }
            }

            if (j_max - j0 >= 32) {
                _mm256_storeu_ps(&C[i * B_cols + j0], c0);
                _mm256_storeu_ps(&C[i * B_cols + j0 + 8], c1);
                _mm256_storeu_ps(&C[i * B_cols + j0 + 16], c2);
                _mm256_storeu_ps(&C[i * B_cols + j0 + 24], c3);
            }
        }
    }
}

//=============================================================================
// Utilities
//=============================================================================

void generate_sparse_matrix(float* M, int rows, int cols, float sparsity) {
    for (int i = 0; i < rows * cols; i++) {
        if (static_cast<float>(rand()) / RAND_MAX > sparsity) {
            M[i] = static_cast<float>(rand()) / RAND_MAX;
        } else {
            M[i] = 0.0f;
        }
    }
}

//=============================================================================
// Main Benchmark
//=============================================================================

int main() {
    std::cout << "=== x86 Sparse Matrix Optimization Strategies ===\n\n";

    const int M = 4096;
    const int N = 4096;
    const float sparsity = 0.95f;  // 95% zeros

    std::cout << "Matrix size: " << M << " x " << N << "\n";
    std::cout << "Sparsity: " << (sparsity * 100) << "% zeros\n";
    std::cout << "Expected nnz: ~" << static_cast<int>(M * N * (1 - sparsity)) << "\n\n";

    // Generate sparse matrix
    float* dense = static_cast<float*>(aligned_alloc(64, M * N * sizeof(float)));
    generate_sparse_matrix(dense, M, N, sparsity);

    // Create different sparse formats
    CSRMatrix csr;
    csr.from_dense(dense, M, N);
    std::cout << "Actual nnz: " << csr.nnz << " (" 
              << (100.0f * csr.nnz / (M * N)) << "% density)\n\n";

    CSCMatrix csc;
    csc.from_csr(csr);

    BCSRMatrix bcsr;
    bcsr.from_dense(dense, M, N, 4, 4);

    ELLMatrix ell;
    ell.from_csr(csr);

    // Allocate vectors
    float* x = static_cast<float*>(aligned_alloc(64, N * sizeof(float)));
    float* y = static_cast<float*>(aligned_alloc(64, M * sizeof(float)));
    float* y_ref = static_cast<float*>(aligned_alloc(64, M * sizeof(float)));

    for (int i = 0; i < N; i++) {
        x[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Reference result
    spmv_csr_naive(csr, x, y_ref);

    auto benchmark_spmv = [&](const char* name, auto func) {
        // Warmup
        func();

        const int iterations = 100;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            func();
        }
        auto end = std::chrono::high_resolution_clock::now();

        double time_sec = std::chrono::duration<double>(end - start).count() / iterations;
        double gflops = (2.0 * csr.nnz / time_sec) / 1e9;
        double bandwidth = ((csr.nnz * 12 + M * 4 + N * 4) / time_sec) / 1e9;

        bool correct = true;
        for (int i = 0; i < M && correct; i++) {
            if (fabs(y[i] - y_ref[i]) > 1e-3f) correct = false;
        }

        std::cout << name << ":\n";
        std::cout << "  Time: " << (time_sec * 1e6) << " us\n";
        std::cout << "  GFLOPS: " << gflops << "\n";
        std::cout << "  Bandwidth: " << bandwidth << " GB/s\n";
        std::cout << "  Correct: " << (correct ? "Yes" : "NO!") << "\n\n";
    };

    std::cout << "=== SpMV Benchmarks ===\n\n";

    benchmark_spmv("CSR Naive", [&]() { spmv_csr_naive(csr, x, y); });
    benchmark_spmv("CSR OpenMP", [&]() { spmv_csr_omp(csr, x, y); });
    benchmark_spmv("CSR Unrolled", [&]() { spmv_csr_unrolled(csr, x, y); });
    benchmark_spmv("CSC", [&]() { spmv_csc(csc, x, y); });
    benchmark_spmv("BCSR 4x4", [&]() { spmv_bcsr_4x4(bcsr, x, y); });
    benchmark_spmv("ELL SIMD", [&]() { spmv_ell_simd(ell, x, y); });

    //=========================================================================
    // Comparison: Dense vs Sparse
    //=========================================================================

    std::cout << "=== Dense vs Sparse Strategies ===\n\n";

    std::cout << "+------------------+------------------------+------------------------+\n";
    std::cout << "| Aspect           | Dense Matrix           | Sparse Matrix          |\n";
    std::cout << "+------------------+------------------------+------------------------+\n";
    std::cout << "| Storage          | O(n^2)                 | O(nnz)                 |\n";
    std::cout << "| Access Pattern   | Regular, predictable   | Irregular, indirect    |\n";
    std::cout << "| SIMD             | Trivial to vectorize   | Difficult (gathers)    |\n";
    std::cout << "| Cache Usage      | Streaming, efficient   | Random access, poor    |\n";
    std::cout << "| Parallelization  | Easy (tiles)           | Load balancing issues  |\n";
    std::cout << "| Bottleneck       | Compute-bound          | Memory-bound           |\n";
    std::cout << "| Key Optimization | Blocking, FMA          | Format selection       |\n";
    std::cout << "+------------------+------------------------+------------------------+\n\n";

    std::cout << "=== When to Use Each Format ===\n\n";

    std::cout << "CSR (Compressed Sparse Row):\n";
    std::cout << "  - General purpose, most common\n";
    std::cout << "  - Good for row-wise access (SpMV)\n";
    std::cout << "  - Supported by most libraries (MKL, cuSPARSE)\n\n";

    std::cout << "CSC (Compressed Sparse Column):\n";
    std::cout << "  - Good for column-wise access\n";
    std::cout << "  - Transpose of CSR\n";
    std::cout << "  - Useful for A^T * x operations\n\n";

    std::cout << "BCSR (Block CSR):\n";
    std::cout << "  - When matrix has natural block structure\n";
    std::cout << "  - Finite element matrices\n";
    std::cout << "  - Enables SIMD within blocks\n\n";

    std::cout << "ELL (ELLPACK):\n";
    std::cout << "  - Uniform row lengths\n";
    std::cout << "  - GPU-friendly (coalesced access)\n";
    std::cout << "  - Wastes space if row lengths vary\n\n";

    std::cout << "COO (Coordinate):\n";
    std::cout << "  - Construction and modification\n";
    std::cout << "  - Easy to build, convert to CSR/CSC\n";
    std::cout << "  - Poor performance for computation\n\n";

    std::cout << "=== Interview Questions ===\n\n";

    std::cout << "Q: Why is SpMV memory-bound while dense GEMM is compute-bound?\n";
    std::cout << "A: SpMV arithmetic intensity = 2 FLOP / (8+4) bytes = 0.17\n";
    std::cout << "   Dense GEMM with blocking can achieve AI > 10\n";
    std::cout << "   SpMV cannot reuse data - each element accessed once\n\n";

    std::cout << "Q: How to choose between CSR and BCSR?\n";
    std::cout << "A: Use BCSR when:\n";
    std::cout << "   - Matrix has natural block structure\n";
    std::cout << "   - Block density is reasonable (>50%)\n";
    std::cout << "   - Block size matches SIMD width\n";
    std::cout << "   CSR is default for unstructured matrices\n\n";

    std::cout << "Q: Why is gather expensive on x86?\n";
    std::cout << "A: _mm256_i32gather_ps has high latency (~12-20 cycles)\n";
    std::cout << "   Each element may come from different cache line\n";
    std::cout << "   Prefer structured access patterns when possible\n\n";

    std::cout << "Q: How to parallelize sparse operations?\n";
    std::cout << "A: Challenges:\n";
    std::cout << "   - Row lengths vary (load imbalance)\n";
    std::cout << "   - CSC has race conditions on y vector\n";
    std::cout << "   Solutions:\n";
    std::cout << "   - Dynamic scheduling for CSR\n";
    std::cout << "   - Row segmentation for better balance\n";
    std::cout << "   - Thread-local accumulators for CSC\n\n";

    free(dense); free(x); free(y); free(y_ref);

    return 0;
}
