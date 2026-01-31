#!/usr/bin/env python3
"""
Python Matrix Multiplication Optimization Strategies

This file demonstrates various optimization techniques for matrix multiplication
in Python, from naive triple loops to highly optimized NumPy/library calls.

Optimization Levels:
1. Naive triple loop (pure Python)
2. NumPy dot product (BLAS-optimized)
3. Tiled/blocked implementation
4. Numba JIT-compiled versions
5. CuPy GPU acceleration (optional)

Run: python matmul.py

Requirements:
    pip install numpy numba
    pip install cupy-cuda12x  # Optional, for GPU
"""

import numpy as np
import time
from typing import Callable, Tuple
import sys

# Try to import optional dependencies
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available. Install with: pip install numba")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


# =============================================================================
# Utility Functions
# =============================================================================

def init_matrices(M: int, N: int, K: int, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize random matrices A (MxN) and B (NxK)."""
    np.random.seed(42)
    A = np.random.randn(M, N).astype(dtype)
    B = np.random.randn(N, K).astype(dtype)
    return A, B


def calculate_gflops(M: int, N: int, K: int, time_sec: float) -> float:
    """Calculate GFLOPS for matrix multiply."""
    flops = 2.0 * M * N * K  # multiply-add = 2 ops
    return (flops / time_sec) / 1e9


def verify_result(C: np.ndarray, C_ref: np.ndarray, rtol: float = 1e-3) -> bool:
    """Verify result against reference."""
    return np.allclose(C, C_ref, rtol=rtol)


def benchmark(func: Callable, A: np.ndarray, B: np.ndarray,
              iterations: int = 5, warmup: int = 1) -> Tuple[np.ndarray, float]:
    """Benchmark a matrix multiply function."""
    # Warmup
    for _ in range(warmup):
        C = func(A, B)

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        C = func(A, B)
    end = time.perf_counter()

    avg_time = (end - start) / iterations
    return C, avg_time


# =============================================================================
# Level 1: Naive Triple Loop (Pure Python)
# =============================================================================

def matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Naive triple-loop matrix multiplication.

    C[i,j] = sum_k A[i,k] * B[k,j]

    Complexity: O(M * N * K)
    Performance: ~0.001 GFLOPS (extremely slow due to Python overhead)

    Why it's slow:
    - Python loop overhead (bytecode interpretation)
    - No vectorization
    - Poor cache utilization
    - No parallelism
    """
    M, N = A.shape
    N2, K = B.shape
    assert N == N2, "Matrix dimensions don't match"

    C = np.zeros((M, K), dtype=A.dtype)

    for i in range(M):
        for j in range(K):
            total = 0.0
            for k in range(N):
                total += A[i, k] * B[k, j]
            C[i, j] = total

    return C


# =============================================================================
# Level 2: NumPy (BLAS-Optimized)
# =============================================================================

def matmul_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    NumPy matrix multiplication using @ operator.

    Under the hood, this calls BLAS (Basic Linear Algebra Subprograms):
    - OpenBLAS (default on most systems)
    - Intel MKL (if installed)
    - Apple Accelerate (on macOS)

    Performance: ~100-500 GFLOPS (depends on BLAS implementation)

    Why it's fast:
    - Optimized C/Fortran code
    - SIMD vectorization (AVX2, AVX-512)
    - Cache-optimized blocking
    - Multi-threaded
    """
    return A @ B


def matmul_numpy_dot(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Alternative using np.dot (same performance as @)."""
    return np.dot(A, B)


def matmul_numpy_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Alternative using np.matmul (same performance as @)."""
    return np.matmul(A, B)


# =============================================================================
# Level 3: Tiled/Blocked (Pure Python - Educational)
# =============================================================================

def matmul_blocked(A: np.ndarray, B: np.ndarray, block_size: int = 64) -> np.ndarray:
    """
    Blocked/tiled matrix multiplication.

    Divides matrices into blocks that fit in L2 cache.
    Still slow in pure Python, but demonstrates the concept.

    Cache Analysis:
    - Block size B: 3 blocks of B x B floats should fit in cache
    - L2 cache (256KB): B ~= sqrt(256KB / 12) ~= 147
    - Typical choice: 32-128

    Performance: Still ~0.001 GFLOPS in pure Python
    (The blocking helps in compiled languages, not interpreted Python)
    """
    M, N = A.shape
    N2, K = B.shape
    assert N == N2

    C = np.zeros((M, K), dtype=A.dtype)

    # Block over all three dimensions
    for i0 in range(0, M, block_size):
        for k0 in range(0, N, block_size):
            for j0 in range(0, K, block_size):
                # Compute block
                i_end = min(i0 + block_size, M)
                k_end = min(k0 + block_size, N)
                j_end = min(j0 + block_size, K)

                for i in range(i0, i_end):
                    for k in range(k0, k_end):
                        a_ik = A[i, k]
                        for j in range(j0, j_end):
                            C[i, j] += a_ik * B[k, j]

    return C


def matmul_blocked_numpy(A: np.ndarray, B: np.ndarray, block_size: int = 256) -> np.ndarray:
    """
    Blocked matrix multiplication using NumPy for inner blocks.

    This combines Python-level blocking with NumPy's optimized operations.
    Useful when you need custom blocking behavior (e.g., for out-of-core computation).

    Performance: ~50-200 GFLOPS (overhead from Python blocking)
    """
    M, N = A.shape
    N2, K = B.shape
    assert N == N2

    C = np.zeros((M, K), dtype=A.dtype)

    for i0 in range(0, M, block_size):
        i_end = min(i0 + block_size, M)
        for j0 in range(0, K, block_size):
            j_end = min(j0 + block_size, K)
            for k0 in range(0, N, block_size):
                k_end = min(k0 + block_size, N)
                # Use NumPy for the inner block computation
                C[i0:i_end, j0:j_end] += A[i0:i_end, k0:k_end] @ B[k0:k_end, j0:j_end]

    return C


# =============================================================================
# Level 4: Numba JIT-Compiled
# =============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, fastmath=True)
    def matmul_numba_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Numba JIT-compiled naive matrix multiplication.

        Numba compiles Python to machine code via LLVM.
        - nopython=True: No Python objects, full optimization
        - fastmath=True: Allow reordering of floating-point ops

        Performance: ~5-20 GFLOPS (single-threaded, vectorized)
        """
        M, N = A.shape
        K = B.shape[1]
        C = np.zeros((M, K), dtype=A.dtype)

        for i in range(M):
            for k in range(N):
                a_ik = A[i, k]
                for j in range(K):
                    C[i, j] += a_ik * B[k, j]

        return C


    @jit(nopython=True, fastmath=True, parallel=True)
    def matmul_numba_parallel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Numba parallel matrix multiplication.

        Uses prange() for parallel outer loop.

        Performance: ~20-100 GFLOPS (multi-threaded)
        """
        M, N = A.shape
        K = B.shape[1]
        C = np.zeros((M, K), dtype=A.dtype)

        for i in prange(M):
            for k in range(N):
                a_ik = A[i, k]
                for j in range(K):
                    C[i, j] += a_ik * B[k, j]

        return C


    @jit(nopython=True, fastmath=True, parallel=True)
    def matmul_numba_blocked(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Numba parallel blocked matrix multiplication.

        Combines blocking with parallelism for better cache utilization.

        Performance: ~30-150 GFLOPS
        """
        M, N = A.shape
        K = B.shape[1]
        C = np.zeros((M, K), dtype=A.dtype)

        BLOCK = 64

        # Parallel over output row blocks
        for i0 in prange(0, M, BLOCK):
            for k0 in range(0, N, BLOCK):
                for j0 in range(0, K, BLOCK):
                    i_end = min(i0 + BLOCK, M)
                    k_end = min(k0 + BLOCK, N)
                    j_end = min(j0 + BLOCK, K)

                    for i in range(i0, i_end):
                        for k in range(k0, k_end):
                            a_ik = A[i, k]
                            for j in range(j0, j_end):
                                C[i, j] += a_ik * B[k, j]

        return C


# =============================================================================
# Level 5: CuPy GPU Acceleration
# =============================================================================

if CUPY_AVAILABLE:
    def matmul_cupy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        CuPy GPU-accelerated matrix multiplication.

        CuPy provides a NumPy-compatible interface for NVIDIA GPUs.
        Under the hood, it uses cuBLAS for matrix operations.

        Performance: ~1000-15000 GFLOPS (depending on GPU)

        Note: Includes data transfer overhead (host <-> device)
        For best performance, keep data on GPU.
        """
        # Transfer to GPU
        A_gpu = cp.asarray(A)
        B_gpu = cp.asarray(B)

        # Compute on GPU
        C_gpu = A_gpu @ B_gpu

        # Transfer back to CPU
        return cp.asnumpy(C_gpu)


    def matmul_cupy_no_transfer(A_gpu, B_gpu):
        """
        CuPy matrix multiply without data transfer.
        Use this when data is already on GPU.
        """
        return A_gpu @ B_gpu


# =============================================================================
# Main Benchmark
# =============================================================================

def main():
    print("=" * 60)
    print("Python Matrix Multiplication Optimization Strategies")
    print("=" * 60)
    print()

    # Matrix dimensions
    M, N, K = 512, 512, 512  # Smaller for naive methods
    M_large, N_large, K_large = 2048, 2048, 2048  # For optimized methods

    print(f"Small matrices: {M} x {N} x {K}")
    print(f"Large matrices: {M_large} x {N_large} x {K_large}")
    print(f"FLOPS (small): {2.0 * M * N * K / 1e9:.3f} GFLOP")
    print(f"FLOPS (large): {2.0 * M_large * N_large * K_large / 1e9:.3f} GFLOP")
    print()

    # Initialize matrices
    A_small, B_small = init_matrices(M, N, K)
    A_large, B_large = init_matrices(M_large, N_large, K_large)

    # Reference result
    C_ref_small = A_small @ B_small
    C_ref_large = A_large @ B_large

    results = []

    # ---------------------------------------------------------------------
    # Benchmark naive (only on small matrices)
    # ---------------------------------------------------------------------
    print("-" * 60)
    print("Level 1: Naive Triple Loop (Pure Python)")
    print("-" * 60)

    if M <= 256:  # Only run naive on very small matrices
        C, time_sec = benchmark(matmul_naive, A_small, B_small, iterations=1, warmup=0)
        gflops = calculate_gflops(M, N, K, time_sec)
        correct = verify_result(C, C_ref_small)
        print(f"  Time: {time_sec*1000:.1f} ms")
        print(f"  Performance: {gflops:.4f} GFLOPS")
        print(f"  Correct: {correct}")
        results.append(("Naive (Python)", gflops))
    else:
        print("  Skipped (too slow for this matrix size)")
    print()

    # ---------------------------------------------------------------------
    # Benchmark NumPy
    # ---------------------------------------------------------------------
    print("-" * 60)
    print("Level 2: NumPy (BLAS-Optimized)")
    print("-" * 60)

    C, time_sec = benchmark(matmul_numpy, A_large, B_large, iterations=10)
    gflops = calculate_gflops(M_large, N_large, K_large, time_sec)
    correct = verify_result(C, C_ref_large)
    print(f"  Time: {time_sec*1000:.2f} ms")
    print(f"  Performance: {gflops:.2f} GFLOPS")
    print(f"  Correct: {correct}")
    results.append(("NumPy (@)", gflops))

    # Check which BLAS is being used
    print(f"  NumPy config: {np.__config__.show() if hasattr(np.__config__, 'show') else 'N/A'}")
    print()

    # ---------------------------------------------------------------------
    # Benchmark blocked NumPy
    # ---------------------------------------------------------------------
    print("-" * 60)
    print("Level 3: Blocked with NumPy Inner Ops")
    print("-" * 60)

    C, time_sec = benchmark(lambda A, B: matmul_blocked_numpy(A, B, 256),
                            A_large, B_large, iterations=5)
    gflops = calculate_gflops(M_large, N_large, K_large, time_sec)
    correct = verify_result(C, C_ref_large)
    print(f"  Time: {time_sec*1000:.2f} ms")
    print(f"  Performance: {gflops:.2f} GFLOPS")
    print(f"  Correct: {correct}")
    results.append(("Blocked NumPy", gflops))
    print()

    # ---------------------------------------------------------------------
    # Benchmark Numba
    # ---------------------------------------------------------------------
    if NUMBA_AVAILABLE:
        print("-" * 60)
        print("Level 4: Numba JIT-Compiled")
        print("-" * 60)

        # Numba naive (on smaller matrix)
        C, time_sec = benchmark(matmul_numba_naive, A_small, B_small, iterations=5)
        gflops = calculate_gflops(M, N, K, time_sec)
        correct = verify_result(C, C_ref_small)
        print(f"  Numba Naive ({M}x{N}x{K}):")
        print(f"    Time: {time_sec*1000:.2f} ms")
        print(f"    Performance: {gflops:.2f} GFLOPS")
        print(f"    Correct: {correct}")
        results.append(("Numba Naive", gflops))

        # Numba parallel
        C, time_sec = benchmark(matmul_numba_parallel, A_large, B_large, iterations=5)
        gflops = calculate_gflops(M_large, N_large, K_large, time_sec)
        correct = verify_result(C, C_ref_large)
        print(f"  Numba Parallel ({M_large}x{N_large}x{K_large}):")
        print(f"    Time: {time_sec*1000:.2f} ms")
        print(f"    Performance: {gflops:.2f} GFLOPS")
        print(f"    Correct: {correct}")
        results.append(("Numba Parallel", gflops))

        # Numba blocked parallel
        C, time_sec = benchmark(matmul_numba_blocked, A_large, B_large, iterations=5)
        gflops = calculate_gflops(M_large, N_large, K_large, time_sec)
        correct = verify_result(C, C_ref_large)
        print(f"  Numba Blocked Parallel:")
        print(f"    Time: {time_sec*1000:.2f} ms")
        print(f"    Performance: {gflops:.2f} GFLOPS")
        print(f"    Correct: {correct}")
        results.append(("Numba Blocked", gflops))
        print()

    # ---------------------------------------------------------------------
    # Benchmark CuPy (GPU)
    # ---------------------------------------------------------------------
    if CUPY_AVAILABLE:
        print("-" * 60)
        print("Level 5: CuPy GPU (cuBLAS)")
        print("-" * 60)

        # With transfer
        C, time_sec = benchmark(matmul_cupy, A_large, B_large, iterations=10)
        gflops = calculate_gflops(M_large, N_large, K_large, time_sec)
        correct = verify_result(C, C_ref_large)
        print(f"  CuPy (with transfer):")
        print(f"    Time: {time_sec*1000:.2f} ms")
        print(f"    Performance: {gflops:.2f} GFLOPS")
        print(f"    Correct: {correct}")
        results.append(("CuPy (w/ transfer)", gflops))

        # Without transfer
        A_gpu = cp.asarray(A_large)
        B_gpu = cp.asarray(B_large)

        # Warmup
        _ = A_gpu @ B_gpu
        cp.cuda.Stream.null.synchronize()

        start = time.perf_counter()
        for _ in range(10):
            C_gpu = A_gpu @ B_gpu
        cp.cuda.Stream.null.synchronize()
        end = time.perf_counter()

        time_sec = (end - start) / 10
        gflops = calculate_gflops(M_large, N_large, K_large, time_sec)
        print(f"  CuPy (no transfer):")
        print(f"    Time: {time_sec*1000:.2f} ms")
        print(f"    Performance: {gflops:.2f} GFLOPS")
        results.append(("CuPy (no transfer)", gflops))
        print()

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------
    print("=" * 60)
    print("Performance Summary")
    print("=" * 60)
    print()
    print(f"{'Method':<25} {'GFLOPS':>10}")
    print("-" * 36)
    for name, gflops in results:
        print(f"{name:<25} {gflops:>10.2f}")
    print()

    # ---------------------------------------------------------------------
    # Interview Questions
    # ---------------------------------------------------------------------
    print("=" * 60)
    print("Interview Discussion Points")
    print("=" * 60)
    print("""
Q: Why is pure Python so slow for matrix multiply?
A: - Python is interpreted (bytecode overhead)
   - Dynamic typing (type checks at runtime)
   - No SIMD vectorization
   - GIL prevents true parallelism

Q: How does NumPy achieve high performance?
A: - Calls optimized BLAS libraries (OpenBLAS, MKL)
   - Written in C/Fortran with SIMD intrinsics
   - Cache-optimized blocking algorithms
   - Multi-threaded (OpenMP)

Q: When would you use Numba over NumPy?
A: - Custom algorithms not expressible as NumPy ops
   - Avoiding temporary arrays in complex expressions
   - GPU kernels with @cuda.jit
   - When NumPy's overhead matters (small arrays)

Q: What's the difference between @ and np.dot?
A: - @ (matmul): Matrix multiplication, broadcast rules
   - np.dot: Inner product, different broadcast for >2D
   - For 2D arrays, they're equivalent

Q: How to choose between CPU and GPU for GEMM?
A: - Small matrices (<512): CPU (transfer overhead dominates)
   - Large matrices (>1024): GPU usually faster
   - Batch operations: Keep data on GPU, amortize transfer
   - Memory-bound ops: May not benefit from GPU
""")


if __name__ == "__main__":
    main()
