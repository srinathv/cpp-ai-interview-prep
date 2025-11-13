#ifndef MATMUL_UTILS_HIP_H
#define MATMUL_UTILS_HIP_H
#include <iostream>
#include <hip/hip_runtime.h>
#include <cmath>
#define HIP_CHECK(call) do { hipError_t e = call; if (e != hipSuccess) { std::cerr << "HIP error: " << hipGetErrorString(e) << std::endl; exit(1); } } while(0)
void matmulCPU(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) for (int j = 0; j < K; j++) {
        float sum = 0.0f; for (int k = 0; k < N; k++) sum += A[i*N+k] * B[k*K+j]; C[i*K+j] = sum;
    }
}
bool verifyMatmul(const float *g, const float *c, int s) {
    for (int i = 0; i < s; i++) if (fabs(g[i] - c[i]) > 1e-3f) return false; return true;
}
double calculateGFLOPS(int M, int N, int K, float ms) { return (2.0*M*N*K) / (ms/1000.0) / 1e9; }
void initMatrix(float *m, int s) { for (int i = 0; i < s; i++) m[i] = float(rand()) / RAND_MAX; }
#endif
