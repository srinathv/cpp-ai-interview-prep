#include <iostream>
#include <hip/hip_runtime.h>
#include "matmul_utils.h"

__global__ void matmulNaive(const float *A, const float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) sum += A[row*N+i] * B[i*K+col];
        C[row*K+col] = sum;
    }
}

int main() {
    std::cout << "=== Naive Matrix Multiply (HIP) ===\n";
    const int M=1024, N=1024, K=1024;
    float *h_A = new float[M*N], *h_B = new float[N*K], *h_C = new float[M*K];
    initMatrix(h_A, M*N); initMatrix(h_B, N*K);
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, M*N*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, N*K*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, M*K*sizeof(float)));
    HIP_CHECK(hipMemcpy(d_A, h_A, M*N*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, N*K*sizeof(float), hipMemcpyHostToDevice));
    dim3 block(16,16), grid((K+15)/16, (M+15)/16);
    hipLaunchKernelGGL(matmulNaive, grid, block, 0, 0, d_A, d_B, d_C, M, N, K);
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for(int i=0; i<10; i++) hipLaunchKernelGGL(matmulNaive, grid, block, 0, 0, d_A, d_B, d_C, M, N, K);
    HIP_CHECK(hipEventRecord(stop)); HIP_CHECK(hipEventSynchronize(stop));
    float ms; HIP_CHECK(hipEventElapsedTime(&ms, start, stop)); ms /= 10;
    HIP_CHECK(hipMemcpy(h_C, d_C, M*K*sizeof(float), hipMemcpyDeviceToHost));
    std::cout << "Time: " << ms << " ms, " << calculateGFLOPS(M,N,K,ms) << " GFLOPS\n";
    delete[] h_A; delete[] h_B; delete[] h_C;
    HIP_CHECK(hipFree(d_A)); HIP_CHECK(hipFree(d_B)); HIP_CHECK(hipFree(d_C));
    return 0;
}
