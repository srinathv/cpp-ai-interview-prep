#include <iostream>
#include <hip/hip_runtime.h>
#include "matmul_utils.h"
#define TILE_SIZE 32

__global__ void matmulTiled(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE], Bs[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * TILE_SIZE + threadIdx.y, col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < N) ? A[row*N+a_col] : 0.0f;
        int b_row = t * TILE_SIZE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (b_row < N && col < K) ? B[b_row*K+col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_SIZE; k++) sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < K) C[row*K+col] = sum;
}

int main() {
    std::cout << "=== Tiled Matrix Multiply (HIP) ===\n";
    const int M=2048, N=2048, K=2048;
    float *h_A = new float[M*N], *h_B = new float[N*K], *h_C = new float[M*K];
    initMatrix(h_A, M*N); initMatrix(h_B, N*K);
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, M*N*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, N*K*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, M*K*sizeof(float)));
    HIP_CHECK(hipMemcpy(d_A, h_A, M*N*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, N*K*sizeof(float), hipMemcpyHostToDevice));
    dim3 block(TILE_SIZE, TILE_SIZE), grid((K+TILE_SIZE-1)/TILE_SIZE, (M+TILE_SIZE-1)/TILE_SIZE);
    hipLaunchKernelGGL(matmulTiled, grid, block, 0, 0, d_A, d_B, d_C, M, N, K);
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for(int i=0; i<10; i++) hipLaunchKernelGGL(matmulTiled, grid, block, 0, 0, d_A, d_B, d_C, M, N, K);
    HIP_CHECK(hipEventRecord(stop)); HIP_CHECK(hipEventSynchronize(stop));
    float ms; HIP_CHECK(hipEventElapsedTime(&ms, start, stop)); ms /= 10;
    std::cout << "Time: " << ms << " ms, " << calculateGFLOPS(M,N,K,ms) << " GFLOPS\n";
    delete[] h_A; delete[] h_B; delete[] h_C;
    HIP_CHECK(hipFree(d_A)); HIP_CHECK(hipFree(d_B)); HIP_CHECK(hipFree(d_C));
    return 0;
}
