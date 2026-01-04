#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                 << cudaGetErrorString(err) << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CUDA kernel for Matrix-Matrix multiplication
// C = A * B where A is MxK, B is KxN, C is MxN
__global__ void matmul_kernel(const double* A, const double* B, double* C,
                              int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        double sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// CUDA kernel for Matrix-Vector multiplication
// y = A * x where A is MxN, x is N, y is M
__global__ void matvec_kernel(const double* A, const double* x, double* y,
                              int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[row * N + j] * x[j];
        }
        y[row] = sum;
    }
}

// Initialize matrix with values (row-major)
void init_matrix(vector<double>& mat, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            mat[i * cols + j] = (i * cols + j) * 0.001;
        }
    }
}

// Initialize vector with values
void init_vector(vector<double>& vec, size_t size) {
    for (size_t i = 0; i < size; i++) {
        vec[i] = i * 0.001;
    }
}

// Calculate FLOPS
double calculate_gflops(size_t operations, double time_ms) {
    return (operations / 1e9) / (time_ms / 1000.0);
}

void benchmark_matmul_cuda(size_t M, size_t K, size_t N) {
    // Allocate host memory
    vector<double> h_A(M * K);
    vector<double> h_B(K * N);
    vector<double> h_C(M * N);
    
    // Initialize
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);
    
    // Allocate device memory
    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(double)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(double), cudaMemcpyHostToDevice));
    
    // Configure kernel
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    // Warmup
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark (compute only)
    auto start = high_resolution_clock::now();
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = high_resolution_clock::now();
    
    double compute_time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    
    // Benchmark with transfers
    start = high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(double), cudaMemcpyHostToDevice));
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost));
    end = high_resolution_clock::now();
    
    double total_time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    
    size_t ops = 2ULL * M * N * K;
    double compute_gflops = calculate_gflops(ops, compute_time_ms);
    double total_gflops = calculate_gflops(ops, total_time_ms);
    
    cout << "Mat*Mat (" << M << "x" << K << " * " << K << "x" << N << "):" << endl;
    cout << "  Compute: " << fixed << setprecision(3) << compute_time_ms << " ms, "
         << compute_gflops << " GFLOPS" << endl;
    cout << "  Total:   " << total_time_ms << " ms, "
         << total_gflops << " GFLOPS (with transfers)" << endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

void benchmark_matvec_cuda(size_t M, size_t N) {
    // Allocate host memory
    vector<double> h_A(M * N);
    vector<double> h_x(N);
    vector<double> h_y(M);
    
    // Initialize
    init_matrix(h_A, M, N);
    init_vector(h_x, N);
    
    // Allocate device memory
    double *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(double)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    
    // Configure kernel
    int block = 256;
    int grid = (M + block - 1) / block;
    
    // Warmup
    matvec_kernel<<<grid, block>>>(d_A, d_x, d_y, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark (compute only)
    auto start = high_resolution_clock::now();
    matvec_kernel<<<grid, block>>>(d_A, d_x, d_y, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = high_resolution_clock::now();
    
    double compute_time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    
    // Benchmark with transfers
    start = high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    matvec_kernel<<<grid, block>>>(d_A, d_x, d_y, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, M * sizeof(double), cudaMemcpyDeviceToHost));
    end = high_resolution_clock::now();
    
    double total_time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    
    size_t ops = 2ULL * M * N;
    double compute_gflops = calculate_gflops(ops, compute_time_ms);
    double total_gflops = calculate_gflops(ops, total_time_ms);
    
    cout << "Mat*Vec (" << M << "x" << N << " * " << N << "):" << endl;
    cout << "  Compute: " << fixed << setprecision(3) << compute_time_ms << " ms, "
         << compute_gflops << " GFLOPS" << endl;
    cout << "  Total:   " << total_time_ms << " ms, "
         << total_gflops << " GFLOPS (with transfers)" << endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
}

int main() {
    // Get GPU properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    cout << "=== CUDA Performance Comparison ===" << endl;
    cout << "GPU: " << prop.name << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
    cout << "Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << endl;
    cout << endl;
    
    vector<size_t> sizes = {128, 256, 512, 1024, 2048, 4096};
    
    for (size_t n : sizes) {
        cout << "\n========== Size: " << n << " ==========" << endl;
        
        benchmark_matmul_cuda(n, n, n);
        cout << endl;
        benchmark_matvec_cuda(n, n);
        
        size_t matmul_ops = 2ULL * n * n * n;
        size_t matvec_ops = 2ULL * n * n;
        double ops_ratio = (double)matmul_ops / matvec_ops;
        
        cout << "\nOps ratio (MatMul/MatVec): " << ops_ratio << "x" << endl;
    }
    
    cout << "\n=== GPU Performance Analysis ===" << endl;
    cout << "Mat*Mat characteristics:" << endl;
    cout << "  - High parallelism: many independent elements" << endl;
    cout << "  - Compute-intensive: amortizes kernel launch overhead" << endl;
    cout << "  - Memory transfers less dominant for large sizes" << endl;
    cout << "\nMat*Vec characteristics:" << endl;
    cout << "  - Lower parallelism: fewer output elements" << endl;
    cout << "  - Memory-bound: limited arithmetic intensity" << endl;
    cout << "  - Transfer overhead more significant" << endl;
    cout << "  - May not fully utilize GPU for small sizes" << endl;
    
    return 0;
}
