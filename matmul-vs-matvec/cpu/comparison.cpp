#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>

using namespace std;
using namespace std::chrono;

// Matrix-Matrix Multiplication: C = A * B
// A: M x K, B: K x N, C: M x N
void matmul(const vector<vector<double>>& A, 
            const vector<vector<double>>& B,
            vector<vector<double>>& C) {
    size_t M = A.size();
    size_t K = A[0].size();
    size_t N = B[0].size();
    
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < K; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// Matrix-Vector Multiplication: y = A * x
// A: M x N, x: N, y: M
void matvec(const vector<vector<double>>& A,
            const vector<double>& x,
            vector<double>& y) {
    size_t M = A.size();
    size_t N = A[0].size();
    
    for (size_t i = 0; i < M; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < N; j++) {
            sum += A[i][j] * x[j];
        }
        y[i] = sum;
    }
}

// Initialize matrix with values
void init_matrix(vector<vector<double>>& mat, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            mat[i][j] = (i * cols + j) * 0.001;
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

void benchmark_matmul(size_t M, size_t K, size_t N) {
    // Allocate matrices
    vector<vector<double>> A(M, vector<double>(K));
    vector<vector<double>> B(K, vector<double>(N));
    vector<vector<double>> C(M, vector<double>(N));
    
    // Initialize
    init_matrix(A, M, K);
    init_matrix(B, K, N);
    
    // Warmup
    matmul(A, B, C);
    
    // Benchmark
    auto start = high_resolution_clock::now();
    matmul(A, B, C);
    auto end = high_resolution_clock::now();
    
    double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    size_t ops = 2ULL * M * N * K; // multiply-add counts as 2 ops
    double gflops = calculate_gflops(ops, time_ms);
    
    cout << "Mat*Mat (" << M << "x" << K << " * " << K << "x" << N << "): "
         << fixed << setprecision(3) << time_ms << " ms, "
         << gflops << " GFLOPS" << endl;
}

void benchmark_matvec(size_t M, size_t N) {
    // Allocate matrix and vectors
    vector<vector<double>> A(M, vector<double>(N));
    vector<double> x(N);
    vector<double> y(M);
    
    // Initialize
    init_matrix(A, M, N);
    init_vector(x, N);
    
    // Warmup
    matvec(A, x, y);
    
    // Benchmark
    auto start = high_resolution_clock::now();
    matvec(A, x, y);
    auto end = high_resolution_clock::now();
    
    double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    size_t ops = 2ULL * M * N; // multiply-add counts as 2 ops
    double gflops = calculate_gflops(ops, time_ms);
    
    cout << "Mat*Vec (" << M << "x" << N << " * " << N << "): "
         << fixed << setprecision(3) << time_ms << " ms, "
         << gflops << " GFLOPS" << endl;
}

int main() {
    cout << "=== CPU Single-Threaded Performance Comparison ===" << endl;
    cout << endl;
    
    vector<size_t> sizes = {128, 256, 512, 1024, 2048};
    
    for (size_t n : sizes) {
        cout << "\nSize: " << n << endl;
        cout << string(50, '-') << endl;
        
        // Matrix-Matrix multiplication (square matrices)
        benchmark_matmul(n, n, n);
        
        // Matrix-Vector multiplication
        benchmark_matvec(n, n);
        
        // Calculate theoretical speedup
        size_t matmul_ops = 2ULL * n * n * n;
        size_t matvec_ops = 2ULL * n * n;
        double ops_ratio = (double)matmul_ops / matvec_ops;
        
        cout << "Ops ratio (MatMul/MatVec): " << ops_ratio << "x" << endl;
    }
    
    cout << "\n=== Analysis ===" << endl;
    cout << "Mat*Mat has O(n³) complexity" << endl;
    cout << "Mat*Vec has O(n²) complexity" << endl;
    cout << "For size n, Mat*Mat does n times more operations than Mat*Vec" << endl;
    
    return 0;
}
