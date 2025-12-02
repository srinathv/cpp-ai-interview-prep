#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <omp.h>

using namespace std;
using namespace std::chrono;

// Matrix-Matrix Multiplication: C = A * B (OpenMP parallelized)
// A: M x K, B: K x N, C: M x N
void matmul_omp(const vector<vector<double>>& A, 
                const vector<vector<double>>& B,
                vector<vector<double>>& C) {
    size_t M = A.size();
    size_t K = A[0].size();
    size_t N = B[0].size();
    
    #pragma omp parallel for collapse(2)
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

// Matrix-Vector Multiplication: y = A * x (OpenMP parallelized)
// A: M x N, x: N, y: M
void matvec_omp(const vector<vector<double>>& A,
                const vector<double>& x,
                vector<double>& y) {
    size_t M = A.size();
    size_t N = A[0].size();
    
    #pragma omp parallel for
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

void benchmark_matmul(size_t M, size_t K, size_t N, int num_threads) {
    // Set number of threads
    omp_set_num_threads(num_threads);
    
    // Allocate matrices
    vector<vector<double>> A(M, vector<double>(K));
    vector<vector<double>> B(K, vector<double>(N));
    vector<vector<double>> C(M, vector<double>(N));
    
    // Initialize
    init_matrix(A, M, K);
    init_matrix(B, K, N);
    
    // Warmup
    matmul_omp(A, B, C);
    
    // Benchmark
    auto start = high_resolution_clock::now();
    matmul_omp(A, B, C);
    auto end = high_resolution_clock::now();
    
    double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    size_t ops = 2ULL * M * N * K;
    double gflops = calculate_gflops(ops, time_ms);
    
    cout << "Mat*Mat (" << M << "x" << K << " * " << K << "x" << N << ") [" 
         << num_threads << " threads]: "
         << fixed << setprecision(3) << time_ms << " ms, "
         << gflops << " GFLOPS" << endl;
}

void benchmark_matvec(size_t M, size_t N, int num_threads) {
    // Set number of threads
    omp_set_num_threads(num_threads);
    
    // Allocate matrix and vectors
    vector<vector<double>> A(M, vector<double>(N));
    vector<double> x(N);
    vector<double> y(M);
    
    // Initialize
    init_matrix(A, M, N);
    init_vector(x, N);
    
    // Warmup
    matvec_omp(A, x, y);
    
    // Benchmark
    auto start = high_resolution_clock::now();
    matvec_omp(A, x, y);
    auto end = high_resolution_clock::now();
    
    double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    size_t ops = 2ULL * M * N;
    double gflops = calculate_gflops(ops, time_ms);
    
    cout << "Mat*Vec (" << M << "x" << N << " * " << N << ") [" 
         << num_threads << " threads]: "
         << fixed << setprecision(3) << time_ms << " ms, "
         << gflops << " GFLOPS" << endl;
}

int main() {
    cout << "=== Multi-Threaded (OpenMP) Performance Comparison ===" << endl;
    cout << "Max threads available: " << omp_get_max_threads() << endl;
    cout << endl;
    
    vector<size_t> sizes = {128, 256, 512, 1024, 2048};
    vector<int> thread_counts = {1, 2, 4, 8};
    
    // Filter thread counts to available
    int max_threads = omp_get_max_threads();
    thread_counts.erase(
        remove_if(thread_counts.begin(), thread_counts.end(),
                  [max_threads](int t) { return t > max_threads; }),
        thread_counts.end()
    );
    
    for (size_t n : sizes) {
        cout << "\n========== Size: " << n << " ==========" << endl;
        
        for (int threads : thread_counts) {
            cout << "\n--- " << threads << " Thread(s) ---" << endl;
            
            benchmark_matmul(n, n, n, threads);
            benchmark_matvec(n, n, threads);
        }
        
        cout << "\nTheoretical ops ratio: " << n << "x" << endl;
    }
    
    cout << "\n=== Thread Scaling Analysis ===" << endl;
    cout << "Mat*Mat benefits more from parallelization due to:" << endl;
    cout << "  1. Higher computational intensity (O(n³) vs O(n²))" << endl;
    cout << "  2. Better work distribution across threads" << endl;
    cout << "  3. More opportunity to amortize thread creation overhead" << endl;
    cout << "\nMat*Vec challenges:" << endl;
    cout << "  1. Lower work per thread" << endl;
    cout << "  2. More memory-bound" << endl;
    cout << "  3. Thread overhead can dominate for small sizes" << endl;
    
    return 0;
}
