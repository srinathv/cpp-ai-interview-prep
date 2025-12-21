#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

// Parallel for loop
void parallelFor() {
    std::cout << "\n=== Parallel For ===" << std::endl;
    
    const int N = 1000000;
    std::vector<int> data(N);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        data[i] = i * i;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Parallel time: " << duration.count() << " ms" << std::endl;
    std::cout << "Number of threads: " << omp_get_max_threads() << std::endl;
}

// Reduction operation
void parallelReduction() {
    std::cout << "\n=== Parallel Reduction ===" << std::endl;
    
    const int N = 100000000;
    long long sum = 0;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        sum += i;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Sum: " << sum << std::endl;
    std::cout << "Time: " << duration.count() << " ms" << std::endl;
}

// Parallel sections
void parallelSections() {
    std::cout << "\n=== Parallel Sections ===" << std::endl;
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            std::cout << "Section 1 by thread " << omp_get_thread_num() << std::endl;
        }
        
        #pragma omp section
        {
            std::cout << "Section 2 by thread " << omp_get_thread_num() << std::endl;
        }
        
        #pragma omp section
        {
            std::cout << "Section 3 by thread " << omp_get_thread_num() << std::endl;
        }
    }
}

// Matrix multiplication
void matrixMultiply(const std::vector<std::vector<double>>& A,
                    const std::vector<std::vector<double>>& B,
                    std::vector<std::vector<double>>& C) {
    int n = A.size();
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Critical section
void demonstrateCritical() {
    std::cout << "\n=== Critical Section ===" << std::endl;
    
    int counter = 0;
    
    #pragma omp parallel for
    for (int i = 0; i < 1000; ++i) {
        #pragma omp critical
        {
            ++counter;
        }
    }
    
    std::cout << "Counter: " << counter << std::endl;
}

// Atomic operations
void demonstrateAtomic() {
    std::cout << "\n=== Atomic Operations ===" << std::endl;
    
    int counter = 0;
    
    #pragma omp parallel for
    for (int i = 0; i < 1000; ++i) {
        #pragma omp atomic
        ++counter;
    }
    
    std::cout << "Counter: " << counter << std::endl;
}

int main() {
    std::cout << "=== OpenMP Basics ===" << std::endl;
    std::cout << "Max threads: " << omp_get_max_threads() << std::endl;
    
    parallelFor();
    parallelReduction();
    parallelSections();
    demonstrateCritical();
    demonstrateAtomic();
    
    // Matrix multiplication
    std::cout << "\n=== Matrix Multiplication ===" << std::endl;
    int n = 500;
    std::vector<std::vector<double>> A(n, std::vector<double>(n, 1.0));
    std::vector<std::vector<double>> B(n, std::vector<double>(n, 2.0));
    std::vector<std::vector<double>> C(n, std::vector<double>(n, 0.0));
    
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiply(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Matrix multiply time: " << duration.count() << " ms" << std::endl;
    std::cout << "Result[0][0]: " << C[0][0] << std::endl;
    
    return 0;
}
