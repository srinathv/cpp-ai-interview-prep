#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace chrono;

// Baseline implementation
int baselineImpl(vector<vector<int>>& grid) {
    int N = grid.size();
    vector<int> rowMax(N, 0);
    vector<int> colMax(N, 0);
    
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            rowMax[i] = max(rowMax[i], grid[i][j]);
        }
    }
    
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            colMax[j] = max(colMax[j], grid[i][j]);
        }
    }
    
    int sum = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            sum += min(rowMax[i], colMax[j]) - grid[i][j];
        }
    }
    
    return sum;
}

// CPU optimized implementation
int cpuOptimizedImpl(vector<vector<int>>& grid) {
    int N = grid.size();
    vector<int> rowMax(N, 0);
    vector<int> colMax(N, 0);
    
    for (int i = 0; i < N; ++i) {
        int maxVal = 0;
        #pragma GCC ivdep
        for (int j = 0; j < N; ++j) {
            maxVal = max(maxVal, grid[i][j]);
        }
        rowMax[i] = maxVal;
    }
    
    for (int j = 0; j < N; ++j) {
        int maxVal = 0;
        for (int i = 0; i < N; ++i) {
            maxVal = max(maxVal, grid[i][j]);
        }
        colMax[j] = maxVal;
    }
    
    int totalSum = 0;
    for (int i = 0; i < N; ++i) {
        int rowMaxVal = rowMax[i];
        int rowSum = 0;
        #pragma GCC ivdep
        for (int j = 0; j < N; ++j) {
            rowSum += min(rowMaxVal, colMax[j]) - grid[i][j];
        }
        totalSum += rowSum;
    }
    
    return totalSum;
}

#ifdef _OPENMP
// OpenMP parallel implementation
int openmpImpl(vector<vector<int>>& grid) {
    int N = grid.size();
    vector<int> rowMax(N, 0);
    vector<int> colMax(N, 0);
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        int maxVal = 0;
        for (int j = 0; j < N; ++j) {
            maxVal = max(maxVal, grid[i][j]);
        }
        rowMax[i] = maxVal;
    }
    
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < N; ++j) {
        int maxVal = 0;
        for (int i = 0; i < N; ++i) {
            maxVal = max(maxVal, grid[i][j]);
        }
        colMax[j] = maxVal;
    }
    
    int totalSum = 0;
    #pragma omp parallel for reduction(+:totalSum) schedule(static)
    for (int i = 0; i < N; ++i) {
        int rowMaxVal = rowMax[i];
        for (int j = 0; j < N; ++j) {
            totalSum += min(rowMaxVal, colMax[j]) - grid[i][j];
        }
    }
    
    return totalSum;
}
#endif

// Generate random grid
vector<vector<int>> generateGrid(int N, int maxHeight = 100) {
    vector<vector<int>> grid(N, vector<int>(N));
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, maxHeight);
    
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            grid[i][j] = dis(gen);
        }
    }
    
    return grid;
}

// Benchmark function
template<typename Func>
double benchmark(Func func, vector<vector<int>>& grid, int iterations = 10) {
    // Warm-up
    func(grid);
    
    auto start = high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        volatile int result = func(grid);
        (void)result; // Prevent optimization
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    
    return duration / (double)iterations;
}

void printResults(const string& name, double time, double baseline_time = 0) {
    cout << setw(20) << left << name 
         << setw(15) << right << fixed << setprecision(2) << time << " µs";
    
    if (baseline_time > 0) {
        double speedup = baseline_time / time;
        cout << setw(15) << right << fixed << setprecision(2) << speedup << "x";
    }
    
    cout << endl;
}

int main() {
    cout << "========================================" << endl;
    cout << "  Max Increase Skyline - Benchmarks" << endl;
    cout << "========================================" << endl;
    cout << endl;
    
    vector<int> sizes = {64, 128, 256, 512, 1024, 2048};
    
    for (int N : sizes) {
        cout << "Grid Size: " << N << "x" << N << endl;
        cout << string(60, '-') << endl;
        
        auto grid = generateGrid(N);
        
        cout << setw(20) << left << "Implementation" 
             << setw(15) << right << "Time (µs)" 
             << setw(15) << right << "Speedup" << endl;
        cout << string(60, '-') << endl;
        
        // Baseline
        double baseline_time = benchmark(baselineImpl, grid, 5);
        printResults("Baseline", baseline_time);
        
        // CPU Optimized
        double cpu_opt_time = benchmark(cpuOptimizedImpl, grid, 5);
        printResults("CPU Optimized", cpu_opt_time, baseline_time);
        
        #ifdef _OPENMP
        // OpenMP
        int num_threads = omp_get_max_threads();
        cout << "OpenMP (" << num_threads << " threads):" << endl;
        double openmp_time = benchmark(openmpImpl, grid, 5);
        printResults("  OpenMP Parallel", openmp_time, baseline_time);
        #endif
        
        cout << endl;
    }
    
    // Verify correctness
    cout << "Correctness Test:" << endl;
    cout << string(60, '-') << endl;
    
    vector<vector<int>> testGrid = {
        {3, 0, 8, 4}, 
        {2, 4, 5, 7}, 
        {9, 2, 6, 3}, 
        {0, 3, 1, 0}
    };
    
    int baseline_result = baselineImpl(testGrid);
    int cpu_opt_result = cpuOptimizedImpl(testGrid);
    
    cout << "Expected result: 35" << endl;
    cout << "Baseline result: " << baseline_result << " " 
         << (baseline_result == 35 ? "✓" : "✗") << endl;
    cout << "CPU Optimized result: " << cpu_opt_result << " " 
         << (cpu_opt_result == 35 ? "✓" : "✗") << endl;
    
    #ifdef _OPENMP
    int openmp_result = openmpImpl(testGrid);
    cout << "OpenMP result: " << openmp_result << " " 
         << (openmp_result == 35 ? "✓" : "✗") << endl;
    #endif
    
    cout << endl;
    
    return 0;
}
