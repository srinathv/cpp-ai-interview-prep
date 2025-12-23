#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

using namespace std;

/**
 * OpenMP Parallel Implementation
 * 
 * Parallelization strategies:
 * 1. Parallel row max calculation
 * 2. Parallel column max calculation (with care for cache)
 * 3. Parallel reduction for sum calculation
 * 4. Dynamic scheduling for load balancing
 */

int maxIncreaseKeepingSkyline(vector<vector<int>>& grid) {
    int N = grid.size();
    
    vector<int> rowMax(N, 0);
    vector<int> colMax(N, 0);
    
    // Parallel row max calculation
    // Each row is independent, perfect for parallelization
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        int maxVal = 0;
        for (int j = 0; j < N; ++j) {
            maxVal = max(maxVal, grid[i][j]);
        }
        rowMax[i] = maxVal;
    }
    
    // Parallel column max calculation
    // Columns are independent, but cache-unfriendly
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < N; ++j) {
        int maxVal = 0;
        for (int i = 0; i < N; ++i) {
            maxVal = max(maxVal, grid[i][j]);
        }
        colMax[j] = maxVal;
    }
    
    // Parallel sum calculation with reduction
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

// Advanced version with nested parallelism for very large grids
int maxIncreaseKeepingSkylineNested(vector<vector<int>>& grid) {
    int N = grid.size();
    
    vector<int> rowMax(N, 0);
    vector<int> colMax(N, 0);
    
    // Enable nested parallelism if grid is very large
    omp_set_nested(1);
    
    #pragma omp parallel sections
    {
        // Section 1: Calculate row maximums
        #pragma omp section
        {
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < N; ++i) {
                int maxVal = 0;
                for (int j = 0; j < N; ++j) {
                    maxVal = max(maxVal, grid[i][j]);
                }
                rowMax[i] = maxVal;
            }
        }
        
        // Section 2: Calculate column maximums (can run concurrently)
        #pragma omp section
        {
            #pragma omp parallel for schedule(dynamic)
            for (int j = 0; j < N; ++j) {
                int maxVal = 0;
                for (int i = 0; i < N; ++i) {
                    maxVal = max(maxVal, grid[i][j]);
                }
                colMax[j] = maxVal;
            }
        }
    }
    
    // Wait for both sections to complete, then calculate sum
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

int main() {
    vector<vector<int>> grid = {
        {3, 0, 8, 4}, 
        {2, 4, 5, 7}, 
        {9, 2, 6, 3}, 
        {0, 3, 1, 0}
    };
    
    // Set number of threads (defaults to number of cores)
    omp_set_num_threads(omp_get_max_threads());
    
    cout << "Using " << omp_get_max_threads() << " threads" << endl;
    cout << "OpenMP Parallel - Total increase: " << maxIncreaseKeepingSkyline(grid) << endl;
    
    // For very large grids, use nested version
    if (grid.size() > 512) {
        cout << "Nested Parallel - Total increase: " << maxIncreaseKeepingSkylineNested(grid) << endl;
    }
    
    return 0;
}
