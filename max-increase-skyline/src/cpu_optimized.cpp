#include <iostream>
#include <vector>
#include <algorithm>
#include <immintrin.h>

using namespace std;

/**
 * CPU Optimized Implementation
 * 
 * Optimizations:
 * 1. Cache-friendly memory access patterns
 * 2. Loop unrolling hints for compiler
 * 3. SIMD vectorization where possible
 * 4. Reduced branching with std::max
 * 5. Better data locality
 */

int maxIncreaseKeepingSkyline(vector<vector<int>>& grid) {
    int N = grid.size();
    
    vector<int> rowMax(N, 0);
    vector<int> colMax(N, 0);
    
    // Optimized row max calculation with better cache locality
    // Process row-by-row (cache-friendly)
    for (int i = 0; i < N; ++i) {
        int maxVal = 0;
        // Compiler can auto-vectorize this loop
        #pragma GCC ivdep
        for (int j = 0; j < N; ++j) {
            maxVal = max(maxVal, grid[i][j]);
        }
        rowMax[i] = maxVal;
    }
    
    // Optimized column max calculation
    // Although column-wise access is cache-unfriendly, we minimize passes
    for (int j = 0; j < N; ++j) {
        int maxVal = 0;
        for (int i = 0; i < N; ++i) {
            maxVal = max(maxVal, grid[i][j]);
        }
        colMax[j] = maxVal;
    }
    
    // Calculate sum with reduced overhead
    int totalSum = 0;
    
    // Process row by row for better cache performance
    for (int i = 0; i < N; ++i) {
        int rowMaxVal = rowMax[i];
        int rowSum = 0;
        
        // Inner loop can be vectorized
        #pragma GCC ivdep
        for (int j = 0; j < N; ++j) {
            rowSum += min(rowMaxVal, colMax[j]) - grid[i][j];
        }
        
        totalSum += rowSum;
    }
    
    return totalSum;
}

// Alternative SIMD-optimized version for large grids (N >= 8)
#ifdef __AVX2__
int maxIncreaseKeepingSkylineSIMD(vector<vector<int>>& grid) {
    int N = grid.size();
    
    vector<int> rowMax(N, 0);
    vector<int> colMax(N, 0);
    
    // Calculate row maximums
    for (int i = 0; i < N; ++i) {
        __m256i vmax = _mm256_setzero_si256();
        int j = 0;
        
        // Process 8 integers at a time with AVX2
        for (; j + 7 < N; j += 8) {
            __m256i vals = _mm256_loadu_si256((__m256i*)&grid[i][j]);
            vmax = _mm256_max_epi32(vmax, vals);
        }
        
        // Horizontal max of vector
        __m128i vmax_low = _mm256_castsi256_si128(vmax);
        __m128i vmax_high = _mm256_extracti128_si256(vmax, 1);
        __m128i vmax128 = _mm_max_epi32(vmax_low, vmax_high);
        
        int temp[4];
        _mm_storeu_si128((__m128i*)temp, vmax128);
        int maxVal = max({temp[0], temp[1], temp[2], temp[3]});
        
        // Handle remaining elements
        for (; j < N; ++j) {
            maxVal = max(maxVal, grid[i][j]);
        }
        
        rowMax[i] = maxVal;
    }
    
    // Calculate column maximums (cannot vectorize as easily due to memory layout)
    for (int j = 0; j < N; ++j) {
        int maxVal = 0;
        for (int i = 0; i < N; ++i) {
            maxVal = max(maxVal, grid[i][j]);
        }
        colMax[j] = maxVal;
    }
    
    // Calculate sum
    int totalSum = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            totalSum += min(rowMax[i], colMax[j]) - grid[i][j];
        }
    }
    
    return totalSum;
}
#endif

int main() {
    vector<vector<int>> grid = {
        {3, 0, 8, 4}, 
        {2, 4, 5, 7}, 
        {9, 2, 6, 3}, 
        {0, 3, 1, 0}
    };
    
    cout << "CPU Optimized - Total increase: " << maxIncreaseKeepingSkyline(grid) << endl;
    
    #ifdef __AVX2__
    cout << "SIMD Optimized - Total increase: " << maxIncreaseKeepingSkylineSIMD(grid) << endl;
    #endif
    
    return 0;
}
