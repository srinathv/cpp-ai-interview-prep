#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

/**
 * Baseline implementation - straightforward solution
 * Time Complexity: O(N^2)
 * Space Complexity: O(N)
 */
int maxIncreaseKeepingSkyline(vector<vector<int>>& grid) {
    int N = grid.size();
    
    vector<int> myRowMax(N);
    vector<int> myColumnMax(N);
    
    // Find maximum in each row
    for (int i = 0; i < N; ++i) {
        myRowMax[i] = grid[i][0];
        for (int j = 0; j < N; ++j) {
            if (myRowMax[i] < grid[i][j]) {
                myRowMax[i] = grid[i][j];
            }
        }
    }
    
    // Find maximum in each column
    for (int j = 0; j < N; ++j) {
        myColumnMax[j] = grid[0][j];
        for (int i = 0; i < N; ++i) {
            if (myColumnMax[j] < grid[i][j]) {
                myColumnMax[j] = grid[i][j];
            }
        }
    }
    
    // Calculate total increase
    int sum = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            sum += min(myRowMax[i], myColumnMax[j]) - grid[i][j];
        }
    }
    
    return sum;
}

int main() {
    vector<vector<int>> grid = {
        {3, 0, 8, 4}, 
        {2, 4, 5, 7}, 
        {9, 2, 6, 3}, 
        {0, 3, 1, 0}
    };
    
    cout << "Total increase: " << maxIncreaseKeepingSkyline(grid) << endl;
    return 0;
}
