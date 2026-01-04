#include <iostream>
#include <vector>
#include <algorithm>

// Fibonacci - Classic DP example
int fibonacci(int n) {
    if (n <= 1) return n;
    std::vector<int> dp(n + 1);
    dp[0] = 0;
    dp[1] = 1;
    for (int i = 2; i <= n; ++i) {
        dp[i] = dp[i-1] + dp[i-2];
    }
    return dp[n];
}

// Longest Common Subsequence
int longestCommonSubsequence(const std::string& text1, const std::string& text2) {
    int m = text1.length(), n = text2.length();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
    
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (text1[i-1] == text2[j-1])
                dp[i][j] = dp[i-1][j-1] + 1;
            else
                dp[i][j] = std::max(dp[i-1][j], dp[i][j-1]);
        }
    }
    return dp[m][n];
}

// Longest Increasing Subsequence
int lengthOfLIS(const std::vector<int>& nums) {
    if (nums.empty()) return 0;
    std::vector<int> dp(nums.size(), 1);
    
    for (size_t i = 1; i < nums.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            if (nums[i] > nums[j]) {
                dp[i] = std::max(dp[i], dp[j] + 1);
            }
        }
    }
    return *std::max_element(dp.begin(), dp.end());
}

// 0/1 Knapsack Problem
int knapsack(const std::vector<int>& weights, const std::vector<int>& values, int capacity) {
    int n = weights.size();
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(capacity + 1, 0));
    
    for (int i = 1; i <= n; ++i) {
        for (int w = 0; w <= capacity; ++w) {
            if (weights[i-1] <= w) {
                dp[i][w] = std::max(dp[i-1][w], 
                                   dp[i-1][w - weights[i-1]] + values[i-1]);
            } else {
                dp[i][w] = dp[i-1][w];
            }
        }
    }
    return dp[n][capacity];
}

// Coin Change - minimum coins
int coinChange(const std::vector<int>& coins, int amount) {
    std::vector<int> dp(amount + 1, amount + 1);
    dp[0] = 0;
    
    for (int i = 1; i <= amount; ++i) {
        for (int coin : coins) {
            if (coin <= i) {
                dp[i] = std::min(dp[i], dp[i - coin] + 1);
            }
        }
    }
    return dp[amount] > amount ? -1 : dp[amount];
}

// Edit Distance (Levenshtein)
int editDistance(const std::string& word1, const std::string& word2) {
    int m = word1.length(), n = word2.length();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));
    
    for (int i = 0; i <= m; ++i) dp[i][0] = i;
    for (int j = 0; j <= n; ++j) dp[0][j] = j;
    
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (word1[i-1] == word2[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = 1 + std::min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]});
            }
        }
    }
    return dp[m][n];
}

int main() {
    std::cout << "Fibonacci(10): " << fibonacci(10) << std::endl;
    
    std::string s1 = "abcde", s2 = "ace";
    std::cout << "LCS of '" << s1 << "' and '" << s2 << "': " 
              << longestCommonSubsequence(s1, s2) << std::endl;
    
    std::vector<int> nums = {10, 9, 2, 5, 3, 7, 101, 18};
    std::cout << "Length of LIS: " << lengthOfLIS(nums) << std::endl;
    
    std::vector<int> weights = {1, 3, 4, 5};
    std::vector<int> values = {1, 4, 5, 7};
    std::cout << "Knapsack max value: " << knapsack(weights, values, 7) << std::endl;
    
    std::vector<int> coins = {1, 2, 5};
    std::cout << "Coin change for 11: " << coinChange(coins, 11) << std::endl;
    
    std::string w1 = "horse", w2 = "ros";
    std::cout << "Edit distance: " << editDistance(w1, w2) << std::endl;
    
    return 0;
}
