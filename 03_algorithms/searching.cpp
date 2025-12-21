#include <iostream>
#include <vector>
#include <algorithm>

// Linear Search - O(n)
int linearSearch(const std::vector<int>& arr, int target) {
    for (size_t i = 0; i < arr.size(); ++i) {
        if (arr[i] == target) return i;
    }
    return -1;
}

// Binary Search - O(log n) - requires sorted array
int binarySearch(const std::vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

// Binary Search Recursive
int binarySearchRecursive(const std::vector<int>& arr, int left, int right, int target) {
    if (left > right) return -1;
    int mid = left + (right - left) / 2;
    if (arr[mid] == target) return mid;
    if (arr[mid] > target) return binarySearchRecursive(arr, left, mid - 1, target);
    return binarySearchRecursive(arr, mid + 1, right, target);
}

// Find first and last position in sorted array
std::vector<int> searchRange(const std::vector<int>& nums, int target) {
    auto lower = std::lower_bound(nums.begin(), nums.end(), target);
    auto upper = std::upper_bound(nums.begin(), nums.end(), target);
    
    if (lower == nums.end() || *lower != target)
        return {-1, -1};
    
    return {static_cast<int>(lower - nums.begin()), 
            static_cast<int>(upper - nums.begin() - 1)};
}

// Search in rotated sorted array
int searchRotated(const std::vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        
        if (nums[left] <= nums[mid]) {
            if (nums[left] <= target && target < nums[mid])
                right = mid - 1;
            else
                left = mid + 1;
        } else {
            if (nums[mid] < target && target <= nums[right])
                left = mid + 1;
            else
                right = mid - 1;
        }
    }
    return -1;
}

int main() {
    std::vector<int> arr = {2, 5, 8, 12, 16, 23, 38, 45, 56, 67, 78};
    
    int target = 23;
    std::cout << "Linear search for " << target << ": " << linearSearch(arr, target) << std::endl;
    std::cout << "Binary search for " << target << ": " << binarySearch(arr, target) << std::endl;
    std::cout << "Binary search (recursive) for " << target << ": " 
              << binarySearchRecursive(arr, 0, arr.size() - 1, target) << std::endl;
    
    std::vector<int> arr2 = {5, 7, 7, 8, 8, 10};
    auto range = searchRange(arr2, 8);
    std::cout << "Range of 8: [" << range[0] << ", " << range[1] << "]" << std::endl;
    
    std::vector<int> rotated = {4, 5, 6, 7, 0, 1, 2};
    std::cout << "Search 0 in rotated array: " << searchRotated(rotated, 0) << std::endl;
    
    return 0;
}
