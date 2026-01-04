#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>

// Interview question: Two Sum
std::vector<int> twoSum(const std::vector<int>& nums, int target) {
    std::unordered_map<int, int> seen;
    for (size_t i = 0; i < nums.size(); ++i) {
        int complement = target - nums[i];
        if (seen.count(complement)) {
            return {seen[complement], static_cast<int>(i)};
        }
        seen[nums[i]] = i;
    }
    return {};
}

// Interview question: First non-repeating character
char firstNonRepeating(const std::string& s) {
    std::unordered_map<char, int> freq;
    for (char c : s) freq[c]++;
    for (char c : s) {
        if (freq[c] == 1) return c;
    }
    return '\0';
}

// Interview question: Group anagrams
std::vector<std::vector<std::string>> groupAnagrams(const std::vector<std::string>& strs) {
    std::unordered_map<std::string, std::vector<std::string>> groups;
    for (const auto& str : strs) {
        std::string key = str;
        std::sort(key.begin(), key.end());
        groups[key].push_back(str);
    }
    std::vector<std::vector<std::string>> result;
    for (auto& [key, group] : groups) {
        result.push_back(std::move(group));
    }
    return result;
}

int main() {
    std::cout << "=== Hash Map Operations ===" << std::endl;
    
    std::unordered_map<std::string, int> ages;
    ages["Alice"] = 30;
    ages["Bob"] = 25;
    
    std::cout << "Alice's age: " << ages["Alice"] << std::endl;
    
    std::cout << "\n=== Interview Questions ===" << std::endl;
    
    std::vector<int> nums = {2, 7, 11, 15};
    auto indices = twoSum(nums, 9);
    std::cout << "Two sum indices: [" << indices[0] << ", " << indices[1] << "]" << std::endl;
    
    std::string s = "leetcode";
    std::cout << "First non-repeating: " << firstNonRepeating(s) << std::endl;
    
    return 0;
}
