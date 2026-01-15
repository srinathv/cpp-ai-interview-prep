#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>

// Pattern 1: Swap function - MUST use references
void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

// Pattern 2: Output parameters - use references
bool divide(int numerator, int denominator, int& result) {
    if (denominator == 0) return false;
    result = numerator / denominator;
    return true;
}

// Pattern 3: Multiple return values via references
void getMinMax(const std::vector<int>& vec, int& min, int& max) {
    if (vec.empty()) return;
    min = max = vec[0];
    for (int val : vec) {
        if (val < min) min = val;
        if (val > max) max = val;
    }
}

// Pattern 4: Builder/Fluent Interface - return reference for chaining
class StringBuilder {
private:
    std::string data;
    
public:
    StringBuilder& append(const std::string& str) {
        data += str;
        return *this;
    }
    
    StringBuilder& appendLine(const std::string& str) {
        data += str + "\n";
        return *this;
    }
    
    std::string build() const {
        return data;
    }
};

// Pattern 5: Range-based for loops - const ref for reading
void printAllStrings(const std::vector<std::string>& strings) {
    for (const std::string& str : strings) {
        std::cout << str << std::endl;
    }
}

// Pattern 6: Range-based for loops - ref for modifying
void uppercaseAll(std::vector<std::string>& strings) {
    for (std::string& str : strings) {
        for (char& c : str) {
            c = std::toupper(c);
        }
    }
}

// Pattern 7: Accepting containers - const ref is standard
int countOccurrences(const std::map<std::string, int>& map, int value) {
    int count = 0;
    for (const auto& [key, val] : map) {
        if (val == value) count++;
    }
    return count;
}

// Pattern 8: Factory functions - return by value (RVO applies)
std::vector<int> createRange(int start, int end) {
    std::vector<int> result;
    for (int i = start; i < end; ++i) {
        result.push_back(i);
    }
    return result;
}

// Pattern 9: When you explicitly want a copy to modify locally
std::string removeSpaces(std::string str) {
    str.erase(
        std::remove(str.begin(), str.end(), ' '),
        str.end()
    );
    return str;
}

int main() {
    std::cout << "=== SWAP ===" << std::endl;
    int x = 10, y = 20;
    std::cout << "Before: x=" << x << ", y=" << y << std::endl;
    swap(x, y);
    std::cout << "After: x=" << x << ", y=" << y << std::endl;
    
    std::cout << "\n=== OUTPUT PARAMETERS ===" << std::endl;
    int result;
    if (divide(10, 3, result)) {
        std::cout << "10 / 3 = " << result << std::endl;
    }
    
    std::cout << "\n=== MULTIPLE OUTPUTS ===" << std::endl;
    std::vector<int> numbers = {5, 2, 8, 1, 9, 3};
    int min, max;
    getMinMax(numbers, min, max);
    std::cout << "Min: " << min << ", Max: " << max << std::endl;
    
    std::cout << "\n=== FLUENT INTERFACE ===" << std::endl;
    StringBuilder sb;
    std::string message = sb.append("Hello")
                           .append(" ")
                           .append("World")
                           .appendLine("!")
                           .append("Chaining works")
                           .build();
    std::cout << message << std::endl;
    
    std::cout << "\n=== RANGE-BASED LOOPS ===" << std::endl;
    std::vector<std::string> words = {"hello", "world", "cpp"};
    std::cout << "Original:" << std::endl;
    printAllStrings(words);
    
    uppercaseAll(words);
    std::cout << "After uppercase:" << std::endl;
    printAllStrings(words);
    
    std::cout << "\n=== FACTORY PATTERN ===" << std::endl;
    auto range = createRange(1, 6);
    for (int n : range) std::cout << n << " ";
    std::cout << std::endl;
    
    std::cout << "\n=== LOCAL MODIFICATION ===" << std::endl;
    std::string text = "Hello World Program";
    std::cout << "Original: " << text << std::endl;
    std::string noSpaces = removeSpaces(text);
    std::cout << "No spaces: " << noSpaces << std::endl;
    std::cout << "Original unchanged: " << text << std::endl;
    
    return 0;
}
