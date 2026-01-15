#include <iostream>
#include <string>
#include <vector>

// PASS BY VALUE - Creates a copy
void passByValue(int x) {
    x = 100;
    std::cout << "Inside passByValue: " << x << std::endl;
}

int square(int num) {
    return num * num;
}

// PASS BY REFERENCE - No copy, direct access to original
void passByReference(int& x) {
    x = 100;
    std::cout << "Inside passByReference: " << x << std::endl;
}

void modifyString(std::string& str) {
    str += " - modified";
}

// PASS BY CONST REFERENCE - Read-only access, no copy
void printVector(const std::vector<int>& vec) {
    for (int val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

int sumVector(const std::vector<int>& vec) {
    int sum = 0;
    for (int val : vec) {
        sum += val;
    }
    return sum;
}

void displayString(const std::string& str) {
    std::cout << "String: " << str << std::endl;
}

int main() {
    std::cout << "=== PASS BY VALUE ===" << std::endl;
    int a = 50;
    std::cout << "Before: " << a << std::endl;
    passByValue(a);
    std::cout << "After: " << a << std::endl;
    std::cout << "Square of 5: " << square(5) << std::endl;
    
    std::cout << "\n=== PASS BY REFERENCE ===" << std::endl;
    int b = 50;
    std::cout << "Before: " << b << std::endl;
    passByReference(b);
    std::cout << "After: " << b << std::endl;
    
    std::string text = "Hello";
    std::cout << "Before: " << text << std::endl;
    modifyString(text);
    std::cout << "After: " << text << std::endl;
    
    std::cout << "\n=== PASS BY CONST REFERENCE ===" << std::endl;
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    printVector(numbers);
    std::cout << "Sum: " << sumVector(numbers) << std::endl;
    displayString("Efficient string passing");
    
    return 0;
}
