#include <iostream>
#include <vector>
#include <string>
#include <cmath>

// Large custom class to demonstrate copy overhead
class LargeObject {
private:
    std::vector<int> data;
    std::string name;
    
public:
    LargeObject(int size) : name("LargeObject") {
        data.resize(size, 42);
        std::cout << "Constructor called (size: " << size << ")" << std::endl;
    }
    
    // Copy constructor - expensive!
    LargeObject(const LargeObject& other) 
        : data(other.data), name(other.name) {
        std::cout << "Copy constructor called (EXPENSIVE!)" << std::endl;
    }
    
    int getSize() const { return data.size(); }
};

// BAD: Pass by value - creates expensive copy
void processByValue(LargeObject obj) {
    std::cout << "Processing object of size: " << obj.getSize() << std::endl;
}

// GOOD: Pass by const reference - no copy
void processByConstRef(const LargeObject& obj) {
    std::cout << "Processing object of size: " << obj.getSize() << std::endl;
}

// GOOD: Pass by reference when modifying
void modifyByReference(std::vector<int>& vec) {
    for (int& val : vec) {
        val *= 2;
    }
}

// Demonstrating when value is okay
struct Point {
    double x, y;
};

// Small POD types - value is fine
double distance(Point p1, Point p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// But even here, const ref works well
double distanceRef(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

int main() {
    std::cout << "=== PERFORMANCE IMPACT ===" << std::endl;
    
    std::cout << "\n--- Creating LargeObject ---" << std::endl;
    LargeObject large(1000000);
    
    std::cout << "\n--- Pass by VALUE (watch for copy!) ---" << std::endl;
    processByValue(large);
    
    std::cout << "\n--- Pass by CONST REFERENCE (no copy) ---" << std::endl;
    processByConstRef(large);
    
    std::cout << "\n=== MODIFYING COLLECTIONS ===" << std::endl;
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    std::cout << "Before: ";
    for (int n : numbers) std::cout << n << " ";
    std::cout << std::endl;
    
    modifyByReference(numbers);
    std::cout << "After: ";
    for (int n : numbers) std::cout << n << " ";
    std::cout << std::endl;
    
    std::cout << "\n=== SMALL TYPES (both work) ===" << std::endl;
    Point a{0, 0}, b{3, 4};
    std::cout << "Distance: " << distance(a, b) << std::endl;
    std::cout << "Distance (ref): " << distanceRef(a, b) << std::endl;
    
    return 0;
}
