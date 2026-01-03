#include <iostream>
#include <chrono>

class LargeObject {
public:
    int data[1000];
    LargeObject() { data[0] = 42; }
};

void stackAllocation() {
    auto start = std::chrono::high_resolution_clock::now();
    LargeObject obj;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "Stack allocation time: " << duration.count() << " ns" << std::endl;
    std::cout << "Object data: " << obj.data[0] << std::endl;
}

void heapAllocation() {
    auto start = std::chrono::high_resolution_clock::now();
    LargeObject* obj = new LargeObject();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "Heap allocation time: " << duration.count() << " ns" << std::endl;
    std::cout << "Object data: " << obj->data[0] << std::endl;
    delete obj;
}

int main() {
    std::cout << "=== Stack vs Heap Comparison ===" << std::endl;
    std::cout << "\nKey Differences:" << std::endl;
    std::cout << "1. Stack: Fast, automatic cleanup, limited size" << std::endl;
    std::cout << "2. Heap: Slower, manual cleanup, large size" << std::endl;
    std::cout << "\nPerformance comparison:\n" << std::endl;
    stackAllocation();
    heapAllocation();
    return 0;
}
