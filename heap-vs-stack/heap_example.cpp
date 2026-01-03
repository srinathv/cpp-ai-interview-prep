#include <iostream>
#include <memory>

void heapAllocation() {
    std::cout << "=== Heap Allocation Example ===" << std::endl;
    std::unique_ptr<int[]> largeArray(new int[1000000]);
    largeArray[0] = 42;
    largeArray[999999] = 100;
    std::cout << "Large heap allocation succeeded!" << std::endl;
    std::cout << "First element: " << largeArray[0] << std::endl;
    std::cout << "Last element: " << largeArray[999999] << std::endl;
}

void deepRecursionWithHeap(int depth, int maxDepth) {
    auto data = std::make_unique<int>(depth);
    if (depth % 1000 == 0) {
        std::cout << "Recursion depth: " << depth << std::endl;
    }
    if (depth < maxDepth) {
        deepRecursionWithHeap(depth + 1, maxDepth);
    }
}

int main() {
    heapAllocation();
    std::cout << "\n=== Deep Recursion with Heap Storage ===" << std::endl;
    deepRecursionWithHeap(0, 10000);
    std::cout << "Completed 10,000 recursive calls successfully!" << std::endl;
    return 0;
}
