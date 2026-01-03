#include <iostream>

void blowStack(int depth) {
    std::cout << "Recursion depth: " << depth << std::endl;
    blowStack(depth + 1);
}

void largeStackAllocation() {
    int largeArray[1000000];
    largeArray[0] = 42;
    std::cout << "Large stack allocation succeeded: " << largeArray[0] << std::endl;
}

int main() {
    std::cout << "=== Stack Overflow Example ===" << std::endl;
    std::cout << "Warning: Uncomment blowStack(0) or largeStackAllocation() to crash!" << std::endl;
    return 0;
}
