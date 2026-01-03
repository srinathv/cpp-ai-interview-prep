#include <iostream>
#include <iomanip>

void printAddress(const char* label, void* addr) {
    std::cout << std::setw(30) << label << ": 0x" 
              << std::hex << reinterpret_cast<uintptr_t>(addr) << std::dec << std::endl;
}

// Example 1: Stack variable lifecycle
void stackLifecycle() {
    std::cout << "\n=== Stack Variable Lifecycle ===" << std::endl;
    
    int x = 10;
    printAddress("Stack var 'x'", &x);
    std::cout << "Value of x after allocation: " << x << std::endl;
    
    x = 20;
    std::cout << "Value of x after modification: " << x << std::endl;
    
    // x goes out of scope and is automatically freed when function returns
    std::cout << "x will be freed when function returns" << std::endl;
}

// Example 2: Heap variable lifecycle
void heapLifecycle() {
    std::cout << "\n=== Heap Variable Lifecycle ===" << std::endl;
    
    int* ptr = new int(10);
    printAddress("Heap var 'ptr' (ptr itself)", &ptr);
    printAddress("Heap var (data pointed to)", ptr);
    std::cout << "Value after allocation: " << *ptr << std::endl;
    
    *ptr = 20;
    std::cout << "Value after modification: " << *ptr << std::endl;
    
    delete ptr;
    std::cout << "Memory freed with delete" << std::endl;
    
    // DANGER: Use after free
    std::cout << "\nDANGER - Use after free:" << std::endl;
    std::cout << "Value after delete (undefined behavior): " << *ptr << std::endl;
    
    ptr = nullptr;
    std::cout << "Pointer set to nullptr for safety" << std::endl;
}

// Example 3: Function scope demonstration
int* dangerousFunction() {
    int localVar = 42;
    printAddress("Local var in function", &localVar);
    return &localVar;  // DANGER: Returning address of stack variable
}

void scopeExample() {
    std::cout << "\n=== Function Scope Example ===" << std::endl;
    
    int* dangerousPtr = dangerousFunction();
    std::cout << "DANGER - Using returned stack address:" << std::endl;
    printAddress("Returned pointer", dangerousPtr);
    std::cout << "Value (undefined behavior): " << *dangerousPtr << std::endl;
}

// Example 4: Multiple allocations showing stack growth
void stackGrowth(int depth) {
    if (depth > 5) return;
    
    int localVar = depth;
    printAddress(("Stack var at depth " + std::to_string(depth)).c_str(), &localVar);
    std::cout << "Value: " << localVar << std::endl;
    
    stackGrowth(depth + 1);
    
    std::cout << "Returning from depth " << depth 
              << ", value still: " << localVar << std::endl;
}

void demonstrateStackGrowth() {
    std::cout << "\n=== Stack Growth (Recursion) ===" << std::endl;
    stackGrowth(0);
}

// Example 5: Memory leak demonstration
void memoryLeak() {
    std::cout << "\n=== Memory Leak Example ===" << std::endl;
    
    for (int i = 0; i < 3; i++) {
        int* leak = new int(i);
        printAddress(("Leaked allocation " + std::to_string(i)).c_str(), leak);
        std::cout << "Value: " << *leak << std::endl;
        // Missing delete! Memory is leaked
    }
    std::cout << "Function ends - all 3 allocations leaked!" << std::endl;
}

// Example 6: Proper heap usage with RAII
class Resource {
    int* data;
public:
    Resource(int val) : data(new int(val)) {
        std::cout << "Resource allocated: " << *data;
        printAddress(" at", data);
    }
    
    ~Resource() {
        std::cout << "Resource freed: " << *data;
        printAddress(" at", data);
        delete data;
    }
    
    void use() {
        std::cout << "Using resource: " << *data << std::endl;
    }
};

void raiiExample() {
    std::cout << "\n=== RAII (Proper Resource Management) ===" << std::endl;
    
    Resource r1(100);
    r1.use();
    
    {
        Resource r2(200);
        r2.use();
        std::cout << "r2 scope ending..." << std::endl;
    }  // r2 automatically freed here
    
    std::cout << "r1 still valid..." << std::endl;
    r1.use();
    
    std::cout << "Function ending, r1 will be freed..." << std::endl;
}

int main() {
    std::cout << "=== Memory Allocation and Usage Lifecycle ===" << std::endl;
    
    stackLifecycle();
    heapLifecycle();
    scopeExample();
    demonstrateStackGrowth();
    memoryLeak();
    raiiExample();
    
    std::cout << "\nProgram ending - all stack variables automatically freed" << std::endl;
    std::cout << "Heap leaks remain until program terminates!" << std::endl;
    
    return 0;
}
