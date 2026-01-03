#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>

// Helper to print memory addresses
void printAddr(const char* label, void* addr) {
    std::cout << "  " << std::setw(40) << label << ": 0x" 
              << std::hex << reinterpret_cast<uintptr_t>(addr) << std::dec << std::endl;
}

class DataObject {
    int id;
    char buffer[100];
public:
    DataObject(int i) : id(i) {
        std::cout << "  Constructor called for object " << id << std::endl;
    }
    ~DataObject() {
        std::cout << "  Destructor called for object " << id << std::endl;
    }
    int getId() const { return id; }
};

// Example 1: Stack allocation - use and cleanup
void stackAllocationExample() {
    std::cout << "\n=== 1. Stack Allocation - Allocation, Use, Deallocation ===" << std::endl;
    
    std::cout << "Before allocation:" << std::endl;
    
    int x = 42;
    std::cout << "After allocation:" << std::endl;
    printAddr("Variable x (stack)", &x);
    
    std::cout << "Using variable:" << std::endl;
    std::cout << "  x = " << x << std::endl;
    x = 100;
    std::cout << "  x after modification = " << x << std::endl;
    
    DataObject obj(1);
    printAddr("Object on stack", &obj);
    std::cout << "Using object: id = " << obj.getId() << std::endl;
    
    std::cout << "End of function - automatic cleanup..." << std::endl;
}

// Example 2: Heap allocation - manual memory management
void heapAllocationExample() {
    std::cout << "\n=== 2. Heap Allocation - Manual Memory Management ===" << std::endl;
    
    std::cout << "Before allocation:" << std::endl;
    
    int* ptr = new int(42);
    std::cout << "After allocation:" << std::endl;
    printAddr("Pointer 'ptr' itself (stack)", &ptr);
    printAddr("Data pointed to (heap)", ptr);
    
    std::cout << "Using heap variable:" << std::endl;
    std::cout << "  *ptr = " << *ptr << std::endl;
    *ptr = 100;
    std::cout << "  *ptr after modification = " << *ptr << std::endl;
    
    DataObject* objPtr = new DataObject(2);
    printAddr("Object pointer (stack)", &objPtr);
    printAddr("Object on heap", objPtr);
    std::cout << "Using heap object: id = " << objPtr->getId() << std::endl;
    
    std::cout << "Manual cleanup:" << std::endl;
    delete ptr;
    std::cout << "  Deleted int" << std::endl;
    delete objPtr;
    
    std::cout << "End of function - stack cleaned, heap manually freed" << std::endl;
}

// Example 3: Smart pointers - automatic heap cleanup
void smartPointerExample() {
    std::cout << "\n=== 3. Smart Pointers - Automatic Heap Management ===" << std::endl;
    
    std::cout << "Creating unique_ptr:" << std::endl;
    auto ptr = std::make_unique<int>(42);
    printAddr("unique_ptr (stack)", &ptr);
    printAddr("Data (heap)", ptr.get());
    
    std::cout << "Using unique_ptr:" << std::endl;
    std::cout << "  *ptr = " << *ptr << std::endl;
    *ptr = 100;
    std::cout << "  *ptr after modification = " << *ptr << std::endl;
    
    std::cout << "Creating unique_ptr to object:" << std::endl;
    auto objPtr = std::make_unique<DataObject>(3);
    std::cout << "Using object: id = " << objPtr->getId() << std::endl;
    
    std::cout << "End of function - automatic cleanup..." << std::endl;
}

// Example 4: Function call stack demonstration
void functionC() {
    std::cout << "  In functionC" << std::endl;
    int c = 3;
    printAddr("Variable c in functionC", &c);
    std::cout << "  functionC returning..." << std::endl;
}

void functionB() {
    std::cout << "  In functionB" << std::endl;
    int b = 2;
    printAddr("Variable b in functionB", &b);
    functionC();
    std::cout << "  Back in functionB, b still valid: " << b << std::endl;
    std::cout << "  functionB returning..." << std::endl;
}

void functionA() {
    std::cout << "  In functionA" << std::endl;
    int a = 1;
    printAddr("Variable a in functionA", &a);
    functionB();
    std::cout << "  Back in functionA, a still valid: " << a << std::endl;
    std::cout << "  functionA returning..." << std::endl;
}

void functionCallExample() {
    std::cout << "\n=== 4. Function Call Stack ===" << std::endl;
    std::cout << "Calling nested functions (watch stack addresses):" << std::endl;
    functionA();
    std::cout << "All functions returned, stack unwound" << std::endl;
}

// Example 5: Array allocation comparison
void arrayAllocationExample() {
    std::cout << "\n=== 5. Array Allocation Comparison ===" << std::endl;
    
    std::cout << "Stack array:" << std::endl;
    int stackArray[5] = {1, 2, 3, 4, 5};
    printAddr("stackArray (stack)", stackArray);
    std::cout << "  Using: stackArray[0] = " << stackArray[0] << std::endl;
    stackArray[0] = 99;
    std::cout << "  After modification: stackArray[0] = " << stackArray[0] << std::endl;
    
    std::cout << "\nHeap array:" << std::endl;
    int* heapArray = new int[5]{1, 2, 3, 4, 5};
    printAddr("heapArray pointer (stack)", &heapArray);
    printAddr("heapArray data (heap)", heapArray);
    std::cout << "  Using: heapArray[0] = " << heapArray[0] << std::endl;
    heapArray[0] = 99;
    std::cout << "  After modification: heapArray[0] = " << heapArray[0] << std::endl;
    
    std::cout << "\nVector (heap-managed):" << std::endl;
    std::vector<int> vec = {1, 2, 3, 4, 5};
    printAddr("vector object (stack)", &vec);
    printAddr("vector data (heap)", vec.data());
    std::cout << "  Using: vec[0] = " << vec[0] << std::endl;
    vec[0] = 99;
    std::cout << "  After modification: vec[0] = " << vec[0] << std::endl;
    
    std::cout << "\nCleanup:" << std::endl;
    delete[] heapArray;
    std::cout << "  Heap array deleted manually" << std::endl;
    std::cout << "  Stack array and vector auto-cleaned at scope end" << std::endl;
}

// Example 6: Memory reuse demonstration
void memoryReuseExample() {
    std::cout << "\n=== 6. Memory Reuse Patterns ===" << std::endl;
    
    void* addr1;
    void* addr2;
    
    {
        std::cout << "First scope:" << std::endl;
        int x = 10;
        addr1 = &x;
        printAddr("Variable x in first scope", &x);
        std::cout << "  x = " << x << std::endl;
    }
    
    {
        std::cout << "\nSecond scope (stack reused):" << std::endl;
        int y = 20;
        addr2 = &y;
        printAddr("Variable y in second scope", &y);
        std::cout << "  y = " << y << std::endl;
        
        if (addr1 == addr2) {
            std::cout << "  Notice: Same stack location reused!" << std::endl;
        }
    }
    
    std::cout << "\nHeap doesn't reuse like stack:" << std::endl;
    int* p1 = new int(10);
    printAddr("First heap allocation", p1);
    delete p1;
    
    int* p2 = new int(20);
    printAddr("Second heap allocation", p2);
    std::cout << "  Heap addresses may differ" << std::endl;
    delete p2;
}

int main() {
    std::cout << "=== Comprehensive Memory Usage Patterns ===" << std::endl;
    std::cout << "Demonstrating allocation -> use -> deallocation for each pattern\n" << std::endl;
    
    stackAllocationExample();
    heapAllocationExample();
    smartPointerExample();
    functionCallExample();
    arrayAllocationExample();
    memoryReuseExample();
    
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Stack: Allocate -> Use -> Auto-cleanup (LIFO)" << std::endl;
    std::cout << "Heap:  Allocate -> Use -> Manual cleanup (or smart pointers)" << std::endl;
    std::cout << "       Heap persists across function calls until freed" << std::endl;
    
    return 0;
}
