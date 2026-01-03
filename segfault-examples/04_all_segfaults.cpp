#include <iostream>
#include <vector>
#include <memory>

// 1. Null Pointer Dereference
void nullPointer() {
    std::cout << "\n=== 1. Null Pointer Dereference ===" << std::endl;
    int* ptr = nullptr;
    *ptr = 42;  // CRASH
}

// 2. Use-After-Free
void useAfterFree() {
    std::cout << "\n=== 2. Use-After-Free ===" << std::endl;
    int* ptr = new int(42);
    delete ptr;
    *ptr = 100;  // CRASH
}

// 3. Dangling Pointer
int* danglingPointer() {
    int local = 42;
    return &local;  // Returns address of destroyed variable
}

void useDangling() {
    std::cout << "\n=== 3. Dangling Pointer ===" << std::endl;
    int* ptr = danglingPointer();
    std::cout << *ptr << std::endl;  // CRASH
}

// 4. Buffer Overflow
void bufferOverflow() {
    std::cout << "\n=== 4. Buffer Overflow ===" << std::endl;
    int arr[10];
    arr[100] = 42;  // CRASH
}

// 5. Stack Overflow
void stackOverflow(int depth = 0) {
    std::cout << "Depth: " << depth << std::endl;
    stackOverflow(depth + 1);  // CRASH: infinite recursion
}

// 6. Uninitialized Pointer
void uninitializedPointer() {
    std::cout << "\n=== 6. Uninitialized Pointer ===" << std::endl;
    int* ptr;  // Garbage value
    *ptr = 42;  // CRASH: random address
}

// 7. Double Free
void doubleFree() {
    std::cout << "\n=== 7. Double Free ===" << std::endl;
    int* ptr = new int(42);
    delete ptr;
    delete ptr;  // CRASH: freeing already freed memory
}

// 8. Write to Read-Only Memory
void writeReadOnly() {
    std::cout << "\n=== 8. Write to Read-Only Memory ===" << std::endl;
    char* str = (char*)"Hello";  // String literal is read-only
    str[0] = 'h';  // CRASH
}

// 9. Virtual Function on Deleted Object
class Base {
public:
    virtual void method() { std::cout << "Base" << std::endl; }
    virtual ~Base() {}
};

void deletedVirtual() {
    std::cout << "\n=== 9. Virtual Function on Deleted Object ===" << std::endl;
    Base* obj = new Base();
    delete obj;
    obj->method();  // CRASH: vtable is destroyed
}

// 10. Iterator Invalidation
void iteratorInvalidation() {
    std::cout << "\n=== 10. Iterator Invalidation ===" << std::endl;
    std::vector<int> vec = {1, 2, 3};
    auto it = vec.begin();
    vec.push_back(4);
    vec.push_back(5);  // May reallocate
    std::cout << *it << std::endl;  // CRASH: iterator invalidated
}

int main(int argc, char* argv[]) {
    std::cout << "=== All Segfault Examples ===" << std::endl;
    std::cout << "Usage: " << argv[0] << " <number 1-10>" << std::endl;
    std::cout << "\n1.  Null Pointer Dereference" << std::endl;
    std::cout << "2.  Use-After-Free" << std::endl;
    std::cout << "3.  Dangling Pointer" << std::endl;
    std::cout << "4.  Buffer Overflow" << std::endl;
    std::cout << "5.  Stack Overflow (infinite recursion)" << std::endl;
    std::cout << "6.  Uninitialized Pointer" << std::endl;
    std::cout << "7.  Double Free" << std::endl;
    std::cout << "8.  Write to Read-Only Memory" << std::endl;
    std::cout << "9.  Virtual Function on Deleted Object" << std::endl;
    std::cout << "10. Iterator Invalidation" << std::endl;
    
    if (argc != 2) {
        std::cout << "\nNo crash selected. Exiting safely." << std::endl;
        return 0;
    }
    
    int choice = std::atoi(argv[1]);
    
    switch (choice) {
        case 1: nullPointer(); break;
        case 2: useAfterFree(); break;
        case 3: useDangling(); break;
        case 4: bufferOverflow(); break;
        case 5: stackOverflow(); break;
        case 6: uninitializedPointer(); break;
        case 7: doubleFree(); break;
        case 8: writeReadOnly(); break;
        case 9: deletedVirtual(); break;
        case 10: iteratorInvalidation(); break;
        default:
            std::cout << "Invalid choice" << std::endl;
    }
    
    return 0;
}
