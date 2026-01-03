#include <iostream>
#include <memory>

// Example 1: Classic use-after-free
void useAfterFree() {
    std::cout << "\n=== Use After Free ===" << std::endl;
    
    int* ptr = new int(42);
    std::cout << "Allocated: *ptr = " << *ptr << std::endl;
    
    delete ptr;
    std::cout << "Deleted ptr" << std::endl;
    
    // CRASH: Using freed memory
    std::cout << "Attempting to use freed memory..." << std::endl;
    *ptr = 100;
    std::cout << "Value after free: " << *ptr << std::endl;
}

// Example 2: Dangling pointer from stack
int* danglingStackPointer() {
    int local = 42;
    return &local;  // Returns address of destroyed stack variable
}

void useDanglingStack() {
    std::cout << "\n=== Dangling Stack Pointer ===" << std::endl;
    
    int* ptr = danglingStackPointer();
    std::cout << "Got pointer to destroyed stack variable" << std::endl;
    
    // CRASH: Accessing destroyed stack memory
    std::cout << "Attempting to use: " << *ptr << std::endl;
}

// Example 3: Reference to destroyed object
class MyClass {
public:
    int value;
    MyClass(int v) : value(v) {
        std::cout << "  Constructor: value = " << value << std::endl;
    }
    ~MyClass() {
        std::cout << "  Destructor: value was " << value << std::endl;
    }
};

MyClass& returnReference() {
    MyClass local(42);
    return local;  // Returns reference to object that's about to be destroyed
}

void useDanglingReference() {
    std::cout << "\n=== Dangling Reference ===" << std::endl;
    
    MyClass& ref = returnReference();
    std::cout << "Got reference to destroyed object" << std::endl;
    
    // CRASH: Object is destroyed
    std::cout << "Attempting to access: " << ref.value << std::endl;
}

// Example 4: Iterator invalidation
#include <vector>

void iteratorInvalidation() {
    std::cout << "\n=== Iterator Invalidation ===" << std::endl;
    
    std::vector<int> vec = {1, 2, 3, 4, 5};
    auto it = vec.begin();
    
    std::cout << "Iterator points to: " << *it << std::endl;
    
    // This invalidates all iterators
    vec.push_back(6);
    vec.push_back(7);
    vec.push_back(8);  // May cause reallocation
    
    // CRASH: Iterator is now invalid
    std::cout << "Attempting to use invalidated iterator..." << std::endl;
    std::cout << *it << std::endl;
}

// Example 5: Double use after free
void doubleUseAfterFree() {
    std::cout << "\n=== Multiple Uses After Free ===" << std::endl;
    
    int* ptr = new int(42);
    delete ptr;
    
    // CRASH: First use after free
    *ptr = 100;
    
    // CRASH: Second use after free
    std::cout << *ptr << std::endl;
}

// SAFE ALTERNATIVES
void safeAlternatives() {
    std::cout << "\n=== SAFE ALTERNATIVES ===" << std::endl;
    
    // 1. Smart pointers prevent use-after-free
    {
        std::cout << "\n1. Using unique_ptr:" << std::endl;
        std::unique_ptr<int> ptr = std::make_unique<int>(42);
        std::cout << "  Value: " << *ptr << std::endl;
        
        // After ptr is destroyed, you can't access it anymore
        // (compile-time safety with move semantics)
    }
    
    // 2. Shared ownership with shared_ptr
    {
        std::cout << "\n2. Using shared_ptr:" << std::endl;
        std::shared_ptr<int> ptr1 = std::make_shared<int>(42);
        {
            std::shared_ptr<int> ptr2 = ptr1;
            std::cout << "  ptr2 value: " << *ptr2 << std::endl;
        }  // ptr2 destroyed, but memory still valid
        std::cout << "  ptr1 still valid: " << *ptr1 << std::endl;
    }
    
    // 3. Use values instead of pointers when possible
    {
        std::cout << "\n3. Using values:" << std::endl;
        MyClass obj(100);
        std::cout << "  Direct value access: " << obj.value << std::endl;
        // Automatic cleanup, no pointers needed
    }
    
    // 4. RAII pattern
    {
        std::cout << "\n4. RAII with unique_ptr:" << std::endl;
        auto obj = std::make_unique<MyClass>(200);
        std::cout << "  Object value: " << obj->value << std::endl;
    }  // Automatic destruction
    
    std::cout << "\nAll memory safely managed!" << std::endl;
}

int main() {
    std::cout << "=== Use-After-Free Segfault Examples ===" << std::endl;
    std::cout << "Uncomment ONE example to see the crash\n" << std::endl;
    
    // UNCOMMENT ONE TO SEE CRASH:
    // useAfterFree();
    // useDanglingStack();
    // useDanglingReference();
    // iteratorInvalidation();
    // doubleUseAfterFree();
    
    // This is safe to run:
    safeAlternatives();
    
    std::cout << "\nProgram completed safely" << std::endl;
    
    return 0;
}
