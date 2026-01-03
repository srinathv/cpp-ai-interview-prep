#include <iostream>

// Example 1: Direct null pointer dereference
void nullPointerDereference() {
    std::cout << "\n=== Null Pointer Dereference ===" << std::endl;
    std::cout << "Creating nullptr and attempting to dereference..." << std::endl;
    
    int* ptr = nullptr;
    std::cout << "ptr = nullptr" << std::endl;
    std::cout << "Attempting: *ptr = 42" << std::endl;
    
    // CRASH: Dereferencing null pointer
    *ptr = 42;
    
    std::cout << "This line will never execute" << std::endl;
}

// Example 2: Null pointer in function
int* returnNull() {
    return nullptr;
}

void nullFromFunction() {
    std::cout << "\n=== Null from Function ===" << std::endl;
    std::cout << "Getting nullptr from function..." << std::endl;
    
    int* ptr = returnNull();
    std::cout << "Attempting to use returned pointer..." << std::endl;
    
    // CRASH: Using null pointer
    *ptr = 100;
}

// Example 3: Conditional null check forgotten
void forgottenNullCheck(int* ptr) {
    std::cout << "\n=== Forgotten Null Check ===" << std::endl;
    std::cout << "Function expects valid pointer but got nullptr..." << std::endl;
    
    // Should check: if (ptr == nullptr) return;
    // CRASH: No null check
    *ptr = 42;
}

// Example 4: Array access on null
void nullArrayAccess() {
    std::cout << "\n=== Null Array Access ===" << std::endl;
    
    int* arr = nullptr;
    std::cout << "Attempting to access arr[0]..." << std::endl;
    
    // CRASH: Array indexing on null
    arr[0] = 10;
}

// Safe alternative: Using references and optional
#include <optional>

void safeVersion() {
    std::cout << "\n=== SAFE VERSION ===" << std::endl;
    
    // Option 1: Use references (can't be null)
    int value = 42;
    int& ref = value;
    ref = 100;  // Safe, references can't be null
    std::cout << "Using reference (can't be null): " << ref << std::endl;
    
    // Option 2: Use std::optional
    std::optional<int> opt = std::nullopt;
    if (opt.has_value()) {
        std::cout << "Value: " << *opt << std::endl;
    } else {
        std::cout << "Optional is empty (safe check)" << std::endl;
    }
    
    opt = 42;
    if (opt) {
        std::cout << "Value after assignment: " << *opt << std::endl;
    }
}

int main() {
    std::cout << "=== Null Pointer Segfault Examples ===" << std::endl;
    std::cout << "Uncomment ONE example to see the crash\n" << std::endl;
    
    // UNCOMMENT ONE TO SEE CRASH:
    // nullPointerDereference();
    // nullFromFunction();
    // forgottenNullCheck(nullptr);
    // nullArrayAccess();
    
    // This is safe to run:
    safeVersion();
    
    std::cout << "\nProgram completed safely" << std::endl;
    std::cout << "To see crashes, uncomment one of the dangerous functions" << std::endl;
    
    return 0;
}
