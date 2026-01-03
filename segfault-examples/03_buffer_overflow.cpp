#include <iostream>
#include <vector>
#include <string.h>

// Example 1: Array out-of-bounds write
void arrayOverflow() {
    std::cout << "\n=== Array Buffer Overflow ===" << std::endl;
    
    int arr[5] = {1, 2, 3, 4, 5};
    std::cout << "Array of 5 elements" << std::endl;
    
    // CRASH: Writing way past array bounds
    std::cout << "Writing to arr[100]..." << std::endl;
    arr[100] = 42;
    
    std::cout << "This may or may not crash immediately" << std::endl;
}

// Example 2: Vector unchecked access
void vectorUncheckedAccess() {
    std::cout << "\n=== Vector Unchecked Access ===" << std::endl;
    
    std::vector<int> vec = {1, 2, 3};
    std::cout << "Vector with 3 elements" << std::endl;
    
    // CRASH: operator[] doesn't bounds-check!
    std::cout << "Accessing vec[10]..." << std::endl;
    vec[10] = 42;
    
    std::cout << "Undefined behavior" << std::endl;
}

// Example 3: String buffer overflow
void stringBufferOverflow() {
    std::cout << "\n=== String Buffer Overflow ===" << std::endl;
    
    char buffer[10];
    const char* longString = "This is a very long string that will overflow";
    
    std::cout << "Copying long string to small buffer..." << std::endl;
    
    // CRASH: strcpy doesn't check bounds!
    strcpy(buffer, longString);
    
    std::cout << "Buffer: " << buffer << std::endl;
}

// Example 4: Off-by-one error
void offByOne() {
    std::cout << "\n=== Off-By-One Error ===" << std::endl;
    
    int arr[10];
    std::cout << "Filling array of 10 elements..." << std::endl;
    
    // CRASH: Loop goes one past the end
    for (int i = 0; i <= 10; i++) {  // Should be i < 10
        arr[i] = i;
    }
    
    std::cout << "Off-by-one can cause corruption or crash" << std::endl;
}

// Example 5: Negative index
void negativeIndex() {
    std::cout << "\n=== Negative Index ===" << std::endl;
    
    int arr[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::cout << "Accessing arr[-5]..." << std::endl;
    
    // CRASH: Negative index accesses memory before array
    std::cout << "Value: " << arr[-5] << std::endl;
}

// Example 6: Stack buffer overflow
void stackBufferOverflow() {
    std::cout << "\n=== Stack Buffer Overflow ===" << std::endl;
    
    char buffer[8];
    const char* input = "VeryLongInputString";
    
    std::cout << "Using unsafe strcpy..." << std::endl;
    
    // CRASH: Classic stack buffer overflow
    strcpy(buffer, input);
}

// SAFE ALTERNATIVES
void safeAlternatives() {
    std::cout << "\n=== SAFE ALTERNATIVES ===" << std::endl;
    
    // 1. Vector with at() for bounds checking
    {
        std::cout << "\n1. Vector with at():" << std::endl;
        std::vector<int> vec = {1, 2, 3};
        
        try {
            vec.at(10) = 42;  // Throws exception instead of crashing
        } catch (const std::out_of_range& e) {
            std::cout << "  Caught exception: " << e.what() << std::endl;
        }
    }
    
    // 2. Use std::string instead of char arrays
    {
        std::cout << "\n2. Using std::string:" << std::endl;
        std::string str = "Hello";
        str += " World!";  // Safe concatenation
        std::cout << "  String: " << str << std::endl;
    }
    
    // 3. Use std::array with bounds checking
    {
        std::cout << "\n3. Using std::array:" << std::endl;
        std::array<int, 5> arr = {1, 2, 3, 4, 5};
        
        try {
            arr.at(10) = 42;  // Bounds checked
        } catch (const std::out_of_range& e) {
            std::cout << "  Caught exception: " << e.what() << std::endl;
        }
    }
    
    // 4. Safe string copy
    {
        std::cout << "\n4. Safe string copy:" << std::endl;
        char buffer[10];
        const char* source = "VeryLongString";
        
        strncpy(buffer, source, sizeof(buffer) - 1);
        buffer[sizeof(buffer) - 1] = '\0';
        
        std::cout << "  Buffer (truncated): " << buffer << std::endl;
    }
    
    // 5. Range-based loops
    {
        std::cout << "\n5. Range-based loops (can't overflow):" << std::endl;
        std::vector<int> vec = {1, 2, 3, 4, 5};
        
        for (int val : vec) {
            std::cout << "  " << val;
        }
        std::cout << std::endl;
    }
    
    // 6. Iterators with bounds checking
    {
        std::cout << "\n6. Checked iteration:" << std::endl;
        std::vector<int> vec = {1, 2, 3};
        
        for (size_t i = 0; i < vec.size(); i++) {
            std::cout << "  vec[" << i << "] = " << vec[i] << std::endl;
        }
    }
}

int main() {
    std::cout << "=== Buffer Overflow Segfault Examples ===" << std::endl;
    std::cout << "Uncomment ONE example to see the crash\n" << std::endl;
    
    // UNCOMMENT ONE TO SEE CRASH:
    // arrayOverflow();
    // vectorUncheckedAccess();
    // stringBufferOverflow();
    // offByOne();
    // negativeIndex();
    // stackBufferOverflow();
    
    // This is safe to run:
    safeAlternatives();
    
    std::cout << "\nProgram completed safely" << std::endl;
    std::cout << "Tip: Compile with -fsanitize=address to catch these bugs!" << std::endl;
    
    return 0;
}
