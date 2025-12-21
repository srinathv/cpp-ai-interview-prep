/*
 * Move Semantics in Modern C++
 * 
 * Key Concepts:
 * - Move constructors and move assignment operators
 * - std::move and rvalue references
 * - Perfect forwarding with std::forward
 * - The Rule of Five
 * 
 * Interview Topics:
 * - When does the compiler generate move constructors?
 * - What's the difference between std::move and std::forward?
 * - How to implement move semantics for custom classes?
 */

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <utility>

// Example 1: Basic Move Semantics
class Buffer {
private:
    int* data;
    size_t size;

public:
    // Constructor
    Buffer(size_t sz) : size(sz), data(new int[sz]) {
        std::cout << "Constructor: Allocated " << size << " ints\n";
    }

    // Destructor
    ~Buffer() {
        delete[] data;
        std::cout << "Destructor: Freed memory\n";
    }

    // Copy constructor (deep copy)
    Buffer(const Buffer& other) : size(other.size), data(new int[other.size]) {
        std::copy(other.data, other.data + size, data);
        std::cout << "Copy Constructor: Deep copy of " << size << " ints\n";
    }

    // Copy assignment operator
    Buffer& operator=(const Buffer& other) {
        if (this != &other) {
            delete[] data;
            size = other.size;
            data = new int[size];
            std::copy(other.data, other.data + size, data);
            std::cout << "Copy Assignment: Deep copy of " << size << " ints\n";
        }
        return *this;
    }

    // Move constructor (transfer ownership)
    Buffer(Buffer&& other) noexcept : size(other.size), data(other.data) {
        other.data = nullptr;
        other.size = 0;
        std::cout << "Move Constructor: Transferred ownership\n";
    }

    // Move assignment operator
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
            std::cout << "Move Assignment: Transferred ownership\n";
        }
        return *this;
    }

    size_t getSize() const { return size; }
};

// Example 2: std::move in practice
void demonstrateStdMove() {
    std::cout << "\n=== std::move Demo ===\n";
    
    std::string str1 = "Hello, World!";
    std::string str2 = std::move(str1); // str1 is now in valid but unspecified state
    
    std::cout << "str2: " << str2 << "\n";
    std::cout << "str1 after move: \"" << str1 << "\" (empty)\n";
    
    // Moving vectors avoids copying all elements
    std::vector<int> vec1 = {1, 2, 3, 4, 5};
    std::vector<int> vec2 = std::move(vec1); // vec1's internal buffer transferred
    
    std::cout << "vec2 size: " << vec2.size() << "\n";
    std::cout << "vec1 size after move: " << vec1.size() << "\n";
}

// Example 3: Move semantics with smart pointers
class Resource {
private:
    std::unique_ptr<int[]> data;
    size_t size;

public:
    Resource(size_t sz) : data(std::make_unique<int[]>(sz)), size(sz) {
        std::cout << "Resource created with size " << sz << "\n";
    }

    // unique_ptr is move-only, so our class is also move-only by default
    Resource(Resource&& other) noexcept = default;
    Resource& operator=(Resource&& other) noexcept = default;

    // Delete copy operations (unique_ptr is not copyable)
    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;

    size_t getSize() const { return size; }
};

// Example 4: Perfect forwarding with std::forward
template<typename T>
void wrapper(T&& arg) {
    // Without std::forward, arg is always an lvalue inside this function
    // std::forward preserves the value category (lvalue or rvalue)
    process(std::forward<T>(arg));
}

void process(const std::string& str) {
    std::cout << "Processing lvalue: " << str << "\n";
}

void process(std::string&& str) {
    std::cout << "Processing rvalue: " << str << "\n";
}

// Example 5: Return value optimization (RVO) vs Move
Buffer createBuffer(size_t size) {
    Buffer b(size);
    // RVO: Compiler may optimize away the move/copy entirely
    // If RVO doesn't apply, move constructor is used
    return b;
}

// Example 6: Move in STL containers
void demonstrateMoveInContainers() {
    std::cout << "\n=== Move in Containers ===\n";
    
    std::vector<Buffer> buffers;
    buffers.reserve(3); // Reserve space to avoid reallocation
    
    // emplace_back constructs in-place (no copy/move)
    buffers.emplace_back(100);
    
    // push_back with temporary (move constructor called)
    buffers.push_back(Buffer(200));
    
    // Move existing object into vector
    Buffer b(300);
    buffers.push_back(std::move(b)); // Explicitly move
}

// Example 7: Conditional move with std::move_if_noexcept
template<typename T>
void safeMove(T& source, T& dest) {
    // Uses move if move constructor is noexcept, otherwise copy
    dest = std::move_if_noexcept(source);
}

// Example 8: Common pitfalls
void commonPitfalls() {
    std::cout << "\n=== Common Pitfalls ===\n";
    
    // Pitfall 1: Using std::move on const objects
    const Buffer b1(100);
    Buffer b2(std::move(b1)); // Calls copy constructor, not move! const can't be moved
    
    // Pitfall 2: Accessing moved-from objects
    Buffer b3(100);
    Buffer b4(std::move(b3));
    // b3 is now in valid but unspecified state
    // Don't use b3 except to reassign or destroy
    
    // Pitfall 3: std::move doesn't actually move anything
    // It just casts to rvalue reference - the move happens in the constructor/assignment
    std::string s1 = "text";
    std::move(s1); // Does nothing! Need to actually use the result
    std::string s2 = std::move(s1); // Now it moves
}

// Example 9: Move semantics in function parameters
void processByValue(std::string str) {
    // str is moved into this function if called with rvalue
    std::cout << "Processing: " << str << "\n";
}

void processByRvalueRef(std::string&& str) {
    // Can only accept rvalues
    std::cout << "Processing rvalue: " << str << "\n";
}

void processByUniversalRef(auto&& str) {
    // Universal/forwarding reference - can accept both lvalues and rvalues
    std::cout << "Processing universal ref: " << str << "\n";
}

int main() {
    std::cout << "=== Move Constructor Demo ===\n";
    Buffer b1(1000);
    Buffer b2(std::move(b1)); // Move constructor called
    
    std::cout << "\n=== Move Assignment Demo ===\n";
    Buffer b3(500);
    Buffer b4(200);
    b4 = std::move(b3); // Move assignment called
    
    demonstrateStdMove();
    
    std::cout << "\n=== Resource Move Demo ===\n";
    Resource r1(100);
    Resource r2(std::move(r1)); // Must use std::move, no copy constructor
    
    std::cout << "\n=== RVO Demo ===\n";
    Buffer b5 = createBuffer(1000); // RVO likely applies
    
    demonstrateMoveInContainers();
    commonPitfalls();
    
    std::cout << "\n=== Function Parameter Demo ===\n";
    std::string temp = "temporary";
    processByValue(std::move(temp));
    processByRvalueRef("literal string");
    processByUniversalRef(temp);
    processByUniversalRef("another literal");
    
    return 0;
}

/*
 * Key Takeaways:
 * 
 * 1. Move semantics transfer ownership instead of copying resources
 * 2. std::move is just a cast to rvalue reference - doesn't move anything itself
 * 3. Move constructors/assignments should be noexcept when possible
 * 4. After moving from an object, it's in a valid but unspecified state
 * 5. Compiler-generated moves are available if no user-defined copy/move/destructor
 * 6. Rule of Five: If you define any of (destructor, copy ctor, copy assign, 
 *    move ctor, move assign), consider defining all five
 * 7. std::forward preserves value categories in template forwarding
 * 8. RVO can optimize away moves entirely in some cases
 * 
 * Interview Questions:
 * Q: When is a move constructor called?
 * A: When initializing from an rvalue (temporary or std::move'd object)
 * 
 * Q: What's the difference between std::move and std::forward?
 * A: std::move unconditionally casts to rvalue, std::forward conditionally 
 *    casts based on the template argument (preserves value category)
 * 
 * Q: Why make move operations noexcept?
 * A: STL containers use std::move_if_noexcept for strong exception guarantee
 *    - they'll copy instead of move if move can throw
 */
