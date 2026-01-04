#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

// Move semantics - transfer ownership instead of copying
class String {
private:
    char* data_;
    size_t size_;
    
public:
    // Constructor
    String(const char* str = "") {
        size_ = std::strlen(str);
        data_ = new char[size_ + 1];
        std::strcpy(data_, str);
        std::cout << "Constructor: " << data_ << std::endl;
    }
    
    // Destructor
    ~String() {
        std::cout << "Destructor: " << (data_ ? data_ : "null") << std::endl;
        delete[] data_;
    }
    
    // Copy constructor - deep copy
    String(const String& other) : size_(other.size_) {
        data_ = new char[size_ + 1];
        std::strcpy(data_, other.data_);
        std::cout << "Copy constructor: " << data_ << std::endl;
    }
    
    // Copy assignment
    String& operator=(const String& other) {
        if (this != &other) {
            delete[] data_;
            size_ = other.size_;
            data_ = new char[size_ + 1];
            std::strcpy(data_, other.data_);
            std::cout << "Copy assignment: " << data_ << std::endl;
        }
        return *this;
    }
    
    // Move constructor - transfer ownership
    String(String&& other) noexcept : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
        std::cout << "Move constructor: " << data_ << std::endl;
    }
    
    // Move assignment
    String& operator=(String&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
            std::cout << "Move assignment: " << data_ << std::endl;
        }
        return *this;
    }
    
    const char* c_str() const { return data_ ? data_ : ""; }
};

// Perfect forwarding example
template<typename T>
void wrapper(T&& arg) {
    // std::forward preserves lvalue/rvalue nature
    process(std::forward<T>(arg));
}

void process(const String& s) {
    std::cout << "Processing lvalue: " << s.c_str() << std::endl;
}

void process(String&& s) {
    std::cout << "Processing rvalue: " << s.c_str() << std::endl;
}

String createString() {
    return String("temporary");
}

void demonstrateMoveSemantics() {
    std::cout << "=== Move Semantics ===" << std::endl;
    
    // Copy
    String s1("Hello");
    String s2 = s1; // Copy constructor
    
    // Move
    String s3 = std::move(s1); // Move constructor
    std::cout << "s1 after move: " << s1.c_str() << std::endl;
    
    // RVO (Return Value Optimization)
    String s4 = createString(); // Likely no copy or move due to RVO
    
    // Move with vectors
    std::cout << "\n=== Vector Move ===" << std::endl;
    std::vector<String> vec;
    vec.push_back(String("First"));  // Move
    vec.emplace_back("Second");      // Construct in-place
    
    // Move elements
    String str1("Original");
    vec.push_back(std::move(str1));
    std::cout << "str1 after move: " << str1.c_str() << std::endl;
}

// Performance comparison
void performanceDemo() {
    std::cout << "\n=== Performance Comparison ===" << std::endl;
    
    // Without move - expensive copies
    std::vector<std::vector<int>> vec1;
    std::vector<int> large(1000000, 42);
    vec1.push_back(large); // Copy - slow
    
    // With move - cheap transfer
    std::vector<std::vector<int>> vec2;
    vec2.push_back(std::move(large)); // Move - fast
    std::cout << "large size after move: " << large.size() << std::endl;
}

int main() {
    demonstrateMoveSemantics();
    performanceDemo();
    
    std::cout << "\n=== Program End ===" << std::endl;
    return 0;
}
