#include <iostream>
#include <vector>
#include <string>
#include <chrono>

// Return Value Optimization (RVO), Named RVO (NRVO), and Copy Elision

class Tracked {
    std::string name;
    std::vector<int> data;
    
public:
    explicit Tracked(const std::string& n, size_t size = 1000) : name(n) {
        data.resize(size, 42);
        std::cout << "  Constructor: " << name << std::endl;
    }
    
    Tracked(const Tracked& other) : name(other.name), data(other.data) {
        std::cout << "  Copy Constructor: " << name << std::endl;
    }
    
    Tracked(Tracked&& other) noexcept 
        : name(std::move(other.name)), data(std::move(other.data)) {
        std::cout << "  Move Constructor: " << name << std::endl;
    }
    
    Tracked& operator=(const Tracked& other) {
        if (this != &other) {
            name = other.name;
            data = other.data;
            std::cout << "  Copy Assignment: " << name << std::endl;
        }
        return *this;
    }
    
    Tracked& operator=(Tracked&& other) noexcept {
        if (this != &other) {
            name = std::move(other.name);
            data = std::move(other.data);
            std::cout << "  Move Assignment: " << name << std::endl;
        }
        return *this;
    }
    
    ~Tracked() {
        std::cout << "  Destructor: " << name << std::endl;
    }
    
    const std::string& getName() const { return name; }
};

// Pattern 1: Return by value - RVO (Return Value Optimization)
// The temporary is constructed directly in the caller's storage
Tracked createWithRVO() {
    std::cout << "Creating with RVO:" << std::endl;
    return Tracked("RVO_Object");  // No copy/move due to RVO
}

// Pattern 2: Named Return Value Optimization (NRVO)
// Named object constructed directly in caller's storage
Tracked createWithNRVO() {
    std::cout << "Creating with NRVO:" << std::endl;
    Tracked obj("NRVO_Object");
    // Some operations on obj
    return obj;  // Likely NRVO (compiler-dependent)
}

// Pattern 3: Multiple return paths - NRVO might not apply
Tracked createConditional(bool flag) {
    std::cout << "Creating conditional:" << std::endl;
    if (flag) {
        Tracked obj1("Conditional_True");
        return obj1;  // NRVO might not apply
    } else {
        Tracked obj2("Conditional_False");
        return obj2;  // NRVO might not apply
    }
}

// Pattern 4: Explicitly preventing copy elision with std::move (DON'T DO THIS!)
Tracked createWithBadMove() {
    std::cout << "Creating with bad std::move:" << std::endl;
    Tracked obj("Bad_Move");
    return std::move(obj);  // BAD: Prevents RVO/NRVO, forces move
}

// Pattern 5: Return by value for large objects - still efficient
std::vector<int> createLargeVector(size_t size) {
    std::vector<int> result;
    result.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        result.push_back(i);
    }
    return result;  // RVO applies, no copy
}

// Pattern 6: Copy elision in initialization (C++17 guaranteed)
void demonstrateCopyElision() {
    std::cout << "\n=== Copy Elision in Initialization ===" << std::endl;
    Tracked obj = createWithRVO();  // Direct construction, no copy/move
    std::cout << "Object name: " << obj.getName() << std::endl;
}

// Pattern 7: Returning different objects - when RVO doesn't apply
Tracked selectObject(bool first) {
    std::cout << "\nSelecting object:" << std::endl;
    Tracked obj1("First");
    Tracked obj2("Second");
    return first ? obj1 : obj2;  // RVO can't apply, will move
}

// Pattern 8: Performance comparison
void performanceComparison() {
    std::cout << "\n=== Performance Comparison ===" << std::endl;
    
    const size_t iterations = 1000000;
    
    // Return by value with RVO
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        auto vec = createLargeVector(100);
        (void)vec;  // Prevent optimization
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Return by value (with RVO): " << duration.count() << "ms" << std::endl;
}

// Pattern 9: Modern factory pattern - return by value
template<typename T, typename... Args>
T createObject(Args&&... args) {
    return T(std::forward<Args>(args)...);  // RVO applies
}

// Pattern 10: Chaining with return by value
class Builder {
    std::string data;
    
public:
    Builder() = default;
    
    // Return by value for chaining (move semantics apply)
    Builder append(const std::string& str) && {
        data += str;
        return std::move(*this);  // Explicit move for rvalue
    }
    
    // Return by reference for lvalue chaining
    Builder& append(const std::string& str) & {
        data += str;
        return *this;
    }
    
    std::string build() && {
        return std::move(data);
    }
    
    std::string build() const & {
        return data;
    }
};

// Pattern 11: std::optional return
#include <optional>

std::optional<Tracked> maybeCreate(bool create) {
    if (create) {
        return Tracked("Optional_Object");  // RVO applies
    }
    return std::nullopt;
}

// Pattern 12: Structured bindings and return
struct Point {
    int x, y;
};

Point createPoint(int x, int y) {
    return {x, y};  // Aggregate initialization, no copy
}

// GUIDELINES for return by value vs reference

// Return by VALUE when:
// - Creating new objects (factory functions)
// - Cheap to move types (with move constructor)
// - Local objects (RVO/NRVO applies)
// - std::optional, std::variant, etc.

// Return by REFERENCE when:
// - Returning member variables (const T&)
// - Returning parameters (const T&)
// - Returning static/global data
// NEVER return reference to local variable!

// Return by POINTER when:
// - Returning nullptr is valid
// - Returning optional non-owning access

int main() {
    std::cout << "=== RVO AND OPTIMIZATION ===" << std::endl;
    
    std::cout << "\n--- RVO Example ---" << std::endl;
    Tracked obj1 = createWithRVO();
    
    std::cout << "\n--- NRVO Example ---" << std::endl;
    Tracked obj2 = createWithNRVO();
    
    std::cout << "\n--- Conditional Return ---" << std::endl;
    Tracked obj3 = createConditional(true);
    
    std::cout << "\n--- Bad std::move Example ---" << std::endl;
    Tracked obj4 = createWithBadMove();
    
    demonstrateCopyElision();
    
    std::cout << "\n--- Select Object (No RVO) ---" << std::endl;
    Tracked obj5 = selectObject(true);
    
    std::cout << "\n--- Builder Pattern ---" << std::endl;
    Builder b1;
    std::string result1 = b1.append("Hello").append(" ").append("World").build();
    std::cout << "Built: " << result1 << std::endl;
    
    std::string result2 = Builder().append("Temporary").append("!").build();
    std::cout << "Built from temp: " << result2 << std::endl;
    
    std::cout << "\n--- Optional Return ---" << std::endl;
    if (auto opt = maybeCreate(true)) {
        std::cout << "Created: " << opt->getName() << std::endl;
    }
    
    std::cout << "\n--- Structured Binding ---" << std::endl;
    auto [x, y] = createPoint(10, 20);
    std::cout << "Point: (" << x << ", " << y << ")" << std::endl;
    
    performanceComparison();
    
    std::cout << "\n=== Program ending ===" << std::endl;
    return 0;
}
