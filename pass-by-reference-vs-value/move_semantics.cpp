#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <utility>

// Move semantics - pass by value vs rvalue reference

class Resource {
private:
    std::vector<int> data;
    std::string name;
    
public:
    Resource(const std::string& n, size_t size) : name(n) {
        data.resize(size, 42);
        std::cout << "Constructor: " << name << " (size: " << size << ")" << std::endl;
    }
    
    // Copy constructor
    Resource(const Resource& other) : data(other.data), name(other.name) {
        std::cout << "Copy constructor: " << name << std::endl;
    }
    
    // Move constructor
    Resource(Resource&& other) noexcept 
        : data(std::move(other.data)), name(std::move(other.name)) {
        std::cout << "Move constructor: " << name << std::endl;
    }
    
    // Copy assignment
    Resource& operator=(const Resource& other) {
        if (this != &other) {
            data = other.data;
            name = other.name;
            std::cout << "Copy assignment: " << name << std::endl;
        }
        return *this;
    }
    
    // Move assignment
    Resource& operator=(Resource&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
            name = std::move(other.name);
            std::cout << "Move assignment: " << name << std::endl;
        }
        return *this;
    }
    
    const std::string& getName() const { return name; }
    size_t getSize() const { return data.size(); }
};

// Pattern 1: Pass by value for sink functions (takes ownership)
// Modern C++: Let the compiler optimize with move semantics
void storeByValue(std::vector<Resource> resources) {
    std::cout << "Storing " << resources.size() << " resources" << std::endl;
    // resources is moved or copied depending on caller
}

// Pattern 2: Pass by rvalue reference (explicit move-only)
void storeByRvalueRef(std::vector<Resource>&& resources) {
    std::cout << "Storing " << resources.size() << " resources (rvalue ref)" << std::endl;
    // Caller must explicitly move
}

// Pattern 3: Pass by const lvalue ref (no ownership transfer)
void inspectByConstRef(const std::vector<Resource>& resources) {
    std::cout << "Inspecting " << resources.size() << " resources" << std::endl;
}

// Pattern 4: Pass by value for cheap-to-move types
class Widget {
    std::unique_ptr<int> data;
    
public:
    Widget(int val) : data(std::make_unique<int>(val)) {}
    
    // Move-only type (deleted copy)
    Widget(const Widget&) = delete;
    Widget& operator=(const Widget&) = delete;
    Widget(Widget&&) = default;
    Widget& operator=(Widget&&) = default;
    
    int getValue() const { return data ? *data : 0; }
};

void consumeWidget(Widget w) {
    std::cout << "Consumed widget with value: " << w.getValue() << std::endl;
}

// Pattern 5: Return by value with move semantics
Resource createResource(const std::string& name, size_t size) {
    Resource r(name, size);
    return r;  // NRVO or move, never copy
}

// Pattern 6: Returning rvalue reference (DANGEROUS - for demonstration only)
// DON'T DO THIS - it's a dangling reference!
// Resource&& createBadResource() {
//     Resource r("bad", 100);
//     return std::move(r);  // DANGER: returns reference to local
// }

// Pattern 7: Perfect forwarding with templates (covered in separate file)
// Pattern 8: std::move vs std::forward

void processLvalue(Resource& r) {
    std::cout << "Processing lvalue: " << r.getName() << std::endl;
}

void processRvalue(Resource&& r) {
    std::cout << "Processing rvalue: " << r.getName() << std::endl;
}

template<typename T>
void forwardToProcess(T&& arg) {
    // std::forward preserves value category
    if constexpr (std::is_lvalue_reference_v<T>) {
        processLvalue(arg);
    } else {
        processRvalue(std::forward<T>(arg));
    }
}

// Pattern 9: String parameters - modern best practice
class TextProcessor {
public:
    // Old way: const reference
    void setTextOld(const std::string& text) {
        text_ = text;  // Always copies
    }
    
    // Old way: two overloads
    void setTextTwoOverloads(const std::string& text) {
        text_ = text;
    }
    void setTextTwoOverloads(std::string&& text) {
        text_ = std::move(text);
    }
    
    // Modern way: pass by value + move
    void setText(std::string text) {
        text_ = std::move(text);  // One function handles both
    }
    
    const std::string& getText() const { return text_; }
    
private:
    std::string text_;
};

int main() {
    std::cout << "=== MOVE SEMANTICS ===" << std::endl;
    
    std::cout << "\n--- Creating and moving resources ---" << std::endl;
    Resource r1("Resource1", 1000);
    Resource r2 = std::move(r1);  // Move constructor
    
    std::cout << "\n--- Pass by value with rvalue ---" << std::endl;
    std::vector<Resource> vec;
    vec.push_back(Resource("Temp", 100));  // Move into vector
    storeByValue(std::move(vec));  // Move into function
    
    std::cout << "\n--- Move-only types ---" << std::endl;
    Widget w1(42);
    consumeWidget(std::move(w1));  // Must use std::move
    
    std::cout << "\n--- Return by value (RVO/move) ---" << std::endl;
    Resource r3 = createResource("Factory", 500);
    
    std::cout << "\n--- Perfect forwarding ---" << std::endl;
    Resource r4("Forwardable", 200);
    forwardToProcess(r4);  // Lvalue
    forwardToProcess(Resource("Temporary", 300));  // Rvalue
    
    std::cout << "\n--- Modern string parameter passing ---" << std::endl;
    TextProcessor tp;
    
    std::string str1 = "Hello World";
    tp.setText(str1);  // Copies str1, then moves into text_
    std::cout << "Set text (lvalue): " << tp.getText() << std::endl;
    
    tp.setText("Temporary String");  // Creates temp, moves into text_
    std::cout << "Set text (rvalue): " << tp.getText() << std::endl;
    
    tp.setText(std::move(str1));  // Moves str1 into text_
    std::cout << "Set text (moved): " << tp.getText() << std::endl;
    std::cout << "Original string now: '" << str1 << "'" << std::endl;
    
    return 0;
}
