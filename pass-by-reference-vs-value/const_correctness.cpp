#include <iostream>
#include <vector>
#include <string>
#include <memory>

// Const Correctness: Why it matters for pass-by-reference

class Data {
    mutable int accessCount = 0;  // mutable allows modification in const methods
    std::string content;
    
public:
    explicit Data(const std::string& c) : content(c) {}
    
    // Const member function - promises not to modify object state
    const std::string& getContent() const {
        ++accessCount;  // OK: mutable member
        return content;
    }
    
    // Non-const member function - can modify state
    void setContent(const std::string& c) {
        content = c;
    }
    
    int getAccessCount() const { return accessCount; }
};

// Pattern 1: Const reference parameters - read-only access
void printData(const Data& data) {
    std::cout << "Data: " << data.getContent() << std::endl;
    // data.setContent("new");  // ERROR: can't call non-const method
}

// Pattern 2: Non-const reference parameters - intent to modify
void modifyData(Data& data) {
    data.setContent("Modified");
    std::cout << "Modified data" << std::endl;
}

// Pattern 3: Const correctness with pointers
void processConstPtr(const Data* data) {
    if (data) {
        std::cout << "Processing: " << data->getContent() << std::endl;
        // data->setContent("new");  // ERROR
    }
}

void processNonConstPtr(Data* data) {
    if (data) {
        data->setContent("Changed via pointer");
    }
}

// Pattern 4: Const and references with containers
void printVector(const std::vector<int>& vec) {
    // vec.push_back(10);  // ERROR: can't modify const reference
    for (const int& val : vec) {  // const reference in range-for
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

void doubleValues(std::vector<int>& vec) {
    for (int& val : vec) {  // Non-const reference to modify
        val *= 2;
    }
}

// Pattern 5: Const member functions and overloading
class Container {
    std::vector<int> data;
    
public:
    Container(std::initializer_list<int> init) : data(init) {}
    
    // Const version - returns const reference
    const int& at(size_t index) const {
        std::cout << "  const at() called" << std::endl;
        return data.at(index);
    }
    
    // Non-const version - returns non-const reference
    int& at(size_t index) {
        std::cout << "  non-const at() called" << std::endl;
        return data.at(index);
    }
    
    // Const iterator access
    auto begin() const { return data.begin(); }
    auto end() const { return data.end(); }
    
    // Non-const iterator access
    auto begin() { return data.begin(); }
    auto end() { return data.end(); }
};

// Pattern 6: Top-level const vs low-level const
void demonstrateConstLevels() {
    std::cout << "\n=== Const Levels ===" << std::endl;
    
    int value = 42;
    
    // Top-level const (const applies to the variable itself)
    const int topLevel = value;  // Can't modify topLevel
    // topLevel = 100;  // ERROR
    
    // Low-level const (const applies to what pointer points to)
    const int* lowLevel = &value;  // Can't modify *lowLevel
    // *lowLevel = 100;  // ERROR
    lowLevel = &topLevel;  // OK: can change what it points to
    
    // Both levels
    const int* const both = &value;
    // *both = 100;  // ERROR
    // both = &topLevel;  // ERROR
    
    std::cout << "Value through lowLevel: " << *lowLevel << std::endl;
}

// Pattern 7: Const with return types
class Resource {
    std::string name;
public:
    explicit Resource(const std::string& n) : name(n) {}
    
    // Return const reference to member - prevents modification
    const std::string& getName() const {
        return name;
    }
    
    // Return by value - caller gets copy
    std::string getNameCopy() const {
        return name;
    }
    
    // DANGER: Return non-const reference to member
    // Only do this if you want to allow external modification
    std::string& getNameRef() {
        return name;
    }
};

// Pattern 8: Const with smart pointers
void constSmartPointers() {
    std::cout << "\n=== Const Smart Pointers ===" << std::endl;
    
    // Const shared_ptr - can't change what it points to
    const std::shared_ptr<Data> constPtr = std::make_shared<Data>("Const");
    // constPtr = nullptr;  // ERROR
    constPtr->setContent("Modified");  // OK: object not const
    
    // Pointer to const data
    std::shared_ptr<const Data> ptrToConst = std::make_shared<Data>("ConstData");
    std::cout << ptrToConst->getContent() << std::endl;
    // ptrToConst->setContent("new");  // ERROR: object is const
    
    // Both const
    const std::shared_ptr<const Data> bothConst = std::make_shared<Data>("BothConst");
    // bothConst = nullptr;  // ERROR
    // bothConst->setContent("new");  // ERROR
}

// Pattern 9: Const with templates
template<typename T>
void processConst(const T& item) {
    std::cout << "Processing const: " << item << std::endl;
    // item += 10;  // ERROR if T is not const-friendly
}

template<typename T>
void processNonConst(T& item) {
    std::cout << "Processing non-const: " << item << std::endl;
    item += 10;  // Modifies the original
}

// Pattern 10: Const cast (use sparingly!)
void demonstrateConstCast() {
    std::cout << "\n=== Const Cast (antipattern) ===" << std::endl;
    
    const int constValue = 100;
    
    // const_cast removes const (DANGEROUS!)
    int& modified = const_cast<int&>(constValue);
    // modified = 200;  // Undefined behavior!
    
    std::cout << "Original const value: " << constValue << std::endl;
    
    // Legitimate use: calling legacy non-const API
    // But prefer fixing the API instead
}

// Pattern 11: Const propagation in class hierarchies
class Base {
public:
    virtual void process() const {
        std::cout << "Base const process" << std::endl;
    }
    
    virtual void modify() {
        std::cout << "Base modify" << std::endl;
    }
    
    virtual ~Base() = default;
};

class Derived : public Base {
public:
    void process() const override {
        std::cout << "Derived const process" << std::endl;
    }
    
    void modify() override {
        std::cout << "Derived modify" << std::endl;
    }
};

void usePolymorphism(const Base& obj) {
    obj.process();  // OK: const method
    // obj.modify();  // ERROR: non-const method
}

// Pattern 12: Why const correctness matters for optimization
void optimizationBenefit() {
    std::cout << "\n=== Optimization Benefits ===" << std::endl;
    
    std::vector<int> data = {1, 2, 3, 4, 5};
    
    // Const allows compiler optimizations
    // Compiler knows function won't modify data
    auto sum = [](const std::vector<int>& vec) {
        int total = 0;
        for (int val : vec) {
            total += val;
        }
        return total;
    };
    
    std::cout << "Sum: " << sum(data) << std::endl;
    
    // Without const, compiler must assume modifications
    auto sumNonConst = [](std::vector<int>& vec) {
        int total = 0;
        for (int val : vec) {
            total += val;
        }
        return total;
    };
    
    std::cout << "Sum (non-const): " << sumNonConst(data) << std::endl;
}

int main() {
    std::cout << "=== CONST CORRECTNESS ===" << std::endl;
    
    Data data("Original");
    
    std::cout << "\n--- Const reference (read-only) ---" << std::endl;
    printData(data);
    
    std::cout << "\n--- Non-const reference (modifiable) ---" << std::endl;
    modifyData(data);
    printData(data);
    
    std::cout << "\n--- Const with containers ---" << std::endl;
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    printVector(numbers);
    doubleValues(numbers);
    printVector(numbers);
    
    std::cout << "\n--- Const overloading ---" << std::endl;
    Container c = {10, 20, 30};
    const Container cc = {40, 50, 60};
    
    c.at(0) = 100;  // Calls non-const version
    std::cout << "Modified: " << c.at(0) << std::endl;
    
    std::cout << "Const container: " << cc.at(0) << std::endl;  // Calls const version
    // cc.at(0) = 100;  // ERROR
    
    demonstrateConstLevels();
    
    std::cout << "\n--- Const return values ---" << std::endl;
    Resource res("MyResource");
    const std::string& nameRef = res.getName();
    std::cout << "Name: " << nameRef << std::endl;
    // nameRef[0] = 'X';  // ERROR: const reference
    
    std::string& modifiableRef = res.getNameRef();
    modifiableRef[0] = 'X';
    std::cout << "Modified name: " << res.getName() << std::endl;
    
    constSmartPointers();
    
    std::cout << "\n--- Templates with const ---" << std::endl;
    int x = 5;
    processConst(x);
    std::cout << "x after const: " << x << std::endl;
    processNonConst(x);
    std::cout << "x after non-const: " << x << std::endl;
    
    std::cout << "\n--- Polymorphism with const ---" << std::endl;
    Derived d;
    usePolymorphism(d);
    
    optimizationBenefit();
    
    std::cout << "\n=== Access count (mutable) ===" << std::endl;
    std::cout << "Data accessed " << data.getAccessCount() << " times" << std::endl;
    
    return 0;
}
