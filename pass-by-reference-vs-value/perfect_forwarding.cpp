#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <type_traits>

// Perfect Forwarding and Universal/Forwarding References

// Pattern 1: Basic perfect forwarding
template<typename T>
void wrapper(T&& arg) {
    // T&& is a forwarding reference (not rvalue reference!)
    // It can bind to both lvalues and rvalues
    std::cout << "Forwarding argument..." << std::endl;
    
    // std::forward preserves the value category
    if constexpr (std::is_lvalue_reference_v<T>) {
        std::cout << "Received lvalue reference" << std::endl;
    } else {
        std::cout << "Received rvalue reference" << std::endl;
    }
}

// Pattern 2: Factory function with perfect forwarding
template<typename T, typename... Args>
std::unique_ptr<T> make_unique_verbose(Args&&... args) {
    std::cout << "Creating object with " << sizeof...(args) << " arguments" << std::endl;
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

class Item {
    std::string name;
    int value;
    
public:
    Item(const std::string& n, int v) : name(n), value(v) {
        std::cout << "Item(const string&, int): " << name << std::endl;
    }
    
    Item(std::string&& n, int v) : name(std::move(n)), value(v) {
        std::cout << "Item(string&&, int): " << name << std::endl;
    }
    
    void display() const {
        std::cout << "Item: " << name << ", value: " << value << std::endl;
    }
};

// Pattern 3: Variadic template with perfect forwarding
template<typename Func, typename... Args>
auto measureCall(Func&& func, Args&&... args) {
    std::cout << "Calling function with " << sizeof...(args) << " arguments" << std::endl;
    return std::forward<Func>(func)(std::forward<Args>(args)...);
}

// Pattern 4: Forwarding with member functions
class Container {
    std::vector<std::string> data;
    
public:
    template<typename T>
    void add(T&& item) {
        // Perfect forwarding to vector's push_back
        data.push_back(std::forward<T>(item));
        std::cout << "Added item (forwarded)" << std::endl;
    }
    
    template<typename... Args>
    void emplace(Args&&... args) {
        // Perfect forwarding to vector's emplace_back
        data.emplace_back(std::forward<Args>(args)...);
        std::cout << "Emplaced item" << std::endl;
    }
    
    void display() const {
        std::cout << "Container contents: ";
        for (const auto& item : data) {
            std::cout << item << " ";
        }
        std::cout << std::endl;
    }
};

// Pattern 5: Reference collapsing rules demonstration
template<typename T>
void analyzeType(T&& arg) {
    using PlainType = std::remove_reference_t<T>;
    
    std::cout << "Type analysis:" << std::endl;
    std::cout << "  is_lvalue_reference: " << std::is_lvalue_reference_v<T> << std::endl;
    std::cout << "  is_rvalue_reference: " << std::is_rvalue_reference_v<T> << std::endl;
    std::cout << "  is_reference: " << std::is_reference_v<T> << std::endl;
    
    // Reference collapsing:
    // T& && -> T&
    // T&& & -> T&
    // T& & -> T&
    // T&& && -> T&&
}

// Pattern 6: Conditional forwarding
template<typename T>
void conditionalForward(T&& arg) {
    if constexpr (std::is_rvalue_reference_v<T&&>) {
        std::cout << "Moving the argument" << std::endl;
        // Process as rvalue
    } else {
        std::cout << "Keeping lvalue reference" << std::endl;
        // Process as lvalue
    }
}

// Pattern 7: Forwarding multiple arguments with different types
template<typename T1, typename T2>
class Pair {
    T1 first_;
    T2 second_;
    
public:
    template<typename U1, typename U2>
    Pair(U1&& first, U2&& second)
        : first_(std::forward<U1>(first))
        , second_(std::forward<U2>(second)) {
        std::cout << "Pair constructed with perfect forwarding" << std::endl;
    }
    
    void display() const {
        std::cout << "Pair: (" << first_ << ", " << second_ << ")" << std::endl;
    }
};

// Pattern 8: std::invoke with perfect forwarding
template<typename Callable, typename... Args>
decltype(auto) invokeAndLog(Callable&& func, Args&&... args) {
    std::cout << "Invoking callable..." << std::endl;
    return std::invoke(std::forward<Callable>(func), std::forward<Args>(args)...);
}

// Pattern 9: Forwarding with SFINAE
template<typename T, 
         typename = std::enable_if_t<std::is_move_constructible_v<T>>>
void moveIfPossible(T&& arg) {
    std::cout << "Type is move constructible" << std::endl;
    auto moved = std::forward<T>(arg);
}

int main() {
    std::cout << "=== PERFECT FORWARDING ===" << std::endl;
    
    std::cout << "\n--- Basic forwarding ---" << std::endl;
    int x = 42;
    wrapper(x);           // Lvalue
    wrapper(100);         // Rvalue
    wrapper(std::move(x)); // Rvalue
    
    std::cout << "\n--- Factory with perfect forwarding ---" << std::endl;
    std::string name = "ItemA";
    auto item1 = make_unique_verbose<Item>(name, 10);  // Lvalue
    auto item2 = make_unique_verbose<Item>("ItemB", 20);  // Rvalue
    item1->display();
    item2->display();
    
    std::cout << "\n--- Variadic forwarding ---" << std::endl;
    auto printer = [](const std::string& msg, int val) {
        std::cout << msg << ": " << val << std::endl;
        return val * 2;
    };
    
    int result = measureCall(printer, "Result", 21);
    std::cout << "Returned: " << result << std::endl;
    
    std::cout << "\n--- Container with forwarding ---" << std::endl;
    Container c;
    std::string str = "Hello";
    c.add(str);              // Lvalue - copies
    c.add("World");          // Rvalue - moves
    c.emplace("Emplaced");   // Constructed in-place
    c.display();
    
    std::cout << "\n--- Type analysis ---" << std::endl;
    int y = 10;
    analyzeType(y);          // Lvalue
    analyzeType(20);         // Rvalue
    
    std::cout << "\n--- Pair with forwarding ---" << std::endl;
    std::string first = "First";
    Pair<std::string, int> p1(first, 42);  // Lvalue, rvalue
    Pair<std::string, int> p2("Second", 84);  // Rvalue, rvalue
    p1.display();
    p2.display();
    
    std::cout << "\n--- std::invoke forwarding ---" << std::endl;
    auto multiply = [](int a, int b) { return a * b; };
    int product = invokeAndLog(multiply, 6, 7);
    std::cout << "Product: " << product << std::endl;
    
    return 0;
}
