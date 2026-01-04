/*
 * Templates in Modern C++
 * 
 * Key Concepts:
 * - Function templates
 * - Class templates
 * - Template specialization (full and partial)
 * - Variadic templates
 * - SFINAE and template constraints
 * - Template template parameters
 * 
 * Interview Topics:
 * - How does template instantiation work?
 * - What is template specialization?
 * - How to handle variadic template arguments?
 */

#include <iostream>
#include <vector>
#include <string>
#include <type_traits>
#include <concepts>

// Example 1: Basic function template
template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}

// Multiple template parameters
template<typename T, typename U>
auto add(T a, U b) -> decltype(a + b) {
    return a + b;
}

void demonstrateFunctionTemplates() {
    std::cout << "=== Function Templates ===\n";
    
    std::cout << "max(5, 10) = " << max(5, 10) << "\n";
    std::cout << "max(3.14, 2.72) = " << max(3.14, 2.72) << "\n";
    std::cout << "max('a', 'z') = " << max('a', 'z') << "\n";
    
    // Template argument deduction
    auto result1 = add(5, 3.14);        // T=int, U=double
    auto result2 = add(std::string("Hello "), std::string("World"));
    
    std::cout << "add(5, 3.14) = " << result1 << "\n";
    std::cout << "add(strings) = " << result2 << "\n";
    
    // Explicit template arguments
    auto result3 = max<double>(5, 10); // Forces double comparison
    std::cout << "max<double>(5, 10) = " << result3 << "\n";
}

// Example 2: Class template
template<typename T>
class Stack {
private:
    std::vector<T> elements;

public:
    void push(const T& elem) {
        elements.push_back(elem);
    }
    
    void pop() {
        if (!elements.empty()) {
            elements.pop_back();
        }
    }
    
    T& top() {
        return elements.back();
    }
    
    bool empty() const {
        return elements.empty();
    }
    
    size_t size() const {
        return elements.size();
    }
};

void demonstrateClassTemplates() {
    std::cout << "\n=== Class Templates ===\n";
    
    Stack<int> intStack;
    intStack.push(1);
    intStack.push(2);
    intStack.push(3);
    
    std::cout << "Int stack size: " << intStack.size() << "\n";
    std::cout << "Top element: " << intStack.top() << "\n";
    
    Stack<std::string> stringStack;
    stringStack.push("Hello");
    stringStack.push("World");
    
    std::cout << "String stack top: " << stringStack.top() << "\n";
}

// Example 3: Template specialization
// Generic template
template<typename T>
class Printer {
public:
    void print(const T& value) {
        std::cout << "Generic: " << value << "\n";
    }
};

// Full specialization for bool
template<>
class Printer<bool> {
public:
    void print(const bool& value) {
        std::cout << "Bool: " << (value ? "true" : "false") << "\n";
    }
};

// Full specialization for char*
template<>
class Printer<const char*> {
public:
    void print(const char* const& value) {
        std::cout << "C-string: \"" << value << "\"\n";
    }
};

void demonstrateSpecialization() {
    std::cout << "\n=== Template Specialization ===\n";
    
    Printer<int> intPrinter;
    intPrinter.print(42);
    
    Printer<bool> boolPrinter;
    boolPrinter.print(true);
    
    Printer<const char*> strPrinter;
    strPrinter.print("Hello");
}

// Example 4: Partial specialization
// Generic template for pairs
template<typename T, typename U>
class Pair {
private:
    T first;
    U second;

public:
    Pair(T f, U s) : first(f), second(s) {}
    
    void print() const {
        std::cout << "Pair<T,U>: (" << first << ", " << second << ")\n";
    }
};

// Partial specialization when both types are the same
template<typename T>
class Pair<T, T> {
private:
    T first;
    T second;

public:
    Pair(T f, T s) : first(f), second(s) {}
    
    void print() const {
        std::cout << "Pair<T,T> (same types): (" << first << ", " << second << ")\n";
    }
    
    bool areSame() const {
        return first == second;
    }
};

// Partial specialization for pointer types
template<typename T>
class Pair<T*, T*> {
private:
    T* first;
    T* second;

public:
    Pair(T* f, T* s) : first(f), second(s) {}
    
    void print() const {
        std::cout << "Pair<T*,T*> (pointers): (" 
                  << *first << ", " << *second << ")\n";
    }
};

void demonstratePartialSpecialization() {
    std::cout << "\n=== Partial Specialization ===\n";
    
    Pair<int, double> p1(42, 3.14);
    p1.print();
    
    Pair<int, int> p2(10, 20);
    p2.print();
    std::cout << "Are same? " << p2.areSame() << "\n";
    
    int x = 5, y = 10;
    Pair<int*, int*> p3(&x, &y);
    p3.print();
}

// Example 5: Variadic templates
template<typename T>
T sum(T value) {
    return value;
}

template<typename T, typename... Args>
T sum(T first, Args... args) {
    return first + sum(args...);
}

// Variadic template class
template<typename... Types>
class Tuple;

template<>
class Tuple<> {
public:
    void print() const {
        std::cout << "()\n";
    }
};

template<typename T, typename... Rest>
class Tuple<T, Rest...> : private Tuple<Rest...> {
private:
    T value;

public:
    Tuple(T v, Rest... rest) : Tuple<Rest...>(rest...), value(v) {}
    
    void print() const {
        std::cout << value;
        if constexpr (sizeof...(Rest) > 0) {
            std::cout << ", ";
            Tuple<Rest...>::print();
        }
    }
};

void demonstrateVariadicTemplates() {
    std::cout << "\n=== Variadic Templates ===\n";
    
    std::cout << "sum(1, 2, 3, 4, 5) = " << sum(1, 2, 3, 4, 5) << "\n";
    std::cout << "sum(1.1, 2.2, 3.3) = " << sum(1.1, 2.2, 3.3) << "\n";
    
    Tuple<int, double, std::string> t(42, 3.14, "hello");
    std::cout << "Tuple: (";
    t.print();
    std::cout << ")\n";
}

// Example 6: SFINAE (Substitution Failure Is Not An Error)
// Enable function only for integral types
template<typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
twice(T value) {
    std::cout << "Integral version: ";
    return value * 2;
}

// Enable function only for floating point types
template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
twice(T value) {
    std::cout << "Floating point version: ";
    return value * 2.0;
}

void demonstrateSFINAE() {
    std::cout << "\n=== SFINAE ===\n";
    
    std::cout << twice(21) << "\n";      // Calls integral version
    std::cout << twice(3.14) << "\n";    // Calls floating point version
}

// Example 7: Template template parameters
template<typename T, template<typename> class Container>
class MyContainer {
private:
    Container<T> data;

public:
    void add(const T& value) {
        data.push_back(value);
    }
    
    void print() const {
        std::cout << "Container contents: ";
        for (const auto& item : data) {
            std::cout << item << " ";
        }
        std::cout << "\n";
    }
};

void demonstrateTemplateTemplateParameters() {
    std::cout << "\n=== Template Template Parameters ===\n";
    
    MyContainer<int, std::vector> container;
    container.add(1);
    container.add(2);
    container.add(3);
    container.print();
}

// Example 8: Non-type template parameters
template<typename T, size_t N>
class Array {
private:
    T data[N];

public:
    T& operator[](size_t index) {
        return data[index];
    }
    
    constexpr size_t size() const {
        return N;
    }
    
    void fill(const T& value) {
        for (size_t i = 0; i < N; ++i) {
            data[i] = value;
        }
    }
};

void demonstrateNonTypeTemplateParameters() {
    std::cout << "\n=== Non-Type Template Parameters ===\n";
    
    Array<int, 5> arr;
    arr.fill(42);
    
    std::cout << "Array size: " << arr.size() << "\n";
    std::cout << "First element: " << arr[0] << "\n";
    
    Array<double, 3> arr2;
    arr2[0] = 1.1;
    arr2[1] = 2.2;
    arr2[2] = 3.3;
    
    std::cout << "arr2[1] = " << arr2[1] << "\n";
}

// Example 9: Template aliases (using)
template<typename T>
using Vec = std::vector<T>;

template<typename T>
using PtrVec = std::vector<T*>;

template<typename K, typename V>
using StrMap = std::pair<K, V>;

void demonstrateTemplateAliases() {
    std::cout << "\n=== Template Aliases ===\n";
    
    Vec<int> numbers = {1, 2, 3, 4, 5};
    std::cout << "Vec<int> size: " << numbers.size() << "\n";
    
    int x = 10, y = 20;
    PtrVec<int> ptrs = {&x, &y};
    std::cout << "PtrVec first element: " << *ptrs[0] << "\n";
}

// Example 10: Fold expressions (C++17)
template<typename... Args>
auto sumFold(Args... args) {
    return (args + ...); // Unary right fold
}

template<typename... Args>
void printAll(Args... args) {
    (std::cout << ... << args) << "\n"; // Unary left fold
}

template<typename... Args>
bool allTrue(Args... args) {
    return (args && ...);
}

void demonstrateFoldExpressions() {
    std::cout << "\n=== Fold Expressions ===\n";
    
    std::cout << "sumFold(1, 2, 3, 4, 5) = " << sumFold(1, 2, 3, 4, 5) << "\n";
    
    std::cout << "printAll: ";
    printAll(1, " ", 2, " ", 3);
    
    std::cout << "allTrue(true, true, true) = " << allTrue(true, true, true) << "\n";
    std::cout << "allTrue(true, false, true) = " << allTrue(true, false, true) << "\n";
}

// Example 11: if constexpr (C++17)
template<typename T>
auto getValue(T t) {
    if constexpr (std::is_pointer<T>::value) {
        std::cout << "Pointer type: ";
        return *t;
    } else {
        std::cout << "Non-pointer type: ";
        return t;
    }
}

void demonstrateIfConstexpr() {
    std::cout << "\n=== if constexpr ===\n";
    
    int x = 42;
    std::cout << getValue(x) << "\n";
    std::cout << getValue(&x) << "\n";
}

// Example 12: Template metaprogramming - Factorial at compile time
template<int N>
struct Factorial {
    static constexpr int value = N * Factorial<N - 1>::value;
};

template<>
struct Factorial<0> {
    static constexpr int value = 1;
};

void demonstrateTemplateMetaprogramming() {
    std::cout << "\n=== Template Metaprogramming ===\n";
    
    // Computed at compile time!
    constexpr int fact5 = Factorial<5>::value;
    std::cout << "5! = " << fact5 << "\n";
    
    constexpr int fact10 = Factorial<10>::value;
    std::cout << "10! = " << fact10 << "\n";
}

int main() {
    demonstrateFunctionTemplates();
    demonstrateClassTemplates();
    demonstrateSpecialization();
    demonstratePartialSpecialization();
    demonstrateVariadicTemplates();
    demonstrateSFINAE();
    demonstrateTemplateTemplateParameters();
    demonstrateNonTypeTemplateParameters();
    demonstrateTemplateAliases();
    demonstrateFoldExpressions();
    demonstrateIfConstexpr();
    demonstrateTemplateMetaprogramming();
    
    return 0;
}

/*
 * Key Takeaways:
 * 
 * 1. Templates enable generic programming - write code once, use with many types
 * 
 * 2. Template Instantiation:
 *    - Happens at compile time
 *    - Compiler generates code for each type used
 *    - Can lead to code bloat if overused
 * 
 * 3. Specialization allows custom behavior for specific types:
 *    - Full specialization: completely custom implementation
 *    - Partial specialization: customize based on type patterns
 * 
 * 4. Variadic templates handle arbitrary number of arguments
 *    - Recursive unpacking or fold expressions
 *    - Perfect for type-safe printf, tuples, etc.
 * 
 * 5. SFINAE enables conditional compilation based on type traits
 *    - Now often replaced by concepts in C++20
 * 
 * 6. Modern features:
 *    - if constexpr: compile-time branching
 *    - Fold expressions: concise variadic operations
 *    - Template aliases: cleaner type names
 * 
 * Interview Questions:
 * 
 * Q: What's the difference between template specialization and overloading?
 * A: Specialization provides alternative implementation for specific template
 *    arguments. Overloading creates different functions with different signatures.
 *    Specialization participates in template argument deduction.
 * 
 * Q: How does template instantiation work?
 * A: Compiler generates concrete code for each unique type/value combination
 *    used with the template. Happens at compile time. Unused template functions
 *    are never instantiated.
 * 
 * Q: What is SFINAE?
 * A: Substitution Failure Is Not An Error - when template substitution fails,
 *    it's not an error, just removes that overload from consideration. Enables
 *    conditional template instantiation based on type traits.
 * 
 * Q: What are variadic templates?
 * A: Templates that accept variable number of arguments. Use parameter packs
 *    (typename... Args) and pack expansion (args...). Process recursively or
 *    with fold expressions.
 */
