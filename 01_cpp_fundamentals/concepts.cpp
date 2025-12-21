/*
 * C++20 Concepts
 * 
 * Key Concepts:
 * - Defining custom concepts
 * - Standard library concepts
 * - Requires clauses and expressions
 * - Concept composition
 * - Constraining templates
 * 
 * Interview Topics:
 * - What are concepts and why do we need them?
 * - How do concepts improve error messages?
 * - What's the difference between SFINAE and concepts?
 */

#include <iostream>
#include <concepts>
#include <vector>
#include <string>
#include <type_traits>

// Example 1: Basic concept definition
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

// Use concept in template constraint
template<Numeric T>
T add(T a, T b) {
    return a + b;
}

void demonstrateBasicConcepts() {
    std::cout << "=== Basic Concepts ===\n";
    
    std::cout << "add(5, 10) = " << add(5, 10) << "\n";
    std::cout << "add(3.14, 2.72) = " << add(3.14, 2.72) << "\n";
    
    // This would fail at compile time with clear error:
    // std::cout << add(std::string("hello"), std::string("world")) << "\n";
}

// Example 2: Standard library concepts
void demonstrateStandardConcepts() {
    std::cout << "\n=== Standard Library Concepts ===\n";
    
    // std::integral
    auto printIfIntegral = []<std::integral T>(T value) {
        std::cout << "Integral value: " << value << "\n";
    };
    
    printIfIntegral(42);
    printIfIntegral('A');
    // printIfIntegral(3.14); // Error: not integral
    
    // std::floating_point
    auto printIfFloat = []<std::floating_point T>(T value) {
        std::cout << "Floating point value: " << value << "\n";
    };
    
    printIfFloat(3.14);
    printIfFloat(2.72f);
}

// Example 3: Requires clauses
template<typename T>
requires std::integral<T>
T multiply(T a, T b) {
    return a * b;
}

// Alternative syntax
template<typename T>
T divide(T a, T b) requires std::floating_point<T> {
    return a / b;
}

// Trailing requires clause
auto square(auto x) requires std::is_arithmetic_v<decltype(x)> {
    return x * x;
}

void demonstrateRequiresClauses() {
    std::cout << "\n=== Requires Clauses ===\n";
    
    std::cout << "multiply(6, 7) = " << multiply(6, 7) << "\n";
    std::cout << "divide(10.0, 3.0) = " << divide(10.0, 3.0) << "\n";
    std::cout << "square(5) = " << square(5) << "\n";
}

// Example 4: Requires expressions
template<typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
};

template<typename T>
concept Printable = requires(T t) {
    { std::cout << t } -> std::same_as<std::ostream&>;
};

template<Addable T>
T sum(T a, T b) {
    return a + b;
}

template<Printable T>
void print(const T& value) {
    std::cout << "Value: " << value << "\n";
}

void demonstrateRequiresExpressions() {
    std::cout << "\n=== Requires Expressions ===\n";
    
    std::cout << "sum(5, 10) = " << sum(5, 10) << "\n";
    std::cout << "sum(strings) = " << sum(std::string("Hello "), std::string("World")) << "\n";
    
    print(42);
    print(3.14);
    print(std::string("Hello"));
}

// Example 5: Complex concept with multiple requirements
template<typename T>
concept Container = requires(T t) {
    typename T::value_type;
    typename T::iterator;
    { t.begin() } -> std::same_as<typename T::iterator>;
    { t.end() } -> std::same_as<typename T::iterator>;
    { t.size() } -> std::convertible_to<size_t>;
};

template<Container C>
void printContainer(const C& container) {
    std::cout << "Container size: " << container.size() << ", elements: ";
    for (const auto& elem : container) {
        std::cout << elem << " ";
    }
    std::cout << "\n";
}

void demonstrateComplexConcepts() {
    std::cout << "\n=== Complex Concepts ===\n";
    
    std::vector<int> vec = {1, 2, 3, 4, 5};
    printContainer(vec);
    
    std::vector<std::string> words = {"hello", "world"};
    printContainer(words);
}

// Example 6: Concept composition
template<typename T>
concept SignedNumeric = Numeric<T> && std::is_signed_v<T>;

template<typename T>
concept UnsignedNumeric = Numeric<T> && std::is_unsigned_v<T>;

template<SignedNumeric T>
T absolute(T value) {
    return value < 0 ? -value : value;
}

void demonstrateConceptComposition() {
    std::cout << "\n=== Concept Composition ===\n";
    
    std::cout << "absolute(-42) = " << absolute(-42) << "\n";
    std::cout << "absolute(-3.14) = " << absolute(-3.14) << "\n";
    
    // This would fail - unsigned types don't satisfy SignedNumeric
    // unsigned int x = 42;
    // std::cout << absolute(x) << "\n";
}

// Example 7: Concept subsumption
template<typename T>
concept Incrementable = requires(T t) {
    { ++t } -> std::same_as<T&>;
};

template<typename T>
concept Decrementable = requires(T t) {
    { --t } -> std::same_as<T&>;
};

template<typename T>
concept Bidirectional = Incrementable<T> && Decrementable<T>;

// More specific overload chosen when both match
template<Incrementable T>
void advance(T& value) {
    std::cout << "Incrementable advance\n";
    ++value;
}

template<Bidirectional T>
void advance(T& value) {
    std::cout << "Bidirectional advance (more specific)\n";
    ++value;
}

void demonstrateSubsumption() {
    std::cout << "\n=== Concept Subsumption ===\n";
    
    int x = 5;
    advance(x); // Calls Bidirectional version (more specific)
}

// Example 8: Custom concepts for specific use cases
template<typename T>
concept Sortable = requires(T a, T b) {
    { a < b } -> std::convertible_to<bool>;
    { a > b } -> std::convertible_to<bool>;
};

template<typename T>
concept Range = requires(T t) {
    { t.begin() };
    { t.end() };
    { *t.begin() };
};

template<Range R>
requires Sortable<typename R::value_type>
void customSort(R& range) {
    std::cout << "Sorting range...\n";
    // Simplified - actual sort implementation would go here
}

void demonstrateCustomConcepts() {
    std::cout << "\n=== Custom Concepts ===\n";
    
    std::vector<int> numbers = {5, 2, 8, 1, 9};
    customSort(numbers);
}

// Example 9: Concepts with multiple template parameters
template<typename T, typename U>
concept Comparable = requires(T t, U u) {
    { t == u } -> std::convertible_to<bool>;
    { t != u } -> std::convertible_to<bool>;
};

template<typename T, typename U>
requires Comparable<T, U>
bool areEqual(const T& a, const U& b) {
    return a == b;
}

void demonstrateMultiParameterConcepts() {
    std::cout << "\n=== Multi-Parameter Concepts ===\n";
    
    std::cout << "areEqual(42, 42) = " << areEqual(42, 42) << "\n";
    std::cout << "areEqual(3.14, 3.14) = " << areEqual(3.14, 3.14) << "\n";
    std::cout << "areEqual(5, 5.0) = " << areEqual(5, 5.0) << "\n";
}

// Example 10: Concepts with class templates
template<typename T>
concept Hashable = requires(T t) {
    { std::hash<T>{}(t) } -> std::convertible_to<size_t>;
};

template<Hashable T>
class HashSet {
private:
    std::vector<T> data;

public:
    void insert(const T& value) {
        size_t hash = std::hash<T>{}(value);
        std::cout << "Inserting with hash: " << hash << "\n";
        data.push_back(value);
    }
    
    size_t size() const { return data.size(); }
};

void demonstrateConceptsWithClasses() {
    std::cout << "\n=== Concepts with Class Templates ===\n";
    
    HashSet<int> intSet;
    intSet.insert(42);
    intSet.insert(100);
    
    HashSet<std::string> stringSet;
    stringSet.insert("hello");
    stringSet.insert("world");
}

// Example 11: Concepts vs SFINAE comparison
// Old way with SFINAE
template<typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
oldSquare(T value) {
    return value * value;
}

// New way with concepts
template<std::integral T>
T newSquare(T value) {
    return value * value;
}

void demonstrateConceptsVsSFINAE() {
    std::cout << "\n=== Concepts vs SFINAE ===\n";
    
    std::cout << "oldSquare(5) = " << oldSquare(5) << "\n";
    std::cout << "newSquare(5) = " << newSquare(5) << "\n";
    
    std::cout << "\nConcepts provide:\n";
    std::cout << "- Better error messages\n";
    std::cout << "- More readable code\n";
    std::cout << "- Subsumption for overload resolution\n";
}

// Example 12: Concepts for function objects
template<typename F, typename... Args>
concept Callable = requires(F f, Args... args) {
    { f(args...) };
};

template<typename F, typename T>
requires Callable<F, T>
void applyToEach(const std::vector<T>& vec, F func) {
    for (const auto& elem : vec) {
        func(elem);
    }
}

void demonstrateCallableConcept() {
    std::cout << "\n=== Callable Concept ===\n";
    
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    
    applyToEach(numbers, [](int n) {
        std::cout << n * 2 << " ";
    });
    std::cout << "\n";
}

// Example 13: Concepts for iterators
template<typename T>
concept InputIterator = requires(T it) {
    { *it };
    { ++it } -> std::same_as<T&>;
    { it++ } -> std::convertible_to<T>;
};

template<InputIterator It>
void printRange(It begin, It end) {
    std::cout << "Range: ";
    for (auto it = begin; it != end; ++it) {
        std::cout << *it << " ";
    }
    std::cout << "\n";
}

void demonstrateIteratorConcepts() {
    std::cout << "\n=== Iterator Concepts ===\n";
    
    std::vector<int> vec = {10, 20, 30, 40, 50};
    printRange(vec.begin(), vec.end());
}

int main() {
    demonstrateBasicConcepts();
    demonstrateStandardConcepts();
    demonstrateRequiresClauses();
    demonstrateRequiresExpressions();
    demonstrateComplexConcepts();
    demonstrateConceptComposition();
    demonstrateSubsumption();
    demonstrateCustomConcepts();
    demonstrateMultiParameterConcepts();
    demonstrateConceptsWithClasses();
    demonstrateConceptsVsSFINAE();
    demonstrateCallableConcept();
    demonstrateIteratorConcepts();
    
    return 0;
}

/*
 * Key Takeaways:
 * 
 * 1. Concepts provide compile-time constraints on template parameters
 *    - Better than SFINAE: clearer intent, better errors
 *    - Self-documenting: concept name describes requirement
 * 
 * 2. Syntax variants:
 *    - template<ConceptName T> - most common
 *    - requires ConceptName<T> - explicit requires clause
 *    - template<typename T> requires ... - trailing requires
 * 
 * 3. Requires expressions check:
 *    - Type requirements: typename T::value_type
 *    - Simple requirements: { expr }
 *    - Type requirements: { expr } -> Type
 *    - Compound requirements: { expr } -> Concept<Type>
 * 
 * 4. Standard library concepts (C++20):
 *    - std::integral, std::floating_point
 *    - std::same_as, std::convertible_to, std::derived_from
 *    - std::copyable, std::movable
 *    - Iterator concepts: input_iterator, forward_iterator, etc.
 * 
 * 5. Concept composition:
 *    - Combine with &&, ||
 *    - Subsumption rules for overload resolution
 *    - More specific concepts chosen automatically
 * 
 * 6. Benefits over SFINAE:
 *    - Much clearer error messages
 *    - More readable code
 *    - Can overload based on concepts (subsumption)
 *    - Checked at point of use, not instantiation
 * 
 * Interview Questions:
 * 
 * Q: What problem do concepts solve?
 * A: Concepts provide compile-time constraints on templates with clear,
 *    readable syntax and better error messages. Before concepts, we used
 *    SFINAE which was verbose and produced cryptic errors.
 * 
 * Q: What's the difference between a requires clause and a requires expression?
 * A: Requires clause constrains a template parameter (requires IntegralType<T>).
 *    Requires expression defines what operations must be valid on a type
 *    (requires(T t) { t + t; }).
 * 
 * Q: How do concepts improve error messages?
 * A: With concepts, compiler checks constraints at template declaration and
 *    produces clear errors like "T does not satisfy Sortable". With SFINAE,
 *    you get pages of template instantiation errors.
 * 
 * Q: What is concept subsumption?
 * A: When multiple overloads match, the most specific concept is chosen.
 *    If concept A requires everything B requires plus more, A subsumes B,
 *    and A's overload is selected when both match.
 * 
 * Q: Can you use concepts with non-template functions?
 * A: Concepts are for constraining templates. But you can use abbreviated
 *    function templates with auto parameters: void f(ConceptName auto x)
 * 
 * Compiler requirements:
 * - Requires C++20 support
 * - Compile with: g++ -std=c++20 concepts.cpp
 */
