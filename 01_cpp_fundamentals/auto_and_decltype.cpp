/**
 * Auto and Decltype - Type Deduction in Modern C++
 * 
 * Key interview points:
 * - auto deduces types from initializers
 * - decltype preserves exact type including references
 * - Use auto to avoid verbose type names
 * - Be careful with auto and references
 */

#include <iostream>
#include <vector>
#include <map>
#include <string>

// Example 1: Basic auto usage
void basic_auto() {
    std::cout << "=== Basic Auto Usage ===\n";
    
    auto x = 42;                    // int
    auto y = 3.14;                  // double
    auto s = std::string("hello");  // std::string
    
    std::cout << "x = " << x << " (int)\n";
    std::cout << "y = " << y << " (double)\n";
    std::cout << "s = " << s << " (std::string)\n\n";
}

// Example 2: Auto with iterators (much cleaner!)
void auto_with_iterators() {
    std::cout << "=== Auto with Iterators ===\n";
    
    std::map<std::string, int> ages{{"Alice", 30}, {"Bob", 25}};
    
    // Without auto (verbose!)
    for (std::map<std::string, int>::const_iterator it = ages.begin();
         it != ages.end(); ++it) {
        std::cout << it->first << ": " << it->second << "\n";
    }
    
    // With auto (clean!)
    for (auto it = ages.begin(); it != ages.end(); ++it) {
        std::cout << it->first << ": " << it->second << "\n";
    }
    
    // Even better: range-based for
    for (const auto& [name, age] : ages) {  // C++17 structured binding
        std::cout << name << ": " << age << "\n";
    }
    std::cout << "\n";
}

// Example 3: Auto with references
void auto_with_references() {
    std::cout << "=== Auto with References ===\n";
    
    std::vector<int> vec{1, 2, 3, 4, 5};
    
    // auto copies!
    for (auto val : vec) {
        val *= 2;  // Modifies copy, not original
    }
    std::cout << "After auto copy: " << vec[0] << "\n";  // Still 1
    
    // auto& modifies original
    for (auto& val : vec) {
        val *= 2;  // Modifies original
    }
    std::cout << "After auto&: " << vec[0] << "\n";  // Now 2
    
    // const auto& for read-only (no copy)
    for (const auto& val : vec) {
        std::cout << val << " ";
    }
    std::cout << "\n\n";
}

// Example 4: decltype - preserves exact type
void decltype_examples() {
    std::cout << "=== Decltype Examples ===\n";
    
    int x = 5;
    decltype(x) y = 10;      // y is int
    
    const int& ref = x;
    decltype(ref) z = x;     // z is const int&
    
    std::vector<int> vec{1, 2, 3};
    decltype(vec[0]) first = vec[0];  // first is int& (reference!)
    
    first = 100;  // Modifies vec[0]!
    std::cout << "vec[0] = " << vec[0] << "\n\n";
}

// Example 5: decltype(auto) - perfect forwarding of return type
template<typename Container>
decltype(auto) get_element(Container& c, size_t idx) {
    return c[idx];  // Returns reference if Container::operator[] returns reference
}

void decltype_auto_example() {
    std::cout << "=== Decltype(auto) ===\n";
    
    std::vector<int> vec{10, 20, 30};
    
    // Returns int& (reference), so we can modify
    get_element(vec, 0) = 100;
    
    std::cout << "vec[0] = " << vec[0] << "\n\n";
}

// Example 6: Common pitfalls
void common_pitfalls() {
    std::cout << "=== Common Pitfalls ===\n";
    
    // Pitfall 1: auto doesn't preserve const
    const int x = 42;
    auto y = x;       // y is int, not const int!
    y = 100;          // OK, y is mutable
    
    // Need explicit const
    const auto z = x; // z is const int
    // z = 100;       // Error!
    
    // Pitfall 2: auto doesn't preserve references
    int a = 5;
    int& ref = a;
    auto b = ref;     // b is int (copy), not int&!
    b = 10;           // Doesn't modify a
    
    auto& c = ref;    // c is int& (reference)
    c = 20;           // Modifies a
    
    std::cout << "a = " << a << "\n\n";
}

// Example 7: When to use auto (interview advice)
void when_to_use_auto() {
    std::cout << "=== When to Use Auto ===\n\n";
    
    std::cout << "✓ USE auto when:\n";
    std::cout << "  - Type is obvious from context: auto x = 42;\n";
    std::cout << "  - Type is verbose: auto it = map.begin();\n";
    std::cout << "  - Type is template-dependent\n";
    std::cout << "  - Exact type doesn't matter\n\n";
    
    std::cout << "✗ AVOID auto when:\n";
    std::cout << "  - Type is not obvious: auto data = getData();\n";
    std::cout << "  - Implicit conversions matter\n";
    std::cout << "  - You need specific type for documentation\n\n";
}

int main() {
    basic_auto();
    auto_with_iterators();
    auto_with_references();
    decltype_examples();
    decltype_auto_example();
    common_pitfalls();
    when_to_use_auto();
    
    return 0;
}
