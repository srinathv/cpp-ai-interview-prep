/*
 * C++20 Ranges Library
 * 
 * Key Concepts:
 * - Range views (lazy evaluation)
 * - Range adaptors
 * - Range algorithms
 * - Composability and pipelines
 * - Projections
 * 
 * Interview Topics:
 * - What are the benefits of ranges over traditional iterators?
 * - How do range views enable lazy evaluation?
 * - What is the difference between views and actions?
 */

#include <iostream>
#include <vector>
#include <ranges>
#include <algorithm>
#include <string>
#include <numeric>

namespace rng = std::ranges;
namespace views = std::views;

// Example 1: Basic range concepts
void demonstrateBasicRanges() {
    std::cout << "=== Basic Ranges ===\n";
    
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // Traditional approach
    std::cout << "Traditional: ";
    for (auto it = numbers.begin(); it != numbers.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << "\n";
    
    // Range-based for loop
    std::cout << "Range-based: ";
    for (auto n : numbers) {
        std::cout << n << " ";
    }
    std::cout << "\n";
    
    // Range algorithm
    std::cout << "Range algorithm: ";
    rng::for_each(numbers, [](int n) { std::cout << n << " "; });
    std::cout << "\n";
}

// Example 2: Views - lazy evaluation
void demonstrateViews() {
    std::cout << "\n=== Views (Lazy Evaluation) ===\n";
    
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // Views are lazy - no computation until iteration
    auto evens = numbers | views::filter([](int n) { return n % 2 == 0; });
    
    std::cout << "Even numbers: ";
    for (int n : evens) {
        std::cout << n << " ";
    }
    std::cout << "\n";
    
    // Views can be composed
    auto result = numbers 
        | views::filter([](int n) { return n % 2 == 0; })  // Keep evens
        | views::transform([](int n) { return n * n; })     // Square them
        | views::take(3);                                    // Take first 3
    
    std::cout << "First 3 squared evens: ";
    for (int n : result) {
        std::cout << n << " ";
    }
    std::cout << "\n";
}

// Example 3: Common range adaptors
void demonstrateAdaptors() {
    std::cout << "\n=== Range Adaptors ===\n";
    
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // filter - keep elements matching predicate
    std::cout << "Filter (> 5): ";
    for (int n : numbers | views::filter([](int n) { return n > 5; })) {
        std::cout << n << " ";
    }
    std::cout << "\n";
    
    // transform - apply function to each element
    std::cout << "Transform (*2): ";
    for (int n : numbers | views::transform([](int n) { return n * 2; })) {
        std::cout << n << " ";
    }
    std::cout << "\n";
    
    // take - first N elements
    std::cout << "Take 5: ";
    for (int n : numbers | views::take(5)) {
        std::cout << n << " ";
    }
    std::cout << "\n";
    
    // drop - skip first N elements
    std::cout << "Drop 5: ";
    for (int n : numbers | views::drop(5)) {
        std::cout << n << " ";
    }
    std::cout << "\n";
    
    // reverse
    std::cout << "Reverse: ";
    for (int n : numbers | views::reverse) {
        std::cout << n << " ";
    }
    std::cout << "\n";
}

// Example 4: More advanced views
void demonstrateAdvancedViews() {
    std::cout << "\n=== Advanced Views ===\n";
    
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    
    // take_while - take elements while predicate is true
    std::cout << "Take while < 4: ";
    for (int n : numbers | views::take_while([](int n) { return n < 4; })) {
        std::cout << n << " ";
    }
    std::cout << "\n";
    
    // drop_while - skip elements while predicate is true
    std::cout << "Drop while < 4: ";
    for (int n : numbers | views::drop_while([](int n) { return n < 4; })) {
        std::cout << n << " ";
    }
    std::cout << "\n";
    
    // split - split range into subranges
    std::string text = "hello world from ranges";
    std::cout << "Split by space:\n";
    for (auto word : text | views::split(' ')) {
        for (char c : word) {
            std::cout << c;
        }
        std::cout << "\n";
    }
    
    // join - flatten nested ranges
    std::vector<std::vector<int>> nested = {{1, 2}, {3, 4}, {5, 6}};
    std::cout << "Join nested: ";
    for (int n : nested | views::join) {
        std::cout << n << " ";
    }
    std::cout << "\n";
}

// Example 5: iota and other generators
void demonstrateGenerators() {
    std::cout << "\n=== Range Generators ===\n";
    
    // iota - infinite sequence starting from value
    std::cout << "First 10 from iota(1): ";
    for (int n : views::iota(1) | views::take(10)) {
        std::cout << n << " ";
    }
    std::cout << "\n";
    
    // iota with limit
    std::cout << "iota(5, 15): ";
    for (int n : views::iota(5, 15)) {
        std::cout << n << " ";
    }
    std::cout << "\n";
    
    // repeat - repeat a value (C++23, may not be available)
    // std::cout << "Repeat 42, 5 times: ";
    // for (int n : views::repeat(42) | views::take(5)) {
    //     std::cout << n << " ";
    // }
    // std::cout << "\n";
}

// Example 6: Projections in range algorithms
struct Person {
    std::string name;
    int age;
};

void demonstrateProjections() {
    std::cout << "\n=== Projections ===\n";
    
    std::vector<Person> people = {
        {"Alice", 30},
        {"Bob", 25},
        {"Charlie", 35},
        {"Diana", 28}
    };
    
    // Sort by age using projection
    rng::sort(people, {}, &Person::age);
    
    std::cout << "Sorted by age:\n";
    for (const auto& p : people) {
        std::cout << p.name << " (" << p.age << ")\n";
    }
    
    // Find by name using projection
    auto it = rng::find(people, "Charlie", &Person::name);
    if (it != people.end()) {
        std::cout << "Found: " << it->name << ", age " << it->age << "\n";
    }
}

// Example 7: Range algorithms vs traditional algorithms
void demonstrateRangeAlgorithms() {
    std::cout << "\n=== Range Algorithms ===\n";
    
    std::vector<int> numbers = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    
    // Traditional: requires begin/end iterators
    // std::sort(numbers.begin(), numbers.end());
    
    // Ranges: can pass the whole container
    rng::sort(numbers);
    
    std::cout << "Sorted: ";
    for (int n : numbers) {
        std::cout << n << " ";
    }
    std::cout << "\n";
    
    // count
    int count = rng::count(numbers, 5);
    std::cout << "Count of 5: " << count << "\n";
    
    // find
    auto it = rng::find(numbers, 7);
    if (it != numbers.end()) {
        std::cout << "Found 7 at position " << (it - numbers.begin()) << "\n";
    }
    
    // any_of, all_of, none_of
    bool has_even = rng::any_of(numbers, [](int n) { return n % 2 == 0; });
    bool all_positive = rng::all_of(numbers, [](int n) { return n > 0; });
    
    std::cout << "Has even: " << has_even << "\n";
    std::cout << "All positive: " << all_positive << "\n";
}

// Example 8: Composing complex pipelines
void demonstrateComplexPipelines() {
    std::cout << "\n=== Complex Pipelines ===\n";
    
    // Generate squares of even numbers from 1 to 20
    auto result = views::iota(1, 21)
        | views::filter([](int n) { return n % 2 == 0; })
        | views::transform([](int n) { return n * n; });
    
    std::cout << "Squares of evens 1-20: ";
    for (int n : result) {
        std::cout << n << " ";
    }
    std::cout << "\n";
    
    // Fibonacci-like processing
    std::vector<int> fibs = {1, 1, 2, 3, 5, 8, 13, 21, 34, 55};
    
    auto processed = fibs
        | views::drop(2)              // Skip first two
        | views::take_while([](int n) { return n < 30; })  // Until < 30
        | views::transform([](int n) { return n * 2; })     // Double
        | views::reverse;             // Reverse order
    
    std::cout << "Processed fibonacci: ";
    for (int n : processed) {
        std::cout << n << " ";
    }
    std::cout << "\n";
}

// Example 9: Zip view (C++23, may not be available in all compilers)
void demonstrateZip() {
    std::cout << "\n=== Zip (if available) ===\n";
    
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    std::vector<std::string> words = {"one", "two", "three", "four", "five"};
    
    // C++23 zip view
    // for (auto [num, word] : views::zip(numbers, words)) {
    //     std::cout << num << " -> " << word << "\n";
    // }
    
    // Manual zip alternative for pre-C++23
    std::cout << "Manual zip:\n";
    for (size_t i = 0; i < std::min(numbers.size(), words.size()); ++i) {
        std::cout << numbers[i] << " -> " << words[i] << "\n";
    }
}

// Example 10: Practical example - data processing
struct Student {
    std::string name;
    int score;
    bool isActive;
};

void demonstratePracticalExample() {
    std::cout << "\n=== Practical Example ===\n";
    
    std::vector<Student> students = {
        {"Alice", 85, true},
        {"Bob", 92, true},
        {"Charlie", 78, false},
        {"Diana", 95, true},
        {"Eve", 88, true},
        {"Frank", 72, false}
    };
    
    // Get top 3 active students with score >= 85, sorted by score
    auto topStudents = students
        | views::filter(&Student::isActive)
        | views::filter([](const Student& s) { return s.score >= 85; })
        | views::transform([](const Student& s) { return s; });
    
    std::vector<Student> result(rng::begin(topStudents), rng::end(topStudents));
    rng::sort(result, rng::greater{}, &Student::score);
    
    std::cout << "Top active students (score >= 85):\n";
    for (const auto& s : result | views::take(3)) {
        std::cout << s.name << ": " << s.score << "\n";
    }
}

// Example 11: Performance considerations
void demonstratePerformance() {
    std::cout << "\n=== Performance Considerations ===\n";
    
    std::vector<int> large(1000);
    std::iota(large.begin(), large.end(), 1);
    
    // Views are lazy - no intermediate containers created
    auto pipeline = large
        | views::filter([](int n) { return n % 2 == 0; })
        | views::transform([](int n) { return n * n; })
        | views::take(10);
    
    // Computation happens only during iteration
    std::cout << "First 10 squared evens from 1-1000: ";
    for (int n : pipeline) {
        std::cout << n << " ";
    }
    std::cout << "\n";
    
    std::cout << "Note: No intermediate vectors created!\n";
}

int main() {
    demonstrateBasicRanges();
    demonstrateViews();
    demonstrateAdaptors();
    demonstrateAdvancedViews();
    demonstrateGenerators();
    demonstrateProjections();
    demonstrateRangeAlgorithms();
    demonstrateComplexPipelines();
    demonstrateZip();
    demonstratePracticalExample();
    demonstratePerformance();
    
    return 0;
}

/*
 * Key Takeaways:
 * 
 * 1. Ranges provide a more expressive and composable way to work with sequences
 * 
 * 2. Views are lazy - they don't create intermediate containers
 *    - Better performance and memory usage
 *    - Enable infinite sequences
 * 
 * 3. Pipe operator (|) enables readable composition
 *    - data | filter | transform | take
 *    - Left-to-right reading, like Unix pipes
 * 
 * 4. Range algorithms work with whole containers, not just iterator pairs
 *    - Simpler syntax
 *    - Support projections natively
 * 
 * 5. Projections eliminate need for custom comparators
 *    - sort(people, {}, &Person::age)
 *    - More readable and less code
 * 
 * 6. Common views:
 *    - filter, transform: modify/select elements
 *    - take, drop: subsequences
 *    - reverse, split, join: restructure
 *    - iota: generate sequences
 * 
 * Interview Questions:
 * 
 * Q: What's the difference between a range view and a range action?
 * A: Views are lazy and don't modify the underlying data - they create
 *    lightweight adapters. Actions (not in standard yet) eagerly create
 *    new containers with modifications applied.
 * 
 * Q: Why are range views more efficient than traditional approaches?
 * A: Views are lazy and composable - no intermediate containers created.
 *    Traditional approach with multiple transforms creates intermediate
 *    vectors at each step.
 * 
 * Q: What is a projection in ranges?
 * A: A projection is a callable that transforms elements before applying
 *    an operation. Allows sorting/comparing by a member without custom
 *    comparators: sort(people, {}, &Person::age)
 * 
 * Q: Can you modify elements through a view?
 * A: Some views allow modification (like transform with non-const lambda),
 *    but many are read-only. Views generally don't own data, they adapt it.
 * 
 * Compiler requirements:
 * - Requires C++20 support
 * - Compile with: g++ -std=c++20 ranges.cpp
 */
