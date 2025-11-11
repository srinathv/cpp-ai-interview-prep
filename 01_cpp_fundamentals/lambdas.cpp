/**
 * Lambda Expressions - Anonymous Functions in C++
 * 
 * Key interview points:
 * - Lambdas are anonymous function objects
 * - Capture modes: [=] (by value), [&] (by reference), [this]
 * - Generic lambdas (C++14) with auto parameters
 * - Lambdas in STL algorithms
 * - Stateful lambdas using mutable
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>

// Example 1: Basic lambda syntax
void basic_lambdas() {
    std::cout << "=== Basic Lambdas ===\n";
    
    // Simple lambda
    auto add = [](int a, int b) { return a + b; };
    std::cout << "5 + 3 = " << add(5, 3) << "\n";
    
    // Lambda with explicit return type
    auto divide = [](double a, double b) -> double {
        if (b == 0) return 0.0;
        return a / b;
    };
    std::cout << "10 / 3 = " << divide(10, 3) << "\n\n";
}

// Example 2: Capture modes
void capture_modes() {
    std::cout << "=== Capture Modes ===\n";
    
    int x = 10, y = 20;
    
    // Capture by value [=]
    auto by_value = [=]() { return x + y; };
    std::cout << "By value: " << by_value() << "\n";
    
    // Capture by reference [&]
    auto by_ref = [&]() { x += 5; return x + y; };
    std::cout << "By reference: " << by_ref() << "\n";
    std::cout << "x is now: " << x << "\n";
    
    // Capture specific variables
    auto mixed = [x, &y]() { y += 10; return x + y; };
    std::cout << "Mixed capture: " << mixed() << "\n";
    std::cout << "y is now: " << y << "\n";
    
    // Capture with initialization (C++14)
    auto init_capture = [z = x * 2]() { return z; };
    std::cout << "Init capture: " << init_capture() << "\n\n";
}

// Example 3: Lambdas with STL algorithms
void lambdas_with_stl() {
    std::cout << "=== Lambdas with STL ===\n";
    
    std::vector<int> vec{5, 2, 8, 1, 9, 3};
    
    // Sort with custom comparator
    std::sort(vec.begin(), vec.end(), [](int a, int b) {
        return a > b;  // Descending order
    });
    
    std::cout << "Sorted (desc): ";
    for (int v : vec) std::cout << v << " ";
    std::cout << "\n";
    
    // Find first even number
    auto it = std::find_if(vec.begin(), vec.end(), [](int x) {
        return x % 2 == 0;
    });
    if (it != vec.end()) {
        std::cout << "First even: " << *it << "\n";
    }
    
    // Count numbers > 5
    int count = std::count_if(vec.begin(), vec.end(), [](int x) {
        return x > 5;
    });
    std::cout << "Numbers > 5: " << count << "\n";
    
    // Transform: multiply by 2
    std::transform(vec.begin(), vec.end(), vec.begin(), [](int x) {
        return x * 2;
    });
    
    std::cout << "Multiplied by 2: ";
    for (int v : vec) std::cout << v << " ";
    std::cout << "\n\n";
}

// Example 4: Generic lambdas (C++14)
void generic_lambdas() {
    std::cout << "=== Generic Lambdas (C++14) ===\n";
    
    // Auto parameter - works with any type
    auto print = [](const auto& x) {
        std::cout << x << "\n";
    };
    
    print(42);
    print(3.14);
    print("Hello");
    
    // Generic binary operation
    auto max = [](const auto& a, const auto& b) {
        return a > b ? a : b;
    };
    
    std::cout << "max(10, 20) = " << max(10, 20) << "\n";
    std::cout << "max(3.14, 2.71) = " << max(3.14, 2.71) << "\n\n";
}

// Example 5: Mutable lambdas (stateful)
void mutable_lambdas() {
    std::cout << "=== Mutable Lambdas ===\n";
    
    int counter = 0;
    
    // Mutable allows modifying captured-by-value variables
    auto increment = [counter]() mutable {
        return ++counter;
    };
    
    std::cout << "Call 1: " << increment() << "\n";
    std::cout << "Call 2: " << increment() << "\n";
    std::cout << "Call 3: " << increment() << "\n";
    std::cout << "Original counter: " << counter << "\n\n";  // Still 0!
}

// Example 6: Returning lambdas (closures)
auto make_adder(int x) {
    return [x](int y) { return x + y; };
}

void lambda_closures() {
    std::cout << "=== Lambda Closures ===\n";
    
    auto add5 = make_adder(5);
    auto add10 = make_adder(10);
    
    std::cout << "add5(3) = " << add5(3) << "\n";
    std::cout << "add10(3) = " << add10(3) << "\n\n";
}

// Example 7: std::function for storing lambdas
void storing_lambdas() {
    std::cout << "=== Storing Lambdas ===\n";
    
    std::vector<std::function<int(int)>> operations;
    
    operations.push_back([](int x) { return x * 2; });
    operations.push_back([](int x) { return x * x; });
    operations.push_back([](int x) { return x + 10; });
    
    int input = 5;
    for (size_t i = 0; i < operations.size(); ++i) {
        std::cout << "Operation " << i << "(" << input << ") = " 
                  << operations[i](input) << "\n";
    }
    std::cout << "\n";
}

// Example 8: Real-world use case - DataFrame-like operations
struct DataPoint {
    std::string name;
    double value;
};

void dataframe_operations() {
    std::cout << "=== DataFrame-like Operations ===\n";
    
    std::vector<DataPoint> data{
        {"Alice", 95.5},
        {"Bob", 87.2},
        {"Charlie", 92.8},
        {"David", 78.5}
    };
    
    // Filter: scores > 90
    std::vector<DataPoint> high_scores;
    std::copy_if(data.begin(), data.end(), std::back_inserter(high_scores),
                 [](const DataPoint& dp) { return dp.value > 90.0; });
    
    std::cout << "High scores (>90):\n";
    for (const auto& dp : high_scores) {
        std::cout << "  " << dp.name << ": " << dp.value << "\n";
    }
    
    // Compute average
    double sum = 0.0;
    std::for_each(data.begin(), data.end(), [&sum](const DataPoint& dp) {
        sum += dp.value;
    });
    std::cout << "Average: " << sum / data.size() << "\n\n";
}

int main() {
    basic_lambdas();
    capture_modes();
    lambdas_with_stl();
    generic_lambdas();
    mutable_lambdas();
    lambda_closures();
    storing_lambdas();
    dataframe_operations();
    
    std::cout << "Key takeaway: Lambdas are essential for\n";
    std::cout << "STL algorithms and functional programming in C++!\n";
    
    return 0;
}
