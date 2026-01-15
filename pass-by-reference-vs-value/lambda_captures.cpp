#include <iostream>
#include <functional>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>

// Lambda Captures: By Value vs By Reference

// Pattern 1: Capture by value vs reference
void basicCaptures() {
    std::cout << "=== Basic Captures ===" << std::endl;
    
    int x = 10;
    std::string str = "Hello";
    
    // Capture by value - creates copies
    auto byValue = [x, str]() {
        std::cout << "Captured by value: x=" << x << ", str=" << str << std::endl;
        // x++; // ERROR: cannot modify captured value
    };
    
    // Capture by reference - no copies
    auto byRef = [&x, &str]() {
        std::cout << "Captured by reference: x=" << x << ", str=" << str << std::endl;
        x++;  // OK: can modify
        str += " World";
    };
    
    // Mutable capture by value
    auto byValueMutable = [x, str]() mutable {
        x++;  // OK: modifies the copy
        str += "!";
        std::cout << "Mutable capture: x=" << x << ", str=" << str << std::endl;
    };
    
    byValue();
    std::cout << "After byValue - x=" << x << ", str=" << str << std::endl;
    
    byRef();
    std::cout << "After byRef - x=" << x << ", str=" << str << std::endl;
    
    byValueMutable();
    std::cout << "After byValueMutable - x=" << x << ", str=" << str << std::endl;
}

// Pattern 2: Capture all by value [=] vs by reference [&]
void captureAll() {
    std::cout << "\n=== Capture All ===" << std::endl;
    
    int a = 1, b = 2, c = 3;
    
    auto captureAllByValue = [=]() {
        std::cout << "All by value: " << a << " " << b << " " << c << std::endl;
    };
    
    auto captureAllByRef = [&]() {
        std::cout << "All by ref: " << a << " " << b << " " << c << std::endl;
        a++; b++; c++;
    };
    
    captureAllByValue();
    captureAllByRef();
    std::cout << "After captures: a=" << a << " b=" << b << " c=" << c << std::endl;
}

// Pattern 3: Mixed captures
void mixedCaptures() {
    std::cout << "\n=== Mixed Captures ===" << std::endl;
    
    int counter = 0;
    std::string msg = "Count";
    
    // Capture counter by ref, msg by value
    auto mixed = [&counter, msg]() {
        counter++;
        std::cout << msg << ": " << counter << std::endl;
    };
    
    mixed();
    mixed();
    std::cout << "Final counter: " << counter << std::endl;
}

// Pattern 4: Init captures (C++14) - generalized lambda capture
void initCaptures() {
    std::cout << "\n=== Init Captures ===" << std::endl;
    
    auto ptr = std::make_unique<int>(42);
    
    // Capture by move
    auto lambda = [p = std::move(ptr)]() {
        std::cout << "Moved unique_ptr value: " << *p << std::endl;
    };
    
    // ptr is now nullptr
    std::cout << "Original ptr is null: " << (ptr == nullptr) << std::endl;
    lambda();
    
    // Init capture with expression
    int x = 10;
    auto lambda2 = [y = x * 2]() {
        std::cout << "Init capture y = " << y << std::endl;
    };
    lambda2();
}

// Pattern 5: Capturing this
class Counter {
    int count = 0;
    
public:
    void increment() {
        // Capture this by value (C++17: [*this] copies the object)
        auto byThis = [this]() {
            count++;  // Modifies member
            std::cout << "Count (this): " << count << std::endl;
        };
        
        // C++17: Capture *this by value
        auto byCopy = [*this]() mutable {
            count++;  // Modifies the copy
            std::cout << "Count (copy): " << count << std::endl;
        };
        
        byThis();
        byCopy();
        std::cout << "Actual count: " << count << std::endl;
    }
};

// Pattern 6: Returning lambdas - dangling references
std::function<void()> createLambda() {
    int local = 100;
    
    // DANGER: Captures local by reference
    // auto bad = [&local]() {
    //     std::cout << local << std::endl;  // Dangling reference!
    // };
    
    // SAFE: Capture by value
    auto good = [local]() {
        std::cout << "Safe capture: " << local << std::endl;
    };
    
    return good;
}

// Pattern 7: Lambdas with auto parameters (C++14)
void genericLambdas() {
    std::cout << "\n=== Generic Lambdas ===" << std::endl;
    
    auto printer = [](const auto& item) {
        std::cout << "Item: " << item << std::endl;
    };
    
    printer(42);
    printer("Hello");
    printer(3.14);
    
    // Perfect forwarding in generic lambda
    auto forwarder = [](auto&& arg) {
        std::cout << "Forwarding..." << std::endl;
        return std::forward<decltype(arg)>(arg);
    };
    
    int x = 10;
    forwarder(x);        // Lvalue
    forwarder(20);       // Rvalue
}

// Pattern 8: Capture and transform with STL algorithms
void capturesWithSTL() {
    std::cout << "\n=== Captures with STL ===" << std::endl;
    
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    int multiplier = 10;
    
    // Capture multiplier by value
    std::vector<int> result;
    std::transform(numbers.begin(), numbers.end(), 
                   std::back_inserter(result),
                   [multiplier](int n) { return n * multiplier; });
    
    std::cout << "Multiplied by " << multiplier << ": ";
    for (int n : result) std::cout << n << " ";
    std::cout << std::endl;
    
    // Capture by reference to accumulate
    int sum = 0;
    std::for_each(numbers.begin(), numbers.end(),
                  [&sum](int n) { sum += n; });
    std::cout << "Sum: " << sum << std::endl;
}

// Pattern 9: Capture with std::function
void captureWithFunction() {
    std::cout << "\n=== Capture with std::function ===" << std::endl;
    
    int state = 0;
    
    std::function<void()> func = [&state]() {
        state++;
        std::cout << "State: " << state << std::endl;
    };
    
    func();
    func();
    func();
    std::cout << "Final state: " << state << std::endl;
}

// Pattern 10: Immediately invoked lambda (IIFE)
void immediatelyInvokedLambda() {
    std::cout << "\n=== Immediately Invoked Lambda ===" << std::endl;
    
    int x = 10;
    int y = 20;
    
    // IIFE for complex initialization
    const int result = [&]() {
        if (x > y) return x * 2;
        else return y * 3;
    }();
    
    std::cout << "Result from IIFE: " << result << std::endl;
}

int main() {
    basicCaptures();
    captureAll();
    mixedCaptures();
    initCaptures();
    
    std::cout << "\n=== Capturing this ===" << std::endl;
    Counter c;
    c.increment();
    
    std::cout << "\n=== Returning lambda ===" << std::endl;
    auto lambda = createLambda();
    lambda();
    
    genericLambdas();
    capturesWithSTL();
    captureWithFunction();
    immediatelyInvokedLambda();
    
    return 0;
}
