# Pass by Reference vs Pass by Value in C++

Comprehensive examples demonstrating when to use pass-by-reference versus pass-by-value in C++, including modern C++ features and best practices.

## Table of Contents
- [Quick Reference Guide](#quick-reference-guide)
- [Files in This Directory](#files-in-this-directory)
- [Decision Trees](#decision-trees)
- [Key Takeaways](#key-takeaways)
- [Common Interview Questions](#common-interview-questions)

## Quick Reference Guide

### Pass by VALUE
```cpp
void func(int x)
void func(std::string str)  // Modern: for sink parameters
```
**Use when:**
- Small types (int, char, bool, float, double, pointers) - ≤ 16 bytes
- Function takes ownership (sink parameters) - Modern C++
- Cheap-to-move types with move semantics
- Explicit copy needed for local modifications

**Advantages:**
- Simple and safe (no unintended modifications)
- Good for primitive types
- Enables move semantics for sink functions

**Disadvantages:**
- Expensive for large non-movable objects
- Cannot modify the original variable

---

### Pass by REFERENCE
```cpp
void func(int& x)
void func(std::vector<int>& vec)
```
**Use when:**
- Need to modify the original variable
- Implementing swap, output parameters
- Builder/fluent interface patterns
- Large objects where modification is required

**Advantages:**
- No copy overhead
- Can modify original data
- Efficient for large objects

**Disadvantages:**
- Caller's data can be modified (side effects)
- Requires lvalue (cannot pass temporary)
- Less clear about ownership semantics

---

### Pass by CONST REFERENCE
```cpp
void func(const std::string& str)
void func(const std::vector<int>& vec)
```
**Use when:**
- Large objects that shouldn't be modified
- Reading from containers (vector, string, map, etc.)
- **Most function parameters in modern C++**
- Want efficiency without allowing modifications
- Observing smart pointers without ownership

**Advantages:**
- No copy overhead
- Cannot accidentally modify original
- Can accept temporaries and rvalues
- Best practice for most complex types
- Clear intent: read-only access

**Disadvantages:**
- Slightly more complex syntax
- Small overhead for tiny types (use value instead)

---

### Pass by RVALUE REFERENCE (&&)
```cpp
void func(std::string&& str)
void func(std::vector<int>&& vec)
```
**Use when:**
- Explicitly want to accept only rvalues
- Move constructors and move assignment operators
- Rarely used in regular functions (use pass-by-value instead)

---

### Pass by FORWARDING REFERENCE (template T&&)
```cpp
template<typename T>
void func(T&& arg)
```
**Use when:**
- Perfect forwarding in templates
- Factory functions and wrappers
- Variadic templates
- Generic code that preserves value categories

---

## Files in This Directory

### Basic Concepts

1. **basic_examples.cpp**
   - Fundamental examples of all three passing mechanisms
   - Pass by value, reference, and const reference
   - Clear behavior demonstrations
   - Good starting point for beginners

2. **performance_comparison.cpp**
   - Demonstrates performance impact of copying large objects
   - Shows when const reference prevents expensive copies
   - Compares different approaches
   - Includes copy/move constructor tracking

3. **common_patterns.cpp**
   - Real-world patterns and use cases
   - Swap functions, output parameters
   - Range-based for loops
   - Fluent interfaces and method chaining
   - Factory functions
   - Container operations

### Modern C++ Features

4. **move_semantics.cpp**
   - Move constructors and move assignment
   - Rvalue references (&&)
   - std::move usage and semantics
   - Pass by value for sink functions
   - Move-only types (unique_ptr)
   - Modern string parameter passing
   - When to use std::move vs when to avoid it

5. **perfect_forwarding.cpp**
   - Forwarding references (T&&)
   - std::forward for perfect forwarding
   - Reference collapsing rules
   - Variadic templates with forwarding
   - Factory functions with perfect forwarding
   - Type deduction with forwarding references
   - std::invoke with forwarding

6. **lambda_captures.cpp**
   - Capture by value [=] vs by reference [&]
   - Mutable lambdas
   - Init captures (C++14)
   - Capturing this and *this
   - Generic lambdas with auto
   - Returning lambdas safely
   - std::function with captures
   - Immediately invoked function expressions (IIFE)

7. **smart_pointers.cpp**
   - unique_ptr passing patterns
   - shared_ptr passing patterns
   - weak_ptr usage
   - When to pass by value (ownership transfer)
   - When to pass by const reference (observation)
   - When to use raw pointers from smart pointers
   - Smart pointers in containers
   - Best practices and antipatterns

8. **rvo_and_optimization.cpp**
   - Return Value Optimization (RVO)
   - Named Return Value Optimization (NRVO)
   - Copy elision (C++17 guaranteed)
   - When RVO applies and when it doesn't
   - Performance comparisons
   - Why NOT to use std::move on return
   - std::optional returns
   - Builder pattern with move semantics

9. **const_correctness.cpp**
   - Const member functions
   - Const overloading
   - Top-level vs low-level const
   - Const with pointers and references
   - Const with smart pointers
   - Mutable keyword
   - Const propagation in hierarchies
   - Optimization benefits of const
   - const_cast (and why to avoid it)

## Decision Trees

### Basic Decision Tree
```
Is it a primitive type (int, char, bool, float, double, pointer)?
├─ YES → Pass by VALUE
└─ NO → Is it a complex type (string, vector, custom class)?
    ├─ YES → Do you need to modify it?
    │   ├─ YES → Pass by REFERENCE
    │   └─ NO → Pass by CONST REFERENCE
    └─ Is it a small POD struct (≤16 bytes)?
        ├─ YES → Pass by VALUE or CONST REFERENCE (either works)
        └─ NO → Pass by CONST REFERENCE
```

### Modern C++ Decision Tree (with move semantics)
```
What's the intent?
├─ Transfer ownership (sink)?
│   └─ Pass by VALUE + std::move internally
├─ Modify the original?
│   └─ Pass by non-const REFERENCE
├─ Read-only access?
│   ├─ Small type (≤16 bytes)? → Pass by VALUE
│   └─ Large type? → Pass by CONST REFERENCE
├─ Template/generic code?
│   └─ Use FORWARDING REFERENCE (T&&) + std::forward
└─ Smart pointer?
    ├─ Transfer ownership? → Pass unique_ptr by VALUE
    ├─ Share ownership? → Pass shared_ptr by VALUE
    └─ Just observe? → Pass RAW POINTER or REFERENCE
```

### Smart Pointer Passing
```
unique_ptr:
├─ Transfer ownership? → Pass by value (std::move)
└─ Just observe? → Pass raw pointer (ptr.get()) or reference (*ptr)

shared_ptr:
├─ Share ownership? → Pass by value
├─ Just observe? → Pass raw pointer (ptr.get()) or reference (*ptr)
└─ Check lifetime? → Pass weak_ptr

DON'T pass shared_ptr by const reference just to observe!
```

## Compilation and Running

```bash
# Using the Makefile
make all              # Compile all examples
make run_basic        # Run basic examples
make run_advanced     # Run modern C++ examples
make run_all          # Compile and run everything
make clean            # Remove compiled binaries

# Or compile individually
g++ -std=c++17 -Wall basic_examples.cpp -o basic_examples
./basic_examples
```

## Key Takeaways

### General Rules
1. **Default for primitives**: Pass by value
2. **Default for complex types**: Pass by const reference
3. **When modifying**: Pass by reference
4. **Modern C++ prefers**: const reference over value for non-primitives
5. **Performance matters**: Large objects should use references
6. **Safety matters**: Use const when you don't need to modify

### Modern C++ (C++11 and later)
1. **Sink functions**: Pass by value and move internally
2. **Move semantics**: Enable efficient transfer of resources
3. **Perfect forwarding**: Use T&& for templates
4. **Return by value**: RVO/NRVO makes it efficient
5. **Smart pointers**: Pass by value for ownership, raw pointer for observation
6. **Lambda captures**: By reference for modification, by value for safety
7. **Const correctness**: Enables compiler optimizations

### Anti-patterns to Avoid
1. ❌ Returning reference to local variable
2. ❌ Using std::move on return statement (blocks RVO)
3. ❌ Passing shared_ptr by value just to observe
4. ❌ Capturing local variables by reference in returned lambdas
5. ❌ Using const_cast to modify const objects
6. ❌ Passing unique_ptr by reference to modify (return new one instead)
7. ❌ Overusing std::move (trust RVO/NRVO)

## Common Interview Questions

**Q: Why use const reference instead of value?**
A: Avoids expensive copies while preventing modifications. Best for complex types.

**Q: When is pass by value better than const reference?**
A: For small types (primitives, small POD structs) where copying is cheap, and for sink functions with move semantics.

**Q: Can you modify a const reference?**
A: No, that's the point - it provides read-only access and enables compiler optimizations.

**Q: What's the difference between reference and pointer?**
A: References cannot be null, must be initialized, and cannot be reseated. Pointers can be null and reassigned.

**Q: Does const reference work with temporaries?**
A: Yes! Const references extend the lifetime of temporaries. Non-const references cannot bind to temporaries.

```cpp
void func(const std::string& s);
func("Hello");  // OK - temporary created and bound

void func2(std::string& s);
func2("Hello");  // ERROR - cannot bind non-const ref to temporary
```

**Q: What is perfect forwarding?**
A: A template technique using forwarding references (T&&) and std::forward to preserve the value category (lvalue/rvalue) of arguments when passing them to another function.

**Q: When should I use std::move?**
A: When you want to explicitly transfer ownership and the object won't be used again. Common in move constructors, move assignment, and when passing to sink functions.

**Q: What is RVO/NRVO?**
A: Return Value Optimization / Named RVO - compiler optimizations that eliminate copies when returning objects by value. In C++17, copy elision is guaranteed in many cases.

**Q: How do I pass unique_ptr to a function?**
A: 
- By value (std::move) if transferring ownership
- By raw pointer (ptr.get()) or reference (*ptr) if just observing
- Never by const reference to transfer ownership

**Q: What's a forwarding reference vs rvalue reference?**
A: `T&&` where T is a template parameter is a forwarding reference (can bind to lvalue or rvalue). `std::string&&` where the type is concrete is an rvalue reference (only binds to rvalues).

**Q: Why is const correctness important?**
A: 
1. Prevents accidental modifications
2. Enables compiler optimizations
3. Documents intent clearly
4. Allows overloading on const
5. Enables use with const objects

## Best Practices Summary

### Function Parameters
```cpp
// Primitives - pass by value
void process(int value);

// Read-only complex types - pass by const reference  
void process(const std::string& str);
void process(const std::vector<int>& vec);

// Modify original - pass by reference
void modify(std::string& str);
void modify(std::vector<int>& vec);

// Sink (take ownership) - pass by value
void store(std::string str);  // Caller can move
void store(std::unique_ptr<Data> ptr);

// Templates - use forwarding reference
template<typename T>
void forward(T&& arg);

// Smart pointers - usually raw pointer/reference
void observe(const Data* ptr);  // Not shared_ptr!
void observe(const Data& ref);
```

### Return Types
```cpp
// Return by value - RVO applies
std::string create();
std::vector<int> makeVector();

// Return const reference to member
const std::string& getName() const;

// Return by reference for chaining
Builder& append(std::string str) &;
Builder append(std::string str) &&;

// Never return reference to local!
// std::string& bad() { std::string local; return local; }  // DANGER!
```

### Lambda Captures
```cpp
// Capture by value for safety
[=]() { }
[x, y]() { }

// Capture by reference for modification
[&]() { }
[&x, &y]() { }

// Init capture for move
[ptr = std::move(ptr)]() { }

// Capture this carefully
[this]() { }      // Pointer to this
[*this]() { }     // C++17: Copy of object
```

## Additional Resources

- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
  - F.15-F.17: Function parameter passing
  - F.42-F.45: Return values
  - ES.20: Always initialize an object
- [Effective Modern C++](https://www.oreilly.com/library/view/effective-modern-c/9781491908419/) by Scott Meyers
- [C++ Move Semantics - The Complete Guide](https://www.cppmove.com/) by Nicolai Josuttis
- [CppCon Talks](https://www.youtube.com/user/CppCon) - Search for "move semantics", "perfect forwarding"

## License

These examples are provided for educational purposes.
