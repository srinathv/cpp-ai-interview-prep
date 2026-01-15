# Pass by Reference vs Pass by Value in C++

This directory contains examples demonstrating when to use pass-by-reference versus pass-by-value in C++.

## Quick Reference Guide

### Pass by VALUE
```cpp
void func(int x)
void func(Point p)
```
**Use when:**
- Small types (int, char, bool, float, double, pointers)
- Function needs its own copy to modify
- Type is cheap to copy (typically ≤ 16 bytes)

**Advantages:**
- Simple and safe (no unintended modifications)
- Good for primitive types

**Disadvantages:**
- Expensive for large objects (copies entire object)
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
- Working with large objects to avoid copies

**Advantages:**
- No copy overhead
- Can modify original data
- Efficient for large objects

**Disadvantages:**
- Caller's data can be modified (side effects)
- Requires lvalue (cannot pass temporary)

---

### Pass by CONST REFERENCE
```cpp
void func(const std::string& str)
void func(const std::vector<int>& vec)
```
**Use when:**
- Large objects that shouldn't be modified
- Reading from containers (vector, string, map, etc.)
- Most function parameters in modern C++
- Want efficiency without allowing modifications

**Advantages:**
- No copy overhead
- Cannot accidentally modify original
- Can accept temporaries and rvalues
- Best practice for most complex types

**Disadvantages:**
- Slightly more complex syntax
- Small overhead for tiny types (use value instead)

---

## Decision Tree

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

## Files in This Directory

1. **basic_examples.cpp**
   - Fundamental examples of all three passing mechanisms
   - Shows behavior differences clearly
   - Good starting point

2. **performance_comparison.cpp**
   - Demonstrates performance impact of copying large objects
   - Shows when const reference prevents expensive copies
   - Includes timing examples

3. **common_patterns.cpp**
   - Real-world patterns and use cases
   - Swap functions, output parameters
   - Range-based for loops
   - Fluent interfaces
   - Factory functions
   - Container operations

## Compilation and Running

```bash
# Using the Makefile
make all        # Compile all examples
make run_all    # Compile and run all examples
make clean      # Remove compiled binaries

# Or compile individually
g++ -std=c++17 -Wall basic_examples.cpp -o basic_examples
./basic_examples
```

## Key Takeaways

1. **Default for primitives**: Pass by value
2. **Default for complex types**: Pass by const reference
3. **When modifying**: Pass by reference
4. **Modern C++ prefers**: const reference over value for non-primitives
5. **Performance matters**: Large objects should use references
6. **Safety matters**: Use const when you don't need to modify

## Common Interview Questions

**Q: Why use const reference instead of value?**
A: Avoids expensive copies while preventing modifications.

**Q: When is pass by value better than const reference?**
A: For small types (primitives, small POD structs) where copying is cheap.

**Q: Can you modify a const reference?**
A: No, that's the point - it provides read-only access.

**Q: What's the difference between reference and pointer?**
A: References cannot be null, must be initialized, and cannot be reseated. Pointers can be null and reassigned.

**Q: Does const reference work with temporaries?**
A: Yes! Const references can bind to temporaries, non-const references cannot.

```cpp
void func(const std::string& s);
func("Hello");  // OK - temporary created and bound

void func2(std::string& s);
func2("Hello");  // ERROR - cannot bind non-const ref to temporary
```

## Best Practices

1. **Use const reference by default** for function parameters that are non-primitive types
2. **Use value for primitives** (int, char, bool, pointers)
3. **Use non-const reference only when you need to modify** the original
4. **Avoid returning references to local variables** (dangling reference!)
5. **Consider return value optimization (RVO)** - modern compilers avoid copies when returning by value

## Additional Resources

- [C++ Core Guidelines: F.15-F.17](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#S-functions)
- Effective C++ by Scott Meyers
- C++ Primer by Lippman, Lajoie, and Moo
