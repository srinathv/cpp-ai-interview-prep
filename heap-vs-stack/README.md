# Heap vs Stack Examples

This directory demonstrates the differences between heap and stack memory allocation in both C++ and Rust, with detailed examples showing allocation, usage, and deallocation patterns.

## Files Overview

### Basic Examples

#### C++ Examples
- **stack_overflow.cpp** - Demonstrates stack overflow through infinite recursion
- **heap_example.cpp** - Shows safe heap allocation for large data
- **stack_vs_heap.cpp** - Compares performance and characteristics of both

#### Rust Examples
- **stack_overflow.rs** - Demonstrates stack overflow through infinite recursion
- **heap_example.rs** - Shows safe heap allocation using Box and Vec
- **stack_vs_heap.rs** - Compares performance and demonstrates ownership

### Extended Examples (NEW!)

#### Memory Lifecycle Examples
- **memory_lifecycle.cpp** - C++ memory lifecycle: allocation → use → deallocation
  - Stack variable lifecycle with automatic cleanup
  - Heap variable lifecycle with manual management
  - Use-after-free dangers (undefined behavior)
  - Dangling pointer problems
  - Stack growth visualization during recursion
  - Memory leaks demonstration
  - RAII (Resource Acquisition Is Initialization) pattern

- **memory_lifecycle.rs** - Rust memory lifecycle with safety guarantees
  - Stack variable lifecycle with automatic cleanup
  - Heap variable lifecycle with Box
  - Ownership transfer and move semantics
  - Borrowing (immutable and mutable references)
  - Lifetime tracking across scopes
  - Stack growth visualization during recursion
  - RAII with Drop trait
  - Copy vs Move semantics
  - Shared ownership with Rc (reference counting)

#### Comprehensive Usage Examples
- **comprehensive_usage.cpp** - Detailed C++ memory patterns
  - Stack allocation, use, and cleanup
  - Heap allocation with manual memory management
  - Smart pointers (unique_ptr) for automatic heap management
  - Function call stack demonstration
  - Array allocation comparison (stack vs heap vs vector)
  - Memory reuse patterns
  - Complete lifecycle tracking with memory addresses

- **comprehensive_usage.rs** - Detailed Rust memory patterns
  - Stack allocation with automatic cleanup
  - Heap allocation with Box
  - Ownership transfer mechanics
  - Borrowing rules enforcement
  - Function call stack demonstration
  - Collection allocation (arrays, Vec, Box)
  - Lifetime tracking across nested scopes
  - Shared ownership with Rc
  - Copy vs Move semantics comparison
  - Scope-based resource cleanup

## Key Differences

### Stack
- **Fast allocation** - Just moving a pointer
- **Automatic cleanup** - Variables freed when out of scope
- **Limited size** - Typically 1-8 MB
- **LIFO structure** - Last in, first out
- **Deterministic lifetime** - Lives until scope ends
- **Use for**: Small, short-lived variables, function parameters, local variables

### Heap
- **Slower allocation** - Memory manager finds free space
- **Manual cleanup** - C++ needs delete/smart pointers, Rust uses RAII
- **Large size** - Limited by available RAM
- **Fragmentation possible** - Non-contiguous allocation
- **Flexible lifetime** - Can outlive creating function
- **Use for**: Large data, dynamic sizes, data that outlives function scope

### C++ vs Rust Memory Safety

**C++ Challenges:**
- Use-after-free bugs (accessing freed memory)
- Double-free bugs (freeing same memory twice)
- Memory leaks (forgetting to free)
- Dangling pointers (pointers to freed memory)
- Data races in concurrent code

**Rust Guarantees:**
- No use-after-free (compile-time prevention)
- No double-free (ownership system)
- No memory leaks (RAII + Drop trait)
- No dangling references (lifetime checking)
- No data races (borrow checker)

## Building and Running

### Quick Start
```bash
# Build everything
make all

# Run basic examples
make run-cpp
make run-rust

# Run memory lifecycle examples
make run-lifecycle

# Run comprehensive usage examples
make run-comprehensive

# See all options
make help
```

### C++ Detailed Build
```bash
# Compile individual files
g++ -std=c++17 -Wall -Wextra -o memory_lifecycle memory_lifecycle.cpp
g++ -std=c++17 -Wall -Wextra -o comprehensive_usage comprehensive_usage.cpp

# Run
./memory_lifecycle
./comprehensive_usage
```

### Rust Detailed Build
```bash
# Compile and run
rustc memory_lifecycle.rs && ./memory_lifecycle_rs
rustc comprehensive_usage.rs && ./comprehensive_usage_rs
```

## Example Output Highlights

### Memory Addresses
Both C++ and Rust examples print memory addresses to show:
- Stack variables have addresses that decrease with nested calls
- Heap allocations have different address patterns
- Stack memory is reused after scope ends
- Heap memory persists until explicitly freed

### Lifecycle Tracking
Examples demonstrate:
1. **Allocation** - When and where memory is allocated
2. **Usage** - How to access and modify the memory
3. **Deallocation** - When and how memory is freed

### C++ Dangers Shown
- Use-after-free undefined behavior
- Returning pointers to local variables
- Memory leaks from missing delete calls

### Rust Safety Shown
- Compile-time prevention of use-after-free
- Ownership transfer to prevent double-free
- Automatic cleanup via Drop trait
- Borrow checker preventing dangling references

## Stack Overflow Examples

Both C++ and Rust examples include functions that will blow the stack:

1. **Infinite recursion** - Each function call adds a stack frame
2. **Large stack allocation** - Allocating huge arrays on the stack
3. **Deep recursion** - Eventually exhausts stack space

**Warning**: These examples will crash your program! Uncomment the calls in main() to see them in action.

## Interview Tips

### When discussing heap vs stack:
- **Stack overflow** happens when you exceed stack memory limits
- **Common causes**: infinite/deep recursion, large local arrays
- **Solution**: Use heap allocation for large data structures
- **Performance**: Stack is ~5x faster than heap (see stack_vs_heap examples)

### Memory Management Concepts:
- **RAII** - Resource Acquisition Is Initialization (automatic cleanup)
- **Smart pointers** - C++ std::unique_ptr, std::shared_ptr
- **Ownership** - Rust's zero-cost abstraction for memory safety
- **Borrowing** - Rust's reference system with compile-time checks
- **Lifetimes** - Ensuring references don't outlive their data

### Key Differences Between C++ and Rust:
- **C++**: Manual memory management (or smart pointers), runtime errors possible
- **Rust**: Ownership system, compile-time memory safety, no runtime overhead
- **Both**: Support RAII pattern for automatic resource cleanup
- **Rust advantage**: Prevents entire classes of bugs at compile time
- **C++ advantage**: More flexibility (can be dangerous if misused)

## Learning Path

1. **Start with basic examples** - Understand stack vs heap fundamentals
2. **Explore lifecycle examples** - See allocation → use → deallocation
3. **Study comprehensive examples** - Deep dive into all patterns
4. **Compare C++ vs Rust** - Understand memory safety approaches
5. **Experiment** - Modify examples to test your understanding

## Common Interview Questions Covered

1. What's the difference between stack and heap?
2. When should I use each?
3. What causes stack overflow?
4. How does memory management differ in C++ vs Rust?
5. What is RAII and why is it important?
6. Explain ownership and borrowing in Rust
7. What are smart pointers and when should I use them?
8. How can I prevent memory leaks?
9. What is use-after-free and how to avoid it?
10. How does the function call stack work?

All these questions are answered with working code examples in this directory!
