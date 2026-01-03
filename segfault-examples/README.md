# Segmentation Fault Examples

This directory contains comprehensive examples of all common segmentation faults in C++ and demonstrates how Rust prevents them through compile-time and runtime safety guarantees.

## Overview

A segmentation fault (SIGSEGV) occurs when a program tries to access memory it doesn't have permission to access. This directory demonstrates 10 major categories of segfaults and Rust's prevention mechanisms.

## Files

### Individual Category Examples

| C++ Example | Rust Safety Demo | Category |
|------------|------------------|----------|
| `01_null_pointer.cpp` | `01_null_pointer.rs` | Null pointer dereference |
| `02_use_after_free.cpp` | `02_use_after_free.rs` | Use-after-free / dangling pointers |
| `03_buffer_overflow.cpp` | `03_buffer_overflow.rs` | Buffer overflow / out-of-bounds access |

### Comprehensive Examples

- **04_all_segfaults.cpp** - All 10 segfault categories in one program
  - Run with `./04_all_segfaults <number>` to trigger specific crash
- **04_rust_safety.rs** - Shows how Rust prevents all 10 categories

## The 10 Segfault Categories

### 1. Null Pointer Dereference
```cpp
// C++ - CRASH
int* ptr = nullptr;
*ptr = 42;
```
```rust
// Rust - SAFE (no null pointers)
let maybe: Option<i32> = None;
// Must handle None explicitly
```

### 2. Use-After-Free
```cpp
// C++ - CRASH
int* ptr = new int(42);
delete ptr;
*ptr = 100;
```
```rust
// Rust - COMPILE ERROR
let data = Box::new(42);
drop(data);
// println!("{}", data); // ERROR: value used after move
```

### 3. Dangling Pointer
```cpp
// C++ - CRASH
int* getPtr() {
    int local = 42;
    return &local;  // Returns address of destroyed variable
}
```
```rust
// Rust - COMPILE ERROR
// fn bad() -> &i32 {
//     let x = 42;
//     &x  // ERROR: returns reference with insufficient lifetime
// }
```

### 4. Buffer Overflow
```cpp
// C++ - CRASH
int arr[10];
arr[100] = 42;
```
```rust
// Rust - PANIC (safe crash)
let arr = [1, 2, 3];
// arr[100] = 42; // Panics instead of corrupting memory
```

### 5. Stack Overflow
```cpp
// C++ - CRASH
void f() { f(); }  // Infinite recursion
```
```rust
// Rust - Same issue, detected by OS
fn f() { f(); }  // OS terminates safely
```

### 6. Uninitialized Pointer
```cpp
// C++ - CRASH
int* ptr;  // Garbage value
*ptr = 42;
```
```rust
// Rust - COMPILE ERROR
// let ptr: &i32; // ERROR if used uninitialized
```

### 7. Double Free
```cpp
// C++ - CRASH
int* ptr = new int(42);
delete ptr;
delete ptr;
```
```rust
// Rust - COMPILE ERROR
let data = Box::new(42);
drop(data);
// drop(data); // ERROR: use of moved value
```

### 8. Write to Read-Only Memory
```cpp
// C++ - CRASH
char* str = "hello";
str[0] = 'H';
```
```rust
// Rust - COMPILE ERROR or requires unsafe
let s = "hello";
// s[0] = 'H'; // ERROR: cannot mutate string literal
```

### 9. Virtual Function on Deleted Object
```cpp
// C++ - CRASH
Base* obj = new Base();
delete obj;
obj->virtualMethod();
```
```rust
// Rust - COMPILE ERROR
// obj.method(); // ERROR: value used after move
```

### 10. Iterator Invalidation
```cpp
// C++ - CRASH
auto it = vec.begin();
vec.push_back(x);  // Invalidates iterator
*it;  // Crash
```
```rust
// Rust - COMPILE ERROR
// for val in &vec {
//     vec.push(4); // ERROR: cannot borrow as mutable
// }
```

## Building and Running

### Quick Start
```bash
# Build everything
make all

# Run all Rust examples (safe!)
make run-rust

# Build with AddressSanitizer
make safe

# See all options
make help
```

### Running C++ Examples (DANGEROUS)
```bash
# Build C++ examples
make cpp

# These WILL CRASH:
./01_null_pointer
./02_use_after_free
./03_buffer_overflow

# Run specific crash type (1-10)
./04_all_segfaults 1  # Null pointer
./04_all_segfaults 2  # Use-after-free
# ... etc
```

### Running with AddressSanitizer (Recommended)
```bash
# Build with sanitizers
make safe

# These will detect bugs and report them clearly:
./01_null_pointer_safe
./02_use_after_free_safe
./03_buffer_overflow_safe
```

### Running Rust Examples (SAFE)
```bash
# Build Rust examples
make rust

# All safe to run:
./01_null_pointer_rs
./02_use_after_free_rs
./03_buffer_overflow_rs
./04_rust_safety_rs

# Or run all at once:
make run-rust
```

## How Rust Prevents Segfaults

### Compile-Time Prevention
- **Ownership System** - Prevents use-after-free and double-free
- **Borrow Checker** - Prevents dangling references and data races
- **No Null Pointers** - Option<T> makes absence explicit
- **Initialization Checking** - All variables must be initialized
- **Lifetime Tracking** - Ensures references don't outlive their data

### Runtime Prevention
- **Bounds Checking** - All array/vec accesses are checked
- **Safe Panics** - Out-of-bounds causes controlled panic, not corruption
- **Integer Overflow Checks** - Debug mode catches overflows

### Result
- **Most segfaults**: Prevented at **compile time**
- **Remaining issues**: Safe **runtime panics** instead of undefined behavior
- **Zero-cost abstractions**: No runtime overhead for safety

## Debugging Tools

### C++ Tools
```bash
# AddressSanitizer (detects memory errors)
g++ -fsanitize=address -g program.cpp

# Valgrind (memory checker)
valgrind --leak-check=full ./program

# For GPU code
cuda-memcheck ./cuda_program
compute-sanitizer ./cuda_program

# For ROCm
rocgdb ./rocm_program
```

### Rust Tools
```bash
# Miri (interpreter that catches undefined behavior)
cargo +nightly miri run

# RUST_BACKTRACE for panics
RUST_BACKTRACE=1 cargo run
```

## Interview Prep

### Key Questions Covered

1. **What causes segmentation faults?**
   - All 10 categories demonstrated with examples

2. **How can you prevent segfaults in C++?**
   - Smart pointers, RAII, bounds checking, sanitizers
   - Examples in safe alternatives sections

3. **How does Rust prevent segfaults?**
   - Ownership, borrowing, lifetimes, Option<T>
   - Demonstrated in all Rust examples

4. **What's the difference between compile-time and runtime safety?**
   - Rust catches most at compile time
   - Remaining issues cause safe panics

5. **When would you use Rust over C++?**
   - Memory safety critical
   - Concurrent systems
   - Systems where undefined behavior is unacceptable

6. **What tools detect memory errors?**
   - AddressSanitizer, Valgrind, cuda-memcheck
   - Demonstrated in Makefile targets

### Memory Safety Comparison

| Issue | C++ | Rust |
|-------|-----|------|
| Null pointer | Runtime crash | Compile-time prevention (Option<T>) |
| Use-after-free | Undefined behavior | Compile-time prevention (ownership) |
| Buffer overflow | Memory corruption | Runtime panic (bounds check) |
| Double free | Crash/corruption | Compile-time prevention (move) |
| Dangling reference | Undefined behavior | Compile-time prevention (lifetimes) |
| Uninitialized | Undefined behavior | Compile-time error |
| Data races | Undefined behavior | Compile-time prevention (Send/Sync) |

## Why This Matters for AI/ML Systems

In GPU computing and AI/ML workloads:

- **Memory bugs are catastrophic** - Silent corruption worse than crashes
- **Debugging is expensive** - GPU memory errors are hard to track down
- **Concurrency is everywhere** - Data races cause non-deterministic failures
- **Performance is critical** - Rust provides safety with zero overhead

Rust's guarantees make it ideal for:
- CUDA/HIP kernel host code
- ML framework backends
- High-performance data pipelines
- Concurrent training systems

## Learning Path

1. **Run Rust examples** - See how safety works
2. **Try C++ crashes** - Understand the problems
3. **Use sanitizers** - Learn detection tools
4. **Read the code** - Study prevention techniques
5. **Experiment** - Modify examples to test understanding

## Additional Resources

- [Rust Book - Ownership](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html)
- [AddressSanitizer](https://github.com/google/sanitizers/wiki/AddressSanitizer)
- [Valgrind Documentation](https://valgrind.org/docs/manual/quick-start.html)
- [CUDA Memory Checker](https://docs.nvidia.com/cuda/cuda-memcheck/)

---

**Warning**: The C++ examples will crash your program! Only run them if you want to see actual segfaults. Use the sanitizer versions or Rust examples for safe exploration.
