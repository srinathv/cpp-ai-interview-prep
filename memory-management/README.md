# Memory Management

Modern C++ memory management techniques and best practices.

## Topics Covered

### 1. Smart Pointers
- **smart_pointers.cpp** - `unique_ptr`, `shared_ptr`, `weak_ptr`
- RAII principles
- Avoiding memory leaks and dangling pointers
- Custom deleters

### 2. RAII (Resource Acquisition Is Initialization)
- **raii.cpp** - Automatic resource management
- File handlers, locks, timers
- Rule of 5 (destructor, copy/move constructors, copy/move assignment)
- Exception safety

### 3. Move Semantics
- **move_semantics.cpp** - Efficient resource transfer
- Move constructors and assignment operators
- `std::move` and perfect forwarding
- RVO (Return Value Optimization)

## Smart Pointer Guidelines

| Pointer Type | Use Case | Ownership |
|--------------|----------|-----------|
| `unique_ptr` | Exclusive ownership | Single owner |
| `shared_ptr` | Shared ownership | Multiple owners |
| `weak_ptr` | Non-owning reference | Break cycles |
| Raw pointer | Non-owning view | Observer only |

## Best Practices

1. **Prefer `unique_ptr` by default** - Cheapest and safest
2. **Use `make_shared`/`make_unique`** - Exception safe, single allocation
3. **Avoid raw `new`/`delete`** - Use smart pointers instead
4. **Follow Rule of 5** - If you define one, define all five
5. **Use move semantics for large objects** - Avoid expensive copies

## Common Pitfalls

- Creating `shared_ptr` from raw pointer twice (double delete)
- Circular references with `shared_ptr` (use `weak_ptr`)
- Forgetting to `std::move` when transferring ownership
- Returning references to local variables

## Interview Questions

- Explain RAII and its benefits
- Difference between `unique_ptr` and `shared_ptr`
- What is move semantics and why is it important?
- How do you prevent memory leaks in C++?
- When would you use `weak_ptr`?
