# Contributing to cpp-ai-interview-prep

This is a living, growing repository. Contributions are welcome!

## How to Contribute

### Adding New Examples

1. Choose the appropriate directory (01_cpp_fundamentals, 02_data_structures, etc.)
2. Create a well-commented .cpp file
3. Include:
   - Clear explanation at the top
   - Multiple examples demonstrating the concept
   - Common interview questions
   - Expected output
4. Update the directory's README.md

### Example Template

```cpp
/**
 * Topic Name - Brief Description
 * 
 * Key interview points:
 * - Point 1
 * - Point 2
 * - Point 3
 */

#include <iostream>

// Example 1: Basic usage
void example1() {
    std::cout << "=== Example 1 ===\n";
    // Your code here
}

int main() {
    example1();
    return 0;
}
```

### Adding New Topics

If you want to add a new major topic:
1. Create a new directory with naming convention: `##_topic_name`
2. Add a README.md explaining the topic
3. Update main README.md

### Code Style

- Use modern C++ (C++17/20)
- Clear variable names
- Comments explaining "why" not just "what"
- Include complexity analysis where relevant

## Topics Needed

Help wanted for:
- [ ] LeetCode-style algorithm problems
- [ ] HIP/ROCm GPU examples
- [ ] AI/ML numerical computing patterns
- [ ] Performance optimization examples
- [ ] Real interview questions you've encountered

## Questions?

Open an issue or submit a PR!
