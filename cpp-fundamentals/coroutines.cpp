/*
 * C++20 Coroutines
 * 
 * Key Concepts:
 * - co_await, co_yield, co_return keywords
 * - Promise types and coroutine handles
 * - Generators and async operations
 * - Coroutine state management
 * 
 * Interview Topics:
 * - What are coroutines and how do they differ from regular functions?
 * - What is the promise type and how does it control coroutine behavior?
 * - How do coroutines help with asynchronous programming?
 * 
 * Note: Coroutines are a low-level facility. Most users will use
 * libraries built on top of coroutines rather than implementing
 * promise types themselves.
 */

#include <iostream>
#include <coroutine>
#include <exception>
#include <memory>
#include <vector>

// Example 1: Simple generator using coroutines
template<typename T>
struct Generator {
    struct promise_type {
        T current_value;
        std::exception_ptr exception;
        
        Generator get_return_object() {
            return Generator{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        
        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        
        std::suspend_always yield_value(T value) {
            current_value = value;
            return {};
        }
        
        void return_void() {}
        
        void unhandled_exception() {
            exception = std::current_exception();
        }
    };
    
    std::coroutine_handle<promise_type> handle;
    
    explicit Generator(std::coroutine_handle<promise_type> h) : handle(h) {}
    
    ~Generator() {
        if (handle) {
            handle.destroy();
        }
    }
    
    // Make it move-only
    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;
    
    Generator(Generator&& other) noexcept : handle(other.handle) {
        other.handle = nullptr;
    }
    
    Generator& operator=(Generator&& other) noexcept {
        if (this != &other) {
            if (handle) {
                handle.destroy();
            }
            handle = other.handle;
            other.handle = nullptr;
        }
        return *this;
    }
    
    bool next() {
        handle.resume();
        return !handle.done();
    }
    
    T value() const {
        return handle.promise().current_value;
    }
};

// Simple number generator
Generator<int> generateNumbers(int start, int end) {
    for (int i = start; i < end; ++i) {
        co_yield i;
    }
}

void demonstrateBasicGenerator() {
    std::cout << "=== Basic Generator ===\n";
    
    auto gen = generateNumbers(1, 6);
    
    std::cout << "Generated numbers: ";
    while (gen.next()) {
        std::cout << gen.value() << " ";
    }
    std::cout << "\n";
}

// Example 2: Fibonacci generator
Generator<long long> fibonacci(int count) {
    long long a = 0, b = 1;
    
    for (int i = 0; i < count; ++i) {
        co_yield a;
        auto next = a + b;
        a = b;
        b = next;
    }
}

void demonstrateFibonacci() {
    std::cout << "\n=== Fibonacci Generator ===\n";
    
    auto fib = fibonacci(10);
    
    std::cout << "First 10 Fibonacci numbers: ";
    while (fib.next()) {
        std::cout << fib.value() << " ";
    }
    std::cout << "\n";
}

// Example 3: Task-like coroutine for async operations
template<typename T>
struct Task {
    struct promise_type {
        T value;
        std::exception_ptr exception;
        
        Task get_return_object() {
            return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        
        std::suspend_never initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        
        void return_value(T v) {
            value = v;
        }
        
        void unhandled_exception() {
            exception = std::current_exception();
        }
    };
    
    std::coroutine_handle<promise_type> handle;
    
    explicit Task(std::coroutine_handle<promise_type> h) : handle(h) {}
    
    ~Task() {
        if (handle) {
            handle.destroy();
        }
    }
    
    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;
    
    Task(Task&& other) noexcept : handle(other.handle) {
        other.handle = nullptr;
    }
    
    Task& operator=(Task&& other) noexcept {
        if (this != &other) {
            if (handle) {
                handle.destroy();
            }
            handle = other.handle;
            other.handle = nullptr;
        }
        return *this;
    }
    
    T get() {
        if (!handle.done()) {
            handle.resume();
        }
        return handle.promise().value;
    }
};

// Simple async computation
Task<int> asyncCompute(int x, int y) {
    std::cout << "Computing " << x << " + " << y << "\n";
    co_return x + y;
}

void demonstrateTask() {
    std::cout << "\n=== Task Coroutine ===\n";
    
    auto task = asyncCompute(5, 10);
    std::cout << "Result: " << task.get() << "\n";
}

// Example 4: Generator with filtering
Generator<int> generateEvens(int start, int end) {
    for (int i = start; i < end; ++i) {
        if (i % 2 == 0) {
            co_yield i;
        }
    }
}

void demonstrateFilteredGenerator() {
    std::cout << "\n=== Filtered Generator (Evens) ===\n";
    
    auto gen = generateEvens(1, 20);
    
    std::cout << "Even numbers 1-20: ";
    while (gen.next()) {
        std::cout << gen.value() << " ";
    }
    std::cout << "\n";
}

// Example 5: Generator that transforms values
Generator<int> generateSquares(int count) {
    for (int i = 1; i <= count; ++i) {
        co_yield i * i;
    }
}

void demonstrateTransformGenerator() {
    std::cout << "\n=== Transform Generator (Squares) ===\n";
    
    auto gen = generateSquares(10);
    
    std::cout << "First 10 squares: ";
    while (gen.next()) {
        std::cout << gen.value() << " ";
    }
    std::cout << "\n";
}

// Example 6: Lazy evaluation demonstration
Generator<int> lazyRange(int start, int end, int step = 1) {
    std::cout << "Starting lazy range generation\n";
    for (int i = start; i < end; i += step) {
        std::cout << "Yielding " << i << "\n";
        co_yield i;
    }
    std::cout << "Lazy range completed\n";
}

void demonstrateLazyEvaluation() {
    std::cout << "\n=== Lazy Evaluation ===\n";
    
    std::cout << "Creating generator (no computation yet)...\n";
    auto gen = lazyRange(0, 10, 2);
    
    std::cout << "\nNow requesting values:\n";
    int count = 0;
    while (gen.next() && count < 3) {
        std::cout << "Got value: " << gen.value() << "\n";
        count++;
    }
    std::cout << "\nStopped early - remaining values never computed!\n";
}

// Example 7: String generator
Generator<std::string> generateWords() {
    co_yield "Hello";
    co_yield "World";
    co_yield "from";
    co_yield "Coroutines";
}

void demonstrateStringGenerator() {
    std::cout << "\n=== String Generator ===\n";
    
    auto gen = generateWords();
    
    std::cout << "Words: ";
    while (gen.next()) {
        std::cout << gen.value() << " ";
    }
    std::cout << "\n";
}

// Example 8: Practical example - Tree traversal
struct TreeNode {
    int value;
    TreeNode* left;
    TreeNode* right;
    
    TreeNode(int v) : value(v), left(nullptr), right(nullptr) {}
};

Generator<int> inorderTraversal(TreeNode* root) {
    if (root == nullptr) {
        co_return;
    }
    
    // This is simplified - a real implementation would need
    // to manually manage the traversal stack
    if (root->left) {
        auto leftGen = inorderTraversal(root->left);
        while (leftGen.next()) {
            co_yield leftGen.value();
        }
    }
    
    co_yield root->value;
    
    if (root->right) {
        auto rightGen = inorderTraversal(root->right);
        while (rightGen.next()) {
            co_yield rightGen.value();
        }
    }
}

void demonstrateTreeTraversal() {
    std::cout << "\n=== Tree Traversal with Coroutines ===\n";
    
    // Create a simple tree:
    //       4
    //      / \
    //     2   6
    //    / \ / \
    //   1  3 5  7
    
    TreeNode* root = new TreeNode(4);
    root->left = new TreeNode(2);
    root->right = new TreeNode(6);
    root->left->left = new TreeNode(1);
    root->left->right = new TreeNode(3);
    root->right->left = new TreeNode(5);
    root->right->right = new TreeNode(7);
    
    auto gen = inorderTraversal(root);
    
    std::cout << "Inorder traversal: ";
    while (gen.next()) {
        std::cout << gen.value() << " ";
    }
    std::cout << "\n";
    
    // Cleanup
    delete root->right->right;
    delete root->right->left;
    delete root->left->right;
    delete root->left->left;
    delete root->right;
    delete root->left;
    delete root;
}

// Example 9: Demonstrating coroutine state
Generator<int> countWithState() {
    std::cout << "Coroutine started\n";
    
    int counter = 0;
    
    while (counter < 5) {
        std::cout << "About to yield " << counter << "\n";
        co_yield counter;
        std::cout << "Resumed after yielding " << counter << "\n";
        counter++;
    }
    
    std::cout << "Coroutine ending\n";
}

void demonstrateCoroutineState() {
    std::cout << "\n=== Coroutine State Management ===\n";
    
    auto gen = countWithState();
    
    std::cout << "\n--- Requesting first value ---\n";
    gen.next();
    std::cout << "Value: " << gen.value() << "\n";
    
    std::cout << "\n--- Requesting second value ---\n";
    gen.next();
    std::cout << "Value: " << gen.value() << "\n";
    
    std::cout << "\n--- Getting remaining values ---\n";
    while (gen.next()) {
        std::cout << "Value: " << gen.value() << "\n";
    }
}

int main() {
    demonstrateBasicGenerator();
    demonstrateFibonacci();
    demonstrateTask();
    demonstrateFilteredGenerator();
    demonstrateTransformGenerator();
    demonstrateLazyEvaluation();
    demonstrateStringGenerator();
    demonstrateTreeTraversal();
    demonstrateCoroutineState();
    
    std::cout << "\n=== Summary ===\n";
    std::cout << "Coroutines enable:\n";
    std::cout << "1. Lazy evaluation of sequences\n";
    std::cout << "2. Simplified async code\n";
    std::cout << "3. State preservation across suspensions\n";
    std::cout << "4. More readable generators and iterators\n";
    
    return 0;
}

/*
 * Key Takeaways:
 * 
 * 1. Coroutines are functions that can suspend and resume execution
 *    - Use co_await, co_yield, or co_return keywords
 *    - State is preserved across suspensions
 * 
 * 2. Three coroutine keywords:
 *    - co_yield: suspend and return a value (generators)
 *    - co_await: suspend and wait for result (async)
 *    - co_return: complete the coroutine with a value
 * 
 * 3. Promise type controls coroutine behavior:
 *    - get_return_object(): creates the return object
 *    - initial_suspend(): suspend before first statement?
 *    - final_suspend(): suspend after last statement?
 *    - yield_value(): handle co_yield
 *    - return_value()/return_void(): handle co_return
 * 
 * 4. Coroutine handle manages coroutine state:
 *    - resume(): continue execution
 *    - done(): check if completed
 *    - destroy(): cleanup coroutine frame
 *    - promise(): access promise object
 * 
 * 5. Common patterns:
 *    - Generator: lazy sequence generation with co_yield
 *    - Task: async computation with co_return
 *    - Awaitable: async operations with co_await
 * 
 * 6. Benefits:
 *    - Lazy evaluation - values computed only when needed
 *    - Memory efficient - don't need to store entire sequence
 *    - State preservation - local variables survive suspension
 *    - Cleaner async code - looks synchronous
 * 
 * 7. Coroutine frame:
 *    - Heap-allocated by default
 *    - Contains promise, parameters, local variables, suspension point
 *    - Destroyed when coroutine handle destroyed
 * 
 * Interview Questions:
 * 
 * Q: How do coroutines differ from regular functions?
 * A: Regular functions run to completion. Coroutines can suspend execution
 *    (co_yield, co_await), preserve state, and resume later. State persists
 *    across suspensions in a coroutine frame.
 * 
 * Q: What is the promise type in coroutines?
 * A: The promise type customizes coroutine behavior. It defines how the
 *    coroutine starts/ends, what return object is created, and how values
 *    are yielded/returned. Required for any coroutine.
 * 
 * Q: What's the difference between co_yield and co_return?
 * A: co_yield suspends and returns control to caller (can resume later).
 *    co_return completes the coroutine and returns final value. Generators
 *    use co_yield, async tasks use co_return.
 * 
 * Q: Are coroutines expensive?
 * A: Coroutine frame is heap-allocated by default, but compilers can optimize
 *    to stack allocation (HALO). Once created, suspending/resuming is cheap -
 *    just jumping to a saved point. More efficient than threads.
 * 
 * Q: When should you use coroutines?
 * A: Generators (lazy sequences), async I/O, state machines, parsers,
 *    iterators over complex structures. Generally when you need to maintain
 *    state across multiple calls without manual state management.
 * 
 * Compiler requirements:
 * - Requires C++20 support
 * - Compile with: g++ -std=c++20 -fcoroutines coroutines.cpp
 *   or: clang++ -std=c++20 -stdlib=libc++ coroutines.cpp
 */
