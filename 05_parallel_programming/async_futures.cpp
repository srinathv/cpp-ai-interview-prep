#include <iostream>
#include <future>
#include <thread>
#include <vector>
#include <chrono>
#include <numeric>

// Function to run asynchronously
int computeSquare(int x) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    return x * x;
}

// Async with std::launch::async
void demonstrateAsync() {
    std::cout << "\n=== std::async ===" << std::endl;
    
    // Launch async task
    std::future<int> result = std::async(std::launch::async, computeSquare, 10);
    
    std::cout << "Doing other work while async task runs..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Get result (blocks if not ready)
    std::cout << "Result: " << result.get() << std::endl;
}

// Multiple async tasks
void demonstrateMultipleAsync() {
    std::cout << "\n=== Multiple Async Tasks ===" << std::endl;
    
    std::vector<std::future<int>> futures;
    
    for (int i = 1; i <= 5; ++i) {
        futures.push_back(std::async(std::launch::async, computeSquare, i));
    }
    
    std::cout << "All tasks launched, collecting results..." << std::endl;
    
    for (size_t i = 0; i < futures.size(); ++i) {
        std::cout << "Result " << i << ": " << futures[i].get() << std::endl;
    }
}

// Promise and future
void demonstratePromise() {
    std::cout << "\n=== Promise and Future ===" << std::endl;
    
    std::promise<int> promise;
    std::future<int> future = promise.get_future();
    
    // Thread that sets the promise
    std::thread t([&promise]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        promise.set_value(42);
    });
    
    std::cout << "Waiting for promise..." << std::endl;
    std::cout << "Got value: " << future.get() << std::endl;
    
    t.join();
}

// Shared future
void demonstrateSharedFuture() {
    std::cout << "\n=== Shared Future ===" << std::endl;
    
    std::promise<int> promise;
    std::shared_future<int> shared_future = promise.get_future().share();
    
    // Multiple threads can wait on shared_future
    auto waiter = [](std::shared_future<int> f, int id) {
        std::cout << "Thread " << id << " waiting..." << std::endl;
        int value = f.get();
        std::cout << "Thread " << id << " got: " << value << std::endl;
    };
    
    std::thread t1(waiter, shared_future, 1);
    std::thread t2(waiter, shared_future, 2);
    std::thread t3(waiter, shared_future, 3);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    promise.set_value(100);
    
    t1.join();
    t2.join();
    t3.join();
}

// Parallel accumulate using async
template<typename Iterator>
long long parallelAccumulate(Iterator first, Iterator last) {
    long long length = std::distance(first, last);
    
    if (length < 1000) {
        return std::accumulate(first, last, 0LL);
    }
    
    Iterator mid = first;
    std::advance(mid, length / 2);
    
    auto handle = std::async(std::launch::async, parallelAccumulate<Iterator>, first, mid);
    long long second_half = parallelAccumulate(mid, last);
    
    return handle.get() + second_half;
}

// Exception handling with futures
void demonstrateExceptions() {
    std::cout << "\n=== Exceptions with Futures ===" << std::endl;
    
    auto task = std::async(std::launch::async, []() -> int {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        throw std::runtime_error("Task failed!");
        return 42;
    });
    
    try {
        int result = task.get();
        std::cout << "Result: " << result << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "=== Async and Futures ===" << std::endl;
    
    demonstrateAsync();
    demonstrateMultipleAsync();
    demonstratePromise();
    demonstrateSharedFuture();
    demonstrateExceptions();
    
    // Parallel accumulate
    std::cout << "\n=== Parallel Accumulate ===" << std::endl;
    std::vector<int> data(10000000);
    std::iota(data.begin(), data.end(), 1);
    
    auto start = std::chrono::high_resolution_clock::now();
    long long sum = parallelAccumulate(data.begin(), data.end());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Sum: " << sum << std::endl;
    std::cout << "Time: " << duration.count() << " ms" << std::endl;
    
    return 0;
}
