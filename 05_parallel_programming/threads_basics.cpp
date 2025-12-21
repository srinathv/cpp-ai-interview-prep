#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <chrono>

// Basic thread creation
void printMessage(const std::string& msg, int count) {
    for (int i = 0; i < count; ++i) {
        std::cout << msg << " " << i << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// Thread with mutex for synchronization
std::mutex mtx;
int shared_counter = 0;

void incrementCounter(int iterations) {
    for (int i = 0; i < iterations; ++i) {
        std::lock_guard<std::mutex> lock(mtx); // RAII lock
        ++shared_counter;
    }
}

// Without mutex (race condition)
int unsafe_counter = 0;

void unsafeIncrement(int iterations) {
    for (int i = 0; i < iterations; ++i) {
        ++unsafe_counter; // RACE CONDITION!
    }
}

// Thread with return value using std::ref
void computeSum(const std::vector<int>& data, int& result) {
    result = 0;
    for (int val : data) {
        result += val;
    }
}

// Demonstrating thread join and detach
void demonstrateJoinDetach() {
    std::cout << "\n=== Join vs Detach ===" << std::endl;
    
    std::thread t1([]() {
        std::cout << "Thread 1 running" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        std::cout << "Thread 1 done" << std::endl;
    });
    
    t1.join(); // Wait for thread to finish
    std::cout << "Main: Thread 1 joined" << std::endl;
    
    std::thread t2([]() {
        std::cout << "Thread 2 running in background" << std::endl;
    });
    
    t2.detach(); // Thread runs independently
    std::cout << "Main: Thread 2 detached" << std::endl;
}

int main() {
    std::cout << "=== Basic Thread Creation ===" << std::endl;
    
    // Create threads
    std::thread t1(printMessage, "Thread-1", 3);
    std::thread t2(printMessage, "Thread-2", 3);
    
    // Wait for completion
    t1.join();
    t2.join();
    
    // Thread-safe counter
    std::cout << "\n=== Thread-Safe Counter ===" << std::endl;
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back(incrementCounter, 1000);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    std::cout << "Safe counter: " << shared_counter << std::endl;
    
    // Unsafe counter (race condition)
    std::cout << "\n=== Unsafe Counter (Race Condition) ===" << std::endl;
    std::vector<std::thread> unsafe_threads;
    for (int i = 0; i < 10; ++i) {
        unsafe_threads.emplace_back(unsafeIncrement, 1000);
    }
    
    for (auto& t : unsafe_threads) {
        t.join();
    }
    std::cout << "Unsafe counter: " << unsafe_counter << " (should be 10000)" << std::endl;
    
    // Thread with return value
    std::cout << "\n=== Thread with Return Value ===" << std::endl;
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int result = 0;
    std::thread t3(computeSum, std::cref(data), std::ref(result));
    t3.join();
    std::cout << "Sum: " << result << std::endl;
    
    demonstrateJoinDetach();
    
    std::cout << "\nHardware concurrency: " << std::thread::hardware_concurrency() << std::endl;
    
    return 0;
}
