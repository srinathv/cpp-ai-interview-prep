#include <iostream>
#include <fstream>
#include <mutex>
#include <vector>
#include <cstring>

// RAII - Resource Acquisition Is Initialization
// Resources are acquired in constructor and released in destructor

// Example 1: File handler with RAII
class FileHandler {
private:
    std::FILE* file_;
    
public:
    FileHandler(const char* filename, const char* mode) {
        file_ = std::fopen(filename, mode);
        if (!file_) {
            throw std::runtime_error("Failed to open file");
        }
        std::cout << "File opened: " << filename << std::endl;
    }
    
    ~FileHandler() {
        if (file_) {
            std::fclose(file_);
            std::cout << "File closed" << std::endl;
        }
    }
    
    // Delete copy operations
    FileHandler(const FileHandler&) = delete;
    FileHandler& operator=(const FileHandler&) = delete;
    
    // Allow move operations
    FileHandler(FileHandler&& other) noexcept : file_(other.file_) {
        other.file_ = nullptr;
    }
    
    void write(const char* data) {
        if (file_) {
            std::fputs(data, file_);
        }
    }
};

// Example 2: Memory buffer with RAII
class Buffer {
private:
    char* data_;
    size_t size_;
    
public:
    explicit Buffer(size_t size) : size_(size) {
        data_ = new char[size];
        std::cout << "Buffer allocated: " << size << " bytes" << std::endl;
    }
    
    ~Buffer() {
        delete[] data_;
        std::cout << "Buffer deallocated" << std::endl;
    }
    
    // Rule of 5
    Buffer(const Buffer& other) : size_(other.size_) {
        data_ = new char[size_];
        std::memcpy(data_, other.data_, size_);
    }
    
    Buffer& operator=(const Buffer& other) {
        if (this != &other) {
            delete[] data_;
            size_ = other.size_;
            data_ = new char[size_];
            std::memcpy(data_, other.data_, size_);
        }
        return *this;
    }
    
    Buffer(Buffer&& other) noexcept : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    char* data() { return data_; }
    size_t size() const { return size_; }
};

// Example 3: Lock guard (similar to std::lock_guard)
class ScopedLock {
private:
    std::mutex& mutex_;
    
public:
    explicit ScopedLock(std::mutex& m) : mutex_(m) {
        mutex_.lock();
        std::cout << "Lock acquired" << std::endl;
    }
    
    ~ScopedLock() {
        mutex_.unlock();
        std::cout << "Lock released" << std::endl;
    }
    
    ScopedLock(const ScopedLock&) = delete;
    ScopedLock& operator=(const ScopedLock&) = delete;
};

// Example 4: Timer for measuring execution time
class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    std::string name_;
    
public:
    explicit Timer(const std::string& name) : name_(name) {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
        std::cout << name_ << " took " << duration.count() << " ms" << std::endl;
    }
};

void demonstrateRAII() {
    std::cout << "=== RAII Examples ===" << std::endl;
    
    // File handler - automatically closed even if exception occurs
    try {
        FileHandler file("/tmp/test.txt", "w");
        file.write("Hello RAII\n");
        // File automatically closed when going out of scope
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
    
    // Buffer - automatically deallocated
    {
        Buffer buf(1024);
        // Buffer automatically freed when going out of scope
    }
    
    // Lock guard
    std::mutex mtx;
    {
        ScopedLock lock(mtx);
        // Critical section
        // Lock automatically released when going out of scope
    }
    
    // Timer
    {
        Timer timer("Operation");
        // Some operation
        std::vector<int> v(1000000);
    }
}

int main() {
    demonstrateRAII();
    std::cout << "\n=== Program End ===" << std::endl;
    return 0;
}
