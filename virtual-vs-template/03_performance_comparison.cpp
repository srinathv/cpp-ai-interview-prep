/**
 * Performance Comparison: Virtual vs Template
 * 
 * This benchmark demonstrates the concrete performance differences between
 * virtual functions (runtime polymorphism) and templates (compile-time polymorphism).
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <memory>
#include <random>

// ============================================================================
// VIRTUAL FUNCTION APPROACH
// ============================================================================

class VirtualAnimal {
public:
    virtual ~VirtualAnimal() = default;
    virtual int compute(int x) const = 0;
};

class VirtualDog : public VirtualAnimal {
public:
    int compute(int x) const override {
        return x * 2 + 1;
    }
};

class VirtualCat : public VirtualAnimal {
public:
    int compute(int x) const override {
        return x * 3 - 2;
    }
};

// ============================================================================
// TEMPLATE APPROACH
// ============================================================================

class TemplateDog {
public:
    int compute(int x) const {
        return x * 2 + 1;
    }
};

class TemplateCat {
public:
    int compute(int x) const {
        return x * 3 - 2;
    }
};

template<typename T>
int templateCompute(const T& animal, int x) {
    return animal.compute(x);
}

// ============================================================================
// BENCHMARKS
// ============================================================================

void benchmarkVirtual() {
    const int iterations = 100'000'000;
    
    VirtualDog dog;
    VirtualAnimal* ptr = &dog;
    
    long long sum = 0;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        sum += ptr->compute(i);  // Virtual call - vtable lookup
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Virtual function:\n";
    std::cout << "  Time: " << duration.count() << " ms\n";
    std::cout << "  Sum: " << sum << " (prevents optimization)\n";
    std::cout << "  Per call: " << (duration.count() * 1000000.0) / iterations << " ns\n";
}

void benchmarkTemplate() {
    const int iterations = 100'000'000;
    
    TemplateDog dog;
    
    long long sum = 0;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        sum += templateCompute(dog, i);  // Direct call - can inline
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Template function:\n";
    std::cout << "  Time: " << duration.count() << " ms\n";
    std::cout << "  Sum: " << sum << " (prevents optimization)\n";
    std::cout << "  Per call: " << (duration.count() * 1000000.0) / iterations << " ns\n";
}

// ============================================================================
// POLYMORPHIC COLLECTION BENCHMARK
// ============================================================================

void benchmarkVirtualCollection() {
    const int size = 1'000'000;
    const int iterations = 100;
    
    std::vector<std::unique_ptr<VirtualAnimal>> animals;
    
    // Mix of dogs and cats
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(0, 1);
    
    for (int i = 0; i < size; i++) {
        if (dis(gen) == 0) {
            animals.push_back(std::make_unique<VirtualDog>());
        } else {
            animals.push_back(std::make_unique<VirtualCat>());
        }
    }
    
    long long sum = 0;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; iter++) {
        for (const auto& animal : animals) {
            sum += animal->compute(iter);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Virtual collection (" << size << " objects, " << iterations << " iterations):\n";
    std::cout << "  Time: " << duration.count() << " ms\n";
    std::cout << "  Sum: " << sum << "\n";
}

void benchmarkTemplateCollection() {
    const int size = 500'000;  // Split into two vectors
    const int iterations = 100;
    
    std::vector<TemplateDog> dogs;
    std::vector<TemplateCat> cats;
    
    // Must use separate containers for different types
    dogs.resize(size);
    cats.resize(size);
    
    long long sum = 0;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; iter++) {
        for (const auto& dog : dogs) {
            sum += dog.compute(iter);
        }
        for (const auto& cat : cats) {
            sum += cat.compute(iter);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Template collections (2x " << size << " objects, " << iterations << " iterations):\n";
    std::cout << "  Time: " << duration.count() << " ms\n";
    std::cout << "  Sum: " << sum << "\n";
    std::cout << "  Note: Better cache locality (same-type objects together)\n";
}

// ============================================================================
// MEMORY OVERHEAD COMPARISON
// ============================================================================

void compareMemoryOverhead() {
    std::cout << "\n=== Memory Overhead ===\n\n";
    
    VirtualDog vdog;
    TemplateDog tdog;
    
    std::cout << "Object sizes:\n";
    std::cout << "  VirtualDog: " << sizeof(vdog) << " bytes (includes vptr)\n";
    std::cout << "  TemplateDog: " << sizeof(tdog) << " bytes (no vptr)\n";
    std::cout << "  vptr overhead: " << (sizeof(vdog) - sizeof(tdog)) << " bytes per object\n";
    
    std::cout << "\nFor 1,000,000 objects:\n";
    std::cout << "  Virtual: " << (sizeof(vdog) * 1'000'000) / 1024 << " KB\n";
    std::cout << "  Template: " << (sizeof(tdog) * 1'000'000) / 1024 << " KB\n";
    std::cout << "  Extra memory for vptrs: " 
              << ((sizeof(vdog) - sizeof(tdog)) * 1'000'000) / 1024 << " KB\n";
}

int main() {
    std::cout << "=== Virtual vs Template Performance Comparison ===\n\n";
    
    std::cout << "=== Single Call Performance ===\n\n";
    benchmarkVirtual();
    std::cout << "\n";
    benchmarkTemplate();
    
    std::cout << "\n=== Collection Performance ===\n\n";
    benchmarkVirtualCollection();
    std::cout << "\n";
    benchmarkTemplateCollection();
    
    compareMemoryOverhead();
    
    std::cout << "\n=== Key Takeaways ===\n\n";
    std::cout << "Virtual Functions:\n";
    std::cout << "  + Can store different types in same container\n";
    std::cout << "  + Runtime type selection\n";
    std::cout << "  + Smaller code size\n";
    std::cout << "  - Slower (vtable indirection)\n";
    std::cout << "  - Cannot inline\n";
    std::cout << "  - Memory overhead (vptr)\n";
    
    std::cout << "\nTemplates:\n";
    std::cout << "  + Faster (direct calls, inlining)\n";
    std::cout << "  + No memory overhead\n";
    std::cout << "  + Better cache locality (same types together)\n";
    std::cout << "  - Must know types at compile time\n";
    std::cout << "  - Separate containers for different types\n";
    std::cout << "  - Larger binary (code duplication)\n";
    
    return 0;
}
