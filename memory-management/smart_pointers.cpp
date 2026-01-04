#include <iostream>
#include <memory>
#include <vector>

class Resource {
public:
    Resource(const std::string& name) : name_(name) {
        std::cout << "Resource " << name_ << " created" << std::endl;
    }
    ~Resource() {
        std::cout << "Resource " << name_ << " destroyed" << std::endl;
    }
    void use() { std::cout << "Using " << name_ << std::endl; }
private:
    std::string name_;
};

// unique_ptr - exclusive ownership
void demonstrateUniquePtr() {
    std::cout << "\n=== unique_ptr ===" << std::endl;
    
    std::unique_ptr<Resource> ptr1 = std::make_unique<Resource>("unique1");
    ptr1->use();
    
    // Transfer ownership
    std::unique_ptr<Resource> ptr2 = std::move(ptr1);
    if (!ptr1) {
        std::cout << "ptr1 is now null" << std::endl;
    }
    ptr2->use();
    
    // Custom deleter
    auto deleter = [](Resource* r) {
        std::cout << "Custom deleter called" << std::endl;
        delete r;
    };
    std::unique_ptr<Resource, decltype(deleter)> ptr3(new Resource("custom"), deleter);
}

// shared_ptr - shared ownership
void demonstrateSharedPtr() {
    std::cout << "\n=== shared_ptr ===" << std::endl;
    
    std::shared_ptr<Resource> ptr1 = std::make_shared<Resource>("shared1");
    std::cout << "Use count: " << ptr1.use_count() << std::endl;
    
    {
        std::shared_ptr<Resource> ptr2 = ptr1;
        std::cout << "Use count after copy: " << ptr1.use_count() << std::endl;
        ptr2->use();
    }
    
    std::cout << "Use count after scope: " << ptr1.use_count() << std::endl;
}

// weak_ptr - break circular references
class Node {
public:
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> prev; // Use weak_ptr to avoid circular reference
    int value;
    
    Node(int v) : value(v) {
        std::cout << "Node " << value << " created" << std::endl;
    }
    ~Node() {
        std::cout << "Node " << value << " destroyed" << std::endl;
    }
};

void demonstrateWeakPtr() {
    std::cout << "\n=== weak_ptr ===" << std::endl;
    
    auto node1 = std::make_shared<Node>(1);
    auto node2 = std::make_shared<Node>(2);
    
    node1->next = node2;
    node2->prev = node1; // weak_ptr prevents memory leak
    
    std::cout << "Use count of node1: " << node1.use_count() << std::endl;
    
    // Convert weak_ptr to shared_ptr to access
    if (auto prevNode = node2->prev.lock()) {
        std::cout << "Previous node value: " << prevNode->value << std::endl;
    }
}

// Common pitfalls
void demonstratePitfalls() {
    std::cout << "\n=== Common Pitfalls ===" << std::endl;
    
    // DON'T create shared_ptr from raw pointer twice
    Resource* raw = new Resource("raw");
    // std::shared_ptr<Resource> ptr1(raw); // DON'T
    // std::shared_ptr<Resource> ptr2(raw); // This would cause double delete!
    delete raw; // Clean up properly
    
    // DO use make_shared instead
    auto ptr = std::make_shared<Resource>("proper");
}

// Performance comparison
void performanceComparison() {
    std::cout << "\n=== Performance ===" << std::endl;
    
    // make_shared is more efficient (single allocation)
    auto shared1 = std::make_shared<Resource>("make_shared");
    
    // shared_ptr constructor does two allocations
    std::shared_ptr<Resource> shared2(new Resource("constructor"));
}

int main() {
    demonstrateUniquePtr();
    demonstrateSharedPtr();
    demonstrateWeakPtr();
    demonstratePitfalls();
    performanceComparison();
    
    std::cout << "\n=== Program End ===" << std::endl;
    return 0;
}
