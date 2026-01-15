#include <iostream>
#include <memory>
#include <vector>
#include <string>

// Smart Pointer Passing - Best Practices

class Resource {
    std::string name;
public:
    explicit Resource(const std::string& n) : name(n) {
        std::cout << "Resource created: " << name << std::endl;
    }
    ~Resource() {
        std::cout << "Resource destroyed: " << name << std::endl;
    }
    const std::string& getName() const { return name; }
    void doWork() const {
        std::cout << name << " is working..." << std::endl;
    }
};

// ===== UNIQUE_PTR PATTERNS =====

// Pattern 1: Pass unique_ptr by value for ownership transfer (sink)
void takeOwnership(std::unique_ptr<Resource> resource) {
    std::cout << "Took ownership of: " << resource->getName() << std::endl;
    resource->doWork();
    // resource is destroyed here
}

// Pattern 2: Pass unique_ptr by const reference for observation (NO ownership transfer)
void observeUnique(const std::unique_ptr<Resource>& resource) {
    std::cout << "Observing: " << resource->getName() << std::endl;
    resource->doWork();
    // resource is NOT destroyed here
}

// Pattern 3: Pass raw pointer for non-owning observation (preferred for unique_ptr)
void observeRaw(const Resource* resource) {
    if (resource) {
        std::cout << "Observing via raw ptr: " << resource->getName() << std::endl;
        resource->doWork();
    }
}

// Pattern 4: Pass by reference for guaranteed non-null non-owning access
void observeRef(const Resource& resource) {
    std::cout << "Observing via reference: " << resource.getName() << std::endl;
    resource.doWork();
}

// Pattern 5: Return unique_ptr by value (factory pattern)
std::unique_ptr<Resource> createResource(const std::string& name) {
    return std::make_unique<Resource>(name);
}

// Pattern 6: unique_ptr in containers
void uniquePtrInContainers() {
    std::cout << "\n=== unique_ptr in containers ===" << std::endl;
    
    std::vector<std::unique_ptr<Resource>> resources;
    resources.push_back(std::make_unique<Resource>("R1"));
    resources.push_back(std::make_unique<Resource>("R2"));
    
    // Access without transferring ownership
    for (const auto& r : resources) {
        r->doWork();
    }
    
    // Transfer ownership out
    auto taken = std::move(resources[0]);
    std::cout << "Taken: " << taken->getName() << std::endl;
    // resources[0] is now nullptr
}

// ===== SHARED_PTR PATTERNS =====

// Pattern 7: Pass shared_ptr by value for shared ownership
void shareOwnership(std::shared_ptr<Resource> resource) {
    std::cout << "Sharing ownership (count: " << resource.use_count() << "): " 
              << resource->getName() << std::endl;
    resource->doWork();
}

// Pattern 8: Pass shared_ptr by const reference when not retaining
void observeShared(const std::shared_ptr<Resource>& resource) {
    std::cout << "Observing shared (count: " << resource.use_count() << "): " 
              << resource->getName() << std::endl;
}

// Pattern 9: Pass raw pointer/reference instead of shared_ptr for observation
// This is the PREFERRED approach - avoid passing shared_ptr unnecessarily
void observeFromShared(const Resource* resource) {
    if (resource) {
        std::cout << "Observing via raw ptr from shared_ptr: " 
                  << resource->getName() << std::endl;
    }
}

// Pattern 10: Return shared_ptr by value
std::shared_ptr<Resource> createSharedResource(const std::string& name) {
    return std::make_shared<Resource>(name);
}

// Pattern 11: weak_ptr to avoid circular references
class Node : public std::enable_shared_from_this<Node> {
    std::string data;
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> prev;  // weak_ptr to break cycle
    
public:
    explicit Node(const std::string& d) : data(d) {
        std::cout << "Node created: " << data << std::endl;
    }
    ~Node() {
        std::cout << "Node destroyed: " << data << std::endl;
    }
    
    void setNext(std::shared_ptr<Node> n) {
        next = n;
        if (n) n->prev = shared_from_this();
    }
    
    std::shared_ptr<Node> getNext() const { return next; }
    std::shared_ptr<Node> getPrev() const { return prev.lock(); }
    
    const std::string& getData() const { return data; }
};

// Pattern 12: Pass weak_ptr when you need to check if resource still exists
void tryUseWeakPtr(std::weak_ptr<Resource> weak) {
    if (auto shared = weak.lock()) {
        std::cout << "Resource still alive: " << shared->getName() << std::endl;
        shared->doWork();
    } else {
        std::cout << "Resource was destroyed" << std::endl;
    }
}

// ===== ANTIPATTERNS - WHAT NOT TO DO =====

// WRONG: Don't pass unique_ptr by reference to modify (use raw pointer or return new one)
// void badModifyUnique(std::unique_ptr<Resource>& resource) { ... }

// WRONG: Don't pass shared_ptr by reference to modify
// void badModifyShared(std::shared_ptr<Resource>& resource) { ... }

// WRONG: Don't pass shared_ptr just to observe (use raw pointer or reference)
// void badObserveShared(std::shared_ptr<Resource> resource) { ... }  // Creates unnecessary copy

// ===== COMPARISON AND GUIDELINES =====

void demonstrateGuidelines() {
    std::cout << "\n=== Smart Pointer Passing Guidelines ===" << std::endl;
    
    // unique_ptr - single ownership
    auto u1 = std::make_unique<Resource>("Unique1");
    
    std::cout << "\nObserving unique_ptr (no ownership transfer):" << std::endl;
    observeRaw(u1.get());          // PREFERRED
    observeRef(*u1);               // Also good
    observeUnique(u1);             // OK but less clear
    
    std::cout << "\nTransferring unique_ptr ownership:" << std::endl;
    takeOwnership(std::move(u1));  // u1 is now nullptr
    
    // shared_ptr - shared ownership
    auto s1 = std::make_shared<Resource>("Shared1");
    std::cout << "\nInitial shared_ptr count: " << s1.use_count() << std::endl;
    
    std::cout << "\nObserving shared_ptr (no ownership):" << std::endl;
    observeFromShared(s1.get());   // PREFERRED - no ref count change
    observeShared(s1);             // OK - no ref count change
    std::cout << "Count after observe: " << s1.use_count() << std::endl;
    
    std::cout << "\nSharing ownership:" << std::endl;
    shareOwnership(s1);            // Ref count increases temporarily
    std::cout << "Count after share: " << s1.use_count() << std::endl;
    
    // weak_ptr - non-owning reference
    std::weak_ptr<Resource> weak = s1;
    std::cout << "\nUsing weak_ptr:" << std::endl;
    tryUseWeakPtr(weak);
    
    s1.reset();  // Destroy the resource
    tryUseWeakPtr(weak);  // Should report destroyed
}

void demonstrateWeakPtr() {
    std::cout << "\n=== weak_ptr for Circular References ===" << std::endl;
    
    auto node1 = std::make_shared<Node>("Node1");
    auto node2 = std::make_shared<Node>("Node2");
    auto node3 = std::make_shared<Node>("Node3");
    
    node1->setNext(node2);
    node2->setNext(node3);
    
    std::cout << "\nTraversing forward:" << std::endl;
    auto current = node1;
    while (current) {
        std::cout << "  " << current->getData() << std::endl;
        current = current->getNext();
    }
    
    std::cout << "\nTraversing backward from node3:" << std::endl;
    current = node3;
    while (current) {
        std::cout << "  " << current->getData() << std::endl;
        current = current->getPrev();
    }
    
    std::cout << "\nNodes will be destroyed when exiting scope..." << std::endl;
}

int main() {
    std::cout << "=== SMART POINTER PASSING ===" << std::endl;
    
    uniquePtrInContainers();
    demonstrateGuidelines();
    demonstrateWeakPtr();
    
    std::cout << "\n=== Program ending ===" << std::endl;
    return 0;
}
