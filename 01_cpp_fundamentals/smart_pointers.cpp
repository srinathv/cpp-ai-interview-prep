/*
 * Smart Pointers in Modern C++
 * 
 * Key Concepts:
 * - std::unique_ptr - Exclusive ownership
 * - std::shared_ptr - Shared ownership with reference counting
 * - std::weak_ptr - Non-owning observer to break circular references
 * - Custom deleters
 * 
 * Interview Topics:
 * - When to use each type of smart pointer?
 * - How does reference counting work in shared_ptr?
 * - How to avoid memory leaks with circular references?
 */

#include <iostream>
#include <memory>
#include <vector>
#include <string>

// Example 1: std::unique_ptr basics
class Resource {
private:
    std::string name;
    int* data;

public:
    Resource(const std::string& n) : name(n), data(new int[100]) {
        std::cout << "Resource '" << name << "' created\n";
    }

    ~Resource() {
        delete[] data;
        std::cout << "Resource '" << name << "' destroyed\n";
    }

    void use() const {
        std::cout << "Using resource '" << name << "'\n";
    }

    const std::string& getName() const { return name; }
};

void demonstrateUniquePtr() {
    std::cout << "=== std::unique_ptr Demo ===\n";
    
    // Create unique_ptr
    std::unique_ptr<Resource> ptr1 = std::make_unique<Resource>("ptr1");
    ptr1->use();
    
    // unique_ptr cannot be copied
    // std::unique_ptr<Resource> ptr2 = ptr1; // Error!
    
    // But can be moved
    std::unique_ptr<Resource> ptr2 = std::move(ptr1);
    // ptr1 is now nullptr
    if (!ptr1) {
        std::cout << "ptr1 is null after move\n";
    }
    ptr2->use();
    
    // Array version
    std::unique_ptr<int[]> arr = std::make_unique<int[]>(10);
    arr[0] = 42;
    std::cout << "arr[0] = " << arr[0] << "\n";
    
    // Manual release of ownership
    Resource* raw = ptr2.release();
    std::cout << "After release, ptr2 is " << (ptr2 ? "valid" : "null") << "\n";
    delete raw; // Now we must manually delete
    
    // Reset to manage new object
    ptr2.reset(new Resource("ptr2_new"));
    ptr2->use();
    
} // ptr2 automatically deleted here

// Example 2: Custom deleters
void customDeleterDemo() {
    std::cout << "\n=== Custom Deleter Demo ===\n";
    
    // Custom deleter for FILE*
    auto fileDeleter = [](FILE* fp) {
        if (fp) {
            std::cout << "Closing file\n";
            fclose(fp);
        }
    };
    
    std::unique_ptr<FILE, decltype(fileDeleter)> file(
        fopen("/tmp/test.txt", "w"),
        fileDeleter
    );
    
    if (file) {
        fprintf(file.get(), "Hello, World!\n");
    }
    
    // Custom deleter for array with logging
    auto arrayDeleter = [](int* p) {
        std::cout << "Deleting array with custom deleter\n";
        delete[] p;
    };
    
    std::unique_ptr<int[], decltype(arrayDeleter)> arr(
        new int[10],
        arrayDeleter
    );
}

// Example 3: std::shared_ptr basics
void demonstrateSharedPtr() {
    std::cout << "\n=== std::shared_ptr Demo ===\n";
    
    // Create shared_ptr
    std::shared_ptr<Resource> sptr1 = std::make_shared<Resource>("shared1");
    std::cout << "sptr1 use_count: " << sptr1.use_count() << "\n";
    
    {
        // Copy creates another owner
        std::shared_ptr<Resource> sptr2 = sptr1;
        std::cout << "sptr1 use_count after copy: " << sptr1.use_count() << "\n";
        std::cout << "sptr2 use_count: " << sptr2.use_count() << "\n";
        
        // Another copy
        std::shared_ptr<Resource> sptr3 = sptr2;
        std::cout << "use_count with 3 owners: " << sptr1.use_count() << "\n";
        
        // All three point to the same object
        std::cout << "All point to same? " 
                  << (sptr1.get() == sptr2.get() && sptr2.get() == sptr3.get()) 
                  << "\n";
    } // sptr2 and sptr3 destroyed, ref count decreases
    
    std::cout << "sptr1 use_count after inner scope: " << sptr1.use_count() << "\n";
} // sptr1 destroyed, ref count reaches 0, resource deleted

// Example 4: shared_ptr with vectors and containers
void sharedPtrInContainers() {
    std::cout << "\n=== shared_ptr in Containers ===\n";
    
    std::vector<std::shared_ptr<Resource>> resources;
    
    auto res1 = std::make_shared<Resource>("res1");
    auto res2 = std::make_shared<Resource>("res2");
    
    resources.push_back(res1);
    resources.push_back(res2);
    resources.push_back(res1); // Share ownership of res1
    
    std::cout << "res1 use_count: " << res1.use_count() << "\n"; // 2 (vector + res1)
    std::cout << "res2 use_count: " << res2.use_count() << "\n"; // 2 (vector + res2)
    
    resources.clear();
    std::cout << "After clear - res1 use_count: " << res1.use_count() << "\n";
}

// Example 5: Circular reference problem
class Node {
public:
    std::string name;
    std::shared_ptr<Node> next; // Circular reference potential
    
    Node(const std::string& n) : name(n) {
        std::cout << "Node '" << name << "' created\n";
    }
    
    ~Node() {
        std::cout << "Node '" << name << "' destroyed\n";
    }
};

void circularReferenceProblem() {
    std::cout << "\n=== Circular Reference Problem ===\n";
    
    auto node1 = std::make_shared<Node>("node1");
    auto node2 = std::make_shared<Node>("node2");
    
    node1->next = node2;
    node2->next = node1; // Circular reference!
    
    std::cout << "node1 use_count: " << node1.use_count() << "\n"; // 2
    std::cout << "node2 use_count: " << node2.use_count() << "\n"; // 2
    
    // Memory leak! Both nodes won't be destroyed because ref count never reaches 0
}

// Example 6: std::weak_ptr to solve circular references
class NodeFixed {
public:
    std::string name;
    std::shared_ptr<NodeFixed> next;
    std::weak_ptr<NodeFixed> prev; // weak_ptr breaks the cycle
    
    NodeFixed(const std::string& n) : name(n) {
        std::cout << "NodeFixed '" << name << "' created\n";
    }
    
    ~NodeFixed() {
        std::cout << "NodeFixed '" << name << "' destroyed\n";
    }
};

void demonstrateWeakPtr() {
    std::cout << "\n=== std::weak_ptr Demo ===\n";
    
    auto node1 = std::make_shared<NodeFixed>("node1");
    auto node2 = std::make_shared<NodeFixed>("node2");
    
    node1->next = node2;
    node2->prev = node1; // weak_ptr doesn't increase ref count
    
    std::cout << "node1 use_count: " << node1.use_count() << "\n"; // 1
    std::cout << "node2 use_count: " << node2.use_count() << "\n"; // 2 (node1->next + node2)
    
    // To use weak_ptr, must lock it to get shared_ptr
    if (auto prevNode = node2->prev.lock()) {
        std::cout << "Previous node: " << prevNode->name << "\n";
        std::cout << "use_count during lock: " << prevNode.use_count() << "\n"; // Temporarily 2
    }
    
    std::cout << "node1 use_count after lock released: " << node1.use_count() << "\n"; // Back to 1
    
} // Both nodes properly destroyed

// Example 7: weak_ptr expiration
void weakPtrExpiration() {
    std::cout << "\n=== weak_ptr Expiration Demo ===\n";
    
    std::weak_ptr<Resource> weakPtr;
    
    {
        auto sharedPtr = std::make_shared<Resource>("temporary");
        weakPtr = sharedPtr;
        
        std::cout << "Inside scope - expired: " << weakPtr.expired() << "\n";
        
        if (auto locked = weakPtr.lock()) {
            locked->use();
        }
    } // sharedPtr destroyed
    
    std::cout << "Outside scope - expired: " << weakPtr.expired() << "\n";
    
    if (auto locked = weakPtr.lock()) {
        locked->use();
    } else {
        std::cout << "Cannot lock expired weak_ptr\n";
    }
}

// Example 8: Factory pattern with shared_ptr
class Factory {
public:
    static std::shared_ptr<Resource> createResource(const std::string& name) {
        return std::make_shared<Resource>(name);
    }
    
    static std::unique_ptr<Resource> createUniqueResource(const std::string& name) {
        return std::make_unique<Resource>(name);
    }
};

// Example 9: enable_shared_from_this
class SelfAware : public std::enable_shared_from_this<SelfAware> {
private:
    std::string name;

public:
    SelfAware(const std::string& n) : name(n) {
        std::cout << "SelfAware '" << name << "' created\n";
    }
    
    ~SelfAware() {
        std::cout << "SelfAware '" << name << "' destroyed\n";
    }
    
    std::shared_ptr<SelfAware> getPtr() {
        return shared_from_this();
    }
    
    void doSomething() {
        std::cout << "SelfAware '" << name << "' doing something\n";
    }
};

void demonstrateEnableSharedFromThis() {
    std::cout << "\n=== enable_shared_from_this Demo ===\n";
    
    auto obj = std::make_shared<SelfAware>("obj1");
    std::cout << "Initial use_count: " << obj.use_count() << "\n";
    
    auto ptr = obj->getPtr();
    std::cout << "After getPtr use_count: " << obj.use_count() << "\n"; // 2
    
    ptr->doSomething();
}

// Example 10: Aliasing constructor
struct Person {
    std::string name;
    int age;
    
    Person(const std::string& n, int a) : name(n), age(a) {}
};

void demonstrateAliasingConstructor() {
    std::cout << "\n=== Aliasing Constructor Demo ===\n";
    
    auto person = std::make_shared<Person>("Alice", 30);
    
    // Create shared_ptr to member, keeping person alive
    std::shared_ptr<std::string> namePtr(person, &person->name);
    std::shared_ptr<int> agePtr(person, &person->age);
    
    std::cout << "Name: " << *namePtr << ", Age: " << *agePtr << "\n";
    std::cout << "person use_count: " << person.use_count() << "\n"; // 3
    
    person.reset(); // Release our reference
    std::cout << "After person.reset() use_count: " << namePtr.use_count() << "\n"; // 2
    
    // Person object still alive because namePtr and agePtr hold references
    std::cout << "Name still accessible: " << *namePtr << "\n";
}

// Example 11: Performance considerations
void performanceComparison() {
    std::cout << "\n=== Performance Considerations ===\n";
    
    // make_shared is more efficient - single allocation
    auto shared1 = std::make_shared<Resource>("shared1");
    
    // Two allocations: one for Resource, one for control block
    std::shared_ptr<Resource> shared2(new Resource("shared2"));
    
    // unique_ptr has zero overhead compared to raw pointer
    auto unique = std::make_unique<Resource>("unique");
    
    std::cout << "sizeof(Resource*): " << sizeof(Resource*) << "\n";
    std::cout << "sizeof(unique_ptr<Resource>): " << sizeof(unique) << "\n";
    std::cout << "sizeof(shared_ptr<Resource>): " << sizeof(shared1) << "\n";
}

int main() {
    demonstrateUniquePtr();
    customDeleterDemo();
    demonstrateSharedPtr();
    sharedPtrInContainers();
    circularReferenceProblem();
    demonstrateWeakPtr();
    weakPtrExpiration();
    
    std::cout << "\n=== Factory Demo ===\n";
    auto res = Factory::createResource("factory");
    res->use();
    
    demonstrateEnableSharedFromThis();
    demonstrateAliasingConstructor();
    performanceComparison();
    
    return 0;
}

/*
 * Key Takeaways:
 * 
 * 1. unique_ptr: Exclusive ownership, move-only, zero overhead
 *    - Use for: Resource ownership in single owner scenarios
 *    - Factory returns, class members, function returns
 * 
 * 2. shared_ptr: Shared ownership with reference counting
 *    - Use for: Multiple owners, object sharing
 *    - Thread-safe reference counting
 *    - Overhead: control block allocation
 * 
 * 3. weak_ptr: Non-owning observer, breaks circular references
 *    - Use for: Caches, observers, breaking cycles
 *    - Must lock() to access object
 *    - Can check if object still exists with expired()
 * 
 * 4. Prefer make_shared and make_unique
 *    - Exception safety
 *    - Better performance (single allocation for make_shared)
 * 
 * 5. Custom deleters allow managing non-memory resources
 * 
 * Interview Questions:
 * 
 * Q: When should you use unique_ptr vs shared_ptr?
 * A: Use unique_ptr by default for exclusive ownership. Only use shared_ptr
 *    when you genuinely need shared ownership. unique_ptr has zero overhead
 *    and can be converted to shared_ptr if needed later.
 * 
 * Q: How does shared_ptr reference counting work?
 * A: shared_ptr maintains a control block with reference counts. When copied,
 *    the count increases. When destroyed, it decreases. When count reaches 0,
 *    the object is deleted. Thread-safe for count operations.
 * 
 * Q: How do you prevent memory leaks from circular references?
 * A: Use weak_ptr for one direction of the relationship. For example, parent
 *    uses shared_ptr to child, child uses weak_ptr to parent.
 * 
 * Q: What's the difference between make_shared and shared_ptr constructor?
 * A: make_shared does single allocation for object + control block (better
 *    performance and exception safety). Constructor does two allocations.
 */
