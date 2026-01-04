/**
 * Virtual Functions and Runtime Polymorphism
 * 
 * HOW VTABLES WORK:
 * 
 * When a class has virtual functions, the compiler creates a vtable (virtual table)
 * for that class. Each object of that class contains a hidden pointer (vptr) to its
 * class's vtable.
 * 
 * Memory Layout:
 * 
 *   Object in Memory:
 *   +------------------+
 *   | vptr (hidden)    |  --> Points to vtable
 *   +------------------+
 *   | member data      |
 *   +------------------+
 * 
 *   Vtable for Dog:
 *   +------------------+
 *   | &Dog::speak      |  --> Address of Dog's speak() implementation
 *   | &Dog::move       |  --> Address of Dog's move() implementation
 *   +------------------+
 * 
 *   Vtable for Cat:
 *   +------------------+
 *   | &Cat::speak      |  --> Address of Cat's speak() implementation
 *   | &Cat::move       |  --> Address of Cat's move() implementation
 *   +------------------+
 * 
 * CALL PROCESS (Runtime):
 * 
 *   animal->speak();
 *   
 *   1. Follow vptr to find vtable
 *   2. Look up speak() entry in vtable
 *   3. Indirect jump to function address
 *   
 *   Assembly pseudocode:
 *     mov rax, [animal]          ; Get vptr
 *     mov rbx, [rax]             ; Get first vtable entry (speak)
 *     call rbx                   ; Indirect call - RUNTIME OVERHEAD
 * 
 * PROS:
 *   - Runtime polymorphism (type determined at runtime)
 *   - One compiled function can handle many types
 *   - Smaller binary size (one function compiled once)
 *   - Can store different types in same container
 * 
 * CONS:
 *   - Indirect function call (2 pointer dereferences)
 *   - Cannot be inlined by compiler
 *   - Memory overhead (vptr per object, vtable per class)
 *   - Cache unfriendly (vtable lookups)
 */

#include <iostream>
#include <vector>
#include <memory>
#include <chrono>

// Base class with virtual functions
class Animal {
public:
    // Virtual destructor (essential for polymorphic base classes)
    virtual ~Animal() = default;
    
    // Virtual functions - resolved at RUNTIME
    virtual void speak() const = 0;  // Pure virtual
    virtual void move() const = 0;   // Pure virtual
    
    // Non-virtual function - resolved at COMPILE TIME
    void describe() const {
        std::cout << "I am an animal\n";
    }
};

// Derived class 1
class Dog : public Animal {
public:
    // Override virtual functions
    void speak() const override {
        std::cout << "Woof!\n";
    }
    
    void move() const override {
        std::cout << "Dog runs\n";
    }
};

// Derived class 2
class Cat : public Animal {
public:
    void speak() const override {
        std::cout << "Meow!\n";
    }
    
    void move() const override {
        std::cout << "Cat prowls\n";
    }
};

// Derived class 3
class Bird : public Animal {
public:
    void speak() const override {
        std::cout << "Tweet!\n";
    }
    
    void move() const override {
        std::cout << "Bird flies\n";
    }
};

// Function using runtime polymorphism
void makeAnimalSpeak(const Animal& animal) {
    // Runtime dispatch through vtable
    // Compiler doesn't know which speak() until runtime
    animal.speak();
}

// Demonstrate vtable overhead
void demonstrateVtableLayout() {
    std::cout << "=== Vtable Memory Layout ===\n\n";
    
    Dog dog;
    Cat cat;
    
    // Show object sizes (includes vptr overhead)
    std::cout << "Size of Dog object: " << sizeof(dog) << " bytes\n";
    std::cout << "Size of Cat object: " << sizeof(cat) << " bytes\n";
    std::cout << "(Includes hidden vptr: " << sizeof(void*) << " bytes)\n\n";
    
    // Show polymorphic behavior
    Animal* animals[2] = {&dog, &cat};
    
    std::cout << "Calling through base pointer (vtable dispatch):\n";
    for (int i = 0; i < 2; i++) {
        // Each call:
        // 1. Read vptr from object
        // 2. Find speak() in vtable
        // 3. Indirect call to actual function
        animals[i]->speak();
    }
}

int main() {
    std::cout << "=== Virtual Functions (Runtime Polymorphism) ===\n\n";
    
    // Demonstrate vtable layout
    demonstrateVtableLayout();
    
    std::cout << "\n=== Polymorphic Container ===\n\n";
    
    // KEY ADVANTAGE: Can store different types in same container
    std::vector<std::unique_ptr<Animal>> zoo;
    zoo.push_back(std::make_unique<Dog>());
    zoo.push_back(std::make_unique<Cat>());
    zoo.push_back(std::make_unique<Bird>());
    zoo.push_back(std::make_unique<Dog>());
    
    std::cout << "Animals in zoo:\n";
    for (const auto& animal : zoo) {
        animal->speak();  // Runtime dispatch via vtable
    }
    
    std::cout << "\n=== Runtime Type Selection ===\n\n";
    
    // Can choose type at runtime
    std::cout << "Choose animal (1=Dog, 2=Cat, 3=Bird): ";
    int choice = 2;  // Simulating user input
    
    std::unique_ptr<Animal> selectedAnimal;
    if (choice == 1) {
        selectedAnimal = std::make_unique<Dog>();
    } else if (choice == 2) {
        selectedAnimal = std::make_unique<Cat>();
    } else {
        selectedAnimal = std::make_unique<Bird>();
    }
    
    std::cout << "Your animal says: ";
    selectedAnimal->speak();  // Type decided at runtime!
    
    std::cout << "\n=== Performance Characteristics ===\n\n";
    
    const int iterations = 10'000'000;
    Dog dog;
    Animal* animalPtr = &dog;
    
    // Measure virtual call overhead
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        animalPtr->speak();  // Virtual call (vtable lookup)
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto virtual_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Virtual calls: " << virtual_time.count() << " μs\n";
    std::cout << "Each call includes vtable lookup overhead\n";
    
    return 0;
}
