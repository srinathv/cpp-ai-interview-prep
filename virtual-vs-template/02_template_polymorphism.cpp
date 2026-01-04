/**
 * Templates and Compile-Time Polymorphism
 * 
 * HOW TEMPLATES WORK:
 * 
 * Templates are resolved at COMPILE TIME through a process called template instantiation.
 * The compiler generates a SEPARATE copy of the code for each type used.
 * 
 * Template Instantiation Process:
 * 
 *   makeAnimalSpeak<Dog>(dog);
 *   makeAnimalSpeak<Cat>(cat);
 *   
 *   Compiler generates TWO separate functions:
 *   
 *   void makeAnimalSpeak_Dog(const Dog& animal) {
 *       animal.speak();  // Direct call to Dog::speak()
 *   }
 *   
 *   void makeAnimalSpeak_Cat(const Cat& animal) {
 *       animal.speak();  // Direct call to Cat::speak()
 *   }
 * 
 * CALL PROCESS (Compile-Time):
 * 
 *   makeAnimalSpeak(dog);
 *   
 *   1. Compiler knows exact type (Dog) at compile time
 *   2. Calls Dog-specific instantiated function
 *   3. Direct call to Dog::speak() - NO INDIRECTION
 *   
 *   Assembly pseudocode:
 *     call Dog::speak    ; DIRECT call - can be inlined!
 * 
 * NO VTABLE:
 *   - No hidden vptr in objects
 *   - No vtable lookups
 *   - No runtime overhead
 *   - Compiler can inline everything
 * 
 * PROS:
 *   - Zero runtime overhead (direct function calls)
 *   - Can be fully inlined by compiler
 *   - No memory overhead (no vptr, no vtable)
 *   - Compiler can optimize each instantiation separately
 *   - Duck typing (doesn't require inheritance)
 * 
 * CONS:
 *   - Type must be known at compile time
 *   - Code bloat (separate copy for each type)
 *   - Larger binary size
 *   - Cannot store different types in same container (without std::variant)
 *   - Longer compile times
 */

#include <iostream>
#include <chrono>

// No base class needed - duck typing!
// Just need to have the same interface

class Dog {
public:
    void speak() const {
        std::cout << "Woof!\n";
    }
    
    void move() const {
        std::cout << "Dog runs\n";
    }
};

class Cat {
public:
    void speak() const {
        std::cout << "Meow!\n";
    }
    
    void move() const {
        std::cout << "Cat prowls\n";
    }
};

class Bird {
public:
    void speak() const {
        std::cout << "Tweet!\n";
    }
    
    void move() const {
        std::cout << "Bird flies\n";
    }
};

// Template function - COMPILE TIME polymorphism
template<typename T>
void makeAnimalSpeak(const T& animal) {
    // Compiler generates SEPARATE function for each T
    // Direct call - NO vtable lookup
    // CAN BE INLINED
    animal.speak();
}

// Template class example
template<typename AnimalType>
class Zoo {
    AnimalType animal;
public:
    Zoo(AnimalType a) : animal(a) {}
    
    void makeNoise() const {
        animal.speak();  // Direct call, can be inlined
    }
};

// Demonstrate template instantiation
void demonstrateTemplateInstantiation() {
    std::cout << "=== Template Instantiation ===\n\n";
    
    Dog dog;
    Cat cat;
    
    // Show object sizes (NO vptr overhead)
    std::cout << "Size of Dog object: " << sizeof(dog) << " bytes\n";
    std::cout << "Size of Cat object: " << sizeof(cat) << " bytes\n";
    std::cout << "(No hidden vptr - just data members)\n\n";
    
    // Compiler generates two separate functions:
    // makeAnimalSpeak<Dog>(const Dog&)
    // makeAnimalSpeak<Cat>(const Cat&)
    
    std::cout << "Calling template function (direct dispatch):\n";
    makeAnimalSpeak(dog);  // makeAnimalSpeak<Dog>
    makeAnimalSpeak(cat);  // makeAnimalSpeak<Cat>
    
    std::cout << "\nCompiler generated separate functions for each type!\n";
}

// Demonstrate compile-time type deduction
template<typename T>
void processAnimal(const T& animal) {
    // Compile-time type information
    std::cout << "Processing animal of type: " << typeid(T).name() << "\n";
    animal.speak();
    animal.move();
}

// Demonstrate template specialization
template<typename T>
class AnimalTraits {
public:
    static const char* name() { return "Unknown"; }
};

// Specialized for Dog
template<>
class AnimalTraits<Dog> {
public:
    static const char* name() { return "Canine"; }
};

// Specialized for Cat
template<>
class AnimalTraits<Cat> {
public:
    static const char* name() { return "Feline"; }
};

int main() {
    std::cout << "=== Templates (Compile-Time Polymorphism) ===\n\n";
    
    // Demonstrate template instantiation
    demonstrateTemplateInstantiation();
    
    std::cout << "\n=== Type Deduction ===\n\n";
    
    Dog dog;
    Cat cat;
    Bird bird;
    
    // Type automatically deduced at compile time
    makeAnimalSpeak(dog);   // T = Dog
    makeAnimalSpeak(cat);   // T = Cat
    makeAnimalSpeak(bird);  // T = Bird
    
    std::cout << "\n=== Template Classes ===\n\n";
    
    // Each Zoo<T> is a DIFFERENT type
    Zoo<Dog> dogZoo(dog);
    Zoo<Cat> catZoo(cat);
    
    std::cout << "Dog zoo: ";
    dogZoo.makeNoise();
    
    std::cout << "Cat zoo: ";
    catZoo.makeNoise();
    
    // Note: Cannot do this with templates:
    // std::vector<Zoo<?>> zoos;  // ERROR: Must know type
    
    std::cout << "\n=== Template Specialization ===\n\n";
    
    std::cout << "Dog is a: " << AnimalTraits<Dog>::name() << "\n";
    std::cout << "Cat is a: " << AnimalTraits<Cat>::name() << "\n";
    
    std::cout << "\n=== Performance Characteristics ===\n\n";
    
    const int iterations = 10'000'000;
    
    // Measure template call performance
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        makeAnimalSpeak(dog);  // Direct call, likely inlined
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto template_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Template calls: " << template_time.count() << " μs\n";
    std::cout << "Direct calls - compiler can inline everything!\n";
    
    std::cout << "\n=== Code Size Trade-off ===\n\n";
    std::cout << "makeAnimalSpeak instantiated for:\n";
    std::cout << "  - Dog (separate function generated)\n";
    std::cout << "  - Cat (separate function generated)\n";
    std::cout << "  - Bird (separate function generated)\n";
    std::cout << "Binary contains 3 separate copies of the code!\n";
    
    return 0;
}
