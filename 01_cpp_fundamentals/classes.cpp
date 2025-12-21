/*
 * C++ Classes and Object-Oriented Programming Fundamentals
 * 
 * Key Concepts:
 * - Class definition and member functions
 * - Constructors and destructors
 * - Access specifiers (public, private, protected)
 * - Member initialization
 * - The Rule of Three/Five/Zero
 * - Inheritance and polymorphism
 * - Virtual functions and abstract classes
 * - Operator overloading
 * 
 * Interview Topics:
 * - What's the difference between struct and class?
 * - What is the Rule of Five?
 * - How does virtual dispatch work?
 */

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cstring>

// Example 1: Basic class definition
class Rectangle {
private:
    double width;
    double height;

public:
    // Constructor
    Rectangle(double w, double h) : width(w), height(h) {
        std::cout << "Rectangle(" << width << ", " << height << ") constructed\n";
    }
    
    // Destructor
    ~Rectangle() {
        std::cout << "Rectangle(" << width << ", " << height << ") destroyed\n";
    }
    
    // Member functions
    double area() const {
        return width * height;
    }
    
    double perimeter() const {
        return 2 * (width + height);
    }
    
    void scale(double factor) {
        width *= factor;
        height *= factor;
    }
    
    // Getters
    double getWidth() const { return width; }
    double getHeight() const { return height; }
    
    // Setters
    void setWidth(double w) { width = w; }
    void setHeight(double h) { height = h; }
};

void demonstrateBasicClass() {
    std::cout << "=== Basic Class ===\n";
    
    Rectangle rect(5.0, 3.0);
    std::cout << "Area: " << rect.area() << "\n";
    std::cout << "Perimeter: " << rect.perimeter() << "\n";
    
    rect.scale(2.0);
    std::cout << "After scaling: Area = " << rect.area() << "\n";
}

// Example 2: Struct vs Class (only difference is default access)
struct Point {  // Members are public by default
    double x;
    double y;
    
    Point(double x = 0, double y = 0) : x(x), y(y) {}
    
    double distanceFromOrigin() const {
        return std::sqrt(x * x + y * y);
    }
};

void demonstrateStructVsClass() {
    std::cout << "\n=== Struct vs Class ===\n";
    
    Point p(3.0, 4.0);
    std::cout << "Point(" << p.x << ", " << p.y << ")\n";  // Direct access OK
    std::cout << "Distance from origin: " << p.distanceFromOrigin() << "\n";
}

// Example 3: Constructor variations
class Person {
private:
    std::string name;
    int age;
    std::string email;

public:
    // Default constructor
    Person() : name("Unknown"), age(0), email("") {
        std::cout << "Default constructor\n";
    }
    
    // Parameterized constructor
    Person(const std::string& n, int a) : name(n), age(a), email("") {
        std::cout << "Parameterized constructor\n";
    }
    
    // Constructor with all parameters
    Person(const std::string& n, int a, const std::string& e) 
        : name(n), age(a), email(e) {
        std::cout << "Full constructor\n";
    }
    
    // Copy constructor
    Person(const Person& other) 
        : name(other.name), age(other.age), email(other.email) {
        std::cout << "Copy constructor\n";
    }
    
    void display() const {
        std::cout << "Person: " << name << ", " << age << " years old";
        if (!email.empty()) {
            std::cout << ", email: " << email;
        }
        std::cout << "\n";
    }
};

void demonstrateConstructors() {
    std::cout << "\n=== Constructors ===\n";
    
    Person p1;
    Person p2("Alice", 30);
    Person p3("Bob", 25, "bob@example.com");
    Person p4(p2);  // Copy constructor
    
    p1.display();
    p2.display();
    p3.display();
    p4.display();
}

// Example 4: The Rule of Three (pre-C++11)
class Buffer {
private:
    char* data;
    size_t size;

public:
    // Constructor
    Buffer(size_t sz) : size(sz), data(new char[sz]) {
        std::memset(data, 0, size);
        std::cout << "Buffer(" << size << ") constructed\n";
    }
    
    // Destructor
    ~Buffer() {
        delete[] data;
        std::cout << "Buffer destroyed\n";
    }
    
    // Copy constructor
    Buffer(const Buffer& other) : size(other.size), data(new char[other.size]) {
        std::memcpy(data, other.data, size);
        std::cout << "Buffer copy constructed\n";
    }
    
    // Copy assignment operator
    Buffer& operator=(const Buffer& other) {
        if (this != &other) {
            delete[] data;
            size = other.size;
            data = new char[size];
            std::memcpy(data, other.data, size);
            std::cout << "Buffer copy assigned\n";
        }
        return *this;
    }
    
    size_t getSize() const { return size; }
};

void demonstrateRuleOfThree() {
    std::cout << "\n=== Rule of Three ===\n";
    
    Buffer b1(100);
    Buffer b2(b1);       // Copy constructor
    Buffer b3(50);
    b3 = b1;             // Copy assignment
}

// Example 5: The Rule of Five (C++11 and later)
class String {
private:
    char* data;
    size_t length;

public:
    // Constructor
    String(const char* str = "") {
        length = std::strlen(str);
        data = new char[length + 1];
        std::strcpy(data, str);
        std::cout << "String(\"" << data << "\") constructed\n";
    }
    
    // Destructor
    ~String() {
        std::cout << "String(\"" << data << "\") destroyed\n";
        delete[] data;
    }
    
    // Copy constructor
    String(const String& other) : length(other.length) {
        data = new char[length + 1];
        std::strcpy(data, other.data);
        std::cout << "String copy constructed\n";
    }
    
    // Copy assignment
    String& operator=(const String& other) {
        if (this != &other) {
            delete[] data;
            length = other.length;
            data = new char[length + 1];
            std::strcpy(data, other.data);
            std::cout << "String copy assigned\n";
        }
        return *this;
    }
    
    // Move constructor
    String(String&& other) noexcept : data(other.data), length(other.length) {
        other.data = nullptr;
        other.length = 0;
        std::cout << "String move constructed\n";
    }
    
    // Move assignment
    String& operator=(String&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            length = other.length;
            other.data = nullptr;
            other.length = 0;
            std::cout << "String move assigned\n";
        }
        return *this;
    }
    
    const char* c_str() const { return data; }
};

void demonstrateRuleOfFive() {
    std::cout << "\n=== Rule of Five ===\n";
    
    String s1("Hello");
    String s2(s1);              // Copy
    String s3(std::move(s1));   // Move
    
    String s4("World");
    s4 = s2;                    // Copy assignment
    s4 = std::move(s3);         // Move assignment
}

// Example 6: Inheritance
class Animal {
protected:
    std::string name;
    int age;

public:
    Animal(const std::string& n, int a) : name(n), age(a) {
        std::cout << "Animal constructed\n";
    }
    
    virtual ~Animal() {
        std::cout << "Animal destroyed\n";
    }
    
    virtual void makeSound() const {
        std::cout << name << " makes a sound\n";
    }
    
    void displayInfo() const {
        std::cout << "Name: " << name << ", Age: " << age << "\n";
    }
};

class Dog : public Animal {
private:
    std::string breed;

public:
    Dog(const std::string& n, int a, const std::string& b) 
        : Animal(n, a), breed(b) {
        std::cout << "Dog constructed\n";
    }
    
    ~Dog() override {
        std::cout << "Dog destroyed\n";
    }
    
    void makeSound() const override {
        std::cout << name << " barks: Woof!\n";
    }
    
    void fetch() const {
        std::cout << name << " is fetching the ball\n";
    }
};

class Cat : public Animal {
public:
    Cat(const std::string& n, int a) : Animal(n, a) {
        std::cout << "Cat constructed\n";
    }
    
    ~Cat() override {
        std::cout << "Cat destroyed\n";
    }
    
    void makeSound() const override {
        std::cout << name << " meows: Meow!\n";
    }
};

void demonstrateInheritance() {
    std::cout << "\n=== Inheritance ===\n";
    
    Dog dog("Buddy", 3, "Golden Retriever");
    dog.displayInfo();
    dog.makeSound();
    dog.fetch();
    
    std::cout << "\n";
    
    Cat cat("Whiskers", 2);
    cat.displayInfo();
    cat.makeSound();
}

// Example 7: Polymorphism
void demonstratePolymorphism() {
    std::cout << "\n=== Polymorphism ===\n";
    
    std::vector<std::unique_ptr<Animal>> animals;
    animals.push_back(std::make_unique<Dog>("Rex", 4, "Labrador"));
    animals.push_back(std::make_unique<Cat>("Mittens", 3));
    animals.push_back(std::make_unique<Dog>("Max", 2, "Beagle"));
    
    std::cout << "All animals making sounds:\n";
    for (const auto& animal : animals) {
        animal->makeSound();  // Virtual dispatch
    }
}

// Example 8: Abstract class (interface)
class Shape {
public:
    virtual ~Shape() = default;
    
    // Pure virtual functions (abstract methods)
    virtual double area() const = 0;
    virtual double perimeter() const = 0;
    virtual void draw() const = 0;
};

class Circle : public Shape {
private:
    double radius;

public:
    Circle(double r) : radius(r) {}
    
    double area() const override {
        return 3.14159 * radius * radius;
    }
    
    double perimeter() const override {
        return 2 * 3.14159 * radius;
    }
    
    void draw() const override {
        std::cout << "Drawing circle with radius " << radius << "\n";
    }
};

class Square : public Shape {
private:
    double side;

public:
    Square(double s) : side(s) {}
    
    double area() const override {
        return side * side;
    }
    
    double perimeter() const override {
        return 4 * side;
    }
    
    void draw() const override {
        std::cout << "Drawing square with side " << side << "\n";
    }
};

void demonstrateAbstractClass() {
    std::cout << "\n=== Abstract Class ===\n";
    
    // Shape s;  // Error: cannot instantiate abstract class
    
    std::vector<std::unique_ptr<Shape>> shapes;
    shapes.push_back(std::make_unique<Circle>(5.0));
    shapes.push_back(std::make_unique<Square>(4.0));
    
    for (const auto& shape : shapes) {
        shape->draw();
        std::cout << "  Area: " << shape->area() << "\n";
        std::cout << "  Perimeter: " << shape->perimeter() << "\n";
    }
}

// Example 9: Operator overloading
class Complex {
private:
    double real;
    double imag;

public:
    Complex(double r = 0, double i = 0) : real(r), imag(i) {}
    
    // Arithmetic operators
    Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }
    
    Complex operator-(const Complex& other) const {
        return Complex(real - other.real, imag - other.imag);
    }
    
    Complex operator*(const Complex& other) const {
        return Complex(
            real * other.real - imag * other.imag,
            real * other.imag + imag * other.real
        );
    }
    
    // Comparison operator
    bool operator==(const Complex& other) const {
        return real == other.real && imag == other.imag;
    }
    
    // Stream insertion operator (friend function)
    friend std::ostream& operator<<(std::ostream& os, const Complex& c) {
        os << c.real;
        if (c.imag >= 0) os << "+";
        os << c.imag << "i";
        return os;
    }
    
    // Unary operator
    Complex operator-() const {
        return Complex(-real, -imag);
    }
    
    // Compound assignment
    Complex& operator+=(const Complex& other) {
        real += other.real;
        imag += other.imag;
        return *this;
    }
};

void demonstrateOperatorOverloading() {
    std::cout << "\n=== Operator Overloading ===\n";
    
    Complex c1(3, 4);
    Complex c2(1, 2);
    
    std::cout << "c1 = " << c1 << "\n";
    std::cout << "c2 = " << c2 << "\n";
    
    Complex c3 = c1 + c2;
    std::cout << "c1 + c2 = " << c3 << "\n";
    
    Complex c4 = c1 * c2;
    std::cout << "c1 * c2 = " << c4 << "\n";
    
    Complex c5 = -c1;
    std::cout << "-c1 = " << c5 << "\n";
}

// Example 10: Static members
class Counter {
private:
    static int count;
    int id;

public:
    Counter() : id(++count) {
        std::cout << "Counter " << id << " created (total: " << count << ")\n";
    }
    
    ~Counter() {
        std::cout << "Counter " << id << " destroyed\n";
    }
    
    static int getCount() {
        return count;
    }
    
    int getId() const {
        return id;
    }
};

// Static member must be defined outside class
int Counter::count = 0;

void demonstrateStaticMembers() {
    std::cout << "\n=== Static Members ===\n";
    
    std::cout << "Initial count: " << Counter::getCount() << "\n";
    
    Counter c1;
    Counter c2;
    Counter c3;
    
    std::cout << "Count after creating 3: " << Counter::getCount() << "\n";
}

// Example 11: Const correctness
class Data {
private:
    int value;
    mutable int accessCount;  // Can be modified even in const methods

public:
    Data(int v) : value(v), accessCount(0) {}
    
    // Const member function - cannot modify members (except mutable ones)
    int getValue() const {
        accessCount++;  // OK because it's mutable
        return value;
    }
    
    // Non-const member function
    void setValue(int v) {
        value = v;
    }
    
    int getAccessCount() const {
        return accessCount;
    }
};

void demonstrateConstCorrectness() {
    std::cout << "\n=== Const Correctness ===\n";
    
    Data data(42);
    std::cout << "Value: " << data.getValue() << "\n";
    std::cout << "Access count: " << data.getAccessCount() << "\n";
    
    const Data constData(100);
    std::cout << "Const value: " << constData.getValue() << "\n";
    // constData.setValue(200);  // Error: cannot call non-const method
}

int main() {
    demonstrateBasicClass();
    demonstrateStructVsClass();
    demonstrateConstructors();
    demonstrateRuleOfThree();
    demonstrateRuleOfFive();
    demonstrateInheritance();
    demonstratePolymorphism();
    demonstrateAbstractClass();
    demonstrateOperatorOverloading();
    demonstrateStaticMembers();
    demonstrateConstCorrectness();
    
    return 0;
}

/*
 * Key Takeaways:
 * 
 * 1. Classes encapsulate data and behavior
 *    - Private members: internal implementation
 *    - Public members: interface to outside world
 *    - Protected members: accessible to derived classes
 * 
 * 2. Constructors initialize objects, destructors clean up
 *    - Use member initializer lists
 *    - Destructor called automatically when object goes out of scope
 * 
 * 3. Rule of Three (C++98): If you define one, define all three:
 *    - Destructor
 *    - Copy constructor
 *    - Copy assignment operator
 * 
 * 4. Rule of Five (C++11): Add two more:
 *    - Move constructor
 *    - Move assignment operator
 * 
 * 5. Rule of Zero: Prefer using smart pointers and STL containers
 *    - Let compiler generate special members
 *    - Only define custom ones when managing resources directly
 * 
 * 6. Inheritance models "is-a" relationship
 *    - Base class: common interface/behavior
 *    - Derived class: specialized behavior
 *    - Use virtual for polymorphic behavior
 * 
 * 7. Virtual functions enable runtime polymorphism
 *    - Virtual dispatch through vtable
 *    - Always make destructor virtual in base classes
 *    - Use override keyword in derived classes
 * 
 * 8. Abstract classes define interfaces
 *    - Pure virtual functions (= 0)
 *    - Cannot be instantiated
 *    - Derived classes must implement pure virtual functions
 * 
 * Interview Questions:
 * 
 * Q: What's the difference between struct and class in C++?
 * A: Only default access specifier. Struct members are public by default,
 *    class members are private by default. Otherwise identical.
 * 
 * Q: What is the Rule of Five?
 * A: If a class manages resources and defines any of these five special
 *    member functions, it should define all five: destructor, copy constructor,
 *    copy assignment, move constructor, move assignment.
 * 
 * Q: Why make destructors virtual?
 * A: When deleting through base class pointer, virtual destructor ensures
 *    derived class destructor is called. Without it, only base destructor
 *    runs, causing resource leaks.
 * 
 * Q: What is const correctness?
 * A: Marking methods const when they don't modify object state. Allows
 *    calling methods on const objects. Const member functions can't modify
 *    members (except mutable ones) or call non-const methods.
 */
