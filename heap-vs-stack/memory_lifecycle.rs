fn print_address<T>(label: &str, ptr: &T) {
    println!("{:30}: {:p}", label, ptr);
}

// Example 1: Stack variable lifecycle
fn stack_lifecycle() {
    println!("\n=== Stack Variable Lifecycle ===");
    
    let x = 10;
    print_address("Stack var 'x'", &x);
    println!("Value of x after allocation: {}", x);
    
    let x = 20;  // Shadowing (not mutation)
    println!("Value of x after shadowing: {}", x);
    
    println!("x will be dropped when function returns");
}

// Example 2: Heap variable lifecycle
fn heap_lifecycle() {
    println!("\n=== Heap Variable Lifecycle ===");
    
    let ptr = Box::new(10);
    print_address("Box 'ptr' (on stack)", &ptr);
    print_address("Data pointed to (on heap)", &*ptr);
    println!("Value after allocation: {}", *ptr);
    
    let mut ptr = ptr;  // Take ownership
    *ptr = 20;
    println!("Value after modification: {}", *ptr);
    
    drop(ptr);
    println!("Memory freed with drop()");
    
    // SAFETY: Cannot use after free - Rust prevents this at compile time!
    // println!("Value after drop: {}", *ptr);  // COMPILE ERROR!
    println!("Rust prevents use-after-free at compile time!");
}

// Example 3: Ownership and borrowing
fn ownership_example() {
    println!("\n=== Ownership Transfer ===");
    
    let s1 = String::from("Hello");
    print_address("String s1", &s1);
    println!("s1 value: {}", s1);
    
    let s2 = s1;  // Ownership transferred
    print_address("String s2 (same data)", &s2);
    println!("s2 value: {}", s2);
    
    // println!("{}", s1);  // COMPILE ERROR! s1 no longer valid
    println!("s1 is no longer accessible (moved to s2)");
}

// Example 4: Borrowing demonstration
fn borrowing_example() {
    println!("\n=== Borrowing (References) ===");
    
    let s = String::from("Hello");
    print_address("Original string", &s);
    
    let r1 = &s;
    let r2 = &s;
    print_address("Reference r1", &r1);
    print_address("Reference r2", &r2);
    
    println!("r1: {}, r2: {}", r1, r2);
    println!("Original s still valid: {}", s);
    
    println!("\nMultiple immutable borrows are allowed!");
}

// Example 5: Mutable borrowing
fn mutable_borrowing() {
    println!("\n=== Mutable Borrowing ===");
    
    let mut s = String::from("Hello");
    print_address("Original string", &s);
    println!("Before: {}", s);
    
    let r = &mut s;
    r.push_str(", World!");
    println!("After mutation via reference: {}", r);
    
    // let r2 = &mut s;  // COMPILE ERROR! Only one mutable borrow allowed
    println!("Only one mutable borrow allowed at a time!");
}

// Example 6: Lifetime demonstration
fn lifetime_example() {
    println!("\n=== Lifetimes ===");
    
    let outer = String::from("Outer");
    print_address("Outer string", &outer);
    
    {
        let inner = String::from("Inner");
        print_address("Inner string", &inner);
        println!("Both valid in inner scope: {}, {}", outer, inner);
    }  // inner dropped here
    
    println!("Outer still valid: {}", outer);
    // println!("{}", inner);  // COMPILE ERROR! inner out of scope
}

// Example 7: Stack growth with recursion
fn stack_growth(depth: i32, max_depth: i32) {
    if depth > max_depth {
        return;
    }
    
    let local_var = depth;
    print_address(&format!("Stack var at depth {}", depth), &local_var);
    println!("Value: {}", local_var);
    
    stack_growth(depth + 1, max_depth);
    
    println!("Returning from depth {}, value still: {}", depth, local_var);
}

fn demonstrate_stack_growth() {
    println!("\n=== Stack Growth (Recursion) ===");
    stack_growth(0, 5);
}

// Example 8: RAII with Drop trait
struct Resource {
    data: Box<i32>,
    name: String,
}

impl Resource {
    fn new(val: i32, name: &str) -> Self {
        println!("Resource '{}' allocated: {}", name, val);
        Resource {
            data: Box::new(val),
            name: name.to_string(),
        }
    }
    
    fn use_resource(&self) {
        println!("Using resource '{}': {}", self.name, *self.data);
    }
}

impl Drop for Resource {
    fn drop(&mut self) {
        println!("Resource '{}' freed: {}", self.name, *self.data);
    }
}

fn raii_example() {
    println!("\n=== RAII (Automatic Resource Management) ===");
    
    let r1 = Resource::new(100, "r1");
    r1.use_resource();
    
    {
        let r2 = Resource::new(200, "r2");
        r2.use_resource();
        println!("r2 scope ending...");
    }  // r2 automatically dropped here
    
    println!("r1 still valid...");
    r1.use_resource();
    
    println!("Function ending, r1 will be dropped...");
}

// Example 9: Demonstrating Copy vs Move semantics
fn copy_vs_move() {
    println!("\n=== Copy vs Move Semantics ===");
    
    // Stack types implement Copy
    let x = 5;
    let y = x;  // Copied
    println!("x: {}, y: {} (both valid, Copy trait)", x, y);
    
    // Heap types use Move
    let s1 = String::from("Hello");
    let s2 = s1;  // Moved
    println!("s2: {} (s1 no longer valid, Move semantics)", s2);
    // println!("{}", s1);  // COMPILE ERROR!
}

// Example 10: Reference counting for shared ownership
use std::rc::Rc;

fn shared_ownership() {
    println!("\n=== Shared Ownership with Rc ===");
    
    let data = Rc::new(String::from("Shared data"));
    print_address("Rc data", &data);
    println!("Strong count: {}", Rc::strong_count(&data));
    
    {
        let data2 = Rc::clone(&data);
        print_address("Cloned Rc", &data2);
        println!("Strong count: {}", Rc::strong_count(&data));
        println!("data: {}, data2: {}", data, data2);
    }  // data2 dropped, count decreases
    
    println!("Strong count after data2 dropped: {}", Rc::strong_count(&data));
    println!("data still valid: {}", data);
}

fn main() {
    println!("=== Memory Allocation and Usage Lifecycle in Rust ===");
    
    stack_lifecycle();
    heap_lifecycle();
    ownership_example();
    borrowing_example();
    mutable_borrowing();
    lifetime_example();
    demonstrate_stack_growth();
    raii_example();
    copy_vs_move();
    shared_ownership();
    
    println!("\nProgram ending - all resources automatically freed!");
    println!("Rust guarantees memory safety at compile time!");
}
