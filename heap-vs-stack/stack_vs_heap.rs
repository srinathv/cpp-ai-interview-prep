use std::time::Instant;

struct LargeObject {
    data: [i32; 1000],
}

impl LargeObject {
    fn new() -> Self {
        LargeObject { data: [0; 1000] }
    }
}

fn stack_allocation() {
    let start = Instant::now();
    let obj = LargeObject::new();
    let duration = start.elapsed();
    println!("Stack allocation time: {:?}", duration);
    println!("Object data: {}", obj.data[0]);
}

fn heap_allocation() {
    let start = Instant::now();
    let obj = Box::new(LargeObject::new());
    let duration = start.elapsed();
    println!("Heap allocation time: {:?}", duration);
    println!("Object data: {}", obj.data[0]);
}

fn demonstrate_ownership() {
    println!("\n=== Rust Ownership Demo ===");
    let stack_str = "Hello, Stack!";
    println!("Stack string: {}", stack_str);
    let heap_str = String::from("Hello, Heap!");
    println!("Heap string: {}", heap_str);
    let moved_str = heap_str;
    println!("Moved string: {}", moved_str);
}

fn main() {
    println!("=== Stack vs Heap Comparison ===");
    println!("\nKey Differences:");
    println!("1. Stack: Fast, automatic cleanup, limited size, Copy semantics");
    println!("2. Heap: Slower, automatic cleanup (RAII), large size, Move semantics");
    println!("\nPerformance comparison:\n");
    stack_allocation();
    heap_allocation();
    demonstrate_ownership();
}
