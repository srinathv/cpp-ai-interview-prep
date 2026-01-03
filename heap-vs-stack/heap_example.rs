fn heap_allocation() {
    println!("=== Heap Allocation Example ===");
    let large_array = Box::new([0i32; 1_000_000]);
    println!("Large heap allocation succeeded!");
    println!("First element: {}", large_array[0]);
    println!("Last element: {}", large_array[999_999]);
}

fn deep_recursion_with_heap(depth: u64, max_depth: u64) {
    let data = Box::new(depth);
    if depth % 1000 == 0 {
        println!("Recursion depth: {}", depth);
    }
    if depth < max_depth {
        deep_recursion_with_heap(depth + 1, max_depth);
    }
}

fn dynamic_heap_allocation() {
    println!("\n=== Dynamic Heap Allocation with Vec ===");
    let mut vec = Vec::new();
    for i in 0..1_000_000 {
        vec.push(i);
    }
    println!("Vec with {} elements allocated on heap", vec.len());
    println!("First element: {}", vec[0]);
    println!("Last element: {}", vec[vec.len() - 1]);
}

fn main() {
    heap_allocation();
    println!("\n=== Deep Recursion with Heap Storage ===");
    deep_recursion_with_heap(0, 10_000);
    println!("Completed 10,000 recursive calls successfully!");
    dynamic_heap_allocation();
}
