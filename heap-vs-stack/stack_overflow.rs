fn blow_stack(depth: u64) {
    println!("Recursion depth: {}", depth);
    blow_stack(depth + 1);
}

fn large_stack_allocation() {
    let large_array: [i32; 1_000_000] = [0; 1_000_000];
    println!("Large stack allocation succeeded: {}", large_array[0]);
}

fn deep_recursion(depth: u64) {
    let _local_var = depth;
    if depth % 1000 == 0 {
        println!("Recursion depth: {}", depth);
    }
    deep_recursion(depth + 1);
}

fn main() {
    println!("=== Stack Overflow Example ===");
    println!("Warning: This will crash!");
    println!("Uncomment blow_stack(0), large_stack_allocation(), or deep_recursion(0) to crash");
}
