// Rust prevents buffer overflows through bounds checking!

fn main() {
    println!("=== Buffer Overflow Safety in Rust ===\n");
    
    // Example 1: Array access is bounds-checked
    println!("Example 1: Array bounds checking");
    let arr = [1, 2, 3, 4, 5];
    println!("  Array: {:?}", arr);
    
    // Accessing within bounds
    println!("  arr[2] = {}", arr[2]);
    
    // This would PANIC at runtime (not compile error, but safe crash):
    // println!("  arr[100] = {}", arr[100]);  // PANIC: index out of bounds
    println!("  Out-of-bounds access would panic (safe crash, not undefined behavior)");
    
    // Example 2: Vec access is also bounds-checked
    println!("\nExample 2: Vec bounds checking");
    let vec = vec![1, 2, 3];
    println!("  Vec: {:?}", vec);
    
    // Safe access
    println!("  vec[1] = {}", vec[1]);
    
    // This would PANIC:
    // println!("  vec[10] = {}", vec[10]);  // PANIC: index out of bounds
    println!("  Out-of-bounds access causes panic, not memory corruption");
    
    // Example 3: Safe checked access with get()
    println!("\nExample 3: Safe access with get()");
    let vec = vec![1, 2, 3];
    
    match vec.get(10) {
        Some(val) => println!("  Value: {}", val),
        None => println!("  Index 10 is out of bounds (returned None)"),
    }
    
    // Example 4: Iterators prevent overflow
    println!("\nExample 4: Iterator safety");
    let arr = [1, 2, 3, 4, 5];
    
    for (i, val) in arr.iter().enumerate() {
        println!("  arr[{}] = {}", i, val);
    }
    println!("  Iterators can't go out of bounds!");
    
    // Example 5: Slices are bounds-checked
    println!("\nExample 5: Slice bounds checking");
    let arr = [1, 2, 3, 4, 5];
    let slice = &arr[1..4];
    println!("  Slice: {:?}", slice);
    
    // This would PANIC:
    // let bad_slice = &arr[1..100];  // PANIC: range end out of bounds
    println!("  Invalid slice ranges cause panic");
    
    // Example 6: String indexing is safe
    println!("\nExample 6: String safety");
    let s = String::from("Hello");
    
    // Can't directly index strings (prevents invalid UTF-8)
    // let c = s[0];  // COMPILE ERROR: cannot index into String
    
    // Must use safe methods
    if let Some(c) = s.chars().nth(0) {
        println!("  First char: {}", c);
    }
    
    // Example 7: No off-by-one errors with ranges
    println!("\nExample 7: Safe loops with ranges");
    let arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    
    for i in 0..arr.len() {  // Range is exclusive of end
        print!("{} ", arr[i]);
    }
    println!("\n  Range syntax prevents off-by-one errors");
    
    // Example 8: Checked arithmetic
    println!("\nExample 8: Checked indexing");
    let vec = vec![1, 2, 3, 4, 5];
    let index: usize = 2;
    
    // Safe access with bounds check
    if index < vec.len() {
        println!("  vec[{}] = {}", index, vec[index]);
    }
    
    // Using get for Option return
    if let Some(val) = vec.get(index) {
        println!("  Using get: {}", val);
    }
    
    // Example 9: Array initialization is safe
    println!("\nExample 9: Safe array initialization");
    let arr = [0; 10];  // Array of 10 zeros
    println!("  Array length: {}", arr.len());
    println!("  All elements initialized safely");
    
    // Example 10: Buffer copying is safe
    println!("\nExample 10: Safe buffer operations");
    let mut dest = vec![0u8; 5];
    let src = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    
    // copy_from_slice checks lengths!
    if dest.len() <= src.len() {
        let len = dest.len(); dest.copy_from_slice(&src[0..len]);
        println!("  Safely copied: {:?}", dest);
    }
    
    // This would PANIC if lengths don't match:
    // dest.copy_from_slice(&src);  // PANIC: source slice length (8) != destination (5)
    
    println!("\n=== Key Takeaways ===");
    println!("1. All array/vec accesses are bounds-checked");
    println!("2. Out-of-bounds access panics (safe crash) not undefined behavior");
    println!("3. get() method returns Option for safe access");
    println!("4. Iterators prevent index errors");
    println!("5. Slice ranges are validated");
    println!("6. String indexing prevents invalid UTF-8");
    println!("7. Range syntax prevents off-by-one errors");
    println!("8. No silent memory corruption");
    println!("\nResult: Buffer overflows are prevented at RUNTIME!");
    println!("(And many are caught at compile time)");
}
