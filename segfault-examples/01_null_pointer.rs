// Rust prevents null pointer dereferences at compile time!

fn main() {
    println!("=== Null Pointer Safety in Rust ===\n");
    
    // Example 1: Rust doesn't have null pointers!
    println!("Example 1: No null pointers in Rust");
    println!("Rust uses Option<T> instead of null");
    
    // This would be a compile error:
    // let ptr: &i32 = null;  // ERROR: 'null' doesn't exist!
    
    // Example 2: Option<T> - the safe way
    println!("\nExample 2: Using Option<T>");
    let maybe_value: Option<i32> = None;
    
    // Can't directly dereference - must handle None case
    match maybe_value {
        Some(value) => println!("  Value: {}", value),
        None => println!("  No value present (safe!)"),
    }
    
    // Example 3: Safe unwrapping with default
    println!("\nExample 3: Safe unwrapping");
    let value = maybe_value.unwrap_or(0);
    println!("  Value with default: {}", value);
    
    // Example 4: References can't be null
    println!("\nExample 4: References are always valid");
    let x = 42;
    let r: &i32 = &x;  // Reference is GUARANTEED to be valid
    println!("  Reference value: {}", *r);
    println!("  Rust guarantees this reference points to valid memory");
    
    // Example 5: Option with actual value
    println!("\nExample 5: Option with value");
    let some_value: Option<i32> = Some(42);
    
    if let Some(val) = some_value {
        println!("  Got value: {}", val);
    }
    
    // Example 6: Chaining operations safely
    println!("\nExample 6: Safe chaining");
    let result = maybe_value
        .map(|x| x * 2)
        .unwrap_or_else(|| {
            println!("  No value, using default");
            0
        });
    println!("  Result: {}", result);
    
    // Example 7: Pointer types in Rust (still safe)
    println!("\nExample 7: Raw pointers (require unsafe)");
    let x = 42;
    let ptr: *const i32 = &x as *const i32;
    
    // Can create null raw pointer
    let null_ptr: *const i32 = std::ptr::null();
    
    // But dereferencing requires unsafe block
    unsafe {
        // This is safe because ptr points to valid memory
        println!("  Value through pointer: {}", *ptr);
        
        // Dereferencing null would crash, but it's explicit:
        // println!("{}", *null_ptr);  // WOULD CRASH
    }
    
    println!("\n=== Key Takeaways ===");
    println!("1. Rust has NO null pointers by default");
    println!("2. Option<T> explicitly handles 'no value' cases");
    println!("3. References are guaranteed to be valid");
    println!("4. Raw pointers exist but require 'unsafe' blocks");
    println!("5. Compiler enforces handling of None cases");
    println!("\nResult: Null pointer dereference is prevented at COMPILE TIME!");
}
