// Rust prevents ALL these segfaults at compile time or runtime!

use std::rc::Rc;

fn main() {
    println!("=== How Rust Prevents All Segfault Categories ===\n");
    
    println!("1. NULL POINTER DEREFERENCE");
    println!("   C++: int* ptr = nullptr; *ptr = 42; // CRASH");
    println!("   Rust: No null pointers! Use Option<T>");
    let maybe: Option<i32> = None;
    if let Some(val) = maybe {
        println!("   Value: {}", val);
    } else {
        println!("   ✓ Must handle None case explicitly");
    }
    
    println!("\n2. USE-AFTER-FREE");
    println!("   C++: delete ptr; *ptr = 42; // CRASH");
    println!("   Rust: Ownership prevents use after move/drop");
    let data = Box::new(42);
    drop(data);
    // println!("{}", data);  // COMPILE ERROR: value used after move
    println!("   ✓ Compile error: value used after move");
    
    println!("\n3. DANGLING POINTER");
    println!("   C++: int* f() {{ int x=42; return &x; }} // CRASH");
    println!("   Rust: Cannot return reference to local variable");
    // fn bad() -> &i32 { let x = 42; &x }  // COMPILE ERROR
    println!("   ✓ Compile error: returns reference with insufficient lifetime");
    
    println!("\n4. BUFFER OVERFLOW");
    println!("   C++: arr[100] = 42; // CRASH");
    println!("   Rust: Bounds checking at runtime");
    let arr = [1, 2, 3, 4, 5];
    // arr[100] = 42;  // PANIC (safe crash, not undefined behavior)
    if let Some(val) = arr.get(2) {
        println!("   ✓ Safe access: arr[2] = {}", val);
    }
    println!("   Out-of-bounds would panic (safe) not corrupt memory");
    
    println!("\n5. STACK OVERFLOW");
    println!("   C++: void f() {{ f(); }} // CRASH");
    println!("   Rust: Same issue, but detected by OS");
    // fn stack_overflow() { stack_overflow(); }
    println!("   ✓ OS detects stack overflow and terminates safely");
    
    println!("\n6. UNINITIALIZED POINTER");
    println!("   C++: int* ptr; *ptr = 42; // CRASH");
    println!("   Rust: All variables must be initialized");
    // let ptr: &i32;  // COMPILE ERROR if used
    // println!("{}", ptr);  // ERROR: use of possibly-uninitialized variable
    let ptr: &i32 = &42;
    println!("   ✓ Must initialize before use: {}", ptr);
    
    println!("\n7. DOUBLE FREE");
    println!("   C++: delete ptr; delete ptr; // CRASH");
    println!("   Rust: Ownership prevents double-free");
    let data = Box::new(42);
    drop(data);
    // drop(data);  // COMPILE ERROR: use of moved value
    println!("   ✓ Compile error: can't drop twice");
    
    println!("\n8. WRITE TO READ-ONLY MEMORY");
    println!("   C++: char* s = \"hi\"; s[0] = 'H'; // CRASH");
    println!("   Rust: Immutable by default, explicit mut needed");
    let s = "Hello";
    // s[0] = 'h';  // COMPILE ERROR: cannot index and cannot mutate
    let mut s = String::from("Hello");
    unsafe {
        let bytes = s.as_bytes_mut();
        bytes[0] = b'h';
    }
    println!("   ✓ Requires mut and unsafe for byte mutation: {}", s);
    
    println!("\n9. VIRTUAL FUNCTION ON DELETED OBJECT");
    println!("   C++: delete obj; obj->method(); // CRASH");
    println!("   Rust: Ownership prevents use after drop");
    struct MyStruct;
    impl MyStruct {
        fn method(&self) { println!("method"); }
    }
    let obj = MyStruct;
    drop(obj);
    // obj.method();  // COMPILE ERROR: value used after move
    println!("   ✓ Compile error: can't call method on moved value");
    
    println!("\n10. ITERATOR INVALIDATION");
    println!("   C++: auto it = vec.begin(); vec.push_back(x); *it; // CRASH");
    println!("   Rust: Borrow checker prevents modification while iterating");
    let mut vec = vec![1, 2, 3];
    for val in &vec {
        // vec.push(4);  // COMPILE ERROR: cannot borrow as mutable
        println!("   Value: {}", val);
    }
    println!("   ✓ Cannot modify vec while borrowed by iterator");
    
    println!("\n=== SUMMARY ===");
    println!("Rust Prevention Mechanisms:");
    println!("• Ownership system (prevents use-after-free, double-free)");
    println!("• Borrow checker (prevents dangling references, data races)");
    println!("• Option<T> instead of null (prevents null dereference)");
    println!("• Bounds checking (prevents buffer overflow)");
    println!("• Initialization checking (prevents uninitialized access)");
    println!("• Immutable by default (prevents unwanted mutation)");
    println!("• Type system (lifetime tracking)");
    println!("\n✓ ALL SEGFAULT CATEGORIES PREVENTED!");
    println!("  Most at COMPILE TIME, rest with safe runtime checks");
}
