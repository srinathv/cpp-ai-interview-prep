// Rust prevents use-after-free at compile time through ownership!

struct MyStruct {
    value: i32,
}

impl MyStruct {
    fn new(v: i32) -> Self {
        println!("  Constructor: value = {}", v);
        MyStruct { value: v }
    }
}

impl Drop for MyStruct {
    fn drop(&mut self) {
        println!("  Destructor: value was {}", self.value);
    }
}

fn main() {
    println!("=== Use-After-Free Safety in Rust ===\n");
    
    // Example 1: Rust prevents use-after-drop
    println!("Example 1: Ownership prevents use-after-free");
    {
        let ptr = Box::new(42);
        println!("  Allocated: *ptr = {}", ptr);
        drop(ptr);  // Explicitly free
        
        // COMPILE ERROR: Can't use after drop!
        // println!("  Value: {}", *ptr);  // ERROR: value used after move
    }
    println!("  Rust prevents use-after-free at compile time!");
    
    // Example 2: Can't return reference to local variable
    println!("\nExample 2: No dangling references");
    println!("  This would be a compile error:");
    println!("  fn bad() -> &i32 {{ ");
    println!("      let x = 42;");
    println!("      &x  // ERROR: returns reference to local variable");
    println!("  }}");
    
    // Example 3: Proper ownership transfer
    println!("\nExample 3: Safe ownership transfer");
    let s1 = String::from("Hello");
    println!("  s1 = {}", s1);
    
    let s2 = s1;  // Ownership moved
    println!("  s2 = {}", s2);
    
    // COMPILE ERROR: Can't use s1 after move!
    // println!("  s1 = {}", s1);  // ERROR: value used after move
    println!("  s1 is no longer accessible (moved to s2)");
    
    // Example 4: Borrowing prevents use-after-free
    println!("\nExample 4: Borrowing rules");
    let mut data = vec![1, 2, 3];
    
    {
        let r = &data[0];
        println!("  Reference to first element: {}", r);
        
        // COMPILE ERROR: Can't modify while borrowed!
        // data.push(4);  // ERROR: cannot borrow `data` as mutable
    }  // Borrow ends here
    
    data.push(4);  // Now safe to modify
    println!("  Added element after borrow ended");
    
    // Example 5: No iterator invalidation
    println!("\nExample 5: Iterator safety");
    let vec = vec![1, 2, 3, 4, 5];
    
    for val in &vec {
        println!("  Value: {}", val);
        // COMPILE ERROR: Can't modify vec while iterating!
        // vec.push(6);  // ERROR: cannot borrow as mutable
    }
    
    // Example 6: Rc for shared ownership
    println!("\nExample 6: Shared ownership with Rc");
    use std::rc::Rc;
    
    let data = Rc::new(42);
    println!("  Created Rc: {}", data);
    
    {
        let data2 = Rc::clone(&data);
        println!("  Cloned Rc: {}", data2);
        println!("  Strong count: {}", Rc::strong_count(&data));
    }  // data2 dropped, but data still valid
    
    println!("  Original still valid: {}", data);
    println!("  Strong count: {}", Rc::strong_count(&data));
    
    // Example 7: Objects with Drop trait
    println!("\nExample 7: RAII with Drop");
    {
        let obj = MyStruct::new(100);
        println!("  Using object: {}", obj.value);
    }  // Automatically dropped
    println!("  Object automatically cleaned up");
    
    // Example 8: Box prevents use-after-free
    println!("\nExample 8: Box ownership");
    let boxed = Box::new(MyStruct::new(200));
    println!("  Boxed value: {}", boxed.value);
    
    // Moving Box transfers ownership
    let boxed2 = boxed;
    println!("  Moved to boxed2: {}", boxed2.value);
    
    // COMPILE ERROR: Can't use boxed after move!
    // println!("{}", boxed.value);  // ERROR: value used after move
    
    println!("\n=== Key Takeaways ===");
    println!("1. Ownership system prevents use-after-free");
    println!("2. Can't return references to local variables");
    println!("3. Move semantics prevent using freed memory");
    println!("4. Borrowing rules prevent dangling references");
    println!("5. Iterator invalidation prevented by borrow checker");
    println!("6. Rc/Arc provide safe shared ownership");
    println!("7. Drop trait ensures cleanup");
    println!("\nResult: Use-after-free is prevented at COMPILE TIME!");
}
