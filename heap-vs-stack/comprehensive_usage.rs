use std::rc::Rc;

fn print_addr<T>(label: &str, ptr: &T) {
    println!("  {:40}: {:p}", label, ptr);
}

struct DataObject {
    id: i32,
}

impl DataObject {
    fn new(id: i32) -> Self {
        println!("  Constructor called for object {}", id);
        DataObject { id }
    }
    
    fn get_id(&self) -> i32 {
        self.id
    }
}

impl Drop for DataObject {
    fn drop(&mut self) {
        println!("  Destructor called for object {}", self.id);
    }
}

// Example 1: Stack allocation - use and cleanup
fn stack_allocation_example() {
    println!("\n=== 1. Stack Allocation - Allocation, Use, Deallocation ===");
    
    println!("Before allocation:");
    
    let x = 42;
    println!("After allocation:");
    print_addr("Variable x (stack)", &x);
    
    println!("Using variable:");
    println!("  x = {}", x);
    let x = 100;  // Shadowing
    println!("  x after shadowing = {}", x);
    
    let obj = DataObject::new(1);
    print_addr("Object on stack", &obj);
    println!("Using object: id = {}", obj.get_id());
    
    println!("End of function - automatic cleanup...");
}

// Example 2: Heap allocation - Box
fn heap_allocation_example() {
    println!("\n=== 2. Heap Allocation - Box (Automatic Management) ===");
    
    println!("Before allocation:");
    
    let ptr = Box::new(42);
    println!("After allocation:");
    print_addr("Box 'ptr' itself (stack)", &ptr);
    print_addr("Data pointed to (heap)", &*ptr);
    
    println!("Using heap variable:");
    println!("  *ptr = {}", *ptr);
    let mut ptr = ptr;
    *ptr = 100;
    println!("  *ptr after modification = {}", *ptr);
    
    let obj_ptr = Box::new(DataObject::new(2));
    print_addr("Box pointer (stack)", &obj_ptr);
    print_addr("Object on heap", &*obj_ptr);
    println!("Using heap object: id = {}", obj_ptr.get_id());
    
    println!("End of function - automatic cleanup...");
}

// Example 3: Ownership transfer
fn ownership_transfer_example() {
    println!("\n=== 3. Ownership Transfer ===");
    
    println!("Creating owned value:");
    let s1 = String::from("Hello");
    print_addr("String s1", &s1);
    println!("  s1 = {}", s1);
    
    println!("\nTransferring ownership:");
    let s2 = s1;
    print_addr("String s2 (same heap data)", &s2);
    println!("  s2 = {}", s2);
    println!("  s1 is no longer valid (moved)");
    // println!("{}", s1);  // COMPILE ERROR!
    
    println!("\nCloning to keep both:");
    let s3 = s2.clone();
    print_addr("String s3 (new heap data)", &s3);
    println!("  s2 = {}, s3 = {}", s2, s3);
    println!("  Both s2 and s3 are valid");
}

// Example 4: Borrowing - immutable and mutable
fn borrowing_example() {
    println!("\n=== 4. Borrowing - References ===");
    
    let mut data = vec![1, 2, 3, 4, 5];
    print_addr("Vector (stack)", &data);
    print_addr("Vector data (heap)", unsafe { &*data.as_ptr() });
    
    println!("\nImmutable borrow:");
    let r1 = &data;
    let r2 = &data;
    println!("  r1[0] = {}, r2[0] = {}", r1[0], r2[0]);
    println!("  Original still accessible: data[0] = {}", data[0]);
    
    println!("\nMutable borrow:");
    let r = &mut data;
    r[0] = 99;
    println!("  Modified via mutable reference: r[0] = {}", r[0]);
    // println!("{}", data);  // COMPILE ERROR while mutable borrow exists!
    
    println!("\nAfter mutable borrow ends:");
    println!("  data[0] = {}", data[0]);
}

// Example 5: Function call stack
fn function_c() {
    println!("  In function_c");
    let c = 3;
    print_addr("Variable c in function_c", &c);
    println!("  function_c returning...");
}

fn function_b() {
    println!("  In function_b");
    let b = 2;
    print_addr("Variable b in function_b", &b);
    function_c();
    println!("  Back in function_b, b still valid: {}", b);
    println!("  function_b returning...");
}

fn function_a() {
    println!("  In function_a");
    let a = 1;
    print_addr("Variable a in function_a", &a);
    function_b();
    println!("  Back in function_a, a still valid: {}", a);
    println!("  function_a returning...");
}

fn function_call_example() {
    println!("\n=== 5. Function Call Stack ===");
    println!("Calling nested functions (watch stack addresses):");
    function_a();
    println!("All functions returned, stack unwound");
}

// Example 6: Array/Vector allocation
fn array_allocation_example() {
    println!("\n=== 6. Array/Collection Allocation ===");
    
    println!("Stack array:");
    let stack_array = [1, 2, 3, 4, 5];
    print_addr("stack_array (stack)", &stack_array);
    println!("  Using: stack_array[0] = {}", stack_array[0]);
    let mut stack_array = stack_array;
    stack_array[0] = 99;
    println!("  After modification: stack_array[0] = {}", stack_array[0]);
    
    println!("\nVec (heap-allocated):");
    let mut vec = vec![1, 2, 3, 4, 5];
    print_addr("Vec object (stack)", &vec);
    print_addr("Vec data (heap)", unsafe { &*vec.as_ptr() });
    println!("  Using: vec[0] = {}", vec[0]);
    vec[0] = 99;
    println!("  After modification: vec[0] = {}", vec[0]);
    
    println!("\nBoxed array (heap):");
    let mut boxed = Box::new([1, 2, 3, 4, 5]);
    print_addr("Box (stack)", &boxed);
    print_addr("Array data (heap)", &boxed[0]);
    println!("  Using: boxed[0] = {}", boxed[0]);
    boxed[0] = 99;
    println!("  After modification: boxed[0] = {}", boxed[0]);
    
    println!("\nAll cleaned up automatically at scope end");
}

// Example 7: Lifetime tracking
fn lifetime_example() {
    println!("\n=== 7. Lifetime Tracking ===");
    
    let outer = String::from("Outer");
    print_addr("Outer string", &outer);
    println!("  outer = {}", outer);
    
    {
        let inner = String::from("Inner");
        print_addr("Inner string", &inner);
        println!("  In inner scope: outer = {}, inner = {}", outer, inner);
    }  // inner dropped here
    
    println!("  After inner scope: outer = {}", outer);
    println!("  inner is no longer accessible");
}

// Example 8: Reference counting
fn shared_ownership_example() {
    println!("\n=== 8. Shared Ownership with Rc ===");
    
    println!("Creating Rc:");
    let data = Rc::new(DataObject::new(10));
    print_addr("Rc (stack)", &data);
    println!("  Strong count: {}", Rc::strong_count(&data));
    
    {
        println!("\nCloning Rc (shares ownership):");
        let data2 = Rc::clone(&data);
        print_addr("Cloned Rc (stack)", &data2);
        println!("  Strong count: {}", Rc::strong_count(&data));
        println!("  Both point to same heap data");
        println!("  data id: {}, data2 id: {}", data.get_id(), data2.get_id());
    }  // data2 dropped, count decreases
    
    println!("\nAfter data2 scope:");
    println!("  Strong count: {}", Rc::strong_count(&data));
    println!("  data still valid: id = {}", data.get_id());
    
    println!("\nEnd of function, last Rc dropped...");
}

// Example 9: Copy vs Move semantics
fn copy_vs_move_example() {
    println!("\n=== 9. Copy vs Move Semantics ===");
    
    println!("Copy trait (stack primitives):");
    let x = 42;
    print_addr("x", &x);
    let y = x;  // Copied
    print_addr("y", &y);
    println!("  x = {}, y = {} (both valid)", x, y);
    
    println!("\nMove semantics (heap types):");
    let s1 = String::from("Hello");
    print_addr("s1", &s1);
    let s2 = s1;  // Moved
    print_addr("s2", &s2);
    println!("  s2 = {}", s2);
    println!("  s1 is no longer accessible (moved)");
    // println!("{}", s1);  // COMPILE ERROR!
}

// Example 10: Scope-based cleanup
fn scope_cleanup_example() {
    println!("\n=== 10. Scope-Based Resource Cleanup ===");
    
    println!("Outer scope:");
    let obj1 = DataObject::new(100);
    println!("  Created obj1");
    
    {
        println!("\n  Inner scope:");
        let _obj2 = DataObject::new(200);
        println!("    Created obj2");
        println!("    Both obj1 and obj2 valid");
        println!("  Inner scope ending...");
    }  // obj2 dropped here
    
    println!("\nBack in outer scope:");
    println!("  obj1 still valid: id = {}", obj1.get_id());
    println!("Outer scope ending...");
}  // obj1 dropped here

fn main() {
    println!("=== Comprehensive Memory Usage Patterns in Rust ===");
    println!("Demonstrating allocation -> use -> deallocation for each pattern\n");
    
    stack_allocation_example();
    heap_allocation_example();
    ownership_transfer_example();
    borrowing_example();
    function_call_example();
    array_allocation_example();
    lifetime_example();
    shared_ownership_example();
    copy_vs_move_example();
    scope_cleanup_example();
    
    println!("\n=== Summary ===");
    println!("Stack: Allocate -> Use -> Auto-cleanup (LIFO, Copy semantics)");
    println!("Heap:  Allocate -> Use -> Auto-cleanup via ownership/RAII");
    println!("       Rust enforces memory safety at compile time!");
    println!("       No use-after-free, no double-free, no data races");
}
