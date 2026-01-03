CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -g
SANITIZE = -fsanitize=address -fsanitize=undefined

CPP_TARGETS = 01_null_pointer 02_use_after_free 03_buffer_overflow 04_all_segfaults
RUST_TARGETS = 01_null_pointer_rs 02_use_after_free_rs 03_buffer_overflow_rs 04_rust_safety_rs

# Sanitized versions
CPP_SAFE_TARGETS = 01_null_pointer_safe 02_use_after_free_safe 03_buffer_overflow_safe

.PHONY: all cpp rust safe clean help run-rust

all: cpp rust

cpp: $(CPP_TARGETS)

rust: $(RUST_TARGETS)

safe: $(CPP_SAFE_TARGETS)

# Regular C++ compilation (will crash)
01_null_pointer: 01_null_pointer.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

02_use_after_free: 02_use_after_free.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

03_buffer_overflow: 03_buffer_overflow.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

04_all_segfaults: 04_all_segfaults.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Safe versions with sanitizers (will catch bugs)
01_null_pointer_safe: 01_null_pointer.cpp
	$(CXX) $(CXXFLAGS) $(SANITIZE) -o $@ $<

02_use_after_free_safe: 02_use_after_free.cpp
	$(CXX) $(CXXFLAGS) $(SANITIZE) -o $@ $<

03_buffer_overflow_safe: 03_buffer_overflow.cpp
	$(CXX) $(CXXFLAGS) $(SANITIZE) -o $@ $<

# Rust compilation
01_null_pointer_rs: 01_null_pointer.rs
	rustc -o $@ $<

02_use_after_free_rs: 02_use_after_free.rs
	rustc -o $@ $<

03_buffer_overflow_rs: 03_buffer_overflow.rs
	rustc -o $@ $<

04_rust_safety_rs: 04_rust_safety.rs
	rustc -o $@ $<

# Run Rust examples (all safe to run)
run-rust: rust
	@echo "=== Running Rust Safety Examples ===\n"
	@echo "1. Null Pointer Safety:"
	@./01_null_pointer_rs
	@echo "\n2. Use-After-Free Safety:"
	@./02_use_after_free_rs
	@echo "\n3. Buffer Overflow Safety:"
	@./03_buffer_overflow_rs
	@echo "\n4. Comprehensive Safety:"
	@./04_rust_safety_rs

help:
	@echo "Segfault Examples Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make all           - Build all C++ and Rust examples"
	@echo "  make cpp           - Build C++ examples (WILL CRASH if run)"
	@echo "  make rust          - Build Rust examples (safe to run)"
	@echo "  make safe          - Build C++ with AddressSanitizer"
	@echo "  make run-rust      - Run all Rust examples (demonstrates safety)"
	@echo "  make clean         - Remove all binaries"
	@echo ""
	@echo "C++ Examples (DANGEROUS - will crash):"
	@echo "  ./01_null_pointer             - Null pointer dereference"
	@echo "  ./02_use_after_free           - Use-after-free bugs"
	@echo "  ./03_buffer_overflow          - Buffer overflow bugs"
	@echo "  ./04_all_segfaults <1-10>     - All segfault types"
	@echo ""
	@echo "C++ with Sanitizer (catches bugs):"
	@echo "  ./01_null_pointer_safe        - Detects null dereference"
	@echo "  ./02_use_after_free_safe      - Detects use-after-free"
	@echo "  ./03_buffer_overflow_safe     - Detects buffer overflow"
	@echo ""
	@echo "Rust Examples (SAFE - no crashes):"
	@echo "  ./01_null_pointer_rs          - No null pointers in Rust"
	@echo "  ./02_use_after_free_rs        - Ownership prevents UAF"
	@echo "  ./03_buffer_overflow_rs       - Bounds checking"
	@echo "  ./04_rust_safety_rs           - Comprehensive safety demo"

clean:
	rm -f $(CPP_TARGETS) $(RUST_TARGETS) $(CPP_SAFE_TARGETS)
	rm -rf *.dSYM
