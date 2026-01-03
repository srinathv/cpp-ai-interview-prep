CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra

CPP_TARGETS = stack_overflow heap_example stack_vs_heap memory_lifecycle comprehensive_usage
RUST_TARGETS = stack_overflow_rs heap_example_rs stack_vs_heap_rs memory_lifecycle_rs comprehensive_usage_rs

.PHONY: all cpp rust clean run-cpp run-rust run-lifecycle run-comprehensive help

all: cpp rust

cpp: $(CPP_TARGETS)

rust: $(RUST_TARGETS)

# C++ compilation
stack_overflow: stack_overflow.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

heap_example: heap_example.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

stack_vs_heap: stack_vs_heap.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

memory_lifecycle: memory_lifecycle.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

comprehensive_usage: comprehensive_usage.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Rust compilation
stack_overflow_rs: stack_overflow.rs
	rustc -o $@ $<

heap_example_rs: heap_example.rs
	rustc -o $@ $<

stack_vs_heap_rs: stack_vs_heap.rs
	rustc -o $@ $<

memory_lifecycle_rs: memory_lifecycle.rs
	rustc -o $@ $<

comprehensive_usage_rs: comprehensive_usage.rs
	rustc -o $@ $<

# Run targets
run-cpp: cpp
	@echo "=== Running C++ heap example ==="
	./heap_example
	@echo "\n=== Running C++ stack vs heap ==="
	./stack_vs_heap

run-rust: rust
	@echo "=== Running Rust heap example ==="
	./heap_example_rs
	@echo "\n=== Running Rust stack vs heap ==="
	./stack_vs_heap_rs

run-lifecycle: memory_lifecycle memory_lifecycle_rs
	@echo "=== C++ Memory Lifecycle ==="
	./memory_lifecycle
	@echo "\n\n=== Rust Memory Lifecycle ==="
	./memory_lifecycle_rs

run-comprehensive: comprehensive_usage comprehensive_usage_rs
	@echo "=== C++ Comprehensive Usage ==="
	./comprehensive_usage
	@echo "\n\n=== Rust Comprehensive Usage ==="
	./comprehensive_usage_rs

help:
	@echo "Available targets:"
	@echo "  make all              - Build all C++ and Rust examples"
	@echo "  make cpp              - Build all C++ examples"
	@echo "  make rust             - Build all Rust examples"
	@echo "  make run-cpp          - Run basic C++ examples"
	@echo "  make run-rust         - Run basic Rust examples"
	@echo "  make run-lifecycle    - Run memory lifecycle examples"
	@echo "  make run-comprehensive - Run comprehensive usage examples"
	@echo "  make clean            - Remove all binaries"

clean:
	rm -f $(CPP_TARGETS) $(RUST_TARGETS)
	rm -rf *.dSYM
