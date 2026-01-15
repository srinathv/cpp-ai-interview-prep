CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

# All targets
TARGETS = basic_examples \
          performance_comparison \
          common_patterns \
          move_semantics \
          perfect_forwarding \
          lambda_captures \
          smart_pointers \
          rvo_and_optimization \
          const_correctness

all: $(TARGETS)

# Basic examples
basic_examples: basic_examples.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

performance_comparison: performance_comparison.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

common_patterns: common_patterns.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Modern C++ examples
move_semantics: move_semantics.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

perfect_forwarding: perfect_forwarding.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

lambda_captures: lambda_captures.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

smart_pointers: smart_pointers.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

rvo_and_optimization: rvo_and_optimization.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

const_correctness: const_correctness.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Phony targets
.PHONY: all clean run_all run_basic run_advanced

clean:
	rm -f $(TARGETS)

run_basic: basic_examples performance_comparison common_patterns
	@echo "=== Running Basic Examples ==="
	./basic_examples
	@echo ""
	@echo "=== Running Performance Comparison ==="
	./performance_comparison
	@echo ""
	@echo "=== Running Common Patterns ==="
	./common_patterns

run_advanced: move_semantics perfect_forwarding lambda_captures smart_pointers rvo_and_optimization const_correctness
	@echo "=== Running Move Semantics ==="
	./move_semantics
	@echo ""
	@echo "=== Running Perfect Forwarding ==="
	./perfect_forwarding
	@echo ""
	@echo "=== Running Lambda Captures ==="
	./lambda_captures
	@echo ""
	@echo "=== Running Smart Pointers ==="
	./smart_pointers
	@echo ""
	@echo "=== Running RVO and Optimization ==="
	./rvo_and_optimization
	@echo ""
	@echo "=== Running Const Correctness ==="
	./const_correctness

run_all: all run_basic run_advanced
	@echo ""
	@echo "=== All examples completed ==="
