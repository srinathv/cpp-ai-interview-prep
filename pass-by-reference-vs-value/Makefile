CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

TARGETS = basic_examples performance_comparison common_patterns

all: $(TARGETS)

basic_examples: basic_examples.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

performance_comparison: performance_comparison.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

common_patterns: common_patterns.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)

run_all: all
	@echo "=== Running basic_examples ==="
	./basic_examples
	@echo ""
	@echo "=== Running performance_comparison ==="
	./performance_comparison
	@echo ""
	@echo "=== Running common_patterns ==="
	./common_patterns

.PHONY: all clean run_all
