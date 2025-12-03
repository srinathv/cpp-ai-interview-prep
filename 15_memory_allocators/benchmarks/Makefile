CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -pthread

# Targets
TARGETS = basic_bench multithreaded_bench

all: $(TARGETS)

basic_bench: basic_bench.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

multithreaded_bench: multithreaded_bench.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Build with different allocators
basic_tcmalloc: basic_bench.cpp
	$(CXX) $(CXXFLAGS) -DUSE_TCMALLOC -o basic_bench_tcmalloc $< -ltcmalloc

basic_jemalloc: basic_bench.cpp
	$(CXX) $(CXXFLAGS) -DUSE_JEMALLOC -o basic_bench_jemalloc $< -ljemalloc

basic_mimalloc: basic_bench.cpp
	$(CXX) $(CXXFLAGS) -DUSE_MIMALLOC -o basic_bench_mimalloc $< -lmimalloc

multi_tcmalloc: multithreaded_bench.cpp
	$(CXX) $(CXXFLAGS) -DUSE_TCMALLOC -o multithreaded_bench_tcmalloc $< -ltcmalloc

multi_jemalloc: multithreaded_bench.cpp
	$(CXX) $(CXXFLAGS) -DUSE_JEMALLOC -o multithreaded_bench_jemalloc $< -ljemalloc

multi_mimalloc: multithreaded_bench.cpp
	$(CXX) $(CXXFLAGS) -DUSE_MIMALLOC -o multithreaded_bench_mimalloc $< -lmimalloc

# Run tests
test: all
	@echo "=== Testing with system allocator ==="
	./basic_bench
	@echo ""
	./multithreaded_bench

clean:
	rm -f $(TARGETS) *_tcmalloc *_jemalloc *_mimalloc

.PHONY: all test clean
