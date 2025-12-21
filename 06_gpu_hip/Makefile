# Makefile for HIP examples
# Requires HIP/ROCm to be installed

HIPCC = hipcc
CXXFLAGS = -O3 -std=c++17

# For AMD GPUs, specify architecture (e.g., gfx906, gfx908, gfx90a)
# AMD_ARCH = --amdgpu-target=gfx908

# For NVIDIA GPUs with HIP
# NVIDIA_ARCH = --cuda-gpu-arch=sm_70

TARGETS = vector_add matrix_multiply memory_patterns

all: $(TARGETS)

vector_add: vector_add.hip.cpp
	$(HIPCC) $(CXXFLAGS) $< -o $@

matrix_multiply: matrix_multiply.hip.cpp
	$(HIPCC) $(CXXFLAGS) $< -o $@

memory_patterns: memory_patterns.hip.cpp
	$(HIPCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean
