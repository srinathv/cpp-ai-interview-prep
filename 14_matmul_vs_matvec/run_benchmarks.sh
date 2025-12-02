#!/bin/bash

# Script to run all benchmarks and collect results

echo "=========================================="
echo "Matrix-Matrix vs Matrix-Vector"
echo "Performance Benchmark Suite"
echo "=========================================="

OUTPUT_DIR="results"
mkdir -p $OUTPUT_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Run CPU benchmark
if [ -f "cpu/comparison" ]; then
    echo -e "\n=== Running CPU Single-Threaded Benchmark ==="
    ./cpu/comparison | tee $OUTPUT_DIR/cpu_${TIMESTAMP}.txt
else
    echo "CPU benchmark not built. Run ./build_all.sh first."
fi

# Run OpenMP benchmark
if [ -f "multithreaded/comparison_omp" ]; then
    echo -e "\n=== Running OpenMP Multi-Threaded Benchmark ==="
    ./multithreaded/comparison_omp | tee $OUTPUT_DIR/omp_${TIMESTAMP}.txt
else
    echo "OpenMP benchmark not built. Run ./build_all.sh first."
fi

# Run CUDA benchmark
if [ -f "cuda/comparison" ]; then
    echo -e "\n=== Running CUDA GPU Benchmark ==="
    ./cuda/comparison | tee $OUTPUT_DIR/cuda_${TIMESTAMP}.txt
else
    echo "CUDA benchmark not built (requires CUDA toolkit)."
fi

echo -e "\n=========================================="
echo "Benchmark results saved to $OUTPUT_DIR/"
echo "=========================================="
