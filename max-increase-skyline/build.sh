#!/bin/bash

# Build script for Max Increase Skyline implementations

set -e

echo "=========================================="
echo "  Building Max Increase Skyline"
echo "=========================================="
echo ""

# Create build directory
mkdir -p build
cd build

# Default options
ENABLE_OPENMP=ON
ENABLE_CUDA=OFF
ENABLE_ROCM=OFF
BUILD_TYPE=Release

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            ENABLE_CUDA=ON
            shift
            ;;
        --rocm)
            ENABLE_ROCM=ON
            shift
            ;;
        --no-openmp)
            ENABLE_OPENMP=OFF
            shift
            ;;
        --debug)
            BUILD_TYPE=Debug
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--cuda] [--rocm] [--no-openmp] [--debug]"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Build Type: $BUILD_TYPE"
echo "  OpenMP: $ENABLE_OPENMP"
echo "  CUDA: $ENABLE_CUDA"
echo "  ROCm: $ENABLE_ROCM"
echo ""

# Run CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DENABLE_OPENMP=$ENABLE_OPENMP \
    -DENABLE_CUDA=$ENABLE_CUDA \
    -DENABLE_ROCM=$ENABLE_ROCM

# Build
echo ""
echo "Building..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "Build complete!"
echo ""
echo "Executables:"
ls -lh baseline cpu_optimized openmp_parallel benchmark_skyline 2>/dev/null || true
ls -lh cuda_gpu rocm_gpu 2>/dev/null || true

echo ""
echo "To run benchmarks:"
echo "  ./benchmark_skyline"
echo ""
echo "To run individual implementations:"
echo "  ./baseline"
echo "  ./cpu_optimized"
echo "  ./openmp_parallel"
if [ "$ENABLE_CUDA" = "ON" ]; then
    echo "  ./cuda_gpu"
fi
if [ "$ENABLE_ROCM" = "ON" ]; then
    echo "  ./rocm_gpu"
fi
