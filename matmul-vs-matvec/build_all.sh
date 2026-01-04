#!/bin/bash

# Build script for all implementations

echo "=========================================="
echo "Building Matrix-Matrix vs Matrix-Vector"
echo "Performance Comparison Suite"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Build CPU version
echo -e "\n${YELLOW}Building CPU single-threaded version...${NC}"
cd cpu
if make clean && make; then
    echo -e "${GREEN}✓ CPU build successful${NC}"
else
    echo -e "${RED}✗ CPU build failed${NC}"
fi
cd ..

# Build multithreaded version
echo -e "\n${YELLOW}Building OpenMP multi-threaded version...${NC}"
cd multithreaded
if make clean && make; then
    echo -e "${GREEN}✓ OpenMP build successful${NC}"
else
    echo -e "${RED}✗ OpenMP build failed${NC}"
    echo -e "${YELLOW}Note: Requires OpenMP support (install libomp on macOS)${NC}"
fi
cd ..

# Build CUDA version
echo -e "\n${YELLOW}Building CUDA version...${NC}"
cd cuda
if command -v nvcc &> /dev/null; then
    if make clean && make; then
        echo -e "${GREEN}✓ CUDA build successful${NC}"
    else
        echo -e "${RED}✗ CUDA build failed${NC}"
    fi
else
    echo -e "${YELLOW}⊘ CUDA compiler (nvcc) not found - skipping CUDA build${NC}"
fi
cd ..

echo -e "\n${GREEN}=========================================="
echo "Build process complete!"
echo "==========================================${NC}"
echo ""
echo "To run the benchmarks:"
echo "  CPU:           cd cpu && ./comparison"
echo "  Multi-thread:  cd multithreaded && ./comparison_omp"
echo "  CUDA:          cd cuda && ./comparison"
