cmake_minimum_required(VERSION 3.18)
project(MaxIncreaseSkyline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options for different implementations
option(ENABLE_OPENMP "Enable OpenMP parallelization" ON)
option(ENABLE_CUDA "Enable CUDA GPU acceleration" OFF)
option(ENABLE_ROCM "Enable ROCm/HIP GPU acceleration" OFF)
option(BUILD_BENCHMARKS "Build benchmark suite" ON)

# Compiler flags for optimization
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native")

# Baseline implementation
add_executable(baseline src/baseline.cpp)

# CPU optimized version
add_executable(cpu_optimized src/cpu_optimized.cpp)
target_compile_options(cpu_optimized PRIVATE -O3 -march=native -mavx2 -mfma)

# OpenMP parallel version
if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        add_executable(openmp_parallel src/openmp_parallel.cpp)
        target_link_libraries(openmp_parallel PRIVATE OpenMP::OpenMP_CXX)
        message(STATUS "OpenMP enabled")
    else()
        message(WARNING "OpenMP not found, skipping OpenMP build")
    endif()
endif()

# CUDA GPU version
if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        add_executable(cuda_gpu src/cuda_gpu.cu)
        set_target_properties(cuda_gpu PROPERTIES 
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "60;70;75;80;86"
        )
        target_compile_options(cuda_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
        message(STATUS "CUDA enabled")
    else()
        message(WARNING "CUDA not found, skipping CUDA build")
    endif()
endif()

# ROCm/HIP GPU version
if(ENABLE_ROCM)
    if(NOT DEFINED ROCM_PATH)
        set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
    endif()
    
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
    find_package(hip)
    
    if(hip_FOUND)
        add_executable(rocm_gpu src/rocm_gpu.cpp)
        target_include_directories(rocm_gpu PRIVATE ${ROCM_PATH}/include)
        target_link_libraries(rocm_gpu PRIVATE hip::host)
        set_target_properties(rocm_gpu PROPERTIES 
            CXX_STANDARD 17
            COMPILE_FLAGS "-O3"
        )
        message(STATUS "ROCm/HIP enabled")
    else()
        message(WARNING "ROCm/HIP not found, skipping ROCm build")
    endif()
endif()

# Benchmark suite
if(BUILD_BENCHMARKS)
    add_executable(benchmark_skyline benchmarks/benchmark.cpp)
    target_compile_options(benchmark_skyline PRIVATE -O3 -march=native)
    
    if(OpenMP_CXX_FOUND)
        target_link_libraries(benchmark_skyline PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# Installation
install(TARGETS baseline cpu_optimized DESTINATION bin)

if(TARGET openmp_parallel)
    install(TARGETS openmp_parallel DESTINATION bin)
endif()

if(TARGET cuda_gpu)
    install(TARGETS cuda_gpu DESTINATION bin)
endif()

if(TARGET rocm_gpu)
    install(TARGETS rocm_gpu DESTINATION bin)
endif()

# Display build configuration
message(STATUS "Build configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  OpenMP: ${ENABLE_OPENMP}")
message(STATUS "  CUDA: ${ENABLE_CUDA}")
message(STATUS "  ROCm: ${ENABLE_ROCM}")
message(STATUS "  Benchmarks: ${BUILD_BENCHMARKS}")
