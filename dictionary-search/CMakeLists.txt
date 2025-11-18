cmake_minimum_required(VERSION 3.18)
project(DictionarySearch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Serial version
add_subdirectory(serial)

# Pthreads version
add_subdirectory(pthreads)

# MPI version (optional)
find_package(MPI)
if(MPI_FOUND)
    add_subdirectory(mpi)
    message(STATUS "MPI found - building MPI version")
else()
    message(STATUS "MPI not found - skipping MPI version")
endif()

# CUDA version (optional)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_subdirectory(cuda)
    message(STATUS "CUDA found - building CUDA version")
else()
    message(STATUS "CUDA not found - skipping CUDA version")
endif()

# HIP version (optional)
if(DEFINED ENV{ROCM_PATH} OR EXISTS "/opt/rocm")
    add_subdirectory(hip)
    message(STATUS "ROCm found - building HIP version")
else()
    message(STATUS "ROCm not found - skipping HIP version")
endif()
