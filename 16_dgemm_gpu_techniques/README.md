# DGEMM GPU Techniques and Assembly Enhancements

This directory covers unique C++ techniques and assembly-level optimizations used in DGEMM implementations on AMD GPUs, particularly relevant to rocBLAS and Tensile.

## Contents

1. cpp_techniques.cpp - Template metaprogramming, CRTP, constexpr patterns
2. assembly_enhancements.md - MFMA instructions, software pipelining, register optimization
3. mfma_examples.cpp - Practical MFMA intrinsic usage

## Interview Focus

For numerical/math library interviews: Emphasize the mathematical foundations - how MFMA computes outer products, why tile sizes affect arithmetic intensity, and how Tensile ML approach optimizes for specific problem dimensions.
