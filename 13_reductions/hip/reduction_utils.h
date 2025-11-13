#ifndef REDUCTION_UTILS_HIP_H
#define REDUCTION_UTILS_HIP_H
#include <iostream>
#include <hip/hip_runtime.h>
#define HIP_CHECK(call) do { hipError_t e = call; if (e != hipSuccess) { std::cerr << "HIP: " << hipGetErrorString(e) << std::endl; exit(1); } } while(0)
float reduceCPU(const float *data, int N) { float sum = 0.0f; for (int i = 0; i < N; i++) sum += data[i]; return sum; }
void initData(float *data, int N) { for (int i = 0; i < N; i++) data[i] = 1.0f; }
double calculateBandwidth(int N, float ms) { return (N * sizeof(float)) / (ms / 1000.0) / 1e9; }
#endif
