#ifndef TRANSPOSE_UTILS_HIP_H
#define TRANSPOSE_UTILS_HIP_H

#include <iostream>
#include <hip/hip_runtime.h>

// Error checking macro
#define HIP_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << hipGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CPU transpose for verification
void transposeCPU(const float *input, float *output, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            output[j * height + i] = input[i * width + j];
        }
    }
}

// Verify results
bool verifyTranspose(const float *gpu_result, const float *cpu_result, int size) {
    const float epsilon = 1e-5f;
    for (int i = 0; i < size; i++) {
        if (fabs(gpu_result[i] - cpu_result[i]) > epsilon) {
            std::cerr << "Mismatch at index " << i << ": GPU=" << gpu_result[i] 
                      << " CPU=" << cpu_result[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Calculate effective bandwidth
double calculateBandwidth(int width, int height, float time_ms) {
    double bytes = 2.0 * width * height * sizeof(float);
    double time_s = time_ms / 1000.0;
    return (bytes / time_s) / 1e9;  // GB/s
}

#endif // TRANSPOSE_UTILS_HIP_H
