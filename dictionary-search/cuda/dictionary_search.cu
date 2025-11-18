#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

using namespace std;

// Dictionary of 20 words
const char* DICTIONARY[] = {
    "algorithm", "binary", "compiler", "database", "encryption",
    "framework", "gateway", "hashmap", "interface", "java",
    "kernel", "library", "memory", "network", "object",
    "protocol", "query", "runtime", "socket", "thread"
};

const int DICT_SIZE = 20;
const int MAX_WORD_LEN = 32;
const int MAX_PREFIX_LEN = 3;

// CUDA kernel to search for matching words
__global__ void searchKernel(const char* dictionary, int dictSize, int maxWordLen,
                             const char* prefix, int prefixLen, int* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= dictSize) return;

    // Get pointer to this word in dictionary
    const char* word = dictionary + (idx * maxWordLen);

    // Check if word length is sufficient
    int wordLen = 0;
    while (word[wordLen] != '\0' && wordLen < maxWordLen) {
        wordLen++;
    }

    if (wordLen < prefixLen) {
        results[idx] = 0;
        return;
    }

    // Compare prefix (case-insensitive)
    bool match = true;
    for (int i = 0; i < prefixLen; i++) {
        char wordChar = word[i];
        char prefixChar = prefix[i];

        // Convert to lowercase for comparison
        if (wordChar >= 'A' && wordChar <= 'Z') {
            wordChar = wordChar - 'A' + 'a';
        }
        if (prefixChar >= 'A' && prefixChar <= 'Z') {
            prefixChar = prefixChar - 'A' + 'a';
        }

        if (wordChar != prefixChar) {
            match = false;
            break;
        }
    }

    results[idx] = match ? 1 : 0;
}

// Helper function to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                 << cudaGetErrorString(err) << endl; \
            exit(1); \
        } \
    } while(0)

// Function to prepare dictionary for GPU
char* prepareDictionaryForGPU(const char** dict, int size, int maxLen) {
    char* flatDict = new char[size * maxLen];

    // Initialize with zeros
    for (int i = 0; i < size * maxLen; i++) {
        flatDict[i] = '\0';
    }

    // Copy words into flat array
    for (int i = 0; i < size; i++) {
        const char* word = dict[i];
        int j = 0;
        while (word[j] != '\0' && j < maxLen - 1) {
            flatDict[i * maxLen + j] = word[j];
            j++;
        }
        flatDict[i * maxLen + j] = '\0';
    }

    return flatDict;
}

// Function to perform GPU-accelerated search
vector<string> searchOnGPU(const char** dictionary, int dictSize, const string& prefix) {
    vector<string> results;

    // Prepare dictionary for GPU
    char* h_dictionary = prepareDictionaryForGPU(dictionary, dictSize, MAX_WORD_LEN);

    // Prepare prefix
    char h_prefix[MAX_PREFIX_LEN + 1];
    string lowerPrefix = prefix;
    transform(lowerPrefix.begin(), lowerPrefix.end(), lowerPrefix.begin(), ::tolower);

    for (size_t i = 0; i < lowerPrefix.length() && i < MAX_PREFIX_LEN; i++) {
        h_prefix[i] = lowerPrefix[i];
    }
    h_prefix[lowerPrefix.length()] = '\0';

    // Allocate device memory
    char* d_dictionary;
    char* d_prefix;
    int* d_results;

    CUDA_CHECK(cudaMalloc(&d_dictionary, dictSize * MAX_WORD_LEN * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_prefix, (MAX_PREFIX_LEN + 1) * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_results, dictSize * sizeof(int)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_dictionary, h_dictionary, dictSize * MAX_WORD_LEN * sizeof(char),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prefix, h_prefix, (MAX_PREFIX_LEN + 1) * sizeof(char),
                         cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (dictSize + threadsPerBlock - 1) / threadsPerBlock;

    searchKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_dictionary, dictSize, MAX_WORD_LEN, d_prefix, lowerPrefix.length(), d_results
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    int* h_results = new int[dictSize];
    CUDA_CHECK(cudaMemcpy(h_results, d_results, dictSize * sizeof(int), cudaMemcpyDeviceToHost));

    // Collect matching words
    for (int i = 0; i < dictSize; i++) {
        if (h_results[i] == 1) {
            results.push_back(dictionary[i]);
        }
    }

    // Cleanup
    delete[] h_dictionary;
    delete[] h_results;
    CUDA_CHECK(cudaFree(d_dictionary));
    CUDA_CHECK(cudaFree(d_prefix));
    CUDA_CHECK(cudaFree(d_results));

    return results;
}

int main() {
    cout << "=== Dictionary Search - CUDA Version ===" << endl;

    // Check CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        cerr << "No CUDA devices found!" << endl;
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cout << "Using GPU: " << prop.name << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
    cout << "Dictionary size: " << DICT_SIZE << " words" << endl;
    cout << endl;

    // Display dictionary
    cout << "Dictionary words:" << endl;
    for (int i = 0; i < DICT_SIZE; i++) {
        cout << "  " << (i + 1) << ". " << DICTIONARY[i] << endl;
    }
    cout << endl;

    // Test searches
    vector<string> testInputs = {"al", "bin", "da", "enc", "fr", "ha", "int", "ke", "lib", "net", "pro", "que", "run", "so", "th"};

    cout << "=== Running GPU-Accelerated Searches ===" << endl;

    double totalSearchTime = 0.0;
    int totalResults = 0;

    // Use for loop for searches
    for (size_t i = 0; i < testInputs.size(); i++) {
        const string& input = testInputs[i];

        auto startSearch = chrono::high_resolution_clock::now();
        vector<string> results = searchOnGPU(DICTIONARY, DICT_SIZE, input);
        auto endSearch = chrono::high_resolution_clock::now();

        chrono::duration<double, micro> searchTime = endSearch - startSearch;
        totalSearchTime += searchTime.count();
        totalResults += results.size();

        cout << "Search \"" << input << "\": ";
        if (results.empty()) {
            cout << "No matches found";
        } else {
            cout << results.size() << " match(es) - ";
            for (size_t j = 0; j < results.size(); j++) {
                cout << results[j];
                if (j < results.size() - 1) cout << ", ";
            }
        }
        cout << " (" << searchTime.count() << " μs)" << endl;
    }

    cout << endl;
    cout << "=== Summary ===" << endl;
    cout << "Total searches: " << testInputs.size() << endl;
    cout << "Total results found: " << totalResults << endl;
    cout << "Average search time: " << (totalSearchTime / testInputs.size()) << " μs" << endl;
    cout << "GPU: " << prop.name << endl;
    cout << "Note: GPU search includes data transfer overhead" << endl;

    return 0;
}
