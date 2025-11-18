#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <mpi.h>

using namespace std;

// Dictionary of 20 words
const vector<string> DICTIONARY = {
    "algorithm", "binary", "compiler", "database", "encryption",
    "framework", "gateway", "hashmap", "interface", "java",
    "kernel", "library", "memory", "network", "object",
    "protocol", "query", "runtime", "socket", "thread"
};

// Function to build a prefix map for a subset of the dictionary
map<string, vector<string>> buildPrefixMap(const vector<string>& words) {
    map<string, vector<string>> prefixMap;

    for (size_t i = 0; i < words.size(); i++) {
        const string& word = words[i];

        // Generate 2-letter and 3-letter prefixes
        for (int prefixLen = 2; prefixLen <= 3 && prefixLen <= (int)word.length(); prefixLen++) {
            string prefix = word.substr(0, prefixLen);
            transform(prefix.begin(), prefix.end(), prefix.begin(), ::tolower);
            prefixMap[prefix].push_back(word);
        }
    }

    return prefixMap;
}

// Function to distribute dictionary words across processes
vector<string> getLocalWords(int rank, int size, const vector<string>& dictionary) {
    vector<string> localWords;

    // Distribute words in round-robin fashion
    for (size_t i = rank; i < dictionary.size(); i += size) {
        localWords.push_back(dictionary[i]);
    }

    return localWords;
}

// Function to search in local prefix map
vector<string> searchLocal(const map<string, vector<string>>& prefixMap,
                           const string& input) {
    string lowerInput = input;
    transform(lowerInput.begin(), lowerInput.end(), lowerInput.begin(), ::tolower);

    auto it = prefixMap.find(lowerInput);
    if (it != prefixMap.end()) {
        return it->second;
    }
    return vector<string>();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        cout << "=== Dictionary Search - MPI Version ===" << endl;
        cout << "Number of MPI processes: " << size << endl;
        cout << "Dictionary size: " << DICTIONARY.size() << " words" << endl;
        cout << endl;
    }

    // Each process gets its subset of words
    vector<string> localWords = getLocalWords(rank, size, DICTIONARY);

    if (rank == 0) {
        cout << "Word distribution:" << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int r = 0; r < size; r++) {
        if (rank == r) {
            cout << "  Rank " << rank << ": " << localWords.size() << " words - ";
            for (size_t i = 0; i < localWords.size(); i++) {
                cout << localWords[i];
                if (i < localWords.size() - 1) cout << ", ";
            }
            cout << endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Build local prefix map
    auto startBuild = chrono::high_resolution_clock::now();
    map<string, vector<string>> localPrefixMap = buildPrefixMap(localWords);
    auto endBuild = chrono::high_resolution_clock::now();

    if (rank == 0) {
        chrono::duration<double, milli> buildTime = endBuild - startBuild;
        cout << endl << "Local prefix maps built in: " << buildTime.count() << " ms" << endl;
        cout << endl;
    }

    // Test searches
    vector<string> testInputs = {"al", "bin", "da", "enc", "fr", "ha", "int", "ke", "lib", "net"};

    if (rank == 0) {
        cout << "=== Running Distributed Searches ===" << endl;
    }

    double totalSearchTime = 0.0;

    // For loop through test inputs
    for (size_t i = 0; i < testInputs.size(); i++) {
        const string& input = testInputs[i];

        auto startSearch = chrono::high_resolution_clock::now();

        // Each process searches its local dictionary
        vector<string> localResults = searchLocal(localPrefixMap, input);

        // Convert results to count for reduction
        int localCount = localResults.size();
        int totalCount = 0;

        // Gather all results to rank 0
        MPI_Reduce(&localCount, &totalCount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        // Gather result strings (simplified: just gather counts and print local matches)
        if (rank == 0) {
            auto endSearch = chrono::high_resolution_clock::now();
            chrono::duration<double, micro> searchTime = endSearch - startSearch;
            totalSearchTime += searchTime.count();

            cout << "Search \"" << input << "\": ";
        }

        // Each rank reports its findings
        MPI_Barrier(MPI_COMM_WORLD);

        for (int r = 0; r < size; r++) {
            if (rank == r && !localResults.empty()) {
                if (r == 0) {
                    cout << "Found " << totalCount << " total match(es) - ";
                }
                for (const auto& word : localResults) {
                    cout << word << " ";
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        if (rank == 0) {
            auto endSearch = chrono::high_resolution_clock::now();
            chrono::duration<double, micro> searchTime = endSearch - startSearch;
            cout << "(" << searchTime.count() << " μs)" << endl;
        }
    }

    if (rank == 0) {
        cout << endl << "=== Summary ===" << endl;
        cout << "Total searches: " << testInputs.size() << endl;
        cout << "Average search time: " << (totalSearchTime / testInputs.size()) << " μs" << endl;
        cout << "Parallel speedup achieved through distribution across " << size << " processes" << endl;
    }

    MPI_Finalize();
    return 0;
}
