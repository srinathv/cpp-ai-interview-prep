#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <pthread.h>
#include <mutex>

using namespace std;

// Dictionary of 20 words
const vector<string> DICTIONARY = {
    "algorithm", "binary", "compiler", "database", "encryption",
    "framework", "gateway", "hashmap", "interface", "java",
    "kernel", "library", "memory", "network", "object",
    "protocol", "query", "runtime", "socket", "thread"
};

// Thread data structure
struct ThreadData {
    int threadId;
    int numThreads;
    vector<string> dictionary;
    string searchPrefix;
    vector<string> results;
};

// Mutex for thread-safe output
mutex outputMutex;

// Function to build prefix map for a subset of words
map<string, vector<string>> buildPrefixMap(const vector<string>& words) {
    map<string, vector<string>> prefixMap;

    for (size_t i = 0; i < words.size(); i++) {
        const string& word = words[i];

        for (int prefixLen = 2; prefixLen <= 3 && prefixLen <= (int)word.length(); prefixLen++) {
            string prefix = word.substr(0, prefixLen);
            transform(prefix.begin(), prefix.end(), prefix.begin(), ::tolower);
            prefixMap[prefix].push_back(word);
        }
    }

    return prefixMap;
}

// Thread worker function for building prefix maps
void* buildPrefixMapThread(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);

    // Get subset of words for this thread
    vector<string> localWords;
    for (size_t i = data->threadId; i < data->dictionary.size(); i += data->numThreads) {
        localWords.push_back(data->dictionary[i]);
    }

    {
        lock_guard<mutex> lock(outputMutex);
        cout << "  Thread " << data->threadId << " processing " << localWords.size() << " words: ";
        for (size_t i = 0; i < localWords.size(); i++) {
            cout << localWords[i];
            if (i < localWords.size() - 1) cout << ", ";
        }
        cout << endl;
    }

    pthread_exit(nullptr);
}

// Thread worker function for searching
void* searchThread(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);

    // Get subset of words for this thread
    vector<string> localWords;
    for (size_t i = data->threadId; i < data->dictionary.size(); i += data->numThreads) {
        localWords.push_back(data->dictionary[i]);
    }

    // Build local prefix map
    map<string, vector<string>> localPrefixMap = buildPrefixMap(localWords);

    // Search for the prefix
    string lowerPrefix = data->searchPrefix;
    transform(lowerPrefix.begin(), lowerPrefix.end(), lowerPrefix.begin(), ::tolower);

    auto it = localPrefixMap.find(lowerPrefix);
    if (it != localPrefixMap.end()) {
        data->results = it->second;
    }

    pthread_exit(nullptr);
}

int main() {
    const int NUM_THREADS = 4;

    cout << "=== Dictionary Search - Pthreads Version ===" << endl;
    cout << "Number of threads: " << NUM_THREADS << endl;
    cout << "Dictionary size: " << DICTIONARY.size() << " words" << endl;
    cout << endl;

    // Display dictionary
    cout << "Dictionary words:" << endl;
    for (size_t i = 0; i < DICTIONARY.size(); i++) {
        cout << "  " << (i + 1) << ". " << DICTIONARY[i] << endl;
    }
    cout << endl;

    // Demonstrate thread distribution
    cout << "=== Thread Work Distribution ===" << endl;

    pthread_t threads[NUM_THREADS];
    ThreadData threadData[NUM_THREADS];

    // Create threads to show work distribution
    for (int i = 0; i < NUM_THREADS; i++) {
        threadData[i].threadId = i;
        threadData[i].numThreads = NUM_THREADS;
        threadData[i].dictionary = DICTIONARY;

        int rc = pthread_create(&threads[i], nullptr, buildPrefixMapThread, &threadData[i]);
        if (rc) {
            cerr << "Error creating thread " << i << endl;
            return 1;
        }
    }

    // Wait for all threads to complete
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], nullptr);
    }

    cout << endl;

    // Test searches with multiple threads
    vector<string> testInputs = {"al", "bin", "da", "enc", "fr", "ha", "int", "ke", "lib", "net", "pro", "que", "run", "so", "th"};

    cout << "=== Running Parallel Searches ===" << endl;

    double totalSearchTime = 0.0;
    int totalResults = 0;

    // Use for loop for searches
    for (size_t searchIdx = 0; searchIdx < testInputs.size(); searchIdx++) {
        const string& input = testInputs[searchIdx];

        auto startSearch = chrono::high_resolution_clock::now();

        // Create threads for parallel search
        pthread_t searchThreads[NUM_THREADS];
        ThreadData searchData[NUM_THREADS];

        for (int i = 0; i < NUM_THREADS; i++) {
            searchData[i].threadId = i;
            searchData[i].numThreads = NUM_THREADS;
            searchData[i].dictionary = DICTIONARY;
            searchData[i].searchPrefix = input;

            int rc = pthread_create(&searchThreads[i], nullptr, searchThread, &searchData[i]);
            if (rc) {
                cerr << "Error creating search thread " << i << endl;
                return 1;
            }
        }

        // Wait for all search threads to complete
        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_join(searchThreads[i], nullptr);
        }

        // Aggregate results from all threads
        vector<string> allResults;
        for (int i = 0; i < NUM_THREADS; i++) {
            for (const auto& result : searchData[i].results) {
                allResults.push_back(result);
            }
        }

        auto endSearch = chrono::high_resolution_clock::now();
        chrono::duration<double, micro> searchTime = endSearch - startSearch;
        totalSearchTime += searchTime.count();
        totalResults += allResults.size();

        cout << "Search \"" << input << "\": ";
        if (allResults.empty()) {
            cout << "No matches found";
        } else {
            cout << allResults.size() << " match(es) - ";
            for (size_t j = 0; j < allResults.size(); j++) {
                cout << allResults[j];
                if (j < allResults.size() - 1) cout << ", ";
            }
        }
        cout << " (" << searchTime.count() << " μs)" << endl;
    }

    cout << endl;
    cout << "=== Summary ===" << endl;
    cout << "Total searches: " << testInputs.size() << endl;
    cout << "Total results found: " << totalResults << endl;
    cout << "Average search time: " << (totalSearchTime / testInputs.size()) << " μs" << endl;
    cout << "Parallel processing with " << NUM_THREADS << " threads" << endl;

    return 0;
}
