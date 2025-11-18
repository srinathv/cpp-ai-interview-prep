#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>

using namespace std;

// Dictionary of 20 words
const vector<string> DICTIONARY = {
    "algorithm", "binary", "compiler", "database", "encryption",
    "framework", "gateway", "hashmap", "interface", "java",
    "kernel", "library", "memory", "network", "object",
    "protocol", "query", "runtime", "socket", "thread"
};

// Function to build a prefix map for efficient searching
map<string, vector<string>> buildPrefixMap(const vector<string>& dictionary) {
    map<string, vector<string>> prefixMap;

    // Use for loop to iterate through dictionary
    for (size_t i = 0; i < dictionary.size(); i++) {
        const string& word = dictionary[i];

        // Generate 2-letter and 3-letter prefixes
        for (int prefixLen = 2; prefixLen <= 3 && prefixLen <= (int)word.length(); prefixLen++) {
            string prefix = word.substr(0, prefixLen);

            // Convert to lowercase for case-insensitive matching
            transform(prefix.begin(), prefix.end(), prefix.begin(), ::tolower);

            // Add word to the prefix map
            prefixMap[prefix].push_back(word);
        }
    }

    return prefixMap;
}

// Function to search for words matching the input prefix
vector<string> searchWords(const map<string, vector<string>>& prefixMap,
                           const string& input) {
    string lowerInput = input;
    transform(lowerInput.begin(), lowerInput.end(), lowerInput.begin(), ::tolower);

    // Search in the map
    auto it = prefixMap.find(lowerInput);

    if (it != prefixMap.end()) {
        return it->second;
    }

    return vector<string>(); // Empty vector if not found
}

int main() {
    cout << "=== Dictionary Search - Serial Version ===" << endl;
    cout << "Dictionary size: " << DICTIONARY.size() << " words" << endl;
    cout << endl;

    // Display dictionary
    cout << "Dictionary words:" << endl;
    for (size_t i = 0; i < DICTIONARY.size(); i++) {
        cout << "  " << (i + 1) << ". " << DICTIONARY[i] << endl;
    }
    cout << endl;

    // Build prefix map
    auto startBuild = chrono::high_resolution_clock::now();
    map<string, vector<string>> prefixMap = buildPrefixMap(DICTIONARY);
    auto endBuild = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> buildTime = endBuild - startBuild;
    cout << "Prefix map built in: " << buildTime.count() << " ms" << endl;
    cout << "Total prefixes: " << prefixMap.size() << endl;
    cout << endl;

    // Test searches with various inputs
    vector<string> testInputs = {"al", "bin", "da", "enc", "fr", "ha", "int", "ke", "lib", "net", "pro", "que", "run", "so", "th"};

    cout << "=== Running Test Searches ===" << endl;

    double totalSearchTime = 0.0;
    int totalResults = 0;

    // Use for loop for searches
    for (size_t i = 0; i < testInputs.size(); i++) {
        const string& input = testInputs[i];

        auto startSearch = chrono::high_resolution_clock::now();
        vector<string> results = searchWords(prefixMap, input);
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

    // Interactive search
    cout << endl << "=== Interactive Search ===" << endl;
    cout << "Enter 2-3 letter prefix (or 'quit' to exit):" << endl;

    string userInput;
    while (true) {
        cout << "> ";
        cin >> userInput;

        if (userInput == "quit" || userInput == "q") {
            break;
        }

        if (userInput.length() < 2 || userInput.length() > 3) {
            cout << "Please enter 2-3 letters only." << endl;
            continue;
        }

        vector<string> results = searchWords(prefixMap, userInput);

        if (results.empty()) {
            cout << "No matches found." << endl;
        } else {
            cout << "Found " << results.size() << " match(es):" << endl;
            for (const auto& word : results) {
                cout << "  - " << word << endl;
            }
        }
    }

    cout << "Goodbye!" << endl;

    return 0;
}
