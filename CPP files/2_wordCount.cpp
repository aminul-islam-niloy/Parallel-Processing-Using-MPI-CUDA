#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>

using namespace std;

// Function to clean and normalize a word
string cleanWord(const string& word) {
    string cleanedWord;
    for (char ch : word) {
        if (isalpha(ch)) {
            cleanedWord += tolower(ch);
        }
    }
    return cleanedWord;
}

int main() {
    // Open the file
    ifstream inputFile("example.txt");

    // Check if the file is opened successfully
    if (!inputFile.is_open()) {
        cerr << "Error opening the file." << endl;
        return 1;
    }

    // Map to store word frequencies
    map<string, int> wordFrequency;

    // Read the file and count word frequencies
    string line;
    while (getline(inputFile, line)) {
        stringstream ss(line);
        string word;
        while (ss >> word) {
            word = cleanWord(word);
            if (!word.empty()) {
                wordFrequency[word]++;
            }
        }
    }

    // Close the file
    inputFile.close();

    // Vector to store pairs of words and their frequencies
    vector<pair<string, int>> wordFrequencyVector(wordFrequency.begin(), wordFrequency.end());

    // Sort the vector in descending order of frequency
    sort(wordFrequencyVector.begin(), wordFrequencyVector.end(),
         [](const pair<string, int>& a, const pair<string, int>& b) {
             return a.second > b.second;
         });

    // Output the sorted word frequencies
    cout << "Word Frequencies (Descending Order):" << endl;
    for (const auto& entry : wordFrequencyVector) {
        cout << entry.first << ": " << entry.second << " times" << endl;
    }

    return 0;
}
