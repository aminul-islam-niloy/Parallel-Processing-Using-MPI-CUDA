#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>
#include <cstring>
#include <ctime>
#include <mpi.h>

using namespace std;

// Function to count word occurrences in a portion of text
map<string, int> countWords(const string& text) {
    map<string, int> wordCount;
    istringstream iss(text);
    string word;

    while (iss >> word) {
        transform(word.begin(), word.end(), word.begin(), ::tolower);
        wordCount[word]++;
    }

    return wordCount;
}

// Function to merge word count maps
map<string, int> mergeWordCounts(const map<string, int>& localWordCount) {
    map<string, int> globalWordCount;

    for (const auto& pair : localWordCount) {
        globalWordCount[pair.first] += pair.second;
    }

    return globalWordCount;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " <number_of_processes> <filename>" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    int numProcesses = stoi(argv[1]);
    string filename = argv[2];

    double startTime = MPI_Wtime();

    string fileContent;  // Declare fileContent outside the if block

    // Read file
    if (rank == 0) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Error: Unable to open file." << endl;
            MPI_Finalize();
            return 1;
        }

        // Read file content into a string
        stringstream buffer;
        buffer << file.rdbuf();
        fileContent = buffer.str();  // Assign the value here

        // Broadcast file content size to all processes
        int fileSize = fileContent.size() + 1;
        MPI_Bcast(&fileSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Broadcast file content to all processes
       // Broadcast file content to all processes
        MPI_Bcast(const_cast<char*>(fileContent.data()), fileSize, MPI_CHAR, 0, MPI_COMM_WORLD);


        file.close();
    }
    else {
        // Receive file content size from process 0
        int fileSize;
        MPI_Bcast(&fileSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Resize fileContent buffer
        fileContent.resize(fileSize);

        // Receive file content from process 0
       // Broadcast file content to all processes
        MPI_Bcast(const_cast<char*>(fileContent.data()), fileSize, MPI_CHAR, 0, MPI_COMM_WORLD);

    }

    // Calculate local word counts
    map<string, int> localWordCount = countWords(fileContent);

    // Gather word counts from all processes
    map<string, int> globalWordCount = mergeWordCounts(localWordCount);

    // Sort word counts in descending order
    vector<pair<string, int>> sortedWordCount(globalWordCount.begin(), globalWordCount.end());
    sort(sortedWordCount.begin(), sortedWordCount.end(),
        [](const pair<string, int>& a, const pair<string, int>& b) {
            return a.second > b.second;
        }
    );

    double endTime = MPI_Wtime();

    // Print total time
    if (rank == 0) {
        cout << "Total time: " << endTime - startTime << " seconds" << endl;

        // Print top 10 occurrences
        cout << "Top 10 occurrences:" << endl;
        int count = min(10, static_cast<int>(sortedWordCount.size()));
        for (int i = 0; i < count; ++i) {
            cout << sortedWordCount[i].first << ": " << sortedWordCount[i].second << " times" << endl;
        }
    }

    MPI_Finalize();

    return 0;
}


//mpiexec -n 4 ./GetingStartedCPP 4 input.txt
