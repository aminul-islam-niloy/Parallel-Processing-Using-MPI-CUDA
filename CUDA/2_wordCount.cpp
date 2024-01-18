#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>
#include <ctime>
#include <cuda_runtime.h>

using namespace std;

const int BLOCK_SIZE = 256;

// CUDA kernel to count word occurrences in a portion of text
__global__ void countWordsKernel(const char* text, int* wordCounts, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (tid < size) {
        int start = tid;
        int end = tid;

        // Find the start and end indices of the current word
        while (end < size && text[end] != ' ' && text[end] != '\n') {
            end++;
        }

        // Increment the count for the current word
        if (end > start) {
            atomicAdd(&wordCounts[start], 1);
        }

        tid += stride;
    }
}

// Function to count word occurrences in the text using CUDA
map<string, int> countWordsCUDA(const string& text) {
    int size = text.size();

    char* d_text;
    int* d_wordCounts;

    // Allocate device memory
    cudaMalloc((void**)&d_text, size * sizeof(char));
    cudaMalloc((void**)&d_wordCounts, size * sizeof(int));

    // Copy text to device
    cudaMemcpy(d_text, text.c_str(), size * sizeof(char), cudaMemcpyHostToDevice);

    // Initialize word counts on the device
    cudaMemset(d_wordCounts, 0, size * sizeof(int));

    // Define grid and block dimensions
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    // Launch kernel
    countWordsKernel<<<dimGrid, dimBlock>>>(d_text, d_wordCounts, size);

    // Copy result from device to host
    int* wordCounts = new int[size];
    cudaMemcpy(wordCounts, d_wordCounts, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_text);
    cudaFree(d_wordCounts);

    // Process word counts on the host
    map<string, int> wordCountMap;
    istringstream iss(text);
    string word;
    int index = 0;

    while (iss >> word) {
        wordCountMap[word] += wordCounts[index++];
    }

    // Free host memory
    delete[] wordCounts;

    return wordCountMap;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <filename>" << endl;
        return 1;
    }

    string filename = argv[1];

    double startTime = clock();

    // Read file
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Unable to open file." << endl;
        return 1;
    }

    // Read file content into a string
    stringstream buffer;
    buffer << file.rdbuf();
    string fileContent = buffer.str();

    // Close the file
    file.close();

    // Count words using CUDA
    map<string, int> wordCountMap = countWordsCUDA(fileContent);

    double endTime = clock();

    // Sort word counts in descending order
    vector<pair<string, int>> sortedWordCount(wordCountMap.begin(), wordCountMap.end());
    sort(sortedWordCount.begin(), sortedWordCount.end(),
        [](const pair<string, int>& a, const pair<string, int>& b) {
            return a.second > b.second;
        }
    );

    // Print total time
    cout << "Total time: " << (endTime - startTime) / CLOCKS_PER_SEC << " seconds" << endl;

    // Print top 10 occurrences
    cout << "Top 10 occurrences:" << endl;
    int count = min(10, static_cast<int>(sortedWordCount.size()));
    for (int i = 0; i < count; ++i) {
        cout << sortedWordCount[i].first << ": " << sortedWordCount[i].second << " times" << endl;
    }

    return 0;
}

// nvcc -o word_count_cuda word_count_cuda.cu
// ./word_count_cuda input.txt
