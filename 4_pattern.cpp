#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

// Function to count occurrences of a pattern in a given text
int countPatternOccurrences(const string& text, const string& pattern) {
    int count = 0;
    size_t pos = text.find(pattern);

    while (pos != string::npos) {
        count++;
        pos = text.find(pattern, pos + 1);
    }

    return count;
}

int main(int argc, char** argv) {
    int world_size, world_rank;
    double start_time, end_time;

    MPI_Init(&argc, &argv);  // Initialize MPI

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc != 3) {
        if (world_rank == 0) {
            cerr << "Usage: " << argv[0] << " <filename> <pattern>" << endl;
        }
        MPI_Finalize();  // Finalize MPI
        return 1;
    }

    string filename = argv[1];
    string pattern = argv[2];

    string data;
    int data_size = 0;

    if (world_rank == 0) {
        ifstream file(filename);
        if (!file) {
            cerr << "Error opening file: " << filename << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Read the entire file into a string
        stringstream buffer;
        buffer << file.rdbuf();
        data = buffer.str();

        data_size = data.size();
    }

    // Broadcast data size
    MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for local data
    vector<char> local_data(data_size + 1);

    // Scatter data to all processes
    MPI_Scatter(data.data(), data_size, MPI_CHAR, local_data.data(), data_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    local_data[data_size] = '\0'; // Null terminate the local string

    // Start timing the search
    start_time = MPI_Wtime();

    // Perform local search
    int local_occurrences = countPatternOccurrences(string(local_data.begin(), local_data.end()), pattern);

    // Sum local occurrences across all processes
    int global_occurrences;
    MPI_Reduce(&local_occurrences, &global_occurrences, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // End timing the search
    end_time = MPI_Wtime();

    if (world_rank == 0) {
        cout << "Total time: " << (end_time - start_time) << " seconds\n";
        cout << "Occurrences of pattern \"" << pattern << "\": " << global_occurrences << endl;
    }

    MPI_Finalize();  // Finalize MPI
    return 0;
}


//mpiexec -n 4 ./GetingStarted2CPP pattern.txt %pattern%

