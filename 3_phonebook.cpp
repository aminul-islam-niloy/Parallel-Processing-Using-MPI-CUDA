#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#define MAX_LINE_LENGTH 100
#define MAX_NAME_LENGTH 50

using namespace std;

// Function to search for contacts in the given data chunk
void searchContacts(const string& data, const  string& name, int rank) {
    istringstream iss(data);
    string line;
    cout << "Process " << rank << " results:" ;

    while (getline(iss, line)) {
        if (line.find(name) !=  string::npos) {
             cout << line <<  endl; // Print matching line
        }
    }
}

int main(int argc, char** argv) {
    int world_size, world_rank;
    double start_time, end_time;

    MPI_Init(&argc, &argv);  // Initialize MPI

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc != 3) {
        if (world_rank == 0) {
             cerr << "Usage: " << argv[0] << " <filename> <target_name>" <<  endl;
        }
        MPI_Finalize();  // Finalize MPI
        return 1;
    }

     string filename = argv[1];
     string target_name = argv[2];

     string data;
    int data_size = 0;

    if (world_rank == 0) {
         ifstream file(filename);
        if (!file) {
             cerr << "Error opening file: " << filename <<  endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Read the entire file into a string
         stringstream buffer;
        buffer << file.rdbuf();
        data = buffer.str();

        data_size = data.size() / world_size;
    }

    // Distribute data size
    MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for local data
     vector<char> local_data(data_size + 1);

    // Scatter data to all processes
    MPI_Scatter(data.data(), data_size, MPI_CHAR, local_data.data(), data_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    local_data[data_size] = '\0'; // Null terminate the local string

    // Start timing the search
    start_time = MPI_Wtime();

    // Perform local search
    searchContacts( string(local_data.begin(), local_data.end()), target_name, world_rank);

    // End timing the search
    end_time = MPI_Wtime();

    if (world_rank == 0) {
         cout << "Total time: " << (end_time - start_time) << " seconds\n";
    }

    MPI_Finalize();  // Finalize MPI
    return 0;
}


//mpiexec -n 4 ./GetingStarted2CPP phonebook.txt John

