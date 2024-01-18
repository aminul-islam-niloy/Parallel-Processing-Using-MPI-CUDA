#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#define MAX_LINE_LENGTH 100
#define MAX_NAME_LENGTH 50

using namespace std;

// CUDA kernel to search for contacts in the given data chunk
__global__ void searchContactsCUDA(const char *data, int data_size, const char *name)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < data_size)
    {
        // Search for the target name in the data
        int i = 0;
        while (name[i] != '\0' && data[tid + i] == name[i])
        {
            i++;
        }

        // If the target name is found, print the matching line
        if (name[i] == '\0')
        {
            printf("%s\n", &data[tid]);
        }

        tid += blockDim.x * gridDim.x;
    }
}

int main(int argc, char **argv)
{
    int world_size, world_rank;
    double start_time, end_time;

    MPI_Init(&argc, &argv); // Initialize MPI

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc != 3)
    {
        if (world_rank == 0)
        {
            cerr << "Usage: " << argv[0] << " <filename> <target_name>" << endl;
        }
        MPI_Finalize(); // Finalize MPI
        return 1;
    }

    string filename = argv[1];
    string target_name = argv[2];

    string data;
    int data_size = 0;

    if (world_rank == 0)
    {
        ifstream file(filename);
        if (!file)
        {
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
    char *d_local_data;
    cudaMalloc((void **)&d_local_data, data_size);

    if (world_rank == 0)
    {
        // Copy data to device
        cudaMemcpy(d_local_data, data.c_str(), data_size, cudaMemcpyHostToDevice);
    }

    // Synchronize before entering timing section
    MPI_Barrier(MPI_COMM_WORLD);

    // Start timing the search
    start_time = MPI_Wtime();

    // Launch CUDA kernel
    int block_size = 256;
    int num_blocks = (data_size + block_size - 1) / block_size;

    searchContactsCUDA<<<num_blocks, block_size>>>(d_local_data, data_size, target_name.c_str());

    // Synchronize after the kernel execution
    cudaDeviceSynchronize();

    // End timing the search
    end_time = MPI_Wtime();

    if (world_rank == 0)
    {
        cout << "Total time: " << (end_time - start_time) << " seconds" << endl;
    }

    // Clean up
    cudaFree(d_local_data);
    MPI_Finalize(); // Finalize MPI

    return 0;
}
// mpiexec -n 4 ./search_contacts_cuda phonebook.txt John
