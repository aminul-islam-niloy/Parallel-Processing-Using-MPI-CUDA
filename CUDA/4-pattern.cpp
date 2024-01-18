// mpiexec -n 4 ./count_pattern_occurrences_cuda input.txt %pattern%
// nvcc -o count_pattern_occurrences_cuda count_pattern_occurrences_cuda.cu
// mpiexec -n 4 ./count_pattern_occurrences_cuda input.txt %pattern%

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

using namespace std;

// CUDA kernel to count occurrences of a pattern in a given text
__global__ void countPatternOccurrencesCUDA(const char *text, int text_size, const char *pattern, int *result)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < text_size)
    {
        int i = 0;
        while (pattern[i] != '\0' && text[tid + i] == pattern[i])
        {
            i++;
        }

        if (pattern[i] == '\0')
        {
            atomicAdd(result, 1); // Atomically increment the result
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
            cerr << "Usage: " << argv[0] << " <filename> <pattern>" << endl;
        }
        MPI_Finalize(); // Finalize MPI
        return 1;
    }

    string filename = argv[1];
    string pattern = argv[2];

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
    int block_size = BLOCK_SIZE;
    int num_blocks = (data_size + block_size - 1) / block_size;

    // Allocate memory for result on device
    int *d_result;
    cudaMalloc((void **)&d_result, sizeof(int));
    cudaMemset(d_result, 0, sizeof(int));

    countPatternOccurrencesCUDA<<<num_blocks, block_size>>>(d_local_data, data_size, pattern.c_str(), d_result);

    // Copy result from device to host
    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Synchronize after the kernel execution
    cudaDeviceSynchronize();

    // End timing the search
    end_time = MPI_Wtime();

    if (world_rank == 0)
    {
        cout << "Total time: " << (end_time - start_time) << " seconds" << endl;
        cout << "Occurrences of pattern \"" << pattern << "\": " << result << endl;
    }

    // Clean up
    cudaFree(d_local_data);
    cudaFree(d_result);
    MPI_Finalize(); // Finalize MPI

    return 0;
}
