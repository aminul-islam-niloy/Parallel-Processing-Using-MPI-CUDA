#include <iostream>
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the start time
    double start_time = MPI_Wtime();

    // Calculate the range of numbers to print for each process
    int numbersPerProcess = 10 / size;
    int startNumber = rank * numbersPerProcess + 1;
    int endNumber = (rank + 1) * numbersPerProcess;

    // Ensure the last process prints any remaining numbers
    if (rank == size - 1)
    {
        endNumber = 10;
    }

    // Print numbers for each process
    for (int i = startNumber; i <= endNumber; ++i)
    {
        std::cout << "Process " << rank << ": " << i << std::endl;
    }

    double end_time = MPI_Wtime();

    // Calculate and print the execution time for each process
    double execution_time = end_time - start_time;
    std::cout << "Process " << rank << " execution time: " << execution_time << " seconds" << std::endl;

    MPI_Finalize();

    return 0;
}

// Note: How to run
//  Build this project or file
// mpiexec -n 4 ./LabTest
// mpiexec -n numofProcess ./FileName
