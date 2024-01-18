// Add this line at the beginning of your program
// For MSVC (Visual Studio)
#if defined(_MSC_VER) && (_MSC_VER < 1900)
#define _SCL_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <iostream>
#include <iomanip>
#include <ctime>
#include <cstdlib> // Include cstdlib for atoi
#include <mpi.h>
#include <fstream>

using namespace std;

void matrixMultiplication(int* A, int* B, int* C, int M, int N, int P) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            C[i * P + j] = 0;
            for (int k = 0; k < N; ++k) {
                C[i * P + j] += A[i * N + k] * B[k * P + j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        cerr << "This program requires at least 2 processes." << endl;
        MPI_Finalize();
        return 1;
    }

    if (argc != 5) {
        if (rank == 0) {
            cout << "Usage: " << argv[0] << " <K> <M> <N> <P>" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    int K = atoi(argv[1]);
    int M = atoi(argv[2]);
    int N = atoi(argv[3]);
    int P = atoi(argv[4]);

    if (rank == 0) {
        cout << "Matrix dimensions: A(" << M << "x" << N << "), B(" << N << "x" << P << ")" << endl;
    }

    int* A = new int[K * M * N];
    int* B = new int[K * N * P];
    int* C = new int[K * M * P];

    if (rank == 0) {
        // Initialize matrices A and B with random values
        srand(time(NULL));
        for (int i = 0; i < K * M * N; ++i) {
            A[i] = rand() % 10;
        }
        for (int i = 0; i < K * N * P; ++i) {
            B[i] = rand() % 10;
        }
    }

    // Scatter A and B to all processes
    MPI_Bcast(A, K * M * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, K * N * P, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute local matrix multiplication
    int local_M = M / size;
    int* local_C = new int[K * local_M * P];
    for (int i = 0; i < K; ++i) {
        matrixMultiplication(&A[i * M * N], &B[i * N * P], &local_C[i * local_M * P], local_M, N, P);
    }

    // Gather local results to process 0
    MPI_Gather(local_C, K * local_M * P, MPI_INT, C, K * local_M * P, MPI_INT, 0, MPI_COMM_WORLD);

    // Print time taken for multiplication on process 0
    if (rank == 0) {
        cout << "Matrix multiplication completed." << endl;

        // Print the result matrix C
        if (rank == 0) {
            cout << "Result matrix C:" << endl;
            for (int i = 0; i < K; ++i) {
                cout << "Matrix " << i + 1 << ":" << endl;
                for (int j = 0; j < M; ++j) {
                    for (int k = 0; k < P; ++k) {
                        cout << setw(5) << C[i * M * P + j * P + k] << " ";
                    }
                    cout << endl;
                }
                cout << endl;
            }

            // Save the result matrix C to a file if needed
            // Uncomment the following code to save to a file
            
            ofstream outFile("result_matrix.txt");
            if (outFile.is_open()) {
                for (int i = 0; i < K * M * P; ++i) {
                    outFile << C[i] << " ";
                }
                outFile.close();
                cout << "Result matrix C saved to result_matrix.txt" << endl;
            } else {
                cerr << "Unable to open file for saving the result matrix." << endl;
            }
            
        }


        delete[] A;
        delete[] B;
        delete[] C;
    }

    MPI_Finalize();

    return 0;
}


//mpiexec -n 3 ./GetingStartedCPP 1 3 2 4

