#include <iostream>
#include <iomanip>
#include <ctime>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

const int BLOCK_SIZE = 16;

// CUDA kernel for matrix multiplication
__global__ void matrixMultiplicationKernel(int *A, int *B, int *C, int M, int N, int P)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < P)
    {
        int sum = 0;
        for (int k = 0; k < N; ++k)
        {
            sum += A[row * N + k] * B[k * P + col];
        }
        C[row * P + col] = sum;
    }
}

// Function to perform matrix multiplication on the GPU
void matrixMultiplicationGPU(int *A, int *B, int *C, int M, int N, int P)
{
    int *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc((void **)&d_A, M * N * sizeof(int));
    cudaMalloc((void **)&d_B, N * P * sizeof(int));
    cudaMalloc((void **)&d_C, M * P * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_A, A, M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * P * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimGrid((P - 1) / BLOCK_SIZE + 1, (M - 1) / BLOCK_SIZE + 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    // Launch kernel
    matrixMultiplicationKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, P);

    // Copy result from device to host
    cudaMemcpy(C, d_C, M * P * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        cout << "Usage: " << argv[0] << " <K> <M> <N> <P>" << endl;
        return 1;
    }

    int K = atoi(argv[1]);
    int M = atoi(argv[2]);
    int N = atoi(argv[3]);
    int P = atoi(argv[4]);

    cout << "Matrix dimensions: A(" << M << "x" << N << "), B(" << N << "x" << P << ")" << endl;

    int *A = new int[K * M * N];
    int *B = new int[K * N * P];
    int *C = new int[K * M * P];

    // Initialize matrices A and B with random values
    srand(time(NULL));
    for (int i = 0; i < K * M * N; ++i)
    {
        A[i] = rand() % 10;
    }
    for (int i = 0; i < K * N * P; ++i)
    {
        B[i] = rand() % 10;
    }

    // Perform matrix multiplication on the GPU
    matrixMultiplicationGPU(A, B, C, M, N, P);

    // Print the result matrix C
    cout << "Result matrix C:" << endl;
    for (int i = 0; i < K; ++i)
    {
        cout << "Matrix " << i + 1 << ":" << endl;
        for (int j = 0; j < M; ++j)
        {
            for (int k = 0; k < P; ++k)
            {
                cout << setw(5) << C[i * M * P + j * P + k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    // Save the result matrix C to a file if needed
    // Uncomment the following code to save to a file

    ofstream outFile("result_matrix_cuda.txt");
    if (outFile.is_open())
    {
        for (int i = 0; i < K * M * P; ++i)
        {
            outFile << C[i] << " ";
        }
        outFile.close();
        cout << "Result matrix C saved to result_matrix_cuda.txt" << endl;
    }
    else
    {
        cerr << "Unable to open file for saving the result matrix." << endl;
    }

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

// !apt update
// !apt install -y nvidia-cuda-toolkit
// !nvcc -o matrix_multiplication_cuda matrix_multiplication_cuda.cu
// !./matrix_multiplication_cuda <K> <M> <N> <P>

// # Example input for CUDA matrix multiplication
// K = 2
// M = 3
// N = 2
// P = 4

// !nvcc -o matrix_multiplication_cuda matrix_multiplication_cuda.cu
// !./matrix_multiplication_cuda $K $M $N $P
