#include <iostream>
#include <vector>

using namespace std;

// Function to multiply two matrices
vector<vector<int>> multiplyMatrices(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int m = A.size();
    int n = A[0].size();
    int p = B[0].size();

    vector<vector<int>> result(m, vector<int>(p, 0));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            for (int k = 0; k < n; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}

int main() {
    int K, M, N, P;

    // Input dimensions
    cout << "Enter the number of matrices (K): ";
    cin >> K;
    cout << "Enter the dimensions of matrices (M N P): ";
    cin >> M >> N >> P;

    // Check constraints
    if (K * M * N > 1000000 || K * N * P > 1000000 || K * M * P > 1000000) {
        cout << "Invalid input. Constraints not satisfied." << endl;
        return 1;
    }

    // Input matrices
    vector<vector<vector<int>>> matricesA(K, vector<vector<int>>(M, vector<int>(N)));
    vector<vector<vector<int>>> matricesB(K, vector<vector<int>>(N, vector<int>(P)));

    cout << "Enter matrices A: " << endl;
    for (int k = 0; k < K; ++k) {
        cout << "Matrix A" << k + 1 << ":" << endl;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                cin >> matricesA[k][i][j];
            }
        }
    }

    cout << "Enter matrices B: " << endl;
    for (int k = 0; k < K; ++k) {
        cout << "Matrix B" << k + 1 << ":" << endl;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < P; ++j) {
                cin >> matricesB[k][i][j];
            }
        }
    }

    // Multiply matrices
    vector<vector<vector<int>>> resultMatrices(K);
    for (int k = 0; k < K; ++k) {
        resultMatrices[k] = multiplyMatrices(matricesA[k], matricesB[k]);
    }

    // Output result matrices
    cout << "Result matrices:" << endl;
    for (int k = 0; k < K; ++k) {
        cout << "Matrix " << k + 1 << ":" << endl;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < P; ++j) {
                cout << resultMatrices[k][i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    return 0;
}


// Enter the number of matrices (K): 2
// Enter the dimensions of matrices (M N P): 2 3 2

// Enter matrices A:
// Matrix A1:
// 1 2 3
// 4 5 6
// Matrix A2:
// 7 8 9
// 10 11 12

// Enter matrices B:
// Matrix B1:
// 13 14
// 15 16
// 17 18
// Matrix B2:
// 19 20
// 21 22
// 23 24

