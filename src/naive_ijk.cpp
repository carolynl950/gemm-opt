#include "alloc.hpp"
#include "check.hpp"

void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K){
    for (int i = 0; i < M; i++) {            // which row of C (and of A) we're filling
        const float* a_row = A + i * K;      // start of row i of A — A is K wide, so rows sit K apart
        float* c_row = C + i * N;            // start of row i of C — C is N wide, so rows sit N apart

        for (int j = 0; j < N; j++) {        // which column of C (and of B) we're filling
            float sum = 0.0f;                // accumulator for one output cell; stays in a register
            for (int k = 0; k < K; k++) {    // the shared dimension: across A's row, down B's column
                sum += a_row[k] * B[k * N + j];   // B row k, column j — k*N skips k whole rows
            }
            c_row[j] = sum;                  // one store per output cell
        }
    }
};

int main() {
    const int M = 2, N = 2, K = 2;

    float* A = alloc_matrix(M, K);
    float* B = alloc_matrix(K, N);
    float* C = alloc_matrix(M, N);
    float* expected = alloc_matrix(M, N);

    // A = [1 2]
    //     [3 4]
    A[idx(0, 0, K)] = 1.0f;
    A[idx(0, 1, K)] = 2.0f;
    A[idx(1, 0, K)] = 3.0f;
    A[idx(1, 1, K)] = 4.0f;

    // B = [5 6]
    //     [7 8]
    B[idx(0, 0, N)] = 5.0f;
    B[idx(0, 1, N)] = 6.0f;
    B[idx(1, 0, N)] = 7.0f;
    B[idx(1, 1, N)] = 8.0f;

    matmul_naive(A, B, C, M, N, K);

    expected[idx(0, 0, K)] = 19; 
    expected[idx(0, 1, K)] = 22; 
    expected[idx(1, 0, K)] = 43; 
    expected[idx(1, 1, K)] = 50; 

    bool ok = matrices_match(expected, C, M, N, N, K);
    printf("naive 2x2: %s\n", ok ? "PASS" : "FAIL");

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
    free_matrix(expected);
    return 0;
}