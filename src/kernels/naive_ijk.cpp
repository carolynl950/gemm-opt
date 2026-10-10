#include "gemm.h"

void matmul_naive(const float* A, const float* B, float* C,
                  int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; i++) {            // which row of C (and of A) we're filling
        const float* a_row = A + i * lda;    // start of row i of A — rows sit lda apart
        float* c_row = C + i * ldc;          // start of row i of C — rows sit ldc apart

        for (int j = 0; j < N; j++) {        // which column of C (and of B) we're filling
            float sum = 0.0f;                // accumulator for one output cell; stays in a register
            for (int k = 0; k < K; k++) {    // the shared dimension: across A's row, down B's column
                sum += a_row[k] * B[k * ldb + j];  // B row k, column j — k*ldb skips k whole rows
            }
            c_row[j] += sum;                 // one store per output cell; += per the contract in gemm.h
        }
    }
}
