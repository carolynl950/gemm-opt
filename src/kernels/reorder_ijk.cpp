#include "gemm.h"

// Stage 1: ikj loop order. Same math as naive; the inner loop now walks
// rows of B and C contiguously instead of a column of B.
void matmul_ikj(const float* A, const float* B, float* C,
                int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; i++) {                // which row of C (and of A) we're filling
        const float* a_row = A + i * lda;        // start of row i of A
        float* c_row = C + i * ldc;              // start of row i of C — reused for every k

        for (int k = 0; k < K; k++) {            // the shared dimension, now the middle loop
            const float a = a_row[k];            // A[i][k], loaded once; constant across the j loop
            const float* b_row = B + k * ldb;    // start of row k of B

            for (int j = 0; j < N; j++) {        // contiguous in both b_row and c_row: every float of each cache line is used
                c_row[j] += a * b_row[j];        // no reduction -> independent updates -> compiler can vectorize this loop
            }
        }
    }
}
