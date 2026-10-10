// Stage 0 sanity check: naive kernel against a hand-computed 2x2.

#include "alloc.hpp"
#include "check.hpp"
#include "gemm.h"

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

    // Kernels accumulate (C += A*B), so C must start at zero
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    matmul_naive(A, B, C, M, N, K, /*lda=*/K, /*ldb=*/N, /*ldc=*/N);

    expected[idx(0, 0, N)] = 19;
    expected[idx(0, 1, N)] = 22;
    expected[idx(1, 0, N)] = 43;
    expected[idx(1, 1, N)] = 50;

    bool ok = matrices_match(expected, C, M, N, N, K);
    printf("naive 2x2: %s\n", ok ? "PASS" : "FAIL");

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
    free_matrix(expected);
    return ok ? 0 : 1;   // nonzero exit is how CTest/CI sees a failure
}
