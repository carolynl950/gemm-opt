// Vendor BLAS wrapper — only compiled when CMake finds a BLAS.
// Not part of the optimization ladder; it's the reference ceiling.

#include "gemm.h"

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

void matmul_blas(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // alpha = 1, beta = 1  ->  C = A*B + C, matching the += contract
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, lda, B, ldb, 1.0f, C, ldc);
}
