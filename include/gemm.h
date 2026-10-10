#pragma once

// Kernel contract — every implementation in src/kernels/ follows it:
//
//   C += A * B            (accumulate; caller zeroes C for a plain product)
//
//   A is M x K, row stride lda   (lda >= K)
//   B is K x N, row stride ldb   (ldb >= N)
//   C is M x N, row stride ldc   (ldc >= N)
//
// Row-major throughout: element (r, c) of X lives at X[r * ldX + c].
// Strides are separate from widths so a kernel can work on a sub-block
// of a bigger matrix (tiling) and so tests can pad rows to catch kernels
// that assume ld == width.
using Kernel = void (*)(const float* A, const float* B, float* C,
                        int M, int N, int K,
                        int lda, int ldb, int ldc);

void matmul_naive(const float* A, const float* B, float* C,
                  int M, int N, int K, int lda, int ldb, int ldc);

#ifdef GEMM_HAVE_BLAS
// Vendor BLAS (Accelerate on macOS, OpenBLAS on Linux) — the ceiling to compare against.
void matmul_blas(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc);
#endif
