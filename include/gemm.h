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

// Stage 1: ikj loop order — inner loop streams rows of B and C contiguously.
void matmul_ikj(const float* A, const float* B, float* C,
                int M, int N, int K, int lda, int ldb, int ldc);

// Stage 2: cache blocking — ikj restricted to MC x KC x NC tiles.
void matmul_blocked(const float* A, const float* B, float* C,
                    int M, int N, int K, int lda, int ldb, int ldc);
// Stage 2b: blocking + packing B tiles into a contiguous aligned buffer.
void matmul_packed(const float* A, const float* B, float* C,
                   int M, int N, int K, int lda, int ldb, int ldc);
// Set blocked's and packed's tile sizes (values <= 0 leave that size unchanged). Not thread-safe.
void set_block_sizes(int mc, int kc, int nc);

// Stage 3: packed A and B + 8x12 register-blocked micro-kernel (NEON on arm64, plain C++ elsewhere).
void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc);

#ifdef GEMM_HAVE_OPENMP
// Stage 4: simd parallelized over B panels (jr loop); M < 8 uses split-K ikj + reduction.
// Thread count from OMP_NUM_THREADS.
void matmul_omp(const float* A, const float* B, float* C,
                int M, int N, int K, int lda, int ldb, int ldc);
#endif

#ifdef GEMM_HAVE_BLAS
// Vendor BLAS (Accelerate on macOS, OpenBLAS on Linux) — the ceiling to compare against.
void matmul_blas(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc);
#endif
