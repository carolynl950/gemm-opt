// gemm_cuda.cuh — Stage 5: CUDA kernels. Plain C++ declarations only (no __global__ here),
// so host-only .cpp files (bench_cuda, test_cuda) can include it.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Every CUDA API call returns an error code; ignoring it hides bugs until much later.
#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                  \
            std::fprintf(stderr, "CUDA error: %s\n  at %s:%d: %s\n",                \
                         cudaGetErrorString(err_), __FILE__, __LINE__, #call);      \
            std::exit(1);                                                           \
        }                                                                           \
    } while (0)

// Same contract as gemm.h, but A, B, C are DEVICE pointers:
//   C += A * B, row-major, A is M x K (lda), B is K x N (ldb), C is M x N (ldc).
// Each function launches on the default stream and returns without waiting;
// the caller synchronizes before reading C.
using CudaKernel = void (*)(const float* A, const float* B, float* C,
                            int M, int N, int K, int lda, int ldb, int ldc);

void cuda_naive(const float* A, const float* B, float* C,
                int M, int N, int K, int lda, int ldb, int ldc);

// Stage 5b: naive with threads mapped for coalesced global-memory access.
void cuda_coalesced(const float* A, const float* B, float* C,
                    int M, int N, int K, int lda, int ldb, int ldc);

// Stage 5c: shared-memory tiling (32x32 tiles of A and B reused 32x from shared memory).
void cuda_smem(const float* A, const float* B, float* C,
               int M, int N, int K, int lda, int ldb, int ldc);

// Stage 5d: register tiling — each thread computes an 8x8 patch of C from registers.
void cuda_regtile(const float* A, const float* B, float* C,
                  int M, int N, int K, int lda, int ldb, int ldc);

// cuBLAS SGEMM: the ceiling to compare against (NVIDIA's tuned library).
void cuda_cublas(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc);

struct CudaImpl {
    const char* name;
    CudaKernel fn;
};

// Single list for bench_cuda and test_cuda: add each new kernel here.
inline constexpr CudaImpl CUDA_IMPLS[] = {
    {"cuda_naive", cuda_naive},
    {"cuda_coalesced", cuda_coalesced},
    {"cuda_smem", cuda_smem},
    {"cuda_regtile", cuda_regtile},
    {"cublas", cuda_cublas},
};
