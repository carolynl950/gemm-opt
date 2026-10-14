// Stage 5a: naive CUDA SGEMM — one thread computes one element of C.
// GPU counterpart of src/kernels/naive_ijk.cpp: correct, simple, and slow on purpose.
#include "gemm_cuda.cuh"

// Runs once per thread. Every thread executes this same code; the only thing that
// differs between threads is their blockIdx / threadIdx, which picks the C element.
__global__ void naive_kernel(const float* A, const float* B, float* C,
                             int M, int N, int K, int lda, int ldb, int ldc) {
    // x -> row on purpose. A warp is 32 threads with consecutive threadIdx.x, so here a
    // warp covers 32 consecutive ROWS of C. Stage 5b measures why this mapping is slow.
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    // The grid is rounded up to whole blocks: threads past the edge of C do nothing.
    if (row >= M || col >= N) return;

    float sum = 0.0f;                                  // in a register, like naive_ijk's sum
    for (int k = 0; k < K; k++) {
        sum += A[row * lda + k] * B[k * ldb + col];    // strides, not K / N: tests use padded rows
    }
    C[row * ldc + col] += sum;                         // += : the contract in gemm_cuda.cuh
}

// Host-side launcher: decides the grid shape and launches the kernel.
void cuda_naive(const float* A, const float* B, float* C,
                int M, int N, int K, int lda, int ldb, int ldc) {
    const dim3 block(32, 32);                                  // 1024 threads per block, the maximum
    const dim3 grid((M + block.x - 1) / block.x,               // x covers rows, rounded up
                    (N + block.y - 1) / block.y);              // y covers cols, rounded up
    naive_kernel<<<grid, block>>>(A, B, C, M, N, K, lda, ldb, ldc);
    // No sync or error check here: the caller checks cudaGetLastError() after the call.
}
