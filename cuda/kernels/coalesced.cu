// Stage 5b: coalesced naive SGEMM — same algorithm as naive.cu, threads mapped so a
// warp (32 consecutive threadIdx.x) covers 32 consecutive COLUMNS of one row of C.
// Then B[k][col] and C[row][col] are 32 consecutive floats per warp (coalesced), and
// A[row][k] is one address for the whole warp (broadcast).
#include "gemm_cuda.cuh"

__global__ void coalesced_kernel(const float* A, const float* B, float* C,
                                 int M, int N, int K, int lda, int ldb, int ldc) {
    // The only change from naive.cu: x -> col, y -> row.
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) return;   // grid is rounded up to whole blocks

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        // A[row][k]: same address for all 32 threads of a warp -> one broadcast load
        // B[k][col]: 32 consecutive floats across the warp      -> 4 x 32 B transactions
        sum += A[row * lda + k] * B[k * ldb + col];
    }
    C[row * ldc + col] += sum;          // consecutive across the warp: coalesced store
}

void cuda_coalesced(const float* A, const float* B, float* C,
                    int M, int N, int K, int lda, int ldb, int ldc) {
    const dim3 block(32, 32);
    const dim3 grid((N + block.x - 1) / block.x,    // x covers cols, rounded up
                    (M + block.y - 1) / block.y);   // y covers rows, rounded up
    coalesced_kernel<<<grid, block>>>(A, B, C, M, N, K, lda, ldb, ldc);
}
