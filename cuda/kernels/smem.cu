// Stage 5c: shared-memory tiling — GPU version of Stage 2 cache blocking.
//
// Each 32x32 thread block computes a 32x32 tile of C. It walks K in steps of TILE:
// all 1024 threads cooperatively load one 32x32 tile of A and one of B from global
// memory into shared memory (one float each, coalesced), wait for everyone, then each
// thread does 32 multiply-adds reading only shared memory. Every value loaded from
// global memory is reused 32 times instead of once.
#include "gemm_cuda.cuh"

constexpr int TILE = 32;

__global__ void smem_kernel(const float* A, const float* B, float* C,
                            int M, int N, int K, int lda, int ldb, int ldc) {
    __shared__ float As[TILE][TILE];   // tile of A: rows of this block, cols k0..k0+31
    __shared__ float Bs[TILE][TILE];   // tile of B: rows k0..k0+31, cols of this block

    const int tx = threadIdx.x, ty = threadIdx.y;   // position inside the tile
    const int row = blockIdx.y * TILE + ty;         // this thread's C element
    const int col = blockIdx.x * TILE + tx;         // (x -> col: keeps 5b's coalescing)

    float sum = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        // Cooperative load: each thread brings in ONE element of each tile.
        // Across a warp (consecutive tx, same ty): A reads A[row][k0..k0+31] and B reads
        // B[k0+ty][col..col+31], 32 consecutive floats each -> both coalesced.
        // Out-of-range elements become 0, so the last partial tile needs no bounds check below.
        As[ty][tx] = (row < M && k0 + tx < K) ? A[row * lda + k0 + tx] : 0.0f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * ldb + col] : 0.0f;

        __syncthreads();   // no thread reads a tile until every thread has written its part

        #pragma unroll
        for (int kk = 0; kk < TILE; kk++) {
            // As[ty][kk]: same address for the whole warp -> broadcast
            // Bs[kk][tx]: 32 consecutive floats -> one per shared-memory bank, no conflict
            sum += As[ty][kk] * Bs[kk][tx];
        }

        __syncthreads();   // no thread overwrites a tile (next k0) while others still read it
    }

    // Bounds check at the END, not an early return: threads outside C must still reach
    // every __syncthreads() and load their share of each tile, or the block deadlocks / reads garbage.
    if (row < M && col < N) C[row * ldc + col] += sum;
}

void cuda_smem(const float* A, const float* B, float* C,
               int M, int N, int K, int lda, int ldb, int ldc) {
    const dim3 block(TILE, TILE);
    const dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    smem_kernel<<<grid, block>>>(A, B, C, M, N, K, lda, ldb, ldc);
}
