// Stage 5d: register tiling — GPU version of the CPU 8x12 micro-kernel (Stage 3).
//
// 5c needed 2 shared-memory reads per FMA. Here each thread owns an 8x8 patch of C in
// registers (64 accumulators). Per k step it reads 8 values of A and 8 of B from shared
// memory and does 64 FMAs: 16 shared reads per 64 FMAs instead of 2 per 1 (8x fewer).
//
//   block  = 256 threads, computes a 128 x 128 tile of C
//   thread = 8 x 8 outputs, strided (rows ty + 16i, cols tx + 16j; see below)
//   K walked in steps of BK = 8: the block loads a 128x8 tile of A and an 8x128 tile of B
//
// 256-thread blocks (vs 1024 in 5c) also let 2 blocks share an SM, so one block's
// __syncthreads() stall can be hidden by the other block's math.
#include "gemm_cuda.cuh"

constexpr int BM = 128, BN = 128, BK = 8;   // block tile of C, and k-step
constexpr int TM = 8, TN = 8;               // per-thread tile of C
constexpr int TPR = BN / TN;                // threads per row of the thread grid: 16
constexpr int THREADS = (BM / TM) * (BN / TN);   // 16 x 16 = 256
constexpr int APAD = 4;                     // shared-memory padding for As, see the load below

static_assert((BM * BK) % THREADS == 0 && (BK * BN) % THREADS == 0, "tiles must split evenly across threads");

__global__ void __launch_bounds__(THREADS)
regtile_kernel(const float* A, const float* B, float* C,
               int M, int N, int K, int lda, int ldb, int ldc) {
    // A tile stored TRANSPOSED (k-major): the compute loop then reads 8 rows of A at one k
    // as As[k][...], the same layout as Bs. +APAD columns: see the bank-conflict note below.
    __shared__ float As[BK][BM + APAD];
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;
    const int tx = tid % TPR;                   // 0..15: which column group
    const int ty = tid / TPR;                   // 0..15: which row group
    const int bm = blockIdx.y * BM;             // this block's first row of C
    const int bn = blockIdx.x * BN;             // this block's first col of C

    // Strided ownership: thread (ty, tx) owns rows ty + 16*i and cols tx + 16*j (i, j = 0..7).
    // Why strided and not 8 consecutive cols: for a fixed j, the 16 tx values of a warp then
    // read Bs[k][tx + 16j] = 16 consecutive floats (16 different banks, no conflict) and
    // write C at 16 consecutive columns (coalesced). With consecutive ownership (tx*8 + j)
    // the warp's reads would land 8 floats apart: only 4 distinct banks, a 4-way conflict.
    float acc[TM][TN] = {};                     // 64 accumulators: stay in registers (fully unrolled)
    float a[TM], b[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        // ---- cooperative load: 1024 floats of A and 1024 of B, 4 + 4 per thread ----
        #pragma unroll
        for (int l = 0; l < (BM * BK) / THREADS; l++) {
            const int idx = tid + l * THREADS;
            const int r = idx / BK, c = idx % BK;          // 8 consecutive threads read 8 consecutive floats of one A row
            const int gr = bm + r, gc = k0 + c;
            // Store transposed. Without padding, the 8 threads with the same r but c = 0..7 would
            // write As[c][r] = addresses 128 apart = the same bank (8-way conflict). With row
            // length 132, c*132 + r spreads them over 8 different banks.
            As[c][r] = (gr < M && gc < K) ? A[gr * lda + gc] : 0.0f;
        }
        #pragma unroll
        for (int l = 0; l < (BK * BN) / THREADS; l++) {
            const int idx = tid + l * THREADS;
            const int r = idx / BN, c = idx % BN;          // 32 consecutive threads read 32 consecutive floats of a B row
            const int gr = k0 + r, gc = bn + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * ldb + gc] : 0.0f;
        }
        __syncthreads();   // tiles complete before anyone computes

        // ---- compute: outer product of 8 A values x 8 B values, BK times ----
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[k][ty + i * (BM / TM)];   // warp: 2 distinct ty -> broadcast
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][tx + j * TPR];        // warp: 16 consecutive -> no conflict
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a[i] * b[j];                                // 64 FMAs from registers
        }
        __syncthreads();   // nobody overwrites the tiles (next k0) while others still read them
    }

    // ---- write back: C += acc, bounds-checked (edge blocks are partly outside C) ----
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        const int row = bm + ty + i * (BM / TM);
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            const int col = bn + tx + j * TPR;
            if (row < M && col < N) C[row * ldc + col] += acc[i][j];
        }
    }
}

void cuda_regtile(const float* A, const float* B, float* C,
                  int M, int N, int K, int lda, int ldb, int ldc) {
    const dim3 block(THREADS);                              // 1-D: tx / ty computed from threadIdx.x
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    regtile_kernel<<<grid, block>>>(A, B, C, M, N, K, lda, ldb, ldc);
}
