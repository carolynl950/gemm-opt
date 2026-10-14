#include "gemm.h"
#include "alloc.hpp"   // alloc_matrix / free_matrix for the pack buffer
#include <algorithm>   // std::min

// Stage 2: cache blocking. Same ikj inner loops as Stage 1, but restricted to
// an MC x KC x NC tile so a KC x NC block of B is reused by MC rows of A
// while it's still in cache, instead of re-streaming all of B for every row.

// Block sizes: runtime-settable so bench can sweep them. Defaults are a first guess:
// B tile = KC * NC * 4 B = 64 KB, half of the M4 Pro's 128 KB P-core L1d.
// NC is a multiple of 4 so tile starts keep 16 B alignment (see results/stage1_ikj.md §6).
static int g_mc = 64, g_kc = 64, g_nc = 256;

void set_block_sizes(int mc, int kc, int nc) {
    if (mc > 0) g_mc = mc;
    if (kc > 0) g_kc = kc;
    if (nc > 0) g_nc = nc;
}

void matmul_blocked(const float* A, const float* B, float* C,
                    int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i0 = 0; i0 < M; i0 += g_mc) {                 // block of rows of A and C
        const int i_end = std::min(i0 + g_mc, M);          // last block may be short: remainder rows
        for (int k0 = 0; k0 < K; k0 += g_kc) {             // block of the shared dimension
            const int k_end = std::min(k0 + g_kc, K);
            for (int j0 = 0; j0 < N; j0 += g_nc) {         // block of columns of B and C
                const int j_end = std::min(j0 + g_nc, N);

                // ---- tile: same ikj as Stage 1, bounds limited to the tile ----
                for (int i = i0; i < i_end; i++) {
                    const float* a_row = A + i * lda;
                    float* c_row = C + i * ldc;
                    for (int k = k0; k < k_end; k++) {
                        const float a = a_row[k];          // A[i][k], constant across the j loop
                        const float* b_row = B + k * ldb;  // row k of B; this tile reads [j0, j_end)
                        for (int j = j0; j < j_end; j++) {
                            c_row[j] += a * b_row[j];      // contiguous in b_row and c_row: vectorizes like ikj
                        }
                    }
                }
            }
        }
    }
}

// Stage 2b: blocking + packing. Each KC x NC tile of B is copied into a
// contiguous, aligned buffer before use, so the tile's cache footprint no
// longer depends on ldb: no power-of-two set conflicts, every row 16 B aligned.
// Loop order k0 -> j0 -> i: each tile is packed once and reused by all M rows.
void matmul_packed(const float* A, const float* B, float* C,
                   int M, int N, int K, int lda, int ldb, int ldc) {
    const int pld_max = (g_nc + 3) & ~3;                   // widest tile, rounded up to 4 floats (16 B)
    float* Bp = alloc_matrix(g_kc, pld_max);               // one buffer, reused for every tile
    if (!Bp) return;                                       // leaves C unchanged; the test would catch it

    for (int k0 = 0; k0 < K; k0 += g_kc) {
        const int k_end = std::min(k0 + g_kc, K);
        for (int j0 = 0; j0 < N; j0 += g_nc) {
            const int j_end = std::min(j0 + g_nc, N);
            const int w   = j_end - j0;                    // this tile's width (short on the last tile)
            const int pld = (w + 3) & ~3;                  // packed row stride: contiguous rows, each 16 B aligned

            // ---- pack: B[k0:k_end, j0:j_end] -> Bp, row-major with stride pld ----
            for (int k = k0; k < k_end; k++) {
                const float* src = B + k * ldb + j0;
                float* dst = Bp + (k - k0) * pld;
                for (int j = 0; j < w; j++) dst[j] = src[j];
            }

            // ---- compute: every row of A against the packed tile ----
            for (int i0 = 0; i0 < M; i0 += g_mc) {
                const int i_end = std::min(i0 + g_mc, M);
                for (int i = i0; i < i_end; i++) {
                    const float* a_row = A + i * lda;
                    float* c_row = C + i * ldc + j0;       // this tile's slice of row i of C
                    for (int k = k0; k < k_end; k++) {
                        const float a = a_row[k];
                        const float* bp_row = Bp + (k - k0) * pld;
                        for (int j = 0; j < w; j++) {
                            c_row[j] += a * bp_row[j];     // same ikj inner loop, reading the packed copy
                        }
                    }
                }
            }
        }
    }
    free_matrix(Bp);
}
