// Stage 3: register-blocked micro-kernel with packed A and B (GotoBLAS / BLIS layout).
//
// Stage 2's inner loop needs 3 memory ops (load B, load C, store C) per FMA.
// Here an MR x NR tile of C lives in registers for the whole k loop: each k step
// loads MR floats of A + NR floats of B and does MR*NR FMAs. C is touched once per tile.
//
//   jc loop (NC cols of B)            B block  KC x NC  -> packed, lives in L2
//     pc loop (KC of shared dim)
//       ic loop (MC rows of A)        A block  MC x KC  -> packed, lives in L2
//         jr loop (NR cols)           B panel  KC x NR  -> stays in L1 across ir
//           ir loop (MR rows)         A panel  KC x MR
//             micro-kernel: C[MR x NR] += A_panel * B_panel

#include "gemm.h"
#include "alloc.hpp"
#include <algorithm>   // std::min

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

constexpr int MR = 8;     // rows of the register tile
constexpr int NR = 12;    // cols of the register tile: 3 NEON vectors of 4 floats
// 8 x 3 = 24 accumulator registers + 3 for B + 2 for A = 29 of NEON's 32.

constexpr int MC = 128;   // A block: 128 x 256 x 4 B = 128 KB (L2)
constexpr int KC = 256;   // B panel: 256 x 12 x 4 B  = 12 KB  (L1)
constexpr int NC = 1536;  // B block: 256 x 1536 x 4 B = 1.5 MB (L2); multiple of NR

static int round_up(int x, int m) { return (x + m - 1) / m * m; }

// ---- packing ----------------------------------------------------------------
// A block (mc x kc, at A) -> panels of MR rows. Within a panel, column p is stored
// as MR consecutive floats: exactly the order the micro-kernel reads them.
// Rows past mc are zero-filled. The micro-kernel computes them anyway (no bounds
// checks inside it); their results land in discarded rows of the edge tile, so the
// zeros aren't needed for correctness, only to keep garbage (NaN, denormals) out of the FMAs.
static void pack_A(const float* A, int lda, int mc, int kc, float* Ap) {
    for (int ir = 0; ir < mc; ir += MR) {
        const int rows = std::min(MR, mc - ir);
        for (int p = 0; p < kc; p++) {
            for (int r = 0; r < rows; r++) Ap[r] = A[(ir + r) * lda + p];
            for (int r = rows; r < MR; r++) Ap[r] = 0.0f;
            Ap += MR;
        }
    }
}

// B block (kc x nc, at B) -> panels of NR cols. Within a panel, row p is NR
// consecutive floats. Cols past nc are zero-filled (same reasoning as pack_A).
static void pack_B(const float* B, int ldb, int kc, int nc, float* Bp) {
    for (int jr = 0; jr < nc; jr += NR) {
        const int cols = std::min(NR, nc - jr);
        for (int p = 0; p < kc; p++) {
            const float* src = B + p * ldb + jr;
            for (int j = 0; j < cols; j++) Bp[j] = src[j];
            for (int j = cols; j < NR; j++) Bp[j] = 0.0f;
            Bp += NR;
        }
    }
}

// ---- micro-kernel: C[MR x NR] += Ap (kc x MR) * Bp (kc x NR) ----------------
#if defined(__aarch64__)

static void micro_kernel(int kc, const float* Ap, const float* Bp, float* C, int ldc) {
    // 24 accumulators: row r, vector v (cols 4v..4v+3). Named, not an array, so they stay in registers.
    float32x4_t c00 = vdupq_n_f32(0), c01 = c00, c02 = c00, c10 = c00, c11 = c00, c12 = c00,
                c20 = c00, c21 = c00, c22 = c00, c30 = c00, c31 = c00, c32 = c00,
                c40 = c00, c41 = c00, c42 = c00, c50 = c00, c51 = c00, c52 = c00,
                c60 = c00, c61 = c00, c62 = c00, c70 = c00, c71 = c00, c72 = c00;

    for (int p = 0; p < kc; p++) {
        const float32x4_t b0 = vld1q_f32(Bp), b1 = vld1q_f32(Bp + 4), b2 = vld1q_f32(Bp + 8);
        const float32x4_t a0 = vld1q_f32(Ap), a1 = vld1q_f32(Ap + 4);   // A[0..3], A[4..7] of this column

        // row r += A[r] * (b0 b1 b2). The lane index must be a compile-time constant, hence the macro.
#define ROW(r, a, lane)                                  \
        c##r##0 = vfmaq_laneq_f32(c##r##0, b0, a, lane); \
        c##r##1 = vfmaq_laneq_f32(c##r##1, b1, a, lane); \
        c##r##2 = vfmaq_laneq_f32(c##r##2, b2, a, lane);
        ROW(0, a0, 0) ROW(1, a0, 1) ROW(2, a0, 2) ROW(3, a0, 3)
        ROW(4, a1, 0) ROW(5, a1, 1) ROW(6, a1, 2) ROW(7, a1, 3)
#undef ROW
        Ap += MR;
        Bp += NR;
    }

    // C += tile: one load + one store per C element for the whole kc loop.
#define STORE(r)                                                                  \
    vst1q_f32(C + r * ldc + 0, vaddq_f32(vld1q_f32(C + r * ldc + 0), c##r##0));  \
    vst1q_f32(C + r * ldc + 4, vaddq_f32(vld1q_f32(C + r * ldc + 4), c##r##1));  \
    vst1q_f32(C + r * ldc + 8, vaddq_f32(vld1q_f32(C + r * ldc + 8), c##r##2));
    STORE(0) STORE(1) STORE(2) STORE(3) STORE(4) STORE(5) STORE(6) STORE(7)
#undef STORE
}

#else  // portable fallback (x86 CI until the AVX2 version exists): same tile, plain C++

static void micro_kernel(int kc, const float* Ap, const float* Bp, float* C, int ldc) {
    float c[MR][NR] = {};
    for (int p = 0; p < kc; p++) {
        for (int r = 0; r < MR; r++) {
            const float a = Ap[r];
            for (int j = 0; j < NR; j++) c[r][j] += a * Bp[j];
        }
        Ap += MR;
        Bp += NR;
    }
    for (int r = 0; r < MR; r++)
        for (int j = 0; j < NR; j++) C[r * ldc + j] += c[r][j];
}

#endif

// Edge tiles (fewer than MR rows or NR cols left): run the full micro-kernel into
// a zeroed scratch tile, then add only the valid part into C. Packing zero-filled
// the missing rows/cols, so the extra lanes compute zeros and are discarded.
static void micro_kernel_edge(int kc, const float* Ap, const float* Bp,
                              float* C, int ldc, int m, int n) {
    float tile[MR * NR] = {};
    micro_kernel(kc, Ap, Bp, tile, NR);
    for (int r = 0; r < m; r++)
        for (int j = 0; j < n; j++) C[r * ldc + j] += tile[r * NR + j];
}

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    // Small-M dispatch: with fewer rows than the 8-row tile, most FMAs hit zero padding and
    // packing B costs as much as using it. Measured crossover (QKV shape, cold, see
    // results/stage3_dispatch.md): ikj wins for M < 8, simd wins from M = 8.
    if (M < MR) { matmul_ikj(A, B, C, M, N, K, lda, ldb, ldc); return; }
    // Buffers sized for the largest block this call will actually use (small for decode shapes).
    const int kc_max = std::min(K, KC);
    float* Ap = alloc_matrix(round_up(std::min(M, MC), MR), kc_max);
    float* Bp = alloc_matrix(round_up(std::min(N, NC), NR), kc_max);
    if (!Ap || !Bp) { free_matrix(Ap); free_matrix(Bp); return; }   // C unchanged; the test would catch it

    for (int jc = 0; jc < N; jc += NC) {
        const int nc = std::min(NC, N - jc);
        for (int pc = 0; pc < K; pc += KC) {
            const int kc = std::min(KC, K - pc);
            pack_B(B + pc * ldb + jc, ldb, kc, nc, Bp);

            for (int ic = 0; ic < M; ic += MC) {
                const int mc = std::min(MC, M - ic);
                pack_A(A + ic * lda + pc, lda, mc, kc, Ap);

                for (int jr = 0; jr < nc; jr += NR) {
                    const int n = std::min(NR, nc - jr);
                    const float* Bpanel = Bp + (jr / NR) * kc * NR;
                    for (int ir = 0; ir < mc; ir += MR) {
                        const int m = std::min(MR, mc - ir);
                        const float* Apanel = Ap + (ir / MR) * kc * MR;
                        float* Ct = C + (ic + ir) * ldc + (jc + jr);
                        if (m == MR && n == NR) micro_kernel(kc, Apanel, Bpanel, Ct, ldc);
                        else                    micro_kernel_edge(kc, Apanel, Bpanel, Ct, ldc, m, n);
                    }
                }
            }
        }
    }
    free_matrix(Ap);
    free_matrix(Bp);
}
