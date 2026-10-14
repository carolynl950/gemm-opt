// tests/test_kernels.cpp — every kernel in IMPLS vs naive, on shapes built to break it
#include <cmath>     // NAN, std::isnan
#include <cstdio>
#include <cstring>   // std::memcpy
#include "alloc.hpp"
#include "check.hpp"
#include "kernels.hpp"

struct Shape { int M, N, K; };

static const Shape SHAPES[] = {
    // square awkward
    {1, 1, 1},
    {7, 7, 7},
    {63, 63, 63},
    {65, 65, 65},
    {129, 129, 129},

    // all-different
    {7, 13, 5},
    {65, 31, 129},

    // degenerate dims
    {1, 65, 31},     // M = 1 (decode-like)
    {65, 1, 31},     // N = 1
    {65, 31, 1},     // K = 1

    // block boundaries for blocked's defaults (MC=64, KC=64, NC=256)
    {64, 256, 64},    // exact multiples: one full block each way, no remainder
    {65, 257, 65},    // one past every block edge: remainder of 1 in all three loops
    {128, 512, 128},  // two full blocks each way: block loops actually iterate
};

// Second pass for blocked/packed with tiny, mutually prime tiles, so nearly every
// shape above hits a remainder in all three block loops.
static const int TINY_MC = 3, TINY_KC = 5, TINY_NC = 7;

static const int PAD = 3;   // extra columns per row; odd so rows aren't aligned

// True if every padding slot (cols [width, ld)) of every row still holds NaN.
static bool padding_intact(const float* X, int rows, int width, int ld) {
    for (int r = 0; r < rows; r++) {
        for (int c = width; c < ld; c++) {
            int i = idx(r, c, ld);
            if (!std::isnan(X[i])) return false;
        }
    }
    return true;
}

// Poison every slot, padding included; untouched padding must still be NaN after the kernel runs.
static void fill_nan(float* X, int count) {
    for (int i = 0; i < count; i++) {
        X[i] = NAN;
    }
}

// dst[i] = |src[i]| for all count slots. NaN padding stays NaN (fabs(NaN) is NaN).
static void abs_copy(float* dst, const float* src, int count) {
    for (int i = 0; i < count; i++) {
        dst[i] = std::fabs(src[i]);
    }
}

static bool run_case(const Impl& impl, Shape s) {
    // strides
    const int M = s.M, N = s.N, K = s.K;     // unpack the shape so you can write M instead of s.M
    const int lda = K + PAD;                 // A is M x K  -> its rows are K wide
    const int ldb = N + PAD;                 // B is K x N  -> its rows are N wide
    const int ldc = N + PAD;                 // C is M x N  -> its rows are N wide

    // alloc: (number of rows, stride)
    float* A     = alloc_matrix(M, lda);
    float* B     = alloc_matrix(K, ldb);
    float* C     = alloc_matrix(M, ldc);
    float* C_ref = alloc_matrix(M, ldc);
    if (!A || !B || !C || !C_ref) {
        std::printf("alloc failed M=%d N=%d K=%d\n", M, N, K);
        return false;
    }

    // poison ALL rows*ld slots; C_ref gets C's bytes (padding included) from the memcpy below
    fill_nan(A, M * lda);
    fill_nan(B, K * ldb);
    fill_nan(C, M * ldc);

    // fill the logical part only; padding stays NaN
    fill_random(A, M, K, lda, 1u);
    fill_random(B, K, N, ldb, 2u);
    fill_random(C, M, N, ldc, 3u);           // non-zero start: catches C = A*B instead of C += A*B

    // same starting C for both
    std::memcpy(C_ref, C, sizeof(float) * M * ldc);

    // Tolerance scale = |C0| + |A|·|B|: the magnitudes actually summed, before any cancellation.
    // Naive does C += A·B, so starting scale at |C0| and multiplying |A|·|B| into it gives exactly that.
    // All terms >= 0, so scale itself has no cancellation and its own rounding is negligible.
    float* absA  = alloc_matrix(M, lda);
    float* absB  = alloc_matrix(K, ldb);
    float* scale = alloc_matrix(M, ldc);
    if (!absA || !absB || !scale) {
        std::printf("alloc failed M=%d N=%d K=%d\n", M, N, K);
        return false;
    }
    abs_copy(absA,  A, M * lda);
    abs_copy(absB,  B, K * ldb);
    abs_copy(scale, C, M * ldc);              // must run before impl.fn changes C
    matmul_naive(absA, absB, scale, M, N, K, lda, ldb, ldc);

    matmul_naive(A, B, C_ref, M, N, K, lda, ldb, ldc);
    impl.fn(     A, B, C,     M, N, K, lda, ldb, ldc);

    bool values_ok  = matrices_match_bound(C_ref, C, scale, M, N, ldc, K);
    bool padding_ok = padding_intact(C, M, N, ldc)
                   && padding_intact(A, M, K, lda)   // kernels must not write their inputs either
                   && padding_intact(B, K, N, ldb);
    bool ok = values_ok && padding_ok;
    if (!ok) {
        std::printf("FAIL %s M=%d N=%d K=%d (%s)\n", impl.name, M, N, K,
                    values_ok ? "padding overwritten" : "values mismatch");
    }

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
    free_matrix(C_ref);
    free_matrix(absA);
    free_matrix(absB);
    free_matrix(scale);
    return ok;
}

int main() {
    int failures = 0, total = 0;
    for (const Impl& impl : IMPLS) {
        for (Shape s : SHAPES) {
            total++;
            if (!run_case(impl, s)) failures++;
        }
    }

    // Last, so it can't leak tiny tiles into the default-size pass above.
    set_block_sizes(TINY_MC, TINY_KC, TINY_NC);
    const Impl tiny[] = {{"blocked(3,5,7)", matmul_blocked}, {"packed(3,5,7)", matmul_packed}};
    for (const Impl& impl : tiny) {
        for (Shape s : SHAPES) {
            total++;
            if (!run_case(impl, s)) failures++;
        }
    }

    std::printf("test_kernels: %d/%d cases passed\n", total - failures, total);
    return failures ? 1 : 0;
}
