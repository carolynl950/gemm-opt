#pragma once
#include "alloc.hpp"   // idx(), used by fill_random
#include "gemm.h"
#include <random>

struct Impl {
    const char* name; 
    Kernel fn;
};

inline constexpr Impl IMPLS[] = { //inline gives one shared def
    {"naive_ijk", matmul_naive},
    {"ikj", matmul_ikj},
    {"blocked", matmul_blocked},
    {"packed", matmul_packed},
    {"simd", matmul_simd},
#ifdef GEMM_HAVE_BLAS
    {"blas", matmul_blas},
#endif
};

inline void fill_random(float* X, int rows, int cols, int ld, unsigned seed) {
    std::mt19937 rng(seed);                                   // same seed -> same sequence every run, so results are reproducible
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);  // small, signed values: sums stay well-scaled and cancellation gets exercised
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            X[idx(r, c, ld)] = dist(rng);                     // row-major walk -> sequential writes; ld (not cols) is the row stride
        }
    }
}