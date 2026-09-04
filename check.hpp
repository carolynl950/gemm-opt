#pragma once

#include "alloc.hpp"
#include <cfloat>   // FLT_EPSILON
#include <cmath>    // std::fabs
#include <cstdio>   // printf

// Compare two result matrices for equality within floating-point tolerance
//
// Why not just use ==:
//   Float addition rounds, and rounding is order-dependent. Every stage of this
//   project reorders the summation, so two correct kernels produce results that
//   differ in the last bits. Exact comparison would fail on working code.
//
//   ref  = the trusted result (naive kernel)
//   test = the result being checked
//   ld   = row stride in memory (how far apart rows sit)
//   K    = accumulations per element; drives how much drift to allow
inline bool matrices_match(const float* ref, const float* test,
                           int rows, int cols, int ld, int K) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int i = idx(r, c, ld);

            // Tolerance, per element:
            //   FLT_EPSILON  - relative size of one rounding step (~1.19e-7)
            //   * fabs(ref)  - scale it to this element's magnitude, since
            //                  rounding error is relative, not absolute
            //   K *          - K accumulations, so K roundings can stack
            //   + 1e-6f      - floor, so elements that are exactly 0 don't
            //                  get a tolerance of 0 and fail on any difference
            float tol = K * FLT_EPSILON * std::fabs(ref[i]) + 1e-6f;

            if (std::fabs(ref[i] - test[i]) > tol) {
                printf("mismatch at (%d, %d): ref=%g test=%g diff=%g tol=%g\n",
                       r, c, ref[i], test[i],
                       std::fabs(ref[i] - test[i]), tol);
                return false;   // stop at first failure
            }
        }
    }
    return true;
}