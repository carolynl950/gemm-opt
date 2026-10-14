// bench.cpp — Stage 1.4 benchmark harness
// CSV rows -> stdout:  implementation,size,run,seconds,gflops
// Progress, mismatches, checksums -> stderr (so `./bench > results.csv` stays clean)
//
// Usage: bench [--impl NAME] [--reps R] [--mc X --kc Y --nc Z] [n ...]
//   no sizes -> default sweep; no --impl -> every kernel in IMPLS

#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <vector>

#include "alloc.hpp"
#include "check.hpp"
#include "gemm.h"   // kernel declarations + the Kernel pointer type
#include "kernels.hpp"


static const int DEFAULT_SIZES[] = {128, 256, 512, 1024, 2048};
constexpr int DEFAULT_REPS = 5;


// Seconds for ONE call of fn.
// Zero C BEFORE t0 — makes the harness correct for Stage 2+ kernels that accumulate (C += ...).
//   auto t0 = std::chrono::steady_clock::now();
//   double s = std::chrono::duration<double>(t1 - t0).count();
static double time_one(Kernel fn, const float* A, const float* B, float* C, int n) {
    std::memset(C, 0, sizeof(float) * n * n);           // reset C outside the timed region; all-zero bits == 0.0f
    auto t0 = std::chrono::steady_clock::now();         // steady_clock: monotonic, never jumps if the wall clock is adjusted
    fn(A, B, C, n, n, n, n, n, n);                      // square, tightly packed: M = N = K = lda = ldb = ldc = n
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();  // duration<double> defaults to seconds, keeps sub-ns precision
}

// 2*n^3 flops (one mul + one add per inner-loop iteration).
// Compute in double — 2048^3 = 8.6e9 overflows a 32-bit int.
static double gflops(int n, double seconds) {
    double nd = static_cast<double>(n);        // convert first, so n*n*n is done in double, never in int
    double flops = 2.0 * nd * nd * nd;         // n^3 inner iterations x (1 mul + 1 add)
    return flops / seconds / 1e9;              // flops per second, scaled to giga
}

int main(int argc, char** argv) {
    const char* only = nullptr;              // --impl NAME: run just this kernel
    int reps = DEFAULT_REPS;
    std::vector<int> sizes;
    for (int a = 1; a < argc; a++) {
        if (!std::strcmp(argv[a], "--impl") && a + 1 < argc)      only = argv[++a];
        else if (!std::strcmp(argv[a], "--reps") && a + 1 < argc) reps = std::atoi(argv[++a]);
        else if (!std::strcmp(argv[a], "--mc") && a + 1 < argc)   set_block_sizes(std::atoi(argv[++a]), 0, 0);
        else if (!std::strcmp(argv[a], "--kc") && a + 1 < argc)   set_block_sizes(0, std::atoi(argv[++a]), 0);
        else if (!std::strcmp(argv[a], "--nc") && a + 1 < argc)   set_block_sizes(0, 0, std::atoi(argv[++a]));
        else if (std::atoi(argv[a]) > 0)                          sizes.push_back(std::atoi(argv[a]));
        else { std::fprintf(stderr, "usage: bench [--impl NAME] [--reps R] [--mc X --kc Y --nc Z] [n ...]\n"); return 2; }
    }
    if (sizes.empty()) sizes.assign(std::begin(DEFAULT_SIZES), std::end(DEFAULT_SIZES));
    if (only) {                                                  // fail loudly on a typo instead of printing an empty CSV
        bool found = false;
        for (const Impl& impl : IMPLS) found = found || !std::strcmp(impl.name, only);
        if (!found) { std::fprintf(stderr, "unknown --impl %s\n", only); return 2; }
    }

    std::printf("implementation,size,run,seconds,gflops\n");

    for (int n : sizes) {
        // Square matrices, tightly packed: ld = n
        float* A     = alloc_matrix(n, n);
        float* B     = alloc_matrix(n, n);
        float* C     = alloc_matrix(n, n);
        float* C_ref = alloc_matrix(n, n);
        if (!A || !B || !C || !C_ref) {                         // posix_memalign failed (2048^2 floats = 16 MB each)
            std::fprintf(stderr, "alloc failed at n=%d\n", n);
            return 1;                                           // free(nullptr) is a no-op, but just exit; the OS reclaims the rest
        }

        // Different seeds -> A != B, so a kernel that swaps or transposes operands gives a wrong answer and gets caught
        fill_random(A, n, n, n, 1u);
        fill_random(B, n, n, n, 2u);

        // Reference answer: naive kernel, computed once, not timed. time_one also zeroes C_ref first.
        time_one(matmul_naive, A, B, C_ref, n);
        std::fprintf(stderr, "n=%d: reference computed\n", n);

        for (const Impl& impl : IMPLS) {
            if (only && std::strcmp(impl.name, only)) continue;
            // Correctness gate + warmup in one call: warms caches/TLB, faults in pages, and checks the answer.
            // The time is thrown away.
            time_one(impl.fn, A, B, C, n);
            if (!matrices_match(C_ref, C, n, n, n, /*K=*/n)) {  // K = n accumulations per output cell -> tolerance scale
                std::fprintf(stderr, "%s n=%d: MISMATCH, skipping timing\n", impl.name, n);
                continue;                                       // never write timing rows for a wrong kernel
            }

            // Timed runs: one CSV row each. Keep every run (not just the min/median) so the analysis can pick later.
            for (int run = 0; run < reps; run++) {
                double s = time_one(impl.fn, A, B, C, n);
                std::printf("%s,%d,%d,%.9f,%.4f\n", impl.name, n, run, s, gflops(n, s));
            }

            // Checksum of the last run's C. Printing it makes the result observable, so the
            // optimizer can't prove the kernel's writes are unused and delete the calls.
            double checksum = 0.0;                              // double: summing 4M floats in float would lose precision
            for (int i = 0; i < n * n; i++) checksum += C[i];
            std::fprintf(stderr, "%s n=%d: checksum=%.6f\n", impl.name, n, checksum);
        }

        free_matrix(A);
        free_matrix(B);
        free_matrix(C);
        free_matrix(C_ref);
    }
    return 0;
}