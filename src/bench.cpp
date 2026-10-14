// bench.cpp — benchmark harness
// CSV rows -> stdout:  implementation,m,n,k,run,seconds,gflops
// Progress, mismatches, checksums -> stderr (so `./bench > results.csv` stays clean)
//
// Usage: bench [--impl NAME] [--reps R] [--mc X --kc Y --nc Z] [--cold]
//              [--shape MxNxK ...] [n ...]
//   n          square n x n x n
//   --shape    C (M x N) += A (M x K) * B (K x N); repeatable
//   --cold     evict caches before every timed run (weights come from DRAM, like real decode)
//   no sizes/shapes -> default square sweep; no --impl -> every kernel in IMPLS
//
// Results from before the --shape change have header implementation,size,run,seconds,gflops.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <vector>

#include "alloc.hpp"
#include "check.hpp"
#include "gemm.h"   // kernel declarations + the Kernel pointer type
#include "kernels.hpp"

struct Shape { int M, N, K; };

static const int DEFAULT_SIZES[] = {128, 256, 512, 1024, 2048};
constexpr int DEFAULT_REPS = 5;

// Bigger than every cache level on the dev machine (16 MB L2 on M4 Pro), so
// touching it evicts A, B and C. Written, not just read, so lines are owned
// by this buffer rather than shared with the matrices.
constexpr size_t FLUSH_BYTES = 64u << 20;
static std::vector<char> g_flush;
static volatile char g_flush_sink;          // keeps the flush loop from being optimized away

static void flush_caches() {
    if (g_flush.empty()) g_flush.assign(FLUSH_BYTES, 0);
    for (size_t i = 0; i < g_flush.size(); i += 64) g_flush[i] += 1;
    g_flush_sink = g_flush[g_flush.size() / 2];
}

// Seconds for ONE call of fn.
// Zero C BEFORE t0 — makes the harness correct for kernels that accumulate (C += ...).
static double time_one(Kernel fn, const float* A, const float* B, float* C, Shape s, bool cold) {
    std::memset(C, 0, sizeof(float) * s.M * s.N);       // reset C outside the timed region; all-zero bits == 0.0f
    if (cold) flush_caches();                           // after the memset, which would otherwise re-warm C
    auto t0 = std::chrono::steady_clock::now();         // steady_clock: monotonic, never jumps if the wall clock is adjusted
    fn(A, B, C, s.M, s.N, s.K, s.K, s.N, s.N);          // tightly packed: lda = K, ldb = ldc = N
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

// 2*M*N*K flops (one mul + one add per inner-loop iteration). In double: 2048^3 overflows int.
static double gflops(Shape s, double seconds) {
    double flops = 2.0 * s.M * static_cast<double>(s.N) * s.K;
    return flops / seconds / 1e9;
}

int main(int argc, char** argv) {
    const char* only = nullptr;              // --impl NAME: run just this kernel
    int reps = DEFAULT_REPS;
    bool cold = false;
    std::vector<Shape> shapes;
    for (int a = 1; a < argc; a++) {
        int m, n, k;
        if (!std::strcmp(argv[a], "--impl") && a + 1 < argc)      only = argv[++a];
        else if (!std::strcmp(argv[a], "--reps") && a + 1 < argc) reps = std::atoi(argv[++a]);
        else if (!std::strcmp(argv[a], "--mc") && a + 1 < argc)   set_block_sizes(std::atoi(argv[++a]), 0, 0);
        else if (!std::strcmp(argv[a], "--kc") && a + 1 < argc)   set_block_sizes(0, std::atoi(argv[++a]), 0);
        else if (!std::strcmp(argv[a], "--nc") && a + 1 < argc)   set_block_sizes(0, 0, std::atoi(argv[++a]));
        else if (!std::strcmp(argv[a], "--cold"))                 cold = true;
        else if (!std::strcmp(argv[a], "--shape") && a + 1 < argc
                 && std::sscanf(argv[a + 1], "%dx%dx%d", &m, &n, &k) == 3 && m > 0 && n > 0 && k > 0) {
            shapes.push_back({m, n, k}); a++;
        }
        else if (std::atoi(argv[a]) > 0) { int s = std::atoi(argv[a]); shapes.push_back({s, s, s}); }
        else {
            std::fprintf(stderr, "usage: bench [--impl NAME] [--reps R] [--mc X --kc Y --nc Z] [--cold] [--shape MxNxK ...] [n ...]\n");
            return 2;
        }
    }
    if (shapes.empty()) for (int s : DEFAULT_SIZES) shapes.push_back({s, s, s});
    if (only) {                                                  // fail loudly on a typo instead of printing an empty CSV
        bool found = false;
        for (const Impl& impl : IMPLS) found = found || !std::strcmp(impl.name, only);
        if (!found) { std::fprintf(stderr, "unknown --impl %s\n", only); return 2; }
    }

    std::printf("implementation,m,n,k,run,seconds,gflops\n");

    for (Shape s : shapes) {
        const int M = s.M, N = s.N, K = s.K;
        float* A     = alloc_matrix(M, K);
        float* B     = alloc_matrix(K, N);
        float* C     = alloc_matrix(M, N);
        float* C_ref = alloc_matrix(M, N);
        float* absA  = alloc_matrix(M, K);
        float* absB  = alloc_matrix(K, N);
        float* scale = alloc_matrix(M, N);
        if (!A || !B || !C || !C_ref || !absA || !absB || !scale) {
            std::fprintf(stderr, "alloc failed at %dx%dx%d\n", M, N, K);
            return 1;                                           // the OS reclaims the rest
        }

        // Different seeds -> A != B, so a kernel that swaps or transposes operands gives a wrong answer and gets caught
        fill_random(A, M, K, K, 1u);
        fill_random(B, K, N, N, 2u);

        // Reference answer + tolerance scale, computed once, not timed.
        // scale = |A|·|B| (C starts at 0): the magnitudes summed, so the check survives cancellation (see NOTES.md).
        time_one(matmul_naive, A, B, C_ref, s, false);
        for (int i = 0; i < M * K; i++) absA[i] = std::fabs(A[i]);
        for (int i = 0; i < K * N; i++) absB[i] = std::fabs(B[i]);
        time_one(matmul_naive, absA, absB, scale, s, false);
        std::fprintf(stderr, "%dx%dx%d: reference computed\n", M, N, K);

        for (const Impl& impl : IMPLS) {
            if (only && std::strcmp(impl.name, only)) continue;
            // Correctness gate + warmup in one call: faults in pages, warms TLB, and checks the answer.
            time_one(impl.fn, A, B, C, s, false);
            if (!matrices_match_bound(C_ref, C, scale, M, N, N, K)) {
                std::fprintf(stderr, "%s %dx%dx%d: MISMATCH, skipping timing\n", impl.name, M, N, K);
                continue;                                       // never write timing rows for a wrong kernel
            }

            // Timed runs: one CSV row each. Keep every run so the analysis can pick later.
            for (int run = 0; run < reps; run++) {
                double secs = time_one(impl.fn, A, B, C, s, cold);
                std::printf("%s,%d,%d,%d,%d,%.9f,%.4f\n", impl.name, M, N, K, run, secs, gflops(s, secs));
            }

            // Checksum of the last run's C: makes the result observable, so the
            // optimizer can't prove the kernel's writes are unused and delete the calls.
            double checksum = 0.0;
            for (int i = 0; i < M * N; i++) checksum += C[i];
            std::fprintf(stderr, "%s %dx%dx%d: checksum=%.6f\n", impl.name, M, N, K, checksum);
        }

        free_matrix(A);
        free_matrix(B);
        free_matrix(C);
        free_matrix(C_ref);
        free_matrix(absA);
        free_matrix(absB);
        free_matrix(scale);
    }
    return 0;
}
