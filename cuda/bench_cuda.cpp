// bench_cuda.cpp — GPU benchmark harness. Same CLI and CSV as src/bench.cpp.
// CSV rows -> stdout:  implementation,m,n,k,run,seconds,gflops
// Progress, mismatches, checksums -> stderr
//
// Usage: bench_cuda [--impl NAME] [--reps R] [--cold] [--warmup-ms W] [--shape MxNxK ...] [n ...]
//   --cold   overwrite a 256 MB device buffer before each timed run, evicting L2
//            (T4 L2 = 4 MB), so weights come from DRAM like real decode
//   --warmup-ms W   run each kernel untimed for at least W ms before timing (default 300):
//            GPU clocks ramp up from idle, and one warmup run isn't enough (measured on T4:
//            naive at n=1024 went 44.6 -> 60.8 GFLOPS over its first 5 timed runs)
//
// Timing: CUDA events around the kernel launch on the GPU's own clock. C is reset with
// cudaMemset outside the timed region. One untimed warmup per kernel per shape also
// serves as the correctness gate (vs cuBLAS, magnitude-based tolerance).
// Exits 1 if any kernel fails its gate.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

#include "check.hpp"
#include "kernels.hpp"     // fill_random
#include "gemm_cuda.cuh"

struct Shape { int M, N, K; };

static const int DEFAULT_SIZES[] = {256, 512, 1024, 2048, 4096};
constexpr int DEFAULT_REPS = 10;
constexpr size_t FLUSH_BYTES = 256u << 20;

static float* device_copy(const std::vector<float>& h) {
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

static double gflops(Shape s, double seconds) {
    return 2.0 * s.M * static_cast<double>(s.N) * s.K / seconds / 1e9;
}

int main(int argc, char** argv) {
    const char* only = nullptr;
    int reps = DEFAULT_REPS;
    bool cold = false;
    float warmup_ms = 300.0f;
    std::vector<Shape> shapes;
    for (int a = 1; a < argc; a++) {
        int m, n, k;
        if (!std::strcmp(argv[a], "--impl") && a + 1 < argc)      only = argv[++a];
        else if (!std::strcmp(argv[a], "--reps") && a + 1 < argc) reps = std::atoi(argv[++a]);
        else if (!std::strcmp(argv[a], "--cold"))                 cold = true;
        else if (!std::strcmp(argv[a], "--warmup-ms") && a + 1 < argc) warmup_ms = static_cast<float>(std::atof(argv[++a]));
        else if (!std::strcmp(argv[a], "--shape") && a + 1 < argc
                 && std::sscanf(argv[a + 1], "%dx%dx%d", &m, &n, &k) == 3 && m > 0 && n > 0 && k > 0) {
            shapes.push_back({m, n, k}); a++;
        }
        else if (std::atoi(argv[a]) > 0) { int s = std::atoi(argv[a]); shapes.push_back({s, s, s}); }
        else {
            std::fprintf(stderr, "usage: bench_cuda [--impl NAME] [--reps R] [--cold] [--warmup-ms W] [--shape MxNxK ...] [n ...]\n");
            return 2;
        }
    }
    if (shapes.empty()) for (int s : DEFAULT_SIZES) shapes.push_back({s, s, s});
    if (only) {
        bool found = false;
        for (const CudaImpl& impl : CUDA_IMPLS) found = found || !std::strcmp(impl.name, only);
        if (!found) { std::fprintf(stderr, "unknown --impl %s\n", only); return 2; }
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::fprintf(stderr, "GPU: %s (sm_%d%d), %d SMs, L2 %d KB\n",
                 prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.l2CacheSize / 1024);

    float* flush = nullptr;
    if (cold) CUDA_CHECK(cudaMalloc(&flush, FLUSH_BYTES));
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    std::printf("implementation,m,n,k,run,seconds,gflops\n");
    int mismatches = 0;

    for (Shape s : shapes) {
        const int M = s.M, N = s.N, K = s.K;
        const size_t a_n = static_cast<size_t>(M) * K, b_n = static_cast<size_t>(K) * N, c_n = static_cast<size_t>(M) * N;
        const size_t c_bytes = c_n * sizeof(float);

        std::vector<float> A(a_n), B(b_n), absA(a_n), absB(b_n);
        fill_random(A.data(), M, K, K, 1u);
        fill_random(B.data(), K, N, N, 2u);
        for (size_t i = 0; i < a_n; i++) absA[i] = std::fabs(A[i]);
        for (size_t i = 0; i < b_n; i++) absB[i] = std::fabs(B[i]);

        float* dA = device_copy(A);
        float* dB = device_copy(B);
        float* dAbsA = device_copy(absA);
        float* dAbsB = device_copy(absB);
        float* dC = nullptr;
        float* dRef = nullptr;
        CUDA_CHECK(cudaMalloc(&dC, c_bytes));
        CUDA_CHECK(cudaMalloc(&dRef, c_bytes));

        // Reference (cuBLAS) and tolerance scale = |A|·|B| (C starts at 0), computed once, not timed.
        std::vector<float> ref(c_n), scale(c_n), got(c_n);
        CUDA_CHECK(cudaMemset(dRef, 0, c_bytes));
        cuda_cublas(dA, dB, dRef, M, N, K, K, N, N);
        CUDA_CHECK(cudaMemcpy(ref.data(), dRef, c_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemset(dRef, 0, c_bytes));
        cuda_cublas(dAbsA, dAbsB, dRef, M, N, K, K, N, N);
        CUDA_CHECK(cudaMemcpy(scale.data(), dRef, c_bytes, cudaMemcpyDeviceToHost));
        std::fprintf(stderr, "%dx%dx%d: reference computed\n", M, N, K);

        for (const CudaImpl& impl : CUDA_IMPLS) {
            if (only && std::strcmp(impl.name, only)) continue;

            // Warmup + correctness gate
            CUDA_CHECK(cudaMemset(dC, 0, c_bytes));
            impl.fn(dA, dB, dC, M, N, K, K, N, N);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(got.data(), dC, c_bytes, cudaMemcpyDeviceToHost));
            if (!matrices_match_bound(ref.data(), got.data(), scale.data(), M, N, N, K)) {
                std::fprintf(stderr, "%s %dx%dx%d: MISMATCH, skipping timing\n", impl.name, M, N, K);
                mismatches++;
                continue;
            }

            // Timed warmup: keep running until the GPU has been busy for warmup_ms, so the
            // clock has ramped up before the first timed run.
            {
                float busy_ms = 0.0f;
                while (busy_ms < warmup_ms) {
                    float ms = 0.0f;
                    CUDA_CHECK(cudaEventRecord(t0));
                    impl.fn(dA, dB, dC, M, N, K, K, N, N);
                    CUDA_CHECK(cudaEventRecord(t1));
                    CUDA_CHECK(cudaEventSynchronize(t1));
                    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
                    busy_ms += ms;
                }
            }

            for (int run = 0; run < reps; run++) {
                CUDA_CHECK(cudaMemset(dC, 0, c_bytes));                       // outside the timed region
                if (cold) CUDA_CHECK(cudaMemset(flush, run & 0xff, FLUSH_BYTES));
                CUDA_CHECK(cudaEventRecord(t0));
                impl.fn(dA, dB, dC, M, N, K, K, N, N);
                CUDA_CHECK(cudaEventRecord(t1));
                CUDA_CHECK(cudaEventSynchronize(t1));
                CUDA_CHECK(cudaGetLastError());
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
                const double secs = ms / 1e3;
                std::printf("%s,%d,%d,%d,%d,%.9f,%.4f\n", impl.name, M, N, K, run, secs, gflops(s, secs));
            }

            // Checksum of the last run: makes the result observable (and catches a kernel
            // that passes the gate once but misbehaves on later runs).
            CUDA_CHECK(cudaMemcpy(got.data(), dC, c_bytes, cudaMemcpyDeviceToHost));
            double checksum = 0.0;
            for (size_t i = 0; i < c_n; i++) checksum += got[i];
            std::fprintf(stderr, "%s %dx%dx%d: checksum=%.6f\n", impl.name, M, N, K, checksum);
        }

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dAbsA));
        CUDA_CHECK(cudaFree(dAbsB));
        CUDA_CHECK(cudaFree(dC));
        CUDA_CHECK(cudaFree(dRef));
    }
    if (flush) CUDA_CHECK(cudaFree(flush));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return mismatches ? 1 : 0;
}
