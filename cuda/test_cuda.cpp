// test_cuda.cpp — every CUDA kernel vs the CPU naive reference, same checks as tests/test_kernels.cpp:
// awkward shapes, padded strides, NaN-poisoned padding, random starting C, magnitude-based tolerance.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "alloc.hpp"
#include "check.hpp"
#include "gemm.h"
#include "kernels.hpp"     // fill_random
#include "gemm_cuda.cuh"

struct Shape { int M, N, K; };

static const Shape SHAPES[] = {
    {1, 1, 1}, {7, 7, 7}, {31, 31, 31}, {32, 32, 32}, {33, 33, 33},   // around a 32-wide warp / tile
    {63, 63, 63}, {64, 64, 64}, {65, 65, 65}, {129, 129, 129},
    {7, 13, 5}, {65, 31, 129}, {33, 130, 67},                          // all different
    {1, 2304, 768},                                                    // decode-shaped (M = 1)
    {65, 1, 31}, {65, 31, 1},                                          // N = 1, K = 1
    {128, 128, 128}, {257, 255, 129},                                  // around 64/128-wide block tiles
    {129, 127, 9}, {256, 384, 17},                                     // regtile: partial 128-tiles, K not a multiple of 8
};

static const int PAD = 3;   // extra columns per row; odd so rows aren't aligned

static bool padding_intact(const float* X, int rows, int width, int ld) {
    for (int r = 0; r < rows; r++)
        for (int c = width; c < ld; c++)
            if (!std::isnan(X[idx(r, c, ld)])) return false;
    return true;
}

// Host buffer of rows x ld floats, all NaN; caller fills the logical part.
static std::vector<float> nan_buffer(int rows, int ld) {
    return std::vector<float>(static_cast<size_t>(rows) * ld, NAN);
}

static float* to_device(const std::vector<float>& h) {
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

static bool run_case(const CudaImpl& impl, Shape s) {
    const int M = s.M, N = s.N, K = s.K;
    const int lda = K + PAD, ldb = N + PAD, ldc = N + PAD;

    std::vector<float> A = nan_buffer(M, lda), B = nan_buffer(K, ldb), C = nan_buffer(M, ldc);
    fill_random(A.data(), M, K, lda, 1u);
    fill_random(B.data(), K, N, ldb, 2u);
    fill_random(C.data(), M, N, ldc, 3u);   // non-zero start: catches C = A*B instead of C += A*B

    // CPU reference + tolerance scale (|C0| + |A||B|), as in test_kernels
    std::vector<float> C_ref = C, absA(A.size()), absB(B.size()), scale(C.size());
    for (size_t i = 0; i < A.size(); i++) absA[i] = std::fabs(A[i]);
    for (size_t i = 0; i < B.size(); i++) absB[i] = std::fabs(B[i]);
    for (size_t i = 0; i < C.size(); i++) scale[i] = std::fabs(C[i]);
    matmul_naive(A.data(), B.data(), C_ref.data(), M, N, K, lda, ldb, ldc);
    matmul_naive(absA.data(), absB.data(), scale.data(), M, N, K, lda, ldb, ldc);

    // GPU: copy whole buffers (padding included), run, copy C back
    float* dA = to_device(A);
    float* dB = to_device(B);
    float* dC = to_device(C);
    impl.fn(dA, dB, dC, M, N, K, lda, ldb, ldc);
    CUDA_CHECK(cudaGetLastError());          // launch errors (bad grid/block config)
    CUDA_CHECK(cudaDeviceSynchronize());     // errors during execution (out-of-bounds, etc.)
    CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> A_after(A.size()), B_after(B.size());
    CUDA_CHECK(cudaMemcpy(A_after.data(), dA, A.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(B_after.data(), dB, B.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    bool values_ok  = matrices_match_bound(C_ref.data(), C.data(), scale.data(), M, N, ldc, K);
    bool padding_ok = padding_intact(C.data(), M, N, ldc)
                   && padding_intact(A_after.data(), M, K, lda)
                   && padding_intact(B_after.data(), K, N, ldb);
    bool ok = values_ok && padding_ok;
    if (!ok) {
        std::printf("FAIL %s M=%d N=%d K=%d (%s)\n", impl.name, M, N, K,
                    values_ok ? "padding overwritten" : "values mismatch");
    }
    return ok;
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    int failures = 0, total = 0;
    for (const CudaImpl& impl : CUDA_IMPLS) {
        for (Shape s : SHAPES) {
            total++;
            if (!run_case(impl, s)) failures++;
        }
    }
    std::printf("test_cuda: %d/%d cases passed\n", total - failures, total);
    return failures ? 1 : 0;
}
