// cuBLAS reference: NVIDIA's tuned SGEMM. The GPU counterpart of src/kernels/blas.cpp.
#include "gemm_cuda.cuh"
#include <cublas_v2.h>

static cublasHandle_t handle() {
    // One handle for the whole process: creating one costs milliseconds and would
    // land inside the timed region if done per call.
    static cublasHandle_t h = [] {
        cublasHandle_t x;
        if (cublasCreate(&x) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "cublasCreate failed\n");
            std::exit(1);
        }
        // Plain FP32 math, no TF32 tensor cores: same arithmetic as our kernels.
        // (T4 has no TF32 anyway; this keeps results comparable on newer GPUs.)
        cublasSetMathMode(x, CUBLAS_DEFAULT_MATH);
        return x;
    }();
    return h;
}

void cuda_cublas(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    const float one = 1.0f;   // alpha = beta = 1  ->  C = A*B + C, the += contract
    // cuBLAS is column-major. A row-major matrix read as column-major is its transpose,
    // so row-major C = A*B is column-major C^T = B^T * A^T: pass B first, swap M and N,
    // and no explicit transposes are needed.
    cublasStatus_t st = cublasSgemm(handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                                    N, M, K,
                                    &one, B, ldb, A, lda,
                                    &one, C, ldc);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasSgemm failed: status %d\n", static_cast<int>(st));
        std::exit(1);
    }
}
