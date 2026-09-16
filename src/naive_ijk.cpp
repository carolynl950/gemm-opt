#include "alloc.hpp"
#include "check.hpp"

void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K){
    for (int i = 0; i < M; i++) {
        const float* a_row = A + i * K;
        float* c_row = C + i * N;

        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a_row[k] * B[k * N + j];
            }
            c_row[j] = sum;
        }
    }
}