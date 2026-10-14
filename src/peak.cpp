// peak.cpp — single-core FP32 FMA throughput microbenchmark.
//
// Runs C independent chains of vector FMAs (acc_c += x * y), registers only,
// no memory traffic. With C = 1 every FMA waits on the previous one: that's FMA
// *latency*. As C grows, more FMAs are in flight until every pipe is busy every
// cycle: the plateau is the core's *throughput* peak, the real ceiling for GEMM.
//
// CSV -> stdout: chains,run,seconds,gflops

#include <chrono>
#include <cstdio>

#if defined(__APPLE__)
#include <pthread/qos.h>   // ask for a P-core
#endif

#if defined(__aarch64__)
#include <arm_neon.h>

constexpr long VEC_FMAS = 1L << 30;   // total vector FMAs per measurement, split across the chains

// One measurement with C chains. Returns seconds. The result is summed and returned
// through *sink so the compiler can't drop the work.
template <int C>
static double run(float* sink) {
    // x, y read through volatile so the compiler can't fold the chain into a closed form.
    volatile float xv = 1e-7f, yv = 1.0f;   // acc grows by 1e-7 per step: no overflow, no denormals
    const float32x4_t x = vdupq_n_f32(xv), y = vdupq_n_f32(yv);
    float32x4_t acc[C];
    for (int c = 0; c < C; c++) acc[c] = vdupq_n_f32(1.0f + c);   // distinct starts

    const long iters = VEC_FMAS / C;
    auto t0 = std::chrono::steady_clock::now();
    for (long i = 0; i < iters; i++) {
        // Accumulator must be the FIRST operand: fmla adds into its destination register.
        // Written as y + acc*x instead, every FMA needs an extra register copy, and at
        // 20+ chains the copies run out of registers and spill (measured: 2x slower).
#pragma clang loop unroll(full)
        for (int c = 0; c < C; c++) acc[c] = vfmaq_f32(acc[c], x, y);   // acc += x*y: one fmla, depends only on its own chain
    }
    auto t1 = std::chrono::steady_clock::now();

    float32x4_t s = acc[0];
    for (int c = 1; c < C; c++) s = vaddq_f32(s, acc[c]);
    *sink += vaddvq_f32(s);
    return std::chrono::duration<double>(t1 - t0).count();
}

template <int C>
static void report(int reps, float* sink) {
    const long iters = VEC_FMAS / C;
    const double flops = 2.0 * 4.0 * static_cast<double>(iters) * C;   // 2 flops x 4 lanes per vector FMA
    run<C>(sink);                                                       // warmup: clock ramp-up
    for (int r = 0; r < reps; r++) {
        double s = run<C>(sink);
        std::printf("%d,%d,%.6f,%.2f\n", C, r, s, flops / s / 1e9);
    }
}

int main() {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);   // hint: run on a performance core
#endif
    float sink = 0.0f;
    const int reps = 5;
    std::printf("chains,run,seconds,gflops\n");
    report<1>(reps, &sink);  report<2>(reps, &sink);  report<4>(reps, &sink);
    report<6>(reps, &sink);  report<8>(reps, &sink);  report<12>(reps, &sink);
    report<16>(reps, &sink); report<20>(reps, &sink); report<24>(reps, &sink);
    report<28>(reps, &sink); report<30>(reps, &sink); report<32>(reps, &sink);
    std::fprintf(stderr, "sink=%g\n", sink);
    return 0;
}

#else
int main() {
    std::fprintf(stderr, "peak: NEON only for now (x86 AVX2/FMA version not written yet)\n");
    return 0;
}
#endif
