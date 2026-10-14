# CLAUDE.md: working on GEMM-opt

## What this project is

A stage-by-stage single-precision GEMM optimization in C++ (naive → loop reorder → cache blocking → AVX2 → OpenMP → vendor-BLAS comparison). Its value is **measured, attributed speedups**: every stage must explain *why* it's faster using hardware counters, not just show a lower time.

It's a portfolio project for hardware/performance roles (Amazon Annapurna, NVIDIA) and ML-systems/MLE roles. The ML angle: benchmarking on real transformer layer shapes (prefill vs decode).

See `README.md` for stages and status, and `NOTES.md` for per-stage findings.

## How I want to work with you

I'm writing this code myself to learn it. **Do not hand me finished implementations.**

For each new step:
1. **Concepts first, as bullets.** What the technique is, why it helps, which hardware behavior it changes.
2. **Then a skeleton or pseudocode with `// TODO`s** for me to fill in.
3. **Then review my attempt.** Point to specific bugs and the line they're on, edge cases I missed, and anything that would make my measurements wrong.

Style:
- Bullets over prose. Direct and terse.
- If I'm wrong, say so plainly.
- Don't pad with encouragement.

## Rules for this repo

- **Numbers must be real.** Never invent benchmark results. README and resume bullets use `[X]` placeholders until I've measured.
- **Correctness before speed.** Every kernel must pass `ctest` (vs naive, K-scaled tolerance, awkward sizes, non-square shapes, padded strides, non-zero initial C) before it gets benchmarked.
- **Floating point is load-bearing.** Tolerance must scale with K. Summation-order differences between kernels are expected and measurable; don't "fix" them by loosening tolerance blindly.
- **Benchmark hygiene:** fixed seed, setup outside the timer, warmup run (doubles as correctness gate), multiple reps, observable checksum against dead-code elimination. For counter runs, use one shape + one kernel per process (`bench --impl X --shape MxNxK`). Decode-like shapes need `--cold`.
- **Platform split.** Mac M4 (ARM64) is fine for Stages 0–2. AVX2, OpenMP, and final numbers need x86-64. CI's Ubuntu runner can check AVX2 *correctness* but not timing. `perf` is Linux-only.
- **BLAS reference** is Accelerate (AMX) on macOS, OpenBLAS on Linux. Always label which one a number came from.
- **Build system is CMake** with presets `release`, `asan`, `ubsan` (Ninja, `build/<preset>/`). Full check: `cmake --build --preset <p> && ctest --preset <p>`. `ctest` does not rebuild. New kernel → add to `add_library(gemm ...)`; new test → add to the `foreach(t ...)` list.
- **Mechanism in one sentence.** Each stage gets a one-line "why it's faster" in `NOTES.md` before moving on.

## Current status

- Done: infrastructure (CMake presets incl. `novec`, `test_kernels` with magnitude-based tolerance, bench CLI with `--shape`/`--cold`, CI), Stage 0 naive, Stage 1 `ikj`, Stage 2a blocked, Stage 2b packed, transformer prefill/decode sweep, Stage 3 NEON micro-kernel (`simd`), FMA peak microbenchmark (`peak`, ~135 GFLOPS/core). Each has a write-up in `results/*.md`.
- Next: small-M dispatch → Stage 4 OpenMP (re-measure decode bandwidth wall) → Stage 3 AVX2 on x86 → x86 numbers → counter confirmation of the set-conflict / prefetcher hypotheses
- Decided: Stage 3 is NEON on M4 first, AVX2 later. Open: Mac counter tool (Instruments/xctrace) or x86-only attribution?

Update this section as stages finish.

## Interview framing to keep in mind

For each stage, I should be able to answer:
- What bottleneck did this remove?
- Which counter proves it?
- Where does it stop helping, and why? (e.g., blocking barely helps at decode T=1 because it's memory-bandwidth-bound)