# GEMM-opt

Single-precision matrix multiply (`C = A · B`) in C++, optimized stage by stage from a naive triple loop toward library performance, with every speedup measured and attributed to hardware behavior (cache misses, vectorization, threading).

The goal is not just a fast kernel. Each stage answers a question: *what changed in the hardware's behavior, and how do the numbers prove it?*

## Why GEMM

Matrix multiply dominates ML training and inference. Every transformer layer is mostly GEMMs (QKV projection, attention output, MLP up and down). Making it fast exercises the full memory hierarchy: registers, L1/L2/L3, DRAM bandwidth, SIMD units, and cores.

## Stages

| Stage | Kernel | Idea | Status |
|---|---|---|---|
| 0 | `naive` (ijk) | Textbook triple loop; correctness reference | Done |
| 1 | `ikj` | Loop reorder so the inner loop streams B and C rows contiguously | Done ([analysis](results/stage1_ikj.md)) |
| 2 | `blocked` | Cache blocking (tiling) with runtime block size and remainder handling | Done: [2a blocking](results/stage2_blocked.md), [2b packing](results/stage2b_packed.md) |
| 3 | `simd` | Packed A + B, 8×12 register-blocked micro-kernel: NEON (M4) done, AVX2 (x86-64) planned | NEON done ([analysis](results/stage3_simd.md)) |
| 4 | `omp` | OpenMP multithreading over blocks | Planned |
| — | `blas` | Vendor BLAS reference ceiling (Accelerate on macOS, OpenBLAS on Linux) | Wired up |

Each stage records its mechanism (one sentence: *why* it's faster) and its measured effect in `NOTES.md`.

## Results

> Numbers are filled in only after measurement. Bracketed values are placeholders.

| Kernel | GFLOPS (N=1024) | Speedup vs naive | Key counter change |
|---|---|---|---|
| naive | 2.44 | 1.0× | — |
| ikj | 32.7 | 13.4× | [X] (split measured: ~3.2× access pattern × ~4.2× SIMD) |
| blocked (64/64/256) | 29.6 | 12.1× | [X] (slower than ikj at 1024: power-of-two set conflicts; +34% at n=1000) |
| packed (64/64/256) | 41.9 | 17.2× | [X] (packing removes the power-of-two cliff) |
| simd (NEON 8×12) | 115.3 | 47× | [X] (~0.2 memory ops per FMA vs 3 in Stages 1–2); ~85% of measured 135 GFLOPS core peak ([peak](results/peak_fma.md)) |
| omp | [X] | [X]× | [X] |
| BLAS (Accelerate, AMX) | 2164 | 887× | — |

Hardware (current numbers): Apple M4 Pro, single thread, 128 KB L1d / 16 MB L2 / 128 B lines; Apple clang 17, `-O3 -mcpu=native`; BLAS = Accelerate. Median of 5 reps. Final x86 numbers: [X]

On Apple silicon the BLAS reference is Accelerate, which uses the AMX matrix units. It is reported separately and is not a target for the hand-written kernels.

### Transformer workloads

GPT-2-small layer shapes (d=768, MLP=3072), prefill (T=512) and decode (T=1), single thread on M4 Pro. Decode is measured **cold** (caches flushed before each run), since a real model's weights don't fit in cache. Full analysis: [results/transformer_shapes.md](results/transformer_shapes.md).

| Layer | Shape (M×K · K×N) | Prefill: packed | Prefill: simd | Prefill: Accelerate | Decode: ikj | Decode: simd | Decode: Accelerate |
|---|---|---|---|---|---|---|---|
| QKV projection | T×768 · 768×2304 | 42.3 | 113.3 | 3198 | 30.2 | 4.2 | 44.9 |
| Attention out | T×768 · 768×768 | 40.3 | 115.6 | 1599 | 29.9 | 3.6 | 44.6 |
| MLP up | T×768 · 768×3072 | 42.5 | 114.0 | 2425 | 30.6 | 4.3 | 45.0 |
| MLP down | T×3072 · 3072×768 | 43.2 | 112.6 | 2240 | 30.3 | 3.5 | 44.2 |

GFLOPS, median.

- **Prefill** reuses each weight 512 times (~135 flops/byte): compute-bound. Packing beats ikj by ~30%, and fixes MLP up, where N=3072 gives a 12 KB row stride that makes plain blocking slower than ikj.
- **Decode** uses each weight once (0.5 flops/byte). Accelerate streams weights at 88–90 GB/s on every layer and lands at exactly 0.5 × 90 = 45 GFLOPS: it's on the bandwidth roofline. Its lead over ikj collapses from ~70× (prefill) to ~1.5×.
- **The register-blocked simd kernel is 2.7× packed in prefill but 7–8× slower than ikj in decode**: with M=1 it fills 1 of 8 tile rows and packs all of B every call. Small M needs a GEMV path, not a GEMM kernel.
- **Blocking and packing hurt decode** (2.5–3.5× slower than ikj): with T=1 there's nothing to reuse a packed tile for, so the copy just adds memory traffic. The right kernel depends on the phase.

## Repo layout

```
include/
  alloc.hpp        aligned allocation, idx(row, col, ld) row-major indexing
  check.hpp        tolerance-scaled correctness comparison (scales with K)
  gemm.h           kernel contract (C += A·B, explicit strides) + declarations
src/
  kernels/         one file per stage, plus blas.cpp (vendor reference)
  bench.cpp        benchmark harness (CSV output)
tests/             correctness tests (CTest)
scripts/plot.py    plots from bench CSV
NOTES.md           per-stage mechanism, measurements, dead ends
CMakeLists.txt     build definition
CMakePresets.json  release / asan / ubsan configurations
```

## Build and run

Requires CMake ≥ 3.25 and Ninja. BLAS is optional; the `blas` reference kernel is built only if one is found.

```bash
cmake --preset release                 # configure into build/release/
cmake --build --preset release         # build kernels, bench, tests
ctest --preset release                 # run tests; nonzero exit = failure

# Sanitizer builds (separate build dirs, no clean needed)
cmake --preset asan  && cmake --build --preset asan  && ctest --preset asan
cmake --preset ubsan && cmake --build --preset ubsan && ctest --preset ubsan

./build/release/bench > results.csv    # CSV: implementation,size,run,seconds,gflops
./build/release/bench --impl ikj --reps 5 1000 2048   # one kernel, chosen sizes

# Experiment build: auto-vectorization off (separates loop-order gains from SIMD)
cmake --preset novec && cmake --build --preset novec
```

`ctest` does not rebuild. Always build before testing.

Options: `-DGEMM_NATIVE=OFF` (no `-march`/`-mcpu=native`), `-DGEMM_USE_BLAS=OFF`.

## Methodology

- **Correctness first.** Every kernel is checked against the naive reference with a K-scaled floating-point tolerance before it is timed. Planned: an all-kernels test on awkward sizes (1, 7, 63, 65, 129), non-square M≠N≠K, padded strides, and non-zero initial C.
- **Timing.** Fixed seed, setup outside the timer, one untimed warmup that also gates correctness, multiple timed reps with `steady_clock`, checksum of C printed to stderr to defeat dead-code elimination.
- **GFLOPS** = 2·M·N·K / seconds / 1e9.
- **Attribution.** Speedups are explained with hardware counters, not just wall time: `perf stat` on bare-metal x86-64. [Mac counter tool for Stages 0–2: TBD.]
- **Platforms.** Stages 0–2 developed on Apple M4 (ARM64). AVX2, OpenMP, and final numbers are measured on x86-64.

## CI

GitHub Actions (`.github/workflows/ci.yml`) on every push and PR, as a 2 × 3 matrix: Ubuntu (x86-64, OpenBLAS) and macOS (ARM64, Accelerate) × presets `release`, `asan`, `ubsan`. Each cell:

1. Builds and runs `ctest` (every kernel vs naive on awkward sizes, padded strides, NaN-poisoned padding)
2. Runs a bench smoke test on square and rectangular shapes; bench exits nonzero if any kernel fails its correctness gate. Timings from shared runners are discarded.

## Dead ends

Things that didn't work, and why. [Filled in as they happen.]