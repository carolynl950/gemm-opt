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
| 2 | `blocked` | Cache blocking (tiling) with runtime block size and remainder handling | Planned |
| 3 | `simd` | AVX2 (x86-64) vectorized micro-kernel | Planned |
| 4 | `omp` | OpenMP multithreading over blocks | Planned |
| — | `blas` | Vendor BLAS reference ceiling (Accelerate on macOS, OpenBLAS on Linux) | Wired up |

Each stage records its mechanism (one sentence: *why* it's faster) and its measured effect in `NOTES.md`.

## Results

> Numbers are filled in only after measurement. Bracketed values are placeholders.

| Kernel | GFLOPS (N=1024) | Speedup vs naive | Key counter change |
|---|---|---|---|
| naive | 2.44 | 1.0× | — |
| ikj | 32.7 | 13.4× | [X] (split measured: ~3.2× access pattern × ~4.2× SIMD) |
| blocked | [X] | [X]× | [X] |
| simd | [X] | [X]× | [X] |
| omp | [X] | [X]× | [X] |
| BLAS (Accelerate, AMX) | 2164 | 887× | — |

Hardware (current numbers): Apple M4 Pro, single thread, 128 KB L1d / 16 MB L2 / 128 B lines; Apple clang 17, `-O3 -mcpu=native`; BLAS = Accelerate. Median of 5 reps. Final x86 numbers: [X]

On Apple silicon the BLAS reference is Accelerate, which uses the AMX matrix units. It is reported separately and is not a target for the hand-written kernels.

### Transformer workloads

Benchmarked on GPT-2-small layer shapes (d=768, MLP=3072), for prefill (T=512) and decode (T=1):

| Layer | Shape (M×K · K×N) | Prefill GFLOPS | Decode GFLOPS |
|---|---|---|---|
| QKV projection | T×768 · 768×2304 | [X] | [X] |
| MLP up | T×768 · 768×3072 | [X] | [X] |
| MLP down | T×3072 · 3072×768 | [X] | [X] |

Prefill is compute-bound, so blocking and SIMD pay off. Decode (T=1) is effectively matrix-vector and memory-bandwidth-bound, so cache blocking barely helps. [Fill in the measured explanation.]

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

Planned (not yet set up). GitHub Actions on every push:

1. Build and correctness tests on Ubuntu (x86-64) and macOS (ARM64)
2. The same tests under AddressSanitizer + UBSan, to catch out-of-bounds bugs in tile edge handling
3. A benchmark smoke run (checks that it runs; shared runners are too noisy for timing)

## Dead ends

Things that didn't work, and why. [Filled in as they happen.]