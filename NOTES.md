# NOTES.md — GEMM optimization project

## Machine (development)
- MacBook Pro, **Apple M4 Pro** (ARM64), Apple clang 17.0.0
- P-core: 128 KB L1d, 16 MB shared L2, **128 B cache lines**, 16 KB pages (`sysctl hw.perflevel0.*`, `hw.cachelinesize`, `hw.pagesize`)

## Reference numbers
- `sizeof(float)` = 4 → **32 floats per 128 B line on M4 Pro** (16 per 64 B line on x86), **4 per NEON register**, **8 per AVX2 register**
- `FLT_EPSILON` ≈ 1.19e-7 (2^-23, from 23 mantissa bits)
- Matrix memory footprint (single matrix; ×3 for A, B, C):

| N | row | matrix |
|---|---|---|
| 128 | 512 B | 64 KB |
| 512 | 2 KB | 1 MB |
| 1024 | 4 KB | 4 MB |
| 2048 | 8 KB | 16 MB |

---

## Step 1.1 — allocation and indexing (done)
- 32-byte aligned via `posix_memalign`; freed with `free`, not `delete`.
- Row-major throughout, `idx(row, col, ld) = row * ld + col`.
- `ld` (leading dimension) passed explicitly 
---

## Step 1.2 — correctness checker (done)

Float addition is not associative. Every stage of this project reorders
summation, so no two stages produce bit-identical output and `==` would
reject correct code.

**Tolerance:** tol = K * FLT_EPSILON * fabs(ref) + 1e-6f, where
- `FLT_EPSILON` — relative size of one rounding step
- `* fabs(ref)` — rounding error is relative; scale to magnitude
- `K *` — K accumulations per element, so K roundings can stack
- `+ 1e-6f` — floor, or elements equal to 0 get tol = 0 and fail on anything

### Experiment: order-dependent summation
Array of 10000 floats, sorted: 5000 × 1e6 followed by 5000 × small.
Summed forward and backward. Float spacing near 5e9 is **512**.

| small value | 5000 × small | vs 512 spacing | fwd vs bwd |
|---|---|---|---|
| 0.01 | 50 | ~10× below | identical |
| 1.0 | 5000 | ~10× above | **differ by 512** |
| 10.0 | 50000 | ~100× above | differ by 49664 |

**Mechanism.** Forward adds each small value to an already-huge total; each
one is far below half a spacing and rounds away individually. Backward adds
the small values *to each other* first, where they are comparable in size
and nothing is lost, then meets the large total as a single aggregate.
Whether that aggregate survives depends on `count × small` vs the spacing.

At small = 1.0 the two sums land on **adjacent representable floats** —
the 512 gap is exactly one ulp at 5e9. There is no possible answer between
them, and neither is more correct than the other.

### Experiment: the K term is load-bearing
Same inputs, same 1.2e6 drift. Only the claimed accumulation count changed:

| K passed | tolerance | result |
|---|---|---|
| 10000 (true) | ~6.0e6 | pass |
| 1 (false) | 602 | **fail** |

A tolerance without K either rejects valid results at large K or accepts
garbage at small K.

### Checker verified three ways
1. identical matrices → pass
2. one element corrupted by +100 → fail (tol there is the 1e-6 floor,
   since ref is 0)
3. same sum, reversed order → pass

### Bug: NaN slipped through the checker
`fabs(ref - test) > tol` is **false** when `test` is NaN, because every
comparison with NaN is false. So a NaN result passed. Changed to
`!(fabs(ref - test) <= tol)`, which fails on NaN.

---

## Step 1.6 — all-kernels correctness test (`tests/test_kernels.cpp`)

Every kernel in `IMPLS` vs naive, on shapes built to break it:

| Case | Catches |
|---|---|
| awkward sizes 1, 7, 63, 65, 129 | remainder-loop bugs (Stage 2+ blocking, SIMD) |
| M≠N≠K | swapped loop bounds, invisible on square |
| M=1 / N=1 / K=1 | degenerate edges; M=1 is decode |
| padded strides (`ld = width + 3`) | kernel uses `N` where it should use `ldb` |
| C starts random, not zero | kernel does `C = A·B` instead of `C += A·B` |

**NaN poisoning.** Every buffer is filled with NaN before the logical part
gets random values. After the kernel runs, padding must still be NaN:
- kernel **writes** padding → `padding_intact` fails
- kernel **reads** padding → NaN propagates into results → `matrices_match`
  fails (only after the NaN fix above)

Kernel list lives in `include/kernels.hpp`, shared with `bench.cpp`, so a
kernel can't be benchmarked without also being tested.

### Finding: `|ref|`-relative tolerance fails under cancellation
First run: 19/20 pass. Failure is BLAS (Accelerate) vs naive at 129×129×129:

```
mismatch at (19, 47): ref=0.00332272 test=0.0033215 diff=1.21724e-06 tol=1.0511e-06
```

Neither kernel is wrong. The tolerance is wrong for this element.

- Tolerance scales with `|ref|`, the size of the **answer**.
- Rounding error scales with the size of the **terms being added**. Inputs
  are in [-1, 1), so the 129 products and the running sums are O(1).
- Here they cancel to 0.0033. The summed magnitudes at that element are
  34.5 (measured, see below): the answer is ~10,000× smaller than what was
  added, so the tolerance was ~10,000× too tight.
- Naive vs naive never hits this (same order, bit-identical). BLAS sums in a
  different order (blocking, FMA), so the rounding differs.

**Principled bound** (standard dot-product error analysis):

```
|error| ≤ K · ε · ( |C₀| + Σₖ |a_ik| · |b_kj| )
```

Scale by the magnitudes that were summed, not by the result. Computable
with one extra naive call on `|A|`, `|B|`, `|C₀|`.

Not fixed by raising the constant: that would also hide real bugs on large
values.

**Fix.** `matrices_match_bound` in `check.hpp` takes a per-element `scale`
matrix. `test_kernels` computes it as `naive(|A|, |B|, |C₀|)`: naive does
`C += A·B`, so seeding with `|C₀|` gives the formula exactly. All terms ≥ 0,
so `scale` itself can't cancel.

**Result.** 20/20 on release, asan, ubsan.

| at element (19, 47), 129³, BLAS | value |
|---|---|
| ref | 0.00332272 |
| scale (Σ magnitudes) | 34.4753 |
| old tol (`K·ε·|ref|`) | 1.05e-6 |
| new tol (`K·ε·scale`) | ~5.3e-4 |
| actual diff | 1.22e-6 |

**Still strict?** Injected `+1e-3` into one C element after the kernel:
0/20 pass. Caught on every shape.

**Caveat.** `K·ε` is the *worst-case* bound (every rounding error same
sign). Observed diff is ~400× under it. Real errors behave more like
random walk, ~√K·ε. So the test can't catch errors smaller than about
K·ε·scale; that's the price of never rejecting a correct kernel.

---

## Stage 1 — `ikj` loop reorder (done)

**Why it's faster:** swapping `j` and `k` turns naive's column walk down B
into contiguous row walks over B and C, which uses every float of each cache
line *and* removes the serial add chain, so the inner loop vectorizes to
`fmla.4s`.

- 10–27× over naive on M4 Pro; measured split ≈ 2.6× access pattern × 4.3×
  SIMD at n=512.
- Naive's 2048 cliff is the power-of-two stride (2050 is 1.7× faster).
- ikj loses ~25% when rows aren't 16 B aligned (n not a multiple of 4).

Full write-up with tables, hypotheses and evidence: `results/stage1_ikj.md`.

---

## Stage 2a — cache blocking (done)

**Why it's faster (when it is):** tiling lets one KC × NC tile of B be reused
by MC rows of A from L1: +34% over ikj at n = 1000, 2000. At powers of two
the tile's rows collide in the same cache sets and blocking loses to ikj
(0.83× at 2048) unless the tile is contiguous.

- KC dominates; MC barely matters. Same 64 KB tile: contiguous 43 vs
  strided 30 GFLOPS.
- No single block size is best across n → pack B (Stage 2b).

Full write-up: `results/stage2_blocked.md`.

---

## Stage 2b — packing B (done)

**Why it's faster:** copying each B tile into a contiguous aligned buffer
makes its cache footprint independent of `ldb`, removing the power-of-two
cliff (1024: 29.6 → 41.9; 2048: 26.8 → 37.8) and beating ikj at every n.

- Performance now depends on tile **size**, not shape: ≤ 128 KB (L1) runs
  36–43, ≥ 256 KB falls back to ikj's ~32.
- Still ~30% of estimated peak: the inner loop needs 2 loads + 1 store per
  FMA. Next lever is register blocking (micro-kernel).

Full write-up: `results/stage2b_packed.md`.

---

## Transformer shapes — prefill vs decode (done)

**Takeaway:** prefill (~135 flop/byte) is compute-bound and rewards
blocking/packing; decode (0.5 flop/byte) is weight-bandwidth-bound. Cold
decode caps Accelerate at 88–90 GB/s ≈ 45 GFLOPS on every layer, so its
lead over ikj shrinks from ~70× to ~1.5×, and packing makes decode 3× worse.

- MLP up (N = 3072, 12 KB stride) reproduces the Stage 2a blocked cliff on
  a real model shape.
- bench now has `--shape MxNxK`, `--cold`, and uses the bound check.

Full write-up: `results/transformer_shapes.md`.

---

## Stage 3 — NEON register-blocked micro-kernel (done on M4; AVX2 pending)

**Why it's faster:** an 8 × 12 tile of C stays in 24 NEON registers for the
whole k loop, so each step loads 20 floats for 96 FMAs instead of 3 memory
ops per FMA: ~115 GFLOPS, 2.8× packed, 47× naive at 1024, **~85% of the
measured ~135 GFLOPS** single-core peak (`results/peak_fma.md`).

- Prefill: ~113–116 on every GPT-2 layer.
- Decode: 3.5–4.3, **worst of all kernels**: M = 1 wastes 7/8 of the tile
  and packs B every call. Needs small-M dispatch to a GEMV path.
- Negative control lesson: removing pack zero-fill isn't a correctness bug
  (padded lanes only reach discarded columns).

Full write-up: `results/stage3_simd.md`.

---

## Peak FMA microbenchmark (done)

One P-core: **~135 GFLOPS** FP32, linear from 1 to 16 independent FMA
chains (8.8 GFLOPS each), flat after. 16 in flight = latency × pipes.
First version was wrong (`vfmaq_f32(y, acc, x)` forced a register copy
per FMA → spills at 20 chains); always check the assembly.

Full write-up: `results/peak_fma.md`.

---

## Small-M dispatch (done)

Measured simd vs ikj over M = 1..64 (QKV, cold): crossover exactly at
M = 8 = MR. `matmul_simd` now calls ikj for M < 8: decode 3.5–4.3 →
27–30 GFLOPS. Weak spot left at M = 9–11 (one full tile + a mostly-empty
edge tile). Full write-up: `results/stage3_dispatch.md`.

---

## Stage 4 — OpenMP (done on M4)

**Why it's faster:** splitting the `jr` loop (12-col B panels) across
threads, with shared packed A/B and `omp for` barriers: ~115 → ~440 GFLOPS
(3.8× at 6 threads). Flat to 8, worse with E-cores (10–12).

- 8 concurrent `peak` processes = ~960 GFLOPS, so the plateau is the
  parallel design (~46% of 8-core peak), not clocks.
- Decode: column-split was 3× slower than serial ikj at 1 thread (strided B
  walk defeats the prefetcher); split-K fixed it. Threads give ≤ 1.25× in
  decode and hurt small layers.

Full write-up: `results/stage4_omp.md`.

---

## Stage 5 — CUDA on T4: naive → coalesced → shared memory → register tiling (done)

**Why 5b is faster:** mapping a warp's 32 threads to consecutive columns
makes each warp load of B 4 contiguous transactions instead of a scatter:
61 → 502–652 GFLOPS (8–12×), same arithmetic.

- T4 is **power-limited**: cuBLAS sustains ~3.3–3.6 TFLOPS under the 70 W
  cap (clock log shows reason 0x4); an earlier 7,000 was a pre-cap burst.
- GPU decode is bandwidth-bound too: cuBLAS 138–212 GB/s of ~320 spec.
- 5c shared-memory tiling: 1.3–1.5× coalesced (~880–985 GFLOPS, 17–25% of
  cuBLAS), not 32×: bottleneck moves to 2 shared loads per FMA. Slower than
  coalesced in decode (nothing to reuse at M = 1).
- 5d register tiling (8×8 per thread, 128×128 block tiles): **3,545 GFLOPS
  = 94% of cuBLAS at n = 2048, 57× naive**, same run. Falls to 49–78% on
  shapes with fewer blocks than the 40 SMs can hold; worst kernel in decode
  (needs small-M dispatch, like the CPU).

Full write-up: `results/stage5_cuda.md`.

---

## Next
- Stage 3 x86: AVX2 micro-kernel
- 2-D (ic × jr) thread grid to get past the ~440 GFLOPS plateau
- STREAM-style multi-thread bandwidth benchmark (is ~90 GB/s per-core or chip?)
- Counter confirmation of the set-conflict and prefetcher hypotheses