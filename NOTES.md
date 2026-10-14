# NOTES.md — GEMM optimization project

## Machine (development)
- MacBook Pro, Apple M4 (ARM64), Apple Clang 17

## Reference numbers
- `sizeof(float)` = 4 → **16 floats per 64 B cache line**, **8 per AVX2 register**
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

## Next
- Stage 1 `ikj`
- Bench CLI (`--impl --m --n --k --reps`)