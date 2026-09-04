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

---

## Next
- 1.3 — Stage 0 naive `ijk`. Watch for the compiler deleting the
  computation if nothing reads C.
- 1.4 — benchmark harness (warmup discarded, multiple reps, CSV out)
- 1.5 — baseline + `-fopt-info-vec` auto-vectorization check (x86 only)