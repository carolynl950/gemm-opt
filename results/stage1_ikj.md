# Stage 1: `ikj` loop reorder — results and analysis

**Mechanism (one sentence):** swapping the `j` and `k` loops turns naive's
column walk down B into contiguous row walks over B and C, which uses every
float of each cache line *and* removes the serial add chain, so the compiler
can vectorize the inner loop with fused multiply-adds.

## Setup

| | |
|---|---|
| CPU | Apple M4 Pro, single thread (P-core: 128 KB L1d, 16 MB shared L2, 128 B lines, 16 KB pages) |
| Compiler | Apple clang 17.0.0, `-O3 -mcpu=native` (preset `release`) |
| No-SIMD build | same + `-fno-vectorize -fno-slp-vectorize` (preset `novec`) |
| Matrices | square, row-major, `ld = n`, fp32, values in [-1, 1), fixed seeds |
| Timing | 1 warmup (also correctness-checked vs naive), 5 timed reps, median reported |
| BLAS | Accelerate (uses AMX; not comparable to scalar/NEON code) |

All numbers are GFLOPS = 2n³ / seconds / 1e9. Raw CSVs are in this folder.

## 1. Headline

`stage1_release_default.csv`

| n | naive | ikj | Accelerate | ikj / naive |
|---|---|---|---|---|
| 128 | 2.25 | 25.4 | 786 | 11.3× |
| 256 | 3.05 | 31.3 | 1312 | 10.2× |
| 512 | 2.89 | 33.3 | 2378 | 11.5× |
| 1024 | 2.44 | 32.7 | 2164 | 13.4× |
| 2048 | 1.21 | 32.4 | 2175 | 26.8× |

- ikj: flat ~32 GFLOPS from 256 to 2048, reps within 1–3% at 512+.
- naive: halves at 2048.
- Accelerate: 30–70× above ikj, reps vary up to 2× (likely AMX and/or
  multithreading). Reported as a ceiling only.

## 2. Hypotheses going in

| # | Hypothesis | Test |
|---|---|---|
| H1 | ikj speedup = better cache-line use **plus** SIMD; the split is unknown | Rebuild with auto-vectorization off (`novec`) |
| H2 | Naive is not vectorized (float reduction can't be reordered) | Compiler remarks + disassembly |
| H3 | Naive's drop at 2048 is cache-**set conflicts** from a power-of-two stride, not capacity | Run n = 2000, 2048, 2050 |
| H4 | ikj at ~32 GFLOPS is load/store-bound, not memory-bound | Not tested yet (needs counters) |

## 3. What the compiler actually generated (H2: partly wrong)

From `-Rpass=loop-vectorize` and `objdump -d` on the kernel objects:

| kernel / build | inner-loop instructions |
|---|---|
| ikj `release` | `fmla.4s`: vector fused multiply-add, 4 floats, interleaved ×4 |
| ikj `novec` | scalar `fmadd` only |
| naive `release` | vector `fmul.4s` **followed by a strictly in-order chain of scalar `fadd`** |
| naive `novec` | scalar `fmadd` |

- **H2 was wrong as stated.** LLVM *does* vectorize naive's k loop, but as
  an *ordered* reduction: multiplies run 4-wide, adds still go one at a time
  in the original order (reordering would change the float result).
- So naive is bound by **one serial add chain**: each `fadd` waits for the
  previous one. SIMD can't help that, which section 4 confirms.
- Side effect: naive `release` rounds twice per term (`fmul` then `fadd`),
  naive `novec` and ikj round once (fused). The kernels are therefore not
  guaranteed bit-identical even with the same summation order. Within
  tolerance; noted for later.

## 4. Splitting the speedup: loop order vs SIMD (H1)

`stage1_release_default.csv` vs `stage1_novec_default.csv`

| n | naive novec | ikj novec | ikj release | **order alone** (ikj novec / naive novec) | **SIMD** (ikj release / ikj novec) |
|---|---|---|---|---|---|
| 128 | 2.65 | 4.92 | 25.4 | 1.9× | 5.2× |
| 256 | 3.23 | 7.47 | 31.3 | 2.3× | 4.2× |
| 512 | 2.98 | 7.73 | 33.3 | 2.6× | 4.3× |
| 1024 | 2.38 | 7.70 | 32.7 | 3.2× | 4.2× |
| 2048 | 1.19 | 7.73 | 32.4 | 6.5× | 4.2× |

- **Naive gets nothing from vectorization** (release ≈ novec at every size).
  Consistent with section 3: the serial add chain is the bottleneck.
- **SIMD is a steady ~4.2×** for ikj at n ≥ 256: matches the 4-wide vectors.
  Slightly above 4, likely from the 4× interleave giving independent chains
  and fewer loop-control instructions per element (hypothesis).
- **Loop order alone is 2–6.5×**, and grows with n. Small matrices fit in
  cache either way, so the column walk costs little; at 2048 the column
  walk falls apart (section 5) and ikj doesn't.
- So "ikj is 11× faster at 512" = **2.6× from memory access pattern ×
  4.3× from SIMD**. The SIMD half only exists *because* the reorder removed
  the reduction.

## 5. Naive's cliff at 2048 is the power-of-two stride (H3: supported)

`stage1_stride_naive.csv`, `stage1_stride_ikj.csv`

| n | row stride | naive | ikj |
|---|---|---|---|
| 1000 | 4000 B | 2.64 | 31.5 |
| 1024 | 4096 B | **2.06** | 32.3 |
| 1050 | 4200 B | 2.39 | 26.1 |
| 2000 | 8000 B | 2.33 | 31.6 |
| 2048 | 8192 B | **1.31** | 31.8 |
| 2050 | 8200 B | 2.27 | 25.8 |

- **2050 is bigger than 2048 and 1.7× faster.** That rules out capacity
  (more data, not less) and TLB reach (same pages-per-column). What's left
  is the exact power of two.
- **Mechanism:** a cache picks a set from middle address bits. Walking a
  column with an 8192 B stride keeps those bits nearly constant, so all of
  B's column lands in a handful of sets and evicts itself, even though the
  cache is mostly empty. 1024 (4096 B stride) shows a smaller dip, as
  expected from a less extreme stride.
- Not yet confirmed with counters (would need L1 miss counts per size).
  M4 Pro L1 associativity is not published; the argument doesn't depend on
  the exact number.
- ikj is unaffected by 2048 because it never walks a column.

## 6. Surprise: ikj is ~20% slower when rows aren't 16-byte aligned

The table above showed ikj dropping at 1050 and 2050. Follow-up,
`stage1_align_ikj.csv`:

| n | row bytes | mod 16 | mod 64 | ikj |
|---|---|---|---|---|
| 1048 | 4192 | 0 | 32 | 31.7 |
| 1049 | 4196 | 4 | 36 | 24.0 |
| 1050 | 4200 | 8 | 40 | 26.0 |
| 1051 | 4204 | 12 | 44 | 23.8 |
| 1052 | 4208 | 0 | 48 | 31.7 |
| 1056 | 4224 | 0 | 0 | 32.0 |

- **Clean split on 16-byte alignment:** every n with row bytes ≡ 0 mod 16
  runs ~32; every other n runs 24–26. 64 B and 128 B alignment make no
  difference (1048, 1052, 1056 all ~32).
- 16 B = one NEON `q` register. The matrix base is 32 B aligned, so row 0 is
  always aligned; row i starts at `i · n · 4` bytes, which is 16 B aligned
  only when n is a multiple of 4.
- **Hypothesis for the cost:** unaligned 16 B loads/stores sometimes
  straddle a cache line and cost an extra access. Not confirmed; the exact
  penalty mechanism needs counters or a microbenchmark.
- **Practical consequence:** padding `ld` up to a multiple of 4 floats
  would recover ~25% at these sizes. Worth remembering for Stage 2 (packed
  buffers can be aligned by construction).
- **Measurement consequence:** the default sweep uses only powers of two,
  which are both the *worst* case for naive (section 5) and the *best* case
  for ikj (aligned). Headline ratios at 1024/2048 flatter ikj. Future sweeps
  should include non-power-of-two sizes.

## 7. Where ikj stops helping (open)

- ikj is flat at ~32 GFLOPS regardless of n, so it is not running out of
  cache yet in a way that costs throughput.
- Rough single-P-core FP32 peak: 4 FMA pipes × 4 lanes × 2 flops × ~4.5 GHz
  ≈ 140 GFLOPS. This is an estimate, not a datasheet figure. If roughly
  right, ikj is at ~23% of peak.
- **H4 (untested):** each inner iteration does 2 loads (B, C) and 1 store
  (C) per FMA, so ikj is likely limited by load/store bandwidth, not by
  FMA units or DRAM. Stage 2's register blocking (keep a tile of C in
  registers across k) attacks exactly that.

## Summary of claims and evidence

| Claim | Status | Evidence |
|---|---|---|
| ikj is 10–27× faster than naive | measured | §1 |
| ~4.2× of that is SIMD | measured | §4, novec build |
| Naive can't use SIMD because of the ordered add chain | measured + disassembly | §3, §4 |
| Naive's 2048 cliff is power-of-two set conflicts | supported by experiment, no counters | §5 |
| ikj loses ~25% on rows not 16 B aligned | measured | §6 |
| Why unaligned costs that much | hypothesis | §6 |
| ikj is load/store-bound at ~23% of peak | hypothesis | §7 |

## Reproduce

```bash
cmake --preset release && cmake --build --preset release
cmake --preset novec   && cmake --build --preset novec
./build/release/bench > results/stage1_release_default.csv
./build/novec/bench   > results/stage1_novec_default.csv
./build/release/bench --impl naive_ijk 1000 1024 1050 2000 2048 2050 > results/stage1_stride_naive.csv
./build/release/bench --impl ikj       1000 1024 1050 2000 2048 2050 > results/stage1_stride_ikj.csv
./build/release/bench --impl ikj       1048 1049 1050 1051 1052 1056 > results/stage1_align_ikj.csv

# vectorization evidence
c++ -std=c++17 -O3 -mcpu=native -Iinclude -c src/kernels/reorder_ijk.cpp -o /dev/null -Rpass=loop-vectorize
objdump -d build/release/CMakeFiles/gemm.dir/src/kernels/reorder_ijk.cpp.o | grep fmla
```
