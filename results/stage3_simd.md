# Stage 3: register-blocked NEON micro-kernel — results and analysis

**Mechanism (one sentence):** keeping an 8 × 12 tile of C in 24 NEON
registers for the whole k loop turns each step into 20 floats loaded for 96
fused multiply-adds (vs 3 memory ops per FMA in Stages 1–2), which moves the
kernel from load/store-bound to FMA-bound: ~115 GFLOPS, 2.8–3× packed and
~47× naive at n = 1024.

## Setup

Same as earlier stages: Apple M4 Pro, single thread, Apple clang 17,
`-O3 -mcpu=native`, median of 5 reps (50 for decode).

Kernel: `matmul_simd` in `src/kernels/simd.cpp`.

| | |
|---|---|
| Register tile | MR × NR = 8 × 12: 24 accumulators + 3 B + 2 A = 29 of 32 NEON registers |
| Inner op | `vfmaq_laneq_f32(c, b, a, lane)`: B vector × one lane of A, fused |
| Packing | A into 8-row panels, B into 12-col panels, edges zero-filled |
| Blocks | MC = 128 (A block 128 KB, L2), KC = 256 (B panel 12 KB, L1), NC = 1536 (B block 1.5 MB, L2) |
| Edges | full micro-kernel into a scratch tile, add only the valid part |
| x86 | portable C++ micro-kernel with the same structure (AVX2 intrinsics: not yet written) |

Generated code: 48 lane-indexed `fmla.4s` in the object file = 24 per k step,
unrolled 2×. The register tile compiled as intended (no spills checked
beyond that count).

Correctness: `test_kernels` 128/128 on release, asan, ubsan, with new shapes
crossing every block edge (8×12×7, 9×13×300, 33×1537×257). Portable fallback
passes under ASan + UBSan. Negative controls (each caught):

| Injected bug | simd cases failed |
|---|---|
| `pack_A` uses `kc` instead of `lda` | 14 / 16 |
| edge tile drops its last column | 15 / 16 |
| panel offset uses `KC` instead of `kc` | 12 / 16 |

Removing the zero-fill in `pack_B` was **not** caught, correctly: padded
lanes only feed discarded columns of edge tiles. The zeros are hygiene
(keep NaN/denormals out of the FMAs), not correctness.

## Prediction going in

Memory ops per FMA drop from 3 to ~0.2, so square and prefill should jump
2–3× over packed. Decode should not improve (bandwidth-bound, and M = 1
fills 1 of 8 tile rows).

## 1. Square sizes

`stage3_simd_square.csv` vs Stage 2 CSVs

| n | ikj | packed | simd | simd / packed | simd / naive |
|---|---|---|---|---|---|
| 512 | 31.7 | 39.6 | 111.3 | 2.81× | — |
| 1000 | 31.8 | 41.2 | 98.6 | 2.39× | — |
| 1024 | 32.4 | 41.9 | **115.3** | **2.76×** | **47×** (naive 2.44, Stage 1) |
| 2000 | 30.9 | 38.0 | 114.2 | 3.00× | — |
| 2048 | 32.1 | 37.8 | 116.4 | 3.08× | — |

- Prediction confirmed: 2.4–3.1× over packed.
- No power-of-two cliff (A and B are both packed).
- The 1000 dip did **not** reproduce. `stage3_simd_edges.csv`, a re-run
  over 996–1032:

  | n | 996 | 1000 | 1004 | 1008 | 1020 | 1032 |
  |---|---|---|---|---|---|---|
  | GFLOPS | 105.3 | 112.7 | 108.6 | 112.0 | 111.2 | 105.9 |

  No pattern with n mod 8 or n mod 12 (edge tiles). Run-to-run noise at
  this speed is ~±10%; the first 98.6 was a low run.

## 2. How close to peak?

- Rough single P-core FP32 peak: 4 FMA pipes × 4 lanes × 2 flops ×
  ~4.5 GHz ≈ **144 GFLOPS**. This is an estimate (pipe count and sustained
  clock not verified).
- simd at ~115 → **~80% of that estimate**. Stages 1–2 were at 22–30%.
- To make this a measured number: a microbenchmark of independent `fmla`
  chains with no memory traffic gives the real per-core ceiling. Next step
  before quoting "% of peak" anywhere.

## 3. Transformer prefill (T = 512, warm)

`stage3_simd_prefill_warm.csv` vs `transformer_prefill_warm.csv`

| Layer | ikj | packed | simd | Accelerate |
|---|---|---|---|---|
| QKV | 31.6 | 42.3 | **113.3** | 3198 |
| attn out | 32.4 | 40.3 | **115.6** | 1599 |
| MLP up | 32.5 | 42.5 | **114.0** | 2425 |
| MLP down | 31.5 | 43.2 | **112.6** | 2240 |

- Same ~2.7× over packed on every layer; shape doesn't matter.
- Accelerate is still 14–28× ahead. It uses the AMX matrix units, separate
  hardware from NEON. On NEON alone, simd is near the ceiling (§2).

## 4. Transformer decode (T = 1, cold): simd is the worst kernel

`stage3_simd_decode_cold.csv` vs `transformer_decode_cold.csv`

| Layer | ikj | packed | simd | Accelerate |
|---|---|---|---|---|
| QKV | 30.2 | 8.9 | **4.2** | 44.9 |
| attn out | 29.9 | 9.0 | **3.6** | 44.6 |
| MLP up | 30.6 | 8.2 | **4.3** | 45.0 |
| MLP down | 30.3 | 8.6 | **3.5** | 44.2 |

- **7–8× slower than ikj.** Prediction ("no improvement") was too kind.
- Why (arithmetic, not measured separately):
  1. **7/8 of the FMAs are wasted.** M = 1 fills one of the tile's 8 rows;
     the other 7 multiply zero padding.
  2. **Every tile takes the edge path** (m = 1 < 8): scratch tile + scalar
     add-back, every time.
  3. **B is packed every call**, which reads and writes all of B before
     computing, the same 3× memory traffic that sank packed in decode.
- The fix is **dispatch, not tuning**: for small M, call a GEMV-style kernel
  (ikj already is one: ~30 GFLOPS, 60 GB/s, close to the ~45 GFLOPS
  bandwidth roofline). Production BLAS libraries do the same.

## 5. Where this stops helping

- Compute: ~80% of an estimated peak. The rest is plausibly packing
  overhead, edge tiles, loop control, and the C load/store per tile. Each
  needs a counter or experiment to attribute.
- Beyond one core: Stage 4 (OpenMP). Prefill should scale with cores.
  Decode is already near the single-thread DRAM wall for Accelerate; more
  threads should raise that wall, which Stage 4 can measure.
- x86: the portable fallback runs on CI but isn't tuned. An AVX2 version
  (8-wide, 16 YMM registers → a smaller tile, e.g. 6 × 16) is the x86 half
  of this stage.

## Summary of claims and evidence

| Claim | Status | Evidence |
|---|---|---|
| simd ~115 GFLOPS, 2.4–3.1× packed, 47× naive at 1024 | measured | §1 |
| Register tile compiled to lane-indexed `fmla.4s` | disassembly | Setup |
| ~80% of single-core peak | estimate (peak not measured) | §2 |
| ~2.7× over packed on all prefill layers | measured | §3 |
| simd is 7–8× slower than ikj in decode | measured | §4 |
| Decode slowdown = wasted tile rows + edge path + B packing | arithmetic, not measured separately | §4 |
| No measurable edge-tile effect on square sizes ±10% | measured | §1 |

## Reproduce

```bash
cmake --preset release && cmake --build --preset release
B=./build/release/bench
$B --impl simd --reps 5 512 1000 1024 2000 2048 > results/stage3_simd_square.csv
$B --impl simd --reps 5 996 1000 1004 1008 1020 1032 > results/stage3_simd_edges.csv
shapes() { T=$1; echo "--shape ${T}x2304x768 --shape ${T}x768x768 --shape ${T}x3072x768 --shape ${T}x768x3072"; }
$B --impl simd --reps 5         $(shapes 512) > results/stage3_simd_prefill_warm.csv
$B --impl simd --reps 50 --cold $(shapes 1)   > results/stage3_simd_decode_cold.csv
objdump -d build/release/CMakeFiles/gemm.dir/src/kernels/simd.cpp.o | grep -cE "fmla\.4s.*\[[0-3]\]"
```
