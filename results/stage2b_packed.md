# Stage 2b: blocking + packing B — results and analysis

**Mechanism (one sentence):** copying each KC × NC tile of B into a
contiguous, 16 B-aligned buffer makes the tile's cache footprint independent
of `ldb`, which removes the power-of-two set-conflict cliff from Stage 2a
(1024: 29.6 → 41.9 GFLOPS; 2048: 26.8 → 37.8) and makes packed faster than
ikj at every size tested.

## Setup

Same as Stages 1 and 2a: Apple M4 Pro, single thread, Apple clang 17,
`-O3 -mcpu=native`, square row-major `ld = n`, median of 5 reps (3 in
sweeps). Defaults MC = 64, KC = 64, NC = 256.

Kernel: `matmul_packed` in `src/kernels/blocked.cpp`. Loop order
`k0 → j0 → pack B tile → i0 → i → k → j`; packed row stride = tile width
rounded up to 4 floats. Inner loop identical to ikj, reading the packed copy.

Correctness: `test_kernels` 91/91 on release, asan, ubsan, including a pass
with 3×5×7 tiles. Negative control: packing one column too few fails 25
cases.

## Prediction going in

From `stage2_blocked.md` §5: ~42 GFLOPS at every n, no dip at 1024 or 2048.

## 1. Packed vs blocked vs ikj

`stage2_ikj.csv`, `stage2_blocked_default.csv`, `stage2b_packed_default.csv`

| n | ikj | blocked | packed | packed / ikj | packed / blocked |
|---|---|---|---|---|---|
| 512 | 31.7 | 35.5 | 39.6 | 1.25× | 1.12× |
| 1000 | 31.8 | 42.6 | 41.2 | 1.30× | 0.97× |
| 1024 | 32.4 | 29.6 | **41.9** | 1.29× | **1.42×** |
| 2000 | 30.9 | 42.1 | 38.0 | 1.23× | 0.90× |
| 2048 | 32.1 | 26.8 | **37.8** | 1.18× | **1.41×** |

- **The power-of-two cliff is gone.** 1024 and 2048 went from the worst
  blocked sizes to normal. Prediction confirmed at 1024 (41.9 vs ~42);
  2048 is ~10% under it.
- **Packed beats ikj at every size** (1.18–1.30×). Blocked didn't.
- **Packed is slightly slower than blocked at 1000 and 2000** (−3%, −10%),
  where blocked had no conflicts to fix. So packing isn't free; see §3.

## 2. With packing, tile **size** matters, not tile **shape**

`stage2b_sweep_1024.csv`, `stage2b_sweep_2048.csv` (MC = 64 fixed)

| KC | NC | tile size | n = 1024 | n = 2048 |
|---|---|---|---|---|
| 16 | 256 | 16 KB | 40.8 | 31.2 |
| 16 | 512 | 32 KB | 40.9 | 36.7 |
| 16 | 1024 | 64 KB | 40.6 | **40.6** |
| 64 | 256 | 64 KB | **42.7** | 38.0 |
| 128 | 256 | 128 KB | 41.0 | 37.8 |
| 64 | 512 | 128 KB | 39.3 | 35.9 |
| 64 | 1024 | 256 KB | 32.6 | 31.5 |
| 128 | 512 | 256 KB | 31.8 | 31.0 |
| 128 | 1024 | 512 KB | 32.3 | 31.8 |
| 256 | 256 | 256 KB | 31.2 | 30.5 |
| 256 | 512 | 512 KB | 31.9 | 31.6 |
| 256 | 1024 | 1 MB | 32.3 | 32.0 |

- **Clean capacity cliff at L1 size.** Every tile ≤ 128 KB (the P-core L1d)
  runs 36–43; every tile ≥ 256 KB falls to ~31–32, which is ikj's speed:
  once the tile doesn't fit L1, blocking has nothing left to give.
- Contrast with 2a, where two 64 KB tiles ran 43 vs 30 depending on shape.
  After packing, the two 64 KB tiles (16×1024, 64×256) are within 5% of each
  other at 1024. **Packing turned a layout problem into a pure capacity
  problem**, which is the point.
- The default 64/256 is best at 1024 and within 7% of best at 2048.
- One outlier: 16 × 256 at 2048 (31.2). Small KC means each C element is
  loaded and stored once per K/KC = 128 passes, and at n = 2048 the C slice
  for one tile (2048 × 256 floats = 2 MB) lives in L2, not L1. Hypothesis:
  C traffic dominates there. Not tested.

## 3. Cost of packing, and loop order (hypotheses)

Why packed is slower than blocked at 1000 and 2000:

- **Copy cost:** each B element is copied once per call (K × N floats).
  That's small next to 2·M·N·K flops, but not zero.
- **Loop order changed:** blocked runs `i0 → k0 → j0`, so a 64-row block of
  C stays nearby while all of K streams past. Packed runs `k0 → j0 → all i`
  so each tile is packed once, which means every tile pass walks **all M
  rows** of a C slice (M × NC floats: 1 MB at 1000, 2 MB at 2000) and C
  comes from L2 instead of L1.
- Untested which of the two dominates. A test: pack once per (i0, k0, j0)
  in blocked's loop order and compare. Production BLAS also packs **A**,
  and adds a register-level micro-kernel, which changes this balance.

## 4. Where this stops helping

- Best packed result ~42 GFLOPS ≈ 30% of the rough ~140 GFLOPS single
  P-core estimate (estimate, not datasheet).
- The inner loop is still ikj's: per FMA, load B, load C, store C. Packing
  fixed **where B comes from** (L1, at any n). It did nothing about
  **how many memory operations each FMA needs**.
- Next lever: **register blocking**. Keep an mr × nr tile of C in NEON
  registers across the whole k loop, so each loaded B vector feeds mr FMAs
  and C is loaded/stored once per tile instead of once per k. That's the
  micro-kernel step (Stage 3 territory: explicit SIMD).

## Summary of claims and evidence

| Claim | Status | Evidence |
|---|---|---|
| Packing removes the power-of-two cliff (1024: 1.42×, 2048: 1.41× over blocked) | measured | §1 |
| Packed beats ikj at all sizes tested (1.18–1.30×) | measured | §1 |
| After packing, performance depends on tile size, not shape | measured | §2 |
| Cliff when tile exceeds L1 (128 KB) → back to ikj speed | measured | §2 |
| Packed slower than blocked at 1000/2000 due to copy cost and/or C locality | hypothesis | §3 |
| Remaining limit is memory ops per FMA, fixed by register blocking | hypothesis | §4 |

## Reproduce

```bash
cmake --preset release && cmake --build --preset release
B=./build/release/bench
$B --impl packed --reps 5 512 1000 1024 2000 2048 > results/stage2b_packed_default.csv
for n in 1024 2048; do
  echo "kc,nc,implementation,size,run,seconds,gflops" > results/stage2b_sweep_$n.csv
  for kc in 16 64 128 256; do for nc in 256 512 1024; do
    $B --impl packed --reps 3 --kc $kc --nc $nc $n 2>/dev/null \
      | tail -n +2 | sed "s/^/$kc,$nc,/" >> results/stage2b_sweep_$n.csv
  done; done
done
```
