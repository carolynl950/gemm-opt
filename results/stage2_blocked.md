# Stage 2a: cache blocking — results and analysis

**Mechanism (one sentence):** tiling ikj into MC × KC × NC blocks lets one
KC × NC tile of B be reused by MC rows of A while it sits in L1, which gives
+34% over ikj at non-power-of-two sizes; at power-of-two sizes the tile's
rows are spaced exactly one power-of-two stride apart, collide in the same
cache sets, and blocking *loses* to ikj unless the tile is contiguous.

## Setup

Same as Stage 1 (`stage1_ikj.md`): Apple M4 Pro, single thread, Apple clang
17, `-O3 -mcpu=native`, square row-major `ld = n`, median of 5 reps (3 in the
sweep). Defaults: MC = 64, KC = 64, NC = 256 (B tile = 64 KB = half of L1d).

Correctness: `test_kernels` 65/65 on release, asan, ubsan, including block
edge shapes (64×256×64, 65×257×65, 128×512×128) and a second pass with
3×5×7 tiles. Negative control: removing the `min()` on `j_end` is caught by
ASan immediately.

## Prediction going in

From Stage 1, ikj was flat at ~32 GFLOPS to n = 2048, and B at 2048 (16 MB)
nearly fits the 16 MB L2. Predicted: **small gain at best** on this machine.

## 1. Blocked (defaults) vs ikj

`stage2_ikj.csv`, `stage2_blocked_default.csv`

| n | ikj | blocked 64/64/256 | blocked / ikj |
|---|---|---|---|
| 512 | 31.7 | 35.5 | 1.12× |
| 1000 | 31.8 | **42.6** | **1.34×** |
| 1024 | 32.4 | 29.6 | **0.91×** |
| 2000 | 30.9 | **42.1** | **1.36×** |
| 2048 | 32.1 | 26.8 | **0.83×** |

- Prediction was half right: the gain is bigger than expected at
  non-power-of-two sizes (+34–36%), and blocking is **slower** than ikj at
  1024 and 2048.
- Rep spread ≤ 5% everywhere, so these gaps are real, not noise.

## 2. Block-size sweep at n = 1024

`stage2_sweep_1024.csv` — 27 configs, MC, KC ∈ {16, 64, 256}, NC ∈ {64, 256, 1024}.

| KC | NC | best GFLOPS (over MC) | B tile shape |
|---|---|---|---|
| 16 | 1024 | **43.1** | 16 full rows = 64 KB, **contiguous** |
| 16 | 256 | 41.8 | 16 chunks of 1 KB, 4 KB apart |
| 64 | 1024 | 32.4 | 64 full rows = 256 KB (> L1) |
| 64 | 256 | 29.6 | 64 chunks of 1 KB, 4 KB apart (the default) |
| 256 | 256 | 29.2 | 256 chunks, 4 KB apart |
| any | 64 | 20.5–26.4 | short rows: loop overhead dominates |

- **MC barely matters** (within ~5% at fixed KC, NC).
- **KC matters most.** KC = 16 → 40–43; KC ≥ 64 → 29–32.
- **Same tile size, different shape, different speed:** KC=16/NC=1024 and
  KC=64/NC=256 are *both* 64 KB tiles, but run 43 vs 30 GFLOPS. The fast one
  is 16 consecutive full rows (one contiguous 64 KB block); the slow one is
  64 separate 1 KB pieces spaced 4 KB apart.
- **NC = 64 is slowest** regardless: inner loop of 64 floats = 16 NEON
  vectors, too short to amortize loop setup per (i, k).

## 3. Why: set conflicts inside the B tile (hypothesis, strongly supported)

- A cache picks a set from middle address bits. Rows of B are `n · 4` bytes
  apart. At n = 1024 that's 4096 B, a power of two, so **row r of the tile
  and row r+1 land in the same sets**. The tile's 64 pieces compete for a
  few sets and evict each other, even though 64 KB fits in a 128 KB L1 by
  capacity.
- Same story as naive's cliff at 2048 (Stage 1 §5), now hitting the tile.
- Evidence:
  1. Default blocking wins at 1000/2000 and loses at 1024/2048: only the
     stride changed.
  2. Fewer tile rows (KC = 16) → fewer pieces competing → fast.
  3. A contiguous tile (NC = N) of the same size → fast.
- Not confirmed with counters (L1 miss counts per config would settle it).
  L1 associativity on M4 Pro isn't published; the argument doesn't need it.

## 4. No single block size works everywhere

`stage2_blocked_kc16.csv` — the sweep winner (64/16/1024) at other sizes:

| n | ikj | blocked 64/64/256 | blocked 64/16/1024 |
|---|---|---|---|
| 512 | 31.7 | 35.5 | **41.9** |
| 1000 | 31.8 | **42.6** | 41.3 |
| 1024 | 32.4 | 29.6 | **37.9** |
| 2000 | 30.9 | **42.1** | 36.3 |
| 2048 | 32.1 | 26.8 | **31.0** |

- 64/16/1024 fixes 1024 but is worse than the default at 2000, and still
  doesn't beat ikj at 2048 (NC = 1024 is only half a row there, so the tile
  is 16 pieces spaced 8 KB apart: strided again).
- **The tile's performance depends on where B's rows happen to fall in
  memory.** Tuning block sizes per n is fighting the symptom.

## 5. What fixes it: packing (Stage 2b)

- Copy each KC × NC tile of B into a **contiguous, aligned buffer** before
  using it. Then the tile's layout no longer depends on `ldb`: no set
  conflicts at any n, always 16 B aligned (Stage 1 §6), and the copy cost
  (KC·NC floats) is amortized over MC rows.
- This is what production BLAS does (GotoBLAS / BLIS packing).
- Prediction: packed blocking ≈ the 42 GFLOPS seen at non-power-of-two
  sizes, at *every* n including 1024 and 2048.

## 6. Where blocking stops helping

- Even the best blocked run (~43) is ~30% of the rough ~140 GFLOPS P-core
  estimate. The inner loop is still ikj's: 2 loads + 1 store per FMA.
- Blocking fixes **where** B comes from (L1 instead of L2). It doesn't fix
  **how many** loads and stores each FMA needs. That's register blocking
  (keep a small tile of C in registers across k), the next lever after
  packing.
- On x86 with 1–2 MB L2 per core, ikj should fall off at large n where it
  doesn't here, so blocking's gain over ikj should be larger. To measure.

## Summary of claims and evidence

| Claim | Status | Evidence |
|---|---|---|
| Blocking gives +34–36% over ikj at n = 1000, 2000 | measured | §1 |
| Blocking is 9–17% slower than ikj at n = 1024, 2048 | measured | §1 |
| KC dominates; MC barely matters | measured | §2 |
| Same-size tiles differ 43 vs 30 by shape (contiguous vs strided) | measured | §2 |
| Cause is cache-set conflicts from power-of-two row stride | supported by 3 experiments, no counters | §3 |
| No single block size is best across n | measured | §4 |
| Packing will fix it | prediction | §5 |

## Reproduce

```bash
cmake --preset release && cmake --build --preset release
B=./build/release/bench
$B --impl ikj     --reps 5 512 1000 1024 2000 2048 > results/stage2_ikj.csv
$B --impl blocked --reps 5 512 1000 1024 2000 2048 > results/stage2_blocked_default.csv
$B --impl blocked --reps 5 --mc 64 --kc 16 --nc 1024 512 1000 1024 2000 2048 > results/stage2_blocked_kc16.csv

echo "mc,kc,nc,implementation,size,run,seconds,gflops" > results/stage2_sweep_1024.csv
for mc in 16 64 256; do for kc in 16 64 256; do for nc in 64 256 1024; do
  $B --impl blocked --reps 3 --mc $mc --kc $kc --nc $nc 1024 2>/dev/null \
    | tail -n +2 | sed "s/^/$mc,$kc,$nc,/" >> results/stage2_sweep_1024.csv
done; done; done
```
