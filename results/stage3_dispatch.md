# Small-M dispatch for `simd` — results

**Change (one line in `matmul_simd`):** if M < MR (8), call `matmul_ikj`
instead of packing and running the 8 × 12 micro-kernel.

**Why:** with fewer rows than the tile, most FMAs multiply zero padding,
every tile takes the edge path, and packing B costs as much as using it
(`stage3_simd.md` §4).

## 1. Picking the threshold: measured crossover

QKV weight shape (M × 2304 × 768), cold cache, 30 reps, median GFLOPS.
`dispatch_crossover_{ikj,simd}.csv` (simd measured *before* dispatch).

| M | 1 | 2 | 3 | 4 | 6 | **8** | 12 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ikj | 26.2 | 31.9 | 32.7 | 32.6 | 32.4 | 32.0 | 32.5 | 32.6 | 33.0 | 32.6 | 32.9 | 32.7 |
| simd | 4.4 | 8.2 | 13.1 | 17.5 | 25.2 | **35.3** | 42.2 | 55.8 | 68.3 | 77.0 | 85.5 | 92.4 |
| simd / ikj | 0.17 | 0.26 | 0.40 | 0.54 | 0.78 | **1.10** | 1.30 | 1.71 | 2.07 | 2.36 | 2.60 | 2.82 |

- **Crossover is exactly at the tile height**: simd loses at M = 6, wins at
  M = 8. Below MR, a fraction (8 − M)/8 of each tile is padding.
- simd's throughput rises roughly in proportion to M up to ~32, then
  flattens toward the ~115 seen at large shapes. ikj is flat at ~32 for
  every M: its inner loop doesn't care how many rows there are.
- These are batched-decode sizes: M = number of tokens processed together.
  Batching 8+ sequences is exactly what moves decode onto the GEMM path.

## 2. After dispatch

Decode, all four GPT-2 layers, cold, 50 reps (`stage3_dispatch_decode_cold.csv`):

| Layer | simd before | simd after | ikj (for reference) |
|---|---|---|---|
| QKV | 4.2 | **27.3** | 30.2 |
| attn out | 3.6 | **29.5** | 29.9 |
| MLP up | 4.3 | **30.4** | 30.6 |
| MLP down | 3.5 | **29.7** | 30.3 |

~7× better at T = 1; now matches ikj, as it should (it *is* ikj there).

Around the threshold (`stage3_dispatch_edge.csv`, QKV, cold):

| M | 4 | 6 | 7 | 8 | 9 | 12 |
|---|---|---|---|---|---|---|
| simd (after) | 31.6 | 32.5 | 32.0 | 34.4 | **29.0** | 41.2 |

- M = 9 dips below ikj: one full tile plus a 1-row edge tile that wastes
  7/8 of its FMAs. M = 9–11 is the remaining weak spot. Fixing it would
  mean handling leftover rows with ikj (split dispatch), not worth the
  complexity until a workload needs those sizes.

## 3. Correctness

`test_kernels` passes on release, asan, ubsan. Shapes with M < 8 now go
through ikj (already tested). The micro-kernel's partial-row edge path is
still covered by M ≥ 8 shapes with leftovers: 9, 33, 63, 65, 129.

## Reproduce

```bash
args=""; for m in 1 2 3 4 6 8 12 16 24 32 48 64; do args="$args --shape ${m}x2304x768"; done
./build/release/bench --impl ikj  --reps 30 --cold $args > results/dispatch_crossover_ikj.csv
./build/release/bench --impl simd --reps 30 --cold $args > results/dispatch_crossover_simd.csv   # before the dispatch line
```
