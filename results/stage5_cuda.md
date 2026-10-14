# Stage 5: CUDA SGEMM on a Tesla T4 — results and analysis

**Headline:** from a naive kernel at ~62 GFLOPS to a register-tiled kernel at
**3,545 GFLOPS (57×), 94% of cuBLAS** at n = 2048, measured in the same run
on the same power-capped T4. Each step targets one bottleneck: memory
coalescing (8–12×), shared-memory reuse (1.3–1.5×), register tiling (2–4×).

**5d mechanism (one sentence):** giving each thread an 8×8 patch of C in
registers turns each k-step into 16 shared-memory reads for 64 FMAs instead
of 2 reads per FMA, and 256-thread blocks let two blocks share an SM, taking
the kernel from shared-memory-bound to within 6–13% of cuBLAS on large shapes.

**5c mechanism (one sentence):** loading 32×32 tiles of A and B into shared
memory once per block and reusing each value 32 times cuts global-memory
traffic ~32×, but only buys **1.3–1.5×** over coalesced (to ~880–985 GFLOPS,
17–25% of cuBLAS), because every FMA still needs two shared-memory reads, so
the bottleneck moves from global memory to shared memory and instruction issue.

**5b mechanism (one sentence):** mapping a warp's 32 threads to 32
consecutive *columns* instead of 32 consecutive *rows* turns each warp's
load of B into 4 contiguous 32-byte transactions instead of a scatter, which
makes the same algorithm **~9–11× faster** (61 → 502–652 GFLOPS) with no
change to the arithmetic.

## Setup

| | |
|---|---|
| GPU | Tesla T4 (Turing, sm_75): 40 SMs, 4 MB L2, 70 W board power limit, boost 1590 MHz |
| Spec FP32 peak | 2560 cores × 2 flops × 1.59 GHz ≈ 8.1 TFLOPS (**only at boost clock; see §1**) |
| Spec memory | ~320 GB/s GDDR6 |
| Host | Google Colab; CUDA via CMake (`GEMM_USE_CUDA`, arch detected = sm_75) |
| Timing | CUDA events; ≥ 300 ms untimed warmup per kernel per shape; 10 reps (square), 20 (prefill), 100 (decode) |
| Correctness | each kernel gated against cuBLAS before timing (magnitude-based tolerance); `test_cuda` vs CPU naive on padded, NaN-poisoned shapes: **51/51** (naive, coalesced, cuBLAS × 17 shapes) |
| Clock logging | `nvidia-smi` every 500 ms during each benchmark → `gpu_clocks_*.csv` |

CSVs, one set per Colab run (each run measures every kernel that existed then):

| run | files | kernels |
|---|---|---|
| 5a first run (1-run warmup, no clock log) | `stage5a_square_t4_first_run.csv` | naive, cuBLAS |
| 5b run | `stage5b_{square,prefill,decode_cold}_t4.csv`, `stage5b_gpu_clocks_*.csv` | + coalesced |
| 5c run | `stage5c_{square,prefill,decode_cold}_t4.csv`, `stage5c_gpu_clocks_*.csv` | + smem |
| 5d run | `stage5d_{square,prefill,decode_cold}_t4.csv`, `stage5d_gpu_clocks_*.csv` | + regtile |

**Compare kernels within one run, not across runs.** Separate Colab sessions
can land on different physical T4s and thermal states: the same coalesced
kernel measured 652 vs 714 GFLOPS (n = 512) and 537 vs 602 (n = 2048) in the
5b and 5c runs, ~10–12% apart. Ratios within a run are stable.

## 1. The T4 is power-limited: cuBLAS sustains ~3.3–5.7 TFLOPS, not 7–8

| | first run (1 warmup run) | current run (300 ms warmup) |
|---|---|---|
| cuBLAS, n = 2048 | 7,020 (one run), dropping to 5,700 | **3,498–3,571**, all 10 runs |
| cuBLAS, n = 4096 | 5,315 first run, then ~3,040 | **3,232–3,300**, all 10 runs |

Clock log during the benchmarks:

| run | samples | throttled | reason | max power |
|---|---|---|---|---|
| square | 77 | 12 | `0x4` (software power cap) | 85.8 W |
| prefill | 15 | 7 | `0x4` | 86.2 W |
| decode | 12 | 11 | `0x4` | 76.6 W |

- Dense FMA at the 1590 MHz boost clock pulls more than the T4's 70 W
  limit, so the driver lowers the clock (logged as low as 660–765 MHz in
  brief dips).
- The first run's ~7,000 was a **burst** measured before the power cap
  engaged. With a proper warmup, cuBLAS is measured in its **sustained**
  state, and it's very stable (±2% across runs).
- Our kernels draw far less power (they wait on memory), so they run at
  full clock. Comparisons below use cuBLAS **sustained**, which is what a
  real workload gets.
- The 5c run (a later Colab session) measured cuBLAS at 3,816–5,705 with
  the same warmup, still power-capped (22 of 83 samples). So "sustained"
  depends on the session's card and temperature: report the range,
  ~3.3–5.7 TFLOPS, and compare kernels within a run.
- Takeaway: on a power-limited part, the spec-sheet peak is not reachable
  for dense compute. Measure the clock, don't assume it.

## 2. Naive → coalesced (5a → 5b), from the 5b run

### Square (GFLOPS, median)

| n | naive | coalesced | coalesced / naive | cuBLAS (sustained) | coalesced / cuBLAS |
|---|---|---|---|---|---|
| 512 | 55.7 | 652.1 | 11.7× | 3,745 | 17% |
| 1024 | 60.7 | 573.9 | 9.5× | 3,178 | 18% |
| 2048 | 61.4 | 537.0 | 8.7× | 3,499 | 15% |
| 4096 | 61.7 | 501.9 | 8.1× | 3,298 | 15% |

### GPT-2 prefill (T = 512)

| Layer | naive | coalesced | cuBLAS |
|---|---|---|---|
| QKV (512×2304×768) | 61.0 | 484.3 | 3,346 |
| attn out (512×768×768) | 59.2 | 641.1 | 4,757 |
| MLP up (512×3072×768) | 60.9 | 499.9 | 3,497 |
| MLP down (512×768×3072) | 59.3 | 534.1 | 3,055 |

### Why: what one warp reads per k step

| access | naive (x → row) | coalesced (x → col) |
|---|---|---|
| `A[row][k]` | 32 addresses `lda` apart → up to 32 transactions | 1 address → broadcast |
| `B[k][col]` | 1 address → broadcast | 32 consecutive floats → 4 × 32 B |
| `C[row][col]` store | 32 scattered | 32 consecutive |

- Same flops, same algorithm, same number of loads per thread. Only the
  *pattern* changed, and it bought ~9–11×.
- The GPU version of Stage 1's lesson (ijk → ikj on the CPU): the access
  pattern of the innermost "loop" decides speed. On a GPU the innermost
  loop is the 32 threads of a warp.
- Coalesced slows from 652 to 502 as n grows. Hypothesis: every thread
  still re-reads its whole row of A and column of B from global memory
  (no reuse in fast storage), so as matrices outgrow the 4 MB L2, more
  of those reads go to DRAM. Shared-memory tiling (5c) targets exactly that.

## 3. Decode (T = 1, cold L2), from the 5b run

| Layer | naive | coalesced | cuBLAS | cuBLAS weight GB/s |
|---|---|---|---|---|
| QKV | 17.9 | 48.7 | 96.0 | 192 |
| attn out | 11.9 | 35.1 | 69.0 | 138 |
| MLP up | 16.3 | 47.8 | 106.0 | 212 |
| MLP down | 12.6 | 37.4 | 88.6 | 177 |

- GPU decode is bandwidth-bound like the CPU: 0.5 flops/byte, so GFLOPS
  × 2 = GB/s of weights. cuBLAS streams 138–212 GB/s against a ~320 GB/s
  spec. Bigger layers get closer (more bytes per launch to amortize
  launch and ramp-up).
- Compared to the M4 Pro CPU, cold, single thread (Accelerate ~45 GFLOPS,
  ~90 GB/s): the T4 is ~2× in decode, vs ~1.2–1.5× in prefill (Accelerate
  ~2,200–3,200 vs cuBLAS sustained ~3,000–4,800). Different machines and
  libraries, so a rough comparison only.
- 11 of 12 clock samples were power-capped during decode, a reminder that
  decode numbers on this card include clock effects too.

## 4. Shared-memory tiling (5c), from the 5c run

Kernel: `cuda/kernels/smem.cu`. 32×32 thread block → 32×32 tile of C. Per
k-step of 32: every thread loads one float of the A tile and one of the B
tile (both coalesced), `__syncthreads()`, 32 FMAs from shared memory,
`__syncthreads()`. Out-of-range tile elements are zero-filled; the C bounds
check is at the end so every thread reaches every barrier.

`test_cuda`: bench gate passed for smem on every shape; full `test_cuda`
confirmed in the 5d run (95/95, below).

### Square (GFLOPS, median of 10; min–max shown for the new kernel)

| n | naive | coalesced | **smem** (min–max) | smem / coalesced | cuBLAS | smem / cuBLAS |
|---|---|---|---|---|---|---|
| 512 | 56.0 | 714.5 | **950.0** (944–961) | 1.33× | 3,852 | 25% |
| 1024 | 60.7 | 722.2 | **985.4** (963–1008) | 1.36× | 5,705 | 17% |
| 2048 | 61.8 | 602.4 | **904.6** (900–913) | 1.50× | 4,083 | 22% |
| 4096 | 61.7 | 577.9 | **876.1** (871–890) | 1.52× | 4,040 | 22% |

### GPT-2 prefill (T = 512)

| Layer | coalesced | smem | smem / coalesced | cuBLAS |
|---|---|---|---|---|
| QKV (512×2304×768) | 561.4 | 843.8 | 1.50× | 4,018 |
| attn out (512×768×768) | 682.3 | 918.3 | 1.35× | 3,830 |
| MLP up (512×3072×768) | 553.1 | 817.0 | 1.48× | 4,335 |
| MLP down (512×768×3072) | 578.0 | 876.9 | 1.52× | 3,816 |

- **1.3–1.5×, not 32×.** Global loads per FMA dropped ~32×, so global
  memory is no longer the limit. What's left per FMA: 2 shared-memory reads
  (`As[ty][kk]` broadcast, `Bs[kk][tx]`) + 1 FMA. The kernel is now bound by
  shared-memory bandwidth / load-instruction issue (hypothesis; not
  profiled with Nsight Compute).
- **The gain grows with n** (1.33× at 512 → 1.52× at 4096): coalesced slows
  as data outgrows the 4 MB L2 (§2), smem barely does (950 → 876). Supports
  §2's hypothesis that coalesced's slowdown was global-memory traffic.
- **Occupancy:** a 1024-thread block fills a T4 SM's 1024-thread limit, so
  one block per SM; nothing hides the `__syncthreads()` stalls. Smaller
  blocks with more work per thread (5d) address this and the 2-loads-per-FMA
  problem together, the same fix as the CPU's register tile (Stage 3).
- No bank conflicts by construction: `Bs[kk][tx]` hits 32 different banks,
  `As[ty][kk]` is a broadcast.

### Decode (T = 1, cold): smem is *slower* than coalesced

| Layer | naive | coalesced | smem | cuBLAS |
|---|---|---|---|---|
| QKV | 20.4 | 52.3 | 25.0 | 96.1 |
| attn out | 13.7 | 38.2 | 17.0 | 70.6 |
| MLP up | 18.4 | 51.5 | 23.3 | 107.0 |
| MLP down | 14.3 | 40.5 | 18.0 | 89.5 |

- smem is ~0.45× coalesced in decode. With M = 1, only 1 of each block's 32
  rows has work, so there's no reuse to gain, but the block still pays a
  barrier every 32 k-steps. And N / 32 = 24–96 blocks for 40 SMs leaves too
  little parallel work to hide those stalls (hypothesis). Same lesson as
  the CPU: a reuse optimization costs you when there's nothing to reuse.

### Clocks during the 5c run

| run | throttled samples | reason | min clock | max power |
|---|---|---|---|---|
| square | 22 / 83 | `0x4` power cap | 750 MHz | 74.5 W |
| prefill | 9 / 17 | `0x4` | 810 MHz | 68.7 W |
| decode | 12 / 16 | `0x4` | 1350 MHz | 71.7 W |

## 5. Register tiling (5d), from the 5d run

Kernel: `cuda/kernels/regtile.cu`.

Correctness: `test_cuda` **95/95** (5 kernels × 19 shapes, including partial
128-wide tiles and K not a multiple of 8), padded strides and NaN-poisoned
padding included.

| | 5c smem | 5d regtile |
|---|---|---|
| outputs per thread | 1 | 64 (8×8, registers) |
| shared reads per FMA | 2 | 0.25 |
| threads per block | 1024 | 256 |
| C tile per block | 32×32 | 128×128 |
| k-step | 32 | 8 |

Design details that matter (all in the code comments):
- **Strided ownership** (thread owns cols `tx + 16j`, rows `ty + 16i`): a
  warp's `Bs` reads hit 16 consecutive floats (no bank conflict) and its C
  writes are coalesced. Consecutive ownership (`tx*8 + j`) would make the
  reads 4-way bank-conflicted.
- **A tile stored transposed with 4 floats of padding**: without padding the
  8 threads writing one A row would hit the same bank (8-way conflict).
- **`__launch_bounds__(256)`** so the compiler budgets registers for 64
  accumulators without spilling.

### Square (GFLOPS, median of 10)

| n | naive | coalesced | smem | **regtile** (min–max) | cuBLAS | **regtile / cuBLAS** | regtile / smem |
|---|---|---|---|---|---|---|---|
| 512 | 55.7 | 660.2 | 899.5 | **1,926** (1,910–1,928) | 3,904 | 49% | 2.1× |
| 1024 | 60.6 | 583.7 | 879.4 | **3,091** (3,086–3,099) | 3,543 | 87% | 3.5× |
| 2048 | 61.5 | 572.5 | 866.1 | **3,545** (3,491–3,725) | 3,787 | **94%** | 4.1× |
| 4096 | 61.7 | 534.7 | 842.5 | **3,168** (3,148–3,181) | 3,651 | 87% | 3.8× |

### GPT-2 prefill (T = 512)

| Layer | smem | **regtile** | cuBLAS | regtile / cuBLAS | blocks (128×128 tiles) |
|---|---|---|---|---|---|
| attn out (512×768×768) | 1,017.5 | 2,236.6 | 4,346.9 | 51% | 24 |
| MLP down (512×768×3072) | 816.7 | 2,331.8 | 3,466.6 | 67% | 24 |
| QKV (512×2304×768) | 758.2 | 2,856.0 | 3,656.1 | 78% | 72 |
| MLP up (512×3072×768) | 741.6 | 2,701.4 | 3,502.6 | 77% | 96 |

### Where regtile falls short: not enough blocks to fill 40 SMs

Blocks launched = ⌈M/128⌉ × ⌈N/128⌉. At 2 blocks per SM the T4 holds 80 at once.

| shape | blocks | fraction of 80 slots | regtile / cuBLAS |
|---|---|---|---|
| 512² | 16 | 0.2 | 49% |
| attn out, MLP down | 24 | 0.3 | 51–67% |
| 1024² | 64 | 0.8 | 87% |
| QKV, MLP up | 72–96 | 0.9–1.2 | 77–78% |
| 2048² | 256 | 3.2 | 94% |

- The pattern is clean: with fewer blocks than SMs, some SMs sit idle the
  whole kernel. 512² launches 16 blocks for 40 SMs, so 60% of the GPU does
  nothing. Big 128×128 tiles are the right choice for big matrices and the
  wrong one for small ones (hypothesis for the cause, but the correlation
  across all 8 shapes is strong; occupancy not profiled).
- cuBLAS picks among many kernels (tile sizes, split-K) per shape; ours has
  one. Fix options: smaller tiles (64×64 → 4× the blocks) or split-K for
  small M·N. Not done.
- 4096 (87%) < 2048 (94%) despite plenty of blocks: hypothesis is L2 misses
  (a 128×4096 strip of A and B per block row no longer fits the 4 MB L2).
  Not tested.

### Decode (T = 1, cold): regtile is the worst kernel

| Layer | coalesced | smem | regtile | cuBLAS |
|---|---|---|---|---|
| QKV | 50.4 | 23.5 | 14.6 | 95.7 |
| attn out | 36.0 | 16.0 | 5.1 | 67.5 |
| MLP up | 50.0 | 22.0 | 19.2 | 105.8 |
| MLP down | 39.0 | 16.9 | 5.5 | 88.6 |

- M = 1 fills 1 of each block's 128 rows: 127/128 of the FMAs multiply zero
  padding, and N/128 = 6–24 blocks leave most SMs idle. Exactly the CPU
  Stage 3 result (8×12 tile at M = 1), and the same fix applies: dispatch
  small M to a different kernel (coalesced is the best of ours, ~half of
  cuBLAS). Not done yet on the GPU.

### Clocks during the 5d run

| run | throttled samples | reason | min clock | max power |
|---|---|---|---|---|
| square | 26 / 87 | `0x4` power cap | 585 MHz | 77.6 W |
| prefill | 11 / 21 | `0x4` | 735 MHz | 75.4 W |
| decode | 19 / 20 | `0x4` | 1320 MHz | 70.7 W |

regtile now draws enough power to be capped too, so regtile and cuBLAS are
compared under the same conditions.

## Summary of claims and evidence

| Claim | Status | Evidence |
|---|---|---|
| Coalesced mapping is 8–12× naive, same arithmetic | measured | §2 |
| Mechanism is transactions per warp load | standard CUDA memory model; not profiled (Nsight not run) | §2 |
| T4 cuBLAS sustains ~3.3–3.6 TFLOPS under the 70 W cap; 7,000 was a pre-cap burst | measured (clock log + two runs) | §1 |
| GPU decode is weight-bandwidth-bound (138–212 GB/s) | measured + arithmetic | §3 |
| Coalesced slowdown with n is L2 → DRAM spill | hypothesis, supported by smem not slowing (§4) | §2, §4 |
| smem is 1.3–1.5× coalesced on prefill/square; 17–25% of cuBLAS | measured | §4 |
| smem is now shared-memory / issue bound | hypothesis (no Nsight) | §4 |
| smem is 0.45× coalesced in decode | measured | §4 |
| Cross-session variance ~10–12% on the same kernel | measured | Setup |
| regtile 3,545 GFLOPS = 94% of cuBLAS at 2048; 57× naive | measured, same run; test_cuda 95/95 | §5 |
| regtile 2–4× smem (8× fewer shared reads per FMA + 2 blocks/SM) | measured; mechanism from arithmetic, not profiled | §5 |
| regtile's gap to cuBLAS on small shapes tracks blocks < SMs | strong correlation across 8 shapes; not profiled | §5 |
| regtile is the worst kernel in decode (M = 1) | measured | §5 |

## Next (optional)

- GPU small-M dispatch: send M < some threshold to coalesced (measure the
  crossover, as on the CPU).
- Smaller tiles or split-K for shapes with fewer blocks than SMs.
- Nsight Compute profiles to confirm the bottleneck hypotheses (shared-memory
  bound in 5c, occupancy in 5d).
