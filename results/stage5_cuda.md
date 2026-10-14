# Stage 5: CUDA SGEMM on a Tesla T4 — results and analysis (in progress)

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

`test_cuda`: **68/68** expected (4 kernels × 17 shapes); bench gate passed
for smem on every shape.

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

## Next

- **5d: register tiling** — each thread computes several C elements from
  registers (GPU version of Stage 3).
