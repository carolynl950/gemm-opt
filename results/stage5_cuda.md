# Stage 5: CUDA SGEMM on a Tesla T4 — results and analysis (in progress)

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

CSVs: `stage5_square_t4.csv`, `stage5_prefill_t4.csv`,
`stage5_decode_cold_t4.csv`, `gpu_clocks_{square,prefill,decode}.csv`.
First run (1-run warmup, no clock log): `stage5a_square_t4_first_run.csv`.

## 1. The T4 is power-limited: cuBLAS sustains ~3.3–3.6 TFLOPS, not 7

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
- Takeaway: on a power-limited part, the spec-sheet peak is not reachable
  for dense compute. Measure the clock, don't assume it.

## 2. Naive → coalesced (5a → 5b)

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

## 3. Decode (T = 1, cold L2)

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

## Summary of claims and evidence

| Claim | Status | Evidence |
|---|---|---|
| Coalesced mapping is 8–12× naive, same arithmetic | measured | §2 |
| Mechanism is transactions per warp load | standard CUDA memory model; not profiled (Nsight not run) | §2 |
| T4 cuBLAS sustains ~3.3–3.6 TFLOPS under the 70 W cap; 7,000 was a pre-cap burst | measured (clock log + two runs) | §1 |
| GPU decode is weight-bandwidth-bound (138–212 GB/s) | measured + arithmetic | §3 |
| Coalesced slowdown with n is L2 → DRAM spill | hypothesis | §2 |

## Next

- **5c: shared-memory tiling** — each block loads a tile of A and B into
  shared memory once and reuses it 32 times (GPU version of Stage 2
  blocking).
- **5d: register tiling** — each thread computes several C elements from
  registers (GPU version of Stage 3).
