# Stage 4: OpenMP multithreading — results and analysis

**Mechanism (one sentence):** splitting the micro-kernel's 12-column B
panels (the `jr` loop) across threads, with shared packed A and B guarded by
`omp for` barriers, takes prefill from ~115 to ~440 GFLOPS (3.8× at 6
threads); beyond 6 threads it stops scaling at ~46% of the measured 8-core
FMA peak, so the limit is the parallel design, not the cores.

## Setup

Apple M4 Pro: 8 P-cores + 4 E-cores. Apple clang 17 + Homebrew libomp 5.1,
`-O3 -mcpu=native`. Thread count via `OMP_NUM_THREADS`. macOS gives no
thread pinning: with > 8 threads the OS places some on E-cores.

Kernel: `matmul_omp` in `src/kernels/simd.cpp` (same packing and 8×12
micro-kernel as Stage 3).

| | |
|---|---|
| M ≥ 8 | all threads run the jc/pc/ic loops in lockstep; `omp for` over B-panel packing, A-panel packing, and the `jr` loop. Each `omp for` ends in an implicit barrier, so no thread repacks a shared buffer another is still reading |
| M < 8 | split-K: each thread streams a contiguous range of B's rows into a private M × N partial; partials summed into C |

Why `jr` and not `ic`: prefill M = 512 with MC = 128 is only 4 row blocks
for 8 cores. `jr` gives 64–128 panels per block.

Correctness: `test_kernels` 144/144 on release, asan, ubsan; `omp` passes at
1, 2, 3, 8 and 12 threads. Negative control: removing the barrier after
packing A (`nowait`) fails 9–11 `omp` cases every run at 2 and 8 threads.

## 1. Prefill / large shapes: thread scaling

`stage4_scaling.csv`, median of 7

| threads | n = 2048 | speedup | efficiency | prefill QKV (512×2304×768) | speedup |
|---|---|---|---|---|---|
| 1 | 116.9 | 1.00 | 1.00 | 114.8 | 1.00 |
| 2 | 214.2 | 1.83 | 0.92 | 204.5 | 1.78 |
| 4 | 372.4 | 3.19 | 0.80 | 358.6 | 3.12 |
| 6 | **442.2** | **3.78** | 0.63 | **427.4** | **3.72** |
| 8 | 439.5 | 3.76 | 0.47 | 426.2 | 3.71 |
| 10 | 377.2 | 3.23 | 0.32 | 371.8 | 3.24 |
| 12 | 376.3 | 3.22 | 0.27 | 381.3 | 3.32 |

Accelerate (fresh run, `stage4_blas_ref.csv`): 2834 (2048), 2930 (QKV).

- Good scaling to 4 threads (80% efficiency), flat from 6 to 8, **worse**
  at 10–12. Static scheduling gives every thread an equal share, so an
  E-core thread finishes last and every barrier waits for it.

## 2. Is the plateau the hardware? No.

`peak_parallel_{P}_*.csv`: P copies of the single-thread FMA benchmark run
at once (16-chain result, per process):

| processes | per-process GFLOPS | total |
|---|---|---|
| 1 | 128 | 128 |
| 4 | 125 × 4 | 499 |
| 6 | 118–125 | 736 |
| 8 | 117–123 | **963** |

- All 8 P-cores together sustain **~960 GFLOPS**; clocks drop at most ~8%
  under full load.
- `omp` at 8 threads (~440) is **~46% of that**. Single-thread `simd` is at
  ~85% of single-core peak. Something about running in parallel costs ~half.

Hypotheses for the plateau (none tested yet):
1. **Shared bandwidth.** All threads read the same packed A block and
   different B panels; the P-cores may be split across clusters with
   separate L2s, so shared data crosses between them.
2. **Barriers and serial phases.** 3 barriers per (jc, pc, ic) step; packing
   A is only 16 panels per block, so 8 threads get 2 each and wait.
3. **Load imbalance at block edges.** At n = 2048, the second jc block has
   512 columns = 43 panels for 8 threads.
4. Production BLAS libraries parallelize **both** `ic` and `jr` (2-D thread
   grid) and size blocks per cluster. Next experiment.

## 3. Decode (M = 1, cold): the first design was wrong

### Column split (first version): slower than serial ikj

`stage4_decode_cold.csv`. Each thread ran ikj on a 256-column strip of N.

| threads | 1 | 2 | 4 | 8 | 12 |
|---|---|---|---|---|---|
| QKV GFLOPS | **9.6** | 12.7 | 18.0 | 16.1 | 24.7 |

- With **1 thread**, 9.6 vs plain ikj's ~30. Same work, same code, only the
  access pattern differs: a 256-column strip walks B in 1 KB pieces one
  full row apart, instead of one front-to-back stream.
- This is the **same slowdown blocked had in cold decode** (~11 GFLOPS,
  `transformer_shapes.md` §2), now isolated to a single variable. Strong
  support for that write-up's hypothesis: DRAM streaming depends on the
  prefetcher seeing one long sequential stream.

### Split-K (current version)

`stage4_decode_cold_splitk.csv`. Each thread streams a contiguous range of
B's rows, then partials are summed. GFLOPS (weight GB/s).

| threads | QKV | attn out | MLP up | MLP down |
|---|---|---|---|---|
| 1 | 25.3 (51) | 23.8 (48) | 28.4 (57) | 28.7 (57) |
| 2 | 30.7 (61) | 16.9 (34) | 34.0 (68) | 26.6 (53) |
| 4 | 30.3 (61) | 15.9 (32) | **36.0 (72)** | 30.2 (60) |
| 8 | 22.6 (45) | 9.2 (18) | 30.9 (62) | 25.4 (51) |
| 12 | 19.4 (39) | 7.9 (16) | 26.1 (52) | 25.4 (51) |

- 1 thread back near ikj (25–29 vs ~30): the remaining gap is the zeroing
  and reduction of partials.
- More threads help **at most ~1.25×** (MLP up, 4 threads), and hurt the
  small attention-out layer (2.4 MB, ~40 µs per call), where thread
  start-up and the reduction cost more than the streaming they split.
- Run-to-run spread is large in cold decode (up to 6×, min to max); medians
  of 50.

### What this says about the decode bandwidth wall

- Best multi-threaded `omp` decode: ~72 GB/s. Accelerate single call:
  ~90 GB/s. Apple's advertised M4 Pro memory bandwidth is far higher.
- So **this experiment doesn't settle** whether ~90 GB/s is a per-core or a
  chip limit: our decode kernel doesn't reach 90 GB/s even with threads.
  The clean test is a multi-threaded read-bandwidth microbenchmark
  (STREAM-style), the memory counterpart of `peak`. Not done.

## Summary of claims and evidence

| Claim | Status | Evidence |
|---|---|---|
| omp prefill ~440 GFLOPS, 3.8× at 6 threads, flat to 8, worse at 10–12 | measured | §1 |
| 8 P-cores sustain ~960 GFLOPS FMA; clocks drop ≤ 8% | measured | §2 |
| The plateau is the parallel design (~46% of 8-core peak) | measured + inference | §2 |
| Which part of the design (bandwidth, barriers, imbalance) | hypothesis | §2 |
| Column-split decode is 3× slower than serial ikj at 1 thread | measured | §3 |
| DRAM streaming needs one sequential stream per thread (prefetcher) | strongly supported (single-variable experiment), no counters | §3 |
| Split-K decode gains ≤ 1.25× from threads; small layers lose | measured | §3 |
| Whether ~90 GB/s is a per-core or chip limit | open | §3 |

## Reproduce

```bash
cmake --preset release && cmake --build --preset release   # needs: brew install libomp
B=./build/release/bench
for t in 1 2 4 6 8 10 12; do OMP_NUM_THREADS=$t $B --impl omp --reps 7 2048 --shape 512x2304x768; done
for t in 1 2 4 8 12; do OMP_NUM_THREADS=$t $B --impl omp --reps 50 --cold \
  --shape 1x2304x768 --shape 1x768x768 --shape 1x3072x768 --shape 1x768x3072; done
for P in 1 4 6 8; do for i in $(seq $P); do ./build/release/peak > results/peak_parallel_${P}_$i.csv & done; wait; done
```
