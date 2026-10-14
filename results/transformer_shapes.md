# Transformer shapes: prefill vs decode — results and analysis

**One sentence:** prefill reuses every weight T = 512 times and is
compute-bound, so the Stage 1–2 optimizations (and Accelerate's AMX) pay off
fully; decode (T = 1) uses every weight once and is bound by how fast
weights stream from DRAM, which caps *every* kernel at ~45 GFLOPS on this
machine and shrinks Accelerate's lead over plain ikj from ~70× to ~1.5×.

## Setup

| | |
|---|---|
| Model | GPT-2 small: d = 768, MLP = 3072 |
| CPU | Apple M4 Pro, single thread, 128 KB L1d, 16 MB L2 |
| Compiler | Apple clang 17, `-O3 -mcpu=native` |
| GEMM | C (T × N) += A (T × K) · B (K × N); A = activations, B = weights |
| Prefill | T = 512, 5 reps |
| Decode | T = 1, 50 reps (each call is 10–300 µs) |
| Warm | back-to-back reps: weights stay in L2 (all ≤ 9.4 MB < 16 MB) |
| Cold | 64 MB written before each timed run: weights come from DRAM, like a real model where all layers' weights don't fit in cache |
| Blocked/packed | MC = 64, KC = 64, NC = 256 (Stage 2 defaults) |
| BLAS | Accelerate (AMX) |

| Layer | M × N × K | Weights |
|---|---|---|
| QKV projection | T × 2304 × 768 | 7.1 MB |
| Attention out | T × 768 × 768 | 2.4 MB |
| MLP up | T × 3072 × 768 | 9.4 MB |
| MLP down | T × 768 × 3072 | 9.4 MB |

Medians in GFLOPS. CSVs: `transformer_{prefill,decode}_{warm,cold}.csv`.

## Arithmetic intensity: why the two phases differ

- **Prefill**, QKV: 2·512·2304·768 = 1.8 GFLOP over ~13 MB of A + B + C
  ≈ **135 flops per byte**. Every weight is loaded once and used 512 times.
- **Decode**, any layer: 2·N·K flops over 4·N·K bytes of weights
  ≈ **0.5 flops per byte**. Every weight is loaded once and used once.
- At 0.5 flop/byte, a machine that streams B at X GB/s can't exceed
  0.5·X GFLOPS no matter how good its arithmetic is.

## 1. Prefill (T = 512)

| Layer | naive | ikj | blocked | packed | Accelerate |
|---|---|---|---|---|---|
| QKV | 2.7 | 31.6 | 42.5 | 42.3 | 3198 |
| attn out | 2.8 | 32.4 | 43.3 | 40.3 | 1599 |
| MLP up | 2.2 | 32.5 | **27.0** | **42.5** | 2425 |
| MLP down | 2.4 | 31.5 | 42.2 | 43.2 | 2240 |

(warm; cold is within ~5% for our kernels, see §3)

- Same ranking as the square benchmarks: naive ≪ ikj < blocked ≈ packed.
- **MLP up reproduces the Stage 2a cliff on a real shape.** N = 3072 →
  B's row stride is 12,288 B = 3 × 4096, so tile rows alias in the cache
  sets, and blocked (27.0) falls *below* ikj (32.5). Packing restores 42.5.
  Real model dimensions hit this, not just artificial powers of two.

## 2. Decode (T = 1), warm vs cold

GFLOPS, with the weight-streaming rate (K·N·4 bytes / time) in parentheses.

| Layer | ikj warm | ikj cold | blocked cold | packed cold | Accel. warm | Accel. cold |
|---|---|---|---|---|---|---|
| QKV | 32.7 (65 GB/s) | 30.2 (60) | 11.1 (22) | 8.9 (18) | 327 (653) | 44.9 (90) |
| attn out | 32.6 (65) | 29.9 (60) | 12.1 (24) | 9.0 (18) | 308 (616) | 44.6 (89) |
| MLP up | 32.6 (65) | 30.6 (61) | 10.9 (22) | 8.2 (16) | 298 (596) | 45.0 (90) |
| MLP down | 31.7 (63) | 30.3 (61) | 11.7 (23) | 8.6 (17) | 303 (606) | 44.2 (88) |

naive: 2.2–2.7 GFLOPS in all decode cases.

### Accelerate hits a bandwidth wall
- Warm, Accelerate streams weights at ~600 GB/s (from L2). Cold, it drops
  **7×** to **88–90 GB/s on every layer**, regardless of layer size.
- The same rate on four different shapes is what a bandwidth limit looks
  like. At 0.5 flop/byte, 90 GB/s → 45 GFLOPS, which is exactly what it
  gets: Accelerate is **on the roofline** for decode.
- Hypothesis: ~90 GB/s is the DRAM bandwidth one thread (plus whatever
  Accelerate uses internally) can pull on this machine. The M4 Pro's
  advertised system bandwidth is higher; one core typically can't saturate
  it. Not verified with a separate bandwidth test.

### ikj barely notices
- ikj cold is only ~7% slower than warm (30 vs 32), at 60 GB/s.
- 60 GB/s is under the ~90 GB/s Accelerate shows is available, so ikj is
  **not** bandwidth-bound even in decode. It's still limited by its own
  inner loop (~32 GFLOPS everywhere since Stage 1).
- ikj walks B row by row, front to back: one perfectly sequential stream,
  ideal for the hardware prefetcher.

### Blocking and packing hurt decode
- Blocked and packed cold are **2.5–3.5× slower than ikj**.
- **Packed:** with T = 1 there's nothing to reuse a packed tile for. The
  copy reads and writes all of B, *then* the compute reads it again:
  ~3× the memory traffic of ikj for the same flops. Warm, it's already at
  half of ikj (14–16 vs 32); cold makes it worse.
- **Blocked:** no copy, but the tile walk jumps through B in 1 KB pieces
  (64 rows × 256 columns per tile) instead of one sequential stream.
  Warm, that costs ~15%; cold, it costs 2.5×. Hypothesis: the prefetcher
  can't run ahead across the jumps, so DRAM latency is exposed. Not
  verified with counters.

## 3. Prefill doesn't care about cold caches

| Layer | packed warm | packed cold | Accel. warm | Accel. cold |
|---|---|---|---|---|
| QKV | 42.3 | 42.5 | 3198 | 2055 |
| attn out | 40.3 | 41.0 | 1599 | 1287 |
| MLP up | 42.5 | 43.3 | 2425 | 1969 |
| MLP down | 43.2 | 43.8 | 2240 | 2174 |

- Our kernels: no change. Loading 7 MB of weights once from DRAM at
  ~90 GB/s takes ~80 µs; the multiply takes ~40 ms. Compute-bound.
- Accelerate loses up to 36% cold (QKV): at ~3 TFLOPS the whole multiply
  takes ~0.6 ms, so the one-time DRAM load is no longer negligible.

## 4. The ML takeaway

| | Prefill | Decode (cold) |
|---|---|---|
| Bound by | compute | weight bandwidth |
| Best own kernel | packed, ~42 GFLOPS | ikj, ~30 GFLOPS |
| Accelerate | ~1600–3200 GFLOPS | ~45 GFLOPS |
| Accelerate / best own | ~40–75× | **~1.5×** |
| Blocking/packing | +30% over ikj | **2.5–3.5× slower** than ikj |

- In decode, the AMX's arithmetic doesn't matter; the memory system does.
  This is why LLM inference work focuses on **batching** (T > 1 raises
  flops per byte), **weight quantization** (fewer bytes per weight), and
  **KV caching**, not faster matmul arithmetic.
- **The right kernel depends on the phase.** A library tuned for prefill
  (blocking + packing) is the wrong choice at T = 1. Production stacks use
  separate GEMV paths for decode.

## Summary of claims and evidence

| Claim | Status | Evidence |
|---|---|---|
| Prefill ranking matches square benchmarks | measured | §1 |
| MLP up (N = 3072) triggers the blocked set-conflict cliff; packing fixes it | measured | §1 |
| Accelerate decode cold is flat at 88–90 GB/s ≈ 45 GFLOPS on all layers | measured | §2 |
| That's a DRAM bandwidth limit (roofline at 0.5 flop/byte) | strongly supported, no bandwidth microbenchmark | §2 |
| ikj isn't bandwidth-bound in decode (60 < 90 GB/s) | measured + inference | §2 |
| Packing triples memory traffic at T = 1 | arithmetic | §2 |
| Blocked decode slowdown is from defeating the prefetcher | hypothesis | §2 |
| Prefill is insensitive to cold caches for our kernels | measured | §3 |

## Caveats

- Single thread. Multi-threaded decode would pull more DRAM bandwidth and
  move the wall; Stage 4 (OpenMP) should measure that.
- Cold decode reps vary up to 2–3× (min to max); medians of 50 are used.
  Warm prefill is within ~10% except Accelerate (up to 2.3×).
- GPT-2 small weights total ~500 MB in fp32, far past L2, so "cold" is the
  realistic decode case; "warm" is an upper bound.

## Reproduce

```bash
cmake --preset release && cmake --build --preset release
B=./build/release/bench
shapes() { T=$1; echo "--shape ${T}x2304x768 --shape ${T}x768x768 --shape ${T}x3072x768 --shape ${T}x768x3072"; }
$B --reps 5         $(shapes 512) > results/transformer_prefill_warm.csv
$B --reps 5  --cold $(shapes 512) > results/transformer_prefill_cold.csv
$B --reps 50        $(shapes 1)   > results/transformer_decode_warm.csv
$B --reps 50 --cold $(shapes 1)   > results/transformer_decode_cold.csv
```
