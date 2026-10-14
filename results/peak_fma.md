# Single-core FP32 FMA peak — measured

**Result:** one M4 Pro P-core sustains **~135 GFLOPS** FP32 (median of the
16- and 20-chain plateau; max single run 137.4). The Stage 3 `simd` kernel's
~115 GFLOPS is therefore **~85% of measured peak**, not "~80% of an
estimate".

## Method

`src/peak.cpp` (`./build/release/peak > results/peak_fma.csv`):

- C independent chains `acc_c += x * y`, each one NEON `fmla.4s`
  (4 lanes × 2 flops = 8 flops). Registers only: no loads or stores in the
  timed loop. 2³⁰ vector FMAs per measurement, 1 warmup + 5 reps, thread
  asked for a P-core via `QOS_CLASS_USER_INTERACTIVE`.
- Verified in the generated assembly: the timed loop for C chains contains
  exactly C `fmla.4s`, zero register copies, and zero vector loads/stores
  for C ≤ 28 (30 and 32 spill a few registers).

## Results

| chains | 1 | 2 | 4 | 6 | 8 | 12 | **16** | **20** | 24 | 28 | 30 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GFLOPS | 8.8 | 17.6 | 35.0 | 52.6 | 70.6 | 103.5 | **135.2** | **134.2** | 109.8 | 127.8 | 107.0 | 106.0 |

- **Perfectly linear from 1 to 16 chains** (8.8 per chain), then flat. A
  single chain is latency-bound: each FMA waits for the previous result.
- **16 FMAs in flight saturate the core.** That number = FMA latency ×
  number of FMA pipes. 1 chain = 1.1 G vector-FMA/s → one FMA every
  ~0.91 ns. Common published figures for Apple P-cores (4 pipes, 4-cycle
  latency) fit both numbers (4 × 4 = 16) at an implied ~4.2 GHz, but pipe
  count and clock are not measured separately here.
- **This is why GEMM needs many accumulators.** Stages 1–2 have one
  dependency chain per output element in the inner loop; Stage 3's 24
  accumulators keep ≥ 16 FMAs in flight.
- 30–32 chains drop because of spills (seen in the assembly).
- **24 and 28 chains dip without spills** (110, 128). Unexplained; the
  `simd` kernel itself runs 24 accumulators at ~115, so it's a property of
  this loop (alignment, front end), not a hardware limit at 24. Not chased.

## A bug worth remembering

The first version wrote `acc = vfmaq_f32(y, acc, x)`, i.e. `y + acc·x`.
ARM's `fmla` adds **into its first operand**, so the compiler had to copy
`y` into a fresh register before every FMA. That doubled the instruction
count, ran out of registers at 20 chains, and spilled to memory: the
benchmark measured **112 GFLOPS peak and a 2× collapse at 20+ chains**,
both wrong. Swapping to `acc = vfmaq_f32(acc, x, y)` fixed it. The
micro-kernel already had the right form (`vfmaq_laneq_f32(c, b, a, lane)`).
Lesson: a microbenchmark is only as good as its assembly; check it.

## Updated Stage 3 standing

| | GFLOPS | % of measured single-core peak |
|---|---|---|
| peak (this benchmark) | ~135 | 100% |
| simd (Stage 3, n = 1024) | 115.3 | **~85%** |
| packed (Stage 2b) | 41.9 | ~31% |
| ikj (Stage 1) | 32.4 | ~24% |
| naive | 2.44 | ~2% |
| Accelerate (AMX) | ~2100–3200 | not comparable: separate matrix hardware |
