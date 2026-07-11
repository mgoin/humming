# Track E: epilogue-side wins (TMEM accumulator multi-staging + TS drain path)

Findings log, workbook style. GPU 4 of 8x B300 SXM6 (cc 10.3, sm_103a),
CUDA 13.0, torch 2.11.0. Branch `prototype/e-epilogue-tmem`.

## Milestone 1: SS-mode baseline (same-GPU, 2026-07-11)

`benchmarks/bench_track_e_baseline.py`. tcgen05 = production WS config
(BM=128 BN=128 BK=128 stages=4, WS+TMA, no stream-k); mma.sync = BM=64
BN=128 BK=64 s3 non-WS reference (same as bench_tcgen05_vs_wmma's "mma").
uint4 + zp, gs=128, bf16 A/scales. 50 iters after 10 warmup.

```
shape                M  mma.sync us    tcg-SS us  tcg/mma
Llama70B gate       16        166.1        232.2    0.72x   (tcg M padded to 128)
Llama70B gate      128        329.9        232.2    1.42x
Llama70B gate      512       1062.3        791.2    1.34x
Llama70B gate     2048       3994.2       2829.3    1.41x
Llama70B gate     4096       7904.8       5555.2    1.42x

Llama70B down       16        280.7        397.6    0.71x   (tcg M padded to 128)
Llama70B down      128        281.0        397.7    0.71x
Llama70B down      512       1117.1        797.5    1.40x
Llama70B down     2048       3866.1       2777.9    1.39x
Llama70B down     4096       7724.7       5549.3    1.39x

4096x4096          128         43.3         61.8    0.70x
4096x4096          256         43.2         61.8    0.70x
4096x4096          512         82.2         61.8    1.33x
4096x4096         2048        276.8        234.4    1.18x

Llama8B qkv        128         43.4         61.8    0.70x
Llama8B qkv        256         84.2         61.8    1.36x
Llama8B qkv       2048        445.9        346.5    1.29x
```

Numbers match workbook B.35/B.36 (Llama70B-down M=2048 was 2787 us there,
2778 us here -- same GPU class, within noise). Test suites green at HEAD:
`test_tcgen05.py` 44 passed / 1 xfail, `test_tcgen05_dtypes.py` 23 passed
/ 24 skipped.

Tile-count arithmetic for Part-1 shape selection (BM=BN=128, 148-160 SMs):

* Llama70B down M=2048: 16 x 64 = 1024 tiles => ~6-7 tiles/CTA. Epilogue
  overlap opportunity exists but K=28672 (224 K-blocks) makes the mainloop
  dominate.
* 4096x4096 M=2048: 16 x 32 = 512 tiles, K-blocks = 32 => epilogue-exposed
  AND multi-tile. Best shape to see Part-1 movement.
* 4096x4096 M=256 (one wave, 64 tiles < #SMs): 1 tile/CTA -- per-tile
  accumulator rotation can win NOTHING here (no tile i+1 in the same CTA).
  Included in the baseline to demonstrate that limit honestly; the
  "epilogue-exposed regime" that Part-1 can help is small-K multi-wave,
  not single-wave.

## Milestone 2: Part-1 archaeology (session 2, 2026-07-11)

Predecessor died mid-flight with uncommitted edits. Triage decisions:

* config.py / storage.cuh / humming.cuh / tcgen05_mma.cuh / humming_ws.cuh
  diff = coherent Part-1 implementation (per-tile TMEM accumulator
  rotation, `tcgen05_acc_stages` config, dedicated `smem.reduce` when
  staged, per-buffer commit mbarriers). KEPT and committed.
* humming_ws.cuh carries a hard-coded `#define TCGEN05_ACC2_NO_DEFER 1`
  bisection switch: shipped drain ordering but all the new
  storage/alloc/rotation plumbing. KEPT for now (see hang below), to be
  flipped to the deferred path once NO_DEFER is proven correct.
* Found predecessor's `/tmp/test_acc2_order.py` HUNG for 5h on GPU 4
  (M=128 N=128 K=4096, acc_stages 2 vs 1, s3). Killed it. This is
  presumably why the bisection flag exists: some earlier revision of the
  acc2 path hangs, likely in the loop-exit pending drain (M=128 N=128 is
  a single tile => the only drain IS the loop-exit one) -- or in
  `tcgen05_alloc<256>`. Unknown which .cuh revision that process had
  compiled; needs re-testing against the current tree.
* benchmarks/exp_b33_alt.py (untracked): docstring referenced a
  `TCGEN05_EXP_ALT_MODE` macro that no longer exists in the diff.
  REWRITTEN to drive `tcgen05_acc_stages` {1,2} through HummingKernel,
  with correctness checks (incl. M=2112 tail for the odd-tile pending
  drain) + both regimes' perf shapes.
* Hazard audit of the deferred path (for when NO_DEFER is dropped):
  `load_channel` copies channel input scale / channel weight scale /
  bias into epilogue regs PER TILE, so a deferred drain of tile i would
  use tile i+1's values. Bias is already static_asserted out; channel
  scales and indexed/grouped gemm (smem rd/wr_row_index refilled by the
  producer) still need guards. To add: static_asserts on
  `Ctx::kIsDenseGemm`, `!kIsChannelWeightScale`,
  `!(kHasInputScale && kInputScaleGroupSize == 0)` for kAccStages > 1.
* GPU 4 is NOT exclusive right now: root runs a 128 GB
  `vLLM-Omni::DiffusionWorker-0` on it. Perf numbers this session may be
  noisier than Milestone 1's.
