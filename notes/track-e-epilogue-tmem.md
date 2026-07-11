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
  noisier than Milestone 1's. (Verified idle-resident during the
  Milestone-3 measurements below: full 2032 MHz clocks, my kernel the
  only one executing.)

## Milestone 3: 4x acc1 regression found + fixed (2026-07-11)

Predecessor's plumbing regressed the SHIPPED path (acc_stages=1) 4x:
Llama70B-down M=2048 went 2781 -> 11210 us. NOT environment: pristine
0954db9 in a /tmp worktree measured 2780.6 us in the same session.

Root cause: `uint32_t mbar_phase_[kAccStages]` indexed by the runtime
member `acc_buf_`. Dynamic indexing of a member array demotes the whole
TCGEN05 object (including its hot state) to a local-memory stack frame:
`cuobjdump -res-usage` shows STACK 16 -> 1104 B between the fast and
slow cubins, REG unchanged at 168. Fix (commit 8c71b67): scalar
`mbar_phase_bits_` bitfield + `accum_buf()`/`accum_col_off()` accessors
that fold to compile-time 0 at kAccStages==1.

After fix, same GPU, same session (Llama70B down M=2048):

```
a1s4 (shipped)        2778.6 us   (Milestone-1 baseline: 2777.9)
a1s3                  2844.4 us   (s3 costs +2.4% -- acc2 needs the
                                   32KB dedicated reduce, s4 does not
                                   fit: cuFuncSetAttribute fails)
a2s3 NO_DEFER         2892.4 us   (+1.7% rotation plumbing overhead)
```

LESSON for any future TMEM multi-staging work: never dynamically index
member arrays in the MMA/pipeline objects; nvcc demotes the aggregate.

## Milestone 4: deferred drain debugging -- missing tcgen05.wait::ld

Enabling the deferred drain (NO_DEFER off) corrupted output with a
crisp pattern: at M=2048 4096x4096 exactly ~148 bad tiles = each CTA's
SECOND-TO-LAST tile (strided persistent scheduling); single-tile-CTA
shapes (M=512: 128 tiles < 148 CTAs) were RANDOMLY ~70-90% bad,
run-to-run varying. Bad tiles held mostly-right data with ~half the
elements per row wrong and a time-gradient across rows.

Bisection matrix (M=512 / M=1024, 3 runs each):
* NO_DEFER (drain immediately, new plumbing): clean.
* Test A (defer by one tile, drain BEFORE consumer.arrive -- zero
  producer overlap): BAD (worse: 119/128).
* Test C (pending/snapshot/set_accum_buf machinery, drain in the SAME
  iteration): clean -- machinery correct, so the corruption depended
  only on WHERE in the instruction stream the drain ran.

ROOT CAUSE: the t2r drain reads `tmp` immediately after
`tcgen05_ld_32x32b_x32`, and NO `tcgen05.wait::ld` exists anywhere in
humming. tcgen05.ld is ASYNC -- destination registers are undefined
until the wait. The SHIPPED SS epilogue has the same latent UB and
passes tests through SASS-scheduling luck; the deferred builds shuffle
the schedule and expose it. (The predecessor's tmem_ld_bench.py itself
uses tcgen05.wait::ld -- the pattern was known.)

Fix: `tcgen05_wait_ld()` after each t2r in drain_accum.

CROSS-TRACK: track B's TS kernel drains TMEM through the same
tcgen05.ld path -- if it doesn't wait::ld it has the same latent bug
masked by luck. Flag when merging b-ts-staging.

After the wait::ld fix the corruption became deterministic and moved:
single-tile/loop-exit drains CLEAN, but each CTA's second-to-last tile
(the drain issued in the LAST loop iteration, i.e. the one whose
smem.reduce is overwritten by the loop-exit drain right behind it)
stayed bad, with ZERO-filled bands. Two more latent bugs:

* LATENT BUG 2: no `fence.proxy.async.shared::cta` anywhere on the
  TMA-C store path -- smem.reduce is written through the generic proxy
  and read by cp.async.bulk.tensor through the async proxy. Fixed in
  gmem_writer.write_tma.
* LATENT BUG 3: `tma_commit_store_group()` had ZERO call sites, so
  EVERY `tma_wait_store_group` in the codebase waits on zero committed
  groups = no-op. The shipped flow survives because a full mainloop
  separates consecutive smem.reduce writers; back-to-back deferred
  drains do not. Fixed: commit after store issue in write_tma + a
  math barrier after the deferred-path waits (the wait is per-thread;
  without the barrier the OTHER math warps can overwrite smem.reduce
  early). This also silently un-no-ops the stream-k TmaC waits.

With all three fixes the deferred drain is CLEAN: 0 bad tiles x 3 runs
at M in {128, 512, 1024, 2048, 2112} on 4096x4096 (+128x128x4096).

## Milestone 5: Part-1 verdict on the SS kernel -- honest negative

Perf with all fixes (GPU 4, 50 iters, DiffusionWorker idle-resident):

```
shape              a1s4        a1s3        a2s3(deferred)
L70B-down M=2048   2773.7 us   2845.3 us   2883.4 us
L70B-down M=4096   5543.5 us   5680.4 us   5756.9 us
4096^2    M=2048    235.5 us    242.1 us    244.1 us
4096^2    M=256      61.9 us     63.8 us     64.0 us
```

* Shipped a1s4 is UNCHANGED by the three correctness fixes (2773.7 vs
  2777.9 baseline) -- they are free.
* Deferred rotation LOSES everywhere on the SS kernel: +1.3% vs a1s3
  at identical stages, and it cannot use s4 (dedicated reduce +32KB
  exceeds max SMEM), so its real deficit vs shipped is ~4%.
* Why no win: the deferred drain still runs INLINE in the math
  threads; it only overlaps PRODUCER loads, and at these shapes the
  producer is not the bottleneck (mainloop-dominated; s4->s3 alone
  costs 2.4%). Epilogue-exposed M=256 is single-wave (1 tile/CTA), so
  rotation has nothing to overlap by construction.
* REMAINING HOPE for rotation: the TS kernel (BK64 s4, ~81KB SMEM),
  where +32KB dedicated reduce still fits (~113KB < 116KB 2-CTA
  bound) and the epilogue is the top headroom (track B's notes).
