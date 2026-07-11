# Track a-ws-pipeline: warp-specialized Transform→MMA pipeline (SS mode)

Findings log, workbook style. Branch `prototype/a-ws-pipeline` off
`sm100-tcgen05-v2`. Goal: take the dequant scatter off the MMA critical
path via a producer/consumer mbarrier pipeline between transform warps
and an MMA-issuing warp, keeping SS mode (r2s scatter + smem.b_dequant).

## M1: SS-mode baseline on THIS GPU (B300 SXM6 GPU0, cc 10.3, 2026-07-11)

`benchmarks/bench_ws_pipeline_baseline.py` — mma.sync (heuristic wmma
config) vs tcgen05 prod config (BlockM=128@M>=128 / 64@M<128, BlockN=128,
BlockK=128, stages=4, WS+TMA), bf16 x uint4 gs=128 zp=True:

```
shape                M  mma.sync us  tcg-prod us    ratio
Llama70B gate       16       165.98       230.05    0.72x
Llama70B gate      128       329.82       231.70    1.42x
Llama70B gate      512      1061.78       795.39    1.33x
Llama70B gate     2048      3998.38      2834.37    1.41x
Llama70B down       16       280.57       393.47    0.71x
Llama70B down      128       281.13       397.79    0.71x
Llama70B down      512      1133.32       796.45    1.42x
Llama70B down     2048      3883.65      2779.80    1.40x
```

Matches workbook B.35/B.36 (2787us at down M=2048 then, 2780 now).
Correctness baseline: `test_tcgen05.py + test_tcgen05_dtypes.py` =
67 passed / 24 skipped / 1 xfailed.

## M2: design — WS Transform→MMA pipeline

### What we replace

Today every math warp per K-iter (16 K-bf16): dequant (4 i-calls,
4-way redundant across M-warps) → scatter 2 KB to
`smem.b_dequant[iter%2]` → `bar.sync 1,256` → warp-0 lane issues
`tcgen05.mma`. The barrier makes the MMA issue wait for the SLOWEST
warp's scatter every iter; warps can never run ahead. Swordfish
evidence (swordfish-notes.md §0, §7.1): hiding the scatter behind a
producer/consumer pipeline is the structural fix, TS-mode not required.

### Pipeline granularity: one full G2S stage (BlockK), not one K-iter

`smem.b_dequant[2]` slots already hold a full BlockN x BlockK tile
(32 KB at 128x128); today they ping-pong per K-iter with half-wasted
slots. New: slot = k_block % 2 (one slot per G2S stage), transform
warps fill ALL kWarpIters (=8) 16-K chunks of a k-block into one slot,
MMA warp drains 8 tcgen05.mma issues per slot. This amortizes the
handshake 8x and exactly matches Swordfish's kT2MStages=2 32 KB
compute-buffer ping-pong. Zero SMEM growth.

### Warp roles (cooperative — all math warps transform, warp 0 also issues)

kMathWarps = 8 at prod config. Warp w keeps its existing loader_b /
loader_bs n-slice (n = (w % kNWarps) * WarpN) and takes the i-call
subset i ≡ (w / kNWarps) mod min(kMathWarps/kNWarps, kCalls) of the
kCalls = WarpN/16 dequant calls. At prod config: 8 warps x 1 i-call
each = 8x less scatter per warp than today (2 KB → 512 B per warp per
K-iter), zero redundancy. (B.25's negative result gated warps *inside
the bar.sync structure*, where the idle warps still waited per-iter;
here there is no per-iter barrier at all, so the tradeoff changes.)
Warp 0 additionally issues the MMAs; dedicated-idle-MMA-warp variants
can be tried later.

### mbarriers (new, in SharedStorage under IF_USE_TCGEN05)

* `t2m_full[2]`  — expected count = kMathWarps (lane-0 arrive per warp
  after scatter + `fence.proxy.async.shared::cta`). Waiter: warp 0
  before issuing the slot's 8 MMAs. Replaces the per-iter bar.sync.
* `t2m_empty[2]` — expected count = 1, arrived by
  `tcgen05.commit.mbarrier::arrive` issued by warp 0 right after the
  slot's 8 MMAs (the PipelineUmmaConsumerAsync consumer_release
  pattern). Waiters: transform warps before REWRITING the slot
  (k_block T+2 waits the commit from T); warp 0 instead waits it
  immediately after its own commit (see below), so each warp waits
  exactly once per slot use and phase parity stays consistent.

### G2S stage release (math_mbar, count = kMathWarps unchanged)

* Transform warps arrive after their last s2r read of the stage
  (post-transform-loop). The wait for stage+1 is placed before the
  final s2r prefetch (which reads next stage's codes) — same ordering
  contract as today's kWarpIters-2 placement.
* Warp 0 arrives only after `mbarrier_wait(t2m_empty[slot])` — i.e.,
  after its 8 MMAs RETIRED, so the producer can never overwrite
  smem.a under an in-flight MMA. This is strictly safer than today's
  arrive-at-kWarpIters-2 (which releases with 2 MMAs not yet issued);
  the retire-wait overlaps the other warps' transform of the next
  slot, so it should be off the critical path (transform ~2k cyc >
  8-issue MMA chain exec).

### Per-tile flow (all math warps; roles branch inside)

```
zero_accum; seek; wait_stage<first>; prime s2r(0,0)
while slice_iters: for stage_id in 0..kNumStages-1:
  slot = slot_ctr & 1
  if warp != 0 and slot_ctr >= 2: wait t2m_empty[slot] (phase^)
  for it in 0..kWarpIters-1:
    if it == kWarpIters-1 and slice_iters > 1: wait_stage(stage+1)
    s2r load (stage, it+1)              # regs_qb/scales double-buffer
    dequant my i-subset (buf = it % 2)
    scatter my i-subset -> b_dequant[slot] @ k_base = it*16
  fence.proxy.async.shared::cta; lane0: arrive t2m_full[slot]
  if warp == 0:
    wait t2m_full[slot] (phase^)
    for it in 0..kWarpIters-1: issue tcgen05.mma(stage, slot, it)
    elect1: tcgen05.commit -> t2m_empty[slot]
    wait t2m_empty[slot] (phase^)
  lane0: arrive math_mbar[stage]; slot_ctr++
epilogue: unchanged (final commit -> tcgen05_mbar, all-warp t2r)
```

`slot_ctr`, phase parities persist across tiles; the tile-end drain
(commit to tcgen05_mbar) is ordered after all per-slot commits, so no
cross-tile hazards. Pre-dealloc `ctx.sync_math_threads()` + cluster
barrier pairing preserved.

### Staging depth

2 slots (existing) to start. 3+ slots would need +32 KB and we are at
~224/232 KiB — not available at BlockK=128 stages=4. If the empty-wait
shows up as a stall, trade G2S stages for T2M slots later (measure).

### Gating

New `TuningConfig.use_ws_pipeline` (default False) → `kUseWsPipeline`;
new mainloop branch in humming_ws.cuh under
`kMmaType == TCGEN05 && kUseWsPipeline`. Old SS path untouched so the
existing test matrix keeps passing unchanged.

## M3: implementation findings (2026-07-11)

* First working version passed the prod config immediately (M=128
  N=256 K=512) but two design iterations were forced by measurement:

1. **Retire-wait on the MMA warp serializes the whole pipeline**
   (measured 0.31-0.62x vs mma.sync, i.e. ~2.3x SLOWER than the
   classic tcgen05 path). Warp 0 waiting its own commit (MMA
   retirement) before releasing the G2S stage puts the full MMA
   execution inside warp 0's per-stage loop; since warp 0 also
   transforms, the next block's t2m_full then waits on it -> transform
   and MMA exec alternate instead of overlapping. Removed.

2. **Releasing the G2S stage at commit-ISSUE races the producer at
   large K** (max|err|=4.0 of ref 536 at M=512 N=1024 K=4096; small
   shapes pass). Unlike the classic path -- where the per-iter
   bar.sync keeps the MMA queue empty so "issued" ~= "retired" -- the
   pipeline can hold 8-16 MMAs in flight, so the producer's refill of
   buffer T-2 (triggered by arrivals for stage T-1) can TMA over
   smem.a still being read. Fix with zero added waits: warp 0 DEFERS
   its math_mbar arrival for block T-1 until after its t2m_full wait
   of block T. At that point transforms(T) completed, which required
   empty[slot(T)] = commit(T-2) COMPLETION, and T-2's buffer is
   exactly the one the producer refills next -> provably retired.
   Costs nothing (the arrival is just delayed, producer keeps
   kNumStages-2 of slack). Deadlocks at kNumStages=2 (static_assert +
   Python assert; prod configs are stages>=3).

* Also: the next-stage G2S wait must come AFTER the current stage's
  arrivals (classic interleaves them at kWarpIters-2); the cross-stage
  s2r prefetch moved after the arrive/wait pair accordingly.

## M4: correctness evidence

* **WS-pipeline output is BIT-EXACT vs the classic tcgen05 path**
  (M=512 N=1024 K=4096, prod config, torch.equal) -- the pipeline
  reorders scheduling, not math (same tcgen05.mma sequence over the
  same dequantised values). This is the sharpest correctness criterion
  and is now pinned by `test_tcgen05_ws_pipeline_bitexact_vs_classic`.
* Important tolerance gotcha: at K=4096 BOTH paths show ~0.7% of
  cells beyond atol=0.5 vs the fp32-reference GEMM (max|err|=2.0,
  identical cells) -- pure bf16 accumulation drift, matching the
  dtype-test suite's atol=2.0 at that K. Do NOT chase atol=0.5
  "failures" at K>=4096.
* Infra gotcha: the shared ~/.humming/tmp/lock/launcher.lock is
  contended across concurrent tracks (a track-E process held it for
  8+ min -> our runs appeared hung with GPU util 0%, main thread in
  nanosleep). Fix: export HUMMING_TMP_DIR=$PWD/.humming-tmp (kernel
  cache stays shared via the default HUMMING_CACHE_DIR).

## M5: perf debugging

* v2 (retire-wait removed) still 0.62x vs mma.sync (6240us at gate
  M=2048 vs classic tcgen05's 2834us). NCU at down M=2048:
  Compute SM 16.5% (classic 57%), warp cyc/inst 16.4 (classic 5.05),
  81.8% no-eligible, long_scoreboard 12.95 cyc/inst (classic 1.52),
  wait 1.65, barrier 0.00 (bar.sync is really gone).
* **Root cause: register-array spills from the runtime i_first.**
  l1tex local traffic: 471M ld + 780M st sectors (classic ~0). The
  WS i-call subset made `i` runtime, so regs_b_tmp[i*4], regs_qb
  selection inside dequant<>, and arith regs_zp/regs_bs indexing all
  became runtime register-array indexes -> ptxas demoted them to
  local memory. Classic path's fully-unrolled `i` never hits this.
* Fix: kIFirst as a template parameter + recursive warp-uniform
  dispatch (i_first in [0, kIStepWs), depth <= 4).

### Ablation bisection at down M=2048 (prod config, all bit-times on GPU0)

```
loads-only (s2r + skeleton + full-mbar + G2S flow):   1033 us
+ dequant(1 i-call) + scatter(4 stores) per iter:     3531 us  (+2498!)
+ MMA issue/commit + empty waits (full v3):           4887 us  (+1356)
classic tcgen05 (does 4 i-calls + 16 stores + bar):   2780 us
```

One i-call in the WS path cost MORE than classic's four -> the
per-K-iter i_first dispatch put 4 guarded copies of dequant+scatter
inside the (nominally unrolled) iter loop; the bloat breaks unrolling
so iter%2 buffer ids go runtime -> regs_qb/regs_b_tmp local spills
return (residual 8.3M local sectors even after templating kIFirst).
Fix: hoist the dispatch to once per k-block with the whole iter loop
inside the instantiation (transform_kblock_ws).

### Perf fix ladder (down M=2048, prod config, bit-exact at every step)

```
v1 retire-wait + per-iter dispatch (runtime i):     6341 us   0.44x classic
v2 no retire-wait (still runtime i_first):          6269 us
v3 kIFirst templated (per-iter dispatch):           4887 us
v4 dispatch hoisted to per-k-block:                 4058 us
v5 ws stage loop unroll(1):                         2726 us   1.02x classic  <- crossover
classic tcgen05 SS (bar.sync per iter):             2780 us
```

Stall progression (cyc/inst): long_scoreboard 12.95 -> 4.36,
no_instruction (I-fetch) surfaced at 6.18 after v4's bloat, killed by
v5. Code size is a first-class constraint in this kernel: 4 dispatch
instantiations x unrolled stages was enough to starve instruction
fetch.

```
v6 bitmask mbar phases (arrays spilled at 185M
   local-ld sectors once the stage loop stopped
   unrolling):                                       2389 us   1.16x classic
```

### Remaining levers (not yet tried, ordered by expected value)

1. Stage-front batched code loads (Swordfish "regs first" pattern):
   the loads-only skeleton was 1033us, so s2r latency is no longer
   the wall, but batching the 8x4B per-warp code words would shorten
   the transform critical path further (uint4 fast path: i-call i
   reads exactly regs_qb word i).
2. Early consumer release for transform warps (arrive math_mbar at
   the last current-stage s2r read instead of post-loop).
3. 3 T2M slots at BlockK=64 configs (SMEM allows) to decouple the
   empty-wait chain.
4. Dedicated MMA warp (warp 0 not transforming) -- likely a wash now
   that transform is cheap, but untested.

## M6: three-way bench (deliverable table, GPU0, merged tree 27a4146)

`bench_ws_pipeline_baseline.py --three-way` -- mma.sync heuristic vs
classic tcgen05 SS (BlockM=128@M>=128/64 else, BN128, BK128, s4, WS)
vs the same geometry + use_ws_pipeline:

```
shape                M   mma.sync   tcg-clas    tcg-wsp  wsp/clas   wsp/mma
Llama70B gate       16     166.13     229.89     200.96     1.14x     0.83x
Llama70B gate      128     329.63     231.83     203.45     1.14x     1.62x
Llama70B gate      512    1062.19     795.14     696.09     1.14x     1.53x
Llama70B gate     2048    3996.90    2833.27    2478.20     1.14x     1.61x
Llama70B down       16     280.66     393.29     344.68     1.14x     0.81x
Llama70B down      128     280.85     397.92     349.18     1.14x     0.80x
Llama70B down      512    1133.38     796.38     689.93     1.15x     1.64x
Llama70B down     2048    3883.89    2779.90    2387.37     1.16x     1.63x
```

Uniform 1.14-1.16x over classic SS at every M on both shapes (the
pipeline removes a per-iter cost, so the ratio is M-independent).
mma.sync still wins at M=16 both shapes and down M=128.

## M7: synccheck finding (tool artifact, pre-existing)

compute-sanitizer synccheck on the ws-pipeline kernel reports "Barrier
error: divergent thread(s) in block" + the kernel then aborts
(unspecified launch failure). CONTROL: the SHIPPED classic warp-spec SS
path (test_tcgen05_warp_spec, untouched code) fails identically
(13696 errors). Cause: the partial-block named barrier
`bar.sync 1, kNumMathThreads` (producer warps legitimately absent) trips
synccheck's divergence instrumentation -- pre-existing for the whole
warp-spec tcgen05 family, NOT a pipeline regression. Track B's clean
synccheck run was over their TS suite whose barrier is also named but
apparently tolerated at their config; our correctness bar stays the
suite + the bit-exact-vs-classic test.

## M8: COMPOSITION (headline question) -- NEGATIVE RESULT, quantified

Merged prototype/b-ts-staging (merge 27a4146; combined suite 94 pass)
and implemented the composed kernel (use_ws_pipeline + use_tcgen05_ts,
commit c0d21f8): track B's TS core (dequant -> tcgen05.st -> swapped
MMA + per-slot WAR commit/mbar) with the per-K-iter 128-thread
bar.sync on the issue path replaced by the track-A t2m_full mbarrier
handshake (only warp 0 waits + fences + issues) and the deferred G2S
release (stage T-1 released at (T, iter1), where transform_b(iter1)'s
WAR wait transitively proves commit(T-1, last) completed). CORRECT:
bit-exact vs plain TS at M=512 N=1024 K=8192; 97-test suite green.

`benchmarks/bench_compose.py`, all five kernels at IDENTICAL geometry
(BlockM=128@M>=128 else 64, BN=128, BK=64, s4, WS+TMA), GPU0:

```
shape             M       mma        ss    ss-wsp        ts    ts-wsp  ts/tswsp
Llama70B-gate    16     166.2     246.4     244.8     165.6     215.7     0.77x
Llama70B-gate   128     309.9     246.8     249.5     203.5     256.8     0.79x
Llama70B-gate   512    1070.5     851.2     856.8     698.2     883.5     0.79x
Llama70B-gate  2048    3823.8    3028.7    3063.4    2489.0    3152.7     0.79x
Llama70B-down    16     280.9     421.8     420.3     281.3     371.2     0.76x
Llama70B-down   128     532.3     426.8     424.3     347.1     441.0     0.79x
Llama70B-down   512    1060.6     851.5     855.3     689.8     877.3     0.79x
Llama70B-down  2048    3701.2    2965.4    2980.5    2399.1    3057.5     0.78x
```

(ts-wsp shown with the stage loop fully unrolled -- the PRAGMA_
UNROLL_COUNT(1) first cut was 0.71x; the SS path's I-fetch-starvation
rationale does not transfer to the tiny TS transform.)

VERDICT: the composition LOSES to plain TS by 0.76-0.79x everywhere;
B's TS stays the fastest humming config. Two structural reasons:

1. There is nothing left to hide. The TS transform (8 lop3/hsub/hmul
   + one tcgen05.st) is ~free next to the SS scatter, so the per-iter
   bar.sync among 4 symmetric warps costs almost nothing -- replacing
   it with 4 mbar arrivals + a warp-0 mbar spin-wait + fence per iter
   ADDS latency to the MMA-issue path instead of removing any.
   Corroborated in-table: at this BK=64 geometry even ss-wsp is a
   WASH vs ss (2980 vs 2965 at down-2048) -- the pipeline's 1.14-1.16x
   win is specific to BK=128 (8 iters/stage, 2 KB/warp/iter scatter);
   at BK=64 nothing is barrier-bound.
2. Run-ahead is capped at 2 K-iters by the 2 TMEM staging slots
   (8 cols each), so the pipeline cannot buy the multi-stage
   decoupling that made it win on SS. Deeper TMEM staging (112 free
   cols at BlockM=128 -> up to 14 slots) would lift the cap, but with
   the transform ~free there is nothing to decouple.

B's "batch commit every 2 iters" lever was NOT tried on the composed
kernel (task gates it on composition winning). It remains plausible
headroom for the PLAIN TS path where the commit is also per-iter.

The composed path stays in-tree (correct, tested, one if-constexpr
branch) as the recorded experiment; the config default keeps it off.

### Track-A levers that DO transfer to TS (follow-ups for next owner)

* Deferred G2S release: B's own notes flag the consumer.arrive at
  kWarpIters-2 racing the stage's last TS MMAs (same latent hazard the
  SS pipeline hit as REAL wrong cells at K=4096). The composed branch
  carries the fix; porting just the deferral into the plain TS
  mainloop is cheap insurance.
* TS integration with track D's production packer is still pending
  (tests/test_ts_packing_cross_track.py pins the layout divergence).
