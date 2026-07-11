# Track C: cta_group::2 (2x1SM) on the SS-mode tcgen05 kernel

Findings log, workbook style. GPU 2 of 8x B300 SXM6 (cc 10.3, sm_103a,
CUDA 13.0, torch 2.11.0+cu130). Branch `prototype/c-2sm-ss`.

## Goal

Cluster-pair MMA on the EXISTING SS-mode kernel: leader CTA issues one
`tcgen05.mma.cta_group::2` covering both CTAs' halves. Swordfish rules
adopted verbatim:
(a) each CTA dequants/stages ONLY its own half of the weight tile
    (disjoint plain TMA, no multicast on weights);
(b) activations use the 2SM multicast TMA load — one arrival on the
    leader's mbarrier covers both CTA halves.

## Milestone 1: SS-mode cg1 baseline (this GPU)

`benchmarks/bench_cg2_ss.py`, production config BM=128 BN=128 BK=128
stages=4 WS+TMA vs mma.sync reference (BM=64 BK=64 s=4 cp.async).
50 iters after 10 warmup, times in µs.

```
shape            M      mma.sync   tcg-ss cg1
Llama70B gate    16        160.1        232.0
Llama70B gate    128       317.7        232.0
Llama70B gate    512      1020.9        788.7
Llama70B gate    2048     3854.8       2836.1
Llama70B gate    4096     7626.8       5567.4

Llama70B down    16        266.8        398.0
Llama70B down    128       267.2        398.5
Llama70B down    512      1057.5        798.7
Llama70B down    2048     3690.6       2782.0
Llama70B down    4096     7376.1       5557.6
```

Consistent with workbook B.35 (Llama70B-down M=2048: 2787 µs there,
2782 here). tests/test_tcgen05.py: 44 passed / 1 xfailed on this GPU.

Every perf claim below is vs THIS table.

## Milestone 2: standalone cg2 issue mechanics validated (POC)

`benchmarks/bench_tcgen05_cg2_poc.py` (Jinzhen's POC, copied verbatim —
same-project code). Raw issue-rate on this GPU (B300, sm_103a):

```
cg1 m128 n256 k16 (p=1, RAW chain):   2321.3 TFLOPS
cg2 m256 n256 k16 (p=0, no chain):    2321.4 TFLOPS
cg1 m128 n128 k16:                    2318.5 TFLOPS
cg2 m256 n128 k16:                    2318.0 TFLOPS
```

Facts learned:
* cg2 issue mechanics (cluster launch via CU_LAUNCH_ATTRIBUTE_CLUSTER_
  DIMENSION, cta_group::2 alloc/mma/commit, leader-only issue, idesc.M
  = total-M across the pair) work as-is on this machine/toolchain.
* BOTH cg1 and cg2 saturate ~2320 TF at these shapes. The raw MMA issue
  rate is NOT the limiter for our kernel in either mode — one CTA/SM
  issuing M128N128K16 back-to-back already hits peak.
* Implication: the cg2 win in the real SS kernel must come from the
  DATAFLOW, not issue rate: with total-M=2*BlockM and B N-split across
  the pair, each CTA dequants/scatters only HALF the weight tile
  (the scatter is the measured bottleneck, workbook B.34/B.36), and
  each weight tile is dequanted once per PAIR of M-tiles instead of
  once per M-tile.

## Milestone 3: design — cg2 on the SS-mode WS kernel

Verified empirically before design: tcgen05 WS + `multi_cast_size_b=2`
(cluster of 2, multicast weight codes, per-CTA cg1 MMAs) is CORRECT on
this tree (m512 n512 k1024 BM128 BN128 BK128 s4: identical error to
mc_b=1, allclose passes). So the cluster launch, scheduler pairing,
multicast-B TMA, cross-CTA math_mbar arrive, and the end-of-kernel
cluster barrier all work under tcgen05 today — B.29's hang is gone on
this base. cg2 builds on that substrate.

### Operand geometry (from CUTLASS SM100_MMA_F16BF16_2x1SM_SS traits)

For MMA shape (M_tot, N, K) over a 2-CTA cluster:
* ALayout: CTA r supplies A rows [r*M_tot/2, (r+1)*M_tot/2) from ITS
  OWN SMEM at the (single, leader-broadcast) a_desc address.
* BLayout: CTA r supplies B cols [r*N/2, (r+1)*N/2) from its own SMEM
  at the b_desc address (fragment starts at row 0 of the descriptor).
* CLayout: CTA r's TMEM holds (M_tot/2 rows, FULL N cols).

Mapped to humming (A=activations M-split, B=weights N-split):
* M_tot = 2*BlockM; each CTA keeps loading its OWN activation M-tile
  (plain per-CTA TMA, unchanged loader_a) — the scheduler's mc_b
  pairing (m_block_id = 2*m + cluster_rank, same n_block_id) is
  EXACTLY the tiling cg2 needs.
* Weight tile (BlockN rows) is SHARED by the pair; CTA rank r dequants
  and scatters ONLY rows [r*BlockN/2, ...+BlockN/2), COMPACTED to row 0
  of a half-size b_dequant buffer ((BlockN/2) x BlockK per slot).
  Weight codes still arrive via multicast TMA (full tile to both CTAs;
  disjoint loads are a follow-up optimization).
* Per-CTA TMEM D = (BlockM=128 rows, BlockN=128 f32 cols) — identical
  to the cg1 M=128 accumulator; t2r + epilogue + gmem writer UNCHANGED,
  each CTA writes its own (m,n) output tile.
* Dequant/scatter work per CTA HALVES (each weight tile transformed
  once per PAIR of M-tiles) — this is the perf lever (scatter is the
  measured bottleneck), since raw MMA issue rate is already saturated
  (milestone 2).

### Sync design (Swordfish rules adapted)

* Per-K-iter pair rendezvous replaces cg1's local-only publish:
  after scatter + `fence.proxy.async.shared::cta` (per math thread) +
  `bar.sync` math threads, ONE elected math thread per CTA arrives on
  BOTH CTAs' `tcgen05_pair_mbar` (local + remote via
  mbarrier.arrive.shared::cluster on the peer-mapped address); ALL math
  threads then wait their LOCAL pair mbar (count=2, parity per iter).
  Leader-only elect-one then issues tcgen05.mma.cta_group::2 with
  idesc.M = 2*BlockM. This guarantees (a) both halves' scatter visible
  to the pair MMA, (b) both CTAs' A-stage data arrived (arrival is
  after each CTA's stage wait, program order), (c) CTAs stay in
  per-iter lockstep so the b_dequant slot / A-stage WAR margins match
  the empirically-safe cg1 timing.
* MMA drain for the epilogue: leader elect-one issues
  `tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster
  .multicast::cluster.b64 [tcgen05_mbar], 0x3` (CUTLASS
  umma_arrive_multicast_2x1SM) — ONE commit lands an arrival on BOTH
  CTAs' local mbar; each CTA waits locally and t2r's its own half.
  Avoids the unproven B.32 "both CTAs commit" workaround (a peer that
  issued no MMA may commit an empty batch = immediate arrival = race).
* Epilogue-vs-next-tile WAR: covered by the iter-0 rendezvous of the
  next tile (peer's arrival is after its epilogue in program order).
* alloc/dealloc/relinquish: cta_group::2 variants from BOTH CTAs (the
  POC recipe); one extra pair rendezvous before dealloc.
* PTX gotchas honored (workbook B.32): no {m0..m3} mask on the cg2 mma
  (the existing `tcgen05_mma_ss_bf16_2cta` wrapper HAS the mask — must
  be fixed); no mixing cg1/cg2 tcgen05 ops in one kernel; never
  try_wait a cluster-mapped mbar (all waits are local).

### Config plumbing

`TuningConfig.use_tcgen05_cg2` (auto-codegens kUseTcgen05Cg2 +
HUMMING_USE_TCGEN05_CG2). Python-side: forces multi_cast_size_b=2,
requires use_tcgen05 + use_warp_spec + BlockM=128 + BlockN=128 (v1).
Launcher/scheduler untouched (mc plumbing does it all).

### v1 restrictions

BlockM=128 (per-CTA TMEM layout = proven cg1 M=128 atom; BlockM=64
would need the 2SM M=128 half-lane TMEM layout re-derivation),
BlockN=128, warp-spec path only. BlockK any of {64,128,256}.

## Milestone 4: cg2 SS kernel landed (29d24bf)

See commit message: bit-identical to cg1 at production shapes, suites
green (67p/24s/1xf) + 6 cg2 tests. B.32 mask bug fixed in the cg2 mma
wrapper.

## Milestone 5: warp-redundancy redesign of the cg2 dequant

### Triage of the mid-flight dirty state (predecessor killed mid-edit)

Found dirty: tcgen05_mma.cuh + context.cuh. Hunk-by-hunk decisions:

* KEPT: `n_warp_id()` override in context.cuh (cg2 => every warp loads
  the CTA's own BlockN/2 slice, id = pair rank). Verified the tcgen05
  epilogue computes its own warp->TMEM mapping from raw warp_id
  (tcgen05_mma.cuh:637-638) and reads smem.bias directly, so the
  override only affects the B s2r loaders (loader_b/bs/bzp) as claimed.
  smem_writer + loader_bias are not on the tcgen05 path.
* KEPT: transform_b ungating (all warps dequant the CTA's own half).
  Rationale from the predecessor's measurement: gating half the warps
  off starves the SM of latency-hiding warps -- 1.5x slower end-to-end
  (long_scoreboard 1.54 -> 3.20, barrier 0.44 -> 1.71 per inst).
* FINISHED: the scatter path still called the deleted
  cg2_warp_owns_half() (would not compile). Completed the redesign:
  removed the scatter_active gate -- under cg2 ALL warps scatter the
  CTA's own half redundantly to the same addresses (n_base = 0
  compacted), 2x the cg1 path's benign store redundancy.
* DISCARDED: `#define TCGEN05_CG2_DEBUG_NO_RENDEZVOUS 1` and
  `#define TCGEN05_CG2_DEBUG_NO_FENCE 1` were left ENABLED -- these are
  timing-only experiments that break correctness (leader can issue
  before the peer's scatter lands). Restored to commented-out knobs.
  The `#ifdef` plumbing for them is kept (harmless, consistent with the
  file's other debug knobs).

NOTE the perf implication of this redesign: with full warp redundancy
each CTA performs the SAME total dequant+scatter instruction count as
cg1 (8 warps x half tile vs 8 warps' slices of the full tile with 4-way
M-warp store redundancy). The remaining cg2 levers are (a) multicast
weight-code loads (one L2/DRAM fetch serves the pair), (b) leader-only
MMA issue. Milestone 2 showed issue rate is saturated either way, so
expect modest gains at best; measurement below decides.

## Design note: composing cg2 with track B's TS-mode kernel (NOT implemented)

Track B (prototype/b-ts-staging, tcgen05_ts_mma.cuh) swaps A<->B:
MMA-A = weights from TMEM (kMmaM = BlockN = 128 rows), MMA-B =
activations from SMEM descriptor (MmaN = BlockM). Stacking cta_group::2
on it therefore mirrors the SS design across the A/B swap:

### Pairing flips from mc_b to mc_a

2x1SM M-splits the A operand across the pair: M_tot = 256 weight rows
= CTA r supplies weight rows [r*128, (r+1)*128) FROM ITS OWN TMEM.
So the CTA pair covers two ADJACENT N-TILES at the SAME M-TILE
(n_block_id = 2n + rank, same m_block_id) -- the multi_cast_size_A=2
pairing, opposite of SS-cg2's mc_b=2. Consequences:
* Weights: per-CTA TMEM staging halves are DISJOINT weight tiles --
  each CTA g2s-loads, dequants, and tcgen05.st's its OWN 128-row
  weight tile exactly as in cg1 TS. NO weight multicast, no
  half-tile compaction, loader_b/bs/bzp and the TS s2r contract
  UNTOUCHED. (Much cleaner than SS-cg2's half-tile dequant.)
* Activations: SHARED by the pair, but the 2x1SM BLayout N-splits the
  B fragment: CTA r supplies MMA-B cols [r*BlockM/2, +BlockM/2) = its
  half of the activation M-tile from its own SMEM at the (common)
  b_desc address. A plain mc_a=2 multicast (full duplicate tile in
  both CTAs) does NOT match: each CTA needs its HALF compacted at the
  stage base. That is exactly SM100_TMA_2SM_LOAD semantics:
  `cp.async.bulk.tensor.cta_group::2`, leader-issued, HW splits the
  box across the pair's SMEM, ONE mbarrier arrival covers both halves.

### Exact touch points in B's code

1. config.py: `use_tcgen05_ts_cg2` forcing the mc_a=2-style pairing
   (scheduler: adjacent n_block at same m_block; mc plumbing gives the
   cluster launch + cluster-mapped mbar init, as SS-cg2 reused mc_b).
2. memory/g2s_loader/loader_a.cuh:64-76 (`load_tma`): add a cg2 branch
   -- leader-only issue of the 2SM TMA (new ptx wrapper
   `tma_load_2d_cg2`), expect-tx on the LEADER's stage mbar = full-tile
   bytes; box M = BlockM covering both halves. The peer's producer
   thread issues nothing for A. NOTE: the peer CTA's math threads never
   read A from SMEM (s2r skips loader_a under TCGEN05; only the
   MMA-B descriptor reads it), and under cg2 only the LEADER issues the
   MMA, so only the leader's stage-arrival mbar matters for RAW; the
   peer's stage lifecycle needs an explicit review of consumer.arrive
   accounting (stage refill WAR: peer's half is overwritten by the
   NEXT leader-issued 2SM TMA -- the WAR gate must prove the pair MMA
   retired, which the multicast cg2 commit (below) gives both CTAs).
3. mma/tcgen05_ts_mma.cuh transform_b(): unchanged dequant + st (own
   tile, own TMEM). The per-slot WAR gate (arrivals_/waits_ vs
   tcgen05_ts_mbar[slot]) must become PAIR-scoped: slot commit switches
   to the leader-issued multicast cg2 commit
   (umma_arrive_multicast_2x1SM, mask 0x3) landing on BOTH CTAs' slot
   mbars -- same drain pattern SS-cg2 uses for its epilogue, applied
   per slot. Peer never commits (it issues no MMA).
4. tcgen05_ts_mma.cuh run(): after ctx.sync_math_threads() +
   fence_after_thread_sync, add the SS-cg2 pair rendezvous
   (cg2_rendezvous: cluster-mapped mbar arrive local+remote, local
   wait) so the peer's tcgen05.st of ITS half is complete before the
   leader issues. Then leader-only elect-one issues
   tcgen05.mma.cta_group::2 TS variant with idesc M = 256 (M_tot),
   a_tmem = staging slot col (same col index in both CTAs by paired
   alloc), b_desc = CTA-local activation half (fragment starts at row 0
   of the descriptor -- holds because the 2SM TMA compacts each half at
   the stage base). No {m0..m3} mask on the cg2 PTX (B.32).
   OPEN QUESTION to verify on-die: whether tcgen05.st (TMEM store) by
   the PEER is made visible to the leader's cta_group::2 MMA by
   fence::before_thread_sync + cluster mbar arrive/wait alone --
   CUTLASS's 2SM mixed-input kernels use exactly this pattern
   (tmem_store -> fence -> cluster barrier -> leader mma), so expected
   yes, but this is the first thing to synccheck.
5. Epilogue final_regs_c_as_ptr(): UNCHANGED math -- 2x1SM CLayout
   gives CTA r TMEM D = (M_tot/2 = its own 128 weight rows) x full
   MmaN = BlockM cols, i.e. exactly the cg1 TS transposed D. Each CTA
   drains its own D and writes its own (m_block, n_block) output tile.
   Drain commit is already leader-multicast under (3)'s scheme (reuse
   SS-cg2's epilogue drain).
6. humming_ws.cuh alloc/dealloc: reuse SS-cg2's cta_group::2
   alloc/relinquish/dealloc + pre-dealloc rendezvous verbatim; TMEM
   cols = TS's kTcgen05TmemCols (16 + BlockM rounded) per CTA.

### v1 constraints (inherit both prototypes' intersections)

BlockN=128 (one 128-row tile per CTA, M_tot=256 <= cg2 idesc max),
BlockM in {64,128} -> MmaN in {64,128} (cg2 N range ok), BlockK=64,
WS-only, bf16 A, uint4 B, int zp. Even n-tile count per M handled by
the mc scheduler's existing edge behavior (verify grid-edge odd-N
behaves like mc_a=2 does today: unpaired tail tile must fall back or
pad -- same question SS-cg2 answered for odd M-tiles).

### Why this composition is attractive

SS-cg2's dequant-halving lever died (warp starvation, milestone 5);
TS-cg2's lever is DIFFERENT: the activation halves. Each CTA's
SMEM stage traffic for A halves (BlockM/2 rows), and the pair
fetches each activation tile from L2/DRAM once instead of twice --
at large M activations dominate bandwidth, which is where cg1 TS is
strongest. Weight-side work is untouched (already the TS win).

### Milestone 5 validation (this session)

* tests/test_tcgen05.py -k cg2: 6/6 passed (incl. bitwise-vs-cg1).
* NEW grid-edge tests test_tcgen05_cg2_odd_tile_grid_edge[320|384|640]:
  odd M-tile counts, rank-1 tail CTA partially/fully past shape_m --
  all bitwise-match cg1, no deadlock (scheduler launches full pairs;
  tail CTA runs on TMA-zero-filled data, predicated stores). The old
  "shape_m must be a multiple of 2*BlockM" comment was wrong; fixed.
* Full suites: 76 passed / 24 skipped / 1 xfailed (was 73/24/1).
* compute-sanitizer, cg2 BM128/BN128/BK128 s4 m512 n512 k1024:
  - synccheck: 0 errors.
  - racecheck: 4 hazards -- but the cg1 SS baseline shows the
    IDENTICAL 4 write->read hazard-pair structure (512 instances
    each), i.e. pre-existing mbarrier-blind false positives on the WS
    producer-store vs consumer-read pipeline (racecheck does not model
    mbarrier arrive/wait). Not cg2-introduced. Logs:
    /tmp/racecheck_cg{1,2}_full.log (this box).

## Milestone 6: PERF VERDICT — cg2-on-SS is a NEGATIVE RESULT

bench_cg2_ss.py --cg2, GPU 2, 50 iters / 10 warmup, µs. cg1 numbers
reproduce milestone 1 within noise.

```
shape            M      mma.sync   tcg-ss cg1   tcg-ss cg2   cg2/cg1
Llama70B gate    512      1021.3        788.8       1115.0     1.41x
Llama70B gate    2048     3853.1       2835.8       4006.2     1.41x
Llama70B gate    4096     7626.7       5566.9       7847.0     1.41x
Llama70B down    512      1057.5        798.3       1113.3     1.39x
Llama70B down    2048     3690.2       2782.4       3882.5     1.40x
Llama70B down    4096     7375.2       5557.0       7757.8     1.40x
```

cg2 is uniformly ~1.40x SLOWER than cg1 at every large-M point on both
Llama70B shapes — worse than mma.sync at some points. The large-M
tensor-core-utilization hypothesis is REFUTED for the SS kernel.

### Attribution (timing-only experiment, correctness knowingly broken)

Re-ran M=2048 with TCGEN05_CG2_DEBUG_NO_RENDEZVOUS +
TCGEN05_CG2_DEBUG_NO_FENCE (skip the per-K-iter pair handshake +
async-proxy fence entirely; reverted after the run):

```
                 cg1     cg2      cg2-no-rdv   handshake cost
gate  M=2048    2834.4  4006.2      3110.8        ~895 µs
down  M=2048    2780.9  3882.5      3470.5        ~412 µs
```

* The per-K-iter cluster rendezvous accounts for 40-75% of the gap
  depending on shape, i.e. hundreds of µs — the two-CTA lockstep
  (cluster-mapped mbarrier arrive + local wait every K-iter) is
  fundamentally expensive on this mainloop, which has a K-iter every
  ~2000 cycles (the scatter).
* Even with the handshake FREE (broken-correctness bound), cg2 is
  still 1.10-1.25x SLOWER than cg1. With the warp-redundancy dequant
  (milestone 5) each CTA does the same transform work as cg1, so
  there is NO work saving left; the residual loss is plausibly the
  leader-only single MMA issue stream + lockstep scheduling rigidity
  vs two independent CTAs. Milestone 2 already showed cg1 issue rate
  saturates the tensor cores — cg2 had nothing to add.

### Conclusion for the track

cg2-on-SS: CORRECT (bit-identical to cg1, suites green, synccheck
clean, grid-edge tested) but a clean perf negative. Keep the code
behind use_tcgen05_cg2 (off by default) as the validated cta_group::2
substrate — alloc/dealloc, pair rendezvous, multicast commit, mask-free
cg2 PTX, scheduler pairing are all proven, and the TS composition
(design note above) reuses ALL of that machinery with a DIFFERENT perf
lever (activation-half SMEM traffic + single fetch of each activation
tile via the 2SM TMA), so the negative SS result does not condemn
TS+cg2. Do not enable cg2 in any SS heuristic.
