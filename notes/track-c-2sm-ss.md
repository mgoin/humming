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
