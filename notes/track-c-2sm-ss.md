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
