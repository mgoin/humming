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
