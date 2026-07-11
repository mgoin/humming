# Track b-ts-staging — TS-mode tcgen05 staging (method 2 proper)

Findings log, workbook style. Facts, measured numbers, dead ends with WHY.

## Milestone 1: SS-mode baseline on THIS GPU (B300 SXM6 AC, GPU 1, sm_103a)

`CUDA_VISIBLE_DEVICES=1 .venv/bin/python benchmarks/bench_ts_baseline.py`
(50 iters, 10 warmup, wall clock; branch sm100-tcgen05-v2 @ 4c4ab24)

```
shape                M  mma.sync us    tcg-SS us              cfg   ss/mma
Llama70B-gate       16        166.0        230.0     M64K128s4+ws    0.72x
Llama70B-gate      128        328.9        232.0    M128K128s4+ws    1.42x
Llama70B-gate      512       1061.2        788.5    M128K128s4+ws    1.35x
Llama70B-gate     2048       3988.8       2836.6    M128K128s4+ws    1.41x
Llama70B-down       16        279.7        393.2     M64K128s4+ws    0.71x
Llama70B-down      128        280.7        397.7    M128K128s4+ws    0.71x
Llama70B-down      512       1132.1        797.5    M128K128s4+ws    1.42x
Llama70B-down     2048       3882.3       2779.6    M128K128s4+ws    1.40x
```

Matches workbook B.35/B.36 numbers within noise (Llama70B-down M=2048 was
2787 us there, 2780 us here). Every TS-mode perf claim below is vs THESE
numbers, same GPU, same timing loop.

Shapes: Llama70B-gate = N=28672 K=8192; Llama70B-down = N=8192 K=28672.
The TS prototype target to beat at M in {128, 512, 2048} is the tcg-SS
column.
