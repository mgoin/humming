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

## Milestone 3: TS-mode design (TMEM col map, mbar phases, pipeline)

Prototype constraints (asserted in `mma/tcgen05_ts_mma.cuh`):
M_WARPS==1 (WarpM==BlockM), K_WARPS==1 (WarpK==BlockK), WarpN==32,
BlockN==128 (exactly one MMA-M tile, 4 math warps = 128 math threads),
BlockK==64 bf16, ElementA==bf16, ElementB==uint4, group weight scale,
cta_group::1.

### Operand roles (A<->B swap)

MMA-A = dequantized weights from TMEM, MmaM = 128 = BlockN (weight rows,
lanes). MMA-B = activations from SMEM via descriptor (the SAME
Swizzle<3,4,3> desc + iter*2-uint128 advance the SS kernel uses for its
A operand), MmaN = BlockM (<=256, mult of 16 for M=128 atom -- both 64
and 128 OK). MmaK = 16 (kind::f16). TMEM D is transposed: D[lane =
weight row n][col = activation m].

CUTLASS confirms the TS PTX takes the {m0..m3} mask operand
(mma_sm100_umma.hpp SM100_MMA_F16BF16_TS::fma):
  tcgen05.mma.cta_group::1.kind::f16 [d], [a_tmem], b_desc, idesc,
    {0,0,0,0}, p;
Jinzhen's codegen omits the mask -- do NOT copy that (our SS work found
the mask-less form parses to a different variant and hangs).

### TMEM column map (single alloc, kTmemCols = pow2(16 + BlockM))

  base + 0  .. 8   : W staging slot 0 (16 K bf16 = 8 cols, 2 bf16/cell)
  base + 8  .. 16  : W staging slot 1
  base + 16 .. 16+BlockM : D accumulator (f32, MmaN=BlockM cols)

BlockM=64 -> alloc 128 cols; BlockM=128 -> alloc 256.

### Transform2Mma handshake (fixes Jinzhen bug #1, WAR race)

Per-slot mbarrier `tcgen05_ts_mbar[2]` (expected_count=1). All math
threads keep per-slot counters arrivals_[s] / waits_[s] (consistent
because every math thread executes run()/transform_b() in the same
order):
  * run(iter): elect-one issues MMA on slot s = iter%2, then
    tcgen05.commit -> ts_mbar[s]; ALL math threads increment
    arrivals_[s].
  * transform_b(s): after dequant (register work, no hazard), if
    arrivals_[s] > waits_[s]: mbarrier_wait(ts_mbar[s], waits_[s]&1),
    waits_[s]++. Then tcgen05.st into slot s.
This uniformly handles the pre-loop transform_b(0) of every tile and
the dangling end-of-tile transform (humming's mainloop emits one
transform per run PLUS one pre-loop + one trailing) with no
tile-boundary special case, and keeps mbar phase parity = waits_&1
consistent forever. commit batches all prior MMAs, so waiting on
arrival k transitively guarantees the slot-s MMA retired.

### Cross-warp st -> mma visibility (fixes Jinzhen bug #2)

transform_b ends with tcgen05.wait::st + tcgen05.fence::before_thread_sync
(producer warps). run() starts with ctx.sync_math_threads() (bar.sync of
the 128 math threads -- the real thread sync PTX requires) followed by
tcgen05.fence::after_thread_sync before the elect-one MMA issue.

### K-packing (fixes Jinzhen bug #3)

Solved by the CONTRACT + M2 reference packer: the TS s2r loader gives
thread (w,l) its row's 16-K chunk as 2 uint32 whose nibbles are
pre-interleaved so the lop3 dequant emits reg r = (K=2r, K=2r+1). The
TS path does its own dequant inline (plain (v - zp_or_8) * scale in
bf16x2; no exp-offset tricks, so no epilogue rescale is needed) and its
own per-lane scale/zp reads (lane = row ownership) -- humming's
fragment-ownership dequant/arith path is bypassed entirely.

### s2r loads (bypass loader_a fragment path AND loader_b half-group path)

Thread (w,l), iter i: codes = uint2 at smem.stages[stage].b byte offset
i*BlockN*8 + w*256 + l*8 (coalesced 256B/warp). Scale: bf16 at
smem.stages[stage].bs[n], n = 32w+l (gs=128 >= BlockK so 1 group/stage).
Zp: nibble n of smem.stages[stage].bzp group row.

### Epilogue (prototype: correctness-first t2r drain)

Commit + wait on the existing tcgen05_mbar, fence, then per warp
BlockM/32 x tcgen05.ld.32x32b.x32 at D cols; tcgen05.wait::ld. Thread
(w,l) holds out[m0..m0+31][n=32w+l]; writes bf16 SCALAR stores into
smem.reduce in gmem_writer's sectioned XOR-swizzled layout
(smem_row = (n/64)*BlockM + m, col swizzle as SS). 2-byte stores are
bank-ugly but correct; track e-epilogue-tmem owns the real epilogue.
gmem_writer then drains as usual (EpiloguePipeline already skips
smem_writer for kMmaType==TCGEN05).

### SMEM win vs SS (why this matters beyond the scatter)

smem.b_dequant (2 x BlockN x BlockK bf16) is NOT emitted for TS. At the
SS prod config that buffer is 64KB (BK=128) / 32KB (BK=64); TS at
BM=128 BN=128 BK=64 s=4 pencils out to ~81KB/CTA total -- under the
116KB 2-CTAs/SM bound, which SS could never reach.
