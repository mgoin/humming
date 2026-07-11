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

## Milestone 4: TS-mode MMA fires correctly (2026-07-11)

`tests/test_tcgen05_ts.py` -- 10/10 pass on B300 GPU 1:
* 512^3 bf16 x uint4 gs=128, zp on AND off
* stages {2, 3, 4} (kNumStages==2 exercises the deferred-load path)
* BlockM {64, 128} (MMA-N = 64 and 128 atoms)
* has_bias=True (per-lane bias add in the t2r drain)
* warp-spec + TMA on
* M=128 N=1024 K=8192 (multi-N-block, 128-iteration K walk, s=4)

SS regression suites untouched: test_tcgen05.py + test_tcgen05_dtypes.py
+ test_sm100_smoke.py = 77 passed / 24 skipped / 1 xfailed (identical to
pre-change).

### Tolerance finding (worth keeping)

Against the fp32-weight reference the K=8192 shape shows 21/131072
elements over atol=0.5 (max 0.875). Against a reference whose weights
are rounded to bf16 after (code-zp)*scale -- exactly what the kernel's
__hmul2 dequant produces -- mean|err| drops 0.093 -> 4e-4 and max err
is 1.0 = precisely 1 bf16 ulp at |out| in [128,256). I.e. the TS path
is bit-faithful to its dequant semantics; the drift is bf16 weight
rounding, same class as workbook B.38's "looser atol at prod shapes"
note. The TS test rounds the reference weights instead of loosening
atol -- strictly tighter check.

### What made it work first-try (for the record)

* The Python contract simulation (M2) validated the pack -> smem ->
  regs -> dequant mapping before any CUDA ran. The historically
  hardest part of this port (the (thread, reg) -> (n, k) mapping,
  cf. Phase B.10's weeks of scatter debugging) was verified offline.
* TS PTX form taken from CUTLASS SM100_MMA_F16BF16_TS::fma verbatim
  (WITH the {m0..m3} mask operand Jinzhen's codegen dropped).
* The mbar counter handshake (design note above) needed no tile-
  boundary special cases; phases stayed consistent across scheduler
  blocks including the stream-k-style trailing transform.

## Milestone 5: TS-mode perf vs same-GPU baselines (GPU 1, 50 iters)

`benchmarks/bench_ts_vs_ss.py`. SS column = the M1 baseline config
(BM128 BK128 s4 WS, humming's prod heuristic). TS column = best of
stages {4,6} x ws {on,off} (s=4+ws won every point; s=6 never helped).

```
shape               M    mma us     SS us         TS cfg     TS us   SS/TS  mma/TS
Llama70B-gate      16     160.1     229.9    M64K64s4+ws     166.2   1.38x   0.96x
Llama70B-gate     128     317.7     231.8   M128K64s4+ws     203.3   1.14x   1.56x
Llama70B-gate     512    1024.2     788.3   M128K64s4+ws     697.4   1.13x   1.47x
Llama70B-gate    2048    3852.2    2836.6   M128K64s4+ws    2486.3   1.14x   1.55x
Llama70B-down      16     266.6     393.2    M64K64s4+ws     281.2   1.40x   0.95x
Llama70B-down     128     266.9     397.8   M128K64s4+ws     347.8   1.14x   0.77x
Llama70B-down     512    1057.2     797.8   M128K64s4+ws     690.4   1.16x   1.53x
Llama70B-down    2048    3687.1    2780.6   M128K64s4+ws    2399.5   1.16x   1.54x
```

* TS beats the SHIPPED SS config (which gets BlockK=128!) by
  1.13-1.16x at M >= 512 on both shapes, and by 1.14x at gate M=128.
* At M=16 TS narrows the tcgen05-vs-mma.sync gap from 0.71-0.72x (SS)
  to 0.95-0.96x -- the crossover point drops substantially.
* mma.sync still wins Llama70B-down M=128 (0.77x) -- same window SS
  loses (0.71x); small-M work belongs to the decode-path track.
* The prototype TS kernel with a scalar-store epilogue and a per-K-iter
  commit+mbar handshake ALREADY beats a heavily-tuned SS kernel. The
  epilogue (track e) and cta_group::2 headroom are on top of this.

## Milestone 6: r2t-vs-r2s IN SITU (identical geometry)

`benchmarks/bench_ts_m6_insitu.py`: both kernels at BM=128 BN=128
BK=64 s=4 WS+TMA; the ONLY difference is the staging path
(r2s scatter + bar.sync 256 vs r2t + bar.sync 128 + fences + per-iter
commit/mbar WAR gate):

```
shape               M  SS-K64 us  TS-K64 us SS-K64/TS
Llama70B-gate     128      247.3      203.2     1.22x
Llama70B-gate     512      842.4      697.6     1.21x
Llama70B-gate    2048     3031.3     2485.6     1.22x
Llama70B-down     128      426.6      347.4     1.23x
Llama70B-down     512      851.6      690.6     1.23x
Llama70B-down    2048     2968.1     2399.3     1.24x
```

ANSWER to the M6 question: of the 50x primitive-cost gap (43 vs 2052
cyc in the microbench), what survives in the full kernel is a uniform
**1.21-1.24x wall-time win**. The Swordfish reading was right: the
kernel partially hides the scatter (57% compute SoL), so you get ~20%
not 50x -- but it's a REAL 20%+ that also frees 32-64KB SMEM/CTA and
removes the 256-thread barrier. Note TS-K64 also beats SS at SS's own
best K128 config (M5 table), so the win is not an artifact of pinning
SS to K64.

### SMEM footprints (from kernel.cu SMEM_SIZE, BM128 BN128 BK64 s4 WS)

TS drops smem.b_dequant entirely; the stage budget is ~81KB/CTA vs
~114KB (SS K64) / ~224KB (SS K128 prod). 2 CTAs/SM (needs <= ~116KB)
is now in reach for the first time -- future work with TMEM col
budgeting (2x (16 + BlockM) <= 512 holds for BM128: 2x144=288 OK... but
tcgen05.alloc pow2 -> 2x256 = 512, exactly fits).

## Open items / known limitations of the prototype (honest list)

* Config space pinned: BlockN==128, WarpN==32, BlockK==64, M_WARPS==1,
  K_WARPS==1, uint4 + group scale (gs >= BlockK), bf16 A. Widening
  BlockK needs section-major TMEM staging; BlockN=256 needs
  kNumMmaMTiles=2 (Jinzhen's tile geometry salvage covers the math).
* Per-K-iter tcgen05.commit + mbar wait is the crude version of
  CUTLASS's Transform2Mma pipeline; batching the commit every 2 iters
  (or an ld/st-scoreboard scheme) is untried perf headroom.
* Epilogue drain uses 2-byte scalar smem stores (heavily
  bank-conflicted); track e-epilogue-tmem owns the real one. Perf
  above INCLUDES this handicap.
* The consumer.arrive(stage) at kWarpIters-2 releases the activation
  stage while the last 1-2 TS MMAs of the stage may still read its
  SMEM through the descriptor (same latent exposure the SS kernel
  ships with; tests pass, but a producer that refills faster could
  expose it -- inherited, not introduced).
* Weights must be packed by tests/ts_contract_pack.py (throwaway);
  swap in track d-packing's production packer behind the same
  CONTRACT.

## Sanitizer check

`compute-sanitizer --tool synccheck` over test_tcgen05_ts.py::test_ts_stages
(stages 2/3/4, the deepest handshake coverage): 3 passed, ERROR SUMMARY:
0 errors. The Transform2Mma mbar handshake and the st->sync->mma fence
chain are clean under the tool that catches "Missing wait" mbar bugs.
