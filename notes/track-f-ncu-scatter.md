# Track f: NCU re-baseline + closed-form SS scatter

Workbook-style findings log. Branch `prototype/f-ncu-scatter` off
`sm100-tcgen05-v2`. GPU 5 of 8x B300 SXM6 (cc 10.3, sm_103a, CUDA 13.0).

## F.1 Same-GPU SS baseline (2026-07-11)

`benchmarks/bench_track_f_baseline.py` — fixed prod-WS config
(BlockM=128 BlockN=128 BlockK=128 stages=4 WS+TMA; M=16 uses BlockM=64
and pads up). wmma = mma.sync reference (bm=64 bk=64 s3). 50 iters,
10 warmup, perf_counter.

```
shape                M    wmma us     tcg us          tcg cfg  tcg/wmma
Llama70B gate     16     165.91     230.63     M64K128s4+ws     0.72x
Llama70B gate    128     329.29     232.24    M128K128s4+ws     1.42x
Llama70B gate    512    1061.24     789.27    M128K128s4+ws     1.34x
Llama70B gate   2048    3990.20    2839.00    M128K128s4+ws     1.41x

Llama70B down     16     280.68     393.87     M64K128s4+ws     0.71x
Llama70B down    128     281.01     398.45    M128K128s4+ws     0.71x
Llama70B down    512    1125.04     798.17    M128K128s4+ws     1.41x
Llama70B down   2048    3875.60    2790.03    M128K128s4+ws     1.39x
```

Matches workbook B.36 (down M=2048 was 2787 us pre-port) — the port to
latest main did not regress the SS kernel. Correctness:
`tests/test_tcgen05.py` 44 passed / 1 xfail on this branch before any
change.

## F.2 NCU re-baseline on v2 branch (Part 1)

Protocol identical to workbook B.34/B.36: Llama70B-down M=2048,
prod-WS config (128/128/128 s4 WS), `ncu --launch-skip 2
--launch-count 1 --kernel-name regex:humming --set full` (no sudo
needed on this box). Target: `benchmarks/ncu_target_prodws.py`.
Report: /tmp/trackf_full_v2.ncu-rep. Grid (148,1,1)x(384,1,1).

### SoL vs B.36 (pre-port, stages=4)

```
metric                     v2 (now)   B.36    delta
Compute (SM) Throughput     57.28 %   57.03   +0.3 pp
Memory Throughput           56.79 %   53.24   +3.6 pp
L1/TEX Cache Throughput     58.83 %   55.79   +3.0 pp
DRAM Throughput              2.79 %    2.92   ~
L2 Cache Throughput          6.41 %     n/a
L2 Hit Rate                 71.70 %   71.69 (B.34)
Executed IPC (issued)        2.37      2.38 (B.34)
Achieved Occupancy          18.75 %   18.75  (1 CTA/SM)
Registers/thread             168       168
Dynamic SMEM/CTA           232.45 KiB 230.78  (+1.7 KiB, upstream
                                               port grew storage a bit)
Elapsed cycles             5.33 M @ 1.09 GHz (ncu-clocked)
```

### Stall breakdown (cycles per issued inst, 5.05 total — identical to B.34/B.36)

```
                     v2 (now)   B.36 s4   delta
long_scoreboard :      1.54       1.52     +0.02
wait (mbar)     :      0.66       0.66      0.00
not_selected    :      0.45       0.45      0.00
barrier         :      0.44       0.44      0.00
math_pipe_throt :      0.30       n/a
no_instruction  :      0.27       n/a
mio_throttle    :      0.10       0.08     +0.02
short_scoreboard:      0.10       0.09     +0.01
selected        :      1.00
```

**Conclusion: upstream's compile-time offset folding (e2f5813) and L2
raster (29e0313) did NOT shift the profile.** Every stall bucket is
within 0.02 cyc/inst of B.36. Compute SoL still 57%; barrier still
~8.7% of warp cycles; long_scoreboard still ~30%.

### Pipe utilization (% of peak sustained active)

```
ALU pipe (inst)  : 45.79 %   (B.34: 46.28)
LSU pipe (inst)  : 41.44 %   (B.34: 40.66)
FMA pipe cycles  : 36.48 %
tensor cycles    : 15.48 %
SMEM st wavefronts: 31.03 % of peak elapsed  (243.9 M wavefronts)
SMEM ld wavefronts: 18.79 % of peak elapsed  (147.7 M wavefronts)
```

### Source-level stall attribution (warp stall sampling, 185 k samples)

```
site                                      samples   dominant stall
top-4 spin-wait BRAs (producer/consumer
  mbarrier try_wait loops)                 27.5 %   long_scoreboard
all barrier stalls (per-K-iter bar.sync,
  spread over unrolled bodies)              8.8 %   barrier
plain STS (the r2s scatter, 234.9 M inst)   7.3 %   (issue slots)
LOP3 total (dequant + scatter addr math)   12.1 %
SHF  total (dequant nibble shifts)          8.1 %
```

**New facts the pre-port profiles missed:**

1. **The scatter is NOT intra-warp bank-conflicted.** Source counters
   for the scatter STS: `L1 Wavefronts Shared` = `Ideal` =
   234,881,172, `Excessive` = 0, max N-way conflict = 1. The
   workbook's "4-way bank conflict by design" is actually **cross-warp
   write duplication** (the 4 M-warps store identical bytes to the
   same addresses; each warp's store is conflict-free). NCU counts
   wavefronts per-warp, so the redundancy shows up as 4x instruction
   count (235 M scatter STS where ~59 M unique would do), not as
   conflict serialization. Swordfish's "conflict-free by construction"
   scatter is therefore NOT an advantage over ours at the wavefront
   level — their win is addressing cost + warp specialization.
2. **The biggest single stall block is producer/consumer mbarrier
   spin loops** (4 BRA sites = 27.5% of all samples, long_scoreboard):
   warps parked in `mbarrier.try_wait` spin loops. This is
   pipeline-structure cost (what track a attacks), not scatter cost.
3. Scatter instruction budget per math-warp per K-iter (from SASS of
   one steady-state body, `~/.humming/cache/c87a4a277a219f02`):
   16 STS + ~32 LOP3 of address math (per store: one 3-way OR
   `LOP3 0xfe` merging base|imm|base2, then one XOR `LOP3 0x3c` with
   the hoisted swizzle-phase register). ptxas already hoists the XOR
   phase to a per-thread register (it proved `(n0+16i+8f)&7 == n0&7`)
   but still burns 2 LOP3 + 16 live address registers per store
   because it applies OR-imm *before* the XOR. One K-iter body is
   ~195 inst, so scatter address math is ~16% of the math-warp
   instruction stream.
4. Total kernel: 2.17 G inst executed; scatter STS = 235 M (10.8%);
   epilogue STS.128 = 0.29 M (negligible).

## F.3 Closed-form scatter derivation (Part 2)

Definitions (from `tcgen05_mma.cuh::run()`, BlockK >= 64 so
kKPerSectionB = 64, kRowBytes = 128):

```
t    = lane (0..31)
n0   = n_base + t/4          n_base = (warp % kNWarps) * 64
k_lo = 16*iter + 2*(t%4) + 8*pair        (pair in {0,1})
n    = n0 + 16*i + 8*frag                (i in 0..3, frag in {0,1})

linear = (iter/4)*kBSectionSizeBytes         ; K-section crossing
       + n*128 + ((16*iter)%64 + 2*(t%4) + 8*pair)*2
swizzled = linear ^ ((( (smem_base>>7) + n) & 7) << 4)
```

Bit-occupancy argument (BlockN <= 256):

```
term                       bits
4*(t%4)                    2-3
32*(iter%4)                5-6
16*pair                    4
(t/4)*128                  7-9
frag*1024                  10
i*2048                     11-12
n_base*128                 13-14        (n_base in {0,64,128,192})
(iter/4)*sectionBytes      14+ (16384 for BN=128)
```

All additive terms occupy pairwise-disjoint bit ranges, so + == | == ^
among them. And the XOR phase `((base+n) & 7)` is invariant in i,
frag, pair AND iter (n changes only in multiples of 8; k terms don't
reach bit 7 within a section) — it is a pure function of the
destination row's low 3 bits, i.e. of (n0 + smem_base/128) & 7.

Therefore:

```
mask  = (((smem_base>>7) + n0) & 7) << 4          ; per-thread constant
pre   = n0*128 + 4*(t%4)                          ; per-thread constant
                                                   (bits 4-6 of pre = 0)
base0 = (pre ^ mask) ^ 32*(iter%4) + secOff       ; 2 ALU per K-iter
base1 = base0 ^ 16                                ; 1 ALU per K-iter
store(i, frag, pair):  STS [base{pair} + (i*2048 + frag*1024)]
                                                   ; immediate offset,
                                                   ; ZERO per-store ALU
```

The `^ 32*(iter%4)` must be XOR (not add) because base0's bits 5-6
already hold mask bits; all other combines are carry-free. This
reduces per-K-iter scatter address math from ~32 LOP3 + 16 addr regs
to 3 ALU + 2 addr regs, and per-store cost to exactly one STS with an
immediate offset. Validated by a compile-time `static_assert` that
enumerates (t, i, frag, pair, iter) against the original formula
(see `tcgen05_mma.cuh`).

Prediction from F.2 fact 3: removes ~30 of ~195 inst per K-iter body
(~15% of math-warp inst stream), but the kernel is NOT issue-bound
(0.59 issue/cycle/smsp; stalls dominated by long_sb+wait+barrier =
52%), so wall-time gain should be well under 15%. HYPOTHESIS (task):
second-order vs the bar.sync serialization.

Generated SASS confirms the design: per K-iter the scatter is exactly
16 `STS [Rbase{0,1}+imm]` off two base registers with compile-time
immediates, zero per-store ALU (verified in
/tmp/trackf_cache/c00e90c3.../kernel.cubin; ordering STS -> BAR.SYNC
-> UTCHMMA -> MEMBAR intact).

## F.4 Microbench: the workbook's 2052-cyc r2s figure was an artifact

`benchmarks/bench_r2s_vs_r2t.cu` extended with two variants (B300,
1 CTA, 256 threads, 10000 iters):

```
Path 1  (r2s synthetic addr + bar.sync)   : 2052.1 cyc/iter
Path 1' (r2s synthetic, no bar)           : 2048.1 cyc/iter
Path 1p (r2s PRODUCTION addressing + bar) :  292.0 cyc/iter   NEW
Path 1c (r2s CLOSED-FORM addr + bar)      :  143.5 cyc/iter   NEW
Path 2a (r2t 8-warp + fence)              :   43.0 cyc/iter
Path 2b (r2t 2-warp + fence + bar.sync)   :   53.5 cyc/iter
```

The original Path-1 kernel used a synthetic offset pattern
`(tid*16+s+it*7) & 1023` whose lane stride is 16 words == 2 banks
apart -> ~16-way intra-warp bank conflicts. Humming's REAL scatter is
conflict-free by swizzle construction (F.2 fact 1), and costs 292
cyc/iter, not 2052. Consequences:

* The Path-2 doc's "r2s is ~50x more expensive than r2t" TL;DR is
  wrong: the true primitive gap is 292/43 = 6.8x (old addressing) or
  143/43 = 3.3x (closed form). The TS-mode pipeline-restructure
  argument survives (Swordfish errata already said the restructure is
  where the cycles are), but the r2s-primitive-cost motivation is
  much weaker than documented.
* The closed form halves the isolated scatter primitive (292 -> 143
  cyc/iter, 2.03x) by eliminating ~2 LOP3 per store.

RE-MEASUREMENT (same GPU 5, final committed bench binary, stable
across 3 runs at 2032 MHz): Path 1 = 4449, Path 1' = 4446, Path 1p
= 780, Path 1c = 143.5, Path 2a = 43.0, Path 2b = 53.5 cyc/iter.
Paths 1c/2a/2b reproduce the table above exactly; Paths 1/1p read
~2.2x/2.7x HIGHER than first recorded (cause unknown -- possibly an
earlier binary build; the first-recorded 2052/292 were not
re-reproducible). Use the re-measured numbers: closed form cuts the
in-kernel-style r2s primitive 780 -> 143.5 (5.4x); r2t remains 3.3x
cheaper than the best r2s (143.5 vs 43). The qualitative conclusions
(synthetic 16-way-conflict figure was an artifact; restructure, not
raw primitive cost, is TS-mode's main win) stand.

## F.5 The BlockK=64 WS+TMA corruption: a pre-existing producer-side
## race that the closed-form scatter amplifies (2026-07-11)

The closed-form scatter is address-exact (compile-time
`static_assert scatter_closed_form_matches()` enumerates every
(t, warp, iter, i, frag, pair) against the element-wise formula), yet
enabling it at the SAFE_PROD_WS_CONFIG BlockK=64 configs produced
wrong outputs. Full experiment matrix, uint4 zp=T (512,512,4096)
block (128,128,64) s3 unless noted, GPU 5 (B300), bf16-noise
baseline max|err| = 2.0:

```
E1  element-wise BK64, WS+TMA / WS-only / plain     : 2.0 everywhere (x12 runs
                                                      incl. E4 stress) CLEAN
E2  closed-form  BK64, WS+TMA                       : 356 / 343 / 374  FAIL
E2  closed-form  BK64, WS-only (cp.async), plain    : 2.0              CLEAN
E2b closed-form  BK64, WS+TMA, seeded identical x2  : NOT bit-exact ->
    genuine nondeterministic race. 1.77% bad cells spread over ALL
    16 output tiles; within a tile bad cols cluster in n%128 = 32..63.
E3  E2 + fence.proxy.async after scatter            : 441 / 295 / 370  FAIL
    (TCGEN05_DEBUG_SCATTER_PROXY_FENCE -- does NOT fix)
E5a closed-form  BK128 s4 uint4 zp=T (B.37 broken)  : 313 / 295 / 215  FAIL
E5a closed-form  BK128 uint4 zp=F s4, uint8 zp=T s3 : 2.0              CLEAN
E5b element-wise BK128 s4 uint4 zp=T (HEAD-equiv)   : 4.0 / 2.0 / 2.0  MARGINAL
    (B.37 documented this config failing outright at the same shape --
    it is WHY uint4 was demoted to BK64 in SAFE_PROD_WS_CONFIG)
```

Plus predecessor probes (per staged code comments): MMA-drain before
every scatter (TCGEN05_DEBUG_DRAIN_PER_ITER) and defer-arrive-until-
MMA-drain at stage release (TCGEN05_DEBUG_DEFER_ARRIVE) do NOT fix
E2 either.

**Verdict: this is NOT a bug in the closed-form scatter and NOT the
scatter->mma visibility gap. It is the pre-existing workbook-B.37
WS+TMA race** ("disabling TMA alone fixes it" -- exactly reproduced
here: WS-only is clean at identical geometry). The race lives on the
producer (TMA) side of the WS pipeline; the closed-form scatter only
compresses math-warp K-iter time (~30 fewer inst/iter), which shifts
consumer timing enough to pull BK64 s3 into the failing envelope that
BK128 s4 uint4 zp=T was already inside at HEAD.

Ruled out by direct experiment:
* b_dequant ping-pong WAR vs async MMA reads (DRAIN_PER_ITER no-fix)
* A-stage release WAR (consumer.arrive at kWarpIters-2 vs in-flight
  MMA descriptor reads; DEFER_ARRIVE no-fix) -- track B's latent
  audit item is NOT this bug (still worth fixing for hygiene)
* generic->async proxy visibility of the scatter STS
  (SCATTER_PROXY_FENCE no-fix). NOTE: the fence is still formally
  required by the PTX memory model -- transform_b's
  fence_proxy_async_shared_cta() executes BEFORE the deferred
  scatter stores it is supposed to publish (SASS: STS -> BAR.SYNC ->
  UTCHMMA -> MEMBAR per K-iter, membar on the WRONG side of the
  UTCHMMA). Latent hazard; fix independent of this race.
* BOTH drain probes together (E7: DRAIN_PER_ITER + DEFER_ARRIVE
  simultaneously, closed form on): 322 / 370 / 328 -- STILL FAILS.
  This is decisive: with a full MMA drain before every scatter and
  before every stage release there is no in-flight UTCHMMA read left
  to race with, yet the output is still corrupt. The corruption is in
  the CONSUMED INPUT DATA (A / B-codes / scales / zp as delivered by
  the TMA producer), not in any math-side WAR.
* compute-sanitizer racecheck (closed form BK64 WS+TMA): 62,467
  errors, all displayed records ONE site pair -- WAR at stage-SMEM
  bytes (~0x5500-0x6100 region), Read = math-warp thread in the
  dequant block (SASS ~+0x4150), Write = producer thread 256
  `@!P0 STS.128 [R3+0x6100]` (SASS +0x1610: an LDG.E.128 ->
  STS.128 -> ARRIVES.LDGSTSBAR path, i.e. the producer's NON-TMA
  side-channel for zp/scales when use_tma_bzp=0). Racecheck may not
  fully model the mbar handshake (hazards also present in passing
  runs; corruption itself vanishes under sanitizer timing, err=2.0),
  so treat as a pointer, not proof. Full log:
  /tmp/trackf_racecheck_full.log; SASS: /tmp/trackf_bk64_sass.txt.
* NOT the math_mbar protocol shape: expected count =
  kNumMathThreads/32 (all math warps), consumer.arrive fires from
  lane 0 of every math warp -- accounting is consistent.

Best remaining hypothesis: premature stage-ready flip on the TMA
side -- e.g. `tma_commit_mbarrier(&load_mbar[stage], load_bytes.x)`
expect_tx undercounting vs what actually lands (the producer ALSO
does LDG->STS.128 zp/scale stores into the same stage between
mbarrier ops), so the consumer's wait_stage returns while part of
the stage (zp/scales or B codes) is still in flight. That is
nondeterministic, TMA-specific, dtype-dependent (zp=T adds
transfers), BlockK-dependent (transfer sizes), and consumer-speed
dependent (faster math = reads closer behind the premature flip) --
matches every observation including B.37's BK128 uint4 zp=T failure
with the SLOW element-wise scatter. Next probe for a future session:
TCGEN05_DEBUG_CONST_B (output = rowsum(A), independent of B/zp/bs)
to split A-path vs B/zp/bs-path corruption; then audit load_bytes
accounting in g2s_pipeline.cuh::load_stage vs the loaders' actual
TMA + STS traffic at (uint4, zp=T, BK64, s3).

Mitigation shipped on this branch: `kUseClosedFormScatter =
BlockShape::K >= 128` keeps the element-wise scatter at BlockK=64
(12/12 clean) and enables the closed form at BlockK>=128, where the
only affected config (uint4 zp=T s4) is already excluded from
SAFE_PROD_WS_CONFIG by B.37. Root cause of the producer-side race
remains OPEN -- next probes should target the TMA mbarrier
expect_tx/arrive accounting in g2s_pipeline.cuh and the producer's
`load_stage` overwrite timing, not the math-warp side.

### Is track B (TS mode) exposed? NO (LOUD VERDICT)

* Architecturally: the TS kernel (tcgen05_ts_mma.cuh) has no
  generic-proxy STS into descriptor-read SMEM at all -- dequant goes
  regs -> tcgen05.st -> TMEM with the documented
  fence::before_thread_sync / bar.sync / fence::after_thread_sync
  pattern; activations are TMA(async proxy) -> act_desc(async proxy)
  with mbarrier ordering.
* Empirically (GPU 1, /tmp/wt-b-check @ origin/prototype/b-ts-staging
  52f39ef): tests/test_tcgen05_ts.py 10/10 x6 consecutive runs, AND a
  targeted stress at the EXACT failing SS geometry -- uint4 zp=T
  (512,512,4096) block (128,128,64) s3 and s4, WS+TMA, 5 reps each:
  max|err| = 2.0 on all 10 runs (script:
  /tmp/wt-b-check/benchmarks/stress_ts_bk64_race.py).
* Caveat: the underlying producer-side WS+TMA race is
  timing-dependent and the TS consumer has different timing; "not
  exposed at every geometry we can hit" is the honest claim. The B.37
  root cause should still be found before TS ships as default.
