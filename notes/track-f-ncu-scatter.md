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
