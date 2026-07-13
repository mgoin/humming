# TCGEN05 TS-mode mainloop ("Path 2") — investigation notes

Status: **deferred** as of 2026-05-20. This document captures what we
learned about moving humming's TCGEN05 mainloop from SS mode (both
operands from SMEM) to TS mode (A from TMEM, B from SMEM with A↔B
swapped). The current SS-mode implementation in
`humming/include/humming/mma/tcgen05_mma.cuh` is what we ship; this
doc is for the next person who picks up the refactor.

## TL;DR

* The current per-K-iter `r2s scatter + bar.sync 1, 256` costs
  **2052 cyc** on B300 sm_103a. The bar itself is essentially free
  (4 cyc); the SMEM scatter is the bottleneck.
* Per-warp `tcgen05.st` (r2t) costs **43 cyc** at the same data volume.
  That's the **~50×** primitive-cost gap.
* CUTLASS **defaults to TS mode** for mixed-input F16/BF16 on SM100.
  The fallback to SS exists only when A needs MN-major access. For
  2-CTA cluster mixed input, TS is the only mode they offer.
* Wall-time win in the full kernel will be smaller than 50× because
  the kernel today partly overlaps the scatter with other work
  (NCU: 57 % Compute SoL), but freeing the per-K-iter SMEM-store
  budget unlocks a meaningfully higher ceiling.

## Why SS mode is what we have today

Path 1 = "SS mode" (Source-Source: both A and B come from SMEM):

```
transform_b():  int4 codes  --(dequant)-->  regs_b_tmp[buffer_id]  (RMEM)
run():
  r2s scatter:  regs_b_tmp[buffer_id]  -->  smem.b_dequant[buffer_id]
  bar.sync 1, 256                                # math-only barrier
  tcgen05.mma.cta_group::1.kind::f16 [tmem_c], a_desc, b_desc, idesc
                                                  ^         ^
                                            smem.a    smem.b_dequant
```

* `smem.b_dequant[2][BlockN*BlockK]` is the staging buffer. Sized for
  two K-iter ping-pong slots (`iter_id % 2`).
* TMEM holds only the accumulator (128 cols).
* `bar.sync` is structurally load-bearing: in Phase B.36b an
  experimental fence-only replacement caused the kernel to slow ~7×
  because the bar gates the tcgen05.mma issue pipeline ordering.

## Why TS mode (Path 2) is the higher-ceiling architecture

Path 2 = "TS mode" (TMEM-Source: A from TMEM, B from SMEM), with the
A↔B operand-role swap:

```
transform_b():  int4 codes  --(dequant)-->  regs_b_tmp        (RMEM)
              r2t (tcgen05.st):  regs_b_tmp  -->  tmem.a_dequant
run():
  tcgen05.fence::before_thread_sync                # warp-local fence
  tcgen05.mma.cta_group::1.kind::f16 [tmem_c], [tmem_a_dequant], b_desc, idesc
                                                ^                   ^
                                          new TMEM region    smem.a (= activations)
```

In humming naming: the dequant'd weights (what we call "B") land in
TMEM and become the LEFT operand of the MMA. The activations stay
in SMEM as the RIGHT operand. The output geometry (M×N) is what we
want directly -- A↔B swap is conceptual: we *rename* the operands
to match the tcgen05.mma TS-mode signature, the math is unchanged.

### Concrete advantages

1. **The per-K-iter SMEM scatter goes away.** r2t is one
   `tcgen05.st.32x32b.x32` per warp -- one inst, not 16 stores per
   thread × 256 threads.
2. **The 256-thread `bar.sync` is replaced by a per-warp fence.**
   `tcgen05.fence::before_thread_sync` orders subsequent tcgen05.mma
   issues from the same warp relative to the prior tcgen05.st. No
   cross-warp barrier needed for that ordering.
3. **64 KiB of SMEM freed.** `smem.b_dequant` disappears. We're at
   230 KiB / 232 KiB cap today (Phase B.36); freeing 64 KiB takes us
   to ~166 KiB. That's still over the 116 KiB cap for 2 CTAs/SM at
   BlockM=BlockN=128, but **combined with one more move** (e.g.
   splitting `smem.reduce` out of the union, ~32 KiB more, vs the
   workbook B.33 blocker which had no budget) the path to 2/SM
   becomes feasible. Note this requires a careful audit of the
   smem.reduce union -- the current implementation reuses
   `smem.b_dequant` as the epilogue output buffer, so a Path 2
   refactor must either keep a small dedicated reduce buffer or
   share with `smem.a/b`.
4. **TMEM has its own bandwidth port.** r2t doesn't compete with TMA
   loads or SMEM-to-RMEM reads on L1TEX (NCU L1/TEX cache throughput
   is 56 % today; TMEM is independent).
5. **The 2-CTA cluster path becomes viable.** Workbook B.32
   documented a producer/consumer mbar pipeline blocker for
   `cta_group::2` in the SS-mode kernel. CUTLASS only offers TS for
   2-CTA mixed input, which is consistent: TS sidesteps the cluster-
   shared SMEM-pipeline requirement.

### Concrete costs

1. **TMEM budget grows.** The accumulator uses 128 cols today. The
   dequant'd A staging needs another ~64-128 cols depending on
   layout (one ping-pong slot per K-iter is one tile of BlockM ×
   BlockK bf16). Total ~192-256 cols out of 512. Plenty of headroom.
2. **The TiledMma + WS pipeline structure changes.** Today humming
   has a 2-stage WS (producer + consumer). CUTLASS's TS path uses
   **3 pipelines**:
   - `Load2Transform`: TMA G2S into SMEM
   - `Transform2Mma`: dequant warp r2t's into TMEM
   - `Mma2Accum`: MMA writes to accumulator TMEM
   Each pipeline has its own buffer count. The math-warp role splits
   into "transform warps" and "MMA warps".
3. **The transform warp's `r2t` target layout must match what
   `tcgen05.mma` TS-mode expects.** Per
   `cute/arch/mma_sm100_umma.hpp:178`: `static_assert(a_major ==
   UMMA::Major::K, "SM100_MMA_F16BF16 A from TMEM can't be
   transposed")`. K-major in TMEM is the only option. Humming's
   weight layout (sectionised, K-stride within each section) already
   matches this; the r2t partition needs to write the same physical
   layout.

## How CUTLASS structures this

`sm100_make_trivial_mixed_input_tiled_mma` (`sm100_common.inl:584`):

```cpp
if constexpr (UmmaMajorA == Major::K && !MixedInputSmemSchedule) {
  return SM100_MMA_F16BF16_TS<...>{};      // ← DEFAULT
} else {
  return SM100_MMA_F16BF16_SS<...>{};      // ← FALLBACK
}
```

For 2-CTA mode (same file, line 621-631), SS isn't offered for mixed
input at all -- only `SM100_MMA_F16BF16_2x1SM_TS`.

### Pipeline + TMEM layout

`sm100_mma_warpspecialized_mixed_input.hpp`:

```cpp
// line 1106 + 1138:
fragment_compute.data() = accumulators.data().get()
                        + find_tmem_tensor_col_offset(accumulators);
// line 1108:
auto r2t_tiled_copy = make_tmem_copy(ComputeCopyAtomA, fragment_compute(_,_,_,0));
```

So the dequant'd A and the accumulator share one TMEM allocation
(`accumulators`), with A at a col offset within it. `make_tmem_copy`
generates `tcgen05.st`-based r2t with the partition that matches the
TS-mode A descriptor.

### Verification points (read these for the refactor)

* `cute/arch/mma_sm100_umma.hpp` lines 171-209 -- `SM100_MMA_F16BF16_TS`
  PTX (the TS-mode tcgen05.mma issue).
* `cute/arch/mma_sm100_umma.hpp` line 191 -- the SS PTX, for compare:
  `[%0], %1, %2, %3, ...` vs TS's `[%0], [%1], %2, %3, ...`. The
  extra `[]` on operand 1 is what makes it a TMEM address.
* `cute/atom/copy_traits_sm100.hpp` -- `SM100_TMEM_STORE_*` copy
  traits, the underlying r2t copy atom.
* `cutlass/gemm/collective/sm100_mma_warpspecialized_mixed_input.hpp`
  lines 1080-1146 -- the `setup_copy_ops` template that selects
  between r2s (SS) and r2t (TS) based on whether the compute
  destination is in TMEM or SMEM.
* `cutlass/gemm/dispatch_policy.hpp:1151` --
  `MainloopSm100TmaUmmaWarpSpecializedMixedInput` with its
  Load2Transform / Transform2Mma stage counts.

## Microbench evidence

An r2s-vs-r2t microbench isolates the per-K-iter "move
4 KiB of dequant'd B from RMEM to staging area" primitive. B300
sm_103a, 10000 iters:

```
Path 1  (r2s + bar.sync, 16 stores/thread):   2052 cyc/iter
Path 1' (r2s NO bar, store cost only):        2048 cyc/iter
Path 2a (r2t 8-warp + fence):                   43 cyc/iter
Path 2b (r2t 2-warp + fence + bar.sync):        54 cyc/iter
```

* The bar.sync is ~4 cyc. The store cost is ~2048 cyc.
* r2t is ~50× cheaper per iter than r2s scatter.
* The "barrier" stall NCU attributes (~8 % of issued inst in the
  full kernel) isn't a barrier cost -- it's warps idling at the bar
  because everyone finishes the scatter together.

## Scope of the refactor

Touches:

* `humming/include/humming/utils/ptx/tcgen05.cuh` -- add `tcgen05_st_32x32b_x32`
  and `tcgen05_mma_ts_bf16` wrappers (the TS-mode mma_ss analogs).
* `humming/include/humming/mma/tcgen05_mma.cuh` -- rewrite the
  TCGEN05 class so `transform_b` does r2t to TMEM instead of
  preparing RMEM regs, `run` issues TS-mode mma, and remove the
  per-K-iter scatter + bar.sync.
* `humming/include/humming/utils/storage.cuh` -- drop `b_dequant`,
  grow the tcgen05_tmem allocation to hold both accumulator and
  dequant'd A (`alloc<256>` instead of `alloc<128>`).
* `humming/include/humming/kernel/humming_ws.cuh` -- split the math-
  side warps into "transform" and "mma" roles, with the new
  Transform2Mma pipeline (mbar between them).
* `humming/include/humming/memory/s2r_pipeline.cuh` and adjacent --
  the transform warp's r2t partition replaces the current "regs_b
  + scatter" plumbing.
* `humming/include/humming/epilogue/pipeline.cuh` -- the epilogue's
  t2r path may need to read from a different TMEM col range than
  today (accumulator now starts after the dequant'd A region).

Rough size estimate: 600-800 LoC of changes. The CUTLASS file
`sm100_mma_warpspecialized_mixed_input.hpp` (1296 lines) is the
working reference -- our refactor can crib the pipeline structure
and copy atom selection directly.

## Why we deferred

The current SS-mode implementation is working, correct across all
dtype combos in `test_tcgen05_dtypes.py`, and **1.2-1.55× faster
than humming's mma.sync path** at M >= 128 on realistic Llama-3
shapes (workbook B.35). Shipping the SS path lets the rest of the
project consume the tcgen05 work today; the Path 2 refactor is
opt-in future perf.

When the time comes to land Path 2, this document + the CUTLASS
references above + the microbench should make the refactor much
faster than starting from scratch.
