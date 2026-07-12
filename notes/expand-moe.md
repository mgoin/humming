# TS-mode tcgen05 MoE / grouped GEMM — GA notes (track: expand/moe)

Base: integration/round3. Goal: ship GROUPED_CONTIGUOUS + GROUPED_MASKED
for the TS kernel at the shipped uint4 W4A16 dtype. Per the POC this is
~90% gate + validation, not kernel work — confirmed.

## M1 — gate flip (commit 03dbaba)

`humming/tune/sm100.py`:
- `supports_tcgen05_ts`: dropped `if meta.num_experts: return False`.
  Grouping lives above the MMA; the TS mainloop/drain are
  grouping-agnostic (POC-proven). Diff kept to exactly this clause so
  the weight-dtypes track's uint4-only-clause drop stays a clean 2-way
  merge.
- `get_config`: TS branch gate widened from `gemm_type == DENSE` to
  `gemm_type in {DENSE, GROUPED_CONTIGUOUS, GROUPED_MASKED}`, with a
  grouped-aware block_m:
    - DENSE: `128 if shape_m >= 128 else 64` (unchanged).
    - GROUPED: shape_m is the total/padded token count over all experts,
      but the scheduler tiles the MMA per-expert, so
      `tokens_per_expert = shape_m // num_experts` drives the choice:
      `128 if tokens_per_expert >= 128 else 64`. Fine-grained experts
      (few tokens each) get BlockM=64 to cut per-expert MMA-row waste.
  Dense TS config byte-identical (verified block_m 64/128/128 at
  shape_m 64/128/512).

## M2 — correctness suite (commit d936e38)

`tests/test_tcgen05_ts_moe.py`, 32 cells, all pass:
- `test_ts_moe_matrix`: {contiguous,masked} x zp{off,on} x
  block_m{64,128} x use_tma{T,F}, E=8 (16 cells). use_tma=False = the
  legacy-C epilogue (write_legacy row masking) — the path the POC left
  thin.
- `test_ts_moe_experts`: {contiguous,masked} x E{4,8,128,256},
  block_m=64, zp on, TMA-C (8 cells).
- `test_ts_moe_masked_zeroing`: masked rows past each expert's token
  count are exactly 0, on TMA-C and legacy-C (2 cells).
- `test_ts_moe_dispatch`: grouped TCGEN05 meta selects the TS config
  with the grouped-aware block_m; no silent fallback (6 cells, no GPU).
- Reference = per-expert dequant matmul vs bf16-rounded weight — the
  same ground truth test_moe.py::test_grouped_gemm validates its
  mma.sync grouped path against, so matching it pins TS to the proven
  scatter. (A direct TS-vs-mma.sync launch cross-check was dropped: the
  mma.sync grouped path uses a different scale-fold/centering
  convention (to_apply_on_c) that made the harness brittle without
  adding coverage the tighter dequant reference lacks.)
- Tolerance: `assert_close(rtol=1e-2, atol=0.5)`. Raw max|err| reaches
  ~1.0 at K=1024 (1 bf16 ulp on the largest outputs), covered by the
  rtol term; mean|err| ~1e-5. Same bound as the dense TS suite.
- Non-regression: test_tcgen05_ts.py + test_tcgen05_ts_packing.py = 53
  passed.

Packing note: MoE weight/scale/zp go through
`prepare_humming_weight` / `prepare_humming_weight_scale` /
`prepare_humming_zero_point(..., packed=False)` with
`use_tcgen05_ts=True` (all handle the leading expert dim). `packed=True`
would unpack — wrong for generate_random_weight's already-unpacked
codes. `use_tma_bzp` must be False (a TMA desc over the packed TS zp
stream fails `make_tma_desc`).

## M3 — HW verification (compute-sanitizer, failing-est config)

Config: grouped_masked, 256 fine-grained experts (top-8), block_m=128
(max padding waste), zp on, TMA-C. Output max=1.0 mean=0.00000
finite=True; masked rows zeroed.

- **memcheck: CLEAN.** Zero out-of-bounds global/shared accesses. The
  34 reported "errors" are all
  `CUDA_ERROR_INVALID_VALUE ... cuGetProcAddress_v2` — a benign CUDA-13
  driver-probe artifact, not a memory access. So the per-expert scatter,
  masked row-zeroing, and per-expert TMA-C tensor-map update are
  memory-clean.
- **synccheck: pre-existing WS artifact, not MoE-specific.** Reports
  "Barrier error detected. Divergent thread(s) in block" (block 6). But
  the *shipped* DENSE TS kernel trips the identical error class
  (dense: 11360 errors, thread 64; masked grouped: 13568 errors,
  thread 32). Warp-specialized kernels drive producer/consumer warps to
  different named barriers by design, which synccheck flags. Both
  configs produce correct finite output without the sanitizer. Grouping
  introduces NO new synchronization divergence beyond the dense TS
  baseline.

## M4 — realistic-shape e2e + bench (benchmarks/bench_ts_moe.py)

TS grouped (heuristic BlockM) vs mma.sync grouped, uint4 W4A16 gs=128,
zp on, 512 tokens, B300. Every point correctness-verified (Y) against
the per-expert dequant reference before timing. TS/mma < 1 means TS
faster. contiguous and masked track within noise; contiguous shown:

  shape                   E    tk  bm   TS us   mma us  TS/mma
  Qwen3 gate/up (1536x4096) 8   2  128    47.9    61.9   0.77
  Qwen3 gate/up            128  8   64   470.3   417.6   1.13
  Qwen3 down (4096x1536)     8  2  128    58.0    70.4   0.82
  Qwen3 down               128  8   64   486.0   380.4   1.28
  DeepSeek gate/up (2048x7168) 8 2 128   152.1   227.8   0.67
  DeepSeek gate/up         256  8   64  2029.9  1639.9   1.24
  DeepSeek down (7168x2048)  8  2  128   119.7   138.7   0.86
  DeepSeek down            256  8   64  2146.3  1664.9   1.29
  Mixtral gate/up (14336x4096) 8 2 128  431.8   514.8   0.84
  Mixtral down (4096x14336)  8  2  128   436.2   707.8   0.62

Story: **TS wins at coarse experts** (E=8, ~128 tokens/expert fills the
BlockM=128 tile: 0.62-0.86x, i.e. up to 1.6x faster) and **loses at
fine-grained experts** (E=128/256, BlockM=64: 1.13-1.31x slower). The
crossover is exactly the token-tile granularity flag below.

## M5 research flag — token-tile granularity for fine-grained MoE

(Documented, not solved — see moe-grouped-gemm.md §Hardest.) TS pins the
token tile = BlockShape::M in {64,128} (MMA-N, forced by
WarpShape::N==32 / M_WARPS==1; static_assert tcgen05_ts_mma.cuh:89).
mma.sync grouped runs BlockM as small as 16. With load-balanced MoE the
per-expert token count is `num_tokens * top_k / num_experts`, so:

  - DeepSeek 256-expert, 512 tok, top-8 => ~16 tokens/expert. TS BlockM=64
    => ceil(16/64)=1 tile of 64 rows, **16 real => 4x MMA-row waste**;
    mma.sync BlockM=16 => 0 waste. Measured wall-clock 1.24-1.29x slower.
  - Qwen3 128-expert, 512 tok, top-8 => ~32 tokens/expert. TS BlockM=64
    => **2x MMA-row waste**. Measured 1.13-1.31x slower.

The wall-clock gap (1.1-1.3x) is smaller than the raw MMA-row waste
(2-4x) because scale/zp loads and the epilogue amortize across the
padded rows; the MMA itself is not the sole cost. Still throughput, not
correctness — masking keeps every fine-grained point bit-correct (Y).

Concrete next step (option b, deferred): relax the BlockM==64||128
static_assert to admit BlockM=32. The TS drain already loops in
kBlockM/32 chunks (tmem_ts_drain.cuh), so a 32-row token tile is
plausibly near-free on the epilogue side and would halve the waste at
16-32 tokens/expert. Needs the MMA-N=32 atom to hold with WarpShape::N
unchanged; verify before committing. Option (c) two-experts-per-CTA
packing is larger. Recommendation: ship BlockM=64 for fine-grained now
(correct, and the TS coarse-expert win + the dtype-breadth roadmap are
the higher-value items); pick up BlockM=32 as a focused perf follow-up.
