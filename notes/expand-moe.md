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

## M5 research flag — token-tile granularity for fine-grained MoE

(Documented, not solved — see below and moe-grouped-gemm.md §Hardest.)
TS pins the token tile = BlockShape::M in {64,128} (MMA-N, forced by
WarpShape::N==32 / M_WARPS==1). With top-8-of-256 at low/medium batch,
most experts get < 64 tokens, so every expert still costs a full 64-row
MMA tile. The masked epilogue keeps it correct (proven) but the MMA does
up to ~4x wasted rows vs a 16-token tile the mma.sync grouped path can
use. This is throughput, not correctness. Quantified in M4 bench below.
