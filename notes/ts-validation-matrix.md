# TS-mode tcgen05 — validation matrix & coverage report

Track: SHIPPED-PATH GAP CLOSURE + VALIDATION BACKBONE (branch `expand/validation`).
Env: B300 SXM6 AC, GPU 3, sm_103a, torch 2.13+cu130. All numbers below measured on this
checkout of `integration/round3`.

Deliverables:
- `tests/test_tcgen05_ts_edge.py` — Milestone 1, shipped-uint4-path gap closure (50 tests).
- `tests/test_tcgen05_ts_matrix.py` — Milestone 2, per-model realistic matrix (61 pass + 8 xfail).
- this report — Milestone 3, cell-by-cell coverage + the caveats that must stay documented.

Reference methodology (workbook / track-B convention, unchanged): reference = bf16-rounded-weight
dequant GEMM (`weight_ref.to(bf16).float()` then fp32 matmul). Tolerance K-scaled: `rtol=1e-2`;
`atol = 0.5 / 1.5 / 2.0` for `K <= 1024 / <= 2048 / larger`. **Verified sufficient to K=28672**:
at Llama-70B-down (K=28672) the max abs error is 8.0, which is exactly ONE bf16 ulp of the
largest output (ref=-1104; ulp near 1024 = 8) — `rtol=1e-2` gives 0 element-wise violations. This
is NOT a loosening; the tail is bf16 rounding of the output, not accumulation error (mean|err|
~3e-3). Second reference mode = SS/mma.sync on identical codes, `_assert_ts_dispatched` guards
against a silent SS fallback masquerading as a pass.

## What ships today (gate, POC-confirmed)

`Sm100Heuristics.supports_tcgen05_ts` + `mma/tcgen05_ts_mma.cuh` static_asserts pin the TS opt-in
(`mma_type="tcgen05"`, used at EVERY M — no per-M crossover): bf16 A × uint4 B, `bs_dtype==bf16`,
group scale `gs >= BlockK=64`, int-or-no zero point, `N%128==0`, `K%64==0`, `num_experts==0`,
BlockN=128, WarpN=32, WarpM=BlockM∈{64,128}, WarpK=BlockK=64. Stream-K pinned False,
raster_group_m=1 (`sm100.py:225-226`). Heuristic picks `block_m = 128 if M>=128 else 64`,
num_stages=4, warp-spec + TMA.

## Coverage matrix

Legend: **C** covered-green-today · **G** gap now closed by this track · **B[track]**
blocked-on-feature (owning track) · **N/A**.

### Scale-type × zero-point × gemm-mode (weight = uint4, bf16 A)

| scale type    | zp   | dense                    | MoE-contig   | MoE-masked   |
| ---           | ---  | ---                      | ---          | ---          |
| group gs=128  | int  | **C** (ts/e2e/edge/matrix) | B[moe]     | B[moe]       |
| group gs=128  | none | **C** (e2e/edge/matrix)  | B[moe]       | B[moe]       |
| group gs=64   | int  | **G→C** (edge/matrix)    | B[moe]       | B[moe]       |
| group gs=64   | none | **G→C** (edge)           | B[moe]       | B[moe]       |
| group gs=32   | int  | B[scale]                 | B[moe]       | B[moe]       |
| channelwise   | -    | B[scale]                 | B[moe]       | B[moe]       |
| e8m0 mx       | none | B[scale]                 | B[moe]       | B[moe]       |
| group gs=128  | fp   | B[scale]                 | B[moe]       | B[moe]       |

### Weight-dtype (group gs=128 int-zp, dense, bf16 A)

| dtype                 | status                      |
| ---                   | ---                         |
| uint4                 | **C**                       |
| uint8                 | B[weight-dtypes] (packer ✓, kernel dequant `0x4300` trick breaks ≥128) |
| uint2                 | B[weight-dtypes]            |
| fp4 / fp6 / fp8       | B[weight-dtypes]            |
| uint3 / uint5 / uint7 | N/A (32%bits!=0 — research)  |

### M coverage (uint4 gs128 dense) — TS runs at EVERY M

| M           | before this track | now                         |
| ---         | ---               | ---                         |
| 1, 2, 7     | **GAP** (decode)  | **G→C** (edge + matrix sweep) |
| 16, 17, 63  | partial           | **C** (edge)                |
| 64, 65      | partial           | **C** (edge, block_m boundary) |
| 128, 129    | 128 only          | **C** incl. 129 ragged tail |
| 130, 191    | —                 | **C** (edge partial-tile)   |
| 512, 2048   | **C**             | **C** (matrix sweep)        |

### Shape-family coverage (uint4 gs128 dense, per-model, real N/K)

All green in `test_tcgen05_ts_matrix.py` at M∈{1,256} + full sweep on the two canonical shapes:

| model            | projections covered                    |
| ---              | ---                                    |
| Llama-3-70B      | gate/up, down (fat-K K=28672), qkv, o  |
| Llama-3-8B       | gate/up, down, qkv                     |
| Qwen2.5-7B       | gate/up, down                          |
| Qwen3-MoE/Mixtral| per-expert gate/up (dense proxy)       |
| DeepSeek-V3      | MLA qkv proj (fat-K), MoE expert (dense proxy) |

MoE **mode** (grouped_contiguous / grouped_masked) is B[moe]; the per-expert GEMM *shape* runs
today as a dense proxy. Full-sweep fat-N (llama70b gate) + fat-K (llama70b down) M∈{1,2,7,16,17,
64,65,128,129,512,2048} all green.

### Adversarial edge shapes (uint4 dense) — all `test_tcgen05_ts_edge.py`, green

| edge case                        | result                                   |
| ---                              | ---                                      |
| single N-tile (N=128)            | **C**                                    |
| odd N-tile count (N=384, 640)    | **C** (not %256)                         |
| N=256 exact                      | **C**                                    |
| K=64 minimum (1 stage)           | **C**                                    |
| K %64 not %128 (192, 320)        | **C**                                    |
| single-tile-per-CTA (64×128×64)  | **C** (smallest grid)                    |
| ragged M-tail (65,129,130,191)   | **C** (+ last-row finite/non-zero check) |
| N %128 != 0 (192, 320)           | **C** rejected loudly at TS-legal gate   |
| N=129 odd                        | **C** rejected loudly (pack %32 guard)   |
| K %64 != 0 (96, 160, 224)        | **C** rejected loudly at TS-legal gate   |

## Caveats — documented, must stay pinned

1. **Stream-K pinned False** (`sm100.py:225`). The TS epilogue (TMA-C store) has no cross-CTA
   partial-K reduction, so enabling stream-K corrupts any K-split output (e2e regression test
   `test_ss_tcgen05_default_path_stream_k_regression` shows mean|err|~0.03 with it on).
   **Correctness question resolved:** is stream-K-off a *correctness* limit for low-M fat-K? No —
   it is perf-left-on-the-table only. `test_matrix_m_sweep_fat_k[m1]` runs Llama-70B-down
   (N=8192, K=28672) at M=1 with stream-K OFF and matches the dequant reference (max|err| = 1 ulp,
   0 tol violations). At low M + huge K stream-K is exactly what *would* help occupancy, so the
   caveat is a perf limit, not a correctness bug, as long as the pin stays. A TS stream-K epilogue
   (cross-CTA D reduction before TMA-C) is a separate `large` work item. Keep the pin + the
   regression test.
2. **raster_group_m=1** (`sm100.py:226`). L2-locality perf only; no correctness impact. No matrix
   cell depends on it.
3. **BlockK=64 only** (`tcgen05_ts_mma.cuh:86`). All cells run at BlockK=64. BlockK>64 is a `large`
   follow-up (activation SMEM descriptor `kSwizzleSizeK >= WarpK`) and interacts with the future
   sub-stage scale splits (gs=32, e8m0-32) — land those at BlockK=64 first.

## Merge-readiness signal

The 8 `xfail(strict=False)` gate-level hooks in `test_tcgen05_ts_matrix.py`
(`test_matrix_future_*`) assert `supports_tcgen05_ts` accepts the future meta. They xfail today and
flip to **XPASS** the moment the owning track lifts the gate:

- weight-dtypes track: `future_weight_dtype[uint8]`, `[uint2]`.
- scale track: `future_gs32`, `future_channelwise`, `future_e8m0_scale`, `future_fp_zero_point`.
- moe track: `future_moe[moe-8]`, `future_moe[moe-256]`.

`CUDA_VISIBLE_DEVICES=3 pytest tests/test_tcgen05_ts_matrix.py -rX` prints the XPASS set — that is
the acceptance backbone for synthesis: a track is "matrix-ready" when its hook XPASSes AND the
corresponding real-GEMM cells (currently `xfail`/absent) are un-xfailed and green.

## Run commands

```
CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m pytest tests/test_tcgen05_ts_edge.py -q     # 50 pass
CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m pytest tests/test_tcgen05_ts_matrix.py -q   # 61 pass + 8 xfail
CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m pytest tests/test_tcgen05_ts.py tests/test_tcgen05_ts_e2e.py -q  # 24 pass (shipped baseline)
```
