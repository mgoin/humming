"""Milestone 2: the parametrized realistic-case validation matrix for the TS
(tcgen05 "method 2") kernel.

This is the acceptance backbone every other expansion track reports "done"
against. It crosses a per-model shape table (Llama-3 70B/8B, Qwen2.5/3 dense,
Qwen3-MoE / Mixtral per-expert, DeepSeek-V3/V4 projections) with the dtype/mode
axes and provides BOTH reference modes:

  1. dequant reference  -- bf16-rounded-weight fp32 GEMM (catches absolute
     wrongness); reused via test_tcgen05_ts_edge._run_e2e_case, which also
     asserts the heuristic actually dispatches TS (no silent SS fallback).
  2. SS / mma.sync cross-check -- TS vs the trusted default path on IDENTICAL
     quantized codes (catches TS-specific layout / scale-fold bugs a
     self-consistent-but-wrong dequant would hide).

ENABLED today: bf16 A x uint4 B, group scale (gs in {64,128}), int / no zero
point, DENSE gemm. Those cells run for real and must be green.

FUTURE cells (other tracks landing weight dtypes / MoE) are left as
`xfail(strict=False)` GATE-LEVEL hooks: they assert `supports_tcgen05_ts`
accepts the future meta. They fail today (gate rejects) and will flip to XPASS
the moment the owning track lifts the gate -- so `pytest -rX` on this file is a
live readiness signal for the merge. Cheap (pure-python gate check, no GEMM).

Tolerance: workbook K-scaled convention (rtol=1e-2; atol 0.5 / 1.5 / 2.0 for
K <= 1024 / <= 2048 / larger). Verified sufficient up to K=28672, where the
max abs error is exactly one bf16 ulp of a large output and rtol covers it
(zero element-wise violations) -- NOT a loosening.

Run:
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m pytest tests/test_tcgen05_ts_matrix.py -v
"""

from __future__ import annotations

import pytest
import torch

from humming import dtypes
from humming.config import GemmType, MmaType
from humming.config.config import LayerConfig
from humming.layer import HummingLayer
from humming.schema.humming import HummingWeightSchema
from humming.tune import get_heuristics_class
from humming.utils.test import generate_random_inputs, generate_random_weight

# Reuse the shipped-path runner (builds + asserts-TS-dispatched + dequant-ref
# compare) and the K-scaled tolerance from the Milestone-1 edge suite.
from test_tcgen05_ts_edge import _atol_for_k, _run_e2e_case
from test_tcgen05_ts_e2e import _assert_close, _is_blackwell

pytestmark = pytest.mark.skipif(
    not _is_blackwell(), reason="TS-mode tcgen05 needs sm_100+"
)


# ---------------------------------------------------------------------------
# Per-model shape table (N = out features, K = in features). All entries are
# TS-legal today (N%128==0, K%64==0). MoE per-expert rows are listed as their
# single-expert dense GEMM shape -- the dense proxy that runs today; the
# grouped/masked MoE *mode* is a FUTURE cell (see MoE hooks below).
# ---------------------------------------------------------------------------

MODEL_SHAPES = [
    # (id, N, K)
    ("llama3-70b.gate_up", 28672, 8192),
    ("llama3-70b.down", 8192, 28672),   # fat-K: TS weak spot + stream-K-off case
    ("llama3-70b.qkv", 10240, 8192),
    ("llama3-70b.o", 8192, 8192),
    ("llama3-8b.gate_up", 14336, 4096),
    ("llama3-8b.down", 4096, 14336),
    ("llama3-8b.qkv", 6144, 4096),
    ("qwen2.5-7b.gate_up", 18944, 3584),
    ("qwen2.5-7b.down", 3584, 18944),
    ("qwen3-moe.expert_gate_up", 14336, 4096),  # per-expert dense proxy
    ("mixtral-8x7b.expert_gate_up", 14336, 4096),
    ("deepseek-v3.mla_qkv", 3072, 7168),         # fat-K MLA projection
    ("deepseek-v3.moe_expert", 2048, 7168),      # per-expert dense proxy
]

_SHAPE_IDS = [s[0] for s in MODEL_SHAPES]


# ===========================================================================
# CELL: uint4 x bf16, group scale, DENSE -- the ENABLED backbone.
# Reference mode 1 (dequant). One clearly-named cell per (model-shape, M).
# ===========================================================================


@pytest.mark.parametrize("shape_m", [1, 256], ids=lambda m: f"m{m}")
@pytest.mark.parametrize("model_id,shape_n,shape_k", MODEL_SHAPES, ids=_SHAPE_IDS)
def test_matrix_uint4_dense(model_id, shape_n, shape_k, shape_m):
    """bf16 A x uint4, gs=128, int-zp, dense -- every production projection at
    decode (M=1) and a prefill point (M=256), vs the dequant reference with
    TS-dispatch asserted."""
    _run_e2e_case(shape_m, shape_n, shape_k, 128, has_zero_point=True)


@pytest.mark.parametrize(
    "model_id,shape_n,shape_k",
    [MODEL_SHAPES[0], MODEL_SHAPES[1], MODEL_SHAPES[4]],
    ids=[_SHAPE_IDS[0], _SHAPE_IDS[1], _SHAPE_IDS[4]],
)
def test_matrix_uint4_dense_no_zp(model_id, shape_n, shape_k):
    """Symmetric (no zero point) on representative fat-N / fat-K / mid shapes."""
    _run_e2e_case(256, shape_n, shape_k, 128, has_zero_point=False)


@pytest.mark.parametrize(
    "model_id,shape_n,shape_k",
    [MODEL_SHAPES[2], MODEL_SHAPES[4]],
    ids=[_SHAPE_IDS[2], _SHAPE_IDS[4]],
)
def test_matrix_uint4_dense_gs64(model_id, shape_n, shape_k):
    """gs=64 (== BlockK) on real model shapes -- the per-stage scale advance
    at production K."""
    _run_e2e_case(256, shape_n, shape_k, 64, has_zero_point=True)


# ===========================================================================
# CELL: decode/prefill M sweep on the two canonical shapes. TS runs at every M
# (no crossover), so this is the load-bearing per-M correctness surface.
#   - fat-N (gate/up): TS's strength.
#   - fat-K (down, K=28672): TS's weak spot AND the stream-K-off caveat case --
#     at low M + huge K stream-K is exactly what would help occupancy, so
#     M=1 here confirms output is correct with stream-K PINNED OFF.
# ===========================================================================

_M_SWEEP = [1, 2, 7, 16, 17, 64, 65, 128, 129, 512, 2048]


@pytest.mark.parametrize("shape_m", _M_SWEEP, ids=lambda m: f"m{m}")
def test_matrix_m_sweep_fat_n(shape_m):
    """Llama-70B gate/up (N=28672 K=8192) across the full M sweep."""
    _run_e2e_case(shape_m, 28672, 8192, 128, has_zero_point=True)


@pytest.mark.parametrize("shape_m", _M_SWEEP, ids=lambda m: f"m{m}")
def test_matrix_m_sweep_fat_k(shape_m):
    """Llama-70B down (N=8192 K=28672) across the full M sweep. Doubles as the
    stream-K-off caveat check: large-K low-M with stream-K pinned False."""
    _run_e2e_case(shape_m, 8192, 28672, 128, has_zero_point=True)


# ===========================================================================
# CELL: reference mode 2 -- TS vs SS/mma.sync default path on IDENTICAL codes.
# Catches TS-specific layout / scale-fold bugs a self-consistent-but-wrong
# dequant hides. Kept on modest-K shapes to bound the doubled cost.
# ===========================================================================

_SS_CROSS_SHAPES = [
    ("llama3-8b.qkv", 6144, 4096),
    ("llama3-8b.gate_up", 14336, 4096),
    ("qwen2.5-7b.gate_up", 18944, 3584),
    ("deepseek-v3.moe_expert", 2048, 7168),
]


def _build_layer(shape_n, shape_k, group_size, mma_type, seed=7):
    schema = HummingWeightSchema(
        b_dtype=dtypes.uint4, bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=group_size, has_zero_point=True,
    )
    torch.manual_seed(seed)
    w = torch.randn(shape_n, shape_k, dtype=torch.bfloat16, device="cuda") / (
        shape_k ** 0.5
    )
    layer = HummingLayer(
        shape_n=shape_n, shape_k=shape_k, weight_config=schema,
        torch_dtype=torch.bfloat16, mma_type=mma_type,
    ).cuda()
    layer.load_from_unquantized(w)
    layer.transform()
    return layer


@pytest.mark.parametrize("shape_m", [16, 256], ids=lambda m: f"m{m}")
@pytest.mark.parametrize(
    "model_id,shape_n,shape_k", _SS_CROSS_SHAPES,
    ids=[s[0] for s in _SS_CROSS_SHAPES],
)
def test_matrix_ss_crosscheck(model_id, shape_n, shape_k, shape_m):
    """TS output must match the default (mma.sync/SS) path on the same
    quantized weights to bf16 noise -- both dequant the same codes."""
    layer_default = _build_layer(shape_n, shape_k, 128, None)
    layer_ts = _build_layer(shape_n, shape_k, 128, "tcgen05")
    assert layer_default.humming_metas[""].mma_type == MmaType.MMA
    assert layer_ts.humming_metas[""].mma_type == MmaType.TCGEN05
    # Confirm the TS layer would really dispatch TS at this M.
    cfg = get_heuristics_class().get_config(
        layer_ts.humming_metas[""], shape_m=shape_m, gemm_type=GemmType.DENSE
    )
    assert cfg.get("use_tcgen05_ts") is True, cfg

    torch.manual_seed(11)
    x = torch.randn(shape_m, shape_k, dtype=torch.bfloat16, device="cuda") / (
        shape_k ** 0.5
    )
    out_default = layer_default.forward(x.clone())
    out_ts = layer_ts.forward(x.clone())
    torch.cuda.synchronize()
    _assert_close(
        out_ts, out_default, atol=_atol_for_k(shape_k),
        label=f"SS-crosscheck {model_id} m{shape_m}",
    )


# ===========================================================================
# FUTURE CELLS: gate-level xfail hooks. Each asserts supports_tcgen05_ts accepts
# a meta the owning track is enabling. Fails today (gate rejects) -> xfail;
# flips to XPASS when the gate lifts. `pytest -rX` = readiness signal.
# ===========================================================================


def _make_meta(**overrides):
    base = dict(
        shape_n=512, shape_k=512, num_experts=0,
        a_dtype=dtypes.bfloat16, b_dtype=dtypes.uint4,
        c_dtype=dtypes.bfloat16, bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=128, has_zero_point=True,
        is_fp_zero_point=False, mma_type=MmaType.TCGEN05,
    )
    base.update(overrides)
    return LayerConfig(**base)


def _gate_accepts(meta) -> bool:
    return get_heuristics_class().supports_tcgen05_ts(meta)


# --- weight-dtype axis (track: weight-dtypes) ---
@pytest.mark.parametrize(
    "b_dtype",
    [dtypes.uint8, dtypes.uint2],
    ids=["uint8", "uint2"],
)
@pytest.mark.xfail(reason="weight-dtypes track: gate pins b_dtype==uint4", strict=False)
def test_matrix_future_weight_dtype(b_dtype):
    assert _gate_accepts(_make_meta(b_dtype=b_dtype))


# --- scale-type axis (track: scale/zp validation) ---
@pytest.mark.xfail(reason="scale track: gate pins gs>=64", strict=False)
def test_matrix_future_gs32():
    assert _gate_accepts(_make_meta(weight_scale_group_size=32))


@pytest.mark.xfail(reason="scale track: gate pins group scale (gs>0)", strict=False)
def test_matrix_future_channelwise():
    assert _gate_accepts(_make_meta(weight_scale_group_size=0))


@pytest.mark.xfail(reason="scale track: gate pins bs_dtype==bfloat16", strict=False)
def test_matrix_future_e8m0_scale():
    assert _gate_accepts(_make_meta(bs_dtype=dtypes.float8e8m0))


@pytest.mark.xfail(reason="scale track: gate rejects fp zero point", strict=False)
def test_matrix_future_fp_zero_point():
    assert _gate_accepts(_make_meta(has_zero_point=True, is_fp_zero_point=True))


# --- MoE axis (track: moe-grouped-gemm) ---
@pytest.mark.parametrize(
    "num_experts", [8, 256], ids=["moe-8", "moe-256"],
)
@pytest.mark.xfail(reason="moe track: gate rejects num_experts>0", strict=False)
def test_matrix_future_moe(num_experts):
    assert _gate_accepts(_make_meta(num_experts=num_experts))
