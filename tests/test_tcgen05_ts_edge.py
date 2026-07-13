"""Milestone 1: correctness gap-closure for the ALREADY-SHIPPED uint4 TS path.

The shipped TS path is the HummingLayer opt-in (`mma_type="tcgen05"`): once a
layer is TS-legal (`supports_tcgen05_ts`) the heuristic dispatches the TS kernel
at EVERY M with block_m = 128 if M >= 128 else 64, warp-spec + TMA, num_stages=4,
stream-K OFF (tune/sm100.py:200-227). test_tcgen05_ts_e2e.py already covers
gs=128 dense at M in {16,128,256,512}. This file closes the remaining gaps in
that SAME shipped path -- they are real correctness holes, not new features:

  * group_size=64 (== BlockK): only gs=128 was ever exercised. gs==BlockK
    advances the scale row EVERY stage (vs every 2 stages at gs=128); if the
    g2s scale-row index aliased at the boundary it would be wrong here only.
  * low / decode M in {1,2,7,16,17,63,64,65,129}: TS runs at every M, so
    decode-M correctness is load-bearing. The e2e suite only tested M >= 16;
    M in {1,2,7} was a genuine gap even for the shipped uint4 path.
  * adversarial edge shapes: N tails (single N-tile, odd N-tile count), K=64
    minimum, K multiples of 64 that are not 128, single-tile-per-CTA,
    grid-edge partial M-tiles (M=65, M=129 -> a 1-row 2nd tile).

Reference = bf16-rounded-weight dequant GEMM at the workbook K-scaled tol
(reuses test_tcgen05_ts_e2e._assert_close). Every case ASSERTS the heuristic
actually dispatches TS (`_assert_ts_dispatched`) so a silent SS fallback that
happened to be correct cannot pass. If a case FAILS, that is a finding -- the
tol is NOT loosened to paper over it.

Run:
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m pytest tests/test_tcgen05_ts_edge.py -v
"""

from __future__ import annotations

import pytest
import torch

# Reuse the shipped-path harness so this suite exercises the EXACT helpers the
# green e2e suite does (tests/ is on sys.path under pytest prepend import mode;
# no __init__.py, so a bare module import resolves to the sibling test file).
from test_tcgen05_ts_e2e import (
    _assert_close,
    _assert_ts_dispatched,
    _build_ts_layer,
    _is_blackwell,
)

from humming import dtypes
from humming.layer import HummingLayer
from humming.schema.humming import HummingWeightSchema
from humming.utils.test import generate_random_inputs, generate_random_weight

pytestmark = pytest.mark.skipif(
    not _is_blackwell(), reason="TS-mode tcgen05 needs sm_100+"
)


def _atol_for_k(shape_k: int) -> float:
    """K-scaled tolerance, identical convention to the shipped suite
    (test_tcgen05_ts_e2e.py:157, test_tcgen05_dtypes.py)."""
    if shape_k <= 1024:
        return 0.5
    if shape_k <= 2048:
        return 1.5
    return 2.0


def _run_e2e_case(shape_m, shape_n, shape_k, group_size, has_zero_point, seed=0xBEEF):
    """Drive the shipped HummingLayer TS opt-in path end to end and compare
    against the bf16-rounded dequant reference. Asserts TS is dispatched."""
    torch.manual_seed(seed)
    (weight_orig, weight_ref, _c, _s, _z, _g) = generate_random_weight(
        n=shape_n, k=shape_k, group_size=group_size,
        dtype=dtypes.uint4, scale_dtype=dtypes.bfloat16,
        has_zero_point=has_zero_point,
    )

    layer = _build_ts_layer(
        shape_n, shape_k, group_size, has_zero_point, weight_orig
    )
    _assert_ts_dispatched(layer, shape_m)

    _, inputs_ref, inputs, _ = generate_random_inputs(
        m=shape_m, k=shape_k, group_size=0, dtype=dtypes.bfloat16,
    )
    weight_ref_bf16 = weight_ref.to(torch.bfloat16).float()
    outputs_ref = inputs_ref.matmul(weight_ref_bf16.T).to(torch.bfloat16)
    torch.cuda.synchronize()

    outputs = layer.forward(inputs.clone())
    torch.cuda.synchronize()

    assert outputs.shape == (shape_m, shape_n)
    assert outputs.dtype == torch.bfloat16
    assert torch.isfinite(outputs).all()
    _assert_close(
        outputs, outputs_ref, atol=_atol_for_k(shape_k),
        label=(
            f"uint4 gs{group_size} zp={has_zero_point} "
            f"m{shape_m} n{shape_n} k{shape_k}"
        ),
    )
    return outputs


# ---------------------------------------------------------------------------
# GAP 1: group_size == 64 (== BlockK). Only gs=128 was tested before. gs==BlockK
# is the boundary case where the scale-row index advances every stage.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("has_zero_point", [True, False])
@pytest.mark.parametrize(
    "shape_m,shape_n,shape_k",
    [
        (16, 512, 512),
        (128, 512, 512),
        (256, 1024, 2048),
        (512, 256, 1024),
    ],
)
def test_ts_edge_gs64(shape_m, shape_n, shape_k, has_zero_point):
    _run_e2e_case(shape_m, shape_n, shape_k, 64, has_zero_point)


def test_ts_edge_gs64_vs_gs128_boundary():
    """Same shape, gs=64 and gs=128, both must hit the dequant reference --
    pins that the gs==BlockK per-stage scale advance did not alias."""
    _run_e2e_case(256, 512, 1024, 64, True, seed=0x64)
    _run_e2e_case(256, 512, 1024, 128, True, seed=0x64)


# ---------------------------------------------------------------------------
# GAP 2: low / decode M. TS runs at every M; M in {1,2,7} was never tested.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape_m", [1, 2, 7, 16, 17, 63, 64, 65, 129])
@pytest.mark.parametrize("has_zero_point", [True, False])
def test_ts_edge_low_m(shape_m, has_zero_point):
    # Small square-ish legal shape keeps it fast; K=512 -> tight atol.
    _run_e2e_case(shape_m, 256, 512, 128, has_zero_point)


@pytest.mark.parametrize("shape_m", [1, 2, 7, 65])
def test_ts_edge_low_m_gs64(shape_m):
    """Decode M crossed with the gs=64 gap -- the two Milestone-1 gaps
    together (block_m=64 path + per-stage scale advance)."""
    _run_e2e_case(shape_m, 256, 512, 64, True)


# ---------------------------------------------------------------------------
# GAP 3: adversarial N/K edge shapes (all TS-legal: N%128==0, K%64==0).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape_n",
    [
        128,   # single N-tile per CTA (1 MMA-N tile)
        256,   # exact 2 tiles
        384,   # odd tile count (3 tiles) -- not %256
        640,   # 5 tiles -- not %256
    ],
)
def test_ts_edge_n_tiling(shape_n):
    _run_e2e_case(128, shape_n, 512, 128, True)


@pytest.mark.parametrize(
    "shape_k",
    [
        64,    # K minimum (one stage, gs must be <= 64 here)
        128,
        192,   # 3 * 64, not %128
        320,   # 5 * 64, not %128
    ],
)
def test_ts_edge_k_tiling(shape_k):
    # gs must divide K and be >= 64; use gs=64 so every K here is legal.
    _run_e2e_case(128, 256, shape_k, 64, True)


def test_ts_edge_single_tile_per_cta():
    """M=64 N=128 K=64: exactly one M-tile, one N-tile, one K-stage --
    the smallest possible TS grid (1 output tile)."""
    _run_e2e_case(64, 128, 64, 64, True)


@pytest.mark.parametrize("shape_m", [65, 129, 130, 191])
def test_ts_edge_partial_m_tile(shape_m):
    """Grid-edge partial M-tiles: M=65 -> block_m=64 with a 1-row 2nd tile;
    M=129 -> block_m=128 with a 1-row 2nd tile; the ragged M-tail is where a
    predication-off-by-one would corrupt or NaN the last rows."""
    out = _run_e2e_case(shape_m, 256, 512, 128, True)
    # The ragged tail rows specifically must be finite and non-degenerate.
    assert torch.isfinite(out[-1]).all()
    assert out[-1].abs().sum() > 0


# ---------------------------------------------------------------------------
# Fail-closed guards: TS-ILLEGAL edge shapes must be REJECTED at transform,
# not silently mispacked / silently fall back to a correct-but-not-TS path.
# ---------------------------------------------------------------------------


def _build_illegal(shape_n, shape_k, group_size=128):
    schema = HummingWeightSchema(
        b_dtype=dtypes.uint4, bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=group_size, has_zero_point=True,
    )
    w = torch.randn(shape_n, shape_k, dtype=torch.bfloat16, device="cuda") / (
        shape_k ** 0.5
    )
    layer = HummingLayer(
        shape_n=shape_n, shape_k=shape_k, weight_config=schema,
        torch_dtype=torch.bfloat16, mma_type="tcgen05",
    ).cuda()
    layer.load_from_unquantized(w)
    return layer


@pytest.mark.parametrize("shape_n", [192, 320])
def test_ts_edge_n_not_mult_128_rejected(shape_n):
    """N%128 != 0 is not TS-legal (BlockN==128). transform must fail loudly
    at the TS-legal gate (not silently mispack or fall back to a
    correct-but-not-TS path)."""
    layer = _build_illegal(shape_n, 512)
    with pytest.raises(AssertionError, match="TS-legal"):
        layer.transform()


def test_ts_edge_n_odd_rejected():
    """N=129 (odd, not %128): still rejected loudly -- here the generic
    weight-pack %32 guard trips before the TS-legal check, which is fine:
    the contract is 'fail loudly, never silently mispack', not a specific
    message. Pins that no silent path exists for odd N."""
    with pytest.raises(AssertionError):
        layer = _build_illegal(129, 512)
        layer.transform()


@pytest.mark.parametrize("shape_k", [96, 160, 224])
def test_ts_edge_k_not_mult_64_rejected(shape_k):
    """K%64 != 0 is not TS-legal (BlockK==64). transform must fail loudly.
    gs = shape_k (one group, %8==0, divides K) so the rejection is on the
    K%64 rule, not a quant-group or gs/K mismatch."""
    layer = _build_illegal(128, shape_k, group_size=shape_k)
    with pytest.raises(AssertionError, match="TS-legal"):
        layer.transform()
