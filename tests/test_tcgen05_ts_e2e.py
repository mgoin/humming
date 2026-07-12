"""End-to-end layer-API validation for the TS-mode tcgen05 path.

Drives the FULL production path a real caller uses -- HummingLayer with
mma_type="tcgen05" -> load_from_unquantized (real bf16 weights in) ->
transform() (D's production slot-paired TS packer, kUseTcgen05Ts) ->
heuristic dispatch -> TS kernel -> output -- and compares against the
humming.utils.test dequant reference GEMM.

Distinct from tests/test_tcgen05_ts.py in two ways that matter for the
"e2e formats" gate:
  1. It compares against the dequant *reference* (bf16-rounded weights,
     fp32 matmul -- track B's subtlety), not against the SS/mma.sync
     path. A silent SS fallback that happened to also be correct would
     still pass test_ts_layer_opt_in_matches_default; here we additionally
  2. ASSERT the heuristic actually returns the TS config
     (use_tcgen05_ts=True), so "the TS kernel really ran" is pinned, not
     assumed.

Formats validated e2e here: bf16 A x uint4, group_size=128, zero-point
ON and OFF. uint8 / uint2 / uint3 are NOT reachable through this path --
the TS kernel static_asserts ElementB::kBits == 4 and Sm100Heuristics.
supports_tcgen05_ts gates b_dtype to uint4; see the "e2e formats"
section of workbook.md for the scoping note.

Run with:
  .venv/bin/python -m pytest humming/tests/test_tcgen05_ts_e2e.py -x -v
"""

from __future__ import annotations

import pytest
import torch

from humming import dtypes
from humming.config import GemmType, MmaType
from humming.layer import HummingLayer
from humming.schema.humming import HummingWeightSchema
from humming.tune import get_heuristics_class
from humming.utils.test import generate_random_inputs, generate_random_weight


def _is_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(
    not _is_blackwell(), reason="TS-mode tcgen05 needs sm_100+"
)


def _assert_close(outputs, outputs_ref, label="", atol=0.5):
    abs_err = (outputs.float() - outputs_ref.float()).abs()
    ref_abs = outputs_ref.float().abs()
    print(
        f"\n  [{label}] max|err|={abs_err.max().item():.3e} "
        f"mean|err|={abs_err.mean().item():.3e} "
        f"|ref|.mean={ref_abs.mean().item():.3e} "
        f"|ref|.max={ref_abs.max().item():.3e} atol={atol}"
    )
    # The TS dequant rounds (code - zp) * scale to bf16 per element, and
    # the reference rounds the dequantised weight to bf16 identically, so
    # per-element the two dequants agree exactly -- the residual is pure
    # fp32-accumulation-order noise from the K-reduction (mean|err| stays
    # ~1e-5..6e-5 here). Its tail grows with K: max|err| <= 0.25 at
    # K<=1024, up to 1.0 at K=2048. This is the same K-scaled tolerance
    # the shipped prod-WS suite uses (tests/test_tcgen05_dtypes.py uses
    # atol=2.0 at K=4096), NOT a blanket loosening.
    torch.testing.assert_close(outputs, outputs_ref, rtol=1e-2, atol=atol)


def _build_ts_layer(shape_n, shape_k, group_size, has_zero_point, weight_orig,
                    b_dtype=dtypes.uint4):
    """Build + load + transform a TS-opted-in HummingLayer from a real
    (unquantized) bf16 weight, exactly as a production caller would."""
    schema = HummingWeightSchema(
        b_dtype=b_dtype,
        bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=group_size,
        has_zero_point=has_zero_point,
    )
    layer = HummingLayer(
        shape_n=shape_n,
        shape_k=shape_k,
        weight_config=schema,
        torch_dtype=torch.bfloat16,
        mma_type="tcgen05",
    ).cuda()
    layer.load_from_unquantized(weight_orig)
    layer.transform()
    return layer


def _assert_ts_dispatched(layer, shape_m):
    """Fail if the heuristic would silently pick an SS / mma.sync config
    for this meta at `shape_m` instead of the TS kernel."""
    meta = layer.humming_metas[""]
    assert meta.mma_type == MmaType.TCGEN05, meta.mma_type
    cls = get_heuristics_class()
    assert cls.supports_tcgen05_ts(meta), "meta is not TS-legal"
    cfg = cls.get_config(meta, shape_m=shape_m, gemm_type=GemmType.DENSE)
    assert cfg.get("use_tcgen05_ts") is True, (
        f"heuristic did NOT dispatch TS at M={shape_m}: {cfg}"
    )
    assert cfg.get("mma_type") == "tcgen05", cfg
    assert cfg.get("use_tcgen05") is True, cfg


# ---------------------------------------------------------------------------
# bf16 A x uint4, group_size=128, zero-point ON and OFF -- the production
# W4A16 target, driven through the full HummingLayer path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("has_zero_point", [True, False])
@pytest.mark.parametrize(
    "shape_m,shape_n,shape_k",
    [
        (16, 512, 512),      # decode-ish M (TS runs at every M)
        (128, 512, 512),     # square
        (256, 1024, 2048),   # multi-block N/K walk
        (512, 512, 512),     # larger M
    ],
)
def test_ts_e2e_uint4_layer(shape_m, shape_n, shape_k, has_zero_point):
    group_size = 128

    torch.manual_seed(0xBEEF)
    (weight_orig, weight_ref, _codes, _scale, _zp, _gs) = generate_random_weight(
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

    # Reference: round the dequantised weight to bf16 (the kernel dequants
    # to bf16 per element), then fp32 matmul -- track B's subtlety, keeps
    # the comparison tight without loosening atol.
    weight_ref_bf16 = weight_ref.to(torch.bfloat16).float()
    outputs_ref = inputs_ref.matmul(weight_ref_bf16.T).to(torch.bfloat16)
    torch.cuda.synchronize()

    outputs = layer.forward(inputs.clone())
    torch.cuda.synchronize()

    assert outputs.shape == (shape_m, shape_n)
    assert outputs.dtype == torch.bfloat16
    assert torch.isfinite(outputs).all()
    atol = 0.5 if shape_k <= 1024 else 1.5
    _assert_close(
        outputs, outputs_ref, atol=atol,
        label=f"uint4 gs128 zp={has_zero_point} m{shape_m} n{shape_n} k{shape_k}",
    )


@pytest.mark.parametrize("has_zero_point", [True, False])
@pytest.mark.parametrize(
    "shape_m,shape_n,shape_k",
    [
        (16, 512, 512),      # decode-ish M (TS runs at every M)
        (256, 1024, 2048),   # multi-block N/K walk
    ],
)
def test_ts_e2e_uint2_layer(shape_m, shape_n, shape_k, has_zero_point):
    """uint2 driven through the full HummingLayer path vs the dequant
    reference. Also pins that the heuristic actually dispatches TS."""
    group_size = 128

    torch.manual_seed(0xBEEF)
    (weight_orig, weight_ref, _codes, _scale, _zp, _gs) = generate_random_weight(
        n=shape_n, k=shape_k, group_size=group_size,
        dtype=dtypes.uint2, scale_dtype=dtypes.bfloat16,
        has_zero_point=has_zero_point,
    )

    layer = _build_ts_layer(
        shape_n, shape_k, group_size, has_zero_point, weight_orig,
        b_dtype=dtypes.uint2,
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
    assert torch.isfinite(outputs).all()
    atol = 0.5 if shape_k <= 1024 else 1.5
    _assert_close(
        outputs, outputs_ref, atol=atol,
        label=f"uint2 gs128 zp={has_zero_point} m{shape_m} n{shape_n} k{shape_k}",
    )


def test_ss_tcgen05_default_path_stream_k_regression():
    """The DEFAULT (non-opt-in) sm100 tcgen05 fast-path also had
    use_stream_k defaulting to True via HummingKernel, which corrupts
    large-K outputs (the tcgen05 epilogue has no stream-K reduction).
    Drive a shape that is _is_tcgen05_eligible with large K through the
    plain HummingLayer forward and check it matches the dequant
    reference -- would fail with mean|err| ~0.03 before the fix."""
    shape_m, shape_n, shape_k, group_size = 256, 4096, 4096, 128

    torch.manual_seed(0xD00D)
    (weight_orig, weight_ref, _c, _s, _z, _g) = generate_random_weight(
        n=shape_n, k=shape_k, group_size=group_size,
        dtype=dtypes.uint4, scale_dtype=dtypes.bfloat16, has_zero_point=True,
    )
    schema = HummingWeightSchema(
        b_dtype=dtypes.uint4, bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=group_size, has_zero_point=True,
    )
    # No mma_type -> default heuristic path (mma.sync/SS tcgen05 mix).
    layer = HummingLayer(
        shape_n=shape_n, shape_k=shape_k, weight_config=schema,
        torch_dtype=torch.bfloat16,
    ).cuda()
    layer.load_from_unquantized(weight_orig)
    layer.transform()

    meta = layer.humming_metas[""]
    cfg = get_heuristics_class().get_config(
        meta, shape_m=shape_m, gemm_type=GemmType.DENSE
    )
    # This shape is TCGEN05-eligible; pin that so the test actually
    # exercises the SS tcgen05 path (not mma.sync) and that stream-K
    # is off.
    assert cfg.get("mma_type") == "tcgen05", cfg
    assert cfg.get("use_stream_k") is False, cfg

    _, inputs_ref, inputs, _ = generate_random_inputs(
        m=shape_m, k=shape_k, group_size=0, dtype=dtypes.bfloat16,
    )
    weight_ref_bf16 = weight_ref.to(torch.bfloat16).float()
    outputs_ref = inputs_ref.matmul(weight_ref_bf16.T).to(torch.bfloat16)
    torch.cuda.synchronize()

    outputs = layer.forward(inputs.clone())
    torch.cuda.synchronize()
    # K=4096: bf16 accumulation tail, same convention as the shipped
    # prod-WS suite (tests/test_tcgen05_dtypes.py uses atol=2.0 at K=4096).
    _assert_close(
        outputs, outputs_ref, atol=2.0,
        label="SS tcgen05 default m256 n4096 k4096",
    )


def test_ts_e2e_uint8_deferred():
    """uint8 is NOT e2e-reachable: the TS kernel static_asserts
    ElementB::kBits == 4 and supports_tcgen05_ts gates b_dtype to uint4.
    Requesting mma_type=tcgen05 with uint8 must fail loudly at transform
    (not silently mispack or fall back to SS). Pins the scoping note."""
    schema = HummingWeightSchema(
        b_dtype=dtypes.uint8, bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=128, has_zero_point=True,
    )
    w = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda") / (512 ** 0.5)
    layer = HummingLayer(
        shape_n=512, shape_k=512, weight_config=schema,
        torch_dtype=torch.bfloat16, mma_type="tcgen05",
    ).cuda()
    layer.load_from_unquantized(w)
    with pytest.raises(AssertionError):
        layer.transform()
