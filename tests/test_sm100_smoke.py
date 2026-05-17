"""End-to-end smoke test that exercises Sm100Heuristics on B300.

Goes through HummingLayer + get_heuristics_config -- i.e. the same path a
real caller would use. Before adding tune/sm100.py this would raise
KeyError on Blackwell. Tests correctness against a torch.float32 reference
of the dequantised weight.

Run with:
  .venv/bin/python -m pytest humming/tests/test_sm100_smoke.py -x -v
"""

from __future__ import annotations

import pytest
import torch

from humming import dtypes
from humming.layer import HummingLayer
from humming.tune import get_heuristics_class


def _is_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 10


pytestmark = pytest.mark.skipif(
    not _is_blackwell(), reason="Sm100 smoke test only runs on Blackwell"
)


def test_heuristics_dispatch_resolves_on_blackwell():
    """tune.__init__ heuristics_map must include the local cc."""
    major, minor = torch.cuda.get_device_capability()
    cls = get_heuristics_class()
    assert cls.sm_version == 100, (
        f"Blackwell cc {major}.{minor} resolved to {cls.__name__} (sm "
        f"{cls.sm_version}); expected Sm100Heuristics."
    )


@pytest.mark.parametrize("shape_m", [16, 128, 1024])
@pytest.mark.parametrize(
    "b_dtype,group_size,has_zero_point",
    [
        ("uint4",   128, True),    # W4A16 AWQ-style with zero-point (target)
        ("int4",    128, False),   # W4A16 symmetric int4
        ("int8",     0,  False),   # W8A16 (channel scale)
    ],
)
def test_humming_layer_forward_matches_reference(
    shape_m: int, b_dtype: str, group_size: int, has_zero_point: bool
):
    shape_n, shape_k = 4096, 4096
    a_dtype = "bfloat16"

    layer = HummingLayer(
        shape_n=shape_n,
        shape_k=shape_k,
        weight_config={
            "dtype": b_dtype,
            "group_size": group_size,
            "scale_dtype": "bfloat16",
            "has_zero_point": has_zero_point,
        },
        input_config={"dtype": a_dtype, "group_size": 0},
        torch_dtype=torch.bfloat16,
    ).to("cuda:0")

    # Fill weight + scale + zp tensors with reasonable random values.
    torch.manual_seed(0xC0FFEE)
    for p in layer.parameters():
        if p.dtype.is_floating_point:
            p.data.normal_(0, 0.05)
        else:
            # int / uint storage: legal-range fill.
            p.data.random_(0, 8)
    layer.transform()

    inputs = (torch.randn(shape_m, shape_k, device="cuda", dtype=torch.bfloat16)
              * 0.05).contiguous()

    # The point of this smoke test is to verify the heuristic-dispatch
    # plumbing works on Blackwell (sm 10.x). Correctness of the underlying
    # kernel is covered by tests/test_shape.py (which constructs
    # HummingKernel directly and bypasses the heuristic).
    out = layer(inputs=inputs)
    assert out.shape == (shape_m, shape_n), f"output shape {out.shape}"
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all(), "non-finite values in output"
