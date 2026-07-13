"""TCGEN05 TS-mode grouped GEMM (MoE) correctness tests.

Grouping lives entirely *above* the MMA (scheduler -> g2s loaders ->
epilogue); the TS mainloop/drain are grouping-agnostic, so both
GROUPED_CONTIGUOUS and GROUPED_MASKED run through the production
slot-paired TS weight/scale/zp packing (docs/tcgen05_ts_packing.md)
with zero kernel changes -- see .scratch/humming-v2/expand-plans/
moe-grouped-gemm.md. This suite proves that across the realistic
space instead of just the single POC point:

  mode{contiguous,masked} x zp{on,off} x block_m{64,128}
    x use_tma{True,False} x num_experts{4,8,128,256}.

use_tma=False is the legacy-C epilogue (write_legacy row masking),
which the POC only partly exercised (it ran TMA-C for both modes).

Reference = per-expert dequant matmul against the bf16-rounded weight
(the TS dequant rounds (code - zp) * scale to bf16 per element, so the
reference is rounded identically -- same convention as the dense TS
suite test_tcgen05_ts.py). This is the same ground truth the mma.sync
grouped path is itself validated against in test_moe.py::
test_grouped_gemm, so matching it pins TS to the proven grouped scatter.
"""

from __future__ import annotations

import pytest
import torch

from humming import dtypes, ops
from humming.kernel.humming import HummingKernel
from humming.layer import HummingLayerMeta
from humming.tune import get_heuristics_config
from humming.utils.test import (
    generate_random_inputs,
    generate_random_moe_tensors,
    generate_random_weight,
)
from humming.utils.weight import (
    prepare_humming_weight,
    prepare_humming_weight_scale,
    prepare_humming_zero_point,
)


def _is_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(not _is_blackwell(), reason="tcgen05 needs sm_100+")

N, K, GROUP_SIZE = 1024, 1024, 128


def _pick_moe(num_experts: int) -> tuple[int, int, int]:
    """(m, top_k, expert_max_tokens) sized so every expert fits under
    expert_max_tokens with a wide statistical margin (avoids the
    generate_random_moe_tensors capacity assertion firing)."""
    m, top_k = (512, 2) if num_experts <= 8 else (512, 8)
    avg = (m * top_k + num_experts - 1) // num_experts
    expert_max_tokens = max(64, 4 * avg)
    return m, top_k, expert_max_tokens


def _make_moe_layout(gemm_type, num_experts, m, top_k, expert_max_tokens):
    emt = None if gemm_type == "grouped_contiguous" else expert_max_tokens
    for _ in range(6):
        try:
            _, expert_layout, *_ = generate_random_moe_tensors(
                m, num_experts=num_experts, top_k=top_k,
                gemm_type=gemm_type, expert_max_tokens=emt,
            )
            return expert_layout, emt
        except AssertionError as e:
            if emt is not None and "expert_max_tokens" in str(e):
                emt *= 2
                continue
            raise
    raise RuntimeError("could not size expert_max_tokens")


def _run_ts_moe(
    gemm_type, num_experts, block_m, use_tma, has_zero_point,
    use_warp_spec=True,
):
    a_dtype = dtypes.bfloat16
    b_dtype = dtypes.uint4
    c_dtype = dtypes.bfloat16
    bs_dtype = dtypes.bfloat16
    torch_dtype = dtypes.torch_dtype_map[c_dtype]

    m, top_k, expert_max_tokens = _pick_moe(num_experts)
    torch.manual_seed(0)
    expert_layout, emt = _make_moe_layout(
        gemm_type, num_experts, m, top_k, expert_max_tokens)

    _, weight_ref, weight_codes, weight_scale, zero_point, _ = generate_random_weight(
        n=N, k=K, group_size=GROUP_SIZE, dtype=b_dtype, scale_dtype=bs_dtype,
        num_experts=num_experts, has_zero_point=has_zero_point,
    )
    weight = prepare_humming_weight(
        weight_codes, b_dtype, a_dtype,
        zero_point=zero_point if has_zero_point else None,
        use_tcgen05_ts=True,
    )
    weight_scale_p = prepare_humming_weight_scale(
        weight_scale, to_apply_on_c=False, use_tcgen05_ts=True)
    zero_point_p = None
    if has_zero_point:
        zero_point_p = prepare_humming_zero_point(
            zero_point, b_dtype, packed=False, use_tcgen05_ts=True)

    m_new = m * top_k if gemm_type == "grouped_contiguous" else num_experts * emt
    _, inputs_ref, inputs, _ = generate_random_inputs(
        m=m_new, k=K, group_size=0, dtype=a_dtype)

    kernel = HummingKernel(
        shape_n=N, shape_k=K,
        block_shape=(block_m, 128, 64),
        warp_shape=(block_m, 32, 64),
        a_dtype=a_dtype, b_dtype=b_dtype, c_dtype=c_dtype, bs_dtype=bs_dtype,
        num_experts=num_experts, num_stages=4,
        use_warp_spec=use_warp_spec, has_bias=False,
        has_zero_point=has_zero_point,
        weight_scale_group_size=GROUP_SIZE,
        mma_type="tcgen05", use_tcgen05=True, use_tcgen05_ts=True,
        use_tma=use_tma, use_cp_async=not use_tma, use_mbarrier=True,
        use_tma_bzp=False, use_stream_k=False,
        gemm_type=gemm_type,
    )
    # No silent fallback: this must be the TS-mode tcgen05 kernel.
    assert kernel.use_tcgen05_ts is True

    outputs = torch.zeros((m_new, N), dtype=torch_dtype, device=inputs.device)
    launch_kwargs = dict(
        configs=[kernel.kernel_id], inputs=inputs, weight=weight,
        outputs=outputs, weight_scale=weight_scale_p,
        expert_layout=expert_layout,
    )
    if zero_point_p is not None:
        launch_kwargs["zero_point"] = zero_point_p
    outputs = ops.launch_kernel(**launch_kwargs).view(-1, N)

    ref = torch.zeros_like(outputs)
    weight_ref = weight_ref.to(torch.bfloat16).float()
    for e in range(num_experts):
        if gemm_type == "grouped_contiguous":
            o1 = int(expert_layout[e])
            o2 = m_new if e == num_experts - 1 else int(expert_layout[e + 1])
        else:
            o1 = emt * e
            o2 = o1 + int(expert_layout[e])
        if o2 == o1:
            continue
        ref[o1:o2] = inputs_ref[o1:o2].matmul(weight_ref[e].T).to(torch_dtype)

    return outputs, ref, expert_layout, emt


def _assert_close(outputs, ref):
    abs_err = (outputs.float() - ref.float()).abs()
    print(
        f"\n  max|err|={abs_err.max().item():.3e} "
        f"mean|err|={abs_err.mean().item():.3e}"
    )
    assert torch.isfinite(outputs).all()
    # atol=0.5 + rtol=1e-2: at K=1024 the largest outputs reach ~1 bf16
    # ulp (~1.0 abs), covered by the rtol term. Same bound the dense TS
    # suite uses (test_tcgen05_ts.py::_assert_close).
    torch.testing.assert_close(outputs, ref, rtol=1e-2, atol=0.5)


# --------------------------------------------------------------------------
# Core matrix: mode x zp x block_m x use_tma at a small expert count.
# The use_tma=False cells are the legacy-C epilogue (incl. masked
# row-zeroing), the residual path the POC flagged as only-partly-tested.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("gemm_type", ["grouped_contiguous", "grouped_masked"])
@pytest.mark.parametrize("has_zero_point", [False, True])
@pytest.mark.parametrize("block_m", [32, 64, 128])
@pytest.mark.parametrize("use_tma", [True, False])
def test_ts_moe_matrix(gemm_type, has_zero_point, block_m, use_tma):
    outputs, ref, _, _ = _run_ts_moe(
        gemm_type, num_experts=8, block_m=block_m,
        use_tma=use_tma, has_zero_point=has_zero_point,
    )
    _assert_close(outputs, ref)


# --------------------------------------------------------------------------
# Expert-count sweep incl. fine-grained (128 / 256). block_m=64 is the
# realistic grouped tile for fine-grained MoE (few tokens/expert).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("gemm_type", ["grouped_contiguous", "grouped_masked"])
@pytest.mark.parametrize("num_experts", [4, 8, 128, 256])
def test_ts_moe_experts(gemm_type, num_experts):
    outputs, ref, _, _ = _run_ts_moe(
        gemm_type, num_experts=num_experts, block_m=64,
        use_tma=True, has_zero_point=True,
    )
    _assert_close(outputs, ref)


# --------------------------------------------------------------------------
# BlockM=32 is a validated-correct TS atom (M128N32K16) but perf-negative
# for fine-grained MoE (perf-negative on a B300 BlockM=32 sweep), so the
# heuristic keeps 64. These cells guard the small-tile drain's correctness
# at fine-grained expert counts so the atom stays trustworthy if revisited.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("gemm_type", ["grouped_contiguous", "grouped_masked"])
@pytest.mark.parametrize("num_experts", [128, 256])
def test_ts_moe_experts_blockm32(gemm_type, num_experts):
    outputs, ref, _, _ = _run_ts_moe(
        gemm_type, num_experts=num_experts, block_m=32,
        use_tma=True, has_zero_point=True,
    )
    _assert_close(outputs, ref)


# --------------------------------------------------------------------------
# Masked row-zeroing: rows past each expert's token count must stay 0.
# Exercised on both TMA-C and legacy-C epilogues.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("use_tma", [True, False])
def test_ts_moe_masked_zeroing(use_tma):
    outputs, ref, expert_layout, emt = _run_ts_moe(
        "grouped_masked", num_experts=8, block_m=64,
        use_tma=use_tma, has_zero_point=True,
    )
    num_experts = expert_layout.numel()
    for e in range(num_experts):
        lo = emt * e + int(expert_layout[e])
        hi = emt * (e + 1)
        if hi > lo:
            assert torch.count_nonzero(outputs[lo:hi]) == 0, (
                f"expert {e} masked rows [{lo}:{hi}] not zeroed")
    _assert_close(outputs, ref)


# --------------------------------------------------------------------------
# Heuristic dispatch: a grouped TCGEN05 meta must select the TS config
# (no silent fallback) with the grouped-aware block_m.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("gemm_type", ["grouped_contiguous", "grouped_masked"])
@pytest.mark.parametrize(
    "num_experts,shape_m,expected_block_m",
    # BlockM=32 is a validated-correct atom but perf-negative (see
    # sm100.py), so production keeps the fine-grained tile at 64.
    [
        (8, 2048, 128),   # 256 tok/expert -> coarse
        (8, 512, 64),     # 64 tok/expert  -> mid
        (128, 2048, 64),  # 16 tok/expert  -> fine-grained (Qwen3-MoE-like)
        (256, 512, 64),   # 2  tok/expert  -> fine-grained (DeepSeek-like)
    ],
)
def test_ts_moe_dispatch(gemm_type, num_experts, shape_m, expected_block_m):
    meta = HummingLayerMeta(
        shape_n=1536, shape_k=4096,
        a_dtype=dtypes.bfloat16, b_dtype=dtypes.uint4, c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.bfloat16, weight_scale_group_size=128,
        num_experts=num_experts, has_zero_point=True, mma_type="tcgen05",
    )
    cfg = get_heuristics_config(meta, shape_m=shape_m, gemm_type=gemm_type)
    assert cfg.get("use_tcgen05_ts") is True
    assert cfg["block_shape"][0] == expected_block_m
    assert cfg["use_stream_k"] is False
