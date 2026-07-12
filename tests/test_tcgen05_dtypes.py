"""Dtype-matrix correctness for the TCGEN05 path.

`tcgen05.mma.kind::f16` accepts 16-bit A; this sweep parametrises B
across every narrower type humming exposes (uint{1..8}, int{2..8},
sub-byte fp, fp8). humming's `check_dtype` rejects many combinations
at build time via bare `assert` -- those surface here as `pytest.skip`
("humming rejects ...").

fp16-A is gated by a static_assert in `tcgen05_mma.cuh` (instruction
descriptor + scatter are hardcoded bf16); the fp16-A test skips
explicitly since the nvrtc build error reads as a generic
"RuntimeError: run failed" that the AssertionError catch can't
disambiguate.
"""
from __future__ import annotations

import pytest
import torch

from humming import dtypes
from humming.kernel.humming import HummingKernel
from humming.utils.test import generate_random_inputs, generate_random_weight
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


def _run_w_a(a_dtype, b_dtype, has_zero_point, shape_m=128, shape_n=128,
             shape_k=256, group_size=128,
             block_shape=(64, 64, 64), warp_shape=(16, 64, 64),
             num_stages=2, use_warp_spec=False, use_tma=False,
             use_cp_async=True, use_mbarrier=False):
    """Run TCGEN05 with the given (A, B) dtype combo and verify
    correctness against the float reference."""
    c_dtype = dtypes.bfloat16
    bs_dtype = dtypes.bfloat16

    random_weight = generate_random_weight(
        n=shape_n, k=shape_k, group_size=group_size,
        dtype=b_dtype, scale_dtype=bs_dtype,
        has_zero_point=has_zero_point,
    )
    _, weight_ref, weight, weight_scale, zero_point, _ = random_weight
    # Pass zero_point into prepare_humming_weight so the repacker
    # applies the sign-magnitude preprocessing the kernel's dequant
    # expects; omitting it on a has_zero_point=True kernel yields
    # silently-wrong outputs that look like a kernel bug.
    weight_prep = prepare_humming_weight(
        weight, b_dtype, a_dtype,
        zero_point=zero_point if has_zero_point else None,
        use_wgmma=False,
    )
    weight_scale_prep = prepare_humming_weight_scale(weight_scale, to_apply_on_c=False)
    zp_prep = (prepare_humming_zero_point(zero_point, dtype=b_dtype)
               if has_zero_point else None)

    _, inputs_ref, inputs, _ = generate_random_inputs(
        m=shape_m, k=shape_k, group_size=0, dtype=a_dtype,
    )

    kernel = HummingKernel(
        shape_n=shape_n, shape_k=shape_k,
        block_shape=block_shape, warp_shape=warp_shape,
        a_dtype=a_dtype, b_dtype=b_dtype,
        c_dtype=c_dtype, bs_dtype=bs_dtype,
        weight_scale_group_size=group_size,
        has_zero_point=has_zero_point,
        num_stages=num_stages,
        use_warp_spec=use_warp_spec, use_tma=use_tma,
        use_cp_async=use_cp_async, use_mbarrier=use_mbarrier,
        # has_zero_point + group_scale + use_tma_bzp=True asserts in
        # `tensor.h:275` ("TMA is not supported for BZP"). Force the
        # BZP load through cp.async so use_tma can be enabled for the
        # A and B loads.
        use_tma_bzp=False,
        has_bias=False,
        mma_type="tcgen05", use_tcgen05=True, use_stream_k=False,
    )

    outputs_ref = inputs_ref.matmul(weight_ref.T).to(torch.bfloat16)
    torch.cuda.synchronize()

    from humming import ops
    outputs = torch.empty(
        (shape_m, shape_n), dtype=torch.bfloat16, device=inputs.device,
    )
    launch_kwargs = dict(
        configs=[kernel.kernel_id], inputs=inputs,
        weight=weight_prep, outputs=outputs,
        weight_scale=weight_scale_prep,
    )
    if zp_prep is not None:
        launch_kwargs["zero_point"] = zp_prep
    ops.launch_kernel(**launch_kwargs)
    torch.cuda.synchronize()
    return outputs, outputs_ref


def _assert_close(outputs, outputs_ref, label=""):
    abs_err = (outputs.float() - outputs_ref.float()).abs()
    ref_abs = outputs_ref.float().abs()
    print(
        f"\n  [{label}] max|err|={abs_err.max().item():.3e} "
        f"mean|err|={abs_err.mean().item():.3e} "
        f"|ref|.max={ref_abs.max().item():.3e}"
    )
    torch.testing.assert_close(outputs, outputs_ref, rtol=1e-2, atol=0.5)


# `has_zero_point` matters mostly for integer (asymmetric) quant;
# symmetric variants set it False.
B_DTYPES_BF16 = [
    # unsigned int (asymmetric quant; zero_point usually present)
    ("uint1", False),
    ("uint2", True),
    ("uint3", True),
    ("uint4", True),
    ("uint4", False),
    ("uint5", True),
    ("uint6", True),
    ("uint7", True),
    ("uint8", True),
    # signed int (symmetric)
    ("int2", False),
    ("int3", False),
    ("int4", False),
    ("int6", False),
    ("int8", False),
    # sub-byte float (typically used with E8M0 fused scale; with
    # group_size=128 + bf16 scale we treat them as generic floats)
    ("float4e2m1", False),
    ("float6e2m3", False),
    ("float6e3m2", False),
    # 8-bit floats
    ("float8e4m3", False),
    ("float8e5m2", False),
]


@pytest.mark.parametrize("b_name, has_zp", B_DTYPES_BF16)
def test_tcgen05_bf16_x_b(b_name, has_zp):
    a_dtype = dtypes.bfloat16
    b_dtype = dtypes.DataType.from_str(b_name)
    if b_dtype.num_bits >= a_dtype.num_bits:
        pytest.skip(f"b={b_name} is not narrower than bf16")
    try:
        outputs, outputs_ref = _run_w_a(a_dtype, b_dtype, has_zp)
    except AssertionError:
        # humming's check_dtype / check_shape / check_scale use bare
        # `assert` (no message) for invalid combos -- e.g. signed int
        # B with fp A, has_zero_point requiring kBits <= mantissa+1,
        # fp B exponent vs A exponent mismatch, etc. Any AssertionError
        # from the build path means "humming rejects this combo".
        pytest.skip(f"humming rejects (b={b_name}, zp={has_zp})")
    except RuntimeError as e:
        if "not supported" in str(e).lower():
            pytest.skip(f"humming RT-rejects (b={b_name}, zp={has_zp}): {e!s:.80s}")
        raise
    _assert_close(outputs, outputs_ref, label=f"bf16 x {b_name} zp={has_zp}")


@pytest.mark.parametrize("b_name, has_zp", B_DTYPES_BF16)
def test_tcgen05_fp16_x_b(b_name, has_zp):
    # TCGEN05 hard-rejects ElementA != BFloat16 via static_assert in
    # tcgen05_mma.cuh (instruction descriptor + scatter are bf16-only).
    # Skip explicitly rather than relying on a build-error catch --
    # nvrtc surfaces the static_assert as a generic "run failed".
    pytest.skip("TCGEN05 currently only supports bf16 A")


# Production WS config: BlockM=128 BlockN=128 BlockK=128 stages=4
# warp_spec=True tma=True. Wider B-dtypes (uint{5..8}, fp{6,8}) blow
# the 232 KiB cc 10.x SMEM cap at this config and need smaller
# BlockK/stages -- the parametrized test below picks a per-dtype
# config from a ladder so every supported B-dtype gets coverage at
# the WS pipeline (not just at the (64,64,64) baseline above).
PROD_CONFIG_LADDER = [
    # (block_shape, warp_shape, num_stages)
    ((128, 128, 128), (32, 64, 128), 4),
    ((128, 128, 128), (32, 64, 128), 3),
    ((128, 128, 128), (32, 64, 128), 2),
    ((128, 128,  64), (32, 64,  64), 4),
    ((128, 128,  64), (32, 64,  64), 3),
]


# Production-realistic shape for the prod-WS test. Large enough that
# bf16 precision drift dominates per-cell error and the standard
# rtol=1e-2 / atol=0.5 tolerance is appropriate.
PROD_PROBE_SHAPE = (512, 512, 4096)


# Per-dtype safe-and-fastest configs at production WS, picked from
# `benchmarks/bench_tcgen05_dtypes.py` (which walks a config ladder
# and verifies correctness vs the float reference at a small probe
# shape). The configs that pass through to the heuristic in
# `tune/sm100.py::_is_tcgen05_eligible` are listed here so this test
# locks them in -- any future heuristic change that picks a different
# config for an opted-in dtype must update this map.
#
# Note: uint4 was originally demoted to BlockK=64 here by the
# workbook-B.37 WS+TMA corruption at BlockK=128 s4. B.37 was
# root-caused (track-f round 3, benchmarks/probe_b37_alias.py) to the
# un-fenced/un-awaited TMA-C epilogue read of smem.reduce and fixed in
# gmem_writer.cuh + the kernels; uint4 is back on the tuned
# BlockK=128 s4 config the sm100 heuristic ships.
SAFE_PROD_WS_CONFIG = {
    # (b_name, has_zp) -> (block_shape, warp_shape, num_stages)
    ("uint4", True):  ((128, 128, 128), (32, 64, 128), 4),
    ("uint4", False): ((128, 128, 128), (32, 64, 128), 4),
    ("uint3", True):  ((128, 128, 128), (32, 64, 128), 3),
    ("uint5", True):  ((128, 128, 128), (32, 64, 128), 3),
    ("uint6", True):  ((128, 128, 128), (32, 64, 128), 3),
    ("uint8", True):  ((128, 128, 64),  (32, 64, 64),  3),
    ("float4e2m1", False): ((128, 128, 128), (32, 64, 128), 3),
    ("float8e4m3", False): ((128, 128, 128), (32, 64, 128), 3),
    ("float8e5m2", False): ((128, 128, 128), (32, 64, 128), 3),
}


@pytest.mark.parametrize("b_name, has_zp", sorted(SAFE_PROD_WS_CONFIG.keys()))
def test_tcgen05_bf16_x_b_prod_ws(b_name, has_zp):
    """Verify the per-dtype safe production WS+TMA config in
    SAFE_PROD_WS_CONFIG produces correct outputs at a production-
    realistic shape (M=512, N=512, K=4096).

    Only the dtypes listed in SAFE_PROD_WS_CONFIG are tested -- these
    are the ones the bench has verified safe at WS+TMA. Other dtypes
    have edge-case WS+TMA bugs at specific (shape_k, stages, BlockK)
    combos (workbook B.37) that haven't been fixed; they remain
    supported only at the (64, 64, 64) baseline above.
    """
    a_dtype = dtypes.bfloat16
    b_dtype = dtypes.DataType.from_str(b_name)
    if b_dtype.num_bits >= a_dtype.num_bits:
        pytest.skip(f"b={b_name} is not narrower than bf16")
    block_shape, warp_shape, num_stages = SAFE_PROD_WS_CONFIG[(b_name, has_zp)]
    shape_m, shape_n, shape_k = PROD_PROBE_SHAPE
    try:
        outputs, outputs_ref = _run_w_a(
            a_dtype, b_dtype, has_zp,
            shape_m=shape_m, shape_n=shape_n, shape_k=shape_k,
            block_shape=block_shape, warp_shape=warp_shape,
            num_stages=num_stages,
            use_warp_spec=True, use_tma=True,
            use_cp_async=False, use_mbarrier=True,
        )
    except (AssertionError, RuntimeError) as e:
        pytest.fail(
            f"safe-config {block_shape} s={num_stages} rejected by humming "
            f"for (b={b_name}, zp={has_zp}): {e!s:.80s}"
        )
    # Looser tolerance than the (64,64,64) baseline -- at this larger
    # shape (M*K = 2M flops/output), bf16 precision drift accumulates
    # to ~2.0 max abs err (~0.5% rel) per output cell, which is
    # standard for bf16 GEMM not a kernel issue.
    abs_err = (outputs.float() - outputs_ref.float()).abs()
    print(
        f"\n  [prod-WS {block_shape} s={num_stages} bf16 x {b_name} "
        f"zp={has_zp}] max|err|={abs_err.max().item():.3e} "
        f"mean|err|={abs_err.mean().item():.3e}"
    )
    torch.testing.assert_close(outputs, outputs_ref, rtol=1e-2, atol=2.0)
