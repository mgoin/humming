"""TCGEN05 TS-mode (track b-ts-staging) correctness tests.

TS mode = dequant to registers, tcgen05.st into double-buffered TMEM
staging, A<->B-swapped TS-mode tcgen05.mma ([d_tmem], [w_tmem],
act_smem_desc). Weights use the PRODUCTION slot-paired TS layout
(docs/tcgen05_ts_packing.md) via the real repack path:
prepare_humming_weight(use_tcgen05_ts=True) -> kUseTcgen05Ts CUDA
repack -> loader_b half-group gather. Scale/zp streams come from
humming.utils.ts_packing (bit-identical to the retired throwaway
packer's streams -- pinned by tests/test_ts_packing_cross_track.py).

Prototype config space (asserted in mma/tcgen05_ts_mma.cuh):
  BlockN == 128, WarpN == 32, WarpM == BlockM in {64, 128},
  WarpK == BlockK == 64, bf16 x uint4, group scale (gs >= BlockK).
"""

from __future__ import annotations

import pytest
import torch

from humming import dtypes
from humming.kernel.humming import HummingKernel
from humming.utils.test import generate_random_inputs, generate_random_weight
from humming.utils.ts_packing import (
    pack_scales_tcgen05_ts,
    pack_zero_point_tcgen05_ts,
)
from humming.utils.weight import prepare_humming_weight


def _is_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(not _is_blackwell(), reason="tcgen05 needs sm_100+")


def _run_ts(
    shape_m, shape_n, shape_k,
    block_shape=(64, 128, 64),
    num_stages=3,
    has_zero_point=True,
    has_bias=False,
    group_size=128,
    use_warp_spec=False,
):
    a_dtype = dtypes.bfloat16
    b_dtype = dtypes.uint4
    c_dtype = dtypes.bfloat16
    bs_dtype = dtypes.bfloat16

    torch.manual_seed(123)
    random_weight = generate_random_weight(
        n=shape_n, k=shape_k, group_size=group_size,
        dtype=b_dtype, scale_dtype=bs_dtype,
        has_zero_point=has_zero_point,
    )
    _, weight_ref, weight_codes, weight_scale, zero_point, _ = random_weight

    weight = prepare_humming_weight(
        weight_codes, b_dtype, a_dtype, zero_point=zero_point,
        use_wgmma=False, use_tcgen05_ts=True,
    )
    weight_scale_p = pack_scales_tcgen05_ts(weight_scale).cuda()
    zero_point_p = None
    if has_zero_point:
        zero_point_p = pack_zero_point_tcgen05_ts(
            zero_point.to(torch.int32), b_dtype.num_bits).cuda()

    _, inputs_ref, inputs, _ = generate_random_inputs(
        m=shape_m, k=shape_k, group_size=0, dtype=a_dtype,
    )
    bias = None
    if has_bias:
        torch.manual_seed(456)
        bias = torch.randn(shape_n, dtype=torch.bfloat16, device=inputs.device)

    kernel = HummingKernel(
        shape_n=shape_n,
        shape_k=shape_k,
        block_shape=block_shape,
        warp_shape=(block_shape[0], 32, block_shape[2]),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        bs_dtype=bs_dtype,
        weight_scale_group_size=group_size,
        has_zero_point=has_zero_point,
        num_stages=num_stages,
        use_warp_spec=use_warp_spec,
        use_tma=use_warp_spec,
        use_cp_async=not use_warp_spec,
        use_mbarrier=use_warp_spec,
        use_tma_bzp=False,
        has_bias=has_bias,
        mma_type="tcgen05",
        use_tcgen05=True,
        use_tcgen05_ts=True,
        use_stream_k=False,
    )

    # The TS dequant rounds (code - zp) * scale to bf16 per element
    # (plain __hmul2, same as the SS path's bs application) -- round the
    # reference weights identically so the comparison stays tight at
    # large K. Against the fp32-weight reference the K=8192 prod shape
    # drifts to ~0.9 abs err (bf16 accumulation noise, cf. workbook
    # B.38); against this reference mean err is ~4e-4 with max 1 ulp.
    weight_ref = weight_ref.to(torch.bfloat16).float()
    outputs_ref = inputs_ref.matmul(weight_ref.T)
    if has_bias:
        outputs_ref = outputs_ref + bias
    outputs_ref = outputs_ref.to(torch.bfloat16)
    torch.cuda.synchronize()

    from humming import ops
    outputs = torch.empty(
        (shape_m, shape_n), dtype=torch.bfloat16, device=inputs.device,
    )
    launch_kwargs = dict(
        configs=[kernel.kernel_id],
        inputs=inputs, weight=weight, outputs=outputs,
        weight_scale=weight_scale_p,
    )
    if zero_point_p is not None:
        launch_kwargs["zero_point"] = zero_point_p
    if bias is not None:
        launch_kwargs["bias"] = bias
    ops.launch_kernel(**launch_kwargs)
    torch.cuda.synchronize()
    return outputs, outputs_ref


def _assert_close(outputs, outputs_ref):
    abs_err = (outputs.float() - outputs_ref.float()).abs()
    ref_abs = outputs_ref.float().abs()
    print(
        f"\n  max|err|={abs_err.max().item():.3e} "
        f"mean|err|={abs_err.mean().item():.3e} "
        f"|ref|.mean={ref_abs.mean().item():.3e} "
        f"|ref|.max={ref_abs.max().item():.3e}"
    )
    torch.testing.assert_close(outputs, outputs_ref, rtol=1e-2, atol=0.5)


# ---------------------------------------------------------------------------
# Milestone 4 target: ONE config, M=N=K=512, bf16 x uint4 gs=128.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("has_zero_point", [False, True])
def test_ts_512_cubed(has_zero_point):
    outputs, outputs_ref = _run_ts(
        shape_m=512, shape_n=512, shape_k=512,
        has_zero_point=has_zero_point,
    )
    _assert_close(outputs, outputs_ref)


@pytest.mark.parametrize("num_stages", [2, 3, 4])
def test_ts_stages(num_stages):
    outputs, outputs_ref = _run_ts(
        shape_m=128, shape_n=128, shape_k=512, num_stages=num_stages,
    )
    _assert_close(outputs, outputs_ref)


@pytest.mark.parametrize("block_m", [64, 128])
def test_ts_block_m(block_m):
    outputs, outputs_ref = _run_ts(
        shape_m=256, shape_n=256, shape_k=1024,
        block_shape=(block_m, 128, 64),
    )
    _assert_close(outputs, outputs_ref)


def test_ts_bias():
    outputs, outputs_ref = _run_ts(
        shape_m=128, shape_n=256, shape_k=512, has_bias=True,
    )
    _assert_close(outputs, outputs_ref)


def test_ts_warp_spec():
    outputs, outputs_ref = _run_ts(
        shape_m=512, shape_n=512, shape_k=512, use_warp_spec=True,
    )
    _assert_close(outputs, outputs_ref)


def test_ts_warp_spec_large_k():
    """Guards the deferred G2S stage release: at large K the TMA
    producer laps the MMA queue, the regime where releasing a stage at
    kWarpIters-2 produced real corruption on track A's SS pipeline."""
    outputs, outputs_ref = _run_ts(
        shape_m=512, shape_n=1024, shape_k=8192,
        block_shape=(128, 128, 64), num_stages=4, use_warp_spec=True,
    )
    _assert_close(outputs, outputs_ref)


def test_ts_prod_shape():
    """Llama70B-gate slice at modest M -- multi-block N/K walk."""
    outputs, outputs_ref = _run_ts(
        shape_m=128, shape_n=1024, shape_k=8192,
        block_shape=(128, 128, 64), num_stages=4,
    )
    _assert_close(outputs, outputs_ref)
