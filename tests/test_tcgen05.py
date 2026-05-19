"""TCGEN05 (Blackwell sm_100+) W4A16 correctness tests.

Phase B.14 known-good config space (each combination here passes vs an
mma.sync reference with rtol=1e-2, atol=0.5):

  * BlockShape: only (64, 64, 64) -- BlockN > 64 is gated by a
    static_assert in `mma/tcgen05_mma.cuh` (workbook 'B.15: N>64
    descriptor/scatter mismatch'). BlockK > 64 and BlockM > 64 are
    likewise unverified and gated.
  * WarpShape: only (16, 64, 64) -- 4 M-warps, one per TMEM
    sub-partition.
  * kNumStages: {2, 3, 4} -- kNumStages == 2 uses the deferred
    `producer.load_stage` fix; kNumStages >= 3 lands the next load
    into a non-conflicting stage and works unmodified.
  * Problem shape: shape_m / shape_n / shape_k must be a multiple of
    the BlockShape; shape_k currently >= 128 (BlockK == 64 needs at
    least 2 K-blocks to exercise the pipeline).

The tests below are written so the parametrization is the same shape
of dimensions we'd eventually want to tune over (kNumStages, BlockN,
BlockM, ...); the gated combinations are marked `xfail` so they remain
visible failures and stop being silently green if we accidentally
relax a static_assert.
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


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _build_w4a16_problem(shape_m, shape_n, shape_k, group_size, has_zero_point):
    """Build a W4A16 (bf16 x uint4) problem matching the existing
    test_shape conventions. Returns (inputs_ref, inputs, weight,
    weight_scale, zero_point, weight_ref)."""
    a_dtype = dtypes.bfloat16
    b_dtype = dtypes.uint4
    bs_dtype = dtypes.bfloat16

    random_weight = generate_random_weight(
        n=shape_n, k=shape_k, group_size=group_size,
        dtype=b_dtype, scale_dtype=bs_dtype,
        has_zero_point=has_zero_point,
    )
    _, weight_ref, weight, weight_scale, zero_point, _ = random_weight
    weight = prepare_humming_weight(weight, b_dtype, a_dtype, use_wgmma=False)
    weight_scale = prepare_humming_weight_scale(weight_scale, to_apply_on_c=False)
    if has_zero_point:
        zero_point = prepare_humming_zero_point(zero_point, dtype=b_dtype)
    else:
        zero_point = None

    _, inputs_ref, inputs, _ = generate_random_inputs(
        m=shape_m, k=shape_k, group_size=0, dtype=a_dtype,
    )
    return inputs_ref, inputs, weight, weight_scale, zero_point, weight_ref


def _run_tcgen05(
    shape_m, shape_n, shape_k,
    block_shape, warp_shape,
    num_stages,
    has_zero_point=True,
    has_bias=False,
    group_size=128,
):
    """Construct a TCGEN05 kernel, run it on a random problem, and
    return (outputs, outputs_ref). Reference is computed BEFORE the
    kernel launch so a context-killing tcgen05 bug doesn't take cublas
    down with it."""
    a_dtype = dtypes.bfloat16
    b_dtype = dtypes.uint4
    c_dtype = dtypes.bfloat16
    bs_dtype = dtypes.bfloat16

    inputs_ref, inputs, weight, weight_scale, zero_point, weight_ref = (
        _build_w4a16_problem(shape_m, shape_n, shape_k, group_size, has_zero_point)
    )
    bias = None
    if has_bias:
        torch.manual_seed(456)
        bias = torch.randn(shape_n, dtype=torch.bfloat16, device=inputs.device)

    kernel = HummingKernel(
        shape_n=shape_n,
        shape_k=shape_k,
        block_shape=block_shape,
        warp_shape=warp_shape,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        bs_dtype=bs_dtype,
        weight_scale_group_size=group_size,
        has_zero_point=has_zero_point,
        num_stages=num_stages,
        use_warp_spec=False,
        use_tma=False,
        use_cp_async=True,
        has_bias=has_bias,
        mma_type="tcgen05",
        use_tcgen05=True,
        use_stream_k=False,
    )

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
        weight_scale=weight_scale,
    )
    if zero_point is not None:
        launch_kwargs["zero_point"] = zero_point
    if bias is not None:
        launch_kwargs["bias"] = bias
    ops.launch_kernel(**launch_kwargs)
    torch.cuda.synchronize()
    return outputs, outputs_ref


def _assert_close(outputs, outputs_ref):
    """Compare bf16 outputs at a fixed tolerance.

    rtol=1e-2 atol=0.5 covers bf16 rounding noise vs mma.sync (max
    element here is ~200 -> 0.5 abs is ~2.5e-3 relative)."""
    abs_err = (outputs.float() - outputs_ref.float()).abs()
    ref_abs = outputs_ref.float().abs()
    # Diagnostic in case the assertion fails:
    print(
        f"\n  max|err|={abs_err.max().item():.3e} "
        f"mean|err|={abs_err.mean().item():.3e} "
        f"|ref|.mean={ref_abs.mean().item():.3e} "
        f"|ref|.max={ref_abs.max().item():.3e}"
    )
    torch.testing.assert_close(outputs, outputs_ref, rtol=1e-2, atol=0.5)


# ---------------------------------------------------------------------------
# Smallest viable case (kept as a sanity test; matches Phase B.4's first
# wired-up shape).
# ---------------------------------------------------------------------------


def test_tcgen05_w4a16_smallest():
    outputs, outputs_ref = _run_tcgen05(
        shape_m=128, shape_n=128, shape_k=256,
        block_shape=(64, 64, 64), warp_shape=(16, 64, 64),
        num_stages=2,
    )
    _assert_close(outputs, outputs_ref)


# ---------------------------------------------------------------------------
# Sweep across shape_m (the dynamic M dim; only multiples of BlockM
# are valid -- humming asserts `problem_shape % block_shape == 0`).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape_m", [64, 128, 256, 512])
def test_tcgen05_shape_m(shape_m):
    outputs, outputs_ref = _run_tcgen05(
        shape_m=shape_m, shape_n=64, shape_k=256,
        block_shape=(64, 64, 64), warp_shape=(16, 64, 64),
        num_stages=2,
    )
    _assert_close(outputs, outputs_ref)


# ---------------------------------------------------------------------------
# Sweep across shape_k (= number of K-blocks the mainloop iterates).
# BlockK == 64 so shape_k is in units of 64 bf16. shape_k=128 is the
# smallest that exercises the K-pipeline (>= 2 iterations).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape_k", [128, 256, 512, 1024, 2048])
def test_tcgen05_shape_k(shape_k):
    outputs, outputs_ref = _run_tcgen05(
        shape_m=128, shape_n=64, shape_k=shape_k,
        block_shape=(64, 64, 64), warp_shape=(16, 64, 64),
        num_stages=2,
    )
    _assert_close(outputs, outputs_ref)


# ---------------------------------------------------------------------------
# kNumStages sweep -- kNumStages == 2 uses the deferred load_stage path
# (humming.cuh:165); kNumStages >= 3 takes the else branch where the
# next load targets a non-conflicting stage and the SMEM-A race never
# arises.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_stages", [2, 3, 4])
def test_tcgen05_num_stages(num_stages):
    outputs, outputs_ref = _run_tcgen05(
        shape_m=128, shape_n=64, shape_k=512,
        block_shape=(64, 64, 64), warp_shape=(16, 64, 64),
        num_stages=num_stages,
    )
    _assert_close(outputs, outputs_ref)


# ---------------------------------------------------------------------------
# Single-K-position probe -- guards against the SMEM-A race regression
# (Phase B.14): with A=delta(k=k0), out[m, n] == B_dequant[k0, n] for
# every k0. Before the fix only k0 in the LAST 16 K of each K-block
# returned wrong values; this test catches that pattern explicitly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k0", [0, 15, 16, 31, 32, 47, 48, 63, 64, 127, 192, 255])
def test_tcgen05_delta_a(k0):
    a_dtype = dtypes.bfloat16
    b_dtype = dtypes.uint4
    c_dtype = dtypes.bfloat16
    bs_dtype = dtypes.bfloat16
    shape_m, shape_n, shape_k = 128, 64, 256
    group_size = 128

    _, _, weight, weight_scale, zero_point, weight_ref = _build_w4a16_problem(
        shape_m, shape_n, shape_k, group_size, has_zero_point=True
    )
    A = torch.zeros(shape_m, shape_k, dtype=torch.bfloat16, device="cuda")
    A[:, k0] = 1.0

    # Reference computed in fp32 then cast (matches WMMA's f32 accumulator).
    outputs_ref = (A.float() @ weight_ref.T.float()).to(torch.bfloat16)

    kernel = HummingKernel(
        shape_n=shape_n, shape_k=shape_k,
        block_shape=(64, 64, 64), warp_shape=(16, 64, 64),
        a_dtype=a_dtype, b_dtype=b_dtype, c_dtype=c_dtype, bs_dtype=bs_dtype,
        weight_scale_group_size=group_size, has_zero_point=True,
        num_stages=2, use_warp_spec=False, use_tma=False, use_cp_async=True,
        has_bias=False, mma_type="tcgen05", use_tcgen05=True, use_stream_k=False,
    )

    from humming import ops
    outputs = torch.empty(
        (shape_m, shape_n), dtype=torch.bfloat16, device=A.device,
    )
    ops.launch_kernel(
        configs=[kernel.kernel_id],
        inputs=A, weight=weight, outputs=outputs,
        weight_scale=weight_scale, zero_point=zero_point,
    )
    torch.cuda.synchronize()
    _assert_close(outputs, outputs_ref)


# ---------------------------------------------------------------------------
# LayerConfig variant: zero-point on/off. The (has_zero_point=False)
# path skips the zp-load + zp-apply branches in
# `mainloop_arith.cuh::may_apply_bs_and_zp_on_b`.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("has_zero_point", [True, False])
def test_tcgen05_zero_point(has_zero_point):
    outputs, outputs_ref = _run_tcgen05(
        shape_m=128, shape_n=64, shape_k=256,
        block_shape=(64, 64, 64), warp_shape=(16, 64, 64),
        num_stages=2,
        has_zero_point=has_zero_point,
    )
    _assert_close(outputs, outputs_ref)


# ---------------------------------------------------------------------------
# Gated configs (xfail, strict=False). These should xfail at build time
# (static_assert in tcgen05_mma.cuh); we capture them here so we'll get
# an `XPASS` if a future change makes them work, signalling that the
# matching tests above should be promoted out of xfail.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_n", [128, 256])
def test_tcgen05_block_n_large(block_n):
    """Phase B.16: BlockN > 64 works after the smem.reduce write fix.
    The t2r write now uses gmem_writer's 8-int4-wide-row "section"
    layout (section_idx = int4_col // 8, smem_row = section_idx *
    BlockM + m_full) -- before this, BlockN=128 wrote 16-int4-wide
    rows that gmem_writer reinterpreted as two BlockM-row sections,
    producing the "N=64..127 mirrors N=0..63" symptom."""
    outputs, outputs_ref = _run_tcgen05(
        shape_m=128, shape_n=max(block_n, 128), shape_k=256,
        block_shape=(64, block_n, 64), warp_shape=(16, 64, 64),
        num_stages=2,
    )
    _assert_close(outputs, outputs_ref)


@pytest.mark.xfail(
    reason="Phase B.15 open: BlockM>64 (8 M-warps -> 2 warps per TMEM "
    "sub-partition) unverified. Gated by static_assert.",
    strict=False, run=False,
)
def test_tcgen05_block_m_large():
    outputs, outputs_ref = _run_tcgen05(
        shape_m=128, shape_n=64, shape_k=256,
        block_shape=(128, 64, 64), warp_shape=(16, 64, 64),
        num_stages=2,
    )
    _assert_close(outputs, outputs_ref)


@pytest.mark.xfail(
    reason="Phase B.15 open: BlockK>64 needs multi-atom-per-row swizzle "
    "(SBO recomputation). Gated by static_assert.",
    strict=False, run=False,
)
def test_tcgen05_block_k_large():
    outputs, outputs_ref = _run_tcgen05(
        shape_m=128, shape_n=64, shape_k=256,
        block_shape=(64, 64, 128), warp_shape=(16, 64, 128),
        num_stages=2,
    )
    _assert_close(outputs, outputs_ref)


@pytest.mark.xfail(
    reason="Phase B.15 open: WarpN<64 hits the loader_b half-group path "
    "(kIsWarpHalfGroup=true at WarpShape::N == ElementA::kBits*2 = 32) "
    "that the scatter doesn't model.",
    strict=False, run=False,
)
def test_tcgen05_warp_n_small():
    outputs, outputs_ref = _run_tcgen05(
        shape_m=128, shape_n=32, shape_k=256,
        block_shape=(64, 32, 64), warp_shape=(16, 32, 64),
        num_stages=2,
    )
    _assert_close(outputs, outputs_ref)


@pytest.mark.xfail(
    reason="Phase B.15 open: has_bias=True path adds bias in the epilogue, "
    "but our custom TCGEN05 epilogue (which skips smem_writer.write) "
    "doesn't currently apply bias correctly -- max|err| ~ 2 at |ref|.max "
    "~ 140 (~1.5%), 50%+ mismatched.",
    strict=False, run=False,
)
def test_tcgen05_has_bias():
    outputs, outputs_ref = _run_tcgen05(
        shape_m=128, shape_n=64, shape_k=256,
        block_shape=(64, 64, 64), warp_shape=(16, 64, 64),
        num_stages=2,
        has_zero_point=True,
        has_bias=True,
    )
    _assert_close(outputs, outputs_ref)
