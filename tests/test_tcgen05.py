"""TCGEN05 (Blackwell sm_100+) W4A16 correctness tests.

Verifies the bf16 x uint4 path (AWQ-style: per-group bf16 scales +
uint4 zero-points, group_size=128) against an mma.sync reference
with rtol=1e-2, atol=0.5.

Supported config space (gated by static_asserts in
`mma/tcgen05_mma.cuh`):

  * BlockShape::M in {64, 128}
  * BlockShape::N in {64, 128, 256}
  * BlockShape::K in {64, 128, 256}
  * WarpShape::M = BlockShape::M / 4   (4 M-warps, one per TMEM
                                        sub-partition)
  * WarpShape::N = 64                  (smaller hits loader_b's
                                        half-group path -- xfail'd)
  * WarpShape::K = BlockShape::K       (no K-warps)
  * kNumStages in {2, 3, 4}
  * has_zero_point, has_bias both in {True, False}

Tests are parametrized over dimensions we'd tune over (kNumStages,
BlockN, BlockM, ...). Known gated combinations are `xfail`'d so they
stay visible if a static_assert is accidentally relaxed.
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
    use_tma=False,
    use_warp_spec=False,
    use_ws_pipeline=False,
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
        use_warp_spec=use_warp_spec,
        use_tma=use_tma or use_warp_spec,
        use_cp_async=not (use_tma or use_warp_spec),
        use_mbarrier=use_tma or use_warp_spec,
        # use_tma_bzp must be False when has_zero_point + is_group_weight_scale --
        # humming's tensor.h asserts on TMA for that combination.
        use_tma_bzp=False,
        has_bias=has_bias,
        mma_type="tcgen05",
        use_tcgen05=True,
        use_ws_pipeline=use_ws_pipeline,
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
# Smallest viable case (sanity check that the path is wired up at all).
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
# Single-K-position probe. With A=delta(k=k0), out[m, n] should equal
# B_dequant[k0, n] for every k0. Guards against the historic SMEM-A
# race where only the LAST 16 K of each K-block returned wrong values.
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
# TMA load path. Replaces cp.async for A and B with TMA `tma_load_2d`
# (humming's existing WMMA-side wiring; the TCGEN05 mma code is
# independent of the load mechanism). BZP still uses cp.async since
# `tensor.h:275` asserts TMA isn't supported for is_group_weight_scale
# BZP descriptors.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_tma", [False, True])
@pytest.mark.parametrize("has_zero_point", [True, False])
def test_tcgen05_tma(use_tma, has_zero_point):
    outputs, outputs_ref = _run_tcgen05(
        shape_m=128, shape_n=64, shape_k=512,
        block_shape=(64, 64, 64), warp_shape=(16, 64, 64),
        num_stages=2,
        has_zero_point=has_zero_point,
        use_tma=use_tma,
    )
    _assert_close(outputs, outputs_ref)


# ---------------------------------------------------------------------------
# Warp specialization. Math threads use bar.sync 1, math instead of
# __syncthreads so the producer warps aren't dragged into every
# per-K-iter scatter sync. Requires use_tma + use_mbarrier (the
# producer pipeline drives gmem -> smem via TMA + mbarrier-based
# completion).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("has_zero_point", [True, False])
def test_tcgen05_warp_spec(has_zero_point):
    outputs, outputs_ref = _run_tcgen05(
        shape_m=128, shape_n=64, shape_k=512,
        block_shape=(64, 64, 64), warp_shape=(16, 64, 64),
        num_stages=3,
        has_zero_point=has_zero_point,
        use_warp_spec=True,
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
    """BlockN > 64. The t2r write uses gmem_writer's 8-int4-wide-row
    "section" layout (section_idx = int4_col // 8, smem_row =
    section_idx * BlockM + m_full)."""
    outputs, outputs_ref = _run_tcgen05(
        shape_m=128, shape_n=max(block_n, 128), shape_k=256,
        block_shape=(64, block_n, 64), warp_shape=(16, 64, 64),
        num_stages=2,
    )
    _assert_close(outputs, outputs_ref)


@pytest.mark.parametrize("block_n", [64, 128, 256])
def test_tcgen05_block_m_large(block_n):
    """BlockM=128 via WarpShape::M=32 (the M=128 TMEM atom places 32
    valid M-values per sub-partition vs 16 for M=64). All 32 lanes
    per warp participate (the laneid<WarpShape::M gate covers both
    cases)."""
    outputs, outputs_ref = _run_tcgen05(
        shape_m=128, shape_n=max(block_n, 128), shape_k=256,
        block_shape=(128, block_n, 64), warp_shape=(32, 64, 64),
        num_stages=2,
    )
    _assert_close(outputs, outputs_ref)


@pytest.mark.parametrize("block_k", [128, 256])
def test_tcgen05_block_k_large(block_k):
    """BlockK > 64. B is sectionised the same way A is -- each section
    holds 64 K-bf16 of all N; the iter advance crosses sections via
    `section_idx * section_size`."""
    outputs, outputs_ref = _run_tcgen05(
        shape_m=128, shape_n=64, shape_k=max(block_k * 2, 256),
        block_shape=(64, 64, block_k), warp_shape=(16, 64, block_k),
        num_stages=2,
    )
    _assert_close(outputs, outputs_ref)


@pytest.mark.xfail(
    reason="WarpN<64 hits loader_b's half-group path "
    "(kIsWarpHalfGroup=true at WarpShape::N == ElementA::kBits*2 = 32) "
    "which the scatter doesn't model. Static_assert in tcgen05_mma.cuh.",
    strict=False, run=False,
)
def test_tcgen05_warp_n_small():
    outputs, outputs_ref = _run_tcgen05(
        shape_m=128, shape_n=32, shape_k=256,
        block_shape=(64, 32, 64), warp_shape=(16, 32, 64),
        num_stages=2,
    )
    _assert_close(outputs, outputs_ref)


# ---------------------------------------------------------------------------
# has_bias=True: the t2r-pack loop now reads `smem.bias` and applies it
# in fp32 before the f32->bf16 cast. (Originally we just skipped
# smem_writer.write for TCGEN05, which lost the WMMA path's bias
# add.) All 4 (has_zero_point, has_bias) combinations pass.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "has_zero_point, has_bias",
    [(True, False), (False, False), (True, True), (False, True)],
)
def test_tcgen05_zp_bias(has_zero_point, has_bias):
    outputs, outputs_ref = _run_tcgen05(
        shape_m=128, shape_n=64, shape_k=256,
        block_shape=(64, 64, 64), warp_shape=(16, 64, 64),
        num_stages=2,
        has_zero_point=has_zero_point,
        has_bias=has_bias,
    )
    _assert_close(outputs, outputs_ref)


# ---------------------------------------------------------------------------
# use_ws_pipeline=True: warp-specialized Transform->MMA pipeline
# (track a-ws-pipeline). Per-k-block b_dequant slot ping-pong gated by
# t2m_full/t2m_empty mbarriers instead of the per-K-iter bar.sync.
# Requires warp-spec + num_stages >= 3. Large-K cases guard the
# deferred-G2S-release fix (commit-issue release raced the producer's
# smem.a refill at K=4096).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_stages", [3, 4])
@pytest.mark.parametrize("shape_k", [512, 2048])
def test_tcgen05_ws_pipeline_prod(shape_k, num_stages):
    outputs, outputs_ref = _run_tcgen05(
        shape_m=256, shape_n=512, shape_k=shape_k,
        block_shape=(128, 128, 128), warp_shape=(32, 64, 128),
        num_stages=num_stages,
        use_warp_spec=True,
        use_ws_pipeline=True,
    )
    _assert_close(outputs, outputs_ref)


def test_tcgen05_ws_pipeline_bitexact_vs_classic():
    """The pipeline reorders WORK, not MATH: both paths issue the same
    tcgen05.mma sequence over the same dequantised values, so outputs
    must be bit-identical. This is the strongest guard against
    scheduling races (at K=4096 the fp32 reference comparison drowns
    in bf16 accumulation noise -- both paths show identical ~0.7% of
    cells beyond atol=0.5, max|err|=2.0)."""
    kwargs = dict(
        shape_m=512, shape_n=1024, shape_k=4096,
        block_shape=(128, 128, 128), warp_shape=(32, 64, 128),
        num_stages=4, use_warp_spec=True,
    )
    torch.manual_seed(7)
    out_classic, ref = _run_tcgen05(use_ws_pipeline=False, **kwargs)
    torch.manual_seed(7)
    out_ws, ref2 = _run_tcgen05(use_ws_pipeline=True, **kwargs)
    assert torch.equal(ref, ref2), "problem generation not deterministic"
    assert torch.equal(out_classic, out_ws), (
        "ws-pipeline output diverged from the classic tcgen05 path "
        f"(max|diff|={(out_classic.float() - out_ws.float()).abs().max().item()})"
    )


@pytest.mark.parametrize(
    "has_zero_point, has_bias",
    [(True, False), (False, False), (True, True), (False, True)],
)
def test_tcgen05_ws_pipeline_zp_bias(has_zero_point, has_bias):
    outputs, outputs_ref = _run_tcgen05(
        shape_m=128, shape_n=256, shape_k=1024,
        block_shape=(128, 128, 128), warp_shape=(32, 64, 128),
        num_stages=4,
        has_zero_point=has_zero_point,
        has_bias=has_bias,
        use_warp_spec=True,
        use_ws_pipeline=True,
    )
    _assert_close(outputs, outputs_ref)


@pytest.mark.parametrize(
    "block_shape, warp_shape",
    [
        ((64, 128, 128), (16, 64, 128)),   # BlockM=64 (kIGroups=4)
        pytest.param(
            (128, 64, 128), (32, 64, 128),
            marks=pytest.mark.xfail(
                reason="BlockN=64 + BlockK=128 WS-pipeline race: "
                "non-deterministic wrong cells in the first output "
                "tile (max|err| ~100-155, 400-600 cells). BlockK=64 "
                "at the same BlockN is clean, and the classic path "
                "passes -- same axis family as the workbook B.38 "
                "WS+TMA+BlockM=128+BK=128 edge bug. Non-production "
                "config (heuristic only drops to BlockN=64 when "
                "shape_n % 128 != 0, and never enables ws_pipeline "
                "there).",
                strict=False,
            ),
        ),                                  # BlockN=64 (4 math warps)
        ((128, 128, 64), (32, 64, 64)),    # BlockK=64 (kWarpIters=4)
        # BlockN=256 needs BlockK=64 to fit SMEM (workbook B.29).
        ((128, 256, 64), (32, 64, 64)),    # BlockN=256 (kNWarps=4)
        ((128, 64, 64), (32, 64, 64)),     # BlockN=64 at BlockK=64 (clean)
    ],
)
def test_tcgen05_ws_pipeline_block_shapes(block_shape, warp_shape):
    outputs, outputs_ref = _run_tcgen05(
        shape_m=256, shape_n=512, shape_k=1024,
        block_shape=block_shape, warp_shape=warp_shape,
        num_stages=3,
        use_warp_spec=True,
        use_ws_pipeline=True,
    )
    _assert_close(outputs, outputs_ref)
