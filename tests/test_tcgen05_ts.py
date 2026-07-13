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
  BlockN == 128, WarpN == 32, WarpM == BlockM in {32, 64, 128},
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
    b_dtype=dtypes.uint4,
    a_dtype=dtypes.bfloat16,
    is_fp_zero_point=False,
):
    # Activation dtype drives the whole f16-family compute path: scales,
    # zero-point dequant base, MMA operand format, and the drain convert
    # all follow it. c/bs match a_dtype for the W(u4)A16 path.
    c_dtype = a_dtype
    bs_dtype = a_dtype
    torch_a = torch.float16 if a_dtype == dtypes.float16 else torch.bfloat16

    torch.manual_seed(123)
    random_weight = generate_random_weight(
        n=shape_n, k=shape_k, group_size=group_size,
        dtype=b_dtype, scale_dtype=bs_dtype,
        has_zero_point=has_zero_point, is_fp_zero_point=is_fp_zero_point,
    )
    _, weight_ref, weight_codes, weight_scale, zero_point, _ = random_weight

    # FP zero point is a bf16 float, not folded into the weight repack
    # (the kernel subtracts it post-dequant). Pass None to the weight
    # repack and stream it as bf16 like the scale.
    repack_zp = None if is_fp_zero_point else zero_point
    weight = prepare_humming_weight(
        weight_codes, b_dtype, a_dtype, zero_point=repack_zp,
        use_wgmma=False, use_tcgen05_ts=True,
    )
    weight_scale_p = pack_scales_tcgen05_ts(weight_scale).cuda()
    zero_point_p = None
    if has_zero_point and is_fp_zero_point:
        zero_point_p = pack_scales_tcgen05_ts(zero_point).cuda()
    elif has_zero_point:
        zero_point_p = pack_zero_point_tcgen05_ts(
            zero_point.to(torch.int32), b_dtype.num_bits).cuda()

    _, inputs_ref, inputs, _ = generate_random_inputs(
        m=shape_m, k=shape_k, group_size=0, dtype=a_dtype,
    )
    bias = None
    if has_bias:
        torch.manual_seed(456)
        bias = torch.randn(shape_n, dtype=torch_a, device=inputs.device)

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
        is_fp_zero_point=is_fp_zero_point,
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

    if group_size == 0:
        # Channelwise: the kernel folds the per-row scale in the DRAIN,
        # i.e. it dequants (code - zp) to the a_dtype (exact for the
        # integer), K-sums, THEN multiplies each output column by the
        # a_dtype-rounded scale. Mirror that apply order so the reference
        # rounds identically (baking scale per-element as in the group
        # path would round in the wrong place).
        if has_zero_point:
            zp = zero_point.float().reshape(shape_n, 1)
        else:
            zp = float(2 ** (b_dtype.num_bits - 1))
        w_int = (weight_codes.float() - zp).to(torch_a).float()
        acc = inputs_ref.matmul(w_int.T)
        acc = acc * weight_scale.float().reshape(1, shape_n)
        outputs_ref = acc
        if has_bias:
            outputs_ref = outputs_ref + bias.float()
        outputs_ref = outputs_ref.to(torch_a)
    else:
        # The TS dequant rounds (code - zp) * scale to bf16 per element
        # (plain __hmul2, same as the SS path's bs application) -- round the
        # reference weights identically so the comparison stays tight at
        # large K. Against the fp32-weight reference the K=8192 prod shape
        # drifts to ~0.9 abs err (bf16 accumulation noise, cf. workbook
        # B.38); against this reference mean err is ~4e-4 with max 1 ulp.
        weight_ref = weight_ref.to(torch_a).float()
        outputs_ref = inputs_ref.matmul(weight_ref.T)
        if has_bias:
            outputs_ref = outputs_ref + bias.float()
        outputs_ref = outputs_ref.to(torch_a)
    torch.cuda.synchronize()

    from humming import ops
    outputs = torch.empty(
        (shape_m, shape_n), dtype=torch_a, device=inputs.device,
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


def _assert_close(outputs, outputs_ref, atol=0.5):
    abs_err = (outputs.float() - outputs_ref.float()).abs()
    ref_abs = outputs_ref.float().abs()
    print(
        f"\n  max|err|={abs_err.max().item():.3e} "
        f"mean|err|={abs_err.mean().item():.3e} "
        f"|ref|.mean={ref_abs.mean().item():.3e} "
        f"|ref|.max={ref_abs.max().item():.3e} atol={atol}"
    )
    torch.testing.assert_close(outputs, outputs_ref, rtol=1e-2, atol=atol)


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


# ---------------------------------------------------------------------------
# fp16 ACTIVATION (milestone 1, scalar-formats): a_dtype=float16. Weights
# dequant to fp16 (0x6400 base), fp16 scales, idesc a/b format F16 (still
# kind::f16), drain converts f32->fp16. Reference is the fp16-rounded-weight
# (0x6400 dequant) GEMM -- (code-zp)*scale rounded to fp16 per element,
# matching the kernel's half2 arithmetic. Output dtype must be float16.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape_m", [16, 128, 256, 512], ids=lambda m: f"m{m}")
@pytest.mark.parametrize("has_zero_point", [False, True])
def test_ts_fp16_uint4(has_zero_point, shape_m):
    block_m = 128 if shape_m >= 128 else 64
    outputs, outputs_ref = _run_ts(
        shape_m=shape_m, shape_n=512, shape_k=512,
        block_shape=(block_m, 128, 64),
        has_zero_point=has_zero_point, a_dtype=dtypes.float16,
    )
    assert outputs.dtype == torch.float16
    assert outputs_ref.dtype == torch.float16
    _assert_close(outputs, outputs_ref)


def test_ts_fp16_uint4_prod_shape():
    """fp16 A x uint4 on a Llama70B-gate slice at large K (fp16-accum tail)."""
    outputs, outputs_ref = _run_ts(
        shape_m=128, shape_n=1024, shape_k=8192,
        block_shape=(128, 128, 64), num_stages=4,
        has_zero_point=True, a_dtype=dtypes.float16,
    )
    assert outputs.dtype == torch.float16
    _assert_close(outputs, outputs_ref, atol=2.0)


# ---------------------------------------------------------------------------
# FP ZERO POINT (milestone 4, scalar-formats): is_fp_zero_point=True. The
# integer 0x4300|zp fold does NOT apply -- ts_dequant returns the raw code,
# transform_b subtracts a per-lane bf16 zp (a new bf16 operand streamed like
# the scale) BEFORE the group scale: (code - zp_fp) * scale, matching the SS
# order. Reference is the standard bf16-rounded-weight GEMM (the kernel and
# generate_random_weight round (code-zp) then *scale identically).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape_m", [16, 128, 256], ids=lambda m: f"m{m}")
def test_ts_fp_zero_point_uint4(shape_m):
    block_m = 128 if shape_m >= 128 else 64
    outputs, outputs_ref = _run_ts(
        shape_m=shape_m, shape_n=512, shape_k=512,
        block_shape=(block_m, 128, 64),
        has_zero_point=True, is_fp_zero_point=True,
    )
    _assert_close(outputs, outputs_ref)


@pytest.mark.parametrize("group_size", [32, 128], ids=["gs32", "gs128"])
def test_ts_fp_zero_point_gs(group_size):
    """fp zp composes with group scale at gs=128 and sub-stage gs=32."""
    outputs, outputs_ref = _run_ts(
        shape_m=256, shape_n=512, shape_k=512,
        block_shape=(128, 128, 64), group_size=group_size,
        has_zero_point=True, is_fp_zero_point=True,
    )
    _assert_close(outputs, outputs_ref)


def test_ts_fp_zero_point_prod_shape():
    outputs, outputs_ref = _run_ts(
        shape_m=128, shape_n=1024, shape_k=8192,
        block_shape=(128, 128, 64), num_stages=4,
        has_zero_point=True, is_fp_zero_point=True,
    )
    _assert_close(outputs, outputs_ref, atol=2.0)


# ---------------------------------------------------------------------------
# gs=32 SUB-STAGE weight scale (milestone 3, scalar-formats): a BlockK=64
# stage spans two 32-K groups. Each 16-K MMA iter stays within one group
# (16 | 32), so the scale is still folded per-code in transform_b -- only
# the per-iter group index changes (iter/2). Reference is the standard
# per-element (code-zp)*scale bf16-rounded-weight GEMM (matches).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape_m", [16, 128, 256], ids=lambda m: f"m{m}")
@pytest.mark.parametrize("has_zero_point", [False, True])
def test_ts_gs32_uint4(has_zero_point, shape_m):
    block_m = 128 if shape_m >= 128 else 64
    outputs, outputs_ref = _run_ts(
        shape_m=shape_m, shape_n=512, shape_k=512,
        block_shape=(block_m, 128, 64),
        group_size=32, has_zero_point=has_zero_point,
    )
    _assert_close(outputs, outputs_ref)


def test_ts_gs32_uint4_prod_shape():
    outputs, outputs_ref = _run_ts(
        shape_m=128, shape_n=1024, shape_k=8192,
        block_shape=(128, 128, 64), num_stages=4,
        group_size=32, has_zero_point=True,
    )
    _assert_close(outputs, outputs_ref, atol=2.0)


# ---------------------------------------------------------------------------
# CHANNELWISE weight scale (milestone 2, scalar-formats): group_size=0, one
# bf16 scale per output row over all K. Applied via epilogue-fold in the TS
# drain (commutes with the K-sum) rather than per-code in transform_b. zp is
# staged once in bzp_c. Reference matches the kernel's apply order: dequant
# (code-zp) to bf16, K-sum, THEN multiply the row by the bf16 scale.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape_m", [16, 128, 256], ids=lambda m: f"m{m}")
@pytest.mark.parametrize("has_zero_point", [False, True])
def test_ts_channelwise_uint4(has_zero_point, shape_m):
    block_m = 128 if shape_m >= 128 else 64
    outputs, outputs_ref = _run_ts(
        shape_m=shape_m, shape_n=512, shape_k=512,
        block_shape=(block_m, 128, 64),
        group_size=0, has_zero_point=has_zero_point,
    )
    _assert_close(outputs, outputs_ref)


def test_ts_channelwise_uint4_prod_shape():
    outputs, outputs_ref = _run_ts(
        shape_m=128, shape_n=1024, shape_k=8192,
        block_shape=(128, 128, 64), num_stages=4,
        group_size=0, has_zero_point=True,
    )
    _assert_close(outputs, outputs_ref, atol=2.0)


# ---------------------------------------------------------------------------
# uint2 weight dtype (milestone b): same integer uint_to_f16 path, kWpr=1
# half-group loader, no-zp midpoint 2 vs int-zp. Reference is the same
# bf16-rounded-weight dequant GEMM.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("has_zero_point", [False, True])
def test_ts_uint2_512_cubed(has_zero_point):
    outputs, outputs_ref = _run_ts(
        shape_m=512, shape_n=512, shape_k=512,
        has_zero_point=has_zero_point, b_dtype=dtypes.uint2,
    )
    _assert_close(outputs, outputs_ref)


def test_ts_uint2_block_m128_fatn():
    """uint2 on a fat-N gate/up slice with BlockM=128, multi-block N/K."""
    outputs, outputs_ref = _run_ts(
        shape_m=256, shape_n=1024, shape_k=1024,
        block_shape=(128, 128, 64), b_dtype=dtypes.uint2,
    )
    _assert_close(outputs, outputs_ref)


def test_ts_uint2_minimal_tile():
    """N=128 K=64 single minimal tile, no zp."""
    outputs, outputs_ref = _run_ts(
        shape_m=128, shape_n=128, shape_k=64,
        group_size=64, has_zero_point=False, b_dtype=dtypes.uint2,
    )
    _assert_close(outputs, outputs_ref)


# ---------------------------------------------------------------------------
# float4e2m1 weight dtype (milestone c, headline production dtype): fp_to_fp
# decode + a constant 2^126 (get_dtype_dequant_exp_offset<bf16,fp4>) weight
# mul; no zero point, no epilogue exp-offset. The group-scale hmul2 in
# transform_b applies unchanged. Reference: same bf16-rounded dequant GEMM.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape_m,shape_n,shape_k",
    [(512, 512, 512), (128, 1024, 1024), (128, 128, 64)],
)
def test_ts_fp4_shapes(shape_m, shape_n, shape_k):
    gs = 64 if shape_k == 64 else 128
    outputs, outputs_ref = _run_ts(
        shape_m=shape_m, shape_n=shape_n, shape_k=shape_k,
        group_size=gs, has_zero_point=False, b_dtype=dtypes.float4e2m1,
        block_shape=(128, 128, 64) if shape_m >= 128 else (64, 128, 64),
    )
    _assert_close(outputs, outputs_ref)


def test_ts_fp4_prod_shape():
    """Llama70B-gate slice, fat-N/K walk at large K (bf16-accum tail)."""
    outputs, outputs_ref = _run_ts(
        shape_m=128, shape_n=1024, shape_k=8192,
        block_shape=(128, 128, 64), num_stages=4,
        has_zero_point=False, b_dtype=dtypes.float4e2m1,
    )
    _assert_close(outputs, outputs_ref, atol=2.0)


# ---------------------------------------------------------------------------
# uint8 (milestone d): normalized_uint_to_fp (8 > bf16 mantissa) + a split
# 2^133 exp offset (2^127 lifts the subnormal dequant to normal range, then
# 2^6), 8-bit zp byte-extract, regs_qb[2][4] int4 load. SMEM fits at
# BlockK=64 stages=4.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("has_zero_point", [False, True])
def test_ts_uint8_512_cubed(has_zero_point):
    outputs, outputs_ref = _run_ts(
        shape_m=512, shape_n=512, shape_k=512,
        has_zero_point=has_zero_point, b_dtype=dtypes.uint8,
    )
    _assert_close(outputs, outputs_ref)


def test_ts_uint8_block_m128():
    outputs, outputs_ref = _run_ts(
        shape_m=256, shape_n=1024, shape_k=1024,
        block_shape=(128, 128, 64), b_dtype=dtypes.uint8,
    )
    _assert_close(outputs, outputs_ref)


def test_ts_uint8_prod_shape():
    outputs, outputs_ref = _run_ts(
        shape_m=128, shape_n=1024, shape_k=8192,
        block_shape=(128, 128, 64), num_stages=4, b_dtype=dtypes.uint8,
    )
    _assert_close(outputs, outputs_ref, atol=2.0)


# ---------------------------------------------------------------------------
# float8e4m3 (milestone d): fp_to_fp decode + a single 2^120 exp offset
# (<=127, fully in the mainloop weight); no zp; regs_qb[2][4].
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape_m,shape_n,shape_k", [(512, 512, 512), (256, 1024, 1024)],
)
def test_ts_fp8_shapes(shape_m, shape_n, shape_k):
    outputs, outputs_ref = _run_ts(
        shape_m=shape_m, shape_n=shape_n, shape_k=shape_k,
        has_zero_point=False, b_dtype=dtypes.float8e4m3,
        block_shape=(128, 128, 64) if shape_m >= 128 else (64, 128, 64),
    )
    _assert_close(outputs, outputs_ref)


def test_ts_fp8_prod_shape():
    outputs, outputs_ref = _run_ts(
        shape_m=128, shape_n=1024, shape_k=8192,
        block_shape=(128, 128, 64), num_stages=4,
        has_zero_point=False, b_dtype=dtypes.float8e4m3,
    )
    _assert_close(outputs, outputs_ref, atol=2.0)


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


# ---------------------------------------------------------------------------
# End-to-end opt-in through the HummingLayer wrapper: mma_type="tcgen05"
# threads meta.mma_type=TCGEN05 -> TS packing (transform) -> TS kernel
# (get_heuristics_config). Compared against the trusted default
# (mma.sync/SS) path on the SAME quantized weights -- both dequant the
# same codes, so they agree to bf16 noise.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape_m", [64, 256])
def test_ts_layer_opt_in_matches_default(shape_m):
    from humming.config import MmaType
    from humming.layer import HummingLayer
    from humming.schema.humming import HummingWeightSchema

    shape_n, shape_k, group_size = 512, 512, 128
    schema = HummingWeightSchema(
        b_dtype=dtypes.uint4, bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=group_size, has_zero_point=True,
    )

    def build(mma_type):
        torch.manual_seed(7)
        w = torch.randn(shape_n, shape_k, dtype=torch.bfloat16,
                        device="cuda") / (shape_k ** 0.5)
        layer = HummingLayer(
            shape_n=shape_n, shape_k=shape_k, weight_config=schema,
            torch_dtype=torch.bfloat16, mma_type=mma_type,
        ).cuda()
        layer.load_from_unquantized(w)
        layer.transform()
        return layer

    layer_default = build(None)
    layer_ts = build("tcgen05")
    assert layer_default.humming_metas[""].mma_type == MmaType.MMA
    assert layer_ts.humming_metas[""].mma_type == MmaType.TCGEN05

    torch.manual_seed(11)
    x = torch.randn(shape_m, shape_k, dtype=torch.bfloat16,
                    device="cuda") / (shape_k ** 0.5)
    out_default = layer_default.forward(x.clone())
    out_ts = layer_ts.forward(x.clone())
    torch.cuda.synchronize()
    _assert_close(out_ts, out_default)


def test_ts_layer_illegal_shape_rejected():
    """mma_type=tcgen05 on a TS-illegal shape (N not %128) must fail at
    transform, not silently mispack."""
    from humming.layer import HummingLayer
    from humming.schema.humming import HummingWeightSchema

    schema = HummingWeightSchema(
        b_dtype=dtypes.uint4, bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=128, has_zero_point=True,
    )
    w = torch.randn(192, 512, dtype=torch.bfloat16, device="cuda") / (512 ** 0.5)
    layer = HummingLayer(
        shape_n=192, shape_k=512, weight_config=schema,
        torch_dtype=torch.bfloat16, mma_type="tcgen05",
    ).cuda()
    layer.load_from_unquantized(w)
    with pytest.raises(AssertionError, match="TS-legal"):
        layer.transform()
