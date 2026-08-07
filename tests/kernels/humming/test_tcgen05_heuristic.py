import pytest

from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType, WeightScale2Type, WeightScaleType
from humming.testing import skip_if_unsupported
from humming.tune import get_heuristics_class, get_heuristics_config

SHAPE_N = 14336
SHAPE_K = 4096


def _layer_config(**overrides) -> LayerConfig:
    values = {
        "shape_n": SHAPE_N,
        "shape_k": SHAPE_K,
        "a_dtype": dtypes.bfloat16,
        "b_dtype": dtypes.uint4,
        "c_dtype": dtypes.bfloat16,
        "bs_dtype": dtypes.bfloat16,
        "weight_scale_group_size": 128,
        "has_zero_point": True,
    }
    return LayerConfig(**(values | overrides))


@pytest.fixture(autouse=True)
def _requires_tcgen05():
    skip_if_unsupported(mma_type="tcgen05")


def test_heuristics_class_resolves_to_sm100():
    assert get_heuristics_class().sm_version == 100


@pytest.mark.parametrize("shape_m,block_m", [(1, 64), (17, 64), (127, 64), (128, 128), (4096, 128)])
def test_ts_opt_in_config(shape_m, block_m):
    """A tcgen05 layer runs TS at every shape_m; only the M tile moves."""
    config = get_heuristics_config(_layer_config(mma_type=MmaType.TCGEN05), shape_m=shape_m)
    assert config["mma_type"] == "tcgen05"
    assert config["use_tcgen05"] is True
    assert config["use_tcgen05_ts"] is True
    assert config["block_shape"] == (block_m, 128, 64)
    assert config["warp_shape"] == (block_m, 32, 64)
    assert config["num_stages"] >= 3
    assert config["use_warp_spec"] is True
    assert config["use_tma"] is True
    # The TS epilogue drains TMEM straight into the TMA-C store and has no
    # cross-CTA partial-K reduction, and the launcher rejects a TMA BZP
    # descriptor for a group weight scale with a zero point.
    assert config["use_stream_k"] is False
    assert config["use_tma_bzp"] is False
    assert config["raster_group_m"] == 1


@pytest.mark.parametrize(
    "overrides,num_stages",
    [
        # uint4's zero-point-folded uint_to_f16 hides a fifth stage; uint8+zp,
        # the one bf16 arm that splits its exponent offset across two
        # multiplies, peaks at three and loses 8% by five.
        ({}, 5),
        ({"has_zero_point": False}, 4),
        ({"b_dtype": dtypes.uint8}, 3),
        ({"b_dtype": dtypes.uint8, "has_zero_point": False}, 4),
        ({"b_dtype": dtypes.uint2}, 4),
        ({"b_dtype": dtypes.float4e2m1, "has_zero_point": False}, 4),
        # An fp zero point is a post-dequant subtract rather than a folded
        # bias, so it takes the zero-point-free depth on both widths.
        ({"is_fp_zero_point": True}, 4),
        ({"b_dtype": dtypes.uint8, "is_fp_zero_point": True}, 4),
        # ElementA picks the arm, not the depth: fp16 uint4 measures the same.
        ({"a_dtype": dtypes.float16, "c_dtype": dtypes.float16, "bs_dtype": dtypes.float16}, 5),
    ],
    ids=str,
)
def test_ts_num_stages_is_per_dequant_arm(overrides, num_stages):
    """TS pipeline depth is tuned per weight dtype, not capped globally."""
    config = get_heuristics_config(_layer_config(mma_type=MmaType.TCGEN05, **overrides), shape_m=2048)
    assert config["num_stages"] == num_stages


def test_ts_grouped_keeps_the_default_pipeline():
    """Grouped tiles a short K per expert; both tuned dense depths lose
    1.6-3.8% at E=8, so the grouped scheduler keeps four stages."""
    config = get_heuristics_config(
        _layer_config(mma_type=MmaType.TCGEN05, num_experts=8),
        shape_m=1024,
        gemm_type=GemmType.GROUPED_CONTIGUOUS,
    )
    assert config["use_tcgen05_ts"] is True
    assert config["num_stages"] == 4


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        # Fat-K, few-output-tile and wide-dtype layers: the whole M >= 128
        # window that SS used to take over before selection became opt-in.
        {"shape_n": 8192, "shape_k": 28672},
        {"shape_n": 6144},
        {"b_dtype": dtypes.uint3},
        {"b_dtype": dtypes.uint8},
    ],
    ids=str,
)
@pytest.mark.parametrize("shape_m", [1, 16, 64, 128, 256, 512, 1024, 2048, 4096])
def test_default_layer_never_selects_tcgen05(overrides, shape_m):
    """Without the opt-in sm100 resolves to the mma.sync config it always did."""
    config = get_heuristics_config(_layer_config(**overrides), shape_m=shape_m)
    assert config.get("mma_type") is None
    assert not config.get("use_tcgen05")
    assert not config.get("use_tcgen05_ts")


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"weight_scale_group_size": 0, "has_zero_point": False},
        {"weight_scale_group_size": 16},
        {"weight_scale_group_size": 32},
        {"weight_scale_group_size": 64},
        {"b_dtype": dtypes.uint2},
        {"b_dtype": dtypes.uint8},
        {"b_dtype": dtypes.float4e2m1, "has_zero_point": False},
        {"b_dtype": dtypes.float8e4m3, "has_zero_point": False},
        {"is_fp_zero_point": True},
        {"b_dtype": dtypes.uint8, "is_fp_zero_point": True},
        {"b_dtype": dtypes.float8e5m2, "has_zero_point": False},
        {"b_dtype": dtypes.DataType.from_str("float4e3m0"), "has_zero_point": False},
        {"b_dtype": dtypes.DataType.from_str("float8e1m6"), "has_zero_point": False},
        # fp16 activations: bs_dtype and c_dtype track a_dtype.
        {"a_dtype": dtypes.float16, "c_dtype": dtypes.float16, "bs_dtype": dtypes.float16},
        {
            "a_dtype": dtypes.float16,
            "c_dtype": dtypes.float16,
            "bs_dtype": dtypes.float16,
            "b_dtype": dtypes.uint8,
        },
        {
            "a_dtype": dtypes.float16,
            "c_dtype": dtypes.float16,
            "bs_dtype": dtypes.float16,
            "is_fp_zero_point": True,
        },
        # 8-bit group scales, decoded to ElementA at the s2r load.
        {"bs_dtype": dtypes.float8e8m0, "weight_scale_group_size": 32},
        {"bs_dtype": dtypes.float8e4m3},
        {"a_dtype": dtypes.float16, "c_dtype": dtypes.float16, "bs_dtype": dtypes.float8e4m3},
        {"num_experts": 256},
        {"shape_n": 512, "shape_k": 512},
    ],
    ids=str,
)
def test_ts_legal_layers(overrides):
    config = _layer_config(**overrides)
    assert config.tcgen05_supported
    assert get_heuristics_class().supports_tcgen05_ts(config)


@pytest.mark.parametrize(
    "overrides",
    [
        # BlockN is 128 and BlockK is 64, so the problem shape must tile.
        {"shape_n": 192},
        {"shape_k": 4128, "weight_scale_group_size": 32},
        # A 16-K MMA iteration must stay inside one weight-scale group.
        {"weight_scale_group_size": 48, "shape_k": 4128},
        # Weight dtypes with no ts_dequant_b_pair arm (the TS packer holds
        # only 32 % num_bits == 0 widths).
        {"b_dtype": dtypes.uint6},
        {"b_dtype": dtypes.DataType.from_str("float6e4m1"), "has_zero_point": False},
        # TS activations are the two 16-bit float dtypes.
        {"a_dtype": dtypes.float8e4m3, "b_dtype": dtypes.float8e4m3, "has_zero_point": False},
        # A 16-bit group scale is reinterpreted as ElementA bit-for-bit, so it
        # must be a_dtype; e8m0 shares its exponent bias with bf16 only.
        {"a_dtype": dtypes.float16, "c_dtype": dtypes.float16},
        {"bs_dtype": dtypes.float16},
        {
            "a_dtype": dtypes.float16,
            "c_dtype": dtypes.float16,
            "bs_dtype": dtypes.float8e8m0,
            "weight_scale_group_size": 32,
        },
        # A channelwise scale is folded into the drain as ElementBS.
        {
            "bs_dtype": dtypes.float8e4m3,
            "weight_scale_group_size": 0,
            "has_zero_point": False,
        },
        # The launcher types the fp zero point as c_dtype but s2r reads it as
        # ElementA.
        {"a_dtype": dtypes.float16, "bs_dtype": dtypes.float16, "is_fp_zero_point": True},
        # weight_scale_2 is applied in the epilogue smem writer, which the TMEM
        # drain bypasses -- admitting it would drop the scale silently.
        {"weight_scale_2_type": WeightScale2Type.CHANNEL},
        {"weight_scale_2_type": WeightScale2Type.TENSOR},
    ],
    ids=str,
)
def test_ts_illegal_layers(overrides):
    config = _layer_config(**overrides)
    assert not config.tcgen05_supported
    assert not get_heuristics_class().supports_tcgen05_ts(config)


@pytest.mark.parametrize(
    "overrides",
    [
        # Weight dtypes with no ts_dequant_b_pair arm, one per dequant family.
        {"b_dtype": dtypes.uint1},
        {"b_dtype": dtypes.uint7},
        {"b_dtype": dtypes.DataType.from_str("float3e2m0"), "has_zero_point": False},
        {"b_dtype": dtypes.DataType.from_str("float7e6m0"), "has_zero_point": False},
        # 8-bit group scales are dequantised into bf16 by the generic mainloop.
        # Pinned to a weight dtype TS has no arm for, so they reach SS.
        {"b_dtype": dtypes.uint7, "bs_dtype": dtypes.float8e8m0, "weight_scale_group_size": 32},
        {"b_dtype": dtypes.uint7, "bs_dtype": dtypes.float8e4m3},
        # A block scale is one f32 per warp N-tile, and SS pins WarpN at 64.
        {
            "bs_dtype": dtypes.float32,
            "weight_scale_type": WeightScaleType.BLOCK,
            "weight_scale_group_size": 64,
            "weight_scale_group_size_n": 64,
            "has_zero_point": False,
        },
    ],
    ids=str,
)
def test_ss_legal_layers(overrides):
    config = _layer_config(**overrides)
    assert not config.tcgen05_supported
    assert get_heuristics_class().supports_tcgen05_ss(config)


@pytest.mark.parametrize(
    "overrides",
    [
        # Applied on C by the epilogue smem writer both drains bypass.
        {"weight_scale_2_type": WeightScale2Type.CHANNEL},
        {"weight_scale_2_type": WeightScale2Type.TENSOR},
        {"weight_scale_group_size": 0, "has_zero_point": False},
        {
            "bs_dtype": dtypes.float32,
            "weight_scale_type": WeightScaleType.TENSOR,
            "weight_scale_group_size": 0,
            "has_zero_point": False,
        },
        # A 16-bit group scale is reinterpreted as ElementA bit-for-bit.
        {"bs_dtype": dtypes.float16},
        # A block scale narrower than the warp N-tile is not indexed per N.
        {
            "bs_dtype": dtypes.float32,
            "weight_scale_type": WeightScaleType.BLOCK,
            "weight_scale_group_size": 64,
            "weight_scale_group_size_n": 32,
            "has_zero_point": False,
        },
    ],
    ids=str,
)
def test_ss_illegal_layers(overrides):
    config = _layer_config(b_dtype=dtypes.uint3, **overrides)
    assert not get_heuristics_class().supports_tcgen05_ss(config)


@pytest.mark.parametrize(
    "overrides",
    [
        # Neither tuning table carries this weight dtype.
        {"b_dtype": dtypes.float6e3m2, "has_zero_point": False},
        # BlockN bottoms out at 64 and BlockK at 64, so the shape must tile.
        {"shape_n": SHAPE_N + 32},
        {"shape_k": SHAPE_K + 32, "weight_scale_group_size": 32},
        # A BlockK stage must hold whole weight-scale groups.
        {"weight_scale_group_size": 48, "shape_k": SHAPE_K + 32},
        # SS applies only the scale kinds the mainloop carries, and TS needs a
        # wired dequant arm.
        {"b_dtype": dtypes.uint7, "weight_scale_group_size": 0, "has_zero_point": False},
        {"b_dtype": dtypes.uint7, "weight_scale_2_type": WeightScale2Type.CHANNEL},
        {"weight_scale_2_type": WeightScale2Type.TENSOR},
        # SS is bf16-only, so an fp16 layer TS turns down has no fallback. Both
        # of these run to completion and return garbage without their gate
        # clause: the TS s2r branch reads a 16-bit group scale and the fp zero
        # point as ElementA whatever the launcher typed them as.
        {"a_dtype": dtypes.float16, "c_dtype": dtypes.float16},
        {
            "a_dtype": dtypes.float16,
            "bs_dtype": dtypes.float16,
            "is_fp_zero_point": True,
        },
    ],
    ids=str,
)
def test_opt_in_rejects_illegal_layer(overrides):
    config = _layer_config(mma_type=MmaType.TCGEN05, **overrides)
    with pytest.raises(AssertionError, match="legal for neither"):
        get_heuristics_config(config, shape_m=512)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"gemm_type": GemmType.INDEXED}, "does not support"),
        ({"use_f16_accum": True}, "f32 TMEM"),
        ({"use_batch_invariant": True}, "batch-invariant"),
    ],
    ids=["indexed", "f16-accum", "batch-invariant"],
)
def test_ts_opt_in_rejects_unsupported_compute(kwargs, message):
    config = _layer_config(mma_type=MmaType.TCGEN05)
    with pytest.raises(AssertionError, match=message):
        get_heuristics_config(config, shape_m=512, **kwargs)


@pytest.mark.parametrize(
    "overrides,block_shape",
    [
        # Weight dtypes with no ts_dequant_b_pair arm.
        ({"b_dtype": dtypes.uint3}, (128, 128, 128)),
        ({"b_dtype": dtypes.uint6}, (128, 128, 128)),
        ({"b_dtype": dtypes.uint7}, (128, 128, 128)),
        ({"b_dtype": dtypes.DataType.from_str("float6e4m1"), "has_zero_point": False}, (128, 128, 128)),
        # shape_n only tiles at BlockN=64, which TS does not accept.
        ({"shape_n": 6208}, (128, 64, 128)),
        # shape_k only tiles at BlockK=64.
        ({"b_dtype": dtypes.uint3, "shape_k": 4160}, (128, 128, 64)),
        # uint8 needs the narrow K tile to keep the b_dequant buffer in SMEM.
        ({"b_dtype": dtypes.uint8, "shape_n": 6208}, (128, 64, 64)),
    ],
    ids=str,
)
@pytest.mark.parametrize("shape_m", [1, 128, 2048])
def test_ss_opt_in_fallback(overrides, block_shape, shape_m):
    """An opted-in layer TS is illegal for runs SS at every shape_m; the same
    layer without the opt-in stays on mma.sync."""
    config = get_heuristics_config(_layer_config(mma_type=MmaType.TCGEN05, **overrides), shape_m=shape_m)
    assert config["mma_type"] == "tcgen05"
    assert config["use_tcgen05"] is True
    assert not config.get("use_tcgen05_ts")
    assert config["block_shape"] == block_shape
    assert config["warp_shape"] == (32, 64, block_shape[2])
    assert config["num_stages"] >= 3
    assert config["use_stream_k"] is False
    assert config["use_warp_spec"] is True

    default = get_heuristics_config(_layer_config(**overrides), shape_m=shape_m)
    assert not default.get("use_tcgen05")
