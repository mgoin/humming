import pytest

from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType
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


@pytest.mark.parametrize("shape_m", [1, 16, 64, 128, 512, 2048])
def test_default_layer_never_selects_ts(shape_m):
    config = get_heuristics_config(_layer_config(), shape_m=shape_m)
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
        # The fp zero point is subtracted before the scale, which only the
        # uint_to_f16 dequant bodies carry.
        {"b_dtype": dtypes.uint8, "is_fp_zero_point": True},
        # Weight dtypes with no ts_dequant_b_pair arm.
        {"b_dtype": dtypes.uint6},
        {"b_dtype": dtypes.float8e5m2, "has_zero_point": False},
        # TS is bf16 x narrow-B only, with bf16 scales.
        {"a_dtype": dtypes.float8e4m3, "b_dtype": dtypes.float8e4m3, "has_zero_point": False},
        {"bs_dtype": dtypes.float8e8m0},
    ],
    ids=str,
)
def test_ts_illegal_layers(overrides):
    config = _layer_config(**overrides)
    assert not config.tcgen05_supported
    assert not get_heuristics_class().supports_tcgen05_ts(config)


def test_ts_opt_in_rejects_illegal_layer():
    config = _layer_config(shape_n=192, mma_type=MmaType.TCGEN05)
    with pytest.raises(AssertionError, match="not legal"):
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
    "overrides,shape_m,selects_ss",
    [
        # Fat-N: mma.sync's smaller tiles win below the M cutoff.
        ({}, 1, False),
        ({}, 64, False),
        ({}, 128, True),
        ({}, 2048, True),
        # Fat-K only pays off from M=512 up.
        ({"shape_n": 8192, "shape_k": 28672}, 256, False),
        ({"shape_n": 8192, "shape_k": 28672}, 512, True),
        # Below the output-tile floor most SMs would idle.
        ({"shape_n": 6144}, 128, False),
        ({"shape_n": 6144}, 256, True),
        # Outside the SS-mode dtype and scale window.
        ({"a_dtype": dtypes.float8e4m3, "b_dtype": dtypes.float8e4m3, "has_zero_point": False}, 512, False),
        ({"b_dtype": dtypes.uint7}, 512, False),
        ({"weight_scale_group_size": 0, "has_zero_point": False}, 512, False),
    ],
    ids=str,
)
def test_ss_auto_selection(overrides, shape_m, selects_ss):
    config = get_heuristics_config(_layer_config(**overrides), shape_m=shape_m)
    assert (config.get("mma_type") == "tcgen05") == selects_ss
    assert bool(config.get("use_tcgen05")) == selects_ss
    if selects_ss:
        assert not config.get("use_tcgen05_ts")
        assert config["use_stream_k"] is False
        assert config["use_warp_spec"] is True


@pytest.mark.parametrize(
    "overrides,block_shape",
    [
        ({}, (128, 128, 128)),
        # shape_k only tiles at BlockK=64.
        ({"shape_k": 4160}, (128, 128, 64)),
        # shape_n only tiles at BlockN=64.
        ({"shape_n": 6208}, (128, 64, 128)),
        # uint8 needs the narrow K tile to keep the b_dequant buffer in SMEM.
        ({"b_dtype": dtypes.uint8}, (128, 128, 64)),
    ],
    ids=str,
)
def test_ss_block_shape(overrides, block_shape):
    config = get_heuristics_config(_layer_config(**overrides), shape_m=512)
    assert config["mma_type"] == "tcgen05"
    assert config["block_shape"] == block_shape
    assert config["warp_shape"] == (32, 64, block_shape[2])
