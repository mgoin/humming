import pytest

from humming import dtypes
from humming.config import (
    GemmType,
    LayerConfig,
    MmaType,
    TuningConfig,
    WeightScale2Type,
    WeightScaleType,
)
from humming.config.config import TCGEN05_TS_B_DTYPES
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


def _to_tuning_config(config: dict) -> TuningConfig:
    fields = TuningConfig.__dataclass_fields__
    return TuningConfig(**{k: v for k, v in config.items() if k in fields})


@pytest.fixture(autouse=True)
def _requires_tcgen05():
    skip_if_unsupported(mma_type="tcgen05")


TS_SHAPE_MS = (1, 17, 127, 128, 4096)


@pytest.mark.parametrize("shape_m", TS_SHAPE_MS)
def test_ts_opt_in_config(shape_m):
    config = get_heuristics_config(_layer_config(mma_type=MmaType.TCGEN05), shape_m=shape_m)
    assert config["mma_type"] == "tcgen05"
    assert config["use_tcgen05"] is True
    assert config["use_tcgen05_ts"] is True
    # tcgen05_ts_mma.cuh static_asserts BlockN=128, BlockK=64 and a single
    # M-warp/K-warp; block_m is the one dimension the heuristic picks, and it
    # must not tile past the problem.
    block_m, block_n, block_k = config["block_shape"]
    assert (block_n, block_k) == (128, 64)
    assert config["warp_shape"] == (block_m, 32, 64)
    assert block_m <= max(64, 1 << (shape_m - 1).bit_length())
    assert config["num_stages"] >= 3
    assert config["use_warp_spec"] is True
    assert config["use_tma"] is True
    # No cross-CTA partial-K reduction in the drain, and the zero point is
    # loaded by TMA.
    assert config["use_stream_k"] is False
    assert config["use_tma_bzp"] is True
    assert config["raster_group_m"] == 1


def test_ts_block_m_grows_with_shape_m():
    block_ms = [
        get_heuristics_config(_layer_config(mma_type=MmaType.TCGEN05), shape_m=shape_m)["block_shape"][0]
        for shape_m in TS_SHAPE_MS
    ]
    assert block_ms == sorted(block_ms)
    assert (block_ms[0], block_ms[-1]) == (64, 128)


def _ts_num_stages(shape_m: int = 2048, **overrides) -> int:
    config = get_heuristics_config(_layer_config(mma_type=MmaType.TCGEN05, **overrides), shape_m=shape_m)
    return config["num_stages"]


def test_ts_num_stages_is_keyed_on_weight_dtype_alone():
    # A latency cap per weight dtype: every wired dtype sits between the
    # mainloop's static_assert floor and the default depth, uint8 is the one
    # dtype tuned below it, and nothing else about the layer moves the choice.
    default = _ts_num_stages()
    assert default >= 3
    for b_dtype in TCGEN05_TS_B_DTYPES:
        depth = _ts_num_stages(b_dtype=b_dtype, has_zero_point=b_dtype.is_integer_type)
        assert 3 <= depth <= default
    assert _ts_num_stages(b_dtype=dtypes.uint8) < default

    assert _ts_num_stages(has_zero_point=False) == default
    assert _ts_num_stages(a_dtype=dtypes.float16, c_dtype=dtypes.float16, bs_dtype=dtypes.float16) == default
    assert _ts_num_stages(b_dtype=dtypes.uint8, is_fp_zero_point=True) == _ts_num_stages(b_dtype=dtypes.uint8)


@pytest.mark.parametrize("b_dtype", [dtypes.uint4, dtypes.uint8], ids=str)
def test_ts_grouped_takes_the_dense_pipeline(b_dtype):
    config = get_heuristics_config(
        _layer_config(mma_type=MmaType.TCGEN05, num_experts=8, b_dtype=b_dtype),
        shape_m=1024,
        gemm_type=GemmType.GROUPED_CONTIGUOUS,
    )
    assert config["use_tcgen05_ts"] is True
    assert config["num_stages"] == _ts_num_stages(shape_m=1024, b_dtype=b_dtype)


@pytest.mark.parametrize("tokens_per_expert", [64, 256])
def test_ts_grouped_block_m_follows_tokens_per_expert(tokens_per_expert):
    # shape_m counts padded tokens over all experts, but the scheduler tiles per
    # expert, so grouped must tile as dense would at tokens-per-expert.
    num_experts = 8
    grouped = get_heuristics_config(
        _layer_config(mma_type=MmaType.TCGEN05, num_experts=num_experts),
        shape_m=tokens_per_expert * num_experts,
        gemm_type=GemmType.GROUPED_CONTIGUOUS,
    )
    dense = get_heuristics_config(_layer_config(mma_type=MmaType.TCGEN05), shape_m=tokens_per_expert)
    assert grouped["block_shape"] == dense["block_shape"]


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"shape_n": 8192, "shape_k": 28672},
        {"shape_n": 6144},
        {"b_dtype": dtypes.uint3},
        {"b_dtype": dtypes.uint8},
    ],
    ids=str,
)
@pytest.mark.parametrize("shape_m", [1, 128, 4096])
def test_default_layer_never_selects_tcgen05(overrides, shape_m):
    config = get_heuristics_config(_layer_config(**overrides), shape_m=shape_m)
    assert config.get("mma_type") is None
    assert not config.get("use_tcgen05")
    assert not config.get("use_tcgen05_ts")


@pytest.mark.parametrize(
    "overrides",
    [
        # BlockN is 128 and BlockK is 64, so the problem shape must tile.
        {"shape_n": 192},
        {"shape_k": 4128, "weight_scale_group_size": 32},
        # A 16-K MMA iteration must stay inside one weight-scale group.
        {"weight_scale_group_size": 48, "shape_k": 4128},
        # The TS packer holds only 32 % num_bits == 0 widths.
        {"b_dtype": dtypes.uint6},
        {"b_dtype": dtypes.DataType.from_str("float6e4m1"), "has_zero_point": False},
        # TS activations are the two 16-bit float dtypes.
        {"a_dtype": dtypes.float8e4m3, "b_dtype": dtypes.float8e4m3, "has_zero_point": False},
        # A 16-bit group scale is read as ElementA; e8m0 shares its exponent
        # bias with bf16 only.
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
        # The launcher types the fp zero point as c_dtype; s2r reads ElementA.
        {"a_dtype": dtypes.float16, "bs_dtype": dtypes.float16, "is_fp_zero_point": True},
        # weight_scale_2 is applied in the epilogue smem writer the drain
        # bypasses, so admitting it would drop the scale silently.
        {"weight_scale_2_type": WeightScale2Type.CHANNEL},
        {"weight_scale_2_type": WeightScale2Type.TENSOR},
    ],
    ids=str,
)
def test_ts_illegal_layers(overrides):
    config = _layer_config(**overrides)
    assert not config.tcgen05_ts_supported
    assert not get_heuristics_class().supports_tcgen05_ts(config)


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
        # drain_accum converts with __floats2bfloat162_rn and bypasses the
        # epilogue smem writer, so a non-bf16 c_dtype would receive bf16 bits.
        {"c_dtype": dtypes.float16},
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


# The rest of the dispatch-site rejections run as ILLEGAL_CASES in
# test_tcgen05.py. These cannot: the weight schema rejects the mismatched
# scale tensor before the layer ever reaches the gate.
@pytest.mark.parametrize(
    "overrides",
    [
        {"a_dtype": dtypes.float16, "c_dtype": dtypes.float16},
        {"a_dtype": dtypes.float16, "bs_dtype": dtypes.float16, "is_fp_zero_point": True},
        # TS turns the weight dtype down and SS is bf16-out only.
        {"b_dtype": dtypes.uint3, "c_dtype": dtypes.float16},
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
    "overrides",
    [
        # Weight dtypes with no ts_dequant_b_pair arm.
        {"b_dtype": dtypes.uint3},
        {"b_dtype": dtypes.uint6},
        {"b_dtype": dtypes.uint7},
        {"b_dtype": dtypes.DataType.from_str("float6e4m1"), "has_zero_point": False},
        # shape_n only tiles at BlockN=64, which TS does not accept.
        {"shape_n": 6208},
        {"b_dtype": dtypes.uint8, "shape_n": 6208},
        # shape_k only tiles at BlockK=64.
        {"b_dtype": dtypes.uint3, "shape_k": 4160},
    ],
    ids=str,
)
@pytest.mark.parametrize("shape_m", [1, 2048])
def test_ss_opt_in_fallback(overrides, shape_m):
    layer_config = _layer_config(mma_type=MmaType.TCGEN05, **overrides)
    assert not layer_config.tcgen05_ts_supported
    config = get_heuristics_config(layer_config, shape_m=shape_m)
    assert config["mma_type"] == "tcgen05"
    assert config["use_tcgen05"] is True
    assert not config.get("use_tcgen05_ts")

    block_m, block_n, block_k = config["block_shape"]
    group_size = layer_config.weight_scale_group_size
    # The problem and the weight-scale group must tile, and the mainloop pins
    # the 32x64 warp tile; block_k is otherwise a tuning choice.
    assert block_m == 128
    assert block_n == (128 if layer_config.shape_n % 128 == 0 else 64)
    assert layer_config.shape_k % block_k == 0
    assert not (group_size % block_k and block_k % group_size)
    assert config["warp_shape"] == (32, 64, block_k)
    assert config["num_stages"] >= 3
    assert config["use_stream_k"] is False
    assert config["use_warp_spec"] is True

    default = get_heuristics_config(_layer_config(**overrides), shape_m=shape_m)
    assert not default.get("use_tcgen05")


@pytest.mark.parametrize("overrides", [{}, {"shape_n": 6208}], ids=str)
def test_tcgen05_never_launches_with_pdl(overrides):
    # tcgen05 allocates TMEM at entry, before the griddepcontrol handshake.
    layer_config = _layer_config(**overrides)
    assert _to_tuning_config(get_heuristics_config(layer_config, shape_m=2048)).use_pdl

    tcgen05 = _layer_config(mma_type=MmaType.TCGEN05, **overrides)
    assert not _to_tuning_config(get_heuristics_config(tcgen05, shape_m=2048)).use_pdl

    with pytest.raises(AssertionError, match="unaudited"):
        TuningConfig(
            block_shape=(128, 128, 64),
            warp_shape=(128, 32, 64),
            use_tcgen05_ts=True,
            use_warp_spec=True,
            num_stages=4,
            use_pdl=True,
        )
