import os

import pytest
import torch

from humming import dtypes
from humming.config import (
    ComputeConfig,
    GemmType,
    LayerConfig,
    MmaType,
    WeightScale2Type,
    WeightScaleType,
)
from humming.testing import (
    KernelTestCase,
    KernelTestRunner,
    assert_kernel_test_shape_coverage,
    skip_if_unsupported,
)
from humming.testing.runner import TEST_TUNING_SOURCE_ENV
from humming.tune import get_heuristics_config

SHAPE_N = 512
SHAPE_K = 512

# tcgen05 ships exactly the geometry the sm100 heuristic emits: sampled
# geometries are rejected by the TS static asserts, and TS has no
# batch-invariant reduction (both pinned in test_tcgen05_heuristic.py).
TUNING_SOURCE = os.environ.get(TEST_TUNING_SOURCE_ENV, "heuristic")
pytestmark = pytest.mark.skipif(
    TUNING_SOURCE != "heuristic",
    reason=f"tcgen05 is heuristic-dispatched only, tuning source is {TUNING_SOURCE}",
)


def _case(
    name: str,
    *,
    b_dtype=dtypes.uint4,
    bs_dtype=dtypes.bfloat16,
    group_size: int = 128,
    group_size_n: int = 0,
    weight_scale_type: WeightScaleType | None = None,
    weight_scale_2_type: WeightScale2Type | None = None,
    has_zero_point: bool = False,
    is_fp_zero_point: bool = False,
    has_bias: bool = False,
    shape_n: int = SHAPE_N,
    shape_k: int = SHAPE_K,
    mma_type=MmaType.TCGEN05,
    atol: float = 0.05,
) -> KernelTestCase:
    return KernelTestCase(
        name=name,
        layer_config=LayerConfig(
            shape_n=shape_n,
            shape_k=shape_k,
            a_dtype=dtypes.bfloat16,
            b_dtype=b_dtype,
            c_dtype=dtypes.bfloat16,
            bs_dtype=bs_dtype,
            weight_scale_group_size=group_size,
            weight_scale_group_size_n=group_size_n,
            weight_scale_type=weight_scale_type,
            weight_scale_2_type=weight_scale_2_type,
            has_zero_point=has_zero_point,
            is_fp_zero_point=is_fp_zero_point,
            has_bias=has_bias,
            mma_type=mma_type,
        ),
        compute_config=ComputeConfig(gemm_type=GemmType.DENSE),
        seed=2026,
        atol=atol,
    )


# Every weight dtype wired into ts_dequant_b_pair, with the zero-point modes
# each one supports: integer zp for the uint_to_f16 dtypes, none for the fp
# dtypes (their dequant carries a constant exponent offset instead).
TS_WEIGHT_DTYPE_CASES = (
    _case("uint2", b_dtype=dtypes.uint2),
    _case("uint2-zp", b_dtype=dtypes.uint2, has_zero_point=True),
    _case("uint4", b_dtype=dtypes.uint4),
    _case("uint4-zp", b_dtype=dtypes.uint4, has_zero_point=True),
    _case("uint8", b_dtype=dtypes.uint8),
    _case("uint8-zp", b_dtype=dtypes.uint8, has_zero_point=True),
    _case("fp4e2m1", b_dtype=dtypes.float4e2m1),
    _case("fp8e4m3", b_dtype=dtypes.float8e4m3),
)

# gs=0 folds the row scale into the TMEM drain, gs=64 advances the scale row
# every stage, and gs<64 puts two groups in one BlockK=64 stage.
TS_SCALE_CASES = (
    _case("uint4-channel-scale", group_size=0),
    _case("uint4-zp-gs16", has_zero_point=True, group_size=16),
    _case("uint4-zp-gs32", has_zero_point=True, group_size=32),
    _case("uint4-zp-gs64", has_zero_point=True, group_size=64),
    _case("uint4-fp-zp-gs128", has_zero_point=True, is_fp_zero_point=True),
    _case("uint4-fp-zp-gs32", has_zero_point=True, is_fp_zero_point=True, group_size=32),
    _case("uint2-fp-zp-gs128", b_dtype=dtypes.uint2, has_zero_point=True, is_fp_zero_point=True),
    _case("uint4-zp-bias", has_zero_point=True, has_bias=True),
)

# TS is legal for shape_n % 128 == 0 and shape_k % 64 == 0; these walk the
# tile edges of that space, and the runner's shape_m sweep adds the partial
# M-tiles (m=1/17/257 against block_m 64/128).
TS_SHAPE_CASES = (
    _case("single-tile", has_zero_point=True, group_size=64, shape_n=128, shape_k=64),
    _case("odd-n-tiles", has_zero_point=True, shape_n=384),
    _case("k-not-multiple-128", has_zero_point=True, group_size=64, shape_n=256, shape_k=320),
    _case("fat-k", has_zero_point=True, shape_n=1024, shape_k=8192, atol=0.1),
)

TS_CASES = TS_WEIGHT_DTYPE_CASES + TS_SCALE_CASES + TS_SHAPE_CASES

# mma_type="tcgen05" is an opt-in to the tcgen05 family, so a layer neither
# mainloop is legal for must be rejected while packing instead of falling back.
# The weight_scale_2 and CHANNEL/TENSOR-scale entries are the fail-closed half
# of that contract: both drains bypass the epilogue smem writer that applies
# them, so an admitted layer would return silently wrong results.
ILLEGAL_CASES = (
    _case("shape-n-not-multiple-64", shape_n=SHAPE_N + 32, has_zero_point=True),
    _case("shape-k-not-multiple-64", shape_k=SHAPE_K + 32, group_size=32, has_zero_point=True),
    _case("straddling-scale-group", group_size=48, has_zero_point=True, shape_k=384),
    _case("weight-dtype-not-wired", b_dtype=dtypes.float6e3m2),
    _case("ts-layer-bs2-channel", has_zero_point=True, weight_scale_2_type=WeightScale2Type.CHANNEL),
    _case("ts-layer-bs2-tensor", has_zero_point=True, weight_scale_2_type=WeightScale2Type.TENSOR),
    _case(
        "ss-layer-bs2-channel",
        b_dtype=dtypes.uint3,
        has_zero_point=True,
        weight_scale_2_type=WeightScale2Type.CHANNEL,
    ),
    _case(
        "ss-layer-bs2-tensor",
        b_dtype=dtypes.uint3,
        has_zero_point=True,
        weight_scale_2_type=WeightScale2Type.TENSOR,
    ),
    _case("ss-layer-channel-scale", b_dtype=dtypes.uint3, group_size=0),
    _case(
        "ss-layer-tensor-scale",
        b_dtype=dtypes.uint3,
        bs_dtype=dtypes.float32,
        group_size=0,
        weight_scale_type=WeightScaleType.TENSOR,
    ),
)

# shape_n % 128 == 0 gives SS BlockN=128, anything else BlockN=64. A weight
# dtype TS also carries only reaches SS at the latter.
SS_SHAPE_N_BLOCK128 = 4096
SS_SHAPE_N_BLOCK64 = 4160


def _fp(name: str) -> dtypes.DataType:
    """A narrow float weight dtype with no humming.dtypes constant."""
    return dtypes.DataType.from_str(name)


def _ss_case(
    name: str,
    *,
    b_dtype=dtypes.uint4,
    has_zero_point: bool = False,
    is_fp_zero_point: bool = False,
    bs_dtype=dtypes.bfloat16,
    group_size: int = 128,
    group_size_n: int = 0,
    weight_scale_type: WeightScaleType | None = None,
    shape_n: int = SS_SHAPE_N_BLOCK128,
) -> KernelTestCase:
    return _case(
        name,
        b_dtype=b_dtype,
        bs_dtype=bs_dtype,
        group_size=group_size,
        group_size_n=group_size_n,
        weight_scale_type=weight_scale_type,
        has_zero_point=has_zero_point,
        is_fp_zero_point=is_fp_zero_point,
        shape_n=shape_n,
        shape_k=2048,
        mma_type=MmaType.TCGEN05,
        atol=0.1,
    )


# One case per weight dtype in the SS tuning table, in each zero-point mode the
# dtype carries: SS reads the ordinary mma.sync weight layout, so the table is
# tuning and this suite is what makes it a tested capability. The integer zero
# point is an unsigned-integer-B feature, so the fp dtypes appear once. shape_n
# alternates so both BlockN are exercised; the five dtypes TS also carries are
# pinned to BlockN=64 because TS takes every shape_n % 128 == 0 layer.
SS_WEIGHT_DTYPE_CASES = (
    _ss_case("ss-uint1", b_dtype=dtypes.uint1),
    _ss_case("ss-uint1-zp", b_dtype=dtypes.uint1, has_zero_point=True, shape_n=SS_SHAPE_N_BLOCK64),
    _ss_case("ss-uint2", b_dtype=dtypes.uint2, shape_n=SS_SHAPE_N_BLOCK64),
    _ss_case("ss-uint2-zp", b_dtype=dtypes.uint2, has_zero_point=True, shape_n=SS_SHAPE_N_BLOCK64),
    _ss_case("ss-uint3", b_dtype=dtypes.uint3),
    _ss_case("ss-uint3-zp", b_dtype=dtypes.uint3, has_zero_point=True, shape_n=SS_SHAPE_N_BLOCK64),
    _ss_case("ss-uint4", b_dtype=dtypes.uint4, shape_n=SS_SHAPE_N_BLOCK64),
    _ss_case("ss-uint4-zp", b_dtype=dtypes.uint4, has_zero_point=True, shape_n=SS_SHAPE_N_BLOCK64),
    _ss_case("ss-uint5", b_dtype=dtypes.uint5),
    _ss_case("ss-uint5-zp", b_dtype=dtypes.uint5, has_zero_point=True, shape_n=SS_SHAPE_N_BLOCK64),
    _ss_case("ss-uint6", b_dtype=dtypes.uint6),
    _ss_case("ss-uint6-zp", b_dtype=dtypes.uint6, has_zero_point=True, shape_n=SS_SHAPE_N_BLOCK64),
    _ss_case("ss-uint7", b_dtype=dtypes.uint7),
    _ss_case("ss-uint7-zp", b_dtype=dtypes.uint7, has_zero_point=True, shape_n=SS_SHAPE_N_BLOCK64),
    _ss_case("ss-uint8", b_dtype=dtypes.uint8, shape_n=SS_SHAPE_N_BLOCK64),
    _ss_case("ss-uint8-zp", b_dtype=dtypes.uint8, has_zero_point=True, shape_n=SS_SHAPE_N_BLOCK64),
    _ss_case("ss-fp3e1m1", b_dtype=_fp("float3e1m1")),
    _ss_case("ss-fp3e2m0", b_dtype=_fp("float3e2m0"), shape_n=SS_SHAPE_N_BLOCK64),
    _ss_case("ss-fp4e2m1", b_dtype=dtypes.float4e2m1, shape_n=SS_SHAPE_N_BLOCK64),
    _ss_case("ss-fp4e3m0", b_dtype=_fp("float4e3m0")),
    _ss_case("ss-fp5e2m2", b_dtype=_fp("float5e2m2")),
    _ss_case("ss-fp5e4m0", b_dtype=_fp("float5e4m0"), shape_n=SS_SHAPE_N_BLOCK64),
    _ss_case("ss-fp6e2m3", b_dtype=dtypes.float6e2m3),
    _ss_case("ss-fp6e4m1", b_dtype=_fp("float6e4m1"), shape_n=SS_SHAPE_N_BLOCK64),
    _ss_case("ss-fp7e2m4", b_dtype=_fp("float7e2m4")),
    _ss_case("ss-fp7e4m2", b_dtype=_fp("float7e4m2"), shape_n=SS_SHAPE_N_BLOCK64),
    _ss_case("ss-fp7e6m0", b_dtype=_fp("float7e6m0")),
    _ss_case("ss-fp8e1m6", b_dtype=_fp("float8e1m6"), shape_n=SS_SHAPE_N_BLOCK64),
    _ss_case("ss-fp8e4m3", b_dtype=dtypes.float8e4m3, shape_n=SS_SHAPE_N_BLOCK64),
    _ss_case("ss-fp8e5m2", b_dtype=dtypes.float8e5m2),
)

# The quantisation parameters SS admits beyond a bf16 group scale. Each is
# applied by the generic mainloop on B, which is the half of the arithmetic
# drain_accum replicates.
SS_SCALE_CASES = (
    _ss_case(
        "ss-uint8-fp-zp",
        b_dtype=dtypes.uint8,
        has_zero_point=True,
        is_fp_zero_point=True,
        shape_n=SS_SHAPE_N_BLOCK64,
    ),
    _ss_case("ss-bs-e8m0-gs32", bs_dtype=dtypes.float8e8m0, group_size=32),
    _ss_case(
        "ss-bs-e4m3",
        bs_dtype=dtypes.float8e4m3,
        has_zero_point=True,
        shape_n=SS_SHAPE_N_BLOCK64,
    ),
    _ss_case(
        "ss-block64x64",
        bs_dtype=dtypes.float32,
        group_size=64,
        group_size_n=64,
        weight_scale_type=WeightScaleType.BLOCK,
    ),
)

SS_CASES = SS_WEIGHT_DTYPE_CASES + SS_SCALE_CASES


def _run(test_case: KernelTestCase, *, expect_ts: bool = False) -> list:
    results = KernelTestRunner(test_case).run()
    for result in results:
        if expect_ts:
            assert result.tuning_values["use_tcgen05_ts"] is True
        torch.testing.assert_close(
            result.outputs,
            result.outputs_ref,
            rtol=test_case.rtol,
            atol=test_case.atol,
        )
    assert_kernel_test_shape_coverage(results)
    return results


@pytest.mark.parametrize("test_case", TS_CASES, ids=str)
def test_tcgen05_ts(test_case):
    config = test_case.layer_config
    skip_if_unsupported(a_dtype=config.a_dtype, mma_type=config.mma_type.value)
    assert config.tcgen05_supported
    _run(test_case, expect_ts=True)


@pytest.mark.parametrize("test_case", ILLEGAL_CASES, ids=str)
def test_tcgen05_rejects_illegal_layer(test_case):
    config = test_case.layer_config
    skip_if_unsupported(mma_type=config.mma_type.value)
    assert not config.tcgen05_supported
    with pytest.raises(AssertionError, match="legal for neither"):
        KernelTestRunner(test_case)


@pytest.mark.parametrize("test_case", SS_CASES, ids=str)
def test_tcgen05_ss(test_case):
    """The opt-in dispatches SS -- never mma.sync -- where TS is illegal."""
    config = test_case.layer_config
    skip_if_unsupported(a_dtype=config.a_dtype, mma_type="tcgen05")
    assert config.mma_type == MmaType.TCGEN05
    assert not config.tcgen05_supported
    heuristic_config = get_heuristics_config(config, shape_m=1024)
    assert heuristic_config["mma_type"] == "tcgen05"
    assert heuristic_config["use_tcgen05"] is True
    assert not heuristic_config.get("use_tcgen05_ts")
    results = _run(test_case)
    assert all(result.tuning_values.get("use_tcgen05") for result in results)
    assert not any(result.tuning_values.get("use_tcgen05_ts") for result in results)


@pytest.mark.parametrize("shape_m", [17, 256])
def test_tcgen05_ts_layer_opt_in(shape_m):
    """mma_type reaches LayerConfig through HummingLayer, and the opted-in
    layer matches the default path on the same unquantized weights."""
    from humming.layer import HummingLayer
    from humming.schema import HummingWeightSchema

    skip_if_unsupported(mma_type="tcgen05")
    shape_n, shape_k = 512, 512
    schema = HummingWeightSchema(
        b_dtype=dtypes.uint4,
        bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=128,
        has_zero_point=True,
    )

    def build(mma_type):
        torch.manual_seed(2026)
        weight = torch.randn(shape_n, shape_k, dtype=torch.bfloat16, device="cuda") / shape_k**0.5
        layer = HummingLayer(
            shape_n=shape_n,
            shape_k=shape_k,
            weight_config=schema,
            torch_dtype=torch.bfloat16,
            mma_type=mma_type,
        ).cuda()
        layer.load_from_unquantized(weight)
        layer.transform()
        return layer

    layer_default = build(None)
    layer_ts = build("tcgen05")
    assert layer_default.humming_metas[""].mma_type == MmaType.MMA
    assert layer_ts.humming_metas[""].mma_type == MmaType.TCGEN05

    torch.manual_seed(11)
    inputs = torch.randn(shape_m, shape_k, dtype=torch.bfloat16, device="cuda") / shape_k**0.5
    outputs_default = layer_default.forward(inputs.clone())
    outputs_ts = layer_ts.forward(inputs.clone())
    torch.testing.assert_close(outputs_ts, outputs_default, rtol=0.01, atol=0.05)


def test_tcgen05_layer_rejects_illegal_shape():
    from humming.layer import HummingLayer
    from humming.schema import HummingWeightSchema

    skip_if_unsupported(mma_type="tcgen05")
    shape_n, shape_k = 160, 512
    schema = HummingWeightSchema(
        b_dtype=dtypes.uint4,
        bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=128,
        has_zero_point=True,
    )
    layer = HummingLayer(
        shape_n=shape_n,
        shape_k=shape_k,
        weight_config=schema,
        torch_dtype=torch.bfloat16,
        mma_type="tcgen05",
    ).cuda()
    layer.load_from_unquantized(torch.randn(shape_n, shape_k, dtype=torch.bfloat16, device="cuda"))
    with pytest.raises(AssertionError, match="legal for neither"):
        layer.transform()


def test_tcgen05_case_coverage():
    assert all(case.layer_config.mma_type == MmaType.TCGEN05 for case in TS_CASES)
    assert {case.layer_config.b_dtype for case in TS_WEIGHT_DTYPE_CASES} == {
        dtypes.uint2,
        dtypes.uint4,
        dtypes.uint8,
        dtypes.float4e2m1,
        dtypes.float8e4m3,
    }
    assert {case.layer_config.weight_scale_group_size for case in TS_CASES} == {0, 16, 32, 64, 128}

    zero_point_modes = {
        (case.layer_config.has_zero_point, case.layer_config.is_fp_zero_point) for case in TS_CASES
    }
    assert zero_point_modes == {(False, False), (True, False), (True, True)}
    assert any(case.layer_config.has_bias for case in TS_CASES)

    shape_ns = {case.layer_config.shape_n for case in TS_SHAPE_CASES}
    shape_ks = {case.layer_config.shape_k for case in TS_SHAPE_CASES}
    assert all(shape_n % 128 == 0 for shape_n in shape_ns)
    assert all(shape_k % 64 == 0 for shape_k in shape_ks)
    assert min(shape_ns) == 128 and min(shape_ks) == 64
    assert any(shape_k % 128 for shape_k in shape_ks)
    assert max(shape_ks) >= 8192

    # Every weight dtype the SS table carries must have a running case, at both
    # BlockN, and no SS case may be one the opt-in would route to TS instead.
    from humming.tune.sm100 import _SS_B_DTYPE_CONFIG

    ss_configs = [case.layer_config for case in SS_CASES]
    assert {config.b_dtype for config in ss_configs} == set(_SS_B_DTYPE_CONFIG)
    assert {config.shape_n % 128 == 0 for config in ss_configs} == {False, True}
    assert all(config.mma_type == MmaType.TCGEN05 for config in ss_configs)

    # Both zero-point modes for every dtype that carries one: the integer zero
    # point is unsigned-integer-B only, the fp one rides along on uint8.
    weight_configs = [case.layer_config for case in SS_WEIGHT_DTYPE_CASES]
    assert {config.b_dtype for config in weight_configs if not config.has_zero_point} == set(
        _SS_B_DTYPE_CONFIG
    )
    assert {config.b_dtype for config in weight_configs if config.has_zero_point} == {
        b_dtype for b_dtype in _SS_B_DTYPE_CONFIG if b_dtype.is_integer_type
    }
    assert any(config.is_fp_zero_point for config in ss_configs)

    assert {config.weight_scale_type for config in ss_configs} == {
        WeightScaleType.GROUP,
        WeightScaleType.BLOCK,
    }
    assert {config.bs_dtype for config in ss_configs} == {
        dtypes.bfloat16,
        dtypes.float8e4m3,
        dtypes.float8e8m0,
        dtypes.float32,
    }

    # Fail-closed coverage: the scale kinds the tcgen05 drains would drop.
    illegal_configs = [case.layer_config for case in ILLEGAL_CASES]
    assert {config.weight_scale_2_type for config in illegal_configs} == set(WeightScale2Type)
    assert {WeightScaleType.CHANNEL, WeightScaleType.TENSOR} <= {
        config.weight_scale_type for config in illegal_configs
    }
