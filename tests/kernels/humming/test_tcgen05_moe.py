import os

import pytest
import torch

from humming import dtypes
from humming.config import ComputeConfig, GemmType, LayerConfig, MmaType
from humming.testing import (
    KernelTestCase,
    KernelTestRunner,
    assert_kernel_test_shape_coverage,
    skip_if_unsupported,
)
from humming.testing.runner import TEST_TUNING_SOURCE_ENV
from humming.tune import get_heuristics_config

TUNING_SOURCE = os.environ.get(TEST_TUNING_SOURCE_ENV, "heuristic")
pytestmark = pytest.mark.skipif(
    TUNING_SOURCE != "heuristic",
    reason=f"tcgen05 is heuristic-dispatched only, tuning source is {TUNING_SOURCE}",
)

SHAPE_N = 1024
SHAPE_K = 1024
NUM_EXPERTS = 8
TOP_K = 2


def _case(
    name: str,
    gemm_type: GemmType,
    *,
    b_dtype=dtypes.uint4,
    group_size: int = 128,
    has_zero_point: bool = False,
    has_bias: bool = False,
    num_experts: int = NUM_EXPERTS,
    top_k: int = TOP_K,
    expert_max_tokens: int | None = None,
) -> KernelTestCase:
    return KernelTestCase(
        name=name,
        layer_config=LayerConfig(
            shape_n=SHAPE_N,
            shape_k=SHAPE_K,
            num_experts=num_experts,
            a_dtype=dtypes.bfloat16,
            b_dtype=b_dtype,
            c_dtype=dtypes.bfloat16,
            bs_dtype=dtypes.bfloat16,
            weight_scale_group_size=group_size,
            has_zero_point=has_zero_point,
            has_bias=has_bias,
            mma_type=MmaType.TCGEN05,
        ),
        compute_config=ComputeConfig(gemm_type=gemm_type),
        top_k=top_k,
        expert_max_tokens=expert_max_tokens,
        seed=2026,
        input_std_scale=0.5,
        weight_std_scale=0.5,
        bias_std_scale=0.5,
        atol=0.2,
    )


# Grouping lives above the MMA, so these pin that the grouped scatter and the
# shared TS mainloop agree.
MOE_CASES = (
    _case("grouped-contiguous", GemmType.GROUPED_CONTIGUOUS),
    _case("grouped-contiguous-zp", GemmType.GROUPED_CONTIGUOUS, has_zero_point=True),
    _case("grouped-contiguous-zp-bias", GemmType.GROUPED_CONTIGUOUS, has_zero_point=True, has_bias=True),
    _case("grouped-contiguous-fp4", GemmType.GROUPED_CONTIGUOUS, b_dtype=dtypes.float4e2m1),
    _case("grouped-masked", GemmType.GROUPED_MASKED),
    _case("grouped-masked-zp", GemmType.GROUPED_MASKED, has_zero_point=True),
    _case("grouped-masked-zp-gs64", GemmType.GROUPED_MASKED, has_zero_point=True, group_size=64),
    _case(
        "grouped-contiguous-fine-grained",
        GemmType.GROUPED_CONTIGUOUS,
        has_zero_point=True,
        num_experts=128,
        top_k=1,
    ),
    _case(
        "grouped-masked-fine-grained",
        GemmType.GROUPED_MASKED,
        has_zero_point=True,
        num_experts=128,
        top_k=1,
        expert_max_tokens=128,
    ),
)


@pytest.mark.parametrize("test_case", MOE_CASES, ids=str)
def test_tcgen05_moe(test_case):
    config = test_case.layer_config
    skip_if_unsupported(a_dtype=config.a_dtype, mma_type=config.mma_type.value)
    assert config.tcgen05_supported
    results = KernelTestRunner(test_case).run()
    for result in results:
        assert result.tuning_values["use_tcgen05_ts"] is True
        torch.testing.assert_close(
            result.outputs,
            result.outputs_ref,
            rtol=test_case.rtol,
            atol=test_case.atol,
        )
    assert_kernel_test_shape_coverage(results)


@pytest.mark.parametrize(
    "gemm_type",
    [GemmType.GROUPED_CONTIGUOUS, GemmType.GROUPED_MASKED],
    ids=lambda value: value.value,
)
@pytest.mark.parametrize(
    "num_experts,shape_m,block_m",
    [(8, 2048, 128), (8, 512, 64), (128, 2048, 64), (256, 512, 64)],
)
def test_tcgen05_moe_block_m(gemm_type, num_experts, shape_m, block_m):
    config = _case("dispatch", gemm_type, num_experts=num_experts).layer_config
    skip_if_unsupported(mma_type=config.mma_type.value)
    heuristic_config = get_heuristics_config(config, shape_m=shape_m, gemm_type=gemm_type)
    assert heuristic_config["use_tcgen05_ts"] is True
    assert heuristic_config["block_shape"][0] == block_m
    assert heuristic_config["use_stream_k"] is False


def test_tcgen05_moe_case_coverage():
    assert {case.compute_config.gemm_type for case in MOE_CASES} == {
        GemmType.GROUPED_CONTIGUOUS,
        GemmType.GROUPED_MASKED,
    }
    assert {case.layer_config.has_zero_point for case in MOE_CASES} == {False, True}
    assert {case.layer_config.num_experts for case in MOE_CASES} == {NUM_EXPERTS, 128}
    assert {case.layer_config.weight_scale_group_size for case in MOE_CASES} == {64, 128}
    assert any(case.layer_config.has_bias for case in MOE_CASES)
    assert any(case.layer_config.b_dtype == dtypes.float4e2m1 for case in MOE_CASES)
