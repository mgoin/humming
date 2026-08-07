"""tcgen05 TS mode vs mma.sync for grouped MoE GEMMs, W4A16.

The sweep behind the grouped block_m rule in humming/tune/sm100.py.
"""

import math

import torch
import triton

from humming.config import GemmType
from humming.layer import HummingLayer
from humming.testing import generate_random_moe_tensors, random_fill_tensor
from humming.tune.sm100 import Sm100Heuristics

WEIGHT_CONFIG = {
    "dtype": "uint4",
    "group_size": 128,
    "scale_dtype": "bfloat16",
    "has_zero_point": True,
}

# Each model's expert config plus a coarse E=8 point, for the tile-fill compare.
SHAPES = [
    ("Qwen3-MoE gate/up", 1536, 4096, [(8, 2), (128, 8)]),
    ("Qwen3-MoE down", 4096, 1536, [(8, 2), (128, 8)]),
    ("DeepSeek-V3 gate/up", 2048, 7168, [(8, 2), (256, 8)]),
    ("DeepSeek-V3 down", 7168, 2048, [(8, 2), (256, 8)]),
    ("Mixtral gate/up", 14336, 4096, [(8, 2)]),
    ("Mixtral down", 4096, 14336, [(8, 2)]),
]
NUM_TOKENS = 512
GEMM_TYPES = [GemmType.GROUPED_CONTIGUOUS, GemmType.GROUPED_MASKED]


def build_layer(shape_n: int, shape_k: int, num_experts: int, mma_type: str | None = None):
    torch.manual_seed(2026)
    layer = HummingLayer(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
        weight_config=WEIGHT_CONFIG,
        torch_dtype=torch.bfloat16,
        mma_type=mma_type,
    ).to("cuda:0")
    for tensor in layer.parameters():
        random_fill_tensor(tensor)
    layer.transform()
    return layer


def bench(layer: HummingLayer, config: dict, **launch_kwargs) -> float:
    def run():
        return layer(tuning_config=[[0, 1 << 30, config]], **launch_kwargs)

    run()
    torch.cuda.synchronize()
    return triton.testing.do_bench(run, warmup=100, rep=1000) * 1e3


def make_problem(shape_k: int, num_experts: int, top_k: int, gemm_type: GemmType) -> dict:
    expert_max_tokens = math.ceil(NUM_TOKENS * top_k / num_experts)
    _, expert_layout, *_ = generate_random_moe_tensors(
        shape_m=NUM_TOKENS,
        num_experts=num_experts,
        top_k=top_k,
        gemm_type=gemm_type,
        balanced=True,
        expert_max_tokens=expert_max_tokens,
    )
    if gemm_type == GemmType.GROUPED_CONTIGUOUS:
        shape_m = NUM_TOKENS * top_k
        valid_shape_m = 0
    else:
        shape_m = num_experts * expert_max_tokens
        valid_shape_m = NUM_TOKENS * top_k
    inputs = torch.randn((shape_m, shape_k), dtype=torch.bfloat16, device="cuda:0")
    return {
        "inputs": inputs,
        "expert_layout": expert_layout,
        "top_k": top_k,
        "valid_shape_m": valid_shape_m,
        "compute_config": {"gemm_type": gemm_type.value},
    }


def main() -> None:
    print(f"device: {torch.cuda.get_device_name(0)}")
    header = (
        f"{'shape':<21}{'mode':<12}{'E':>5}{'top_k':>6}{'M':>7}"
        f"{'block_m':>8}{'mma us':>10}{'TS us':>10}{'mma/TS':>9}"
    )
    print(header)
    print("-" * len(header))
    for label, shape_n, shape_k, points in SHAPES:
        for num_experts, top_k in points:
            layer = build_layer(shape_n, shape_k, num_experts)
            ts_layer = build_layer(shape_n, shape_k, num_experts, mma_type="tcgen05")
            for gemm_type in GEMM_TYPES:
                problem = make_problem(shape_k, num_experts, top_k, gemm_type)
                shape_m = problem["inputs"].shape[0]
                ts_config = Sm100Heuristics._ts_config(ts_layer.humming_config, shape_m, gemm_type)
                mma_config = Sm100Heuristics.get_config(
                    layer_config=layer.humming_config,
                    shape_m=shape_m,
                    gemm_type=gemm_type,
                )
                mma_us = bench(layer, mma_config, **problem)
                ts_us = bench(ts_layer, ts_config, **problem)
                print(
                    f"{label:<21}{gemm_type.value.split('_')[1]:<12}{num_experts:>5}{top_k:>6}"
                    f"{shape_m:>7}{ts_config['block_shape'][0]:>8}{mma_us:>10.1f}"
                    f"{ts_us:>10.1f}{mma_us / ts_us:>8.2f}x"
                )
            del layer, ts_layer
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
