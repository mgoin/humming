"""tcgen05 TS mode vs mma.sync for grouped MoE GEMMs.

The sweep behind the grouped block_m rule in humming/tune/sm100.py: --block_m
overrides the tile the tokens-per-expert threshold picks.
"""

import argparse
import math

import torch
import triton

from humming import dtypes
from humming.config import GemmType
from humming.layer import HummingLayer
from humming.testing import (
    generate_random_moe_tensors,
    random_fill_tensor,
    save_benchmark_result,
)
from humming.tune import get_heuristics_config


def build_layer(args, torch_dtype, mma_type: str | None = None) -> HummingLayer:
    torch.manual_seed(2026)
    layer = HummingLayer(
        shape_n=args.shape_n,
        shape_k=args.shape_k,
        num_experts=args.num_experts,
        weight_config={
            "dtype": args.b_dtype,
            "group_size": args.weight_scale_group_size,
            "scale_dtype": args.bs_dtype,
            "has_zero_point": args.zero_point,
        },
        torch_dtype=torch_dtype,
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


def make_problem(args, torch_dtype, gemm_type: GemmType) -> dict:
    expert_max_tokens = math.ceil(args.shape_m * args.top_k / args.num_experts)
    _, expert_layout, *_ = generate_random_moe_tensors(
        shape_m=args.shape_m,
        num_experts=args.num_experts,
        top_k=args.top_k,
        gemm_type=gemm_type,
        balanced=True,
        expert_max_tokens=expert_max_tokens,
    )
    if gemm_type == GemmType.GROUPED_CONTIGUOUS:
        shape_m = args.shape_m * args.top_k
        valid_shape_m = 0
    else:
        shape_m = args.num_experts * expert_max_tokens
        valid_shape_m = args.shape_m * args.top_k
    inputs = torch.randn((shape_m, args.shape_k), dtype=torch_dtype, device="cuda:0")
    return {
        "inputs": inputs,
        "expert_layout": expert_layout,
        "top_k": args.top_k,
        "valid_shape_m": valid_shape_m,
        "compute_config": {"gemm_type": gemm_type.value},
    }


def with_block_m(config: dict, block_m: int) -> dict:
    _, block_n, block_k = config["block_shape"]
    # TS pins one M-warp, so the warp tile follows block_m.
    return config | {"block_shape": (block_m, block_n, block_k), "warp_shape": (block_m, 32, block_k)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape_n", type=int, required=True)
    parser.add_argument("--shape_k", type=int, required=True)
    parser.add_argument("--num_experts", type=int, required=True)
    parser.add_argument("--top_k", type=int, required=True)
    parser.add_argument("--shape_m", type=int, default=512)
    parser.add_argument("--a_dtype", type=str, choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--b_dtype", type=str, default="uint4")
    parser.add_argument("--bs_dtype", type=str, default="bfloat16")
    parser.add_argument("--weight_scale_group_size", type=int, default=128)
    parser.add_argument("--zero_point", default=False, action="store_true")
    # Overrides the tile the tokens-per-expert threshold picks.
    parser.add_argument("--block_m", type=int, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    torch_dtype = dtypes.torch_dtype_map[dtypes.DataType.from_str(args.a_dtype)]
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"N={args.shape_n} K={args.shape_k} E={args.num_experts} top_k={args.top_k}")
    layer = build_layer(args, torch_dtype)
    ts_layer = build_layer(args, torch_dtype, mma_type="tcgen05")

    header = f"{'mode':<12}{'M':>8}{'block_m':>9}{'mma us':>10}{'TS us':>10}{'mma/TS':>9}"
    print(header)
    print("-" * len(header))
    benchmark_result: list[dict[str, int | float]] = []
    for gemm_type in (GemmType.GROUPED_CONTIGUOUS, GemmType.GROUPED_MASKED):
        problem = make_problem(args, torch_dtype, gemm_type)
        shape_m = problem["inputs"].shape[0]
        ts_config = get_heuristics_config(ts_layer.humming_config, shape_m=shape_m, gemm_type=gemm_type)
        if args.block_m is not None:
            ts_config = with_block_m(ts_config, args.block_m)
        mma_config = get_heuristics_config(layer.humming_config, shape_m=shape_m, gemm_type=gemm_type)
        mma_us = bench(layer, mma_config, **problem)
        ts_us = bench(ts_layer, ts_config, **problem)
        block_m = ts_config["block_shape"][0]
        mode = gemm_type.value.split("_")[1]
        print(f"{mode:<12}{shape_m:>8}{block_m:>9}{mma_us:>10.1f}{ts_us:>10.1f}{mma_us / ts_us:>8.2f}x")
        benchmark_result.append(
            {
                "shape_m": f"{mode}-{shape_m}",
                "block_m": block_m,
                "mma_time": mma_us,
                "ts_time": ts_us,
            }
        )

    save_benchmark_result(benchmark_result, args)


if __name__ == "__main__":
    main()
