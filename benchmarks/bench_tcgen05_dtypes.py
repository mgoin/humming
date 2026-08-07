"""tcgen05 SS mode vs mma.sync across the weight-dtype matrix, bf16 activations.

The sweep behind the SS geometry rule in humming/tune/sm100.py: --b_dtype walks
TCGEN05_SS_B_DTYPES and --block_k / --num_stages walk the ladder the rule picks
from. Every SS point is checked against the mma.sync output on the same layer.
"""

import argparse
import itertools

import torch
import triton

from humming.config import LayerConfig
from humming.layer import HummingLayer
from humming.testing import random_fill_tensor, save_benchmark_result
from humming.tune import get_heuristics_config
from humming.tune.sm100 import TCGEN05_SS_B_DTYPES


def build_layer(args, b_dtype: str) -> HummingLayer:
    torch.manual_seed(2026)
    layer = HummingLayer(
        shape_n=args.shape_n,
        shape_k=args.shape_k,
        weight_config={
            "dtype": b_dtype,
            "group_size": args.weight_scale_group_size,
            "scale_dtype": args.bs_dtype,
            # Unsigned codes carry an integer zero point; float ones are symmetric.
            "has_zero_point": args.zero_point and "uint" in b_dtype,
        },
        torch_dtype=torch.bfloat16,
    ).to("cuda:0")
    for tensor in layer.parameters():
        random_fill_tensor(tensor)
    layer.transform()
    return layer


def bench(layer: HummingLayer, inputs: torch.Tensor, config: dict) -> float:
    def run():
        return layer(inputs=inputs, tuning_config=[[0, 1 << 30, config]])

    run()
    torch.cuda.synchronize()
    return triton.testing.do_bench(run, warmup=100, rep=1000) * 1e3


def ss_config(layer_config: LayerConfig, block_k: int, num_stages: int) -> dict:
    block_n = 128 if layer_config.shape_n % 128 == 0 else 64
    return {
        "block_shape": (128, block_n, block_k),
        "warp_shape": (32, 64, block_k),
        "num_ctas_per_sm": 1,
        "num_write_splits": 1,
        "mma_type": "tcgen05",
        "use_tcgen05": True,
        "use_warp_spec": True,
        "use_tma": True,
        "use_cp_async": False,
        "use_mbarrier": True,
        "use_tma_bzp": False,
        "use_stream_k": False,
        "raster_group_m": 1,
        "num_stages": num_stages,
    }


def best_ss(args, layer: HummingLayer, inputs: torch.Tensor, reference: torch.Tensor) -> tuple:
    """Fastest correct ladder entry, or (None, reason)."""
    best: tuple[float, str] | None = None
    for block_k, num_stages in itertools.product(args.block_k, args.num_stages):
        if layer.humming_config.shape_k % block_k:
            continue
        config = ss_config(layer.humming_config, block_k, num_stages)
        try:
            outputs = layer(inputs=inputs, tuning_config=[[0, 1 << 30, config]])
            torch.cuda.synchronize()
        except Exception:
            continue
        tolerance = 0.02 * reference.float().abs().max().item()
        if (outputs.float() - reference.float()).abs().max().item() > tolerance:
            return None, "WRONG"
        microseconds = bench(layer, inputs, config)
        if best is None or microseconds < best[0]:
            best = (microseconds, f"K{block_k}s{num_stages}")
    return best if best is not None else (None, "--")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape_n", type=int, required=True)
    parser.add_argument("--shape_k", type=int, required=True)
    parser.add_argument("--b_dtype", type=str, default=None, nargs="+")
    parser.add_argument("--bs_dtype", type=str, default="bfloat16")
    parser.add_argument("--weight_scale_group_size", type=int, default=128)
    parser.add_argument("--zero_point", default=False, action="store_true")
    parser.add_argument("--shape_m_list", type=int, default=[256, 2048], nargs="+")
    # Widest first: the bf16 b_dequant staging buffer pushes wide weight dtypes
    # over the SMEM cap at the deeper entries.
    parser.add_argument("--block_k", type=int, default=[128, 64], nargs="+")
    parser.add_argument("--num_stages", type=int, default=[4, 3], nargs="+")
    parser.add_argument("--output_file", type=str, default=None)
    # SS is bf16-only (mma/tcgen05_mma.cuh static_asserts it); recorded so the
    # saved result carries the activation dtype.
    parser.set_defaults(a_dtype="bfloat16")
    args = parser.parse_args()

    b_dtypes = args.b_dtype or [str(b_dtype) for b_dtype in TCGEN05_SS_B_DTYPES]
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"N={args.shape_n} K={args.shape_k}")
    header = f"{'weight':<12}{'zp':>6}{'M':>6}{'mma us':>10}{'SS us':>10}{'SS cfg':>9}{'mma/SS':>9}"
    print(header)
    print("-" * len(header))
    benchmark_result: list[dict[str, int | float | str]] = []
    for b_dtype in b_dtypes:
        layer = build_layer(args, b_dtype)
        has_zero_point = layer.humming_config.has_zero_point
        for shape_m in args.shape_m_list:
            inputs = torch.randn((shape_m, args.shape_k), dtype=torch.bfloat16, device="cuda:0")
            mma_config = get_heuristics_config(layer.humming_config, shape_m=shape_m)
            reference = layer(inputs=inputs, tuning_config=[[0, 1 << 30, mma_config]]).clone()
            mma_us = bench(layer, inputs, mma_config)
            ss_us, ss_label = best_ss(args, layer, inputs, reference)
            ratio = f"{mma_us / ss_us:>8.2f}x" if ss_us else f"{'--':>9}"
            print(
                f"{b_dtype:<12}{str(has_zero_point):>6}{shape_m:>6}{mma_us:>10.1f}"
                f"{ss_us if ss_us else float('nan'):>10.1f}{ss_label:>9}{ratio}"
            )
            benchmark_result.append(
                {
                    "shape_m": f"{b_dtype}-{shape_m}",
                    "has_zero_point": has_zero_point,
                    "mma_time": mma_us,
                    "ss_time": ss_us if ss_us else float("nan"),
                    "ss_config": ss_label,
                }
            )
        del layer
        torch.cuda.empty_cache()

    save_benchmark_result(benchmark_result, args)


if __name__ == "__main__":
    main()
