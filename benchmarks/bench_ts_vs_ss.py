"""tcgen05 TS mode vs SS mode vs mma.sync on sm100.

The sweep behind TS being the mainloop the tcgen05 opt-in selects, and behind
_TS_B_DTYPE_STAGES in humming/tune/sm100.py: --b_dtype picks the weight dtype
and --num_stages overrides the TS pipeline depth that table caps.
"""

import argparse

import torch
import triton

from humming import dtypes
from humming.config import GemmType
from humming.layer import HummingLayer
from humming.testing import random_fill_tensor, save_benchmark_result
from humming.tune import get_heuristics_config
from humming.tune.sm100 import Sm100Heuristics


def build_layer(args, torch_dtype, mma_type: str | None = None) -> HummingLayer:
    torch.manual_seed(2026)
    layer = HummingLayer(
        shape_n=args.shape_n,
        shape_k=args.shape_k,
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


def bench(layer: HummingLayer, inputs: torch.Tensor, config: dict) -> float:
    def run():
        return layer(inputs=inputs, tuning_config=[[0, 1 << 30, config]])

    run()
    torch.cuda.synchronize()
    return triton.testing.do_bench(run, warmup=100, rep=1000) * 1e3


def config_label(config: dict) -> str:
    block_m, _, block_k = config["block_shape"]
    return f"M{block_m}K{block_k}s{config['num_stages']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape_n", type=int, required=True)
    parser.add_argument("--shape_k", type=int, required=True)
    parser.add_argument("--a_dtype", type=str, choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--b_dtype", type=str, default="uint4")
    parser.add_argument("--bs_dtype", type=str, default="bfloat16")
    parser.add_argument("--weight_scale_group_size", type=int, default=128)
    parser.add_argument("--zero_point", default=False, action="store_true")
    parser.add_argument("--shape_m_list", type=int, default=[16, 128, 512, 2048], nargs="+")
    # Overrides the depth _TS_B_DTYPE_STAGES caps; unset runs the heuristic's.
    parser.add_argument("--num_stages", type=int, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    torch_dtype = dtypes.torch_dtype_map[dtypes.DataType.from_str(args.a_dtype)]
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"N={args.shape_n} K={args.shape_k} {args.a_dtype} x {args.b_dtype}")
    layer = build_layer(args, torch_dtype)
    ts_layer = build_layer(args, torch_dtype, mma_type="tcgen05")
    layer_config = layer.humming_config
    # SS has no public entry point: the opt-in prefers TS for every weight dtype
    # TS is wired for. It reads the ordinary weight layout, so it runs on the
    # layer that did not opt in.
    ss_config = Sm100Heuristics._ss_config(layer_config, GemmType.DENSE)

    header = f"{'M':>6}{'mma us':>10}{'SS us':>10}{'TS us':>10}{'TS cfg':>10}{'mma/TS':>9}{'SS/TS':>8}"
    print(header)
    print("-" * len(header))
    benchmark_result: list[dict[str, int | float]] = []
    for shape_m in args.shape_m_list:
        inputs = torch.randn((shape_m, args.shape_k), dtype=torch_dtype, device="cuda:0")
        ts_config = get_heuristics_config(ts_layer.humming_config, shape_m=shape_m)
        if args.num_stages is not None:
            ts_config = ts_config | {"num_stages": args.num_stages}
        mma_us = bench(layer, inputs, get_heuristics_config(layer_config, shape_m=shape_m))
        ss_us = bench(layer, inputs, ss_config) if ss_config is not None else float("nan")
        ts_us = bench(ts_layer, inputs, ts_config)
        print(
            f"{shape_m:>6}{mma_us:>10.1f}{ss_us:>10.1f}{ts_us:>10.1f}"
            f"{config_label(ts_config):>10}{mma_us / ts_us:>8.2f}x{ss_us / ts_us:>7.2f}x"
        )
        benchmark_result.append(
            {
                "shape_m": shape_m,
                "num_stages": ts_config["num_stages"],
                "mma_time": mma_us,
                "ss_time": ss_us,
                "ts_time": ts_us,
            }
        )

    save_benchmark_result(benchmark_result, args)


if __name__ == "__main__":
    main()
