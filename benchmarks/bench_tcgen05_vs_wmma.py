"""tcgen05 SS mode vs mma.sync across LLM projection shapes, W4A16.

Sweeps M for the projection shapes of Llama-3 and Mixtral and marks the points
where the sm100 heuristic actually selects SS mode, which is where its cutoffs
in humming/tune/sm100.py come from.
"""

import torch
import triton

from humming.config import GemmType, LayerConfig
from humming.layer import HummingLayer
from humming.testing import random_fill_tensor
from humming.tune.raster import raster_group_m_for_config
from humming.tune.sm100 import Sm100Heuristics

WEIGHT_CONFIG = {
    "dtype": "uint4",
    "group_size": 128,
    "scale_dtype": "bfloat16",
    "has_zero_point": True,
}

SHAPES = [
    ("Llama8B qkv", 6144, 4096),
    ("Llama8B gate", 14336, 4096),
    ("Llama8B down", 4096, 14336),
    ("Llama70B qkv", 10240, 8192),
    ("Llama70B gate", 28672, 8192),
    ("Llama70B down", 8192, 28672),
    ("Mixtral gate", 14336, 4096),
]
SHAPE_MS = [1, 16, 64, 128, 256, 512, 1024, 2048]


class _MmaSyncHeuristics(Sm100Heuristics):
    """The sm100 heuristic with SS mode switched off: the mma.sync baseline."""

    @classmethod
    def _ss_config(cls, *args, **kwargs) -> dict | None:
        return None


def build_layer(shape_n: int, shape_k: int) -> HummingLayer:
    torch.manual_seed(2026)
    layer = HummingLayer(
        shape_n=shape_n,
        shape_k=shape_k,
        weight_config=WEIGHT_CONFIG,
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


def mma_sync_config(layer_config: LayerConfig, shape_m: int) -> dict:
    config = _MmaSyncHeuristics.get_config(layer_config=layer_config, shape_m=shape_m)
    config["raster_group_m"] = raster_group_m_for_config(layer_config, config["block_shape"])
    return config


def main() -> None:
    print(f"device: {torch.cuda.get_device_name(0)}")
    header = f"{'M':>6}{'mma us':>10}{'SS us':>10}{'mma/SS':>9}{'shipped':>9}"
    for label, shape_n, shape_k in SHAPES:
        layer = build_layer(shape_n, shape_k)
        layer_config = layer.humming_config
        # Only the SS profitability gates depend on shape_m, not the geometry it
        # returns, so take the config from above the cutoff and time it at every
        # M -- that is what makes the cutoff itself measurable here.
        ss_config = Sm100Heuristics._ss_config(layer_config, 1 << 20, False, GemmType.DENSE)
        print(f"\n{label}: N={shape_n} K={shape_k}")
        print(header)
        print("-" * len(header))
        for shape_m in SHAPE_MS:
            inputs = torch.randn((shape_m, shape_k), dtype=torch.bfloat16, device="cuda:0")
            mma_us = bench(layer, inputs, mma_sync_config(layer_config, shape_m))
            ss_us = bench(layer, inputs, ss_config)
            shipped = Sm100Heuristics.get_config(layer_config, shape_m).get("use_tcgen05", False)
            print(
                f"{shape_m:>6}{mma_us:>10.1f}{ss_us:>10.1f}"
                f"{mma_us / ss_us:>8.2f}x{'SS' if shipped else 'mma':>9}"
            )


if __name__ == "__main__":
    main()
