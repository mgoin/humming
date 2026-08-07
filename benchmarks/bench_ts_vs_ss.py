"""tcgen05 TS mode vs SS mode vs mma.sync on sm100, W4A16.

The three paths run the same layer, so the columns differ only in the mainloop:
mma.sync dequantises into registers, SS mode stages the dequantised weights in
SMEM for tcgen05.mma, and TS mode stages them in TMEM. The mma.sync column is
what the sm100 heuristic emits without the mma_type="tcgen05" opt-in, so it is
also the upstream baseline.
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
    ("Llama70B gate", 28672, 8192),
    ("Llama70B down", 8192, 28672),
]
SHAPE_MS = [16, 128, 512, 2048]


def build_layer(shape_n: int, shape_k: int, mma_type: str | None = None) -> HummingLayer:
    torch.manual_seed(2026)
    layer = HummingLayer(
        shape_n=shape_n,
        shape_k=shape_k,
        weight_config=WEIGHT_CONFIG,
        torch_dtype=torch.bfloat16,
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


def mma_sync_config(layer_config: LayerConfig, shape_m: int) -> dict:
    config = Sm100Heuristics.get_config(layer_config=layer_config, shape_m=shape_m)
    config["raster_group_m"] = raster_group_m_for_config(layer_config, config["block_shape"])
    return config


def config_label(config: dict) -> str:
    block_m, _, block_k = config["block_shape"]
    return f"M{block_m}K{block_k}s{config['num_stages']}"


def main() -> None:
    print(f"device: {torch.cuda.get_device_name(0)}")
    header = (
        f"{'shape':<15}{'M':>6}{'mma us':>10}{'SS us':>10}{'TS us':>10}"
        f"{'TS cfg':>10}{'mma/TS':>9}{'SS/TS':>8}"
    )
    print(header)
    print("-" * len(header))
    for label, shape_n, shape_k in SHAPES:
        layer = build_layer(shape_n, shape_k)
        ts_layer = build_layer(shape_n, shape_k, mma_type="tcgen05")
        # SS reads the ordinary weight layout, so it runs on the non-opted-in
        # layer; its geometry does not depend on shape_m.
        ss_config = Sm100Heuristics._ss_config(layer.humming_config, GemmType.DENSE)
        for shape_m in SHAPE_MS:
            inputs = torch.randn((shape_m, shape_k), dtype=torch.bfloat16, device="cuda:0")
            ts_config = Sm100Heuristics._ts_config(ts_layer.humming_config, shape_m, GemmType.DENSE)
            mma_us = bench(layer, inputs, mma_sync_config(layer.humming_config, shape_m))
            ss_us = bench(layer, inputs, ss_config)
            ts_us = bench(ts_layer, inputs, ts_config)
            print(
                f"{label:<15}{shape_m:>6}{mma_us:>10.1f}{ss_us:>10.1f}{ts_us:>10.1f}"
                f"{config_label(ts_config):>10}{mma_us / ts_us:>8.2f}x{ss_us / ts_us:>7.2f}x"
            )
        print()


if __name__ == "__main__":
    main()
