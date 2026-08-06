"""tcgen05 SS mode vs mma.sync across the weight-dtype matrix, bf16 activations.

For each weight dtype this walks the (block_k, num_stages) ladder the SS mainloop
accepts, checks the result against the mma.sync path on the same layer, and
reports the fastest correct entry -- the sweep behind _SS_B_DTYPE_CONFIG in
humming/tune/sm100.py. Dtypes that no ladder entry fits, or that the SS mainloop
rejects, print "--" and stay out of that table.
"""

import torch
import triton

from humming.config import LayerConfig
from humming.layer import HummingLayer
from humming.testing import random_fill_tensor
from humming.tune.raster import raster_group_m_for_config
from humming.tune.sm100 import Sm100Heuristics

GROUP_SIZE = 128

# (weight dtype, has_zero_point) -- unsigned codes carry an integer zero point,
# signed and float codes are symmetric.
B_DTYPES = [
    ("uint2", True),
    ("uint3", True),
    ("uint4", True),
    ("uint4", False),
    ("uint5", True),
    ("uint6", True),
    ("uint7", True),
    ("uint8", True),
    ("int4", False),
    ("int8", False),
    ("float4e2m1", False),
    ("float6e2m3", False),
    ("float6e3m2", False),
    ("float8e4m3", False),
    ("float8e5m2", False),
]

SHAPES = [
    ("Llama8B gate", 14336, 4096),
    ("Llama70B down", 8192, 28672),
]
SHAPE_MS = [256, 2048]

# Widest first; wide weight dtypes need fewer stages or block_k=64 because the
# bf16 b_dequant staging buffer pushes them over the SMEM cap.
SS_LADDER = [(128, 4), (128, 3), (64, 4), (64, 3)]


class _MmaSyncHeuristics(Sm100Heuristics):
    """The sm100 heuristic with SS mode switched off: the mma.sync baseline."""

    @classmethod
    def _ss_config(cls, *args, **kwargs) -> dict | None:
        return None


def build_layer(shape_n: int, shape_k: int, b_dtype: str, has_zero_point: bool) -> HummingLayer:
    torch.manual_seed(2026)
    layer = HummingLayer(
        shape_n=shape_n,
        shape_k=shape_k,
        weight_config={
            "dtype": b_dtype,
            "group_size": GROUP_SIZE,
            "scale_dtype": "bfloat16",
            "has_zero_point": has_zero_point,
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


def mma_sync_config(layer_config: LayerConfig, shape_m: int) -> dict:
    config = _MmaSyncHeuristics.get_config(layer_config=layer_config, shape_m=shape_m)
    config["raster_group_m"] = raster_group_m_for_config(layer_config, config["block_shape"])
    return config


def ss_config(block_k: int, num_stages: int) -> dict:
    return {
        "block_shape": (128, 128, block_k),
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


def best_ss(layer: HummingLayer, inputs: torch.Tensor, reference: torch.Tensor) -> tuple:
    best: tuple[float, str] | None = None
    for block_k, num_stages in SS_LADDER:
        if layer.humming_config.shape_k % block_k:
            continue
        config = ss_config(block_k, num_stages)
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
    print(f"device: {torch.cuda.get_device_name(0)}")
    header = f"{'weight':<12}{'zp':>6}{'M':>6}{'mma us':>10}{'SS us':>10}{'SS cfg':>9}{'mma/SS':>9}"
    for label, shape_n, shape_k in SHAPES:
        print(f"\n{label}: N={shape_n} K={shape_k}")
        print(header)
        print("-" * len(header))
        for b_dtype, has_zero_point in B_DTYPES:
            layer = build_layer(shape_n, shape_k, b_dtype, has_zero_point)
            for shape_m in SHAPE_MS:
                inputs = torch.randn((shape_m, shape_k), dtype=torch.bfloat16, device="cuda:0")
                mma_config = mma_sync_config(layer.humming_config, shape_m)
                reference = layer(inputs=inputs, tuning_config=[[0, 1 << 30, mma_config]]).clone()
                mma_us = bench(layer, inputs, mma_config)
                ss_us, ss_label = best_ss(layer, inputs, reference)
                ratio = f"{mma_us / ss_us:>8.2f}x" if ss_us else f"{'--':>9}"
                print(
                    f"{b_dtype:<12}{str(has_zero_point):>6}{shape_m:>6}{mma_us:>10.1f}"
                    f"{ss_us if ss_us else float('nan'):>10.1f}{ss_label:>9}{ratio}"
                )


if __name__ == "__main__":
    main()
