"""Track-E baseline: SS-mode TCGEN05 (production WS config) vs mma.sync.

Shapes: workbook standard (Llama70B gate/down) at M in {16, 128, 512, 2048}
plus the epilogue-exposed regime (small tile count: M=256 on 4096x4096, and
M=128 on Llama8B qkv).

Run:  CUDA_VISIBLE_DEVICES=4 .venv/bin/python benchmarks/bench_track_e_baseline.py
"""
import sys
import time

import torch

from humming import dtypes, ops
from humming.kernel.humming import HummingKernel
from humming.utils.test import generate_random_inputs, generate_random_weight
from humming.utils.weight import (
    prepare_humming_weight,
    prepare_humming_weight_scale,
    prepare_humming_zero_point,
)

A_DTYPE = dtypes.bfloat16
B_DTYPE = dtypes.uint4
C_DTYPE = dtypes.bfloat16
BS_DTYPE = dtypes.bfloat16
GROUP_SIZE = 128


def time_kernel(launch_fn, warmup=10, iters=50):
    for _ in range(warmup):
        launch_fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        launch_fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


def build_launcher(shape_m, shape_n, shape_k, mma_type, block_m=128,
                   block_n=128, block_k=128, num_stages=4,
                   use_warp_spec=True):
    if mma_type == "tcgen05":
        block_shape = (block_m, block_n, block_k)
        warp_shape = (block_m // 4, 64, block_k)
    else:
        block_shape = (64, block_n, 64)
        warp_shape = (16, 64, 64)
        use_warp_spec = False
        num_stages = 3

    torch.manual_seed(123)
    w = generate_random_weight(
        n=shape_n, k=shape_k, group_size=GROUP_SIZE,
        dtype=B_DTYPE, scale_dtype=BS_DTYPE, has_zero_point=True,
    )
    _, _, weight, weight_scale, zero_point, _ = w
    weight_p = prepare_humming_weight(
        weight, B_DTYPE, A_DTYPE, zero_point=zero_point, use_wgmma=False,
    )
    weight_scale_p = prepare_humming_weight_scale(weight_scale, to_apply_on_c=False)
    zero_point_p = prepare_humming_zero_point(zero_point, dtype=B_DTYPE)
    _, _, inputs, _ = generate_random_inputs(
        m=shape_m, k=shape_k, group_size=0, dtype=A_DTYPE,
    )
    outputs = torch.empty((shape_m, shape_n), dtype=torch.bfloat16, device="cuda")
    kernel = HummingKernel(
        shape_n=shape_n, shape_k=shape_k,
        block_shape=block_shape, warp_shape=warp_shape,
        a_dtype=A_DTYPE, b_dtype=B_DTYPE, c_dtype=C_DTYPE, bs_dtype=BS_DTYPE,
        weight_scale_group_size=GROUP_SIZE, has_zero_point=True,
        num_stages=num_stages,
        use_warp_spec=use_warp_spec,
        use_tma=use_warp_spec,
        use_cp_async=not use_warp_spec,
        use_mbarrier=use_warp_spec,
        use_tma_bzp=False,
        has_bias=False, mma_type=mma_type,
        use_tcgen05=(mma_type == "tcgen05"), use_stream_k=False,
    )

    def launch():
        ops.launch_kernel(
            configs=[kernel.kernel_id], inputs=inputs, weight=weight_p,
            outputs=outputs, weight_scale=weight_scale_p,
            zero_point=zero_point_p,
        )

    return launch


def round_up(x, m):
    return ((x + m - 1) // m) * m


CASES = [
    # (label, N, K, [Ms])
    ("Llama70B gate", 28672, 8192, (16, 128, 512, 2048, 4096)),
    ("Llama70B down", 8192, 28672, (16, 128, 512, 2048, 4096)),
    ("4096x4096",      4096, 4096, (128, 256, 512, 2048)),
    ("Llama8B qkv",    6144, 4096, (128, 256, 2048)),
]


def main():
    torch.cuda.set_device(0)
    print(f"# device={torch.cuda.get_device_name(0)} "
          f"cc={torch.cuda.get_device_capability(0)}")
    print(f"{'shape':<16s} {'M':>5s} {'mma.sync us':>12s} {'tcg-SS us':>12s} "
          f"{'tcg/mma':>8s}  cfg")
    for label, n, k, ms in CASES:
        for m in ms:
            sm = round_up(max(m, 128), 128)
            t_mma = time_kernel(build_launcher(max(m, 1), n, k, "mma"))
            t_tcg = time_kernel(build_launcher(sm, n, k, "tcgen05"))
            ratio = t_mma / t_tcg
            print(f"{label:<16s} {m:>5d} {t_mma:>12.1f} {t_tcg:>12.1f} "
                  f"{ratio:>7.2f}x  BM128/BN128/BK128 s4 WS (tcg M pad {sm})")
        print()


if __name__ == "__main__":
    main()
