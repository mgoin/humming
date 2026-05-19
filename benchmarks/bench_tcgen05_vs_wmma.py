"""Benchmark TCGEN05 vs mma.sync across realistic LLM weight shapes.

Sweeps M (= batch × seq) from 1 to 2048 for several common projection
weight shapes from Llama-3 / Mixtral. Both kernels use the same
W4A16 setup (bf16 A × uint4 B with bf16 group scales, zero-points).

Phase B.23 result (B300, sm_103a): TCGEN05 wins consistently at
M >= 128 by 1.1×-1.3×. M < 64 is slower (TCGEN05 path has BlockM=64
minimum and pads up; the M<64 case is dominated by the per-CTA setup
overhead with very few output tiles). The crossover is around M=128
for thin layers and M=256 for fat-K layers.

Run with: ~/venvs/vllm-rel/bin/python benchmarks/bench_tcgen05_vs_wmma.py
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
    return (time.perf_counter() - t0) / iters * 1e6  # microseconds


def build_launcher(shape_m, shape_n, shape_k, mma_type, block_m=64,
                   block_n=128, block_k=64, num_stages=3,
                   use_warp_spec=False):
    if mma_type == "tcgen05":
        # TCGEN05 path: BlockM ∈ {64, 128}, WarpM = BlockM/4.
        block_shape = (block_m, block_n, block_k)
        warp_shape = (block_m // 4, 64, block_k)
    else:
        block_shape = (block_m, block_n, 64)
        warp_shape = (16, 64, 64)
        use_warp_spec = False  # WS for wmma reference disabled

    torch.manual_seed(123)
    w = generate_random_weight(
        n=shape_n, k=shape_k, group_size=GROUP_SIZE,
        dtype=B_DTYPE, scale_dtype=BS_DTYPE, has_zero_point=True,
    )
    _, _, weight, weight_scale, zero_point, _ = w
    # Phase B.30: `prepare_humming_weight` must receive `zero_point` so the
    # repacker takes the sign-magnitude with-zp preprocessing path the
    # kernel expects (utils/weight.py:203-211). Without it, the repacker
    # silently produces no-zp-shaped weight bytes and the kernel runs the
    # with-zp dequant on them -- the timing is similar but the output is
    # garbage. We don't check correctness here, but if you ever copy this
    # block to a correctness test, remember the kwarg.
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
            outputs=outputs, weight_scale=weight_scale_p, zero_point=zero_point_p,
        )

    return launch


def round_up(x, m):
    return ((x + m - 1) // m) * m


def bench_one(shape_m, shape_n, shape_k, mma_type, **kw):
    sn = round_up(shape_n, 64)
    sk = round_up(shape_k, 64)
    block_m = kw.get("block_m", 64)
    sm = (round_up(max(shape_m, block_m), block_m)
          if mma_type == "tcgen05" else max(shape_m, 1))
    try:
        return time_kernel(build_launcher(sm, sn, sk, mma_type, **kw))
    except Exception as e:
        return f"FAIL: {e!s:.50s}"


SHAPES = [
    # (label, N, K) -- N is output dim, K is contraction dim.
    ("Llama8B qkv",         6144,  4096),
    ("Llama8B gate",       14336,  4096),
    ("Llama8B down",        4096, 14336),
    ("Llama70B qkv",       10240,  8192),
    ("Llama70B gate",      28672,  8192),
    ("Llama70B down",       8192, 28672),
    ("Mixtral expert gate",14336,  4096),
]
MS = [1, 16, 64, 128, 256, 512, 1024, 2048]


def fmt(t):
    return f"{t:>9.2f}" if isinstance(t, float) else f"{t:>9s}"[:9]


def best_tcgen05(m, n, k):
    """Pick the best TCGEN05 launcher across (block_m, block_k,
    use_warp_spec)."""
    candidates = []
    for bm in [64, 128] if m >= 128 else [64]:
        for bk in [64, 128]:
            if bk == 128 and k % 128 != 0:
                continue
            for ws in [False, True]:
                # bk=128 + stages=4 exceeds SMEM budget at BlockN=128.
                ns = 3 if bk == 128 else 4
                t = bench_one(m, n, k, "tcgen05",
                              block_m=bm, block_k=bk,
                              num_stages=ns, use_warp_spec=ws)
                if isinstance(t, float):
                    candidates.append((t, bm, bk, ws))
    if not candidates:
        return None, None, None, None
    t, bm, bk, ws = min(candidates)
    return t, bm, bk, ws


def main():
    print(
        f"{'shape':<22s} {'M':>5s} {'wmma µs':>10s} {'tcg µs':>10s} "
        f"{'tcg cfg':>13s} {'tcg/wmma':>10s}"
    )
    for label, n, k in SHAPES:
        for m in MS:
            t_wmma = bench_one(m, n, k, "mma")
            t_tcg, bm, bk, ws = best_tcgen05(m, n, k)
            if isinstance(t_wmma, float) and isinstance(t_tcg, float):
                ratio = f"{t_wmma / t_tcg:>9.2f}x"
                cfg = f"M{bm}K{bk}{'+ws' if ws else ''}"
            else:
                ratio = "   --   "
                cfg = "  --  "
            print(f"  {label:<20s} {m:>5d} {fmt(t_wmma):>10s} "
                  f"{fmt(t_tcg):>10s} {cfg:>13s} {ratio:>10s}")
        print()


if __name__ == "__main__":
    main()
