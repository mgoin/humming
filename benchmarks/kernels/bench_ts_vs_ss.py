"""Track b-ts-staging M5: TS-mode vs SS-mode tcgen05 vs mma.sync.

Same shapes / timing loop as bench_ts_baseline.py (the recorded M1
baseline). TS prototype config: BlockN=128, BlockK=64, WarpN=32,
BlockM in {64,128}, stages swept where SMEM allows.

Run: CUDA_VISIBLE_DEVICES=1 .venv/bin/python benchmarks/bench_ts_vs_ss.py
"""
import time

import torch

from humming import dtypes, ops  # noqa: E402
from humming.kernel.humming import HummingKernel  # noqa: E402
from humming.utils.test import (  # noqa: E402
    generate_random_inputs,
    generate_random_weight,
)
from humming.utils.ts_packing import (  # noqa: E402
    pack_scales_tcgen05_ts,
    pack_zero_point_tcgen05_ts,
)
from humming.utils.weight import (  # noqa: E402
    prepare_humming_weight,
    prepare_humming_weight_scale,
    prepare_humming_zero_point,
)

A_DTYPE = dtypes.bfloat16
B_DTYPE = dtypes.uint4
C_DTYPE = dtypes.bfloat16
BS_DTYPE = dtypes.bfloat16
GROUP_SIZE = 128

SHAPES = [
    ("Llama70B-gate", 28672, 8192),
    ("Llama70B-down", 8192, 28672),
]
MS = [16, 128, 512, 2048]


def time_kernel(launch_fn, warmup=10, iters=50):
    for _ in range(warmup):
        launch_fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        launch_fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def _gen_problem(shape_m, shape_n, shape_k):
    torch.manual_seed(123)
    w = generate_random_weight(
        n=shape_n, k=shape_k, group_size=GROUP_SIZE,
        dtype=B_DTYPE, scale_dtype=BS_DTYPE, has_zero_point=True,
    )
    _, _, weight_codes, weight_scale, zero_point, _ = w
    _, _, inputs, _ = generate_random_inputs(
        m=shape_m, k=shape_k, group_size=0, dtype=A_DTYPE,
    )
    outputs = torch.empty(
        (shape_m, shape_n), dtype=torch.bfloat16, device="cuda")
    return weight_codes, weight_scale, zero_point, inputs, outputs


def build_launcher(shape_m, shape_n, shape_k, mode, block_m=128,
                   block_k=64, num_stages=4, use_warp_spec=True):
    weight_codes, weight_scale, zero_point, inputs, outputs = _gen_problem(
        shape_m, shape_n, shape_k)

    if mode == "ts":
        weight_p = prepare_humming_weight(
            weight_codes, B_DTYPE, A_DTYPE, zero_point=zero_point,
            use_wgmma=False, use_tcgen05_ts=True)
        weight_scale_p = pack_scales_tcgen05_ts(weight_scale).cuda()
        zero_point_p = pack_zero_point_tcgen05_ts(
            zero_point.to(torch.int32), B_DTYPE.num_bits).cuda()
        block_shape = (block_m, 128, 64)
        warp_shape = (block_m, 32, 64)
    else:
        weight_p = prepare_humming_weight(
            weight_codes, B_DTYPE, A_DTYPE, zero_point=zero_point,
            use_wgmma=False)
        weight_scale_p = prepare_humming_weight_scale(
            weight_scale, to_apply_on_c=False)
        zero_point_p = prepare_humming_zero_point(zero_point, dtype=B_DTYPE)
        if mode == "tcgen05":
            block_shape = (block_m, 128, block_k)
            warp_shape = (block_m // 4, 64, block_k)
        else:
            block_shape = (64, 128, 64)
            warp_shape = (16, 64, 64)
            use_warp_spec = False

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
        has_bias=False, mma_type="mma" if mode == "mma" else "tcgen05",
        use_tcgen05=(mode != "mma"),
        use_tcgen05_ts=(mode == "ts"),
        use_stream_k=False,
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


def bench(shape_m, shape_n, shape_k, mode, **kw):
    bm = kw.get("block_m", 128)
    sm = round_up(max(shape_m, bm), bm) if mode != "mma" else shape_m
    try:
        return time_kernel(build_launcher(sm, shape_n, shape_k, mode, **kw))
    except Exception as e:
        return f"FAIL:{e!s:.40s}"


def fmt(t):
    return f"{t:>9.1f}" if isinstance(t, float) else f"{t:>9s}"[:9]


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'shape':<15s} {'M':>5s} {'mma us':>9s} {'SS us':>9s} "
          f"{'TS cfg':>14s} {'TS us':>9s} {'SS/TS':>7s} {'mma/TS':>7s}")
    for label, n, k in SHAPES:
        for m in MS:
            t_mma = bench(m, n, k, "mma")
            bm = 128 if m >= 128 else 64
            t_ss = bench(m, n, k, "tcgen05", block_m=bm, block_k=128,
                         num_stages=4, use_warp_spec=True)
            # TS: sweep stages x ws, keep the best
            best = None
            for s in (4, 6):
                for ws in (True, False):
                    t = bench(m, n, k, "ts", block_m=bm, num_stages=s,
                              use_warp_spec=ws)
                    if isinstance(t, float) and (
                            best is None or t < best[0]):
                        best = (t, s, ws)
            if best:
                t_ts, s, ws = best
                cfg = f"M{bm}K64s{s}{'+ws' if ws else ''}"
            else:
                t_ts, cfg = "FAIL", "--"
            r_ss = (f"{t_ss / t_ts:>6.2f}x"
                    if isinstance(t_ts, float) and isinstance(t_ss, float)
                    else "--")
            r_mma = (f"{t_mma / t_ts:>6.2f}x"
                     if isinstance(t_ts, float) and isinstance(t_mma, float)
                     else "--")
            print(f"{label:<15s} {m:>5d} {fmt(t_mma)} {fmt(t_ss)} "
                  f"{cfg:>14s} {fmt(t_ts)} {r_ss:>7s} {r_mma:>7s}")
    print()


if __name__ == "__main__":
    main()
