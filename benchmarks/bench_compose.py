"""Track a-ws-pipeline M8: composition matrix at IDENTICAL geometry.

Five kernels at the same block geometry (BlockM=128@M>=128 else 64,
BlockN=128, BlockK=64, stages=4, warp-spec + TMA):

  mma.sync   heuristic wmma reference
  ss         classic SS tcgen05 (r2s scatter + per-iter bar.sync)
  ss-wsp     SS + WS Transform->MMA pipeline (track a)
  ts         TS-mode tcgen05 (track b: r2t staging, per-iter bar.sync)
  ts-wsp     COMPOSED: TS mainloop in the WS pipeline (this track)

Run: CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_compose.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from ts_contract_pack import (  # noqa: E402
    pack_ts_weight,
    pack_ts_weight_scale,
    pack_ts_zero_point,
)

from humming import dtypes, ops  # noqa: E402
from humming.kernel.humming import HummingKernel  # noqa: E402
from humming.utils.test import (  # noqa: E402
    generate_random_inputs,
    generate_random_weight,
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
                   num_stages=4):
    weight_codes, weight_scale, zero_point, inputs, outputs = _gen_problem(
        shape_m, shape_n, shape_k)
    is_ts = mode.startswith("ts")
    wsp = mode.endswith("wsp")
    use_warp_spec = mode != "mma"

    if is_ts:
        weight_p = pack_ts_weight(weight_codes.cpu().to(torch.int32)).cuda()
        weight_scale_p = pack_ts_weight_scale(weight_scale).cuda()
        zero_point_p = pack_ts_zero_point(
            zero_point.cpu().to(torch.int32)).cuda()
        block_shape = (block_m, 128, 64)
        warp_shape = (block_m, 32, 64)
    else:
        weight_p = prepare_humming_weight(
            weight_codes, B_DTYPE, A_DTYPE, zero_point=zero_point,
            use_wgmma=False)
        weight_scale_p = prepare_humming_weight_scale(
            weight_scale, to_apply_on_c=False)
        zero_point_p = prepare_humming_zero_point(zero_point, dtype=B_DTYPE)
        if mode == "mma":
            block_shape = (block_m, 128, 64)
            warp_shape = (16, 64, 64)
        else:
            block_shape = (block_m, 128, 64)
            warp_shape = (block_m // 4, 64, 64)

    kernel = HummingKernel(
        shape_n=shape_n, shape_k=shape_k,
        block_shape=block_shape, warp_shape=warp_shape,
        a_dtype=A_DTYPE, b_dtype=B_DTYPE, c_dtype=C_DTYPE, bs_dtype=BS_DTYPE,
        weight_scale_group_size=GROUP_SIZE, has_zero_point=True,
        num_stages=num_stages if mode != "mma" else 3,
        use_warp_spec=use_warp_spec,
        use_tma=use_warp_spec,
        use_cp_async=not use_warp_spec,
        use_mbarrier=use_warp_spec,
        use_tma_bzp=False,
        has_bias=False, mma_type="mma" if mode == "mma" else "tcgen05",
        use_tcgen05=(mode != "mma"),
        use_tcgen05_ts=is_ts,
        use_ws_pipeline=wsp,
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


def ratio(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return f"{a / b:>6.2f}x"
    return "    --"


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    modes = ["mma", "ss", "ss-wsp", "ts", "ts-wsp"]
    hdr = " ".join(f"{m:>9s}" for m in modes)
    print(f"{'shape':<15s} {'M':>5s} {hdr} {'ts/tswsp':>8s} {'best-par/tswsp':>14s}")
    for label, n, k in SHAPES:
        for m in MS:
            bm = 128 if m >= 128 else 64
            ts = {mode: bench(m, n, k, mode, block_m=bm) for mode in modes}
            parents = [t for t in (ts["ss-wsp"], ts["ts"])
                       if isinstance(t, float)]
            best_parent = min(parents) if parents else None
            row = " ".join(fmt(ts[mode]) for mode in modes)
            print(f"{label:<15s} {m:>5d} {row} "
                  f"{ratio(ts['ts'], ts['ts-wsp']):>8s} "
                  f"{ratio(best_parent, ts['ts-wsp']):>14s}", flush=True)
    print()


if __name__ == "__main__":
    main()
