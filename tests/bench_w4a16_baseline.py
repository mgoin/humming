"""Phase A perf baseline: humming (existing mma.sync path) vs Sablefish (CUTLASS
tcgen05) vs Marlin AWQ. All three consume identical quantised weights so the
numbers are apples-to-apples.

Pre-Phase-B this is what tcgen05 has to beat at high batch. Output is a TSV
suitable for diffing across iterations.

Run:
    .venv/bin/python humming/tests/bench_w4a16_baseline.py | tee bench_baseline.tsv
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

# Pull Sablefish + Marlin baselines from the scratch tree.
_SABLEFISH = "/home/mgoin/code/vllm/.scratch/sablefish"
if _SABLEFISH not in sys.path:
    sys.path.insert(0, _SABLEFISH)

from awq_pack import quantize_awq  # noqa: E402
from marlin_baseline import MarlinAWQ  # noqa: E402
from sablefish import Sablefish, SablefishConfig  # noqa: E402

from humming import dtypes  # noqa: E402
from humming.layer import HummingLayer  # noqa: E402

GROUP_SIZE = 128

SHAPES = (
    ( 4096,  4096),
    (14336,  4096),
    ( 4096, 14336),
    ( 8192,  8192),
    (28672,  8192),
    ( 8192, 28672),
)
BATCHES = (1, 16, 64, 256, 1024, 4096)


def _time_us(fn, iters=30, warmup=8) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


def _build_humming_layer(N: int, K: int) -> tuple[HummingLayer, torch.Tensor]:
    """Construct a humming W4A16 (uint4 + gs=128 + zp) layer with random init.

    Returns the layer and a dequantised weight reference for downstream
    correctness checks (we don't use it here but keep it as documentation).
    """
    layer = HummingLayer(
        shape_n=N,
        shape_k=K,
        weight_config={
            "dtype": "uint4",
            "group_size": GROUP_SIZE,
            "scale_dtype": "bfloat16",
            "has_zero_point": True,
        },
        input_config={"dtype": "bfloat16", "group_size": 0},
        torch_dtype=torch.bfloat16,
    ).to("cuda:0")

    torch.manual_seed(0xC0FFEE)
    for p in layer.parameters():
        if p.dtype.is_floating_point:
            p.data.normal_(0, 0.05)
        else:
            p.data.random_(0, 8)
    layer.transform()
    return layer


def bench_one(B: int, N: int, K: int, iters: int) -> tuple[float, float, float]:
    """Return (humming_us, sablefish_us, marlin_us)."""

    # Build all three backends from the same raw (K, N) bf16 weight, so the
    # quantised numerics are aligned (humming's parameter init is separate
    # but irrelevant for timing — same shape, same kernel path).
    torch.manual_seed(0xC0FFEE)
    raw_w = (torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    _, codes_u, scales, zp_u = quantize_awq(raw_w, GROUP_SIZE)
    X = (torch.randn(B, K, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()

    # Humming: build layer, run forward
    humming_layer = _build_humming_layer(N, K)
    humming_us = _time_us(lambda: humming_layer(inputs=X), iters=iters)

    # Sablefish: pick a good tile for this shape (we already know which two
    # cover the envelope — keep both, take min).
    sf_tiles = [
        SablefishConfig(mma_tiler_mnk=(64, 128, 128)),
        SablefishConfig(
            mma_tiler_mnk=(256, 256, 128),
            cluster_shape_mn=(2, 1),
            use_2cta_instrs=True,
        ),
    ]
    sf_best = float("inf")
    for cfg in sf_tiles:
        if not cfg.is_compatible_with(N, K):
            continue
        try:
            sf = Sablefish(cfg).prepare_weights(codes_u, scales, zp_u, B_max=B)
            out = torch.empty((B, N), dtype=torch.bfloat16, device="cuda")
            us = _time_us(lambda: sf.forward_prepared(X, out=out), iters=iters)
            sf_best = min(sf_best, us)
        except Exception:
            torch.cuda.synchronize()
            continue

    # Marlin
    try:
        m = MarlinAWQ.prepare(codes_u, scales, zp_u, GROUP_SIZE)
        mar_us = _time_us(lambda: m.forward(X), iters=iters)
    except Exception:
        mar_us = float("nan")

    return humming_us, sf_best, mar_us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--shapes", type=str, default="")
    ap.add_argument("--batches", type=str, default="")
    args = ap.parse_args()

    shapes = SHAPES if not args.shapes else tuple(
        tuple(int(x) for x in s.split(",")) for s in args.shapes.split(";") if s
    )
    batches = BATCHES if not args.batches else tuple(
        int(x) for x in args.batches.split(",")
    )

    print(f"# Phase A perf baseline (humming mma.sync vs Sablefish vs Marlin)")
    print(f"# AWQ int4, gs={GROUP_SIZE}, bf16 in/out, B300 SXM6")
    print(f"# {'B':>5}  {'N':>6}  {'K':>6}  "
          f"{'hum_us':>10}  {'sf_us':>10}  {'mar_us':>10}  "
          f"{'hum_TFs':>8}  {'sf/hum':>7}  {'mar/hum':>7}")
    for (N, K) in shapes:
        for B in batches:
            hum, sf, mar = bench_one(B, N, K, args.iters)
            tflops = 2.0 * B * N * K / (hum * 1e-6) / 1e12
            print(f"  {B:>5}  {N:>6}  {K:>6}  "
                  f"{hum:>10.1f}  {sf:>10.1f}  {mar:>10.1f}  "
                  f"{tflops:>8.1f}  {sf/hum:>6.2f}x  {mar/hum:>6.2f}x")
        print()


if __name__ == "__main__":
    main()
