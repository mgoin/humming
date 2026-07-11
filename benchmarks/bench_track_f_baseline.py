"""Track-f baseline: SS-mode tcgen05 vs mma.sync at fixed prod-WS config.

Fixed config (workbook B.35/B.36 heuristic): BlockM=128 BlockN=128
BlockK=128 stages=4 warp-spec+TMA. mma.sync reference uses the bench's
default (bm=64 bk=64 s3). Shapes: Llama70B gate (28672x8192) and down
(8192x28672); M in {16, 128, 512, 2048}. M=16 pads to BlockM (noted).

Run: CUDA_VISIBLE_DEVICES=5 .venv/bin/python benchmarks/bench_track_f_baseline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bench_tcgen05_vs_wmma import bench_one, fmt

SHAPES = [
    ("Llama70B gate", 28672, 8192),
    ("Llama70B down", 8192, 28672),
]
MS = [16, 128, 512, 2048]


def main():
    print(f"{'shape':<16s} {'M':>5s} {'wmma us':>10s} {'tcg us':>10s} "
          f"{'tcg cfg':>16s} {'tcg/wmma':>9s}")
    for label, n, k in SHAPES:
        for m in MS:
            bm = 128 if m >= 128 else 64
            t_wmma = bench_one(m, n, k, "mma")
            t_tcg = bench_one(m, n, k, "tcgen05", block_m=bm, block_k=128,
                              num_stages=4, use_warp_spec=True)
            if isinstance(t_wmma, float) and isinstance(t_tcg, float):
                ratio = f"{t_wmma / t_tcg:>8.2f}x"
            else:
                ratio = "   --   "
            cfg = f"M{bm}K128s4+ws"
            print(f"  {label:<14s} {m:>5d} {fmt(t_wmma):>10s} "
                  f"{fmt(t_tcg):>10s} {cfg:>16s} {ratio:>9s}")
        print()


if __name__ == "__main__":
    main()
