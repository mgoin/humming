"""Track d-packing same-GPU SS-mode baseline.

Times humming mma.sync vs the shipped SS-mode TCGEN05 kernel at the
workbook standard shapes (Llama70B gate 28672x8192 and down 8192x28672),
M in {16, 128, 512, 2048}. Production tcgen05 config per workbook B.35:
BlockM=128 BlockN=128 BlockK=128 stages=4 WS (BlockM=64 when M < 128).

Run: CUDA_VISIBLE_DEVICES=3 .venv/bin/python benchmarks/bench_baseline_track_d.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bench_tcgen05_vs_wmma import bench_one  # noqa: E402

SHAPES = [
    ("Llama70B gate", 28672, 8192),
    ("Llama70B down", 8192, 28672),
]
MS = [16, 128, 512, 2048]


def main():
    print(f"{'shape':<16s} {'M':>5s} {'mma.sync us':>12s} {'tcg-ss us':>12s} "
          f"{'cfg':>16s} {'tcg/mma':>8s}")
    for label, n, k in SHAPES:
        for m in MS:
            t_wmma = bench_one(m, n, k, "mma")
            bm = 128 if m >= 128 else 64
            t_tcg = bench_one(m, n, k, "tcgen05", block_m=bm, block_k=128,
                              num_stages=4, use_warp_spec=True)
            cfg = f"M{bm}K128s4+ws"
            if isinstance(t_wmma, float) and isinstance(t_tcg, float):
                ratio = f"{t_wmma / t_tcg:.2f}x"
            else:
                ratio = "--"
            def f(t):
                return f"{t:12.1f}" if isinstance(t, float) else f"{t:>12s}"[:12]
            print(f"{label:<16s} {m:>5d} {f(t_wmma)} {f(t_tcg)} {cfg:>16s} {ratio:>8s}")


if __name__ == "__main__":
    main()
