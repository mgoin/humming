"""SS-mode tcgen05 baseline for track a-ws-pipeline.

Times the production tcgen05 config (BlockM=128 BlockN=128 BlockK=128
stages=4 WS+TMA) and the mma.sync reference at the workbook's standard
Llama70B shapes, on THIS GPU. Every perf claim in
notes/track-a-ws-pipeline.md is relative to this table.

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/bench_ws_pipeline_baseline.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_tcgen05_vs_wmma import bench_one, fmt  # noqa: E402

SHAPES = [
    ("Llama70B gate", 28672, 8192),
    ("Llama70B down", 8192, 28672),
]
MS = [16, 128, 512, 2048]


def main():
    three_way = len(sys.argv) > 1 and sys.argv[1] == "--three-way"
    extra = {}
    if len(sys.argv) > 1 and sys.argv[1] == "--ws-pipeline":
        extra["use_ws_pipeline"] = True
    if three_way:
        print(f"{'shape':<16s} {'M':>5s} {'mma.sync':>10s} {'tcg-clas':>10s} "
              f"{'tcg-wsp':>10s} {'wsp/clas':>9s} {'wsp/mma':>9s}")
    else:
        print(f"{'shape':<16s} {'M':>5s} {'mma.sync us':>12s} {'tcg-prod us':>12s} {'ratio':>8s}")
    for label, n, k in SHAPES:
        for m in MS:
            t_wmma = bench_one(m, n, k, "mma")
            bm = 128 if m >= 128 else 64
            if three_way:
                t_c = bench_one(m, n, k, "tcgen05", block_m=bm, block_k=128,
                                num_stages=4, use_warp_spec=True)
                t_w = bench_one(m, n, k, "tcgen05", block_m=bm, block_k=128,
                                num_stages=4, use_warp_spec=True,
                                use_ws_pipeline=True)
                ok = all(isinstance(t, float) for t in (t_wmma, t_c, t_w))
                r1 = f"{t_c / t_w:>8.2f}x" if ok else "  --  "
                r2 = f"{t_wmma / t_w:>8.2f}x" if ok else "  --  "
                print(f"{label:<16s} {m:>5d} {fmt(t_wmma):>10s} {fmt(t_c):>10s} "
                      f"{fmt(t_w):>10s} {r1:>9s} {r2:>9s}", flush=True)
                continue
            t_tcg = bench_one(
                m, n, k, "tcgen05", block_m=bm, block_k=128,
                num_stages=4, use_warp_spec=True, **extra)
            ratio = (f"{t_wmma / t_tcg:>7.2f}x"
                     if isinstance(t_wmma, float) and isinstance(t_tcg, float)
                     else "  --  ")
            print(f"{label:<16s} {m:>5d} {fmt(t_wmma):>12s} {fmt(t_tcg):>12s} {ratio:>8s}",
                  flush=True)


if __name__ == "__main__":
    main()
