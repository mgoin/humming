"""Track b-ts-staging M6: r2t-vs-r2s evidence IN SITU.

The microbench (benchmarks/bench_r2s_vs_r2t.cu) says the staging
primitive costs 43 (r2t) vs 2052 (r2s scatter + bar.sync) cycles per
K-iter. Swordfish evidence says the pipeline restructure, not the
primitive, is where wall-time lives. This bench pins both kernels to
the IDENTICAL block geometry (BM=128 BN=128 BK=64 s=4 WS+TMA) so the
ONLY difference is the staging path + its sync:
  SS: dequant -> r2s scatter (smem.b_dequant) -> bar.sync 256 -> mma(SS)
  TS: dequant -> r2t (tcgen05.st) -> bar.sync 128 + fences ->
      mma(TS) + per-iter commit/mbar WAR gate

Run: CUDA_VISIBLE_DEVICES=1 .venv/bin/python benchmarks/bench_ts_m6_insitu.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from bench_ts_vs_ss import SHAPES, bench, fmt  # noqa: E402


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Identical geometry BM=128 BN=128 BK=64 s=4 WS "
          "(only the staging path differs):")
    print(f"{'shape':<15s} {'M':>5s} {'SS-K64 us':>10s} {'TS-K64 us':>10s} "
          f"{'SS-K64/TS':>9s}")
    for label, n, k in SHAPES:
        for m in (128, 512, 2048):
            t_ss64 = bench(m, n, k, "tcgen05", block_m=128, block_k=64,
                           num_stages=4, use_warp_spec=True)
            t_ts = bench(m, n, k, "ts", block_m=128, num_stages=4,
                         use_warp_spec=True)
            r = (f"{t_ss64 / t_ts:>8.2f}x"
                 if isinstance(t_ss64, float) and isinstance(t_ts, float)
                 else "--")
            print(f"{label:<15s} {m:>5d} {fmt(t_ss64):>10s} "
                  f"{fmt(t_ts):>10s} {r:>9s}")
    print()


if __name__ == "__main__":
    main()
