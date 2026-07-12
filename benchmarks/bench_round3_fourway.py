"""Round 3 integration four-way bench: mma.sync / SS / TS.

Reuses bench_ts_vs_ss.build_launcher. The SS column reflects whatever
kUseClosedFormScatter is compiled to in tcgen05_mma.cuh: run once with
the shipped constant (=true -> SS-closed-form) and once with it flipped
to false (-> SS-classic element-wise scatter) to fill both SS columns of
the four-way table. TS is the warp-spec TMEM-staging path with the
vectorized transposed drain (tmem_ts_drain.cuh).

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/bench_round3_fourway.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from bench_ts_vs_ss import SHAPES, MS, bench, fmt  # noqa: E402


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
