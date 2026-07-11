"""Fast repro for the BlockK=64 prod-WS race exposed by the
closed-form scatter (track f).

Runs uint4 zp=True at (M=512, N=512, K=4096), block (128,128,64) s=3,
sweeping (use_tma, use_warp_spec) and printing max|err| over several
runs each. Usage:
  CUDA_VISIBLE_DEVICES=5 .venv/bin/python benchmarks/repro_bk64_race.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import torch
from test_tcgen05_dtypes import _run_w_a
from humming import dtypes

CONFIGS = [
    ("WS+TMA  ", dict(use_warp_spec=True, use_tma=True,
                      use_cp_async=False, use_mbarrier=True)),
    ("WS only ", dict(use_warp_spec=True, use_tma=False,
                      use_cp_async=True, use_mbarrier=True)),
    ("plain   ", dict(use_warp_spec=False, use_tma=False,
                      use_cp_async=True, use_mbarrier=False)),
]

for label, kw in CONFIGS:
    errs = []
    for rep in range(3):
        out, ref = _run_w_a(
            dtypes.bfloat16, dtypes.uint4, True,
            shape_m=512, shape_n=512, shape_k=4096,
            block_shape=(128, 128, 64), warp_shape=(32, 64, 64),
            num_stages=3, **kw)
        errs.append((out.float() - ref.float()).abs().max().item())
    print(f"{label}: max|err| over 3 runs = "
          + ", ".join(f"{e:.1f}" for e in errs))
