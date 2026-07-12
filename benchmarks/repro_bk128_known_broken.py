"""Workbook B.37 PROD_WS_KNOWN_BROKEN probe: the BK128 prod-WS configs
that fail at HEAD (uint4 zp=T/F s4, uint8 zp=T s3). Run with and
without the scatter proxy fence to test the cross-proxy-race theory.
Usage:
  CUDA_VISIBLE_DEVICES=5 .venv/bin/python benchmarks/repro_bk128_known_broken.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import torch
from test_tcgen05_dtypes import _run_w_a
from humming import dtypes

kw = dict(use_warp_spec=True, use_tma=True, use_cp_async=False,
          use_mbarrier=True)

CASES = [
    ("uint4 zp=T K128 s4", dtypes.uint4, True, 4),
    ("uint4 zp=F K128 s4", dtypes.uint4, False, 4),
    ("uint8 zp=T K128 s3", dtypes.uint8, True, 3),
]

for label, b_dtype, zp, stages in CASES:
    errs = []
    for rep in range(3):
        out, ref = _run_w_a(
            dtypes.bfloat16, b_dtype, zp,
            shape_m=512, shape_n=512, shape_k=4096,
            block_shape=(128, 128, 128), warp_shape=(32, 64, 128),
            num_stages=stages, **kw)
        errs.append((out.float() - ref.float()).abs().max().item())
    print(f"{label}: max|err| over 3 runs = "
          + ", ".join(f"{e:.1f}" for e in errs))
