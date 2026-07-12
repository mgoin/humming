"""Value forensics for the B.37 corruption: what do the bad cells hold?

smem.reduce aliases stages[0].a in the SMEM union. If the TMA-C engine
reads stale (pre-drain) SMEM, corrupted C cells hold bf16 A-activation
bytes (|v| ~ few) instead of C-magnitude values (|v| ~ sqrt(K)). If the
corruption were an epilogue math bug, |out_bad| would be C-scale.

Usage:
  CUDA_VISIBLE_DEVICES=5 .venv/bin/python benchmarks/probe_b37_forensics.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import torch
from test_tcgen05_dtypes import _run_w_a
from humming import dtypes

for rep in range(6):
    out, ref = _run_w_a(
        dtypes.bfloat16, dtypes.uint4, True,
        shape_m=512, shape_n=512, shape_k=4096,
        block_shape=(128, 128, 128), warp_shape=(32, 64, 128),
        num_stages=4,
        use_warp_spec=True, use_tma=True, use_cp_async=False,
        use_mbarrier=True)
    err = (out.float() - ref.float()).abs()
    bad = (err > 8.0).nonzero()
    if not bad.numel():
        print(f"rep {rep}: clean")
        continue
    m_tiles = sorted(set((bad[:, 0] // 128).tolist()))
    n_tiles = sorted(set((bad[:, 1] // 128).tolist()))
    cols = sorted(set((bad[:, 1] % 128).tolist()))
    rows = sorted(set((bad[:, 0] % 128).tolist()))
    out_bad = out.float()[bad[:, 0], bad[:, 1]]
    ref_bad = ref.float()[bad[:, 0], bad[:, 1]]
    print(f"rep {rep}: bad={bad.shape[0]} m_tiles={m_tiles} n_tiles={n_tiles}")
    print(f"  cols%128 range=[{cols[0]}..{cols[-1]}] unique={len(cols)}; "
          f"rows%128 range=[{rows[0]}..{rows[-1]}] unique={len(rows)}")
    print(f"  |out_bad|: mean={out_bad.abs().mean():.2f} "
          f"max={out_bad.abs().max():.2f}  vs |ref_bad|: "
          f"mean={ref_bad.abs().mean():.2f} max={ref_bad.abs().max():.2f}")
    print(f"  |ref| overall: mean={ref.float().abs().mean():.2f} "
          f"max={ref.float().abs().max():.2f}")
    samples = [(int(bad[i,0]), int(bad[i,1]), float(out_bad[i]), float(ref_bad[i]))
               for i in range(0, min(8, bad.shape[0]))]
    print(f"  samples (m,n,out,ref): {samples}")
