"""N-rep runner for the B.37 config (uint4 zp=T (512,512,4096)
block (128,128,128) s4 WS+TMA). Prints per-rep max|err| and a verdict.

Usage:
  CUDA_VISIBLE_DEVICES=5 .venv/bin/python benchmarks/repro_b37_loop.py [reps] [label]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

from test_tcgen05_dtypes import _run_w_a
from humming import dtypes

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
LABEL = sys.argv[2] if len(sys.argv) > 2 else "run"

errs = []
for rep in range(REPS):
    out, ref = _run_w_a(
        dtypes.bfloat16, dtypes.uint4, True,
        shape_m=512, shape_n=512, shape_k=4096,
        block_shape=(128, 128, 128), warp_shape=(32, 64, 128),
        num_stages=4,
        use_warp_spec=True, use_tma=True, use_cp_async=False,
        use_mbarrier=True)
    errs.append((out.float() - ref.float()).abs().max().item())

fails = sum(e > 8.0 for e in errs)
marginal = sum(2.0 < e <= 8.0 for e in errs)
print(f"[{LABEL}] {REPS} reps: fails(>8)={fails} marginal(2..8]={marginal} "
      f"errs=" + ",".join(f"{e:.0f}" for e in errs))
