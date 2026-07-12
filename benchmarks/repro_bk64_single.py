"""Single-run BK64 WS+TMA repro for compute-sanitizer sessions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import torch
from test_tcgen05_dtypes import _run_w_a
from humming import dtypes

torch.manual_seed(1234)
out, ref = _run_w_a(
    dtypes.bfloat16, dtypes.uint4, True,
    shape_m=512, shape_n=512, shape_k=4096,
    block_shape=(128, 128, 64), warp_shape=(32, 64, 64),
    num_stages=3, use_warp_spec=True, use_tma=True,
    use_cp_async=False, use_mbarrier=True)
print("max|err| =", (out.float() - ref.float()).abs().max().item())
