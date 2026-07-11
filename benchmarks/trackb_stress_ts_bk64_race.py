"""Track-f cross-check: is TS mode (track B) exposed to the SS-mode
WS+TMA BlockK=64 race at the exact failing SS geometry?

SS fails nondeterministically at uint4 zp=T (512,512,4096) block
(128,128,64) s3 WS+TMA (max|err| ~350 vs bf16-noise 2.0). Run the TS
kernel at the same geometry (s3 and s4) repeatedly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import torch
from test_tcgen05_ts import _run_ts

for stages in (3, 4):
    for rep in range(5):
        torch.manual_seed(1000 + rep)
        out, ref = _run_ts(512, 512, 4096, block_shape=(128, 128, 64),
                           num_stages=stages, use_warp_spec=True)
        err = (out.float() - ref.float()).abs().max().item()
        print(f"TS WS+TMA (512,512,4096) BK64 s{stages} rep{rep}: "
              f"max|err|={err:.1f}", flush=True)
