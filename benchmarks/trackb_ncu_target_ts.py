"""NCU profile target: TS-mode (track B) kernel, Llama70B-down M=2048.

Config = track B's winning M128/BN128/BK64 s4 WS. Same protocol as
track f's ncu_target_prodws.py (skip 2, count 1).

Run:
  CUDA_VISIBLE_DEVICES=1 /usr/local/cuda/bin/ncu --launch-skip 2 \
    --launch-count 1 --kernel-name regex:humming --set full \
    -o /tmp/trackf_ts_m2048 .venv/bin/python benchmarks/ncu_target_ts.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bench_ts_vs_ss import build_launcher

import torch

M, N, K = 2048, 8192, 28672

launch = build_launcher(M, N, K, "ts", block_m=128, num_stages=4,
                        use_warp_spec=True)
for _ in range(6):
    launch()
torch.cuda.synchronize()
print("done")
