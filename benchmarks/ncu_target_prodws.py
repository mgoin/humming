"""NCU profile target: prod-WS tcgen05 kernel, Llama70B-down M=2048.

Config matches workbook B.34/B.36 protocol: BlockM=128 BlockN=128
BlockK=128 stages=4 ws=True. Launches the kernel a few times so
`--launch-skip 2 --launch-count 1` lands on a steady-state launch.

Run under ncu:
  CUDA_VISIBLE_DEVICES=5 /usr/local/cuda/bin/ncu --launch-skip 2 \
    --launch-count 1 --kernel-name regex:humming --set full \
    .venv/bin/python benchmarks/ncu_target_prodws.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bench_tcgen05_vs_wmma import build_launcher

import torch

M, N, K = 2048, 8192, 28672

launch = build_launcher(M, N, K, "tcgen05", block_m=128, block_n=128,
                        block_k=128, num_stages=4, use_warp_spec=True)
for _ in range(6):
    launch()
torch.cuda.synchronize()
print("done")
