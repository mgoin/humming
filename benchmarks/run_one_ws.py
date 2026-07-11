"""Launch one config a few times for NCU profiling.

Usage: python benchmarks/run_one_ws.py [--classic]
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_tcgen05_vs_wmma import build_launcher  # noqa: E402

ws_pipeline = "--classic" not in sys.argv
launch = build_launcher(
    2048, 8192, 28672, "tcgen05", block_m=128, block_k=128,
    num_stages=4, use_warp_spec=True, use_ws_pipeline=ws_pipeline)
for _ in range(4):
    launch()
import torch  # noqa: E402
torch.cuda.synchronize()
print("done ws_pipeline=", ws_pipeline)
