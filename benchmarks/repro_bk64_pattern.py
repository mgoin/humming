"""Fingerprint the BlockK=64 WS+TMA failure: deterministic? where?

Seeds torch, runs the kernel twice on identical data, reports
bit-exactness across runs and the spatial error pattern vs reference.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import torch
from test_tcgen05_dtypes import _run_w_a
from humming import dtypes

kw = dict(use_warp_spec=True, use_tma=True, use_cp_async=False,
          use_mbarrier=True)

torch.manual_seed(1234)
out1, ref = _run_w_a(dtypes.bfloat16, dtypes.uint4, True,
                     shape_m=512, shape_n=512, shape_k=4096,
                     block_shape=(128, 128, 64), warp_shape=(32, 64, 64),
                     num_stages=3, **kw)
torch.manual_seed(1234)
out2, _ = _run_w_a(dtypes.bfloat16, dtypes.uint4, True,
                   shape_m=512, shape_n=512, shape_k=4096,
                   block_shape=(128, 128, 64), warp_shape=(32, 64, 64),
                   num_stages=3, **kw)

print("bit-exact across seeded runs:", torch.equal(out1, out2))
err = (out1.float() - ref.float()).abs()
bad = err > 8.0
print(f"max|err|={err.max().item():.1f}  bad cells={bad.sum().item()} "
      f"of {err.numel()} ({100.0*bad.sum().item()/err.numel():.2f}%)")
# Block-level heat map: 128x128 output tiles -> 4x4 tiles
M, N = err.shape
bm, bn = 128, 128
print("bad-cell count per (m_tile, n_tile):")
for mt in range(M // bm):
    row = [int(bad[mt*bm:(mt+1)*bm, nt*bn:(nt+1)*bn].sum().item())
           for nt in range(N // bn)]
    print("  ", row)
# Within the worst tile, where?
flat = torch.tensor([[bad[mt*bm:(mt+1)*bm, nt*bn:(nt+1)*bn].sum()
                      for nt in range(N // bn)] for mt in range(M // bm)])
mt, nt = divmod(int(flat.argmax()), N // bn)
tile = bad[mt*bm:(mt+1)*bm, nt*bn:(nt+1)*bn]
print(f"worst tile ({mt},{nt}); bad rows:",
      torch.nonzero(tile.any(dim=1)).flatten().tolist()[:20])
print("bad cols:", torch.nonzero(tile.any(dim=0)).flatten().tolist()[:40])
