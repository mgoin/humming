"""Cross-track comparison: track d's reference packer vs track
b-ts-staging's throwaway packer (both written independently to the same
register-layout CONTRACT).

Skips unless the b-ts-staging worktree is present. Documents the ONE
known divergence so it is decided consciously, not discovered in a
kernel bring-up:

* scale stream, zero-point stream, nibble interleave, word-pair
  semantics: IDENTICAL.
* weight word placement within a K-chunk row: DIFFERENT.
    track b: col = (n/32)*64  + (n%32)*2 + j   (band-major flat;
             needs a NEW TS s2r gather: 8 B per thread at w*256+l*8)
    track d: col = (n/64)*128 + (n%32)*4 + ((n%64)/32)*2 + j
             (slot-paired; byte-compatible with the EXISTING loader_b
             WarpN==32 half-group gather, zero loader changes)
  This is exactly decision point 2 in docs/tcgen05_ts_packing.md.
"""

import sys
from pathlib import Path

import pytest
import torch

from humming.utils import ts_packing as d

B_TESTS = Path("/home/mgoin/code/vllm/.scratch/humming-v2/wt-b-ts-staging/tests")

pytestmark = pytest.mark.skipif(
    not (B_TESTS / "ts_contract_pack.py").exists(),
    reason="track b-ts-staging packer not available",
)


@pytest.fixture()
def b_pack():
    sys.path.insert(0, str(B_TESTS))
    import ts_contract_pack

    yield ts_contract_pack
    sys.path.remove(str(B_TESTS))


def test_scale_and_zp_streams_identical(b_pack):
    g = torch.Generator().manual_seed(1)
    ws = torch.randn(128, 4, generator=g).to(torch.bfloat16)
    zp = torch.randint(0, 16, (128, 4), generator=g, dtype=torch.int32)
    assert torch.equal(b_pack.pack_ts_weight_scale(ws), d.pack_scales_tcgen05_ts(ws))
    zb = b_pack.pack_ts_zero_point(zp)
    zd = d.pack_zero_point_tcgen05_ts(zp, 4)
    assert torch.equal(zb.reshape(zd.shape), zd)


def test_weight_layouts_encode_same_weights_but_differ(b_pack):
    """Both satisfy the register contract via their own kernel read
    pattern; the packed tensors are NOT interchangeable."""
    g = torch.Generator().manual_seed(2)
    n, k = 128, 64
    codes = torch.randint(0, 16, (n, k), generator=g, dtype=torch.int32)
    pb = b_pack.pack_ts_weight(codes)
    pd = d.pack_weight_tcgen05_ts(codes, 4)
    assert pb.shape == pd.shape
    assert not torch.equal(pb, pd), "layouts unexpectedly converged -- update docs!"
    assert torch.equal(b_pack.unpack_ts_weight(pb, n, k), codes)
    assert torch.equal(d.unpack_weight_tcgen05_ts(pd, n, k, 4), codes)
    # The divergence is pure word placement: word (n, j) maps
    # b: (n//32)*64 + (n%32)*2 + j   vs   d: (n//64)*128 + (n%32)*4
    #    + ((n%64)//32)*2 + j. Re-permuting b's cols by that relation
    # must reproduce d's tensor exactly (nibble contents identical).
    nn = torch.arange(n)
    perm = torch.empty(2 * n, dtype=torch.long)
    for j in range(2):
        col_b = (nn // 32) * 64 + (nn % 32) * 2 + j
        col_d = (nn // 64) * 128 + (nn % 32) * 4 + ((nn % 64) // 32) * 2 + j
        perm[col_d] = col_b
    assert torch.equal(pb[:, perm], pd)


def test_b_layout_fails_d_loader_contract(b_pack):
    """Feeding track b's tensor to the existing loader_b half-group
    gather violates the contract (motivates the decision point)."""
    g = torch.Generator().manual_seed(3)
    codes = torch.randint(0, 16, (128, 16), generator=g, dtype=torch.int32)
    pb = b_pack.pack_ts_weight(codes)
    wrong = 0
    for w in range(4):
        for lane in range(32):
            regs = d.simulate_ts_thread_regs(pb, w, lane, 0, block_n=128)
            if not torch.equal(regs, codes[32 * (w % 4) + lane, :16]):
                wrong += 1
    assert wrong == 124, f"expected 124/128 mismatched threads, got {wrong}"
