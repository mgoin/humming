"""Round-trip + contract validation for the throwaway TS-mode reference
packer (tests/ts_contract_pack.py). Pure CPU/GPU tensor math -- no
kernel launches. Guards the register-layout CONTRACT that the TS-mode
kernel and track d-packing's production packer both build against."""

from __future__ import annotations

import pytest
import torch

from ts_contract_pack import (
    pack_ts_weight,
    pack_ts_weight_scale,
    pack_ts_zero_point,
    simulate_thread_regs,
    unpack_ts_weight,
)


@pytest.fixture
def rand_codes():
    torch.manual_seed(7)
    n, k = 256, 512
    return torch.randint(0, 16, (n, k), dtype=torch.int32), n, k


def test_pack_unpack_round_trip(rand_codes):
    codes, n, k = rand_codes
    packed = pack_ts_weight(codes)
    assert packed.dtype == torch.int32
    assert packed.shape == (k // 16, n * 2)
    out = unpack_ts_weight(packed, n, k)
    assert torch.equal(out, codes)


def test_shape_matches_humming_pack(rand_codes):
    """The TS pack must be byte-compatible (shape/dtype) with
    prepare_humming_weight output so the launcher shape checks and the
    dumb g2s tile copies work unchanged."""
    if not torch.cuda.is_available():
        pytest.skip("prepare_humming_weight needs CUDA")
    from humming import dtypes
    from humming.utils.weight import prepare_humming_weight

    codes, n, k = rand_codes
    ref = prepare_humming_weight(
        codes.cuda(), dtypes.uint4, dtypes.bfloat16, use_wgmma=False)
    packed = pack_ts_weight(codes)
    assert packed.shape == ref.shape, (packed.shape, ref.shape)
    assert packed.dtype == ref.dtype


def test_contract_thread_ownership(rand_codes):
    """Thread (warp w, lane l) of tile n_block must hold, after the
    simulated s2r load + lop3 dequant, exactly
    W[n_block*BlockN + 32w + l, 16*k_iter + 0..15] in ascending K."""
    codes, n, k = rand_codes
    packed = pack_ts_weight(codes)
    block_n = 128
    for n_block in (0, 1):
        for k_iter in (0, 3, k // 16 - 1):
            for warp in range(block_n // 32):
                for lane in (0, 1, 17, 31):
                    row = n_block * block_n + warp * 32 + lane
                    got = simulate_thread_regs(
                        packed, n, k, block_n, n_block, k_iter, warp, lane)
                    want = codes[row, k_iter * 16:(k_iter + 1) * 16].tolist()
                    assert got == want, (
                        f"n_block={n_block} k_iter={k_iter} w={warp} "
                        f"l={lane}: {got} != {want}")


def test_scale_zp_lane_ownership():
    torch.manual_seed(11)
    n, groups = 128, 4
    scale = torch.randn(n, groups, dtype=torch.bfloat16)
    zp = torch.randint(0, 16, (n, groups), dtype=torch.int32)

    s = pack_ts_weight_scale(scale)
    assert s.shape == (groups, n)
    # lane = row ownership: scale for row r of group g is at [g, r]
    assert torch.equal(s[2, 37], scale[37, 2])

    z = pack_ts_zero_point(zp)
    assert z.shape == (groups, n * 4 // 32)
    zb = z.view(torch.uint8).view(groups, n // 2)
    for g in (0, 3):
        for r in (0, 1, 76, 127):
            nib = (int(zb[g, r // 2].item()) >> (4 * (r % 2))) & 0xF
            assert nib == int(zp[r, g].item())
