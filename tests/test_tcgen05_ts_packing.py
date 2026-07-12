"""Tests for the tcgen05 TS-mode weight/scale/zp packing.

Guards the register-layout contract in docs/tcgen05_ts_packing.md:
after loader_b's gather + the lop3 dequant, thread (warp w, lane l)
must hold W[32*(w%4) + l, 16*chunk + 2r .. 2r+1] in reg r, ascending K,
with scales/zero-points owned per lane = row.

Layers (cheapest first):
1. Python pack -> unpack round-trip (pure layout bijection).
2. Loader/dequant simulation vs the contract (word/bit level).
3. Python inverse of the EXISTING mma.sync packer vs the CUDA
   `ops.repack_weight` output (same logical weights in, same code
   values out -- proves the two layouts encode identical weights).
4. CUDA `weight_repack_nk` TS variant bit-exact vs the Python
   reference on GPU, at gs in {64, 128} and the shape matrix.
"""

import pytest
import torch

from humming import ops
from humming.utils.ts_packing import (
    pack_scales_tcgen05_ts,
    pack_weight_tcgen05_ts,
    pack_zero_point_tcgen05_ts,
    simulate_ts_thread_regs,
    unpack_scales_tcgen05_ts,
    unpack_weight_mma_sync,
    unpack_weight_tcgen05_ts,
    unpack_zero_point_tcgen05_ts,
)

SHAPES = [(64, 64), (128, 128), (128, 256), (256, 512), (512, 1024), (192, 320)]


def _rand_codes(n, k, weight_bits=4, seed=0, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, 1 << weight_bits, (n, k), generator=g, dtype=torch.int32).to(device)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("weight_bits", [2, 4, 8])
def test_ts_pack_roundtrip(shape, weight_bits):
    n, k = shape
    codes = _rand_codes(n, k, weight_bits)
    packed = pack_weight_tcgen05_ts(codes, weight_bits)
    assert packed.shape == (k // 16, n * 16 * weight_bits // 32)
    out = unpack_weight_tcgen05_ts(packed, n, k, weight_bits)
    assert torch.equal(out, codes)


def test_ts_pack_roundtrip_moe():
    codes = torch.stack([_rand_codes(128, 128, seed=s) for s in range(3)])
    packed = pack_weight_tcgen05_ts(codes, 4)
    assert packed.shape == (3, 8, 256)
    out = unpack_weight_tcgen05_ts(packed, 128, 128, 4)
    assert torch.equal(out, codes)


@pytest.mark.parametrize("weight_bits", [2, 4, 8])
def test_ts_register_contract_simulation(weight_bits):
    """The load-bearing test: simulate loader_b's WarpN==32 half-group
    gather + the lop3 dequant slot extraction and check every thread's
    regs hold its own row's 16-K chunk in ascending-K reg order."""
    n, k = 128, 64
    codes = _rand_codes(n, k, weight_bits, seed=1)
    packed = pack_weight_tcgen05_ts(codes, weight_bits)
    for k_chunk in range(k // 16):
        for w in range(4):  # 4 warps covering the 128-row tile
            for lane in range(32):
                row = 32 * (w % 4) + lane
                regs = simulate_ts_thread_regs(
                    packed, w, lane, k_chunk, block_n=128, weight_bits=weight_bits
                )
                expect = codes[row, 16 * k_chunk : 16 * k_chunk + 16]
                assert torch.equal(regs, expect), (
                    f"contract violated at warp={w} lane={lane} chunk={k_chunk}"
                )


def test_ts_contract_block_n64():
    """MmaM = 64 tiles (BlockN=64): 2 warps cover the tile, bands 0/1."""
    n, k = 64, 32
    codes = _rand_codes(n, k, seed=2)
    packed = pack_weight_tcgen05_ts(codes, 4)
    for k_chunk in range(2):
        for w in range(2):
            for lane in range(32):
                regs = simulate_ts_thread_regs(packed, w, lane, k_chunk, block_n=64)
                expect = codes[32 * w + lane, 16 * k_chunk : 16 * k_chunk + 16]
                assert torch.equal(regs, expect)


def test_scale_stream_roundtrip():
    g = torch.Generator().manual_seed(3)
    ws = torch.randn(128, 8, generator=g, dtype=torch.float32).to(torch.bfloat16)
    packed = pack_scales_tcgen05_ts(ws)
    assert packed.shape == (8, 128)
    # lane = row ownership: packed[g, n] is row n's scale for group g.
    assert torch.equal(packed[5, 77], ws[77, 5])
    assert torch.equal(unpack_scales_tcgen05_ts(packed), ws)


@pytest.mark.parametrize("weight_bits", [2, 4, 8])
def test_zero_point_stream_roundtrip(weight_bits):
    g = torch.Generator().manual_seed(4)
    zp = torch.randint(0, 1 << weight_bits, (128, 4), generator=g, dtype=torch.int32)
    packed = pack_zero_point_tcgen05_ts(zp, weight_bits)
    zp_bits = 4 if weight_bits <= 4 else 8
    assert packed.shape == (4, 128 * zp_bits // 32)
    # Thread with row n extracts slot n % V of word n // V.
    vpw = 32 // zp_bits
    n = 77
    word = int(packed[2, n // vpw].view(torch.uint32))
    assert (word >> ((n % vpw) * zp_bits)) & ((1 << zp_bits) - 1) == int(zp[n, 2])
    assert torch.equal(unpack_zero_point_tcgen05_ts(packed, 128, weight_bits), zp)


# ---------------------------------------------------------------------------
# GPU: cross-validation against the existing CUDA packers
# ---------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@requires_cuda
@pytest.mark.parametrize("shape", [(64, 64), (128, 128), (256, 512), (128, 28672 // 16)])
def test_mma_sync_inverse_matches_cuda_repack(shape):
    """unpack_weight_mma_sync(ops.repack_weight(W)) == W for the
    production u4/bf16 path (interleave_mode=3, no preprocessing).
    Proves the Python model of the EXISTING layout -- the base the TS
    interleave derivation stands on -- and that both layouts round-trip
    the same logical weights."""
    n, k = shape
    codes = _rand_codes(n, k, seed=5, device="cuda")
    packed = ops.repack_weight(
        inputs=codes,
        weight_bits=4,
        activation_bits=16,
        is_weight_packed=False,
    )
    out = unpack_weight_mma_sync(packed.cpu(), n, k)
    assert torch.equal(out, codes.cpu())
    # Same logical weights land in both layouts:
    ts = pack_weight_tcgen05_ts(codes.cpu(), 4)
    assert torch.equal(unpack_weight_tcgen05_ts(ts, n, k), out)


CUDA_TS_SHAPES = [(64, 64), (128, 128), (128, 256), (256, 512), (512, 1024), (1024, 4096)]


@requires_cuda
@pytest.mark.parametrize("shape", CUDA_TS_SHAPES)
@pytest.mark.parametrize("weight_bits", [2, 4, 8])
def test_cuda_ts_repack_matches_reference(shape, weight_bits):
    n, k = shape
    codes = _rand_codes(n, k, weight_bits, seed=6, device="cuda")
    packed = ops.repack_weight(
        inputs=codes,
        weight_bits=weight_bits,
        activation_bits=16,
        is_weight_packed=False,
        use_tcgen05_ts=True,
    )
    ref = pack_weight_tcgen05_ts(codes.cpu(), weight_bits)
    assert torch.equal(packed.cpu(), ref)


@requires_cuda
@pytest.mark.parametrize("group_size", [64, 128])
def test_cuda_ts_repack_group_sizes(group_size):
    """gs only affects the scale/zp streams (the weight repack itself is
    zp-free for u4/bf16: should_preprocess_for_int2fp is False), but run
    the full prepare path at both group sizes to pin the E2E artifacts."""
    n, k = 256, 512
    codes = _rand_codes(n, k, seed=7, device="cuda")
    packed = ops.repack_weight(
        inputs=codes,
        weight_bits=4,
        activation_bits=16,
        is_weight_packed=False,
        use_tcgen05_ts=True,
    )
    ref = pack_weight_tcgen05_ts(codes.cpu(), 4)
    assert torch.equal(packed.cpu(), ref)

    g = torch.Generator().manual_seed(8)
    ws = torch.randn(n, k // group_size, generator=g).to(torch.bfloat16).cuda()
    zp = torch.randint(0, 16, (n, k // group_size), generator=g, dtype=torch.int32).cuda()
    ws_p = pack_scales_tcgen05_ts(ws)
    zp_p = pack_zero_point_tcgen05_ts(zp, 4)
    assert torch.equal(unpack_scales_tcgen05_ts(ws_p), ws)
    assert torch.equal(unpack_zero_point_tcgen05_ts(zp_p, n, 4), zp)


@requires_cuda
def test_cuda_ts_repack_packed_input():
    """is_weight_packed=True input path (bit-packed rows, as produced by
    ops.pack_weight) must produce the same TS layout."""
    n, k = 128, 256
    codes = _rand_codes(n, k, seed=9, device="cuda")
    packed_in = ops.pack_weight(codes, 4)
    out = ops.repack_weight(
        inputs=packed_in,
        weight_bits=4,
        activation_bits=16,
        is_weight_packed=True,
        use_tcgen05_ts=True,
    )
    ref = pack_weight_tcgen05_ts(codes.cpu(), 4)
    assert torch.equal(out.cpu(), ref)


@requires_cuda
def test_cuda_ts_repack_moe():
    codes = torch.stack([_rand_codes(128, 128, seed=s, device="cuda") for s in range(2)])
    out = ops.repack_weight(
        inputs=codes.contiguous(),
        weight_bits=4,
        activation_bits=16,
        is_weight_packed=False,
        use_tcgen05_ts=True,
    )
    ref = pack_weight_tcgen05_ts(codes.cpu(), 4)
    assert torch.equal(out.cpu(), ref)
