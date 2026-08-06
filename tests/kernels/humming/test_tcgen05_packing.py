"""Contract tests for the tcgen05 TS-mode weight/scale/zero-point layouts.

The register-layout contract (docs/tcgen05_ts_packing.md) is that after
loader_b's half-group gather and the lop3 dequant, thread (warp w, lane l)
holds W[32 * (w % 4) + l, 16 * chunk .. + 16] in ascending-K register order,
with scales and zero points owned per lane = weight row.
"""

import pytest
import torch

from humming import ops
from humming.testing import skip_if_unsupported
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
WEIGHT_BITS = [2, 4, 8]


def _random_codes(shape_n, shape_k, weight_bits=4, seed=0, device="cpu"):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    codes = torch.randint(0, 1 << weight_bits, (shape_n, shape_k), generator=generator, dtype=torch.int32)
    return codes.to(device)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("weight_bits", WEIGHT_BITS)
def test_weight_pack_roundtrip(shape, weight_bits):
    shape_n, shape_k = shape
    codes = _random_codes(shape_n, shape_k, weight_bits)
    packed = pack_weight_tcgen05_ts(codes, weight_bits)
    assert packed.shape == (shape_k // 16, shape_n * 16 * weight_bits // 32)
    assert torch.equal(unpack_weight_tcgen05_ts(packed, shape_n, shape_k, weight_bits), codes)


def test_weight_pack_roundtrip_moe():
    codes = torch.stack([_random_codes(128, 128, seed=seed) for seed in range(3)])
    packed = pack_weight_tcgen05_ts(codes, 4)
    assert packed.shape == (3, 8, 256)
    assert torch.equal(unpack_weight_tcgen05_ts(packed, 128, 128, 4), codes)


@pytest.mark.parametrize("weight_bits", WEIGHT_BITS)
def test_register_contract(weight_bits):
    """Simulate the WarpN=32 half-group gather plus the lop3 slot extraction."""
    shape_n, shape_k = 128, 64
    codes = _random_codes(shape_n, shape_k, weight_bits, seed=1)
    packed = pack_weight_tcgen05_ts(codes, weight_bits)
    for k_chunk in range(shape_k // 16):
        for warp in range(4):
            for lane in range(32):
                regs = simulate_ts_thread_regs(
                    packed,
                    warp,
                    lane,
                    k_chunk,
                    block_n=128,
                    weight_bits=weight_bits,
                )
                expected = codes[32 * warp + lane, 16 * k_chunk : 16 * k_chunk + 16]
                assert torch.equal(regs, expected), f"warp={warp} lane={lane} chunk={k_chunk}"


def test_register_contract_block_n64():
    """A 64-row MMA-M tile is covered by two warps, bands 0 and 1."""
    shape_n, shape_k = 64, 32
    codes = _random_codes(shape_n, shape_k, seed=2)
    packed = pack_weight_tcgen05_ts(codes, 4)
    for k_chunk in range(shape_k // 16):
        for warp in range(2):
            for lane in range(32):
                regs = simulate_ts_thread_regs(packed, warp, lane, k_chunk, block_n=64)
                expected = codes[32 * warp + lane, 16 * k_chunk : 16 * k_chunk + 16]
                assert torch.equal(regs, expected)


def test_scale_stream_roundtrip():
    generator = torch.Generator().manual_seed(3)
    weight_scale = torch.randn(128, 8, generator=generator, dtype=torch.float32).to(torch.bfloat16)
    packed = pack_scales_tcgen05_ts(weight_scale)
    assert packed.shape == (8, 128)
    assert torch.equal(packed[5, 77], weight_scale[77, 5])
    assert torch.equal(unpack_scales_tcgen05_ts(packed), weight_scale)


@pytest.mark.parametrize("weight_bits", WEIGHT_BITS)
def test_zero_point_stream_roundtrip(weight_bits):
    generator = torch.Generator().manual_seed(4)
    shape_n, num_groups = 128, 4
    zero_point = torch.randint(
        0,
        1 << weight_bits,
        (shape_n, num_groups),
        generator=generator,
        dtype=torch.int32,
    )
    packed = pack_zero_point_tcgen05_ts(zero_point, weight_bits)
    zp_bits = 4 if weight_bits <= 4 else 8
    assert packed.shape == (num_groups, shape_n * zp_bits // 32)

    # A thread owning row n extracts slot n % values_per_word of word n // vpw.
    values_per_word = 32 // zp_bits
    row = 77
    word = int(packed[2, row // values_per_word].view(torch.uint32))
    slot = (word >> ((row % values_per_word) * zp_bits)) & ((1 << zp_bits) - 1)
    assert slot == int(zero_point[row, 2])
    assert torch.equal(unpack_zero_point_tcgen05_ts(packed, shape_n, weight_bits), zero_point)


@pytest.mark.parametrize("shape", [(64, 64), (128, 128), (256, 512), (128, 1792)])
def test_mma_sync_inverse_matches_cuda_repack(shape):
    """Both layouts must round-trip the same logical weights.

    The mma.sync inverse is the base the TS interleave derivation stands on,
    so pinning it keeps the TS reference packer honest.
    """
    skip_if_unsupported()
    shape_n, shape_k = shape
    codes = _random_codes(shape_n, shape_k, seed=5, device="cuda")
    packed = ops.repack_weight(
        inputs=codes,
        weight_bits=4,
        activation_bits=16,
        is_weight_packed=False,
    )
    unpacked = unpack_weight_mma_sync(packed.cpu(), shape_n, shape_k)
    assert torch.equal(unpacked, codes.cpu())
    packed_ts = pack_weight_tcgen05_ts(codes.cpu(), 4)
    assert torch.equal(unpack_weight_tcgen05_ts(packed_ts, shape_n, shape_k), unpacked)


@pytest.mark.parametrize("shape", SHAPES + [(1024, 4096)])
@pytest.mark.parametrize("weight_bits", WEIGHT_BITS)
def test_cuda_repack_matches_reference(shape, weight_bits):
    skip_if_unsupported()
    shape_n, shape_k = shape
    codes = _random_codes(shape_n, shape_k, weight_bits, seed=6, device="cuda")
    packed = ops.repack_weight(
        inputs=codes,
        weight_bits=weight_bits,
        activation_bits=16,
        is_weight_packed=False,
        use_tcgen05_ts=True,
    )
    assert torch.equal(packed.cpu(), pack_weight_tcgen05_ts(codes.cpu(), weight_bits))


def test_cuda_repack_packed_input():
    """Bit-packed rows (as produced by ops.pack_weight) reach the same layout."""
    skip_if_unsupported()
    shape_n, shape_k = 128, 256
    codes = _random_codes(shape_n, shape_k, seed=9, device="cuda")
    packed = ops.repack_weight(
        inputs=ops.pack_weight(codes, 4),
        weight_bits=4,
        activation_bits=16,
        is_weight_packed=True,
        use_tcgen05_ts=True,
    )
    assert torch.equal(packed.cpu(), pack_weight_tcgen05_ts(codes.cpu(), 4))


def test_cuda_repack_moe():
    skip_if_unsupported()
    codes = torch.stack([_random_codes(128, 128, seed=seed, device="cuda") for seed in range(2)])
    packed = ops.repack_weight(
        inputs=codes.contiguous(),
        weight_bits=4,
        activation_bits=16,
        is_weight_packed=False,
        use_tcgen05_ts=True,
    )
    assert torch.equal(packed.cpu(), pack_weight_tcgen05_ts(codes.cpu(), 4))
