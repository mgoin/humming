import itertools

import pytest
import torch
from _ts_packing_ref import (
    pack_weight_tcgen05_ts,
    simulate_ts_thread_regs,
    unpack_weight_mma_sync,
    unpack_weight_tcgen05_ts,
    unpack_zero_point_tcgen05_ts,
)

from humming import dtypes, ops
from humming.config import LayerConfig, MmaType
from humming.schema import HummingWeightSchema
from humming.testing import skip_if_unsupported
from humming.testing.data import generate_random_tensor
from humming.transform import transform_humming_tensors, transform_humming_weight_scale
from humming.utils.ts_packing import pack_zero_point_tcgen05_ts

SHAPES = [(64, 64), (128, 128), (128, 256), (256, 512), (512, 1024), (192, 320)]
WEIGHT_BITS = [2, 4, 8]


def _random_codes(shape_n, shape_k, weight_bits=4, seed=0, device="cpu"):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    codes = torch.randint(0, 1 << weight_bits, (shape_n, shape_k), generator=generator, dtype=torch.int32)
    return codes.to(device)


def _quantized_w4a16_layer(mma_type: MmaType) -> tuple[LayerConfig, dict]:
    config = LayerConfig(
        shape_n=256,
        shape_k=256,
        a_dtype=dtypes.bfloat16,
        b_dtype=dtypes.uint4,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=128,
        has_zero_point=True,
        mma_type=mma_type,
    )
    schema = HummingWeightSchema(
        b_dtype=config.b_dtype,
        bs_dtype=config.bs_dtype,
        weight_scale_group_size=config.weight_scale_group_size,
        weight_scale_type=config.weight_scale_type,
        has_zero_point=True,
    )
    torch.manual_seed(11)
    weight = generate_random_tensor(
        (config.shape_n, config.shape_k),
        dtype=torch.bfloat16,
        group_size=config.weight_scale_group_size,
    )
    return config, schema.quant_tensor(weight, schema, torch.bfloat16)


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
    shape_n, shape_k = 64, 32
    codes = _random_codes(shape_n, shape_k, seed=2)
    packed = pack_weight_tcgen05_ts(codes, 4)
    for k_chunk in range(shape_k // 16):
        for warp in range(2):
            for lane in range(32):
                regs = simulate_ts_thread_regs(packed, warp, lane, k_chunk, block_n=64)
                expected = codes[32 * warp + lane, 16 * k_chunk : 16 * k_chunk + 16]
                assert torch.equal(regs, expected)


def test_scale_stream_keeps_natural_row_order():
    # The thread owning row n reads its scale at index n: no permutation.
    generator = torch.Generator().manual_seed(3)
    weight_scale = torch.randn(128, 8, generator=generator, dtype=torch.float32).to(torch.bfloat16)
    packed = transform_humming_weight_scale(weight_scale, use_tcgen05_ts=True)
    assert packed.shape == (8, 128)
    assert torch.equal(packed.transpose(0, 1), weight_scale)


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


def _ss_scatter_geometry(block_n: int, block_k: int) -> dict:
    # bf16 A: 16 K per tcgen05.mma issue, staged section-major at 64 K per section.
    part_mma_shape_k = 16
    k_per_section = min(block_k, 64)
    row_bytes = k_per_section * 2
    return {
        "part_mma_shape_k": part_mma_shape_k,
        "k_per_section": k_per_section,
        "row_bytes": row_bytes,
        "section_bytes": block_n * row_bytes,
        "k_iters_per_section": k_per_section // part_mma_shape_k,
    }


def _ss_scatter_ref_offset(g, t, n_base, iteration, i, frag, pair, base_div128):
    # Element-wise form: PTX Table 32 (n, k) for the fragment, then the byte
    # address under loader_a's section-major staging and Swizzle<3,4,3>.
    n = n_base + i * 16 + 8 * frag + t // 4
    k_lo = iteration * g["part_mma_shape_k"] + 2 * (t % 4) + 8 * pair
    k_section, k_in_section = divmod(k_lo, g["k_per_section"])
    linear_in_section = n * g["row_bytes"] + k_in_section * 2
    linear = k_section * g["section_bytes"] + linear_in_section
    xor_shift = (base_div128 + (linear_in_section >> 7)) & 7
    return linear ^ (xor_shift << 4)


def _ss_scatter_closed_base0(g, t, n_base, iteration, base_div128):
    n0 = n_base + t // 4
    pre = n0 * g["row_bytes"] + (t % 4) * 4
    mask = ((base_div128 + n0) & 7) << 4
    iters = g["k_iters_per_section"]
    in_section = (pre ^ mask) ^ ((iteration % iters) * g["part_mma_shape_k"] * 2)
    return in_section + (iteration // iters) * g["section_bytes"]


def _ss_scatter_closed_offset(g, t, n_base, iteration, i, frag, pair, base_div128):
    base0 = _ss_scatter_closed_base0(g, t, n_base, iteration, base_div128)
    return (base0 ^ (pair * 16)) + i * 16 * g["row_bytes"] + frag * 8 * g["row_bytes"]


@pytest.mark.parametrize("block_n", [64, 128, 256])
@pytest.mark.parametrize("block_k", [64, 128, 256])
def test_ss_scatter_closed_form_matches_reference(block_n, block_k):
    # The SS mainloop (mma/tcgen05_mma.cuh) addresses its r2s dequant scatter
    # with scatter_closed_base0 plus immediates, on the claim that every
    # additive term occupies a disjoint bit range. Exhaustive over the whole
    # index space and all 8 SMEM base phases, for every shape the kernel's
    # static_asserts admit.
    warp_n = 64
    g = _ss_scatter_geometry(block_n, block_k)
    space = itertools.product(
        range(8),  # SMEM base phase
        range(max(block_n // warp_n, 1)),  # scatter warp
        range(32),  # lane
        range(block_k // g["part_mma_shape_k"]),  # 16-K iter
        range(warp_n // 16),  # m16n8 fragment pair
        range(2),  # fragment half
        range(2),  # k pair
    )
    for base_div128, warp, t, iteration, i, frag, pair in space:
        args = (g, t, warp * warp_n, iteration, i, frag, pair, base_div128)
        ref, closed = _ss_scatter_ref_offset(*args), _ss_scatter_closed_offset(*args)
        assert ref == closed, f"{base_div128=} {warp=} {t=} {iteration=} {i=} {frag=} {pair=}"


@pytest.mark.parametrize("capability", [(8, 0), (9, 0), (10, 0), (12, 0)])
def test_transform_layout_does_not_depend_on_the_packing_host(capability, monkeypatch):
    # A checkpoint packed anywhere must load on Blackwell, so mma_type="tcgen05"
    # has to pick the TS layout from the layer alone.
    skip_if_unsupported()
    config, tensors = _quantized_w4a16_layer(MmaType.TCGEN05)
    assert config.tcgen05_ts_supported
    expected = transform_humming_tensors(config, tensors)

    mma_config, mma_tensors = _quantized_w4a16_layer(MmaType.MMA)
    assert not torch.equal(transform_humming_tensors(mma_config, mma_tensors)["weight"], expected["weight"])

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: capability)
    packed = transform_humming_tensors(config, tensors)
    assert packed.keys() == expected.keys()
    for name, tensor in expected.items():
        assert torch.equal(packed[name].view(torch.uint8), tensor.view(torch.uint8)), name


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
