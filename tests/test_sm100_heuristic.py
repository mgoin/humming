"""Test that Sm100Heuristics picks TCGEN05 vs mma.sync at the right
shapes (Phase B.26).
"""
from __future__ import annotations

import pytest
import torch

from humming import dtypes
from humming.config import GemmType
from humming.layer import HummingLayerMeta
from humming.tune.sm100 import Sm100Heuristics, _is_tcgen05_eligible


def _is_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(not _is_blackwell(), reason="sm_100+ only")


def _make_meta(shape_n, shape_k, has_zero_point=True, weight_scale_group_size=128):
    return HummingLayerMeta(
        shape_n=shape_n, shape_k=shape_k,
        a_dtype=dtypes.bfloat16, b_dtype=dtypes.uint4,
        c_dtype=dtypes.bfloat16, bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=weight_scale_group_size,
        has_zero_point=has_zero_point,
    )


@pytest.mark.parametrize("shape_m, want_tcg", [
    # Decode / small batch: never TCGEN05 (BlockM=64 padding overhead
    # dominates).
    (1, False),
    (16, False),
    (64, False),
    # Crossover: M=128 is where the bench shows TCGEN05 wins on fat-N
    # shapes.
    (128, True),
    (256, True),
    (512, True),
    (2048, True),
])
def test_heuristic_fat_n_crossover(shape_m, want_tcg):
    """Llama-8B gate: N=14336, K=4096. N >> K (fat-N). Should pick
    TCGEN05 at M >= 128."""
    meta = _make_meta(14336, 4096)
    eligible = _is_tcgen05_eligible(meta, shape_m, GemmType.DENSE)
    assert eligible == want_tcg


@pytest.mark.parametrize("shape_m, want_tcg", [
    # Fat-K (down projection): bench shows TCGEN05 only wins at
    # M >= 512 for these. At M < 512 the per-K-iter sync overhead
    # dominates.
    (128, False),
    (256, False),
    (512, True),
    (1024, True),
])
def test_heuristic_fat_k_crossover(shape_m, want_tcg):
    """Llama-70B down: N=8192, K=28672. K >> N (fat-K)."""
    meta = _make_meta(8192, 28672)
    eligible = _is_tcgen05_eligible(meta, shape_m, GemmType.DENSE)
    assert eligible == want_tcg


def test_heuristic_returns_tcgen05_config():
    """Verify the config dict has the right TCGEN05 fields set."""
    # shape_k=4096 (% 128 == 0) → bk=128 + stages=3 (bigger MMA per
    # issue, halves the K-iter scatter/sync count; ~5% win at all
    # M >= 128 vs bk=64).
    meta = _make_meta(14336, 4096)
    cfg = Sm100Heuristics.get_config(meta, shape_m=128)
    assert cfg["mma_type"] == "tcgen05"
    assert cfg["use_tcgen05"] is True
    assert cfg["use_warp_spec"] is True
    assert cfg["use_tma"] is True
    assert cfg["use_tma_bzp"] is False  # asserted False by tensor.h
    assert cfg["block_shape"] == (128, 128, 128)
    assert cfg["warp_shape"] == (32, 64, 128)
    assert cfg["num_stages"] == 3

    # Same shape at M=2048 - same config picked.
    cfg = Sm100Heuristics.get_config(meta, shape_m=2048)
    assert cfg["block_shape"] == (128, 128, 128)
    assert cfg["num_stages"] == 3


def test_heuristic_falls_back_to_blockk_64_for_unusual_k():
    """shape_k not divisible by 128 (only by 64) → bk=64 + stages=4."""
    meta = _make_meta(14336, 4160)  # 4160 = 64 * 65 (not div by 128)
    cfg = Sm100Heuristics.get_config(meta, shape_m=512)
    assert cfg["mma_type"] == "tcgen05"
    assert cfg["block_shape"] == (128, 128, 64)
    assert cfg["num_stages"] == 4


def test_heuristic_falls_back_to_mma_for_small_m():
    """Small batch (decode) should get the mma.sync path, which means
    no `mma_type=tcgen05` in the returned config."""
    meta = _make_meta(14336, 4096)
    cfg = Sm100Heuristics.get_config(meta, shape_m=16)
    assert cfg.get("mma_type") != "tcgen05"


def test_heuristic_blockn_fallback_for_unusual_n():
    """N divisible by 64 but not 128 -> heuristic should fall back
    to BlockN=64 in the TCGEN05 path."""
    # N=192 = 64 * 3 (not divisible by 128).
    meta = _make_meta(192, 4096)
    cfg = Sm100Heuristics.get_config(meta, shape_m=512)
    if cfg.get("mma_type") == "tcgen05":
        assert cfg["block_shape"][1] == 64


def test_heuristic_non_w4a16_falls_through():
    """A != bf16 -> not a TCGEN05-eligible shape."""
    meta = HummingLayerMeta(
        shape_n=14336, shape_k=4096,
        a_dtype=dtypes.float8e4m3, b_dtype=dtypes.float8e4m3,
        c_dtype=dtypes.bfloat16, bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=128, has_zero_point=False,
    )
    cfg = Sm100Heuristics.get_config(meta, shape_m=512)
    assert cfg.get("mma_type") != "tcgen05"
