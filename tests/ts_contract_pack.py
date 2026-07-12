"""THROWAWAY reference packer for the tcgen05 TS-mode register-layout
CONTRACT (track b-ts-staging). Test-only; the production packer is track
d-packing's job. Slow reshape/permute implementation, no CUDA.

CONTRACT (jinzhen-umma-notes.md section 4):
  * An MMA-M tile = min(BlockN, 128) weight rows ("N" dim = weight rows).
  * Within a 128-row tile, warp w (of kWarpsPerMmaTile = MmaM/32 warps)
    owns rows (w % 4) * 32 + lane.
  * Per K-iter (16 bf16 of K) each thread holds one full 16-K chunk of
    its single row: 8 x uint32 after dequant.
  * In-register order after dequant: reg r = bf16 pair (K=2r lo half,
    K=2r+1 hi half), ascending K. TMEM cell (lane, col c) = W[row,
    2c..2c+1] -- K-major, the only layout tcgen05 TS-mode A accepts.
  * Scales / zero-points follow lane = row ownership.
  * The lop3 u4->bf16 dequant's (i, i+4) in-word interleave is
    pre-compensated at pack time.

Physical gmem layout (matches humming's g2s loader_b geometry so the
loader stays a dumb byte copy): int32 tensor of shape (K/16, N*2).
Viewed as (K/16 k-iter rows) x (N/32 chunks) x (32 lanes) x (2 words):
  * chunk c covers weight rows 32c .. 32c+31; lane l -> row 32c+l.
  * word 0 = K-local 0..7, word 1 = K-local 8..15 of the k-iter.
  * nibble j of a word holds K-local index INTERLEAVE[j] within its
    8-K half, where INTERLEAVE = [0, 2, 4, 6, 1, 3, 5, 7]. This is the
    pre-compensation for the lop3 dequant, which emits nibble pairs
    (j, j+4) into one uint32: after dequant, reg r = (K=2r, K=2r+1).

In-kernel read (what the TS s2r loader does): thread (warp w, lane l)
of a BlockN=128 tile reads the 8-byte word pair at byte offset
  iter * BlockN * 8 + w * 256 + l * 8
of the stage's smem.b -- 32 consecutive 8B words per warp, fully
coalesced.
"""

from __future__ import annotations

import torch

# lop3 pre-compensation: nibble position j holds K-local INTERLEAVE[j].
INTERLEAVE = [0, 2, 4, 6, 1, 3, 5, 7]


def pack_ts_weight(codes: torch.Tensor) -> torch.Tensor:
    """Pack uint4 codes (N, K) int32 in [0, 16) into the TS-contract
    layout: int32 tensor (K // 16, N * 2)."""
    assert codes.ndim == 2
    n, k = codes.shape
    assert n % 32 == 0 and k % 16 == 0
    assert codes.dtype in (torch.int32, torch.int64, torch.uint8)
    c = codes.to(torch.int64)
    # (nchunk, lane, kiter, word, nibble_slot)
    c = c.reshape(n // 32, 32, k // 16, 2, 8)
    perm = torch.tensor(INTERLEAVE, dtype=torch.int64, device=codes.device)
    c = c[..., perm]  # nibble slot j <- K-local INTERLEAVE[j]
    shifts = torch.arange(8, dtype=torch.int64, device=codes.device) * 4
    words = (c << shifts).sum(dim=-1)  # (nchunk, lane, kiter, word)
    words = words.permute(2, 0, 1, 3).contiguous()  # (kiter, nchunk, lane, word)
    return words.reshape(k // 16, n * 2).to(torch.int32)


def unpack_ts_weight(packed: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Inverse of pack_ts_weight: (K//16, N*2) int32 -> (N, K) int32."""
    assert packed.shape == (k // 16, n * 2)
    words = packed.to(torch.int64).reshape(k // 16, n // 32, 32, 2)
    words = words.permute(1, 2, 0, 3)  # (nchunk, lane, kiter, word)
    shifts = torch.arange(8, dtype=torch.int64, device=packed.device) * 4
    nibbles = (words.unsqueeze(-1) >> shifts) & 0xF  # (..., word, slot)
    inv = torch.empty(8, dtype=torch.int64, device=packed.device)
    inv[torch.tensor(INTERLEAVE, device=packed.device)] = torch.arange(
        8, device=packed.device)
    nibbles = nibbles[..., inv]  # slot -> K-local order
    return nibbles.reshape(n // 32, 32, k // 16, 16) \
        .permute(0, 1, 2, 3).reshape(n // 32, 32, k) \
        .reshape(n, k).to(torch.int32)


def pack_ts_weight_scale(weight_scale: torch.Tensor) -> torch.Tensor:
    """(N, num_groups) -> (num_groups, N), identity N order (lane = row
    ownership; no fragment permute)."""
    return weight_scale.transpose(-1, -2).contiguous()


def pack_ts_zero_point(zero_point: torch.Tensor) -> torch.Tensor:
    """uint4 zp codes (N, num_groups) int32 -> int32 (num_groups, N/8),
    linear nibble order: byte b of a group row = zp[2b] | zp[2b+1] << 4."""
    n = zero_point.size(-2)
    zp = zero_point.transpose(-1, -2).contiguous().to(torch.uint8).view(-1)
    zp = zp[1::2] * 16 + zp[::2]
    return zp.view(torch.int32).view(-1, n * 4 // 32)


# ---------------------------------------------------------------------------
# Kernel-view simulation: replicate the in-kernel smem read + lop3
# dequant register semantics, used to validate the CONTRACT before any
# kernel work.
# ---------------------------------------------------------------------------


def simulate_thread_regs(packed: torch.Tensor, n: int, k: int,
                         block_n: int, n_block: int, k_iter: int,
                         warp: int, lane: int) -> list[int]:
    """Return the 16 uint4 codes thread (warp, lane) holds after the TS
    s2r load + lop3 dequant, in ascending-K contract order (reg r holds
    codes [2r], [2r+1]).

    Models: smem.b stage row = packed[k_iter], tile slice
    [n_block*block_n*... : ...], thread reads 2 words at
    w*256B + l*8B; lop3 emits nibble pairs (j, j+4) per output reg.
    """
    row_words = packed[k_iter].to(torch.int64)  # (N*2,) int32 words
    # tile slice: chunks n_block*(block_n//32) .. + block_n//32
    chunk = n_block * (block_n // 32) + warp
    base = chunk * 64 + lane * 2  # int32 index within row
    w0 = int(row_words[base].item()) & 0xFFFFFFFF
    w1 = int(row_words[base + 1].item()) & 0xFFFFFFFF
    out = []
    for word in (w0, w1):
        for j in range(4):
            lo = (word >> (4 * j)) & 0xF          # nibble j
            hi = (word >> (4 * j + 16)) & 0xF     # nibble j + 4
            out.extend([lo, hi])
    return out
