"""Reference (torch) packer for the tcgen05 TS-mode weight layout.

This module is the executable specification for the packed-weight /
scale / zero-point layouts consumed by the TS-mode (TMEM-A) tcgen05
mainloop. See ``docs/tcgen05_ts_packing.md`` for the derivation and
the interface contract.

Register-layout contract (what the kernel sees after ``dequant()``):

* An MMA-M tile covers ``min(BlockN, 128)`` weight rows (humming "N").
* Within a 128-row tile, warp ``w`` (of the ``MmaM / 32`` warps that
  cover the tile) owns rows ``(w % 4) * 32 + lane``.
* Per 16-K chunk each thread holds its single row's 16 bf16 in 8
  uint32: reg ``r`` = bf16 pair ``(K = 2r`` in lo half, ``K = 2r + 1``
  in hi half), ascending K. TMEM cell ``(lane, col c)`` =
  ``W[row, 2c .. 2c+1]`` -- K-major, the layout TS-mode A requires.
* Scales / zero-points follow the same ``lane = row`` ownership.

Packed-word bit layout (the lop3 ``(i, i+4)`` pre-compensation): the
``uint_to_f16`` dequant extracts, per output reg, the value at bits
``[b*s, b*s+kBits)`` (lo half) and ``[16+b*s, ...)`` (hi half) of the
shifted word. Pre-compensating so reg ``r`` comes out K-ascending
means value-slot ``s`` of a word holds K-element
``e = (s % (V/2)) * 2 + s // (V/2)`` where ``V = 32 / kBits`` values
per word; equivalently element ``e`` lands in slot
``s = (e % 2) * (V/2) + e // 2``.

All functions accept 2-D ``[N, K]``-shaped code tensors or 3-D
``[E, N, K]`` (MoE) and operate on the last two dims.
"""

import torch

__all__ = [
    "pack_weight_tcgen05_ts",
    "unpack_weight_tcgen05_ts",
    "unpack_weight_mma_sync",
    "pack_scales_tcgen05_ts",
    "unpack_scales_tcgen05_ts",
    "pack_zero_point_tcgen05_ts",
    "unpack_zero_point_tcgen05_ts",
    "simulate_ts_thread_regs",
]


def _check_codes(codes: torch.Tensor, weight_bits: int):
    assert codes.dtype == torch.int32
    assert weight_bits in (2, 4, 8), "reference packer covers 32 % bits == 0 widths {2, 4, 8}"
    n, k = codes.shape[-2], codes.shape[-1]
    assert n % 64 == 0, "N must be padded to a multiple of 64"
    assert k % 16 == 0, "K must be a multiple of kPartMmaShapeK = 16"
    assert (codes >= 0).all() and (codes < (1 << weight_bits)).all()
    return n, k


def pack_weight_tcgen05_ts(codes: torch.Tensor, weight_bits: int = 4) -> torch.Tensor:
    """Pack ``[.., N, K]`` integer codes into the tcgen05-TS layout.

    Returns int32 ``[.., K/16, N * 16 * weight_bits / 32]`` (identical
    shape/tiling to the existing mma.sync packed tensor; only the
    permutation inside each 64-row x 16-K block differs).

    Word/bit position of code ``W[n, k]``:
      c   = k // 16          (packed row)
      j   = (k % 16) * weight_bits // 32   (word within the thread pair-slot)
      e   = (k % 16) % (32 // weight_bits) (K-ascending element in word)
      s   = (e % 2) * (V/2) + e // 2, V = 32 // weight_bits (bit slot)
      B   = n // 64          (64-row block)
      l   = n % 32           (lane)
      h   = (n % 64) // 32   (32-row band half within the block)
      W_r = 16 * weight_bits // 32         (words per row per 16-K chunk)
      col = B * 64 * W_r + l * 2 * W_r + h * W_r + j
      bits [s * weight_bits, (s+1) * weight_bits) of out[c, col]
    """
    n, k = _check_codes(codes, weight_bits)
    dev = codes.device
    vpw = 32 // weight_bits  # values per word
    wpr = 16 * weight_bits // 32  # words per row per 16-K chunk
    num_rows = k // 16
    num_cols = n * wpr

    # Build the inverse map: for output (col, slot) -> (n, k_in_chunk).
    col = torch.arange(num_cols, device=dev, dtype=torch.long)
    B = col // (64 * wpr)
    r = col % (64 * wpr)
    lane = r // (2 * wpr)
    q = r % (2 * wpr)
    h = q // wpr
    j = q % wpr
    src_n = B * 64 + h * 32 + lane  # [num_cols]

    s = torch.arange(vpw, device=dev, dtype=torch.long)
    e = (s % (vpw // 2)) * 2 + s // (vpw // 2)  # [vpw]
    k_in = j.unsqueeze(1) * vpw + e.unsqueeze(0)  # [num_cols, vpw]

    # gather: vals[.., c, col, s] = codes[.., src_n[col], c*16 + k_in[col, s]]
    codes_v = codes.reshape(*codes.shape[:-2], n, num_rows, 16)
    # -> [.., num_rows, n, 16]
    codes_v = codes_v.movedim(-2, -3)
    flat_idx = (src_n.unsqueeze(1) * 16 + k_in).reshape(-1)  # [num_cols*vpw]
    gathered = codes_v.reshape(*codes_v.shape[:-2], n * 16)[..., flat_idx].reshape(
        *codes_v.shape[:-2], num_cols, vpw
    )

    shifts = torch.arange(vpw, device=dev, dtype=torch.int32) * weight_bits
    out = (gathered.to(torch.int64) << shifts.to(torch.int64)).sum(-1)
    return (out & 0xFFFFFFFF).to(torch.uint32).view(torch.int32).contiguous()


def unpack_weight_tcgen05_ts(
    packed: torch.Tensor, shape_n: int, shape_k: int, weight_bits: int = 4
) -> torch.Tensor:
    """Inverse of :func:`pack_weight_tcgen05_ts` -> ``[.., N, K]`` codes."""
    assert packed.dtype == torch.int32
    vpw = 32 // weight_bits
    wpr = 16 * weight_bits // 32
    num_rows = shape_k // 16
    num_cols = shape_n * wpr
    assert packed.shape[-2:] == (num_rows, num_cols), (
        f"expected [..,{num_rows},{num_cols}], got {tuple(packed.shape[-2:])}"
    )
    dev = packed.device

    words = packed.view(torch.uint32).to(torch.int64)  # [.., c, col]
    s = torch.arange(vpw, device=dev, dtype=torch.int64)
    vals = (words.unsqueeze(-1) >> (s * weight_bits)) & ((1 << weight_bits) - 1)
    # vals[.., c, col, s] -> codes[.., n, k]
    col = torch.arange(num_cols, device=dev, dtype=torch.long)
    B = col // (64 * wpr)
    r = col % (64 * wpr)
    lane = r // (2 * wpr)
    q = r % (2 * wpr)
    h = q // wpr
    j = q % wpr
    src_n = B * 64 + h * 32 + lane
    e = (s % (vpw // 2)) * 2 + s // (vpw // 2)
    k_in = j.unsqueeze(1) * vpw + e.unsqueeze(0)  # [num_cols, vpw]

    out = torch.empty((*packed.shape[:-2], shape_n, shape_k), dtype=torch.int32, device=dev)
    n_idx = src_n.unsqueeze(1).expand(num_cols, vpw).reshape(-1)
    c_idx = torch.arange(num_rows, device=dev, dtype=torch.long)
    k_idx = c_idx.view(-1, 1) * 16 + k_in.reshape(-1).view(1, -1)  # [num_rows, num_cols*vpw]
    out_flat = out.reshape(*packed.shape[:-2], shape_n * shape_k)
    flat_dst = n_idx.view(1, -1) * shape_k + k_idx  # [num_rows, num_cols*vpw]
    out_flat.scatter_(
        -1,
        flat_dst.reshape(1, -1).expand(*packed.shape[:-2], -1) if packed.dim() > 2 else flat_dst.reshape(-1),
        vals.reshape(*packed.shape[:-2], -1).to(torch.int32),
    )
    return out


def unpack_weight_mma_sync(
    packed: torch.Tensor, shape_n: int, shape_k: int, weight_bits: int = 4
) -> torch.Tensor:
    """Python inverse of the EXISTING mma.sync repack (interleave_mode=3,
    no wgmma mini-block transpose, no int2fp preprocessing -- i.e. the
    u4/bf16 W4A16 production path).

    Word/bit position of code ``W[n, k]`` in the mma.sync layout:
      c    = k // 16;  k_in = k % 16
      B    = n // 64;  n_in = n % 64
      w    = n_in // 16                 (word within the thread's 4)
      tid  = 4 * (n_in % 8) + (k_in % 8) // 2
      e'   = 4 * ((n_in % 16) // 8) + 2 * (k_in // 8) + (k_in % 2)
      s    = (e' % 2) * 4 + e' // 2     (same lop3 pre-compensation)
      col  = 128 * B + 4 * tid + w
    """
    assert weight_bits == 4, "mma.sync reference inverse implemented for u4"
    assert packed.dtype == torch.int32
    num_rows = shape_k // 16
    num_cols = shape_n * 2
    assert packed.shape[-2:] == (num_rows, num_cols)
    dev = packed.device

    n = torch.arange(shape_n, device=dev, dtype=torch.long).view(-1, 1)
    k_in = torch.arange(16, device=dev, dtype=torch.long).view(1, -1)
    B = n // 64
    n_in = n % 64
    w = n_in // 16
    tid = 4 * (n_in % 8) + (k_in % 8) // 2
    e = 4 * ((n_in % 16) // 8) + 2 * (k_in // 8) + (k_in % 2)
    s = (e % 2) * 4 + e // 2
    col = 128 * B + 4 * tid + w  # [shape_n, 16]

    words = packed.view(torch.uint32).to(torch.int64)  # [.., c, col]
    gathered = words[..., col.reshape(-1)].reshape(*packed.shape[:-2], num_rows, shape_n, 16)
    vals = (gathered >> (s.reshape(-1).view(1, 1, -1).reshape(1, shape_n, 16) * 4)) & 0xF
    # [.., c, n, k_in] -> [.., n, c*16 + k_in]
    vals = vals.movedim(-3, -2).reshape(*packed.shape[:-2], shape_n, shape_k)
    return vals.to(torch.int32)


# ---------------------------------------------------------------------------
# Scale / zero-point streams (lane = row ownership)
# ---------------------------------------------------------------------------


def pack_scales_tcgen05_ts(weight_scale: torch.Tensor) -> torch.Tensor:
    """``[.., N, K/gs]`` scales -> ``[.., K/gs, N]``, natural row order.

    Thread (band ``b``, lane ``l``) of a 64-row block reads its single
    row's scale at index ``n = block*64 + b*32 + l`` -- no permutation
    (unlike the mma.sync stream's 8x8 transpose within 64-row blocks).
    """
    return weight_scale.transpose(-1, -2).contiguous()


def unpack_scales_tcgen05_ts(packed: torch.Tensor) -> torch.Tensor:
    return packed.transpose(-1, -2).contiguous()


def pack_zero_point_tcgen05_ts(zero_point: torch.Tensor, weight_bits: int = 4) -> torch.Tensor:
    """``[.., N, K/gs]`` integer zero-points -> nibble/byte-packed
    ``[.., K/gs, N * zp_bits / 32]`` int32, natural row order.

    zp_bits = 4 for weight_bits <= 4 else 8. Word ``w`` of a group row
    holds rows ``[w * V, (w+1) * V)`` with row ``w*V + i`` at bits
    ``[i * zp_bits, (i+1) * zp_bits)`` (V = 32 / zp_bits). Thread with
    row ``n`` reads word ``n // V`` and extracts slot ``n % V``.
    """
    assert zero_point.dtype in (torch.int32, torch.uint8)
    zp_bits = 4 if weight_bits <= 4 else 8
    vpw = 32 // zp_bits
    zp = zero_point.transpose(-1, -2).contiguous().to(torch.int64)
    *lead, g, n = zp.shape
    assert n % vpw == 0
    zp = zp.reshape(*lead, g, n // vpw, vpw)
    shifts = torch.arange(vpw, device=zp.device, dtype=torch.int64) * zp_bits
    out = (zp << shifts).sum(-1)
    return (out & 0xFFFFFFFF).to(torch.uint32).view(torch.int32).contiguous()


def unpack_zero_point_tcgen05_ts(packed: torch.Tensor, shape_n: int, weight_bits: int = 4) -> torch.Tensor:
    zp_bits = 4 if weight_bits <= 4 else 8
    vpw = 32 // zp_bits
    words = packed.view(torch.uint32).to(torch.int64)
    s = torch.arange(vpw, device=packed.device, dtype=torch.int64)
    vals = (words.unsqueeze(-1) >> (s * zp_bits)) & ((1 << zp_bits) - 1)
    vals = vals.reshape(*packed.shape[:-1], shape_n)
    return vals.transpose(-1, -2).contiguous().to(torch.int32)


# ---------------------------------------------------------------------------
# Kernel-side simulation (loader_b half-group gather + lop3 dequant)
# ---------------------------------------------------------------------------


def simulate_ts_thread_regs(
    packed: torch.Tensor,
    n_warp_id: int,
    lane: int,
    k_chunk: int,
    block_n: int = 128,
    n_block_id: int = 0,
    weight_bits: int = 4,
) -> torch.Tensor:
    """Emulate loader_b's WarpN==32 half-group gather + the dequant
    value-slot extraction for one thread and one 16-K chunk.

    Returns the 16 integer codes the thread's dequant regs would hold,
    in reg order: entry ``2r`` = reg r lo half (K = 2r), entry
    ``2r + 1`` = reg r hi half (K = 2r + 1). If the pack honors the
    contract these equal ``W[row, 16*k_chunk : 16*k_chunk + 16]`` for
    ``row = n_block_id * block_n + 32 * n_warp_id + lane``.
    """
    assert packed.dim() == 2
    vpw = 32 // weight_bits
    wpr = 16 * weight_bits // 32
    words_row = packed[k_chunk].view(torch.uint32).to(torch.int64)

    # g2s: kSmemStride words of this K-chunk row, offset by the n-block.
    smem_base_word = n_block_id * block_n * wpr
    # s2r half-group gather (loader_b.cuh): LoadType covers
    # kNumIntsPerThread = kBits/2 words; int-index of the first word:
    #   idx  = 32 * (n_warp_id / 2) + lane        (16-B slot index)
    #   word = idx * 2*wpr + (n_warp_id % 2) * wpr + j,  j in [0, wpr)
    slot = 32 * (n_warp_id // 2) + lane
    first = smem_base_word + slot * 2 * wpr + (n_warp_id % 2) * wpr
    my_words = words_row[first : first + wpr]

    out = torch.empty(16, dtype=torch.int32)
    for j in range(wpr):
        word = int(my_words[j])
        for r_in in range(vpw // 2):  # regs produced from this word
            lo_slot = r_in
            hi_slot = r_in + vpw // 2
            lo = (word >> (lo_slot * weight_bits)) & ((1 << weight_bits) - 1)
            hi = (word >> (hi_slot * weight_bits)) & ((1 << weight_bits) - 1)
            r = j * (vpw // 2) + r_in
            out[2 * r] = lo
            out[2 * r + 1] = hi
    return out
