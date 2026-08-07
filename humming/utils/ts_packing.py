"""Runtime packers for the tcgen05 TS-mode weight layout.

``docs/tcgen05_ts_packing.md`` derives the packed weight / scale /
zero-point layouts the TS-mode (TMEM-A) tcgen05 mainloop consumes and
states the register-layout contract they target; the torch reference
packer and its inverses live with the tests
(``tests/kernels/humming/_ts_packing_ref.py``).

Functions accept 2-D ``[N, K]``-shaped tensors or 3-D ``[E, N, K]``
(MoE) and operate on the last two dims.
"""

import torch

__all__ = ["pack_zero_point_tcgen05_ts"]


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
