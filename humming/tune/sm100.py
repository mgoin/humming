"""sm_100 (Blackwell) tuning heuristics.

Phase A: returns Sm89-style mma.sync configs with the larger SMEM budget.
This is a *correctness*-first stub that unblocks the normal Python entry
point on B100/B200/B300 — `get_heuristics_class()` was previously raising
`KeyError` for any Blackwell cc.

Phase B (separate commit) will extend this to return tcgen05 configs above
a per-shape batch crossover. See workbook.md.
"""

from humming.tune.sm8x import Sm89Heuristics


class Sm100Heuristics(Sm89Heuristics):
    # Blackwell datacenter dies (GB100/GB200/B300) have 228 KiB shared
    # memory per SM available to a CTA; round down for a small safety
    # margin against the driver's reserved bytes (same convention as the
    # Sm90 class).
    max_smem_size: int = 227 * 1024
    sm_version: int = 100
