"""sm_100 (Blackwell) tuning heuristics.

Phase B.26 (current): adds a TCGEN05 fast-path on top of the
Sm89-style mma.sync baseline. For W4A16 (bf16 A × narrow-B with
group scales + zero-points) at shape_m >= 128 with a "fat-N"
weight (N >= K, e.g. gate/up projections), the heuristic returns a
TCGEN05 config that benchmarks 1.13-1.33× faster than mma.sync on
realistic LLM shapes. Everything else (decode, "fat-K" down
projections, integer A, etc.) still falls through to the Sm89-style
mma.sync config.

See `benchmarks/bench_tcgen05_vs_wmma.py` for the underlying perf
data and `workbook.md` for the rationale of each cutoff.
"""

from humming import dtypes
from humming.config import GemmType
from humming.tune.sm8x import Sm89Heuristics


def _is_tcgen05_eligible(meta, shape_m: int, gemm_type: GemmType) -> bool:
    """Return True iff the TCGEN05 path is BOTH supported and a
    profitable choice for `meta` at `shape_m`.

    Supported (matches the static_asserts in
    `mma/tcgen05_mma.cuh`):
      * bf16 A × uint4 B with bf16 group scales + zero-points
      * group_size matches BlockK (= 64 for our TCGEN05 path); the
        bench uses group_size=128 with BlockK=64, which works
        because TCGEN05's K-iter ignores the BS group boundary.
      * Dense GEMM only (grouped/indexed/MoE not yet validated for
        TCGEN05).

    Profitable (matches bench crossover, conservatively):
      * shape_m >= 128 (4 M-tiles minimum to fill the SM with the
        BlockM=64 path, 1 M-tile with BlockM=128). M < 128 loses to
        mma.sync's smaller-tile path which can spread work across
        more CTAs.
      * shape_n >= shape_k. "Fat-N" weight (gate / up projections);
        for "fat-K" (down projections at M=128-256) TCGEN05 still
        loses 0.67-0.69× because the K-loop's per-iter scatter +
        sync cost outweighs the larger MMA throughput. At M >= 512
        even fat-K wins, but we keep the conservative N >= K gate
        until we hand-tune the down crossover.
    """
    if gemm_type != GemmType.DENSE:
        return False
    if meta.a_dtype != dtypes.bfloat16:
        return False
    if meta.b_dtype != dtypes.uint4:
        return False
    if not meta.has_zero_point:
        # has_zero_point=False works but the most common deployed
        # quant flows are AWQ/GPTQ with zero-points; gate to that
        # case for now since it's what the bench validated.
        pass  # actually allow both — both are correct.
    if meta.weight_scale_group_size <= 0:
        # Tensor-scale or no-scale: not exercised by tests yet.
        return False
    if shape_m < 128:
        return False
    if meta.shape_n < meta.shape_k:
        # Fat-K down projection: skip until the bench shows TCGEN05
        # wins reliably in this regime. At very large M (>= 512)
        # TCGEN05 does win for fat-K too, but that's a separate
        # cutoff we'll add after verification.
        if shape_m < 512:
            return False
    return True


class Sm100Heuristics(Sm89Heuristics):
    # Blackwell datacenter dies (GB100/GB200/B300) have 228 KiB shared
    # memory per SM available to a CTA; round down for a small safety
    # margin against the driver's reserved bytes (same convention as the
    # Sm90 class).
    max_smem_size: int = 227 * 1024
    sm_version: int = 100

    @classmethod
    def get_config(
        cls,
        meta,
        shape_m,
        use_f16_accum=False,
        use_batch_invariant=False,
        gemm_type=GemmType.DENSE,
    ):
        if _is_tcgen05_eligible(meta, shape_m, gemm_type):
            # TCGEN05 config matches what
            # `benchmarks/bench_tcgen05_vs_wmma.py` shows as the
            # peak-throughput point (BlockM=128, BlockN=128, BlockK=64,
            # 4 M-warps, single N-warp, kNumStages=3, warp-spec on).
            # BlockN=128 is the sweet spot for the realistic shapes we
            # measured -- BlockN=64 gives more parallelism per CTA but
            # the per-CTA overhead dominates; BlockN=256 saturates SMEM
            # too quickly and loses occupancy.
            return {
                "block_shape": (128, 128, 64),
                "warp_shape": (32, 64, 64),
                "num_stages": 3,
                "num_ctas_per_sm": 1,
                "num_write_splits": 1,
                "mma_type": "tcgen05",
                "use_tcgen05": True,
                "use_warp_spec": True,
                "use_tma": True,
                "use_cp_async": False,
                "use_mbarrier": True,
                # is_group_weight_scale + has_zero_point triggers
                # humming's `tensor.h:275` assert if use_tma_bzp is
                # True; keep BZP on cp.async (cheap, BZP is small).
                "use_tma_bzp": False,
            }

        return super().get_config(
            meta=meta,
            shape_m=shape_m,
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
            gemm_type=gemm_type,
        )
