"""sm_100 (Blackwell) tuning heuristics.

Adds a TCGEN05 fast-path on top of the Sm89-style mma.sync baseline.
For W4A16 (bf16 A x narrow-B with group scales + zero-points) at
shape_m >= 128 with a "fat-N" weight (N >= K, e.g. gate/up
projections), the heuristic returns a TCGEN05 config that benchmarks
1.20-1.55x faster than mma.sync on realistic LLM shapes. Everything
else (decode, "fat-K" down projections at small M, integer A, etc.)
falls through to the Sm89-style mma.sync config.

See `benchmarks/bench_tcgen05_vs_wmma.py` for the underlying perf
data and `workbook.md` for the rationale of each cutoff.
"""

from humming import dtypes
from humming.config import GemmType, MmaType
from humming.tune.sm8x import Sm89Heuristics

# B-dtypes opted in for the TCGEN05 path. Each entry maps to a
# (block_k, num_stages) config in `_tcgen05_config_for_b_dtype`. The
# set is conservative: only dtypes whose WS+TMA configs were verified
# correct vs the no-WS reference at production shapes (workbook B.37,
# `benchmarks/bench_tcgen05_dtypes.py`).
_TCGEN05_OPTED_IN_B_DTYPES = frozenset({
    dtypes.uint3,
    dtypes.uint4,
    dtypes.uint5,
    dtypes.uint6,
    dtypes.uint8,
    dtypes.float4e2m1,
    dtypes.float8e4m3,
    dtypes.float8e5m2,
})


def _tcgen05_config_for_b_dtype(b_dtype, shape_k_aligned_128):
    """Return (block_k, num_stages) for the TCGEN05 path given the
    B dtype and whether shape_k is a multiple of 128.

    Wider B-dtypes (uint{5..8}, fp8) blow the 232 KiB SMEM cap at
    BlockK=128 stages=4 once they're combined with the bf16
    b_dequant staging buffer, so they're tuned with BlockK=128
    stages=3 (uint5/6, fp8) or BlockK=64 stages=3 (uint8) which
    matches the bench's safe-and-fastest picks.

    uint4 keeps BlockK=128 stages=4 (the existing tuned config from
    workbook B.35) when shape_k allows it.

    All configs are verified correct at production-realistic shapes
    by `tests/test_tcgen05_dtypes.py::test_tcgen05_bf16_x_b_prod_ws`.
    """
    if not shape_k_aligned_128:
        # BlockK=128 needs shape_k divisible by 128. Fall back to
        # BlockK=64 stages=4 for all dtypes.
        return 64, 4
    # BlockK=128 path: pick stages per SMEM budget for this dtype.
    if b_dtype in (dtypes.uint3, dtypes.uint4, dtypes.float4e2m1):
        return 128, 4
    if b_dtype in (dtypes.uint5, dtypes.uint6,
                   dtypes.float8e4m3, dtypes.float8e5m2):
        return 128, 3
    if b_dtype == dtypes.uint8:
        # uint8's raw codes are 2x wider than uint4; even at stages=3
        # BlockK=128 trips the SMEM cap. Use BlockK=64 stages=4.
        return 64, 4
    # Fallback for any future opt-in: BlockK=64 stages=4.
    return 64, 4


# TS-mode weight-dtype allowlist. Mirrors the ts_dequant_b_pair dispatch +
# static_assert allowlist in mma/tcgen05_ts_mma.cuh; extended one dtype per
# weight-dtype milestone. Start: {uint4}.
_TS_OPTED_IN_B_DTYPES = frozenset({dtypes.uint2, dtypes.uint4})


def supports_tcgen05_ts(meta) -> bool:
    """M-independent legality of the TS-mode tcgen05 kernel for `meta`
    (mirrors the static_asserts in `mma/tcgen05_ts_mma.cuh`). TS-mode
    needs the slot-paired TS weight/scale/zp packing
    (docs/tcgen05_ts_packing.md), which is NOT interchangeable with the
    layout mma.sync / SS-tcgen05 read -- so a TS layer runs TS at
    every M, and the opt-in lives on the meta (mma_type == TCGEN05),
    not on a per-M crossover.
    """
    if meta.num_experts:
        return False
    if meta.a_dtype != dtypes.bfloat16 or meta.b_dtype not in _TS_OPTED_IN_B_DTYPES:
        return False
    if meta.bs_dtype != dtypes.bfloat16:
        return False
    if meta.has_zero_point and meta.is_fp_zero_point:
        return False
    # One scale group per BlockK=64 stage (kernel asserts gs >= BlockK).
    if meta.weight_scale_group_size < 64:
        return False
    # BlockN == 128 (exactly one MMA-M tile), BlockK == 64.
    if meta.shape_n % 128 != 0 or meta.shape_k % 64 != 0:
        return False
    return True


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
    # B-dtypes opted in for the TCGEN05 fast-path. Each has a
    # corresponding (block_k, num_stages) entry in
    # `_tcgen05_config_for_b_dtype` below, picked from
    # `benchmarks/bench_tcgen05_dtypes.py` (which sweeps a config
    # ladder per dtype and reports the fastest WS+TMA config that
    # matches the no-WS reference within bf16 noise).
    if meta.b_dtype not in _TCGEN05_OPTED_IN_B_DTYPES:
        return False
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
    # Tile-count check: TCGEN05's BlockM=128 + BlockN=128 means each
    # CTA covers 128*128 output cells. For shapes with few output
    # tiles (e.g. Llama-8B qkv at M=128 has just 1 M-tile × 48
    # N-tiles = 48 CTAs, under 1/3 of B300's 144 SMs), mma.sync's
    # smaller per-CTA tile size spreads the work across more SMs and
    # wins by ~30%. Require at least 64 output tiles per slice
    # before opting into TCGEN05.
    n_tiles = (meta.shape_n + 127) // 128
    m_tiles = (shape_m + 127) // 128
    if n_tiles * m_tiles < 64:
        return False
    return True


class Sm100Heuristics(Sm89Heuristics):
    # Blackwell datacenter dies (GB100/GB200/B300) have 228 KiB shared
    # memory per SM available to a CTA; round down for a small safety
    # margin against the driver's reserved bytes (same convention as the
    # Sm90 class).
    max_smem_size: int = 227 * 1024
    sm_version: int = 100
    b8_allowed_dtypes: list[dtypes.DataType] = [
        dtypes.int8,
        dtypes.float8e4m3,
        dtypes.float8e5m2,
    ]

    @classmethod
    def supports_tcgen05_ts(cls, meta) -> bool:
        return supports_tcgen05_ts(meta)

    @classmethod
    def get_config(
        cls,
        meta,
        shape_m,
        use_f16_accum=False,
        use_batch_invariant=False,
        gemm_type=GemmType.DENSE,
    ):
        # TS-mode tcgen05 ("method 2"): explicit opt-in via
        # meta.mma_type == TCGEN05 because it changes the packed
        # weight/scale/zp layouts for the whole layer (see
        # transform_humming_layer). Where legal it is used at EVERY M
        # (no mma.sync fallback can read TS packing). Measured on B300
        # GPU0 vs the per-M mma.sync/SS mix of the default path:
        # ties mma.sync at M <= 64 (0.93-0.96x), loses only at fat-K
        # M=128 (0.77x), and wins 1.5x+ at M >= 128 fat-N / M >= 256
        # fat-K; beats closed-form SS BK128 1.01-1.22x everywhere.
        # Default metas keep the SS/mma.sync behavior below -- TS as
        # the sm100 default stays gated on the open producer-race
        # root cause (synthesis round-2 SS2b).
        if (
            gemm_type == GemmType.DENSE
            and meta.mma_type == MmaType.TCGEN05
            and supports_tcgen05_ts(meta)
        ):
            block_m = 128 if shape_m >= 128 else 64
            return {
                "block_shape": (block_m, 128, 64),
                "warp_shape": (block_m, 32, 64),
                "num_stages": 4,
                "num_ctas_per_sm": 1,
                "num_write_splits": 1,
                "mma_type": "tcgen05",
                "use_tcgen05": True,
                "use_tcgen05_ts": True,
                "use_warp_spec": True,
                "use_tma": True,
                "use_cp_async": False,
                "use_mbarrier": True,
                "use_tma_bzp": False,
                # Stream-K's cross-CTA partial-K reduction is NOT
                # supported by the TS epilogue (TMA-C store path): it
                # corrupts outputs whenever K is split across CTAs
                # (mean|err| ~0.1 at M=256 N=1024 K=2048 vs ~4e-5 with
                # it off). HummingKernel defaults use_stream_k=True, so
                # it MUST be pinned False here -- the whole TS
                # validation suite (tests/test_tcgen05_ts.py,
                # test_tcgen05_ts_e2e.py) runs with it off.
                "use_stream_k": False,
                "raster_group_m": 1,
            }

        if _is_tcgen05_eligible(meta, shape_m, gemm_type):
            # TCGEN05 config selected per benchmarks/bench_blocksize +
            # bench_blockk_fatk:
            #
            #   BlockN=128: sweet spot. (BlockN=64 underuses each CTA;
            #     BlockN=256 saturates SMEM and loses occupancy.)
            #   BlockK=128 + stages=3 is BEST whenever shape_k % 128
            #     == 0 -- the bigger MMA per issue halves the K-iter
            #     count and wins 3-8% over BlockK=64 across ALL
            #     measured shapes / M values (including M=128 with a
            #     single M-tile per CTA, where the noise in earlier
            #     sweeps had pointed the other way; see B.28 clean
            #     re-bench).
            #   BlockK=64 + stages=4 is the fallback when shape_k
            #     isn't divisible by 128.
            block_n = 128
            # shape_n must be a multiple of block_n; humming asserts
            # this in `check_shape`. Fall back to 64 if shape_n's only
            # divisible by 64.
            if meta.shape_n % 128 != 0:
                if meta.shape_n % 64 == 0:
                    block_n = 64
                else:
                    # No valid TCGEN05 BlockN for this shape; fall
                    # through to the mma.sync heuristic.
                    return super().get_config(
                        meta=meta, shape_m=shape_m,
                        use_f16_accum=use_f16_accum,
                        use_batch_invariant=use_batch_invariant,
                        gemm_type=gemm_type,
                    )
            # BlockK + num_stages depend on B-dtype's SMEM footprint;
            # `_tcgen05_config_for_b_dtype` keeps the per-dtype
            # mapping next to the opted-in set above.
            block_k, num_stages = _tcgen05_config_for_b_dtype(
                meta.b_dtype, meta.shape_k % 128 == 0,
            )
            return {
                "block_shape": (128, block_n, block_k),
                "warp_shape": (32, 64, block_k),
                "num_stages": num_stages,
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
                # Same stream-K hazard as the TS config above: the
                # tcgen05 epilogue does not implement stream-K's
                # cross-CTA partial-K reduction, so leaving the
                # HummingKernel default (True) corrupts outputs at
                # large K (mean|err| ~0.03 at M=256 N=512 K=4096 vs
                # ~4e-6 off). Every tcgen05 test builds kernels with
                # use_stream_k=False, so this was never exercised
                # through the heuristic. Pin it off.
                "use_stream_k": False,
                # L2 rasterization grouping (tune/raster.py) is tuned
                # for the mma.sync/wgmma kernels; the tcgen05 configs
                # were benched without it. Pin 1 to opt out until a
                # tcgen05 raster sweep says otherwise.
                "raster_group_m": 1,
            }

        return super().get_config(
            meta=meta,
            shape_m=shape_m,
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
            gemm_type=gemm_type,
        )
