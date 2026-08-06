"""sm_100 (Blackwell) tuning heuristics.

Two tcgen05 paths sit on top of the mma.sync baseline:

* TS mode, opted into per layer with ``mma_type="tcgen05"``. It packs weights,
  scales and zero points in a layout no other kernel reads, so a TS layer runs
  TS at every shape_m -- there is no per-M fallback, and an illegal shape is an
  error rather than a silent downgrade.
* SS mode, auto-selected for a narrow bf16-activation window where it
  benchmarks 1.20-1.55x faster than mma.sync (benchmarks/bench_ts_vs_ss.py).
"""

import math

from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType
from humming.tune.sm8x import Sm80Heuristics
from humming.utils.smem import estimate_smem_size_layer

# B dtypes opted into SS mode, mapped to the (block_k, num_stages) each was
# tuned to by benchmarks/bench_tcgen05_dtypes.py. Wide dtypes drop to fewer
# stages (or BlockK=64) because the bf16 b_dequant staging buffer pushes
# BlockK=128 stages=4 over the SMEM cap.
_SS_B_DTYPE_CONFIG: dict[dtypes.DataType, tuple[int, int]] = {
    dtypes.uint3: (128, 4),
    dtypes.uint4: (128, 4),
    dtypes.float4e2m1: (128, 4),
    dtypes.uint5: (128, 3),
    dtypes.uint6: (128, 3),
    dtypes.float8e4m3: (128, 3),
    dtypes.float8e5m2: (128, 3),
    dtypes.uint8: (64, 4),
}

_TS_GEMM_TYPES = (GemmType.DENSE, GemmType.GROUPED_CONTIGUOUS, GemmType.GROUPED_MASKED)


class Sm100Heuristics(Sm80Heuristics):
    # Blackwell datacenter dies expose 228 KiB of shared memory per SM to a
    # CTA; round down for the driver's reserved bytes (same as Sm90).
    max_smem_size: int = 227 * 1024
    sm_version: int = 100
    b8_allowed_dtypes: list[dtypes.DataType] = [dtypes.int8, dtypes.float8e4m3, dtypes.float8e5m2]

    @classmethod
    def supports_tcgen05_ts(cls, layer_config: LayerConfig) -> bool:
        return layer_config.tcgen05_supported

    @classmethod
    def get_config(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        if layer_config.mma_type == MmaType.TCGEN05:
            assert cls.supports_tcgen05_ts(layer_config), (
                "mma_type='tcgen05' selects the TS kernel, which this layer is "
                "not legal for (see LayerConfig.tcgen05_supported)"
            )
            assert gemm_type in _TS_GEMM_TYPES, f"tcgen05 TS does not support {gemm_type}"
            assert not use_f16_accum, "tcgen05 TS accumulates in f32 TMEM"
            assert not use_batch_invariant, "tcgen05 TS does not implement batch-invariant reduction"
            return cls._ts_config(layer_config, shape_m, gemm_type)

        ss_config = cls._ss_config(layer_config, shape_m, use_batch_invariant, gemm_type)
        if ss_config is not None:
            return ss_config

        return super().get_config(
            layer_config=layer_config,
            shape_m=shape_m,
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
            gemm_type=gemm_type,
        )

    @classmethod
    def _ts_config(cls, layer_config: LayerConfig, shape_m: int, gemm_type: GemmType) -> dict:
        if gemm_type == GemmType.DENSE:
            block_m = 128 if shape_m >= 128 else 64
        else:
            # Grouped: shape_m counts padded tokens over all experts, but the
            # scheduler tiles per expert, so tokens-per-expert sets occupancy.
            # BlockM=32 is a legal TS atom but is never selected -- the
            # fine-grained cost is per-tile r2t/handshake overhead rather than
            # MMA-row waste, so a smaller token tile does not recover it.
            tokens_per_expert = shape_m // max(layer_config.num_experts, 1)
            block_m = 128 if tokens_per_expert >= 128 else 64

        config = {
            "block_shape": (block_m, 128, 64),
            "warp_shape": (block_m, 32, 64),
            "num_ctas_per_sm": 1,
            "num_write_splits": 1,
            "mma_type": "tcgen05",
            "use_tcgen05": True,
            "use_tcgen05_ts": True,
            "use_warp_spec": True,
            "use_tma": True,
            "use_cp_async": False,
            "use_mbarrier": True,
            # A group weight scale with a zero point trips the launcher's
            # BZP assert under TMA; BZP is small, so keep it on cp.async.
            "use_tma_bzp": False,
            # The TS epilogue drains TMEM straight to the TMA-C store and has
            # no cross-CTA partial-K reduction, so stream-K would corrupt any
            # output whose K is split across CTAs.
            "use_stream_k": False,
            # tune/raster.py's grouping is tuned for the mma.sync/wgmma
            # kernels; opt out until a tcgen05 raster sweep says otherwise.
            "raster_group_m": 1,
        }
        config["num_stages"] = cls._fit_num_stages(layer_config, config, gemm_type, 4)
        return config

    @classmethod
    def _ss_config(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_batch_invariant: bool,
        gemm_type: GemmType,
    ) -> dict | None:
        """SS-mode config, or None when mma.sync should win.

        Supported (mirrors the static_asserts in mma/tcgen05_mma.cuh): dense
        bf16 x narrow-B with a group weight scale. Profitable cutoffs come from
        benchmarks/bench_ts_vs_ss.py: below M=128 mma.sync's smaller tiles
        spread better, fat-K weights only win from M=512 up, and below ~64
        output tiles a 128x128 CTA tile leaves most SMs idle.
        """
        if gemm_type != GemmType.DENSE or use_batch_invariant:
            return None
        if layer_config.a_dtype != dtypes.bfloat16:
            return None
        if layer_config.b_dtype not in _SS_B_DTYPE_CONFIG:
            return None
        if layer_config.weight_scale_group_size <= 0:
            return None
        if shape_m < 128:
            return None
        if layer_config.shape_n < layer_config.shape_k and shape_m < 512:
            return None
        num_tiles = math.ceil(layer_config.shape_n / 128) * math.ceil(shape_m / 128)
        if num_tiles < 64:
            return None

        block_n = 128
        if layer_config.shape_n % 128 != 0:
            if layer_config.shape_n % 64 != 0:
                return None
            block_n = 64

        # BlockK=128 halves the K-iter count and wins 3-8% over BlockK=64
        # wherever shape_k allows it; the per-dtype entry picks the stage count
        # that still fits in SMEM.
        block_k, num_stages = _SS_B_DTYPE_CONFIG[layer_config.b_dtype]
        if layer_config.shape_k % 128 != 0:
            block_k, num_stages = 64, 4

        config = {
            "block_shape": (128, block_n, block_k),
            "warp_shape": (32, 64, block_k),
            "num_ctas_per_sm": 1,
            "num_write_splits": 1,
            "mma_type": "tcgen05",
            "use_tcgen05": True,
            "use_warp_spec": True,
            "use_tma": True,
            "use_cp_async": False,
            "use_mbarrier": True,
            "use_tma_bzp": False,
            "use_stream_k": False,
            "raster_group_m": 1,
        }
        config["num_stages"] = cls._fit_num_stages(layer_config, config, gemm_type, num_stages)
        return config

    @classmethod
    def _fit_num_stages(
        cls,
        layer_config: LayerConfig,
        config: dict,
        gemm_type: GemmType,
        max_num_stages: int,
    ) -> int:
        best = 2
        for num_stages in range(3, max_num_stages + 1):
            smem_size = estimate_smem_size_layer(
                layer_config,
                config["block_shape"],
                gemm_type,
                num_stages,
                warp_shape=config["warp_shape"],
                use_mbarrier=True,
                use_warp_spec=config["use_warp_spec"],
                num_write_splits=config["num_write_splits"],
                use_tcgen05=config["use_tcgen05"],
                use_tcgen05_ts=config.get("use_tcgen05_ts", False),
            )
            if smem_size <= cls.max_smem_size:
                best = num_stages
        return best
