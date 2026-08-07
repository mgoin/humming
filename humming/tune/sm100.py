from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType, WeightScale2Type, WeightScaleType
from humming.tune.sm8x import Sm80Heuristics
from humming.utils.smem import estimate_smem_size_layer

# Weight dtypes the SS mainloop is wired for; membership gates SS.
TCGEN05_SS_B_DTYPES = tuple(
    dtypes.DataType.from_str(name)
    for name in (
        "uint1",
        "uint2",
        "uint3",
        "uint4",
        "uint5",
        "uint6",
        "uint7",
        "uint8",
        "float3e1m1",
        "float3e2m0",
        "float4e2m1",
        "float4e3m0",
        "float5e2m2",
        "float5e4m0",
        "float6e2m3",
        "float6e4m1",
        "float7e2m4",
        "float7e4m2",
        "float7e6m0",
        "float8e1m6",
        "float8e4m3",
        "float8e5m2",
    )
)

# SS pipeline depth: _fit_num_stages takes the deepest that SMEM holds under
# this cap. At BlockN=128/BlockK=128 the bf16 b_dequant staging buffer binds and
# that is four stages up to 3-bit codes and three above; the half-width
# BlockN=64 tiles hold four throughout. Deeper than four measures as noise, and
# no weight dtype wants a shallower pipeline than SMEM forces.
# Retune with benchmarks/bench_tcgen05_dtypes.py --block_k --num_stages.
_SS_MAX_NUM_STAGES = 4

_SS_GROUP_BS_DTYPES = (
    dtypes.bfloat16,
    dtypes.float8e4m3,
    dtypes.float8e5m2,
    dtypes.float8e8m0,
)

_TS_GEMM_TYPES = (GemmType.DENSE, GemmType.GROUPED_CONTIGUOUS, GemmType.GROUPED_MASKED)

# Per-dtype TS pipeline depth cap; five is the knee except for uint8, whose
# stage is wide enough that a fifth costs throughput.
# Retune with benchmarks/bench_ts_vs_ss.py --b_dtype --num_stages.
_TS_B_DTYPE_STAGES: dict[dtypes.DataType, int] = {
    dtypes.uint8: 4,
}
_TS_DEFAULT_NUM_STAGES = 5

# The TS mainloop static_asserts kNumStages >= 3 (kernel/humming_ws.cuh) and the
# SS depths are tuned from 3 up, so nothing shallower is offered.
_MIN_NUM_STAGES = 3


# TODO (mgoin): add proper heuristics
class Sm100Heuristics(Sm80Heuristics):
    max_smem_size: int = 227 * 1024
    sm_version: int = 100
    b8_allowed_dtypes: list[dtypes.DataType] = [dtypes.int8, dtypes.float8e4m3, dtypes.float8e5m2]

    @classmethod
    def supports_tcgen05_ts(cls, layer_config: LayerConfig) -> bool:
        # Selecting this class is the device gate; the rest is layer legality.
        return layer_config.tcgen05_ts_supported

    @classmethod
    def supports_tcgen05_ss(cls, layer_config: LayerConfig) -> bool:
        if layer_config.a_dtype != dtypes.bfloat16 or layer_config.c_dtype != dtypes.bfloat16:
            # mma/tcgen05_mma.cuh static_asserts ElementA == ElementC == BFloat16:
            # the SS r2s scatter is written against bf16 bit patterns, and
            # drain_accum converts and writes the output as bf16.
            return False
        if layer_config.b_dtype not in TCGEN05_SS_B_DTYPES:
            return False
        if layer_config.weight_scale_2_type != WeightScale2Type.NONE:
            return False
        # Only the scale kinds the mainloop applies on B are legal.
        if layer_config.weight_scale_type == WeightScaleType.GROUP:
            if layer_config.bs_dtype not in _SS_GROUP_BS_DTYPES:
                return False
        elif layer_config.weight_scale_type == WeightScaleType.BLOCK:
            # The block scale is read as one f32 per warp N-tile, and SS pins
            # WarpShape::N at 64.
            if layer_config.bs_dtype != dtypes.float32:
                return False
            if layer_config.weight_scale_group_size_n % 64:
                return False
        else:
            return False
        # GROUP and BLOCK both carry weight_scale_group_size > 0 (LayerConfig).
        group_size = layer_config.weight_scale_group_size
        if group_size % 64 and 64 % group_size:
            # A BlockK stage must hold whole weight-scale groups.
            return False
        return layer_config.shape_n % 64 == 0 and layer_config.shape_k % 64 == 0

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
            assert gemm_type in _TS_GEMM_TYPES, f"tcgen05 does not support {gemm_type}"
            assert not use_f16_accum, "tcgen05 accumulates in f32 TMEM"
            assert not use_batch_invariant, "tcgen05 does not implement batch-invariant reduction"
            if cls.supports_tcgen05_ts(layer_config):
                return cls._ts_config(layer_config, shape_m, gemm_type)
            ss_config = cls._ss_config(layer_config, gemm_type)
            assert ss_config is not None, (
                "mma_type='tcgen05' is legal for neither the TS kernel (see "
                "LayerConfig.tcgen05_ts_supported) nor the SS fallback"
            )
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
            # Retune with benchmarks/bench_ts_moe.py --block_m.
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
            # The TS packed zero point is the row-major [K/gs, N*zp_bits/32]
            # array make_tma_desc_bzp already describes, so TMA loads it
            # bit-identically and the load-bound mainloop keeps the 6-21% it
            # buys on the 2- and 4-bit codes (uint8 is a wash, +-2% by shape).
            # Retune with benchmarks/bench_ts_vs_ss.py --b_dtype.
            "use_tma_bzp": True,
            # The TS epilogue has no cross-CTA partial-K reduction, so stream-K
            # would corrupt any output whose K is split across CTAs.
            "use_stream_k": False,
            # TODO: tune/raster.py's grouping is tuned for mma.sync/wgmma.
            "raster_group_m": 1,
        }
        config["num_stages"] = cls._fit_num_stages(
            layer_config, config, gemm_type, cls._ts_max_num_stages(layer_config)
        )
        return config

    @classmethod
    def _ts_max_num_stages(cls, layer_config: LayerConfig) -> int:
        return _TS_B_DTYPE_STAGES.get(layer_config.b_dtype, _TS_DEFAULT_NUM_STAGES)

    @classmethod
    def _ss_config(cls, layer_config: LayerConfig, gemm_type: GemmType) -> dict | None:
        if gemm_type != GemmType.DENSE or not cls.supports_tcgen05_ss(layer_config):
            return None

        block_n = 128 if layer_config.shape_n % 128 == 0 else 64

        # BlockK=128 halves the K-iter count where shape_k and the group allow it.
        block_k = 128
        group_size = layer_config.weight_scale_group_size
        if layer_config.shape_k % block_k or (group_size % block_k and block_k % group_size):
            block_k = 64

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
        config["num_stages"] = cls._fit_num_stages(layer_config, config, gemm_type, _SS_MAX_NUM_STAGES)
        return config

    @classmethod
    def _fit_num_stages(
        cls,
        layer_config: LayerConfig,
        config: dict,
        gemm_type: GemmType,
        max_num_stages: int,
    ) -> int:
        def smem_size(num_stages: int) -> int:
            return estimate_smem_size_layer(
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

        assert max_num_stages >= _MIN_NUM_STAGES, f"max_num_stages is below the {_MIN_NUM_STAGES}-stage floor"
        best = 0
        for num_stages in range(_MIN_NUM_STAGES, max_num_stages + 1):
            if smem_size(num_stages) <= cls.max_smem_size:
                best = num_stages
        assert best, (
            f"tcgen05 needs {smem_size(_MIN_NUM_STAGES)} B of SMEM for this layer at "
            f"num_stages={_MIN_NUM_STAGES}, over the {cls.max_smem_size} B limit"
        )
        return best
