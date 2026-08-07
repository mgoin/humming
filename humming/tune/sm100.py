"""sm_100 (Blackwell) tuning heuristics.

Two tcgen05 mainloops sit on top of the mma.sync baseline, and neither is
reachable without the per-layer ``mma_type="tcgen05"`` opt-in: a default
LayerConfig resolves to exactly the mma.sync config it did before tcgen05
existed.

* TS mode stages the dequantised weights in TMEM. It is the tcgen05 default,
  1.38-1.42x faster than SS at every M where both are legal
  (benchmarks/bench_ts_vs_ss.py). Against the mma.sync config this heuristic
  otherwise emits, TS wins 1.08-1.11x at M=2048, is level at M=512 and loses
  below: with use_stream_k off and raster_group_m=1 the grid is N/BlockN CTAs,
  so at M=16 only 64 of 148 SMs have work. TS packs weights, scales and zero
  points in a layout no other kernel reads, so a TS layer runs TS at every
  shape_m -- there is no per-M fallback. Its pipeline depth is per-dtype
  (_TS_B_DTYPE_STAGES) rather than a single cap: the depth that suits one
  dequant arm costs up to 8% on another.
* SS mode stages them in SMEM. It is the substrate TS was built on and the
  tcgen05 fallback for layers TS is not legal for. It never beats mma.sync at
  any shape measured -- 0.13-0.91x in benchmarks/bench_tcgen05_vs_wmma.py and
  0.56-0.94x in benchmarks/bench_tcgen05_dtypes.py -- which is exactly why
  selection is opt-in rather than automatic.
"""

from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType, WeightScale2Type, WeightScaleType
from humming.tune.sm8x import Sm80Heuristics
from humming.utils.smem import estimate_smem_size_layer

# Every weight dtype the library pairs with a bf16 activation, mapped to the
# (block_k, num_stages) SS runs it at. SS reads the ordinary mma.sync weight
# layout, so this is a tuning table and not a capability gate: the entries
# benchmarks/bench_tcgen05_dtypes.py swept keep their tuned values, and the rest
# take the deepest pipeline estimate_smem_size_layer fits at BlockK=128 -- the
# bf16 b_dequant staging buffer binds there (uint4 at four stages needs
# 234,496 B against the device's 232,448 B cap).
_SS_B_DTYPE_CONFIG: dict[dtypes.DataType, tuple[int, int]] = {
    dtypes.DataType.from_str(name): value
    for name, value in {
        "uint1": (128, 4),
        "uint2": (128, 4),
        "uint3": (128, 4),
        "uint4": (128, 3),
        "uint5": (128, 3),
        "uint6": (128, 3),
        "uint7": (128, 3),
        "uint8": (64, 4),
        "float3e1m1": (128, 4),
        "float3e2m0": (128, 4),
        "float4e2m1": (128, 3),
        "float4e3m0": (128, 3),
        "float5e2m2": (128, 3),
        "float5e4m0": (128, 3),
        "float6e2m3": (128, 3),
        "float6e4m1": (128, 3),
        "float7e2m4": (128, 3),
        "float7e4m2": (128, 3),
        "float7e6m0": (128, 3),
        "float8e1m6": (128, 3),
        "float8e4m3": (128, 3),
        "float8e5m2": (128, 3),
    }.items()
}

# Weight-scale dtypes the generic mainloop dequantises into ElementA before
# applying them on B. A 16-bit scale is reinterpreted as ElementA bit-for-bit,
# so only bs_dtype == a_dtype is legal there.
_SS_GROUP_BS_DTYPES = (
    dtypes.bfloat16,
    dtypes.float8e4m3,
    dtypes.float8e5m2,
    dtypes.float8e8m0,
)

_TS_GEMM_TYPES = (GemmType.DENSE, GemmType.GROUPED_CONTIGUOUS, GemmType.GROUPED_MASKED)

# TS num_stages for the dense cells that want something other than the default
# four, keyed on (b_dtype, integer zero point). TS stages only the packed
# weights -- there is no bf16 b_dequant buffer -- so nothing here is SMEM-bound
# below nine stages and the depth is pure latency tuning of ts_dequant_b_pair:
# uint4's zero-point-folded uint_to_f16 hides a fifth stage, while uint8+zp,
# the one bf16 arm that splits its exponent offset across two multiplies, peaks
# at three and loses 8% by five. The key is the dequant arm rather than the
# weight width: an fp zero point is a separate post-dequant subtract and tracks
# the zero-point-free timing at four.
_TS_B_DTYPE_STAGES: dict[tuple[dtypes.DataType, bool], int] = {
    (dtypes.uint4, True): 5,
    (dtypes.uint8, True): 3,
}
_TS_DEFAULT_NUM_STAGES = 4


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
    def supports_tcgen05_ss(cls, layer_config: LayerConfig) -> bool:
        """Whether the SS-mode tcgen05 kernel can run `layer_config`.

        Mirrors the static_asserts in mma/tcgen05_mma.cuh: bf16 activations
        against a narrow B dtype, read from the ordinary mma.sync weight layout.
        Every quantisation parameter is admitted by name rather than by absence
        of a check, because the SS drain replicates only what the mainloop
        applies: anything the epilogue smem writer would have applied is
        silently dropped instead of raising. Gates dispatch and packing, which
        must agree, so transform_humming_tensors calls it too.
        """
        if layer_config.a_dtype != dtypes.bfloat16:
            # Deliberately narrower than the TS gate, which also admits fp16:
            # mma/tcgen05_mma.cuh static_asserts ElementA == BFloat16 because
            # the SS r2s scatter and its drain_accum epilogue are written
            # against bf16 bit patterns (__nv_bfloat162 / __floats2bfloat162_rn).
            # An fp16 layer TS cannot take is rejected, not downgraded to SS.
            return False
        if layer_config.b_dtype not in _SS_B_DTYPE_CONFIG:
            return False
        if layer_config.weight_scale_2_type != WeightScale2Type.NONE:
            # weight_scale_2 lives in EpilogueArithmetic::may_apply_on_smem_write,
            # which drain_accum bypasses.
            return False
        # CHANNEL and TENSOR weight scales are applied on C, and
        # should_apply_bs_on_c is False for TCGEN05, so only the scale kinds the
        # mainloop applies on B are legal.
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
                "LayerConfig.tcgen05_supported) nor the SS fallback"
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
        config["num_stages"] = cls._fit_num_stages(
            layer_config, config, gemm_type, cls._ts_max_num_stages(layer_config, gemm_type)
        )
        return config

    @classmethod
    def _ts_max_num_stages(cls, layer_config: LayerConfig, gemm_type: GemmType) -> int:
        if gemm_type != GemmType.DENSE:
            # Both tuned dense depths lose 1.6-3.8% against the default at
            # E=8, so the grouped scheduler keeps it.
            return _TS_DEFAULT_NUM_STAGES
        has_int_zero_point = layer_config.has_zero_point and not layer_config.is_fp_zero_point
        key = (layer_config.b_dtype, has_int_zero_point)
        return _TS_B_DTYPE_STAGES.get(key, _TS_DEFAULT_NUM_STAGES)

    @classmethod
    def _ss_config(cls, layer_config: LayerConfig, gemm_type: GemmType) -> dict | None:
        """SS-mode config, or None when SS cannot run this layer.

        Reached only under the mma_type="tcgen05" opt-in, and only where TS is
        illegal. SS never beats the mma.sync config this heuristic otherwise
        emits, so it carries no profitability cutoff: an opted-in layer gets
        tcgen05 at every shape_m, or an error.
        """
        if gemm_type != GemmType.DENSE or not cls.supports_tcgen05_ss(layer_config):
            return None

        block_n = 128 if layer_config.shape_n % 128 == 0 else 64

        # BlockK=128 halves the K-iter count and wins 3-8% over BlockK=64
        # wherever shape_k and the weight-scale group allow it; the per-dtype
        # entry picks the deepest pipeline that still fits in SMEM.
        block_k, num_stages = _SS_B_DTYPE_CONFIG[layer_config.b_dtype]
        group_size = layer_config.weight_scale_group_size
        if layer_config.shape_k % block_k or (group_size % block_k and block_k % group_size):
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
