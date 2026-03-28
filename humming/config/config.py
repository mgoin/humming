import dataclasses
import math
from typing import ClassVar

from typing_extensions import Self

from humming.config.base import BaseHummingConfig
from humming.config.enum import MmaType, WeightScaleType


@dataclasses.dataclass
class SchedulerConfig(BaseHummingConfig):
    use_stream_k: bool = True
    use_m_major_scheduler: bool = False


@dataclasses.dataclass
class PipelineConfig(BaseHummingConfig):
    use_warp_spec: bool | None = None
    num_stages: int = 2
    num_ctas_per_sm: int = 1
    use_mbarrier: bool | None = None
    use_cp_async: bool | None = None
    use_tma: bool | None = None
    use_tma_a: bool | None = None
    use_tma_b: bool | None = None
    use_tma_c: bool | None = None
    use_tma_bs: bool | None = None
    use_tma_bzp: bool | None = None
    use_tma_bias: bool | None = None
    num_write_splits: int = 1
    multi_cast_size: int = 1

    _cpp_extra_names: ClassVar[tuple[str, ...]] = (
        "num_threads",
        "num_math_threads",
        "num_load_threads",
    )

    _name_map = {
        "use_mbarrier": "kUseMBarrier",
        "use_tma_bs": "kUseTmaBS",
        "use_tma_bzp": "kUseTmaBZP",
    }

    def __post_init__(self):
        self.num_threads = 0
        self.num_load_threads = 0
        self.num_math_threads = 0

    @classmethod
    def from_dict(cls, raw_config) -> Self:
        clean_config = cls._preprocess_dict(raw_config)
        config = cls(**clean_config)

        block_shape = raw_config["block_shape"]
        warp_shape = raw_config["warp_shape"]
        config.num_math_threads = math.prod(block_shape) // math.prod(warp_shape) * 32

        if config.use_warp_spec is None:
            config.use_warp_spec = False

        if config.use_warp_spec:
            config.num_load_threads = 128
            config.num_threads = config.num_math_threads + 128
        else:
            config.num_load_threads = config.num_math_threads
            config.num_threads = config.num_math_threads

        if config.use_tma is None:
            config.use_tma = False

        for name in dir(config):
            if not name.startswith("use_tma_"):
                continue
            if not config.use_tma:
                assert getattr(config, name) is not True
            if getattr(config, name) is None:
                setattr(config, name, config.use_tma)

        if config.use_mbarrier is None:
            config.use_mbarrier = False

        if config.use_cp_async is None:
            config.use_cp_async = raw_config["sm_version"] >= 80

        return config


@dataclasses.dataclass
class QuantParamConfig(BaseHummingConfig):
    input_scale_group_size: int = 0
    weight_scale_group_size: int = 0
    weight_scale_group_size_n: int = 0
    weight_scale_type: WeightScaleType | None = None
    use_int_weight_scale: bool = False
    has_zero_point: bool = False
    is_fp_zero_point: bool = False

    _cpp_extra_names: ClassVar[tuple[str, ...]] = (
        "is_channel_weight_scale",
        "is_block_weight_scale",
        "is_group_weight_scale",
        "is_tensor_weight_scale",
    )

    def __post_init__(self):
        super().__post_init__()
        if self.weight_scale_type is None:
            if self.weight_scale_group_size_n > 1:
                self.weight_scale_type = WeightScaleType.BLOCK
            elif self.weight_scale_group_size > 0:
                self.weight_scale_type = WeightScaleType.GROUP
            elif self.weight_scale_group_size == 0:
                self.weight_scale_type = WeightScaleType.CHANNEL

        self.is_channel_weight_scale = self.weight_scale_type == WeightScaleType.CHANNEL
        self.is_tensor_weight_scale = self.weight_scale_type in [
            WeightScaleType.TENSOR,
            WeightScaleType.GROUP_TENSOR,
        ]
        self.is_block_weight_scale = self.weight_scale_type == WeightScaleType.BLOCK
        self.is_group_weight_scale = self.weight_scale_type in [
            WeightScaleType.GROUP,
            WeightScaleType.GROUP_TENSOR,
        ]

    @classmethod
    def from_dict(cls, raw_config) -> Self:
        config = super().from_dict(raw_config)
        config.__post_init__()
        return config


@dataclasses.dataclass
class EpilogueConfig(BaseHummingConfig):
    has_bias: bool = False


@dataclasses.dataclass
class MoEConfig(BaseHummingConfig):
    top_k: int = 0
    is_moe: bool = False
    is_moe_down: bool = False

    _name_map = {
        "is_moe": "kIsMoE",
        "is_moe_down": "kIsMoEDown",
    }


@dataclasses.dataclass
class MmaConfig(BaseHummingConfig):
    mma_type: MmaType | None | str = None
    use_f16_accum: bool = False

    @classmethod
    def from_dict(cls, raw_config) -> Self:
        clean_config = cls._preprocess_dict(raw_config)
        config = cls(**clean_config)

        if config.mma_type is None:
            if raw_config["sm_version"] == 90:
                config.mma_type = MmaType.WGMMA
            elif raw_config["sm_version"] >= 75:
                config.mma_type = MmaType.MMA
        elif config.mma_type == "mma":
            config.mma_type = MmaType.MMA
        elif config.mma_type == "wgmma":
            config.mma_type = MmaType.WGMMA

        return config
