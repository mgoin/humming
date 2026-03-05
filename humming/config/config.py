import dataclasses
import re

from humming.config.base import BaseHummingConfigClass
from humming.config.enum import ActivationType, MmaType


@dataclasses.dataclass
class SchedulerConfig(BaseHummingConfigClass):
    use_stream_k: bool = True


@dataclasses.dataclass
class PipelineConfig(BaseHummingConfigClass):
    use_warp_spec: bool | None = None
    num_stages: int = 2
    num_threads: int | None = None
    num_ctas_per_sm: int = 1
    num_math_threads: int | None = None
    num_load_threads: int | None = None
    use_mbarrier: bool | None = None
    use_cp_async: bool | None = None
    use_tma: bool | None = None
    use_tma_a: bool | None = None
    use_tma_b: bool | None = None
    use_tma_c: bool | None = None
    use_tma_bs: bool | None = None
    use_tma_bzp: bool | None = None
    use_tma_bias: bool | None = None

    _name_map = {
        "use_mbarrier": "kUseMBarrier",
        "use_tma_bs": "kUseTmaBS",
        "use_tma_bzp": "kUseTmaBZP",
    }

    @classmethod
    def from_dict(cls, raw_config):
        clean_config = cls._preprocess_dict(raw_config)
        config = cls(**clean_config)

        assert config.num_math_threads is not None

        if config.use_warp_spec is None:
            config.use_warp_spec = raw_config["sm_version"] >= 90

        if config.use_warp_spec:
            config.num_load_threads = 128
            config.num_threads = config.num_math_threads + 128
        else:
            config.num_load_threads = config.num_math_threads
            config.num_threads = config.num_math_threads

        if config.use_tma is None:
            config.use_tma = raw_config["sm_version"] >= 90

        for name in dir(config):
            if not name.startswith("use_tma_"):
                continue
            if not config.use_tma:
                assert getattr(config, name) is not True
            if getattr(config, name) is None:
                setattr(config, name, config.use_tma)

        if config.use_mbarrier is None:
            config.use_mbarrier = config.use_warp_spec or config.use_tma

        if config.use_cp_async is None:
            config.use_cp_async = raw_config["sm_version"] >= 80

        return config


@dataclasses.dataclass
class QuantParamConfig(BaseHummingConfigClass):
    has_input_scale: bool | None = None
    has_weight_scale: bool = True
    input_scale_group_size: int = 0
    weight_scale_group_size: int = 0
    has_global_scale: bool = False
    has_zero_point: bool = False


@dataclasses.dataclass
class EpilogueConfig(BaseHummingConfigClass):
    has_bias: bool = False
    activation_type: ActivationType = ActivationType.NONE
    custom_activation_func_impl: str | None = None

    _cpp_ignore_names = {"custom_activation_func_impl"}

    def prepare_custom_activation_func(self):
        impl_text = self.custom_activation_func_impl
        if self.activation_type == ActivationType.CUSTOM:
            template = (
                "template <>\n"
                "CUDA_INLINE\n"
                "float activation_func<ActivationType::CUSTOM>(const float a) {{\n"
                "{impl_text}\n"
                "}};"
            )
        elif self.activation_type == ActivationType.CUSTOM_GLU:
            template = (
                "template <>\n"
                "CUDA_INLINE\n"
                "float activation_func<ActivationType::CUSTOM_GLU>(const float2 a) {{\n"
                "{impl_text}\n"
                "}};"
            )
        else:
            return ""

        assert isinstance(impl_text, str) and impl_text.strip()
        impl_text = impl_text.strip()
        impl_text = re.sub("\\s+\n\\s+", "\n  ", impl_text)
        return template.format(impl_text=impl_text)


@dataclasses.dataclass
class MoEConfig(BaseHummingConfigClass):
    top_k: int = 0
    is_moe: bool = False
    is_moe_down: bool = False

    _name_map = {
        "is_moe": "kIsMoE",
        "is_moe_down": "kIsMoEDown",
    }


@dataclasses.dataclass
class MmaConfig(BaseHummingConfigClass):
    mma_type: MmaType | None = None
    use_f16_accum: bool = False

    @classmethod
    def from_dict(cls, raw_config):
        clean_config = cls._preprocess_dict(raw_config)
        config = cls(**clean_config)

        if config.mma_type is None:
            if raw_config["sm_version"] == 90:
                config.mma_type = MmaType.WGMMA
            elif raw_config["sm_version"] >= 75:
                config.mma_type = MmaType.MMA

        return config
