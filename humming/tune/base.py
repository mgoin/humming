import torch

from humming.layer import HummingLayerMeta


class DeviceHeuristics:
    max_smem_size: int = 0

    @classmethod
    def get_num_sms(cls):
        return torch.cuda.get_device_properties().multi_processor_count

    @classmethod
    def get_l2_cache_size(cls):
        return torch.cuda.get_device_properties().L2_cache_size

    @classmethod
    def get_config_b4(
        cls,
        meta: HummingLayerMeta,
        shape_m: int,
        use_stream_k: bool,
        use_f16_accum: bool,
    ):
        raise NotImplementedError

    @classmethod
    def get_config_b8(
        cls,
        meta: HummingLayerMeta,
        shape_m: int,
        use_stream_k: bool,
        use_f16_accum: bool,
    ):
        raise NotImplementedError

    @classmethod
    def get_config_b16(
        cls,
        meta: HummingLayerMeta,
        shape_m: int,
        use_stream_k: bool,
        use_f16_accum: bool,
    ):
        raise NotImplementedError

    @classmethod
    def get_config(
        cls,
        meta: HummingLayerMeta,
        shape_m: int,
        use_stream_k: bool,
        use_f16_accum: bool,
    ):
        get_config_func = cls.get_config_b16
        if meta.a_dtype.num_bits == 8:
            get_config_func = cls.get_config_b8
        elif meta.a_dtype.num_bits == 4:
            get_config_func = cls.get_config_b4

        return get_config_func(meta, shape_m, use_stream_k, use_f16_accum)

    @classmethod
    def get_configs(
        cls,
        meta: HummingLayerMeta,
        use_stream_k: bool,
        use_f16_accum: bool,
    ):
        last_shape_m = 0
        configs: list[list[int | dict]] = []
        last_config_str: str = ""

        if meta.num_experts is None:
            max_shape_m = 8192
        else:
            max_shape_m = int(meta.num_experts / meta.top_k * 256)

        for shape_m in range(16, max_shape_m, 16):
            config = cls.get_config(meta, shape_m, use_stream_k, use_f16_accum)
            config_str = str(config)

            if last_config_str == config_str:
                configs[-1][1] = shape_m
            else:
                configs.append([last_shape_m, shape_m, config])

            last_config_str = config_str
            last_shape_m = shape_m

        configs[-1][1] = 1 << 30

        return configs
