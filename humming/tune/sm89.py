import math

import numpy as np

from humming import dtypes
from humming.layer import HummingLayerMeta
from humming.tune.base import DeviceHeuristics
from humming.utils.smem import estimate_smem_size_layer


class Sm89Heuristics(DeviceHeuristics):
    max_smem_size: int = 99 * 1024

    @classmethod
    def get_max_block_shape_nk(cls, meta: HummingLayerMeta):
        max_block_shape_n = 256
        while meta.shape_n % max_block_shape_n != 0:
            max_block_shape_n = max_block_shape_n // 2
        assert max_block_shape_n >= 64

        max_block_shape_k = 256
        if meta.shape_k % max_block_shape_k != 0:
            max_block_shape_k = max_block_shape_k // 2
        assert meta.shape_k % max_block_shape_k == 0

        return max_block_shape_n, max_block_shape_k

    @classmethod
    def get_block_shape_m(
        cls,
        max_block_shape_m: int,
        shape_m: int,
        num_experts: int | None = None,
        top_k: int = 0,
    ):
        if num_experts is not None:
            shape_m = int(top_k * shape_m / num_experts / 0.9)
        if shape_m <= max_block_shape_m:
            block_shape_m = math.ceil(shape_m / 16) * 16
        else:
            blocks = [math.ceil(shape_m / ((i + 1) * 16)) for i in range(max_block_shape_m // 16)]
            block_shape_m = np.argmin(blocks).item() * 16 + 16

        return block_shape_m

    @classmethod
    def get_config_b4(
        cls,
        meta: HummingLayerMeta,
        shape_m: int,
        use_stream_k: bool,
        use_f16_accum: bool,
    ):
        assert meta.a_dtype == dtypes.int4
        assert not use_f16_accum
        max_block_shape_n, max_block_shape_k = cls.get_max_block_shape_nk(meta)

        max_block_shape_m = 64
        block_shape_m = cls.get_block_shape_m(
            max_block_shape_m=max_block_shape_m,
            shape_m=shape_m,
            num_experts=meta.num_experts,
            top_k=meta.top_k,
        )
        warp_shape_m = block_shape_m
        warp_shape_n, warp_shape_k = 32, 128
        block_shape_n = min(max_block_shape_n, 256)
        num_warps_n = block_shape_n // warp_shape_n
        num_warps_k = 8 // num_warps_n
        block_shape_k = min(max_block_shape_k, num_warps_k * warp_shape_k)

        block_shape = (block_shape_m, block_shape_n, 256)
        smem_size = estimate_smem_size_layer(meta, block_shape, 3)
        if smem_size < cls.max_smem_size:
            warp_shape_k *= 256 // block_shape_k
            block_shape_k = 256

        return {
            "block_shape": (block_shape_m, block_shape_n, block_shape_k),
            "warp_shape": (warp_shape_m, warp_shape_n, warp_shape_k),
            "use_f16_accum": use_f16_accum,
            "num_sms": cls.get_num_sms(),
            "num_stages": 3,
            "num_ctas_per_sm": 1,
        }

    @classmethod
    def get_config_b8(
        cls,
        meta: HummingLayerMeta,
        shape_m: int,
        use_stream_k: bool,
        use_f16_accum: bool,
    ):
        max_block_shape_n, max_block_shape_k = cls.get_max_block_shape_nk(meta)

        min_group_size = min(
            meta.input_scale_group_size or meta.shape_k,
            meta.weight_scale_group_size or meta.shape_k,
        )
        if min_group_size == meta.shape_k:
            min_group_size = 0

        max_block_shape_m = 128 if use_f16_accum or min_group_size == 0 else 64
        block_shape_m = cls.get_block_shape_m(
            max_block_shape_m=max_block_shape_m,
            shape_m=shape_m,
            num_experts=meta.num_experts,
            top_k=meta.top_k,
        )
        warp_shape_m = block_shape_m
        warp_shape_n, warp_shape_k = 32, 64
        block_shape_n = min(max_block_shape_n, 256)
        num_warps_n = block_shape_n // warp_shape_n
        num_warps_k = 8 // num_warps_n
        block_shape_k = min(max_block_shape_k, num_warps_k * warp_shape_k)

        if min_group_size >= 128 or min_group_size == 0 and max_block_shape_k == 128:
            if block_shape_n >= 128 and meta.b_dtype.num_bits <= 4:
                block_shape = (block_shape_m, block_shape_n, 128)
                smem_size = estimate_smem_size_layer(meta, block_shape, 3)
                if smem_size < cls.max_smem_size:
                    warp_shape_k *= 128 // block_shape_k
                    block_shape_k = 128

        return {
            "block_shape": (block_shape_m, block_shape_n, block_shape_k),
            "warp_shape": (warp_shape_m, warp_shape_n, warp_shape_k),
            "use_f16_accum": use_f16_accum,
            "num_sms": cls.get_num_sms(),
            "num_stages": 3,
            "num_ctas_per_sm": 1,
        }

    @classmethod
    def get_config_b16(
        cls,
        meta: HummingLayerMeta,
        shape_m: int,
        use_stream_k: bool,
        use_f16_accum: bool,
    ):
        max_block_shape_n, max_block_shape_k = cls.get_max_block_shape_nk(meta)
        max_block_shape_m = 128 if use_f16_accum else 64
        block_shape_m = cls.get_block_shape_m(
            max_block_shape_m=max_block_shape_m,
            shape_m=shape_m,
            num_experts=meta.num_experts,
            top_k=meta.top_k,
        )
        warp_shape_m = block_shape_m
        warp_shape_n, warp_shape_k = 64, 32
        block_shape_n = min(max_block_shape_n, 256)
        num_warps_n = block_shape_n // warp_shape_n
        num_warps_k = 8 // num_warps_n
        block_shape_k = min(max_block_shape_k, num_warps_k * warp_shape_k)

        return {
            "block_shape": (block_shape_m, block_shape_n, block_shape_k),
            "warp_shape": (warp_shape_m, warp_shape_n, warp_shape_k),
            "use_f16_accum": use_f16_accum,
            "num_sms": cls.get_num_sms(),
            "num_stages": 3,
            "num_ctas_per_sm": 1,
        }
