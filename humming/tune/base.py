import math

import numpy as np
import torch

from humming import dtypes
from humming.layer import HummingLayerMeta
from humming.utils.smem import estimate_smem_size_layer


class DeviceHeuristics:
    max_smem_size: int = 0
    b16_allowed_dtypes: list[dtypes.DataType] = []
    b8_allowed_dtypes: list[dtypes.DataType] = []
    b4_allowed_dtypes: list[dtypes.DataType] = []
    sm_version: int = 0

    @classmethod
    def get_block_shape_k_and_num_ctas_per_sm(
        cls,
        meta: HummingLayerMeta,
        max_num_ctas_per_sm: int,
        num_warps_m: int,
        num_warps_n: int,
        warp_shape_k: int,
        block_shape_m: int,
        block_shape_n: int,
        num_stages: int,
    ):
        assert max_num_ctas_per_sm in [1, 2]
        max_num_warps = 8
        while max_num_ctas_per_sm * num_warps_n * num_warps_m > max_num_warps:
            max_num_ctas_per_sm -= 1

        assert max_num_ctas_per_sm > 0

        num_warps_k = max_num_warps // max_num_ctas_per_sm // num_warps_n // num_warps_m
        block_shape_k = num_warps_k * warp_shape_k
        while meta.shape_k % block_shape_k:
            block_shape_k = block_shape_k // 2
            assert block_shape_k % warp_shape_k == 0

        block_shape = (block_shape_m, block_shape_n, block_shape_k)
        while block_shape_k > warp_shape_k:
            smem_size = estimate_smem_size_layer(meta, block_shape, num_stages)
            smem_size = smem_size * max_num_ctas_per_sm
            if smem_size * max_num_ctas_per_sm <= cls.max_smem_size:
                break
            block_shape_k = block_shape_k // 2
            block_shape = (block_shape_m, block_shape_n, block_shape_k)

        block_shape = (block_shape_m, block_shape_n, block_shape_k)
        smem_size = estimate_smem_size_layer(meta, block_shape, num_stages)
        if smem_size * max_num_ctas_per_sm > cls.max_smem_size:
            assert max_num_ctas_per_sm > 1
            return cls.get_block_shape_k_and_num_ctas_per_sm(
                meta=meta,
                max_num_ctas_per_sm=max_num_ctas_per_sm - 1,
                num_warps_m=num_warps_m,
                num_warps_n=num_warps_n,
                warp_shape_k=warp_shape_k,
                block_shape_m=block_shape_m,
                block_shape_n=block_shape_n,
                num_stages=num_stages,
            )

        return block_shape_k, max_num_ctas_per_sm

    @classmethod
    def get_special_config_b8_g128(
        cls,
        meta: HummingLayerMeta,
        block_shape: tuple[int, int, int],
        num_ctas_per_sm: int,
        use_stream_k: bool,
    ):
        num_stages = 2 if cls.sm_version < 80 else 3
        group_size = min(
            meta.input_scale_group_size or 1024,
            meta.weight_scale_group_size or 1024,
        )
        if meta.a_dtype.num_bits != 8 or group_size != 128:
            return None

        if meta.shape_n % 256 != 0:
            return None

        if block_shape[1] != 128 or num_ctas_per_sm != 2:
            return None

        new_block_shape = (block_shape[0], 256, 128)
        smem_size = estimate_smem_size_layer(meta, new_block_shape, num_stages)
        if smem_size > cls.max_smem_size:
            return None

        return {
            "block_shape": (block_shape[0], 256, 128),
            "warp_shape": (block_shape[0], 32, 128),
            "use_stream_k": use_stream_k,
            "use_f16_accum": False,
            "num_sms": cls.get_num_sms(),
            "num_stages": num_stages,
            "num_ctas_per_sm": 1,
        }

    @classmethod
    def get_config(
        cls,
        meta: HummingLayerMeta,
        shape_m: int,
        use_stream_k: bool,
        use_f16_accum: bool,
    ):
        if meta.a_dtype.num_bits == 16:
            assert meta.a_dtype in cls.b16_allowed_dtypes
        elif meta.a_dtype.num_bits == 8:
            assert meta.a_dtype in cls.b8_allowed_dtypes
        elif meta.a_dtype.num_bits == 4:
            assert meta.a_dtype in cls.b4_allowed_dtypes
        else:
            raise AssertionError(f"unsupported a_dtype {meta.a_dtype} on sm{cls.sm_version}")

        # 1. determine block_shape_m, init warp_shape_m
        max_block_shape_m = 64
        has_group_scale = (meta.input_scale_group_size or meta.weight_scale_group_size) > 0
        if meta.a_dtype.num_bits in [4, 8] and not has_group_scale:
            max_block_shape_m = 128
            if meta.b_dtype.num_bits <= 4 and cls.sm_version == 75:
                max_block_shape_m = 64
        block_shape_m = cls.get_block_shape_m(
            max_block_shape_m=max_block_shape_m,
            shape_m=shape_m,
            num_experts=meta.num_experts,
            top_k=meta.top_k,
        )
        warp_shape_m = block_shape_m
        num_write_splits = 1
        if cls.sm_version == 75 and block_shape_m == 64:
            # shared memory of sm75 of quiet small,
            # we use 2 splits for epilogue, so we need less shared memory
            warp_shape_m = block_shape_m
            num_write_splits = 2

        # 2. init block_shape_n, warp_shape_n, warp_shape_k
        block_shape_n = 256
        while meta.shape_n % block_shape_n != 0:
            block_shape_n = block_shape_n // 2
        assert block_shape_n >= 64
        warp_shape_n = 64
        warp_shape_k = 512 // meta.a_dtype.num_bits
        if meta.a_dtype.num_bits in [4, 8] and not has_group_scale and block_shape_m > 64:
            warp_shape_n = 32
        if meta.a_dtype.num_bits != 16 and has_group_scale and block_shape_m > 32:
            warp_shape_n = 32
            block_shape_n = min(block_shape_n, 128)

        # 3. init num_ctas_per_sm, reduce num_ctas_per_sm and block_shape_n / warp_shape_n
        #    for small m (to avoid too many ctas processing the same mn_block)
        num_blocks_n = meta.shape_n // block_shape_n
        num_blocks_m = cls.estimate_num_blocks_m(meta, shape_m, block_shape_m)
        num_sms = cls.get_num_sms()
        num_warps_mn = block_shape_m * block_shape_n // (warp_shape_m * warp_shape_n)
        if num_warps_mn >= 8:
            num_ctas_per_sm = 1
        else:
            num_ctas_per_sm = 2
        num_ctas_per_mn_block = num_sms * num_ctas_per_sm / (num_blocks_m * num_blocks_n)
        while num_ctas_per_mn_block > 2:
            if num_blocks_m > 1 or num_ctas_per_mn_block < 3:
                # for compute-bound, allow the same mn_block to be processe by at most 3 ctas
                if meta.a_dtype.num_bits == 8 and warp_shape_n == 64:
                    warp_shape_n = 32
                    block_shape_n = block_shape_n // 2
                    num_ctas_per_mn_block = num_ctas_per_mn_block / 2
                break

            if num_ctas_per_sm > 1 and (meta.a_dtype.num_bits == 16 or warp_shape_n == 32):
                num_ctas_per_sm = 1
                num_ctas_per_mn_block = num_ctas_per_mn_block / 2
                continue

            if block_shape_n == 64:
                break

            block_shape_n = block_shape_n // 2
            num_blocks_n = num_blocks_n * 2
            num_ctas_per_mn_block = num_ctas_per_mn_block / 2
            if meta.a_dtype.num_bits == 8 and warp_shape_n == 64:
                warp_shape_n = 32

        # 4. special case for b8 activation + groupsize 128: force the warp_shape_k to be 128
        if num_ctas_per_sm < 3:
            block_shape_k = warp_shape_k
            config = cls.get_special_config_b8_g128(
                meta=meta,
                block_shape=(block_shape_m, block_shape_n, block_shape_k),
                num_ctas_per_sm=num_ctas_per_sm,
                use_stream_k=use_f16_accum,
            )
            if config is not None:
                return config

        # 5. determine block_shape_k and num_ctas_per_sm based on shared memory size
        num_warps_n = block_shape_n // warp_shape_n
        num_stages = 2 if cls.sm_version < 80 else 3
        block_shape_k, num_ctas_per_sm = cls.get_block_shape_k_and_num_ctas_per_sm(
            meta=meta,
            max_num_ctas_per_sm=num_ctas_per_sm,
            num_warps_m=block_shape_m // warp_shape_m,
            num_warps_n=num_warps_n,
            warp_shape_k=warp_shape_k,
            block_shape_m=block_shape_m,
            block_shape_n=block_shape_n,
            num_stages=num_stages,
        )

        # 6. reduce warp_shape if possible, to ensure we have enough warps per sm
        num_warps = num_warps_n * (block_shape_k // warp_shape_k)
        if num_warps == 2 and warp_shape_m in [32, 64]:
            warp_shape_m = warp_shape_m // 2
            num_warps = 4
        if meta.a_dtype.num_bits != 16 and num_warps * num_ctas_per_sm < 8 and warp_shape_n == 64:
            warp_shape_n = 32
            num_warps = num_warps * 2

        # 7. increase warp_shape_k and num_stages if shared memory is enough
        block_shape = (block_shape_m, block_shape_n, block_shape_k)
        if warp_shape_k == 512 // meta.a_dtype.num_bits and meta.shape_k % (block_shape_k * 2) == 0:
            block_shape_new = (block_shape_m, block_shape_n, block_shape_k * 2)
            smem_size = estimate_smem_size_layer(meta, block_shape_new, num_stages)
            if smem_size * num_ctas_per_sm < cls.max_smem_size:
                block_shape_k = block_shape_k * 2
                warp_shape_k = warp_shape_k * 2
                block_shape = block_shape_new
        smem_size = estimate_smem_size_layer(meta, block_shape, num_stages + 1)
        if smem_size * num_ctas_per_mn_block < cls.max_smem_size:
            num_stages = num_stages + 1

        smem_size = estimate_smem_size_layer(meta, block_shape, num_stages)
        assert smem_size * num_ctas_per_sm <= cls.max_smem_size, f"{block_shape} {num_stages}"

        return {
            "block_shape": (block_shape_m, block_shape_n, block_shape_k),
            "warp_shape": (warp_shape_m, warp_shape_n, warp_shape_k),
            "use_stream_k": use_stream_k,
            "use_f16_accum": use_f16_accum,
            "num_sms": min(num_blocks_m * num_blocks_n * 4, num_sms),
            "num_stages": num_stages,
            "num_ctas_per_sm": num_ctas_per_sm,
            "num_write_splits": num_write_splits,
        }

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
    def estimate_num_blocks_m(cls, meta: HummingLayerMeta, shape_m: int, block_shape_m: int):
        if meta.num_experts is None:
            estimated_num_blocks_m = math.ceil(shape_m / block_shape_m)
        elif shape_m * meta.top_k < meta.num_experts:
            estimated_num_blocks_m = shape_m * meta.top_k
        else:
            estimated_num_blocks_m = meta.num_experts

        return estimated_num_blocks_m

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
    def get_num_sms(cls):
        return torch.cuda.get_device_properties().multi_processor_count

    @classmethod
    def get_configs(cls, meta: HummingLayerMeta, use_stream_k: bool, use_f16_accum: bool):
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
