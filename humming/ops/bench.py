import torch
import triton

from humming import dtypes
from humming.kernel.tops_bench import TopsBenchKernel


def tops_bench(dtype: str, mma_type: str | None = None, use_f16_accum: bool = False) -> int:
    if mma_type is None:
        mma_type = "wgmma" if torch.cuda.get_device_capability()[0] == 9 else "mma"

    if mma_type == "mma":
        mma_shape_m = 16
        mma_shape_n = 8
        mma_shape_k = 256 // dtypes.DataType.from_str(dtype).num_bits
    else:
        mma_shape_m = 64
        mma_shape_n = 256
        mma_shape_k = 256 // dtypes.DataType.from_str(dtype).num_bits

    if "float" in dtype:
        out_dtype = "float32"
    else:
        out_dtype = "int32"

    if use_f16_accum:
        assert dtype in ["float16", "float8e4m3", "float8e5m2"]
        out_dtype = "float16"

    kernel = TopsBenchKernel(
        mma_type=mma_type,
        mma_shape_m=mma_shape_m,
        mma_shape_n=mma_shape_n,
        mma_shape_k=mma_shape_k,
        ab_dtype=dtype,
        cd_dtype=out_dtype,
        repeat_count=65536,
        unroll_count=64,
    )

    ops_per_call = kernel.ops_per_call
    t = triton.testing.do_bench(kernel, warmup=100, rep=1000)
    return 65536 * ops_per_call / t / 1e9
