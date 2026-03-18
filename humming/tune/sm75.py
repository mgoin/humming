from humming import dtypes
from humming.tune.base import DeviceHeuristics


class Sm75Heuristics(DeviceHeuristics):
    max_smem_size: int = 64 * 1024
    sm_version: int = 75
    b16_allowed_dtypes: list[dtypes.DataType] = [dtypes.float16]
    b8_allowed_dtypes: list[dtypes.DataType] = [dtypes.int8]
