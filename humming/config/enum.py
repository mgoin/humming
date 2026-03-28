import enum


class MmaType(enum.Enum):
    MMA = "mma"
    WGMMA = "wgmma"


class WeightScaleType(enum.Enum):
    GROUP = "group"
    BLOCK = "block"
    CHANNEL = "channel"
    TENSOR = "tensor"
    GROUP_TENSOR = "group_tensor"
