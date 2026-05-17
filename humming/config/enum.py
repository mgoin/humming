import enum


class MmaType(enum.Enum):
    MMA = "mma"
    WGMMA = "wgmma"
    # Blackwell tcgen05.mma -- accumulator in TMEM, both operands in SMEM.
    # Wired up for sm_100+; see kernel/tcgen05_mma.cuh.
    TCGEN05 = "tcgen05"


class WeightScaleType(enum.Enum):
    GROUP = "group"
    BLOCK = "block"
    CHANNEL = "channel"
    TENSOR = "tensor"
    GROUP_TENSOR = "group_tensor"


class GemmType(enum.Enum):
    DENSE = "dense"
    INDEXED = "indexed"
    GROUPED_CONTIGUOUS = "grouped_contiguous"
    GROUPED_MASKED = "grouped_masked"
