import enum


class MmaType(enum.Enum):
    MMA = "mma"
    WGMMA = "wgmma"
    # Blackwell tcgen05.mma (UMMA): accumulator in TMEM, operands in SMEM/TMEM.
    # Ordinal 2 matches MmaType::TCGEN05 in include/humming/utils/enum.cuh.
    TCGEN05 = "tcgen05"
    MXMMA = "mxmma"


class WeightScaleType(enum.Enum):
    GROUP = "group"
    BLOCK = "block"
    CHANNEL = "channel"
    TENSOR = "tensor"


class WeightScale2Type(enum.Enum):
    NONE = "none"
    CHANNEL = "channel"
    TENSOR = "tensor"


class GemmType(enum.Enum):
    DENSE = "dense"
    INDEXED = "indexed"
    GROUPED_CONTIGUOUS = "grouped_contiguous"
    GROUPED_MASKED = "grouped_masked"
