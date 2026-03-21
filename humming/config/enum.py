import enum


class MmaType(enum.Enum):
    MMA = "mma"
    WGMMA = "wgmma"


class ActivationType(enum.Enum):
    NONE = "none"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    GELU = "gelu"
    FASTGELU = "fastgelu"
    QUICKGELU = "quickgelu"
    SILU = "silu"
    CUSTOM = "custom"
    SILU_GLU = "silu_glu"
    CUSTOM_GLU = "custom_glu"
